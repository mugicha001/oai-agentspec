"""L1: `_adapters.governance` の純ロジック検証（fake 注入・Runner 実行非依存）。

`_load_policy`（YAML 読込 / 未知キー fail-fast / 非強制フィールド警告 / policy オブジェクト
素通し・検証）、`_evaluate_tool`（allowlist 短絡 / blocked_patterns 生文字列照合 / JSON
エスケープ正規化照合 / パース不能フォールバック）、`_field_default`（default /
default_factory / 必須）を直接検証する。加えて `_make_audit_hooks` の MCP ツール評価
（origin 判定 / 記録順 / fail-closed / agent_id / 利用者フック非到達 / policy 既定 None での
非評価）を fake sink・fake policy・fake 拒否例外の注入で検証する。

ポリシー評価は原則 fake policy（呼び出し記録付き）を注入し AGT 非依存で検証する。実
`GovernancePolicy`（dataclass フィールド集合と照合実装が挙動を決める）が必要なのは、YAML 読込と
`blocked_patterns` の照合結果を `spec.tools` 経路と同一入力で突き合わせる検証（`_evaluate_tool`
の複製検知）で、いずれもテスト内 `pytest.importorskip` により governance extra 未導入環境では
skip する。
"""

from __future__ import annotations

import warnings
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

import pytest
from agents import FunctionTool, ToolOrigin, ToolOriginType
from agents.tool_context import ToolContext

from oai_agentspec._adapters.governance import (
    _evaluate_tool,
    _field_default,
    _load_policy,
    _make_audit_hooks,
    _require_agt,
)

pytestmark = pytest.mark.unit


def _agt_policy_cls() -> Any:
    """AGT 実 `GovernancePolicy` を返す（governance extra 未導入環境では skip）。"""
    mod = pytest.importorskip(
        "openai_agents_trust", reason="governance extra（agent-governance-toolkit）未導入"
    )
    return mod.GovernancePolicy


class _FakePolicy:
    """`check_tool` / `check_content` を呼び出し記録付きで応答する fake policy。"""

    def __init__(self, *, tool_reason: Any = None, content_deny_substr: str | None = None) -> None:
        """拒否条件を設定する（None なら許可）。"""
        self.tool_calls: list[str] = []
        self.content_calls: list[str] = []
        self._tool_reason = tool_reason
        self._deny = content_deny_substr

    def check_tool(self, tool_name: str) -> Any:
        """ツール名照合（設定された理由をそのまま返す）。"""
        self.tool_calls.append(tool_name)
        return self._tool_reason

    def check_content(self, content: str) -> str | None:
        """引数テキスト照合（部分文字列一致で拒否）。"""
        self.content_calls.append(content)
        if self._deny is not None and self._deny in content:
            return f"blocked: {self._deny}"
        return None


# ----------------------------------------------------------------------
# _load_policy: YAML パス読込
# ----------------------------------------------------------------------


def test_load_policy_yaml_path_builds_policy(tmp_path: Path) -> None:
    """強制対象フィールドのみの YAML は警告なしで `GovernancePolicy` を構築する。"""
    cls = _agt_policy_cls()
    path = tmp_path / "policy.yaml"
    path.write_text('allowed_tools: ["a", "b"]\nblocked_patterns: ["rm"]\n', encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        policy = _load_policy(str(path), cls)
    assert isinstance(policy, cls)
    assert policy.allowed_tools == ["a", "b"]
    assert policy.blocked_patterns == ["rm"]


def test_load_policy_accepts_pathlike(tmp_path: Path) -> None:
    """`os.PathLike`（`Path`）でも YAML 読込経路に入る。"""
    cls = _agt_policy_cls()
    path = tmp_path / "policy.yaml"
    path.write_text('allowed_tools: ["only"]\n', encoding="utf-8")
    policy = _load_policy(path, cls)
    assert policy.allowed_tools == ["only"]


def test_load_policy_empty_yaml_raises_value_error(tmp_path: Path) -> None:
    """空 YAML は全既定（allowlist 無効 = 全許可）へ静かに化けるため ValueError で拒否する。"""
    cls = _agt_policy_cls()
    path = tmp_path / "policy.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="空です"):
        _load_policy(str(path), cls)


def test_load_policy_falsy_non_mapping_roots_raise(tmp_path: Path) -> None:
    """falsy 非マッピングのルート（[] / false / 0）も or {} で潰さず ValueError で拒否する。"""
    cls = _agt_policy_cls()
    for content in ("[]\n", "false\n", "0\n"):
        path = tmp_path / "policy.yaml"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="マッピング"):
            _load_policy(str(path), cls)


def test_load_policy_non_mapping_raises_value_error(tmp_path: Path) -> None:
    """マッピングでない YAML（リスト等）は ValueError。"""
    cls = _agt_policy_cls()
    path = tmp_path / "policy.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="マッピング"):
        _load_policy(str(path), cls)


def test_load_policy_unknown_key_raises_value_error(tmp_path: Path) -> None:
    """未知キー（`allowed_tool:` のような typo）は黙殺せず ValueError（footgun 防止）。"""
    cls = _agt_policy_cls()
    path = tmp_path / "policy.yaml"
    path.write_text('allowed_tool: ["x"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="未知のキー") as excinfo:
        _load_policy(str(path), cls)
    # typo キーと有効キーの両方がメッセージに含まれる（修正手がかり）。
    assert "allowed_tool" in str(excinfo.value)
    assert "allowed_tools" in str(excinfo.value)


def test_load_policy_missing_file_raises(tmp_path: Path) -> None:
    """存在しない YAML パスは FileNotFoundError。"""
    cls = _agt_policy_cls()
    with pytest.raises(FileNotFoundError):
        _load_policy(str(tmp_path / "missing.yaml"), cls)


# ----------------------------------------------------------------------
# _load_policy: 非強制フィールドの警告
# ----------------------------------------------------------------------


def test_load_policy_non_enforced_field_warns(tmp_path: Path) -> None:
    """非強制フィールド（max_tokens 等）を既定値以外で指定すると RuntimeWarning。"""
    cls = _agt_policy_cls()
    path = tmp_path / "policy.yaml"
    path.write_text("max_tokens: 5\n", encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="max_tokens"):
        policy = _load_policy(str(path), cls)
    # 警告は出るがロード自体は成功する。
    assert isinstance(policy, cls)


def test_load_policy_non_enforced_field_default_value_no_warn(tmp_path: Path) -> None:
    """非強制フィールドでも既定値ちょうどの指定なら警告しない。"""
    cls = _agt_policy_cls()
    default = next(f.default for f in fields(cls) if f.name == "max_tokens")
    path = tmp_path / "policy.yaml"
    path.write_text(f"max_tokens: {default}\n", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        policy = _load_policy(str(path), cls)
    assert policy.max_tokens == default


def test_load_policy_benign_name_field_no_warn(tmp_path: Path) -> None:
    """メタフィールド `name` は挙動に影響しないため警告対象外。"""
    cls = _agt_policy_cls()
    path = tmp_path / "policy.yaml"
    path.write_text("name: custom\n", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        policy = _load_policy(str(path), cls)
    assert policy.name == "custom"


# ----------------------------------------------------------------------
# _load_policy: policy オブジェクト素通し / 検証
# ----------------------------------------------------------------------


def test_load_policy_object_passthrough() -> None:
    """評価メソッドを持つオブジェクトは同一オブジェクトのまま素通しされる。"""
    policy = _FakePolicy()
    assert _load_policy(policy, object) is policy


def test_load_policy_object_missing_check_content_raises_type_error() -> None:
    """`check_content` 欠如オブジェクトは build 時に TypeError（fail-fast）。"""

    class _OnlyCheckTool:
        def check_tool(self, tool_name: str) -> None:
            return None

    with pytest.raises(TypeError, match="check_content"):
        _load_policy(_OnlyCheckTool(), object)


def test_load_policy_object_non_callable_check_tool_raises_type_error() -> None:
    """`check_tool` が callable でないオブジェクトも TypeError。"""

    class _NonCallable:
        check_tool = "not-callable"

        def check_content(self, content: str) -> None:
            return None

    with pytest.raises(TypeError, match="check_tool"):
        _load_policy(_NonCallable(), object)


# ----------------------------------------------------------------------
# _evaluate_tool: allowlist / blocked_patterns / 正規化照合
# ----------------------------------------------------------------------


def test_evaluate_tool_allowlist_deny_short_circuits_content_check() -> None:
    """ツール名拒否は引数照合より先に短絡し、その理由を返す。"""
    policy = _FakePolicy(tool_reason="tool not allowed")
    reason = _evaluate_tool(policy, "rmtool", '{"x": 1}')
    assert reason == "tool not allowed"
    assert policy.tool_calls == ["rmtool"]
    assert policy.content_calls == []  # check_content は呼ばれない


def test_evaluate_tool_tool_reason_coerced_to_str() -> None:
    """`check_tool` の非 str 理由は str へ変換して返す。"""
    policy = _FakePolicy(tool_reason=42)
    assert _evaluate_tool(policy, "t", "{}") == "42"


def test_evaluate_tool_blocked_pattern_denies_raw_string() -> None:
    """生 JSON 文字列がパターンに一致したら違反理由を返す。"""
    policy = _FakePolicy(content_deny_substr="rm ")
    reason = _evaluate_tool(policy, "sh", '{"cmd": "rm -rf /"}')
    assert reason == "blocked: rm "


def test_evaluate_tool_json_escape_normalization_denies() -> None:
    """JSON エスケープ（\\u0072m = rm）による回避は正規化文字列照合で deny する（fail-closed）。"""
    policy = _FakePolicy(content_deny_substr="rm ")
    raw = '{"cmd": "\\u0072m -rf /"}'  # 生文字列に "rm" は現れない
    assert "rm " not in raw
    reason = _evaluate_tool(policy, "sh", raw)
    assert reason == "blocked: rm "
    # 生 + 正規化の両方が照合される。
    assert policy.content_calls == [raw, '{"cmd": "rm -rf /"}']


def test_evaluate_tool_content_reason_coerced_to_str() -> None:
    """`check_content` の非 str 理由も str へ変換して返す。"""

    class _ObjReasonPolicy:
        def check_tool(self, tool_name: str) -> None:
            return None

        def check_content(self, content: str) -> Any:
            return 7

    assert _evaluate_tool(_ObjReasonPolicy(), "t", "{}") == "7"


def test_evaluate_tool_unparseable_input_falls_back_to_raw_only() -> None:
    """JSON パース不能な入力は生文字列のみ照合する（フォールバック・1 回だけ呼ばれる）。"""
    policy = _FakePolicy()
    assert _evaluate_tool(policy, "t", "not json {{") is None
    assert policy.content_calls == ["not json {{"]


def test_evaluate_tool_unparseable_input_still_denies_on_raw() -> None:
    """パース不能でも生文字列の照合は維持される（fail-closed の取りこぼし無し）。"""
    policy = _FakePolicy(content_deny_substr="rm ")
    assert _evaluate_tool(policy, "t", "rm -rf /") == "blocked: rm "


def test_evaluate_tool_json_null_skips_normalization() -> None:
    """`null` をパースした None は正規化照合に回さない（生文字列のみ）。"""
    policy = _FakePolicy()
    assert _evaluate_tool(policy, "t", "null") is None
    assert policy.content_calls == ["null"]


def test_evaluate_tool_allowed_returns_none_after_all_candidates() -> None:
    """許可時は None を返す（生 + 正規化 + デコード済み文字列を重複排除して照合済み）。"""
    policy = _FakePolicy()
    raw = '{"q": "ok"}'
    assert _evaluate_tool(policy, "t", raw) is None
    assert policy.tool_calls == ["t"]
    # 正規化文字列は raw と同一のため重複排除され、デコード済みのキー / 値が続く。
    assert policy.content_calls == [raw, "q", "ok"]


# ----------------------------------------------------------------------
# _field_default / _require_agt
# ----------------------------------------------------------------------


def test_field_default_handles_default_factory_and_missing() -> None:
    """default / default_factory / 必須（MISSING）の 3 形を判別する。"""

    @dataclass
    class _D:
        required: int
        plain: int = 1
        factory: list[int] = field(default_factory=list)

    by_name = {f.name: f for f in fields(_D)}
    assert _field_default(by_name["plain"]) == 1
    assert _field_default(by_name["factory"]) == []
    assert _field_default(by_name["required"]) is MISSING


def test_require_agt_missing_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """AGT 未導入相当（import 失敗）では install hint 付き ImportError を送出する。"""
    import sys

    monkeypatch.setitem(sys.modules, "openai_agents_trust", None)
    with pytest.raises(ImportError, match=r"oai-agentspec\[governance\]"):
        _require_agt()


# ----------------------------------------------------------------------
# load_policy_bundle（bundle YAML: default + agents・実 GovernancePolicy 必要）
# ----------------------------------------------------------------------


def _write_bundle(tmp_path: Path, content: str) -> Path:
    """bundle YAML を一時ファイルへ書き出す。"""
    path = tmp_path / "governance.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_policy_bundle_default_and_agents(tmp_path: Path) -> None:
    """default + agents の bundle が (既定ポリシー, per-agent dict) に構築される。"""
    _agt_policy_cls()  # extra 未導入環境では skip
    from oai_agentspec._adapters.governance import load_policy_bundle

    path = _write_bundle(
        tmp_path,
        "default:\n"
        "  allowed_tools: [lookup]\n"
        "agents:\n"
        "  support:\n"
        "    allowed_tools: [lookup, refund]\n",
    )
    default_policy, agent_policies = load_policy_bundle(path)

    assert default_policy.allowed_tools == ["lookup"]
    assert set(agent_policies) == {"support"}
    assert agent_policies["support"].allowed_tools == ["lookup", "refund"]


def test_load_policy_bundle_default_only(tmp_path: Path) -> None:
    """agents セクション省略時は per-agent dict が空になる（default のみで有効）。"""
    _agt_policy_cls()
    from oai_agentspec._adapters.governance import load_policy_bundle

    path = _write_bundle(tmp_path, "default:\n  allowed_tools: [lookup]\n")
    default_policy, agent_policies = load_policy_bundle(path)

    assert default_policy.allowed_tools == ["lookup"]
    assert agent_policies == {}

    # `agents:`（空値 = None）はセクション未指定と同じ扱い（省略可）。
    path_empty = _write_bundle(tmp_path, "default:\n  allowed_tools: [lookup]\nagents:\n")
    _, agent_policies_empty = load_policy_bundle(path_empty)
    assert agent_policies_empty == {}


def test_load_policy_bundle_missing_default_raises(tmp_path: Path) -> None:
    """default セクション欠落は ValueError（既定ポリシー必須の維持）。"""
    _agt_policy_cls()
    from oai_agentspec._adapters.governance import load_policy_bundle

    path = _write_bundle(tmp_path, "agents:\n  support:\n    allowed_tools: [refund]\n")
    with pytest.raises(ValueError, match="'default'"):
        load_policy_bundle(path)


def test_load_policy_bundle_unknown_top_key_raises(tmp_path: Path) -> None:
    """トップレベル未知キー（defaults 等の typo）は ValueError で fail-fast。"""
    _agt_policy_cls()
    from oai_agentspec._adapters.governance import load_policy_bundle

    path = _write_bundle(tmp_path, "defaults:\n  allowed_tools: [lookup]\n")  # "defaults" は typo
    with pytest.raises(ValueError, match="未知のキー"):
        load_policy_bundle(path)


def test_load_policy_bundle_section_unknown_key_raises(tmp_path: Path) -> None:
    """セクション内の未知キー（allowed_tool 等の typo）も単一 YAML と同一の ValueError。"""
    _agt_policy_cls()
    from oai_agentspec._adapters.governance import load_policy_bundle

    path = _write_bundle(
        tmp_path,
        "default:\n  allowed_tools: [lookup]\nagents:\n  support:\n    allowed_tool: [refund]\n",
    )
    with pytest.raises(ValueError, match=r"agents\['support'\]"):
        load_policy_bundle(path)


def test_load_policy_bundle_non_mapping_shapes_raise(tmp_path: Path) -> None:
    """bundle 全体 / default / agents / セクションがマッピングでない場合は ValueError。"""
    _agt_policy_cls()
    from oai_agentspec._adapters.governance import load_policy_bundle

    with pytest.raises(ValueError, match="マッピング"):
        load_policy_bundle(_write_bundle(tmp_path, "- not-a-mapping\n"))
    with pytest.raises(ValueError, match="'default'"):
        load_policy_bundle(_write_bundle(tmp_path, "default: not-a-mapping\n"))
    with pytest.raises(ValueError, match="'agents'"):
        load_policy_bundle(
            _write_bundle(tmp_path, "default:\n  allowed_tools: [a]\nagents: not-a-mapping\n")
        )
    # falsy 非マッピング（agents: []）も黙殺せず型エラー（overrides が静かに消える footgun 防止）。
    with pytest.raises(ValueError, match="'agents'"):
        load_policy_bundle(_write_bundle(tmp_path, "default:\n  allowed_tools: [a]\nagents: []\n"))
    with pytest.raises(ValueError, match=r"agents\['support'\]"):
        load_policy_bundle(
            _write_bundle(
                tmp_path, "default:\n  allowed_tools: [a]\nagents:\n  support: not-a-mapping\n"
            )
        )


def test_load_policy_bundle_warns_non_enforced_fields(tmp_path: Path) -> None:
    """セクション内の非強制フィールド指定は単一 YAML と同様に RuntimeWarning を出す。"""
    _agt_policy_cls()
    from oai_agentspec._adapters.governance import load_policy_bundle

    path = _write_bundle(
        tmp_path,
        "default:\n  allowed_tools: [lookup]\n  max_tool_calls: 3\n",
    )
    with pytest.warns(RuntimeWarning, match="max_tool_calls"):
        load_policy_bundle(path)


# ----------------------------------------------------------------------
# _policy_from_mapping: 値形状・キー型・正規表現の fail-fast 検証
# ----------------------------------------------------------------------


def test_load_policy_scalar_allowed_tools_raises(tmp_path: Path) -> None:
    """スカラ文字列の allowed_tools（list typo）は部分文字列 allowlist 化を防ぐため ValueError。"""
    cls = _agt_policy_cls()
    path = tmp_path / "policy.yaml"
    path.write_text("allowed_tools: refund_tool\n", encoding="utf-8")
    with pytest.raises(ValueError, match="allowed_tools"):
        _load_policy(str(path), cls)


def test_load_policy_allowed_tools_non_str_items_raise(tmp_path: Path) -> None:
    """allowed_tools のリスト要素に非文字列が混ざる場合も ValueError（null 単体は許容）。"""
    cls = _agt_policy_cls()
    path = tmp_path / "policy.yaml"
    path.write_text("allowed_tools:\n  - lookup\n  - 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="allowed_tools"):
        _load_policy(str(path), cls)
    # allowed_tools: null（allowlist なし）は明示指定として有効。
    path.write_text("name: open\nallowed_tools: null\n", encoding="utf-8")
    assert _load_policy(str(path), cls).allowed_tools is None


def test_load_policy_blocked_patterns_shape_raises(tmp_path: Path) -> None:
    """blocked_patterns はスカラ文字列 / null / 非文字列要素を ValueError で拒否する。"""
    cls = _agt_policy_cls()
    path = tmp_path / "policy.yaml"
    for content in (
        'blocked_patterns: "rm -rf"\n',  # スカラ（1 文字ずつ正規表現化する footgun）
        "blocked_patterns: null\n",  # AGT 側で実行時 TypeError になる
        "blocked_patterns:\n  - 1\n",
    ):
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="blocked_patterns"):
            _load_policy(str(path), cls)


def test_load_policy_invalid_regex_raises_at_load(tmp_path: Path) -> None:
    """compile 不能な正規表現はロード時 ValueError（実行時 re.error への遅延を防ぐ）。"""
    cls = _agt_policy_cls()
    path = tmp_path / "policy.yaml"
    path.write_text('blocked_patterns:\n  - "(unclosed"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="正規表現"):
        _load_policy(str(path), cls)


def test_load_policy_non_str_keys_raise(tmp_path: Path) -> None:
    """YAML 1.1 暗黙型付けで bool / null 化したキー（on: / null:）は ValueError で案内する。"""
    cls = _agt_policy_cls()
    path = tmp_path / "policy.yaml"
    path.write_text("on: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="文字列である必要"):
        _load_policy(str(path), cls)


def test_load_policy_bundle_non_str_keys_raise(tmp_path: Path) -> None:
    """bundle のトップレベル / agents キーの非文字列（on: 等）は ValueError で案内する。"""
    _agt_policy_cls()
    from oai_agentspec._adapters.governance import load_policy_bundle

    top = _write_bundle(tmp_path, "default:\n  allowed_tools: [a]\ntrue: {}\n")
    with pytest.raises(ValueError, match="トップレベルキーは文字列"):
        load_policy_bundle(top)

    agents = _write_bundle(
        tmp_path, "default:\n  allowed_tools: [a]\nagents:\n  on:\n    allowed_tools: [b]\n"
    )
    with pytest.raises(ValueError, match="エージェント名"):
        load_policy_bundle(agents)


# ----------------------------------------------------------------------
# _evaluate_tool: デコード値照合・RecursionError 耐性
# ----------------------------------------------------------------------


def test_evaluate_tool_denies_on_decoded_string_values() -> None:
    """エスケープ表現（\\n）と実改行の差による \\s 系パターン回避をデコード値照合で塞ぐ。"""
    cls = _agt_policy_cls()
    policy = cls(name="p", blocked_patterns=[r"rm\s+-rf"])
    raw = '{"cmd": "rm\\n-rf /"}'  # 生 / 正規化とも実改行を含まない（\\n エスケープのまま）
    reason = _evaluate_tool(policy, "sh", raw)
    assert reason is not None
    assert "rm" in reason


def test_evaluate_tool_deeply_nested_input_does_not_crash() -> None:
    """深ネスト引数で json.loads が RecursionError でも評価は生文字列照合で継続する。"""
    policy = _FakePolicy()
    deep = "[" * 60000
    assert _evaluate_tool(policy, "t", deep) is None
    assert policy.content_calls == [deep]  # フォールバック（クラッシュ・監査欠落なし）


def test_iter_decoded_strings_walks_nested_structures() -> None:
    """_iter_decoded_strings はネストした dict / list から文字列スカラとキーを集める。"""
    from oai_agentspec._adapters.governance import _iter_decoded_strings

    out = _iter_decoded_strings({"a": ["x", {"b": "y"}, 1], "c": None})
    assert set(out) == {"a", "b", "c", "x", "y"}


# ----------------------------------------------------------------------
# _make_audit_hooks: MCP ツールのポリシー評価（fake 注入・AGT 非依存）
# ----------------------------------------------------------------------


class _DenyExc(Exception):
    """`denied_exc` として注入する fake 拒否例外（AGT `PolicyViolationError` 相当）。"""


class _Sink:
    """`record(...)` 呼び出しを 4 つ組で記録する fake 監査 sink（記録順と件数の検証用）。"""

    def __init__(self) -> None:
        """記録リストを初期化する。"""
        self.records: list[tuple[Any, str, str, dict[str, Any] | None]] = []

    def record(
        self,
        agent_id: str,
        action: str,
        decision: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """監査レコードを記録順に積む（AGT `AuditLog.record` と同シグネチャ）。"""
        self.records.append((agent_id, action, decision, details))


class _Named:
    """`name` だけを持つ最小の agent / tool スタブ。"""

    def __init__(self, name: str) -> None:
        """名前を保持する。"""
        self.name = name


class _RecordingInnerHooks:
    """利用者フック（`spec.hooks`）を模す記録フック（deny 時の非到達検証用・duck typing）。"""

    def __init__(self) -> None:
        """イベント記録を初期化する。"""
        self.events: list[str] = []

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        """ツール開始を記録する。"""
        self.events.append(f"tool_start:{tool.name}")

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        """ツール終了を記録する。"""
        self.events.append(f"tool_end:{tool.name}")


class _ArgsCtx:
    """`tool_arguments` に任意の値を持つ最小コンテキスト（非 str / None の検証用）。"""

    def __init__(self, tool_arguments: Any) -> None:
        """ツール引数を保持する。"""
        self.tool_arguments = tool_arguments


def _origin_tool(
    name: str = "mcp_read",
    *,
    origin_type: ToolOriginType | str | None = ToolOriginType.MCP,
    emit_origin: bool = True,
    invoked: list[str] | None = None,
) -> FunctionTool:
    """指定 origin の `FunctionTool` を作る（`MCPUtil.to_function_tool` を通さず origin を偽装）。

    Args:
        name: ツール名。
        origin_type: 載せる `ToolOriginType`（None なら `_tool_origin` を付けない）。生 str を
            渡した場合は `ToolOrigin(type=<生 str>)` になる（型検証を持たない frozen dataclass
            のため成立する・identity 比較の退行検知に使う）。
        emit_origin: `_emit_tool_origin`（False なら origin 解決結果が None になる）。
        invoked: 実ツール本体（`on_invoke_tool`）が呼ばれたら引数 JSON を積むリスト。

    Returns:
        origin メタを持つ `FunctionTool`。
    """

    async def _on_invoke_tool(ctx: Any, input_json: str) -> str:
        if invoked is not None:
            invoked.append(input_json)
        return "ok"

    origin: Any = None
    if origin_type is ToolOriginType.MCP:
        origin = ToolOrigin(type=origin_type, mcp_server_name="srv")
    elif origin_type is ToolOriginType.AGENT_AS_TOOL:
        origin = ToolOrigin(type=origin_type, agent_name="sub")
    elif origin_type is not None:
        origin = ToolOrigin(type=origin_type)
    return FunctionTool(
        name=name,
        description="fake tool",
        params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
        on_invoke_tool=_on_invoke_tool,
        _tool_origin=origin,
        _emit_tool_origin=emit_origin,
    )


def _tool_ctx(name: str, arguments: str) -> ToolContext:
    """`on_tool_start` へ渡す最小 `ToolContext`（実 SDK 型・`tool_arguments` を持つ）を作る。"""
    return ToolContext(context=None, tool_name=name, tool_call_id="c1", tool_arguments=arguments)


async def test_mcp_tool_deny_raises_and_does_not_invoke_tool_body() -> None:
    """A1: MCP origin ツールの deny は `denied_exc` を送出し、実ツール本体を呼ばない。"""
    sink = _Sink()
    invoked: list[str] = []
    tool = _origin_tool(invoked=invoked)
    policy = _FakePolicy(tool_reason="tool not allowed")
    hooks = _make_audit_hooks(sink, None, policy=policy, denied_exc=_DenyExc, agent_name="support")

    with pytest.raises(_DenyExc, match="mcp_read"):
        await hooks.on_tool_start(_tool_ctx("mcp_read", '{"q": "x"}'), _Named("support"), tool)

    assert invoked == []  # 実ツール本体は実行されない
    assert policy.tool_calls == ["mcp_read"]  # 評価は行われた


@pytest.mark.parametrize(
    ("origin_type", "emit_origin", "evaluated"),
    [
        (ToolOriginType.MCP, True, True),
        ("mcp", True, True),
        (ToolOriginType.FUNCTION, True, False),
        (ToolOriginType.AGENT_AS_TOOL, True, False),
        (None, False, False),
    ],
    ids=["mcp", "mcp_raw_str", "function", "agent_as_tool", "origin_none"],
)
async def test_only_mcp_origin_tools_are_evaluated(
    origin_type: ToolOriginType | str | None, emit_origin: bool, evaluated: bool
) -> None:
    """A2: 評価対象は MCP origin のみ（FUNCTION / AGENT_AS_TOOL / origin None は非評価）。

    非 MCP ケースはツール名を allowlist に載せない（= 評価されたら必ず deny になる）状態で、
    例外が出ないことと `tool:{name}` レコードが 0 件であることの 2 点で非評価を固定する
    （positive 判定への pin。negative 判定だと `spec.tools` の as_tool が二重評価される）。

    `mcp_raw_str` ケースは生 str origin（`ToolOrigin(type="mcp")`）の回帰 pin。
    `ToolOriginType` は `str` 派生 Enum で `ToolOrigin` は型検証を持たない frozen dataclass の
    ため、第三者ラッパ・シリアライズ経路から生 str が渡り得る。実装の origin 判定を `!=` から
    `is not`（identity 比較）へ戻すと、同値なのに不一致となり統治が無警告でスキップされる
    （fail-open）。enum ケースと同一の期待値（評価される）で固定する。
    """
    sink = _Sink()
    tool = _origin_tool(origin_type=origin_type, emit_origin=emit_origin)
    policy = _FakePolicy(tool_reason="tool not allowed")
    hooks = _make_audit_hooks(sink, None, policy=policy, denied_exc=_DenyExc, agent_name="support")
    ctx = _tool_ctx("mcp_read", '{"q": "x"}')
    agent = _Named("support")

    if evaluated:
        with pytest.raises(_DenyExc):
            await hooks.on_tool_start(ctx, agent, tool)
    else:
        await hooks.on_tool_start(ctx, agent, tool)  # 例外は出ない

    tool_records = [r for r in sink.records if r[1] == "tool:mcp_read"]
    assert len(tool_records) == (1 if evaluated else 0)
    assert policy.tool_calls == (["mcp_read"] if evaluated else [])


async def test_mcp_tool_deny_records_start_then_deny_in_order() -> None:
    """A3: deny 時は `tool_start:`(allow) -> `tool:`(deny) の順で raise 前に両方残る。"""
    sink = _Sink()
    tool = _origin_tool()
    policy = _FakePolicy(tool_reason="tool not allowed")
    hooks = _make_audit_hooks(sink, None, policy=policy, denied_exc=_DenyExc, agent_name="bot")

    with pytest.raises(_DenyExc):
        await hooks.on_tool_start(_tool_ctx("mcp_read", '{"q": "x"}'), _Named("bot"), tool)

    # 記録列全体を == で比較する（順序反転・重複記録・記録欠落のいずれも検知する）。
    assert sink.records == [
        ("bot", "tool_start:mcp_read", "allow", None),
        (
            "bot",
            "tool:mcp_read",
            "deny",
            {"reason": "tool not allowed", "arguments": '{"q": "x"}'},
        ),
    ]


async def test_mcp_tool_allow_records_single_allow_entry_with_arguments() -> None:
    """A4: allow 時は `tool:`(allow) が引数付きでちょうど 1 件残る（監査形式の対称性）。"""
    sink = _Sink()
    tool = _origin_tool()
    policy = _FakePolicy()
    hooks = _make_audit_hooks(sink, None, policy=policy, denied_exc=_DenyExc, agent_name="bot")

    await hooks.on_tool_start(_tool_ctx("mcp_read", '{"q": "x"}'), _Named("bot"), tool)

    assert sink.records == [
        ("bot", "tool_start:mcp_read", "allow", None),
        ("bot", "tool:mcp_read", "allow", {"arguments": '{"q": "x"}'}),
    ]
    assert len([r for r in sink.records if r[1] == "tool:mcp_read"]) == 1  # 二重評価なし


async def test_mcp_tool_blocked_patterns_deny_json_escaped_arguments() -> None:
    """A5: MCP 経路でも JSON エスケープ（\\u0072m = rm）の回避が塞がる（`_evaluate_tool` 共用）。

    既存 `test_blocked_patterns_deny_json_escaped_arguments`（L2・govern ラップ経路）と
    同一入力・同一結果になることを固定し、MCP 側で評価ロジックが複製・乖離するのを防ぐ。
    """
    cls = _agt_policy_cls()
    policy = cls(name="p", blocked_patterns=["rm -rf"])
    escaped = '{"text": "\\u0072m -rf /"}'  # 生文字列に "rm -rf" は現れない
    expected_reason = _evaluate_tool(policy, "sh", escaped)
    assert expected_reason is not None and "blocked pattern" in expected_reason

    sink = _Sink()
    invoked: list[str] = []
    tool = _origin_tool("sh", invoked=invoked)
    hooks = _make_audit_hooks(sink, None, policy=policy, denied_exc=_DenyExc, agent_name="bot")

    with pytest.raises(_DenyExc, match="sh"):
        await hooks.on_tool_start(_tool_ctx("sh", escaped), _Named("bot"), tool)

    assert invoked == []
    assert sink.records[-1] == (
        "bot",
        "tool:sh",
        "deny",
        {"reason": expected_reason, "arguments": escaped},
    )


@pytest.mark.parametrize(
    "context",
    [object(), _ArgsCtx(None), _ArgsCtx(123), _ArgsCtx({"q": "x"})],
    ids=["missing_attr", "none", "int", "dict"],
)
async def test_mcp_tool_unavailable_arguments_fail_closed(context: Any) -> None:
    """A6: `tool_arguments` が欠落 / 非 str なら fail-closed で deny する（名前照合へ縮退しない）。

    ツール名は allowlist に載せた状態（`_FakePolicy` は常に許可）で試験するため、deny の
    原因は引数の取得不能に限定される。
    """
    sink = _Sink()
    tool = _origin_tool()
    policy = _FakePolicy()
    hooks = _make_audit_hooks(sink, None, policy=policy, denied_exc=_DenyExc, agent_name="support")

    with pytest.raises(_DenyExc, match="mcp_read"):
        await hooks.on_tool_start(context, _Named("support"), tool)

    assert sink.records == [
        ("support", "tool_start:mcp_read", "allow", None),
        (
            "support",
            "tool:mcp_read",
            "deny",
            {
                "reason": "tool arguments unavailable for policy evaluation",
                "arguments": None,
            },
        ),
    ]
    # 引数が取れない時点で deny するため、ポリシー評価そのものには入らない。
    assert policy.tool_calls == []


async def test_non_function_tool_passes_through_without_evaluation() -> None:
    """A7: 非 `FunctionTool`（hosted tool 相当）は例外にならず素通しされる。"""
    sink = _Sink()
    policy = _FakePolicy(tool_reason="tool not allowed")
    hooks = _make_audit_hooks(sink, None, policy=policy, denied_exc=_DenyExc, agent_name="bot")

    await hooks.on_tool_start(
        _tool_ctx("web_search", '{"q": "x"}'), _Named("bot"), _Named("web_search")
    )

    assert sink.records == [("bot", "tool_start:web_search", "allow", None)]
    assert policy.tool_calls == []


async def test_policy_none_skips_evaluation_for_mcp_tool() -> None:
    """A8: `policy=None`（既存の 2 引数呼び出し相当）では MCP origin でも評価しない。"""
    sink = _Sink()
    tool = _origin_tool()

    hooks = _make_audit_hooks(sink, None)  # policy / denied_exc / agent_name は既定 None
    await hooks.on_tool_start(_tool_ctx("mcp_read", '{"q": "x"}'), _Named("bot"), tool)

    assert sink.records == [("bot", "tool_start:mcp_read", "allow", None)]


async def test_tool_record_uses_spec_agent_name_not_runtime_agent_name() -> None:
    """A9: `tool:` 行の agent_id は `agent_name`（`spec.name`）・`tool_start:` 行は agent.name。"""
    sink = _Sink()
    tool = _origin_tool()
    policy = _FakePolicy(tool_reason="tool not allowed")
    hooks = _make_audit_hooks(sink, None, policy=policy, denied_exc=_DenyExc, agent_name="support")

    with pytest.raises(_DenyExc):
        await hooks.on_tool_start(_tool_ctx("mcp_read", "{}"), _Named("other"), tool)

    assert [(r[0], r[1], r[2]) for r in sink.records] == [
        ("other", "tool_start:mcp_read", "allow"),
        ("support", "tool:mcp_read", "deny"),
    ]


async def test_mcp_tool_deny_does_not_reach_inner_hooks() -> None:
    """A10: MCP deny 時は利用者フック（`inner`）の `on_tool_start` へ到達しない。"""
    sink = _Sink()
    inner = _RecordingInnerHooks()
    tool = _origin_tool()
    policy = _FakePolicy(tool_reason="tool not allowed")
    hooks = _make_audit_hooks(sink, inner, policy=policy, denied_exc=_DenyExc, agent_name="support")

    with pytest.raises(_DenyExc):
        await hooks.on_tool_start(_tool_ctx("mcp_read", "{}"), _Named("support"), tool)

    assert inner.events == []  # 評価が委譲より後ろへ動いたら非空になる


async def test_allowed_mcp_tool_reaches_inner_hooks() -> None:
    """A10 対照: allow なら利用者フックへ委譲される（deny の非到達が raise 由来である担保）。"""
    sink = _Sink()
    inner = _RecordingInnerHooks()
    tool = _origin_tool()
    hooks = _make_audit_hooks(
        sink, inner, policy=_FakePolicy(), denied_exc=_DenyExc, agent_name="support"
    )

    await hooks.on_tool_start(_tool_ctx("mcp_read", "{}"), _Named("support"), tool)

    assert inner.events == ["tool_start:mcp_read"]
