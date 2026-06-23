"""L1: `GovernedAgentBuilder` の装飾ロジック検証（fake 注入・Runner 実行非依存）。

`build(spec)` が govern 済み spec（tools 差し替え + hooks 合成）を `inner.build` へ渡すこと、
`policy` / `audit_sink` が `govern_spec` へ素通しされること、`inner=None` で `DefaultAgentBuilder`
が使われること、既定 sink の生成・共有（`audit_sink` プロパティ）、extra 未導入時の挙動
（コンストラクトは成功し build で install hint 付き ImportError）を検証する。

builder は `build` 内で `from ..._adapters import ...` する（関数内遅延 import）ため、monkeypatch
対象は使用箇所パス `oai_agentspec._adapters.*`（`govern_spec` / `new_audit_sink` /
`DefaultAgentBuilder`）。AGT 実依存が要るのは実 `govern_spec` を通すテストのみで、`agt_symbols`
フィクスチャ（conftest）で extra 未導入環境では skip する。
"""

from __future__ import annotations

from typing import Any

import pytest

from oai_agentspec import AgentSpec, function_tool
from oai_agentspec._adapters.governance import _GOVERNANCE_INSTALL_HINT
from oai_agentspec.protocols import AgentBuilder
from oai_agentspec.runtime.governance import GovernedAgentBuilder

pytestmark = pytest.mark.unit


class _RecordingBuilder:
    """`build(spec)` 呼び出しの spec を記録する fake inner builder。"""

    def __init__(self) -> None:
        """記録リストを初期化する。"""
        self.specs: list[Any] = []

    def build(self, spec: AgentSpec) -> Any:
        """spec を記録し、識別可能なダミー Agent を返す（spec の属性には依存しない）。"""
        self.specs.append(spec)
        return ("agent", getattr(spec, "name", None))


def _make_tool(name: str = "echo") -> Any:
    """実 `FunctionTool` を 1 つ作る（govern ラップ対象・実行はしない）。"""

    @function_tool(name_override=name)
    def _tool(text: str) -> str:
        """エコーする。"""
        return text

    return _tool


# ----------------------------------------------------------------------
# AgentBuilder Protocol 適合
# ----------------------------------------------------------------------


def test_governed_builder_satisfies_agent_builder_protocol() -> None:
    """`GovernedAgentBuilder` が DI 拡張点 `AgentBuilder`（runtime_checkable）に構造適合する。"""
    assert isinstance(GovernedAgentBuilder(policy=object()), AgentBuilder)


# ----------------------------------------------------------------------
# build: govern 済み spec が inner へ渡る（実 govern_spec・AGT 必要）
# ----------------------------------------------------------------------


def test_build_passes_governed_spec_to_inner(
    agt_symbols: tuple[Any, Any, Any],  # noqa: ARG001 - extra 未導入時 skip のためのみ使用
    allow_all_policy: Any,
    recording_sink: Any,
) -> None:
    """inner へは tools 差し替え + 監査 hooks 合成済みの新 spec が渡る（元 spec 不変）。"""
    tool = _make_tool()
    spec = AgentSpec(name="bot", instructions="i", tools=[tool])
    inner = _RecordingBuilder()
    builder = GovernedAgentBuilder(policy=allow_all_policy, audit_sink=recording_sink, inner=inner)

    result = builder.build(spec)

    # inner.build の戻り値がそのまま返る。
    assert result == ("agent", "bot")
    assert len(inner.specs) == 1
    governed = inner.specs[0]
    # 新 spec（非破壊置換）で、宣言メタは不変。
    assert governed is not spec
    assert governed.name == "bot"
    assert governed.instructions == "i"
    # tools: 同名 FunctionTool だが on_invoke_tool が差し替わった新オブジェクト。
    assert governed.tools[0] is not tool
    assert governed.tools[0].name == tool.name
    assert governed.tools[0].on_invoke_tool is not tool.on_invoke_tool
    # hooks: spec.hooks=None でも監査フックが装着される。
    assert governed.hooks is not None
    # 元 spec は不変。
    assert spec.tools == [tool]
    assert spec.hooks is None


# ----------------------------------------------------------------------
# build: policy / audit_sink の govern_spec への素通し（fake govern_spec）
# ----------------------------------------------------------------------


def test_policy_and_audit_sink_passed_to_govern_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """利用者指定の policy / audit_sink が不透明値のまま `govern_spec` へ渡る。"""
    seen: dict[str, Any] = {}
    sentinel_governed = object()

    def _fake_govern_spec(spec: Any, *, policy: Any, audit_sink: Any = None) -> Any:
        seen.update(spec=spec, policy=policy, audit_sink=audit_sink)
        return sentinel_governed

    monkeypatch.setattr("oai_agentspec._adapters.govern_spec", _fake_govern_spec)
    sentinel_policy = object()
    sentinel_sink = object()
    inner = _RecordingBuilder()
    builder = GovernedAgentBuilder(policy=sentinel_policy, audit_sink=sentinel_sink, inner=inner)
    spec = AgentSpec(name="a", instructions="x")

    builder.build(spec)

    assert seen["spec"] is spec
    assert seen["policy"] is sentinel_policy
    assert seen["audit_sink"] is sentinel_sink
    # govern_spec の戻り値（govern 済み spec）が inner へ渡る。
    assert inner.specs == [sentinel_governed]


def test_inner_none_uses_default_agent_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """`inner=None` のとき `_adapters` の `DefaultAgentBuilder` で Agent 化される。"""
    created: list[Any] = []
    sentinel_governed = object()

    class _FakeDefaultBuilder:
        def __init__(self) -> None:
            created.append(self)
            self.specs: list[Any] = []

        def build(self, spec: Any) -> str:
            self.specs.append(spec)
            return "built-by-default"

    monkeypatch.setattr("oai_agentspec._adapters.DefaultAgentBuilder", _FakeDefaultBuilder)
    monkeypatch.setattr(
        "oai_agentspec._adapters.govern_spec",
        lambda spec, *, policy, audit_sink=None: sentinel_governed,
    )
    builder = GovernedAgentBuilder(policy=object(), audit_sink=object())

    result = builder.build(AgentSpec(name="a", instructions="x"))

    assert result == "built-by-default"
    assert len(created) == 1
    assert created[0].specs == [sentinel_governed]


# ----------------------------------------------------------------------
# audit_sink プロパティ: build 前の値・既定 sink の生成と共有
# ----------------------------------------------------------------------


def test_audit_sink_property_before_build_returns_user_value_or_none() -> None:
    """build 前は利用者指定の sink、未指定なら None を返す（AGT は import しない）。"""
    sentinel_sink = object()
    assert GovernedAgentBuilder(policy=object(), audit_sink=sentinel_sink).audit_sink is (
        sentinel_sink
    )
    assert GovernedAgentBuilder(policy=object()).audit_sink is None


def test_default_sink_created_on_first_build_and_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    """`audit_sink=None` の既定 sink は初回 build で 1 度だけ生成され以降共有される。"""
    sentinel_sink = object()
    new_sink_calls: list[None] = []
    sinks_seen: list[Any] = []

    def _fake_new_audit_sink() -> Any:
        new_sink_calls.append(None)
        return sentinel_sink

    def _fake_govern_spec(spec: Any, *, policy: Any, audit_sink: Any = None) -> Any:
        sinks_seen.append(audit_sink)
        return spec

    monkeypatch.setattr("oai_agentspec._adapters.new_audit_sink", _fake_new_audit_sink)
    monkeypatch.setattr("oai_agentspec._adapters.govern_spec", _fake_govern_spec)
    builder = GovernedAgentBuilder(policy=object(), inner=_RecordingBuilder())

    builder.build(AgentSpec(name="a", instructions="x"))
    builder.build(AgentSpec(name="b", instructions="x"))

    # 生成は初回 build の 1 回のみ・両 build で同一 sink が govern_spec へ渡る（チェーン連続）。
    assert len(new_sink_calls) == 1
    assert sinks_seen == [sentinel_sink, sentinel_sink]
    assert builder.audit_sink is sentinel_sink


def test_explicit_audit_sink_skips_default_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """利用者指定 sink があるときは既定 sink を生成しない（`new_audit_sink` 非呼出）。"""
    sinks_seen: list[Any] = []

    def _fail_new_audit_sink() -> Any:
        raise AssertionError("audit_sink 指定時に new_audit_sink が呼ばれた")

    monkeypatch.setattr("oai_agentspec._adapters.new_audit_sink", _fail_new_audit_sink)
    monkeypatch.setattr(
        "oai_agentspec._adapters.govern_spec",
        lambda spec, *, policy, audit_sink=None: sinks_seen.append(audit_sink) or spec,
    )
    sentinel_sink = object()
    builder = GovernedAgentBuilder(
        policy=object(), audit_sink=sentinel_sink, inner=_RecordingBuilder()
    )

    builder.build(AgentSpec(name="a", instructions="x"))

    assert sinks_seen == [sentinel_sink]
    assert builder.audit_sink is sentinel_sink


# ----------------------------------------------------------------------
# extra 未導入耐性: コンストラクトは成功し、build で install hint 付き ImportError
# ----------------------------------------------------------------------


def test_constructor_does_not_require_agt_and_build_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AGT 未導入相当（`_require_agt` 失敗）でも構築でき、build で案内付き ImportError。"""

    def _raise_import_error() -> Any:
        raise ImportError(_GOVERNANCE_INSTALL_HINT)

    monkeypatch.setattr("oai_agentspec._adapters.governance._require_agt", _raise_import_error)
    # __init__ は AGT を import しない（extra 未導入でもコンストラクト可能）。
    builder = GovernedAgentBuilder(policy=object())
    assert builder.audit_sink is None
    # build（既定 sink 生成 = new_audit_sink → _require_agt）で初めて失敗する。
    with pytest.raises(ImportError, match=r"oai-agentspec\[governance\]"):
        builder.build(AgentSpec(name="a", instructions="x"))


# ----------------------------------------------------------------------
# overrides: per-agent ポリシーの選択・フォールバック・未適用キー検知
# ----------------------------------------------------------------------


def _patch_govern_spec_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, Any]]:
    """`govern_spec` を記録 fake に差し替え、(spec 名, policy) の適用履歴を返す。"""
    applied: list[tuple[str, Any]] = []

    def _fake_govern_spec(spec: Any, *, policy: Any, audit_sink: Any = None) -> Any:
        applied.append((spec.name, policy))
        return spec

    monkeypatch.setattr("oai_agentspec._adapters.govern_spec", _fake_govern_spec)
    monkeypatch.setattr("oai_agentspec._adapters.new_audit_sink", lambda: object())
    return applied


def test_override_policy_selected_for_listed_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """overrides 掲載エージェントは override ポリシー・未掲載は既定へフォールバックする。"""
    applied = _patch_govern_spec_recorder(monkeypatch)
    default_policy = object()
    support_policy = object()
    builder = GovernedAgentBuilder(
        policy=default_policy,
        overrides={"support": support_policy},
        inner=_RecordingBuilder(),
    )

    builder.build(AgentSpec(name="triage", instructions="x"))
    builder.build(AgentSpec(name="support", instructions="x"))

    assert applied == [("triage", default_policy), ("support", support_policy)]


def test_override_key_matching_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """overrides キーの引き当ては `spec.name` との完全一致のみ（正規化なし）。"""
    applied = _patch_govern_spec_recorder(monkeypatch)
    default_policy = object()
    override_policy = object()
    builder = GovernedAgentBuilder(
        policy=default_policy,
        overrides={"Support": override_policy},  # 大文字始まり（不一致）
        inner=_RecordingBuilder(),
    )

    builder.build(AgentSpec(name="support", instructions="x"))

    # 大文字小文字は正規化されず既定へフォールバックする。
    assert applied == [("support", default_policy)]
    assert builder.unapplied_overrides == frozenset({"Support"})


def test_unapplied_overrides_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """`unapplied_overrides` は build 前=全キー・適用後に減少・typo キーは残留する。"""
    _patch_govern_spec_recorder(monkeypatch)
    builder = GovernedAgentBuilder(
        policy=object(),
        overrides={"support": object(), "suport": object()},  # "suport" は typo 相当
        inner=_RecordingBuilder(),
    )

    # build 前は全キーが未適用。
    assert builder.unapplied_overrides == frozenset({"support", "suport"})

    builder.build(AgentSpec(name="support", instructions="x"))
    builder.build(AgentSpec(name="triage", instructions="x"))

    # 適用済みキーは除かれ、typo キーのみ残る（検知の根拠）。
    assert builder.unapplied_overrides == frozenset({"suport"})


def test_unapplied_overrides_empty_without_overrides() -> None:
    """overrides 未指定なら `unapplied_overrides` は空集合（既存利用と完全互換）。"""
    assert GovernedAgentBuilder(policy=object()).unapplied_overrides == frozenset()


def test_overrides_mapping_is_copied(monkeypatch: pytest.MonkeyPatch) -> None:
    """渡した overrides を後から書き換えても builder の引き当てに影響しない（防御的コピー）。"""
    applied = _patch_govern_spec_recorder(monkeypatch)
    default_policy = object()
    override_policy = object()
    mapping: dict[str, Any] = {"bot": override_policy}
    builder = GovernedAgentBuilder(
        policy=default_policy, overrides=mapping, inner=_RecordingBuilder()
    )

    mapping.clear()  # 外部で書き換え
    builder.build(AgentSpec(name="bot", instructions="x"))

    assert applied == [("bot", override_policy)]


# ----------------------------------------------------------------------
# from_yaml: extra 未導入相当では install hint 付き ImportError
# ----------------------------------------------------------------------


def test_from_yaml_missing_extra_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """bundle 構築（from_yaml）は AGT 未導入相当で install hint 付き ImportError を送出する。"""

    def _raise_import_error(path: Any) -> Any:
        raise ImportError(_GOVERNANCE_INSTALL_HINT)

    monkeypatch.setattr("oai_agentspec._adapters.load_policy_bundle", _raise_import_error)
    with pytest.raises(ImportError, match=r"oai-agentspec\[governance\]"):
        GovernedAgentBuilder.from_yaml(tmp_path / "governance.yaml")


def test_override_not_marked_applied_when_build_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """override 適用 build が失敗した場合はキーを適用済みにしない（失敗 override の診断可能性）。"""

    def _raising_govern_spec(spec: Any, *, policy: Any, audit_sink: Any = None) -> Any:
        raise ValueError("policy load failed")

    monkeypatch.setattr("oai_agentspec._adapters.govern_spec", _raising_govern_spec)
    monkeypatch.setattr("oai_agentspec._adapters.new_audit_sink", lambda: object())
    builder = GovernedAgentBuilder(
        policy=object(), overrides={"bot": object()}, inner=_RecordingBuilder()
    )

    with pytest.raises(ValueError, match="policy load failed"):
        builder.build(AgentSpec(name="bot", instructions="x"))

    # 失敗した override は未適用のまま残る（成功後にのみ適用済み記録）。
    assert builder.unapplied_overrides == frozenset({"bot"})
