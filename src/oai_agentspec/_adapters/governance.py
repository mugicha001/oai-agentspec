"""AGT ガバナンス統合窓口（ツール単位ポリシー強制 + 監査を `_adapters` に閉じる・NFR-1）。

`from agents import ...` と `agent-governance-toolkit`（AGT）の import を本モジュールに局在化する。
`govern_spec` は宣言層の `AgentSpec` を受け、各 `FunctionTool` の `on_invoke_tool` をポリシー評価
付きラップへ非破壊置換し（許可なら実関数を実行・違反なら実関数を実行せず `PolicyViolationError` を
送出）、ライフサイクル監査を記録する合成 `AgentHooks` を `spec.hooks` と合成した新 `AgentSpec` を
返す（build-don't-run・実行は SDK Runner に委ねる）。

ポリシー評価・監査 sink・拒否例外は AGT の `[openai-agents]` 連携（`openai_agents_trust` の
`GovernancePolicy` / `AuditLog` と core の `PolicyViolationError`）をそのまま使い、自前で再実装しな
い。SDK 型を知る FunctionTool ラップと監査 `AgentHooks` 合成のみ本モジュールが担う（AGT は build 時
結線用の FunctionTool ラッパ / `AgentHooks` を提供しないため）。AGT の import は関数内遅延に閉じ、未
導入時は install hint 付き `ImportError`（`_adapters/lightning.py` の `_require_agentlightning` /
`_LIGHTNING_INSTALL_HINT` と同型）。policy / audit_sink は引数 DI で受け、env 参照は持たない。
"""

from __future__ import annotations

import inspect
import json
import os
import re
import warnings
from dataclasses import MISSING as _MISSING
from dataclasses import fields as _dataclass_fields
from dataclasses import replace as _dataclass_replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents import AgentHooks, FunctionTool

# govern ラップが元 on_invoke_tool の第 1 引数注釈（'ToolContext[Any]' 等の文字列注釈）を
# 引き継いだ際、SDK 側の get_type_hints が本モジュールの globals で解決できるようにするための
# import（本文では直接参照しない）。
from agents.run_context import RunContextWrapper  # noqa: F401
from agents.tool_context import ToolContext  # noqa: F401

if TYPE_CHECKING:
    from ..spec import AgentSpec

# governance extra（agent-governance-toolkit）未導入時の案内。
_GOVERNANCE_INSTALL_HINT = (
    "AGT ガバナンス（ツール単位ポリシー強制と監査）には agent-governance-toolkit が必要です。"
    "次でインストールしてください: pip install 'oai-agentspec[governance]'"
)

# `govern_spec` が実際に強制するポリシーフィールド（MVP）。`allowed_tools` は `check_tool`
# （ツール名 allowlist）、`blocked_patterns` は `check_content`（ツール引数 JSON への照合）で
# 評価される。これ以外のフィールドは本統合では強制されない（YAML ロード時に警告する）。
_ENFORCED_POLICY_FIELDS = frozenset({"allowed_tools", "blocked_patterns"})

# 非強制だが警告不要のメタフィールド（違反メッセージ等に使うだけで挙動には影響しない）。
_BENIGN_POLICY_FIELDS = frozenset({"name"})

# policy オブジェクトに必須の評価メソッド（build 時に存在を検証する）。
_REQUIRED_POLICY_METHODS = ("check_tool", "check_content")


def _require_agt() -> tuple[Any, Any, Any]:
    """AGT の openai-agents 連携シンボルを遅延 import する（未導入時は案内付き ImportError）。

    ポリシー型 / 監査 sink 型は integrations 側（`openai_agents_trust`）、拒否例外は core 側
    （`agent_os.exceptions`）に居る。core パッケージ（`agent_os`）の import は legacy パッケージ名を
    告知する `DeprecationWarning` を出すため、ノイズ抑止のため import 中のみ抑制する。

    Returns:
        `(GovernancePolicy, AuditLog, PolicyViolationError)` の 3 つ組（いずれも AGT 型）。

    Raises:
        ImportError: agent-governance-toolkit が未導入の場合（案内文字列付き）。
    """
    try:
        from openai_agents_trust import AuditLog, GovernancePolicy

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from agent_os.exceptions import PolicyViolationError
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise ImportError(_GOVERNANCE_INSTALL_HINT) from exc
    return GovernancePolicy, AuditLog, PolicyViolationError


def new_audit_sink() -> Any:
    """AGT 既定の監査 sink（tamper-evident な `AuditLog`）を 1 つ生成して返す。

    AGT の遅延 import を維持するため `_require_agt()` 経由で `AuditLog` を構築する。
    `GovernedAgentBuilder` が既定 sink を build 間で共有するためのファクトリ（spec ごとに sink を
    分断させずハッシュチェーンを連続させる）。

    Returns:
        新しい AGT `AuditLog` インスタンス。

    Raises:
        ImportError: governance extra（agent-governance-toolkit）が未導入の場合（案内付き）。
    """
    _, audit_log_cls, _ = _require_agt()
    return audit_log_cls()


def resolve_policy(policy: object) -> Any:
    """ポリシー定義（YAML パス or オブジェクト）を評価可能なポリシーオブジェクトへ解決する。

    `GovernedAgentBuilder` が各ポリシーを **1 度だけ** 読み込み・検証してスナップショットとして
    build 間で共有するための入口。YAML パスを build のたびに再読込すると、同一 registry 解決内で
    エージェント間のポリシー不整合（読み込みタイミング差・TOCTOU）や非強制フィールド警告の重複
    発火が起きるため、解決済みオブジェクトへ正規化してから使う。検証内容は `_load_policy` と同一
    （オブジェクトの素通し検証を含む・冪等）。

    Args:
        policy: YAML ファイルパス、または AGT ポリシー互換オブジェクト。

    Returns:
        解決済みのポリシーオブジェクト。

    Raises:
        ImportError: governance extra（agent-governance-toolkit）が未導入の場合（案内付き）。
        FileNotFoundError: YAML パスが存在しない場合。
        ValueError: YAML の構造・キー・値形状が不正な場合。
        TypeError: policy オブジェクトに callable な `check_tool` / `check_content` が無い場合。
    """
    governance_policy, _, _ = _require_agt()
    return _load_policy(policy, governance_policy)


def policy_violation_error_type() -> type[Exception]:
    """AGT のポリシー違反例外クラス（`PolicyViolationError`）を返す。

    `oai_agentspec.runtime.governance` 公開窓口からの再エクスポートに使う取得口。AGT の import は
    `_require_agt` の関数内遅延に閉じ、core パッケージ import 時の `DeprecationWarning` も同所で
    抑制済みのため、利用側は警告抑制ボイラープレートなしで例外クラスを取得できる。

    Returns:
        AGT が送出する `PolicyViolationError` クラスそのもの（isinstance 互換）。

    Raises:
        ImportError: governance extra（agent-governance-toolkit）が未導入の場合（案内付き）。
    """
    _, _, policy_violation_error = _require_agt()
    return policy_violation_error


def _field_default(field: Any) -> Any:
    """dataclass フィールドの既定値を返す（default / default_factory どちらにも対応）。

    既定値が定義されていない（必須）フィールドは `dataclasses.MISSING` を返す。

    Args:
        field: dataclass の `Field` オブジェクト。

    Returns:
        フィールドの既定値。未定義なら `dataclasses.MISSING`。
    """
    if field.default is not _MISSING:
        return field.default
    if field.default_factory is not _MISSING:  # type: ignore[misc]
        return field.default_factory()
    return _MISSING


def _warn_non_enforced_fields(raw: dict[str, Any], fields: dict[str, Any]) -> None:
    """YAML で本統合が強制しないフィールドが既定値以外で指定された場合に警告する。

    `allowed_tools`（`check_tool`）と `blocked_patterns`（`check_content`）のみが強制対象。
    それ以外（`max_tokens` / `max_tool_calls` / `min_trust_score` / `require_identity` 等）を
    既定値以外で指定しても silent no-op になるため、false sense of security を防ぐべく警告する
    （`name` 等のメタフィールドは挙動に影響しないため対象外）。

    Args:
        raw: YAML から読み込んだ生のマッピング（既知キーのみ・未知キーは呼び出し側で除外済み）。
        fields: フィールド名 -> `Field` の mapping（既定値の参照に使う）。
    """
    for key, value in raw.items():
        if key in _ENFORCED_POLICY_FIELDS or key in _BENIGN_POLICY_FIELDS:
            continue
        default = _field_default(fields[key])
        if default is _MISSING or value != default:
            warnings.warn(
                f"governance policy フィールド {key!r} は本統合では強制されません"
                "（強制対象は allowed_tools / blocked_patterns のみ）。指定値は無視されます",
                RuntimeWarning,
                stacklevel=2,
            )


def _check_policy_object(policy: object) -> None:
    """policy オブジェクトが評価メソッドを持つことを build 時に検証する（fail-fast）。

    `check_tool` / `check_content` が callable でない場合は `TypeError` を即時送出する
    （現状の素通しだと最初のツール呼び出しまで `AttributeError` が遅延するため）。

    Args:
        policy: 検証対象のポリシーオブジェクト。

    Raises:
        TypeError: `check_tool` / `check_content` のいずれかが callable でない場合。
    """
    for method in _REQUIRED_POLICY_METHODS:
        if not callable(getattr(policy, method, None)):
            raise TypeError(
                f"governance policy オブジェクトには callable な {method!r} が必要です"
                "（YAML パス、または AGT GovernancePolicy 互換オブジェクトを渡してください）: "
                f"{type(policy).__name__}"
            )


def _load_yaml_mapping(path: str | os.PathLike[str], *, what: str) -> dict[str, Any]:
    """YAML ファイルをマッピングとして読み込む（空 / 非マッピングは fail-fast）。

    空ファイル・空ドキュメント（None ルート）は「制限なしの全既定ポリシーへ静かに化ける」
    footgun のため `ValueError` で拒否する（意図的に制限を置かない場合も `name:` 等を持つ明示の
    マッピングとして書く）。`[]` / `false` / `0` 等の falsy 非マッピングも型エラーとして拒否する。

    Args:
        path: YAML ファイルパス。
        what: エラーメッセージに使う対象名（例: "governance policy YAML"）。

    Returns:
        読み込んだマッピング。

    Raises:
        FileNotFoundError: パスが存在しない場合。
        yaml.YAMLError: YAML の構文が不正な場合。
        ValueError: ルートが空（None）またはマッピングでない場合。
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"{what} が空です（制限なしの意図でも明示のマッピングを書いてください）")
    if not isinstance(raw, dict):
        raise ValueError(f"{what} はマッピングである必要があります: {type(raw).__name__}")
    return raw


def _load_policy(policy: object, policy_cls: Any) -> Any:
    """`policy`（YAML パス or ポリシーオブジェクト）を AGT `GovernancePolicy` へ解決する。

    `str` / `os.PathLike` のときは YAML を読み、`GovernancePolicy` のフィールドで構築する
    （`pyyaml` はコア依存）。未知キーは黙殺せず `ValueError`（`allowed_tool:` のような typo が
    allowlist 無効化 = 全ツール許可へ化ける footgun を防ぐ・`build_agent` の extra 未知キー →
    `ValueError` と整合）。空 YAML / falsy 非マッピングも `ValueError`（全既定 = 全許可への
    無言フォールバック防止）。強制されないフィールドの指定は警告する
    （`_warn_non_enforced_fields`）。それ以外（既構築のポリシーオブジェクト）は評価メソッドの
    存在を検証してそのまま返す（duck typing・build 時 fail-fast）。

    Args:
        policy: YAML ファイルパス、または AGT ポリシーオブジェクト。
        policy_cls: AGT の `GovernancePolicy` クラス（YAML 構築先）。

    Returns:
        解決済みのポリシーオブジェクト。

    Raises:
        FileNotFoundError: YAML パスが存在しない場合。
        yaml.YAMLError: YAML の構文が不正な場合。
        ValueError: YAML が空 / マッピングでない / 未知キー・非文字列キーを含む /
            強制対象フィールドの値形状が不正な場合。
        TypeError: policy オブジェクトに callable な `check_tool` / `check_content` が無い場合。
    """
    if not isinstance(policy, (str, os.PathLike)):
        _check_policy_object(policy)
        return policy

    raw = _load_yaml_mapping(policy, what="governance policy YAML")
    return _policy_from_mapping(raw, policy_cls, context="governance policy YAML")


def _validate_enforced_field_values(raw: dict[str, Any], *, context: str) -> None:
    """強制対象フィールド（allowed_tools / blocked_patterns）の値形状を検証する。

    値の型 typo は黙殺すると致命的に化ける: スカラ文字列の `allowed_tools` は AGT の
    `in` 判定が**部分文字列照合**になり意図しないツール名を許可し、スカラ文字列の
    `blocked_patterns` は 1 文字ずつ正規表現として照合される。また不正な正規表現は実行時の
    最初のツール呼び出しまで顕在化しないため、ロード時に compile 検証する（fail-fast）。

    Args:
        raw: ポリシーフィールドのマッピング（既知キーのみ）。
        context: エラーメッセージに使う文脈。

    Raises:
        ValueError: allowed_tools が文字列リスト（または null）でない / blocked_patterns が
            文字列リストでない / blocked_patterns に compile 不能な正規表現が含まれる場合。
    """
    if "allowed_tools" in raw:
        allowed = raw["allowed_tools"]
        if allowed is not None and (
            not isinstance(allowed, list) or any(not isinstance(t, str) for t in allowed)
        ):
            raise ValueError(
                f"{context} の allowed_tools は文字列のリスト（または null）である必要が"
                f"あります: {allowed!r}"
                "（スカラ文字列は部分文字列照合の allowlist になり意図しないツールを許可します）"
            )
    if "blocked_patterns" in raw:
        patterns = raw["blocked_patterns"]
        if not isinstance(patterns, list) or any(not isinstance(p, str) for p in patterns):
            raise ValueError(
                f"{context} の blocked_patterns は文字列のリストである必要があります: {patterns!r}"
            )
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"{context} の blocked_patterns に不正な正規表現が含まれます: "
                    f"{pattern!r}（{exc}）"
                ) from exc


def _policy_from_mapping(raw: dict[str, Any], policy_cls: Any, *, context: str) -> Any:
    """`GovernancePolicy` フィールドのマッピングからポリシーを構築する（fail-fast 共通部）。

    非文字列キー（YAML 1.1 暗黙型付けで bool / null 化した `on:` / `yes:` / `null:` 等）と
    未知キーは黙殺せず `ValueError`（typo footgun の防止）、強制対象フィールドの値形状も検証し
    （`_validate_enforced_field_values`）、強制されないフィールドの指定は
    `_warn_non_enforced_fields` で警告する。単一ポリシー YAML（`_load_policy`）と bundle YAML の
    各セクション（`load_policy_bundle`）が共用する。

    Args:
        raw: ポリシーフィールドのマッピング（YAML 由来）。
        policy_cls: AGT の `GovernancePolicy` クラス（構築先）。
        context: エラーメッセージに使う文脈（例: "governance policy YAML" /
            "governance bundle YAML の agents['support']"）。

    Returns:
        構築済みのポリシーオブジェクト。

    Raises:
        ValueError: 非文字列キー / 未知キー / 強制対象フィールドの不正な値形状を含む場合。
    """
    non_str_keys = [k for k in raw if not isinstance(k, str)]
    if non_str_keys:
        raise ValueError(
            f"{context} のキーは文字列である必要があります: {non_str_keys!r}"
            "（YAML 1.1 では on / yes / null 等が bool / null に暗黙変換されるため、"
            "キーに使う場合は引用符で囲んでください）"
        )
    fields = {f.name: f for f in _dataclass_fields(policy_cls)}
    unknown = sorted(raw.keys() - fields.keys())
    if unknown:
        raise ValueError(
            f"{context} に未知のキーが含まれます: {unknown}（有効キー: {sorted(fields)}）"
        )
    _validate_enforced_field_values(raw, context=context)
    _warn_non_enforced_fields(raw, fields)
    return policy_cls(**raw)


# bundle YAML のトップレベル有効キー（default は必須・agents は任意）。
_BUNDLE_TOP_KEYS = frozenset({"default", "agents"})


def load_policy_bundle(path: str | os.PathLike[str]) -> tuple[Any, dict[str, Any]]:
    """bundle YAML（`default` + `agents`）を読み、(既定ポリシー, per-agent ポリシー) を構築する。

    制限の全量を 1 ファイルに宣言する形式。`default`（必須）は既定ポリシーのフィールド
    マッピング、`agents`（任意）はエージェント名 -> フィールドマッピングで、各セクションは
    単一ポリシー YAML と同一の fail-fast 検証（未知キー `ValueError`・非強制フィールド警告）を
    受ける。

    ```yaml
    default:
      allowed_tools: [lookup_order]
    agents:
      support:
        allowed_tools: [lookup_order, refund]
    ```

    Args:
        path: bundle YAML のファイルパス。

    Returns:
        `(既定ポリシー, {エージェント名: ポリシー})` の 2 つ組
        （`GovernedAgentBuilder(policy=..., overrides=...)` へそのまま渡せる形）。

    Raises:
        ImportError: governance extra（agent-governance-toolkit）が未導入の場合（案内付き）。
        FileNotFoundError: パスが存在しない場合。
        yaml.YAMLError: YAML の構文が不正な場合。
        ValueError: マッピングでない / トップレベルに未知キーがある / `default` が無い /
            各セクションがマッピングでない / セクションに未知キーがある場合。
    """
    governance_policy, _, _ = _require_agt()

    raw = _load_yaml_mapping(path, what="governance bundle YAML")
    non_str_top = [k for k in raw if not isinstance(k, str)]
    if non_str_top:
        raise ValueError(
            f"governance bundle YAML のトップレベルキーは文字列である必要があります: "
            f"{non_str_top!r}（YAML 1.1 の on / yes / null 等は引用符で囲んでください）"
        )
    unknown_top = sorted(raw.keys() - _BUNDLE_TOP_KEYS)
    if unknown_top:
        raise ValueError(
            f"governance bundle YAML のトップレベルに未知のキーが含まれます: {unknown_top}"
            f"（有効キー: {sorted(_BUNDLE_TOP_KEYS)}）"
        )
    if "default" not in raw:
        raise ValueError("governance bundle YAML には既定ポリシーの 'default' セクションが必要です")

    def _section(section: Any, label: str) -> Any:
        if not isinstance(section, dict):
            raise ValueError(
                f"governance bundle YAML の {label} はマッピングである必要があります: "
                f"{type(section).__name__}"
            )
        return _policy_from_mapping(
            section, governance_policy, context=f"governance bundle YAML の {label}"
        )

    default_policy = _section(raw["default"], "'default'")
    # None（セクション未指定 / 空値）のみ省略扱い。`agents: []` 等の falsy 非マッピングは
    # overrides が黙って消える footgun になるため型エラーとして拒否する（fail-fast）。
    agents_raw = raw.get("agents")
    if agents_raw is None:
        agents_raw = {}
    if not isinstance(agents_raw, dict):
        raise ValueError(
            "governance bundle YAML の 'agents' はマッピングである必要があります: "
            f"{type(agents_raw).__name__}"
        )
    non_str_agents = [k for k in agents_raw if not isinstance(k, str)]
    if non_str_agents:
        raise ValueError(
            "governance bundle YAML の agents キー（エージェント名）は文字列である必要が"
            f"あります: {non_str_agents!r}"
            "（YAML 1.1 の on / yes / null 等のエージェント名は引用符で囲んでください）"
        )
    agent_policies = {
        name: _section(section, f"agents[{name!r}]") for name, section in agents_raw.items()
    }
    return default_policy, agent_policies


def _iter_decoded_strings(value: Any) -> list[str]:
    """パース済み JSON 構造からデコード済み文字列スカラ（と dict キー）を集める。

    blocked_patterns を「ツールが実際に受け取る値」へ照合するための候補列。JSON ワイヤ文字列上
    では実改行が `\\n`（バックスラッシュ + n）のエスケープ表現になり `\\s` 系パターンが回避できる
    ため、デコード済みの実文字列にも照合する。深いネストでも落ちないよう再帰でなく明示スタックで
    走査する。

    Args:
        value: `json.loads` 済みの値。

    Returns:
        構造内の文字列スカラと dict キー（文字列のもの）のリスト。
    """
    out: list[str] = []
    stack: list[Any] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            for key, child in item.items():
                if isinstance(key, str):
                    out.append(key)
                stack.append(child)
        elif isinstance(item, list):
            stack.extend(item)
    return out


def _evaluate_tool(policy: Any, tool_name: str, input_json: str) -> str | None:
    """ツール呼び出し（関数名 + 引数）をポリシー評価し、違反理由（あれば）を返す。

    まずツール名を allowlist（`check_tool`）で照合し、許可なら引数 JSON を blocked_patterns
    （`check_content`）で照合する。いずれかが違反理由を返したらそれを返す（許可なら None）。

    `input_json` は SDK が渡すツール引数の生 JSON 文字列（LLM 出力でありインジェクション誘導可能）。
    照合候補は 3 系統: (1) 生ワイヤ文字列、(2) パース可能なら `json.dumps(parsed,
    ensure_ascii=False)` の正規化文字列（`\\u0072m` = `rm` のようなエスケープ別表現の中和）、
    (3) デコード済みの文字列スカラ群（実改行・タブ等の制御文字に対する `\\s` 系パターンの照合・
    エスケープ表現と実文字の差による回避の封鎖）。**いずれか**が違反を返したら deny する
    （fail-closed）。パース不能時（深すぎるネストによる `RecursionError` を含む）は生文字列のみの
    照合へフォールバックし、評価自体は失敗させない（監査記録前のクラッシュ防止）。

    Args:
        policy: AGT ポリシーオブジェクト（`check_tool` / `check_content` を持つ）。
        tool_name: 評価対象のツール名。
        input_json: ツール引数の生 JSON 文字列。

    Returns:
        違反理由の文字列。許可なら None。
    """
    reason = policy.check_tool(tool_name)
    if reason is not None:
        return str(reason)
    candidates = [input_json]
    try:
        parsed = json.loads(input_json)
    except (ValueError, TypeError, RecursionError):
        parsed = None
    if parsed is not None:
        try:
            candidates.append(json.dumps(parsed, ensure_ascii=False))
        except (ValueError, TypeError, RecursionError):  # pragma: no cover - 深さ依存の防御
            pass
        candidates.extend(_iter_decoded_strings(parsed))
    for text in dict.fromkeys(candidates):
        content_reason = policy.check_content(text)
        if content_reason is not None:
            return str(content_reason)
    return None


def _govern_tool(
    tool: FunctionTool,
    *,
    policy: Any,
    sink: Any,
    denied_exc: Any,
    agent_name: str,
) -> FunctionTool:
    """`FunctionTool` の `on_invoke_tool` をポリシー評価付きラップへ非破壊置換した新 tool を返す。

    実行時、ツール呼び出し直前にポリシーを評価し、許可なら監査 sink に "allow" を記録して実関数を
    実行する。違反なら "deny" を記録し、実関数を実行せず `denied_exc`（AGT `PolicyViolationError`）
    を送出する。`name` / `description` / `params_json_schema` / `needs_approval` 等の宣言メタは維持
    し、差し替えるのは実行本体のみ（`mock_spec_tools` / `attach_tool_guardrails` と同型の非破壊）。

    元 `on_invoke_tool` の第 1 引数注釈はラッパーへ引き継ぐ。SDK は本注釈で渡すコンテキスト型
    （full `ToolContext` か縮約 `RunContextWrapper` か）を選ぶため、注釈を `Any` のままにすると
    `RunContextWrapper` 契約のツールに full `ToolContext` が渡り、SDK の縮約（実行時メタの漏えい
    防止）が無効化される。

    Args:
        tool: ラップ対象の `FunctionTool`。
        policy: AGT ポリシーオブジェクト。
        sink: 監査 sink（`record(agent_id, action, decision, details)` を持つ）。
        denied_exc: ポリシー違反時に送出する例外クラス（AGT `PolicyViolationError`）。
        agent_name: 監査記録に使うエージェント名（`spec.name`）。

    Returns:
        govern ラップ済みの新しい `FunctionTool`（元 tool は不変）。
    """
    original = tool.on_invoke_tool
    tool_name = tool.name

    async def _on_invoke_tool(ctx: Any, input_json: str) -> Any:
        reason = _evaluate_tool(policy, tool_name, input_json)
        if reason is not None:
            sink.record(
                agent_id=agent_name,
                action=f"tool:{tool_name}",
                decision="deny",
                details={"reason": reason, "arguments": input_json},
            )
            raise denied_exc(f"governance denied tool {tool_name!r}: {reason}")
        sink.record(
            agent_id=agent_name,
            action=f"tool:{tool_name}",
            decision="allow",
            details={"arguments": input_json},
        )
        return await original(ctx, input_json)

    try:
        first = next(iter(inspect.signature(original).parameters.values()))
        if first.annotation is not inspect.Parameter.empty:
            _on_invoke_tool.__annotations__["ctx"] = first.annotation
    except (StopIteration, TypeError, ValueError):  # pragma: no cover - 異形シグネチャの防御
        pass

    return _dataclass_replace(tool, on_invoke_tool=_on_invoke_tool)


async def _delegate(inner: Any, method: str, *args: Any) -> None:
    """`inner`（既存 `spec.hooks`）の同名ライフサイクルメソッドへ委譲する（await 可能なら await）。

    `inner` が None、または当該メソッドを持たない場合は何もしない（合成フックが既存フックの
    振る舞いを失わせないための薄い委譲）。

    Args:
        inner: 既存の `spec.hooks`（None 可）。
        method: 委譲先のライフサイクルメソッド名。
        *args: そのメソッドへ渡す引数。
    """
    if inner is None:
        return
    fn = getattr(inner, method, None)
    if fn is None:
        return
    result = fn(*args)
    if inspect.isawaitable(result):
        await result


def _make_audit_hooks(sink: Any, inner: Any) -> AgentHooks[Any]:
    """ライフサイクル事象を監査 sink へ記録し既存 `spec.hooks` へ委譲する合成 `AgentHooks` を作る。

    各ライフサイクルメソッドで「監査記録 → 既存フックの同名メソッドへ委譲」の順に呼ぶ
    （上書きでなく合成）。`inner`（既存 `spec.hooks`）が None のときは監査記録のみ行う。
    `on_llm_start` / `on_llm_end` は監査対象外だが、既存フックを失わないため委譲だけ行う。

    Args:
        sink: 監査 sink（`record(agent_id, action, decision, details)` を持つ）。
        inner: 既存の `spec.hooks`（None 可）。

    Returns:
        監査 + 委譲を行う合成 `AgentHooks` インスタンス。
    """

    class _AuditAgentHooks(AgentHooks[Any]):
        """監査記録を上乗せし既存フックへ委譲する合成 `AgentHooks`。"""

        async def on_start(self, context: Any, agent: Any) -> None:
            sink.record(agent_id=agent.name, action="agent_start", decision="allow")
            await _delegate(inner, "on_start", context, agent)

        async def on_end(self, context: Any, agent: Any, output: Any) -> None:
            sink.record(agent_id=agent.name, action="agent_end", decision="allow")
            await _delegate(inner, "on_end", context, agent, output)

        async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
            sink.record(
                agent_id=agent.name,
                action=f"tool_start:{getattr(tool, 'name', '')}",
                decision="allow",
            )
            await _delegate(inner, "on_tool_start", context, agent, tool)

        async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
            sink.record(
                agent_id=agent.name,
                action=f"tool_end:{getattr(tool, 'name', '')}",
                decision="allow",
            )
            await _delegate(inner, "on_tool_end", context, agent, tool, result)

        async def on_handoff(self, context: Any, agent: Any, source: Any) -> None:
            sink.record(
                agent_id=getattr(source, "name", ""),
                action=f"handoff:{getattr(agent, 'name', '')}",
                decision="allow",
            )
            await _delegate(inner, "on_handoff", context, agent, source)

        async def on_llm_start(
            self, context: Any, agent: Any, system_prompt: Any, input_items: Any
        ) -> None:
            await _delegate(inner, "on_llm_start", context, agent, system_prompt, input_items)

        async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
            await _delegate(inner, "on_llm_end", context, agent, response)

    return _AuditAgentHooks()


def govern_spec(
    spec: AgentSpec,
    *,
    policy: object,
    audit_sink: object | None = None,
) -> AgentSpec:
    """`spec.tools` を govern ラップし、監査 `AgentHooks` を `spec.hooks` と合成した新 spec を返す。

    各 `FunctionTool` の `on_invoke_tool` を `dataclasses.replace` でポリシー評価付きラップへ
    非破壊置換し（違反は実関数を実行せず `PolicyViolationError` を送出）、`FunctionTool` 以外
    （hosted tool 等）は素通しする。監査 `AgentHooks` を生成し、`spec.hooks` があれば
    「監査記録 → 既存フックへ委譲」の順で呼ぶ合成フックを作る（`spec.hooks is None` なら監査単体）。
    `spec.handoffs` は変更せず、tools / hooks のみ置換した新 `AgentSpec` を返す（元 spec は不変・
    build-don't-run で実行は SDK Runner に委ねる）。

    監査の `details` には**ツール引数 JSON が全文記録される**（rationale の「誰が・どの引数で呼んだ
    か」を残す監査要件どおり）。機密引数を扱う場合は記録先を `audit_sink` で選定して考慮する。

    既知の境界（govern 対象外）: `sub_agents` の as_tool は registry が build 後に注入するため
    per-call の allow/deny 評価・監査レコードを持たない（監査フックの tool_start / tool_end 記録の
    み・サブエージェント自身の内部 `FunctionTool` は別途 build されていれば govern 済み）。
    `register_factory` 経路は builder を通らないため govern 対象外。SDK の HITL 承認
    （`needs_approval`）はツール実行前の承認フローとして govern ラップ（実行本体）より**先に**
    走るため、ポリシーが拒否する呼び出しでも承認要求は先に発生し得る（承認後に deny される。
    `needs_approval` の宣言メタは不変に維持する方針のため、承認前に弾きたい場合はポリシー対象と
    承認対象のツールを設計で分ける）。

    Args:
        spec: govern 対象の `AgentSpec`（plain・コア型）。
        policy: ポリシー定義（YAML ファイルパス、または AGT ポリシーオブジェクト）。
        audit_sink: 監査ログ出力先（`record(...)` を持つ任意オブジェクト）。None で AGT 既定
            （`AuditLog`・tamper-evident なハッシュチェーン）を新規生成する。

    Returns:
        tools / hooks を govern 化した新しい `AgentSpec`。

    Raises:
        ImportError: governance extra（agent-governance-toolkit）が未導入の場合（案内付き）。
        FileNotFoundError: policy が指す YAML パスが存在しない場合。
        yaml.YAMLError: policy が指す YAML の構文が不正な場合。
        ValueError: policy YAML がマッピングでない、または未知キーを含む場合。
        TypeError: policy オブジェクトに callable な `check_tool` / `check_content` が無い場合。
    """
    governance_policy, audit_log_cls, policy_violation_error = _require_agt()
    policy_obj = _load_policy(policy, governance_policy)
    sink = audit_sink if audit_sink is not None else audit_log_cls()
    agent_name = spec.name

    new_tools: list[Any] = []
    for tool in spec.tools:
        if isinstance(tool, FunctionTool):
            new_tools.append(
                _govern_tool(
                    tool,
                    policy=policy_obj,
                    sink=sink,
                    denied_exc=policy_violation_error,
                    agent_name=agent_name,
                )
            )
        else:
            new_tools.append(tool)

    audit_hooks = _make_audit_hooks(sink, spec.hooks)
    return _dataclass_replace(spec, tools=new_tools, hooks=audit_hooks)
