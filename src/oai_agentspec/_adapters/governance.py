"""AGT ガバナンス統合窓口（ツール単位ポリシー強制 + 監査を `_adapters` に閉じる・NFR-1）。

`from agents import ...` と `agent-governance-toolkit`（AGT）の import を本モジュールに局在化する。
`govern_spec` は宣言層の `AgentSpec` を受け、各 `FunctionTool` の `on_invoke_tool` をポリシー評価
付きラップへ非破壊置換し（許可なら実関数を実行・違反なら実関数を実行せず `PolicyViolationError` を
送出）、ライフサイクル監査を記録する `AgentHooks` を `spec.hooks` と合成した新 `AgentSpec` を
返す（build-don't-run・実行は SDK Runner に委ねる）。オプトイン時に registry 第 3 段 post-process
から呼ばれる `govern_ungoverned_tools`（`sub_agents` の as_tool）/ `govern_agent`（factory Agent）
も同じラップと監査フックを使う。

ポリシー評価・監査 sink・拒否例外は AGT の `[openai-agents]` 連携（`openai_agents_trust` の
`GovernancePolicy` / `AuditLog` と core の `PolicyViolationError`）をそのまま使い、自前で再実装しな
い。SDK 型を知る FunctionTool ラップと監査 `AgentHooks` の生成のみ本モジュールが担う（AGT は build
時結線用の FunctionTool ラッパ / `AgentHooks` を提供しないため）。既存 `spec.hooks` への委譲は本
モジュールで手書きせず `_adapters/hooks.py` の `chain_agent_hooks` に委ねる（委譲実体の一元化）。
AGT の import は関数内遅延に閉じ、未
導入時は install hint 付き `ImportError`（`_adapters/lightning.py` の `_require_agentlightning` /
`_LIGHTNING_INSTALL_HINT` と同型）。policy / audit_sink は引数 DI で受け、env 参照は持たない。
"""

from __future__ import annotations

import inspect
import json
import os
import re
import warnings
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import MISSING as _MISSING
from dataclasses import fields as _dataclass_fields
from dataclasses import replace as _dataclass_replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from agents import (
    AgentHooks,
    FunctionTool,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolOriginType,
)

# `AgentHooksBase` は `agents` トップレベルに export されていないためサブモジュールから import する
# （`_make_audit_hooks` の戻り値注釈用。`agents.AgentHooks` は `TAgent = Agent` を主張するが、
# 合成結果は `inner` 自身になり得るため保証できない）。
from agents.lifecycle import AgentHooksBase

# govern ラップが元 on_invoke_tool の第 1 引数注釈（'ToolContext[Any]' 等の文字列注釈）を
# 引き継いだ際、SDK 側の get_type_hints が本モジュールの globals で解決できるようにするための
# import（本文では直接参照しない）。
from agents.run_context import RunContextWrapper  # noqa: F401

# `get_function_tool_origin` は `agents` トップレベルに export されていないためサブモジュールから
# import する（`_AuditAgentHooks.on_tool_start` の MCP origin 判定用）。
# 以下の 3 つは SDK 非公開要素。`_FailureHandlingFunctionToolInvoker` は `_GovernedInvoker` の
# 継承元で承認強化の型判定の対象、`_SYNC_FUNCTION_TOOL_MARKER` は timeout 設定の検証が参照する
# 印（いずれも ADR-0045 Decision 3）。`_FUNCTION_TOOL_WRAPPED_CALLABLE_MARKER` は `@function_tool`
# が実行関数へ付ける元関数の印で、govern 済みツールの `FunctionTool.__wrapped__` が govern なしと
# 同じ元関数を返すよう実行本体へ写す（SDK はこの印を `__wrapped__` の解決にだけ使い実行経路では
# 読まない。参照箇所は SDK バージョン耐性トリップワイヤのテストで検査する）。
from agents.tool import (
    _FUNCTION_TOOL_WRAPPED_CALLABLE_MARKER,
    _SYNC_FUNCTION_TOOL_MARKER,
    _FailureHandlingFunctionToolInvoker,
    get_function_tool_origin,
)
from agents.tool_context import ToolContext

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

# MCP 由来ツールの引数が str として取れず評価不能なときの拒否理由（fail-closed の固定文言）。
_ARGUMENTS_UNAVAILABLE_REASON = "tool arguments unavailable for policy evaluation"

# 統治ガードレールの name（`RunResult.tool_input_guardrail_results` で内容検査の行と区別する）。
_MCP_GOVERNANCE_GUARDRAIL_NAME = "mcp_governance_guardrail"

# govern ラップ済みの実行本体（`_govern_tool` のラッパ関数、または生成時に自身を登録する
# `_GovernedInvoker`）の登録簿（id -> 弱参照）。判定は登録したオブジェクトとの同一性（`is`）で
# 行う。ラッパ関数の経路は `dataclasses.replace` / `copy` を経ても同じ関数を指し、invoker の経路は
# SDK の再束縛が作る別インスタンスも生成時に登録されるため、どちらも統治済みと判定される。
# 弱参照のためラッパが GC されるとエントリも消える。
_GOVERNED_WRAPPERS: dict[int, weakref.ref[Callable[..., Any]]] = {}


def _register_governed(fn: Callable[..., Any]) -> None:
    """govern ラップ済みの実行本体を統治済みとして `_GOVERNED_WRAPPERS` へ登録する。

    弱参照のコールバックは、同じ id で後から登録された別のエントリを消さない（自分が
    登録したエントリのときだけ削除する）。

    Args:
        fn: 登録する実行本体（`_govern_tool` のラッパ関数、または `_GovernedInvoker`）。
    """
    fn_id = id(fn)

    def _drop(ref: weakref.ref[Callable[..., Any]], key: int = fn_id) -> None:
        if _GOVERNED_WRAPPERS.get(key) is ref:
            del _GOVERNED_WRAPPERS[key]

    _GOVERNED_WRAPPERS[fn_id] = weakref.ref(fn, _drop)


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


def _deny_tool_call(
    *,
    sink: Any,
    agent_name: str | None,
    tool_name: str,
    reason: str,
    arguments: str | None,
    denied_exc: Any,
) -> NoReturn:
    """拒否を監査 sink へ記録してから拒否例外を送出する（記録が送出より前であることを保証）。

    build 時ラップ（`_govern_tool`）と run 時フック（`_make_audit_hooks` の `on_tool_start`）の
    両経路が共有する。監査レコードの形（`action` / `decision` / `details` のキー）と例外メッセージ
    書式を 1 箇所へ集約し、経路ごとに drift しないようにする（形が揃っていることは利用側の
    `action.startswith("tool:")` 抽出の前提であり、ずれても例外は出ない）。

    Args:
        sink: 監査 sink（`record(agent_id, action, decision, details)` を持つ）。
        agent_name: 監査記録に使うエージェント名（`spec.name`）。
        tool_name: 拒否したツールの公開名。
        reason: 拒否理由（ポリシー由来の文言、または評価不能を示す固定文言）。
        arguments: 評価対象の生ワイヤ引数。取得できなかった場合は None を渡す。
        denied_exc: 送出する例外クラス（AGT `PolicyViolationError`）。

    Raises:
        denied_exc: 常に送出する（戻らない）。送出する例外は
            `details={"tool_name": ..., "reason": ...}` を持ち、利用側はメッセージ文言を
            パースせずに拒否ツール名と理由を取得できる。ツール引数は `details` に載せない
            （引数の置き場は監査 sink のまま。ADR-0030）。
    """
    sink.record(
        agent_id=agent_name,
        action=f"tool:{tool_name}",
        decision="deny",
        details={"reason": reason, "arguments": arguments},
    )
    raise denied_exc(
        f"governance denied tool {tool_name!r}: {reason}",
        details={"tool_name": tool_name, "reason": reason},
    )


async def _governed_invoke(
    call: Callable[[Any, str], Awaitable[Any]],
    ctx: Any,
    input_json: str,
    *,
    policy: Any,
    sink: Any,
    denied_exc: Any,
    agent_name: str,
    tool_name: str,
) -> Any:
    """ツール呼び出しをポリシーで評価し、許可なら記録して `call` を実行する（govern の実行手順）。

    invoker 経路（`_GovernedInvoker`）と素の関数経路（`_govern_tool` のラッパ）が共有する唯一の
    実行手順。違反なら "deny" を記録して `denied_exc` を送出し `call` を実行しない。許可なら
    "allow" を記録してから `call` を await する。2 経路で評価・記録の順序と監査の形式を揃える
    （ADR-0045 Decision 1）。

    Args:
        call: 評価後に実行する元の実行本体（SDK の invoker か素の `on_invoke_tool`）。
        ctx: SDK が渡すツールコンテキスト。
        input_json: ツール引数の生 JSON 文字列（評価と記録はこの文字列で行う）。
        policy: AGT ポリシーオブジェクト。
        sink: 監査 sink（`record(agent_id, action, decision, details)` を持つ）。
        denied_exc: ポリシー違反時に送出する例外クラス（AGT `PolicyViolationError`）。
        agent_name: 監査記録に使うエージェント名（`spec.name`）。
        tool_name: 評価・記録に使うツールの公開名。

    Returns:
        `call` の実行結果。
    """
    reason = _evaluate_tool(policy, tool_name, input_json)
    if reason is not None:
        _deny_tool_call(
            sink=sink,
            agent_name=agent_name,
            tool_name=tool_name,
            reason=reason,
            arguments=input_json,
            denied_exc=denied_exc,
        )
    sink.record(
        agent_id=agent_name,
        action=f"tool:{tool_name}",
        decision="allow",
        details={"arguments": input_json},
    )
    return await call(ctx, input_json)


class _GovernedInvoker(_FailureHandlingFunctionToolInvoker):
    """SDK の失敗ハンドラ付き invoker に準拠した govern ラップ（ADR-0045）。

    SDK は `on_invoke_tool` が失敗ハンドラ付き invoker のインスタンスのときだけ、callable な
    `needs_approval` の前に引数を入力モデルで事前検証する。本クラスはその型判定を満たし、govern
    済みツールの承認判定を govern なしと揃える。

    - 基底へ渡す実行本体（`_invoke_tool_impl`）そのものを統治済みの関数にする。SDK が `__call__` を
      通らず `_invoke_tool_impl` を直接呼んでも評価を迂回できない。
    - `__call__` は基底の失敗処理を通さない。deny の例外が `failure_error_function` に吸われず
      伝播し、失敗処理は元の invoker（`inner`）が担う。
    - 事前検証の材料（`__agents_prepare_arguments__`）と元関数の印は `inner` の実装関数から写す。
    - 統治済みの印は `__init__` で登録する。SDK の再束縛（`dataclasses.replace` / `copy` 時の
      `__agents_bind_function_tool__`）が作る別インスタンスも生成時に登録される。
    - policy / sink 等はインスタンス属性に置かずクロージャに閉じる（理由は `__init__` 内の注記）。
    """

    def __init__(
        self,
        inner: _FailureHandlingFunctionToolInvoker,
        *,
        policy: Any,
        sink: Any,
        denied_exc: Any,
        agent_name: str,
        tool_name: str,
        function_tool: FunctionTool | None,
    ) -> None:
        """統治済みの実行本体を組み、基底を初期化して統治済みとして登録する。

        Args:
            inner: 元の `on_invoke_tool`（SDK の失敗ハンドラ付き invoker）。
            policy: AGT ポリシーオブジェクト。
            sink: 監査 sink（`record(agent_id, action, decision, details)` を持つ）。
            denied_exc: ポリシー違反時に送出する例外クラス（AGT `PolicyViolationError`）。
            agent_name: 監査記録に使うエージェント名（`spec.name`）。
            tool_name: 評価・記録に使うツールの公開名。
            function_tool: 束縛先の `FunctionTool`（未束縛なら None）。
        """

        async def governed_impl(ctx: ToolContext[Any], input_json: str) -> Any:
            return await _governed_invoke(
                inner,
                ctx,
                input_json,
                policy=policy,
                sink=sink,
                denied_exc=denied_exc,
                agent_name=agent_name,
                tool_name=tool_name,
            )

        # SDK の承認の事前検証は材料（`__agents_prepare_arguments__`）を `_invoke_tool_impl` の
        # 関数属性から辿るため、inner の実装関数が持つものを写す（写さないと govern 済みツールだけ
        # 事前検証が効かない。ADR-0045 Decision 4）。元関数の印も写し、`__wrapped__` を
        # govern なしと揃える（`__wrapped__` は未統治の元関数を返す。govern なしと同じ公開面）。
        inner_impl = inner._invoke_tool_impl
        for attr in ("__agents_prepare_arguments__", _FUNCTION_TOOL_WRAPPED_CALLABLE_MARKER):
            if hasattr(inner_impl, attr):
                setattr(governed_impl, attr, getattr(inner_impl, attr))

        # policy / sink 等はインスタンス属性に置かずクロージャに閉じる。属性に置くと、SDK の
        # エージェント同一性シグネチャが dataclass の Model 経由で invoker を deepcopy する際、
        # sink が持つロックで pickle に失敗する。
        def rebind(function_tool: FunctionTool) -> _GovernedInvoker:
            return _GovernedInvoker(
                inner.__agents_bind_function_tool__(function_tool),
                policy=policy,
                sink=sink,
                denied_exc=denied_exc,
                agent_name=agent_name,
                tool_name=tool_name,
                function_tool=function_tool,
            )

        super().__init__(governed_impl, inner._on_handled_error, function_tool=function_tool)
        self._rebind = rebind
        if getattr(inner, _SYNC_FUNCTION_TOOL_MARKER, False):
            setattr(self, _SYNC_FUNCTION_TOOL_MARKER, True)
        _register_governed(self)

    async def __call__(self, ctx: ToolContext[Any], input_json: str) -> Any:
        """統治済みの実行本体を基底の失敗処理を通さずに実行する。

        Args:
            ctx: SDK が渡すツールコンテキスト（注釈は基底と同じ。SDK はこの注釈で型を選ぶ）。
            input_json: ツール引数の生 JSON 文字列。

        Returns:
            ツールの実行結果（失敗処理は `inner` が担う）。
        """
        return await self._invoke_tool_impl(ctx, input_json)

    def __agents_bind_function_tool__(self, function_tool: FunctionTool) -> _GovernedInvoker:
        """`function_tool` へ束縛した invoker を返す（SDK の再束縛プロトコル）。

        基底の再束縛は基底クラスのインスタンスを作り、上書きした `__call__` と登録が失われる
        ため使わない。

        Args:
            function_tool: 束縛先の `FunctionTool`。

        Returns:
            束縛先が同じなら self。違えば `inner` を再束縛した新しい `_GovernedInvoker`。
        """
        if self._function_tool is function_tool:
            return self
        return self._rebind(function_tool)


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

    ラップは元 `on_invoke_tool` の種類で 2 経路に分かれる（ADR-0045）。

    - SDK の失敗ハンドラ付き invoker のとき: `_GovernedInvoker` で包む。SDK の承認の事前検証が
      govern なしと同じく効く。コンテキスト型は基底と同じ `ToolContext[Any]` 注釈から解決される。
    - それ以外（利用者が `FunctionTool` を直接組んだ場合等）: 素の async 関数で包み、元
      `on_invoke_tool` の第 1 引数注釈をラッパーへ引き継ぐ。SDK は本注釈で渡すコンテキスト型
      （full `ToolContext` か縮約 `RunContextWrapper` か）を選ぶため、注釈を `Any` のままにすると
      `RunContextWrapper` 契約のツールに full `ToolContext` が渡り、SDK の縮約（実行時メタの
      漏えい防止）が無効化される。

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

    if isinstance(original, _FailureHandlingFunctionToolInvoker):
        # replace の `__post_init__` が再束縛を呼び、新しい tool へ束縛した別インスタンス（生成時に
        # 登録済み）が on_invoke_tool になる。
        governed = _GovernedInvoker(
            original,
            policy=policy,
            sink=sink,
            denied_exc=denied_exc,
            agent_name=agent_name,
            tool_name=tool_name,
            function_tool=None,
        )
        return _dataclass_replace(tool, on_invoke_tool=governed)

    async def _on_invoke_tool(ctx: Any, input_json: str) -> Any:
        return await _governed_invoke(
            original,
            ctx,
            input_json,
            policy=policy,
            sink=sink,
            denied_exc=denied_exc,
            agent_name=agent_name,
            tool_name=tool_name,
        )

    try:
        first = next(iter(inspect.signature(original).parameters.values()))
        if first.annotation is not inspect.Parameter.empty:
            _on_invoke_tool.__annotations__["ctx"] = first.annotation
    except (StopIteration, TypeError, ValueError):  # pragma: no cover - 異形シグネチャの防御
        pass

    _register_governed(_on_invoke_tool)
    return _dataclass_replace(tool, on_invoke_tool=_on_invoke_tool)


def _is_governed(tool: FunctionTool) -> bool:
    """`tool` の実行本体が `_GOVERNED_WRAPPERS` に登録された実行本体そのものかを返す。

    判定は登録したオブジェクトとの同一性（`is`）で行い、hash / eq に依存しない。このため
    関数属性の手付け・`functools.wraps` によるコピー・hash / eq を参照先へ委譲するプロキシは
    統治済みと判定されない（未統治として扱う = fail-closed）。ラッパが GC されるとエントリは
    消える。統治済みの印は「いずれかのポリシーで統治済み」を示し、どのエージェント・どの
    ポリシーで統治したかは区別しない。`_GovernedInvoker` を `copy.deepcopy` した複製は
    `__init__` を通らず登録されないため未統治と判定される（再 govern で評価と記録が二重になる
    fail-closed。`dataclasses.replace` / `copy.copy` は SDK の再束縛で登録される）。

    Args:
        tool: 判定対象の `FunctionTool`。

    Returns:
        `_govern_tool` が作った（登録済みの）実行本体を持つなら True。
    """
    fn = tool.on_invoke_tool
    ref = _GOVERNED_WRAPPERS.get(id(fn))
    return ref is not None and ref() is fn


class _AuditAgentHooks(AgentHooks[Any]):
    """監査記録と MCP 由来ツール評価（`policy` 指定時）を行う `AgentHooks`。

    既存フックへの委譲は `chain_agent_hooks` が担う。`policy` が None のときは監査記録のみを
    行う。統治ガードレール（`mcp_governance_guardrail`）が `data.agent.hooks` から本クラスの
    インスタンスを辿って同じポリシー・sink・`agent_name` で評価できるよう、モジュール水準に置き
    状態をインスタンス属性に持つ。ガードレールで評価済みの呼び出しは `_evaluated` に
    評価したツール名が印として記録され、`on_tool_start` は印の名前が自身の評価に使う名前と
    一致する呼び出しだけ評価を省く（印が無い・名前が一致しない場合は従来どおり評価する安全網。
    倒れる先は二重評価の側）。
    """

    def __init__(
        self,
        *,
        sink: Any,
        policy: Any,
        denied_exc: Any,
        agent_name: str | None,
    ) -> None:
        """監査 sink とポリシー評価の材料を保持する新規インスタンスを初期化する。

        Args:
            sink: 監査 sink（`record(agent_id, action, decision, details)` を持つ）。
            policy: AGT ポリシーオブジェクト。None なら MCP 由来ツールを評価しない。
            denied_exc: ポリシー違反時に送出する例外クラス（AGT `PolicyViolationError`）。
            agent_name: `tool:` レコードの `agent_id` に使うエージェント名（`spec.name`）。
        """
        super().__init__()
        self._sink = sink
        self._policy = policy
        self._denied_exc = denied_exc
        self._agent_name = agent_name
        # 統治ガードレールで評価済みの `ToolContext` -> ガードレールが評価に使ったツール名
        # （キーは同一性で判定・呼び出し終了後は GC で消える）。
        self._evaluated: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()

    def __deepcopy__(self, memo: dict[int, Any]) -> _AuditAgentHooks:
        """自身を返す（深いコピーで sink を複製しない）。

        SDK のエージェント同一性シグネチャ（`agents/_run_state_agent_identity.py`）は hooks を
        `_normalize_capability_identity_value` で正規化し deepcopy しない。ただし `model` 等の
        フィールドが dataclass で、そこから Agent へ参照が届く場合は `dataclasses.asdict` が
        Agent ごと深くコピーし、その過程で本フックも deepcopy される（同ファイルの invoker 側の
        注記と同じ経路）。利用者が `copy.deepcopy(agent)` した場合も同じく到達する。AuditLog の
        ロックは複製できず、監査の記録先も複製してはならないため、self を返す。以前の形（関数内
        クラスで状態をクロージャに持つ）はコピーが別インスタンスになり同じ sink・ポリシーを共有
        していたが、いまはコピーが同一インスタンスそのものになる（`_evaluated` の印も共有する）。
        ADR-0045 の invoker の統治済みの印（deepcopy で外れる）とは対象が別で、本メソッドは
        その挙動に影響しない。

        Args:
            memo: `copy.deepcopy` のメモ（使わない）。

        Returns:
            self。
        """
        return self

    def _govern_call(self, tool_name: str, args: Any) -> None:
        """MCP 由来ツールの呼び出し 1 件をポリシーで評価し、判定を `tool:` として記録する。

        `on_tool_start` と統治ガードレールが共有する評価手順。引数が str でなければ名前照合へ
        縮退せず deny する（fail-closed）。

        Args:
            tool_name: 評価・記録に使うツールの公開名。
            args: ツール引数（生 JSON 文字列であるべき値）。

        Raises:
            denied_exc: ポリシー違反、または引数が取得できず評価不能な場合。
        """
        if not isinstance(args, str):
            # 引数が取れないときは名前照合へ縮退せず deny する（fail-closed）。
            _deny_tool_call(
                sink=self._sink,
                agent_name=self._agent_name,
                tool_name=tool_name,
                reason=_ARGUMENTS_UNAVAILABLE_REASON,
                arguments=None,
                denied_exc=self._denied_exc,
            )
        reason = _evaluate_tool(self._policy, tool_name, args)
        if reason is not None:
            _deny_tool_call(
                sink=self._sink,
                agent_name=self._agent_name,
                tool_name=tool_name,
                reason=reason,
                arguments=args,
                denied_exc=self._denied_exc,
            )
        self._sink.record(
            agent_id=self._agent_name,
            action=f"tool:{tool_name}",
            decision="allow",
            details={"arguments": args},
        )

    async def on_start(self, context: Any, agent: Any) -> None:
        self._sink.record(agent_id=agent.name, action="agent_start", decision="allow")

    async def on_end(self, context: Any, agent: Any, output: Any) -> None:
        self._sink.record(agent_id=agent.name, action="agent_end", decision="allow")

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        name = getattr(tool, "name", "")
        self._sink.record(agent_id=agent.name, action=f"tool_start:{name}", decision="allow")
        if self._policy is None or not isinstance(tool, FunctionTool):
            return
        origin = get_function_tool_origin(tool)
        # MCP 由来のみ評価する（positive 判定）。FUNCTION は build 時の govern ラップ、
        # AGENT_AS_TOOL は対象外のため、ここで評価すると二重評価・意味変更になる
        # （オプトイン時の AGENT_AS_TOOL は第 3 段 post-process の実行本体ラップで評価済み）。
        # 比較は `is` でなく `!=` を使う: `ToolOriginType` は `str` 派生 Enum で
        # `ToolOrigin` は型検証を持たない frozen dataclass のため、生 str の
        # `ToolOrigin(type="mcp")` が渡り得る（公開型なので第三者ラッパ・シリアライズ経路で
        # 成立する）。`is` だと同値でも不一致になり、統治が無警告でスキップされる。
        if origin is None or origin.type != ToolOriginType.MCP:
            return
        # 統治ガードレールが本フックで同じ名前を評価済みの呼び出しは再評価しない（`tool:` を
        # 1 件に保つ）。印はガードレールが評価したときにだけ付き、評価した名前と本フックが評価に
        # 使う名前が一致するときだけ省く。ガードレールが辿れなかった呼び出し・名前が食い違う
        # 呼び出しはここで必ず評価される（安全網。倒れる先は二重評価の側）。
        # `in` を先に使う: 弱参照を作れない context では `get` が TypeError を送出するが、
        # `in` は False を返し評価へ進む（WeakSet だった頃と同じ挙動）。
        if context in self._evaluated and self._evaluated[context] == name:
            return
        self._govern_call(name, getattr(context, "tool_arguments", None))

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        self._sink.record(
            agent_id=agent.name,
            action=f"tool_end:{getattr(tool, 'name', '')}",
            decision="allow",
        )

    async def on_handoff(self, context: Any, agent: Any, source: Any) -> None:
        self._sink.record(
            agent_id=getattr(source, "name", ""),
            action=f"handoff:{getattr(agent, 'name', '')}",
            decision="allow",
        )


def _make_audit_hooks(
    sink: Any,
    inner: Any,
    *,
    policy: Any = None,
    denied_exc: Any = None,
    agent_name: str | None = None,
) -> AgentHooksBase[Any, Any]:
    """ライフサイクル事象を監査 sink へ記録し、MCP 由来ツールを評価するフックを合成して返す。

    監査記録（と `policy` 指定時の MCP 由来ツール評価）を行う `AgentHooks` を作り、
    `chain_agent_hooks` で既存フックと宣言順 `(監査, 既存)` に合成する（上書きでなく合成）。
    これにより各ライフサイクルメソッドは「監査記録 → 既存フックの同名メソッドへ委譲」の順に
    呼ばれる。`inner`（既存 `spec.hooks`）が None のときは合成ラッパを被せず監査フック自身を
    返す。`on_llm_start` / `on_llm_end` は監査対象外で、監査フック側は基底の no-op のまま既存
    フックの同名メソッドだけが呼ばれる。

    `policy` を渡した場合、`on_tool_start` は MCP 由来ツール（`ToolOriginType.MCP`）のみを
    `_evaluate_tool` で評価する（build 時にラップ対象が存在しない run 時注入ツールの統治）。
    許可なら "allow" を、違反なら "deny" を記録して `denied_exc` を送出する（送出により合成
    チェーンの後段＝利用者フックへは到達しない）。統治ガードレール（`mcp_governance_guardrail`）
    が本フックで同じツール名を評価済みの呼び出しは再評価しない。`policy` が None のときは評価
    せず従来どおり監査記録のみを行う。

    Args:
        sink: 監査 sink（`record(agent_id, action, decision, details)` を持つ）。
        inner: 既存の `spec.hooks`（None 可・部分実装可）。
        policy: AGT ポリシーオブジェクト（`check_tool` / `check_content` を持つ）。None なら
            MCP 由来ツールの評価を行わない（監査記録のみ）。**指定する場合は `denied_exc` /
            `agent_name` も同時に渡す**（3 つで 1 組。`policy` のみ渡すと違反検出時に
            `raise None(...)` となり `TypeError` へ化ける。呼び出し元は `govern_spec` /
            `govern_agent` で `_require_agt()` の戻りから必ず 3 つ揃うため、防御コードは置かない）。
        denied_exc: ポリシー違反時に送出する例外クラス（AGT `PolicyViolationError`）。
        agent_name: `tool:` レコードの `agent_id` に使うエージェント名（`spec.name`）。

    Returns:
        監査記録と既存フックへの委譲を行う合成済み `AgentHooksBase` インスタンス。

    Raises:
        denied_exc: 返されたフックの `on_tool_start` が、MCP 由来ツールでポリシー違反を検出した
            場合、または引数（`context.tool_arguments`）が取得できず評価不能な場合（fail-closed）
            に送出する（`policy` 指定時のみ）。送出する例外は `_deny_tool_call` と同じ
            `details={"tool_name": ..., "reason": ...}` を持つ（ツール引数は載せない）。
    """
    # `chain_agent_hooks` は関数内遅延 import に留める（トップレベル禁止）。
    # `import oai_agentspec` -> `_adapters/__init__.py` -> `governance` の連鎖で本モジュールは
    # 常時ロードされるため、トップレベル import にすると `_adapters.hooks` も常時ロードされ、
    # `tests/runtime/hooks/test_init_pep562_l1.py` の「窓口 import だけでは `_adapters.hooks` が
    # 載らない」probe（PEP 562 遅延窓口の契約）が赤になる。
    from .hooks import chain_agent_hooks

    audit = _AuditAgentHooks(sink=sink, policy=policy, denied_exc=denied_exc, agent_name=agent_name)
    return chain_agent_hooks(audit, inner)


def _collect_audit_hooks(hooks: Any) -> list[_AuditAgentHooks]:
    """`hooks` を根に、ポリシーを持つ lib の監査フックを宣言順に集める。

    辿る規則は 3 つ（ADR-0048）: (1) ポリシーを持つ `_AuditAgentHooks` は評価対象に加える
    （中は辿らない）、(2) `_ChainedAgentHooks` は `_hooks` を宣言順に同じ規則で辿る、
    (3) それ以外（None・利用者フック・duck-typed のラッパ・ポリシーを持たない監査フック）は
    そこで止まる。辿るのは同じパッケージの lib 型だけで、SDK の型には依存しない。

    Args:
        hooks: 辿る根（通常は `data.agent.hooks`）。

    Returns:
        評価対象の監査フック（宣言順）。対象が無ければ空リスト。
    """
    # `_adapters.hooks` は関数内遅延 import に留める（`_make_audit_hooks` の注記と同じ理由）。
    from .hooks import _ChainedAgentHooks

    if isinstance(hooks, _AuditAgentHooks):
        return [hooks] if hooks._policy is not None else []
    if isinstance(hooks, _ChainedAgentHooks):
        return [target for child in hooks._hooks for target in _collect_audit_hooks(child)]
    return []


async def _mcp_governance_guardrail_function(data: Any) -> ToolGuardrailFunctionOutput:
    """統治ガードレールの本体。`data.agent.hooks` から辿った各監査フックで呼び出しを評価する。

    評価は宣言順に行い、最初の deny で `tool:` deny を記録して拒否例外を送出する（後続は評価
    しない。SDK は `UserError` で包み `__cause__` に原例外を載せる）。allow は `tool:` allow
    （`details.arguments`）を記録し、その監査フックの評価済みの印として `data.context` に
    評価に使ったツール名（`context.tool_name`）を記録する（`on_tool_start` は自身が評価に使う
    名前と一致するときだけ再評価を省く）。対象が無ければ記録せずに allow する。
    `reject_content` / `raise_exception` は使わない。

    Args:
        data: SDK の `ToolInputGuardrailData`（`context.tool_name` / `context.tool_arguments` /
            `agent.hooks` を参照する）。

    Returns:
        常に `ToolGuardrailFunctionOutput.allow()`（deny は送出で表す）。

    Raises:
        PolicyViolationError: いずれかの監査フックのポリシーが deny した場合、または引数が
            str として取得できず評価不能な場合（fail-closed）。
    """
    context = data.context
    tool_name = context.tool_name
    args = getattr(context, "tool_arguments", None)
    for target in _collect_audit_hooks(getattr(data.agent, "hooks", None)):
        target._govern_call(tool_name, args)
        target._evaluated[context] = tool_name
    return ToolGuardrailFunctionOutput.allow()


def mcp_governance_guardrail() -> ToolInputGuardrail[Any]:
    """MCPServer の `tool_input_guardrails` の先頭に置く統治ガードレールを生成して返す。

    評価は呼び出したエージェント自身に装着された lib の監査フック（ポリシー・sink・
    `spec.name`）で行うため、builder に結び付かず、共有サーバでも per-agent で評価される。
    詳細は `govern_spec` の docstring（MCP 節）を参照する。

    Returns:
        name が `"mcp_governance_guardrail"` の SDK `ToolInputGuardrail`。
    """
    return ToolInputGuardrail(
        guardrail_function=_mcp_governance_guardrail_function,
        name=_MCP_GOVERNANCE_GUARDRAIL_NAME,
    )


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
    run 時解決の MCP 由来ツールの引数も同形で全文記録される（本経路は従来 `tool_start:` の記録のみ
    で `details` を持たなかったため、記録される情報の範囲が広がっている）。

    既知の境界（govern 対象外）: 既定（registry の `post_processor` 引数へ
    `GovernedAgentBuilder.post_processor(...)` を渡さない場合）では、`sub_agents` の as_tool は
    registry が build 後に注入するため per-call の allow/deny 評価・監査レコードを持たない
    （監査フックの tool_start / tool_end 記録のみ・サブエージェント自身の内部 `FunctionTool` は
    別途 build されていれば govern 済み）。`register_factory` 経路は builder を通らないため
    govern 対象外。オプトイン時は `govern_ungoverned_tools`（`sub_agents` の as_tool）/
    `govern_agent`（factory Agent）が registry 第 3 段 post-process で統治する
    （`post_processor(sub_agent_tools=True)` / `post_processor(factory_agents=True)` を registry の
    `post_processor` 引数へ渡した場合）。SDK の HITL 承認（`needs_approval`）はツール実行前の
    承認フローとして govern ラップ（実行本体）より**先に**走るため、ポリシーが拒否する呼び出し
    でも承認要求は先に発生し得る（承認後に deny される。
    `needs_approval` の宣言メタは不変に維持する方針のため、承認前に弾きたい場合はポリシー対象と
    承認対象のツールを設計で分ける）。

    MCP 由来ツール（`ToolOriginType.MCP`）は run 時に SDK が解決するため build 時のラップ対象が
    存在せず、監査フックの `AgentHooks.on_tool_start` で評価する。MCPServer の
    `tool_input_guardrails` の先頭に統治ガードレール（`mcp_governance_guardrail()`）を付けた
    場合は、入力ガードレールの位置で評価し（`tool:` 行が `tool_start:` より先に記録される）、
    評価済みの呼び出しは `on_tool_start` で再評価しない。付けていない・付けたが監査フックへ辿れ
    ない呼び出しは従来どおり `on_tool_start` で評価する（安全網。ADR-0048）。この経路の境界:
    (1) 統治ガードレールを付けない場合、tool 入力ガードレール（内容検査）が `reject_content` した
    呼び出しは `on_tool_start` へ到達しないため評価も監査も発生しない。統治ガードレールを先頭に
    置けば統治が内容検査より前に評価・記録し、deny した呼び出しでは内容検査の検知器を呼ばない。
    統治 allow の後に内容検査が reject した呼び出しは `tool:` allow だけが残り `tool_start:` /
    `tool_end:` は残らない（`tool:` は判定の記録であり実行の記録ではない）、(2) HITL 承認
    （`needs_approval`）は `on_tool_start` と入力ガードレールより前に走るため MCP 経路でも
    「承認後に deny」になり得る。統治ガードレールを付け、`RunConfig.tool_execution` の
    `pre_approval_tool_input_guardrails` を真にすると承認要求の前に deny できる（着地は同じ
    `UserError`・`__cause__` が `PolicyViolationError`）。この場合 allow は承認の前後で 2 回評価
    され `tool:` allow が 2 行残り、承認が却下されると実行されない呼び出しの allow 行だけが残る、
    (3) deny は raise で合成チェーンを中断するため利用者の `spec.hooks.on_tool_start` へ
    **到達しない**（`spec.tools` の deny では実行本体のラップで弾くため到達する非対称）。統治
    ガードレールを付けた経路では deny が `on_tool_start` より前に送出されるため、
    `RunHooks.on_tool_start` も開始されない。付けない経路では `RunHooks.on_tool_start` は SDK が
    並行実行するため deny 時も開始済みになり得る（開始後は最初の await で取り消されうる）、
    (4) `AGENT_AS_TOOL` origin（`sub_agents`
    の as_tool）は対象外
    （機構上は同じフックで評価しうるが、既存 `allowed_tools` 宣言の意味を変えるため評価しない。
    オプトイン時の as_tool はフックでなく実行本体のラップで評価する）、
    (5) `tool:` 行の `agent_id` は宣言時の `spec.name`（build 時捕獲）で、`tool_start:` 行の
    `agent.name`（runtime agent）とは取得元が違うため `Agent.clone(name=...)` すると食い違う、
    (6) `RealtimeAgentSpec` の `mcp_servers` は別 registry / 別 builder Protocol 経路のため govern
    対象外、(7) 照合対象は SDK が解決した公開ツール名であり
    `mcp_config["include_server_in_tool_names"]` を真にすると `mcp_{サーバ名}__{ツール名}` を
    基本形とする名前になるため `allowed_tools` の宣言も追随が必要。サーバ名部分 / ツール名部分は
    それぞれ ASCII 英数字 / `_` / `-` 以外の文字が `_` へ置換され、前後の `_-` が strip される
    （strip 後に空になると `server` / `tool` へフォールバックする。置換が 0 文字でも `--` / `__` /
    空文字は空になるため該当する）。基本形が長さ上限を超える場合は切り詰めてハッシュを付け、
    `spec.tools` / handoff / as_tool のツール名（SDK が予約名として渡す集合）や同一解決バッチ
    （同一 agent の全 MCP サーバ）内の他ツールと衝突する場合は上限以内でもハッシュが付く。
    基本形をそのまま宣言すると全不一致＝当該ツールが常時 deny になりうるため、実際の公開名を
    確認して宣言する、
    (8) hosted MCP（Responses API のサーバ側 MCP・`HostedMCPTool`）はモデルプロバイダ側で実行され
    `FunctionTool` でもないため `on_tool_start` が発火せず、**評価も監査も一切発生しない**
    （本経路が統治するのは client-side MCP = `spec.mcp_servers` 経由のツールのみ）。統治対象は
    MCP の**ツール呼び出し**に限られ、`list_prompts` / `get_prompt` / resources 経由でサーバから
    取得した文面を利用者が `instructions` 等へ流し込む使い方は評価も監査も受けない、
    (9) allowlist は名前照合であり、MCP ツールの実体はターンごとに再解決されるため同名のまま
    schema / 意味だけ差し替える変更は検知しない（サーバ単位で名前空間を分ける
    `include_server_in_tool_names` の併用が有効）、(10) deny は `UserError` として run を終了させ、
    モデルへエラー文字列を返して会話を継続する degradation は行わない（MCP ツール自身の実行時
    例外が `mcp_config["failure_error_function"]` でモデルへ返るのとは挙動が違う）。deny は
    per-call であり**ターン単位のロールバックではない**: 同一ターンに複数のツール呼び出しがある
    場合、SDK は各呼び出しを並行タスクで起動するため、deny 発生時点で兄弟呼び出しが既に実行済み /
    実行中ならその副作用は残る（「deny で run が終わった = 何も起きていない」とは読めない）、
    (11) build 後に `agent.tools` へ直接注入したツールは build 時ラップを受けず、FUNCTION origin の
    ままなら本フックでも評価されない（positive 判定の帰結）。利用者が MCP origin の
    `FunctionTool` を自前で `spec.tools` に置いた場合は build ラップと本フックの二重評価になり
    allow 時に `tool:` レコードが 2 件残る、
    (12) build 後に `Agent.hooks` を差し替える（`clone(hooks=...)` 含む・いずれも SDK の公開 API）と
    **MCP 経路の強制と監査がともに失われる**（例外も警告も出ない）。`spec.tools` 経路は実行本体の
    ラップが tool オブジェクト自身に焼き込まれるため強制も per-call の `tool:` レコードも残り、
    この点で 2 経路は非対称である。差し替えで両経路とも失われるのはフック由来のライフサイクル
    記録（`agent_start` / `tool_start:` / `tool_end:` / `handoff:` / `agent_end`）で、これは本
    フックの導入時から存在する性質だが、強制と per-call レコードまで失われるのは MCP 経路のみ。
    差し替えでなく合成したい場合は `spec.hooks` へ自前フックを宣言して builder に合成させる
    （本モジュールが `chain_agent_hooks` で合成するため利用者フックは失われない）。統治
    ガードレールは lib の合成型（`_ChainedAgentHooks`）と監査フックしか辿らないため、差し替え先が
    元の監査フックへ委譲する利用者のラッパであっても辿れず評価しない。その場合は委譲された
    `on_tool_start` が安全網として評価する（強制は残り、`tool:` は `tool_start:` の後に
    記録される）、
    (13) `get_function_tool_origin` が `None` を返すツールは MCP 由来であっても評価されない
    （fail-open・例外も警告も出ない）。SDK が非公開の `FunctionTool._emit_tool_origin` を False に
    したラッパ（現行 SDK では `build_litellm_json_tool_call` の合成ツール）や、第三者ラッパが同
    フィールドを落とした場合が該当する。逆に引数が `str` として取れない場合は名前照合へ縮退せず
    deny する（fail-closed）ため、正常な MCP 呼び出しでも `ToolContext.tool_arguments` の契約が
    変われば一律 deny になる、
    (14) 評価対象は**ツール名と引数のみ**で、MCP サーバが返す**結果は評価も content 照合も受けず**
    モデル文脈へ入る（`on_tool_end` は `tool_end:` を記録するだけ）。MCP は第三者プロセス / リモート
    のサーバであり、`allowed_tools` で許可したツールの戻り値が間接プロンプトインジェクションの主
    経路になる。`spec.tools` でも同じだが、サーバが自前でない MCP では影響が大きい。サーバを信頼
    境界の外に置く場合は SDK の出力ガードレールを併用する。MCP ツールの戻り値へ掛けるには
    MCPServer のコンストラクタへ `tool_output_guardrails=[...]` を渡す（SDK がサーバの全ツールへ
    付ける。`guard_tool` は run 時に解決される MCP ツールへ届かない）。出力の `reject_content` は
    置き換え後の値だけを Session に残すため redact として使える（置き換え文言に機密を入れない）。

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
        TypeError: policy オブジェクトに callable な `check_tool` / `check_content` が無い場合、
            または `spec.hooks` が run 単位フック（`RunHooksBase` インスタンス）の場合、
            または `spec.hooks` が `on_*` を 1 つも持たないオブジェクト（`*` の付け忘れで
            渡した list 等）の場合（監査フックとの合成が `chain_agent_hooks` を通るため。
            ADR-0017）。
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

    audit_hooks = _make_audit_hooks(
        sink,
        spec.hooks,
        policy=policy_obj,
        denied_exc=policy_violation_error,
        agent_name=agent_name,
    )
    return _dataclass_replace(spec, tools=new_tools, hooks=audit_hooks)


def govern_ungoverned_tools(
    agent: Any,
    *,
    policy: object,
    audit_sink: object,
    agent_name: str,
) -> None:
    """構築済み `Agent` の tools のうち印の無い `FunctionTool` だけをその場で govern ラップする。

    registry 第 3 段 post-process の spec 経路（`GovernedAgentBuilder.post_processor(
    sub_agent_tools=True)` を registry の `post_processor` 引数へ渡した場合）専用。build 時に
    `govern_spec` でラップ済みの tool は印（`_GOVERNED_WRAPPERS` への登録。判定は
    `_is_governed`）で判別して素通しし、印の無い `FunctionTool`
    （wire が注入した `sub_agents` の as_tool 等）を `_govern_tool` でラップする。`FunctionTool`
    以外は素通しする。origin では絞らない。`agent.hooks` は触らない（build 時に監査フックを
    合成済みのため。再合成するとライフサイクル記録・MCP 評価が重複する）。ラップ・評価・監査の
    境界は `govern_spec` の docstring を参照する。

    所有契約: `agent.tools` の list オブジェクトを再束縛せず、要素をその場で置換する。結線の
    途中で作られた clone のうち、tools の list を共有するもの（`Agent.clone` に `tools=` を
    渡さない既定）に限り統治が届く。`tools=` に新しい list を渡した clone や、list をコピー
    した Agent には届かない。
    `agent.tools` は builder が作った list（利用者の `spec.tools` とは別物）である前提で書き換える。

    境界: 印は「いずれかのポリシーで統治済み」を示し、この agent のポリシーで統治したかは区別
    しない。自作 inner builder が別 agent / 別 builder で統治済みの tool を足すと、その tool は
    元のポリシーだけで評価される。逆に自作 inner が独自に追加した印の無い `FunctionTool` は
    本関数で統治される。

    Args:
        agent: 統治対象の構築済み `Agent`（lib 所有・spec 経路）。
        policy: ポリシー定義（YAML ファイルパス、または AGT ポリシーオブジェクト）。
        audit_sink: 監査ログ出力先（`record(...)` を持つ任意オブジェクト）。
        agent_name: `tool:` レコードの `agent_id` に使うエージェント名（登録名）。

    Raises:
        ImportError: governance extra（agent-governance-toolkit）が未導入の場合（案内付き）。
        FileNotFoundError: policy が指す YAML パスが存在しない場合。
        yaml.YAMLError: policy が指す YAML の構文が不正な場合。
        ValueError: policy YAML がマッピングでない、または未知キーを含む場合。
        TypeError: policy オブジェクトに callable な `check_tool` / `check_content` が無い場合。
    """
    governance_policy, _, policy_violation_error = _require_agt()
    policy_obj = _load_policy(policy, governance_policy)
    tools = agent.tools
    for i, tool in enumerate(tools):
        if isinstance(tool, FunctionTool) and not _is_governed(tool):
            tools[i] = _govern_tool(
                tool,
                policy=policy_obj,
                sink=audit_sink,
                denied_exc=policy_violation_error,
                agent_name=agent_name,
            )


def govern_agent(
    agent: Any,
    *,
    policy: object,
    audit_sink: object,
    agent_name: str,
) -> Any:
    """構築済み `Agent` を clone し、全 `FunctionTool` の govern ラップと監査フックを合成して返す。

    registry 第 3 段 post-process の factory 経路（`GovernedAgentBuilder.post_processor(
    factory_agents=True)` を registry の `post_processor` 引数へ渡した場合）専用。`agent`
    （利用者所有）は一切変更せず、全 `FunctionTool` を `_govern_tool` でラップした新 list と、
    監査フックを `agent.hooks` と「監査記録 → 既存フックへ委譲」の順に合成したフック
    （`agent.hooks is None` なら監査単体）で `agent.clone(tools=..., hooks=...)` を返す。
    `FunctionTool` 以外は同一オブジェクトのまま残す。ラップ・評価・監査の境界は `govern_spec` の
    docstring を参照する。

    境界: 印（`_is_governed`）は見ない（統治済みを skip すると factory 名のポリシーが黙って
    効かなくなるため）。factory が統治済みの Agent（例: `registry.get` の戻り）を返すと、元の
    ポリシーと factory 名のポリシーの両方で評価され、`tool:` レコードとライフサイクル記録が
    2 回ずつ残る（どちらかが deny なら deny）。

    Args:
        agent: 統治対象の構築済み `Agent`（利用者所有・factory 経路）。
        policy: ポリシー定義（YAML ファイルパス、または AGT ポリシーオブジェクト）。
        audit_sink: 監査ログ出力先（`record(...)` を持つ任意オブジェクト）。
        agent_name: `tool:` レコードの `agent_id` に使うエージェント名（登録名）。

    Returns:
        tools / hooks を govern 化した `agent` の clone。

    Raises:
        ImportError: governance extra（agent-governance-toolkit）が未導入の場合（案内付き）。
        FileNotFoundError: policy が指す YAML パスが存在しない場合。
        yaml.YAMLError: policy が指す YAML の構文が不正な場合。
        ValueError: policy YAML がマッピングでない、または未知キーを含む場合。
        TypeError: policy オブジェクトに callable な `check_tool` / `check_content` が無い場合、
            または `agent.hooks` が run 単位フック（`RunHooksBase` インスタンス）の場合、
            または `agent.hooks` が `on_*` を 1 つも持たないオブジェクトの場合（監査フックとの
            合成が `chain_agent_hooks` を通るため）。
    """
    governance_policy, _, policy_violation_error = _require_agt()
    policy_obj = _load_policy(policy, governance_policy)
    tools = [
        _govern_tool(
            tool,
            policy=policy_obj,
            sink=audit_sink,
            denied_exc=policy_violation_error,
            agent_name=agent_name,
        )
        if isinstance(tool, FunctionTool)
        else tool
        for tool in agent.tools
    ]
    hooks = _make_audit_hooks(
        audit_sink,
        agent.hooks,
        policy=policy_obj,
        denied_exc=policy_violation_error,
        agent_name=agent_name,
    )
    return agent.clone(tools=tools, hooks=hooks)
