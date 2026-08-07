"""エージェントの宣言的定義（`AgentSpec`）とハンドオフ設定（`HandoffConfig`）。

`AgentSpec` は openai-agents の `Agent` の薄い宣言的 Wrapper である。基本的に `Agent`
と同じフィールドを持ち（`agents` には依存せず構造的に保持）、それに加えてハンドオフ /
サブエージェントを**エージェント名で**宣言できるグラフ連携機能を持つ。

プロンプト合成は本 spec の責務ではなく、`PromptStore.compose` が生成した値を
`instructions` に渡す形をとる（SDK の `Agent.instructions` と同じ使い心地）。

ハンドオフ設定の対称性（HandoffConfig / DynamicHandoff）について:

静的エッジ（`HandoffConfig`）と動的エッジ（`DynamicHandoff`）は `description` /
`on_handoff` / `input_type` / `input_filter` / `is_enabled` / `options` の 6 フィールドを
**同名・同型・同既定値**で共有する（AC #7「静的・動的の学習統一性」）。両 dataclass を
共通基底に統合せず別物として保つ理由:

- `HandoffConfig` は `frozen=True`、`DynamicHandoff` はミュータブル（registry が
  `_update_handoffs` で上書きするため）。Python の dataclass は frozen / unfrozen の
  混在継承を許可しない（型エラー）。
- 動的エッジは `tool_name` 必須・`candidates` / `resolver` の追加フィールドを持ち、構造が
  完全には一致しない。
- 共有 6 フィールドのパリティは `tests/test_handoff_config.py` の
  `test_handoff_config_and_dynamic_handoff_share_field_names` で機械的に保証される。

フィールド追加・型変更は両 dataclass に同時に行うこと（テストが失敗して検知する）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HandoffConfig:
    """ハンドオフ 1 エッジの設定（SDK `handoff()` の主要引数を型付きで保持）。

    専用フィールドに無い `handoff()` 引数は `options` で素通しする（`AgentSpec.extra`
    と同じ「専用フィールド + 残りは素通し」思想）。

    Attributes:
        description: `handoff(tool_description_override=...)`。ハンドオフ tool の説明。
        tool_name: `handoff(tool_name_override=...)`。ハンドオフ tool 名。
        on_handoff: `handoff(on_handoff=...)`。ハンドオフ発火時のコールバック。
        input_type: `handoff(input_type=...)`。転送時に LLM が埋める構造化入力の型。
        input_filter: `handoff(input_filter=...)`。次エージェントへ渡す履歴の変換。
        is_enabled: `handoff(is_enabled=...)`。動的有効化（bool or callable）。
        options: 上記以外の `handoff()` kwarg 素通し用 dict。
    """

    description: str | None = None
    tool_name: str | None = None
    on_handoff: Any = None
    input_type: Any = None
    input_filter: Any = None
    is_enabled: Any = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamicHandoff:
    """動的ハンドオフ宣言（`on_invoke_handoff` で候補から転送先を実行時に選ぶ）。

    固定 1 ターゲットの通常ハンドオフと異なり、resolver が候補名から転送先を返す。
    registry が候補名を解決する `on_invoke_handoff` を生成して結線する。

    `on_handoff` / `input_type` / `input_filter` / `is_enabled` / `options` は静的エッジの
    `HandoffConfig` と同じ意味を持ち、利用側に「静的・動的」の二重学習を強いない（型付き
    フィールド優先・`options` は SDK 固有の追加 kwarg 専用の裏口で、型付きフィールドと
    同義の予約キーは `_adapters` 側で `ValueError` として弾く）。

    Attributes:
        tool_name: ハンドオフ tool 名（必須。動的なので転送先名からは導出しない）。
        candidates: 転送先候補のエージェント名（validate / 依存解決の対象）。
        resolver: `(context, input_json) -> 転送先名`。戻り名は candidates 内に限る。
        description: ハンドオフ tool の説明（生 `Handoff.tool_description` へ反映）。
        on_handoff: ハンドオフ発火時のコールバック（転送先決定後・Agent 返却前に発火）。
            `input_type` 指定時は `(context, parsed_input)`、未指定時は `(context,)` を受ける。
        input_type: 転送時に LLM が埋める構造化入力の型（pydantic モデル等）。指定時は
            `_adapters` が JSON Schema を生成し `Handoff.input_json_schema` を差し替える。
        input_filter: 次エージェントへ渡す履歴の変換（`Handoff.input_filter` へ素通し）。
        is_enabled: 動的有効化（bool または `(context, agent) -> bool` callable）。
        options: 上記以外の生 `Handoff` kwarg 素通し用 dict。型付きフィールドと同義の
            予約キー（`tool_name` / `tool_description` / `input_json_schema` /
            `on_invoke_handoff` / `agent_name` / `input_filter` / `is_enabled`）は禁止。
    """

    tool_name: str
    candidates: list[str]
    resolver: Callable[..., Any]
    description: str | None = None
    on_handoff: Any = None
    input_type: Any = None
    input_filter: Any = None
    is_enabled: Any = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSpec:
    """宣言的なエージェント定義（`Agent` の薄い Wrapper）。

    `instructions` / `prompt` / `tools` / `model` / `model_settings` / `hooks` /
    `input_guardrails` / `output_guardrails` / `mcp_servers` / `mcp_config` は `agents.Agent` と
    同じ意味を持つ。`handoffs` /
    `sub_agents` はエージェント名の参照で、registry が遅延構築時に解決する（グラフ連携の追加機能）。
    その他の `Agent` kwarg は `extra` で素通しする。

    Attributes:
        name: エージェント名（registry 内で一意）。
        instructions: システムプロンプト。文字列、または (context, agent) の 2 引数
            callable（`PromptStore.compose` の戻り値を渡せる）。
        instructions_append: 静的 `instructions` の末尾へ run ごとに評価される断片を宣言順に
            連結する追記関数のリスト。各要素は (context, agent) の 2 引数 callable
            （async 可）で `str` を返す。`instructions` が callable の場合は併用できない。
            容器は `list` / `tuple` を受理し、`set` や generator は宣言時に拒否する
            （順序が保証されない・検証で消費されて追記が無言に消えるのを防ぐため）。
        prompt: `Agent.prompt`（agents.Prompt / DynamicPromptFunction。Responses API 用）。
        tools: Agent に渡すツール（SDK の Tool）。
        model: モデル指定（str | agents.Model | None）。
        model_settings: モデル設定（agents.ModelSettings）。
        hooks: エージェントフック（agents.AgentHooks）。
        input_guardrails: 入力ガードレールのリスト（`agents.Agent.input_guardrails` と同型・
            `runtime.guardrails` の factory が返す guardrail オブジェクト）。
        output_guardrails: 出力ガードレールのリスト（`agents.Agent.output_guardrails` と同型・
            `runtime.guardrails` の factory が返す guardrail オブジェクト）。
        guardrails: ガードレールの**登録名**リスト（実体は registry が build 時に
            `GuardrailProvider` で解決し、宣言境界に応じて `input_guardrails` /
            `output_guardrails` へ連結する。連結順序は「専用フィールド -> 名前参照」）。
        mcp_servers: `Agent.mcp_servers`。接続する MCP サーバのリスト。接続 / 切断
            （`connect()` / `cleanup()`）は利用者責務であり lib は宣言を素通しするだけで
            lifecycle を持たない（`agents.Agent.mcp_servers` と同じ。
            `agents.mcp.MCPServerManager` の利用を検討する）。MCP サーバのツール定義・ツール
            出力は信頼境界の外側から model context へ入るため、必要なら
            `oai_agentspec.runtime.guardrails` を併用する。
        mcp_config: `Agent.mcp_config`。MCP 設定（`convert_schemas_to_strict` /
            `include_server_in_tool_names` / `failure_error_function`）。未指定（None）の場合は
            build 時に kwargs へ積まず SDK の既定（空 dict）に委ねる。SDK の `MCPConfig` に無い
            キーは検証されず SDK 側で無視される（綴り誤りは silent に効かない）。
            `failure_error_function` の戻り値は LLM へ渡るため、例外原文（接続 URL / トークンを
            含みうる）をそのまま返さない。build 時は dict をコピーせず参照を渡すため、宣言後に
            渡した dict を mutate すると構築済み `Agent` へ伝播する（registry 経由の `freeze()`
            は `_copy_spec` が dict を複製するため遮断される）。
        handoffs: ハンドオフ先エージェント名リスト（グラフ連携）。
        handoff_options: dst 名 -> HandoffConfig の per-edge 設定。
        sub_agents: as_tool 配線するサブエージェント名リスト（グラフ連携）。
        sub_agent_tools: サブ名 -> (tool_name, tool_description) の as_tool 上書き。
        dynamic_handoffs: 動的ハンドオフ宣言（on_invoke による候補選択）。
        extra: 上記以外の agents.Agent kwarg 素通し用 dict。
    """

    name: str
    instructions: str | Callable[..., Any] | None = None
    prompt: Any = None
    tools: list[Any] = field(default_factory=list)
    model: Any = None
    model_settings: Any = None
    hooks: Any = None
    # kw_only: 既存フィールドの位置引数束縛を保つため（handoffs 等のズレ防止）。
    input_guardrails: list[Any] = field(default_factory=list, kw_only=True)
    output_guardrails: list[Any] = field(default_factory=list, kw_only=True)
    guardrails: list[str] = field(default_factory=list, kw_only=True)
    instructions_append: list[Callable[..., Any]] = field(default_factory=list, kw_only=True)
    mcp_servers: list[Any] = field(default_factory=list, kw_only=True)
    mcp_config: dict[str, Any] | None = field(default=None, kw_only=True)
    handoffs: list[str] = field(default_factory=list)
    handoff_options: dict[str, HandoffConfig] = field(default_factory=dict)
    sub_agents: list[str] = field(default_factory=list)
    sub_agent_tools: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)
    dynamic_handoffs: list[DynamicHandoff] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxAgentSpec(AgentSpec):
    """サンドボックスエージェントの宣言的定義（`SandboxAgent` の薄い Wrapper）。

    `agents.sandbox.SandboxAgent`（`agents.Agent` のサブクラス）向けの宣言 dataclass。
    `AgentSpec` の全フィールドを継承し、サンドボックス固有の 4 フィールドを追加する。
    4 フィールドはいずれも未指定（None）の場合、build 時に kwargs へ積まれず SDK の
    既定値に委ねられる（SDK の `Capabilities.default()` 等をここで再現・ハードコード
    しない）。`capabilities` 未指定時の SDK 既定はシェル実行を含む機能群を有効化しうる
    ため、最小権限にしたい場合は明示指定すること。`base_instructions` の callable arity
    検証は build 時（`_adapters`）に行う。

    デフォルト builder（`_adapters.build_agent`）が `SandboxAgent` へ渡すのは本クラスで
    宣言済みのフィールドのみ。本クラスをさらに継承して独自フィールドを追加しても
    デフォルト builder は関知しない（カスタム `AgentBuilder` の注入が必要）。

    Attributes:
        default_manifest: `SandboxAgent.default_manifest`（SDK `Manifest` 相当の不透明型）。
        capabilities: `SandboxAgent.capabilities`（SDK `Sequence[Capability]` 相当の不透明型）。
        run_as: `SandboxAgent.run_as`（SDK `User | str` 相当の不透明型）。
        base_instructions: サンドボックス用ベースプロンプト。文字列、または
            (context, agent) の 2 引数 callable。
    """

    default_manifest: Any = field(default=None, kw_only=True)
    capabilities: Any = field(default=None, kw_only=True)
    run_as: Any = field(default=None, kw_only=True)
    base_instructions: str | Callable[..., Any] | None = field(default=None, kw_only=True)
