"""RealtimeAgent の宣言的定義（`RealtimeAgentSpec`）とハンドオフ設定（`RealtimeHandoffConfig`）。

`RealtimeAgentSpec` は openai-agents の `RealtimeAgent`（`agents.realtime.RealtimeAgent`）の
薄い宣言的 Wrapper である。`agents` には依存せず、`RealtimeAgent` が受け付けるフィールドのみを
構造的に保持し、それに加えてハンドオフを**エージェント名で**宣言できるグラフ連携機能を持つ。

通常ルートの `AgentSpec` と異なり、`RealtimeAgent` が受け付けない非対応フィールド
（`model` / `model_settings` / `input_guardrails` / `sub_agents` / `sub_agent_tools` /
`dynamic_handoffs` / `output_type` / `tool_use_behavior`）は**型レベルで排除**する
（フィールドとして定義しない）。build 時の第二防御（`extra` 検証）と合わせた二段防御により、
非対応機能の誤用を宣言段階で不可能にする。

`RealtimeHandoffConfig` は SDK `realtime_handoff()` の主要引数を型付きで保持する。通常ルートの
`HandoffConfig` / `DynamicHandoff` とは別物として定義し、`input_filter` を型として持たない
（`realtime_handoff()` が `input_filter` パラメータを持たないため）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RealtimeHandoffConfig:
    """Realtime ハンドオフ 1 エッジの設定（SDK `realtime_handoff()` の主要引数を型付きで保持）。

    通常ルートの `HandoffConfig` とは別物として定義する。`realtime_handoff()` は
    `input_filter` パラメータを持たないため、本設定も `input_filter` を型として持たない
    （`extra` / `options` 経由の指定も build 時に `ValueError` で弾く）。

    Attributes:
        on_handoff: `realtime_handoff(on_handoff=...)`。ハンドオフ発火時のコールバック。
            `input_type` 指定時は `(context, parsed_input)`、未指定時は `(context,)` を受ける。
        input_type: `realtime_handoff(input_type=...)`。転送時に LLM が埋める構造化入力の型。
        tool_name_override: `realtime_handoff(tool_name_override=...)`。ハンドオフ tool 名。
        tool_description_override: `realtime_handoff(tool_description_override=...)`。tool の説明。
        is_enabled: `realtime_handoff(is_enabled=...)`。動的有効化（bool または callable）。
    """

    on_handoff: Any = None
    input_type: Any = None
    tool_name_override: str | None = None
    tool_description_override: str | None = None
    is_enabled: Any = True


@dataclass
class RealtimeAgentSpec:
    """宣言的な RealtimeAgent 定義（`RealtimeAgent` の薄い Wrapper）。

    `instructions` / `prompt` / `tools` / `hooks` / `output_guardrails` /
    `handoff_description` / `mcp_servers` / `mcp_config` は `agents.realtime.RealtimeAgent`
    と同じ意味を持つ。`handoffs` はエージェント名の参照で、registry が遅延構築時に解決する
    （グラフ連携の追加機能）。その他の `RealtimeAgent` kwarg は `extra` で素通しする。

    非対応フィールド（`model` / `model_settings` / `input_guardrails` / `sub_agents` /
    `sub_agent_tools` / `dynamic_handoffs` / `output_type` / `tool_use_behavior`）は
    `RealtimeAgent` が受け付けないため、型レベルで排除する（フィールドとして定義しない）。

    Attributes:
        name: エージェント名（registry 内で一意）。
        instructions: システムプロンプト。文字列、または (context, agent) の 2 引数
            callable（`PromptStore.compose` の戻り値を渡せる）。
        prompt: `RealtimeAgent.prompt`（agents.Prompt | None のみ）。`RealtimeAgent` は
            通常 `Agent` と異なり `DynamicPromptFunction`（callable）を受け付けない
            （session が agent.prompt を直接使い解決 callback を呼ばないため）。callable を
            渡すと build 時に `ValueError` になる（第二防御）。
        tools: Agent に渡すツール（SDK の Tool）。
        hooks: エージェントフック（agents.realtime のフック型）。
        output_guardrails: 出力ガードレールのリスト（`RealtimeAgent.output_guardrails` と同型）。
        handoff_description: `RealtimeAgent.handoff_description`。ハンドオフ先としての説明。
        mcp_servers: `RealtimeAgent.mcp_servers`。接続する MCP サーバのリスト。
        mcp_config: `RealtimeAgent.mcp_config`。MCP 設定。
        handoffs: ハンドオフ先エージェント名リスト（グラフ連携）。
        handoff_options: dst 名 -> RealtimeHandoffConfig の per-edge 設定。
        extra: 上記以外の agents.realtime.RealtimeAgent kwarg 素通し用 dict。
    """

    name: str
    instructions: str | Callable[..., Any] | None = None
    prompt: Any = None
    tools: list[Any] = field(default_factory=list)
    hooks: Any = None
    handoff_description: str | None = None
    mcp_servers: list[Any] = field(default_factory=list)
    mcp_config: Any = None
    # kw_only: 既存フィールドの位置引数束縛を保つため（handoffs 等のズレ防止）。
    output_guardrails: list[Any] = field(default_factory=list, kw_only=True)
    handoffs: list[str] = field(default_factory=list)
    handoff_options: dict[str, RealtimeHandoffConfig] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
