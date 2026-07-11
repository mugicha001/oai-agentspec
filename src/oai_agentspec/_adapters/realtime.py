"""RealtimeAgent / realtime handoff 構築アダプタ（SDK 結合を `_adapters` に閉じる・NFR-1）。

`build_realtime_agent`（デフォルト RealtimeAgentBuilder の構築本体）/ `make_realtime_handoff` /
`DefaultRealtimeAgentBuilder`（`RealtimeAgentBuilder` Protocol 適合）を提供する。
`from agents.realtime import ...` の SDK 結合は本モジュール内に閉じる。

`RealtimeAgent` は `model` / `model_settings` / `output_type` / `tool_use_behavior` /
`input_guardrails` をフィールドに持たない（非対応）。`RealtimeAgentSpec` はこれらを型レベルで
排除するが、`extra` 経由の素通し指定は本アダプタが有効キー集合（`RealtimeAgent` の dataclass
fields）と照合して `ValueError` で弾く（FR-4 第二防御）。
"""

from __future__ import annotations

from dataclasses import fields as _dataclass_fields
from typing import TYPE_CHECKING, Any

from agents.realtime import RealtimeAgent, realtime_handoff

from .._validation import ensure_static_prompt, validate_extra_kwargs

if TYPE_CHECKING:
    from ..realtime.spec import RealtimeAgentSpec, RealtimeHandoffConfig

# 専用フィールド名（RealtimeAgentSpec 側で別扱いするため extra から除外する RealtimeAgent kwarg）。
_DEDICATED_REALTIME_AGENT_KWARGS = frozenset(
    {
        "name",
        "instructions",
        "prompt",
        "tools",
        "hooks",
        "handoff_description",
        "mcp_servers",
        "mcp_config",
        "handoffs",
        "output_guardrails",
    }
)

# RealtimeAgent が受け付ける有効な kwarg 名（extra の早期検証に使う）。
_REALTIME_AGENT_FIELD_NAMES = frozenset(f.name for f in _dataclass_fields(RealtimeAgent))


def build_realtime_agent(
    spec: RealtimeAgentSpec, *, handoffs: list[Any] | None = None
) -> RealtimeAgent:
    """spec から RealtimeAgent を 1 つ構築する（デフォルト RealtimeAgentBuilder の本体）。

    `instructions` / `prompt` / `tools` / `hooks` / `output_guardrails` /
    `handoff_description` / `mcp_servers` / `mcp_config` を `RealtimeAgent` にそのまま渡す。
    handoffs は既定で空（registry が後付け結線）。None の任意フィールドは kwargs に積まず
    SDK 既定に委ねる。

    Args:
        spec: 構築対象の RealtimeAgentSpec。
        handoffs: 構築時に渡す handoffs（省略時は空。registry は空で構築し後付け結線する）。

    Returns:
        agents.realtime.RealtimeAgent（handoffs は既定で空）。

    Raises:
        ValueError: extra に専用フィールド名と同名のキー、または RealtimeAgent が受け付けない
            未知のキーが含まれる場合（agent 名 + 該当キー名を含む）。
    """
    extra = dict(spec.extra)
    validate_extra_kwargs(
        spec.name,
        extra,
        dedicated=_DEDICATED_REALTIME_AGENT_KWARGS,
        field_names=_REALTIME_AGENT_FIELD_NAMES,
        agent_label="agents.realtime.RealtimeAgent",
    )
    # RealtimeAgent.prompt は Prompt | None のみで DynamicPromptFunction（callable）非対応
    # （session が agent.prompt を解決 callback 抜きで直接使うため）。register 時の前倒し検証と
    # 同一の共有ヘルパで第二防御として reject する（判定・メッセージの単一ソース化）。
    ensure_static_prompt(spec.name, spec.prompt)

    # FR-2: 未指定（None / 空）のフィールドは kwargs に積まず SDK 既定に委ねる。
    kwargs: dict[str, Any] = {
        "name": spec.name,
        "instructions": spec.instructions,
        **extra,
    }
    if spec.tools:
        kwargs["tools"] = list(spec.tools)
    if handoffs:
        kwargs["handoffs"] = list(handoffs)
    if spec.output_guardrails:
        kwargs["output_guardrails"] = list(spec.output_guardrails)
    if spec.mcp_servers:
        kwargs["mcp_servers"] = list(spec.mcp_servers)
    if spec.prompt is not None:
        kwargs["prompt"] = spec.prompt
    if spec.hooks is not None:
        kwargs["hooks"] = spec.hooks
    if spec.handoff_description is not None:
        kwargs["handoff_description"] = spec.handoff_description
    if spec.mcp_config is not None:
        kwargs["mcp_config"] = spec.mcp_config
    return RealtimeAgent(**kwargs)


def make_realtime_handoff(agent: RealtimeAgent, config: RealtimeHandoffConfig | None) -> Any:
    """SDK の realtime_handoff() を RealtimeHandoffConfig で生成する。

    型付きフィールド（on_handoff / input_type / tool_name_override /
    tool_description_override / is_enabled）を realtime_handoff() の対応引数へマップする。
    `realtime_handoff()` は `input_filter` を持たないため一切渡さない（FR-4）。config 省略
    （None）時は SDK 既定（transfer_to_<name> のツール名）で結線する。

    Args:
        agent: ハンドオフ先の RealtimeAgent インスタンス。
        config: ハンドオフ設定。None なら既定設定で結線する。

    Returns:
        agents.Handoff（RealtimeAgent 用）。
    """
    if config is None:
        return realtime_handoff(agent)
    kwargs: dict[str, Any] = {}
    if config.tool_name_override is not None:
        kwargs["tool_name_override"] = config.tool_name_override
    if config.tool_description_override is not None:
        kwargs["tool_description_override"] = config.tool_description_override
    if config.on_handoff is not None:
        kwargs["on_handoff"] = config.on_handoff
    if config.input_type is not None:
        kwargs["input_type"] = config.input_type
    if config.is_enabled is not True:
        kwargs["is_enabled"] = config.is_enabled
    return realtime_handoff(agent, **kwargs)


class DefaultRealtimeAgentBuilder:
    """`build_realtime_agent` / `make_realtime_handoff` をラップするデフォルト実装。

    `RealtimeAgentBuilder` Protocol に構造的に適合する（registry の既定 builder）。
    """

    def build(self, spec: RealtimeAgentSpec) -> RealtimeAgent:
        """spec から handoffs 空の RealtimeAgent を構築する。"""
        return build_realtime_agent(spec)

    def make_handoff(self, agent: RealtimeAgent, config: RealtimeHandoffConfig | None) -> Any:
        """構築済み target とエッジ設定からハンドオフを生成する。"""
        return make_realtime_handoff(agent, config)
