"""L2: _adapters.realtime の SDK 結合検証（build_realtime_agent / make_realtime_handoff）。

`build_realtime_agent` のマッピング成功系と extra reject 系、`make_realtime_handoff` の
`realtime_handoff` 委譲、および `RealtimeRunner` + `FakeRealtimeModel` による handoff 実委譲を
検証する。`_adapters.realtime` は未実装のため本モジュールの import は collection error（RED）想定。
"""

from __future__ import annotations

import pytest
from agents.realtime import RealtimeAgent
from agents.realtime.model_events import RealtimeModelToolCallEvent
from agents.realtime.runner import RealtimeRunner
from pydantic import BaseModel

from oai_agentspec._adapters.realtime import (
    DefaultRealtimeAgentBuilder,
    build_realtime_agent,
    make_realtime_handoff,
)
from oai_agentspec.realtime.protocols import RealtimeAgentBuilder
from oai_agentspec.realtime.registry import RealtimeAgentRegistry
from oai_agentspec.realtime.spec import RealtimeAgentSpec, RealtimeHandoffConfig

from _helpers.fake_realtime_model import FakeRealtimeModel

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# build_realtime_agent: マッピング成功系
# ---------------------------------------------------------------------------
def test_build_maps_supported_fields() -> None:
    """name / instructions / tools / output_guardrails / handoff_description の反映と空 handoffs 検証。"""  # noqa: E501
    tool = object()
    guardrail = object()
    spec = RealtimeAgentSpec(
        name="triage",
        instructions="route",
        tools=[tool],
        output_guardrails=[guardrail],
        handoff_description="triage desk",
    )
    agent = build_realtime_agent(spec)
    assert isinstance(agent, RealtimeAgent)
    assert agent.name == "triage"
    assert agent.instructions == "route"
    assert agent.tools == [tool]
    assert agent.output_guardrails == [guardrail]
    assert agent.handoff_description == "triage desk"
    # handoffs は registry の後付け結線に委ねるため build 時は空。
    assert agent.handoffs == []


def test_build_omits_none_optional_fields() -> None:
    """None の任意フィールド（prompt / hooks）は SDK 既定（None）のまま積まれない。"""
    agent = build_realtime_agent(RealtimeAgentSpec(name="a", instructions="x"))
    assert agent.prompt is None
    assert agent.hooks is None


def test_build_passes_prompt_when_set() -> None:
    """prompt が指定された場合は構築物へ渡る。"""
    prompt = {"id": "p1"}
    agent = build_realtime_agent(RealtimeAgentSpec(name="a", instructions="x", prompt=prompt))
    assert agent.prompt == prompt


# ---------------------------------------------------------------------------
# build_realtime_agent: extra reject 系（FR-4 第二防御）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["model", "model_settings", "output_type", "tool_use_behavior"])
def test_build_rejects_unsupported_kwarg_in_extra(key: str) -> None:
    """RealtimeAgent が受け付けない非対応キーを extra に積むと ValueError（agent 名 + キー名）。"""
    spec = RealtimeAgentSpec(name="voice", instructions="x", extra={key: object()})
    with pytest.raises(ValueError, match=r"voice") as excinfo:
        build_realtime_agent(spec)
    assert key in str(excinfo.value)


def test_build_rejects_unknown_key_in_extra() -> None:
    """RealtimeAgent が知らない未知キーを extra に積むと ValueError（キー名を含む）。"""
    spec = RealtimeAgentSpec(name="voice", instructions="x", extra={"nonexistent_kw": 1})
    with pytest.raises(ValueError, match=r"voice") as excinfo:
        build_realtime_agent(spec)
    assert "nonexistent_kw" in str(excinfo.value)


def test_build_rejects_dedicated_field_collision_in_extra() -> None:
    """extra に専用フィールド同名キー（name）を積むと ValueError で弾く。"""
    spec = RealtimeAgentSpec(name="voice", instructions="x", extra={"name": "dup"})
    with pytest.raises(ValueError, match=r"voice"):
        build_realtime_agent(spec)


def test_build_rejects_callable_prompt() -> None:
    """prompt に callable（DynamicPromptFunction）を渡すと agent 名 + prompt 入り ValueError。

    RealtimeAgent.prompt は Prompt | None のみ対応で callable 非対応のため、第二防御で弾く。
    """
    spec = RealtimeAgentSpec(name="voice", instructions="x", prompt=lambda ctx, agent: "p")
    with pytest.raises(ValueError, match=r"voice") as excinfo:
        build_realtime_agent(spec)
    assert "prompt" in str(excinfo.value)


# ---------------------------------------------------------------------------
# make_realtime_handoff: realtime_handoff への委譲
# ---------------------------------------------------------------------------
def test_make_handoff_default_config_derives_tool_name() -> None:
    """config 省略（None）時は SDK 既定（transfer_to_<name>）のツール名で結線する。"""
    target = build_realtime_agent(RealtimeAgentSpec(name="billing", instructions="b"))
    handoff = make_realtime_handoff(target, None)
    assert handoff.tool_name == "transfer_to_billing"
    assert handoff.agent_name == "billing"


def test_make_handoff_applies_overrides() -> None:
    """tool_name_override / tool_description_override が realtime_handoff へ渡る。"""
    target = build_realtime_agent(RealtimeAgentSpec(name="billing", instructions="b"))
    config = RealtimeHandoffConfig(
        tool_name_override="go_billing",
        tool_description_override="route to billing",
    )
    handoff = make_realtime_handoff(target, config)
    assert handoff.tool_name == "go_billing"
    assert handoff.tool_description == "route to billing"


def test_make_handoff_never_sets_input_filter() -> None:
    """realtime_handoff は input_filter 非対応のため、生成 Handoff の input_filter は常に None。"""
    target = build_realtime_agent(RealtimeAgentSpec(name="billing", instructions="b"))
    handoff = make_realtime_handoff(target, RealtimeHandoffConfig())
    assert handoff.input_filter is None


def test_make_handoff_with_input_type_and_callback() -> None:
    """input_type 指定時は 2 引数 on_handoff を伴って Handoff が生成される。"""
    target = build_realtime_agent(RealtimeAgentSpec(name="billing", instructions="b"))

    def _on_handoff(context: object, parsed: object) -> None:
        return None

    class _Input(BaseModel):
        reason: str = ""

    config = RealtimeHandoffConfig(on_handoff=_on_handoff, input_type=_Input)
    handoff = make_realtime_handoff(target, config)
    assert handoff.agent_name == "billing"


# ---------------------------------------------------------------------------
# RealtimeRunner + FakeRealtimeModel による handoff 実委譲
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_handoff_delegation_switches_current_agent() -> None:
    """handoff ツール呼び出しイベントを流すと現在エージェントが委譲先へ切り替わる。"""
    billing = build_realtime_agent(RealtimeAgentSpec(name="billing", instructions="b"))
    triage = build_realtime_agent(RealtimeAgentSpec(name="triage", instructions="t"))
    triage.handoffs.append(make_realtime_handoff(billing, RealtimeHandoffConfig()))

    model = FakeRealtimeModel()
    # async_tool_calls=False で tool call を同期処理し、委譲完了を決定的に観測する。
    runner = RealtimeRunner(triage, model=model, config={"async_tool_calls": False})
    session = await runner.run()
    async with session:
        assert model.connected
        assert session._current_agent is triage  # noqa: SLF001 - 委譲前の現在エージェント検証
        await model.emit(
            RealtimeModelToolCallEvent(name="transfer_to_billing", call_id="c1", arguments="{}")
        )
        # handoff により現在エージェントが billing へ切り替わる。
        assert session._current_agent is billing  # noqa: SLF001 - 委譲後の現在エージェント検証
    assert model.closed


# ---------------------------------------------------------------------------
# デフォルトビルダー: Protocol 適合と registry 実運用経路
# ---------------------------------------------------------------------------
def test_default_builder_satisfies_protocol() -> None:
    """DefaultRealtimeAgentBuilder は RealtimeAgentBuilder（runtime_checkable）に適合する。"""
    assert isinstance(DefaultRealtimeAgentBuilder(), RealtimeAgentBuilder)


def test_default_builder_builds_real_realtime_agent() -> None:
    """builder 省略の RealtimeAgentRegistry().get() が実際の SDK RealtimeAgent を構築する。"""
    reg = RealtimeAgentRegistry()  # agent_builder 省略 → _adapters デフォルトを遅延生成。
    reg.register(RealtimeAgentSpec(name="solo", instructions="s"))
    agent = reg.get("solo")
    assert isinstance(agent, RealtimeAgent)
    assert agent.name == "solo"
    assert agent.instructions == "s"


def test_default_builder_wires_handoff_via_real_sdk() -> None:
    """デフォルトビルダー経路で handoff 結線され、SDK Handoff（委譲先名保持）が張られる。"""
    reg = RealtimeAgentRegistry()
    reg.register(RealtimeAgentSpec(name="triage", instructions="t", handoffs=["billing"]))
    reg.register(RealtimeAgentSpec(name="billing", instructions="b"))
    triage = reg.get("triage")
    billing = reg.get("billing")
    assert isinstance(triage, RealtimeAgent)
    # handoffs には SDK Handoff が結線され、agent_name で委譲先を指す。
    assert len(triage.handoffs) == 1
    assert triage.handoffs[0].agent_name == "billing"
    assert triage.handoffs[0].tool_name == "transfer_to_billing"
    assert isinstance(billing, RealtimeAgent)
