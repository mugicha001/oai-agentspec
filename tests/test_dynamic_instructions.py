"""L2: 実 Agent + FakeModel + Runner で動的 instructions の実注入を検証。"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from agents import Runner

from oai_agentspec import AgentRegistry, AgentSpec

from _helpers.fake_model import FakeModel


@dataclass
class Ctx:
    user_name: str
    plan: str


@pytest.mark.asyncio
async def test_sync_callable_instructions_injected() -> None:
    model = FakeModel().queue_text("done")
    reg = AgentRegistry()
    reg.register(
        AgentSpec(
            name="concierge",
            instructions=lambda ctx, agent: f"Serve {ctx.context.user_name} ({ctx.context.plan}).",
            model=model,
        )
    )
    agent = reg.get("concierge")
    await Runner.run(agent, input="hi", context=Ctx(user_name="Mugi", plan="premium"))
    assert model.calls[0].system_instructions == "Serve Mugi (premium)."


@pytest.mark.asyncio
async def test_async_callable_instructions_injected() -> None:
    model = FakeModel().queue_text("done")

    async def instr(ctx: object, agent: object) -> str:
        return f"Hello {ctx.context.user_name}."  # type: ignore[attr-defined]

    reg = AgentRegistry()
    reg.register(AgentSpec(name="a", instructions=instr, model=model))
    agent = reg.get("a")
    await Runner.run(agent, input="hi", context=Ctx(user_name="Mugi", plan="free"))
    assert model.calls[0].system_instructions == "Hello Mugi."


@pytest.mark.asyncio
async def test_static_instructions_injected() -> None:
    model = FakeModel().queue_text("done")
    reg = AgentRegistry()
    reg.register(AgentSpec(name="a", instructions="Static prompt.", model=model))
    agent = reg.get("a")
    await Runner.run(agent, input="hi", context=Ctx("x", "y"))
    assert model.calls[0].system_instructions == "Static prompt."
