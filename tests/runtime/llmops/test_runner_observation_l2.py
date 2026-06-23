"""L2: `DefaultRunnerAdapter.run_with_observation` を FakeModel で検証する。

生 `RunResult` を `_adapters` 内で消費し plain `(RunOutcome, ObservedRun)` のみを返すこと
（生 `RunResult` を外へ出さない・NFR-1）・最終出力 / route / tool_calls の plain 抽出を確認する。
"""

from __future__ import annotations

import pytest

from oai_agentspec import AgentSpec
from oai_agentspec._adapters import DefaultRunnerAdapter, build_agent
from oai_agentspec._adapters.runner import RunOutcome
from oai_agentspec.runtime.llmops import ObservedRun, RouteStep

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.integration


def _agent(name: str = "bot", text: str = "hi"):
    """FakeModel を据えた構築済み Agent を返す。"""
    return build_agent(AgentSpec(name=name, instructions="i", model=FakeModel().queue_text(text)))


@pytest.mark.asyncio
async def test_run_with_observation_returns_plain_tuple() -> None:
    """plain な (RunOutcome, ObservedRun) を返す（生 RunResult を外に出さない）。"""
    runner = DefaultRunnerAdapter()
    outcome, observation = await runner.run_with_observation(_agent(text="answer"), "ask")
    assert isinstance(outcome, RunOutcome)
    assert isinstance(observation, ObservedRun)
    # plain 型のみ（RunResult 型でないこと）。
    assert type(outcome).__name__ == "RunOutcome"
    assert type(observation).__name__ == "ObservedRun"


@pytest.mark.asyncio
async def test_run_with_observation_captures_final_output() -> None:
    """RunOutcome.final_output に最終応答テキストが入る。"""
    runner = DefaultRunnerAdapter()
    outcome, _ = await runner.run_with_observation(_agent(text="the answer"), "ask")
    assert outcome.interrupted is False
    assert outcome.final_output == "the answer"


@pytest.mark.asyncio
async def test_run_with_observation_route_includes_last_agent() -> None:
    """単体実行で handoff が無くても route に最終応答 agent が含まれる。"""
    runner = DefaultRunnerAdapter()
    _, observation = await runner.run_with_observation(_agent(name="solo"), "ask")
    assert observation.route.last_agent == "solo"
    assert observation.route.steps == [RouteStep(agent="solo", handoff_from=None)]


@pytest.mark.asyncio
async def test_run_with_observation_no_tool_calls_for_text_only() -> None:
    """ツールを呼ばないテキスト応答では tool_calls は空。"""
    runner = DefaultRunnerAdapter()
    _, observation = await runner.run_with_observation(_agent(), "ask")
    assert observation.tool_calls == []
