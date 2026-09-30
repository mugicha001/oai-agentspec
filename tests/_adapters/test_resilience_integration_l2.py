"""T8 の L2 統合テスト。Runner 実測で FR-4/FR-5 の受け入れ基準を担保する。

`test_resilience_l2.py`（ユニット L2・hooks/build を直接叩く）とは別に、本モジュールは
SDK `Runner.run` / `run_streamed` / `run_sync` を実際に走らせ、resilience 系が SDK 内部で
正しく機能することを実測で pin する:

- A: `Runner.run` 経由の `RunBudgetExceeded`（トークン上限・非 streaming）と正常完了。
- B: `Runner.run_streamed` 経由の予算超過（elapsed 上限・stream_events 消費で観測）。
- C: `Runner.run_sync` 経由の予算超過（run と等価な hooks 挙動）。
- D: SDK `error_handlers` との共存（正常時は素通し・`RunBudgetExceeded` は非捕捉で伝播）。
- E: `ModelSettings.resolve` による retry マージ（Runner 側が優先・SDK 委譲の検証）。
- F: 有効条件ゼロ x max_retries 正 の `ValueError`（build_model_retry へ到達しない）。
- G: usage 欠損の warning（トークン 3 項目がすべて 0 で判定し requests を見ない・ADR-0046）。

SDK 直 import は L2 のため許容する。build 関数・宣言型は公開窓口
`oai_agentspec.runtime.resilience` から取得する（NFR-1 隔離を利用者視点でも維持）。
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from agents import (
    Agent,
    ModelSettings,
    RunErrorHandlerResult,
    Runner,
    Usage,
)
from agents.items import ModelResponse
from agents.models.interface import Model
from openai.types.responses import ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from oai_agentspec._adapters import build_agent
from oai_agentspec.constants import RESILIENCE_LOGGER_NAME
from oai_agentspec.exceptions import RunBudgetExceeded
from oai_agentspec.runtime.resilience import (
    ModelRetryPolicy,
    RunBudgetPolicy,
    build_model_retry,
    build_run_budget_hooks,
)
from oai_agentspec.workflow import END, START, WorkflowGraph

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# ヘルパ
# ---------------------------------------------------------------------------
def _agent_with(model: FakeModel, name: str = "budget_agent") -> Agent[Any]:
    """FakeModel を据えた最小 Agent を作る（SDK 直結）。"""
    return Agent(name=name, instructions="test", model=model)


def _max_turns_handler(_inp: Any) -> RunErrorHandlerResult:
    """SDK `error_handlers['max_turns']` に載せるダミーハンドラ（発火時は stopped を返す）。"""
    return RunErrorHandlerResult(final_output="stopped")


# ===========================================================================
# A. Runner.run 経由の RunBudgetExceeded（非 streaming）
# ===========================================================================
async def test_A1_run経由でトークン上限超過時にRunBudgetExceededをraiseする() -> None:
    """usage 付き応答の累積が上限超過するターンで `RunBudgetExceeded` が Runner から伝播する。"""
    model = FakeModel().queue_text_with_usage("done", total_tokens=60)
    agent = _agent_with(model, name="agent_a1")
    hooks = build_run_budget_hooks(RunBudgetPolicy(max_total_tokens=50))

    with pytest.raises(RunBudgetExceeded) as exc_info:
        await Runner.run(agent, input="hi", hooks=hooks)

    exc = exc_info.value
    assert exc.usage.total_tokens > 50
    assert exc.context["exceeded"] == "max_total_tokens"
    assert exc.context["agent_name"] == "agent_a1"
    assert exc.context["llm_calls"] >= 1


async def test_A2_上限に達しなければRunner_runは正常完了する() -> None:
    """超過しない上限（100 万）では予算 hooks を挟んでも RunResult を正常取得できる。"""
    model = FakeModel().queue_text_with_usage("ok", total_tokens=60)
    agent = _agent_with(model, name="agent_a2")
    hooks = build_run_budget_hooks(RunBudgetPolicy(max_total_tokens=1_000_000))

    result = await Runner.run(agent, input="hi", hooks=hooks)

    assert result.final_output == "ok"


# ===========================================================================
# B. Runner.run_streamed 経由の RunBudgetExceeded（streaming）
# ===========================================================================
def _slow_workflow_agent(name: str, *, sleep_seconds: float) -> Agent[Any]:
    """関数ノードで一定時間ブロックする WorkflowModel Agent（elapsed 予算超過の実測用）。

    `WorkflowModel` は usage を返さない（token 予算では超過を作れない）ため、確実な超過は
    elapsed 上限で作る。関数ノードで `time.sleep` を挟み、model 呼び出し（on_llm_start〜
    on_llm_end）の window を上限より確実に長くする。
    """
    wf = WorkflowGraph(name)
    wf.add_function_node("slow", fn=lambda msg, ctx: (time.sleep(sleep_seconds), msg)[1])
    wf.add_edge(START, "slow")
    wf.add_edge("slow", END)
    return build_agent(wf.as_agent_spec(name))


async def test_B1_run_streamedはstream_events消費中に予算超過をraiseする() -> None:
    """elapsed 上限超過は `stream_events()` 消費中に `RunBudgetExceeded` として観測される。

    設計 D9（streaming は stream_events を回さないと例外が観測されない）を pin するため、
    for ループの外側の except（`pytest.raises`）で捕まえる形にする。
    """
    agent = _slow_workflow_agent("stream_budget", sleep_seconds=0.02)
    hooks = build_run_budget_hooks(RunBudgetPolicy(max_elapsed_seconds=0.001))

    streamed = Runner.run_streamed(agent, input="hi", hooks=hooks)
    with pytest.raises(RunBudgetExceeded) as exc_info:
        async for _event in streamed.stream_events():
            pass

    assert exc_info.value.context["exceeded"] == "max_elapsed_seconds"


# ===========================================================================
# C. Runner.run_sync 経由の RunBudgetExceeded
# ===========================================================================
def test_C1_run_sync経由でもトークン上限超過でRunBudgetExceededをraiseする() -> None:
    """`run_sync` は内部で `run` を呼ぶため hooks の予算判定は run と等価に働く。"""
    model = FakeModel().queue_text_with_usage("done", total_tokens=60)
    agent = _agent_with(model, name="agent_c1")
    hooks = build_run_budget_hooks(RunBudgetPolicy(max_total_tokens=50))

    with pytest.raises(RunBudgetExceeded) as exc_info:
        Runner.run_sync(agent, input="hi", hooks=hooks)

    assert exc_info.value.context["exceeded"] == "max_total_tokens"
    assert exc_info.value.context["agent_name"] == "agent_c1"


# ===========================================================================
# D. SDK error_handlers との共存（非干渉）
# ===========================================================================
async def test_D1_error_handlersと予算hooksの重ねがけでも正常時は素通しする() -> None:
    """budget hooks と `error_handlers['max_turns']` を同時に渡しても正常完了する。"""
    model = FakeModel().queue_text_with_usage("ok", total_tokens=10)
    agent = _agent_with(model, name="agent_d1")
    hooks = build_run_budget_hooks(RunBudgetPolicy(max_total_tokens=1_000_000))

    result = await Runner.run(
        agent,
        input="hi",
        hooks=hooks,
        error_handlers={"max_turns": _max_turns_handler},
    )

    assert result.final_output == "ok"


async def test_D2_RunBudgetExceededはerror_handlersに捕まらず呼び出し元まで伝播する() -> None:
    """`error_handlers['max_turns']` を渡しても budget 超過は `RunBudgetExceeded` で伝播する。

    予算超過は `MaxTurnsExceeded` ではないため handler は発火せず、"stopped" にはならない。
    """
    model = FakeModel().queue_text_with_usage("done", total_tokens=60)
    agent = _agent_with(model, name="agent_d2")
    hooks = build_run_budget_hooks(RunBudgetPolicy(max_total_tokens=50))

    with pytest.raises(RunBudgetExceeded):
        await Runner.run(
            agent,
            input="hi",
            hooks=hooks,
            error_handlers={"max_turns": _max_turns_handler},
        )


# ===========================================================================
# E. Agent / Runner マージ（ModelRetryPolicy）
# ===========================================================================
def test_E1_retry設定はresolveでRunner側のmax_retriesが優先される() -> None:
    """SDK `ModelSettings.resolve`（`_merge_retry_settings`）で override（Runner 側）が優先。

    lib 側にマージ機構は無く SDK 委譲であるため、Agent 側=2 と Runner 側=5 を resolve して
    結果が 5 になることで「Runner 側優先」の SDK 挙動を実測 pin する。
    """
    agent_settings = ModelSettings(retry=build_model_retry(ModelRetryPolicy(max_retries=2)))
    run_settings = ModelSettings(retry=build_model_retry(ModelRetryPolicy(max_retries=5)))

    merged = agent_settings.resolve(run_settings)

    assert merged.retry is not None
    assert merged.retry.max_retries == 5


# ===========================================================================
# F. 有効条件ゼロ x max_retries 正 の ValueError（build-time）
# ===========================================================================
def test_F1_有効な_retry条件ゼロでmax_retries正はbuild前にValueError() -> None:
    """全 retry_on_* False かつ extra 空で max_retries>0 は宣言時に `ValueError`。

    `ModelRetryPolicy.__post_init__` で fail-fast するため `build_model_retry` へ到達しない。
    """
    with pytest.raises(ValueError):
        ModelRetryPolicy(
            max_retries=3,
            retry_on_network_error=False,
            retry_on_timeout=False,
            retry_on_rate_limit=False,
            retry_on_server_error=False,
            retry_on_retry_after=False,
        )


# ===========================================================================
# G. usage 欠損の warning（ADR-0046: トークン 3 項目がすべて 0 で判定し requests を見ない）
# ===========================================================================
class _UsageStreamingModel(Model):
    """ストリーム終端の `ResponseCompletedEvent` に任意の usage を載せる検証用 Model。

    FakeModel は stream_response 非対応のため、終端 usage の形を制御する最小の fake を置く。
    SDK のストリームループは終端 `Response.usage` が在ればトークンを写して `requests=1` の
    `Usage` を組み、無ければ `Usage()` を組んで `on_llm_end` へ渡す。

    Attributes:
        terminal_usage: 終端 `Response.usage` に載せる値（None なら usage を載せない）。
    """

    def __init__(self, terminal_usage: ResponseUsage | None) -> None:
        """終端に載せる usage を受け取る。"""
        self.terminal_usage = terminal_usage

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """非ストリーム経路は本テストでは使わない。"""
        raise NotImplementedError("_UsageStreamingModel はストリーム専用")

    async def stream_response(  # type: ignore[override]
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Any]:
        """テキスト応答を completed イベント 1 件で終端し、終端に usage を載せる。"""
        from oai_agentspec._adapters import _completed_event
        from oai_agentspec.runtime.deterministic import text_response

        event = _completed_event(
            text_response("done").output, 0, response_id="resp_usage", model="usage-fake"
        )
        event.response.usage = self.terminal_usage
        yield event


def _zero_response_usage() -> ResponseUsage:
    """トークンがすべて 0 の `ResponseUsage`（usage の欠けた応答をストリーム終端で表す）。"""
    return ResponseUsage(
        input_tokens=0,
        input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
        output_tokens=0,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=0,
    )


def _model_with_usage(usage: Usage) -> FakeModel:
    """任意の `Usage` を載せたテキスト応答を 1 件返す FakeModel を作る。"""
    from oai_agentspec.runtime.deterministic import text_response

    model = FakeModel()
    model.responses.append(dataclasses.replace(text_response("done"), usage=usage))
    return model


def _usage_missing_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """resilience logger が出した WARNING のレコードだけを抜き出す。"""
    return [
        r
        for r in caplog.records
        if r.name == RESILIENCE_LOGGER_NAME and r.levelno == logging.WARNING
    ]


async def test_G1_非ストリームでrequests1かつトークン0の応答はusage欠損としてwarningを1件出す(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Model が `Usage(requests=1)`（トークンがすべて 0）を返すと warning が 1 件出る。

    SDK 組み込みモデルは usage の欠けた応答でも requests を 1 と数える。欠損判定は requests を
    見ずトークン 3 項目で行うため、この形も欠損として検知する（ADR-0046）。
    """
    agent = _agent_with(_model_with_usage(Usage(requests=1)), name="agent_g1")
    hooks = build_run_budget_hooks(RunBudgetPolicy(max_total_tokens=1_000_000))

    with caplog.at_level(logging.WARNING, logger=RESILIENCE_LOGGER_NAME):
        await Runner.run(agent, input="hi", hooks=hooks)

    assert len(_usage_missing_warnings(caplog)) == 1


async def test_G2_ストリーム終端のusageがトークン0ならwarningを1件出す(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ストリーム終端にトークンがすべて 0 の usage を載せると warning が 1 件出る。

    SDK のストリームループはこの形で `requests=1` の `Usage` を組むため、requests を見る判定
    では検知できない。トークン 3 項目で判定すると欠損になる（ADR-0046）。
    """
    agent = Agent(
        name="agent_g2", instructions="test", model=_UsageStreamingModel(_zero_response_usage())
    )
    hooks = build_run_budget_hooks(RunBudgetPolicy(max_total_tokens=1_000_000))

    with caplog.at_level(logging.WARNING, logger=RESILIENCE_LOGGER_NAME):
        streamed = Runner.run_streamed(agent, input="hi", hooks=hooks)
        async for _event in streamed.stream_events():
            pass

    assert len(_usage_missing_warnings(caplog)) == 1


async def test_G3_totalを埋めずin_outを持つusageはwarningを出さない(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`Usage(requests=0, input=5, output=3, total=0)` は欠損ではなく warning は 0 件。

    total_tokens を埋めないモデル実装でも in / out が在ればトークンは取得できている
    （ADR-0046 Decision 1。total だけを見る判定・requests を見る判定はここで落ちる）。
    """
    usage = Usage(requests=0, input_tokens=5, output_tokens=3, total_tokens=0)
    agent = _agent_with(_model_with_usage(usage), name="agent_g3")
    hooks = build_run_budget_hooks(RunBudgetPolicy(max_total_tokens=1_000_000))

    with caplog.at_level(logging.WARNING, logger=RESILIENCE_LOGGER_NAME):
        await Runner.run(agent, input="hi", hooks=hooks)

    assert _usage_missing_warnings(caplog) == []


async def test_G4_ストリーム終端にusageが無い場合もwarningを1件出す(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """回帰テスト: ストリーム終端の usage が None でも warning が 1 件出る。

    SDK は終端 usage が無いとき `Usage()`（requests も 0）を組むため、判定条件の変更前から
    欠損として検知される。判定条件の変更後もこの経路が欠損のままであることを固定する。
    """
    agent = Agent(name="agent_g4", instructions="test", model=_UsageStreamingModel(None))
    hooks = build_run_budget_hooks(RunBudgetPolicy(max_total_tokens=1_000_000))

    with caplog.at_level(logging.WARNING, logger=RESILIENCE_LOGGER_NAME):
        streamed = Runner.run_streamed(agent, input="hi", hooks=hooks)
        async for _event in streamed.stream_events():
            pass

    assert len(_usage_missing_warnings(caplog)) == 1
