"""L2: SDK 組み込みモデル経路での usage 欠損 pin（ADR-0046 の Confirmation 項目）。

`FakeModel`（テスト専用の最小 Model）ではなく、SDK 組み込みの `OpenAIResponsesModel` /
`OpenAIChatCompletionsModel` が usage を欠いた応答を実際にパースし、その結果を
`_adapters.resilience` の予算 hooks（warning）と `_adapters.intent.run_filler_prompt`
（`AgentRunUsage` への None 詰め替え）が正しく検知することを pin する。

ネットワーク・課金は一切発生しない: `openai.AsyncOpenAI` に `httpx2.AsyncClient` +
`httpx2.MockTransport` を挟み、`base_url` は解決不能ドメイン（`http://test.invalid/v1`）に
する。実 HTTP 接続は `MockTransport` がソケットを開かず完結させるため発生せず、
`tests/conftest.py` の loopback 限定ネットワークガードにも抵触しない。usage フィールドを
意図的に省いた応答 JSON を返すことで、SDK 0.22.3 の usage 欠損時挙動
（Chat: `openai_chatcompletions.py`、Responses: `openai_responses.py` / `usage.py`）を経由した
実測パスを固定する。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from agents import Agent, OpenAIChatCompletionsModel, OpenAIResponsesModel, Runner
from openai import AsyncOpenAI

from oai_agentspec._adapters.intent import AgentRunUsage, run_filler_prompt
from oai_agentspec.constants import RESILIENCE_LOGGER_NAME
from oai_agentspec.runtime.resilience import RunBudgetPolicy, build_run_budget_hooks

pytestmark = pytest.mark.integration

_MODEL_NAME = "gpt-usage-missing-pin"

# usage フィールドは意図的に省く（欠損応答の再現）。
_RESPONSES_BODY: dict[str, Any] = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "status": "completed",
    "model": _MODEL_NAME,
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hi", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}

# usage フィールドは意図的に省く（欠損応答の再現）。
_CHAT_BODY: dict[str, Any] = {
    "id": "cc_1",
    "object": "chat.completion",
    "created": 0,
    "model": _MODEL_NAME,
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hi"},
        }
    ],
}


def _sse_responses() -> bytes:
    """Responses API の SSE ストリーム（usage 欠損の completed イベント）を組み立てる。"""
    events = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**_RESPONSES_BODY, "status": "in_progress", "output": []},
        },
        {"type": "response.completed", "sequence_number": 1, "response": _RESPONSES_BODY},
    ]
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


def _sse_chat() -> bytes:
    """Chat Completions API の SSE ストリーム（usage 欠損のチャンク列）を組み立てる。"""
    chunks = [
        {
            "id": "cc_1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": _MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hi"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "cc_1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": _MODEL_NAME,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"


def _handler(request: httpx2.Request) -> httpx2.Response:
    """usage を持たない Responses / Chat 応答を返す最小 handler（実ネットワーク不使用）。"""
    body = json.loads(request.content or b"{}")
    stream = bool(body.get("stream"))
    path = request.url.path
    if path.endswith("/responses"):
        if stream:
            return httpx2.Response(
                200, content=_sse_responses(), headers={"content-type": "text/event-stream"}
            )
        return httpx2.Response(200, json=_RESPONSES_BODY)
    if path.endswith("/chat/completions"):
        if stream:
            return httpx2.Response(
                200, content=_sse_chat(), headers={"content-type": "text/event-stream"}
            )
        return httpx2.Response(200, json=_CHAT_BODY)
    return httpx2.Response(404, json={"error": path})  # pragma: no cover - 想定外パス


def _mock_client() -> AsyncOpenAI:
    """`MockTransport` を挟んだ `AsyncOpenAI` を作る（解決不能ドメイン・実接続なし）。"""
    return AsyncOpenAI(
        api_key="test",
        base_url="http://test.invalid/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(_handler)),
        max_retries=0,
    )


def _responses_model() -> OpenAIResponsesModel:
    """usage 欠損応答を返す `OpenAIResponsesModel`（SDK 組み込み実装そのもの）を作る。"""
    return OpenAIResponsesModel(_MODEL_NAME, openai_client=_mock_client())


def _chat_model() -> OpenAIChatCompletionsModel:
    """usage 欠損応答を返す `OpenAIChatCompletionsModel`（SDK 組み込み実装そのもの）を作る。"""
    return OpenAIChatCompletionsModel(_MODEL_NAME, openai_client=_mock_client())


_MODEL_FACTORIES: list[tuple[str, Callable[[], Any]]] = [
    ("responses", _responses_model),
    ("chat_completions", _chat_model),
]


def _usage_missing_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """resilience logger が出した WARNING のレコードだけを抜き出す。"""
    return [
        r
        for r in caplog.records
        if r.name == RESILIENCE_LOGGER_NAME and r.levelno == logging.WARNING
    ]


# ===========================================================================
# A. resilience budget hooks x Runner.run（非 streaming）
# ===========================================================================
@pytest.mark.parametrize(("label", "make_model"), _MODEL_FACTORIES)
async def test_A_非stream応答でresilienceのusage欠損warningが1件出る(
    label: str, make_model: Callable[[], Any], caplog: pytest.LogCaptureFixture
) -> None:
    """組み込みモデルが usage 欠損の非 stream 応答を返すと resilience warning が 1 件出る。"""
    agent = Agent(name=f"agent_{label}", instructions="test", model=make_model())
    hooks = build_run_budget_hooks(RunBudgetPolicy(max_total_tokens=1_000_000))

    with caplog.at_level(logging.WARNING, logger=RESILIENCE_LOGGER_NAME):
        result = await Runner.run(agent, input="go", hooks=hooks)

    assert result.final_output == "hi"
    assert len(_usage_missing_warnings(caplog)) == 1


# ===========================================================================
# B. resilience budget hooks x Runner.run_streamed
# ===========================================================================
@pytest.mark.parametrize(("label", "make_model"), _MODEL_FACTORIES)
async def test_B_stream応答でもresilienceのusage欠損warningが1件出る(
    label: str, make_model: Callable[[], Any], caplog: pytest.LogCaptureFixture
) -> None:
    """`Runner.run_streamed` 経由でも usage 欠損の resilience warning が 1 件出る。"""
    agent = Agent(name=f"agent_stream_{label}", instructions="test", model=make_model())
    hooks = build_run_budget_hooks(RunBudgetPolicy(max_total_tokens=1_000_000))

    with caplog.at_level(logging.WARNING, logger=RESILIENCE_LOGGER_NAME):
        streamed = Runner.run_streamed(agent, input="go", hooks=hooks)
        async for _event in streamed.stream_events():
            pass

    assert streamed.final_output == "hi"
    assert len(_usage_missing_warnings(caplog)) == 1


# ===========================================================================
# C. intent run_filler_prompt 経由の usage None 詰め替え
# ===========================================================================
@pytest.mark.parametrize(("label", "make_model"), _MODEL_FACTORIES)
async def test_C_intentのfiller経由でusage欠損はNoneに詰め替わる(
    label: str, make_model: Callable[[], Any]
) -> None:
    """`run_filler_prompt` は usage 欠損応答を `AgentRunUsage` の None トークンへ詰め替える。

    Args:
        label: モデル種別ラベル（parametrize id 用）。
        make_model: usage 欠損応答を返す組み込みモデルを作るファクトリ。
    """
    agent = Agent(name=f"agent_filler_{label}", instructions="test", model=make_model())

    text, usage = await run_filler_prompt(agent, (), "go")

    assert text == "hi"
    assert usage == AgentRunUsage(model_calls=1, input_tokens=None, output_tokens=None)
