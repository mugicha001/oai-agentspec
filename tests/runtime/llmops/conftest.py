"""LLMOps 評価テスト用の共通フィクスチャ・ヘルパ（FakeModel Judge / fake RunResult）。

評価対象 Agent は FakeModel（実 LLM を呼ばない）。DeepEval 採点は `_adapters.judge` /
`judge_tools` を monkeypatch で固定 `CriterionResult` 返却に差し替える、または DeepEval metric を
モックすることで外部実通信なしで検証する。Langfuse クライアントは patch でモックする。
"""

from __future__ import annotations

from typing import Any

import pytest
from agents.items import ModelResponse
from agents.models.interface import Model

from oai_agentspec.runtime.deterministic import text_response
from oai_agentspec.runtime.llmops import JudgeConfig


class JudgeFakeModel(Model):
    """採点用 Judge モデルのスタブ（SDK Model ABC 実装・実 LLM を呼ばない）。

    `_make_judge_model` 経由で DeepEval custom model にラップされ得るが、本テスト群では
    DeepEval metric / `judge` 自体をモックするため LLM 呼び出しまで到達しない想定。到達した
    場合に備えて固定テキストを返す。
    """

    def __init__(self, text: str = "judge-ok") -> None:
        """固定応答テキストを設定する。"""
        self._text = text

    async def get_response(
        self,
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        """固定テキスト応答を返す。"""
        return text_response(self._text)

    async def stream_response(  # type: ignore[override]
        self,
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """未使用（非ストリーミング採点のみ）。"""
        raise NotImplementedError("JudgeFakeModel はストリーミング非対応")
        yield  # pragma: no cover - 到達しない（型のため）


@pytest.fixture
def judge_config() -> JudgeConfig:
    """FakeModel を採点モデルに据えた `JudgeConfig` を返す。"""
    return JudgeConfig(model=JudgeFakeModel())
