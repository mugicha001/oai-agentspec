"""意図予測テスト用の共通 Fake（`IntentFakeModel`）。

意図予測は LLM 出力を JSON として parse する。FakeModel はテストごとに異なる JSON
応答を返せるよう、コンストラクタで固定テキストを受け取る形にする（llmops の
JudgeFakeModel と同型）。
"""

from __future__ import annotations

from typing import Any

from agents.items import ModelResponse
from agents.models.interface import Model

from .responses import text_response


class IntentFakeModel(Model):
    """意図予測用 LLM のスタブ（SDK Model ABC 実装・実 LLM を呼ばない）。

    コンストラクタで固定応答テキスト（通常は JSON 文字列）を渡す。テストごとに
    期待する意図予測 JSON を注入できるようにする（`JudgeFakeModel` と同型）。
    """

    def __init__(self, text: str = '{"candidates": [], "report": null, "metadata": null}') -> None:
        """固定応答テキスト（通常は JSON 文字列）を設定する。"""
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
        """未使用（非ストリーミング意図予測のみ）。"""
        raise NotImplementedError("IntentFakeModel はストリーミング非対応")
        yield  # pragma: no cover - 到達しない（型のため）


class RecordingFakeModel(Model):
    """`get_response` の呼び出し引数（system_instructions・input）を記録する Fake。

    新 signature の `run_intent_prompt(model, system, history_items, user_content, *, context)`
    が SDK 側 `Agent`/`Runner.run` に対して何を渡したかを検証するために使う。

    `text` に `None` を指定すると `final_output` が `None` となる応答を返す
    （adapter 側が None を空文字に変換する契約を pin する目的）。
    """

    def __init__(self, text: str | None = "OK") -> None:
        """固定応答テキスト（`None` の場合は final_output が None になる）を設定する。"""
        self._text = text
        self.calls: list[dict[str, Any]] = []

    async def get_response(
        self,
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        model_settings: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        """呼び出し引数を記録し、固定応答（または None 応答）を返す。"""
        self.calls.append(
            {
                "system_instructions": system_instructions,
                "input": input,
                "model_settings": model_settings,
                "args": args,
                "kwargs": kwargs,
            }
        )
        if self._text is None:
            from agents import Usage
            from agents.items import ModelResponse as _ModelResponse

            return _ModelResponse(output=[], usage=Usage(), response_id=None)
        return text_response(self._text)

    async def stream_response(  # type: ignore[override]
        self,
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """未使用。"""
        raise NotImplementedError("RecordingFakeModel はストリーミング非対応")
        yield  # pragma: no cover - 到達しない（型のため）
