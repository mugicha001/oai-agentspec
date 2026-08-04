"""決定的応答モデルの 1 本例: ルール関数と応答ビルダで実 API 抜きに run を完走させる。

`DeterministicResponseModel` は、実 LLM を呼ばずに **利用者が渡すルール関数が入力から応答を
決める** ステートレスな `Model` 実装である。想定用途は自動テスト・実 API を呼ばないオフライン
開発・デモ実行・決定的なシナリオ再生の 4 つで、テスト専用の機構ではない。

本例で示すこと:

- ルール関数は `ModelRequest` を受け、応答ビルダ（`text_response` 等）の戻り値を返す純関数。
  同期関数と async 関数の双方を受理する。
- **ステートレス**: 内部キューを消費しないため、同じ入力の run を何度実行しても応答は変わらず、
  同一インスタンスを複数の Agent で共有しても呼び出し順序に依存しない。
- `None` を返すと空テキスト応答になり run は正常終了する。ルール関数が送出した例外は
  握り潰されず伝播する。
- 応答ビルダは `from agents` / `from openai` を書かずに応答オブジェクトを組み立てる。

実行は利用者責務である（build-don't-run）。本 example は `Runner.run` を呼ぶが、
ライブラリ側に実行 API は無い。

モデル呼び出しは実 API へ接続しない（ネットワーク不要）。

実行:
    uv run python examples/deterministic/01_rule_model.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from agents import Agent, Runner

from oai_agentspec.runtime.deterministic import (
    DeterministicResponseModel,
    ModelRequest,
    text_response,
    text_response_with_usage,
)


def rule(request: ModelRequest) -> Any:
    """入力から応答を決める純関数（同期）。

    Args:
        request: 1 回分のモデル呼び出し入力。

    Returns:
        応答オブジェクト、または応答を決められない場合は `None`。
    """
    if "使用量" in request.user_text:
        # usage を載せると run 予算の検証にも使える
        return text_response_with_usage("使用量つきで返します", total_tokens=120, requests=1)
    if "無言" in request.user_text:
        return None  # 空テキスト応答として扱われる
    return text_response(f"echo: {request.user_text}")


async def async_rule(request: ModelRequest) -> Any:
    """async のルール関数（awaitable も受理される）。

    Args:
        request: 1 回分のモデル呼び出し入力。

    Returns:
        応答オブジェクト。
    """
    await asyncio.sleep(0)
    return text_response(f"async echo: {request.user_text}")


class RuleError(RuntimeError):
    """ルール関数が送出する例外（伝播の確認用）。"""


async def main() -> None:
    """基本・ステートレス性・None・async・例外伝播を順に示す。"""
    model = DeterministicResponseModel(rule)
    agent = Agent(name="assistant", instructions="アシスタントです", model=model)

    print("--- 基本")
    result = await Runner.run(agent, input="こんにちは")
    print(f"final_output = {result.final_output!r}")

    print("--- ステートレス（同一インスタンスで 2 回実行しても同じ）")
    first = await Runner.run(agent, input="こんにちは")
    second = await Runner.run(agent, input="こんにちは")
    print(f"1 回目 == 2 回目: {first.final_output == second.final_output}")

    print("--- 同一インスタンスを複数 Agent で共有")
    other = Agent(name="other", instructions="別のエージェント", model=model)
    shared = await Runner.run(other, input="共有できる")
    print(f"final_output = {shared.final_output!r}")

    print("--- usage を載せる")
    usage_result = await Runner.run(agent, input="使用量を教えて")
    print(f"final_output = {usage_result.final_output!r}")

    print("--- ルール関数が None を返す")
    empty = await Runner.run(agent, input="無言で")
    print(f"final_output = {empty.final_output!r}（空文字）")

    print("--- async のルール関数")
    async_agent = Agent(
        name="async",
        instructions="async",
        model=DeterministicResponseModel(async_rule),
    )
    async_result = await Runner.run(async_agent, input="非同期")
    print(f"final_output = {async_result.final_output!r}")

    print("--- ルール関数の例外は握り潰されず伝播する")

    def failing_rule(request: ModelRequest) -> Any:
        raise RuleError("ルール関数の中で失敗した")

    failing_agent = Agent(
        name="failing",
        instructions="failing",
        model=DeterministicResponseModel(failing_rule),
    )
    try:
        await Runner.run(failing_agent, input="失敗させる")
    except RuleError as exc:
        print(f"RuleError: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
