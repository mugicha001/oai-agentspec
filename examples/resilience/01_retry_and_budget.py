"""Resilience 系宣言型の 1 本例: Model Retry + Run Budget + streaming の例外観測。

本例では以下を一度に示す:

- `ModelRetryPolicy` で「まっとうな retry」を 1 行で宣言（silent no-op 排除）。
  Agent 単位（`Agent.model_settings.retry`）と Runner 共通（`RunConfig.model_settings.retry`）
  の両方を指定し、SDK `_merge_retry_settings` により Runner 側が優先されることを示す。
- `RunBudgetPolicy` で run 全体の累積トークン上限を宣言し、`build_run_budget_hooks(policy)`
  を `Runner.run(hooks=...)` に渡す。
- ハード timeout が必要な場合は利用者が `asyncio.wait_for` を被せる（本 lib はターン境界
  の graceful 判定のみ・docstring で案内）。
- streaming（`Runner.run_streamed`）では例外は `stream_events()` 消費時に初めて raise
  される（`async for` の外側の `try/except` で捕捉する）ことを併記する。

Azure OpenAI の環境変数（AZURE_OPENAI_*・examples/_shared/_azure.py 参照）を設定して実行:

    uv run python examples/resilience/01_retry_and_budget.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agents import Agent, ModelSettings, RunConfig, Runner
from agents.lifecycle import RunHooksBase

from oai_agentspec.exceptions import RunBudgetExceeded
from oai_agentspec.runtime.hooks import chain_hooks
from oai_agentspec.runtime.resilience import (
    ModelRetryPolicy,
    RunBudgetPolicy,
    build_model_retry,
    build_run_budget_hooks,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402


async def _run_normal() -> None:
    """通常フロー: Agent 単位と Runner 共通の Model retry を併用し、run 予算を設定する。"""
    # Agent 単位: この agent 独自に max_retries=2 を宣言する（既定フラグで OK）
    agent = Agent(
        name="assistant",
        instructions="You are a helpful assistant. Reply in one sentence.",
        model=azure_model(),
        model_settings=ModelSettings(
            retry=build_model_retry(ModelRetryPolicy(max_retries=2)),
        ),
    )

    # Runner 共通: SDK _merge_retry_settings により Runner 側の max_retries=4 が優先される
    common_retry = build_model_retry(
        ModelRetryPolicy(
            max_retries=4,
            # 既定で network / timeout / 429 / 5xx / Retry-After の全てが有効
            extra_retry_statuses=(408,),  # 追加で 408 Request Timeout も retry 対象に
        )
    )

    # 累積上限: 1 run の合計トークンが 100,000 を超えたら RunBudgetExceeded
    hooks = build_run_budget_hooks(
        RunBudgetPolicy(max_total_tokens=100_000, max_elapsed_seconds=60.0)
    )

    try:
        # 併用時は SDK 側で先に処理される順序: (a) Model retry は SDK 内部で自動再試行、
        # (b) Run Budget は各ターンの on_llm_end 境界で判定、(c) 例外は SDK error_handlers
        # に捕まらず（MaxTurnsExceeded / ModelRefusalError 以外は対象外）呼び出し元まで
        # 素通しで伝播する。
        result = await Runner.run(
            agent,
            "Hello, what is the capital of France?",
            hooks=hooks,
            run_config=RunConfig(model_settings=ModelSettings(retry=common_retry)),
        )
        print("[normal] final_output =", result.final_output)
    except RunBudgetExceeded as exc:
        # 予算超過時の情報充実（設計 D7）
        print("[normal] budget exceeded:", exc)
        print("  exceeded    =", exc.context.get("exceeded"))
        print("  llm_calls   =", exc.context.get("llm_calls"))
        print("  total_tokens=", exc.usage.total_tokens)


async def _run_streaming_observation_pattern() -> None:
    """streaming 時の例外観測パターン: `async for` の外側の `try/except` で捕捉する。

    設計 D9 の重要な注意点: streaming では `RunBudgetExceeded` は background の run_loop
    タスクで raise され、`RunResult._stored_exception` に格納される。利用者が
    `stream_events()` を消費して初めてその例外が再送出される。イベントを回さないと
    観測されない SDK 仕様に依存するため、`async for` を必ず回す運用が前提。
    """
    agent = Agent(
        name="assistant",
        instructions="You are a helpful assistant. Reply in one sentence.",
        model=azure_model(),
    )

    hooks = build_run_budget_hooks(RunBudgetPolicy(max_total_tokens=100_000))

    streamed = Runner.run_streamed(agent, "Hello.", hooks=hooks)
    try:
        # 例外は for ループ内で raise される可能性があるため except は外側に置く。
        async for _event in streamed.stream_events():
            pass  # 実運用では event を UI 等に流す
        print("[streaming] final_output =", streamed.final_output)
    except RunBudgetExceeded as exc:
        print("[streaming] budget exceeded during stream_events():", exc)


async def _run_hard_timeout_pattern() -> None:
    """ハード timeout（tool 実行中を含めた即時中断）が必要な場合の案内パターン。

    本 lib の `RunBudgetPolicy(max_elapsed_seconds=...)` はターン境界のみで判定する
    graceful 挙動。tool 実行中でも切りたい場合は利用者が `asyncio.wait_for` を被せる。
    """
    agent = Agent(
        name="assistant",
        instructions="Reply briefly.",
        model=azure_model(),
    )

    try:
        result = await asyncio.wait_for(Runner.run(agent, "Hello."), timeout=30.0)
        print("[hard-timeout] final_output =", result.final_output)
    except TimeoutError:
        print("[hard-timeout] hard timeout tripped by asyncio.wait_for")


class _LoggingHooks(RunHooksBase):  # type: ignore[type-arg]
    """LLM 呼び出し境界を print で記録する軽量な自作 hooks（chain_hooks 実演用）。"""

    def __init__(self, label: str) -> None:
        super().__init__()
        self._label = label
        self._calls = 0

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:  # noqa: ANN001
        self._calls += 1
        print(f"[{self._label}] on_llm_start #{self._calls}")

    async def on_llm_end(self, context, agent, response) -> None:  # noqa: ANN001
        print(f"[{self._label}] on_llm_end")


async def _run_chain_hooks_pattern() -> None:
    """`chain_hooks` で budget hooks と自作 logging hooks を合成する（Issue #31）。

    `Runner.run(hooks=...)` は単数の `RunHooksBase` しか受け付けない。複数 hook を併用したい
    場合は `chain_hooks(*hooks)` で宣言順に順次 `await` する単一 hook にまとめる。前段が
    raise したら後段は呼ばれず例外がそのまま伝播する（fail-fast）。
    """
    agent = Agent(
        name="assistant",
        instructions="Reply briefly.",
        model=azure_model(),
    )

    budget_hooks = build_run_budget_hooks(RunBudgetPolicy(max_total_tokens=100_000))
    logging_hooks = _LoggingHooks(label="chain")

    # 宣言順に順次 await される: budget_hooks -> logging_hooks
    hooks = chain_hooks(budget_hooks, logging_hooks)

    try:
        result = await Runner.run(agent, "Hello.", hooks=hooks)
        print("[chain] final_output =", result.final_output)
    except RunBudgetExceeded as exc:
        print("[chain] budget exceeded:", exc)


async def main() -> None:
    """4 パターンを順に実行する。"""
    await _run_normal()
    await _run_streaming_observation_pattern()
    await _run_hard_timeout_pattern()
    await _run_chain_hooks_pattern()


if __name__ == "__main__":
    asyncio.run(main())
