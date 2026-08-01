"""Next-Turn Agent Override の 1 本例: streaming 実行（`Runner.run_streamed`）での次ターン解決。

01-03 は非 streaming（`Runner.run`）だが、次ターン上書きと到達時ハンドオフ禁止はどちらも
streaming でも同じように使える。本例では以下を示す:

- `Runner.run_streamed` は coroutine ではなく `RunResultStreaming` を即座に返す（await しない）。
  実行が進むのは `stream_events()` を消費している間なので、**イベントを消費し切ってから**
  `next_turn_agent` / `resolve_next_agent` へ渡す。消費前の結果を渡すと `new_items` が空で
  ハンドオフを観測できず、上書きが発動しない。
- 消費完了後の `RunResultStreaming` は `new_items` と `last_agent` を `RunResult` と同名・
  同義で持つため、判定材料の読み取りは非 streaming と同一になる（型ではなく構造にのみ依存）。
- 到達時ハンドオフ禁止も streaming で効く。SDK は streaming 経路でも同じ `get_handoffs` を
  使って handoff の有効性を評価するため、禁止対象へ到達したターンではその出辺がモデルへ
  提示されない。本例の出力で見えるのは「billing が support へ回さず自分で答えた」までで、
  非提示そのものは LLM の選択と区別できない（機械的な証明は
  `tests/_adapters/test_next_turn_registry_l2.py` が `get_handoffs` の直接観測で担う）。
- 次ターンの継続入力は `to_input_list()` で組む（これも非 streaming と同じ）。

非 streaming の主経路は `01_next_turn_override.py`、単体 API での自前分岐は
`02_resolve_only.py`、禁止のみルールは `03_prohibit_only.py` を参照。

Azure OpenAI の環境変数（AZURE_OPENAI_*・examples/_shared/_azure.py 参照）を設定して実行:

    uv run python examples/next_turn/04_streaming.py

本例は実 API を呼ぶ（合計 2 回・いずれも streaming）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agents import Runner

from oai_agentspec import (
    AgentRegistry,
    AgentSpec,
    HandoffGraph,
    NextTurnPolicy,
    NextTurnRule,
    PromptLayout,
    PromptStore,
    apply_next_turn_policy,
    next_turn_agent,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402
from _run_path import print_run_path  # noqa: E402

PROMPT_VARS = {"company": "AgentSpec Inc."}
LAYOUT = PromptLayout(base="base", parts="parts", agents="agents")

# billing へ到達したターンは billing に答えさせ切り（たらい回しの遮断）、次ターンは triage から。
POLICY = NextTurnPolicy(
    rules={"billing": NextTurnRule(next_agent="triage", no_handoff_on_arrival=True)}
)


def build_registry() -> AgentRegistry:
    """triage -> billing / support のハンドオフを持つ registry を組む。

    Returns:
        登録済みの registry（`apply_next_turn_policy` へ渡す前の素の状態）。
    """
    store = PromptStore(Path(__file__).resolve().parent.parent / "prompts", LAYOUT)
    registry = AgentRegistry()
    model = azure_model()

    for name in ("triage", "billing", "support"):
        registry.register(
            AgentSpec(
                name=name,
                instructions=store.compose(
                    agent=name, base="main", parts=["style", "safety"], vars=PROMPT_VARS
                ),
                model=model,
            )
        )

    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "billing", description="請求・支払いの問い合わせは請求担当へ")
    graph.edge("triage", "support", description="技術的な問い合わせはサポートへ")
    # billing の出辺。到達時ハンドオフ禁止が効くターンでは、この辺が無効化される。
    graph.edge("billing", "support", description="技術的な内容はサポートへ")
    graph.edge("support", "triage", description="担当外は振り分けへ戻す")
    graph.apply(registry)
    registry.validate()

    return registry


async def drain(result: object) -> None:
    """`stream_events()` を最後まで消費し、届いたイベントの種別を数えて表示する。

    判定材料（`new_items` / `last_agent`）が揃うのは消費完了後なので、`next_turn_agent`
    へ渡す前に必ずこの消費を終える。イベント本体の逐次表示は本例の主題ではないため、
    種別ごとの件数だけを出す。

    Args:
        result: `Runner.run_streamed` の戻り値（`RunResultStreaming`）。
    """
    counts: dict[str, int] = {}
    async for event in result.stream_events():  # type: ignore[attr-defined]
        counts[event.type] = counts.get(event.type, 0) + 1
    summary = " / ".join(f"{name}={n}" for name, n in sorted(counts.items()))
    print(f"--- stream events: {summary} ---")


async def main() -> None:
    """streaming 経路: run_streamed -> 消費完了 -> `next_turn_agent` -> 次ターンも streaming。"""
    runtime_registry = apply_next_turn_policy(POLICY, build_registry())

    # await しない。ここでは実行はまだ進んでいない。
    result = Runner.run_streamed(
        runtime_registry.get("triage"), input="先月の請求額が想定より高いのですが確認できますか"
    )
    await drain(result)

    # 消費完了後は RunResult と同じ読み取りができる。triage -> billing のハンドオフが現れ、
    # billing は自分で回答を終える（到達時ハンドオフ禁止により billing -> support が
    # このターンだけモデルへ提示されないため、そもそも選べない）。
    print_run_path(result)
    print(result.final_output)

    agent = next_turn_agent(POLICY, result, runtime_registry)
    if agent is None:
        print("次ターンの開始エージェントを決定できませんでした")
        return
    print(f"--- 次ターンの開始エージェント: {agent.name} ---")

    # 次ターンの実行は利用者コードが呼ぶ（lib は解決までで実行しない）。継続入力の組み方も
    # 非 streaming と同じ。禁止は run を跨がないため、この triage は元の handoff 構成で動く。
    follow_up = Runner.run_streamed(
        agent,
        input=result.to_input_list()
        + [{"role": "user", "content": "ではパスワードの再設定手順も教えてください"}],
    )
    await drain(follow_up)
    print_run_path(follow_up)
    print(follow_up.final_output)


if __name__ == "__main__":
    asyncio.run(main())
