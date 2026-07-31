"""Next-Turn Agent Override の 1 本例: 主経路（宣言 -> 適用 -> 解決 -> 次ターン継続）。

SDK 標準の last_agent 継続（`Runner.run(result.last_agent, ...)`）では、ハンドオフで専門
エージェントへ遷移したあと次ターンも専門エージェントから始まる。本例では以下を示す:

- `NextTurnPolicy(rules={回答者名: ルール})` を 1 回宣言し、`apply_next_turn_policy` で
  名前整合検証済みの派生 registry を得る（元 registry は変更されない）。以降の `get` と
  `next_turn_agent` には必ずこの派生 registry を渡す（元 registry では禁止が働かない）。
- 主経路: 宣言 -> `apply_next_turn_policy` -> `Runner.run` -> `next_turn_agent` ->
  次ターンの `Runner.run`。次ターンの実行を呼ぶのは利用者コード（build-don't-run）。
- 値位置の多態: ルールの列（到達元条件付き + 包括）と、次ターン名の str 略記。到達元で
  戻し先を変えたい場合は `source` 付きルールを先に並べる（選定は「一致 source -> 包括」）。
- `no_handoff_on_arrival=True`（到達時ハンドオフ禁止）を宣言すると、ハンドオフで到達した
  ターンだけ当該エージェントの全 handoff が無効化され、たらい回しが止まる。記録は run 内
  一時状態なので次ターンには持ち越されない（次ターンの billing は元の handoff 構成で動く）。
- `next_turn_agent` は上書き発動時に Y の Agent、非発動時に `result.last_agent` を返す。
  `last_agent` も取得できない場合のみ None（開始エージェント決定不能）。

単体 API での自前分岐は `02_resolve_only.py`、禁止のみルールは `03_prohibit_only.py` を参照。

Azure OpenAI の環境変数（AZURE_OPENAI_*・examples/_shared/_azure.py 参照）を設定して実行:

    uv run python examples/next_turn/01_next_turn_override.py

本例は実 API を呼ぶ（合計 2 回）。
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

# 回答者名 -> 上書きルール。billing は到達元で戻し先を変え、support は str 略記。
POLICY = NextTurnPolicy(
    rules={
        "billing": (
            # triage からの到達: handoff を止めて billing に答えさせ、次ターンは triage から。
            NextTurnRule(next_agent="triage", no_handoff_on_arrival=True, source="triage"),
            # それ以外（support 等）からの到達: 禁止のみ（次ターンは last_agent 継続）。
            NextTurnRule(no_handoff_on_arrival=True),
        ),
        "support": "triage",  # 次ターン指定だけの単一ルールは名前の str 略記で書ける
    }
)


def build_registry() -> tuple[AgentRegistry, HandoffGraph]:
    """triage <-> billing / support のハンドオフを持つ registry を組む。

    Returns:
        登録済み registry と、宣言したハンドオフグラフ。
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

    return registry, graph


async def main() -> None:
    """主経路: 宣言 -> 適用 -> run -> `next_turn_agent` -> 次ターン継続。"""
    registry, graph = build_registry()
    print("--- handoff graph ---")
    print(graph.mermaid())

    # 名前整合を検証し、到達時ハンドオフ禁止を結線した派生 registry を返す。
    # 元 registry は変更されない（適用は全登録の完了後に行うこと）。
    runtime_registry = apply_next_turn_policy(POLICY, registry)

    result = await Runner.run(
        runtime_registry.get("triage"), input="先月の請求額が想定より高いのですが確認できますか"
    )
    # triage -> billing のハンドオフが現れ、billing は自分で回答を終える
    # （到達時ハンドオフ禁止により billing -> support の辺が当該ターンだけ無効）。
    print_run_path(result)
    print(result.final_output)

    # 上書き発動時は Y（triage）の Agent、非発動時は result.last_agent が返る。
    agent = next_turn_agent(POLICY, result, runtime_registry)
    if agent is None:
        print("次ターンの開始エージェントを決定できませんでした")
        return
    print(f"--- 次ターンの開始エージェント: {agent.name} ---")

    # 次ターンの実行は利用者コードが呼ぶ（lib は解決までで実行しない）。
    # 禁止は run を跨がないため、この triage は元の handoff 構成で動く。
    follow_up = await Runner.run(
        agent,
        input=result.to_input_list()
        + [{"role": "user", "content": "ではパスワードの再設定手順も教えてください"}],
    )
    print_run_path(follow_up)
    print(follow_up.final_output)


if __name__ == "__main__":
    asyncio.run(main())
