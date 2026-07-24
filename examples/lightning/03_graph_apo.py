"""ハンドオフを含む系全体を `prompt_slots` でまとめて最適化する例（実 API）。

`prompt_slots` は列挙したエージェント分の `Slot` を一括生成し `{名前: Slot}` mapping を返す。これを
`optimize(graph, slot=slots, registry=registry)` に渡すと、各スロットの `build` から rebind が
自動導出され、手書きの rebind / build なしでグラフ全体 APO が実質 2 行で書ける。最適化対象は
`agents=[...]` で列挙したエージェントのみで、未掲載のエージェント（ここでは support）のプロンプトは
固定される。`registry` はクローンされて差し替えられるため、利用者の registry は汚れない。

APO は agentlightning 0.3 では単一プロンプト最適化のため、複数スロット mapping はライブラリが
順次 APO へ通す（前のスロットの最良で次のスロットの seed コンテキストを更新）。APO 計算用クライ
アントは `apo_client=` で直接渡す。検証データ `val` は必須。

Azure OpenAI の環境変数を設定して実行:
    uv run python examples/lightning/03_graph_apo.py

導入: pip install 'oai-agentspec[lightning]'
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph, PromptLayout, PromptStore
from oai_agentspec.runtime.lightning import (
    OptimizeCase,
    judge,
    optimize,
    prompt_slots,
    train_val_split,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_client, azure_model  # noqa: E402

PROMPTS_AGENTS = Path(__file__).resolve().parent.parent / "prompts" / "agents"
LAYOUT = PromptLayout(base="base", parts="parts", agents="agents")


def build_registry(model: object) -> tuple[AgentRegistry, HandoffGraph]:
    """triage / billing / support を登録し triage 起点のハンドオフ系を組む。"""
    registry = AgentRegistry()
    for name in ("triage", "billing", "support"):
        spec = AgentSpec(name=name, instructions="(seed は store から読む)", model=model)
        registry.register(spec)

    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "billing", description="請求関連")
    graph.edge("triage", "support", description="技術問い合わせ")
    graph.apply(registry)
    registry.validate()
    return registry, graph


async def main() -> None:
    model = azure_model()
    registry, graph = build_registry(model)
    store = PromptStore(PROMPTS_AGENTS, LAYOUT)

    # triage と billing のプロンプトのみ系全体で同時最適化する（support は固定）。tune 省略時は
    # 各 slot で agent セグメントのみ最適化される（従来動作と同一）。エージェントごとに異なる
    # セレクタを使いたい場合は `tune={"billing": ["main", "billing"]}` のように agent 名をキーと
    # する dict を渡す（未指定 agent は tune=None に縮退・agents に無いキーは fail-closed）。
    slots = prompt_slots(
        store, registry, agents=["triage", "billing"], vars={"company": "AgentSpec Inc."}
    )

    data = [
        OptimizeCase(input="二重請求の返金をお願いしたいです"),
        OptimizeCase(input="請求書の宛名を変更したい"),
        OptimizeCase(input="支払い方法をクレジットカードに変えたい"),
        OptimizeCase(input="領収書を再発行してほしい"),
    ]
    train, val = train_val_split(data, val_ratio=0.25, seed=0)

    result = await optimize(
        graph,  # 第1引数は最適化対象（ここではグラフ）。スロット mapping は slot= で渡す。
        train=train,
        val=val,
        reward=judge(
            rubric="請求に関する依頼を billing 担当が的確に処理できる応答になっているか。",
            model=model,
        ),
        slot=slots,
        registry=registry,  # グラフ最適化では registry 必須（未指定は CONFIG_MISSING エラー）。
        apo_client=azure_client(),
        # E2E 動作確認用に最小 APO 設定（1 ラウンド・1 候補）。本番では rounds / beam を増やす。
        rounds=1,
        apo_beam_width=1,
        apo_branch_factor=1,
    )

    print(f"train_score={result.train_score:.3f}")

    # 各 slot の変更点（diff）を全文表示する。base/parts は全 slot 共通で長いため、before/after
    # を切り詰めて並べると共通部しか見えず差が読み取れない。diff なら変更箇所だけが ± で浮かぶ。
    diff_dict = result.diff if isinstance(result.diff, dict) else {}
    for name, diff_text in diff_dict.items():
        print()
        print(f"=== [{name}] ===")
        if diff_text:
            print(diff_text)
        else:
            print("(変更なし: APO が seed より良い候補を採用しなかった)")


if __name__ == "__main__":
    asyncio.run(main())
