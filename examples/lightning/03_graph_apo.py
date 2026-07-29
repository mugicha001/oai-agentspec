"""ハンドオフを含む系全体を `prompt_slot_factory` でまとめて最適化する例（実 API）。

`prompt_slot_factory(store, registry, base=..., vars=...)` は per-agent で共通する既定値
（`store` / `registry` / `base` / `vars`）を束ねた `make_slot(agent, **overrides)` を返す。これを
エージェントごとに呼んでリストへ集め `optimize(graph, slot=slots, registry=registry)` に渡すと、
各スロットの `build` から rebind が自動導出され、手書きの rebind / build なしでグラフ全体 APO が
実質数行で書ける。最適化対象は `make_slot` を呼んだエージェントのみで、呼んでいないエージェント
（ここでは support）のプロンプトは固定される。`registry` はクローンされて差し替えられるため、
利用者の registry は汚れない。

APO は agentlightning 0.3 では単一プロンプト最適化のため、複数スロット mapping はライブラリが
順次 APO へ通す（前のスロットの最良で次のスロットの seed コンテキストを更新）。APO 計算用クライ
アントは `apo_client=` で直接渡す。検証データ `val` は必須。

注意（graph target の pre-flight route coverage・既定有効）:
`optimize()` は APO へ委譲する前に seed 状態で `train` 全件を 1 巡 rollout し、`slot` に挙げた
エージェント（本例では triage / billing）が routing で 1 度も到達しないと
`OptimizeError(FailureKind.CONFIG_MISSING)` で fail-fast する（未到達 slot の silent no-op 防止）。
そのぶん **`train` の件数だけ実 API 呼び出しが追加**される（本例では 3 件）。動的 routing で seed
状態では判定できない構成や、この追加コストを避けたい場合は `skip_coverage_check=True` で opt-out
できる。詳細は `docs/adr/0009-lightning-preflight-coverage.md` を参照。

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
    prompt_slot_factory,
    train_val_split,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import api_style, azure_client, azure_deployment, azure_model  # noqa: E402

PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "prompts"
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
    store = PromptStore(PROMPTS_ROOT, LAYOUT)

    # triage と billing のプロンプトのみ系全体で同時最適化する（support は固定）。
    # per-agent で `parts` / `tune` が違うため `prompt_slot_factory` で共通既定値（`store` /
    # `registry` / `base` / `vars`）を束ね、差分だけを `make_slot()` の kwarg として上書きする。
    # 返り値の列は `optimize(slot=)` が `Slot.name` をキーとする mapping へ正規化する。
    make_slot = prompt_slot_factory(
        store, registry, base="main", vars={"company": "AgentSpec Inc."}
    )
    slots = [
        # triage は routing 指針を parts に足し、agent セグメントのみ最適化（tune 省略）。
        make_slot("triage", parts=["style", "routing"]),
        # billing は billing_rules を parts に足し、base 側も同時に最適化する（tune 明示）。
        make_slot("billing", parts=["style", "billing_rules"], tune=["main", "billing"]),
    ]

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
        # APO の gradient / apply-edit 用モデルは rollout と同じものへ明示的に揃える
        # （既定 gpt-5.4-mini はプロバイダ / ゲートウェイによっては存在しないため）。
        apo_gradient_model=azure_deployment(),
        apo_apply_edit_model=azure_deployment(),
        # gradient / apply-edit の API はプロバイダ設定（OPENAI_API_STYLE）に揃えて明示する。
        apo_api=api_style(),
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
