"""合成プロンプトを `prompt_slot`（新 shape・複数 tune）で最適化する例（PromptStore 連携・実 API）。

`prompt_slot` は `PromptStore.compose` と同じ構成指定（`agent` / `base` / `parts` / `layout`）を
受け付け、`tune` セレクタで最適化対象セグメントを選べる（新 shape）。ここでは:

    prompt_slot(
        store, registry,
        agent="triage_minimal", base="main_minimal", parts=["style"],
        tune=["main_minimal", "triage_minimal"],   # base と agent の両方を最適化
        vars={"company": "AgentSpec Inc."},
    )

- base（`main_minimal`）と agent（`triage_minimal`）の 2 セグメントを **同時に最適化**
- part（`style`）は固定（tune 対象外・APO 中も不変）
- vars（`${company}`）は最適化対象外で seed に保持され、rollout 時に再注入される

内部的には `Slot.segments` に構成順のセグメント列が確定し、各 tune セグメントは境界マーカー
（`${oas_boundary_N}`）で挟まれた 1 本の候補プロンプトとして APO に渡り、beam の best 候補は
境界マーカーで再分解して各セグメントに書き戻す。境界マーカーの順序・個数が候補側で壊れた場合は
fail-closed で seed へフォールバックする（NFR-3）。

seed は **意図的に弱く**（`triage_minimal` = 「依頼を確認して対応してください」・`main_minimal` =
「あなたは ${company} の担当者です」のみ）保っている。routing 指示や billing / support のツール
選択指示は書かず、APO が rollout の reward シグナルから改善案を textual gradient で生成する。
完成された seed を渡すと APO は改善余地がなく diff が空になる（07 と同じ規範）。

PromptStore は読み取り / 複製経由のみで一切改変しない（依存方向 runtime/lightning -> core を
一方向に保つ）。

APO は内部で追加 LLM（textual gradient + prompt edit）を使うため、利用者は `apo_client=` で
AsyncOpenAI 互換クライアント（ここでは Azure OpenAI）を直接渡す。検証データ `val` は必須。

Azure OpenAI の環境変数を設定して実行:
    uv run python examples/lightning/02_prompt_slot_apo.py

導入: pip install 'oai-agentspec[lightning]'
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from oai_agentspec import AgentRegistry, AgentSpec, PromptLayout, PromptStore
from oai_agentspec.runtime.lightning import (
    OptimizeCase,
    judge,
    optimize,
    prompt_slot,
    train_val_split,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import api_style, azure_client, azure_deployment, azure_model  # noqa: E402

PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "prompts"
LAYOUT = PromptLayout(base="base", parts="parts", agents="agents")


async def main() -> None:
    model = azure_model()

    # 最適化対象 spec を registry に登録。既定 build がこの spec を複製して instructions を候補で
    # 差し替える。spec 名は PromptStore の agent md ファイル名（agents/triage_minimal.md）と一致
    # させる必要がある（既定 build が `agent=` の名前で registry を lookup するため）。
    spec = AgentSpec(
        name="triage_minimal", instructions="(seed は PromptStore から読む)", model=model
    )
    registry = AgentRegistry()
    registry.register(spec)

    store = PromptStore(PROMPTS_ROOT, LAYOUT)

    # 新 shape: 構成 = base:main_minimal + agent:triage_minimal + part:style（compose と同一順）。
    # tune=["main_minimal", "triage_minimal"] で base と agent の両方を最適化対象にし、part:style は
    # 固定（tune 対象外・APO 中も不変）。tune=None にすると agent セグメントのみが最適化対象になる。
    slot = prompt_slot(
        store,
        registry,
        agent="triage_minimal",
        base="main_minimal",
        parts=["style"],
        tune=["main_minimal", "triage_minimal"],
        vars={"company": "AgentSpec Inc."},
    )

    print("--- Slot 構成（compose と同一順） ---")
    for seg in slot.segments:
        mark = "[tune]" if seg.tune else "[fixed]"
        print(f"  {mark} {seg.ref}")

    data = [
        OptimizeCase(input="先月の請求書のPDFが欲しいです", expected_output="billing"),
        OptimizeCase(input="ログインできず困っています", expected_output="support"),
        OptimizeCase(input="解約の手続きを教えてください", expected_output="billing"),
        OptimizeCase(input="今月の請求書を見たい", expected_output="billing"),
        OptimizeCase(input="アプリが起動しない", expected_output="support"),
        OptimizeCase(input="パスワードを忘れた", expected_output="support"),
        OptimizeCase(input="支払い方法を変更したい", expected_output="billing"),
        OptimizeCase(input="エラーメッセージが出る", expected_output="support"),
    ]
    train, val = train_val_split(data, val_ratio=0.25, seed=0)

    result = await optimize(
        spec,
        train=train,
        val=val,
        reward=judge(
            rubric=(
                "出力に expected_output の語（billing / support）が含まれ、依頼が正しく分類できて"
                "いれば高評価。全く分類が行われず一般的な返答のみなら低評価。"
            ),
            model=model,
        ),
        slot=slot,
        registry=registry,
        apo_client=azure_client(),
        # APO の gradient / apply-edit 用モデルは rollout と同じものへ明示的に揃える
        # （既定 gpt-5.4-mini はプロバイダ / ゲートウェイによっては存在しないため）。
        apo_gradient_model=azure_deployment(),
        apo_apply_edit_model=azure_deployment(),
        # gradient / apply-edit の API はプロバイダ設定（OPENAI_API_STYLE）に揃えて明示する。
        apo_api=api_style(),
        # 複数セグメント tune のため 1 ラウンドで各 tune 対象に 1 候補ずつ生成する最小構成。
        # 本番では rounds / beam を増やして候補多様性を確保する。
        rounds=1,
        apo_beam_width=1,
        apo_branch_factor=1,
    )

    print()
    print(f"train_score={result.train_score:.3f}")
    print("--- BEFORE（rollout 時の合成済み本文・${company} は vars 展開済み） ---")
    print(result.seed)
    print()
    print("--- AFTER（最適化済み合成済みプロンプト） ---")
    print(result.prompt)
    print()
    print("--- DIFF（unified diff・part:style は不変・base / agent の tune 部分だけ ± で表示） ---")
    if result.diff:
        print(result.diff)
    else:
        print("(変更なし: APO が seed より良い候補を採用しなかった)")
        print("  rounds / beam を増やすと変化が起きやすい")


if __name__ == "__main__":
    asyncio.run(main())
