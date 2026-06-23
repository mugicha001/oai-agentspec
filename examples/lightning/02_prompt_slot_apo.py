"""合成プロンプトを `prompt_slot` で最適化する例（PromptStore 連携・実 API）。

`prompt_slot` は `PromptStore` の公開メソッドを**読み取るのみ**で seed（`${var}` 未展開・
プレースホルダ保持）を取得し、既定 build（registry 登録 `AgentSpec` を複製して `instructions`
だけ候補で差し替え）を内包した `Slot` を返す。`build` から rebind が自動導出されるため、利用者は
手書きの rebind を書かなくてよい。`vars`（ここでは `${company}`）は最適化対象外で seed に保持され、
rollout 時に内部で再注入される。

PromptStore は読み取り / 複製経由のみで一切改変しない（依存方向 runtime/lightning -> core を一方向に
保つ）。本例では `examples/prompts/agents` を store の root に据え、各エージェントの本文
テンプレートを `store.get(name)` でフラット解決している。

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
from _azure import azure_client, azure_model  # noqa: E402

# agents/ を root にすると各本文テンプレートが store.get(name) でフラット解決できる。
PROMPTS_AGENTS = Path(__file__).resolve().parent.parent / "prompts" / "agents"
LAYOUT = PromptLayout(base="base", parts="parts", agents="agents")


async def main() -> None:
    model = azure_model()

    # 最適化対象 triage を registry に登録（既定 build がこの spec を複製して instructions を
    # 差し替える）。tools / handoffs / model は登録 spec から複製され、利用者は再宣言しなくてよい。
    spec = AgentSpec(name="triage", instructions="(seed は PromptStore から読む)", model=model)
    registry = AgentRegistry()
    registry.register(spec)

    store = PromptStore(PROMPTS_AGENTS, LAYOUT)
    # seed = agents/triage.md（${company} 保持）。vars は最適化対象外で rollout 時に再注入される。
    slot = prompt_slot(store, registry, tune="triage", vars={"company": "AgentSpec Inc."})

    data = [
        OptimizeCase(input="先月の請求書のPDFが欲しいです"),
        OptimizeCase(input="ログインできず困っています"),
        OptimizeCase(input="解約の手続きを教えてください"),
        OptimizeCase(input="今月の請求書を見たい"),
    ]
    train, val = train_val_split(data, val_ratio=0.25, seed=0)

    result = await optimize(
        spec,  # 第1引数は常に最適化対象。スロットは slot= キーワードで渡す。
        train=train,
        val=val,
        reward=judge(
            rubric="出力が依頼を適切な担当（billing / support）へ正しく案内できているか。",
            model=model,
        ),
        slot=slot,
        registry=registry,
        apo_client=azure_client(),
        # E2E 動作確認用に最小 APO 設定（1 ラウンド・1 候補）。本番では rounds / beam を増やす。
        rounds=1,
        apo_beam_width=1,
        apo_branch_factor=1,
    )

    print(f"train_score={result.train_score:.3f}")
    print("--- before（rollout 時の合成済み triage 本文・${company} は vars 展開済み）---")
    print(result.seed)
    print("--- after（最適化済み合成済みプロンプト）---")
    print(result.prompt)
    print("--- diff（unified diff・base/parts は不変・tune 部分だけ ± で表示）---")
    print(result.diff or "(no change)")


if __name__ == "__main__":
    asyncio.run(main())
