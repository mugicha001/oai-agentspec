"""`vars=callable` + `context_factory` で rollout ごとの実行時 context を注入する例（実 API）。

Issue #40 FR-2 (`context_factory`) / FR-3 (`vars=callable`) の end-to-end 動作例。
本番の HandoffGraph は triage が context を確定して後段エージェントに引き継ぐ動線
（例: triage の routing 決定 / ID 抽出）を持つ。本例は最小構成でこれを再現し、`planner` の
プロンプト内 `${routed_from}` / `${tone}` が「rollout ごとに新鮮な context」から動的注入
されつつ、**APO の成果物（`OptimizeResult.prompt`）には `${var}` のまま保持** されることを
示す（具体値がベイクされない・compose(vars=callable) と同一契約）。

構成:
- `HandoffGraph`: triage -> planner / advisor
- `context_factory=lambda: ConversationContext(...)`: rollout 開始時に新鮮な context を生成
- triage: 通常 build（vars=None）
- planner / advisor: `vars=lambda ctx: {"tone": ..., "routed_from": ...}` の callable を渡す。
  既定 build が SDK 動的 Instructions 規約 `(context, agent) -> str` の instructions を据え、
  rollout 時に `vars_fn(context)` を評価して `${tone}` / `${routed_from}` へ注入する

注意（本例のスコープと限界）:
- **本例は `context_factory` の「rollout ごとに新鮮な context」と `vars=callable` の「動的注入」
  のパイプ配線のみを示す最小例**。「triage の tool 呼び出しで context を確定 -> 後段が読む」
  というエンドツーエンドの動線までは示していない。実運用では triage 側で
  `@function_tool` の中で `ctx.context.routed_from = ...` を書き込む形になる（本例では
  factory が固定値をセットしているため `routed_from='triage'` の初期値がそのまま planner で
  読まれる）
- APO は planner / advisor のプロンプトのみ最適化する（triage 側は登録 spec のまま固定）
- Slot は per-agent 構成（個別 `prompt_slot`）で組み立てる。共通 base/parts がないため
  `prompt_slots` の一括生成は使わない
- **`OptimizeResult.diff` は本例では通常空**（単一 slot・短い期待出力・capable LLM の組み合わせで
  reward の S/N が低く、APO 候補は生成されるものの beam 選抜で seed に落ちる）。`${var}` 保持
  や context 注入の動線確認が本例の目的で、安定した diff を見たい場合は 07
  （複数 slot + 複合 reward）を参照

Azure OpenAI の環境変数を設定して実行:
    uv run python examples/lightning/08_dynamic_vars_callable_apo.py

導入: pip install 'oai-agentspec[lightning]'
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph, PromptLayout, PromptStore
from oai_agentspec.runtime.lightning import (
    OptimizeCase,
    contains,
    optimize,
    prompt_slot,
    train_val_split,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_client, azure_model  # noqa: E402


@dataclass
class ConversationContext:
    """rollout ごとに新鮮なインスタンスを生成する会話 context（`context_factory` で作る）。

    Attributes:
        tone: 応対トーン（会話全体で共通・callable が `${tone}` に注入する）。
        routed_from: 直前のハンドオフ元 agent 名（triage 起点なので初期値は "triage"）。
            実運用では triage 側の tool 呼び出しで動的に書き換わる想定だが、本例では初期値
            そのままを planner / advisor が読む形で最小構成を示す。
    """

    tone: str = "polite"
    routed_from: str = "triage"


# 意図的に弱い seed（07 と同じ規範）。`${var}` プレースホルダは APO 中も保持され rollout 時に
# 注入される。APO は routing 指示・応答方針を textual gradient で追加する余地を持つ。
TRIAGE_SEED = (
    "あなたはトリアージ担当です。ユーザーの依頼を読み、手続き系なら planner に、"
    "案内・情報提供系なら advisor にハンドオフしてください。"
)
PLANNER_SEED = "${tone} で応答。ハンドオフ元 ${routed_from}。"
ADVISOR_SEED = "${tone} で応答。ハンドオフ元 ${routed_from}。"


def _make_store(root: Path) -> PromptStore:
    """本例専用の一時 `PromptStore` を組む（agents/ 配下に 3 テンプレート）。"""
    (root / "agents").mkdir()
    (root / "agents" / "triage.md").write_text(TRIAGE_SEED, encoding="utf-8")
    (root / "agents" / "planner.md").write_text(PLANNER_SEED, encoding="utf-8")
    (root / "agents" / "advisor.md").write_text(ADVISOR_SEED, encoding="utf-8")
    return PromptStore(root, PromptLayout(base="base", parts="parts", agents="agents"))


async def main() -> None:
    model = azure_model()
    registry = AgentRegistry()
    registry.register(AgentSpec(name="triage", instructions="(seed)", model=model))
    registry.register(AgentSpec(name="planner", instructions="(seed)", model=model))
    registry.register(AgentSpec(name="advisor", instructions="(seed)", model=model))

    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "planner", description="手続き系")
    graph.edge("triage", "advisor", description="案内・情報提供系")
    graph.apply(registry)
    registry.validate()

    with tempfile.TemporaryDirectory() as td:
        store = _make_store(Path(td))

        # planner / advisor に vars=callable を渡す（rollout ごとに context から動的注入される）。
        # 既定 build は SDK 規約 `(context, agent) -> str` の動的 instructions を据え、rollout 時に
        # `vars_fn(context)` を評価して `${tone}` / `${routed_from}` へ注入する。APO 候補は
        # 内部で `${var}` 保持のまま扱われ、`OptimizeResult.prompt` にも保持される。
        def _dyn_vars(ctx: object) -> dict[str, object]:
            """`RunContextWrapper.context` から `ConversationContext` を取り出す。"""
            return {
                "tone": ctx.context.tone,  # type: ignore[attr-defined]
                "routed_from": ctx.context.routed_from,  # type: ignore[attr-defined]
            }

        slots = {
            "planner": prompt_slot(store, registry, agent="planner", vars=_dyn_vars),
            "advisor": prompt_slot(store, registry, agent="advisor", vars=_dyn_vars),
        }

        data = [
            OptimizeCase(input="請求書のPDFが欲しい", expected_output="請求"),
            OptimizeCase(input="料金プランを教えて", expected_output="プラン"),
            OptimizeCase(input="今月分の明細ください", expected_output="明細"),
            OptimizeCase(input="法人プランはある？", expected_output="法人"),
        ]
        train, val = train_val_split(data, val_ratio=0.5, seed=0)

        # contains() で expected_output の語を判定する（seed に指示が薄いため APO が routing 指示や
        # 応答方針を textual gradient で追加する余地を持つ・07 の設計と同じ規範）。
        reward = contains()

        result = await optimize(
            graph,
            train=train,
            val=val,
            reward=reward,
            slot=slots,
            registry=registry,
            # rollout ごとに新鮮な context を生成（rollout 間で状態を共有しない）。
            # 承認 resume ループ内は SDK `RunState` 内包の同一 context を再利用する。
            context_factory=lambda: ConversationContext(tone="polite", routed_from="triage"),
            apo_client=azure_client(),
            # E2E 動作確認用の最小 APO 設定。本番では rounds / beam を増やす。
            rounds=1,
            apo_beam_width=1,
            apo_branch_factor=1,
        )

        print(f"train_score={result.train_score:.3f}")
        for slot_name in ("planner", "advisor"):
            print()
            print(f"=== [{slot_name}] ===")
            print("--- BEFORE (seed・${var} 保持) ---")
            seed = result.seed if isinstance(result.seed, str) else result.seed[slot_name]
            print(seed)
            print("--- AFTER (最適化済み・${var} 保持) ---")
            prompt = result.prompt if isinstance(result.prompt, str) else result.prompt[slot_name]
            print(prompt)
            print("--- DIFF (unified) ---")
            diff = (
                result.diff if isinstance(result.diff, str) else (result.diff or {}).get(slot_name)
            )
            print(diff or "(変更なし: APO が seed より良い候補を採用しなかった)")
        # ${tone} / ${routed_from} は AFTER にも literal で残る（APO は具体値をベイクしない）。
        # 実 rollout では ctx.context.tone / ctx.context.routed_from の値が注入される。


if __name__ == "__main__":
    asyncio.run(main())
