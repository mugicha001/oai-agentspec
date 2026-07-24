"""APO の「全部入り」例: `OptimizeCase` + 複合 reward + 系全体最適化 + vars 再注入（実 API）。

llmops `EvalCase` が `input` / `expected_output` / `expected_tools` / `expected_route` /
`expected_approvals` を 1 ケースに集約するのと同じ発想で、APO のデータセットも `OptimizeCase`
で**期待回答・期待ツール・期待ルート・期待最終 agent**を 1 ケースに集約する。`OptimizeCase` の
標準フィールド名は reward ファクトリの既定値（`expected_output` / `expected_tools` /
`expected_route` / `expected_last_agent` / `expected_approvals`）に揃っているため、reward は
**フィールド名を渡さずに**呼べる（`reward=contains()` で十分）。

reward は個別ファクトリ（`contains` / `tool_match` / `route_match` / `last_agent_match`）を
AND（または重み付き平均）で合成すれば、「正しい担当へ handoff し、期待ツールを呼び、期待回答に
近い応答を返す」プロンプトを 1 つの reward で APO 学習できる。

設計上の重要ポイント:
  - **APO の改善余地を確保するため、seed は意図的に弱く保つ**: triage の routing 指示や billing /
    support のツール選択指示を seed には書かず、APO が rollout の reward シグナルから「billing
    系は billing にハンドオフして lookup_invoice を呼ぶ」「support 系は support にハンドオフして
    restart_service を呼ぶ」を学習する。完成された seed を渡すと APO は何も改善できず diff が
    空になる。
  - **複合 reward の各観点が最適化対象に「効く」配置**: 系全体（triage / billing / support）を
    一括で最適化対象にする（`contains` は billing/support の応答内容、`tool_match` は billing/
    support のツール選択、`route_match` と `last_agent_match` は triage の routing 判断）。
  - **vars 再注入**: `${company}` などの vars は `Slot.vars` に保持され、最適化対象外（不変）の
    まま rollout 時に再注入される（APO 候補が `${company}` を喪失した場合は fail-closed で 0.0）。
  - PromptStore は本例では使わない（02 / 03 で十分示している）。`Slot` を手書きすることで「seed /
    build / vars」のミニマル構成を見せる。

`OptimizeResult` の出力（手書き `Slot` は `segments` 空 = 合成なし・run_apo 返却を素通し）:
  - `result.seed`  : 各 slot の seed テキスト（before・rollout 実体と一致・`${company}` は vars
                     展開済み・APO 候補生成は内部で `${var}` 保持のまま扱う）
  - `result.prompt`: 各 slot の最適化済みテキスト（after・rollout 実体と一致・vars 展開済み）
  - `result.diff`  : unified diff（変更箇所だけ ± で表示・空文字なら「変更なし」）

`RolloutResult` は以下の plain 観測を持つ:
    case               入力ケース（`OptimizeCase` または利用者定義 dict）
    output             rollout の最終出力テキスト（`contains` / `exact` / `judge` が使う）
    tool_calls         観測したツール呼び出し名の列（`tool_match` が使う）
    fired_approvals    中断時に承認ゲートが発火したツール名の列（`approval_match` が使う）
    route_steps        実行経路（起点を含む agent 名の列・`route_match` が使う）
    last_agent         最終応答した agent 名（`last_agent_match` が使う）

NOTE: APO は内部で追加 LLM（textual gradient + prompt edit）を使うため、利用者は `apo_client=`
で AsyncOpenAI 互換クライアントを直接渡す。検証データ `val` は必須。

Azure OpenAI の環境変数を設定して実行:
    uv run python examples/lightning/07_composite_reward_apo.py

導入: pip install 'oai-agentspec[lightning]'
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
from pathlib import Path

from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph, function_tool
from oai_agentspec.runtime.lightning import (
    OptimizeCase,
    RolloutResult,
    Slot,
    contains,
    last_agent_match,
    optimize,
    route_match,
    tool_match,
    train_val_split,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_client, azure_model  # noqa: E402

# rollout 時の vars（`${company}` プレースホルダに展開）。最適化対象外で全 slot 共通。
VARS = {"company": "AgentSpec Inc."}

# **意図的に弱い** seed: routing 指示なし / tool 指示なし。`${company}` だけ保持。APO が rollout
# の reward シグナルから「billing 系は billing にハンドオフして lookup_invoice を呼ぶ」等の
# 改善案を textual gradient で生成して候補に追加する。
WEAK_SEEDS: dict[str, str] = {
    "triage": (
        "あなたは ${company} のお客様対応窓口です。ユーザーの依頼を確認し、丁寧に対応してください。"
    ),
    "billing": "あなたは ${company} の請求担当です。簡潔に応答してください。",
    "support": "あなたは ${company} のサポート担当です。簡潔に応答してください。",
}


# 各担当が使うダミーツール（副作用なし・承認不要・最適化中も実行されてよい）。tool_match の観測
# 対象として「billing は lookup_invoice を、support は restart_service を呼ぶ」を学習させる。
@function_tool
def lookup_invoice(invoice_id: str) -> str:
    """指定 ID の請求書を参照する（billing 担当が利用するダミーツール・例示用）。"""
    return f"invoice {invoice_id}: status=ok, amount=1000"


@function_tool
def restart_service(service: str) -> str:
    """指定サービスを再起動する（support 担当が利用するダミーツール・例示用）。"""
    return f"service {service} restarted"


def build_registry(model: object) -> tuple[AgentRegistry, HandoffGraph]:
    """triage / billing / support を登録し triage 起点のハンドオフ系を組む。

    各 spec の `instructions` は placeholder（`Slot.build` が登録 spec を複製して instructions を
    候補で差し替えるため、ここでの値は使われない）。billing には `lookup_invoice`、support には
    `restart_service` を持たせる（APO が tool 呼び出しを学習する対象）。"""
    registry = AgentRegistry()
    registry.register(AgentSpec(name="triage", instructions="(seed は Slot から渡る)", model=model))
    registry.register(
        AgentSpec(
            name="billing",
            instructions="(seed は Slot から渡る)",
            tools=[lookup_invoice],
            model=model,
        )
    )
    registry.register(
        AgentSpec(
            name="support",
            instructions="(seed は Slot から渡る)",
            tools=[restart_service],
            model=model,
        )
    )

    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "billing", description="請求関連")
    graph.edge("triage", "support", description="技術問い合わせ")
    graph.apply(registry)
    registry.validate()
    return registry, graph


def composite_reward(result: RolloutResult) -> float:
    """4 観点の AND（平均）合成 reward。

    各 reward ファクトリは `OptimizeCase` の標準フィールド名（`expected_output` /
    `expected_tools` / `expected_route` / `expected_last_agent`）を既定値に持つため、フィールド名
    を渡さずに呼べる。各観点が 1.0 / 0.0 を返し、その平均（0.0..1.0）を最終 reward とする。
    利用者は重み付け / `all([...])` のような AND 厳密合成など自由に設計してよい。
    """
    scores = [
        contains()(result),  # expected_output（billing/support の応答テキストに期待語が含まれるか）
        tool_match()(result),  # expected_tools（billing/support が期待ツールを呼んだか）
        route_match()(result),  # expected_route（triage → billing/support の経路が正しいか）
        last_agent_match()(result),  # expected_last_agent（最終応答が想定担当か）
    ]
    return sum(scores) / len(scores)


async def main() -> None:
    model = azure_model()
    registry, graph = build_registry(model)

    # 系全体（triage / billing / support）を順次最適化する。手書き `Slot` を 3 個構築:
    #   - seed: WEAK_SEEDS の弱い prompt（`${company}` 保持・APO の改善余地を確保）
    #   - build: 登録 spec を複製して instructions を候補で差し替える（tools / handoffs / model
    #     は複製で保持・利用者 registry の登録 spec は不変）
    #   - vars: 全 slot 共通の `${company}` を rollout 時に再注入
    # APO 0.3 は単一プロンプト最適化のため、ライブラリが triage → billing → support の順で
    # 各 slot を APO に通し、最良候補で次の slot の rollout コンテキストを更新する。
    def _make_slot(name: str) -> Slot:
        return Slot(
            name=name,
            seed=WEAK_SEEDS[name],
            build=lambda candidate, _name=name: dataclasses.replace(
                registry._specs[_name],  # noqa: SLF001 - example の最小化のため
                instructions=candidate,
            ),
            vars=VARS,
        )

    slots = {name: _make_slot(name) for name in ("triage", "billing", "support")}

    # 総合データセット: 1 `OptimizeCase` に input + 期待回答 + 期待ツール + 期待ルート + 期待
    # 最終 agent を集約する。reward は各 OptimizeCase フィールドを既定で参照する（フィールド名
    # の指定は不要）。各観点が「最適化対象のどのプロンプトに効くか」が明確（contains は billing/
    # support の応答内容、tool_match は billing/support のツール選択、route_match と
    # last_agent_match は triage の routing 判断）。
    data = [
        OptimizeCase(
            input="請求書 INV-001 の返金について教えてほしい",
            expected_output="返金",
            expected_tools=["lookup_invoice"],
            expected_route=["triage", "billing"],
            expected_last_agent="billing",
        ),
        OptimizeCase(
            input="アプリが起動しません",
            expected_output="サポート",
            expected_tools=["restart_service"],
            expected_route=["triage", "support"],
            expected_last_agent="support",
        ),
        OptimizeCase(
            input="請求 INV-002 の二重請求を調べてほしい",
            expected_output="請求",
            expected_tools=["lookup_invoice"],
            expected_route=["triage", "billing"],
            expected_last_agent="billing",
        ),
        OptimizeCase(
            input="ログインエラーが直りません",
            expected_output="サポート",
            expected_tools=["restart_service"],
            expected_route=["triage", "support"],
            expected_last_agent="support",
        ),
    ]
    train, val = train_val_split(data, val_ratio=0.25, seed=0)

    result = await optimize(
        graph,
        train=train,
        val=val,
        reward=composite_reward,
        slot=slots,
        registry=registry,  # グラフ最適化では registry 必須（未指定は CONFIG_MISSING）。
        apo_client=azure_client(),
        # E2E 動作確認用に最小 APO 設定（1 ラウンド・1 候補）。本番では rounds / beam を増やす。
        rounds=1,
        apo_beam_width=1,
        apo_branch_factor=1,
    )

    print(f"train_score={result.train_score:.3f} | val_score={result.val_score}")

    # 各 slot の変更点（diff）を全文表示する。`(変更なし)` は APO が seed より良い候補を採用
    # しなかったことを意味する（rounds / beam を増やすと変化が起きやすい）。
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
