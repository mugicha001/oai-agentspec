"""エージェント単体のプロンプトを APO で最適化する最小例（実 API）。

`optimize` に静的 `AgentSpec` と学習データ・報酬を渡し、`${var}` 保持の最適化済みプロンプトを得る。
`slot` を省略すると静的 `AgentSpec` の `instructions` を既定スロット（最適化対象の seed）として
扱う。データセットは `OptimizeCase`（typed なケース型）で記述すると reward ファクトリ（`contains` /
`tool_match` / `route_match` / `last_agent_match` / `approval_match`）はフィールド名を渡さずに
呼べる（既定 field が `OptimizeCase` の標準フィールド名に揃っている）。

APO は agentlightning の textual gradient + beam search による反復最適化で、内部で gradient 計算と
prompt 編集に LLM を 2 つ追加で使う（既定 gpt-5-mini / gpt-4.1-mini）。利用者は採点 LLM（rollout
実行用）とは別に APO 計算用の AsyncOpenAI 互換クライアントを `apo_client=` で直接渡す（パワー
ユーザーは `config=OptimizeConfig(...)` も使える・両方同時指定はエラー）。検証データ `val` は必須
（APO の beam search が必要）で、`train_val_split` で決定的に分割する。

`algorithm=` は省略可（既定 `"apo"`）。本 extra では `"apo"` のみ受理し、`"rl"` は別 extra
`oai-agentspec[lightning-rl]` で提供される。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/lightning/01_single_agent_apo.py

導入: pip install 'oai-agentspec[lightning]'
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from oai_agentspec import AgentSpec
from oai_agentspec.runtime.lightning import OptimizeCase, contains, optimize, train_val_split

# examples/ 共有ヘルパ _azure を解決するため _shared を import パスへ。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import api_style, azure_client, azure_deployment, azure_model  # noqa: E402


async def main() -> None:
    # 最適化対象（静的 AgentSpec）。instructions が既定スロットの seed になる。
    target = AgentSpec(
        name="jp-router",
        instructions="ユーザーの依頼を分類し、billing / support / other のいずれか1語だけ返す。",
        model=azure_model(),
    )

    # データセットは利用者供給（lib はケースを同梱しない）。`OptimizeCase` を使うと reward は
    # `contains()` のように field 引数を渡さずに呼べる（既定 field=expected_output が読まれる）。
    # dict 経路（例: {"input": "...", "expected": "billing"}）も併存し、その場合は
    # `contains("expected")` のように自由フィールド名を明示する。
    data = [
        OptimizeCase(input="請求書の再発行をお願いします", expected_output="billing"),
        OptimizeCase(input="アプリが起動しません", expected_output="support"),
        OptimizeCase(input="営業時間を教えてください", expected_output="other"),
        OptimizeCase(input="二重に請求されています", expected_output="billing"),
    ]
    # 決定的分割（seed 固定）。自前分割の結果をそのまま train / val に渡してもよい。
    train, val = train_val_split(data, val_ratio=0.25, seed=0)

    # slot 省略 = 静的 AgentSpec の instructions を最適化対象スロットとする。
    # apo_client は直接 kwargs で渡せる（最小ケースの推奨経路・rounds 等もここに並べられる）。
    # APO の gradient / edit モデルは既定 `gpt-5.4-mini`（oai-agentspec 標準）。Azure 利用時は
    # 当該デプロイ名を `apo_gradient_model=` / `apo_apply_edit_model=` で上書きする。
    # E2E 動作確認用に最小 APO 設定（1 ラウンド・1 候補のみ生成）。本番運用では rounds を増やし
    # apo_beam_width / apo_branch_factor を 4 等に上げて多候補から最良を選ぶ。
    result = await optimize(
        target,
        train=train,
        val=val,
        reward=contains(),  # 既定 field=expected_output で OptimizeCase.expected_output を読む。
        apo_client=azure_client(),
        # APO の gradient / apply-edit 用モデルは rollout と同じものへ明示的に揃える
        # （既定 gpt-5.4-mini はプロバイダ / ゲートウェイによっては存在しないため）。
        apo_gradient_model=azure_deployment(),
        apo_apply_edit_model=azure_deployment(),
        # gradient / apply-edit の API はプロバイダ設定（OPENAI_API_STYLE）に揃えて明示する。
        apo_api=api_style(),
        rounds=1,
        apo_beam_width=1,
        apo_branch_factor=1,
    )

    print(f"train_score={result.train_score:.3f} | val_score={result.val_score}")
    print("--- before（rollout 時の合成済み instructions）---")
    print(result.seed)
    print("--- after（最適化済みプロンプト・rollout 実体と一致）---")
    print(result.prompt)
    print("--- diff（unified diff）---")
    print(result.diff or "(no change)")

    # history は各スロット 1 件の HistoryEntry。`placeholder_fallback=True` のとき APO 最良候補が
    # seed の `${var}` を喪失したため seed にフォールバックしたことを示す（公開契約）。利用者は
    # warning に依存せず programmatic に検出できる。
    entry = result.history[0]
    print(
        f"--- history ---  placeholder_fallback={entry['placeholder_fallback']} "
        f"best_score={entry['best_score']} best_version={entry['best_version']}"
    )

    # 結果保存は opt-in。リポジトリを汚さないよう一時ディレクトリへ書き出す（save は任意パス）。
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "optimized_prompt.txt"
        result.save(out)
        print(f"saved (temp): {out} | exists={out.exists()}")


if __name__ == "__main__":
    asyncio.run(main())
