"""決定的応答モデルの公開窓口（extra 不要・追加依存なし・再エクスポート専用）。

実 API を呼ばず「入力からルール関数が応答を決める」ステートレス SDK `Model` 実装
（`DeterministicResponseModel`）・ルール関数の引数型（`ModelRequest`）・ルール関数が返す
応答を組み立てる純関数ビルダ 5 種を再エクスポートする。`AgentSpec(model=...)` /
`agents.Agent(model=...)` へ入れれば、ネットワークへ到達できない環境でも `Runner.run` /
`Runner.run_streamed` が完走する（自動テスト・オフライン開発・デモ実行・決定的なシナリオ
再生の 4 用途を想定し、テスト専用機能ではない）。

本窓口は実装本体を持たず、SDK 結合（`agents` / `openai`）は `_adapters/deterministic.py` に
閉じる（NFR-1）。実行 API は持たず、公開するのはモデル宣言と応答オブジェクト構築のみ
（build-don't-run）。コア `__init__` の `__all__` には載せない（コア `__all__` は宣言層
シンボルのみという原則に従う）。
"""

from __future__ import annotations

from ..._adapters.deterministic import (
    DeterministicResponseModel,
    ModelRequest,
    mixed_response,
    multi_tool_call_response,
    text_response,
    text_response_with_usage,
    tool_call_response,
)

__all__ = [
    # 決定的応答モデル（ルール関数を受け取るステートレス SDK Model）
    "DeterministicResponseModel",
    # ルール関数の引数型（frozen・型注釈とエディタ補完のため公開する）
    "ModelRequest",
    # 応答ビルダ（純関数）
    "text_response",
    "text_response_with_usage",
    "tool_call_response",
    "multi_tool_call_response",
    "mixed_response",
]
