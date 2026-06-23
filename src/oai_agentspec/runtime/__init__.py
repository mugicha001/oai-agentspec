"""実行寄り層（runtime）の予約 namespace パッケージ。

会話サービス（`conversation`）・FastAPI サーバ入口（`serve`）・CLI クライアント（`cli`）を
`runtime/` 配下へ集約する。コア（宣言層）は本パッケージへ依存しない単方向依存を保つ（NFR-5）。

本 `__init__` は再エクスポートしない（extra 未導入耐性のため `import oai_agentspec.runtime`
で serve/cli のトップ import を連鎖させない）。利用側は
`from oai_agentspec.runtime.conversation import ...` のようにサブパッケージ公開窓口を
直接参照する。将来の `runtime/<feature>`（例: `runtime/llmops`）も同規約で追加する。
"""

from __future__ import annotations
