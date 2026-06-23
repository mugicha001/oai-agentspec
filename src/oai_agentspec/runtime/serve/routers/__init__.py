"""serve のルーティング層（REST / WebSocket・serve extra・agents 非依存）。

REST ルータ構築（`rest`）と WebSocket ルート登録（`ws`）を責務別サブモジュールへ分割する。
app factory（`serve.app.create_app`）がここから router を取り込み登録する。
"""

from __future__ import annotations
