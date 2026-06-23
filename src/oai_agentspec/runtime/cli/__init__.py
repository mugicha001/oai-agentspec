"""会話 CLI クライアント（cli extra・agents 非依存・別プロセス）。

起動中の会話サーバへ HTTP/WS で接続する CLI。`main` のみを公開する。httpx /
websockets は cli extra のため、依存を import する `client` / `chat` は本 `__init__` から
トップ import しない（`main` 内で遅延 import）。これにより cli extra 未導入でも
`import oai_agentspec.runtime.cli` 自体は壊れない。SDK（`agents`）は import しない（NFR-1）。
"""

from __future__ import annotations

from .main import main

__all__ = ["main"]
