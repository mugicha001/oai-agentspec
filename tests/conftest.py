"""テスト共通設定。

外部 API への通信を構造的に遮断する: (1) トレーシング無効化、(2) OPENAI_API_KEY 削除、
(3) ループバック以外の TCP 接続を即時失敗させるネットワークガード。
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def _no_external_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        from agents import set_tracing_disabled

        set_tracing_disabled(True)
    except Exception:  # pragma: no cover - tracing API 非提供時のベストエフォート
        pass

    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: object) -> object:
        # AF_UNIX や (host, port) 以外（イベントループ self-pipe 等）は許可。
        if isinstance(address, tuple) and address:
            host = address[0]
            if host not in _LOOPBACK:
                raise RuntimeError(f"テスト中の外部接続をブロックしました: {host}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    yield
