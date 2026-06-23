"""WS メッセージ種別の serve / cli 二重定義の値一致トリップワイヤ。

serve（`serve.protocol` の `WsClientMsg` / `WsServerMsg` StrEnum）と cli（`cli.client` の
`WS_TYPE_*` 定数）は同じ WS メッセージ種別文字列を別々に定義している。両者がズレると
サーバ送信メッセージをクライアントが取りこぼす / 誤判定するため、値一致を担保する。
serve / cli 両 extra（fastapi / httpx 等）導入時のみ実行する。
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("websockets")

from oai_agentspec.runtime.cli import client as cli_client  # noqa: E402
from oai_agentspec.runtime.serve.protocol import WsClientMsg, WsServerMsg  # noqa: E402

pytestmark = pytest.mark.integration


def test_ws_type_values_match_between_serve_and_cli() -> None:
    """serve の WS メッセージ種別値と cli の WS_TYPE_* 定数が一致する。

    serve / cli の WS 種別二重定義のズレを検知するトリップワイヤ。turn / token / done /
    error の各値が両側で同一であることを担保する。
    """
    assert WsClientMsg.TURN.value == cli_client.WS_TYPE_TURN
    assert WsServerMsg.TOKEN.value == cli_client.WS_TYPE_TOKEN
    assert WsServerMsg.DONE.value == cli_client.WS_TYPE_DONE
    assert WsServerMsg.ERROR.value == cli_client.WS_TYPE_ERROR
