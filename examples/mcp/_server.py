"""example 用の最小 MCP サーバ（stdio）。

`examples/mcp/01_declarative_mcp_servers.py` が `MCPServerStdio` から
`python examples/mcp/_server.py` として起動する。外部パッケージの追加導入は不要
（`mcp` は openai-agents の依存として必ず入る）。

ネットワークへ出ず、in-memory の固定データだけを返す（example の再現性を保つため）。
本ファイルは MCP サーバ側の実装例であり、oai-agentspec の宣言面とは無関係
（利用者が自前 / OSS の MCP サーバを持つ場合はそちらを `mcp_servers` へ渡す）。

直接起動して疎通確認もできる（stdio なので手動での対話には向かない）:
    uv run python examples/mcp/_server.py
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-inventory")

_STOCK = {"SKU-1": 12, "SKU-2": 0, "SKU-3": 340}


@mcp.tool()
def get_stock(sku: str) -> str:
    """在庫数を返す（MCP サーバ側のツール）。

    Args:
        sku: 在庫を調べる SKU。

    Returns:
        在庫数の説明文（未登録 SKU は「未登録」を返す）。
    """
    if sku not in _STOCK:
        return f"{sku}: 未登録の SKU です"
    return f"{sku}: 在庫 {_STOCK[sku]} 個"


@mcp.tool()
def list_skus() -> str:
    """登録済み SKU の一覧を返す（MCP サーバ側のツール）。

    Returns:
        カンマ区切りの SKU 一覧。
    """
    return ", ".join(sorted(_STOCK))


if __name__ == "__main__":
    mcp.run(transport="stdio")
