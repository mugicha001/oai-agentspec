# MCP サーバの宣言（`AgentSpec.mcp_servers` / `mcp_config`）

MCP（Model Context Protocol）サーバを `AgentSpec` の**専用フィールド**で宣言する例。
`extra` へ SDK kwarg を素通しする必要はなく、型・補完が効き、綴り誤りは build 時に
`ValueError` で検出される。

追加の extra 導入は不要（`mcp` は openai-agents の依存として必ず入る）。

## 宣言

```python
from oai_agentspec import AgentRegistry, AgentSpec

registry.register(
    AgentSpec(
        name="inventory",
        instructions="...",
        model=azure_model(),
        mcp_servers=[server],                                   # kw_only
        mcp_config={"include_server_in_tool_names": False},     # kw_only・省略可
    )
)
```

`mcp_servers` / `mcp_config` は build 時に `agents.Agent` の同名フィールドへ渡る。
`mcp_servers` は空なら kwargs へ積まず、非空なら **コピーして** 渡す（宣言後に list を
mutate しても構築済み `Agent` へ伝播しない）。`mcp_config` は未指定（`None`）なら積まず、
SDK 既定の空 dict に委ねる。

`extra={"mcp_servers": ...}` のように `extra` へ同名キーを積むと、専用フィールドとの衝突として
build 時に `ValueError` で拒否される（どちらが勝つかの曖昧さを作らない）。

## MCP ツールは run 時に解決される

MCP のツールは `spec.tools` に載らない。SDK が **run 時**にサーバへ list_tools して
`FunctionTool` へ変換する（ターンごとに再解決される）。したがって `tools` が空でもモデルは
MCP のツールを呼べる。

```
build 時                        run 時（ターンごと）
--------                        ------------------
spec.tools -> Agent.tools       Agent.get_all_tools()
                                  -> MCPServer.list_tools()
                                  -> MCPUtil.to_function_tool()   <- ここで FunctionTool になる
```

ツールの公開名は既定でサーバ上の名前そのまま（例: `get_stock`）。
`mcp_config={"include_server_in_tool_names": True}` にすると `mcp_{サーバ名}__{ツール名}` を基本形
とする prefix が付くため、名前でツールを参照する仕組み（ポリシー等）を併用している場合は追随が
必要。基本形がそのまま公開名になるのは「ASCII 英数字 / `_` / `-` のみ・長さ上限以内・`spec.tools` /
handoff / as_tool のツール名や同一解決バッチ（同一 agent の全 MCP サーバ）内の他ツールと非衝突」の
場合で、外れると SDK が文字を置換したりハッシュ付きへ切り詰めるため、実際の公開名を確認して
宣言する。サーバ名 / ツール名が置換と strip の結果空になる場合（`--` 等）は `server` / `tool` へ
フォールバックする。

## サーバの接続 / 切断は利用者責務

lib は宣言を素通しするだけで lifecycle を持たない（build-don't-run）。`connect()` /
`cleanup()` は利用者が呼ぶ（`agents.Agent.mcp_servers` の契約と同じ）。本例は
`MCPServerStdio` を async context manager として使い、`with` を抜けるときに切断する。
複数サーバをまとめて扱う場合は `agents.mcp.MCPServerManager` を検討する。

```python
async with MCPServerStdio(name="demo", params={...}) as server:
    registry = build_registry(server, azure_model())
    agent = registry.get("inventory")
    await Runner.run(agent, input="...")
```

## `mcp_config` の注意点

| 項目 | 内容 |
|---|---|
| 未知キー | SDK の `MCPConfig`（`convert_schemas_to_strict` / `include_server_in_tool_names` / `failure_error_function`）に無いキーは検証されず SDK 側で無視される（綴り誤りは silent に効かない） |
| `failure_error_function` | 戻り値は LLM へ渡る。例外原文（接続 URL / トークンを含みうる）をそのまま返さない |
| dict のコピー | build 時に dict をコピーせず参照を渡す。宣言後に mutate すると構築済み `Agent` へ伝播する（registry の `freeze()` は複製するため遮断される） |

MCP サーバのツール定義・ツール出力は信頼境界の外側から model context へ入る。必要に応じて
`oai_agentspec.runtime.guardrails`（`examples/guardrails/`）を併用する。

## サンプル

| ファイル | 内容 |
|---|---|
| `01_declarative_mcp_servers.py` | `mcp_servers` / `mcp_config` の宣言・run 時のツール解決・接続 lifecycle の最小例 |
| `_server.py` | example が起動する最小 MCP サーバ（stdio・in-memory の固定在庫データを返す。ネットワークへ出ない） |

## 実行

Azure OpenAI の環境変数（`AZURE_OPENAI_*`・`examples/_shared/_azure.py` 参照）を `.env` に設定して
実行する。MCP サーバは example が `python examples/mcp/_server.py` として自動起動するため、
別途の準備は不要。

```bash
uv run python examples/mcp/01_declarative_mcp_servers.py
```

詳細仕様は `docs/architecture.md` の `AgentSpec` 節を参照。
