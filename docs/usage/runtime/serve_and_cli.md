# serve と cli（クライアント・サーバ型）

## 何を解決するか

`ConversationService` を別プロセスから使いたい場合の入口です。`oai-agentspec serve` は FastAPI + WebSocket のサーバ入口（`[serve]` extra）、`oai-agentspec chat` は httpx / websockets で接続する CLI クライアント（`[cli]` extra）です。dev 用途（localhost・認証なし）を想定しています。

サーバは `ConversationService` へ委譲する薄い接着層であり、認証・TLS・スケール要件が要る場合は自前のサーバから `ConversationService` を呼ぶ形にします。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| in-process `ConversationService` | 同プロセス | 組込・自作 UI |
| `serve` + `cli`（別プロセス） | REST + WS | dev のマルチクライアント |
| serve のみ | 自作クライアントから叩く | 独自 UI 開発 |
| cli のみ（自作サーバへ） | serve と互換の API を自作 | 認証・TLS 要件 |

## 使い方

- import: `from oai_agentspec.runtime.serve import create_app, start_server, DEFAULT_HOST, DEFAULT_PORT`
- コマンド: `oai-agentspec serve --registry <module:attr>` / `oai-agentspec chat`
- extras: `pip install oai-agentspec[serve]` および `pip install oai-agentspec[cli]`
- 依存 env: serve は `XDG_DATA_HOME`（session db 既定位置。未設定時 `~/.local/share/oai-agentspec/`）を参照。cli 本体は env を読まず、接続先は `--url http://localhost:8000`（既定）または `--host` / `--port` で指定する

```python
# 起動側
from oai_agentspec.runtime.serve import start_server
start_server(registry, host="127.0.0.1", port=8000)

# 別プロセスから接続
# $ oai-agentspec chat --host 127.0.0.1 --port 8000
```

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `start_server(registry_or_service, *, host=DEFAULT_HOST, port=DEFAULT_PORT, session_policy=None, entry_agent=None)`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `registry_or_service` | `AgentRegistry \| ConversationService` | 必須 | どちらかを受ける |
| `host` | `str` | `DEFAULT_HOST` (`"127.0.0.1"`) | バインド先 |
| `port` | `int` | `DEFAULT_PORT` (`8000`) | バインド先 |
| `session_policy` | `SessionPolicy \| None` | `None` | registry 渡し時のみ |
| `entry_agent` | `str \| None` | `None` | registry 渡し時のみ |

### `create_app(service)`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `service` | `ConversationService` | 必須 | 委譲先 |

### 定数

`DEFAULT_HOST = "127.0.0.1"` / `DEFAULT_PORT = 8000`。

### CLI コマンド（`[project.scripts]` 由来）

- `oai-agentspec serve --registry <module:attr> [--host 127.0.0.1] [--port 8000] [--session-db PATH]`
- `oai-agentspec chat [--url http://localhost:8000] [--host 127.0.0.1] [--port 8000]` — 別プロセスからサーバへ接続（`--url` 指定時は `--host` / `--port` を上書き）

## 判断軸

- 同プロセスで完結するなら **in-process `ConversationService`**（余分な依存不要）
- dev で複数クライアントから同一 registry を触りたい → **`serve` + `cli`**
- 認証・TLS・スケールが要る本番用途は **本ツールを使わず**、`ConversationService` を自前のサーバに埋め込む

## 落とし穴

- `serve` は localhost・認証なし。**本番運用の想定外**
- env 参照は CLI 境界に閉じる。サーバ / `ConversationService` 本体は env 非依存
- `oai-agentspec chat` は entry（登録順の先頭）エージェント起点。切り替えたい場合はサーバ側 registry 側で調整

## 参照

- 詳細設計: `docs/architecture.md`（会話 Helper 節）
- 具体例: `examples/conversation/03_serve_and_cli.py` / `examples/conversation/05_hitl_serve.py`

## 次

実 API を呼ばずに決定的な応答で動かす方法は [runtime/deterministic.md](./deterministic.md) を参照してください。より深い設計・不変条件は [docs/architecture.md](../../architecture.md) を参照してください。
