# Realtime エージェント（音声）

## 何を解決するか

`agents.realtime.RealtimeAgent` を対象とする専用の宣言ルートです。`RealtimeAgentSpec` は RealtimeAgent が対応するフィールドのみを持ち、非対応フィールド（`model` / `model_settings` / `input_guardrails` / `sub_agents` / `sub_agent_tools` / `dynamic_handoffs` / `output_type` / `tool_use_behavior`）を型レベルで排除します。専用の `RealtimeAgentRegistry` が名前ベース handoff を遅延構築（循環も解決）します。

シンボルはコア `__all__` に載せず、`oai_agentspec.realtime` 窓口から取得します（通常 Agent 側と混同しないための分離）。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| 通常 `AgentSpec` | テキスト応答・tool 実行 | 一般的なチャット・タスク |
| `RealtimeAgentSpec` | 低レイテンシ音声 I/O 前提 | 音声チャット・電話応対 |
| `spec.handoffs` 直接宣言 | 名前ベース handoff | 少数エッジ |
| `RealtimeHandoffGraph` | ノード / エッジ宣言 + mermaid | 多エッジ・可視化が要る |

## 使い方

- import: `from oai_agentspec.realtime import RealtimeAgentSpec, RealtimeAgentRegistry, RealtimeHandoffGraph, RealtimeHandoffConfig, RealtimeHandoffEdge, from_specs`
- extras: なし（`agents.realtime` は SDK 側）
- 依存 env: Realtime Model 接続に必要な env

```python
from agents.realtime import RealtimeRunner
from oai_agentspec.realtime import RealtimeAgentRegistry, RealtimeAgentSpec

registry = RealtimeAgentRegistry()
registry.register(RealtimeAgentSpec(name="triage", instructions="受付", handoffs=["support"]))
registry.register(RealtimeAgentSpec(name="support", instructions="技術サポート"))
registry.validate()

entry = registry.get("triage")
runner = RealtimeRunner(entry, config={"model_settings": {"model_name": "gpt-4o-realtime-preview"}})
```

`model_settings`（`model_name` / `voice` / `modalities` 等）は宣言側が持たない実行時 Config。セッション開始時に `RealtimeRunner` へ渡します。

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `RealtimeAgentSpec`（dataclass）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `name` | `str` | 必須 | エージェント名 |
| `instructions` | `str \| Callable[..., Any] \| None` | `None` | 静的または `(context, agent) -> str` |
| `prompt` | `Any` | `None` | `agents.Prompt \| None`（callable 不可・build 時 `ValueError`） |
| `tools` | `list[Any]` | `[]` | Tool リスト |
| `hooks` | `Any` | `None` | `agents.realtime` のフック型 |
| `handoff_description` | `str \| None` | `None` | ハンドオフ先としての説明 |
| `mcp_servers` | `list[Any]` | `[]` | MCP サーバ |
| `mcp_config` | `Any` | `None` | MCP 設定 |
| `output_guardrails` | `list[Any]` | `[]` | 出力ガードレール（kw_only） |
| `handoffs` | `list[str]` | `[]` | ハンドオフ先名 |
| `handoff_options` | `dict[str, RealtimeHandoffConfig]` | `{}` | dst 名 -> RealtimeHandoffConfig |
| `extra` | `dict[str, Any]` | `{}` | 素通し用 |

### `RealtimeHandoffConfig`（frozen）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `on_handoff` | `Any` | `None` | 発火時コールバック |
| `input_type` | `Any` | `None` | 構造化入力の型 |
| `tool_name_override` | `str \| None` | `None` | ハンドオフ tool 名 |
| `tool_description_override` | `str \| None` | `None` | ハンドオフ tool の説明 |
| `is_enabled` | `Any` | `True` | 動的有効化 |

### `RealtimeAgentRegistry.__init__(agent_builder=None)`

1 引数（`RealtimeAgentBuilder | None`）。省略で `_adapters.realtime.DefaultRealtimeAgentBuilder`。

### `RealtimeHandoffGraph`

`entry: str | None = None` / `edges: list[RealtimeHandoffEdge] = []`。宣言メソッドは `edge(src, dst, *, on_handoff=None, input_type=None, tool_name=None, tool_description=None, is_enabled=True)` / `extend(iter)` / `apply(specs)` / `entry_agent(registry)` / `mermaid()`。

### `from_specs(specs, entry=None)`

2 引数。

## 判断軸

- 音声 I/O が要件でないなら **通常 `AgentSpec`** を使う（Realtime は Config 制約が強い）
- handoff エッジ数が少なく可視化不要なら **`spec.handoffs`** 直接、多エッジや mermaid が要るなら **`RealtimeHandoffGraph`**

## 落とし穴

- `RealtimeAgentSpec` に `model` / `model_settings` / `input_guardrails` は書けない（型レベル排除）。実行時 Config は `RealtimeRunner` 側
- コア `AgentRegistry` と `RealtimeAgentRegistry` は別物。混在させない
- `RealtimeAgentSpec.prompt` に callable を渡すと build 時 `ValueError`

## 参照

- 詳細設計: `docs/architecture.md`（Realtime 節）
- 具体例: `examples/realtime/basic_declaration.py` / `handoff_session.py` / `voice_chat.py`

## 次

[conversation.md](./conversation.md) — 会話 Helper と HITL
