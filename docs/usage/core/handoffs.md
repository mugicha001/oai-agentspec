# HandoffGraph（宣言的ハンドオフ）

## 何を解決するか

エージェント間の転送を SDK の `handoff()` で直接組むと、宣言箇所が spec に散らばり、循環（A⇄B）解決や未解決参照の検出が難しくなります。`HandoffGraph` は名前ベースで edge を宣言し、`apply()` で registry に反映、`validate()` で run 前にタイポと未解決参照を検出、`mermaid()` で可視化します。

固定 1 ターゲットの静的 edge に加え、実行時に候補から転送先を選ぶ動的 edge（`dynamic_edge`）も同じ宣言面で扱えます。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| `graph.edge(src, dst, ...)` | 静的 1:1 転送 | 転送先が設計時に決まる |
| `graph.dynamic_edge(src, candidates, resolver, tool_name=...)` | resolver 関数が候補内から実行時選択 | 入力に応じて候補内で分岐 |
| 循環 edge（A → B / B → A） | 遅延構築で相互解決 | 状態遷移的な会話ループ |

より広い（handoff vs agent-as-tool vs WorkflowGraph の）使い分けは [multi_agent](./multi_agent.md) を参照。ハンドオフ後の次ターン開始エージェントを宣言で上書きする（到達時ハンドオフ禁止を含む）方法は [next_turn](./next_turn.md) を参照。

## 使い方

- import: `from oai_agentspec import HandoffGraph, HandoffEdge, from_specs`
- extras: なし
- 依存 env: なし

```python
from oai_agentspec import HandoffGraph

graph = HandoffGraph(entry="triage")
graph.edge("triage", "billing", description="請求関連")
graph.edge("triage", "support", description="技術問い合わせ")
graph.dynamic_edge(
    "triage", ["billing", "support"],
    resolver=lambda ctx, inp: "billing" if "invoice" in inp else "support",
    tool_name="route", description="動的に担当を決定",
)
graph.apply(registry)
registry.validate()
print(graph.mermaid())
```

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `HandoffGraph`（dataclass・宣言のみ）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `entry` | `str \| None` | `None` | エントリエージェント名 |
| `edges` | `list[HandoffEdge]` | `[]` | 静的エッジ蓄積用 |
| `dynamic` | `list[DynamicHandoffEdge]` | `[]` | 動的エッジ蓄積用 |

### `HandoffGraph.edge(src, dst, description=None, *, ...)`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `src` | `str` | 必須 | ハンドオフ元名 |
| `dst` | `str` | 必須 | ハンドオフ先名 |
| `description` | `str \| None` | `None` | tool 説明・mermaid ラベル |
| `tool_name` | `str \| None` | `None` | ハンドオフ tool 名（kw_only） |
| `on_handoff` | `Any` | `None` | 発火時コールバック |
| `input_type` | `Any` | `None` | 構造化入力の型 |
| `input_filter` | `Any` | `None` | 履歴フィルタ |
| `is_enabled` | `Any` | `True` | 動的有効化 |
| `options` | `dict[str, Any] \| None` | `None` | 素通し用 kwarg |

### `HandoffGraph.dynamic_edge(src, candidates, resolver, *, tool_name, ...)`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `src` | `str` | 必須 | ハンドオフ元名 |
| `candidates` | `Iterable[str]` | 必須 | 転送先候補名 |
| `resolver` | `Callable[..., Any]` | 必須 | `(ctx, input_json) -> 候補名` |
| `tool_name` | `str` | 必須（kw_only） | ハンドオフ tool 名 |
| `description` | `str \| None` | `None` | tool 説明 |
| `on_handoff` / `input_type` / `input_filter` / `is_enabled` / `options` | 上表と同じ | 上表と同じ | 静的 `edge` と同意 |

### `from_specs(specs, entry=None)`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `specs` | `Iterable[Any]` | 必須 | AgentSpec 群 |
| `entry` | `str \| None` | `None` | エントリエージェント名 |

## 判断軸

- 転送先が入力に依存しないなら **静的 edge**、資源判定・feature flag 等で条件付けたい場合のみ **dynamic_edge**
- 相互参照（循環）は宣言可能。SDK では循環参照が構築時エラーになるが、本 lib は遅延解決で許容する
- `description` は tool 説明（LLM の選択材料）+ `mermaid()` のラベル。ターゲット側の既定説明 `Agent.handoff_description` とは別物

## 落とし穴

- `dynamic_edge` の resolver 戻り値は candidates 名リスト内に強制される（不一致は run 時例外）
- `graph.apply(registry)` は当該 src を replace で上書きする。差分適用ではない

## 参照

- 詳細設計: `docs/architecture.md`（循環ハンドオフ節）
- 検討経緯: `docs/rationale/handoff-cycle-resolution.md`
- 具体例: `examples/basic/cyclic_handoff.py` / `examples/basic/dynamic_edge.py`

## 次

[multi_agent.md](./multi_agent.md) — オーケストレーション手段の選び方
