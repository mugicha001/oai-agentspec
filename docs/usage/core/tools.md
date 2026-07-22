# ToolRegistry と HITL 宣言

## 何を解決するか

`agents.function_tool` を直接呼ぶと、tool メタデータ（`enabled` / `needs_approval` / `timeout` / `failure_error_function` 等）が呼び出し箇所に散らばり、複数 agent で共有する場合に一貫性維持が難しくなります。`ToolRegistry` は `ToolSpec` で宣言的にメタデータを一元管理し、属性アクセス（`registry.<name>`）で `function_tool` を遅延構築・キャッシュします。

`needs_approval=True` を宣言したツールは HITL（Human-In-The-Loop）対象になり、実行前に承認待ちとなります（承認フロー実行は [runtime/conversation](../runtime/conversation.md)）。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| `ToolRegistry` + `ToolSpec` | 宣言的メタデータ + 遅延構築 | 複数 agent で共有・feature flag 制御・メタデータを一箇所で見たい |
| 直接 `function_tool` | 関数デコレータ | 1 箇所でしか使わない・メタデータ最小 |
| `needs_approval=True` | HITL 対象化 | 危険・不可逆・高コスト操作 |
| `needs_approval=None`（既定） | SDK 既定に委譲 | 副作用のない参照系 |

## 使い方

- import: `from oai_agentspec import ToolRegistry, ToolSpec`
- extras: なし
- 依存 env: なし

```python
from oai_agentspec import AgentSpec, ToolRegistry, ToolSpec

def search(q: str) -> str:
    return f"result: {q}"

def delete_user(uid: str) -> str:
    return "deleted"

tools = ToolRegistry()
tools.register(ToolSpec(func=search))  # name 省略で func.__name__ を採用
tools.register(ToolSpec(func=delete_user, needs_approval=True, timeout=30))

spec = AgentSpec(name="assistant", instructions="...", tools=[tools.search, tools.delete_user])
```

`ToolSpec.enabled` は属性代入で動的にトグル可能（再構築不要）。feature flag 用途に使えます。

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `ToolSpec`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `func` | `Any` | 必須（第 1 引数） | Tool 実体（sync/async callable） |
| `name` | `str \| None` | `None` | 登録キー。省略時は `func.__name__` |
| `enabled` | `bool` | `True` | 有効フラグ（動的トグル可） |
| `needs_approval` | `Any` | `None` | HITL 承認要否（bool / callable） |
| `timeout` | `float \| None` | `None` | Tool 実行タイムアウト秒 |
| `timeout_behavior` | `str \| None` | `None` | SDK `timeout_behavior` |
| `timeout_error_function` | `Any` | `None` | SDK `timeout_error_function` |
| `failure_error_function` | `Any` | `TOOL_UNSET`（センチネル） | 未指定と None 明示を区別する 3 値 |
| `name_override` | `str \| None` | `None` | SDK `name_override`（Registry キーとは独立） |
| `description_override` | `str \| None` | `None` | SDK `description_override` |
| `strict_mode` | `bool \| None` | `None` | SDK `strict_mode`（None で kwarg 未渡し） |
| `extra` | `dict[str, Any]` | `{}` | `agents.function_tool` へ素通しする追加 kwarg |

### `ToolRegistry`

引数なし（`__init__()`）。`register(spec)` / `names()` / `metadata(name)` メソッドを提供し、`registry.<name>` の属性アクセスで `FunctionTool` を返す。

## 判断軸

- 副作用のあるツールは既定で **`needs_approval=True`**、参照系のみ False にする
- 1 モジュール内でしか使わない tool は直接 `function_tool` で足りるが、agent 間共有 or feature flag 要件があるなら **`ToolRegistry`** に集約する

## 落とし穴

- `ToolRegistry.<name>` は属性アクセス。`registry["search"]` ではない
- `needs_approval=True` の tool は `ConversationService.send` の戻り値が `pending` になる。承認 API を呼ばないと会話が進まない
- `ToolSpec` の第 1 引数は `func`（`fn` ではない）

## 参照

- 詳細設計: `docs/architecture.md`（Tool Registry 節）
- 設計判断: `docs/adr/0001-tool-metadata-centralization.md`
- 具体例: `examples/tool_registry/`

## 次

[handoffs.md](./handoffs.md) — HandoffGraph と dynamic_edge
