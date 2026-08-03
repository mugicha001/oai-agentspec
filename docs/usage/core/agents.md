# AgentSpec と AgentRegistry

## 何を解決するか

`agents.Agent` は kwarg が多く、フィールド名のタイポや専用フィールドとの二重指定が実行時まで発覚しにくいです。`AgentSpec` は `Agent` の薄い宣言的ラッパーで、専用フィールド + `extra` の二段構えでミスを build 時に弾きます。

`AgentRegistry` は複数の spec を名前空間に登録し、依存（`handoffs` ∪ `sub_agents` ∪ 動的候補）を遅延解決して `Agent` を構築します。循環参照や未解決参照は `validate()` で run 前に検出できます。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| `AgentSpec` | 通常の Agent 宣言（handoffs / sub_agents / extra 対応） | 標準ユースケース |
| `SandboxAgentSpec` | shell / code interpreter を最小権限で使う派生宣言 | サンドボックス実行が必要 |
| 静的 `instructions` | 文字列 or 合成済みプロンプト | 内容が run で変わらない |
| 動的 `instructions`（`(context, agent) -> str`） | 2 引数 callable | tenant / user 状況で切り替えたい |

## 使い方

- import: `from oai_agentspec import AgentSpec, AgentRegistry, SandboxAgentSpec, HandoffConfig, RegistryFrozenError`
- extras: なし
- 依存 env: なし

```python
from oai_agentspec import AgentRegistry, AgentSpec

registry = AgentRegistry()
registry.register(AgentSpec(
    name="triage",
    instructions="受付担当",
    handoffs=["billing"],
    sub_agents=["researcher"],
    extra={"handoff_description": "振り分け担当"},  # 未対応キーは extra へ
))
registry.register(AgentSpec(name="billing", instructions="請求担当"))
registry.register(AgentSpec(name="researcher", instructions="調査担当"))
registry.validate()
agent = registry.get("triage")  # 依存解決して agents.Agent を構築
```

動的 instructions は `PromptStore.compose(vars=callable)` に `RunContextWrapper -> dict` を渡します（[prompts](./prompts.md)）。

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `AgentSpec`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `name` | `str` | 必須 | エージェント名（registry 内で一意） |
| `instructions` | `str \| Callable[..., Any] \| None` | `None` | 静的文字列または `(context, agent) -> str` |
| `prompt` | `Any` | `None` | `agents.Prompt` / `DynamicPromptFunction`（Responses API 用） |
| `tools` | `list[Any]` | `[]` | SDK Tool のリスト |
| `model` | `Any` | `None` | モデル指定（str / `agents.Model` / None） |
| `model_settings` | `Any` | `None` | `agents.ModelSettings` |
| `hooks` | `Any` | `None` | `agents.AgentHooks`（複数宣言は `chain_agent_hooks` で合成。下記参照） |
| `input_guardrails` | `list[Any]` | `[]` | 入力ガードレール（kw_only） |
| `output_guardrails` | `list[Any]` | `[]` | 出力ガードレール（kw_only） |
| `handoffs` | `list[str]` | `[]` | ハンドオフ先エージェント名リスト |
| `handoff_options` | `dict[str, HandoffConfig]` | `{}` | dst 名 -> HandoffConfig |
| `sub_agents` | `list[str]` | `[]` | as_tool 配線するサブエージェント名 |
| `sub_agent_tools` | `dict[str, tuple[str \| None, str \| None]]` | `{}` | サブ名 -> (tool_name, tool_description) |
| `dynamic_handoffs` | `list[DynamicHandoff]` | `[]` | 動的ハンドオフ宣言 |
| `extra` | `dict[str, Any]` | `{}` | 上記以外の `agents.Agent` kwarg 素通し |

#### `hooks` を複数宣言する（`chain_agent_hooks`）

`hooks` は単一スロットのため、エージェント単位のフックを複数持たせる場合は
`chain_agent_hooks` で 1 つに合成して渡します（追加 extra は不要）。

```python
from typing import Any

from agents.lifecycle import AgentHooksBase

from oai_agentspec import AgentSpec
from oai_agentspec.runtime.hooks import chain_agent_hooks

class MetricsHooks(AgentHooksBase[Any, Any]):     # 全 on_* を持つ通常の実装
    async def on_start(self, context, agent): ...
    async def on_end(self, context, agent, output): ...

class ToolLogger:                                  # 部分実装（継承せず 2 メソッドのみ）
    async def on_tool_start(self, context, agent, tool): ...
    async def on_tool_end(self, context, agent, tool, result): ...

enable_audit = False                               # 条件付きで有効化するフックの例

spec = AgentSpec(
    name="triage",
    instructions="...",
    hooks=chain_agent_hooks(
        MetricsHooks(),
        MetricsHooks() if enable_audit else None,   # 無効時は None のまま渡せる
        ToolLogger(),                                # 部分実装も混在可
    ),
)
```

- 宣言順（左から右）に順次 await します。前段が例外を送出したら後段は呼ばれず、その例外が
  そのまま伝播します（fail-fast）。
- `None` は無視されるため、条件付きで有効化するフックを分岐なしに列挙できます。
- run 単位フック（`agents.lifecycle.RunHooksBase` のインスタンス）は渡せません。渡すと build 時に
  `TypeError` になります（メソッド名が異なり `on_handoff` の引数意味も違うため）。run 単位の合成は
  `chain_hooks` を使ってください（[resilience](../safety/resilience.md)）。
- `AgentHooksBase` を継承するフックの `on_*` は `async` で定義してください。合成ラッパ経由では
  同期メソッドも呼び出せますが、実効 1 件かつ `AgentHooksBase` インスタンスの場合はそのフック
  自身が返って正規化を経ないため、同期メソッドは SDK 側の await で `TypeError` になります
  （同期許容は部分実装のための緩さで、フック件数に依存しない保証ではありません）。
- `on_*` を 1 つも持たないオブジェクトも渡せません（build 時に `TypeError`）。包むと全メソッドが
  no-op になり、フックが 1 度も発火しないのに例外が出ないためです。`*` の付け忘れ
  （`chain_agent_hooks(my_hooks_list)`）や型違いがこれで検知されます。
- 要素は `agents.lifecycle.AgentHooksBase` 継承クラスのインスタンスに限らず、`on_*` の一部だけを持つ
  オブジェクト（部分実装）も渡せます。持たないメソッドはスキップされます。
- `None` を除いた実効件数が 0 件なら全メソッド no-op のフック、1 件で
  `agents.lifecycle.AgentHooksBase` のインスタンスならその要素自身が返り、余分なラッパは
  作られません（型判定に使えるのは非ジェネリックな基底 `AgentHooksBase` です。添字付き
  エイリアス `agents.AgentHooks` は `isinstance` の第 2 引数に使えません）。
- 戻り値は `AgentSpec(hooks=...)` へそのまま渡せます（`agents.Agent(hooks=...)` へ素通し）。
- run 単位（`Runner.run(hooks=...)`）の合成は `chain_hooks` です（[resilience](../safety/resilience.md)）。
  メソッド名が `on_start` / `on_end` 対 `on_agent_start` / `on_agent_end` で異なるため、両者は
  互換ではありません。

### `SandboxAgentSpec`（`AgentSpec` 継承 + 4 kw_only フィールド）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `default_manifest` | `Any` | `None` | `SandboxAgent.default_manifest`（未指定は SDK 既定） |
| `capabilities` | `Any` | `None` | `SandboxAgent.capabilities`（未指定は SDK 既定・最小権限は明示必須） |
| `run_as` | `Any` | `None` | `SandboxAgent.run_as` |
| `base_instructions` | `str \| Callable[..., Any] \| None` | `None` | サンドボックス用ベースプロンプト |

### `AgentRegistry`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `agent_builder` | `AgentBuilder \| None` | `None` | Agent 構築の Protocol 実装。省略時は `_adapters` の既定 |

### `HandoffConfig`（frozen）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `description` | `str \| None` | `None` | `handoff(tool_description_override=...)` |
| `tool_name` | `str \| None` | `None` | `handoff(tool_name_override=...)` |
| `on_handoff` | `Any` | `None` | 発火時コールバック |
| `input_type` | `Any` | `None` | 転送時に LLM が埋める構造化入力の型 |
| `input_filter` | `Any` | `None` | 次エージェントへ渡す履歴の変換 |
| `is_enabled` | `Any` | `True` | 動的有効化（bool or callable） |
| `options` | `dict[str, Any]` | `{}` | 素通し用 kwarg |

## 判断軸

- 静的な役割宣言で足りるなら **`AgentSpec`** を使い、tenant / user で挙動を分けたい場合のみ **動的 instructions** を導入する
- shell / code interpreter を触るなら通常の `AgentSpec` ではなく **`SandboxAgentSpec`** で最小権限を宣言する
- 実行時に spec を差し替えたい場合は `registry.update(new_spec)` / `registry.unregister(name)`。依存元は自動で無効化される

## 落とし穴

- `extra` に専用フィールドと同名キーを入れると `ValueError`（二重指定防止）
- `extra` の未知キー（`output_typ` 等のタイポ）は `ValueError` で弾く
- `registry.freeze()` 後に register / update すると `RegistryFrozenError`（`lockdown(root, registry=registry)` は内部で `freeze()` を呼ぶ）

## 参照

- 詳細設計: `docs/architecture.md`（AgentSpec / SandboxAgentSpec 節）
- 具体例: `examples/basic/basic.py` / `examples/basic/dynamic_context.py` / `examples/basic/runtime_update.py` / `examples/sandbox/` / `examples/hooks/01_chain_agent_hooks.py`（`hooks` の複数宣言）/ `examples/hooks/02_chain_hooks.py`（run 単位との非対称）
- 設計判断: `docs/adr/0016-agent-hooks-chain-helper.md`（`chain_agent_hooks`）/ `docs/adr/0017-reject-run-hooks-in-chain-agent-hooks.md`（run 単位フックの拒否）

## 次

[prompts.md](./prompts.md) — PromptStore による合成
