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
| `hooks` | `Any` | `None` | `agents.AgentHooks` |
| `input_guardrails` | `list[Any]` | `[]` | 入力ガードレール（kw_only） |
| `output_guardrails` | `list[Any]` | `[]` | 出力ガードレール（kw_only） |
| `handoffs` | `list[str]` | `[]` | ハンドオフ先エージェント名リスト |
| `handoff_options` | `dict[str, HandoffConfig]` | `{}` | dst 名 -> HandoffConfig |
| `sub_agents` | `list[str]` | `[]` | as_tool 配線するサブエージェント名 |
| `sub_agent_tools` | `dict[str, tuple[str \| None, str \| None]]` | `{}` | サブ名 -> (tool_name, tool_description) |
| `dynamic_handoffs` | `list[DynamicHandoff]` | `[]` | 動的ハンドオフ宣言 |
| `extra` | `dict[str, Any]` | `{}` | 上記以外の `agents.Agent` kwarg 素通し |

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
- 具体例: `examples/basic/basic.py` / `examples/basic/dynamic_context.py` / `examples/basic/runtime_update.py` / `examples/sandbox/`

## 次

[prompts.md](./prompts.md) — PromptStore による合成
