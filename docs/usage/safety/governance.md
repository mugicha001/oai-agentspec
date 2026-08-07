# ガバナンス（ツール単位ポリシー + 監査ログ）

## 何を解決するか

エージェントが「何をできるか」をツール単位のポリシーで許可 / 拒否し、決定を監査ログに残す仕組みです。`GovernedAgentBuilder` を `AgentRegistry(agent_builder=...)` に注入すると、registry の遅延構築経路を通る全 spec の tools が govern ラップされ、監査 `AgentHooks` が装着されます。`AgentSpec` / `tools` / `AgentBuilder` Protocol の宣言面は不変です。

外部依存の Agent Governance Toolkit（AGT）に委譲します。ポリシー違反時は AGT 由来の `PolicyViolationError` が送出されます。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| 素の `AgentRegistry` | ポリシーなし | 検証・PoC・単体テスト |
| `GovernedAgentBuilder(policy=...)` DI | 全 spec に一括で policy + 監査 | 本番・監査要件がある |
| `GovernedAgentBuilder(policy=..., overrides={...})` | エージェント別に policy 上書き | agent 毎に権限を差し替えたい |
| `GovernedAgentBuilder.from_yaml(path)` | bundle YAML から一括構築 | 制限を宣言ファイルに一元化 |

## 使い方

- import: `from oai_agentspec.runtime.governance import GovernedAgentBuilder`
- extras: `pip install oai-agentspec[governance]`（`agent-governance-toolkit[openai-agents]`）
- 依存 env: AGT が要求する env（audit sink 等）

```python
from oai_agentspec import AgentRegistry
from oai_agentspec.runtime.governance import GovernedAgentBuilder

builder = GovernedAgentBuilder(policy="policy.yaml", audit_sink=my_sink)
registry = AgentRegistry(agent_builder=builder)
# 以降 register / get は通常通り。tools は自動で govern ラップされる
```

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `GovernedAgentBuilder.__init__`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `policy` | `str \| os.PathLike[str] \| object` | 必須（kw_only） | YAML パスまたは AGT ポリシーオブジェクト |
| `audit_sink` | `object \| None` | `None` | 監査ログ出力先。None で初回 build 時に AGT 既定 sink を生成 |
| `inner` | `AgentBuilder \| None` | `None` | 装飾対象。None で `DefaultAgentBuilder` |
| `overrides` | `Mapping[str, str \| os.PathLike[str] \| object] \| None` | `None` | エージェント名 -> ポリシー上書き |

### `GovernedAgentBuilder.from_yaml(path, *, audit_sink=None, inner=None)`

bundle YAML（`default` + `agents`）からの構築。3 引数以下のためコメントで列挙。

### 主要プロパティ

- `unapplied_overrides: frozenset[str]` — 一度も適用されていない override キー（typo 検知）
- `audit_sink: object | None` — 現在の監査 sink

## 強制点は 2 つ（宣言は共通）

同じ `allowed_tools` / `blocked_patterns` が両方に効きます（ポリシー宣言の規約は 1 本のまま）。

| 対象 | 評価位置 | 理由 |
|---|---|---|
| `spec.tools` の `FunctionTool` | build 時に実行本体（`on_invoke_tool`）を差し替え | tool オブジェクトが build 時に存在する |
| `spec.mcp_servers` 経由の MCP ツール | 装着した監査 `AgentHooks.on_tool_start` | SDK が **run 時**（ターンごと）にサーバへ list_tools して `FunctionTool` を生成するため、build 時にラップ対象が無い |

MCP を使う場合も利用者の記述は変わりません（builder を注入し、`allowed_tools` に MCP ツールの公開名を書くだけ）。実行例は `examples/governance/05_mcp_tool_governance.py` を参照してください。

## 判断軸

- 監査要件・ポリシー強制が要るなら **`GovernedAgentBuilder` DI** を registry に注入する。宣言面（`AgentSpec`）は触らない
- policy を差し替えたい場合は builder ごと入れ替える。spec 側に policy を書かない
- extra 未導入時は `build()` 内部の遅延 import で `ImportError`（案内付き）。fail-fast 設計

## 落とし穴

- AGT の import は関数内遅延。窓口 import 自体は extra 未導入でも壊れないが、`build()` 実行時にアクセスで例外
- `sub_agents` の as_tool・`register_factory` 経路は govern 対象外
- clone された registry は builder を共有する（監査チェーンが混ざる）。系を分けたい場合は builder を別々に注入
- **hosted MCP**（Responses API のサーバ側 MCP・`HostedMCPTool`）はモデルプロバイダ側で実行されるため評価も監査も発生しない。統治されるのは client-side MCP（`spec.mcp_servers`）のみ。`RealtimeAgentSpec` の `mcp_servers` も別 builder 経路のため対象外
- **MCP の deny は利用者の `spec.hooks.on_tool_start` へ到達しない**（`spec.tools` の deny では到達する）。利用者フックで監査・計測している場合は観測が欠ける
- MCP の deny は run を `UserError` で終了させる。MCP ツール自身の実行時例外が `mcp_config["failure_error_function"]` でモデルへ返り会話が継続するのとは挙動が違う
- `mcp_config={"include_server_in_tool_names": True}` にするとツールの公開名が `mcp_{サーバ名}__{ツール名}` になるため、`allowed_tools` の宣言も追随が必要（未対応なら全 deny になり安全側で顕在化する）
- `allowed_tools` は名前照合で、MCP ツールの実体はターンごとに再解決される。同名のまま schema / 意味だけ差し替える変更は検知しない
- 監査の `details` には MCP ツールの引数も全文記録される（URL / 接続情報が入りうるため `audit_sink` の永続先を考慮する）

## 参照

- 詳細設計: `docs/architecture.md`（AGT ガバナンス節）
- 検討経緯: `docs/rationale/agt-governance-integration.md`
- 具体例: `examples/governance/01_policy_enforcement.py` 〜 `04_policy_bundle.py`

## 次

[integrity.md](./integrity.md) — lockdown と manifest 検証
