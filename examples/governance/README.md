# AGT ガバナンス（oai-agentspec[governance]）の使い方

宣言したエージェントが「何をできるか」をツール単位のポリシーで許可 / 拒否し、許可 / 拒否を監査
ログへ記録する支援層。`AgentRegistry(agent_builder=GovernedAgentBuilder(policy=...))` の
**builder 差し替え 1 行**とポリシー定義だけで後付けでき、`AgentSpec` / `tools` の宣言面は不変。

「何をできるか」を制御する AGT ガバナンスと、「何を言うか」を検査する内容ガードレール
（`oai-agentspec[guardrails]`・`examples/guardrails/`）は直交する役割分担であり、相互に置き換え
ない。lib 内部状態 / ディスク改竄の検知（`lockdown`・`examples/integrity/`）とも補完関係にある
（起動時の一括ゲート vs ツール呼び出しごとの実行時強制。`docs/integrity.md` 参照）。

## インストール（extra）

```bash
pip install 'oai-agentspec[governance]'   # agent-governance-toolkit[openai-agents] を取り込む
```

依存は AGT（MIT）の base / core / integrations と `structlog` のみ（grpc / azure / opa / torch 等の
重い推移依存なし）。extra 未導入でも `import oai_agentspec` と
`oai_agentspec.runtime.governance` の import は壊れず、`build` 時に導入を案内する
`ImportError` になる。

## 結線の仕組み（build 時・実行は SDK Runner）

`GovernedAgentBuilder` は `AgentBuilder` Protocol を満たす装飾 builder で、registry の遅延構築
（唯一の構築経路）に差し込まれる。build 時に各 `FunctionTool` の実行本体（`on_invoke_tool`）を
ポリシー評価付きラップへ非破壊置換し、ライフサイクル監査の `AgentHooks` を装着する。ポリシー
評価・監査記録が動くのは実行時（SDK `Runner` がツールを呼ぶ直前）で、lib は実行エンジンを
持たない（build-don't-run）。

```python
from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.governance import GovernedAgentBuilder

registry = AgentRegistry(
    agent_builder=GovernedAgentBuilder(policy="policies/support.yaml"),
)
registry.register(AgentSpec(name="support", tools=[refund, lookup_order]))  # 宣言面は不変
agent = registry.get("support")  # 各 tool が govern 済み・監査フック装着済み
```

- `spec.hooks` に利用者フックがあっても上書きされず、「監査記録 -> 既存フックへ委譲」の順で
  合成される。
- `policy` は YAML ファイルパスでも、構築済みの AGT `GovernancePolicy` オブジェクトでもよい。

## per-agent ポリシー（overrides）

エージェントごとにポリシーを出し分ける場合は `overrides`（エージェント名 -> ポリシー）を渡す。
掲載エージェントはそのポリシー、未掲載は `policy`（既定）へフォールバックする（`spec.name` との
完全一致で引き当て・正規化なし）。同一ツールでもエージェントによって allow / deny を分けられる。

```python
builder = GovernedAgentBuilder(
    policy="policies/readonly.yaml",          # 既定（基準線）
    overrides={"support": "policies/support.yaml"},  # support だけ上書き
)
```

- overrides の値は `policy` と同形式（YAML パス / ポリシーオブジェクト）で、同一の fail-fast
  検証を受ける。`None` は不正値（既定へ戻す意図はキーの削除で表現する）。
- キーの typo は黙って既定へフォールバックするため、全エージェント build 後に
  `builder.unapplied_overrides` が空集合であることを確認する（typo 検知の opt-in 手段）。
- 監査 sink は overrides を使っても builder で 1 本共有される。

### bundle YAML（制限の全量を 1 ファイルへ）

既定 / per-agent の制限を単一ファイルに宣言したい場合は bundle YAML + `from_yaml` を使う。
制限定義が YAML（ポリシー本体）とコード（エージェント名との対応付け）に分離せず、
1 ファイルの監査で「どのエージェントが何をできるか」を把握できる。

```yaml
# governance.yaml
default:                  # 必須（overrides 未掲載の全エージェントへ適用）
  allowed_tools: [lookup_order]
agents:                   # 任意（エージェント名 -> ポリシー）
  support:
    allowed_tools: [lookup_order, refund]
```

```python
builder = GovernedAgentBuilder.from_yaml("governance.yaml")  # コード側は参照 1 行
```

bundle は構築糖衣であり、通常コンストラクタで同内容を組んだ場合と等価に動く（fail-fast 検証・
`unapplied_overrides`・sink 共有も同一）。コードでポリシーオブジェクトを組む形式・YAML パスを
個別に渡す形式・bundle の 3 形式はどれでも使え、既定 / per-agent とも自由に選べる。

## 拒否例外の捕捉

拒否例外 `PolicyViolationError` は公開窓口から取得する（AGT 内部パッケージからの import や
DeprecationWarning 抑制は不要）。SDK `Runner` 経由では SDK 例外にラップされ得るため `__cause__`
も確認する。

```python
from oai_agentspec.runtime.governance import PolicyViolationError
```

## 強制範囲

| ポリシーフィールド | 強制内容 | いつ動く |
|---|---|---|
| `allowed_tools` | ツール名 allowlist。未掲載ツールの呼び出しを拒否 | ツール実行直前（LLM の tool call 後・実関数の実行前） |
| `blocked_patterns` | ツール引数 JSON への正規表現照合。生のワイヤ文字列と JSON 正規化文字列（`\uXXXX` 等のエスケープ別表現を展開した形）の両方に適用 | 同上 |

違反時は実関数を実行せず AGT `PolicyViolationError` を送出する。SDK `Runner` 経由では SDK
例外にラップされ得るため、捕捉時は `__cause__` も確認する（`01_policy_enforcement.py` 参照）。
拒否で run は中断されるため、拒否されたツールの `tool_end` 以降の監査記録は残らない（deny
レコードが終端）。

YAML の未知キー（`allowed_tool:` のような typo）は読み込み時に `ValueError` で拒否される
（allowlist が黙って無効化され全ツール許可に化ける事故の防止）。`max_tool_calls` 等の本統合で
強制されない `GovernancePolicy` フィールドを指定した場合は `RuntimeWarning` で警告される。

## 監査ログ

監査は 2 系統が同じ sink に記録される。

- **決定記録（tool 単位）**: 「どのエージェントが・どのツールを・どの引数で呼び・許可 / 拒否
  されたか」。`details` には**ツール引数 JSON が全文記録される**ため、機密引数を扱う場合は
  sink の永続先を考慮して選定する。
- **ライフサイクル監査（hooks）**: `agent_start` / `tool_start` / `tool_end` / `handoff` /
  `agent_end`。

既定 sink（AGT `AuditLog`・tamper-evident ハッシュチェーン）は builder が初回 build で生成して
以降の build と共有するため、複数エージェントでも 1 本のチェーンになる。
`GovernedAgentBuilder.audit_sink` プロパティで取得し、`get_entries()` / `verify_chain()` で参照・
検証する。`audit_sink=` に `record(agent_id, action, decision, details=None)` を持つ任意
オブジェクトを DI して出力先を差し替えられる（env 参照は持たない・引数 DI のみ）。

## 強制点は 2 つ（`spec.tools` と MCP）

`spec.tools` の `FunctionTool` は build 時に実行本体（`on_invoke_tool`）をラップして評価する。
`spec.mcp_servers` 経由の MCP ツールは SDK が **run 時**（ターンごと）に解決するため build 時の
ラップ対象が存在せず、装着した `AgentHooks.on_tool_start` で評価する。宣言は同じ
`allowed_tools` / `blocked_patterns` で足り、ポリシーの規約は 1 本のまま
（`05_mcp_tool_governance.py` 参照）。

MCP 経路の非対称（利用者が観測しうる差）:

- MCP の deny は `on_tool_start` からの送出で合成チェーンを中断するため、**利用者の
  `spec.hooks.on_tool_start` へ到達しない**（`spec.tools` の deny は実行本体のラップで弾くため
  到達する）。利用者フックで監査・計測している場合は観測が欠ける。
- MCP の deny は run を `UserError` で終了させる。MCP ツール自身の実行時例外が
  `mcp_config["failure_error_function"]` でモデルへ文字列返却され会話が継続するのとは挙動が違う。
- `tool:` レコードの `agent_id` は宣言時の `spec.name`、`tool_start:` は runtime の `agent.name`。
- `allowed_tools` は名前照合で、MCP ツールの実体はターンごとに再解決される。同名のまま
  schema / 意味だけ差し替える変更は検知しない（`mcp_config["include_server_in_tool_names"]` で
  サーバ単位に名前空間を分けると識別しやすい。真にした場合は `allowed_tools` の宣言も
  `mcp_{サーバ名}__{ツール名}` へ追随する）。

## 既知の境界（govern 対象外）

- `sub_agents` の as_tool は registry が build 後に注入するため、per-call の allow / deny 評価・
  決定記録の対象外（hooks の `tool_start` / `tool_end` 記録のみ。サブエージェント自身の内部
  `FunctionTool` は同 builder 経由で govern 済み）。
- **hosted MCP**（Responses API のサーバ側 MCP・`HostedMCPTool`）はモデルプロバイダ側で実行され
  `on_tool_start` が発火しないため、評価も監査も発生しない。統治されるのは client-side MCP
  （`spec.mcp_servers`）のみ。`RealtimeAgentSpec` の `mcp_servers` も別 builder 経路のため対象外。
- `register_factory` 経路は builder を通らないため govern 対象外。
- hosted tool 等の非 `FunctionTool` は素通し（ポリシー強制境界は関数ツールの呼び出し）。
- SDK の HITL 承認（`needs_approval`）はツール実行前の承認フローとして govern ラップより先に
  走るため、ポリシーが拒否する呼び出しでも承認要求は先に発生し得る（承認後に deny される）。
- 既定監査 sink・override 適用記録は **builder インスタンス単位**。`AgentRegistry.clone()` は
  builder を共有するため、系（本番 / 評価等）を分けたい場合は builder を registry ごとに分ける。

## サンプル

| ファイル | 内容 |
|---|---|
| `01_policy_enforcement.py` | YAML ポリシーで allowlist / blocked_patterns を強制する最小例。許可 / 拒否 / 引数内容での拒否と、監査ログの確認 |
| `02_audit_log.py` | 複数エージェントでの既定 sink 共有（1 本のチェーン）・`audit_sink` プロパティ・`verify_chain()`・利用者 hooks との合成。ポリシーはオブジェクト形 |
| `03_per_agent_policy.py` | per-agent ポリシー（overrides）。同一ツールのエージェント別 allow / deny の出し分け・既定へのフォールバック・`unapplied_overrides` での typo 検知。ポリシーはコード（オブジェクト）形 |
| `04_policy_bundle.py` | bundle YAML（`from_yaml`）。既定 + per-agent の制限を 1 ファイルに宣言し、コード側は参照 1 行 |
| `05_mcp_tool_governance.py` | MCP サーバ経由のツールにも同じ `allowed_tools` を効かせる。run 時解決ツールの allow / deny と `tool:{name}` 監査（同梱の最小 MCP サーバ `examples/mcp/_server.py` を stdio で自動起動） |
| `policies/support.yaml` | 単一ポリシー定義の例（`01_policy_enforcement.py` が読む） |
| `policies/governance.yaml` | bundle 定義の例（`04_policy_bundle.py` が読む） |

## 実行

Azure OpenAI の環境変数（`AZURE_OPENAI_*`・`examples/_shared/_azure.py` 参照）を `.env` に設定して
実行する:

```bash
uv run python examples/governance/01_policy_enforcement.py
uv run python examples/governance/02_audit_log.py
```

詳細仕様は `docs/architecture.md` の「AGT ガバナンス」節、検討経緯は
`docs/rationale/agt-governance-integration.md` を参照。
