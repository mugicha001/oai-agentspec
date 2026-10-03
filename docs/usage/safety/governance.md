# ガバナンス（ツール単位ポリシー + 監査ログ）

## 何を解決するか

エージェントが「何をできるか」をツール単位のポリシーで許可 / 拒否し、決定を監査ログに残す仕組みです。`GovernedAgentBuilder` を `AgentRegistry(agent_builder=...)` に注入すると、registry の遅延構築経路を通る全 spec の tools が govern ラップされ、監査 `AgentHooks` が装着されます。`AgentSpec` / `tools` / `AgentBuilder` Protocol の宣言面は不変です。

外部依存の Agent Governance Toolkit（AGT）に委譲します。ポリシー違反時は AGT 由来の `PolicyViolationError` が送出され、SDK `Runner` 経由では SDK の `UserError` にラップされて着地します。`UserError` にはツール実行中の他の例外も包まれて届き、`__cause__` が None のこともあるため、`isinstance(exc.__cause__, PolicyViolationError)` を確認してから `exc.__cause__.details.get("tool_name")` で拒否されたツール名を取得し、違反でなければ再送出します（ラップ前の経路で `PolicyViolationError` を直接捕捉する場合は `exc.details.get("tool_name")`）。具体形は `examples/governance/01_policy_enforcement.py` を参照してください。

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

### `GovernedAgentBuilder.post_processor(*, sub_agent_tools=False, factory_agents=False)`

`sub_agents` の as_tool と `register_factory` の Agent を統治対象に加えるオプトイン。戻り値を `AgentRegistry` の `post_processor` 引数へ渡したときだけ、registry が構築・結線の完了後に統治する（渡さなければ統治は build 時だけで完結する）。ポリシー・監査 sink・override の適用記録は生成元の builder と共有する。

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `sub_agent_tools` | `bool` | `False` | True で registry が注入した `sub_agents` の as_tool を親エージェントのポリシーで評価・記録する（親の `allowed_tools` に as_tool の公開名を書く） |
| `factory_agents` | `bool` | `False` | True で `register_factory` の Agent を clone して統治する（factory 名で override を引き当てる・`registry.get` は clone を返す） |

両方 `False` は `ValueError`。

```python
builder = GovernedAgentBuilder(policy="policy.yaml", audit_sink=my_sink)
registry = AgentRegistry(
    agent_builder=builder,  # 装飾 builder で包んでもよい
    post_processor=builder.post_processor(sub_agent_tools=True, factory_agents=True),
)
```

### `mcp_governance_guardrail()`

MCP サーバのツール入力ガードレールとして統治を評価するオブジェクトを返す（引数なし・戻り値は SDK のツール入力ガードレールで不透明値として扱う）。`MCPServer` のコンストラクタの `tool_input_guardrails` の先頭に渡す。ガードレールの name は固定の `"mcp_governance_guardrail"`。import は `from oai_agentspec.runtime.governance import mcp_governance_guardrail`。使い方は「MCP サーバに統治ガードレールを付ける」を参照。

## 強制点は 2 つ（宣言は共通）

同じ `allowed_tools` / `blocked_patterns` が両方に効きます（ポリシー宣言の規約は 1 本のまま）。

| 対象 | 評価位置 | 理由 |
|---|---|---|
| `spec.tools` の `FunctionTool` | build 時に実行本体（`on_invoke_tool`）を差し替え | tool オブジェクトが build 時に存在する |
| `spec.mcp_servers` 経由の MCP ツール | 統治ガードレール（`mcp_governance_guardrail()`）を付けたサーバでは、ツール入力ガードレールの位置（先頭）。付けないサーバでは、装着した監査 `AgentHooks.on_tool_start` | SDK が **run 時**（ターンごと）にサーバへ list_tools して `FunctionTool` を生成するため、build 時にラップ対象が無い。サーバのコンストラクタに渡したガードレールは、SDK が生成する全ツールへ付く |

MCP を使う場合も利用者の記述は変わりません（builder を注入し、`allowed_tools` に MCP ツールの公開名を書くだけ）。実行例は `examples/governance/05_mcp_tool_governance.py` を参照してください。

### MCP サーバに統治ガードレールを付ける

内容検査のガードレール（`tool_guardrail` 等）を MCP サーバに併用する場合は、統治ガードレールを `tool_input_guardrails` の**先頭**に置きます。入力ガードレールは先頭から順に評価され、最初の reject / 送出で止まるため、統治を先に置くと次が成り立ちます。

- ポリシー違反は内容検査の結果に関わらず監査の `tool:` deny に残る
- deny された呼び出しでは内容検査の検知器が呼ばれない

```python
from agents.mcp import MCPServerStdio
from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.governance import GovernedAgentBuilder, mcp_governance_guardrail
from oai_agentspec.runtime.guardrails import tool_guardrail

governed = GovernedAgentBuilder(policy="policy.yaml", audit_sink=my_sink)
server = MCPServerStdio(
    params=...,
    tool_input_guardrails=[mcp_governance_guardrail(), tool_guardrail(d_in, on="input")],  # 統治を先頭に
    tool_output_guardrails=[tool_guardrail(d_out, on="output")],
)
registry = AgentRegistry(agent_builder=governed)
registry.register(AgentSpec(name="bot", instructions="...", mcp_servers=[server]))
```

統治ガードレールは、呼び出したエージェント自身に装着された監査フックのポリシー・監査 sink・`spec.name` で評価します。同じサーバを複数のエージェント（別の builder のエージェントを含む）で共有しても、各エージェントのポリシーで評価され、未統治のエージェントは素通しします。付け忘れたサーバ・評価できなかった呼び出しは、従来どおり `on_tool_start` で評価されます（lib の監査フックがエージェントの hooks に装着されている限り統治は外れません。build 後に hooks を差し替える場合の境界は「落とし穴」の `Agent.hooks` の差し替えの項を参照してください）。内容検査と出力の redact の付け方は [guardrails.md](./guardrails.md) の「MCP サーバ単位で掛ける」を参照してください。

## 判断軸

- 監査要件・ポリシー強制が要るなら **`GovernedAgentBuilder` DI** を registry に注入する。宣言面（`AgentSpec`）は触らない
- policy を差し替えたい場合は builder ごと入れ替える。spec 側に policy を書かない
- extra 未導入時は `build()` 内部の遅延 import で `ImportError`（案内付き）。fail-fast 設計

## 落とし穴

- AGT の import は関数内遅延。窓口 import 自体は extra 未導入でも壊れないが、`build()` 実行時にアクセスで例外
- `sub_agents` の as_tool・`register_factory` 経路は既定では govern 対象外。`post_processor(sub_agent_tools=True, factory_agents=True)` を registry の `post_processor` 引数へ渡すと対象になる
- builder 自体を `AgentRegistry(post_processor=builder)` に渡すと `TypeError`。`builder.post_processor(...)` の戻り値を渡す
- `sub_agent_tools=True` では as_tool の入力文が監査 `tool:` レコードの `details.arguments` に全文記録される（個人情報等を含みうるため `audit_sink` の永続先を考慮する）
- 統治済みの Agent（`registry.get` の戻り等）を返す factory を `factory_agents=True` で統治すると、元のポリシーと factory 名のポリシーの両方で評価され、`tool:` レコードが 2 行残る（どちらかが deny なら deny）。境界の詳細は `docs/architecture.md`（AGT ガバナンス節）を参照
- clone された registry は builder を共有する（監査チェーンが混ざる）。系を分けたい場合は builder を別々に注入
- **hosted MCP**（Responses API のサーバ側 MCP・`HostedMCPTool`）はモデルプロバイダ側で実行されるため評価も監査も発生しない。統治されるのは client-side MCP（`spec.mcp_servers`）のみ。`RealtimeAgentSpec` の `mcp_servers` も別 builder 経路のため対象外。同様に `list_prompts` / `get_prompt` / resources 経由で取得した文面を `instructions` 等へ流し込む使い方はツール呼び出しでないため統治対象外
- origin が取得できない（MCP 由来かどうか判定できない）ツールも同様に評価も監査もされない（fail-open・無警告）
- 評価対象はツール名と引数のみで、ツールの戻り値は評価も content 照合も受けない。第三者の MCP サーバを使う場合は戻り値が間接プロンプトインジェクションの経路になるため、信頼境界の外に置く MCP サーバには出力ガードレールを併用する（付け方は [guardrails.md](./guardrails.md) の「MCP サーバ単位で掛ける」）
- **MCP の deny は利用者の `spec.hooks.on_tool_start` へ到達しない**（`spec.tools` の deny では到達する）。利用者フックで監査・計測している場合は観測が欠ける。統治ガードレールを付けたサーバでは deny が `on_tool_start` より前に送出されるため、`RunHooks.on_tool_start` も開始されない（付けないサーバでは、SDK が `RunHooks` と `AgentHooks` の `on_tool_start` を並行実行するため、`RunHooks.on_tool_start` は deny 時も開始済みになりうる）
- 統治ガードレールは `tool_input_guardrails` の**先頭**に置く。内容検査より後ろに置くと、前の内容検査が reject した呼び出しは統治の評価も記録も受けない（`on_tool_start` にも到達しないため安全網も働かない）。lib は並び順を検査しない
- 統治ガードレールは `MCPServer` にだけ付ける。`function_tool(tool_input_guardrails=[...])` など `spec.tools` に付けると、build 時ラップとの二重評価になる（`tool:` レコードが 2 行残り、どちらかが deny なら deny）
- 統治ガードレールを付けたサーバでは監査列の順序が変わる。allow は `tool:`（allow）-> `tool_start:` -> `tool_end:` の順になり、deny では `tool_start:` が残らない（`tool:` deny のみ）。付けないサーバでは `tool_start:` -> `tool:` の順。`tool:` レコードの形（`agent_id` = `spec.name`・`details.arguments`）はどちらも同じ
- `tool:` レコードはポリシー判定の記録であり、実行の記録ではない。統治 allow の後に内容検査が reject した場合と、pre_approval 下で承認前に allow と判定した後に人が承認を却下した場合は、`tool:` allow が残るが呼び出しは実行されない。実行の有無は `tool_start:` / `tool_end:` で判別する
- 内容検査の結果は `RunResult.tool_input_guardrail_results` で観測する。統治ガードレールの allow も呼び出しごとに載るため、`guardrail.get_name() == "mcp_governance_guardrail"` の行を除いて読む（統治の deny は送出が先に起きるため載らない）
- `RunConfig(tool_execution=ToolExecutionConfig(pre_approval_tool_input_guardrails=True))` にすると、承認を要する MCP ツールの呼び出しで、統治ガードレールの deny が承認要求より前に出る。allow の場合は承認後にもう一度評価されるため、同じ呼び出しの `tool:` allow が 2 行残りうる
- MCP の deny は run を `UserError` で終了させる。MCP ツール自身の実行時例外が `mcp_config["failure_error_function"]` でモデルへ返り会話が継続するのとは挙動が違う
- deny で run が中断すると、stdio 接続の MCP サーバーの切断時に SDK が `Error cleaning up server: unhandled errors in a TaskGroup` を ERROR でログ出力する（非致命的。切断自体は完了し終了コードも変わらない）。deny が起きない実行では出ないため、ログ監視のしきい値設定で誤検知になりうる
- `mcp_config={"include_server_in_tool_names": True}` にすると公開名は基本形（`mcp_{サーバ名}__{ツール名}`）から SDK が変形を加える場合がある。`allowed_tools` は実際の公開名を確認して宣言する必要がある（未対応なら全 deny になり安全側で顕在化する。詳細は `docs/architecture.md` を参照）
- `allowed_tools` は名前照合で、MCP ツールの実体はターンごとに再解決される。同名のまま schema / 意味だけ差し替える変更は検知しない
- build 後に `Agent.hooks` を差し替える（`clone(hooks=...)` を含む）と、MCP 経路は強制と監査がともに失われる（`spec.tools` 経路は強制と per-call の記録が残る）。差し替えでなく合成したい場合は `spec.hooks` へ自前フックを宣言する
- SDK の HITL 承認（`needs_approval`）は govern の評価より先に走るため、ポリシーが拒否する呼び出しでも承認要求は先に出る。govern 済みツールの条件付き承認（callable な `needs_approval`）の判定は govern なしと同じで、型変換される引数（数値の文字列表現等）も SDK の引数事前検証を経てから判定される。設計判断は `docs/adr/0045-govern-wrapper-sdk-invoker-conformance.md`
- deny は per-call であり、同一ターンに複数のツール呼び出しがある場合、deny 発生時点で並行実行済みの兄弟呼び出しの副作用は残る（ターン単位のロールバックではない）
- 監査の `details` には MCP ツールの引数も全文記録される（URL / 接続情報が入りうるため `audit_sink` の永続先を考慮する）
- 一方で**例外**の `details` には引数を含めない（`tool_name` / `reason` のみ）。引数の取得先は監査 sink であり、例外からは辿れない
- `reason` は AGT が生成する説明文で、`allowed_tools` の全量やブロック用の正規表現パターンといった防御構成を含む。エンドユーザー向けのエラーレスポンスへそのまま載せず、`tool_name` のみを使うか定型文へ写す
- 「引数は例外からは辿れない」は lib が能動的に載せないという意味であり、`policy` は duck typing で利用者のオブジェクトを受け入れ `reason` を文字列として透過するため、自作 policy が `reason` に引数断片を埋めた場合はその文字列が例外へ載る

## 参照

- 詳細設計: `docs/architecture.md`（AGT ガバナンス節）
- 検討経緯: `docs/rationale/agt-governance-integration.md`
- 設計判断（MCP の統治ガードレール）: `docs/adr/0048-mcp-governance-as-tool-input-guardrail.md`
- 具体例: `examples/governance/01_policy_enforcement.py` 〜 `04_policy_bundle.py`

## 次

[integrity.md](./integrity.md) — lockdown と manifest 検証
