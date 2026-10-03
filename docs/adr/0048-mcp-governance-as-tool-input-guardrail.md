# 0048: MCP 由来ツールの統治を MCPServer のツール入力ガードレールとしても提供し、on_tool_start を安全網として残す

- Status: accepted
- Date: 2026-10-03

## Context

### 出発点

ADR-0025 は、MCP 由来ツールの統治（`allowed_tools` / `blocked_patterns` の評価と `tool:` 監査）を、
`_make_audit_hooks` が作る `AgentHooks.on_tool_start` で行うと決めた。同 ADR は SDK の tool 入力ガードレールを
評価点の候補から外しており、その理由の前半は「MCP 由来 `FunctionTool` には `tool_input_guardrails` が付かない」
（ADR-0025:27）だった。

openai-agents 0.22.1 で `MCPServer` のコンストラクタが kw-only 引数 `tool_input_guardrails` /
`tool_output_guardrails` を受け取るようになり、SDK は `MCPUtil.to_function_tool` でターンごとに生成する
`FunctionTool` へそのリストを付ける。ADR-0025:27 の前半の前提はこれで成り立たなくなった（後半の
「`guard_tool` は run 時生成の MCP ツールに装着対象が無い」は引き続き成り立つ）。ADR-0043 は依存追随の判断の中で
この変化を記録し、評価点を MCPServer 単位のガードレールへ移すかは別の判断として保留した（ADR-0043:293-295）。
本 ADR がその判断である。

### 公式ドキュメントでの位置付け

SDK のドキュメントは、ツールガードレールを呼び出しの検査・阻止の手段、ライフサイクルフックを観測の手段として
位置付けている。

- guardrails: 「Tool guardrails wrap `FunctionTool` instances and let you validate or block calls to those tools
  before and after execution.」「You can also set `tool_input_guardrails` and `tool_output_guardrails` on a local
  MCP server; the SDK attaches those lists to every tool exposed by that server.」「input tool guardrails normally
  run after approval and immediately before execution.」
- agents（lifecycle）: 「Sometimes, you want to observe the lifecycle of an agent. For example, you may want to
  log events, pre-fetch data, or record usage when certain events occur.」

### 実測

環境は openai-agents 0.22.3 / mcp 1.28.1。課金 API には接続せず、FakeModel とスタブの `MCPServer` で実行した。

- S1（出力ガードレールが `reject_content`）: 順序は入力ガードレール -> `on_tool_start`（lib の統治評価を含む）
  -> `call_tool` -> 出力ガードレール -> `on_tool_end`。Session の `function_call_output` は置き換え後の値だけで、
  元の出力文字列は Session のどこにも残らなかった。
- S2（入力ガードレールが `reject_content`）: `on_tool_start` は呼ばれず、`call_tool` にも到達しない。監査は
  `agent_start` / `agent_end` だけで、`tool_start:` も `tool:` も残らなかった。lib の統治を内容検査と併用すると、
  内容検査が止めた呼び出しは統治の監査から見えない。
- S3（ガードレール関数内で例外を送出）: AGT の `PolicyViolationError` を送出すると、`Runner.run` は
  `UserError` を送出し、その `__cause__` は送出した例外そのもの（`is` 一致）だった。`RuntimeError` でも同形。
  ガードレールはツール名・引数 JSON・エージェント名を受け取れた。入力ガードレールはリスト順に逐次評価され、
  最初の reject / raise で止まる。
- S4（試作 (g)。Agent のサブクラスで `get_mcp_tools` を上書きし、統治ガードレールを先頭に差し込む）: ポリシーが
  deny する呼び出しでは統治だけが評価され、後ろの内容検査の検知器は呼ばれず、監査は `tool:read` deny の 1 行
  だった。ポリシーが allow する呼び出しでは統治 -> 内容検査の順に評価され、`tool:read` allow が残った。

S3 から、統治の deny をガードレールの位置で行っても、ADR-0030 の着地（`UserError` の `__cause__` に
`PolicyViolationError`）は保てる。S2 / S4 から、統治を内容検査より前に置けば、違反は必ず `tool:` に残り、
deny された呼び出しで内容検査の検知器は呼ばれない。

### 検討した案

- (a) 新しい API は足さず docs で案内する。統治は `on_tool_start` のまま、内容検査と併存させる
- (b1) `AgentSpec` 等に宣言を足し、builder が利用者の `MCPServer` の属性へ統治ガードレールを注入する
- (b2) builder が `MCPServer` を proxy で包んで差し替える
- (b3) `GuardrailRegistry.mcp_server_kwargs(names)` を足す
- (c) 統治をガードレールへ移し、`on_tool_start` での評価を廃止する
- (d) 内容検査が止めた呼び出しを統治監査へ書くヘルパを足す
- (g) Agent のサブクラスで `get_mcp_tools` を上書きし、統治ガードレールを自動で先頭に差し込む
- (e2) 統治ガードレールを提供し（利用者が先頭に付ける・オプトイン）、`on_tool_start` を安全網として残す

| 観点 | (a) | (b1)/(b2) | (b3) | (c) | (d) | (g) | (e2) |
|---|---|---|---|---|---|---|---|
| deny の証跡（内容検査と併用したとき） | 内容検査が先に止めると `tool:` deny が残らない | 注入の仕方次第 | (a) と同じ | 残る | 別形式で残る | 残る | 残る（統治が先頭） |
| deny の着地（ADR-0030） | 不変 | 不変 | 不変 | S3 により保てる | 不変 | 保てる | 保てる（S3） |
| deny された呼び出しで内容検査の検知器が呼ばれるか | 呼ばれる | — | 呼ばれる | 呼ばれない | 呼ばれる | 呼ばれない | 呼ばれない（S4） |
| 利用者の `MCPServer` インスタンス | 触らない | 書き換える / 包む | 触らない | 注入が要る | 触らない | 触らない | 触らない（利用者がコンストラクタで渡す） |
| 付け忘れ時 | — | — | — | 統治が外れる | — | 付け忘れが起きない | 安全網で統治される |
| 既定の挙動 | 不変 | 変わる | 不変 | 変わる | 不変 | 統治済みの全エージェントで監査列が変わる | 不変（オプトイン） |
| SDK への依存 | なし | proxy は非公開ヘルパに依存 | kwarg 名 | なし | なし | Agent のサブクラス化と `get_mcp_tools` の上書き。inner builder が別の Agent 型を作る経路と両立しにくい | 公開のガードレール型と、ガードレールと `on_tool_start` が同じ `ToolContext` を受けること |
| 公開 API | 不変 | `AgentSpec` が増える | メソッドが増える | 破壊的変更 | 増える | 不変 | `runtime.governance` に 1 シンボル |

(e2) を採る理由:

- (c) の利点（deny の証跡・内容検査のコスト・SDK の正式な手段）を得ながら、(c) の欠点（付け忘れで統治が外れる・
  既定の挙動変更）を安全網とオプトインで避ける。
- (g) は利用者の作業を不要にする代わりに、SDK の Agent のサブクラス化と既定の挙動変更を伴う。統治を利用者が
  明示的に宣言する形は宣言的 lib の流儀に合う。
- (c) の却下理由として想定していた「deny の着地が変わる」は S3 で成り立たない。(c) に残る却下理由は、付け忘れで
  統治が外れることと既定の挙動変更である。

あわせて次の 3 案を比較して却下した。

| 案 | 却下理由 |
|---|---|
| 統治ガードレールを builder に結び付ける（`GovernedAgentBuilder.mcp_tool_guardrail()` が自 builder のポリシー・sink で評価し、他の builder のエージェントは素通しする） | builder の印を `govern_spec` / `govern_agent` / `_make_audit_hooks` へ通す機構が要る。複数の builder で共有するサーバでは、他の builder のエージェントが内容検査より後の `on_tool_start` に回り、「内容検査より前に評価する」が builder ごとにしか成り立たない |
| `on_tool_start` の評価を省く条件を「ツールの `tool_input_guardrails` に統治ガードレールが付いているか」にする | 付いていても、ガードレールが監査フックへ辿れず評価しなかった場合（利用者が hooks を duck-typed のラッパで包んだ場合など）に、`on_tool_start` まで評価を省いて統治が無言で外れる |
| pre_approval 下の `tool:` allow の重複を `tool_call_id` で除く | 承認の前後で `ToolContext` が別のオブジェクトになるため、呼び出しをまたぐ状態が要る。承認の却下や内容検査の reject で消費されない id が長命の Agent に溜まり、上限の定数も要る。得られるのは SDK のオプトイン設定時の allow 1 行の重複を消すことだけである |

## Decision

### 1. 公開形

`oai_agentspec.runtime.governance.mcp_governance_guardrail() -> object` を追加する。引数は無く、戻り値は
不透明値（実体は SDK の `ToolInputGuardrail`）である。

- ガードレールの name は固定の `"mcp_governance_guardrail"` とする。`RunResult.tool_input_guardrail_results` の
  `guardrail.get_name()` で、統治の行と内容検査の行を区別するための名前である。
- `runtime.governance.__all__` は `GovernedAgentBuilder` / `PolicyViolationError` / `mcp_governance_guardrail`
  になる。コアの `__all__` は変えない。
- 本体は `_adapters/governance.py` に置く。公開関数は `runtime/governance/guardrail.py` に置き、`_adapters` へは
  関数内の遅延 import で委譲する（governance extra が未導入でも窓口の import は壊れない）。`__init__.py` は
  再エクスポートだけを行う。
- 利用者は `MCPServer` のコンストラクタの `tool_input_guardrails` に、内容検査より前（先頭）に置く。

```python
server = MCPServerStdio(
    params=...,
    tool_input_guardrails=[mcp_governance_guardrail(), tool_guardrail(d_in, on="input")],
    tool_output_guardrails=[tool_guardrail(d_out, on="output")],
)
```

### 2. 評価と安全網

- 統治ガードレールは、受け取ったエージェント（`data.agent`）の hooks から lib の監査フック
  （`_AuditAgentHooks`）を辿り、各監査フックが持つポリシー・sink・`spec.name` で
  `data.context.tool_name` / `data.context.tool_arguments` を評価する。判定は既存の `_evaluate_tool` を、
  送出は既存の `_deny_tool_call` を使う（判定の意味論を 1 か所に留める ADR-0025 の要点を維持する）。
- deny: `tool:` deny を記録してから `PolicyViolationError` を送出する。後続の監査フック・後続のガードレールは
  評価しない。`reject_content` / `raise_exception` は返さない。
- allow: `tool:` allow（`details.arguments` 付き）を記録し、その監査フックの印に `data.context` と評価した
  ツール名（`data.context.tool_name`）を記録して、`ToolGuardrailFunctionOutput.allow()` を返す。
- 辿れる監査フックが無い（未統治のエージェント・hooks を差し替えたエージェント）: 記録せずに allow を返す。
- 印は監査フックごとの `weakref.WeakKeyDictionary[ToolContext, str]`（値は評価したツール名）とする。
  `ToolContext` は同一性で hash されるため、呼び出しが終われば印も消え、蓄積しない。
- `_AuditAgentHooks.on_tool_start` は従来どおり `tool_start:` を記録し、MCP origin の呼び出しについて、
  印の名前が `on_tool_start` の評価に使うツール名（`tool.name`）と一致するときだけ評価を省き、印が無い・名前が
  ずれる場合は従来どおり評価する（安全網）。ガードレールの `context.tool_name` と `on_tool_start` の `tool.name`
  がずれても安全網が外れない多層防御で、倒れる先は二重評価の側である。統治ガードレールを付けない利用者では
  印が無いため、従来の分岐をそのまま通る。
- `_AuditAgentHooks` は状態（sink・ポリシー・印）をインスタンスに持つため、`__deepcopy__` は self を返す。
  AuditLog のロックは複製できず、監査の記録先も複製しない。SDK のエージェント同一性シグネチャが `model` 等の
  dataclass から Agent へ届いて深くコピーする場合も、利用者の `copy.deepcopy(agent)` も、コピーは同じ監査フックを
  共有し統治は外れない。ADR-0045 の invoker の統治済みの印（deepcopy で外れる）とは対象が別である。
- `_AuditAgentHooks` はガードレールから辿るためモジュール水準へ移す。クラス名・合成形・要素数・宣言順は
  変えない（ADR-0025 の合成形の決定を維持する）。
- fail-closed（`tool_arguments` が `str` として取得できない場合は名前照合へ縮退させず deny する）は、
  ガードレールの位置でも同じに適用する。

### 3. hooks を辿る規則

根は `data.agent.hooks` で、次の 3 規則で再帰する。

1. 要素が `_AuditAgentHooks` でポリシーを持つ -> 評価対象に加える（中は辿らない）
2. 要素が `_ChainedAgentHooks` -> その要素列を宣言順に、1〜3 の規則で辿る
3. それ以外（`None`・利用者のフック・duck-typed のラッパ・ポリシーを持たない監査フック）-> そこで止まる

`spec.hooks` が `None` の場合は `chain_agent_hooks` の最適化で根が `_AuditAgentHooks` そのものになり、規則 1 が
扱う。評価対象は宣言順に評価し、最初の deny で送出する。辿るのは同じパッケージの lib 型だけで、SDK の型には
依存しない。

この規則により、次の経路はそれぞれ次のように扱われる。

- 共有サーバ: 各エージェントの hooks が自分のポリシーと `spec.name` を持つため、builder が違っても per-agent に
  評価される。未統治のエージェントは評価対象が無いので素通しする。
- `clone()`: hooks を保つので統治される（`tool:` の `agent_id` は元の `spec.name`）。`clone(hooks=...)` と
  `Agent.hooks` の差し替えは辿れず、`on_tool_start` も無いので未統治になる（ADR-0025 の範囲での境界と同じ）。
  差し替え先が元の監査フックへ委譲するラッパなら、ガードレールは辿れないが `on_tool_start` の安全網が評価する。
- `register_factory` + `post_processor(factory_agents=True)`（ADR-0042）・入れ子の `GovernedAgentBuilder`:
  入れ子の `_ChainedAgentHooks` を辿り、各ポリシーを宣言順に評価する。
- `sub_agents` の as_tool は MCP ではないため対象外。サブエージェント自身の MCP サーバは、そのサブエージェントの
  hooks で評価される。

### 4. `tool:` は判定の記録であること・pre_approval・並び順

- 統治の deny は `on_tool_start` より前に送出されるため、利用者の `spec.hooks.on_tool_start` にも
  `RunHooks.on_tool_start` にも到達しない。
- `RunConfig.tool_execution.pre_approval_tool_input_guardrails` を真にすると、承認を要する呼び出しで統治の deny が
  承認要求の前に出る。allow なら承認後にもう一度評価され、`tool:` allow が 2 行残る。この重複は受け入れる。
- `tool:` はポリシー判定の記録であり、実行の記録ではない。統治 allow の後に内容検査が reject した場合と、
  pre_approval 下で承認前の評価の後に人が承認を却下した場合は、`tool:` allow が残るが呼び出しは実行されない。
  実行の有無は `tool_start:` / `tool_end:` で判別する。
- SDK は統治ガードレールの allow も `RunResult.tool_input_guardrail_results` に載せる。内容検査の結果は
  `get_name() == "mcp_governance_guardrail"` の行を除いて読む。統治の deny は送出が先に起きるため載らない。
- 統治ガードレールを先頭以外に置くと、前にある内容検査が reject したときに統治の評価も記録も起きない。lib は
  並び順を検査できないため、docs で先頭に置くことを求める。

### 5. `spec.tools` へ付けた場合

`ToolInputGuardrailData` はツールの origin を持たないため、ガードレールは MCP 由来かどうかを判定できない。
`function_tool(tool_input_guardrails=[mcp_governance_guardrail()])` と付けると build 時ラップとの二重評価になる
（`tool:` が 2 行残り、どちらかが deny なら deny）。統治としては安全側のため検査は足さず、`MCPServer` にだけ
付けることを docs で求める。

### 6. ADR-0025 のうち置き換える範囲

ADR-0025 の Decision の評価点のうち、統治ガードレールを付けた MCP サーバのツールの評価点と、Context の
「評価点の候補と却下理由」の tool 入力ガードレールの行の前半（「MCP 由来 `FunctionTool` には
`tool_input_guardrails` が付かない」）を本 ADR で置き換える。ADR-0025 のそれ以外（MCP origin の positive 判定・
fail-closed / fail-open・監査形式・合成形・宣言面・統治ガードレールを付けないサーバでの `on_tool_start` 評価）は
有効なまま残る。

## Consequences

- + 統治ガードレールを先頭に付けたサーバでは、deny の証跡が内容検査の有無に関わらず `tool:` に残る。
- + deny された呼び出しで内容検査の検知器が呼ばれないため、検知器（外部サービス・LLM 判定を含む）のコストが減る。
- + pre_approval を真にすれば、ポリシーが拒否する呼び出しを承認要求の前に止められる。
- + 付け忘れても統治は外れない（`on_tool_start` の安全網が評価する）。付けたが辿れない場合も安全網が評価し、
  失敗したときに倒れる先は二重評価の側である。
- + deny 時に利用者の `RunHooks.on_tool_start` も開始されない。ADR-0025:105-108 の「`RunHooks.on_tool_start` は
  deny 時も開始済みになり得る」は、統治ガードレールを付けた経路では成り立たない（付けない経路では引き続き成り立つ）。
- - `runtime.governance` の公開シンボルが 1 つ増える。
- - 統治ガードレールを付けた場合は監査列の順序が変わる（`tool:` が `tool_start:` より先に来る。deny では
  `tool_start:` が残らない）。`tool:` 行の形（`agent_id` = `spec.name`・`details.arguments`）は変わらない。
- - `tool:` は判定の記録で、実行されないことがある（内容検査の reject、pre_approval 下での承認却下）。
- - pre_approval 下では `tool:` allow が重複しうる。
- - `RunResult.tool_input_guardrail_results` に統治の allow も毎回載るため、内容検査の結果は name で除外して読む。
- - 並び順は lib が検査できない。先頭以外に置くと、前にある内容検査が止めた呼び出しは統治の評価も記録も受けない
  （安全網の `on_tool_start` にも到達しない）。
- - `spec.tools` へ付けると二重評価になる。
- - ガードレールと `on_tool_start` が同じ `ToolContext` を受けることに依存する。崩れた場合は二重評価（`tool:` が
  2 行）に倒れ、統治は外れない。
- - hosted MCP（`HostedMCPTool`）と `RealtimeAgentSpec` の `mcp_servers` は引き続き対象外。

## Confirmation

強制手段は `tests/_adapters/test_governance_l2.py`（FakeModel とスタブの `MCPServer` で SDK の実行経路を通す
結合テスト）と `tests/_adapters/test_governance_l1.py`（ガードレールの単体テスト）である。不変条件と強制手段の
対応は `docs/QUALITY-GUARANTEES.md` に登録する（source = ADR-0048）。テストの追加・改名に追随する可変層は
台帳へ一本化し、本 ADR には設計方針だけを残す。

本 ADR が成立に必要とする性質は次のとおりで、いずれも保証対象を壊す変異を注入して当該テストが RED になることを
実行で確認する。

1. 統治ガードレールを付けたとき、deny は送出の前に `tool:` deny を記録し、後続の内容検査の検知器を呼ばない。
   着地は `UserError` の `__cause__` に `PolicyViolationError` で、`details` は `tool_name` / `reason` のまま。
2. ガードレールで評価した呼び出しは `on_tool_start` で再評価しない（`tool:` は 1 行）。hooks の 2 形（根が監査
   フック / 合成）のどちらでも成り立つ。
3. ガードレールが評価しなかった呼び出しは `on_tool_start` が必ず評価する。付け忘れた場合も、付けたが辿れない
   場合も、ガードレールの評価名と `on_tool_start` のツール名がずれた場合も成り立つ。
4. 共有サーバでは、各エージェント自身のポリシーと `spec.name` で評価する。
5. `MCPServer` の出力ガードレールの `reject_content` は、置き換え後の値だけを Session に永続化する（lib の保証では
   なく、利用者向け docs の「redact として使える」という案内が前提にする SDK の性質を検知するトリップワイヤ）。
6. 統治済みの Agent（およびその監査フック）を `copy.deepcopy` しても、コピーは同一の監査フックを共有し、統治と
   監査の記録先は外れない。
