# 0025: MCP 由来ツールの統治を AgentHooks.on_tool_start で行う

- Status: accepted
- Date: 2026-08-07

## Context

`govern_spec` は `spec.tools` の各 `FunctionTool` の実行本体（`on_invoke_tool`）を build 時に
ポリシー評価付きへ非破壊置換することで統治していた。MCP サーバー経由のツールはこの経路に
乗らない。SDK は `Agent.get_all_tools` から `MCPUtil.to_function_tool` を呼び、**run 時
（ターンごと）** にサーバーへ list_tools して `FunctionTool` を生成するため、build 時には
ラップ対象のオブジェクトが存在しない。

結果として、宣言した `allowed_tools` / `blocked_patterns` が「`spec.tools` では効くが MCP
ツールでは効かない」非対称が生まれていた。任意 URL へ外部 HTTP を送るツールがガバナンスの網の
外に置かれる点で運用上のリスクがあり、利用側（別リポジトリのアプリ）は本体無改修の制約下で
同じ責務（`AgentHooks.on_tool_start` での評価・引数照合の候補生成・非公開
`load_policy_bundle` への依存）を自前実装していた。判定の意味論が 2 箇所に存在し乖離しうる
状態だった。

### 評価点の候補と却下理由

| 候補 | 却下理由 |
|---|---|
| MCP 由来 `FunctionTool` の `on_invoke_tool` の内側へ評価を差し込む | 当該ツールは失敗ハンドラで包まれ、内部例外を `failure_error_function`（既定 `default_tool_error_function`）で**文字列化してモデルへ返す**。拒否が単なるツールエラーメッセージへ退化し会話が継続する（deny が無効化される） |
| SDK ネイティブの `tool_filter` / `create_static_tool_filter` | 列挙段階のフィルタであり呼び出し引数を見られないため `blocked_patterns` を表現できない。callable filter 内から監査 sink へ記録することは可能だが、記録できるのは列挙時の可否であり per-call の allow/deny 決定記録（1 呼び出し 1 行）にならない。宣言が per-server 引数のためポリシー宣言 1 本の規約からも外れる |
| SDK の tool 入力ガードレール（`ToolInputGuardrail`）+ `guard_tool` | MCP 由来 `FunctionTool` には `tool_input_guardrails` が付かない。`guard_tool` は build 時に既存 tool オブジェクトへ後付けする方式で、run 時生成の MCP ツールには装着対象が存在しない |
| build 時に MCP ツールの proxy `FunctionTool` を `spec.tools` へ置く | `list_tools` が async かつサーバー接続済みを要求するため build 時に I/O とサーバー接続を持ち込み build-don't-run に反する。ツールは毎ターン再解決されるため build 時スナップショットは陳腐化し、サーバー側のツール追加を取りこぼす |
| `RunHooks`（run 単位フック）側での評価 | `AgentSpec` に run 単位フックの宣言スロットが無く、装着は利用者が `Runner.run(hooks=...)` で行うため build-don't-run の lib からは結線できない。`chain_agent_hooks` は `RunHooksBase` を build 時に `TypeError` で拒否する既存契約（ADR-0017）とも衝突する |
| AGT の MCP Security Gateway コンポーネント | sidecar / 別プロセスのゲートウェイを前提とし、in-process・build 時結線・実行ループ非所有という取り込みの篩を満たさない（`docs/rationale/agt-governance-integration.md` で範囲外と判断済み） |
| ポリシーロードの公開窓口化（`load_policy_bundle` の公開） | 利用側が自前で MCP を評価するために非公開 API へ依存していた動機は、本 ADR の決定により消える。使われない公開 API を増やさない |

### origin 判定方式の候補

negative 判定（`ToolOriginType.FUNCTION` 以外を評価する）は却下した。利用者が
`agent.as_tool(...)` で作った `FunctionTool` を `spec.tools` へ直接置く宣言は現在も可能で、
その tool は build 時に govern ラップされ、かつ origin が `AGENT_AS_TOOL` になる。negative
判定はこれを `on_tool_start` でも評価するため二重評価・監査レコード二重化が現行 SDK で成立
する。将来 SDK が新 origin 型を追加した場合に評価対象へ暗黙に取り込む点も難点である。

`AGENT_AS_TOOL` origin（`sub_agents` の as_tool）を同時に対象化する案も却下した。機構上は
同じフックで評価できるが、既存利用者の `allowed_tools` にサブエージェントの as_tool 名は
書かれておらず、対象化すると既存デプロイが即 deny で機能停止する。`sub_agents` の統治は
`allowed_tools` へ as_tool 名を書く規約の追加を伴う独立した意思決定である。

### `PolicyViolationError` の基底

`AgentsException` 派生へ変える案も検討したが却下した。当該例外は lib が定義しておらず AGT core
由来（`Exception` 派生）で、基底を変えるには lib 側で別例外型を新設して AGT 例外を置き換える
ことになり、「AGT 由来・`isinstance` 互換で再エクスポート」という公開契約と AGT への委譲方針を
壊す。SDK が非 `AgentsException` を `UserError(...) from e` にラップする性質は既存の
`spec.tools` 経路にも同じく当てはまり、本 ADR の決定で新たな非対称は生まれない。

## Decision

ADR-0016 の「`_make_audit_hooks` が作る `AgentHooks` は監査記録のみを行う」（`0016:62`）および
「governance の監査フックは監査記録の責務だけを持つ」（`0016:81`）を**部分的に撤回**し、責務を
ポリシー評価まで広げる。ADR-0016 本文は append-only のため書き換えず、Status に
`partially superseded by 0025` を追記する。ADR-0016 の他の判断（委譲実体を `_adapters/hooks.py`
へ一元化・`chain_agent_hooks(audit, inner)` の合成形と宣言順・`AgentHooksBase` 継承の維持・
`chain_agent_hooks` の関数内遅延 import）は有効なまま残る。

- **評価点**: `_make_audit_hooks` が作る `AgentHooks` の `on_tool_start` で評価する。この位置は
  実ツール呼び出しの task 生成より前に await が完了するため、送出により実行本体へ到達しない。
- **対象の絞り込み**: `get_function_tool_origin(tool)` の `type` が `ToolOriginType.MCP` の
  ときだけ評価する（positive 判定）。比較は `is` ではなく `!=` を使う。`ToolOriginType` は
  `str` 派生 Enum で `ToolOrigin` は型検証を持たない frozen dataclass のため、生 str の
  `ToolOrigin(type="mcp")` が渡り得る。identity 比較では同値でも不一致になり、統治が無警告で
  スキップされる。
- **判定の再利用**: 引数照合は既存 `_evaluate_tool` をそのまま呼ぶ。`on_tool_start` に渡る
  `ToolContext.tool_arguments` は `on_invoke_tool` の `input_json` と同一の生ワイヤ文字列で
  あり、3 系統の照合（生 / JSON 正規化 / デコード済み文字列スカラ）が MCP でも同一に効く。
  判定の意味論を 1 箇所に留めることが本決定の要点である。
- **fail-closed**: `tool_arguments` が `str` として取得できない場合は名前照合のみへ縮退させず
  deny する。縮退させると `blocked_patterns` が無警告で効かなくなる。
- **fail-open**: origin が取得できない（`None`）場合と MCP でない場合は評価しない。origin メタが
  落ちた場合に deny へ倒すと、MCP に限らず build 時に govern 済みの `spec.tools` 由来 tool まで
  一律 deny になり正当な呼び出しが機能停止する。SDK 契約の変化は CI のトリップワイヤで検知する。
- **監査形式**: 既存 `_govern_tool` と同一の `tool:{name}` 形式で記録し、`details` に引数を
  載せる。`tool:` の `agent_id` は宣言時の `spec.name`。deny 時は記録を残してから送出する。
- **合成形**: `chain_agent_hooks(_AuditAgentHooks(), inner)` のまま。要素数・宣言順・内部クラス名
  を変えない（`inner is None` の実効 1 件最適化を維持する）。追加引数は kw-only の任意引数
  （既定 `None` = 評価しない）とし、既存の 2 引数呼び出しを無変更で通す。
- **宣言面**: `AgentSpec` に `mcp_servers` / `mcp_config` を kw_only の専用フィールドとして
  持たせ、`extra` 素通しをやめる。`extra` への同名キーは既存の衝突検査で `ValueError` になる
  （特別扱いコードは書かない）。MCP 配線の有無で govern フックの装着を分岐させない
  （origin 判定が MCP 以外を no-op にするため、検出条件を列挙し続ける保守負債を避ける）。
- **build-don't-run**: 本決定は build 時にフックを組んで `spec.hooks` へ合成するだけで、実行する
  のは SDK `Runner` が呼ぶ `on_tool_start` の中身である。既存の `spec.tools` 統治と同じ性質で
  あり、build-don't-run の例外を新設しない。

## Consequences

- **+** 判定の意味論が `_evaluate_tool` の 1 箇所に留まる。MCP と `spec.tools` で照合規則が
  乖離しない。
- **+** ポリシー宣言が 1 本に統一される（MCP ツールも同じ `allowed_tools` / `blocked_patterns`
  で扱える）。利用側は builder を注入するだけでよく、自前のフック実装と非公開 API 依存を撤去
  できる。
- **+** 宣言面が型付きになり、綴り誤りが build 時に検出される。境界を docstring に書く場所が
  できる。
- **-** `extra={"mcp_servers": ...}` / `extra={"mcp_config": ...}` は build 時 `ValueError` に
  なる（限定的な破壊的変更）。移行は宣言 1 行の機械的書き換えで済む。
- **-** `allowed_tools` を宣言している既存 spec で MCP を使っている場合、未掲載の MCP ツールが
  deny になる（挙動変更）。ポリシー宣言への追記が必要。
- **-** MCP の deny は `on_tool_start` からの送出で合成チェーンを中断するため、利用者の
  `spec.hooks.on_tool_start` へ到達しない（`spec.tools` の deny では到達する非対称）。利用者
  フックで監査・計測している場合は観測が欠ける。`RunHooks.on_tool_start` は SDK が
  `asyncio.gather` で並行実行するため deny 時も開始済みになり得る。
- **-** MCP の deny は run を `UserError` で終了させる。MCP ツール自身の実行時例外が
  `mcp_config["failure_error_function"]` でモデルへ文字列返却され会話が継続するのとは挙動が違う。
- **-** 強制の成立が SDK の非公開メタ（`FunctionTool._tool_origin` / `._emit_tool_origin` /
  `agents.tool.get_function_tool_origin`。いずれもトップレベル未 export）に依存する。契約が
  壊れると fail-open のため MCP が無警告で未統治になる。検知手段は CI のトリップワイヤのみで、
  **利用者環境で SDK を単独 upgrade した場合は検知されない**。
- **-** fail-open は現行 SDK でも到達しうる。`build_litellm_json_tool_call` が生成する合成ツール
  （LiteLLM + `output_schema` 時の構造化出力デシリアライズ用）が `_emit_tool_origin=False` を
  設定するため、この tool の `on_tool_start` では origin が `None` になり評価されない。当該
  ツールは MCP ツールではないので統治の穴ではなく、ここを deny に倒すと無関係な機能が壊れる。
- **-** 監査の `details` に MCP ツールの引数が全文記録されるようになった（本経路は従来
  `tool_start:` の記録のみで `details` を持たなかった）。MCP ツールの引数には URL・接続情報が
  入りやすいため、記録先の選定に影響する。
- **-** 統治されるのは client-side MCP（`spec.mcp_servers`）のみ。hosted MCP
  （`HostedMCPTool`）はモデルプロバイダ側で実行され `on_tool_start` が発火しないため、評価も
  監査も発生しない。`RealtimeAgentSpec` の `mcp_servers` も別 builder 経路のため対象外。
- **-** `allowed_tools` は名前照合であり、MCP ツールの実体はターンごとに再解決される。同名の
  まま schema / 意味だけ差し替える変更は検知しない。
- **-** `mcp_config["include_server_in_tool_names"]` を真にすると、SDK が生成する公開名は
  `mcp_{サーバ名}__{ツール名}` を基本形とし、文字置換や長さ超過時の切り詰めなどの変形を加える
  場合があるため、`allowed_tools` は実際の公開名を確認して宣言する必要がある。
- **-** 評価対象はツール名と引数のみで、MCP サーバが返す結果は評価も content 照合も受けずモデル
  文脈へ入る（`on_tool_end` は記録のみ）。MCP はサーバが第三者であるため、許可したツールの戻り値
  が間接プロンプトインジェクションの主経路になる。信頼境界の外に置く場合は SDK の出力ガードレール
  を併用する。

## Confirmation

不変条件と強制手段の対応は `docs/QUALITY-GUARANTEES.md` に登録する（テストの追加・改名に
追随する可変層を台帳へ一本化し、本 ADR には改名で古くならない設計方針だけを残す）。

本 ADR が成立に必要とする性質は次の 4 つで、いずれも**保証対象を壊す変異を注入して当該テストが
RED になることを実行で確認**してある。

1. **評価対象が MCP origin に限られること**（negative 判定へ退行すると `spec.tools` に置かれた
   as_tool が二重評価され、監査レコードが二重化する）。origin 比較が生 str も捕捉すること
   （identity 比較へ戻すと統治が無警告でスキップされる）。
2. **deny の証跡が送出より前に残ること**（順序が入れ替わると deny の記録が失われ、利用側の
   `action.startswith("tool:")` による抽出が空になる）。
3. **引数が取得できない場合に名前照合へ縮退しないこと**（縮退すると `blocked_patterns` が無警告
   で効かなくなる）。
4. **SDK 側の origin 契約が upgrade で silent に壊れないこと**（fail-open のため、壊れても例外も
   警告も出ずに MCP が未統治になる）。実 SDK の MCP ツール生成経路を通した検証が必要で、
   手で組んだ偽装 tool だけでは SDK が origin 付与をやめた退行を検知できない。
