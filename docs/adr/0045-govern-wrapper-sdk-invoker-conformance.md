# 0045: govern の実行本体ラップを SDK の失敗ハンドラ付き invoker に準拠させ、統治済みの印を invoker の生成時に登録する

- Status: accepted
- Date: 2026-09-30

## Context

### 決定時点の govern ラップと承認の実行順序

- `_adapters/governance.py` の `_govern_tool` は、`FunctionTool.on_invoke_tool` を素の async 関数（ポリシー評価 →
  deny なら記録して送出 / allow なら記録して元の `on_invoke_tool` を await）へ置き換え、`dataclasses.replace` で
  新しい tool を作る。元の `on_invoke_tool` は SDK 内部の失敗ハンドラ付き invoker
  （`agents.tool._FailureHandlingFunctionToolInvoker`）である。
- SDK の run loop は 1 回の tool call を、承認判定（callable な `needs_approval`）→ tool input guardrails →
  `on_tool_start` フック → `on_invoke_tool` の順に処理する。govern の評価は最後の段にある。
- 統治済みの印（ADR-0042 の Decision 6）は、`_govern_tool` が作ったラッパ関数を `_GOVERNED_WRAPPERS` へ弱参照で
  登録し、`tool.on_invoke_tool` が登録済みの関数そのものかで判定する。`dataclasses.replace` / `copy` を経ても
  同じ関数を指すことに依存する。

### SDK 0.22.x の承認強化と govern の逆転（ADR-0043 の実測）

- SDK 0.22.3 は、`on_invoke_tool` が失敗ハンドラ付き invoker のインスタンスのときだけ、callable な
  `needs_approval` の前に引数をツールの入力モデルで検証する。検証で引数が変わる場合（緩い型変換・既定値の
  補完・予測できないスキーマ）は、判定関数を呼ばずに承認を必須にする。
- govern は実行本体を素の関数で置き換えるので、この強化を受けない。引数の表現を変えるだけで govern 済み
  ツールの判定をすり抜けられた（すり抜けた入力は ADR-0043 の「要修正」の govern の項を正とする）。govern を
  足すと承認が弱くなる逆転である。
- SDK の `FunctionTool.__post_init__` は、`on_invoke_tool` が再束縛プロトコル（`__agents_bind_function_tool__`）を
  持てば新しい tool へ束縛し直す。基底の invoker は束縛先が違えば別インスタンスを作る。invoker を
  `on_invoke_tool` に置くと、複製のたびに別オブジェクトになり、ADR-0042 の印の前提（複製が包み直さない）が
  成り立たなくなる。

### 検討した候補

| 観点 | SDK の invoker に準拠する案（採用） | `needs_approval` を包む案 | ポリシー評価を tool input guardrail へ移す案 |
|---|---|---|---|
| SDK 非公開要素への依存 | Decision の列挙のとおり | 入力の pydantic モデルは `function_tool` の閉包内にしか無く非公開。使うなら非公開依存、使わないなら lib 独自の JSON Schema 検証（新規依存か自作） | 無い（`ToolInputGuardrail` / `FunctionTool.tool_input_guardrails` は公開） |
| SDK 0.22.x の承認強化の適用 | 適用される（invoker の型判定を満たす） | 適用されない（実行本体は素の関数のまま）。lib が SDK の判定（緩い型変換・既定値の補完・予測できないスキーマ）を複製し、SDK との乖離が続く | 適用される見込み（実行本体を置き換えない。未試作） |
| 統治済みの印（ADR-0042 の Decision 6） | 判定方式（登録済みインスタンスとの同一性）は維持し、登録の時点を invoker の生成時へ移す | 影響なし | 印をガードレールの同一性へ移す必要があり、Decision 6 を置き換える |
| deny の着地 | 不変（評価は基底の失敗処理の外。`PolicyViolationError` がそのまま伝播） | 不変 | 変わる（tripwire は `ToolInputGuardrailTripwireTriggered`、reject_content はモデルへ文言返却）。利用者向けの deny の捕捉案内と ADR-0030 の details 契約に触れる |
| 監査記録の時点 | 不変（実行本体の直前） | 不変 | ガードレール段へ移る。利用者の `attach_tool_guardrails` との順序によっては、利用者側が reject すると監査が残らない |
| SDK の前提変化の検知 | parity テストとトリップワイヤ（Confirmation） | 複製した判定と SDK の判定の一致テスト（SDK 更新のたびに乖離しうる） | 公開 API のため低リスク |
| 試作 | 近い方式を試作済み（印を同一性とサブクラス判定の併用で保つ形。試したケースすべてで govern なしと一致し、governance 系テストの退行は 0 件。ADR-0043）。本 ADR の形（実行本体そのものの統治・生成時の登録）は未試作 | 未試作 | 未試作 |

依存が最少の「tool input guardrail へ移す案」を採らない理由:

- セキュリティ経路の公開挙動（deny の例外型・着地点・監査の時点）を変える。
- ADR-0042 の Decision 6 の印を置き換える。
- 未試作で、依存追随の作業の範囲を超える。

「`needs_approval` を包む案」を採らない理由: 依存を減らせても SDK の判定を lib が複製し続けるため、
「govern なしと同じ判定」を構造的に保証できない。

「SDK の invoker に準拠する案」は SDK 非公開要素への依存が増えるが、openai-agents の宣言範囲を 1 つの minor に
閉じる（ADR-0043 の R1・R2）ので依存の変化は次の追随作業で必ず検査される。加えて、前提の変化はテストで
検知でき、公開挙動を変えず、近い方式の試作で govern なしとの一致を確認している。

サブクラス判定（`isinstance(fn, _GovernedInvoker)`）を印に使わない理由: 利用者が private クラスを生成・継承
すれば印を得られるので、ADR-0042 が属性印を却下した理由（利用者が印を付けられる）と同じ弱さを持つ。
登録済みインスタンスとの同一性より弱くなるので、判定方式は変えず、登録の時点だけを変える。

## Decision

### 1. `_GovernedInvoker` を SDK の失敗ハンドラ付き invoker のサブクラスとして置く

`_adapters/governance.py`（SDK import は `_adapters` 内に閉じる）に
`_GovernedInvoker(_FailureHandlingFunctionToolInvoker)` を置く。

- `__init__(inner, *, policy, sink, denied_exc, agent_name, tool_name, function_tool)`:
  - 統治済みの実行本体 `governed_impl(ctx, input_json)` を組む。手順は従来の govern ラップと同じで、ポリシー
    評価 → deny なら記録して送出 / allow なら記録して `await inner(ctx, input_json)` とする。`inner` は元の
    `on_invoke_tool`（SDK の失敗ハンドラ付き invoker）で、失敗処理は `inner` が担う。
  - 基底の `__init__` へ `invoke_tool_impl=governed_impl` を渡す。`inner` の `_invoke_tool_impl` は渡さない。
    SDK が `__call__` を通らず `_invoke_tool_impl` を直接呼んでも評価を迂回できないようにするためである。
    `on_handled_error` は `inner` のものを、`function_tool` は引数どおりを渡す。
  - 同期関数ツールのマーカー（`_SYNC_FUNCTION_TOOL_MARKER`）は、`inner` が持つ場合だけ自分にも付ける（SDK の
    再束縛と同じ規則）。
  - 最後に `_register_governed(self)` で登録する。登録はここに一本化し、生成経路によらず生成された
    インスタンスは必ず登録済みになる。
- `__call__(ctx: ToolContext[Any], input_json: str)` は `return await self._invoke_tool_impl(ctx, input_json)` に
  上書きする。基底の try / except を通さないので、deny の `PolicyViolationError` が `failure_error_function` に
  吸われない。`ctx` の注釈を基底の `__call__` と同じ `ToolContext[Any]` にするのは、SDK がコンテキスト型を
  `on_invoke_tool` の第 1 引数の注釈で選ぶためで、govern なしの invoker と同じ解決経路になる（0.17.4 の
  ソースを読んだ結果。0.22.x で同じ解決経路であることはテストで確認する）。
- `__agents_bind_function_tool__(tool)` は、束縛先が同じなら self を返す。違えば
  `inner.__agents_bind_function_tool__(tool)` で `inner` を再束縛し、同じ policy 等で新しい `_GovernedInvoker` を
  作って返す（登録は `__init__` が行う）。基底の再束縛は基底クラスのインスタンスを作り、上書きした `__call__`
  と登録が失われるので使わない。

### 2. `_govern_tool` の分岐

- 元の `on_invoke_tool` が失敗ハンドラ付き invoker のインスタンスなら、`_GovernedInvoker(inner=元,
  function_tool=None, ...)` を作り `dataclasses.replace(tool, on_invoke_tool=...)` へ渡す。replace の
  `__post_init__` が再束縛を呼び、新しい tool へ束縛した別インスタンスが作られる。これも `__init__` で登録
  されるので、返る tool は統治済みと判定される。最初に作ったインスタンスは参照が切れれば GC され、弱参照の
  削除コールバックは同じ id で後から登録されたエントリを消さない（ADR-0042 の Decision 6）。
- 元の `on_invoke_tool` が invoker でない場合（利用者が `FunctionTool` を直接組んだ場合等）は、従来の素の関数
  ラップ（コンテキスト注釈の引継ぎを含む）を使う。この場合は govern なしでも SDK の事前検証が効かないので、
  govern の有無で判定は変わらない。

### 3. 依存する SDK 非公開要素

- `agents.tool._FailureHandlingFunctionToolInvoker`（継承元で、承認強化の型判定の対象）
- そのコンストラクタ引数と内部属性（`_invoke_tool_impl` / `_on_handled_error` / `_function_tool`）
- `__agents_bind_function_tool__` の再束縛プロトコル（`FunctionTool.__post_init__` が呼ぶ）
- `_SYNC_FUNCTION_TOOL_MARKER`（timeout 設定の検証が参照する）
- 承認の事前検証を invoker の型で切り替える SDK 0.22.x の挙動
- SDK が invoker を実行する経路が `__call__` か `_invoke_tool_impl` の直接呼び出しに限られること。両方に評価を
  置いている。`_function_tool` から元の関数を辿る等の第 3 の経路が SDK に増えると迂回されうる。0.17.4 で
  `_invoke_tool_impl` を呼ぶのは `__call__` のみである。0.22.x の参照箇所は実装時に SDK ソースで全件列挙して
  確認する
- 基底クラスが `__slots__` を持たず、インスタンスを弱参照できること（0.17.4 は `__slots__` なし）

### 4. 事前検証の材料の出どころによる分岐

SDK 0.22.x の承認の事前検証が、材料（入力モデル・検証関数等）を `on_invoke_tool._invoke_tool_impl` やそこに
付いた属性から辿る実装だった場合、差し替えた `governed_impl` にはその材料が無く、govern 済みツールだけ検証が
効かないか例外になる。実装時に SDK ソースを読み、次のとおり扱う。

- 材料を invoker 本体・`_function_tool`・tool から辿る: 本 ADR の設計のまま実装する。
- 材料を `_invoke_tool_impl`（または付随属性）から辿り、属性として写せる: その属性を `governed_impl` へ
  明示的に写す。写した属性は追加の非公開依存であり、SDK バージョン耐性トリップワイヤのテスト docstring に
  列挙する。
- 写せない形（閉包内のみ等）: 本 ADR の方式は成立しない。新しい ADR を起こして方式を決め直す。

いずれの場合も、govern の有無での判定一致を確かめるテスト（Confirmation）が、govern 済みツールだけ事前検証が
効かない退行を検知する。

### 5. 変えないもの

- 統治済みの印の判定方式（登録済みインスタンスとの同一性。hash / eq に依存しない）。
- deny の着地（`PolicyViolationError` の伝播）、監査記録の時点、監査レコードの形式。
- 監査の `details.arguments` は、モデルが送った生の引数を記録する（ポリシーの照合と同じ入力）。承認判定の
  一致とは別の契約である。
- `govern_spec` が `needs_approval` 等の宣言メタを変えないこと。承認前に deny したい場合の設計指針。
- ADR-0025 の決定（`spec.tools` は build 時の実行本体ラップ・MCP 由来ツールは監査フックで評価）。本 ADR は
  関数ツールの評価点を動かさない。

## Consequences

- + govern 済みツールの条件付き承認（callable な `needs_approval`）が、govern なしと同じ判定になる。引数の
  表現を変えるだけで判定をすり抜ける逆転が無くなる。
- + deny の例外型・着地点・監査の時点を変えないので、利用者の deny の捕捉と ADR-0030 の details 契約は
  そのまま成り立つ。
- - 引数の事前検証で値が変わる呼び出し（既定値を省略した呼び出し等）では承認要求が増える。govern なしの SDK
  既定と同じ挙動に揃う結果である。
- - SDK 非公開要素への依存が増える（Decision 3）。前提の変化はトリップワイヤと govern の有無での判定一致の
  テストで検知する。宣言範囲を 1 つの minor に閉じているので、依存の変化は範囲を上げる作業の中で検査される。
- - invoker の経路では、ADR-0042 の Decision 6 のうち「`dataclasses.replace` / `copy` を経ても同じ関数を指す」
  ことと、同 ADR の Consequences の「SDK の `FunctionTool` 複製が `on_invoke_tool` を包み直さないことへの依存」は
  成り立たない。印の判定は、複製時に SDK の再束縛が作る `_GovernedInvoker` を `__init__` で登録することで保つ。
  invoker でない `on_invoke_tool` の経路では ADR-0042 の Decision 6 がそのまま成り立つ。
- - 統治済みツールを再度 govern すると（ADR-0042 の「統治済みの Agent を返す factory」）、`inner` が
  `_GovernedInvoker` になる。外側の再束縛は `inner.__agents_bind_function_tool__` を呼ぶので内側も連鎖して作り
  直され登録される。評価と `tool:` の記録は外側・内側で 1 回ずつ残り、どちらかが deny なら deny になる。
  ADR-0042 の Consequences が述べる同じ帰結である。
- - 本 ADR の形（実行本体そのものの統治・生成時の登録）は、決定時点で未試作である。実装時に SDK 0.22.x の
  一時環境で試作し、既存の governance 系テストと、ADR-0043 のすり抜け実測の入力で govern なしと一致することを
  確認したうえで確定する。

## Confirmation

本 ADR は設計フェーズで受理したものであり、以下の強制手段は実装フェーズで追加する対象である。追加・更新した
テストは `docs/QUALITY-GUARANTEES.md` へ登録する（source = ADR-0045）。個別 assert とテスト名の確定はテスト
実装時に行い、一次情報は各テストの docstring とする。

- govern 済み / govern なしの同じツール（callable な `needs_approval`）に、型変換される引数（数値の文字列表現・
  小数表現・真偽値の文字列表現、既定値なしツールでの数値の文字列表現）と対照の整数引数を渡し、承認要求の
  有無と判定関数へ渡った引数が両者で一致することを pin するテスト
- govern 済みツールの `on_invoke_tool` が SDK の失敗ハンドラ付き invoker のインスタンスで弱参照を作れ、
  `dataclasses.replace` / `copy.copy` の後も統治済みと判定され、実行時の `tool:` 監査レコードが 1 行であることを
  pin するテスト（ADR-0042 の印のトリップワイヤをこの形へ更新する）
- deny 時に `PolicyViolationError` が伝播し、`failure_error_function` の文言がモデルへ返らないことを pin する
  テスト
- 同期関数の `function_tool` を govern したとき、timeout 設定の検証結果が govern なしと同じであることを pin する
  テスト
- invoker でない `on_invoke_tool` を持つ `FunctionTool` が従来のラップで統治され、allow / deny とも従来どおりで
  あることを pin するテスト
- 依存する SDK 非公開シンボル（失敗ハンドラ付き invoker と再束縛プロトコル）が存在することを検査する SDK
  バージョン耐性トリップワイヤ（`docs/architecture.md` のトリップワイヤ節の方式に倣う）
- govern 済みツールの `on_invoke_tool._invoke_tool_impl` を直接呼んでも、deny ポリシーなら
  `PolicyViolationError` になり deny が記録されること（複製後の再束縛済みインスタンスでも同じ）を pin する
  テスト
- 統治済みツールを別ポリシーで再度 govern すると評価と `tool:` 記録が外側・内側の両方で残り、片方が deny なら
  deny になること（複製後も同じ）を pin するテスト
- 内側の関数のコンテキスト注釈が `RunContextWrapper` のものと `ToolContext` のものの双方で、govern 済み /
  govern なしのツールが受け取るコンテキストの型が一致することを pin するテスト
