# 0029: 実行可能アクションの公開 API 形（明示コンストラクタ + `bind` が返す `ActionPlanner` + `Slot` 1 型）

- Status: accepted
- Date: 2026-08-11

## Context

実行可能アクションの宣言基盤は、利用者が最も長く触れる公開 API である。宣言の書き方・結線の
渡し方・毎ターンの呼び出し形・スロットのデータ構造という 4 つの選択が、いずれも diff からは
読み取れない。複数の案を実測込みで比較したうえで形を決めた。

### 宣言の書き方（decorator を採るか）

関数シグネチャからパラメータ宣言を導く decorator（`@catalog.action`）案は**技術的に成立する**
ことを実測で確認した。`Annotated[T, Field(...), marker]` を使えば SDK tool の schema と
`parameters_model()` の schema がバイト一致し、最短経路は 19 行に収まる。

それでも採らない理由:

| 観点 | 採らない理由 |
|---|---|
| 既存流儀 | `ToolSpec` / `AgentSpec` / `NextTurnRule` / `NextTurnPolicy` はすべて明示コンストラクタで宣言する。宣言層の書き方を 1 つに保つ |
| ADR 0001 の残る論点 | ADR 0001 却下案 2 の「Tool 定義ファイルが lib 依存になる（利用者は生の Python 関数を lib 非依存のまま散在ファイルに置きたい）」が decorator 案に部分的に当てはまる（マーカーは `agents` に依存しないが `oai_agentspec.runtime.intent` には依存する）。明示コンストラクタなら実行関数は素の Python 関数のままで済む |
| 2 経路併存の回避 | decorator を主経路にしつつ明示経路を残すと宣言の書き方が 2 つになる |

**受け入れるコスト 2 件**:

1. **型・`description` の二重記述**: パラメータ 1 件につき `param(...)` と実行関数の signature の
   2 箇所に型を書く（アクション 20 件 × 平均 3 パラメータで 60 箇所）。
2. **宣言パラメータ名と tool 引数名のずれ（F4）が残る**: decorator なら構造的に発生しない。
   明示経路では起こりうるため、build 時検証を分離先（関数実行エージェントの組み立て）が持つ。
   実測では `param("second", int)`（`s` 抜け）に対し
   `ValueError: 宣言パラメータ名と tool 引数名の不一致: ['second', 'seconds']` が組み立て時点で
   送出され、無症状にはならないことを確認した。

マーカー（`from_context()` / `predicted()`）も導入しない。`param(..., from_context=(...),
by_llm=True, ...)` のキーワード引数形で足りる。

### 結線の渡し方

`registry` / `prompts` / `guardrail_registry` / 候補の出どころ / 不足の埋め方という 3 つの関心事が
混ざる。平坦な引数列にすると「どれがどの目的の部品か」が呼び出し側から読み取れない。

結線の保持先も 2 案を比較した。当初案は「`bind()` が `ActionCatalog` 自身を変異させ、以降の
`validate()` / `plan()` が catalog のメソッドとして結線を読む」形だった。この形は
「`bind()` を呼ばずに `plan()` / `validate()` を呼ぶと `RuntimeError`」という実行時検査を
必要とし、呼び出し順を誤ると実行時まで結線漏れが検出されない。また「`bind()` を 2 回呼んだら
どうなるか」という意味論（上書きか・エラーか）を別途定義する必要が残っていた。

予測エージェントについては、当初「利用者がエージェントの実体を渡す」形も検討した。しかし
本ライブラリの原則は「利用者は `AgentSpec` を宣言し、実体は registry が遅延構築する」であり、
実体を受け取ると lib からは名前が読めず（SDK 属性へ触れないため）、エージェント名を利用者に
書かせるという不整合が生じていた。

予測された値は `Slot.value` -> `plan.input_json`（型検証）-> `Runner.run` -> ツールの引数という
経路で**実行入力になる**。型検証は通っても内容が危険な値（削除対象のパス・想定外のホスト名・
注入文字列）は通過するため、内容検査を挟む口が要る。ガードレールの渡し方は「登録名参照」と
「実体の直渡し」の 2 案を比較した。

| 観点 | 名前参照 | 実体の直渡し |
|---|---|---|
| フィールド数 | 2（`guardrails` + `guardrail_registry`） | 2（`input_guardrails` + `output_guardrails`。境界ごとに分ける必要がある） |
| 境界の判定 | `AgentRegistry._wire()` の既存 dispatch が振り分ける | lib が実体から境界を判別するには SDK 型の内省が要り、SDK 隔離に抵触する |
| タイポの検出 | 起動時検証で落ちる | 検出手段が無い |

名前解決と境界振り分けは `AgentRegistry._wire()` にあり、`DefaultAgentBuilder` 単体では起きない
ことを実測で確認した（`DefaultAgentBuilder` 経由では装着 0 件・registry 経由では装着される）。

### 毎ターンの呼び出し形

当初案は毎ターン `suggest_executable_intents` -> `plan_slots` -> `predict_params` の
**3 呼び出しを利用者に書き写させていた**。

### スロットのデータ構造

当初案は `Resolved` / `NeedsAgent` / `NeedsConfirmation` / `NeedsUser` の 4 クラスを公開し、
`state` を文字列で、値の出どころを `"run_context:<パス>"` / `"agent:<名前>"` という**連結文字列**で
契約していた。利用者が接頭辞を切り出すコードを書く必要があり、実際にレビュー工程で
`s.origin.startswith("user:")` が `origin is None` の状態で `AttributeError` になる誤りが 1 度発生した。

### 用語の不統一

「LLM で値を埋める」という 1 つの概念が、宣言 `param(..., by_agent=True)`・結線 `llm_filler`・
値の出どころ `Origin.AGENT`・スロット状態 `SlotState.NEEDS_AGENT` と複数の名前で呼ばれていた。
さらに `agent` という語は本ライブラリでは `action_agent`（実行エージェント）・
`action_next_turn_agent`（次ターンの窓口）という別概念を既に指しており、`by_agent` が
「実行エージェントが埋める」と誤読される余地があった。

### 却下した中間案

- decorator を採る案（宣言 decorator あり・パラメータのみ明示の中間形を含む）。
- 4 クラスの判別付き union（`Field(discriminator="state")`）を維持したまま `origin` だけを
  enum 化する案。文字列契約は 1 つ減るが、公開シンボル数と「どのクラスがどのフィールドを持つか」の
  学習コストが残り、`plan.pending` のような横断的な導出プロパティが union 型を跨ぐため書きにくい。
- `plan()` を「便利メソッド」として追加し 3 関数を公開のまま残す案。同じことをする経路が 2 つに
  なり、どちらを使うべきかの判断を利用者へ押し付ける。計測基準テストも 2 経路ぶん必要になる。
- 候補列を直接受け取る引数（`plan(query, candidates=...)`）を足す案。候補の出どころが 2 つになり、
  allowlist 除外・WARNING の経路が分岐する。自作ランキング結果を渡したい利用者は
  `CandidateGenerator` Protocol を数行で実装すればよい。
- catalog 変異型 `bind`（`catalog.bind(...)` が catalog を変異させ、`validate()` / `plan()` を
  catalog のメソッドに置く案）。却下理由: (a) 本設計は「実行時検査より、間違いを構造的に不可能に
  する」を一貫原則としており（連結文字列の enum 化・`from_user` の導出・予測エージェントの内製化が
  同じ方向）、API の入口だけが「順序を誤ると実行時 `RuntimeError`」という実行時検査に依存するのは
  原則の不徹底である。結線済みオブジェクトを返す形なら、未結線の `plan()` は「呼べない」
  （`ActionCatalog` にメソッドが存在しない）。(b) 前例として引いていた `AgentRegistry` は DI を
  **コンストラクタ**で受けており（`registry.py:50-55`）、後から変異させるメソッドではない。
  前例はむしろ「結線を受けた不変オブジェクトが仕事をする」形を支持する。(c) 結線済みオブジェクトを
  返す形なら、テストで同じ宣言簿を別結線（別 registry・別 generator）で使い回せる。
  「`bind()` を 2 回」の意味論も「独立した結線オブジェクトが 2 つできる」という自明な形に解ける。
  frozen と `validate()` 実行済みフラグ（`plan()` 初回の自動 validate を 1 度にする仕組み）の
  両立は pydantic の private attribute（公開フィールドに現れない可変キャッシュ）で成立することを
  実測で確認した。
- 押下後の実行まで面倒を見る実行ヘルパー（`run_action(plan, ...)`）。build-don't-run の逸脱が
  5 例目に増える（ADR 0026 で 4 例目に増やす判断と同じターンでもう 1 件増やすのは原則の空洞化）。
  実行は 1 行で書ける。
- `apply` 時に確定値を `run_context` へ自動で書き戻す案。`run_context` は利用者の任意型
  （不透明値）であり、frozen dataclass / Mapping / 属性アクセスのどれであるかを lib は知らないため
  汎用 setter が構造的に書けない。frozen なら書き込み自体が失敗する。`from_context` を
  読み取り専用と定義した用語定義とも矛盾する。

## Decision

1. **宣言は明示コンストラクタ（`ActionSpec` + `param`）を維持し、decorator を採らない。**
   上記のコスト 2 件を受け入れる。
2. **結線は `ActionCatalog.bind` へ一元化し、`bind` は catalog を変異させず frozen な結線済み
   オブジェクト `ActionPlanner` を返す。関心事は frozen 宣言型へ分ける。**
   シグネチャは `bind(*, registry, prompts=None, guardrail_registry=None, candidates=None,
   llm_filler=None)` の 5 引数で、`CandidateSource`（`generator` / `context_builder` /
   `history_limit`）と `LLMFiller`（`model` / `on_invalid_response` / `guardrails`）が関心事を
   束ねる。前例は `AgentRegistry.__init__(agent_builder=...)`（DI をコンストラクタで受け、
   受けた後は変異させない）と、既存の `SessionPolicy` / `CompactionConfig` / `IntentPolicy` /
   `NextTurnPolicy` である。`bind` は DI 対象と bind 時点の宣言簿スナップショットを
   `ActionPlanner` へ載せるだけで実行しない。両宣言型の検証（`context_builder` と
   `history_limit` の排他・`on_invalid_response` の `Literal`）は各型の validator が持ち、
   `bind` まで持ち越さない。
   - **(0) `ActionCatalog` は純粋な宣言簿に戻る。** 公開メソッドは `register` / `names` /
     `get` / `bind` の 4 つで、`validate` / `plan` は `ActionPlanner` 側の公開メソッドである。
     未結線のまま `plan()` を呼ぶ形は構造的に存在しない（`ActionCatalog` にメソッドが無い）。
     `bind()` を複数回呼ぶと独立した `ActionPlanner` が複数できるだけであり、bind 後の
     `register()` は既存の `ActionPlanner` に影響しない。
   - **(0a) 戻り値の型名は `ActionPlanner` とする。** 既存の型名は役割で名付けられており
     （`CandidateSource` = 候補の出どころ・`LLMFiller` = 不足の埋め方・`AgentBuilder` /
     `ContextBuilder`）、`BoundCatalog` のような「作られ方」による名前は流儀から外れる。
     `planner.plan(query)` が生成物 `ActionPlan` と対応して読め、`ActionSpec` / `ActionCatalog` /
     `ActionPlan` と同じ `Action` 接頭辞で族が揃う。`ActionPlanner` の frozen と `validate()`
     実行済みフラグ（`plan()` 初回の自動 validate を 1 度に抑えるキャッシュ）の両立は、公開
     フィールドに現れない private な可変キャッシュで実現する（実測で成立を確認済み）。
   - **(a) 予測エージェントの実体は lib が構築する。** 利用者が渡すのは `LLMFiller(model=...)` の
     `model` のみで、`runtime/intent` が `AgentSpec(name=<lib 固定名>, instructions=<合成済み>,
     model=<利用者の model>)` を宣言し、`_adapters` が実体化する。エージェント名は lib の宣言定数
     であり、利用者に書かせない。「予測エージェントは業務エージェントとは別に置き `session` を
     渡さない」は、利用者が実体に触れられないため**構造的に保証される**。
   - **(b) `LLMFiller(spec=...)` の逃げ道と `max_turns` の公開は見送る**（YAGNI）。`max_turns` は
     内部定数 `1` とし、将来ツールを持たせる判断をしたら公開契約を変えずに内部で上げる。
   - **(c) 予測値の内容検査は `LLMFiller.guardrails`（登録名の tuple・既定 `()`）で opt-in させ、
     実体の直渡しは受けない。** 解決簿は `bind(guardrail_registry=...)` 側に置く（`registry` /
     `prompts` と同列）。新規機構は作らず既存 `runtime/guardrails` と `AgentSpec.guardrails` を
     そのまま使う。解決は**予測エージェント専用の `AgentRegistry`** で行い、`_wire()` の名前解決と
     境界振り分けを再利用する（dispatch を二重実装しない・要件「業務エージェントとは別の registry に
     置く」を文字どおり満たす）。不整合（解決簿の欠落・登録名のタイポ）は起動時検証で落とす。
   - **(d) ガードレール発火時は SDK 例外を伝播し、`on_invalid_response="skip"` の後退を適用しない。**
     `on_invalid_response` が扱うのは「応答が壊れている」であり回復が妥当な事象だが、ガードレール
     発火は「危険な内容を検出した」安全事象であり、既定値での実行続行は宣言意図に反する。後退が
     必要な利用者は `failsafe_call` で `planner.plan()` を包める。
3. **毎ターンの 3 関数を `await planner.plan(query, *, predict=True, detail=False)` へ畳み、
   公開シンボルから外す**（実装は各モジュールに残す）。低レベル用途は 2 引数で賄う。`predict=False`
   は予測段を実行せず（候補生成器が非 LLM なら全経路で LLM 0 回）、`detail=True` は
   `PlanResult(plans, suggestion, usage)` を返して `report` / `metadata` / 使用量を捨てない。
   `detail=False` は戻り値を絞るだけで情報を捨てる設計ではない。
4. **スロットは `Slot` 1 型 + `SlotState` / `Origin` の 2 enum へ畳む**（公開シンボルは 4 -> 3）。
   連結文字列を `origin`（enum）+ `detail`（パス / エージェント名）の 2 フィールドへ分解し、
   利用者が接頭辞を切り出すコードを書かないようにする。1 型化で失われる構造的制約
   （4 クラスでは `NeedsAgent` に `origin` フィールドが存在しなかった）は `Slot` の
   `model_validator` へ移し、状態とフィールドの整合 6 条件を強制する。
5. **穴埋め後の書き戻しは lib の責務にせず、`slot.from_user` を材料として公開する。**
   `origin` が `USER_INPUT` / `USER_CONFIRMED` のいずれかという enum 判定を利用者に書かせず、
   `origin is None` でも `False` を返すため、当初案で実際に起きた誤りが構造的に発生しない。
   次ターンでの再利用は `param(..., from_context=(...))` の宣言で表現する。
6. **必須結線が欠けた場合の規則を 3 つ定める。** `candidates=None` で得た `ActionPlanner` の
   `plan()` は `RuntimeError`（`validate()` だけの利用は妨げない）、`prompts=None` かつ
   セグメント宣言ありでの `validate()` は `RuntimeError`。**`llm_filler=None` のみ例外ではなく
   スキップ**とし、決定的段の結果と `ParamUsage(runs=0, ...)` を返す。`LLMFiller` を渡さないこと
   自体が「穴埋め経路を持たない構成」という利用者の明示的な意思表示であり、例外にするとその
   意思表示が使えなくなる。段階リリース（予測段の実装前でも `plan(query)` を既定引数のまま呼べる）と
   `predict` の既定値 `True` の両立でもある。当初 4 つ目としてあった「`bind()` 未呼び出しの
   `plan()` / `validate()` は `RuntimeError`」は、Decision 2 (0) により構造的に不可能となり
   規則として不要になった。
7. **`ActionSpec.action_agent` を改名しない。** 改名すると隣接する 2 型で同名フィールドが別物を
   指す（`spec.agent` は名前・`plan.agent` は実体）構図になる。撤回後は「`action_agent` は常に
   名前（`str`）、`agent` は registry で解決した実体」という規則が全型で一貫する。
   - **(a) 公開面を削減する。** 判定基準は「**ライブラリ側の知識を含むもの・生データは残す。
     利用者が書ける単なる別名は落とす**」。`ActionPlan` から `agent`（registry の選択を隠すため・
     ADR 0026 と同じく実行は利用者が書く）/ `needs_agent` / `needs_user` / `parameters` を落とし、
     `Slot` から `resolved`（`state` と同値であり状態を 2 通りで表現する）を落とす。`slots_model()` は
     予測が内部へ畳まれた結果、利用者が呼ぶ場面が無くなったため非公開にする（`parameters_model()` は
     UI のフォーム生成という外向きの用途があるため公開のまま残す）。`ready` / `pending` /
     `from_user` は状態集合・enum 判定の**定義**を含むため残す。

8. **「LLM で値を埋める」概念の名前を `llm` へ統一する。** 宣言は `param(..., by_llm=True)`、
   値の出どころは `Origin.LLM`、スロット状態は `SlotState.NEEDS_LLM` とする（`by_agent` /
   `Origin.AGENT` / `NEEDS_AGENT` は採らない）。1 概念 1 名の統一に加え、`agent` という語を
   `action_agent`（実行エージェント）系の概念へ専有させ、「実行エージェントが埋める」という
   誤読を断つ。`NEEDS_AGENT` を揃えの対象に含めるのは、同状態が「`by_llm=True` のパラメータが
   未充足」という同一概念の状態表現であるため（`needs_agent` 導出プロパティは既に公開面から
   削減済みで、影響はミラーの enum 値のみ）。
   - **(a) `Origin.LLM` の `detail` は `None` とする。** 旧 `Origin.AGENT` の `detail` は
     予測エージェント名だったが、予測エージェントの内製化（Decision 2 (b)）により名前は
     ライブラリ内部の固定定数となり利用者にとって情報量が無い。載せると内部定数名が
     `model_dump(mode="json")` の公開契約へ漏れ、内部名の変更が契約変更になる。`detail` は
     `str | None` のまま残るため、将来情報を載せる判断（例: モデル識別子）は非破壊の拡張で
     行える（逆方向の削除は破壊的であり、YAGNI 側へ倒す）。これに伴い `Slot` validator の
     条件 6 は「`detail` 非 `None` は `origin` が `RUN_CONTEXT` のときのみ」へ縮小する
     （6 条件の構成自体は不変）。
9. **`param(..., extra={})` を削除する。** 要件書の AC 120 件のいずれも `extra` を消費して
   おらず、消費者のいない拡張口である。`LLMFiller(spec=...)` の逃げ道・`max_turns` の公開を
   見送った Decision 2 (b) と同じ YAGNI 基準を適用する。必要になった時点での既定値付き
   フィールド追加は非破壊で行える。

現在仕様の SoT は `docs/architecture.md`（「意図予測（`runtime/intent`）」節の
「実行可能意図の宣言と決定的スロット確定」小節）とし、本 ADR は判断・却下案のみを記録する。

## Consequences

- + 宣言層の書き方が明示コンストラクタ 1 つに揃い、実行関数は素の Python 関数のままでよい。
- + 結線が 1 箇所（`bind`）に集約され、DI 注入点が読み取りやすい。`llm_filler` を渡さなければ
  穴埋め経路が存在しないことがコードから読め、従量課金の発生条件が API 形に現れる。
- + 未結線の `plan()` / `validate()` が構造的に呼べなくなり、「bind 忘れの実行時
  `RuntimeError`」という検査と、その検査を固定するテストが不要になる。同じ宣言簿を
  別結線でテストに使い回せる。
- + 「LLM で埋める」概念の名前が宣言・結線・出どころ・状態の 4 箇所で `llm` に揃い、
  `agent` は実行エージェント系の概念だけを指す。
- + 予測エージェントの分離（業務エージェントと別・`session` 非伝播）が構造的に保証される。
- + 毎ターンの呼び出しが 1 行になり、順序の書き写しと「呼び忘れると永久に `ready=False`」という
  無症状の footgun が消える。
- + `state` / `origin` の文字列契約と連結規則が消え、利用者が接頭辞を切り出すコードを書かない。
- - 型・`description` の二重記述（パラメータ 1 件につき 2 箇所）が残る。
- - 宣言パラメータ名と tool 引数名のずれは型では防げず、分離先の build 時検証に依存する。
- - `LLMFiller` にツール付きエージェントを渡す逃げ道が無いため、必要になった時点でフィールド追加
  （非破壊）が要る。

## Confirmation

強制手段（`tests/_adapters/test_intent_adapter_l2.py` のみ既存ファイルへの追記。他は**新規作成**）:

- `tests/runtime/intent/test_catalog_l1.py`: `bind()` が frozen な `ActionPlanner` を返し
  catalog を変異させないこと / `ActionCatalog` が `validate` / `plan` を持たないこと /
  `bind()` 2 回で独立した `ActionPlanner` が 2 つでき、bind 後の `register()` が既存の
  `ActionPlanner` に影響しないこと / `candidates` 未結線の `plan()` が `RuntimeError` /
  `prompts` 未結線 + セグメント宣言ありで `validate()` が `RuntimeError` / `llm_filler` 未結線で
  予測段をスキップし `ParamUsage(runs=0, ...)` を返すこと / `plan()` 初回に `validate()` が
  1 度だけ走ること（frozen のまま実行済みキャッシュが機能すること） / `bind` が実行しないこと。
- `tests/runtime/intent/test_binding_l1.py`: `CandidateSource` の排他 validator・`LLMFiller` の
  `Literal`・両型の frozen。
- `tests/runtime/intent/test_slots_l1.py`: `Slot` の `model_validator` が 6 条件を強制すること
  （9 通りの組み合わせを parametrize し、8 通りが `ValidationError`・意図的に通す
  `RESOLVED` + `value=None` の 1 通りのみが成功する。`RESOLVED` + `origin=None` と
  `NEEDS_CONFIRMATION` + `suggestions=()` の 2 ケースを含める）/ `slot.from_user` が
  `origin=None` で `False` を返すこと / `plan.pending` が `NEEDS_CONFIRMATION` と `NEEDS_USER`
  のみを返すこと / `Slot.model_dump(mode="json")` に `state` が必ず載ること /
  **`ActionPlan` が `AgentRegistry` を参照するフィールド・プロパティを持たないこと**
  （`plan.agent` を生やす回帰を防ぐ。実体を lib 側で抱えると到達記録と `is_enabled` ゲートの
  双方が欠落する）。
- `tests/runtime/intent/test_catalog_plan_l2.py`: `predict=False` で LLM 0 回 /
  `detail=True` が `PlanResult(plans, suggestion, usage)` を返し `report` / `metadata` を
  捨てないこと。
- `tests/_adapters/test_intent_adapter_l2.py`（既存ファイルへ追記）: 予測エージェント専用
  `AgentRegistry` で `AgentSpec` から実体化して 1 回走らせること・`max_turns` が内部定数 1 で
  あること・`guardrails` の登録名が境界へ振り分けて装着されること・`guardrails` 空なら装着 0 件で
  あること・発火時に SDK 例外が伝播し後退しないこと。

`docs/QUALITY-GUARANTEES.md` に登録済み（source = ADR-0029）。
