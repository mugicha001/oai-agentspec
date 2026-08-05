# 0023: run スコープの instructions 追記と canary トークンの run ごと再解決

- Status: accepted
- Date: 2026-08-05

## Context

システムプロンプト本体は静的な `str` のまま保ちたい一方で、カナリートークンのように run ごとに
値が変わる断片をプロンプトへ埋め込みたい要求がある。同時に、出力側でその漏洩を検知する
`canary_guardrail` も、run ごとに変わるトークンを逐語照合できる必要がある。

現行の資産では次の 2 点が満たせない。

- `AgentSpec.instructions` を callable にすると instructions 全体が動的になり、静的本文の
  `str` 性（外部での再利用・最適化対象としての抽出）が失われる。利用側は「全体 callable 化 +
  独自属性で静的部分を退避」という回避策を取ることになる。
- `canary_guardrail` が受ける検知器の契約は `Callable[[str], Detection]` であり、run
  コンテキストが届かない。run ごとに変わるトークンを扱うにはプレフィクス正規表現照合のような
  劣化した近似へ落ちる。

新規設計の前に、既存資産・SDK 標準機能で同じ目的を満たせるかを検討した。

- **SDK 動的 instructions（`(context, agent) -> str`）**: 単体では上記の欠陥そのもの。ただし
  **合成後の実行経路としては採用**する（build が合成した callable を SDK のこの機構へ乗せ、
  lib 独自の実行機構を作らない）。
- **`PromptStore.compose(vars=callable)`（ADR 0005）**: 却下。`compose` は build 時評価で
  `RunContextWrapper` を受け取らない。合成結果の `str` を静的本文として渡す互換は維持する。
- **`Agent.prompt` / `DynamicPromptFunction`**: 却下。Responses API 専用で instructions とは
  別系統。
- **`predicate_guardrail` / `predicate_detector`**: 却下。契約が `Callable[[str], ...]` で
  context が届かない。
- **`unwrap_run_context`（`_adapters/run_context.py`）**: 却下。同ヘルパの契約は
  「`Runner.run(context=...)` へ forward する前に開く」用途であり、追記関数・resolver へは
  SDK 鏡写しで wrapper のまま渡す。
- **`make_arrival_gate` の「callable なら `(context, agent)` で呼ぶ」規約 /
  `validate_instructions_callable` / `_call_detect` の awaitable 正規化**: いずれも踏襲・再利用
  する（引数規約・arity 検証・await 正規化）。

検知側の接着経路として検討した選択肢:

1. **検知器の契約自体を context 対応へ広げる（単体では却下）**: 既存 detector 契約
   `Callable[[str], Detection]` を破壊し、単独利用（webhook 等）の DX も損なう。
2. **接着層に context 対応の guardrail ビルダを 1 本追加する（採用）**: 既存 detector 契約と
   固定値経路を不変のまま残し、resolver 経路だけを並走させられる。

## Decision

`AgentSpec.instructions_append`（埋め込み側）と `canary_guardrail` の resolver 受理（検知側）を
互いに独立した汎用機構として追加する。両者はカナリア用途に限定せず、片方だけの利用も成立する。

**build-don't-run の例外には当たらない**。追記関数の合成 callable は SDK の動的 instructions
機構へ載せて `Agent.get_system_prompt` が評価し、resolver は SDK が呼ぶ guardrail 実行の中で
評価される。いずれも評価主体は SDK であり、lib は宣言 → build 時結線に徹して独自の実行ループを
持たない（`./CLAUDE.md` の例外リストは 3 件のまま増やさない）。

主要な設計判断:

1. **追記関数の引数規約は `(context, agent)` 2 引数で、context は `RunContextWrapper` のまま
   渡す**。既存の動的 instructions / `is_enabled` / `make_arrival_gate` と同一の SDK 鏡写し規約に
   揃え、利用側は `ctx.context.<attr>` で run context を開く。arity 検証は
   `validate_instructions_callable` を要素ごとに再利用し、既存 instructions と同じく registry
   登録時（`_validate_spec`）に行う。
2. **`instructions` が callable のときの追記併用は拒否する**。検出は registry 登録時へ前倒しし
   （既存 instructions の arity 検証と同じタイミング）、`build_agent` にも同一チェックを防御と
   して置く（custom builder / registry を経由しない直接 build 対策）。要求は「静的 `str` +
   追記」のみであり、callable 本体との合成は sync / async 混在の意味論を複雑化させるため受理
   しない（エラーメッセージで instructions 全体の callable 化を案内する）。`instructions=None`
   + 追記のみは許容する。
3. **追記関数が送出した例外は伝播させる（fail-fast）。縮退しない**。カナリア埋め込みの失敗を
   ログ + 断片省略で縮退させると、検知側の guardrail が沈黙して漏洩検知が無意味になる
   （セキュリティ機能の silent degradation）。run を落として顕在化させる方が安全であり、
   縮退しないためログ要件も発生しない。戻り値が `str` でない場合は `TypeError`。
4. **追記関数は async を許容する**。合成 callable を常に `async def` で生成し、各追記の戻り値が
   awaitable なら await する（`_call_detect` と同型）。SDK は async の instructions callable を
   await するため追加機構は不要で、sync 限定にしても実装は簡単にならない。合成 callable は
   SDK 側のパラメータ数検査（厳密に 2 個）を満たすため、名前付き 2 引数で生成する
   （`*args` 形は不可）。
5. **`SandboxAgentSpec` では `instructions` 側にのみ自動適用し、`base_instructions` には適用
   しない**。build 経路が `instructions` の kwargs を共有するため sandbox でも追記は効くが、
   `base_instructions` は SDK 既定文との関係が別系統であり本 ADR の対象外とする（動的にしたい
   場合は callable を直接渡す）。
6. **canary resolver の契約は `(context, agent) -> str | Iterable[str] | None`** で、context は
   `RunContextWrapper` のまま渡す（判断 1 と同じ理由）。構築時は arity の bind 検証のみを行い、
   resolver を評価しない。
7. **検知側の接着は接着層に context 対応の output guardrail ビルダを 1 本新設して行う**。
   factories 側は「呼び出しごとに resolver を評価し、その値で検知器を組んで逐語照合する」
   クロージャを渡す。既存 detector 契約と固定値経路は不変のまま互換を保つ。context 対応版は
   output 境界のみ新設する（カナリアは出力専用。他境界は必要になった時点で追加する）。
8. **resolver が `None` / 空を返した場合は発火しない（`triggered=False`）**。「この run には
   カナリアが無い」状態として扱い、既存の空文字スキップと整合させる。
9. **resolver の公開契約は同期のみとする**。トークン取得は run context の属性読み出しで I/O を
   伴わない。接着層の内部実装は awaitable 正規化を持つ形にしてよいが、公開契約としての async
   許容は行わない（開放する場合は新 ADR で記録する）。
10. **糖衣ヘルパー（「context 属性名 + テンプレート」から追記関数を作る helper）は追加しない**。
    追記関数は 1 行のラムダで表現でき削減効果が小さく、テンプレート引数を lib API に持つと
    プロンプト非同梱原則と摩擦し、属性パス指定のミニ DSL 化は過剰である。使い勝手は
    `docs/usage/` のレシピ掲載で補う。

11. **`instructions_append` の容器型検証は要素ループより前に置く**。`Sequence` 判定を要素の走査より
    先に済ませることで、使い切り容器（generator / iterator）が検証ループで消費される事態を防ぐ。
    消費されると build 側で追記が無言で 0 件になり、カナリア埋め込みの消失＝検知側の恒久的な
    fail-open（漏洩を検知できないまま run が成功し続ける）になる。宣言時に fail-loud で弾く方が
    安全であり、`list` / `tuple` 以外の容器を受理する DX 上の必要もない。エラーメッセージへ値を
    載せないのは、カナリア埋め込み文を取り違えて渡した場合にトークンがログ・例外文へ漏れるため。
12. **`instructions_append` を宣言した spec の APO は早期 fail-closed で拒否する**。代替案として
    lightning 側で追記を折り込む（候補テキスト + 追記の合成を build が組み立てる）案を検討したが
    却下した。折り込むと `OptimizeResult.prompt` が rollout 時の実 instructions と一致しなくなり、
    「返されたテキストがそのまま実行されたプロンプトである」という契約が drift する。追記を持つ
    spec は最適化対象から外すか、利用者が `build=` で合成責務を明示的に引き受ける形に倒す。
    構築時（`prompt_slot` 呼び出し時）に倒すのは、rollout 中の `build_agent` まで遅延させると
    判断 2 の併用拒否エラーが lib 生成 callable を指して原因不明になるため。
13. **canary resolver の戻り値は `str` / `Iterable[str]` / `None` に限定し、`Mapping` を明示的に
    拒否する**。dict は iterable であるため素朴な型検査を通過するが、反復されるのはキー列であり、
    値側にある実トークンが一切照合されない。例外も警告も出ないまま「検知しないカナリア」が
    成立する（fail-open）ため、iterable 判定より先に `Mapping` を弾く。`bytes` / `bytearray` も
    同様に「バイト列を反復して整数が出る」誤りを防ぐため拒否する。メッセージへ解決値を載せない
    理由は判断 11 と同じ。要素検査の前に tuple 化するのは、使い切り iterable を検査で消費して
    照合対象を空にしないため（判断 11 と同型の fail-open 回避）。
14. **async resolver は構築時に `ValueError` で拒否する**。判断 9 の「公開契約は同期のみ」を
    実行時の暗黙の失敗に委ねず、宣言時点で倒す。検査は関数自体（`inspect.iscoroutinefunction`。
    `functools.partial(async def)` も解ける）と `type(resolver).__call__` の両方に掛ける。
    `async def __call__` を持つ callable object は前者だけでは検出できず、受理すると検知時に
    coroutine オブジェクトが str として扱われて照合が常に不一致になる（fail-open）。同期
    `__call__` を持つ callable object は正当な利用形態なので受理する。

空文字列の意味論: `instructions=None` かつすべての追記が `""` を返した場合、合成 callable は
`None` ではなく空文字列 `""` を返す（連結結果をそのまま返し、`None` へ畳み込まない）。

公開契約への影響: `AgentSpec.instructions_append` は kw_only の新規フィールド（既定は空）で
位置引数の束縛は不変。`canary_guardrail` の第 1 引数は既存の `str | Iterable[str]` に callable が
加わるのみで、既存呼び出しは経路ごと不変。コア `__all__` は不変。

## Consequences

- + 静的本文を `str` のまま保ったまま、run ごとに変わる断片を宣言的に追記できる（利用側の
  「全体 callable 化 + 独自属性」回避策が不要になる）。
- + 検知側が run ごとにトークンを再解決するため、逐語照合のまま run 単位のカナリアを扱える
  （正規表現による近似照合への劣化が不要になる）。
- + 追記側と検知側が独立した汎用機構であり、カナリア以外の run スコープ断片・run スコープ
  検知にも使える。
- - `instructions` が callable の場合に追記を併用できない非対称が生まれる（登録時 `ValueError`
   で明示する。将来の緩和余地は残る）。
- - 追記関数の失敗が run 全体を落とす（意図的な fail-fast のトレードオフ）。
- - 検知側に固定値経路と resolver 経路の 2 本が並走する（既存契約の互換維持と引き換えの
   実装分岐）。
- - 追記の容器が `list` / `tuple` に限定され、generator / set を直に渡せない（fail-open 回避と
   引き換えの受理幅の狭さ。利用側は `list(...)` で包む）。
- - 追記を宣言した spec は APO の対象にできない（`build=` の明示が必要になる）。
- - resolver の戻り値・同期性の検査が増え、`Mapping` を返す実装・`async def` の resolver は
   受理されない（いずれも fail-open を作る形であり、意図的な非受理）。

## Confirmation

本 ADR の判断は以下のテストで強制する。

- 追記関数が build / 登録の時点では評価されず、run ごとに再評価されること:
  `tests/_adapters/test_instructions_append_l2.py::test_append_functions_are_not_called_at_build`
  （呼ばれたら失敗する sentinel callable による「build 時未評価」pin）/
  `::test_fragments_are_reevaluated_per_run`（run を 2 回行って system prompt に異なる値が
  入ることの検証）。
- canary resolver が構築 / 登録の時点では評価されず、検知呼び出しごとに再解決されること:
  `tests/runtime/guardrails/test_canary_resolver_l2.py::test_構築時にresolverは評価されない` /
  `::test_検知呼び出しごとにresolverが再評価されトークンが切り替わる`。facade 経路は
  `tests/runtime/guardrails/test_canary_resolver_facade_l2.py::test_facade登録の実体はrunごとのトークンで照合する`。
- 上記の評価タイミング 2 件は、保証対象を壊す変異（追記の合成結果のキャッシュ / build 時の 1 度
  評価 / resolver 解決値のキャッシュ / 構築時の 1 度評価）を注入して当該テストが RED になることを
  実行確認済みであり、強制手段が実体で pin されていることを確認している。
- 追記要素の arity 検証・callable 本体との併用拒否・容器型 / 要素型検証: `tests/test_instructions_append_l1.py`
  （宣言層の `ValueError` を pin）と
  `tests/_adapters/test_instructions_append_l2.py::test_register_and_build_share_identical_error_message`
  （registry 登録時と `build_agent` の両層が同一文言で弾くことの検証）。
- 追記を宣言した spec の APO 未サポートの早期拒否:
  `tests/runtime/lightning/test_slots_l1.py::test_prompt_slot_rejects_target_spec_with_instructions_append`。
- 評価タイミングの 2 件は `docs/QUALITY-GUARANTEES.md` に source = ADR-0023 として登録している
  （相互参照）。
