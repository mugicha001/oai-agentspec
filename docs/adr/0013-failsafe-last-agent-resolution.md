# 0013: Failsafe 着地結果から実行中エージェントを参照する決定モデル（`last_agent`）

- Status: accepted
- Date: 2026-07-30

## Context

`docs/adr/0012-failsafe-declarative-landing.md` は Failsafe を「Runner の外側へ伝播した任意例外を
宣言 1 回 + `failsafe_call` で着地値へ丸める」機構として導入し、`FailsafeResult` を
`final_output` / `exception` / `matched_type` の 3 フィールドで構成した。0012 の運用後、
「着地しても会話を終わらせず、もともと実行中だったエージェントを起点に継続実行したい」という
要求が生じた。SDK は例外に `run_data`（`RunErrorDetails` 相当。`last_agent` を含む）を添付する
経路を持ち、lib 独自例外 `RunBudgetExceeded` にも同種の情報を持たせられる。本 ADR はこれらから
「実行中だったエージェント」を `FailsafeResult.last_agent` として参照できるようにする決定モデルを
定める。0012 の Status・本文は変更せず、本 ADR で補完する。

検討した選択肢:

1. **暗黙の自動導出を挟む 3 段モデル（却下）**: 「例外ごとの指定」「全体規定」に加え、
   両方無指定なら常に例外から自動導出を試みる中間段を挟む。宣言側の意図（「実行中の agent を
   使いたい」という明示）と自動導出（宣言なしでも動く）が両側に存在することになり、
   「なぜこの着地だけ last_agent が付くのか」が宣言を読むだけでは分からなくなる。解決を
   `RUNNING_AGENT` という明示的な指定値の配置に一本化し、指定が無ければ一切導出しない
   opt-in モデルの方が、宣言を読めば挙動が決まる単純なメンタルモデルで説明できる。
2. **`failsafe_call(policy, thunk, *, agent=agent)` の明示引数案（却下）**: 呼び出し側が
   実行対象の agent を毎回引数で渡す。`thunk`（`lambda: Runner.run(agent, ...)`）の内側に
   既に agent を書いているため、同じ値を呼び出し引数にも重複して書かせることになり、
   両者が食い違う（引数だけ更新し忘れる等）リスクを生む。宣言（policy）側に集約し、
   `failsafe_call` のシグネチャは変更しない方針を採る。
3. **並列マッピング `last_agents: Mapping[type[Exception], Any]` 案（却下）**: `handlers` とは
   別に例外型ごとの `last_agent` マッピングを持つ。`handlers` のキー集合と `last_agents` の
   キー集合が同期していることを利用者が手動で保証する必要があり、キーの追加・削除で
   両辞書が乖離する（片方だけ更新し忘れる）宣言の分裂を招く。`FailsafeHandler` で
   `handlers` の値位置に `fallback` と `last_agent` を同居させ、1 つの宣言単位にまとめる
   方が乖離を構造的に防げる。
4. **`FailsafeResult.to_input_list()` の自前実装案（却下）**: 着地結果から次の `Runner.run` へ渡す
   会話履歴を lib 側で組み立てるヘルパを提供する。SDK の `mode`（`"append"` 等）・reasoning item の
   扱い・input item の正規化ロジックは SDK 内部の関心事であり、lib 側で複製すると SDK の将来変更に
   静かに追随できなくなる。また `RunBudgetExceeded` のように `run_data` を持たない例外では
   会話履歴そのものが存在せず、`to_input_list()` は「履歴なし」を正直に表現できない設計になる。
   会話継続は SDK `Session` を第一選択とし、`run_data` を持つ例外に限り SDK 資産
   （`ItemHelpers.input_to_new_input_list` 等）を利用者コード側で組み合わせる案内に留める
   （`docs/usage/safety/resilience.md` 参照）。
5. **`RunBudgetExceeded` へ `run_data`（`RunErrorDetails`）の部分模倣を添付する案（却下）**: lib 独自
   例外に SDK 型と同名・同形の `run_data` 属性を持たせ、SDK 例外と同じ読み取り経路に統一する。
   `RunErrorDetails` は `input` / `new_items` / `raw_responses` / `last_agent` / `context_wrapper` /
   `input_guardrail_results` / `output_guardrail_results` の 7 フィールド契約を持ち、lib が
   `last_agent` 以外を持たない不完全な模倣を提供すると「`run_data` を持つ」という利用者の期待に
   嘘をつくことになる。`RunBudgetExceeded` は素直に `last_agent` 属性のみを追加し、解決側
   （`_derive_last_agent`）が `run_data.last_agent` -> `last_agent` の順に duck typing で読むことで
   SDK 例外・lib 独自例外の双方を同じ規約でカバーする。

## Decision

`last_agent` の決定は次の 2 段 + `RUNNING_AGENT` sentinel による opt-in 解決で構成する。

1. **段 1（例外ごとの指定）**: `FailsafeHandler.last_agent`
2. **段 2（全体規定）**: `FailsafePolicy.fallback_last_agent`

各段は「具体の agent」または公開 sentinel `RUNNING_AGENT`（「実際に動いていた Agent を使う」
ことを表す指定値）を置ける。判定は `is None` / `is RUNNING_AGENT` の同一性のみで行い、
`RUNNING_AGENT` を置いた段でのみ例外からの解決（`exc.run_data.last_agent` -> `exc.last_agent`。
いずれも `getattr` の duck typing）を試みる。解決できなければ次の段へ落ち、最後まで
決まらなければ `None` になる。何も指定しなければ解決は一切走らない（自動導出はしない）。
解決は防御的に行い、属性の読み出し自体が例外を送出しても着地（`FailsafeResult` の返却・
warning・`on_apply`）を壊さず、`logger.debug` に記録して解決不能として扱う。

シグネチャは `failsafe_call(policy, thunk)` を変更せず、決定に必要な入力（`FailsafeHandler` /
`fallback_last_agent`）はすべて宣言（`FailsafePolicy` / `handlers` の値）側に集約する。
`FailsafeResult.from_exception` は `failsafe_call` の外側（利用者が自前の except で捕捉した
場合）から同じ結果型・同じ `last_agent` の意味へ手動着地するファクトリで、policy を受け取らない
ため段 1 相当（明示指定 または `RUNNING_AGENT`）のみを持つ。

会話履歴の継続手段（`to_input_list()` 相当）は lib に実装せず、SDK `Session` を第一選択、
`run_data` を持つ例外に限り SDK 資産（`ItemHelpers.input_to_new_input_list` 等）を利用者側で
組み合わせる案内に留める。

## Consequences

- + 着地後も「もともと実行中だったエージェント」を参照でき、会話の継続実行が可能になる。
  指定しない限り挙動は変わらない（既存の 3 フィールド構築・`FailsafePolicy` の既定動作は不変）。
- + 解決は宣言（`RUNNING_AGENT` の配置）を読むだけで判別でき、暗黙の自動導出による驚きがない。
- + `handlers` とキー集合が同期しない別マッピングを持たないため、宣言の乖離が構造的に起きない。
- - `RUNNING_AGENT` を着地値位置（`FailsafeHandler.fallback` / `handlers` の値位置）に誤配置すると
  意図しない挙動になるため、build-time `ValueError` による fail-fast が必須になった
  （実装コスト。ネスト宣言の拒否と対称に扱う）。
- - `last_agent` は機微（システムプロンプト・資格情報を含みうる Agent 実体）を保持しうるため、
  `FailsafeResult` / `FailsafeHandler` / `FailsafePolicy` の該当フィールドに `repr=False` を
  追加する必要があった（監査ログ・メトリクス送信時に丸ごとシリアライズしないことは
  利用者側の責務として `docs/usage/safety/resilience.md` の落とし穴に明示する）。
- - 会話履歴の継続は lib のヘルパを持たないため、`run_data` を持たない例外（`RunBudgetExceeded` 等）
  では利用者が「履歴を再構成する手段がない」ことを認識したうえで `Session` 等の代替に頼る必要がある。
- 禁止例外列挙に `ExceptionGroup` を追加した（ADR 0012 の列挙を補完する。0012 の本文は変更しない）。
  マッチが `isinstance` である以上、`ExceptionGroup` を宣言すると `TaskGroup` 等が束ねた無関係な
  例外（バグ由来を含む）まで丸ごと着地させる広すぎる捕捉になり、`Exception` を禁止した理由と同一。
  `BaseExceptionGroup` は `Exception` 非派生として別の段で拒否されるため、両者の扱いが対称になる。
  禁止は列挙メンバーそのものに限り、利用者定義のサブクラス（捕捉範囲が限定される）は許容する。
- `FailsafeResult` は直接構築でも `__post_init__` で `exception` / `matched_type` を検証する
  （`from_exception` と同一の契約・同一のメッセージ）。同一の公開型に 2 つの検証契約が並ぶ状態を
  避けるためで、`last_agent` は不透明値として検証しない。
- `failsafe_call` の thunk 受理契約検査（`inspect.isawaitable`）は `handlers` の宣言有無より前で
  1 度だけ行う。`handlers` が空（no-op 宣言）でも lib のメッセージで fail-fast し、宣言を足した
  ときに診断メッセージが変わらない。

## Confirmation

- 決定モデル（2 段・opt-in・段の遷移・falsy 値の扱い・防御的解決・sentinel の複製耐性）の強制手段:
  `tests/runtime/resilience/test_failsafe_l1.py`（`agents` 非依存の純ロジック層検証。決定表・
  R1〜R3・S1・S2・D の各セクション）。QUALITY-GUARANTEES.md に該当行を登録済み（source =
  ADR-0013）。
- `RUNNING_AGENT` の着地値位置への誤配置（`FailsafeHandler.fallback` / `handlers` の値位置）が
  build-time `ValueError` になることの強制手段: 同ファイルの該当テスト（QUALITY-GUARANTEES.md に
  登録済み）。
- `last_agent` を保持する 3 フィールドが `repr` に出ないことの強制手段: 同ファイルの R2/S1
  セクション（QUALITY-GUARANTEES.md に登録済み）。
- SDK 隔離の強制手段: SDK 隔離 grep（`grep -rnE "(from agents|import agents)"
  src/oai_agentspec/ | grep -v _adapters` が空であること）。
