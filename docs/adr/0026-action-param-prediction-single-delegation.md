# 0026: パラメータ予測の 1 回委譲を build-don't-run の 4 例目として境界を切る

- Status: accepted
- Date: 2026-08-11

## Context

実行可能アクションの宣言基盤（`runtime/intent` の `ActionCatalog`）では、決定的に埋まらなかった
パラメータを LLM に予測させる段が必要になる。この段は `Runner.run` を駆動するため、lib の
不変条件である build-don't-run（宣言・build-time 検証・薄い結線に徹し、実行は SDK `Runner.run` に
寄せる。公開の実行 API を持たない）からの逸脱にあたる。既存の逸脱は
`fit_ml_estimator`（ADR 0004）/ `failsafe_call`（ADR 0012）/ observability のグローバル結線
（ADR 0022）の 3 例であり、本件を 4 例目として追加してよいか、追加するならどこまでを逸脱範囲と
するかを決める必要があった。

逸脱一覧は「実行を 1 回駆動する関数の数」として不変に保つ整理（ADR 0014 Decision 9）であり、
1 つの機能追加で複数件増やさないことも判断の前提とした。

検討した選択肢:

1. **候補ごとに 1 回ずつ予測を呼ぶ（却下）**: 実装は単純だが、候補件数に比例して `Runner.run` の
   回数と従量課金が増える。「押下の瞬間の待ち時間と課金をゼロにするために事前予測へ寄せる」という
   本機能の狙いに対し、事前予測側のコストが候補数に比例するのは割に合わない。
2. **strict structured output（`output_type=`）を使う（却下）**: 予測応答の型付けは SDK 側の
   structured output に寄せられるが、既存 `_adapters/intent.py` の `run_intent_prompt` は
   `str` を返す形で SDK 隔離を保っており、`output_type=` を使うと lib 側がスキーマモデルを SDK へ
   渡す経路が増える。加えて `max_length` を付けた派生モデルは上限超過を切り捨てではなく
   `ValidationError` にするため、`max_suggestions` の「超過分は切り捨て」という契約と噛み合わない。
   raw `str` を受けて lib 側で parse・sort・truncate する既存経路を再利用する。
3. **予測段に再試行を内蔵する（却下）**: 再試行・タイムアウト・フォールバックは
   `runtime/resilience`（`ModelRetryPolicy` / `failsafe_call`）が担う関心事であり、予測段に
   独自の再試行ループを持たせると同じ関心事の実装が 2 箇所になる。逸脱範囲も「1 回駆動」から
   「実行ループを持つ」へ質的に広がる。
4. **押下後の実行まで面倒を見る実行ヘルパー（`run_action(plan, ...)`）を足す（却下）**:
   逸脱が 5 例目に増える。実行は `Runner.run(registry.get(plan.action_agent), input=plan.input_json)`
   の 1 行で書けるため、逸脱を増やして得られるものが小さい。
5. **`ActionPlanner.plan()` の内部でパラメータ予測を 1 ターンあたり 1 回だけ駆動する（採用）**:
   不足パラメータを持つ候補が 1 件でもあるときに限り、全候補・全不足パラメータを 1 本の
   プロンプトへ合成して `Runner.run` を 1 回だけ駆動する。不足が無ければ 0 回、`predict=False`
   なら 0 回。

## Decision

build-don't-run の逸脱に `ActionPlanner.plan()` 内のパラメータ予測を 4 例目として加え、逸脱範囲を
次のとおり限定する。

- 駆動するのは 1 ターンあたり `Runner.run` **ちょうど 1 回**であり、候補件数・不足パラメータ件数に
  比例しない。不足が 0 件、または `predict=False`、または `llm_filler` を結線していない場合は
  0 回である。
- lib 独自の実行ループ・再試行・タイムアウト・モデル切替を持たない。再試行等が必要な場合は
  `runtime/resilience`（`ModelRetryPolicy` / `failsafe_call`）へ委ねる。
- SDK 接触は `_adapters/intent.py` の `run_filler_prompt` 1 関数に閉じる。`max_turns` は
  同モジュールの内部定数 `1` で固定し、公開引数にしない。
- 押下後のアクション実行は lib の責務ではなく、利用側が
  `Runner.run(registry.get(plan.action_agent), input=plan.input_json)` を書く。lib は実行 API を
  持たない。
- `./CLAUDE.md`「設計の核」の build-don't-run 項目と `docs/architecture.md` の逸脱一覧に
  4 例目として明記し、逸脱範囲を grep 的に検知可能な状態に保つ。

現在仕様の SoT は `docs/architecture.md`（「意図予測（`runtime/intent`）」節の
「実行可能意図の宣言と決定的スロット確定」小節）とし、本 ADR は判断・却下案のみを記録して
仕様詳細を重複させない（ADR 0004 と同じ分担）。

## Consequences

- + 事前予測のコストが候補件数に依存せず、1 ターンあたり LLM 1 回で上限が読める。
- + 押下の瞬間は LLM 0 回であり、待ち時間も従量課金も発生しない。
- + 逸脱範囲が「1 回駆動」に閉じるため、将来 lib が実行エンジン化しようとした場合に
  `CLAUDE.md` の例外文言の範囲逸脱として検知できる。
- - build-don't-run の逸脱が 3 例から 4 例へ増える（意図的なトレードオフ）。以後の機能追加で
  さらに増やす場合は、同じ厳しさで逸脱範囲を切ることが求められる。
- - 全候補を 1 本のプロンプトへ合成するため、候補が多い構成では 1 回あたりの入力トークンが
  増える。候補件数の上限は候補生成器（`IntentPolicy.max_candidates` 等）側で制御する。

## Confirmation

強制手段:

- `tests/runtime/intent/test_catalog_plan_l2.py`（**新規作成**）: 「候補 3 件・不足 5 件で
  `Runner.run` が 1 回」「不足 0 件で 0 回」「`predict=False` で 0 回」を `FakeModel.calls` の
  件数で pin する。
- `tests/runtime/intent/test_predict_l1.py`（**新規作成**）: 同一プロンプトセグメントを 2 候補が
  要求したとき、合成結果に当該セグメントの本文が 1 回だけ現れることを pin する。
- `tests/_adapters/test_intent_adapter_l2.py`（既存ファイルへ追記）: `run_filler_prompt` が
  `AgentSpec` から `AgentBuilder` 経由で実体化して 1 回走らせること、`RunResult` からの usage 抽出と
  未取得判定を検証する。
- SDK 隔離 grep（`_adapters` 外に `from agents` / `import agents` を許さない既存計測）に
  新規モジュールを含める。

`docs/QUALITY-GUARANTEES.md` に登録済み（source = ADR-0026）。
