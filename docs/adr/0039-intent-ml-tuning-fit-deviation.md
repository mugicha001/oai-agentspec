# 0039: CV チューニング支援の fit 駆動を private ヘルパ 1 箇所へ集約する

- Status: accepted
- Date: 2026-09-02

## Context

`runtime/intent` の ML 支援層は学習済み推論器の受け取りと単発 fit までを支援するが、ハイパー
パラメータチューニング（CV ベースの探索）を lib 側で扱う入口がない。利用者は sklearn 互換の
探索器を自前で fit してから推論成果物を組み直す必要があり、`best_params` / `best_score` /
`cv_results` といったチューニング副産物と後段（意図分類器の組立）が分断されている。sklearn 互換の
探索器を注入するだけで CV チューニング結果をそのまま意図分類器へ直結できる薄いラッパを追加する
にあたり、探索の fit をどこで駆動するかを検討した。

本 ADR は `docs/adr/0004-intent-ml-fit-deviation.md` の Decision（L34）と Confirmation（L62-64）の
**記述位置**を読み替える。逸脱の実体（lib が利用者の学習器の `fit()` を駆動する物理点が 1 箇所で
あること、および推論側 `_ml.py` に fit 相当のコードが無いこと）は不変であり、その位置が
`_ml_training.py` の private ヘルパ `_fit_once` へ移る。入口は `fit_ml_estimator` と
`tune_ml_estimator` の 2 つになるが、両者は同一の物理点を共有する。0004 の判断そのもの（逸脱を
1 モジュールへ物理隔離する）は覆さないため、0004 に `superseded by` は付さない。

検討した選択肢:

1. **`fit_ml_estimator` へ委譲し、その戻り値を再包装する（却下）**: 駆動行は既存の 1 行のまま
   維持できるが、推論 callable と成果物の `estimator` が探索器そのものにバインドされ、「探索の
   最良推定器から成果物を組む」という要件が成立しない。動作するのは `SearchCV` が
   `predict_proba` / `classes_` を最良推定器へ委譲するという sklearn の実装詳細に依存した結果で
   あり、estimator を duck-typed な `Any` として扱う既存前提と整合しない。加えて探索器の再学習を
   無効にした構成では、原因（再学習が無効である）に到達できない `predict_proba` 欠落の
   `AttributeError` として現れる。
2. **`fit_ml_estimator` へ委譲したうえで最良推定器から推論器を再構築する（却下）**: 委譲経路では
   推論器の組立検査の対象が探索器そのものになるため、`predict_proba` / `classes_` を最良推定器へ
   委譲しない自作探索器では、探索が正常終了しても構築時点で必ず失敗する。「探索器の種別を問わず
   同一入口で扱える」という要件が成立しない。捨てるための推論 callable を必ず 1 個組む構造という
   無駄も伴う。
3. **チューニング関数の本体が探索器を直接 fit し、private ヘルパで駆動点を共有する（採用）**:
   0004 の Decision の文言（逸脱は `fit_ml_estimator` 内の 1 点のみ）は字義通りには偽になるが、
   逸脱の実体（lib が `.fit` を呼ぶ物理点の数）は 1 箇所のまま保たれ、sklearn の実装詳細への依存も
   持ち込まない。

**`fit_ml_estimator` 内に `fit()` を残したまま新経路を足す案は却下する**: `.fit(` の呼び出しが lib 内
2 箇所へ増え、駆動点の集合一致による機械検査も成立せず、逸脱の実体（物理点の数）が悪化する。

実 `scikit-learn` を用いた end-to-end 検証の依存経路も本 ADR で決着させた。テスト実行環境に
sklearn は導入されておらず、CI の依存同期は extra のみを対象とするため、非既定の依存グループに
宣言しても導入されない。条件付き skip は CI でもローカルでも常に skip され、検証が一度も
実行されないため採らない。

## Decision

CV チューニング支援は `tune_ml_estimator` 本体が private ヘルパ `_fit_once` 経由で探索器を fit し、
探索器の最良推定器から推論器を組む。

- build-don't-run からの逸脱の物理点は `_ml_training.py` の `_fit_once` ただ 1 箇所とし、
  `fit_ml_estimator` と `tune_ml_estimator` の 2 入口がこれを共有する。lib は探索アルゴリズムを
  識別する分岐・再試行・独自の学習ループ・`Runner` 代行を持たない。
- 逸脱コードは `_ml_training.py` に物理隔離するという 0004 の判断を維持する（推論側 `_ml.py` には
  fit 相当のコードを一切持たない）。
- 実 sklearn を用いた end-to-end 検証のため、`scikit-learn` は開発依存グループにのみ追加する。
  配布物の依存・extra は変えない。PEP 735 の依存グループは配布物メタデータに載らず、利用者の
  インストール面・依存グラフを一切変えないため、「新規 extra の追加」には当たらない。lib 本体が
  ML フレームワークを import しない（duck-typed estimator 契約）という不変条件とも両立する。
- `./CLAUDE.md`「設計の核」の build-don't-run 項目には 6 例目を足さず、例外 (1) の文を
  `fit_ml_estimator` / `tune_ml_estimator` が private ヘルパ 1 箇所で利用者の学習器を 1 回 fit する
  形へ改める。両者は同一の物理駆動点を共有するため 1 例として数える（既存の例外 (3) が 2 関数を
  1 例として数えているのと同じ数え方であり、逸脱台帳の件数も維持される）。

現在仕様の SoT は `docs/architecture.md`（「意図予測（`runtime/intent`）」節の「ML ベース分類器支援」
小節）とし、本 ADR は判断・却下案のみを記録して仕様詳細を重複させない（0004 と同じ分担）。

## Consequences

- + sklearn 互換の探索器を注入するだけで、探索の駆動・最良推定器からの推論器組立・チューニング
  副産物の保持・既存の意図分類器組立入口への直結までが 1 呼び出しで完結する。探索器を差し替えても
  呼び出し側は不変であり、lib に探索種別の分岐が入らない。
- + 逸脱の実体（lib が `.fit` を呼ぶ物理点）は 1 箇所のまま増えず、0004 の趣旨である物理隔離が
  保存される。担保手段も機械検証へ強化される（Confirmation 参照）。
- + 探索器の再学習を無効にした構成を、推論器の組立前に固有の診断メッセージで落とせる。sklearn の
  設定体系（`refit` 等の探索器固有パラメータ）を lib が読んで分岐することはなく、メッセージ文言と
  してのみ言及する。
- - 探索器そのものは成果物に保持しない。探索器は利用者自身が構築して引数として渡したオブジェクト
  であり、成果物に載せても利用者が失う情報を何も回復しないためである（探索器が内部生成する最良
  推定器とは事情が異なる）。必要が生じた時点で、既定値つきの keyword-only フィールドとして後方
  互換に追加できる。
- - 成果物の副産物フィールドは同一性比較の対象外とする。親型で成功する同一性比較・ハッシュが子型で
  例外にならないよう、親型と同じ同一性・ハッシュの挙動を保つための措置であり、診断用の表示には
  副産物が残る。
- - 開発・CI 環境に scikit-learn とその推移依存（numpy / scipy 等）が入る。ロックファイルには既に
  これらが記録済みのため依存ツリーの新規解決は起きないが、cold cache 時の CI 実行時間が数十秒
  増える。実 sklearn の end-to-end が緑になることの価値がこれを上回ると判断した。
- - 探索器の再学習を無効にした構成は本経路では扱えず、明示的な失敗として落ちる（最良推定器が
  存在しない以上、推論器を組めないため意図的なトレードオフ）。

## Confirmation

0004 Confirmation が担保手段としていた「モジュール構成 + コードレビュー」を、本 ADR では AST 走査に
よる機械検証（`tests/test_build_dont_run_isolation_l1.py`）へ置き換える。ただし本検査は素直な `.fit(`
の追加に対する回帰網であり、意図的な迂回（`getattr(target, "fit")(...)` / 束縛の再代入 /
`functools.partial` / `operator.methodcaller`）を防ぐものではない。迂回の検出は引き続きコード
レビューが担う。

強制手段:

- lib 内の学習駆動が 1 箇所に閉じること:
  `tests/test_build_dont_run_isolation_l1.py::test_fit_駆動箇所の集合が_fit_once_ただ一つと一致する`
  （`src/oai_agentspec/` 配下を `ast` で走査し、検出した (相対パス, 囲む関数名) の集合が
  `_ml_training.py` の `_fit_once` ただ 1 つと一致することを検査する。件数カウントではなく集合一致に
  するのは、無関係な実在 API の `fit` が将来入った際に、原因の結びつかない失敗になるのを防ぐため）
- lib 本体が ML フレームワークを import しないこと:
  `tests/test_build_dont_run_isolation_l1.py::test_lib_本体は_sklearn_numpy_scipy_を_import_しない`
- 探索アルゴリズムを識別する分岐引数がシグネチャに現れないこと:
  `tests/runtime/intent/test_ml_tuning_l1.py::test_tune_ml_estimator_のシグネチャに探索固有の引数が現れない`
- 同一性契約が親型と等価であること:
  `tests/runtime/intent/test_ml_tuning_l1.py::test_hash_は親と同値で成立する` /
  `tests/runtime/intent/test_ml_tuning_l1.py::test_ndarray_を含む_cv_results_でも等価比較が例外にならない`
- 探索の駆動が 1 回だけであること:
  `tests/runtime/intent/test_ml_tuning_l2.py::test_tune_は_search_の_fit_を_1_回だけ駆動する`
- duck-typed fake 探索器での契約全体（副産物の透過 / エンコード経路 / 必須属性の fail-fast /
  任意属性の `None` 既定 / 非単射なラベル対応の拒否）: `tests/runtime/intent/test_ml_tuning_l2.py`
- 実 sklearn の end-to-end:
  `tests/runtime/intent/test_ml_tuning_sklearn_l2.py::test_gridsearchcv_で調整した分類器が意図を分類できる`

上記のうち AST 走査 2 件は `docs/QUALITY-GUARANTEES.md` に登録し、台帳側の source 列に
`ADR-0039` を安定アンカーとして記載して相互参照する。
