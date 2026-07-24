# 0004: ML 分類器支援層で estimator.fit() を 1 関数に隔離して駆動する

- Status: accepted
- Date: 2026-07-23

## Context

`runtime/intent` に ML 分類器（sklearn / 軽量 Transformer / ONNX 等・方式非依存）を
`CandidateGenerator` Protocol へ差し込む支援層を追加するにあたり、学習（fit）駆動をライブラリが
持つか、利用者責務のままにするかを検討した。

lib の不変条件は build-don't-run（宣言・build-time 検証・薄い結線に徹し、実行は SDK
`Runner.run` に寄せる。公開の実行 API を持たない）であり、`estimator.fit()` を lib が呼ぶことは
この原則からの逸脱にあたる。一方、ユーザー要件には「自体が Fit しないと結構使いにくくない？」
という声があり、sklearn 互換学習器を素のまま渡すだけで学習が完結する体験（ゼロコード fit）への
実需が確認された。

検討した選択肢:

1. **fit 駆動を一切持たず、学習済み推論 callable の受け取りのみとする（却下）**: build-don't-run を
   完全に維持できるが、利用者は毎回 `estimator.fit(X, y)` 呼び出しと推論 callable への変換を
   自前で書く必要があり、sklearn 利用時の定型コストが高い。
2. **fit 駆動を lib の中心 API とする（却下）**: 学習ループ・ハイパーパラメータ管理まで lib が
   持つ設計は実行エンジン化であり、build-don't-run から大きく逸脱する。lib のスコープを
   「分類結果の型・規約の提供」から「ML パイプライン管理」へ拡大してしまう。
3. **学習手段非依存の最小 Protocol（FR-4a）+ sklearn 互換学習器のゼロコード fit（FR-4b）の
   二段構え（採用）**: 学習手段を問わない最小契約（`TrainedIntentEstimator` 成果物型 +
   `IntentTrainer` 型エイリアス + `make_trained_estimator` builder）を主契約とし、lib は
   trainer を呼び出さない。加えて sklearn 互換 estimator に限り、`fit_ml_estimator` という
   単一の便宜関数でのみ `estimator.fit()` を 1 回駆動する。ユーザー合意済み。

## Decision

build-don't-run からの逸脱は `fit_ml_estimator` 内の `estimator.fit()` 呼び出し 1 点のみに限定する。

- 逸脱コードは `_ml_training.py` に物理隔離する（推論側 `_ml.py` には fit 相当のコードを一切
  持たない）。
- `CLAUDE.md`「設計の核」の build-don't-run 項目に例外 1 文を明記し、逸脱範囲を grep 的に
  検知可能な状態に保つ。
- 主契約は FR-4a（`TrainedIntentEstimator` / `IntentTrainer` / `make_trained_estimator`）とし、
  sklearn 以外の学習手段（軽量 Transformer / ONNX 等）は FR-4a に沿って利用者が
  `TrainedIntentEstimator` を自作すれば同じ下流（`intent_classifier_from_ml_inference`）に
  接続できる。`fit_ml_estimator`（FR-4b）はその上に載る sklearn 互換限定の便宜レイヤであり、
  必須経路ではない。

現在仕様の SoT は `docs/architecture.md`（「意図予測（`runtime/intent`）」節の
「ML ベース分類器支援」小節）とし、本 ADR は判断・却下案のみを記録して仕様詳細を重複させない。

## Consequences

- + sklearn 互換学習器を持つ利用者は `fit_ml_estimator(estimator, x_train=..., y_train=...,
  policy=...)` の 1 呼び出しで学習済み推論器を得られ、ゼロコード fit の実需を満たす。
- + 逸脱範囲が 1 関数・1 ファイルに閉じるため、将来 lib が学習ループ管理へスコープを拡大しようと
  した場合、CLAUDE.md の例外文言の範囲逸脱として検知できる。
- + FR-4a の学習手段非依存 Protocol が主契約であり、sklearn 以外の学習手段でも lib の恩恵
  （推論側の mapper・dedup・allowlist・sort・truncate・ファクトリ結線）を受けられる。
- - `fit_ml_estimator` は sklearn 互換 API（`fit` / `predict_proba` / `classes_`）を前提とし、
  この形に合わない学習器は FR-4a の手動経路を使う必要がある（意図的なトレードオフ）。

## Confirmation

- 逸脱範囲の物理隔離（`estimator.fit()` 呼び出しが `_ml_training.py` の `fit_ml_estimator` に
  閉じること）と推論側（`_ml.py`）に fit 相当のコードがないことは、モジュール構成そのもの
  （ファイル分割）とコードレビューで担保する。
- SDK 隔離・duck-typed estimator（ML フレームワーク非 import）の強制手段: 強制手段:
  `tests/runtime/intent/test_ml_l2.py` / `test_ml_training_l2.py`（duck-typed fake estimator で
  sklearn 非依存に検証）。
