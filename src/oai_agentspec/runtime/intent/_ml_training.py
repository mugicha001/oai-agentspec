"""ML 意図分類器の学習支援層（FR-4a/4b/4c）。

FR-4a: 学習手段非依存の最小契約（`IntentTrainer` 型エイリアス + `TrainedIntentEstimator`
frozen dataclass + `make_trained_estimator` builder）。ライブラリは trainer を呼び出さない
（実行主体は利用者・build-don't-run の逸脱に当たらない）。

FR-4c: 学習済み sklearn 互換 estimator から `fit` を駆動せずに推論 callable を
組み立てる薄い factory（`ml_inference_from_estimator`）。

FR-4b: sklearn 互換 estimator の `fit(X, y)` を 1 回駆動して推論 callable を含む
`TrainedIntentEstimator` を組み立てる便宜関数（`fit_ml_estimator`・実装済み）。

build-don't-run 逸脱: 本モジュールの `fit_ml_estimator`（FR-4b）で `estimator.fit()`
を 1 回駆動する。逸脱範囲はその 1 関数のみ（ADR 0004）。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .types import IntentContext, IntentPolicy

# 内部エイリアス（非公開・_ml.py の MLCandidateGenerator inference 型と同型）。
_MLInference = Callable[
    [IntentContext[Any]],
    Sequence[tuple[str, float]] | Awaitable[Sequence[tuple[str, float]]],
]

# 公開エイリアス。学習手順（trainer）は呼び出された結果 TrainedIntentEstimator を
# 返す、という戻り値契約のみを型で表現する（引数は利用者が自由に定義してよい）。
IntentTrainer = Callable[..., "TrainedIntentEstimator"]


@dataclass(frozen=True)
class TrainedIntentEstimator:
    """FR-4a 学習成果物。FR-3/FR-5 へ直結する推論 callable を含む束。

    Attributes:
        inference: `IntentContext` を受け取り (ラベル, スコア) 列を返す推論 callable
            （同期/非同期）。
        estimator: FR-4b 用の学習済み estimator（再利用・保存のため保持。既定 None）。
        decoder: ラベル逆写像（既定 None = 恒等）。カスタムエンコーディング時に FR-4b が
            設定する。
        policy: 意図ポリシー（既定 None）。`fit_ml_estimator` の成果物に保持され、
            `intent_classifier_from_ml_inference`（FR-5）の policy 省略時に自動解決
            される。
    """

    inference: _MLInference
    estimator: Any | None = None
    decoder: Callable[[Any], str] | None = None
    policy: IntentPolicy | None = None


def make_trained_estimator(
    *,
    inference: _MLInference,
    estimator: Any | None = None,
    decoder: Callable[[Any], str] | None = None,
    policy: IntentPolicy | None = None,
) -> TrainedIntentEstimator:
    """`TrainedIntentEstimator` を組み立てる薄い builder。

    `TrainedIntentEstimator(...)` を直接呼ぶのと機能は同じだが、利用者に安定な
    公開 API 面を提供する。

    Args:
        inference: `IntentContext` を受け取り (ラベル, スコア) 列を返す推論 callable
            （同期/非同期）。keyword-only。
        estimator: FR-4b 用の学習済み estimator（再利用・保存のため保持）。keyword-only。
            既定 None。
        decoder: ラベル逆写像（既定 None = 恒等）。keyword-only。
        policy: 意図ポリシー（既定 None）。keyword-only。

    Returns:
        組み立てられた `TrainedIntentEstimator`。
    """
    return TrainedIntentEstimator(
        inference=inference, estimator=estimator, decoder=decoder, policy=policy
    )


def ml_inference_from_estimator(
    estimator: Any,
    *,
    transform: Callable[[IntentContext[Any]], Any] | None = None,
    decoder: Callable[[Any], str] | None = None,
) -> Callable[[IntentContext[Any]], Sequence[tuple[str, float]]]:
    """学習済み sklearn 互換 estimator から `fit` なしで推論 callable を組み立てる。

    `estimator.predict_proba` と `estimator.classes_` を用いて (ラベル, スコア) 列を
    返す同期 callable を返す。学習（`fit`）は駆動しないため build-don't-run の逸脱には
    当たらない。`estimator` に `predict_proba` / `classes_` のいずれかが欠けている場合は
    構築時に fail-fast する。

    Args:
        estimator: `predict_proba` メソッドと `classes_` 属性を持つ学習済み
            sklearn 互換オブジェクト。
        transform: `IntentContext` を `predict_proba` の入力へ変換する callable。
            keyword-only。None のとき `[context.utterance]`（単一サンプル列）を渡す
            （既定・sklearn のサンプル列契約に整合）。
        decoder: `classes_` の各値を文字列ラベルへ復号する callable。keyword-only。
            None のとき恒等（`classes_` の値をそのまま用いる。既定）。

    Returns:
        `IntentContext` を受け取り (ラベル, スコア) 列を `classes_` の順で返す同期
        callable。

    Raises:
        AttributeError: `estimator` に `predict_proba` または `classes_` が無い場合。
    """
    if getattr(estimator, "predict_proba", None) is None:
        raise AttributeError("estimator must provide a 'predict_proba' method")
    if getattr(estimator, "classes_", None) is None:
        raise AttributeError("estimator must provide a 'classes_' attribute")

    def _inference(context: IntentContext[Any]) -> list[tuple[str, float]]:
        x = transform(context) if transform is not None else [context.utterance]
        proba_row = estimator.predict_proba(x)[0]
        classes = estimator.classes_
        labels = classes if decoder is None else [decoder(c) for c in classes]
        return list(zip(labels, proba_row, strict=True))

    return _inference


def fit_ml_estimator(
    estimator: Any,
    *,
    x_train: Any,
    y_train: Any,
    policy: IntentPolicy,
    transform: Callable[[IntentContext[Any]], Any] | None = None,
    label_encoding: Mapping[str, Any] | None = None,
) -> TrainedIntentEstimator:
    """sklearn 互換 estimator の `fit(X, y)` を駆動し推論 callable を組み立てる。

    `estimator.fit(x_train, y_encoded)` を 1 回だけ呼び、FR-3/FR-5 に直結する推論
    callable を含む `TrainedIntentEstimator` を返す。`label_encoding` を渡した場合は
    `y_train` を数値ラベル列へエンコードして fit へ渡し、推論時に逆写像で文字列ラベルへ
    復号する（元の `y_train` は破壊的変更を受けない）。`estimator` に `fit` が無い場合は
    構築時に fail-fast する。

    Note:
        本関数は build-don't-run 不変条件からの明示的逸脱（`estimator.fit()` の 1 回
        駆動）。逸脱範囲はここのみ。詳細は `docs/adr/0004-intent-ml-fit-deviation.md`
        を参照。

    Args:
        estimator: `fit(X, y)` / `predict_proba` / `classes_` を持つ sklearn 互換
            オブジェクト。
        x_train: `estimator.fit` の第 1 引数へ渡す学習入力。keyword-only。
        y_train: 学習ラベル列。keyword-only。`label_encoding` が None のとき参照そのまま
            を fit の第 2 引数へ渡す。
        policy: 意図ポリシー。keyword-only。成果物に保持され FR-5 の policy 省略時に
            自動解決される。fit 時ラベル検証の将来拡張ポイント。
        transform: `IntentContext` を `predict_proba` の入力へ変換する callable。
            keyword-only。None のとき `[context.utterance]`（単一サンプル列）を渡す
            （既定・sklearn のサンプル列契約に整合）。
        label_encoding: 文字列ラベルから数値ラベルへの写像。keyword-only。None のとき
            `y_train` を素通しし復号もしない（既定）。

    Returns:
        fit 済み estimator を保持する `TrainedIntentEstimator`。

    Raises:
        AttributeError: `estimator` に `fit` メソッドが無い場合。
    """
    if getattr(estimator, "fit", None) is None:
        raise AttributeError("estimator must provide a 'fit' method")

    if label_encoding is None:
        y_encoded = y_train
        decoder: Callable[[Any], str] | None = None
    else:
        y_encoded = [label_encoding[y] for y in y_train]
        _reverse = dict(zip(label_encoding.values(), label_encoding.keys(), strict=True))

        def decoder(value: Any) -> str:
            return _reverse[value]

    estimator.fit(x_train, y_encoded)

    inference = ml_inference_from_estimator(estimator, transform=transform, decoder=decoder)
    return TrainedIntentEstimator(
        inference=inference, estimator=estimator, decoder=decoder, policy=policy
    )
