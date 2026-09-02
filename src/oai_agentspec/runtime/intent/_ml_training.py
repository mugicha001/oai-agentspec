"""ML 意図分類器の学習支援層（FR-4a/4b/4c）。

FR-4a: 学習手段非依存の最小契約（`IntentTrainer` 型エイリアス + `TrainedIntentEstimator`
frozen dataclass + `make_trained_estimator` builder）。ライブラリは trainer を呼び出さない
（実行主体は利用者・build-don't-run の逸脱に当たらない）。

FR-4c: 学習済み sklearn 互換 estimator から `fit` を駆動せずに推論 callable を
組み立てる薄い factory（`ml_inference_from_estimator`）。

FR-4b: sklearn 互換 estimator の `fit(X, y)` を 1 回駆動して推論 callable を含む
`TrainedIntentEstimator` を組み立てる便宜関数（`fit_ml_estimator`・実装済み）。

チューニング支援: sklearn 互換の CV 探索器を 1 回駆動し、`best_estimator_` から推論
callable を組み立ててチューニング副産物を保持する `TunedIntentEstimator` を返す便宜関数
（`tune_ml_estimator`）。

build-don't-run 逸脱: 本モジュールの private ヘルパ `_fit_once` が `fit()` を 1 回駆動する
唯一の物理点であり、`fit_ml_estimator`（FR-4b）と `tune_ml_estimator` の 2 つの入口が
この同一の物理点を共有する（ADR 0004 / ADR 0039）。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .types import IntentContext, IntentPolicy

# 内部エイリアス（非公開・_ml.py の MLCandidateGenerator inference 型と同型）。
_MLInference = Callable[
    [IntentContext[Any]],
    Sequence[tuple[str, float]] | Awaitable[Sequence[tuple[str, float]]],
]


def _encode_labels(
    y_train: Any, label_encoding: Mapping[str, Any] | None
) -> tuple[Any, Callable[[Any], str] | None]:
    """ラベル列を数値エンコーディングへ変換し、逆写像 decoder と対で返す。

    `label_encoding` が None の場合は `y_train` を素通しし、decoder も None（恒等）を
    返す。指定がある場合は各ラベルをエンコード値へ写像した新しい列を作り、元の
    `y_train` は破壊的変更を受けない。

    Args:
        y_train: 学習ラベル列。
        label_encoding: 文字列ラベルから数値ラベルへの写像。None のとき素通し。

    Returns:
        エンコード済みラベル列（`label_encoding` が None のとき `y_train` そのもの）と、
        エンコード値から元の文字列ラベルへ復号する callable（None のとき恒等）の組。

    Raises:
        ValueError: `label_encoding` が単射でない（複数キーが同一のエンコード値へ
            衝突する）場合。
        ValueError: `label_encoding` が `y_train` に現れるラベルを被覆していない場合。
            メッセージにはラベル値そのものを載せず、被覆されていない相異なるラベルの
            件数のみを含む（利用者の実データが例外文字列やログへ流入しないため）。
    """
    if label_encoding is None:
        return y_train, None
    if len(set(label_encoding.values())) != len(label_encoding):
        raise ValueError("label_encoding must be injective (duplicate encoded values)")
    # 被覆性の検査と変換は同一ループで行う。事前の別パスとして走査を足すと、
    # `y_train` が generator 等の 1 回限り iterable のとき 1 回目で消費され、
    # 空のラベル列で fit してしまう（silent な退行）。
    y_encoded: list[Any] = []
    unmapped: set[Any] = set()
    for label in y_train:
        if label in label_encoding:
            y_encoded.append(label_encoding[label])
        else:
            unmapped.add(label)
    if unmapped:
        raise ValueError(
            f"label_encoding does not cover all labels in y_train ({len(unmapped)} unmapped)"
        )
    _reverse = dict(zip(label_encoding.values(), label_encoding.keys(), strict=True))

    def decoder(value: Any) -> str:
        return _reverse[value]

    return y_encoded, decoder


def _fit_once(target: Any, x_train: Any, y_encoded: Any) -> None:
    """`target.fit(x_train, y_encoded)` を 1 回駆動する。

    build-don't-run 不変条件からの逸脱（lib が利用者の学習器の `fit()` を駆動する）が
    許容される唯一の物理点。この関数以外の箇所で `.fit(` を呼び出してはならない。
    詳細は `docs/adr/0004-intent-ml-fit-deviation.md` および
    `docs/adr/0039-intent-ml-tuning-fit-deviation.md` を参照。

    Args:
        target: `fit(X, y)` を持つ sklearn 互換オブジェクト（estimator または
            探索器）。
        x_train: `target.fit` の第 1 引数へ渡す学習入力。
        y_encoded: `target.fit` の第 2 引数へ渡すラベル列（エンコード済みの場合を含む）。

    Returns:
        None。`target` は破壊的に fit される。
    """
    target.fit(x_train, y_encoded)


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


@dataclass(frozen=True)
class TunedIntentEstimator(TrainedIntentEstimator):
    """CV 探索の成果物。親の全契約に加えチューニング副産物を保持する。

    `TrainedIntentEstimator` の frozen サブクラスであり `isinstance` 判定を通るため、
    `intent_classifier_from_ml_inference`（FR-5）へそのまま渡せる。親フィールドの位置
    引数束縛は不変で、追加した副産物 3 フィールドは keyword-only。

    副産物 3 フィールドはすべて `compare=False` とする（親型との置換可能性の保全）。
    `best_params` の実値は unhashable な `dict`、`cv_results` は値に `numpy.ndarray` を
    含みうる `dict` であり、`compare=True`（既定）のままだと dataclass が生成する
    `__hash__` / `__eq__` がこれらを参照するため、親では成功する `hash()` が
    `TypeError: unhashable type: 'dict'` になり、`==` が ndarray の真偽値評価で
    `ValueError: truth value of an array ... is ambiguous` になる。`compare=False` に
    することで同一性比較の対象フィールドとハッシュ値は親と同一になり（dataclass が
    生成する `__eq__` は `other.__class__ is self.__class__` を要求するため、親型の
    インスタンスとの `==` は `compare` の指定によらず元より成立しない）、`repr` には
    副産物が残るため診断性は失われない。

    Attributes:
        best_params: 探索が選んだ最良パラメータ（`search.best_params_` 由来・必須・
            keyword-only）。lib は中身を解釈せずそのまま保持する。
        best_score: 最良パラメータの CV スコア（`search.best_score_` 由来・既定 None）。
            探索器が当該属性を持たない場合は None。
        cv_results: 探索の全試行結果（`search.cv_results_` 由来・既定 None）。キー構成は
            探索器の scoring 設定に依存するため lib は中身を解釈しない。
    """

    best_params: Mapping[str, Any] = field(kw_only=True, compare=False)
    best_score: float | None = field(default=None, kw_only=True, compare=False)
    cv_results: Mapping[str, Any] | None = field(default=None, kw_only=True, compare=False)


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
        本関数は build-don't-run 不変条件からの明示的逸脱を伴う。`estimator.fit()` の
        駆動は private ヘルパ `_fit_once` に閉じており、本関数はそれを 1 回呼ぶ。
        詳細は `docs/adr/0004-intent-ml-fit-deviation.md` を参照。

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
        ValueError: `label_encoding` が単射でない（複数キーが同一のエンコード値へ
            衝突する）場合。
        ValueError: `label_encoding` が `y_train` に現れるラベルを被覆していない場合。
            メッセージにはラベル値そのものを載せず、被覆されていない相異なるラベルの
            件数のみを含む（利用者の実データが例外文字列やログへ流入しないため）。
    """
    if getattr(estimator, "fit", None) is None:
        raise AttributeError("estimator must provide a 'fit' method")

    y_encoded, decoder = _encode_labels(y_train, label_encoding)
    _fit_once(estimator, x_train, y_encoded)

    inference = ml_inference_from_estimator(estimator, transform=transform, decoder=decoder)
    return TrainedIntentEstimator(
        inference=inference, estimator=estimator, decoder=decoder, policy=policy
    )


def tune_ml_estimator(
    search: Any,
    *,
    x_train: Any,
    y_train: Any,
    policy: IntentPolicy,
    transform: Callable[[IntentContext[Any]], Any] | None = None,
    label_encoding: Mapping[str, Any] | None = None,
) -> TunedIntentEstimator:
    """sklearn 互換の CV 探索器を 1 回駆動し、最良推定器から推論 callable を組み立てる。

    `search.fit(x_train, y_encoded)` を 1 回だけ呼び、fit 後の `best_estimator_` から
    推論 callable を組んで `TunedIntentEstimator` として返す。推論 callable も成果物の
    `estimator` も探索器そのものではなく `best_estimator_` に束縛されるため、探索器が
    `predict_proba` を委譲するかどうかに依存しない。lib は探索アルゴリズムを識別する
    分岐を持たないため、`GridSearchCV` / `RandomizedSearchCV` / `HalvingGridSearchCV` /
    自作探索器のいずれも同じ入口で扱える（探索の設定は探索器側に閉じる）。

    `label_encoding` を渡した場合は `y_train` を数値ラベル列へエンコードして fit へ渡し、
    推論時に逆写像で文字列ラベルへ復号する（元の `y_train` は破壊的変更を受けない）。
    `fit` 属性の欠落と `label_encoding` の非単射は、高コストな探索を走らせる前に
    fail-fast する。

    `refit` を無効にした探索器（sklearn の `refit=False` 等）は使用できない。最良
    パラメータでの再学習が行われず `best_estimator_` が生成されないため、推論 callable
    を組む対象が存在しないためである。

    副産物のフィールド名は sklearn の末尾アンダースコア付き属性名（`best_params_` /
    `best_score_` / `cv_results_`）を `getattr` で読み、lib 側の語彙（`best_params` /
    `best_score` / `cv_results`）で保持したものである。

    Note:
        本関数は build-don't-run 不変条件からの明示的逸脱を伴う。`search.fit()` の駆動は
        private ヘルパ `_fit_once` に閉じており、本関数はそれを 1 回呼ぶ
        （`fit_ml_estimator` と同一の物理点を共有する）。詳細は
        `docs/adr/0004-intent-ml-fit-deviation.md` および
        `docs/adr/0039-intent-ml-tuning-fit-deviation.md` を参照。

    Args:
        search: `fit(X, y)` を持ち、fit 後に `best_estimator_` / `best_params_` を生やす
            sklearn 互換の CV 探索器。
        x_train: `search.fit` の第 1 引数へ渡す学習入力。keyword-only。
        y_train: 学習ラベル列。keyword-only。`label_encoding` が None のとき参照そのまま
            を fit の第 2 引数へ渡す。
        policy: 意図ポリシー。keyword-only。成果物に保持され FR-5 の policy 省略時に
            自動解決される。
        transform: `IntentContext` を `predict_proba` の入力へ変換する callable。
            keyword-only。None のとき `[context.utterance]`（単一サンプル列）を渡す
            （既定・sklearn のサンプル列契約に整合）。
        label_encoding: 文字列ラベルから数値ラベルへの写像。keyword-only。None のとき
            `y_train` を素通しし復号もしない（既定）。

    Returns:
        `best_estimator_` を `estimator` に保持し、チューニング副産物を併せ持つ
        `TunedIntentEstimator`。`best_score_` / `cv_results_` を持たない探索器では
        対応するフィールドが None になる（欠落は例外にしない）。

    Raises:
        AttributeError: `search` に `fit` メソッドが無い場合（fit 前検査）。
        ValueError: `label_encoding` が単射でない（複数キーが同一のエンコード値へ
            衝突する）場合（fit 前検査）。
        ValueError: `label_encoding` が `y_train` に現れるラベルを被覆していない場合
            （fit 前検査）。メッセージにはラベル値そのものを載せず、被覆されていない
            相異なるラベルの件数のみを含む（利用者の実データが例外文字列やログへ
            流入しないため）。
        AttributeError: fit 後の `search` に `best_estimator_` が無い、またはその値が
            `None` の場合（`refit` を無効にした探索器）。属性を 1 回だけ読み、値が
            `None` なら欠落として扱う。
        AttributeError: fit 後の `search` に `best_params_` が無い、またはその値が
            `None` の場合。空 dict は「探索対象が無い」正当な構成として受理する。
        AttributeError: `search.best_estimator_` に `predict_proba` または `classes_`
            が無い場合（`ml_inference_from_estimator` が送出）。このときのメッセージは
            `estimator must provide a 'predict_proba' method` のように主語が
            `estimator` になるが、その `estimator` が指すのは引数の `search` ではなく
            `search.best_estimator_` である。
    """
    if getattr(search, "fit", None) is None:
        raise AttributeError("search must provide a 'fit' method")

    y_encoded, decoder = _encode_labels(y_train, label_encoding)
    _fit_once(search, x_train, y_encoded)

    best_estimator = getattr(search, "best_estimator_", None)
    if best_estimator is None:
        raise AttributeError(
            "search must provide a 'best_estimator_' attribute after fit (refit disabled?)"
        )
    best_params = getattr(search, "best_params_", None)
    if best_params is None:
        raise AttributeError("search must provide a 'best_params_' attribute after fit")
    best_score = getattr(search, "best_score_", None)
    cv_results = getattr(search, "cv_results_", None)

    inference = ml_inference_from_estimator(best_estimator, transform=transform, decoder=decoder)
    return TunedIntentEstimator(
        inference=inference,
        estimator=best_estimator,
        decoder=decoder,
        policy=policy,
        best_params=best_params,
        best_score=best_score,
        cv_results=cv_results,
    )
