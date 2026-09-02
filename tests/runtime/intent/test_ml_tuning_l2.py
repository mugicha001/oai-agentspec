"""L2: `_ml_training.py` の `tune_ml_estimator`（CV チューニング支援）の振る舞い契約 pin。

`tune_ml_estimator` は sklearn 互換の CV 探索器を 1 回だけ fit し、`best_estimator_`
から推論器を組み立て、チューニング副産物を保持した `TunedIntentEstimator` を返す。
本 L2 は duck-typed の fake 探索器を用いて以下を統合契約として pin する:

- 探索の駆動が 1 回だけであること（build-don't-run 逸脱の局所化）。
- 推論器が探索器そのものではなく `best_estimator_` から組まれること。
- 副産物（`best_params` / `best_score` / `cv_results`）の成果物への透過。
- 必須属性欠落の fail-fast（`best_estimator_` / `best_params_`）と任意属性の None 既定。
- ラベルエンコード経路（fit へはエンコード値・推論では文字列へ復号・元の y は非破壊）。
- fit 前検査（`fit` 欠落 / 非単射 `label_encoding`）が探索を走らせる前に落ちること。
- 成果物を `intent_classifier_from_ml_inference` へ直渡しできること（後方互換）。

各テスト内でのローカル import は、RED 先行フェーズで「未実装なら import 自体が失敗する」
形を取るために導入したものをそのまま維持している（`test_ml_training_l2.py` と同じ作法）。
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from oai_agentspec.runtime.intent.factories import intent_classifier_from_ml_inference
from oai_agentspec.runtime.intent.types import (
    IntentCategory,
    IntentContext,
    IntentPolicy,
    IntentQuery,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# 共有ヘルパ
# ---------------------------------------------------------------------------


def _ctx(utt: str = "返金") -> IntentContext[Any]:
    """テスト用の整形済み IntentContext を返す。"""
    return IntentContext(utterance=utt, history_items=(), run_context=None)


def _policy(max_candidates: int = 3) -> IntentPolicy:
    """テスト用の最小 IntentPolicy を返す (refund / cancel / other)。"""
    return IntentPolicy(
        categories=(
            IntentCategory(name="refund", description="返金"),
            IntentCategory(name="cancel", description="解約"),
            IntentCategory(name="other", description="その他"),
        ),
        max_candidates=max_candidates,
    )


def _thresholds() -> dict[str, float]:
    """`intent_classifier_from_ml_inference` へ渡す閾値マッピング。"""
    return {"certain": 0.90, "high": 0.75, "medium": 0.50, "low": 0.25, "speculative": 0.0}


class FakeBestEstimator:
    """探索の `best_estimator_` に相当する duck-typed の学習済み estimator。

    `predict_proba` は 1 サンプル分の 2D 配列 `[[p1, p2, ...]]` を返し、渡された入力を
    `received_x` に記録する。
    """

    def __init__(self, classes: tuple[Any, ...], proba_rows: list[list[float]]) -> None:
        self.classes_ = classes
        self._proba = proba_rows
        self.received_x: list[Any] = []

    def predict_proba(self, X: Any) -> list[list[float]]:
        self.received_x.append(X)
        return self._proba


class FakeFittableEstimator(FakeBestEstimator):
    """`fit_ml_estimator` と挙動を突き合わせるための学習可能 estimator。

    `fit` は sklearn 慣行に従い自身を返し、`(X, y)` を `fit_calls` に記録する。
    """

    def __init__(self, classes: tuple[Any, ...], proba_rows: list[list[float]]) -> None:
        super().__init__(classes=classes, proba_rows=proba_rows)
        self.fit_calls: list[tuple[Any, Any]] = []

    def fit(self, X: Any, y: Any) -> FakeFittableEstimator:
        self.fit_calls.append((X, y))
        return self


# 探索器 fake の 3 状態（設計 5.3 の実 sklearn 実測表に対応）。
REFIT_ENABLED = "refit_enabled"  # 単一 scoring・refit 有効（既定）
REFIT_DISABLED = "refit_disabled"  # refit=False 相当（best_estimator_ / predict_proba なし）
REFIT_CALLABLE = "refit_callable"  # refit=<callable> 相当（best_score_ なし）

# 属性そのものを生やさないことを表すセンチネル（None 代入は実物より寛容なので使わない）。
OMIT = object()

_DEFAULT_BEST_PARAMS = {"clf__C": 10.0}
_DEFAULT_BEST_SCORE = 0.8333333333333334
_DEFAULT_CV_RESULTS = {"mean_test_score": [0.75, 0.8333333333333334]}

# 探索器側 predict_proba が使われたら判別できるよう、best_estimator_ とは別の値を返す。
_SEARCH_SIDE_CLASSES = ("search_side_a", "search_side_b", "search_side_c")
_SEARCH_SIDE_PROBA = [[0.34, 0.33, 0.33]]


class FakeSearch:
    """duck-typed の CV 探索器 fake（実 sklearn の属性有無を状態機械として再現する）。

    未 fit の時点では `best_estimator_` / `best_params_` / `best_score_` /
    `cv_results_` / `predict_proba` のいずれも持たない（`hasattr` が False）。
    `fit` は sklearn 慣行に従い自身を返し、`(X, y)` を `fit_calls` に記録したうえで
    `mode` に応じた属性のみを生やす（設計 5.3 の実測表）:

    - `REFIT_ENABLED`: 全属性あり + 探索器側 `predict_proba` / `classes_` あり
    - `REFIT_DISABLED`: `best_estimator_` なし・探索器側 `predict_proba` なし
    - `REFIT_CALLABLE`: `best_score_` なし

    `best_params` / `best_score` / `cv_results` に `OMIT` を渡すと、その属性は fit 後も
    生えない（実 sklearn には無い構成だが自作探索器では起こりうる）。
    """

    def __init__(
        self,
        *,
        best: Any = None,
        mode: str = REFIT_ENABLED,
        best_params: Any = _DEFAULT_BEST_PARAMS,
        best_score: Any = _DEFAULT_BEST_SCORE,
        cv_results: Any = _DEFAULT_CV_RESULTS,
    ) -> None:
        self._best = best
        self._mode = mode
        self._best_params = best_params
        self._best_score = best_score
        self._cv_results = cv_results
        self.fit_calls: list[tuple[Any, Any]] = []
        self.search_side_proba_calls: list[Any] = []

    def _search_side_predict_proba(self, X: Any) -> list[list[float]]:
        """探索器側の推論（`best_estimator_` のものとは別物であることを判別する）。"""
        self.search_side_proba_calls.append(X)
        return _SEARCH_SIDE_PROBA

    def fit(self, X: Any, y: Any) -> FakeSearch:
        self.fit_calls.append((X, y))
        if self._best_params is not OMIT:
            self.best_params_ = self._best_params
        if self._cv_results is not OMIT:
            self.cv_results_ = self._cv_results
        if self._mode != REFIT_CALLABLE and self._best_score is not OMIT:
            self.best_score_ = self._best_score
        if self._mode != REFIT_DISABLED:
            self.best_estimator_ = self._best
            self.classes_ = _SEARCH_SIDE_CLASSES
            self.predict_proba = self._search_side_predict_proba
        return self


class FitlessSearch(FakeSearch):
    """`fit` を持たない探索器（`getattr(search, "fit", None)` が None になる）。"""

    fit = None  # type: ignore[assignment]


def _best(proba_rows: list[list[float]] | None = None) -> FakeBestEstimator:
    """既定の 3 クラス `best_estimator_` を返す。"""
    return FakeBestEstimator(
        classes=("refund", "cancel", "other"),
        proba_rows=proba_rows if proba_rows is not None else [[0.7, 0.2, 0.1]],
    )


# ---------------------------------------------------------------------------
# 受け入れ基準: 探索の駆動は 1 回だけ
# ---------------------------------------------------------------------------


def test_tune_は_search_の_fit_を_1_回だけ駆動する() -> None:
    """`search.fit(x, y)` はちょうど 1 回・引数は素通しで呼ばれる。"""
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    search = FakeSearch(best=_best())
    x_train = [[0.0], [1.0], [0.0]]
    y_train = ["refund", "cancel", "refund"]

    tune_ml_estimator(search, x_train=x_train, y_train=y_train, policy=_policy())

    assert len(search.fit_calls) == 1
    called_x, called_y = search.fit_calls[0]
    assert called_x is x_train
    assert called_y is y_train


# ---------------------------------------------------------------------------
# 受け入れ基準: best_estimator_ から推論器が組まれる（探索器そのものではない）
# ---------------------------------------------------------------------------


def test_推論器は_best_estimator_から組まれる() -> None:
    """推論は `best_estimator_` の `predict_proba` / `classes_` を使う。"""
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    best = _best(proba_rows=[[0.7, 0.2, 0.1]])
    search = FakeSearch(best=best)

    tuned = tune_ml_estimator(search, x_train=[[0.0]], y_train=["refund"], policy=_policy())
    result = list(tuned.inference(_ctx(utt="返金")))

    assert result == [("refund", 0.7), ("cancel", 0.2), ("other", 0.1)]
    assert best.received_x == [["返金"]]
    # 探索器側にも predict_proba があるが、そちらは使われない。
    assert search.search_side_proba_calls == []


def test_成果物の_estimator_は_best_estimator_である() -> None:
    """`.estimator` には探索器ではなく `best_estimator_` が入る。"""
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    best = _best()
    search = FakeSearch(best=best)

    tuned = tune_ml_estimator(search, x_train=[[0.0]], y_train=["refund"], policy=_policy())

    assert tuned.estimator is best
    assert tuned.estimator is not search


def test_custom_transform_が_best_estimator_へ渡る() -> None:
    """`transform` の戻り値がそのまま `best_estimator_.predict_proba` へ渡る。"""
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    best = _best()
    search = FakeSearch(best=best)
    sentinel = object()

    tuned = tune_ml_estimator(
        search,
        x_train=[[0.0]],
        y_train=["refund"],
        policy=_policy(),
        transform=lambda ctx: sentinel,
    )
    tuned.inference(_ctx())

    assert best.received_x == [sentinel]


# ---------------------------------------------------------------------------
# 受け入れ基準: 副産物の透過（FR-2a）
# ---------------------------------------------------------------------------


def test_副産物が成果物へ透過する() -> None:
    """`best_params_` / `best_score_` / `cv_results_` が成果物へそのまま載る。"""
    from oai_agentspec.runtime.intent._ml_training import (
        TunedIntentEstimator,
        tune_ml_estimator,
    )

    best_params = {"clf__C": 10.0, "tfidf__ngram_range": (2, 3)}
    cv_results = {"mean_test_score": [0.75, 0.83], "params": [{"clf__C": 1.0}]}
    search = FakeSearch(
        best=_best(), best_params=best_params, best_score=0.83, cv_results=cv_results
    )
    policy = _policy()

    tuned = tune_ml_estimator(search, x_train=[[0.0]], y_train=["refund"], policy=policy)

    assert isinstance(tuned, TunedIntentEstimator)
    assert tuned.best_params is best_params
    assert tuned.best_score == 0.83
    assert tuned.cv_results is cv_results
    assert tuned.policy is policy


def test_best_score_が無くても例外にせず_None_を既定にする() -> None:
    """`refit=<callable>` 相当（`best_score_` 欠落）でも成果物は組み上がる。"""
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    search = FakeSearch(best=_best(), mode=REFIT_CALLABLE)

    tuned = tune_ml_estimator(search, x_train=[[0.0]], y_train=["refund"], policy=_policy())

    assert not hasattr(search, "best_score_")
    assert tuned.best_score is None
    assert tuned.cv_results is not None


def test_cv_results_が無くても例外にせず_None_を既定にする() -> None:
    """`cv_results_` を持たない自作探索器でも成果物は組み上がる。"""
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    search = FakeSearch(best=_best(), cv_results=OMIT)

    tuned = tune_ml_estimator(search, x_train=[[0.0]], y_train=["refund"], policy=_policy())

    assert not hasattr(search, "cv_results_")
    assert tuned.cv_results is None


# ---------------------------------------------------------------------------
# 受け入れ基準: 必須属性欠落の fail-fast（C3 / C4）
# ---------------------------------------------------------------------------


def test_best_estimator_が無いと_AttributeError() -> None:
    """`refit=False` 相当では refit への言及を含む AttributeError で落ちる。"""
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    search = FakeSearch(best=_best(), mode=REFIT_DISABLED)
    message = "search must provide a 'best_estimator_' attribute after fit (refit disabled?)"

    with pytest.raises(AttributeError, match=re.escape(message)):
        tune_ml_estimator(search, x_train=[[0.0]], y_train=["refund"], policy=_policy())

    assert not hasattr(search, "best_estimator_")
    assert len(search.fit_calls) == 1


def test_best_params_が無いと_AttributeError() -> None:
    """`best_params_` の欠落は None 既定にせず fail-fast する（FR-2a）。"""
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    search = FakeSearch(best=_best(), best_params=OMIT)
    message = "search must provide a 'best_params_' attribute after fit"

    with pytest.raises(AttributeError, match=re.escape(message)):
        tune_ml_estimator(search, x_train=[[0.0]], y_train=["refund"], policy=_policy())


def test_best_estimator_が_None_でも欠落として_AttributeError() -> None:
    """属性が生えていても値が None なら欠落扱いにする（推論器を組む対象が無いため）。

    属性の有無だけを見る判定（`hasattr`）では通過してしまい、後段の
    `ml_inference_from_estimator` が `predict_proba` 欠落として落ちて真因に到達できない。
    """
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    search = FakeSearch()  # best の既定は None
    message = "search must provide a 'best_estimator_' attribute after fit (refit disabled?)"

    with pytest.raises(AttributeError, match=re.escape(message)):
        tune_ml_estimator(search, x_train=[[0.0]], y_train=["refund"], policy=_policy())

    assert search.best_estimator_ is None
    assert hasattr(search, "best_estimator_")


def test_best_params_が_None_でも欠落として_AttributeError() -> None:
    """`best_params_` も値が None なら欠落扱いにする（`best_params` は必須フィールド）。

    空 dict は「探索するパラメータが無い」正当な構成なので `is None` 比較で判定し、
    falsy 判定にはしない（`{}` は通過させる）。
    """
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    search = FakeSearch(best=_best(), best_params=None)
    message = "search must provide a 'best_params_' attribute after fit"

    with pytest.raises(AttributeError, match=re.escape(message)):
        tune_ml_estimator(search, x_train=[[0.0]], y_train=["refund"], policy=_policy())

    assert search.best_params_ is None
    assert hasattr(search, "best_params_")


def test_best_params_が空_dict_なら通過する() -> None:
    """`best_params_ = {}` は「探索対象が無い」正当な構成として受理する。"""
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    search = FakeSearch(best=_best(), best_params={})

    tuned = tune_ml_estimator(search, x_train=[[0.0]], y_train=["refund"], policy=_policy())

    assert tuned.best_params == {}


# ---------------------------------------------------------------------------
# 受け入れ基準: ラベルエンコード経路
# ---------------------------------------------------------------------------


def test_label_encoding_でエンコードして_fit_へ渡し推論で復号する() -> None:
    """fit の第 2 引数はエンコード値・推論の戻り値は復号済み文字列・元の y は不変。"""
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    best = FakeBestEstimator(classes=(0, 1, 2), proba_rows=[[0.7, 0.2, 0.1]])
    search = FakeSearch(best=best)
    x_train = [[0.0], [1.0], [0.0]]
    y_train = ["refund", "cancel", "other"]

    tuned = tune_ml_estimator(
        search,
        x_train=x_train,
        y_train=y_train,
        policy=_policy(),
        label_encoding={"refund": 0, "cancel": 1, "other": 2},
    )

    called_x, called_y = search.fit_calls[0]
    assert called_x is x_train
    assert list(called_y) == [0, 1, 2]
    assert y_train == ["refund", "cancel", "other"]
    assert list(tuned.inference(_ctx())) == [("refund", 0.7), ("cancel", 0.2), ("other", 0.1)]
    assert tuned.decoder is not None


def test_label_encoding_なしなら_decoder_は_None() -> None:
    """`label_encoding=None` は y を素通しし decoder を作らない（既定）。"""
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    search = FakeSearch(best=_best())

    tuned = tune_ml_estimator(search, x_train=[[0.0]], y_train=["refund"], policy=_policy())

    assert tuned.decoder is None


def test_1_回限りの_iterable_な_y_train_でも正しくエンコードされる() -> None:
    """generator の y でも正しくエンコードされて探索器の fit へ届く。

    被覆性検査を「事前の別パス」として足すと generator が 1 回目の走査で消費され、
    空のラベル列で fit してしまう。本テストはその silent な退行を検知する pin であり、
    検査と変換が同一ループであること（`y_train` の走査が 1 回だけであること）を要求する。
    """
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    best = FakeBestEstimator(classes=(0, 1, 2), proba_rows=[[0.7, 0.2, 0.1]])
    search = FakeSearch(best=best)
    x_train = [[0.0], [1.0], [0.0]]
    y_train = (label for label in ["refund", "cancel", "other"])

    tune_ml_estimator(
        search,
        x_train=x_train,
        y_train=y_train,
        policy=_policy(),
        label_encoding={"refund": 0, "cancel": 1, "other": 2},
    )

    assert len(search.fit_calls) == 1
    called_x, called_y = search.fit_calls[0]
    assert called_x is x_train
    assert list(called_y) == [0, 1, 2]


# ---------------------------------------------------------------------------
# 受け入れ基準: fit 前検査（C1 / C2）が探索を走らせる前に落ちる
# ---------------------------------------------------------------------------


def test_fit_を持たない探索器は_fit_前に_AttributeError() -> None:
    """`fit` 属性検査は探索の駆動前に行われる（高コストな探索を走らせない）。"""
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    search = FitlessSearch(best=_best())

    with pytest.raises(AttributeError, match="search must provide a 'fit' method"):
        tune_ml_estimator(search, x_train=[[0.0]], y_train=["refund"], policy=_policy())

    assert search.fit_calls == []
    assert not hasattr(search, "best_params_")


def test_非単射_label_encoding_は_fit_前に_ValueError() -> None:
    """非単射検証は探索の駆動前に行われ、探索器の fit は一度も呼ばれない。"""
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    search = FakeSearch(best=_best())

    with pytest.raises(ValueError, match="label_encoding must be injective"):
        tune_ml_estimator(
            search,
            x_train=[[0.0]],
            y_train=["refund", "cancel", "other"],
            policy=_policy(),
            label_encoding={"refund": 0, "cancel": 0, "other": 1},
        )

    assert search.fit_calls == []
    assert not hasattr(search, "best_params_")


def test_非単射_label_encoding_のメッセージは_fit_ml_estimator_と同一() -> None:
    """`_encode_labels` の共有により両関数の ValueError 文言は完全一致する。"""
    from oai_agentspec.runtime.intent._ml_training import fit_ml_estimator, tune_ml_estimator

    encoding = {"refund": 0, "cancel": 0, "other": 1}
    y_train = ["refund", "cancel", "other"]

    with pytest.raises(ValueError) as tune_exc:
        tune_ml_estimator(
            FakeSearch(best=_best()),
            x_train=[[0.0]],
            y_train=y_train,
            policy=_policy(),
            label_encoding=encoding,
        )
    fittable = FakeFittableEstimator(classes=(0, 1), proba_rows=[[0.6, 0.4]])
    with pytest.raises(ValueError) as fit_exc:
        fit_ml_estimator(
            fittable,
            x_train=[[0.0]],
            y_train=y_train,
            policy=_policy(),
            label_encoding=encoding,
        )

    # どちらも fit 前検査であり、学習は駆動されない。
    assert fittable.fit_calls == []
    assert str(tune_exc.value) == str(fit_exc.value)
    assert str(tune_exc.value) == "label_encoding must be injective (duplicate encoded values)"


def test_未被覆_label_encoding_は_fit_前に_ValueError() -> None:
    """`label_encoding` が y のラベルを被覆しないと探索の駆動前に ValueError になる。

    メッセージは件数のみを含み、ラベル値そのものは載せない（ラベルが利用者の実データ
    由来のとき、トレースやログへ実データが流入するのを避けるため）。末尾の
    "(1 unmapped)" は正規表現の括弧として解釈されるため、`match=` には括弧を含まない
    部分文字列を渡し、件数とラベル値の非露出は文字列比較で pin する。
    """
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    search = FakeSearch(best=_best())

    with pytest.raises(ValueError, match="does not cover all labels in y_train") as excinfo:
        tune_ml_estimator(
            search,
            x_train=[[0.0], [1.0]],
            y_train=["refund", "secret_label"],
            policy=_policy(),
            label_encoding={"refund": 0, "cancel": 1},
        )

    message = str(excinfo.value)
    assert "(1 unmapped)" in message
    # ラベル値そのものはメッセージへ載せない（この修正の主目的）。
    assert "secret_label" not in message
    # 被覆性検査は fit 前検査であり、探索は駆動されない。
    assert search.fit_calls == []
    assert not hasattr(search, "best_params_")


def test_未被覆ラベルの件数は_distinct_件数である() -> None:
    """未被覆ラベルの件数は distinct 件数（同一ラベルが複数回出ても 1 と数える）。"""
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    search = FakeSearch(best=_best())

    with pytest.raises(ValueError) as excinfo:
        tune_ml_estimator(
            search,
            x_train=[[0.0], [1.0], [0.0], [1.0], [0.0]],
            y_train=["refund", "alpha", "beta", "alpha", "beta"],
            policy=_policy(),
            label_encoding={"refund": 0},
        )

    message = str(excinfo.value)
    # 未被覆は alpha / beta の 2 種（出現は計 4 回）。
    assert "(2 unmapped)" in message
    assert "alpha" not in message
    assert "beta" not in message
    assert search.fit_calls == []


# ---------------------------------------------------------------------------
# 受け入れ基準: FR-5 意図分類器組立入口への直結（後方互換）
# ---------------------------------------------------------------------------


async def test_成果物を意図分類器へ直渡しすると_policy_が自動解決される() -> None:
    """`TunedIntentEstimator` を直渡しすると policy 省略でも分類が動く。"""
    from oai_agentspec.runtime.intent._ml_training import tune_ml_estimator

    search = FakeSearch(best=_best(proba_rows=[[0.95, 0.03, 0.02]]))
    tuned = tune_ml_estimator(search, x_train=[[0.0]], y_train=["refund"], policy=_policy())

    clf = intent_classifier_from_ml_inference(tuned, thresholds=_thresholds())
    prediction = await clf.classify(IntentQuery(utterance="返金してほしい"))

    assert prediction.candidates
    assert prediction.candidates[0].text == "refund"
