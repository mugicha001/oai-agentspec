"""意図予測の 1 行ヘルパ（factory）。

`intent_classifier_from_model` は model + prompt + policy から
`DefaultIntentClassifier`（`DefaultContextBuilder` + `LLMCandidateGenerator`）を
組み立てる薄い便宜関数。`intent_classifier_from_generator` はその対称形で、
自作 `CandidateGenerator` から同構成を組み立てる（LLM 不使用の分類器等）。
`intent_classifier_from_ml_inference` は ML 推論 callable（または
`TrainedIntentEstimator`）から同構成を組み立てる、LLM 版の対称形。
ContextBuilder まで差し替える場合は `DefaultIntentClassifier` を直接組み立てる。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ._default import DefaultContextBuilder, DefaultIntentClassifier
from ._llm import LLMCandidateGenerator
from ._ml import MLCandidateGenerator, confidence_mapper_from_thresholds
from ._ml_training import TrainedIntentEstimator
from .protocols import CandidateGenerator
from .types import ConfidenceLevel, IntentContext, IntentPolicy


def intent_classifier_from_model(
    model: Any,
    prompt: Callable[[IntentContext[Any]], str],
    *,
    policy: IntentPolicy,
    history_limit: int = 20,
    include_policy_in_system: bool = True,
    model_settings: Any | None = None,
) -> DefaultIntentClassifier:
    """LLM モデルから既定構成の `DefaultIntentClassifier` を組み立てる。

    Args:
        model: LLM モデル（`agents.Model` 相当・不透明型）。
        prompt: `IntentContext` から user 入力文字列を組み立てる callable。
        policy: 分類器が守る契約。
        history_limit: `DefaultContextBuilder` が history から取得する上限件数。
        include_policy_in_system: True なら `policy.render_prompt()` を system に注入する。
        model_settings: agents.ModelSettings 相当（不透明型）。None なら SDK 既定に委ねる。
            `LLMCandidateGenerator` へそのまま pass-through する。

    Returns:
        `DefaultContextBuilder` + `LLMCandidateGenerator` を束ねた
        `DefaultIntentClassifier`。
    """
    return DefaultIntentClassifier(
        context_builder=DefaultContextBuilder(history_limit=history_limit),
        generator=LLMCandidateGenerator(
            model,
            prompt,
            policy=policy,
            include_policy_in_system=include_policy_in_system,
            model_settings=model_settings,
        ),
    )


def intent_classifier_from_generator(
    generator: CandidateGenerator,
    *,
    history_limit: int = 20,
) -> DefaultIntentClassifier:
    """自作 `CandidateGenerator` から既定構成の `DefaultIntentClassifier` を組み立てる。

    `intent_classifier_from_model` の対称形。LLM を使わない generator（キーワード
    マッチ・embedding 等）を Protocol DI で差し込む際の 1 行ヘルパ。generator の
    型検証は行わず素通しで格納する（既存 factory と同一の非検証契約。誤った
    オブジェクトを渡した場合は初回 `classify()` 時に顕在化する）。

    Args:
        generator: `CandidateGenerator` Protocol を満たす自作実装。`IntentPolicy` の
            強制（allowlist / sort / truncate）は generator 実装の責務
            （`protocols.CandidateGenerator` の docstring 参照）。
        history_limit: `DefaultContextBuilder` が history から取得する上限件数。

    Returns:
        `DefaultContextBuilder` + 渡された generator を束ねた `DefaultIntentClassifier`。
    """
    return DefaultIntentClassifier(
        context_builder=DefaultContextBuilder(history_limit=history_limit),
        generator=generator,
    )


def intent_classifier_from_ml_inference(
    inference: Any,
    *,
    policy: IntentPolicy | None = None,
    mapper: Callable[[float], ConfidenceLevel] | None = None,
    thresholds: Mapping[str, float] | None = None,
    history_limit: int = 20,
) -> DefaultIntentClassifier:
    """ML 推論 callable から既定構成の `DefaultIntentClassifier` を組み立てる。

    `intent_classifier_from_model` の対称形。`inference` は `IntentContext` を受け取り
    (ラベル, スコア) 列を返す callable、または `fit_ml_estimator` 等が返す
    `TrainedIntentEstimator` をそのまま渡してよい（後者の場合は内部で `.inference`
    を取り出す）。

    Args:
        inference: (ラベル, スコア) 列を返す推論 callable、または
            `TrainedIntentEstimator`。
        policy: 分類器が守る契約。keyword-only。省略時は `TrainedIntentEstimator`
            直渡しの成果物が保持する policy から自動解決する（明示指定が優先）。
        mapper: スコアを `ConfidenceLevel` に変換する callable。keyword-only。
            `thresholds` と同時指定不可（排他）。
        thresholds: 5 段階名（`certain` / `high` / `medium` / `low` / `speculative`）を
            キーとする閾値マッピング。keyword-only。内部で
            `confidence_mapper_from_thresholds` に展開する。`mapper` と同時指定不可（排他）。
        history_limit: `DefaultContextBuilder` が history から取得する上限件数。
            keyword-only。

    Returns:
        `DefaultContextBuilder` + `MLCandidateGenerator` を束ねた
        `DefaultIntentClassifier`。

    Raises:
        ValueError: `mapper` と `thresholds` を両方指定、または両方省略した場合。
            policy を指定せず、`TrainedIntentEstimator` からも解決できない場合。
        TypeError: `thresholds` が必要な 5 段階名（`certain` / `high` / `medium` /
            `low` / `speculative`）を欠く、または未知のキーを含む場合
            （`confidence_mapper_from_thresholds` への keyword 引数展開時に発生）。
    """
    resolved_policy = policy
    if resolved_policy is None and isinstance(inference, TrainedIntentEstimator):
        resolved_policy = inference.policy
    if resolved_policy is None:
        raise ValueError(
            "policy を指定するか、policy を保持した TrainedIntentEstimator を渡してください"
        )

    if mapper is None and thresholds is None:
        raise ValueError("mapper と thresholds のいずれかを指定してください")
    if mapper is not None and thresholds is not None:
        raise ValueError("mapper と thresholds は同時指定できません")

    resolved_mapper = (
        mapper
        if mapper is not None
        else confidence_mapper_from_thresholds(
            **thresholds  # type: ignore[arg-type]
        )
    )
    resolved_inference = (
        inference.inference if isinstance(inference, TrainedIntentEstimator) else inference
    )

    return intent_classifier_from_generator(
        MLCandidateGenerator(resolved_inference, policy=resolved_policy, mapper=resolved_mapper),
        history_limit=history_limit,
    )
