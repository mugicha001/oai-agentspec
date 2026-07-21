"""意図予測の 1 行ヘルパ（factory）。

`intent_classifier_from_model` は model + prompt + policy から
`DefaultIntentClassifier`（`DefaultContextBuilder` + `LLMCandidateGenerator`）を
組み立てる薄い便宜関数。`intent_classifier_from_generator` はその対称形で、
自作 `CandidateGenerator` から同構成を組み立てる（LLM 不使用の分類器等）。
ContextBuilder まで差し替える場合は `DefaultIntentClassifier` を直接組み立てる。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ._default import DefaultContextBuilder, DefaultIntentClassifier
from ._llm import LLMCandidateGenerator
from .protocols import CandidateGenerator
from .types import IntentContext, IntentPolicy


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
