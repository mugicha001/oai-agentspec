"""意図予測の 1 行ヘルパ（factory）。

`intent_classifier_from_model` は model + prompt + policy から
`DefaultIntentClassifier`（`DefaultContextBuilder` + `LLMCandidateGenerator`）を
組み立てる薄い便宜関数。差し替えは Protocol 経由（`DefaultIntentClassifier` を
直接組み立てる経路）で行える。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ._default import DefaultContextBuilder, DefaultIntentClassifier
from ._llm import LLMCandidateGenerator
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
