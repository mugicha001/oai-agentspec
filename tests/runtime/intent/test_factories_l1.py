"""L1: `runtime.intent.factories.intent_classifier_from_model` の組み立て契約ピン留め。

model / prompt / policy / history_limit / include_policy_in_system が
DefaultIntentClassifier(DefaultContextBuilder + LLMCandidateGenerator) の
対応する属性へそのまま流れることを検証する。実 SDK / 実 LLM は呼ばない。
"""

from __future__ import annotations

from typing import Any

import pytest

from oai_agentspec.runtime.intent._default import (
    DefaultContextBuilder,
    DefaultIntentClassifier,
)
from oai_agentspec.runtime.intent._llm import LLMCandidateGenerator
from oai_agentspec.runtime.intent.factories import intent_classifier_from_model
from oai_agentspec.runtime.intent.types import (
    IntentCategory,
    IntentContext,
    IntentPolicy,
)

pytestmark = pytest.mark.unit


def _policy() -> IntentPolicy:
    """テスト用の最小 IntentPolicy を返す。"""
    return IntentPolicy(
        categories=(
            IntentCategory(name="ask", description="質問"),
            IntentCategory(name="chitchat", description="雑談"),
        ),
    )


def _prompt(ctx: IntentContext[Any]) -> str:
    """テスト用の prompt callable。"""
    return ctx.utterance


def test_factory_returns_default_intent_classifier() -> None:
    """factory は DefaultIntentClassifier インスタンスを返す。"""
    clf = intent_classifier_from_model(object(), _prompt, policy=_policy())
    assert isinstance(clf, DefaultIntentClassifier)


def test_factory_context_builder_default_history_limit_is_20() -> None:
    """context_builder は DefaultContextBuilder で history_limit デフォルト 20。"""
    clf = intent_classifier_from_model(object(), _prompt, policy=_policy())
    assert isinstance(clf.context_builder, DefaultContextBuilder)
    assert clf.context_builder.history_limit == 20


def test_factory_context_builder_history_limit_is_propagated() -> None:
    """history_limit=5 を渡すと DefaultContextBuilder に反映される。"""
    clf = intent_classifier_from_model(object(), _prompt, policy=_policy(), history_limit=5)
    assert isinstance(clf.context_builder, DefaultContextBuilder)
    assert clf.context_builder.history_limit == 5


def test_factory_generator_is_llm_candidate_generator_with_policy() -> None:
    """generator は LLMCandidateGenerator で、policy がそのまま保持される。"""
    policy = _policy()
    clf = intent_classifier_from_model(object(), _prompt, policy=policy)
    assert isinstance(clf.generator, LLMCandidateGenerator)
    assert clf.generator._policy is policy


def test_factory_include_policy_in_system_default_true() -> None:
    """include_policy_in_system のデフォルトは True。"""
    clf = intent_classifier_from_model(object(), _prompt, policy=_policy())
    assert isinstance(clf.generator, LLMCandidateGenerator)
    assert clf.generator._include_policy_in_system is True


def test_factory_include_policy_in_system_false_is_propagated() -> None:
    """include_policy_in_system=False がそのまま generator に反映される。"""
    clf = intent_classifier_from_model(
        object(), _prompt, policy=_policy(), include_policy_in_system=False
    )
    assert isinstance(clf.generator, LLMCandidateGenerator)
    assert clf.generator._include_policy_in_system is False


def test_factory_prompt_callable_is_stored_in_generator() -> None:
    """prompt callable がそのまま LLMCandidateGenerator._prompt に格納される。"""
    clf = intent_classifier_from_model(object(), _prompt, policy=_policy())
    assert isinstance(clf.generator, LLMCandidateGenerator)
    assert clf.generator._prompt is _prompt


def test_factory_model_is_stored_in_generator() -> None:
    """model がそのまま LLMCandidateGenerator._model に格納される（不透明値）。"""
    sentinel = object()
    clf = intent_classifier_from_model(sentinel, _prompt, policy=_policy())
    assert isinstance(clf.generator, LLMCandidateGenerator)
    assert clf.generator._model is sentinel


def test_factory_passes_model_settings_to_generator() -> None:
    """model_settings がそのまま LLMCandidateGenerator._model_settings に格納される（不透明値）。"""
    sentinel = object()
    clf = intent_classifier_from_model(object(), _prompt, policy=_policy(), model_settings=sentinel)
    assert isinstance(clf.generator, LLMCandidateGenerator)
    assert clf.generator._model_settings is sentinel


def test_factory_model_settings_default_none() -> None:
    """model_settings 未指定時、generator の _model_settings は None（既定値 pin）。"""
    clf = intent_classifier_from_model(object(), _prompt, policy=_policy())
    assert isinstance(clf.generator, LLMCandidateGenerator)
    assert clf.generator._model_settings is None


def test_factory_policy_is_keyword_only() -> None:
    """policy は keyword-only（位置引数で渡すと TypeError）。"""
    with pytest.raises(TypeError):
        intent_classifier_from_model(object(), _prompt, _policy())  # type: ignore[misc]
