"""意図予測の公開窓口（`oai-agentspec[intent]` extra・agents 非依存・公開 API）。

意図分類の型（`IntentQuery` / `IntentContext` / `IntentCategory` / `IntentPolicy` /
`IntentPrediction` / `IntentCandidate` / `ConsistencyReport` / `ConfidenceLevel`）、
Protocol（`IntentClassifier` / `ContextBuilder` / `CandidateGenerator`）、
デフォルト実装（`DefaultIntentClassifier` / `LLMCandidateGenerator`）、および 1 行
ヘルパ（`intent_classifier_from_model`）を再エクスポートする。

型は pydantic に依存するため、本窓口は PEP 562 (`__getattr__`) による遅延再エクスポートに
統一する。窓口 import 自体は intent extra 未導入でも壊れず、属性アクセス時に初めて
pydantic を含む依存を import する（未導入時は import 例外が案内される）。SDK
（`agents`）は本窓口で扱わない（`_adapters/intent.py` に閉じる・NFR-1）。

`_CONFIDENCE_LEVEL_DESCRIPTION` は module 内部の実装詳細であり `__all__` に含めない。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ConfidenceLevel",
    "IntentQuery",
    "IntentContext",
    "IntentCategory",
    "IntentPolicy",
    "IntentPrediction",
    "IntentCandidate",
    "ConsistencyReport",
    "IntentClassifier",
    "ContextBuilder",
    "CandidateGenerator",
    "DefaultIntentClassifier",
    "LLMCandidateGenerator",
    "intent_classifier_from_model",
]


_TYPE_SYMBOLS = frozenset(
    {
        "ConfidenceLevel",
        "IntentQuery",
        "IntentContext",
        "IntentCategory",
        "IntentPolicy",
        "IntentPrediction",
        "IntentCandidate",
        "ConsistencyReport",
    }
)
_PROTOCOL_SYMBOLS = frozenset({"IntentClassifier", "ContextBuilder", "CandidateGenerator"})
_DEFAULT_SYMBOLS = frozenset({"DefaultIntentClassifier"})
_LLM_SYMBOLS = frozenset({"LLMCandidateGenerator"})
_FACTORY_SYMBOLS = frozenset({"intent_classifier_from_model"})


def __getattr__(name: str) -> Any:
    """PEP 562: 公開シンボルを遅延 import する。

    intent extra（pydantic）未導入時でも窓口 import は成功し、属性アクセスで初めて
    依存を要求する。取得済みの値は module 属性へキャッシュする。

    Args:
        name: アクセスされた属性名。

    Returns:
        該当する公開シンボル。

    Raises:
        AttributeError: 公開しない属性名の場合。
        ImportError: intent extra が未導入で pydantic が import できない場合。
    """
    if name in _TYPE_SYMBOLS:
        from . import types as _types

        value = getattr(_types, name)
    elif name in _PROTOCOL_SYMBOLS:
        from . import protocols as _protocols

        value = getattr(_protocols, name)
    elif name in _DEFAULT_SYMBOLS:
        from ._default import DefaultIntentClassifier

        value = DefaultIntentClassifier
    elif name in _LLM_SYMBOLS:
        from ._llm import LLMCandidateGenerator

        value = LLMCandidateGenerator
    elif name in _FACTORY_SYMBOLS:
        from .factories import intent_classifier_from_model

        value = intent_classifier_from_model
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """`dir()` に遅延再エクスポート分を未 import でも含める。"""
    return sorted(set(globals()) | set(__all__))
