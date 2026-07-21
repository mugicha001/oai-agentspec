"""L1: `runtime.intent.__init__` の PEP 562 (`__getattr__`) 遅延再エクスポート契約。

窓口の `__all__` メンバ集合・非公開属性の隠蔽・遅延取得後のキャッシュ・未定義属性の
AttributeError・`dir()` への反映を検証する。extra 依存（pydantic）は解決済み前提。
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


_EXPECTED_ALL = {
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
    "intent_classifier_from_generator",
}


def test_all_membership_pinned() -> None:
    """`__all__` は 15 件で設計仕様通りのメンバ集合と一致する。"""
    import oai_agentspec.runtime.intent as intent_mod

    assert set(intent_mod.__all__) == _EXPECTED_ALL
    assert len(intent_mod.__all__) == 15


def test_all_does_not_include_confidence_level_description() -> None:
    """`_CONFIDENCE_LEVEL_DESCRIPTION` は非公開。`__all__` に含まれない。"""
    import oai_agentspec.runtime.intent as intent_mod

    assert "_CONFIDENCE_LEVEL_DESCRIPTION" not in intent_mod.__all__


def test_getattr_lazy_resolves_type_symbol() -> None:
    """`IntentPolicy` は遅延取得され、`types.IntentPolicy` と同一になる。"""
    import oai_agentspec.runtime.intent as intent_mod
    from oai_agentspec.runtime.intent import types as _types

    intent_mod.__dict__.pop("IntentPolicy", None)
    resolved = intent_mod.__getattr__("IntentPolicy")
    assert resolved is _types.IntentPolicy


def test_getattr_caches_resolved_symbol_in_globals() -> None:
    """`__getattr__` 経由の取得で module globals にキャッシュされ、次回同一オブジェクトを返す。"""
    import oai_agentspec.runtime.intent as intent_mod

    intent_mod.__dict__.pop("IntentPolicy", None)
    assert "IntentPolicy" not in intent_mod.__dict__
    first = intent_mod.__getattr__("IntentPolicy")
    assert "IntentPolicy" in intent_mod.__dict__
    second = intent_mod.IntentPolicy
    assert first is second


def test_getattr_lazy_resolves_protocol_symbol() -> None:
    """Protocol シンボル（`IntentClassifier`）も遅延取得できる。"""
    import oai_agentspec.runtime.intent as intent_mod

    intent_mod.__dict__.pop("IntentClassifier", None)
    cls = intent_mod.__getattr__("IntentClassifier")
    assert hasattr(cls, "classify")


def test_all_symbols_are_resolvable_via_getattr() -> None:
    """`__all__` の全シンボルが __getattr__ で解決可能（漏れがない）。"""
    import oai_agentspec.runtime.intent as intent_mod

    for name in intent_mod.__all__:
        intent_mod.__dict__.pop(name, None)
        value = intent_mod.__getattr__(name)
        assert value is not None, f"'{name}' が __getattr__ で解決できない"


def test_getattr_unknown_attribute_raises() -> None:
    """未定義属性は AttributeError を送出する。"""
    import oai_agentspec.runtime.intent as intent_mod

    with pytest.raises(AttributeError):
        intent_mod.__getattr__("Nonexistent")


def test_dir_includes_all_symbols_even_before_access() -> None:
    """`dir()` は未 import 状態でも `__all__` の全シンボルを含む。"""
    import oai_agentspec.runtime.intent as intent_mod

    listing = set(intent_mod.__dir__())
    assert _EXPECTED_ALL.issubset(listing)
