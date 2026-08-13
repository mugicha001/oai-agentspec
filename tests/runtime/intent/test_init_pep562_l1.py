"""L1: `runtime.intent.__init__` の PEP 562 (`__getattr__`) 遅延再エクスポート契約。

窓口の `__all__` メンバ集合・非公開属性の隠蔽・遅延取得後のキャッシュ・未定義属性の
AttributeError・`dir()` への反映を検証する。extra 依存（pydantic）は解決済み前提。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"


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
    "confidence_mapper_from_thresholds",
    "prediction_from_scored_labels",
    "MLCandidateGenerator",
    "IntentTrainer",
    "TrainedIntentEstimator",
    "make_trained_estimator",
    "ml_inference_from_estimator",
    "fit_ml_estimator",
    "intent_classifier_from_model",
    "intent_classifier_from_generator",
    "intent_classifier_from_ml_inference",
    # --- アクション宣言（FR-1/FR-2/FR-3） ---
    "ActionSpec",
    "ActionCatalog",
    "ActionPlanner",
    "ParameterSpec",
    "param",
    "PARAM_UNSET",
    # --- スロットと計画（FR-5/FR-8） ---
    "Slot",
    "SlotState",
    "Origin",
    "SlotSuggestion",
    "ActionPlan",
    "PlanResult",
    "ParamUsage",
    # --- 結線の宣言型（FR-3） ---
    "CandidateSource",
    "LLMFiller",
    # --- 候補契約（FR-4） ---
    "ExecutableIntent",
    "ExecutableSuggestion",
}


def _assert_lazy_submodule_resolution(symbol: str, submodule: str) -> None:
    """窓口 import 直後は `submodule` 未 import、`symbol` アクセス後に import される性質を pin。

    親パッケージ (`oai_agentspec` / `oai_agentspec.runtime`) をクリーンな子プロセスで
    import し直すことで、他テストの import 順序に左右されずに切り分ける。

    Args:
        symbol: 遅延解決される公開シンボル名。
        submodule: 解決先のフル修飾サブモジュール名。

    Raises:
        AssertionError: 遅延解決の性質が壊れている場合。
    """
    probe = (
        "import sys\n"
        "import oai_agentspec.runtime.intent as intent_mod\n"
        f"assert {submodule!r} not in sys.modules, 'import 直後に既に読み込まれている'\n"
        f"_ = intent_mod.{symbol}\n"
        f"assert {submodule!r} in sys.modules, '属性アクセス後に読み込まれていない'\n"
        "print('OK')\n"
    )
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_SRC_DIR) + (os.pathsep + existing if existing else "")
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        timeout=30,
    )
    assert result.stdout.strip() == "OK"


def test_all_membership_pinned() -> None:
    """`__all__` は 41 件で設計仕様通りのメンバ集合と一致する。"""
    import oai_agentspec.runtime.intent as intent_mod

    assert set(intent_mod.__all__) == _EXPECTED_ALL
    assert len(intent_mod.__all__) == 41


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


def test_getattr_lazy_resolves_executable_intent_symbol() -> None:
    """`ExecutableIntent` は `_TYPE_SYMBOLS` 経由で `types.ExecutableIntent` と同一になる。"""
    import oai_agentspec.runtime.intent as intent_mod
    from oai_agentspec.runtime.intent import types as _types

    intent_mod.__dict__.pop("ExecutableIntent", None)
    resolved = intent_mod.__getattr__("ExecutableIntent")
    assert resolved is _types.ExecutableIntent


def test_getattr_lazy_resolves_executable_suggestion_symbol() -> None:
    """`ExecutableSuggestion` は `_TYPE_SYMBOLS` 経由で `types.ExecutableSuggestion` と同一。"""
    import oai_agentspec.runtime.intent as intent_mod
    from oai_agentspec.runtime.intent import types as _types

    intent_mod.__dict__.pop("ExecutableSuggestion", None)
    resolved = intent_mod.__getattr__("ExecutableSuggestion")
    assert resolved is _types.ExecutableSuggestion


def test_getattr_lazy_resolves_action_symbol() -> None:
    """`ActionPlanner` は遅延取得され、`actions.ActionPlanner` と同一になる。

    `ActionSpec` ではなく `ActionPlanner` を対象にする。`slots.py` が
    `from .actions import ActionSpec` を持つため `ActionSpec` は `slots` 経由でも
    解決でき、`_ACTION_SYMBOLS` の解決先差し替え（`actions` -> `slots`）を検知できない
    （pin として無効になる）ため。
    """
    import oai_agentspec.runtime.intent as intent_mod
    from oai_agentspec.runtime.intent import actions as _actions

    intent_mod.__dict__.pop("ActionPlanner", None)
    resolved = intent_mod.__getattr__("ActionPlanner")
    assert resolved is _actions.ActionPlanner


def test_getattr_lazy_resolves_slot_symbol() -> None:
    """`Slot` は遅延取得され、`slots.Slot` と同一になる。"""
    import oai_agentspec.runtime.intent as intent_mod
    from oai_agentspec.runtime.intent import slots as _slots

    intent_mod.__dict__.pop("Slot", None)
    resolved = intent_mod.__getattr__("Slot")
    assert resolved is _slots.Slot


def test_getattr_lazy_resolves_binding_symbol() -> None:
    """`CandidateSource` は遅延取得され、`binding.CandidateSource` と同一になる。"""
    import oai_agentspec.runtime.intent as intent_mod
    from oai_agentspec.runtime.intent import binding as _binding

    intent_mod.__dict__.pop("CandidateSource", None)
    resolved = intent_mod.__getattr__("CandidateSource")
    assert resolved is _binding.CandidateSource


def test_action_symbol_submodule_is_lazily_imported() -> None:
    """`actions` サブモジュールは `ActionPlanner` アクセスまで `sys.modules` に載らない。"""
    _assert_lazy_submodule_resolution("ActionPlanner", "oai_agentspec.runtime.intent.actions")


def test_slot_symbol_submodule_is_lazily_imported() -> None:
    """`slots` サブモジュールは `Slot` アクセスまで `sys.modules` に載らない。"""
    _assert_lazy_submodule_resolution("Slot", "oai_agentspec.runtime.intent.slots")


def test_binding_symbol_submodule_is_lazily_imported() -> None:
    """`binding` サブモジュールは `CandidateSource` アクセスまで `sys.modules` に載らない。"""
    _assert_lazy_submodule_resolution("CandidateSource", "oai_agentspec.runtime.intent.binding")


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
