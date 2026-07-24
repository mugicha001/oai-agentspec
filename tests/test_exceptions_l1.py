"""L1: `oai_agentspec.exceptions` 統一窓口の契約テスト（直 import + PEP 562 遅延）。

`runtime/resilience/__init__.py` の窓口テストパターンを踏襲する。lib 独自例外 9 種を
再エクスポートする窓口で、コア層と外部依存ゼロの extra 例外（7 種）は直 import、
外部依存を持つ extra 例外（`OptimizeError` / `ConversationClientError`）は PEP 562 遅延取得。
定義実体は既存モジュールに残るため isinstance / issubclass は完全互換。
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


_DIRECT_SYMBOLS = {
    "RegistryFrozenError",
    "IntegrityError",
    "PromptTemplateIntegrityError",
    "PromptResolutionError",
    "WorkflowFrozenError",
    "RunBudgetExceeded",
    "ConversationError",
}
_LAZY_SYMBOLS = {
    "OptimizeError",
    "ConversationClientError",
}
_EXPECTED_ALL = _DIRECT_SYMBOLS | _LAZY_SYMBOLS


def test_all_membership_pinned() -> None:
    """`__all__` は 9 件で設計仕様通りのメンバ集合と一致する。"""
    from oai_agentspec import exceptions as mod

    assert set(mod.__all__) == _EXPECTED_ALL
    assert len(mod.__all__) == 9


def test_direct_symbols_are_directly_imported() -> None:
    """直 import 対象 7 種は module import 時点で `__dict__` に載る。"""
    from oai_agentspec import exceptions as mod

    for name in _DIRECT_SYMBOLS:
        assert name in mod.__dict__, f"'{name}' は直 import されているべき"


def test_lazy_symbols_resolve_and_cache() -> None:
    """遅延 2 種は `__getattr__` 経由で解決でき、再取得で同一オブジェクトを返す。"""
    from oai_agentspec import exceptions as mod

    for name in _LAZY_SYMBOLS:
        mod.__dict__.pop(name, None)
        first = getattr(mod, name)
        assert first is not None, f"'{name}' が遅延取得できない"
        assert name in mod.__dict__
        second = getattr(mod, name)
        assert first is second, f"'{name}' がキャッシュされていない"


def test_lazy_symbols_match_definition_module() -> None:
    """遅延 2 種の実体は定義元モジュールのクラスと `is` 一致する（完全互換）。"""
    from oai_agentspec import exceptions as mod
    from oai_agentspec.runtime.cli import _models as cli_models
    from oai_agentspec.runtime.lightning import types as lightning_types

    mod.__dict__.pop("OptimizeError", None)
    mod.__dict__.pop("ConversationClientError", None)
    assert mod.OptimizeError is lightning_types.OptimizeError
    assert mod.ConversationClientError is cli_models.ConversationClientError


def test_direct_symbols_match_definition_modules() -> None:
    """直 import 7 種の実体も定義元と `is` 一致する（isinstance / issubclass 完全互換）。"""
    from oai_agentspec import exceptions as mod
    from oai_agentspec.integrity import IntegrityError, PromptTemplateIntegrityError
    from oai_agentspec.prompts import PromptResolutionError
    from oai_agentspec.registry import RegistryFrozenError
    from oai_agentspec.runtime.conversation.types import ConversationError
    from oai_agentspec.runtime.resilience._errors import RunBudgetExceeded
    from oai_agentspec.workflow.graph import WorkflowFrozenError

    assert mod.RegistryFrozenError is RegistryFrozenError
    assert mod.IntegrityError is IntegrityError
    assert mod.PromptTemplateIntegrityError is PromptTemplateIntegrityError
    assert mod.PromptResolutionError is PromptResolutionError
    assert mod.WorkflowFrozenError is WorkflowFrozenError
    assert mod.RunBudgetExceeded is RunBudgetExceeded
    assert mod.ConversationError is ConversationError


def test_all_symbols_are_resolvable_and_are_exception_classes() -> None:
    """`__all__` の全 9 シンボルは getattr で解決でき、いずれも BaseException のサブクラス。"""
    from oai_agentspec import exceptions as mod

    for name in mod.__all__:
        value = getattr(mod, name)
        assert value is not None, f"'{name}' が解決できない"
        assert issubclass(value, BaseException), f"'{name}' が例外クラスではない"


def test_getattr_unknown_attribute_raises() -> None:
    """未定義属性は AttributeError を送出する。"""
    from oai_agentspec import exceptions as mod

    with pytest.raises(AttributeError):
        mod.__getattr__("nonexistent")


def test_dir_includes_all_symbols_even_before_access() -> None:
    """`dir()` は未 import 状態でも `__all__` の全 9 シンボルを含む。"""
    from oai_agentspec import exceptions as mod

    listing = set(mod.__dir__())
    assert _EXPECTED_ALL.issubset(listing)
