"""L1: 公開窓口からの `PolicyViolationError` 遅延再エクスポート検証（PEP 562）。

`oai_agentspec.runtime.governance` から `PolicyViolationError` を DeprecationWarning なしで取得
できること、AGT が送出する例外クラスと isinstance 互換（同一クラス）であること、extra 未導入相当
では属性アクセス時に install hint 付き `ImportError` となること（窓口 import 自体は壊れない）、
未公開属性は `AttributeError` となることを検証する。

取得値は module 属性へキャッシュされるため、未導入相当のテストでは monkeypatch でキャッシュを
除去してから `__getattr__` 経路を通す。
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

import oai_agentspec.runtime.governance as governance_window
from oai_agentspec._adapters.governance import _GOVERNANCE_INSTALL_HINT

pytestmark = pytest.mark.unit


def test_reexported_symbol_is_agt_exception_class(agt_symbols: tuple[Any, Any, Any]) -> None:
    """再エクスポート値は AGT の `PolicyViolationError` クラスそのもの（isinstance 互換）。"""
    _, _, agt_policy_violation_error = agt_symbols

    with warnings.catch_warnings():
        # 再エクスポート経路では DeprecationWarning を発生させない（エラー昇格で検知）。
        warnings.simplefilter("error", DeprecationWarning)
        from oai_agentspec.runtime.governance import PolicyViolationError

    assert PolicyViolationError is agt_policy_violation_error
    # 送出 / 捕捉が再エクスポート シンボルで成立する。
    with pytest.raises(PolicyViolationError, match="boom"):
        raise agt_policy_violation_error("boom")


def test_reexport_listed_in_all() -> None:
    """公開対象は `GovernedAgentBuilder` と `PolicyViolationError` のみ（`__all__` 契約）。"""
    assert governance_window.__all__ == ["GovernedAgentBuilder", "PolicyViolationError"]


def test_unknown_attribute_raises_attribute_error() -> None:
    """未公開属性へのアクセスは AttributeError（`__getattr__` の素通り防止）。"""
    with pytest.raises(AttributeError, match="no_such_symbol"):
        getattr(governance_window, "no_such_symbol")  # noqa: B009 - __getattr__ 経路を明示発火


def test_missing_extra_raises_install_hint_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """extra 未導入相当では属性アクセス時に install hint 付き ImportError（窓口 import は無事）。"""

    def _raise_import_error() -> Any:
        raise ImportError(_GOVERNANCE_INSTALL_HINT)

    # 取得済みキャッシュを除去して __getattr__ 経路を強制し、取得口を未導入相当へ差し替える。
    monkeypatch.delattr(governance_window, "PolicyViolationError", raising=False)
    monkeypatch.setattr("oai_agentspec._adapters.policy_violation_error_type", _raise_import_error)

    with pytest.raises(ImportError, match=r"oai-agentspec\[governance\]"):
        getattr(governance_window, "PolicyViolationError")  # noqa: B009 - __getattr__ 経路を発火


def test_dir_lists_lazy_reexport_without_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """dir() は未 import（キャッシュ未生成）でも PolicyViolationError を列挙する（__dir__）。"""
    monkeypatch.delattr(governance_window, "PolicyViolationError", raising=False)
    listed = dir(governance_window)
    assert "PolicyViolationError" in listed
    assert "GovernedAgentBuilder" in listed
