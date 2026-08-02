"""L2: `GuardrailRegistry` facade 9 経路と `factories` / `catalog` の同期検証（SDK 結合）。

`registry.py` の import が `_adapters.guardrails` 経由で `agents` を引くため L2。次の 3 群を
検証する。

1. facade メソッド集合（`GuardrailRegistry` の public メソッドから非 facade 7 件を明示除外して
   導出）と、公開窓口 `__all__` から同梱 helper でないシンボルを除いた集合が双方向一致すること。
2. facade 9 個それぞれの `inspect.signature` が対応 `factories` 関数と、許容差分 3 点
   （`name` の型 / 既定値差・`labels` 追加・`severity` 追加）を除いて完全一致すること。
3. `HELPER_DEFAULTS` のキー集合が「facade 集合 − `DI_DEPENDENT_HELPERS`」と双方向一致すること。
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from oai_agentspec.runtime import guardrails
from oai_agentspec.runtime.guardrails import factories
from oai_agentspec.runtime.guardrails.catalog import (
    DETECTOR_FACTORIES,
    DI_DEPENDENT_HELPERS,
    HELPER_DEFAULTS,
    NON_FACTORY_SYMBOLS,
    NON_GUARDRAIL_HELPERS,
)
from oai_agentspec.runtime.guardrails.registry import GuardrailRegistry

pytestmark = pytest.mark.integration

# facade 集合の導出規則（明示除外）: `GuardrailRegistry` の public メソッド（`_` 始まりを除く）
# から、facade（生成 + 登録）でない照会 / 登録メソッド 7 件を名前で明示的に除外する。
_NON_FACADE_METHODS: frozenset[str] = frozenset(
    {
        "register",
        "get",
        "names",
        "metadata",
        "boundary_of",
        "specs",
        "run_config_kwargs",
    }
)


def _facade_method_names() -> frozenset[str]:
    """`GuardrailRegistry` の public メソッドから非 facade 7 件を除いた facade 集合を返す。"""
    public_methods = {
        name
        for name, _ in inspect.getmembers(GuardrailRegistry, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    return frozenset(public_methods - _NON_FACADE_METHODS)


def _bundled_helper_identifiers() -> frozenset[str]:
    """公開窓口 `__all__` から helper ファクトリでない識別子を除いた集合を返す。"""
    return (
        frozenset(guardrails.__all__)
        - NON_GUARDRAIL_HELPERS
        - DETECTOR_FACTORIES
        - NON_FACTORY_SYMBOLS
    )


FACADE_METHOD_NAMES = _facade_method_names()


# ======================================================================
# 群 1: facade メソッド集合と同梱 helper 識別子集合の双方向一致
# ======================================================================


def test_facadeメソッド集合と同梱helper識別子集合は双方向一致する() -> None:
    """facade 集合（コード導出）と helper 識別子集合（`__all__` から除外集合を差し引き導出）が
    `==` で一致し、要素数が 9 であること（過大 = facade 追加漏れ / 過小 = helper 追加漏れの
    双方向を 1 本で検知する）。
    """
    facade = FACADE_METHOD_NAMES
    helpers = _bundled_helper_identifiers()
    assert facade == helpers
    assert len(facade) == 9


def test_guard_toolはfacade集合に含まれない() -> None:
    """`guard_tool` は guardrail 実体を返さないため facade 集合から除外される。"""
    assert "guard_tool" not in FACADE_METHOD_NAMES


def test_tool_guardrailはfacade集合に含まれる() -> None:
    """`tool_guardrail` はツール境界 guardrail を返す facade であり除外されない。"""
    assert "tool_guardrail" in FACADE_METHOD_NAMES


# ======================================================================
# 群 2: facade と factory の `inspect.signature` 同期（許容差分 3 点のみ）
# ======================================================================


def _param_tuple(param: inspect.Parameter) -> tuple[str, Any, Any, Any]:
    """比較に使う 4 属性（名前・kind・既定値・注釈）のタプルを返す。"""
    return (param.name, param.kind, param.default, param.annotation)


def _to_factory_shape(param: inspect.Parameter) -> inspect.Parameter:
    """`name` パラメータのみ factory 側の形（注釈 `str | None`・既定値 `None`）へ戻す。

    位置・`kind` は変更しない（許容差分 1 点目: facade は注釈 `str` で既定値なし）。
    """
    if param.name != "name":
        return param
    return param.replace(annotation="str | None", default=None)


@pytest.mark.parametrize("method", sorted(FACADE_METHOD_NAMES))
def test_facadeとfactoryのシグネチャは許容差分3点を除いて完全一致する(method: str) -> None:
    """facade 9 個それぞれについて、facade から末尾 `labels` / `severity` を落とし `name` を
    factory の形へ戻したパラメータ列が、factory のパラメータ列と完全一致することを検証する。

    引数名・順序・`kind`・既定値・型注釈（文字列表現）のいずれかが 1 つでも変化すると、
    facade と factory の宣言面が乖離するため、追従要否の判断ポイントとして fail させる。
    戻り値注釈は比較対象にしない。
    """
    facade_params = list(inspect.signature(getattr(GuardrailRegistry, method)).parameters.values())
    factory_params = list(inspect.signature(getattr(factories, method)).parameters.values())

    assert facade_params[0].name == "self"
    facade_params = facade_params[1:]

    # 許容差分 2, 3 点目: facade 末尾に labels、その次に severity が追加されている。
    labels_param = facade_params[-2]
    severity_param = facade_params[-1]
    assert labels_param.name == "labels"
    assert _param_tuple(labels_param) == (
        "labels",
        inspect.Parameter.KEYWORD_ONLY,
        None,
        "dict[str, Any] | None",
    )
    assert severity_param.name == "severity"
    assert _param_tuple(severity_param) == (
        "severity",
        inspect.Parameter.KEYWORD_ONLY,
        None,
        "Severity | None",
    )

    # 許容差分 1 点目: facade の name は注釈 `str` で既定値なし（キーワード必須）。
    # `_to_factory_shape` が name を無条件に書き換えるため、書き換え前にここで直接 pin する
    # （書き換え後の突合だけでは name の注釈・既定値の逸脱が吸収されて素通りする）。
    (name_param,) = [param for param in facade_params if param.name == "name"]
    assert name_param.annotation == "str"
    assert name_param.default is inspect.Parameter.empty
    assert name_param.kind is inspect.Parameter.KEYWORD_ONLY

    # 許容差分 1 点目（name の型・既定値差）のみ吸収したうえで factory と完全一致させる。
    coerced = [_to_factory_shape(param) for param in facade_params[:-2]]
    assert [_param_tuple(p) for p in coerced] == [_param_tuple(p) for p in factory_params]


# ======================================================================
# 群 3: HELPER_DEFAULTS のキー集合の双方向一致
# ======================================================================


def test_helper_defaultsのキー集合はfacade集合からdi依存分を除いたものと一致する() -> None:
    """`set(HELPER_DEFAULTS)` は「facade 集合 − `DI_DEPENDENT_HELPERS`」と `==` で一致する
    （キーの過剰 = 未知の facade への既定付与漏れ検知 / 欠落 = 既定分類漏れの双方向を検知）。
    """
    assert set(HELPER_DEFAULTS) == FACADE_METHOD_NAMES - DI_DEPENDENT_HELPERS


def test_di_dependent_helpersはfacade集合の部分集合である() -> None:
    """`DI_DEPENDENT_HELPERS` に facade でない名前が混ざっていないことを検知する。"""
    assert DI_DEPENDENT_HELPERS <= FACADE_METHOD_NAMES
