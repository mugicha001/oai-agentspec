"""L1: 同梱 guardrail helper の分類カタログ（`catalog`）の純検証（agents / SDK 非依存）。

`HELPER_DEFAULTS`（キー集合の固定・各値の labels / severity・不変性）、`HelperDefaults`
（フィールド集合の固定）、除外集合 4 種（`NON_GUARDRAIL_HELPERS` / `DETECTOR_FACTORIES` /
`NON_FACTORY_SYMBOLS` / `DI_DEPENDENT_HELPERS`）の要素集合と型、`HELPER_DEFAULTS` のキーが
`DI_DEPENDENT_HELPERS` と交わらないこと、`DETECTOR_FACTORIES` の識別子が `_detectors` の実
シンボルと一致することを検証する。あわせて `catalog` モジュールのソースが `agents` /
`openai` への import を含まないこと（NFR-1）を固定する。SDK を一切 import しない。
"""

from __future__ import annotations

import dataclasses
from dataclasses import fields
from pathlib import Path

import pytest

from oai_agentspec.runtime.guardrails import _detectors, catalog
from oai_agentspec.runtime.guardrails.catalog import (
    DETECTOR_FACTORIES,
    DI_DEPENDENT_HELPERS,
    HELPER_DEFAULTS,
    NON_FACTORY_SYMBOLS,
    NON_GUARDRAIL_HELPERS,
    HelperDefaults,
)
from oai_agentspec.runtime.guardrails.types import Severity

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# HELPER_DEFAULTS
# ----------------------------------------------------------------------


def test_helper_defaultsのキー集合は2件に固定される() -> None:
    """キー集合は injection_baseline_guardrail / canary_guardrail の 2 件のみ。

    集合の `==` で pin する（キー追加 = 過大と削除 = 過小の両方向を同時に検知するため。
    `in` による個別存在確認だけでは追加・削除の一方を見逃す）。
    """
    assert set(HELPER_DEFAULTS.keys()) == {
        "injection_baseline_guardrail",
        "canary_guardrail",
    }


def test_helper_defaultsのinjection_baseline_guardrailは値を固定される() -> None:
    """`injection_baseline_guardrail` は labels が LLM01・severity が MEDIUM。

    `labels` はキー集合の増減も検知するため辞書全体を `==` で pin する（値の単項チェック
    だけにしない）。
    """
    entry = HELPER_DEFAULTS["injection_baseline_guardrail"]
    assert dict(entry.labels) == {"owasp_llm": "LLM01"}
    assert entry.severity is Severity.MEDIUM


def test_helper_defaultsのcanary_guardrailは値を固定される() -> None:
    """`canary_guardrail` は labels が LLM07・severity が HIGH。"""
    entry = HELPER_DEFAULTS["canary_guardrail"]
    assert dict(entry.labels) == {"owasp_llm": "LLM07"}
    assert entry.severity is Severity.HIGH


def test_helper_defaultsはキーの再代入でTypeErrorになる() -> None:
    """`HELPER_DEFAULTS`（`MappingProxyType`）はキー代入で `TypeError`（不変性の pin）。"""
    with pytest.raises(TypeError):
        HELPER_DEFAULTS["canary_guardrail"] = HelperDefaults(labels={}, severity=Severity.LOW)


def test_helper_defaultsのlabelsはキー代入でTypeErrorになる() -> None:
    """既定値の `labels`（`MappingProxyType`）もキー代入で `TypeError`（不変性の pin）。

    `labels` のラップはエントリごとに個別なので、登録済みの全エントリを回す（1 エントリだけを
    見ると他エントリのラップ漏れが素通りする）。
    """
    for entry in HELPER_DEFAULTS.values():
        with pytest.raises(TypeError):
            entry.labels["owasp_llm"] = "X"


def test_helper_defaultsのseverityは代入でFrozenInstanceErrorになる() -> None:
    """`HelperDefaults` は `frozen=True` のため属性代入で `FrozenInstanceError`（不変性の pin）。"""
    with pytest.raises(dataclasses.FrozenInstanceError):
        HELPER_DEFAULTS["canary_guardrail"].severity = Severity.LOW


# ----------------------------------------------------------------------
# HelperDefaults
# ----------------------------------------------------------------------


def test_helperdefaultsのフィールド集合はlabelsとseverityに固定される() -> None:
    """`HelperDefaults` のフィールドは labels / severity の 2 件のみ。

    集合の `==` で pin する（フィールド追加・削除の両方向を同時に検知するため）。
    """
    assert {f.name for f in fields(HelperDefaults)} == {"labels", "severity"}


# ----------------------------------------------------------------------
# 除外集合 4 種
# ----------------------------------------------------------------------


def test_non_guardrail_helpersはguard_tool1件に固定される() -> None:
    """`NON_GUARDRAIL_HELPERS` は `guard_tool` の 1 件のみで `frozenset`。"""
    assert isinstance(NON_GUARDRAIL_HELPERS, frozenset)
    assert NON_GUARDRAIL_HELPERS == {"guard_tool"}


def test_detector_factoriesは6識別子に固定される() -> None:
    """`DETECTOR_FACTORIES` は 6 件の検知器ファクトリ識別子のみで `frozenset`。"""
    assert isinstance(DETECTOR_FACTORIES, frozenset)
    assert DETECTOR_FACTORIES == {
        "canary_detector",
        "regex_detector",
        "length_detector",
        "allow_deny_detector",
        "predicate_detector",
        "injection_baseline_detector",
    }


def test_non_factory_symbolsは11識別子に固定される() -> None:
    """`NON_FACTORY_SYMBOLS` は 11 件の非ファクトリ公開シンボルのみで `frozenset`。"""
    assert isinstance(NON_FACTORY_SYMBOLS, frozenset)
    assert NON_FACTORY_SYMBOLS == {
        "GuardrailSpec",
        "Boundary",
        "Severity",
        "GuardrailRegistry",
        "Detection",
        "HelperDefaults",
        "HELPER_DEFAULTS",
        "INJECTION_BASELINE_PATTERNS",
        "SQLI_PATTERNS",
        "COMMAND_INJECTION_PATTERNS",
        "PATH_TRAVERSAL_PATTERNS",
    }


def test_di_dependent_helpersは7識別子に固定される() -> None:
    """`DI_DEPENDENT_HELPERS` は 7 件の DI 依存 helper 識別子のみで `frozenset`。"""
    assert isinstance(DI_DEPENDENT_HELPERS, frozenset)
    assert DI_DEPENDENT_HELPERS == {
        "prompt_llm_guardrail",
        "predicate_guardrail",
        "regex_guardrail",
        "length_guardrail",
        "allow_deny_guardrail",
        "external_detector_guardrail",
        "tool_guardrail",
    }


def test_helper_defaultsのキーはdi_dependent_helpersと交わらない() -> None:
    """既定分類を持つ helper と DI 依存 helper は排他（DI 依存に既定値を持たせない不変条件）。"""
    assert set(HELPER_DEFAULTS.keys()).isdisjoint(DI_DEPENDENT_HELPERS)


def test_detector_factoriesの識別子はdetectorsモジュールの実シンボルと一致する() -> None:
    """`DETECTOR_FACTORIES` の 6 識別子が `_detectors` モジュールに実在する（タイポ検知）。"""
    for name in DETECTOR_FACTORIES:
        assert hasattr(_detectors, name), f"_detectors に {name} が存在しません"


# ----------------------------------------------------------------------
# SDK 隔離（NFR-1）
# ----------------------------------------------------------------------


def test_catalogモジュールはagentsとopenaiへの参照を含まない() -> None:
    """`catalog.py` のソースに `agents` / `openai` への import 文が現れない（NFR-1）。"""
    source = Path(catalog.__file__).read_text(encoding="utf-8")
    assert "from agents" not in source
    assert "import agents" not in source
    assert "from openai" not in source
    assert "import openai" not in source
