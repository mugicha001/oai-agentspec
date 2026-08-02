"""L1: 同梱 guardrail helper の分類カタログ（`catalog`）の純検証（agents / SDK 非依存）。

`HELPER_DEFAULTS`（キー集合の固定・各値の labels / severity・不変性）、`HelperDefaults`
（フィールド集合の固定）、除外集合 4 種（`NON_GUARDRAIL_HELPERS` / `DETECTOR_FACTORIES` /
`NON_FACTORY_SYMBOLS` / `DI_DEPENDENT_HELPERS`）の要素集合と型、`HELPER_DEFAULTS` のキーが
`DI_DEPENDENT_HELPERS` と交わらないこと、`DETECTOR_FACTORIES` の識別子が `_detectors` の実
シンボルと一致することを検証する。あわせて docs の分類表（`docs/usage/safety/guardrails.md` の
`## helper の framework 分類と既定危険度`）の framework ラベル列がコード側の分類と一致すること、
`catalog` モジュールのソースが `agents` / `openai` への import を含まないこと（NFR-1）を固定する。
SDK を一切 import しない。
"""

from __future__ import annotations

import dataclasses
import re
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
# docs 分類表との照合（framework ラベル列のみ）
# ----------------------------------------------------------------------

_FRAMEWORK_SECTION_HEADING = "## helper の framework 分類と既定危険度"
_FRAMEWORK_TABLE_HEADER = ["helper 識別子", "適用境界", "framework ラベル", "既定危険度", "備考"]
_LABEL_COLUMN = 2
_DECLARED_BY_USER = "利用者が宣言"
_HELPER_IDENT_CELL = re.compile(r"^`([a-z_]+)`$")
_LABEL_CELL = re.compile(r"^`([a-z_]+): ([^`]+)`$")


def _framework_table() -> list[list[str]]:
    """docs の framework 分類表を見出し起点で parse し、行ごとのセル列を返す（区切り行は除外）。

    行番号にはハードコード依存せず、見出しから次の `## ` 見出しまでの範囲で行頭が `|` の行のみを
    対象にする。見出しが見つからなければ `list.index` が `ValueError` を上げる（表が消えたまま
    assert が空振りするのを防ぐ）。

    Returns:
        表の行（先頭が見出し行）。各行はセル文字列のリスト。
    """
    doc = Path(__file__).resolve().parents[3] / "docs" / "usage" / "safety" / "guardrails.md"
    lines = doc.read_text(encoding="utf-8").splitlines()
    start = lines.index(_FRAMEWORK_SECTION_HEADING)
    rows: list[list[str]] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def _framework_label_column() -> tuple[dict[str, dict[str, str]], set[str]]:
    """表のデータ行を framework ラベル列で 2 分し、既定を持つ helper と DI 依存 helper を返す。

    Returns:
        `({helper 識別子: ラベル dict}, {利用者が宣言の helper 識別子})`。
    """
    defaults: dict[str, dict[str, str]] = {}
    declared_by_user: set[str] = set()
    for cells in _framework_table()[1:]:
        ident = _HELPER_IDENT_CELL.match(cells[0])
        assert ident is not None, f"helper 識別子セルの書式が想定外です: {cells[0]!r}"
        label = cells[_LABEL_COLUMN]
        if label == _DECLARED_BY_USER:
            declared_by_user.add(ident.group(1))
            continue
        matched = _LABEL_CELL.match(label)
        assert matched is not None, f"framework ラベルセルの書式が想定外です: {label!r}"
        defaults[ident.group(1)] = {matched.group(1): matched.group(2)}
    return defaults, declared_by_user


def test_docsのframework分類表は列構成を保つ() -> None:
    """表の見出し行のセル列を `==` で pin する（照合対象の列位置が動かないことの前提）。

    列を入れ替えても parse は成功してしまい、別列を framework ラベルとして読む。列構成ごと
    固定して、ラベル列の位置（`_LABEL_COLUMN`）が有効であることを担保する。
    """
    assert _framework_table()[0] == _FRAMEWORK_TABLE_HEADER


def test_docsのframework分類表の全データ行がラベル列で分類される() -> None:
    """データ行数と 2 群（既定を持つ / 利用者が宣言）の合計が一致する。

    書式が想定外の行は `_framework_label_column` の assert が落とすため、分類から静かに漏れる
    経路はない。本テストが単独で検知するのは (1) 同一 helper 識別子の行が重複していて dict /
    set へ畳み込まれる場合と、(2) データ行が 0 件で後続の集合一致テストが空振りする場合。
    """
    defaults, declared_by_user = _framework_label_column()
    data_rows = _framework_table()[1:]

    assert len(data_rows) > 0
    assert len(data_rows) == len(defaults) + len(declared_by_user)


def test_docsのframeworkラベル列はhelper_defaultsと一致する() -> None:
    """表の framework ラベル列が `HELPER_DEFAULTS` の labels と `==` で一致する。

    docs はコードの投影なので、ラベルの食い違い・helper の過不足を双方向で検知する。既定危険度
    列は照合対象に含めない（ライブラリが付す値は運用組織のポリシー判断で上書きされる出発点で
    あり、docs 表と固定する対象ではないという設計判断。ADR 0015 の該当決定を参照）。
    """
    defaults, _ = _framework_label_column()

    assert defaults == {name: dict(entry.labels) for name, entry in HELPER_DEFAULTS.items()}


def test_docsの利用者宣言行はdi_dependent_helpersと一致する() -> None:
    """表で「利用者が宣言」とされた helper 集合が `DI_DEPENDENT_HELPERS` と `==` で一致する。"""
    _, declared_by_user = _framework_label_column()

    assert declared_by_user == set(DI_DEPENDENT_HELPERS)


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
