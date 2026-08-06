"""L1: FR-6（テスト専用を含意しない命名）の静的照合。

窓口を import せず `ast` で `__init__.py` を静的解析する方式で行う（設計方針 WARN-9 の決定）。
`pkgutil` / `walk_packages` はリポジトリに使用前例が無いため採らず、import 方式は extra
未導入環境で skip され保証が空洞化するため採らない。

`_adapters/` は内部窓口（利用者が直接 import する公開契約ではない）のため照合対象から除外する。
既存の `_adapters.__all__` の `mock_spec_tools` と `runtime/lightning` / `runtime/llmops` の
`tool_mocks` 引数は FR-6 の対象外で改名しないことが要件で明示されている
（`docs/requirements/agent-name-registry-and-deterministic-model.md` FR-6 受け入れ基準）。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# 禁止語彙（FR-6）: 「偽物」を直接含意する 3 語のみ。大小無視の部分一致で照合する
# （`grep -rniE "fake|mock|dummy"` 相当）。
_FORBIDDEN_PATTERN = re.compile(r"fake|mock|dummy", re.IGNORECASE)

# テーマB（決定的応答モデル）の公開窓口シンボル 7 件。コア __all__ に含まれないことを併せて pin。
_THEME_B_SYMBOLS = {
    "DeterministicResponseModel",
    "ModelRequest",
    "text_response",
    "text_response_with_usage",
    "tool_call_response",
    "multi_tool_call_response",
    "mixed_response",
}

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
_RUNTIME_DIR = _SRC_DIR / "oai_agentspec" / "runtime"


def _runtime_subpackage_names() -> list[str]:
    """`runtime/` 直下のサブパッケージ名を列挙する（`__init__.py` を持つディレクトリのみ）。"""
    return sorted(
        entry.name
        for entry in _RUNTIME_DIR.iterdir()
        if entry.is_dir() and (entry / "__init__.py").exists()
    )


def _all_string_elements(init_path: Path) -> list[str]:
    """`__init__.py` を `ast` で解析し、トップレベル `__all__` 代入の文字列要素を取り出す。"""
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "__all__" not in targets:
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        return [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
    return []


def test_runtime直下のサブパッケージ件数は12件に固定される() -> None:
    """窓口追加時にこの pin が落ちることで列挙漏れを検知する。"""
    names = _runtime_subpackage_names()

    assert len(names) == 12, names


def test_runtime配下の各公開窓口のall要素に禁止語彙が含まれない() -> None:
    """各 `runtime/*/__init__.py` の `__all__` 文字列要素へ `fake` / `mock` / `dummy` が 0 件。"""
    violations: list[str] = []
    for name in _runtime_subpackage_names():
        init_path = _RUNTIME_DIR / name / "__init__.py"
        for symbol in _all_string_elements(init_path):
            if _FORBIDDEN_PATTERN.search(symbol):
                violations.append(f"{name}.__all__: {symbol}")

    assert violations == []


def test_コアall__に禁止語彙が含まれない() -> None:
    """コア `oai_agentspec.__all__`（宣言層シンボルのみ）へ FR-6 の禁止語彙照合をかける。"""
    import oai_agentspec

    violations = [name for name in oai_agentspec.__all__ if _FORBIDDEN_PATTERN.search(name)]

    assert violations == []


def test_コアall__にテーマBの7シンボルが含まれない() -> None:
    """テーマB（決定的応答モデル）は `runtime.deterministic` 窓口専用で、コア __all__ には無い。"""
    import oai_agentspec

    assert set(oai_agentspec.__all__).isdisjoint(_THEME_B_SYMBOLS)
