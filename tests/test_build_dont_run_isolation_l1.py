"""L1: build-don't-run 逸脱の物理隔離を AST 走査で機械検証する（ADR 0039 Confirmation）。

本検査は素直な `.fit(` 追加に対する回帰網であり、意図的な迂回（`getattr(target, "fit")(...)` /
束縛の再代入 / `functools.partial` / `operator.methodcaller`）は検知しない。迂回の検出は
コードレビューが担う。

対象が `src/oai_agentspec/` 全体というリポジトリ横断の不変条件のため、`tests/runtime/intent/`
ではなく `tests/` 直下に置く（既存慣行: `tests/test_extra_isolation.py` /
`tests/test_public_all_membership_l1.py` / `tests/test_integrity.py`）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "oai_agentspec"

# NFR-1 前半: lib 本体が import してはならない ML フレームワークのトップレベル名。
_FORBIDDEN_IMPORT_ROOTS = ("sklearn", "numpy", "scipy")


def _fit_call_sites() -> set[tuple[str, str]]:
    """`src/oai_agentspec/` 配下で `<obj>.fit(...)` を呼ぶ (相対パス, 関数名) を集める。

    ast で関数単位まで絞り込む。ネストした関数は最も内側の関数名を採用する。任意の
    対象に対する `.fit(` 呼び出しを検知する必要があるため、呼び出し元オブジェクトの
    型・名前は条件にしない（`node.func.attr == "fit"` のみで判定する）。

    Returns:
        (`src/oai_agentspec/` からの相対パス, 囲む関数名) の集合。
    """
    sites: set[tuple[str, str]] = set()

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stack: list[str] = []

        def visit(node: ast.AST, stack: list[str] = stack, path: Path = path) -> None:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                stack.append(node.name)
                for child in ast.iter_child_nodes(node):
                    visit(child)
                stack.pop()
                return
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "fit"
            ):
                sites.add(
                    (path.relative_to(_SRC_ROOT).as_posix(), stack[-1] if stack else "<module>")
                )
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(tree)

    return sites


def _forbidden_import_sites() -> set[tuple[str, str]]:
    """`src/oai_agentspec/` 配下で `sklearn` / `numpy` / `scipy` を import する箇所を集める。

    `ImportFrom` は `node.level == 0`（絶対 import）のみを対象にする（相対 import は
    ローカルモジュールであり ML フレームワークではないため）。

    Returns:
        (相対パス, import されたモジュール名) の集合。
    """
    sites: set[tuple[str, str]] = set()

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _FORBIDDEN_IMPORT_ROOTS:
                        sites.add((path.relative_to(_SRC_ROOT).as_posix(), alias.name))
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module.split(".")[0] in _FORBIDDEN_IMPORT_ROOTS:
                    sites.add((path.relative_to(_SRC_ROOT).as_posix(), node.module))

    return sites


def test_fit_駆動箇所の集合が_fit_once_ただ一つと一致する() -> None:
    """lib 内の学習駆動が `_fit_once` の 1 箇所へ閉じている（ADR 0004 / ADR 0039）。

    件数カウントではなく集合一致にする。`rich.panel.Panel.fit(...)` のような無関係な
    実在 API が将来入った際に、その変更が本ガードを落として原因が結びつかなくなるのを
    防ぐため（`runtime/cli` は `rich.panel.Panel` を使うが、現時点で `Panel.fit` の
    呼び出しは無い）。
    """
    sites = _fit_call_sites()
    expected = {("runtime/intent/_ml_training.py", "_fit_once")}
    assert sites == expected, f"検出した .fit( 呼び出し箇所: {sorted(sites)}"


def test_lib_本体は_sklearn_numpy_scipy_を_import_しない() -> None:
    """lib（`src/oai_agentspec/`）は ML フレームワークを import しない（NFR-1 前半）。

    開発依存に scikit-learn（と推移依存の numpy / scipy）が入ったことで、テスト緑の
    まま lib 本体へ import が混入しうるため機械検証する。
    """
    sites = _forbidden_import_sites()
    assert sites == set(), f"検出した禁止 import 箇所: {sorted(sites)}"
