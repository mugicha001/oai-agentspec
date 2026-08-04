"""L1: `runtime.deterministic.__init__` の公開窓口契約（`__all__` 集合 pin + 再エクスポート専用）。

決定的応答モデルの公開窓口は追加依存を持たず（extra 不要）、実装本体（`_adapters/deterministic.py`）
を再エクスポートするだけの薄い窓口であることを固定する。subprocess ヘルパーは
`tests/runtime/hooks/test_init_pep562_l1.py` の `_run_in_clean_subprocess` と同型で当該ファイル内に
複製する（`tests/_helpers/` へは切り出さない）。
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# 公開窓口の `__all__` メンバ集合（決定的応答モデル + 応答ビルダ 5 種 + ModelRequest）。
_EXPECTED_ALL = {
    "DeterministicResponseModel",
    "ModelRequest",
    "text_response",
    "text_response_with_usage",
    "tool_call_response",
    "multi_tool_call_response",
    "mixed_response",
}

_SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "src"
_INIT_PATH = _SRC_DIR / "oai_agentspec" / "runtime" / "deterministic" / "__init__.py"


def _run_in_clean_subprocess(probe: str) -> str:
    """`src` を path に通したクリーンな子プロセスで probe スクリプトを実行し標準出力を返す。"""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_SRC_DIR) + (os.pathsep + existing if existing else "")
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return result.stdout.strip()


def test_all_membership_pinned() -> None:
    """`__all__` はちょうど 7 件で、決定的応答モデル関連シンボルの集合と完全一致する。"""
    from oai_agentspec.runtime import deterministic as mod

    assert set(mod.__all__) == _EXPECTED_ALL
    assert len(mod.__all__) == 7


def test_all_symbols_are_importable_and_match_adapter_source() -> None:
    """`__all__` の各シンボルは `_adapters.deterministic` の実体と `is` 一致する（再エクスポート）。

    再エクスポートであること（実装をコピーせず同一オブジェクトを参照すること）を pin する。
    """
    from oai_agentspec import _adapters
    from oai_agentspec.runtime import deterministic as mod

    adapter = _adapters.deterministic
    for name in mod.__all__:
        assert getattr(mod, name) is getattr(adapter, name)


def test_init_module_has_no_own_definitions() -> None:
    """窓口の `__init__.py` は再エクスポート専用で、関数定義・クラス定義を自前で持たない。

    `ast` でモジュールを静的解析し、トップレベルに `FunctionDef` / `AsyncFunctionDef` /
    `ClassDef` が存在しないことを固定する。実装本体を持たない薄い窓口という設計方針の pin。
    """
    tree = ast.parse(_INIT_PATH.read_text(encoding="utf-8"))
    own_definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert own_definitions == []


def test_importing_window_succeeds_in_clean_subprocess() -> None:
    """追加依存が無いため、clean subprocess でも `import` が `ImportError` にならない。"""
    probe = "import oai_agentspec.runtime.deterministic\nprint('ok')\n"
    out = _run_in_clean_subprocess(probe)
    assert out == "ok"


def test_deterministic_symbols_not_in_core_all() -> None:
    """決定的応答モデルのシンボルはコア `__all__`（宣言層のみ）に載らない。"""
    import oai_agentspec

    assert set(oai_agentspec.__all__).isdisjoint(_EXPECTED_ALL)
