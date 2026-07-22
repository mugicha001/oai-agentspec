"""L1: `runtime.hooks.__init__` の公開窓口契約（PEP 562 単一シンボル遅延）。

`chain_hooks` を `_adapters.hooks` から遅延取得する薄い窓口。`import oai_agentspec.runtime.hooks`
時点では `agents.lifecycle` を発火させず、属性アクセス時に初めて `_adapters.hooks` 経由で
実体を取得する（`governance` 窓口と同型）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


_SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "src"


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
    """`__all__` は `chain_hooks` の単一シンボル。"""
    from oai_agentspec.runtime import hooks as mod

    assert set(mod.__all__) == {"chain_hooks"}
    assert len(mod.__all__) == 1


def test_chain_hooks_resolves_and_caches() -> None:
    """`chain_hooks` は `__getattr__` 経由で遅延解決でき、キャッシュされる。"""
    from oai_agentspec.runtime import hooks as mod

    mod.__dict__.pop("chain_hooks", None)
    assert "chain_hooks" not in mod.__dict__
    first = mod.chain_hooks
    assert first is not None
    assert "chain_hooks" in mod.__dict__
    second = mod.chain_hooks
    assert first is second


def test_chain_hooks_matches_adapter_source() -> None:
    """遅延取得した `chain_hooks` は `_adapters.hooks` の実体と `is` 一致する。"""
    from oai_agentspec._adapters import hooks as adapter
    from oai_agentspec.runtime import hooks as mod

    mod.__dict__.pop("chain_hooks", None)
    resolved = mod.chain_hooks
    assert resolved is adapter.chain_hooks


def test_getattr_unknown_attribute_raises() -> None:
    """未定義属性は AttributeError を送出する。"""
    from oai_agentspec.runtime import hooks as mod

    with pytest.raises(AttributeError):
        mod.__getattr__("nonexistent")


def test_dir_includes_all_symbols_even_before_access() -> None:
    """`dir()` は未 import 状態でも `chain_hooks` を含む。"""
    from oai_agentspec.runtime import hooks as mod

    assert "chain_hooks" in set(mod.__dir__())


def test_importing_window_does_not_load_adapter_module() -> None:
    """`import oai_agentspec.runtime.hooks` 時点で `_adapters.hooks` を発火させない（PEP 562）。

    実体 `_adapters/hooks.py` は `agents.lifecycle` を import するが、窓口の `__getattr__` 経由
    でしか import されないため、窓口 module import だけでは実装 module 自体が読み込まれない
    ことを固定する。`agents` は core 依存として本体 import で既に載っているため対象外。
    """
    probe = (
        "import sys\n"
        "import oai_agentspec.runtime.hooks\n"
        "loaded = 'oai_agentspec._adapters.hooks' in sys.modules\n"
        "print('loaded' if loaded else 'not-loaded')\n"
    )
    out = _run_in_clean_subprocess(probe)
    assert out == "not-loaded", f"窓口 import で _adapters.hooks が発火しました: {out}"
