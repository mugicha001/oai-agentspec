"""L1: `runtime.hooks.__init__` の公開窓口契約（PEP 562 の 2 シンボル遅延）。

run 単位 `chain_hooks` と agent 単位 `chain_agent_hooks` を `_adapters.hooks` から遅延取得する
薄い窓口。`import oai_agentspec.runtime.hooks` 時点では合成クラス定義を含む `_adapters.hooks` を
ロードせず、属性アクセス時に初めて実体を取得する（`governance` 窓口と同型）。

遅延の対象は `_adapters.hooks` モジュールであって `agents` ではない。`agents` / `agents.lifecycle`
はコア依存で `oai_agentspec/__init__.py` -> `_adapters/__init__.py` の連鎖によりこの窓口の import
より前にロード済みになるため、本ファイルの probe も `_adapters.hooks` の非ロードのみを検査する。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# 公開窓口の `__all__` メンバ集合（run 単位 / agent 単位の 2 シンボル）。
_EXPECTED_ALL = {"chain_agent_hooks", "chain_hooks"}


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
    """`__all__` は run 単位 `chain_hooks` と agent 単位 `chain_agent_hooks` の 2 シンボル。"""
    from oai_agentspec.runtime import hooks as mod

    assert set(mod.__all__) == _EXPECTED_ALL
    assert len(mod.__all__) == 2


def test_hooks_symbols_are_not_in_core_all() -> None:
    """hooks 合成ヘルパーはコア `__all__` に載らない（実行寄り層の独立窓口として分離する）。

    コア `__all__` は宣言層シンボルのみとする方針の pin。混入すると `import oai_agentspec` の
    公開面が実行寄り層へ広がり、独立窓口という表現が意味を失う。
    """
    import oai_agentspec

    assert set(oai_agentspec.__all__).isdisjoint(_EXPECTED_ALL)


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


def test_chain_agent_hooks_resolves_and_caches() -> None:
    """`chain_agent_hooks` は `__getattr__` 経由で遅延解決でき、キャッシュされる。"""
    from oai_agentspec.runtime import hooks as mod

    mod.__dict__.pop("chain_agent_hooks", None)
    assert "chain_agent_hooks" not in mod.__dict__
    first = mod.chain_agent_hooks
    assert first is not None
    assert "chain_agent_hooks" in mod.__dict__
    second = mod.chain_agent_hooks
    assert first is second


def test_chain_agent_hooks_matches_adapter_source() -> None:
    """遅延取得した `chain_agent_hooks` は `_adapters.hooks` の実体と `is` 一致する。"""
    from oai_agentspec._adapters import hooks as adapter
    from oai_agentspec.runtime import hooks as mod

    mod.__dict__.pop("chain_agent_hooks", None)
    resolved = mod.chain_agent_hooks
    assert resolved is adapter.chain_agent_hooks


def test_getattr_unknown_attribute_raises() -> None:
    """未定義属性は AttributeError を送出する。"""
    from oai_agentspec.runtime import hooks as mod

    with pytest.raises(AttributeError):
        mod.__getattr__("nonexistent")


def test_dir_includes_all_symbols_even_before_access() -> None:
    """`dir()` は未 import 状態でも `chain_hooks` と `chain_agent_hooks` を含む。"""
    from oai_agentspec.runtime import hooks as mod

    assert "chain_hooks" in set(mod.__dir__())
    assert "chain_agent_hooks" in set(mod.__dir__())


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


def test_dir_call_does_not_load_adapter_module() -> None:
    """`dir(mod)` を呼んでも `_adapters.hooks` を発火させない（`__dir__` は名前集合のみを見る）。

    `__dir__` が `__all__` の名前集合と `globals()` のみを参照し実体に触れないことを固定する。
    どちらのシンボルも未アクセスの状態で `dir()` を呼ぶ経路を押さえることで、シンボル追加時に
    `__dir__` が実体解決（`__getattr__` 相当の呼び出し）へ退化する回帰を検知する。
    """
    probe = (
        "import sys\n"
        "import oai_agentspec.runtime.hooks as mod\n"
        "names = set(dir(mod))\n"
        "assert 'chain_hooks' in names and 'chain_agent_hooks' in names, sorted(names)\n"
        "loaded = 'oai_agentspec._adapters.hooks' in sys.modules\n"
        "print('loaded' if loaded else 'not-loaded')\n"
    )
    out = _run_in_clean_subprocess(probe)
    assert out == "not-loaded", f"dir() 呼び出しで _adapters.hooks が発火しました: {out}"
