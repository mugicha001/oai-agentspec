"""L1: `runtime.resilience.__init__` の公開窓口契約（直 import + PEP 562 遅延）。

intent 窓口（`tests/runtime/intent/test_init_pep562_l1.py`）を直接の踏襲元とする。
resilience 固有の差分として、宣言型（`ModelRetryPolicy` / `RunBudgetPolicy` /
`FailsafeHandler` / `FailsafePolicy` / `FailsafeResult`）・sentinel `RUNNING_AGENT` と
関数 `failsafe_call` は外部依存ゼロのため module import 時点で直 import 済みであり、
`build_*` ヘルパと SDK 生型 10 種のみ `__getattr__` で `_adapters.resilience` 経由の
遅延取得になる。
lib 独自例外 `RunBudgetExceeded` の正規経路は `oai_agentspec.exceptions`
（本窓口からは撤去済み）。

遅延の対象は `_adapters.resilience` モジュールであって `agents` ではない。`agents` は
コア依存で `oai_agentspec/__init__.py` -> `_adapters/__init__.py` の連鎖によりこの窓口の
import より前にロード済みになるため、本ファイルの probe も `_adapters.resilience` の
非ロードのみを検査する（`tests/runtime/hooks/` の probe と同型）。
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


# lib 独自 9 種。うち直 import は宣言型 5 + 関数 1 + sentinel 1、遅延は build_* 2。
_DIRECT_SYMBOLS = {
    "ModelRetryPolicy",
    "RunBudgetPolicy",
    "FailsafeHandler",
    "FailsafePolicy",
    "FailsafeResult",
    "failsafe_call",
    "RUNNING_AGENT",
}
_LAZY_BUILD_SYMBOLS = {
    "build_model_retry",
    "build_run_budget_hooks",
}
# SDK 生型 10 種（すべて `_adapters.resilience` 経由の遅延取得）。
_LAZY_SDK_SYMBOLS = {
    "ModelRetrySettings",
    "ModelRetryBackoffSettings",
    "retry_policies",
    "RetryDecision",
    "RetryPolicyContext",
    "ModelRetryNormalizedError",
    "RunErrorHandlers",
    "RunErrorHandlerResult",
    "RunErrorHandlerInput",
    "RunErrorData",
}
_LAZY_SYMBOLS = _LAZY_BUILD_SYMBOLS | _LAZY_SDK_SYMBOLS
_EXPECTED_ALL = _DIRECT_SYMBOLS | _LAZY_SYMBOLS


def test_all_membership_pinned() -> None:
    """`__all__` は 19 件で設計仕様通りのメンバ集合と一致する。"""
    from oai_agentspec.runtime import resilience as mod

    assert set(mod.__all__) == _EXPECTED_ALL
    assert len(mod.__all__) == 19


def test_declaration_symbols_are_directly_imported() -> None:
    """直 import 対象（宣言型 + 関数 + sentinel）は module import 時点で `__dict__` に載る。"""
    from oai_agentspec.runtime import resilience as mod

    for name in _DIRECT_SYMBOLS:
        assert name in mod.__dict__, f"'{name}' は直 import されているべき"


def test_lazy_build_symbol_resolves_and_caches() -> None:
    """`build_model_retry` は遅延取得され、再取得で同一オブジェクトを返す（キャッシュ）。"""
    from oai_agentspec.runtime import resilience as mod

    mod.__dict__.pop("build_model_retry", None)
    assert "build_model_retry" not in mod.__dict__
    first = mod.build_model_retry
    assert first is not None
    assert "build_model_retry" in mod.__dict__
    second = mod.build_model_retry
    assert first is second


def test_lazy_build_symbol_matches_adapter_source() -> None:
    """遅延取得の `build_run_budget_hooks` は `_adapters.resilience` の実体と同一。"""
    from oai_agentspec._adapters import resilience as adapter
    from oai_agentspec.runtime import resilience as mod

    mod.__dict__.pop("build_run_budget_hooks", None)
    resolved = mod.build_run_budget_hooks
    assert resolved is adapter.build_run_budget_hooks


def test_lazy_sdk_raw_types_resolve_and_cache() -> None:
    """SDK 生型 10 種はいずれも遅延取得でき、再取得で同一オブジェクトを返す。"""
    from oai_agentspec.runtime import resilience as mod

    for name in _LAZY_SDK_SYMBOLS:
        mod.__dict__.pop(name, None)
        first = getattr(mod, name)
        assert first is not None, f"'{name}' が遅延取得できない"
        assert name in mod.__dict__
        second = getattr(mod, name)
        assert first is second, f"'{name}' がキャッシュされていない"


def test_all_symbols_are_resolvable_via_getattr() -> None:
    """`__all__` の全 19 シンボルが `getattr` で解決可能（漏れがない）。"""
    from oai_agentspec.runtime import resilience as mod

    for name in mod.__all__:
        mod.__dict__.pop(name, None)
        value = getattr(mod, name)
        assert value is not None, f"'{name}' が解決できない"


def test_getattr_unknown_attribute_raises() -> None:
    """未定義属性は AttributeError を送出する。"""
    from oai_agentspec.runtime import resilience as mod

    with pytest.raises(AttributeError):
        mod.__getattr__("nonexistent")


def test_dir_includes_all_symbols_even_before_access() -> None:
    """`dir()` は未 import 状態でも `__all__` の全 19 シンボルを含む。"""
    from oai_agentspec.runtime import resilience as mod

    listing = set(mod.__dir__())
    assert _EXPECTED_ALL.issubset(listing)


def test_run_budget_exceeded_is_removed_from_window() -> None:
    """`RunBudgetExceeded` は窓口から撤去済み。正規経路は `oai_agentspec.exceptions`。"""
    from oai_agentspec.runtime import resilience as mod

    with pytest.raises(AttributeError):
        mod.__getattr__("RunBudgetExceeded")


def test_importing_window_does_not_load_adapter_module() -> None:
    """`import oai_agentspec.runtime.resilience` 時点で `_adapters.resilience` を発火させない。

    実体 `_adapters/resilience.py` は SDK 生型と build 関数を定義するが、窓口の `__getattr__`
    経由でしか import されないため、窓口 module import だけでは実装 module 自体が読み込まれない
    ことを固定する。`agents` は core 依存として本体 import で既に載っているため対象外。
    `_adapters/__init__.py` がトップレベルで `.resilience` を import する退行（コア import 連鎖で
    常時ロードされ遅延が空振りする）もこの probe が検知する。
    """
    probe = (
        "import sys\n"
        "import oai_agentspec.runtime.resilience\n"
        "loaded = 'oai_agentspec._adapters.resilience' in sys.modules\n"
        "print('loaded' if loaded else 'not-loaded')\n"
    )
    out = _run_in_clean_subprocess(probe)
    assert out == "not-loaded", f"窓口 import で _adapters.resilience が発火しました: {out}"


def test_dir_call_does_not_load_adapter_module() -> None:
    """`dir(mod)` を呼んでも `_adapters.resilience` を発火させない（名前集合のみを見る）。

    `__dir__` が `__all__` の名前集合と `globals()` のみを参照し実体に触れないことを固定する。
    遅延シンボル未アクセスの状態で `dir()` を呼ぶ経路を押さえることで、シンボル追加時に
    `__dir__` が実体解決（`__getattr__` 相当の呼び出し）へ退化する回帰を検知する。
    """
    probe = (
        "import sys\n"
        "import oai_agentspec.runtime.resilience as mod\n"
        "names = set(dir(mod))\n"
        "assert 'build_model_retry' in names and 'ModelRetrySettings' in names, sorted(names)\n"
        "loaded = 'oai_agentspec._adapters.resilience' in sys.modules\n"
        "print('loaded' if loaded else 'not-loaded')\n"
    )
    out = _run_in_clean_subprocess(probe)
    assert out == "not-loaded", f"dir() 呼び出しで _adapters.resilience が発火しました: {out}"


async def test_failsafe_call_is_usable_via_window_import() -> None:
    """`failsafe_call` は窓口経由の import でも呼び出せる（直 import シンボルの疎通確認）。"""
    from oai_agentspec.runtime.resilience import FailsafePolicy, failsafe_call

    async def _thunk() -> str:
        return "ok"

    policy = FailsafePolicy()
    result = await failsafe_call(policy, _thunk)

    assert result == "ok"


async def test_failsafe_handler_is_usable_via_window_import() -> None:
    """`FailsafeHandler` は窓口経由の import でも `handlers` の値位置で機能する。"""
    from oai_agentspec.runtime.resilience import FailsafeHandler, FailsafePolicy, failsafe_call

    class _WindowError(Exception):
        """本 pin 専用の例外型。"""

    async def _thunk() -> str:
        raise _WindowError("boom")

    policy = FailsafePolicy(handlers={_WindowError: FailsafeHandler(fallback="landed")})
    result = await failsafe_call(policy, _thunk)

    assert result.final_output == "landed"


async def test_running_agent_is_usable_via_window_import() -> None:
    """`RUNNING_AGENT` は窓口経由の import でも `fallback_last_agent` の解決に使える。"""
    from oai_agentspec.runtime.resilience import RUNNING_AGENT, FailsafePolicy, failsafe_call

    class _WindowError(Exception):
        """本 pin 専用の例外型。"""

    class _FakeRunData:
        """`exc.run_data.last_agent` を模す最小 fake（`agents` は import しない）。"""

        def __init__(self, last_agent: object) -> None:
            self.last_agent = last_agent

    agent = object()

    async def _thunk() -> str:
        exc = _WindowError("boom")
        exc.run_data = _FakeRunData(agent)  # type: ignore[attr-defined]
        raise exc

    policy = FailsafePolicy(handlers={_WindowError: "landed"}, fallback_last_agent=RUNNING_AGENT)
    result = await failsafe_call(policy, _thunk)

    assert result.final_output == "landed"
    assert result.last_agent is agent
