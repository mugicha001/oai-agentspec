"""L1: `runtime.observability.__init__` の公開窓口契約（`__all__` 集合 pin + 再エクスポート専用）。

オブザーバビリティ連携の公開窓口が、設定 2 型（`_adapters` を経由しない plain dataclass）と
有効化関数 2 つ（`_adapters/observability.py` の実体）を**再エクスポートするだけ**の薄い窓口で
あることを固定する。加えて、窓口の import が観測系 SDK（`opentelemetry` /
`microsoft_agents_a365`）を一切ロードしないこと（有効化関数を呼ぶまで遅延する = extra 未導入
耐性・ADR 0022 Confirmation）を clean subprocess で担保する。

subprocess ヘルパーは `tests/runtime/deterministic/test_init_l1.py` の `_run_in_clean_subprocess`
と同型で当該ファイル内に複製する（`tests/_helpers/` へは切り出さない）。
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# 公開窓口の `__all__` メンバ集合（有効化エントリ 2 種 + 設定型 2 種）。
_EXPECTED_ALL = {
    "enable_agent365_tracing",
    "enable_otel_logging",
    "Agent365TracingConfig",
    "OtelLoggingConfig",
}

# 有効化関数（実体は `_adapters/observability.py`）と設定型（実体は `runtime/.../config.py`）。
_ENABLE_ENTRIES = {"enable_agent365_tracing", "enable_otel_logging"}
_CONFIG_TYPES = {"Agent365TracingConfig", "OtelLoggingConfig"}

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
_INIT_PATH = _SRC_DIR / "oai_agentspec" / "runtime" / "observability" / "__init__.py"


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
    """`__all__` はちょうど 4 件で、有効化エントリ + 設定型の集合と完全一致する。"""
    from oai_agentspec.runtime import observability as mod

    assert set(mod.__all__) == _EXPECTED_ALL
    assert len(mod.__all__) == 4


def test_enable_entries_are_reexported_from_adapter() -> None:
    """有効化関数は `_adapters.observability` の実体と `is` 一致する（再エクスポート）。"""
    from oai_agentspec._adapters import observability as adapter
    from oai_agentspec.runtime import observability as mod

    for name in sorted(_ENABLE_ENTRIES):
        assert getattr(mod, name) is getattr(adapter, name)


def test_config_types_are_reexported_from_config_module() -> None:
    """設定型は同パッケージの `config` モジュールの実体と `is` 一致する（再エクスポート）。"""
    from oai_agentspec.runtime import observability as mod
    from oai_agentspec.runtime.observability import config

    for name in sorted(_CONFIG_TYPES):
        assert getattr(mod, name) is getattr(config, name)


def test_init_module_has_no_own_definitions() -> None:
    """窓口の `__init__.py` は再エクスポート専用で、関数定義・クラス定義を自前で持たない。

    `ast` でモジュールを静的解析し、トップレベルに `FunctionDef` / `AsyncFunctionDef` /
    `ClassDef` が存在しないことを固定する（実装本体を持たない薄い窓口という設計方針の pin）。
    """
    tree = ast.parse(_INIT_PATH.read_text(encoding="utf-8"))
    own_definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    assert own_definitions == []


def test_importing_window_does_not_load_observability_sdks() -> None:
    """窓口 import では観測系 SDK をロードしない（有効化関数を呼ぶまで遅延する）。

    ADR 0022 Confirmation が名指す不変条件のうち「窓口経由の import」側を担保する
    （`import oai_agentspec` 側は `tests/test_extra_isolation.py` が担保する）。他テストの
    副作用を排除するためクリーンな子プロセスで確認する。
    """
    probe = (
        "import sys\n"
        "import oai_agentspec.runtime.observability\n"
        "loaded = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m == 'opentelemetry' or m.startswith('opentelemetry.')\n"
        "    or m == 'microsoft_agents_a365' or m.startswith('microsoft_agents_a365')\n"
        ")\n"
        "print(','.join(loaded))\n"
    )
    out = _run_in_clean_subprocess(probe)
    loaded = [m for m in out.split(",") if m]

    assert loaded == [], f"窓口 import で観測系 SDK がロードされました: {loaded}"


def test_window_symbols_are_importable_in_clean_subprocess() -> None:
    """窓口の全公開シンボルが clean subprocess で import でき、設定型の構築も通る。

    観測系 SDK 非依存の宣言部分（設定型）だけで完結する利用（宣言だけ書いて有効化は別の場所で
    行う）が壊れないことを固定する。
    """
    probe = (
        "from oai_agentspec.runtime.observability import (\n"
        "    Agent365TracingConfig,\n"
        "    OtelLoggingConfig,\n"
        "    enable_agent365_tracing,\n"
        "    enable_otel_logging,\n"
        ")\n"
        "Agent365TracingConfig(service_name='svc', service_namespace='ns')\n"
        "OtelLoggingConfig()\n"
        "print(callable(enable_agent365_tracing) and callable(enable_otel_logging))\n"
    )

    assert _run_in_clean_subprocess(probe) == "True"


def test_observability_symbols_not_in_core_all() -> None:
    """オブザーバビリティのシンボルはコア `__all__`（宣言層のみ）に載らない（FR-7）。"""
    import oai_agentspec

    assert set(oai_agentspec.__all__).isdisjoint(_EXPECTED_ALL)
