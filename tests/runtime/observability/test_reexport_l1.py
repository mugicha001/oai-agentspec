"""L1: `runtime.observability.__init__` の公開窓口契約（`__all__` 集合 pin + 再エクスポート専用）。

オブザーバビリティ連携の公開窓口が、設定 2 型（`_adapters` を経由しない plain dataclass）と
有効化関数 2 つ（`_adapters/observability.py` の実体）を**再エクスポートするだけ**の薄い窓口で
あることを固定する。加えて、窓口の import が観測系モジュール（`opentelemetry` /
`microsoft_agents_a365`）を新たにロードしないこと（判定 A）、および観測系 SDK 本体をロードしない
こと（判定 B。有効化関数を呼ぶまで遅延する = extra 未導入耐性）を clean subprocess で担保する
（範囲は ADR 0047 Decision を正とする）。

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

# mcp 2 系で SDK が `import agents` の時点に読み込む opentelemetry-api を再現する preamble と、
# 判定 B の名前空間ごとにその名前空間にだけ当たる SDK 本体のモジュール。
# tests/test_extra_isolation.py の同名定数に同期する。
_OBSERVABILITY_API_PRELOAD = (
    "import opentelemetry.context\n"
    "import opentelemetry.propagate\n"
    "import opentelemetry.trace\n"
    "print('preloaded')\n"
)
_OBSERVABILITY_SDK_PRELOADS = (
    "opentelemetry.sdk.resources",
    "opentelemetry.exporter.otlp.proto.http",
)

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


def _window_probe(preamble: str = "") -> str:
    """窓口 import の観測系の違反を最終行へ出力する probe を組み立てる。

    `preamble` は `import agents` より前に実行する文で、SDK が観測系モジュールを先に読み込んだ
    状況（baseline に載っている状態）を再現するために使う。
    """
    return (
        "import sys\n"
        f"{preamble}"
        "import agents\n"
        "sdk_baseline = set(sys.modules)\n"
        "import oai_agentspec.runtime.observability\n"
        "def _match(m, names):\n"
        "    return any(m == p or m.startswith(p + '.') for p in names)\n"
        # roots は ADR 0047 Decision 1、sdk_body は Decision 2 を正とし、
        # tests/test_extra_isolation.py の _OBSERVABILITY_ROOTS /
        # _OBSERVABILITY_SDK_BODY に同期する。
        "roots = ['opentelemetry', 'microsoft_agents_a365']\n"
        "sdk_body = ['opentelemetry.sdk', 'opentelemetry.exporter', 'microsoft_agents_a365']\n"
        "added_by_lib = [m for m in set(sys.modules) - sdk_baseline if _match(m, roots)]\n"
        "sdk_body_loaded = [m for m in sys.modules if _match(m, sdk_body)]\n"
        "violations = sorted(set(added_by_lib) | set(sdk_body_loaded))\n"
        "print(','.join(violations))\n"
    )


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
    """窓口 import で観測系モジュールを新たに増やさず SDK 本体もロードしない。

    観測系の import は有効化関数を呼ぶまで遅延する。不変条件のうち「窓口経由の import」側を
    担保する（`import oai_agentspec` 側は `tests/test_extra_isolation.py` が担保する）。
    判定 A（`import agents` 直後の baseline から、窓口 import で観測系ルートが新たに増えて
    いないこと）と判定 B（観測系 SDK 本体が baseline の有無を問わず存在しないこと）の 2 つで
    検査する（ADR 0047 Decision 1）。他テストの副作用を排除するためクリーンな子プロセスで
    確認する。
    """
    out = _run_in_clean_subprocess(_window_probe())
    loaded = [m for m in out.split(",") if m]

    assert loaded == [], f"窓口 import で観測系モジュールがロードされました: {loaded}"


def test_observability_api_loaded_before_agents_is_not_attributed_to_window() -> None:
    """`import agents` より前に載った opentelemetry-api は窓口 import の違反にしない。

    mcp 2 系で SDK が `import agents` の時点に opentelemetry-api を読み込む状況を preamble で
    再現し、窓口 import が新たに読み込んだものだけを違反とする差分判定（ADR 0047 Decision 1）を
    mcp の版に依存せず固定する。preamble が実行されたことは `preloaded` の出力で確かめる。
    """
    pytest.importorskip("opentelemetry.trace")
    out = _run_in_clean_subprocess(_window_probe(_OBSERVABILITY_API_PRELOAD)).splitlines()

    assert out[0] == "preloaded"
    loaded = [m for line in out[1:] for m in line.split(",") if m]
    assert loaded == [], f"SDK 経由で先に載った観測系モジュールが違反になりました: {loaded}"


@pytest.mark.parametrize("sdk_module", _OBSERVABILITY_SDK_PRELOADS)
def test_sdk_body_loaded_before_agents_is_a_window_violation(sdk_module: str) -> None:
    """観測系 SDK 本体は `import agents` より前に載っていても窓口の違反になる（判定 B）。

    判定 B だけが検出できる状況（SDK 本体が baseline に含まれる状態）を preamble で作り、
    判定 B が効いていることを固定する（ADR 0047 Decision 1 / 2）。
    """
    pytest.importorskip(sdk_module)
    out = _run_in_clean_subprocess(_window_probe(f"import {sdk_module}\n"))
    loaded = [m for m in out.split(",") if m]

    assert sdk_module in loaded


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
