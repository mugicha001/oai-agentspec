"""extra 隔離（NFR-1）: 本体 import が各 extra（serve / cli / llmops / observability）を読まない。

`import oai_agentspec` は会話コア・registry・workflow 等の公開 API を提供するが、serve
（fastapi / uvicorn）・cli（httpx / websockets）の入口モジュール、および llmops 採点エンジン
（deepeval）・観測クライアント（langfuse）はその時点で import してはならない（各サブコマンド /
app factory / 評価エントリで遅延 import する前提・NFR-1）。観測系（opentelemetry /
microsoft_agents_a365）は、lib が新たに強制ロードしないこと、および観測系 SDK 本体
（`opentelemetry.sdk` / `opentelemetry.exporter` / `microsoft_agents_a365`）が読まれないことを
別の判定として検査する（範囲は ADR 0047 Decision を正とする）。

検証対象は oai_agentspec 自身が制御する境界に限定する:
- 本体 import 後に `oai_agentspec.runtime.serve.*` / `oai_agentspec.runtime.cli.*` /
  `oai_agentspec.runtime.llmops` 入口モジュールが sys.modules に載っていないこと。
- serve / cli 入口のみが import する extra（fastapi / websockets）・llmops の重い依存
  （deepeval / langfuse）が載っていないこと（トップレベル名の完全一致による絶対禁止）。
- `import agents` 直後の baseline から、lib の本体 import で観測系ルート（opentelemetry /
  microsoft_agents_a365）が新たに増えていないこと（判定 A）、および観測系 SDK 本体が
  baseline の有無を問わず存在しないこと（判定 B）。

注: httpx / uvicorn は SDK（agents / openai）が transitive に import するため本体 import でも
sys.modules に現れうる。これらは oai_agentspec の制御外（SDK 依存）であり、本隔離不変条件の
対象外とする。同様に、`import agents` の時点で SDK 経由で載る観測系モジュール（mcp 2 系での
opentelemetry-api）は判定 A の対象外とする（ADR 0047 Decision 1）。汚染のないクリーンな
subprocess で本体だけを import して検証する。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# serve / cli / llmops だけが import する extra（本体 import で現れてはならない。トップレベル名の
# 完全一致による絶対禁止）。httpx / uvicorn は SDK 経由で transitive に載るため対象外（モジュール
# docstring 参照）。deepeval / langfuse は llmops の重い依存で、評価エントリ / _adapters の関数内
# 遅延 import に閉じる前提（本体 import で載ってはならない）。観測系（opentelemetry /
# microsoft_agents_a365）は判定方式が異なるため別定数（_OBSERVABILITY_ROOTS /
# _OBSERVABILITY_SDK_BODY）に分離する。
_FORBIDDEN_EXTRAS = (
    "fastapi",
    "websockets",
    "deepeval",
    "langfuse",
)

# 観測系の判定 A（lib による追加の禁止）対象ルート。`import agents` 直後の baseline から新たに
# 増えたモジュールのうち、これらに一致するものを違反とする（名前一致は
# `m == p or m.startswith(p + ".")`。ADR 0047 Decision 1）。
_OBSERVABILITY_ROOTS = (
    "opentelemetry",
    "microsoft_agents_a365",
)

# 観測系の判定 B（SDK 本体の絶対禁止）対象の名前空間。baseline に含まれていても違反とする
# （範囲は ADR 0047 Decision 2 を正とする）。
_OBSERVABILITY_SDK_BODY = (
    "opentelemetry.sdk",
    "opentelemetry.exporter",
    "microsoft_agents_a365",
)

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _import_in_clean_subprocess(probe: str) -> str:
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


def test_importing_package_does_not_load_serve_cli_llmops_entrypoints() -> None:
    """`import oai_agentspec` で serve / cli / llmops 入口を import しない（NFR-1）。"""
    probe = (
        "import sys\n"
        "import oai_agentspec\n"
        "loaded = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m.startswith('oai_agentspec.runtime.serve')\n"
        "    or m.startswith('oai_agentspec.runtime.cli')\n"
        "    or m == 'oai_agentspec.runtime.llmops'\n"
        "    or m.startswith('oai_agentspec.runtime.llmops.')\n"
        ")\n"
        "print(','.join(loaded))\n"
    )
    out = _import_in_clean_subprocess(probe)
    loaded = [m for m in out.split(",") if m]
    assert loaded == [], f"本体 import で serve / cli / llmops 入口がロードされました: {loaded}"


def test_importing_package_does_not_force_load_extra_deps() -> None:
    """本体 import で extra 依存・観測系モジュール・観測系 SDK 本体を強制ロードしない。

    extra 依存は `_FORBIDDEN_EXTRAS` の完全一致で検査する。観測系は判定 A（lib による追加の
    禁止）と判定 B（SDK 本体の絶対禁止）の 2 つで検査する（ADR 0047 Decision 1）。
    """
    forbidden = list(_FORBIDDEN_EXTRAS)
    roots = list(_OBSERVABILITY_ROOTS)
    sdk_body = list(_OBSERVABILITY_SDK_BODY)
    probe = (
        "import sys\n"
        "import agents\n"
        "sdk_baseline = set(sys.modules)\n"
        "import oai_agentspec\n"
        f"forbidden = {forbidden!r}\n"
        "forbidden_loaded = [m for m in forbidden if m in sys.modules]\n"
        f"roots = {roots!r}\n"
        f"sdk_body = {sdk_body!r}\n"
        "def _match(m, names):\n"
        "    return any(m == p or m.startswith(p + '.') for p in names)\n"
        "added_by_lib = [m for m in set(sys.modules) - sdk_baseline if _match(m, roots)]\n"
        "sdk_body_loaded = [m for m in sys.modules if _match(m, sdk_body)]\n"
        "violations = sorted(set(forbidden_loaded) | set(added_by_lib) | set(sdk_body_loaded))\n"
        "print(','.join(violations))\n"
    )
    out = _import_in_clean_subprocess(probe)
    loaded = [m for m in out.split(",") if m]
    assert loaded == [], (
        f"本体 import で extra（{forbidden}）または観測系モジュールがロードされました: {loaded}"
    )


def test_importing_package_does_not_chain_import_realtime() -> None:
    """`import oai_agentspec` で realtime を連鎖 import しない（遅延 import 境界）。"""
    probe = (
        "import sys\n"
        "import oai_agentspec\n"
        "loaded = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m == 'oai_agentspec.realtime' or m.startswith('oai_agentspec.realtime.')\n"
        ")\n"
        "print(','.join(loaded))\n"
    )
    out = _import_in_clean_subprocess(probe)
    loaded = [m for m in out.split(",") if m]
    assert loaded == [], f"本体 import で realtime が連鎖ロードされました: {loaded}"


def test_core_all_does_not_contain_realtime_symbols() -> None:
    """コア `__all__` に Realtime シンボルを載せない（専用窓口経由のみで提供する）。"""
    import oai_agentspec

    realtime_symbols = {"RealtimeAgentSpec", "RealtimeHandoffConfig", "RealtimeAgentRegistry"}
    leaked = realtime_symbols & set(oai_agentspec.__all__)
    assert leaked == set(), f"コア __all__ に Realtime シンボルが混入: {sorted(leaked)}"


def test_importing_realtime_window_does_not_load_realtime_sdk() -> None:
    """`import oai_agentspec.realtime` で `agents.realtime` / websockets を強制ロードしない。

    Realtime の SDK 結合は build 時（registry のデフォルトビルダー遅延生成）まで遅延する。
    親パッケージ import に伴う `agents` コアのロードは SDK 依存として対象外。
    """
    probe = (
        "import sys\n"
        "import oai_agentspec.realtime\n"
        "loaded = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m == 'websockets' or m == 'agents.realtime'\n"
        "    or m.startswith('agents.realtime.')\n"
        ")\n"
        "print(','.join(loaded))\n"
    )
    out = _import_in_clean_subprocess(probe)
    loaded = [m for m in out.split(",") if m]
    assert loaded == [], f"realtime 窓口 import で SDK がロードされました: {loaded}"


def test_importing_exceptions_window_does_not_load_lazy_extras() -> None:
    """`import oai_agentspec.exceptions` で lightning / cli 親パッケージを強制ロードしない。

    exceptions 窓口は直 import 7 種（コア + resilience / conversation）と PEP 562 遅延 2 種
    （lightning の `OptimizeError` / cli の `ConversationClientError`）で構成される。窓口
    module import だけでは遅延分の親パッケージ（`oai_agentspec.runtime.lightning` /
    `oai_agentspec.runtime.cli`）が `sys.modules` に載らないことを固定し、PEP 562 遅延の
    実効性を担保する（Issue #31 D1 の頑健性理由）。`agents` は core 依存として本体 import で
    既に載っているため対象外（`_FORBIDDEN_EXTRAS` に含めていない既存不変と整合）。
    """
    probe = (
        "import sys\n"
        "import oai_agentspec.exceptions\n"
        "loaded = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m == 'oai_agentspec.runtime.lightning'\n"
        "    or m.startswith('oai_agentspec.runtime.lightning.')\n"
        "    or m == 'oai_agentspec.runtime.cli'\n"
        "    or m.startswith('oai_agentspec.runtime.cli.')\n"
        ")\n"
        "print(','.join(loaded))\n"
    )
    out = _import_in_clean_subprocess(probe)
    loaded = [m for m in out.split(",") if m]
    assert loaded == [], f"exceptions 窓口 import で lightning / cli がロードされました: {loaded}"


def test_l1_fake_builder_helper_is_agents_free() -> None:
    """L1 用テストダブル（fake_realtime_builder）は `agents` を一切ロードしない。

    L1 テストの「agents 非依存」を import 境界で機械的に固定する（SDK 依存の
    FakeRealtimeModel は fake_realtime_model.py 側に分離されている前提）。
    """
    tests_dir = Path(__file__).resolve().parent
    probe = (
        "import sys\n"
        f"sys.path.insert(0, {str(tests_dir)!r})\n"
        "import _helpers.fake_realtime_builder\n"
        "loaded = sorted(m for m in sys.modules if m == 'agents' or m.startswith('agents.'))\n"
        "print(','.join(loaded))\n"
    )
    out = _import_in_clean_subprocess(probe)
    loaded = [m for m in out.split(",") if m]
    assert loaded == [], f"L1 テストダブルが agents をロードしました: {loaded}"
