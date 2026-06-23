"""extra 隔離（NFR-1）: 本体 import が serve / cli / llmops extra を強制ロードしないことを担保する。

`import oai_agentspec` は会話コア・registry・workflow 等の公開 API を提供するが、serve
（fastapi / uvicorn）・cli（httpx / websockets）の入口モジュール、および llmops 採点エンジン
（deepeval）・観測クライアント（langfuse）はその時点で import してはならない（各サブコマンド /
app factory / 評価エントリで遅延 import する前提・NFR-1）。

検証対象は oai_agentspec 自身が制御する境界に限定する:
- 本体 import 後に `oai_agentspec.runtime.serve.*` / `oai_agentspec.runtime.cli.*` /
  `oai_agentspec.runtime.llmops` 入口モジュールが sys.modules に載っていないこと。
- serve / cli 入口のみが import する extra（fastapi / websockets）・llmops の重い依存
  （deepeval / langfuse）が載っていないこと。

注: httpx / uvicorn は SDK（agents / openai）が transitive に import するため本体 import でも
sys.modules に現れうる。これらは oai_agentspec の制御外（SDK 依存）であり、本隔離不変条件の
対象外とする。汚染のないクリーンな subprocess で本体だけを import して検証する。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# serve / cli / llmops 入口のみが import する extra（本体 import で現れてはならない）。
# httpx / uvicorn は SDK 経由で transitive に載るため対象外（モジュール docstring 参照）。
# deepeval / langfuse は llmops の重い依存で、評価エントリ / _adapters の関数内遅延 import に閉じる
# 前提（本体 import で載ってはならない）。
_FORBIDDEN_EXTRAS = ("fastapi", "websockets", "deepeval", "langfuse")

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
    """`import oai_agentspec` で fastapi / websockets / deepeval / langfuse を強制ロードしない。"""
    forbidden = list(_FORBIDDEN_EXTRAS)
    probe = (
        "import sys\n"
        "import oai_agentspec\n"
        f"forbidden = {forbidden!r}\n"
        "loaded = [m for m in forbidden if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    out = _import_in_clean_subprocess(probe)
    loaded = [m for m in out.split(",") if m]
    assert loaded == [], f"本体 import で extra（{forbidden}）がロードされました: {loaded}"
