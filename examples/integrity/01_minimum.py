"""最小起動: `lockdown(<root>)` 1 行で disk 改竄検知。

git 同梱の `sample_app/` を読み込んで lockdown を実行する。

実行:
    uv run python examples/integrity/01_minimum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import SAMPLE_APP  # noqa: E402

from oai_agentspec import lockdown  # noqa: E402

lockdown(SAMPLE_APP, libs=False)
print("[OK] lockdown 通過（sample_app の sha256 検証成功）")
