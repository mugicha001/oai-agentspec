"""`tests/examples/` 用の import 経路設定。

`examples/` はリポジトリ直下（`src/` レイアウト外）のローカルパッケージで、通常の
sys.path 構成（editable install が追加する `src/` のみ）には含まれない。`examples.*` を
`importlib.import_module` で読み込めるよう、リポジトリ直下を sys.path へ追加する。
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
