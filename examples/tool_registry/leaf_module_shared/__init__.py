"""葉モジュール分散登録パターン: 各 Tool ファイルが import 時に自ら登録する構成。

`_registry.py` に共有 `ToolRegistry` インスタンスを配置し、各 Tool ファイル（`weather.py` /
`docs.py` / `notification.py`）は import 時に `tool_registry.register(...)` を呼んで
登録を発火させる。`from . import weather, docs, notification` の順で全 Tool の登録が完了する
（`__init__.py` が実質的な「登録台帳」になる）。

**組み立て時の中央集権性**（1 つの Registry インスタンスに全 Tool が集約される）は
維持しつつ、**宣言の物理配置**は複数ファイルへ分散できる。Agent 少・Tool 多で個々の Tool を
機能単位でファイル分割したいときに向く。

Tool ファイル本体は lib（`oai_agentspec`）を import するが、これは登録行のみで、
関数実装自体は純粋な Python として書ける。
"""

from __future__ import annotations

# 各 Tool ファイルの import で `tool_registry.register(...)` が発火する。
# 順序に依存はないが、ファイル追加時のフォーマッタ挙動と可読性のためアルファベット順。
from . import docs, notification, weather  # noqa: F401
from ._registry import tool_registry

__all__ = ["tool_registry"]
