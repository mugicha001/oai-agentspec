"""RunContextWrapper 展開の共有ヘルパ（`_adapters` 内専用）。

`Runner.run(context=...)` へ forward する前に、SDK が渡してくる
`RunContextWrapper` を生の run_context に開く定型処理を一元化する。
"""

from __future__ import annotations

from typing import Any


def unwrap_run_context(context: Any) -> Any:
    """`RunContextWrapper` なら `.context` を取り出し、それ以外はそのまま返す。

    Args:
        context: 生の run_context または `RunContextWrapper`（None も可）。

    Returns:
        生の run_context。
    """
    from agents import RunContextWrapper  # 関数内遅延 import（import 時 SDK 非依存を維持）

    return context.context if isinstance(context, RunContextWrapper) else context
