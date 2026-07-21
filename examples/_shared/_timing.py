"""実行時間計測の共有ヘルパー（examples 用）。

LLM 呼び出しを含む処理の所要時間を計測して表示する。examples の実行結果に
レイテンシの目安を添えるための補助で、lib 本体の機能ではない。

使い方:
    from _timing import stopwatch

    with stopwatch("classify"):
        prediction = await classifier.classify(query)
    # -> [TIME] classify: 1.23s
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def stopwatch(label: str = "elapsed") -> Iterator[None]:
    """ブロックの所要時間を計測し ``[TIME] <label>: N.NNs`` 形式で表示する。

    Args:
        label: 表示に使う処理名。
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[TIME] {label}: {time.perf_counter() - t0:.2f}s")
