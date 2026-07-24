"""Mermaid flowchart 整形の共有ヘルパ（agents 非依存・最下層）。

通常ルート（`handoffs.HandoffGraph`）と Realtime 専用ルート
（`realtime.handoffs.RealtimeHandoffGraph`）の `mermaid()` が同一の flowchart 書式を
保つための純フォーマッタ。両ルートは宣言型を共用しないが、可視化出力の書式は
単一ソースで維持する（`_validation` と同じ共有 leaf パターン）。
"""

from __future__ import annotations

from collections.abc import Iterable


def render_flowchart(
    entry: str | None,
    static_edges: Iterable[tuple[str, str, str | None]],
    extra_lines: Iterable[str] = (),
) -> str:
    """(src, dst, label) 列と追加行から Mermaid flowchart 文字列を整形する。

    Args:
        entry: エントリノード名（None なら start 行を出力しない）。
        static_edges: 静的エッジの (src, dst, label) 列。label が falsy なら無ラベル。
        extra_lines: 末尾に加える整形済み行（通常ルートの動的破線エッジ等）。

    Returns:
        Mermaid の `flowchart TD` 文字列。
    """
    lines = ["flowchart TD"]
    if entry:
        lines.append(f"    start([start]) --> {entry}")
    for src, dst, label in static_edges:
        label_part = f"|{label}|" if label else ""
        lines.append(f"    {src} -->{label_part} {dst}")
    lines.extend(extra_lines)
    return "\n".join(lines)
