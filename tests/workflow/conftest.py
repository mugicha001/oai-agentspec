"""workflow テスト用 fixture（tracing 検証局所）。

- `RecordingTracer`: `WorkflowTracer` Protocol を満たす fake。span 開閉イベントを
  events リストに記録する（L1 で発行プロトコル検証に使う）。
- `tracing_enabled` fixture（yield 方式）: setup で `set_tracing_disabled(False)` +
  collector（`TracingProcessor` 実装）を `add_trace_processor` で登録、teardown で
  `set_tracing_disabled(True)` + processor 列を空に戻す（root conftest の autouse
  オーバーライドを yield 寿命で安全に巻き戻す）。
- `_RecordingProcessor`: span name + parent_id を蓄積する最小 `TracingProcessor` 実装。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest

# ------------------------------------------------------------------
# RecordingTracer（L1 用・WorkflowTracer Protocol fake）
# ------------------------------------------------------------------


@dataclass
class RecordingTracer:
    """`WorkflowTracer` Protocol を満たす fake（L1 用）。

    各 span メソッド呼び出しの enter / exit を `events` リストに記録する。
    `("enter", span_kind, name, kwargs)` / `("exit", span_kind, name)` のタプル形式。

    Attributes:
        events: 記録された開閉イベント列。
    """

    events: list[tuple[Any, ...]] = field(default_factory=list)

    @contextmanager
    def workflow_span(self, graph_name: str) -> Iterator[None]:
        """workflow span 発行イベントを記録する。"""
        self.events.append(("enter", "workflow", graph_name, {}))
        try:
            yield None
        finally:
            self.events.append(("exit", "workflow", graph_name))

    @contextmanager
    def node_span(self, node_name: str, kind: str) -> Iterator[None]:
        """node span 発行イベントを記録する。"""
        self.events.append(("enter", "node", node_name, {"kind": kind}))
        try:
            yield None
        finally:
            self.events.append(("exit", "node", node_name))

    @contextmanager
    def condition_span(self, src: str) -> Iterator[None]:
        """condition span 発行イベントを記録する。"""
        self.events.append(("enter", "condition", src, {}))
        try:
            yield None
        finally:
            self.events.append(("exit", "condition", src))

    @contextmanager
    def fan_out_span(self, src: str) -> Iterator[None]:
        """fan-out span 発行イベントを記録する。"""
        self.events.append(("enter", "fan_out", src, {}))
        try:
            yield None
        finally:
            self.events.append(("exit", "fan_out", src))

    @contextmanager
    def fan_in_span(self, dst: str) -> Iterator[None]:
        """fan-in span 発行イベントを記録する。"""
        self.events.append(("enter", "fan_in", dst, {}))
        try:
            yield None
        finally:
            self.events.append(("exit", "fan_in", dst))


# ------------------------------------------------------------------
# _RecordingProcessor（L2 用・最小 TracingProcessor 実装）
# ------------------------------------------------------------------


@dataclass
class _SpanRecord:
    """L2 collector が記録する 1 span の最小プロファイル。

    Attributes:
        span_id: span の一意 ID。
        parent_id: 親 span ID（trace 直下なら None）。
        trace_id: 所属 trace ID。
        name: span name（CustomSpanData.name 由来）。
        data: span data（CustomSpanData.data 由来）。
    """

    span_id: str
    parent_id: str | None
    trace_id: str
    name: str | None
    data: dict[str, Any] | None


@dataclass
class _TraceRecord:
    """L2 collector が記録する 1 trace の最小プロファイル。"""

    trace_id: str
    name: str | None


class _RecordingProcessor:
    """span name + parent_id を蓄積する `TracingProcessor` 実装（L2 collector）。

    `on_span_end` 時点で `CustomSpanData.name` / `CustomSpanData.data` から span 情報を取り出して
    `spans` リストへ蓄積する。trace は `on_trace_start` で `traces` リストへ蓄積する。
    """

    def __init__(self) -> None:
        """spans / traces の蓄積リストを初期化する。"""
        self.spans: list[_SpanRecord] = []
        self.traces: list[_TraceRecord] = []

    def on_trace_start(self, trace: Any) -> None:
        """trace 開始を記録する。"""
        self.traces.append(
            _TraceRecord(
                trace_id=getattr(trace, "trace_id", ""),
                name=getattr(trace, "name", None),
            )
        )

    def on_trace_end(self, trace: Any) -> None:
        """no-op（end 時点では何もしない）。"""
        return None

    def on_span_start(self, span: Any) -> None:
        """no-op（end 時点で記録する）。"""
        return None

    def on_span_end(self, span: Any) -> None:
        """span 終了時に name / parent_id / data を記録する。"""
        span_data = getattr(span, "span_data", None)
        self.spans.append(
            _SpanRecord(
                span_id=getattr(span, "span_id", ""),
                parent_id=getattr(span, "parent_id", None),
                trace_id=getattr(span, "trace_id", ""),
                name=getattr(span_data, "name", None),
                data=getattr(span_data, "data", None),
            )
        )

    def shutdown(self) -> None:
        """no-op（テスト終了時の cleanup は不要）。"""
        return None

    def force_flush(self) -> None:
        """no-op（同期 collector のため即時反映済み）。"""
        return None


@pytest.fixture
def tracing_enabled() -> Iterator[_RecordingProcessor]:
    """tracing を局所オーバーライドで有効化する fixture（yield 方式・teardown 必須）。

    setup: `set_tracing_disabled(False)` + collector を `add_trace_processor` で登録。
    teardown: `set_tracing_disabled(True)` を再設定し、processor 列を空に戻す（root
    conftest の autouse `set_tracing_disabled(True)` を一時的に上書きするため、yield /
    finally で必ず元に戻す。set_trace_processors([]) は登録済み列を完全クリアする）。

    Yields:
        L2 検証用の collector（`spans` / `traces` リストを持つ）。
    """
    from agents import set_tracing_disabled
    from agents.tracing import set_trace_processors

    collector = _RecordingProcessor()
    set_tracing_disabled(False)
    set_trace_processors([collector])
    try:
        yield collector
    finally:
        set_tracing_disabled(True)
        set_trace_processors([])
