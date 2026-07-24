"""Realtime ルート用のテストダブル。

`FakeRealtimeModel` は SDK `RealtimeModel` ABC（`connect` / `add_listener` /
`remove_listener` / `send_event` / `close` の 5 抽象）を全実装し、`connect` は conftest の
ネットワークガード下で通る no-op、listener へ事前設定イベント（`model_events`）や任意イベントを
流し込んで `RealtimeRunner` を駆動する（L2 の handoff 実委譲検証で使う）。

L1（agents 非依存）用の `FakeRealtimeAgent` / `FakeRealtimeAgentBuilder` は、本モジュールが
SDK（`agents.realtime`）をトップレベル import するため `fake_realtime_builder.py` に分離している。
既存の `fake_model.py` の様式を踏襲する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agents.realtime.model import RealtimeModel

if TYPE_CHECKING:
    from agents.realtime.model import RealtimeModelListener


@dataclass
class FakeRealtimeModel(RealtimeModel):
    """本物の WebSocket 接続を張らずに `RealtimeSession` を駆動するテスト用 RealtimeModel。

    `RealtimeModel` ABC の 5 抽象を全実装する。`connect` は no-op（ネットワークガード下でも通る）、
    `send_event` はセッションが送出したイベントを記録し、`emit` / `emit_all` で listener
    （= `RealtimeSession`）へモデルイベントを流し込んで tool call / handoff を発火させる。

    Attributes:
        model_events: `emit_all` で順に流す事前設定イベント列。
        listeners: `add_listener` で登録された listener（通常 `RealtimeSession` 1 件）。
        sent_events: セッションが `send_event` で送ってきたイベントの記録。
        connected: `connect` が呼ばれたか。
        closed: `close` が呼ばれたか。
    """

    model_events: list[Any] = field(default_factory=list)
    listeners: list[Any] = field(default_factory=list)
    sent_events: list[Any] = field(default_factory=list)
    connected: bool = False
    closed: bool = False

    async def connect(self, options: Any) -> None:
        """接続を確立した体で connected フラグのみ立てる no-op。"""
        self.connected = True

    def add_listener(self, listener: RealtimeModelListener) -> None:
        """listener を登録する。"""
        self.listeners.append(listener)

    def remove_listener(self, listener: RealtimeModelListener) -> None:
        """登録済み listener を解除する（未登録なら何もしない）。"""
        if listener in self.listeners:
            self.listeners.remove(listener)

    async def send_event(self, event: Any) -> None:
        """セッションが送ってきたイベントを記録する。"""
        self.sent_events.append(event)

    async def close(self) -> None:
        """セッションを閉じた体で closed フラグのみ立てる no-op。"""
        self.closed = True

    async def emit(self, event: Any) -> None:
        """登録済みの全 listener へ 1 件のモデルイベントを配信する。"""
        for listener in list(self.listeners):
            await listener.on_event(event)

    async def emit_all(self) -> None:
        """事前設定した `model_events` を順に全 listener へ配信する。"""
        for event in self.model_events:
            await self.emit(event)
