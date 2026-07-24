"""Realtime ルート L1 用の agents 非依存テストダブル。

`FakeRealtimeAgent` / `FakeRealtimeAgentBuilder` は `agents` を一切 import しない軽量スタブで、
`RealtimeAgentBuilder` Protocol を満たし L1（agents 非依存）の registry ロジック検証で使う。
SDK 依存のある `FakeRealtimeModel`（L2 用）は `fake_realtime_model.py` に分離しており、
本モジュールの import は `agents.realtime` を読み込まない。
既存の `fake_model.py` / `fake_builder.py` の様式を踏襲する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _FakeRealtimeHandoff:
    """`FakeRealtimeAgentBuilder.make_handoff` が返す結線オブジェクトのスタブ。

    Attributes:
        target: ハンドオフ先の構築済み `FakeRealtimeAgent`。
        config: エッジ設定（`RealtimeHandoffConfig` または None）。
    """

    target: Any
    config: Any = None


@dataclass
class FakeRealtimeAgent:
    """`agents.realtime.RealtimeAgent` の代わりにロジック検証で使う軽量スタブ。"""

    name: str
    instructions: Any = None
    prompt: Any = None
    handoffs: list[Any] = field(default_factory=list)
    tools: list[Any] = field(default_factory=list)


class FakeRealtimeAgentBuilder:
    """`RealtimeAgentBuilder` Protocol を満たすフェイク（`FakeRealtimeAgent` を返す）。

    `agents` を構築せず、registry の遅延構築・循環解決・巻き戻しのロジックを純粋に検証する。
    `build` は handoffs 空で構築し、`make_handoff` は `_FakeRealtimeHandoff` を返す
    （registry が `agent.handoffs.append` で後付け結線する様式）。
    """

    def __init__(self) -> None:
        self.built: list[str] = []

    def build(self, spec: Any) -> Any:
        """spec から handoffs 空の `FakeRealtimeAgent` を 1 つ構築する。"""
        self.built.append(spec.name)
        return FakeRealtimeAgent(
            name=spec.name,
            instructions=spec.instructions,
            prompt=spec.prompt,
            tools=list(spec.tools),
        )

    def make_handoff(self, agent: Any, config: Any) -> Any:
        """構築済み target とエッジ設定から `_FakeRealtimeHandoff` を生成する。"""
        return _FakeRealtimeHandoff(target=agent, config=config)
