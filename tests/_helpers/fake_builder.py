"""L1（agents 非依存）テスト用のフェイク AgentBuilder。

`agents.Agent` を構築せず、registry のロジック（遅延構築・循環解決・差し替え）を
純粋に検証するためのフェイク。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from oai_agentspec.spec import AgentSpec


@dataclass
class FakeAgent:
    """agents.Agent の代わりにロジック検証で使う軽量スタブ。"""

    name: str
    instructions: Any = None
    prompt: Any = None
    handoffs: list[Any] = field(default_factory=list)
    tools: list[Any] = field(default_factory=list)


class FakeAgentBuilder:
    """protocols.AgentBuilder を満たすフェイク（FakeAgent を返す）。"""

    def __init__(self) -> None:
        self.built: list[str] = []

    def build(self, spec: AgentSpec) -> Any:
        self.built.append(spec.name)
        return FakeAgent(
            name=spec.name,
            instructions=spec.instructions,
            prompt=spec.prompt,
            tools=list(spec.tools),
        )
