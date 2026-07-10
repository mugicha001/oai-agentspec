"""RealtimeAgent 専用の宣言ルート（コア・runtime と並列の第 3 の宣言層・公開窓口）。

`RealtimeAgentSpec` / `RealtimeHandoffConfig`（宣言）と `RealtimeAgentRegistry`（登録 /
遅延構築 / 循環 handoff 解決 / validate）を提供する。SDK 結合は `_adapters` に閉じ、本
パッケージは agents を import しない（NFR-1）。コア `oai_agentspec.__all__` は汚さず、本
専用窓口（`oai_agentspec.realtime`）から取得する（`runtime.conversation` と同じ分離方式）。

`RealtimeAgentBuilder`（DI 注入点）は `__all__` に載せず、
`from oai_agentspec.realtime.protocols import RealtimeAgentBuilder` の直接 import で参照する。
"""

from __future__ import annotations

from .registry import RealtimeAgentRegistry
from .spec import RealtimeAgentSpec, RealtimeHandoffConfig

__all__ = [
    "RealtimeAgentRegistry",
    "RealtimeAgentSpec",
    "RealtimeHandoffConfig",
]
