"""共有 `ToolRegistry` インスタンスの葉モジュール。

Tool ファイル群が import して `tool_registry.register(...)` を呼ぶ。この葉モジュールは
`oai_agentspec` しか import せず、Tool 実装や他のアプリコードへの依存を持たないため、
循環 import のリスクを避けられる（AgentRegistry の運用と同型の設計）。
"""

from __future__ import annotations

from oai_agentspec import ToolRegistry

tool_registry = ToolRegistry()
