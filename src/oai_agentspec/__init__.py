"""oai-agentspec: openai-agents 上の宣言的エージェント管理ライブラリ。

公開契約は `__all__` に掲載したシンボルのみ。バージョニングは SemVer に従う。
"""

from __future__ import annotations

from ._adapters import function_tool
from .agent_names import AgentNames, validate_agent_names
from .handoffs import HandoffEdge, HandoffGraph, from_specs
from .integrity import (
    IntegrityCheck,
    IntegrityError,
    PromptTemplateIntegrityError,
    lockdown,
)
from .next_turn import (
    NextTurnPolicy,
    NextTurnRule,
    action_next_turn_agent,
    apply_next_turn_policy,
    next_turn_agent,
    resolve_next_agent,
)
from .prompts import PromptLayout, PromptStore, PromptTemplate, dynamic_prompt
from .registry import AgentRegistry, RegistryFrozenError
from .spec import AgentSpec, HandoffConfig, SandboxAgentSpec
from .tool_registry import ToolRegistry, ToolSpec
from .workflow import (
    END,
    START,
    FacadeMode,
    NodeFn,
    NodeHook,
    NodeResults,
    Router,
    WorkflowFrozenError,
    WorkflowGraph,
    default_input_filter,
)

# 注: AgentBuilder（DI 拡張点）は `oai_agentspec.protocols` にあり、上級者向けのため
# トップレベル公開 API には含めない。SDK 隔離は `_adapters` が担う（docs/architecture.md）。
# 会話シンボル（ConversationService / SessionPolicy / Stream* 等）は実行寄り層へ移り、
# `oai_agentspec.runtime.conversation` 公開窓口経由で参照する（コア __all__ は宣言層のみ）。

__all__ = [
    "END",
    "START",
    "AgentNames",
    "AgentRegistry",
    "AgentSpec",
    "FacadeMode",
    "HandoffConfig",
    "HandoffEdge",
    "HandoffGraph",
    "IntegrityCheck",
    "IntegrityError",
    "NextTurnPolicy",
    "NextTurnRule",
    "NodeFn",
    "NodeHook",
    "NodeResults",
    "PromptLayout",
    "PromptStore",
    "PromptTemplate",
    "PromptTemplateIntegrityError",
    "RegistryFrozenError",
    "Router",
    "SandboxAgentSpec",
    "ToolRegistry",
    "ToolSpec",
    "WorkflowFrozenError",
    "WorkflowGraph",
    "action_next_turn_agent",
    "apply_next_turn_policy",
    "default_input_filter",
    "dynamic_prompt",
    "from_specs",
    "function_tool",
    "lockdown",
    "next_turn_agent",
    "resolve_next_agent",
    "validate_agent_names",
]


def main() -> int:
    """コンソールエントリポイント（会話 CLI へ委譲）。

    会話 CLI の依存（httpx / websockets）は cli extra のため、本体 import 時に強制 import
    しないよう `oai_agentspec.runtime.cli.main` を関数内で遅延 import する。

    Returns:
        会話 CLI の終了コード。
    """
    from .runtime.cli.main import main as cli_main

    return cli_main()
