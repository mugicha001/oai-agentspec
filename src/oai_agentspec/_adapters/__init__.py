"""openai-agents（`agents`）への import 単一窓口。

本パッケージ配下にのみ `from agents import ...` を集約する（NFR-1）。
他モジュールは SDK 型をここからの再エクスポート経由で参照し、`agents` を直接 import しない。

実装は責務別サブモジュール（`builders` / `responses` / `models` / `runner` / `session`）に
分割し、本 `__init__` は全公開シンボル + private ヘルパ + SDK 型を再エクスポートする薄い集約に
徹する（import 互換の単一窓口を維持）。
"""

from __future__ import annotations

from agents import (
    Agent,
    DynamicPromptFunction,
    FunctionTool,
    GenerateDynamicPromptData,
    Handoff,
    ItemHelpers,
    Model,
    ModelSettings,
    Prompt,
    RunContextWrapper,
    Runner,
    Usage,
    function_tool,
)
from agents.items import ModelResponse
from agents.tool_context import ToolContext

from ._pending_store import (
    delete_pending_approval,
    list_pending_approvals,
    load_pending_approval,
    save_pending_approval,
)
from ._session_store import (
    get_session_items,
    list_session_ids,
    list_session_meta,
)
from .approvals import (
    apply_approvals,
    resume_outcome,
    resume_with_observation,
    unresolved_pending,
)
from .builders import (
    DefaultAgentBuilder,
    build_agent,
    make_agent_tool,
    make_dynamic_handoff,
    make_handoff,
    mock_spec_tools,
)
from .governance import (
    govern_spec,
    load_policy_bundle,
    new_audit_sink,
    policy_violation_error_type,
    resolve_policy,
)
from .guardrails import (
    OnTrip,
    attach_tool_guardrails,
    build_async_input_guardrail,
    build_async_output_guardrail,
    build_input_guardrail,
    build_output_guardrail,
    build_tool_input_guardrail,
    build_tool_output_guardrail,
    run_judge_prompt,
)
from .judge import (
    judge,
    judge_tools,
)
from .langfuse import (
    fetch_dataset_items,
    langfuse_send,
    register_dataset_items,
)
from .lightning import (
    judge_score,
    run_apo,
)
from .models import (
    DeterministicToolCallModel,
    WorkflowModel,
    workflow_as_tool,
)
from .responses import (
    _completed_event as _completed_event,
)
from .responses import (
    _text_delta_events as _text_delta_events,
)
from .responses import (
    _text_of as _text_of,
)
from .responses import (
    latest_user_text,
    make_facade_extra,
    make_required_tool_choice_settings,
    text_response,
    tool_call_response,
)
from .routing import (
    observe_run_result,
)
from .runner import (
    ApplyResult,
    DefaultRunnerAdapter,
    RunOutcome,
)
from .serialization import (
    StreamTextDelta,
    StreamTextDone,
    deserialize_state,
    run_streamed_outcome,
    run_streamed_text,
    serialize_state,
)
from .session import (
    close_session,
    make_session,
)
from .tracing import (  # noqa: F401 - workflow 層への内部窓口（公開 __all__ には積まない）
    WorkflowTracer,
    make_workflow_tracer,
)

__all__ = [
    "Agent",
    "DynamicPromptFunction",
    "FunctionTool",
    "GenerateDynamicPromptData",
    "Handoff",
    "ItemHelpers",
    "Model",
    "ModelResponse",
    "ModelSettings",
    "Prompt",
    "RunContextWrapper",
    "Runner",
    "ToolContext",
    "Usage",
    "build_agent",
    "mock_spec_tools",
    "DefaultAgentBuilder",
    "DefaultRunnerAdapter",
    "DeterministicToolCallModel",
    "WorkflowModel",
    "function_tool",
    "make_handoff",
    "make_dynamic_handoff",
    "make_agent_tool",
    "make_session",
    "close_session",
    "list_session_ids",
    "list_session_meta",
    "get_session_items",
    "save_pending_approval",
    "load_pending_approval",
    "delete_pending_approval",
    "list_pending_approvals",
    "run_streamed_text",
    "run_streamed_outcome",
    "StreamTextDelta",
    "StreamTextDone",
    "RunOutcome",
    "ApplyResult",
    "apply_approvals",
    "unresolved_pending",
    "resume_outcome",
    "resume_with_observation",
    "serialize_state",
    "deserialize_state",
    "latest_user_text",
    "make_facade_extra",
    "make_required_tool_choice_settings",
    "text_response",
    "tool_call_response",
    "workflow_as_tool",
    # LLMOps（採点 / 観測 / 実行トレース捕捉の plain 窓口・deepeval/langfuse はトップ非 import）
    "judge",
    "judge_tools",
    "langfuse_send",
    "register_dataset_items",
    "fetch_dataset_items",
    "observe_run_result",
    # Agent Lightning（APO 最適化の単一窓口・agentlightning はトップ非 import）
    "run_apo",
    "judge_score",
    # 内容ガードレール（agent / tool guardrail の SDK 接着窓口・agents はトップ import）
    "OnTrip",
    "build_input_guardrail",
    "build_output_guardrail",
    "build_async_input_guardrail",
    "build_async_output_guardrail",
    "build_tool_input_guardrail",
    "build_tool_output_guardrail",
    "attach_tool_guardrails",
    "run_judge_prompt",
    # AGT ガバナンス（ツール単位ポリシー強制 + 監査の SDK/AGT 結合窓口・AGT はトップ非 import）
    "govern_spec",
    "load_policy_bundle",
    "new_audit_sink",
    "policy_violation_error_type",
    "resolve_policy",
]
