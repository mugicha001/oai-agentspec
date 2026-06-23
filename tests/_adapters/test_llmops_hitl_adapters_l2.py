"""L2: LLMOps の HITL 完了採点アダプタ（mock_spec_tools / resume_with_observation）を検証する。

`mock_spec_tools` は宣言（`AgentSpec`）層でツール実行だけを `dataclasses.replace` で差し替え、
`needs_approval` を含む宣言メタデータを維持し、元 spec / 元 tool を mutate しないことを検証する
（#29 の核: ゲートを bypass せずツール本体だけモックする・宣言層で当てて build する）。
`resume_with_observation` は承認適用済み RunState から再開し plain な `RunOutcome` + `ObservedRun`
を返すことを、実 SDK Runner + 承認必須ツールの中断 → approve → 再開フローで検証する。
"""

from __future__ import annotations

import pytest
from agents import Agent, FunctionTool, Runner

from oai_agentspec import AgentSpec
from oai_agentspec._adapters import (
    apply_approvals,
    build_agent,
    mock_spec_tools,
    resume_with_observation,
)

from _helpers.approval import QueuedFakeModel, ToolRecorder, make_approval_tool
from _helpers.responses import text_response, tool_call_response

pytestmark = pytest.mark.integration


# ----------------------------------------------------------------------
# mock_spec_tools: 宣言層で実行だけ差し替え / メタデータ維持 / 元 spec 不変 / replaced 集合
# ----------------------------------------------------------------------


def _tool_by_name(tools: list[object], name: str) -> FunctionTool:
    """tools リストから指定名の FunctionTool を取り出す。"""
    return next(t for t in tools if isinstance(t, FunctionTool) and t.name == name)


def test_mock_spec_tools_replaces_only_invocation_and_keeps_metadata() -> None:
    """同名ツールの on_invoke_tool だけ差し替え、name / description / 引数スキーマ /
    needs_approval は維持する（エージェントが見るツール定義を変えない）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    spec = AgentSpec(name="bot", instructions="b", model=QueuedFakeModel(), tools=[tool])

    original = _tool_by_name(spec.tools, "danger")
    mocked_spec, replaced = mock_spec_tools(spec, {"danger": "mocked"})
    new_tool = _tool_by_name(mocked_spec.tools, "danger")

    # 実差し替えした (agent, tool) ペア集合を返す（approve 認可判定の根拠）。
    assert replaced == {("bot", "danger")}
    # 実行（on_invoke_tool）は差し替わるが、宣言メタデータ（name / description / 引数スキーマ /
    # needs_approval）は不変。評価対象の「ツールを呼ぶか」の判断が本番と同一になり、変わるのは
    # 実行された時の副作用だけになる。
    assert new_tool.on_invoke_tool is not original.on_invoke_tool
    assert new_tool.name == "danger"
    assert new_tool.description == original.description
    assert new_tool.params_json_schema == original.params_json_schema
    assert new_tool.needs_approval is True


def test_mock_spec_tools_does_not_mutate_original_spec() -> None:
    """元 spec / 元 tool は不変（新 spec を返す・宣言層 mock は元を破壊しない）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    spec = AgentSpec(name="bot", instructions="b", model=QueuedFakeModel(), tools=[tool])
    original_invoke = spec.tools[0].on_invoke_tool

    mocked_spec, _replaced = mock_spec_tools(spec, {"danger": "mocked"})

    # 元 spec は別オブジェクトで、その tool の実行本体は不変。
    assert mocked_spec is not spec
    assert spec.tools[0].on_invoke_tool is original_invoke


@pytest.mark.asyncio
async def test_mock_spec_tools_static_value_is_returned() -> None:
    """静的値モックは入力に関わらず str(値) を返す（本物のツール本体は呼ばれない）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    spec = AgentSpec(name="bot", instructions="b", model=QueuedFakeModel(), tools=[tool])
    mocked_spec, _replaced = mock_spec_tools(spec, {"danger": 42})

    out = await _tool_by_name(mocked_spec.tools, "danger").on_invoke_tool(None, '{"x": "v"}')
    assert out == "42"
    # 本物のツール本体は実行されない（recorder は空）。
    assert recorder.executed == []


@pytest.mark.asyncio
async def test_mock_spec_tools_callable_receives_args() -> None:
    """callable モックは JSON 引数 dict を受け取り戻りを str 化する。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    spec = AgentSpec(name="bot", instructions="b", model=QueuedFakeModel(), tools=[tool])
    mocked_spec, _replaced = mock_spec_tools(spec, {"danger": lambda args: f"got:{args.get('x')}"})

    out = await _tool_by_name(mocked_spec.tools, "danger").on_invoke_tool(None, '{"x": "abc"}')
    assert out == "got:abc"
    assert recorder.executed == []


@pytest.mark.asyncio
async def test_mock_spec_tools_callable_tolerates_malformed_or_nondict_json() -> None:
    """callable モックは不正 JSON / 非 dict JSON でも空 dict を渡して落ちない（防御的）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    spec = AgentSpec(name="bot", instructions="b", model=QueuedFakeModel(), tools=[tool])
    mocked_spec, _replaced = mock_spec_tools(spec, {"danger": lambda args: f"keys:{sorted(args)}"})

    invoke = _tool_by_name(mocked_spec.tools, "danger").on_invoke_tool
    # 不正 JSON（パース不能）。
    assert await invoke(None, "not-json") == "keys:[]"
    # 非 dict JSON（配列）。
    assert await invoke(None, "[1, 2]") == "keys:[]"


def test_mock_spec_tools_empty_is_noop() -> None:
    """tool_mocks 空なら元 spec をそのまま返し replaced は空集合。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    spec = AgentSpec(name="bot", instructions="b", model=QueuedFakeModel(), tools=[tool])
    mocked_spec, replaced = mock_spec_tools(spec, {})
    assert mocked_spec is spec
    assert replaced == set()


def test_mock_spec_tools_unknown_name_is_noop() -> None:
    """spec に無い名前だけ指定したら何も差し替えず元 spec をそのまま返す（replaced 空）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    spec = AgentSpec(name="bot", instructions="b", model=QueuedFakeModel(), tools=[tool])
    mocked_spec, replaced = mock_spec_tools(spec, {"other": "x"})
    assert mocked_spec is spec
    assert replaced == set()


def test_mock_spec_tools_partial_match_collects_only_replaced() -> None:
    """複数ツールのうち tool_mocks に在る名前だけ差し替え、replaced はその (agent, tool) だけ。"""
    recorder = ToolRecorder()
    a = make_approval_tool(recorder, name="danger")
    b = make_approval_tool(recorder, name="wire")
    spec = AgentSpec(name="bot", instructions="b", model=QueuedFakeModel(), tools=[a, b])

    mocked_spec, replaced = mock_spec_tools(spec, {"danger": "mock"})
    assert replaced == {("bot", "danger")}
    # wire は差し替えられず元の on_invoke のまま。
    assert (
        _tool_by_name(mocked_spec.tools, "wire").on_invoke_tool
        is _tool_by_name(spec.tools, "wire").on_invoke_tool
    )


# ----------------------------------------------------------------------
# resume_with_observation: 承認適用 → 再開 → plain RunOutcome + ObservedRun
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_with_observation_completes_after_mock_approve() -> None:
    """中断 → 宣言層 mock + approve → resume で完了し RunOutcome + ObservedRun を返す。"""
    from oai_agentspec._adapters import RunOutcome

    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = (
        QueuedFakeModel()
        .queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
        .queue(text_response("final answer"))
    )
    # 宣言層で mock を当ててから build する（build 済み Agent を mutate しない）。
    spec = AgentSpec(name="bot", instructions="b", model=model, tools=[tool])
    mocked_spec, replaced = mock_spec_tools(spec, {"danger": "mocked-output"})
    assert replaced == {("bot", "danger")}
    agent: Agent = build_agent(mocked_spec)

    # 初回実行は承認待ちで中断する。
    result = await Runner.run(agent, "do it")
    assert result.interruptions
    state = result.to_state()

    # approve を適用してから resume_with_observation で再開する。
    apply_approvals(state, [{"call_id": "c1", "decision": "approve"}])
    outcome, observation = await resume_with_observation(agent, state)

    assert isinstance(outcome, RunOutcome)
    assert outcome.interrupted is False
    assert outcome.final_output == "final answer"
    # 本物のツール本体は実行されない（モック差し替えのため recorder は空）。
    assert recorder.executed == []
    # ObservedRun が plain に返る（route / tool_calls を持つ）。
    assert observation is not None
    assert observation.route.last_agent == "bot"


@pytest.mark.asyncio
async def test_resume_with_observation_rejects_without_executing_tool() -> None:
    """reject 注入で resume すると却下後の応答を返し、ツールは実行されない（モック不要）。"""
    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = (
        QueuedFakeModel()
        .queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
        .queue(text_response("rejected response"))
    )
    agent = build_agent(AgentSpec(name="bot", instructions="b", model=model, tools=[tool]))

    result = await Runner.run(agent, "do it")
    state = result.to_state()
    apply_approvals(state, [{"call_id": "c1", "decision": "reject", "rejection_message": "no"}])
    outcome, _observation = await resume_with_observation(agent, state)

    assert outcome.interrupted is False
    assert outcome.final_output == "rejected response"
    # 却下のためツール本体は実行されない。
    assert recorder.executed == []


# ----------------------------------------------------------------------
# pending への agent_name 付与（approve 認可を (agent, tool) 単位にするため）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_includes_agent_name_from_interruption() -> None:
    """RunOutcome.pending / unresolved_pending は承認待ちを発生させた agent 名を含む。

    実 SDK の中断フローで `ToolApprovalItem.agent.name` を pending dict に付与できること
    （feasibility 検証の回帰防止・approve 認可の (agent, tool) 判定の前提）を確認する。
    """
    from oai_agentspec._adapters import unresolved_pending
    from oai_agentspec._adapters.runner import _outcome_from_result

    recorder = ToolRecorder()
    tool = make_approval_tool(recorder, name="danger")
    model = QueuedFakeModel().queue(tool_call_response("danger", '{"x": "v"}', call_id="c1"))
    agent = build_agent(
        AgentSpec(name="account-agent", instructions="b", model=model, tools=[tool])
    )

    result = await Runner.run(agent, "do it")
    # RunResult 由来の pending（_extract_pending 経由）に agent_name が載る。
    outcome = _outcome_from_result(result)
    assert outcome.pending == [
        {"tool_name": "danger", "call_id": "c1", "agent_name": "account-agent"}
    ]
    # RunState 由来の unresolved_pending にも agent_name が載る。
    state = result.to_state()
    pending = unresolved_pending(state)
    assert pending == [{"tool_name": "danger", "call_id": "c1", "agent_name": "account-agent"}]
