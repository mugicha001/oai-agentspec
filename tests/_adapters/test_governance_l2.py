"""L2: AGT ガバナンス統合の実 SDK 型統合検証（実 `function_tool` + registry + Runner）。

`AgentRegistry(agent_builder=GovernedAgentBuilder(...))` 経由で build した実 Agent に対し、
許可ツールの実行 / 拒否ツールの実関数非実行（`PolicyViolationError`）/ 監査ログの allow・deny
記録と `verify_chain()` / 既存 `spec.hooks` の合成委譲 / 非 FunctionTool 素通しと元 spec・tool の
非破壊性 / 既定 sink の build 間共有（agent_id 跨ぎのチェーン連続）を検証する。build 後に注入
される MCP origin tool（`spec.tools` を通らない経路）の deny が監査フック側の評価で
`UserError.__cause__` に `PolicyViolationError` を載せて着地することも併せて固定する。

SDK / AGT のバージョン耐性トリップワイヤを兼ねる: SDK `AgentHooksBase` の public ライフサイクル
メソッド集合（増えたら `_AuditAgentHooks` の監査記録の追随漏れを検知）と、AGT
`GovernancePolicy.check_tool / check_content`・`AuditLog.record / get_entries / verify_chain`・
`PolicyViolationError` の存在 / シグネチャを固定する。FakeModel で出力を制御し実 LLM を
呼ばない（決定的）。

MCP 経路は統治が fail-open（origin が MCP でなければ素通し）なため、依存する SDK 契約が破れても
例外もログも出ず MCP ツールが無警告で未統治になる。そのため実 `MCPUtil.to_function_tool` の
生成物を使って origin 付与 / 公開名 / 実例外の文字列化を pin し、`agents.tool.
get_function_tool_origin` の存在・`ToolOriginType` のメンバ集合・`on_tool_start` に渡る
`tool_arguments: str` も併せてトリップワイヤ化する（実 SDK 生成物が監査フックで実際に評価される
ことも deny / allow 両方向で固定する）。さらに宣言 `spec.mcp_servers` から SDK の run 時解決を経て
run loop が `on_tool_start` へ dispatch するまでの結合部を実 Runner で通し、deny / allow と
`include_server_in_tool_names` による SDK 生成の照合名（base 名 `mcp_{サーバ名}__{ツール名}` が
そのまま公開名になる単純分岐と、SDK が置換・切り詰め + ハッシュ付与を行う変形分岐の両方）を
pin する（SDK が MCP を専用 dispatch へ移す退行を、build 後注入ベースのテストでは検知できない
ため）。build 後に `Agent.hooks` を差し替えたとき MCP 経路の強制と監査がともに失われ、`spec.tools`
経路は強制と per-call の `tool:` レコードが残るという非対称も併せて固定する。
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import re
import warnings
import weakref
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "openai_agents_trust", reason="governance extra（agent-governance-toolkit）未導入"
)
# `mcp` は openai-agents の無条件依存だが、実 MCP ツール生成経路の pin が
# `mcp.types.Tool` に依存するため明示的にガードする。
pytest.importorskip("mcp", reason="mcp（openai-agents の依存）未導入")

import agents.tool as sdk_tool  # noqa: E402
from agents import Agent, FunctionTool, Runner, ToolOrigin, ToolOriginType, UserError  # noqa: E402
from agents.lifecycle import AgentHooksBase, RunHooksBase  # noqa: E402
from agents.mcp import MCPServer  # noqa: E402
from agents.mcp.util import MCPUtil  # noqa: E402
from agents.run_context import RunContextWrapper  # noqa: E402
from agents.tool import get_function_tool_origin  # noqa: E402
from agents.tool_context import ToolContext  # noqa: E402
from mcp.types import CallToolResult, GetPromptResult, ListPromptsResult  # noqa: E402
from mcp.types import Tool as MCPTool  # noqa: E402
from openai_agents_trust import AuditLog, GovernancePolicy  # noqa: E402

from oai_agentspec import AgentRegistry, AgentSpec, function_tool  # noqa: E402
from oai_agentspec._adapters import (  # noqa: E402
    govern_agent,
    govern_spec,
    govern_ungoverned_tools,
    new_audit_sink,
)
from oai_agentspec._adapters import governance as governance_module  # noqa: E402
from oai_agentspec._adapters.governance import (  # noqa: E402
    _govern_tool,
    _is_governed,
    _make_audit_hooks,
)
from oai_agentspec.runtime.governance import GovernedAgentBuilder  # noqa: E402

from _helpers.fake_model import FakeModel  # noqa: E402

with warnings.catch_warnings():
    # agent_os は legacy パッケージ名告知の DeprecationWarning を出すため抑制する。
    warnings.simplefilter("ignore", DeprecationWarning)
    from agent_os.exceptions import PolicyViolationError  # noqa: E402

pytestmark = pytest.mark.integration


# ----------------------------------------------------------------------
# helper: 実 FunctionTool（副作用フラグ付き）/ ToolContext
# ----------------------------------------------------------------------


def _make_tool(record: list[str], name: str = "echo") -> FunctionTool:
    """実行を `record` に記録する実 `FunctionTool` を作る（非実行の検証用副作用フラグ）。"""

    @function_tool(name_override=name)
    def _tool(text: str) -> str:
        """テキストを記録してエコーする。"""
        record.append(text)
        return f"echo:{text}"

    return _tool


def _mcp_origin_tool(record: list[str], name: str = "mcp_read") -> FunctionTool:
    """MCP origin メタを載せた `FunctionTool` を作る（build 後注入用・origin を偽装）。

    実 MCP サーバへ接続せず `MCPUtil.to_function_tool` を通さずに `_tool_origin` を直接載せる
    （origin 判定に必要なメタのみを再現する）。`record` には実ツール本体の呼び出し引数が積まれる。
    """

    async def _on_invoke_tool(ctx: Any, input_json: str) -> str:
        record.append(input_json)
        return "ok"

    return FunctionTool(
        name=name,
        description="fake mcp tool",
        params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
        on_invoke_tool=_on_invoke_tool,
        _tool_origin=ToolOrigin(type=ToolOriginType.MCP, mcp_server_name="srv"),
    )


def _tool_ctx(name: str, arguments: str) -> ToolContext:
    """govern ラップ済み `on_invoke_tool` を直接呼ぶための最小 `ToolContext` を作る。"""
    return ToolContext(context=None, tool_name=name, tool_call_id="c1", tool_arguments=arguments)


class _RecordingHooks:
    """既存 `spec.hooks` を模す記録フック（合成委譲の検証用・duck typing）。"""

    def __init__(self) -> None:
        """イベント記録を初期化する。"""
        self.events: list[str] = []
        self.llm_events: list[str] = []

    async def on_start(self, context: Any, agent: Any) -> None:
        self.events.append(f"start:{agent.name}")

    async def on_end(self, context: Any, agent: Any, output: Any) -> None:
        self.events.append(f"end:{agent.name}")

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        self.events.append(f"tool_start:{tool.name}")

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        self.events.append(f"tool_end:{tool.name}")

    async def on_handoff(self, context: Any, agent: Any, source: Any) -> None:
        self.events.append(f"handoff:{source.name}->{agent.name}")

    async def on_llm_start(
        self, context: Any, agent: Any, system_prompt: Any, input_items: Any
    ) -> None:
        self.llm_events.append("llm_start")

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        self.llm_events.append("llm_end")


# ----------------------------------------------------------------------
# 許可 / 拒否（registry + Runner 経由のエンドツーエンド）
# ----------------------------------------------------------------------


async def test_allowed_tool_executes_via_registry_and_runner() -> None:
    """許可ツールは実関数が実行され、監査ログに allow 一式が記録されチェーン検証が通る。"""
    calls: list[str] = []
    tool = _make_tool(calls)
    sink = AuditLog()
    policy = GovernancePolicy(name="p", allowed_tools=["echo"])
    reg = AgentRegistry(agent_builder=GovernedAgentBuilder(policy=policy, audit_sink=sink))
    model = FakeModel().queue_tool_call("echo", '{"text": "hi"}').queue_text("done")
    reg.register(AgentSpec(name="bot", instructions="i", model=model, tools=[tool]))
    agent = reg.get("bot")

    result = await Runner.run(agent, input="go")

    assert calls == ["hi"]  # 実関数が実行された
    assert result.final_output == "done"
    entries = sink.get_entries()
    triples = [(e.agent_id, e.action, e.decision) for e in entries]
    # ツール単位の allow + ライフサイクル監査が揃う。
    assert ("bot", "tool:echo", "allow") in triples
    assert ("bot", "agent_start", "allow") in triples
    assert ("bot", "agent_end", "allow") in triples
    assert ("bot", "tool_start:echo", "allow") in triples
    assert ("bot", "tool_end:echo", "allow") in triples
    # allow 記録にはツール引数 JSON が全文残る（監査要件）。
    tool_entry = next(e for e in entries if e.action == "tool:echo")
    assert tool_entry.details == {"arguments": '{"text": "hi"}'}
    assert sink.verify_chain() is True


async def test_denied_tool_not_executed_and_raises_via_runner() -> None:
    """拒否ツールは実関数を実行せず例外で中断し、監査ログに deny が記録される。"""
    calls: list[str] = []
    tool = _make_tool(calls)
    sink = AuditLog()
    policy = GovernancePolicy(name="p", allowed_tools=["other"])
    reg = AgentRegistry(agent_builder=GovernedAgentBuilder(policy=policy, audit_sink=sink))
    model = FakeModel().queue_tool_call("echo", '{"text": "nope"}').queue_text("unreached")
    reg.register(AgentSpec(name="bot", instructions="i", model=model, tools=[tool]))
    agent = reg.get("bot")

    with pytest.raises(Exception) as excinfo:
        await Runner.run(agent, input="go")

    # SDK が tool 実行例外をラップしても原因は PolicyViolationError（生伝搬でも可）。
    err = excinfo.value
    assert isinstance(err, PolicyViolationError) or isinstance(err.__cause__, PolicyViolationError)
    assert calls == []  # 実関数は非実行のまま拒否された
    deny = next(e for e in sink.get_entries() if e.action == "tool:echo")
    assert deny.decision == "deny"
    assert deny.details["arguments"] == '{"text": "nope"}'
    assert "echo" in deny.details["reason"]
    # 拒否で実行が中断されるため tool_end / agent_end は記録されない。
    actions = [e.action for e in sink.get_entries()]
    assert "tool_end:echo" not in actions
    assert "agent_end" not in actions
    assert sink.verify_chain() is True


async def test_mcp_origin_tool_deny_lands_as_user_error_cause_via_runner() -> None:
    """A12: build 後注入した MCP origin tool の deny が `UserError.__cause__` に載って着地する。

    MCP ツールは実行時に SDK 側で agent へ注入されるため、`spec.tools` には現れず build 時の
    govern ラップ（`_govern_tool`）が掛からない。ここでは build 後の `agent.tools.append(...)`
    で経路を再現し、監査フック（`on_tool_start`）側の評価だけで deny が効くことを固定する。
    """
    invoked: list[str] = []
    sink = AuditLog()
    policy = GovernancePolicy(name="p", allowed_tools=["allowed_only"])
    reg = AgentRegistry(agent_builder=GovernedAgentBuilder(policy=policy, audit_sink=sink))
    model = FakeModel().queue_tool_call("mcp_read", '{"q": "x"}').queue_text("unreached")
    reg.register(AgentSpec(name="bot", instructions="i", model=model))
    agent = reg.get("bot")
    # build 後注入（`spec.tools` に置くと build 時ラップも同時に掛かり経路が混ざる）。
    agent.tools.append(_mcp_origin_tool(invoked))

    with pytest.raises(UserError) as excinfo:
        await Runner.run(agent, input="go")

    assert isinstance(excinfo.value.__cause__, PolicyViolationError)
    assert invoked == []  # 実ツール本体は実行されない
    # 記録列全体を `==` で固定する（`in` 判定では `tool:` の重複記録を検知できない）。
    # 拒否で run が中断されるため tool_end / agent_end も現れないことが同時に固定される。
    assert [(e.agent_id, e.action, e.decision) for e in sink.get_entries()] == [
        ("bot", "agent_start", "allow"),
        ("bot", "tool_start:mcp_read", "allow"),
        ("bot", "tool:mcp_read", "deny"),
    ]
    deny = next(e for e in sink.get_entries() if e.action == "tool:mcp_read")
    assert deny.details["arguments"] == '{"q": "x"}'
    assert "mcp_read" in deny.details["reason"]
    assert sink.verify_chain() is True


# ----------------------------------------------------------------------
# 許可 / 拒否（govern ラップ済み on_invoke_tool の直接呼び出し）
# ----------------------------------------------------------------------


async def test_governed_tool_direct_invocation_allow_and_deny() -> None:
    """直接呼び出しでも許可は実関数を実行し、拒否は `PolicyViolationError` を送出する。"""
    calls: list[str] = []
    tool = _make_tool(calls)

    allowed = govern_spec(
        AgentSpec(name="bot", instructions="i", tools=[tool]),
        policy=GovernancePolicy(name="p", allowed_tools=["echo"]),
        audit_sink=AuditLog(),
    ).tools[0]
    out = await allowed.on_invoke_tool(_tool_ctx("echo", '{"text": "hi"}'), '{"text": "hi"}')
    assert out == "echo:hi"
    assert calls == ["hi"]

    denied = govern_spec(
        AgentSpec(name="bot", instructions="i", tools=[tool]),
        policy=GovernancePolicy(name="p", allowed_tools=[]),  # 空 allowlist = 全拒否
        audit_sink=AuditLog(),
    ).tools[0]
    with pytest.raises(PolicyViolationError, match="echo"):
        await denied.on_invoke_tool(_tool_ctx("echo", '{"text": "x"}'), '{"text": "x"}')
    assert calls == ["hi"]  # 拒否側では増えない


async def test_blocked_patterns_deny_json_escaped_arguments() -> None:
    """実 AGT の blocked_patterns でも JSON エスケープ表現（\\u0072m = rm）が deny される。"""
    calls: list[str] = []
    tool = _make_tool(calls, name="sh")
    governed = govern_spec(
        AgentSpec(name="bot", instructions="i", tools=[tool]),
        policy=GovernancePolicy(name="p", blocked_patterns=["rm -rf"]),
        audit_sink=AuditLog(),
    ).tools[0]
    escaped = '{"text": "\\u0072m -rf /"}'  # 生文字列に "rm -rf" は現れない

    with pytest.raises(PolicyViolationError, match="blocked pattern"):
        await governed.on_invoke_tool(_tool_ctx("sh", escaped), escaped)
    assert calls == []  # 実関数は非実行


# ----------------------------------------------------------------------
# govern_agent / govern_ungoverned_tools（構築済み実 Agent への統治焼き込み・実 AGT）
# ----------------------------------------------------------------------


async def test_govern_agent_allow_deny_records_same_shape() -> None:
    """`govern_agent` 経由のラップも allow / deny の監査レコードと拒否 payload が同形になる。

    allow は `details == {"arguments": ...}`、deny は `PolicyViolationError.details ==
    {"tool_name", "reason"}`（引数は載らない）と sink の `{"reason", "arguments"}`。`govern_agent`
    は clone を返す factory 経路専用のため、元 Agent の tool は不変であることも固定する。
    """
    calls: list[str] = []
    tool = _make_tool(calls)
    src = Agent(name="bot", instructions="i", tools=[tool])

    allow_sink = AuditLog()
    allowed = govern_agent(
        src,
        policy=GovernancePolicy(name="p", allowed_tools=["echo"]),
        audit_sink=allow_sink,
        agent_name="bot",
    )
    out = await allowed.tools[0].on_invoke_tool(
        _tool_ctx("echo", '{"text": "hi"}'), '{"text": "hi"}'
    )
    assert out == "echo:hi"
    assert calls == ["hi"]
    assert [(e.agent_id, e.action, e.decision, e.details) for e in allow_sink.get_entries()] == [
        ("bot", "tool:echo", "allow", {"arguments": '{"text": "hi"}'})
    ]

    deny_sink = AuditLog()
    denied = govern_agent(
        src,
        policy=GovernancePolicy(name="p", allowed_tools=[]),  # 空 allowlist = 全拒否
        audit_sink=deny_sink,
        agent_name="bot",
    )
    with pytest.raises(PolicyViolationError, match="echo") as excinfo:
        await denied.tools[0].on_invoke_tool(_tool_ctx("echo", '{"text": "x"}'), '{"text": "x"}')
    assert calls == ["hi"]  # 拒否側では実関数を実行しない

    entries = deny_sink.get_entries()
    assert [(e.agent_id, e.action, e.decision) for e in entries] == [("bot", "tool:echo", "deny")]
    reason = entries[0].details["reason"]
    assert isinstance(reason, str) and reason
    assert entries[0].details == {"reason": reason, "arguments": '{"text": "x"}'}
    assert excinfo.value.details == {"tool_name": "echo", "reason": reason}
    assert src.tools == [tool]  # clone 経路は元 Agent を変えない
    assert src.tools[0] is tool


async def test_govern_ungoverned_tools_keeps_sdk_agent_runnable() -> None:
    """`govern_ungoverned_tools` で統治した実 Agent は同一 list・hooks 不変のまま Runner で動く。

    spec 経路の第 3 段は hooks を再合成しないため、`agent.hooks is None` の Agent では記録は
    実行本体ラップ由来の `tool:` 1 行だけになる（記録列全体を `==` で固定する）。
    """
    calls: list[str] = []
    sink = AuditLog()
    model = FakeModel().queue_tool_call("echo", '{"text": "hi"}').queue_text("done")
    tools = [_make_tool(calls)]
    agent = Agent(name="bot", instructions="i", model=model, tools=tools)

    govern_ungoverned_tools(
        agent,
        policy=GovernancePolicy(name="p", allowed_tools=["echo"]),
        audit_sink=sink,
        agent_name="bot",
    )

    assert agent.tools is tools
    assert agent.hooks is None
    result = await Runner.run(agent, input="go")
    assert result.final_output == "done"
    assert calls == ["hi"]
    assert [(e.agent_id, e.action, e.decision, e.details) for e in sink.get_entries()] == [
        ("bot", "tool:echo", "allow", {"arguments": '{"text": "hi"}'})
    ]
    assert sink.verify_chain() is True


# ----------------------------------------------------------------------
# 既存 spec.hooks の合成（上書きでなく委譲）
# ----------------------------------------------------------------------


async def test_existing_spec_hooks_composed_and_delegated() -> None:
    """既存 `spec.hooks` は上書きされず、監査記録と併走して同名メソッドへ委譲される。"""
    calls: list[str] = []
    tool = _make_tool(calls)
    inner_hooks = _RecordingHooks()
    sink = AuditLog()
    reg = AgentRegistry(
        agent_builder=GovernedAgentBuilder(
            policy=GovernancePolicy(name="p", allowed_tools=["echo"]), audit_sink=sink
        )
    )
    model = FakeModel().queue_tool_call("echo", '{"text": "hi"}').queue_text("done")
    spec = AgentSpec(name="bot", instructions="i", model=model, tools=[tool], hooks=inner_hooks)
    reg.register(spec)
    agent = reg.get("bot")

    # agent.hooks は合成フックに置き換わる（既存フックそのものではない）。
    assert agent.hooks is not inner_hooks
    await Runner.run(agent, input="go")

    # 既存フックへ委譲される（ライフサイクル順）。
    assert inner_hooks.events == ["start:bot", "tool_start:echo", "tool_end:echo", "end:bot"]
    # on_llm_start / on_llm_end は監査対象外だが委譲は行われる。
    assert "llm_start" in inner_hooks.llm_events
    assert "llm_end" in inner_hooks.llm_events
    # 監査記録も並行して残る（委譲がフックを失わせない）。
    triples = [(e.action, e.decision) for e in sink.get_entries()]
    assert ("agent_start", "allow") in triples
    assert ("agent_end", "allow") in triples
    # 元 spec.hooks は不変（非破壊）。
    assert spec.hooks is inner_hooks


async def test_audit_hooks_on_handoff_records_and_delegates() -> None:
    """合成フックの `on_handoff` は source/target 名で監査記録し、既存フックへ委譲する。"""
    inner_hooks = _RecordingHooks()
    sink = AuditLog()
    hooks = _make_audit_hooks(sink, inner_hooks)

    class _Named:
        def __init__(self, name: str) -> None:
            self.name = name

    await hooks.on_handoff(None, _Named("target"), _Named("src"))

    entry = sink.get_entries()[0]
    assert entry.agent_id == "src"
    assert entry.action == "handoff:target"
    assert entry.decision == "allow"
    assert inner_hooks.events == ["handoff:src->target"]


async def test_audit_hooks_without_inner_returns_audit_hooks_itself() -> None:
    """`inner=None` では合成ラッパを被せず監査フック自身を返す（記録のみ・`on_llm_*` は no-op）。

    `_make_audit_hooks` は `chain_agent_hooks(audit, inner)` を返すため、`inner` が `None` の
    ときは実効 1 件かつ `isinstance(audit, AgentHooksBase)` が真であることを根拠に audit 自身が
    `is` 一致で返る。監査専用クラスから基底 `AgentHooks[Any]`（= `AgentHooksBase`）の継承を
    外すと `isinstance` が偽になり不要なラッパが 1 個挟まるため、この pin が当該前提条件を守る。
    """
    sink = AuditLog()
    hooks = _make_audit_hooks(sink, None)

    # 合成ラッパではなく監査専用クラスのインスタンスがそのまま返る。
    assert type(hooks).__name__ == "_AuditAgentHooks"
    assert isinstance(hooks, AgentHooksBase)

    class _Named:
        def __init__(self, name: str) -> None:
            self.name = name

    # 監査対象メソッドは記録される（委譲先が無くても例外を出さない）。
    await hooks.on_tool_start(None, _Named("bot"), _Named("echo"))
    assert [(e.agent_id, e.action, e.decision) for e in sink.get_entries()] == [
        ("bot", "tool_start:echo", "allow")
    ]

    # 監査対象外の on_llm_start は基底の no-op が呼ばれるだけで記録されない。
    await hooks.on_llm_start(None, _Named("bot"), None, [])
    assert len(sink.get_entries()) == 1


async def test_audit_hooks_with_policy_without_inner_returns_audit_hooks_itself() -> None:
    """A11: policy を渡しても `inner=None` なら合成ラッパを被せず監査フック自身を返す。

    MCP ツール評価の追加が `chain_agent_hooks` の要素数・合成条件に影響しないこと
    （`inner=None` 時に余計なラッパが 1 個挟まらないこと）を固定する。
    """
    sink = AuditLog()
    hooks = _make_audit_hooks(
        sink,
        None,
        policy=GovernancePolicy(name="p", allowed_tools=["echo"]),
        denied_exc=PolicyViolationError,
        agent_name="bot",
    )

    assert type(hooks).__name__ == "_AuditAgentHooks"
    assert isinstance(hooks, AgentHooksBase)


async def test_audit_record_precedes_inner_delegation() -> None:
    """合成順が `(監査, 既存フック)` であること（既存フックが raise しても監査記録が残る）。

    `_make_audit_hooks` は `chain_agent_hooks(audit, inner)` を返し、合成は fail-fast である。
    したがって既存フックが `on_start` で例外を送出した場合、
    - 正しい順序 `(audit, inner)`: 監査記録が先に完了し、記録が sink に残る
    - 反転した順序 `(inner, audit)`: 既存フックの例外で後段の監査へ到達せず、記録が失われる
    という観測可能な差が生じる。引数順の反転は `sink` と `inner` を別コレクションで独立に検証する
    テストでは検知できないため、この pin が順序そのものを挙動差として固定する。
    """

    class _RaisingInnerHooks(AgentHooksBase[Any, Any]):
        """`on_start` で必ず例外を送出する既存フック（`spec.hooks` 相当）。"""

        async def on_start(self, context: Any, agent: Any) -> None:
            """常に `RuntimeError` を送出する。"""
            raise RuntimeError("inner boom")

    class _Named:
        def __init__(self, name: str) -> None:
            self.name = name

    sink = AuditLog()
    hooks = _make_audit_hooks(sink, _RaisingInnerHooks())

    with pytest.raises(RuntimeError, match="inner boom"):
        await hooks.on_start(None, _Named("bot"))

    # 監査記録は既存フックの委譲より前に完了しているため残る。
    assert [(e.agent_id, e.action, e.decision) for e in sink.get_entries()] == [
        ("bot", "agent_start", "allow")
    ]


def test_govern_spec_rejects_run_scope_hooks_in_spec_hooks() -> None:
    """`spec.hooks` に run 単位フックを置いた宣言は build 時に `TypeError` で落ちる。

    `_make_audit_hooks` は `chain_agent_hooks(audit, inner)` を通るため、run 単位フックを
    agent スロットへ入れた宣言は合成時に拒否される（ADR-0017）。従来は `on_start` / `on_end`
    が silent skip され `on_handoff` は from/to が反転して誤記録が残っていたため、fail-fast へ
    変えた振る舞い変更の pin。
    """

    class _RunScopeHooks(RunHooksBase[Any, Any]):
        async def on_agent_start(self, context: Any, agent: Any) -> None:
            """run 単位の開始通知（agent 単位の `on_start` とは別名）。"""

    spec = AgentSpec(name="bot", instructions="i", hooks=_RunScopeHooks())

    with pytest.raises(TypeError) as excinfo:
        govern_spec(spec, policy=GovernancePolicy(), audit_sink=AuditLog())

    assert "chain_hooks" in str(excinfo.value)


# ----------------------------------------------------------------------
# 非 FunctionTool 素通し / 元 spec・tool の非破壊性
# ----------------------------------------------------------------------


def test_non_function_tool_passthrough_and_originals_unchanged() -> None:
    """非 FunctionTool は素通しされ、元 spec / tool は一切破壊されない（メタは維持）。"""
    calls: list[str] = []
    tool = _make_tool(calls)
    hosted = object()  # hosted tool 相当のダミー（FunctionTool ではない）
    original_invoke = tool.on_invoke_tool
    spec = AgentSpec(name="bot", instructions="i", tools=[tool, hosted])

    governed = govern_spec(spec, policy=GovernancePolicy(name="p"), audit_sink=AuditLog())

    # 非 FunctionTool は同一オブジェクトのまま素通し。
    assert governed.tools[1] is hosted
    g_tool = governed.tools[0]
    assert isinstance(g_tool, FunctionTool)
    assert g_tool is not tool
    # 宣言メタは維持・差し替えは実行本体のみ。
    assert g_tool.name == tool.name
    assert g_tool.description == tool.description
    assert g_tool.params_json_schema == tool.params_json_schema
    assert g_tool.strict_json_schema == tool.strict_json_schema
    assert g_tool.needs_approval == tool.needs_approval
    assert g_tool.on_invoke_tool is not original_invoke
    # 元 spec / tool は不変。handoffs も変更されない。
    assert spec.tools == [tool, hosted]
    assert spec.hooks is None
    assert tool.on_invoke_tool is original_invoke
    assert governed.handoffs == spec.handoffs


# ----------------------------------------------------------------------
# 既定 sink の build 間共有（agent_id 跨ぎでチェーン連続）
# ----------------------------------------------------------------------


async def test_default_sink_shared_across_builds_with_continuous_chain() -> None:
    """`audit_sink` 未指定の既定 sink は初回 build で生成され spec を跨いで共有される。"""
    calls_a: list[str] = []
    calls_b: list[str] = []
    builder = GovernedAgentBuilder(policy=GovernancePolicy(name="p"))
    assert builder.audit_sink is None  # build 前は未生成
    reg = AgentRegistry(agent_builder=builder)
    reg.register(AgentSpec(name="a", instructions="i", tools=[_make_tool(calls_a, name="ta")]))
    reg.register(AgentSpec(name="b", instructions="i", tools=[_make_tool(calls_b, name="tb")]))
    agent_a = reg.get("a")
    agent_b = reg.get("b")

    sink = builder.audit_sink
    assert isinstance(sink, AuditLog)  # 初回 build で AGT 既定 sink が生成・共有される

    await agent_a.tools[0].on_invoke_tool(_tool_ctx("ta", '{"text": "1"}'), '{"text": "1"}')
    await agent_b.tools[0].on_invoke_tool(_tool_ctx("tb", '{"text": "2"}'), '{"text": "2"}')

    entries = sink.get_entries()
    assert [(e.agent_id, e.action, e.decision) for e in entries] == [
        ("a", "tool:ta", "allow"),
        ("b", "tool:tb", "allow"),
    ]
    # agent_id 跨ぎでハッシュチェーンが連続している（sink 分断なし）。
    assert entries[1].previous_hash == entries[0].entry_hash
    assert sink.verify_chain() is True


def test_new_audit_sink_returns_fresh_agt_audit_log() -> None:
    """`new_audit_sink` は AGT `AuditLog` を都度新規生成する。"""
    sink = new_audit_sink()
    assert isinstance(sink, AuditLog)
    assert new_audit_sink() is not sink


# ----------------------------------------------------------------------
# バージョン耐性トリップワイヤ（SDK AgentHooksBase / AGT API）
# ----------------------------------------------------------------------


def test_sdk_agent_hooks_lifecycle_method_set_tripwire() -> None:
    """SDK `AgentHooksBase` の public ライフサイクルメソッド集合が変化したら fail させる。

    集合が増えた場合、`_make_audit_hooks` の `_AuditAgentHooks` に監査記録を追加しないと
    新メソッドの監査が黙って漏れる（監査記録側の追随漏れの検知）。既存 `spec.hooks` への
    委譲は `chain_agent_hooks` が担うため、委譲漏れは
    `tests/runtime/hooks/test_chain_agent_hooks_l2.py` の SDK パリティ tripwire が検知する。
    """
    expected = {
        "on_start",
        "on_end",
        "on_handoff",
        "on_tool_start",
        "on_tool_end",
        "on_llm_start",
        "on_llm_end",
    }
    actual = {
        name
        for name, _ in inspect.getmembers(AgentHooksBase, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    assert actual == expected, (
        "SDK AgentHooksBase のライフサイクルメソッド集合が変化した。"
        "_adapters/governance.py の _AuditAgentHooks の監査記録を追従させること。"
        f" 差分: {sorted(actual.symmetric_difference(expected))}"
    )


def test_agt_governance_policy_api_tripwire() -> None:
    """AGT `GovernancePolicy` のフィールド集合と評価メソッドの存在 / 挙動を固定する。

    フィールド集合が変化すると `_load_policy` の未知キー検証 / 非強制フィールド警告の前提が
    変わるため、追従要否の判断ポイントとして fail させる。
    """
    field_names = {f.name for f in dataclasses.fields(GovernancePolicy)}
    assert field_names == {
        "name",
        "max_tokens",
        "max_tool_calls",
        "blocked_patterns",
        "allowed_tools",
        "min_trust_score",
        "require_identity",
    }
    # check_tool / check_content: (self, <text>) -> str | None。
    for method in ("check_tool", "check_content"):
        params = list(inspect.signature(getattr(GovernancePolicy, method)).parameters)
        assert len(params) == 2, f"{method} のシグネチャが変化した: {params}"
    policy = GovernancePolicy(name="p", allowed_tools=["a"], blocked_patterns=["rm"])
    assert policy.check_tool("a") is None
    assert isinstance(policy.check_tool("b"), str)
    assert isinstance(policy.check_content("rm -rf /"), str)
    assert policy.check_content("safe") is None


def test_agt_audit_log_api_tripwire() -> None:
    """AGT `AuditLog` の `record / get_entries / verify_chain` シグネチャと挙動を固定する。"""
    params = inspect.signature(AuditLog.record).parameters
    assert list(params) == ["self", "agent_id", "action", "decision", "details"]
    assert params["details"].default is None
    log = AuditLog()
    log.record(agent_id="a", action="x", decision="allow")
    log.record(agent_id="a", action="y", decision="deny", details={"reason": "r"})
    entries = log.get_entries()
    assert len(entries) == 2
    assert entries[1].details == {"reason": "r"}
    # チェーン検証に使うエントリ属性が存在する。
    entry_fields = {f.name for f in dataclasses.fields(entries[0])}
    assert entry_fields >= {
        "agent_id",
        "action",
        "decision",
        "details",
        "previous_hash",
        "entry_hash",
    }
    assert log.verify_chain() is True


def test_agt_policy_violation_error_tripwire() -> None:
    """AGT `PolicyViolationError` は Exception 派生でメッセージ付き送出できる。"""
    assert issubclass(PolicyViolationError, Exception)
    with pytest.raises(PolicyViolationError, match="boom"):
        raise PolicyViolationError("boom")


# ----------------------------------------------------------------------
# govern ラップの SDK 契約トリップワイヤ（注釈引き継ぎ / dataclasses.replace）
# ----------------------------------------------------------------------


def test_govern_wrap_propagates_context_annotation() -> None:
    """govern 済み on_invoke_tool は元実装の第 1 引数注釈を引き継ぐ（SDK のコンテキスト選択用）。

    SDK は注釈で full ToolContext / 縮約 RunContextWrapper を選ぶ
    （agents/tool.py の _get_function_tool_invoke_context）。Any のままだと縮約契約のツールに
    full ToolContext が渡る退行になるため、引き継ぎを固定する。
    """
    tool = _make_tool([], name="annotated")
    original_ann = next(iter(inspect.signature(tool.on_invoke_tool).parameters.values())).annotation
    assert original_ann not in (inspect.Parameter.empty, None)

    spec = AgentSpec(name="bot", instructions="i", tools=[tool])
    governed = govern_spec(spec, policy=GovernancePolicy(name="p"), audit_sink=AuditLog())
    wrapped_ann = next(
        iter(inspect.signature(governed.tools[0].on_invoke_tool).parameters.values())
    ).annotation
    assert wrapped_ann == original_ann


def test_dataclasses_replace_keeps_custom_on_invoke_tool() -> None:
    """dataclasses.replace が on_invoke_tool 差し替えを保持する（SDK の再バインド退行検知）。

    将来 SDK の __post_init__ が on_invoke_tool を自己再バインドする実装になると govern ラップが
    外れるため、置換がそのまま残る現行契約をトリップワイヤとして固定する。
    """
    tool = _make_tool([], name="replaceable")

    async def _custom(ctx: Any, input_json: str) -> str:
        return "custom"

    replaced = dataclasses.replace(tool, on_invoke_tool=_custom)
    assert replaced.on_invoke_tool is _custom


# ----------------------------------------------------------------------
# MCP 経路の SDK 契約トリップワイヤ（origin 付与 / origin 型集合 / 引数型 / 例外の文字列化）
#
# 本経路の統治は fail-open（origin が MCP でなければ素通し）のため、以下の契約が破れても
# 例外もログも出ず MCP ツールが無警告で未統治になる。CI で SDK upgrade を検知する唯一の
# 手段としてトリップワイヤで固定する。
# ----------------------------------------------------------------------


class _StubMCPServer(MCPServer):
    """`MCPUtil.to_function_tool` を通すための最小 MCP サーバー（実接続なし）。

    `_get_failure_error_function` / `_get_needs_approval_for_tool` は SDK 既定の挙動を使うため
    `MCPServer`（abc）を継承して private ヘルパを継承で得る（duck-typed で自前実装すると
    「SDK 既定の `failure_error_function` が効く」という B5 の pin 対象そのものを偽装してしまう）。
    """

    def __init__(
        self,
        name: str = "srv",
        *,
        fail_with: Exception | None = None,
        tools: list[MCPTool] | None = None,
    ) -> None:
        """サーバー名と `call_tool` の失敗挙動 / 公開ツール一覧を設定する。

        Args:
            name: `ToolOrigin.mcp_server_name` に載るサーバー名。
            fail_with: `call_tool` が送出する例外（None なら空結果を返す）。
            tools: `list_tools` が返す MCP ツール一覧（None なら空。SDK が run 時に
                `spec.mcp_servers` から解決する経路を通す e2e で指定する）。
        """
        super().__init__()
        self._name = name
        self._fail_with = fail_with
        self._tools: list[MCPTool] = list(tools) if tools else []
        self.calls: list[tuple[str, Any]] = []

    @property
    def name(self) -> str:
        """サーバー名を返す。"""
        return self._name

    async def connect(self) -> None:
        """接続は行わない（no-op）。"""

    async def cleanup(self) -> None:
        """後始末は行わない（no-op）。"""

    async def list_tools(self, run_context: Any = None, agent: Any = None) -> list[MCPTool]:
        """コンストラクタで受けたツール一覧を返す（既定は空）。

        既定の空は `to_function_tool` を直接呼ぶトリップワイヤ向け。`tools` を渡した場合は
        SDK の run 時解決（`Agent.get_all_tools`）がこの一覧から `FunctionTool` を組む。
        """
        return list(self._tools)

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None, meta: dict[str, Any] | None = None
    ) -> CallToolResult:
        """呼び出しを記録し、`fail_with` があれば送出する。"""
        self.calls.append((tool_name, arguments))
        if self._fail_with is not None:
            raise self._fail_with
        return CallToolResult(content=[])

    async def list_prompts(self) -> ListPromptsResult:
        """プロンプト一覧は未対応。"""
        raise NotImplementedError

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> GetPromptResult:
        """プロンプト取得は未対応。"""
        raise NotImplementedError


def _mcp_tool(name: str) -> MCPTool:
    """MCP サーバーが公開するツール宣言（引数なしの最小 `inputSchema`）を作る。"""
    return MCPTool(name=name, inputSchema={"type": "object"})


def _real_mcp_function_tool(
    server: _StubMCPServer, *, tool_name: str = "read", name_override: str | None = None
) -> FunctionTool:
    """実 SDK の `MCPUtil.to_function_tool` で MCP 由来 `FunctionTool` を生成する。"""
    return MCPUtil.to_function_tool(
        _mcp_tool(tool_name),
        server,
        False,
        tool_name_override=name_override,
    )


def test_sdk_get_function_tool_origin_import_tripwire() -> None:
    """B1: `agents.tool.get_function_tool_origin` が存在し callable であることを固定する。

    本シンボルは `agents` トップレベルに export されていないため（`_adapters/governance.py` は
    サブモジュールから import している）、改名・移動が起きうる。消滅すれば import 時に落ちて
    気付けるが、**改名で別名が増えたのに旧名も残る**場合は静かに古い契約を見続けることに
    なるため、存在そのものを pin する。

    併せて `_adapters/governance.py` が束縛している実体が SDK の関数そのもの（同一オブジェクト）で
    あることも pin する。自前フォールバック実装や等価ラッパへの差し替えが混入すると、SDK 側の
    origin 判定ロジックの変更から静かに乖離し MCP の positive 判定が崩れるため identity で検知する。
    """
    origin_getter = getattr(sdk_tool, "get_function_tool_origin", None)
    assert callable(origin_getter), (
        "SDK の agents.tool.get_function_tool_origin が消滅 / 改名した。"
        "_adapters/governance.py の import と _AuditAgentHooks.on_tool_start の origin 判定を"
        "追従させること（MCP 由来ツールの positive 判定が成立しなくなる）。"
    )
    # `_adapters/governance.py` が束縛している実体が SDK の関数そのものであること
    # （自前フォールバック実装・別シンボルへの差し替えが混入したら検知する）。本テストファイル冒頭
    # の `from agents.tool import get_function_tool_origin`（モジュールグローバルの
    # `get_function_tool_origin`）と比較しても両辺が `agents.tool` の同一属性を指すため恒真になる。
    # よって governance モジュール側の束縛（`governance_module` 経由）を参照する。
    assert governance_module.get_function_tool_origin is sdk_tool.get_function_tool_origin, (
        "_adapters/governance.py が束縛する get_function_tool_origin が SDK の関数そのもので"
        "なくなった（自前フォールバック実装 / 別シンボルへの差し替えの混入）。"
        "MCP 由来ツールの origin 判定が SDK の実装から乖離するため追従させること。"
    )


def test_sdk_tool_origin_type_member_set_tripwire() -> None:
    """B2: `ToolOriginType` のメンバ値集合を固定する（新 origin 型追加時に判断を強制する）。

    `on_tool_start` は MCP のみを評価する positive 判定のため、新しい origin 型が増えても
    例外は出ず黙って非評価になる。評価対象へ含めるかの判断ポイントとして fail させる。
    """
    expected = {"function", "mcp", "agent_as_tool"}
    actual = {member.value for member in ToolOriginType}
    assert actual == expected, (
        "SDK ToolOriginType のメンバ集合が変化した。"
        "_adapters/governance.py の _AuditAgentHooks.on_tool_start の positive 判定"
        "（MCP のみ評価）へ新 origin を含めるか判断すること。"
        f" 差分: {sorted(actual.symmetric_difference(expected))}"
    )


def test_sdk_mcp_to_function_tool_attaches_mcp_origin_tripwire() -> None:
    """B3 / C1: 実 `MCPUtil.to_function_tool` の生成物が MCP origin と公開名を持つことを固定する。

    A 群のテストは `_tool_origin` を手で載せた偽装 tool を使うため、**実 SDK の MCP ツール生成
    経路を 1 本も通らない**。SDK が origin 付与をやめる / `_emit_tool_origin` の既定を反転すると、
    偽装ベースのテストは緑のまま本番の positive 判定が全 MCP ツールを素通しにする（fail-open
    なので例外もログも出ない・最悪の失敗モード）。ここでは `_AuditAgentHooks.on_tool_start` と
    同じ `get_function_tool_origin(tool)` 経由で assert し、既定反転も同時に検知する。

    併せて「SDK が解決した公開名が `FunctionTool.name` に載る」ことも pin する
    （`_evaluate_tool` へ渡す名前 = allowlist 照合対象の前提。`include_server_in_tool_names` の
    prefix 解決結果は `tool_name_override` として渡り、`FunctionTool.name` に反映される）。
    """
    server = _StubMCPServer(name="srv")
    tool = _real_mcp_function_tool(server)

    origin = get_function_tool_origin(tool)
    assert origin is not None, (
        "実 SDK の MCP ツールから origin が取得できない"
        "（_emit_tool_origin の既定が反転した可能性）。"
        "_adapters/governance.py の MCP positive 判定が全ツールを素通しにするため追従が必要。"
    )
    assert origin.type is ToolOriginType.MCP, (
        "実 SDK の MCP ツールに ToolOriginType.MCP が付与されなくなった。"
        "_adapters/governance.py の _AuditAgentHooks.on_tool_start による MCP 統治が"
        "無警告で全て素通しになるため、origin 判定の追従が必須。"
        f" 実際の origin: {origin!r}"
    )
    assert origin.mcp_server_name == "srv"
    # SDK が解決した公開名が FunctionTool.name に載る（allowlist 照合対象の前提）。
    assert tool.name == "read"
    prefixed = _real_mcp_function_tool(server, name_override="mcp_srv__read")
    assert prefixed.name == "mcp_srv__read", (
        "SDK が解決した公開名が FunctionTool.name に載らなくなった。"
        "allowlist 照合（_evaluate_tool へ渡す名前）の前提が崩れるため追従が必要。"
    )


async def test_real_sdk_mcp_function_tool_is_evaluated_by_audit_hooks() -> None:
    """C1: 実 SDK 生成の MCP `FunctionTool` が監査フックで評価される（deny / allow 両方向）。

    偽装 origin ではなく `MCPUtil.to_function_tool` の生成物を `on_tool_start` へ渡し、
    ポリシー評価が実際に走ることをエンドツーエンドで固定する。deny 側だけでは
    「常に deny」変異と区別できないため、同一ツールを allow するポリシーで素通ることも
    併せて確認する。deny 側では実 `PolicyViolationError` の構造化 payload（`details`）も
    同居して固定する（fake 例外ではなく実 AGT 例外で成立することの実証）。
    """
    server = _StubMCPServer(name="srv")
    tool = _real_mcp_function_tool(server)
    ctx = _tool_ctx("read", '{"path": "/etc/passwd"}')

    class _Named:
        def __init__(self, name: str) -> None:
            self.name = name

    deny_sink = AuditLog()
    deny_hooks = _make_audit_hooks(
        deny_sink,
        None,
        policy=GovernancePolicy(name="p", allowed_tools=["other"]),
        denied_exc=PolicyViolationError,
        agent_name="bot",
    )
    with pytest.raises(PolicyViolationError, match="read") as excinfo:
        await deny_hooks.on_tool_start(ctx, _Named("bot"), tool)
    # 実 `PolicyViolationError` 上で拒否 payload が成立することを併せて固定する。`reason` は
    # AGT 生成の人間可読文言（機械判別のキーではない）ため値をリテラル固定せず、非空であること
    # と送出メッセージへ載っていることの 2 点で見る（`in` だけでは空文字が常に真で素通りする）。
    assert set(excinfo.value.details) == {"tool_name", "reason"}, (
        f"拒否例外の payload キー集合が変わった: {excinfo.value.details!r}"
    )
    assert excinfo.value.details["tool_name"] == "read"
    assert excinfo.value.details["reason"]
    assert excinfo.value.details["reason"] in str(excinfo.value)
    assert excinfo.value.error_code == "POLICY_VIOLATION"
    # 本経路は `from_check_result` を通らないため `check_result` は None のまま（ADR-0030 の
    # Consequences。由来の判別に `check_result` を使う利用側コードが従来どおり動く前提）。
    assert excinfo.value.check_result is None
    assert [(e.action, e.decision) for e in deny_sink.get_entries()] == [
        ("tool_start:read", "allow"),
        ("tool:read", "deny"),
    ]
    assert server.calls == []  # 実 MCP 呼び出しは発生しない

    allow_sink = AuditLog()
    allow_hooks = _make_audit_hooks(
        allow_sink,
        None,
        policy=GovernancePolicy(name="p", allowed_tools=["read"]),
        denied_exc=PolicyViolationError,
        agent_name="bot",
    )
    await allow_hooks.on_tool_start(ctx, _Named("bot"), tool)
    assert [(e.action, e.decision) for e in allow_sink.get_entries()] == [
        ("tool_start:read", "allow"),
        ("tool:read", "allow"),
    ]
    allow_entry = allow_sink.get_entries()[1]
    assert allow_entry.details == {"arguments": '{"path": "/etc/passwd"}'}


async def test_sdk_tool_context_carries_str_arguments_tripwire() -> None:
    """B4: `on_tool_start` に渡る context が `tool_arguments: str` を持つことを固定する。

    型が変わると `_AuditAgentHooks.on_tool_start` の fail-closed 分岐が常時発火して
    全 MCP 呼び出しが deny になる（アプリ停止）。逆に空文字へ化けると引数照合
    （blocked_patterns）が黙って効かなくなる。実 Runner + FakeModel で駆動し、
    合成チェーン後段の既存フックで context を捕獲して pin する。
    """
    captured: list[Any] = []

    class _CapturingHooks:
        """`on_tool_start` の context を捕獲する既存 `spec.hooks` 相当。"""

        async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
            captured.append(context)

    invoked: list[str] = []
    sink = AuditLog()
    policy = GovernancePolicy(name="p", allowed_tools=["mcp_read"])
    reg = AgentRegistry(agent_builder=GovernedAgentBuilder(policy=policy, audit_sink=sink))
    model = FakeModel().queue_tool_call("mcp_read", '{"q": "x"}').queue_text("done")
    reg.register(AgentSpec(name="bot", instructions="i", model=model, hooks=_CapturingHooks()))
    agent = reg.get("bot")
    # MCP ツールは run 時に SDK が注入するため build 後注入で経路を再現する。
    agent.tools.append(_mcp_origin_tool(invoked))

    await Runner.run(agent, input="go")

    assert len(captured) == 1
    arguments = getattr(captured[0], "tool_arguments", None)
    assert isinstance(arguments, str), (
        "SDK が on_tool_start の context に str の tool_arguments を渡さなくなった。"
        "_adapters/governance.py の fail-closed 分岐が常時発火し全 MCP 呼び出しが deny になる。"
        f" 実際の型: {type(arguments).__name__}"
    )
    assert arguments == '{"q": "x"}', (
        "SDK が渡す tool_arguments がモデル出力の引数 JSON と一致しない。"
        "blocked_patterns の引数照合が黙って無効化されるため追従が必要。"
    )
    # 引数が評価・監査へそのまま渡っている（allow 記録の details で観測する）。
    allow = next(e for e in sink.get_entries() if e.action == "tool:mcp_read")
    assert allow.details == {"arguments": '{"q": "x"}'}
    assert invoked == ['{"q": "x"}']  # allow なので実ツール本体まで到達する


async def test_sdk_mcp_function_tool_wraps_errors_into_result_tripwire() -> None:
    """B5: MCP 由来 `FunctionTool` の `on_invoke_tool` が内部例外を文字列化して返すことを固定する。

    SDK 既定の `failure_error_function` が効くため、MCP ツールの実行時例外は送出されず
    モデル向けエラー文字列として返る（`agents/tool.py` の
    `_FailureHandlingFunctionToolInvoker.__call__`）。この性質があるため「`on_invoke_tool` の
    内側にポリシー評価を置く」案は成立せず（deny 例外が文字列へ吸われて統治が無効化される）、
    `on_tool_start` を選んだ設計判断の前提になっている。性質が消えたら設計の再検討が必要。
    """
    server = _StubMCPServer(name="srv", fail_with=RuntimeError("stub mcp failure"))
    tool = _real_mcp_function_tool(server)

    result = await tool.on_invoke_tool(_tool_ctx("read", "{}"), "{}")

    assert isinstance(result, str), (
        "MCP 由来 FunctionTool の on_invoke_tool が内部例外を文字列化して返さなくなった。"
        "on_tool_start でポリシー評価する設計判断（deny 例外が文字列へ吸われないため）の"
        "前提が変わるため、_adapters/governance.py の評価位置を再検討すること。"
        f" 実際の型: {type(result).__name__}"
    )
    assert "stub mcp failure" in result
    assert server.calls == [("read", {})]  # 実 MCP 呼び出しは 1 回だけ走った


# ----------------------------------------------------------------------
# D 群: 宣言 `spec.mcp_servers` -> SDK の run 時解決 -> run loop の `on_tool_start` dispatch
#
# 他の MCP テストは build 後の `agent.tools.append` 注入か `on_tool_start` の直接呼び出しで、
# 宣言から run 時解決までの結合部を 1 本も通らない。SDK が MCP ツールを専用 dispatch へ移す
# （`on_tool_start` を発火させない）退行が起きても、fail-open のため全テスト緑のまま統治だけが
# 消えるため、実 Runner で結合部ごと固定する。
# ----------------------------------------------------------------------


def _mcp_server_with_read(name: str = "srv") -> _StubMCPServer:
    """`read` ツール 1 本を公開するスタブ MCP サーバーを作る（run 時解決の入力）。"""
    return _StubMCPServer(name=name, tools=[_mcp_tool("read")])


def _governed_registry(policy: GovernancePolicy, sink: AuditLog) -> AgentRegistry:
    """`GovernedAgentBuilder` を差した registry を作る。"""
    return AgentRegistry(agent_builder=GovernedAgentBuilder(policy=policy, audit_sink=sink))


async def test_declared_mcp_server_tool_deny_end_to_end() -> None:
    """D1: 宣言 `mcp_servers` の run 時解決ツールを deny し、MCP 呼び出しごと止める。

    `spec.mcp_servers` -> SDK の run 時解決 -> run loop の `on_tool_start` dispatch という結合部
    を実 Runner で通す（build 後注入もフック直呼びもしない）。deny は `UserError` として run を
    終了させ（`__cause__` に `PolicyViolationError`）、MCP サーバーの `call_tool` へは 1 度も
    到達しないこと・監査列が厳密に `agent_start` / `tool_start` / `tool` deny の 3 件であることを
    固定する（`tool_end` / `agent_end` が続かない = run が継続していない証跡）。
    """
    server = _mcp_server_with_read()
    sink = AuditLog()
    reg = _governed_registry(GovernancePolicy(name="p", allowed_tools=["nothing"]), sink)
    model = FakeModel().queue_tool_call("read", '{"q": "x"}').queue_text("unreached")
    reg.register(AgentSpec(name="bot", instructions="i", model=model, mcp_servers=[server]))
    agent = reg.get("bot")

    with pytest.raises(UserError) as excinfo:
        await Runner.run(agent, input="go")

    assert isinstance(excinfo.value.__cause__, PolicyViolationError), (
        "宣言 mcp_servers 経路の deny が PolicyViolationError を原因として着地しない。"
        f" 実際の __cause__: {excinfo.value.__cause__!r}"
    )
    assert server.calls == [], "deny なのに MCP サーバーの call_tool へ到達した（統治の抜け）。"
    assert [(e.action, e.decision) for e in sink.get_entries()] == [
        ("agent_start", "allow"),
        ("tool_start:read", "allow"),
        ("tool:read", "deny"),
    ], (
        "宣言 mcp_servers 経路で on_tool_start が発火しなくなった可能性がある"
        "（SDK が MCP ツールを専用 dispatch へ移すと fail-open で統治が消える）。"
    )


async def test_declared_mcp_server_tool_allow_end_to_end() -> None:
    """D2: 宣言 `mcp_servers` の run 時解決ツールを allow し、MCP サーバーまで到達させる。

    D1（deny）だけでは「常に deny」変異と区別できないため、同一経路で allow が素通り、
    `call_tool` へツール名と引数の両方が渡ることを固定する。監査列は allow 経路の 5 件
    （`agent_start` / `tool_start` / `tool` / `tool_end` / `agent_end`）を厳密比較する。
    """
    server = _mcp_server_with_read()
    sink = AuditLog()
    reg = _governed_registry(GovernancePolicy(name="p", allowed_tools=["read"]), sink)
    model = FakeModel().queue_tool_call("read", '{"q": "x"}').queue_text("done")
    reg.register(AgentSpec(name="bot", instructions="i", model=model, mcp_servers=[server]))
    agent = reg.get("bot")

    await Runner.run(agent, input="go")

    assert server.calls == [("read", {"q": "x"})], (
        "allow なのに MCP サーバーへツール名 / 引数がそのまま渡っていない。"
        f" 実際の呼び出し: {server.calls!r}"
    )
    assert [(e.action, e.decision) for e in sink.get_entries()] == [
        ("agent_start", "allow"),
        ("tool_start:read", "allow"),
        ("tool:read", "allow"),
        ("tool_end:read", "allow"),
        ("agent_end", "allow"),
    ]


async def test_declared_mcp_server_tool_name_prefixed_by_sdk_end_to_end() -> None:
    """D3: `include_server_in_tool_names` の照合名を SDK に生成させて両方向を固定する。

    既存トリップワイヤは `tool_name_override` をテスト側から渡しており SDK の prefix 生成
    （`agents/mcp/util.py` の private ヘルパ）を 1 度も通らない。ここでは
    `mcp_config={"include_server_in_tool_names": True}` を宣言して SDK に名前を生成させ、
    prefix 付き名（`mcp_srv__read`）で allow・prefix 前の名前（`read`）で deny という 2 方向を
    pin する（片方だけでは prefix が付いていること自体を固定できない）。SDK 側で形式が変われば
    利用者の `allowed_tools` が全不一致 = 全 deny（機能停止）になるため検知が必須。

    本テストが固定するのは `mcp_{サーバ名}__{ツール名}`（base 名）がそのまま公開名になる**単純
    分岐**、すなわち base 名が ASCII 英数字 / `_` / `-` のみで構成され、SDK の長さ上限以内で、
    同一解決バッチ内の他ツール名や `spec.tools` の名前と衝突しない場合の形式である。それ以外の
    場合は SDK が置換・切り詰め・ハッシュ付与を行う（変形分岐は D4 で別途 pin する）。
    """
    allow_server = _mcp_server_with_read()
    allow_sink = AuditLog()
    allow_reg = _governed_registry(
        GovernancePolicy(name="p", allowed_tools=["mcp_srv__read"]), allow_sink
    )
    allow_model = FakeModel().queue_tool_call("mcp_srv__read", '{"q": "x"}').queue_text("done")
    allow_reg.register(
        AgentSpec(
            name="bot",
            instructions="i",
            model=allow_model,
            mcp_servers=[allow_server],
            mcp_config={"include_server_in_tool_names": True},
        )
    )

    await Runner.run(allow_reg.get("bot"), input="go")

    assert [tool.name for tool in allow_model.calls[0].tools] == ["mcp_srv__read"], (
        "単純分岐（ASCII 英数字のみ・長さ上限以内・非衝突）でも SDK が生成する MCP ツールの"
        "公開名が mcp_{サーバ名}__{ツール名} でなくなった。"
        "allowed_tools の宣言形式に関する記述（spec.py / _adapters/governance.py / docs）が"
        "全て不一致になり全 deny へ化けるため追従が必須。"
        f" 実際の名前: {[tool.name for tool in allow_model.calls[0].tools]!r}"
    )
    assert [(e.action, e.decision) for e in allow_sink.get_entries()] == [
        ("agent_start", "allow"),
        ("tool_start:mcp_srv__read", "allow"),
        ("tool:mcp_srv__read", "allow"),
        ("tool_end:mcp_srv__read", "allow"),
        ("agent_end", "allow"),
    ]
    # MCP サーバーへ渡るのは prefix 前のツール名（prefix は SDK 側の公開名だけに載る）。
    assert allow_server.calls == [("read", {"q": "x"})]

    deny_server = _mcp_server_with_read()
    deny_sink = AuditLog()
    deny_reg = _governed_registry(GovernancePolicy(name="p", allowed_tools=["read"]), deny_sink)
    deny_model = FakeModel().queue_tool_call("mcp_srv__read", '{"q": "x"}').queue_text("unreached")
    deny_reg.register(
        AgentSpec(
            name="bot",
            instructions="i",
            model=deny_model,
            mcp_servers=[deny_server],
            mcp_config={"include_server_in_tool_names": True},
        )
    )

    with pytest.raises(UserError) as excinfo:
        await Runner.run(deny_reg.get("bot"), input="go")

    assert isinstance(excinfo.value.__cause__, PolicyViolationError)
    assert deny_server.calls == []
    assert [(e.action, e.decision) for e in deny_sink.get_entries()] == [
        ("agent_start", "allow"),
        ("tool_start:mcp_srv__read", "allow"),
        ("tool:mcp_srv__read", "deny"),
    ], (
        "prefix 前の名前（read）が allowlist にあるだけで許可された"
        "（照合対象が SDK 解決後の公開名でなくなった可能性）。"
    )


async def test_declared_mcp_server_tool_name_transformed_by_sdk_end_to_end() -> None:
    """D4: SDK が公開名を変形する分岐（非英数字置換 / 長さ超過ハッシュ）を宣言経路で固定する。

    D3 が固定するのは base 名（`mcp_{サーバ名}__{ツール名}`）がそのまま公開名になる単純分岐だけ。
    SDK は `include_server_in_tool_names` の名前解決で (1) ASCII 英数字 / `_` / `-` 以外を `_` へ
    置換し前後の `_-` を strip し、(2) base 名が長さ上限を超える場合は切り詰めて sha1 先頭 8 桁を
    付ける（同一解決バッチ内での base 名重複・`spec.tools` の名前との衝突では短い名前でもハッシュ
    が付く）。この分岐に入ると `allowed_tools=["mcp_<サーバ名>__<ツール名>"]` は全不一致になり当該
    MCP ツールが常時 deny になる（fail-closed なので安全側だが機能停止）。既存テストは緑のままな
    ので、変形が起きること自体をトリップワイヤ化する。

    private ヘルパ（`_build_prefixed_tool_base_name` 等）は直接呼ばず、宣言 `mcp_servers` から SDK
    の run 時解決へ至る公開経路で pin する。ハッシュ値そのものは seed 構成の変更で無意味に赤くなる
    ため固定せず、「base 名と異なる」「長さ上限以下」「`_` + 16 進 8 桁で終わる」の 3 点で固定する。
    """
    # SDK の `_MCP_FUNCTION_TOOL_NAME_MAX_LENGTH`（private 定数のため参照せず値を持つ）。
    max_length = 64

    # (1) 非英数字置換: サーバ名の `.` が `_` へ置換され、その公開名が照合対象になる。
    dotted_server = _mcp_server_with_read("my.srv")
    dotted_sink = AuditLog()
    dotted_reg = _governed_registry(
        GovernancePolicy(name="p", allowed_tools=["mcp_my_srv__read"]), dotted_sink
    )
    dotted_model = FakeModel().queue_tool_call("mcp_my_srv__read", '{"q": "x"}').queue_text("done")
    dotted_reg.register(
        AgentSpec(
            name="bot",
            instructions="i",
            model=dotted_model,
            mcp_servers=[dotted_server],
            mcp_config={"include_server_in_tool_names": True},
        )
    )

    await Runner.run(dotted_reg.get("bot"), input="go")

    assert [tool.name for tool in dotted_model.calls[0].tools] == ["mcp_my_srv__read"], (
        "SDK の公開名生成が変わった（ASCII 英数字 / _ / - 以外を _ へ置換しなくなった）。"
        "allowed_tools の宣言形式に関する docstring / docs の記述"
        "（src/oai_agentspec/spec.py / _adapters/governance.py /"
        " docs/usage/safety/governance.md 等）を追随させること。"
        f" 実際の名前: {[tool.name for tool in dotted_model.calls[0].tools]!r}"
    )
    # 置換後の公開名で allow 照合が成立し、MCP サーバーへは prefix 前の名前が渡る。
    assert dotted_server.calls == [("read", {"q": "x"})]
    assert [(e.action, e.decision) for e in dotted_sink.get_entries()] == [
        ("agent_start", "allow"),
        ("tool_start:mcp_my_srv__read", "allow"),
        ("tool:mcp_my_srv__read", "allow"),
        ("tool_end:mcp_my_srv__read", "allow"),
        ("agent_end", "allow"),
    ]

    # (2) 長さ超過: base 名が上限超のとき公開名は切り詰め + ハッシュ付きへ変形される。
    long_server_name = "my-very-long-mcp-server-name-for-testing"
    long_tool_name = "read_file_with_an_extremely_long_tool_name_for_testing"
    base_name = f"mcp_{long_server_name}__{long_tool_name}"
    assert len(base_name) > max_length  # 前提: 上限超過分岐へ入る base 名であること
    long_server = _StubMCPServer(name=long_server_name, tools=[_mcp_tool(long_tool_name)])
    long_reg = _governed_registry(GovernancePolicy(name="p", allowed_tools=["nothing"]), AuditLog())
    long_model = FakeModel().queue_text("done")
    long_reg.register(
        AgentSpec(
            name="bot",
            instructions="i",
            model=long_model,
            mcp_servers=[long_server],
            mcp_config={"include_server_in_tool_names": True},
        )
    )

    await Runner.run(long_reg.get("bot"), input="go")

    resolved = [tool.name for tool in long_model.calls[0].tools]
    assert len(resolved) == 1
    public_name = resolved[0]
    follow_up = (
        "SDK の公開名生成が変わった。allowed_tools の宣言形式に関する docstring / docs の記述"
        "（src/oai_agentspec/spec.py / _adapters/governance.py /"
        " docs/usage/safety/governance.md 等）を追随させること。"
    )
    assert public_name != base_name, (
        f"{follow_up} 長さ上限超の base 名が変形されず公開名になっている: {public_name!r}"
    )
    assert len(public_name) <= max_length, (
        f"{follow_up} 公開名が長さ上限（{max_length}）へ切り詰められていない:"
        f" {public_name!r}（{len(public_name)} 文字）"
    )
    assert re.search(r"_[0-9a-f]{8}\Z", public_name) is not None, (
        f"{follow_up} 公開名の末尾がハッシュ（_ + 16 進 8 桁）でない: {public_name!r}"
    )


async def test_agent_hooks_replacement_drops_mcp_enforcement_not_spec_tools() -> None:
    """D5: build 後に `Agent.hooks` を差し替えると MCP 経路の強制だけが失われる（境界 (12)）。

    `Agent.hooks` の差し替え（`clone(hooks=...)` 含む）は SDK の公開 API なので利用者が到達しうる。
    MCP 由来ツールの強制はフック（`on_tool_start`）にしか無いため差し替えで消え、`spec.tools` は
    実行本体のラップが tool オブジェクト自身へ焼き込まれるため残る。この非対称は意図したもので、
    利用者が差し替えたときの挙動を固定するためにここで pin する（差し替えを推奨するものではない。
    フックを足したい場合は `spec.hooks` へ宣言して builder に合成させる）。両方向を 1 本で固定する
    （片方だけでは非対称そのものを固定できない）。
    """
    # (a) MCP 経路: 差し替えで強制が消え、deny すべき呼び出しが MCP サーバーへ到達する。
    mcp_server = _mcp_server_with_read()
    mcp_sink = AuditLog()
    mcp_reg = _governed_registry(GovernancePolicy(name="p", allowed_tools=["nothing"]), mcp_sink)
    mcp_model = FakeModel().queue_tool_call("read", '{"q": "x"}').queue_text("done")
    mcp_reg.register(
        AgentSpec(name="bot", instructions="i", model=mcp_model, mcp_servers=[mcp_server])
    )
    mcp_agent = mcp_reg.get("bot")
    mcp_agent.hooks = None  # 利用者による差し替え（監査フックの合成チェーンごと捨てる）

    await Runner.run(mcp_agent, input="go")

    assert mcp_server.calls == [("read", {"q": "x"})], (
        "hooks 差し替え後も MCP 経路の強制が残っている（境界 (12) の前提が変わった）。"
        "_adapters/governance.py の govern_spec 境界 (12) と"
        " runtime/governance/builder.py の記述を追随させること。"
        f" 実際の呼び出し: {mcp_server.calls!r}"
    )
    assert mcp_sink.get_entries() == []  # フック由来の記録も一切残らない

    # (b) `spec.tools` 経路: 同一の差し替えでも build 時ラップによる deny と `tool:` 記録は残る。
    invoked: list[str] = []
    tool_sink = AuditLog()
    tool_reg = _governed_registry(GovernancePolicy(name="p", allowed_tools=["nothing"]), tool_sink)
    tool_model = FakeModel().queue_tool_call("echo", '{"text": "nope"}').queue_text("unreached")
    tool_reg.register(
        AgentSpec(
            name="bot", instructions="i", model=tool_model, tools=[_make_tool(invoked, "echo")]
        )
    )
    tool_agent = tool_reg.get("bot")
    tool_agent.hooks = None

    with pytest.raises(UserError) as excinfo:
        await Runner.run(tool_agent, input="go")

    assert isinstance(excinfo.value.__cause__, PolicyViolationError), (
        "hooks 差し替えで spec.tools 経路の強制まで失われた"
        "（build 時ラップが tool 自身へ焼き込まれる前提が壊れた）。"
        f" 実際の __cause__: {excinfo.value.__cause__!r}"
    )
    # docs が案内する取得形（`exc.__cause__.details`）が実 `Runner.run` 経由の公開経路で
    # 成立することの実証（`__cause__` は参照なので payload がラップで失われる経路は無い）。
    cause = excinfo.value.__cause__
    assert set(cause.details) == {"tool_name", "reason"}, (
        f"公開経路で辿れる拒否 payload のキー集合が変わった: {cause.details!r}"
    )
    assert cause.details["tool_name"] == "echo"
    assert invoked == []  # 実関数へは到達しない
    # per-call の `tool:` レコードはラップ内で記録されるため残る（消えるのはフック由来の記録のみ）。
    assert [(e.agent_id, e.action, e.decision) for e in tool_sink.get_entries()] == [
        ("bot", "tool:echo", "deny"),
    ]


# ----------------------------------------------------------------------
# registry 第 3 段 post-process: 既定不変の pin と sub_agents / factory のオプトイン統治
#
# 1 つの registry に sub agent（`researcher`）・親（`support`: `spec.tools` の関数ツール +
# 利用者が直接置いた as_tool + wire 注入の `researcher` as_tool + 部分実装 hooks）・factory
# （`legacy`）を載せ、FakeModel で各ツールを 1 回ずつ呼ばせる。
# ----------------------------------------------------------------------

_MIXED_ALLOWED = ["fn_r", "fn_s", "fn_l", "direct_tool", "researcher"]


class _SinkObservingHooks:
    """利用者の `spec.hooks`（duck-typed 部分実装）。到達時の sink 末尾 action を併記する。

    `on_start` / `on_end` / `on_tool_start` / `on_tool_end` のみを持つ（`on_handoff` /
    `on_llm_*` を持たない部分実装）。到達時点の sink 末尾を併記することで、「監査記録が先・
    利用者フックへの委譲が後」の順を記録列として観測する。
    """

    def __init__(self, sink: AuditLog) -> None:
        """観測対象の sink と記録リストを初期化する。"""
        self._sink = sink
        self.events: list[tuple[str, str | None]] = []

    def _last_action(self) -> str | None:
        entries = self._sink.get_entries()
        return entries[-1].action if entries else None

    async def on_start(self, context: Any, agent: Any) -> None:
        self.events.append((f"start:{agent.name}", self._last_action()))

    async def on_end(self, context: Any, agent: Any, output: Any) -> None:
        self.events.append((f"end:{agent.name}", self._last_action()))

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        self.events.append((f"tool_start:{tool.name}", self._last_action()))

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        self.events.append((f"tool_end:{tool.name}", self._last_action()))


@dataclasses.dataclass
class _MixedRegistry:
    """`_mixed_registry` が組んだ registry と観測点の束。"""

    registry: AgentRegistry
    sink: AuditLog
    hooks: _SinkObservingHooks
    calls: dict[str, list[str]]
    support_model: FakeModel
    researcher_model: FakeModel
    direct_model: FakeModel
    legacy_model: FakeModel
    legacy_original: list[Agent]


def _mixed_registry(
    *,
    allowed_tools: list[str],
    sub_agent_tools: bool | None = None,
    factory_agents: bool | None = None,
    support_tool_calls: list[tuple[str, str]] | None = None,
) -> _MixedRegistry:
    """sub agent / 親 / factory を 1 つの registry に載せる。

    `sub_agent_tools` / `factory_agents` が両方 None なら registry へ `post_processor` を渡さない
    （既定経路の pin 用）。片方だけ指定した場合はもう片方を False として
    `builder.post_processor(...)` を作り、`AgentRegistry(post_processor=...)` へ渡す。

    Args:
        allowed_tools: 既定ポリシーの `allowed_tools`。
        sub_agent_tools: `post_processor(sub_agent_tools=...)`（None は False 扱い）。
        factory_agents: `post_processor(factory_agents=...)`（None は False 扱い）。
        support_tool_calls: `support` の FakeModel に順に積む (tool 名, 引数 JSON)。None なら
            `fn_s` → `direct_tool` → `researcher` の 3 件。
    """
    sink = AuditLog()
    builder = GovernedAgentBuilder(
        policy=GovernancePolicy(name="p", allowed_tools=allowed_tools), audit_sink=sink
    )
    if sub_agent_tools is None and factory_agents is None:
        registry = AgentRegistry(agent_builder=builder)
    else:
        registry = AgentRegistry(
            agent_builder=builder,
            post_processor=builder.post_processor(
                sub_agent_tools=bool(sub_agent_tools), factory_agents=bool(factory_agents)
            ),
        )
    calls: dict[str, list[str]] = {"fn_r": [], "fn_s": [], "fn_l": []}
    hooks = _SinkObservingHooks(sink)

    researcher_model = FakeModel().queue_text("researched")
    registry.register(
        AgentSpec(
            name="researcher",
            instructions="r",
            model=researcher_model,
            tools=[_make_tool(calls["fn_r"], name="fn_r")],
        )
    )
    direct_model = FakeModel().queue_text("direct-done")
    direct_as_tool = Agent(name="direct", instructions="d", model=direct_model).as_tool(
        tool_name="direct_tool", tool_description="direct"
    )
    support_model = FakeModel()
    for name, arguments in support_tool_calls or [
        ("fn_s", '{"text": "s"}'),
        ("direct_tool", '{"input": "d"}'),
        ("researcher", '{"input": "r"}'),
    ]:
        support_model.queue_tool_call(name, arguments)
    support_model.queue_text("done")
    registry.register(
        AgentSpec(
            name="support",
            instructions="s",
            model=support_model,
            tools=[_make_tool(calls["fn_s"], name="fn_s"), direct_as_tool],
            sub_agents=["researcher"],
            hooks=hooks,
        )
    )
    legacy_model = FakeModel().queue_tool_call("fn_l", '{"text": "l"}').queue_text("l-done")
    legacy_original: list[Agent] = []
    fn_l = _make_tool(calls["fn_l"], name="fn_l")

    def _legacy_factory(_registry: AgentRegistry) -> Agent:
        agent = Agent(name="legacy", instructions="x", tools=[fn_l], model=legacy_model)
        legacy_original.append(agent)
        return agent

    registry.register_factory("legacy", _legacy_factory)
    return _MixedRegistry(
        registry=registry,
        sink=sink,
        hooks=hooks,
        calls=calls,
        support_model=support_model,
        researcher_model=researcher_model,
        direct_model=direct_model,
        legacy_model=legacy_model,
        legacy_original=legacy_original,
    )


def _record_shape(sink: AuditLog) -> list[tuple[str, str, str, list[str]]]:
    """監査レコード列を `(agent_id, action, decision, sorted(details のキー))` へ写す。"""
    return [
        (e.agent_id, e.action, e.decision, sorted((e.details or {}).keys()))
        for e in sink.get_entries()
    ]


async def test_audit_record_sequence_unchanged_by_default() -> None:
    """`post_processor` を渡さない registry は対応前と同一の監査レコード列・identity・委譲順を保つ。

    `spec.tools` の関数ツールと利用者が直接置いた as_tool は `tool:` レコードを持ち（統治）、
    wire 注入の `researcher` as_tool は `tool_start:` / `tool_end:` のみ（非統治）、factory
    `legacy` は統治されず `registry.get` が factory の戻り値そのものを返す。期待列は対応前の
    実装（build 時統治）で実測した列であり、`post_processor` 引数の追加後もこの列が変わらない
    ことが既定不変の証拠になる。部分実装の `spec.hooks` へは各イベントが監査記録の後に届く。
    """
    mixed = _mixed_registry(allowed_tools=_MIXED_ALLOWED)

    support = mixed.registry.get("support")
    result = await Runner.run(support, input="go")
    legacy = mixed.registry.get("legacy")
    legacy_result = await Runner.run(legacy, input="go")

    assert result.final_output == "done"
    assert legacy_result.final_output == "l-done"
    # 各ツールの実体が 1 回ずつ走った（as_tool はサブ側 model が 1 回呼ばれる）。
    assert mixed.calls == {"fn_r": [], "fn_s": ["s"], "fn_l": ["l"]}
    assert len(mixed.researcher_model.calls) == 1
    assert len(mixed.direct_model.calls) == 1
    # factory の戻り値がそのまま返る（clone されない）。
    assert len(mixed.legacy_original) == 1
    assert legacy is mixed.legacy_original[0]
    # 対応前の実装で実測した列（推測で書き換えないこと）。
    assert _record_shape(mixed.sink) == [
        ("support", "agent_start", "allow", []),
        ("support", "tool_start:fn_s", "allow", []),
        ("support", "tool:fn_s", "allow", ["arguments"]),
        ("support", "tool_end:fn_s", "allow", []),
        ("support", "tool_start:direct_tool", "allow", []),
        ("support", "tool:direct_tool", "allow", ["arguments"]),
        ("support", "tool_end:direct_tool", "allow", []),
        ("support", "tool_start:researcher", "allow", []),
        ("researcher", "agent_start", "allow", []),
        ("researcher", "agent_end", "allow", []),
        ("support", "tool_end:researcher", "allow", []),
        ("support", "agent_end", "allow", []),
    ]
    # 利用者フックへは監査記録の直後に届く（到達時の sink 末尾が同イベントの監査記録）。
    assert mixed.hooks.events == [
        ("start:support", "agent_start"),
        ("tool_start:fn_s", "tool_start:fn_s"),
        ("tool_end:fn_s", "tool_end:fn_s"),
        ("tool_start:direct_tool", "tool_start:direct_tool"),
        ("tool_end:direct_tool", "tool_end:direct_tool"),
        ("tool_start:researcher", "tool_start:researcher"),
        ("tool_end:researcher", "tool_end:researcher"),
        ("end:support", "agent_end"),
    ]
    assert mixed.sink.verify_chain() is True


def _tool_records(sink: AuditLog, action: str) -> list[Any]:
    """sink から `action` が一致するレコードだけを取り出す。"""
    return [e for e in sink.get_entries() if e.action == action]


async def test_opt_in_sub_agent_as_tool_allowed_and_recorded() -> None:
    """オプトイン時、wire 注入の sub_agents as_tool は親のポリシーで allow 評価され監査に残る。"""
    mixed = _mixed_registry(allowed_tools=_MIXED_ALLOWED, sub_agent_tools=True, factory_agents=True)

    result = await Runner.run(mixed.registry.get("support"), input="go")

    assert result.final_output == "done"
    assert len(mixed.researcher_model.calls) == 1  # allow なのでサブエージェントが走る
    records = _tool_records(mixed.sink, "tool:researcher")
    assert [(e.agent_id, e.decision, e.details) for e in records] == [
        ("support", "allow", {"arguments": '{"input": "r"}'})
    ]
    assert mixed.sink.verify_chain() is True


async def test_opt_in_sub_agent_as_tool_denied_stops_run() -> None:
    """オプトイン時、ポリシー外の sub_agents as_tool は deny され、サブエージェントは走らない。"""
    mixed = _mixed_registry(
        allowed_tools=[n for n in _MIXED_ALLOWED if n != "researcher"],
        sub_agent_tools=True,
        factory_agents=True,
        support_tool_calls=[("researcher", '{"input": "r"}')],
    )

    with pytest.raises(UserError) as excinfo:
        await Runner.run(mixed.registry.get("support"), input="go")

    assert isinstance(excinfo.value.__cause__, PolicyViolationError)
    assert mixed.researcher_model.calls == []  # サブエージェントの run は起きない
    records = _tool_records(mixed.sink, "tool:researcher")
    assert [(e.agent_id, e.decision) for e in records] == [("support", "deny")]
    assert records[0].details["arguments"] == '{"input": "r"}'
    assert "researcher" in records[0].details["reason"]


async def test_opt_in_factory_agent_governed_as_clone_allow_and_deny() -> None:
    """オプトイン時、factory Agent の tools も allow / deny され、get は clone を返し元は不変。"""
    allowed = _mixed_registry(
        allowed_tools=_MIXED_ALLOWED, sub_agent_tools=True, factory_agents=True
    )

    legacy = allowed.registry.get("legacy")
    result = await Runner.run(legacy, input="go")

    assert result.final_output == "l-done"
    assert allowed.calls["fn_l"] == ["l"]
    records = _tool_records(allowed.sink, "tool:fn_l")
    assert [(e.agent_id, e.decision, e.details) for e in records] == [
        ("legacy", "allow", {"arguments": '{"text": "l"}'})
    ]
    # factory の戻り値とは別オブジェクト（clone）が返り、元の tools / hooks は変更されない。
    assert len(allowed.legacy_original) == 1
    original = allowed.legacy_original[0]
    assert legacy is not original
    assert len(original.tools) == 1
    assert original.tools[0] is not legacy.tools[0]
    assert original.hooks is None
    assert legacy.hooks is not None

    denied = _mixed_registry(
        allowed_tools=[n for n in _MIXED_ALLOWED if n != "fn_l"],
        sub_agent_tools=True,
        factory_agents=True,
    )

    with pytest.raises(UserError) as excinfo:
        await Runner.run(denied.registry.get("legacy"), input="go")

    assert isinstance(excinfo.value.__cause__, PolicyViolationError)
    assert denied.calls["fn_l"] == []  # 実関数は非実行
    records = _tool_records(denied.sink, "tool:fn_l")
    assert [(e.agent_id, e.decision) for e in records] == [("legacy", "deny")]
    assert records[0].details["arguments"] == '{"text": "l"}'


async def test_opt_in_sub_agent_tools_only_leaves_factory_ungoverned() -> None:
    """`post_processor(sub_agent_tools=True)` だけでは factory 経路は従来どおり非統治のまま。"""
    mixed = _mixed_registry(
        allowed_tools=[n for n in _MIXED_ALLOWED if n != "fn_l"],  # fn_l はポリシー外
        sub_agent_tools=True,
    )

    await Runner.run(mixed.registry.get("support"), input="go")
    legacy = mixed.registry.get("legacy")
    await Runner.run(legacy, input="go")

    # sub_agents 経路は統治される。
    assert [e.decision for e in _tool_records(mixed.sink, "tool:researcher")] == ["allow"]
    # factory 経路は同一オブジェクト・ポリシー外の fn_l も評価されず実行される。
    assert legacy is mixed.legacy_original[0]
    assert mixed.calls["fn_l"] == ["l"]
    assert _tool_records(mixed.sink, "tool:fn_l") == []


async def test_opt_in_factory_agents_only_leaves_sub_agent_tools_ungoverned() -> None:
    """`post_processor(factory_agents=True)` だけでは wire 注入の as_tool は従来どおり非統治。"""
    mixed = _mixed_registry(
        allowed_tools=[n for n in _MIXED_ALLOWED if n != "researcher"],  # researcher はポリシー外
        factory_agents=True,
    )

    result = await Runner.run(mixed.registry.get("support"), input="go")
    legacy = mixed.registry.get("legacy")
    await Runner.run(legacy, input="go")

    # sub_agents の as_tool は評価されずサブエージェントまで走る。
    assert result.final_output == "done"
    assert len(mixed.researcher_model.calls) == 1
    assert _tool_records(mixed.sink, "tool:researcher") == []
    # factory 経路は clone されて統治される。
    assert legacy is not mixed.legacy_original[0]
    assert [e.decision for e in _tool_records(mixed.sink, "tool:fn_l")] == ["allow"]


# ----------------------------------------------------------------------
# 既定経路（build 時統治）の回帰 pin: 装飾 builder / 入れ子 builder
#
# 統治は build 時に焼き込まれるため、`build` だけを委譲する装飾 builder や
# `GovernedAgentBuilder` の入れ子でも外れない。registry 経由の Agent で拒否ツールを
# 実行し、例外・実関数の非実行・deny レコードの 3 点で観測する。
# ----------------------------------------------------------------------


class _BuildOnlyDecorator:
    """`build` だけを委譲する装飾 builder（post-processor は registry へ別途明示的に渡す）。"""

    def __init__(self, inner: GovernedAgentBuilder) -> None:
        """委譲先と build した spec 名の記録を初期化する。"""
        self.inner = inner
        self.built: list[str] = []

    def build(self, spec: AgentSpec) -> Agent:
        """spec 名を記録して委譲先の `build` へ渡す。"""
        self.built.append(spec.name)
        return self.inner.build(spec)


def _echo_registry(builder: Any) -> tuple[AgentRegistry, list[str]]:
    """`echo` を 1 回呼ぶ `bot` を登録した registry と、実関数の呼び出し記録を返す。"""
    calls: list[str] = []
    registry = AgentRegistry(agent_builder=builder)
    model = FakeModel().queue_tool_call("echo", '{"text": "x"}').queue_text("done")
    registry.register(
        AgentSpec(name="bot", instructions="i", model=model, tools=[_make_tool(calls)])
    )
    return registry, calls


async def _assert_run_raises_policy_violation(agent: Agent) -> None:
    """Runner 実行が `PolicyViolationError`（直接 or `__cause__`）で中断されることを確かめる。"""
    with pytest.raises(Exception) as excinfo:
        await Runner.run(agent, input="go")
    err = excinfo.value
    assert isinstance(err, PolicyViolationError) or isinstance(err.__cause__, PolicyViolationError)


async def test_decorated_builder_delegating_only_build_still_governs() -> None:
    """`build` だけを委譲する装飾 builder で包んでも、拒否ツールは build 時統治で拒否される。"""
    sink = AuditLog()
    decorator = _BuildOnlyDecorator(
        GovernedAgentBuilder(policy=GovernancePolicy(name="p", allowed_tools=[]), audit_sink=sink)
    )
    registry, calls = _echo_registry(decorator)

    await _assert_run_raises_policy_violation(registry.get("bot"))

    assert decorator.built == ["bot"]
    assert calls == []  # 実関数は非実行
    assert [(e.agent_id, e.decision) for e in _tool_records(sink, "tool:echo")] == [("bot", "deny")]


@pytest.mark.parametrize(
    ("outer_allowed", "inner_allowed", "outer_expected", "inner_expected"),
    [
        (["echo"], [], [], ["deny"]),
        ([], ["echo"], ["deny"], ["allow"]),
        (["echo"], ["echo"], ["allow"], ["allow"]),
    ],
    ids=["outer_allow_inner_deny", "outer_deny_inner_allow", "both_allow"],
)
async def test_nested_governed_builder_still_governs(
    outer_allowed: list[str],
    inner_allowed: list[str],
    outer_expected: list[str],
    inner_expected: list[str],
) -> None:
    """`GovernedAgentBuilder` の入れ子は外側・内側の両ポリシーで評価され、一方が deny なら deny。

    外側の build が先にラップし、内側の build がその上からラップするため、実行時は内側の
    ポリシーが先に評価される（内側が deny なら外側まで届かない）。各 sink の `tool:` 行を
    `==` で固定する。
    """
    outer_sink = AuditLog()
    inner_sink = AuditLog()
    builder = GovernedAgentBuilder(
        policy=GovernancePolicy(name="outer", allowed_tools=outer_allowed),
        audit_sink=outer_sink,
        inner=GovernedAgentBuilder(
            policy=GovernancePolicy(name="inner", allowed_tools=inner_allowed),
            audit_sink=inner_sink,
        ),
    )
    registry, calls = _echo_registry(builder)
    agent = registry.get("bot")

    if "deny" in outer_expected + inner_expected:
        await _assert_run_raises_policy_violation(agent)
        assert calls == []  # 実関数は非実行
    else:
        result = await Runner.run(agent, input="go")
        assert result.final_output == "done"
        assert calls == ["x"]
    assert [e.decision for e in _tool_records(outer_sink, "tool:echo")] == outer_expected
    assert [e.decision for e in _tool_records(inner_sink, "tool:echo")] == inner_expected


# ----------------------------------------------------------------------
# オプトイン（post_processor(sub_agent_tools=True)）: 同一 list の要素置換と追加レコード
# ----------------------------------------------------------------------


@pytest.mark.parametrize("allowed", [True, False], ids=["allow", "deny"])
async def test_factory_clone_during_wire_sees_governed_injected_as_tool(allowed: bool) -> None:
    """結線の途中で factory が作った clone にも、第 3 段で統治した注入 as_tool が届く。

    構成: `a` は handoffs=["f", "b"]、`b` は sub_agents=["sub"]、factory `f` は
    `r.get("b").clone(name="b2")` を返す。`a` の結線中に `f` が呼ばれ、第 3 段より前の `b` を
    clone する（SDK の clone は tools list を共有する）。第 3 段が `b.tools` を再束縛せず同じ
    list の要素を置換していれば、`b2` から呼んだ `sub` の as_tool も `b` のポリシーで評価される。
    """
    sink = AuditLog()
    builder = GovernedAgentBuilder(
        policy=GovernancePolicy(name="p", allowed_tools=["sub"] if allowed else []),
        audit_sink=sink,
    )
    registry = AgentRegistry(
        agent_builder=builder, post_processor=builder.post_processor(sub_agent_tools=True)
    )
    sub_model = FakeModel().queue_text("sub-done")
    registry.register(AgentSpec(name="sub", instructions="s", model=sub_model))
    b_model = FakeModel().queue_tool_call("sub", '{"input": "q"}').queue_text("b-done")
    registry.register(AgentSpec(name="b", instructions="b", model=b_model, sub_agents=["sub"]))
    clones: list[Agent] = []

    def _clone_b(r: AgentRegistry) -> Agent:
        clone = r.get("b").clone(name="b2")
        clones.append(clone)
        return clone

    registry.register_factory("f", _clone_b)
    registry.register(AgentSpec(name="a", instructions="a", handoffs=["f", "b"]))

    registry.get("a")
    b2 = registry.get("f")

    # 前提: clone は結線中に 1 度だけ作られ、b とは別オブジェクト。
    assert clones == [b2]
    assert b2 is not registry.get("b")
    if allowed:
        result = await Runner.run(b2, input="go")
        assert result.final_output == "b-done"
        assert len(sub_model.calls) == 1
        assert [(e.agent_id, e.decision, e.details) for e in _tool_records(sink, "tool:sub")] == [
            ("b", "allow", {"arguments": '{"input": "q"}'})
        ]
    else:
        await _assert_run_raises_policy_violation(b2)
        assert sub_model.calls == []  # サブエージェントの run は起きない
        assert [(e.agent_id, e.decision) for e in _tool_records(sink, "tool:sub")] == [
            ("b", "deny")
        ]


async def test_opt_in_sub_agent_tools_adds_only_injected_tool_record() -> None:
    """`post_processor(sub_agent_tools=True)` は既定の記録列に注入 as_tool の `tool:` 1 行を加える。

    build 時に統治済みの `fn_s` / `direct_tool` は統治済み判定により再ラップされず `tool:` が 2 行に
    ならない。hooks も再合成しないため、ライフサイクル行と利用者フックへの到達列は
    既定（`test_audit_record_sequence_unchanged_by_default`）と同一のまま。
    """
    mixed = _mixed_registry(allowed_tools=_MIXED_ALLOWED, sub_agent_tools=True)

    result = await Runner.run(mixed.registry.get("support"), input="go")

    assert result.final_output == "done"
    assert mixed.calls == {"fn_r": [], "fn_s": ["s"], "fn_l": []}
    assert len(mixed.researcher_model.calls) == 1
    assert len(mixed.direct_model.calls) == 1
    assert _record_shape(mixed.sink) == [
        ("support", "agent_start", "allow", []),
        ("support", "tool_start:fn_s", "allow", []),
        ("support", "tool:fn_s", "allow", ["arguments"]),
        ("support", "tool_end:fn_s", "allow", []),
        ("support", "tool_start:direct_tool", "allow", []),
        ("support", "tool:direct_tool", "allow", ["arguments"]),
        ("support", "tool_end:direct_tool", "allow", []),
        ("support", "tool_start:researcher", "allow", []),
        ("support", "tool:researcher", "allow", ["arguments"]),
        ("researcher", "agent_start", "allow", []),
        ("researcher", "agent_end", "allow", []),
        ("support", "tool_end:researcher", "allow", []),
        ("support", "agent_end", "allow", []),
    ]
    assert mixed.hooks.events == [
        ("start:support", "agent_start"),
        ("tool_start:fn_s", "tool_start:fn_s"),
        ("tool_end:fn_s", "tool_end:fn_s"),
        ("tool_start:direct_tool", "tool_start:direct_tool"),
        ("tool_end:direct_tool", "tool_end:direct_tool"),
        ("tool_start:researcher", "tool_start:researcher"),
        ("tool_end:researcher", "tool_end:researcher"),
        ("end:support", "agent_end"),
    ]
    assert mixed.sink.verify_chain() is True


# ----------------------------------------------------------------------
# オプトイン（post_processor(factory_agents=True)）: 統治済み Agent を返す factory の境界
# ----------------------------------------------------------------------


def _alias_registry(builder: GovernedAgentBuilder) -> tuple[AgentRegistry, list[str]]:
    """spec `a`（`echo` を 1 回呼ぶ）と、`a` をそのまま返す factory `alias` を載せる。

    registry へは `builder.post_processor(factory_agents=True)` を明示的に渡す。
    """
    calls: list[str] = []
    registry = AgentRegistry(
        agent_builder=builder, post_processor=builder.post_processor(factory_agents=True)
    )
    model = FakeModel().queue_tool_call("echo", '{"text": "x"}').queue_text("done")
    registry.register(AgentSpec(name="a", instructions="i", model=model, tools=[_make_tool(calls)]))
    registry.register_factory("alias", lambda r: r.get("a"))
    return registry, calls


async def test_factory_returning_governed_agent_is_evaluated_by_both_policies() -> None:
    """統治済み Agent を返す factory は、factory 名と元 spec 名の両ポリシーで評価される（境界）。

    factory 経路は印で skip しない（skip すると factory 名の override が黙って効かなくなる）。
    許可では `tool:` が factory 名 -> 元 spec 名の順に 2 行残り、factory 名だけを deny にすると
    実関数は走らず deny 1 行で止まる。
    """
    allow_sink = AuditLog()
    registry, calls = _alias_registry(
        GovernedAgentBuilder(
            policy=GovernancePolicy(name="p", allowed_tools=["echo"]),
            audit_sink=allow_sink,
        )
    )
    alias = registry.get("alias")
    assert alias is not registry.get("a")  # 前提: factory 経路は clone を返す

    result = await Runner.run(alias, input="go")

    assert result.final_output == "done"
    assert calls == ["x"]
    assert [(e.agent_id, e.decision) for e in _tool_records(allow_sink, "tool:echo")] == [
        ("alias", "allow"),
        ("a", "allow"),
    ]

    deny_sink = AuditLog()
    registry, calls = _alias_registry(
        GovernedAgentBuilder(
            policy=GovernancePolicy(name="p", allowed_tools=["echo"]),
            overrides={"alias": GovernancePolicy(name="d", allowed_tools=[])},
            audit_sink=deny_sink,
        )
    )

    await _assert_run_raises_policy_violation(registry.get("alias"))

    assert calls == []  # 実関数は非実行
    assert [(e.agent_id, e.decision) for e in _tool_records(deny_sink, "tool:echo")] == [
        ("alias", "deny")
    ]


# ----------------------------------------------------------------------
# 統治済み登録の SDK 契約トリップワイヤ（dataclasses.replace / copy.copy）
# ----------------------------------------------------------------------


def test_governed_registration_survives_replace_and_copy_tripwire() -> None:
    """govern 済みの実行本体は SDK の失敗ハンドラ付き invoker で、複製後も統治済みと判定される。

    ADR-0045: `@function_tool` 由来の tool を govern すると、`on_invoke_tool` は
    `agents.tool._FailureHandlingFunctionToolInvoker` のサブクラス（`_GovernedInvoker`）になる
    （SDK 0.22.x の承認の事前検証はこの型判定で有効になる）。SDK の `FunctionTool.__post_init__` /
    `__copy__` は `__agents_bind_function_tool__` で invoker を新しい tool へ束縛し直し、複製ごとに
    別インスタンスを作る。そのため統治済みの印は「登録済みの同一オブジェクトを指し続けること」では
    なく「再束縛で作られた別インスタンスが生成時に登録されること」で保つ。印が外れると第 3 段が
    build 時統治済みの tool を再ラップし、`tool:` レコードが 2 行になる（fail-closed だが記録が
    変わる）。弱参照の登録簿（`_GOVERNED_WRAPPERS`）に載せるため invoker は弱参照可能である
    （基底が `__slots__` を持たないこと）。
    """
    tool = _make_tool([])
    governed = _govern_tool(
        tool,
        policy=GovernancePolicy(name="p"),
        sink=AuditLog(),
        denied_exc=PolicyViolationError,
        agent_name="bot",
    )
    assert _is_governed(governed) is True
    assert _is_governed(tool) is False
    invoker = governed.on_invoke_tool
    assert isinstance(invoker, sdk_tool._FailureHandlingFunctionToolInvoker)
    assert weakref.ref(invoker)() is invoker

    for derived in (dataclasses.replace(governed, name="renamed"), copy.copy(governed)):
        assert derived is not governed
        assert isinstance(derived.on_invoke_tool, sdk_tool._FailureHandlingFunctionToolInvoker)
        # 再束縛で新しい tool へ束縛した別インスタンスになる（SDK の複製契約）。
        assert derived.on_invoke_tool._function_tool is derived
        assert _is_governed(derived) is True, (
            "SDK の FunctionTool 複製で再束縛された govern 済み invoker が統治済みと判定されない。"
            "_adapters/governance.py の _GovernedInvoker の再束縛と生成時登録を見直すこと。"
        )


async def test_governed_copy_rebinds_inner_invoker_to_the_copied_tool() -> None:
    """複製した govern 済み tool の失敗処理は、複製側の tool の設定で行われる（ADR-0045）。

    SDK は失敗時の文言を invoker が束縛している tool（`_function_tool`）から解決する。
    `_GovernedInvoker` の再束縛が内側の invoker まで束縛し直さないと、複製で失敗時の関数を
    差し替えても複製前の tool の設定が使われる。
    """

    @function_tool(name_override="boom")
    def _boom(text: str) -> str:
        """常に失敗する。"""
        raise ValueError("boom")

    governed = _govern_tool(
        _boom,
        policy=GovernancePolicy(name="p"),
        sink=AuditLog(),
        denied_exc=PolicyViolationError,
        agent_name="bot",
    )

    def _custom(ctx: Any, error: Exception) -> str:
        return f"custom:{error}"

    derived = dataclasses.replace(
        governed, _failure_error_function=_custom, _use_default_failure_error_function=False
    )

    result = await derived.on_invoke_tool(_tool_ctx("boom", '{"text": "x"}'), '{"text": "x"}')

    assert derived.on_invoke_tool._function_tool is derived
    assert result == "custom:boom"


def test_governed_tool_exposes_same_wrapped_callable_as_ungoverned() -> None:
    """govern 済み tool の `__wrapped__` は govern なしと同じ元関数を返す（公開面の parity）。

    SDK の `FunctionTool.__wrapped__` は invoker の実行本体に付いた印から元関数を辿る。
    govern の実行本体へ印を写さないと、govern 済みだけ AttributeError になる。返るのは未統治の
    元関数であり、SDK が実行経路でこの印を読まないことはトリップワイヤ
    （`test_sdk_function_tool_invoker_private_symbols_tripwire`）が検査する。
    """
    tool = _make_tool([])
    governed = _govern_tool(
        tool,
        policy=GovernancePolicy(name="p"),
        sink=AuditLog(),
        denied_exc=PolicyViolationError,
        agent_name="bot",
    )

    assert governed.__wrapped__ is tool.__wrapped__
    assert copy.copy(governed).__wrapped__ is tool.__wrapped__


# ----------------------------------------------------------------------
# オプトインは post-processor の明示的な受け渡し: 装飾 builder / 入れ子 builder
# ----------------------------------------------------------------------


def _researcher_registry(
    builder: Any, post_processor: Any, *, sub_model: FakeModel
) -> AgentRegistry:
    """sub agent `researcher` と、それを 1 回呼ぶ親 `support` を載せた registry を返す。"""
    registry = AgentRegistry(agent_builder=builder, post_processor=post_processor)
    registry.register(AgentSpec(name="researcher", instructions="r", model=sub_model))
    support_model = FakeModel().queue_tool_call("researcher", '{"input": "r"}').queue_text("done")
    registry.register(
        AgentSpec(name="support", instructions="s", model=support_model, sub_agents=["researcher"])
    )
    return registry


async def test_decorated_builder_with_explicit_post_processor_governs_injected_as_tool() -> None:
    """build だけ委譲する装飾 builder でも、明示的に渡した post-processor が注入 as_tool を統治。

    ポリシー外の `researcher` as_tool は deny され（`UserError.__cause__` が
    `PolicyViolationError`）、サブエージェントの model は 1 度も呼ばれず、`tool:` 行は
    親名の deny 1 行だけになる。
    """
    sink = AuditLog()
    governed = GovernedAgentBuilder(
        policy=GovernancePolicy(name="p", allowed_tools=["other"]), audit_sink=sink
    )
    decorator = _BuildOnlyDecorator(governed)
    sub_model = FakeModel().queue_text("researched")
    registry = _researcher_registry(
        decorator, governed.post_processor(sub_agent_tools=True), sub_model=sub_model
    )

    with pytest.raises(UserError) as excinfo:
        await Runner.run(registry.get("support"), input="go")

    assert isinstance(excinfo.value.__cause__, PolicyViolationError)
    assert sorted(decorator.built) == ["researcher", "support"]
    assert sub_model.calls == []  # サブエージェントの run は起きない
    assert [
        (e.agent_id, e.action, e.decision)
        for e in sink.get_entries()
        if e.action.startswith("tool:")
    ] == [("support", "tool:researcher", "deny")]


@pytest.mark.parametrize("source", ["outer", "inner"])
async def test_nested_builder_opt_in_uses_explicit_post_processor_policy(source: str) -> None:
    """入れ子の builder では、post-processor を作った側の sink にだけ・そのポリシーで残る。

    外側は `researcher` を許可、内側は拒否するポリシーにする。外側から作れば allow で
    サブエージェントまで走り、内側から作れば deny で止まる。as_tool の `tool:` 行は作った側の
    sink にだけ 1 行残り、もう一方の sink には残らない。
    """
    outer_sink = AuditLog()
    inner_sink = AuditLog()
    inner = GovernedAgentBuilder(
        policy=GovernancePolicy(name="inner", allowed_tools=[]), audit_sink=inner_sink
    )
    outer = GovernedAgentBuilder(
        policy=GovernancePolicy(name="outer", allowed_tools=["researcher"]),
        audit_sink=outer_sink,
        inner=inner,
    )
    maker = outer if source == "outer" else inner
    sub_model = FakeModel().queue_text("researched")
    registry = _researcher_registry(
        outer, maker.post_processor(sub_agent_tools=True), sub_model=sub_model
    )

    if source == "outer":
        result = await Runner.run(registry.get("support"), input="go")
        assert result.final_output == "done"
        assert len(sub_model.calls) == 1
        expected_outer = [("support", "allow", '{"input": "r"}')]
        expected_inner: list[Any] = []
    else:
        with pytest.raises(UserError) as excinfo:
            await Runner.run(registry.get("support"), input="go")
        assert isinstance(excinfo.value.__cause__, PolicyViolationError)
        assert sub_model.calls == []
        expected_outer = []
        expected_inner = [("support", "deny", '{"input": "r"}')]
        assert "researcher" in _tool_records(inner_sink, "tool:researcher")[0].details["reason"]
    assert [
        (e.agent_id, e.decision, e.details["arguments"])
        for e in _tool_records(outer_sink, "tool:researcher")
    ] == expected_outer
    assert [
        (e.agent_id, e.decision, e.details["arguments"])
        for e in _tool_records(inner_sink, "tool:researcher")
    ] == expected_inner


# ----------------------------------------------------------------------
# govern 済み実行本体の SDK invoker 準拠（ADR-0045）
#
# govern は `@function_tool` 由来の実行本体を SDK の失敗ハンドラ付き invoker のサブクラスで
# 置き換える。SDK 0.22.x の承認の事前検証（callable な `needs_approval` の前に引数を入力モデルで
# 検証し、値が変わる引数は判定関数を呼ばずに承認必須にする）は invoker の型判定で有効になるため、
# 素の関数で置き換えると govern 済みツールだけ事前検証が外れ、引数の表現を変えるだけで条件付き
# 承認をすり抜けられる。
# ----------------------------------------------------------------------


def _derive(tool: FunctionTool, how: str) -> FunctionTool:
    """SDK の複製経路（`dataclasses.replace` / `copy.copy`）で tool を派生させる。"""
    if how == "replace":
        return dataclasses.replace(tool, description="derived")
    if how == "copy":
        return copy.copy(tool)
    return tool


def _approval_tools(seen: list[dict[str, Any]], ran: list[Any]) -> dict[str, FunctionTool]:
    """条件付き承認（callable な `needs_approval`）を持つ実 `function_tool` 一式を作る。

    判定関数は生の型に敏感（`type(v) is int` / `is True`）にしてある。SDK の事前検証を経ずに
    生の引数が判定関数へ届くと、型変換される入力では承認を要求しない（ADR-0043 のすり抜け実測と
    同じ形）。判定関数へ渡った引数は `seen` に、ツール本体が受け取った値は `ran` に積む。
    """

    async def needs_big(ctx: Any, args: dict[str, Any], call_id: str) -> bool:
        seen.append(dict(args))
        amount = args.get("amount")
        return type(amount) is int and amount > 1000

    async def needs_execute(ctx: Any, args: dict[str, Any], call_id: str) -> bool:
        seen.append(dict(args))
        return args.get("execute") is True

    @function_tool(needs_approval=needs_big)
    def pay(amount: int) -> str:
        """支払う（既定値なし）。"""
        ran.append(amount)
        return f"paid {amount}"

    @function_tool(needs_approval=needs_big)
    def pay_memo(amount: int, memo: str = "") -> str:
        """メモ付きで支払う（既定値あり）。"""
        ran.append(amount)
        return f"paid {amount}"

    @function_tool(needs_approval=needs_execute)
    def launch(execute: bool) -> str:
        """実行フラグ付きで起動する。"""
        ran.append(execute)
        return f"launched {execute}"

    return {"pay": pay, "pay_memo": pay_memo, "launch": launch}


@dataclasses.dataclass
class _ApprovalRun:
    """条件付き承認ツールを 1 回走らせた結果（再開用の agent と観測点を含む）。"""

    result: Any
    agent: Agent
    seen: list[dict[str, Any]]
    ran: list[Any]
    sink: AuditLog


async def _run_approval_case(tool_name: str, args_json: str, *, governed: bool) -> _ApprovalRun:
    """`tool_name` を 1 回呼ぶ run を govern あり / なしの registry で実行する。"""
    seen: list[dict[str, Any]] = []
    ran: list[Any] = []
    tool = _approval_tools(seen, ran)[tool_name]
    sink = AuditLog()
    if governed:
        registry = AgentRegistry(
            agent_builder=GovernedAgentBuilder(
                policy=GovernancePolicy(name="p", allowed_tools=[tool_name]), audit_sink=sink
            )
        )
    else:
        registry = AgentRegistry()
    model = FakeModel().queue_tool_call(tool_name, args_json).queue_text("done")
    registry.register(AgentSpec(name="bot", instructions="i", model=model, tools=[tool]))
    agent = registry.get("bot")
    result = await Runner.run(agent, input="go")
    return _ApprovalRun(result=result, agent=agent, seen=seen, ran=ran, sink=sink)


@pytest.mark.parametrize(
    ("tool_name", "args_json", "expected_interruptions"),
    [
        pytest.param("pay", '{"amount": "5000"}', 1, id="int-as-string"),
        pytest.param("pay", '{"amount": 5000.0}', 1, id="int-as-float"),
        pytest.param("launch", '{"execute": "yes"}', 1, id="bool-as-string"),
        pytest.param("pay_memo", '{"amount": 500}', 1, id="default-completion"),
        pytest.param("pay", '{"amount": 5000}', 1, id="control-int-over-limit"),
        pytest.param("pay", '{"amount": 500}', 0, id="control-int-under-limit"),
        pytest.param("launch", '{"execute": true}', 1, id="control-bool"),
    ],
)
async def test_governed_conditional_approval_matches_ungoverned(
    tool_name: str, args_json: str, expected_interruptions: int
) -> None:
    """govern 済み / なしの同じツールで、承認要求の有無と判定関数へ渡る引数が一致する（ADR-0045）。

    型変換される入力（数値の文字列表現・小数表現・真偽値の文字列表現）と既定値の補完が起きる入力
    では、SDK 0.22.x の事前検証が判定関数を呼ばずに承認を必須にする。govern 済みツールの実行本体が
    SDK の失敗ハンドラ付き invoker でないと事前検証が外れ、生の型に敏感な判定関数が承認不要と
    判定して本体が承認なしで実行される（govern を足すと承認が弱くなる逆転）。対照の整数入力では
    両者とも判定関数が呼ばれる。`expected_interruptions` は govern なしの SDK 既定の挙動の前提確認
    である。
    """
    plain = await _run_approval_case(tool_name, args_json, governed=False)
    gov = await _run_approval_case(tool_name, args_json, governed=True)

    assert len(plain.result.interruptions) == expected_interruptions  # 前提: SDK 既定の判定
    assert len(gov.result.interruptions) == len(plain.result.interruptions), (
        f"govern 済みだけ承認要求が変わった: {args_json} "
        f"(govern なし={len(plain.result.interruptions)}, "
        f"govern 済み={len(gov.result.interruptions)})"
    )
    assert gov.seen == plain.seen
    assert gov.ran == plain.ran


async def test_governed_approval_resume_runs_with_validated_arguments_and_records_allow() -> None:
    """型変換される入力で承認待ちになった govern 済みツールは、承認後の再開で検証済みの値で動く。

    `{"amount": "5000"}` は事前検証で承認必須になり、承認前は本体も `tool:` 記録も無い。承認して
    再開すると本体は int の 5000 を受け取り（SDK が準備した引数が govern の実行本体を経て届く）、
    監査には allow がモデルの送った生の引数付きで 1 行残る（ADR-0045 Decision 5）。
    """
    run = await _run_approval_case("pay", '{"amount": "5000"}', governed=True)

    assert len(run.result.interruptions) == 1, "govern 済みツールが承認待ちにならず実行された"
    assert run.ran == []
    assert _tool_records(run.sink, "tool:pay") == []

    state = run.result.to_state()
    for item in run.result.interruptions:
        state.approve(item)
    resumed = await Runner.run(run.agent, state)

    assert resumed.final_output == "done"
    assert run.ran == [5000]
    assert type(run.ran[0]) is int
    assert [
        (e.agent_id, e.decision, e.details["arguments"])
        for e in _tool_records(run.sink, "tool:pay")
    ] == [("bot", "allow", '{"amount": "5000"}')]


@pytest.mark.parametrize("derive", ["original", "replace", "copy"])
async def test_governed_invoker_copy_records_single_tool_entry_per_run(derive: str) -> None:
    """govern 済み tool（複製後を含む）は第 3 段で再ラップされず、1 回の実行で `tool:` が 1 行。

    複製で SDK が再束縛した invoker が統治済みと判定されないと、`govern_ungoverned_tools` が
    再ラップして `tool:` レコードが 2 行になる。
    """
    calls: list[str] = []
    sink = AuditLog()
    policy = GovernancePolicy(name="p", allowed_tools=["echo"])
    governed = govern_spec(
        AgentSpec(name="bot", instructions="i", tools=[_make_tool(calls)]),
        policy=policy,
        audit_sink=sink,
    ).tools[0]
    model = FakeModel().queue_tool_call("echo", '{"text": "hi"}').queue_text("done")
    agent = Agent(name="bot", instructions="i", model=model, tools=[_derive(governed, derive)])
    govern_ungoverned_tools(agent, policy=policy, audit_sink=sink, agent_name="bot")

    result = await Runner.run(agent, input="go")

    assert result.final_output == "done"
    assert calls == ["hi"]
    assert [(e.agent_id, e.decision) for e in _tool_records(sink, "tool:echo")] == [
        ("bot", "allow")
    ]


async def test_deepcopied_governed_tool_is_ungoverned_and_regoverned_fail_closed() -> None:
    """govern 済み tool を `copy.deepcopy` した複製は統治済みの印を保たない（fail-closed）。

    deepcopy は `__init__` を通らないため `_GovernedInvoker` の複製は登録されず、`_is_governed` は
    偽になる。第 3 段（`govern_ungoverned_tools`）で再び govern され、1 回の実行で評価と `tool:`
    記録が 2 回ずつになる（評価が抜けるのではなく二重になる方向）。複製の実行本体は元の評価を
    保つので、deny ポリシーでは本体が実行されない。
    """
    calls: list[str] = []
    sink = AuditLog()
    policy = GovernancePolicy(name="p", allowed_tools=["echo"])
    governed = govern_spec(
        AgentSpec(name="bot", instructions="i", tools=[_make_tool(calls)]),
        policy=policy,
        audit_sink=sink,
    ).tools[0]
    cloned = copy.deepcopy(governed)

    assert _is_governed(governed) is True
    assert _is_governed(cloned) is False

    model = FakeModel().queue_tool_call("echo", '{"text": "hi"}').queue_text("done")
    agent = Agent(name="bot", instructions="i", model=model, tools=[cloned])
    govern_ungoverned_tools(agent, policy=policy, audit_sink=sink, agent_name="bot")

    result = await Runner.run(agent, input="go")

    assert result.final_output == "done"
    assert calls == ["hi"]
    assert [(e.agent_id, e.decision) for e in _tool_records(sink, "tool:echo")] == [
        ("bot", "allow"),
        ("bot", "allow"),
    ]

    deny_sink = AuditLog()
    denied = copy.deepcopy(
        govern_spec(
            AgentSpec(name="bot", instructions="i", tools=[_make_tool(calls)]),
            policy=GovernancePolicy(name="deny", allowed_tools=["other"]),
            audit_sink=deny_sink,
        ).tools[0]
    )
    with pytest.raises(PolicyViolationError):
        await denied.on_invoke_tool(_tool_ctx("echo", '{"text": "no"}'), '{"text": "no"}')
    assert calls == ["hi"]


async def test_governed_deny_propagates_without_failure_error_function_message() -> None:
    """deny は `PolicyViolationError` で run を止め、`failure_error_function` の文言を返さない。

    govern 済みの実行本体を SDK の失敗ハンドラ付き invoker にしても、deny を失敗ハンドラに吸わせない
    （ADR-0045 Decision 5: deny の着地は不変）。吸われると失敗文言がツール出力としてモデルへ返り、
    run が続行する。
    """
    calls: list[str] = []

    @function_tool(name_override="echo", failure_error_function=lambda ctx, err: "TOOL FAILED")
    def _tool(text: str) -> str:
        """テキストを記録してエコーする。"""
        calls.append(text)
        return text

    sink = AuditLog()
    registry = AgentRegistry(
        agent_builder=GovernedAgentBuilder(
            policy=GovernancePolicy(name="p", allowed_tools=[]), audit_sink=sink
        )
    )
    model = FakeModel().queue_tool_call("echo", '{"text": "x"}').queue_text("unreached")
    registry.register(AgentSpec(name="bot", instructions="i", model=model, tools=[_tool]))

    await _assert_run_raises_policy_violation(registry.get("bot"))

    assert calls == []
    assert len(model.calls) == 1  # 失敗文言を載せた 2 回目のモデル呼び出しが無い
    assert [(e.agent_id, e.decision) for e in _tool_records(sink, "tool:echo")] == [("bot", "deny")]


async def test_governed_allow_keeps_failure_error_function_for_tool_errors() -> None:
    """allow 後の本体の失敗は、govern なしと同じく `failure_error_function` の文言がモデルへ返る。

    失敗処理は govern の内側（元の SDK invoker）が担う。
    """

    @function_tool(name_override="boom", failure_error_function=lambda ctx, err: "TOOL FAILED")
    def _tool(text: str) -> str:
        """常に失敗する。"""
        raise RuntimeError("kaboom")

    sink = AuditLog()
    registry = AgentRegistry(
        agent_builder=GovernedAgentBuilder(
            policy=GovernancePolicy(name="p", allowed_tools=["boom"]), audit_sink=sink
        )
    )
    model = FakeModel().queue_tool_call("boom", '{"text": "x"}').queue_text("done")
    registry.register(AgentSpec(name="bot", instructions="i", model=model, tools=[_tool]))

    result = await Runner.run(registry.get("bot"), input="go")

    assert result.final_output == "done"
    assert len(model.calls) == 2
    assert "TOOL FAILED" in str(model.calls[1].input)
    assert [(e.agent_id, e.decision) for e in _tool_records(sink, "tool:boom")] == [
        ("bot", "allow")
    ]


def _timeout_validation_outcome(tool: FunctionTool) -> str | None:
    """`timeout_seconds` を設定した複製を作り、SDK の timeout 設定検証のエラー文言を返す。"""
    try:
        dataclasses.replace(tool, timeout_seconds=1.0)
    except ValueError as exc:
        return str(exc)
    return None


@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
def test_governed_function_tool_sync_marker_and_timeout_validation_match_ungoverned(
    sync: bool,
) -> None:
    """同期関数ツールのマーカーと timeout 設定の検証結果が govern あり / なしで一致する。

    SDK は同期関数ツールの invoker に `__agents_sync_function_tool__` を付け、timeout 設定を
    拒否する（同期関数は timeout で打ち切れないため）。govern 済みの実行本体がマーカーを
    引き継がないと、同期関数ツールに timeout を設定できてしまう（ADR-0045 Decision 1）。
    """
    if sync:

        @function_tool(name_override="echo")
        def _tool(text: str) -> str:
            """同期関数ツール。"""
            return text

    else:

        @function_tool(name_override="echo")
        async def _tool(text: str) -> str:
            """非同期関数ツール。"""
            return text

    governed = govern_spec(
        AgentSpec(name="bot", instructions="i", tools=[_tool]),
        policy=GovernancePolicy(name="p"),
        audit_sink=AuditLog(),
    ).tools[0]
    marker = sdk_tool._SYNC_FUNCTION_TOOL_MARKER

    assert getattr(_tool.on_invoke_tool, marker, False) is sync  # 前提: SDK の付与規則
    assert getattr(governed.on_invoke_tool, marker, False) is sync
    assert (_timeout_validation_outcome(_tool) is not None) is sync  # 前提: SDK の検証
    assert _timeout_validation_outcome(governed) == _timeout_validation_outcome(_tool)


async def test_bare_function_tool_governed_by_plain_wrapper_allow_and_deny() -> None:
    """実行本体が invoker でない `FunctionTool` は従来の素の関数ラップで統治される（allow / deny）。

    利用者が `FunctionTool` を直接組んだ場合は govern なしでも SDK の事前検証が効かないため、
    invoker 化せず従来のラップを使う（ADR-0045 Decision 2）。統治済みの印は複製後も保たれる。
    """
    calls: list[str] = []

    async def _on_invoke(ctx: Any, input_json: str) -> str:
        calls.append(input_json)
        return "ok"

    def _bare() -> FunctionTool:
        return FunctionTool(
            name="bare",
            description="bare tool",
            params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
            on_invoke_tool=_on_invoke,
        )

    allow_sink = AuditLog()
    allowed = govern_spec(
        AgentSpec(name="bot", instructions="i", tools=[_bare()]),
        policy=GovernancePolicy(name="p", allowed_tools=["bare"]),
        audit_sink=allow_sink,
    ).tools[0]
    assert not isinstance(allowed.on_invoke_tool, sdk_tool._FailureHandlingFunctionToolInvoker)
    for derived in (allowed, dataclasses.replace(allowed, description="d"), copy.copy(allowed)):
        assert _is_governed(derived) is True
    assert await allowed.on_invoke_tool(_tool_ctx("bare", "{}"), "{}") == "ok"
    assert calls == ["{}"]
    assert [(e.agent_id, e.decision) for e in _tool_records(allow_sink, "tool:bare")] == [
        ("bot", "allow")
    ]

    deny_sink = AuditLog()
    denied = govern_spec(
        AgentSpec(name="bot", instructions="i", tools=[_bare()]),
        policy=GovernancePolicy(name="p", allowed_tools=[]),
        audit_sink=deny_sink,
    ).tools[0]
    with pytest.raises(PolicyViolationError, match="bare"):
        await denied.on_invoke_tool(_tool_ctx("bare", "{}"), "{}")
    assert calls == ["{}"]
    assert [(e.agent_id, e.decision) for e in _tool_records(deny_sink, "tool:bare")] == [
        ("bot", "deny")
    ]


# SDK（openai-agents 0.22.x）内で実行本体と元関数の印を参照する既知の箇所（ファイル, 行の中身）。
# 行番号は patch 更新でずれるため中身で照合する。
_SDK_INVOKER_REFERENCES = [
    (
        "tool.py",
        '_FUNCTION_TOOL_WRAPPED_CALLABLE_MARKER = "__agents_function_tool_wrapped_callable__"',
    ),
    ("tool.py", "wrapped_callable = instance.on_invoke_tool._get_wrapped_callable()"),
    ("tool.py", 'raise AttributeError("FunctionTool.__wrapped__ is read-only")'),
    ("tool.py", "__wrapped__ = _FunctionToolWrappedCallableDescriptor()"),
    ("tool.py", "self._invoke_tool_impl = invoke_tool_impl"),
    ("tool.py", "def _get_wrapped_callable(self) -> object:"),
    ("tool.py", "self._invoke_tool_impl,"),
    ("tool.py", "_FUNCTION_TOOL_WRAPPED_CALLABLE_MARKER,"),
    ("tool.py", "self._invoke_tool_impl,"),
    ("tool.py", 'prepare = getattr(self._invoke_tool_impl, "__agents_prepare_arguments__", None)'),
    ("tool.py", "return await self._invoke_tool_impl(ctx, input)"),
    ("tool.py", 'inspect.getattr_static(func, "__wrapped__", missing) is not missing'),
    ("tool.py", 'or hasattr(call_descriptor, "__wrapped__")'),
    ("tool.py", "_FUNCTION_TOOL_WRAPPED_CALLABLE_MARKER,"),
]


def test_sdk_function_tool_invoker_private_symbols_tripwire() -> None:
    """govern の invoker 準拠が依存する SDK 非公開要素の存在を検査する（ADR-0045 Decision 3・4）。

    依存点:
    - `agents.tool._FailureHandlingFunctionToolInvoker`（継承元。承認の事前検証の型判定の対象）と
      コンストラクタ引数 `(invoke_tool_impl, on_handled_error, *, function_tool)`・内部属性
      `_invoke_tool_impl` / `_on_handled_error` / `_function_tool`・`__slots__` を持たないこと
      （インスタンスを弱参照の登録簿へ載せるため）
    - 再束縛プロトコル `__agents_bind_function_tool__`（`FunctionTool.__post_init__` / `__copy__` が
      呼び、束縛先が違えば別インスタンスを返す）
    - 同期関数ツールのマーカー `_SYNC_FUNCTION_TOOL_MARKER`（= `__agents_sync_function_tool__`）
    - 事前検証の材料 `__agents_prepare_arguments__` と `__agents_function_tool_wrapped_callable__`
      （`@function_tool` が `_invoke_tool_impl` の関数属性として付与する。govern は統治済みの実行
      本体へ写す）と、それを参照する `prepare_arguments`
    - SDK 内で実行本体（`_invoke_tool_impl`）と元関数の印（`_get_wrapped_callable` /
      `__wrapped__` / `_FUNCTION_TOOL_WRAPPED_CALLABLE_MARKER`）を参照する箇所が、下の既知の集合と
      完全に一致すること。呼び出しの字面に限らず、変数への代入・`getattr`・定義を含む全参照を単語
      境界で列挙する。govern は `__call__` と `_invoke_tool_impl` の両方に評価を置き、元関数の印を
      実行本体へ写す（`__wrapped__` は未統治の元関数を返す）ため、SDK がこれらを新しい経路で参照・
      実行するとポリシー評価を迂回しうる。参照が増減したら SDK の差分を読み、迂回にならないことを
      確かめてから既知の集合を更新する
    """
    invoker_cls = sdk_tool._FailureHandlingFunctionToolInvoker
    params = inspect.signature(invoker_cls.__init__).parameters
    assert list(params) == ["self", "invoke_tool_impl", "on_handled_error", "function_tool"]
    assert params["function_tool"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "__slots__" not in vars(invoker_cls)
    assert sdk_tool._SYNC_FUNCTION_TOOL_MARKER == "__agents_sync_function_tool__"

    tool = _make_tool([])
    invoker = tool.on_invoke_tool
    assert isinstance(invoker, invoker_cls)
    assert invoker._function_tool is tool
    assert callable(invoker._on_handled_error)
    assert callable(invoker.prepare_arguments)
    assert callable(invoker.__agents_bind_function_tool__)
    impl = invoker._invoke_tool_impl
    assert callable(getattr(impl, "__agents_prepare_arguments__", None))
    assert hasattr(impl, "__agents_function_tool_wrapped_callable__")
    copied = copy.copy(tool)
    assert copied.on_invoke_tool is not invoker
    assert copied.on_invoke_tool._function_tool is copied

    reference = re.compile(
        r"\b(_invoke_tool_impl|_get_wrapped_callable|__wrapped__|_FUNCTION_TOOL_WRAPPED_CALLABLE_MARKER"
        r"|__agents_function_tool_wrapped_callable__)\b"
    )
    sdk_root = Path(inspect.getfile(sdk_tool)).parent
    references = sorted(
        (str(path.relative_to(sdk_root)), line.strip())
        for path in sdk_root.rglob("*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
        if reference.search(line)
    )
    assert references == sorted(_SDK_INVOKER_REFERENCES), (
        "SDK 内で実行本体（_invoke_tool_impl）か元関数の印（__wrapped__ 等）を参照する箇所が"
        "増減した。新しい参照がポリシー評価を通らずに実行本体・元関数を呼ぶ経路でないかを SDK の"
        "差分で確かめ、迂回にならない場合だけ _SDK_INVOKER_REFERENCES を更新すること"
        "（ADR-0045 Decision 3・4）。"
    )


@pytest.mark.parametrize("derive", ["original", "replace", "copy"])
async def test_governed_invoke_tool_impl_direct_call_still_denies(derive: str) -> None:
    """govern 済みの `on_invoke_tool._invoke_tool_impl` の直接呼び出しでも deny される（迂回防止）。

    基底 invoker へ渡す実行本体は統治済みのもの（ADR-0045 Decision 1）で、元の実行本体を渡すと
    `_invoke_tool_impl` の直接呼び出しで評価を迂回できる。複製で再束縛されたインスタンスでも同じ。
    """
    calls: list[str] = []
    sink = AuditLog()
    governed = govern_spec(
        AgentSpec(name="bot", instructions="i", tools=[_make_tool(calls)]),
        policy=GovernancePolicy(name="p", allowed_tools=[]),
        audit_sink=sink,
    ).tools[0]
    target = _derive(governed, derive)
    impl = target.on_invoke_tool._invoke_tool_impl

    with pytest.raises(PolicyViolationError, match="echo"):
        await impl(_tool_ctx("echo", '{"text": "x"}'), '{"text": "x"}')

    assert calls == []
    assert [
        (e.agent_id, e.decision, e.details["arguments"]) for e in _tool_records(sink, "tool:echo")
    ] == [("bot", "deny", '{"text": "x"}')]


@pytest.mark.parametrize("derive", ["original", "replace", "copy"])
@pytest.mark.parametrize(
    ("deny_side", "expected"),
    [
        pytest.param("none", [("outer", "allow"), ("inner", "allow")], id="both-allow"),
        pytest.param("outer", [("outer", "deny")], id="outer-deny"),
        pytest.param("inner", [("outer", "allow"), ("inner", "deny")], id="inner-deny"),
    ],
)
async def test_regoverned_tool_is_evaluated_by_outer_and_inner(
    derive: str, deny_side: str, expected: list[tuple[str, str]]
) -> None:
    """統治済みツールの再 govern は外側・内側の両方で評価・記録され、片方の deny で止まる。

    再度の govern では内側の実行本体が govern 済み invoker になる。外側の再束縛は内側も連鎖して
    作り直す（ADR-0045 Consequences）。複製後も同じ評価・記録になる。
    """
    calls: list[str] = []
    sink = AuditLog()

    def _policy(side: str) -> GovernancePolicy:
        return GovernancePolicy(name=side, allowed_tools=[] if deny_side == side else ["echo"])

    inner_tool = govern_spec(
        AgentSpec(name="inner", instructions="i", tools=[_make_tool(calls)]),
        policy=_policy("inner"),
        audit_sink=sink,
    ).tools[0]
    outer_tool = govern_spec(
        AgentSpec(name="outer", instructions="i", tools=[inner_tool]),
        policy=_policy("outer"),
        audit_sink=sink,
    ).tools[0]
    target = _derive(outer_tool, derive)

    if deny_side == "none":
        out = await target.on_invoke_tool(_tool_ctx("echo", '{"text": "x"}'), '{"text": "x"}')
        assert out == "echo:x"
        assert calls == ["x"]
    else:
        with pytest.raises(PolicyViolationError, match="echo"):
            await target.on_invoke_tool(_tool_ctx("echo", '{"text": "x"}'), '{"text": "x"}')
        assert calls == []
    assert [(e.agent_id, e.decision) for e in _tool_records(sink, "tool:echo")] == expected


async def _received_context_type(annotation: str, *, governed: bool) -> type:
    """第 1 引数の注釈が `annotation` の関数ツールを run し、本体が受け取った文脈の型を返す。"""
    received: list[type] = []
    if annotation == "run_context_wrapper":

        @function_tool(name_override="probe")
        def _tool(ctx: RunContextWrapper[Any], text: str) -> str:
            """受け取ったコンテキストの型を記録する。"""
            received.append(type(ctx))
            return text

    else:

        @function_tool(name_override="probe")
        def _tool(ctx: ToolContext[Any], text: str) -> str:
            """受け取ったコンテキストの型を記録する。"""
            received.append(type(ctx))
            return text

    if governed:
        registry = AgentRegistry(
            agent_builder=GovernedAgentBuilder(
                policy=GovernancePolicy(name="p", allowed_tools=["probe"]), audit_sink=AuditLog()
            )
        )
    else:
        registry = AgentRegistry()
    model = FakeModel().queue_tool_call("probe", '{"text": "x"}').queue_text("done")
    registry.register(AgentSpec(name="bot", instructions="i", model=model, tools=[_tool]))
    await Runner.run(registry.get("bot"), input="go")
    assert len(received) == 1
    return received[0]


@pytest.mark.parametrize("annotation", ["run_context_wrapper", "tool_context"])
async def test_governed_tool_receives_same_context_type_as_ungoverned(annotation: str) -> None:
    """内側関数のコンテキスト注釈によらず、govern 済み / なしで受け取るコンテキスト型が一致する。

    SDK は `on_invoke_tool` の第 1 引数の注釈で渡すコンテキスト型を選ぶ。govern 済みの実行本体の
    注釈が SDK の invoker と同じ解決経路にならないと、ツールが受け取る型が変わる（ADR-0045
    Decision 1 の `ToolContext[Any]` 注釈）。
    """
    plain = await _received_context_type(annotation, governed=False)
    gov = await _received_context_type(annotation, governed=True)

    assert gov is plain
