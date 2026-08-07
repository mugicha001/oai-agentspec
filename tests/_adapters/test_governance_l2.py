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

import dataclasses
import inspect
import re
import warnings
from typing import Any

import pytest

pytest.importorskip(
    "openai_agents_trust", reason="governance extra（agent-governance-toolkit）未導入"
)
# `mcp` は openai-agents の無条件依存だが、実 MCP ツール生成経路の pin が
# `mcp.types.Tool` に依存するため明示的にガードする。
pytest.importorskip("mcp", reason="mcp（openai-agents の依存）未導入")

import agents.tool as sdk_tool  # noqa: E402
from agents import FunctionTool, Runner, ToolOrigin, ToolOriginType, UserError  # noqa: E402
from agents.lifecycle import AgentHooksBase, RunHooksBase  # noqa: E402
from agents.mcp import MCPServer  # noqa: E402
from agents.mcp.util import MCPUtil  # noqa: E402
from agents.tool import get_function_tool_origin  # noqa: E402
from agents.tool_context import ToolContext  # noqa: E402
from mcp.types import CallToolResult, GetPromptResult, ListPromptsResult  # noqa: E402
from mcp.types import Tool as MCPTool  # noqa: E402
from openai_agents_trust import AuditLog, GovernancePolicy  # noqa: E402

from oai_agentspec import AgentRegistry, AgentSpec, function_tool  # noqa: E402
from oai_agentspec._adapters import govern_spec, new_audit_sink  # noqa: E402
from oai_agentspec._adapters import governance as governance_module  # noqa: E402
from oai_agentspec._adapters.governance import _make_audit_hooks  # noqa: E402
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
    併せて確認する。
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
    with pytest.raises(PolicyViolationError, match="read"):
        await deny_hooks.on_tool_start(ctx, _Named("bot"), tool)
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
    assert invoked == []  # 実関数へは到達しない
    # per-call の `tool:` レコードはラップ内で記録されるため残る（消えるのはフック由来の記録のみ）。
    assert [(e.agent_id, e.action, e.decision) for e in tool_sink.get_entries()] == [
        ("bot", "tool:echo", "deny"),
    ]
