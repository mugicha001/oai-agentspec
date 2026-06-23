"""L2: AGT ガバナンス統合の実 SDK 型統合検証（実 `function_tool` + registry + Runner）。

`AgentRegistry(agent_builder=GovernedAgentBuilder(...))` 経由で build した実 Agent に対し、
許可ツールの実行 / 拒否ツールの実関数非実行（`PolicyViolationError`）/ 監査ログの allow・deny
記録と `verify_chain()` / 既存 `spec.hooks` の合成委譲 / 非 FunctionTool 素通しと元 spec・tool の
非破壊性 / 既定 sink の build 間共有（agent_id 跨ぎのチェーン連続）を検証する。

SDK / AGT のバージョン耐性トリップワイヤを兼ねる: SDK `AgentHooksBase` の public ライフサイクル
メソッド集合（増えたら `_AuditAgentHooks` の委譲漏れを検知）と、AGT `GovernancePolicy.check_tool /
check_content`・`AuditLog.record / get_entries / verify_chain`・`PolicyViolationError` の存在 /
シグネチャを固定する。FakeModel で出力を制御し実 LLM を呼ばない（決定的）。
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings
from typing import Any

import pytest

pytest.importorskip(
    "openai_agents_trust", reason="governance extra（agent-governance-toolkit）未導入"
)

from agents import FunctionTool, Runner  # noqa: E402
from agents.lifecycle import AgentHooksBase  # noqa: E402
from agents.tool_context import ToolContext  # noqa: E402
from openai_agents_trust import AuditLog, GovernancePolicy  # noqa: E402

from oai_agentspec import AgentRegistry, AgentSpec, function_tool  # noqa: E402
from oai_agentspec._adapters import govern_spec, new_audit_sink  # noqa: E402
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

    集合が増えた場合、`_make_audit_hooks` の `_AuditAgentHooks` に委譲メソッドを追加しないと
    既存 `spec.hooks` の新メソッドが黙って失われる（委譲漏れの検知）。
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
        "_adapters/governance.py の _AuditAgentHooks の委譲を追従させること。"
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
