"""L2: per-agent ポリシー（overrides）の実 AGT 統合検証（registry + 実 govern ラップ）。

同一ツールを持つ 2 エージェントが overrides により allow / deny で分かれること（per-agent の核心
価値）、既定ポリシーへのフォールバック、監査 sink の 1 本共有、override 値の異常系が既定 `policy`
と同一の fail-fast 検証エラーになること（FileNotFoundError / ValueError / TypeError）を、実 AGT
（`agt_symbols`・extra 未導入環境では skip）で検証する。

ツール実行は SDK `Runner` を介さず `on_invoke_tool(ToolContext, 引数JSON)` を直接呼ぶ
（`tests/_adapters/test_governance_l2.py` と同じ流儀・ポリシー評価は govern ラップ内で動く）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from agents.tool_context import ToolContext

from oai_agentspec import AgentRegistry, AgentSpec, function_tool
from oai_agentspec.runtime.governance import GovernedAgentBuilder

pytestmark = pytest.mark.integration


def _make_tool(name: str) -> Any:
    """govern ラップ対象の実 `FunctionTool` を作る。"""

    @function_tool(name_override=name)
    def _tool(text: str) -> str:
        """エコーする。"""
        return f"{name}:{text}"

    return _tool


def _ctx(tool_name: str, arguments: str) -> ToolContext:
    """`on_invoke_tool` 直接呼び出し用の `ToolContext` を作る。"""
    return ToolContext(
        context=None, tool_name=tool_name, tool_call_id="c1", tool_arguments=arguments
    )


@pytest.fixture
def split_registry(
    agt_symbols: tuple[Any, Any, Any],
) -> tuple[AgentRegistry, GovernedAgentBuilder, Any]:
    """同一ツール群の 2 エージェントへ既定 / override の異なるポリシーを適用した registry。

    Returns:
        `(registry, builder, PolicyViolationError)`。
    """
    governance_policy, _, policy_violation_error = agt_symbols
    builder = GovernedAgentBuilder(
        # 既定: lookup のみ許可（triage はこちらへフォールバック）。
        policy=governance_policy(name="readonly", allowed_tools=["lookup"]),
        # support だけ refund も許可。
        overrides={
            "support": governance_policy(name="support", allowed_tools=["lookup", "refund"])
        },
    )
    registry = AgentRegistry(agent_builder=builder)
    for name in ("triage", "support"):
        registry.register(
            AgentSpec(
                name=name,
                instructions="x",
                tools=[_make_tool("lookup"), _make_tool("refund")],
            )
        )
    return registry, builder, policy_violation_error


async def test_same_tool_split_allow_and_deny_per_agent(
    split_registry: tuple[AgentRegistry, GovernedAgentBuilder, Any],
) -> None:
    """同一 `refund` ツールが support では実行され、triage では実行前に拒否される。"""
    registry, _, policy_violation_error = split_registry
    support_tools = {t.name: t for t in registry.get("support").tools}
    triage_tools = {t.name: t for t in registry.get("triage").tools}
    args = '{"text": "A123"}'

    # support（override）: refund は allow され実関数が動く。
    result = await support_tools["refund"].on_invoke_tool(_ctx("refund", args), args)
    assert "refund:" in str(result)

    # triage（既定へフォールバック）: 同じ refund が実行前に deny される。
    with pytest.raises(policy_violation_error, match="refund"):
        await triage_tools["refund"].on_invoke_tool(_ctx("refund", args), args)

    # lookup は両ポリシーの allowlist に載っており双方 allow。
    assert "lookup:" in str(await triage_tools["lookup"].on_invoke_tool(_ctx("lookup", args), args))
    assert "lookup:" in str(
        await support_tools["lookup"].on_invoke_tool(_ctx("lookup", args), args)
    )


async def test_audit_sink_shared_across_override_and_default(
    split_registry: tuple[AgentRegistry, GovernedAgentBuilder, Any],
) -> None:
    """overrides 使用時も監査 sink は 1 本共有され、両エージェントの記録が同一チェーンに並ぶ。"""
    registry, builder, policy_violation_error = split_registry
    support_tools = {t.name: t for t in registry.get("support").tools}
    triage_tools = {t.name: t for t in registry.get("triage").tools}
    args = '{"text": "A123"}'

    await support_tools["refund"].on_invoke_tool(_ctx("refund", args), args)
    with pytest.raises(policy_violation_error):
        await triage_tools["refund"].on_invoke_tool(_ctx("refund", args), args)

    sink = builder.audit_sink
    decisions = {(e.agent_id, e.action, e.decision) for e in sink.get_entries()}
    assert ("support", "tool:refund", "allow") in decisions
    assert ("triage", "tool:refund", "deny") in decisions
    assert sink.verify_chain() is True
    # 全エージェント build 済みで overrides の未適用キーは残らない。
    assert builder.unapplied_overrides == frozenset()


def test_override_value_error_equivalence_with_default_policy(
    agt_symbols: tuple[Any, Any, Any],
    tmp_path: Path,
) -> None:
    """override 値の異常系は既定 `policy` と同一の fail-fast 検証エラーで build 時に拒否される。"""
    spec = AgentSpec(name="bot", instructions="x")

    def _build_with_override(value: Any) -> None:
        GovernedAgentBuilder(policy=object(), overrides={"bot": value}).build(spec)

    # 存在しない YAML パス -> FileNotFoundError。
    with pytest.raises(FileNotFoundError):
        _build_with_override(str(tmp_path / "missing.yaml"))

    # 未知キーを含む YAML -> ValueError（typo footgun の fail-fast）。
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("allowed_tool:\n  - lookup\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未知のキー"):
        _build_with_override(str(bad_yaml))

    # 評価メソッド欠如オブジェクト / None -> TypeError（既定へ戻す意図はキー削除で表現）。
    with pytest.raises(TypeError, match="check_tool"):
        _build_with_override(object())
    with pytest.raises(TypeError, match="check_tool"):
        _build_with_override(None)


async def test_from_yaml_bundle_equivalent_to_constructor(
    agt_symbols: tuple[Any, Any, Any],
    tmp_path: Path,
) -> None:
    """bundle YAML からの構築は通常コンストラクタと等価（既定 / override の引き当て・検知）。"""
    _, _, policy_violation_error = agt_symbols
    bundle = tmp_path / "governance.yaml"
    bundle.write_text(
        "default:\n"
        "  allowed_tools: [lookup]\n"
        "agents:\n"
        "  support:\n"
        "    allowed_tools: [lookup, refund]\n",
        encoding="utf-8",
    )
    builder = GovernedAgentBuilder.from_yaml(bundle)

    # bundle の agents キーは構築直後すべて未適用（通常コンストラクタの overrides と同じ意味論）。
    assert builder.unapplied_overrides == frozenset({"support"})

    registry = AgentRegistry(agent_builder=builder)
    for name in ("triage", "support"):
        registry.register(
            AgentSpec(
                name=name,
                instructions="x",
                tools=[_make_tool("lookup"), _make_tool("refund")],
            )
        )

    # support は override（refund 許可）・triage は default（lookup のみ）で強制される。
    support_refund = {t.name: t for t in registry.get("support").tools}["refund"]
    triage_refund = {t.name: t for t in registry.get("triage").tools}["refund"]
    args = '{"text": "A"}'

    assert "refund:" in str(await support_refund.on_invoke_tool(_ctx("refund", args), args))
    with pytest.raises(policy_violation_error, match="refund"):
        await triage_refund.on_invoke_tool(_ctx("refund", args), args)

    # 全エージェント build 後は未適用キーなし・sink は 1 本共有。
    assert builder.unapplied_overrides == frozenset()
    assert builder.audit_sink is not None


def test_yaml_policy_resolved_once_and_snapshotted(
    agt_symbols: tuple[Any, Any, Any],
    tmp_path: Path,
) -> None:
    """YAML パスのポリシーは初回 build で解決・スナップショットされ、以降再読込しない。

    build ごとの再読込は同一 registry 解決内のエージェント間ポリシー不整合（ファイル更新
    タイミング差）と警告の重複発火を生むため、初回解決後はファイルが壊れても・変わっても
    以降の build に影響しないことを固定する。
    """
    _, _, policy_violation_error = agt_symbols
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("allowed_tools: [lookup]\n", encoding="utf-8")
    builder = GovernedAgentBuilder(policy=str(policy_path))
    registry = AgentRegistry(agent_builder=builder)
    for name in ("a1", "a2"):
        registry.register(AgentSpec(name=name, instructions="x", tools=[_make_tool("lookup")]))

    registry.get("a1")
    # 初回 build 後にファイルを破壊しても、スナップショット済みポリシーで build が続行できる。
    policy_path.write_text("allowed_tool: [typo]\n", encoding="utf-8")
    agent2 = registry.get("a2")

    # 元の内容（lookup のみ許可）が両エージェントに効いている。
    tools = {t.name: t for t in agent2.tools}
    assert set(tools) == {"lookup"}
