"""L2: 評価オーケストレータ（`evaluate` / `_target`）を FakeModel + judge モックで検証する。

評価対象 Agent は FakeModel（実 LLM を呼ばない）。DeepEval 採点は `_adapters.judge` /
`judge_tools` を monkeypatch で固定 `CriterionResult` 返却に差し替え、外部実通信を行わない。
観点オブジェクト（Criterion）駆動の dispatch・必要データ不足の理由付き not_applicable・criteria
に挙げた観点のみ評価（自動付与なし）・単体 / 横断（HandoffGraph / WorkflowGraph + registry）の
verdict 算出・逐次/並列・registry 未供給エラー・未対応型 TypeError・langfuse を網羅する。
"""

from __future__ import annotations

from typing import Any

import pytest

from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph
from oai_agentspec.runtime.llmops import (
    ApprovalGate,
    CriterionResult,
    CriterionStatus,
    EvalCase,
    EvaluationConfig,
    EvaluationResult,
    Faithfulness,
    HandoffRoute,
    Relevance,
    ToolUse,
    Verdict,
    evaluate,
)
from oai_agentspec.runtime.llmops import _target as target_mod
from oai_agentspec.runtime.llmops.evaluator import _build_decisions
from oai_agentspec.workflow import END, START, WorkflowGraph

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.integration

# 観点名（CriterionResult.criterion の値）。
RELEVANCE = "relevance"
TOOL_CORRECTNESS = "tool_correctness"
HANDOFF_CORRECTNESS = "handoff_correctness"
APPROVAL_GATE = "approval_gate"


def _spec(
    name: str = "bot",
    *,
    instructions: str = "be helpful",
    tools: list[Any] | None = None,
    model: Any = None,
) -> AgentSpec:
    """FakeModel を据えた AgentSpec を作る（既定はテキスト応答 1 件）。"""
    fake = model if model is not None else FakeModel().queue_text("hello world")
    return AgentSpec(name=name, instructions=instructions, model=fake, tools=list(tools or []))


def _danger_spec(name: str = "bot", *, tool_name: str = "danger") -> AgentSpec:
    """承認必須ツール（needs_approval=True）を 1 つ持つ AgentSpec を作る。

    宣言層 mock（`mock_spec_tools`）が当該ツールを実差し替えできる（=approve 許可の根拠
    `replaced_tools` に含まれる）ようにする。実行フローは monkeypatch で fake するため本物の本体は
    呼ばれない。
    """
    from oai_agentspec import function_tool

    @function_tool(name_override=tool_name, needs_approval=True)
    def _danger(x: str) -> str:
        """承認必須ツール（テスト用・実行はモックへ差し替えられる）。"""
        return f"real:{x}"

    return _spec(name=name, tools=[_danger])


def _patch_judge(monkeypatch: pytest.MonkeyPatch, *, results: list[CriterionResult]) -> None:
    """`_adapters.judge` を固定 `CriterionResult` 列を返す fake へ差し替える。"""

    async def _fake_judge(**kwargs: Any) -> list[CriterionResult]:
        return list(results)

    monkeypatch.setattr("oai_agentspec._adapters.judge", _fake_judge, raising=True)


def _patch_judge_tools(monkeypatch: pytest.MonkeyPatch, *, result: CriterionResult) -> None:
    """`_adapters.judge_tools` を固定 `CriterionResult` を返す fake へ差し替える。"""

    async def _fake_judge_tools(**kwargs: Any) -> CriterionResult:
        return result

    monkeypatch.setattr("oai_agentspec._adapters.judge_tools", _fake_judge_tools, raising=True)


@pytest.fixture
def stub_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """judge=relevance pass を返す既定スタブ（品質系のみ）。"""
    _patch_judge(
        monkeypatch,
        results=[CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale="ok")],
    )


# ----------------------------------------------------------------------
# 単体評価（AgentSpec・mode=single）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_agent_evaluation_returns_pass(judge_config: Any, stub_judge: None) -> None:
    """単体 AgentSpec を FakeModel で評価し、relevance pass で verdict=pass を得る。"""
    result = await evaluate(
        _spec(),
        [EvalCase(input="hi")],
        judge=judge_config,
        criteria=[Relevance()],
    )
    assert isinstance(result, EvaluationResult)
    assert result.target_id == "bot"
    assert result.verdict == Verdict.PASS
    assert len(result.cases) == 1


@pytest.mark.asyncio
async def test_judge_model_direct_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """judge に model を直接渡すと内部で JudgeConfig にラップされ評価が成立する。"""
    _patch_judge(
        monkeypatch,
        results=[CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale="")],
    )
    result = await evaluate(
        _spec(),
        [EvalCase(input="hi")],
        judge=FakeModel().queue_text("ok"),  # JudgeConfig ではなく model 直接
        criteria=[Relevance()],
    )
    assert result.verdict == Verdict.PASS


@pytest.mark.asyncio
async def test_default_criteria_used_when_none(
    judge_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """criteria=None なら標準品質セット（relevance/safety/conciseness/factual_grounding）を使う。"""
    captured: dict[str, Any] = {}

    async def _fake_judge(**kwargs: Any) -> list[CriterionResult]:
        captured["specs"] = kwargs["specs"]
        return [
            CriterionResult(criterion=name, status=CriterionStatus.PASS, rationale="")
            for name, _metric, _rubric in kwargs["specs"]
        ]

    monkeypatch.setattr("oai_agentspec._adapters.judge", _fake_judge, raising=True)

    # reference_context なしなので factual_grounding は evaluator が NA とし judge へ渡らない。
    result = await evaluate(_spec(), [EvalCase(input="hi")], judge=judge_config)
    spec_names = {name for name, _m, _r in captured["specs"]}
    assert {"relevance", "safety", "conciseness"} <= spec_names
    assert "factual_grounding" not in spec_names  # 参照文脈無で NA（judge へ渡らない）
    fg = next(c for c in result.cases[0].criteria if c.criterion == "factual_grounding")
    assert fg.status == CriterionStatus.NOT_APPLICABLE
    assert "reference_context" in fg.rationale


@pytest.mark.asyncio
async def test_criterion_not_listed_produces_no_row(judge_config: Any, stub_judge: None) -> None:
    """criteria に挙げない観点（tool/handoff）は評価行も not_applicable 行も出ない（MAJOR-2）。"""
    result = await evaluate(
        _spec(),
        [EvalCase(input="hi", expected_tools=["search"], expected_route=["bot"])],
        judge=judge_config,
        criteria=[Relevance()],  # ToolUse / HandoffRoute を入れない
    )
    names = {c.criterion for c in result.cases[0].criteria}
    assert names == {RELEVANCE}
    assert TOOL_CORRECTNESS not in names
    assert HANDOFF_CORRECTNESS not in names


@pytest.mark.asyncio
async def test_tooluse_not_applicable_when_no_expected_tools(
    judge_config: Any, stub_judge: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ToolUse を入れても expected_tools 非在なら evaluator が NA を作り judge_tools を呼ばない。"""
    called = {"judge_tools": False}

    async def _fake_judge_tools(**kwargs: Any) -> CriterionResult:
        called["judge_tools"] = True
        return CriterionResult(
            criterion=TOOL_CORRECTNESS, status=CriterionStatus.PASS, rationale=""
        )

    monkeypatch.setattr("oai_agentspec._adapters.judge_tools", _fake_judge_tools, raising=True)

    result = await evaluate(
        _spec(),  # tools 無し
        [EvalCase(input="hi", expected_tools=None)],
        judge=judge_config,
        criteria=[Relevance(), ToolUse()],
    )
    assert called["judge_tools"] is False
    tool = next(c for c in result.cases[0].criteria if c.criterion == TOOL_CORRECTNESS)
    assert tool.status == CriterionStatus.NOT_APPLICABLE
    assert "expected_tools" in tool.rationale


@pytest.mark.asyncio
async def test_tooluse_evaluated_when_agent_has_no_tools(
    judge_config: Any, stub_judge: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ツール非保有でも能力ゲートで NA にせず評価する（期待ツール未呼び出しは fail）。"""
    captured: dict[str, Any] = {}

    async def _fake_judge_tools(**kwargs: Any) -> CriterionResult:
        captured.update(kwargs)
        return CriterionResult(
            criterion=TOOL_CORRECTNESS, status=CriterionStatus.FAIL, rationale="tools mismatch"
        )

    monkeypatch.setattr("oai_agentspec._adapters.judge_tools", _fake_judge_tools, raising=True)

    result = await evaluate(
        _spec(),  # tools 無し
        [EvalCase(input="hi", expected_tools=["search"])],
        judge=judge_config,
        criteria=[Relevance(), ToolUse()],
    )
    # 能力ゲートを廃したため judge_tools が呼ばれる（NA でない）。recall=0 で fail として現れる。
    assert captured["expected_tools"] == ["search"]
    tool = next(c for c in result.cases[0].criteria if c.criterion == TOOL_CORRECTNESS)
    assert tool.status == CriterionStatus.FAIL


@pytest.mark.asyncio
async def test_tooluse_scored_when_evaluable(
    judge_config: Any, stub_judge: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ツール保有 + expected_tools ありなら judge_tools が呼ばれその結果が載る。"""
    from oai_agentspec import function_tool

    @function_tool
    def search(q: str) -> str:
        """example 用ダミーツール。"""
        return "result"

    captured: dict[str, Any] = {}

    async def _fake_judge_tools(**kwargs: Any) -> CriterionResult:
        captured.update(kwargs)
        return CriterionResult(
            criterion=TOOL_CORRECTNESS, status=CriterionStatus.PASS, rationale="tools matched"
        )

    monkeypatch.setattr("oai_agentspec._adapters.judge_tools", _fake_judge_tools, raising=True)

    result = await evaluate(
        _spec(tools=[search]),
        [EvalCase(input="hi", expected_tools=["search"])],
        judge=judge_config,
        criteria=[Relevance(), ToolUse()],
    )
    # judge_tools が name / expected_tools 付きで呼ばれる（NA でない）。
    assert captured["name"] == TOOL_CORRECTNESS
    assert captured["expected_tools"] == ["search"]
    tool = next(c for c in result.cases[0].criteria if c.criterion == TOOL_CORRECTNESS)
    assert tool.status == CriterionStatus.PASS


@pytest.mark.asyncio
async def test_duplicate_criterion_names_raise(judge_config: Any) -> None:
    """同名 Criterion の重複は明示 ValueError。"""
    with pytest.raises(ValueError, match="duplicate criterion names"):
        await evaluate(
            _spec(),
            [EvalCase(input="hi")],
            judge=judge_config,
            criteria=[Relevance(), Relevance()],
        )


@pytest.mark.asyncio
async def test_interrupted_run_marks_all_criteria_inconclusive_and_fails(
    judge_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HITL/承認で中断した実行は採点せず全観点 inconclusive・verdict fail（Codex P2）。"""
    from types import SimpleNamespace

    from oai_agentspec.runtime.llmops import ObservedRoute, ObservedRun

    observation = ObservedRun(route=ObservedRoute(steps=[], last_agent="bot"), tool_calls=[])

    async def _interrupted(self: Any, agent: Any, value: Any, **kwargs: Any) -> Any:
        # 承認待ちで中断（final_output=None）。判定すべき route/tool 観点を含めても採点されない。
        return (
            SimpleNamespace(final_output=None, interrupted=True, pending=[], state=None),
            observation,
        )

    monkeypatch.setattr(
        "oai_agentspec._adapters.DefaultRunnerAdapter.run_with_observation",
        _interrupted,
        raising=True,
    )

    result = await evaluate(
        _spec(),
        [EvalCase(input="hi", expected_route=["bot"], expected_tools=["t"])],
        judge=judge_config,
        criteria=[Relevance(), HandoffRoute(), ToolUse()],
    )
    case = result.cases[0]
    assert [c.status for c in case.criteria] == [CriterionStatus.INCONCLUSIVE] * 3
    assert all("interrupted" in c.rationale for c in case.criteria)
    assert result.verdict == Verdict.FAIL


# ----------------------------------------------------------------------
# 案B: ApprovalGate（中断時の承認ゲート発火を採点・実行ゼロ）
# ----------------------------------------------------------------------


def _patch_interrupted(monkeypatch: pytest.MonkeyPatch, *, pending: list[dict[str, str]]) -> None:
    """run_with_observation を「承認待ちで中断する outcome」を返す fake へ差し替える。"""
    from oai_agentspec.runtime.llmops import ObservedRoute, ObservedRun

    observation = ObservedRun(route=ObservedRoute(steps=[], last_agent="bot"), tool_calls=[])

    async def _interrupted(self: Any, agent: Any, value: Any, **kwargs: Any) -> Any:
        from types import SimpleNamespace

        outcome = SimpleNamespace(
            final_output=None, interrupted=True, pending=list(pending), state=None
        )
        return outcome, observation

    monkeypatch.setattr(
        "oai_agentspec._adapters.DefaultRunnerAdapter.run_with_observation",
        _interrupted,
        raising=True,
    )


@pytest.mark.asyncio
async def test_approval_gate_passes_when_expected_approval_pending(
    judge_config: Any, stub_judge: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """中断時の承認待ちに expected_approvals が出ていれば ApprovalGate=pass（実行ゼロ）。"""
    _patch_interrupted(monkeypatch, pending=[{"tool_name": "danger", "call_id": "c1"}])

    result = await evaluate(
        _spec(),
        [EvalCase(input="hi", expected_approvals=["danger"])],
        judge=judge_config,
        criteria=[Relevance(), ApprovalGate()],
    )
    case = result.cases[0]
    gate = next(c for c in case.criteria if c.criterion == APPROVAL_GATE)
    assert gate.status == CriterionStatus.PASS
    # ApprovalGate 以外（Relevance）は中断のため inconclusive（混在を許容）。
    relevance = next(c for c in case.criteria if c.criterion == RELEVANCE)
    assert relevance.status == CriterionStatus.INCONCLUSIVE


@pytest.mark.asyncio
async def test_approval_gate_fails_when_expected_approval_absent(
    judge_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """中断時の承認待ちに expected_approvals が無ければ ApprovalGate=fail。"""
    _patch_interrupted(monkeypatch, pending=[{"tool_name": "other", "call_id": "c1"}])

    result = await evaluate(
        _spec(),
        [EvalCase(input="hi", expected_approvals=["danger"])],
        judge=judge_config,
        criteria=[ApprovalGate()],
    )
    gate = result.cases[0].criteria[0]
    assert gate.status == CriterionStatus.FAIL
    assert result.verdict == Verdict.FAIL


@pytest.mark.asyncio
async def test_approval_gate_not_applicable_without_expected_approvals(
    judge_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """expected_approvals 非在なら ApprovalGate は not_applicable（中断時でも NA 判定が先）。"""
    _patch_interrupted(monkeypatch, pending=[{"tool_name": "danger", "call_id": "c1"}])

    result = await evaluate(
        _spec(),
        [EvalCase(input="hi")],  # expected_approvals 無し
        judge=judge_config,
        criteria=[ApprovalGate()],
    )
    gate = result.cases[0].criteria[0]
    assert gate.status == CriterionStatus.NOT_APPLICABLE
    assert "expected_approvals" in gate.rationale


@pytest.mark.asyncio
async def test_approval_gate_records_pending_in_observation(
    judge_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """中断時の承認待ちは observation.pending_approvals / interrupted に載る（Langfuse 用）。"""
    _patch_interrupted(monkeypatch, pending=[{"tool_name": "danger", "call_id": "c1"}])

    result = await evaluate(
        _spec(),
        [EvalCase(input="hi", expected_approvals=["danger"])],
        judge=judge_config,
        criteria=[ApprovalGate()],
    )
    obs = result.cases[0].observation
    assert obs is not None
    assert obs.interrupted is True
    assert [a.tool for a in obs.pending_approvals] == ["danger"]


# ----------------------------------------------------------------------
# 案A: mock-approve（承認自動解決 → 完了採点）/ reject-resume / 安全不変条件
# ----------------------------------------------------------------------


def _patch_resolve_flow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pending: list[dict[str, str]],
    final_output: str = "done",
) -> dict[str, Any]:
    """run_with_observation=中断 / resume_with_observation=完了 を fake する（承認自動解決の流れ）。

    apply_approvals は state を要求するため、最小 fake state を outcome に載せる。apply_approvals
    に渡った decisions を captured へ記録して検証できるようにする。
    """
    from types import SimpleNamespace

    from oai_agentspec.runtime.llmops import ObservedRoute, ObservedRun

    captured: dict[str, Any] = {"decisions": None, "resumed": False}
    observation = ObservedRun(route=ObservedRoute(steps=[], last_agent="bot"), tool_calls=[])

    async def _interrupted(self: Any, agent: Any, value: Any, **kwargs: Any) -> Any:
        outcome = SimpleNamespace(
            final_output=None, interrupted=True, pending=list(pending), state=object()
        )
        return outcome, observation

    async def _resume(agent: Any, state: Any, **kwargs: Any) -> Any:
        captured["resumed"] = True
        outcome = SimpleNamespace(
            final_output=final_output, interrupted=False, pending=[], state=None
        )
        return outcome, observation

    def _apply(state: Any, decisions: list[dict[str, Any]]) -> Any:
        captured["decisions"] = decisions
        return SimpleNamespace(
            applied=[d["call_id"] for d in decisions], unknown=[], already_resolved=[]
        )

    monkeypatch.setattr(
        "oai_agentspec._adapters.DefaultRunnerAdapter.run_with_observation",
        _interrupted,
        raising=True,
    )
    monkeypatch.setattr("oai_agentspec._adapters.resume_with_observation", _resume, raising=True)
    monkeypatch.setattr("oai_agentspec._adapters.apply_approvals", _apply, raising=True)
    return captured


@pytest.mark.asyncio
async def test_mock_approve_resolves_and_scores_completion(
    judge_config: Any, stub_judge: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """approvals で approve + tool_mocks 指定なら中断を解決し完了出力を採点する（案A）。"""
    captured = _patch_resolve_flow(
        monkeypatch,
        pending=[{"tool_name": "danger", "call_id": "c1", "agent_name": "bot"}],
        final_output="final answer",
    )

    result = await evaluate(
        _danger_spec(),
        [EvalCase(input="hi")],
        judge=judge_config,
        criteria=[Relevance()],
        approvals=lambda p: True,  # 全承認 approve
        tool_mocks={"bot": {"danger": "mocked-result"}},  # agent スコープのネスト dict
    )
    case = result.cases[0]
    # 完了採点（Relevance pass）。中断由来の inconclusive ではない。
    relevance = next(c for c in case.criteria if c.criterion == RELEVANCE)
    assert relevance.status == CriterionStatus.PASS
    assert case.output == "final answer"
    assert captured["resumed"] is True
    # approve の decision が apply_approvals へ渡る。
    assert captured["decisions"] == [{"call_id": "c1", "decision": "approve"}]
    assert result.verdict == Verdict.PASS


@pytest.mark.asyncio
async def test_mock_approve_scores_approval_gate_on_completion(
    judge_config: Any, stub_judge: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """承認自動解決で完了した場合、ApprovalGate は発火した承認を pending として採点する。"""
    _patch_resolve_flow(
        monkeypatch, pending=[{"tool_name": "danger", "call_id": "c1", "agent_name": "bot"}]
    )

    result = await evaluate(
        _danger_spec(),
        [EvalCase(input="hi", expected_approvals=["danger"])],
        judge=judge_config,
        criteria=[ApprovalGate()],
        approvals=lambda p: True,
        tool_mocks={"bot": {"danger": "ok"}},
    )
    gate = result.cases[0].criteria[0]
    assert gate.status == CriterionStatus.PASS


@pytest.mark.asyncio
async def test_reject_resume_scores_rejection_response(
    judge_config: Any, stub_judge: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resolver が reject を返すと reject 注入で resume し却下後応答を採点する（ツール非実行）。"""
    captured = _patch_resolve_flow(monkeypatch, pending=[{"tool_name": "danger", "call_id": "c1"}])

    result = await evaluate(
        _spec(),
        [EvalCase(input="hi")],
        judge=judge_config,
        criteria=[Relevance()],
        approvals=lambda p: False,  # 全却下
        tool_mocks=None,  # reject はモック不要（ツールは実行されない）
    )
    # reject decision が渡る（rejection_message 付き）。
    assert captured["decisions"][0]["decision"] == "reject"
    assert "rejection_message" in captured["decisions"][0]
    assert captured["resumed"] is True
    assert result.cases[0].output == "done"


@pytest.mark.asyncio
async def test_interrupted_case_includes_fired_approvals_from_earlier_rounds(
    judge_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """先に approve・発火したゲートが、後続中断残存時の採点に反映される（Codex P2-2）。

    round1 で danger を approve → resume → round2 で別の wire ゲートが残って中断。最終的に中断
    残存だが、ApprovalGate(expected_approvals=["danger"]) は先に発火した danger を認識して pass。
    """
    from types import SimpleNamespace

    from oai_agentspec.runtime.llmops import ObservedRoute, ObservedRun

    observation = ObservedRun(route=ObservedRoute(steps=[], last_agent="bot"), tool_calls=[])

    async def _interrupted(self: Any, agent: Any, value: Any, **kwargs: Any) -> Any:
        # 初回: danger ゲート。
        outcome = SimpleNamespace(
            final_output=None,
            interrupted=True,
            pending=[{"tool_name": "danger", "call_id": "c1", "agent_name": "bot"}],
            state=object(),
        )
        return outcome, observation

    async def _resume(agent: Any, state: Any, **kwargs: Any) -> Any:
        # resume 後: 別の wire ゲートが残って中断継続（danger は既に解決済み）。
        outcome = SimpleNamespace(
            final_output=None,
            interrupted=True,
            pending=[{"tool_name": "wire", "call_id": "c2", "agent_name": "bot"}],
            state=object(),
        )
        return outcome, observation

    monkeypatch.setattr(
        "oai_agentspec._adapters.DefaultRunnerAdapter.run_with_observation",
        _interrupted,
        raising=True,
    )
    monkeypatch.setattr("oai_agentspec._adapters.resume_with_observation", _resume, raising=True)
    monkeypatch.setattr(
        "oai_agentspec._adapters.apply_approvals",
        lambda state, decisions: SimpleNamespace(
            applied=[d["call_id"] for d in decisions], unknown=[], already_resolved=[]
        ),
        raising=True,
    )

    # danger のみ approve（wire は reject されるが、本テストでは差し替え集合に danger のみ）。
    result = await evaluate(
        _danger_spec(),
        [EvalCase(input="hi", expected_approvals=["danger"])],
        judge=judge_config,
        criteria=[ApprovalGate()],
        approvals=lambda p: p.get("tool_name") == "danger",
        tool_mocks={"bot": {"danger": "ok"}},
    )
    gate = result.cases[0].criteria[0]
    # 先に発火・approve した danger を fired として認識 → pass（fired を捨てない）。
    assert gate.status == CriterionStatus.PASS
    # observation にも既発火 danger ＋残存 wire が両方載る。
    obs = result.cases[0].observation
    assert obs is not None
    fired_tools = {a.tool for a in obs.pending_approvals}
    assert fired_tools == {"danger", "wire"}
    assert obs.interrupted is True


@pytest.mark.asyncio
async def test_mock_approve_route_dedups_segment_boundary(
    judge_config: Any, stub_judge: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """単体 agent の mock-approve resume 後、route が ['bot','bot'] でなく ['bot'] になる（P2-1）。

    各 segment の route 末尾は last_agent を含むため単純連結だと ['bot','bot'] になり、経路不変の
    HandoffRoute(['bot']) が誤って fail する。segment 境界の連続重複を畳むことで pass になる。
    """
    from types import SimpleNamespace

    from oai_agentspec.runtime.llmops import ObservedRoute, ObservedRun, RouteStep

    # 各 segment の route は単体 agent の単一ステップ（末尾 = last_agent = bot）。
    seg = ObservedRun(
        route=ObservedRoute(steps=[RouteStep(agent="bot")], last_agent="bot"), tool_calls=[]
    )

    async def _interrupted(self: Any, agent: Any, value: Any, **kwargs: Any) -> Any:
        outcome = SimpleNamespace(
            final_output=None,
            interrupted=True,
            pending=[{"tool_name": "danger", "call_id": "c1", "agent_name": "bot"}],
            state=object(),
        )
        return outcome, seg

    async def _resume(agent: Any, state: Any, **kwargs: Any) -> Any:
        outcome = SimpleNamespace(final_output="done", interrupted=False, pending=[], state=None)
        return outcome, seg

    monkeypatch.setattr(
        "oai_agentspec._adapters.DefaultRunnerAdapter.run_with_observation",
        _interrupted,
        raising=True,
    )
    monkeypatch.setattr("oai_agentspec._adapters.resume_with_observation", _resume, raising=True)
    monkeypatch.setattr(
        "oai_agentspec._adapters.apply_approvals",
        lambda state, decisions: SimpleNamespace(
            applied=[d["call_id"] for d in decisions], unknown=[], already_resolved=[]
        ),
        raising=True,
    )

    result = await evaluate(
        _danger_spec(),
        [EvalCase(input="hi", expected_route=["bot"])],
        judge=judge_config,
        criteria=[HandoffRoute()],
        approvals=lambda p: True,
        tool_mocks={"bot": {"danger": "ok"}},
    )
    handoff = result.cases[0].criteria[0]
    # segment 境界の重複を畳み、観測経路は ['bot']（['bot','bot'] ではない）。
    assert handoff.status == CriterionStatus.PASS
    observed = [s.agent for s in result.cases[0].observation.route.steps]
    assert observed == ["bot"]


@pytest.mark.asyncio
async def test_approve_without_tool_mock_raises_value_error(
    judge_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """安全不変条件: approve したツールが実差し替えされていなければ ValueError（危険阻止）。"""
    _patch_resolve_flow(
        monkeypatch, pending=[{"tool_name": "danger", "call_id": "c1", "agent_name": "bot"}]
    )

    with pytest.raises(ValueError, match="モックへ差し替え"):
        await evaluate(
            _spec(),  # danger ツールを持たない（=実差し替え不能）
            [EvalCase(input="hi")],
            judge=judge_config,
            criteria=[Relevance()],
            approvals=lambda p: True,  # approve するが
            tool_mocks={
                "bot": {"danger": "x"}
            },  # キーはあるが Agent に danger ツールが無く差し替え不能
        )


# ----------------------------------------------------------------------
# 安全不変条件の直接検証（_build_decisions・apply/resume を介さない L1・(agent, tool) 単位）
# ----------------------------------------------------------------------

# 承認待ち 1 件（agent="account-agent" のツール "danger"）。
PENDING = {"tool_name": "danger", "call_id": "c1", "agent_name": "account-agent"}
REPLACED_A = frozenset({("account-agent", "danger")})


@pytest.mark.unit
def test_build_decisions_approve_without_replaced_raises() -> None:
    """approve かつ実差し替え集合に無ければ ValueError（apply/resume から独立して効く）。"""
    with pytest.raises(ValueError, match="モックへ差し替え"):
        _build_decisions([PENDING], resolver=lambda p: True, replaced_tools=frozenset())


@pytest.mark.unit
def test_build_decisions_approve_with_replaced_returns_approve_decision() -> None:
    """approve かつ実差し替え集合 (agent, tool) に在れば approve decision を返す。"""
    decisions = _build_decisions([PENDING], resolver=lambda p: True, replaced_tools=REPLACED_A)
    assert decisions == [{"call_id": "c1", "decision": "approve"}]


@pytest.mark.unit
def test_build_decisions_approve_same_tool_different_agent_raises() -> None:
    """**Codex P1 の核**: 同名ツールでも別 agent の approve は認可しない（ValueError）。

    (account-agent, danger) は mock 済みだが、承認待ちが (other-agent, danger) のとき approve は
    認可されず ValueError（同名ツールすり抜けを防ぐ）。
    """
    other_pending = {"tool_name": "danger", "call_id": "c2", "agent_name": "other-agent"}
    with pytest.raises(ValueError, match="other-agent"):
        _build_decisions([other_pending], resolver=lambda p: True, replaced_tools=REPLACED_A)


@pytest.mark.unit
def test_build_decisions_approve_empty_agent_name_raises() -> None:
    """agent 不明（空文字）の approve は認可しない（安全側・fail-closed）。"""
    no_agent = {"tool_name": "danger", "call_id": "c3", "agent_name": ""}
    with pytest.raises(ValueError, match="モックへ差し替え"):
        _build_decisions([no_agent], resolver=lambda p: True, replaced_tools=REPLACED_A)


@pytest.mark.unit
def test_build_decisions_resolver_receives_agent_name() -> None:
    """resolver に渡る pending dict は agent_name を含む（既存 lambda は壊れない・追加キー）。"""
    seen: dict[str, Any] = {}

    def _resolver(p: dict) -> bool:
        seen.update(p)
        return True

    _build_decisions([PENDING], resolver=_resolver, replaced_tools=REPLACED_A)
    assert seen.get("agent_name") == "account-agent"
    assert seen.get("tool_name") == "danger"


@pytest.mark.unit
def test_build_decisions_reject_without_replaced_is_safe() -> None:
    """reject は実差し替え集合が空でも例外なく reject decision を返す（ツール非実行で安全）。"""
    decisions = _build_decisions([PENDING], resolver=lambda p: False, replaced_tools=frozenset())
    assert decisions[0]["call_id"] == "c1"
    assert decisions[0]["decision"] == "reject"
    assert "rejection_message" in decisions[0]


@pytest.mark.asyncio
async def test_no_resolver_keeps_inconclusive_behavior(
    judge_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """approvals 未指定なら #24 挙動を維持（中断 → inconclusive → fail・後方互換）。"""
    _patch_interrupted(monkeypatch, pending=[{"tool_name": "danger", "call_id": "c1"}])

    result = await evaluate(
        _spec(),
        [EvalCase(input="hi")],
        judge=judge_config,
        criteria=[Relevance()],
        # approvals / tool_mocks を渡さない。
    )
    relevance = result.cases[0].criteria[0]
    assert relevance.status == CriterionStatus.INCONCLUSIVE
    assert result.verdict == Verdict.FAIL


def _patch_reinterrupt_flow(
    monkeypatch: pytest.MonkeyPatch, *, apply_result: Any, counts: dict[str, int]
) -> None:
    """run/resume が常に中断し続け、apply_approvals が固定 ApplyResult を返す flow を fake する。

    `counts["resume"]` に resume 回数を記録する（max_rounds 上限 / no-progress 早期 break の
    回数検証に使う）。
    """
    from types import SimpleNamespace

    from oai_agentspec.runtime.llmops import ObservedRoute, ObservedRun

    observation = ObservedRun(route=ObservedRoute(steps=[], last_agent="bot"), tool_calls=[])

    def _interrupt_outcome() -> Any:
        return SimpleNamespace(
            final_output=None,
            interrupted=True,
            pending=[{"tool_name": "danger", "call_id": "c1", "agent_name": "bot"}],
            state=object(),
        )

    async def _interrupted(self: Any, agent: Any, value: Any, **kwargs: Any) -> Any:
        return _interrupt_outcome(), observation

    async def _resume(agent: Any, state: Any, **kwargs: Any) -> Any:
        counts["resume"] += 1
        return _interrupt_outcome(), observation

    monkeypatch.setattr(
        "oai_agentspec._adapters.DefaultRunnerAdapter.run_with_observation",
        _interrupted,
        raising=True,
    )
    monkeypatch.setattr("oai_agentspec._adapters.resume_with_observation", _resume, raising=True)
    monkeypatch.setattr(
        "oai_agentspec._adapters.apply_approvals",
        lambda state, decisions: apply_result,
        raising=True,
    )


@pytest.mark.asyncio
async def test_resolve_loop_stops_at_max_rounds_when_always_reinterrupts(
    judge_config: Any, stub_judge: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """apply 成功でも resume が中断し続けると max_rounds で打ち切る（無限ループ防止）。"""
    from types import SimpleNamespace

    counts = {"resume": 0}
    # 毎ラウンド適用成功（applied 非空）させ、resume が再中断し続けるシナリオ。
    _patch_reinterrupt_flow(
        monkeypatch,
        apply_result=SimpleNamespace(applied=["c1"], unknown=[], already_resolved=[]),
        counts=counts,
    )

    result = await evaluate(
        _danger_spec(),
        [EvalCase(input="hi", expected_approvals=["danger"])],
        judge=judge_config,
        criteria=[ApprovalGate(), Relevance()],
        approvals=lambda p: True,
        tool_mocks={"bot": {"danger": "ok"}},
    )
    # 上限ちょうどで打ち切る（resume = max_rounds 回）。
    assert counts["resume"] == 5
    case = result.cases[0]
    gate = next(c for c in case.criteria if c.criterion == APPROVAL_GATE)
    assert gate.status == CriterionStatus.PASS
    relevance = next(c for c in case.criteria if c.criterion == RELEVANCE)
    assert relevance.status == CriterionStatus.INCONCLUSIVE


@pytest.mark.asyncio
async def test_resolve_loop_breaks_early_on_no_progress(
    judge_config: Any, stub_judge: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """apply で 1 件も適用できない（applied 空）と進展なしと判定し即 break（resume を呼ばない）。"""
    from types import SimpleNamespace

    counts = {"resume": 0}
    # 適用 0 件（unknown のみ）→ 進展なしで初回ラウンドで break。
    _patch_reinterrupt_flow(
        monkeypatch,
        apply_result=SimpleNamespace(applied=[], unknown=["c1"], already_resolved=[]),
        counts=counts,
    )

    result = await evaluate(
        _danger_spec(),
        [EvalCase(input="hi", expected_approvals=["danger"])],
        judge=judge_config,
        criteria=[ApprovalGate(), Relevance()],
        approvals=lambda p: True,
        tool_mocks={"bot": {"danger": "ok"}},
    )
    # 進展なしで即打ち切るため resume は 1 度も呼ばれない（空回りなし）。
    assert counts["resume"] == 0
    case = result.cases[0]
    # 中断のまま倒れる: ApprovalGate は pending 採点で pass、Relevance は inconclusive。
    gate = next(c for c in case.criteria if c.criterion == APPROVAL_GATE)
    assert gate.status == CriterionStatus.PASS
    relevance = next(c for c in case.criteria if c.criterion == RELEVANCE)
    assert relevance.status == CriterionStatus.INCONCLUSIVE


# ----------------------------------------------------------------------
# 横断評価（HandoffGraph + registry）
# ----------------------------------------------------------------------


def _cross_registry() -> AgentRegistry:
    """triage -> billing の handoff 用 specs を登録した registry を作る。"""
    reg = AgentRegistry()
    reg.register(_spec(name="triage", model=FakeModel().queue_text("routed")))
    reg.register(_spec(name="billing", model=FakeModel().queue_text("routed")))
    return reg


@pytest.mark.asyncio
async def test_cross_evaluation_route_match_pass(judge_config: Any, stub_judge: None) -> None:
    """横断評価で観測経路が expected_route と一致すれば handoff_correctness=pass。"""
    # FakeModel は handoff を実際には起こさない（triage が応答するのみ）。観測経路は ["triage"]。
    reg = _cross_registry()
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "billing")

    result = await evaluate(
        graph,
        [EvalCase(input="route me", expected_route=["triage"])],
        judge=judge_config,
        criteria=[Relevance(), HandoffRoute()],
        registry=reg,
    )
    handoff = next(c for c in result.cases[0].criteria if c.criterion == HANDOFF_CORRECTNESS)
    assert handoff.status == CriterionStatus.PASS
    assert result.target_id == "triage"


@pytest.mark.asyncio
async def test_cross_evaluation_with_tool_mocks_does_not_pollute_user_registry(
    judge_config: Any, stub_judge: None
) -> None:
    """HandoffGraph + tool_mocks の evaluate 後、利用者 registry の entry spec の handoffs が不変。

    クローンへの apply が利用者 registry を汚さないこと（Codex P2）を評価フロー全体で検証する。
    """
    from oai_agentspec import function_tool

    @function_tool(name_override="danger", needs_approval=True)
    def _danger(x: str) -> str:
        """承認必須ツール（下流側・テスト用）。"""
        return f"real:{x}"

    reg = AgentRegistry()
    reg.register(_spec(name="triage", model=FakeModel().queue_text("routed")))
    reg.register(_spec(name="ops", tools=[_danger], model=FakeModel().queue_text("routed")))
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "ops")

    triage_handoffs_before = list(reg._specs["triage"].handoffs)  # noqa: SLF001

    await evaluate(
        graph,
        [EvalCase(input="route me")],
        judge=judge_config,
        criteria=[Relevance()],
        registry=reg,
        tool_mocks={"ops": {"danger": "mock"}},
    )

    # 利用者 registry の entry spec の handoffs は評価前と不変（apply は派生 registry のみ）。
    assert reg._specs["triage"].handoffs == triage_handoffs_before  # noqa: SLF001


@pytest.mark.asyncio
async def test_cross_evaluation_tool_mocks_does_not_consume_graph_applied_srcs(
    judge_config: Any, stub_judge: None
) -> None:
    """mock 評価がグラフの `_applied_srcs` を消費せず、後の再 apply が stale handoffs を消せる。

    apply→エッジ削除→tool_mocks 評価（clone 経路）→再 apply の順で、評価の clone-apply がグラフの
    `_applied_srcs`（差分クリア用 bookkeeping）を汚さないこと（Codex P2）を behavioral に検証する。
    """
    from oai_agentspec import function_tool

    @function_tool(name_override="danger", needs_approval=True)
    def _danger(x: str) -> str:
        """承認必須ツール（下流側・テスト用）。"""
        return f"real:{x}"

    reg = AgentRegistry()
    reg.register(_spec(name="triage", model=FakeModel().queue_text("routed")))
    reg.register(_spec(name="ops", tools=[_danger], model=FakeModel().queue_text("routed")))
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "ops")

    # 1) real registry に apply（triage の handoffs と _applied_srcs をセット）。
    graph.apply(reg)
    assert reg._specs["triage"].handoffs == ["ops"]  # noqa: SLF001
    applied_srcs_before = set(graph._applied_srcs)  # noqa: SLF001

    # 2) グラフからエッジを削除（real registry の handoffs は再 apply まで stale のまま）。
    graph.edges.clear()

    # 3) tool_mocks 評価（clone 経路）。グラフの _applied_srcs を消費してはならない。
    await evaluate(
        graph,
        [EvalCase(input="route me")],
        judge=judge_config,
        criteria=[Relevance()],
        registry=reg,
        tool_mocks={"ops": {"danger": "mock"}},
    )
    # 評価前後でグラフの _applied_srcs は不変（deepcopy したグラフを apply したため）。
    assert graph._applied_srcs == applied_srcs_before  # noqa: SLF001

    # 4) 再 apply で stale handoffs が real registry から正しくクリアされる（triage -> []）。
    graph.apply(reg)
    assert reg._specs["triage"].handoffs == []  # noqa: SLF001


@pytest.mark.asyncio
async def test_cross_evaluation_route_mismatch_fail(judge_config: Any, stub_judge: None) -> None:
    """横断評価で観測経路が expected_route と不一致なら handoff_correctness=fail。"""
    reg = _cross_registry()
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "billing")

    result = await evaluate(
        graph,
        # 期待は triage->billing だが FakeModel は handoff しないため不一致。
        [EvalCase(input="route me", expected_route=["triage", "billing"])],
        judge=judge_config,
        criteria=[Relevance(), HandoffRoute()],
        registry=reg,
    )
    handoff = next(c for c in result.cases[0].criteria if c.criterion == HANDOFF_CORRECTNESS)
    assert handoff.status == CriterionStatus.FAIL
    assert result.verdict == Verdict.FAIL


@pytest.mark.asyncio
async def test_handoff_route_evaluated_on_single_target(
    judge_config: Any, stub_judge: None
) -> None:
    """単体 AgentSpec でも HandoffRoute は横断ゲートで NA にせず評価する（観測経路=最終 agent）。"""
    result = await evaluate(
        _spec(),
        [EvalCase(input="hi", expected_route=["bot"])],
        judge=judge_config,
        criteria=[Relevance(), HandoffRoute()],
    )
    handoff = next(c for c in result.cases[0].criteria if c.criterion == HANDOFF_CORRECTNESS)
    assert handoff.status == CriterionStatus.PASS
    assert "route matched" in handoff.rationale


@pytest.mark.asyncio
async def test_handoff_route_not_applicable_without_expected_route(
    judge_config: Any, stub_judge: None
) -> None:
    """横断でも expected_route 非在なら HandoffRoute は not_applicable。"""
    reg = _cross_registry()
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "billing")

    result = await evaluate(
        graph,
        [EvalCase(input="route me")],  # expected_route 無し
        judge=judge_config,
        criteria=[Relevance(), HandoffRoute()],
        registry=reg,
    )
    handoff = next(c for c in result.cases[0].criteria if c.criterion == HANDOFF_CORRECTNESS)
    assert handoff.status == CriterionStatus.NOT_APPLICABLE


@pytest.mark.asyncio
async def test_cross_evaluation_workflow_graph(judge_config: Any, stub_judge: None) -> None:
    """WorkflowGraph（FUNCTION のみ・LLM 不要）を registry 経由で横断評価する。"""
    wf = WorkflowGraph(name="pipeline")
    wf.add_function_node("upper", fn=lambda msg, ctx: str(msg).upper())
    wf.add_edge(START, "upper")
    wf.add_edge("upper", END)

    result = await evaluate(
        wf,
        [EvalCase(input="hello", expected_route=["workflow"])],
        judge=judge_config,
        criteria=[Relevance(), HandoffRoute()],
        registry=AgentRegistry(),
    )
    assert result.target_id == "workflow"
    # 横断（mode=cross）なので handoff_correctness は決定的比較される。
    handoff = next(c for c in result.cases[0].criteria if c.criterion == HANDOFF_CORRECTNESS)
    assert handoff.status in (CriterionStatus.PASS, CriterionStatus.FAIL)


@pytest.mark.asyncio
async def test_function_only_workflow_graph_without_registry_evaluates(
    judge_config: Any, stub_judge: None
) -> None:
    """関数ノードのみの WorkflowGraph は registry=None でも評価できる（AGENT ノード無）。"""
    wf = WorkflowGraph(name="pipeline")
    wf.add_function_node("upper", fn=lambda msg, ctx: str(msg).upper())
    wf.add_edge(START, "upper")
    wf.add_edge("upper", END)
    result = await evaluate(wf, [EvalCase(input="x")], judge=judge_config, criteria=[Relevance()])
    assert isinstance(result, EvaluationResult)
    assert result.target_id == "workflow"


@pytest.mark.asyncio
async def test_cross_evaluation_requires_registry(judge_config: Any) -> None:
    """横断対象（HandoffGraph）に registry 未供給なら明示 ValueError（spec 実体を持たないため）。"""
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "billing")
    with pytest.raises(ValueError, match="registry"):
        await evaluate(graph, [EvalCase(input="x")], judge=judge_config, criteria=[Relevance()])


# ----------------------------------------------------------------------
# 未対応型・並列・langfuse・knockout 集約
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsupported_target_type_raises_type_error(judge_config: Any) -> None:
    """未対応の target 型は許容型を列挙した TypeError になる。"""
    with pytest.raises(TypeError, match="AgentSpec / HandoffGraph / WorkflowGraph"):
        await evaluate(
            "not-a-target",  # type: ignore[arg-type]
            [EvalCase(input="x")],
            judge=judge_config,
            criteria=[Relevance()],
        )


@pytest.mark.asyncio
async def test_knockout_criterion_fail_forces_verdict_fail(
    judge_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """knockout 観点（Faithfulness・reference_context あり）が fail なら verdict 即 fail。"""
    _patch_judge(
        monkeypatch,
        results=[
            CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""),
            CriterionResult(
                criterion="factual_grounding", status=CriterionStatus.FAIL, rationale="ungrounded"
            ),
        ],
    )
    result = await evaluate(
        _spec(),
        [EvalCase(input="hi", reference_context=["ref"])],
        judge=judge_config,
        criteria=[Relevance(), Faithfulness()],  # Faithfulness は既定 knockout
    )
    assert result.verdict == Verdict.FAIL


@pytest.mark.asyncio
async def test_concurrency_runs_multiple_cases(judge_config: Any, stub_judge: None) -> None:
    """concurrency>1 で複数ケースを評価し全件 CaseResult が返る（並列経路）。"""
    cases = [EvalCase(input=f"q{i}") for i in range(4)]
    result = await evaluate(
        _spec(),
        cases,
        judge=judge_config,
        criteria=[Relevance()],
        config=EvaluationConfig(concurrency=4),
    )
    assert len(result.cases) == 4
    assert result.verdict == Verdict.PASS


@pytest.mark.asyncio
async def test_sequential_runs_multiple_cases(judge_config: Any, stub_judge: None) -> None:
    """concurrency=1（逐次）でも複数ケースを評価し全件返る。"""
    cases = [EvalCase(input=f"q{i}") for i in range(3)]
    result = await evaluate(
        _spec(),
        cases,
        judge=judge_config,
        criteria=[Relevance()],
        config=EvaluationConfig(concurrency=1),
    )
    assert len(result.cases) == 3


@pytest.mark.asyncio
async def test_langfuse_none_returns_local_result_without_send(
    judge_config: Any, stub_judge: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """langfuse 未指定なら langfuse_send を呼ばずローカル結果を返す。"""
    called = {"send": False}

    def _fail_send(*args: Any, **kwargs: Any) -> None:
        called["send"] = True

    monkeypatch.setattr("oai_agentspec._adapters.langfuse_send", _fail_send, raising=True)

    result = await evaluate(
        _spec(),
        [EvalCase(input="hi")],
        judge=judge_config,
        criteria=[Relevance()],
        langfuse=None,
    )
    assert called["send"] is False
    assert isinstance(result, EvaluationResult)


@pytest.mark.asyncio
async def test_langfuse_config_triggers_send(
    judge_config: Any, stub_judge: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """langfuse 指定時は langfuse_send が結果 + cases + prompt_text 付きで呼ばれる。"""
    from oai_agentspec.runtime.llmops import LangfuseConfig

    captured: dict[str, Any] = {}

    def _capture_send(result: Any, config: Any, *, cases: Any, prompt_text: Any = None) -> None:
        captured["result"] = result
        captured["config"] = config
        captured["cases"] = cases
        captured["prompt_text"] = prompt_text

    monkeypatch.setattr("oai_agentspec._adapters.langfuse_send", _capture_send, raising=True)

    spec = _spec(instructions="static prompt body")
    await evaluate(
        spec,
        [EvalCase(input="hi")],
        judge=judge_config,
        criteria=[Relevance()],
        langfuse=LangfuseConfig(prompt_name="p"),
    )
    assert isinstance(captured["result"], EvaluationResult)
    assert len(captured["cases"]) == 1
    # 静的 instructions が prompt_text として抽出される。
    assert captured["prompt_text"] == "static prompt body"


# ----------------------------------------------------------------------
# _target ユニット（normalize / target_id / extract_prompt の分岐）
# ----------------------------------------------------------------------


def test_target_id_for_agent_spec() -> None:
    """target_id は AgentSpec の name を返す。"""
    assert target_mod.target_id(_spec(name="x")) == "x"


def test_target_id_for_handoff_graph_uses_entry() -> None:
    """target_id は HandoffGraph の entry 名を返す。"""
    assert target_mod.target_id(HandoffGraph(entry="triage")) == "triage"


def test_target_id_for_handoff_graph_without_entry() -> None:
    """entry 未指定の HandoffGraph は "handoff_graph" を返す。"""
    assert target_mod.target_id(HandoffGraph()) == "handoff_graph"


def test_extract_prompt_static_instructions() -> None:
    """静的 instructions の AgentSpec はその文字列を抽出する。"""
    assert target_mod.extract_prompt(_spec(instructions="static")) == "static"


def test_extract_prompt_callable_instructions_returns_none() -> None:
    """callable instructions は抽出せず None。"""
    spec = AgentSpec(name="x", instructions=lambda ctx, agent: "dyn", model=FakeModel())
    assert target_mod.extract_prompt(spec) is None


def test_extract_prompt_with_prompt_field_returns_none() -> None:
    """prompt フィールド設定時（動的）は静的 instructions でも抽出しない。"""
    spec = AgentSpec(name="x", instructions="static", prompt=object(), model=FakeModel())
    assert target_mod.extract_prompt(spec) is None


def test_extract_prompt_handoff_graph_returns_none() -> None:
    """横断対象（HandoffGraph）は単一プロンプト不特定で None。"""
    assert target_mod.extract_prompt(HandoffGraph(entry="t")) is None


def test_normalize_agent_spec_builds_agent() -> None:
    """AgentSpec は build_agent 経由で正規化され (Agent, 空集合) を返す（mock 無し）。"""
    agent, replaced = target_mod.normalize(_spec(), None)
    assert agent is not None
    assert replaced == frozenset()


def test_normalize_handoff_graph_without_registry_raises() -> None:
    """HandoffGraph で registry=None は ValueError。"""
    with pytest.raises(ValueError, match="registry"):
        target_mod.normalize(HandoffGraph(entry="triage"), None)


def test_normalize_agent_spec_mocks_tools_without_touching_original() -> None:
    """AgentSpec + tool_mocks は spec を差し替えて build し、元 spec.tools は不変（宣言層）。"""
    from oai_agentspec import function_tool

    @function_tool(name_override="danger", needs_approval=True)
    def _danger(x: str) -> str:
        """承認必須ツール（テスト用）。"""
        return f"real:{x}"

    spec = _spec(tools=[_danger])  # name="bot"
    original_invoke = spec.tools[0].on_invoke_tool
    _agent, replaced = target_mod.normalize(spec, None, tool_mocks={"bot": {"danger": "mock"}})
    # 実差し替えした (agent, tool) ペアが集合に入る。元 spec の tool は不変（mutate しない）。
    assert replaced == frozenset({("bot", "danger")})
    assert spec.tools[0].on_invoke_tool is original_invoke


def test_normalize_handoff_graph_mocks_clone_not_user_registry() -> None:
    """HandoffGraph + tool_mocks はクローンを mock 化し、利用者 registry を汚さない（P2-1）。"""
    from oai_agentspec import function_tool
    from oai_agentspec._adapters import FunctionTool

    @function_tool(name_override="danger", needs_approval=True)
    def _danger(x: str) -> str:
        """承認必須ツール（テスト用）。"""
        return f"real:{x}"

    reg = AgentRegistry()
    reg.register(_spec(name="triage", model=FakeModel().queue_text("ok")))
    reg.register(_spec(name="ops", tools=[_danger], model=FakeModel().queue_text("ok")))
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "ops")

    # 評価前: entry spec の handoffs（apply 前なので空）。
    triage_handoffs_before = list(reg._specs["triage"].handoffs)  # noqa: SLF001

    _agent, replaced = target_mod.normalize(graph, reg, tool_mocks={"ops": {"danger": "mock"}})
    assert replaced == frozenset({("ops", "danger")})

    # 利用者 registry の ops ツールは本物のまま（mock 化されていない）。
    user_ops = reg.get("ops")
    user_tool = next(
        t for t in user_ops.tools if isinstance(t, FunctionTool) and t.name == "danger"
    )
    # 元 spec の tool オブジェクトと同一（差し替えられていない）。
    assert user_tool is reg._specs["ops"].tools[0]  # noqa: SLF001 - 不変検証
    # クローンへの apply が利用者 registry の entry spec の handoffs を汚さない（Codex P2）。
    assert reg._specs["triage"].handoffs == triage_handoffs_before  # noqa: SLF001


def test_normalize_handoff_graph_dynamic_candidate_is_mocked() -> None:
    """dynamic_edge の候補側の承認ツールもクローン経由で mock 化される（P2-2）。"""
    from oai_agentspec import function_tool

    @function_tool(name_override="danger", needs_approval=True)
    def _danger(x: str) -> str:
        """承認必須ツール（テスト用・動的候補側）。"""
        return f"real:{x}"

    reg = AgentRegistry()
    reg.register(_spec(name="triage", model=FakeModel().queue_text("ok")))
    reg.register(_spec(name="ops", tools=[_danger], model=FakeModel().queue_text("ok")))
    graph = HandoffGraph(entry="triage")
    # ops は静的 .handoffs ではなく動的候補（旧設計では到達できず未 mock だった）。
    graph.dynamic_edge("triage", ["ops"], lambda c, i: "ops", tool_name="route")

    _agent, replaced = target_mod.normalize(graph, reg, tool_mocks={"ops": {"danger": "mock"}})
    # 動的候補側のツールも mock 済み（(ops, danger) が replaced に入る → approve 可能）。
    assert replaced == frozenset({("ops", "danger")})


def test_target_id_for_workflow_graph() -> None:
    """target_id は WorkflowGraph に "workflow" を返す。"""
    wf = WorkflowGraph(name="pipeline")
    assert target_mod.target_id(wf) == "workflow"


def test_target_id_fallback_for_unknown_object() -> None:
    """未知オブジェクトは name 属性 / "target" にフォールバックする。"""

    class _Named:
        name = "custom"

    assert target_mod.target_id(_Named()) == "custom"
    assert target_mod.target_id(object()) == "target"


def test_normalize_function_only_workflow_graph_without_registry() -> None:
    """関数ノードのみの WorkflowGraph は registry=None でも正規化できる。"""
    wf = WorkflowGraph(name="pipeline")
    wf.add_function_node("upper", fn=lambda msg, ctx: str(msg))
    wf.add_edge(START, "upper")
    wf.add_edge("upper", END)
    agent, replaced = target_mod.normalize(wf, None)
    assert agent is not None
    assert replaced == frozenset()


def test_normalize_workflow_graph_with_registry() -> None:
    """WorkflowGraph は registry 供給でも正規化できる。"""
    wf = WorkflowGraph(name="pipeline")
    wf.add_function_node("upper", fn=lambda msg, ctx: str(msg))
    wf.add_edge(START, "upper")
    wf.add_edge("upper", END)
    agent, replaced = target_mod.normalize(wf, AgentRegistry())
    assert agent is not None
    assert replaced == frozenset()


def test_normalize_workflow_graph_agent_node_mocks_via_clone() -> None:
    """AGENT ノードを含む WorkflowGraph + tool_mocks はクローン registry 経由で mock 化される。"""
    from oai_agentspec import function_tool

    @function_tool(name_override="danger", needs_approval=True)
    def _danger(x: str) -> str:
        """承認必須ツール（テスト用・AGENT ノード側）。"""
        return f"real:{x}"

    reg = AgentRegistry()
    reg.register(_spec(name="worker", tools=[_danger], model=FakeModel().queue_text("ok")))
    wf = WorkflowGraph(name="pipeline")
    wf.add_agent_node("worker", agent="worker")
    wf.add_edge(START, "worker")
    wf.add_edge("worker", END)

    _agent, replaced = target_mod.normalize(wf, reg, tool_mocks={"worker": {"danger": "mock"}})
    # AGENT ノードが参照する registry agent のツールもクローン経由で mock 済み。
    assert replaced == frozenset({("worker", "danger")})


def test_normalize_unsupported_raises_type_error() -> None:
    """未対応型は TypeError。"""
    with pytest.raises(TypeError):
        target_mod.normalize(123, None)
