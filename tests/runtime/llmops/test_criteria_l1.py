"""L1: 観点オブジェクト（`Criterion`）と組込みファクトリの純検証（外部依存なし）。

各ファクトリが name / metric / knockout / requires / フラグを設計表どおりに組むことと、
`default_criteria` の標準品質セットを検証する。DeepEval / agents 非依存。
"""

from __future__ import annotations

import pytest

from oai_agentspec.runtime.llmops import (
    ApprovalGate,
    Conciseness,
    Criterion,
    Faithfulness,
    GEval,
    HandoffRoute,
    Relevance,
    Safety,
    ToolUse,
)
from oai_agentspec.runtime.llmops.criteria import MetricId, default_criteria

pytestmark = pytest.mark.unit


def test_relevance_factory() -> None:
    """Relevance は AnswerRelevancy・既定 knockout なし。"""
    c = Relevance()
    assert c.name == "relevance"
    assert c.metric is MetricId.ANSWER_RELEVANCY
    assert c.knockout is False
    assert c.requires == frozenset()
    assert Relevance(knockout=True).knockout is True


def test_safety_factory() -> None:
    """Safety は G-Eval・既定 knockout あり・rubric 任意。"""
    c = Safety()
    assert c.name == "safety"
    assert c.metric is MetricId.G_EVAL
    assert c.knockout is True
    assert c.rubric is None
    assert Safety(rubric="安全か", knockout=False).rubric == "安全か"


def test_conciseness_factory() -> None:
    """Conciseness は G-Eval・既定 knockout なし。"""
    c = Conciseness(rubric="簡潔か")
    assert c.name == "conciseness"
    assert c.metric is MetricId.G_EVAL
    assert c.knockout is False
    assert c.rubric == "簡潔か"


def test_faithfulness_factory() -> None:
    """Faithfulness は Faithfulness・既定 knockout あり・reference_context 必須。"""
    c = Faithfulness()
    assert c.name == "factual_grounding"
    assert c.metric is MetricId.FAITHFULNESS
    assert c.knockout is True
    assert c.requires == frozenset({"reference_context"})


def test_geval_factory() -> None:
    """GEval は指定 name・G-Eval・rubric 必須・既定 knockout なし。"""
    c = GEval("politeness", "丁寧か")
    assert c.name == "politeness"
    assert c.metric is MetricId.G_EVAL
    assert c.rubric == "丁寧か"
    assert c.knockout is False
    assert GEval("x", "r", knockout=True).knockout is True


def test_tooluse_factory() -> None:
    """ToolUse は ToolCorrectnessMetric・expected_tools 必須（能力ゲートは持たない）。"""
    c = ToolUse()
    assert c.name == "tool_correctness"
    assert c.metric is MetricId.TOOL_CORRECTNESS
    assert c.requires == frozenset({"expected_tools"})
    assert c.deterministic is False


def test_handoff_route_factory() -> None:
    """HandoffRoute は metric=None・deterministic・expected_route 必須（横断ゲートは持たない）。"""
    c = HandoffRoute()
    assert c.name == "handoff_correctness"
    assert c.metric is None
    assert c.deterministic is True
    assert c.requires == frozenset({"expected_route"})


def test_approval_gate_factory() -> None:
    """ApprovalGate は metric=None・deterministic・expected_approvals 必須（実行ゼロのゲート）。"""
    c = ApprovalGate()
    assert c.name == "approval_gate"
    assert c.metric is None
    assert c.deterministic is True
    assert c.knockout is False
    assert c.requires == frozenset({"expected_approvals"})
    assert ApprovalGate(knockout=True).knockout is True


def test_criterion_is_frozen() -> None:
    """Criterion は frozen（属性代入不可）。"""
    c = Relevance()
    with pytest.raises(AttributeError):
        c.name = "other"  # type: ignore[misc]


def test_default_criteria_is_standard_quality_set() -> None:
    """default_criteria は relevance / safety / conciseness / factual_grounding の 4 観点。"""
    names = [c.name for c in default_criteria()]
    assert names == ["relevance", "safety", "conciseness", "factual_grounding"]
    assert all(isinstance(c, Criterion) for c in default_criteria())
