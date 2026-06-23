"""L1: `compute_verdict` の純ロジック網羅（FR-5・I/O なし・外部依存なし）。

母集合除外（skip / not_applicable）・knockout fail-closed・inconclusive ポリシー・missing-pair
fail-closed・全 pass・空母集合 fail-closed を決定的に検証する。DeepEval / Langfuse / agents に
非依存。
"""

from __future__ import annotations

import pytest

from oai_agentspec.runtime.llmops import (
    CaseResult,
    CriterionResult,
    CriterionStatus,
    Verdict,
)
from oai_agentspec.runtime.llmops.verdict import compute_verdict

pytestmark = pytest.mark.unit

# 観点名は CriterionResult.criterion の値（内部定数だが結果キーとして安定）。テストでは
# verdict 純ロジックの検証用に名前文字列を直接使う。
RELEVANCE = "relevance"
SAFETY = "safety"
CONCISENESS = "conciseness"
FACTUAL_GROUNDING = "factual_grounding"
TOOL_CORRECTNESS = "tool_correctness"
HANDOFF_CORRECTNESS = "handoff_correctness"


def _case(*criteria: CriterionResult) -> CaseResult:
    """観点結果列から `CaseResult` を組む（入力・出力はダミー）。"""
    return CaseResult(case_input="in", output="out", criteria=list(criteria))


def _result(criterion: str, status: CriterionStatus) -> CriterionResult:
    """指定観点・状態の `CriterionResult` を組む。"""
    return CriterionResult(criterion=criterion, status=status, rationale="")


def test_all_pass_returns_pass() -> None:
    """母集合が全 pass なら verdict は pass。"""
    case = _case(
        _result(RELEVANCE, CriterionStatus.PASS),
        _result(CONCISENESS, CriterionStatus.PASS),
    )
    assert compute_verdict([case]) == Verdict.PASS


def test_skip_and_not_applicable_excluded_from_population() -> None:
    """skip / not_applicable は母集合から除外され、残りが全 pass なら pass。"""
    case = _case(
        _result(RELEVANCE, CriterionStatus.PASS),
        _result(CONCISENESS, CriterionStatus.SKIP),
        _result(HANDOFF_CORRECTNESS, CriterionStatus.NOT_APPLICABLE),
    )
    # 母集合は relevance（pass）のみ。skip / not_applicable の fail 相当は影響しない。
    assert compute_verdict([case]) == Verdict.PASS


def test_empty_population_is_fail_closed() -> None:
    """母集合が空（全 skip / not_applicable）なら fail-closed。"""
    case = _case(
        _result(RELEVANCE, CriterionStatus.SKIP),
        _result(FACTUAL_GROUNDING, CriterionStatus.NOT_APPLICABLE),
    )
    assert compute_verdict([case]) == Verdict.FAIL


def test_no_cases_is_fail_closed() -> None:
    """ケースが 1 件も無い場合も fail-closed（母集合が空）。"""
    assert compute_verdict([]) == Verdict.FAIL


def test_knockout_safety_fail_forces_fail() -> None:
    """knockout 観点 safety が fail なら他が全 pass でも即 fail。"""
    case = _case(
        _result(RELEVANCE, CriterionStatus.PASS),
        _result(SAFETY, CriterionStatus.FAIL),
    )
    assert compute_verdict([case], knockout=frozenset({SAFETY})) == Verdict.FAIL


def test_knockout_factual_grounding_fail_forces_fail() -> None:
    """knockout 観点 factual_grounding が fail なら即 fail。"""
    case = _case(
        _result(RELEVANCE, CriterionStatus.PASS),
        _result(FACTUAL_GROUNDING, CriterionStatus.FAIL),
    )
    assert compute_verdict([case], knockout=frozenset({FACTUAL_GROUNDING})) == Verdict.FAIL


def test_knockout_not_applicable_is_not_evaluated() -> None:
    """knockout 観点が not_applicable なら判定対象外（母集合除外・残り pass で pass）。"""
    case = _case(
        _result(RELEVANCE, CriterionStatus.PASS),
        # safety は knockout だが not_applicable なので母集合から除外され fail を誘発しない。
        _result(SAFETY, CriterionStatus.NOT_APPLICABLE),
    )
    assert compute_verdict([case], knockout=frozenset({SAFETY})) == Verdict.PASS


def test_non_knockout_fail_returns_fail_without_knockout_override() -> None:
    """非 knockout 観点の fail は即時ではないが、母集合に fail が残るため最終的に fail。"""
    case = _case(
        _result(RELEVANCE, CriterionStatus.PASS),
        _result(CONCISENESS, CriterionStatus.FAIL),
    )
    assert compute_verdict([case]) == Verdict.FAIL


def test_inconclusive_policy_fail() -> None:
    """inconclusive_policy=FAIL（既定）で母集合の inconclusive は fail へ解決する。"""
    case = _case(
        _result(RELEVANCE, CriterionStatus.PASS),
        _result(CONCISENESS, CriterionStatus.INCONCLUSIVE),
    )
    assert compute_verdict([case], inconclusive_policy=Verdict.FAIL) == Verdict.FAIL


def test_inconclusive_policy_pass() -> None:
    """inconclusive_policy=PASS で母集合の inconclusive は pass へ解決する。"""
    case = _case(
        _result(RELEVANCE, CriterionStatus.PASS),
        _result(CONCISENESS, CriterionStatus.INCONCLUSIVE),
    )
    assert compute_verdict([case], inconclusive_policy=Verdict.PASS) == Verdict.PASS


def test_inconclusive_pass_does_not_mask_real_fail() -> None:
    """inconclusive_policy=PASS でも非 knockout の実 fail は隠さず fail にする。"""
    case = _case(
        _result(RELEVANCE, CriterionStatus.PASS),
        _result(CONCISENESS, CriterionStatus.FAIL),  # 非 knockout の実 fail
        _result(SAFETY, CriterionStatus.INCONCLUSIVE),
    )
    assert compute_verdict([case], inconclusive_policy=Verdict.PASS) == Verdict.FAIL


def test_inconclusive_does_not_bypass_knockout() -> None:
    """inconclusive_policy=PASS でも knockout fail は優先され fail のまま。"""
    case = _case(
        _result(SAFETY, CriterionStatus.FAIL),
        _result(CONCISENESS, CriterionStatus.INCONCLUSIVE),
    )
    assert (
        compute_verdict([case], knockout=frozenset({SAFETY}), inconclusive_policy=Verdict.PASS)
        == Verdict.FAIL
    )


def test_missing_required_criterion_is_fail_closed() -> None:
    """required_criteria が母集合に存在しないと fail-closed。"""
    case = _case(_result(RELEVANCE, CriterionStatus.PASS))
    required = frozenset({RELEVANCE, FACTUAL_GROUNDING})
    # factual_grounding が母集合に無いため missing-pair fail-closed。
    assert compute_verdict([case], required_criteria=required) == Verdict.FAIL


def test_required_criteria_subset_present_passes() -> None:
    """required_criteria が母集合の部分集合なら missing 判定なしで評価が進む。"""
    case = _case(
        _result(RELEVANCE, CriterionStatus.PASS),
        _result(CONCISENESS, CriterionStatus.PASS),
    )
    assert compute_verdict([case], required_criteria=frozenset({RELEVANCE})) == Verdict.PASS


def test_required_criteria_none_uses_present_set() -> None:
    """required_criteria=None なら母集合の観点集合を要求集合とし追加 missing 判定は起きない。"""
    case = _case(_result(RELEVANCE, CriterionStatus.PASS))
    assert compute_verdict([case], required_criteria=None) == Verdict.PASS


def test_custom_knockout_set_passed_by_caller() -> None:
    """knockout 集合を渡せて、当該観点の fail を即 fail にできる。"""
    case = _case(
        _result(RELEVANCE, CriterionStatus.PASS),
        _result(TOOL_CORRECTNESS, CriterionStatus.FAIL),
    )
    # knockout 空（既定）でも母集合に fail が残るため fail。
    assert compute_verdict([case]) == Verdict.FAIL
    # tool_correctness を knockout に加えても fail（経路は異なるが結果は fail）。
    assert compute_verdict([case], knockout=frozenset({TOOL_CORRECTNESS})) == Verdict.FAIL


def test_flat_aggregation_across_multiple_cases() -> None:
    """複数ケースの観点を flat 集計し、いずれかに fail があれば fail。"""
    pass_case = _case(_result(RELEVANCE, CriterionStatus.PASS))
    fail_case = _case(_result(CONCISENESS, CriterionStatus.FAIL))
    assert compute_verdict([pass_case, fail_case]) == Verdict.FAIL


def test_flat_aggregation_all_pass_across_cases() -> None:
    """複数ケースが全 pass なら flat 集計でも pass。"""
    c1 = _case(_result(RELEVANCE, CriterionStatus.PASS))
    c2 = _case(_result(CONCISENESS, CriterionStatus.PASS))
    assert compute_verdict([c1, c2]) == Verdict.PASS
