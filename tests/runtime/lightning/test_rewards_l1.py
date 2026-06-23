"""L1: reward ファクトリ群（`contains` / `exact` / `tool_match` / `approval_match` / `judge`）検証。

`RolloutResult`（plain 観測）・`_case_value`（dict / 属性両対応）・各 reward の包含 / 完全一致 /
ツール包含（recall）/ 承認ゲート発火包含（recall）・期待非在で 0.0 を網羅する。`judge` は async で
`_adapters.judge_score` 経由（使用箇所パス `oai_agentspec._adapters.judge_score` を monkeypatch・
実 LLM を呼ばない）。純データ操作 + adapter モックで外部実通信なし（`@pytest.mark.unit`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from oai_agentspec.runtime.lightning import (
    OptimizeCase,
    RolloutResult,
    approval_match,
    contains,
    exact,
    judge,
    last_agent_match,
    route_match,
    tool_match,
)
from oai_agentspec.runtime.lightning.rewards import _case_value

pytestmark = pytest.mark.unit


def _result(
    case: Any,
    output: str = "",
    tool_calls: list[str] | None = None,
    fired_approvals: list[str] | None = None,
    route_steps: list[str] | None = None,
    last_agent: str | None = None,
) -> RolloutResult:
    """テスト用 RolloutResult を組む。"""
    return RolloutResult(
        case=case,
        output=output,
        tool_calls=list(tool_calls or []),
        fired_approvals=list(fired_approvals or []),
        route_steps=list(route_steps or []),
        last_agent=last_agent,
    )


# ----------------------------------------------------------------------
# RolloutResult / _case_value
# ----------------------------------------------------------------------


def test_rollout_result_defaults() -> None:
    """RolloutResult の plain 観測フィールドの既定値。"""
    result = RolloutResult(case={}, output="out")
    assert result.tool_calls == []
    assert result.fired_approvals == []
    assert result.route_steps == []
    assert result.last_agent is None
    assert result.output == "out"


def test_case_value_dict_access() -> None:
    """dict ケースは `case[field]` を返す（未在キーは None）。"""
    assert _case_value({"expected": "x"}, "expected") == "x"
    assert _case_value({"expected": "x"}, "missing") is None


def test_case_value_attribute_access() -> None:
    """属性ケースは `getattr(case, field, None)` を返す（未在属性は None）。"""

    @dataclass
    class _Case:
        expected: str

    assert _case_value(_Case(expected="y"), "expected") == "y"
    assert _case_value(_Case(expected="y"), "missing") is None


# ----------------------------------------------------------------------
# contains
# ----------------------------------------------------------------------


def test_contains_hit_returns_one() -> None:
    """期待文字列が出力に含まれれば 1.0。"""
    reward = contains("expected")
    assert reward(_result({"expected": "cat"}, output="a cat sat")) == 1.0


def test_contains_miss_returns_zero() -> None:
    """期待文字列が出力に含まれなければ 0.0。"""
    reward = contains("expected")
    assert reward(_result({"expected": "dog"}, output="a cat sat")) == 0.0


def test_contains_none_expected_returns_zero() -> None:
    """期待フィールド非在（None）は 0.0（fail-closed）。"""
    reward = contains("expected")
    assert reward(_result({}, output="anything")) == 0.0


def test_contains_coerces_expected_to_str() -> None:
    """非 str の期待値は str 化して包含判定する。"""
    reward = contains("expected")
    assert reward(_result({"expected": 42}, output="value is 42 here")) == 1.0


# ----------------------------------------------------------------------
# exact
# ----------------------------------------------------------------------


def test_exact_match_returns_one() -> None:
    """出力が期待と完全一致（前後空白は strip）すれば 1.0。"""
    reward = exact("expected")
    assert reward(_result({"expected": "answer"}, output="  answer  ")) == 1.0


def test_exact_mismatch_returns_zero() -> None:
    """出力が期待と一致しなければ 0.0。"""
    reward = exact("expected")
    assert reward(_result({"expected": "answer"}, output="other")) == 0.0


def test_exact_none_expected_returns_zero() -> None:
    """期待フィールド非在は 0.0。"""
    reward = exact("expected")
    assert reward(_result({}, output="answer")) == 0.0


# ----------------------------------------------------------------------
# tool_match
# ----------------------------------------------------------------------


def test_tool_match_all_expected_present_returns_one() -> None:
    """期待ツールが全て呼ばれていれば 1.0（余分な呼び出しは無視）。"""
    reward = tool_match("expected_tools")
    result = _result({"expected_tools": ["search", "fetch"]}, tool_calls=["search", "fetch", "log"])
    assert reward(result) == 1.0


def test_tool_match_missing_expected_returns_zero() -> None:
    """期待ツールに欠落があれば 0.0（recall 不足）。"""
    reward = tool_match("expected_tools")
    result = _result({"expected_tools": ["search", "fetch"]}, tool_calls=["search"])
    assert reward(result) == 0.0


def test_tool_match_empty_expected_returns_zero() -> None:
    """期待ツールが空 / 非在なら 0.0（採点対象外を低評価で扱う）。"""
    reward = tool_match("expected_tools")
    assert reward(_result({"expected_tools": []}, tool_calls=["search"])) == 0.0
    assert reward(_result({}, tool_calls=["search"])) == 0.0


# ----------------------------------------------------------------------
# approval_match（承認ゲート発火の recall）
# ----------------------------------------------------------------------


def test_approval_match_all_expected_fired_returns_one() -> None:
    """期待承認ゲートが全て発火していれば 1.0（余分な発火は無視）。"""
    reward = approval_match("expected_approvals")
    result = _result(
        {"expected_approvals": ["delete_account", "wire_money"]},
        fired_approvals=["delete_account", "wire_money", "send_email"],
    )
    assert reward(result) == 1.0


def test_approval_match_missing_expected_returns_zero() -> None:
    """期待承認ゲートに欠落があれば 0.0（recall 不足）。"""
    reward = approval_match("expected_approvals")
    result = _result(
        {"expected_approvals": ["delete_account", "wire_money"]},
        fired_approvals=["delete_account"],
    )
    assert reward(result) == 0.0


def test_approval_match_empty_expected_returns_zero() -> None:
    """期待承認ゲートが空 / 非在なら 0.0（採点対象外を低評価で扱う）。"""
    reward = approval_match("expected_approvals")
    assert reward(_result({"expected_approvals": []}, fired_approvals=["x"])) == 0.0
    assert reward(_result({}, fired_approvals=["x"])) == 0.0


def test_approval_match_no_fired_returns_zero() -> None:
    """承認ゲートが 1 件も発火していなければ 0.0（期待非空時）。"""
    reward = approval_match("expected_approvals")
    result = _result({"expected_approvals": ["danger"]}, fired_approvals=[])
    assert reward(result) == 0.0


def test_approval_match_independent_from_tool_calls() -> None:
    """approval_match は fired_approvals のみを見る（tool_calls の有無は無関係）。"""
    reward = approval_match("expected_approvals")
    # tool_calls が一致しても fired_approvals が欠ければ 0.0。
    result = _result(
        {"expected_approvals": ["danger"]},
        tool_calls=["danger"],
        fired_approvals=[],
    )
    assert reward(result) == 0.0


# ----------------------------------------------------------------------
# route_match（経路フルパスの完全一致）
# ----------------------------------------------------------------------


def test_route_match_exact_match_returns_one() -> None:
    """期待ルートと観測 route_steps が完全一致なら 1.0。"""
    reward = route_match("expected_route")
    result = _result(
        {"expected_route": ["triage", "billing"]},
        route_steps=["triage", "billing"],
    )
    assert reward(result) == 1.0


def test_route_match_order_mismatch_returns_zero() -> None:
    """順序が異なれば 0.0（経路は順序保持）。"""
    reward = route_match("expected_route")
    result = _result(
        {"expected_route": ["triage", "billing"]},
        route_steps=["billing", "triage"],
    )
    assert reward(result) == 0.0


def test_route_match_length_mismatch_returns_zero() -> None:
    """長さが異なれば 0.0（経由回数まで完全一致）。"""
    reward = route_match("expected_route")
    result = _result(
        {"expected_route": ["triage"]},
        route_steps=["triage", "billing"],
    )
    assert reward(result) == 0.0


def test_route_match_empty_expected_returns_zero() -> None:
    """期待ルートが空 / 非在なら 0.0。"""
    reward = route_match("expected_route")
    assert reward(_result({"expected_route": []}, route_steps=["triage"])) == 0.0
    assert reward(_result({}, route_steps=["triage"])) == 0.0


def test_route_match_single_agent_path() -> None:
    """単体応答（handoff なし）の `["triage"]` 一致も 1.0。"""
    reward = route_match("expected_route")
    result = _result({"expected_route": ["triage"]}, route_steps=["triage"])
    assert reward(result) == 1.0


# ----------------------------------------------------------------------
# last_agent_match（最終応答 agent の一致）
# ----------------------------------------------------------------------


def test_last_agent_match_hit_returns_one() -> None:
    """期待 last_agent と観測 last_agent が一致すれば 1.0。"""
    reward = last_agent_match("expected_last_agent")
    result = _result({"expected_last_agent": "billing"}, last_agent="billing")
    assert reward(result) == 1.0


def test_last_agent_match_miss_returns_zero() -> None:
    """期待 last_agent と観測 last_agent が異なれば 0.0。"""
    reward = last_agent_match("expected_last_agent")
    result = _result({"expected_last_agent": "billing"}, last_agent="support")
    assert reward(result) == 0.0


def test_last_agent_match_none_observed_returns_zero() -> None:
    """observation の last_agent が None（中断時）なら 0.0。"""
    reward = last_agent_match("expected_last_agent")
    result = _result({"expected_last_agent": "billing"}, last_agent=None)
    assert reward(result) == 0.0


def test_last_agent_match_empty_expected_returns_zero() -> None:
    """期待 last_agent が空 / 非在なら 0.0。"""
    reward = last_agent_match("expected_last_agent")
    assert reward(_result({"expected_last_agent": ""}, last_agent="billing")) == 0.0
    assert reward(_result({}, last_agent="billing")) == 0.0


# ----------------------------------------------------------------------
# judge（async・_adapters.judge_score 経由）
# ----------------------------------------------------------------------


async def test_judge_delegates_to_judge_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """judge reward は `_adapters.judge_score` を rubric / model / output / case 付きで呼ぶ。"""
    captured: dict[str, Any] = {}

    async def _fake_judge_score(*, rubric: str, model: Any, output: str, case: Any) -> float:
        captured.update(rubric=rubric, model=model, output=output, case=case)
        return 0.42

    monkeypatch.setattr("oai_agentspec._adapters.judge_score", _fake_judge_score, raising=True)

    model = object()
    reward = judge("be concise", model)
    score = await reward(_result({"id": 1}, output="answer text"))

    assert score == pytest.approx(0.42)
    assert captured == {
        "rubric": "be concise",
        "model": model,
        "output": "answer text",
        "case": {"id": 1},
    }


# ----------------------------------------------------------------------
# 既定 field（OptimizeCase 標準フィールド名）でファクトリを呼ぶ経路
# ----------------------------------------------------------------------


def test_contains_default_field_uses_expected_output() -> None:
    """contains() は既定で expected_output を参照する（OptimizeCase 用）。"""
    case = OptimizeCase(input="hi", expected_output="cat")
    assert contains()(_result(case, output="a cat sat")) == 1.0
    # 期待非在（expected_output=None）は 0.0。
    assert contains()(_result(OptimizeCase(input="hi"), output="cat")) == 0.0


def test_exact_default_field_uses_expected_output() -> None:
    """exact() は既定で expected_output を参照する（OptimizeCase 用）。"""
    case = OptimizeCase(input="hi", expected_output="answer")
    assert exact()(_result(case, output="  answer  ")) == 1.0
    assert exact()(_result(case, output="other")) == 0.0


def test_tool_match_default_field_uses_expected_tools() -> None:
    """tool_match() は既定で expected_tools を参照する（OptimizeCase 用）。"""
    case = OptimizeCase(input="hi", expected_tools=["search", "fetch"])
    assert tool_match()(_result(case, tool_calls=["search", "fetch", "log"])) == 1.0
    assert tool_match()(_result(case, tool_calls=["search"])) == 0.0


def test_route_match_default_field_uses_expected_route() -> None:
    """route_match() は既定で expected_route を参照する（OptimizeCase 用）。"""
    case = OptimizeCase(input="hi", expected_route=["triage", "billing"])
    assert route_match()(_result(case, route_steps=["triage", "billing"])) == 1.0
    assert route_match()(_result(case, route_steps=["triage"])) == 0.0


def test_last_agent_match_default_field_uses_expected_last_agent() -> None:
    """last_agent_match() は既定で expected_last_agent を参照する（OptimizeCase 用）。"""
    case = OptimizeCase(input="hi", expected_last_agent="billing")
    assert last_agent_match()(_result(case, last_agent="billing")) == 1.0
    assert last_agent_match()(_result(case, last_agent="support")) == 0.0


def test_approval_match_default_field_uses_expected_approvals() -> None:
    """approval_match() は既定で expected_approvals を参照する（OptimizeCase 用）。"""
    case = OptimizeCase(input="hi", expected_approvals=["delete_account"])
    assert approval_match()(_result(case, fired_approvals=["delete_account"])) == 1.0
    assert approval_match()(_result(case, fired_approvals=["other"])) == 0.0


def test_reward_factories_with_optimize_case_attribute_access() -> None:
    """OptimizeCase は属性アクセス経路（_case_value）で reward に取り込まれる（dict 化不要）。"""
    case = OptimizeCase(
        input="複合ケース",
        expected_output="期待",
        expected_tools=["t1"],
        expected_route=["a", "b"],
        expected_last_agent="b",
        expected_approvals=["g1"],
    )
    result = _result(
        case,
        output="期待される応答",
        tool_calls=["t1"],
        route_steps=["a", "b"],
        last_agent="b",
        fired_approvals=["g1"],
    )
    assert contains()(result) == 1.0
    assert tool_match()(result) == 1.0
    assert route_match()(result) == 1.0
    assert last_agent_match()(result) == 1.0
    assert approval_match()(result) == 1.0


def test_explicit_field_arg_overrides_default() -> None:
    """`field=` 明示時は既定でなくその名前で dict キーを引く（後方互換）。"""
    case = {"my_expected": "期待"}
    assert contains("my_expected")(_result(case, output="期待される応答")) == 1.0
    # 既定 field を使うと dict には expected_output が無いので 0.0。
    assert contains()(_result(case, output="期待される応答")) == 0.0
