"""L2: `_adapters.judge` / `judge_tools` の分岐網羅（DeepEval metric をモック・実通信なし）。

DeepEval metric クラスを stub に差し替え、spec（name, metric_id, rubric）→ metric 解決・
LLMTestCase 構築・タイムアウト fail-closed・スキーマ不適合 fail-closed・DeepEval→plain
`CriterionResult` 変換・ツール採点の tools_called 構築・テレメトリ opt-out・同期 generate の
running-loop 分岐を検証する。deepeval は導入済みのため import は通る前提で、メトリクスの挙動の
みモックする。必要データ不足の not_applicable は evaluator の責務（test_evaluator_l2 で検証）。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from oai_agentspec._adapters import judge, judge_tools
from oai_agentspec._adapters.judge import _make_judge_model
from oai_agentspec.runtime.llmops import (
    CriterionStatus,
    EvaluationConfig,
    JudgeConfig,
)
from oai_agentspec.runtime.llmops.criteria import MetricId

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.integration

# 観点名（CriterionResult.criterion の値）。
RELEVANCE = "relevance"
SAFETY = "safety"
CONCISENESS = "conciseness"
TOOL_CORRECTNESS = "tool_correctness"


class _StubMetric:
    """DeepEval metric の最小スタブ（measure / a_measure / score / success / reason 固定）。"""

    def __init__(self, *, score: float, success: bool, reason: str) -> None:
        self.score = score
        self.success = success
        self.reason = reason
        self.measured: list[Any] = []

    def measure(self, test_case: Any) -> None:
        self.measured.append(test_case)

    async def a_measure(self, test_case: Any) -> None:
        self.measured.append(test_case)


def _patch_quality_metrics(monkeypatch: pytest.MonkeyPatch, *, factory: Any) -> dict[str, Any]:
    """出力品質 metric（Faithfulness / AnswerRelevancy / GEval）を factory 産の stub に差し替える。

    各 metric クラスは呼ばれた引数を `created` に記録する stub コンストラクタへ置換する。
    """
    created: dict[str, Any] = {}

    def _make(name: str) -> Any:
        def _ctor(*args: Any, **kwargs: Any) -> Any:
            metric = factory()
            created[name] = {"args": args, "kwargs": kwargs, "metric": metric}
            return metric

        return _ctor

    import deepeval.metrics as dm

    monkeypatch.setattr(dm, "FaithfulnessMetric", _make("faithfulness"), raising=True)
    monkeypatch.setattr(dm, "AnswerRelevancyMetric", _make("answer_relevancy"), raising=True)
    monkeypatch.setattr(dm, "GEval", _make("g_eval"), raising=True)
    return created


def _judge_config() -> JudgeConfig:
    """FakeModel を採点モデルにした JudgeConfig。"""
    return JudgeConfig(model=FakeModel().queue_text("ok"))


# ----------------------------------------------------------------------
# judge: spec → metric 解決・plain 変換
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_relevance_pass_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """AnswerRelevancy が success=True を返すと relevance は PASS + score 反映。"""
    _patch_quality_metrics(
        monkeypatch,
        factory=lambda: _StubMetric(score=0.9, success=True, reason="relevant"),
    )
    results = await judge(
        marked_input="q",
        marked_output="a",
        specs=[(RELEVANCE, MetricId.ANSWER_RELEVANCY, None)],
        context=None,
        judge_config=_judge_config(),
        config=EvaluationConfig(timeout_seconds=None),
    )
    assert len(results) == 1
    assert results[0].criterion == RELEVANCE
    assert results[0].status == CriterionStatus.PASS
    assert results[0].score == pytest.approx(0.9)
    assert results[0].rationale == "relevant"


@pytest.mark.asyncio
async def test_judge_geval_fail_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """GEval（safety）が success=False を返すと FAIL に変換される。"""
    _patch_quality_metrics(
        monkeypatch,
        factory=lambda: _StubMetric(score=0.2, success=False, reason="unsafe"),
    )
    results = await judge(
        marked_input="q",
        marked_output="a",
        specs=[(SAFETY, MetricId.G_EVAL, None)],
        context=None,
        judge_config=_judge_config(),
        config=EvaluationConfig(timeout_seconds=None),
    )
    assert results[0].status == CriterionStatus.FAIL
    assert results[0].score == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_judge_faithfulness_with_context_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """factual_grounding は Faithfulness で採点する（context は LLMTestCase へ載る）。"""
    _patch_quality_metrics(
        monkeypatch,
        factory=lambda: _StubMetric(score=0.8, success=True, reason="grounded"),
    )
    results = await judge(
        marked_input="q",
        marked_output="a",
        specs=[("factual_grounding", MetricId.FAITHFULNESS, None)],
        context=["ref doc"],
        judge_config=_judge_config(),
        config=EvaluationConfig(timeout_seconds=None),
    )
    assert results[0].status == CriterionStatus.PASS
    assert results[0].score == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_judge_geval_uses_rubric_from_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """G-Eval 観点は spec の rubric を criteria として渡す。"""
    created = _patch_quality_metrics(
        monkeypatch,
        factory=lambda: _StubMetric(score=1.0, success=True, reason=""),
    )
    await judge(
        marked_input="q",
        marked_output="a",
        specs=[(CONCISENESS, MetricId.G_EVAL, "Be concise rubric")],
        context=None,
        judge_config=_judge_config(),
        config=EvaluationConfig(timeout_seconds=None),
    )
    assert created["g_eval"]["kwargs"]["criteria"] == "Be concise rubric"


@pytest.mark.asyncio
async def test_judge_geval_default_criteria_when_no_rubric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rubric=None の G-Eval は観点名のみの最小定義に倒す（プロンプト非同梱）。"""
    created = _patch_quality_metrics(
        monkeypatch,
        factory=lambda: _StubMetric(score=1.0, success=True, reason=""),
    )
    await judge(
        marked_input="q",
        marked_output="a",
        specs=[(CONCISENESS, MetricId.G_EVAL, None)],
        context=None,
        judge_config=_judge_config(),
        config=EvaluationConfig(timeout_seconds=None),
    )
    assert CONCISENESS in created["g_eval"]["kwargs"]["criteria"]


def _capture_quality_test_case(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """出力品質パスの `LLMTestCase` 構築 kwarg を記録する（expected_output 検証用）。"""
    import deepeval.test_case as dt

    captured: dict[str, Any] = {}
    real_test_case = dt.LLMTestCase

    def _ctor(*args: Any, **kwargs: Any) -> Any:
        captured["kwargs"] = kwargs
        return real_test_case(*args, **kwargs)

    monkeypatch.setattr(dt, "LLMTestCase", _ctor, raising=True)
    return captured


@pytest.mark.asyncio
async def test_judge_geval_includes_expected_output_when_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """expected_output 提供時、G-Eval は EXPECTED_OUTPUT を params に含め test case にも載せる。"""
    from deepeval.test_case import LLMTestCaseParams

    created = _patch_quality_metrics(
        monkeypatch,
        factory=lambda: _StubMetric(score=1.0, success=True, reason=""),
    )
    captured = _capture_quality_test_case(monkeypatch)
    await judge(
        marked_input="q",
        marked_output="a",
        specs=[(CONCISENESS, MetricId.G_EVAL, None)],
        context=None,
        expected_output="golden answer",
        judge_config=_judge_config(),
        config=EvaluationConfig(timeout_seconds=None),
    )
    params = created["g_eval"]["kwargs"]["evaluation_params"]
    assert LLMTestCaseParams.EXPECTED_OUTPUT in params
    assert captured["kwargs"]["expected_output"] == "golden answer"


@pytest.mark.asyncio
async def test_judge_geval_excludes_expected_output_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """expected_output 未指定時、G-Eval は EXPECTED_OUTPUT を含めず test case も None。"""
    from deepeval.test_case import LLMTestCaseParams

    created = _patch_quality_metrics(
        monkeypatch,
        factory=lambda: _StubMetric(score=1.0, success=True, reason=""),
    )
    captured = _capture_quality_test_case(monkeypatch)
    await judge(
        marked_input="q",
        marked_output="a",
        specs=[(CONCISENESS, MetricId.G_EVAL, None)],
        context=None,
        judge_config=_judge_config(),
        config=EvaluationConfig(timeout_seconds=None),
    )
    params = created["g_eval"]["kwargs"]["evaluation_params"]
    assert LLMTestCaseParams.EXPECTED_OUTPUT not in params
    assert captured["kwargs"]["expected_output"] is None


@pytest.mark.asyncio
async def test_judge_multiple_specs_scored_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """複数 spec を順序どおりに採点して結果列を返す。"""
    _patch_quality_metrics(
        monkeypatch,
        factory=lambda: _StubMetric(score=1.0, success=True, reason=""),
    )
    results = await judge(
        marked_input="q",
        marked_output="a",
        specs=[
            (RELEVANCE, MetricId.ANSWER_RELEVANCY, None),
            (CONCISENESS, MetricId.G_EVAL, None),
        ],
        context=None,
        judge_config=_judge_config(),
        config=EvaluationConfig(timeout_seconds=None),
    )
    assert [r.criterion for r in results] == [RELEVANCE, CONCISENESS]


@pytest.mark.asyncio
async def test_judge_timeout_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """採点が timeout を超えると fail_closed_status へ倒す（asyncio.wait_for 経路）。"""

    class _SlowMetric(_StubMetric):
        async def a_measure(self, test_case: Any) -> None:
            await asyncio.sleep(1.0)

    _patch_quality_metrics(
        monkeypatch,
        factory=lambda: _SlowMetric(score=0.0, success=False, reason=""),
    )
    results = await judge(
        marked_input="q",
        marked_output="a",
        specs=[(RELEVANCE, MetricId.ANSWER_RELEVANCY, None)],
        context=None,
        judge_config=_judge_config(),
        config=EvaluationConfig(timeout_seconds=0.01, fail_closed_status=CriterionStatus.FAIL),
    )
    assert results[0].status == CriterionStatus.FAIL
    assert results[0].rationale == "judge timed out"


@pytest.mark.asyncio
async def test_judge_timeout_fail_closed_to_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fail_closed_status=INCONCLUSIVE ならタイムアウトは inconclusive へ倒す。"""

    class _SlowMetric(_StubMetric):
        async def a_measure(self, test_case: Any) -> None:
            await asyncio.sleep(1.0)

    _patch_quality_metrics(
        monkeypatch,
        factory=lambda: _SlowMetric(score=0.0, success=False, reason=""),
    )
    results = await judge(
        marked_input="q",
        marked_output="a",
        specs=[(RELEVANCE, MetricId.ANSWER_RELEVANCY, None)],
        context=None,
        judge_config=_judge_config(),
        config=EvaluationConfig(
            timeout_seconds=0.01, fail_closed_status=CriterionStatus.INCONCLUSIVE
        ),
    )
    assert results[0].status == CriterionStatus.INCONCLUSIVE


@pytest.mark.asyncio
async def test_judge_schema_error_fail_closed_generalizes_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """採点中の例外は fail-closed し rationale に例外型名のみ載せる（情報露出回避）。"""

    class _BoomMetric(_StubMetric):
        async def a_measure(self, test_case: Any) -> None:
            raise ValueError("secret endpoint https://internal")

    _patch_quality_metrics(
        monkeypatch,
        factory=lambda: _BoomMetric(score=0.0, success=False, reason=""),
    )
    results = await judge(
        marked_input="q",
        marked_output="a",
        specs=[(RELEVANCE, MetricId.ANSWER_RELEVANCY, None)],
        context=None,
        judge_config=_judge_config(),
        config=EvaluationConfig(timeout_seconds=None),
    )
    assert results[0].status == CriterionStatus.FAIL
    # 例外メッセージ全文ではなく型名のみ。
    assert results[0].rationale == "judge error: ValueError"
    assert "internal" not in results[0].rationale


@pytest.mark.asyncio
async def test_judge_telemetry_opt_out_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """deepeval_telemetry_opt_out=True で DEEPEVAL_TELEMETRY_OPT_OUT が設定される。"""
    monkeypatch.delenv("DEEPEVAL_TELEMETRY_OPT_OUT", raising=False)
    _patch_quality_metrics(
        monkeypatch,
        factory=lambda: _StubMetric(score=1.0, success=True, reason=""),
    )
    await judge(
        marked_input="q",
        marked_output="a",
        specs=[(RELEVANCE, MetricId.ANSWER_RELEVANCY, None)],
        context=None,
        judge_config=_judge_config(),
        config=EvaluationConfig(timeout_seconds=None, deepeval_telemetry_opt_out=True),
    )
    assert os.environ.get("DEEPEVAL_TELEMETRY_OPT_OUT") == "YES"


@pytest.mark.asyncio
async def test_judge_score_none_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """metric.score が None なら CriterionResult.score も None。"""

    class _NoScoreMetric:
        success = True
        reason = None
        score = None

        async def a_measure(self, test_case: Any) -> None:
            return None

    _patch_quality_metrics(monkeypatch, factory=lambda: _NoScoreMetric())
    results = await judge(
        marked_input="q",
        marked_output="a",
        specs=[(RELEVANCE, MetricId.ANSWER_RELEVANCY, None)],
        context=None,
        judge_config=_judge_config(),
        config=EvaluationConfig(timeout_seconds=None),
    )
    assert results[0].score is None
    assert results[0].rationale == ""


# ----------------------------------------------------------------------
# judge_tools: ToolCorrectnessMetric の tools_called 構築・fail-closed
# （not_applicable は evaluator の責務・本関数は評価可能ケースのみ受ける）
# ----------------------------------------------------------------------


def _patch_tool_metric(monkeypatch: pytest.MonkeyPatch, *, metric: Any) -> dict[str, Any]:
    """ToolCorrectnessMetric / LLMTestCase / ToolCall を stub に差し替え、構築引数を記録する。

    `captured["metric_kwargs"]` に ToolCorrectnessMetric の構築 kwarg を記録する（model 未指定
    退行の回帰ガード用）。
    """
    import deepeval.metrics as dm
    import deepeval.test_case as dt

    captured: dict[str, Any] = {}

    class _FakeToolCall:
        def __init__(self, *, name: str) -> None:
            self.name = name

    class _FakeLLMTestCase:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["test_case_kwargs"] = kwargs

    def _ctor(*args: Any, **kwargs: Any) -> Any:
        captured["metric_kwargs"] = kwargs
        return metric

    monkeypatch.setattr(dm, "ToolCorrectnessMetric", _ctor, raising=True)
    monkeypatch.setattr(dt, "LLMTestCase", _FakeLLMTestCase, raising=True)
    monkeypatch.setattr(dt, "ToolCall", _FakeToolCall, raising=True)
    return captured


@pytest.mark.asyncio
async def test_judge_tools_builds_tools_called_and_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """観測ツール列と expected_tools から ToolCall を組み metric で採点（success→PASS）。"""
    from oai_agentspec.runtime.llmops import ObservedToolCall

    captured = _patch_tool_metric(
        monkeypatch,
        metric=_StubMetric(score=1.0, success=True, reason="all tools correct"),
    )
    result = await judge_tools(
        name=TOOL_CORRECTNESS,
        tool_calls=[ObservedToolCall(tool="search"), ObservedToolCall(tool="fetch")],
        expected_tools=["search", "fetch"],
        judge_config=JudgeConfig(model=FakeModel()),
        config=EvaluationConfig(),
    )
    assert result.criterion == TOOL_CORRECTNESS
    assert result.status == CriterionStatus.PASS
    assert result.score == pytest.approx(1.0)
    # rationale はツール一致状況から自前で組む（LLM reason 非生成）。
    assert result.rationale == "tools matched: ['search', 'fetch']"
    # tools_called が観測ツール名から組まれる。
    tools_called = captured["test_case_kwargs"]["tools_called"]
    assert [tc.name for tc in tools_called] == ["search", "fetch"]
    expected = captured["test_case_kwargs"]["expected_tools"]
    assert [tc.name for tc in expected] == ["search", "fetch"]


@pytest.mark.asyncio
async def test_judge_tools_metric_built_with_model_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """回帰ガード: ToolCorrectnessMetric は model 付き + include_reason/async_mode 無効で構築。

    model 未指定だと DeepEval が既定 GPTModel を初期化し OPENAI_API_KEY を要求して fail-closed
    に倒れる（E2E で判明した退行）。これを二度と起こさないため構築 kwarg を検証する。
    """
    from oai_agentspec.runtime.llmops import ObservedToolCall

    captured = _patch_tool_metric(
        monkeypatch,
        metric=_StubMetric(score=1.0, success=True, reason=""),
    )
    await judge_tools(
        name=TOOL_CORRECTNESS,
        tool_calls=[ObservedToolCall(tool="search")],
        expected_tools=["search"],
        judge_config=JudgeConfig(model=FakeModel()),
        config=EvaluationConfig(),
    )
    metric_kwargs = captured["metric_kwargs"]
    assert metric_kwargs.get("model") is not None
    assert metric_kwargs.get("include_reason") is False
    assert metric_kwargs.get("async_mode") is False


@pytest.mark.asyncio
async def test_judge_tools_fail_when_metric_unsuccessful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ToolCorrectnessMetric が success=False を返すと FAIL。"""
    from oai_agentspec.runtime.llmops import ObservedToolCall

    _patch_tool_metric(
        monkeypatch,
        metric=_StubMetric(score=0.0, success=False, reason="missing tool"),
    )
    result = await judge_tools(
        name=TOOL_CORRECTNESS,
        tool_calls=[ObservedToolCall(tool="search")],
        expected_tools=["search", "fetch"],
        judge_config=JudgeConfig(model=FakeModel()),
        config=EvaluationConfig(),
    )
    assert result.status == CriterionStatus.FAIL


@pytest.mark.asyncio
async def test_judge_tools_exception_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """ToolCorrectnessMetric.measure 例外は fail-closed し型名のみ rationale に載せる。"""
    from oai_agentspec.runtime.llmops import ObservedToolCall

    class _BoomMetric:
        def measure(self, test_case: Any) -> None:
            raise RuntimeError("secret detail")

    _patch_tool_metric(monkeypatch, metric=_BoomMetric())
    result = await judge_tools(
        name=TOOL_CORRECTNESS,
        tool_calls=[ObservedToolCall(tool="search")],
        expected_tools=["search"],
        judge_config=JudgeConfig(model=FakeModel()),
        config=EvaluationConfig(fail_closed_status=CriterionStatus.FAIL),
    )
    assert result.status == CriterionStatus.FAIL
    assert result.rationale == "tool correctness error: RuntimeError"
    assert "secret" not in result.rationale


# ----------------------------------------------------------------------
# custom judge model: 同期 generate の running-loop 分岐
# ----------------------------------------------------------------------


def test_make_judge_model_generate_outside_loop() -> None:
    """running loop が無いとき generate は asyncio.run 経路で動く。"""
    model = _make_judge_model(FakeModel().queue_text("sync-out"))
    out = model.generate("prompt")
    assert out == "sync-out"


@pytest.mark.asyncio
async def test_make_judge_model_generate_inside_running_loop() -> None:
    """running loop 下では generate は別スレッドで専用ループを回して動く（別スレッド経路）。"""
    model = _make_judge_model(FakeModel().queue_text("threaded-out"))
    # 実行中ループ下から同期 generate を直接呼ぶ（running loop 検出 → 別スレッド経路）。
    out = model.generate("prompt")
    assert out == "threaded-out"


@pytest.mark.asyncio
async def test_make_judge_model_a_generate_runs_agent() -> None:
    """a_generate は最小エージェントを 1 ターン実行しテキストを返す。"""
    model = _make_judge_model(FakeModel().queue_text("async-out"))
    out = await model.a_generate("prompt")
    assert out == "async-out"


def test_make_judge_model_metadata() -> None:
    """custom model の load_model / get_model_name が正しく動く。"""
    judge_model = FakeModel()
    model = _make_judge_model(judge_model)
    assert model.load_model() is judge_model
    assert model.get_model_name() == "oai-agentspec-judge"


# ----------------------------------------------------------------------
# _resolve_metric: 未対応 metric_id の ValueError
# ----------------------------------------------------------------------


def test_resolve_metric_unsupported_id_raises() -> None:
    """未対応の MetricId は ValueError を送出する。"""
    from enum import StrEnum

    from oai_agentspec._adapters.judge import _resolve_metric

    class _BogusMetricId(StrEnum):
        BOGUS = "bogus"

    with pytest.raises(ValueError, match="unsupported metric id"):
        _resolve_metric("x", _BogusMetricId.BOGUS, object(), None, has_expected_output=False)
