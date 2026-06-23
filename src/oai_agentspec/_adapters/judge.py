"""DeepEval 統合窓口（採点エンジン結合を `_adapters` に閉じる・NFR-1）。

`import deepeval` を本モジュールの関数内遅延 import に閉じる。利用者 Judge モデル
（`JudgeConfig.model`）を `DeepEvalBaseLLM` 実装でラップし、DeepEval の LLM 呼び出しを
`_adapters` 経由（SDK Runner）に一本化する。観点 → 抽象メトリクス識別子（`criteria.MetricId`）を
DeepEval metric クラスへ解決し、**事前 Spotlighting 済み入力**を受け取って `LLMTestCase` を組み、
採点結果を plain `CriterionResult`（`runtime/llmops/types`）へ変換して返す。

評価ロジック層は DeepEval 型を一切見ない。スキーマ不適合 / タイムアウトは当該観点を
fail-closed（既定 fail）へ倒し、未捕捉例外でプロセスを止めない。DeepEval の Confident AI
テレメトリは既定オフで opt-out する（env 境界はここに閉じる）。

deepeval 未導入時は明示 ImportError + 案内（`_LLMOPS_INSTALL_HINT`）。
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..runtime.llmops.config import EvaluationConfig, JudgeConfig
    from ..runtime.llmops.criteria import MetricId
    from ..runtime.llmops.types import CriterionResult, CriterionStatus, ObservedToolCall

# llmops extra（deepeval）未導入時の案内。
_LLMOPS_INSTALL_HINT = (
    "LLMOps 評価（採点）には deepeval が必要です。"
    "次でインストールしてください: pip install 'oai-agentspec[llmops]'"
)


def _require_deepeval() -> Any:
    """deepeval を遅延 import する（未導入時は案内付き ImportError）。

    Returns:
        deepeval モジュール。

    Raises:
        ImportError: deepeval が未導入の場合（案内文字列付き）。
    """
    try:
        import deepeval  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise ImportError(_LLMOPS_INSTALL_HINT) from exc
    return deepeval


def _apply_telemetry_opt_out(opt_out: bool) -> None:
    """DeepEval の Confident AI テレメトリ opt-out を env 経由で設定する。

    env 直読・設定は本 `_adapters` 境界に閉じる（コア層へ波及させない・NFR-5）。

    Args:
        opt_out: True なら `DEEPEVAL_TELEMETRY_OPT_OUT=YES` を設定する。
    """
    if opt_out:
        os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")


def _make_judge_model(model: Any) -> Any:
    """利用者 Judge モデルを `DeepEvalBaseLLM` 実装でラップする。

    DeepEval の LLM 呼び出しを SDK Runner（`_adapters`）経由に一本化し、外部直叩きを避ける。
    `model` は SDK `Model` / モデル名文字列等の不透明値で、`agents.Runner.run` の `model`
    として渡せる。`generate` / `a_generate` は最小エージェントを 1 ターン実行してテキストを返す。

    Args:
        model: 利用者 Judge モデル（不透明値）。

    Returns:
        DeepEval custom model インスタンス。
    """
    from agents import Agent, Runner
    from deepeval.models import DeepEvalBaseLLM

    class _RunnerJudgeModel(DeepEvalBaseLLM):
        """SDK Runner 経由で採点プロンプトを実行する DeepEval custom model。"""

        def __init__(self, judge_model: Any) -> None:
            self._judge_model = judge_model
            super().__init__()

        def load_model(self, *args: Any, **kwargs: Any) -> Any:
            return self._judge_model

        def get_model_name(self, *args: Any, **kwargs: Any) -> str:
            return "oai-agentspec-judge"

        async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            agent = Agent(name="oai-agentspec-judge", instructions="", model=self._judge_model)
            result = await Runner.run(agent, prompt)
            output = result.final_output
            return "" if output is None else str(output)

        def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            # DeepEval は a_measure → a_generate 経路を使う前提だが、同期 generate へ
            # フォールバックした場合に備える。`evaluate` は実行中 event loop 下で呼ばれるため
            # `asyncio.run` は RuntimeError になる。実行中ループを検出したら別スレッドで
            # 専用ループを回し、無ければ `asyncio.run` で実行する。
            def _run_in_new_loop() -> str:
                return asyncio.run(self.a_generate(prompt, *args, **kwargs))

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return _run_in_new_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(_run_in_new_loop).result()

    return _RunnerJudgeModel(model)


def _resolve_metric(
    criterion: str,
    metric_id: MetricId,
    judge_model: Any,
    rubric: str | None,
    *,
    has_expected_output: bool,
) -> Any:
    """抽象メトリクス識別子を DeepEval metric インスタンスへ解決する。

    G-Eval 観点（safety / conciseness 等）は `rubric`（観点文）を criteria として渡す。rubric が
    None の場合は観点名のみの最小定義に倒す（プロンプト本文は lib 非同梱）。`has_expected_output`
    が True のときのみ G-Eval の `evaluation_params` に `LLMTestCaseParams.EXPECTED_OUTPUT` を
    含める（DeepEval は evaluation_params に挙げた項目が test case に必須で、None だと例外になる
    ため・提供時のみ追加）。

    Args:
        criterion: 観点名。
        metric_id: 抽象メトリクス識別子。
        judge_model: ラップ済み DeepEval custom model。
        rubric: G-Eval 観点文（None なら観点名のみの最小定義）。
        has_expected_output: 当該ケースに expected_output（正解文）が提供されているか。

    Returns:
        DeepEval metric インスタンス。

    Raises:
        ValueError: 未対応の metric_id の場合。
    """
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCaseParams

    from ..runtime.llmops.criteria import MetricId

    if metric_id is MetricId.FAITHFULNESS:
        return FaithfulnessMetric(model=judge_model)
    if metric_id is MetricId.ANSWER_RELEVANCY:
        return AnswerRelevancyMetric(model=judge_model)
    if metric_id is MetricId.G_EVAL:
        criteria_text = rubric or f"Evaluate the output for the criterion: {criterion}."
        params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
        if has_expected_output:
            params.append(LLMTestCaseParams.EXPECTED_OUTPUT)
        return GEval(
            name=criterion,
            criteria=criteria_text,
            evaluation_params=params,
            model=judge_model,
        )
    raise ValueError(f"unsupported metric id: {metric_id}")


def _status_from_metric(metric: Any) -> CriterionStatus:
    """DeepEval metric の `success` から `CriterionStatus` を導出する。

    Args:
        metric: 採点済み DeepEval metric（`success` 属性を持つ）。

    Returns:
        success True なら PASS、False なら FAIL。
    """
    from ..runtime.llmops.types import CriterionStatus

    success = getattr(metric, "success", None)
    return CriterionStatus.PASS if success else CriterionStatus.FAIL


async def _score_one(
    criterion: str,
    metric_id: MetricId,
    *,
    marked_input: str,
    marked_output: str,
    context: list[str] | None,
    expected_output: str | None,
    rubric: str | None,
    judge_model: Any,
    timeout: float | None,
    fail_status: CriterionStatus,
) -> CriterionResult:
    """1 観点を DeepEval で採点し plain `CriterionResult` へ変換する（fail-closed）。

    タイムアウト / スキーマ不適合 / 未捕捉例外は `fail_status` へ倒す（プロセス停止させない）。

    Args:
        criterion: 観点名。
        metric_id: 抽象メトリクス識別子。
        marked_input: Spotlighting 済み入力テキスト。
        marked_output: Spotlighting 済み出力テキスト。
        context: 参照文脈（faithfulness 用）。
        expected_output: 正解文（任意・利用者提供の信頼入力でマーキングしない）。提供時は
            G-Eval が `EXPECTED_OUTPUT` として参照する。None なら従来どおり参照しない。
        rubric: G-Eval 観点文（None なら観点名のみの最小定義）。
        judge_model: ラップ済み DeepEval custom model。
        timeout: 採点タイムアウト秒（None で無制限）。
        fail_status: fail-closed 時に倒す状態（fail / inconclusive）。

    Returns:
        plain `CriterionResult`。
    """
    from deepeval.test_case import LLMTestCase

    from ..runtime.llmops.types import CriterionResult

    try:
        metric = _resolve_metric(
            criterion,
            metric_id,
            judge_model,
            rubric,
            has_expected_output=expected_output is not None,
        )
        # marked_output は domain 側 `_spotlight` でデリミタ済み（untrusted）。内蔵メトリクスは
        # 固定プロンプトでマーカーの意味を知らないため Spotlighting は実質無効、G-Eval では利用者
        # rubric 側でマーカー内部をデータ扱いする指示が要る（_spotlight.py docstring 参照）。
        # expected_output は利用者提供の信頼入力でマーキングしない（context と同じ扱い）。
        test_case = LLMTestCase(
            input=marked_input,
            actual_output=marked_output,
            retrieval_context=context,
            context=context,
            expected_output=expected_output,
        )

        async def _measure() -> None:
            await metric.a_measure(test_case)

        if timeout is not None:
            await asyncio.wait_for(_measure(), timeout=timeout)
        else:
            await _measure()
    except TimeoutError:
        return CriterionResult(
            criterion=criterion,
            status=fail_status,
            rationale="judge timed out",
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed（プロセス停止させない）
        # 例外メッセージ全文は外部送信される（langfuse Scores comment）。認証エラー時に
        # エンドポイント等が混入しうるため例外型名のみに一般化する（情報露出回避）。
        return CriterionResult(
            criterion=criterion,
            status=fail_status,
            rationale=f"judge error: {type(exc).__name__}",
        )

    score = getattr(metric, "score", None)
    reason = getattr(metric, "reason", None)
    return CriterionResult(
        criterion=criterion,
        status=_status_from_metric(metric),
        rationale="" if reason is None else str(reason),
        score=None if score is None else float(score),
    )


async def judge(
    *,
    marked_input: str,
    marked_output: str,
    specs: list[tuple[str, MetricId, str | None]],
    context: list[str] | None,
    expected_output: str | None = None,
    judge_config: JudgeConfig,
    config: EvaluationConfig,
) -> list[CriterionResult]:
    """評価可能な品質観点を DeepEval で採点し plain `CriterionResult` 列を返す（NFR-1）。

    マーキングは domain 側 `_spotlight` の責務で、本関数は **事前処理済みのテキスト**を受け取る
    （untrusted な評価対象 output は Spotlighting 済み・利用者提供 input は非マーキング・設計 §3）。
    `specs` は呼び出し側（evaluator）が Criterion から組んだ `(name, metric_id, rubric)` 列で、
    各 spec を DeepEval metric へ解決して採点する。**必要データ不足の not_applicable 判定は
    evaluator が担い、本関数は評価可能な観点のみを受け取る**（採点に専念・二重 NA を出さない）。
    deepeval テレメトリは config に従って opt-out する。

    Args:
        marked_input: 入力テキスト（信頼入力・非マーキング）。
        marked_output: Spotlighting 済み出力テキスト（untrusted）。
        specs: 採点する観点の `(name, metric_id, rubric)` 列。
        context: 参照文脈（faithfulness 用・LLMTestCase へ載せる）。
        expected_output: 正解文（任意・信頼入力で非マーキング）。提供時は G-Eval が
            `EXPECTED_OUTPUT` として参照する。
        judge_config: Judge 設定（model）。
        config: 評価設定（timeout / fail_closed_status / テレメトリ）。

    Returns:
        観点別の plain `CriterionResult` 列。
    """
    _require_deepeval()
    _apply_telemetry_opt_out(config.deepeval_telemetry_opt_out)
    judge_model = _make_judge_model(judge_config.model)

    results: list[CriterionResult] = []
    for name, metric_id, rubric in specs:
        results.append(
            await _score_one(
                name,
                metric_id,
                marked_input=marked_input,
                marked_output=marked_output,
                context=context,
                expected_output=expected_output,
                rubric=rubric,
                judge_model=judge_model,
                timeout=config.timeout_seconds,
                fail_status=config.fail_closed_status,
            )
        )
    return results


def _tools_rationale(
    *, status_pass: bool, expected_tools: list[str], tool_calls: list[ObservedToolCall]
) -> str:
    """ツール一致状況から rationale を自前で組む（LLM reason 非生成・決定的）。

    Args:
        status_pass: 採点結果が PASS か。
        expected_tools: 期待ツール名のリスト。
        tool_calls: 観測したツール呼び出し列。

    Returns:
        一致なら "tools matched: [...]"、不一致なら "tools mismatch: expected [...] got [...]"。
    """
    observed = [tc.tool for tc in tool_calls]
    if status_pass:
        return f"tools matched: {observed}"
    return f"tools mismatch: expected {list(expected_tools)} got {observed}"


async def judge_tools(
    *,
    name: str,
    tool_calls: list[ObservedToolCall],
    expected_tools: list[str],
    judge_config: JudgeConfig,
    config: EvaluationConfig,
) -> CriterionResult:
    """ツール使用の正しさを DeepEval ToolCorrectnessMetric で採点する（決定的・NFR-1）。

    捕捉済み `ObservedToolCall` 列から `tools_called` を組み、`expected_tools` と決定的比較する。
    比較自体は LLM を呼ばないが、DeepEval の `ToolCorrectnessMetric.__init__` は無条件に
    `initialize_model(model)` を呼ぶため、`model=None` だと既定 GPTModel が `OPENAI_API_KEY` を
    要求し Azure 等で fail する。これを避けるため `judge_config` のラップ済み判定モデル（既存
    `judge()` と同じ `_adapters` 経由 `DeepEvalBaseLLM` ラッパ・NFR-1 整合）を渡す。
    `include_reason=False`（LLM による reason 生成を行わず rationale は一致状況から自前で組む）/
    `async_mode=False`（実行中 event loop 下での measure 内 asyncio 再入を避ける）で決定的運用する。
    `threshold=1.0` + 非 exact-match で **recall**（期待ツールが全て呼ばれていれば pass・余分な
    呼び出しは無視）にする。`tool_calls` に handoff の `transfer_to_*` 呼び出しが含まれても「余分な
    呼び出し」として無視され false fail を生まない（呼び出し回数・順序は見ない）。

    **必要データ不足の not_applicable 判定は evaluator が担い、本関数は expected_tools 充足の
    ケースのみを受け取る**。スキーマ不適合・未捕捉例外は fail-closed。

    Args:
        name: 観点名（結果 `CriterionResult.criterion` に出る値）。
        tool_calls: 観測したツール呼び出し列（下流エージェント・handoff transfer を含みうる）。
        expected_tools: 期待ツール名のリスト（ground truth・非空想定）。
        judge_config: Judge 設定（model）。ToolCorrectnessMetric の model 初期化に使う。
        config: 評価設定（fail-closed 状態 / テレメトリ）。

    Returns:
        plain `CriterionResult`（criterion=name）。
    """
    from ..runtime.llmops.types import CriterionResult, CriterionStatus

    _require_deepeval()
    _apply_telemetry_opt_out(config.deepeval_telemetry_opt_out)
    try:
        from deepeval.metrics import ToolCorrectnessMetric
        from deepeval.test_case import LLMTestCase, ToolCall

        judge_model = _make_judge_model(judge_config.model)
        tools_called = [ToolCall(name=tc.tool) for tc in tool_calls]
        expected = [ToolCall(name=tool) for tool in expected_tools]
        test_case = LLMTestCase(
            input="",
            actual_output="",
            tools_called=tools_called,
            expected_tools=expected,
        )
        metric = ToolCorrectnessMetric(
            model=judge_model,
            include_reason=False,
            async_mode=False,
            threshold=1.0,
        )
        metric.measure(test_case)
    except Exception as exc:  # noqa: BLE001 - fail-closed
        # 例外型名のみ（メッセージ全文の外部送信を避ける・Fix 8 と同方針）。
        return CriterionResult(
            criterion=name,
            status=config.fail_closed_status,
            rationale=f"tool correctness error: {type(exc).__name__}",
        )

    status = _status_from_metric(metric)
    score = getattr(metric, "score", None)
    return CriterionResult(
        criterion=name,
        status=status,
        rationale=_tools_rationale(
            status_pass=status == CriterionStatus.PASS,
            expected_tools=expected_tools,
            tool_calls=tool_calls,
        ),
        score=None if score is None else float(score),
    )
