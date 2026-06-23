"""評価オーケストレータ（`evaluate` 公開エントリの本体・観点オブジェクト駆動）。

対象 build → 実行（route + tool 捕捉）→ 各 `Criterion` を評価可能性で振り分け（必要データ不足は
evaluator が理由付き not_applicable を生成・judge を呼ばない）→ DeepEval 採点（品質系）/
ToolCorrectnessMetric（ツール）/ 決定的経路比較（handoff）→ verdict 計算 → Langfuse 送信（任意）
を結線する。`_adapters`（judge / langfuse / runner）は関数内遅延 import に閉じ、`agents` /
`deepeval` / `langfuse` を本モジュールで import しない（NFR-1）。`RunResult` を一切受け取らず
plain `RunOutcome` / `ObservedRun` のみ扱う。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace as _dataclass_replace
from typing import TYPE_CHECKING, Any, Final

from . import _target
from ._spotlight import spotlight
from .config import EvaluationConfig, JudgeConfig
from .criteria import (
    _APPROVAL_GATE,
    _REQ_CONTEXT,
    _REQ_EXPECTED_APPROVALS,
    _REQ_EXPECTED_ROUTE,
    _REQ_EXPECTED_TOOLS,
    Criterion,
    MetricId,
    default_criteria,
)
from .types import (
    CaseResult,
    CriterionResult,
    CriterionStatus,
    EvaluationResult,
    ObservedApproval,
    ObservedRun,
)
from .verdict import compute_verdict

logger = logging.getLogger(__name__)

# 承認自動解決ループの反復上限（無限ループ防止・段階解決の安全弁）。
_MAX_RESOLVE_ROUNDS: Final[int] = 5

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ...handoffs import HandoffGraph
    from ...registry import AgentRegistry
    from ...spec import AgentSpec
    from ...workflow import WorkflowGraph
    from .config import LangfuseConfig
    from .dataset import EvalCase


def _not_applicable(name: str, reason: str) -> CriterionResult:
    """理由付き not_applicable の `CriterionResult` を作る。

    Args:
        name: 観点名。
        reason: not_applicable の理由（rationale に明記）。

    Returns:
        status=NOT_APPLICABLE の `CriterionResult`。
    """
    return CriterionResult(criterion=name, status=CriterionStatus.NOT_APPLICABLE, rationale=reason)


# 承認待ち（HITL/ツール承認）で中断した実行の rationale。
_INTERRUPTED_REASON = "run interrupted: awaiting approval (HITL)"


def _interrupted_case(
    case: EvalCase,
    *,
    criteria: Sequence[Criterion],
    observation: ObservedRun,
    pending: list[dict[str, str]],
) -> CaseResult:
    """承認待ちで中断した実行のケース結果を組む（ApprovalGate は採点・他は inconclusive）。

    評価対象が HITL / ツール承認で中断すると最終出力が無く（`RunOutcome.final_output=None`）、
    route / tool も部分的になりうる。この状態を空文字出力として採点続行すると、出力非依存の
    観点（handoff_correctness / tool_correctness）が途中経路の一致で誤って pass しうる（Codex P2）。
    そこで judge / route / tool 採点を一切走らせず、それらの観点を理由付き INCONCLUSIVE に倒す。
    `compute_verdict` の `inconclusive_policy`（既定 FAIL）で中断は既定 verdict fail になり、かつ
    「未完了」が rationale で顕在化する（中断を採点完了と誤認しない）。

    ただし **ApprovalGate（案B）は中断時でも採点する**（中断短絡の前に処理する・#29 の核）。
    承認ゲートが期待どおり発火したか（`pending` のツール名集合 ⊇ `expected_approvals`）を決定的に
    判定する。resume も approve もしないため危険ツールは実行されない。expected_approvals 非在の
    ケースでは `_na_reason` 経由で not_applicable になる（混在を許容）。

    Args:
        case: 評価ケース。
        criteria: 評価観点列。
        observation: 中断時点までに捕捉した実行トレース（部分的・pending 情報付き）。
        pending: 中断時点の承認待ち一覧（plain dict 列・ApprovalGate の採点に使う）。

    Returns:
        ApprovalGate のみ採点済み・他観点は INCONCLUSIVE の `CaseResult`（output は空文字）。
    """
    results: list[CriterionResult] = []
    for c in criteria:
        if c.name == _APPROVAL_GATE:
            reason = _na_reason(c, case=case)
            if reason is not None:
                results.append(_not_applicable(c.name, reason))
            else:
                results.append(
                    _approval_gate(
                        c.name,
                        expected_approvals=case.expected_approvals or [],
                        pending=pending,
                    )
                )
            continue
        results.append(
            CriterionResult(
                criterion=c.name,
                status=CriterionStatus.INCONCLUSIVE,
                rationale=_INTERRUPTED_REASON,
            )
        )
    return CaseResult(case_input=case.input, output="", criteria=results, observation=observation)


def _na_reason(criterion: Criterion, *, case: EvalCase) -> str | None:
    """`Criterion` が当該ケースで評価不能なら理由文字列を、評価可能なら None を返す。

    NA 判定は **必要 ground truth の充足のみ**で行う（NA 判定は evaluator に一本化）。対象の能力
    （ツール保有 / 横断モード）では NA にしない: 観点を criteria に入れるかは利用者が決め、能力が
    無く期待を満たさなければ NA ではなく fail として現れる（例: ツール非保有で expected_tools が
    呼ばれなければ recall=0 で fail）。

    Args:
        criterion: 評価観点。
        case: 評価ケース。

    Returns:
        評価不能なら理由文字列、評価可能なら None。
    """
    if _REQ_CONTEXT in criterion.requires and not case.reference_context:
        return "requires reference_context"
    if _REQ_EXPECTED_ROUTE in criterion.requires and case.expected_route is None:
        return "requires expected_route"
    if _REQ_EXPECTED_TOOLS in criterion.requires and case.expected_tools is None:
        return "requires expected_tools"
    if _REQ_EXPECTED_APPROVALS in criterion.requires and case.expected_approvals is None:
        return "requires expected_approvals"
    return None


def _handoff_correctness(
    name: str,
    *,
    expected_route: list[str],
    observation: ObservedRun,
) -> CriterionResult:
    """ルーティングの正しさを決定的比較で判定する（評価可能なケースのみ・自前純比較）。

    観測経路のエージェント系列（`ObservedRoute.steps` の agent 列 = 起点を先頭に・handoff 遷移先を
    順に並べ・末尾に最終応答 agent を含めたフルパス・routing 側で正規化済み）と `expected_route` を
    順序付きで完全一致比較する。`expected_route` は起点を含むフルパスで指定する（例:
    triage→billing は `["triage", "billing"]`）。NA 判定（expected_route 非在）は呼び出し側
    （evaluator）が担う。

    Args:
        name: 観点名。
        expected_route: 期待エージェント名の系列（ground truth・非 None）。
        observation: 観測した実行トレース。

    Returns:
        plain `CriterionResult`（criterion=name）。
    """
    observed = [step.agent for step in observation.route.steps]
    if observed == list(expected_route):
        return CriterionResult(
            criterion=name,
            status=CriterionStatus.PASS,
            rationale=f"route matched: {observed}",
            score=1.0,
        )
    return CriterionResult(
        criterion=name,
        status=CriterionStatus.FAIL,
        rationale=f"route mismatch: expected {list(expected_route)}, observed {observed}",
        score=0.0,
    )


def _approval_gate(
    name: str,
    *,
    expected_approvals: list[str],
    pending: list[dict[str, str]],
) -> CriterionResult:
    """承認ゲートの発火を決定的比較で判定する（中断時点の承認待ちと期待集合を recall 比較）。

    `pending`（`RunOutcome.pending` の `{"tool_name", "call_id"}` 列）のツール名集合に
    `expected_approvals` が全て含まれていれば pass（recall: 期待した承認ゲートが全て発火した）。
    余分な承認待ちは無視する（recall のみを見る）。resume も approve もしないため危険ツールは
    実行されない。NA 判定（expected_approvals 非在）は呼び出し側（evaluator）が担う。

    Args:
        name: 観点名。
        expected_approvals: 期待する承認待ちツール名の集合（ground truth・非 None）。
        pending: 中断時点の承認待ち一覧（plain dict 列・中断なしなら空）。

    Returns:
        plain `CriterionResult`（criterion=name）。
    """
    pending_tools = {p.get("tool_name", "") for p in pending}
    missing = [t for t in expected_approvals if t not in pending_tools]
    if not missing:
        return CriterionResult(
            criterion=name,
            status=CriterionStatus.PASS,
            rationale=f"approval gates fired: {sorted(pending_tools)}",
            score=1.0,
        )
    return CriterionResult(
        criterion=name,
        status=CriterionStatus.FAIL,
        rationale=(
            f"approval gate not fired: expected {list(expected_approvals)}, "
            f"pending {sorted(pending_tools)}"
        ),
        score=0.0,
    )


def _with_approvals(
    observation: ObservedRun,
    *,
    pending: list[dict[str, str]],
    interrupted: bool,
) -> ObservedRun:
    """`ObservedRun` に承認待ち情報（pending_approvals / interrupted）を載せて返す。

    Langfuse metadata（`pending_approvals` / `interrupted`）の根拠を `CaseResult.observation`
    経由で参照できるよう、採点時点の承認待ち集合と中断フラグを plain `ObservedApproval` 列として
    付与した新しい `ObservedRun` を作る（frozen のため `dataclasses.replace`）。

    Args:
        observation: route / tool を捕捉済みの `ObservedRun`。
        pending: 承認待ち一覧（plain dict 列・発火した承認 or 未解決の承認）。
        interrupted: 採点時点で承認待ちが未解決のまま中断していたか。

    Returns:
        承認情報を付与した新しい `ObservedRun`。
    """
    approvals = [
        ObservedApproval(tool=p.get("tool_name", ""), call_id=p.get("call_id", "")) for p in pending
    ]
    return _dataclass_replace(observation, pending_approvals=approvals, interrupted=interrupted)


def _merge_observation(base: ObservedRun, nxt: ObservedRun) -> ObservedRun:
    """2 つの `ObservedRun`（segment）を 1 つへマージする（route/tool を横断連結）。

    承認自動解決ループでは 1 ケースが複数 segment（中断 → resume → ...）に分かれて実行される。
    各 segment の route ステップとツール呼び出しを順に連結し、最終応答 agent / last_agent は
    後段（`nxt`）の値で更新する。`pending_approvals` / `interrupted` は本関数では引き継がず、
    呼び出し側がループ全体の集約として設定する（segment 単位では持たない）。

    各 segment の route 末尾は last_agent（最終応答 agent）を含むため、segment 境界では前段末尾と
    後段先頭が同一 agent になりうる（例: 単体 agent の resume で `['bot']` + `['bot']`）。**境界の
    連続重複だけを畳む**（後段先頭ステップの agent が累積末尾の agent と同一ならその先頭をスキップ・
    Codex P2-1）。別 agent への正当な handoff-back は畳まない（連続が同一 agent のときのみ）。

    Args:
        base: これまでに蓄積した observation。
        nxt: 直近 segment の observation。

    Returns:
        route ステップとツール呼び出しを連結した新しい `ObservedRun`。
    """
    from .types import ObservedRoute

    base_steps = list(base.route.steps)
    nxt_steps = list(nxt.route.steps)
    # segment 境界の連続重複（前段末尾 agent == 後段先頭 agent）を 1 件だけ畳む。
    if base_steps and nxt_steps and base_steps[-1].agent == nxt_steps[0].agent:
        nxt_steps = nxt_steps[1:]
    merged_route = ObservedRoute(
        steps=base_steps + nxt_steps,
        last_agent=nxt.route.last_agent or base.route.last_agent,
    )
    return ObservedRun(
        route=merged_route,
        tool_calls=list(base.tool_calls) + list(nxt.tool_calls),
    )


def _merge_pending(
    fired: list[ObservedApproval], pending: list[dict[str, str]]
) -> list[dict[str, str]]:
    """既発火の承認（fired）と現在残存の承認待ち（pending）を call_id で重複排除してマージする。

    承認自動解決ループで先に発火・解決した承認（fired）を捨てると、ApprovalGate / Langfuse が
    「先に発火したゲートは未発火」と誤報告する（Codex P2-2）。fired を先頭・残存 pending を後続に
    並べ、`call_id` で重複排除した plain dict 列を返す（call_id が空の要素は tool 名で識別不能の
    ため常に残す）。

    Args:
        fired: 全ラウンドで発火した承認（`ObservedApproval` 列）。
        pending: 採点時点で残存している承認待ち（plain dict 列）。

    Returns:
        重複排除済みの承認待ち plain dict 列（`{"tool_name", "call_id"}`）。
    """
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    for a in fired:
        key = a.call_id
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append({"tool_name": a.tool, "call_id": a.call_id})
    for p in pending:
        key = p.get("call_id", "")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append({"tool_name": p.get("tool_name", ""), "call_id": key})
    return merged


def _build_decisions(
    pending: list[dict[str, str]],
    *,
    resolver: Callable[[dict], bool],
    replaced_tools: frozenset[tuple[str, str]],
) -> list[dict[str, Any]]:
    """承認待ち列に resolver を適用し `apply_approvals` 用の decisions を構築する。

    各 pending（plain `{"tool_name", "call_id", "agent_name"}`）を resolver へ渡し、True なら
    approve / False なら reject の decision を組む。**安全不変条件（Codex P1）**: approve を返した
    承認待ちの `(agent_name, tool_name)` ペアが `replaced_tools`（宣言層 mock で**実際にモックへ
    差し替えた** `(agent, tool)` ペア集合）に含まれない場合は `ValueError`（本物の危険ツールを
    構造的に実行させない）。同名ツールでも別 agent 由来や agent 不明（空文字）の approve は認可
    しない（許可の根拠を「ツール名」ではなく「実差し替えした (agent, tool)」に置く fail-closed）。
    tool_mocks にキーがあっても spec 実体を持たない（factory 登録）等で差し替えられなかった
    ペアは `replaced_tools` に入らないため approve はエラーになる。reject は安全（ツール非実行）。

    Args:
        pending: 中断時点の承認待ち一覧（plain dict 列・`agent_name` を含む）。
        resolver: `(pending_dict) -> bool`。approve(True) / reject(False) を返す。
        replaced_tools: 宣言層 mock で実際にモックへ差し替えた `(agent, tool)` ペアの集合。

    Returns:
        `apply_approvals` 用 decisions（`{"call_id", "decision", "rejection_message"}` 列）。

    Raises:
        ValueError: approve を返した `(agent_name, tool_name)` が `replaced_tools` に無い場合。
    """
    decisions: list[dict[str, Any]] = []
    for item in pending:
        tool_name = item.get("tool_name", "")
        call_id = item.get("call_id", "")
        agent_name = item.get("agent_name", "")
        approved = bool(resolver(dict(item)))
        if approved and (agent_name, tool_name) not in replaced_tools:
            raise ValueError(
                f"approval resolver が approve を返したツール {tool_name!r}（agent "
                f"{agent_name!r}）がモックへ差し替えられていません。本物の危険ツールの実行を防ぐ"
                "ため、approve するツールは evaluate(tool_mocks={agent: {tool: 値}}) で当該 agent "
                "のモック実装を指定し、かつ実際に差し替え可能（spec ベース登録の FunctionTool）"
                "である必要があります"
            )
        if approved:
            decisions.append({"call_id": call_id, "decision": "approve"})
        else:
            decisions.append(
                {
                    "call_id": call_id,
                    "decision": "reject",
                    "rejection_message": "rejected by evaluation resolver",
                }
            )
    return decisions


async def _resolve_loop(
    agent: Any,
    *,
    outcome: Any,
    observation: ObservedRun,
    resolver: Callable[[dict], bool],
    replaced_tools: frozenset[tuple[str, str]],
) -> tuple[Any, ObservedRun, list[ObservedApproval]]:
    """中断 → resolver で承認/却下 → resume を完了または上限まで反復する（承認自動解決）。

    各ラウンドで `RunOutcome.pending` を resolver で approve/reject 判定し、`apply_approvals` で
    state へ適用後 `resume_with_observation` で再開する。route/tool は segment 横断でマージする。
    発火した承認待ち（全ラウンドの pending）を `ObservedApproval` 列として集約して返す。

    `apply_approvals` の戻り `ApplyResult` を検査し、そのラウンドで 1 件も適用できなかった
    （`applied` が空・unknown / already_resolved のみ）場合は **進展なしと判定して即 break** する
    （max_rounds を空回りさせない）。中断のまま戻るのは安全側（呼び出し側が `_interrupted_case`
    で INCONCLUSIVE→fail に倒す）。

    Args:
        agent: 正規化済み実行 Agent。
        outcome: 初回実行の `RunOutcome`（interrupted=True 前提）。
        observation: 初回実行の `ObservedRun`。
        resolver: `(pending_dict) -> bool`。approve(True) / reject(False)。
        replaced_tools: 宣言層 mock で実際にモックへ差し替えた `(agent, tool)` ペアの集合
            （安全不変条件の照合に使う）。

    Returns:
        `(最終 RunOutcome, マージ済み ObservedRun, 発火した承認待ち ObservedApproval 列)`。

    Raises:
        ValueError: approve を返した `(agent, tool)` が `replaced_tools` に無い場合（安全不変）。
    """
    from ..._adapters import apply_approvals, resume_with_observation

    # fired = ゲート発火した承認待ち（approve / reject を問わず承認待ちに出たツール全体）。
    # ApprovalGate は recall でゲート発火の有無を見るため、reject 対象も含めてよい。
    fired: list[ObservedApproval] = []
    merged = observation
    current = outcome
    for _round in range(_MAX_RESOLVE_ROUNDS):
        if not current.interrupted:
            break
        pending = list(current.pending)
        fired.extend(
            ObservedApproval(tool=p.get("tool_name", ""), call_id=p.get("call_id", ""))
            for p in pending
        )
        decisions = _build_decisions(pending, resolver=resolver, replaced_tools=replaced_tools)
        applied = apply_approvals(current.state, decisions)
        if applied.unknown or applied.already_resolved:
            logger.debug(
                "approval 適用で未引き当て/解決済みを検出（unknown=%s, already_resolved=%s）",
                applied.unknown,
                applied.already_resolved,
            )
        # 進展なし（1 件も適用できず pending が残る）なら空回りを避け即打ち切る（中断のまま）。
        if not applied.applied:
            break
        current, segment = await resume_with_observation(agent, current.state)
        merged = _merge_observation(merged, segment)
    return current, merged, fired


async def _evaluate_case(
    case: EvalCase,
    *,
    agent: Any,
    criteria: Sequence[Criterion],
    judge_config: JudgeConfig,
    config: EvaluationConfig,
    resolver: Callable[[dict], bool] | None = None,
    replaced_tools: frozenset[tuple[str, str]] = frozenset(),
) -> CaseResult:
    """1 ケースを実行し、各 `Criterion` を振り分けて採点し `CaseResult` を組む。

    各観点は (1) 必要 ground truth 不足なら理由付き not_applicable、(2) 決定的（handoff /
    approval_gate）なら決定的比較、(3) ツール（TOOL_CORRECTNESS）なら ToolCorrectnessMetric、
    (4) それ以外の品質系は DeepEval judge へ振り分ける。判定対象は **criteria に挙げた観点のみ**
    （自動付与しない・MAJOR-2）。

    中断（HITL / ツール承認）時の扱いは `resolver`（承認自動解決）の有無で分岐する:
        - resolver 無し（既定・後方互換）: 採点せず全観点 INCONCLUSIVE に倒す。ただし
          ApprovalGate は中断短絡の前に pending と比較して採点する（混在を許容）。
        - resolver 有り（案A）: 中断を承認/却下で自動解決し完了まで再開してから通常採点する。
          approve するツールは実際にモックへ差し替え済み（本物の危険ツールは実行されない・安全
          不変条件は `_build_decisions` が `replaced_tools` で担保）。

    Args:
        case: 評価ケース。
        agent: 正規化済み実行 Agent（tool_mocks 適用済み）。
        criteria: 評価観点列（データセット共通）。
        judge_config: Judge 設定。
        config: 評価設定。
        resolver: 承認自動解決の判定関数（`(pending_dict) -> bool`）。None で #24 挙動（中断は
            inconclusive、ApprovalGate のみ pending 採点）。
        replaced_tools: 宣言層 mock で実際にモックへ差し替えた `(agent, tool)` ペアの集合
            （安全不変条件の照合に使う）。

    Returns:
        当該ケースの `CaseResult`。
    """
    from ..._adapters import DefaultRunnerAdapter, judge, judge_tools

    runner = DefaultRunnerAdapter()
    outcome, observation = await runner.run_with_observation(agent, case.input)

    # 承認自動解決（案A）: resolver 指定 + 中断時は承認/却下を適用して完了まで再開する。
    # 発火した承認待ち（fired）は ApprovalGate の採点・observation 集約に使う。
    fired: list[ObservedApproval] = []
    if outcome.interrupted and resolver is not None:
        outcome, observation, fired = await _resolve_loop(
            agent,
            outcome=outcome,
            observation=observation,
            resolver=resolver,
            replaced_tools=replaced_tools,
        )

    # 解決後も中断が残る（resolver 無し / reject で詰む / 上限到達）場合は完了採点せず、
    # ApprovalGate は pending を採点し、それ以外の観点は INCONCLUSIVE に倒す（混在を許容）。
    # ApprovalGate を中断短絡の前に採点するのが #29 の核（中断時でもゲート発火を判定する）。
    if outcome.interrupted:
        # 既に発火・解決した承認（fired）＋現在残っている承認待ち（outcome.pending）をマージして
        # 採点する。fired を捨てると「先に発火・approve したゲート」が未発火と誤報告される（Codex
        # P2-2）。call_id で重複排除し、fired を先頭・残存 pending を後続に並べる。
        interrupted_pending = _merge_pending(fired, outcome.pending)
        return _interrupted_case(
            case,
            criteria=criteria,
            observation=_with_approvals(observation, pending=interrupted_pending, interrupted=True),
            pending=interrupted_pending,
        )

    output = outcome.final_output or ""

    # Spotlighting は untrusted な評価対象 output にのみ適用する（設計 §3 / 判断H）。
    # 利用者提供の case.input / expected_output は信頼入力でありマーキングしない。
    marked_input = str(case.input)
    marked_output = spotlight(output)

    # 完了採点時の ApprovalGate は発火した承認待ち（fired・自動解決で完了した場合）を pending
    # 集合として比較する。承認ゲートを通らずに完了した場合は fired が空で expected と照合する。
    approval_pending = [{"tool_name": a.tool, "call_id": a.call_id} for a in fired]

    results: list[CriterionResult] = []
    # judge へまとめて渡す品質系 spec（name, metric_id, rubric）。
    quality_specs: list[tuple[str, MetricId, str | None]] = []

    for criterion in criteria:
        reason = _na_reason(criterion, case=case)
        if reason is not None:
            results.append(_not_applicable(criterion.name, reason))
            continue
        if criterion.deterministic and criterion.name == _APPROVAL_GATE:
            # approval_gate（決定的・発火検証）。expected_approvals は _na_reason で充足済み。
            results.append(
                _approval_gate(
                    criterion.name,
                    expected_approvals=case.expected_approvals or [],
                    pending=approval_pending,
                )
            )
            continue
        if criterion.deterministic:
            # handoff_correctness（決定的経路比較）。expected_route は _na_reason で充足済み。
            expected_route = case.expected_route or []
            results.append(
                _handoff_correctness(
                    criterion.name,
                    expected_route=expected_route,
                    observation=observation,
                )
            )
            continue
        if criterion.metric is MetricId.TOOL_CORRECTNESS:
            # expected_tools は _na_reason で充足済み。
            results.append(
                await judge_tools(
                    name=criterion.name,
                    tool_calls=observation.tool_calls,
                    expected_tools=case.expected_tools or [],
                    judge_config=judge_config,
                    config=config,
                )
            )
            continue
        # 品質系（G-Eval / Faithfulness / AnswerRelevancy）はまとめて judge へ。metric は非 None。
        if criterion.metric is not None:
            quality_specs.append((criterion.name, criterion.metric, criterion.rubric))

    if quality_specs:
        results.extend(
            await judge(
                marked_input=marked_input,
                marked_output=marked_output,
                specs=quality_specs,
                context=case.reference_context,
                expected_output=case.expected_output,
                judge_config=judge_config,
                config=config,
            )
        )

    return CaseResult(
        case_input=case.input,
        output=output,
        criteria=results,
        observation=_with_approvals(
            observation,
            pending=approval_pending,
            interrupted=False,
        ),
    )


async def evaluate(
    target: AgentSpec | WorkflowGraph | HandoffGraph,
    dataset: Sequence[EvalCase],
    *,
    judge: JudgeConfig | Any,
    criteria: Sequence[Criterion] | None = None,
    registry: AgentRegistry | None = None,
    config: EvaluationConfig | None = None,
    langfuse: LangfuseConfig | None = None,
    approvals: Callable[[dict], bool] | None = None,
    tool_mocks: dict[str, dict[str, Any]] | None = None,
) -> EvaluationResult:
    """評価対象をデータセットで採点し統合 verdict 付きの `EvaluationResult` を返す（公開窓口）。

    `target` を `_target.normalize` で実行可能 Agent へ正規化し（AgentSpec=単体 / HandoffGraph・
    WorkflowGraph=横断・registry 経由）、各ケースを実行して route + tool を捕捉する。`criteria` に
    挙げた観点のみを評価し（**criteria に入れない観点は評価行も not_applicable 行も出ず、母集合や
    Langfuse Scores にも現れない**）、必要データ不足の観点は evaluator が理由付き not_applicable と
    する。untrusted な output へ Spotlighting を適用してから DeepEval / ToolCorrectnessMetric /
    決定的経路比較で採点し `compute_verdict` で統合 verdict を出す。

    `langfuse` 指定時のみ Langfuse へ best-effort 送信する（送信失敗でも評価は fail させない・
    NFR-3）。

    `approvals` 指定時は HITL / ツール承認で中断した実行を承認/却下で自動解決し完了まで再開して
    から採点する（HITL 対応エージェントの完了採点）。approve するツールは `tool_mocks` のモック
    実装に差し替えてから実行する（本物の危険ツールは実行されない＝構造的安全）。承認ゲート
    （`needs_approval`）は維持するため HITL 経路は通る。approve 認可は **`(agent_name, tool_name)`
    単位**（resolver に渡す pending dict は `agent_name` を含む）で、同名ツールでも別 agent の
    approve は認可しない（Codex P1）。`approvals` 未指定時は #24 挙動を完全維持（中断 → 全観点
    inconclusive → verdict fail。ただし criteria に `ApprovalGate` を含めた場合は中断時の承認待ちを
    決定的に採点する）。

    Args:
        target: 評価対象（AgentSpec=単体 / HandoffGraph・WorkflowGraph=横断）。
        dataset: 評価ケース列（利用者提供）。
        judge: 採点用 LLM（`JudgeConfig` または model を直接）。model 直接時は内部で
            `JudgeConfig(model=judge)` にラップする。キーワード専用・必須。
        criteria: 評価観点列（データセット共通）。None で標準品質セット
            `(Relevance(), Safety(), Conciseness(), Faithfulness())`。
        registry: 横断対象の specs 供給経路。HandoffGraph は必須、WorkflowGraph は AGENT ノードを
            含む場合のみ必要（関数ノードのみなら不要）。AgentSpec 単体評価では不要。
        config: 評価設定（並列度 / タイムアウト / verdict ポリシー / テレメトリ）。None で既定。
        langfuse: Langfuse 観測シンク設定（任意）。None で送信スキップ（ローカル結果のみ）。
        approvals: 承認自動解決の判定関数（`(pending_dict) -> bool`・plain な承認待ち
            `{"tool_name", "call_id", "agent_name"}` を受け approve(True)/reject(False) を返す）。
            None で自動解決しない（中断は inconclusive / ApprovalGate のみ採点・後方互換）。
        tool_mocks: **agent スコープのネスト dict**（`{agent_name: {tool_name: 値 |
            callable(args: dict)}}`）。例: `{"account-agent": {"delete_account": "(mock)"}}`。
            `approvals` で approve するツールは当該 agent のエントリにモックを必ず指定する（未指定 /
            実差し替え不能なら approve 時に ValueError・本物の危険ツール実行を構造的に阻止）。
            **build 前の宣言層**（AgentSpec.tools / registry の各 spec）で同名ツールの**実行だけ**を
            差し替え、`needs_approval` は維持する（HITL 経路を消さない）。横断対象では利用者
            registry をクローンした派生 registry に適用するため、利用者 registry / キャッシュ済み
            Agent は一切変更しない。**制限**: `register_factory` 登録の agent は spec 実体を持たず
            tools をモック化できないため、その上の承認ツールは差し替えられず（approve 時に
            fail-closed の ValueError になる）。

    Returns:
        統合 verdict 付きの `EvaluationResult`（必ず返る・Langfuse 未設定/失敗でもローカル完結）。

    Raises:
        TypeError: 未対応の target 型の場合。
        ValueError: HandoffGraph に registry 未供給、同名 Criterion の重複、または approvals が
            approve した `(agent, tool)` がモックへ差し替えられていない場合。
        ImportError: deepeval 未導入の場合（採点不能）。langfuse 指定時に langfuse 未導入の場合。
    """
    effective_config = config or EvaluationConfig()
    effective_criteria = tuple(criteria) if criteria is not None else default_criteria()
    _check_duplicate_names(effective_criteria)
    judge_config = judge if isinstance(judge, JudgeConfig) else JudgeConfig(model=judge)
    knockout_names = frozenset(c.name for c in effective_criteria if c.knockout)

    # tool_mocks（agent スコープのネスト dict）指定時は **build 前の宣言（spec / registry）層**で
    # ツール実行を差し替える（build 済み Agent / 利用者 registry を mutate せず P2-1/P2-2 を解消）。
    # `normalize` が派生 Agent と「実際に差し替えた (agent, tool) ペアの集合」を返す。approve 認可
    # の根拠を tool_mocks のキーではなく実差し替えに置き、同名ツールでも別 agent を認可しない
    # （fail-closed・Codex P1）。SDK 型操作は `_adapters`、registry/spec のクローンは `_target`。
    agent, replaced_tools = _target.normalize(target, registry, tool_mocks=tool_mocks)

    semaphore = asyncio.Semaphore(max(1, effective_config.concurrency))

    async def _run_one(case: EvalCase) -> CaseResult:
        async with semaphore:
            return await _evaluate_case(
                case,
                agent=agent,
                criteria=effective_criteria,
                judge_config=judge_config,
                config=effective_config,
                resolver=approvals,
                replaced_tools=replaced_tools,
            )

    case_results = await asyncio.gather(*(_run_one(case) for case in dataset))

    verdict = compute_verdict(
        case_results,
        knockout=knockout_names,
        inconclusive_policy=effective_config.inconclusive_policy,
        required_criteria=effective_config.required_criteria,
    )

    result = EvaluationResult(
        target_id=_target.target_id(target),
        cases=list(case_results),
        verdict=verdict,
    )

    if langfuse is not None:
        from ..._adapters import langfuse_send

        langfuse_send(
            result,
            langfuse,
            cases=list(dataset),
            prompt_text=_target.extract_prompt(target),
        )

    return result


def _check_duplicate_names(criteria: Sequence[Criterion]) -> None:
    """同名 `Criterion` の重複を検出して明示エラーにする。

    Args:
        criteria: 評価観点列。

    Raises:
        ValueError: 同名の Criterion が複数含まれる場合。
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for c in criteria:
        if c.name in seen:
            duplicates.add(c.name)
        seen.add(c.name)
    if duplicates:
        raise ValueError(f"duplicate criterion names: {sorted(duplicates)}")
