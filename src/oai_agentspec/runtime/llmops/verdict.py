"""統合 verdict 計算（FR-5・純粋関数・I/O なし）。

`compute_verdict` に FR-5 の全規則を集約する。観点リストに `tool_correctness` /
`handoff_correctness` を含めても規則は不変（API 互換）。`.types` のみに依存し外部 SDK を
import しない。knockout 観点集合は呼び出し側（evaluator が `Criterion.knockout` から導出）が
渡す（既定は空集合）。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .types import CaseResult, CriterionResult, CriterionStatus, Verdict


def _all_criterion_results(case_results: Iterable[CaseResult]) -> list[CriterionResult]:
    """全ケースの全 `CriterionResult` を flat に集約する。

    Args:
        case_results: ケース結果の反復可能。

    Returns:
        全ケース横断の観点結果リスト（flat）。
    """
    flat: list[CriterionResult] = []
    for case in case_results:
        flat.extend(case.criteria)
    return flat


def compute_verdict(
    case_results: Sequence[CaseResult],
    *,
    knockout: frozenset[str] = frozenset(),
    inconclusive_policy: Verdict = Verdict.FAIL,
    required_criteria: frozenset[str] | None = None,
) -> Verdict:
    """全ケースの観点結果から統合 verdict を計算する（FR-5・純関数）。

    **入力定義**: 全ケースの全 `CriterionResult` を flat 集計して 1 つの合否を出す
    （CI ゲートが「データセット全体で 1 つの合否」を期待するため・§6 推奨）。ケース単位の
    verdict は統合せず、観点結果を横断 flat で評価する。

    規則（順序付き・上から評価し最初に該当した結果を返す）:
        1. 母集合: `skip` / `not_applicable` を除外する（母集合が空なら `fail`・fail-closed）。
        2. knockout（呼び出し側が `Criterion.knockout` から導出して渡す集合）: 当該観点が母集合内で
           `fail` なら verdict 即 `fail`（fail-closed・上書き不可）。`not_applicable` の knockout
           観点は判定対象外（母集合除外済み）。
        3. missing-pair fail-closed: `required_criteria` が母集合に存在しなければ `fail`。
           `required_criteria` 未指定時は母集合に出現した観点集合を要求集合とする
           （= 追加の missing 判定は発生しない）。
        4. 実 fail 優先: 母集合に `fail` が 1 件でもあれば `fail`。`inconclusive_policy=PASS` でも
           実在する fail を inconclusive で隠さない（fail は常に fail へ倒す・inconclusive ポリシー
           より先に評価する）。
        5. inconclusive: 残り（`pass` / `inconclusive` のみ）に `inconclusive` があれば
           `inconclusive_policy`（既定 `fail`）で解決する。
        6. 残りが全 `pass` なら `pass`。

    Args:
        case_results: 評価したケース結果の列。
        knockout: knockout 観点集合（当該観点 fail で即 fail）。
        inconclusive_policy: 母集合に inconclusive があるときに解決する verdict（既定 fail）。
        required_criteria: 母集合に存在を要求する観点集合。None なら母集合の観点集合を要求集合
            とする（追加 missing 判定なし）。

    Returns:
        統合 verdict（`Verdict.PASS` または `Verdict.FAIL`）。
    """
    flat = _all_criterion_results(case_results)

    # 1. 母集合（skip / not_applicable を除外）。
    population = [
        r for r in flat if r.status not in (CriterionStatus.SKIP, CriterionStatus.NOT_APPLICABLE)
    ]
    if not population:
        return Verdict.FAIL

    present = {r.criterion for r in population}

    # 2. knockout fail-closed（母集合内の fail のみ・not_applicable は既に除外済み）。
    for r in population:
        if r.criterion in knockout and r.status == CriterionStatus.FAIL:
            return Verdict.FAIL

    # 3. missing-pair fail-closed。
    required = required_criteria if required_criteria is not None else frozenset(present)
    if not required.issubset(present):
        return Verdict.FAIL

    # 4. 実 fail 優先: 母集合に fail が 1 件でもあれば fail。inconclusive_policy=PASS でも実在する
    #    fail を inconclusive で隠さない（inconclusive ポリシーより先に評価する）。
    if any(r.status == CriterionStatus.FAIL for r in population):
        return Verdict.FAIL

    # 5. inconclusive ポリシー（fail が無い場合のみ・残りは pass / inconclusive）。
    if any(r.status == CriterionStatus.INCONCLUSIVE for r in population):
        return inconclusive_policy

    # 6. 残りは全 pass。
    return Verdict.PASS
