"""HITL 承認/却下を `RunState` へ適用する委譲アダプタ（SDK 結合を `_adapters` に閉じる・NFR-1）。

`apply_approvals`（承認/却下の検証先行 2 パス適用）/ `unresolved_pending`（未解決の承認待ち
算出）/ `resume_outcome`（中断状態からの再開）/ `resume_with_observation`（再開 + 実行トレース
捕捉の合成・LLMOps の HITL 完了採点用）を提供する。SDK 結合（`agents` の `Runner` /
`RunContextWrapper`）は本モジュール内に閉じ、外へは plain な値（plain dict / `RunOutcome` /
`ObservedRun`）のみを渡す。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents import Runner

from .run_context import unwrap_run_context
from .runner import ApplyResult, RunOutcome, _outcome_from_result, _pending_agent_name

if TYPE_CHECKING:
    from ..runtime.llmops.types import ObservedRun

__all__ = [
    "apply_approvals",
    "resume_outcome",
    "resume_with_observation",
    "unresolved_pending",
]


def _approval_status(state: Any, item: Any) -> bool | None:
    """`RunState` の context から当該 ToolApprovalItem の解決状態を読む（True/False/None）。

    SDK の `RunContextWrapper.is_tool_approved(tool_name, call_id)` を引き、approve 済み=True /
    reject 済み=False / 未解決=None を返す。context / API が無い実装では None（未解決扱い）を返す
    （検証先行の安全側）。SDK 内部（`_context` / `is_tool_approved`）への結合は本 `_adapters` に
    閉じる（NFR-1）。

    Args:
        state: 中断状態の SDK `RunState`（不透明 `Any`）。
        item: 対象の `ToolApprovalItem`。

    Returns:
        approve 済み True / reject 済み False / 未解決 None。
    """
    context = getattr(state, "_context", None)
    is_approved = getattr(context, "is_tool_approved", None)
    if is_approved is None:
        return None
    tool_name = getattr(item, "tool_name", None)
    call_id = getattr(item, "call_id", None)
    return is_approved(tool_name, call_id)


def apply_approvals(state: Any, decisions: list[dict[str, Any]]) -> ApplyResult:
    """承認/却下の plain decisions を不透明 `RunState` へ**検証先行（2 パス）**で適用する。

    `decisions` は plain（`{"call_id": str, "decision": "approve"|"reject",
    "rejection_message": str | None}` のリスト）。

    1. **検証パス（state を一切変更しない）**: 全 decisions を `state.interruptions` 内の
       `ToolApprovalItem` へ call_id で引き当て、未引き当て=unknown、既解決（context が True/False
       を返す call_id）=already_resolved を集める。
    2. **適用パス**: unknown / already_resolved が **1 件も無いときに限り** approve/reject を
       state へ適用する。1 件でもあれば state へ一切適用せず unknown/already_resolved のみ返す。

    これにより混在バッチ（正常 + 未知/解決済み）でも部分適用による不整合を生まず、FR-4 の
    「無効な承認操作では会話状態は変化しない」を満たす。未知/解決済みは例外を投げず構造化結果
    で返す（SDK 例外を境界に漏らさない・NFR-5）。

    **fail-closed（NFR-7）**: decision は `"approve"` の明示一致のときだけ approve し、それ以外
    （`"reject"` / 未知値 / typo / 空 / 欠落）はすべて reject（非実行）へ倒す。承認必須ツールは
    曖昧な指定で実行させない安全側を採る。

    Args:
        state: 中断状態の SDK `RunState`（不透明 `Any`）。
        decisions: 適用する承認/却下の plain dict のリスト。

    Returns:
        適用結果（applied / unknown / already_resolved）の `ApplyResult`。
    """
    # SDK の承認待ち取得は型で口が異なる: `RunState` は `get_interruptions()` メソッド、`RunResult`
    # は `interruptions` プロパティ。本関数は不透明 `RunState` を受けるため `get_interruptions()` を
    # 使う（`_outcome_from_result` は `RunResult` 由来なので `result.interruptions` を使う）。属性/
    # メソッドの存在前提は L2 SDK 耐性トリップワイヤで別途検証する想定。`get_interruptions` が無い
    # 退行時は空にフォールバックし、全 decision が unknown になり FR-4 の安全側へ倒れる。
    get_interruptions = getattr(state, "get_interruptions", None)
    interruptions = list(get_interruptions() or []) if callable(get_interruptions) else []
    by_call_id: dict[str, Any] = {}
    for item in interruptions:
        call_id = getattr(item, "call_id", None)
        if call_id is not None:
            by_call_id[str(call_id)] = item

    # --- 検証パス（state は変更しない） ---
    result = ApplyResult(applied=[], unknown=[], already_resolved=[])
    valid: list[tuple[str, Any, dict[str, Any]]] = []
    for decision in decisions:
        call_id = str(decision.get("call_id", ""))
        item = by_call_id.get(call_id)
        if item is None:
            result.unknown.append(call_id)
            continue
        if _approval_status(state, item) is not None:
            result.already_resolved.append(call_id)
            continue
        valid.append((call_id, item, decision))

    # 無効な decision が 1 件でもあれば state へ一切適用しない（部分適用を避ける・FR-4）。
    if result.unknown or result.already_resolved:
        return result

    # --- 適用パス（全件 valid のときのみ state を変更） ---
    for call_id, item, decision in valid:
        # fail-closed: 明示 "approve" のときだけ approve、それ以外（未知値/空/欠落含む）は reject。
        if decision.get("decision") == "approve":
            state.approve(item)
        else:
            state.reject(item, rejection_message=decision.get("rejection_message"))
        result.applied.append(call_id)
    return result


def unresolved_pending(state: Any) -> list[dict[str, str]]:
    """不透明 `RunState` の中から**実際に未解決**の承認待ちを plain dict 列で返す。

    `state.get_interruptions()` の各 `ToolApprovalItem` を `_approval_status`（approve 済み True /
    reject 済み False / 未解決 None）でフィルタし、未解決（None）のものだけを `{"tool_name",
    "call_id", "agent_name"}` の plain dict で返す。共有コアの「残り承認待ち」は保存済みリストの
    差し引きではなく state の実状から算出する（P2-1: 再開失敗後リトライで詰まらせない）。
    `agent_name` は承認待ちを発生させた Agent 名（approve 認可を `(agent_name, tool_name)` 単位で
    行うため・追加キーなので既存利用者は無視してよい）。

    Args:
        state: 中断状態の SDK `RunState`（不透明 `Any`）。

    Returns:
        未解決の承認待ち一覧（plain dict のリスト）。全解決済みなら空リスト。
    """
    get_interruptions = getattr(state, "get_interruptions", None)
    interruptions = list(get_interruptions() or []) if callable(get_interruptions) else []
    pending: list[dict[str, str]] = []
    for item in interruptions:
        if _approval_status(state, item) is not None:
            continue
        call_id = getattr(item, "call_id", None)
        tool_name = getattr(item, "tool_name", None)
        pending.append(
            {
                "tool_name": "" if tool_name is None else str(tool_name),
                "call_id": "" if call_id is None else str(call_id),
                "agent_name": _pending_agent_name(item),
            }
        )
    return pending


async def resume_outcome(
    agent: Any,
    state: Any,
    *,
    context: Any = None,
    **runner_kwargs: Any,
) -> RunOutcome:
    """中断状態 `RunState` から `Runner.run(input=state)` で再開し `RunOutcome` を返す。

    再開後も `interruptions` が残れば再度中断として返す（段階解決）。共有コアは戻りの
    `RunOutcome` だけを見て中断 or 完了を判定する。

    Args:
        agent: 再開に使う SDK Agent（解決済み）。
        state: 承認/却下を適用済みの SDK `RunState`（不透明 `Any`）。
        context: 各実行へ素通しする共有 context（`RunContextWrapper` は `.context` を展開）。
        **runner_kwargs: `Runner.run` へ素通しする残りの kwarg（session 等）。

    Returns:
        再開後の中断 or 完了を表す `RunOutcome`。
    """
    raw_context = unwrap_run_context(context)
    result = await Runner.run(agent, state, context=raw_context, **runner_kwargs)
    return _outcome_from_result(result)


async def resume_with_observation(
    agent: Any,
    state: Any,
    *,
    context: Any = None,
    **runner_kwargs: Any,
) -> tuple[RunOutcome, ObservedRun]:
    """中断状態 `RunState` から再開し plain な `RunOutcome` + `ObservedRun` を返す（LLMOps）。

    `resume_outcome` と `routing.observe_run_result` の合成（生 `RunResult` を 1 回だけ取得し、
    最終出力 / 中断と routing 経路 + ツール呼び出しの両方を抽出する）。**生 `RunResult` は
    `_adapters` 外へ一切出さない**（NFR-1）。LLMOps の HITL 完了採点（承認の自動解決 → 再開 →
    完了採点）で resume 後の route / tool を捕捉するために使う。`resume_outcome` は不変。

    Args:
        agent: 再開に使う SDK Agent（解決済み）。
        state: 承認/却下を適用済みの SDK `RunState`（不透明 `Any`）。
        context: 各実行へ素通しする共有 context（`RunContextWrapper` は `.context` を展開）。
        **runner_kwargs: `Runner.run` へ素通しする残りの kwarg（session 等）。

    Returns:
        再開後の plain な `RunOutcome`（最終出力 / 中断）と `ObservedRun`（route + tool_calls）。
    """
    from .routing import observe_run_result

    raw_context = unwrap_run_context(context)
    result = await Runner.run(agent, state, context=raw_context, **runner_kwargs)
    return _outcome_from_result(result), observe_run_result(result)
