"""会話サービスの HITL（承認待ち）内部ヘルパ群（共有コア・agents 非依存）。

`ConversationService` の HITL 承認まわりの内部ヘルパ（`_*_pending` クラスタ + ストリーム調停
`_stream_outcome`）を、`ConversationService` インスタンスを第 1 引数に受け取るフリー関数として
集約する。会話 store / lock / registry 解決 / `_adapters` / `SessionPolicy` 等の内部状態には
渡された `service` 経由でアクセスし、状態の取り違え・lock の二重取得を起こさない（lock は呼び出し
元の公開メソッドが保持し、本モジュールの関数は取得しない）。

`ConversationService` の公開 HITL メソッド（`pending_approvals` / `resolve_approvals` /
`stream_resolve`）は `service.py` に残り、本モジュールのフリー関数へ委譲する（API 不変）。

実行・Session 生成は `_adapters`（agents 単一窓口）へ委譲し、本モジュールは agents を import
しない（NFR-1）。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ... import _adapters
from .types import (
    ApprovalDecision,
    ApprovalRequired,
    ConversationError,
    ConversationErrorCode,
    PendingApproval,
    StreamDelta,
    StreamDone,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .service import ConversationService
    from .store import ConversationEntry
    from .types import StreamEvent

__all__: list[str] = []


async def has_pending(service: ConversationService, entry: ConversationEntry) -> bool:
    """当該会話が未解決の承認待ちを持つか判定する（必要なら永続から復元・P1）。

    `entry.pending_state` が既に在ればそのまま True。無い場合は永続テーブルから復元を試み、
    復元できれば True（=未解決の承認待ちが存在）。揮発会話/中断無しなら False。新ターン開始の
    可否判定に使い、True のとき send/stream は新たな `Runner.run` を回さない。

    Args:
        service: 会話サービス本体（内部状態へのアクセス元）。
        entry: 対象会話エントリ（ロック取得済み）。

    Returns:
        未解決の承認待ちがあれば True。

    Raises:
        ConversationError: 永続復元時の `RunState` 復元失敗（`restore_pending` 由来）。
    """
    if entry.pending_state is None:
        await restore_pending(service, entry)
    return entry.pending_state is not None


def pending_from_state(
    service: ConversationService, entry: ConversationEntry
) -> list[PendingApproval]:
    """entry.pending_state の**実際の未解決** interruption を `PendingApproval` 列で返す。

    保存済みリストの差し引きではなく `_adapters.unresolved_pending`（state の承認状態で
    フィルタ）で算出する。これにより「state は解決済みだがリストに古い call_id が残る」不整合
    （再開失敗後の詰まり・P2-1）を防ぐ。state が無ければ空。

    Args:
        service: 会話サービス本体（内部状態へのアクセス元）。
        entry: 対象会話エントリ（ロック取得済み）。

    Returns:
        未解決の承認待ち一覧（call_id 単位）。全解決なら空リスト。
    """
    if entry.pending_state is None:
        return []
    return [
        PendingApproval(tool_name=p["tool_name"], call_id=p["call_id"])
        for p in _adapters.unresolved_pending(entry.pending_state)
    ]


def _decision_to_dict(decision: ApprovalDecision | dict[str, Any]) -> dict[str, Any]:
    """承認/却下を `_adapters.apply_approvals` が受ける plain dict 形へ正規化する。

    dict はそのまま通す（後方互換）。`ApprovalDecision` は `approve` 真偽を `"approve"` /
    `"reject"` 文字列へ写し、キー集合（`call_id` / `decision` / `rejection_message`）を揃える。

    Args:
        decision: 承認/却下（型付き `ApprovalDecision` または plain dict）。

    Returns:
        `{"call_id", "decision", "rejection_message"}` 形の plain dict。
    """
    if isinstance(decision, ApprovalDecision):
        return {
            "call_id": decision.call_id,
            "decision": "approve" if decision.approve else "reject",
            "rejection_message": decision.rejection_message,
        }
    return decision


async def prepare_and_apply(
    service: ConversationService,
    entry: ConversationEntry,
    conversation_id: str,
    decisions: list[ApprovalDecision | dict[str, Any]],
) -> list[PendingApproval]:
    """中断状態を確保し decisions を検証先行で適用、未解決の残承認待ちを返す（共通点）。

    `resolve_approvals` / `stream_resolve` の共通前処理。中断状態が無ければ復元し、無ければ
    `NO_PENDING_APPROVAL`。`_adapters.apply_approvals`（検証先行 2 パス）の結果に未知/解決済み
    があれば `UNKNOWN_APPROVAL` / `APPROVAL_ALREADY_RESOLVED` を raise する（このとき state は
    一切変更されない・FR-4）。適用後、残承認待ちは **state の実際の未解決 interruption** から
    算出し entry.pending_approvals へ同期して返す（保存済みリスト差し引きに非依存・P2-1）。

    Args:
        service: 会話サービス本体（内部状態へのアクセス元）。
        entry: 対象会話エントリ（ロック取得済み）。
        conversation_id: 対象会話 ID（エラーメッセージ用）。
        decisions: 適用する承認/却下（`ApprovalDecision` または plain dict の列）。

    Returns:
        未解決のまま残った承認待ち（call_id 単位）。全解決なら空リスト。

    Raises:
        ConversationError: 承認待ち無し / 未知 call_id / 解決済み call_id 再操作の場合
            （いずれも state を変更しない）。
    """
    if entry.pending_state is None:
        await restore_pending(service, entry)
    if entry.pending_state is None:
        raise ConversationError(
            ConversationErrorCode.NO_PENDING_APPROVAL,
            f"承認待ちがありません: {conversation_id!r}",
        )
    normalized = [_decision_to_dict(d) for d in decisions]
    apply = _adapters.apply_approvals(entry.pending_state, normalized)
    if apply.unknown:
        raise ConversationError(
            ConversationErrorCode.UNKNOWN_APPROVAL,
            f"未知の call_id です: {apply.unknown!r}",
        )
    if apply.already_resolved:
        raise ConversationError(
            ConversationErrorCode.APPROVAL_ALREADY_RESOLVED,
            f"既に解決済みの call_id です: {apply.already_resolved!r}",
        )
    # 残りは state の実際の未解決から算出し entry へ同期（再開失敗後リトライで詰まらせない）。
    entry.pending_approvals = pending_from_state(service, entry)
    return list(entry.pending_approvals)


async def stream_outcome(
    service: ConversationService,
    entry: ConversationEntry,
    resolved: str,
    agent: Any,
    input_value: Any,
    *,
    state: Any,
) -> AsyncIterator[StreamEvent | ApprovalRequired]:
    """ストリーム実行/再開を回し `StreamDelta`/`StreamDone`/`ApprovalRequired` を yield する。

    `_adapters.run_streamed_outcome` の `str`（断片）/ `RunOutcome`（終端）を判別し、中断なし
    なら `StreamDone`、中断ありなら `ApprovalRequired` を流す。中断状態の保持/永続化/クリアも
    ここで調停する。SDK 例外は `StreamError` へ畳む。`state` 非 None は再開（入力は RunState）。

    Args:
        service: 会話サービス本体（内部状態へのアクセス元）。
        entry: 対象会話エントリ（ロックは呼び出し側が取得済み）。
        resolved: 解決済みエージェント名（`agent_name` 更新用）。
        agent: 実行する SDK Agent（解決済み・不透明）。
        input_value: 新規ターンならユーザー入力、再開なら `RunState`。
        state: 再開時の `RunState`（不透明）。新規ターンなら None。

    Yields:
        `StreamDelta` → `StreamDone`、中断時 `ApprovalRequired`、エラー時 `StreamError`。
    """
    try:
        async for chunk in _adapters.run_streamed_outcome(
            agent, input_value, session=entry.session
        ):
            if isinstance(chunk, str):
                yield StreamDelta(text=chunk)
                continue
            # chunk は終端の RunOutcome（plain）。
            entry.agent_name = resolved
            if chunk.interrupted:
                await capture_pending(service, entry, chunk)
                yield ApprovalRequired(approvals=list(entry.pending_approvals))
                return
            entry.turn_count += 1
            if state is not None:
                # 完了: 終端 RunOutcome の受領は再開ストリーム run が session へ履歴追記を完了
                # したことを意味する（run_streamed は終端まで回し切ってから RunOutcome を出す・
                # SDK 契約）。その後に専用テーブルの中断状態を DELETE する（履歴コミット→DELETE
                # の順・NFR-4(a)）。DELETE 失敗時は次回復元で再提示され収束する。
                await clear_pending(service, entry)
            yield StreamDone(final_output=chunk.final_output or "")
    except Exception as exc:  # noqa: BLE001 - SDK 例外を構造化エラーへ変換
        if state is not None:
            # 再開ストリームの失敗: state は解決適用済みのまま保持し、entry を state の実状
            # （全解決=空）へ同期して再永続する。空 decisions での再 stream_resolve が resume を
            # やり直せる（一時エラーで詰まらせない・P2-1）。新規ターンの失敗は state を持たない
            # ため同期不要（中断状態は無い）。
            entry.pending_approvals = pending_from_state(service, entry)
            await persist_pending(service, entry)
        # SDK 例外は StreamError へ畳んで 1 件 yield し、即終端する（以降を流さない）。
        yield service._to_conversation_error(exc).to_stream_error()
        return


async def capture_pending(
    service: ConversationService, entry: ConversationEntry, outcome: Any
) -> None:
    """中断 `RunOutcome` を entry へ保持し、永続会話なら専用テーブルへ upsert する。

    中断 `RunOutcome` の受領は当該 run が session への履歴追記を完了して戻ったことを意味する
    （SDK 契約）。本メソッドはその後に呼ばれ、専用テーブルを新中断状態で upsert する（履歴
    コミット→upsert の順・NFR-4(a)）。upsert は冪等で再試行に耐える。

    Args:
        service: 会話サービス本体（内部状態へのアクセス元）。
        entry: 対象会話エントリ（ロック取得済み）。
        outcome: 中断を表す `_adapters.RunOutcome`（`interrupted=True`）。
    """
    entry.pending_state = outcome.state
    entry.pending_approvals = [
        PendingApproval(tool_name=p["tool_name"], call_id=p["call_id"]) for p in outcome.pending
    ]
    await persist_pending(service, entry)


async def persist_pending(service: ConversationService, entry: ConversationEntry) -> None:
    """entry の中断状態を永続会話なら専用テーブルへ upsert する（揮発はメモリのみ）。

    永続キーは `entry.session_id`（FR-10: 復元は同 session_id・新 conversation_id で行う）。
    RunState を生んだ解決済みエージェント名（`entry.agent_name`）も併せて保存し、復元時の
    initial_agent 解決（D-Resume）に使う。

    Args:
        service: 会話サービス本体（内部状態へのアクセス元）。
        entry: 対象会話エントリ（中断状態を保持済み・ロック取得済み）。
    """
    db_path = service._policy.pending_db_path(persist=entry.persist)
    if db_path is None or entry.pending_state is None:
        return
    run_state_json = await _adapters.serialize_state(entry.pending_state)
    pending_json = json.dumps(
        [{"tool_name": p.tool_name, "call_id": p.call_id} for p in entry.pending_approvals]
    )
    _adapters.save_pending_approval(
        db_path,
        entry.session_id,
        entry.agent_name or "",
        run_state_json,
        pending_json,
    )


async def clear_pending(service: ConversationService, entry: ConversationEntry) -> None:
    """entry の中断状態をクリアし、永続会話なら専用テーブルから削除する（FR-6）。

    永続テーブルの削除キーは `entry.session_id`（保存と同じキー・別 session を取り違えない）。

    Args:
        service: 会話サービス本体（内部状態へのアクセス元）。
        entry: 対象会話エントリ（ロック取得済み）。
    """
    entry.pending_state = None
    entry.pending_approvals = []
    db_path = service._policy.pending_db_path(persist=entry.persist)
    if db_path is not None:
        _adapters.delete_pending_approval(db_path, entry.session_id)


async def restore_pending(service: ConversationService, entry: ConversationEntry) -> None:
    """永続会話で entry に中断状態が無い場合、専用テーブルから復元する（D-Resume）。

    専用テーブルを `entry.session_id` で引き、中断状態があれば `RunState` を復元して entry に
    載せる（FR-10: 復元は同 session_id・新 conversation_id で行うため session_id キーで引く）。
    initial_agent は**永続レコードに保存された agent_name**（RunState を生んだ解決済み名）を
    registry 解決して与える。再起動跨ぎで `entry.agent_name` は None に倒れるため、これに依存
    せず永続値を使う。永続 agent_name が空/未解決ならエントリエージェントへフォールバックする。
    揮発会話/中断無しなら何もしない。

    Args:
        service: 会話サービス本体（内部状態へのアクセス元）。
        entry: 対象会話エントリ（ロック取得済み）。
    """
    db_path = service._policy.pending_db_path(persist=entry.persist)
    if db_path is None:
        return
    record = _adapters.load_pending_approval(db_path, entry.session_id)
    if record is None or not record.get("run_state"):
        return
    # 永続された agent_name を優先（再起動跨ぎで entry.agent_name は None のため）。空なら
    # エントリエージェントへフォールバック（D-Resume）。
    saved_agent = record.get("agent_name") or None
    initial_name = service._resolve_entry_name(saved_agent)
    agent = service._resolve_agent(initial_name)
    try:
        # from_string（SDK）は破損 JSON / スキーマ不整合で SDK 例外を投げうる。境界へ生で
        # 漏らさず ConversationError へ変換する（NFR-5）。pending_approvals / resolve_approvals
        # / stream_resolve の呼び出し境界はこの変換済み例外を受ける。
        entry.pending_state = await _adapters.deserialize_state(agent, record["run_state"])
    except ConversationError:
        raise
    except Exception as exc:  # noqa: BLE001 - SDK 例外を構造化エラーへ変換
        raise service._to_conversation_error(exc) from exc
    entry.agent_name = initial_name
    try:
        pending_items = json.loads(record.get("pending") or "[]")
    except (TypeError, ValueError):
        pending_items = []
    entry.pending_approvals = [
        PendingApproval(
            tool_name=str(item.get("tool_name", "")),
            call_id=str(item.get("call_id", "")),
        )
        for item in pending_items
        if isinstance(item, dict)
    ]
