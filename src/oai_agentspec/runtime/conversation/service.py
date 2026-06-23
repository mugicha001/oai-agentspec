"""会話コアサービス（共有コア・agents 非依存・公開 API）。

registry 登録済みエージェントとローカルで会話する単一実装点（NFR-3 の核）。会話 store
（`conversation_id -> エントリ`・会話毎ロック）と session 管理（SDK `Session`・session_id
連動）を束ね、非ストリーミング（`send`）/ ストリーミング（`stream`）の両モードを提供する。

実行・Session 生成は `_adapters`（agents 単一窓口）へ委譲し、本モジュールは agents を
import しない（NFR-1）。Agent / Session は不透明型（`Any`）で扱い、registry.get() の戻りも
不透明に扱う。不正エージェント名・不正 conversation_id・モデル未注入は構造化された
`ConversationError` に変換し、SDK 例外を生で漏らさない。

agents / create_conversation / list_sessions（D5 一覧/復元）/ send / stream を提供する。
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from ... import _adapters
from ...constants import CONVERSATION_ID_PREFIX
from . import approvals
from .session import SessionPolicy
from .store import ConversationEntry, ConversationStore
from .types import (
    ApprovalDecision,
    ApprovalRequired,
    ConversationError,
    ConversationErrorCode,
    PendingApproval,
    SendResult,
    SendStatus,
    SessionInfo,
    StreamDelta,
    StreamDone,
    StreamError,
)

# 復元時に表示する過去履歴の既定取得件数（直近 N 件）。
DEFAULT_HISTORY_LIMIT = 10

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ...registry import AgentRegistry
    from .types import StreamEvent


class ConversationService:
    """共有コア会話サービス（registry + 会話 store + session 管理の単一実装点）。

    利用者提供の `AgentRegistry` を受け取り、会話の作成・送受信（非ストリーミング /
    ストリーミング）を提供する。会話毎ロックで同一 conversation の同時実行を排他する。
    """

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        session_policy: SessionPolicy | None = None,
        entry_agent: str | None = None,
    ) -> None:
        """会話サービスを生成する。

        Args:
            registry: 利用者提供の `AgentRegistry`（名前一覧 / Agent 解決に使う）。
            session_policy: session 生成方針（永続化先・compaction 設定）。None で既定。
            entry_agent: 会話の起点エージェント名（CLI の「エントリ起点」会話に使う）。
                None なら registry 登録順の先頭（`registry.entry_name`）を既定採用する。
        """
        self._registry = registry
        self._policy = session_policy or SessionPolicy()
        self._store = ConversationStore()
        self._entry_agent = entry_agent

    # ------------------------------------------------------------------
    # エージェント一覧 / エントリ
    # ------------------------------------------------------------------
    def agents(self) -> list[str]:
        """登録済みエージェント名の一覧を返す。

        Returns:
            登録済みエージェント名（昇順）。
        """
        return self._registry.names()

    def entry_agent(self) -> str | None:
        """会話の起点エージェント名を返す（CLI の「エントリ起点」会話に使う）。

        コンストラクタで明示された場合はそれを、無指定なら registry 登録順の先頭を返す。
        登録が 1 つも無ければ None。明示エントリ名の実在検証は行わない（利用者責任）。

        Returns:
            起点エージェント名。決定できなければ None。
        """
        if self._entry_agent is not None:
            return self._entry_agent
        return self._registry.entry_name

    # ------------------------------------------------------------------
    # 会話作成
    # ------------------------------------------------------------------
    async def create_conversation(
        self,
        *,
        conversation_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """新規会話を作成し conversation_id を返す。

        `session_id` を明示すると当該 session でファイル永続化（再起動後 resume 可）し、
        無指定なら in-memory（揮発）の session を割り当てる（D3・session_id 連動）。既存の
        session_id（`list_sessions` で得たもの等）を渡せば、その永続化 session に紐づく新規
        会話となり、続く send / stream は過去履歴を踏まえて継続する（D5・復元）。

        Args:
            conversation_id: 会話 ID。None で自動採番（`conv-<uuid4>`）。
            session_id: SDK Session の session_id。明示で永続化、None で in-memory。
                既存 session_id を渡すと過去履歴から続きを開始する（復元）。

        Returns:
            生成された会話 ID。

        Raises:
            ConversationError: conversation_id が既に存在する場合
                （`CONVERSATION_ALREADY_EXISTS`）。
        """
        cid = conversation_id or f"{CONVERSATION_ID_PREFIX}{uuid.uuid4().hex}"
        persist = session_id is not None
        sid = session_id or cid
        session = _adapters.make_session(
            sid,
            db_path=self._policy.db_path_for(persist=persist),
            **self._policy.compaction_kwargs(),
        )
        entry = ConversationEntry(
            conversation_id=cid, session_id=sid, session=session, persist=persist
        )
        # 重複チェックと挿入は store.add が同一ロック内で原子的に行う（TOCTOU レース回避）。
        # 別 get() で事前チェックすると並行作成時にすり抜けて上書きが起きるため使わない。
        if not await self._store.add(entry):
            # 重複/レースで登録されなかった Session は破棄する。file-backed SQLiteSession は
            # sqlite 接続を抱えるため close してリソースリークを防ぐ。
            await _adapters.close_session(session)
            raise ConversationError(
                ConversationErrorCode.CONVERSATION_ALREADY_EXISTS,
                f"conversation_id は既に存在します: {cid!r}",
            )
        return cid

    # ------------------------------------------------------------------
    # session 一覧 / 復元（D5）
    # ------------------------------------------------------------------
    async def list_sessions(self) -> list[SessionInfo]:
        """ファイル永続化された過去 session のメタ情報を一覧する（D5・更新時刻降順）。

        SessionPolicy の永続化先 db（既定 `memory/conversations.db`）を `_adapters` 経由で
        素 SELECT し、各 session の更新時刻・ターン数・先頭発話プレビューを導出する。in-memory
        （session_id 無指定・揮発）会話はプロセス内限定のため列挙対象外。db が未作成なら空リスト。

        Returns:
            `SessionInfo` の列（最終更新の新しい順）。永続化会話が無ければ空リスト。
        """
        db_path = self._policy.persisted_db_path()
        return [
            SessionInfo(
                session_id=meta["session_id"],
                updated_at=meta["updated_at"],
                turn_count=meta["turn_count"],
                preview=meta["preview"],
            )
            for meta in _adapters.list_session_meta(db_path)
        ]

    async def session_history(
        self, session_id: str, *, limit: int | None = DEFAULT_HISTORY_LIMIT
    ) -> list[dict[str, Any]]:
        """指定 session の過去履歴アイテムを時系列で返す（復元時の表示用・D5）。

        永続化先 db を `_adapters` 経由で読み、`session_id` の履歴アイテム（dict）を時系列で
        返す。`limit` 指定時は直近 `limit` 件のみ（既定 `DEFAULT_HISTORY_LIMIT`）。db / session
        が無ければ空リスト。

        Args:
            session_id: 取得対象の session_id。
            limit: 返す最大件数（直近側）。None で全件。既定は直近 10 件。

        Returns:
            履歴アイテム（dict）の時系列リスト。
        """
        db_path = self._policy.persisted_db_path()
        return _adapters.get_session_items(db_path, session_id, limit=limit)

    # ------------------------------------------------------------------
    # 会話送信（非ストリーミング）
    # ------------------------------------------------------------------
    async def send(self, agent_name: str | None, text: str, *, conversation_id: str) -> SendResult:
        """非ストリーミングで 1 ターン会話し最終応答 or 承認待ちを返す。

        **未解決の承認待ちがある間は新ターンを開始しない**（P1・安全性）。当該会話が承認待ち
        （`entry.pending_state` 保持、または永続から復元して存在）を持つ場合、新たな `Runner.run`
        を回さず（session に履歴を追記せず）、既存の承認待ちをそのまま返す（`status="pending"`）。
        新規テキストは処理しない（先に approve/reject で解決させる）。これにより古い `RunState` が
        変異済み session に対して順序外で再開・実行される事故を防ぐ。

        承認待ちが無い場合のみ実行し、`_adapters` の検知で中断ありなら最終出力を返さず承認待ち
        （`status="pending"`）を返す。中断状態は entry に保持し永続会話なら conversations.db の専用
        テーブルへ保存する（FR-10）。中断なしは従来どおり最終応答（`status="final"`）を返す（NFR-6）。

        Args:
            agent_name: 会話相手のエージェント名（registry 登録済み）。None で
                エントリエージェント（`entry_agent()`）を起点に会話する。
            text: ユーザー入力テキスト。
            conversation_id: 対象会話 ID（`create_conversation` の戻り）。

        Returns:
            最終応答（`status="final"`・`output`）または承認待ち（`status="pending"`・`pending`）。

        Raises:
            ConversationError: 不正エージェント名 / 不正 conversation_id / モデル未注入 /
                実行時エラーの場合（SDK 例外は構造化エラーへ変換する）。
        """
        entry = await self._require_entry(conversation_id)
        resolved = self._resolve_entry_name(agent_name)
        agent = self._resolve_agent(resolved)
        async with entry.lock:
            # 未解決の承認待ちがあれば新ターンを開始せず既存の承認待ちを返す（P1）。
            if await approvals.has_pending(self, entry):
                return SendResult(status=SendStatus.PENDING, pending=list(entry.pending_approvals))
            try:
                outcome = await _adapters.DefaultRunnerAdapter(None).run_outcome(
                    agent, text, session=entry.session
                )
            except Exception as exc:  # noqa: BLE001 - SDK 例外を構造化エラーへ変換
                raise self._to_conversation_error(exc) from exc
            entry.agent_name = resolved
            if outcome.interrupted:
                await approvals.capture_pending(self, entry, outcome)
                return SendResult(status=SendStatus.PENDING, pending=list(entry.pending_approvals))
            entry.turn_count += 1
            return SendResult(status=SendStatus.FINAL, output=outcome.final_output or "")

    # ------------------------------------------------------------------
    # 会話送信（ストリーミング）
    # ------------------------------------------------------------------
    async def stream(
        self, agent_name: str | None, text: str, *, conversation_id: str
    ) -> AsyncIterator[StreamEvent | ApprovalRequired]:
        """ストリーミングで 1 ターン会話しテキスト断片 / 完了 / 承認待ちを逐次 yield する。

        **未解決の承認待ちがある間は新ターンを開始しない**（P1・安全性）。当該会話が承認待ちを
        持つ場合、新たな `Runner.run` を回さず（session に履歴を追記せず）、`ApprovalRequired` を
        1 件 yield して `StreamDone` を出さずに終端する。新規テキストは処理しない（先に approve/
        reject で解決させる）。古い `RunState` の順序外再開・実行を防ぐ。

        承認待ちが無い場合のみ、会話毎ロックを取った状態で `_adapters.run_streamed_outcome` を
        回し、`StreamDelta`（逐次 token）を流す。中断なしなら最後に `StreamDone`（最終出力）を
        yield する（既存 3 メンバのみ・NFR-6）。中断ありなら `StreamDone` を出さず、専用イベント
        `ApprovalRequired` を 1 件 yield して終端する（`StreamEvent` Union 非混入・D-Compat）。
        エラーは `StreamError` を 1 件 yield して終端する（SDK 例外を生で漏らさない）。

        Args:
            agent_name: 会話相手のエージェント名（registry 登録済み）。None で
                エントリエージェント（`entry_agent()`）を起点に会話する。
            text: ユーザー入力テキスト。
            conversation_id: 対象会話 ID。

        Yields:
            `StreamDelta`（断片）→ `StreamDone`（最終出力）。中断時は `ApprovalRequired`、
            エラー時は `StreamError`。
        """
        try:
            entry = await self._require_entry(conversation_id)
            resolved = self._resolve_entry_name(agent_name)
            agent = self._resolve_agent(resolved)
        except ConversationError as exc:
            yield exc.to_stream_error()
            return

        async with entry.lock:
            # 未解決の承認待ちがあれば新ターンを開始せず既存の承認待ちを再提示する（P1）。
            try:
                has_pending = await approvals.has_pending(self, entry)
            except ConversationError as exc:
                yield exc.to_stream_error()
                return
            if has_pending:
                yield ApprovalRequired(approvals=list(entry.pending_approvals))
                return
            async for event in approvals.stream_outcome(
                self, entry, resolved, agent, text, state=None
            ):
                yield event

    # ------------------------------------------------------------------
    # 承認待ち取得 / 承認・却下（HITL）
    # ------------------------------------------------------------------
    async def pending_approvals(self, conversation_id: str) -> list[PendingApproval]:
        """現在の承認待ち一覧を返す（冪等・復元直後の再提示にも使う・FR-3/FR-10）。

        entry が中断状態を持てばそのまま返す。永続会話で entry に中断状態が無い場合は
        conversations.db の専用テーブルから読み出して `RunState` を復元し entry に載せる
        （サーバ再起動跨ぎ復元・D-Resume）。中断が無ければ空リスト。

        Args:
            conversation_id: 対象会話 ID。

        Returns:
            承認待ち一覧（call_id 単位の `PendingApproval`）。中断なしなら空リスト。

        Raises:
            ConversationError: conversation_id が未登録の場合。
        """
        entry = await self._require_entry(conversation_id)
        async with entry.lock:
            if entry.pending_state is None:
                await approvals.restore_pending(self, entry)
            return list(entry.pending_approvals)

    async def resolve_approvals(
        self, conversation_id: str, decisions: list[ApprovalDecision | dict[str, Any]]
    ) -> SendResult:
        """承認/却下を call_id 単位で適用し、全解決なら再開して最終応答 or 残承認待ちを返す。

        会話ロック下で `_adapters.apply_approvals` を適用する。全解決（未指定 call_id が残らない）
        なら `Runner.run(input=pending_state)` で再開し、完了なら最終応答（`status="final"`）を、
        再度中断したなら段階解決として残承認待ち（`status="pending"`）を返す。部分解決（未指定
        call_id 残）なら再開せず残承認待ちを返す（FR-7）。完了で中断状態をクリアし、永続会話なら
        専用テーブルから削除する（FR-6/FR-10）。

        Args:
            conversation_id: 対象会話 ID。
            decisions: 適用する承認/却下。`ApprovalDecision` または
                `{"call_id", "decision", "rejection_message"}` dict の列（dict は後方互換）。

        Returns:
            最終応答（`status="final"`）または残承認待ち（`status="pending"`）。

        Raises:
            ConversationError: 承認待ち無し（`NO_PENDING_APPROVAL`）/ 未知 call_id
                （`UNKNOWN_APPROVAL`）/ 解決済み call_id 再操作（`APPROVAL_ALREADY_RESOLVED`）/
                不正 conversation_id / 実行時エラーの場合。
        """
        entry = await self._require_entry(conversation_id)
        async with entry.lock:
            remaining = await approvals.prepare_and_apply(self, entry, conversation_id, decisions)
            if remaining:
                # 部分解決: 再開せず残承認待ちを保持・再提示する（段階解決・FR-7）。
                await approvals.persist_pending(self, entry)
                return SendResult(status=SendStatus.PENDING, pending=list(remaining))
            agent = self._resolve_agent(self._resolve_entry_name(entry.agent_name))
            try:
                outcome = await _adapters.resume_outcome(
                    agent, entry.pending_state, session=entry.session
                )
            except Exception as exc:  # noqa: BLE001 - SDK 例外を構造化エラーへ変換
                # 再開失敗: state は解決適用済みのまま保持し、entry を state の実状（全解決=空）へ
                # 同期して再永続する。これにより空 decisions での再 resolve が resume をやり直せる
                # （一時エラーで会話が詰まらない・P2-1）。SDK 例外は構造化エラーへ変換して返す。
                entry.pending_approvals = approvals.pending_from_state(self, entry)
                await approvals.persist_pending(self, entry)
                raise self._to_conversation_error(exc) from exc
            if outcome.interrupted:
                # 再中断: 終端 RunOutcome 受領（=再開 run の session 追記がコミット済み・SDK 契約）
                # の後に専用テーブルを新中断状態で upsert する（capture_pending）。
                await approvals.capture_pending(self, entry, outcome)
                return SendResult(status=SendStatus.PENDING, pending=list(entry.pending_approvals))
            # 完了: 終端 RunOutcome の受領は再開 run が session へ履歴追記を完了したことを意味する
            # （Runner.run は完了まで待って戻る・SDK 契約）。その後に専用テーブルの中断状態を
            # session_id 条件付き DELETE する（履歴コミット→DELETE の順・NFR-4(a)）。DELETE 失敗時
            # も残存中断状態は次回復元で再提示され収束する（逆順は取りこぼしのため不可）。
            entry.turn_count += 1
            await approvals.clear_pending(self, entry)
            return SendResult(status=SendStatus.FINAL, output=outcome.final_output or "")

    async def stream_resolve(
        self, conversation_id: str, decisions: list[ApprovalDecision | dict[str, Any]]
    ) -> AsyncIterator[StreamEvent | ApprovalRequired]:
        """承認/却下を適用し、全解決ならストリーム再開してテキスト断片 / 完了を逐次 yield する。

        `resolve_approvals` のストリーミング版。全解決なら `_adapters.run_streamed_outcome` で
        `RunState` から再開し `StreamDelta` を逐次流し、完了で `StreamDone`、再度中断で
        `ApprovalRequired` を yield する。部分解決なら再開せず `ApprovalRequired`（残承認待ち）を
        yield する。承認系/実行時エラーは `StreamError` を 1 件 yield して終端する。

        Args:
            conversation_id: 対象会話 ID。
            decisions: 適用する承認/却下。`ApprovalDecision` または
                `{"call_id", "decision", "rejection_message"}` dict の列（dict は後方互換）。

        Yields:
            `StreamDelta` → `StreamDone`（再開完了）。残承認待ち時は `ApprovalRequired`、
            エラー時は `StreamError`。
        """
        try:
            entry = await self._require_entry(conversation_id)
        except ConversationError as exc:
            yield exc.to_stream_error()
            return

        async with entry.lock:
            try:
                remaining = await approvals.prepare_and_apply(
                    self, entry, conversation_id, decisions
                )
            except ConversationError as exc:
                yield exc.to_stream_error()
                return

            if remaining:
                await approvals.persist_pending(self, entry)
                yield ApprovalRequired(approvals=list(remaining))
                return

            resolved_name = self._resolve_entry_name(entry.agent_name)
            agent = self._resolve_agent(resolved_name)
            async for event in approvals.stream_outcome(
                self, entry, resolved_name, agent, entry.pending_state, state=entry.pending_state
            ):
                yield event

    # ------------------------------------------------------------------
    # 内部ヘルパ
    # ------------------------------------------------------------------
    async def _require_entry(self, conversation_id: str) -> ConversationEntry:
        """会話エントリを取得する（未登録なら構造化エラー）。

        Args:
            conversation_id: 対象会話 ID。

        Returns:
            会話エントリ。

        Raises:
            ConversationError: conversation_id が未登録の場合。
        """
        entry = await self._store.get(conversation_id)
        if entry is None:
            raise ConversationError(
                ConversationErrorCode.UNKNOWN_CONVERSATION,
                f"不明な conversation_id: {conversation_id!r}",
            )
        return entry

    def _resolve_entry_name(self, agent_name: str | None) -> str:
        """エージェント名を決める（None ならエントリエージェントへ解決）。

        Args:
            agent_name: 明示エージェント名。None でエントリ起点。

        Returns:
            解決されたエージェント名。

        Raises:
            ConversationError: 名前未指定でエントリも決定できない（registry が空）場合。
        """
        name = agent_name or self.entry_agent()
        if name is None:
            raise ConversationError(
                ConversationErrorCode.UNKNOWN_AGENT,
                "エージェント名が未指定で、エントリエージェントも決定できません"
                "（registry が空です）",
            )
        return name

    def _resolve_agent(self, agent_name: str) -> Any:
        """エージェント名を SDK Agent へ解決する（不正名なら構造化エラー）。

        Args:
            agent_name: registry 上のエージェント名。

        Returns:
            解決済みの Agent（不透明型）。

        Raises:
            ConversationError: 未登録名の場合。
        """
        try:
            return self._registry.get(agent_name)
        except KeyError as exc:
            raise ConversationError(
                ConversationErrorCode.UNKNOWN_AGENT,
                f"不明なエージェント名: {agent_name!r}",
            ) from exc

    @staticmethod
    def _to_conversation_error(exc: Exception) -> ConversationError:
        """SDK / 実行時の例外を構造化された `ConversationError` へ変換する。

        `AgentSpec` の `model` 省略（`model=None`）は SDK デフォルトモデルを使う正当な用法の
        ため、実行前にモデル未注入を構造的に拒否することはしない。モデル不備は実行時に SDK
        例外として現れるので、メッセージ文字列一致（`api_key` / `model`）で `MODEL_NOT_CONFIGURED`
        を**ベストエフォートで推定**する。**既知の限界**: 文字列一致はヒューリスティックで、
        `model` を含む無関係なエラーを誤分類しうる。判別が曖昧な場合は `EXECUTION_ERROR` に倒す。

        Args:
            exc: 変換元の例外。

        Returns:
            構造化された `ConversationError`。
        """
        if isinstance(exc, ConversationError):
            return exc
        message = str(exc)
        lowered = message.lower()
        # fallback ヒューリスティック（誤分類しうる既知の限界・上記 docstring 参照）。
        if "api_key" in lowered or "api key" in lowered or "model" in lowered:
            return ConversationError(
                ConversationErrorCode.MODEL_NOT_CONFIGURED,
                f"モデルが構成されていない可能性があります: {message}",
            )
        return ConversationError(
            ConversationErrorCode.EXECUTION_ERROR,
            f"会話実行中にエラーが発生しました: {message}",
        )


__all__ = [
    "ApprovalRequired",
    "ConversationService",
    "PendingApproval",
    "SendResult",
    "StreamDelta",
    "StreamDone",
    "StreamError",
]
