"""L2: ConversationService.send / stream を実 Agent + FakeModel で検証する。

registry に FakeModel エージェントを登録し、SQLiteSession(:memory:) で送受信・履歴継続・
ストリーミング逐次 delta・構造化エラー（不正名 / 不正 conversation_id）を確認する。
実 LLM は呼ばない。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from agents.items import ModelResponse
from agents.models.interface import Model

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.conversation import (
    CompactionConfig,
    ConversationError,
    ConversationErrorCode,
    ConversationService,
    SessionPolicy,
    StreamDelta,
    StreamDone,
    StreamError,
)
from oai_agentspec.runtime.conversation.service import ConversationService as _SvcForHelper

from _helpers.responses import text_response

pytestmark = pytest.mark.integration


class StreamingFakeModel(Model):
    """カンネドテキストを返し、ストリーミング時は delta + completed を流す Model。

    `_adapters` の workflow 用ストリーミング実装と同じ要領で、最終テキストを
    `ResponseTextDeltaEvent` に区切り `ResponseCompletedEvent` で終端する。FakeModel は
    stream_response 非対応のため、会話ストリーミング検証用に別途用意する。
    """

    def __init__(self) -> None:
        """空のレスポンスキューで生成する。"""
        self._responses: list[ModelResponse] = []
        self.calls: list[Any] = []

    def queue_text(self, text: str) -> StreamingFakeModel:
        """テキスト応答をキューに積む（自身を返す）。"""
        self._responses.append(text_response(text))
        return self

    async def get_response(
        self,
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        """キュー先頭の応答を返す（空なら空テキスト）。"""
        self.calls.append(input)
        if self._responses:
            return self._responses.pop(0)
        return text_response("")

    async def stream_response(  # type: ignore[override]
        self,
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """最終テキストを delta + completed イベントで流す（run_streamed 対応）。"""
        from oai_agentspec._adapters import _completed_event, _text_delta_events, _text_of

        response = await self.get_response(system_instructions, input, *args, **kwargs)
        text = _text_of(response)
        seq = 0
        for event in _text_delta_events(text):
            yield event
            seq = event.sequence_number + 1
        yield _completed_event(response.output, seq)


def _service_registry(name: str = "bot") -> AgentRegistry:
    """1 つの FakeModel エージェントを登録した AgentRegistry を作る。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name=name, instructions="b", model=StreamingFakeModel()))
    return reg


def _service_with_agent(model: Model, name: str = "bot") -> ConversationService:
    """1 つの FakeModel エージェントを登録した ConversationService を作る。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name=name, instructions="b", model=model))
    return ConversationService(reg)


@pytest.mark.asyncio
async def test_agents_lists_registered_names() -> None:
    """agents() が registry の登録名を返す。"""
    svc = _service_with_agent(StreamingFakeModel(), name="bot")
    assert svc.agents() == ["bot"]


@pytest.mark.asyncio
async def test_send_returns_final_output() -> None:
    """send が最終応答テキストを返す（in-memory session）。"""
    model = StreamingFakeModel().queue_text("hi there")
    svc = _service_with_agent(model)
    cid = await svc.create_conversation()
    out = await svc.send("bot", "hello", conversation_id=cid)
    assert out.status == "final"
    assert out.output == "hi there"


@pytest.mark.asyncio
async def test_send_continues_history_in_same_session() -> None:
    """同一会話の連続 send で履歴が session に積まれる（2 ターン目で前ターンが input に乗る）。"""
    model = StreamingFakeModel().queue_text("first").queue_text("second")
    svc = _service_with_agent(model)
    cid = await svc.create_conversation()
    await svc.send("bot", "turn-1", conversation_id=cid)
    await svc.send("bot", "turn-2", conversation_id=cid)
    # 2 ターン目の get_response の input には 1 ターン目の履歴が含まれる（session 継続の証跡）。
    second_input = model.calls[1]
    assert isinstance(second_input, list)
    assert len(second_input) > 1


@pytest.mark.asyncio
async def test_stream_yields_deltas_then_done() -> None:
    """stream が StreamDelta を逐次 yield し、最後に StreamDone を返す。"""
    model = StreamingFakeModel().queue_text("streaming output text")
    svc = _service_with_agent(model)
    cid = await svc.create_conversation()

    deltas: list[str] = []
    done: StreamDone | None = None
    async for event in svc.stream("bot", "go", conversation_id=cid):
        if isinstance(event, StreamDelta):
            deltas.append(event.text)
        elif isinstance(event, StreamDone):
            done = event
    assert len(deltas) >= 1
    assert "".join(deltas) == "streaming output text"
    assert done is not None
    assert done.final_output == "streaming output text"


@pytest.mark.asyncio
async def test_send_unknown_agent_raises_structured_error() -> None:
    """不正エージェント名は ConversationError(UNKNOWN_AGENT) になる。"""
    svc = _service_with_agent(StreamingFakeModel())
    cid = await svc.create_conversation()
    with pytest.raises(ConversationError) as exc:
        await svc.send("nope", "x", conversation_id=cid)
    assert exc.value.code == ConversationErrorCode.UNKNOWN_AGENT


@pytest.mark.asyncio
async def test_send_unknown_conversation_raises_structured_error() -> None:
    """不正 conversation_id は ConversationError(UNKNOWN_CONVERSATION) になる。"""
    svc = _service_with_agent(StreamingFakeModel())
    with pytest.raises(ConversationError) as exc:
        await svc.send("bot", "x", conversation_id="missing")
    assert exc.value.code == ConversationErrorCode.UNKNOWN_CONVERSATION


@pytest.mark.asyncio
async def test_stream_unknown_agent_yields_stream_error() -> None:
    """stream の不正名は StreamError を 1 件 yield して終端する（例外は漏らさない）。"""
    svc = _service_with_agent(StreamingFakeModel())
    cid = await svc.create_conversation()
    events = [e async for e in svc.stream("nope", "x", conversation_id=cid)]
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].code == ConversationErrorCode.UNKNOWN_AGENT.value


@pytest.mark.asyncio
async def test_stream_unknown_conversation_yields_stream_error() -> None:
    """stream の不正 conversation_id は StreamError を yield して終端する。"""
    svc = _service_with_agent(StreamingFakeModel())
    events = [e async for e in svc.stream("bot", "x", conversation_id="missing")]
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].code == ConversationErrorCode.UNKNOWN_CONVERSATION.value


@pytest.mark.asyncio
async def test_create_duplicate_conversation_raises() -> None:
    """同一 conversation_id の重複作成は CONVERSATION_ALREADY_EXISTS になる。"""
    svc = _service_with_agent(StreamingFakeModel())
    await svc.create_conversation(conversation_id="dup")
    with pytest.raises(ConversationError) as exc:
        await svc.create_conversation(conversation_id="dup")
    assert exc.value.code == ConversationErrorCode.CONVERSATION_ALREADY_EXISTS


@pytest.mark.asyncio
async def test_concurrent_create_same_id_only_one_succeeds() -> None:
    """同一 conversation_id の並行作成で先勝ち 1 件のみ成功し残りは衝突する（TOCTOU 回避）。"""
    svc = _service_with_agent(StreamingFakeModel())
    results = await asyncio.gather(
        *(svc.create_conversation(conversation_id="race") for _ in range(8)),
        return_exceptions=True,
    )
    ok = [r for r in results if not isinstance(r, BaseException)]
    conflicts = [
        r
        for r in results
        if isinstance(r, ConversationError)
        and r.code == ConversationErrorCode.CONVERSATION_ALREADY_EXISTS
    ]
    assert len(ok) == 1  # 成功は 1 件だけ（上書きされない）
    assert len(conflicts) == 7  # 残りは全て衝突エラー
    # store には 1 エントリのみ存在する。
    assert await svc._store.get("race") is not None  # noqa: SLF001


@pytest.mark.asyncio
async def test_send_runtime_api_key_error_maps_to_model_not_configured() -> None:
    """実行時の api_key 不備例外は MODEL_NOT_CONFIGURED へ分類する（ヒューリスティック）。"""
    svc = _service_with_agent(RaisingModel(RuntimeError("missing api_key")))
    cid = await svc.create_conversation()
    with pytest.raises(ConversationError) as exc:
        await svc.send("bot", "hi", conversation_id=cid)
    assert exc.value.code == ConversationErrorCode.MODEL_NOT_CONFIGURED


@pytest.mark.asyncio
async def test_stream_runtime_api_key_error_yields_model_not_configured() -> None:
    """stream の実行時 api_key 不備例外は MODEL_NOT_CONFIGURED の StreamError を yield する。"""
    svc = _service_with_agent(RaisingModel(RuntimeError("missing api_key")))
    cid = await svc.create_conversation()
    events = [e async for e in svc.stream("bot", "hi", conversation_id=cid)]
    assert isinstance(events[-1], StreamError)
    assert events[-1].code == ConversationErrorCode.MODEL_NOT_CONFIGURED.value


@pytest.mark.asyncio
async def test_model_none_agent_is_not_rejected_at_preflight(monkeypatch: Any) -> None:
    """model=None（SDK デフォルトモデル用法）は実行前に拒否せず Runner へ到達する（P2 回帰）。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="b"))  # model 未指定（デフォルトモデル）

    class _Result:
        final_output = "default-model-ok"

    async def _fake_run(self: Any, agent: Any, text: Any, *, session: Any = None, **kw: Any) -> Any:
        return _Result()

    monkeypatch.setattr("oai_agentspec._adapters.DefaultRunnerAdapter.run", _fake_run, raising=True)
    svc = ConversationService(reg)
    cid = await svc.create_conversation()
    out = await svc.send("bot", "hi", conversation_id=cid)
    # 前段で MODEL_NOT_CONFIGURED に倒れない（最終応答が返る）。
    assert out.status == "final"
    assert out.output == "default-model-ok"


@pytest.mark.asyncio
async def test_duplicate_create_closes_orphan_session(monkeypatch: Any) -> None:
    """会話 ID 重複時、登録されなかった orphan Session は close される（P3 リーク防止）。"""
    closed: list[str] = []

    class _FakeSession:
        def __init__(self, sid: str) -> None:
            self.sid = sid

        def close(self) -> None:
            closed.append(self.sid)

    def _fake_make_session(
        session_id: str,
        *,
        db_path: Any = None,
        enable_compaction: bool = False,
        client: Any = None,
        model: Any = None,
        compaction_options: Any = None,
    ) -> Any:
        return _FakeSession(session_id)

    monkeypatch.setattr("oai_agentspec._adapters.make_session", _fake_make_session, raising=True)
    svc = _service_with_agent(StreamingFakeModel())
    await svc.create_conversation(conversation_id="dup")
    with pytest.raises(ConversationError):
        await svc.create_conversation(conversation_id="dup")
    assert closed == ["dup"]  # 重複分の orphan のみ close（初回分は登録され生きている）


# ----------------------------------------------------------------------
# compaction 設定の make_session への伝播（plain kwargs 展開・FR-3）
# ----------------------------------------------------------------------


class _FakeClient:
    """実 API を叩かないダミー AsyncOpenAI 互換クライアント。"""


def _patch_capture_make_session(monkeypatch: Any) -> list[dict[str, Any]]:
    """make_session を引数捕捉する fake へ差し替え、捕捉先 list を返す。"""
    captured: list[dict[str, Any]] = []

    class _FakeSession:
        def __init__(self, sid: str) -> None:
            self.sid = sid

    def _fake_make_session(
        session_id: str,
        *,
        db_path: Any = None,
        enable_compaction: bool = False,
        client: Any = None,
        model: Any = None,
        compaction_options: Any = None,
    ) -> Any:
        captured.append(
            {
                "session_id": session_id,
                "db_path": db_path,
                "enable_compaction": enable_compaction,
                "client": client,
                "model": model,
                "compaction_options": compaction_options,
            }
        )
        return _FakeSession(session_id)

    monkeypatch.setattr("oai_agentspec._adapters.make_session", _fake_make_session, raising=True)
    return captured


@pytest.mark.asyncio
async def test_create_conversation_passes_compaction_kwargs(monkeypatch: Any) -> None:
    """compaction=CompactionConfig(enabled=True) で新 plain kwargs が make_session へ渡る。"""
    captured = _patch_capture_make_session(monkeypatch)
    client = _FakeClient()
    policy = SessionPolicy(
        compaction=CompactionConfig(enabled=True, client=client, model="gpt-4.1")
    )
    svc = ConversationService(_service_registry(), session_policy=policy)
    await svc.create_conversation()
    assert len(captured) == 1
    kw = captured[0]
    assert kw["enable_compaction"] is True
    assert kw["client"] is client
    assert kw["model"] == "gpt-4.1"
    assert kw["compaction_options"] == {}


@pytest.mark.asyncio
async def test_create_conversation_no_compaction_disables(monkeypatch: Any) -> None:
    """compaction=None のとき enable_compaction=False（plain 経路）で make_session を呼ぶ。"""
    captured = _patch_capture_make_session(monkeypatch)
    svc = ConversationService(_service_registry())  # 既定 SessionPolicy（compaction=None）
    await svc.create_conversation()
    assert len(captured) == 1
    assert captured[0]["enable_compaction"] is False


# ----------------------------------------------------------------------
# 実行中 SDK 例外 → 構造化エラー変換（EXECUTION_ERROR 経路 / 誤分類ヒューリスティック）
# ----------------------------------------------------------------------


class RaisingModel(Model):
    """get_response / stream_response で任意の例外を送出する Model（実行時エラー経路用）。"""

    def __init__(self, exc: Exception) -> None:
        """送出する例外を設定する。"""
        self._exc = exc

    async def get_response(
        self,
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        """設定済み例外を送出する。"""
        raise self._exc

    async def stream_response(  # type: ignore[override]
        self,
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """設定済み例外を送出する（イベントを流さない）。"""
        raise self._exc
        yield  # pragma: no cover - 到達しない（型のため）


@pytest.mark.asyncio
async def test_send_runtime_error_becomes_execution_error() -> None:
    """実行中の無関係な SDK 例外は EXECUTION_ERROR の構造化エラーへ変換される。"""
    svc = _service_with_agent(RaisingModel(RuntimeError("boom")))
    cid = await svc.create_conversation()
    with pytest.raises(ConversationError) as exc:
        await svc.send("bot", "x", conversation_id=cid)
    assert exc.value.code == ConversationErrorCode.EXECUTION_ERROR


@pytest.mark.asyncio
async def test_stream_runtime_error_yields_execution_error() -> None:
    """stream 実行中の SDK 例外は EXECUTION_ERROR の StreamError を yield して終端する。"""
    svc = _service_with_agent(RaisingModel(RuntimeError("boom")))
    cid = await svc.create_conversation()
    events = [e async for e in svc.stream("bot", "x", conversation_id=cid)]
    assert isinstance(events[-1], StreamError)
    assert events[-1].code == ConversationErrorCode.EXECUTION_ERROR.value


def test_to_conversation_error_passthrough_keeps_code() -> None:
    """既存の ConversationError はそのまま返す（二重ラップしない）。"""
    original = ConversationError(ConversationErrorCode.UNKNOWN_AGENT, "x")
    assert _SvcForHelper._to_conversation_error(original) is original


def test_to_conversation_error_model_keyword_maps_to_model_not_configured() -> None:
    """'model' を含む実行時例外は MODEL_NOT_CONFIGURED へ（既知のヒューリスティック）。"""
    err = _SvcForHelper._to_conversation_error(RuntimeError("invalid model name"))
    assert err.code == ConversationErrorCode.MODEL_NOT_CONFIGURED


def test_to_conversation_error_api_key_maps_to_model_not_configured() -> None:
    """'api_key' を含む実行時例外も MODEL_NOT_CONFIGURED へ分類される。"""
    err = _SvcForHelper._to_conversation_error(RuntimeError("missing api_key"))
    assert err.code == ConversationErrorCode.MODEL_NOT_CONFIGURED


def test_to_conversation_error_unrelated_maps_to_execution_error() -> None:
    """無関係なメッセージの例外は EXECUTION_ERROR へ分類される。"""
    err = _SvcForHelper._to_conversation_error(RuntimeError("network timeout"))
    assert err.code == ConversationErrorCode.EXECUTION_ERROR


# ----------------------------------------------------------------------
# 会話毎ロックの排他（同一 conversation への並行 send が直列化される）
# ----------------------------------------------------------------------


class GatedModel(Model):
    """get_response 内で同時実行数を観測し、最大同時実行数を記録する Model。"""

    def __init__(self) -> None:
        """カウンタを初期化する。"""
        self.active = 0
        self.max_active = 0

    async def get_response(
        self,
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        """実行中の同時実行数を観測し最大値を更新する（直列化されていれば常に 1）。"""
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        # 他コルーチンへ制御を譲り、ロックが無ければ同時侵入を許す窓を作る。
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.active -= 1
        return text_response("ok")

    async def stream_response(  # type: ignore[override]
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Any]:
        """未使用（非ストリーミング検証のみ）。"""
        raise NotImplementedError
        yield  # pragma: no cover


@pytest.mark.asyncio
async def test_concurrent_send_to_same_conversation_is_serialized() -> None:
    """同一 conversation への並行 send は会話毎ロックで直列化される（同時実行数が 1 以下）。"""
    model = GatedModel()
    svc = _service_with_agent(model)
    cid = await svc.create_conversation()
    await asyncio.gather(*(svc.send("bot", f"m{i}", conversation_id=cid) for i in range(5)))
    assert model.max_active == 1
