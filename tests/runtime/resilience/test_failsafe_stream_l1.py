"""L1: `failsafe_stream`（ストリーミングの宣言的な例外着地）の純検証。

`failsafe_stream(policy, source)` が `source`（任意の async iterable）を 1 回だけ消費して
中継し、`__anext__` の await 中に発生した宣言済み例外を、既配信要素を保ったまま末尾 1 要素の
`FailsafeResult` として yield して終了することを pin する。併せて以下を pin する:

- 正常完了は要素を同一オブジェクトのまま順序どおり中継し、`FailsafeResult` を出さない
  （`handlers={StopAsyncIteration: ...}` を宣言しても正常終了は着地しない）。
- 未宣言例外・`handlers` 空・fallback 自身の例外は素通しし、監査（warning / `on_apply`）は
  発火しない。挿入順 first-match・fallback の 3 形・`last_agent` の 2 段解決・監査は
  `failsafe_call` と同じ規則に従う。
- `asyncio.CancelledError` / 明示 `aclose()`（`GeneratorExit`）/ `athrow()` で投げ込まれた
  例外は着地しない。明示 `aclose()` は source の `aclose` へ転送される。
- 受理契約（async iterable）の検査は呼び出し時点・try の外で 1 回だけ行う。

外部依存 (agents / openai) なし（source は自作の async generator / 自作クラスで作る）。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator
from types import MappingProxyType
from typing import Any

import pytest

from oai_agentspec.runtime.resilience._failsafe import (
    RUNNING_AGENT,
    FailsafeHandler,
    FailsafePolicy,
    FailsafeResult,
    failsafe_call,
    failsafe_stream,
)

pytestmark = pytest.mark.unit

_LOGGER_NAME = "oai_agentspec.resilience"


class MyError(Exception):
    """テスト用のユーザー定義例外（`Exception` サブクラス・許容されるキー）。"""


class MySubError(MyError):
    """`MyError` のサブクラス（isinstance マッチと first-match の検証用）。"""


class _FakeRunData:
    """SDK `RunErrorDetails` の代役（解決に必要な `last_agent` のみを持つ最小形）。"""

    def __init__(self, last_agent: Any) -> None:
        self.last_agent = last_agent


class BudgetLikeError(Exception):
    """`RunBudgetExceeded` 相当の代役（例外自身が `last_agent` 属性を持つ）。"""

    def __init__(self, message: str, last_agent: Any) -> None:
        super().__init__(message)
        self.last_agent = last_agent


class HybridError(Exception):
    """`run_data` と `last_agent` の双方を持つ例外（読み取り先の優先順位の検証用）。"""

    def __init__(self, message: str, run_data: Any, last_agent: Any) -> None:
        super().__init__(message)
        self.run_data = run_data
        self.last_agent = last_agent


AGENT_FROM_RUN_DATA = object()
"""`exc.run_data.last_agent` 由来で解決された値を同一性で確認する番兵（Agent の代役）。"""

AGENT_FROM_ATTRIBUTE = object()
"""`exc.last_agent` 由来で解決された値を同一性で確認する番兵（Agent の代役）。"""

AGENT_PER_EXCEPTION = object()
"""段 1（`FailsafeHandler.last_agent`）に置いた具体 agent を同一性で確認する番兵。"""

AGENT_POLICY_FALLBACK = object()
"""段 2（`FailsafePolicy.fallback_last_agent`）に置いた具体 agent を同一性で確認する番兵。"""


def _records_of(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    """resilience logger が出した指定レベルのレコードのみを抽出する。"""
    return [r for r in caplog.records if r.name == _LOGGER_NAME and r.levelno == level]


async def _source(items: list[Any], exc: BaseException | None = None) -> AsyncIterator[Any]:
    """`items` を順に yield し、`exc` があれば最後に送出する async generator。"""
    for item in items:
        yield item
    if exc is not None:
        raise exc


async def _collect(stream: AsyncIterator[Any]) -> list[Any]:
    """`stream` を最後まで消費して yield された要素を列で返す。"""
    return [item async for item in stream]


async def _collect_until_raise(stream: AsyncIterator[Any], received: list[Any]) -> None:
    """`stream` を消費し、受け取った要素を `received` へ追記する（例外はそのまま伝播）。"""
    async for item in stream:
        received.append(item)


class _RaiseThenYieldSource:
    """1 回目の `__anext__` で例外を送出し、2 回目以降は値を返す自作 async iterator。

    例外を送出した async generator は再開できないため、着地後に source が読まれないことを
    観測する目的で `__anext__` を自前実装する。`aclose` は持たない。
    """

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.anext_calls = 0

    def __aiter__(self) -> _RaiseThenYieldSource:
        return self

    async def __anext__(self) -> Any:
        self.anext_calls += 1
        if self.anext_calls == 1:
            raise self.exc
        return "continued"


class _NoAcloseSource:
    """`__aiter__` / `__anext__` のみを持ち `aclose` を持たない自作 async iterator。"""

    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)

    def __aiter__(self) -> _NoAcloseSource:
        return self

    async def __anext__(self) -> Any:
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class _CountingAiterSource:
    """`__aiter__` の呼び出し回数を記録する自作 async iterable。"""

    def __init__(self, items: list[Any]) -> None:
        self._items = items
        self.aiter_calls = 0

    def __aiter__(self) -> AsyncIterator[Any]:
        self.aiter_calls += 1
        return _NoAcloseSource(self._items)


class _CountingAcloseSource:
    """`aclose` の呼び出し回数を数える自作 async iterator（終了経路は `outcome` で切り替える）。

    async generator の source は終了済みになると `aclose()` が no-op になり転送を観測できない
    ため、`aclose` を自前実装して回数を記録する。`items` を返し切った後の `__anext__` は
    `outcome` が None なら `StopAsyncIteration`（正常完了）、例外なら毎回それを送出する。
    """

    def __init__(self, items: list[Any], outcome: Exception | None = None) -> None:
        self._items = list(items)
        self._outcome = outcome
        self.aclose_calls = 0

    def __aiter__(self) -> _CountingAcloseSource:
        return self

    async def __anext__(self) -> Any:
        if self._items:
            return self._items.pop(0)
        if self._outcome is not None:
            raise self._outcome
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.aclose_calls += 1


# ---------------------------------------------------------------------------
# 関数の形（plain def・戻り値は async iterator）
# ---------------------------------------------------------------------------


def test_failsafe_stream_coroutine関数ではなく呼び出しでasync_iteratorを返す() -> None:
    """公開関数は plain def で、呼び出しの戻り値は `__anext__` を持つ async iterator。"""
    policy = FailsafePolicy(handlers={MyError: "landed"})

    assert inspect.iscoroutinefunction(failsafe_stream) is False
    stream = failsafe_stream(policy, _NoAcloseSource([]))
    assert hasattr(stream, "__anext__")
    assert hasattr(stream, "__aiter__")


# ---------------------------------------------------------------------------
# 正常完了 / 着地（0 件 / N 件）/ 着地後の終了
# ---------------------------------------------------------------------------


async def test_failsafe_stream_正常完了は全要素を同一オブジェクトのまま順序どおり中継する() -> None:
    """handlers 非空でも正常完了なら要素をそのまま中継し、`FailsafeResult` を出さない。"""
    items = [object(), object(), object()]
    policy = FailsafePolicy(handlers={MyError: "landed"})

    received = await _collect(failsafe_stream(policy, _source(items)))

    assert len(received) == len(items)
    for got, expected in zip(received, items, strict=True):
        assert got is expected
    assert not any(isinstance(item, FailsafeResult) for item in received)


async def test_failsafe_stream_StopAsyncIterationを宣言しても正常終了は着地しない(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`StopAsyncIteration` は `Exception` 派生だが正常終了として扱い、着地も監査もしない。"""
    items = [object(), object()]
    seen: list[FailsafeResult] = []

    def _on_apply(result: FailsafeResult) -> None:
        seen.append(result)

    policy = FailsafePolicy(handlers={StopAsyncIteration: "landed"}, on_apply=_on_apply)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        received = await _collect(failsafe_stream(policy, _source(items)))

    assert len(received) == 2
    assert received[0] is items[0]
    assert received[1] is items[1]
    assert seen == []
    assert _records_of(caplog, logging.WARNING) == []


async def test_failsafe_stream_0件転送で宣言例外ならFailsafeResult1件のみを受け取る() -> None:
    """最初の `__anext__` で宣言済み例外が出たら、受け取る列は `FailsafeResult` 1 件だけ。"""
    exc = MyError("boom")
    policy = FailsafePolicy(handlers={MyError: "landed"})

    received = await _collect(failsafe_stream(policy, _source([], exc)))

    assert len(received) == 1
    result = received[0]
    assert isinstance(result, FailsafeResult)
    assert result.final_output == "landed"
    assert result.matched_type is MyError
    assert result.exception is exc
    assert result.last_agent is None


async def test_failsafe_stream_N件転送後の宣言例外は既配信の末尾にFailsafeResultが付く() -> None:
    """既配信 N 件は上書きされず同一オブジェクトのまま残り、末尾 1 要素として着地する。"""
    items = [object(), object(), object()]
    exc = MyError("boom")
    policy = FailsafePolicy(handlers={MyError: "landed"})

    received = await _collect(failsafe_stream(policy, _source(items, exc)))

    assert len(received) == len(items) + 1
    for got, expected in zip(received[:-1], items, strict=True):
        assert got is expected
    assert not any(isinstance(item, FailsafeResult) for item in received[:-1])
    last = received[-1]
    assert isinstance(last, FailsafeResult)
    assert last.final_output == "landed"
    assert last.exception is exc
    assert last.matched_type is MyError


async def test_failsafe_stream_着地後は終了しsourceの続きを読まない() -> None:
    """`FailsafeResult` の後の `anext` は `StopAsyncIteration` で、source は再度読まれない。"""
    exc = MyError("boom")
    source = _RaiseThenYieldSource(exc)
    policy = FailsafePolicy(handlers={MyError: "landed"})

    stream = failsafe_stream(policy, source)
    first = await anext(stream)

    assert isinstance(first, FailsafeResult)
    assert first.exception is exc
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert source.anext_calls == 1


# ---------------------------------------------------------------------------
# 透過（未宣言例外 / handlers 空）
# ---------------------------------------------------------------------------


async def test_failsafe_stream_未宣言例外は既配信の後にそのまま伝播し監査しない(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """どのキーにもマッチしない例外は既配信の後に素通しし、chaining も監査もしない。"""
    items = [object(), object()]
    exc = ValueError("unhandled")
    seen: list[FailsafeResult] = []

    def _on_apply(result: FailsafeResult) -> None:
        seen.append(result)

    policy = FailsafePolicy(handlers={MyError: "landed"}, on_apply=_on_apply)
    received: list[Any] = []

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        with pytest.raises(ValueError) as exc_info:
            await _collect_until_raise(failsafe_stream(policy, _source(items, exc)), received)

    assert exc_info.value is exc
    assert exc_info.value.__cause__ is None
    assert len(received) == 2
    assert received[0] is items[0]
    assert received[1] is items[1]
    assert seen == []
    assert _records_of(caplog, logging.WARNING) == []


async def test_failsafe_stream_handlers空は例外をそのまま伝播し監査しない(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """handlers 空なら一切着地せず、例外はそのまま伝播して warning も出ない。"""
    items = [object()]
    exc = MyError("boom")
    policy = FailsafePolicy(handlers={})
    received: list[Any] = []

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        with pytest.raises(MyError) as exc_info:
            await _collect_until_raise(failsafe_stream(policy, _source(items, exc)), received)

    assert exc_info.value is exc
    assert exc_info.value.__cause__ is None
    assert len(received) == 1
    assert received[0] is items[0]
    assert _records_of(caplog, logging.WARNING) == []


async def test_failsafe_stream_handlers空でも正常完了はそのまま中継する() -> None:
    """handlers 空でも正常完了は完全透過し、要素は同一オブジェクトのまま届く。"""
    items = [object(), object()]
    policy = FailsafePolicy(handlers={})

    received = await _collect(failsafe_stream(policy, _source(items)))

    assert len(received) == 2
    assert received[0] is items[0]
    assert received[1] is items[1]


# ---------------------------------------------------------------------------
# 挿入順 first-match
# ---------------------------------------------------------------------------


async def test_failsafe_stream_親型を先に宣言すると子型の例外も親型で着地する() -> None:
    """親キーを先に宣言した場合、サブクラス例外でも親キーが first-match する。"""
    policy = FailsafePolicy(handlers={MyError: "parent", MySubError: "child"})

    received = await _collect(failsafe_stream(policy, _source([], MySubError("boom"))))

    result = received[-1]
    assert isinstance(result, FailsafeResult)
    assert result.matched_type is MyError
    assert result.final_output == "parent"


async def test_failsafe_stream_子型を先に宣言すると子型で着地する() -> None:
    """宣言順を入れ替えると first-match の結果も入れ替わる（順序が意味を持つ）。"""
    policy = FailsafePolicy(handlers={MySubError: "child", MyError: "parent"})

    received = await _collect(failsafe_stream(policy, _source([], MySubError("boom"))))

    result = received[-1]
    assert isinstance(result, FailsafeResult)
    assert result.matched_type is MySubError
    assert result.final_output == "child"


# ---------------------------------------------------------------------------
# fallback の形（値 / sync callable / async callable / 例外）
# ---------------------------------------------------------------------------


async def test_failsafe_stream_非callable値のfallbackはそのまま着地値になる() -> None:
    """handlers の値が callable でなければ、その値自体が final_output になる。"""
    landing = {"answer": 42}
    policy = FailsafePolicy(handlers={MyError: landing})

    received = await _collect(failsafe_stream(policy, _source([], MyError("boom"))))

    result = received[-1]
    assert isinstance(result, FailsafeResult)
    assert result.final_output is landing


async def test_failsafe_stream_sync_callableのfallbackは例外を受け取り戻り値が着地値になる() -> (
    None
):
    """sync callable の fallback は捕捉例外を単一引数に呼ばれ、戻り値が final_output になる。"""
    exc = MyError("boom")
    seen: list[Exception] = []

    def _fb(received: Exception) -> str:
        seen.append(received)
        return f"recovered:{received}"

    policy = FailsafePolicy(handlers={MyError: _fb})

    received = await _collect(failsafe_stream(policy, _source([object()], exc)))

    result = received[-1]
    assert isinstance(result, FailsafeResult)
    assert result.final_output == "recovered:boom"
    assert len(seen) == 1
    assert seen[0] is exc


async def test_failsafe_stream_async_callableのfallbackはawaitされた結果が着地値になる() -> None:
    """async callable の fallback は await され、その結果が final_output になる。"""
    exc = MyError("boom")
    seen: list[Exception] = []

    async def _fb(received: Exception) -> str:
        await asyncio.sleep(0)
        seen.append(received)
        return "async-recovered"

    policy = FailsafePolicy(handlers={MyError: _fb})

    received = await _collect(failsafe_stream(policy, _source([object()], exc)))

    result = received[-1]
    assert isinstance(result, FailsafeResult)
    assert result.final_output == "async-recovered"
    assert len(seen) == 1
    assert seen[0] is exc


async def test_failsafe_stream_fallback自身の例外は素通しし監査は発火しない(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """fallback が失敗したら着地は成立せず、その例外が伝播して warning / on_apply は出ない。"""
    fallback_exc = RuntimeError("fallback failed")
    called: list[str] = []

    def _fb(received: Exception) -> str:
        raise fallback_exc

    def _on_apply(result: FailsafeResult) -> None:
        called.append("on_apply")

    policy = FailsafePolicy(handlers={MyError: _fb}, on_apply=_on_apply)
    items = [object()]
    received: list[Any] = []

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        with pytest.raises(RuntimeError) as exc_info:
            await _collect_until_raise(
                failsafe_stream(policy, _source(items, MyError("boom"))), received
            )

    assert exc_info.value is fallback_exc
    assert len(received) == 1
    assert received[0] is items[0]
    assert called == []
    assert [r for r in caplog.records if r.name == _LOGGER_NAME] == []


# ---------------------------------------------------------------------------
# last_agent の 2 段解決
# ---------------------------------------------------------------------------


async def _landed_last_agent(policy: FailsafePolicy, exc: Exception) -> Any:
    """`exc` を送出する source を `policy` で包み、着地結果の `last_agent` を返す。"""
    received = await _collect(failsafe_stream(policy, _source([object()], exc)))
    result = received[-1]
    assert isinstance(result, FailsafeResult)
    return result.last_agent


async def test_failsafe_stream_RUNNING_AGENTは例外のlast_agent属性から解決される() -> None:
    """段 1 に `RUNNING_AGENT` を置くと `exc.last_agent` から実行中の agent を解決する。"""
    policy = FailsafePolicy(
        handlers={
            BudgetLikeError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT),
        }
    )

    got = await _landed_last_agent(policy, BudgetLikeError("over", AGENT_FROM_ATTRIBUTE))

    assert got is AGENT_FROM_ATTRIBUTE


async def test_failsafe_stream_RUNNING_AGENTはrun_dataのlast_agentを優先する() -> None:
    """`run_data` を持つ例外は `run_data.last_agent` が `exc.last_agent` より優先される。"""
    policy = FailsafePolicy(
        handlers={HybridError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)}
    )
    exc = HybridError("boom", _FakeRunData(AGENT_FROM_RUN_DATA), AGENT_FROM_ATTRIBUTE)

    got = await _landed_last_agent(policy, exc)

    assert got is AGENT_FROM_RUN_DATA


async def test_failsafe_stream_段1で解決できなければfallback_last_agentへ落ちる() -> None:
    """段 1 の `RUNNING_AGENT` が解決できない例外では段 2 の全体規定が採られる。"""
    policy = FailsafePolicy(
        handlers={MyError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)},
        fallback_last_agent=AGENT_POLICY_FALLBACK,
    )

    got = await _landed_last_agent(policy, MyError("boom"))

    assert got is AGENT_POLICY_FALLBACK


async def test_failsafe_stream_具体agentの指定はそのまま入る() -> None:
    """段 1 に具体の agent を置くと、例外の属性によらずその値がそのまま入る。"""
    policy = FailsafePolicy(
        handlers={
            BudgetLikeError: FailsafeHandler(fallback="landed", last_agent=AGENT_PER_EXCEPTION),
        },
        fallback_last_agent=AGENT_POLICY_FALLBACK,
    )

    got = await _landed_last_agent(policy, BudgetLikeError("over", AGENT_FROM_ATTRIBUTE))

    assert got is AGENT_PER_EXCEPTION


async def test_failsafe_stream_last_agent無指定ならNone() -> None:
    """どの段も無指定なら、例外が `last_agent` を持っていても解決は走らず None。"""
    policy = FailsafePolicy(handlers={BudgetLikeError: "landed"})

    got = await _landed_last_agent(policy, BudgetLikeError("over", AGENT_FROM_ATTRIBUTE))

    assert got is None


# ---------------------------------------------------------------------------
# 監査（warning / on_apply）
# ---------------------------------------------------------------------------


async def test_failsafe_stream_既定でwarningログが出てマッチキー名と例外型名を含む(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """log_on_apply 既定 True では WARNING がトレースバック付きで 1 件出る。"""
    policy = FailsafePolicy(handlers={MyError: "landed"})
    exc = MySubError("boom")

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        await _collect(failsafe_stream(policy, _source([object()], exc)))

    records = _records_of(caplog, logging.WARNING)
    assert len(records) == 1
    assert records[0].exc_info[1] is exc
    msg = records[0].getMessage()
    assert msg.startswith("failsafe applied: matched_type=")
    assert f"matched_type={MyError.__name__}" in msg
    assert f"exception_type={MySubError.__name__}" in msg
    assert "boom" in msg


async def test_failsafe_stream_log_on_apply_Falseでwarningが出ない(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """log_on_apply=False なら着地しても warning は emit されない。"""
    policy = FailsafePolicy(handlers={MyError: "landed"}, log_on_apply=False)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        received = await _collect(failsafe_stream(policy, _source([], MyError("boom"))))

    assert isinstance(received[-1], FailsafeResult)
    assert _records_of(caplog, logging.WARNING) == []


async def test_failsafe_stream_on_apply_syncはyieldされた結果と同一のFailsafeResultを受け取る() -> (
    None
):
    """sync の on_apply は着地時に 1 回呼ばれ、受け取る結果は yield された要素そのもの。"""
    exc = MyError("boom")
    seen: list[FailsafeResult] = []

    def _on_apply(result: FailsafeResult) -> None:
        seen.append(result)

    policy = FailsafePolicy(handlers={MyError: "landed"}, on_apply=_on_apply)

    received = await _collect(failsafe_stream(policy, _source([object()], exc)))

    assert len(seen) == 1
    assert seen[0] is received[-1]
    assert seen[0].exception is exc


async def test_failsafe_stream_on_apply_asyncはawaitされyieldされた結果と同一を受け取る() -> None:
    """async の on_apply は await され、受け取る結果は yield された要素そのもの。"""
    seen: list[FailsafeResult] = []

    async def _on_apply(result: FailsafeResult) -> None:
        await asyncio.sleep(0)
        seen.append(result)

    policy = FailsafePolicy(handlers={MyError: "landed"}, on_apply=_on_apply)

    received = await _collect(failsafe_stream(policy, _source([], MyError("boom"))))

    assert len(seen) == 1
    assert seen[0] is received[-1]


async def test_failsafe_stream_on_apply例外はerrorログで握り潰されFailsafeResultはyieldされる(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """on_apply が失敗しても FailsafeResult は yield され、error ログに記録される。"""

    callback_exc = RuntimeError("callback failed")

    def _on_apply(result: FailsafeResult) -> None:
        raise callback_exc

    policy = FailsafePolicy(handlers={MyError: "landed"}, on_apply=_on_apply)
    items = [object()]

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        received = await _collect(failsafe_stream(policy, _source(items, MyError("boom"))))

    assert len(received) == 2
    assert received[0] is items[0]
    assert isinstance(received[1], FailsafeResult)
    assert received[1].final_output == "landed"
    errors = _records_of(caplog, logging.ERROR)
    assert len(errors) == 1
    assert errors[0].exc_info[1] is callback_exc


# ---------------------------------------------------------------------------
# 協調キャンセル（CancelledError / 明示 aclose / athrow）
# ---------------------------------------------------------------------------


async def test_failsafe_stream_sourceのCancelledErrorは着地せず伝播する(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """source が CancelledError を送出しても着地せず伝播し、既配信は届いている。"""
    items = [object()]
    seen: list[FailsafeResult] = []

    def _on_apply(result: FailsafeResult) -> None:
        seen.append(result)

    policy = FailsafePolicy(handlers={MyError: "landed"}, on_apply=_on_apply)
    received: list[Any] = []

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        with pytest.raises(asyncio.CancelledError):
            await _collect_until_raise(
                failsafe_stream(policy, _source(items, asyncio.CancelledError())), received
            )

    assert len(received) == 1
    assert received[0] is items[0]
    assert seen == []
    assert _records_of(caplog, logging.WARNING) == []


async def test_failsafe_stream_宣言検証を迂回したCancelledErrorキーでも着地せず伝播する() -> None:
    """宣言側の検証を迂回した場合でも捕捉側が BaseException 系を素通しする。

    捕捉側（`except Exception` 限定）が BaseException 系を素通しする（ADR の二重防御の
    2 段目）。
    """
    policy = FailsafePolicy(handlers={MyError: "landed"})
    object.__setattr__(policy, "handlers", MappingProxyType({asyncio.CancelledError: "landed"}))
    cancelled = asyncio.CancelledError()
    received: list[Any] = []

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await _collect_until_raise(failsafe_stream(policy, _source(["a"], cancelled)), received)

    assert exc_info.value is cancelled
    assert received == ["a"]


async def test_failsafe_stream_明示acloseはsourceのfinallyへ転送され着地しない() -> None:
    """1 件受け取った後の `aclose()` は source の async generator の finally に到達させる。"""
    finalized: list[str] = []
    seen: list[FailsafeResult] = []

    async def _gen() -> AsyncIterator[Any]:
        try:
            yield "first"
            yield "second"
        finally:
            finalized.append("finally")

    def _on_apply(result: FailsafeResult) -> None:
        seen.append(result)

    policy = FailsafePolicy(handlers={MyError: "landed"}, on_apply=_on_apply)
    stream = failsafe_stream(policy, _gen())

    first = await anext(stream)
    assert first == "first"
    assert finalized == []

    await stream.aclose()  # type: ignore[attr-defined]

    assert finalized == ["finally"]
    assert seen == []
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


async def test_failsafe_stream_反復前のacloseはsourceのacloseへ転送しない() -> None:
    """1 要素も要求する前の `aclose()` は source の `aclose` を呼ばない（意図した契約）。

    戻り値は async generator で、未開始の async generator の `aclose` は本体を実行しない
    （PEP 525）。そのため本体の `finally` に到達せず転送は起きない。`__aiter__` で確保した
    資源の解放は利用者側の責務になる。
    """
    source = _CountingAcloseSource(["a"])
    policy = FailsafePolicy(handlers={MyError: "landed"})

    stream = failsafe_stream(policy, source)
    await stream.aclose()  # type: ignore[attr-defined]

    assert source.aclose_calls == 0


@pytest.mark.parametrize(
    ("outcome", "expect_landed", "expect_raise"),
    [
        (None, False, False),
        (MyError("boom"), True, False),
        (ValueError("unhandled"), False, True),
    ],
    ids=["正常完了", "宣言済み例外で着地", "未宣言例外が伝播"],
)
async def test_failsafe_stream_反復開始後の終了経路ではsourceのacloseへ1回転送する(
    outcome: Exception | None, expect_landed: bool, expect_raise: bool
) -> None:
    """反復を始めた後は、正常完了・着地・未宣言例外の伝播のいずれでも `aclose` が 1 回呼ばれる。"""
    source = _CountingAcloseSource(["a"], outcome)
    policy = FailsafePolicy(handlers={MyError: "landed"})
    received: list[Any] = []

    if expect_raise:
        with pytest.raises(ValueError) as exc_info:
            await _collect_until_raise(failsafe_stream(policy, source), received)
        assert exc_info.value is outcome
    else:
        await _collect_until_raise(failsafe_stream(policy, source), received)

    assert source.aclose_calls == 1
    assert received[0] == "a"
    assert len(received) == (2 if expect_landed else 1)
    if expect_landed:
        assert isinstance(received[1], FailsafeResult)
        assert received[1].exception is outcome


async def test_failsafe_stream_aclose非保持のsourceでもacloseと完走がエラーにならない() -> None:
    """`aclose` を持たない自作 source でも、明示 `aclose()` と完走のどちらも失敗しない。"""
    policy = FailsafePolicy(handlers={MyError: "landed"})

    closing = failsafe_stream(policy, _NoAcloseSource(["a", "b"]))
    assert await anext(closing) == "a"
    await closing.aclose()  # type: ignore[attr-defined]

    completed = await _collect(failsafe_stream(policy, _NoAcloseSource(["a", "b"])))
    assert completed == ["a", "b"]


class _RaisingAcloseSource(_CountingAcloseSource):
    """`aclose` が `aclose_exc` を送出する自作 async iterator（呼び出し回数も数える）。"""

    def __init__(
        self, items: list[Any], aclose_exc: Exception, outcome: Exception | None = None
    ) -> None:
        super().__init__(items, outcome)
        self.aclose_exc = aclose_exc

    async def aclose(self) -> None:
        self.aclose_calls += 1
        raise self.aclose_exc


@pytest.mark.parametrize(
    ("outcome", "explicit_close", "expect_landed"),
    [
        (None, False, False),
        (MyError("boom"), False, True),
        (None, True, False),
    ],
    ids=["正常完了後", "着地後", "明示aclose"],
)
async def test_failsafe_stream_aclose自身の例外は着地せず同一インスタンスで伝播する(
    caplog: pytest.LogCaptureFixture,
    outcome: Exception | None,
    explicit_close: bool,
    expect_landed: bool,
) -> None:
    """source の `aclose` が送出した例外は着地させず、そのインスタンスのまま利用者へ届く。"""
    aclose_exc = OSError("aclose boom")
    source = _RaisingAcloseSource(["a", "b"], aclose_exc, outcome)
    policy = FailsafePolicy(handlers={MyError: "landed", OSError: "must-not-land"})
    received: list[Any] = []

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        with pytest.raises(OSError) as exc_info:
            if explicit_close:
                stream = failsafe_stream(policy, source)
                received.append(await anext(stream))
                await stream.aclose()  # type: ignore[attr-defined]
            else:
                await _collect_until_raise(failsafe_stream(policy, source), received)

    assert exc_info.value is aclose_exc
    assert source.aclose_calls == 1
    assert received[0] == "a"
    landed = [item for item in received if isinstance(item, FailsafeResult)]
    assert all(r.final_output != "must-not-land" for r in landed)
    if expect_landed:
        assert len(landed) == 1
        assert landed[0].exception is outcome
        assert len(_records_of(caplog, logging.WARNING)) == 1
    else:
        assert landed == []
        assert _records_of(caplog, logging.WARNING) == []


async def test_failsafe_stream_未宣言例外の伝播中のaclose例外は元の例外を__context__に残す() -> (
    None
):
    """伝播中に `aclose` が送出すると、届くのは `aclose` 側の例外になる。

    元の例外は `__context__` に残る。
    """
    original = ValueError("unhandled")
    aclose_exc = OSError("aclose boom")
    source = _RaisingAcloseSource(["a"], aclose_exc, original)
    policy = FailsafePolicy(handlers={MyError: "landed"})
    received: list[Any] = []

    with pytest.raises(OSError) as exc_info:
        await _collect_until_raise(failsafe_stream(policy, source), received)

    assert exc_info.value is aclose_exc
    assert exc_info.value.__context__ is original
    assert source.aclose_calls == 1
    assert received == ["a"]


async def test_failsafe_stream_fallbackのStopAsyncIterationはRuntimeErrorへ変換され監査しない(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """fallback が `StopAsyncIteration` を送出すると `RuntimeError` へ変換される。

    async generator の仕様（PEP 525）による変換で、`failsafe_call` ではそのまま伝播する
    （同じ fallback でも観測される型が異なる）。着地は成立していないため warning も
    `on_apply` も発火しない。
    """
    stop = StopAsyncIteration("from fallback")
    called: list[str] = []

    def _fb(received: Exception) -> str:
        raise stop

    def _on_apply(result: FailsafeResult) -> None:
        called.append("on_apply")

    policy = FailsafePolicy(handlers={MyError: _fb}, on_apply=_on_apply)
    received: list[Any] = []

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        with pytest.raises(RuntimeError) as exc_info:
            await _collect_until_raise(
                failsafe_stream(policy, _source(["a"], MyError("boom"))), received
            )

    assert exc_info.value.__cause__ is stop
    assert received == ["a"]
    assert called == []
    assert [r for r in caplog.records if r.name == _LOGGER_NAME] == []

    async def _thunk() -> str:
        raise MyError("boom")

    with pytest.raises(StopAsyncIteration) as call_exc_info:
        await failsafe_call(policy, _thunk)
    assert call_exc_info.value is stop


async def test_failsafe_stream_athrowで投げ込まれた宣言例外は着地せず伝播する() -> None:
    """`yield` は try の外にあるため、`athrow()` の宣言済み例外は着地せずそのまま伝播する。"""
    seen: list[FailsafeResult] = []

    def _on_apply(result: FailsafeResult) -> None:
        seen.append(result)

    policy = FailsafePolicy(handlers={MyError: "landed"}, on_apply=_on_apply)
    stream = failsafe_stream(policy, _source(["first", "second"]))
    thrown = MyError("x")

    assert await anext(stream) == "first"
    with pytest.raises(MyError) as exc_info:
        await stream.athrow(thrown)  # type: ignore[attr-defined]

    assert exc_info.value is thrown
    assert seen == []


# ---------------------------------------------------------------------------
# 受理契約（呼び出し時点の TypeError・aiter は 1 回）
# ---------------------------------------------------------------------------


async def _async_gen_function() -> AsyncIterator[Any]:
    """呼び出し忘れの検証用 async generator 関数（関数そのものは async iterable でない）。"""
    yield "never"


async def _coroutine_function() -> str:
    """coroutine オブジェクトを作るための async 関数。"""
    return "value"


@pytest.mark.parametrize(
    "handlers",
    [{TypeError: "landed"}, {}],
    ids=["TypeError宣言あり", "handlers空"],
)
@pytest.mark.parametrize(
    "bad_source",
    [123, [1, 2], _async_gen_function],
    ids=["int", "list", "未呼び出しのasync_generator関数"],
)
def test_failsafe_stream_非async_iterableは呼び出し時点でTypeErrorになり着地しない(
    handlers: dict[type[Exception], Any], bad_source: Any
) -> None:
    """非 async iterable は `async for` の前・呼び出し時点で TypeError になり着地しない。"""
    policy = FailsafePolicy(handlers=handlers)

    with pytest.raises(TypeError):
        failsafe_stream(policy, bad_source)


@pytest.mark.parametrize(
    "handlers",
    [{TypeError: "landed"}, {}],
    ids=["TypeError宣言あり", "handlers空"],
)
def test_failsafe_stream_coroutineは呼び出し時点でTypeErrorになり着地しない(
    handlers: dict[type[Exception], Any],
) -> None:
    """coroutine オブジェクトは async iterable でないため、呼び出し時点で TypeError になる。"""
    policy = FailsafePolicy(handlers=handlers)
    coro = _coroutine_function()
    try:
        with pytest.raises(TypeError):
            failsafe_stream(policy, coro)  # type: ignore[arg-type]
    finally:
        coro.close()


async def test_failsafe_stream_aiterは呼び出し時点の1回だけ呼ばれる() -> None:
    """`__aiter__` は `failsafe_stream(...)` の呼び出しで 1 回呼ばれ、完走しても増えない。"""
    source = _CountingAiterSource(["a", "b"])
    policy = FailsafePolicy(handlers={MyError: "landed"})

    stream = failsafe_stream(policy, source)
    assert source.aiter_calls == 1

    received = await _collect(stream)

    assert received == ["a", "b"]
    assert source.aiter_calls == 1
