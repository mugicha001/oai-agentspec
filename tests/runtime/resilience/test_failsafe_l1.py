"""L1: Failsafe (`FailsafePolicy` / `FailsafeResult` / `failsafe_call`) の純検証。

FR-1〜6 の受け入れ基準を pin する:

- FR-1: frozen dataclass の既定値・フィールド保持・frozen 性・`__post_init__` の
  build-time 検証（handlers キーの型 / 禁止例外の fail-fast・handlers 値位置への
  `RUNNING_AGENT` 誤配置の拒否・`on_apply` の callable 検査）と、構築後の handlers 不変化。
- FR-2: 正常完了の透過・宣言例外型の着地・isinstance / 宣言順 first-match・
  未宣言例外の素通し・thunk の受理契約違反（coroutine 直渡し / 非 awaitable 戻り値）。
- FR-3: fallback の形（値 / sync callable / async callable）と fallback 自身の例外の素通し。
- FR-5: 監査（warning ログの内容・`log_on_apply` の抑止・`on_apply` の呼び出しと
  その例外の握り潰し）。
- FR-6: lib は例外型を同梱せず、利用者が持ち込む例外型で着地できること。

併せて `last_agent`（着地時に「もともと実行中だったエージェント」を参照する経路）を
pin する。決定は 2 段:

1. 例外ごとの指定: `FailsafeHandler.last_agent`（具体 agent または `RUNNING_AGENT`）
2. 全体規定: `FailsafePolicy.fallback_last_agent`（具体 agent または `RUNNING_AGENT`）

どちらも無指定なら None。例外が運ぶ実行文脈からの解決は `RUNNING_AGENT` を置いた段でのみ
走る opt-in であり、指定が無ければ解決は行われない（`last_agent` は None）。
`RUNNING_AGENT` を置いても解決できなければ次の段へ落ち、最後まで解決できなければ None。
各段の採否は `is None` / `is RUNNING_AGENT` の同一性判定のため、falsy な具体値も正当な
値として採る。加えて、解決は防御的に読み（属性の読み出し自体が失敗しても着地は
成立させる）、`last_agent` は `repr` に出さず、`FailsafeResult.from_exception` は入力
（例外インスタンス / 例外クラス）を build-time 検証する。

さらに以下を pin する:

- 解決の段ごと防御: 一方の読み取り先の読み出しが失敗しても、もう一方の読み取り先を
  飛ばさない（`run_data` -> `exc.last_agent` の連鎖が中断しない）。
- `RUNNING_AGENT` の singleton 強制: 内部 sentinel 型を直接構築してもモジュール変数と
  同一インスタンスになり、`is RUNNING_AGENT` 判定のガードを迂回できない。
- `ExceptionGroup` の禁止: 集約例外を丸ごと飲む宣言を build-time で拒否する
  （ユーザー定義サブクラスは従来どおり許容する）。
- `thunk` 受理契約の検査位置: handlers が空でも lib のメッセージで fail-fast する。
- `FailsafeResult` の直接構築にも `from_exception` と同一の入力検証が効く。
- 例外側が運ぶ値が `RUNNING_AGENT` そのものだった場合は解決不能として扱い、次の
  読み取り先 / 次の段へ落ちる（指定用 sentinel が着地結果に載らない）。
- 監査の独立性: `log_on_apply=False` でも `on_apply` は従来どおり発火する。
- async な `on_apply` の例外も握り潰され、async な fallback callable の例外は素通しする。
- `thunk` の呼び出しは handlers 非空の正常系・着地経路でも 1 回だけで、`thunk` 本体が
  同期的に送出した例外は着地対象にならない。

外部依存 (agents / openai) なし（SDK 例外はテスト内定義の fake で模す）。SDK 実型に対する
構造契約（`AgentsException.run_data` / `RunErrorDetails.last_agent`）の pin は
`tests/_adapters/test_resilience_exception_contract_l2.py` が担う。
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import inspect
import logging
import pickle
import warnings
from collections.abc import Mapping
from typing import Any

import pytest

from oai_agentspec.runtime.resilience._failsafe import (
    RUNNING_AGENT,
    FailsafeHandler,
    FailsafePolicy,
    FailsafeResult,
    _RunningAgentSentinel,
    failsafe_call,
)

pytestmark = pytest.mark.unit


class MyError(Exception):
    """テスト用のユーザー定義例外（`Exception` サブクラス・許容されるキー）。"""


class MyBaseError(BaseException):
    """`Exception` を継承しない `BaseException` 系（禁止列挙以外・拒否されるキー）。"""


def _fallback(exc: Exception) -> str:
    """ハンドラのダミー（宣言時は呼ばれない）。"""
    return "fallback"


# ---------------------------------------------------------------------------
# FailsafePolicy: 既定値・フィールド保持
# ---------------------------------------------------------------------------


def test_failsafe_policy_最小構成の既定値() -> None:
    """`FailsafePolicy()` が生成でき、既定値は handlers 空・log_on_apply True・on_apply None。"""
    policy = FailsafePolicy()
    assert policy.handlers == {}
    assert policy.log_on_apply is True
    assert policy.on_apply is None


def test_failsafe_policy_フル指定でフィールド保持() -> None:
    """全フィールドを明示指定した値がそのまま保持される。"""

    def _on_apply(result: object) -> None:
        return None

    policy = FailsafePolicy(
        handlers={MyError: _fallback},
        log_on_apply=False,
        on_apply=_on_apply,
    )
    assert policy.handlers == {MyError: _fallback}
    assert policy.log_on_apply is False
    assert policy.on_apply is _on_apply


def test_failsafe_policy_handlers_空dictは許容() -> None:
    """handlers が空 dict（no-op）は矛盾ではなく OK。"""
    policy = FailsafePolicy(handlers={})
    assert policy.handlers == {}


def test_failsafe_policy_handlers_ユーザー定義例外キーは許容() -> None:
    """ユーザー定義の `Exception` サブクラスはキーとして許容される。"""
    policy = FailsafePolicy(handlers={MyError: _fallback})
    assert policy.handlers[MyError] is _fallback


def test_failsafe_policy_handlers_組み込み例外サブクラスキーは許容() -> None:
    """`ValueError` / `TypeError` など `Exception` サブクラスはキーとして許容される。"""
    policy = FailsafePolicy(handlers={ValueError: _fallback})
    assert policy.handlers[ValueError] is _fallback


def test_failsafe_policy_handlers_複数キーの正常宣言() -> None:
    """複数の許容キーを同時に宣言できる。"""
    policy = FailsafePolicy(
        handlers={ValueError: _fallback, TypeError: _fallback, MyError: _fallback}
    )
    assert set(policy.handlers) == {ValueError, TypeError, MyError}


def test_failsafe_policy_is_frozen() -> None:
    """frozen dataclass のため属性の書き換えは FrozenInstanceError。"""
    policy = FailsafePolicy(handlers={MyError: _fallback})
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.log_on_apply = False  # type: ignore[misc]


def test_failsafe_policy_handlers_は構築後に不変化される() -> None:
    """構築後の handlers は書き込み不可で、検証を迂回した禁止キーの事後注入ができない。"""
    policy = FailsafePolicy(handlers={MyError: _fallback})

    with pytest.raises(TypeError):
        policy.handlers[Exception] = _fallback  # type: ignore[index]

    assert set(policy.handlers) == {MyError}


def test_failsafe_policy_handlers_は呼び出し側の元dict変更の影響を受けない() -> None:
    """handlers はコピーしてから不変化するため、元 dict 経由の事後注入も効かない。"""
    source: dict[type[Exception], object] = {MyError: _fallback}
    policy = FailsafePolicy(handlers=source)

    source[Exception] = _fallback

    assert set(policy.handlers) == {MyError}


def test_failsafe_policy_on_applyが非callableなら_ValueError() -> None:
    """`on_apply` が None でも callable でもなければ build-time で ValueError。"""
    with pytest.raises(ValueError, match="on_apply"):
        FailsafePolicy(handlers={MyError: _fallback}, on_apply="not-callable")


def test_failsafe_policy_on_applyの_ValueErrorは受け取った型名を含む() -> None:
    """`on_apply` 検査の ValueError メッセージには受け取った値の型名が含まれる。"""
    with pytest.raises(ValueError, match="str"):
        FailsafePolicy(handlers={MyError: _fallback}, on_apply="not-callable")


def test_failsafe_policy_on_applyはNoneとcallableを許容する() -> None:
    """`on_apply` は None（既定）と callable のいずれも許容される。"""

    def _on_apply(result: object) -> None:
        return None

    assert FailsafePolicy(handlers={MyError: _fallback}).on_apply is None
    assert FailsafePolicy(handlers={MyError: _fallback}, on_apply=_on_apply).on_apply is _on_apply


# ---------------------------------------------------------------------------
# FailsafePolicy: build-time ValueError（キーが例外クラスでない）
# ---------------------------------------------------------------------------


def test_failsafe_policy_handlers_非クラスキーは_ValueError() -> None:
    """キーが文字列など非クラス値なら build-time で ValueError（キー名を含む）。"""
    with pytest.raises(ValueError, match="str"):
        FailsafePolicy(handlers={"str": _fallback})  # type: ignore[dict-item]


def test_failsafe_policy_handlers_非例外クラスキーは_ValueError() -> None:
    """キーが `int`（例外でないクラス）なら ValueError（キー名を含む）。"""
    with pytest.raises(ValueError, match="int"):
        FailsafePolicy(handlers={int: _fallback})  # type: ignore[dict-item]


def test_failsafe_policy_handlers_objectキーは_ValueError() -> None:
    """キーが `object`（例外でないクラス）なら ValueError（キー名を含む）。"""
    with pytest.raises(ValueError, match="object"):
        FailsafePolicy(handlers={object: _fallback})  # type: ignore[dict-item]


def test_failsafe_policy_handlers_例外インスタンスキーは_ValueError() -> None:
    """キーが例外クラスでなくインスタンスなら ValueError（キー名を含む）。"""
    with pytest.raises(ValueError, match="ValueError"):
        FailsafePolicy(handlers={ValueError("x"): _fallback})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# FailsafePolicy: build-time ValueError（禁止例外の列挙）
# ---------------------------------------------------------------------------


def test_failsafe_policy_handlers_Exceptionキーは_ValueError() -> None:
    """`Exception` そのものは広すぎるため禁止（ValueError・キー名を含む）。"""
    with pytest.raises(ValueError, match="Exception"):
        FailsafePolicy(handlers={Exception: _fallback})


def test_failsafe_policy_handlers_BaseExceptionキーは_ValueError() -> None:
    """`BaseException` そのものは禁止（ValueError・キー名を含む）。"""
    with pytest.raises(ValueError, match="BaseException"):
        FailsafePolicy(handlers={BaseException: _fallback})  # type: ignore[dict-item]


def test_failsafe_policy_handlers_KeyboardInterruptキーは_ValueError() -> None:
    """`KeyboardInterrupt` の握り潰しは禁止（ValueError・キー名を含む）。"""
    with pytest.raises(ValueError, match="KeyboardInterrupt"):
        FailsafePolicy(handlers={KeyboardInterrupt: _fallback})  # type: ignore[dict-item]


def test_failsafe_policy_handlers_SystemExitキーは_ValueError() -> None:
    """`SystemExit` の握り潰しは禁止（ValueError・キー名を含む）。"""
    with pytest.raises(ValueError, match="SystemExit"):
        FailsafePolicy(handlers={SystemExit: _fallback})  # type: ignore[dict-item]


def test_failsafe_policy_handlers_CancelledErrorキーは_ValueError() -> None:
    """`asyncio.CancelledError`（協調キャンセル）の握り潰しは禁止（ValueError）。"""
    with pytest.raises(ValueError, match="CancelledError"):
        FailsafePolicy(handlers={asyncio.CancelledError: _fallback})  # type: ignore[dict-item]


def test_failsafe_policy_handlers_GeneratorExitキーは_ValueError() -> None:
    """`GeneratorExit` の握り潰しは禁止（ValueError・キー名を含む）。"""
    with pytest.raises(ValueError, match="GeneratorExit"):
        FailsafePolicy(handlers={GeneratorExit: _fallback})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# FailsafePolicy: build-time ValueError（Exception 非派生の BaseException 系）
# ---------------------------------------------------------------------------


def test_failsafe_policy_handlers_Exception非派生のBaseException系キーは_ValueError() -> None:
    """禁止列挙以外でも `Exception` を継承しない `BaseException` 系は ValueError。"""
    with pytest.raises(ValueError, match="MyBaseError"):
        FailsafePolicy(handlers={MyBaseError: _fallback})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# FailsafeResult: フィールド保持・structural 互換・frozen 性
# ---------------------------------------------------------------------------


def test_failsafe_result_4フィールドを保持する() -> None:
    """final_output / exception / matched_type / last_agent を保持して構築でき、値が読める。"""
    exc = MyError("boom")
    agent = object()
    result = FailsafeResult(
        final_output="fallback", exception=exc, matched_type=MyError, last_agent=agent
    )
    assert result.final_output == "fallback"
    assert result.exception is exc
    assert result.matched_type is MyError
    assert result.last_agent is agent
    assert [f.name for f in dataclasses.fields(FailsafeResult)] == [
        "final_output",
        "exception",
        "matched_type",
        "last_agent",
    ]


def test_failsafe_result_final_outputでアクセスできる() -> None:
    """SDK `RunResult` と structural 互換（共通基底は持たず `.final_output` のみ一致）。"""
    result = FailsafeResult(
        final_output={"answer": 42}, exception=MyError("x"), matched_type=MyError
    )
    assert result.final_output == {"answer": 42}
    assert hasattr(result, "final_output")


def test_failsafe_result_isinstanceで判別できる() -> None:
    """呼び出し側が failsafe 適用を `isinstance` で判別できる。"""
    result = FailsafeResult(final_output=None, exception=MyError("x"), matched_type=MyError)
    assert isinstance(result, FailsafeResult)


def test_failsafe_result_is_frozen() -> None:
    """frozen dataclass のため属性の書き換えは FrozenInstanceError。"""
    result = FailsafeResult(final_output="a", exception=MyError("x"), matched_type=MyError)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.final_output = "b"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# failsafe_call 用のテストヘルパ
# ---------------------------------------------------------------------------

_LOGGER_NAME = "oai_agentspec.resilience"

SENTINEL = object()
"""正常完了時の戻り値が「そのまま」返ることを同一性で確認するための番兵。"""


class MySubError(MyError):
    """`MyError` のサブクラス（isinstance マッチの検証用）。"""


class SdkLikeError(Exception):
    """SDK 例外の代役（lib は例外型を知らず利用者が持ち込むことの表現）。"""


def _thunk_returning(value: Any):  # noqa: ANN202
    """`value` をそのまま返す thunk を作る。"""

    async def _thunk() -> Any:
        return value

    return _thunk


def _thunk_raising(exc: BaseException):  # noqa: ANN202
    """`exc` を送出する thunk を作る。"""

    async def _thunk() -> Any:
        raise exc

    return _thunk


def _records_of(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    """resilience logger が出した指定レベルのレコードのみを抽出する。"""
    return [r for r in caplog.records if r.name == _LOGGER_NAME and r.levelno == level]


# ---------------------------------------------------------------------------
# failsafe_call: FR-2 基本挙動（正常完了 / 着地 / マッチ規則 / 透過）
# ---------------------------------------------------------------------------


async def test_failsafe_call_正常完了は戻り値をそのまま返す() -> None:
    """例外が出なければ thunk の戻り値がそのまま返り、`FailsafeResult` でラップされない。"""
    policy = FailsafePolicy(handlers={MyError: "fallback"})

    result = await failsafe_call(policy, _thunk_returning(SENTINEL))

    assert result is SENTINEL
    assert not isinstance(result, FailsafeResult)


async def test_failsafe_call_宣言例外型の送出はFailsafeResultを返す() -> None:
    """宣言済みの例外型が送出されたら着地し、4 フィールド（`last_agent` 既定 None）が返る。"""
    exc = MyError("boom")
    policy = FailsafePolicy(handlers={MyError: "landed"})

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.final_output == "landed"
    assert result.exception is exc
    assert result.matched_type is MyError
    assert result.last_agent is None


async def test_failsafe_call_サブクラス送出は親キーにisinstanceマッチする() -> None:
    """送出型が宣言キーのサブクラスでも着地し、matched_type は宣言キー側になる。"""
    exc = MySubError("boom")
    policy = FailsafePolicy(handlers={MyError: "landed"})

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.exception is exc
    assert result.matched_type is MyError
    assert result.matched_type is not MySubError


async def test_failsafe_call_複数マッチは挿入順first_matchで親が先なら親が勝つ() -> None:
    """親キーを先に宣言した場合、サブクラス例外でも親キーが first-match する。"""
    policy = FailsafePolicy(handlers={MyError: "parent", MySubError: "child"})

    result = await failsafe_call(policy, _thunk_raising(MySubError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.matched_type is MyError
    assert result.final_output == "parent"


async def test_failsafe_call_複数マッチは挿入順first_matchで子が先なら子が勝つ() -> None:
    """宣言順を入れ替えると first-match の結果も入れ替わる（順序が意味を持つ）。"""
    policy = FailsafePolicy(handlers={MySubError: "child", MyError: "parent"})

    result = await failsafe_call(policy, _thunk_raising(MySubError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.matched_type is MySubError
    assert result.final_output == "child"


async def test_failsafe_call_未宣言例外はそのまま再送出される() -> None:
    """どのキーにもマッチしない例外は握り潰さず素通しする（chaining もしない）。"""
    exc = ValueError("unhandled")
    policy = FailsafePolicy(handlers={MyError: "landed"})

    with pytest.raises(ValueError) as excinfo:
        await failsafe_call(policy, _thunk_raising(exc))

    assert excinfo.value is exc
    assert excinfo.value.__cause__ is None


async def test_failsafe_call_handlers空は正常時に戻り値をそのまま返す() -> None:
    """handlers 空（捕捉ゼロ）でも正常完了は完全透過する。"""
    policy = FailsafePolicy(handlers={})

    result = await failsafe_call(policy, _thunk_returning(SENTINEL))

    assert result is SENTINEL


async def test_failsafe_call_handlers空は例外をそのまま伝播する() -> None:
    """handlers 空なら一切捕捉せず、例外はそのまま呼び出し側へ伝播する。"""
    exc = MyError("boom")
    policy = FailsafePolicy(handlers={})

    with pytest.raises(MyError) as excinfo:
        await failsafe_call(policy, _thunk_raising(exc))

    assert excinfo.value is exc


async def test_failsafe_call_coroutine直渡しはTypeErrorでfail_fastする() -> None:
    """thunk ではなく coroutine を直接渡す受理契約違反は TypeError で fail-fast する。"""
    policy = FailsafePolicy(handlers={MyError: "landed"})

    async def _coro() -> str:
        return "value"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        coro = _coro()
        try:
            with pytest.raises(TypeError):
                await failsafe_call(policy, coro)  # type: ignore[arg-type]
        finally:
            coro.close()


async def test_failsafe_call_coroutine直渡しのTypeErrorはTypeError宣言でも着地しない() -> None:
    """`thunk()` は try の外で呼ぶため、TypeError を宣言していても受理契約違反は素通しする。"""
    policy = FailsafePolicy(handlers={TypeError: "landed"})

    async def _coro() -> str:
        return "value"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        coro = _coro()
        try:
            with pytest.raises(TypeError):
                await failsafe_call(policy, coro)  # type: ignore[arg-type]
        finally:
            coro.close()


async def test_failsafe_call_非awaitableを返すthunkはTypeErrorでfail_fastする() -> None:
    """awaitable を返さない callable は受理契約違反として TypeError で fail-fast する。"""
    policy = FailsafePolicy(handlers={MyError: "landed"})

    with pytest.raises(TypeError, match="awaitable"):
        await failsafe_call(policy, lambda: 42)  # type: ignore[arg-type,return-value]


async def test_failsafe_call_非awaitable戻り値のTypeErrorはTypeError宣言でも着地しない() -> None:
    """受理契約の検査は try の外で行うため、TypeError を宣言していても素通しする。"""
    policy = FailsafePolicy(handlers={TypeError: "landed"})

    with pytest.raises(TypeError):
        await failsafe_call(policy, lambda: 42)  # type: ignore[arg-type,return-value]


# ---------------------------------------------------------------------------
# failsafe_call: FR-3 fallback の形（値 / sync callable / async callable / 例外）
# ---------------------------------------------------------------------------


async def test_failsafe_call_非callable値のfallbackはそのまま着地値になる() -> None:
    """handlers の値が callable でなければ、その値自体が final_output になる。"""
    policy = FailsafePolicy(handlers={MyError: {"answer": 42}})

    result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.final_output == {"answer": 42}


async def test_failsafe_call_sync_callableのfallbackは例外を受け取り戻り値が着地値になる() -> None:
    """sync callable の fallback は捕捉例外を単一引数に呼ばれ、戻り値が final_output になる。"""
    exc = MyError("boom")
    seen: list[Exception] = []

    def _fb(received: Exception) -> str:
        seen.append(received)
        return f"recovered:{received}"

    policy = FailsafePolicy(handlers={MyError: _fb})

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.final_output == "recovered:boom"
    assert seen == [exc]
    assert seen[0] is exc


async def test_failsafe_call_async_callableのfallbackはawaitされた結果が着地値になる() -> None:
    """async callable の fallback は await され、その結果が final_output になる。"""
    exc = MyError("boom")
    seen: list[Exception] = []

    async def _fb(received: Exception) -> str:
        seen.append(received)
        return "async-recovered"

    policy = FailsafePolicy(handlers={MyError: _fb})

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.final_output == "async-recovered"
    assert seen[0] is exc


async def test_failsafe_call_fallback自身の例外はそのまま伝播する() -> None:
    """fallback が失敗したら着地は成立せず、その例外がそのまま伝播する。"""
    fallback_exc = RuntimeError("fallback failed")

    def _fb(received: Exception) -> str:
        raise fallback_exc

    policy = FailsafePolicy(handlers={MyError: _fb})

    with pytest.raises(RuntimeError) as excinfo:
        await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert excinfo.value is fallback_exc


# ---------------------------------------------------------------------------
# failsafe_call: FR-5 監査（warning ログ / on_apply コールバック）
# ---------------------------------------------------------------------------


async def test_failsafe_call_既定でwarningログが出る(caplog: pytest.LogCaptureFixture) -> None:
    """log_on_apply 既定 True では resilience logger に WARNING がトレースバック付きで出る。"""
    policy = FailsafePolicy(handlers={MyError: "landed"})

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    records = _records_of(caplog, logging.WARNING)
    assert len(records) == 1
    assert records[0].exc_info is not None
    msg = records[0].getMessage()
    assert MyError.__name__ in msg
    assert "boom" in msg


async def test_failsafe_call_warningログはマッチキー名と例外型名を含む(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """warning は matched_type（宣言キー名）と実際の例外型名の双方を出す（FR-5）。"""
    policy = FailsafePolicy(handlers={MyError: "landed"})

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        await failsafe_call(policy, _thunk_raising(MySubError("boom")))

    msg = _records_of(caplog, logging.WARNING)[0].getMessage()
    assert f"matched_type={MyError.__name__}" in msg
    assert f"exception_type={MySubError.__name__}" in msg


async def test_failsafe_call_log_on_apply_Falseでwarningが出ない(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """log_on_apply=False なら着地しても warning は emit されない。"""
    policy = FailsafePolicy(handlers={MyError: "landed"}, log_on_apply=False)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert _records_of(caplog, logging.WARNING) == []


async def test_failsafe_call_on_apply_syncはFailsafeResultを引数に呼ばれる() -> None:
    """sync の on_apply は着地時に FailsafeResult を受け取って呼ばれる。"""
    exc = MyError("boom")
    seen: list[FailsafeResult] = []

    def _on_apply(result: FailsafeResult) -> None:
        seen.append(result)

    policy = FailsafePolicy(handlers={MyError: "landed"}, on_apply=_on_apply)

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert len(seen) == 1
    assert seen[0] is result
    assert seen[0].final_output == "landed"
    assert seen[0].exception is exc
    assert seen[0].matched_type is MyError


async def test_failsafe_call_on_apply_asyncはawaitされる() -> None:
    """async の on_apply は await され、コールバック本体が最後まで実行される。"""
    called: list[str] = []

    async def _on_apply(result: FailsafeResult) -> None:
        await asyncio.sleep(0)
        called.append("done")

    policy = FailsafePolicy(handlers={MyError: "landed"}, on_apply=_on_apply)

    result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert called == ["done"]


async def test_failsafe_call_on_apply例外はerrorログで握り潰され着地は継続する(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """on_apply が失敗しても FailsafeResult の返却は継続し、error ログに記録される。"""

    def _on_apply(result: FailsafeResult) -> None:
        raise RuntimeError("callback failed")

    policy = FailsafePolicy(handlers={MyError: "landed"}, on_apply=_on_apply)

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.final_output == "landed"
    errors = _records_of(caplog, logging.ERROR)
    assert len(errors) == 1
    assert errors[0].exc_info is not None


async def test_failsafe_call_on_apply未指定なら呼ばれない(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """on_apply 既定 None ではコールバック経路に入らず（error ログ皆無）、着地のみが行われる。"""
    policy = FailsafePolicy(handlers={MyError: "landed"})

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.final_output == "landed"
    assert result.matched_type is MyError
    assert _records_of(caplog, logging.ERROR) == []


async def test_failsafe_call_fallback例外時はwarningもon_applyも発火しない(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """fallback 自身が失敗した場合は着地不成立のため、監査（warning / on_apply）に到達しない。"""
    called: list[str] = []

    def _fb(received: Exception) -> str:
        raise RuntimeError("fallback failed")

    def _on_apply(result: FailsafeResult) -> None:
        called.append("on_apply")

    policy = FailsafePolicy(handlers={MyError: _fb}, on_apply=_on_apply)

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        with pytest.raises(RuntimeError):
            await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert called == []
    assert [r for r in caplog.records if r.name == _LOGGER_NAME] == []


# ---------------------------------------------------------------------------
# failsafe_call: FR-6 SDK 共存（lib は例外型を知らず利用者が持ち込む）
# ---------------------------------------------------------------------------


async def test_failsafe_call_利用者が持ち込む例外型で着地できる() -> None:
    """lib は例外型を同梱せず、利用者が宣言した任意の例外型（SDK 例外の代役）で着地する。"""
    exc = SdkLikeError("guardrail tripped")
    policy = FailsafePolicy(handlers={SdkLikeError: "safe answer"})

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.final_output == "safe answer"
    assert result.matched_type is SdkLikeError
    assert result.exception is exc


# ---------------------------------------------------------------------------
# last_agent 用のテストヘルパ（SDK 例外は fake で模す・`agents` を import しない）
# ---------------------------------------------------------------------------

AGENT_FROM_RUN_DATA = object()
"""`exc.run_data.last_agent` 由来で解決された値を同一性で確認する番兵（Agent の代役）。"""

AGENT_FROM_ATTRIBUTE = object()
"""`exc.last_agent` 由来で解決された値を同一性で確認する番兵（Agent の代役）。"""

AGENT_PER_EXCEPTION = object()
"""段 1（`FailsafeHandler.last_agent`）に置いた具体 agent を同一性で確認する番兵。"""

AGENT_POLICY_FALLBACK = object()
"""段 2（`FailsafePolicy.fallback_last_agent`）に置いた具体 agent を同一性で確認する番兵。"""


class _FakeRunData:
    """SDK `RunErrorDetails` の代役（解決に必要な `last_agent` のみを持つ最小形）。"""

    def __init__(self, last_agent: Any) -> None:
        self.last_agent = last_agent


class _FakeRunDataWithoutLastAgent:
    """`run_data` ではあるが `last_agent` 属性を持たないオブジェクト（防御読みの検証用）。"""


class SdkRunDataError(Exception):
    """SDK 例外の代役（`run_data` を保持する。lib は `getattr` で読むだけ）。"""

    def __init__(self, message: str, run_data: Any) -> None:
        super().__init__(message)
        self.run_data = run_data


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


# ---------------------------------------------------------------------------
# FailsafeHandler: 既定値・フィールド保持・frozen 性・ネスト宣言の build-time 拒否
# ---------------------------------------------------------------------------


def test_failsafe_handler_最小構成の既定値() -> None:
    """`fallback` のみの指定で生成でき、`last_agent` の既定は None（段 1 の指定なし）。"""
    handler = FailsafeHandler(fallback="landed")
    assert handler.fallback == "landed"
    assert handler.last_agent is None


def test_failsafe_handler_フル指定でフィールド保持() -> None:
    """`fallback` / `last_agent` を明示指定した値がそのまま保持される。"""
    handler = FailsafeHandler(fallback=_fallback, last_agent=AGENT_PER_EXCEPTION)
    assert handler.fallback is _fallback
    assert handler.last_agent is AGENT_PER_EXCEPTION


def test_failsafe_handler_is_frozen() -> None:
    """frozen dataclass のため属性の書き換えは FrozenInstanceError。"""
    handler = FailsafeHandler(fallback="landed")
    with pytest.raises(dataclasses.FrozenInstanceError):
        handler.last_agent = AGENT_PER_EXCEPTION  # type: ignore[misc]


def test_failsafe_handler_fallbackへのネスト宣言は_ValueError() -> None:
    """`fallback` に `FailsafeHandler` を入れ子にする誤宣言は build-time で ValueError。"""
    with pytest.raises(ValueError, match="fallback"):
        FailsafeHandler(fallback=FailsafeHandler(fallback="landed"))


def test_failsafe_handler_ネスト宣言の_ValueErrorは型名を手掛かりに含む() -> None:
    """ネスト拒否のメッセージには `FailsafeHandler` の名が含まれ、原因が特定できる。"""
    with pytest.raises(ValueError) as excinfo:
        FailsafeHandler(fallback=FailsafeHandler(fallback="landed"), last_agent=AGENT_PER_EXCEPTION)

    assert FailsafeHandler.__name__ in str(excinfo.value)


def test_failsafe_handler_はhandlersの値として宣言できる() -> None:
    """handlers の値位置に置いても policy の build-time 検証は通り、値がそのまま保持される。"""
    handler = FailsafeHandler(fallback="landed", last_agent=AGENT_PER_EXCEPTION)
    policy = FailsafePolicy(handlers={MyError: handler})
    assert policy.handlers[MyError] is handler


# ---------------------------------------------------------------------------
# FailsafeHandler 経由でも FR-3 の fallback 挙動が同一であること
# ---------------------------------------------------------------------------


async def test_failsafe_call_handler経由の非callable値はそのまま着地値になる() -> None:
    """`FailsafeHandler.fallback` が callable でなければ、その値自体が final_output になる。"""
    policy = FailsafePolicy(handlers={MyError: FailsafeHandler(fallback={"answer": 42})})

    result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.final_output == {"answer": 42}


async def test_failsafe_call_handler経由のsync_callableは例外を受け取る() -> None:
    """`FailsafeHandler.fallback` の sync callable は捕捉例外を単一引数に呼ばれる。"""
    exc = MyError("boom")
    seen: list[Exception] = []

    def _fb(received: Exception) -> str:
        seen.append(received)
        return f"recovered:{received}"

    policy = FailsafePolicy(handlers={MyError: FailsafeHandler(fallback=_fb)})

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.final_output == "recovered:boom"
    assert seen == [exc]
    assert seen[0] is exc


async def test_failsafe_call_handler経由のasync_callableはawaitされる() -> None:
    """`FailsafeHandler.fallback` の async callable は await され、その結果が着地値になる。"""
    exc = MyError("boom")
    seen: list[Exception] = []

    async def _fb(received: Exception) -> str:
        seen.append(received)
        return "async-recovered"

    policy = FailsafePolicy(handlers={MyError: FailsafeHandler(fallback=_fb)})

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.final_output == "async-recovered"
    assert seen[0] is exc


async def test_failsafe_call_handler経由のfallback例外は素通しし監査も発火しない(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`FailsafeHandler` 経由でも fallback 自身の失敗は着地不成立（warning / on_apply 非発火）。"""
    fallback_exc = RuntimeError("fallback failed")
    called: list[str] = []

    def _fb(received: Exception) -> str:
        raise fallback_exc

    def _on_apply(result: FailsafeResult) -> None:
        called.append("on_apply")

    policy = FailsafePolicy(
        handlers={MyError: FailsafeHandler(fallback=_fb, last_agent=AGENT_PER_EXCEPTION)},
        on_apply=_on_apply,
    )

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        with pytest.raises(RuntimeError) as excinfo:
            await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert excinfo.value is fallback_exc
    assert called == []
    assert [r for r in caplog.records if r.name == _LOGGER_NAME] == []


# ---------------------------------------------------------------------------
# RUNNING_AGENT: 解決を opt-in するための公開 sentinel
# ---------------------------------------------------------------------------


def test_running_agent_のreprはRUNNING_AGENT() -> None:
    """デバッグ時に素性が読めるよう、sentinel の repr は名前そのもの。"""
    assert repr(RUNNING_AGENT) == "RUNNING_AGENT"


def test_running_agent_は単一の同一オブジェクトである() -> None:
    """判定は `is RUNNING_AGENT` の同一性で行うため、None とも他の値とも別物である。"""
    assert RUNNING_AGENT is RUNNING_AGENT
    assert RUNNING_AGENT is not None
    assert RUNNING_AGENT is not object()


def test_running_agent_はFailsafeHandlerとFailsafePolicyに指定できる() -> None:
    """段 1 / 段 2 のどちらにも置け、指定値としてそのまま保持される。"""
    handler = FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)
    policy = FailsafePolicy(handlers={MyError: handler}, fallback_last_agent=RUNNING_AGENT)

    assert handler.last_agent is RUNNING_AGENT
    assert policy.fallback_last_agent is RUNNING_AGENT


# ---------------------------------------------------------------------------
# failsafe_call: 指定が無ければ解決しない（RUNNING_AGENT による opt-in）
# ---------------------------------------------------------------------------


async def test_failsafe_call_指定が無ければrun_dataを持つ例外でもlast_agentはNone() -> None:
    """解決は opt-in のため、`run_data.last_agent` を持つ例外でも指定が無ければ None。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    policy = FailsafePolicy(handlers={SdkRunDataError: "landed"})

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None
    assert result.last_agent is not AGENT_FROM_RUN_DATA


async def test_failsafe_call_指定が無ければlast_agent属性を持つ例外でもNone() -> None:
    """budget 相当（`exc.last_agent`）の例外でも、指定が無ければ解決は走らない。"""
    exc = BudgetLikeError("boom", last_agent=AGENT_FROM_ATTRIBUTE)
    policy = FailsafePolicy(handlers={BudgetLikeError: "landed"})

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None


async def test_failsafe_call_段1未指定のFailsafeHandlerでも解決は走らない() -> None:
    """`FailsafeHandler.last_agent` 既定 None は「段 1 の指定なし」であり、解決の合図ではない。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    policy = FailsafePolicy(handlers={SdkRunDataError: FailsafeHandler(fallback="landed")})

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None


async def test_failsafe_call_解決材料の無い例外でも指定が無ければNone() -> None:
    """`run_data` も `last_agent` も持たない例外は当然 None（例外にしない）。"""
    policy = FailsafePolicy(handlers={MyError: "landed"})

    result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None


# ---------------------------------------------------------------------------
# failsafe_call: 段 1 に RUNNING_AGENT を置いたときの解決
# ---------------------------------------------------------------------------


async def test_failsafe_call_段1のRUNNING_AGENTはrun_dataから解決される() -> None:
    """SDK 例外の代役（`run_data.last_agent`）から last_agent が解決される。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    policy = FailsafePolicy(
        handlers={SdkRunDataError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)}
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_FROM_RUN_DATA
    assert result.final_output == "landed"


async def test_failsafe_call_段1のRUNNING_AGENTは例外属性から解決される() -> None:
    """`run_data` を持たず `last_agent` 属性のみを持つ例外（budget 相当）からも解決される。"""
    exc = BudgetLikeError("boom", last_agent=AGENT_FROM_ATTRIBUTE)
    policy = FailsafePolicy(
        handlers={BudgetLikeError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)}
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_FROM_ATTRIBUTE


async def test_failsafe_call_RUNNING_AGENTの解決はrun_dataが例外属性より優先される() -> None:
    """双方を持つ例外では `run_data.last_agent` が先に採用される（読み取り先の優先順位）。"""
    exc = HybridError(
        "boom",
        run_data=_FakeRunData(AGENT_FROM_RUN_DATA),
        last_agent=AGENT_FROM_ATTRIBUTE,
    )
    policy = FailsafePolicy(
        handlers={HybridError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)}
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_FROM_RUN_DATA


async def test_failsafe_call_run_dataにlast_agent属性が無ければ例外属性から解決される() -> None:
    """`run_data` の読み出しは防御読みのため、属性を欠く run_data でも次の読み取り先へ進む。"""
    exc = HybridError(
        "boom",
        run_data=_FakeRunDataWithoutLastAgent(),
        last_agent=AGENT_FROM_ATTRIBUTE,
    )
    policy = FailsafePolicy(
        handlers={HybridError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)}
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_FROM_ATTRIBUTE


async def test_failsafe_call_run_dataにlast_agent属性が無く例外属性も無ければNone() -> None:
    """防御読みで run_data からも例外自身からも取れない場合は None（AttributeError にしない）。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunDataWithoutLastAgent())
    policy = FailsafePolicy(
        handlers={SdkRunDataError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)}
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None


async def test_failsafe_call_解決値が空文字でもその値が採用される() -> None:
    """解決は `is None` の逐次チェックのため、falsy な値（空文字）も正当な値として採る。"""
    exc = BudgetLikeError("boom", last_agent="")
    policy = FailsafePolicy(
        handlers={BudgetLikeError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)}
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent == ""


async def test_failsafe_call_run_dataの解決値がfalsyでも例外属性へ落ちない() -> None:
    """`run_data.last_agent` が 0（falsy）でも例外属性側へは進まない（`or` 連鎖の退行検出）。"""
    exc = HybridError("boom", run_data=_FakeRunData(0), last_agent=AGENT_FROM_ATTRIBUTE)
    policy = FailsafePolicy(
        handlers={HybridError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)}
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent == 0
    assert result.last_agent is not AGENT_FROM_ATTRIBUTE


async def test_failsafe_call_解決値はRUNNING_AGENTそのものにはならない() -> None:
    """`RUNNING_AGENT` は指定側の合図で、結果には解決済みの値か None しか載らない。"""
    resolvable = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    unresolvable = MyError("boom")
    resolvable_policy = FailsafePolicy(
        handlers={SdkRunDataError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)}
    )
    unresolvable_policy = FailsafePolicy(
        handlers={MyError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)}
    )

    landed = await failsafe_call(resolvable_policy, _thunk_raising(resolvable))
    missed = await failsafe_call(unresolvable_policy, _thunk_raising(unresolvable))

    assert isinstance(landed, FailsafeResult)
    assert isinstance(missed, FailsafeResult)
    assert landed.last_agent is not RUNNING_AGENT
    assert missed.last_agent is not RUNNING_AGENT
    assert missed.last_agent is None


# ---------------------------------------------------------------------------
# failsafe_call: 段 1 に具体 agent を置いたとき（解決は走らない）
# ---------------------------------------------------------------------------


async def test_failsafe_call_段1の具体agentはrun_dataより優先される() -> None:
    """段 1 に具体 agent を置けばその値で確定し、例外からの解決は行われない。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    policy = FailsafePolicy(
        handlers={
            SdkRunDataError: FailsafeHandler(fallback="landed", last_agent=AGENT_PER_EXCEPTION)
        }
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_PER_EXCEPTION
    assert result.last_agent is not AGENT_FROM_RUN_DATA
    assert result.final_output == "landed"


async def test_failsafe_call_段1の具体agentは例外属性より優先される() -> None:
    """budget 相当の例外（`last_agent` 属性）でも段 1 の具体 agent が使われる。"""
    exc = BudgetLikeError("boom", last_agent=AGENT_FROM_ATTRIBUTE)
    policy = FailsafePolicy(
        handlers={
            BudgetLikeError: FailsafeHandler(fallback="landed", last_agent=AGENT_PER_EXCEPTION)
        }
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_PER_EXCEPTION


async def test_failsafe_call_段1の具体agentが空文字でも採用される() -> None:
    """段 1 の採否は `is None` / `is RUNNING_AGENT` 判定のため、falsy な具体値も採用される。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    policy = FailsafePolicy(
        handlers={SdkRunDataError: FailsafeHandler(fallback="landed", last_agent="")}
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent == ""


async def test_failsafe_call_on_applyが受け取る結果にもlast_agentが入る() -> None:
    """監査コールバックへ渡る `FailsafeResult` にも解決済みの last_agent が含まれる。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    seen: list[FailsafeResult] = []

    def _on_apply(result: FailsafeResult) -> None:
        seen.append(result)

    policy = FailsafePolicy(
        handlers={SdkRunDataError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)},
        on_apply=_on_apply,
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert len(seen) == 1
    assert seen[0] is result
    assert seen[0].last_agent is AGENT_FROM_RUN_DATA


# ---------------------------------------------------------------------------
# FailsafeResult: last_agent フィールド（既定 None・構築互換）
# ---------------------------------------------------------------------------


def test_failsafe_result_last_agent未指定の既定はNone() -> None:
    """既存の 3 引数構築は引き続き動き、`last_agent` は既定 None になる（構築互換）。"""
    result = FailsafeResult(final_output="a", exception=MyError("x"), matched_type=MyError)
    assert result.last_agent is None


def test_failsafe_result_last_agentは明示指定した値を保持する() -> None:
    """`last_agent` に渡した不透明値がそのまま読める。"""
    result = FailsafeResult(
        final_output="a",
        exception=MyError("x"),
        matched_type=MyError,
        last_agent=AGENT_PER_EXCEPTION,
    )
    assert result.last_agent is AGENT_PER_EXCEPTION


def test_failsafe_result_last_agentもfrozenで書き換え不可() -> None:
    """追加フィールドも frozen の対象で、着地後の差し替え（監査の改竄）はできない。"""
    result = FailsafeResult(
        final_output="a",
        exception=MyError("x"),
        matched_type=MyError,
        last_agent=AGENT_FROM_RUN_DATA,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.last_agent = AGENT_PER_EXCEPTION  # type: ignore[misc]

    assert result.last_agent is AGENT_FROM_RUN_DATA


# ---------------------------------------------------------------------------
# FailsafeResult.from_exception: 外側の except で同じ型へ着地させるファクトリ
# ---------------------------------------------------------------------------


def test_from_exception_はFailsafeResultを返す() -> None:
    """戻り値は `FailsafeResult` インスタンスで、外側の except でも isinstance 判別できる。"""
    exc = MyError("boom")

    result = FailsafeResult.from_exception(exc, final_output="landed")

    assert isinstance(result, FailsafeResult)
    assert result.final_output == "landed"
    assert result.exception is exc


def test_from_exception_matched_type未指定は送出型になる() -> None:
    """手動着地では宣言キーが無いため、既定の matched_type は `type(exception)`。"""
    exc = MySubError("boom")

    result = FailsafeResult.from_exception(exc, final_output="landed")

    assert result.matched_type is MySubError


def test_from_exception_matched_type明示指定はその値になる() -> None:
    """matched_type を明示すれば送出型でなくその値が入る（宣言キー相当を手で与える）。"""
    exc = MySubError("boom")

    result = FailsafeResult.from_exception(exc, final_output="landed", matched_type=MyError)

    assert result.matched_type is MyError


def test_from_exception_last_agent未指定なら解決は走らずNone() -> None:
    """手動着地でも解決は opt-in で、指定が無ければ `run_data` を持つ例外でも None。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))

    result = FailsafeResult.from_exception(exc, final_output="landed")

    assert result.last_agent is None
    assert result.last_agent is not AGENT_FROM_RUN_DATA


def test_from_exception_RUNNING_AGENT指定はrun_dataから解決される() -> None:
    """`RUNNING_AGENT` を指定すると `failsafe_call` と同じ解決（run_data 経由）が働く。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))

    result = FailsafeResult.from_exception(exc, final_output="landed", last_agent=RUNNING_AGENT)

    assert result.last_agent is AGENT_FROM_RUN_DATA


def test_from_exception_RUNNING_AGENT指定は例外属性からも解決される() -> None:
    """`RUNNING_AGENT` を指定すれば例外属性からも解決される（budget 相当）。"""
    exc = BudgetLikeError("boom", last_agent=AGENT_FROM_ATTRIBUTE)

    result = FailsafeResult.from_exception(exc, final_output="landed", last_agent=RUNNING_AGENT)

    assert result.last_agent is AGENT_FROM_ATTRIBUTE


def test_from_exception_RUNNING_AGENT指定でも解決材料が無ければNone() -> None:
    """解決材料が無い例外では None になる（`RUNNING_AGENT` はそのまま載らない）。"""
    result = FailsafeResult.from_exception(
        MyError("boom"), final_output="landed", last_agent=RUNNING_AGENT
    )

    assert result.last_agent is None
    assert result.last_agent is not RUNNING_AGENT


def test_from_exception_具体agentの指定はそのまま入る() -> None:
    """具体 agent を指定すると解決は走らず、その値がそのまま結果に載る。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))

    result = FailsafeResult.from_exception(
        exc, final_output="landed", last_agent=AGENT_PER_EXCEPTION
    )

    assert result.last_agent is AGENT_PER_EXCEPTION
    assert result.last_agent is not AGENT_FROM_RUN_DATA


def test_from_exception_具体agentが空文字でも採用される() -> None:
    """判定は `is None` / `is RUNNING_AGENT` のため、falsy な具体値も指定として採られる。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))

    result = FailsafeResult.from_exception(exc, final_output="landed", last_agent="")

    assert result.last_agent == ""


async def test_from_exception_の解決はfailsafe_call着地時と一致する() -> None:
    """同じ例外と同じ指定なら手動着地と自動着地で last_agent の意味が揃う（解決の共有）。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    policy = FailsafePolicy(
        handlers={SdkRunDataError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)}
    )

    landed = await failsafe_call(policy, _thunk_raising(exc))
    manual = FailsafeResult.from_exception(exc, final_output="landed", last_agent=RUNNING_AGENT)

    assert isinstance(landed, FailsafeResult)
    assert manual.last_agent is landed.last_agent
    assert manual.last_agent is AGENT_FROM_RUN_DATA
    assert manual.exception is landed.exception


def test_from_exception_final_outputはキーワード専用() -> None:
    """`final_output` 以降はキーワード専用で、位置引数での取り違えを防ぐ。"""
    with pytest.raises(TypeError):
        FailsafeResult.from_exception(MyError("boom"), "landed")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# R1: 解決の防御（属性の読み出しが失敗しても着地は成立する）用のヘルパ
# ---------------------------------------------------------------------------


class _FakeRunDataRaisingLastAgent:
    """`last_agent` の読み出し自体が例外を送出する run_data（SDK 側 property 失敗の代役）。"""

    @property
    def last_agent(self) -> Any:
        """常に `RuntimeError` を送出する。"""
        raise RuntimeError("run_data.last_agent access failed")


class RaisingRunDataError(Exception):
    """`run_data` の読み出し自体が例外を送出する例外（1 つ目の読み取り先の防御）。"""

    @property
    def run_data(self) -> Any:
        """常に `RuntimeError` を送出する。"""
        raise RuntimeError("run_data access failed")


class RaisingLastAgentError(Exception):
    """`run_data` を持たず `last_agent` の読み出しが例外を送出する例外（2 つ目の読み取り先）。"""

    @property
    def last_agent(self) -> Any:
        """常に `RuntimeError` を送出する。"""
        raise RuntimeError("last_agent access failed")


class RunDataRaisingLastAgentError(Exception):
    """`run_data` は取れるが、その `last_agent` の読み出しが例外を送出する例外。"""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.run_data = _FakeRunDataRaisingLastAgent()


async def _land_with_audit(
    exc: Exception, caplog: pytest.LogCaptureFixture
) -> tuple[Any, list[FailsafeResult], list[logging.LogRecord]]:
    """`RUNNING_AGENT` 指定で `exc` を着地させ、結果・`on_apply` の受領・warning を返す。

    解決は opt-in のため、防御の検証には段 1 へ `RUNNING_AGENT` を置いて解決経路を
    実際に走らせる必要がある。
    """
    seen: list[FailsafeResult] = []

    def _on_apply(result: FailsafeResult) -> None:
        seen.append(result)

    policy = FailsafePolicy(
        handlers={type(exc): FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)},
        on_apply=_on_apply,
    )
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = await failsafe_call(policy, _thunk_raising(exc))
    return result, seen, _records_of(caplog, logging.WARNING)


# ---------------------------------------------------------------------------
# R1: 解決の防御（failsafe_call）
# ---------------------------------------------------------------------------


async def test_failsafe_call_run_data読み出しが例外でも着地は成立する(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """解決（`run_data` の読み出し）が失敗しても例外は漏れず、last_agent None で着地する。"""
    result, _, _ = await _land_with_audit(RaisingRunDataError("boom"), caplog)

    assert isinstance(result, FailsafeResult)
    assert result.final_output == "landed"
    assert result.last_agent is None
    assert result.last_agent is not RUNNING_AGENT


async def test_failsafe_call_run_data読み出しが例外でも監査は発火する(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """解決の失敗は着地を壊さないため、warning と `on_apply` は通常どおり発火する。"""
    result, seen, records = await _land_with_audit(RaisingRunDataError("boom"), caplog)

    assert len(records) == 1
    assert len(seen) == 1
    assert seen[0] is result


async def test_failsafe_call_例外属性last_agentの読み出しが例外でも着地は成立する(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """解決（`exc.last_agent` の読み出し）が失敗しても例外は漏れず None で着地する。"""
    result, seen, records = await _land_with_audit(RaisingLastAgentError("boom"), caplog)

    assert isinstance(result, FailsafeResult)
    assert result.final_output == "landed"
    assert result.last_agent is None
    assert len(records) == 1
    assert len(seen) == 1


async def test_failsafe_call_run_dataのlast_agent読み出しが例外でも着地は成立する(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`run_data` は取れてもその `last_agent` 読み出しが失敗する場合も着地は成立する。"""
    result, seen, records = await _land_with_audit(RunDataRaisingLastAgentError("boom"), caplog)

    assert isinstance(result, FailsafeResult)
    assert result.final_output == "landed"
    assert result.last_agent is None
    assert len(records) == 1
    assert len(seen) == 1


async def test_failsafe_call_解決が例外でも全体規定へ落ちる(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """読み出し失敗は「解決不能」として扱われ、段 2 の具体 agent が採用される。"""
    policy = FailsafePolicy(
        handlers={
            RaisingRunDataError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)
        },
        fallback_last_agent=AGENT_POLICY_FALLBACK,
    )

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = await failsafe_call(policy, _thunk_raising(RaisingRunDataError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_POLICY_FALLBACK


# ---------------------------------------------------------------------------
# R1: 解決の防御（from_exception も同じ防御が効く）
# ---------------------------------------------------------------------------


def test_from_exception_run_data読み出しが例外でもlast_agentはNone() -> None:
    """手動着地でも解決の失敗は漏らさず、`last_agent` は None になる。"""
    result = FailsafeResult.from_exception(
        RaisingRunDataError("boom"), final_output="landed", last_agent=RUNNING_AGENT
    )

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None
    assert result.final_output == "landed"


def test_from_exception_例外属性last_agentの読み出しが例外でもlast_agentはNone() -> None:
    """`exc.last_agent` の読み出しが失敗する例外でも手動着地は成立する。"""
    result = FailsafeResult.from_exception(
        RaisingLastAgentError("boom"), final_output="landed", last_agent=RUNNING_AGENT
    )

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None


def test_from_exception_run_dataのlast_agent読み出しが例外でもlast_agentはNone() -> None:
    """`run_data` 経由の読み出しが失敗する例外でも手動着地は成立する。"""
    exc = RunDataRaisingLastAgentError("boom")

    result = FailsafeResult.from_exception(exc, final_output="landed", last_agent=RUNNING_AGENT)

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None


# ---------------------------------------------------------------------------
# R2: last_agent は repr に出さない（属性としては参照できる）
# ---------------------------------------------------------------------------


class _SensitiveAgent:
    """repr に機微情報を含むオブジェクト（Agent の代役・`repr=False` の検証用）。"""

    def __repr__(self) -> str:
        """機微情報を含む repr を返す。"""
        return "<Agent secret=SENSITIVE-PROMPT-TOKEN>"


def test_failsafe_result_reprにlast_agentが出ない() -> None:
    """`last_agent` は `repr=False` のため、内容もフィールド名も repr に現れない。"""
    agent = _SensitiveAgent()
    result = FailsafeResult(
        final_output="landed", exception=MyError("x"), matched_type=MyError, last_agent=agent
    )

    text = repr(result)

    assert "SENSITIVE-PROMPT-TOKEN" not in text
    assert "last_agent" not in text


def test_failsafe_result_reprは他フィールドを従来どおり出す() -> None:
    """`repr=False` は `last_agent` 限定で、他フィールドの repr 表示は変わらない。"""
    result = FailsafeResult(
        final_output="landed",
        exception=MyError("x"),
        matched_type=MyError,
        last_agent=_SensitiveAgent(),
    )

    text = repr(result)

    assert "final_output=" in text
    assert "matched_type=" in text
    assert "exception=" in text


def test_failsafe_result_repr非表示でもlast_agentは属性として参照できる() -> None:
    """repr から隠れても値は保持され、継続実行のための参照として読める。"""
    agent = _SensitiveAgent()
    result = FailsafeResult(
        final_output="landed", exception=MyError("x"), matched_type=MyError, last_agent=agent
    )

    assert result.last_agent is agent


def test_failsafe_result_last_agentはreprフィールドから除外されている() -> None:
    """dataclass フィールドのメタデータとして `repr=False` が宣言されている。"""
    by_name = {f.name: f for f in dataclasses.fields(FailsafeResult)}

    assert by_name["last_agent"].repr is False
    assert by_name["final_output"].repr is True


def test_failsafe_result_repr非表示はフィールド集合を変えない() -> None:
    """`repr=False` は表示のみの指定で、フィールド集合（構築互換）は従来どおり。"""
    assert [f.name for f in dataclasses.fields(FailsafeResult)] == [
        "final_output",
        "exception",
        "matched_type",
        "last_agent",
    ]


def test_failsafe_result_repr非表示でも等価比較はlast_agentを含む() -> None:
    """`repr=False` は比較に影響せず、`last_agent` の違いは `==` で区別される。"""
    exc = MyError("x")
    base = FailsafeResult(
        final_output="landed", exception=exc, matched_type=MyError, last_agent=AGENT_PER_EXCEPTION
    )
    same = FailsafeResult(
        final_output="landed", exception=exc, matched_type=MyError, last_agent=AGENT_PER_EXCEPTION
    )
    other = FailsafeResult(
        final_output="landed", exception=exc, matched_type=MyError, last_agent=AGENT_FROM_RUN_DATA
    )

    assert base == same
    assert base != other


def test_failsafe_handler_reprにlast_agentが出ない() -> None:
    """`FailsafeHandler.last_agent` も `repr=False` で、宣言の repr に機微が出ない。"""
    handler = FailsafeHandler(fallback="landed", last_agent=_SensitiveAgent())

    text = repr(handler)

    assert "SENSITIVE-PROMPT-TOKEN" not in text
    assert "last_agent" not in text
    assert "fallback=" in text


def test_failsafe_handler_repr非表示でもlast_agentは属性として参照できる() -> None:
    """repr から隠れても宣言値は保持され、段 1 の入力として読める。"""
    agent = _SensitiveAgent()
    handler = FailsafeHandler(fallback="landed", last_agent=agent)

    assert handler.last_agent is agent


def test_failsafe_handler_last_agentはreprフィールドから除外されている() -> None:
    """dataclass フィールドのメタデータとして `repr=False` が宣言されている。"""
    by_name = {f.name: f for f in dataclasses.fields(FailsafeHandler)}

    assert by_name["last_agent"].repr is False
    assert by_name["fallback"].repr is True


# ---------------------------------------------------------------------------
# R3: from_exception の入力検証（build-time ValueError）
# ---------------------------------------------------------------------------


def test_from_exception_非例外のexceptionは_ValueError() -> None:
    """`exception` が例外インスタンスでなければ ValueError（受け取った型名を含む）。"""
    with pytest.raises(ValueError, match="str"):
        FailsafeResult.from_exception("boom", final_output="landed")  # type: ignore[arg-type]


def test_from_exception_例外クラスをexceptionに渡すと_ValueError() -> None:
    """インスタンスでなくクラスを渡す取り違えも ValueError で fail-fast する。"""
    with pytest.raises(ValueError, match="MyError|type"):
        FailsafeResult.from_exception(MyError, final_output="landed")  # type: ignore[arg-type]


def test_from_exception_非例外クラスのmatched_typeは_ValueError() -> None:
    """`matched_type` が `int`（例外でないクラス）なら ValueError（型名を含む）。"""
    with pytest.raises(ValueError, match="int"):
        FailsafeResult.from_exception(
            MyError("boom"),
            final_output="landed",
            matched_type=int,  # type: ignore[arg-type]
        )


def test_from_exception_文字列のmatched_typeは_ValueError() -> None:
    """`matched_type` が文字列なら ValueError（型名を含む）。"""
    with pytest.raises(ValueError, match="str"):
        FailsafeResult.from_exception(
            MyError("boom"),
            final_output="landed",
            matched_type="MyError",  # type: ignore[arg-type]
        )


def test_from_exception_例外インスタンスのmatched_typeは_ValueError() -> None:
    """`matched_type` はクラスを要求するため、例外インスタンスは ValueError。"""
    with pytest.raises(ValueError, match="MyError"):
        FailsafeResult.from_exception(
            MyError("boom"),
            final_output="landed",
            matched_type=MyError("x"),  # type: ignore[arg-type]
        )


def test_from_exception_matched_type既定Noneは検証を通る() -> None:
    """既定（None）は「送出型を採る」の意味で、入力検証に弾かれない。"""
    result = FailsafeResult.from_exception(MySubError("boom"), final_output="landed")

    assert result.matched_type is MySubError


def test_from_exception_正当な例外サブクラスのmatched_typeは許容される() -> None:
    """`Exception` サブクラスの明示指定は検証を通り、そのまま結果に載る。"""
    result = FailsafeResult.from_exception(
        MySubError("boom"), final_output="landed", matched_type=MyError
    )

    assert result.matched_type is MyError


# ---------------------------------------------------------------------------
# FailsafePolicy.fallback_last_agent（段 2 = 全体規定）のフィールド契約
# ---------------------------------------------------------------------------


def test_failsafe_policy_fallback_last_agentの既定はNone() -> None:
    """段 2 の既定は None（= 全体規定なし）で、既存宣言の意味は変わらない。"""
    assert FailsafePolicy().fallback_last_agent is None
    assert FailsafePolicy(handlers={MyError: _fallback}).fallback_last_agent is None


def test_failsafe_policy_fallback_last_agentは明示指定した値を保持する() -> None:
    """明示指定した不透明値がそのまま読める。"""
    policy = FailsafePolicy(
        handlers={MyError: _fallback}, fallback_last_agent=AGENT_POLICY_FALLBACK
    )

    assert policy.fallback_last_agent is AGENT_POLICY_FALLBACK


def test_failsafe_policy_fallback_last_agentは既存フィールドと併用できる() -> None:
    """既存 3 フィールドと同時指定でき、相互に影響しない。"""

    def _on_apply(result: object) -> None:
        return None

    policy = FailsafePolicy(
        handlers={MyError: _fallback},
        log_on_apply=False,
        on_apply=_on_apply,
        fallback_last_agent=AGENT_POLICY_FALLBACK,
    )

    assert policy.handlers[MyError] is _fallback
    assert policy.log_on_apply is False
    assert policy.on_apply is _on_apply
    assert policy.fallback_last_agent is AGENT_POLICY_FALLBACK


def test_failsafe_policy_fallback_last_agentもfrozenで書き換え不可() -> None:
    """追加フィールドも frozen の対象で、構築後の差し替えはできない。"""
    policy = FailsafePolicy(
        handlers={MyError: _fallback}, fallback_last_agent=AGENT_POLICY_FALLBACK
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.fallback_last_agent = AGENT_PER_EXCEPTION  # type: ignore[misc]

    assert policy.fallback_last_agent is AGENT_POLICY_FALLBACK


# ---------------------------------------------------------------------------
# 決定表: 段 1 に具体 agent（段 2 の内容によらず段 1 が使われる）
# ---------------------------------------------------------------------------


async def test_失敗時_段1が具体agentなら段2の具体agentより優先される() -> None:
    """段 1 の具体 agent は無条件に確定し、段 2 は参照されない。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    policy = FailsafePolicy(
        handlers={
            SdkRunDataError: FailsafeHandler(fallback="landed", last_agent=AGENT_PER_EXCEPTION)
        },
        fallback_last_agent=AGENT_POLICY_FALLBACK,
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_PER_EXCEPTION
    assert result.last_agent is not AGENT_POLICY_FALLBACK
    assert result.last_agent is not AGENT_FROM_RUN_DATA


async def test_失敗時_段1が具体agentなら段2のRUNNING_AGENTより優先される() -> None:
    """段 2 が `RUNNING_AGENT`（解決可能）でも、段 1 の具体 agent が勝つ。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    policy = FailsafePolicy(
        handlers={
            SdkRunDataError: FailsafeHandler(fallback="landed", last_agent=AGENT_PER_EXCEPTION)
        },
        fallback_last_agent=RUNNING_AGENT,
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_PER_EXCEPTION


async def test_失敗時_段1の具体agentが空文字でも段2へ落ちない() -> None:
    """段 1 の採否は同一性判定のため、falsy な具体値が段 2 に飲み込まれない。"""
    policy = FailsafePolicy(
        handlers={MyError: FailsafeHandler(fallback="landed", last_agent="")},
        fallback_last_agent=AGENT_POLICY_FALLBACK,
    )

    result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent == ""
    assert result.last_agent is not AGENT_POLICY_FALLBACK


# ---------------------------------------------------------------------------
# 決定表: 段 1 に RUNNING_AGENT（解決できれば解決値・できなければ段 2 へ落ちる）
# ---------------------------------------------------------------------------


async def test_失敗時_段1のRUNNING_AGENTが解決できれば段2より優先される() -> None:
    """解決に成功した時点で確定し、段 2 の具体 agent は参照されない。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    policy = FailsafePolicy(
        handlers={SdkRunDataError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)},
        fallback_last_agent=AGENT_POLICY_FALLBACK,
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_FROM_RUN_DATA
    assert result.last_agent is not AGENT_POLICY_FALLBACK


async def test_失敗時_段1のRUNNING_AGENTの解決値がfalsyでも段2へ落ちない() -> None:
    """解決値が 0（falsy）でも「解決できた」とみなす（`or` 連鎖への退行検出）。"""
    exc = BudgetLikeError("boom", last_agent=0)
    policy = FailsafePolicy(
        handlers={BudgetLikeError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)},
        fallback_last_agent=AGENT_POLICY_FALLBACK,
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent == 0
    assert result.last_agent is not AGENT_POLICY_FALLBACK


async def test_失敗時_段1のRUNNING_AGENTが解決不能なら段2の具体agentへ落ちる() -> None:
    """解決材料の無い例外では段 2 まで落ち、そこに置いた具体 agent が採用される。"""
    policy = FailsafePolicy(
        handlers={MyError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)},
        fallback_last_agent=AGENT_POLICY_FALLBACK,
    )

    result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_POLICY_FALLBACK


async def test_失敗時_段1と段2の双方がRUNNING_AGENTで解決不能ならNone() -> None:
    """段 2 の `RUNNING_AGENT` も同じ例外から解決するため、結果は None（sentinel は載らない）。"""
    policy = FailsafePolicy(
        handlers={MyError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)},
        fallback_last_agent=RUNNING_AGENT,
    )

    result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None
    assert result.last_agent is not RUNNING_AGENT


async def test_失敗時_段1のRUNNING_AGENTが解決不能で段2未指定ならNone() -> None:
    """段 2 が無ければ「不明」を正直に表現して None になる。"""
    policy = FailsafePolicy(
        handlers={MyError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)}
    )

    result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None


# ---------------------------------------------------------------------------
# 決定表: 段 1 未指定（段 2 の内容で決まる）
# ---------------------------------------------------------------------------


async def test_失敗時_段1未指定なら段2の具体agentが使われる() -> None:
    """素の着地値宣言（段 1 なし）では段 2 の具体 agent が採用される。"""
    policy = FailsafePolicy(handlers={MyError: "landed"}, fallback_last_agent=AGENT_POLICY_FALLBACK)

    result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_POLICY_FALLBACK


async def test_失敗時_last_agent未指定のFailsafeHandlerでも段2が使われる() -> None:
    """`FailsafeHandler.last_agent` 既定 None は「段 1 の指定なし」なので段 2 まで落ちる。"""
    policy = FailsafePolicy(
        handlers={MyError: FailsafeHandler(fallback="landed")},
        fallback_last_agent=AGENT_POLICY_FALLBACK,
    )

    result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_POLICY_FALLBACK


async def test_失敗時_段1未指定で段2がRUNNING_AGENTなら解決値が使われる() -> None:
    """段 2 に `RUNNING_AGENT` を置けば、段 1 が無くても例外から解決される。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    policy = FailsafePolicy(handlers={SdkRunDataError: "landed"}, fallback_last_agent=RUNNING_AGENT)

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_FROM_RUN_DATA


async def test_失敗時_段1未指定で段2がRUNNING_AGENTでも解決不能ならNone() -> None:
    """段 2 の `RUNNING_AGENT` が解決できなければ None（sentinel はそのまま載らない）。"""
    policy = FailsafePolicy(handlers={MyError: "landed"}, fallback_last_agent=RUNNING_AGENT)

    result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None
    assert result.last_agent is not RUNNING_AGENT


async def test_失敗時_段1も段2も未指定ならNone() -> None:
    """どちらも無指定なら解決は一切走らず None になる。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    policy = FailsafePolicy(handlers={SdkRunDataError: "landed"})

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None


async def test_失敗時_段2の具体agentが空文字でも採用される() -> None:
    """段 2 の採否も同一性判定のため、空文字のような falsy 値が None に潰されない。"""
    policy = FailsafePolicy(handlers={MyError: "landed"}, fallback_last_agent="")

    result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent == ""
    assert result.last_agent is not None


async def test_失敗時_段2の具体agentが0でも採用される() -> None:
    """段 1 の `RUNNING_AGENT` が解決不能で落ちた先でも、falsy な具体値は潰されない。"""
    policy = FailsafePolicy(
        handlers={MyError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)},
        fallback_last_agent=0,
    )

    result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent == 0
    assert result.last_agent is not None


async def test_失敗時_段2で確定した値はon_applyが受け取る結果にも入る() -> None:
    """段 2 で確定した last_agent も監査コールバックから参照できる。"""
    seen: list[FailsafeResult] = []

    def _on_apply(result: FailsafeResult) -> None:
        seen.append(result)

    policy = FailsafePolicy(
        handlers={MyError: "landed"},
        on_apply=_on_apply,
        fallback_last_agent=AGENT_POLICY_FALLBACK,
    )

    result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert len(seen) == 1
    assert seen[0] is result
    assert seen[0].last_agent is AGENT_POLICY_FALLBACK


def test_from_exception_はpolicyを受け取らないため段2を持たない() -> None:
    """手動着地に policy は渡らないので、決定は段 1 相当（明示 or RUNNING_AGENT）のみ。"""
    assert "policy" not in inspect.signature(FailsafeResult.from_exception).parameters

    result = FailsafeResult.from_exception(MyError("boom"), final_output="landed")

    assert result.last_agent is None


# ---------------------------------------------------------------------------
# S1: FailsafePolicy.fallback_last_agent も repr に出さない（段 2 の機微保護）
# ---------------------------------------------------------------------------


def test_failsafe_policy_reprにfallback_last_agentが出ない() -> None:
    """段 2 の指定値も `repr=False` のため、内容もフィールド名も repr に現れない。"""
    policy = FailsafePolicy(handlers={MyError: "landed"}, fallback_last_agent=_SensitiveAgent())

    text = repr(policy)

    assert "SENSITIVE-PROMPT-TOKEN" not in text
    assert "fallback_last_agent" not in text


def test_failsafe_policy_reprは他フィールドを従来どおり出す() -> None:
    """`repr=False` は `fallback_last_agent` 限定で、他フィールドの repr 表示は変わらない。"""
    policy = FailsafePolicy(handlers={MyError: "landed"}, fallback_last_agent=_SensitiveAgent())

    text = repr(policy)

    assert "handlers=" in text
    assert "log_on_apply=" in text
    assert "on_apply=" in text


def test_failsafe_policy_repr非表示でもfallback_last_agentは属性として参照できる() -> None:
    """repr から隠れても宣言値は保持され、段 2 の入力として読める。"""
    agent = _SensitiveAgent()
    policy = FailsafePolicy(handlers={MyError: "landed"}, fallback_last_agent=agent)

    assert policy.fallback_last_agent is agent


def test_failsafe_policy_fallback_last_agentはreprフィールドから除外されている() -> None:
    """dataclass フィールドのメタデータとして `repr=False` が宣言されている。"""
    by_name = {f.name: f for f in dataclasses.fields(FailsafePolicy)}

    assert by_name["fallback_last_agent"].repr is False
    assert by_name["handlers"].repr is True


def test_failsafe_policy_repr非表示はフィールド集合と順序を変えない() -> None:
    """`repr=False` は表示のみの指定で、フィールドの集合と順序（構築互換）は従来どおり。"""
    assert [f.name for f in dataclasses.fields(FailsafePolicy)] == [
        "handlers",
        "log_on_apply",
        "on_apply",
        "fallback_last_agent",
    ]


def test_failsafe_policy_repr非表示でも等価比較はfallback_last_agentを含む() -> None:
    """`repr=False` は比較に影響せず、`fallback_last_agent` の違いは `==` で区別される。"""
    base = FailsafePolicy(handlers={MyError: "landed"}, fallback_last_agent=AGENT_POLICY_FALLBACK)
    same = FailsafePolicy(handlers={MyError: "landed"}, fallback_last_agent=AGENT_POLICY_FALLBACK)
    other = FailsafePolicy(handlers={MyError: "landed"}, fallback_last_agent=AGENT_PER_EXCEPTION)

    assert base == same
    assert base != other


# ---------------------------------------------------------------------------
# S2: RUNNING_AGENT は複製しても同一オブジェクトのまま（sentinel の同一性保存）
# ---------------------------------------------------------------------------


def test_running_agent_はcopyで同一オブジェクトのままである() -> None:
    """判定は `is RUNNING_AGENT` のため、浅い複製で別インスタンスになってはならない。"""
    assert copy.copy(RUNNING_AGENT) is RUNNING_AGENT


def test_running_agent_はdeepcopyで同一オブジェクトのままである() -> None:
    """宣言ごと deepcopy されても sentinel の同一性が保たれる（判定が外れない）。"""
    assert copy.deepcopy(RUNNING_AGENT) is RUNNING_AGENT


def test_running_agent_はpickle往復で同一オブジェクトのままである() -> None:
    """プロセス跨ぎ（pickle 往復）でも sentinel は復元されず同一オブジェクトを指す。

    `loads` の入力は同一プロセス内で lib の sentinel から生成した信頼できるバイト列のみで、
    外部入力は扱わない（複製時の同一性という契約の検証が目的）。
    """
    assert pickle.loads(pickle.dumps(RUNNING_AGENT)) is RUNNING_AGENT


async def test_failsafe_call_deepcopyしたRUNNING_AGENTを段1に置いても解決値になる() -> None:
    """複製された sentinel も合図として認識され、例外からの解決が走る。"""
    copied = copy.deepcopy(RUNNING_AGENT)
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    policy = FailsafePolicy(
        handlers={SdkRunDataError: FailsafeHandler(fallback="landed", last_agent=copied)}
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_FROM_RUN_DATA
    assert not isinstance(result.last_agent, type(RUNNING_AGENT))


async def test_failsafe_call_deepcopyしたRUNNING_AGENTが段1で解決不能ならNone() -> None:
    """複製された sentinel も「具体値」ではないため、解決できなければ結果に載らない。"""
    copied = copy.deepcopy(RUNNING_AGENT)
    policy = FailsafePolicy(
        handlers={MyError: FailsafeHandler(fallback="landed", last_agent=copied)}
    )

    result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None
    assert not isinstance(result.last_agent, type(RUNNING_AGENT))


async def test_failsafe_call_deepcopyしたRUNNING_AGENTを段2に置いても解決値になる() -> None:
    """段 2 に置いた複製 sentinel も合図として認識され、例外からの解決が走る。"""
    copied = copy.deepcopy(RUNNING_AGENT)
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    policy = FailsafePolicy(handlers={SdkRunDataError: "landed"}, fallback_last_agent=copied)

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_FROM_RUN_DATA
    assert not isinstance(result.last_agent, type(RUNNING_AGENT))


async def test_failsafe_call_deepcopyしたRUNNING_AGENTが段2で解決不能ならNone() -> None:
    """段 2 の複製 sentinel も解決できなければ None になり、sentinel は結果に載らない。"""
    copied = copy.deepcopy(RUNNING_AGENT)
    policy = FailsafePolicy(handlers={MyError: "landed"}, fallback_last_agent=copied)

    result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None
    assert not isinstance(result.last_agent, type(RUNNING_AGENT))


# ---------------------------------------------------------------------------
# D: FailsafeHandler(fallback=RUNNING_AGENT) の build-time 拒否
# ---------------------------------------------------------------------------


def test_failsafe_handler_fallbackへのRUNNING_AGENT宣言は_ValueError() -> None:
    """`fallback` に指定用 sentinel を置く誤宣言は build-time で ValueError。"""
    with pytest.raises(ValueError, match="fallback"):
        FailsafeHandler(fallback=RUNNING_AGENT)


def test_failsafe_handler_RUNNING_AGENT宣言の_ValueErrorはsentinel名を含む() -> None:
    """拒否のメッセージには `RUNNING_AGENT` の名が含まれ、原因が特定できる。"""
    with pytest.raises(ValueError) as excinfo:
        FailsafeHandler(fallback=RUNNING_AGENT, last_agent=AGENT_PER_EXCEPTION)

    assert "RUNNING_AGENT" in str(excinfo.value)


async def test_failsafe_handler_last_agentへのRUNNING_AGENTは拒否されない() -> None:
    """`last_agent` 側の `RUNNING_AGENT` は正当な指定で、宣言も着地も従来どおり動く。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    handler = FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)
    policy = FailsafePolicy(handlers={SdkRunDataError: handler})

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert handler.last_agent is RUNNING_AGENT
    assert isinstance(result, FailsafeResult)
    assert result.final_output == "landed"
    assert result.last_agent is AGENT_FROM_RUN_DATA


# ---------------------------------------------------------------------------
# D: handlers 値位置の RUNNING_AGENT の build-time 拒否（包む形と対称）
# ---------------------------------------------------------------------------


def test_failsafe_policy_handlers値位置の直接RUNNING_AGENTは_ValueError() -> None:
    """`FailsafeHandler` で包まない素の値位置に置いた sentinel も build-time で ValueError。

    `FailsafeHandler(fallback=RUNNING_AGENT)` の拒否と対称に、`handlers={E: RUNNING_AGENT}`
    も構築時に止める（放置すると指定用 sentinel がそのまま `final_output` に載る）。
    """
    with pytest.raises(ValueError, match="RUNNING_AGENT"):
        FailsafePolicy(handlers={MyError: RUNNING_AGENT})


def test_failsafe_policy_値位置RUNNING_AGENTの_ValueErrorはキー名を含む() -> None:
    """拒否のメッセージには違反キー（例外型名）が含まれ、どの宣言が誤りかを特定できる。"""
    with pytest.raises(ValueError, match="MyError"):
        FailsafePolicy(handlers={MyError: RUNNING_AGENT})


def test_failsafe_policy_値位置のdeepcopy済みRUNNING_AGENTも_ValueError() -> None:
    """`__reduce__` により複製でも同一性が保たれるため、複製 sentinel の誤配置も拒否される。"""
    copied = copy.deepcopy(RUNNING_AGENT)

    with pytest.raises(ValueError, match="RUNNING_AGENT"):
        FailsafePolicy(handlers={MyError: copied})


def test_failsafe_policy_複数キーのうち1件が値位置RUNNING_AGENTでも_ValueError() -> None:
    """検証は全キーを走査するため、正当な宣言に紛れた 1 件の誤配置も検出される。"""
    with pytest.raises(ValueError, match="TypeError"):
        FailsafePolicy(
            handlers={ValueError: "landed", MyError: _fallback, TypeError: RUNNING_AGENT}
        )


class _InconsistentMapping(Mapping[type[Exception], Any]):
    """反復と添字アクセスで別の内容を返す `Mapping`（検証迂回の再現用）。

    `__iter__` は空を装い、`keys()` / `__getitem__` は禁止キーと値位置 sentinel を
    返す。検証と格納で読み取り経路が違うと、この宣言が検証をすり抜けて格納される。
    """

    _real: dict[type[Exception], Any] = {Exception: "swallow-everything", KeyError: RUNNING_AGENT}

    def __iter__(self) -> Any:
        return iter(())

    def __len__(self) -> int:
        return 0

    def __getitem__(self, key: type[Exception]) -> Any:
        return self._real[key]

    def keys(self) -> Any:
        return self._real.keys()


def test_failsafe_policy_反復と添字が食い違うMappingでも検証は迂回できない() -> None:
    """検証対象と格納対象が一致するため、独自 `Mapping` で禁止宣言を紛れ込ませられない。

    検証が `__iter__` 系、格納が `keys()` / `__getitem__` 系を読む実装だと、両者で
    別の内容を返す `Mapping` に禁止キー（`Exception`）と値位置 sentinel を格納され、
    未宣言例外まで着地してしまう。
    """
    with pytest.raises(ValueError, match="Exception"):
        FailsafePolicy(handlers=_InconsistentMapping())


# ---------------------------------------------------------------------------
# D: 値位置検証の過剰拒否ガード（sentinel との同一性判定のみで弾く）
# ---------------------------------------------------------------------------


def test_failsafe_policy_値位置のFailsafeHandler経由のRUNNING_AGENT指定は許容される() -> None:
    """正当な指定形（`last_agent=RUNNING_AGENT`）は値位置検証で弾かれず保持される。"""
    handler = FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)

    policy = FailsafePolicy(handlers={MyError: handler})

    assert policy.handlers[MyError] is handler
    assert policy.handlers[MyError].last_agent is RUNNING_AGENT


def test_failsafe_policy_値位置のfalsyな着地値は許容される() -> None:
    """拒否は `is RUNNING_AGENT` の同一性判定のみのため、falsy な具体値は受理される。"""
    for value in ("", 0, False, None, [], {}):
        policy = FailsafePolicy(handlers={MyError: value})

        assert policy.handlers[MyError] is value


def test_failsafe_policy_値位置のcallableは許容される() -> None:
    """sync / async の callable fallback は従来どおり値位置に置ける。"""

    async def _async_fallback(received: Exception) -> str:
        return "async-fallback"

    policy = FailsafePolicy(handlers={MyError: _fallback, ValueError: _async_fallback})

    assert policy.handlers[MyError] is _fallback
    assert policy.handlers[ValueError] is _async_fallback


def test_failsafe_policy_値位置のAgent相当オブジェクトは許容される() -> None:
    """Agent 相当の不透明値を着地値に置く形は sentinel ではないため受理される。"""
    policy = FailsafePolicy(handlers={MyError: AGENT_PER_EXCEPTION})

    assert policy.handlers[MyError] is AGENT_PER_EXCEPTION


async def test_failsafe_call_値位置のfalsyな着地値はそのままfinal_outputになる() -> None:
    """値位置検証を通った falsy な具体値は着地時も加工されず `final_output` になる。"""
    empty_policy = FailsafePolicy(handlers={MyError: ""})
    none_policy = FailsafePolicy(handlers={MyError: None})

    empty = await failsafe_call(empty_policy, _thunk_raising(MyError("boom")))
    none = await failsafe_call(none_policy, _thunk_raising(MyError("boom")))

    assert isinstance(empty, FailsafeResult)
    assert isinstance(none, FailsafeResult)
    assert empty.final_output == ""
    assert none.final_output is None


# ---------------------------------------------------------------------------
# A: 解決の段ごと防御（片方の読み出し失敗が他方の読み取り先を飛ばさない）用のヘルパ
# ---------------------------------------------------------------------------


class RaisingRunDataWithLastAgentError(Exception):
    """`run_data` の読み出しは失敗するが、`last_agent` 属性は正常に読める例外。"""

    def __init__(self, message: str, last_agent: Any) -> None:
        super().__init__(message)
        self.last_agent = last_agent

    @property
    def run_data(self) -> Any:
        """常に `RuntimeError` を送出する（1 つ目の読み取り先の失敗）。"""
        raise RuntimeError("run_data access failed")


class RunDataRaisingWithLastAgentError(Exception):
    """`run_data` は取れるがその `last_agent` 読み出しが失敗し、例外属性は正常に読める例外。"""

    def __init__(self, message: str, last_agent: Any) -> None:
        super().__init__(message)
        self.run_data = _FakeRunDataRaisingLastAgent()
        self.last_agent = last_agent


class RunDataOkLastAgentRaisingError(Exception):
    """`run_data.last_agent` は読めるが、例外自身の `last_agent` 読み出しが失敗する例外。"""

    def __init__(self, message: str, run_data: Any) -> None:
        super().__init__(message)
        self.run_data = run_data

    @property
    def last_agent(self) -> Any:
        """常に `RuntimeError` を送出する（2 つ目の読み取り先の失敗）。"""
        raise RuntimeError("last_agent access failed")


class RaisingBothError(Exception):
    """`run_data` と `last_agent` の双方の読み出しが失敗する例外（解決不能の下限）。"""

    @property
    def run_data(self) -> Any:
        """常に `RuntimeError` を送出する。"""
        raise RuntimeError("run_data access failed")

    @property
    def last_agent(self) -> Any:
        """常に `RuntimeError` を送出する。"""
        raise RuntimeError("last_agent access failed")


def _running_agent_policy(exc_type: type[Exception]) -> FailsafePolicy:
    """段 1 に `RUNNING_AGENT` を置いた最小 policy を作る（解決経路を実際に走らせる）。"""
    return FailsafePolicy(
        handlers={exc_type: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)}
    )


# ---------------------------------------------------------------------------
# A: 解決の段ごと防御（failsafe_call）
# ---------------------------------------------------------------------------


async def test_failsafe_call_run_data読み出し失敗でも例外属性から解決される() -> None:
    """1 つ目の読み取り先の失敗で連鎖を打ち切らず、`exc.last_agent` まで試みる。"""
    exc = RaisingRunDataWithLastAgentError("boom", AGENT_FROM_ATTRIBUTE)

    result = await failsafe_call(
        _running_agent_policy(RaisingRunDataWithLastAgentError), _thunk_raising(exc)
    )

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_FROM_ATTRIBUTE
    assert result.final_output == "landed"


async def test_failsafe_call_run_dataのlast_agent読み出し失敗でも例外属性から解決される() -> None:
    """`run_data` 自体は取れてもその `last_agent` が失敗する場合も、次の読み取り先へ進む。"""
    exc = RunDataRaisingWithLastAgentError("boom", AGENT_FROM_ATTRIBUTE)

    result = await failsafe_call(
        _running_agent_policy(RunDataRaisingWithLastAgentError), _thunk_raising(exc)
    )

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_FROM_ATTRIBUTE


async def test_failsafe_call_例外属性の読み出し失敗はrun_data由来の解決を妨げない() -> None:
    """逆順（2 つ目の読み取り先が失敗）でも、先に成功した `run_data` 由来の値が確定する。"""
    exc = RunDataOkLastAgentRaisingError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))

    result = await failsafe_call(
        _running_agent_policy(RunDataOkLastAgentRaisingError), _thunk_raising(exc)
    )

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_FROM_RUN_DATA


async def test_failsafe_call_双方の読み出しが失敗すればlast_agentはNone() -> None:
    """どちらの読み取り先も失敗したときだけ解決不能（None）で、着地自体は成立する。"""
    result = await failsafe_call(
        _running_agent_policy(RaisingBothError), _thunk_raising(RaisingBothError("boom"))
    )

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None
    assert result.final_output == "landed"


async def test_failsafe_call_読み出し失敗はdebugに記録され解決は次の読み取り先へ進む(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """失敗した読み取り先は debug に記録するだけで、解決の連鎖は止まらない。"""
    exc = RaisingRunDataWithLastAgentError("boom", AGENT_FROM_ATTRIBUTE)

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        result = await failsafe_call(
            _running_agent_policy(RaisingRunDataWithLastAgentError), _thunk_raising(exc)
        )

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_FROM_ATTRIBUTE
    assert len(_records_of(caplog, logging.DEBUG)) >= 1


async def test_failsafe_call_読み出し失敗があっても監査は従来どおり発火する(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """段ごと防御でも warning と `on_apply` の発火は変わらず、解決値だけが改善する。"""
    exc = RaisingRunDataWithLastAgentError("boom", AGENT_FROM_ATTRIBUTE)

    result, seen, records = await _land_with_audit(exc, caplog)

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_FROM_ATTRIBUTE
    assert len(records) == 1
    assert len(seen) == 1
    assert seen[0] is result


# ---------------------------------------------------------------------------
# A: 解決の段ごと防御（from_exception でも同じ連鎖になる）
# ---------------------------------------------------------------------------


def test_from_exception_run_data読み出し失敗でも例外属性から解決される() -> None:
    """手動着地でも連鎖は中断せず、`failsafe_call` と同じ値に解決される。"""
    exc = RaisingRunDataWithLastAgentError("boom", AGENT_FROM_ATTRIBUTE)

    result = FailsafeResult.from_exception(exc, final_output="landed", last_agent=RUNNING_AGENT)

    assert result.last_agent is AGENT_FROM_ATTRIBUTE


async def test_from_exception_の段ごと防御はfailsafe_call着地時と一致する() -> None:
    """読み出しが失敗する例外でも、手動着地と自動着地で `last_agent` の意味が揃う。"""
    exc = RaisingRunDataWithLastAgentError("boom", AGENT_FROM_ATTRIBUTE)

    landed = await failsafe_call(
        _running_agent_policy(RaisingRunDataWithLastAgentError), _thunk_raising(exc)
    )
    manual = FailsafeResult.from_exception(exc, final_output="landed", last_agent=RUNNING_AGENT)

    assert isinstance(landed, FailsafeResult)
    assert manual.last_agent is landed.last_agent
    assert manual.last_agent is AGENT_FROM_ATTRIBUTE


def test_from_exception_双方の読み出しが失敗すればlast_agentはNone() -> None:
    """手動着地でも「どちらも失敗」のときだけ None になる（着地は成立する）。"""
    result = FailsafeResult.from_exception(
        RaisingBothError("boom"), final_output="landed", last_agent=RUNNING_AGENT
    )

    assert result.last_agent is None
    assert result.final_output == "landed"


# ---------------------------------------------------------------------------
# B1: RUNNING_AGENT の singleton 強制（直接構築でも同一インスタンス）
# ---------------------------------------------------------------------------


def test_running_agent_直接構築してもモジュールsingletonを返す() -> None:
    """内部 sentinel 型を直接構築しても `RUNNING_AGENT` と同一インスタンスになる。"""
    assert _RunningAgentSentinel() is RUNNING_AGENT
    assert _RunningAgentSentinel() is _RunningAgentSentinel()


def test_failsafe_handler_直接構築sentinelのfallback指定は_ValueError() -> None:
    """singleton 強制により、直接構築した sentinel でも fallback 位置のガードが効く。"""
    with pytest.raises(ValueError, match="RUNNING_AGENT"):
        FailsafeHandler(fallback=_RunningAgentSentinel())


def test_failsafe_policy_値位置の直接構築sentinelは_ValueError() -> None:
    """handlers 値位置のガードも迂回できない（sentinel が `final_output` に載らない）。"""
    with pytest.raises(ValueError, match="RUNNING_AGENT"):
        FailsafePolicy(handlers={MyError: _RunningAgentSentinel()})


async def test_failsafe_call_直接構築sentinelを段1に置いても解決の合図になる() -> None:
    """直接構築した sentinel も「具体値」ではなく解決の合図として扱われる。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))
    policy = FailsafePolicy(
        handlers={
            SdkRunDataError: FailsafeHandler(fallback="landed", last_agent=_RunningAgentSentinel())
        }
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_FROM_RUN_DATA
    assert not isinstance(result.last_agent, _RunningAgentSentinel)


# ---------------------------------------------------------------------------
# B2: ExceptionGroup の禁止（集約例外を丸ごと飲む宣言の拒否）
# ---------------------------------------------------------------------------


class MyErrorGroup(ExceptionGroup):
    """ユーザー定義の `ExceptionGroup` サブクラス（禁止列挙メンバーそのものではない）。"""


def test_failsafe_policy_handlers_ExceptionGroupキーは_ValueError() -> None:
    """`ExceptionGroup` そのものは集約例外を丸ごと飲むため禁止（ValueError・キー名を含む）。"""
    with pytest.raises(ValueError, match="ExceptionGroup"):
        FailsafePolicy(handlers={ExceptionGroup: _fallback})  # type: ignore[dict-item]


def test_failsafe_policy_handlers_BaseExceptionGroupキーは_ValueError() -> None:
    """`BaseExceptionGroup` も従来どおり拒否される（ExceptionGroup 側との非対称を解消）。"""
    with pytest.raises(ValueError, match="BaseExceptionGroup"):
        FailsafePolicy(handlers={BaseExceptionGroup: _fallback})  # type: ignore[dict-item]


def test_failsafe_policy_handlers_ユーザー定義ExceptionGroupサブクラスは許容される() -> None:
    """禁止は列挙メンバーそのものに限るため、利用者定義のサブクラスは受理される。"""
    policy = FailsafePolicy(handlers={MyErrorGroup: _fallback})

    assert policy.handlers[MyErrorGroup] is _fallback


async def test_failsafe_call_ユーザー定義ExceptionGroupサブクラスで着地できる() -> None:
    """過剰拒否していないことを着地まで通して確認する（宣言も実行も従来どおり）。"""
    exc = MyErrorGroup("boom", [ValueError("inner")])
    policy = FailsafePolicy(handlers={MyErrorGroup: "landed"})

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.final_output == "landed"
    assert result.matched_type is MyErrorGroup


# ---------------------------------------------------------------------------
# C1: thunk 受理契約の検査は handlers 空でも同じメッセージで行われる
# ---------------------------------------------------------------------------


async def test_failsafe_call_handlers空でも非awaitable戻り値はlibのメッセージでfail_fastする() -> (
    None
):
    """handlers 空（no-op）でも受理契約違反は lib のメッセージで fail-fast する。"""
    with pytest.raises(TypeError, match="thunk must return an awaitable, got 'int'"):
        await failsafe_call(FailsafePolicy(), lambda: 42)  # type: ignore[arg-type,return-value]


async def test_failsafe_call_非awaitableのメッセージはhandlers空でも非空でも一致する() -> None:
    """受理契約の検査位置は handlers の有無に依存しない（メッセージが分岐しない）。"""
    with pytest.raises(TypeError) as empty:
        await failsafe_call(FailsafePolicy(handlers={}), lambda: 42)  # type: ignore[arg-type,return-value]

    with pytest.raises(TypeError) as declared:
        policy = FailsafePolicy(handlers={MyError: "landed"})
        await failsafe_call(policy, lambda: 42)  # type: ignore[arg-type,return-value]

    assert str(empty.value) == str(declared.value)


async def test_failsafe_call_handlers空でもthunkの呼び出しは1回だけ() -> None:
    """検査の前倒しで thunk が二重に呼ばれない（副作用のある thunk の退行検出）。"""
    calls: list[str] = []

    async def _thunk() -> str:
        calls.append("called")
        return "ok"

    result = await failsafe_call(FailsafePolicy(), _thunk)

    assert result == "ok"
    assert calls == ["called"]


# ---------------------------------------------------------------------------
# C3: FailsafeResult の直接構築にも from_exception と同一の入力検証が効く
# ---------------------------------------------------------------------------


def test_failsafe_result_直接構築の非例外exceptionは_ValueError() -> None:
    """`exception` が例外インスタンスでなければ ValueError（受け取った型名を含む）。"""
    with pytest.raises(ValueError, match="str"):
        FailsafeResult(
            final_output="x",
            exception="not an exception",  # type: ignore[arg-type]
            matched_type=ValueError,
        )


def test_failsafe_result_直接構築で例外クラスをexceptionに渡すと_ValueError() -> None:
    """インスタンスでなくクラスを渡す取り違えも直接構築で fail-fast する。"""
    with pytest.raises(ValueError, match="MyError|type"):
        FailsafeResult(
            final_output="x",
            exception=MyError,  # type: ignore[arg-type]
            matched_type=MyError,
        )


def test_failsafe_result_直接構築の非例外クラスmatched_typeは_ValueError() -> None:
    """`matched_type` が `str`（例外でないクラス）なら ValueError（型名を含む）。"""
    with pytest.raises(ValueError, match="str"):
        FailsafeResult(
            final_output="x",
            exception=MyError("boom"),
            matched_type=str,  # type: ignore[arg-type]
        )


def test_failsafe_result_直接構築の例外インスタンスmatched_typeは_ValueError() -> None:
    """`matched_type` はクラスを要求するため、例外インスタンスは ValueError。"""
    with pytest.raises(ValueError, match="MyError"):
        FailsafeResult(
            final_output="x",
            exception=MyError("boom"),
            matched_type=MyError("x"),  # type: ignore[arg-type]
        )


def test_failsafe_result_直接構築の検証メッセージはfrom_exceptionと一致する() -> None:
    """同一型に 2 つの契約を作らないため、文体（メッセージ）も揃える。"""
    with pytest.raises(ValueError) as direct:
        FailsafeResult(
            final_output="x",
            exception="not an exception",  # type: ignore[arg-type]
            matched_type=MyError,
        )

    with pytest.raises(ValueError) as factory:
        FailsafeResult.from_exception("not an exception", final_output="x")  # type: ignore[arg-type]

    assert str(direct.value) == str(factory.value)


def test_failsafe_result_正当な直接構築は従来どおり許容される() -> None:
    """検証追加後も、例外インスタンス + `Exception` サブクラス + 任意 last_agent は通る。"""
    exc = MySubError("boom")

    full = FailsafeResult(
        final_output="landed", exception=exc, matched_type=MyError, last_agent=AGENT_PER_EXCEPTION
    )
    minimal = FailsafeResult(final_output=None, exception=exc, matched_type=MySubError)

    assert full.exception is exc
    assert full.matched_type is MyError
    assert full.last_agent is AGENT_PER_EXCEPTION
    assert minimal.matched_type is MySubError
    assert minimal.last_agent is None


def test_from_exception_は直接構築の検証と二重に弾かれない() -> None:
    """ファクトリ経由の正当な構築は、追加された直接構築の検証にも弾かれない。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(AGENT_FROM_RUN_DATA))

    default_type = FailsafeResult.from_exception(exc, final_output="landed")
    explicit_type = FailsafeResult.from_exception(
        exc, final_output="landed", matched_type=Exception, last_agent=RUNNING_AGENT
    )

    assert default_type.matched_type is SdkRunDataError
    assert explicit_type.matched_type is Exception
    assert explicit_type.last_agent is AGENT_FROM_RUN_DATA


# ---------------------------------------------------------------------------
# V1: 例外側の RUNNING_AGENT は解決不能として扱う（sentinel は結果に載らない）
# ---------------------------------------------------------------------------


async def test_failsafe_call_例外属性のRUNNING_AGENTは解決値にならず段2へ落ちる() -> None:
    """例外の `last_agent` が sentinel そのものなら解決不能扱いで、段 2 の具体 agent が使われる。"""
    exc = BudgetLikeError("boom", last_agent=RUNNING_AGENT)
    policy = FailsafePolicy(
        handlers={BudgetLikeError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)},
        fallback_last_agent=AGENT_POLICY_FALLBACK,
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_POLICY_FALLBACK
    assert result.last_agent is not RUNNING_AGENT


async def test_failsafe_call_例外属性のRUNNING_AGENTで段2未指定ならNone() -> None:
    """解決不能かつ段 2 未指定なら None になり、sentinel が着地結果に漏れない。"""
    exc = BudgetLikeError("boom", last_agent=RUNNING_AGENT)
    policy = FailsafePolicy(
        handlers={BudgetLikeError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)}
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None


async def test_failsafe_call_run_dataのRUNNING_AGENTは次の読み取り先へ進む() -> None:
    """1 つ目の読み取り先が sentinel なら採らず、2 つ目（`exc.last_agent`）から解決する。"""
    exc = HybridError("boom", run_data=_FakeRunData(RUNNING_AGENT), last_agent=AGENT_FROM_ATTRIBUTE)
    policy = FailsafePolicy(
        handlers={HybridError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)}
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_FROM_ATTRIBUTE
    assert result.last_agent is not RUNNING_AGENT


async def test_failsafe_call_run_dataのRUNNING_AGENTで例外属性が無ければ段2へ落ちる() -> None:
    """`run_data` の値が sentinel で他の読み取り先も無ければ、段 2 の具体 agent が使われる。"""
    exc = SdkRunDataError("boom", run_data=_FakeRunData(RUNNING_AGENT))
    policy = FailsafePolicy(
        handlers={SdkRunDataError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)},
        fallback_last_agent=AGENT_POLICY_FALLBACK,
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is AGENT_POLICY_FALLBACK


async def test_failsafe_call_双方の読み取り先がRUNNING_AGENTならNone() -> None:
    """両方の読み取り先が sentinel なら解決不能として None になる。"""
    exc = HybridError("boom", run_data=_FakeRunData(RUNNING_AGENT), last_agent=RUNNING_AGENT)
    policy = FailsafePolicy(
        handlers={HybridError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)}
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent is None


async def test_failsafe_call_sentinelを飛ばした先のfalsyな解決値は採用される() -> None:
    """除外は sentinel との同一性のみで行う（飛ばした先の falsy な具体値は従来どおり採る）。"""
    exc = HybridError("boom", run_data=_FakeRunData(RUNNING_AGENT), last_agent=0)
    policy = FailsafePolicy(
        handlers={HybridError: FailsafeHandler(fallback="landed", last_agent=RUNNING_AGENT)},
        fallback_last_agent=AGENT_POLICY_FALLBACK,
    )

    result = await failsafe_call(policy, _thunk_raising(exc))

    assert isinstance(result, FailsafeResult)
    assert result.last_agent == 0
    assert result.last_agent is not AGENT_POLICY_FALLBACK


def test_from_exception_例外属性のRUNNING_AGENTは解決値にならずNoneになる() -> None:
    """`from_exception` でも sentinel は結果に載らず（段 2 が無いため）None になる。"""
    exc = BudgetLikeError("boom", last_agent=RUNNING_AGENT)

    result = FailsafeResult.from_exception(exc, final_output="landed", last_agent=RUNNING_AGENT)

    assert result.last_agent is None
    assert result.last_agent is not RUNNING_AGENT


def test_from_exception_run_dataのRUNNING_AGENTは次の読み取り先へ進む() -> None:
    """`from_exception` の解決も sentinel を採らず、2 つ目の読み取り先へ進む。"""
    exc = HybridError("boom", run_data=_FakeRunData(RUNNING_AGENT), last_agent=AGENT_FROM_ATTRIBUTE)

    result = FailsafeResult.from_exception(exc, final_output="landed", last_agent=RUNNING_AGENT)

    assert result.last_agent is AGENT_FROM_ATTRIBUTE


# ---------------------------------------------------------------------------
# V5: log_on_apply と on_apply は独立（ログ抑止でも監査コールバックは発火する）
# ---------------------------------------------------------------------------


async def test_failsafe_call_log_on_apply_Falseでもon_applyは発火する(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """機微を伏せるための `log_on_apply=False` は `on_apply` の呼び出しを止めない。"""
    seen: list[FailsafeResult] = []

    def _on_apply(result: FailsafeResult) -> None:
        seen.append(result)

    policy = FailsafePolicy(handlers={MyError: "landed"}, log_on_apply=False, on_apply=_on_apply)

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert _records_of(caplog, logging.WARNING) == []
    assert len(seen) == 1
    assert seen[0] is result


async def test_failsafe_call_log_on_apply_Falseでもon_apply例外はerrorログに残る(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """warning の抑止は監査経路の失敗記録（error ログ）までは抑止しない。"""

    def _on_apply(result: FailsafeResult) -> None:
        raise RuntimeError("callback failed")

    policy = FailsafePolicy(handlers={MyError: "landed"}, log_on_apply=False, on_apply=_on_apply)

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert _records_of(caplog, logging.WARNING) == []
    errors = _records_of(caplog, logging.ERROR)
    assert len(errors) == 1
    assert errors[0].exc_info is not None


# ---------------------------------------------------------------------------
# V4: async な on_apply の例外も握り潰される（await が監査の try の内側にある）
# ---------------------------------------------------------------------------


async def test_failsafe_call_async_on_applyの例外はerrorログで握り潰される(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """await 中に送出する `on_apply` でも例外は漏れず、着地結果の返却は継続する。"""

    async def _on_apply(result: FailsafeResult) -> None:
        await asyncio.sleep(0)
        raise RuntimeError("async callback failed")

    policy = FailsafePolicy(handlers={MyError: "landed"}, on_apply=_on_apply)

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        result = await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert isinstance(result, FailsafeResult)
    assert result.final_output == "landed"
    errors = _records_of(caplog, logging.ERROR)
    assert len(errors) == 1
    assert errors[0].exc_info is not None
    assert f"matched_type={MyError.__name__}" in errors[0].getMessage()


# ---------------------------------------------------------------------------
# V6: thunk の呼び出しは handlers 非空の正常系・着地経路でも 1 回だけ
# ---------------------------------------------------------------------------


async def test_failsafe_call_handlers非空の正常完了でもthunkの呼び出しは1回だけ() -> None:
    """受理契約検査と本体 await で thunk が二重に呼ばれない（副作用の二重実行の退行検出）。"""
    calls: list[str] = []

    def _thunk():  # noqa: ANN202
        calls.append("called")

        async def _inner() -> str:
            return "ok"

        return _inner()

    result = await failsafe_call(FailsafePolicy(handlers={MyError: "landed"}), _thunk)

    assert result == "ok"
    assert calls == ["called"]


async def test_failsafe_call_着地経路でもthunkの呼び出しは1回だけ() -> None:
    """着地する経路（宣言例外の送出）でも thunk の呼び出しは 1 回に留まる。"""
    calls: list[str] = []
    exc = MyError("boom")

    def _thunk():  # noqa: ANN202
        calls.append("called")

        async def _inner() -> str:
            raise exc

        return _inner()

    result = await failsafe_call(FailsafePolicy(handlers={MyError: "landed"}), _thunk)

    assert isinstance(result, FailsafeResult)
    assert result.exception is exc
    assert calls == ["called"]


# ---------------------------------------------------------------------------
# V7: async な fallback callable 自身の例外は素通しする（着地は成立しない）
# ---------------------------------------------------------------------------


async def test_failsafe_call_async_fallback自身の例外はそのまま伝播する(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """await 中に送出する fallback でも着地は成立せず、例外は素通しし監査も発火しない。"""
    fallback_exc = RuntimeError("async fallback failed")
    called: list[str] = []

    async def _fb(received: Exception) -> str:
        await asyncio.sleep(0)
        raise fallback_exc

    def _on_apply(result: FailsafeResult) -> None:
        called.append("on_apply")

    policy = FailsafePolicy(handlers={MyError: _fb}, on_apply=_on_apply)

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        with pytest.raises(RuntimeError) as excinfo:
            await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert excinfo.value is fallback_exc
    assert called == []
    assert [r for r in caplog.records if r.name == _LOGGER_NAME] == []


async def test_failsafe_call_handler経由のasync_fallback例外もそのまま伝播する(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`FailsafeHandler` で包んでも async fallback の例外の扱いは同一（素通し・監査なし）。"""
    fallback_exc = RuntimeError("async fallback failed")
    called: list[str] = []

    async def _fb(received: Exception) -> str:
        await asyncio.sleep(0)
        raise fallback_exc

    def _on_apply(result: FailsafeResult) -> None:
        called.append("on_apply")

    policy = FailsafePolicy(
        handlers={MyError: FailsafeHandler(fallback=_fb, last_agent=AGENT_PER_EXCEPTION)},
        on_apply=_on_apply,
    )

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        with pytest.raises(RuntimeError) as excinfo:
            await failsafe_call(policy, _thunk_raising(MyError("boom")))

    assert excinfo.value is fallback_exc
    assert called == []
    assert [r for r in caplog.records if r.name == _LOGGER_NAME] == []


# ---------------------------------------------------------------------------
# V9: thunk が同期送出した例外は着地対象外（await 中の例外のみ着地させる）
# ---------------------------------------------------------------------------


async def test_failsafe_call_thunkが同期送出した宣言例外は着地せず伝播する(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """awaitable を返す前に送出する thunk は宣言済みの型でも着地せず、素通しする。"""
    exc = MyError("boom")
    called: list[str] = []

    def _sync_raising() -> Any:
        raise exc

    def _on_apply(result: FailsafeResult) -> None:
        called.append("on_apply")

    policy = FailsafePolicy(handlers={MyError: "landed"}, on_apply=_on_apply)

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        with pytest.raises(MyError) as excinfo:
            await failsafe_call(policy, _sync_raising)

    assert excinfo.value is exc
    assert excinfo.value.__cause__ is None
    assert called == []
    assert [r for r in caplog.records if r.name == _LOGGER_NAME] == []
