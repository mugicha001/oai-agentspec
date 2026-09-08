"""L2: `_adapters.context_capture.RunContextCapture` の run スコープ捕捉を実 SDK 型で検証する。

経路C（`as_agent_spec`）で外側 context を `WorkflowModel` へ届けるための捕捉テーブルの単体契約:
lib 所有 `AgentHooks.on_start` が現在 span（`agents.tracing.get_current_span()`）をキーに
`RunContextWrapper` を登録し、`resolve()` が同一 span で同じ wrapper を `is` で返す。現在 span が
無い / 未登録 span では `None`（警告・例外なし）。保持は `WeakKeyDictionary` のため span の参照を
落とすとエントリ（値側の wrapper）も解放される。`chain_agent_hooks` との合成（`is` 素通し・
合成後も捕捉が動き利用者 hook も呼ばれる）と tracing 無効時（`NoOpSpan`）の成立も pin する。

span は `agents.tracing.custom_span(...)` を `with` で current にして作る（tracing 有効時は
`SpanImpl`、無効時は `NoOpSpan`）。本パッケージは tests 配下であり NFR-1 の grep 計測対象外。
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from agents import Agent, RunContextWrapper, set_tracing_disabled
from agents.lifecycle import AgentHooksBase
from agents.tracing import custom_span, get_current_span, trace
from agents.tracing.spans import NoOpSpan, SpanImpl

from oai_agentspec._adapters import context_capture
from oai_agentspec.runtime.hooks import chain_agent_hooks

pytestmark = pytest.mark.integration


@dataclass
class _AppCtx:
    """捕捉対象の利用者 context（同一性を `is` で照合する）。"""

    token: str


@pytest.fixture
def _tracing_enabled() -> Iterator[None]:
    """tracing を一時的に有効化し `custom_span` が `SpanImpl` を返す状態にする。

    ルート conftest の autouse フィクスチャが毎テスト tracing を無効化するため、明示的に
    有効化してから終了時に無効へ戻す。
    """
    set_tracing_disabled(False)
    try:
        yield
    finally:
        set_tracing_disabled(True)


def _agent() -> Agent:
    """`on_start` の第 2 引数に渡す実 Agent（内容は問わない）。"""
    return Agent(name="probe")


# ---------------------------------------------------------------------------
# 捕捉と解決（同一 span で `is` 一致）
# ---------------------------------------------------------------------------
@pytest.mark.usefixtures("_tracing_enabled")
async def test_on_start_registers_wrapper_for_current_span() -> None:
    """`on_start` が現在 span（`SpanImpl`）をキーに wrapper を登録し `resolve()` が同一物を返す。

    登録前は同じ span でも未登録のため None。
    """
    capture = context_capture.RunContextCapture()
    wrapper = RunContextWrapper(context=_AppCtx(token="tk_1"))

    with trace("capture-test"), custom_span("turn") as span:
        assert isinstance(span, SpanImpl)
        assert get_current_span() is span
        # 登録前は同じ span でも未登録のため None。
        assert capture.resolve() is None
        await capture.hooks.on_start(wrapper, _agent())
        resolved = capture.resolve()

    assert resolved is wrapper
    assert resolved.context.token == "tk_1"


@pytest.mark.usefixtures("_tracing_enabled")
async def test_resolve_is_idempotent_within_same_span() -> None:
    """同一 span 内で `resolve()` を複数回呼んでも同じ wrapper を返す（読み取りで消費しない）。

    SDK の `get_response_with_retry` は同一 turn 内で `get_response` を再呼び出しするため、
    最初の解決でエントリを pop する実装だと再試行時に None へ落ちる。
    """
    capture = context_capture.RunContextCapture()
    wrapper = RunContextWrapper(context=_AppCtx(token="tk_retry"))

    with trace("capture-test"), custom_span("turn"):
        await capture.hooks.on_start(wrapper, _agent())
        first = capture.resolve()
        second = capture.resolve()

    assert first is wrapper
    assert second is wrapper


def test_resolve_without_current_span_returns_none() -> None:
    """現在 span が無い（`Runner.run` を経ない直接呼び出し相当）と `resolve()` は None。"""
    capture = context_capture.RunContextCapture()
    assert get_current_span() is None
    assert capture.resolve() is None


@pytest.mark.usefixtures("_tracing_enabled")
async def test_resolve_for_unregistered_span_returns_none() -> None:
    """別 span で登録した wrapper は、未登録の現在 span からは解決されない（span 単位の分離）。"""
    capture = context_capture.RunContextCapture()
    wrapper = RunContextWrapper(context=_AppCtx(token="tk_a"))

    with trace("capture-test"):
        with custom_span("turn-a"):
            await capture.hooks.on_start(wrapper, _agent())
        with custom_span("turn-b") as span_b:
            assert get_current_span() is span_b
            assert capture.resolve() is None


@pytest.mark.usefixtures("_tracing_enabled")
async def test_separate_captures_do_not_share_entries() -> None:
    """`RunContextCapture` インスタンスごとにテーブルは独立し、他インスタンスの登録は見えない。

    1 `WorkflowModel` につき 1 capture（共有レジストリ不採用）の pin。
    """
    capture_a = context_capture.RunContextCapture()
    capture_b = context_capture.RunContextCapture()
    wrapper = RunContextWrapper(context=_AppCtx(token="tk_a"))

    with trace("capture-test"), custom_span("turn"):
        await capture_a.hooks.on_start(wrapper, _agent())
        assert capture_a.resolve() is wrapper
        assert capture_b.resolve() is None


def test_hooks_property_returns_stable_instance() -> None:
    """`hooks` は毎回同じインスタンスを返す（`AgentSpec.hooks` に載せる 1 オブジェクト）。"""
    capture = context_capture.RunContextCapture()
    hooks = capture.hooks
    assert hooks is capture.hooks
    assert isinstance(hooks, AgentHooksBase)


# ---------------------------------------------------------------------------
# 寿命: span の参照を落とすとエントリ（wrapper）が解放される
# ---------------------------------------------------------------------------
async def test_entry_is_released_when_span_is_collected() -> None:
    """span の参照を全て落とし `gc.collect()` すると、登録した wrapper への強参照も消える。

    `WeakKeyDictionary` の値側は強参照のため、テーブルがエントリを保持し続けていれば wrapper の
    weakref は生存し続ける。wrapper の weakref が None になることで「エントリ消滅」を観測する
    （テーブルの私有属性に依存しない）。span は tracing 無効時の `NoOpSpan`（exporter キューが
    保持しないため run 終了で即解放される種類）を使う。
    """
    capture = context_capture.RunContextCapture()
    wrapper: Any = RunContextWrapper(context=_AppCtx(token="tk_gc"))
    wrapper_ref = weakref.ref(wrapper)

    with custom_span("turn") as span:
        assert isinstance(span, NoOpSpan)
        await capture.hooks.on_start(wrapper, _agent())
        assert capture.resolve() is wrapper

    del span
    del wrapper
    gc.collect()

    assert wrapper_ref() is None


# ---------------------------------------------------------------------------
# tracing 無効時（NoOpSpan）でも成立する
# ---------------------------------------------------------------------------
async def test_capture_works_with_noop_span_when_tracing_disabled() -> None:
    """tracing 無効時の `NoOpSpan` もキーになり、同一 span 内で wrapper を解決できる。"""
    capture = context_capture.RunContextCapture()
    wrapper = RunContextWrapper(context=_AppCtx(token="tk_noop"))

    with custom_span("turn") as span:
        assert isinstance(span, NoOpSpan)
        assert get_current_span() is span
        await capture.hooks.on_start(wrapper, _agent())
        assert capture.resolve() is wrapper

    # NoOpSpan は呼び出しごとに一意のため、別 with ブロックからは解決されない。
    with custom_span("turn-2"):
        assert capture.resolve() is None


# ---------------------------------------------------------------------------
# chain_agent_hooks との合成（FR-3）
# ---------------------------------------------------------------------------
def test_chain_agent_hooks_passes_capture_hooks_through() -> None:
    """`chain_agent_hooks(capture.hooks)` は実効 1 件の `AgentHooksBase` として `is` 素通しする。"""
    capture = context_capture.RunContextCapture()
    assert chain_agent_hooks(capture.hooks) is capture.hooks


class _OwnHooks(AgentHooksBase[Any, Any]):
    """利用者フック相当。`on_start` の引数をそのまま記録する。"""

    def __init__(self) -> None:
        self.starts: list[tuple[Any, Any]] = []

    async def on_start(self, context: Any, agent: Any) -> None:
        self.starts.append((context, agent))


@pytest.mark.usefixtures("_tracing_enabled")
async def test_chained_hooks_keep_capture_and_call_own_on_start() -> None:
    """`chain_agent_hooks(capture.hooks, own)` 合成後も捕捉が動き、own の `on_start` も呼ばれる。"""
    capture = context_capture.RunContextCapture()
    own = _OwnHooks()
    chained = chain_agent_hooks(capture.hooks, own)
    assert chained is not capture.hooks  # 2 件合成のため素通しではない

    wrapper = RunContextWrapper(context=_AppCtx(token="tk_chain"))
    agent = _agent()
    with trace("capture-test"), custom_span("turn"):
        await chained.on_start(wrapper, agent)
        assert capture.resolve() is wrapper

    assert len(own.starts) == 1
    seen_context, seen_agent = own.starts[0]
    assert seen_context is wrapper
    assert seen_agent is agent
