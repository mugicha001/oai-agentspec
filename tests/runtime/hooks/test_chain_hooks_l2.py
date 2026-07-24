"""L2: _adapters.hooks.chain_hooks の RunHooks 合成特性化テスト（#31 T3・RED 先行）。

`chain_hooks(*hooks)` が複数の `RunHooksBase` を宣言順に合成した単一の `RunHooksBase` を
返すことを検証する。具体的には (a) 戻り値が `RunHooksBase` インスタンスであること、
(b) 引数なし・単一引数の最適化（no-op / passthrough）、(c) 全 7 メソッドが宣言順に
await されること、(d) メソッド単位の fail-fast（中段 raise で後続 hook を呼ばない・例外を
そのまま伝播）、(e) 引数の無変更転送、(f) 合成メソッドが全て coroutine であることを検証する。

実装未完のため（`_adapters/hooks.py` が未追加）、本モジュールの import は `ImportError` と
なる（collection error = RED 状態が正しい）。
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest
from agents.lifecycle import RunHooksBase

from oai_agentspec._adapters.hooks import chain_hooks

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 全 7 メソッド名（宣言順の検証に使う）
# ---------------------------------------------------------------------------
_METHOD_NAMES = (
    "on_llm_start",
    "on_llm_end",
    "on_agent_start",
    "on_agent_end",
    "on_handoff",
    "on_tool_start",
    "on_tool_end",
)


# ---------------------------------------------------------------------------
# fake hooks
# ---------------------------------------------------------------------------
class RecordingHooks(RunHooksBase[Any, Any]):
    """全 7 メソッドの呼び出しを (hook_id, method_name, args) で共有リストへ記録する fake。"""

    def __init__(self, hook_id: str, calls: list[tuple[str, str, tuple[Any, ...]]]) -> None:
        super().__init__()
        self._id = hook_id
        self._calls = calls

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        self._calls.append((self._id, "on_llm_start", (context, agent, system_prompt, input_items)))

    async def on_llm_end(self, context, agent, response) -> None:
        self._calls.append((self._id, "on_llm_end", (context, agent, response)))

    async def on_agent_start(self, context, agent) -> None:
        self._calls.append((self._id, "on_agent_start", (context, agent)))

    async def on_agent_end(self, context, agent, output) -> None:
        self._calls.append((self._id, "on_agent_end", (context, agent, output)))

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        self._calls.append((self._id, "on_handoff", (context, from_agent, to_agent)))

    async def on_tool_start(self, context, agent, tool) -> None:
        self._calls.append((self._id, "on_tool_start", (context, agent, tool)))

    async def on_tool_end(self, context, agent, tool, result) -> None:
        self._calls.append((self._id, "on_tool_end", (context, agent, tool, result)))


class RaisingHooks(RunHooksBase[Any, Any]):
    """指定メソッドで指定例外を raise する fake（それ以外は共有リストへ記録する）。"""

    def __init__(
        self,
        hook_id: str,
        calls: list[tuple[str, str, tuple[Any, ...]]],
        raise_on: str,
        exc: BaseException,
    ) -> None:
        super().__init__()
        self._id = hook_id
        self._calls = calls
        self._raise_on = raise_on
        self._exc = exc

    def _dispatch(self, method_name: str, args: tuple[Any, ...]) -> None:
        self._calls.append((self._id, method_name, args))
        if method_name == self._raise_on:
            raise self._exc

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        self._dispatch("on_llm_start", (context, agent, system_prompt, input_items))

    async def on_llm_end(self, context, agent, response) -> None:
        self._dispatch("on_llm_end", (context, agent, response))

    async def on_agent_start(self, context, agent) -> None:
        self._dispatch("on_agent_start", (context, agent))

    async def on_agent_end(self, context, agent, output) -> None:
        self._dispatch("on_agent_end", (context, agent, output))

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        self._dispatch("on_handoff", (context, from_agent, to_agent))

    async def on_tool_start(self, context, agent, tool) -> None:
        self._dispatch("on_tool_start", (context, agent, tool))

    async def on_tool_end(self, context, agent, tool, result) -> None:
        self._dispatch("on_tool_end", (context, agent, tool, result))


# ---------------------------------------------------------------------------
# ヘルパ: 各メソッドをダミー引数で呼び出す
# ---------------------------------------------------------------------------
def _call_method(hooks: RunHooksBase, method_name: str, args: tuple[Any, ...]) -> None:
    """`hooks` の `method_name` を `args` で await 実行する。"""
    coro = getattr(hooks, method_name)(*args)
    asyncio.run(coro)


def _args_for(method_name: str) -> tuple[Any, ...]:
    """メソッドごとの引数個数に合わせたダミー sentinel タプルを返す。"""
    arities = {
        "on_llm_start": 4,
        "on_llm_end": 3,
        "on_agent_start": 2,
        "on_agent_end": 3,
        "on_handoff": 3,
        "on_tool_start": 3,
        "on_tool_end": 4,
    }
    return tuple(object() for _ in range(arities[method_name]))


# ---------------------------------------------------------------------------
# 1. 戻り値型
# ---------------------------------------------------------------------------
def test_chain_hooks_returns_run_hooks_base_instance() -> None:
    """通常引数（2 個の hook）で chain_hooks の戻り値が RunHooksBase インスタンスであること。"""
    calls: list[tuple[str, str, tuple[Any, ...]]] = []
    h1 = RecordingHooks("h1", calls)
    h2 = RecordingHooks("h2", calls)

    chained = chain_hooks(h1, h2)

    assert isinstance(chained, RunHooksBase)


# ---------------------------------------------------------------------------
# 2. 引数なし（no-op）
# ---------------------------------------------------------------------------
def test_chain_hooks_no_args_returns_noop_run_hooks_base() -> None:
    """chain_hooks() は素の RunHooksBase を返し、全 7 メソッドを呼んでも例外を出さないこと。"""
    chained = chain_hooks()

    assert isinstance(chained, RunHooksBase)
    for method_name in _METHOD_NAMES:
        _call_method(chained, method_name, _args_for(method_name))


# ---------------------------------------------------------------------------
# 3. 単一引数（passthrough）
# ---------------------------------------------------------------------------
def test_chain_hooks_single_arg_returns_same_instance() -> None:
    """chain_hooks(single) は single そのものを返す（is 一致・単一時の最適化）。"""
    calls: list[tuple[str, str, tuple[Any, ...]]] = []
    single = RecordingHooks("single", calls)

    chained = chain_hooks(single)

    assert chained is single


# ---------------------------------------------------------------------------
# 4. 宣言順に全 7 メソッドを転送
# ---------------------------------------------------------------------------
def test_chain_hooks_forwards_all_seven_methods_in_declared_order() -> None:
    """3 個の hook で全 7 メソッドが宣言順（h1 -> h2 -> h3）に await されること。"""
    calls: list[tuple[str, str, tuple[Any, ...]]] = []
    h1 = RecordingHooks("h1", calls)
    h2 = RecordingHooks("h2", calls)
    h3 = RecordingHooks("h3", calls)

    chained = chain_hooks(h1, h2, h3)

    for method_name in _METHOD_NAMES:
        calls.clear()
        _call_method(chained, method_name, _args_for(method_name))
        assert [(hook_id, name) for hook_id, name, _ in calls] == [
            ("h1", method_name),
            ("h2", method_name),
            ("h3", method_name),
        ]


# ---------------------------------------------------------------------------
# 5. メソッド単位の fail-fast
# ---------------------------------------------------------------------------
def test_chain_hooks_stops_forwarding_when_hook_raises() -> None:
    """中段 h2 が on_llm_end で raise すると h3 の同メソッドは呼ばれず例外が伝播する。

    fail-fast はメソッド単位であり、raise しない別メソッド（on_llm_start）では h1/h2/h3
    すべてが呼ばれることも併せて検証する。
    """
    calls: list[tuple[str, str, tuple[Any, ...]]] = []
    h1 = RecordingHooks("h1", calls)
    h2 = RaisingHooks("h2", calls, raise_on="on_llm_end", exc=RuntimeError("stop"))
    h3 = RecordingHooks("h3", calls)

    chained = chain_hooks(h1, h2, h3)

    # raise するメソッド: h3 は呼ばれず例外が伝播する
    calls.clear()
    with pytest.raises(RuntimeError):
        _call_method(chained, "on_llm_end", _args_for("on_llm_end"))
    assert [(hook_id, name) for hook_id, name, _ in calls] == [
        ("h1", "on_llm_end"),
        ("h2", "on_llm_end"),
    ]

    # raise しない別メソッド: h1/h2/h3 すべて呼ばれる
    calls.clear()
    _call_method(chained, "on_llm_start", _args_for("on_llm_start"))
    assert [(hook_id, name) for hook_id, name, _ in calls] == [
        ("h1", "on_llm_start"),
        ("h2", "on_llm_start"),
        ("h3", "on_llm_start"),
    ]


# ---------------------------------------------------------------------------
# 6. 例外の型・メッセージ保存
# ---------------------------------------------------------------------------
def test_chain_hooks_preserves_exception_type_and_message() -> None:
    """中段 hook が ValueError("boom") を raise した際、型とメッセージがそのまま伝播すること。"""
    calls: list[tuple[str, str, tuple[Any, ...]]] = []
    h1 = RecordingHooks("h1", calls)
    h2 = RaisingHooks("h2", calls, raise_on="on_agent_start", exc=ValueError("boom"))

    chained = chain_hooks(h1, h2)

    with pytest.raises(ValueError, match="boom"):
        _call_method(chained, "on_agent_start", _args_for("on_agent_start"))


# ---------------------------------------------------------------------------
# 7. 引数の無変更転送
# ---------------------------------------------------------------------------
def test_chain_hooks_passes_arguments_unchanged_to_all_hooks() -> None:
    """各メソッドの引数（context / agent / 追加引数）が変更されず全 hook に渡ること。"""
    calls: list[tuple[str, str, tuple[Any, ...]]] = []
    h1 = RecordingHooks("h1", calls)
    h2 = RecordingHooks("h2", calls)

    chained = chain_hooks(h1, h2)

    for method_name in _METHOD_NAMES:
        calls.clear()
        args = _args_for(method_name)
        _call_method(chained, method_name, args)
        for _hook_id, _name, recorded_args in calls:
            assert len(recorded_args) == len(args)
            for expected, actual in zip(args, recorded_args, strict=True):
                assert actual is expected


# ---------------------------------------------------------------------------
# 8. 全メソッドが coroutine
# ---------------------------------------------------------------------------
def test_chained_hooks_overrides_match_sdk_on_methods() -> None:
    """`_ChainedHooks` は SDK `RunHooksBase` の全 on_* メソッドをオーバーライドする（fitness）。

    SDK バージョン更新で `RunHooksBase` に新規 hook メソッドが追加された場合、`_ChainedHooks`
    でオーバーライド漏れがあると新メソッドは合成対象から抜け silent gap になる（module docstring
    「SDK 追随手順」の機械化）。合成インスタンスの型階層に本モジュールの `_ChainedHooks` が
    含まれる想定で、`RunHooksBase` 側の on_* シンボル集合と `_ChainedHooks.__dict__` 側の
    on_* シンボル集合を突き合わせる。
    """
    calls: list[tuple[str, str, tuple[Any, ...]]] = []
    h1 = RecordingHooks("h1", calls)
    h2 = RecordingHooks("h2", calls)
    chained_cls = type(chain_hooks(h1, h2))

    sdk_on_methods = {name for name in vars(RunHooksBase) if name.startswith("on_")}
    chained_on_overrides = {name for name in vars(chained_cls) if name.startswith("on_")}
    missing = sdk_on_methods - chained_on_overrides
    assert missing == set(), (
        f"SDK RunHooksBase の on_* メソッド {sorted(missing)} が _ChainedHooks で "
        "オーバーライドされていません。_adapters/hooks.py の _ChainedHooks に同名の "
        "async オーバーライドを追加してください（module docstring「SDK 追随手順」参照）。"
    )


def test_chain_hooks_all_methods_are_async() -> None:
    """合成インスタンスの 7 メソッドは全て coroutine function であること（戻り値経由で確認）。"""
    calls: list[tuple[str, str, tuple[Any, ...]]] = []
    h1 = RecordingHooks("h1", calls)
    h2 = RecordingHooks("h2", calls)

    chained = chain_hooks(h1, h2)

    for method_name in _METHOD_NAMES:
        method = getattr(chained, method_name)
        assert inspect.iscoroutinefunction(method), method_name
