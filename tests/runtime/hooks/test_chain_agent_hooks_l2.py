"""L2: _adapters.hooks.chain_agent_hooks の AgentHooks 合成特性化テスト。

`chain_agent_hooks(*hooks)` が複数の agent 単位フックを宣言順に合成した単一の
`AgentHooksBase` を返すことを検証する。具体的には (a) 戻り値型と縮退仕様（0 件 / 全 `None` /
1 件・`isinstance` 成立 / 1 件・部分実装 / 2 件以上）、(b) `None` 混在時の除外と宣言順維持、
(c) 全 7 メソッドの宣言順転送と引数の無変更転送、(d) duck-typed 委譲（sync / async /
メソッド欠如 / 非 callable 属性）、(e) メソッド単位の fail-fast、(f) SDK パリティ tripwire、
(g) run 単位クラスとの非混同（MRO・`on_handoff` の引数構成）、(h) 呼び出し回数上限（NFR-5）、
(i) 元引数列の非破壊、(j) 公開シグネチャ、(k) `AgentSpec(hooks=...)` の build 素通しを検証する。

委譲ヘルパー `_delegate_agent_hook` は原則 `chain_agent_hooks` 経由で検証するが、`None` 安全の
分岐のみ合成経路から到達しないため単体で 1 件検証する。
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from agents import Agent
from agents.lifecycle import AgentHooksBase, RunHooksBase

from oai_agentspec._adapters import build_agent
from oai_agentspec._adapters.hooks import (
    _AGENT_HOOK_METHOD_NAMES,
    _ChainedHooks,
    _delegate_agent_hook,
    chain_agent_hooks,
)
from oai_agentspec.spec import AgentSpec

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 全 7 メソッド名（SDK `AgentHooksBase` の宣言順）と引数個数
# ---------------------------------------------------------------------------
_METHOD_NAMES = (
    "on_start",
    "on_end",
    "on_handoff",
    "on_tool_start",
    "on_tool_end",
    "on_llm_start",
    "on_llm_end",
)

_ARITIES = {
    "on_start": 2,
    "on_end": 3,
    "on_handoff": 3,
    "on_tool_start": 3,
    "on_tool_end": 4,
    "on_llm_start": 4,
    "on_llm_end": 3,
}

# 記録形式: (hook_id, method_name, args, kwargs)
_Call = tuple[str, str, tuple[Any, ...], dict[str, Any]]


# ---------------------------------------------------------------------------
# fake hooks（`AgentHooksBase` 継承）
# ---------------------------------------------------------------------------
class RecordingHooks(AgentHooksBase[Any, Any]):
    """全 7 メソッドの呼び出しを (hook_id, method_name, args, kwargs) で共有リストへ記録する fake。

    引数を `*args, **kwargs` で受けるのは、転送された位置引数の個数・順序・同一性に加えて
    「キーワード引数が使われていないこと」まで 1 つの比較で照合するため。
    """

    def __init__(self, hook_id: str, calls: list[_Call]) -> None:
        super().__init__()
        self._id = hook_id
        self._calls = calls

    def _record(self, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self._calls.append((self._id, method_name, args, kwargs))

    async def on_start(self, *args: Any, **kwargs: Any) -> None:
        self._record("on_start", args, kwargs)

    async def on_end(self, *args: Any, **kwargs: Any) -> None:
        self._record("on_end", args, kwargs)

    async def on_handoff(self, *args: Any, **kwargs: Any) -> None:
        self._record("on_handoff", args, kwargs)

    async def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        self._record("on_tool_start", args, kwargs)

    async def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
        self._record("on_tool_end", args, kwargs)

    async def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self._record("on_llm_start", args, kwargs)

    async def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        self._record("on_llm_end", args, kwargs)


class RaisingHooks(RecordingHooks):
    """指定メソッドで指定例外を raise する fake（記録は行ってから raise する）。"""

    def __init__(
        self,
        hook_id: str,
        calls: list[_Call],
        raise_on: str,
        exc: BaseException,
    ) -> None:
        super().__init__(hook_id, calls)
        self._raise_on = raise_on
        self._exc = exc

    def _record(self, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        super()._record(method_name, args, kwargs)
        if method_name == self._raise_on:
            raise self._exc


# ---------------------------------------------------------------------------
# fake hooks（duck-typed・`AgentHooksBase` 非継承）
# ---------------------------------------------------------------------------
class SyncDuckHooks:
    """`on_start` のみを同期メソッドで持つ部分実装（await されない経路の検証用）。"""

    def __init__(self, hook_id: str, calls: list[_Call]) -> None:
        self._id = hook_id
        self._calls = calls

    def on_start(self, *args: Any, **kwargs: Any) -> None:
        self._calls.append((self._id, "on_start", args, kwargs))


class AsyncDuckHooks:
    """`on_end` のみを非同期メソッドで持つ部分実装（await される経路の検証用）。"""

    def __init__(self, hook_id: str, calls: list[_Call]) -> None:
        self._id = hook_id
        self._calls = calls

    async def on_end(self, *args: Any, **kwargs: Any) -> None:
        self._calls.append((self._id, "on_end", args, kwargs))


class EmptyDuckHooks:
    """`on_*` を一切持たない部分実装（ADR-0017 の build 時拒否の検証用）。"""


class OtherMethodDuckHooks:
    """対象メソッド（`on_start`）を持たず別の `on_*` のみを持つ部分実装（skip 経路の検証用）。

    `on_*` を 1 つも持たない要素は build 時に拒否されるため、skip 経路の検証には「1 つは持つが
    対象メソッドは持たない」形を使う。
    """

    async def on_end(self, context: Any, agent: Any, output: Any) -> None:
        """終了のみ受ける（`on_start` は持たない）。"""


class NonCallableAttrHooks:
    """`on_start` が callable でない属性である部分実装（TypeError 伝播の検証用）。"""

    on_start = 1


async def _noop() -> None:
    """awaitable 判定経路を通すための no-op coroutine を返す。"""
    return None


class CountingHooks(AgentHooksBase[Any, Any]):
    """`on_start` の呼び出し回数を**同期部**で数える fake（`AgentHooksBase` 継承版）。

    `async def` の本体でカウントすると「呼んだが await しない」二重呼び出しを検知できない
    （本体は await 時に初めて実行されるため）。呼び出し直後に走る同期部で数え、awaitable
    判定・`await` の経路も通るよう coroutine を返す。
    """

    def __init__(self, hook_id: str, counts: dict[str, int]) -> None:
        super().__init__()
        self._id = hook_id
        self._counts = counts

    def on_start(self, *args: Any, **kwargs: Any) -> Any:
        self._counts[self._id] = self._counts.get(self._id, 0) + 1
        return _noop()


class CountingDuckHooks:
    """`on_start` の呼び出し回数を同期部で数える部分実装（duck-typed 版）。"""

    def __init__(self, hook_id: str, counts: dict[str, int]) -> None:
        self._id = hook_id
        self._counts = counts

    def on_start(self, *args: Any, **kwargs: Any) -> Any:
        self._counts[self._id] = self._counts.get(self._id, 0) + 1
        return _noop()


# ---------------------------------------------------------------------------
# ヘルパ
# ---------------------------------------------------------------------------
async def _call_method(hooks: Any, method_name: str, args: tuple[Any, ...]) -> None:
    """`hooks` の `method_name` を `args` で await 実行する。"""
    await getattr(hooks, method_name)(*args)


def _args_for(method_name: str) -> tuple[Any, ...]:
    """メソッドごとの引数個数に合わせたダミー sentinel タプルを返す。"""
    return tuple(object() for _ in range(_ARITIES[method_name]))


# ---------------------------------------------------------------------------
# 観点 1. 戻り値型
# ---------------------------------------------------------------------------
def test_chain_agent_hooks_returns_agent_hooks_base_instance() -> None:
    """2 件合成の戻り値が `AgentHooksBase` インスタンスであること（AgentSpec.hooks 適合）。"""
    calls: list[_Call] = []
    chained = chain_agent_hooks(RecordingHooks("h1", calls), RecordingHooks("h2", calls))

    assert isinstance(chained, AgentHooksBase)


# ---------------------------------------------------------------------------
# 観点 2. 縮退（0 件）
# ---------------------------------------------------------------------------
async def test_chain_agent_hooks_no_args_returns_plain_agent_hooks_base() -> None:
    """`chain_agent_hooks()` は素の `AgentHooksBase` を返し全 7 メソッドが no-op であること。"""
    chained = chain_agent_hooks()

    assert type(chained) is AgentHooksBase
    for method_name in _METHOD_NAMES:
        await _call_method(chained, method_name, _args_for(method_name))


# ---------------------------------------------------------------------------
# 観点 3. 縮退（全 None）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hooks", [(None,), (None, None)])
async def test_chain_agent_hooks_all_none_returns_plain_agent_hooks_base(
    hooks: tuple[Any, ...],
) -> None:
    """`None` のみを渡しても素の `AgentHooksBase`（`None` を返さない・例外も出さない）。"""
    chained = chain_agent_hooks(*hooks)

    assert type(chained) is AgentHooksBase
    for method_name in _METHOD_NAMES:
        await _call_method(chained, method_name, _args_for(method_name))


# ---------------------------------------------------------------------------
# 観点 4. 縮退（1 件・isinstance 成立）
# ---------------------------------------------------------------------------
def test_chain_agent_hooks_single_instance_returns_same_object() -> None:
    """実効 1 件かつ `AgentHooksBase` インスタンスなら引数自身を返す（`is` 一致・ラッパ非生成）。

    `None` を前後に混ぜた `(a, None)` / `(None, a)` でも同じ縮退が働くことを併せて固定する。
    値等価では複製でも通るため、照合は `is` で行う。
    """
    calls: list[_Call] = []
    single = RecordingHooks("single", calls)

    assert chain_agent_hooks(single) is single
    assert chain_agent_hooks(single, None) is single
    assert chain_agent_hooks(None, single) is single


# ---------------------------------------------------------------------------
# 観点 5. 縮退（1 件・非 isinstance の部分実装）
# ---------------------------------------------------------------------------
def test_chain_agent_hooks_single_duck_typed_is_wrapped() -> None:
    """実効 1 件でも非 `AgentHooksBase` の部分実装はラッパで包み、`isinstance` を満たすこと。"""
    calls: list[_Call] = []
    duck = SyncDuckHooks("duck", calls)

    chained = chain_agent_hooks(duck)

    assert chained is not duck
    assert isinstance(chained, AgentHooksBase)


# ---------------------------------------------------------------------------
# 観点 6. None 混在時の除外と宣言順維持
# ---------------------------------------------------------------------------
async def test_chain_agent_hooks_skips_none_and_keeps_declared_order() -> None:
    """`(None, a, None, b)` は `None` を除外し宣言順（a -> b）で転送すること。"""
    calls: list[_Call] = []
    h1 = RecordingHooks("h1", calls)
    h2 = RecordingHooks("h2", calls)

    chained = chain_agent_hooks(None, h1, None, h2)

    await _call_method(chained, "on_start", _args_for("on_start"))
    assert [(hook_id, name) for hook_id, name, _args, _kwargs in calls] == [
        ("h1", "on_start"),
        ("h2", "on_start"),
    ]


# ---------------------------------------------------------------------------
# 観点 7. 全 7 メソッドの宣言順転送 + 引数の無変更転送
# ---------------------------------------------------------------------------
async def test_chain_agent_hooks_forwards_all_seven_methods_in_declared_order() -> None:
    """全 7 メソッドが宣言順（h1 -> h2 -> h3）に呼ばれ、引数が無変更で転送されること。

    記録した `(args, kwargs)` を期待値と全体照合する（sentinel は既定の同一性比較のため、
    等価比較がそのまま `is` 一致・個数・順序の検証になる）。回数と順序だけでは「引数を捨てて
    別の値で呼ぶ」変異を検知できないため、引数列ごと固定する。
    """
    calls: list[_Call] = []
    h1 = RecordingHooks("h1", calls)
    h2 = RecordingHooks("h2", calls)
    h3 = RecordingHooks("h3", calls)

    chained = chain_agent_hooks(h1, h2, h3)

    for method_name in _METHOD_NAMES:
        calls.clear()
        args = _args_for(method_name)
        await _call_method(chained, method_name, args)
        assert calls == [
            ("h1", method_name, args, {}),
            ("h2", method_name, args, {}),
            ("h3", method_name, args, {}),
        ]


# ---------------------------------------------------------------------------
# 観点 8. duck-typed 委譲（sync / async / None 要素 / メソッド欠如）
# ---------------------------------------------------------------------------
async def test_chain_agent_hooks_calls_sync_duck_typed_method() -> None:
    """部分実装の同期メソッドはそのまま呼ばれる（await 不要・引数素通し）。"""
    calls: list[_Call] = []
    chained = chain_agent_hooks(SyncDuckHooks("sync", calls), RecordingHooks("h1", calls))

    ctx, agent = object(), object()
    await chained.on_start(ctx, agent)

    assert calls == [
        ("sync", "on_start", (ctx, agent), {}),
        ("h1", "on_start", (ctx, agent), {}),
    ]


async def test_chain_agent_hooks_awaits_async_duck_typed_method() -> None:
    """部分実装の非同期メソッドは `await` されて完了すること。"""
    calls: list[_Call] = []
    chained = chain_agent_hooks(AsyncDuckHooks("async", calls), RecordingHooks("h1", calls))

    ctx, agent, output = object(), object(), object()
    await chained.on_end(ctx, agent, output)

    assert calls == [
        ("async", "on_end", (ctx, agent, output), {}),
        ("h1", "on_end", (ctx, agent, output), {}),
    ]


async def test_chain_agent_hooks_ignores_none_element_at_call_time() -> None:
    """`None` 要素は合成対象から除かれ、呼び出し時に何も起こらない（例外なし）。"""
    calls: list[_Call] = []
    chained = chain_agent_hooks(None, SyncDuckHooks("sync", calls), None)

    ctx, agent = object(), object()
    await chained.on_start(ctx, agent)

    assert calls == [("sync", "on_start", (ctx, agent), {})]


async def test_chain_agent_hooks_skips_element_without_target_method() -> None:
    """対象メソッドを持たない要素は skip され `AttributeError` を送出しないこと。"""
    calls: list[_Call] = []
    chained = chain_agent_hooks(OtherMethodDuckHooks(), SyncDuckHooks("sync", calls))

    ctx, agent = object(), object()
    await chained.on_start(ctx, agent)

    assert calls == [("sync", "on_start", (ctx, agent), {})]


async def test_delegate_agent_hook_target_none_is_noop() -> None:
    """委譲ヘルパーは委譲先が `None` でも何もしない（例外なし）。

    `chain_agent_hooks` は合成時に `None` を除外するため、この分岐は本番経路（合成ラッパ）
    からは到達しない。それでも `_delegate_agent_hook` 単体の契約（`None` 安全）として
    維持されるため、ヘルパーを直接呼んで pin する。
    """
    await _delegate_agent_hook(None, "on_start", "ctx", "agent")


# ---------------------------------------------------------------------------
# 観点 9. 非 callable 属性は TypeError をそのまま伝播
# ---------------------------------------------------------------------------
async def test_chain_agent_hooks_propagates_type_error_for_non_callable_attribute() -> None:
    """同名属性が callable でない（`on_start = 1`）場合は `TypeError` がそのまま伝播すること。

    委譲側で callable 判定を行わない（現行 `governance._delegate` と同一挙動）ことの pin。
    """
    chained = chain_agent_hooks(NonCallableAttrHooks())

    with pytest.raises(TypeError, match="not callable"):
        await chained.on_start(object(), object())


# ---------------------------------------------------------------------------
# 観点 10. メソッド単位の fail-fast と例外の保存
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raising_method", _METHOD_NAMES)
async def test_chain_agent_hooks_stops_forwarding_when_element_raises(raising_method: str) -> None:
    """中段が raise すると後段の同メソッドは 0 回呼ばれ、例外が伝播すること（7 メソッド全件）。

    fail-fast はメソッド単位である。7 メソッドそれぞれについて (a) 当該メソッドでは h3 が
    呼ばれず例外が伝播する、(b) raise しない別メソッドでは h1/h2/h3 すべてが呼ばれる、を
    検証する。個別メソッドに `try` / `except` を入れる退行を全メソッドで検知するため
    parametrize する（3 メソッドのみを対象にしていると残り 4 メソッドの変異が生き残る）。
    """
    other_method = next(name for name in _METHOD_NAMES if name != raising_method)

    calls: list[_Call] = []
    h1 = RecordingHooks("h1", calls)
    h2 = RaisingHooks("h2", calls, raise_on=raising_method, exc=RuntimeError("stop"))
    h3 = RecordingHooks("h3", calls)

    chained = chain_agent_hooks(h1, h2, h3)

    calls.clear()
    with pytest.raises(RuntimeError):
        await _call_method(chained, raising_method, _args_for(raising_method))
    assert [(hook_id, name) for hook_id, name, _args, _kwargs in calls] == [
        ("h1", raising_method),
        ("h2", raising_method),
    ]

    calls.clear()
    await _call_method(chained, other_method, _args_for(other_method))
    assert [(hook_id, name) for hook_id, name, _args, _kwargs in calls] == [
        ("h1", other_method),
        ("h2", other_method),
        ("h3", other_method),
    ]


async def test_chain_agent_hooks_preserves_exception_type_and_message() -> None:
    """中段が `ValueError("boom")` を raise した際、型とメッセージがそのまま伝播すること。"""
    calls: list[_Call] = []
    h1 = RecordingHooks("h1", calls)
    h2 = RaisingHooks("h2", calls, raise_on="on_tool_start", exc=ValueError("boom"))

    chained = chain_agent_hooks(h1, h2)

    with pytest.raises(ValueError, match="boom"):
        await _call_method(chained, "on_tool_start", _args_for("on_tool_start"))


# ---------------------------------------------------------------------------
# 観点 11. SDK パリティ tripwire
# ---------------------------------------------------------------------------
def test_chained_agent_hooks_overrides_match_sdk_on_methods() -> None:
    """合成ラッパは SDK `AgentHooksBase` の全 on_* メソッドをオーバーライドする（fitness）。

    SDK バージョン更新で `AgentHooksBase` に新規 hook メソッドが追加された場合、合成ラッパで
    オーバーライド漏れがあると新メソッドは合成対象から抜け silent gap になる（`_adapters/
    hooks.py` の module docstring「SDK 追随手順」の機械化）。`AgentHooksBase` 側の on_*
    シンボル集合から合成ラッパのクラスの `vars()` 上の on_* 集合を引いた差集合が空であることを
    主張する（run 単位 `test_chained_hooks_overrides_match_sdk_on_methods` と同一方向・
    同一比較方式）。

    比較の空振り（`vars()` の取得側が壊れて空集合になり差集合が自明に空になる）を防ぐため、
    SDK 側集合が既知の 7 メソッドを下限として含むことも併せて確認する（等価比較ではないため
    SDK への新規追加は差集合側で検知される）。

    本テストの nodeid は `docs/QUALITY-GUARANTEES.md` が強制手段として参照するため、
    テスト関数名を変更しないこと。
    """
    calls: list[_Call] = []
    h1 = RecordingHooks("h1", calls)
    h2 = RecordingHooks("h2", calls)
    chained_cls = type(chain_agent_hooks(h1, h2))

    sdk_on_methods = {name for name in vars(AgentHooksBase) if name.startswith("on_")}
    assert set(_METHOD_NAMES) <= sdk_on_methods

    chained_on_overrides = {name for name in vars(chained_cls) if name.startswith("on_")}
    missing = sdk_on_methods - chained_on_overrides
    assert missing == set(), (
        f"SDK AgentHooksBase の on_* メソッド {sorted(missing)} が合成ラッパで "
        "オーバーライドされていません。_adapters/hooks.py の _ChainedAgentHooks に同名の "
        "async オーバーライドを追加してください（module docstring「SDK 追随手順」参照）。"
    )


# ---------------------------------------------------------------------------
# 観点 12. on_handoff は (context, agent, source) の 3 引数
# ---------------------------------------------------------------------------
async def test_chain_agent_hooks_forwards_three_handoff_arguments() -> None:
    """`on_handoff` は `(context, agent, source)` の 3 引数を各要素へ転送すること。

    run 単位 `RunHooksBase.on_handoff` の `(context, from_agent, to_agent)` と引数構成が
    異なるため、agent 単位の 3 引数転送を独立に固定する。
    """
    calls: list[_Call] = []
    h1 = RecordingHooks("h1", calls)
    h2 = RecordingHooks("h2", calls)

    chained = chain_agent_hooks(h1, h2)

    context, agent, source = object(), object(), object()
    await chained.on_handoff(context, agent, source)

    assert calls == [
        ("h1", "on_handoff", (context, agent, source), {}),
        ("h2", "on_handoff", (context, agent, source), {}),
    ]


# ---------------------------------------------------------------------------
# 観点 13. MRO に run 単位クラスを含まない
# ---------------------------------------------------------------------------
def test_chained_agent_hooks_mro_excludes_run_scope_chained_hooks() -> None:
    """合成ラッパのクラスの MRO に run 単位クラス `_ChainedHooks` を含まないこと。"""
    calls: list[_Call] = []
    chained_cls = type(chain_agent_hooks(RecordingHooks("h1", calls), RecordingHooks("h2", calls)))

    assert _ChainedHooks not in chained_cls.__mro__


# ---------------------------------------------------------------------------
# 観点 14. 呼び出し回数の上限（NFR-5）
# ---------------------------------------------------------------------------
async def test_chain_agent_hooks_calls_each_element_at_most_once() -> None:
    """要素数 N の合成で 1 回の `on_start` 呼び出しにつき各要素の呼び出しは 1 回以下であること。

    カウントは fake の同期部で行う（coroutine 本体で数えると `await` 回数の計測になり、
    「呼んで await しない」二重呼び出しを検知できない）。対象メソッドを持たない要素は 0 回。
    """
    counts: dict[str, int] = {}
    h1 = CountingHooks("h1", counts)
    h2 = CountingDuckHooks("h2", counts)
    h3 = OtherMethodDuckHooks()

    chained = chain_agent_hooks(h1, h2, h3)

    await chained.on_start(object(), object())

    assert counts == {"h1": 1, "h2": 1}
    assert sum(counts.values()) <= 3


# ---------------------------------------------------------------------------
# 観点 15. 元引数列の非破壊
# ---------------------------------------------------------------------------
def test_chain_agent_hooks_does_not_mutate_argument_sequence() -> None:
    """合成後も呼び出し側が渡した列（`None` 要素を含む）と各オブジェクトが変わらないこと。"""
    calls: list[_Call] = []
    h1 = RecordingHooks("h1", calls)
    h2 = RecordingHooks("h2", calls)
    original = [None, h1, None, h2]
    snapshot = list(original)

    chain_agent_hooks(*original)

    assert len(original) == len(snapshot)
    assert all(actual is expected for actual, expected in zip(original, snapshot, strict=True))
    # 合成（build 時）には要素のメソッドを呼ばない。
    assert calls == []


# ---------------------------------------------------------------------------
# 観点 16. 公開シグネチャの pin
# ---------------------------------------------------------------------------
def test_chain_agent_hooks_signature_is_var_positional_only() -> None:
    """`chain_agent_hooks` は可変長位置引数 1 つのみでキーワード専用引数を持たないこと。"""
    params = list(inspect.signature(chain_agent_hooks).parameters.values())

    assert len(params) == 1
    assert params[0].kind is inspect.Parameter.VAR_POSITIONAL
    assert [p.name for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY] == []
    assert [p.name for p in params if p.kind is inspect.Parameter.VAR_KEYWORD] == []


# ---------------------------------------------------------------------------
# 観点 17. AgentSpec(hooks=...) の build 素通し
# ---------------------------------------------------------------------------
def test_chain_agent_hooks_result_passes_through_agent_spec_build() -> None:
    """`AgentSpec(hooks=chain_agent_hooks(...))` の合成結果が `agent.hooks` へ素通しされること。

    利用者が追加の変換なしに合成結果を宣言へ渡せること（FR-1）の pin。複製でなく同一
    オブジェクトが渡ることを `is` で照合する。
    """
    calls: list[_Call] = []
    chained = chain_agent_hooks(RecordingHooks("h1", calls), SyncDuckHooks("duck", calls))
    spec = AgentSpec(name="bot", instructions="i", hooks=chained)

    agent = build_agent(spec)

    assert isinstance(agent, Agent)
    assert agent.hooks is chained


# ---------------------------------------------------------------------------
# 観点 18. run 単位フックの拒否（ADR-0017）
# ---------------------------------------------------------------------------
class _RunScopeHooks(RunHooksBase[Any, Any]):
    """run 単位フック。agent スロットへ渡すと誤ルートされるため拒否対象。"""

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        """run 単位の開始通知（agent 単位の `on_start` とは別名）。"""


class _BothScopeHooks(AgentHooksBase[Any, Any], RunHooksBase[Any, Any]):
    """両スコープを継承したフック。`on_handoff` の引数意味が一意に決まらないため拒否対象。"""


def test_chain_agent_hooks_rejects_run_scope_hook() -> None:
    """run 単位フック単体を渡すと `TypeError` で拒否されること。

    合成ラッパへ包まれると SDK の型チェックを通過してしまい、`on_start` / `on_end` が
    silent skip され `on_handoff` は from/to が反転する（例外なしの誤記録）。build 時に
    落とすことで構造的に防ぐ（ADR-0017）。
    """
    with pytest.raises(TypeError):
        chain_agent_hooks(_RunScopeHooks())


def test_chain_agent_hooks_rejects_run_scope_hook_mixed_with_agent_scope() -> None:
    """agent 単位フックと混在していても run 単位フックがあれば拒否されること。"""
    calls: list[_Call] = []

    with pytest.raises(TypeError):
        chain_agent_hooks(RecordingHooks("h1", calls), _RunScopeHooks())


def test_chain_agent_hooks_rejects_run_scope_hook_with_none_present() -> None:
    """`None` を併記していても拒否され、案内する位置が利用者の引数位置と一致すること。

    検証は `None` 除外の前に行う。除外後の位置で数えると `None` 混在時に誤った位置を案内する
    （`chain_agent_hooks(None, None, run_hook)` が `hooks[0]` になってしまう）ため、この pin で
    引数位置との一致を固定する。
    """
    with pytest.raises(TypeError) as excinfo:
        chain_agent_hooks(None, None, _RunScopeHooks())

    assert "hooks[2]" in str(excinfo.value)


def test_chain_agent_hooks_rejects_hook_inheriting_both_scopes() -> None:
    """両スコープを継承したフックも拒否されること（`AgentHooksBase` 判定より前に落とす）。"""
    with pytest.raises(TypeError):
        chain_agent_hooks(_BothScopeHooks())


def test_chain_agent_hooks_rejection_message_locates_offending_argument() -> None:
    """拒否メッセージに引数位置・クラス名・`chain_hooks` への誘導が含まれること。"""
    calls: list[_Call] = []

    with pytest.raises(TypeError) as excinfo:
        chain_agent_hooks(RecordingHooks("h1", calls), _RunScopeHooks())

    message = str(excinfo.value)
    assert "hooks[1]" in message
    assert "_RunScopeHooks" in message
    assert "chain_hooks" in message


def test_chain_agent_hooks_none_only_still_returns_plain_instance() -> None:
    """`None` のみの呼び出しは拒否検証の追加後も従来どおり素インスタンスを返すこと（回帰）。"""
    result = chain_agent_hooks(None)

    assert type(result) is AgentHooksBase


# ---------------------------------------------------------------------------
# 観点 19. `on_*` を 1 つも持たない要素の拒否（ADR-0017）
# ---------------------------------------------------------------------------
def test_chain_agent_hooks_rejects_element_without_any_on_method() -> None:
    """`on_*` を 1 つも持たない要素は `TypeError` で拒否されること。

    包んでも全メソッドが skip されるため、フックが 1 つも発火しない no-op が例外なく成立して
    しまう（silent gap）。build 時に落とすことで検知可能にする。
    """
    with pytest.raises(TypeError):
        chain_agent_hooks(EmptyDuckHooks())


def test_chain_agent_hooks_rejects_unpack_omission() -> None:
    """`*` の付け忘れ（list をそのまま渡す）が拒否されること。

    `chain_agent_hooks([h1, h2])` は list 自体が要素になり `on_*` を持たないため、従来は
    全フックが無音で失われていた。実際に起きやすい誤用なので個別に pin する。
    """
    calls: list[_Call] = []
    hooks = [RecordingHooks("h1", calls), RecordingHooks("h2", calls)]

    with pytest.raises(TypeError):
        chain_agent_hooks(hooks)  # type: ignore[arg-type]


def test_chain_agent_hooks_rejects_string_element() -> None:
    """typo 等で文字列が混入した場合に拒否されること。"""
    with pytest.raises(TypeError):
        chain_agent_hooks("typo")


def test_chain_agent_hooks_rejection_message_names_missing_on_methods() -> None:
    """メッセージに引数位置・型名と、`on_*` を持たない旨が含まれること。"""
    with pytest.raises(TypeError) as excinfo:
        chain_agent_hooks(None, "typo")

    message = str(excinfo.value)
    assert "hooks[1]" in message
    assert "str" in message
    assert "on_" in message


def test_agent_hook_method_names_match_sdk_lifecycle_methods() -> None:
    """検証に使うメソッド名集合が SDK `AgentHooksBase` の `on_*` と一致すること。

    `_AGENT_HOOK_METHOD_NAMES` が空になると `on_*` を持つ正当な要素まで全て拒否され、多数の
    テストが一斉に落ちて原因が見えにくくなる（実測 35 件）。この pin が 1 件で導出の破綻を
    指し示す。`dir()` を使うのは、SDK がメソッドを中間基底クラスへ移した場合も継承経由で拾い
    導出が空にならないため（`vars()` は当該クラスの `__dict__` のみを見る）。
    """
    expected = {name for name in dir(AgentHooksBase) if name.startswith("on_")}

    assert set(_AGENT_HOOK_METHOD_NAMES) == expected
    assert len(_AGENT_HOOK_METHOD_NAMES) == 7, (
        f"SDK AgentHooksBase の on_* メソッド数が 7 から変わりました: "
        f"{sorted(_AGENT_HOOK_METHOD_NAMES)}。`_ChainedAgentHooks` のオーバーライドと "
        f"docs の記述（7 メソッド）も追随させてください。"
    )


def test_chain_agent_hooks_accepts_element_with_single_on_method() -> None:
    """`on_*` を 1 つでも持つ部分実装は従来どおり受理されること（回帰）。

    拒否条件を「1 つも持たない」に限定していることの pin。部分実装サポート（FR-3）を
    壊していないことを保証する。
    """

    class _OnlyOnStart:
        async def on_start(self, context: Any, agent: Any) -> None:
            """開始のみ受ける。"""

    result = chain_agent_hooks(_OnlyOnStart())

    assert isinstance(result, AgentHooksBase)
