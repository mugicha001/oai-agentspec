"""L2: `canary_guardrail` の resolver 経路（run ごとのトークン再解決）の SDK 結合検証。

`canary` に callable（resolver）を渡した場合、構築時には評価されず、検知呼び出しごとに
`(context, agent)` で再評価してそのトークンで逐語照合することを検証する。固定値経路
（`str` / `Iterable[str]`）の互換、`None` / 空値の非発火、構築時の arity 検証、プレフィクス
一致へ劣化していないこと（逐語照合）、および `canary_detector` の既存 1 引数契約の不変も pin する。
SDK の `OutputGuardrail.run` 越しに駆動する（`context` / `agent` は SDK と同じ経路で届く）。

resolver の戻り値型の制約（`str` / `Iterable[str]` / `None` 以外は検知時 `TypeError`）と、async
resolver の構築時拒否（公開契約は同期のみ・ADR 0023 判断 9）も本ファイルで pin する。型不正の
例外メッセージにトークン値を載せないこと（漏洩面を広げないこと）も併せて検証する。
"""

from __future__ import annotations

from typing import Any

import pytest
from agents import OutputGuardrail, RunContextWrapper

from oai_agentspec.runtime.guardrails._detectors import Detection, canary_detector
from oai_agentspec.runtime.guardrails.factories import canary_guardrail

pytestmark = pytest.mark.integration


class _Agent:
    """SDK から渡る agent の代役（同一性照合のみに使う不透明スタブ）。"""


def _wrapper(payload: Any = None) -> RunContextWrapper[Any]:
    """`RunContextWrapper` を組む（resolver へ wrapper のまま届くことの照合用）。"""
    return RunContextWrapper(context=payload)


async def _triggered(
    guardrail: Any, output: str, *, context: Any = None, agent: Any = None
) -> bool:
    """guardrail を SDK 経路で 1 回駆動して tripwire の真偽を返す。"""
    result = await guardrail.run(
        context=_wrapper() if context is None else context,
        agent=_Agent() if agent is None else agent,
        agent_output=output,
    )
    return bool(result.output.tripwire_triggered)


# ----------------------------------------------------------------------
# 構築時に resolver を評価しない
# ----------------------------------------------------------------------


def test_構築時にresolverは評価されない() -> None:
    """`canary_guardrail(resolver, ...)` の呼び出し時点で resolver を呼んではならない。

    構築時に解決してしまうと「run ごとに変わるトークン」を扱えず、固定トークンの照合へ静かに
    退行する（呼ばれたら失敗する sentinel で pin する）。
    """

    def sentinel(context: Any, agent: Any) -> str:
        pytest.fail("resolver が構築時に評価された（run ごとの再解決が成立しない）")

    guardrail = canary_guardrail(sentinel, name="canary")
    assert isinstance(guardrail, OutputGuardrail)
    assert guardrail.get_name() == "canary"


# ----------------------------------------------------------------------
# 検知呼び出しごとの再解決（run ごとに異なるトークンで照合が成立する）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_検知呼び出しごとにresolverが再評価されトークンが切り替わる() -> None:
    """検知 4 回で resolver が 4 回呼ばれ、その回のトークンだけで逐語照合が成立する。

    「1 回目は T1 で trip・2 回目は T2 なので T1 では trip しない」を対で押さえることで、
    構築時 1 回だけ解決してキャッシュする変異（および常に trip / 常に非 trip への退行）を
    同時に kill する。カウンタは resolver 本体の同期部で数える。
    """
    tokens = ["T1", "T2", "T1", "T2"]
    calls: list[int] = []

    def resolver(context: Any, agent: Any) -> str:
        index = len(calls)
        calls.append(index)
        return tokens[index]

    guardrail = canary_guardrail(resolver, name="canary")

    assert await _triggered(guardrail, "oops T1 leaked") is True  # 1 回目: T1 で trip
    assert await _triggered(guardrail, "oops T1 leaked") is False  # 2 回目: 現トークンは T2
    assert await _triggered(guardrail, "oops T2 leaked") is False  # 3 回目: 現トークンは T1
    assert await _triggered(guardrail, "oops T2 leaked") is True  # 4 回目: T2 で trip
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_resolverはcontextとagentの2引数で呼ばれ実体がそのまま届く() -> None:
    """resolver は `(context, agent)` で呼ばれ、SDK から渡ったオブジェクトそのものを受ける。

    引数を捨てて別の値（None 等）で呼ぶ変異では run context からトークンを取り出せないため、
    `is` で同一性を照合する。
    """
    seen: list[tuple[Any, ...]] = []
    context = _wrapper({"canary": "CTX-TOKEN"})
    agent = _Agent()

    def resolver(*args: Any, **kwargs: Any) -> str:
        seen.append(args)
        assert kwargs == {}
        return "CTX-TOKEN"

    guardrail = canary_guardrail(resolver, name="canary")
    assert await _triggered(guardrail, "leaked CTX-TOKEN", context=context, agent=agent) is True

    assert len(seen) == 1
    args = seen[0]
    assert len(args) == 2
    assert args[0] is context
    assert args[1] is agent


@pytest.mark.asyncio
async def test_resolverはrun_contextから取り出したトークンで照合できる() -> None:
    """`ctx.context.<attr>` 相当で run context からトークンを読み出す実利用形が成立する。"""

    def resolver(context: Any, agent: Any) -> str:
        return str(context.context["canary"])

    guardrail = canary_guardrail(resolver, name="canary")
    ctx = _wrapper({"canary": "RUN-1"})
    assert await _triggered(guardrail, "leaked RUN-1", context=ctx) is True
    assert await _triggered(guardrail, "leaked RUN-2", context=ctx) is False


# ----------------------------------------------------------------------
# resolver の戻り値の型（単一 str / 複数 Iterable[str]）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolverが単一strを返す場合の逐語照合() -> None:
    """resolver が単一 `str` を返す場合、そのトークンの逐語照合で trip / 非 trip する。"""
    guardrail = canary_guardrail(lambda c, a: "SOLO-TOKEN", name="canary")
    assert await _triggered(guardrail, "here is SOLO-TOKEN") is True
    assert await _triggered(guardrail, "nothing to see") is False


@pytest.mark.asyncio
async def test_resolverが複数トークンを返す場合はいずれの一致でもtripする() -> None:
    """resolver が `Iterable[str]` を返す場合、いずれかのトークン一致で trip する。"""
    guardrail = canary_guardrail(lambda c, a: ["ALPHA-1", "BETA-2"], name="canary")
    assert await _triggered(guardrail, "leaked ALPHA-1") is True
    assert await _triggered(guardrail, "leaked BETA-2") is True
    assert await _triggered(guardrail, "leaked GAMMA-3") is False


@pytest.mark.asyncio
async def test_resolverが返す非list系iterableでも照合できる() -> None:
    """tuple / generator 等の一度きりの iterable でも呼び出しごとに組み直されて照合できる。"""
    guardrail = canary_guardrail(lambda c, a: (t for t in ("GEN-1", "GEN-2")), name="canary")
    assert await _triggered(guardrail, "leaked GEN-2") is True
    # 2 回目も新しい generator が resolver から得られるため照合が成立する（使い切りにならない）。
    assert await _triggered(guardrail, "leaked GEN-1") is True


# ----------------------------------------------------------------------
# None / 空値は発火しない
# ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("resolved", [None, "", [], (), [""], ["", ""]])
async def test_resolverがNoneや空を返すとtripしない(resolved: Any) -> None:
    """`None` / 空文字列 / 空 iterable は「この run にカナリアが無い」状態として発火しない。"""
    guardrail = canary_guardrail(lambda c, a: resolved, name="canary")
    assert await _triggered(guardrail, "any output at all") is False
    assert await _triggered(guardrail, "") is False


# ----------------------------------------------------------------------
# 固定値経路の互換（従来どおり）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_固定値の単一canaryは従来どおり逐語照合でtripする() -> None:
    """`canary_guardrail("TOKEN")` の従来経路が resolver 追加後も不変であること。"""
    guardrail = canary_guardrail("TOKEN", name="canary")
    assert isinstance(guardrail, OutputGuardrail)
    assert await _triggered(guardrail, "oops TOKEN here") is True
    assert await _triggered(guardrail, "clean output") is False


@pytest.mark.asyncio
async def test_固定値の複数canaryは従来どおりいずれの一致でもtripする() -> None:
    """`canary_guardrail(["A", "B"])` の従来経路が不変であること。"""
    guardrail = canary_guardrail(["AAA", "BBB"], name="canary")
    assert await _triggered(guardrail, "has AAA") is True
    assert await _triggered(guardrail, "has BBB") is True
    assert await _triggered(guardrail, "has CCC") is False


def test_固定値経路のguardrail名は既定名にフォールバックする() -> None:
    """`name` 省略時の既定名（`canary_guardrail`）が resolver 追加後も不変であること。"""
    assert canary_guardrail("TOKEN").get_name() == "canary_guardrail"
    assert canary_guardrail(lambda c, a: "TOKEN").get_name() == "canary_guardrail"


# ----------------------------------------------------------------------
# 構築時の arity 検証（`(context, agent)` で bind できない resolver は ValueError）
# ----------------------------------------------------------------------


def test_2引数でbindできないresolverは構築時にValueErrorになる() -> None:
    """引数規約 `(context, agent)` を満たさない resolver は構築時に弾く（実行時まで遅延しない）。

    実行時まで遅延すると、guardrail は登録済みで正常に見えるまま run で初めて壊れる。
    """
    with pytest.raises(ValueError):
        canary_guardrail(lambda: "x", name="canary")
    with pytest.raises(ValueError):
        canary_guardrail(lambda ctx: "x", name="canary")
    with pytest.raises(ValueError):
        canary_guardrail(lambda ctx, agent, extra: "x", name="canary")


def test_可変長引数や既定値を持つresolverは受理される() -> None:
    """`(context, agent)` で bind できれば可変長・既定値付きの resolver も受理する。"""
    assert isinstance(canary_guardrail(lambda *args: "x", name="a"), OutputGuardrail)
    assert isinstance(canary_guardrail(lambda c, a, extra=None: "x", name="b"), OutputGuardrail)


# ----------------------------------------------------------------------
# 逐語照合であること（プレフィクス一致へ劣化していない）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_トークンと先頭が共通する別文字列ではtripしない() -> None:
    """解決トークンの逐語照合であり、プレフィクス一致（正規表現近似）へ劣化していないこと。

    run ごとのトークンを扱うためにプレフィクス正規表現照合へ落とす実装だと、同じ接頭辞を持つ
    無関係な文字列で誤検知する（誤検知は運用上 guardrail の無効化に直結する）。
    """
    guardrail = canary_guardrail(lambda c, a: "CANARY-ABC123", name="canary")
    assert await _triggered(guardrail, "unrelated CANARY-ABC in text") is False
    assert await _triggered(guardrail, "unrelated CANARY-ABCXYZ in text") is False
    assert await _triggered(guardrail, "leaked CANARY-ABC123 in text") is True


# ----------------------------------------------------------------------
# 既存 detector 契約の不変（`Callable[[str], Detection]`）
# ----------------------------------------------------------------------


def test_canary_detectorの1引数契約は不変() -> None:
    """`canary_detector` はテキスト 1 引数で呼べる検知関数を返す（契約を広げていないこと）。"""
    detect = canary_detector("TOKEN")
    hit = detect("oops TOKEN")
    miss = detect("clean")
    assert isinstance(hit, Detection)
    assert hit.triggered is True
    assert miss.triggered is False


# ----------------------------------------------------------------------
# resolver の戻り値型の制約（str / Iterable[str] / None 以外は検知時 TypeError）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolverがMappingを返すと検知時にTypeErrorになる() -> None:
    """resolver が `dict` を返した場合は検知呼び出しで `TypeError`。

    `Mapping` を受理すると `canary_detector` の `tuple(canary)` がキー列を取り、実トークンが
    一切照合されないまま guardrail が恒久 fail-open になる（登録済みで正常に見えるため運用中に
    気付けない）。例外メッセージにはトークン値を載せない（漏洩面を広げない）。
    """
    guardrail = canary_guardrail(lambda c, a: {"session": "CT-7f3a"}, name="canary")
    with pytest.raises(TypeError) as excinfo:
        await _triggered(guardrail, "leaked CT-7f3a")
    message = str(excinfo.value)
    assert "dict" in message
    assert "CT-7f3a" not in message


@pytest.mark.asyncio
async def test_resolverが非iterableのスカラーを返すと検知時にTypeErrorになる() -> None:
    """resolver が `int` を返した場合は検知呼び出しで `TypeError`（型名のみを載せる）。"""
    guardrail = canary_guardrail(lambda c, a: 42, name="canary")
    with pytest.raises(TypeError) as excinfo:
        await _triggered(guardrail, "any output")
    assert "int" in str(excinfo.value)


@pytest.mark.asyncio
async def test_resolverがbytesを返すと検知時にTypeErrorになる() -> None:
    """`bytes` は str の代用として通さない（反復すると int 列になり照合が壊れる）。"""
    guardrail = canary_guardrail(lambda c, a: b"CT-7f3a", name="canary")
    with pytest.raises(TypeError) as excinfo:
        await _triggered(guardrail, "leaked CT-7f3a")
    assert "CT-7f3a" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_resolverが返すiterableに非str要素が混ざると検知時にTypeErrorになる() -> None:
    """要素単位で str を要求する（1 要素でも非 str なら `TypeError`）。

    部分的に正しいトークンを含んでいても照合の健全性は保証できないため縮退させない。正常な
    トークン値も例外メッセージへ載せない。
    """
    guardrail = canary_guardrail(lambda c, a: ["OK-TOKEN", 42], name="canary")
    with pytest.raises(TypeError) as excinfo:
        await _triggered(guardrail, "leaked OK-TOKEN")
    message = str(excinfo.value)
    assert "int" in message
    assert "OK-TOKEN" not in message


@pytest.mark.asyncio
async def test_型制約の追加後も順序なしiterableと使い切りiterableは受理される() -> None:
    """`set` / generator のような順序なし・使い切り iterable が型検証で壊れていないこと。"""
    unordered = canary_guardrail(lambda c, a: {"SET-1", "SET-2"}, name="canary")
    assert await _triggered(unordered, "leaked SET-2") is True
    assert await _triggered(unordered, "clean output") is False

    generated = canary_guardrail(lambda c, a: (t for t in ["GEN-1"]), name="canary")
    assert await _triggered(generated, "leaked GEN-1") is True


# ----------------------------------------------------------------------
# async resolver は構築時に拒否する（公開契約は同期のみ・ADR 0023 判断 9）
# ----------------------------------------------------------------------


def test_async_resolverは構築時にValueErrorで拒否される() -> None:
    """`async def` resolver は構築時に弾く（検知時の未 await coroutine 事故を防ぐ）。

    通してしまうと検知時に `TypeError: 'coroutine' object is not iterable` と
    `coroutine was never awaited` 警告になり、guardrail が壊れたことに気付きにくい。
    """

    async def resolver(context: Any, agent: Any) -> str:
        return "CT-7f3a"

    with pytest.raises(ValueError) as excinfo:
        canary_guardrail(resolver, name="canary")
    assert "同期" in str(excinfo.value)


def test_同期resolverは型検証の追加後も従来どおり受理される() -> None:
    """同期 resolver の受理（非退行）。async 拒否が同期経路へ波及していないこと。"""
    assert isinstance(canary_guardrail(lambda c, a: "TOKEN", name="canary"), OutputGuardrail)


def test_async_callable_objectのresolverも構築時にValueErrorで拒否される() -> None:
    """`async def __call__` を持つ callable object も構築時に弾く（関数形だけの検査では漏れる）。

    `inspect.iscoroutinefunction(resolver)` は callable object に対して False を返すため、
    `__call__` を検査しないと構築時をすり抜け、検知時に `_canary_tokens` の `TypeError` と
    「coroutine was never awaited」警告になる（guardrail が壊れたことに気付きにくい）。
    """

    class AsyncResolver:
        """`__call__` が `async def` の resolver（同期契約に違反する形）。"""

        async def __call__(self, context: Any, agent: Any) -> str:
            return "CT-7f3a"

    with pytest.raises(ValueError) as excinfo:
        canary_guardrail(AsyncResolver(), name="canary")
    assert "同期" in str(excinfo.value)


@pytest.mark.asyncio
async def test_同期callable_objectのresolverは従来どおり受理され照合できる() -> None:
    """同期 `__call__` を持つ callable object は受理され逐語照合が成立する（過剰拒否の検知）。

    `__call__` 検査を「callable object を一律拒否」へ広げると、`Detection` を返す同期
    `__call__` オブジェクトの既存利用が壊れる。
    """

    class SyncResolver:
        """`__call__` が同期の resolver（受理されるべき形）。"""

        def __call__(self, context: Any, agent: Any) -> str:
            return "OBJ-TOKEN"

    guardrail = canary_guardrail(SyncResolver(), name="canary")
    assert isinstance(guardrail, OutputGuardrail)
    assert await _triggered(guardrail, "leaked OBJ-TOKEN") is True
    assert await _triggered(guardrail, "clean output") is False
