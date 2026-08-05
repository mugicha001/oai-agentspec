"""L2: `GuardrailRegistry.canary_guardrail` facade の resolver 受理を検証する。

facade 経由でも resolver を渡して登録でき、境界が OUTPUT のまま・登録実体が factory の戻り値
そのもの（`is`）・構築時に resolver が評価されない・`HELPER_DEFAULTS` 由来の labels / severity が
resolver 経路でも付与される、を pin する。facade / factory のシグネチャ parity は
`test_facade_sync_l2.py` が担うため本ファイルでは扱わない。
"""

from __future__ import annotations

from typing import Any

import pytest
from agents import RunContextWrapper

from oai_agentspec.runtime.guardrails import factories
from oai_agentspec.runtime.guardrails.catalog import HELPER_DEFAULTS
from oai_agentspec.runtime.guardrails.registry import GuardrailRegistry
from oai_agentspec.runtime.guardrails.types import Boundary, Severity

pytestmark = pytest.mark.integration


class _Agent:
    """SDK から渡る agent の代役（同一性照合のみに使う不透明スタブ）。"""


def _wrapper(payload: Any = None) -> RunContextWrapper[Any]:
    """`RunContextWrapper` を組む（resolver へ wrapper のまま届くことの照合用）。"""
    return RunContextWrapper(context=payload)


def _resolver(context: Any, agent: Any) -> str:
    """run context からトークンを読み出す resolver（実利用形）。"""
    return str(context.context["canary"])


# ----------------------------------------------------------------------
# 登録 / 境界 / 検知挙動
# ----------------------------------------------------------------------


def test_facadeへresolverを渡して登録できる() -> None:
    """resolver 経路でも登録され、境界は OUTPUT のまま・`get` で実体が取れる。"""
    reg = GuardrailRegistry()
    spec = reg.canary_guardrail(_resolver, name="canary")

    assert reg.names() == ["canary"]
    assert reg.boundary_of("canary") == "output"
    assert reg.boundary_of("canary") is Boundary.OUTPUT
    assert reg.get("canary") is spec.guardrail


@pytest.mark.asyncio
async def test_facade登録の実体はrunごとのトークンで照合する() -> None:
    """facade 登録した実体が run context 由来のトークンで trip / 非 trip する。

    「resolver を捨てて固定トークンで生成する」変異を behavioral に kill する（trip 側だけでは
    「常に trip」も通るため非該当出力も押さえる）。
    """
    reg = GuardrailRegistry()
    spec = reg.canary_guardrail(_resolver, name="canary")
    guardrail = spec.guardrail

    first = await guardrail.run(
        context=_wrapper({"canary": "RUN-1"}), agent=_Agent(), agent_output="oops RUN-1 leaked"
    )
    assert first.output.tripwire_triggered is True

    second = await guardrail.run(
        context=_wrapper({"canary": "RUN-2"}), agent=_Agent(), agent_output="oops RUN-1 leaked"
    )
    assert second.output.tripwire_triggered is False

    third = await guardrail.run(
        context=_wrapper({"canary": "RUN-2"}), agent=_Agent(), agent_output="oops RUN-2 leaked"
    )
    assert third.output.tripwire_triggered is True


# ----------------------------------------------------------------------
# facade が登録するのは factory の戻り値そのもの
# ----------------------------------------------------------------------


def test_facadeはfactoryの戻り値そのものを登録する(monkeypatch: pytest.MonkeyPatch) -> None:
    """resolver 経路でも facade は factory を 1 回呼び、その戻り値を `is` 同一で登録する。

    値等価の照合では「factory を正しい引数で呼びつつ、戻り値を捨てて別実体（never-trip）を
    登録する」変異が生存するため、モジュール属性の factory を包んで同一性を押さえる。
    """
    original = factories.canary_guardrail
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    produced: list[Any] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, dict(kwargs)))
        result = original(*args, **kwargs)
        produced.append(result)
        return result

    monkeypatch.setattr(factories, "canary_guardrail", spy)

    reg = GuardrailRegistry()
    spec = reg.canary_guardrail(_resolver, name="canary")

    assert calls == [((_resolver,), {"name": "canary"})]
    assert len(produced) == 1
    assert spec.guardrail is produced[0]
    assert reg.get("canary") is produced[0]


# ----------------------------------------------------------------------
# 構築時に resolver を評価しない
# ----------------------------------------------------------------------


def test_facade経由でも構築時にresolverは評価されない() -> None:
    """facade 経路でも登録時点で resolver を呼んではならない（run ごと再解決の pin）。"""

    def sentinel(context: Any, agent: Any) -> str:
        pytest.fail("resolver が facade 登録時に評価された（run ごとの再解決が成立しない）")

    reg = GuardrailRegistry()
    spec = reg.canary_guardrail(sentinel, name="canary")
    assert spec.guardrail is reg.get("canary")


# ----------------------------------------------------------------------
# HELPER_DEFAULTS 由来の labels / severity（resolver 経路でも付与）
# ----------------------------------------------------------------------


def test_facade経由でもasync_resolverは構築時にValueErrorで拒否される() -> None:
    """facade 経路でも `async def` resolver は登録時に弾く（公開契約は同期のみ）。

    factory 側だけで弾いて facade をすり抜けると、登録済みで正常に見えるまま検知時に未 await
    coroutine で壊れる（登録もされないことを `names()` で押さえる）。
    """

    async def resolver(context: Any, agent: Any) -> str:
        return "CT-7f3a"

    reg = GuardrailRegistry()
    with pytest.raises(ValueError) as excinfo:
        reg.canary_guardrail(resolver, name="canary")
    assert "同期" in str(excinfo.value)
    assert reg.names() == []


def test_facade経由でもasync_callable_objectのresolverは構築時に拒否される() -> None:
    """facade 経路でも `async def __call__` を持つ callable object を登録時に弾く。

    `inspect.iscoroutinefunction` は callable object に False を返すため、`__call__` を検査
    しないと登録済みで正常に見えるまま検知時に未 await coroutine で壊れる（副作用なしを
    `names()` で押さえる）。
    """

    class AsyncResolver:
        """`__call__` が `async def` の resolver（同期契約に違反する形）。"""

        async def __call__(self, context: Any, agent: Any) -> str:
            return "CT-7f3a"

    reg = GuardrailRegistry()
    with pytest.raises(ValueError) as excinfo:
        reg.canary_guardrail(AsyncResolver(), name="canary")
    assert "同期" in str(excinfo.value)
    assert reg.names() == []


def test_facade経由の同期callable_objectのresolverは登録できる() -> None:
    """facade 経路でも同期 `__call__` の callable object は登録できる（過剰拒否の検知）。"""

    class SyncResolver:
        """`__call__` が同期の resolver（受理されるべき形）。"""

        def __call__(self, context: Any, agent: Any) -> str:
            return str(context.context["canary"])

    reg = GuardrailRegistry()
    spec = reg.canary_guardrail(SyncResolver(), name="canary")
    assert reg.get("canary") is spec.guardrail
    assert reg.names() == ["canary"]


def test_facade経由の同期resolverは従来どおり登録できる() -> None:
    """async 拒否の追加が facade の同期 resolver 経路へ波及していないこと（非退行）。"""
    reg = GuardrailRegistry()
    spec = reg.canary_guardrail(_resolver, name="canary")
    assert reg.get("canary") is spec.guardrail


def test_resolver経路でも既定のlabelsとseverityが付与される() -> None:
    """`HELPER_DEFAULTS["canary_guardrail"]` 由来の OWASP LLM07 / HIGH が resolver 経路でも付く。"""
    reg = GuardrailRegistry()
    spec = reg.canary_guardrail(_resolver, name="canary")

    assert spec.labels == {"owasp_llm": "LLM07"}
    assert spec.severity is Severity.HIGH
    assert dict(HELPER_DEFAULTS["canary_guardrail"].labels) == {"owasp_llm": "LLM07"}
    assert HELPER_DEFAULTS["canary_guardrail"].severity is Severity.HIGH


def test_resolver経路でも利用者宣言のlabelsとseverityが優先される() -> None:
    """利用者が渡した labels はキー単位でマージされ、severity の明示は既定を上書きする。"""
    reg = GuardrailRegistry()
    spec = reg.canary_guardrail(
        _resolver, name="canary", labels={"team": "sec"}, severity=Severity.LOW
    )

    assert spec.labels == {"owasp_llm": "LLM07", "team": "sec"}
    assert spec.severity is Severity.LOW
