"""L2: `apply_next_turn_policy` が派生 registry へ設置する結線を実 SDK 型で pin する。

到達時ハンドオフ禁止（FR-3）の実現形は「build 時に確定する判定表 + SDK 公式拡張点
（`on_handoff` / `is_enabled`）への合成 + run 単位の到達記録」であり、エージェント実体の
複製・書き換えを伴わない。ここでは実 `agents.Agent` / `agents.Handoff` を構築して次を固定する:

- 判定表に載るエッジ（`src -> X` の流入・X の全出辺）は、素の Agent 直 append 経路でも
  `Handoff` オブジェクトへ昇格し、X の出辺の `is_enabled` は callable になる。
- 判定表に載らないエッジ・禁止を宣言しない宣言・元 registry は従来経路（Agent 直 append）の
  まま（既定挙動不変）。
- 利用者宣言の `on_handoff` は記録の後に chain され、利用者宣言の `is_enabled` は未到達時に
  委譲される（合成が利用者宣言を落とさない）。
- 到達を記録すると当該 run でだけ X の出辺が無効化され、別 run（別 `RunContextWrapper`）では
  無効化されない（run 内一時状態）。
- `clone` は判定表と記録ストアを共有継承するため、派生 registry を clone しても禁止が
  静かに脱落しない。

`agents` を import するため integration マーカー（`tests/_adapters/test_builders_l2.py` と
同じ扱い）。`agents` 非依存で観測できる範囲の pin は `tests/test_registry.py` が担う。
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from agents import Agent, Handoff, RunContextWrapper
from agents.exceptions import UserError
from pydantic import BaseModel

from oai_agentspec import AgentRegistry, AgentSpec, HandoffConfig
from oai_agentspec.next_turn import NextTurnPolicy, NextTurnRule, apply_next_turn_policy
from oai_agentspec.registry import RegistryFrozenError
from oai_agentspec.spec import DynamicHandoff

pytestmark = pytest.mark.integration

_PROHIBIT_BILLING = NextTurnPolicy(rules={"billing": NextTurnRule(no_handoff_on_arrival=True)})
"""billing への到達で billing の全 handoff を無効化する宣言。

判定表に載るのは billing への流入エッジと billing の出辺だけで、他のエッジは対象外。
"""


def _make_registry(
    *,
    triage_options: dict[str, HandoffConfig] | None = None,
    billing_options: dict[str, HandoffConfig] | None = None,
) -> AgentRegistry:
    """triage -> billing -> tech -> triage のハンドオフ構成を持つ実 registry を作る。

    tech -> triage は判定表に載らないエッジ（既定挙動不変の確認用）。

    Args:
        triage_options: triage の per-edge 設定。
        billing_options: billing の per-edge 設定。

    Returns:
        実 `agents.Agent` を構築する `AgentRegistry`。
    """
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="triage",
            instructions="t",
            handoffs=["billing"],
            handoff_options=triage_options or {},
        )
    )
    registry.register(
        AgentSpec(
            name="billing",
            instructions="b",
            handoffs=["tech"],
            handoff_options=billing_options or {},
        )
    )
    registry.register(AgentSpec(name="tech", instructions="x", handoffs=["triage"]))
    return registry


async def _is_enabled(handoff_obj: Handoff, ctx: RunContextWrapper[Any], agent: Agent) -> bool:
    """SDK と同じ呼び方で `Handoff.is_enabled` を評価する。

    Args:
        handoff_obj: 評価対象の handoff。
        ctx: run のコンテキスト wrapper。
        agent: handoff を所有するエージェント（SDK は所有側を第 2 引数に渡す）。

    Returns:
        当該ステップで handoff が有効かどうか。
    """
    attr = handoff_obj.is_enabled
    if isinstance(attr, bool):
        return attr
    return bool(await attr(ctx, agent))


async def _arrive(handoff_obj: Handoff, ctx: RunContextWrapper[Any]) -> Any:
    """SDK と同じ経路（`on_invoke_handoff`）でハンドオフ到達を発生させる。

    Args:
        handoff_obj: 実行する handoff。
        ctx: run のコンテキスト wrapper。

    Returns:
        遷移先の Agent。
    """
    return await handoff_obj.on_invoke_handoff(ctx, "{}")


# ---------------------------------------------------------------------------
# 判定表に載るエッジの昇格とゲート設置
# ---------------------------------------------------------------------------


def test_禁止対象への流入エッジはHandoffへ昇格する() -> None:
    """素の Agent 直 append だった `triage -> billing` が `Handoff` になる（記録の差し込み口）。"""
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, _make_registry())

    edge = derived.get("triage").handoffs[0]

    assert isinstance(edge, Handoff)
    assert edge.agent_name == "billing"


def test_禁止対象Xの出辺はis_enabledがcallableになる() -> None:
    """X の全出辺にはゲートが AND 合成されるため、`is_enabled` は bool のままではない。"""
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, _make_registry())

    edge = derived.get("billing").handoffs[0]

    assert isinstance(edge, Handoff)
    assert callable(edge.is_enabled)


def test_判定表に載らないエッジは素のAgent直appendのまま() -> None:
    """禁止に関係しない `tech -> triage` は昇格せず、従来どおり Agent 実体が入る。"""
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, _make_registry())

    edge = derived.get("tech").handoffs[0]

    assert not isinstance(edge, Handoff)
    assert edge is derived.get("triage")


def test_禁止を宣言しない宣言ではどのエッジも昇格しない() -> None:
    """次ターン指定のみの宣言では合成が設置されず、既定挙動（直 append）が保たれる。"""
    policy = NextTurnPolicy(rules={"billing": NextTurnRule(next_agent="triage")})

    derived = apply_next_turn_policy(policy, _make_registry())

    assert derived.get("triage").handoffs[0] is derived.get("billing")
    assert derived.get("billing").handoffs[0] is derived.get("tech")


def test_元registryの結線は昇格しない() -> None:
    """合成は派生側だけに設置され、元 registry から取得した Agent の結線は変わらない。"""
    registry = _make_registry()

    apply_next_turn_policy(_PROHIBIT_BILLING, registry)

    assert registry.get("triage").handoffs[0] is registry.get("billing")
    assert registry.get("billing").handoffs[0] is registry.get("tech")


# ---------------------------------------------------------------------------
# 到達記録とゲート評価（run 内一時状態）
# ---------------------------------------------------------------------------


async def test_到達前はXの出辺が有効() -> None:
    """記録が無い間は既存 `is_enabled`（既定 True）へ委譲され、handoff は有効なまま。"""
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, _make_registry())
    ctx: RunContextWrapper[Any] = RunContextWrapper(context=None)

    assert (
        await _is_enabled(derived.get("billing").handoffs[0], ctx, derived.get("billing")) is True
    )


async def test_ハンドオフ到達後はXの出辺が無効化される() -> None:
    """`triage -> billing` の到達を経ると、その run では billing の handoff が無効になる。"""
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, _make_registry())
    ctx: RunContextWrapper[Any] = RunContextWrapper(context=None)

    await _arrive(derived.get("triage").handoffs[0], ctx)

    assert (
        await _is_enabled(derived.get("billing").handoffs[0], ctx, derived.get("billing")) is False
    )


async def test_到達記録は別のrunへ持ち越されない() -> None:
    """記録は wrapper 単位のため、別 run では禁止が発動しない（ターンを越えない）。"""
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, _make_registry())
    first: RunContextWrapper[Any] = RunContextWrapper(context=None)
    second: RunContextWrapper[Any] = RunContextWrapper(context=None)

    await _arrive(derived.get("triage").handoffs[0], first)

    assert (
        await _is_enabled(derived.get("billing").handoffs[0], first, derived.get("billing"))
        is False
    )
    assert (
        await _is_enabled(derived.get("billing").handoffs[0], second, derived.get("billing"))
        is True
    )


# ---------------------------------------------------------------------------
# 利用者宣言（on_handoff / is_enabled）が失われないこと
# ---------------------------------------------------------------------------


async def test_利用者宣言のon_handoffは記録の後にchainされる() -> None:
    """記録を前置しても利用者の `on_handoff` は呼ばれ、呼ばれた時点で到達済みになっている。"""
    calls: list[str] = []

    def _user_on_handoff(ctx: RunContextWrapper[Any]) -> None:
        calls.append("user")

    registry = _make_registry(
        triage_options={"billing": HandoffConfig(on_handoff=_user_on_handoff)}
    )
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, registry)
    ctx: RunContextWrapper[Any] = RunContextWrapper(context=None)

    await _arrive(derived.get("triage").handoffs[0], ctx)

    assert calls == ["user"]
    assert (
        await _is_enabled(derived.get("billing").handoffs[0], ctx, derived.get("billing")) is False
    )


async def test_利用者宣言のis_enabled_Falseは未到達でも委譲される() -> None:
    """ゲートは未到達なら既存宣言へ委譲するため、`is_enabled=False` の宣言は無効化されない。"""
    registry = _make_registry(billing_options={"tech": HandoffConfig(is_enabled=False)})
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, registry)
    ctx: RunContextWrapper[Any] = RunContextWrapper(context=None)

    assert (
        await _is_enabled(derived.get("billing").handoffs[0], ctx, derived.get("billing")) is False
    )


async def test_利用者宣言のis_enabled_callableは未到達なら評価される() -> None:
    """既存 callable は未到達時に評価され、その戻り値がゲートの結果になる。"""
    seen: list[str] = []

    def _user_is_enabled(ctx: RunContextWrapper[Any], agent: Any) -> bool:
        seen.append("evaluated")
        return True

    registry = _make_registry(billing_options={"tech": HandoffConfig(is_enabled=_user_is_enabled)})
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, registry)
    ctx: RunContextWrapper[Any] = RunContextWrapper(context=None)

    assert (
        await _is_enabled(derived.get("billing").handoffs[0], ctx, derived.get("billing")) is True
    )
    assert seen == ["evaluated"]


# ---------------------------------------------------------------------------
# clone 継承（判定表と記録ストアの共有継承）
# ---------------------------------------------------------------------------


def test_派生registryをcloneしても昇格とゲートが維持される() -> None:
    """clone が判定表を継承しないと禁止が静かに脱落するため、結線の維持を固定する。"""
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, _make_registry())

    cloned = derived.clone()

    assert isinstance(cloned.get("triage").handoffs[0], Handoff)
    assert callable(cloned.get("billing").handoffs[0].is_enabled)


async def test_cloneした派生registryでも到達で無効化される() -> None:
    """clone 後の registry でも記録とゲートが噛み合い、到達後に handoff が無効になる。"""
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, _make_registry())
    cloned = derived.clone()
    ctx: RunContextWrapper[Any] = RunContextWrapper(context=None)

    assert await _is_enabled(cloned.get("billing").handoffs[0], ctx, cloned.get("billing")) is True

    await _arrive(cloned.get("triage").handoffs[0], ctx)

    assert await _is_enabled(cloned.get("billing").handoffs[0], ctx, cloned.get("billing")) is False


def test_合成を設置していないregistryのcloneは従来どおり() -> None:
    """判定表を持たない通常の registry の clone は、これまでどおり直 append を維持する。"""
    registry = _make_registry()

    cloned = registry.clone()

    assert cloned.get("triage").handoffs[0] is cloned.get("billing")


# ---------------------------------------------------------------------------
# 動的エッジ（DynamicHandoff）経路
#
# 動的エッジは遷移先が実行時（resolver）解決のため、`on_handoff` の時点では到達先が
# 確定しない。候補を一括で記録すると「到達していない X も到達済み」になり FR-3 の意味論が
# 壊れるため、記録は resolver が転送先を確定させた後に、その `(遷移元, 確定先)` が判定表に
# 載るときだけ行われる。ゲートは静的エッジと同じく X の全出辺へ AND 合成される。
# ---------------------------------------------------------------------------

_DYNAMIC_POLICY = NextTurnPolicy(
    rules={
        # billing は到達元不問（triage からの動的到達も判定表に載る）。
        "billing": NextTurnRule(no_handoff_on_arrival=True),
        # tech は billing からの到達に限定するため、(triage, tech) は判定表に載らない。
        "tech": NextTurnRule(no_handoff_on_arrival=True, source="billing"),
    }
)
"""判定表に (triage, billing) と (billing, tech) が載り、(triage, tech) は載らない宣言。"""


def _make_dynamic_registry(
    resolver: Any,
    *,
    on_handoff: Any = None,
    is_enabled: Any = True,
) -> AgentRegistry:
    """triage が billing / tech を候補に持つ動的エッジ 1 本だけを持つ registry を作る。

    Args:
        resolver: `(context, input_json) -> 転送先名` の候補選択関数。
        on_handoff: 動的エッジの利用者宣言コールバック。
        is_enabled: 動的エッジの利用者宣言 `is_enabled`。

    Returns:
        実 `agents.Agent` を構築する `AgentRegistry`。
    """
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="triage",
            instructions="t",
            dynamic_handoffs=[
                DynamicHandoff(
                    tool_name="route",
                    candidates=["billing", "tech"],
                    resolver=resolver,
                    on_handoff=on_handoff,
                    is_enabled=is_enabled,
                )
            ],
        )
    )
    registry.register(AgentSpec(name="billing", instructions="b", handoffs=["tech"]))
    registry.register(AgentSpec(name="tech", instructions="x", handoffs=["triage"]))
    return registry


async def test_動的エッジ経由の到達でもXの出辺が無効化される() -> None:
    """resolver が判定表に載る billing を選ぶと記録され、以降 billing の handoff が無効になる。"""
    derived = apply_next_turn_policy(
        _DYNAMIC_POLICY, _make_dynamic_registry(lambda ctx, input_json: "billing")
    )
    ctx: RunContextWrapper[Any] = RunContextWrapper(context=None)

    assert (
        await _is_enabled(derived.get("billing").handoffs[0], ctx, derived.get("billing")) is True
    )

    await _arrive(derived.get("triage").handoffs[0], ctx)

    assert (
        await _is_enabled(derived.get("billing").handoffs[0], ctx, derived.get("billing")) is False
    )


async def test_動的エッジで判定表に載らない候補へ解決しても記録されない() -> None:
    """候補の一括記録を検知する対の pin。

    (triage, tech) は判定表に載らないため、tech も billing も無効化されない。
    """
    derived = apply_next_turn_policy(
        _DYNAMIC_POLICY, _make_dynamic_registry(lambda ctx, input_json: "tech")
    )
    ctx: RunContextWrapper[Any] = RunContextWrapper(context=None)

    await _arrive(derived.get("triage").handoffs[0], ctx)

    assert await _is_enabled(derived.get("tech").handoffs[0], ctx, derived.get("tech")) is True
    assert (
        await _is_enabled(derived.get("billing").handoffs[0], ctx, derived.get("billing")) is True
    )


async def test_動的エッジでも記録は利用者on_handoffより先に行われる() -> None:
    """利用者コールバックの時点で到達が記録済み（ゲートが False になっている）ことを見る。"""
    seen: list[bool] = []
    holder: dict[str, Any] = {}

    async def _user_on_handoff(ctx: RunContextWrapper[Any]) -> None:
        seen.append(await _is_enabled(holder["edge"], ctx, holder["agent"]))

    derived = apply_next_turn_policy(
        _DYNAMIC_POLICY,
        _make_dynamic_registry(lambda ctx, input_json: "billing", on_handoff=_user_on_handoff),
    )
    holder["edge"] = derived.get("billing").handoffs[0]
    holder["agent"] = derived.get("billing")
    ctx: RunContextWrapper[Any] = RunContextWrapper(context=None)

    await _arrive(derived.get("triage").handoffs[0], ctx)

    assert seen == [False]


async def test_動的エッジの利用者宣言is_enabledは未到達なら評価される() -> None:
    """動的エッジにもゲートが載るが、未到達なら既存 callable の戻り値へ委譲される。"""
    evaluated: list[str] = []

    def _user_is_enabled(ctx: RunContextWrapper[Any], agent: Any) -> bool:
        evaluated.append("evaluated")
        return True

    policy = NextTurnPolicy(rules={"triage": NextTurnRule(no_handoff_on_arrival=True)})
    derived = apply_next_turn_policy(
        policy,
        _make_dynamic_registry(lambda ctx, input_json: "billing", is_enabled=_user_is_enabled),
    )
    ctx: RunContextWrapper[Any] = RunContextWrapper(context=None)

    assert await _is_enabled(derived.get("triage").handoffs[0], ctx, derived.get("triage")) is True
    assert evaluated == ["evaluated"]


async def test_動的エッジの利用者宣言is_enabled_Falseは委譲されFalseのまま() -> None:
    """既存宣言が False の動的エッジは、未到達でもゲートが True に持ち上げない。"""
    policy = NextTurnPolicy(rules={"triage": NextTurnRule(no_handoff_on_arrival=True)})
    derived = apply_next_turn_policy(
        policy, _make_dynamic_registry(lambda ctx, input_json: "billing", is_enabled=False)
    )
    ctx: RunContextWrapper[Any] = RunContextWrapper(context=None)

    assert await _is_enabled(derived.get("triage").handoffs[0], ctx, derived.get("triage")) is False


# ---------------------------------------------------------------------------
# 適用タイミングの仕様（policy 適用は全登録の完了後に行う）
# ---------------------------------------------------------------------------


def test_適用後に登録したエージェントからの到達では禁止が発動しない() -> None:
    """判定表は適用時の登録名から静的展開されるため、後から足した遷移元は対象外になる。

    これは**現在の仕様**であり、バグではない（判定表を build 時に確定させることが FR-3 の
    実現形の前提）。`apply_next_turn_policy` は全エージェントの登録が完了したあとに
    呼ぶこと。後から遷移元を足す場合は、元 registry へ登録し直してから適用する。
    """
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, _make_registry())
    derived.register(
        AgentSpec(name="sales", instructions="s", handoffs=["billing"], handoff_options={})
    )

    # 適用後に登録した sales -> billing は判定表に載らないため昇格せず、記録も入らない。
    assert not isinstance(derived.get("sales").handoffs[0], Handoff)
    assert derived.get("sales").handoffs[0] is derived.get("billing")


async def test_適用後に登録した遷移元からの到達ではXの出辺が無効化されない() -> None:
    """判定表に無い流入エッジ経由では記録が入らないため、X の出辺は有効なまま。"""
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, _make_registry())
    derived.register(
        AgentSpec(name="sales", instructions="s", handoffs=["billing"], handoff_options={})
    )
    ctx: RunContextWrapper[Any] = RunContextWrapper(context=None)

    # 素の Agent 直 append のため on_invoke_handoff の差し込み口が無く、到達は記録されない。
    assert (
        await _is_enabled(derived.get("billing").handoffs[0], ctx, derived.get("billing")) is True
    )


def test_適用済みregistryへの再適用は_ValueError() -> None:
    """1 registry につき policy は 1 つ。重ね掛けは食い違いを生むため build 時に拒否する。"""
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, _make_registry())

    with pytest.raises(ValueError, match="適用済み"):
        apply_next_turn_policy(_PROHIBIT_BILLING, derived)


def test_適用済みregistryへは禁止なしpolicyも再適用できない() -> None:
    """禁止を含まない policy でも、判定表を持つ registry への再適用は拒否する。

    許すと「宣言は禁止なしなのに先行 policy の禁止が clone 継承で残る」食い違いになる。
    """
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, _make_registry())
    policy = NextTurnPolicy(rules={"billing": NextTurnRule(next_agent="triage")})

    with pytest.raises(ValueError, match="適用済み"):
        apply_next_turn_policy(policy, derived)


def test_未適用のregistryには禁止なしpolicyを適用できる() -> None:
    """判定表を持たない registry への適用は従来どおり通る（拒否しすぎない）。"""
    policy = NextTurnPolicy(rules={"billing": NextTurnRule(next_agent="triage")})

    derived = apply_next_turn_policy(policy, _make_registry())

    assert derived.get("triage").handoffs[0] is derived.get("billing")


# ---------------------------------------------------------------------------
# freeze の引き継ぎ（完全性を派生でも保つ）
# ---------------------------------------------------------------------------


async def test_frozenな元registryの派生でも禁止は実際に効く() -> None:
    """派生は frozen で返るが、build / 到達記録 / ゲートの read-only 経路は従来どおり働く。"""
    registry = _make_registry()
    registry.freeze()
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, registry)
    ctx: RunContextWrapper[Any] = RunContextWrapper(context=None)

    await _arrive(derived.get("triage").handoffs[0], ctx)

    assert (
        await _is_enabled(derived.get("billing").handoffs[0], ctx, derived.get("billing")) is False
    )
    with pytest.raises(RegistryFrozenError):
        derived.register(AgentSpec(name="sales", instructions="s"))


# ---------------------------------------------------------------------------
# ファクトリ登録への禁止宣言は build 時に拒否する（silent no-op の防止）
# ---------------------------------------------------------------------------


def _make_factory_registry(factory_name: str) -> AgentRegistry:
    """triage -> billing -> tech 構成のうち 1 つを実 Agent のファクトリ登録にする。

    Args:
        factory_name: ファクトリ登録に置き換えるエージェント名（triage / billing）。

    Returns:
        指定名だけがファクトリ登録の `AgentRegistry`。
    """
    registry = AgentRegistry()
    if factory_name == "triage":
        registry.register_factory(
            "triage",
            lambda r: Agent(name="triage", instructions="t", handoffs=[r.get("billing")]),
        )
    else:
        registry.register(AgentSpec(name="triage", instructions="t", handoffs=["billing"]))
    if factory_name == "billing":
        registry.register_factory(
            "billing",
            lambda r: Agent(name="billing", instructions="b", handoffs=[r.get("tech")]),
        )
    else:
        registry.register(AgentSpec(name="billing", instructions="b", handoffs=["tech"]))
    registry.register(AgentSpec(name="tech", instructions="x", handoffs=["triage"]))
    return registry


def test_ファクトリ登録の禁止対象は結線できないため_ValueError() -> None:
    """X がファクトリ登録だと出辺が生の Agent のままでゲートを載せられない。

    受理してしまうと「宣言は禁止なのに到達後も handoff が提示される」silent failure になる。
    """
    policy = NextTurnPolicy(
        rules={"billing": NextTurnRule(no_handoff_on_arrival=True, source="triage")}
    )

    with pytest.raises(ValueError, match="ファクトリ登録"):
        apply_next_turn_policy(policy, _make_factory_registry("billing"))


def test_sourceで明示指定したファクトリ登録の到達元は_ValueError() -> None:
    """遷移元がファクトリ登録だと流入エッジが昇格せず、記録が入らない（ゲートが素通しになる）。

    実測: 拒否しないと `triage -> billing` は生の Agent 直 append のままで記録の
    差し込み口が無く、billing の出辺のゲートが常に「未到達」と判定して素通しになる。
    """
    policy = NextTurnPolicy(
        rules={"billing": NextTurnRule(no_handoff_on_arrival=True, source="triage")}
    )

    with pytest.raises(ValueError, match="ファクトリ登録"):
        apply_next_turn_policy(policy, _make_factory_registry("triage"))


async def test_無関係なファクトリ登録があっても包括禁止はspec経路で効く() -> None:
    """禁止対象へのエッジを持たないファクトリ登録は拒否せず、spec 経路の禁止は従来どおり働く。"""
    registry = _make_registry()
    registry.register_factory("standalone", lambda r: Agent(name="standalone", instructions="s"))
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, registry)
    ctx: RunContextWrapper[Any] = RunContextWrapper(context=None)

    await _arrive(derived.get("triage").handoffs[0], ctx)

    assert (
        await _is_enabled(derived.get("billing").handoffs[0], ctx, derived.get("billing")) is False
    )


def test_禁止を宣言しなければファクトリ登録でも適用できる() -> None:
    """次ターン指定のみなら `registry.get` で解決するだけのため、ファクトリ登録でも通る。"""
    policy = NextTurnPolicy(rules={"billing": NextTurnRule(next_agent="triage")})

    derived = apply_next_turn_policy(policy, _make_factory_registry("billing"))

    assert derived.get("billing").name == "billing"


# ---------------------------------------------------------------------------
# 利用者宣言の誤りは build 時に落とす（合成が SDK の検証を隠さない）
# ---------------------------------------------------------------------------


class _EscalationInput(BaseModel):
    """`input_type` あり経路の handoff 入力（最小の pydantic モデル）。"""

    reason: str


def test_記録対象エッジの非callableなon_handoffはbuild時に落ちる() -> None:
    """禁止対象への流入エッジでも、非 callable な `on_handoff` は build 時に落ちる。

    合成すると SDK の検証は合成 callable しか見ないため発火せず、run 中の `TypeError` へ
    後ろ倒しになる。`input_type` なしの経路では SDK も `callable()` を見ずに
    `inspect.signature` へ渡すため、例外型は SDK と同じ `TypeError` になる。
    """
    registry = _make_registry(triage_options={"billing": HandoffConfig(on_handoff="not-callable")})
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, registry)

    with pytest.raises(TypeError, match="is not a callable object"):
        derived.get("triage")


def test_記録対象エッジの非callableなon_handoffはinput_typeありなら_UserError() -> None:
    """`input_type` あり経路の非 callable は SDK と同じく `UserError` で落ちる。"""
    registry = _make_registry(
        triage_options={
            "billing": HandoffConfig(on_handoff="not-callable", input_type=_EscalationInput)
        }
    )
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, registry)

    with pytest.raises(UserError, match="must be callable"):
        derived.get("triage")


def test_記録対象エッジのinput_typeあり_on_handoff未宣言はbuild時に_UserError() -> None:
    """禁止対象への流入エッジでも、`input_type` ありの `on_handoff` 未宣言は build 時に落ちる。

    合成が常に `on_handoff` を埋めるため SDK の必須チェックが発火せず、コールバック無しの
    まま handoff が成立してしまうのを防ぐ。例外型も SDK と同じ `UserError` に揃える
    （判定表に載らないエッジと同じ型で落ちることを次のテストと対で pin する）。
    """
    registry = _make_registry(
        triage_options={"billing": HandoffConfig(input_type=_EscalationInput)}
    )
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, registry)

    with pytest.raises(UserError, match="You must provide on_handoff"):
        derived.get("triage")


def test_判定表に載らないエッジのinput_type誤りは従来どおりSDKが落とす() -> None:
    """禁止に関係しないエッジの build 挙動は変えない（SDK の `UserError` のまま）。"""
    registry = _make_registry()
    registry.update(
        AgentSpec(
            name="tech",
            instructions="x",
            handoffs=["triage"],
            handoff_options={"triage": HandoffConfig(input_type=_EscalationInput)},
        )
    )
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, registry)

    with pytest.raises(UserError, match="You must provide on_handoff"):
        derived.get("tech")


def test_ゲートのみ載る出辺のinput_type誤りも従来どおりSDKが落とす() -> None:
    """ゲートだけを合成する出辺は `on_handoff` を埋めないため、SDK の検証が働き続ける。"""
    registry = _make_registry(billing_options={"tech": HandoffConfig(input_type=_EscalationInput)})
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, registry)

    with pytest.raises(UserError, match="You must provide on_handoff"):
        derived.get("billing")


def _two_args(ctx: Any, payload: Any) -> None:
    """2 引数の利用者 `on_handoff`（`input_type` なしのエッジでは arity 誤りになる）。

    Args:
        ctx: run のコンテキスト wrapper。
        payload: handoff 入力。
    """
    return None


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(HandoffConfig(on_handoff=_two_args), UserError, id="arity-mismatch"),
        pytest.param(
            HandoffConfig(input_type=_EscalationInput), UserError, id="input-type-without-handler"
        ),
        pytest.param(
            HandoffConfig(on_handoff="not-callable", input_type=_EscalationInput),
            UserError,
            id="not-callable-with-input-type",
        ),
        pytest.param(HandoffConfig(on_handoff="not-callable"), TypeError, id="not-callable"),
        pytest.param(HandoffConfig(on_handoff=time.time), ValueError, id="unreadable-signature"),
    ],
)
def test_誤宣言の例外型は禁止対象エッジと判定表外エッジで一致する(
    config: HandoffConfig, expected: type[Exception]
) -> None:
    """同一の誤宣言は、禁止を宣言してもしなくても同じ例外型で build 時に落ちる。

    型が変わると、無関係な禁止ルールを足しただけで利用者の `except` 節がすり抜ける
    （禁止を宣言したエッジだけ挙動が変わる非対称）。基準は SDK の `handoff()` が送出する型で、
    `input_type` あり経路の非 callable は `UserError`・なし経路の非 callable は
    `inspect.signature` 由来の `TypeError`・署名取得不能は `ValueError`・arity 誤りは
    `UserError`。派生クラスでの一致に緩めないよう型の同一性で比較する。

    Args:
        config: triage -> billing エッジへ宣言する per-edge 設定（誤宣言）。
        expected: 両経路で送出されるべき例外型。
    """
    options = {"billing": config}
    plain = _make_registry(triage_options=options)
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, _make_registry(triage_options=options))

    with pytest.raises(Exception) as plain_exc:
        plain.get("triage")
    with pytest.raises(Exception) as derived_exc:
        derived.get("triage")

    assert type(plain_exc.value) is expected
    assert type(derived_exc.value) is expected


def test_正しい宣言は禁止対象エッジでも判定表外エッジでもbuildを通る() -> None:
    """対称性 pin が「両方落ちる」だけで満たされないよう、正常系も対で pin する。"""

    def _user_one(ctx: Any) -> None:
        return None

    options = {"billing": HandoffConfig(on_handoff=_user_one)}
    plain = _make_registry(triage_options=options)
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, _make_registry(triage_options=options))

    assert plain.get("triage").name == "triage"
    assert derived.get("triage").name == "triage"


def test_記録対象エッジの正しいon_handoff宣言は従来どおり受理される() -> None:
    """`input_type` ありで 2 引数の `on_handoff` を宣言したエッジは build を通る。"""

    def _user(ctx: RunContextWrapper[Any], payload: _EscalationInput) -> None:
        return None

    registry = _make_registry(
        triage_options={"billing": HandoffConfig(on_handoff=_user, input_type=_EscalationInput)}
    )
    derived = apply_next_turn_policy(_PROHIBIT_BILLING, registry)

    assert isinstance(derived.get("triage").handoffs[0], Handoff)
