"""L1: AgentRegistry のロジック検証（agents 非依存・フェイク注入）。"""

from __future__ import annotations

import pytest

from oai_agentspec import (
    AgentRegistry,
    AgentSpec,
    HandoffGraph,
    IntegrityError,
    RegistryFrozenError,
)
from oai_agentspec.next_turn import NextTurnPolicy, NextTurnRule, apply_next_turn_policy
from oai_agentspec.protocols import AgentBuilder

from _helpers.fake_builder import FakeAgent, FakeAgentBuilder


def make_registry() -> tuple[AgentRegistry, FakeAgentBuilder]:
    builder = FakeAgentBuilder()
    return AgentRegistry(agent_builder=builder), builder


def test_fake_builder_satisfies_agent_builder_protocol() -> None:
    """DI 拡張点 AgentBuilder（runtime_checkable Protocol）の構造的適合を担保する。"""
    assert isinstance(FakeAgentBuilder(), AgentBuilder)


def test_register_duplicate_raises() -> None:
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="x"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(AgentSpec(name="a", instructions="y"))


def test_lazy_build_only_on_get() -> None:
    reg, builder = make_registry()
    reg.register(AgentSpec(name="a", instructions="x"))
    assert builder.built == []
    reg.get("a")
    assert builder.built == ["a"]


def test_get_unknown_raises() -> None:
    reg, _ = make_registry()
    with pytest.raises(KeyError):
        reg.get("missing")


def test_instructions_passthrough() -> None:
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="hello"))
    assert reg.get("a").instructions == "hello"


def test_callable_instructions_arity() -> None:
    reg, _ = make_registry()
    with pytest.raises(ValueError, match="2 引数"):
        reg.register(AgentSpec(name="a", instructions=lambda ctx: "x"))
    reg.register(AgentSpec(name="b", instructions=lambda ctx, agent: "x"))


def test_callable_instructions_accepts_two_arg_callable_variants() -> None:
    """2 引数で呼び出せる callable（デフォルト引数付き / 可変長）は register を通過する。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions=lambda ctx, agent, cfg=None: "x"))
    reg.register(AgentSpec(name="b", instructions=lambda *args: "x"))


def test_callable_instructions_skips_unintrospectable_callable() -> None:
    """シグネチャ取得不能な callable（builtin 等）は検証をスキップし register を通過する。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions=zip))


def test_registration_order_independent() -> None:
    reg, _ = make_registry()
    reg.register(AgentSpec(name="parent", instructions="p", handoffs=["child"]))
    reg.register(AgentSpec(name="child", instructions="c"))
    parent = reg.get("parent")
    assert parent.handoffs[0] is reg.get("child")


def test_entry_name_is_first_registered() -> None:
    """entry_name は登録順の先頭を返す（names() の昇順とは独立）。"""
    reg, _ = make_registry()
    assert reg.entry_name is None
    reg.register(AgentSpec(name="zeta", instructions="z"))
    reg.register(AgentSpec(name="alpha", instructions="a"))
    # names() は昇順（alpha が先頭）だが、entry_name は登録順の先頭（zeta）。
    assert reg.names() == ["alpha", "zeta"]
    assert reg.entry_name == "zeta"


def test_entry_name_after_unregister() -> None:
    """先頭エージェントを unregister すると次の登録が entry_name になる。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="first", instructions="f"))
    reg.register(AgentSpec(name="second", instructions="s"))
    reg.unregister("first")
    assert reg.entry_name == "second"


def test_cyclic_handoff_identity() -> None:
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a", handoffs=["b"]))
    reg.register(AgentSpec(name="b", instructions="b", handoffs=["a"]))
    a = reg.get("a")
    b = reg.get("b")
    assert a.handoffs[0] is b
    assert b.handoffs[0] is a


def test_three_way_cycle() -> None:
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a", handoffs=["b"]))
    reg.register(AgentSpec(name="b", instructions="b", handoffs=["c"]))
    reg.register(AgentSpec(name="c", instructions="c", handoffs=["a"]))
    a, b, c = reg.get("a"), reg.get("b"), reg.get("c")
    assert a.handoffs[0] is b
    assert b.handoffs[0] is c
    assert c.handoffs[0] is a


def test_unreachable_spec_not_built() -> None:
    reg, builder = make_registry()
    reg.register(AgentSpec(name="a", instructions="a"))
    reg.register(AgentSpec(name="orphan", instructions="o"))
    reg.get("a")
    assert "orphan" not in builder.built


def test_unregistered_handoff_dst_errors() -> None:
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a", handoffs=["ghost"]))
    with pytest.raises(KeyError, match="ghost"):
        reg.get("a")


def test_wiring_failure_rolls_back_partial_build() -> None:
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a", handoffs=["late"]))
    with pytest.raises(KeyError):
        reg.get("a")
    assert "a" not in reg._built  # noqa: SLF001 - 残留しないことの検証
    reg.register(AgentSpec(name="late", instructions="late"))
    a = reg.get("a")
    assert a.handoffs[0] is reg.get("late")


def test_update_reinstantiates() -> None:
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="v1"))
    a1 = reg.get("a")
    reg.update(AgentSpec(name="a", instructions="v2"))
    a2 = reg.get("a")
    assert a1 is not a2


def test_update_chains_to_dependents() -> None:
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a", handoffs=["b"]))
    reg.register(AgentSpec(name="b", instructions="b"))
    a_before = reg.get("a")
    reg.update(AgentSpec(name="b", instructions="b2"))
    a_after = reg.get("a")
    assert a_before is not a_after
    assert a_after.handoffs[0] is reg.get("b")


def test_unregister() -> None:
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a"))
    reg.get("a")
    reg.unregister("a")
    with pytest.raises(KeyError):
        reg.get("a")


def test_update_handoffs_append_and_replace() -> None:
    # _update_handoffs は内部プリミティブ（ユーザーは HandoffGraph 経由）。mode 挙動を直接検証。
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a"))
    reg.register(AgentSpec(name="b", instructions="b"))
    reg.register(AgentSpec(name="c", instructions="c"))
    reg._update_handoffs("a", ["b"], mode="append")  # noqa: SLF001
    reg._update_handoffs("a", ["b", "c"], mode="append")  # noqa: SLF001 - 重複排除
    assert reg._specs["a"].handoffs == ["b", "c"]  # noqa: SLF001
    reg._update_handoffs("a", ["c"], mode="replace")  # noqa: SLF001
    assert reg._specs["a"].handoffs == ["c"]  # noqa: SLF001


def test_validate_detects_missing_references() -> None:
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a", handoffs=["ghost"], sub_agents=["nope"]))
    with pytest.raises(KeyError, match="ghost.*nope|nope.*ghost"):
        reg.validate()


def test_validate_passes_when_all_resolved() -> None:
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a", handoffs=["b"]))
    reg.register(AgentSpec(name="b", instructions="b"))
    reg.validate()  # 例外が出なければ OK


# ----------------------------------------------------------------------
# clone（評価で利用者 registry を汚さない派生 registry を作るプリミティブ）
# ----------------------------------------------------------------------


def test_clone_copies_specs_and_preserves_order() -> None:
    """clone は spec を引き継ぎ、登録順（entry_name 基準）も維持する。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a"))
    reg.register(AgentSpec(name="b", instructions="b"))
    cloned = reg.clone()
    assert cloned.names() == ["a", "b"]
    assert cloned.entry_name == "a"


def test_clone_is_independent_from_source() -> None:
    """clone への登録は元 registry に影響しない（独立インスタンス）。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a"))
    cloned = reg.clone()
    cloned.register(AgentSpec(name="b", instructions="b"))
    assert "b" not in reg.names()
    assert "b" in cloned.names()


def test_clone_transform_spec_applies_to_each_spec() -> None:
    """transform_spec が各 spec へ適用され、元 spec は不変。"""
    from dataclasses import replace

    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="orig"))
    cloned = reg.clone(transform_spec=lambda s: replace(s, instructions="mocked"))
    cloned.get("a")
    assert cloned._specs["a"].instructions == "mocked"  # noqa: SLF001 - 検証
    # 元 spec は不変。
    assert reg._specs["a"].instructions == "orig"  # noqa: SLF001 - 検証


def test_clone_carries_factories_without_transform() -> None:
    """factory 登録は spec 実体が無いため transform 対象外で、そのまま引き継ぐ。"""
    reg, _ = make_registry()
    reg.register_factory("f", lambda r: FakeAgent(name="f"))
    calls = {"transform": 0}

    def _transform(s: AgentSpec) -> AgentSpec:
        calls["transform"] += 1
        return s

    cloned = reg.clone(transform_spec=_transform)
    # factory は transform を経由しない。
    assert calls["transform"] == 0
    assert "f" in cloned.names()
    # クローンから factory agent を取得できる。
    assert cloned.get("f").name == "f"


def test_clone_spec_is_independent_object_no_transform() -> None:
    """transform 無しでもクローンの spec は元と別オブジェクト（identity 共有しない・Codex P2）。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a", handoffs=["b"]))
    cloned = reg.clone()
    # 同一オブジェクトを共有しない。
    assert cloned._specs["a"] is not reg._specs["a"]  # noqa: SLF001 - 不変検証
    # 可変コンテナ（handoffs）も別 list。
    assert cloned._specs["a"].handoffs is not reg._specs["a"].handoffs  # noqa: SLF001


def test_clone_mutating_handoffs_does_not_affect_source() -> None:
    """クローン spec の handoffs を変更（append / 再代入）しても元 registry の spec は不変。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a", handoffs=["b"]))
    before = list(reg._specs["a"].handoffs)  # noqa: SLF001
    cloned = reg.clone()

    cloned._specs["a"].handoffs.append("c")  # noqa: SLF001 - in-place 変更
    cloned._specs["a"].handoffs = ["x", "y"]  # noqa: SLF001 - 再代入
    # 元 registry の spec は不変（コンテナ共有なし）。
    assert reg._specs["a"].handoffs == before  # noqa: SLF001


def test_clone_mutating_all_containers_does_not_affect_source() -> None:
    """クローン spec の全可変コンテナを変更しても元 registry の spec のコンテナは不変。"""
    reg, _ = make_registry()
    reg.register(
        AgentSpec(
            name="a",
            instructions="a",
            tools=["t1"],
            input_guardrails=["ig1"],
            output_guardrails=["og1"],
            handoffs=["b"],
            sub_agents=["s1"],
            extra={"k": "v"},
        )
    )
    src = reg._specs["a"]  # noqa: SLF001
    cloned = reg.clone()
    dst = cloned._specs["a"]  # noqa: SLF001

    # 各コンテナが別インスタンス。
    assert dst.tools is not src.tools
    assert dst.input_guardrails is not src.input_guardrails
    assert dst.output_guardrails is not src.output_guardrails
    assert dst.handoffs is not src.handoffs
    assert dst.handoff_options is not src.handoff_options
    assert dst.sub_agents is not src.sub_agents
    assert dst.sub_agent_tools is not src.sub_agent_tools
    assert dst.dynamic_handoffs is not src.dynamic_handoffs
    assert dst.extra is not src.extra

    # クローン側で全コンテナを変更しても元は不変。
    dst.tools.append("t2")
    dst.input_guardrails.append("ig2")
    dst.output_guardrails.append("og2")
    dst.sub_agents.append("s2")
    dst.extra["k2"] = "v2"
    assert src.tools == ["t1"]
    assert src.input_guardrails == ["ig1"]
    assert src.output_guardrails == ["og1"]
    assert src.sub_agents == ["s1"]
    assert src.extra == {"k": "v"}


def test_clone_with_transform_still_copies_untouched_containers() -> None:
    """transform が tools だけ差し替えても、handoffs 等の他コンテナはコピーされ共有しない。"""
    from dataclasses import replace

    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a", handoffs=["b"]))
    # transform は tools のみ差し替え（dataclasses.replace は handoffs を参照共有のまま返す）。
    cloned = reg.clone(transform_spec=lambda s: replace(s, tools=["mock"]))
    # それでも clone 側で handoffs は別 list（_copy_spec が全コンテナをコピー）。
    assert cloned._specs["a"].handoffs is not reg._specs["a"].handoffs  # noqa: SLF001
    cloned._specs["a"].handoffs.append("c")  # noqa: SLF001
    assert reg._specs["a"].handoffs == ["b"]  # noqa: SLF001 - 元は不変


def test_clone_transform_mutating_input_does_not_affect_source() -> None:
    """transform が入力 spec を mutate しても元 registry は不変（コピーを先に渡す・Codex P2）。"""

    def _mutate(s: AgentSpec) -> AgentSpec:
        s.handoffs.append("EVIL")
        s.tools.append("EVIL")
        return s

    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a", handoffs=["b"], tools=["t"]))
    reg.clone(transform_spec=_mutate)
    # transform はコピーを受け取るため、元 registry の spec は汚れない。
    assert reg._specs["a"].handoffs == ["b"]  # noqa: SLF001
    assert reg._specs["a"].tools == ["t"]  # noqa: SLF001


# ----------------------------------------------------------------------
# freeze（runtime インテグリティ防御の遮断面・RegistryFrozenError）
# ----------------------------------------------------------------------
def test_registry_frozen_error_is_runtime_error_not_integrity_error() -> None:
    """``RegistryFrozenError`` は ``RuntimeError`` を継承し、``IntegrityError`` 系統と分離。

    利用者の ``except IntegrityError`` で誤って握り潰されないことを担保する。
    """
    assert issubclass(RegistryFrozenError, RuntimeError)
    assert not issubclass(RegistryFrozenError, IntegrityError)


def test_freeze_is_idempotent() -> None:
    """``freeze()`` を 2 度呼んでも 2 回目は no-op として成功する。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a"))
    reg.freeze()
    reg.freeze()  # 2 回目も成功。
    with pytest.raises(RegistryFrozenError):
        reg.register(AgentSpec(name="b", instructions="b"))


def test_register_raises_after_freeze() -> None:
    """freeze 後の ``register`` は ``RegistryFrozenError`` を raise する。"""
    reg, _ = make_registry()
    reg.freeze()
    with pytest.raises(RegistryFrozenError, match="register"):
        reg.register(AgentSpec(name="a", instructions="a"))


def test_register_factory_raises_after_freeze() -> None:
    """freeze 後の ``register_factory`` は ``RegistryFrozenError`` を raise する。"""
    reg, _ = make_registry()
    reg.freeze()
    with pytest.raises(RegistryFrozenError, match="register_factory"):
        reg.register_factory("f", lambda r: FakeAgent(name="f"))


def test_update_raises_after_freeze() -> None:
    """freeze 後の ``update`` は ``RegistryFrozenError`` を raise する。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a"))
    reg.freeze()
    with pytest.raises(RegistryFrozenError, match="update"):
        reg.update(AgentSpec(name="a", instructions="b"))


def test_unregister_raises_after_freeze() -> None:
    """freeze 後の ``unregister`` は ``RegistryFrozenError`` を raise する。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a"))
    reg.freeze()
    with pytest.raises(RegistryFrozenError, match="unregister"):
        reg.unregister("a")


def test_update_handoffs_direct_raises_after_freeze() -> None:
    """freeze 後の ``_update_handoffs`` 直接呼びも ``RegistryFrozenError`` を raise する。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a"))
    reg.register(AgentSpec(name="b", instructions="b"))
    reg.freeze()
    with pytest.raises(RegistryFrozenError, match="_update_handoffs"):
        reg._update_handoffs("a", ["b"], mode="replace")  # noqa: SLF001


def test_handoff_graph_apply_raises_after_freeze() -> None:
    """freeze 後の ``HandoffGraph.apply`` 経由でも ``RegistryFrozenError`` を raise する。

    apply は内部で ``_update_handoffs`` に委譲するため、apply 経路もここで遮断される。
    """
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a"))
    reg.register(AgentSpec(name="b", instructions="b"))
    graph = HandoffGraph(entry="a")
    graph.edge("a", "b")
    reg.freeze()
    with pytest.raises(RegistryFrozenError):
        graph.apply(reg)


# read-only API は freeze 後も成功
def test_get_succeeds_after_freeze() -> None:
    """``get`` は read-only のため freeze 後も成功する。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a"))
    reg.freeze()
    assert reg.get("a").name == "a"


def test_validate_succeeds_after_freeze() -> None:
    """``validate`` は read-only のため freeze 後も成功する。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a"))
    reg.freeze()
    reg.validate()


def test_entry_name_succeeds_after_freeze() -> None:
    """``entry_name`` は read-only のため freeze 後も読み取れる。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="first", instructions="f"))
    reg.freeze()
    assert reg.entry_name == "first"


def test_names_succeeds_after_freeze() -> None:
    """``names`` は read-only のため freeze 後も読み取れる。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="x", instructions="x"))
    reg.freeze()
    assert reg.names() == ["x"]


def test_clone_does_not_inherit_frozen_state() -> None:
    """``clone()`` の戻り値は freeze 状態を引き継がない（独立した unfrozen registry）。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a"))
    reg.freeze()
    cloned = reg.clone()
    # クローン側は unfrozen のため register が成功する。
    cloned.register(AgentSpec(name="b", instructions="b"))
    assert "b" in cloned.names()
    # 元 registry は依然 frozen。
    with pytest.raises(RegistryFrozenError):
        reg.register(AgentSpec(name="c", instructions="c"))


# ----------------------------------------------------------------------
# freeze 後の spec snapshot（外部参照経由の mutation 遮断）
# ----------------------------------------------------------------------
def test_freeze_snapshots_specs_against_external_mutation() -> None:
    """``freeze()`` 後、外部参照経由の ``spec.instructions = ...`` が build に伝播しない。

    freeze の本質的契約: public な ``AgentSpec`` オブジェクト経由で改竄を仕込めない。
    """
    reg, _ = make_registry()
    spec = AgentSpec(name="triage", instructions="ok")
    reg.register(spec)
    reg.freeze()

    # 外部参照経由で spec を書き換え（攻撃シナリオ）。
    spec.instructions = "EVIL"

    built = reg.get("triage")
    assert built.instructions == "ok"  # snapshot により改竄が反映されない。


def test_freeze_snapshots_handoffs_list() -> None:
    """``freeze()`` 後の ``spec.handoffs.append(...)`` が registry の build に反映されない。"""
    reg, _ = make_registry()
    spec = AgentSpec(name="a", instructions="a", handoffs=["b"])
    reg.register(spec)
    reg.register(AgentSpec(name="b", instructions="b"))
    reg.freeze()

    # 外部参照経由で handoffs を伸ばす（攻撃シナリオ）。
    spec.handoffs.append("evil")

    built = reg.get("a")
    # build 結果に "evil" は紛れ込まない（snapshot で list がコピーされている）。
    assert [getattr(h, "name", h) for h in built.handoffs] == ["b"]


def test_freeze_snapshots_tools_list() -> None:
    """``freeze()`` 後の ``spec.tools.append(...)`` も build に反映されない。"""
    reg, _ = make_registry()
    spec = AgentSpec(name="a", instructions="a", tools=["original_tool"])
    reg.register(spec)
    reg.freeze()

    spec.tools.append("evil_tool")

    built = reg.get("a")
    assert built.tools == ["original_tool"]


def test_freeze_snapshot_invalidates_built_cache() -> None:
    """``freeze()`` 前に build された Agent は freeze 後の get で snapshot から再構築される。

    snapshot 切り替え時に ``_built`` が invalidate されないと、freeze 前の spec から
    build された Agent が改竄に対する防御を持たないまま残ってしまう。
    """
    reg, builder = make_registry()
    spec = AgentSpec(name="a", instructions="ok")
    reg.register(spec)
    reg.get("a")  # 1 回 build しておく。
    assert builder.built == ["a"]
    reg.freeze()
    # snapshot 切り替えと invalidate により、次の get で再 build される。
    reg.get("a")
    assert builder.built == ["a", "a"]


def test_freeze_is_idempotent_does_not_reset_snapshot() -> None:
    """``freeze()`` 2 回目以降は no-op で snapshot を再生成しない（spec 参照が安定）。"""
    reg, _ = make_registry()
    spec = AgentSpec(name="a", instructions="ok")
    reg.register(spec)
    reg.freeze()
    snapshot_after_first = reg._specs["a"]  # noqa: SLF001
    reg.freeze()  # 2 回目（no-op）
    assert reg._specs["a"] is snapshot_after_first  # noqa: SLF001
    # 外部 mutation は依然反映されない（snapshot を再取得しないため）。
    spec.instructions = "EVIL"
    built = reg.get("a")
    assert built.instructions == "ok"


def test_clone_after_freeze_returns_unfrozen() -> None:
    """``freeze()`` 後の ``clone()`` は unfrozen registry を返し、元 registry は frozen のまま。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="a"))
    reg.freeze()
    cloned = reg.clone()
    assert cloned._frozen is False  # noqa: SLF001
    cloned.register(AgentSpec(name="b", instructions="b"))  # unfrozen で成功。
    with pytest.raises(RegistryFrozenError):
        reg.register(AgentSpec(name="c", instructions="c"))


# ----------------------------------------------------------------------
# clone() の _copy_spec 二重呼び出し排除リグレッション
# ----------------------------------------------------------------------
def test_clone_transform_spec_mutation_preserved() -> None:
    """transform_spec が in-place mutate した結果がそのまま clone 後の spec に反映される。

    リグレッション防止: 過去に ``_copy_spec`` を 2 回呼ぶ実装では transform の mutation が
    無効化されるケースがあったため、明示的に「mutate して self を返す」transform の
    結果が clone 後に保持されることを保証する。
    """
    reg, _ = make_registry()
    reg.register(AgentSpec(name="a", instructions="orig", tools=["original"]))

    def _mutate(s: AgentSpec) -> AgentSpec:
        s.tools.append("added_by_transform")  # in-place mutation
        return s

    cloned = reg.clone(transform_spec=_mutate)
    # transform の mutation が cloned 側に保持される。
    assert cloned._specs["a"].tools == ["original", "added_by_transform"]  # noqa: SLF001
    # 元 registry は影響を受けない（_copy_spec 1 回目で独立コピーされているため）。
    assert reg._specs["a"].tools == ["original"]  # noqa: SLF001


# ----------------------------------------------------------------------
# Next-Turn Agent Override: 到達時ハンドオフ禁止の結線（既定挙動不変の側）
#
# 実 SDK 型（`Handoff` への昇格・`is_enabled` ゲート）を伴う結線の pin は
# `tests/_adapters/test_next_turn_registry_l2.py`（integration）が担う。ここでは
# フェイク builder のまま観測できる「合成が入らない経路」を固定する。
# ----------------------------------------------------------------------
def make_handoff_registry() -> AgentRegistry:
    """triage -> billing -> tech のハンドオフ構成を持つフェイク registry を作る。"""
    reg, _ = make_registry()
    reg.register(AgentSpec(name="triage", instructions="t", handoffs=["billing"]))
    reg.register(AgentSpec(name="billing", instructions="b", handoffs=["tech"]))
    reg.register(AgentSpec(name="tech", instructions="x"))
    return reg


def test_next_turn_禁止を宣言しない宣言では素のAgent直appendが維持される() -> None:
    """次ターン指定のみの宣言では合成を設置せず、handoff は従来どおり Agent 実体の直 append。"""
    reg = make_handoff_registry()

    derived = apply_next_turn_policy(
        NextTurnPolicy(rules={"billing": NextTurnRule(next_agent="triage")}), reg
    )

    assert derived.get("triage").handoffs == [derived.get("billing")]
    assert isinstance(derived.get("triage").handoffs[0], FakeAgent)


def test_next_turn_禁止を宣言しても元registryの結線は素のまま() -> None:
    """合成は派生 registry にのみ設置され、元 registry の結線は従来経路のまま。"""
    reg = make_handoff_registry()

    apply_next_turn_policy(
        NextTurnPolicy(rules={"billing": NextTurnRule(no_handoff_on_arrival=True)}), reg
    )

    assert reg.get("triage").handoffs == [reg.get("billing")]
    assert isinstance(reg.get("triage").handoffs[0], FakeAgent)
    assert isinstance(reg.get("billing").handoffs[0], FakeAgent)


def test_next_turn_判定表を持たないregistryのcloneは従来どおり() -> None:
    """合成を設置していない registry の clone は、これまでどおり素の直 append を維持する。"""
    reg = make_handoff_registry()

    cloned = reg.clone()

    assert cloned.get("triage").handoffs == [cloned.get("billing")]
    assert isinstance(cloned.get("triage").handoffs[0], FakeAgent)
