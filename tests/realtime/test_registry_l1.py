"""L1: RealtimeAgentRegistry のロジック検証（agents 非依存・FakeRealtimeAgentBuilder 注入）。

登録 / 遅延構築 / 循環 handoff の遅延バインド解決 / 重複登録エラー / 未登録参照の validate エラー /
ビルド失敗時の巻き戻しを、本物の RealtimeAgent を作らずに検証する。registry は未実装のため本
モジュールの import は collection error（RED）になる想定。
"""

from __future__ import annotations

import pytest

from oai_agentspec.realtime.protocols import RealtimeAgentBuilder
from oai_agentspec.realtime.registry import RealtimeAgentRegistry
from oai_agentspec.realtime.spec import RealtimeAgentSpec, RealtimeHandoffConfig

from _helpers.fake_realtime_builder import FakeRealtimeAgentBuilder


def make_registry() -> tuple[RealtimeAgentRegistry, FakeRealtimeAgentBuilder]:
    builder = FakeRealtimeAgentBuilder()
    return RealtimeAgentRegistry(agent_builder=builder), builder


def test_fake_builder_satisfies_realtime_agent_builder_protocol() -> None:
    """DI 拡張点 RealtimeAgentBuilder（runtime_checkable Protocol）への構造的適合を担保する。"""
    assert isinstance(FakeRealtimeAgentBuilder(), RealtimeAgentBuilder)


def test_register_duplicate_raises() -> None:
    """同名の再登録は ValueError（already registered）で弾く。"""
    reg, _ = make_registry()
    reg.register(RealtimeAgentSpec(name="a", instructions="x"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(RealtimeAgentSpec(name="a", instructions="y"))


def test_lazy_build_only_on_get() -> None:
    """build は register 時ではなく get 時に初めて走る（遅延構築）。"""
    reg, builder = make_registry()
    reg.register(RealtimeAgentSpec(name="a", instructions="x"))
    assert builder.built == []
    reg.get("a")
    assert builder.built == ["a"]


def test_get_unknown_raises() -> None:
    """未登録名の get は KeyError。"""
    reg, _ = make_registry()
    with pytest.raises(KeyError):
        reg.get("missing")


def test_instructions_passthrough() -> None:
    """instructions は構築物にそのまま渡る。"""
    reg, _ = make_registry()
    reg.register(RealtimeAgentSpec(name="a", instructions="hello"))
    assert reg.get("a").instructions == "hello"


def test_callable_instructions_arity() -> None:
    """callable instructions は (context, agent) の 2 引数で呼び出せない場合 ValueError。"""
    reg, _ = make_registry()
    with pytest.raises(ValueError, match="2 引数"):
        reg.register(RealtimeAgentSpec(name="a", instructions=lambda ctx: "x"))
    reg.register(RealtimeAgentSpec(name="b", instructions=lambda ctx, agent: "x"))


def test_callable_instructions_accepts_two_arg_callable_variants() -> None:
    """2 引数で呼び出せる callable（デフォルト引数付き / 可変長）は register を通過する。"""
    reg, _ = make_registry()
    reg.register(RealtimeAgentSpec(name="a", instructions=lambda ctx, agent, cfg=None: "x"))
    reg.register(RealtimeAgentSpec(name="b", instructions=lambda *args: "x"))


def test_callable_instructions_skips_unintrospectable_callable() -> None:
    """シグネチャ取得不能な callable（builtin 等）は検証をスキップし register を通過する。"""
    reg, _ = make_registry()
    reg.register(RealtimeAgentSpec(name="a", instructions=zip))


def test_register_rejects_callable_prompt() -> None:
    """prompt に callable（DynamicPromptFunction）を渡すと register 時に ValueError。"""
    reg, _ = make_registry()
    with pytest.raises(ValueError, match=r"a.*prompt"):
        reg.register(RealtimeAgentSpec(name="a", instructions="x", prompt=lambda c, a: "p"))


def test_register_rejects_unknown_handoff_options_key() -> None:
    """handoffs に存在しないキーの handoff_options は register 時に ValueError。

    タイポによる per-edge 設定の silent drop（デフォルト handoff への置換）を防ぐ。
    """
    reg, _ = make_registry()
    with pytest.raises(ValueError, match=r"a.*suport.*handoffs"):
        reg.register(
            RealtimeAgentSpec(
                name="a",
                instructions="x",
                handoffs=["support"],
                handoff_options={"suport": RealtimeHandoffConfig()},
            )
        )


def test_register_rejects_wrong_on_handoff_arity() -> None:
    """on_handoff の引数個数が SDK 契約と合わない場合 register 時に ValueError。

    input_type ありは (context, input) の 2 引数、なしは (context) の 1 引数
    （SDK realtime_handoff() の厳格検査を register 時に前倒し）。
    """
    reg, _ = make_registry()
    with pytest.raises(ValueError, match=r"'a' -> 'b'.*1 引数.*2 引数"):
        reg.register(
            RealtimeAgentSpec(
                name="a",
                instructions="x",
                handoffs=["b"],
                handoff_options={"b": RealtimeHandoffConfig(on_handoff=lambda c, i: None)},
            )
        )
    with pytest.raises(ValueError, match=r"'c' -> 'd'.*2 引数.*1 引数"):
        reg.register(
            RealtimeAgentSpec(
                name="c",
                instructions="x",
                handoffs=["d"],
                handoff_options={
                    "d": RealtimeHandoffConfig(input_type=object, on_handoff=lambda c: None)
                },
            )
        )


def test_register_accepts_correct_on_handoff_arity() -> None:
    """SDK 契約どおりの on_handoff（input_type なし 1 / あり 2 引数）は register を通過する。"""
    reg, _ = make_registry()
    reg.register(
        RealtimeAgentSpec(
            name="a",
            instructions="x",
            handoffs=["b"],
            handoff_options={"b": RealtimeHandoffConfig(on_handoff=lambda c: None)},
        )
    )
    reg.register(
        RealtimeAgentSpec(
            name="c",
            instructions="x",
            handoffs=["d"],
            handoff_options={
                "d": RealtimeHandoffConfig(input_type=object, on_handoff=lambda c, i: None)
            },
        )
    )


def test_register_rejects_input_type_without_on_handoff() -> None:
    """handoff_options の input_type 指定に on_handoff が伴わない場合 register 時に ValueError。

    SDK realtime_handoff() の必須制約を、get() 時の文脈なし UserError ではなく
    register 時のエージェント名・エッジ名入りエラーへ前倒しする。
    """
    reg, _ = make_registry()
    with pytest.raises(ValueError, match=r"'a' -> 'b'.*on_handoff"):
        reg.register(
            RealtimeAgentSpec(
                name="a",
                instructions="x",
                handoffs=["b"],
                handoff_options={"b": RealtimeHandoffConfig(input_type=object)},
            )
        )


def test_names_sorted() -> None:
    """names() は登録名を昇順で返す。"""
    reg, _ = make_registry()
    reg.register(RealtimeAgentSpec(name="zeta", instructions="z"))
    reg.register(RealtimeAgentSpec(name="alpha", instructions="a"))
    assert reg.names() == ["alpha", "zeta"]


def test_entry_name_is_first_registered() -> None:
    """entry_name は登録順の先頭を返す（names() の昇順とは独立）。"""
    reg, _ = make_registry()
    assert reg.entry_name is None
    reg.register(RealtimeAgentSpec(name="zeta", instructions="z"))
    reg.register(RealtimeAgentSpec(name="alpha", instructions="a"))
    assert reg.entry_name == "zeta"


def test_registration_order_independent_handoff_wiring() -> None:
    """後で登録する handoff 先も遅延バインドで解決され、handoff に結線される。"""
    reg, _ = make_registry()
    reg.register(RealtimeAgentSpec(name="parent", instructions="p", handoffs=["child"]))
    reg.register(RealtimeAgentSpec(name="child", instructions="c"))
    parent = reg.get("parent")
    assert parent.handoffs[0].target is reg.get("child")


def test_cyclic_handoff_identity() -> None:
    """相互参照（a<->b）の循環 handoff が遅延バインドで同一インスタンスへ解決される。"""
    reg, _ = make_registry()
    reg.register(RealtimeAgentSpec(name="a", instructions="a", handoffs=["b"]))
    reg.register(RealtimeAgentSpec(name="b", instructions="b", handoffs=["a"]))
    a = reg.get("a")
    b = reg.get("b")
    assert a.handoffs[0].target is b
    assert b.handoffs[0].target is a


def test_three_way_cycle() -> None:
    """3 者循環（a->b->c->a）も遅延バインドで解決される。"""
    reg, _ = make_registry()
    reg.register(RealtimeAgentSpec(name="a", instructions="a", handoffs=["b"]))
    reg.register(RealtimeAgentSpec(name="b", instructions="b", handoffs=["c"]))
    reg.register(RealtimeAgentSpec(name="c", instructions="c", handoffs=["a"]))
    a, b, c = reg.get("a"), reg.get("b"), reg.get("c")
    assert a.handoffs[0].target is b
    assert b.handoffs[0].target is c
    assert c.handoffs[0].target is a


def test_handoff_option_passed_to_make_handoff() -> None:
    """handoff_options のエッジ設定が make_handoff（結線）へ渡る。"""
    reg, _ = make_registry()
    cfg = RealtimeHandoffConfig(tool_name_override="go_child")
    reg.register(
        RealtimeAgentSpec(
            name="parent",
            instructions="p",
            handoffs=["child"],
            handoff_options={"child": cfg},
        )
    )
    reg.register(RealtimeAgentSpec(name="child", instructions="c"))
    parent = reg.get("parent")
    assert parent.handoffs[0].config is cfg


def test_unreachable_spec_not_built() -> None:
    """handoffs から到達しない spec は get 時に build されない。"""
    reg, builder = make_registry()
    reg.register(RealtimeAgentSpec(name="a", instructions="a"))
    reg.register(RealtimeAgentSpec(name="orphan", instructions="o"))
    reg.get("a")
    assert "orphan" not in builder.built


def test_unregistered_handoff_dst_errors() -> None:
    """未登録の handoff 先を持つ spec の get は KeyError（該当名を含む）。"""
    reg, _ = make_registry()
    reg.register(RealtimeAgentSpec(name="a", instructions="a", handoffs=["ghost"]))
    with pytest.raises(KeyError, match="ghost"):
        reg.get("a")


def test_wiring_failure_rolls_back_partial_build() -> None:
    """結線失敗時に部分ビルドを巻き戻し、依存を後付け登録すれば再度 get が成功する。"""
    reg, _ = make_registry()
    reg.register(RealtimeAgentSpec(name="a", instructions="a", handoffs=["late"]))
    with pytest.raises(KeyError):
        reg.get("a")
    # 巻き戻しにより "a" は built キャッシュに残らない。
    assert "a" not in reg._built  # noqa: SLF001 - 巻き戻し検証
    reg.register(RealtimeAgentSpec(name="late", instructions="late"))
    a = reg.get("a")
    assert a.handoffs[0].target is reg.get("late")


def test_validate_detects_missing_references() -> None:
    """validate は未解決の handoff 参照を検出して KeyError にする。"""
    reg, _ = make_registry()
    reg.register(RealtimeAgentSpec(name="a", instructions="a", handoffs=["ghost"]))
    with pytest.raises(KeyError, match="ghost"):
        reg.validate()


def test_validate_passes_when_all_resolved() -> None:
    """全 handoff 参照が既知名なら validate は例外を出さない。"""
    reg, _ = make_registry()
    reg.register(RealtimeAgentSpec(name="a", instructions="a", handoffs=["b"]))
    reg.register(RealtimeAgentSpec(name="b", instructions="b"))
    reg.validate()


# ------------------------------------------------------------------
# PromptStore の動的合成との互換性
# ------------------------------------------------------------------
def test_promptstore_の動的合成callableはinstructionsとして通る(tmp_path) -> None:
    """PromptStore.compose(vars=callable) の戻り値（(context, agent) -> str）が
    RealtimeAgentSpec.instructions として register 検証を通過し、構築物へそのまま渡る。

    Realtime session は callable instructions を get_system_prompt(context) で解決する
    ため、PromptStore のテンプレート合成（動的変数注入）は Realtime でも機能する。
    """
    from types import SimpleNamespace

    from oai_agentspec import PromptLayout, PromptStore

    (tmp_path / "base").mkdir()
    (tmp_path / "parts").mkdir()
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "triage.md").write_text("あなたは ${user} 担当の受付。", encoding="utf-8")
    store = PromptStore(tmp_path, PromptLayout(base="base", parts="parts", agents="agents"))

    instructions = store.compose(agent="triage", vars=lambda ctx: {"user": ctx.context.user})
    assert callable(instructions)

    reg, builder = make_registry()
    reg.register(RealtimeAgentSpec(name="triage", instructions=instructions))
    agent = reg.get("triage")
    assert agent.instructions is instructions

    # SDK と同じ呼び出し形（(context, agent) の 2 位置引数）で解決できることを確認する。
    ctx = SimpleNamespace(context=SimpleNamespace(user="mugicha"))
    assert agent.instructions(ctx, agent) == "あなたは mugicha 担当の受付。"
