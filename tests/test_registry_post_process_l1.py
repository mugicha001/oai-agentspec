"""L1: `AgentPostProcessor` と registry 構築の第 3 段（post-process）の検証（agents 非依存）。

構築は build（パス 1）-> wire（パス 2）-> post-process（第 3 段）の順に固定される。
spec 経路は全 wire 完了後に各 spec 1 回・in-place で同一オブジェクトを返す契約（違えば
`ValueError`）で、例外時は同じ呼び出しで構築した全 spec を巻き戻す。factory 経路は factory
呼び出し直後に `spec=None` で 1 回呼ばれ、戻り値がキャッシュされる。第 3 段は
`AgentRegistry(post_processor=...)` で明示的に渡したオブジェクトだけを呼び、builder が
`post_process` を持っていても発見しない（未指定なら従来どおり何も差し込まれない）。

sub_agents の as_tool は `_adapters.make_agent_tool` を sentinel を返す関数へ差し替えて、
結線済みであることを `is` で観測する。
"""

from __future__ import annotations

from typing import Any

import pytest

import oai_agentspec
import oai_agentspec.protocols
from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.protocols import AgentBuilder, AgentPostProcessor

from _helpers.fake_builder import FakeAgent, FakeAgentBuilder

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# テスト用フェイク
# ----------------------------------------------------------------------


class _SentinelTool:
    """`make_agent_tool` の代わりに返す as_tool の目印（同一性で比較する）。"""

    def __init__(self, agent: Any, tool_name: str | None, tool_description: str | None) -> None:
        self.agent = agent
        self.tool_name = tool_name
        self.tool_description = tool_description


def _install_sentinel_as_tool(monkeypatch: pytest.MonkeyPatch) -> list[_SentinelTool]:
    """`make_agent_tool` を sentinel を返す関数へ差し替え、生成した sentinel の記録を返す。"""
    made: list[_SentinelTool] = []

    def fake_make_agent_tool(
        agent: Any, *, tool_name: str | None, tool_description: str | None
    ) -> _SentinelTool:
        tool = _SentinelTool(agent, tool_name, tool_description)
        made.append(tool)
        return tool

    monkeypatch.setattr("oai_agentspec._adapters.make_agent_tool", fake_make_agent_tool)
    return made


class _RecordingBuilder(FakeAgentBuilder):
    """build のみを持ち、構築した FakeAgent を名前ごとに記録する builder。

    `log` を渡すと `("build", name)` を追記する（post-processor と共有して順序を観測する）。
    """

    def __init__(self, log: list[tuple[str, str]] | None = None) -> None:
        super().__init__()
        self.agents: dict[str, Any] = {}
        self.log = log

    def build(self, spec: AgentSpec) -> Any:
        agent = super().build(spec)
        self.agents[spec.name] = agent
        if self.log is not None:
            self.log.append(("build", spec.name))
        return agent


class _RecordingPostProcessor:
    """post_process だけを持つ単独の post-processor（builder を継承しない）。

    呼び出しと呼び出し時点の結線状態を記録する。`registry` を設定すると、呼び出し時点で
    `_built` にある全 Agent の tools / handoffs をスナップショットする（全 wire 完了後に
    呼ばれることの観測用）。`log` を渡すと `("post", name)` を追記する。
    """

    def __init__(self, log: list[tuple[str, str]] | None = None) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.snapshots: dict[str, dict[str, tuple[list[Any], list[Any]]]] = {}
        self.registry: AgentRegistry | None = None
        self.log = log

    def post_process(self, agent: Any, *, name: str, spec: AgentSpec | None) -> Any:
        self.calls.append(((agent,), {"name": name, "spec": spec}))
        if self.log is not None:
            self.log.append(("post", name))
        if self.registry is not None:
            self.snapshots[name] = {
                n: (list(a.tools), list(a.handoffs))
                for n, a in self.registry._built.items()  # noqa: SLF001 - 結線状態の観測
            }
        return agent

    def called_names(self) -> list[str]:
        return [kwargs["name"] for _, kwargs in self.calls]


class _ReplacingPostProcessor(_RecordingPostProcessor):
    """`replace_spec` が真の間、spec 経路で別オブジェクトを返す（契約違反）post-processor。"""

    def __init__(self) -> None:
        super().__init__()
        self.replace_spec = True

    def post_process(self, agent: Any, *, name: str, spec: AgentSpec | None) -> Any:
        result = super().post_process(agent, name=name, spec=spec)
        if spec is not None and self.replace_spec:
            return FakeAgent(name=f"{name}-replaced")
        return result


class _RaisingPostProcessor(_RecordingPostProcessor):
    """`fail_on` に一致する名前で RuntimeError を送出する post-processor。"""

    def __init__(self, fail_on: str) -> None:
        super().__init__()
        self.fail_on: str | None = fail_on

    def post_process(self, agent: Any, *, name: str, spec: AgentSpec | None) -> Any:
        result = super().post_process(agent, name=name, spec=spec)
        if name == self.fail_on:
            raise RuntimeError(f"post-process failed: {name}")
        return result


class _FactoryReplacingPostProcessor(_RecordingPostProcessor):
    """factory 経路（spec=None）で新しいインスタンスを返す post-processor。"""

    def __init__(self) -> None:
        super().__init__()
        self.returned: list[Any] = []

    def post_process(self, agent: Any, *, name: str, spec: AgentSpec | None) -> Any:
        result = super().post_process(agent, name=name, spec=spec)
        if spec is None:
            result = FakeAgent(name=f"{name}-processed")
            self.returned.append(result)
        return result


class _BuilderWithPostProcess(_RecordingBuilder):
    """build と post_process の両方を持つ builder（isinstance 発見されないことの確認用）。"""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def post_process(self, agent: Any, *, name: str, spec: AgentSpec | None) -> Any:
        self.calls.append(name)
        return FakeAgent(name=f"{name}-should-not-be-used")


class _BuildOnlyDecorator:
    """build だけを内側の builder へ委譲する装飾 builder（post_process を持たない）。"""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.decorated: list[str] = []

    def build(self, spec: AgentSpec) -> Any:
        self.decorated.append(spec.name)
        return self.inner.build(spec)


class _OnlyPostProcess:
    """post_process だけを持つ最小実装（Protocol 充足の確認用）。"""

    def post_process(self, agent: Any, *, name: str, spec: AgentSpec | None) -> Any:
        return agent


# ----------------------------------------------------------------------
# T1: AgentPostProcessor Protocol
# ----------------------------------------------------------------------


def test_post_process_only_object_satisfies_agent_post_processor() -> None:
    """post_process を持つオブジェクトは AgentPostProcessor を満たす（build は不要）。"""
    assert isinstance(_OnlyPostProcess(), AgentPostProcessor)


def test_builder_with_post_process_is_not_discovered_without_post_processor() -> None:
    """post_process を持つ builder でも、post_processor 未指定なら第 3 段で呼ばれない。

    registry は builder を `isinstance` で post-processor として発見しない（オプトインは
    `post_processor=` の明示的な受け渡しだけ）。spec 経路・factory 経路とも get の戻りは
    builder / factory の戻りと同一で、builder の post_process は 1 度も呼ばれない。
    """
    builder = _BuilderWithPostProcess()
    assert isinstance(builder, AgentBuilder)
    assert isinstance(builder, AgentPostProcessor)
    reg = AgentRegistry(agent_builder=builder)
    reg.register(AgentSpec(name="a", instructions="a"))
    produced = FakeAgent(name="f")
    reg.register_factory("f", lambda _reg: produced)

    assert reg.get("a") is builder.agents["a"]
    assert reg.get("f") is produced
    assert builder.calls == []


def test_post_processor_rejects_non_protocol_object() -> None:
    """AgentPostProcessor を満たさないオブジェクトを渡すと `__init__` で TypeError。

    満たすオブジェクトは受理されることも合わせて確認する（引数そのものが無いことによる
    TypeError と区別する）。
    """
    AgentRegistry(post_processor=_RecordingPostProcessor())
    with pytest.raises(TypeError):
        AgentRegistry(post_processor=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AgentRegistry(post_processor=_RecordingBuilder())  # type: ignore[arg-type]


def test_build_only_fake_builder_is_not_agent_post_processor() -> None:
    """build だけの FakeAgentBuilder は AgentPostProcessor を満たさない。"""
    assert not isinstance(FakeAgentBuilder(), AgentPostProcessor)


def test_agent_post_processor_not_in_core_all() -> None:
    """コア `oai_agentspec.__all__` には AgentPostProcessor を加えない（公開面の不変）。"""
    assert "AgentPostProcessor" not in oai_agentspec.__all__


def test_agent_post_processor_in_protocols_all() -> None:
    """`oai_agentspec.protocols.__all__` に AgentPostProcessor が含まれる。"""
    assert "AgentPostProcessor" in oai_agentspec.protocols.__all__


# ----------------------------------------------------------------------
# T2 (a): post-processor 無しでは従来どおり（既存挙動の pin）
# ----------------------------------------------------------------------


def test_without_post_processor_spec_agent_and_as_tool_are_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """post-processor 無しでは get の戻りが builder の戻りと同一で、as_tool もそのまま入る。"""
    made = _install_sentinel_as_tool(monkeypatch)
    builder = _RecordingBuilder()
    assert not isinstance(builder, AgentPostProcessor)
    reg = AgentRegistry(agent_builder=builder)
    reg.register(AgentSpec(name="worker", instructions="w"))
    reg.register(AgentSpec(name="router", instructions="r", sub_agents=["worker"]))

    router = reg.get("router")

    assert router is builder.agents["router"]
    assert reg.get("worker") is builder.agents["worker"]
    assert len(made) == 1
    assert made[0].agent is builder.agents["worker"]
    assert len(router.tools) == 1
    assert router.tools[0] is made[0]


def test_without_post_processor_factory_result_is_returned_as_is() -> None:
    """post-processor 無しの builder では factory の戻り値がそのまま get の戻りになる。"""
    reg = AgentRegistry(agent_builder=_RecordingBuilder())
    produced = FakeAgent(name="f")
    reg.register_factory("f", lambda _reg: produced)
    assert reg.get("f") is produced
    assert reg.get("f") is produced


def test_factory_only_registry_with_default_builder_returns_factory_result() -> None:
    """agent_builder=None の factory のみ registry でも同一で、既定 builder を遅延生成しない。"""
    reg = AgentRegistry()
    produced = FakeAgent(name="f")
    reg.register_factory("f", lambda _reg: produced)
    assert reg.get("f") is produced
    assert reg.get("f") is produced
    assert reg._agent_builder is None  # noqa: SLF001 - factory 経路で _builder() を呼ばない


# ----------------------------------------------------------------------
# T2 (b): post-processor 付きの呼び出し契約
# ----------------------------------------------------------------------


def test_spec_path_post_process_called_once_per_spec_after_all_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """spec 経路は全 wire 完了後に各 spec 1 回 post_process(agent, name=, spec=) で呼ばれる。"""
    made = _install_sentinel_as_tool(monkeypatch)
    log: list[tuple[str, str]] = []
    builder = _RecordingBuilder(log=log)
    post = _RecordingPostProcessor(log=log)
    reg = AgentRegistry(agent_builder=builder, post_processor=post)
    post.registry = reg
    spec_a = AgentSpec(name="a", instructions="a", handoffs=["b"], sub_agents=["c"])
    spec_b = AgentSpec(name="b", instructions="b", handoffs=["a"])
    spec_c = AgentSpec(name="c", instructions="c")
    for spec in (spec_a, spec_b, spec_c):
        reg.register(spec)

    a = reg.get("a")
    b, c = reg.get("b"), reg.get("c")

    # builder と post-processor の共有ログ: 3 体の build がすべて終わってから第 3 段が始まる。
    assert len(log) == 6
    assert sorted(n for kind, n in log[:3] if kind == "build") == ["a", "b", "c"]
    assert sorted(n for kind, n in log[3:] if kind == "post") == ["a", "b", "c"]

    # 呼び出し記録の全体照合（引数は位置 1 つ + name / spec のキーワード）。
    by_name = {kwargs["name"]: (args, kwargs) for args, kwargs in post.calls}
    assert len(post.calls) == 3
    assert by_name == {
        "a": ((a,), {"name": "a", "spec": spec_a}),
        "b": ((b,), {"name": "b", "spec": spec_b}),
        "c": ((c,), {"name": "c", "spec": spec_c}),
    }
    for name, agent, spec in (("a", a, spec_a), ("b", b, spec_b), ("c", c, spec_c)):
        args, kwargs = by_name[name]
        assert args[0] is agent
        assert kwargs["spec"] is spec

    # どの呼び出しの時点でも、3 体すべての結線（handoffs / as_tool）が完了している。
    assert len(made) == 1
    for name in ("a", "b", "c"):
        snapshot = post.snapshots[name]
        assert set(snapshot) == {"a", "b", "c"}
        a_tools, a_handoffs = snapshot["a"]
        b_tools, b_handoffs = snapshot["b"]
        assert len(a_tools) == 1
        assert a_tools[0] is made[0]
        assert made[0].agent is c
        assert len(a_handoffs) == 1
        assert a_handoffs[0] is b
        assert b_tools == []
        assert len(b_handoffs) == 1
        assert b_handoffs[0] is a


def test_factory_path_post_process_called_once_with_spec_none() -> None:
    """factory 経路は post_process(agent, name=登録名, spec=None) で 1 回呼ばれる。"""
    post = _RecordingPostProcessor()
    reg = AgentRegistry(agent_builder=_RecordingBuilder(), post_processor=post)
    produced = FakeAgent(name="inner")
    reg.register_factory("svc", lambda _reg: produced)

    assert reg.get("svc") is produced
    assert reg.get("svc") is produced

    assert post.calls == [((produced,), {"name": "svc", "spec": None})]
    assert post.calls[0][0][0] is produced


def test_factory_path_new_instance_from_post_process_is_cached() -> None:
    """factory 経路で post_process が別オブジェクトを返すと、それが get の戻り・キャッシュ。"""
    post = _FactoryReplacingPostProcessor()
    reg = AgentRegistry(agent_builder=_RecordingBuilder(), post_processor=post)
    produced = FakeAgent(name="inner")
    reg.register_factory("svc", lambda _reg: produced)

    first = reg.get("svc")
    second = reg.get("svc")

    assert len(post.returned) == 1
    assert first is post.returned[0]
    assert first is not produced
    assert second is first
    assert post.calls == [((produced,), {"name": "svc", "spec": None})]
    assert post.calls[0][0][0] is produced


# ----------------------------------------------------------------------
# T2 (c)(d): spec 経路の失敗と巻き戻し
# ----------------------------------------------------------------------


def test_spec_path_returning_different_object_raises_and_rolls_back() -> None:
    """spec 経路で別オブジェクトを返すと ValueError になり、構築分が巻き戻される。"""
    builder = _RecordingBuilder()
    post = _ReplacingPostProcessor()
    reg = AgentRegistry(agent_builder=builder, post_processor=post)
    reg.register(AgentSpec(name="a", instructions="a", handoffs=["b"]))
    reg.register(AgentSpec(name="b", instructions="b"))

    with pytest.raises(ValueError):
        reg.get("a")
    assert post.calls != []
    assert "a" not in reg._built  # noqa: SLF001 - 残留しないことの検証
    assert "b" not in reg._built  # noqa: SLF001 - 同上

    # 契約を守る挙動へ直すと、再試行で再構築される。
    post.replace_spec = False
    a = reg.get("a")
    assert a is builder.agents["a"]
    assert a.handoffs[0] is reg.get("b")
    assert builder.built == ["a", "b", "a", "b"]


def test_spec_path_exception_in_post_process_rolls_back_all_newly_built() -> None:
    """第 3 段の例外で、同じ呼び出しで構築した全 spec が巻き戻される（既構築は残る）。"""
    post = _RaisingPostProcessor(fail_on="b")
    reg = AgentRegistry(agent_builder=_RecordingBuilder(), post_processor=post)
    reg.register(AgentSpec(name="a", instructions="a", handoffs=["b", "pre"]))
    reg.register(AgentSpec(name="b", instructions="b"))
    reg.register(AgentSpec(name="pre", instructions="p"))
    pre = reg.get("pre")

    with pytest.raises(RuntimeError, match="post-process failed: b"):
        reg.get("a")

    assert set(post.called_names()) >= {"pre", "b"}
    assert "a" not in reg._built  # noqa: SLF001 - 残留しないことの検証
    assert "b" not in reg._built  # noqa: SLF001 - 同上
    assert reg._built == {"pre": pre}  # noqa: SLF001 - 既構築は巻き戻さない

    post.fail_on = None
    a = reg.get("a")
    assert a.handoffs[0] is reg.get("b")
    assert a.handoffs[1] is pre


# ----------------------------------------------------------------------
# T2 (e)(f)(g): 循環・二重適用の防止・第 3 段の位置
# ----------------------------------------------------------------------


def test_cyclic_handoff_identity_preserved_with_post_processor() -> None:
    """循環 a->b->a でも post-process 後に handoffs の同一性が保たれ、各 1 回だけ呼ばれる。"""
    post = _RecordingPostProcessor()
    reg = AgentRegistry(agent_builder=_RecordingBuilder(), post_processor=post)
    reg.register(AgentSpec(name="a", instructions="a", handoffs=["b"]))
    reg.register(AgentSpec(name="b", instructions="b", handoffs=["a"]))

    a = reg.get("a")
    b = reg.get("b")

    assert a.handoffs[0] is b
    assert b.handoffs[0] is a
    assert sorted(post.called_names()) == ["a", "b"]


def test_post_process_not_repeated_on_cached_get() -> None:
    """2 回目の get（キャッシュ返却）では post_process を再度呼ばない。"""
    post = _RecordingPostProcessor()
    reg = AgentRegistry(agent_builder=_RecordingBuilder(), post_processor=post)
    reg.register(AgentSpec(name="a", instructions="a"))

    first = reg.get("a")
    second = reg.get("a")

    assert second is first
    assert post.called_names() == ["a"]


def test_post_process_not_repeated_for_already_built_dependency() -> None:
    """先に構築済みの依存先（b）は、後の get("a") の構築で再度 post-process されない。"""
    post = _RecordingPostProcessor()
    reg = AgentRegistry(agent_builder=_RecordingBuilder(), post_processor=post)
    reg.register(AgentSpec(name="a", instructions="a", handoffs=["b"]))
    reg.register(AgentSpec(name="b", instructions="b"))

    b = reg.get("b")
    assert post.called_names() == ["b"]
    a = reg.get("a")

    assert post.called_names() == ["b", "a"]
    assert a.handoffs[0] is b


def test_post_process_runs_after_sub_agent_as_tool_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """a の post_process 時点で、sub_agent b の as_tool が a.tools に結線済みである。"""
    made = _install_sentinel_as_tool(monkeypatch)
    post = _RecordingPostProcessor()
    reg = AgentRegistry(agent_builder=_RecordingBuilder(), post_processor=post)
    post.registry = reg
    reg.register(AgentSpec(name="a", instructions="a", sub_agents=["b"]))
    reg.register(AgentSpec(name="b", instructions="b"))

    a = reg.get("a")

    assert "a" in post.snapshots
    a_tools_at_call, _ = post.snapshots["a"]["a"]
    assert len(made) == 1
    assert len(a_tools_at_call) == 1
    assert a_tools_at_call[0] is made[0]
    assert made[0].agent is reg.get("b")
    assert a.tools[0] is made[0]


# ----------------------------------------------------------------------
# clone 継承・装飾 builder の下での呼び出し
# ----------------------------------------------------------------------


def test_clone_inherits_post_processor() -> None:
    """clone は post_processor を共有継承し、clone 側の get で clone の Agent に対して呼ばれる。"""
    post = _RecordingPostProcessor()
    reg = AgentRegistry(agent_builder=_RecordingBuilder(), post_processor=post)
    spec_a = AgentSpec(name="a", instructions="a")
    reg.register(spec_a)
    produced = FakeAgent(name="f")
    reg.register_factory("f", lambda _reg: produced)

    cloned = reg.clone()
    a = cloned.get("a")
    f = cloned.get("f")

    assert f is produced
    assert len(post.calls) == 2
    (a_args, a_kwargs), (f_args, f_kwargs) = post.calls
    assert a_args[0] is a
    assert a_kwargs["name"] == "a"
    assert a_kwargs["spec"] is not None
    assert a_kwargs["spec"].name == "a"
    assert a_kwargs["spec"] is not spec_a  # clone は spec を独立コピーする
    assert f_args[0] is produced
    assert f_kwargs == {"name": "f", "spec": None}
    assert reg._built == {}  # noqa: SLF001 - 元 registry は構築されていない


def test_post_processor_called_under_decorating_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build だけ委譲する装飾 builder の下でも、明示的に渡した post_processor が呼ばれる。"""
    made = _install_sentinel_as_tool(monkeypatch)
    inner = _RecordingBuilder()
    decorator = _BuildOnlyDecorator(inner)
    assert not isinstance(decorator, AgentPostProcessor)
    post = _RecordingPostProcessor()
    reg = AgentRegistry(agent_builder=decorator, post_processor=post)
    post.registry = reg
    reg.register(AgentSpec(name="a", instructions="a", sub_agents=["b"]))
    reg.register(AgentSpec(name="b", instructions="b"))
    produced = FakeAgent(name="f")
    reg.register_factory("f", lambda _reg: produced)

    a = reg.get("a")
    f = reg.get("f")

    assert sorted(decorator.decorated) == ["a", "b"]
    assert a is inner.agents["a"]
    assert f is produced
    assert sorted(post.called_names()) == ["a", "b", "f"]
    a_tools_at_call, _ = post.snapshots["a"]["a"]
    assert len(made) == 1
    assert a_tools_at_call == [made[0]]
    assert a_tools_at_call[0] is made[0]
