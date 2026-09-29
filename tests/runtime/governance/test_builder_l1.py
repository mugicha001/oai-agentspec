"""L1: `GovernedAgentBuilder` の装飾ロジック検証（fake 注入・Runner 実行非依存）。

`build(spec)` が govern 済み spec（tools 差し替え + hooks 合成）を `inner.build` へ渡すこと、
`policy` / `audit_sink` が `govern_spec` へ素通しされること、`inner=None` で `DefaultAgentBuilder`
が使われること、既定 sink の生成・共有（`audit_sink` プロパティ）、extra 未導入時の挙動
（コンストラクトは成功し build で install hint 付き ImportError）を検証する。

`GovernedAgentBuilder` は `AgentBuilder` であり `AgentPostProcessor` ではない。registry 第 3 段の
統治は `builder.post_processor(sub_agent_tools=..., factory_agents=...)` が返す post-processor を
`AgentRegistry(post_processor=...)` へ明示的に渡したときだけ走る。その `post_process(agent,
name=..., spec=...)` は spec 経路では `sub_agent_tools=True` のときだけ `govern_ungoverned_tools`、
factory 経路（`spec=None`）では `factory_agents=True` のときだけ `govern_agent` を呼び、それ以外は
受け取った Agent を返す。post-processor は builder のポリシースナップショット・override・既定
sink・適用記録を共有する（`builder.audit_sink` / `builder.unapplied_overrides` に反映される）。
両フラグ False の `post_processor()` は ValueError。

builder は `build` 内で `from ..._adapters import ...` する（関数内遅延 import）ため、monkeypatch
対象は使用箇所パス `oai_agentspec._adapters.*`（`govern_spec` / `new_audit_sink` /
`DefaultAgentBuilder`、post-processor 系は `govern_ungoverned_tools` / `govern_agent` /
`resolve_policy` も）。AGT 実依存が要るのは実 `govern_spec` を通すテストのみで、`agt_symbols`
フィクスチャ（conftest）で extra 未導入環境では skip する。
"""

from __future__ import annotations

from typing import Any

import pytest

from oai_agentspec import AgentRegistry, AgentSpec, function_tool
from oai_agentspec._adapters.governance import _GOVERNANCE_INSTALL_HINT
from oai_agentspec.protocols import AgentBuilder, AgentPostProcessor
from oai_agentspec.runtime.governance import GovernedAgentBuilder

pytestmark = pytest.mark.unit


class _RecordingBuilder:
    """`build(spec)` 呼び出しの spec を記録する fake inner builder。"""

    def __init__(self) -> None:
        """記録リストを初期化する。"""
        self.specs: list[Any] = []

    def build(self, spec: AgentSpec) -> Any:
        """spec を記録し、識別可能なダミー Agent を返す（spec の属性には依存しない）。"""
        self.specs.append(spec)
        return ("agent", getattr(spec, "name", None))


def _make_tool(name: str = "echo") -> Any:
    """実 `FunctionTool` を 1 つ作る（govern ラップ対象・実行はしない）。"""

    @function_tool(name_override=name)
    def _tool(text: str) -> str:
        """エコーする。"""
        return text

    return _tool


# ----------------------------------------------------------------------
# Protocol 適合（AgentBuilder / AgentPostProcessor）
# ----------------------------------------------------------------------


def test_governed_builder_satisfies_agent_builder_protocol() -> None:
    """builder は `AgentBuilder` であり `AgentPostProcessor` ではない。

    `post_processor(sub_agent_tools=True)` の戻りが `AgentPostProcessor` を満たす。
    """
    builder = GovernedAgentBuilder(policy=object())
    assert isinstance(builder, AgentBuilder)
    assert not isinstance(builder, AgentPostProcessor)
    assert isinstance(builder.post_processor(sub_agent_tools=True), AgentPostProcessor)


def test_registry_rejects_governed_builder_as_post_processor() -> None:
    """`AgentRegistry(post_processor=<GovernedAgentBuilder>)` は TypeError（一番起きやすい誤用）。

    同じ builder から作った post-processor は受理されることも確認する（引数の欠如による
    TypeError と区別する）。
    """
    builder = GovernedAgentBuilder(policy=object())
    AgentRegistry(agent_builder=builder, post_processor=builder.post_processor(factory_agents=True))
    with pytest.raises(TypeError):
        AgentRegistry(agent_builder=builder, post_processor=builder)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"sub_agent_tools": False, "factory_agents": False}],
    ids=["omitted", "both_false"],
)
def test_post_processor_requires_at_least_one_flag(kwargs: dict[str, bool]) -> None:
    """`post_processor()` は両フラグ False（省略を含む）で ValueError（渡す = オプトイン）。"""
    builder = GovernedAgentBuilder(policy=object())
    with pytest.raises(ValueError):
        builder.post_processor(**kwargs)


# ----------------------------------------------------------------------
# build: govern 済み spec が inner へ渡る（実 govern_spec・AGT 必要）
# ----------------------------------------------------------------------


def test_build_passes_governed_spec_to_inner(
    agt_symbols: tuple[Any, Any, Any],  # noqa: ARG001 - extra 未導入時 skip のためのみ使用
    allow_all_policy: Any,
    recording_sink: Any,
) -> None:
    """inner へは tools 差し替え + 監査 hooks 合成済みの新 spec が渡る（元 spec 不変）。"""
    tool = _make_tool()
    spec = AgentSpec(name="bot", instructions="i", tools=[tool])
    inner = _RecordingBuilder()
    builder = GovernedAgentBuilder(policy=allow_all_policy, audit_sink=recording_sink, inner=inner)

    result = builder.build(spec)

    # inner.build の戻り値がそのまま返る。
    assert result == ("agent", "bot")
    assert len(inner.specs) == 1
    governed = inner.specs[0]
    # 新 spec（非破壊置換）で、宣言メタは不変。
    assert governed is not spec
    assert governed.name == "bot"
    assert governed.instructions == "i"
    # tools: 同名 FunctionTool だが on_invoke_tool が差し替わった新オブジェクト。
    assert governed.tools[0] is not tool
    assert governed.tools[0].name == tool.name
    assert governed.tools[0].on_invoke_tool is not tool.on_invoke_tool
    # hooks: spec.hooks=None でも監査フックが装着される。
    assert governed.hooks is not None
    # 元 spec は不変。
    assert spec.tools == [tool]
    assert spec.hooks is None


# ----------------------------------------------------------------------
# build: policy / audit_sink の govern_spec への素通し（fake govern_spec）
# ----------------------------------------------------------------------


def test_policy_and_audit_sink_passed_to_govern_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """利用者指定の policy / audit_sink が不透明値のまま `govern_spec` へ渡る。"""
    seen: dict[str, Any] = {}
    sentinel_governed = object()

    def _fake_govern_spec(spec: Any, *, policy: Any, audit_sink: Any = None) -> Any:
        seen.update(spec=spec, policy=policy, audit_sink=audit_sink)
        return sentinel_governed

    monkeypatch.setattr("oai_agentspec._adapters.govern_spec", _fake_govern_spec)
    sentinel_policy = object()
    sentinel_sink = object()
    inner = _RecordingBuilder()
    builder = GovernedAgentBuilder(policy=sentinel_policy, audit_sink=sentinel_sink, inner=inner)
    spec = AgentSpec(name="a", instructions="x")

    builder.build(spec)

    assert seen["spec"] is spec
    assert seen["policy"] is sentinel_policy
    assert seen["audit_sink"] is sentinel_sink
    # govern_spec の戻り値（govern 済み spec）が inner へ渡る。
    assert inner.specs == [sentinel_governed]


def test_inner_none_uses_default_agent_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """`inner=None` のとき `_adapters` の `DefaultAgentBuilder` で Agent 化される。"""
    created: list[Any] = []
    sentinel_governed = object()

    class _FakeDefaultBuilder:
        def __init__(self) -> None:
            created.append(self)
            self.specs: list[Any] = []

        def build(self, spec: Any) -> str:
            self.specs.append(spec)
            return "built-by-default"

    monkeypatch.setattr("oai_agentspec._adapters.DefaultAgentBuilder", _FakeDefaultBuilder)
    monkeypatch.setattr(
        "oai_agentspec._adapters.govern_spec",
        lambda spec, *, policy, audit_sink=None: sentinel_governed,
    )
    builder = GovernedAgentBuilder(policy=object(), audit_sink=object())

    result = builder.build(AgentSpec(name="a", instructions="x"))

    assert result == "built-by-default"
    assert len(created) == 1
    assert created[0].specs == [sentinel_governed]


# ----------------------------------------------------------------------
# audit_sink プロパティ: build 前の値・既定 sink の生成と共有
# ----------------------------------------------------------------------


def test_audit_sink_property_before_build_returns_user_value_or_none() -> None:
    """build 前は利用者指定の sink、未指定なら None を返す（AGT は import しない）。"""
    sentinel_sink = object()
    assert GovernedAgentBuilder(policy=object(), audit_sink=sentinel_sink).audit_sink is (
        sentinel_sink
    )
    assert GovernedAgentBuilder(policy=object()).audit_sink is None


def test_default_sink_created_on_first_build_and_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    """`audit_sink=None` の既定 sink は初回 build で 1 度だけ生成され以降共有される。"""
    sentinel_sink = object()
    new_sink_calls: list[None] = []
    sinks_seen: list[Any] = []

    def _fake_new_audit_sink() -> Any:
        new_sink_calls.append(None)
        return sentinel_sink

    def _fake_govern_spec(spec: Any, *, policy: Any, audit_sink: Any = None) -> Any:
        sinks_seen.append(audit_sink)
        return spec

    monkeypatch.setattr("oai_agentspec._adapters.new_audit_sink", _fake_new_audit_sink)
    monkeypatch.setattr("oai_agentspec._adapters.govern_spec", _fake_govern_spec)
    builder = GovernedAgentBuilder(policy=object(), inner=_RecordingBuilder())

    builder.build(AgentSpec(name="a", instructions="x"))
    builder.build(AgentSpec(name="b", instructions="x"))

    # 生成は初回 build の 1 回のみ・両 build で同一 sink が govern_spec へ渡る（チェーン連続）。
    assert len(new_sink_calls) == 1
    assert sinks_seen == [sentinel_sink, sentinel_sink]
    assert builder.audit_sink is sentinel_sink


def test_explicit_audit_sink_skips_default_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """利用者指定 sink があるときは既定 sink を生成しない（`new_audit_sink` 非呼出）。"""
    sinks_seen: list[Any] = []

    def _fail_new_audit_sink() -> Any:
        raise AssertionError("audit_sink 指定時に new_audit_sink が呼ばれた")

    monkeypatch.setattr("oai_agentspec._adapters.new_audit_sink", _fail_new_audit_sink)
    monkeypatch.setattr(
        "oai_agentspec._adapters.govern_spec",
        lambda spec, *, policy, audit_sink=None: sinks_seen.append(audit_sink) or spec,
    )
    sentinel_sink = object()
    builder = GovernedAgentBuilder(
        policy=object(), audit_sink=sentinel_sink, inner=_RecordingBuilder()
    )

    builder.build(AgentSpec(name="a", instructions="x"))

    assert sinks_seen == [sentinel_sink]
    assert builder.audit_sink is sentinel_sink


# ----------------------------------------------------------------------
# extra 未導入耐性: コンストラクトは成功し、build で install hint 付き ImportError
# ----------------------------------------------------------------------


def test_constructor_does_not_require_agt_and_build_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AGT 未導入相当（`_require_agt` 失敗）でも構築でき、build で案内付き ImportError。"""

    def _raise_import_error() -> Any:
        raise ImportError(_GOVERNANCE_INSTALL_HINT)

    monkeypatch.setattr("oai_agentspec._adapters.governance._require_agt", _raise_import_error)
    # __init__ は AGT を import しない（extra 未導入でもコンストラクト可能）。
    builder = GovernedAgentBuilder(policy=object())
    assert builder.audit_sink is None
    # build（既定 sink 生成 = new_audit_sink → _require_agt）で初めて失敗する。
    with pytest.raises(ImportError, match=r"oai-agentspec\[governance\]"):
        builder.build(AgentSpec(name="a", instructions="x"))


# ----------------------------------------------------------------------
# overrides: per-agent ポリシーの選択・フォールバック・未適用キー検知
# ----------------------------------------------------------------------


def _patch_govern_spec_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, Any]]:
    """`govern_spec` を記録 fake に差し替え、(spec 名, policy) の適用履歴を返す。"""
    applied: list[tuple[str, Any]] = []

    def _fake_govern_spec(spec: Any, *, policy: Any, audit_sink: Any = None) -> Any:
        applied.append((spec.name, policy))
        return spec

    monkeypatch.setattr("oai_agentspec._adapters.govern_spec", _fake_govern_spec)
    monkeypatch.setattr("oai_agentspec._adapters.new_audit_sink", lambda: object())
    return applied


def test_override_policy_selected_for_listed_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """overrides 掲載エージェントは override ポリシー・未掲載は既定へフォールバックする。"""
    applied = _patch_govern_spec_recorder(monkeypatch)
    default_policy = object()
    support_policy = object()
    builder = GovernedAgentBuilder(
        policy=default_policy,
        overrides={"support": support_policy},
        inner=_RecordingBuilder(),
    )

    builder.build(AgentSpec(name="triage", instructions="x"))
    builder.build(AgentSpec(name="support", instructions="x"))

    assert applied == [("triage", default_policy), ("support", support_policy)]


def test_override_key_matching_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """overrides キーの引き当ては `spec.name` との完全一致のみ（正規化なし）。"""
    applied = _patch_govern_spec_recorder(monkeypatch)
    default_policy = object()
    override_policy = object()
    builder = GovernedAgentBuilder(
        policy=default_policy,
        overrides={"Support": override_policy},  # 大文字始まり（不一致）
        inner=_RecordingBuilder(),
    )

    builder.build(AgentSpec(name="support", instructions="x"))

    # 大文字小文字は正規化されず既定へフォールバックする。
    assert applied == [("support", default_policy)]
    assert builder.unapplied_overrides == frozenset({"Support"})


def test_unapplied_overrides_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """`unapplied_overrides` は build 前=全キー・適用後に減少・typo キーは残留する。"""
    _patch_govern_spec_recorder(monkeypatch)
    builder = GovernedAgentBuilder(
        policy=object(),
        overrides={"support": object(), "suport": object()},  # "suport" は typo 相当
        inner=_RecordingBuilder(),
    )

    # build 前は全キーが未適用。
    assert builder.unapplied_overrides == frozenset({"support", "suport"})

    builder.build(AgentSpec(name="support", instructions="x"))
    builder.build(AgentSpec(name="triage", instructions="x"))

    # 適用済みキーは除かれ、typo キーのみ残る（検知の根拠）。
    assert builder.unapplied_overrides == frozenset({"suport"})


def test_unapplied_overrides_empty_without_overrides() -> None:
    """overrides 未指定なら `unapplied_overrides` は空集合（既存利用と完全互換）。"""
    assert GovernedAgentBuilder(policy=object()).unapplied_overrides == frozenset()


def test_overrides_mapping_is_copied(monkeypatch: pytest.MonkeyPatch) -> None:
    """渡した overrides を後から書き換えても builder の引き当てに影響しない（防御的コピー）。"""
    applied = _patch_govern_spec_recorder(monkeypatch)
    default_policy = object()
    override_policy = object()
    mapping: dict[str, Any] = {"bot": override_policy}
    builder = GovernedAgentBuilder(
        policy=default_policy, overrides=mapping, inner=_RecordingBuilder()
    )

    mapping.clear()  # 外部で書き換え
    builder.build(AgentSpec(name="bot", instructions="x"))

    assert applied == [("bot", override_policy)]


# ----------------------------------------------------------------------
# from_yaml: extra 未導入相当では install hint 付き ImportError
# ----------------------------------------------------------------------


def test_from_yaml_missing_extra_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """bundle 構築（from_yaml）は AGT 未導入相当で install hint 付き ImportError を送出する。"""

    def _raise_import_error(path: Any) -> Any:
        raise ImportError(_GOVERNANCE_INSTALL_HINT)

    monkeypatch.setattr("oai_agentspec._adapters.load_policy_bundle", _raise_import_error)
    with pytest.raises(ImportError, match=r"oai-agentspec\[governance\]"):
        GovernedAgentBuilder.from_yaml(tmp_path / "governance.yaml")


def test_override_not_marked_applied_when_build_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """override 適用 build が失敗した場合はキーを適用済みにしない（失敗 override の診断可能性）。"""

    def _raising_govern_spec(spec: Any, *, policy: Any, audit_sink: Any = None) -> Any:
        raise ValueError("policy load failed")

    monkeypatch.setattr("oai_agentspec._adapters.govern_spec", _raising_govern_spec)
    monkeypatch.setattr("oai_agentspec._adapters.new_audit_sink", lambda: object())
    builder = GovernedAgentBuilder(
        policy=object(), overrides={"bot": object()}, inner=_RecordingBuilder()
    )

    with pytest.raises(ValueError, match="policy load failed"):
        builder.build(AgentSpec(name="bot", instructions="x"))

    # 失敗した override は未適用のまま残る（成功後にのみ適用済み記録）。
    assert builder.unapplied_overrides == frozenset({"bot"})


# ----------------------------------------------------------------------
# post_process（registry 第 3 段）: 記録 fake
# ----------------------------------------------------------------------


class _PostProcessRecorder:
    """`govern_ungoverned_tools` / `govern_agent` / `new_audit_sink` / `resolve_policy` の履歴。"""

    def __init__(self) -> None:
        """履歴と factory 経路（`govern_agent`）の戻り値 sentinel を初期化する。"""
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.sinks: list[Any] = []
        self.resolved: list[Any] = []
        self.clone_result = object()


def _patch_post_process_recorder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    govern_agent_error: Exception | None = None,
    resolved_value: Any = None,
) -> _PostProcessRecorder:
    """post_process が使う `_adapters` の 4 関数を記録 fake に差し替える。

    fake の署名は新 API（`agent` + kw-only の `policy` / `audit_sink` / `agent_name`）に固定する
    （呼び出し側の署名が変わると TypeError で検知できる）。`govern_ungoverned_tools`
    は None、`govern_agent` は `clone_result` を返す（`govern_agent_error` 指定時は送出する）。
    `new_audit_sink` は呼ばれるたびに新しい sentinel を返す。`resolve_policy` は `resolved_value`
    が None なら受け取った値を、そうでなければ `resolved_value` を返す。

    Args:
        monkeypatch: pytest の monkeypatch。
        govern_agent_error: `govern_agent` に送出させる例外（None なら送出しない）。
        resolved_value: `resolve_policy` の戻り値（None なら素通し）。

    Returns:
        呼び出し履歴を持つ `_PostProcessRecorder`。
    """
    rec = _PostProcessRecorder()

    def _fake_govern_ungoverned_tools(
        agent: Any, *, policy: Any, audit_sink: Any, agent_name: str
    ) -> None:
        rec.calls.append(
            (
                "govern_ungoverned_tools",
                {
                    "agent": agent,
                    "policy": policy,
                    "audit_sink": audit_sink,
                    "agent_name": agent_name,
                },
            )
        )

    def _fake_govern_agent(agent: Any, *, policy: Any, audit_sink: Any, agent_name: str) -> Any:
        rec.calls.append(
            (
                "govern_agent",
                {
                    "agent": agent,
                    "policy": policy,
                    "audit_sink": audit_sink,
                    "agent_name": agent_name,
                },
            )
        )
        if govern_agent_error is not None:
            raise govern_agent_error
        return rec.clone_result

    def _fake_new_audit_sink() -> Any:
        sink = object()
        rec.sinks.append(sink)
        return sink

    def _fake_resolve_policy(policy: Any) -> Any:
        rec.resolved.append(policy)
        return policy if resolved_value is None else resolved_value

    monkeypatch.setattr(
        "oai_agentspec._adapters.govern_ungoverned_tools", _fake_govern_ungoverned_tools
    )
    monkeypatch.setattr("oai_agentspec._adapters.govern_agent", _fake_govern_agent)
    monkeypatch.setattr("oai_agentspec._adapters.new_audit_sink", _fake_new_audit_sink)
    monkeypatch.setattr("oai_agentspec._adapters.resolve_policy", _fake_resolve_policy)
    return rec


# ----------------------------------------------------------------------
# post-processor: spec 経路（sub_agent_tools のときだけ govern_ungoverned_tools）
# ----------------------------------------------------------------------


@pytest.mark.parametrize("sub_agent_tools", [False, True])
def test_post_process_spec_path_governs_ungoverned_tools_only_when_opted_in(
    monkeypatch: pytest.MonkeyPatch, sub_agent_tools: bool
) -> None:
    """spec 経路: sub_agent_tools=False は何も呼ばず同一 Agent を返し、True だけ親名ポリシーで統治。

    True 時は `govern_ungoverned_tools` を親名の override ポリシー（build で解決済みの
    スナップショット）・build と共有の sink・登録名で 1 回呼ぶ。`resolve_policy` は build の
    1 回だけで、post_process で再解決しない。`govern_agent` は spec 経路では呼ばない
    （False 側は factory_agents=True にして、両フラグ False の ValueError を避ける）。
    """
    resolved = object()
    rec = _patch_post_process_recorder(monkeypatch, resolved_value=resolved)
    monkeypatch.setattr(
        "oai_agentspec._adapters.govern_spec",
        lambda spec, *, policy, audit_sink=None: spec,
    )
    builder = GovernedAgentBuilder(
        policy=object(),
        overrides={"support": "support.yaml"},
        inner=_RecordingBuilder(),
    )
    post = builder.post_processor(
        sub_agent_tools=sub_agent_tools, factory_agents=not sub_agent_tools
    )
    spec = AgentSpec(name="support", instructions="x", tools=[_make_tool()])
    builder.build(spec)
    agent = object()

    result = post.post_process(agent, name="support", spec=spec)

    assert result is agent
    assert rec.resolved == ["support.yaml"]
    assert len(rec.sinks) == 1
    assert builder.audit_sink is rec.sinks[0]
    if sub_agent_tools:
        assert rec.calls == [
            (
                "govern_ungoverned_tools",
                {
                    "agent": agent,
                    "policy": resolved,
                    "audit_sink": rec.sinks[0],
                    "agent_name": "support",
                },
            )
        ]
    else:
        assert rec.calls == []


# ----------------------------------------------------------------------
# post-processor: factory 経路（factory_agents のときだけ govern_agent）
# ----------------------------------------------------------------------


def test_post_process_factory_path_passthrough_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """factory_agents=False の factory 経路（spec=None）は統治せず同一オブジェクトを返す。

    解決・sink 生成も起きない（builder.audit_sink は None のまま）。
    """
    rec = _patch_post_process_recorder(monkeypatch)
    builder = GovernedAgentBuilder(policy=object(), inner=_RecordingBuilder())
    post = builder.post_processor(sub_agent_tools=True)
    agent = object()

    result = post.post_process(agent, name="legacy", spec=None)

    assert result is agent
    assert rec.calls == []
    assert rec.sinks == []
    assert rec.resolved == []
    assert builder.audit_sink is None


@pytest.mark.parametrize(
    ("sub_agent_tools", "factory_agents"),
    [(True, False), (False, True), (True, True)],
    ids=["sub_only", "factory_only", "both"],
)
def test_post_process_flags_select_sub_agent_and_factory_governance(
    monkeypatch: pytest.MonkeyPatch,
    sub_agent_tools: bool,
    factory_agents: bool,
) -> None:
    """2 フラグは独立に効く（sub は spec 経路の統治、factory は factory 経路の統治を選ぶ）。

    オブジェクト形のポリシーは `resolve_policy` を通さずそのまま渡る。sink は経路を跨いで 1 本。
    """
    rec = _patch_post_process_recorder(monkeypatch)
    policy = object()
    sink = object()
    builder = GovernedAgentBuilder(policy=policy, audit_sink=sink, inner=_RecordingBuilder())
    post = builder.post_processor(sub_agent_tools=sub_agent_tools, factory_agents=factory_agents)
    spec = AgentSpec(name="support", instructions="x", tools=[_make_tool()])
    spec_agent = object()
    factory_agent = object()

    spec_result = post.post_process(spec_agent, name="support", spec=spec)
    factory_result = post.post_process(factory_agent, name="legacy", spec=None)

    expected: list[tuple[str, dict[str, Any]]] = []
    if sub_agent_tools:
        expected.append(
            (
                "govern_ungoverned_tools",
                {
                    "agent": spec_agent,
                    "policy": policy,
                    "audit_sink": sink,
                    "agent_name": "support",
                },
            )
        )
    if factory_agents:
        expected.append(
            (
                "govern_agent",
                {
                    "agent": factory_agent,
                    "policy": policy,
                    "audit_sink": sink,
                    "agent_name": "legacy",
                },
            )
        )
    assert rec.calls == expected
    assert spec_result is spec_agent
    assert factory_result is (rec.clone_result if factory_agents else factory_agent)
    assert rec.resolved == []
    assert rec.sinks == []


def test_default_sink_created_on_first_factory_post_process_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """factory だけの registry でオプトインすると、既定 sink は初回の factory 統治で生成される。

    build を経ずに factory 経路の post_process が先に来ても sink は 1 本だけ生成され、
    `builder.audit_sink` と同一である。同じ builder から作った別の post-processor と、
    後続の `builder.build`（`govern_spec`）も同じ sink を使う（監査チェーンが分断されない）。
    """
    rec = _patch_post_process_recorder(monkeypatch)
    govern_spec_sinks: list[Any] = []

    def _fake_govern_spec(spec: Any, *, policy: Any, audit_sink: Any = None) -> Any:
        govern_spec_sinks.append(audit_sink)
        return spec

    monkeypatch.setattr("oai_agentspec._adapters.govern_spec", _fake_govern_spec)
    policy = object()
    builder = GovernedAgentBuilder(policy=policy, inner=_RecordingBuilder())
    post = builder.post_processor(factory_agents=True)
    other_post = builder.post_processor(factory_agents=True)
    first = object()
    second = object()
    third = object()

    assert builder.audit_sink is None
    post.post_process(first, name="f1", spec=None)
    post.post_process(second, name="f2", spec=None)
    other_post.post_process(third, name="f3", spec=None)
    builder.build(AgentSpec(name="support", instructions="x"))

    assert len(rec.sinks) == 1
    assert builder.audit_sink is rec.sinks[0]
    assert govern_spec_sinks == [rec.sinks[0]]
    assert govern_spec_sinks[0] is rec.sinks[0]
    assert rec.calls == [
        (
            "govern_agent",
            {"agent": first, "policy": policy, "audit_sink": rec.sinks[0], "agent_name": "f1"},
        ),
        (
            "govern_agent",
            {"agent": second, "policy": policy, "audit_sink": rec.sinks[0], "agent_name": "f2"},
        ),
        (
            "govern_agent",
            {"agent": third, "policy": policy, "audit_sink": rec.sinks[0], "agent_name": "f3"},
        ),
    ]


# ----------------------------------------------------------------------
# unapplied_overrides: factory 経路は統治した post_process 成功時にのみ減る
# ----------------------------------------------------------------------


def test_unapplied_overrides_kept_when_post_process_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """factory 経路で `govern_agent` が例外を出した post_process ではキーを適用済みにしない。"""
    _patch_post_process_recorder(monkeypatch, govern_agent_error=RuntimeError("govern failed"))
    builder = GovernedAgentBuilder(
        policy=object(), overrides={"legacy": object()}, inner=_RecordingBuilder()
    )
    post = builder.post_processor(factory_agents=True)

    with pytest.raises(RuntimeError, match="govern failed"):
        post.post_process(object(), name="legacy", spec=None)

    assert builder.unapplied_overrides == frozenset({"legacy"})


@pytest.mark.parametrize("factory_agents", [False, True])
def test_unapplied_overrides_on_factory_path_follow_governance(
    monkeypatch: pytest.MonkeyPatch, factory_agents: bool
) -> None:
    """factory 経路の登録名は統治した post_process 成功でのみ減る（非統治の素通しでは減らない）。

    適用記録は post-processor ではなく builder の `unapplied_overrides` に反映される
    （False 側は sub_agent_tools=True にして、両フラグ False の ValueError を避ける）。
    """
    _patch_post_process_recorder(monkeypatch)
    builder = GovernedAgentBuilder(
        policy=object(), overrides={"legacy": object()}, inner=_RecordingBuilder()
    )
    post = builder.post_processor(sub_agent_tools=not factory_agents, factory_agents=factory_agents)

    post.post_process(object(), name="legacy", spec=None)

    expected = frozenset() if factory_agents else frozenset({"legacy"})
    assert builder.unapplied_overrides == expected
