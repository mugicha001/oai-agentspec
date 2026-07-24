"""L1/L2: ``SandboxAgentSpec``（AgentSpec のサンドボックス拡張 dataclass）の検証。

前半（Issue #21 T1）は継承関係・追加 4 フィールドの既定値 None・kw_only 属性・AgentSpec
フィールドの継承・``dataclasses.replace`` でのサブクラス保持・**検証対象である spec.py 自体**の
SDK 隔離（``agents`` 非依存）を確認する（本テストファイル自体は SDK 実体の構築検証のため
``agents`` に依存する）。

後半（Issue #21 T4）は実ビルダー（``AgentRegistry`` デフォルトの ``_adapters.build_agent``）
経由で ``AgentSpec`` と ``SandboxAgentSpec`` を同一レジストリに混在登録した際の受け入れ基準
（相互ハンドオフ・validate・freeze・clone・sub_agents 混在）を検証する。
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
from agents import Agent
from agents.sandbox import SandboxAgent

from oai_agentspec import AgentRegistry, HandoffGraph
from oai_agentspec.spec import AgentSpec, SandboxAgentSpec

SANDBOX_FIELDS = ("default_manifest", "capabilities", "run_as", "base_instructions")


def test_sandbox_spec_is_agent_spec_subclass() -> None:
    """SandboxAgentSpec は AgentSpec のサブクラスであり dataclass である。"""
    assert issubclass(SandboxAgentSpec, AgentSpec)
    assert dataclasses.is_dataclass(SandboxAgentSpec)


def test_sandbox_fields_default_to_none() -> None:
    """追加 4 フィールドの既定値はすべて None（SDK 既定値を再現しない）。"""
    spec = SandboxAgentSpec(name="x")
    assert spec.default_manifest is None
    assert spec.capabilities is None
    assert spec.run_as is None
    assert spec.base_instructions is None


def test_sandbox_fields_are_kw_only() -> None:
    """追加 4 フィールドはすべて kw_only（位置引数の束縛ズレを起こさない）。"""
    fields_by_name = {f.name: f for f in dataclasses.fields(SandboxAgentSpec)}
    for name in SANDBOX_FIELDS:
        assert name in fields_by_name, f"フィールド {name} が定義されていない"
        assert fields_by_name[name].kw_only is True, f"{name} は kw_only であるべき"


def test_inherits_all_agent_spec_fields() -> None:
    """AgentSpec の全フィールドがそのまま継承されている。"""
    base_names = {f.name for f in dataclasses.fields(AgentSpec)}
    sub_names = {f.name for f in dataclasses.fields(SandboxAgentSpec)}
    assert base_names <= sub_names
    assert sub_names - base_names == set(SANDBOX_FIELDS)


def test_constructor_preserves_inherited_and_sandbox_values() -> None:
    """コンストラクタで指定した継承フィールド・追加フィールドの値がそのまま保持される。"""
    manifest = object()
    caps = object()
    spec = SandboxAgentSpec(
        name="sandboxed",
        instructions="do work",
        handoffs=["billing"],
        sub_agents=["helper"],
        extra={"k": "v"},
        default_manifest=manifest,
        capabilities=caps,
        run_as="worker",
        base_instructions="base",
    )
    assert spec.name == "sandboxed"
    assert spec.instructions == "do work"
    assert spec.handoffs == ["billing"]
    assert spec.sub_agents == ["helper"]
    assert spec.extra == {"k": "v"}
    assert spec.default_manifest is manifest
    assert spec.capabilities is caps
    assert spec.run_as == "worker"
    assert spec.base_instructions == "base"


def test_base_instructions_accepts_callable() -> None:
    """base_instructions は callable（動的 instructions）も受け付ける。"""

    def dyn(ctx: object, agent: object) -> str:
        return "dynamic"

    spec = SandboxAgentSpec(name="x", base_instructions=dyn)
    assert spec.base_instructions is dyn


def test_dataclasses_replace_preserves_subclass_and_sandbox_fields() -> None:
    """replace 後もサブクラス型・追加フィールド値が保持される（freeze/clone の基盤）。"""
    manifest = object()
    spec = SandboxAgentSpec(
        name="orig",
        instructions="i",
        default_manifest=manifest,
        run_as="worker",
    )
    replaced = dataclasses.replace(spec, name="renamed")
    assert type(replaced) is SandboxAgentSpec
    assert replaced.name == "renamed"
    assert replaced.instructions == "i"
    assert replaced.default_manifest is manifest
    assert replaced.run_as == "worker"
    assert replaced.capabilities is None


def test_spec_module_does_not_import_agents_sdk() -> None:
    """spec.py のソースに ``agents`` / ``openai`` の import が無い（NFR-1 SDK 隔離）。"""
    spec_path = Path(__file__).resolve().parent.parent / "src" / "oai_agentspec" / "spec.py"
    src = spec_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^\s*(from\s+(agents|openai)\b|import\s+(agents|openai)\b)", re.MULTILINE)
    assert pattern.search(src) is None


# ----------------------------------------------------------------------
# Issue #21 T4: AgentSpec / SandboxAgentSpec 混在レジストリ（実ビルダー・FR-3）
# ----------------------------------------------------------------------


@pytest.mark.integration
def test_mixed_registry_cyclic_handoff_resolves_with_real_types() -> None:
    """通常 AgentSpec と SandboxAgentSpec の相互ハンドオフ（A→B→A）が例外なく解決される。

    b の実体は ``agents.sandbox.SandboxAgent``、a の実体は素の ``agents.Agent`` であり、
    双方の handoffs が識別子（identity）で結線されていることを検証する。
    """
    reg = AgentRegistry()
    reg.register(AgentSpec(name="a", instructions="agent a", handoffs=["b"]))
    reg.register(SandboxAgentSpec(name="b", instructions="agent b", handoffs=["a"]))

    a = reg.get("a")
    b = reg.get("b")

    assert type(a) is Agent
    assert isinstance(b, SandboxAgent)
    assert a.handoffs[0] is b
    assert b.handoffs[0] is a


@pytest.mark.integration
def test_validate_detects_missing_handoff_from_sandbox_spec() -> None:
    """SandboxAgentSpec 由来の未登録 handoff 参照は通常 spec と同じ KeyError で列挙される。"""
    reg = AgentRegistry()
    reg.register(SandboxAgentSpec(name="s", instructions="s", handoffs=["missing"]))
    with pytest.raises(KeyError, match="missing"):
        reg.validate()


@pytest.mark.integration
def test_freeze_preserves_sandbox_fields_after_build() -> None:
    """freeze 後に get() した SandboxAgent でも sandbox 固有フィールドが欠落しない。"""
    reg = AgentRegistry()
    reg.register(SandboxAgentSpec(name="s", instructions="s", run_as="worker"))
    reg.freeze()

    built = reg.get("s")

    assert isinstance(built, SandboxAgent)
    assert built.run_as == "worker"


@pytest.mark.integration
def test_clone_preserves_sandbox_spec_type_and_fields() -> None:
    """clone した registry でも SandboxAgentSpec 型のまま複製され、固有フィールドが保持される。"""
    reg = AgentRegistry()
    reg.register(SandboxAgentSpec(name="s", instructions="s", run_as="worker"))

    cloned = reg.clone()

    assert type(cloned._specs["s"]) is SandboxAgentSpec  # noqa: SLF001 - 型保持の検証
    built = cloned.get("s")
    assert isinstance(built, SandboxAgent)
    assert built.run_as == "worker"


@pytest.mark.integration
def test_sub_agents_mixed_wires_as_tool() -> None:
    """通常 AgentSpec が SandboxAgentSpec をサブエージェント参照しても as_tool 配線が機能する。"""
    reg = AgentRegistry()
    reg.register(SandboxAgentSpec(name="sandbox_sub", instructions="sub"))
    reg.register(AgentSpec(name="orch", instructions="orchestrate", sub_agents=["sandbox_sub"]))

    orch = reg.get("orch")

    assert len(orch.tools) == 1
    assert orch.tools[0].name == "sandbox_sub"


@pytest.mark.integration
def test_handoff_graph_apply_wires_sandbox_spec_too() -> None:
    """HandoffGraph 経由の適用でも SandboxAgentSpec の handoffs が正しく結線される。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="a", instructions="a"))
    reg.register(SandboxAgentSpec(name="b", instructions="b"))
    graph = HandoffGraph(entry="a")
    graph.edge("a", "b")
    graph.edge("b", "a")
    graph.apply(reg)

    a = reg.get("a")
    b = reg.get("b")

    assert isinstance(b, SandboxAgent)
    assert a.handoffs[0] is b
    assert b.handoffs[0] is a


@pytest.mark.integration
def test_freeze_blocks_external_capabilities_mutation() -> None:
    """freeze 後の外部 capabilities リスト mutation が build 結果に伝播しない。

    freeze は spec を独立コピーに置き換えて外部参照経由の mutation を遮断する契約を
    持つ。sandbox 固有の可変コンテナ（capabilities リスト）もこの契約の対象であること
    をピン留めする（tools 等の基底フィールドと同じ遮断挙動）。
    """
    caps: list[object] = [object()]
    reg = AgentRegistry()
    reg.register(SandboxAgentSpec(name="s", instructions="i", capabilities=caps))
    reg.freeze()

    caps.append(object())
    built = reg.get("s")

    assert isinstance(built, SandboxAgent)
    assert len(list(built.capabilities)) == 1
