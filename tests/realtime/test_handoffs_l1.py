"""L1: RealtimeHandoffGraph の宣言・apply・等価性・mermaid 検証（agents 非依存・RED 先行）。

Issue #15 タスク2（`oai_agentspec.realtime.handoffs` 新設）に対する L1 検証。
`RealtimeHandoffEdge` / `RealtimeHandoffGraph` / `from_specs` は未実装のため、本モジュールの
import は collection error（RED）になる想定。

設計 SoT（/tmp/architecture/15.md・15_policy.md）に従い、以下を一次情報として固定する:
- apply は spec.handoffs / spec.handoff_options を in-place 反映し、直後に共有バリデータで検証する
- グラフ apply == spec 直接宣言（同一 registry 結線）の等価性
- 順序非依存（register 済み spec への apply でも不正設定が ValueError）
- 一回性（再 apply で消えた src の handoffs を自動クリアしない・エッジを持たない src は不変）
- mermaid は flowchart TD の静的エッジのみ（tool_description_override をラベル源・破線なし）
- from_specs は各 spec.handoffs から静的エッジを張る（コア from_specs と対称）
- edge() 引数 -> RealtimeHandoffConfig フィールドのマッピング
"""

from __future__ import annotations

import pytest

from oai_agentspec.realtime.handoffs import (
    RealtimeHandoffEdge,
    RealtimeHandoffGraph,
    from_specs,
)
from oai_agentspec.realtime.registry import RealtimeAgentRegistry
from oai_agentspec.realtime.spec import RealtimeAgentSpec, RealtimeHandoffConfig

from _helpers.fake_realtime_builder import FakeRealtimeAgentBuilder


def make_registry() -> RealtimeAgentRegistry:
    """FakeRealtimeAgentBuilder 注入済みの registry を生成する（agents 非依存）。"""
    return RealtimeAgentRegistry(agent_builder=FakeRealtimeAgentBuilder())


def wiring(agent: object) -> list[tuple[str, object]]:
    """構築済み FakeRealtimeAgent の結線内容を (target 名, config) の列で取り出す。"""
    return [(h.target.name, h.config) for h in agent.handoffs]  # type: ignore[attr-defined]


# ------------------------------------------------------------------
# edge() 引数 -> RealtimeHandoffConfig マッピング
# ------------------------------------------------------------------
def test_edge_引数がRealtimeHandoffConfigへマップされる() -> None:
    """edge() の各引数が設計の表どおり RealtimeHandoffConfig の対応フィールドへ入る。

    tool_name -> tool_name_override / tool_description -> tool_description_override /
    on_handoff・input_type・is_enabled は同名。
    """
    fn = lambda c, i: None  # noqa: E731 - テスト用の識別子確認
    graph = RealtimeHandoffGraph(entry="triage")
    graph.edge(
        "triage",
        "billing",
        tool_description="請求",
        tool_name="go_billing",
        on_handoff=fn,
        input_type=object,
        is_enabled=False,
    )
    edge = graph.edges[0]
    assert isinstance(edge, RealtimeHandoffEdge)
    assert (edge.src, edge.dst) == ("triage", "billing")
    cfg = edge.config
    assert cfg.tool_description_override == "請求"
    assert cfg.tool_name_override == "go_billing"
    assert cfg.on_handoff is fn
    assert cfg.input_type is object
    assert cfg.is_enabled is False


def test_edge_はグラフ自身を返し連鎖できる() -> None:
    """edge() は自身（グラフ）を返し、fluent に連鎖できる。"""
    graph = RealtimeHandoffGraph()
    assert graph.edge("a", "b") is graph


def test_extend_と_outgoing() -> None:
    """extend((src, dst) 列) がまとめて追加され、outgoing(src) が dst 列を返す。"""
    graph = RealtimeHandoffGraph()
    graph.extend([("a", "b"), ("a", "c")])
    assert graph.outgoing("a") == ["b", "c"]


# ------------------------------------------------------------------
# apply の spec 反映
# ------------------------------------------------------------------
def test_apply_はspecのhandoffsとoptionsを反映する() -> None:
    """apply(specs) が src spec の handoffs を replace し handoff_options を書き込む。"""
    triage = RealtimeAgentSpec(name="triage", instructions="t")
    billing = RealtimeAgentSpec(name="billing", instructions="b")
    graph = RealtimeHandoffGraph(entry="triage")
    graph.edge("triage", "billing", tool_description="請求")
    graph.apply([triage, billing])
    assert triage.handoffs == ["billing"]
    assert triage.handoff_options["billing"].tool_description_override == "請求"


def test_apply_はエッジを持たないsrcを触らない() -> None:
    """グラフにエッジを持たない src の spec は apply で変異しない（一回性・非破壊）。"""
    triage = RealtimeAgentSpec(name="triage", instructions="t")
    orphan = RealtimeAgentSpec(name="orphan", instructions="o", handoffs=["triage"])
    graph = RealtimeHandoffGraph(entry="triage")
    graph.edge("triage", "billing")
    billing = RealtimeAgentSpec(name="billing", instructions="b")
    graph.apply([triage, orphan, billing])
    # orphan はグラフのエッジ src ではないため既存 handoffs が保持される
    assert orphan.handoffs == ["triage"]
    assert orphan.handoff_options == {}


def test_apply_で未登録srcはKeyError() -> None:
    """グラフに現れる src が specs に無い場合、apply は KeyError（コア apply と対称）。"""
    billing = RealtimeAgentSpec(name="billing", instructions="b")
    graph = RealtimeHandoffGraph()
    graph.edge("ghost", "billing")
    with pytest.raises(KeyError, match="ghost"):
        graph.apply([billing])


# ------------------------------------------------------------------
# apply の検証迂回防止（順序非依存）
# ------------------------------------------------------------------
def test_apply_は不正なinput_type設定をValueError() -> None:
    """input_type ありで on_handoff なしのエッジは apply の mutation 直後検証で ValueError。"""
    triage = RealtimeAgentSpec(name="triage", instructions="t")
    billing = RealtimeAgentSpec(name="billing", instructions="b")
    graph = RealtimeHandoffGraph(entry="triage")
    graph.edge("triage", "billing", input_type=object)
    with pytest.raises(ValueError, match=r"'triage' -> 'billing'.*on_handoff"):
        graph.apply([triage, billing])


def test_apply_は不正なon_handoff_arityをValueError() -> None:
    """on_handoff の引数個数が SDK 契約と合わないエッジは apply で ValueError。"""
    triage = RealtimeAgentSpec(name="triage", instructions="t")
    billing = RealtimeAgentSpec(name="billing", instructions="b")
    graph = RealtimeHandoffGraph(entry="triage")
    graph.edge("triage", "billing", on_handoff=lambda c, i: None)
    with pytest.raises(ValueError, match=r"'triage' -> 'billing'.*1 引数.*2 引数"):
        graph.apply([triage, billing])


def test_順序非依存_register後のapplyでも不正設定は検証される() -> None:
    """register 済み spec を後から apply しても、apply が検証を再実行し ValueError にする。

    コアの通常フロー（register -> apply）を Realtime でも書いた場合の検証迂回を防ぐ設計の核心。
    """
    reg = make_registry()
    triage = RealtimeAgentSpec(name="triage", instructions="t")
    billing = RealtimeAgentSpec(name="billing", instructions="b")
    reg.register(triage)
    reg.register(billing)
    graph = RealtimeHandoffGraph(entry="triage")
    graph.edge("triage", "billing", input_type=object)  # on_handoff 欠落 = 不正
    with pytest.raises(ValueError, match=r"'triage' -> 'billing'.*on_handoff"):
        graph.apply([triage, billing])


# ------------------------------------------------------------------
# 等価性: グラフ apply == spec 直接宣言（registry 結線が一致）
# ------------------------------------------------------------------
def test_等価性_グラフapplyとspec直接宣言の結線が一致する() -> None:
    """同一トポロジをグラフ apply と spec 直接宣言で構築し、registry.get 結線が一致する。"""
    # (1) グラフ DSL + apply
    reg1 = make_registry()
    triage1 = RealtimeAgentSpec(name="triage", instructions="t")
    billing1 = RealtimeAgentSpec(name="billing", instructions="b")
    graph = RealtimeHandoffGraph(entry="triage")
    graph.edge("triage", "billing", tool_description="請求")
    graph.apply([triage1, billing1])
    reg1.register(triage1)
    reg1.register(billing1)

    # (2) spec に直接 handoffs / handoff_options を宣言
    reg2 = make_registry()
    triage2 = RealtimeAgentSpec(
        name="triage",
        instructions="t",
        handoffs=["billing"],
        handoff_options={"billing": RealtimeHandoffConfig(tool_description_override="請求")},
    )
    billing2 = RealtimeAgentSpec(name="billing", instructions="b")
    reg2.register(triage2)
    reg2.register(billing2)

    assert wiring(reg1.get("triage")) == wiring(reg2.get("triage"))


# ------------------------------------------------------------------
# 一回性: 再 apply で消えた src の handoffs を自動クリアしない
# ------------------------------------------------------------------
def test_一回性_再applyで消えたsrcのhandoffsを自動クリアしない() -> None:
    """前回 apply で反映した src が、エッジを持たない別グラフの再 apply で消去されない。"""
    triage = RealtimeAgentSpec(name="triage", instructions="t")
    billing = RealtimeAgentSpec(name="billing", instructions="b")
    graph1 = RealtimeHandoffGraph(entry="triage")
    graph1.edge("triage", "billing")
    graph1.apply([triage, billing])
    assert triage.handoffs == ["billing"]

    # triage を src に持たない別グラフを再 apply しても handoffs は残る
    graph2 = RealtimeHandoffGraph(entry="triage")
    graph2.apply([triage, billing])
    assert triage.handoffs == ["billing"]


# ------------------------------------------------------------------
# 循環トポロジの結線
# ------------------------------------------------------------------
def test_循環_triageとsupportの相互ハンドオフがregistryで結線される() -> None:
    """triage <-> support の循環エッジを apply -> register し、遅延バインドで相互解決される。"""
    reg = make_registry()
    triage = RealtimeAgentSpec(name="triage", instructions="t")
    support = RealtimeAgentSpec(name="support", instructions="s")
    graph = RealtimeHandoffGraph(entry="triage")
    graph.edge("triage", "support")
    graph.edge("support", "triage")
    graph.apply([triage, support])
    reg.register(triage)
    reg.register(support)
    a = reg.get("triage")
    b = reg.get("support")
    assert a.handoffs[0].target is b
    assert b.handoffs[0].target is a


# ------------------------------------------------------------------
# entry_agent
# ------------------------------------------------------------------
def test_entry_agent_はentryを取得する() -> None:
    """entry_agent(registry) は entry 名のエージェントを registry.get で返す。"""
    reg = make_registry()
    triage = RealtimeAgentSpec(name="triage", instructions="t")
    graph = RealtimeHandoffGraph(entry="triage")
    graph.apply([triage])
    reg.register(triage)
    assert graph.entry_agent(reg) is reg.get("triage")


def test_entry_agent_はentry未設定でValueError() -> None:
    """entry 未設定のグラフで entry_agent を呼ぶと ValueError（コア HandoffGraph と対称）。"""
    reg = make_registry()
    graph = RealtimeHandoffGraph()
    with pytest.raises(ValueError, match="entry"):
        graph.entry_agent(reg)


# ------------------------------------------------------------------
# mermaid
# ------------------------------------------------------------------
def test_mermaid_はflowchartとentryとラベルを出力する() -> None:
    """mermaid は flowchart TD / start --> entry / ラベル付き・なしエッジを出力し破線を含まない。

    ラベル源は tool_description_override（未設定時は無ラベル）。動的破線は持たない。
    """
    graph = RealtimeHandoffGraph(entry="triage")
    graph.edge("triage", "billing", tool_description="請求")
    graph.edge("triage", "support")
    out = graph.mermaid()
    assert "flowchart TD" in out
    assert "start([start]) --> triage" in out
    assert "triage -->|請求| billing" in out
    assert "triage --> support" in out
    # 動的破線（-.->）は Realtime に存在しないため出力されない
    assert "-.->" not in out


# ------------------------------------------------------------------
# from_specs
# ------------------------------------------------------------------
def test_from_specs_はspecのhandoffsから静的エッジを張る() -> None:
    """from_specs は各 spec.handoffs から静的エッジを構築し entry を設定する（コアと対称）。"""
    specs = [
        RealtimeAgentSpec(name="triage", instructions="t", handoffs=["billing", "support"]),
        RealtimeAgentSpec(name="billing", instructions="b"),
        RealtimeAgentSpec(name="support", instructions="s"),
    ]
    graph = from_specs(specs, entry="triage")
    assert graph.entry == "triage"
    assert graph.outgoing("triage") == ["billing", "support"]


# ------------------------------------------------------------------
# apply の同名 spec 重複検出 / 公開窓口
# ------------------------------------------------------------------
def test_apply_は同名specの重複をValueError() -> None:
    """specs に同名 spec が複数あると apply が ValueError（silent 後勝ちを防ぐ）。"""
    graph = RealtimeHandoffGraph(entry="dup")
    graph.edge("dup", "dup2")
    specs = [
        RealtimeAgentSpec(name="dup", instructions="a"),
        RealtimeAgentSpec(name="dup", instructions="b"),
        RealtimeAgentSpec(name="dup2", instructions="c"),
    ]
    with pytest.raises(ValueError, match="dup"):
        graph.apply(specs)


def test_公開窓口はグラフDSLの3シンボルを再エクスポートする() -> None:
    """oai_agentspec.realtime の __all__ にグラフ DSL の 3 シンボルが載る（公開契約の固定）。"""
    import oai_agentspec.realtime as window

    assert {"RealtimeHandoffGraph", "RealtimeHandoffEdge", "from_specs"} <= set(window.__all__)
    assert window.RealtimeHandoffGraph is RealtimeHandoffGraph


# ------------------------------------------------------------------
# apply の原子性 / from_specs の重複検出
# ------------------------------------------------------------------
def test_apply_は途中失敗で一切のspecを変異させない() -> None:
    """後続 src の検証失敗（ValueError）時、先行 src の spec も変異しない（原子性）。"""
    graph = RealtimeHandoffGraph(entry="triage")
    graph.edge("triage", "billing")
    graph.edge("support", "billing", input_type=object)  # on_handoff 欠落で不正
    triage = RealtimeAgentSpec(name="triage", instructions="t")
    billing = RealtimeAgentSpec(name="billing", instructions="b")
    support = RealtimeAgentSpec(name="support", instructions="s")
    with pytest.raises(ValueError, match="on_handoff"):
        graph.apply([triage, billing, support])
    assert triage.handoffs == []
    assert triage.handoff_options == {}


def test_apply_は未登録srcでも一切のspecを変異させない() -> None:
    """後続 src の KeyError 時も先行 src の spec は変異しない（原子性）。"""
    graph = RealtimeHandoffGraph(entry="triage")
    graph.edge("triage", "billing")
    graph.edge("ghost", "billing")
    triage = RealtimeAgentSpec(name="triage", instructions="t")
    billing = RealtimeAgentSpec(name="billing", instructions="b")
    with pytest.raises(KeyError, match="ghost"):
        graph.apply([triage, billing])
    assert triage.handoffs == []


def test_from_specs_は同名specの重複をValueError() -> None:
    """from_specs も apply と同様に同名 spec の重複を入口で ValueError にする。"""
    specs = [
        RealtimeAgentSpec(name="dup", instructions="a"),
        RealtimeAgentSpec(name="dup", instructions="b"),
    ]
    with pytest.raises(ValueError, match="dup"):
        from_specs(specs)
