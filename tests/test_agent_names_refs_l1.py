"""L1: 定数簿の値を FR-2 の全参照経路へ渡しても生 str と同一結果になることの検証。

タスク A3 の RED テスト。`src/oai_agentspec/agent_names.py` は未実装のため、本ファイルは
import 時点で失敗する（実装後に緑へ変わる）。

`AgentNames` の宣言値は `str` であり、`spec.py` / `handoffs.py` / `next_turn.py` /
`registry.py` は本件で無変更である（`docs/adr/0018-declarative-agent-name-catalog.md` の
Decision 7）。したがって「定数を渡した構成」と「生 str を渡した構成」は同一でなければならない。
本ファイルはその同一性を、`handoffs` / `handoff_options` のキー / `sub_agents` /
`sub_agent_tools` のキー / `DynamicHandoff.candidates` / `NextTurnRule` の到達元・遷移先 /
`NextTurnPolicy` のキー / entry 名 / 混在宣言 / 位置引数束縛の各経路で pin する。

`agents` 非依存を保つため Agent 構築はフェイク builder（`_helpers.fake_builder.FakeAgentBuilder`
の派生）を注入する。SDK 結線（`handoff()` / `as_tool()`）が触る最小属性だけをスタブへ足す。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from oai_agentspec import AgentRegistry, AgentSpec, HandoffConfig
from oai_agentspec.agent_names import AgentNames
from oai_agentspec.next_turn import (
    NextTurnPolicy,
    NextTurnRule,
    apply_next_turn_policy,
    resolve_next_agent,
)
from oai_agentspec.spec import DynamicHandoff

from _helpers.fake_builder import FakeAgent, FakeAgentBuilder

pytestmark = pytest.mark.unit


class _Names(AgentNames):
    """テスト共通の定数簿（宣言はこの 1 箇所）。"""

    PLANNER = "planner"
    REVIEWER = "reviewer"
    WRITER = "writer"


@dataclass(frozen=True)
class _AgentTool:
    """`as_tool()` の戻り値スタブ（値で比較できる plain データ）。"""

    agent_name: str
    tool_name: str | None
    tool_description: str | None


@dataclass
class _StubAgent(FakeAgent):
    """`FakeAgent` に SDK 結線が触る最小属性を足したスタブ。

    `handoff()` は `agent.name` / `agent.handoff_description` を読み、`as_tool()` は
    サブエージェントのツール化で呼ばれる。いずれも `agents` の Agent を構築せずに
    registry の結線ロジックだけを通すためのスタブ。
    """

    handoff_description: str | None = None

    def as_tool(self, *, tool_name: str | None, tool_description: str | None) -> _AgentTool:
        """サブエージェントのツール化結果を比較可能な plain データとして返す。

        Args:
            tool_name: `sub_agent_tools` の上書き名（未指定なら None）。
            tool_description: `sub_agent_tools` の上書き説明（未指定なら None）。

        Returns:
            比較用の `_AgentTool`。
        """
        return _AgentTool(
            agent_name=self.name, tool_name=tool_name, tool_description=tool_description
        )


class _StubAgentBuilder(FakeAgentBuilder):
    """`_StubAgent` を返す `AgentBuilder`（`FakeAgentBuilder` の派生）。"""

    def build(self, spec: AgentSpec) -> Any:
        """spec から結線前のスタブ Agent を 1 つ構築する。

        Args:
            spec: 構築対象の宣言。

        Returns:
            handoffs 未結線の `_StubAgent`。
        """
        self.built.append(spec.name)
        return _StubAgent(
            name=spec.name,
            instructions=spec.instructions,
            prompt=spec.prompt,
            tools=list(spec.tools),
        )


def _make_registry() -> AgentRegistry:
    """スタブ builder を注入した空の registry を返す。"""
    return AgentRegistry(agent_builder=_StubAgentBuilder())


def _handoff_key(item: Any) -> tuple[str, str]:
    """handoffs の 1 要素を比較用のキーへ落とす。

    per-edge 設定なしのエッジは Agent 実体が直 append され、設定ありのエッジは SDK の
    `Handoff` へ昇格する。両方を同じ形で比較できるようにする。

    Args:
        item: `agent.handoffs` の 1 要素。

    Returns:
        `("agent", 名前)` または `("handoff", tool 名)`。
    """
    if isinstance(item, _StubAgent):
        return ("agent", item.name)
    return ("handoff", item.tool_name)


def _structure(agent: Any) -> dict[str, Any]:
    """構築済み Agent の構成（名前 / instructions / handoffs の並び / tools の並び）を返す。

    Args:
        agent: `registry.get()` の戻り値。

    Returns:
        比較用の plain dict。
    """
    return {
        "name": agent.name,
        "instructions": agent.instructions,
        "handoffs": [_handoff_key(item) for item in agent.handoffs],
        "tools": list(agent.tools),
    }


# ---------------------------------------------------------------------------
# handoffs / sub_agents
# ---------------------------------------------------------------------------
def test_定数と生_str_で同一の_Agent_構成になる() -> None:
    """`handoffs` / `sub_agents` に定数を渡しても生 str と同一の構成になる。

    `handoffs` の並びと `tools`（sub_agents の as_tool 結果）の並びまで含めて比較する。
    """
    const_registry = _make_registry()
    const_registry.register(
        AgentSpec(
            name=_Names.PLANNER,
            instructions="計画を立てる",
            handoffs=[_Names.WRITER, _Names.REVIEWER],
            sub_agents=[_Names.REVIEWER],
        )
    )
    const_registry.register(AgentSpec(name=_Names.WRITER, instructions="本文を書く"))
    const_registry.register(AgentSpec(name=_Names.REVIEWER, instructions="レビューする"))

    raw_registry = _make_registry()
    raw_registry.register(
        AgentSpec(
            name="planner",
            instructions="計画を立てる",
            handoffs=["writer", "reviewer"],
            sub_agents=["reviewer"],
        )
    )
    raw_registry.register(AgentSpec(name="writer", instructions="本文を書く"))
    raw_registry.register(AgentSpec(name="reviewer", instructions="レビューする"))

    const_agent = const_registry.get(_Names.PLANNER)
    raw_agent = raw_registry.get("planner")

    assert _structure(const_agent) == _structure(raw_agent)
    assert _structure(const_agent)["handoffs"] == [("agent", "writer"), ("agent", "reviewer")]
    assert _structure(const_agent)["tools"] == [
        _AgentTool(agent_name="reviewer", tool_name=None, tool_description=None)
    ]
    const_registry.validate()
    raw_registry.validate()


# ---------------------------------------------------------------------------
# handoff_options / sub_agent_tools のキー
# ---------------------------------------------------------------------------
def test_定数を_handoff_options_と_sub_agent_tools_のキーに使っても同一構成になる() -> None:
    """dict キーの参照経路でも定数と生 str が同一の per-edge 設定を引く。"""

    def _build(planner: str, writer: str, reviewer: str) -> AgentRegistry:
        registry = _make_registry()
        registry.register(
            AgentSpec(
                name=planner,
                instructions="計画を立てる",
                handoffs=[writer],
                handoff_options={writer: HandoffConfig(description="執筆担当へ", tool_name="to_w")},
                sub_agents=[reviewer],
                sub_agent_tools={reviewer: ("review_tool", "レビューする")},
            )
        )
        registry.register(AgentSpec(name=writer, instructions="本文を書く"))
        registry.register(AgentSpec(name=reviewer, instructions="レビューする"))
        return registry

    const_agent = _build(_Names.PLANNER, _Names.WRITER, _Names.REVIEWER).get(_Names.PLANNER)
    raw_agent = _build("planner", "writer", "reviewer").get("planner")

    assert _structure(const_agent) == _structure(raw_agent)
    # per-edge 設定が引けている（キーが一致しなければ設定なしの直 append になる）。
    assert _structure(const_agent)["handoffs"] == [("handoff", "to_w")]
    assert _structure(const_agent)["tools"] == [
        _AgentTool(agent_name="reviewer", tool_name="review_tool", tool_description="レビューする")
    ]


# ---------------------------------------------------------------------------
# DynamicHandoff.candidates
# ---------------------------------------------------------------------------
async def test_定数を_dynamic_handoff_の候補に渡しても候補内チェックの結果が同じ() -> None:
    """候補内の名前は解決され、候補外の名前は `ValueError` になる（定数・生 str 同一）。"""

    def _build(planner: str, writer: str, chosen: str) -> AgentRegistry:
        registry = _make_registry()
        registry.register(
            AgentSpec(
                name=planner,
                instructions="計画を立てる",
                dynamic_handoffs=[
                    DynamicHandoff(
                        tool_name="route",
                        candidates=[writer],
                        resolver=lambda _ctx, _payload: chosen,
                    )
                ],
            )
        )
        registry.register(AgentSpec(name=writer, instructions="本文を書く"))
        registry.register(AgentSpec(name="reviewer", instructions="レビューする"))
        return registry

    const_registry = _build(_Names.PLANNER, _Names.WRITER, _Names.WRITER)
    raw_registry = _build("planner", "writer", "writer")
    const_handoff = const_registry.get(_Names.PLANNER).handoffs[0]
    raw_handoff = raw_registry.get("planner").handoffs[0]

    const_target = await const_handoff.on_invoke_handoff(None, "{}")
    raw_target = await raw_handoff.on_invoke_handoff(None, "{}")
    assert const_target.name == raw_target.name == "writer"

    # 候補外の名前を返す resolver は定数・生 str のいずれでも ValueError。
    out_of_candidates = _build(_Names.PLANNER, _Names.WRITER, "reviewer")
    handoff = out_of_candidates.get(_Names.PLANNER).handoffs[0]
    with pytest.raises(ValueError, match="reviewer"):
        await handoff.on_invoke_handoff(None, "{}")


# ---------------------------------------------------------------------------
# NextTurnRule / NextTurnPolicy
# ---------------------------------------------------------------------------
class _FakeAgentRef:
    """run 完了結果が持つ Agent の代役（`name` のみ）。"""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeHandoffItem:
    """handoff アイテムの代役（`source_agent` / `target_agent` を持つ）。"""

    def __init__(self, source: str, target: str) -> None:
        self.source_agent = _FakeAgentRef(source)
        self.target_agent = _FakeAgentRef(target)


class _FakeResult:
    """run 完了結果の代役（`last_agent` / `new_items` を持つ）。"""

    def __init__(self, last_agent: Any, new_items: Any) -> None:
        self.last_agent = last_agent
        self.new_items = new_items


def test_定数を_next_turn_の宣言に使っても解決結果が同じ() -> None:
    """`NextTurnPolicy` のキー・`NextTurnRule` の到達元 / 遷移先が定数でも同一に解決される。"""
    const_policy = NextTurnPolicy(
        {_Names.PLANNER: NextTurnRule(next_agent=_Names.WRITER, source=_Names.REVIEWER)}
    )
    raw_policy = NextTurnPolicy({"planner": NextTurnRule(next_agent="writer", source="reviewer")})
    result = _FakeResult(
        last_agent=_FakeAgentRef("planner"),
        new_items=[_FakeHandoffItem("reviewer", "planner")],
    )

    assert resolve_next_agent(const_policy, result) == "writer"
    assert resolve_next_agent(const_policy, result) == resolve_next_agent(raw_policy, result)

    # 到達元が一致しない場合の「上書きなし」も同一。
    other = _FakeResult(
        last_agent=_FakeAgentRef("planner"),
        new_items=[_FakeHandoffItem("writer", "planner")],
    )
    assert resolve_next_agent(const_policy, other) is None
    assert resolve_next_agent(raw_policy, other) is None


def test_定数を_next_turn_の宣言に使っても派生_registry_の結線が同じ() -> None:
    """`apply_next_turn_policy` の名前整合検証と禁止の結線が定数でも同一に働く。"""

    def _build() -> AgentRegistry:
        registry = _make_registry()
        registry.register(
            AgentSpec(name="planner", instructions="計画を立てる", handoffs=["writer"])
        )
        registry.register(AgentSpec(name="writer", instructions="本文を書く"))
        registry.register(AgentSpec(name="reviewer", instructions="レビューする", handoffs=[]))
        return registry

    const_policy = NextTurnPolicy(
        {
            _Names.PLANNER: NextTurnRule(
                next_agent=_Names.WRITER, no_handoff_on_arrival=True, source=_Names.REVIEWER
            )
        }
    )
    raw_policy = NextTurnPolicy(
        {
            "planner": NextTurnRule(
                next_agent="writer", no_handoff_on_arrival=True, source="reviewer"
            )
        }
    )

    const_derived = apply_next_turn_policy(const_policy, _build())
    raw_derived = apply_next_turn_policy(raw_policy, _build())

    assert const_derived.names() == raw_derived.names()
    assert _structure(const_derived.get(_Names.PLANNER)) == _structure(raw_derived.get("planner"))
    # 禁止対象の出辺は Handoff へ昇格する（ゲート合成の差し込み口）。
    assert _structure(const_derived.get(_Names.PLANNER))["handoffs"] == [
        ("handoff", "transfer_to_writer")
    ]


# ---------------------------------------------------------------------------
# entry 名 / 混在 / 位置引数束縛
# ---------------------------------------------------------------------------
def test_定数で宣言した_entry_名が生_str_と同じに解決される() -> None:
    """`entry_name` は登録順の先頭を返すため、定数宣言でも同じ値になる。"""
    const_registry = _make_registry()
    const_registry.register(AgentSpec(name=_Names.PLANNER, instructions="計画を立てる"))
    const_registry.register(AgentSpec(name=_Names.WRITER, instructions="本文を書く"))

    raw_registry = _make_registry()
    raw_registry.register(AgentSpec(name="planner", instructions="計画を立てる"))
    raw_registry.register(AgentSpec(name="writer", instructions="本文を書く"))

    assert const_registry.entry_name == raw_registry.entry_name == "planner"
    assert const_registry.get(const_registry.entry_name).name == "planner"


def test_定数と生_str_を同一_registry_内で混在させても解決に失敗しない() -> None:
    """名前文字列の一致のみで解決されるため、混在宣言でも参照は解決できる。"""
    registry = _make_registry()
    registry.register(
        AgentSpec(name=_Names.PLANNER, instructions="計画を立てる", handoffs=["writer"])
    )
    registry.register(
        AgentSpec(name="writer", instructions="本文を書く", handoffs=[_Names.REVIEWER])
    )
    registry.register(AgentSpec(name=_Names.REVIEWER, instructions="レビューする"))

    registry.validate()  # 例外が出なければ OK
    planner = registry.get("planner")
    assert _structure(planner)["handoffs"] == [("agent", "writer")]
    assert _structure(registry.get(_Names.WRITER))["handoffs"] == [("agent", "reviewer")]


def test_定数を位置引数で渡しても同一の_Agent_構成になる() -> None:
    """`AgentSpec` の位置引数束縛契約（`spec.py:140` の kw_only 配置）を pin する。

    `input_guardrails` / `output_guardrails` / `guardrails` が `kw_only=True` のため、
    8 番目の位置引数は `handoffs` に束縛される。定数の値は str なのでこの契約は不変。
    """
    positional = AgentSpec(
        _Names.PLANNER, "計画を立てる", None, [], None, None, None, [_Names.WRITER]
    )
    keyword = AgentSpec(name="planner", instructions="計画を立てる", handoffs=["writer"])
    assert positional.name == "planner"
    assert positional.handoffs == ["writer"]
    assert positional == keyword

    registry = _make_registry()
    registry.register(positional)
    registry.register(AgentSpec(_Names.WRITER, "本文を書く"))

    raw_registry = _make_registry()
    raw_registry.register(keyword)
    raw_registry.register(AgentSpec("writer", "本文を書く"))

    assert _structure(registry.get(_Names.PLANNER)) == _structure(raw_registry.get("planner"))
