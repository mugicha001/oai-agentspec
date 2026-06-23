"""L1: 実行トレース捕捉（`_adapters.routing.observe_run_result`）の決定的検証。

fake `RunResult`（`.last_agent` に `.name` を持つダミー agent・`.new_items` に source_agent /
target_agent を持つダミー HandoffOutputItem・tool 名を持つダミー ToolCallItem 相当）で叩き、
plain `ObservedRoute` / `RouteStep` / `ObservedToolCall` への変換・属性ベース判定・last_agent
末尾正規化付加を網羅する。SDK 型 / 実通信に非依存。
"""

from __future__ import annotations

from typing import Any

import pytest

from oai_agentspec._adapters import observe_run_result
from oai_agentspec.runtime.llmops import ObservedRun, ObservedToolCall, RouteStep

pytestmark = pytest.mark.unit


class _FakeAgent:
    """`name` 属性のみを持つダミー agent。"""

    def __init__(self, name: str | None) -> None:
        self.name = name


class _FakeHandoffItem:
    """source_agent / target_agent を持つダミー HandoffOutputItem。"""

    def __init__(self, source: str | None, target: str | None) -> None:
        self.source_agent = _FakeAgent(source)
        self.target_agent = _FakeAgent(target)


class _FakeToolItemToolName:
    """`tool_name` プロパティを持つダミー ToolCallItem 相当。"""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name


class _FakeRawItem:
    """`name` 属性を持つ raw_item（SDK の function_call 相当）。"""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeToolItemRaw:
    """`raw_item.name` 経由でツール名を持つダミー ToolCallItem 相当（tool_name なし）。"""

    def __init__(self, name: str) -> None:
        self.raw_item = _FakeRawItem(name)


class _FakeRunResult:
    """`last_agent` / `new_items` を持つ最小ダミー RunResult。"""

    def __init__(self, last_agent: str | None, new_items: list[Any] | None) -> None:
        self.last_agent = _FakeAgent(last_agent)
        self.new_items = new_items


def test_no_items_yields_single_last_agent_step() -> None:
    """new_items が空なら最終応答 agent の単一ステップに倒す。"""
    observed = observe_run_result(_FakeRunResult("solo", []))
    assert isinstance(observed, ObservedRun)
    assert observed.route.last_agent == "solo"
    assert observed.route.steps == [RouteStep(agent="solo", handoff_from=None)]
    assert observed.tool_calls == []


def test_handoff_item_produces_route_step_with_source() -> None:
    """handoff アイテムは起点を先頭に・target を agent・source を handoff_from にした経路を生む。"""
    items = [_FakeHandoffItem(source="triage", target="billing")]
    observed = observe_run_result(_FakeRunResult("billing", items))
    # 起点 triage を先頭に prepend し遷移先 billing が続く（末尾 last_agent と一致で二重付加なし）。
    assert observed.route.steps == [
        RouteStep(agent="triage", handoff_from=None),
        RouteStep(agent="billing", handoff_from="triage"),
    ]
    assert observed.route.last_agent == "billing"


def test_last_agent_appended_when_differs_from_last_step() -> None:
    """最後の handoff 先が last_agent と異なれば末尾に付加する（起点も先頭に含む）。"""
    items = [_FakeHandoffItem(source="triage", target="billing")]
    observed = observe_run_result(_FakeRunResult("support", items))
    assert observed.route.steps == [
        RouteStep(agent="triage", handoff_from=None),
        RouteStep(agent="billing", handoff_from="triage"),
        RouteStep(agent="support", handoff_from=None),
    ]


def test_last_agent_not_double_appended_when_same() -> None:
    """最後の handoff 先が last_agent と同一なら二重付加しない（起点 a を先頭に含む）。"""
    items = [
        _FakeHandoffItem(source="a", target="b"),
        _FakeHandoffItem(source="b", target="c"),
    ]
    observed = observe_run_result(_FakeRunResult("c", items))
    assert observed.route.steps == [
        RouteStep(agent="a", handoff_from=None),
        RouteStep(agent="b", handoff_from="a"),
        RouteStep(agent="c", handoff_from="b"),
    ]


def test_tool_call_via_tool_name_attribute() -> None:
    """tool_name 属性を持つアイテムから ObservedToolCall を抽出する。"""
    items = [_FakeToolItemToolName("search")]
    observed = observe_run_result(_FakeRunResult("bot", items))
    assert ObservedToolCall(tool="search") in observed.tool_calls


def test_tool_call_via_raw_item_name_fallback() -> None:
    """tool_name が無くても raw_item.name 経由でツール名を抽出する（フォールバック）。"""
    items = [_FakeToolItemRaw("lookup")]
    observed = observe_run_result(_FakeRunResult("bot", items))
    assert observed.tool_calls == [ObservedToolCall(tool="lookup")]


def test_tool_calls_preserve_order() -> None:
    """複数ツール呼び出しは観測順を保持する。"""
    items = [_FakeToolItemToolName("first"), _FakeToolItemToolName("second")]
    observed = observe_run_result(_FakeRunResult("bot", items))
    assert observed.tool_calls == [
        ObservedToolCall(tool="first"),
        ObservedToolCall(tool="second"),
    ]


def test_mixed_handoff_and_tool_items() -> None:
    """handoff とツール呼び出しが混在しても 1 パスで両方抽出する（経路は起点込み）。"""
    items = [
        _FakeToolItemToolName("search"),
        _FakeHandoffItem(source="triage", target="billing"),
    ]
    observed = observe_run_result(_FakeRunResult("billing", items))
    assert observed.tool_calls == [ObservedToolCall(tool="search")]
    assert observed.route.steps == [
        RouteStep(agent="triage", handoff_from=None),
        RouteStep(agent="billing", handoff_from="triage"),
    ]


def test_multi_hop_route_includes_entry_and_all_targets() -> None:
    """多段 handoff（triage->billing->escalation）は起点 + 全遷移先のフルパスになる。"""
    items = [
        _FakeHandoffItem(source="triage", target="billing"),
        _FakeHandoffItem(source="billing", target="escalation"),
    ]
    observed = observe_run_result(_FakeRunResult("escalation", items))
    assert [step.agent for step in observed.route.steps] == ["triage", "billing", "escalation"]


def test_handoff_with_missing_source_uses_none() -> None:
    """source_agent の名前が None なら handoff_from は None・起点 prepend もしない（防御的）。"""
    items = [_FakeHandoffItem(source=None, target="billing")]
    observed = observe_run_result(_FakeRunResult("billing", items))
    assert observed.route.steps == [RouteStep(agent="billing", handoff_from=None)]


def test_missing_last_agent_name_yields_empty() -> None:
    """last_agent の名前が None なら last_agent は空文字・末尾付加もしない。"""
    observed = observe_run_result(_FakeRunResult(None, []))
    assert observed.route.last_agent == ""
    assert observed.route.steps == []


def test_unrecognized_item_is_ignored() -> None:
    """handoff でもツールでもないアイテムは無視される（取りこぼし＝ステップ生成なし）。"""

    class _Plain:
        pass

    observed = observe_run_result(_FakeRunResult("bot", [_Plain()]))
    assert observed.tool_calls == []
    # last_agent のみが単一ステップとして残る。
    assert observed.route.steps == [RouteStep(agent="bot", handoff_from=None)]


def test_new_items_none_treated_as_empty() -> None:
    """new_items が None でも空扱いにし last_agent ステップに倒す（防御的）。"""
    observed = observe_run_result(_FakeRunResult("bot", None))
    assert observed.route.steps == [RouteStep(agent="bot", handoff_from=None)]
    assert observed.tool_calls == []
