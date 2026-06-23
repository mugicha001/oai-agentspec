"""実行トレース捕捉窓口（横断 routing / ツール使用評価用・SDK 結合を閉じる・NFR-1）。

生 `RunResult` を `_adapters` 内で消費し、`new_items` を 1 パス走査して plain
`ObservedRoute` / `RouteStep`（`HandoffOutputItem` の source / target 由来）と
`ObservedToolCall`（`ToolCallItem` 相当のツール名由来）を抽出する。SDK 型
（`RunResult` / `HandoffOutputItem` / `ToolCallItem`）を `_adapters` 外へ出さない。

抽出は防御的 `getattr`（`runner._extract_pending` と同型）で行い、SDK 退行に耐性を持たせる。
plain 戻り型（`ObservedRun` 等）は `runtime/llmops/types` 由来だが、`_adapters` は当該 plain
型を read-only で import するのみで `runtime/llmops` のロジック層には依存しない。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..runtime.llmops.types import ObservedRun


def _agent_name(agent: Any) -> str:
    """SDK Agent から名前を防御的に取り出す（属性消失時は空文字）。

    Args:
        agent: SDK Agent（`name` 属性を持つ前提・防御的に取得）。

    Returns:
        エージェント名。取得不能なら空文字。
    """
    name = getattr(agent, "name", None)
    return "" if name is None else str(name)


def observe_run_result(result: Any) -> ObservedRun:
    """生 `RunResult` から plain `ObservedRun`（route + tool_calls）を 1 パス抽出する。

    `new_items` を走査し、handoff アイテム（`source_agent` / `target_agent` 属性を持つもの）から
    `RouteStep` を、ツール呼び出しアイテム（`tool_name` プロパティ / `raw_item.name` を持つもの）
    から `ObservedToolCall` を組む。`last_agent` は `RunResult.last_agent` の名前を反映する。

    アイテム判定は `type` 文字列の完全一致に依存せず **属性の有無**（`_extract_pending` と同型の
    防御的 `getattr`）で行い、SDK の `type` リテラル変更で静かに取りこぼすのを避ける。

    観測経路（`steps`）は**起点（最初の handoff の source）を先頭に**含め、handoff の遷移先を順に
    並べ、末尾に最終応答 agent（`last_agent`）を付加して**起点・経由・最終を網羅したエージェント
    系列**にする（`expected_route` との順序比較に使う・フルパス指定）。起点は遷移先として記録され
    ないため明示的に先頭へ付加する（起点が自身に応答した＝handoff が無い場合は最終 agent として
    末尾に現れる）。handoff が 1 件も無いときは最終 agent の単一ステップに倒す。同一 agent の二重
    付加（最後の handoff 先が last_agent と一致）はしない。起点 source 名が取れない / 起点と最初の
    遷移先が同一の場合は起点 prepend をしない（防御的）。

    Args:
        result: SDK `RunResult`（`last_agent` / `new_items` を持つ前提）。

    Returns:
        plain `ObservedRun`（SDK 型を含まない・`route.steps` 末尾に last_agent を含む）。
    """
    from ..runtime.llmops.types import (
        ObservedRoute,
        ObservedRun,
        ObservedToolCall,
        RouteStep,
    )

    last_agent_name = _agent_name(getattr(result, "last_agent", None))
    items = list(getattr(result, "new_items", None) or [])

    steps: list[RouteStep] = []
    tool_calls: list[ObservedToolCall] = []
    for item in items:
        # handoff 判定: source/target agent 属性の有無で見る（type 文字列に依存しない）。
        has_source = hasattr(item, "source_agent")
        has_target = hasattr(item, "target_agent")
        if has_source and has_target:
            source = _agent_name(getattr(item, "source_agent", None))
            target = _agent_name(getattr(item, "target_agent", None))
            steps.append(RouteStep(agent=target, handoff_from=source or None))
            continue
        # tool 呼び出し判定: tool 名属性の有無で見る（type 文字列に依存しない）。
        tool_name = getattr(item, "tool_name", None)
        if tool_name is None:
            raw = getattr(item, "raw_item", None)
            tool_name = getattr(raw, "name", None)
        if tool_name is not None:
            tool_calls.append(ObservedToolCall(tool=str(tool_name)))

    # 経路の先頭に起点（最初の handoff の source）を含める。起点は遷移先として記録されないため
    # 明示的に prepend する（起点が自身に応答した＝handoff が無い場合は last_agent として末尾に
    # 現れる）。source 名が取れない / 起点と最初の遷移先が同一なら prepend しない（防御的）。
    if steps and steps[0].handoff_from and steps[0].handoff_from != steps[0].agent:
        entry = steps[0].handoff_from
        steps.insert(0, RouteStep(agent=entry, handoff_from=None))

    # 観測経路のエージェント系列の末尾に最終応答 agent を含める。
    if last_agent_name and (not steps or steps[-1].agent != last_agent_name):
        steps.append(RouteStep(agent=last_agent_name, handoff_from=None))

    route = ObservedRoute(steps=steps, last_agent=last_agent_name)
    return ObservedRun(route=route, tool_calls=tool_calls)
