"""L2: `Runner.run_streamed` の完了結果でも FR-2 の判定材料が読めることの pin。

`resolve_next_agent` / `next_turn_agent` は run 完了結果を `extract_turn_observation` で
ダックタイピング読みするため、非 streaming（`RunResult`）と streaming
（`RunResultStreaming`）のどちらを渡しても同じ観測が得られる必要がある。SDK の
`RunResultStreaming` は `RunResultBase` を継承して `new_items` を持ち、`last_agent` を
property として公開する（非 streaming と同名・同義）。この構造が SDK upgrade で変わると、
**例外も型エラーも出ないまま streaming 経路だけ上書きが効かなくなる**ため、実型で固定する。

あわせて `last_agent` が「参照解放後にアクセスすると例外を送出する property」である実型の
事実（`RunResult` / `RunResultStreaming` の双方）を pin する。属性欠落だけでなくアクセス時の
例外まで安全側へ倒す防御的読み取り（NFR-5）が机上の仮定ではないことの根拠になる。

`agents` を import するため integration マーカー（`tests/_adapters/test_builders_l2.py` と
同じ扱い）。
"""

from __future__ import annotations

import dataclasses
import gc
import inspect
from typing import Any

import pytest
from agents import Agent, RunContextWrapper
from agents.items import HandoffOutputItem
from agents.result import RunResult, RunResultStreaming

from oai_agentspec._adapters.next_turn import extract_turn_observation

pytestmark = pytest.mark.integration


def _handoff_item(source: Agent, target: Agent) -> HandoffOutputItem:
    """実 SDK 型のハンドオフアイテムを 1 件組む。

    Args:
        source: 遷移元エージェント。
        target: 遷移先エージェント。

    Returns:
        `source_agent` / `target_agent` を持つ実 `HandoffOutputItem`。
    """
    return HandoffOutputItem(
        agent=source,
        raw_item={"role": "user", "content": "handoff"},
        source_agent=source,
        target_agent=target,
    )


def _base_fields(last_agent_items: list[Any]) -> dict[str, Any]:
    """`RunResultBase` 由来の必須フィールドをまとめて組む。

    Args:
        last_agent_items: 完了結果の `new_items` に載せるアイテム列。

    Returns:
        `RunResult` / `RunResultStreaming` に共通で渡すキーワード引数。
    """
    return {
        "input": "hello",
        "new_items": last_agent_items,
        "raw_responses": [],
        "final_output": None,
        "input_guardrail_results": [],
        "output_guardrail_results": [],
        "tool_input_guardrail_results": [],
        "tool_output_guardrail_results": [],
        "context_wrapper": RunContextWrapper(context=None),
    }


def _streaming_result(last_agent: Agent, items: list[Any]) -> RunResultStreaming:
    """streaming の完了結果を組む。

    Args:
        last_agent: 最終回答者（`current_agent` として渡す）。
        items: `new_items` に載せるアイテム列。

    Returns:
        実 `RunResultStreaming`。
    """
    return RunResultStreaming(
        **_base_fields(items),
        current_agent=last_agent,
        current_turn=1,
        max_turns=10,
        _current_agent_output_schema=None,
        trace=None,
    )


# ---------------------------------------------------------------------------
# 構造契約: streaming 完了結果も同名の判定材料を持つ
# ---------------------------------------------------------------------------


def test_streaming結果はnew_itemsフィールドを持つ() -> None:
    """`RunResultStreaming` は `RunResultBase` 由来の `new_items` を持つ。"""
    names = {f.name for f in dataclasses.fields(RunResultStreaming)}

    assert "new_items" in names


def test_last_agentは両結果型でpropertyとして公開される() -> None:
    """`last_agent` は非 streaming / streaming の双方で property（読み出しに副作用がありうる）。"""
    assert isinstance(inspect.getattr_static(RunResult, "last_agent"), property)
    assert isinstance(inspect.getattr_static(RunResultStreaming, "last_agent"), property)


# ---------------------------------------------------------------------------
# 観測の同等性（streaming / 非 streaming で同じ判定材料が得られる）
# ---------------------------------------------------------------------------


def test_streaming結果から最終回答者と遷移が観測できる() -> None:
    """streaming の完了結果でも `(遷移元, 遷移先)` と最終回答者名が抽出できる。"""
    triage, billing = Agent(name="triage"), Agent(name="billing")

    observation = extract_turn_observation(
        _streaming_result(billing, [_handoff_item(triage, billing)])
    )

    assert observation.last_agent == "billing"
    assert observation.handoffs == (("triage", "billing"),)


def test_streamingと非streamingで同じ観測になる() -> None:
    """同じ内容なら結果型が違っても観測は一致する（経路差で上書き判定がぶれない）。"""
    triage, billing = Agent(name="triage"), Agent(name="billing")

    streamed = extract_turn_observation(
        _streaming_result(billing, [_handoff_item(triage, billing)])
    )
    non_streamed = extract_turn_observation(
        RunResult(**_base_fields([_handoff_item(triage, billing)]), _last_agent=billing)
    )

    assert streamed.last_agent == non_streamed.last_agent == "billing"
    assert streamed.handoffs == non_streamed.handoffs == (("triage", "billing"),)


# ---------------------------------------------------------------------------
# 防御的読み取りの実型根拠（解放後の last_agent は例外を送出する）
# ---------------------------------------------------------------------------


def test_解放後のlast_agentは例外を送出する() -> None:
    """`last_agent` は参照解放後のアクセスで例外を送出する（属性欠落ではない）。"""
    result = RunResult(**_base_fields([]), _last_agent=Agent(name="billing"))
    result._release_last_agent_reference()  # noqa: SLF001 - 解放後アクセスの実型再現
    gc.collect()

    with pytest.raises(Exception, match="no longer available"):
        _ = result.last_agent


def test_解放後の結果からの抽出は例外にならずNoneへ倒す() -> None:
    """実型の例外送出 property でも抽出は例外を伝播させず安全側（None）へ倒す（NFR-5）。"""
    result = RunResult(**_base_fields([]), _last_agent=Agent(name="billing"))
    result._release_last_agent_reference()  # noqa: SLF001 - 解放後アクセスの実型再現
    gc.collect()

    observation = extract_turn_observation(result)

    assert observation.last_agent is None
    assert observation.handoffs == ()
