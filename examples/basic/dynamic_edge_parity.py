"""動的エッジで `on_handoff` / `input_type` / `input_filter` / `is_enabled` / `options` を
実 LLM 経由で exercise する E2E 例。

通常の `dynamic_edge` は resolver による転送先選択のみだが、本例ではすべての per-edge 設定を
指定し、LLM がハンドオフ tool を呼ぶたびに以下が起きることを観測する:

  - LLM は `EscalationInput`（Pydantic）の各 `Field(description=...)` を見て構造化引数を埋める
  - resolver（`route_by_priority`）は raw `input_json` で priority を見て billing / support を決定
  - `on_handoff_callback` が parsed `EscalationInput` を受け取り副作用として記録する
  - `input_filter`（`remove_all_tools`）で次エージェントの履歴から tool call / tool output が落ちる
  - `is_enabled=True` でハンドオフ tool が提示される
  - `options={"nest_handoff_history": True}` で履歴ネスト挙動が有効化される

実行:
    uv run python examples/basic/dynamic_edge_parity.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Literal

from agents import Runner
from agents.extensions.handoff_filters import remove_all_tools
from pydantic import BaseModel, Field

from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402
from _run_path import print_run_path  # noqa: E402

CANDIDATES = ["billing", "support"]


class EscalationInput(BaseModel):
    """ハンドオフ tool が LLM に埋めさせる構造化引数。

    各フィールドの `Field(description=...)` とモデル docstring は JSON Schema 経由で
    ハンドオフ tool の `parameters` として LLM に届く。
    """

    priority: Literal["low", "high"] = Field(
        description=(
            "エスカレーションの優先度。high の場合は請求担当、"
            "low の場合はサポート担当へ振り分ける。"
        ),
    )
    reason: str = Field(
        description="エスカレーション理由（顧客への説明にも使える形で）",
    )


def route_by_priority(context: Any, input_json: str) -> str:
    """resolver は raw `input_json` を受け取る現行契約のまま。

    `input_type` 指定時でも resolver には未パース文字列が渡るため、必要なら自前で
    `json.loads` する（parsed オブジェクトは on_handoff にだけ届く）。
    """
    payload = json.loads(input_json) if input_json else {}
    return "billing" if payload.get("priority") == "high" else "support"


_HANDOFF_TRACE: list[dict[str, str]] = []


def on_handoff_callback(context: Any, parsed: EscalationInput) -> None:
    """転送先決定の直後に発火し、parsed Pydantic オブジェクトを受け取る副作用フック。"""
    record = {"priority": parsed.priority, "reason": parsed.reason}
    _HANDOFF_TRACE.append(record)
    print(f"[on_handoff] parsed: priority={parsed.priority!r} reason={parsed.reason!r}")


def build_registry() -> tuple[AgentRegistry, HandoffGraph]:
    registry = AgentRegistry()
    model = azure_model()

    registry.register(
        AgentSpec(
            name="triage",
            instructions=(
                "あなたは振り分け担当。ユーザーの問い合わせから優先度を判断し、"
                "`route` ハンドオフ tool を呼んで担当へ転送する。"
                "請求の話なら high、それ以外は low。"
            ),
            model=model,
        )
    )
    registry.register(
        AgentSpec(
            name="billing",
            instructions="あなたは請求担当。優先度 high の問い合わせに簡潔に回答する。",
            model=model,
        )
    )
    registry.register(
        AgentSpec(
            name="support",
            instructions="あなたはサポート担当。優先度 low の問い合わせに簡潔に回答する。",
            model=model,
        )
    )

    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        CANDIDATES,
        route_by_priority,
        tool_name="route",
        description="エスカレーション情報に基づき担当を実行時に決定する",
        on_handoff=on_handoff_callback,
        input_type=EscalationInput,
        input_filter=remove_all_tools,
        is_enabled=True,
        options={"nest_handoff_history": True},
    )
    graph.apply(registry)
    registry.validate()
    return registry, graph


async def main() -> None:
    registry, graph = build_registry()
    print("--- handoff graph ---")
    print(graph.mermaid())
    print("---------------------")

    entry = graph.entry_agent(registry)

    print("\n=== ケース1: 請求関連（high 優先度 → billing） ===")
    result = await Runner.run(entry, input="先月の請求書のPDFが届かないので至急対応してほしい")
    print_run_path(result)
    print(f"最終応答: {result.final_output}")

    print("\n=== ケース2: 技術質問（low 優先度 → support） ===")
    result = await Runner.run(entry, input="ログイン画面の表示が崩れている気がします")
    print_run_path(result)
    print(f"最終応答: {result.final_output}")

    print("\n--- on_handoff 発火履歴 ---")
    for i, rec in enumerate(_HANDOFF_TRACE, 1):
        print(f"  {i}. priority={rec['priority']!r} reason={rec['reason']!r}")

    # ケース3: is_enabled=False で同一構成を組み、ハンドオフ tool が LLM に提示されないため
    # triage が転送せず自分で応答することを観測する（is_enabled の効きを単独で検証）。
    print("\n=== ケース3: is_enabled=False（ハンドオフ tool を LLM に提示しない） ===")
    registry2 = AgentRegistry()
    model = azure_model()
    registry2.register(
        AgentSpec(
            name="triage",
            instructions=(
                "あなたは振り分け担当。`route` ハンドオフ tool があれば優先度を判断して呼ぶ。"
                "tool が無ければ自分でユーザーに簡潔に応答する。"
            ),
            model=model,
        )
    )
    registry2.register(
        AgentSpec(name="billing", instructions="あなたは請求担当。", model=model)
    )
    graph2 = HandoffGraph(entry="triage")
    graph2.dynamic_edge(
        "triage",
        ["billing"],
        route_by_priority,
        tool_name="route",
        description="エスカレーション情報に基づき担当を決定する",
        on_handoff=on_handoff_callback,
        input_type=EscalationInput,
        is_enabled=False,
    )
    graph2.apply(registry2)
    registry2.validate()
    pre_trace_len = len(_HANDOFF_TRACE)
    result3 = await Runner.run(
        graph2.entry_agent(registry2),
        input="先月の請求書のPDFが届かないので至急対応してほしい",
    )
    print_run_path(result3)
    print(f"最終応答: {result3.final_output}")
    handoff_fired = len(_HANDOFF_TRACE) > pre_trace_len
    assert not handoff_fired, "is_enabled=False なのに on_handoff が発火している（バグ）"
    print(
        "[verify] is_enabled=False: on_handoff 未発火 / 最終エージェント=",
        result3.last_agent.name,
        "（triage のまま＝ハンドオフ tool が LLM に提示されなかった）",
    )


if __name__ == "__main__":
    asyncio.run(main())
