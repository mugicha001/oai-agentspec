"""RealtimeAgentSpec を宣言し RealtimeAgentRegistry で RealtimeAgent を構築する最小例。

Realtime 専用宣言ルート（`oai_agentspec.realtime`）の基本フローを示す:

    RealtimeAgentSpec 宣言 -> RealtimeHandoffGraph でトポロジ宣言 -> apply(specs)
    -> RealtimeAgentRegistry.register -> validate
    -> get で RealtimeAgent を遅延構築（handoffs は 2 パスで後付け結線）

ハンドオフは spec の `handoffs` に直接書いても、グラフ DSL（本例）で宣言しても
構造的に同一の結線になる。グラフは `mermaid()` でトポロジを可視化できる。

実 API（OpenAI Realtime）へは接続しない。宣言と build-time 検証・構築までで完結し、
構築された RealtimeAgent のフィールドを print で確認するだけの例。

RealtimeAgent が非対応とするフィールド（model / model_settings / voice 等の実行時 Config）は
`RealtimeAgentSpec` が型として持たない。それらは実行時に `RealtimeRunner` へ渡す責務であり、
handoff_session.py で扱う。

実行:
    uv run python examples/realtime/basic_declaration.py
"""

from __future__ import annotations

from oai_agentspec.realtime import (
    RealtimeAgentRegistry,
    RealtimeAgentSpec,
    RealtimeHandoffGraph,
)


def build_graph() -> RealtimeHandoffGraph:
    """triage <-> support の相互 handoff（循環）をノードとエッジで宣言する。

    Returns:
        エッジ宣言済みの RealtimeHandoffGraph（entry は triage）。
    """
    graph = RealtimeHandoffGraph(entry="triage")
    graph.edge(
        "triage",
        "support",
        tool_name="transfer_to_support",
        tool_description="技術的な問い合わせをサポート担当へ引き継ぐ。",
    )
    # 相互参照（triage <-> support の循環）。registry の 2 パス遅延バインドが解決する
    graph.edge("support", "triage")
    return graph


def build_registry() -> RealtimeAgentRegistry:
    """spec 宣言 + グラフ適用で 2 エージェントを登録し registry を返す。

    Returns:
        validate 済みの RealtimeAgentRegistry（entry は最初に登録した triage）。
    """
    # spec はエージェントの中身のみ宣言する（トポロジはグラフ側の責務）。
    specs = [
        RealtimeAgentSpec(
            name="triage",
            instructions="ユーザーの要望を聞き取り、必要ならサポート担当へ引き継ぐ受付担当。",
            handoff_description="最初の受付・振り分け担当。",
        ),
        RealtimeAgentSpec(
            name="support",
            instructions=(
                "製品の技術的な問い合わせに答えるサポート担当。"
                "解決したら、または技術以外の話題になったら受付担当へ戻す。"
            ),
            handoff_description="技術サポート担当。",
        ),
    ]

    # グラフを spec 群へ一括反映する（spec.handoffs 直接宣言と同一の結線になる）。
    graph = build_graph()
    graph.apply(specs)
    print("--- mermaid ---")
    print(graph.mermaid())

    registry = RealtimeAgentRegistry()
    for spec in specs:
        registry.register(spec)

    # handoffs 参照のタイポ等を build 前に一括検出する。
    registry.validate()
    return registry


def main() -> None:
    registry = build_registry()

    print("--- registry ---")
    print("登録エージェント:", registry.names())
    print("entry:", registry.entry_name)

    # get で遅延構築（到達可能な spec を 2 パスで build し handoff を後付け結線）。
    triage = registry.get("triage")
    support = registry.get("support")

    print("--- built RealtimeAgent ---")
    print("triage.name:", triage.name)
    print("triage.handoff_description:", triage.handoff_description)
    print("triage.handoffs (tool 名):", [h.tool_name for h in triage.handoffs])
    print("support.name:", support.name)
    print("support.handoffs:", [h.tool_name for h in support.handoffs])


if __name__ == "__main__":
    main()
