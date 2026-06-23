"""実行時の再構成をユーザー操作で示す例（モデル呼び出しなし）。

カバーする機能:
    1. from_specs（AgentSpec.handoffs 宣言から HandoffGraph を導出）
    2. HandoffGraph の編集 + 再 apply によるトポロジ再構成（apply は replace 適用）
    3. update（同名 spec の置換）/ unregister（削除）と validate による参照ミス検出

Agent の構築はネットワークを要しないため、本例は環境変数なしで実行できる:
    uv run python examples/runtime_update.py
"""

from __future__ import annotations

from oai_agentspec import AgentRegistry, AgentSpec, from_specs


def handoff_targets(registry: AgentRegistry, name: str) -> list[str]:
    # handoffs は Agent（設定なしエッジ）と Handoff（設定付きエッジ）の混在。
    names: list[str] = []
    for h in registry.get(name).handoffs:
        names.append(getattr(h, "tool_name", None) or f"transfer_to_{h.name}")
    return names


def main() -> None:
    registry = AgentRegistry()

    # --- 1. from_specs: spec の handoffs 宣言からグラフを導出 ---
    specs = [
        AgentSpec(name="triage", instructions="依頼を振り分ける。", handoffs=["billing"]),
        AgentSpec(name="billing", instructions="請求に対応する。"),
        AgentSpec(name="support", instructions="技術問い合わせに対応する。"),
    ]
    for spec in specs:
        registry.register(spec)

    graph = from_specs(specs, entry="triage")
    graph.apply(registry)
    registry.validate()
    print("初期 handoffs:", handoff_targets(registry, "triage"))  # -> transfer_to_billing

    # --- 2. グラフを編集して再 apply（replace でトポロジを上書き）---
    graph.edge("triage", "support", description="技術問い合わせはこちら")
    graph.apply(registry)
    print("再 apply 後 handoffs:", handoff_targets(registry, "triage"))  # billing + support

    # --- 3a. update: 同名 spec を置換（次回 get から反映）---
    registry.update(AgentSpec(name="billing", instructions="請求・返金・領収書に対応する。"))
    print("update 後の billing:", registry.get("billing").instructions)

    # --- 3b. unregister + validate: ぶら下がり参照を検出 ---
    registry.unregister("support")
    try:
        registry.validate()
    except KeyError as exc:
        print("validate が参照ミスを検出:", exc)


if __name__ == "__main__":
    main()
