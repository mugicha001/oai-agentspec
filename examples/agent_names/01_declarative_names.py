"""エージェント名の宣言的定数簿（`AgentNames`）の 1 本例。

エージェント名を 1 箇所のクラス属性宣言へ集約し、以降の参照を定数経由にすることで、
タイポを次の 3 点で捕捉する。

- 宣言済み名は登録操作なしに静的なクラス属性として解決でき、`dir()` に載る（エディタ補完が効く）
- 未宣言名アクセスは宣言済み属性名の一覧つき `AttributeError` になる（`Names.PLANER` の類）
- 到達不能な宣言・不正な値・注釈のみの宣言・値の重複はクラス定義時に `ValueError` になる

値は `str` なので、既存の名前参照フィールド（`AgentSpec.name` / `handoffs` /
`handoff_options` のキー / `sub_agents` / `sub_agent_tools` のキー /
`DynamicHandoff.candidates` / `NextTurnRule` の到達元・遷移先 / entry 名）へ変換なしに渡せる。
生 str による従来の宣言はそのまま有効で、定数簿の利用は任意（opt-in）である。

`validate_agent_names` は定数簿の宣言集合と `AgentRegistry` の登録名集合を突き合わせ、
両方向の差分（宣言のみ / 登録のみ）を全件集約して単一の `KeyError` で報告する。
`registry.validate()`（参照の解決可否）とは別の検査で、両方を run 前に呼べる。

モデル呼び出しは行わない（実 API を使わない）。

実行:
    uv run python examples/agent_names/01_declarative_names.py
"""

from __future__ import annotations

from oai_agentspec import (
    AgentNames,
    AgentRegistry,
    AgentSpec,
    validate_agent_names,
)


class Names(AgentNames):
    """本アプリのエージェント名（宣言はここ 1 箇所）。"""

    PLANNER = "planner"
    WRITER = "writer"
    REVIEWER = "reviewer"


def show_declaration() -> None:
    """宣言済み名の解決・一覧・str としての扱いを示す。"""
    print("--- 宣言")
    print(f"Names.PLANNER = {Names.PLANNER!r}（型: {type(Names.PLANNER).__name__}）")
    print(f"Names.names() = {Names.names()}")
    print(f"dir() に含まれる: {'PLANNER' in dir(Names)}")


def show_typo_detection() -> None:
    """未宣言名アクセスが一覧つき `AttributeError` になることを示す。"""
    print("--- タイポ検出")
    try:
        Names.PLANER  # noqa: B018  # 意図的なタイポ（PLANNER の打ち間違い）
    except AttributeError as exc:
        print(f"AttributeError: {exc}")


def show_definition_time_rejection() -> None:
    """クラス定義時に弾かれる宣言を示す（値の重複・注釈のみ）。"""
    print("--- 定義時の拒否")
    try:
        type(Names)("Dup", (AgentNames,), {"PLANNER": "planner", "PLANER": "planner"})
    except ValueError as exc:
        print(f"値の重複 -> ValueError: {exc}")

    try:
        type(Names)("AnnotationOnly", (AgentNames,), {"__annotations__": {"PLANNER": str}})
    except ValueError as exc:
        print(f"注釈のみ -> ValueError: {exc}")


def show_reference_paths() -> AgentRegistry:
    """全参照経路で定数を使って registry を組む。

    Returns:
        定数だけで宣言した registry。
    """
    print("--- 参照（定数だけで宣言する）")
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            Names.PLANNER,
            "計画を立てる",
            handoffs=[Names.WRITER],
            sub_agents=[Names.REVIEWER],
        )
    )
    registry.register(AgentSpec(Names.WRITER, "本文を書く"))
    registry.register(AgentSpec(Names.REVIEWER, "レビューする"))
    print(f"registry.names() = {registry.names()}")
    print(f"entry_name       = {registry.entry_name!r}")
    return registry


def show_consistency_check(registry: AgentRegistry) -> None:
    """整合検査の成功と、差分がある場合の両方向報告を示す。"""
    print("--- 整合検査")
    validate_agent_names(Names, registry)
    registry.validate()
    print("差分 0 件（例外なし）")

    class Drifted(AgentNames):
        """registry とずれた定数簿（宣言のみ / 登録のみが同時に起きる例）。"""

        PLANNER = "planner"
        WRITER = "writer"
        ARCHIVER = "archiver"  # 宣言のみ（registry に無い）

    try:
        # registry 側には reviewer があり定数簿に無い（登録のみ）
        validate_agent_names(Drifted, registry)
    except KeyError as exc:
        print(f"KeyError: {exc.args[0]}")


def main() -> None:
    """定数簿の宣言・タイポ検出・定義時拒否・参照・整合検査を順に示す。"""
    show_declaration()
    show_typo_detection()
    show_definition_time_rejection()
    registry = show_reference_paths()
    show_consistency_check(registry)


if __name__ == "__main__":
    main()
