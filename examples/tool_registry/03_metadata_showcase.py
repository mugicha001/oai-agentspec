"""ToolSpec の全メタデータの実演: SDK FunctionTool にどう反映されるかを確認する。

01/02 では enabled / needs_approval / timeout のみを扱ったが、本例では以下を追加で示す:
- failure_error_function の 3 値（未指定 = SDK 既定 / callable 指定 / None 明示 = 例外素通し）
- name_override / description_override（Registry 登録名と LLM 提示名の分離）
- strict_mode（Structured Outputs の厳格モード切替）
- extra 素通し（`defer_loading` 等の SDK 引数を型付きフィールド外で渡す）

Runner の実行は行わず、`build_function_tool` 後の FunctionTool 属性の反映を print で確認する
（Azure OpenAI の API キー不要）。

Usage:
    uv run python examples/tool_registry/03_metadata_showcase.py
"""

from __future__ import annotations

from oai_agentspec import ToolRegistry, ToolSpec


# --- Tool 関数群（純関数） -----------------------------------------------
async def search_web(query: str) -> str:
    """Web 検索。"""
    return f"results for {query!r}"


async def charge_card(amount: int) -> str:
    """課金処理（副作用あり）。"""
    return f"charged: {amount}"


def _custom_failure(_ctx: object, err: Exception) -> str:
    """独自の失敗文言 formatter（モデルへ返すエラー文字列を組み立て）。"""
    return f"[custom] tool failed: {type(err).__name__}: {err}"


def main() -> None:
    reg = ToolRegistry()

    # 1) failure_error_function 未指定 → SDK 既定 formatter に委ねる
    #    （Registry は kwarg を渡さない・None-omission）
    reg.register(ToolSpec(func=search_web, name="search_default"))

    # 2) failure_error_function に callable 指定 → モデルへ返す文言をアプリで組み立て
    reg.register(
        ToolSpec(
            func=search_web,
            name="search_custom_fmt",
            failure_error_function=_custom_failure,
        )
    )

    # 3) failure_error_function=None を明示 → SDK 側で例外を素通し（呼び出し元へ生例外）
    #    Runner 側で try/except したい場合や、Resilience 層に渡したい場合の入口。
    reg.register(
        ToolSpec(
            func=search_web,
            name="search_raw_error",
            failure_error_function=None,
        )
    )

    # 4) name_override / description_override で LLM 提示名と説明を上書き
    reg.register(
        ToolSpec(
            func=charge_card,
            name="charge_v2",  # Registry の登録キー（属性アクセス名）
            name_override="charge_card_v2",  # LLM 提示名（Registry 名と独立）
            description_override="Charge a credit card. Requires prior user confirmation.",
        )
    )

    # 5) strict_mode=False で Structured Outputs の厳格モードを解除
    #    （optional 引数だらけの既存関数を Tool にしたいときの逃げ道）
    reg.register(
        ToolSpec(
            func=search_web,
            name="search_lax",
            strict_mode=False,
        )
    )

    # 6) extra 素通しで型付きフィールド外の SDK 引数を渡す（例: defer_loading）
    reg.register(
        ToolSpec(
            func=search_web,
            name="search_deferred",
            extra={"defer_loading": True},
        )
    )

    # --- FunctionTool への反映を確認 ---
    print(f"[registered] {reg.names()}\n")

    for name in reg.names():
        tool = getattr(reg, name)
        print(f"# {name}")
        print(f"  tool.name             = {tool.name!r}")
        print(f"  tool.description      = {tool.description!r}")
        print(f"  tool.strict_json_schema = {tool.strict_json_schema}")
        print(f"  tool.defer_loading    = {tool.defer_loading}")
        # 内部属性（private）だが 3 値の反映確認のため参照する（本番コードでは通常触らない）。
        print(f"  _use_default_failure_error_function = {tool._use_default_failure_error_function}")
        print(f"  _failure_error_function is None     = {tool._failure_error_function is None}")
        print()


if __name__ == "__main__":
    main()
