"""ToolRegistry の登録/照会エラーを教育的にまとめて実演する。

以下の 5 種のエラーを try/except で捕まえて挙動を示す:

1. 二重登録 (ValueError: "tool already registered: ...")
2. 予約キー衝突 (属性アクセス時 build で ValueError)
3. 未知キー (属性アクセス時 build で ValueError)
4. Python 予約語 (register 時 ValueError)
5. 未登録アクセス (KeyError = metadata / AttributeError = 属性アクセス)

Azure OpenAI 不要（純ロジック検証のみ）。

Usage:
    uv run python examples/tool_registry/04_error_handling.py
"""

from __future__ import annotations

from oai_agentspec import ToolRegistry, ToolSpec


async def hello(name: str) -> str:
    """挨拶。"""
    return f"hello, {name}"


def _demonstrate(label: str, action) -> None:  # type: ignore[no-untyped-def]
    """try/except で action を実行し、発生した例外の型とメッセージを表示する。"""
    print(f"## {label}")
    try:
        action()
    except (ValueError, KeyError, AttributeError) as err:
        print(f"  -> {type(err).__name__}: {err}\n")
    else:
        print("  -> (no exception raised)\n")


def main() -> None:
    reg = ToolRegistry()
    reg.register(ToolSpec(func=hello, name="greet"))

    # 1) 二重登録は register 時に検出される
    _demonstrate(
        "1) 二重登録",
        lambda: reg.register(ToolSpec(func=hello, name="greet")),
    )

    # 2) 予約キー衝突: extra に型付きフィールドと同名のキーを入れて登録。
    #    register() は成功し、属性アクセス時の build_function_tool で検出される。
    reg.register(ToolSpec(func=hello, name="bad_reserved", extra={"is_enabled": True}))
    _demonstrate(
        "2) 予約キー衝突 (extra={'is_enabled': True} → 属性アクセスで build 発火)",
        lambda: reg.bad_reserved,
    )

    # 3) 未知キー: SDK function_tool が受け付けないキーを extra で渡す。
    reg.register(ToolSpec(func=hello, name="bad_unknown", extra={"unknown_kw": 1}))
    _demonstrate(
        "3) 未知キー (extra={'unknown_kw': 1} → 属性アクセスで build 発火)",
        lambda: reg.bad_unknown,
    )

    # 4) Python 予約語（Codex review [P2] 修正で追加されたガード）
    _demonstrate(
        "4) Python 予約語 (name='class')",
        lambda: reg.register(ToolSpec(func=hello, name="class")),
    )

    # 5a) 未登録名の metadata() は KeyError
    _demonstrate(
        "5a) 未登録名の metadata() アクセス",
        lambda: reg.metadata("unknown_tool"),
    )

    # 5b) 未登録名の属性アクセスは AttributeError（文言は metadata と共有）
    _demonstrate(
        "5b) 未登録名の属性アクセス",
        lambda: reg.unknown_tool,
    )

    print(f"[final registered] {reg.names()}")


if __name__ == "__main__":
    main()
