"""SDK `Session` Protocol を模した呼び出し記録付き fake（読み取り専用契約の測定用）。

契約の出所（SDK 実ソース `agents/memory/session.py` の `Session` Protocol）:
    - 属性 `session_id: str`
    - `async def get_items(limit: int | None = None) -> list[item]`
    - `async def add_items(items: list[item]) -> None`
    - `async def pop_item() -> item | None`
    - `async def clear_session() -> None`

本 fake は **全メソッドの呼び出しを `calls` へ記録する**。`dataset_from_session` の
「読み取り専用（`get_items` のみ呼ぶ）」契約は、書込系メソッドを例外にして塞ぐのではなく
`calls == ["get_items"]` のようにテスト側で assert して測る（例外で塞ぐと「呼ばれたが
握り潰された」経路と「呼ばれていない」経路を区別できず、実 Protocol より寛容/厳格の
どちらにも倒れるため。04-pytest §6）。

`get_items` の異常系（ストア破損等）は `get_items_error` に例外を注入して模倣する。
"""

from __future__ import annotations

from typing import Any

# `get_items` が引数なしで呼ばれたことを判別する sentinel（`limit=None` 明示と区別する）。
# 実 Protocol の既定値は None だが、None を既定にすると「引数なし」と「None 明示」が
# 記録上区別できないため、既定を sentinel にして呼び出し形の記録能力を持たせる
# （None / int を渡した場合の受理形・挙動は実 Protocol と同一）。
UNSET: Any = object()


class FakeSession:
    """呼び出しメソッド記録付きの fake Session。

    Attributes:
        session_id: セッション id（Protocol の必須属性）。
        calls: 呼び出されたメソッド名の発生順
            （`"get_items"` / `"add_items"` / `"pop_item"` / `"clear_session"`）。
        get_items_limits: `get_items` が受け取った limit 値の発生順
            （引数なしの呼び出しは `UNSET` を記録する）。
    """

    def __init__(
        self,
        items: list[dict[str, Any]] | None = None,
        *,
        session_id: str = "fake-session",
        get_items_error: BaseException | None = None,
    ) -> None:
        """fake Session を生成する。

        Args:
            items: `get_items` が返す履歴 items（plain dict の列）。None なら空履歴。
            session_id: セッション id。
            get_items_error: 非 None のとき `get_items` が記録後にこの例外を送出する。
        """
        self.session_id = session_id
        self._items: list[dict[str, Any]] = list(items or [])
        self._get_items_error = get_items_error
        self.calls: list[str] = []
        self.get_items_limits: list[Any] = []

    async def get_items(self, limit: int | None = UNSET) -> list[dict[str, Any]]:
        """履歴 items を返す（実 Protocol と同形の受理形・引数を記録する）。

        Args:
            limit: 取得件数の上限。未指定 / None なら全件、指定時は末尾 N 件（時系列順）。

        Returns:
            履歴 items のリスト。

        Raises:
            BaseException: `get_items_error` を注入した場合はその例外。
        """
        self.calls.append("get_items")
        self.get_items_limits.append(limit)
        if self._get_items_error is not None:
            raise self._get_items_error
        if limit is UNSET or limit is None:
            return list(self._items)
        return list(self._items[-limit:]) if limit > 0 else []

    async def add_items(self, items: list[dict[str, Any]]) -> None:
        """items を履歴へ追加する（呼び出しを記録する・例外にはしない）。

        Args:
            items: 追加する items。
        """
        self.calls.append("add_items")
        self._items.extend(items)

    async def pop_item(self) -> dict[str, Any] | None:
        """最新 item を取り除いて返す（呼び出しを記録する・例外にはしない）。

        Returns:
            最新の item。履歴が空なら None。
        """
        self.calls.append("pop_item")
        return self._items.pop() if self._items else None

    async def clear_session(self) -> None:
        """全 items を破棄する（呼び出しを記録する・例外にはしない）。"""
        self.calls.append("clear_session")
        self._items.clear()
