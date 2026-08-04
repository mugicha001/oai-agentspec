"""エージェント名の宣言的定数簿（`AgentNames` / `validate_agent_names`・agents 非依存）。

エージェント名を 1 箇所のクラス属性宣言へ集約し、以降の参照を定数経由にするためのコア
公開 API。`AgentSpec` / `AgentRegistry` / `HandoffGraph` / `NextTurnPolicy` からは完全に
独立で、生 str による宣言と `registry.validate()` / `get()` の実行時検出は不変である
（定数簿は任意の追加手段）。

タイポは次の 3 点で捕捉する。

1. 宣言済み名は登録操作なしに静的なクラス属性として解決でき、`dir()` に載る
2. 未宣言名アクセスは宣言済み**属性名**の一覧つき `AttributeError` になる
3. 到達不能な宣言・不正な値・注釈のみの宣言・値の重複はクラス定義時に `ValueError` になる

検査対象はクラス body の名前空間のみである。定義後の属性代入（`Names.EXTRA = "x"`）と
定数簿以外の基底クラス（mixin）由来の公開 str 属性は検査を通らずに宣言集合へ載る。

到達不能名規則（非空 + `str.isidentifier()` / `_` 始まり禁止 / `keyword.iskeyword()` 禁止 /
予約属性名との衝突禁止）の SoT は `ToolRegistry._validate_name`（`tool_registry.py`）であり、
本モジュールは同一規則を予約属性名だけ差し替えて適用する。例外メッセージの言語は踏襲元の
同一トピックへ揃え、属性アクセス・識別子規則に属する `ValueError` / `AttributeError` は英語
（`ToolRegistry` 踏襲）、整合検査の集約 `KeyError` は日本語（`AgentRegistry.validate()` 踏襲）
とする。実行時 import は標準ライブラリのみ（`agents` / `openai` 非依存のコア最下層リーフ）。

詳細は docs/architecture.md の「エージェント名定数簿」節および
docs/adr/0018-declarative-agent-name-catalog.md を参照。
"""

from __future__ import annotations

import inspect
import keyword
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .registry import AgentRegistry

#: 定数簿の宣言名として使えない属性名。`names` は唯一の公開メソッド、`mro` は非データ記述子
#: `type.mro` を隠すと `cls.mro()` による introspection が壊れるため予約する
#: （`ToolRegistry._RESERVED_METHOD_NAMES` と同じ表現）。
_RESERVED_ATTRIBUTE_NAMES: frozenset[str] = frozenset({"names", "mro"})


def _is_dunder(name: str) -> bool:
    """`__x__` 形（dunder）の名前かどうかを返す。

    Args:
        name: 判定対象の名前。

    Returns:
        dunder なら True。
    """
    return name.startswith("__") and name.endswith("__")


def _validate_attribute_name(name: str) -> None:
    """宣言の属性名が属性アクセス可能な公開識別子であることを検証する。

    規則の SoT は `ToolRegistry._validate_name`（`tool_registry.py`）で、本関数は同一の
    4 分岐を予約集合だけ `_RESERVED_ATTRIBUTE_NAMES` へ差し替えて適用する。

    Args:
        name: 検証対象の属性名。

    Raises:
        ValueError: `str.isidentifier()` 偽（空文字を含む）、`_` 始まり、Python 予約語
            （`class` / `from` 等・`isidentifier()` は True だが属性アクセスで SyntaxError に
            なるため到達不能）、予約属性名との衝突のいずれか。
    """
    if not name or not name.isidentifier():
        raise ValueError(f"invalid agent name declaration (not a valid identifier): {name!r}")
    if name.startswith("_"):
        raise ValueError(
            f"invalid agent name declaration (must not start with underscore): {name!r}"
        )
    if keyword.iskeyword(name):
        raise ValueError(
            f"invalid agent name declaration (Python keyword is not reachable via attribute "
            f"access): {name!r}"
        )
    if name in _RESERVED_ATTRIBUTE_NAMES:
        raise ValueError(
            f"invalid agent name declaration (collides with reserved attribute): {name!r}"
        )


def _validate_namespace(namespace: dict[str, Any]) -> None:
    """クラス body の名前空間が宣言専用であることを検証する。

    dunder（`__module__` / `__qualname__` / `__doc__` / `__annotations__` 等）は検査対象・
    宣言集合の双方から除外する（除外しないと docstring 付きクラスが定義できない）。

    Args:
        namespace: クラス定義時の名前空間（クラス body 相当）。

    Raises:
        ValueError: 到達不能な属性名、非空 str 以外の値（callable / descriptor / 型
            オブジェクトを含む）のいずれか。注釈のみの宣言はクラス生成後に
            `_validate_no_annotation_only` が検査する。
    """
    for attr, value in namespace.items():
        if _is_dunder(attr):
            continue
        _validate_attribute_name(attr)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"invalid agent name declaration (value must be a non-empty str): {attr!r}"
            )


def _validate_no_annotation_only(cls: type) -> None:
    """値の代入を伴わない注釈のみの宣言（`PLANNER: str`）が無いことを検証する。

    注釈の取得には `inspect.get_annotations(cls)` を使う。クラス body の名前空間から
    `__annotations__` を直接読む形は CPython の実装詳細に依存し、PEP 649/749 適用後
    （Python 3.12 / 3.14 の両実機で確認）は名前空間へ `__annotations__` が入らないため
    検査が無言で失効する。`inspect.get_annotations` はそのクラス自身の注釈のみを返し
    継承分を含まないため、MRO 集約とは独立に扱える（クラス生成後に呼ぶ必要がある）。

    Args:
        cls: 生成済みの定数簿クラス。

    Raises:
        ValueError: 注釈だけがあり同名の値が宣言されていない属性がある場合。
    """
    for attr in inspect.get_annotations(cls):
        if _is_dunder(attr) or attr in vars(cls):
            continue
        raise ValueError(
            f"invalid agent name declaration (annotation without a value is not declared): {attr!r}"
        )


def _declared_names(cls: type) -> dict[str, str]:
    """MRO 集約後の宣言（属性名 -> エージェント名）を返す。

    `dir()` が親クラスの属性を含むため、集約しないと `names()` と `dir()` が食い違う。
    同一属性名をサブクラスが override した場合はサブクラス側の値が残る。

    Args:
        cls: 定数簿クラス（`AgentNames` またはそのサブクラス）。

    Returns:
        属性名からエージェント名への dict。
    """
    declared: dict[str, str] = {}
    for klass in reversed(cls.__mro__):
        for attr, value in vars(klass).items():
            if attr.startswith("_") or not isinstance(value, str):
                continue
            declared[attr] = value
    return declared


def _validate_no_duplicate_values(declared: dict[str, str]) -> None:
    """同一のエージェント名が複数の属性名へ割り当てられていないことを検証する。

    `PLANNER = "planner"` / `PLANER = "planner"` は防ぎたいタイポの一類型でありながら、
    拒否しなければ属性アクセスも整合検査も通ってしまい、どの検出網にも掛からない。
    同一属性名の override は 1 名前として扱うため対象外（集約後の dict で判定する）。

    Args:
        declared: MRO 集約後の宣言（属性名 -> エージェント名）。

    Raises:
        ValueError: 同一値が 2 つ以上の異なる属性名へ割り当てられている場合（全件を列挙）。
    """
    by_value: dict[str, list[str]] = {}
    for attr, value in declared.items():
        by_value.setdefault(value, []).append(attr)
    duplicates = [
        f"{value!r} is declared as {', '.join(sorted(attrs))}"
        for value, attrs in sorted(by_value.items())
        if len(attrs) > 1
    ]
    if duplicates:
        raise ValueError("duplicate agent name value: " + "; ".join(duplicates))


def _unknown_agent_name_message(cls: type, name: str) -> str:
    """未宣言名アクセスのエラー文言を組み立てる（`ToolRegistry._unknown_tool_message` 同型）。

    Args:
        cls: アクセスされた定数簿クラス。
        name: 未宣言のアクセス名。

    Returns:
        `unknown agent name: <name>. declared agent names: <一覧>` 形式の文字列。
        宣言が 0 件の場合は一覧を `(none)` と表示する。
    """
    declared = ", ".join(sorted(_declared_names(cls))) or "(none)"
    return f"unknown agent name: {name}. declared agent names: {declared}"


class _AgentNamesMeta(type):
    """`AgentNames` の宣言をクラス定義時に検査する非公開メタクラス（公開契約ではない）。

    `__new__` が宣言専用の性質（到達可能な属性名・非空 str の値・注釈のみの宣言の禁止・
    値の重複禁止）を定義時に保証し、`__getattr__` が未宣言名の検出専用の入口として
    一覧つき `AttributeError` を送出する。属性名と値の検査は namespace を走査して
    `type.__new__` の前に、注釈と値の重複の検査は生成済みクラスに対してその後に行う。
    """

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> _AgentNamesMeta:
        """定数簿クラスを生成し、宣言の妥当性をクラス定義時に検査する。

        基底 `AgentNames` 自身の生成（`bases` に本メタクラスのインスタンスを含まない
        呼び出し）では検査を skip する（基底の `names` classmethod が「値が非 str」で
        落ちるのを防ぐため）。

        Args:
            name: 生成するクラス名。
            bases: 基底クラスの列。
            namespace: クラス body 相当の名前空間。
            **kwargs: `type.__new__` へ素通しするクラスキーワード引数。

        Returns:
            生成した定数簿クラス。

        Raises:
            ValueError: 宣言が到達不能・値が非空 str 以外・注釈のみ・値が重複のいずれか。
        """
        is_catalog = any(isinstance(base, _AgentNamesMeta) for base in bases)
        if is_catalog:
            _validate_namespace(namespace)
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if is_catalog:
            _validate_no_annotation_only(cls)
            _validate_no_duplicate_values(_declared_names(cls))
        return cls

    def __getattr__(cls, name: str) -> Any:
        """未宣言名アクセスを一覧つき `AttributeError` として報告する（検出専用の入口）。

        宣言済み名は通常のクラス属性として解決されるため本メソッドを通らない。`_` 始まり
        （dunder を含む）は素の `AttributeError` で素通しし、`inspect` / `copy` / `pickle` /
        pytest の内部プロトコル探索へ誤誘導メッセージを返さない
        （`ToolRegistry.__getattr__` の `_` 始まり素通しと同型）。

        Args:
            name: アクセスされた属性名。

        Raises:
            AttributeError: 常に送出する（`_` 始まりは素の文言、それ以外は宣言済み属性名の
                一覧つき文言）。
        """
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(_unknown_agent_name_message(cls, name))


class AgentNames(metaclass=_AgentNamesMeta):
    """エージェント名をクラス属性として宣言するための基底クラス（継承して使う）。

    宣言はクラス body の代入 1 行のみで、実行時の登録操作を持たない。値は `str` のため、
    既存の名前参照フィールド（`AgentSpec.name` / `handoffs` / `handoff_options` のキー /
    `sub_agents` / `sub_agent_tools` のキー / `DynamicHandoff.candidates` /
    `NextTurnRule` の到達元・遷移先 / entry 名）へ変換なしに渡せる。

    多段継承を許可し、宣言集合は MRO 集約する。予約属性名は `{"names", "mro"}` で、
    公開メソッドは `names()` の 1 件のみ。

    Example:
        ```python
        class Names(AgentNames):
            \"\"\"本アプリのエージェント名（宣言はここ 1 箇所）。\"\"\"

            PLANNER = "planner"
            WRITER = "writer"


        registry.register(AgentSpec(Names.PLANNER, "計画を立てる", handoffs=[Names.WRITER]))
        ```
    """

    @classmethod
    def names(cls) -> list[str]:
        """宣言済みエージェント名（値）の昇順リストを返す。

        MRO 集約後の宣言値を返すため、親クラスの宣言も含まれる。`AttributeError` が列挙
        するのは属性名で、本メソッドが返すのは値である点が非対称であることに注意する。

        Returns:
            宣言済みエージェント名の昇順リスト（宣言が 0 件なら空リスト）。
        """
        return sorted(_declared_names(cls).values())


def validate_agent_names(names: type[AgentNames], registry: AgentRegistry) -> None:
    """定数簿の宣言集合と registry の登録名集合の整合を検査する。

    registry 側の既知名は既存公開 API の `registry.names()`（spec と factory の和集合）
    のみを読み、private 状態には触れない（`register_factory` 登録名を未登録と誤検知しない）。
    `AgentRegistry` から定数簿への依存辺を作らないため、型注釈は `TYPE_CHECKING` 配下でのみ
    import する（`next_turn.py` と同型）。

    報告は「定数簿に宣言済みだが registry へ未登録」「registry へ登録済みだが定数簿に未宣言」
    の 2 群へ全件を集約し、群ごとに接頭辞を付けて `"; "` で連結、群が 2 つ揃うときのみ
    `" | "` で連結して単一の `KeyError` を送出する（`AgentRegistry.validate()` と同型）。

    Args:
        names: 検査対象の定数簿クラス（`AgentNames` のサブクラス）。
        registry: 突き合わせる `AgentRegistry`。

    Raises:
        KeyError: 両方向のいずれかに差分が 1 件以上ある場合（全件を列挙）。
    """
    declared = set(names.names())
    known = set(registry.names())
    messages: list[str] = []
    unregistered = sorted(declared - known)
    if unregistered:
        messages.append(
            "定数簿に宣言済みだが registry へ未登録: "
            + "; ".join(repr(name) for name in unregistered)
        )
    undeclared = sorted(known - declared)
    if undeclared:
        messages.append(
            "registry へ登録済みだが定数簿に未宣言: " + "; ".join(repr(name) for name in undeclared)
        )
    if messages:
        raise KeyError(" | ".join(messages))


__all__ = ["AgentNames", "validate_agent_names"]
