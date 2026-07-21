"""Tool メタデータの中央集権的な宣言・遅延構築を担う Registry（コア層）。

`agents` には依存せず（SDK 型は関数内遅延 import で `_adapters` を経由）、
`ToolSpec` として宣言されたメタデータを保持し、属性アクセス（`registry.<name>`）
時にのみ `_adapters.build_function_tool` を呼んで SDK `FunctionTool` を組み立て、
`_built` キャッシュに保持する。単一スレッド / 単一イベントループ前提
（並行制御は利用者責任）。設計判断・要件の詳細は docs/architecture.md および
docs/adr/0001-tool-metadata-centralization.md を参照。
"""

from __future__ import annotations

import keyword
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# センチネル / 内部ヘルパ
# ---------------------------------------------------------------------------
#: `failure_error_function` の「未指定」を表す module-level センチネル。
#: SDK private センチネルを import せずに 3 値（未指定 / callable / None）を表現する。
_UNSET: Any = object()

#: `ToolRegistry` の公開メソッド名。属性アクセスと衝突する Tool 名の登録を拒否するために使う。
_RESERVED_METHOD_NAMES: frozenset[str] = frozenset({"register", "names", "metadata"})


@dataclass
class ToolSpec:
    """Tool のメタデータ宣言（mutable dataclass）。

    `ToolRegistry.register()` に渡して登録する。`enabled` は登録後に属性代入で
    トグルでき、次回 run から `is_enabled` closure 経由で SDK ネイティブに反映される
    （FR-4・再構築なし）。`name` / `func` の登録後変更は非サポート（`__setattr__`
    ガードは設けないため利用者責任・Registry キーとの desync を起こす）。

    Attributes:
        func: Tool 実体（sync/async callable・必須）。lib 非依存の純関数を想定。
        name: 登録キー。省略時は `func.__name__`。
        enabled: 有効フラグ。動的トグル可（is_enabled closure 経由で即時反映）。
        needs_approval: HITL 承認要否。SDK `needs_approval` へ委譲（bool | callable）。
        timeout: Tool 実行のタイムアウト秒。SDK 委譲。
        timeout_behavior: SDK `timeout_behavior` へ委譲。
        timeout_error_function: SDK `timeout_error_function` へ委譲。
        failure_error_function: SDK `failure_error_function` へ委譲。既定 `_UNSET`
            は「未指定 = kwarg を渡さない」を意味し、None を明示指定した場合は
            None が渡る（3 値・None-omission）。
        name_override: SDK `name_override` へ委譲（Registry キーとは独立）。
        description_override: SDK `description_override` へ委譲。
        strict_mode: SDK `strict_mode` へ委譲。None なら kwarg を渡さない（SDK 既定に委ねる）。
        extra: `agents.function_tool` に素通しする任意 kwarg。予約キー・未知キーは
            `_adapters.build_function_tool` 側で検証される。
    """

    func: Any
    name: str | None = None
    enabled: bool = True
    needs_approval: Any = None
    timeout: float | None = None
    timeout_behavior: str | None = None
    timeout_error_function: Any = None
    failure_error_function: Any = _UNSET
    name_override: str | None = None
    description_override: str | None = None
    strict_mode: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Tool メタデータの中央集権的な登録・遅延構築レジストリ（単一スレッド前提）。

    並行制御は利用者責任。`register()` で宣言を保持し、属性アクセス
    （`registry.<tool_name>`）で初回のみ `_adapters.build_function_tool` を呼び、
    以降は `_built` キャッシュから同一インスタンスを返す。`metadata(name)` は
    live な `ToolSpec` を返し、`enabled` への属性代入で is_enabled closure を
    通じて次 run から SDK ネイティブに反映される（FR-4・再構築なし）。
    """

    def __init__(self) -> None:
        """空の Registry を生成する。"""
        # `__getattr__` は `__getattribute__` が見つけられなかった時のみ呼ばれるため、
        # `_specs` / `_built` を `__init__` で確実に初期化して再帰を避ける。
        self._specs: dict[str, ToolSpec] = {}
        self._built: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 登録
    # ------------------------------------------------------------------
    def register(self, spec: ToolSpec) -> None:
        """`ToolSpec` を登録する（ビルドは属性アクセス時まで遅延）。

        Args:
            spec: 登録する `ToolSpec`。`name` 省略時は `spec.func.__name__` が使われる。

        Raises:
            ValueError: 名前が到達不能（`_` 始まり / 非識別子 / 公開メソッド名衝突）、
                または既登録名との重複。
        """
        name = spec.name if spec.name is not None else spec.func.__name__
        self._validate_name(name)
        if name in self._specs:
            raise ValueError(f"tool already registered: {name}")
        self._specs[name] = spec

    def names(self) -> list[str]:
        """登録済み Tool 名を昇順で返す。"""
        return sorted(self._specs.keys())

    def metadata(self, name: str) -> ToolSpec:
        """登録済み `ToolSpec` の live インスタンスを返す（属性代入で状態更新可能）。

        Args:
            name: 登録名。

        Returns:
            登録時に渡された `ToolSpec` そのもの（copy ではない）。

        Raises:
            KeyError: `name` が未登録の場合。文言は `__getattr__` と共通
                （`_unknown_tool_message` で単一ソース化）。
        """
        if name not in self._specs:
            raise KeyError(self._unknown_tool_message(name))
        return self._specs[name]

    # ------------------------------------------------------------------
    # 属性アクセス（遅延構築 + キャッシュ）
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """`registry.<tool_name>` で SDK `FunctionTool` を遅延構築して返す。

        `_` 始まりの名前は通常の `AttributeError` を送出する（内部プロトコル探索保護）。
        未登録名は登録済み名一覧付き `AttributeError`（`metadata()` KeyError と同一文言）。

        Args:
            name: アクセスされた属性名。

        Returns:
            構築済み `FunctionTool` インスタンス（同一名アクセスはキャッシュから返る）。

        Raises:
            AttributeError: `_` 始まり、または未登録名。
        """
        if name.startswith("_"):
            raise AttributeError(name)
        # `_specs` 未初期化時（例: unpickle 等）は `_unknown_tool_message` が
        # 内部で `self.names()` → `self._specs` を辿るため防御パスが破綻する。
        # 未初期化時は素の AttributeError で早期終了し、初期化済みのときのみ
        # 登録済み名一覧つきの分かりやすいメッセージを組み立てる。
        specs = self.__dict__.get("_specs")
        if specs is None:
            raise AttributeError(name)
        if name not in specs:
            raise AttributeError(self._unknown_tool_message(name))
        built = self.__dict__.setdefault("_built", {})
        if name in built:
            return built[name]
        # 関数内遅延 import で SDK 隔離を維持する（registry.py L403-408 同型）。
        from . import _adapters

        tool = _adapters.build_function_tool(
            specs[name],
            lambda n=name: self._specs[n].enabled,
        )
        built[name] = tool
        return tool

    # ------------------------------------------------------------------
    # 内部ヘルパ
    # ------------------------------------------------------------------
    def _validate_name(self, name: str) -> None:
        """登録名が属性アクセス可能な公開識別子であることを検証する。

        Args:
            name: 検証対象の名前。

        Raises:
            ValueError: `_` 始まり、`str.isidentifier()` 偽、Python 予約語
                （`class` / `from` 等・`isidentifier()` は True を返すが属性アクセスで
                SyntaxError になるため到達不能）、公開メソッド名衝突のいずれか。
        """
        if not name or not name.isidentifier():
            raise ValueError(f"invalid tool name (not a valid identifier): {name!r}")
        if name.startswith("_"):
            raise ValueError(f"invalid tool name (must not start with underscore): {name!r}")
        if keyword.iskeyword(name):
            raise ValueError(
                f"invalid tool name (Python keyword is not reachable via attribute access): "
                f"{name!r}"
            )
        if name in _RESERVED_METHOD_NAMES:
            raise ValueError(
                f"invalid tool name (collides with ToolRegistry public method): {name!r}"
            )

    def _unknown_tool_message(self, name: str) -> str:
        """未登録名エラーの単一ソース文言を組み立てる（`metadata` / `__getattr__` 共有）。

        Args:
            name: 未登録アクセス名。

        Returns:
            `unknown tool: <name>. registered tools: <一覧>` 形式の文字列。
            登録が空の場合は一覧を `(none)` と表示する。
        """
        registered = ", ".join(self.names()) or "(none)"
        return f"unknown tool: {name}. registered tools: {registered}"


__all__ = ["ToolRegistry", "ToolSpec"]
