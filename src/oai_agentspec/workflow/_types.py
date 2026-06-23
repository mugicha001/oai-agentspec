"""ワークフローの公開定数・型エイリアス・列挙と低レベル共有ヘルパ。

`START` / `END` 番兵、内部 sentinel（`_UNSET` / `_PENDING`）、フック / FUNCTION / router の型
エイリアス（`NodeHook` / `NodeFn` / `Router`）、ノード種別 `NodeKind`、ファサード入口種別
`FacadeMode`、および宣言／検証で共有する低レベルヘルパ（`_as_targets` / `_check_reserved_run_keys`）
を提供する。SDK には依存しない（NFR-1）。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Hashable
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._declarations import NodeResults

__all__ = [
    "END",
    "FacadeMode",
    "NodeFn",
    "NodeHook",
    "NodeKind",
    "Router",
    "START",
]


class _Sentinel:
    """START / END / 未指定を表す sentinel（一意な identity を持つ）。"""

    __slots__ = ("_label",)

    def __init__(self, label: str) -> None:
        self._label = label

    def __repr__(self) -> str:  # pragma: no cover - 表示用
        return self._label

    def __hash__(self) -> int:
        return id(self)


# START / END 番兵（入力の入口 / 終端）。エッジ端点に使える公開定数（FR-2）。
START: Any = _Sentinel("START")
END: Any = _Sentinel("END")

# 未指定を None と区別する内部 sentinel（デフォルト引数で None opt-in を識別する）。
_UNSET: Any = _Sentinel("<UNSET>")

# fan-in の未充足経路を表す内部 sentinel（最終出力候補から除外する）。
_PENDING: Any = _Sentinel("<PENDING>")

# ノード前後フックの型。`(node_name, NodeResults, context) -> None | Awaitable[None]`
# （FR-13）。制御フロー介入（中断指示等）は持たない（build-don't-run の線引き C-3）。
NodeHook = Callable[[str, "NodeResults", Any], "None | Awaitable[None]"]

# FUNCTION ノードの callable 型。`(msg, ctx) -> 出力`（sync / async 両対応）。
NodeFn = Callable[..., Any]

# 条件エッジ router の型。`(msg, ctx) -> 判定キー`（mapping のキーを選ぶ・FR-2）。
Router = Callable[[Any, Any], Hashable]


class NodeKind(str, Enum):  # noqa: UP042 - 規約で str, Enum 併用を許可（01-python 4）
    """ノード種別（初版は 2 種。SUBWORKFLOW は初版スコープ外・C-9）。"""

    AGENT = "agent"
    FUNCTION = "function"


class FacadeMode(str, Enum):  # noqa: UP042 - 規約で str, Enum 併用を許可（01-python 4）
    """ファサード（`as_facade_spec`）の入口モデル種別。

    入口に何を据えるかで「実 LLM 呼び出し回数 / 決定性 / 入出力を LLM が整形するか」が変わる。
    いずれの mode でも外側 context は tool 経由で内部ノードへ透過する（context を渡せない
    経路C との差別化点）。

    Attributes:
        DETERMINISTIC: 決定論ステートレスモデルを入口に据える。実 LLM 0 回・決定論。入力は
            素通し（LLM が整形しない）。tool 結果がそのまま最終出力（stop_on_first_tool）。
        LLM_INPUT: 実 LLM が tool 入力を整形して 1 回呼ぶ（出口の要約なし・stop_on_first_tool）。
            既定値で、従来の経路A と同一挙動（後方互換）。
        LLM_INPUT_OUTPUT: 実 LLM が入力整形に加え tool 結果も要約する（実 LLM 2 回）。
            stop_on_first_tool を付けず、SDK の `reset_tool_choice` 既定で 2 ターン目の無限
            ツール呼び出しを防ぐ。
    """

    DETERMINISTIC = "deterministic"
    LLM_INPUT = "llm_input"
    LLM_INPUT_OUTPUT = "llm_input_output"


def _as_targets(value: Any) -> list[Any]:
    """条件エッジの行き先（ノード名 | END | それらのリスト）をリストに正規化する。"""
    return list(value) if isinstance(value, list) else [value]


def _check_reserved_run_keys(
    options: dict[str, Any] | None, *, allow_session: bool, where: str
) -> None:
    """Runner kwargs の予約キー（input / context、ノード側は session も）を弾く。

    `input` はノードの msg、`context` は lib が経路A で透過するため lib 管理。ノード単位の
    `run_options` では `session` も禁止（session はグラフ既定 run_defaults でのみ設定し、並列
    ガードを成立させる）。
    """
    if not options:
        return
    reserved = {"input", "context"} if allow_session else {"input", "context", "session"}
    bad = reserved & options.keys()
    if bad:
        raise ValueError(
            f"{where} に予約キーが含まれます: {sorted(bad)}"
            "（input/context は lib 管理、session はグラフ既定 run_defaults で設定してください）"
        )
