"""APO データセットの plain 型（`OptimizeCase`）+ データ分割ヘルパ（`train_val_split`）。

llmops `EvalCase` と同じ発想で、APO のデータセットも `OptimizeCase` で 1 ケースに**期待回答 / 期待
ツール / 期待ルート / 期待最終 agent / 期待承認ゲート**を集約した typed なケース型を提供する。
reward ファクトリ（`contains` / `tool_match` / `route_match` / `last_agent_match` /
`approval_match`）の既定 `field` は `OptimizeCase` の標準フィールド名に揃っているため、
`OptimizeCase` を使うときは**フィールド名を渡さずに**ファクトリを呼べる（`reward=contains()`）。
dict ケースも併存し、自由なフィールド名で `contains("自由名")` のように明示すれば従来どおり採点
できる。

`train_val_split` は sklearn 風の決定的データ分割で、SDK / `PromptStore` / 外部クライアントに触れ
ない純データ操作（NFR-5・依存方向に影響しない）。利用者自前分割（スライス / 層化 / 時系列等）の
結果も同じく `train` / `val` として `optimize` に渡せる（FR-9）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class OptimizeCase:
    """APO データセットの 1 ケース（llmops `EvalCase` 相当の typed 集約・FR-9）。

    `input` + 期待観点（出力 / ツール / 経路 / 最終 agent / 承認ゲート）を 1 つにまとめた typed な
    データセット型。reward ファクトリの既定フィールド名（`expected_output` / `expected_tools` /
    `expected_route` / `expected_last_agent` / `expected_approvals`）と一致するため、`contains()` /
    `tool_match()` / `route_match()` / `last_agent_match()` / `approval_match()` を**フィールド名
    無しで**そのまま呼べる（`reward=contains()` で十分）。dict ケースとも併存する: 既存の
    `[{"input": ..., "expected": ...}, ...]` のような自由 dict も `_case_value`（dict / 属性
    両対応）によりそのまま採点できる。reward callable には `case` フィールドとして渡る
    （`RolloutResult.case`）。

    `metadata` は採点に使わない補助情報（id 補足・タグ等）の保管場所で、reward は参照しない。

    Attributes:
        input: rollout への入力テキスト（必須）。
        id: ケース識別子（任意・ログ / 失敗解析用）。
        expected_output: 期待出力テキスト（`contains` / `exact` の既定フィールド）。
        expected_tools: 期待ツール名の列（`tool_match` の既定フィールド）。
        expected_route: 期待経路（起点込みの agent 名の列・`route_match` の既定フィールド）。
        expected_last_agent: 期待最終応答 agent 名（`last_agent_match` の既定フィールド）。
        expected_approvals: 期待承認ゲート名の列（`approval_match` の既定フィールド）。
        metadata: 採点に使わない任意の補助情報（reward は参照しない）。
    """

    input: str
    id: str | None = None
    expected_output: str | None = None
    expected_tools: list[str] = field(default_factory=list)
    expected_route: list[str] = field(default_factory=list)
    expected_last_agent: str | None = None
    expected_approvals: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def train_val_split[T](
    data: Sequence[T],
    *,
    val_ratio: float = 0.2,
    seed: int = 0,
    shuffle: bool = True,
) -> tuple[list[T], list[T]]:
    """データを `(train, val)` に決定的に分割する（opt-in・FR-9）。

    `shuffle=True` のとき `seed` 固定の `random.Random` で決定的にシャッフルしてから分割する
    （`shuffle=False` は入力順を保つ）。`val` 件数は `round(len(data) * val_ratio)` で、入力を
    一切改変せず新リストを返す（純データ操作）。

    Args:
        data: 分割対象のケース列（利用者供給・`OptimizeCase` / dict / 任意型）。
        val_ratio: val に回す割合（0.0..1.0）。
        seed: シャッフルの乱数シード（決定的）。
        shuffle: True で分割前にシャッフルする。False で入力順を保つ。

    Returns:
        `(train, val)` のタプル（いずれも新リスト・入力は不変）。

    Raises:
        ValueError: `val_ratio` が 0.0..1.0 の範囲外の場合。
    """
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError(f"val_ratio は 0.0..1.0 の範囲で指定してください: {val_ratio}")
    items = list(data)
    if shuffle:
        random.Random(seed).shuffle(items)
    n_val = round(len(items) * val_ratio)
    if n_val == 0:
        return items, []
    return items[n_val:], items[:n_val]
