"""宣言的ガードレール登録の plain 型層（agents 非依存・検証なし・保持に徹する）。

適用境界の列挙 `Boundary`（agent 境界 2 種 + ツール境界 2 種）・深刻度の全順序列挙
`Severity`（LOW / MEDIUM / HIGH / CRITICAL）・宣言 1 件を表す `GuardrailSpec`（名前 /
境界 / guardrail 本体 / labels / severity）を提供する。`GuardrailSpec` は与えられた値を
解釈せずそのまま保持し、値域検証と Enum への正規化は登録簿側の責務とする。

本層は agents パッケージを一切 import せず、SDK なしで単体検証できる（NFR-1）。`guardrail`
フィールドは SDK 互換オブジェクトを不透明型（`Any`）として保持するのみで、その構造を見ない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class Boundary(str, Enum):  # noqa: UP042 - 規約で str, Enum 併用を許可（01-python 4）
    """ガードレールの適用境界（agent 境界 2 種 / ツール境界 2 種）。

    `str` 併用のため素の文字列との等価比較・dict キー互換が成立し、値域内の文字列からは
    `Boundary("tool_output")` のように復元できる（値域外は Enum 標準どおり `ValueError`）。

    Attributes:
        INPUT: agent への入力に適用する境界。
        OUTPUT: agent の出力に適用する境界。
        TOOL_INPUT: ツール呼び出しの入力に適用する境界。
        TOOL_OUTPUT: ツール呼び出しの出力に適用する境界。
    """

    INPUT = "input"
    OUTPUT = "output"
    TOOL_INPUT = "tool_input"
    TOOL_OUTPUT = "tool_output"


class Severity(IntEnum):
    """ガードレール発火の深刻度（1 始まりの 4 段階・比較演算子で全順序）。

    `IntEnum` のため `Severity.LOW < Severity.HIGH` のように比較でき、1 始まりのため
    どのメンバも真値になる（0 始まりによる真偽値の取り違えを避ける）。

    Attributes:
        LOW: 情報レベル（記録のみで運用判断に影響しない程度）。
        MEDIUM: 注意レベル（監視・レビュー対象）。
        HIGH: 重大レベル（是正が必要）。
        CRITICAL: 最重大レベル（即時対応が必要）。
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class GuardrailSpec:
    """ガードレール宣言 1 件の plain 表現（値を解釈せず保持する・フィールドは再代入不可）。

    値域外の `boundary` / `severity` を渡しても構築は成功し、値域内の文字列を渡しても
    Enum へ変換せず渡された型のまま保持する（検証と正規化は登録簿の責務）。

    `frozen=True` なのは、登録時の検証（宣言境界と実体境界の一致・可視名の一致）を通った宣言が
    後から書き換えられて検証済みの不変条件を失う経路を閉じるためである。境界を書き換えられると、
    出力境界の宣言が入力側へ結線されて対象を一度も検査しないまま一覧上は「登録済み」に見える。
    ただし frozen が禁止するのは**属性の再代入**のみで、`labels` は `dict` なのでキー単位の
    更新（`spec.labels["k"] = v`）は通る（宣言後のラベル追記を許す意図的な設計）。

    Attributes:
        name: 宣言の識別名（登録簿内での参照キーに使う）。
        boundary: 適用境界（`Boundary` または同等の文字列）。
        guardrail: guardrail 本体。不透明型として保持し、構造を解釈しない。
        labels: 任意のラベル（フィルタ / 分類用）。既定は空 dict で、値は正規化しない。
        severity: 深刻度（任意）。既定は None で、既定値からの推定は行わない。
    """

    name: str
    boundary: Boundary | str
    guardrail: Any
    labels: dict[str, Any] = field(default_factory=dict)
    severity: Severity | None = None
