"""評価観点オブジェクト（`Criterion`）と組込みファクトリ（DeepEval 非依存）。

観点は自己完結の frozen dataclass `Criterion`（名前・抽象メトリクス識別子・knockout・rubric・
必要データ・決定的フラグ）に集約する。利用者は組込みファクトリ
（`Relevance` / `Safety` / `Conciseness` / `Faithfulness` / `GEval` / `ToolUse` / `HandoffRoute`）で
typed に観点を宣言する。観点 → 採点メトリクスの対応は `Criterion.metric`（抽象識別子 `MetricId`）が
保持し DeepEval クラスを一切 import しない（依存規則）。識別子の DeepEval metric クラスへの解決は
`_adapters/judge.py` が担う。

観点名（`CriterionResult.criterion` に出る値）は内部文字列定数で固定する（結果キー継続）。
プロンプト本文（G-Eval rubric）は lib に同梱せず `Criterion.rubric` 経由で利用者が渡す。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from ..._validation import validate_bool

# 観点名（CriterionResult.criterion の値・内部定数。公開はファクトリ経由）。
_FACTUAL_GROUNDING: Final[str] = "factual_grounding"
_SAFETY: Final[str] = "safety"
_RELEVANCE: Final[str] = "relevance"
_CONCISENESS: Final[str] = "conciseness"
_TOOL_CORRECTNESS: Final[str] = "tool_correctness"
_HANDOFF_CORRECTNESS: Final[str] = "handoff_correctness"
_APPROVAL_GATE: Final[str] = "approval_gate"

# 必要データキー（Criterion.requires の要素・EvalCase の対応フィールド名と一致）。
_REQ_CONTEXT: Final[str] = "reference_context"
_REQ_EXPECTED_TOOLS: Final[str] = "expected_tools"
_REQ_EXPECTED_ROUTE: Final[str] = "expected_route"
_REQ_EXPECTED_APPROVALS: Final[str] = "expected_approvals"


class MetricId(StrEnum):
    """抽象メトリクス識別子（DeepEval クラス非依存・内部）。

    `_adapters/judge.py` が本識別子を DeepEval metric クラス（Faithfulness /
    AnswerRelevancy / GEval / ToolCorrectnessMetric 等）へ解決する。ドメイン（criteria）が
    DeepEval に結合しないための間接層。`deterministic=True` の観点（HandoffRoute）は metric を
    持たない（`Criterion.metric=None`）。

    Attributes:
        FAITHFULNESS: context ベースの事実整合性（DeepEval Faithfulness）。
        ANSWER_RELEVANCY: 出力の関連性（DeepEval Answer Relevancy）。
        G_EVAL: 観点文（rubric）ベースの汎用採点（DeepEval G-Eval）。
        TOOL_CORRECTNESS: ツール使用の正しさ（DeepEval ToolCorrectnessMetric・決定的）。
    """

    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    G_EVAL = "g_eval"
    TOOL_CORRECTNESS = "tool_correctness"


@dataclass(frozen=True)
class Criterion:
    """評価観点 1 件の自己完結宣言（plain frozen・agents/deepeval 非依存）。

    Attributes:
        name: 観点名（結果 `CriterionResult.criterion` に出る値）。
        metric: 抽象メトリクス識別子（judge が DeepEval metric へ解決）。`deterministic=True` の
            観点（HandoffRoute / ApprovalGate）は None で、evaluator が決定的比較経路へ回す。
        knockout: 当該観点が fail なら verdict 即 fail（fail-closed）。
        rubric: G-Eval 観点文（G-Eval メトリクス時のみ意味を持つ・lib 非同梱で利用者提供）。
        requires: 必要データキー集合（{"reference_context"} / {"expected_tools"} /
            {"expected_route"} / {"expected_approvals"}）。充足しない場合のみ evaluator が理由付き
            not_applicable を生成する（NA は ground truth 非在のみが根拠で、対象の能力=ツール保有 /
            横断モードでは判定しない。観点の適用可否は利用者の criteria 選択に委ねる）。
        deterministic: LLM 非使用の決定的比較か（handoff_correctness / approval_gate）。
    """

    name: str
    metric: MetricId | None = None
    knockout: bool = False
    rubric: str | None = None
    requires: frozenset[str] = field(default_factory=frozenset)
    deterministic: bool = False

    def __post_init__(self) -> None:
        """`knockout` / `deterministic` が bool であることを構築時に検証する。

        Raises:
            ValueError: `knockout` または `deterministic` が bool でない場合。
        """
        validate_bool(self.knockout, "knockout")
        validate_bool(self.deterministic, "deterministic")


def Relevance(*, knockout: bool = False) -> Criterion:  # noqa: N802 - 観点ファクトリ（型名様の API）
    """出力の関連性観点（DeepEval Answer Relevancy）。

    Args:
        knockout: True で当該観点 fail が verdict 即 fail（既定 False）。

    Returns:
        relevance の `Criterion`。
    """
    return Criterion(name=_RELEVANCE, metric=MetricId.ANSWER_RELEVANCY, knockout=knockout)


def Safety(*, rubric: str | None = None, knockout: bool = True) -> Criterion:  # noqa: N802
    """安全性観点（DeepEval G-Eval・既定 knockout）。

    Args:
        rubric: G-Eval 観点文（任意・利用者提供）。None なら judge 側の最小定義に倒す。
        knockout: True で当該観点 fail が verdict 即 fail（既定 True）。

    Returns:
        safety の `Criterion`。
    """
    return Criterion(name=_SAFETY, metric=MetricId.G_EVAL, knockout=knockout, rubric=rubric)


def Conciseness(*, rubric: str | None = None, knockout: bool = False) -> Criterion:  # noqa: N802
    """簡潔性観点（DeepEval G-Eval）。

    Args:
        rubric: G-Eval 観点文（任意・利用者提供）。None なら judge 側の最小定義に倒す。
        knockout: True で当該観点 fail が verdict 即 fail（既定 False）。

    Returns:
        conciseness の `Criterion`。
    """
    return Criterion(name=_CONCISENESS, metric=MetricId.G_EVAL, knockout=knockout, rubric=rubric)


def Faithfulness(*, knockout: bool = True) -> Criterion:  # noqa: N802
    """事実整合性観点（DeepEval Faithfulness・参照文脈必須・既定 knockout）。

    `requires={"reference_context"}` のため、`EvalCase.reference_context` が無いケースは
    evaluator が not_applicable とする。

    Args:
        knockout: True で当該観点 fail が verdict 即 fail（既定 True）。

    Returns:
        factual_grounding の `Criterion`。
    """
    return Criterion(
        name=_FACTUAL_GROUNDING,
        metric=MetricId.FAITHFULNESS,
        knockout=knockout,
        requires=frozenset({_REQ_CONTEXT}),
    )


def GEval(name: str, rubric: str, *, knockout: bool = False) -> Criterion:  # noqa: N802
    """利用者定義の G-Eval 観点（rubric 必須・カスタム）。

    Args:
        name: 観点名（組込み名と衝突させない想定）。
        rubric: G-Eval 観点文（必須・利用者提供）。
        knockout: True で当該観点 fail が verdict 即 fail（既定 False）。

    Returns:
        指定 name の `Criterion`。
    """
    return Criterion(name=name, metric=MetricId.G_EVAL, knockout=knockout, rubric=rubric)


def ToolUse(*, knockout: bool = False) -> Criterion:  # noqa: N802
    """ツール使用の正しさ観点（DeepEval ToolCorrectnessMetric・決定的・recall）。

    `requires={"expected_tools"}` のため `EvalCase.expected_tools` 非在のケースのみ evaluator が
    not_applicable とする。評価対象のツール保有有無では NA にしない（ツール非保有で期待ツールが
    呼ばれなければ recall=0 で fail）。比較は recall（期待ツールが全て呼ばれていれば pass・余分な
    呼び出しや handoff の `transfer_to_*` は無視）。

    Args:
        knockout: True で当該観点 fail が verdict 即 fail（既定 False）。

    Returns:
        tool_correctness の `Criterion`。
    """
    return Criterion(
        name=_TOOL_CORRECTNESS,
        metric=MetricId.TOOL_CORRECTNESS,
        knockout=knockout,
        requires=frozenset({_REQ_EXPECTED_TOOLS}),
    )


def HandoffRoute(*, knockout: bool = False) -> Criterion:  # noqa: N802
    """ルーティングの正しさ観点（決定的経路比較・LLM 非使用）。

    `requires={"expected_route"}` のため `EvalCase.expected_route` が無いケースのみ evaluator が
    not_applicable とする（横断モードかどうかでは NA にしない。単体対象に明示で入れた場合は観測
    経路 = 最終応答 agent の単一系列と比較する。適用可否は利用者の criteria 選択に委ねる）。

    Args:
        knockout: True で当該観点 fail が verdict 即 fail（既定 False）。

    Returns:
        handoff_correctness の `Criterion`（metric=None・deterministic）。
    """
    return Criterion(
        name=_HANDOFF_CORRECTNESS,
        metric=None,
        knockout=knockout,
        requires=frozenset({_REQ_EXPECTED_ROUTE}),
        deterministic=True,
    )


def ApprovalGate(*, knockout: bool = False) -> Criterion:  # noqa: N802
    """承認ゲートの発火を検証する観点（決定的・実行ゼロ・LLM 非使用）。

    評価対象を実行し HITL / ツール承認で中断した時点の承認待ち（`RunOutcome.pending` のツール名
    集合）と `expected_approvals` を決定的に比較する（recall: expected が全て pending に出ていれば
    pass）。resume も approve もしないため危険ツールは実行されない（ゲートが正しく発火するかだけ
    を採点する）。`requires={"expected_approvals"}` のため `EvalCase.expected_approvals` 非在の
    ケースのみ evaluator が not_applicable とする。

    中断時でも採点する点が他観点と異なる（中断短絡の前に処理する）。同一ケースで ApprovalGate
    以外の観点を併記した場合、それらは中断時 INCONCLUSIVE に倒れる（混在を許容）。

    Args:
        knockout: True で当該観点 fail が verdict 即 fail（既定 False）。

    Returns:
        approval_gate の `Criterion`（metric=None・deterministic）。
    """
    return Criterion(
        name=_APPROVAL_GATE,
        metric=None,
        knockout=knockout,
        requires=frozenset({_REQ_EXPECTED_APPROVALS}),
        deterministic=True,
    )


# 標準品質セット（evaluate の criteria 既定）。
def default_criteria() -> tuple[Criterion, ...]:
    """`evaluate(criteria=None)` の既定観点セットを返す（標準品質 4 観点）。

    Returns:
        `(Relevance(), Safety(), Conciseness(), Faithfulness())`。
    """
    return (Relevance(), Safety(), Conciseness(), Faithfulness())
