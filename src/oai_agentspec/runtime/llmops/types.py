"""LLMOps 評価の plain 結果型・状態 Enum・実行トレース plain 型（外部 SDK 非依存）。

本モジュールは openai-agents（`agents`）・DeepEval（`deepeval`）・Langfuse（`langfuse`）を
一切 import しない。`_adapters` が SDK / 採点エンジン / 観測クライアントの型を本モジュールの
plain dataclass / Enum へ変換し、評価ロジック層（`evaluator` / `verdict`）と公開窓口は
この plain 型のみを扱う（NFR-1）。

すべて `@dataclass(frozen=True)`（会話の `SendResult` 等と一致・Pydantic 非導入）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..._validation import validate_bool


class CriterionStatus(StrEnum):
    """観点 1 件の判定状態。

    Attributes:
        PASS: 合格。
        FAIL: 不合格。
        INCONCLUSIVE: 判定不能（タイムアウト / スキーマ不適合等）。
        SKIP: 利用者が当該観点を評価対象に含めなかった。
        NOT_APPLICABLE: 必要 ground truth 非在で適用不能（reference_context / expected_route /
            expected_tools / expected_approvals のいずれかが当該ケースに無い）。対象の能力
            （ツール保有 / 横断モード）では NA にしない。
    """

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    SKIP = "skip"
    NOT_APPLICABLE = "not_applicable"


class Verdict(StrEnum):
    """データセット全体の統合 verdict（CI ゲート用の 1 合否）。

    Attributes:
        PASS: 合格。
        FAIL: 不合格。
    """

    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True)
class CriterionResult:
    """観点 1 件の判定結果（plain・採点エンジン型に非依存）。

    Attributes:
        criterion: 観点名（"factual_grounding" / "tool_correctness" /
            "handoff_correctness" 等）。
        status: 判定状態。
        rationale: 判定根拠（DeepEval reason / 経路・ツール比較結果）。
        score: 採点スコア（0.0..1.0 等）。決定的比較・スコア非在時は None。
    """

    criterion: str
    status: CriterionStatus
    rationale: str
    score: float | None = None


@dataclass(frozen=True)
class RouteStep:
    """処理経路の 1 ステップ（plain・SDK 型に非依存）。

    Attributes:
        agent: その時点で処理したエージェント名。
        handoff_from: handoff 由来なら遷移元エージェント名（`HandoffOutputItem.source_agent`
            の名前）。handoff を伴わない処理なら None。`expected_route` との決定的比較に必要な
            最小情報（遷移の有無と方向を判定可能）。
    """

    agent: str
    handoff_from: str | None = None


@dataclass(frozen=True)
class ObservedRoute:
    """観測された処理経路（plain・SDK 型に非依存）。

    Attributes:
        steps: `new_items` から抽出した処理経路（`HandoffOutputItem` の source / target を反映）。
            起点を先頭に・遷移先を順に・最終応答 agent を末尾に含むフルパス（routing 側で正規化）。
        last_agent: 最終的に応答したエージェント名（`RunResult.last_agent` の名前）。
    """

    steps: list[RouteStep] = field(default_factory=list)
    last_agent: str = ""


@dataclass(frozen=True)
class ObservedToolCall:
    """観測されたツール呼び出し 1 件（plain・最小情報）。

    args は持たせない（投機回避）。ツール名のみを保持する。

    Attributes:
        tool: 呼び出されたツール名。
    """

    tool: str


@dataclass(frozen=True)
class ObservedApproval:
    """観測された承認待ち（HITL / ツール承認ゲート）1 件（plain・最小情報）。

    Attributes:
        tool: 承認待ちになったツール名。
        call_id: 承認待ちのツール呼び出し ID（`RunOutcome.pending` の call_id）。
    """

    tool: str
    call_id: str = ""


@dataclass(frozen=True)
class ObservedRun:
    """1 実行で観測したルーティング経路 + ツール呼び出し + 承認待ち列（plain・SDK 型に非依存）。

    Attributes:
        route: ルーティング経路。
        tool_calls: 呼び出されたツール列（順序保持）。
        pending_approvals: 観測した承認待ち列（HITL / ツール承認ゲートの発火・順序保持）。
            中断を伴わない実行や承認ゲートを通らない実行では空。Langfuse metadata
            （`pending_approvals` / `interrupted`）の根拠に使う。
        interrupted: 採点時点で承認待ちが未解決のまま中断していたか（承認の自動解決で完了採点
            した場合や中断しなかった場合は False）。
    """

    route: ObservedRoute
    tool_calls: list[ObservedToolCall] = field(default_factory=list)
    pending_approvals: list[ObservedApproval] = field(default_factory=list)
    interrupted: bool = False

    def __post_init__(self) -> None:
        """`interrupted` が bool であることを構築時に検証する。

        Raises:
            ValueError: `interrupted` が bool でない場合。
        """
        validate_bool(self.interrupted, "interrupted")


@dataclass(frozen=True)
class CaseResult:
    """1 ケースの評価結果（plain）。

    Attributes:
        case_input: ケース入力（利用者提供の任意型）。
        output: 評価対象が生成した最終出力テキスト。
        criteria: 観点別結果（出力品質 + tool_correctness + handoff_correctness を含む）。
        observation: 捕捉した実行トレース（route + tool_calls）。捕捉不能なら None。
    """

    case_input: Any
    output: str
    criteria: list[CriterionResult] = field(default_factory=list)
    observation: ObservedRun | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """評価全体の構造化結果（plain・必ず返る・FR-5）。

    Attributes:
        target_id: 評価対象の識別子（spec.name / グラフ識別子・FR-1）。
        cases: 各ケースの結果。
        verdict: 統合 verdict（データセット全体での 1 合否・FR-5）。
    """

    target_id: str
    cases: list[CaseResult] = field(default_factory=list)
    verdict: Verdict = Verdict.FAIL
