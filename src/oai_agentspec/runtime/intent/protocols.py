"""意図予測の Protocol（DI 拡張点・agents 非依存）。

すべて @runtime_checkable + async 統一。IntentClassifier が分類器全体の差し替え口、
ContextBuilder + CandidateGenerator が DefaultIntentClassifier の内部段。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .types import IntentContext, IntentPrediction, IntentQuery


@runtime_checkable
class IntentClassifier(Protocol):
    """分類器全体の Protocol（利用側が丸ごと差し替え可）。"""

    async def classify(self, query: IntentQuery[Any]) -> IntentPrediction:
        """利用者の発話から意図予測を行う。

        Args:
            query: 分類対象の発話・履歴・run_context を含む入力。

        Returns:
            分類結果の予測。
        """
        ...


@runtime_checkable
class ContextBuilder(Protocol):
    """IntentQuery を IntentContext へ整形する前処理段。"""

    async def build(self, query: IntentQuery[Any]) -> IntentContext[Any]:
        """IntentQuery を整形済みの IntentContext へ変換する。

        Args:
            query: 分類対象の発話・履歴・run_context を含む入力。

        Returns:
            整形済みの内部コンテキスト。
        """
        ...


@runtime_checkable
class CandidateGenerator(Protocol):
    """整形済み IntentContext から IntentPrediction を生成する段。

    IntentPolicy の強制（allowlist / ConfidenceLevel 降順 sort / max_candidates
    truncate）は generator 実装の責務。既定の `LLMCandidateGenerator` はこれを
    post-hoc 3 段として実装している。独自 generator を差し込む場合、policy を
    守らせたいなら実装側で同等の適用を行うこと（`DefaultIntentClassifier` は
    generator の出力を素通しし、policy を強制しない）。
    """

    async def generate(self, context: IntentContext[Any]) -> IntentPrediction:
        """整形済みコンテキストから意図予測を生成する。

        Args:
            context: ContextBuilder が組み立てた整形済みコンテキスト。

        Returns:
            分類結果の予測。candidates は ConfidenceLevel 降順を推奨（型上は強制
            されない・既定実装のみが保証する）。
        """
        ...
