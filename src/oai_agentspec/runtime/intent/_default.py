"""意図予測のデフォルト実装（LLM 非依存の骨格）。

DefaultContextBuilder: IntentQuery.history から直近 N 件を取得し history_items に
tuple 化して pass-through する。
DefaultIntentClassifier: 2 段（ContextBuilder + CandidateGenerator）を束ねる薄いオーケストレーター。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .protocols import CandidateGenerator, ContextBuilder
from .types import IntentContext, IntentPrediction, IntentQuery


class DefaultContextBuilder:
    """ContextBuilder Protocol のデフォルト実装。

    IntentQuery.history から直近 N 件を取得し history_items に tuple 化して
    pass-through する。history は Session 互換の duck-typed 呼び出し
    （``get_items(limit=...)``）で取得する。
    """

    def __init__(self, history_limit: int = 20) -> None:
        """コンストラクタ。

        Args:
            history_limit: history から取得する直近アイテムの上限件数（1 以上）。

        Raises:
            ValueError: history_limit が 1 未満の場合。負値を Session.get_items に渡すと
                SQLite の LIMIT -1 は「無制限」となり全履歴が LLM に流れるため、明示的に拒否する。
        """
        if history_limit < 1:
            raise ValueError(f"history_limit must be >= 1, got {history_limit}")
        self._history_limit = history_limit

    @property
    def history_limit(self) -> int:
        """history 取得件数の上限。"""
        return self._history_limit

    async def build(self, query: IntentQuery[Any]) -> IntentContext[Any]:
        """IntentQuery から IntentContext を組み立てる。

        Args:
            query: 入力クエリ。

        Returns:
            history_items を含んだ IntentContext。history が None の場合、
            history_items は空 tuple となる。
        """
        if query.history is None:
            items: tuple[Mapping[str, Any], ...] = ()
        else:
            raw = await query.history.get_items(limit=self._history_limit)
            items = tuple(raw or ())
        return IntentContext(
            utterance=query.utterance,
            history_items=items,
            run_context=query.run_context,
        )


class DefaultIntentClassifier:
    """IntentClassifier Protocol のデフォルト実装。

    ContextBuilder と CandidateGenerator の 2 段を束ねる薄いオーケストレーター。
    """

    def __init__(
        self,
        context_builder: ContextBuilder,
        generator: CandidateGenerator,
    ) -> None:
        """コンストラクタ。

        Args:
            context_builder: IntentContext を構築する ContextBuilder。
            generator: IntentContext から IntentPrediction を生成する CandidateGenerator。
        """
        self._context_builder = context_builder
        self._generator = generator

    @property
    def context_builder(self) -> ContextBuilder:
        """束ねている ContextBuilder。"""
        return self._context_builder

    @property
    def generator(self) -> CandidateGenerator:
        """束ねている CandidateGenerator。"""
        return self._generator

    async def classify(self, query: IntentQuery[Any]) -> IntentPrediction:
        """IntentQuery を分類する。

        Args:
            query: 入力クエリ。

        Returns:
            CandidateGenerator が返した IntentPrediction。
        """
        context = await self._context_builder.build(query)
        return await self._generator.generate(context)
