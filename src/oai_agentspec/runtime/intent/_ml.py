"""L2 supplementary: ML 意図分類器のための推論側支援関数。

sklearn 互換等の外部 ML モデルが返す (ラベル, スコア) 列を、lib の契約型
`IntentPrediction` へ橋渡しするための低レベル部品を提供する。推論 callable
（FR-3 `MLCandidateGenerator`・T2 で別途追記）の内部で利用する想定。

提供シンボル:
- `confidence_mapper_from_thresholds` (FR-1): 5 段階の下限スコア境界から float
  スコアを `ConfidenceLevel` へ変換する呼び出し可能オブジェクトを構築する。
  境界包含 (`>=`)・単調性検証・許容範囲外の error/clamp 挙動を持つ。
- `prediction_from_scored_labels` (FR-2): (ラベル, スコア) 列を post-hoc 適用
  （重複集約 / allowlist フィルタ / mapper 適用 / レベル降順 sort / truncate）の
  うえ `IntentPrediction` へ変換する。`_llm.py` の post-hoc 3 段と同じトーンを
  踏襲した独立実装。
- `MLCandidateGenerator` (FR-3): 利用側の推論 callable（同期 / 非同期）を
  `CandidateGenerator` Protocol へ橋渡しする薄いアダプタ。返った (ラベル, スコア)
  列を `prediction_from_scored_labels` に委譲して `IntentPrediction` を組み立てる。

方針:
- build-don't-run 該当なし（純関数・SDK 非依存・LLM/実 API を呼ばない）。
- ログのフォーマット文字列は `%s` プレースホルダを使う（f-string 不可）。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

from ._llm import _LEVEL_ORDER
from .types import (
    ConfidenceLevel,
    IntentCandidate,
    IntentContext,
    IntentPolicy,
    IntentPrediction,
)

logger = logging.getLogger(__name__)

# 許容スコア範囲（確率・正規化済みスコアを前提とする閉区間）。
_SCORE_MIN: float = 0.0
_SCORE_MAX: float = 1.0


def confidence_mapper_from_thresholds(
    *,
    certain: float,
    high: float,
    medium: float,
    low: float,
    speculative: float,
    on_out_of_range: Literal["error", "clamp"] = "error",
) -> Callable[[float], ConfidenceLevel]:
    """5 段階の下限スコア境界から `ConfidenceLevel` 変換 callable を構築する。

    返される callable は、スコアを上位側から順に下限境界と比較し（`>=`）、最初に
    満たした段の `ConfidenceLevel` を返す。最下位境界 `speculative` 未満のスコアも
    `SPECULATIVE` に丸める。下限に等しいスコアは上位側の段に含める（境界包含）。

    Args:
        certain: `CERTAIN` と判定する下限スコア。
        high: `HIGH` と判定する下限スコア。
        medium: `MEDIUM` と判定する下限スコア。
        low: `LOW` と判定する下限スコア。
        speculative: `SPECULATIVE` と判定する下限スコア。
        on_out_of_range: 許容範囲 (0.0〜1.0) 外スコアの扱い。`"error"`（既定）は
            マッパ呼び出し時に `ValueError` を送出する（fail-fast）。`"clamp"` は
            0.0 未満を 0.0、1.0 超を 1.0 に丸めてから閾値判定する。

    Returns:
        float スコアを 1 つ受け取り `ConfidenceLevel` を返す callable。

    Raises:
        ValueError: 閾値境界が単調非増加
            (`certain >= high >= medium >= low >= speculative`) でない場合。
    """
    thresholds = (certain, high, medium, low, speculative)
    if any(upper < lower for upper, lower in zip(thresholds[:-1], thresholds[1:], strict=True)):
        raise ValueError(
            "confidence thresholds must be monotonically non-increasing "
            "(certain >= high >= medium >= low >= speculative), "
            f"got {thresholds}"
        )

    # 上位段から (下限, レベル) を並べ、最初に満たした段を採用する。
    _bands: tuple[tuple[float, ConfidenceLevel], ...] = (
        (certain, ConfidenceLevel.CERTAIN),
        (high, ConfidenceLevel.HIGH),
        (medium, ConfidenceLevel.MEDIUM),
        (low, ConfidenceLevel.LOW),
        (speculative, ConfidenceLevel.SPECULATIVE),
    )

    def _mapper(score: float) -> ConfidenceLevel:
        """float スコアを `ConfidenceLevel` に変換する。

        Args:
            score: 変換対象スコア。

        Returns:
            対応する `ConfidenceLevel`。

        Raises:
            ValueError: `on_out_of_range="error"` かつスコアが 0.0〜1.0 外の場合。
                スコアが NaN の場合は `on_out_of_range` の値に関わらず送出する
                （NaN は clamp で救えない不正値のため）。
        """
        if math.isnan(score):
            raise ValueError(f"score must be within [{_SCORE_MIN}, {_SCORE_MAX}], got {score}")
        if score < _SCORE_MIN or score > _SCORE_MAX:
            if on_out_of_range == "error":
                raise ValueError(f"score must be within [{_SCORE_MIN}, {_SCORE_MAX}], got {score}")
            score = min(max(score, _SCORE_MIN), _SCORE_MAX)
        for lower, level in _bands:
            if score >= lower:
                return level
        return ConfidenceLevel.SPECULATIVE

    return _mapper


def prediction_from_scored_labels(
    scored_labels: Sequence[tuple[str, float]],
    *,
    policy: IntentPolicy,
    mapper: Callable[[float], ConfidenceLevel],
) -> IntentPrediction:
    """(ラベル, スコア) 列を post-hoc 適用のうえ `IntentPrediction` に変換する。

    `_llm.py` の post-hoc 段を踏襲した独立実装。処理順は
    (1) 重複集約（同一ラベルは最高スコアの 1 件のみ）、(2) allowlist フィルタ
    （`policy.categories` の name 集合外を除外・除外時 WARNING ログ）、
    (3) mapper でスコア→`ConfidenceLevel` 変換、(4) レベル降順 stable sort、
    (5) `policy.max_candidates` で truncate。空入力・全除外は空 `candidates` の
    `IntentPrediction` を返す（例外は送出しない）。

    Args:
        scored_labels: (ラベル, スコア) の列。順序は問わない。
        policy: 分類器が守る契約（allowlist / max_candidates）。
        mapper: スコアを `ConfidenceLevel` に変換する callable
            （`confidence_mapper_from_thresholds` 等）。

    Returns:
        allowlist で許可された候補のみをレベル降順に並べ、`max_candidates` で
        切り詰めた `IntentPrediction`（`report=None` / `metadata=None`）。

    Raises:
        ValueError: mapper がスコアを範囲外と判定した場合（mapper から伝播）。
    """
    # post-hoc (1): 重複集約（同一ラベルは最高スコアを残す・初出順を保持）
    best_scores: dict[str, float] = {}
    for label, score in scored_labels:
        current = best_scores.get(label)
        if current is None or score > current:
            best_scores[label] = score

    # post-hoc (2): allowlist フィルタ + 除外時 WARNING ログ
    allowed_names = {c.name for c in policy.categories}
    accepted_scores: dict[str, float] = {}
    rejected_texts: list[str] = []
    for label, score in best_scores.items():
        if label in allowed_names:
            accepted_scores[label] = score
        else:
            rejected_texts.append(label)

    if rejected_texts:
        # ラベルは外部由来テキストを含みうるため repr 化（ログフォージング CWE-117 対策）
        logger.warning(
            "intent classifier removed %d candidates outside allowlist: %s",
            len(rejected_texts),
            [repr(t) for t in rejected_texts],
        )

    # post-hoc (3): mapper でスコア→ConfidenceLevel 変換
    accepted: list[IntentCandidate] = [
        IntentCandidate(text=label, level=mapper(score)) for label, score in accepted_scores.items()
    ]

    # post-hoc (4): レベル降順 stable sort
    accepted.sort(key=lambda c: _LEVEL_ORDER[c.level])

    # post-hoc (5): max_candidates で切り詰め
    accepted = accepted[: policy.max_candidates]

    return IntentPrediction(candidates=tuple(accepted), report=None, metadata=None)


class MLCandidateGenerator:
    """ML 推論 callable を `CandidateGenerator` Protocol へ橋渡しする薄いアダプタ。

    利用側が渡す推論 callable（sklearn 互換モデル等をラップした同期関数、または
    非同期関数）を呼び、返った (ラベル, スコア) 列を FR-2
    `prediction_from_scored_labels` に委譲して `IntentPrediction` を組み立てる。
    `_llm.py` の `LLMCandidateGenerator` と同じく `CandidateGenerator` Protocol
    （`@runtime_checkable`）を duck typing で満たす（継承不要）。

    同期 / 非同期の判別は構築時に `inspect.iscoroutinefunction` で 1 度だけ行い、
    結果を private 属性に保持する。同期 callable は `asyncio.to_thread` で別スレッド
    実行し、イベントループをブロックしない。推論の送出例外は握り潰さず
    `generate()` から伝播する。
    """

    def __init__(
        self,
        inference: Callable[
            [IntentContext[Any]],
            Sequence[tuple[str, float]] | Awaitable[Sequence[tuple[str, float]]],
        ],
        *,
        policy: IntentPolicy,
        mapper: Callable[[float], ConfidenceLevel],
    ) -> None:
        """ML 分類器アダプタを初期化する。

        Args:
            inference: `IntentContext` を受け取り (ラベル, スコア) 列を返す推論
                callable。同期関数 / 非同期関数（コルーチン関数）のいずれも受け付ける。
            policy: 分類器が守る契約（allowlist / max_candidates）。keyword-only。
            mapper: スコアを `ConfidenceLevel` に変換する callable
                （`confidence_mapper_from_thresholds` 等）。keyword-only。
        """
        self._inference = inference
        self._policy = policy
        self._mapper = mapper
        self._is_async = inspect.iscoroutinefunction(inference)

    async def generate(self, context: IntentContext[Any]) -> IntentPrediction:
        """推論 callable を呼び出し、FR-2 変換済みの `IntentPrediction` を返す。

        非同期 inference は `await` で、同期 inference は `asyncio.to_thread` で
        別スレッド実行する（イベントループをブロックしない）。`context` は変換せず
        そのまま inference に素通しする。

        Args:
            context: ContextBuilder が組み立てた整形済み文脈。

        Returns:
            allowlist で許可された候補のみをレベル降順に並べ、`max_candidates` で
            切り詰めた `IntentPrediction`（`report=None` / `metadata=None`）。

        Raises:
            Exception: inference が送出した例外は握り潰さずそのまま伝播する。
            ValueError: mapper がスコアを範囲外と判定した場合（mapper から伝播）。
        """
        if self._is_async:
            result = await self._inference(context)  # type: ignore[misc]
        else:
            result = await asyncio.to_thread(self._inference, context)
        # `inspect.iscoroutinefunction` は callable オブジェクトの async `__call__` や
        # coroutine を返す同期ラッパーを検出できない。呼び出し後に awaitable かを
        # 判定し、そうであれば await して型注釈どおりの契約を満たす。
        if inspect.isawaitable(result):
            result = await result
        return prediction_from_scored_labels(result, policy=self._policy, mapper=self._mapper)
