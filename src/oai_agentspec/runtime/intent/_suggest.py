"""候補取得と allowlist 除外（`planner.plan()` の段 (1)・設計 §3.13）。

`ContextBuilder` を選んで `IntentContext` を組み、`CandidateGenerator` を 1 回だけ呼び、
返った候補を宣言済み `action_id` の allowlist で絞って `ExecutableSuggestion` にまとめる。

方針:
- **除外は 1 経路にまとめる。** 未登録 `action_id` の候補と、そもそも `ExecutableIntent`
  ではない候補（`IntentPrediction.candidates` の宣言型は親型 `IntentCandidate` であり
  素の候補が混じりうる）を同じ判定で落とし、同一の WARNING 1 行で報告する（設計 §3.4c）。
  `getattr(c, "action_id", None)` のような防御的読み取りで擬似的に受け入れると、
  `action_id` を持たない候補が決定的段へ流れて別の失敗にすり替わる。
- 除外名は既存 `_llm.py` と同じく `repr` 化して 1 レコードへ載せる。候補テキストは
  LLM 由来で制御文字・改行を含みうるため（ログフォージング CWE-117 対策）。
- 候補の順序・フィールドは加工しない（並べ替えと切り詰めは generator 側の関心事）。
  `report` / `metadata` は素通しで、lib 側が予約キーを差し込まない。
- `generator` / `context_builder` の例外は握り潰さず伝播する（FR-4）。
- `catalog` を受け取らず allowlist を名前の集合として受ける（設計 §2.1 の依存図）。
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Any

from ._default import DefaultContextBuilder
from .binding import CandidateSource
from .protocols import ContextBuilder
from .types import ExecutableIntent, ExecutableSuggestion, IntentQuery

logger = logging.getLogger(__name__)


async def _suggest_intents(
    query: IntentQuery[Any],
    source: CandidateSource,
    allowed: Collection[str],
) -> ExecutableSuggestion:
    """候補を生成し、宣言済み `action_id` だけに絞った `ExecutableSuggestion` を返す。

    Args:
        query: 入力クエリ。`ContextBuilder` へそのまま渡す。
        source: 候補の出どころ。`context_builder` が `None` なら `DefaultContextBuilder`
            を使い、`history_limit` が `None` ならその既定（20 件）に委ねる。
        allowed: 宣言済み `action_id` の集まり。ここに無い候補は通さない（NFR-6）。

    Returns:
        残った候補（generator の順序のまま）と、生成に使った `IntentContext`、
        `report` / `metadata` の素通しを載せた `ExecutableSuggestion`。全候補が除外されても
        例外にはせず、空の `candidates` を持つ結果を返す。

    Raises:
        Exception: `context_builder.build()` / `generator.generate()` が送出した例外は
            種別を問わずそのまま伝播する。ここで握ると候補が空になった理由が失われる。
    """
    builder: ContextBuilder = source.context_builder
    if builder is None:
        builder = (
            DefaultContextBuilder()
            if source.history_limit is None
            else DefaultContextBuilder(history_limit=source.history_limit)
        )

    context = await builder.build(query)
    prediction = await source.generator.generate(context)

    allowed_ids = set(allowed)
    accepted: list[ExecutableIntent] = []
    rejected: list[str] = []
    for candidate in prediction.candidates:
        # 派生も通すため isinstance で判定する（`type(...) is` にすると利用者の派生が落ちる）。
        executable = candidate if isinstance(candidate, ExecutableIntent) else None
        if executable is not None and executable.action_id in allowed_ids:
            accepted.append(executable)
        else:
            rejected.append(executable.action_id if executable is not None else candidate.text)

    if rejected:
        # 候補テキストは制御文字/改行を含みうるため repr 化 (ログフォージング CWE-117 対策)
        logger.warning(
            "intent suggestion removed %d candidates outside the declared actions: %s",
            len(rejected),
            [repr(t) for t in rejected],
        )

    return ExecutableSuggestion(
        candidates=tuple(accepted),
        context=context,
        report=prediction.report,
        metadata=prediction.metadata,
    )
