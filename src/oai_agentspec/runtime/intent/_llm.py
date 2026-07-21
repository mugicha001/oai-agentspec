"""LLMCandidateGenerator: LLM を呼び post-hoc 3 段の policy 適用を行う。

3 段の内訳: allowlist フィルタ + WARNING ログ / レベル降順 sort / max_candidates truncate。

`_adapters.intent.run_intent_prompt` を経由して LLM を呼び、返却 JSON を
`IntentPrediction` にパースしたうえで `IntentPolicy` を post-hoc 適用する。
適用段は (1) allowlist によるフィルタ + 除外時 WARNING ログ、
(2) ConfidenceLevel 降順への stable sort、(3) `max_candidates` での truncate の 3 段。

方針:
- `require_rationale` フィルタは行わない（フィールド廃止済み）。
- LLM 出力 metadata は pass-through で伝搬し、lib 側で `rejected` 等の予約キーを
  差し込まない（利用側が触った状態を尊重する）。
- SDK 隔離のため `agents` は import せず、モデルは不透明型で受ける。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from ..._adapters.intent import run_intent_prompt
from .types import (
    ConfidenceLevel,
    IntentCandidate,
    IntentContext,
    IntentPolicy,
    IntentPrediction,
)

logger = logging.getLogger(__name__)

# ConfidenceLevel 宣言順（CERTAIN → HIGH → MEDIUM → LOW → SPECULATIVE）を降順ソートキーに転用。
# enum を単一ソースとし、値追加時の二重管理を避ける（`test_confidence_level_ordering` で順序 pin）。
_LEVEL_ORDER: dict[ConfidenceLevel, int] = {level: idx for idx, level in enumerate(ConfidenceLevel)}


# Markdown コードフェンス（```json ... ``` / ``` ... ```）で全体が包まれた応答にマッチする。
# 改行の有無・言語タグ直後に本文が続く形（```json{...}```）も 1 パターンで剥がす。
_FENCE_RE = re.compile(r"^```[\w-]*\s*(.*?)\s*```$", re.DOTALL)


def _strip_code_fence(raw: str) -> str:
    """LLM 応答が Markdown コードフェンスで包まれていた場合に中身だけを取り出す。

    低精度・高速モデルは「JSON のみ」の指示にもかかわらず ```json ... ``` で
    包んで返すことがあるため、パース前の耐性として剥がす。単一行フェンス
    （```json {...}```）や言語タグ直後に本文が続く形も対象。フェンスでない応答は
    そのまま返す（strip 以外の加工はしない）。

    Args:
        raw: LLM の生応答テキスト。

    Returns:
        フェンスを剥がした（または元のままの）テキスト。
    """
    text = raw.strip()
    match = _FENCE_RE.match(text)
    return match.group(1).strip() if match else text


class LLMCandidateGenerator:
    """LLM を用いた `CandidateGenerator` Protocol の実装。

    `run_intent_prompt` を呼び、返却 JSON を `IntentPrediction` にパースし、
    `IntentPolicy` の allowlist / sort / truncate を post-hoc 3 段で適用する。
    `require_rationale` フィルタは行わず、`metadata.rejected` の記録もしない
    （metadata は LLM 出力を pass-through する）。
    """

    def __init__(
        self,
        model: Any,
        prompt: Callable[[IntentContext[Any]], str],
        *,
        policy: IntentPolicy,
        include_policy_in_system: bool = True,
    ) -> None:
        """LLM 分類器を初期化する。

        Args:
            model: LLM モデル（agents.Model 相当・不透明型）。
            prompt: `IntentContext` から user 入力文字列を組み立てる callable。
            policy: 分類器が守る契約（allowlist / max_candidates 等）。
            include_policy_in_system: True なら `policy.render_prompt()` を system に注入する。
                False の場合、categories と出力 JSON schema を LLM に伝達する責務は
                `prompt` callable 側に移る（利用側が全制御するための escape hatch）。
        """
        self._model = model
        self._prompt = prompt
        self._policy = policy
        self._include_policy_in_system = include_policy_in_system

    async def generate(self, context: IntentContext[Any]) -> IntentPrediction:
        """LLM を呼び出し、post-hoc 3 段適用済みの `IntentPrediction` を返す。

        Args:
            context: ContextBuilder が組み立てた整形済み文脈。

        Returns:
            allowlist で許可された候補のみをレベル降順に並べ、`max_candidates` で
            切り詰めた `IntentPrediction`。metadata は LLM 出力を pass-through する。

        Raises:
            pydantic.ValidationError: LLM の返却 JSON が `IntentPrediction` の
                契約を満たさない（不正 JSON・未知の ConfidenceLevel 等）場合。
                例外メッセージには LLM の生出力の一部が含まれるため、外部露出
                （ログ・API レスポンス）前に握り替えを検討すること。
            ValueError: prompt callable の返す user_content と history_items の
                両方が空の場合（adapter の fail-fast から伝播）。
        """
        system = self._policy.render_prompt() if self._include_policy_in_system else ""
        user_content = self._prompt(context)
        raw = await run_intent_prompt(
            self._model,
            system,
            context.history_items,
            user_content,
            context=context.run_context,
        )
        parsed = IntentPrediction.model_validate_json(_strip_code_fence(raw))

        # post-hoc (1): allowlist フィルタ + 除外時 WARNING ログ
        allowed_names = {c.name for c in self._policy.categories}
        accepted: list[IntentCandidate] = []
        rejected_texts: list[str] = []
        for cand in parsed.candidates:
            if cand.text in allowed_names:
                accepted.append(cand)
            else:
                rejected_texts.append(cand.text)

        if rejected_texts:
            # LLM 由来テキストは制御文字/改行を含みうるため repr 化
            # (ログフォージング CWE-117 対策)
            logger.warning(
                "intent classifier removed %d candidates outside allowlist: %s",
                len(rejected_texts),
                [repr(t) for t in rejected_texts],
            )

        # post-hoc (2): レベル降順 stable sort（同レベル内は LLM 出力順を保持）
        accepted.sort(key=lambda c: _LEVEL_ORDER[c.level])

        # post-hoc (3): max_candidates で切り詰め
        accepted = accepted[: self._policy.max_candidates]

        return IntentPrediction(
            candidates=tuple(accepted),
            report=parsed.report,
            metadata=parsed.metadata,  # pass-through: lib 側で予約キーを差し込まない
        )
