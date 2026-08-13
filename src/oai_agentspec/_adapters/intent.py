"""意図予測用 LLM 呼び出しのアダプタ（SDK 隔離窓口）。

runtime/intent/ の非 _adapters ファイルは agents を import せず、この薄いラッパを介して
SDK に触れる（judge.py と同型）。agents.Model を Any で受け、Agent + Runner.run で
文字列応答を得る最小関数のみを提供する。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

FILLER_MAX_TURNS: Final[int] = 1
"""パラメータ予測エージェントに許すターン数（1 ターンあたり 1 回のモデル呼び出し）。"""


@dataclass(frozen=True)
class AgentRunUsage:
    """1 回の Runner 実行で観測したモデル利用量（agents 非依存の値型）。

    Attributes:
        model_calls: モデル応答の件数（`raw_responses` の実測値）。
        input_tokens: 入力トークン合計。usage を取得できない場合は None。
        output_tokens: 出力トークン合計。usage を取得できない場合は None。
    """

    model_calls: int
    input_tokens: int | None
    output_tokens: int | None


async def run_intent_prompt(
    model: Any,
    system: str,
    history_items: tuple[Mapping[str, Any], ...],
    user_content: str,
    *,
    context: Any = None,
    model_settings: Any = None,
) -> str:
    """agents.Agent + Runner.run を薄くラップして意図予測 LLM 応答を str で返す。

    Args:
        model: agents.Model 相当（呼び出し側 DI）。lib 内では Any として扱う。
        system: LLM に渡す system instructions。空文字は `instructions=None` として扱う。
        history_items: 過去 turn の SDK 互換 dict tuple。
        user_content: 現在発話の user content。空文字の場合は user turn を追加せず
            履歴のみを送る。
        context: RunContext。`RunContextWrapper` の場合は `.context` を展開して forward、
            None も可（keyword-only）。
        model_settings: agents.ModelSettings 相当（不透明型・keyword-only）。None なら
            SDK 既定に委ねる。reasoning effort / verbosity / max_tokens 等のチューニングを
            利用側 DI で渡すための pass-through。

    Returns:
        LLM の final_output を str 化したもの。None は空文字。

    Raises:
        ValueError: user_content と history_items の両方が空の場合（utterance が空でも
            prompt callable が非空を返せば送信は行われる点に注意）。
        Exception: モデル呼び出しで発生した例外はそのまま伝播する（catch しない）。
    """
    from agents import Agent, Runner  # 関数内遅延 import（NFR-1）

    from .run_context import unwrap_run_context

    agent_kwargs: dict[str, Any] = {
        "name": "intent-classifier",
        "instructions": system or None,
        "model": model,
    }
    if model_settings is not None:
        agent_kwargs["model_settings"] = model_settings
    agent = Agent(**agent_kwargs)
    input_items: list[Mapping[str, Any]] = list(history_items)
    if user_content:
        input_items.append({"role": "user", "content": user_content})
    if not input_items:
        raise ValueError("intent classification requires a non-empty utterance or history items")
    raw_ctx = unwrap_run_context(context)
    result = await Runner.run(agent, input=input_items, context=raw_ctx)
    output = result.final_output
    return "" if output is None else str(output)


def _collect_usage(raw_responses: Iterable[Any]) -> AgentRunUsage:
    """モデル応答列から `AgentRunUsage` を組み立てる。

    SDK の `Usage` は全フィールド非 Optional かつ既定 0 のため、0 と未取得を型で区別できない。
    全応答が `requests == 0` かつ `total_tokens == 0` の場合のみ未取得と判定し、トークンを
    None にする。件数は usage の内容に依存せず常に実測値を残す。

    Args:
        raw_responses: `RunResult.raw_responses` 相当のモデル応答列。

    Returns:
        件数とトークン合計（未取得なら None）を載せた `AgentRunUsage`。
    """
    responses = list(raw_responses)
    input_total = 0
    output_total = 0
    measured = False
    for response in responses:
        usage = response.usage
        if usage.requests or usage.total_tokens:
            measured = True
        input_total += usage.input_tokens
        output_total += usage.output_tokens
    if not measured:
        return AgentRunUsage(model_calls=len(responses), input_tokens=None, output_tokens=None)
    return AgentRunUsage(
        model_calls=len(responses), input_tokens=input_total, output_tokens=output_total
    )


async def run_filler_prompt(
    agent: Any,
    history_items: tuple[Mapping[str, Any], ...],
    user_content: str,
    *,
    context: Any = None,
) -> tuple[str, AgentRunUsage]:
    """構築済みのパラメータ予測エージェントを 1 回だけ走らせて応答と利用量を返す。

    予測エージェント専用 `AgentRegistry` の生成・`AgentSpec` の宣言・ガードレール登録名の
    解決は上位層（`runtime/intent`）の責務であり、本関数は解決済みの不透明な実体をそのまま
    `Runner.run` へ渡す。会話履歴は `session` ではなく `history_items` として明示的に渡す。

    Args:
        agent: 構築済みの agents.Agent 相当（不透明型）。
        history_items: 過去 turn の SDK 互換 dict tuple。
        user_content: 現在発話の user content。空文字の場合は user turn を追加しない。
        context: RunContext。`RunContextWrapper` の場合は `.context` を展開して forward、
            None も可（keyword-only）。

    Returns:
        `(応答テキスト, AgentRunUsage)`。final_output が None の場合テキストは空文字。

    Raises:
        ValueError: user_content と history_items の両方が空の場合。
        Exception: モデル呼び出し・ガードレール発火・ターン上限超過で発生した SDK 例外は
            握り潰さずそのまま伝播する。
    """
    from agents import Runner  # 関数内遅延 import（NFR-1）

    from .run_context import unwrap_run_context

    input_items: list[Mapping[str, Any]] = list(history_items)
    if user_content:
        input_items.append({"role": "user", "content": user_content})
    if not input_items:
        raise ValueError("parameter filling requires a non-empty utterance or history items")
    raw_ctx = unwrap_run_context(context)
    result = await Runner.run(agent, input=input_items, context=raw_ctx, max_turns=FILLER_MAX_TURNS)
    output = result.final_output
    text = "" if output is None else str(output)
    return text, _collect_usage(result.raw_responses)
