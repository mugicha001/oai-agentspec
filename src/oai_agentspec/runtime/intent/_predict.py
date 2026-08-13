"""不足パラメータの予測委譲（`planner.plan()` の段 (3)・設計 §3.13 / ADR 0026）。

`NEEDS_LLM` のスロットを持つ計画がある場合に限り、予測エージェントを **1 ターンあたり
1 回だけ**駆動して値を埋める。候補件数・不足件数に比例して呼び出しが増えないよう、全候補の
不足パラメータを `candidate_<index>` を持つ 1 つの複合モデルへまとめて 1 回で問い合わせる。

方針:
- **`ActionCatalog` を受け取らない**（設計 §3.13）。読むのは `plan.spec.parameters` と
  `plan.resolved_*` と `IntentContext` だけであり、既定マージ解決は決定的段で済んでいる。
- **提示スキーマと応答検証モデルを分ける**（設計 §3.8）。提示側は `slots._slots_model` の
  宣言型そのままで `maxItems` を含み、検証側は値を `Any` として受ける緩いモデルである。
  検証側まで宣言型で縛ると、1 スロットの型違いで応答全体が `ValidationError` になり、
  FR-7 が `on_invalid_slot` の管轄と定めた「型に合わない値」が `on_invalid_response` へ
  倒れてしまう。宣言型の検査はスロット単位に `TypeAdapter` で行う（`slots.apply` と同型）。
- 予測エージェントの宣言・実体化・ガードレール登録名の解決は本モジュールが行い、
  `_adapters.intent.run_filler_prompt` へは不透明値として渡す（設計 §3.4・NFR-1）。
  専用の `AgentRegistry` を毎回組むため、利用者の業務 registry には触れない。
- **発話（`IntentQuery.utterance`）を system 指示部へ連結しない**（NFR-6）。会話は
  `history_items` として会話部分に渡り、指示部にはセグメント本文だけが積まれる。
- 同一セグメントの本文は重複排除する（NFR-5 / ADR 0026 (b)）。候補が増えても指示部が
  同じ本文で膨らまない。
- 応答の parse 失敗（`on_invalid_response`）と、スロット単位の欠落・型不一致
  （`on_invalid_slot`）を別の管轄として扱う。後者の後退は当該スロットに閉じ、決定的段で
  確定済みのスロットと他候補の計画は保持する（FR-7）。
"""

from __future__ import annotations

import json
from typing import Any, Final

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from ..._adapters.intent import run_filler_prompt
from ...registry import AgentRegistry
from ...spec import AgentSpec
from ._llm import _LEVEL_ORDER, _strip_code_fence
from ._models import build_frozen_model, derive_optional_model
from .actions import ParameterSpec, _model_name, _resolve_path
from .slots import (
    ActionPlan,
    Origin,
    ParamUsage,
    Slot,
    SlotState,
    SlotSuggestion,
    _settle_slot,
    _slots_model,
)
from .types import IntentContext

FILLER_AGENT_NAME: Final[str] = "oai_agentspec_action_param_filler"
"""予測エージェントの宣言名。lib が固定で持ち、利用者の業務 registry とは無関係である。"""

#: 複合応答モデル（提示用 / 検証用）のクラス名。`model_json_schema()` の `title` になる。
_COMPOSITE_MODEL_NAME: Final[str] = "ActionCandidateSlots"

#: 予測を行わなかったターンの実行量。`Runner.run` が 0 回であることを値として表す。
_NO_USAGE: Final[ParamUsage] = ParamUsage(
    runs=0, model_calls=0, candidates=0, input_tokens=None, output_tokens=None
)


async def _predict_params(
    plans: tuple[ActionPlan, ...],
    context: IntentContext[Any],
    *,
    llm_filler: Any,
    prompts: Any = None,
    guardrail_registry: Any = None,
) -> tuple[tuple[ActionPlan, ...], ParamUsage]:
    """`NEEDS_LLM` のスロットを 1 回の予測でまとめて埋める（FR-6 / FR-7）。

    Args:
        plans: 決定的段が返した計画列。順序と件数はそのまま保たれる。
        context: 会話部分（`history_items`）と `run_context` を持つ `IntentContext`。
        llm_filler: 穴埋めの結線（`LLMFiller`）。`model` / `on_invalid_response` /
            `guardrails` を読む。
        prompts: セグメントを解決する `PromptStore`。セグメント宣言が 1 件も無ければ
            `None` でよい。
        guardrail_registry: ガードレール登録名を解決する `GuardrailRegistry`。
            `llm_filler.guardrails` が空なら `None` でよい。

    Returns:
        `(埋めた後の計画列, ParamUsage)`。`NEEDS_LLM` が 1 件も無ければ計画列は入力
        そのままで、`ParamUsage` は 0 件を表す値になる（`Runner.run` は 0 回）。

    Raises:
        RuntimeError: 予測対象があるのに会話部分（`context.history_items`）が空の場合。
            発話は指示部へ連結しない（NFR-6）ため、会話が予測エージェントへ届く唯一の
            経路は `history_items` であり、空のまま駆動するとモデルは文脈ゼロで当て推量を
            返す（暗黙失敗）。`run_filler_prompt` を呼ぶ前に落とす。文言は
            `IntentQuery(history=...)` の結線に加えて、現在発話をセッションへ積む必要
            （初回ターンで積み忘れると `history_items` が空になる）まで誘導する。
        RuntimeError: セグメント宣言があるのに `prompts` が未結線の場合
            （`bind` の規則 2 と同じ扱い）。
        ValidationError: 応答を parse できず `on_invalid_response="error"`（既定）の場合。
        ValueError: スロットの値が欠落または宣言型に合わず、当該計画の
            `resolved_on_invalid_slot` が `"error"` の場合。文言にパラメータ名を載せる。
        Exception: ガードレール発火・ターン上限超過などの SDK 例外はそのまま伝播する
            （後退挙動を適用しない）。
    """
    targets = tuple((index, plan) for index, plan in enumerate(plans) if _pending_params(plan))
    if not targets:
        return plans, _NO_USAGE
    if not context.history_items:
        raise RuntimeError(
            "parameters must be predicted but the conversation history is empty; pass "
            "IntentQuery(history=session) and make sure the current utterance has been "
            "appended to the session before planning: "
            f"{sorted({plan.action_id for _index, plan in targets})}"
        )

    agent = _build_filler_agent(
        _compose_instructions(targets, context, prompts), llm_filler, guardrail_registry
    )
    raw, run_usage = await run_filler_prompt(
        agent,
        context.history_items,
        _render_user_content(targets),
        context=context.run_context,
    )
    usage = ParamUsage(
        runs=1,
        model_calls=run_usage.model_calls,
        candidates=len(targets),
        input_tokens=run_usage.input_tokens,
        output_tokens=run_usage.output_tokens,
    )

    parsed = _parse_response(raw, targets, llm_filler)
    filled = list(plans)
    for index, plan in targets:
        answer = None if parsed is None else getattr(parsed, _field_name(index), None)
        filled[index] = _fill_plan(plan, answer)
    return tuple(filled), usage


def _pending_params(plan: ActionPlan) -> tuple[ParameterSpec, ...]:
    """`NEEDS_LLM` のスロットに対応するパラメータ宣言を宣言順で返す。

    `Slot` は `prompt` / `confirm` / `max_suggestions` / `default` を持たないため、予測段が
    必要とする値はすべて宣言側から読む（設計 §3.13 の表）。

    Args:
        plan: 対象の計画。

    Returns:
        `NEEDS_LLM` のパラメータ宣言の tuple。該当が無ければ空 tuple。
    """
    names = {slot.name for slot in plan.slots if slot.state is SlotState.NEEDS_LLM}
    return tuple(param for param in plan.spec.parameters if param.name in names)


def _field_name(index: int) -> str:
    """候補の位置から複合モデルのフィールド名を組む。

    同一 `action_id` の候補が複数あっても衝突しないよう、鍵は `action_id` ではなく候補列
    での位置にする（FR-6 L170）。

    Args:
        index: 計画列での位置。

    Returns:
        `candidate_<index>` 形式のフィールド名。
    """
    return f"candidate_{index}"


# ---- プロンプト合成（タスク 2-2） ----


def _compose_instructions(
    targets: tuple[tuple[int, ActionPlan], ...],
    context: IntentContext[Any],
    prompts: Any,
) -> str | None:
    """予測対象の計画が要求するセグメント本文を重複なく積んだ system 指示部を組む。

    積むのは「計画のマージ済みセグメント」と「`NEEDS_LLM` のパラメータが宣言した
    セグメント」だけである。既に確定したパラメータのセグメントを積むと、埋める必要が
    ないパラメータの指示が指示部を占める（FR-6 L167）。

    Args:
        targets: `(位置, 計画)` の組。予測対象の計画だけを含む。
        context: `prompt_vars` のパス解決に使う `run_context` を持つ `IntentContext`。
        prompts: セグメントを解決する `PromptStore`。

    Returns:
        セグメント本文を `\\n\\n` で連結した文字列。セグメント宣言が 1 件も無ければ
        `None`（空文字は `AgentSpec.instructions` として送らない）。

    Raises:
        RuntimeError: セグメント宣言があるのに `prompts` が未結線の場合。
        PromptResolutionError: セグメントを解決できない場合（起動時検証を経ていれば
            起きないため、ここでは捕捉せず伝播させる）。
    """
    requested = tuple((plan, _segment_names(plan)) for _index, plan in targets)
    declared = [name for _plan, names in requested for name in names]
    if not declared:
        return None
    if prompts is None:
        raise RuntimeError(
            "prompt segments are declared but no PromptStore is wired; pass "
            f"bind(prompts=...) to fill parameters with them: {declared}"
        )

    bodies: dict[str, None] = {}
    for plan, names in requested:
        variables = _resolved_vars(plan, context)
        for name in names:
            bodies.setdefault(prompts.compose(layout=[name], vars=variables), None)
    return "\n\n".join(bodies)


def _segment_names(plan: ActionPlan) -> tuple[str, ...]:
    """当該計画の穴埋めで積むセグメント名を重複なく宣言順で返す。

    Args:
        plan: 対象の計画。

    Returns:
        マージ済みセグメントに `NEEDS_LLM` のパラメータのセグメントを続けた tuple。
    """
    names: dict[str, None] = dict.fromkeys(plan.resolved_prompt)
    for param in _pending_params(plan):
        if param.prompt is not None:
            names.setdefault(param.prompt)
    return tuple(names)


def _resolved_vars(plan: ActionPlan, context: IntentContext[Any]) -> dict[str, Any]:
    """`resolved_prompt_vars` の各値をパスとして `run_context` から解決する（FR-6 L168）。

    解決できなかったパスは空文字へ展開する。`None` を str 化するとリテラルの `"None"` が
    プロンプトへ載り、モデルがそれを値として読む。

    Args:
        plan: 対象の計画。
        context: 解決先の `run_context` を持つ `IntentContext`。

    Returns:
        `PromptStore.compose(vars=...)` へ渡す変数の辞書。
    """
    resolved: dict[str, Any] = {}
    for key, path in plan.resolved_prompt_vars.items():
        value = _resolve_path(context.run_context, path)
        resolved[key] = "" if value is None else str(value)
    return resolved


def _render_user_content(targets: tuple[tuple[int, ActionPlan], ...]) -> str:
    """候補と `candidate_<index>` の対応表と、出力形式のスキーマを載せた user content を組む。

    発話は一切連結しない（NFR-6）。会話は `history_items` として別に渡る。

    Args:
        targets: `(位置, 計画)` の組。

    Returns:
        対応表・スキーマ・「JSON のみを返す」制約を含む user content。
    """
    schema = _presentation_model(targets).model_json_schema()
    lines = ["Fill the missing parameters for each candidate below.", "", "Candidates:"]
    lines += [
        f"- {_field_name(index)}: action_id={plan.action_id!r}, label={plan.label!r}"
        for index, plan in targets
    ]
    lines += [
        "",
        "Answer with json only, with no prose and no code fence, matching this schema:",
        json.dumps(schema, ensure_ascii=False, indent=2),
    ]
    return "\n".join(lines)


# ---- 複合応答モデル（タスク 2-3） ----


def _presentation_model(targets: tuple[tuple[int, ActionPlan], ...]) -> type[BaseModel]:
    """LLM へ提示する複合スキーマモデルを組む（宣言型そのまま・`maxItems` を含む）。

    Args:
        targets: `(位置, 計画)` の組。

    Returns:
        `candidate_<index>` を計画ごとのスキーマモデルへ対応させた frozen なモデル。
    """
    fields = {
        _field_name(index): (_slots_model(plan), Field(description=plan.spec.description))
        for index, plan in targets
    }
    return build_frozen_model(_COMPOSITE_MODEL_NAME, fields)


def _parse_model(targets: tuple[tuple[int, ActionPlan], ...]) -> type[BaseModel]:
    """応答検証に使う複合モデルを組む（全フィールド任意・値は `Any`）。

    外側（`candidate_<index>`）も内側（各パラメータ）も任意にする。欠落を
    `ValidationError` にすると、1 スロットが欠けただけで応答全体が捨てられ、スロット単位の
    後退（FR-7 L222）が成立しない。値を `Any` で受けるのは、宣言型の検査を
    `_validated_suggestions` がスロット単位で行うためである。

    Args:
        targets: `(位置, 計画)` の組。

    Returns:
        `candidate_<index>` を任意フィールドとして持つ frozen なモデル。
    """
    fields = {
        _field_name(index): (
            derive_optional_model(_loose_slots_model(plan)) | None,
            Field(default=None),
        )
        for index, plan in targets
    }
    return build_frozen_model(_COMPOSITE_MODEL_NAME, fields)


def _loose_slots_model(plan: ActionPlan) -> type[BaseModel]:
    """`NEEDS_LLM` のパラメータを値型 `Any` で受けるスキーマモデルを組む。

    `slots._slots_model` との違いは、フィールド型が `SlotSuggestion[T]` ではなく
    `SlotSuggestion[Any]` である点と、`max_length` を持たない点である（設計 §3.8）。

    Args:
        plan: 対象の計画。

    Returns:
        `NEEDS_LLM` のパラメータを宣言順に持つ frozen なモデル。
    """
    fields: dict[str, tuple[Any, Any]] = {
        param.name: (
            list[SlotSuggestion[Any]] if param.max_suggestions > 1 else SlotSuggestion[Any],
            Field(description=param.description),
        )
        for param in _pending_params(plan)
    }
    return build_frozen_model(_model_name(plan.action_id, "Slots"), fields)


# ---- 予測エージェント（タスク 2-4） ----


def _build_filler_agent(instructions: str | None, llm_filler: Any, guardrail_registry: Any) -> Any:
    """予測エージェント専用の `AgentRegistry` を組んで実体を 1 つ返す（設計 §3.4）。

    登録名から実体への解決と境界別の振り分けは registry の結線に委ねる。戻り値は不透明値
    として扱い、属性へは触れない（NFR-1）。

    Args:
        instructions: 合成済みの system 指示部。セグメントが無ければ `None`。
        llm_filler: `model` / `guardrails` を持つ `LLMFiller`。
        guardrail_registry: ガードレール登録名の解決簿。宣言が無ければ `None` でよい。

    Returns:
        構築済みのエージェント（不透明値）。

    Raises:
        KeyError: 宣言されたガードレール登録名を解決できない場合（起動時検証を経ていれば
            起きない）。
    """
    registry = AgentRegistry(guardrail_registry=guardrail_registry)
    registry.register(
        AgentSpec(
            name=FILLER_AGENT_NAME,
            instructions=instructions,
            model=llm_filler.model,
            guardrails=list(llm_filler.guardrails),
        )
    )
    return registry.get(FILLER_AGENT_NAME)


def _parse_response(
    raw: str,
    targets: tuple[tuple[int, ActionPlan], ...],
    llm_filler: Any,
) -> BaseModel | None:
    """応答テキストを複合モデルへ parse する（コードフェンスは剥がす）。

    Args:
        raw: 予測エージェントの応答テキスト。
        targets: `(位置, 計画)` の組。
        llm_filler: `on_invalid_response` を持つ `LLMFiller`。

    Returns:
        parse 済みのモデル。parse に失敗し `on_invalid_response="skip"` の場合は `None`
        （全 `NEEDS_LLM` をスロット単位の後退へ倒す合図）。

    Raises:
        ValidationError: parse に失敗し `on_invalid_response="error"`（既定）の場合。
            文言には LLM の生出力が含まれるため、そのままログや API 応答へ流さないこと。
    """
    try:
        return _parse_model(targets).model_validate_json(_strip_code_fence(raw))
    except ValidationError:
        if llm_filler.on_invalid_response == "error":
            raise
        return None


# ---- スロットへの反映と後退（タスク 2-4 / 2-6） ----


def _fill_plan(plan: ActionPlan, answer: BaseModel | None) -> ActionPlan:
    """1 計画ぶんの `NEEDS_LLM` スロットを予測結果で置き換えた新しい計画を返す。

    `NEEDS_LLM` 以外のスロットはそのまま引き継ぐ（NFR-6）。

    Args:
        plan: 対象の計画。
        answer: 当該候補ぶんの parse 済み応答。応答不正・当該候補の欠落なら `None`。

    Returns:
        スロットを差し替えた新しい `ActionPlan`。

    Raises:
        ValueError: 欠落または型不一致があり `resolved_on_invalid_slot` が `"error"` の場合。
    """
    params = {param.name: param for param in _pending_params(plan)}
    slots = tuple(
        _fill_slot(params[slot.name], answer, plan.resolved_on_invalid_slot)
        if slot.state is SlotState.NEEDS_LLM
        else slot
        for slot in plan.slots
    )
    return plan.model_copy(update={"slots": slots})


def _fill_slot(param: ParameterSpec, answer: BaseModel | None, on_invalid_slot: str) -> Slot:
    """パラメータ 1 つを予測結果で埋める。埋まらなければ後退させる（FR-6 / FR-7）。

    Args:
        param: 対象のパラメータ宣言。
        answer: 当該候補ぶんの parse 済み応答。
        on_invalid_slot: 当該計画で解決済みの不正スロット時の挙動。

    Returns:
        `confirm=True` なら `NEEDS_CONFIRMATION`、偽なら `RESOLVED` の `Slot`。値を得られ
        なければ `default` 由来の `Slot`（`Origin.DEFAULT`）か `NEEDS_USER` の `Slot`。

    Raises:
        ValueError: 値が欠落または宣言型に合わず `on_invalid_slot` が `"error"` の場合。
    """
    raw = None if answer is None else getattr(answer, param.name, None)
    suggestions = _validated_suggestions(param, raw, on_invalid_slot)
    if not suggestions:
        return _fallback_slot(param)
    suggestions.sort(key=lambda suggestion: _LEVEL_ORDER[suggestion.level])
    chosen = suggestions[: param.max_suggestions]
    if param.confirm:
        return Slot(
            name=param.name,
            state=SlotState.NEEDS_CONFIRMATION,
            origin=Origin.LLM,
            suggestions=tuple(chosen),
        )
    return Slot(name=param.name, state=SlotState.RESOLVED, value=chosen[0].value, origin=Origin.LLM)


def _validated_suggestions(
    param: ParameterSpec,
    raw: Any,
    on_invalid_slot: str,
) -> list[SlotSuggestion[Any]]:
    """応答の値をスロット単位で宣言型へ検証し、通った候補値だけを返す（FR-7 L222）。

    検証をフィールド単位で行うのは、型不一致を `on_invalid_slot` の管轄に留めるためである。
    複合モデル全体の 1 回の検証に任せると、1 スロットの型違いが応答全体の
    `ValidationError` になり `on_invalid_response` へ倒れてしまう。

    Args:
        param: 対象のパラメータ宣言。
        raw: 応答から取り出した値（`SlotSuggestion[Any]` かその list か `None`）。
        on_invalid_slot: 当該計画で解決済みの不正スロット時の挙動。

    Returns:
        宣言型に合った候補値のリスト（宣言順のまま・空なら後退対象）。

    Raises:
        ValueError: 値が欠落または宣言型に合わず `on_invalid_slot` が `"error"` の場合。
            文言にはパラメータ名を載せる（どの宣言を直すべきかが読み取れないため）。
    """
    items = raw if isinstance(raw, list) else [] if raw is None else [raw]
    if not items:
        if on_invalid_slot == "error":
            raise ValueError(f"the prediction response has no value for parameter {param.name!r}")
        return []
    adapter = TypeAdapter(param.annotation)
    accepted: list[SlotSuggestion[Any]] = []
    for item in items:
        try:
            value = adapter.validate_python(item.value)
        except ValidationError as exc:
            if on_invalid_slot == "error":
                raise ValueError(
                    f"the predicted value for parameter {param.name!r} does not match the "
                    f"declared type"
                ) from exc
            continue
        accepted.append(SlotSuggestion(value=value, level=item.level, rationale=item.rationale))
    return accepted


def _fallback_slot(param: ParameterSpec) -> Slot:
    """予測で埋まらなかったパラメータの後退先を決める（FR-7 L221 / L222）。

    Args:
        param: 対象のパラメータ宣言。

    Returns:
        `default` 宣言があれば `Origin.DEFAULT` の `Slot`（`confirm=True` なら
        `NEEDS_CONFIRMATION`）、無ければ `NEEDS_USER` の `Slot`。
    """
    if param.has_default:
        return _settle_slot(param, param.default, Origin.DEFAULT, None)
    return Slot(name=param.name, state=SlotState.NEEDS_USER)
