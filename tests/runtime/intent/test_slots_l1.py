"""L1: `runtime.intent.slots` のスロット表現と計画型 (`SlotState` / `Origin` / `Slot` /
`SlotSuggestion` / `ActionPlan` / `PlanResult` / `ParamUsage`) の純検証。

FR-5 の受け入れ基準（タスク 1-4 / 1-5）を pin する。`Slot` を 1 型へ畳んだことで構造的には
作れてしまう不整合な組み合わせを validator 6 条件（設計 §3.5 の表）が拒否すること、状態と
同値の導出プロパティ (`resolved`) を公開しないこと、`ActionPlan` が実行先エージェントの
実体を保持せず `model_dump()` に `spec` / `action_agent` / `resolved_*` と `run_context`
由来のキーを出さないことを対象とする。外部依存 (agents / openai) なし。
"""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from oai_agentspec.runtime.intent.actions import ActionSpec, param

# ruff の isort は存在しないモジュールを第一者と判定できないため、slots.py が未実装の
# 間だけ I001 を報告する（実装後は既存ファイルと同じ並びで警告なしになる）。
from oai_agentspec.runtime.intent.slots import (
    ActionPlan,
    Origin,
    ParamUsage,
    PlanResult,
    Slot,
    SlotState,
    SlotSuggestion,
)
from oai_agentspec.runtime.intent.types import (
    ConfidenceLevel,
    ExecutableSuggestion,
    IntentContext,
)

pytestmark = pytest.mark.unit


def _make_spec(**overrides: Any) -> ActionSpec:
    """テスト用の最小 ActionSpec を組む。"""
    fields: dict[str, Any] = {
        "action_id": "run_load_test",
        "description": "負荷試験を実行する",
        "action_agent": "load_test_runner",
        "label": "${target} に ${seconds} 秒の負荷試験",
        "parameters": (param("target", str), param("seconds", int)),
    }
    fields.update(overrides)
    return ActionSpec(**fields)


def _resolved(name: str, value: Any, origin: Origin = Origin.CANDIDATE) -> Slot:
    """解決済みスロットを組む。"""
    return Slot(name=name, state=SlotState.RESOLVED, value=value, origin=origin)


def _make_plan(**overrides: Any) -> ActionPlan:
    """テスト用の最小 ActionPlan を組む。"""
    spec = overrides.pop("spec", None) or _make_spec()
    fields: dict[str, Any] = {
        "action_id": spec.action_id,
        "slots": (_resolved("target", "api.example.com"), _resolved("seconds", 30)),
        "spec": spec,
        "action_agent": spec.action_agent,
        "resolved_prompt": (),
        "resolved_prompt_vars": {},
        "resolved_on_invalid_slot": "skip",
    }
    fields.update(overrides)
    return ActionPlan(**fields)


# ---------------------------------------------------------------------------
# SlotState / Origin の値と文字列契約 (FR-5 L184 / 設計 §3.5)
# ---------------------------------------------------------------------------


def test_slot_state_has_exactly_four_values() -> None:
    """SlotState は RESOLVED / NEEDS_LLM / NEEDS_CONFIRMATION / NEEDS_USER の 4 値 (FR-5 L184)。

    状態を増やすと利用者側の分岐が黙って抜けるため、集合そのものを固定する。
    """
    assert issubclass(SlotState, StrEnum)
    assert {member.name for member in SlotState} == {
        "RESOLVED",
        "NEEDS_LLM",
        "NEEDS_CONFIRMATION",
        "NEEDS_USER",
    }


def test_slot_state_values_are_prefixless_snake_case() -> None:
    """SlotState の値は接頭辞なし snake_case であり公開契約である (設計 §3.5)。"""
    assert SlotState.RESOLVED == "resolved"
    assert SlotState.NEEDS_LLM == "needs_llm"
    assert SlotState.NEEDS_CONFIRMATION == "needs_confirmation"
    assert SlotState.NEEDS_USER == "needs_user"


def test_origin_has_exactly_six_values() -> None:
    """Origin は 6 値の StrEnum である (FR-5 L184)。"""
    assert issubclass(Origin, StrEnum)
    assert {member.name for member in Origin} == {
        "CANDIDATE",
        "RUN_CONTEXT",
        "DEFAULT",
        "LLM",
        "USER_CONFIRMED",
        "USER_INPUT",
    }


def test_origin_values_are_prefixless_snake_case() -> None:
    """Origin の値は接頭辞なし snake_case で揃う（案 1 の "user:" 接頭辞を断つ）(設計 §3.5)。

    連結文字列の名残が残ると利用者が接頭辞を切り出すコードを書くことになる。
    """
    assert Origin.CANDIDATE == "candidate"
    assert Origin.RUN_CONTEXT == "run_context"
    assert Origin.DEFAULT == "default"
    assert Origin.LLM == "llm"
    assert Origin.USER_CONFIRMED == "user_confirmed"
    assert Origin.USER_INPUT == "user_input"


def test_slot_dump_json_carries_the_snake_case_strings() -> None:
    """model_dump(mode="json") に state / origin の snake_case 文字列が現れる (FR-5 L184)。

    JSON へ落とした後も 4 状態と 6 出どころを判別できることが公開契約である。
    """
    slot = Slot(
        name="target",
        state=SlotState.RESOLVED,
        value="api.example.com",
        origin=Origin.RUN_CONTEXT,
        detail="current_env.host",
    )
    dumped = slot.model_dump(mode="json")
    assert dumped["state"] == "resolved"
    assert dumped["origin"] == "run_context"
    assert dumped["detail"] == "current_env.host"


# ---------------------------------------------------------------------------
# SlotSuggestion (FR-1 L104)
# ---------------------------------------------------------------------------


def test_slot_suggestion_defaults_level_to_certain() -> None:
    """level 省略時は ConfidenceLevel.CERTAIN になる (FR-1 L104)。"""
    suggestion = SlotSuggestion(value=60)
    assert suggestion.level is ConfidenceLevel.CERTAIN
    assert suggestion.rationale is None


def test_slot_suggestion_is_frozen_and_serializable() -> None:
    """SlotSuggestion は frozen で直列化が成立する (FR-1 L103 / L105)。"""
    suggestion = SlotSuggestion(value=60, level=ConfidenceLevel.HIGH, rationale="直前の発言")
    assert isinstance(suggestion, BaseModel)
    with pytest.raises(ValidationError):
        suggestion.value = 30  # type: ignore[misc]
    assert suggestion.model_dump(mode="json")["level"] == "high"


def test_slot_rebuilds_suggestions_but_preserves_equality() -> None:
    """パラメータ化ジェネリック注釈のため検証で作り直されるが、等価性は保たれる。

    apply の USER_CONFIRMED 判別は値の等価性で行う契約（FR-8）であり、identity へ
    依存する実装を後続タスクが書かないよう挙動を固定する。
    """
    suggestion = SlotSuggestion(value=60)
    slot = Slot(name="seconds", state=SlotState.NEEDS_USER, suggestions=(suggestion,))
    assert slot.suggestions[0] is not suggestion
    assert slot.suggestions[0] == suggestion


# ---------------------------------------------------------------------------
# Slot の基本形 (FR-5 L180-L184 / 設計 §3.5)
# ---------------------------------------------------------------------------


def test_slot_is_frozen() -> None:
    """Slot は frozen な pydantic BaseModel である (FR-1 L103)。"""
    slot = _resolved("seconds", 30)
    assert isinstance(slot, BaseModel)
    with pytest.raises(ValidationError):
        slot.value = 60  # type: ignore[misc]


def test_slot_unresolved_fields_default_to_none_and_empty() -> None:
    """未解決スロットは name と state だけで成立する (設計 §3.5 の状態表)。"""
    slot = Slot(name="seconds", state=SlotState.NEEDS_LLM)
    assert slot.value is None
    assert slot.origin is None
    assert slot.detail is None
    assert slot.suggestions == ()


def test_slot_is_serializable() -> None:
    """Slot は model_dump / model_json_schema が成立する (FR-1 L105)。"""
    slot = _resolved("seconds", 30)
    assert isinstance(slot.model_dump(), dict)
    assert isinstance(slot.model_json_schema(), dict)


def test_slot_declares_exactly_six_fields() -> None:
    """Slot のフィールドは name / state / value / origin / detail / suggestions の 6 件。

    設計 §3.5b の公開面（6 フィールド + 導出 `from_user` の 7 件）を pin する。
    """
    assert set(Slot.model_fields) == {
        "name",
        "state",
        "value",
        "origin",
        "detail",
        "suggestions",
    }
    assert isinstance(_resolved("seconds", 30).from_user, bool)


def test_slot_does_not_expose_resolved() -> None:
    """`resolved` は公開しない（`state is RESOLVED` と同値のため削除済み）(設計 §3.5b)。

    状態を 2 通りで表現すると、片方だけを見る利用者コードが生まれる。
    """
    slot = _resolved("seconds", 30)
    assert not hasattr(slot, "resolved")
    assert "resolved" not in Slot.model_fields


# ---------------------------------------------------------------------------
# Slot.from_user (FR-5 L185 / 設計 §3.5a)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("origin", [Origin.USER_INPUT, Origin.USER_CONFIRMED])
def test_slot_from_user_is_true_for_user_origins(origin: Origin) -> None:
    """origin が USER_INPUT / USER_CONFIRMED のときだけ from_user は真 (FR-5 L185)。

    アプリが「自分のストアへ書き戻すべき値」を選り分ける唯一の判定である。
    """
    assert _resolved("seconds", 60, origin=origin).from_user is True


@pytest.mark.parametrize("origin", [Origin.CANDIDATE, Origin.DEFAULT, Origin.LLM])
def test_slot_from_user_is_false_for_non_user_origins(origin: Origin) -> None:
    """候補 / 既定値 / LLM 予測の値は from_user が偽である (設計 §3.5a の表)。"""
    assert _resolved("seconds", 60, origin=origin).from_user is False


def test_slot_from_user_is_false_for_run_context_origin() -> None:
    """run context から読んだ値は from_user が偽である（既にアプリが持つ値）(設計 §3.5a)。"""
    slot = Slot(
        name="target",
        state=SlotState.RESOLVED,
        value="api.example.com",
        origin=Origin.RUN_CONTEXT,
        detail="current_env.host",
    )
    assert slot.from_user is False


@pytest.mark.parametrize(
    "state",
    [SlotState.NEEDS_LLM, SlotState.NEEDS_USER],
)
def test_slot_from_user_is_false_when_origin_is_none(state: SlotState) -> None:
    """origin が None のスロットでも from_user は例外を出さず偽を返す (FR-5 L185)。

    `s.origin.startswith(...)` 形の判定が `AttributeError` になる事故を構造的に消す。
    """
    assert Slot(name="seconds", state=state).from_user is False


# ---------------------------------------------------------------------------
# Slot の validator 6 条件 (FR-5 L186 / 設計 §3.5 の表・第 4 回レビューの 9 通り)
# ---------------------------------------------------------------------------
#
# 案 1 の 4 クラスでは `NeedsAgent` に origin フィールドが存在しなかったため、不整合な
# 組み合わせは構造的に作れなかった。1 型化した `Slot` は全フィールドを持つため、
# 6 条件を validator で強制しないと FR-5 の各契約が型でも実行時でも検査されない。

_A_SUGGESTION = SlotSuggestion(value=60, level=ConfidenceLevel.HIGH)

#: 第 4 回レビューが列挙した 9 通り。意図的に通す 1 通り以外はすべて ValidationError。
_SLOT_COMBINATIONS: list[tuple[str, dict[str, Any], bool]] = [
    (
        "needs-llm-with-origin",
        {"name": "seconds", "state": SlotState.NEEDS_LLM, "origin": Origin.LLM},
        False,
    ),
    (
        "needs-llm-with-value",
        {"name": "seconds", "state": SlotState.NEEDS_LLM, "value": 60},
        False,
    ),
    (
        "needs-llm-with-suggestions",
        {"name": "seconds", "state": SlotState.NEEDS_LLM, "suggestions": (_A_SUGGESTION,)},
        False,
    ),
    (
        "needs-user-with-origin",
        {"name": "seconds", "state": SlotState.NEEDS_USER, "origin": Origin.USER_INPUT},
        False,
    ),
    (
        "needs-user-with-value",
        {"name": "seconds", "state": SlotState.NEEDS_USER, "value": 60},
        False,
    ),
    (
        "needs-confirmation-with-value",
        {
            "name": "seconds",
            "state": SlotState.NEEDS_CONFIRMATION,
            "value": 60,
            "origin": Origin.LLM,
            "suggestions": (_A_SUGGESTION,),
        },
        False,
    ),
    (
        "needs-confirmation-without-suggestions",
        {
            "name": "seconds",
            "state": SlotState.NEEDS_CONFIRMATION,
            "origin": Origin.LLM,
            "suggestions": (),
        },
        False,
    ),
    (
        "resolved-without-origin",
        {"name": "seconds", "state": SlotState.RESOLVED, "value": 60},
        False,
    ),
    (
        "resolved-with-none-value",
        {"name": "seconds", "state": SlotState.RESOLVED, "value": None, "origin": Origin.DEFAULT},
        True,
    ),
]


@pytest.mark.parametrize(
    ("fields", "is_valid"),
    [
        pytest.param(fields, is_valid, id=case_id)
        for case_id, fields, is_valid in _SLOT_COMBINATIONS
    ],
)
def test_slot_validator_accepts_only_the_intentional_combination(
    fields: dict[str, Any], is_valid: bool
) -> None:
    """9 通りのうち 8 通りは ValidationError、意図的に通す 1 通りのみ成立する (FR-5 L186 / L187)。

    RESOLVED + value=None は `param(..., default=None)` を明示宣言したときに正当に発生する
    組み合わせであり、value の None は「未解決」ではなく「値が None であること」を意味する。
    """
    if is_valid:
        slot = Slot(**fields)
        assert slot.state is SlotState.RESOLVED
        assert slot.value is None
    else:
        with pytest.raises(ValidationError):
            Slot(**fields)


def test_slot_rejects_resolved_without_origin_because_from_user_would_silently_be_false() -> None:
    """RESOLVED + origin=None は拒否する（条件 5）(FR-5 L186)。

    通してしまうと from_user が常に偽になり、利用者が決めた値が書き戻し対象から黙って漏れる。
    宣言が `origin: Origin | None = None` である以上、既定の型検証では落ちない。
    """
    with pytest.raises(ValidationError):
        Slot(name="seconds", state=SlotState.RESOLVED, value=60)


def test_slot_rejects_needs_confirmation_without_suggestions() -> None:
    """NEEDS_CONFIRMATION + suggestions=() は拒否する（条件 4）(FR-5 L186)。

    通してしまうと UI に選択肢が出ず、さらに apply が「値が suggestions のいずれかと等しいか」
    で判別するため USER_CONFIRMED を判定できず USER_INPUT へ誤分類する。
    """
    with pytest.raises(ValidationError):
        Slot(name="seconds", state=SlotState.NEEDS_CONFIRMATION, origin=Origin.LLM)


@pytest.mark.parametrize(
    "origin",
    [Origin.CANDIDATE, Origin.DEFAULT, Origin.LLM, Origin.USER_CONFIRMED, Origin.USER_INPUT],
)
def test_slot_rejects_detail_when_origin_is_not_run_context(origin: Origin) -> None:
    """detail が非 None なら origin は RUN_CONTEXT でなければならない（条件 6）(FR-5 L186)。

    detail は run context の解決パスを入れるフィールドであり、Origin.LLM の detail は常に
    None である（予測エージェント名は公開契約に載せない）。
    """
    with pytest.raises(ValidationError):
        Slot(
            name="seconds",
            state=SlotState.RESOLVED,
            value=60,
            origin=origin,
            detail="current_env.host",
        )


def test_slot_accepts_detail_with_run_context_origin() -> None:
    """RUN_CONTEXT の detail には解決に成功したパスが入る (FR-5 L184)。"""
    slot = Slot(
        name="target",
        state=SlotState.RESOLVED,
        value="api.example.com",
        origin=Origin.RUN_CONTEXT,
        detail="current_env.host",
    )
    assert slot.detail == "current_env.host"


def test_slot_accepts_needs_confirmation_with_suggestions_and_origin() -> None:
    """NEEDS_CONFIRMATION は suggestions 1 件以上 + origin を持ち value は None (設計 §3.5)。"""
    slot = Slot(
        name="seconds",
        state=SlotState.NEEDS_CONFIRMATION,
        origin=Origin.LLM,
        suggestions=(_A_SUGGESTION,),
    )
    assert slot.value is None
    assert slot.suggestions == (_A_SUGGESTION,)


def test_slot_accepts_needs_user_with_suggestions() -> None:
    """NEEDS_USER は suggestions を持てる（既定は空 tuple）(設計 §3.5 の状態表)。"""
    slot = Slot(name="seconds", state=SlotState.NEEDS_USER, suggestions=(_A_SUGGESTION,))
    assert slot.suggestions == (_A_SUGGESTION,)
    assert slot.origin is None
    assert slot.value is None


# ---------------------------------------------------------------------------
# ActionPlan の保持と直列化除外 (FR-5 L178 / L189 / NFR-6)
# ---------------------------------------------------------------------------


def test_action_plan_is_frozen() -> None:
    """ActionPlan は frozen な pydantic BaseModel である (FR-1 L103)。"""
    plan = _make_plan()
    assert isinstance(plan, BaseModel)
    with pytest.raises(ValidationError):
        plan.action_id = "other"  # type: ignore[misc]


def test_action_plan_holds_the_spec_and_resolved_defaults() -> None:
    """spec / action_agent / resolved_* は属性として読める (FR-5 L178)。"""
    spec = _make_spec()
    plan = _make_plan(
        spec=spec,
        resolved_prompt=("intent/common", "actions/load_test"),
        resolved_prompt_vars={"host": "current_env.host"},
        resolved_on_invalid_slot="error",
    )
    assert plan.spec is spec
    assert plan.action_agent == "load_test_runner"
    assert plan.resolved_prompt == ("intent/common", "actions/load_test")
    assert plan.resolved_prompt_vars == {"host": "current_env.host"}
    assert plan.resolved_on_invalid_slot == "error"


def test_action_plan_dump_contains_only_action_id_and_slots() -> None:
    """model_dump() は action_id と slots のみを含む (FR-5 L189 / NFR-6)。

    spec / action_agent / resolved_* の 5 件は Field(exclude=True) であり、run_context 由来の
    キーは ActionPlan がそもそも保持しない。
    """
    plan = _make_plan()
    dumped = plan.model_dump()
    assert set(dumped) == {"action_id", "slots"}


@pytest.mark.parametrize(
    "excluded",
    [
        "spec",
        "action_agent",
        "resolved_prompt",
        "resolved_prompt_vars",
        "resolved_on_invalid_slot",
    ],
)
def test_action_plan_dump_excludes_the_five_fields(excluded: str) -> None:
    """5 件の Field(exclude=True) は model_dump() / model_dump_json() に現れない (FR-5 L178)。"""
    plan = _make_plan()
    assert excluded not in plan.model_dump()
    assert excluded not in plan.model_dump_json()


def test_action_plan_dump_does_not_leak_run_context() -> None:
    """model_dump() に run_context / IntentContext 由来のキーが現れない (FR-5 L189 / NFR-6)。

    ActionPlan は run_context を保持しない（露出するのは ExecutableSuggestion 側のみ）。
    """
    plan = _make_plan()
    dumped_json = plan.model_dump_json()
    assert "run_context" not in ActionPlan.model_fields
    assert "context" not in ActionPlan.model_fields
    assert "run_context" not in dumped_json


def test_action_plan_json_schema_still_lists_the_excluded_fields() -> None:
    """Field(exclude=True) は model_json_schema() の properties には残る (設計 §7.2 / 実測 7-4)。

    LLM へ提示するスキーマを ActionPlan から作ると spec が漏れるため、予測段は専用の
    スキーマモデルから作る。この性質を pin して誤用を検出できるようにする。
    """
    properties = ActionPlan.model_json_schema()["properties"]
    assert "spec" in properties


# ---------------------------------------------------------------------------
# ActionPlan は実行先エージェントの実体を保持しない (FR-8 L238 / 設計 §3.4d)
# ---------------------------------------------------------------------------


def test_action_plan_does_not_expose_agent() -> None:
    """`agent` フィールド・プロパティを持たない（registry の選択を隠さない）(FR-8 L238)。

    実行は利用者が `Runner.run(registry.get(plan.action_agent), ...)` と明示的に書く。
    どの registry から解決したかが呼び出し箇所に現れることで、派生 registry の
    取り違えが隠れない。
    """
    plan = _make_plan()
    assert not hasattr(plan, "agent")
    assert "agent" not in ActionPlan.model_fields


def test_action_plan_does_not_expose_a_registry() -> None:
    """AgentRegistry を参照するフィールド・プロパティを一切持たない (FR-8 L238)。"""
    plan = _make_plan()
    assert not hasattr(plan, "registry")
    assert "registry" not in ActionPlan.model_fields


def test_action_plan_action_agent_is_a_plain_name() -> None:
    """action_agent は ActionSpec と同じ実行先エージェント名の str である (FR-8 L237)。"""
    plan = _make_plan()
    assert plan.action_agent == plan.spec.action_agent
    assert isinstance(plan.action_agent, str)


# ---------------------------------------------------------------------------
# ActionPlan の導出 (FR-5 L188 / 設計 §3.5b)
# ---------------------------------------------------------------------------


def test_action_plan_pending_lists_confirmation_and_user_slots_in_declaration_order() -> None:
    """pending は NEEDS_CONFIRMATION と NEEDS_USER を宣言順に並べた tuple (FR-5 L188)。

    NEEDS_USER を先に宣言する。SlotState の値で並べると
    "needs_confirmation" < "needs_user" となり逆順になるため、状態でソートする実装を弾ける。
    """
    needs_user = Slot(name="region", state=SlotState.NEEDS_USER)
    needs_confirmation = Slot(
        name="seconds",
        state=SlotState.NEEDS_CONFIRMATION,
        origin=Origin.CANDIDATE,
        suggestions=(_A_SUGGESTION,),
    )
    plan = _make_plan(
        slots=(
            _resolved("target", "api.example.com"),
            needs_user,
            needs_confirmation,
        )
    )
    assert plan.pending == (needs_user, needs_confirmation)


def test_action_plan_pending_excludes_resolved_and_needs_llm() -> None:
    """RESOLVED と NEEDS_LLM は pending に入らない（利用者へ聞く対象ではない）(FR-5 L188)。"""
    plan = _make_plan(
        slots=(
            _resolved("target", "api.example.com"),
            Slot(name="seconds", state=SlotState.NEEDS_LLM),
        )
    )
    assert plan.pending == ()


def test_action_plan_ready_is_true_when_all_slots_are_resolved() -> None:
    """ready は全スロットが RESOLVED のときだけ真 (FR-5 L188)。"""
    assert _make_plan().ready is True


@pytest.mark.parametrize(
    "unresolved",
    [
        pytest.param(Slot(name="seconds", state=SlotState.NEEDS_LLM), id="needs-llm"),
        pytest.param(Slot(name="seconds", state=SlotState.NEEDS_USER), id="needs-user"),
        pytest.param(
            Slot(
                name="seconds",
                state=SlotState.NEEDS_CONFIRMATION,
                origin=Origin.CANDIDATE,
                suggestions=(_A_SUGGESTION,),
            ),
            id="needs-confirmation",
        ),
    ],
)
def test_action_plan_ready_is_false_when_any_slot_is_unresolved(unresolved: Slot) -> None:
    """RESOLVED 以外のスロットが 1 件でもあれば ready は偽 (FR-5 L188)。"""
    plan = _make_plan(slots=(_resolved("target", "api.example.com"), unresolved))
    assert plan.ready is False


@pytest.mark.parametrize("dropped", ["needs_agent", "needs_user"])
def test_action_plan_does_not_expose_the_dropped_derivations(dropped: str) -> None:
    """needs_agent / needs_user は公開しない（slots から一意に書ける）(設計 §3.5b)。

    `any(s.state is SlotState.NEEDS_LLM for s in plan.slots)` / `bool(plan.pending)` と
    同値であり、SlotState は公開されているため利用者が書ける。
    """
    plan = _make_plan()
    assert not hasattr(plan, dropped)


def test_action_plan_does_not_expose_parameters() -> None:
    """parameters は公開しない（input_json と同じデータの別表現）(設計 §3.5b)。"""
    plan = _make_plan()
    assert not hasattr(plan, "parameters")


def test_action_plan_label_substitutes_resolved_values() -> None:
    """label は ${name} を解決済みスロットの値で render する (FR-5 L190)。"""
    plan = _make_plan()
    assert plan.label == "api.example.com に 30 秒の負荷試験"


def test_action_plan_label_renders_unresolved_slots_as_ellipsis() -> None:
    """未解決スロットは "…" として render される (FR-5 L190)。"""
    plan = _make_plan(
        slots=(
            _resolved("target", "api.example.com"),
            Slot(name="seconds", state=SlotState.NEEDS_LLM),
        )
    )
    assert plan.label == "api.example.com に … 秒の負荷試験"


def test_action_plan_label_raises_key_error_for_an_unknown_placeholder() -> None:
    """パラメータ名でないプレースホルダは `KeyError` になる (FR-5 L190 / 設計 §3.5b)。

    render は `string.Template.substitute` で行う。`safe_substitute` に差し替えた実装は
    `${unknown}` を綴りのまま UI へ出し、宣言の取りこぼしを黙って通す。プレースホルダと
    パラメータ名の対応は起動時検証（`planner.validate()`）の担当であり、そこを抜けた
    宣言はここで落ちる契約である。
    """
    spec = _make_spec(label="${target} / ${unknown}", parameters=(param("target", str),))
    plan = _make_plan(spec=spec, slots=(_resolved("target", "api.example.com"),))
    with pytest.raises(KeyError):
        _ = plan.label


# ---------------------------------------------------------------------------
# ParamUsage / PlanResult の型定義（第 1 段で型だけ置く。値の詰め替えは第 2 段）
# ---------------------------------------------------------------------------


def test_param_usage_holds_the_five_fields() -> None:
    """ParamUsage は runs / model_calls / candidates / input_tokens / output_tokens を持つ。"""
    usage = ParamUsage(runs=0, model_calls=0, candidates=0, input_tokens=None, output_tokens=None)
    assert usage.runs == 0
    assert usage.model_calls == 0
    assert usage.candidates == 0
    assert usage.input_tokens is None
    assert usage.output_tokens is None


def test_param_usage_accepts_measured_tokens() -> None:
    """usage を取得できたときは input_tokens / output_tokens に実測値が入る (FR-6 L209)。"""
    usage = ParamUsage(runs=1, model_calls=1, candidates=2, input_tokens=120, output_tokens=34)
    assert (usage.input_tokens, usage.output_tokens) == (120, 34)


def test_param_usage_is_frozen() -> None:
    """ParamUsage は frozen である (FR-1 L103 と同型の宣言型契約)。"""
    usage = ParamUsage(runs=0, model_calls=0, candidates=0, input_tokens=None, output_tokens=None)
    with pytest.raises(ValidationError):
        usage.runs = 1  # type: ignore[misc]


def test_plan_result_holds_plans_suggestion_and_usage() -> None:
    """PlanResult は plans / suggestion / usage を保持する (FR-4 L169 / FR-6 L196)。"""
    plan = _make_plan()
    usage = ParamUsage(runs=0, model_calls=0, candidates=0, input_tokens=None, output_tokens=None)
    suggestion = ExecutableSuggestion(
        candidates=(), context=IntentContext(utterance="負荷試験をしたい")
    )
    result = PlanResult(plans=(plan,), suggestion=suggestion, usage=usage)
    assert result.plans == (plan,)
    assert result.suggestion is suggestion
    assert result.usage is usage


# ---------------------------------------------------------------------------
# ActionPlan.resolved_prompt_vars も中身を書き換えられない (レビュー 2 巡目・指摘 #88-W2 系)
# ---------------------------------------------------------------------------
#
# `ActionCatalog.prompt_vars` / `ActionSpec.prompt_vars` / `ActionPlanner.prompt_vars` は
# 読み取り専用へ揃えたが、4 つ目のキャリアである `ActionPlan.resolved_prompt_vars` は
# `Mapping[str, str]` を素の dict として保持したままである。`resolve_prompt_vars()` が毎回
# 新しい dict を返し、frozen な `ActionPlan` は属性の再束縛しか禁じない。この値は起動時検証
# (検査 6 / 7) を通過した宣言をマージした結果であり、実際に LLM プロンプトへ展開される
# 直前の値そのものであるため、脅威モデル上もっとも遅い差し替え点になる。
# 併せて、読み取り専用化で複製・永続化が壊れないことを対で pin する
# (`ActionPlan` は Field(exclude=True) のフィールドを 5 件持つため、復元後もそれらの値が
# 保たれることを確かめる)。

#: 複製・永続化の 3 経路。どれか 1 つでも落ちれば計画をプロセス外へ運べない。
_CLONE_ROUTES: list[Any] = [
    pytest.param(copy.deepcopy, id="deepcopy"),
    pytest.param(lambda obj: obj.model_copy(deep=True), id="model-copy-deep"),
    pytest.param(lambda obj: pickle.loads(pickle.dumps(obj)), id="pickle"),
]


def test_action_plan_resolved_prompt_vars_rejects_item_assignment() -> None:
    """resolved_prompt_vars への要素追加は TypeError (指摘 #88-W2 系)。"""
    plan = _make_plan(resolved_prompt_vars={"host": "current_env.host"})
    with pytest.raises(TypeError) as excinfo:
        plan.resolved_prompt_vars["secret"] = "credentials.api_key"
    assert type(excinfo.value) is TypeError


def test_action_plan_resolved_prompt_vars_rejects_item_overwrite_and_deletion() -> None:
    """既存キーの上書きと削除も TypeError (指摘 #88-W2 系)。

    マージ済みの解決値をプロンプト展開の直前に別のパスへ向け直せる経路であるため、
    追加だけでなく上書き・削除も塞ぐ。
    """
    plan = _make_plan(resolved_prompt_vars={"host": "current_env.host"})
    with pytest.raises(TypeError) as overwrite:
        plan.resolved_prompt_vars["host"] = "credentials.api_key"
    assert type(overwrite.value) is TypeError
    with pytest.raises(TypeError) as deletion:
        del plan.resolved_prompt_vars["host"]
    assert type(deletion.value) is TypeError


def test_action_plan_resolved_prompt_vars_keeps_its_values_after_a_rejected_write() -> None:
    """書き込みが弾かれた後も resolved_prompt_vars は組み立て時のまま (指摘 #88-W2 系)。"""
    plan = _make_plan(resolved_prompt_vars={"host": "current_env.host"})
    with pytest.raises(TypeError):
        plan.resolved_prompt_vars["host"] = "credentials.api_key"
    with pytest.raises(TypeError):
        plan.resolved_prompt_vars["secret"] = "credentials.api_key"
    assert dict(plan.resolved_prompt_vars) == {"host": "current_env.host"}


def test_action_plan_copies_the_given_resolved_prompt_vars() -> None:
    """組み立て時に渡した dict を後から書き換えても中身が透けない (指摘 #88-W2 系)。"""
    source = {"host": "current_env.host"}
    plan = _make_plan(resolved_prompt_vars=source)
    source["host"] = "credentials.api_key"
    source["secret"] = "credentials.api_key"
    assert dict(plan.resolved_prompt_vars) == {"host": "current_env.host"}


def test_action_plan_resolved_prompt_vars_is_still_readable_as_a_mapping() -> None:
    """読み取り専用にしても Mapping としての読み取りは従来どおり (指摘 #88-W2 系の回帰防止)。"""
    plan = _make_plan(resolved_prompt_vars={"host": "current_env.host", "who": "profile.name"})
    assert plan.resolved_prompt_vars["host"] == "current_env.host"
    assert sorted(plan.resolved_prompt_vars) == ["host", "who"]
    assert len(plan.resolved_prompt_vars) == 2
    assert dict(plan.resolved_prompt_vars) == {
        "host": "current_env.host",
        "who": "profile.name",
    }


@pytest.mark.parametrize("clone", _CLONE_ROUTES)
def test_action_plan_survives_cloning_with_its_excluded_fields(clone: Callable[[Any], Any]) -> None:
    """ActionPlan は 3 経路で複製でき exclude=True の 5 件も保たれる (指摘 #88-W2 系の退行)。

    `Field(exclude=True)` は直列化から外すだけであり、複製・永続化では失われない。
    読み取り専用化で `mappingproxy` を持ち込むと、この 3 経路が `TypeError` で落ちる。
    """
    plan = _make_plan(
        resolved_prompt=("intent/common",),
        resolved_prompt_vars={"host": "current_env.host"},
        resolved_on_invalid_slot="error",
    )
    restored = clone(plan)
    assert restored.action_id == plan.action_id
    assert restored.spec.action_id == plan.spec.action_id
    assert restored.action_agent == plan.action_agent
    assert restored.resolved_prompt == ("intent/common",)
    assert dict(restored.resolved_prompt_vars) == {"host": "current_env.host"}
    assert restored.resolved_on_invalid_slot == "error"
    assert restored.resolved_prompt_vars is not plan.resolved_prompt_vars


@pytest.mark.parametrize("clone", _CLONE_ROUTES)
def test_action_plan_resolved_prompt_vars_stay_read_only_after_cloning(
    clone: Callable[[Any], Any],
) -> None:
    """複製後の resolved_prompt_vars も読み取り専用のまま (指摘 #88-W2 系の退行)。

    複製を通すために素の dict へ戻す修正だと値の一致だけは緑になるため、書き込みが
    `TypeError` であることを複製の先でも確かめる。
    """
    plan = _make_plan(resolved_prompt_vars={"host": "current_env.host"})
    restored = clone(plan)
    with pytest.raises(TypeError) as excinfo:
        restored.resolved_prompt_vars["secret"] = "credentials.api_key"
    assert type(excinfo.value) is TypeError
    assert dict(restored.resolved_prompt_vars) == {"host": "current_env.host"}
