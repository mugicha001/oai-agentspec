"""L1: `plan.apply(answers)` / `plan.input_json` / `plan.slots`（タスク 1-8・FR-8）の純検証。

利用者の確認結果と穴埋め入力を計画へ合流させ、型検証済みの実行入力を取り出す契約を pin
する。対象は「`apply` が新しい `ActionPlan` を返し元を変更しないこと」「`USER_CONFIRMED` /
`USER_INPUT` の判別が **`suggestions` の値との等価（`==`）**で行われること」「未知キー・既
`RESOLVED` キー・型不一致の拒否」「`ready=False` での `input_json` 参照が未解決スロット名を
列挙した `ValueError` になること」「`input_json` が `parameters_model()` で型検証した JSON
文字列であること」「`plan.slots` が宣言順の公開読み取り経路であること」。

`ActionPlan` はテスト側で直接組み立てるため、決定的段（タスク 1-6）の実装に依存しない。
`suggestions` の突き合わせは `is` / `id()` を使わない（`tuple[SlotSuggestion[Any], ...]` は
検証で作り直され identity が保存されない = バッチ 1 の申し送り）。
外部依存 (agents / openai) なし。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from oai_agentspec.runtime.intent.actions import ActionSpec, param
from oai_agentspec.runtime.intent.slots import (
    ActionPlan,
    Origin,
    Slot,
    SlotState,
    SlotSuggestion,
)
from oai_agentspec.runtime.intent.types import ConfidenceLevel

pytestmark = pytest.mark.unit


def _spec(*parameters: Any, label: str = "${target} / ${seconds}") -> ActionSpec:
    """テスト用の ActionSpec を組む。"""
    return ActionSpec(
        action_id="run_load_test",
        description="負荷試験を実行する",
        action_agent="load_test_runner",
        label=label,
        parameters=parameters or (param("target", str), param("seconds", int)),
    )


def _plan(spec: ActionSpec, slots: tuple[Slot, ...], **overrides: Any) -> ActionPlan:
    """テスト用の ActionPlan を組む。"""
    fields: dict[str, Any] = {
        "action_id": spec.action_id,
        "slots": slots,
        "spec": spec,
        "action_agent": spec.action_agent,
        "resolved_prompt": (),
        "resolved_prompt_vars": {},
        "resolved_on_invalid_slot": "skip",
    }
    fields.update(overrides)
    return ActionPlan(**fields)


def _resolved(name: str, value: Any, origin: Origin = Origin.CANDIDATE) -> Slot:
    """`RESOLVED` のスロットを組む。"""
    return Slot(name=name, state=SlotState.RESOLVED, value=value, origin=origin)


def _needs_user(name: str) -> Slot:
    """`NEEDS_USER` のスロットを組む。"""
    return Slot(name=name, state=SlotState.NEEDS_USER)


def _needs_confirmation(name: str, *values: Any, origin: Origin = Origin.CANDIDATE) -> Slot:
    """`NEEDS_CONFIRMATION` のスロットを組む。"""
    return Slot(
        name=name,
        state=SlotState.NEEDS_CONFIRMATION,
        origin=origin,
        suggestions=tuple(SlotSuggestion(value=value) for value in values),
    )


def _slot(plan: ActionPlan, name: str) -> Slot:
    """名前でスロットを引く。"""
    return next(slot for slot in plan.slots if slot.name == name)


def _pending_plan() -> ActionPlan:
    """`target` が解決済み・`seconds` が `NEEDS_USER` の計画を組む。"""
    spec = _spec()
    return _plan(spec, (_resolved("target", "api.example.com"), _needs_user("seconds")))


# ---------------------------------------------------------------------------
# apply は新しい ActionPlan を返す (FR-8 L230)
# ---------------------------------------------------------------------------


def test_apply_returns_a_new_action_plan() -> None:
    """`apply` は新しい `ActionPlan` を返す (FR-8 L230)。"""
    plan = _pending_plan()
    applied = plan.apply({"seconds": 30})
    assert isinstance(applied, ActionPlan)
    assert applied is not plan


def test_apply_does_not_mutate_the_original_plan() -> None:
    """元のインスタンスを変更しない (FR-8 L230)。

    UI が「押す前の計画」を握ったまま `apply` を呼ぶため、元が書き換わると
    再描画の基準が消える。
    """
    plan = _pending_plan()
    plan.apply({"seconds": 30})
    assert _slot(plan, "seconds").state is SlotState.NEEDS_USER
    assert _slot(plan, "seconds").value is None
    assert not plan.ready


def test_apply_transitions_the_answered_slot_to_resolved() -> None:
    """`answers` のキーに対応するスロットが `RESOLVED` へ遷移する (FR-8 L230)。"""
    applied = _pending_plan().apply({"seconds": 30})
    slot = _slot(applied, "seconds")
    assert slot.state is SlotState.RESOLVED
    assert slot.value == 30
    assert applied.ready


def test_apply_leaves_other_slots_untouched() -> None:
    """`answers` に無いスロットはそのまま引き継がれる (FR-8 L230)。"""
    plan = _pending_plan()
    applied = plan.apply({"seconds": 30})
    assert _slot(applied, "target") == _slot(plan, "target")


def test_apply_keeps_declaration_order() -> None:
    """`apply` 後もスロットは宣言順を保つ (FR-8 L240 / 設計 §3.5a)。"""
    applied = _pending_plan().apply({"seconds": 30})
    assert tuple(slot.name for slot in applied.slots) == ("target", "seconds")


def test_apply_keeps_the_resolved_defaults() -> None:
    """`apply` 後も `spec` / `action_agent` / `resolved_*` が引き継がれる (FR-5 L178)。"""
    spec = _spec()
    plan = _plan(
        spec,
        (_resolved("target", "api.example.com"), _needs_user("seconds")),
        resolved_prompt=("seg",),
        resolved_prompt_vars={"host": "current_env.host"},
        resolved_on_invalid_slot="error",
    )
    applied = plan.apply({"seconds": 30})
    assert applied.spec == spec
    assert applied.action_agent == "load_test_runner"
    assert applied.resolved_prompt == ("seg",)
    assert dict(applied.resolved_prompt_vars) == {"host": "current_env.host"}
    assert applied.resolved_on_invalid_slot == "error"


def test_apply_accepts_empty_answers() -> None:
    """空の `answers` は何も変えずに新しい計画を返す (FR-8 L230)。"""
    plan = _pending_plan()
    applied = plan.apply({})
    assert applied is not plan
    assert applied.slots == plan.slots


def test_apply_can_answer_several_slots_at_once() -> None:
    """複数スロットを 1 回の `apply` で埋められる (FR-8 L230)。"""
    spec = _spec()
    plan = _plan(spec, (_needs_user("target"), _needs_user("seconds")))
    applied = plan.apply({"target": "api.example.com", "seconds": 30})
    assert applied.ready


def test_apply_can_be_chained() -> None:
    """`apply` を続けて呼べる（1 回目の結果が 2 回目の入力になる）(FR-8 L230)。"""
    spec = _spec()
    plan = _plan(spec, (_needs_user("target"), _needs_user("seconds")))
    applied = plan.apply({"target": "api.example.com"}).apply({"seconds": 30})
    assert applied.ready
    assert _slot(applied, "target").value == "api.example.com"


# ---------------------------------------------------------------------------
# USER_CONFIRMED / USER_INPUT の判別 (FR-8 L231)
# ---------------------------------------------------------------------------


def test_answer_matching_a_suggestion_is_user_confirmed() -> None:
    """`NEEDS_CONFIRMATION` で `suggestions` の値と等しければ `USER_CONFIRMED` (FR-8 L231)。"""
    spec = _spec()
    plan = _plan(spec, (_resolved("target", "api"), _needs_confirmation("seconds", 30)))
    slot = _slot(plan.apply({"seconds": 30}), "seconds")
    assert slot.state is SlotState.RESOLVED
    assert slot.origin is Origin.USER_CONFIRMED
    assert slot.from_user


def test_confirmation_match_uses_equality_not_identity() -> None:
    """突き合わせは `==` で行う（`is` / `id()` を使わない）(FR-8 L231)。

    `tuple[SlotSuggestion[Any], ...]` はパラメータ化ジェネリック注釈のため検証で作り直され、
    identity が保存されない（等価性は保たれる）。identity で判定する実装は L1 では緑に
    見えても、値が別オブジェクトとして届く統合段で `USER_INPUT` へ誤分類する。
    """
    spec = _spec(param("target", str), param("region", str), label="${target} ${region}")
    plan = _plan(spec, (_resolved("target", "api"), _needs_confirmation("region", "tokyo")))
    answer = "".join(["to", "kyo"])
    assert answer is not plan.slots[1].suggestions[0].value
    assert answer == plan.slots[1].suggestions[0].value
    assert _slot(plan.apply({"region": answer}), "region").origin is Origin.USER_CONFIRMED


def test_answer_matching_any_suggestion_is_user_confirmed() -> None:
    """複数候補のいずれかと等しければ `USER_CONFIRMED` (FR-8 L231)。"""
    spec = _spec()
    plan = _plan(spec, (_resolved("target", "api"), _needs_confirmation("seconds", 30, 60, 90)))
    assert _slot(plan.apply({"seconds": 60}), "seconds").origin is Origin.USER_CONFIRMED


def test_answer_not_in_suggestions_is_user_input() -> None:
    """`suggestions` に無い値は `USER_INPUT` (FR-8 L231)。"""
    spec = _spec()
    plan = _plan(spec, (_resolved("target", "api"), _needs_confirmation("seconds", 30)))
    slot = _slot(plan.apply({"seconds": 45}), "seconds")
    assert slot.origin is Origin.USER_INPUT
    assert slot.from_user


def test_needs_user_answer_is_user_input() -> None:
    """`NEEDS_USER` のスロットへの入力は `USER_INPUT` (FR-8 L231)。"""
    slot = _slot(_pending_plan().apply({"seconds": 30}), "seconds")
    assert slot.origin is Origin.USER_INPUT


def test_needs_llm_answer_is_user_input() -> None:
    """`NEEDS_LLM` のスロットへの入力も `USER_INPUT` (FR-8 L231)。

    予測が走らなかった構成（`llm_filler` 未結線）でも利用者が直接埋められる。
    """
    spec = _spec()
    plan = _plan(
        spec,
        (_resolved("target", "api"), Slot(name="seconds", state=SlotState.NEEDS_LLM)),
    )
    slot = _slot(plan.apply({"seconds": 30}), "seconds")
    assert slot.state is SlotState.RESOLVED
    assert slot.origin is Origin.USER_INPUT


def test_applied_slot_has_no_detail() -> None:
    """利用者由来のスロットは `detail` を持たない (FR-5 L184 の validator 条件 6)。"""
    assert _slot(_pending_plan().apply({"seconds": 30}), "seconds").detail is None


def test_confirmation_with_non_default_level_still_matches_by_value() -> None:
    """突き合わせは `SlotSuggestion.value` で行い `level` を要求しない (FR-8 L231)。"""
    spec = _spec()
    slot = Slot(
        name="seconds",
        state=SlotState.NEEDS_CONFIRMATION,
        origin=Origin.LLM,
        suggestions=(SlotSuggestion(value=30, level=ConfidenceLevel.LOW),),
    )
    plan = _plan(spec, (_resolved("target", "api"), slot))
    assert _slot(plan.apply({"seconds": 30}), "seconds").origin is Origin.USER_CONFIRMED


# ---------------------------------------------------------------------------
# 拒否する入力 (FR-8 L232-L234)
# ---------------------------------------------------------------------------


def test_unknown_answer_key_raises_value_error() -> None:
    """宣言済みパラメータ名でないキーは `ValueError` (FR-8 L232)。"""
    with pytest.raises(ValueError) as excinfo:
        _pending_plan().apply({"secnods": 30})
    assert "secnods" in str(excinfo.value)


def test_unknown_answer_keys_are_all_listed() -> None:
    """未知キーは全件列挙される (FR-8 L232)。"""
    with pytest.raises(ValueError) as excinfo:
        _pending_plan().apply({"secnods": 30, "targt": "api"})
    message = str(excinfo.value)
    assert "secnods" in message
    assert "targt" in message


def test_answer_for_a_resolved_slot_raises_value_error() -> None:
    """既に `RESOLVED` のスロットを指すキーは `ValueError` (FR-8 L233)。

    確定済み値の黙示的な上書きを禁止する。
    """
    with pytest.raises(ValueError) as excinfo:
        _pending_plan().apply({"target": "evil.example.com"})
    assert "target" in str(excinfo.value)


def test_resolved_overwrite_keys_are_all_listed() -> None:
    """既 `RESOLVED` キーは全件列挙される (FR-8 L233)。"""
    spec = _spec()
    plan = _plan(spec, (_resolved("target", "api"), _resolved("seconds", 30)))
    with pytest.raises(ValueError) as excinfo:
        plan.apply({"target": "other", "seconds": 60})
    message = str(excinfo.value)
    assert "target" in message
    assert "seconds" in message


def test_resolved_overwrite_is_rejected_before_any_change() -> None:
    """拒否したときは元の計画が変わらない (FR-8 L230 / L233)。"""
    plan = _pending_plan()
    with pytest.raises(ValueError):
        plan.apply({"target": "other", "seconds": 30})
    assert _slot(plan, "seconds").state is SlotState.NEEDS_USER


def test_type_mismatch_raises_validation_error() -> None:
    """宣言型に合わない値は `ValidationError` (FR-8 L234)。

    未解決スロットが残る段では全件モデルによる検証を行えないため、当該パラメータ単体を
    `TypeAdapter(<param の annotation>)` で検証する。
    """
    with pytest.raises(ValidationError):
        _pending_plan().apply({"seconds": "not-a-number"})


def test_type_mismatch_is_a_validation_error_not_a_plain_value_error() -> None:
    """型不一致は未知キー・上書きの `ValueError` と区別できる (FR-8 L232-L234)。"""
    with pytest.raises(ValidationError) as excinfo:
        _pending_plan().apply({"seconds": []})
    assert isinstance(excinfo.value, ValidationError)


def test_valid_value_of_the_declared_type_is_accepted() -> None:
    """宣言型に合う値はそのまま受け付ける (FR-8 L234)。"""
    assert _slot(_pending_plan().apply({"seconds": 30}), "seconds").value == 30


def test_type_check_does_not_require_the_whole_model_to_be_complete() -> None:
    """未解決スロットが残っていても単体検証で通る (FR-8 L234)。

    全件モデルで検証する実装だと、残りの未解決スロットが必須フィールドとして落ちる。
    """
    spec = _spec()
    plan = _plan(spec, (_needs_user("target"), _needs_user("seconds")))
    applied = plan.apply({"seconds": 30})
    assert _slot(applied, "seconds").state is SlotState.RESOLVED
    assert _slot(applied, "target").state is SlotState.NEEDS_USER


# ---------------------------------------------------------------------------
# input_json (FR-8 L235 / L236)
# ---------------------------------------------------------------------------


def test_input_json_raises_when_the_plan_is_not_ready() -> None:
    """`ready=False` で参照すると `ValueError` (FR-8 L235)。"""
    with pytest.raises(ValueError):
        _ = _pending_plan().input_json


def test_input_json_error_lists_unresolved_slot_names() -> None:
    """`ValueError` に未解決スロット名が列挙される (FR-8 L235)。"""
    spec = _spec()
    plan = _plan(spec, (_needs_user("target"), Slot(name="seconds", state=SlotState.NEEDS_LLM)))
    with pytest.raises(ValueError) as excinfo:
        _ = plan.input_json
    message = str(excinfo.value)
    assert "target" in message
    assert "seconds" in message


def test_input_json_error_omits_resolved_slot_names() -> None:
    """解決済みスロットは未解決の列挙に現れない (FR-8 L235)。"""
    with pytest.raises(ValueError) as excinfo:
        _ = _pending_plan().input_json
    assert "seconds" in str(excinfo.value)
    assert "target" not in str(excinfo.value)


def test_input_json_returns_a_json_string() -> None:
    """`ready=True` なら `str` の JSON を返す (FR-8 L236)。"""
    plan = _pending_plan().apply({"seconds": 30})
    assert isinstance(plan.input_json, str)
    assert json.loads(plan.input_json) == {"target": "api.example.com", "seconds": 30}


def test_input_json_keeps_declaration_order() -> None:
    """JSON のキーは宣言順に並ぶ (FR-8 L236 / 設計 §3.5a)。"""
    plan = _pending_plan().apply({"seconds": 30})
    assert list(json.loads(plan.input_json)) == ["target", "seconds"]


def test_input_json_is_type_validated_by_parameters_model() -> None:
    """全スロットの値を `spec.parameters_model()` で型検証する (FR-8 L236)。

    検証せず素の値を直列化する実装だと、候補生成器が載せた型違いの値が実行入力へ抜ける。
    """
    spec = _spec()
    plan = _plan(spec, (_resolved("target", "api"), _resolved("seconds", ["not", "an", "int"])))
    with pytest.raises(ValidationError):
        _ = plan.input_json


def test_input_json_covers_every_declared_parameter() -> None:
    """宣言した全パラメータが実行入力に現れる (FR-8 L236)。"""
    spec = _spec(
        param("target", str),
        param("seconds", int),
        param("region", str),
        label="${target} ${seconds} ${region}",
    )
    plan = _plan(
        spec,
        (
            _resolved("target", "api"),
            _resolved("seconds", 30),
            _resolved("region", "tokyo", origin=Origin.USER_INPUT),
        ),
    )
    assert set(json.loads(plan.input_json)) == {"target", "seconds", "region"}


def test_input_json_serialises_an_explicit_none_value() -> None:
    """`RESOLVED` + `value=None` も実行入力へ載る (FR-5 L187 / FR-8 L236)。"""
    spec = _spec(param("target", str | None), label="${target}")
    plan = _plan(spec, (_resolved("target", None, origin=Origin.DEFAULT),))
    assert json.loads(plan.input_json) == {"target": None}


def test_input_json_is_a_read_only_property() -> None:
    """`input_json` はメソッドではなく参照する property である (FR-8 L235 / L236)。"""
    assert isinstance(ActionPlan.input_json, property)


# ---------------------------------------------------------------------------
# 公開面 (FR-8 L237-L240 / 設計 §3.5b)
# ---------------------------------------------------------------------------


def test_plan_does_not_expose_parameters() -> None:
    """`parameters` プロパティは公開しない (設計 §3.5b)。

    `input_json` と同じデータの別表現であり、値は `slots` にもある。型付きで触りたい
    利用者は公開されている `spec.parameters_model()` から自分で組める。
    """
    plan = _pending_plan().apply({"seconds": 30})
    assert not hasattr(plan, "parameters")


def test_plan_does_not_expose_an_agent_instance() -> None:
    """実行先エージェントの実体を解決しない (FR-8 L238 / 設計 §3.4d)。"""
    plan = _pending_plan()
    assert not hasattr(plan, "agent")
    assert plan.action_agent == "load_test_runner"


def test_slots_is_readable_after_apply() -> None:
    """`plan.slots` から `name` / `state` / `value` / `origin` / `detail` を読める (FR-8 L240)。"""
    applied = _pending_plan().apply({"seconds": 30})
    readback = [
        (slot.name, slot.state, slot.value, slot.origin, slot.detail) for slot in applied.slots
    ]
    assert readback == [
        ("target", SlotState.RESOLVED, "api.example.com", Origin.CANDIDATE, None),
        ("seconds", SlotState.RESOLVED, 30, Origin.USER_INPUT, None),
    ]


def test_from_user_selects_only_user_supplied_slots() -> None:
    """`from_user` で書き戻し対象のスロットだけを選り分けられる (FR-8 L240 / 設計 §3.5a)。"""
    applied = _pending_plan().apply({"seconds": 30})
    assert [slot.name for slot in applied.slots if slot.from_user] == ["seconds"]


def test_pending_is_empty_after_every_slot_is_answered() -> None:
    """`apply` で全スロットが埋まると `pending` が空になる (FR-5 L188 / FR-8 L230)。"""
    applied = _pending_plan().apply({"seconds": 30})
    assert applied.pending == ()


def test_apply_result_excludes_declaration_from_model_dump() -> None:
    """`apply` 後も `model_dump()` に宣言と結線が漏れない (FR-5 L189)。"""
    dumped = _pending_plan().apply({"seconds": 30}).model_dump()
    assert set(dumped) == {"action_id", "slots"}


# ---------------------------------------------------------------------------
# input_json の未解決ガードそのもの (FR-8 L235)
#
# 「未解決なら例外」だけを見る検証は、ガードを取り除いても `parameters_model()` 側の
# 必須フィールド検証が代わりに `ValidationError`（`ValueError` 派生）を投げるため緑の
# まま通る。ガードが**それ自身の判断で**落ちていることを、文言・派生型・宣言型の 3 面
# から固定する。
# ---------------------------------------------------------------------------


def test_input_json_unresolved_error_message_has_the_declared_prefix() -> None:
    """未解決ガードの文言は決められた接頭辞で始まる (FR-8 L235)。

    「未解決だから落ちた」ことを利用者が文言で判別できる契約である。`parameters_model()`
    の必須フィールド検証へ倒れた場合は pydantic の文言になり、この接頭辞にならない。
    """
    with pytest.raises(
        ValueError, match="cannot build the execution input while slots are unresolved"
    ):
        _ = _pending_plan().input_json


def test_input_json_unresolved_guard_is_a_plain_value_error() -> None:
    """未解決ガードは `ValidationError` ではなく素の `ValueError` である (FR-8 L235)。

    `ValidationError` は `ValueError` 派生であるため `pytest.raises(ValueError)` では
    区別できない。ガードを外して型検証へ倒れた実装をここで弾く。
    """
    with pytest.raises(ValueError) as excinfo:
        _ = _pending_plan().input_json
    assert type(excinfo.value) is ValueError


def test_input_json_guard_fires_even_when_the_declaration_accepts_none() -> None:
    """`None` を受理する宣言でも `ready=False` なら `ValueError` (FR-8 L235)。

    `param("a", Any)` は未解決スロットの `value`（`None`）をそのまま受理するため、
    ガードを外した実装は例外を出さず `{"a": null}` を実行入力として返してしまう。
    宣言型の寛容さに依存せずガード自身が落ちることを固定する。
    """
    spec = ActionSpec(
        action_id="run_load_test",
        description="負荷試験を実行する",
        action_agent="load_test_runner",
        label="固定ラベル",
        parameters=(param("a", Any),),
    )
    plan = _plan(spec, (Slot(name="a", state=SlotState.NEEDS_USER),))
    with pytest.raises(ValueError) as excinfo:
        _ = plan.input_json
    assert type(excinfo.value) is ValueError
    assert "a" in str(excinfo.value)


# ---------------------------------------------------------------------------
# USER_CONFIRMED は NEEDS_CONFIRMATION 限定である (FR-8 L231)
# ---------------------------------------------------------------------------


def test_answer_equal_to_a_suggestion_of_a_needs_user_slot_is_user_input() -> None:
    """`NEEDS_USER` は `suggestions` と等しい値でも `USER_INPUT` になる (FR-8 L231)。

    validator が `suggestions` を禁じているのは `NEEDS_LLM` だけであり、`NEEDS_USER` は
    候補を提示したまま利用者へ聞ける。突き合わせの状態条件（`NEEDS_CONFIRMATION`）を
    落とした実装は、確認を経ていない入力を `USER_CONFIRMED` と記録してしまう。
    """
    spec = _spec()
    slot = Slot(
        name="seconds",
        state=SlotState.NEEDS_USER,
        suggestions=(SlotSuggestion(value=30),),
    )
    plan = _plan(spec, (_resolved("target", "api"), slot))
    answered = _slot(plan.apply({"seconds": 30}), "seconds")
    assert answered.state is SlotState.RESOLVED
    assert answered.origin is Origin.USER_INPUT


# ---------------------------------------------------------------------------
# apply の宣言順保存（先頭スロットを埋めた場合）(FR-8 L240 / 設計 §3.5a)
# ---------------------------------------------------------------------------


def test_apply_keeps_declaration_order_when_the_first_slot_is_answered() -> None:
    """先頭スロットを `apply` しても宣言順のまま (FR-8 L240 / 設計 §3.5a)。

    最後のスロットだけを埋める検証は「埋めたスロットを末尾へ並べ直す」実装でも緑に
    なる。先頭を埋めて順序が動かないことまで固定する。
    """
    spec = _spec()
    plan = _plan(spec, (_needs_user("target"), _needs_user("seconds")))
    applied = plan.apply({"target": "api.example.com"})
    assert [slot.name for slot in applied.slots] == ["target", "seconds"]


def test_input_json_keeps_declaration_order_when_the_first_slot_is_answered() -> None:
    """先頭スロットを `apply` した後も実行入力のキーは宣言順である (FR-8 L236)。"""
    spec = _spec()
    plan = _plan(spec, (_needs_user("target"), _resolved("seconds", 30)))
    applied = plan.apply({"target": "api.example.com"})
    assert list(json.loads(applied.input_json)) == ["target", "seconds"]


# ---------------------------------------------------------------------------
# apply の検証順（未知キー / 既 RESOLVED -> 型検証）(FR-8 L232-L234)
# ---------------------------------------------------------------------------


def test_unknown_key_is_rejected_before_type_validation() -> None:
    """未知キーと型不一致値が同時に来たら `ValueError` になる (FR-8 L232 / L234)。

    型検証を先に走らせる実装だと `ValidationError` が先に出て、未知キーの列挙という
    診断そのものが利用者へ届かなくなる。`ValidationError` は `ValueError` 派生なので
    派生型まで固定する。
    """
    with pytest.raises(ValueError) as excinfo:
        _pending_plan().apply({"secnods": 30, "seconds": "not-a-number"})
    assert type(excinfo.value) is ValueError
    assert "secnods" in str(excinfo.value)


def test_resolved_overwrite_is_rejected_before_type_validation() -> None:
    """既 `RESOLVED` キーに型不一致値を渡しても `ValueError` になる (FR-8 L233 / L234)。

    上書き禁止の判定は型検証より前に全件行う。順序が入れ替わると「確定済み値は上書き
    できない」という拒否理由が型エラーに隠れる。
    """
    with pytest.raises(ValueError) as excinfo:
        _pending_plan().apply({"target": 123})
    assert type(excinfo.value) is ValueError
    assert "target" in str(excinfo.value)
