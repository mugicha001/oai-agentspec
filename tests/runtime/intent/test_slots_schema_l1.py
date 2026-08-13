"""L1: 予測段のスキーマモデル生成（タスク 1-7・FR-2 L124-L127）の純検証。

`NEEDS_LLM` 状態のパラメータだけをフィールドに持つモデルを組み、`max_suggestions` に応じて
`SlotSuggestion[T]` / `list[SlotSuggestion[T]]` + `max_length` を使い分けること、その parse 用
派生が全 Optional かつ `max_length` を持たないこと（設計 §3.8）、そして当該モデルの生成関数が
**公開 API ではない**こと（設計 §3.5b の `slots_model()` 非公開化）を対象とする。

関数名は設計に明示が無いため、案 1 の `slots_model()` を非公開化した
`slots._slots_model(plan)` として固定する。`ActionPlan` はテスト側で直接組み立てるため、
本ファイルは決定的段（タスク 1-6）の実装に依存しない。外部依存 (agents / openai) なし。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from oai_agentspec.runtime.intent import slots as slots_module
from oai_agentspec.runtime.intent._models import derive_optional_model
from oai_agentspec.runtime.intent.actions import ActionSpec, param
from oai_agentspec.runtime.intent.slots import (
    ActionPlan,
    Origin,
    Slot,
    SlotState,
    SlotSuggestion,
)

pytestmark = pytest.mark.unit


def _spec(*parameters: Any, label: str = "${target}") -> ActionSpec:
    """テスト用の ActionSpec を組む。"""
    return ActionSpec(
        action_id="run_load_test",
        description="負荷試験を実行する",
        action_agent="load_test_runner",
        label=label,
        parameters=parameters,
    )


def _plan(spec: ActionSpec, slots: tuple[Slot, ...]) -> ActionPlan:
    """テスト用の ActionPlan を組む。"""
    return ActionPlan(
        action_id=spec.action_id,
        slots=slots,
        spec=spec,
        action_agent=spec.action_agent,
        resolved_prompt=(),
        resolved_prompt_vars={},
        resolved_on_invalid_slot="skip",
    )


def _needs_llm(name: str) -> Slot:
    """`NEEDS_LLM` のスロットを組む。"""
    return Slot(name=name, state=SlotState.NEEDS_LLM)


def _resolved(name: str, value: Any) -> Slot:
    """`RESOLVED` のスロットを組む。"""
    return Slot(name=name, state=SlotState.RESOLVED, value=value, origin=Origin.CANDIDATE)


def _mixed_plan() -> ActionPlan:
    """解決済み 1 件・`NEEDS_LLM` 2 件・`NEEDS_USER` 1 件の計画を組む。"""
    spec = _spec(
        param("target", str),
        param("seconds", int, by_llm=True, description="負荷をかける秒数"),
        param("region", str, by_llm=True),
        param("ticket", str),
        label="${target} ${seconds} ${region} ${ticket}",
    )
    return _plan(
        spec,
        (
            _resolved("target", "api.example.com"),
            _needs_llm("seconds"),
            _needs_llm("region"),
            Slot(name="ticket", state=SlotState.NEEDS_USER),
        ),
    )


# ---------------------------------------------------------------------------
# フィールド集合 (FR-2 L124)
# ---------------------------------------------------------------------------


def test_schema_model_has_only_needs_llm_fields() -> None:
    """`NEEDS_LLM` 状態のパラメータのみをフィールドに持つ (FR-2 L124)。

    解決済みのパラメータを載せると、LLM が既に確定した値を上書きできてしまう。
    """
    model = slots_module._slots_model(_mixed_plan())
    assert list(model.model_fields) == ["seconds", "region"]


def test_schema_model_keeps_declaration_order() -> None:
    """フィールドは宣言順に並ぶ (FR-2 L124 / 設計 §3.5a)。"""
    spec = _spec(
        param("region", str, by_llm=True),
        param("seconds", int, by_llm=True),
        label="${region} ${seconds}",
    )
    plan = _plan(spec, (_needs_llm("region"), _needs_llm("seconds")))
    assert list(slots_module._slots_model(plan).model_fields) == ["region", "seconds"]


def test_schema_model_is_empty_when_nothing_needs_the_llm() -> None:
    """`NEEDS_LLM` が 1 件も無ければフィールド 0 件のモデルになる (FR-2 L124)。"""
    spec = _spec(param("target", str))
    plan = _plan(spec, (_resolved("target", "api.example.com"),))
    assert slots_module._slots_model(plan).model_fields == {}


def test_schema_model_is_a_frozen_base_model_subclass() -> None:
    """生成物は frozen な `BaseModel` サブクラスである (設計 §3.1)。"""
    model = slots_module._slots_model(_mixed_plan())
    assert issubclass(model, BaseModel)
    instance = model(seconds={"value": 60}, region={"value": "tokyo"})
    with pytest.raises(ValidationError):
        instance.seconds = {"value": 1}


def test_schema_model_name_is_a_public_identifier() -> None:
    """生成クラス名は公開識別子である（`title` として LLM へ渡る出力面）。"""
    name = slots_module._slots_model(_mixed_plan()).__name__
    assert name.isidentifier()
    assert not name.startswith("_")


def test_schema_model_reflects_parameter_description() -> None:
    """`param(description=...)` がスキーマへ反映される (FR-2 L122 / L124)。"""
    schema = slots_module._slots_model(_mixed_plan()).model_json_schema()
    assert "負荷をかける秒数" in str(schema["properties"]["seconds"])


# ---------------------------------------------------------------------------
# max_suggestions == 1: SlotSuggestion[T] (FR-2 L126)
# ---------------------------------------------------------------------------


def test_single_suggestion_field_is_a_slot_suggestion() -> None:
    """`max_suggestions` が 1 なら型は `SlotSuggestion[T]` である (FR-2 L126)。"""
    model = slots_module._slots_model(_mixed_plan())
    parsed = model.model_validate({"seconds": {"value": 60}, "region": {"value": "tokyo"}})
    assert isinstance(parsed.seconds, SlotSuggestion)
    assert parsed.seconds.value == 60


def test_single_suggestion_field_validates_the_declared_inner_type() -> None:
    """`SlotSuggestion[T]` の `T` は `param` の第 2 引数である (FR-2 L126)。

    `SlotSuggestion[Any]` で組むと、宣言型に合わない値が素通りして後段で落ちる。
    """
    model = slots_module._slots_model(_mixed_plan())
    with pytest.raises(ValidationError):
        model.model_validate({"seconds": {"value": "not-a-number"}, "region": {"value": "tokyo"}})


def test_single_suggestion_field_rejects_a_bare_value() -> None:
    """裸の値ではなく `SlotSuggestion` の形を要求する (FR-2 L126)。"""
    model = slots_module._slots_model(_mixed_plan())
    with pytest.raises(ValidationError):
        model.model_validate({"seconds": 60, "region": {"value": "tokyo"}})


def test_single_suggestion_field_is_not_a_list() -> None:
    """`max_suggestions` が 1 のフィールドに `maxItems` は付かない (FR-2 L126)。"""
    schema = slots_module._slots_model(_mixed_plan()).model_json_schema()
    assert "maxItems" not in schema["properties"]["seconds"]


def test_single_suggestion_carries_level_and_rationale() -> None:
    """`level` / `rationale` を伴う応答を受け取れる (FR-2 L126 / FR-6 L207)。"""
    model = slots_module._slots_model(_mixed_plan())
    parsed = model.model_validate(
        {
            "seconds": {"value": 60, "level": "high", "rationale": "直近の会話から"},
            "region": {"value": "tokyo"},
        }
    )
    assert parsed.seconds.level == "high"
    assert parsed.seconds.rationale == "直近の会話から"


# ---------------------------------------------------------------------------
# max_suggestions >= 2: list[SlotSuggestion[T]] + max_length (FR-2 L125)
# ---------------------------------------------------------------------------


def _multi_plan(max_suggestions: int = 3) -> ActionPlan:
    """`max_suggestions >= 2` のパラメータを持つ計画を組む。"""
    spec = _spec(param("seconds", int, by_llm=True, max_suggestions=max_suggestions))
    return _plan(spec, (_needs_llm("seconds"),))


def test_multi_suggestion_field_is_a_list() -> None:
    """`max_suggestions` が 2 以上なら型は `list[SlotSuggestion[T]]` である (FR-2 L125)。"""
    model = slots_module._slots_model(_multi_plan())
    parsed = model.model_validate({"seconds": [{"value": 60}, {"value": 90}]})
    assert [suggestion.value for suggestion in parsed.seconds] == [60, 90]
    assert all(isinstance(suggestion, SlotSuggestion) for suggestion in parsed.seconds)


def test_multi_suggestion_field_validates_the_declared_inner_type() -> None:
    """list の内側も宣言型で検証される (FR-2 L125)。"""
    model = slots_module._slots_model(_multi_plan())
    with pytest.raises(ValidationError):
        model.model_validate({"seconds": [{"value": "not-a-number"}]})


def test_multi_suggestion_field_reports_max_items_in_json_schema() -> None:
    """`max_suggestions` が `model_json_schema()` の `maxItems` に反映される (FR-2 L125)。

    LLM への上限提示に使うため、スキーマ提示用のモデルには `max_length` を付ける
    （設計 §3.8）。
    """
    schema = slots_module._slots_model(_multi_plan(max_suggestions=3)).model_json_schema()
    assert schema["properties"]["seconds"]["maxItems"] == 3


def test_multi_suggestion_max_items_follows_the_declaration() -> None:
    """`maxItems` は宣言値そのものである（固定値を返す実装を弾く）(FR-2 L125)。"""
    schema = slots_module._slots_model(_multi_plan(max_suggestions=2)).model_json_schema()
    assert schema["properties"]["seconds"]["maxItems"] == 2


def test_schema_model_rejects_more_than_max_suggestions() -> None:
    """スキーマ提示用のモデルは上限超過を `ValidationError` にする (設計 §3.8)。"""
    model = slots_module._slots_model(_multi_plan(max_suggestions=2))
    with pytest.raises(ValidationError):
        model.model_validate({"seconds": [{"value": 1}, {"value": 2}, {"value": 3}]})


def test_schema_model_accepts_exactly_max_suggestions() -> None:
    """上限ちょうどは受け付ける (設計 §3.8)。"""
    model = slots_module._slots_model(_multi_plan(max_suggestions=2))
    parsed = model.model_validate({"seconds": [{"value": 1}, {"value": 2}]})
    assert len(parsed.seconds) == 2


def test_mixed_max_suggestions_use_different_shapes_in_one_model() -> None:
    """同一モデル内で 1 件のフィールドと複数件のフィールドが共存する (FR-2 L125 / L126)。"""
    spec = _spec(
        param("seconds", int, by_llm=True),
        param("region", str, by_llm=True, max_suggestions=2),
        label="${seconds} ${region}",
    )
    plan = _plan(spec, (_needs_llm("seconds"), _needs_llm("region")))
    schema = slots_module._slots_model(plan).model_json_schema()
    assert "maxItems" not in schema["properties"]["seconds"]
    assert schema["properties"]["region"]["maxItems"] == 2


# ---------------------------------------------------------------------------
# parse 用派生 (FR-2 L127 / 設計 §3.8)
# ---------------------------------------------------------------------------


def test_parse_model_accepts_missing_fields() -> None:
    """parse 派生は一部フィールドが欠落した JSON でも成功する (FR-2 L127)。"""
    parse_model = derive_optional_model(slots_module._slots_model(_mixed_plan()))
    parsed = parse_model.model_validate_json('{"seconds": {"value": 60}}')
    assert parsed.seconds.value == 60
    assert parsed.region is None


def test_parse_model_accepts_an_empty_object() -> None:
    """全フィールド欠落でも parse できる (FR-2 L127)。"""
    parse_model = derive_optional_model(slots_module._slots_model(_mixed_plan()))
    parsed = parse_model.model_validate_json("{}")
    assert parsed.seconds is None
    assert parsed.region is None


def test_parse_model_fields_are_optional_with_none_default() -> None:
    """parse 派生の全フィールドが `X | None`（既定 `None`）である (FR-2 L127)。"""
    parse_model = derive_optional_model(slots_module._slots_model(_mixed_plan()))
    for info in parse_model.model_fields.values():
        assert not info.is_required()
        assert info.default is None


def test_parse_model_has_no_max_length_constraint() -> None:
    """parse 派生は `max_length` を持たない (FR-2 L127 / 設計 §3.8)。

    上限超過を `ValidationError` にすると、LLM が 1 件超えただけで応答全体が落ち、
    `on_invalid_response` の判断材料が失われる。切り捨ては lib 側が行う。
    """
    parse_model = derive_optional_model(slots_module._slots_model(_multi_plan(max_suggestions=2)))
    parsed = parse_model.model_validate_json(
        '{"seconds": [{"value": 1}, {"value": 2}, {"value": 3}]}'
    )
    assert len(parsed.seconds) == 3


def test_parse_model_json_schema_has_no_max_items() -> None:
    """parse 派生のスキーマに `maxItems` が現れない (設計 §3.8)。"""
    parse_model = derive_optional_model(slots_module._slots_model(_multi_plan(max_suggestions=2)))
    assert "maxItems" not in str(parse_model.model_json_schema()["properties"]["seconds"])


def test_parse_model_still_validates_the_declared_inner_type() -> None:
    """parse 派生でも宣言型の検証は残る (FR-2 L127 / FR-7 L222)。

    型不一致は「値が欠落した」ではなく `on_invalid_slot` の対象として扱うため、
    全 Optional 化で型検証まで失ってはならない。
    """
    parse_model = derive_optional_model(slots_module._slots_model(_mixed_plan()))
    with pytest.raises(ValidationError):
        parse_model.model_validate_json('{"seconds": {"value": "not-a-number"}}')


# ---------------------------------------------------------------------------
# 非公開性 (FR-2 L124 / 設計 §3.5b)
# ---------------------------------------------------------------------------


def test_schema_model_builder_is_not_public() -> None:
    """スキーマモデルの生成は公開 API として提供しない (FR-2 L124 / 設計 §3.5b)。

    予測が `planner.plan()` の内部へ畳まれたため、利用者が呼ぶ場面が無い。
    公開すると「UI フォーム生成には `parameters_model()`」という導線が二重になる。
    """
    from oai_agentspec.runtime import intent as intent_package

    assert slots_module._slots_model.__name__.startswith("_")
    assert "slots_model" not in intent_package.__all__
    assert "_slots_model" not in intent_package.__all__
