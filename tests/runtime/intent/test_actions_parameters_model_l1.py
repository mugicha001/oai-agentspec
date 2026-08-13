"""L1: `ActionSpec.parameters_model()`（タスク 1-3・FR-2）の純検証。

宣言した型を単一の出どころとして実行入力の検証・LLM スキーマ・UI フォーム生成へ使い回す
契約を pin する。対象は「全 `ParameterSpec` をフィールド化した frozen モデルを返すこと」
「フィールド型が `param` の第 2 引数であること」「`Field(description=...)` が反映されること」
「2 回以上呼ぶと**同一のクラスオブジェクト**を返すこと（キャッシュの契約なので identity で
判定する）」「pydantic が扱えない型はパラメータ名を添えた `ValueError` になること」。

生成の実体は `_models.build_frozen_model`（タスク 1-1・`test_models_l1.py` で検証済み）で
あり、本ファイルは宣言型からの橋渡しとキャッシュのみを対象とする。
外部依存 (agents / openai) なし。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from oai_agentspec.runtime.intent.actions import ActionSpec, param

pytestmark = pytest.mark.unit


class _Unschemable:
    """pydantic がフィールド型として扱えないプレーンなクラス。"""


def _make_spec(**overrides: Any) -> ActionSpec:
    """テスト用の最小 ActionSpec を組む。"""
    fields: dict[str, Any] = {
        "action_id": "run_load_test",
        "description": "負荷試験を実行する",
        "action_agent": "load_test_runner",
        "label": "${target} に ${seconds} 秒の負荷試験",
        "parameters": (
            param("target", str, description="対象ホスト"),
            param("seconds", int),
        ),
    }
    fields.update(overrides)
    return ActionSpec(**fields)


# ---------------------------------------------------------------------------
# 生成物の性質 (FR-2 L122)
# ---------------------------------------------------------------------------


def test_parameters_model_returns_base_model_subclass() -> None:
    """`parameters_model()` は `BaseModel` サブクラスを返す (FR-2 L122)。"""
    model = _make_spec().parameters_model()
    assert isinstance(model, type)
    assert issubclass(model, BaseModel)


def test_parameters_model_is_frozen() -> None:
    """生成モデルのインスタンスは frozen である (FR-2 L122)。

    実行入力として下流へ渡る値であり、渡した後に書き換えられると
    `input_json` で検証した内容と実際に実行される内容が食い違う。
    """
    model = _make_spec().parameters_model()
    instance = model(target="api.example.com", seconds=30)
    with pytest.raises(ValidationError):
        instance.target = "evil.example.com"


def test_parameters_model_has_all_declared_parameters_as_fields() -> None:
    """全 `ParameterSpec` が宣言順にフィールド化される (FR-2 L122)。"""
    model = _make_spec().parameters_model()
    assert list(model.model_fields) == ["target", "seconds"]


def test_parameters_model_field_type_is_the_second_param_argument() -> None:
    """フィールド型は `param` の第 2 引数そのものである (FR-2 L122)。"""
    model = _make_spec().parameters_model()
    assert model.model_fields["target"].annotation is str
    assert model.model_fields["seconds"].annotation is int


def test_parameters_model_validates_against_the_declared_type() -> None:
    """宣言型に合わない値は `ValidationError` になる (FR-2 L122)。

    annotation を「載せるだけ」で検証に効かせない実装を弾く。
    """
    model = _make_spec().parameters_model()
    with pytest.raises(ValidationError):
        model(target="api.example.com", seconds="not-a-number")


def test_parameters_model_reflects_description_in_field() -> None:
    """`param(description=...)` が `Field(description=...)` へ反映される (FR-2 L122)。"""
    model = _make_spec().parameters_model()
    assert model.model_fields["target"].description == "対象ホスト"


def test_parameters_model_reflects_description_in_json_schema() -> None:
    """description が `model_json_schema()` に載る (FR-2 L122・UI フォーム生成の用途)。"""
    schema = _make_spec().parameters_model().model_json_schema()
    assert schema["properties"]["target"]["description"] == "対象ホスト"


def test_parameters_model_omits_description_when_not_declared() -> None:
    """description 未宣言のパラメータには description が載らない (FR-2 L122)。"""
    model = _make_spec().parameters_model()
    assert model.model_fields["seconds"].description is None


def test_parameters_model_name_is_a_public_identifier() -> None:
    """生成クラス名は公開識別子である (設計 §3.1 の `_models` 入口検証と整合)。

    クラス名は `model_json_schema()` の `title` として LLM / UI へ渡る出力面である。
    具体的な綴りは契約にしないが、識別子であること・`_` 始まりでないことは固定する。
    """
    name = _make_spec().parameters_model().__name__
    assert name.isidentifier()
    assert not name.startswith("_")


def test_parameters_model_accepts_a_spec_without_parameters() -> None:
    """パラメータ 0 件の宣言でもフィールド 0 件のモデルとして成立する (FR-2 L122)。"""
    spec = _make_spec(label="固定ラベル", parameters=())
    assert spec.parameters_model().model_fields == {}


# ---------------------------------------------------------------------------
# 宣言した default の反映 (FR-2 L122 / 設計 §3.7 の has_default)
# ---------------------------------------------------------------------------


def test_parameters_model_field_is_required_when_no_default_is_declared() -> None:
    """`default` 未宣言のパラメータは必須フィールドになる (FR-2 L122)。"""
    model = _make_spec().parameters_model()
    assert model.model_fields["seconds"].is_required()


def test_parameters_model_reflects_declared_default() -> None:
    """`param(default=...)` を宣言したパラメータは当該既定を持つ (FR-2 L122・設計 §3.7)。

    「未宣言」と「明示的な `default=None`」を分ける `has_default` は、生成モデルの
    必須 / 任意の別として現れるのが唯一の観測点である。
    """
    spec = _make_spec(
        parameters=(param("target", str), param("seconds", int, default=60)),
    )
    model = spec.parameters_model()
    assert not model.model_fields["seconds"].is_required()
    assert model(target="api.example.com").seconds == 60


def test_parameters_model_reflects_explicit_none_default() -> None:
    """明示的な `default=None` は「未宣言」ではなく既定 `None` として反映される (設計 §3.7)。"""
    spec = _make_spec(
        label="${target}",
        parameters=(param("target", str | None, default=None),),
    )
    model = spec.parameters_model()
    assert not model.model_fields["target"].is_required()
    assert model().target is None


# ---------------------------------------------------------------------------
# キャッシュ (FR-2 L123)
# ---------------------------------------------------------------------------


def test_parameters_model_returns_the_same_class_object_on_second_call() -> None:
    """2 回以上呼ぶと**同一のクラスオブジェクト**を返す (FR-2 L123)。

    識別子の同一性そのものがキャッシュの契約であるため `is` で判定する
    （`==` では毎回作り直す実装を通してしまう）。
    """
    spec = _make_spec()
    assert spec.parameters_model() is spec.parameters_model()


def test_parameters_model_cache_survives_three_calls() -> None:
    """3 回目以降も同一のクラスオブジェクトを返す (FR-2 L123)。"""
    spec = _make_spec()
    first = spec.parameters_model()
    spec.parameters_model()
    assert spec.parameters_model() is first


def test_parameters_model_instances_are_accepted_by_the_cached_class() -> None:
    """キャッシュされたクラスで作った値が同じクラスの `isinstance` を満たす (FR-2 L123)。

    毎回別クラスを返す実装では、先に組んだインスタンスが後の呼び出しで得たモデルの
    `isinstance` を満たさず、下流の型検証が黙って落ちる。
    """
    spec = _make_spec()
    instance = spec.parameters_model()(target="api.example.com", seconds=30)
    assert isinstance(instance, spec.parameters_model())


def test_parameters_model_cache_is_not_shared_between_specs() -> None:
    """別々の `ActionSpec` は別々のクラスオブジェクトを得る (FR-2 L123)。

    クラス単位のキャッシュにすると、宣言が違うのに同じモデルが返る。
    """
    first = _make_spec()
    second = _make_spec(
        action_id="send_notice",
        label="${target} へ通知",
        parameters=(param("target", str),),
    )
    assert first.parameters_model() is not second.parameters_model()
    assert list(second.parameters_model().model_fields) == ["target"]


def test_parameters_model_of_a_model_copy_reflects_the_copied_parameters() -> None:
    """`model_copy(update=...)` したコピーは**自分の宣言**のモデルを返す (FR-2 L123)。

    キャッシュは「同一の宣言に対する同一性」を保証するためのものであり、宣言が違えば
    別のモデルでなければならない（`test_parameters_model_cache_is_not_shared_between_specs`
    と同じ契約）。`model_copy` は `PrivateAttr` ごとコピーするため、元で
    `parameters_model()` を呼んだ後にコピーすると、コピーが**元の宣言**のモデルを返す。
    実行入力の検証（`ActionPlan.input_json`）がコピー後の宣言を見なくなり、宣言に無い
    パラメータで検証される。
    """
    spec = _make_spec()
    original = spec.parameters_model()
    copied = spec.model_copy(update={"parameters": (param("region", str),)})
    assert list(copied.parameters_model().model_fields) == ["region"]
    assert copied.parameters_model() is not original


def test_parameters_model_of_a_model_copy_reflects_the_copied_action_id() -> None:
    """`action_id` を差し替えたコピーは自分の `action_id` 由来の title を持つ (FR-2 L123)。

    `action_id` は生成クラスの `__name__` と `model_json_schema()` の `title` になる出力面
    （LLM / UI へ渡る）。キャッシュの鍵から `action_id` が落ちると、コピーが別アクションの
    title を持つモデルを返す。綴り全体は契約にしないため語幹のみを見る。
    """
    spec = _make_spec()
    original = spec.parameters_model()
    copied = spec.model_copy(update={"action_id": "other_action"})
    assert copied.parameters_model() is not original
    assert copied.parameters_model().__name__.startswith("OtherAction")


def test_parameters_model_cache_does_not_break_frozen_declaration() -> None:
    """キャッシュを持っても `ActionSpec` の frozen 契約は保たれる (FR-1 / FR-2 L123)。"""
    spec = _make_spec()
    spec.parameters_model()
    with pytest.raises(ValidationError):
        spec.action_id = "other"


def test_parameters_model_cache_is_not_visible_as_a_public_field() -> None:
    """キャッシュは公開フィールドとして現れない (FR-2 L123・設計 §3.5b)。

    宣言簿を `model_dump()` した結果に生成クラスが混ざると、宣言の直列化が壊れる。
    """
    spec = _make_spec()
    spec.parameters_model()
    assert "parameters_model" not in type(spec).model_fields
    assert all(not key.startswith("_") for key in spec.model_dump())


# ---------------------------------------------------------------------------
# 扱えない型の変換 (FR-2 L128)
# ---------------------------------------------------------------------------


def test_parameters_model_raises_value_error_for_unusable_annotation() -> None:
    """pydantic が扱えない型は `ValueError` になる (FR-2 L128)。

    素の `PydanticSchemaGenerationError` は `RuntimeError` 派生であり、素通しでは
    `ValueError` を捕捉する呼び出し側の網を抜ける。
    """
    spec = _make_spec(
        parameters=(param("target", str), param("seconds", _Unschemable)),
    )
    with pytest.raises(ValueError):
        spec.parameters_model()


def test_parameters_model_value_error_names_the_offending_parameter() -> None:
    """変換後の `ValueError` に当該パラメータ名が含まれる (FR-2 L128)。"""
    spec = _make_spec(
        parameters=(param("target", str), param("seconds", _Unschemable)),
    )
    with pytest.raises(ValueError) as excinfo:
        spec.parameters_model()
    assert "seconds" in str(excinfo.value)


def test_parameters_model_failure_is_not_cached_as_a_model() -> None:
    """生成に失敗した宣言は 2 回目も同じ `ValueError` になる (FR-2 L123 / L128)。

    失敗を握り潰して壊れたクラスや `None` をキャッシュする実装を弾く。
    """
    spec = _make_spec(parameters=(param("seconds", _Unschemable),))
    with pytest.raises(ValueError):
        spec.parameters_model()
    with pytest.raises(ValueError):
        spec.parameters_model()


def test_parameters_model_call_does_not_change_equality() -> None:
    """`parameters_model()` の呼び出しが `ActionSpec` の等価性を変えない (FR-2 L123)。

    キャッシュは実装の詳細であり、宣言の内容で決まる等価性へ混ざってはならない。
    読み取りに見えるアクセサが観測結果を変えると、呼び出し側から予測できない。
    """
    spec = _make_spec()
    other = _make_spec()
    assert spec == other
    spec.parameters_model()
    assert spec == other, "parameters_model() の呼び出しが等価性を変えています"


def test_equality_still_reflects_the_declaration() -> None:
    """等価性は宣言の内容で決まる（キャッシュの有無に依らない）(FR-2 L123)。"""
    spec = _make_spec()
    spec.parameters_model()
    assert spec != _make_spec(action_id="other_action")
    assert spec != _make_spec(parameters=(param("region", str),))
    assert spec == _make_spec()


def test_equality_with_a_foreign_type_is_not_forced_to_false() -> None:
    """異なる型との比較は NotImplemented を返し判断を相手側へ委ねる (FR-2 L123)。

    False に固定すると、相手側が定義した __eq__ が呼ばれなくなる。
    """
    spec = _make_spec()
    assert spec.__eq__(1) is NotImplemented
    assert spec != 1

    class _AlwaysEqual:
        def __eq__(self, other: object) -> bool:
            return True

    assert spec == _AlwaysEqual()


def test_equality_requires_the_exact_same_class() -> None:
    """サブクラスとは宣言が同一でも不等（pydantic 既定と同じ厳密一致）(FR-2 L123)。

    isinstance で緩めると、派生型が親型として等価に扱われ、宣言の同一性という
    契約が型をまたいで曖昧になる。
    """

    class _Sub(ActionSpec):
        pass

    spec = _make_spec()
    sub = _Sub(**spec.model_dump())
    assert spec != sub
    assert sub != spec
