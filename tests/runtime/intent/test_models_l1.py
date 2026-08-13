"""L1: `runtime.intent._models` の frozen モデル生成ビルダ (`build_frozen_model` /
`derive_optional_model`) の純検証。

FR-2 が要求する生成物の性質を pin する。生成モデルの frozen 性・`Field(description=...)` の
`model_json_schema()` 反映・`list[X]` + `max_length` の `maxItems` 反映・
`PydanticSchemaGenerationError` からフィールド名付き `ValueError` への変換 (設計 §3.9)・
parse 派生が全フィールドを `X | None` (既定 `None`) にし `max_length` を落とすこと (設計 §3.8)、
および `_models.py` がドメイン型を 1 つも import しないこと (設計 §3.1 の循環回避契約) を対象と
する。ビルダはドメイン型を知らないため、annotation はすべてテスト側のダミー型で組んで渡す。
外部依存 (agents / openai) なし。
"""

from __future__ import annotations

import ast
import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, ForwardRef, Literal, Optional

import pytest
from pydantic import BaseModel, Field, ValidationError
from pydantic.errors import PydanticSchemaGenerationError
from pydantic.fields import FieldInfo

from oai_agentspec.runtime.intent._models import build_frozen_model, derive_optional_model
from oai_agentspec.runtime.intent.actions import param
from oai_agentspec.runtime.intent.types import IntentPrediction, IntentQuery

pytestmark = pytest.mark.unit

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
_MODELS_PATH = _SRC_DIR / "oai_agentspec" / "runtime" / "intent" / "_models.py"

# `_models.py` が import してよいトップレベルパッケージ (設計 §3.1: ドメイン型を 1 つも
# import しない)。ここに `oai_agentspec` 由来の名前が増えたら循環回避契約が崩れている。
_ALLOWED_IMPORT_ROOTS = frozenset({"__future__", "collections", "keyword", "pydantic", "typing"})


# ビルダはドメイン型を知らない契約 (設計 §3.1) のため、本番の SlotSuggestion に相当する
# 「呼び出し側が組み立てて渡す annotation」をテスト内のダミー型で再現する。
class _Suggestion[T](BaseModel):
    """`SlotSuggestion[T]` に相当するテスト用のダミー generic 型。"""

    model_config = {"frozen": True}
    value: T


def _fields(**spec: tuple[Any, FieldInfo]) -> dict[str, tuple[Any, FieldInfo]]:
    """`build_frozen_model` へ渡す fields マッピングを組む。"""
    return dict(spec)


def _simple_model(name: str = "Params") -> type[BaseModel]:
    """必須 2 フィールドの最小モデルを組む。"""
    return build_frozen_model(
        name,
        _fields(
            target=(str, Field(description="送信先ホスト")),
            seconds=(int, Field(description="待機秒数")),
        ),
    )


# ---------------------------------------------------------------------------
# build_frozen_model: 生成物の基本性質 (FR-2 L122)
# ---------------------------------------------------------------------------


def test_build_frozen_model_returns_base_model_subclass() -> None:
    """build_frozen_model は BaseModel のサブクラス (型そのもの) を返す (FR-2 L122)。"""
    model = _simple_model()
    assert isinstance(model, type)
    assert issubclass(model, BaseModel)


def test_build_frozen_model_uses_given_name() -> None:
    """第 1 引数がクラス名になる (FR-2 L122)。"""
    model = build_frozen_model("RunLoadTestParams", _fields(target=(str, Field())))
    assert model.__name__ == "RunLoadTestParams"


def test_build_frozen_model_maps_annotations_to_fields() -> None:
    """fields の第 1 要素 (annotation) がそのままフィールド型になる (FR-2 L122)。"""
    model = _simple_model()
    assert set(model.model_fields) == {"target", "seconds"}
    assert model.model_fields["target"].annotation is str
    assert model.model_fields["seconds"].annotation is int


def test_build_frozen_model_instance_is_frozen() -> None:
    """生成モデルのインスタンスは frozen (フィールド代入は ValidationError) (FR-2 L122)。"""
    instance = _simple_model()(target="host", seconds=60)
    assert instance.target == "host"
    assert instance.seconds == 60
    with pytest.raises(ValidationError):
        instance.target = "other"


def test_build_frozen_model_fields_are_required_by_default() -> None:
    """既定を持たない FieldInfo を渡したフィールドは必須のまま (FR-2 L122)。"""
    model = _simple_model()
    assert model.model_fields["target"].is_required() is True
    with pytest.raises(ValidationError):
        model(target="host")


def test_build_frozen_model_reflects_description_in_json_schema() -> None:
    """Field(description=...) が model_json_schema() に現れる (FR-2 L122)。"""
    schema = _simple_model().model_json_schema()
    assert schema["properties"]["target"]["description"] == "送信先ホスト"
    assert schema["properties"]["seconds"]["description"] == "待機秒数"
    assert sorted(schema["required"]) == ["seconds", "target"]


def test_build_frozen_model_accepts_parametrized_generic_annotation() -> None:
    """呼び出し側が組んだ generic の実体化型 (SlotSuggestion[T] 相当) を受け取れる (設計 §3.1)。"""
    model = build_frozen_model(
        "SingleSuggestionParams",
        _fields(target=(_Suggestion[str], Field(description="single"))),
    )
    schema = model.model_json_schema()
    assert schema["properties"]["target"]["description"] == "single"
    assert "_Suggestion_str_" in schema["$defs"]


def test_build_frozen_model_reflects_max_length_as_max_items() -> None:
    """list[X] + Field(max_length=N) が model_json_schema() の maxItems になる (FR-2 L125)。"""
    model = build_frozen_model(
        "MultiSuggestionParams",
        _fields(seconds=(list[_Suggestion[int]], Field(description="up to 3", max_length=3))),
    )
    prop = model.model_json_schema()["properties"]["seconds"]
    assert prop["type"] == "array"
    assert prop["maxItems"] == 3
    assert prop["description"] == "up to 3"


def test_build_frozen_model_enforces_max_length_at_validation() -> None:
    """max_length を付けたモデルは上限超過を ValidationError で弾く (設計 §3.8 の前提)。"""
    model = build_frozen_model(
        "MultiSuggestionParams",
        _fields(seconds=(list[_Suggestion[int]], Field(max_length=2))),
    )
    assert len(model(seconds=[{"value": 1}, {"value": 2}]).seconds) == 2
    with pytest.raises(ValidationError):
        model(seconds=[{"value": 1}, {"value": 2}, {"value": 3}])


def test_build_frozen_model_accepts_empty_fields() -> None:
    """フィールド 0 件でもモデルを生成できる (パラメータ無しアクションのため)。"""
    model = build_frozen_model("NoParams", {})
    assert model.model_fields == {}
    assert model().model_dump() == {}


# ---------------------------------------------------------------------------
# build_frozen_model: ビルダ自体の性質 (キャッシュは 1-3 の責務)
# ---------------------------------------------------------------------------


def test_build_frozen_model_returns_a_new_class_on_each_call() -> None:
    """同一入力で 2 回呼ぶと別のクラスオブジェクトを返す (ビルダはキャッシュを持たない)。

    `parameters_model()` の「2 回目も同一クラスオブジェクト」契約 (FR-2 L123) は
    呼び出し側 (`ActionSpec`) のキャッシュ責務であり、汎用ビルダの責務ではない。
    """
    first = _simple_model()
    second = _simple_model()
    assert first is not second
    assert first.model_fields.keys() == second.model_fields.keys()


def test_build_frozen_model_does_not_mutate_the_given_fields_mapping() -> None:
    """引数で渡した fields マッピングをビルダが書き換えない。"""
    fields = _fields(target=(str, Field(description="送信先ホスト")))
    snapshot = dict(fields)
    build_frozen_model("Params", fields)
    assert fields == snapshot


# ---------------------------------------------------------------------------
# build_frozen_model: 例外変換 (FR-2 L128 / 設計 §3.9)
# ---------------------------------------------------------------------------


class _Unschemable:
    """pydantic がフィールド型として扱えないプレーンなクラス。"""


def test_build_frozen_model_converts_schema_generation_error_to_value_error() -> None:
    """扱えない型は PydanticSchemaGenerationError を捕捉し ValueError へ変換する (FR-2 L128)。"""
    with pytest.raises(ValueError) as excinfo:
        build_frozen_model("Params", _fields(target=(_Unschemable, Field())))
    assert not isinstance(excinfo.value, PydanticSchemaGenerationError)


def test_build_frozen_model_value_error_names_the_offending_field() -> None:
    """変換後の ValueError のメッセージに当該パラメータ名が含まれる (FR-2 L128)。"""
    with pytest.raises(ValueError) as excinfo:
        build_frozen_model(
            "Params",
            _fields(target=(str, Field()), seconds=(_Unschemable, Field())),
        )
    assert "seconds" in str(excinfo.value)


def test_build_frozen_model_chains_the_original_exception() -> None:
    """`raise ... from exc` により __cause__ が元の例外になる (FR-2 L128)。"""
    with pytest.raises(ValueError) as excinfo:
        build_frozen_model("Params", _fields(target=(_Unschemable, Field())))
    assert isinstance(excinfo.value.__cause__, PydanticSchemaGenerationError)


def test_build_frozen_model_does_not_leak_runtime_error() -> None:
    """PydanticSchemaGenerationError は RuntimeError 派生なので素通ししない (設計 §3.9)。"""
    assert issubclass(PydanticSchemaGenerationError, RuntimeError)
    with pytest.raises(ValueError):
        build_frozen_model("Params", _fields(target=(_Unschemable, Field())))


# ---------------------------------------------------------------------------
# derive_optional_model (FR-2 L127 / 設計 §3.8)
# ---------------------------------------------------------------------------


def test_derive_optional_model_returns_a_distinct_base_model_subclass() -> None:
    """derive_optional_model は元と別の BaseModel サブクラスを返す (FR-2 L127)。"""
    source = _simple_model()
    derived = derive_optional_model(source)
    assert isinstance(derived, type)
    assert issubclass(derived, BaseModel)
    assert derived is not source


def test_derive_optional_model_does_not_mutate_the_source_model() -> None:
    """派生を作っても元モデルのフィールドは必須のまま (FR-2 L127)。"""
    source = _simple_model()
    derive_optional_model(source)
    assert source.model_fields["target"].is_required() is True


def test_derive_optional_model_makes_every_field_optional() -> None:
    """全フィールドが既定 None の任意フィールドになる (FR-2 L127)。"""
    derived = derive_optional_model(_simple_model())
    assert set(derived.model_fields) == {"target", "seconds"}
    for name in ("target", "seconds"):
        assert derived.model_fields[name].is_required() is False
        assert derived.model_fields[name].default is None


def test_derive_optional_model_widens_annotations_to_optional() -> None:
    """annotation 自体が X | None へ広がる (FR-2 L127)。

    既定値を None にするだけでは pydantic が既定値を検証しないため素通りする。
    annotation の Optional 化は明示 null を受け取るために load-bearing である。
    """
    derived = derive_optional_model(_simple_model())
    assert derived.model_fields["target"].annotation == str | None
    assert derived.model_fields["seconds"].annotation == int | None


def test_derive_optional_model_accepts_explicit_null_values() -> None:
    """明示 null を含む JSON を受理する (FR-2 L127)。

    LLM が埋められないフィールドを null で返すのは典型的な応答形であり、
    annotation が元の型のままだとここで ValidationError になり応答全体が落ちる。
    """
    derived = derive_optional_model(_simple_model())
    instance = derived.model_validate_json('{"target": null, "seconds": null}')
    assert instance.target is None
    assert instance.seconds is None


def test_derive_optional_model_allows_construction_without_arguments() -> None:
    """引数なしで生成でき、全フィールドが None (FR-2 L127)。"""
    instance = derive_optional_model(_simple_model())()
    assert instance.target is None
    assert instance.seconds is None


def test_derive_optional_model_accepts_json_with_missing_fields() -> None:
    """一部フィールドが欠落した JSON でも model_validate_json が成功する (FR-2 L127)。"""
    derived = derive_optional_model(_simple_model())
    instance = derived.model_validate_json('{"target": "host"}')
    assert instance.target == "host"
    assert instance.seconds is None


def test_derive_optional_model_accepts_empty_json_object() -> None:
    """空の JSON オブジェクトでも model_validate_json が成功する (FR-2 L127)。"""
    derived = derive_optional_model(_simple_model())
    assert derived.model_validate_json("{}").target is None


def test_derive_optional_model_still_validates_present_values() -> None:
    """欠落を許すだけで、与えられた値の型検証は維持される (FR-2 L127)。"""
    derived = derive_optional_model(_simple_model())
    with pytest.raises(ValidationError):
        derived.model_validate_json('{"seconds": "not-an-int"}')


def test_derive_optional_model_drops_max_length_constraint() -> None:
    """parse 派生は max_length を落とし、上限超過でも ValidationError にならない (設計 §3.8)。"""
    source = build_frozen_model(
        "MultiSuggestionParams",
        _fields(seconds=(list[_Suggestion[int]], Field(description="up to 2", max_length=2))),
    )
    derived = derive_optional_model(source)
    instance = derived.model_validate_json(
        '{"seconds": [{"value": 1}, {"value": 2}, {"value": 3}]}'
    )
    assert len(instance.seconds) == 3


def test_derive_optional_model_has_no_max_items_in_json_schema() -> None:
    """parse 派生の model_json_schema() に maxItems が現れない (設計 §3.8)。"""
    source = build_frozen_model(
        "MultiSuggestionParams",
        _fields(seconds=(list[_Suggestion[int]], Field(max_length=2))),
    )
    schema = derive_optional_model(source).model_json_schema()
    assert "maxItems" not in str(schema["properties"]["seconds"])


# ---------------------------------------------------------------------------
# ドメイン型を import しない契約 (設計 §3.1 / NFR-1)
# ---------------------------------------------------------------------------


def test_models_module_has_no_relative_import() -> None:
    """`_models.py` は相対 import を 1 つも持たない (設計 §3.1 の循環回避契約)。

    `_models` がドメイン型 (`SlotSuggestion` など) を import すると `slots` との真の循環に
    なるため、annotation は呼び出し側が組み立てて引数で渡す契約になっている。
    """
    tree = ast.parse(_MODELS_PATH.read_text(encoding="utf-8"))
    relative = [
        node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level > 0
    ]
    assert relative == [], f"_models.py に相対 import があります: {[n.module for n in relative]}"


def test_models_module_imports_only_allowed_roots() -> None:
    """`_models.py` の import は標準ライブラリと pydantic に限られる (設計 §3.1 / NFR-1)。"""
    tree = ast.parse(_MODELS_PATH.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= _ALLOWED_IMPORT_ROOTS, f"想定外の import があります: {sorted(roots)}"


def test_models_module_source_has_no_sdk_reference() -> None:
    """`_models.py` のソースに agents / openai / _adapters が現れない (NFR-1)。"""
    source = _MODELS_PATH.read_text(encoding="utf-8")
    assert "from agents" not in source
    assert "import agents" not in source
    assert "import openai" not in source
    assert "_adapters" not in source


# ---------------------------------------------------------------------------
# build_frozen_model は入口で自衛する (セキュリティレビュー指摘 #88-4)
# ---------------------------------------------------------------------------
#
# 汎用ビルダであり、呼び出し側が宣言層 (`actions.py`) とは限らない。宣言層の検証が
# 効いていることに依存せず、自分の入口で name / フィールド名を検証する。


@pytest.mark.parametrize(
    ("name", "branch"),
    [
        ("", "空文字"),
        ("A B", "空白入り"),
        ("A B\n<script>", "改行と記号入り"),
        ("Params-1", "ハイフン入り"),
        ("1Params", "数字始まり"),
        ("Pa\x00rams", "NUL 入り"),
    ],
)
def test_build_frozen_model_rejects_non_identifier_name(name: str, branch: str) -> None:
    """name が識別子でなければ ValueError (指摘 #88-4a)。

    現状は素通しし、生成クラスの `__name__` と `model_json_schema()["title"]` に
    そのまま載る。スキーマは LLM / UI へ渡る出力面であるため、入口で落とす。
    """
    with pytest.raises(ValueError):
        build_frozen_model(name, _fields(target=(str, Field())))


@pytest.mark.parametrize("name", ["Params", "RunLoadTestParams", "NoParams", "P1"])
def test_build_frozen_model_still_accepts_identifier_names(name: str) -> None:
    """識別子の name は従来どおり受理される (指摘 #88-4a の回帰防止)。"""
    assert build_frozen_model(name, _fields(target=(str, Field()))).__name__ == name


@pytest.mark.parametrize(
    ("field_name", "branch"),
    [
        ("", "空文字"),
        ("a-b", "ハイフン入り"),
        ("a b", "空白入り"),
        ("1a", "数字始まり"),
        ("a\nb", "改行入り"),
    ],
)
def test_build_frozen_model_rejects_non_identifier_field_name(field_name: str, branch: str) -> None:
    """fields のキーが識別子でなければ ValueError (指摘 #88-4b)。

    `a-b` のような非識別子キーは現状そのままフィールド名になり、属性としては到達不能な
    フィールドが黙って生まれる。`ParameterSpec._validate_name` と同じ規則を入口に置く。
    """
    with pytest.raises(ValueError):
        build_frozen_model("Params", {field_name: (str, Field())})


@pytest.mark.parametrize("field_name", ["_secret", "_", "__module__", "__config__"])
def test_build_frozen_model_rejects_underscore_prefixed_field_name(field_name: str) -> None:
    """fields のキーが _ 始まりなら ValueError (指摘 #88-4b)。

    `_secret` は pydantic の `NameError` (「Fields must not use names with leading
    underscores」) が素通しで漏れ、`__module__` などの dunder は例外も警告もなく消えて
    フィールド 0 件のモデルになる。いずれも `ValueError` を契約する Raises 節に反する。
    """
    with pytest.raises(ValueError):
        build_frozen_model("Params", {field_name: (str, Field())})


@pytest.mark.parametrize("field_name", ["target", "seconds", "a1"])
def test_build_frozen_model_still_accepts_identifier_field_names(field_name: str) -> None:
    """識別子かつ _ 始まりでないキーは従来どおり受理される (指摘 #88-4b の回帰防止)。"""
    model = build_frozen_model("Params", {field_name: (str, Field())})
    assert set(model.model_fields) == {field_name}


# ---------------------------------------------------------------------------
# 例外文言の名前は repr 化する (セキュリティレビュー指摘 #88-5 / CWE-117)
# ---------------------------------------------------------------------------
#
# `_llm.py:148-154` が LLM 由来テキストを repr 化してからログへ載せているのと同じ理由。
# 例外文言はログにもエラーレスポンスにも載るため、制御文字・改行を生のまま通さない。


def test_build_frozen_model_value_error_reprs_the_field_name() -> None:
    """例外文言のフィールド名が repr 化されている (指摘 #88-5)。"""
    field_name = "bad\nname\x00\x1b[31m"
    with pytest.raises(ValueError) as excinfo:
        build_frozen_model("Params", {field_name: (_Unschemable, Field())})
    message = str(excinfo.value)
    assert field_name not in message, "フィールド名が生のまま例外文言へ載っています"
    assert repr(field_name) in message


def test_build_frozen_model_value_error_reprs_the_model_name() -> None:
    """例外文言のモデル名が repr 化されている (指摘 #88-5)。"""
    name = "Bad\nName\x00"
    with pytest.raises(ValueError) as excinfo:
        build_frozen_model(name, _fields(target=(str, Field())))
    message = str(excinfo.value)
    assert name not in message, "モデル名が生のまま例外文言へ載っています"
    assert repr(name) in message


# ---------------------------------------------------------------------------
# build_frozen_model: annotation の自衛 (生成層の多層防御)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("annotation", ["int", "str", "list[int]", "SomeUndefinedType"])
def test_build_frozen_model_rejects_str_annotation(annotation: str) -> None:
    """annotation が str の場合はビルダ自身が ValueError で拒否する。

    pydantic は str の annotation を前方参照として eval するため、型として受理すると
    任意の式が実行される。宣言層 (param) の検証だけでは、予測段のスキーマ生成のように
    param を経由せずビルダを直接呼ぶ経路が塞がらないため、生成層でも自衛する。
    """
    with pytest.raises(ValueError):
        build_frozen_model("Params", {"target": (annotation, Field())})


def test_build_frozen_model_does_not_evaluate_a_str_annotation() -> None:
    """str の annotation を渡してもその式が評価されない (副作用が起きない)。

    拒否するだけでは不十分で、拒否前に評価されていれば意味がない。現状は
    create_model と _unschemable_field_names の切り分けで 2 回評価される。
    """
    key = "OAI_PWNED_MODELS"
    expression = f"__import__('os').environ.setdefault({key!r}, '1')"
    os.environ.pop(key, None)
    try:
        with pytest.raises(ValueError):
            build_frozen_model("Params", {"target": (expression, Field())})
        assert key not in os.environ, "str annotation が評価され副作用が発生しています"
    finally:
        os.environ.pop(key, None)


def test_build_frozen_model_does_not_re_evaluate_a_str_annotation_while_isolating() -> None:
    """失敗時の切り分け経路でも str の annotation を評価し直さない。

    現状は 1 回目に `create_model()` が、2 回目に `_unschemable_field_names` の
    1 フィールドずつの切り分け再試行が評価するため、副作用が 2 回起きる。入口で弾けば
    どちらの経路にも到達しないことを、評価回数を数えるプローブで pin する。
    """
    key = "OAI_PWNED_MODELS_COUNT"
    expression = (
        "__import__('os').environ.__setitem__("
        f"{key!r}, str(int(__import__('os').environ.get({key!r}, '0')) + 1))"
    )
    os.environ.pop(key, None)
    try:
        with pytest.raises(ValueError):
            build_frozen_model(
                "Params", {"target": (expression, Field()), "seconds": (str, Field())}
            )
        evaluations = int(os.environ.get(key, "0"))
        assert evaluations == 0, f"str annotation が {evaluations} 回評価されました"
    finally:
        os.environ.pop(key, None)


def test_build_frozen_model_still_accepts_real_types_as_annotation() -> None:
    """型オブジェクトの annotation は従来どおり受理される (生成層の自衛の回帰防止)。"""
    model = build_frozen_model(
        "Params",
        _fields(
            target=(str, Field()),
            tags=(list[str], Field()),
            one=(_Suggestion[int], Field()),
        ),
    )
    assert model.model_fields["target"].annotation is str
    assert model.model_fields["tags"].annotation == list[str]


# ---------------------------------------------------------------------------
# derive_optional_model はパラメータ化 generic を扱える (セキュリティレビュー指摘 #88-6)
# ---------------------------------------------------------------------------
#
# `IntentQuery[dict]` のようにパラメータ化した pydantic generic は `__name__` が
# `"IntentQuery[dict]"` になる。派生名を `f"{model.__name__}Parse"` で組むと
# `"IntentQuery[dict]Parse"` となり、モデル名検証 (指摘 #88-4a) が識別子でないとして
# 落とす。予測段は generic なドメイン型から parse 派生を組むため、この経路が塞がると
# LLM 応答の検証そのものが不能になる。名前検証を緩めるのではなく、派生名を
# サニタイズして識別子へ落とし込むことで両立させる。

_TYPES_PATH = _SRC_DIR / "oai_agentspec" / "runtime" / "intent" / "types.py"


def test_types_module_imports_only_allowed_roots() -> None:
    """本節が使う `types.py` は標準ライブラリと pydantic しか import しない (L1 の前提)。

    パラメータ化 generic の実例として `IntentQuery[dict]` を使うため、その定義元が
    agents / openai へ依存していないことを先に固定する。依存が入ると本ファイルが L1
    (外部依存なし) でなくなる。
    """
    tree = ast.parse(_TYPES_PATH.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= frozenset({"__future__", "collections", "enum", "pydantic", "typing"}), (
        f"types.py に想定外の import があります: {sorted(roots)}"
    )


def test_derive_optional_model_accepts_a_parametrized_generic_source() -> None:
    """パラメータ化 generic を派生元にしても ValueError にならない (指摘 #88-6)。

    `IntentQuery[dict].__name__` は `"IntentQuery[dict]"` であり、派生名がそのまま
    `"IntentQuery[dict]Parse"` になるとモデル名検証で落ちる。予測段が generic な
    ドメイン型から parse 派生を組む経路を塞がないため、派生名をサニタイズする。
    """
    derived = derive_optional_model(IntentQuery[dict])
    assert issubclass(derived, BaseModel)
    assert set(derived.model_fields) == set(IntentQuery[dict].model_fields)


def test_derive_optional_model_name_is_an_identifier_for_a_parametrized_generic() -> None:
    """generic 由来の派生名も識別子になる (指摘 #88-6)。

    派生名は生成クラスの `__name__` と `model_json_schema()["title"]` に載り LLM / UI へ
    渡る出力面であるため、`[` `]` などの記号を残したまま通してはならない。
    """
    derived = derive_optional_model(IntentQuery[dict])
    assert derived.__name__.isidentifier(), f"派生名が識別子ではありません: {derived.__name__!r}"


def test_derive_optional_model_of_a_parametrized_generic_parses_partial_json() -> None:
    """generic 由来の派生も欠落フィールドを許す parse 用モデルとして機能する (指摘 #88-6)。"""
    derived = derive_optional_model(IntentQuery[dict])
    instance = derived.model_validate_json('{"utterance": "こんにちは"}')
    assert instance.utterance == "こんにちは"
    assert instance.run_context is None


def test_derive_optional_model_keeps_working_for_a_non_generic_source() -> None:
    """非 generic のドメイン型は従来どおり派生できる (指摘 #88-6 の回帰防止)。"""
    derived = derive_optional_model(IntentPrediction)
    assert derived.__name__.isidentifier()
    assert set(derived.model_fields) == set(IntentPrediction.model_fields)


def test_derive_optional_model_name_stays_unchanged_for_plain_names() -> None:
    """派生元名が既に識別子ならサニタイズは名前を変えない (指摘 #88-6 の回帰防止)。"""
    assert derive_optional_model(_simple_model("Params")).__name__ == "ParamsParse"


# ---------------------------------------------------------------------------
# build_frozen_model のフィールド名は Python 予約語を拒否する (指摘 #88-7)
# ---------------------------------------------------------------------------
#
# 宣言層 (`ParameterSpec._validate_name`) は identifier / `_` 始まり / 予約語の 3 分岐で
# 弾くのに、生成層は予約語だけ素通しする非対称がある。`class` をフィールド名にすると
# `instance.class` が SyntaxError となり属性アクセスで到達できず、宣言が黙って失われる。


@pytest.mark.parametrize("field_name", ["class", "None", "lambda", "return"])
def test_build_frozen_model_rejects_python_keyword_field_name(field_name: str) -> None:
    """fields のキーが Python 予約語なら ValueError (指摘 #88-7)。

    `str.isidentifier()` は予約語に対して True を返すため現状は素通しし、属性として
    到達できないフィールドが生まれる。宣言層の 3 分岐と同一の規則を生成層にも置く。
    """
    with pytest.raises(ValueError):
        build_frozen_model("Params", {field_name: (str, Field())})


# ---------------------------------------------------------------------------
# build_frozen_model のフィールド名は str 以外を ValueError で拒否する (指摘 #88-8)
# ---------------------------------------------------------------------------
#
# 現状は非 str のキーが `field_name.isidentifier()` で `AttributeError` になり、
# `ValueError` を契約する Raises 節に反する。呼び出し側は `ValueError` だけを捕捉して
# 宣言エラーとして扱うため、`AttributeError` は捕捉網を抜けて上位まで素通しする。


@pytest.mark.parametrize("field_name", [1, None, (1,), b"target", 1.5])
def test_build_frozen_model_rejects_non_str_field_name(field_name: Any) -> None:
    """fields のキーが str でなければ ValueError (指摘 #88-8)。"""
    with pytest.raises(ValueError):
        build_frozen_model("Params", {field_name: (str, Field())})


@pytest.mark.parametrize("field_name", [1, None, (1,)])
def test_build_frozen_model_non_str_field_name_does_not_raise_attribute_error(
    field_name: Any,
) -> None:
    """非 str のキーで AttributeError が漏れない (指摘 #88-8)。

    `ValueError` の部分型でない `AttributeError` は、宣言エラーとして `ValueError` のみを
    捕捉する呼び出し側の網を抜ける。型の異なる例外へ化けないことを明示的に pin する。
    """
    with pytest.raises(ValueError) as excinfo:
        build_frozen_model("Params", {field_name: (str, Field())})
    assert not isinstance(excinfo.value, AttributeError)


# ---------------------------------------------------------------------------
# build_frozen_model のモデル名はフィールド名と同じ規則で検証する (指摘 #88-9)
# ---------------------------------------------------------------------------
#
# モデル名は `str.isidentifier()` しか見ていないため、`_Secret` や `class` が通る。
# フィールド名側は `_` 始まりを弾いており規則が非対称である。モデル名も生成クラスの
# `__name__` として参照される公開識別子であり、`_` 始まりは非公開の見た目で、予約語は
# 名前として書き下せない。同一の 3 分岐へ揃える。


@pytest.mark.parametrize("name", ["class", "None", "lambda", "return"])
def test_build_frozen_model_rejects_python_keyword_name(name: str) -> None:
    """name が Python 予約語なら ValueError (指摘 #88-9)。"""
    with pytest.raises(ValueError):
        build_frozen_model(name, _fields(target=(str, Field())))


@pytest.mark.parametrize("name", ["_Secret", "__init__", "_", "_Params"])
def test_build_frozen_model_rejects_underscore_prefixed_name(name: str) -> None:
    """name が `_` 始まりなら ValueError (指摘 #88-9)。

    フィールド名側は `_` 始まりを弾いているのにモデル名側は素通しする非対称があり、
    生成クラスが `_Secret` のような非公開に見える名前で `model_json_schema()["title"]`
    へ載る。フィールド名と同一の規則へ揃える。
    """
    with pytest.raises(ValueError):
        build_frozen_model(name, _fields(target=(str, Field())))


@pytest.mark.parametrize("name", ["Params", "P1", "ParamsParse", "パラメータ"])
def test_build_frozen_model_still_accepts_public_identifier_names(name: str) -> None:
    """`_` 始まりでも予約語でもない識別子は従来どおり受理される (指摘 #88-9 の回帰防止)。"""
    assert build_frozen_model(name, _fields(target=(str, Field()))).__name__ == name


@pytest.mark.parametrize("name", [1, None, (1,), b"Params", 1.5])
def test_build_frozen_model_rejects_non_str_model_name(name: Any) -> None:
    """モデル名が str でなければ ValueError (フィールド名側と規則を揃える)。

    str でない値は isidentifier() の呼び出しが AttributeError になり、ValueError だけを
    捕捉する呼び出し側の網を抜ける。フィールド名側 (指摘 #88-8) と同型の契約違反。
    """
    with pytest.raises(ValueError):
        build_frozen_model(name, {"target": (str, Field())})


@pytest.mark.parametrize("name", [1, None, (1,)])
def test_build_frozen_model_non_str_model_name_does_not_raise_attribute_error(
    name: Any,
) -> None:
    """非 str のモデル名で AttributeError が漏れない (契約は ValueError)。"""
    with pytest.raises(ValueError):
        build_frozen_model(name, {"target": (str, Field())})


def test_derive_optional_model_strips_leading_underscore_from_a_plain_source() -> None:
    """ブラケットを持たない _ 始まりのクラスからも公開識別子の派生名を組む。

    派生名は _validate_model_name の 3 分岐 (識別子 / _ 始まり / 予約語) をすべて
    満たす必要がある。記号を落とすだけの実装だと _Secret が _SecretParse となり
    自分自身の検証に落ちる。
    """

    class _Secret(BaseModel):
        model_config = {"frozen": True}
        x: int = 1

    derived = derive_optional_model(_Secret)
    assert derived.__name__.isidentifier()
    assert not derived.__name__.startswith("_")
    assert derived.__name__ == "SecretParse"


# ---------------------------------------------------------------------------
# ネストした前方参照も拒否する (指摘 #88-11)
# ---------------------------------------------------------------------------
#
# `_validate_annotation` は annotation 全体が str かどうかしか見ないため、`list["..."]` /
# `Optional["..."]` / `ForwardRef("...")` のように 1 段でも包むと素通しし、`create_model()`
# が前方参照として eval する。さらに評価結果がスキーマ化できないと
# `_unschemable_field_names` の切り分けが同じ annotation をもう一度組み直すため、任意式が
# 2 回実行される。検査は annotation の内側まで再帰しなければならない。
#
# ただし再帰は素朴に `get_args` を辿るだけでは誤検知する。`Literal["a", "b"]` の引数は値の
# str であり型ではなく、`Annotated[int, "note"]` の第 2 引数以降はメタデータである。どちらも
# 拒否すると正当な宣言が通らなくなるため、通す側も併せて pin する。

#: 評価されると `_evil` の NameError になるプローブ式。評価しない限り無害な str である。
_FORWARD_REF_PROBE = "_evil()"

_NESTED_FORWARD_REFS = [
    pytest.param(list[_FORWARD_REF_PROBE], id="list"),
    pytest.param(ForwardRef(_FORWARD_REF_PROBE), id="forward-ref"),
    pytest.param(Optional[_FORWARD_REF_PROBE], id="optional"),  # noqa: UP045
    pytest.param(dict[str, _FORWARD_REF_PROBE], id="dict-value"),
    pytest.param(tuple[_FORWARD_REF_PROBE, ...], id="tuple-ellipsis"),
    pytest.param(list[list[_FORWARD_REF_PROBE]], id="list-of-list"),
]


@pytest.mark.parametrize("annotation", _NESTED_FORWARD_REFS)
def test_build_frozen_model_rejects_a_nested_forward_ref(annotation: Any) -> None:
    """annotation の内側に前方参照が 1 つでもあれば ValueError (指摘 #88-11)。

    最上位の str しか見ない検査は `list["<式>"]` のように 1 段包むだけで迂回でき、
    pydantic が任意式として eval する経路がそのまま残る。
    """
    with pytest.raises(ValueError):
        build_frozen_model("Params", {"target": (annotation, Field())})


def test_build_frozen_model_does_not_evaluate_a_nested_forward_ref() -> None:
    """ネストした前方参照を渡してもその式が評価されない (指摘 #88-11)。

    拒否するだけでは不十分で、拒否前に評価されていれば任意式の実行を許したことになる。
    評価されるたびに 1 ずつ増えるカウンタ式を渡し、増分が 0 であることで未評価を検知する。
    """
    key = "OAI_PWNED_MODELS_NESTED"
    expression = (
        "__import__('os').environ.__setitem__("
        f"{key!r}, str(int(__import__('os').environ.get({key!r}, '0')) + 1))"
    )
    os.environ.pop(key, None)
    try:
        with pytest.raises(ValueError):
            build_frozen_model("Params", {"target": (list[expression], Field())})
        evaluations = int(os.environ.get(key, "0"))
        assert evaluations == 0, f"ネストした前方参照が {evaluations} 回評価されました"
    finally:
        os.environ.pop(key, None)


def test_build_frozen_model_does_not_re_evaluate_a_nested_forward_ref_while_isolating() -> None:
    """失敗時の切り分け経路でもネストした前方参照を評価し直さない (指摘 #88-11)。

    評価結果がスキーマ化できない場合、1 回目に `create_model()` が、2 回目に
    `_unschemable_field_names` の 1 フィールドずつの再試行が評価する。入口で弾けば
    どちらの経路にも到達しないことを、評価回数を数えるプローブで pin する。
    """
    key = "OAI_PWNED_MODELS_NESTED_COUNT"
    expression = (
        "__import__('os').environ.__setitem__("
        f"{key!r}, str(int(__import__('os').environ.get({key!r}, '0')) + 1)) or object()"
    )
    os.environ.pop(key, None)
    try:
        with pytest.raises(ValueError):
            build_frozen_model(
                "Params",
                {"target": (list[expression], Field()), "seconds": (str, Field())},
            )
        evaluations = int(os.environ.get(key, "0"))
        assert evaluations == 0, f"ネストした前方参照が {evaluations} 回評価されました"
    finally:
        os.environ.pop(key, None)


_NESTED_FORWARD_REF_FREE_ANNOTATIONS = [
    # 予測段が組み立てる SlotSuggestion[T] 相当。パラメータ化済み pydantic generic は
    # 具象クラスであり get_args() が () を返すため、辿る対象そのものが無い。
    pytest.param(_Suggestion[int], id="generic"),
    pytest.param(list[_Suggestion[int]], id="list-generic"),
    pytest.param(_Suggestion[int] | None, id="generic-optional"),
    pytest.param(list[_Suggestion[int]] | None, id="list-generic-optional"),
    # Literal の引数は型ではなく値の str である。辿ると必ず前方参照と誤認する。
    pytest.param(Literal["a", "b"], id="literal"),
    pytest.param(Literal["a"] | None, id="literal-optional"),
    pytest.param(list[Literal["a", "b"]], id="list-literal"),
    pytest.param(dict[str, Literal["a"]], id="dict-literal"),
    # Annotated の第 2 引数以降はメタデータであり、str の注記を置くのが通常の使い方。
    pytest.param(Annotated[int, "note"], id="annotated-str-metadata"),
    pytest.param(Annotated[list[str], Field(description="d")], id="annotated-field-metadata"),
    pytest.param(dict[str, int], id="dict"),
    pytest.param(list[str], id="list-str"),
    pytest.param(int | None, id="optional-int"),
    pytest.param(IntentPrediction, id="domain-model"),
    pytest.param(IntentQuery[dict], id="domain-generic"),
]


@pytest.mark.parametrize("annotation", _NESTED_FORWARD_REF_FREE_ANNOTATIONS)
def test_build_frozen_model_still_accepts_nested_annotations_without_forward_refs(
    annotation: Any,
) -> None:
    """前方参照を含まない合成 annotation は従来どおり受理される (指摘 #88-11 の回帰防止)。

    再帰検査が Literal の値や Annotated のメタデータを前方参照と誤認すると、正当な宣言と
    予測段のスキーマ生成が両方塞がる。誤検知しないことを拒否側と対で pin する。
    """
    model = build_frozen_model("Params", {"target": (annotation, Field())})
    assert "target" in model.model_fields


def test_build_frozen_model_still_accepts_a_mixed_field_map_without_forward_refs() -> None:
    """複数フィールドを混在させても再帰検査が誤検知しない (指摘 #88-11 の回帰防止)。"""
    model = build_frozen_model(
        "Params",
        _fields(
            target=(str, Field()),
            mode=(Literal["fast", "safe"], Field()),
            note=(Annotated[str, "free text"], Field()),
            one=(_Suggestion[int], Field()),
            many=(list[_Suggestion[int]] | None, Field(default=None)),
        ),
    )
    assert set(model.model_fields) == {"target", "mode", "note", "one", "many"}


# ---------------------------------------------------------------------------
# derive_optional_model は再帰検査を挟んでも既存型で壊れない (指摘 #88-11 の回帰防止)
# ---------------------------------------------------------------------------
#
# derive_optional_model は全フィールドを `X | None` へ包み直してから build_frozen_model を
# 呼ぶ。再帰検査はその包み直した annotation を通るため、誤検知するとドメイン型からの
# parse 派生 (LLM 応答の検証そのもの) が丸ごと不能になる。


@pytest.mark.parametrize("source", [IntentPrediction, IntentQuery[dict]])
def test_derive_optional_model_still_works_for_existing_domain_types(
    source: type[BaseModel],
) -> None:
    """既存のドメイン型からの parse 派生が再帰検査で塞がらない (指摘 #88-11 の回帰防止)。"""
    derived = derive_optional_model(source)
    assert set(derived.model_fields) == set(source.model_fields)


def test_derive_optional_model_still_works_for_literal_and_generic_fields() -> None:
    """Literal / Annotated / generic を持つモデルからも派生できる (指摘 #88-11 の回帰防止)。

    `X | None` へ包み直すと Literal も Annotated も Union の内側へ入る。再帰検査が
    そこで誤検知すると、派生元が正当なモデルでも ValueError になる。
    """
    source = build_frozen_model(
        "Params",
        _fields(
            mode=(Literal["fast", "safe"], Field(description="モード")),
            note=(Annotated[str, "free text"], Field()),
            many=(list[_Suggestion[int]], Field()),
        ),
    )
    derived = derive_optional_model(source)
    assert set(derived.model_fields) == {"mode", "note", "many"}
    assert derived.model_validate_json("{}").mode is None


# ---------------------------------------------------------------------------
# Callable の引数リストに隠れた前方参照も拒否する (セキュリティ BLOCKER)
# ---------------------------------------------------------------------------
#
# `get_args(Callable[["expr"], int])` は `([ForwardRef("expr")], int)` を返し、第 1 要素が
# **list** である。`_has_forward_ref` は str / ForwardRef / Literal / Annotated / それ以外
# (get_args を再帰) の 4 分岐しか持たず、list の中身を辿らないため False を返す。結果として
# 前方参照が pydantic へ渡り eval される (`Callable` は宣言で普通に書ける型であり、
# 「get_args に現れない場所へ隠した前方参照」という許容済みの残余ではない)。
#
# 戻り値側 (`Callable[..., "expr"]`) は `get_args` が `(Ellipsis, ForwardRef("expr"))` を
# 返すため現状でも検知される。同型の見落としを対で pin するため併せて置く。


def _probe_expression(key: str) -> str:
    """評価されると環境変数 `key` を立てるプローブ式を組む。

    評価しない限りただの str であり、危険なコマンドは一切実行しない
    (`os.system` / `subprocess` を含まない)。

    Args:
        key: 評価の痕跡として立てる環境変数名。

    Returns:
        前方参照として `eval` された時にのみ副作用を起こす式の文字列。
    """
    return f"__import__('os').environ.setdefault({key!r}, '1')"


#: 評価されると `_evil` の NameError になるプローブ式 (Callable の引数リスト検査用)。
_CALLABLE_PROBE = _FORWARD_REF_PROBE

_CALLABLE_FORWARD_REFS = [
    pytest.param(Callable[[_CALLABLE_PROBE], int], id="callable-argument-list"),
    pytest.param(Callable[..., _CALLABLE_PROBE], id="callable-return"),
    pytest.param(Callable[[int, _CALLABLE_PROBE], int], id="callable-second-argument"),
    pytest.param(list[Callable[[_CALLABLE_PROBE], int]], id="list-of-callable"),
    pytest.param(dict[str, Callable[[_CALLABLE_PROBE], int]], id="dict-value-callable"),
    pytest.param(Callable[[_CALLABLE_PROBE], int] | None, id="callable-optional"),
]


@pytest.mark.parametrize("annotation", _CALLABLE_FORWARD_REFS)
def test_build_frozen_model_rejects_a_forward_ref_inside_callable(annotation: Any) -> None:
    """`Callable` の内側に前方参照があれば ValueError (セキュリティ BLOCKER)。

    `Callable` の引数リストは `get_args` の第 1 要素が list であり、再帰が list の中身を
    辿らないと素通りする。ネスト検査 (指摘 #88-11) と同じ契約が `Callable` にも及ぶ。
    """
    with pytest.raises(ValueError, match="is evaluated by pydantic") as excinfo:
        build_frozen_model("Params", {"target": (annotation, Field())})
    assert type(excinfo.value) is ValueError


def test_build_frozen_model_does_not_evaluate_a_forward_ref_in_a_callable_argument_list() -> None:
    """`Callable` の引数リストに置いた前方参照が評価されない (セキュリティ BLOCKER)。

    「例外が出る」だけでは不十分で、拒否より前に評価されていれば任意式の実行を許した
    ことになる。評価された時にだけ立つ環境変数で副作用の不在を検知する。
    """
    key = "OAI_PWNED_MODELS_CALLABLE_ARG"
    annotation = Callable[[_probe_expression(key)], int]
    os.environ.pop(key, None)
    try:
        with pytest.raises(ValueError, match="is evaluated by pydantic"):
            build_frozen_model("Params", {"target": (annotation, Field())})
        assert key not in os.environ, "Callable の引数リストの前方参照が評価されています"
    finally:
        os.environ.pop(key, None)


def test_build_frozen_model_does_not_evaluate_a_forward_ref_nested_under_callable() -> None:
    """`list[Callable[["expr"], int]]` のように包んでも評価されない (セキュリティ BLOCKER)。"""
    key = "OAI_PWNED_MODELS_CALLABLE_NESTED"
    annotation = list[Callable[[_probe_expression(key)], int]]
    os.environ.pop(key, None)
    try:
        with pytest.raises(ValueError, match="is evaluated by pydantic"):
            build_frozen_model("Params", {"target": (annotation, Field())})
        assert key not in os.environ, "入れ子にした Callable の前方参照が評価されています"
    finally:
        os.environ.pop(key, None)


_CALLABLE_FORWARD_REF_FREE_ANNOTATIONS = [
    pytest.param(Callable[[int], str], id="callable-int-to-str"),
    pytest.param(Callable[[], None], id="callable-no-argument"),
    pytest.param(Callable[..., int], id="callable-ellipsis"),
    pytest.param(Callable[[int, str], bool], id="callable-two-arguments"),
    pytest.param(list[Callable[[int], str]], id="list-of-callable"),
    pytest.param(Callable[[int], str] | None, id="callable-optional"),
    pytest.param(Callable[[Callable[[int], str]], str], id="callable-of-callable"),
]


@pytest.mark.parametrize("annotation", _CALLABLE_FORWARD_REF_FREE_ANNOTATIONS)
def test_build_frozen_model_still_accepts_a_callable_without_forward_refs(
    annotation: Any,
) -> None:
    """前方参照を含まない `Callable` は従来どおり受理される (誤検知の防止)。

    これが無いと、修正が「`Callable` を一律拒否する」方向へ倒れても拒否側だけが緑になる。
    """
    model = build_frozen_model("Params", {"target": (annotation, Field())})
    assert "target" in model.model_fields


# 宣言層 (`param`) 側の入口。`ParameterSpec` の field_validator が同じ `_has_forward_ref` を
# 使うため、`Callable` の見落としは宣言層でもそのまま素通りする。pydantic の validator 内で
# 送出した `ValueError` は `ValidationError` (ValueError 派生) へ包まれる。


@pytest.mark.parametrize("annotation", _CALLABLE_FORWARD_REFS)
def test_param_rejects_a_forward_ref_inside_callable(annotation: Any) -> None:
    """`param` も `Callable` の内側の前方参照を拒否する (セキュリティ BLOCKER)。"""
    with pytest.raises(ValidationError, match="invalid parameter annotation") as excinfo:
        param("cb", annotation)
    assert type(excinfo.value) is ValidationError


def test_param_does_not_evaluate_a_forward_ref_in_a_callable_argument_list() -> None:
    """`param` 経由でも `Callable` 内の前方参照が評価されない (セキュリティ BLOCKER)。

    宣言層で弾けなければ `ActionSpec.parameters_model()` が `create_model()` へ渡し、
    その時点で式が実行される。宣言の受理そのものを拒否することで到達させない。
    """
    key = "OAI_PWNED_ACTIONS_CALLABLE_ARG"
    annotation = Callable[[_probe_expression(key)], int]
    os.environ.pop(key, None)
    try:
        with pytest.raises(ValidationError, match="invalid parameter annotation"):
            param("cb", annotation)
        assert key not in os.environ, "param 経由で Callable 内の前方参照が評価されています"
    finally:
        os.environ.pop(key, None)


@pytest.mark.parametrize("annotation", _CALLABLE_FORWARD_REF_FREE_ANNOTATIONS)
def test_param_still_accepts_a_callable_without_forward_refs(annotation: Any) -> None:
    """前方参照を含まない `Callable` は `param` でも従来どおり宣言できる (誤検知の防止)。"""
    spec = param("cb", annotation)
    assert spec.annotation == annotation


# ---------------------------------------------------------------------------
# BaseModel の予約属性と衝突する名前を拒否する (セキュリティ BLOCKER)
# ---------------------------------------------------------------------------
#
# `model_config` は「非 str / 非識別子 / `_` 始まり / 予約語」の 4 分岐をすべて通過するが、
# `create_model` はフィールド定義をクラス名前空間へ入れるため、pydantic の `ModelMetaclass`
# が `model_config` を**設定**として解釈しフィールドを捨てる。例外も警告も出ない。
# `model_fields` / `model_computed_fields` は `UserWarning`（"shadows an attribute in parent
# BaseModel"）が出るだけでフィールドとしては残る、同じ系統の取りこぼしである。
#
# 実害: 候補が値を載せても `plan.input_json` から消える一方、`plan.slots` には当該 Slot が
# RESOLVED で残り `ready` も True になる。「確定済みの値が実行入力にだけ入っていない」
# silent data loss になる。`_validate_field_name` の docstring 自身が「`create_model` の
# 予約引数名と衝突して例外も警告もなく消える」ことを塞ぐと宣言しており、その保護が
# dunder のみで不完全であることが本欠陥である。

_RESERVED_MODEL_ATTRIBUTE_NAMES = [
    pytest.param("model_config", id="model-config-silently-dropped"),
    pytest.param("model_fields", id="model-fields-shadows-attribute"),
    pytest.param("model_computed_fields", id="model-computed-fields-shadows-attribute"),
]

#: `model` で始まるだけの正当な名前。予約属性と衝突せずフィールドとして機能する。
_NON_RESERVED_NAMES = ["model", "config", "model_name", "model_id", "configuration", "env"]


@pytest.mark.parametrize("field_name", _RESERVED_MODEL_ATTRIBUTE_NAMES)
def test_build_frozen_model_rejects_a_reserved_base_model_attribute_name(field_name: str) -> None:
    """`BaseModel` の予約属性と衝突するフィールド名は ValueError (セキュリティ BLOCKER)。

    生成層は宣言層と独立した多層防御であり、`param` を経由しない直接呼び出し
    （予測段のスキーマ生成）も同じ規則で守られる必要がある。
    """
    with pytest.raises(ValueError, match="invalid field name") as excinfo:
        build_frozen_model("Params", {field_name: (str, Field()), "env": (str, Field())})
    assert type(excinfo.value) is ValueError
    assert repr(field_name) in str(excinfo.value)


@pytest.mark.parametrize("field_name", _RESERVED_MODEL_ATTRIBUTE_NAMES)
def test_build_frozen_model_does_not_silently_drop_a_reserved_name(field_name: str) -> None:
    """予約属性名の宣言が「黙って消える」形で通過しない (セキュリティ BLOCKER)。

    「例外が出る」だけでなく「宣言したフィールドが欠けたモデルが返らない」ことを pin する。
    現状は `model_config` が例外も警告もなくフィールド集合から消え、宣言と生成物が食い違う。
    """
    try:
        model = build_frozen_model("Params", {field_name: (str, Field()), "env": (str, Field())})
    except ValueError:
        return
    assert set(model.model_fields) == {field_name, "env"}, (
        f"宣言した {field_name!r} が例外も警告もなく生成モデルから消えています"
    )


@pytest.mark.parametrize("field_name", _NON_RESERVED_NAMES)
def test_build_frozen_model_still_accepts_names_that_only_look_reserved(field_name: str) -> None:
    """`model` / `config` / `model_name` などは従来どおり受理される (誤検知の防止)。

    これが無いと、修正が「`model` で始まる名前を一律拒否する」方向へ倒れても緑になる。
    """
    model = build_frozen_model("Params", {field_name: (str, Field())})
    assert set(model.model_fields) == {field_name}
    assert model(**{field_name: "v"}).model_dump() == {field_name: "v"}


@pytest.mark.parametrize("field_name", _RESERVED_MODEL_ATTRIBUTE_NAMES)
def test_param_rejects_a_reserved_base_model_attribute_name(field_name: str) -> None:
    """`param` も `BaseModel` の予約属性と衝突する名前を拒否する (セキュリティ BLOCKER)。

    宣言層で弾けば、宣言と `parameters_model()` の生成物が食い違う状態そのものが作れない。
    """
    with pytest.raises(ValidationError, match="invalid parameter name") as excinfo:
        param(field_name, str)
    assert type(excinfo.value) is ValidationError
    assert repr(field_name) in str(excinfo.value)


@pytest.mark.parametrize("field_name", _NON_RESERVED_NAMES)
def test_param_still_accepts_names_that_only_look_reserved(field_name: str) -> None:
    """予約属性と紛らわしいだけの名前は `param` でも従来どおり宣言できる (誤検知の防止)。"""
    assert param(field_name, str).name == field_name
