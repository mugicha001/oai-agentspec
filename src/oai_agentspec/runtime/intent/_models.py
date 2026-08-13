"""frozen な pydantic モデルを実行時生成する汎用ビルダ（`create_model` の薄い包み）。

2 つの内訳: 宣言から frozen モデルを組む `build_frozen_model` / 既存モデルから全フィールドを
`X | None`（既定 `None`）にした parse 用派生を組む `derive_optional_model`。

`ActionSpec.parameters_model()`（公開）と予測段のスキーマモデル（内部）は、どちらも
`create_model` 呼び出しと `PydanticSchemaGenerationError` の変換を必要とする。この実装を
宣言型の置き場（`actions.py` / `slots.py`）へ混ぜないための分離モジュールである。

方針:
- **ドメイン型を 1 つも import しない。** 予測段のフィールド型は `SlotSuggestion[T]` /
  `list[SlotSuggestion[T]]` であり、`SlotSuggestion` は `slots.py` に置かれる。本モジュールが
  この型を自分で組み立てると `_models <-> slots` の真の循環になるため、annotation は
  呼び出し側が組み立てて引数で渡す契約とする（設計 §3.1）。
- `PydanticSchemaGenerationError` は `RuntimeError` 派生であり素通しでは `ValueError` に
  ならない。`create_model()` の呼び出し時点で送出されるため、`try` は
  `create_model()` そのものへ掛ける（設計 §3.9）。
- parse 用派生には `max_length` を持ち込まない。上限超過を `ValidationError` にすると
  LLM 応答が 1 件超えただけで全体が落ち、後退判断の材料が失われる（設計 §3.8）。
- 生成結果はキャッシュしない。「同一クラスオブジェクトを返す」契約は呼び出し側
  （`ActionSpec.parameters_model()`）の責務である。
- **入口で自衛する。** 汎用ビルダであり呼び出し側が宣言層（`actions.py`）とは限らない
  （予測段のスキーマ生成は `param()` を経由せず本モジュールを直接呼ぶ）。宣言層の検証が
  効いていることに依存せず、モデル名・フィールド名・annotation を自分の入口で検証する。
- **annotation に前方参照を受け取らない。** pydantic は `str` / `ForwardRef` を `eval` する
  ため、型として受理すると任意式の実行経路になる。`create_model()` へ渡す**前**に
  `_has_forward_ref` で内側まで検査して弾き、失敗時の切り分け経路にも到達させない。
  ただしこれは完全な保証ではなく、宣言で普通に書ける範囲のネストを塞ぐ多層防御である
  （`get_args` に現れない場所へ隠した前方参照は残る。詳細は `_has_forward_ref` の Note）。
- 例外文言へ載せる名前は `repr()` 化する。制御文字・改行を生のままログやエラー応答へ
  流さないため（CWE-117・`_llm.py:148-154` と同じ理由）。
"""

from __future__ import annotations

import keyword
from collections.abc import Mapping
from typing import Annotated, Any, Final, ForwardRef, Literal, get_args, get_origin

from pydantic import BaseModel, Field, create_model
from pydantic.errors import PydanticSchemaGenerationError
from pydantic.fields import FieldInfo

#: pydantic が `BaseModel` のクラス名前空間で特別扱いする属性名。`create_model` はフィールド
#: 定義をクラス名前空間へ入れるため、これらの名前は「フィールドとして宣言したのに生成物へ
#: 現れない」経路になる。`model_config` は `ModelMetaclass` が**設定**として解釈しフィールドを
#: 例外も警告もなく捨て、`model_fields` / `model_computed_fields` は
#: `UserWarning: Field name "..." shadows an attribute in parent "BaseModel"` を伴う。
#: 接頭辞で一律に弾かないのは、`model` / `model_name` のような正当な名前を誤検知しないため。
_RESERVED_MODEL_ATTRIBUTE_NAMES: Final[frozenset[str]] = frozenset(
    {"model_config", "model_fields", "model_computed_fields"}
)

#: `_RESERVED_MODEL_ATTRIBUTE_NAMES` との衝突を伝える文言（フィールド名・パラメータ名で共有）。
_RESERVED_MODEL_ATTRIBUTE_REASON: Final[str] = (
    "collides with an attribute pydantic reserves on BaseModel"
)


def build_frozen_model(
    name: str,
    fields: Mapping[str, tuple[Any, FieldInfo]],
) -> type[BaseModel]:
    """annotation と `FieldInfo` の対応表から frozen な `BaseModel` サブクラスを組む。

    Args:
        name: 生成するクラスの名前（`model_json_schema()` の `title` にもなる）。
        fields: フィールド名 -> `(annotation, FieldInfo)` の対応表。annotation は
            呼び出し側が組み立てたものをそのまま使う（本モジュールはドメイン型を
            知らない・設計 §3.1）。空の対応表を渡すとフィールド 0 件のモデルになる。

    Returns:
        `fields` をフィールドに持つ frozen な `BaseModel` サブクラス。呼び出しごとに
        新しいクラスオブジェクトを返す（キャッシュしない）。

    Raises:
        ValueError: name が公開識別子でない場合（`str.isidentifier()` 偽 / `_` 始まり /
            Python 予約語。生成クラスの `__name__` と `model_json_schema()` の `title` に
            載り LLM / UI へ渡る出力面であるため入口で落とす）、フィールド名が str でない
            か公開識別子でない場合（同じ 3 分岐）、annotation が `str`（前方参照）の場合、
            または annotation の 1 つ以上を
            pydantic がフィールド型として扱えない場合。最後のケースは元の
            `PydanticSchemaGenerationError` が `RuntimeError` 派生で `ValueError` に
            ならないため、該当フィールド名を添えて変換し `raise ... from exc` で連鎖させる。
    """
    _validate_model_name(name)
    for field_name, (annotation, _info) in fields.items():
        _validate_field_name(name, field_name)
        _validate_annotation(name, field_name, annotation)
    definitions: dict[str, Any] = dict(fields)
    try:
        return create_model(name, __config__={"frozen": True}, **definitions)
    except PydanticSchemaGenerationError as exc:
        offenders = _unschemable_field_names(name, fields)
        raise ValueError(
            f"cannot build model {name!r}: pydantic cannot use the declared annotation as a "
            f"field type for: {', '.join(repr(offender) for offender in offenders)}"
        ) from exc


def derive_optional_model(model: type[BaseModel]) -> type[BaseModel]:
    """既存モデルから、全フィールドを `X | None`（既定 `None`）にした parse 用派生を組む。

    LLM 応答の検証に使う派生であり、`Field(description=...)` のみを引き継いで
    `max_length` などの制約は落とす（設計 §3.8）。これにより一部フィールドが欠落した
    JSON も、上限を超えた件数を含む JSON も `model_validate_json` を通り、切り捨てや
    後退の判断を呼び出し側が行えるようになる。

    Args:
        model: 派生元のモデル。本関数は派生元を変更しない。

    Returns:
        全フィールドが任意（既定 `None`）の frozen な `BaseModel` サブクラス。

    Raises:
        ValueError: 派生元のフィールド型を `X | None` にしたものを pydantic が扱えない場合。
    """
    fields: dict[str, tuple[Any, FieldInfo]] = {
        field_name: (
            info.annotation | None,
            Field(default=None, description=info.description),
        )
        for field_name, info in model.model_fields.items()
    }
    return build_frozen_model(f"{_name_stem(model.__name__)}Parse", fields)


def _name_stem(source_name: str) -> str:
    """派生モデル名の語幹を、`_validate_model_name` の全分岐を満たす形へ正規化する。

    パラメータ化した pydantic generic の `__name__` は `IntentQuery[dict]` のように記号を
    含み、そのまま `f"{...}Parse"` にするとモデル名検証で落ちる。名前検証を緩めるのでは
    なく、派生名の側を識別子へ落とし込むことで両立させる。

    規則:

    - 既に公開識別子（識別子かつ `_` 始まりでない）ならそのまま返す。`Params` ->
      `Params`（呼び出し側で `ParamsParse` になる）で、綴りも大小も変えない。
    - そうでなければ英数字と `_` 以外を区切りとして分割し、各断片の先頭の `_` を落として
      先頭 1 文字を大文字化し、連結する。`IntentQuery[dict]` -> `IntentQueryDict` /
      `IntentQuery[str]` -> `IntentQueryStr`（型引数名を残すため両者は衝突しない）。
      `_Secret[int]` -> `SecretInt`（`_` 始まりの拒否と衝突しない）。
    - 連結結果が識別子にならない（空、または数字始まり）場合のみ `Model` を前置する。

    語幹に `Parse` が付くため、結果が Python 予約語になることはない。

    Args:
        source_name: 派生元モデルの `__name__`。

    Returns:
        公開識別子として使える語幹。
    """
    if source_name.isidentifier() and not source_name.startswith("_"):
        return source_name
    segments = [
        trimmed[0].upper() + trimmed[1:]
        for part in _split_on_symbols(source_name)
        if (trimmed := part.lstrip("_"))
    ]
    stem = "".join(segments)
    return stem if stem.isidentifier() else f"Model{stem}"


def _split_on_symbols(source_name: str) -> list[str]:
    """英数字と `_` 以外を区切りとして名前を分割する（`re` を使わない・NFR-6）。

    Args:
        source_name: 分割対象の名前。

    Returns:
        区切り文字を含まない断片のリスト（空の断片は含まない）。
    """
    parts: list[str] = []
    current: list[str] = []
    for char in source_name:
        if char.isalnum() or char == "_":
            current.append(char)
        elif current:
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))
    return parts


def _check_public_identifier(
    value: Any,
    *,
    what: str,
    reserved: frozenset[str] = frozenset(),
    reserved_reason: str = "collides with a reserved name",
) -> None:
    """名前が到達可能な公開識別子であり予約集合と衝突しないことを検証する。

    到達不能名の判定はモデル名・フィールド名・パラメータ名・`action_id` の 4 箇所で必要に
    なる。分岐本体を各所へ複製すると「1 箇所直しても他が残る」形になるため（`model_config`
    の取りこぼしがその実例）、分岐は本関数だけが持ち、呼び出し側は文言（`what`）と予約集合
    だけを渡す。`re` は使わない（NFR-6）。

    Args:
        value: 検証対象の名前。`str` でない値もそのまま受け取る。
        what: 文言に載せる名前の種別（`"model name"` / `"parameter name"` など）。
        reserved: 追加で拒否する名前の集合。既定は空集合。
        reserved_reason: 予約集合と衝突したときに文言へ載せる理由。

    Raises:
        ValueError: `str` でない場合（`isidentifier()` の呼び出しが `AttributeError` に
            なり、`ValueError` だけを捕捉する呼び出し側の網を抜けるため先に弾く）、
            `str.isidentifier()` が偽（空文字・NUL 入りを含む）、`_` 始まり（pydantic が
            「Fields must not use names with leading underscores」の `NameError` を投げ、
            `__module__` などの dunder は `create_model` の予約引数名と衝突して例外も警告も
            なく消える）、Python 予約語（`instance.class` が SyntaxError になり属性アクセス
            で到達できない）、または `reserved` に含まれる場合。文言へは制御文字を生のまま
            載せないよう `repr()` 化した名前を添える（CWE-117）。
    """
    if not isinstance(value, str):
        raise ValueError(f"invalid {what} (must be a str): {value!r}")
    if not value.isidentifier():
        raise ValueError(f"invalid {what} (not a valid identifier): {value!r}")
    if value.startswith("_"):
        raise ValueError(f"invalid {what} (must not start with underscore): {value!r}")
    if keyword.iskeyword(value):
        raise ValueError(
            f"invalid {what} (Python keyword is not reachable via attribute access): {value!r}"
        )
    if value in reserved:
        raise ValueError(f"invalid {what} ({reserved_reason}): {value!r}")


def _validate_model_name(name: str) -> None:
    """生成するモデル名が到達可能な公開識別子であることを検証する。

    モデル名は生成クラスの `__name__` になり、`model_json_schema()` の `title` として
    LLM や UI フォームへ渡る出力面である。空白・改行・記号・NUL を含む名前をそのまま
    通すと、スキーマの読み手側で解釈される文字列を宣言側から差し込めてしまう。

    判定は `_check_public_identifier` が単独で持ち、本関数は文言だけを渡す。予約集合は
    渡さない（クラス名は pydantic のフィールド名前空間へ入らない）。

    Args:
        name: 検証対象のモデル名。

    Raises:
        ValueError: 非 str / 非識別子 / `_` 始まり / Python 予約語の場合。
    """
    _check_public_identifier(name, what="model name")


def _validate_field_name(name: str, field_name: str) -> None:
    """フィールド名が到達可能な公開識別子であり予約属性と衝突しないことを検証する。

    規則は `ParameterSpec._validate_name`（`actions.py`）と共通で、宣言層を経由しない
    呼び出し（予測段のスキーマ生成）でも同一の規則が効くように生成層へも置く。判定本体は
    `_check_public_identifier` が単独で持ち、本関数は文言と予約集合だけを渡す。

    `_RESERVED_MODEL_ATTRIBUTE_NAMES` を渡すのは、`model_config` のように「4 分岐を通過
    するのに生成モデルからフィールドが消える」名前を塞ぐためである（dunder だけを弾く形
    では不完全だった）。

    Args:
        name: 生成しようとしているモデル名（文言に添える）。
        field_name: 検証対象のフィールド名。

    Raises:
        ValueError: 非 str / 非識別子 / `_` 始まり / Python 予約語 / pydantic が `BaseModel`
            で予約する属性名の場合。
    """
    _check_public_identifier(
        field_name,
        what=f"field name for model {name!r}",
        reserved=_RESERVED_MODEL_ATTRIBUTE_NAMES,
        reserved_reason=_RESERVED_MODEL_ATTRIBUTE_REASON,
    )


def _validate_annotation(name: str, field_name: str, annotation: Any) -> None:
    """annotation が前方参照を含まないことを、評価する前に検証する。

    pydantic は前方参照（`str` / `ForwardRef`）を `eval` する。したがって `create_model()`
    へ渡した時点で任意の式が実行され、失敗しても `_unschemable_field_names` の切り分けで
    もう一度実行される。最上位だけでなく内側まで `_has_forward_ref` で検査し、判定は
    型の照合と `typing.get_args` の走査のみで行って評価経路へ到達させない。

    Args:
        name: 生成しようとしているモデル名（文言に添える）。
        field_name: 当該フィールド名。
        annotation: 検証対象の annotation。評価は行わない。

    Raises:
        ValueError: annotation 自身または内側に前方参照が含まれる場合。
    """
    if _has_forward_ref(annotation):
        raise ValueError(
            f"invalid annotation for field {field_name!r} of model {name!r}: a forward reference "
            f"(str or ForwardRef) is evaluated by pydantic; pass the type objects themselves"
        )


def _has_forward_ref(annotation: Any) -> bool:
    """annotation 自身または内側に前方参照が含まれるかを、評価せずに判定する。

    `typing.get_args` を辿るだけで、annotation を呼び出したり `eval` したりはしない。
    最上位しか見ない検査は `list["<式>"]` のように 1 段包むだけで迂回できるため再帰する。
    5 分岐で構成する。

    1. `str` / `ForwardRef` -> 前方参照そのもの。True を返す
    2. `list` / `tuple`（型ではないコンテナ）-> 中身を再帰。`get_args` は常に型を返すとは
       限らず、`get_args(Callable[[X], Y])` は `([X], Y)` のように**素の list** を第 1 要素
       に持つ。`get_origin(list_instance)` は `None`、`get_args(list_instance)` は `()` に
       なるため、この分岐が無いと `Callable[["<式>"], int]` の前方参照を取りこぼす
    3. `Literal` -> **辿らない**。`get_args(Literal["a"])` が返すのは型ではなく**値**として
       の str であり、辿ると `Literal["fast", "safe"]` のような正当な宣言を必ず誤検知する
    4. `Annotated` -> `get_args(...)[:1]`（型部分）のみ辿る。第 2 引数以降はメタデータで
       あり、`Annotated[int, "note"]` のように str の注記を置くのが通常の使い方である
    5. それ以外 -> `get_args(...)` を再帰。`list[X]` / `dict[K, V]` / `X | None` が該当する

    分岐 2 を `str` / `ForwardRef` の直後・`get_origin` 判定より前に置くのは、素のコンテナが
    `Literal` / `Annotated` の origin を持たないためである。`Literal` の値としての str は
    分岐 3 で辿るのをやめるため、コンテナ分岐には到達しない。

    Note:
        **完全な保証ではなく、宣言で普通に書ける範囲のネストを塞ぐ多層防御である。**
        `get_args` の返り値に現れる前方参照は、素の `list` / `tuple` に包まれたものも含めて
        すべて辿る。残るのは `get_args` に現れない場所へ隠す形であり、`TypedDict` /
        `NamedTuple` がフィールドとして保持する注釈、および自己参照モデルが内部に保持する
        未解決参照は、annotation の引数を辿っても到達できない。この形は
        `typing.TypedDict("TD", {"a": "<式>"})` のような**関数形コンストラクタ**でも作れる
        （クラス本体を書く必要はなく、注釈の位置に文字列を 1 個置くだけで成立する）。
        しかも `create_model()` が失敗した場合、`_unschemable_field_names` の切り分けが
        同じ annotation を**もう 1 度**評価する。最後の砦は「信頼できない入力から
        annotation を組み立てない」という呼び出し側の運用であり、本検査はその手前で事故を
        減らすためのものである。

    Args:
        annotation: 判定対象の annotation。評価は行わない。

    Returns:
        前方参照を含むなら True。

    """
    if isinstance(annotation, str | ForwardRef):
        return True
    if isinstance(annotation, list | tuple):
        return any(_has_forward_ref(arg) for arg in annotation)
    origin = get_origin(annotation)
    if origin is Literal:
        return False
    if origin is Annotated:
        return any(_has_forward_ref(arg) for arg in get_args(annotation)[:1])
    return any(_has_forward_ref(arg) for arg in get_args(annotation))


def _unschemable_field_names(
    name: str,
    fields: Mapping[str, tuple[Any, FieldInfo]],
) -> list[str]:
    """`create_model` が失敗した対応表から、原因となったフィールド名を宣言順で拾い出す。

    `PydanticSchemaGenerationError` のメッセージは扱えなかった型しか含まずフィールド名を
    持たない。FR-2 が「パラメータ名を添えた `ValueError`」を契約しているため、
    1 フィールドずつ組み直して切り分ける（失敗時にのみ通る経路であり常用しない）。

    Args:
        name: 生成しようとしたクラスの名前（切り分け用のモデルにも流用する）。
        fields: `build_frozen_model` へ渡された対応表。

    Returns:
        単体でも `create_model` が失敗したフィールド名の宣言順リスト。1 件も特定できな
        かった場合は組み合わせ由来とみなし、全フィールド名を返す。
    """
    offenders = [
        field_name
        for field_name, definition in fields.items()
        if not _is_schemable(name, field_name, definition)
    ]
    return offenders or list(fields)


def _is_schemable(name: str, field_name: str, definition: tuple[Any, FieldInfo]) -> bool:
    """単一フィールドだけのモデルを組めるかどうかを返す。

    Args:
        name: 生成しようとしたクラスの名前。
        field_name: 判定対象のフィールド名。
        definition: 当該フィールドの `(annotation, FieldInfo)`。

    Returns:
        pydantic がフィールド型として扱えるなら True。

    Note:
        捕捉を `PydanticSchemaGenerationError` に絞らず `Exception` まで広げている。
        切り分け中に別種の例外が出るとそちらが呼び出し元へ伝播し、本来の原因である
        `PydanticSchemaGenerationError` が隠れてしまうためである。ここは既に失敗が
        確定した経路であり、判定は「単体でも組めなかった」で十分である。
    """
    try:
        create_model(name, __config__={"frozen": True}, **{field_name: definition})
    except Exception:
        return False
    return True
