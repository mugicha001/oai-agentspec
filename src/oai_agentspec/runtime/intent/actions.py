"""実行可能アクションの宣言簿（`ActionSpec` / `ParameterSpec` / `ActionCatalog` / `param`）と、
`ActionCatalog.bind()` が返す結線済みの `ActionPlanner`。

宣言はここへ一元化し、候補生成方式（ルール / 学習モデル / LLM）を差し替えても下流の
パラメータ契約と実行結線を変えずに済むようにする。`action_agent` はエージェント名の str
として保持し、実体の解決は利用者が `AgentRegistry.get()` で行う（`agents` / `openai` を
import しない・NFR-1）。

既定（`prompt` / `prompt_vars` / `on_invalid_slot`）の解決はモジュールレベルの純関数 3 件
（`resolve_prompt` / `resolve_prompt_vars` / `resolve_on_invalid_slot`）に一元化する。
`ActionSpec` にも `ActionCatalog` にもメソッドとして埋めないのは、両者を引数で受ける関数に
すれば起動時検証と予測段の双方から同一実装を呼べるためである。

`param()` の公開シグネチャは `default=PARAM_UNSET` を既定に取るが、`ParameterSpec` は
sentinel をフィールドとして保持しない。`default` + `has_default` の 2 フィールドへ正規化する
（設計 §3.7）。
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from ._models import (
    _RESERVED_MODEL_ATTRIBUTE_NAMES,
    _RESERVED_MODEL_ATTRIBUTE_REASON,
    _check_public_identifier,
    _has_forward_ref,
    build_frozen_model,
)
from ._readonly import ReadOnlyStrMapping, _to_read_only_mapping

if TYPE_CHECKING:
    # `slots.py -> actions.py` はモジュールレベルの一方向であるため、型注釈のためだけの
    # 逆向き参照はここへ閉じる（実行時 import は `plan()` の中の遅延 import・設計 §2.1）。
    from .slots import ActionPlan, PlanResult

#: `param(default=...)` の「未宣言」を表す module-level センチネル。既存 `TOOL_UNSET`
#: （`constants.py`）と同型だが、あちらは `ToolSpec.failure_error_function` 専用と docstring で
#: 明示されているため用途を上書きせず、intent 側に専用のものを置く（コアへ intent 専用定数を
#: 持ち込まない・NFR-2）。`param()` が identity 判定に使うため、必ず単一インスタンスを共有する。
#: `ParameterSpec` のフィールドとしては保持しない（設計 §3.7）。
PARAM_UNSET: Final[object] = object()

#: 既定引数に mutable リテラルを置かないための空 Mapping。宣言簿は既定を書き換えない。
_EMPTY_PROMPT_VARS: Final[Mapping[str, str]] = MappingProxyType({})

#: `action_id` として使えない名前。`ActionCatalog` の公開メソッド 4 件と衝突すると、
#: 将来 `catalog.<action_id>` 形の参照を足したときに到達不能になるため宣言時に弾く。
#: `validate` / `plan` は `ActionPlanner` 側の公開メソッドであり `ActionCatalog` には無いので
#: 予約しない（`ActionCatalog` は純粋な宣言簿に徹する・FR-1）。
_RESERVED_ACTION_IDS: Final[frozenset[str]] = frozenset({"register", "names", "get", "bind"})

#: `on_invalid_slot` が取り得る値。宣言時に落とし、予測段まで持ち越さない（FR-1）。
_ON_INVALID_SLOT_VALUES: Final[tuple[str, ...]] = ("error", "skip")


class _DeclaredFieldsEq:
    """等価性を宣言フィールドのみで決める mixin。private attribute を比較へ混ぜない。

    pydantic の既定の `__eq__` は `__pydantic_private__` も比較するため、キャッシュや実行
    済みフラグを `PrivateAttr` で持つ型では**読み取りに見えるアクセサの呼び出しが等価性を
    変える**。`spec.parameters_model()` を片側だけで呼ぶと、宣言が同一の 2 インスタンスが
    `==` で不等になる。キャッシュは実装の詳細であり、呼び出し側から予測できない観測結果の
    変化を持ち込まないためにここで比較対象から外す。

    frozen かつ private attribute を持つ本パッケージの型が共有する。`ActionPlanner`
    （`validate()` 実行済みフラグを `PrivateAttr` で持つ・ADR 0029）も本 mixin を継承する
    だけで同じ性質を得られる。比較はまず `__dict__` 同士で行い、一致しない場合に宣言
    フィールド名だけで絞り直す（pydantic 既定と同じ 2 段構え）。`__dict__` には
    `functools.cached_property` の計算結果のように宣言フィールド以外も入りうるため、
    絞り直しが無いと**キャッシュの計算有無が等価性へ漏れて本 mixin の目的が失われる**。

    pydantic 既定との差分は次の 3 点に限られる（いずれも実測で確認済み。継承側が同じ検証を
    繰り返さずに済むよう結論を残す）。

    - **非 generic モデル専用**。pydantic 既定は `__pydantic_generic_metadata__` の origin を
      見て `Model(v=1) == Model[Any](v=1)` を成立させるが、本 mixin は `__class__` の厳密
      一致で判定するためこの同一視が失われる（本パッケージの `SlotSuggestion[T]` /
      `IntentContext[T]` / `IntentQuery[T]` には適用しない）。
    - 非推奨の `BaseModel.copy(exclude=...)` で宣言フィールドが `__dict__` から欠落した場合、
      「欠落」と「値が `None`」を区別しない（pydantic 既定は sentinel で区別する）。成立には
      当該フィールドの値が `None` であることも要り、`model_construct` は既定値を埋めるため
      再現しない。本パッケージに `.copy(` の使用は無く稼働経路へは届かない。
    - `extra` は pydantic 既定に合わせ `None` と `{}` を同一視する（検証時に `extra` の
      挙動が制御されうるため）。本パッケージの型は `extra` を宣言しないため、この比較は
      共有先のためにある。

    `cached_property` と `extra` の挙動はテストで pin していない。本パッケージに該当する型が
    無く、pin すると仕様に無い挙動を固定してしまうためである（2 条件で判断を揃えている）。

    `__hash__` は定義しない。`__eq__` を定義したクラスの `__hash__` は `None` になるが、
    pydantic は frozen モデルのクラス自身へ `__hash__` を設定するため MRO 上そちらが優先
    され、継承側のハッシュ可能性は本 mixin の導入前後で変わらない。
    """

    def __eq__(self, other: object) -> bool:
        """宣言フィールド（と extra）のみを比較する。

        Args:
            other: 比較対象。

        Returns:
            同一クラスであり宣言フィールドの値と `extra` がすべて等しければ True。クラスが
            異なる場合は pydantic 既定と同じく `NotImplemented` を返し、比較の判断を相手側と
            Python へ委ねる。`False` を返すと、`ActionSpec` との比較に応じる相手
            （`unittest.mock.ANY` など）との比較結果が pydantic 既定から変わってしまう。
        """
        if other.__class__ is not self.__class__:
            return NotImplemented
        fields_equal = self.__dict__ == other.__dict__ or all(
            self.__dict__.get(name) == other.__dict__.get(name) for name in type(self).model_fields
        )
        extra_equal = (getattr(self, "__pydantic_extra__", None) or {}) == (
            getattr(other, "__pydantic_extra__", None) or {}
        )
        return fields_equal and extra_equal


class ParameterSpec(BaseModel):
    """アクション 1 件のパラメータ 1 つの宣言。生成は `param()` を使う。"""

    model_config = {"frozen": True}
    name: str = Field(description="Parameter name. Must be a valid Python identifier.")
    annotation: Any = Field(
        description=(
            "Field type used to build the parameters model. Any pydantic-usable type."
            " Never build it from untrusted input: pydantic evaluates forward references."
        )
    )
    from_context: tuple[str, ...] = Field(
        default=(), description="Dotted paths into the run context, tried in declaration order."
    )
    by_llm: bool = Field(
        default=False, description="Whether the prediction stage may fill this parameter."
    )
    prompt: str | None = Field(
        default=None, description="Prompt segment name used when filling this parameter."
    )
    description: str | None = Field(
        default=None, description="Field description surfaced to the LLM and to UI forms."
    )
    default: Any = Field(
        default=None,
        description="Declared default value. Meaningful only when has_default is true.",
    )
    has_default: bool = Field(
        default=False,
        description=(
            "Whether a default was declared at all. Separates 'not declared' from an explicit"
            " default of None (the sentinel itself is never stored as a field value)."
        ),
    )
    max_suggestions: int = Field(
        default=1, ge=1, description="Upper bound of suggested values for this parameter (>=1)."
    )
    confirm: bool = Field(
        default=False, description="Whether a filled value still needs user confirmation."
    )
    filled_by_candidate: bool = Field(
        default=False, description="Whether the candidate itself supplies this parameter."
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        """name が到達可能な公開識別子であることを検証する。

        判定本体は `_models._check_public_identifier` が単独で持ち、ここでは文言と予約集合
        だけを渡す（生成層の `_validate_field_name` と同一の規則になる）。name は生成する
        pydantic モデルのフィールド名になるため、到達できない名前は宣言が黙って失われる
        経路になる。

        Args:
            v: バリデーション対象の name。

        Returns:
            検証済みの name。

        Raises:
            ValueError: 空文字を含め `str.isidentifier()` が偽の場合、`_` 始まりの場合
                （`_secret` は pydantic が「Fields must not use names with leading
                underscores」の `NameError` を投げ、`__module__` などの dunder は
                `create_model` の予約引数名と衝突して例外も警告もなく消える）、
                Python 予約語の場合（`isidentifier()` は True だが属性アクセスで
                SyntaxError になるため到達不能）、または pydantic が `BaseModel` で予約する
                属性名（`model_config` 等）の場合。
        """
        _check_public_identifier(
            v,
            what="parameter name",
            reserved=_RESERVED_MODEL_ATTRIBUTE_NAMES,
            reserved_reason=_RESERVED_MODEL_ATTRIBUTE_REASON,
        )
        return v

    @field_validator("annotation")
    @classmethod
    def _validate_annotation(cls, v: Any) -> Any:
        """annotation が前方参照を含まないことを、評価する前に検証する。

        pydantic は前方参照（`str` / `ForwardRef`）を `eval` するため、宣言層で受理すると
        任意式の実行経路になる。最上位だけを見る検査は `list["<式>"]` のように 1 段包む
        だけで迂回できるため、`_models._has_forward_ref` で内側まで再帰的に判定する
        （`Literal` の値と `Annotated` のメタデータは辿らない。判定規則の SoT は当該関数の
        docstring）。判定は型の照合と `typing.get_args` の走査のみで、評価は一切行わない。

        生成層（`_models.build_frozen_model`）も同じ関数で検査しており、宣言層を経由
        しない呼び出しに備えた多層防御になっている。**完全な保証ではない**（`get_args` に
        現れない場所へ隠した前方参照は残る）。

        Args:
            v: バリデーション対象の annotation。評価は行わない。

        Returns:
            検証済みの annotation。

        Raises:
            ValueError: annotation 自身または内側に前方参照が含まれる場合。
        """
        if _has_forward_ref(v):
            raise ValueError(
                "invalid parameter annotation: a forward reference (str or ForwardRef) is "
                "evaluated by pydantic; pass the type objects themselves"
            )
        return v


def param(
    name: str,
    annotation: Any,
    *,
    from_context: str | tuple[str, ...] | None = None,
    by_llm: bool = False,
    prompt: str | None = None,
    description: str | None = None,
    default: Any = PARAM_UNSET,
    max_suggestions: int = 1,
    confirm: bool = False,
    filled_by_candidate: bool = False,
) -> ParameterSpec:
    """`ParameterSpec` を宣言する。

    `from_context` の str を 1 要素 tuple へ正規化し、`default` の sentinel を
    `default` + `has_default` の 2 フィールドへ正規化する（設計 §3.7）。

    Args:
        name: パラメータ名。生成モデルのフィールド名になる。`_` 始まり・Python 予約語は
            到達不能なため受け付けない。
        annotation: フィールド型。`parameters_model()` がそのまま pydantic へ渡す。
            型オブジェクトのみを受け付け、前方参照の str は受け付けない。前方参照の検査は
            多層防御であって完全ではないため（`_models._has_forward_ref` の Note）、
            **信頼できない入力から annotation を組み立てないこと**。pydantic は前方参照を
            `eval` するため、組み立て元が汚染されると任意式の実行経路になる。
        from_context: run context からの取得元パス。str なら 1 要素 tuple へ正規化し、
            tuple なら宣言順を保持する。None なら取得元なし。
        by_llm: 予測段による穴埋めを許すか。
        prompt: 穴埋め時に使うプロンプトセグメント名。
        description: `Field(description=...)` に載せる説明。
        default: 既定値。省略（`PARAM_UNSET`）なら「未宣言」として扱い、明示的な `None` とは
            `has_default` で区別する。
        max_suggestions: 候補値の上限件数（1 以上）。
        confirm: 埋まった値にユーザー確認を要するか。
        filled_by_candidate: 候補側がこのパラメータを供給するか。

    Returns:
        正規化済みの `ParameterSpec`。

    Raises:
        ValueError: name が `str.isidentifier()` 偽 / `_` 始まり / Python 予約語のいずれかの
            場合、annotation が前方参照の str の場合、または max_suggestions が 1 未満の
            場合（いずれも `ParameterSpec` の検証で落ちる）。
    """
    has_default = default is not PARAM_UNSET
    return ParameterSpec(
        name=name,
        annotation=annotation,
        from_context=(from_context,) if isinstance(from_context, str) else (from_context or ()),
        by_llm=by_llm,
        prompt=prompt,
        description=description,
        default=default if has_default else None,
        has_default=has_default,
        max_suggestions=max_suggestions,
        confirm=confirm,
        filled_by_candidate=filled_by_candidate,
    )


class ActionSpec(_DeclaredFieldsEq, BaseModel):
    """実行可能アクション 1 件の宣言。`ActionCatalog.register` で宣言簿へ載せる。

    等価性は宣言フィールドのみで決まる（`_DeclaredFieldsEq`）。`parameters_model()` の
    キャッシュを `PrivateAttr` で持つため、pydantic 既定の `__eq__` のままだと当該メソッドの
    呼び出しが等価性を変えてしまう。
    """

    model_config = {"frozen": True}
    #: `parameters_model()` の `(鍵, 生成結果)`。FR-2 L123 の「2 回以上呼ぶと同一のクラス
    #: オブジェクトを返す」を満たすためのキャッシュで、`PrivateAttr` に置くのは frozen
    #: モデルでも代入でき（`model_config` の frozen は宣言フィールドにのみ効く）、公開
    #: フィールドとして `model_dump()` / `model_json_schema()` へ漏れないためである。
    #: モジュールレベルの dict へ `id(spec)` を鍵に持たせる形は、宣言が GC された後に同じ
    #: アドレスの別オブジェクトへ当たるため採らない。
    #:
    #: 鍵と生成結果を 1 つの private attr へ**組にして**持つのは、`model_copy(update=...)`
    #: が private attr をそのまま引き継ぐためである（鍵の詳細は `parameters_model()` の
    #: Note）。2 つの private attr へ分けると「鍵だけが更新された」中間状態が型として
    #: 表現可能になるが、組にすれば鍵と結果は常に同じ生成に由来する。
    _parameters_model_cache: tuple[tuple[Any, ...], type[BaseModel]] | None = PrivateAttr(
        default=None
    )
    action_id: str = Field(description="Unique action identifier. A reachable public identifier.")
    description: str = Field(description="What this action does. Shown to the candidate source.")
    action_agent: str = Field(
        description="Name of the agent that executes this action. A plain str, never an instance."
    )
    label: str = Field(
        description=(
            "UI label template. ${param} placeholders are substituted."
            " Write a literal dollar sign as '$$' (startup validation rejects labels that"
            " string.Template cannot render, e.g. 'Pay $100')."
        )
    )
    parameters: tuple[ParameterSpec, ...] = Field(
        description="Parameter declarations in declaration order. Names must be unique."
    )
    prompt: tuple[str, ...] = Field(
        default=(), description="Prompt segments merged after the catalog-wide ones."
    )
    # 既定は default_factory で毎回新しい dict を作る。`_EMPTY_PROMPT_VARS`（mappingproxy）を
    # 既定に置くと pydantic の既定値 deepcopy が mappingproxy を複製できずに落ちる。
    # `validate_default=True` を付けるのは、既定値も `ReadOnlyStrMapping` の正規化を通し、
    # 「宣言時に渡した場合だけ読み取り専用」という穴を作らないためである。
    prompt_vars: ReadOnlyStrMapping = Field(
        default_factory=dict,
        validate_default=True,
        description="Prompt variables merged over the catalog-wide ones.",
    )
    on_invalid_slot: Literal["error", "skip"] | None = Field(
        default=None, description="Overrides the catalog-wide setting. None means 'not declared'."
    )

    @field_validator("action_id")
    @classmethod
    def _validate_action_id(cls, v: str) -> str:
        """action_id が到達可能な公開識別子であり予約名と衝突しないことを検証する。

        規則の SoT は `ToolRegistry._validate_name`（`tool_registry.py`）で、本 validator は
        `agent_names._validate_attribute_name` と同型の分岐を、判定本体を持つ
        `_models._check_public_identifier` へ予約集合 `_RESERVED_ACTION_IDS` を渡す形で
        適用する。

        Args:
            v: バリデーション対象の action_id。

        Returns:
            検証済みの action_id。

        Raises:
            ValueError: `str.isidentifier()` 偽（空文字を含む）、`_` 始まり、Python 予約語
                （`class` / `None` 等・`isidentifier()` は True だが属性アクセスで
                SyntaxError になるため到達不能）、`ActionCatalog` の公開メソッド名との衝突の
                いずれか。
        """
        _check_public_identifier(
            v,
            what="action_id",
            reserved=_RESERVED_ACTION_IDS,
            reserved_reason="collides with an ActionCatalog method",
        )
        return v

    @field_validator("parameters")
    @classmethod
    def _validate_parameters(cls, v: tuple[ParameterSpec, ...]) -> tuple[ParameterSpec, ...]:
        """同一 ActionSpec 内にパラメータ名の重複がないことを検証する。

        Args:
            v: バリデーション対象の parameters tuple。

        Returns:
            検証済みの parameters tuple。

        Raises:
            ValueError: 同名の `ParameterSpec` が 2 件以上ある場合。名前は生成モデルの
                フィールド名になるため、重複すると宣言の一方が黙って消える。
        """
        names = [p.name for p in v]
        if len(names) != len(set(names)):
            raise ValueError("ActionSpec.parameters has duplicate names")
        return v

    def parameters_model(self) -> type[BaseModel]:
        """全パラメータ宣言をフィールド化した frozen な `BaseModel` サブクラスを返す。

        宣言した型を単一の出どころとして、実行入力の検証（`ActionPlan.input_json`）・
        LLM へのスキーマ提示・UI のフォーム生成へ使い回すための窓口である（FR-2）。
        `has_default` が真なら当該既定を持つ任意フィールド、偽なら必須フィールドになる。
        「未宣言」と「明示的な `default=None`」の分離（設計 §3.7）を生成モデルまで通すのは、
        必須 / 任意の別がこの区別の唯一の観測点であるためである。

        生成結果は初回呼び出し時にキャッシュし、2 回目以降は同一のクラスオブジェクトを返す
        （FR-2 L123）。毎回組み直すと、先に組んだインスタンスが後の呼び出しで得たモデルの
        `isinstance` を満たさず、下流の型検証が黙って落ちる。生成に失敗した場合は何も
        キャッシュしないため、2 回目以降も同じ `ValueError` になる。

        Returns:
            パラメータ宣言をフィールドに持つ frozen な `BaseModel` サブクラス。

        Raises:
            ValueError: いずれかの annotation を pydantic がフィールド型として扱えない場合。
                当該パラメータ名を添えた文言へ変換される（`_models.build_frozen_model`）。

        Note:
            キャッシュは `(action_id, parameters)` を鍵として引く。この 2 つは
            `build_frozen_model` へ渡す入力（クラス名とフィールド定義）そのものであり、
            他の宣言フィールドは生成結果に影響しない。鍵を持たず「1 度作ったら再利用」
            にすると、`model_copy(update=...)` が private attr をそのまま引き継ぐため、
            `parameters_model()` を呼んだ宣言から派生させたコピーが**自分の宣言と無関係な
            モデル**を返す。フィールド集合が重なる派生（型だけ狭める・`default` だけ
            差し替える等）では例外にもならず、宣言と検証型が食い違ったまま
            `ActionPlan.input_json` が通ってしまう。`model_copy(update=...)` は frozen
            pydantic の派生の標準手段であり、`spec` は `ActionPlan` の公開フィールドとして
            利用者から到達できるため、到達しうる経路として塞ぐ。
        """
        key = (self.action_id, self.parameters)
        cached = self._parameters_model_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        fields: dict[str, tuple[Any, Any]] = {
            spec.name: (
                spec.annotation,
                Field(default=spec.default, description=spec.description)
                if spec.has_default
                else Field(description=spec.description),
            )
            for spec in self.parameters
        }
        model = build_frozen_model(_model_name(self.action_id, "Params"), fields)
        self._parameters_model_cache = (key, model)
        return model


class ActionCatalog:
    """アクション宣言の簿冊。plain な mutable クラス（frozen 契約の対象外）。

    公開するのは `register` / `names` / `get` / `bind` の 4 メソッドのみで、`validate` /
    `plan` は `bind()` が返す `ActionPlanner` 側の公開メソッドである。宣言簿と結線・実行の
    関心を分けるための線引きであり、ここへ検証や計画を足さないこと（FR-1）。

    設定 3 件（`prompt` / `prompt_vars` / `on_invalid_slot`）は読み取り専用 property であり、
    実体は `_` 始まりの属性が持つ。plain 属性のままだと `catalog.on_invalid_slot = "bogus"`
    が `__init__` の値検証（FR-1 L116）を迂回し、不正値を予測段まで持ち越せてしまう。
    `resolve_on_invalid_slot` は検証済みの 2 値を前提にしているため、宣言後に書き換わらない
    ことを契約とする。宣言簿としての可変性（`register`）は従来どおり維持する。
    """

    def __init__(
        self,
        *,
        prompt: tuple[str, ...] = (),
        prompt_vars: Mapping[str, str] = _EMPTY_PROMPT_VARS,
        on_invalid_slot: str = "skip",
    ) -> None:
        """コンストラクタ。全 `ActionSpec` 共通の既定を受け取る。

        Args:
            prompt: 全アクション共通のプロンプトセグメントの列（tuple / list）。各 `ActionSpec`
                の同名フィールドとマージされ、アクション側が後に積まれる。bare str は
                受け付けない（`tuple("billing")` が 1 文字ずつ分解されるため）。
            prompt_vars: 全アクション共通のプロンプト変数。同名キーはアクション側が勝つ。
            on_invalid_slot: 不正スロット時の既定挙動（`"error"` / `"skip"`）。

        Raises:
            ValueError: on_invalid_slot が `"error"` / `"skip"` のいずれでもない場合、または
                prompt に bare str を渡した場合。いずれも宣言時に落とし、予測段まで
                持ち越さない（FR-1）。
        """
        if on_invalid_slot not in _ON_INVALID_SLOT_VALUES:
            raise ValueError(
                f"invalid on_invalid_slot (must be one of {_ON_INVALID_SLOT_VALUES}): "
                f"{on_invalid_slot!r}"
            )
        # bare str を `tuple(...)` へ通すと 1 文字ずつのセグメント名として黙って成立する。
        # 同じ契約を持つ `ActionSpec.prompt` は宣言時に弾くため、非対称を残さない。
        if isinstance(prompt, str):
            raise ValueError(
                f"invalid prompt (must be a sequence of segment names, not a str): {prompt!r}"
            )
        self._prompt: tuple[str, ...] = tuple(prompt)
        # 正規化の規則と根拠は `_readonly` に一元化する（pydantic を経由しない本クラスは
        # 型エイリアスではなく正規化関数を直接呼ぶ）。
        self._prompt_vars: Mapping[str, str] = _to_read_only_mapping(prompt_vars)
        # 上の分岐を通過した時点で 2 値のいずれかであることが確定している。
        self._on_invalid_slot: Literal["error", "skip"] = cast(
            'Literal["error", "skip"]', on_invalid_slot
        )
        self._specs: dict[str, ActionSpec] = {}

    @property
    def prompt(self) -> tuple[str, ...]:
        """全アクション共通のプロンプトセグメント（宣言後は変更できない）。"""
        return self._prompt

    @property
    def prompt_vars(self) -> Mapping[str, str]:
        """全アクション共通のプロンプト変数（読み取り専用ビュー。中身も変更できない）。"""
        return self._prompt_vars

    @property
    def on_invalid_slot(self) -> Literal["error", "skip"]:
        """不正スロット時の既定挙動（宣言後は変更できない）。"""
        return self._on_invalid_slot

    def register(self, spec: ActionSpec) -> None:
        """アクション宣言を簿冊へ載せる。

        Args:
            spec: 登録する `ActionSpec`。宣言そのものを保持し、複製も差し替えもしない。

        Raises:
            ValueError: 同一 action_id が既に登録済みの場合。後勝ちで黙って上書きすると、
                どちらの宣言が効いているかが実行時まで分からなくなる。
        """
        if spec.action_id in self._specs:
            raise ValueError(f"action_id is already registered: {spec.action_id!r}")
        self._specs[spec.action_id] = spec

    def names(self) -> list[str]:
        """登録済みの action_id を昇順で返す。

        Returns:
            action_id の昇順リスト。未登録なら空リスト。
        """
        return sorted(self._specs)

    def get(self, action_id: str) -> ActionSpec:
        """登録済みのアクション宣言を取り出す。

        Args:
            action_id: 取り出す宣言の action_id。

        Returns:
            登録時に渡された `ActionSpec` そのもの。

        Raises:
            KeyError: 当該 action_id が未登録の場合。
        """
        if action_id not in self._specs:
            raise KeyError(f"unknown action_id: {action_id!r}")
        return self._specs[action_id]

    def bind(
        self,
        *,
        registry: Any,
        prompts: Any = None,
        guardrail_registry: Any = None,
        candidates: Any = None,
        llm_filler: Any = None,
    ) -> ActionPlanner:
        """結線と宣言簿のスナップショットを載せた frozen な `ActionPlanner` を返す。

        自分自身は一切変更せず、実行もしない（LLM 0 回）。同じ宣言簿を別の結線で何度でも
        bind でき、得られる `ActionPlanner` は互いに独立である。bind 後の `register()` は
        既に返した `ActionPlanner` へ届かない（スナップショットを取るため）。

        結線値は型を検査せず不透明値のまま保持する。`CandidateSource` / `LLMFiller` の値域
        検証は各宣言型の validator が済ませており（設計 §3.4a）、解決簿 3 件は利用者の
        アプリ資産であって `agents` / `openai` へ触れずに検査できないためである（NFR-1）。

        Args:
            registry: 実行先エージェントを解決する `AgentRegistry`。
            prompts: プロンプトセグメントを解決する `PromptStore`。
            guardrail_registry: ガードレール登録名を解決する `GuardrailRegistry`。
            candidates: 候補の出どころ（`CandidateSource`）。`None` なら `plan()` は
                `RuntimeError`（`validate()` だけの利用は妨げない・設計 §3.4a の規則 1）。
            llm_filler: 不足パラメータの埋め方（`LLMFiller`）。`None` なら穴埋め経路その
                ものが存在しない（規則 3）。

        Returns:
            bind 時点の宣言簿スナップショットと結線を載せた frozen な `ActionPlanner`。
        """
        return ActionPlanner(
            specs=tuple(self._specs[action_id] for action_id in self.names()),
            prompt=self._prompt,
            prompt_vars=dict(self._prompt_vars),
            on_invalid_slot=self._on_invalid_slot,
            registry=registry,
            prompts=prompts,
            guardrail_registry=guardrail_registry,
            candidates=candidates,
            llm_filler=llm_filler,
        )


class ActionPlanner(_DeclaredFieldsEq, BaseModel):
    """`ActionCatalog.bind()` が返す frozen な結線済みオブジェクト（ADR 0029 Decision 2）。

    公開メソッドは `validate` / `plan` の 2 つで、宣言簿側には存在しない。未結線のまま
    `plan()` を呼ぶ形が構造的に成立しないようにするための線引きである。

    宣言簿は `ActionCatalog` そのものではなく **bind 時点のスナップショット**として保持する
    （`specs` と設定 3 件）。`ActionCatalog` は plain な mutable クラスであり、フィールドと
    して抱えると (1) bind 後の `register()` が既存 planner の挙動を変え、(2) 等価性が
    宣言の同一性ではなくオブジェクトの同一性になってしまう。検証と計画に必要な
    `ActionCatalog` はスナップショットから組み直して `PrivateAttr` に置く
    （`_validate` / `slots` の受け口が `ActionCatalog` であるため、ここで橋渡しする）。

    **派生状態（組み直した宣言簿・検証済みかどうか）は常に宣言フィールドから導出し直す。**
    `model_copy(update=...)` は frozen pydantic の派生の標準手段だが `model_post_init` を
    呼ばず private attribute をそのまま引き継ぐため、「宣言フィールドは新しいのに `plan()` が
    参照する宣言簿と検証状態は古い」コピーが作れてしまう（コピーで除外したはずのアクションが
    allowlist に残り、差し替えた結線に対する起動時検証も走らない）。したがって宣言簿は
    `ActionSpec.parameters_model()` と同じ鍵付きキャッシュにし、検証済みかどうかは真偽値では
    なく「どのフィールド一式に対して検証したか」（`__dict__` のスナップショット）で持つ。

    等価性は宣言フィールドのみで決まる（`_DeclaredFieldsEq`）。検証済みスナップショットと
    組み直した宣言簿はいずれも `PrivateAttr` であり、pydantic 既定の `__eq__` のままだと
    `validate()` / `plan()` の呼び出しが等価性を変えてしまう。
    """

    model_config = {"frozen": True}
    #: スナップショットから組み直した宣言簿の `(鍵, 生成結果)`。鍵は宣言簿を組む材料
    #: （`specs` と設定 3 件）であり、`model_copy(update=...)` で 1 つでも変われば組み直す。
    #: `parameters_model()` と同じく鍵と結果を 1 つの private attr へ組にして持つ（分けると
    #: 「鍵だけが更新された」中間状態が表現可能になる）。
    _catalog_cache: tuple[tuple[Any, ...], ActionCatalog] | None = PrivateAttr(default=None)
    #: `validate()` を通した時点のフィールド一式（`__dict__` の浅いコピー）。`plan()` の段 (0)
    #: は現在の `__dict__` と `==` で照合し、一致するときだけ自動検証を省く（明示的な
    #: `validate()` は常に実行する）。結線値は任意のオブジェクトであるため、比較は各値の
    #: 既定の `==`（多くは同一性）に委ねる。真偽値で持つと、宣言フィールドや結線を差し替えた
    #: コピーが「検証済み」を名乗ってしまう。
    #:
    #: スナップショットは**浅い**コピーであるため、`model_copy(update=...)` で渡された可変
    #: オブジェクト（validator を通らないので読み取り専用化もされない素の dict 等）を、
    #: 検証後に in-place で書き換えられた場合は検知できない（同一オブジェクトを指したまま
    #: `==` が成立する）。派生は `model_validate` 経由で作ること。
    _validated_fields: dict[str, Any] | None = PrivateAttr(default=None)
    specs: tuple[ActionSpec, ...] = Field(
        description="Snapshot of the declarations at bind time, in action_id order."
    )
    prompt: tuple[str, ...] = Field(description="Catalog-wide prompt segments at bind time.")
    prompt_vars: ReadOnlyStrMapping = Field(
        description="Catalog-wide prompt variables at bind time."
    )
    on_invalid_slot: Literal["error", "skip"] = Field(
        description="Catalog-wide behaviour for invalid slots at bind time."
    )
    registry: Any = Field(description="AgentRegistry-ish object. Held opaquely (NFR-1).")
    prompts: Any = Field(default=None, description="PromptStore-ish object, or None.")
    guardrail_registry: Any = Field(
        default=None, description="GuardrailRegistry-ish object, or None."
    )
    candidates: Any = Field(
        default=None, description="CandidateSource, or None when plan() is not usable."
    )
    llm_filler: Any = Field(
        default=None, description="LLMFiller, or None when there is no filling path at all."
    )

    def model_post_init(self, context: Any, /) -> None:
        """構築時にスナップショットから宣言簿を組み、鍵付きキャッシュを温める。

        組み直し自体は `_current_catalog()` が持つ。ここで 1 度呼ぶのは、宣言簿として
        成立しない `specs`（同一 action_id の重複）を `plan()` まで持ち越さずに構築時点で
        落とすためである。

        Args:
            context: pydantic が渡す検証コンテキスト。使わない。
        """
        self._current_catalog()

    def _current_catalog(self) -> ActionCatalog:
        """現在の宣言フィールドから組んだ宣言簿を返す（鍵付きキャッシュ）。

        鍵は宣言簿を組む材料そのもの（`specs` と設定 3 件）であり、`model_copy(update=...)`
        で 1 つでも変われば組み直す。鍵を持たずに「1 度作ったら再利用」にすると、コピーの
        `plan()` が自分の宣言と無関係な allowlist で動く。

        Returns:
            現在の宣言フィールドと一致する `ActionCatalog`。
        """
        key = (self.specs, self.prompt, dict(self.prompt_vars), self.on_invalid_slot)
        cached = self._catalog_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        catalog = ActionCatalog(
            prompt=self.prompt,
            prompt_vars=self.prompt_vars,
            on_invalid_slot=self.on_invalid_slot,
        )
        for spec in self.specs:
            catalog.register(spec)
        self._catalog_cache = (key, catalog)
        return catalog

    def validate(self, *, context: Any = None) -> None:
        """宣言簿と結線の整合を検査する（起動時検証・FR-3）。

        LLM / network / env を一切参照しないため同期メソッドである。検査の実装は
        `_validate._validate_catalog` が単独で持ち、ここでは呼び出しと「どのフィールド一式に
        対して検証したか」（`_validated_fields`）の記録だけを行う（検査を 2 箇所に持つと
        片方だけが直される）。

        明示的に呼ばれた場合は記録があっても再度検査する。宣言簿はスナップショット
        だが結線先（`registry` / `prompts`）は利用者のオブジェクトであり、こちら側で
        「前回と同じ結果になる」と決めつけない。

        Args:
            context: run context の代表インスタンス。渡したときだけパスの構造検査
                （検査 7）を行う。

        Returns:
            `None`。全検査を通過したことを表す。

        Raises:
            KeyError: 未登録の `action_agent` または解決できないガードレール登録名がある
                場合。
            ValueError: `label` / プロンプト変数 / `from_context` などの宣言に違反がある
                場合、および `guardrails` が非空なのに解決簿が未結線の場合。
            RuntimeError: セグメント宣言があるのに `prompts` が未結線の場合
                （設計 §3.4a の規則 2）。
            PromptTemplateIntegrityError: lockdown 済みの `PromptStore` が manifest 未掲載の
                セグメントを拒否した場合（`_validate` が捕捉せずそのまま伝播させる・
                設計 §3.12）。
        """
        from ._validate import _validate_catalog

        _validate_catalog(
            self._current_catalog(),
            registry=self.registry,
            prompts=self.prompts,
            guardrail_registry=self.guardrail_registry,
            llm_filler=self.llm_filler,
            context=context,
        )
        self._validated_fields = dict(self.__dict__)
        return None

    async def plan(
        self,
        query: Any,
        *,
        predict: bool = True,
        detail: bool = False,
    ) -> tuple[ActionPlan, ...] | PlanResult:
        """毎ターンの窓口。候補生成からスロット確定までを 1 呼び出しへ畳む（設計 §3.13）。

        内部順序は (0) 現在のフィールド一式に対して未検証なら `validate()` -> (1) 候補生成と
        allowlist 除外 -> (2) 決定的なスロット確定 -> (3) 不足パラメータの予測、である。
        段 (0) を先に置くのは、宣言と結線の不整合を候補生成器（従量課金が起きうる）を呼ぶ前に
        落とすためである。検証済みかどうかは真偽値ではなく検証時のフィールド一式との一致で
        判定するため、`model_copy(update=...)` で宣言または結線を差し替えたコピーは未検証
        扱いへ戻る。

        段 (3) は `predict=True`（既定）かつ `llm_filler` が結線されている場合にだけ
        `_predict._predict_params` へ委譲する（設計 §3.4a の規則 3）。委譲は `plan()` 1 回に
        つき 1 回であり、候補件数・不足件数に比例しない（ADR 0026）。委譲しなかった場合は
        段 (2) の計画をそのまま返し、`usage` は 0 件を表す `ParamUsage` になる。

        Args:
            query: 発話と `run_context` を載せた `IntentQuery`。
            predict: 段 (3) を実行するか。`False` なら `NEEDS_LLM` のスロットは埋まらず、
                従量課金も発生しない。
            detail: `True` なら `PlanResult(plans, suggestion, usage)` を返し、候補生成器の
                `report` / `metadata` と実行量を捨てない。

        Returns:
            `detail=False` なら候補と同順・同数の `tuple[ActionPlan, ...]`、`detail=True`
            なら `PlanResult`。

        Raises:
            RuntimeError: `candidates` が未結線の場合（設計 §3.4a の規則 1）。候補の
                出どころが無いまま毎ターンの窓口を呼ぶのは結線漏れである。段 (3) が
                セグメント宣言に対する `prompts` の未結線を検出した場合と、予測対象が
                あるのに会話（`IntentContext.history_items`）が空だった場合も同じ型で
                落ちる（会話が予測エージェントへ届く唯一の経路であるため）。
            Exception: 段 (0) の検証と、`context_builder` / `generator` / 段 (3) が送出した
                例外は種別を問わずそのまま伝播する。
        """
        from ._suggest import _suggest_intents
        from .slots import ParamUsage, PlanResult, _plan_slots

        if self._validated_fields != self.__dict__:
            self.validate()
        if self.candidates is None:
            raise RuntimeError(
                "plan() requires a candidate source; pass bind(candidates=CandidateSource(...))"
            )

        catalog = self._current_catalog()
        suggestion = await _suggest_intents(query, self.candidates, catalog.names())
        plans = _plan_slots(suggestion.candidates, catalog, suggestion.context)
        usage = ParamUsage(
            runs=0, model_calls=0, candidates=0, input_tokens=None, output_tokens=None
        )
        if predict and self.llm_filler is not None:
            from ._predict import _predict_params

            plans, usage = await _predict_params(
                plans,
                suggestion.context,
                llm_filler=self.llm_filler,
                prompts=self.prompts,
                guardrail_registry=self.guardrail_registry,
            )
        if detail:
            return PlanResult(plans=plans, suggestion=suggestion, usage=usage)
        return plans


def _model_name(action_id: str, suffix: str) -> str:
    """`action_id` から実行時生成モデルのクラス名を組む。

    クラス名は `model_json_schema()` の `title` として LLM / UI へ渡る出力面であるため、
    snake_case の `action_id` を CamelCase へ寄せて用途の接尾辞を付ける
    （`run_load_test` + `"Params"` -> `RunLoadTestParams`）。`action_id` は宣言時に
    「識別子・`_` 始まりでない・予約語でない」まで検証済みであり、`_` で分割して各断片の
    先頭を大文字化した結果も必ず公開識別子になる。

    `parameters_model()`（本モジュール）と予測段のスキーマモデル（`slots._slots_model`）が
    共有する。命名規則を 2 箇所に書くと、片方だけが直される drift が起きる。

    Args:
        action_id: 検証済みの action_id。
        suffix: 用途を表す接尾辞（`"Params"` / `"Slots"`）。

    Returns:
        `model_json_schema()` の `title` として使えるクラス名。
    """
    stem = "".join(part[:1].upper() + part[1:] for part in action_id.split("_"))
    return f"{stem}{suffix}"


def _resolve_path(obj: Any, path: str) -> Any:
    """`.` 区切りのパスで run context を辿り、解決できなければ `None` を返す。

    利用者は `_validate.py`（構造的に解決できるかの検査）・`slots.py`（`from_context` の
    解決）・`_predict.py`（`prompt_vars` の解決）の 3 つあり、本関数へ一元化する
    （設計 §3.4b）。`actions.py` は intent 側の依存グラフの最下層であり、3 者すべてが
    既に本モジュールを import している。

    各セグメントは、対象が `collections.abc.Mapping` ならキーで、そうでなければ属性で
    辿る。`hasattr(obj, "__getitem__")` で判定すると `str` / `list` まで mapping 扱いに
    なり、`Mapping` を先に見ないと `{"items": 7}` のようにキー名が dict のメソッド名と
    衝突する場合に属性（`dict.items`）へ落ちる。

    非 Mapping は `getattr` で辿るため、宣言したパスは run context の**任意の属性**
    （`property` を含む）へ到達しうる。`property` は読み出し自体が任意のコードを走らせる。
    したがって `from_context` / `prompt_vars` のパスは、利用者入力や候補由来ではなく
    **開発者が宣言する値に限る**こと。

    Args:
        obj: 起点のオブジェクト。`None` でもよい（`run_context` 未指定の場合）。
        path: `.` 区切りのパス。

    Returns:
        解決できた値。途中のセグメントが解決できない場合と、起点が `None` の場合は
        `None`。解決できない状態は宣言順に次のパスを試す正常系であり、例外にしない
        （FR-3 L152 / FR-5 L179）。
    """
    current = obj
    for segment in path.split("."):
        if current is None:
            return None
        if isinstance(current, Mapping):
            if segment not in current:
                return None
            current = current[segment]
        else:
            current = getattr(current, segment, None)
    return current


def resolve_prompt(catalog: ActionCatalog, spec: ActionSpec) -> tuple[str, ...]:
    """カタログ既定とアクション宣言のプロンプトセグメントをマージする。

    Args:
        catalog: 既定を持つ `ActionCatalog`。
        spec: 対象の `ActionSpec`。

    Returns:
        カタログ側を先、アクション側を後に積んだセグメント名の tuple。
    """
    return (*catalog.prompt, *spec.prompt)


def resolve_prompt_vars(catalog: ActionCatalog, spec: ActionSpec) -> Mapping[str, str]:
    """カタログ既定とアクション宣言のプロンプト変数をマージする。

    Args:
        catalog: 既定を持つ `ActionCatalog`。
        spec: 対象の `ActionSpec`。

    Returns:
        同名キーはアクション側が勝つマージ結果。
    """
    return {**catalog.prompt_vars, **spec.prompt_vars}


def resolve_on_invalid_slot(catalog: ActionCatalog, spec: ActionSpec) -> Literal["error", "skip"]:
    """不正スロット時の挙動を上書き規則で解決する。

    `prompt` / `prompt_vars` と違いマージの意味がない単一の選択であるため、アクション側の
    宣言があればそれが勝ち、無ければカタログ既定を採る。

    Args:
        catalog: 既定を持つ `ActionCatalog`。
        spec: 対象の `ActionSpec`。

    Returns:
        `"error"` または `"skip"`。双方とも宣言時に値検証済みである。
    """
    if spec.on_invalid_slot is not None:
        return spec.on_invalid_slot
    return catalog.on_invalid_slot
