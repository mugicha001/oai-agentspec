"""アクション実行のスロット表現と計画型（`Slot` / `ActionPlan` / `PlanResult`）。

`runtime/lightning/slots.py`（`prompt_slot` = プロンプトの分割単位）とは**別概念**である。
本モジュールのスロットは「アクション 1 件のパラメータ 1 つの充足状態」を指す。

案 1 の 4 クラス（`Resolved` / `NeedsAgent` / `NeedsConfirmation` / `NeedsUser`）を `Slot`
1 型 + `SlotState` / `Origin` の 2 enum へ畳んでいる。1 型化すると 4 クラスが構造的に持って
いた制約（「未解決状態には `origin` フィールドが無い」など）が失われるため、その埋め戻しを
`Slot` の `model_validator(mode="after")` 6 条件が担う（設計 §3.5）。ここが緩むと FR-5 の
各契約が型でも実行時でも検査されない宣言になる。

方針:
- 状態の表現は `state` ただ 1 つ。`resolved` のような `state` と同値の導出プロパティは
  持たない（状態を 2 通りで表現すると片方だけを見る利用者コードが生まれる・設計 §3.5b）。
- `from_user` は残す。`origin` が `USER_INPUT` / `USER_CONFIRMED` のどちらかという enum の
  判定規則を利用者に書かせないためであり、`origin is None` でも例外を出さず `False` を返す。
- `ActionPlan` は**実行先エージェントの実体を保持しない**。名前（`action_agent`）だけを
  公開し、解決は利用者が `registry.get()` で行う（設計 §3.4d）。
- `spec` / `action_agent` / `resolved_*` の 5 件は `Field(exclude=True)`。`model_dump()` に
  宣言と結線が漏れないようにするためで、`model_json_schema()` の properties には残る
  （実測 7-4。LLM へ提示するスキーマは本型から作らない）。
- `label` の render は `string.Template.substitute` を使う（`re` を使わない・NFR-6）。
- 決定的段（`_plan_slots`）と予測段のスキーマ生成（`_slots_model`）は**非公開**である。
  案 1 の 3 呼び出しは `ActionPlanner.plan()` へ畳まれ、実装だけが各モジュールへ残る
  （設計 §3.13 / §3.5b）。
- パス解決は自前で書かず `actions._resolve_path` を呼ぶ。同じ規則を 2 箇所へ書くと片方
  だけが直される drift が起きる（設計 §3.4b。利用者は本モジュールを含めて 3 つある）。
- 本モジュールから `actions.py` への依存は一方向である（逆向きの辺を作らない）。
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from string import Template
from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from ._models import build_frozen_model
from ._readonly import ReadOnlyStrMapping
from .actions import (
    ActionCatalog,
    ActionSpec,
    ParameterSpec,
    _model_name,
    _resolve_path,
    resolve_on_invalid_slot,
    resolve_prompt,
    resolve_prompt_vars,
)
from .types import ConfidenceLevel, ExecutableIntent, ExecutableSuggestion, IntentContext

#: 未解決スロットを `label` へ render するときの差し込み文字。値がまだ無いことを 1 文字で
#: 示すためのもので、`model_dump()` ではなく表示専用の経路にのみ現れる。
_UNRESOLVED_LABEL = "…"


class SlotState(StrEnum):
    """スロット 1 つの充足状態。値は接頭辞なし snake_case で公開契約（設計 §3.5）。"""

    RESOLVED = "resolved"
    NEEDS_LLM = "needs_llm"
    NEEDS_CONFIRMATION = "needs_confirmation"
    NEEDS_USER = "needs_user"


class Origin(StrEnum):
    """解決済みスロットの値の出どころ。値は接頭辞なし snake_case で公開契約（設計 §3.5）。

    案 1 の `"run_context:<パス>"` / `"agent:<名前>"` という連結文字列を、`Origin` と
    `Slot.detail` の 2 フィールドへ分解した。利用者が接頭辞を切り出すコードを書かない。
    """

    CANDIDATE = "candidate"
    RUN_CONTEXT = "run_context"
    DEFAULT = "default"
    LLM = "llm"
    USER_CONFIRMED = "user_confirmed"
    USER_INPUT = "user_input"


#: `from_user` が真になる出どころ。利用者の意思が通った値だけをアプリが記憶する
#: （`CANDIDATE` / `RUN_CONTEXT` / `DEFAULT` / `LLM` は記憶しない・設計 §3.5a の表）。
_USER_ORIGINS: frozenset[Origin] = frozenset({Origin.USER_INPUT, Origin.USER_CONFIRMED})


class SlotSuggestion[T](BaseModel):
    """スロット 1 件に対する候補値 1 つ。"""

    model_config = {"frozen": True}
    value: T = Field(description="Suggested value for the slot.")
    level: ConfidenceLevel = Field(
        default=ConfidenceLevel.CERTAIN,
        description=(
            "Confidence in this suggestion. Defaults to CERTAIN because a deterministically"
            " resolved value is not a guess."
        ),
    )
    rationale: str | None = Field(default=None, description="One-sentence justification.")


class Slot(BaseModel):
    """アクション 1 件のパラメータ 1 つの充足状態。"""

    model_config = {"frozen": True}
    name: str = Field(description="Parameter name this slot corresponds to.")
    state: SlotState = Field(description="Which of the four states this slot is in.")
    value: Any = Field(
        default=None,
        description="Resolved value. Meaningful only when state is RESOLVED.",
    )
    origin: Origin | None = Field(
        default=None, description="Where the value came from. None while unresolved."
    )
    detail: str | None = Field(
        default=None,
        description="Resolved run context path. Non-None only when origin is RUN_CONTEXT.",
    )
    suggestions: tuple[SlotSuggestion[Any], ...] = Field(
        default=(), description="Candidate values to show the user."
    )

    @property
    def from_user(self) -> bool:
        """値を利用者自身が入力または確認したかどうか。

        アプリが「自分のストアへ書き戻すべき値」を選り分ける唯一の判定である。`origin` が
        `None` のスロット（`NEEDS_LLM` / `NEEDS_USER`）でも例外を出さず `False` を返す。

        Returns:
            `origin` が `USER_INPUT` / `USER_CONFIRMED` のいずれかなら True。
        """
        return self.origin in _USER_ORIGINS

    @model_validator(mode="after")
    def _check_state_fields(self) -> Slot:
        """状態とフィールドの整合を 6 条件で強制する（設計 §3.5 の表）。

        `mode="after"` なのは、既定値が入った後の最終的な組み合わせを判定する必要がある
        ためである（`mode="before"` だと省略されたフィールドが未確定で、9 通りの一部が
        誤って通る）。

        Returns:
            検証済みの自分自身。

        Raises:
            ValueError: 次のいずれかに該当する場合。(1) `NEEDS_LLM` / `NEEDS_USER` で
                `origin` または `value` が非 `None`（未解決スロットが値を持つ）、
                (2) `NEEDS_LLM` で `suggestions` が非空（予測待ちなのに選択肢が載る）、
                (3) `NEEDS_CONFIRMATION` で `value` が非 `None`（値が `value` と
                `suggestions` へ二重に載る）、(4) `NEEDS_CONFIRMATION` で `suggestions` が
                空（UI に選択肢が出ず、`apply` が `USER_CONFIRMED` を判別できず
                `USER_INPUT` へ誤分類する）、(5) `RESOLVED` で `origin` が `None`
                （`from_user` が常に偽になり利用者が決めた値が書き戻し対象から黙って
                漏れる）、(6) `detail` が非 `None` で `origin` が `RUN_CONTEXT` でない
                （`detail` は run context の解決パスであり、`Origin.LLM` の `detail` は
                常に `None`）。

        Note:
            `RESOLVED` + `value is None` は**意図的に通す**。`param(..., default=None)` を
            明示宣言した場合に正当に発生する組み合わせであり、`value` の `None` は
            「未解決」ではなく「値が `None` であること」を意味する。未解決かどうかは
            `state` が持つ。
        """
        if self.state in (SlotState.NEEDS_LLM, SlotState.NEEDS_USER) and (
            self.origin is not None or self.value is not None
        ):
            raise ValueError(f"{self.state} slot must not carry origin or value: {self.name!r}")
        if self.state is SlotState.NEEDS_LLM and self.suggestions:
            raise ValueError(f"{self.state} slot must not carry suggestions: {self.name!r}")
        if self.state is SlotState.NEEDS_CONFIRMATION:
            if self.value is not None:
                raise ValueError(
                    f"{self.state} slot must keep value None and use suggestions: {self.name!r}"
                )
            if not self.suggestions:
                raise ValueError(
                    f"{self.state} slot must have at least one suggestion: {self.name!r}"
                )
        if self.state is SlotState.RESOLVED and self.origin is None:
            raise ValueError(f"{self.state} slot must record an origin: {self.name!r}")
        if self.detail is not None and self.origin is not Origin.RUN_CONTEXT:
            raise ValueError(
                f"detail is only meaningful for {Origin.RUN_CONTEXT} origin: {self.name!r}"
            )
        return self


class ActionPlan(BaseModel):
    """候補 1 件に対する実行計画（宣言順のスロット列 + 解決済みの既定）。"""

    model_config = {"frozen": True}
    action_id: str = Field(description="The action this plan will execute.")
    slots: tuple[Slot, ...] = Field(description="Slots in ActionSpec.parameters declaration order.")
    spec: ActionSpec = Field(
        exclude=True,
        description="The declaration as-declared (before merging catalog-wide defaults).",
    )
    action_agent: str = Field(
        exclude=True,
        description="Name of the executing agent. The instance itself is never held.",
    )
    resolved_prompt: tuple[str, ...] = Field(
        exclude=True, description="Prompt segments after merging the catalog-wide ones."
    )
    # 読み取り専用へ正規化する（`_readonly.ReadOnlyStrMapping`）。マージ結果は LLM プロンプト
    # へ展開される**直前**の値であり、`Field(exclude=True)` は `model_dump()` からの除外で
    # あって書き換えを防がない（`resolve_prompt_vars()` は毎回新しい素の dict を返すため、
    # 保持した後に読み取り専用化する）。
    resolved_prompt_vars: ReadOnlyStrMapping = Field(
        exclude=True, description="Prompt variables after merging the catalog-wide ones."
    )
    resolved_on_invalid_slot: Literal["error", "skip"] = Field(
        exclude=True, description="Behaviour for invalid slots after override resolution."
    )

    @property
    def pending(self) -> tuple[Slot, ...]:
        """利用者に聞く必要があるスロットを宣言順で返す。

        「どの状態を利用者に聞くべきか」の定義はライブラリ側の知識であるため導出して
        提供する。`RESOLVED` は確定済みで、`NEEDS_LLM` は予測段が埋める対象である。

        Returns:
            `NEEDS_CONFIRMATION` と `NEEDS_USER` のスロットを宣言順に並べた tuple。
        """
        return tuple(
            slot
            for slot in self.slots
            if slot.state in (SlotState.NEEDS_CONFIRMATION, SlotState.NEEDS_USER)
        )

    @property
    def ready(self) -> bool:
        """全スロットが確定していて実行入力を組めるかどうか。

        Returns:
            全スロットが `RESOLVED` なら True（スロット 0 件なら True）。
        """
        return all(slot.state is SlotState.RESOLVED for slot in self.slots)

    @property
    def label(self) -> str:
        """`ActionSpec.label` の `${name}` をスロットの値で render した表示名を返す。

        未解決スロットは `"…"` として render する。`re` は使わず `string.Template` の
        `substitute` で置換する（NFR-6）。プレースホルダとパラメータ名の対応は起動時検証
        （`planner.validate()`）が確認済みである前提で、`safe_substitute` ではなく
        `substitute` を使い、宣言の取りこぼしを黙って通さない。

        Returns:
            render 済みの表示名。
        """
        values = {
            slot.name: slot.value if slot.state is SlotState.RESOLVED else _UNRESOLVED_LABEL
            for slot in self.slots
        }
        return Template(self.spec.label).substitute(values)

    @property
    def input_json(self) -> str:
        """全スロットの値を型検証したうえで直列化した実行入力を返す（FR-8 L235 / L236）。

        `Runner.run(registry.get(plan.action_agent), input=plan.input_json)` の引数に
        そのまま渡せる形である。検証せず素の値を直列化すると、候補生成器が載せた型違いの
        値が実行入力へ抜ける。型付きインスタンスが必要な利用者は、公開されている
        `spec.parameters_model()` から自分で組める（設計 §3.5b）。

        Returns:
            `spec.parameters_model()` で検証した値を宣言順に並べた JSON 文字列。

        Raises:
            ValueError: `ready` が偽の場合。文言には**未解決スロット名だけ**を列挙する
                （解決済みの名前を混ぜると、どれを埋めればよいのかが読み取れない）。
            ValidationError: いずれかのスロットの値が宣言型に合わない場合。文言には検証に
                失敗した入力値そのものが載る。候補 / LLM 由来の値を含むため、この例外を
                そのままログや API 応答へ流さないこと。
        """
        unresolved = [slot.name for slot in self.slots if slot.state is not SlotState.RESOLVED]
        if unresolved:
            raise ValueError(
                f"cannot build the execution input while slots are unresolved: "
                f"{', '.join(unresolved)}"
            )
        model = self.spec.parameters_model()
        return model(**{slot.name: slot.value for slot in self.slots}).model_dump_json()

    def apply(self, answers: Mapping[str, Any]) -> ActionPlan:
        """利用者の確認結果と穴埋め入力を合流させた**新しい**計画を返す（FR-8 L230-L234）。

        元のインスタンスは変更しない。UI は「押す前の計画」を握ったまま本メソッドを呼ぶ
        ため、元が書き換わると再描画の基準が消える。

        検証は「未知キー全件 -> 既 `RESOLVED` 全件 -> 型検証」の順に、いずれも `answers`
        全体に対して行ってから 1 つも欠けずに適用する。1 件ずつ処理して途中で落とすと、
        「拒否したときは元が不変」と「違反キーを全件列挙する」が同時に壊れる。

        `USER_CONFIRMED` / `USER_INPUT` の判別は、対象が `NEEDS_CONFIRMATION` で答えが
        `suggestions` のいずれかの値と**等しい**（`==`）かどうかで行う。`is` / `id()` で
        判定しないのは、`tuple[SlotSuggestion[Any], ...]` がパラメータ化ジェネリック注釈
        であり検証で作り直されて identity が保存されないためである（等価性は保たれる）。

        Args:
            answers: パラメータ名 -> 利用者が確定させた値。空でもよい。

        Returns:
            対象スロットを `RESOLVED` へ遷移させた新しい `ActionPlan`。`spec` /
            `action_agent` / `resolved_*` の 5 件はそのまま引き継がれる。

        Raises:
            ValueError: 宣言済みパラメータ名でないキーが含まれる場合（全件列挙する）、
                または既に `RESOLVED` のスロットを指すキーが含まれる場合（確定済み値の
                黙示的な上書きを禁止する・全件列挙する）。
            ValidationError: いずれかの値が宣言型に合わない場合。未解決スロットが残る段
                では全件モデルによる検証を行えないため、当該パラメータ単体を
                `TypeAdapter(<param の annotation>)` で検証する。文言には検証に失敗した
                入力値そのものが載る。候補 / LLM 由来の値を含むため、この例外をそのまま
                ログや API 応答へ流さないこと。
        """
        slots_by_name = {slot.name: slot for slot in self.slots}
        unknown = [name for name in answers if name not in slots_by_name]
        if unknown:
            raise ValueError(f"unknown answer keys for {self.action_id!r}: {', '.join(unknown)}")
        overwritten = [name for name in answers if slots_by_name[name].state is SlotState.RESOLVED]
        if overwritten:
            raise ValueError(
                f"cannot overwrite slots that are already resolved: {', '.join(overwritten)}"
            )
        params_by_name = {param.name: param for param in self.spec.parameters}
        validated = {
            name: TypeAdapter(params_by_name[name].annotation).validate_python(value)
            for name, value in answers.items()
        }
        applied = tuple(
            _answer_slot(slot, validated[slot.name]) if slot.name in validated else slot
            for slot in self.slots
        )
        return self.model_copy(update={"slots": applied})


class ParamUsage(BaseModel):
    """予測段の実行量。予測を行わなかった場合は 0 件を表す値が入る（FR-6）。"""

    model_config = {"frozen": True}
    runs: int = Field(description="How many times the runner was driven (0 or 1).")
    model_calls: int = Field(description="How many model calls the run consumed.")
    candidates: int = Field(description="How many candidates the prediction covered.")
    input_tokens: int | None = Field(
        default=None, description="Input tokens, or None when usage was not reported."
    )
    output_tokens: int | None = Field(
        default=None, description="Output tokens, or None when usage was not reported."
    )


class PlanResult(BaseModel):
    """`planner.plan(detail=True)` の戻り。計画・候補生成の結果一式・実行量を束ねる。"""

    model_config = {"frozen": True}
    plans: tuple[ActionPlan, ...] = Field(
        description="Plans in candidate order, one per accepted candidate."
    )
    suggestion: ExecutableSuggestion = Field(
        description="The candidate generation result, passed through without loss."
    )
    usage: ParamUsage = Field(description="How much the prediction stage actually consumed.")


def _plan_slots(
    candidates: tuple[ExecutableIntent, ...],
    catalog: ActionCatalog,
    context: IntentContext[Any],
) -> tuple[ActionPlan, ...]:
    """候補ごとに空欄の確定を行い、候補と同順・同数の `ActionPlan` を返す（FR-5）。

    LLM 実行アダプタ・ネットワーク・環境変数を一切参照しないため同期関数である。同一の
    候補列と同一の `run_context` に対し常に同一結果を返す（`await` を要さないことが、
    この段が LLM を呼ばないことの構造的な表現になる）。候補も宣言簿も変更しない。

    非公開なのは、案 1 の 3 呼び出しが `ActionPlanner.plan()` へ畳まれ、実装だけが各
    モジュールへ残る契約に従うためである（設計 §3.13）。

    Args:
        candidates: 登録済み `action_id` へ絞り込み済みの候補列（段 (1) の出力）。
        catalog: 宣言簿。`action_id` の解決と既定マージ解決に使う。
        context: 発話と `run_context` を載せた `IntentContext`。

    Returns:
        候補と同順・同数の `ActionPlan` の tuple。候補 0 件なら空 tuple。

    Raises:
        KeyError: 候補の `action_id` が宣言簿に無い場合（段 (1) が allowlist 除外を
            済ませている前提であり、ここで防御的に受け入れると未登録候補が下流へ流れる）。
    """
    return tuple(_plan_candidate(candidate, catalog, context) for candidate in candidates)


def _plan_candidate(
    candidate: ExecutableIntent,
    catalog: ActionCatalog,
    context: IntentContext[Any],
) -> ActionPlan:
    """候補 1 件ぶんの `ActionPlan` を組む。

    既定（`prompt` / `prompt_vars` / `on_invalid_slot`）の解決は `actions.py` の純関数
    3 件へ委ね、結果を `resolved_*` へ運ぶ。`spec` はマージ結果を書き戻さず as-declared の
    まま保持する。起動時検証の「当該 `ActionSpec` 自身の宣言に限る」検査（FR-3 L156）が、
    マージ済みの値を見てしまうと成立しなくなるためである（設計 §3.4a）。

    Args:
        candidate: 対象の候補 1 件。
        catalog: 宣言簿。
        context: 発話と `run_context` を載せた `IntentContext`。

    Returns:
        宣言順のスロット列と解決済み既定を載せた `ActionPlan`。
    """
    spec = catalog.get(candidate.action_id)
    return ActionPlan(
        action_id=spec.action_id,
        slots=tuple(_resolve_slot(param, candidate, context) for param in spec.parameters),
        spec=spec,
        action_agent=spec.action_agent,
        resolved_prompt=resolve_prompt(catalog, spec),
        resolved_prompt_vars=resolve_prompt_vars(catalog, spec),
        resolved_on_invalid_slot=resolve_on_invalid_slot(catalog, spec),
    )


def _resolve_slot(
    param: ParameterSpec,
    candidate: ExecutableIntent,
    context: IntentContext[Any],
) -> Slot:
    """パラメータ 1 つの値を解決順に沿って決め、対応する `Slot` を組む（FR-5 L179-L184）。

    解決順は「候補の `parameters`」->「`from_context` の宣言順パス解決」->「`by_llm` に
    よる予測（この段では未実施）」->「`default`」->「利用者入力」である。

    `by_llm` を `default` より**先**に見ることが要点である。`by_llm=True` かつ `default`
    宣言ありのパラメータをここで `DEFAULT` へ倒すと、予測が一度も走らない。宣言した
    `default` は FR-7 の後退先として予測段が使う。

    Args:
        param: 対象のパラメータ宣言。
        candidate: 値を載せている可能性がある候補。
        context: `from_context` の解決対象 `run_context` を持つ `IntentContext`。

    Returns:
        解決結果に対応する `Slot`。値を得られなければ `NEEDS_LLM`（`by_llm=True`）または
        `NEEDS_USER`（`by_llm=False`）。
    """
    if param.name in candidate.parameters:
        return _settle_slot(param, candidate.parameters[param.name], Origin.CANDIDATE, None)
    for path in param.from_context:
        value = _resolve_path(context.run_context, path)
        if value is not None:
            return _settle_slot(param, value, Origin.RUN_CONTEXT, path)
    if param.by_llm:
        return Slot(name=param.name, state=SlotState.NEEDS_LLM)
    if param.has_default:
        return _settle_slot(param, param.default, Origin.DEFAULT, None)
    return Slot(name=param.name, state=SlotState.NEEDS_USER)


def _settle_slot(param: ParameterSpec, value: Any, origin: Origin, detail: str | None) -> Slot:
    """値を得たパラメータを `confirm` 宣言に応じた状態の `Slot` へ落とす（FR-5 L180 / L181）。

    `confirm=True` の場合に値を `value` ではなく `suggestions` の 1 件として載せるのは、
    同じ値が 2 箇所へ二重に載ることを `Slot` の validator が禁じているためである
    （設計 §3.5 の条件 3）。`level` は宣言側の既定値（`CERTAIN`）に委ねる。決定的に解決
    した値は推測ではないためであり、ここで明示すると既定の意味が 2 箇所に分かれる。

    Args:
        param: 対象のパラメータ宣言。
        value: 解決した値。`None` でもよい（明示的な `default=None` の場合）。
        origin: 値の出どころ。
        detail: `RUN_CONTEXT` なら解決に成功したパス、それ以外は `None`。

    Returns:
        `confirm=True` なら `NEEDS_CONFIRMATION`、偽なら `RESOLVED` の `Slot`。
    """
    if param.confirm:
        return Slot(
            name=param.name,
            state=SlotState.NEEDS_CONFIRMATION,
            origin=origin,
            detail=detail,
            suggestions=(SlotSuggestion(value=value),),
        )
    return Slot(
        name=param.name,
        state=SlotState.RESOLVED,
        value=value,
        origin=origin,
        detail=detail,
    )


def _answer_slot(slot: Slot, value: Any) -> Slot:
    """利用者が確定させた値でスロットを `RESOLVED` へ遷移させる（FR-8 L231）。

    突き合わせは `SlotSuggestion.value` の等価（`==`）で行い、`level` の一致は要求しない。
    `is` / `id()` を使わないのは、`suggestions` が検証で作り直されて identity が保存され
    ないためである（バッチ 1 で pin 済み）。

    Args:
        slot: 遷移させる元のスロット。
        value: 型検証済みの値。

    Returns:
        `RESOLVED` の新しい `Slot`。元が `NEEDS_CONFIRMATION` で値が `suggestions` の
        いずれかと等しければ `origin=USER_CONFIRMED`、それ以外は `origin=USER_INPUT`。
        `detail` は利用者由来のため常に `None` である（validator 条件 6）。
    """
    confirmed = slot.state is SlotState.NEEDS_CONFIRMATION and any(
        suggestion.value == value for suggestion in slot.suggestions
    )
    return Slot(
        name=slot.name,
        state=SlotState.RESOLVED,
        value=value,
        origin=Origin.USER_CONFIRMED if confirmed else Origin.USER_INPUT,
    )


def _slots_model(plan: ActionPlan) -> type[BaseModel]:
    """`NEEDS_LLM` のパラメータだけをフィールドに持つ提示用スキーマモデルを組む（FR-2 L124）。

    解決済みのパラメータを載せると、LLM が既に確定した値を上書きできてしまう。フィールド
    型は `max_suggestions` が 1 なら `SlotSuggestion[T]`、2 以上なら
    `list[SlotSuggestion[T]]` + `max_length`（`model_json_schema()` の `maxItems` として
    LLM への上限提示に使う）である。応答検証に使う parse 派生は
    `_models.derive_optional_model` が組み、そちらには `max_length` が残らない（設計 §3.8）。

    annotation（`SlotSuggestion[T]`）の組み立てを本モジュールで行うのは、`_models` が
    ドメイン型を 1 つも import しない契約であるためである（設計 §3.1）。生成そのものは
    `build_frozen_model` へ委ねる。

    非公開なのは、予測が `planner.plan()` の内部へ畳まれ利用者が呼ぶ場面が無いためである。
    公開すると「UI のフォーム生成には `parameters_model()`」という導線が二重になる
    （設計 §3.5b）。

    Args:
        plan: 対象の計画。スロットの状態と `spec.parameters` の宣言順を参照する。

    Returns:
        `NEEDS_LLM` のパラメータを宣言順にフィールドとして持つ frozen なモデル。該当が
        1 件も無ければフィールド 0 件のモデル。

    Raises:
        ValueError: いずれかの annotation を pydantic がフィールド型として扱えない場合。
    """
    needs_llm = {slot.name for slot in plan.slots if slot.state is SlotState.NEEDS_LLM}
    fields: dict[str, tuple[Any, Any]] = {
        param.name: (
            list[SlotSuggestion[param.annotation]],  # type: ignore[misc, name-defined]
            Field(description=param.description, max_length=param.max_suggestions),
        )
        if param.max_suggestions > 1
        else (
            SlotSuggestion[param.annotation],  # type: ignore[misc, name-defined]
            Field(description=param.description),
        )
        for param in plan.spec.parameters
        if param.name in needs_llm
    }
    return build_frozen_model(_model_name(plan.spec.action_id, "Slots"), fields)
