"""L1: `runtime.intent.binding` の結線宣言型 (`CandidateSource` / `LLMFiller`) の純検証。

FR-3 の受け入れ基準（タスク 1-2b）を pin する。両型が frozen であること、`generator` /
`context_builder` / `model` を不透明値として型検証せずに保持すること、`bind` まで持ち越さずに
宣言時へ前倒した 2 つの検証（`context_builder` と `history_limit` の排他・
`on_invalid_response` の値域）が生成時に落ちることを対象とする。外部依存 (agents / openai) なし。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

# ruff の isort は存在しないモジュールを第一者と判定できないため、binding.py が未実装の
# 間だけ I001 を報告する（実装後は既存ファイルと同じ並びで警告なしになる）。
from oai_agentspec.runtime.intent.binding import CandidateSource, LLMFiller

pytestmark = pytest.mark.unit


class _OpaqueGenerator:
    """`CandidateGenerator` Protocol を満たさない不透明値。型検証されないことの検出に使う。"""


class _OpaqueModel:
    """モデルの代わりに渡す不透明値。lib は中身を解釈しない。"""


def _builder(query: Any) -> Any:
    """`ContextBuilder` の代わりに渡す不透明値（呼ばれないこと自体は本ファイルの対象外）。"""
    return query


# ---------------------------------------------------------------------------
# CandidateSource の宣言と frozen 性 (FR-3 L137)
# ---------------------------------------------------------------------------


def test_candidate_source_holds_the_generator_as_an_opaque_value() -> None:
    """generator は不透明値としてそのまま保持される（型検証しない）(FR-3 L137)。

    Protocol を満たさないオブジェクトでも宣言時に落ちないことを固定する。候補生成方式
    （ルール / 学習 / LLM）に依存しない結線を宣言時に縛らないための契約である。
    """
    generator = _OpaqueGenerator()
    source = CandidateSource(generator=generator)
    assert source.generator is generator


def test_candidate_source_optional_fields_default_to_none() -> None:
    """context_builder / history_limit の既定はいずれも None (FR-3 L137 / L138)。

    `history_limit` の既定を None の sentinel にすることで「明示された 20」と「既定の 20」を
    区別できる（既定 builder は None のとき 20 を使う）。
    """
    source = CandidateSource(generator=_OpaqueGenerator())
    assert source.context_builder is None
    assert source.history_limit is None


def test_candidate_source_is_frozen() -> None:
    """CandidateSource は frozen な pydantic BaseModel である (FR-3 L137)。"""
    source = CandidateSource(generator=_OpaqueGenerator())
    assert isinstance(source, BaseModel)
    with pytest.raises(ValidationError):
        source.generator = _OpaqueGenerator()  # type: ignore[misc]


def test_candidate_source_holds_the_context_builder_as_an_opaque_value() -> None:
    """context_builder も不透明値として保持される（型検証しない）(FR-3 L137)。"""
    source = CandidateSource(generator=_OpaqueGenerator(), context_builder=_builder)
    assert source.context_builder is _builder


def test_candidate_source_accepts_history_limit_alone() -> None:
    """history_limit だけを渡す形は既定 builder への便宜引数として成立する (FR-3 L138)。"""
    source = CandidateSource(generator=_OpaqueGenerator(), history_limit=20)
    assert source.history_limit == 20
    assert source.context_builder is None


def test_candidate_source_rejects_both_context_builder_and_history_limit() -> None:
    """context_builder と history_limit の双方に非 None を渡すと ValidationError (FR-3 L138)。

    history_limit は既定 builder 専用の便宜引数であり、差し替えた builder には効かない。
    黙って無視されると「20 件に絞ったつもり」の宣言が効かないまま実行される。
    """
    with pytest.raises(ValidationError):
        CandidateSource(generator=_OpaqueGenerator(), context_builder=_builder, history_limit=20)


def test_candidate_source_requires_a_generator() -> None:
    """generator は必須である（候補生成は代替不能）(FR-3 L137)。"""
    with pytest.raises(ValidationError):
        CandidateSource()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# LLMFiller の宣言と frozen 性 (FR-3 L139 / L141)
# ---------------------------------------------------------------------------


def test_llm_filler_holds_the_model_as_an_opaque_value() -> None:
    """model は不透明値としてそのまま保持される（型検証しない）(FR-3 L139)。

    利用者はエージェントの実体を渡さず model のみを渡す（実体は lib が構築する）。
    """
    model = _OpaqueModel()
    filler = LLMFiller(model=model)
    assert filler.model is model


def test_llm_filler_optional_fields_have_declared_defaults() -> None:
    """on_invalid_response の既定は "error"、guardrails の既定は空 tuple (FR-3 L139)。

    guardrails が空なら予測エージェントにガードレールは 1 件も装着されない（opt-in）。
    """
    filler = LLMFiller(model=_OpaqueModel())
    assert filler.on_invalid_response == "error"
    assert filler.guardrails == ()


def test_llm_filler_is_frozen() -> None:
    """LLMFiller は frozen な pydantic BaseModel である (FR-3 L139)。"""
    filler = LLMFiller(model=_OpaqueModel())
    assert isinstance(filler, BaseModel)
    with pytest.raises(ValidationError):
        filler.on_invalid_response = "skip"  # type: ignore[misc]


def test_llm_filler_requires_a_model() -> None:
    """model は必須である（どこへ接続するかは lib が決められない）(FR-3 L139)。"""
    with pytest.raises(ValidationError):
        LLMFiller()  # type: ignore[call-arg]


@pytest.mark.parametrize("allowed", ["error", "skip"])
def test_llm_filler_accepts_the_two_declared_values(allowed: str) -> None:
    """on_invalid_response は "error" / "skip" の 2 値を受け付ける (FR-3 L141)。"""
    filler = LLMFiller(model=_OpaqueModel(), on_invalid_response=allowed)  # type: ignore[arg-type]
    assert filler.on_invalid_response == allowed


@pytest.mark.parametrize("bad", ["halt", "ERROR", "", "raise", "Skip", None, 0])
def test_llm_filler_rejects_other_on_invalid_response_values(bad: Any) -> None:
    """ "error" / "skip" 以外を渡すと生成時に ValidationError (FR-3 L141)。

    宣言時に落とし、予測が失敗した瞬間まで不正値を持ち越さない。
    """
    with pytest.raises(ValidationError):
        LLMFiller(model=_OpaqueModel(), on_invalid_response=bad)


def test_llm_filler_holds_the_guardrail_names() -> None:
    """guardrails は登録名の tuple として保持される（実体の直渡しは受けない）(FR-3 L139)。"""
    filler = LLMFiller(model=_OpaqueModel(), guardrails=("no_pii", "no_secrets"))
    assert filler.guardrails == ("no_pii", "no_secrets")


def test_llm_filler_does_not_hold_a_guardrail_registry() -> None:
    """解決簿は LLMFiller のフィールドではなく bind の引数である (設計 §3.4a)。

    「guardrails が非空なのに解決簿が無い」の検出は起動時検証（`planner.validate()`）が担う。
    LLMFiller 単体では解決簿を知らないため宣言時には落とせない。
    """
    assert "guardrail_registry" not in LLMFiller.model_fields


def test_llm_filler_declares_exactly_three_fields() -> None:
    """LLMFiller のフィールドは model / on_invalid_response / guardrails の 3 件 (FR-3 L139)。"""
    assert set(LLMFiller.model_fields) == {"model", "on_invalid_response", "guardrails"}


def test_candidate_source_declares_exactly_three_fields() -> None:
    """CandidateSource のフィールドは generator / context_builder / history_limit の 3 件。"""
    assert set(CandidateSource.model_fields) == {
        "generator",
        "context_builder",
        "history_limit",
    }


def test_candidate_source_rejects_a_history_limit_below_one() -> None:
    """history_limit は 0 以下を宣言時に落とす（FR-3）。

    DefaultContextBuilder は実行時に ValueError を出すが、それは planner.plan() の
    初回（毎ターンの窓口）であり、bind() / validate() を素通りする。binding の宣言型は
    「検証を bind まで持ち越さない」方針なので宣言時へ前倒す。
    """
    with pytest.raises(ValidationError):
        CandidateSource(generator=_OpaqueGenerator(), history_limit=0)


# ---------------------------------------------------------------------------
# generator は非 None であることを宣言時に落とす (レビュー指摘 #88-R2)
# ---------------------------------------------------------------------------
#
# `generator: Any` は `None` を宣言時に通し、誤りは `plan()` の中の
# `source.generator.generate(...)` で `AttributeError` になって初めて表面化する。
# 候補が押された瞬間まで発覚しない誤りを宣言時へ前倒すのが binding の方針であり
# （排他 validator・`LLMFiller` の Literal と同じ）、`generator` だけが例外になっている。


class _FalsyGenerator:
    """真偽値が偽になる不透明値。`if not generator` 実装の取りこぼし検出に使う。"""

    def __bool__(self) -> bool:
        """常に False を返す。

        Returns:
            常に False。
        """
        return False


def test_candidate_source_rejects_a_none_generator() -> None:
    """generator に None を渡すと宣言時に ValidationError (指摘 #88-R2)。

    型が `Any` であるため pydantic の必須検証を素通りする。`plan()` の実行時
    `AttributeError` へ持ち越さず、結線の宣言時に落とす。
    """
    with pytest.raises(ValidationError, match="generator") as excinfo:
        CandidateSource(generator=None)
    assert type(excinfo.value) is ValidationError


def test_candidate_source_rejects_a_none_generator_on_the_validate_path() -> None:
    """model_validate 経路でも None の generator は落ちる (指摘 #88-R2)。"""
    with pytest.raises(ValidationError, match="generator"):
        CandidateSource.model_validate({"generator": None})


def test_candidate_source_accepts_a_falsy_generator() -> None:
    """真偽値が偽の generator は通る（拒否条件は None かどうかである）(指摘 #88-R2)。

    `if not self.generator` で実装すると、`__bool__` が偽の generator まで巻き込んで
    落ちる。lib は generator を不透明値として扱うため、中身で判断してはならない。
    """
    generator = _FalsyGenerator()
    assert CandidateSource(generator=generator).generator is generator


@pytest.mark.parametrize(
    "generator",
    [_OpaqueGenerator(), object(), lambda: None, 0, "", False],
    ids=["opaque", "object", "callable", "zero", "empty-str", "false"],
)
def test_candidate_source_accepts_any_non_none_generator(generator: Any) -> None:
    """非 None の値はどれも不透明値として通る (指摘 #88-R2 の誤検知防止)。"""
    assert CandidateSource(generator=generator).generator is generator


def test_candidate_source_still_rejects_both_context_builder_and_history_limit() -> None:
    """generator の検証を足しても既存の排他 validator は効いたままである (指摘 #88-R2)。

    どちらの validator が落としたかを取り違えないよう、排他側のメッセージで照合する。
    """
    with pytest.raises(ValidationError, match="history_limit"):
        CandidateSource(generator=_OpaqueGenerator(), context_builder=_builder, history_limit=20)
