"""L1: `runtime.intent.types` に追加する候補型 (`ExecutableIntent` / `ExecutableSuggestion`)
の純検証。

FR-4 の受け入れ基準（タスク 1-9）を pin する。`ExecutableIntent` が既存 `IntentCandidate` の
サブクラスとして成立し `text` を `action_id` から自動補完すること、`ExecutableSuggestion` が
`IntentPrediction` を経由せず `tuple[ExecutableIntent, ...]` を直接持つこと（設計 §3.10・
実測 1）、`run_context` に任意型が載るため直列化の成立を契約に含めないことを対象とする。
外部依存 (agents / openai) なし。
"""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from oai_agentspec.runtime.intent.types import (
    ConfidenceLevel,
    ConsistencyReport,
    ExecutableIntent,
    ExecutableSuggestion,
    IntentCandidate,
    IntentContext,
    IntentPrediction,
)

pytestmark = pytest.mark.unit


class _OpaqueRunContext:
    """利用者の任意型 run_context の代わり。直列化できないことに意味がある。"""

    def __init__(self, host: str) -> None:
        self.host = host


def _make_intent(**overrides: Any) -> ExecutableIntent:
    """テスト用の最小 ExecutableIntent を組む。"""
    fields: dict[str, Any] = {
        "action_id": "run_load_test",
        "parameters": {"seconds": 30},
        "level": ConfidenceLevel.HIGH,
        "source": "rule",
    }
    fields.update(overrides)
    return ExecutableIntent(**fields)


# ---------------------------------------------------------------------------
# ExecutableIntent の型と text 自動補完 (FR-4 L162 / L163)
# ---------------------------------------------------------------------------


def test_executable_intent_is_an_intent_candidate_subclass() -> None:
    """ExecutableIntent は既存 IntentCandidate のサブクラスである (FR-4 L162)。

    既存の候補契約（`IntentPrediction.candidates`）へそのまま載せられることが前提である。
    """
    intent = _make_intent()
    assert isinstance(intent, IntentCandidate)
    assert issubclass(ExecutableIntent, IntentCandidate)


def test_executable_intent_is_frozen() -> None:
    """ExecutableIntent は frozen である (FR-4 L162)。"""
    intent = _make_intent()
    assert isinstance(intent, BaseModel)
    with pytest.raises(ValidationError):
        intent.action_id = "other"  # type: ignore[misc]


def test_executable_intent_fills_text_from_action_id() -> None:
    """text は action_id から自動補完される（利用者は text を渡さない）(FR-4 L162)。

    親型の必須フィールド `text` を利用者に二重記述させないための補完である。
    """
    intent = _make_intent()
    assert intent.text == "run_load_test"


def test_executable_intent_fills_text_on_the_validate_path() -> None:
    """dict からの検証経路でも text が補完される（mode="before" の validator）(FR-4 L162)。"""
    intent = ExecutableIntent.model_validate(
        {"action_id": "run_load_test", "level": "high", "source": "rule"}
    )
    assert intent.text == "run_load_test"


def test_executable_intent_accepts_a_matching_explicit_text() -> None:
    """text と action_id を双方明示して一致するなら成立する (FR-4 L163)。"""
    intent = _make_intent(text="run_load_test")
    assert intent.text == "run_load_test"


def test_executable_intent_rejects_a_mismatching_explicit_text() -> None:
    """text と action_id が不一致なら ValueError (FR-4 L163)。

    黙ってどちらかを採ると、候補の表示名と実行先の宣言が食い違ったまま下流へ流れる。
    """
    with pytest.raises(ValueError):
        _make_intent(text="run_smoke_test")


def test_executable_intent_defaults_parameters_to_an_empty_mapping() -> None:
    """parameters を省略すると空の Mapping として成立する (FR-4 L165)。"""
    intent = ExecutableIntent(action_id="run_load_test", level=ConfidenceLevel.LOW, source="rule")
    assert dict(intent.parameters) == {}


def test_executable_intent_holds_the_declared_parameters() -> None:
    """parameters は宣言した Mapping をそのまま保持する (FR-4 L162)。"""
    intent = _make_intent(parameters={"seconds": 60, "target": "api.example.com"})
    assert dict(intent.parameters) == {"seconds": 60, "target": "api.example.com"}


@pytest.mark.parametrize("level", list(ConfidenceLevel))
def test_executable_intent_accepts_all_five_confidence_levels(level: ConfidenceLevel) -> None:
    """level は既存 ConfidenceLevel の 5 段階に限定される (FR-4 L166)。"""
    assert _make_intent(level=level).level is level


def test_executable_intent_rejects_an_unknown_level() -> None:
    """ConfidenceLevel 以外の level は ValidationError (FR-4 L166)。"""
    with pytest.raises(ValidationError):
        _make_intent(level="bogus")


def test_executable_intent_keeps_source_as_an_unvalidated_str() -> None:
    """source は候補の生成系統を str として保持し、値の集合は検証しない (FR-4 L167)。"""
    intent = _make_intent(source="my_experimental_ranker")
    assert intent.source == "my_experimental_ranker"


def test_executable_intent_requires_source() -> None:
    """source は必須である（候補の生成系統は generator 側が常に知っている）(FR-4 L167)。

    省略可能へ後退させると、除外ログ（FR-4 の WARNING 1 行）や複数 generator を併用した
    ときの切り分けが黙って効かなくなる。
    """
    with pytest.raises(ValidationError):
        ExecutableIntent(action_id="run_load_test", level=ConfidenceLevel.HIGH)


def test_executable_intent_rationale_defaults_to_none() -> None:
    """rationale の既定は None（親型 IntentCandidate と同じ）(FR-4 L162)。"""
    assert _make_intent().rationale is None


# ---------------------------------------------------------------------------
# IntentPrediction へ載せたときのサブクラス保持 (FR-4 L164 / 実測 1)
# ---------------------------------------------------------------------------


def test_executable_intent_survives_intent_prediction_instance_path() -> None:
    """インスタンス経路では検証後もサブクラスと追加フィールドが保持される (FR-4 L164)。"""
    intent = _make_intent()
    prediction = IntentPrediction(candidates=(intent,))
    kept = prediction.candidates[0]
    assert isinstance(kept, ExecutableIntent)
    assert kept.action_id == "run_load_test"
    assert dict(kept.parameters) == {"seconds": 30}


def test_executable_intent_survives_model_validate_of_instances() -> None:
    """model_validate へインスタンスを渡す経路でも保持される (FR-4 L164 / 実測 1-C)。"""
    intent = _make_intent()
    prediction = IntentPrediction.model_validate({"candidates": [intent]})
    assert isinstance(prediction.candidates[0], ExecutableIntent)


def test_intent_prediction_loses_the_subclass_on_the_plain_dict_path() -> None:
    """純 dict 経路では IntentCandidate へ coerce され追加フィールドが落ちる（実測 1-D）。

    これが `ExecutableSuggestion.candidates` を `tuple[ExecutableIntent, ...]` として
    直接持つ理由である（設計 §3.10）。JSON から復元した `IntentPrediction` を経由すると
    `action_id` / `parameters` が消えるため、経由させない形を固定する。
    """
    prediction = IntentPrediction.model_validate(
        {"candidates": [{"text": "run_load_test", "level": "high"}]}
    )
    candidate = prediction.candidates[0]
    assert type(candidate) is IntentCandidate
    assert not hasattr(candidate, "action_id")


def test_intent_prediction_dump_drops_the_subclass_fields() -> None:
    """親型スキーマで直列化されるため dump に action_id / parameters が現れない（実測 1-E）。

    dump 経路が契約外であることを明示的に pin する（設計 §3.10 の 2 点目）。
    """
    dumped = IntentPrediction(candidates=(_make_intent(),)).model_dump()
    assert set(dumped["candidates"][0]) == {"text", "level", "rationale"}


# ---------------------------------------------------------------------------
# ExecutableSuggestion (FR-4 L169 / 設計 §3.10)
# ---------------------------------------------------------------------------


def test_executable_suggestion_declares_exactly_four_fields() -> None:
    """フィールドは candidates / context / report / metadata の 4 件 (FR-4 L169 / §3.10)。

    `IntentPrediction` を丸ごと持たず、`report` / `metadata` は素通しで分解して持つ。
    """
    assert set(ExecutableSuggestion.model_fields) == {
        "candidates",
        "context",
        "report",
        "metadata",
    }
    assert "prediction" not in ExecutableSuggestion.model_fields


def test_executable_suggestion_holds_executable_intents_directly() -> None:
    """candidates は tuple[ExecutableIntent, ...] であり、サブクラスのフィールドが落ちない。

    IntentPrediction を経由しないため、検証後も action_id / parameters を読める
    （設計 §3.10・実測 1）。
    """
    intent = _make_intent()
    suggestion = ExecutableSuggestion(
        candidates=(intent,), context=IntentContext(utterance="負荷試験をしたい")
    )
    kept = suggestion.candidates[0]
    assert isinstance(kept, ExecutableIntent)
    assert kept.action_id == "run_load_test"
    assert dict(kept.parameters) == {"seconds": 30}


def test_executable_suggestion_keeps_subclass_fields_on_the_validate_path() -> None:
    """model_validate 経路でも ExecutableIntent のフィールドが保持される (設計 §3.10)。

    candidates の宣言型が ExecutableIntent であるため、純 dict からでも action_id が生きる。
    """
    suggestion = ExecutableSuggestion.model_validate(
        {
            "candidates": [{"action_id": "run_load_test", "level": "high", "source": "rule"}],
            "context": IntentContext(utterance="負荷試験をしたい"),
        }
    )
    assert suggestion.candidates[0].action_id == "run_load_test"


def test_executable_suggestion_passes_report_and_metadata_through() -> None:
    """report / metadata は generator が返した値を素通しで保持する (FR-4 L169)。"""
    report = ConsistencyReport(conflicts=("直前の発言と矛盾",))
    suggestion = ExecutableSuggestion(
        candidates=(_make_intent(),),
        context=IntentContext(utterance="負荷試験をしたい"),
        report=report,
        metadata={"generator": "rule"},
    )
    assert suggestion.report is report
    assert suggestion.metadata == {"generator": "rule"}


def test_executable_suggestion_report_and_metadata_default_to_none() -> None:
    """report / metadata の既定は None（判定しない generator もある）(FR-4 L169)。"""
    suggestion = ExecutableSuggestion(
        candidates=(), context=IntentContext(utterance="負荷試験をしたい")
    )
    assert suggestion.report is None
    assert suggestion.metadata is None


def test_executable_suggestion_is_frozen() -> None:
    """ExecutableSuggestion は frozen である (FR-1 L103)。"""
    suggestion = ExecutableSuggestion(
        candidates=(), context=IntentContext(utterance="負荷試験をしたい")
    )
    assert isinstance(suggestion, BaseModel)
    with pytest.raises(ValidationError):
        suggestion.candidates = (_make_intent(),)  # type: ignore[misc]


def test_executable_suggestion_accepts_an_arbitrary_run_context() -> None:
    """IntentContext.run_context に任意型が載るため直列化の成立を契約に含めない (FR-1 L105)。

    直列化できない利用者の型が run_context に載った状態でも宣言が成立することを固定する
    （dump が成立することは契約ではないため、ここでは dump を検査しない）。
    """
    run_context = _OpaqueRunContext("api.example.com")
    suggestion = ExecutableSuggestion(
        candidates=(_make_intent(),),
        context=IntentContext(utterance="負荷試験をしたい", run_context=run_context),
    )
    assert suggestion.context.run_context is run_context
    assert suggestion.context.utterance == "負荷試験をしたい"


def test_executable_intent_before_validator_passes_through_non_mapping_input() -> None:
    """action_id を持たない入力は補完せずそのまま通す (tester 申し送り 5)。

    text 補完は mode="before" で走るため、親型そのままの dict や Mapping でない値も
    到達する。ここで例外を出すと ExecutableIntent 以外の経路まで巻き込んで落ちる。
    """
    with pytest.raises(ValidationError) as non_mapping:
        ExecutableIntent.model_validate(["not-a-mapping"])
    assert non_mapping.value.errors()[0]["type"] == "model_type"

    with pytest.raises(ValidationError) as without_action_id:
        ExecutableIntent.model_validate({"text": "負荷試験", "source": "rule"})
    assert without_action_id.value.errors()[0]["type"] == "missing"


# ---------------------------------------------------------------------------
# ExecutableIntent.parameters は中身も書き換えられない (セキュリティレビュー指摘 #88-W2)
# ---------------------------------------------------------------------------
#
# `model_config = {"frozen": True}` は属性の再束縛だけを禁じる。`parameters` は
# `Mapping[str, Any]` を素の dict として保持するため `intent.parameters["k"] = v` が通り、
# 候補が運ぶ実行入力を宣言後に差し替えられる。`parameters` はスロット確定の第 1 優先の
# 値であり、下流の `ActionPlan.input_json` を通じて実行入力へ届く。


def test_executable_intent_parameters_rejects_item_assignment() -> None:
    """parameters への要素追加は TypeError (指摘 #88-W2)。"""
    intent = _make_intent()
    with pytest.raises(TypeError):
        intent.parameters["target"] = "api.internal"


def test_executable_intent_parameters_rejects_item_overwrite_and_deletion() -> None:
    """既存キーの上書きと削除も TypeError (指摘 #88-W2)。"""
    intent = _make_intent()
    with pytest.raises(TypeError):
        intent.parameters["seconds"] = 9999
    with pytest.raises(TypeError):
        del intent.parameters["seconds"]


def test_executable_intent_parameters_keep_their_values_after_a_rejected_write() -> None:
    """書き込みが弾かれた後も parameters は宣言時のまま (指摘 #88-W2)。"""
    intent = _make_intent()
    with pytest.raises(TypeError):
        intent.parameters["seconds"] = 9999
    assert dict(intent.parameters) == {"seconds": 30}


def test_executable_intent_default_parameters_are_read_only() -> None:
    """既定の空 parameters も書き換えられない (指摘 #88-W2)。"""
    intent = ExecutableIntent(action_id="run_load_test", level=ConfidenceLevel.LOW, source="rule")
    with pytest.raises(TypeError):
        intent.parameters["seconds"] = 30


def test_executable_intent_parameters_survive_the_validate_path_as_read_only() -> None:
    """model_validate 経路の parameters も読み取り専用 (指摘 #88-W2)。"""
    intent = ExecutableIntent.model_validate(
        {"action_id": "run_load_test", "level": "high", "source": "rule", "parameters": {"a": 1}}
    )
    with pytest.raises(TypeError):
        intent.parameters["a"] = 2


def test_executable_intent_copies_the_given_parameters() -> None:
    """宣言時に渡した dict を後から書き換えても中身が透けない (指摘 #88-W2)。"""
    source: dict[str, Any] = {"seconds": 30}
    intent = _make_intent(parameters=source)
    source["seconds"] = 9999
    source["target"] = "api.internal"
    assert dict(intent.parameters) == {"seconds": 30}


def test_executable_intent_parameters_are_still_readable_as_a_mapping() -> None:
    """読み取り専用にしても Mapping としての読み取りは従来どおり (指摘 #88-W2 の回帰防止)。"""
    intent = _make_intent(parameters={"seconds": 60, "target": "api.example.com"})
    assert intent.parameters["seconds"] == 60
    assert sorted(intent.parameters) == ["seconds", "target"]
    assert len(intent.parameters) == 2
    assert dict(intent.parameters) == {"seconds": 60, "target": "api.example.com"}


# ---------------------------------------------------------------------------
# 読み取り専用にした parameters でも複製・永続化できる (レビュー 2 巡目・指摘 #88-W2 の退行)
# ---------------------------------------------------------------------------
#
# 指摘 #88-W2 の修正で `parameters` を `MappingProxyType` へ正規化したところ、`mappingproxy`
# が pickle 不可であるため `copy.deepcopy` / `model_copy(deep=True)` / `pickle` の 3 経路が
# `TypeError` で落ちるようになった (修正前は素の dict で 3 経路とも成立していた)。
# `ExecutableIntent` は候補として利用者コードやセッション層を渡り歩く公開型であり、複製・
# 永続化は起こりうる。読み取り専用性と複製可能性を両立させ、「pickle を通すために素の dict へ
# 戻す」修正で #88-W2 が無言で巻き戻ることも同時に防ぐ。

#: 複製・永続化の 3 経路。どれか 1 つでも落ちれば候補をプロセス外へ運べない。
_CLONE_ROUTES: list[Any] = [
    pytest.param(copy.deepcopy, id="deepcopy"),
    pytest.param(lambda obj: obj.model_copy(deep=True), id="model-copy-deep"),
    pytest.param(lambda obj: pickle.loads(pickle.dumps(obj)), id="pickle"),
]


@pytest.mark.parametrize("clone", _CLONE_ROUTES)
def test_executable_intent_survives_cloning_with_its_parameters(
    clone: Callable[[Any], Any],
) -> None:
    """ExecutableIntent は 3 経路で複製でき parameters の値と実体の別が保たれる (#88-W2 の退行)。"""
    intent = _make_intent(parameters={"seconds": 30, "target": "api.example.com"})
    restored = clone(intent)
    assert dict(restored.parameters) == {"seconds": 30, "target": "api.example.com"}
    assert restored.action_id == intent.action_id
    assert restored.text == intent.text
    assert restored.parameters is not intent.parameters


@pytest.mark.parametrize("clone", _CLONE_ROUTES)
def test_executable_intent_parameters_stay_read_only_after_cloning(
    clone: Callable[[Any], Any],
) -> None:
    """複製後の parameters も読み取り専用のまま (#88-W2 の退行)。

    複製を通すために素の dict へ戻す修正だと値の一致だけは緑になるため、書き込みが
    `TypeError` であることを複製の先でも確かめる。
    """
    intent = _make_intent(parameters={"seconds": 30})
    restored = clone(intent)
    with pytest.raises(TypeError) as excinfo:
        restored.parameters["target"] = "api.internal"
    assert type(excinfo.value) is TypeError
    assert dict(restored.parameters) == {"seconds": 30}


def test_executable_intent_deepcopy_does_not_share_nested_parameter_values() -> None:
    """deepcopy した候補の parameters は入れ子の値まで共有しない (#88-W2 の退行)。

    Mapping 自体を使い回すと、読み取り専用でも入れ子の list / dict 経由で片方の変更が
    他方へ透ける。deepcopy の意味論を入れ子の 1 段で確かめる。
    """
    intent = _make_intent(parameters={"targets": ["api.example.com"]})
    restored = copy.deepcopy(intent)
    restored.parameters["targets"].append("api.internal")
    assert dict(intent.parameters) == {"targets": ["api.example.com"]}


# ---------------------------------------------------------------------------
# ExecutableSuggestion.metadata も中身を書き換えられない (レビュー指摘 #88-R4)
# ---------------------------------------------------------------------------
#
# 兄弟の `ExecutableIntent.parameters` は読み取り専用 Mapping へ正規化されているのに、
# `metadata` は素の `Mapping[str, Any] | None` の pass-through であり
# `suggestion.metadata["k"] = v` が通る。`frozen` が守るのは属性の再束縛だけであり、
# 同じファイル・同じ用途（宣言後に読むだけの Mapping）で守りが片側にしか無い状態になる。
# `context` は Mapping ではなくモデル（`IntentContext`）なので対象外。


def _make_suggestion(**overrides: Any) -> ExecutableSuggestion:
    """テスト用の最小 ExecutableSuggestion を組む。"""
    fields: dict[str, Any] = {
        "candidates": (_make_intent(),),
        "context": IntentContext(utterance="負荷試験をしたい"),
    }
    fields.update(overrides)
    return ExecutableSuggestion(**fields)


def test_executable_suggestion_metadata_rejects_item_assignment() -> None:
    """metadata への要素追加は TypeError (指摘 #88-R4)。"""
    suggestion = _make_suggestion(metadata={"generator": "rule"})
    with pytest.raises(TypeError) as excinfo:
        suggestion.metadata["generator"] = "llm"  # type: ignore[index]
    assert type(excinfo.value) is TypeError


def test_executable_suggestion_metadata_rejects_item_deletion() -> None:
    """metadata からの要素削除も TypeError (指摘 #88-R4)。"""
    suggestion = _make_suggestion(metadata={"generator": "rule"})
    with pytest.raises(TypeError):
        del suggestion.metadata["generator"]  # type: ignore[union-attr]


def test_executable_suggestion_metadata_keeps_its_values_after_a_rejected_write() -> None:
    """書き込みが弾かれた後も metadata は宣言時のまま (指摘 #88-R4)。"""
    suggestion = _make_suggestion(metadata={"generator": "rule"})
    with pytest.raises(TypeError):
        suggestion.metadata["generator"] = "llm"  # type: ignore[index]
    assert dict(suggestion.metadata) == {"generator": "rule"}  # type: ignore[arg-type]


def test_executable_suggestion_copies_the_given_metadata() -> None:
    """宣言時に渡した dict を後から書き換えても中身が透けない (指摘 #88-R4)。"""
    source: dict[str, Any] = {"generator": "rule"}
    suggestion = _make_suggestion(metadata=source)
    source["generator"] = "llm"
    source["injected"] = True
    assert dict(suggestion.metadata) == {"generator": "rule"}  # type: ignore[arg-type]


def test_executable_suggestion_metadata_survives_the_validate_path_as_read_only() -> None:
    """model_validate 経路の metadata も読み取り専用 (指摘 #88-R4)。"""
    suggestion = ExecutableSuggestion.model_validate(
        {
            "candidates": [{"action_id": "run_load_test", "level": "high", "source": "rule"}],
            "context": IntentContext(utterance="負荷試験をしたい"),
            "metadata": {"generator": "rule"},
        }
    )
    with pytest.raises(TypeError):
        suggestion.metadata["generator"] = "llm"  # type: ignore[index]


def test_executable_suggestion_metadata_accepts_none() -> None:
    """metadata は None を受け付ける（判定材料を返さない generator もある）(指摘 #88-R4)。

    読み取り専用化で Optional が潰れると、既定の None すら宣言できなくなる。
    """
    assert _make_suggestion(metadata=None).metadata is None
    assert _make_suggestion().metadata is None


def test_executable_suggestion_metadata_accepts_an_empty_mapping() -> None:
    """空の metadata は None へ潰されず空の Mapping のまま保持される (指摘 #88-R4)。"""
    suggestion = _make_suggestion(metadata={})
    assert suggestion.metadata is not None
    assert dict(suggestion.metadata) == {}


def test_executable_suggestion_metadata_is_still_readable_as_a_mapping() -> None:
    """読み取り専用にしても Mapping としての読み取りは従来どおり (指摘 #88-R4 の回帰防止)。"""
    suggestion = _make_suggestion(metadata={"generator": "rule", "elapsed_ms": 12})
    assert suggestion.metadata["generator"] == "rule"  # type: ignore[index]
    assert sorted(suggestion.metadata) == ["elapsed_ms", "generator"]  # type: ignore[arg-type]
    assert len(suggestion.metadata) == 2  # type: ignore[arg-type]
    assert suggestion.metadata == {"generator": "rule", "elapsed_ms": 12}


#: `ExecutableSuggestion` の複製経路。`pickle` を含めないのは、`context` に載る
#: `IntentContext[Any]`（parametrized generic の pydantic モデル）が metadata の実装に
#: 関わらず pickle 不可であり、直列化の成立が本型の契約に含まれないためである（FR-1 L105）。
_SUGGESTION_CLONE_ROUTES: list[Any] = [
    pytest.param(copy.deepcopy, id="deepcopy"),
    pytest.param(lambda obj: obj.model_copy(deep=True), id="model-copy-deep"),
]


@pytest.mark.parametrize("clone", _SUGGESTION_CLONE_ROUTES)
def test_executable_suggestion_survives_cloning_with_its_metadata(
    clone: Callable[[Any], Any],
) -> None:
    """ExecutableSuggestion は 3 経路で複製でき metadata の値と実体の別が保たれる (指摘 #88-R4)。

    `parameters` と同じく、読み取り専用化で複製・永続化が落ちてはならない
    （`MappingProxyType` を使うと 3 経路とも `TypeError` になる）。
    """
    suggestion = _make_suggestion(metadata={"generator": "rule", "elapsed_ms": 12})
    restored = clone(suggestion)
    assert dict(restored.metadata) == {"generator": "rule", "elapsed_ms": 12}
    assert restored.candidates[0].action_id == "run_load_test"
    assert restored.metadata is not suggestion.metadata


@pytest.mark.parametrize("clone", _SUGGESTION_CLONE_ROUTES)
def test_executable_suggestion_metadata_stays_read_only_after_cloning(
    clone: Callable[[Any], Any],
) -> None:
    """複製後の metadata も読み取り専用のまま (指摘 #88-R4)。

    複製を通すために素の dict へ戻す修正だと値の一致だけは緑になるため、書き込みが
    `TypeError` であることを複製の先でも確かめる。
    """
    suggestion = _make_suggestion(metadata={"generator": "rule"})
    restored = clone(suggestion)
    with pytest.raises(TypeError) as excinfo:
        restored.metadata["generator"] = "llm"
    assert type(excinfo.value) is TypeError
    assert dict(restored.metadata) == {"generator": "rule"}


@pytest.mark.parametrize("clone", _SUGGESTION_CLONE_ROUTES)
def test_executable_suggestion_with_none_metadata_survives_cloning(
    clone: Callable[[Any], Any],
) -> None:
    """metadata が None の候補一式も 3 経路で複製できる (指摘 #88-R4)。"""
    restored = clone(_make_suggestion())
    assert restored.metadata is None
