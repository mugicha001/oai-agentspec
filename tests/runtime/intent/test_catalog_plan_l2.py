"""L2: `ActionPlanner.plan()` の毎ターン契約（設計 §5 タスク 1-12 / §3.13 / §3.4a）。

段 (1) `_suggest` -> 段 (2) `slots` -> 段 (3) `_predict` の畳み込みと、`predict` / `detail`
の分岐、必須結線が欠けた場合の 2 規則（`candidates` 未結線 = `RuntimeError` / `llm_filler`
未結線 = 段 (3) をスキップして `ParamUsage(runs=0, ...)`）を pin する。あわせて初回 `plan()`
での `validate()` 1 回実行とその非再実行（`PrivateAttr` の実行済みフラグ）を対象とする。

**第 1 段では段 (3) の中身が存在しない**（`_predict` はタスク 2-7）。したがって `llm_filler`
を結線しても `NEEDS_LLM` は埋まらず `usage.runs == 0` のままであることを固定する。設計 §5 の
タスク 1-12 行「`predict=True` / `detail=True` の**分岐の存在**もここで定義し、段 (3) の中身
だけを 2-7 で埋める」に従った判断である。

実 SDK / 実 LLM は使わない。候補生成器は Fake で、LLM 実行アダプタ
（`_adapters.intent.run_intent_prompt`）は「呼ばれたら失敗する」関数へ差し替えて 0 回である
ことを構造的に確かめる。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import pytest

from oai_agentspec._adapters import intent as intent_adapter
from oai_agentspec.runtime.intent import _llm, _predict
from oai_agentspec.runtime.intent.actions import ActionCatalog, ActionSpec, param
from oai_agentspec.runtime.intent.binding import CandidateSource, LLMFiller
from oai_agentspec.runtime.intent.slots import (
    ActionPlan,
    Origin,
    ParamUsage,
    PlanResult,
    SlotState,
    _plan_slots,
)
from oai_agentspec.runtime.intent.types import (
    ConfidenceLevel,
    ConsistencyReport,
    ExecutableIntent,
    ExecutableSuggestion,
    IntentContext,
    IntentPrediction,
    IntentQuery,
)

pytestmark = pytest.mark.integration


# ---- Fake ----


class _CountingAgentRegistry:
    """`AgentRegistry` の Fake。`names()` の呼び出し回数を数える。"""

    def __init__(self, *names: str) -> None:
        self._names = sorted(names)
        self.name_calls = 0

    def names(self) -> list[str]:
        """登録済みエージェント名を昇順で返し、呼び出しを数える。"""
        self.name_calls += 1
        return list(self._names)

    def get(self, name: str) -> object:
        """未登録名なら `KeyError`（本物と同じ契約）。"""
        if name not in self._names:
            raise KeyError(name)
        return object()


class _RecordingGenerator:
    """`CandidateGenerator` の Fake。固定の予測を返し context を記録する。"""

    def __init__(self, prediction: IntentPrediction) -> None:
        self._prediction = prediction
        self.contexts: list[IntentContext[Any]] = []

    async def generate(self, context: IntentContext[Any]) -> IntentPrediction:
        """記録して固定の予測を返す。"""
        self.contexts.append(context)
        return self._prediction


class _RunContext:
    """`from_context` の解決先。"""

    def __init__(self, host: str = "api.example.com") -> None:
        self.host = host


# ---- ヘルパ ----


def _spec(
    action_id: str = "run_load_test",
    *,
    action_agent: str = "load_test_agent",
    label: str = "負荷試験",
    parameters: tuple[Any, ...] | None = None,
) -> ActionSpec:
    """健全な `ActionSpec` を組む（`label` は既定でプレースホルダを持たない）。"""
    return ActionSpec(
        action_id=action_id,
        description="負荷試験を実行する",
        action_agent=action_agent,
        label=label,
        parameters=parameters if parameters is not None else (param("seconds", int, default=30),),
    )


def _catalog(*specs: ActionSpec) -> ActionCatalog:
    """宣言簿を組んで `specs` を登録した `ActionCatalog` を返す。"""
    catalog = ActionCatalog()
    for spec in specs:
        catalog.register(spec)
    return catalog


def _intent(action_id: str, **parameters: Any) -> ExecutableIntent:
    """テスト用の `ExecutableIntent` を組み立てる。"""
    return ExecutableIntent(
        action_id=action_id,
        level=ConfidenceLevel.HIGH,
        parameters=parameters,
        source="rule",
    )


def _planner(
    catalog: ActionCatalog,
    prediction: IntentPrediction,
    *,
    registry: _CountingAgentRegistry | None = None,
    llm_filler: LLMFiller | None = None,
) -> Any:
    """候補生成器を結線した `ActionPlanner` を返す。"""
    return catalog.bind(
        registry=registry if registry is not None else _CountingAgentRegistry("load_test_agent"),
        candidates=CandidateSource(generator=_RecordingGenerator(prediction)),
        llm_filler=llm_filler,
    )


def _no_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM 実行アダプタ 2 件を「呼ばれたら失敗する」関数へ差し替える。

    段 (1) の入口（`run_intent_prompt`）と段 (3) の入口（`run_filler_prompt`）の**両方**を
    塞ぐ。`_predict` はモジュールレベルで `run_filler_prompt` を import しているため、定義元
    （`_adapters.intent`）だけを差し替えても段 (3) には効かず、実 API へ到達しうる
    （`model=` に実在モデル名を渡すテストがあるため従量課金にも倒れる）。定義元と参照元の
    双方を差し替え、**差し替えが実際に効いていること**を identity で照合してから返す。
    照合が無いと「呼ばれなかった」と「差し替えが的外れで素通しだった」を区別できない。

    wrapper は `async def` ではなく同期の `def` にし、`pytest.fail`（`BaseException` 派生）を
    送出する。await されない呼び出しでも失敗し、`except Exception` にも握り潰されない。
    """

    def _boom(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("the LLM adapter must not be called in the deterministic path")

    monkeypatch.setattr("oai_agentspec._adapters.intent.run_intent_prompt", _boom)
    monkeypatch.setattr("oai_agentspec._adapters.intent.run_filler_prompt", _boom)
    monkeypatch.setattr("oai_agentspec.runtime.intent._llm.run_intent_prompt", _boom)
    monkeypatch.setattr("oai_agentspec.runtime.intent._predict.run_filler_prompt", _boom)

    assert intent_adapter.run_intent_prompt is _boom
    assert intent_adapter.run_filler_prompt is _boom
    assert _llm.run_intent_prompt is _boom
    assert _predict.run_filler_prompt is _boom


# ---- 戻り値の形（§3.13） ----


async def test_plan_returns_action_plans_in_candidate_order() -> None:
    """既定では候補と同順・同数の `ActionPlan` の tuple を返す。"""
    catalog = _catalog(_spec("run_load_test"), _spec("open_dashboard"))
    prediction = IntentPrediction(
        candidates=(_intent("open_dashboard"), _intent("run_load_test"), _intent("open_dashboard"))
    )
    plans = await _planner(catalog, prediction).plan(IntentQuery(utterance="hi"))

    assert isinstance(plans, tuple)
    assert all(isinstance(p, ActionPlan) for p in plans)
    assert [p.action_id for p in plans] == ["open_dashboard", "run_load_test", "open_dashboard"]


async def test_plan_excludes_unregistered_candidates() -> None:
    """段 (1) の allowlist 除外を経た候補だけが計画になる（NFR-6）。"""
    catalog = _catalog(_spec("run_load_test"))
    prediction = IntentPrediction(
        candidates=(_intent("run_load_test"), _intent("delete_everything"))
    )
    plans = await _planner(catalog, prediction).plan(IntentQuery(utterance="hi"))

    assert [p.action_id for p in plans] == ["run_load_test"]


async def test_plan_returns_empty_when_all_candidates_are_excluded() -> None:
    """全候補が除外されても例外にせず空 tuple を返す。"""
    catalog = _catalog(_spec("run_load_test"))
    prediction = IntentPrediction(candidates=(_intent("delete_everything"),))
    plans = await _planner(catalog, prediction).plan(IntentQuery(utterance="hi"))

    assert plans == ()


async def test_plan_detail_returns_plan_result() -> None:
    """`detail=True` は `PlanResult(plans, suggestion, usage)` を返す。"""
    catalog = _catalog(_spec("run_load_test"))
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    result = await _planner(catalog, prediction).plan(IntentQuery(utterance="hi"), detail=True)

    assert type(result) is PlanResult
    assert isinstance(result.suggestion, ExecutableSuggestion)
    assert type(result.usage) is ParamUsage
    assert [p.action_id for p in result.plans] == ["run_load_test"]


async def test_plan_detail_preserves_report_and_metadata() -> None:
    """`detail=True` は候補生成器の `report` / `metadata` を捨てない（FR-4）。"""
    report = ConsistencyReport(conflicts=("c1",))
    prediction = IntentPrediction(
        candidates=(_intent("run_load_test"),), report=report, metadata={"generator": "rule-v3"}
    )
    result = await _planner(_catalog(_spec()), prediction).plan(
        IntentQuery(utterance="hi"), detail=True
    )

    assert result.suggestion.report == report
    assert result.suggestion.metadata == {"generator": "rule-v3"}


async def test_plan_without_detail_returns_only_plans() -> None:
    """`detail=False`（既定）では `PlanResult` を返さない。"""
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    plans = await _planner(_catalog(_spec()), prediction).plan(IntentQuery(utterance="hi"))

    assert not isinstance(plans, PlanResult)


# ---- 決定的段（FR-5 / NFR-5） ----


async def test_plan_resolves_slots_deterministically() -> None:
    """候補の値と `from_context` が段 (2) で確定する。"""
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, default=30),
                param("target", str, from_context="host"),
            )
        )
    )
    prediction = IntentPrediction(candidates=(_intent("run_load_test", seconds=60),))
    query = IntentQuery(utterance="hi", run_context=_RunContext())
    plans = await _planner(catalog, prediction).plan(query)

    by_name = {slot.name: slot for slot in plans[0].slots}
    assert by_name["seconds"].value == 60
    assert by_name["seconds"].origin is Origin.CANDIDATE
    assert by_name["target"].value == "api.example.com"
    assert by_name["target"].origin is Origin.RUN_CONTEXT


async def test_plan_is_deterministic_across_calls() -> None:
    """同一の候補列・同一の `run_context` に対し常に同一結果を返す（FR-5）。"""
    catalog = _catalog(_spec())
    prediction = IntentPrediction(candidates=(_intent("run_load_test", seconds=60),))
    planner = _planner(catalog, prediction)
    query = IntentQuery(utterance="hi")

    assert await planner.plan(query) == await planner.plan(query)


async def test_plan_does_not_call_the_llm_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """`llm_filler` 未結線なら `predict=True`（既定）でも LLM 実行アダプタを 1 度も呼ばない。

    第 1 段では「`llm_filler` を結線しても段 (3) の中身が無いので呼ばれない」を固定していたが、
    段 (3) が入った時点でその前提は消えた（結線してあれば実際に呼ばれるのが正しい挙動であり、
    塞がないまま残すと実 API・従量課金へ到達する）。段 (3) 導入後も成立する不変条件は
    「穴埋め経路そのものが無い場合は入口へ到達しない」（設計 §3.4a の規則 3）であるため、
    `NEEDS_LLM` のスロットを持たせたまま結線だけを外して構造的に観測する。結線してある場合の
    委譲は段 (3) の節（`test_plan_delegates_to_predict_params_once_with_the_bound_wiring` 他）
    が受け持つ。
    """
    _no_llm(monkeypatch)
    catalog = _catalog(_spec(parameters=(param("note", str, by_llm=True),)))
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    planner = _planner(catalog, prediction, llm_filler=None)

    plans = await planner.plan(IntentQuery(utterance="hi"))

    assert plans[0].slots[0].state is SlotState.NEEDS_LLM


async def test_plan_predict_false_does_not_call_the_llm_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`predict=False` は段 (3) を実行しない（FR-4 の保証を引数で取り出せる）。"""
    _no_llm(monkeypatch)
    catalog = _catalog(_spec(parameters=(param("note", str, by_llm=True),)))
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    planner = _planner(catalog, prediction, llm_filler=LLMFiller(model="gpt-x"))

    plans = await planner.plan(IntentQuery(utterance="hi"), predict=False)

    assert [p.action_id for p in plans] == ["run_load_test"]


async def test_plan_predict_false_with_detail_reports_zero_usage() -> None:
    """`predict=False` の `usage` は 0 件を表す値になる（FR-6）。"""
    catalog = _catalog(_spec(parameters=(param("note", str, by_llm=True),)))
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    planner = _planner(catalog, prediction, llm_filler=LLMFiller(model="gpt-x"))

    result = await planner.plan(IntentQuery(utterance="hi"), predict=False, detail=True)

    assert result.usage == ParamUsage(
        runs=0, model_calls=0, candidates=0, input_tokens=None, output_tokens=None
    )


# ---- 必須結線が欠けた場合（§3.4a の 3 規則） ----


async def test_plan_without_candidates_raises_runtime_error() -> None:
    """`candidates` 未結線の `plan()` は `RuntimeError`（規則 1・実測 30-11）。"""
    planner = _catalog(_spec()).bind(registry=_CountingAgentRegistry("load_test_agent"))
    with pytest.raises(RuntimeError) as excinfo:
        await planner.plan(IntentQuery(utterance="hi"))

    assert type(excinfo.value) is RuntimeError


async def test_plan_without_llm_filler_skips_stage_three() -> None:
    """`llm_filler` 未結線でも例外にせず `runs=0` を返す（規則 3・第 1 段の単体リリース）。"""
    catalog = _catalog(_spec(parameters=(param("note", str, by_llm=True),)))
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    planner = _planner(catalog, prediction, llm_filler=None)

    result = await planner.plan(IntentQuery(utterance="hi"), detail=True)

    assert result.usage.runs == 0
    assert result.usage.model_calls == 0
    assert result.plans[0].slots[0].state is SlotState.NEEDS_LLM
    assert result.plans[0].ready is False


# `test_plan_with_llm_filler_still_skips_stage_three_in_phase_one` はここにあったが削除した。
# 「`llm_filler` を結線しても段 (3) の中身が無いので `runs=0`」という**第 1 段限定**の契約を
# 固定するものであり、段 (3) が入った時点で偽になる。後継は段 (3) の節にある
# `test_plan_delegates_to_predict_params_once_with_the_bound_wiring` /
# `test_plan_returns_the_plans_from_predict_params` /
# `test_plan_detail_reports_the_usage_from_predict_params` の 3 件である。


async def test_zero_needs_llm_slots_report_zero_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """`NEEDS_LLM` が 1 件も無ければ `runs=0`（FR-6）。

    段 (3) は予測対象が 0 件なら実行器を駆動せず入力の plans をそのまま返す。その短絡が
    将来壊れても実 API・従量課金へ倒れないよう、LLM 実行アダプタを構造的に塞いだうえで
    観測する（`llm_filler` に実在しうるモデル名を渡しているため）。
    """
    _no_llm(monkeypatch)
    prediction = IntentPrediction(candidates=(_intent("run_load_test", seconds=60),))
    planner = _planner(_catalog(_spec()), prediction, llm_filler=LLMFiller(model="gpt-x"))

    result = await planner.plan(IntentQuery(utterance="hi"), detail=True)

    assert result.usage.runs == 0
    assert result.plans[0].ready is True


# ---- 初回 plan() の validate()（§3.13 段 (0)・実測 30-4） ----


async def test_first_plan_runs_validate_once() -> None:
    """初回 `plan()` で `validate()` が走り、2 回目では再実行されない。"""
    catalog = _catalog(_spec())
    registry = _CountingAgentRegistry("load_test_agent")
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    planner = _planner(catalog, prediction, registry=registry)
    query = IntentQuery(utterance="hi")

    await planner.plan(query)
    after_first = registry.name_calls
    await planner.plan(query)

    assert after_first > 0
    assert registry.name_calls == after_first


async def test_explicit_validate_is_not_repeated_by_plan() -> None:
    """`validate()` 済みなら `plan()` は再検証しない。"""
    catalog = _catalog(_spec())
    registry = _CountingAgentRegistry("load_test_agent")
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    planner = _planner(catalog, prediction, registry=registry)

    planner.validate()
    after_validate = registry.name_calls
    await planner.plan(IntentQuery(utterance="hi"))

    assert after_validate > 0
    assert registry.name_calls == after_validate


async def test_plan_propagates_validation_failure() -> None:
    """検証に落ちる宣言では `plan()` が段 (1) へ進む前に落ちる。"""
    catalog = _catalog(_spec(action_agent="ghost_agent"))
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    generator = _RecordingGenerator(prediction)
    planner = catalog.bind(
        registry=_CountingAgentRegistry("load_test_agent"),
        candidates=CandidateSource(generator=generator),
    )
    with pytest.raises(KeyError):
        await planner.plan(IntentQuery(utterance="hi"))

    assert generator.contexts == []


async def test_plan_does_not_change_planner_equality() -> None:
    """`plan()` の実行済みフラグは等価性に漏れない（`_DeclaredFieldsEq`）。"""
    catalog = _catalog(_spec())
    registry = _CountingAgentRegistry("load_test_agent")
    source = CandidateSource(
        generator=_RecordingGenerator(IntentPrediction(candidates=(_intent("run_load_test"),)))
    )
    used = catalog.bind(registry=registry, candidates=source)
    untouched = catalog.bind(registry=registry, candidates=source)

    await used.plan(IntentQuery(utterance="hi"))

    assert used == untouched


# ---- 段 (1) との結線 ----


async def test_plan_calls_the_generator_once_per_call() -> None:
    """`plan()` 1 回につき `generate()` は 1 回だけ（NFR-5）。"""
    catalog = _catalog(_spec())
    generator = _RecordingGenerator(IntentPrediction(candidates=(_intent("run_load_test"),)))
    planner = catalog.bind(
        registry=_CountingAgentRegistry("load_test_agent"),
        candidates=CandidateSource(generator=generator),
    )
    query = IntentQuery(utterance="hi")

    await planner.plan(query)
    await planner.plan(query)

    assert len(generator.contexts) == 2


async def test_plan_passes_the_query_through_the_context_builder() -> None:
    """段 (1) は `query` から `IntentContext` を組み `run_context` を運ぶ。"""
    catalog = _catalog(_spec())
    generator = _RecordingGenerator(IntentPrediction(candidates=(_intent("run_load_test"),)))
    planner = catalog.bind(
        registry=_CountingAgentRegistry("load_test_agent"),
        candidates=CandidateSource(generator=generator),
    )
    run_context = _RunContext()

    await planner.plan(IntentQuery(utterance="発話", run_context=run_context))

    assert generator.contexts[0].utterance == "発話"
    assert generator.contexts[0].run_context is run_context


async def test_validation_precedes_the_candidates_wiring_check() -> None:
    """段 (0) の `validate()` は `candidates` 未結線の `RuntimeError` より先に走る（§3.13）。

    宣言が不正かつ `candidates` 未結線という二重の欠陥では、順序を入れ替えても
    どちらかの例外で落ちるため外からは同じに見える。設計が定めた順序
    「(0) 未検証なら validate -> candidates 未結線なら RuntimeError -> (1) 候補生成」
    を、送出される例外の型で固定する。
    """
    catalog = _catalog(_spec(action_agent="ghost_agent"))
    planner = catalog.bind(registry=_CountingAgentRegistry("load_test_agent"))
    with pytest.raises(KeyError, match="ghost_agent") as excinfo:
        await planner.plan(IntentQuery(utterance="hi"))

    assert type(excinfo.value) is KeyError


async def test_plan_uses_the_bound_snapshot_not_later_registrations() -> None:
    """bind 後に `register()` したアクションは既存 planner の allowlist に入らない。"""
    catalog = _catalog(_spec("run_load_test"))
    prediction = IntentPrediction(
        candidates=(_intent("run_load_test"), _intent("added_later")),
    )
    planner = _planner(catalog, prediction)

    catalog.register(_spec("added_later"))
    plans = await planner.plan(IntentQuery(utterance="hi"))

    assert [p.action_id for p in plans] == ["run_load_test"]


# ---- model_copy(update=...) が派生状態を引き継がない（セキュリティレビュー指摘 #88-W3） ----
#
# `_catalog` は `model_post_init` でのみ組まれ `model_copy` では再構築されず、`_validated`
# もそのまま引き継がれる。結果、宣言フィールド（`specs` 等）は新しいのに `plan()` が実際に
# 参照する宣言簿と検証状態が古い planner を作れる（コピーで除外したはずのアクションが
# allowlist に残り、差し替えた結線に対する起動時検証も走らない）。
#
# 採る方針は `model_copy` の禁止ではなく、派生状態を宣言フィールドから常に導出し直すこと
# （`ActionSpec.parameters_model()` の鍵付きキャッシュと同じ規則）。したがって以下の 4 件は
# 「コピーが新しい宣言で動くこと」「コピーが未検証扱いへ戻ること（宣言フィールド / 結線の
# どちらを差し替えても）」「何も変えないコピーは既存の等価性 pin と矛盾しないこと」を pin する。


def _validated_planner(
    catalog: ActionCatalog,
    prediction: IntentPrediction,
    *,
    registry: _CountingAgentRegistry | None = None,
) -> Any:
    """検証済み（`_validated=True`）の状態にした `ActionPlanner` を返す。

    派生状態の引き継ぎを観測するには、コピー元が「宣言簿を組み終え検証も済ませた」状態で
    ある必要がある。`validate()` を明示的に呼んで到達させる。
    """
    planner = _planner(catalog, prediction, registry=registry)
    planner.validate()
    return planner


async def test_model_copy_update_specs_uses_the_new_allowlist() -> None:
    """`update={"specs": ...}` したコピーは新しい specs の allowlist で動く（指摘 #88-W3）。

    コピー元の宣言簿が残ると、コピーで除外したはずの `run_load_test` が候補として通り、
    差し替えた `open_dashboard` が除外される。
    """
    catalog = _catalog(_spec("run_load_test"))
    prediction = IntentPrediction(candidates=(_intent("run_load_test"), _intent("open_dashboard")))
    planner = _validated_planner(catalog, prediction)

    copied = planner.model_copy(update={"specs": (_spec("open_dashboard"),)})
    plans = await copied.plan(IntentQuery(utterance="hi"))

    assert [p.action_id for p in plans] == ["open_dashboard"]


async def test_model_copy_update_specs_is_treated_as_unvalidated() -> None:
    """宣言フィールドを差し替えたコピーの `plan()` は起動時検証を走らせる（指摘 #88-W3）。

    不正な結線（未登録の `action_agent`）へ差し替えたコピーが検証由来の `KeyError` で落ちる
    ことで、`_validated` が引き継がれていないことを観測する。
    """
    catalog = _catalog(_spec("run_load_test"))
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    planner = _validated_planner(catalog, prediction)

    copied = planner.model_copy(
        update={"specs": (_spec("run_load_test", action_agent="ghost_agent"),)}
    )
    with pytest.raises(KeyError, match="ghost_agent") as excinfo:
        await copied.plan(IntentQuery(utterance="hi"))

    assert type(excinfo.value) is KeyError


async def test_model_copy_update_wiring_only_is_treated_as_unvalidated() -> None:
    """結線だけを差し替えたコピーも未検証扱いになる（指摘 #88-W3）。

    `registry` は宣言フィールドではないが、起動時検証（検査 1）の判定材料である。差し替えを
    未検証扱いにしないと、宣言簿が解決できない登録簿へ向いたまま `plan()` が通る。
    """
    catalog = _catalog(_spec("run_load_test"))
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    planner = _validated_planner(catalog, prediction)

    copied = planner.model_copy(update={"registry": _CountingAgentRegistry("other_agent")})
    with pytest.raises(KeyError, match="load_test_agent") as excinfo:
        await copied.plan(IntentQuery(utterance="hi"))

    assert type(excinfo.value) is KeyError


async def test_model_copy_without_update_keeps_equality_and_behaviour() -> None:
    """何も変えないコピーは宣言フィールドが同一であり挙動も変わらない（指摘 #88-W3）。

    派生状態を宣言フィールドから導出し直す方針が、既存の等価性 pin
    （`test_plan_does_not_change_planner_equality`）と矛盾しないことを固定する。再検証が
    走ってもよいが、結果は同じ allowlist・同じ計画でなければならない。
    """
    catalog = _catalog(_spec("run_load_test"))
    prediction = IntentPrediction(candidates=(_intent("run_load_test"), _intent("open_dashboard")))
    planner = _validated_planner(catalog, prediction)

    copied = planner.model_copy()
    plans = await copied.plan(IntentQuery(utterance="hi"))

    assert copied == planner
    assert [p.action_id for p in plans] == ["run_load_test"]


# ---- 段 (3) の中身（設計 §5 タスク 2-7 / §3.13） ----
#
# 第 2 段で段 (3) の中身（`_predict._predict_params` への委譲）が入る。ここでは委譲そのもの
# （呼ばれる回数・渡る引数一式・戻り値の反映・例外の伝播・段 (2) との順序）を pin する。
# `_predict_params` の内側の挙動は L1（`test_predict_l1.py`）が持ち、本ファイルは
# `ActionPlanner.plan()` から見た結線だけを対象にする。
#
# 観測は `_predict._predict_params` を spy へ差し替えて行う。spy は**同期の `def`**で
# coroutine を返すだけであり、カウンタは呼び出しの同期部で進む（`async def` の内側に置くと
# await されない二重呼び出しを取り逃す）。


class _Prompts:
    """`prompts` 結線の目印。セグメント宣言が無いため起動時検証からは触られない。"""


class _Guardrails:
    """`guardrail_registry` 結線の目印。ガードレール宣言が無いため触られない。"""


class _PredictBoom(Exception):
    """段 (3) が送出する例外。`plan()` が握り潰さないことの観測に使う。"""


def _predict_reference(
    plans: tuple[ActionPlan, ...],
    context: IntentContext[Any],
    *,
    llm_filler: Any,
    prompts: Any = None,
    guardrail_registry: Any = None,
) -> None:
    """`_predict._predict_params` の確定済みシグネチャ（引数の照合基準）。

    記録した `(args, kwargs)` をこのシグネチャで束ね、位置渡し / キーワード渡しの違いを
    正規化してから期待値テーブルと `==` で照合する。実装モジュールの `signature` を使うと
    実装が未完のときに束ね自体が失敗し、何が違うのかが読めなくなる。
    """


_PREDICT_SIGNATURE = inspect.signature(_predict_reference)

#: spy が返す `ParamUsage`。定数 `ParamUsage(runs=0, ...)` と全フィールドで異なる値にして、
#: 段 (3) の戻りを捨てて定数を返す実装を検知できるようにする。
_SPY_USAGE = ParamUsage(runs=1, model_calls=2, candidates=3, input_tokens=11, output_tokens=7)


@dataclass
class _PredictSpy:
    """`_predict._predict_params` の spy。受け取った plans を反転して返す。

    反転を返すのは、段 (3) の戻りを反映せず段 (2) の plans をそのまま返す実装と区別する
    ためである（同じ集合・違う順序であり、件数を数えるだけの検査では通ってしまう）。
    """

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    error: Exception | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """呼び出しを同期部で記録し、結果を返す coroutine を返す。"""
        self.calls.append((args, dict(kwargs)))
        bound = _PREDICT_SIGNATURE.bind(*args, **kwargs)
        return self._answer(tuple(bound.arguments["plans"]))

    async def _answer(self, plans: tuple[ActionPlan, ...]) -> tuple[tuple[ActionPlan, ...], Any]:
        """反転した plans と目印の `ParamUsage` を返す（`error` があれば送出する）。"""
        if self.error is not None:
            raise self.error
        return tuple(reversed(plans)), _SPY_USAGE

    @property
    def once(self) -> dict[str, Any]:
        """ちょうど 1 回呼ばれたことを確かめ、既定を埋めた引数一式を返す。"""
        assert len(self.calls) == 1
        args, kwargs = self.calls[0]
        bound = _PREDICT_SIGNATURE.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)


def _spy_predict(monkeypatch: pytest.MonkeyPatch, *, error: Exception | None = None) -> _PredictSpy:
    """`_predict._predict_params` を spy へ差し替える（定義元を差し替える）。"""
    spy = _PredictSpy(error=error)
    monkeypatch.setattr("oai_agentspec.runtime.intent._predict._predict_params", spy)
    return spy


def _wired_planner(
    catalog: ActionCatalog,
    prediction: IntentPrediction,
    *,
    llm_filler: LLMFiller | None,
    prompts: Any = None,
    guardrail_registry: Any = None,
) -> Any:
    """解決簿 2 件まで結線した `ActionPlanner` を返す（段 (3) への受け渡しの観測用）。"""
    return catalog.bind(
        registry=_CountingAgentRegistry("load_test_agent", "dashboard_agent"),
        prompts=prompts,
        guardrail_registry=guardrail_registry,
        candidates=CandidateSource(generator=_RecordingGenerator(prediction)),
        llm_filler=llm_filler,
    )


def _llm_spec(action_id: str = "run_load_test", *, action_agent: str = "load_test_agent") -> Any:
    """`by_llm` の空欄（`NEEDS_LLM`）と `from_context` を 1 件ずつ持つ宣言を返す。"""
    return _spec(
        action_id,
        action_agent=action_agent,
        parameters=(
            param("note", str, by_llm=True),
            param("target", str, from_context="host"),
        ),
    )


async def test_plan_delegates_to_predict_params_once_with_the_bound_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """段 (3) は `_predict_params` を 1 回だけ呼び、bind で受けた結線をそのまま渡す。

    引数は期待値テーブルと `==` で全体照合する。個別フィールドだけを見ると、結線値を捨てて
    別の値（`None` や新規に組んだオブジェクト）で呼ぶ変異を取り逃す。
    """
    spy = _spy_predict(monkeypatch)
    catalog = _catalog(_llm_spec())
    prediction = IntentPrediction(
        candidates=(_intent("run_load_test"), _intent("delete_everything"))
    )
    filler = LLMFiller(model="gpt-x")
    prompts = _Prompts()
    guardrails = _Guardrails()
    planner = _wired_planner(
        catalog, prediction, llm_filler=filler, prompts=prompts, guardrail_registry=guardrails
    )
    query = IntentQuery(utterance="hi", run_context=_RunContext())

    await planner.plan(query)

    context = planner.candidates.generator.contexts[0]
    assert spy.once == {
        "plans": _plan_slots((_intent("run_load_test"),), catalog, context),
        "context": context,
        "llm_filler": filler,
        "prompts": prompts,
        "guardrail_registry": guardrails,
    }


async def test_plan_passes_the_context_object_used_by_the_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """段 (3) が受け取る `context` は段 (1) が候補生成器へ渡したものと等価である。

    `run_context` は**同一オブジェクト**でなければならない（利用者のアプリ状態であり、
    複製すると宣言した `from_context` / `prompt_vars` のパスが別の実体を指す）。`context`
    そのものの identity は要求しない。`ExecutableSuggestion` のフィールドとして持ち回る
    過程で pydantic が等価な別インスタンスへ再構築するためである（実測）。
    """
    spy = _spy_predict(monkeypatch)
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    planner = _wired_planner(_catalog(_llm_spec()), prediction, llm_filler=LLMFiller(model="gpt-x"))
    run_context = _RunContext()

    await planner.plan(IntentQuery(utterance="発話", run_context=run_context))

    assert spy.once["context"] == planner.candidates.generator.contexts[0]
    assert spy.once["context"].run_context is run_context
    assert spy.once["context"].utterance == "発話"


async def test_plan_receives_the_stage_two_plans_not_the_raw_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """段 (3) は段 (2) の**後**に走り、確定済みスロットを載せた plans を受け取る（順序）。

    `from_context` 由来の値が既に載っていることで、段 (2) を経ていることを観測する。
    """
    spy = _spy_predict(monkeypatch)
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    planner = _wired_planner(_catalog(_llm_spec()), prediction, llm_filler=LLMFiller(model="gpt-x"))

    await planner.plan(IntentQuery(utterance="hi", run_context=_RunContext()))

    received = spy.once["plans"]
    by_name = {slot.name: slot for slot in received[0].slots}
    assert [p.action_id for p in received] == ["run_load_test"]
    assert by_name["target"].value == "api.example.com"
    assert by_name["target"].origin is Origin.RUN_CONTEXT
    assert by_name["note"].state is SlotState.NEEDS_LLM


async def test_plan_returns_the_plans_from_predict_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`plan()` の戻りは段 (3) が返した plans である（段 (2) の plans ではない）。"""
    _spy_predict(monkeypatch)
    catalog = _catalog(_llm_spec(), _llm_spec("open_dashboard", action_agent="dashboard_agent"))
    prediction = IntentPrediction(candidates=(_intent("run_load_test"), _intent("open_dashboard")))
    planner = _wired_planner(catalog, prediction, llm_filler=LLMFiller(model="gpt-x"))

    plans = await planner.plan(IntentQuery(utterance="hi", run_context=_RunContext()))

    assert [p.action_id for p in plans] == ["open_dashboard", "run_load_test"]


async def test_plan_detail_reports_the_usage_from_predict_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`detail=True` の `usage` は段 (3) が返した値であり、0 件の定数ではない（FR-6）。"""
    _spy_predict(monkeypatch)
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    planner = _wired_planner(_catalog(_llm_spec()), prediction, llm_filler=LLMFiller(model="gpt-x"))

    result = await planner.plan(IntentQuery(utterance="hi", run_context=_RunContext()), detail=True)

    assert result.usage == _SPY_USAGE
    assert [p.action_id for p in result.plans] == ["run_load_test"]


async def test_plan_predict_false_does_not_reach_predict_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`predict=False` は段 (3) を 1 度も呼ばず、段 (2) の plans と 0 件の usage を返す。

    第 1 段では段 (3) の中身が無く `predict` フラグの効力が観測できなかった。段 (3) が
    入って初めて「フラグを無視して常に予測する」変異を検知できる。
    """
    spy = _spy_predict(monkeypatch)
    catalog = _catalog(_llm_spec())
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    planner = _wired_planner(catalog, prediction, llm_filler=LLMFiller(model="gpt-x"))
    query = IntentQuery(utterance="hi", run_context=_RunContext())

    result = await planner.plan(query, predict=False, detail=True)

    context = planner.candidates.generator.contexts[0]
    assert spy.calls == []
    assert result.plans == _plan_slots((_intent("run_load_test"),), catalog, context)
    assert result.usage == ParamUsage(
        runs=0, model_calls=0, candidates=0, input_tokens=None, output_tokens=None
    )


async def test_plan_without_llm_filler_does_not_reach_predict_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`llm_filler` 未結線なら `predict=True`（既定）でも段 (3) へ入らない（規則 3）。"""
    spy = _spy_predict(monkeypatch)
    catalog = _catalog(_llm_spec())
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    planner = _wired_planner(catalog, prediction, llm_filler=None)
    query = IntentQuery(utterance="hi", run_context=_RunContext())

    result = await planner.plan(query, detail=True)

    context = planner.candidates.generator.contexts[0]
    assert spy.calls == []
    assert result.plans == _plan_slots((_intent("run_load_test"),), catalog, context)
    assert result.usage == ParamUsage(
        runs=0, model_calls=0, candidates=0, input_tokens=None, output_tokens=None
    )


async def test_plan_calls_predict_params_once_per_plan_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`plan()` 1 回につき段 (3) は 1 回だけ走る（候補件数に比例しない・ADR 0026）。"""
    spy = _spy_predict(monkeypatch)
    catalog = _catalog(_llm_spec(), _llm_spec("open_dashboard", action_agent="dashboard_agent"))
    prediction = IntentPrediction(candidates=(_intent("run_load_test"), _intent("open_dashboard")))
    planner = _wired_planner(catalog, prediction, llm_filler=LLMFiller(model="gpt-x"))
    query = IntentQuery(utterance="hi", run_context=_RunContext())

    await planner.plan(query)
    await planner.plan(query)

    assert len(spy.calls) == 2


async def test_plan_propagates_the_error_from_predict_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """段 (3) が送出した例外は `plan()` が握り潰さずそのまま伝播する。"""
    _spy_predict(monkeypatch, error=_PredictBoom("predict stage exploded"))
    prediction = IntentPrediction(candidates=(_intent("run_load_test"),))
    planner = _wired_planner(_catalog(_llm_spec()), prediction, llm_filler=LLMFiller(model="gpt-x"))

    with pytest.raises(_PredictBoom, match="predict stage exploded") as excinfo:
        await planner.plan(IntentQuery(utterance="hi", run_context=_RunContext()))

    assert type(excinfo.value) is _PredictBoom
