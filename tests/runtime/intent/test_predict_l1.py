"""L1: `runtime.intent._predict` の予測委譲（設計 §5 タスク 2-2〜2-6 / §3.13 / ADR 0026）。

FR-6 / FR-7 の受け入れ基準と NFR-5 / NFR-6 の計測基準を pin する。対象は次の 5 つである。

- 2-2 プロンプト合成（セグメント重複排除 / `resolved_prompt_vars` のパス解決 /
  `history_items` の会話部分化 / 出力形式スキーマの提示）。`catalog` は受け取らない。
- 2-3 複合応答モデル（`candidate_<index>` -> parse 派生）と対応表。
- 2-4 予測委譲の本体（`Runner.run` を 1 回 / 0 回・コードフェンス剥がし・`confirm` による
  遷移・`ConfidenceLevel` 降順 stable sort + 切り捨て・`origin=Origin.LLM` / `detail=None`）。
- 2-5 `ParamUsage` の詰め替え（`AgentRunUsage` -> `runs` / `model_calls` / `candidates` /
  tokens）。
- 2-6 後退挙動（`default` への後退 / `NEEDS_USER` への遷移・他スロットの保持）。

観測の取り方（実装の内部関数名に依存しないための取り決め）:

- `Runner.run` の**呼び出し回数**は `_adapters.intent.run_filler_prompt` を包む spy で数える。
  spy の wrapper は `async def` ではなく**同期の `def`**（coroutine を返すだけ）であり、
  カウンタは呼び出しの同期部で進む。`async def` の内側に置くと、await されない二重呼び出し
  （候補ごとに 1 回呼ぶ実装）を取り逃す。
- system 指示部は `FakeModel.calls[0].system_instructions`（実際に送られた値）で観測する。
  `Agent.instructions` は callable でもありうるため、送信結果の側を見る。
- user content と `history_items` と装着済みガードレールは spy が捕らえた引数・実体で観測する。

実 API は呼ばない（`FakeModel` + 実 `Runner`）。予測エージェントは lib が専用
`AgentRegistry` で構築するため、業務 registry は `_predict` へ渡らない。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from agents import InputGuardrailTripwireTriggered
from pydantic import ValidationError

from oai_agentspec._adapters import intent as intent_adapter
from oai_agentspec._adapters.guardrails import build_input_guardrail, build_output_guardrail
from oai_agentspec._adapters.intent import AgentRunUsage
from oai_agentspec.prompts import PromptLayout, PromptStore
from oai_agentspec.runtime.guardrails._detectors import Detection
from oai_agentspec.runtime.guardrails.registry import GuardrailRegistry
from oai_agentspec.runtime.guardrails.types import Boundary, GuardrailSpec
from oai_agentspec.runtime.intent import _predict
from oai_agentspec.runtime.intent.actions import ActionCatalog, ActionSpec, param
from oai_agentspec.runtime.intent.binding import LLMFiller
from oai_agentspec.runtime.intent.slots import (
    ActionPlan,
    Origin,
    ParamUsage,
    SlotState,
    _plan_slots,
)
from oai_agentspec.runtime.intent.types import (
    ConfidenceLevel,
    ExecutableIntent,
    IntentContext,
)

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.unit

# 直接 `from ..._predict import ...` と書かず、パッケージ属性から取り出す。ruff の isort は
# 実在しないモジュールへの直接 import をサードパーティと分類するため、実装が入る前後で
# import ブロックの並びが変わってしまう（実装後に I001 が出る）。
_predict_params = _predict._predict_params
FILLER_AGENT_NAME = _predict.FILLER_AGENT_NAME


_LAYOUT = PromptLayout(base="base", parts="parts", agents="agents")
#: 宣言には現れない発話（`context.utterance` 専用の目印）。既定では会話部分にも現れず、
#: system 指示部へ混入していないかの目印に使う。
_UTTERANCE = "MARKER_UTTERANCE_9f3a"
#: 既定の会話部分の本文。`_UTTERANCE` と別マーカーにして「発話の漏れ」と「会話の漏れ」を
#: 独立に観測できるようにする（同一マーカーだと 2 経路を区別できない）。
_DEFAULT_HISTORY_BODY = "MARKER_DEFAULT_HISTORY_BODY_5e7d"
#: `_context()` の既定の会話部分。予測を駆動するテストは会話がある経路を通す（空 history で
#: 予測を駆動すると暗黙失敗の検知として `RuntimeError` になるため）。
_DEFAULT_HISTORY: tuple[Mapping[str, Any], ...] = (
    {"role": "user", "content": _DEFAULT_HISTORY_BODY},
)


# ---- Fake / spy ----


@dataclass
class _FillerCall:
    """`run_filler_prompt` 1 回ぶんの引数の記録。"""

    agent: Any
    history_items: tuple[Mapping[str, Any], ...]
    user_content: str
    context: Any


@dataclass
class _FillerSpy:
    """`run_filler_prompt` の呼び出し記録。カウントは同期部で進む。"""

    calls: list[_FillerCall] = field(default_factory=list)

    @property
    def once(self) -> _FillerCall:
        """ちょうど 1 回呼ばれたことを確かめたうえで記録を返す。"""
        assert len(self.calls) == 1
        return self.calls[0]


def _spy_filler(monkeypatch: pytest.MonkeyPatch) -> _FillerSpy:
    """`run_filler_prompt` を「記録してから本物へ委譲する」同期 wrapper へ差し替える。

    差し替えは定義元（`_adapters.intent`）と参照元（`_predict`）の双方へ行う。参照元が
    モジュールレベル import でも関数内遅延 import でも spy が効くようにするためである。
    """
    original = intent_adapter.run_filler_prompt
    spy = _FillerSpy()

    def _wrapper(
        agent: Any,
        history_items: tuple[Mapping[str, Any], ...],
        user_content: str,
        *,
        context: Any = None,
    ) -> Any:
        """記録してから本物のコルーチンを返す（同期関数なので記録は呼び出し時に進む）。"""
        spy.calls.append(
            _FillerCall(
                agent=agent,
                history_items=history_items,
                user_content=user_content,
                context=context,
            )
        )
        return original(agent, history_items, user_content, context=context)

    monkeypatch.setattr(intent_adapter, "run_filler_prompt", _wrapper)
    if hasattr(_predict, "run_filler_prompt"):
        monkeypatch.setattr(_predict, "run_filler_prompt", _wrapper)
    return spy


def _stub_filler(monkeypatch: pytest.MonkeyPatch, text: str, usage: AgentRunUsage) -> None:
    """`run_filler_prompt` を固定の応答と利用量を返す stub へ差し替える。

    `ParamUsage` への詰め替えだけを対象にしたいテスト向け。`max_turns=1` の実行では
    `model_calls` を 1 以外にできず、`runs` との取り違えを検知できないため stub を使う。
    """

    async def _stub(*args: Any, **kwargs: Any) -> tuple[str, AgentRunUsage]:
        """固定の応答テキストと利用量を返す。"""
        return text, usage

    monkeypatch.setattr(intent_adapter, "run_filler_prompt", _stub)
    if hasattr(_predict, "run_filler_prompt"):
        monkeypatch.setattr(_predict, "run_filler_prompt", _stub)


def _forbid_filler(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_filler_prompt` を「呼ばれたら失敗する」同期 wrapper へ差し替える。

    同期 `def` にするのは、await されない呼び出し（coroutine を作っただけの経路）も
    捕らえるためである。`async def` の内側で失敗させると呼び出しを取り逃す。
    """

    def _never(*args: Any, **kwargs: Any) -> Any:
        """呼ばれた時点で失敗させる（従量課金の発生前に落ちることの保証）。"""
        pytest.fail("run_filler_prompt must not be called")

    monkeypatch.setattr(intent_adapter, "run_filler_prompt", _never)
    if hasattr(_predict, "run_filler_prompt"):
        monkeypatch.setattr(_predict, "run_filler_prompt", _never)


def _detect_clean(text: str) -> Detection:
    """常に非検知を返す plain 検知器。"""
    return Detection(triggered=False)


def _detect_trip(text: str) -> Detection:
    """常に検知する plain 検知器（tripwire を立てる）。"""
    return Detection(triggered=True, reason="dangerous")


class _Tenant:
    """`prompt_vars` のパス解決先（入れ子）。"""

    id = "t-001"


class _Ctx:
    """`run_context` の代表インスタンス。"""

    def __init__(self) -> None:
        self.tenant = _Tenant()


# ---- ヘルパ ----


def _store(tmp_path: Path, **bodies: str) -> PromptStore:
    """`parts/<name>.md` を書き出した実物の `PromptStore` を返す。"""
    parts = tmp_path / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for name, body in bodies.items():
        (parts / f"{name}.md").write_text(body, encoding="utf-8")
    return PromptStore(tmp_path, _LAYOUT)


def _spec(
    action_id: str = "run_load_test",
    *,
    action_agent: str = "load_test_agent",
    label: str = "負荷試験",
    parameters: tuple[Any, ...] | None = None,
    prompt: tuple[str, ...] = (),
    prompt_vars: Mapping[str, str] | None = None,
    on_invalid_slot: str | None = None,
) -> ActionSpec:
    """健全な `ActionSpec` を組む（`label` は既定でプレースホルダを持たない）。"""
    return ActionSpec(
        action_id=action_id,
        description="負荷試験を実行する",
        action_agent=action_agent,
        label=label,
        parameters=parameters if parameters is not None else (param("seconds", int, by_llm=True),),
        prompt=prompt,
        prompt_vars=dict(prompt_vars or {}),
        on_invalid_slot=on_invalid_slot,
    )


def _catalog(*specs: ActionSpec, **kwargs: Any) -> ActionCatalog:
    """宣言簿を組んで `specs` を登録した `ActionCatalog` を返す。"""
    catalog = ActionCatalog(**kwargs)
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


def _context(
    *,
    run_context: Any = None,
    history_items: tuple[Mapping[str, Any], ...] = _DEFAULT_HISTORY,
) -> IntentContext[Any]:
    """`IntentContext` を組む（`utterance` は system 混入検知用の目印）。

    `history_items` は既定で非空にする。予測は会話部分からしか文脈を得られないため、
    予測を駆動するテストは会話がある経路を通す必要がある。空 history を意図する場合は
    `history_items=()` を明示的に渡す。
    """
    return IntentContext(utterance=_UTTERANCE, history_items=history_items, run_context=run_context)


def _build_plans(
    catalog: ActionCatalog,
    candidates: tuple[ExecutableIntent, ...],
    context: IntentContext[Any],
) -> tuple[ActionPlan, ...]:
    """決定的段（段 (2)）を通して `ActionPlan` を組む。"""
    return _plan_slots(candidates, catalog, context)


def _suggestion(value: Any, level: str = "high", rationale: str | None = None) -> dict[str, Any]:
    """予測エージェントが返す `SlotSuggestion` 1 件ぶんの dict。"""
    return {"value": value, "level": level, "rationale": rationale}


def _response(*per_candidate: Mapping[str, Any]) -> str:
    """候補位置順の dict を `candidate_<index>` へ載せた応答 JSON を組む。"""
    return json.dumps(
        {f"candidate_{index}": body for index, body in enumerate(per_candidate)},
        ensure_ascii=False,
    )


def _filler(text: str, **kwargs: Any) -> tuple[LLMFiller, FakeModel]:
    """固定応答を返す `FakeModel` を載せた `LLMFiller` を返す。"""
    model = FakeModel().queue_text(text)
    return LLMFiller(model=model, **kwargs), model


def _slot(plan: ActionPlan, name: str) -> Any:
    """名前でスロットを 1 件取り出す。"""
    return next(slot for slot in plan.slots if slot.name == name)


# ======================================================================
# 2-2: プロンプト合成
# ======================================================================


async def test_duplicate_segment_body_appears_once_in_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同一セグメントを 2 候補が要求しても本文は 1 回だけ積まれる（NFR-5 / ADR 0026 (b)）。"""
    _spy_filler(monkeypatch)
    catalog = _catalog(
        _spec("run_load_test", prompt=("part:shared",)),
        _spec(
            "open_dashboard", prompt=("part:shared",), parameters=(param("q", str, by_llm=True),)
        ),
    )
    prompts = _store(tmp_path, shared="SHARED_SEGMENT_BODY")
    context = _context()
    plans = _build_plans(catalog, (_intent("run_load_test"), _intent("open_dashboard")), context)
    filler, model = _filler(_response({"seconds": _suggestion(60)}, {"q": _suggestion("x")}))

    await _predict_params(plans, context, llm_filler=filler, prompts=prompts)

    system = model.calls[0].system_instructions or ""
    assert system.count("SHARED_SEGMENT_BODY") == 1


async def test_param_segment_is_included_only_for_needs_llm_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`NEEDS_LLM` のパラメータの `param.prompt` だけが積まれる（FR-6 L167）。"""
    _spy_filler(monkeypatch)
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, by_llm=True, prompt="part:missing_hint"),
                param("region", str, by_llm=True, prompt="part:filled_hint"),
            )
        )
    )
    prompts = _store(tmp_path, missing_hint="MISSING_HINT", filled_hint="FILLED_HINT")
    context = _context()
    # region は候補が値を載せているため決定的段で確定し、予測対象ではない。
    plans = _build_plans(catalog, (_intent("run_load_test", region="jp"),), context)
    filler, model = _filler(_response({"seconds": _suggestion(60)}))

    await _predict_params(plans, context, llm_filler=filler, prompts=prompts)

    system = model.calls[0].system_instructions or ""
    assert "MISSING_HINT" in system
    assert "FILLED_HINT" not in system


async def test_prompt_vars_are_resolved_from_run_context_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resolved_prompt_vars` の各キーが `run_context` のパス解決値で置換される（FR-6 L168）。"""
    _spy_filler(monkeypatch)
    catalog = _catalog(
        _spec(prompt=("part:hint",), prompt_vars={"tenant": "tenant.id"}),
    )
    prompts = _store(tmp_path, hint="テナント ${tenant} 向けのヒント")
    context = _context(run_context=_Ctx())
    plans = _build_plans(catalog, (_intent("run_load_test"),), context)
    filler, model = _filler(_response({"seconds": _suggestion(60)}))

    await _predict_params(plans, context, llm_filler=filler, prompts=prompts)

    system = model.calls[0].system_instructions or ""
    assert "テナント t-001 向けのヒント" in system
    assert "${tenant}" not in system


async def test_prompt_var_resolving_to_none_becomes_an_empty_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """パス解決できなかった `prompt_vars` は空文字へ展開する（`"None"` を載せない）。"""
    _spy_filler(monkeypatch)
    catalog = _catalog(_spec(prompt=("part:hint",), prompt_vars={"tenant": "tenant.missing"}))
    prompts = _store(tmp_path, hint="テナント[${tenant}]のヒント")
    context = _context(run_context=_Ctx())
    plans = _build_plans(catalog, (_intent("run_load_test"),), context)
    filler, model = _filler(_response({"seconds": _suggestion(60)}))

    await _predict_params(plans, context, llm_filler=filler, prompts=prompts)

    system = model.calls[0].system_instructions or ""
    assert "テナント[]のヒント" in system
    assert "None" not in system


async def test_missing_prompt_store_with_declared_segments_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """セグメント宣言があるのに `prompts` が未結線なら `RuntimeError`（設計 §3.4a の規則 2）。

    起動時検証が同じ条件を落とすが、`_predict` を直接呼ぶ経路でも黙ってセグメントを捨てない
    ことを固定する（捨てると宣言したヒントが効かないまま従量課金だけが発生する）。
    """
    _spy_filler(monkeypatch)
    catalog = _catalog(_spec(prompt=("part:hint",)))
    context = _context()
    plans = _build_plans(catalog, (_intent("run_load_test"),), context)
    filler, _ = _filler(_response({"seconds": _suggestion(60)}))

    with pytest.raises(RuntimeError, match="PromptStore") as excinfo:
        await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert type(excinfo.value) is RuntimeError


async def test_history_items_are_sent_as_the_conversation_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`context.history_items` は会話部分として渡り、system 指示部には積まれない（FR-6 L168）。"""
    spy = _spy_filler(monkeypatch)
    history = ({"role": "user", "content": "HISTORY_BODY_7c21"},)
    context = _context(history_items=history)
    plans = _build_plans(_catalog(_spec()), (_intent("run_load_test"),), context)
    filler, model = _filler(_response({"seconds": _suggestion(60)}))

    await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert tuple(spy.once.history_items) == history
    assert "HISTORY_BODY_7c21" not in (model.calls[0].system_instructions or "")


async def test_utterance_is_never_concatenated_into_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """発話を system 指示部へ連結する経路を持たない（NFR-6 の計測基準・プロンプト注入経路）。

    発話（`_UTTERANCE`）と既定の会話本文（`_DEFAULT_HISTORY_BODY`）を別マーカーにして、
    「発話の漏れ」と「会話の漏れ」を独立に観測する。同一マーカーだと会話本文の混入を
    発話の混入と取り違える（またはその逆）。
    """
    spy = _spy_filler(monkeypatch)
    context = _context()
    plans = _build_plans(_catalog(_spec()), (_intent("run_load_test"),), context)
    filler, model = _filler(_response({"seconds": _suggestion(60)}))

    await _predict_params(plans, context, llm_filler=filler, prompts=None)

    system = model.calls[0].system_instructions or ""
    assert _UTTERANCE not in system
    assert _UTTERANCE not in spy.once.user_content
    # 会話部分は `history_items` としてのみ渡り、system 指示部へは積まれない。
    assert _DEFAULT_HISTORY_BODY not in system


async def test_run_context_is_forwarded_as_the_run_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """`context.run_context` が `Runner.run` の `context` として渡る（FR-6）。"""
    spy = _spy_filler(monkeypatch)
    ctx = _Ctx()
    context = _context(run_context=ctx)
    plans = _build_plans(_catalog(_spec()), (_intent("run_load_test"),), context)
    filler, _ = _filler(_response({"seconds": _suggestion(60)}))

    await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert spy.once.context is ctx


async def test_user_content_presents_the_response_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """出力形式として複合応答モデルのスキーマと JSON のみを返す制約を含める（FR-6 L168）。"""
    spy = _spy_filler(monkeypatch)
    catalog = _catalog(
        _spec(parameters=(param("seconds", int, by_llm=True, max_suggestions=3, confirm=True),))
    )
    context = _context()
    plans = _build_plans(catalog, (_intent("run_load_test"),), context)
    filler, _ = _filler(_response({"seconds": [_suggestion(60)]}))

    await _predict_params(plans, context, llm_filler=filler, prompts=None)

    content = spy.once.user_content
    assert "candidate_0" in content
    assert "seconds" in content
    # 上限は提示用スキーマの maxItems として渡る（設計 §3.8・parse 派生には付けない）。
    assert "maxItems" in content
    assert "json" in content.lower()


# ======================================================================
# 2-3: 複合応答モデルと対応表
# ======================================================================


async def test_user_content_maps_candidate_index_to_action_id_and_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`candidate_<index>` と `action_id` / `label` の対応表を含める（FR-6 L170）。"""
    spy = _spy_filler(monkeypatch)
    catalog = _catalog(
        _spec("run_load_test", label="負荷試験ラベル"),
        _spec(
            "open_dashboard",
            label="ダッシュボードラベル",
            parameters=(param("q", str, by_llm=True),),
        ),
    )
    context = _context()
    plans = _build_plans(catalog, (_intent("run_load_test"), _intent("open_dashboard")), context)
    filler, _ = _filler(_response({"seconds": _suggestion(60)}, {"q": _suggestion("x")}))

    await _predict_params(plans, context, llm_filler=filler, prompts=None)

    content = spy.once.user_content
    assert "candidate_0" in content
    assert "candidate_1" in content
    assert "run_load_test" in content
    assert "open_dashboard" in content
    assert "負荷試験ラベル" in content
    assert "ダッシュボードラベル" in content


async def test_same_action_id_twice_is_addressed_by_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一 `action_id` の候補が複数あっても位置で区別され衝突しない（FR-6 L170）。

    応答が `candidate_1` だけを載せた場合、埋まるのは 2 件目だけである。1 件目は値を得られず
    `default` 未宣言のため `NEEDS_USER` へ後退する（`on_invalid_slot` の既定 `"skip"`）。
    """
    _spy_filler(monkeypatch)
    catalog = _catalog(_spec())
    context = _context()
    plans = _build_plans(catalog, (_intent("run_load_test"), _intent("run_load_test")), context)
    filler, _ = _filler(json.dumps({"candidate_1": {"seconds": _suggestion(60)}}))

    filled, _usage = await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert _slot(filled[0], "seconds").state is SlotState.NEEDS_USER
    assert _slot(filled[1], "seconds").state is SlotState.RESOLVED
    assert _slot(filled[1], "seconds").value == 60


# ======================================================================
# 2-4: 予測委譲の本体
# ======================================================================


async def test_runner_is_driven_once_for_three_candidates_with_five_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """候補 3 件・不足 5 件でも `Runner.run` は 1 回だけである（ADR 0026 / NFR-5）。"""
    spy = _spy_filler(monkeypatch)
    catalog = _catalog(
        _spec(
            "run_load_test",
            parameters=(
                param("seconds", int, by_llm=True),
                param("region", str, by_llm=True),
                param("note", str, by_llm=True),
            ),
        ),
        _spec("open_dashboard", parameters=(param("q", str, by_llm=True),)),
        _spec("send_report", parameters=(param("to", str, by_llm=True),)),
    )
    context = _context()
    plans = _build_plans(
        catalog,
        (_intent("run_load_test"), _intent("open_dashboard"), _intent("send_report")),
        context,
    )
    filler, model = _filler(
        _response(
            {
                "seconds": _suggestion(60),
                "region": _suggestion("jp"),
                "note": _suggestion("n"),
            },
            {"q": _suggestion("x")},
            {"to": _suggestion("ops@example.com")},
        )
    )

    _filled, usage = await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert len(spy.calls) == 1
    assert len(model.calls) == 1
    assert usage.runs == 1
    assert usage.candidates == 3


async def test_no_missing_parameter_skips_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """`NEEDS_LLM` が 1 件も無ければ `Runner.run` は 0 回で決定的段の計画をそのまま返す。"""
    spy = _spy_filler(monkeypatch)
    catalog = _catalog(_spec())
    context = _context()
    plans = _build_plans(catalog, (_intent("run_load_test", seconds=30),), context)
    filler, model = _filler("{}")

    filled, usage = await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert spy.calls == []
    assert model.calls == []
    assert filled == plans
    assert usage == ParamUsage(
        runs=0, model_calls=0, candidates=0, input_tokens=None, output_tokens=None
    )


async def test_code_fenced_response_is_stripped_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """コードフェンス付きの応答も `_strip_code_fence` を経て parse される（FR-6）。"""
    _spy_filler(monkeypatch)
    context = _context()
    plans = _build_plans(_catalog(_spec()), (_intent("run_load_test"),), context)
    body = _response({"seconds": _suggestion(60)})
    filler, _ = _filler(f"```json\n{body}\n```")

    filled, _usage = await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert _slot(filled[0], "seconds").value == 60


async def test_predicted_value_without_confirm_becomes_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`confirm=False` の予測値は `RESOLVED` / `origin=LLM` / `detail=None` になる（FR-6）。"""
    _spy_filler(monkeypatch)
    context = _context()
    plans = _build_plans(_catalog(_spec()), (_intent("run_load_test"),), context)
    filler, _ = _filler(_response({"seconds": _suggestion(60)}))

    filled, _usage = await _predict_params(plans, context, llm_filler=filler, prompts=None)

    slot = _slot(filled[0], "seconds")
    assert slot.state is SlotState.RESOLVED
    assert slot.value == 60
    assert slot.origin is Origin.LLM
    assert slot.detail is None


async def test_predicted_value_with_confirm_becomes_needs_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`confirm=True` の予測値は `NEEDS_CONFIRMATION` へ遷移し `suggestions` に載る（FR-6）。"""
    _spy_filler(monkeypatch)
    catalog = _catalog(_spec(parameters=(param("seconds", int, by_llm=True, confirm=True),)))
    context = _context()
    plans = _build_plans(catalog, (_intent("run_load_test"),), context)
    filler, _ = _filler(_response({"seconds": _suggestion(60, level="medium")}))

    filled, _usage = await _predict_params(plans, context, llm_filler=filler, prompts=None)

    slot = _slot(filled[0], "seconds")
    assert slot.state is SlotState.NEEDS_CONFIRMATION
    assert slot.value is None
    assert slot.origin is Origin.LLM
    assert slot.detail is None
    assert [(s.value, s.level) for s in slot.suggestions] == [(60, ConfidenceLevel.MEDIUM)]


async def test_suggestions_are_sorted_by_level_desc_and_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ConfidenceLevel` 降順の stable sort 後に `max_suggestions` で切り捨てる（FR-6）。

    宣言順は `10(medium) / 20(certain) / 30(medium) / 40(high)` で、期待値
    `[20, 40, 10]` は宣言順の先頭 3 件（`[10, 20, 30]`）と一致しない。降順 sort・切り捨て・
    同レベル内の安定性（`10` が `30` より先）の 3 つが同時に効いていないと通らない。
    """
    _spy_filler(monkeypatch)
    catalog = _catalog(
        _spec(parameters=(param("seconds", int, by_llm=True, max_suggestions=3, confirm=True),))
    )
    context = _context()
    plans = _build_plans(catalog, (_intent("run_load_test"),), context)
    filler, _ = _filler(
        _response(
            {
                "seconds": [
                    _suggestion(10, level="medium"),
                    _suggestion(20, level="certain"),
                    _suggestion(30, level="medium"),
                    _suggestion(40, level="high"),
                ]
            }
        )
    )

    filled, _usage = await _predict_params(plans, context, llm_filler=filler, prompts=None)

    slot = _slot(filled[0], "seconds")
    assert [s.value for s in slot.suggestions] == [20, 40, 10]
    assert [s.level for s in slot.suggestions] == [
        ConfidenceLevel.CERTAIN,
        ConfidenceLevel.HIGH,
        ConfidenceLevel.MEDIUM,
    ]


async def test_already_resolved_slots_are_untouched_by_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """予測段は `NEEDS_LLM` 以外のスロットを書き換えない（NFR-6）。"""
    _spy_filler(monkeypatch)
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, by_llm=True),
                param("region", str, from_context="tenant.id"),
            )
        )
    )
    context = _context(run_context=_Ctx())
    plans = _build_plans(catalog, (_intent("run_load_test"),), context)
    filler, _ = _filler(_response({"seconds": _suggestion(60)}))

    filled, _usage = await _predict_params(plans, context, llm_filler=filler, prompts=None)

    region = _slot(filled[0], "region")
    assert region.state is SlotState.RESOLVED
    assert region.value == "t-001"
    assert region.origin is Origin.RUN_CONTEXT
    assert region.detail == "tenant.id"


async def test_prediction_agent_is_declared_by_the_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """予測エージェントは lib の固定名で宣言され、業務 registry は関与しない（ADR 0029 (2b)）。"""
    spy = _spy_filler(monkeypatch)
    context = _context()
    plans = _build_plans(_catalog(_spec()), (_intent("run_load_test"),), context)
    filler, _ = _filler(_response({"seconds": _suggestion(60)}))

    await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert isinstance(FILLER_AGENT_NAME, str) and FILLER_AGENT_NAME
    assert spy.once.agent.name == FILLER_AGENT_NAME


# ======================================================================
# 2-4: ガードレール（ADR 0029 (2d)）
# ======================================================================


async def test_declared_guardrails_are_attached_to_both_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """登録名は専用 registry の `_wire` が解決し、実体そのものが境界別に装着される。"""
    spy = _spy_filler(monkeypatch)
    guardrails = GuardrailRegistry()
    guardrails.register(
        GuardrailSpec(
            name="in_gr",
            boundary=Boundary.INPUT,
            guardrail=build_input_guardrail("in_gr", _detect_clean),
        )
    )
    guardrails.register(
        GuardrailSpec(
            name="out_gr",
            boundary=Boundary.OUTPUT,
            guardrail=build_output_guardrail("out_gr", _detect_clean),
        )
    )
    context = _context()
    plans = _build_plans(_catalog(_spec()), (_intent("run_load_test"),), context)
    model = FakeModel().queue_text(_response({"seconds": _suggestion(60)}))
    filler = LLMFiller(model=model, guardrails=("in_gr", "out_gr"))

    await _predict_params(
        plans, context, llm_filler=filler, prompts=None, guardrail_registry=guardrails
    )

    agent = spy.once.agent
    # 値等価では複製でも通るため、解決簿の戻り値そのものであることを identity で照合する。
    assert [g is guardrails.get("in_gr") for g in agent.input_guardrails] == [True]
    assert [g is guardrails.get("out_gr") for g in agent.output_guardrails] == [True]


async def test_no_guardrail_is_attached_when_none_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`guardrails` が空（既定）なら 1 件も装着されない（FR-6）。"""
    spy = _spy_filler(monkeypatch)
    context = _context()
    plans = _build_plans(_catalog(_spec()), (_intent("run_load_test"),), context)
    filler, _ = _filler(_response({"seconds": _suggestion(60)}))

    await _predict_params(plans, context, llm_filler=filler, prompts=None)

    agent = spy.once.agent
    assert agent.input_guardrails == []
    assert agent.output_guardrails == []


async def test_guardrail_tripwire_propagates_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ガードレール発火は伝播し、`on_invalid_response="skip"` の後退を適用しない。"""
    _spy_filler(monkeypatch)
    guardrails = GuardrailRegistry()
    guardrails.register(
        GuardrailSpec(
            name="in_gr",
            boundary=Boundary.INPUT,
            guardrail=build_input_guardrail("in_gr", _detect_trip, run_in_parallel=False),
        )
    )
    catalog = _catalog(_spec(parameters=(param("seconds", int, by_llm=True, default=30),)))
    context = _context()
    plans = _build_plans(catalog, (_intent("run_load_test"),), context)
    model = FakeModel().queue_text(_response({"seconds": _suggestion(60)}))
    filler = LLMFiller(model=model, on_invalid_response="skip", guardrails=("in_gr",))

    with pytest.raises(InputGuardrailTripwireTriggered):
        await _predict_params(
            plans, context, llm_filler=filler, prompts=None, guardrail_registry=guardrails
        )


# ======================================================================
# 2-5: ParamUsage の詰め替え
# ======================================================================


async def test_usage_is_transferred_field_by_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """`AgentRunUsage` -> `ParamUsage` の詰め替えが各フィールドで正しい（FR-6）。

    `runs` / `model_calls` / `candidates` / tokens をすべて異なる値にして、1 フィールドを
    別のフィールドへ取り違える変異が個別に落ちるようにする。予測対象は 3 件で、`NEEDS_LLM`
    を持たない 4 件目は `candidates` に数えない（`len(plans)` との取り違えも検知する）。
    """
    catalog = _catalog(
        _spec("run_load_test"),
        _spec("open_dashboard", parameters=(param("q", str, by_llm=True),)),
        _spec("send_report", parameters=(param("to", str, by_llm=True),)),
        _spec("show_help", parameters=(param("topic", str, default="all"),)),
    )
    context = _context()
    plans = _build_plans(
        catalog,
        (
            _intent("run_load_test"),
            _intent("open_dashboard"),
            _intent("send_report"),
            _intent("show_help"),
        ),
        context,
    )
    _stub_filler(
        monkeypatch,
        _response(
            {"seconds": _suggestion(60)},
            {"q": _suggestion("x")},
            {"to": _suggestion("ops@example.com")},
            {},
        ),
        AgentRunUsage(model_calls=7, input_tokens=11, output_tokens=13),
    )
    filler = LLMFiller(model=FakeModel())

    _filled, usage = await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert len(plans) == 4
    assert usage == ParamUsage(
        runs=1, model_calls=7, candidates=3, input_tokens=11, output_tokens=13
    )


async def test_usage_tokens_stay_none_when_not_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """usage 未取得（tokens が `None`）はそのまま `ParamUsage` へ運ばれる（FR-6）。"""
    context = _context()
    plans = _build_plans(_catalog(_spec()), (_intent("run_load_test"),), context)
    _stub_filler(
        monkeypatch,
        _response({"seconds": _suggestion(60)}),
        AgentRunUsage(model_calls=1, input_tokens=None, output_tokens=None),
    )
    filler = LLMFiller(model=FakeModel())

    _filled, usage = await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert usage.input_tokens is None
    assert usage.output_tokens is None


# ======================================================================
# 2-6: 後退挙動
# ======================================================================


async def test_unparsable_response_raises_when_on_invalid_response_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """parse できない応答は既定（`"error"`）で例外になる（FR-7 L220）。"""
    _spy_filler(monkeypatch)
    catalog = _catalog(_spec(parameters=(param("seconds", int, by_llm=True, default=30),)))
    context = _context()
    plans = _build_plans(catalog, (_intent("run_load_test"),), context)
    filler, _ = _filler("これは JSON ではありません")

    with pytest.raises(ValidationError, match="Invalid JSON") as excinfo:
        await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert type(excinfo.value) is ValidationError


async def test_unparsable_response_falls_back_when_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """`"skip"` なら全 `NEEDS_LLM` を `default` / `NEEDS_USER` へ倒して続行する（FR-7 L221）。

    後退は `NEEDS_LLM` のスロットだけに適用され、決定的段で確定済みのスロットと他候補の
    計画はそのまま保たれる。
    """
    _spy_filler(monkeypatch)
    catalog = _catalog(
        _spec(
            "run_load_test",
            parameters=(
                param("seconds", int, by_llm=True, default=30),
                param("note", str, by_llm=True),
                param("confirmed", int, by_llm=True, default=5, confirm=True),
                param("region", str, from_context="tenant.id"),
            ),
        ),
        _spec("open_dashboard", parameters=(param("q", str, default="all"),)),
    )
    context = _context(run_context=_Ctx())
    plans = _build_plans(catalog, (_intent("run_load_test"), _intent("open_dashboard")), context)
    filler, _ = _filler("не json", on_invalid_response="skip")

    filled, usage = await _predict_params(plans, context, llm_filler=filler, prompts=None)

    seconds = _slot(filled[0], "seconds")
    assert (seconds.state, seconds.value, seconds.origin) == (
        SlotState.RESOLVED,
        30,
        Origin.DEFAULT,
    )
    note = _slot(filled[0], "note")
    assert (note.state, note.value, note.origin) == (SlotState.NEEDS_USER, None, None)
    confirmed = _slot(filled[0], "confirmed")
    assert confirmed.state is SlotState.NEEDS_CONFIRMATION
    assert confirmed.origin is Origin.DEFAULT
    assert [s.value for s in confirmed.suggestions] == [5]
    # 他スロット・他候補は保持される。
    region = _slot(filled[0], "region")
    assert (region.state, region.value) == (SlotState.RESOLVED, "t-001")
    assert filled[1] == plans[1]
    assert usage.runs == 1


async def test_missing_field_falls_back_per_slot_when_on_invalid_slot_is_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """欠落したフィールドだけを後退させ、同一応答内の他スロットは保持する（FR-7 L222）。"""
    _spy_filler(monkeypatch)
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, by_llm=True, default=30),
                param("note", str, by_llm=True),
                param("region", str, by_llm=True),
            )
        )
    )
    context = _context()
    plans = _build_plans(catalog, (_intent("run_load_test"),), context)
    filler, _ = _filler(_response({"region": _suggestion("jp")}))

    filled, _usage = await _predict_params(plans, context, llm_filler=filler, prompts=None)

    seconds = _slot(filled[0], "seconds")
    assert (seconds.state, seconds.value, seconds.origin) == (
        SlotState.RESOLVED,
        30,
        Origin.DEFAULT,
    )
    assert _slot(filled[0], "note").state is SlotState.NEEDS_USER
    region = _slot(filled[0], "region")
    assert (region.state, region.value, region.origin) == (
        SlotState.RESOLVED,
        "jp",
        Origin.LLM,
    )


async def test_missing_field_raises_when_on_invalid_slot_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`on_invalid_slot="error"` なら欠落フィールドで例外になる（FR-7 L223）。"""
    _spy_filler(monkeypatch)
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, by_llm=True, default=30),
                param("note", str, by_llm=True),
            ),
            on_invalid_slot="error",
        )
    )
    context = _context()
    plans = _build_plans(catalog, (_intent("run_load_test"),), context)
    filler, _ = _filler(_response({"seconds": _suggestion(60)}))

    with pytest.raises(ValueError, match="note") as excinfo:
        await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert type(excinfo.value) is ValueError


async def test_type_mismatch_is_handled_per_slot_not_as_a_response_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """宣言型に合わない値は**スロット単位**で後退させる（FR-7 L222）。

    `on_invalid_response="error"`（既定）でも例外にならないことを同じ 1 本で固定する。応答
    全体は JSON として parse できているため `on_invalid_response` の管轄へ倒してはならず、
    倒す実装（モデル全体の検証 1 回で済ませる実装）はここで落ちる。

    後退は当該スロットに閉じ、同一計画内の正しい LLM 値・決定的段で確定済みのスロット・
    他候補の計画は保持される。
    """
    _spy_filler(monkeypatch)
    catalog = _catalog(
        _spec(
            "run_load_test",
            parameters=(
                param("seconds", int, by_llm=True, default=30),
                param("note", str, by_llm=True),
                param("region", str, by_llm=True),
            ),
        ),
        _spec("open_dashboard", parameters=(param("q", str, default="all"),)),
    )
    context = _context()
    plans = _build_plans(catalog, (_intent("run_load_test"), _intent("open_dashboard")), context)
    filler, _ = _filler(
        _response(
            {
                # int 宣言に str / str 宣言に int（pydantic は既定でこの 2 方向を強制変換しない）。
                "seconds": _suggestion("abc"),
                "note": _suggestion(5),
                "region": _suggestion("jp"),
            },
            {},
        )
    )
    assert filler.on_invalid_response == "error"

    filled, usage = await _predict_params(plans, context, llm_filler=filler, prompts=None)

    seconds = _slot(filled[0], "seconds")
    assert (seconds.state, seconds.value, seconds.origin) == (
        SlotState.RESOLVED,
        30,
        Origin.DEFAULT,
    )
    note = _slot(filled[0], "note")
    assert (note.state, note.value, note.origin) == (SlotState.NEEDS_USER, None, None)
    region = _slot(filled[0], "region")
    assert (region.state, region.value, region.origin) == (SlotState.RESOLVED, "jp", Origin.LLM)
    assert filled[1] == plans[1]
    assert usage.runs == 1


async def test_type_mismatch_raises_when_on_invalid_slot_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`on_invalid_slot="error"` なら宣言型に合わない値で `ValueError` になる（FR-7 L223）。"""
    _spy_filler(monkeypatch)
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, by_llm=True, default=30),
                param("note", str, by_llm=True),
            ),
            on_invalid_slot="error",
        )
    )
    context = _context()
    plans = _build_plans(catalog, (_intent("run_load_test"),), context)
    filler, _ = _filler(_response({"seconds": _suggestion(60), "note": _suggestion(5)}))

    with pytest.raises(ValueError, match="note") as excinfo:
        await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert type(excinfo.value) is ValueError


# ======================================================================
# 会話部分の欠落検知（暗黙失敗の検知）
# ======================================================================


async def test_empty_history_with_prediction_targets_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """予測対象があるのに会話が空なら、駆動する前に `RuntimeError` になる。

    発話は system 指示部へ連結されない（NFR-6）ため、予測エージェントへ会話が届く経路は
    `context.history_items` だけである。空のまま駆動すると当て推量の値が例外なしで返り、
    利用者から見て「それっぽい誤り」になる（暗黙失敗）。既存の「予測経路を結線したのに
    必要な入力が欠けている」系の `RuntimeError`（セグメント宣言あり + `prompts` 未結線）と
    同じクラスとして扱う。
    """
    _forbid_filler(monkeypatch)
    catalog = _catalog(_spec("run_load_test"))
    context = _context(history_items=())
    plans = _build_plans(catalog, (_intent("run_load_test"),), context)
    filler, model = _filler(_response({"seconds": _suggestion(60)}))

    with pytest.raises(RuntimeError, match="history") as excinfo:
        await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert type(excinfo.value) is RuntimeError
    # どの候補の予測が落ちたかを利用者が特定できること。
    assert "run_load_test" in str(excinfo.value)
    # 課金前に落ちる保証（モデルへも到達しない）。
    assert model.calls == []


async def test_empty_history_message_guides_appending_the_current_utterance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """会話が空のときの誘導は「history を渡す」だけで終わらない。

    `IntentQuery(history=session)` を渡していても、現在発話をまだセッションへ積んでいない
    初回ターンでは `history_items` が空になる（`DefaultContextBuilder` は
    `query.history.get_items()` からのみ会話部分を作り `utterance` を含めない）。
    「history を渡せ」だけの文言はこの利用者を救えないため、発話をセッションへ積む旨の
    誘導も必須とする。実装が必ず含める語は `"utterance"` と `"session"` の 2 語である。
    """
    _forbid_filler(monkeypatch)
    catalog = _catalog(_spec("run_load_test"))
    context = _context(history_items=())
    plans = _build_plans(catalog, (_intent("run_load_test"),), context)
    filler, _model = _filler(_response({"seconds": _suggestion(60)}))

    with pytest.raises(RuntimeError) as excinfo:
        await _predict_params(plans, context, llm_filler=filler, prompts=None)

    message = str(excinfo.value)
    assert type(excinfo.value) is RuntimeError
    assert "utterance" in message
    assert "session" in message


async def test_non_empty_history_still_drives_the_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """会話が非空なら従来どおり予測が 1 回駆動される（欠落検知の誤検知防止）。"""
    spy = _spy_filler(monkeypatch)
    history = ({"role": "user", "content": "HISTORY_BODY_4b19"},)
    context = _context(history_items=history)
    plans = _build_plans(_catalog(_spec()), (_intent("run_load_test"),), context)
    filler, _ = _filler(_response({"seconds": _suggestion(60)}))

    filled, usage = await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert len(spy.calls) == 1
    assert tuple(spy.once.history_items) == history
    assert _slot(filled[0], "seconds").value == 60
    assert usage.runs == 1


async def test_empty_history_without_prediction_targets_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """予測対象が 0 件なら会話が空でも正常系である（欠落検知の誤検知防止）。

    検知の条件は「予測を駆動する」ことであり、`history_items` が空であること単体ではない。
    条件を取り違えた実装（対象の有無を見ずに落とす実装）はここで落ちる。
    """
    _forbid_filler(monkeypatch)
    catalog = _catalog(_spec())
    context = _context(history_items=())
    plans = _build_plans(catalog, (_intent("run_load_test", seconds=30),), context)
    filler, model = _filler("{}")

    filled, usage = await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert filled == plans
    assert model.calls == []
    assert usage == ParamUsage(
        runs=0, model_calls=0, candidates=0, input_tokens=None, output_tokens=None
    )


async def test_on_invalid_slot_is_read_per_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """`resolved_on_invalid_slot` は計画ごとに読まれる（`ActionSpec` の上書きが効く）。"""
    _spy_filler(monkeypatch)
    catalog = _catalog(
        _spec("run_load_test", parameters=(param("seconds", int, by_llm=True, default=30),)),
        _spec(
            "open_dashboard",
            parameters=(param("q", str, by_llm=True, default="all"),),
            on_invalid_slot="error",
        ),
        on_invalid_slot="skip",
    )
    context = _context()
    plans = _build_plans(catalog, (_intent("run_load_test"), _intent("open_dashboard")), context)
    filler, _ = _filler(_response({}, {"q": _suggestion("x")}))

    filled, _usage = await _predict_params(plans, context, llm_filler=filler, prompts=None)

    assert _slot(filled[0], "seconds").origin is Origin.DEFAULT
    assert _slot(filled[1], "q").origin is Origin.LLM
