"""L1: `ActionCatalog.bind()` と `ActionPlanner` の構造（設計 §5 タスク 1-12 / §3.4a）。

FR-3 の受け入れ基準のうち、結線と `validate()` に関わる部分を pin する。設計の核心は
**`bind()` が catalog を変異させず frozen な `ActionPlanner` を返す**ことであり、その帰結
（2 回の `bind()` が独立・bind 後の `register()` が既存 planner へ届かない・未結線の `plan()` が
`ActionCatalog` にメソッドが無いため構造的に呼べない）を対象とする。

`ActionPlanner` はシンボルを直接 import せず `bind()` の戻り値から観測する。実装前でも
このモジュールが収集でき、失敗が「`bind` が未実装だから」に限られるようにするためである。

外部依存（agents / openai）なし。`AgentRegistry` / `PromptStore` は L1 の範囲で
duck-typed な Fake と tmp_path 上の実物を使う。`plan()` の挙動は `test_catalog_plan_l2.py`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from oai_agentspec.prompts import PromptLayout, PromptStore
from oai_agentspec.runtime.intent.actions import ActionCatalog, ActionSpec, param
from oai_agentspec.runtime.intent.binding import CandidateSource, LLMFiller
from oai_agentspec.runtime.intent.types import IntentContext, IntentPrediction

pytestmark = pytest.mark.unit


_LAYOUT = PromptLayout(base="base", parts="parts", agents="agents")


# ---- Fake ----


class _CountingAgentRegistry:
    """`AgentRegistry` の Fake。`names()` の呼び出し回数を数える（検証回数の観測用）。"""

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


class _FakeGuardrailRegistry:
    """`GuardrailRegistry` の Fake。"""

    def __init__(self, *names: str) -> None:
        self._names = sorted(names)

    def names(self) -> list[str]:
        """登録済みガードレール名を昇順で返す。"""
        return list(self._names)

    def get(self, name: str) -> object:
        """未登録名なら `KeyError`（本物と同じ契約）。"""
        if name not in self._names:
            raise KeyError(name)
        return object()


class _CountingGenerator:
    """`CandidateGenerator` の Fake。`generate()` の呼び出しを数える。"""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, context: IntentContext[Any]) -> IntentPrediction:
        """空の予測を返し、呼び出しを数える。"""
        self.calls += 1
        return IntentPrediction(candidates=())


class _Ctx:
    """検査 7 用の代表インスタンス。"""

    region = "jp"


# ---- ヘルパ ----


def _spec(
    action_id: str = "run_load_test",
    *,
    action_agent: str = "load_test_agent",
    label: str = "負荷試験 ${seconds} 秒",
    parameters: tuple[Any, ...] | None = None,
    prompt: tuple[str, ...] = (),
) -> ActionSpec:
    """他の検査に触れない健全な `ActionSpec` を組む。"""
    return ActionSpec(
        action_id=action_id,
        description="負荷試験を実行する",
        action_agent=action_agent,
        label=label,
        parameters=parameters if parameters is not None else (param("seconds", int, default=30),),
        prompt=prompt,
    )


def _catalog(*specs: ActionSpec, **kwargs: Any) -> ActionCatalog:
    """宣言簿を組んで `specs` を登録した `ActionCatalog` を返す。"""
    catalog = ActionCatalog(**kwargs)
    for spec in specs:
        catalog.register(spec)
    return catalog


def _store(tmp_path: Path, **bodies: str) -> PromptStore:
    """`parts/<name>.md` を書き出した実物の `PromptStore` を返す。"""
    parts = tmp_path / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for name, body in bodies.items():
        (parts / f"{name}.md").write_text(body, encoding="utf-8")
    return PromptStore(tmp_path, _LAYOUT)


def _bind(catalog: ActionCatalog, **kwargs: Any) -> Any:
    """`registry` の既定を補って `catalog.bind()` を呼ぶ。"""
    kwargs.setdefault("registry", _CountingAgentRegistry("load_test_agent"))
    return catalog.bind(**kwargs)


# ---- bind の戻り値（§3.4a・実測 30-1 / 30-2） ----


def test_bind_returns_an_action_planner() -> None:
    """`bind()` は `actions.ActionPlanner`（pydantic モデル）を返す。"""
    from oai_agentspec.runtime.intent import actions as actions_mod

    planner = _bind(_catalog(_spec()))

    assert type(planner).__name__ == "ActionPlanner"
    assert type(planner) is actions_mod.ActionPlanner
    assert isinstance(planner, BaseModel)


def test_bound_planner_is_frozen() -> None:
    """`ActionPlanner` は frozen（結線を後から差し替えられない）。"""
    planner = _bind(_catalog(_spec()))

    assert type(planner).model_config.get("frozen") is True
    # 未宣言名への代入は、非 frozen なら素の ValueError、frozen なら ValidationError。
    # ValidationError は ValueError 派生なので `pytest.raises(ValueError)` では区別できない。
    with pytest.raises(ValidationError):
        planner.registry = object()


def test_planner_exposes_validate_and_plan() -> None:
    """公開メソッドは `validate` / `plan` の 2 つ（§4a-3）。"""
    planner = _bind(_catalog(_spec()))

    assert callable(planner.validate)
    assert callable(planner.plan)


def test_catalog_has_no_plan_or_validate() -> None:
    """未結線の `plan()` / `validate()` は構造的に呼べない（実測 30-10）。"""
    catalog = _catalog(_spec())

    assert not hasattr(catalog, "plan")
    assert not hasattr(catalog, "validate")


def test_bind_is_keyword_only() -> None:
    """結線引数はすべてキーワード専用（どの部品かを呼び出し側で読めるようにする）。"""
    catalog = _catalog(_spec())
    with pytest.raises(TypeError):
        catalog.bind(_CountingAgentRegistry("load_test_agent"))


def test_bind_requires_registry() -> None:
    """`registry` は必須（実行先の解決簿が無い結線を作らせない）。"""
    with pytest.raises(TypeError):
        _catalog(_spec()).bind()


# ---- bind は catalog を変異させない（ADR 0029 Decision 2・実測 30-1） ----


def test_bind_does_not_mutate_the_catalog() -> None:
    """`bind()` の前後で宣言簿の状態は変わらない。"""
    catalog = _catalog(_spec(), prompt=("part:common",), prompt_vars={"tenant": "tenant.id"})
    before = (catalog.names(), catalog.prompt, dict(catalog.prompt_vars), catalog.on_invalid_slot)

    _bind(catalog)

    assert catalog.names() == before[0]
    assert catalog.prompt == before[1]
    assert dict(catalog.prompt_vars) == before[2]
    assert catalog.on_invalid_slot == before[3]


def test_bind_does_not_modify_registered_specs() -> None:
    """登録済み `ActionSpec` の内容にも触れない（同一オブジェクトのまま）。"""
    spec = _spec()
    catalog = _catalog(spec)

    _bind(catalog)

    assert catalog.get("run_load_test") is spec


def test_bind_does_not_run_the_generator() -> None:
    """`bind()` は結線を保持するだけで実行しない（LLM 0 回）。"""
    generator = _CountingGenerator()
    registry = _CountingAgentRegistry("load_test_agent")

    _bind(_catalog(_spec()), registry=registry, candidates=CandidateSource(generator=generator))

    assert generator.calls == 0
    assert registry.name_calls == 0


# ---- 2 回の bind は独立（実測 30-5 / 30-6 / 30-7） ----


def test_bind_twice_returns_distinct_planners() -> None:
    """`bind()` を 2 回呼ぶと別インスタンスが得られる。"""
    catalog = _catalog(_spec())

    assert _bind(catalog) is not _bind(catalog)


def test_bind_twice_returns_equal_planners() -> None:
    """同じ宣言簿・同じ結線から得た 2 つの planner は等価である。"""
    catalog = _catalog(_spec())
    registry = _CountingAgentRegistry("load_test_agent")

    assert catalog.bind(registry=registry) == catalog.bind(registry=registry)


def test_validate_does_not_change_planner_equality() -> None:
    """`validate()` の実行済みフラグは等価性に漏れない（`_DeclaredFieldsEq`）。"""
    catalog = _catalog(_spec())
    registry = _CountingAgentRegistry("load_test_agent")
    validated = catalog.bind(registry=registry)
    untouched = catalog.bind(registry=registry)

    validated.validate()

    assert validated == untouched


def test_register_after_bind_does_not_reach_an_existing_planner() -> None:
    """bind 後の `register()` は既存 planner へ届かない（宣言簿のスナップショット）。"""
    catalog = _catalog(_spec())
    planner = _bind(catalog)

    catalog.register(_spec("broken_action", action_agent="ghost_agent"))

    # 宣言簿には登録済みだが、スナップショットを持つ既存 planner の検証は通る。
    assert "broken_action" in catalog.names()
    assert planner.validate() is None
    # 同じ宣言簿を bind し直すと、新しい planner は当然落ちる。
    with pytest.raises(KeyError):
        _bind(catalog).validate()


def test_the_same_catalog_can_be_bound_to_different_registries() -> None:
    """同じ宣言簿を別結線で使い回せる（実測 30-7）。"""
    catalog = _catalog(_spec())
    good = catalog.bind(registry=_CountingAgentRegistry("load_test_agent"))
    bad = catalog.bind(registry=_CountingAgentRegistry("other_agent"))

    assert good.validate() is None
    with pytest.raises(KeyError):
        bad.validate()


def test_validate_state_is_independent_between_planners() -> None:
    """一方の `validate()` 実行は他方の検証状態へ影響しない（実測 30-5）。"""
    catalog = _catalog(_spec())
    registry_a = _CountingAgentRegistry("load_test_agent")
    registry_b = _CountingAgentRegistry("load_test_agent")
    planner_a = catalog.bind(registry=registry_a)
    catalog.bind(registry=registry_b)

    planner_a.validate()

    assert registry_a.name_calls > 0
    assert registry_b.name_calls == 0


# ---- validate() は結線を使う（FR-3・1-11 の検査を ActionPlanner 経由で） ----


def test_validate_returns_none_for_a_valid_catalog() -> None:
    """整合している宣言では `None` を返す。"""
    assert _bind(_catalog(_spec())).validate() is None


def test_validate_uses_the_wired_registry() -> None:
    """検査 1 は結線された `registry` を見る（未登録の実行先は `KeyError`）。"""
    planner = _catalog(_spec(action_agent="ghost_agent")).bind(
        registry=_CountingAgentRegistry("load_test_agent")
    )
    with pytest.raises(KeyError) as excinfo:
        planner.validate()

    assert type(excinfo.value) is KeyError
    assert "ghost_agent" in str(excinfo.value)


def test_validate_uses_the_wired_prompts(tmp_path: Path) -> None:
    """検査 3 は結線された `prompts` を見る（未解決セグメントは `ValueError`）。"""
    prompts = _store(tmp_path, hint="ヒント本文")
    catalog = _catalog(
        _spec(
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:missing_segment",),
        )
    )
    with pytest.raises(ValueError) as excinfo:
        _bind(catalog, prompts=prompts).validate()

    assert type(excinfo.value) is ValueError
    assert "missing_segment" in str(excinfo.value)


def test_validate_raises_runtime_error_when_prompts_are_not_wired() -> None:
    """セグメント宣言があるのに `prompts` 未結線なら `RuntimeError`（§3.4a の規則 2）。"""
    catalog = _catalog(
        _spec(
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:hint",),
        )
    )
    with pytest.raises(RuntimeError) as excinfo:
        _bind(catalog).validate()

    assert type(excinfo.value) is RuntimeError


def test_validate_passes_without_prompts_when_no_segment_is_declared() -> None:
    """セグメント宣言が無ければ `prompts` 未結線でも通る。"""
    assert _bind(_catalog(_spec())).validate() is None


def test_validate_only_usage_without_candidates_is_allowed() -> None:
    """`candidates` を省略した結線でも `validate()` の利用は妨げない（§3.4a）。"""
    assert _bind(_catalog(_spec()), candidates=None).validate() is None


def test_validate_accepts_a_context_keyword() -> None:
    """`validate(context=...)` でパスの構造検査を有効にできる（検査 7）。"""
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, default=30),
                param("target", str, from_context="tenant.nonexistent"),
            )
        )
    )
    planner = _bind(catalog)

    assert planner.validate() is None
    with pytest.raises(ValueError) as excinfo:
        planner.validate(context=_Ctx())

    assert "tenant.nonexistent" in str(excinfo.value)


def test_validate_uses_the_wired_guardrail_registry() -> None:
    """検査 9 は結線された `guardrail_registry` を見る。"""
    catalog = _catalog(_spec())
    filler = LLMFiller(model="gpt-x", guardrails=("ghost_guard",))

    with pytest.raises(KeyError):
        _bind(
            catalog, guardrail_registry=_FakeGuardrailRegistry("pii"), llm_filler=filler
        ).validate()

    assert (
        _bind(
            catalog,
            guardrail_registry=_FakeGuardrailRegistry("ghost_guard"),
            llm_filler=filler,
        ).validate()
        is None
    )


def test_guardrails_without_a_registry_raise_value_error() -> None:
    """`guardrails` 非空 + 解決簿未結線は `ValueError`（名前の未登録と型で分ける）。"""
    planner = _bind(_catalog(_spec()), llm_filler=LLMFiller(model="gpt-x", guardrails=("pii",)))
    with pytest.raises(ValueError) as excinfo:
        planner.validate()

    assert type(excinfo.value) is ValueError


def test_validate_is_idempotent() -> None:
    """`validate()` を 2 回呼んでも副作用が無く、同じ結果を返す。"""
    planner = _bind(_catalog(_spec()))

    assert planner.validate() is None
    assert planner.validate() is None
