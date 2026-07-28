"""L2: `_rollout._apply_candidate` の build 段階 fail-closed（Issue #40 T6・RED）。

境界マーカー崩れ・その他の `ValueError`（`_new_default_build` の build から raise）を
`_apply_candidate` が catch し `_reinject_vars` の None と同一の per-candidate 無効化経路
（呼び出し元 rollout の reward 0.0）へ倒すことを検証する。build 段階以外の想定外例外
（`RuntimeError` 等）は暴走防止のため catch せず伝搬することも合わせて検証する。
`TypeError`（vars=callable の非 dict 戻り値）は build 呼び出し時点でなく rollout 実行時に
発生するため本ファイルのスコープ外（T5 でカバー済み）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.prompts import PromptLayout, PromptStore
from oai_agentspec.runtime.lightning import Slot
from oai_agentspec.runtime.lightning import prompt_slot as _prompt_slot
from oai_agentspec.runtime.lightning._rollout import _apply_candidate, _observe_route_steps
from oai_agentspec.runtime.lightning.types import _CandidateInvalid

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.unit


def _store_new_shape(tmp_path: Path) -> PromptStore:
    """新 shape（agent= / tune=）テスト用ストア（`test_slots_l1._store_new_shape` と同構成）。"""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "triage.md").write_text("Triage seed ${tone}", encoding="utf-8")
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "main.md").write_text("BASE ${org}", encoding="utf-8")
    return PromptStore(tmp_path, PromptLayout(base="base", parts="parts", agents="agents"))


def _registry() -> AgentRegistry:
    reg = AgentRegistry()
    reg.register(AgentSpec(name="triage", instructions="orig", model=FakeModel()))
    return reg


def _marker_slot(tmp_path: Path) -> Slot:
    """`n_tune=2`（base:main + agent:triage）のマーカー付き新 shape slot。"""
    return _prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        tune=["main", "triage"],
        vars={"org": "AgentSpec", "tone": "formal"},
    )


# ----------------------------------------------------------------------
# build 段階のマーカー崩れ・ValueError -> None（per-candidate 無効化）
# ----------------------------------------------------------------------


def test_apply_candidate_marker_broken_returns_none(tmp_path: Path) -> None:
    """マーカー重複候補は build で `_CandidateInvalid` を送出し None が返る。

    マーカーそのものを候補から欠落させると `${var}` 喪失検査（`_reinject_vars`）が先に None を
    返してしまい build の `_CandidateInvalid` 経路を通らない（マーカーも `${...}` 記法のため
    seed の placeholder 検査対象に含まれる）。build 段階の `_CandidateInvalid` 捕捉を確実に
    検証するため、必要な `${var}` / マーカーはすべて候補に残しつつ **マーカーを重複させて**
    順序不整合を発生させる（FU: 旧 T6 の generic ValueError から内部 sentinel へ移行）。
    """
    slot = _marker_slot(tmp_path)
    target = AgentSpec(name="triage", instructions="orig", model=FakeModel())
    candidate = {
        "triage": (
            "OPTIMIZED_MAIN ${org}\n\n${oas_boundary_1}\n\n"
            "${oas_boundary_1}\n\nOPTIMIZED_TRIAGE ${tone}"
        ),
    }

    result = _apply_candidate(
        target=target, registry=None, slots={"triage": slot}, rebind=None, candidate=candidate
    )

    assert result is None


def test_apply_candidate_valid_marker_succeeds(tmp_path: Path) -> None:
    """正しいマーカー入り候補は build が呼ばれ `(target, registry)` が返る（None でない）。"""
    slot = _marker_slot(tmp_path)
    target = AgentSpec(name="triage", instructions="orig", model=FakeModel())
    candidate = {
        "triage": "OPTIMIZED_MAIN ${org}\n\n${oas_boundary_1}\n\nOPTIMIZED_TRIAGE ${tone}",
    }

    result = _apply_candidate(
        target=target, registry=None, slots={"triage": slot}, rebind=None, candidate=candidate
    )

    assert result is not None
    built_target, built_registry = result
    assert built_target.instructions == "OPTIMIZED_MAIN AgentSpec\n\nOPTIMIZED_TRIAGE formal"
    assert built_registry is None


def test_apply_candidate_build_candidateinvalid_returns_none() -> None:
    """`build` が sentinel `_CandidateInvalid` を送出したとき None が返る。

    FU（C3 fix）: 旧 T6 は generic `ValueError` を catch していたが、これは旧 shape の
    利用者 build から出た正当な validation エラーまで silent 化する regression を生んでいた。
    今は内部の `_CandidateInvalid` sentinel のみを catch する（generic ValueError は伝搬）。
    """

    def _raise_candidate_invalid(_candidate: str) -> AgentSpec:
        raise _CandidateInvalid("marker broken")

    slot = Slot(name="x", seed="seed text", build=_raise_candidate_invalid)
    target = AgentSpec(name="x", instructions="orig", model=FakeModel())

    result = _apply_candidate(
        target=target,
        registry=None,
        slots={"x": slot},
        rebind=None,
        candidate={"x": "seed text"},
    )

    assert result is None


def test_apply_candidate_build_generic_valueerror_propagates() -> None:
    """旧 shape の利用者 build が投げる generic `ValueError` は伝搬する（C3 regression fix）。

    T6 では generic ValueError を silent に None に化けさせていたため、旧 shape の
    `prompt_slot(build=my_build)` を使う既存ユーザーの fail-closed 検証も silent 化されていた。
    FU（C3 fix）で内部 sentinel `_CandidateInvalid` のみに catch を絞り、generic ValueError の
    診断性を復元する。
    """

    def _raise_value_error(_candidate: str) -> AgentSpec:
        raise ValueError("user validation failed")

    slot = Slot(name="x", seed="seed text", build=_raise_value_error)
    target = AgentSpec(name="x", instructions="orig", model=FakeModel())

    with pytest.raises(ValueError, match="user validation failed"):
        _apply_candidate(
            target=target,
            registry=None,
            slots={"x": slot},
            rebind=None,
            candidate={"x": "seed text"},
        )


# ----------------------------------------------------------------------
# build 段階以外の想定外 Exception は伝搬する（暴走防止・ValueError のみ catch の意）
# ----------------------------------------------------------------------


def test_apply_candidate_unexpected_exception_propagates() -> None:
    """`slot.build` が `RuntimeError` を送出した場合は catch せず伝搬する。"""

    def _raise_runtime_error(_candidate: str) -> AgentSpec:
        raise RuntimeError("unexpected")

    slot = Slot(name="x", seed="seed text", build=_raise_runtime_error)
    target = AgentSpec(name="x", instructions="orig", model=FakeModel())

    with pytest.raises(RuntimeError, match="unexpected"):
        _apply_candidate(
            target=target,
            registry=None,
            slots={"x": slot},
            rebind=None,
            candidate={"x": "seed text"},
        )


# ----------------------------------------------------------------------
# グラフ経路（target が非 AgentSpec・registry.clone(transform_spec=...) 経由）
# ----------------------------------------------------------------------


def test_apply_candidate_graph_route_marker_broken_returns_none() -> None:
    """グラフ経路でも `registry.clone` 中の `_CandidateInvalid` は None を返す。

    `target` が `AgentSpec` でない場合、`_apply_candidate` は `registry.clone(transform_spec=...)`
    経由で各 slot の `build` を呼ぶ。build が `_CandidateInvalid` を送出したときの catch は
    単一 AgentSpec 経路とは別コードパスのため、専用に検証する（FU: 旧 T6 の ValueError から
    内部 sentinel への移行）。
    """

    def _raise_candidate_invalid(_candidate: str) -> AgentSpec:
        raise _CandidateInvalid("marker broken")

    slot = Slot(name="triage", seed="seed text", build=_raise_candidate_invalid)
    registry = _registry()

    result = _apply_candidate(
        target="triage",
        registry=registry,
        slots={"triage": slot},
        rebind=None,
        candidate={"triage": "seed text"},
    )

    assert result is None


def test_apply_candidate_graph_route_generic_valueerror_propagates() -> None:
    """グラフ経路でも generic ValueError（利用者 build や `_resolve_spec`）は伝搬する（C3 fix）。"""

    def _raise_value_error(_candidate: str) -> AgentSpec:
        raise ValueError("user validation failed")

    slot = Slot(name="triage", seed="seed text", build=_raise_value_error)
    registry = _registry()

    with pytest.raises(ValueError, match="user validation failed"):
        _apply_candidate(
            target="triage",
            registry=registry,
            slots={"triage": slot},
            rebind=None,
            candidate={"triage": "seed text"},
        )


def test_apply_candidate_graph_route_runtime_error_propagates() -> None:
    """グラフ経路でも `slot.build` の `RuntimeError` は catch せず伝搬する。"""

    def _raise_runtime_error(_candidate: str) -> AgentSpec:
        raise RuntimeError("unexpected")

    slot = Slot(name="triage", seed="seed text", build=_raise_runtime_error)
    registry = _registry()

    with pytest.raises(RuntimeError, match="unexpected"):
        _apply_candidate(
            target="triage",
            registry=registry,
            slots={"triage": slot},
            rebind=None,
            candidate={"triage": "seed text"},
        )


# ----------------------------------------------------------------------
# `_observe_route_steps` の fail-closed 2 経路（pre-flight route coverage）
# ----------------------------------------------------------------------


async def test_observe_route_steps_applied_none_returns_empty_no_reach_claim() -> None:
    """`_apply_candidate` が None（必要 `${var}` 喪失）を返すとき `((), False)` を返す。

    誤って「全 slot 到達済み」を申告すると pre-flight が本来検出すべき未到達 slot を
    silent に見逃す（本 PR が防ごうとしている silent no-op の再発）。
    """
    slot = Slot(
        name="x",
        seed="seed ${var} text",
        build=lambda c: AgentSpec(name="x", instructions=c, model=FakeModel()),
    )
    target = AgentSpec(name="x", instructions="orig", model=FakeModel())

    # seeds に seed が持つ ${var} プレースホルダが含まれない → `_reinject_vars` が None を返し
    # `_apply_candidate` も None を返す（applied is None 経路）。
    steps, interrupted = await _observe_route_steps(
        target=target,
        registry=None,
        slots={"x": slot},
        seeds={"x": "seed text without placeholder"},
        case={"input": "hi"},
        approvals=None,
        tool_mocks=None,
        context_factory=None,
    )

    assert (steps, interrupted) == ((), False)


async def test_observe_route_steps_candidate_invalid_returns_empty_no_reach_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_run_one` 実行中の `_CandidateInvalid` は catch され `((), False)` を返す。

    利用者 `build=` / `vars=callable` が rollout 実行時（SDK `Runner.run` 経由）に候補を無効化
    する経路。誤って「全 slot 到達済み」を申告すると同様に未到達 slot を見逃す。
    """

    async def _raise_candidate_invalid(self: Any, agent: Any, value: Any, **kwargs: Any) -> Any:
        raise _CandidateInvalid("boundary marker broken")

    monkeypatch.setattr(
        "oai_agentspec._adapters.DefaultRunnerAdapter.run_with_observation",
        _raise_candidate_invalid,
        raising=True,
    )

    slot = Slot(
        name="x",
        seed="seed text",
        build=lambda c: AgentSpec(name="x", instructions=c, model=FakeModel()),
    )
    target = AgentSpec(name="x", instructions="orig", model=FakeModel())

    steps, interrupted = await _observe_route_steps(
        target=target,
        registry=None,
        slots={"x": slot},
        seeds={"x": "seed text"},
        case={"input": "hi"},
        approvals=None,
        tool_mocks=None,
        context_factory=None,
    )

    assert (steps, interrupted) == ((), False)
