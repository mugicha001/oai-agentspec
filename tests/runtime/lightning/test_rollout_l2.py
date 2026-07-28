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

from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph
from oai_agentspec.prompts import PromptLayout, PromptStore
from oai_agentspec.runtime.lightning import FailureKind, OptimizeError, Slot
from oai_agentspec.runtime.lightning import prompt_slot as _prompt_slot
from oai_agentspec.runtime.lightning._rollout import (
    _apply_candidate,
    _check_route_coverage,
    _observe_route_steps,
)
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


async def test_observe_route_steps_applied_none_returns_invalid_signal() -> None:
    """`_apply_candidate` が None（必要 `${var}` 喪失）を返すとき `(None, False)` を返す。

    `()` へ畳むと「実行済みだが観測が空」（防御的経路）と区別できず、無効化 case が
    「未到達確定」と誤診断される（実測で確認済みの欠陥）。None は「観測なし」の標識。

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

    assert (steps, interrupted) == (None, False)


async def test_observe_route_steps_candidate_invalid_returns_invalid_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_run_one` 実行中の `_CandidateInvalid` は catch され `(None, False)` を返す。

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

    assert (steps, interrupted) == (None, False)


# --- _check_route_coverage: 観測失敗時の部分診断保全 -------------------------------------


def _cov_slots(*names: str) -> dict[str, Slot]:
    """`_check_route_coverage` の slots 引数用の最小 Slot mapping。"""
    return {
        name: Slot(
            name=name,
            seed=f"{name} seed",
            build=lambda c, _n=name: AgentSpec(name=_n, instructions=c, model=FakeModel()),
        )
        for name in names
    }


async def _run_check_coverage(
    monkeypatch: pytest.MonkeyPatch,
    *,
    observe: Any,
    train: list[Any],
    slots: dict[str, Slot],
    timeout_seconds: float | None = None,
) -> None:
    """`_observe_route_steps` を差し替えて `_check_route_coverage` を駆動する。"""
    monkeypatch.setattr(
        "oai_agentspec.runtime.lightning._rollout._observe_route_steps", observe, raising=True
    )
    await _check_route_coverage(
        target=HandoffGraph(entry=next(iter(slots))),
        registry=None,
        slots=slots,
        seeds={name: f"{name} seed" for name in slots},
        train=train,
        approvals=None,
        tool_mocks=None,
        context_factory=None,
        timeout_seconds=timeout_seconds,
    )


async def test_check_route_coverage_attaches_partial_report_on_observation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """観測が途中で失敗したら、そこまでの到達を `coverage` に載せて TRAINER_FAILED を送出する。

    支払い済みの API コストで得た部分観測が `logger.warning` の文字列にしか残らないと、
    利用者は except 節からプログラム的に診断できない（案 B の動機がこの経路で満たされない）。
    """
    calls = {"n": 0}

    async def _observe(**kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] < 3:
            return ("triage",), False
        raise RuntimeError("boom")

    with pytest.raises(OptimizeError) as excinfo:
        await _run_check_coverage(
            monkeypatch,
            observe=_observe,
            train=[{"input": "a"}, {"input": "b"}, {"input": "c"}],
            slots=_cov_slots("triage", "billing"),
        )

    exc = excinfo.value
    assert exc.kind is FailureKind.TRAINER_FAILED
    assert exc.coverage is not None
    assert exc.coverage.complete is False
    assert exc.coverage.covered == frozenset({"triage"})
    # 失敗した 3 件目は per_case に入らない（steps が得られていないため）。
    assert len(exc.coverage.per_case) == 2
    assert isinstance(exc.__cause__, RuntimeError)


async def test_check_route_coverage_failure_message_carries_case_and_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """失敗メッセージは case 位置・例外型名・例外本文を含む（既存 assert 資産も満たす）。"""

    async def _observe(**kwargs: Any) -> Any:
        raise RuntimeError("boom")

    with pytest.raises(OptimizeError) as excinfo:
        await _run_check_coverage(
            monkeypatch,
            observe=_observe,
            train=[{"input": "a"}, {"input": "b"}],
            slots=_cov_slots("triage"),
        )

    message = excinfo.value.message
    assert "pre-flight" in message.lower()
    assert "case 1/2" in message
    assert "RuntimeError" in message
    assert "boom" in message


async def test_check_route_coverage_timeout_message_states_limit_and_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """timeout は上限秒と型名をメッセージに残す（`str(TimeoutError())` が空でも読める）。

    型名のあとに空本文を連結すると `'... TimeoutError: '` とコロンで終わり情報がゼロになる。
    本文が空のときは連結しないことを固定する。
    """
    import asyncio

    async def _observe(**kwargs: Any) -> Any:
        await asyncio.sleep(3600)

    with pytest.raises(OptimizeError) as excinfo:
        await _run_check_coverage(
            monkeypatch,
            observe=_observe,
            train=[{"input": "a"}],
            slots=_cov_slots("triage"),
            timeout_seconds=0.01,
        )

    message = excinfo.value.message
    assert "TimeoutError" in message
    assert "0.01" in message
    assert not message.rstrip("\n").splitlines()[0].endswith(": ")
    assert isinstance(excinfo.value.__cause__, TimeoutError)


async def test_check_route_coverage_no_timeout_message_states_unlimited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`timeout_seconds=None`（上限なし）でもメッセージが適用状況を明示する。"""

    async def _observe(**kwargs: Any) -> Any:
        raise RuntimeError("boom")

    with pytest.raises(OptimizeError) as excinfo:
        await _run_check_coverage(
            monkeypatch,
            observe=_observe,
            train=[{"input": "a"}],
            slots=_cov_slots("triage"),
        )

    assert "上限なし" in excinfo.value.message


async def test_check_route_coverage_passes_through_optimize_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """観測中の `OptimizeError` は再ラップしない（kind / message / coverage を保つ）。

    pre-flight は approvals を本番同値で素通しするため、承認安全違反（NFR-8 fail-closed）の
    `CONFIG_MISSING` がループ内から飛びうる。無条件に再ラップすると kind が TRAINER_FAILED へ
    変質し、docs に明記済みの契約が壊れる（回帰 pin）。
    """

    async def _observe(**kwargs: Any) -> Any:
        raise OptimizeError(FailureKind.CONFIG_MISSING, "承認済みツールがモック未差し替えです")

    with pytest.raises(OptimizeError) as excinfo:
        await _run_check_coverage(
            monkeypatch,
            observe=_observe,
            train=[{"input": "a"}],
            slots=_cov_slots("triage"),
        )

    exc = excinfo.value
    assert exc.kind is FailureKind.CONFIG_MISSING
    assert exc.message == "承認済みツールがモック未差し替えです"
    assert exc.coverage is None


async def test_check_route_coverage_partial_report_does_not_leak_case_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """部分レポート経路でも case 本文は message / repr に出さない（PII 方針の維持）。"""

    async def _observe(**kwargs: Any) -> Any:
        raise RuntimeError("boom")

    with pytest.raises(OptimizeError) as excinfo:
        await _run_check_coverage(
            monkeypatch,
            observe=_observe,
            train=[{"input": "SECRET-CASE-MARKER-1"}],
            slots=_cov_slots("triage"),
        )

    exc = excinfo.value
    assert "SECRET-CASE-MARKER-1" not in exc.message
    assert "SECRET-CASE-MARKER-1" not in repr(exc.coverage)


# --- _check_route_coverage: candidate 無効化の誤診断修正（None / () / 非空 の 3 値） ----------


async def test_check_route_coverage_all_invalid_does_not_claim_unreached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全 train が無効化されたとき「一度も routing されなかった」と主張しない。

    rollout は 1 件も観測できておらず、未到達は確定していない。誤った救済策
    （train 追加・edge 見直し）を提示せず、無効化の原因（${var} / vars_fn / マーカー）へ
    誘導する。診断は invalid_cases / per_case=None で構造化される。
    """

    async def _observe(**kwargs: Any) -> Any:
        return None, False

    with pytest.raises(OptimizeError) as excinfo:
        await _run_check_coverage(
            monkeypatch,
            observe=_observe,
            train=[{"input": "a"}, {"input": "b"}],
            slots=_cov_slots("triage", "billing"),
        )

    exc = excinfo.value
    assert exc.kind is FailureKind.CONFIG_MISSING
    assert exc.coverage is not None
    assert exc.coverage.invalid_cases == 2
    assert exc.coverage.complete is True
    assert [steps for _, steps in exc.coverage.per_case] == [None, None]
    assert "一度も routing されませんでした" not in exc.message
    assert "候補無効化" in exc.message
    assert "${var}" in exc.message
    # 誤誘導する救済策を載せない。
    assert "train ケースに未到達 slot を経由する入力を追加する" not in exc.message


async def test_check_route_coverage_partial_invalid_message_separates_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """部分無効化 + 未到達ありでは、観測件数と無効化件数を分離して示す。

    「train 全 N 件の rollout で」という全数主張は無効化があるとき偽になるため、
    観測できた件数の範囲での判定であることを明示する。
    """
    calls = {"n": 0}

    async def _observe(**kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return ("triage",), False
        return None, False

    with pytest.raises(OptimizeError) as excinfo:
        await _run_check_coverage(
            monkeypatch,
            observe=_observe,
            train=[{"input": "a"}, {"input": "b"}],
            slots=_cov_slots("triage", "billing"),
        )

    exc = excinfo.value
    assert exc.kind is FailureKind.CONFIG_MISSING
    assert exc.coverage.invalid_cases == 1
    assert [steps for _, steps in exc.coverage.per_case] == [("triage",), None]
    assert "候補無効化" in exc.message
    assert "1/2" in exc.message
    assert "billing" in exc.message


async def test_check_route_coverage_pass_with_invalid_warns(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """全 slot カバー済みなら無効化があっても fail させず warning で通知する。

    無効化の是正は coverage 検査の責務外（判定に必要な観測は揃っている）。ただし silent に
    しない — seed 状態での無効化は本来起きないはずの構成問題のシグナルのため。
    """
    calls = {"n": 0}

    async def _observe(**kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return ("triage",), False
        return None, False

    with caplog.at_level("WARNING"):
        await _run_check_coverage(
            monkeypatch,
            observe=_observe,
            train=[{"input": "a"}, {"input": "b"}],
            slots=_cov_slots("triage"),
        )

    text = caplog.text
    assert "candidate invalidated at case 2/2" in text
    assert "coverage passed with 1/2 invalidated" in text


async def test_check_route_coverage_executed_empty_is_not_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """実行済み・観測空（`()`）は無効化と区別され invalid_cases に数えない。

    防御的経路（route 構築で steps が空）は「rollout は実行された」事実を持つ。
    無効化（観測なし）と同一視すると診断の 3 値が崩れる。
    """

    async def _observe(**kwargs: Any) -> Any:
        return (), False

    with pytest.raises(OptimizeError) as excinfo:
        await _run_check_coverage(
            monkeypatch,
            observe=_observe,
            train=[{"input": "a"}],
            slots=_cov_slots("triage"),
        )

    exc = excinfo.value
    assert exc.coverage.invalid_cases == 0
    assert [steps for _, steps in exc.coverage.per_case] == [()]
    # 無効化ゼロなら現行の未到達メッセージ（全数主張が真のため）。
    assert "一度も routing されませんでした" in exc.message
    # 「候補無効化」の診断文言が出ないこと（escape hatch の「本検査を無効化する」とは別語）。
    assert "候補無効化" not in exc.message


async def test_check_route_coverage_partial_report_counts_invalid_and_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """観測失敗の部分 report でも invalid_cases が載り、観測完了数は非 None エントリで数える。

    無効化 case を per_case に `None` で積む設計では `len(per_case)` は「観測完了数」でなく
    「記録数」。観測完了の過大計上（無効化の混入）を防ぐ。
    """
    calls = {"n": 0}

    async def _observe(**kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return None, False
        raise RuntimeError("boom")

    with pytest.raises(OptimizeError) as excinfo:
        await _run_check_coverage(
            monkeypatch,
            observe=_observe,
            train=[{"input": "a"}, {"input": "b"}],
            slots=_cov_slots("triage"),
        )

    exc = excinfo.value
    assert exc.kind is FailureKind.TRAINER_FAILED
    assert exc.coverage.complete is False
    assert exc.coverage.invalid_cases == 1
    assert "観測完了 case: 0/2" in exc.message


async def test_check_route_coverage_optimize_error_does_not_log_observation_failed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """承認 fail-closed の `OptimizeError` で「observation failed」warning を出さない。

    承認安全違反（NFR-8）は意図的な fail-closed であって観測の失敗ではない。両方出すと
    「観測が失敗した（error=OptimizeError）」と「CONFIG_MISSING」の矛盾する 2 診断が並ぶ。
    """

    async def _observe(**kwargs: Any) -> Any:
        raise OptimizeError(FailureKind.CONFIG_MISSING, "承認済みツールがモック未差し替えです")

    with caplog.at_level("WARNING"), pytest.raises(OptimizeError):
        await _run_check_coverage(
            monkeypatch,
            observe=_observe,
            train=[{"input": "a"}],
            slots=_cov_slots("triage"),
        )

    assert "observation failed" not in caplog.text
