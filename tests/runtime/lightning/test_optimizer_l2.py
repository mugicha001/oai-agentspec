"""L2: 最適化エントリ `optimize` + 内部オーケストレータを FakeModel + monkeypatch で検証する。

`optimize` は関数内で `from ..._adapters import run_apo` するため monkeypatch 対象は使用箇所パス
`oai_agentspec._adapters.run_apo`。設定不在（algorithm / train / reward / val / config /
apo_client）の CONFIG_MISSING・extra 不在（ImportError → EXTRA_MISSING）・Trainer 失敗
（RuntimeError →
TRAINER_FAILED + cause）・rollout 内遅延 CONFIG_MISSING の kind 保持（TRAINER_FAILED に化けない）・
実 rollout 経路（FakeModel + reward 呼び出し）・安全不変条件（approve したツールが未差し替えで
ValueError・危険ツール非実行）・seed 経路（生 seed + rebind）・vars 再注入・正規化分岐を網羅する。
実 LLM / 実 Trainer は呼ばない（`run_apo` を fake へ差し替えて到達阻止）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from agents.items import ModelResponse
from agents.models.interface import Model

from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph, function_tool
from oai_agentspec.runtime.lightning import (
    FailureKind,
    OptimizeCase,
    OptimizeConfig,
    OptimizeError,
    OptimizeResult,
    RolloutResult,
    Slot,
    contains,
    optimize,
)
from oai_agentspec.runtime.lightning.optimizer import (
    _build_decisions,
    _extract_case_input,
    _normalize_slots,
    _reinject_vars,
    _seeds_of,
)

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.integration


# APO 必須 client の sentinel（fake AsyncOpenAI 互換）。`_build_apo` まで到達させないため、
# `run_apo` を fake へ差し替える経路では中身は使われない（None でなければ十分）。
_FAKE_APO_CLIENT = object()


def _apo_config() -> OptimizeConfig:
    """APO の前提（apo_client）を満たす最小 `OptimizeConfig` を作る。"""
    return OptimizeConfig(apo_client=_FAKE_APO_CLIENT)


# APO 必須引数（val / config）を補う既定値（rollout 経路では fake run_apo が val も rollout する）。
_DEFAULT_VAL: list[dict[str, Any]] = [{"input": "v", "expected": "expected"}]


class _RepeatModel(Model):
    """常に同一テキストを返すモデル（複数 rollout で枯渇しない・実 LLM 非依存）。

    `FakeModel().queue_text(...)` はキューを消費するため train + val の複数 rollout で 2 回目に
    空テキストへ落ちる。reward を rollout 横断で安定させるため、入力に依らず固定テキストを返す。
    """

    def __init__(self, text: str) -> None:
        """固定応答テキストを設定する。"""
        self._text = text

    async def get_response(
        self,
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        """固定テキスト応答を返す。"""
        from _helpers.responses import text_response

        return text_response(self._text)

    async def stream_response(  # type: ignore[override]
        self,
        system_instructions: str | None = None,
        input: Any = None,  # noqa: A002 - SDK シグネチャに追従
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """未使用（非ストリーミング）。"""
        raise NotImplementedError("_RepeatModel はストリーミング非対応")
        yield  # pragma: no cover - 到達しない（型のため）


def _spec(
    name: str = "bot",
    *,
    instructions: str = "be helpful",
    tools: list[Any] | None = None,
    output_text: str = "hello expected world",
) -> AgentSpec:
    """固定テキストを返すモデルを据えた AgentSpec を作る（複数 rollout で枯渇しない）。"""
    return AgentSpec(
        name=name,
        instructions=instructions,
        model=_RepeatModel(output_text),
        tools=list(tools or []),
    )


def _patch_run_apo(monkeypatch: pytest.MonkeyPatch, fake: Any) -> None:
    """使用箇所パス `oai_agentspec._adapters.run_apo` を fake へ差し替える。"""
    monkeypatch.setattr("oai_agentspec._adapters.run_apo", fake, raising=True)


def _calling_run_apo() -> Any:
    """渡された rollout を train 各ケースに実際に適用し平均を返す薄い fake run_apo。

    実 Trainer / agentlightning を呼ばずに、rollout（=候補適用 + reward）が確実に駆動されることを
    検証するための fake。最良候補 = seeds（1 ラウンド評価）として OptimizeResult を組む。
    """

    async def _fake(
        *,
        seeds: dict[str, str],
        train: Any,
        val: Any,
        rollout: Any,
        config: Any,
        vars_per_slot: Any = None,  # noqa: ARG001 - run_apo の新パラメータに合わせる
    ) -> OptimizeResult:
        candidate = dict(seeds)
        train_list = list(train)
        total = 0.0
        for case in train_list:
            total += float(await rollout(dict(candidate), case))
        train_score = total / len(train_list) if train_list else 0.0
        val_score: float | None = None
        if val is not None:
            val_list = list(val)
            v_total = 0.0
            for case in val_list:
                v_total += float(await rollout(dict(candidate), case))
            val_score = v_total / len(val_list) if val_list else 0.0
        prompt: str | dict[str, str] = (
            candidate[next(iter(candidate))] if len(candidate) == 1 else dict(candidate)
        )
        return OptimizeResult(prompt=prompt, train_score=train_score, val_score=val_score)

    return _fake


# ----------------------------------------------------------------------
# 設定不在（CONFIG_MISSING）: run_apo へ到達せず即エラー
# ----------------------------------------------------------------------


async def test_algorithm_rl_raises_config_missing() -> None:
    """algorithm='rl' は別 extra 案内付きの CONFIG_MISSING。"""
    with pytest.raises(OptimizeError) as exc:
        await optimize(_spec(), algorithm="rl", train=[{"input": "x"}], reward=contains("e"))
    assert exc.value.kind == FailureKind.CONFIG_MISSING


async def test_unknown_algorithm_raises_config_missing() -> None:
    """未対応 algorithm は CONFIG_MISSING。"""
    with pytest.raises(OptimizeError) as exc:
        await optimize(_spec(), algorithm="bogus", train=[{"input": "x"}], reward=contains("e"))
    assert exc.value.kind == FailureKind.CONFIG_MISSING


async def test_empty_train_raises_config_missing() -> None:
    """train が空なら CONFIG_MISSING。"""
    with pytest.raises(OptimizeError) as exc:
        await optimize(_spec(), algorithm="apo", train=[], reward=contains("e"))
    assert exc.value.kind == FailureKind.CONFIG_MISSING


async def test_none_reward_raises_config_missing() -> None:
    """reward=None は CONFIG_MISSING。"""
    with pytest.raises(OptimizeError) as exc:
        await optimize(
            _spec(),
            algorithm="apo",
            train=[{"input": "x"}],
            reward=None,  # type: ignore[arg-type]
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


async def test_raw_seed_without_rebind_raises_config_missing() -> None:
    """生 seed（str slot）で rebind 未指定なら CONFIG_MISSING（rollout に到達せず）。"""
    with pytest.raises(OptimizeError) as exc:
        await optimize(
            HandoffGraph(entry="t"),
            algorithm="apo",
            train=[{"input": "x"}],
            val=_DEFAULT_VAL,
            reward=contains("e"),
            slot="raw seed text",  # 生 seed
            config=_apo_config(),
            # rebind を渡さない。
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


async def test_mixed_slot_dict_raises_config_missing() -> None:
    """slot dict に Slot と生 seed(str) が混在すると CONFIG_MISSING（fail-closed）。"""
    slot = Slot(name="a", seed="s", build=lambda c: c)
    with pytest.raises(OptimizeError) as exc:
        await optimize(
            HandoffGraph(entry="t"),
            algorithm="apo",
            train=[{"input": "x"}],
            val=_DEFAULT_VAL,
            reward=contains("e"),
            slot={"a": slot, "b": "raw"},  # type: ignore[dict-item]
            config=_apo_config(),
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


async def test_empty_slot_dict_raises_config_missing() -> None:
    """slot={} は最適化対象が無い不正設定で CONFIG_MISSING（prompt_slots(agents=[]) 等を阻止）。

    空 mapping を許すと prompt={} で誤った成功になるため fail-closed する（Codex P2 回帰防止）。
    """
    with pytest.raises(OptimizeError) as exc:
        await optimize(
            _spec(),
            algorithm="apo",
            train=[{"input": "x"}],
            val=_DEFAULT_VAL,
            reward=contains("e"),
            slot={},
            config=_apo_config(),
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


async def test_agentspec_target_with_mismatched_slot_name_raises_config_missing() -> None:
    """target=AgentSpec(name='A') に slot.name='B' を渡すと CONFIG_MISSING（fail-closed）。

    `_apply_candidate` の AgentSpec 分岐は `next(iter(slots.values()))` で 1 件目を取り、その
    `slot.name` で registry の spec を resolve する。target と異なる slot 名を許すと、利用者が
    target に渡した spec ではなく registry の別 agent が黙って最適化されるため、整合性が崩れる。
    """
    target = _spec(name="A")
    slot = Slot(name="B", seed="s", build=lambda c: c)
    with pytest.raises(OptimizeError) as exc:
        await optimize(
            target,
            algorithm="apo",
            train=[{"input": "x"}],
            val=_DEFAULT_VAL,
            reward=contains("e"),
            slot=slot,
            config=_apo_config(),
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


async def test_slot_dict_key_mismatched_with_slot_name_raises_config_missing() -> None:
    """slot dict のキーと `Slot.name` が一致しないと CONFIG_MISSING で fail-closed
    （Codex P2: KeyError や wrong agent build を防ぐ）。"""
    slot_a = Slot(name="A", seed="s", build=lambda c: c)
    with pytest.raises(OptimizeError) as exc:
        await optimize(
            HandoffGraph(entry="t"),
            algorithm="apo",
            train=[{"input": "x"}],
            val=_DEFAULT_VAL,
            reward=contains("e"),
            # dict key="other_key" だが Slot.name="A" でズレている。
            slot={"other_key": slot_a},
            config=_apo_config(),
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


async def test_agentspec_target_with_multiple_slots_raises_config_missing() -> None:
    """target=AgentSpec(name='A') に slot={'A': ..., 'B': ...} の multi-slot を渡すと
    CONFIG_MISSING（Codex P2: `next(iter(slots.values()))` の辞書順序依存で wrong agent が
    最適化されたり余剰スロットが silent に無視されたりするのを fail-closed で防ぐ）。"""
    target = _spec(name="A")
    slot_a = Slot(name="A", seed="s", build=lambda c: c)
    slot_b = Slot(name="B", seed="s", build=lambda c: c)
    with pytest.raises(OptimizeError) as exc:
        await optimize(
            target,
            algorithm="apo",
            train=[{"input": "x"}],
            val=_DEFAULT_VAL,
            reward=contains("e"),
            slot={"A": slot_a, "B": slot_b},
            config=_apo_config(),
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


async def test_agentspec_target_with_mismatched_slot_dict_keys_raises_config_missing() -> None:
    """target=AgentSpec(name='A') に slot={'B': Slot('B'), ...} を渡すと CONFIG_MISSING。"""
    target = _spec(name="A")
    slot_b = Slot(name="B", seed="s", build=lambda c: c)
    with pytest.raises(OptimizeError) as exc:
        await optimize(
            target,
            algorithm="apo",
            train=[{"input": "x"}],
            val=_DEFAULT_VAL,
            reward=contains("e"),
            slot={"B": slot_b},
            config=_apo_config(),
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


async def test_algorithm_defaults_to_apo(monkeypatch: pytest.MonkeyPatch) -> None:
    """algorithm 引数を省略すると既定値 "apo" で受理され rollout が回る（最小ケース）。"""
    _patch_run_apo(monkeypatch, _calling_run_apo())

    result = await optimize(
        _spec(output_text="the expected answer"),
        # algorithm を渡さない（既定 "apo" にフォールバック）。
        train=[{"input": "hi", "expected": "expected"}],
        val=_DEFAULT_VAL,
        reward=contains("expected"),
        config=_apo_config(),
    )
    assert isinstance(result, OptimizeResult)
    assert result.train_score == pytest.approx(1.0)


async def test_direct_apo_client_kwarg_drives_optimize(monkeypatch: pytest.MonkeyPatch) -> None:
    """`config=` を渡さず `apo_client=` 直接渡しでも APO 設定が成立して rollout が回る。"""
    _patch_run_apo(monkeypatch, _calling_run_apo())

    result = await optimize(
        _spec(output_text="the expected answer"),
        train=[{"input": "hi", "expected": "expected"}],
        val=_DEFAULT_VAL,
        reward=contains("expected"),
        apo_client=_FAKE_APO_CLIENT,  # 直接 kwargs（最小ケース）。
        rounds=2,
    )
    assert isinstance(result, OptimizeResult)
    assert result.train_score == pytest.approx(1.0)


async def test_direct_tracer_kwarg_flows_into_effective_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """直接 kwargs で渡した `tracer=` は最終 `OptimizeConfig.tracer` まで到達する。"""
    captured: dict[str, Any] = {}

    async def _fake(
        *,
        seeds: dict[str, str],
        train: Any,
        val: Any,
        rollout: Any,
        config: Any,
        vars_per_slot: Any = None,  # noqa: ARG001 - run_apo の新パラメータに合わせる
    ) -> OptimizeResult:
        captured["config"] = config
        return OptimizeResult(prompt=seeds[next(iter(seeds))], train_score=0.0, val_score=0.0)

    _patch_run_apo(monkeypatch, _fake)

    sentinel_tracer = object()
    await optimize(
        _spec(),
        train=[{"input": "x"}],
        val=_DEFAULT_VAL,
        reward=contains("e"),
        apo_client=_FAKE_APO_CLIENT,
        tracer=sentinel_tracer,  # 直接 kwargs
    )
    assert captured["config"].tracer is sentinel_tracer


async def test_config_and_direct_kwargs_both_specified_raises() -> None:
    """`config=` と直接 kwargs（apo_client 等）を同時指定すると CONFIG_MISSING（曖昧禁止）。"""
    with pytest.raises(OptimizeError) as exc:
        await optimize(
            _spec(),
            train=[{"input": "x"}],
            val=_DEFAULT_VAL,
            reward=contains("e"),
            config=_apo_config(),
            apo_client=_FAKE_APO_CLIENT,  # 同時指定で曖昧。
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING
    assert "同時に指定" in str(exc.value)


async def test_direct_kwargs_only_without_apo_client_raises() -> None:
    """直接 kwargs で apo_client を渡さないと CONFIG_MISSING（必須・案内文に直接渡し言及）。"""
    with pytest.raises(OptimizeError) as exc:
        await optimize(
            _spec(),
            train=[{"input": "x"}],
            val=_DEFAULT_VAL,
            reward=contains("e"),
            rounds=2,  # apo_client は未指定。
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING
    assert "apo_client" in str(exc.value)


async def test_handoff_graph_with_rebind_without_slot_raises_config_missing() -> None:
    """HandoffGraph + rebind 指定だが slot 未指定は CONFIG_MISSING（Codex P2 回帰防止）。

    `_normalize_slots` が None を返し（target は AgentSpec 以外）、rebind が在るので rebind 必須
    チェックは通るが、`_seeds_of(None, None)` は `{}` を返す。空 seeds で run_apo に到達すると
    Trainer 側で obscure に落ちるため、optimize 側で fail-closed する。
    """
    graph = HandoffGraph(entry="t")
    with pytest.raises(OptimizeError) as exc:
        await optimize(
            graph,
            algorithm="apo",
            train=[{"input": "x"}],
            val=_DEFAULT_VAL,
            reward=contains("e"),
            rebind=lambda candidate: graph,  # rebind だけ与える（slot は未指定）
            config=_apo_config(),
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING
    assert "slot=" in str(exc.value)


async def test_val_none_raises_config_missing() -> None:
    """val=None は CONFIG_MISSING（APO は検証用ケース列必須・新契約）。"""
    with pytest.raises(OptimizeError) as exc:
        await optimize(
            _spec(),
            algorithm="apo",
            train=[{"input": "x"}],
            val=None,
            reward=contains("e"),
            config=_apo_config(),
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


async def test_val_empty_raises_config_missing() -> None:
    """val=[] は CONFIG_MISSING（空 val は不可・新契約）。"""
    with pytest.raises(OptimizeError) as exc:
        await optimize(
            _spec(),
            algorithm="apo",
            train=[{"input": "x"}],
            val=[],
            reward=contains("e"),
            config=_apo_config(),
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


async def test_config_none_raises_config_missing() -> None:
    """config=None は CONFIG_MISSING（apo_client 必須のため・新契約）。"""
    with pytest.raises(OptimizeError) as exc:
        await optimize(
            _spec(),
            algorithm="apo",
            train=[{"input": "x"}],
            val=_DEFAULT_VAL,
            reward=contains("e"),
            config=None,
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


async def test_config_without_apo_client_raises_config_missing() -> None:
    """OptimizeConfig.apo_client=None は CONFIG_MISSING（APO は AsyncOpenAI 必須・新契約）。"""
    with pytest.raises(OptimizeError) as exc:
        await optimize(
            _spec(),
            algorithm="apo",
            train=[{"input": "x"}],
            val=_DEFAULT_VAL,
            reward=contains("e"),
            config=OptimizeConfig(apo_client=None),
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


# ----------------------------------------------------------------------
# extra 不在 / Trainer 失敗（run_apo の例外 → 構造化エラーへ変換）
# ----------------------------------------------------------------------


async def test_import_error_maps_to_extra_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_apo が ImportError を送出すると EXTRA_MISSING へ変換され cause がチェーンされる。"""

    async def _raise_import(**kwargs: Any) -> Any:
        raise ImportError("agentlightning が必要です")

    _patch_run_apo(monkeypatch, _raise_import)

    with pytest.raises(OptimizeError) as exc:
        await optimize(
            _spec(),
            algorithm="apo",
            train=[{"input": "x"}],
            val=_DEFAULT_VAL,
            reward=contains("e"),
            config=_apo_config(),
        )
    assert exc.value.kind == FailureKind.EXTRA_MISSING
    assert isinstance(exc.value.__cause__, ImportError)


async def test_runtime_error_maps_to_trainer_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_apo の RuntimeError は TRAINER_FAILED へ変換され cause がチェーンされる。"""

    async def _raise_runtime(**kwargs: Any) -> Any:
        raise RuntimeError("trainer exploded")

    _patch_run_apo(monkeypatch, _raise_runtime)

    with pytest.raises(OptimizeError) as exc:
        await optimize(
            _spec(),
            algorithm="apo",
            train=[{"input": "x"}],
            val=_DEFAULT_VAL,
            reward=contains("e"),
            config=_apo_config(),
        )
    assert exc.value.kind == FailureKind.TRAINER_FAILED
    assert isinstance(exc.value.__cause__, RuntimeError)


async def test_rollout_config_missing_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rollout 内で遅延送出された CONFIG_MISSING は kind を保つ（TRAINER_FAILED に化けない・FR-8）。

    グラフ最適化で registry=None を渡すと `_apply_candidate` が rollout 内で
    OptimizeError(CONFIG_MISSING) を送出する。run_apo を「rollout を実際に呼ぶ fake」にすると、
    optimize の `except OptimizeError: raise` で CONFIG_MISSING のまま伝播する（BLOCKER 回帰防止）。
    """

    async def _call_rollout(
        *,
        seeds: Any,
        train: Any,
        val: Any,
        rollout: Any,
        config: Any,
        vars_per_slot: Any = None,  # noqa: ARG001 - run_apo の新パラメータに合わせる
    ) -> Any:
        # train 先頭ケースで rollout を呼ぶ（内部で _apply_candidate が走る）。
        await rollout(dict(seeds), list(train)[0])
        return None  # 到達しない（rollout が送出する）。

    _patch_run_apo(monkeypatch, _call_rollout)

    slot = Slot(name="triage", seed="seed", build=lambda c: _spec(name="triage"))
    graph = HandoffGraph(entry="triage")

    with pytest.raises(OptimizeError) as exc:
        await optimize(
            graph,
            algorithm="apo",
            train=[{"input": "x"}],
            val=_DEFAULT_VAL,
            reward=contains("e"),
            slot=slot,
            registry=None,  # グラフ最適化に必須の registry を渡さない → rollout 内で遅延検知。
            config=_apo_config(),
        )
    # TRAINER_FAILED に化けず CONFIG_MISSING のまま。
    assert exc.value.kind == FailureKind.CONFIG_MISSING


# ----------------------------------------------------------------------
# 実 rollout 経路（FakeModel + reward 駆動）
# ----------------------------------------------------------------------


async def test_static_agent_spec_default_slot_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    """静的 AgentSpec（slot=None 既定スロット）で rollout が回り reward が呼ばれ score が返る。"""
    _patch_run_apo(monkeypatch, _calling_run_apo())

    result = await optimize(
        _spec(output_text="the expected answer"),
        algorithm="apo",
        train=[{"input": "hi", "expected": "expected"}],
        val=[{"input": "v", "expected": "expected"}],
        reward=contains("expected"),
        config=_apo_config(),
    )
    assert isinstance(result, OptimizeResult)
    # FakeModel 出力に "expected" を含むため reward=1.0 → train_score=1.0。
    assert result.train_score == pytest.approx(1.0)
    # val 指定 → val_score も 1.0。
    assert result.val_score == pytest.approx(1.0)
    # 既定スロットの seed = instructions が prompt として返る。
    assert result.prompt == "be helpful"


async def test_rollout_reward_miss_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """期待文字列が出力に無ければ reward=0.0 が train_score に反映される。"""
    _patch_run_apo(monkeypatch, _calling_run_apo())

    result = await optimize(
        _spec(output_text="totally unrelated"),
        algorithm="apo",
        train=[{"input": "hi", "expected": "nowhere"}],
        val=_DEFAULT_VAL,
        reward=contains("expected"),
        config=_apo_config(),
    )
    assert result.train_score == pytest.approx(0.0)


async def test_rollout_with_val_computes_val_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """val 指定時は val_score が算出される（fake run_apo が val を rollout する）。"""
    _patch_run_apo(monkeypatch, _calling_run_apo())

    result = await optimize(
        _spec(output_text="the expected answer"),
        algorithm="apo",
        train=[{"input": "hi", "expected": "expected"}],
        val=[{"input": "v", "expected": "expected"}],
        reward=contains("expected"),
        config=_apo_config(),
    )
    assert result.val_score == pytest.approx(1.0)


async def test_async_reward_is_awaited(monkeypatch: pytest.MonkeyPatch) -> None:
    """async reward は await されて score が返る（同期 / async 両対応）。"""
    _patch_run_apo(monkeypatch, _calling_run_apo())

    async def _async_reward(result: RolloutResult) -> float:
        return 0.5

    result = await optimize(
        _spec(),
        algorithm="apo",
        train=[{"input": "hi"}],
        val=_DEFAULT_VAL,
        reward=_async_reward,
        config=_apo_config(),
    )
    assert result.train_score == pytest.approx(0.5)


async def test_rollout_passes_observed_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """rollout が RolloutResult.case / output を reward へ渡す（plain 観測の受け渡し）。"""
    _patch_run_apo(monkeypatch, _calling_run_apo())
    captured: list[dict[str, Any]] = []

    def _reward(result: RolloutResult) -> float:
        captured.append(
            {"case": result.case, "output": result.output, "tool_calls": result.tool_calls}
        )
        return 1.0

    await optimize(
        _spec(output_text="answer-text"),
        algorithm="apo",
        train=[{"input": "hi", "id": 7}],
        val=_DEFAULT_VAL,
        reward=_reward,
        config=_apo_config(),
    )
    # train / val 両方で rollout される。train の id=7 ケースが含まれていることを確認。
    train_cases = [c for c in captured if c["case"].get("id") == 7]
    assert len(train_cases) == 1
    assert train_cases[0]["case"] == {"input": "hi", "id": 7}
    assert train_cases[0]["output"] == "answer-text"
    assert isinstance(train_cases[0]["tool_calls"], list)


async def test_rollout_passes_route_steps_and_last_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_make_rollout` が observation.route を route_steps / last_agent に詰めて reward へ渡す。"""
    _patch_run_apo(monkeypatch, _calling_run_apo())

    captured: list[RolloutResult] = []

    def _reward(result: RolloutResult) -> float:
        captured.append(result)
        return 0.0

    # 単体 AgentSpec → route_steps=["bot"]・last_agent="bot"（_RepeatModel が即応答）。
    await optimize(
        _spec(name="bot", output_text="answer"),
        algorithm="apo",
        train=[{"input": "hi"}],
        val=_DEFAULT_VAL,
        reward=_reward,
        config=_apo_config(),
    )
    assert captured, "rollout reward が呼ばれていない"
    train_case = next((c for c in captured if c.case.get("input") == "hi"), None)
    assert train_case is not None
    assert train_case.route_steps == ["bot"]
    assert train_case.last_agent == "bot"


# ----------------------------------------------------------------------
# 生 seed + rebind 経路
# ----------------------------------------------------------------------


async def test_raw_seed_with_rebind_drives_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    """生 seed（str）+ rebind 経路で rebind が候補を受け取り宣言物を組み直す。"""
    _patch_run_apo(monkeypatch, _calling_run_apo())
    seen: dict[str, Any] = {}

    def _rebind(candidate: Any) -> AgentSpec:
        seen["candidate"] = candidate
        return _spec(output_text="rebound expected output")

    result = await optimize(
        _spec(),  # 元 target は AgentSpec だが slot=str + rebind で生 seed 経路。
        algorithm="apo",
        train=[{"input": "hi", "expected": "expected"}],
        val=_DEFAULT_VAL,
        reward=contains("expected"),
        slot="raw seed prompt",
        rebind=_rebind,
        config=_apo_config(),
    )
    # 生 seed が候補として rebind に渡る（fake run_apo は seeds をそのまま候補にする）。
    assert seen["candidate"] == "raw seed prompt"
    assert result.train_score == pytest.approx(1.0)
    # 単一スロットなので prompt は str。
    assert result.prompt == "raw seed prompt"


# ----------------------------------------------------------------------
# 横断（HandoffGraph）最適化の rollout 経路（_target.normalize / _mocked_registry / clone）
# ----------------------------------------------------------------------


async def test_graph_optimization_with_slots_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    """HandoffGraph + Slot mapping + registry で rollout が回り reward score が返る（横断経路）。"""
    _patch_run_apo(monkeypatch, _calling_run_apo())

    reg = AgentRegistry()
    reg.register(_spec(name="triage", output_text="triage expected output"))
    reg.register(_spec(name="billing", output_text="billing output"))
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "billing")

    # triage のみ最適化対象スロット（既定 build = 登録 spec 複製で instructions 差し替え）。
    triage_slot = Slot(
        name="triage",
        seed="triage seed",
        build=lambda c: _spec(name="triage", instructions=c, output_text="triage expected output"),
    )

    result = await optimize(
        graph,
        algorithm="apo",
        train=[{"input": "route me", "expected": "expected"}],
        val=_DEFAULT_VAL,
        reward=contains("expected"),
        slot=triage_slot,
        registry=reg,
        config=_apo_config(),
    )
    assert isinstance(result, OptimizeResult)
    assert result.train_score == pytest.approx(1.0)
    assert result.prompt == "triage seed"


async def test_graph_optimization_with_tool_mocks_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HandoffGraph + tool_mocks で clone 経由の rollout が回り、元 registry が不変に保たれる。"""
    _patch_run_apo(monkeypatch, _calling_run_apo())

    @function_tool(name_override="danger", needs_approval=True)
    def _danger(x: str) -> str:
        """承認必須ツール（テスト用・下流側）。"""
        return f"real:{x}"

    reg = AgentRegistry()
    reg.register(_spec(name="triage", output_text="triage expected"))
    reg.register(_spec(name="ops", tools=[_danger], output_text="ops output"))
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "ops")

    triage_handoffs_before = list(reg._specs["triage"].handoffs)  # noqa: SLF001
    triage_slot = Slot(
        name="triage",
        seed="triage seed",
        build=lambda c: _spec(name="triage", instructions=c, output_text="triage expected"),
    )

    result = await optimize(
        graph,
        algorithm="apo",
        train=[{"input": "go", "expected": "expected"}],
        val=_DEFAULT_VAL,
        reward=contains("expected"),
        slot=triage_slot,
        registry=reg,
        tool_mocks={"ops": {"danger": "mocked"}},
        config=_apo_config(),
    )
    assert result.train_score == pytest.approx(1.0)
    # 利用者 registry の entry spec の handoffs は最適化前後で不変（clone 経路）。
    assert reg._specs["triage"].handoffs == triage_handoffs_before  # noqa: SLF001


# ----------------------------------------------------------------------
# 安全不変条件（NFR-8）: approve したツールが未差し替えなら ValueError（危険ツール非実行）
# ----------------------------------------------------------------------


async def test_approve_without_tool_mock_raises_and_halts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """approve したツールが実差し替え集合に無ければ ValueError で停止する（危険ツール非実行）。

    needs_approval=True の function_tool を持つ spec で tool_mocks を渡さず approve すると、
    `_build_decisions` の安全不変条件で ValueError になり、危険ツールが実行される前に停止する。
    """
    executed: list[str] = []

    @function_tool(name_override="danger", needs_approval=True)
    def _danger(x: str) -> str:
        """本物の危険ツール（呼ばれてはならない）。"""
        executed.append(x)
        return f"real:{x}"

    spec = AgentSpec(
        name="bot",
        instructions="use the tool",
        model=FakeModel().queue_tool_call("danger", '{"x": "v"}'),
        tools=[_danger],
    )

    async def _call_rollout(
        *,
        seeds: Any,
        train: Any,
        val: Any,
        rollout: Any,
        config: Any,
        vars_per_slot: Any = None,  # noqa: ARG001 - run_apo の新パラメータに合わせる
    ) -> Any:
        await rollout(dict(seeds), list(train)[0])
        return None  # 到達しない。

    _patch_run_apo(monkeypatch, _call_rollout)

    with pytest.raises(OptimizeError) as exc:
        await optimize(
            spec,
            algorithm="apo",
            train=[{"input": "do it"}],
            val=_DEFAULT_VAL,
            reward=contains("x"),
            approvals=lambda p: True,  # approve するが tool_mocks を渡さない。
            # tool_mocks を渡さない → 差し替え集合は空 → approve は認可されない。
            config=_apo_config(),
        )
    # `_build_decisions` の安全違反は `OptimizeError(CONFIG_MISSING)` で送出され、`optimize` の
    # `except OptimizeError: raise` で kind を保持したまま伝搬する（TRAINER_FAILED へ化けない・
    # FR-8 / NFR-8 整合）。
    assert exc.value.kind == FailureKind.CONFIG_MISSING
    # 本物の危険ツールは実行されていない（fail-closed で停止）。
    assert executed == []


# ----------------------------------------------------------------------
# 承認自動解決ループ（_run_one）と vars 喪失の fail-closed（rollout return 0.0）
# ----------------------------------------------------------------------


async def test_mock_approve_resolves_and_scores_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """approvals で approve + tool_mocks 指定なら中断を承認自動解決し完了出力を採点する。"""
    from types import SimpleNamespace

    from oai_agentspec.runtime.llmops import ObservedRoute, ObservedRun

    _patch_run_apo(monkeypatch, _calling_run_apo())
    observation = ObservedRun(route=ObservedRoute(steps=[], last_agent="bot"), tool_calls=[])
    resumed = {"count": 0}

    async def _interrupted(self: Any, agent: Any, value: Any, **kwargs: Any) -> Any:
        outcome = SimpleNamespace(
            final_output=None,
            interrupted=True,
            pending=[{"tool_name": "danger", "call_id": "c1", "agent_name": "bot"}],
            state=object(),
        )
        return outcome, observation

    async def _resume(agent: Any, state: Any, **kwargs: Any) -> Any:
        resumed["count"] += 1
        outcome = SimpleNamespace(
            final_output="expected final", interrupted=False, pending=[], state=None
        )
        return outcome, observation

    monkeypatch.setattr(
        "oai_agentspec._adapters.DefaultRunnerAdapter.run_with_observation",
        _interrupted,
        raising=True,
    )
    monkeypatch.setattr("oai_agentspec._adapters.resume_with_observation", _resume, raising=True)
    monkeypatch.setattr(
        "oai_agentspec._adapters.apply_approvals",
        lambda state, decisions: SimpleNamespace(
            applied=[d["call_id"] for d in decisions], unknown=[], already_resolved=[]
        ),
        raising=True,
    )

    @function_tool(name_override="danger", needs_approval=True)
    def _danger(x: str) -> str:
        """承認必須ツール（テスト用）。"""
        return f"real:{x}"

    spec = AgentSpec(name="bot", instructions="i", model=FakeModel(), tools=[_danger])

    result = await optimize(
        spec,
        algorithm="apo",
        train=[{"input": "hi", "expected": "expected"}],
        val=_DEFAULT_VAL,
        reward=contains("expected"),
        approvals=lambda p: True,
        tool_mocks={"bot": {"danger": "mocked"}},
        config=_apo_config(),
    )
    # 承認自動解決して完了出力 "expected final" を採点 → reward=1.0。
    # train と val 両方で rollout が走るため resume は 2 回呼ばれる。
    assert resumed["count"] == 2
    assert result.train_score == pytest.approx(1.0)


async def test_mock_approve_merges_resumed_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """承認 resume 後の segment で観測されたツール呼び出しは tool_calls にマージされる。

    承認前 segment は空 / 承認後 segment に "danger" を観測 → reward(tool_match) が recall で 1.0。
    マージしないと resume 後のツールが見えず 0.0 になる（Codex P2 回帰防止）。
    """
    from types import SimpleNamespace

    from oai_agentspec.runtime.lightning import tool_match
    from oai_agentspec.runtime.llmops import ObservedRoute, ObservedRun, ObservedToolCall

    _patch_run_apo(monkeypatch, _calling_run_apo())
    pre_observation = ObservedRun(route=ObservedRoute(steps=[], last_agent="bot"), tool_calls=[])
    post_observation = ObservedRun(
        route=ObservedRoute(steps=[], last_agent="bot"),
        tool_calls=[ObservedToolCall(tool="danger")],
    )

    async def _interrupted(self: Any, agent: Any, value: Any, **kwargs: Any) -> Any:
        outcome = SimpleNamespace(
            final_output=None,
            interrupted=True,
            pending=[{"tool_name": "danger", "call_id": "c1", "agent_name": "bot"}],
            state=object(),
        )
        return outcome, pre_observation

    async def _resume(agent: Any, state: Any, **kwargs: Any) -> Any:
        outcome = SimpleNamespace(final_output="done", interrupted=False, pending=[], state=None)
        return outcome, post_observation

    monkeypatch.setattr(
        "oai_agentspec._adapters.DefaultRunnerAdapter.run_with_observation",
        _interrupted,
        raising=True,
    )
    monkeypatch.setattr("oai_agentspec._adapters.resume_with_observation", _resume, raising=True)
    monkeypatch.setattr(
        "oai_agentspec._adapters.apply_approvals",
        lambda state, decisions: SimpleNamespace(
            applied=[d["call_id"] for d in decisions], unknown=[], already_resolved=[]
        ),
        raising=True,
    )

    @function_tool(name_override="danger", needs_approval=True)
    def _danger(x: str) -> str:
        """承認必須ツール（テスト用）。"""
        return f"real:{x}"

    captured: list[RolloutResult] = []

    def _capture_reward(r: RolloutResult) -> float:
        captured.append(r)
        return 1.0 if "danger" in r.tool_calls else 0.0

    spec = AgentSpec(name="bot", instructions="i", model=FakeModel(), tools=[_danger])

    result = await optimize(
        spec,
        algorithm="apo",
        train=[{"input": "delete u-1", "expected_tools": ["danger"]}],
        val=_DEFAULT_VAL,
        reward=_capture_reward,
        approvals=lambda p: True,
        tool_mocks={"bot": {"danger": "mocked"}},
        config=_apo_config(),
    )
    # resume 後の "danger" 呼び出しが tool_calls にマージされている。
    assert any("danger" in r.tool_calls for r in captured)
    # reward は tool_match 相当 → 1.0。
    assert result.train_score == pytest.approx(1.0)
    # 改めて tool_match ファクトリでも recall 評価できる。
    assert tool_match("expected_tools")(captured[0]) == pytest.approx(1.0)


async def test_mock_approve_merges_resumed_route_and_last_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """承認 resume 後の segment の route_steps / last_agent もマージされる（Codex P2 回帰防止）。

    承認前 segment は `["triage"]` で last_agent="triage" → 承認後 segment が `["billing"]` で
    last_agent="billing"（triage が billing にハンドオフ）。route_match / last_agent_match の reward
    が **resume 後の最終状態**で採点されないと、承認フローを伴うグラフ最適化で誤った reward
    に倒れる（Codex P2 回帰防止）。
    """
    from types import SimpleNamespace

    from oai_agentspec.runtime.lightning import last_agent_match, route_match
    from oai_agentspec.runtime.llmops import ObservedRoute, ObservedRun, RouteStep

    _patch_run_apo(monkeypatch, _calling_run_apo())
    pre_observation = ObservedRun(
        route=ObservedRoute(
            steps=[RouteStep(agent="triage", handoff_from=None)],
            last_agent="triage",
        ),
        tool_calls=[],
    )
    post_observation = ObservedRun(
        route=ObservedRoute(
            steps=[RouteStep(agent="billing", handoff_from="triage")],
            last_agent="billing",
        ),
        tool_calls=[],
    )

    async def _interrupted(self: Any, agent: Any, value: Any, **kwargs: Any) -> Any:
        outcome = SimpleNamespace(
            final_output=None,
            interrupted=True,
            pending=[{"tool_name": "danger", "call_id": "c1", "agent_name": "triage"}],
            state=object(),
        )
        return outcome, pre_observation

    async def _resume(agent: Any, state: Any, **kwargs: Any) -> Any:
        outcome = SimpleNamespace(final_output="done", interrupted=False, pending=[], state=None)
        return outcome, post_observation

    monkeypatch.setattr(
        "oai_agentspec._adapters.DefaultRunnerAdapter.run_with_observation",
        _interrupted,
        raising=True,
    )
    monkeypatch.setattr("oai_agentspec._adapters.resume_with_observation", _resume, raising=True)
    monkeypatch.setattr(
        "oai_agentspec._adapters.apply_approvals",
        lambda state, decisions: SimpleNamespace(
            applied=[d["call_id"] for d in decisions], unknown=[], already_resolved=[]
        ),
        raising=True,
    )

    @function_tool(name_override="danger", needs_approval=True)
    def _danger(x: str) -> str:
        """承認必須ツール（テスト用）。"""
        return f"real:{x}"

    captured: list[RolloutResult] = []

    def _capture(r: RolloutResult) -> float:
        captured.append(r)
        return 0.0

    spec = AgentSpec(name="triage", instructions="i", model=FakeModel(), tools=[_danger])

    await optimize(
        spec,
        algorithm="apo",
        train=[
            {
                "input": "x",
                "expected_route": ["triage", "billing"],
                "expected_last_agent": "billing",
            }
        ],
        val=_DEFAULT_VAL,
        reward=_capture,
        approvals=lambda p: True,
        tool_mocks={"triage": {"danger": "mocked"}},
        config=_apo_config(),
    )
    # resume 後の route_steps / last_agent が反映されている。
    train_observation = captured[0]
    assert train_observation.route_steps == ["triage", "billing"]
    assert train_observation.last_agent == "billing"
    # route_match / last_agent_match reward も完全一致で 1.0 になる。
    assert route_match("expected_route")(train_observation) == pytest.approx(1.0)
    assert last_agent_match("expected_last_agent")(train_observation) == pytest.approx(1.0)


async def test_fired_approvals_populated_and_approval_match_rewards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """中断時の pending tool_name が fired_approvals に積まれ approval_match の recall に使える。

    `_run_one` は初回 outcome.pending を fired に積む。`approval_match("expected_approvals")` は
    fired の集合が期待を recall すれば 1.0（HITL 同等性ギャップ埋め）。
    """
    from types import SimpleNamespace

    from oai_agentspec.runtime.lightning import approval_match
    from oai_agentspec.runtime.llmops import ObservedRoute, ObservedRun

    _patch_run_apo(monkeypatch, _calling_run_apo())
    obs = ObservedRun(route=ObservedRoute(steps=[], last_agent="bot"), tool_calls=[])

    async def _interrupted(self: Any, agent: Any, value: Any, **kwargs: Any) -> Any:
        outcome = SimpleNamespace(
            final_output=None,
            interrupted=True,
            pending=[
                {"tool_name": "delete_account", "call_id": "c1", "agent_name": "bot"},
                {"tool_name": "wire_money", "call_id": "c2", "agent_name": "bot"},
            ],
            state=object(),
        )
        return outcome, obs

    async def _resume(agent: Any, state: Any, **kwargs: Any) -> Any:
        outcome = SimpleNamespace(final_output="done", interrupted=False, pending=[], state=None)
        return outcome, obs

    monkeypatch.setattr(
        "oai_agentspec._adapters.DefaultRunnerAdapter.run_with_observation",
        _interrupted,
        raising=True,
    )
    monkeypatch.setattr("oai_agentspec._adapters.resume_with_observation", _resume, raising=True)
    monkeypatch.setattr(
        "oai_agentspec._adapters.apply_approvals",
        lambda state, decisions: SimpleNamespace(
            applied=[d["call_id"] for d in decisions], unknown=[], already_resolved=[]
        ),
        raising=True,
    )

    @function_tool(name_override="delete_account", needs_approval=True)
    def _danger1(x: str) -> str:
        """承認必須ツール 1（テスト用）。"""
        return f"real:{x}"

    @function_tool(name_override="wire_money", needs_approval=True)
    def _danger2(x: str) -> str:
        """承認必須ツール 2（テスト用）。"""
        return f"real:{x}"

    captured: list[RolloutResult] = []

    def _capture_reward(r: RolloutResult) -> float:
        captured.append(r)
        return 0.0

    spec = AgentSpec(name="bot", instructions="i", model=FakeModel(), tools=[_danger1, _danger2])

    await optimize(
        spec,
        algorithm="apo",
        train=[
            {"input": "do dangerous things", "expected_approvals": ["delete_account", "wire_money"]}
        ],
        val=_DEFAULT_VAL,
        reward=_capture_reward,
        approvals=lambda p: True,
        tool_mocks={"bot": {"delete_account": "mocked", "wire_money": "mocked"}},
        config=_apo_config(),
    )
    # fired_approvals に両方の承認ゲートが積まれている。
    assert captured, "rollout reward が呼ばれていない"
    train_case = next((c for c in captured if c.case.get("input") == "do dangerous things"), None)
    assert train_case is not None
    assert set(train_case.fired_approvals) == {"delete_account", "wire_money"}
    # approval_match が full recall で 1.0 を返す。
    assert approval_match("expected_approvals")(train_case) == pytest.approx(1.0)


async def test_fired_approvals_populated_even_without_approvals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`approvals=None` で中断したまま戻る場合も初回 pending は fired_approvals に積まれる。

    reward callable が承認ゲート発火を観測できることが目的（rollout 完了とは独立に承認挙動を採点）。
    """
    from types import SimpleNamespace

    from oai_agentspec.runtime.llmops import ObservedRoute, ObservedRun

    _patch_run_apo(monkeypatch, _calling_run_apo())
    obs = ObservedRun(route=ObservedRoute(steps=[], last_agent="bot"), tool_calls=[])

    async def _interrupted(self: Any, agent: Any, value: Any, **kwargs: Any) -> Any:
        outcome = SimpleNamespace(
            final_output=None,
            interrupted=True,
            pending=[{"tool_name": "danger", "call_id": "c1", "agent_name": "bot"}],
            state=object(),
        )
        return outcome, obs

    monkeypatch.setattr(
        "oai_agentspec._adapters.DefaultRunnerAdapter.run_with_observation",
        _interrupted,
        raising=True,
    )

    @function_tool(name_override="danger", needs_approval=True)
    def _danger(x: str) -> str:
        """承認必須ツール（テスト用）。"""
        return f"real:{x}"

    captured: list[RolloutResult] = []

    def _capture(r: RolloutResult) -> float:
        captured.append(r)
        return 0.0

    spec = AgentSpec(name="bot", instructions="i", model=FakeModel(), tools=[_danger])
    await optimize(
        spec,
        algorithm="apo",
        train=[{"input": "x"}],
        val=_DEFAULT_VAL,
        reward=_capture,
        # approvals 未指定 → resume せず中断のまま戻る。
        config=_apo_config(),
    )
    # 中断時の pending tool_name が fired_approvals に入っている。
    train_case = next((c for c in captured if c.case.get("input") == "x"), None)
    assert train_case is not None
    assert train_case.fired_approvals == ["danger"]


async def test_resolve_loop_breaks_on_no_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """apply で 1 件も適用できない（applied 空）と進展なしで即 break（resume を呼ばない）。"""
    from types import SimpleNamespace

    from oai_agentspec.runtime.llmops import ObservedRoute, ObservedRun

    _patch_run_apo(monkeypatch, _calling_run_apo())
    observation = ObservedRun(route=ObservedRoute(steps=[], last_agent="bot"), tool_calls=[])
    resumed = {"count": 0}

    async def _interrupted(self: Any, agent: Any, value: Any, **kwargs: Any) -> Any:
        outcome = SimpleNamespace(
            final_output=None,
            interrupted=True,
            pending=[{"tool_name": "danger", "call_id": "c1", "agent_name": "bot"}],
            state=object(),
        )
        return outcome, observation

    async def _resume(agent: Any, state: Any, **kwargs: Any) -> Any:
        resumed["count"] += 1
        return SimpleNamespace(final_output="x", interrupted=False, pending=[], state=None), (
            observation
        )

    monkeypatch.setattr(
        "oai_agentspec._adapters.DefaultRunnerAdapter.run_with_observation",
        _interrupted,
        raising=True,
    )
    monkeypatch.setattr("oai_agentspec._adapters.resume_with_observation", _resume, raising=True)
    # applied 空 → 進展なしで break。
    monkeypatch.setattr(
        "oai_agentspec._adapters.apply_approvals",
        lambda state, decisions: SimpleNamespace(applied=[], unknown=["c1"], already_resolved=[]),
        raising=True,
    )

    @function_tool(name_override="danger", needs_approval=True)
    def _danger(x: str) -> str:
        """承認必須ツール（テスト用）。"""
        return f"real:{x}"

    spec = AgentSpec(name="bot", instructions="i", model=FakeModel(), tools=[_danger])

    result = await optimize(
        spec,
        algorithm="apo",
        train=[{"input": "hi"}],
        val=_DEFAULT_VAL,
        reward=lambda r: 0.0,
        approvals=lambda p: True,
        tool_mocks={"bot": {"danger": "mocked"}},
        config=_apo_config(),
    )
    # 進展なしで即 break するため resume は呼ばれない（空回りなし）。
    assert resumed["count"] == 0
    assert isinstance(result, OptimizeResult)


async def test_workflow_graph_optimization_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    """関数ノードのみの WorkflowGraph は registry=None でも rollout が回る（WF 経路）。"""
    from oai_agentspec.workflow import END, START, WorkflowGraph

    _patch_run_apo(monkeypatch, _calling_run_apo())

    wf = WorkflowGraph(name="pipeline")
    wf.add_function_node("upper", fn=lambda msg, ctx: f"EXPECTED {str(msg).upper()}")
    wf.add_edge(START, "upper")
    wf.add_edge("upper", END)

    # 生 seed + rebind 経路（WorkflowGraph は AgentSpec でないため slot=None は rebind 必須）。
    result = await optimize(
        wf,
        algorithm="apo",
        train=[{"input": "hello", "expected": "EXPECTED"}],
        val=_DEFAULT_VAL,
        reward=contains("expected"),
        slot="wf seed",
        rebind=lambda candidate: wf,  # 候補に依らず同じ wf を返す（FUNCTION ノードのみ）。
        config=_apo_config(),
    )
    assert isinstance(result, OptimizeResult)
    assert result.train_score == pytest.approx(1.0)


async def test_var_loss_candidate_scores_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """候補が必要 `${var}` を喪失すると rollout は無効化され 0.0（fail-closed・低評価）。

    seed が `${role}` を含み vars に role があるとき、Trainer 候補が `${role}` を落とすと
    `_apply_candidate` が None を返し rollout が 0.0 を返す。run_apo を「seed から var を落とした
    候補で rollout する」fake にして検証する。
    """

    async def _drop_var(
        *,
        seeds: Any,
        train: Any,
        val: Any,
        rollout: Any,
        config: Any,
        vars_per_slot: Any = None,  # noqa: ARG001 - run_apo の新パラメータに合わせる
    ) -> Any:
        # var を落とした候補（${role} 無し）で rollout する。
        candidate = dict.fromkeys(seeds, "stripped candidate without placeholder")
        case = list(train)[0]
        score = await rollout(candidate, case)
        return OptimizeResult(prompt="x", train_score=score)

    _patch_run_apo(monkeypatch, _drop_var)

    slot = Slot(
        name="bot",
        seed="hi ${role}",
        build=lambda c: _spec(name="bot", instructions=c, output_text="expected"),
        vars={"role": "agent"},
    )
    result = await optimize(
        _spec(),
        algorithm="apo",
        train=[{"input": "hi", "expected": "expected"}],
        val=_DEFAULT_VAL,
        reward=contains("expected"),
        slot=slot,
        config=_apo_config(),
    )
    # 候補が ${role} を喪失 → _apply_candidate が None → rollout 0.0。
    assert result.train_score == pytest.approx(0.0)


# ----------------------------------------------------------------------
# 内部ヘルパ直接検証（_normalize_slots / _seeds_of / _reinject_vars / _build_decisions）
# ----------------------------------------------------------------------


@pytest.mark.unit
def test_normalize_slots_default_for_static_agent_spec() -> None:
    """slot=None + 静的 AgentSpec は instructions を seed とする既定 Slot を 1 件作る。"""
    spec = AgentSpec(name="bot", instructions="static body", model=FakeModel())
    slots = _normalize_slots(spec, None)
    assert slots is not None
    assert set(slots) == {"bot"}
    assert slots["bot"].seed == "static body"
    # 既定 build は instructions を候補で差し替える。
    rebuilt = slots["bot"].build("new body")
    assert isinstance(rebuilt, AgentSpec)
    assert rebuilt.instructions == "new body"


@pytest.mark.unit
def test_normalize_slots_callable_instructions_returns_none() -> None:
    """slot=None + callable instructions の AgentSpec は既定スロット導出不可で None を返す。"""
    spec = AgentSpec(name="bot", instructions=lambda ctx, agent: "dyn", model=FakeModel())
    assert _normalize_slots(spec, None) is None


@pytest.mark.unit
def test_normalize_slots_single_slot_object() -> None:
    """Slot 1 件は `{name: slot}` へ正規化する。"""
    slot = Slot(name="a", seed="s", build=lambda c: c)
    assert _normalize_slots(object(), slot) == {"a": slot}


@pytest.mark.unit
def test_normalize_slots_all_slot_dict() -> None:
    """全 Slot の dict はそのまま `{名前: Slot}` へ正規化する。"""
    a = Slot(name="a", seed="sa", build=lambda c: c)
    b = Slot(name="b", seed="sb", build=lambda c: c)
    result = _normalize_slots(object(), {"a": a, "b": b})
    assert result == {"a": a, "b": b}


@pytest.mark.unit
def test_normalize_slots_raw_str_returns_none() -> None:
    """生 seed（str）は rebind 必須経路で None。"""
    assert _normalize_slots(object(), "raw seed") is None


@pytest.mark.unit
def test_normalize_slots_all_raw_dict_returns_none() -> None:
    """全て生 seed(str) の dict は rebind 必須経路で None（Slot 混在なし）。"""
    assert _normalize_slots(object(), {"a": "x", "b": "y"}) is None


@pytest.mark.unit
def test_normalize_slots_empty_dict_raises_config_missing() -> None:
    """slot={} は不正設定で OptimizeError(CONFIG_MISSING)（Codex P2 回帰防止）。"""
    with pytest.raises(OptimizeError, match="slot の dict が空") as exc:
        _normalize_slots(object(), {})
    assert exc.value.kind == FailureKind.CONFIG_MISSING


@pytest.mark.unit
def test_extract_case_input_from_dict() -> None:
    """dict ケースからは `case["input"]` を取り出す。"""
    assert _extract_case_input({"input": "hi", "expected": "x"}) == "hi"
    # input キーが無ければ None。
    assert _extract_case_input({"expected": "x"}) is None


@pytest.mark.unit
def test_extract_case_input_from_optimize_case() -> None:
    """OptimizeCase からは `.input` 属性を取り出す。"""
    case = OptimizeCase(input="ユーザー依頼", expected_output="ans")
    assert _extract_case_input(case) == "ユーザー依頼"


@pytest.mark.unit
def test_extract_case_input_from_str_falls_back_to_self() -> None:
    """属性も "input" キーも無い生 str / 任意オブジェクトは case 自身を返す（後方互換）。"""
    assert _extract_case_input("raw input string") == "raw input string"


# ----------------------------------------------------------------------
# OptimizeCase（typed ケース）経由の rollout 経路
# ----------------------------------------------------------------------


async def test_optimize_case_drives_rollout_via_attribute_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OptimizeCase は `case.input` を rollout 入力として渡し reward を駆動する。"""
    _patch_run_apo(monkeypatch, _calling_run_apo())

    captured: list[RolloutResult] = []

    def _reward(r: RolloutResult) -> float:
        captured.append(r)
        return 1.0 if isinstance(r.case, OptimizeCase) else 0.0

    train_case = OptimizeCase(input="hi train", expected_output="expected")
    val_case = OptimizeCase(input="hi val", expected_output="expected")

    result = await optimize(
        _spec(output_text="the expected answer"),
        algorithm="apo",
        train=[train_case],
        val=[val_case],
        reward=_reward,
        config=_apo_config(),
    )
    # case が OptimizeCase のまま reward へ渡る（dict 化されない）。
    assert all(isinstance(c.case, OptimizeCase) for c in captured)
    assert result.train_score == pytest.approx(1.0)


async def test_optimize_case_default_field_rewards_via_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OptimizeCase + 既定 field の reward ファクトリで採点が成立する（field 引数省略）。"""
    _patch_run_apo(monkeypatch, _calling_run_apo())

    case = OptimizeCase(input="please", expected_output="expected")
    result = await optimize(
        _spec(output_text="the expected answer"),
        algorithm="apo",
        train=[case],
        val=[case],
        reward=contains(),  # 既定 field=expected_output で OptimizeCase.expected_output を読む。
        config=_apo_config(),
    )
    assert result.train_score == pytest.approx(1.0)


@pytest.mark.unit
def test_seeds_of_from_slots() -> None:
    """slots ありなら各 Slot.seed を `{名前: seed}` で返す。"""
    a = Slot(name="a", seed="sa", build=lambda c: c)
    assert _seeds_of({"a": a}, None) == {"a": "sa"}


@pytest.mark.unit
def test_seeds_of_from_raw_str() -> None:
    """生 seed（str）は `{"prompt": seed}` で返す。"""
    assert _seeds_of(None, "raw") == {"prompt": "raw"}


@pytest.mark.unit
def test_seeds_of_from_raw_dict() -> None:
    """生 seed dict は値を str 化して返す。"""
    assert _seeds_of(None, {"a": "x", "b": "y"}) == {"a": "x", "b": "y"}


@pytest.mark.unit
def test_seeds_of_none_returns_empty() -> None:
    """slots も slot も無い場合は空 dict。"""
    assert _seeds_of(None, None) == {}


@pytest.mark.unit
def test_reinject_vars_substitutes_present_placeholders() -> None:
    """候補に `${var}` が残っていれば vars を再注入する。"""
    slot = Slot(name="a", seed="hi ${name}", build=lambda c: c, vars={"name": "world"})
    assert _reinject_vars(slot, "greet ${name}!") == "greet world!"


@pytest.mark.unit
def test_reinject_vars_missing_required_placeholder_returns_none() -> None:
    """seed にあった必要 `${var}` を候補が喪失していたら None（fail-closed）。"""
    slot = Slot(name="a", seed="hi ${name}", build=lambda c: c, vars={"name": "world"})
    # 候補が ${name} を含まない → 無効化。
    assert _reinject_vars(slot, "no placeholder here") is None


@pytest.mark.unit
def test_reinject_vars_keeps_undefined_placeholders() -> None:
    """vars に無い `${var}` は safe_substitute で保持する。"""
    slot = Slot(name="a", seed="${kept}", build=lambda c: c, vars={})
    assert _reinject_vars(slot, "value ${kept}") == "value ${kept}"


@pytest.mark.unit
def test_reinject_vars_validates_seed_placeholders_not_only_vars_keys() -> None:
    """seed の `${var}` 検査は `slot.vars` のキーに限らず seed 全 placeholder を見る。

    Codex P2 回帰防止: vars に渡されていない placeholder（例: `${role}`）でも、APO 公開契約
    「最適化済みテンプレートは `${var}` を保持する」を満たすため、候補が落とせば無効化する。
    旧実装は `for key in slot.vars` でしか検査していなかったため、vars 未指定の placeholder を
    候補が消去しても素通しになっていた。
    """
    # vars 未指定だが seed には `${role}` がある（保持されるべき）。
    slot = Slot(name="a", seed="あなたは ${role} です", build=lambda c: c, vars={})
    # 候補が `${role}` を喪失 → 無効化（None）。
    assert _reinject_vars(slot, "あなたは AI です") is None
    # 候補が `${role}` を保持していれば素通し（vars 未指定なので展開もしない）。
    assert _reinject_vars(slot, "あなたは ${role} で頑張る") == "あなたは ${role} で頑張る"


@pytest.mark.unit
def test_reinject_vars_checks_multiple_seed_placeholders() -> None:
    """seed に複数 placeholder（vars に部分的にしかキーが無い）でも全てを検査する。"""
    slot = Slot(
        name="a",
        seed="${role} of ${company} for ${customer}",
        build=lambda c: c,
        vars={"company": "AgentSpec"},  # role / customer は vars 未指定
    )
    # 全 placeholder 保持 → vars にあるキーだけ展開され、残りは ${...} のまま。
    assert (
        _reinject_vars(slot, "${role} of ${company} for ${customer}")
        == "${role} of AgentSpec for ${customer}"
    )
    # `${role}` を 1 個でも落とせば無効化。
    assert _reinject_vars(slot, "AI of ${company} for ${customer}") is None


# ----------------------------------------------------------------------
# _build_decisions（安全不変条件の直接検証・apply/resume を介さない）
# ----------------------------------------------------------------------

_PENDING = {"tool_name": "danger", "call_id": "c1", "agent_name": "bot"}
_REPLACED = frozenset({("bot", "danger")})


@pytest.mark.unit
def test_build_decisions_approve_without_replaced_raises() -> None:
    """approve かつ実差し替え集合に無ければ OptimizeError(CONFIG_MISSING)（危険ツール非実行）。"""
    with pytest.raises(OptimizeError, match="モックへ差し替え") as exc:
        _build_decisions([_PENDING], resolver=lambda p: True, replaced_tools=frozenset())
    assert exc.value.kind == FailureKind.CONFIG_MISSING


@pytest.mark.unit
def test_build_decisions_approve_with_replaced_returns_approve() -> None:
    """approve かつ実差し替え集合に在れば approve decision を返す。"""
    decisions = _build_decisions([_PENDING], resolver=lambda p: True, replaced_tools=_REPLACED)
    assert decisions == [{"call_id": "c1", "decision": "approve"}]


@pytest.mark.unit
def test_build_decisions_approve_different_agent_raises() -> None:
    """同名ツールでも別 agent の approve は認可しない（OptimizeError・同名すり抜け阻止）。"""
    other = {"tool_name": "danger", "call_id": "c2", "agent_name": "other"}
    with pytest.raises(OptimizeError, match="other") as exc:
        _build_decisions([other], resolver=lambda p: True, replaced_tools=_REPLACED)
    assert exc.value.kind == FailureKind.CONFIG_MISSING


@pytest.mark.unit
def test_build_decisions_reject_is_safe_without_replaced() -> None:
    """reject は実差し替え集合が空でも例外なく reject decision を返す（ツール非実行）。"""
    decisions = _build_decisions([_PENDING], resolver=lambda p: False, replaced_tools=frozenset())
    assert decisions[0]["decision"] == "reject"
    assert "rejection_message" in decisions[0]


@pytest.mark.unit
def test_build_decisions_resolver_receives_pending_dict() -> None:
    """resolver には agent_name / tool_name を含む pending dict が渡る。"""
    seen: dict[str, Any] = {}

    def _resolver(p: dict) -> bool:
        seen.update(p)
        return True

    _build_decisions([_PENDING], resolver=_resolver, replaced_tools=_REPLACED)
    assert seen.get("agent_name") == "bot"
    assert seen.get("tool_name") == "danger"


# ----------------------------------------------------------------------
# _target（lightning 専用複製）の直接検証（target_id / normalize 分岐）
# ----------------------------------------------------------------------


@pytest.mark.unit
def test_target_id_for_agent_spec() -> None:
    """target_id は AgentSpec の name を返す。"""
    from oai_agentspec.runtime.lightning import _target as target_mod

    assert target_mod.target_id(_spec(name="x")) == "x"


@pytest.mark.unit
def test_target_id_for_handoff_graph_uses_entry() -> None:
    """target_id は HandoffGraph の entry 名を返す（entry 未指定は "handoff_graph"）。"""
    from oai_agentspec.runtime.lightning import _target as target_mod

    assert target_mod.target_id(HandoffGraph(entry="triage")) == "triage"
    assert target_mod.target_id(HandoffGraph()) == "handoff_graph"


@pytest.mark.unit
def test_target_id_for_workflow_graph() -> None:
    """target_id は WorkflowGraph に "workflow" を返す。"""
    from oai_agentspec.runtime.lightning import _target as target_mod
    from oai_agentspec.workflow import WorkflowGraph

    assert target_mod.target_id(WorkflowGraph(name="pipeline")) == "workflow"


@pytest.mark.unit
def test_target_id_fallback_for_unknown_object() -> None:
    """未知オブジェクトは name 属性 / "target" にフォールバックする。"""
    from oai_agentspec.runtime.lightning import _target as target_mod

    class _Named:
        name = "custom"

    assert target_mod.target_id(_Named()) == "custom"
    assert target_mod.target_id(object()) == "target"


@pytest.mark.unit
def test_normalize_agent_spec_builds_agent() -> None:
    """AgentSpec は build_agent 経由で正規化され (Agent, 空集合) を返す（mock 無し）。"""
    from oai_agentspec.runtime.lightning import _target as target_mod

    agent, replaced = target_mod.normalize(_spec(), None)
    assert agent is not None
    assert replaced == frozenset()


@pytest.mark.unit
def test_normalize_handoff_graph_without_registry_raises() -> None:
    """HandoffGraph で registry=None は ValueError。"""
    from oai_agentspec.runtime.lightning import _target as target_mod

    with pytest.raises(ValueError, match="registry"):
        target_mod.normalize(HandoffGraph(entry="triage"), None)


@pytest.mark.unit
def test_normalize_unsupported_type_raises_type_error() -> None:
    """未対応 target 型は許容型を列挙した TypeError。"""
    from oai_agentspec.runtime.lightning import _target as target_mod

    with pytest.raises(TypeError, match="AgentSpec / HandoffGraph / WorkflowGraph"):
        target_mod.normalize(123, None)


@pytest.mark.unit
def test_normalize_workflow_graph_agent_node_mocks_via_clone() -> None:
    """AGENT ノードを含む WorkflowGraph + tool_mocks はクローン registry 経由で mock 化される。"""
    from oai_agentspec.runtime.lightning import _target as target_mod
    from oai_agentspec.workflow import END, START, WorkflowGraph

    @function_tool(name_override="danger", needs_approval=True)
    def _danger(x: str) -> str:
        """承認必須ツール（テスト用・AGENT ノード側）。"""
        return f"real:{x}"

    reg = AgentRegistry()
    reg.register(_spec(name="worker", tools=[_danger]))
    wf = WorkflowGraph(name="pipeline")
    wf.add_agent_node("worker", agent="worker")
    wf.add_edge(START, "worker")
    wf.add_edge("worker", END)

    _agent, replaced = target_mod.normalize(wf, reg, tool_mocks={"worker": {"danger": "mock"}})
    # AGENT ノードが参照する registry agent のツールもクローン経由で mock 済み。
    assert replaced == frozenset({("worker", "danger")})


# ----------------------------------------------------------------------
# Issue #40 T7: 結果整形パス（新 shape slot の vars_per_slot 受け渡し・
# run_apo 返却後の full 再合成 + diff 再計算）
# ----------------------------------------------------------------------


def _store_new_shape(tmp_path: Path) -> Any:
    """新 shape（agent= 指定）テスト用ストア（agents/ ディレクトリにセグメントを配置）。"""
    from oai_agentspec.prompts import PromptLayout, PromptStore

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "triage.md").write_text("Triage seed ${tone}", encoding="utf-8")
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "main.md").write_text("BASE ${org}", encoding="utf-8")
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    (parts_dir / "style.md").write_text("STYLE part", encoding="utf-8")
    return PromptStore(tmp_path, PromptLayout(base="base", parts="parts", agents="agents"))


async def test_optimize_new_shape_slot_passes_vars_per_slot_to_run_apo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """新 shape（segments 非空）の slot は run_apo へ `vars_per_slot` を渡し、固定セグメントの
    full 再合成は optimizer の `_recompose_new_shape_results` が segments SoT から担う。"""
    from oai_agentspec.runtime.lightning import prompt_slot

    store = _store_new_shape(tmp_path)
    slot = prompt_slot(
        store,
        AgentRegistry(),
        agent="triage",
        base="main",
        parts=["style"],
        vars={"org": "AgentSpec"},
    )

    captured: dict[str, Any] = {}

    async def _fake(
        *,
        seeds: dict[str, str],
        train: Any,
        val: Any,
        rollout: Any,
        config: Any,
        vars_per_slot: Any = None,
    ) -> OptimizeResult:
        captured["vars_per_slot"] = vars_per_slot
        return OptimizeResult(prompt=seeds[next(iter(seeds))], train_score=0.0, val_score=0.0)

    _patch_run_apo(monkeypatch, _fake)

    await optimize(
        _spec(name="triage"),
        train=[{"input": "x"}],
        val=_DEFAULT_VAL,
        reward=contains("e"),
        slot=slot,
        config=_apo_config(),
    )
    assert captured["vars_per_slot"]["triage"] == {"org": "AgentSpec"}


async def test_optimize_custom_build_slot_result_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """custom build 経路（`Slot.segments = ()`）は run_apo の返却をそのまま `OptimizeResult` に
    詰める（`_recompose_new_shape_results` の segments 空スキップ契約の regression guard）。

    `_recompose_new_shape_results` の `if not slot.segments: continue` が silent regression した
    場合、custom build 利用者の OptimizeResult.prompt が build 出力を無視して recompose された
    文字列に書き換わるため、本テストが検知する。
    """
    from oai_agentspec.runtime.lightning import prompt_slot

    store = _store_new_shape(tmp_path)

    def _custom_build(candidate: str) -> AgentSpec:
        return _spec(name="triage", instructions=candidate)

    slot = prompt_slot(
        store,
        AgentRegistry(),
        agent="triage",
        base="main",
        vars={"org": "AgentSpec"},
        build=_custom_build,
    )
    assert slot.segments == ()

    async def _fake(
        *,
        seeds: dict[str, str],
        train: Any,
        val: Any,
        rollout: Any,
        config: Any,
        vars_per_slot: Any = None,
    ) -> OptimizeResult:
        return OptimizeResult(
            prompt="RAW PROMPT", train_score=1.0, val_score=1.0, seed="RAW SEED", diff="RAW DIFF"
        )

    _patch_run_apo(monkeypatch, _fake)

    result = await optimize(
        _spec(name="triage"),
        train=[{"input": "x"}],
        val=_DEFAULT_VAL,
        reward=contains("e"),
        slot=slot,
        config=_apo_config(),
    )
    # segments 空 slot は再合成対象外・run_apo 返却がそのまま OptimizeResult に載る。
    assert result.prompt == "RAW PROMPT"
    assert result.seed == "RAW SEED"
    assert result.diff == "RAW DIFF"


async def test_optimize_vars_callable_slot_passes_empty_vars_per_slot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`vars=callable` の slot は `vars_per_slot` に空 dict を渡す（callable キーは最適化ループへ
    非伝搬・論点 G）。"""
    from oai_agentspec.runtime.lightning import prompt_slot

    store = _store_new_shape(tmp_path)
    slot = prompt_slot(store, AgentRegistry(), agent="triage", vars=lambda ctx: {"org": ctx})

    captured: dict[str, Any] = {}

    async def _fake(
        *,
        seeds: dict[str, str],
        train: Any,
        val: Any,
        rollout: Any,
        config: Any,
        vars_per_slot: Any = None,
    ) -> OptimizeResult:
        captured["vars_per_slot"] = vars_per_slot
        return OptimizeResult(prompt=seeds[next(iter(seeds))], train_score=0.0, val_score=0.0)

    _patch_run_apo(monkeypatch, _fake)

    await optimize(
        _spec(name="triage"),
        train=[{"input": "x"}],
        val=_DEFAULT_VAL,
        reward=contains("e"),
        slot=slot,
        config=_apo_config(),
    )
    assert captured["vars_per_slot"]["triage"] == {}


async def test_optimize_new_shape_result_prompt_is_full_composed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """新 shape slot は run_apo 返却後に `compose_from_marked` で full 再合成した `prompt` を返す
    （固定セグメント本文を含み境界マーカーは含まない・論点 G）。"""
    from oai_agentspec.runtime.lightning import prompt_slot

    store = _store_new_shape(tmp_path)
    slot = prompt_slot(
        store,
        AgentRegistry(),
        agent="triage",
        base="main",
        parts=["style"],
        vars={"org": "AgentSpec"},
    )

    async def _fake(
        *,
        seeds: dict[str, str],
        train: Any,
        val: Any,
        rollout: Any,
        config: Any,
        vars_per_slot: Any = None,
    ) -> OptimizeResult:
        return OptimizeResult(
            prompt="OPTIMIZED TUNE TEXT",
            train_score=1.0,
            val_score=1.0,
            seed=seeds["triage"],
            diff="OLD_DIFF",
        )

    _patch_run_apo(monkeypatch, _fake)

    result = await optimize(
        _spec(name="triage"),
        train=[{"input": "x"}],
        val=_DEFAULT_VAL,
        reward=contains("e"),
        slot=slot,
        config=_apo_config(),
    )
    assert result.prompt == "BASE AgentSpec\n\nSTYLE part\n\nOPTIMIZED TUNE TEXT"
    assert "oas_boundary" not in result.prompt


async def test_optimize_new_shape_result_seed_is_full_composed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """新 shape slot は run_apo 返却後に `compose_from_marked` で full 再合成した `seed` を返す。"""
    from oai_agentspec.runtime.lightning import prompt_slot

    store = _store_new_shape(tmp_path)
    slot = prompt_slot(
        store,
        AgentRegistry(),
        agent="triage",
        base="main",
        parts=["style"],
        vars={"org": "AgentSpec"},
    )

    async def _fake(
        *,
        seeds: dict[str, str],
        train: Any,
        val: Any,
        rollout: Any,
        config: Any,
        vars_per_slot: Any = None,
    ) -> OptimizeResult:
        return OptimizeResult(
            prompt="OPTIMIZED TUNE TEXT",
            train_score=1.0,
            val_score=1.0,
            seed=seeds["triage"],
            diff="OLD_DIFF",
        )

    _patch_run_apo(monkeypatch, _fake)

    result = await optimize(
        _spec(name="triage"),
        train=[{"input": "x"}],
        val=_DEFAULT_VAL,
        reward=contains("e"),
        slot=slot,
        config=_apo_config(),
    )
    assert result.seed == "BASE AgentSpec\n\nSTYLE part\n\nTriage seed ${tone}"


async def test_optimize_new_shape_result_diff_is_recomputed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """新 shape slot は run_apo が返す tune-only diff を破棄し、full 合成後の seed/prompt から
    `difflib.unified_diff` で再計算した `diff` を返す（論点 G）。"""
    from oai_agentspec.runtime.lightning import prompt_slot

    store = _store_new_shape(tmp_path)
    slot = prompt_slot(
        store,
        AgentRegistry(),
        agent="triage",
        base="main",
        parts=["style"],
        vars={"org": "AgentSpec"},
    )

    async def _fake(
        *,
        seeds: dict[str, str],
        train: Any,
        val: Any,
        rollout: Any,
        config: Any,
        vars_per_slot: Any = None,
    ) -> OptimizeResult:
        return OptimizeResult(
            prompt="OPTIMIZED TUNE TEXT",
            train_score=1.0,
            val_score=1.0,
            seed=seeds["triage"],
            diff="OLD_DIFF",
        )

    _patch_run_apo(monkeypatch, _fake)

    result = await optimize(
        _spec(name="triage"),
        train=[{"input": "x"}],
        val=_DEFAULT_VAL,
        reward=contains("e"),
        slot=slot,
        config=_apo_config(),
    )
    assert "OLD_DIFF" not in result.diff
    # SSoT ヘルパ `unified_diff_labeled` の統一ラベル（before / after）で再計算されている。
    assert "--- before" in result.diff
    assert "+++ after" in result.diff


async def test_optimize_new_shape_multi_slot_dict_recompose_per_slot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """複数 slot（HandoffGraph + `prompt_slots`）で run_apo が dict 型 `OptimizeResult` を返す
    経路の per-slot 再合成を検証する（`_recompose_new_shape_results` の isinstance(prompt, dict)
    分岐の regression guard）。

    全 slot について `result.prompt[name]` / `result.seed[name]` が固定セグメント込みの full
    合成テキストになり、`result.diff[name]` が `unified_diff_labeled` の統一ラベルで再計算される
    ことを固定する。将来 dict 分岐で 1 slot 分の合成しか反映されず他 slot が tune-only のまま
    silent 通過する regression を検出する。
    """
    from oai_agentspec.prompts import PromptLayout, PromptStore
    from oai_agentspec.runtime.lightning import prompt_slots

    # 2 agent 用のローカルストア（_store_new_shape は triage のみのため）。
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "triage.md").write_text("Triage seed", encoding="utf-8")
    (agents_dir / "second.md").write_text("Second seed", encoding="utf-8")
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "main.md").write_text("BASE ${org}", encoding="utf-8")
    store = PromptStore(tmp_path, PromptLayout(base="base", parts="parts", agents="agents"))

    registry = AgentRegistry()
    registry.register(_spec(name="triage"))
    registry.register(_spec(name="second"))
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "second", description="次のエージェント")
    graph.apply(registry)
    registry.validate()

    slots = prompt_slots(
        store, registry, agents=["triage", "second"], base="main", vars={"org": "AgentSpec"}
    )

    async def _fake(
        *,
        seeds: dict[str, str],
        train: Any,
        val: Any,
        rollout: Any,
        config: Any,
        vars_per_slot: Any = None,
    ) -> OptimizeResult:
        # dict 型で複数 slot 分の tune-only テキストを返す（run_apo の複数 slot 契約）。
        return OptimizeResult(
            prompt={"triage": "TUNED_TRIAGE", "second": "TUNED_SECOND"},
            seed={"triage": seeds["triage"], "second": seeds["second"]},
            diff={"triage": "OLD_TRIAGE_DIFF", "second": "OLD_SECOND_DIFF"},
            train_score=1.0,
            val_score=1.0,
        )

    _patch_run_apo(monkeypatch, _fake)

    result = await optimize(
        graph,
        train=[{"input": "x"}],
        val=_DEFAULT_VAL,
        reward=contains("e"),
        slot=slots,
        registry=registry,
        config=_apo_config(),
    )
    # dict 型が保持され、全 slot について base が prepend された full 合成になっている。
    assert isinstance(result.prompt, dict)
    assert isinstance(result.seed, dict)
    assert isinstance(result.diff, dict)
    for name, tuned in [("triage", "TUNED_TRIAGE"), ("second", "TUNED_SECOND")]:
        assert result.prompt[name] == f"BASE AgentSpec\n\n{tuned}"
        assert result.seed[name].startswith("BASE AgentSpec\n\n")
        # 古い diff は破棄され、unified_diff_labeled の統一ラベルで再計算されている。
        assert f"OLD_{name.upper()}_DIFF" not in result.diff[name]
        assert "--- before" in result.diff[name]
        assert "+++ after" in result.diff[name]


async def test_optimize_new_shape_result_prompt_matches_rollout_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """rollout build の instructions と optimizer 結果整形の prompt が同一（SSoT drift 検出）。

    新 shape の multi-tune slot で、rollout 経路（`slot.build(candidate).instructions`）と
    optimizer 経路（`run_apo` 返却後の `compose_from_marked` 再合成 prompt）が同一 SSoT
    （`compose_from_marked`）を通ることを固定する。両者がズレると
    "OptimizeResult.prompt == rollout instructions" 契約が壊れる。
    """
    from oai_agentspec.runtime.lightning import prompt_slot

    store = _store_new_shape(tmp_path)
    registry = AgentRegistry()
    registry.register(_spec(name="triage"))
    slot = prompt_slot(
        store,
        registry,
        agent="triage",
        base="main",
        tune=["main", "triage"],
        vars={"org": "AgentSpec", "tone": "formal"},
    )
    # 境界マーカー入り連結の具体的な candidate（`${var}` は温存）。
    candidate = "OPTIMIZED_MAIN ${org}\n\n${oas_boundary_1}\n\nOPTIMIZED_TRIAGE ${tone}"

    # rollout 経路: build が返す agent.instructions（compose_from_marked 経由）。
    rollout_instructions = slot.build(candidate).instructions

    async def _fake(
        *,
        seeds: dict[str, str],
        train: Any,
        val: Any,
        rollout: Any,
        config: Any,
        vars_per_slot: Any = None,
    ) -> OptimizeResult:
        return OptimizeResult(
            prompt=candidate, seed=candidate, train_score=0.5, val_score=None, diff="dummy"
        )

    _patch_run_apo(monkeypatch, _fake)

    result = await optimize(
        _spec(name="triage"),
        train=[{"input": "x"}],
        val=_DEFAULT_VAL,
        reward=contains("e"),
        slot=slot,
        config=_apo_config(),
    )
    # optimizer 経路の prompt が rollout build と一致（SSoT drift 検出）。
    assert result.prompt == rollout_instructions


# ----------------------------------------------------------------------
# T10: `context_factory` の rollout ごとの新鮮な context 素通し（Issue #40 FR-2）
# ----------------------------------------------------------------------


async def test_optimize_with_context_factory_calls_factory_per_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`context_factory` は rollout ごとに呼ばれ、戻り値が `run_with_observation` の
    `context=` へ素通しされる（rollout 間で同一オブジェクトを共有しない）。"""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from oai_agentspec.runtime.llmops import ObservedRoute, ObservedRun

    _patch_run_apo(monkeypatch, _calling_run_apo())
    observation = ObservedRun(route=ObservedRoute(steps=[], last_agent="bot"), tool_calls=[])
    captured_contexts: list[Any] = []

    async def _fake_run(
        self: Any, agent: Any, value: Any, *, context: Any = None, **kwargs: Any
    ) -> Any:
        captured_contexts.append(context)
        outcome = SimpleNamespace(
            final_output="expected", interrupted=False, pending=[], state=None
        )
        return outcome, observation

    monkeypatch.setattr(
        "oai_agentspec._adapters.DefaultRunnerAdapter.run_with_observation",
        _fake_run,
        raising=True,
    )

    factory = MagicMock(side_effect=lambda: object())

    result = await optimize(
        _spec(),
        train=[
            {"input": "hi", "expected": "expected"},
            {"input": "yo", "expected": "expected"},
        ],
        val=_DEFAULT_VAL,
        reward=contains("expected"),
        config=_apo_config(),
        context_factory=factory,
    )

    # train(2) + val(1) = 3 rollouts → factory も 3 回呼ばれ、各回で別オブジェクトを渡す。
    assert factory.call_count == 3
    assert len(captured_contexts) == 3
    assert len({id(c) for c in captured_contexts}) == 3
    assert result.train_score == pytest.approx(1.0)


async def test_optimize_without_context_factory_uses_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """`context_factory` 未指定時は既存動作のまま `context=None` で `run_with_observation` を呼ぶ
    （後方互換）。"""
    from types import SimpleNamespace

    from oai_agentspec.runtime.llmops import ObservedRoute, ObservedRun

    _patch_run_apo(monkeypatch, _calling_run_apo())
    observation = ObservedRun(route=ObservedRoute(steps=[], last_agent="bot"), tool_calls=[])
    captured_contexts: list[Any] = []

    async def _fake_run(
        self: Any, agent: Any, value: Any, *, context: Any = None, **kwargs: Any
    ) -> Any:
        captured_contexts.append(context)
        outcome = SimpleNamespace(
            final_output="expected", interrupted=False, pending=[], state=None
        )
        return outcome, observation

    monkeypatch.setattr(
        "oai_agentspec._adapters.DefaultRunnerAdapter.run_with_observation",
        _fake_run,
        raising=True,
    )

    result = await optimize(
        _spec(),
        train=[{"input": "hi", "expected": "expected"}],
        val=_DEFAULT_VAL,
        reward=contains("expected"),
        config=_apo_config(),
        # context_factory を渡さない。
    )

    assert captured_contexts == [None, None]
    assert result.train_score == pytest.approx(1.0)


async def test_optimize_context_factory_reaches_dynamic_instructions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`context_factory` が生成した context が `vars=callable` の動的 instructions まで実際に
    届く（SDK Runner 実行を経由・fake run_with_observation を挟まない）。"""
    from types import SimpleNamespace

    from oai_agentspec.runtime.lightning import prompt_slot

    store = _store_new_shape(tmp_path)
    registry = AgentRegistry()
    registry.register(_spec(name="triage", output_text="expected"))

    captured_contexts: list[Any] = []

    def _vars_fn(ctx: Any) -> dict[str, Any]:
        value = ctx.context.triage_result if ctx is not None and ctx.context else "none"
        captured_contexts.append(value)
        return {"tone": value, "org": "AgentSpec"}

    slot = prompt_slot(store, registry, agent="triage", base="main", parts=["style"], vars=_vars_fn)

    _patch_run_apo(monkeypatch, _calling_run_apo())

    result = await optimize(
        _spec(name="triage"),
        train=[{"input": "hi", "expected": "expected"}],
        val=_DEFAULT_VAL,
        reward=contains("expected"),
        slot=slot,
        config=_apo_config(),
        context_factory=lambda: SimpleNamespace(triage_result="OK"),
    )

    assert "OK" in captured_contexts
    assert result.train_score == pytest.approx(1.0)


async def test_optimize_context_factory_not_recreated_within_resume_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """承認 resume ループ内では `context_factory` が再呼び出しされない（1 rollout = 1 回のみ・
    SDK RunState 内包 context の再利用）。"""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from oai_agentspec.runtime.llmops import ObservedRoute, ObservedRun

    _patch_run_apo(monkeypatch, _calling_run_apo())
    observation = ObservedRun(route=ObservedRoute(steps=[], last_agent="bot"), tool_calls=[])

    async def _interrupted(self: Any, agent: Any, value: Any, **kwargs: Any) -> Any:
        outcome = SimpleNamespace(
            final_output=None,
            interrupted=True,
            pending=[{"tool_name": "danger", "call_id": "c1", "agent_name": "bot"}],
            state=object(),
        )
        return outcome, observation

    async def _resume(agent: Any, state: Any, **kwargs: Any) -> Any:
        outcome = SimpleNamespace(
            final_output="expected final", interrupted=False, pending=[], state=None
        )
        return outcome, observation

    monkeypatch.setattr(
        "oai_agentspec._adapters.DefaultRunnerAdapter.run_with_observation",
        _interrupted,
        raising=True,
    )
    monkeypatch.setattr("oai_agentspec._adapters.resume_with_observation", _resume, raising=True)
    monkeypatch.setattr(
        "oai_agentspec._adapters.apply_approvals",
        lambda state, decisions: SimpleNamespace(
            applied=[d["call_id"] for d in decisions], unknown=[], already_resolved=[]
        ),
        raising=True,
    )

    @function_tool(name_override="danger", needs_approval=True)
    def _danger(x: str) -> str:
        """承認必須ツール（テスト用）。"""
        return f"real:{x}"

    spec = AgentSpec(name="bot", instructions="i", model=FakeModel(), tools=[_danger])
    factory = MagicMock(side_effect=lambda: SimpleNamespace())

    result = await optimize(
        spec,
        train=[{"input": "hi", "expected": "expected"}],
        val=_DEFAULT_VAL,
        reward=contains("expected"),
        approvals=lambda p: True,
        tool_mocks={"bot": {"danger": "mocked"}},
        config=_apo_config(),
        context_factory=factory,
    )

    # train + val = 2 rollout。各 rollout は承認 resume で 2 ラウンド実行されるが、
    # context_factory は rollout 単位で 1 回のみ（resume では再生成しない）= 合計 2 回。
    assert factory.call_count == 2
    assert result.train_score == pytest.approx(1.0)


# ----------------------------------------------------------------------
# NFR-4: 複数セグメント指定は連結後の 1 候補として最適化するため、単一セグメント指定と比較して
# APO の最適化ループ回数（run_apo 呼び出し回数）を増加させない
# ----------------------------------------------------------------------


async def test_optimize_new_shape_single_vs_multi_tune_matches_call_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NFR-4: 複数セグメント指定（新 shape・tune=[複数]）は単一セグメント指定（新 shape・
    agent= のみ）と同じ `run_apo` 呼び出し回数（= APO ループ数）で完了する。

    複数セグメントは連結後の 1 候補テキスト（境界マーカー入り 1 本の seed）として `run_apo` へ
    渡るため、セグメント数が増えても `run_apo` の呼び出し自体は常に 1 回（optimize 1 回につき
    run_apo は 1 回しか呼ばれない設計）。単一 tune・複数 tune の双方で同じ回数（1 回）になる
    ことを固定し、将来 `optimize` がセグメントごとにループを回す実装へ回帰した場合に検知する。
    """
    from oai_agentspec.runtime.lightning import prompt_slot

    call_count = 0

    async def _fake_run_apo(
        *,
        seeds: dict[str, str],
        train: Any,
        val: Any,
        rollout: Any,
        config: Any,
        vars_per_slot: Any = None,
    ) -> OptimizeResult:
        nonlocal call_count
        call_count += 1
        name = next(iter(seeds))
        return OptimizeResult(
            prompt=seeds[name],
            seed=seeds[name],
            diff="",
            train_score=0.0,
            val_score=0.0,
        )

    _patch_run_apo(monkeypatch, _fake_run_apo)

    same_config = _apo_config()

    # 単一セグメント指定（新 shape・agent= のみ・segments=[agent:triage] の 1 tune）。
    # ストアごとに別ディレクトリを使う（同一 tmp_path で複数 store を作ると衝突するため）。
    single_root = tmp_path / "single"
    single_root.mkdir()
    store_single = _store_new_shape(single_root)
    slot_single = prompt_slot(store_single, AgentRegistry(), agent="triage")
    await optimize(
        _spec(name="triage"),
        train=[{"input": "x"}],
        val=_DEFAULT_VAL,
        reward=contains("e"),
        slot=slot_single,
        config=same_config,
    )
    count_single = call_count

    # 複数セグメント指定（新 shape・tune=[複数]・segments 非空）。同一 OptimizeConfig を再利用。
    call_count = 0
    multi_root = tmp_path / "multi"
    multi_root.mkdir()
    store_multi = _store_new_shape(multi_root)
    slot_multi = prompt_slot(
        store_multi,
        AgentRegistry(),
        agent="triage",
        base="main",
        tune=["main", "triage"],
        vars={"org": "AgentSpec"},
    )
    await optimize(
        _spec(name="triage"),
        train=[{"input": "x"}],
        val=_DEFAULT_VAL,
        reward=contains("e"),
        slot=slot_multi,
        config=same_config,
    )
    count_multi = call_count

    # NFR-4: 同一 OptimizeConfig（rounds / beam_width / branch_factor）で
    # 単一・複数セグメント指定の run_apo 呼び出し回数（= APO ループ数）が一致する。
    assert count_multi == count_single
