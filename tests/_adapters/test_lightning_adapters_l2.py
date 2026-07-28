"""L2: Agent Lightning 統合窓口（`run_apo` / `judge_score` / `_parse_score` / 変換ヘルパ）検証。

agentlightning 0.3.0 導入済みのため `_require_agentlightning` は通る前提で、Trainer / APO /
PromptTemplate / emit_reward を fake へ差し替えて `run_apo` を直接呼び、実 Trainer 配線（algorithm
/ initial_resources / store / n_runners passthrough・`Trainer.fit(litagent, train, val_dataset=)` の
to_thread 経由呼び出し・`get_best_prompt()` から `${var}` 復元・複数 seed の順次最適化と history
記録）を網羅する。LitAgent サブクラスは `_make_litagent` を直接呼んで rollout_async / emit_reward の
契約を検証する。`_to_jinja` / `_from_jinja` ラウンドトリップは識別子境界 / 空白許容 / 非識別子の
非変換を直接検証する。`judge_score` は `agents` を import するため FakeModel を model に渡して 1
ターン実行する。`_parse_score` は数値抽出 / クランプ / 抽出不能 0.0 を直接検証する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from oai_agentspec._adapters import judge_score, run_apo
from oai_agentspec._adapters.lightning import (
    _from_jinja,
    _make_litagent,
    _parse_score,
    _require_agentlightning,
    _to_jinja,
)
from oai_agentspec.runtime.lightning import OptimizeConfig, OptimizeResult

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.integration

_FAKE_APO_CLIENT = object()


def _const_rollout(score: float) -> Any:
    """常に固定 score を返す async rollout（候補 / ケースに依らず）。"""

    async def _rollout(candidate: dict[str, str], case: Any) -> float:
        return score

    return _rollout


@dataclass
class _FakePromptTemplate:
    """`agentlightning.PromptTemplate` の最小スタブ（`.template` 属性のみ持つ）。"""

    template: str
    engine: str = "jinja"
    resource_type: str = "prompt_template"


@dataclass
class _FakeAPO:
    """`agentlightning.APO` の最小スタブ（受け取った kwargs を保存・`get_best_prompt` を提供）。

    `_history_best_score` / `_history_best_version` は `run_apo` の history エントリ生成で
    参照される。
    """

    kwargs: dict[str, Any] = field(default_factory=dict)
    best_template_text: str = "(optimized) hi {{ var }}"
    _history_best_score: float = 0.85
    _history_best_version: str = "v1"

    def get_best_prompt(self) -> _FakePromptTemplate:
        return _FakePromptTemplate(template=self.best_template_text, engine="jinja")


@dataclass
class _FakeTrainer:
    """`agentlightning.Trainer` の最小スタブ（fit 呼び出しを記録）。"""

    kwargs: dict[str, Any] = field(default_factory=dict)
    fit_calls: list[tuple[Any, list[Any], list[Any] | None]] = field(default_factory=list)

    def fit(
        self, litagent: Any, train_dataset: list[Any], *, val_dataset: list[Any] | None = None
    ) -> None:
        """sync fit。to_thread 経由で呼ばれる前提（test 側は呼ばれた事実のみ検証）。"""
        self.fit_calls.append((litagent, train_dataset, val_dataset))


@pytest.fixture
def fake_apo_factory(monkeypatch: pytest.MonkeyPatch) -> list[_FakeAPO]:
    """`agentlightning.APO` を fake へ差し替え、生成インスタンスを順に記録するフィクスチャ。"""
    created: list[_FakeAPO] = []

    def _factory(**kwargs: Any) -> _FakeAPO:
        inst = _FakeAPO(kwargs=dict(kwargs))
        created.append(inst)
        return inst

    monkeypatch.setattr("agentlightning.APO", _factory, raising=True)
    return created


@pytest.fixture
def fake_trainer_factory(monkeypatch: pytest.MonkeyPatch) -> list[_FakeTrainer]:
    """`agentlightning.Trainer` を fake へ差し替え、生成インスタンスを順に記録するフィクスチャ。"""
    created: list[_FakeTrainer] = []

    def _factory(**kwargs: Any) -> _FakeTrainer:
        inst = _FakeTrainer(kwargs=dict(kwargs))
        created.append(inst)
        return inst

    monkeypatch.setattr("agentlightning.Trainer", _factory, raising=True)
    return created


@pytest.fixture
def fake_prompt_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """`agentlightning.PromptTemplate` を fake へ差し替え（pydantic 検証を回避）。"""
    monkeypatch.setattr("agentlightning.PromptTemplate", _FakePromptTemplate, raising=True)


@pytest.fixture
def captured_rewards(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """`agentlightning.emit_reward` を fake へ差し替え、emit された値を捕捉するフィクスチャ。"""
    rewards: list[float] = []

    def _emit(reward: float, **kwargs: Any) -> None:
        rewards.append(float(reward))

    monkeypatch.setattr("agentlightning.emit_reward", _emit, raising=True)
    return rewards


# ----------------------------------------------------------------------
# _require_agentlightning
# ----------------------------------------------------------------------


def test_require_agentlightning_returns_module() -> None:
    """agentlightning 導入済みなら module を返す（未導入は ImportError・本環境では到達不可）。"""
    module = _require_agentlightning()
    assert module is not None


# ----------------------------------------------------------------------
# `${var}` <-> `{{ var }}` 変換ヘルパ
# ----------------------------------------------------------------------


@pytest.mark.unit
def test_to_jinja_converts_identifier_placeholders() -> None:
    """`${var}` を `{{ var }}` に変換する（識別子のみ・複数同居）。"""
    assert _to_jinja("hello ${name}, you are ${role}") == "hello {{ name }}, you are {{ role }}"


@pytest.mark.unit
def test_to_jinja_ignores_non_identifier() -> None:
    """`${1abc}` / `${a-b}` のような非識別子は変換しない（placeholder 喪失で fail-closed）。"""
    assert _to_jinja("${1abc} ${a-b}") == "${1abc} ${a-b}"


@pytest.mark.unit
def test_from_jinja_converts_identifier_placeholders() -> None:
    """`{{ var }}` を `${var}` に変換する（空白許容・複数同居）。"""
    src = "hi {{ name }}, role={{role}}"
    assert _from_jinja(src) == "hi ${name}, role=${role}"


@pytest.mark.unit
def test_from_jinja_extra_whitespace_tolerated() -> None:
    """`{{  var  }}`（任意空白）も変換対象。"""
    assert _from_jinja("{{  a  }} {{b}}") == "${a} ${b}"


@pytest.mark.unit
def test_jinja_roundtrip_preserves_identifiers() -> None:
    """`${var}` → jinja → `${var}` のラウンドトリップで元に戻る（識別子のみ）。"""
    original = "begin ${a} mid ${name_2} end"
    assert _from_jinja(_to_jinja(original)) == original


@pytest.mark.unit
def test_from_jinja_keeps_unknown_constructs() -> None:
    """`{{ var | filter }}` のような jinja 固有構文は本変換では維持される（後段で fail-closed）。"""
    assert _from_jinja("{{ x | default('y') }}") == "{{ x | default('y') }}"


# ----------------------------------------------------------------------
# run_apo: 実 Trainer 配線（Trainer / APO / PromptTemplate / emit_reward を fake へ）
# ----------------------------------------------------------------------


async def test_run_apo_single_seed_trainer_wired_with_apo_algorithm(
    fake_apo_factory: list[_FakeAPO],
    fake_trainer_factory: list[_FakeTrainer],
    fake_prompt_template: None,  # noqa: ARG001 - fixture 副作用のみ使用
    captured_rewards: list[float],  # noqa: ARG001 - fixture 副作用のみ使用
) -> None:
    """単一 seed で APO + Trainer が正しく構築され fit が呼ばれて best prompt が返る。"""
    config = OptimizeConfig(apo_client=_FAKE_APO_CLIENT)
    result = await run_apo(
        seeds={"bot": "hi ${var}"},
        train=[{"input": "t1"}, {"input": "t2"}],
        val=[{"input": "v1"}],
        rollout=_const_rollout(0.5),
        config=config,
    )
    assert isinstance(result, OptimizeResult)
    # APO は 1 回作られ apo_client を受け取る。
    assert len(fake_apo_factory) == 1
    assert fake_apo_factory[0].kwargs["async_openai_client"] is _FAKE_APO_CLIENT
    # Trainer は 1 回作られ algorithm=APO / initial_resources にスロット名キーで PromptTemplate。
    assert len(fake_trainer_factory) == 1
    trainer = fake_trainer_factory[0]
    assert trainer.kwargs["algorithm"] is fake_apo_factory[0]
    assert set(trainer.kwargs["initial_resources"]) == {"bot"}
    assert trainer.kwargs["initial_resources"]["bot"].template == "hi {{ var }}"
    # fit が 1 回呼ばれ train/val が list 化されて渡る。
    assert len(trainer.fit_calls) == 1
    _litagent, train_list, val_list = trainer.fit_calls[0]
    assert train_list == [{"input": "t1"}, {"input": "t2"}]
    assert val_list == [{"input": "v1"}]
    # get_best_prompt から ${var} 復元したテキストが prompt に入る。
    assert result.prompt == "(optimized) hi ${var}"
    # seed には最適化前の seed テキストが prompt と同じ shape（単一 = str）で入る。
    assert result.seed == "hi ${var}"


async def test_run_apo_substitutes_vars_in_tune(
    fake_apo_factory: list[_FakeAPO],  # noqa: ARG001
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """`vars_per_slot=` 経路: `OptimizeResult.seed` / `prompt` は tune 側の `${var}` を
    `substitute_braced`（braced のみ・bare `$var` 不変）で再注入する（rollout 実体一致）。"""
    result = await run_apo(
        seeds={"bot": "hi ${var}"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.5),
        config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        vars_per_slot={"bot": {"var": "WORLD"}},
    )
    assert result.seed == "hi WORLD"
    assert result.prompt == "(optimized) hi WORLD"


async def test_run_apo_does_not_touch_bare_dollar_vars(
    fake_apo_factory: list[_FakeAPO],  # noqa: ARG001
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """braced `${var}` のみ置換し、bare `$var` には触らない（tune 側 seed の literal `$5` / `$PATH`
    は vars に同名キーがあっても置換されない）。"""
    result = await run_apo(
        seeds={"bot": "hi ${var} price=$5 shell=$name"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.5),
        config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        # `"5": "FIVE"` は intentionally dead key（PLACEHOLDER_RE は数字始まり identifier に match
        # しない）。`bare $name` も触らないことが load-bearing なアサーション対象。
        vars_per_slot={"bot": {"var": "WORLD", "name": "NG", "5": "FIVE"}},
    )
    # ${var} → WORLD（braced・置換）。$5 と $name（bare）は触らない。
    assert "hi WORLD" in str(result.seed)
    assert "price=$5" in str(result.seed)
    assert "shell=$name" in str(result.seed)


async def test_run_apo_diff_empty_when_no_change(
    fake_apo_factory: list[_FakeAPO],
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """APO が候補を採用せず seed と prompt が同一なら diff は空文字（差分なし表現）。"""
    # APO の best_template_text を seed と同じに固定する。
    fake_apo_factory.append  # noqa: B018 - 使用しないが lint 抑止用に参照
    # _FakeAPO の既定 `best_template_text` を seed と同じに揃える。
    import oai_agentspec._adapters.lightning as _lightning_mod

    # fake_apo_factory フィクスチャは monkeypatch で置換済みのため、テスト内で更に差し替えるのは
    # 複雑。ここでは _from_jinja の戻り値を seed と一致させる経路を直接確かめる代わりに
    # seed と同じ jinja を best_template_text として返す fake で run_apo を駆動する。
    captured: dict[str, Any] = {"best_text": "hi {{ var }}"}  # seed と等価（${var} 復元後）

    @dataclass
    class _NoChangeAPO:
        kwargs: dict[str, Any] = field(default_factory=dict)
        _history_best_score: float = 0.5
        _history_best_version: str = "v0"

        def get_best_prompt(self) -> Any:
            class _T:
                template = captured["best_text"]
                engine = "jinja"

            return _T()

    import pytest

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("agentlightning.APO", lambda **kwargs: _NoChangeAPO(kwargs=dict(kwargs)))
    try:
        result = await _lightning_mod.run_apo(
            seeds={"bot": "hi ${var}"},
            train=[{"input": "t"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.5),
            config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        )
    finally:
        monkeypatch.undo()
    # seed == prompt のため diff は空文字。
    assert result.seed == result.prompt
    assert result.diff == ""


async def test_run_apo_multi_seed_seed_field_is_dict(
    fake_apo_factory: list[_FakeAPO],  # noqa: ARG001
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """複数 seed では `OptimizeResult.seed` も dict（prompt と同じ shape）になる。"""
    # _FakeAPO の既定 best_template_text は `{{ var }}` を含むため、seed 側 placeholder も `${var}`
    # に揃える（不一致だと placeholder 喪失 fallback の warn が発生する）。
    result = await run_apo(
        seeds={"a": "seed-a ${var}", "b": "seed-b ${var}"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.5),
        config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
    )
    assert isinstance(result.seed, dict)
    assert result.seed == {"a": "seed-a ${var}", "b": "seed-b ${var}"}
    # prompt も dict shape（複数 slot のため）。
    assert isinstance(result.prompt, dict)
    assert set(result.prompt) == {"a", "b"}


async def test_run_apo_passes_apo_kwargs_from_config(
    fake_apo_factory: list[_FakeAPO],
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001 - fixture 副作用のみ
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """OptimizeConfig の APO 固有設定が APO 引数へ passthrough される（rounds は beam_rounds）。"""
    config = OptimizeConfig(
        apo_client=_FAKE_APO_CLIENT,
        apo_gradient_model="gpt-grad",
        apo_apply_edit_model="gpt-edit",
        apo_beam_width=5,
        apo_branch_factor=3,
        rounds=7,
    )
    await run_apo(
        seeds={"bot": "s"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.0),
        config=config,
    )
    kwargs = fake_apo_factory[0].kwargs
    assert kwargs["gradient_model"] == "gpt-grad"
    assert kwargs["apply_edit_model"] == "gpt-edit"
    assert kwargs["beam_width"] == 5
    assert kwargs["branch_factor"] == 3
    # rounds は beam_rounds 名前空間にマップ。
    assert kwargs["beam_rounds"] == 7


async def test_run_apo_passes_trainer_kwargs_from_config(
    fake_apo_factory: list[_FakeAPO],  # noqa: ARG001
    fake_trainer_factory: list[_FakeTrainer],
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """OptimizeConfig の store は Trainer 引数へ・concurrency は strategy.n_runners へ passthrough。

    新挙動: 既定で `SharedMemoryExecutionStrategy` を Trainer.strategy に渡す（macOS spawn pickling
    回避）。`concurrency` は `strategy.n_runners` 経由で並列度を表現する（n_runners 直接 kwargs で
    なく strategy 経由）。
    """
    sentinel_store = object()
    config = OptimizeConfig(
        apo_client=_FAKE_APO_CLIENT,
        store=sentinel_store,
        concurrency=3,
    )
    await run_apo(
        seeds={"bot": "s"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.0),
        config=config,
    )
    tkwargs = fake_trainer_factory[0].kwargs
    assert tkwargs["store"] is sentinel_store
    # concurrency は strategy.n_runners 経由（n_runners 直接 kwargs ではなく）。
    strategy = tkwargs["strategy"]
    assert strategy.n_runners == 3
    assert strategy.main_thread == "algorithm"  # 常に algorithm（runner thread の協調停止のため）


async def test_run_apo_omits_unspecified_kwargs(
    fake_apo_factory: list[_FakeAPO],
    fake_trainer_factory: list[_FakeTrainer],
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """None 設定は APO に渡さない / Trainer は既定 strategy（n_runners=1・runner main）を渡す。"""
    config = OptimizeConfig(apo_client=_FAKE_APO_CLIENT)  # 他は全て None
    await run_apo(
        seeds={"bot": "s"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.0),
        config=config,
    )
    apo_kwargs = fake_apo_factory[0].kwargs
    # async_openai_client + APO モデル名既定（gpt-5.4-mini）が渡る。他の None 項目はキー無し。
    assert set(apo_kwargs) == {"async_openai_client", "gradient_model", "apply_edit_model"}
    assert apo_kwargs["gradient_model"] == "gpt-5.4-mini"
    assert apo_kwargs["apply_edit_model"] == "gpt-5.4-mini"
    trainer_kwargs = fake_trainer_factory[0].kwargs
    # algorithm / initial_resources / strategy / adapter / tracer は必須（store/n_runners は無い）。
    assert set(trainer_kwargs) == {
        "algorithm",
        "initial_resources",
        "strategy",
        "adapter",
        "tracer",
    }
    # concurrency 未指定 → 既定 strategy（n_runners=1, main_thread="algorithm"）。
    # main_thread="algorithm" なら algorithm 完了時に runner thread を協調停止できる
    # （"runner" だと runner が無限に待ち続け終了しない）。
    strategy = trainer_kwargs["strategy"]
    assert strategy.n_runners == 1
    assert strategy.main_thread == "algorithm"
    # APO は TraceToMessages 必須（既定 TracerTraceToTriplet だと実行時失敗）。
    from agentlightning.adapter import TraceToMessages

    assert isinstance(trainer_kwargs["adapter"], TraceToMessages)
    # tracer 既定: agent-lightning の既定構成（agentops_managed=True / instrument_managed=True）。
    # agentops_managed=False は AgentOps の TracerProvider 未初期化で実行時に落ちるため使わない。
    from agentlightning.tracer import AgentOpsTracer

    tracer = trainer_kwargs["tracer"]
    assert isinstance(tracer, AgentOpsTracer)
    assert tracer.agentops_managed is True
    assert tracer.instrument_managed is True


async def test_silences_agentops_logs_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
    fake_apo_factory: list[_FakeAPO],  # noqa: ARG001
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """`AGENTOPS_API_KEY` 未設定時は agentops のログ / ファイル出力を抑制する env を入れる。

    本物のキーが無ければ AgentOps クラウドへ silent fail するだけなので、warning や Session
    Replay URL は不要。本物のキーが入っているときは触らず通常レベルで動かす（意図尊重）。
    """
    monkeypatch.delenv("AGENTOPS_API_KEY", raising=False)
    monkeypatch.delenv("AGENTOPS_LOG_LEVEL", raising=False)
    monkeypatch.delenv("AGENTOPS_LOGGING_TO_FILE", raising=False)

    config = OptimizeConfig(apo_client=_FAKE_APO_CLIENT)
    await run_apo(
        seeds={"bot": "s"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.0),
        config=config,
    )
    import os

    # API キー未設定時のみ env を setdefault する。
    assert os.environ.get("AGENTOPS_LOG_LEVEL") == "ERROR"
    assert os.environ.get("AGENTOPS_LOGGING_TO_FILE") == "False"


async def test_does_not_touch_agentops_env_when_api_key_present(
    monkeypatch: pytest.MonkeyPatch,
    fake_apo_factory: list[_FakeAPO],  # noqa: ARG001
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """`AGENTOPS_API_KEY` 設定時は agentops 関連 env を勝手に上書きしない（利用者意図尊重）。"""
    monkeypatch.setenv("AGENTOPS_API_KEY", "real-key")
    monkeypatch.delenv("AGENTOPS_LOG_LEVEL", raising=False)
    monkeypatch.delenv("AGENTOPS_LOGGING_TO_FILE", raising=False)

    config = OptimizeConfig(apo_client=_FAKE_APO_CLIENT)
    await run_apo(
        seeds={"bot": "s"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.0),
        config=config,
    )
    import os

    # API キー有り時は env を一切いじらない（agentops の通常動作）。
    assert "AGENTOPS_LOG_LEVEL" not in os.environ
    assert "AGENTOPS_LOGGING_TO_FILE" not in os.environ


async def test_attaches_agentlightning_console_handler_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
    fake_apo_factory: list[_FakeAPO],  # noqa: ARG001
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """API キー未設定時は agentlightning ロガーに console handler を取り付け、INFO 以上で出力する
    （rollout 進捗が利用者から見えるようにする）。重複登録は防ぐ。"""
    import logging as _logging

    monkeypatch.delenv("AGENTOPS_API_KEY", raising=False)

    # フラグ / 既存 handler をリセット（テスト独立性）。
    import oai_agentspec._adapters.lightning as _lightning_mod

    monkeypatch.setattr(_lightning_mod, "_AGENTLIGHTNING_CONSOLE_HANDLER_ATTACHED", False)
    al_logger = _logging.getLogger("agentlightning")
    original_handlers = list(al_logger.handlers)
    original_level = al_logger.level
    for h in original_handlers:
        al_logger.removeHandler(h)
    al_logger.setLevel(_logging.NOTSET)
    try:
        config = OptimizeConfig(apo_client=_FAKE_APO_CLIENT)
        await run_apo(
            seeds={"bot": "s"},
            train=[{"input": "t"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.0),
            config=config,
        )
        # handler が 1 個追加され、レベルが INFO 以下になっている。
        stream_handlers = [
            h
            for h in al_logger.handlers
            if isinstance(h, _logging.StreamHandler) and not isinstance(h, _logging.FileHandler)
        ]
        assert len(stream_handlers) == 1
        assert al_logger.level <= _logging.INFO
        # 2 度目の run_apo では重複登録しない。
        await run_apo(
            seeds={"bot": "s"},
            train=[{"input": "t"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.0),
            config=config,
        )
        stream_handlers_after = [
            h
            for h in al_logger.handlers
            if isinstance(h, _logging.StreamHandler) and not isinstance(h, _logging.FileHandler)
        ]
        assert len(stream_handlers_after) == 1
    finally:
        for h in list(al_logger.handlers):
            al_logger.removeHandler(h)
        for h in original_handlers:
            al_logger.addHandler(h)
        al_logger.setLevel(original_level)


async def test_console_logging_preserves_explicit_logger_level(
    monkeypatch: pytest.MonkeyPatch,
    fake_apo_factory: list[_FakeAPO],  # noqa: ARG001
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """利用者が `logging.getLogger("agentlightning").setLevel(logging.WARNING)` で明示的に
    mute している場合、`_setup_agentlightning_console_logging` はその level を尊重して INFO に
    引き下げない（Codex P3）。"""
    import logging as _logging

    monkeypatch.delenv("AGENTOPS_API_KEY", raising=False)

    import oai_agentspec._adapters.lightning as _lightning_mod

    monkeypatch.setattr(_lightning_mod, "_AGENTLIGHTNING_CONSOLE_HANDLER_ATTACHED", False)
    al_logger = _logging.getLogger("agentlightning")
    original_handlers = list(al_logger.handlers)
    original_level = al_logger.level
    for h in original_handlers:
        al_logger.removeHandler(h)
    # 利用者が明示的に WARNING に設定（progress を mute したい意図）。
    al_logger.setLevel(_logging.WARNING)
    try:
        await run_apo(
            seeds={"bot": "s"},
            train=[{"input": "t"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.0),
            config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        )
        # handler は追加されるが、level は WARNING のまま（INFO へ下げない）。
        assert al_logger.level == _logging.WARNING
    finally:
        for h in list(al_logger.handlers):
            al_logger.removeHandler(h)
        for h in original_handlers:
            al_logger.addHandler(h)
        al_logger.setLevel(original_level)


async def test_user_explicit_log_level_is_preserved_even_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    fake_apo_factory: list[_FakeAPO],  # noqa: ARG001
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """API キー未設定でも、利用者が `AGENTOPS_LOG_LEVEL` を明示していれば setdefault で上書きしない
    （unmute 経路）。"""
    monkeypatch.delenv("AGENTOPS_API_KEY", raising=False)
    monkeypatch.setenv("AGENTOPS_LOG_LEVEL", "DEBUG")  # 利用者が明示

    config = OptimizeConfig(apo_client=_FAKE_APO_CLIENT)
    await run_apo(
        seeds={"bot": "s"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.0),
        config=config,
    )
    import os

    assert os.environ.get("AGENTOPS_LOG_LEVEL") == "DEBUG"  # 利用者の値を保持


async def test_run_apo_passes_explicit_tracer_through(
    fake_apo_factory: list[_FakeAPO],  # noqa: ARG001
    fake_trainer_factory: list[_FakeTrainer],
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """`OptimizeConfig.tracer` を明示すれば既定 AgentOpsTracer を使わずそれが Trainer に渡る。"""
    sentinel_tracer = object()
    config = OptimizeConfig(apo_client=_FAKE_APO_CLIENT, tracer=sentinel_tracer)
    await run_apo(
        seeds={"bot": "s"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.0),
        config=config,
    )
    trainer_kwargs = fake_trainer_factory[0].kwargs
    assert trainer_kwargs["tracer"] is sentinel_tracer


async def test_run_apo_multi_seed_sequential_per_slot(
    fake_apo_factory: list[_FakeAPO],
    fake_trainer_factory: list[_FakeTrainer],
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """複数 seed は順次最適化される（seed 数だけ Trainer/APO が作られる）。

    initial_resources は各ラウンドで対象スロットのみを持つ。
    """
    # 各スロットで best が異なる候補を返す（呼び出し順で best_template_text を差し替え）。
    bests = iter(["best_a {{ var }}", "best_b {{ var }}"])

    original_apo_factory = fake_apo_factory  # reference for closure

    def _on_create(**kwargs: Any) -> _FakeAPO:
        inst = _FakeAPO(kwargs=dict(kwargs), best_template_text=next(bests))
        original_apo_factory.append(inst)
        return inst

    # APO factory を差し替え（fixture が登録した factory を上書き）。
    import agentlightning

    agentlightning.APO = _on_create  # type: ignore[assignment]

    result = await run_apo(
        seeds={"a": "seed_a ${var}", "b": "seed_b ${var}"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.0),
        config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
    )
    # 2 スロット分 APO/Trainer が順次生成された。
    assert len(fake_apo_factory) == 2
    assert len(fake_trainer_factory) == 2
    # 各 Trainer の initial_resources のキーは対象スロットのみ（順序保存）。
    assert set(fake_trainer_factory[0].kwargs["initial_resources"]) == {"a"}
    assert set(fake_trainer_factory[1].kwargs["initial_resources"]) == {"b"}
    # prompt は mapping で各スロットに best が入る（jinja → ${var} 復元）。
    assert result.prompt == {"a": "best_a ${var}", "b": "best_b ${var}"}


async def test_run_apo_records_history_per_slot(
    fake_apo_factory: list[_FakeAPO],  # noqa: ARG001
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """history には各スロットの best_score / best_version が記録される。"""
    result = await run_apo(
        seeds={"bot": "s"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.0),
        config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
    )
    assert len(result.history) == 1
    entry = result.history[0]
    assert entry["slot"] == "bot"
    assert entry["best_score"] == pytest.approx(0.85)
    assert entry["best_version"] == "v1"
    # happy path（fallback が起きていない）では `placeholder_fallback` は False で、利用者が
    # 「APO スコアが本物か / seed フォールバックが起きたか」を programmatic に区別できる
    # （Codex 第4 round の dual coverage 強化）。
    assert entry["placeholder_fallback"] is False


async def test_run_apo_records_history_normalizes_inf(
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_history_best_score` が `-inf`（全ラウンド未更新）なら history.best_score は None。"""
    inf_apo_holder: list[_FakeAPO] = []

    def _inf_factory(**kwargs: Any) -> _FakeAPO:
        inst = _FakeAPO(kwargs=dict(kwargs))
        inst._history_best_score = float("-inf")  # noqa: SLF001 - テスト用直接設定
        inf_apo_holder.append(inst)
        return inst

    monkeypatch.setattr("agentlightning.APO", _inf_factory, raising=True)

    result = await run_apo(
        seeds={"bot": "s"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.0),
        config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
    )
    # `-inf` は JSON 非互換 / 利用者誤読防止のため None に正規化される。
    assert result.history[0]["best_score"] is None


async def test_run_apo_train_val_recomputed_after_fit(
    fake_apo_factory: list[_FakeAPO],  # noqa: ARG001
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """fit 完了後に合成候補で train / val を再計算する（rollout 平均が train/val_score）。"""
    # 可変 score で train が 2 件・val が 1 件を想定。LitAgent rollout（emit_reward）で消費される
    # はずだが、fake fit はそれを呼ばないため最終 _score_candidate が rollout を train + val で
    # 3 回呼ぶ。
    scores = iter([0.2, 0.4, 0.9])

    async def _rollout(candidate: dict[str, str], case: Any) -> float:
        return next(scores)

    result = await run_apo(
        seeds={"bot": "s"},
        train=[{"input": "t1"}, {"input": "t2"}],
        val=[{"input": "v1"}],
        rollout=_rollout,
        config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
    )
    # train_score = (0.2 + 0.4)/2 = 0.3、val_score = 0.9。
    assert result.train_score == pytest.approx(0.3)
    assert result.val_score == pytest.approx(0.9)


# ----------------------------------------------------------------------
# _make_litagent（rollout_async / emit_reward 契約・SDK 結合の境界）
# ----------------------------------------------------------------------


async def test_litagent_rollout_extracts_candidate_and_emits_reward(
    captured_rewards: list[float],
) -> None:
    """rollout_async: resources から候補抽出 → oai-agentspec rollout → emit_reward(score)。"""
    seen_candidates: list[dict[str, str]] = []
    seen_tasks: list[Any] = []

    async def _rollout(candidate: dict[str, str], case: Any) -> float:
        seen_candidates.append(dict(candidate))
        seen_tasks.append(case)
        return 0.42

    litagent = _make_litagent(
        target_slot="triage",
        seeds={"triage": "seed_triage", "billing": "seed_billing"},
        rollout=_rollout,
    )
    resources = {"triage": _FakePromptTemplate(template="optimized {{ var }} hi")}
    await litagent.rollout_async({"input": "x"}, resources, rollout_obj=object())

    # 候補は target_slot のみ更新・他は seeds そのまま。${var} へ jinja から復元。
    assert seen_candidates == [{"triage": "optimized ${var} hi", "billing": "seed_billing"}]
    assert seen_tasks == [{"input": "x"}]
    # emit_reward が float で呼ばれる。
    assert captured_rewards == [pytest.approx(0.42)]


async def test_litagent_rollout_exception_emits_zero(
    captured_rewards: list[float],
) -> None:
    """rollout 例外時は emit_reward(0.0) で継続（fail-closed・最適化は止めない）。"""

    async def _failing(candidate: dict[str, str], case: Any) -> float:
        raise RuntimeError("rollout crashed")

    litagent = _make_litagent(target_slot="bot", seeds={"bot": "s"}, rollout=_failing)
    resources = {"bot": _FakePromptTemplate(template="any {{ var }}")}
    # 例外は捕捉され外には漏れない。
    await litagent.rollout_async({"input": "x"}, resources, rollout_obj=object())
    assert captured_rewards == [pytest.approx(0.0)]


async def test_litagent_rollout_reraises_optimize_error(
    captured_rewards: list[float],
) -> None:
    """rollout が `OptimizeError`（安全違反 NFR-8 など）を送出すると握り潰さず再 raise する。

    `_build_decisions` の安全違反は `OptimizeError(CONFIG_MISSING)` で送出される。汎用
    `except Exception` で握り潰されると FR-8 の構造化失敗種別が消えるため、`OptimizeError` のみ
    例外的に再 raise して Trainer / `asyncio.to_thread` / `optimize` まで伝搬させる。
    """
    from oai_agentspec.runtime.lightning import FailureKind, OptimizeError

    async def _raise_optimize_error(candidate: dict[str, str], case: Any) -> float:
        raise OptimizeError(FailureKind.CONFIG_MISSING, "未差し替えツールへの approve")

    litagent = _make_litagent(target_slot="bot", seeds={"bot": "s"}, rollout=_raise_optimize_error)
    resources = {"bot": _FakePromptTemplate(template="any {{ var }}")}

    with pytest.raises(OptimizeError) as exc:
        await litagent.rollout_async({"input": "x"}, resources, rollout_obj=object())
    assert exc.value.kind == FailureKind.CONFIG_MISSING
    # 安全違反は emit_reward に倒れず、Trainer 経路まで伝搬する（emit されない）。
    assert captured_rewards == []


async def test_run_apo_falls_back_to_seed_when_best_drops_placeholder(
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """APO 最良候補が seed の `${var}` を喪失していたら seed にフォールバック + warn する
    （契約: 「最適化済みテキストは `${var}` を保持する」を fail-closed で守る）。"""

    @dataclass
    class _DropVarAPO:
        kwargs: dict[str, Any] = field(default_factory=dict)
        _history_best_score: float = 0.5
        _history_best_version: str = "v1"

        def get_best_prompt(self) -> _FakePromptTemplate:
            # seed には `${var}` があるが best 候補は喪失（jinja {{ var }} を含まない）。
            return _FakePromptTemplate(template="(optimized) hi", engine="jinja")

    monkeypatch.setattr(
        "agentlightning.APO", lambda **kwargs: _DropVarAPO(kwargs=dict(kwargs)), raising=True
    )

    with pytest.warns(RuntimeWarning, match="`\\${var}` を喪失"):
        result = await run_apo(
            seeds={"bot": "hi ${var}"},
            train=[{"input": "t"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.5),
            config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        )
    # フォールバック: best_text が seed と一致 → diff も空 / prompt は seed そのもの。
    assert result.prompt == "hi ${var}"
    assert result.seed == "hi ${var}"
    assert result.diff == ""
    # history: 破棄された候補の score / version が誤って残らないこと（Codex 第3 round）。
    # placeholder_fallback フラグが True で、best_score / best_version は None に上書き。
    entry = result.history[0]
    assert entry["placeholder_fallback"] is True
    assert entry["best_score"] is None
    assert entry["best_version"] is None


async def test_run_apo_single_slot_normal_placeholder_missing_falls_back_to_seed(
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """回帰確認: 通常 placeholder（`oas_boundary_` 接頭辞なし）が best から欠落したら seed
    フォールバックする（post-fit フォールバック判定の既存挙動・T8 の拡張前提が壊れていない）。"""

    @dataclass
    class _DropOtherVarAPO:
        kwargs: dict[str, Any] = field(default_factory=dict)
        _history_best_score: float = 0.5
        _history_best_version: str = "v1"

        def get_best_prompt(self) -> _FakePromptTemplate:
            # seed には `${other_var}` があるが best 候補は喪失。
            return _FakePromptTemplate(template="(optimized) hi", engine="jinja")

    monkeypatch.setattr(
        "agentlightning.APO", lambda **kwargs: _DropOtherVarAPO(kwargs=dict(kwargs)), raising=True
    )

    with pytest.warns(RuntimeWarning, match="`\\${var}` を喪失"):
        result = await run_apo(
            seeds={"bot": "hi ${other_var}"},
            train=[{"input": "t"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.5),
            config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        )
    assert result.prompt == "hi ${other_var}"
    assert result.history[0]["placeholder_fallback"] is True


async def test_run_apo_single_slot_boundary_marker_missing_falls_back(
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """予約境界マーカー `${oas_boundary_1}`（exact-once 検査対象）が best から完全に欠落した
    場合も seed フォールバックする（既存の存在検査で既にカバーされる経路・回帰確認）。"""

    @dataclass
    class _DropBoundaryAPO:
        kwargs: dict[str, Any] = field(default_factory=dict)
        _history_best_score: float = 0.5
        _history_best_version: str = "v1"

        def get_best_prompt(self) -> _FakePromptTemplate:
            # seed には `${oas_boundary_1}` があるが best 候補では欠落。
            return _FakePromptTemplate(template="(optimized) start end", engine="jinja")

    monkeypatch.setattr(
        "agentlightning.APO", lambda **kwargs: _DropBoundaryAPO(kwargs=dict(kwargs)), raising=True
    )

    with pytest.warns(RuntimeWarning, match="`\\${var}` を喪失"):
        result = await run_apo(
            seeds={"bot": "start ${oas_boundary_1} end"},
            train=[{"input": "t"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.5),
            config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        )
    assert result.prompt == "start ${oas_boundary_1} end"
    assert result.history[0]["placeholder_fallback"] is True


async def test_run_apo_single_slot_boundary_marker_duplicated_falls_back(
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新規契約（T8）: 予約境界マーカー `${oas_boundary_1}` は seed と同じ回数（exact-once）で
    best にも存在すべき。best で重複（2 回以上）していたら、存在はしていても seed フォールバック
    対象に含める（現状は set ベースの存在検査のみのため未検出＝RED）。"""

    @dataclass
    class _DuplicateBoundaryAPO:
        kwargs: dict[str, Any] = field(default_factory=dict)
        _history_best_score: float = 0.5
        _history_best_version: str = "v1"

        def get_best_prompt(self) -> _FakePromptTemplate:
            # seed には `${oas_boundary_1}` が 1 回のみだが best では 2 回重複している。
            return _FakePromptTemplate(
                template="(optimized) start {{ oas_boundary_1 }} mid {{ oas_boundary_1 }} end",
                engine="jinja",
            )

    monkeypatch.setattr(
        "agentlightning.APO",
        lambda **kwargs: _DuplicateBoundaryAPO(kwargs=dict(kwargs)),
        raising=True,
    )

    with pytest.warns(RuntimeWarning, match="`\\${var}` を喪失"):
        result = await run_apo(
            seeds={"bot": "start ${oas_boundary_1} end"},
            train=[{"input": "t"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.5),
            config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        )
    # 重複していた best は破棄され seed にフォールバックする（境界マーカーの exact-once 契約）。
    assert result.prompt == "start ${oas_boundary_1} end"
    assert result.history[0]["placeholder_fallback"] is True


async def test_run_apo_single_slot_boundary_marker_order_swap_falls_back(
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FU C2 fix: seed と best のマーカー出現順が入れ替わっている場合も fallback を発火させる。

    従来の count 一致検査（`str.count`）は order-agnostic のため、seed の
    `${oas_boundary_1}...${oas_boundary_2}` に対し best 側で 2 が先・1 が後の swap が
    count 一致で素通ししていた。post-fit fallback を `boundary_intact`（連番列の順序込み比較）
    に一本化することで、silent に literal マーカーが `OptimizeResult` に漏れるのを防ぐ。"""

    @dataclass
    class _OrderSwapBoundaryAPO:
        kwargs: dict[str, Any] = field(default_factory=dict)
        _history_best_score: float = 0.5
        _history_best_version: str = "v1"

        def get_best_prompt(self) -> _FakePromptTemplate:
            # seed は 1 -> 2 の順・best は 2 -> 1 の順で swap。count 一致・order 不整合。
            return _FakePromptTemplate(
                template="A {{ oas_boundary_2 }} B {{ oas_boundary_1 }} C",
                engine="jinja",
            )

    monkeypatch.setattr(
        "agentlightning.APO",
        lambda **kwargs: _OrderSwapBoundaryAPO(kwargs=dict(kwargs)),
        raising=True,
    )

    with pytest.warns(RuntimeWarning, match="`\\${var}` を喪失"):
        result = await run_apo(
            seeds={"bot": "A ${oas_boundary_1} B ${oas_boundary_2} C"},
            train=[{"input": "t"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.5),
            config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        )
    assert result.prompt == "A ${oas_boundary_1} B ${oas_boundary_2} C"
    assert result.history[0]["placeholder_fallback"] is True


async def test_run_apo_single_slot_boundary_marker_exact_once_passes(
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """seed / best とも `${oas_boundary_1}` がちょうど 1 回ずつであれば exact-once を満たすため
    フォールバックせず best 候補をそのまま採用する。"""

    @dataclass
    class _ExactOnceBoundaryAPO:
        kwargs: dict[str, Any] = field(default_factory=dict)
        _history_best_score: float = 0.5
        _history_best_version: str = "v1"

        def get_best_prompt(self) -> _FakePromptTemplate:
            return _FakePromptTemplate(
                template="(optimized) start {{ oas_boundary_1 }} end", engine="jinja"
            )

    monkeypatch.setattr(
        "agentlightning.APO",
        lambda **kwargs: _ExactOnceBoundaryAPO(kwargs=dict(kwargs)),
        raising=True,
    )

    result = await run_apo(
        seeds={"bot": "start ${oas_boundary_1} end"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.5),
        config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
    )
    # フォールバックせず best 候補（optimized 済みテキスト）が採用される。
    assert result.prompt == "(optimized) start ${oas_boundary_1} end"
    assert result.history[0]["placeholder_fallback"] is False


async def test_run_apo_warns_when_history_attr_missing(
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """agent-lightning の private 属性 `_history_best_score` が欠落した場合、warnings で顕在化する
    （API rename の silent failure を防ぐ・version pin の補助）。"""

    @dataclass
    class _RenamedAPO:
        kwargs: dict[str, Any] = field(default_factory=dict)

        # `_history_best_score` / `_history_best_version` を意図的に持たない（rename を再現）。

        def get_best_prompt(self) -> _FakePromptTemplate:
            return _FakePromptTemplate(template="x {{ var }}", engine="jinja")

    monkeypatch.setattr(
        "agentlightning.APO", lambda **kwargs: _RenamedAPO(kwargs=dict(kwargs)), raising=True
    )

    with pytest.warns(RuntimeWarning, match="_history_best_score"):
        result = await run_apo(
            seeds={"bot": "s"},
            train=[{"input": "t"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.0),
            config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        )
    # 属性欠落でも history.best_score は None で続行（fail-soft）。
    assert result.history[0]["best_score"] is None


async def test_litagent_records_optimize_error_in_sentinel(
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """`OptimizeError` 発生時、`LitAgent.critical_error` sentinel に保持される。

    agent-lightning の worker thread が catch-all で例外を握り潰した場合、re-raise だけでは
    呼び出し側へ伝搬しない可能性がある。post-fit で sentinel を check できるよう保持する。
    """
    from oai_agentspec.runtime.lightning import FailureKind, OptimizeError

    async def _raise_optimize_error(candidate: dict[str, str], case: Any) -> float:
        raise OptimizeError(FailureKind.CONFIG_MISSING, "未差し替えツールへの approve")

    litagent = _make_litagent(target_slot="bot", seeds={"bot": "s"}, rollout=_raise_optimize_error)
    resources = {"bot": _FakePromptTemplate(template="any {{ var }}")}

    with pytest.raises(OptimizeError):
        await litagent.rollout_async({"input": "x"}, resources, rollout_obj=object())

    # sentinel に保持される（worker thread が swallow してもこれで検出可能）。
    assert litagent.critical_error is not None
    assert litagent.critical_error.kind == FailureKind.CONFIG_MISSING


async def test_run_apo_reraises_swallowed_optimize_error(
    fake_apo_factory: list[_FakeAPO],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """worker thread が `OptimizeError` を握り潰しても run_apo は post-fit に sentinel から
    検出して再 raise する（NFR-8 fail-closed 維持）。

    agent-lightning の SharedMemoryExecutionStrategy worker thread が rollout 例外を catch-all で
    ログに落として続行するシナリオを `_SwallowingTrainer` で再現する。
    `_OaiAgentSpecLitAgent.rollout_async` が raise する `OptimizeError` を `fit` 内で
    swallow しても、
    `critical_error` sentinel が保持されており `run_apo` が post-fit でそれを check して再 raise
    することを検証する。
    """
    from oai_agentspec.runtime.lightning import FailureKind, OptimizeError

    @dataclass
    class _SwallowingTrainer:
        kwargs: dict[str, Any] = field(default_factory=dict)

        def fit(
            self,
            litagent: Any,
            train_dataset: list[Any],
            *,
            val_dataset: Any = None,  # noqa: ARG002
        ) -> None:
            """rollout_async を呼んで例外を握り潰す（agent-lightning worker の挙動を再現）。"""
            import asyncio as _asyncio

            for case in train_dataset:
                try:
                    _asyncio.run(
                        litagent.rollout_async(
                            case,
                            {"bot": _FakePromptTemplate(template="any")},
                            object(),
                        )
                    )
                except Exception:
                    # worker thread が catch-all で握り潰すシナリオ（fail-open リスク）。
                    pass

    monkeypatch.setattr(
        "agentlightning.Trainer", lambda **kwargs: _SwallowingTrainer(kwargs=dict(kwargs))
    )

    async def _raise_optimize_error(candidate: dict[str, str], case: Any) -> float:
        raise OptimizeError(FailureKind.CONFIG_MISSING, "未差し替えツールへの approve")

    with pytest.raises(OptimizeError) as exc:
        await run_apo(
            seeds={"bot": "s"},
            train=[{"input": "t"}],
            val=[{"input": "v"}],
            rollout=_raise_optimize_error,
            config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


# ----------------------------------------------------------------------
# judge_score: 最小エージェント 1 ターン実行 + _parse_score
# ----------------------------------------------------------------------


async def test_judge_score_parses_model_output() -> None:
    """judge_score は最小エージェントを 1 ターン実行し出力数値を 0.0..1.0 で返す。"""
    model = FakeModel().queue_text("0.8")
    score = await judge_score(rubric="be concise", model=model, output="answer", case={"id": 1})
    assert score == pytest.approx(0.8)


async def test_judge_score_no_number_returns_zero() -> None:
    """採点出力に数値が無ければ 0.0（fail-closed）。"""
    model = FakeModel().queue_text("no number here")
    score = await judge_score(rubric="r", model=model, output="o", case={})
    assert score == pytest.approx(0.0)


async def test_judge_score_clamps_above_one() -> None:
    """1.0 超の採点出力は 1.0 にクランプする。"""
    model = FakeModel().queue_text("score: 5")
    score = await judge_score(rubric="r", model=model, output="o", case={})
    assert score == pytest.approx(1.0)


# ----------------------------------------------------------------------
# _parse_score: 数値抽出 / クランプ / 抽出不能
# ----------------------------------------------------------------------


@pytest.mark.unit
def test_parse_score_plain_number() -> None:
    """先頭の数値をそのまま 0.0..1.0 範囲で抽出する。"""
    assert _parse_score("0.7") == pytest.approx(0.7)


@pytest.mark.unit
def test_parse_score_embedded_in_text() -> None:
    """テキスト中の最初の数値を抽出する（'score: 0.7'）。"""
    assert _parse_score("score: 0.7 (good)") == pytest.approx(0.7)


@pytest.mark.unit
def test_parse_score_clamps_above_one() -> None:
    """1.0 を超える値は 1.0 にクランプする。"""
    assert _parse_score("1.5") == pytest.approx(1.0)


@pytest.mark.unit
def test_parse_score_clamps_below_zero() -> None:
    """負値は 0.0 にクランプする。"""
    assert _parse_score("-0.3") == pytest.approx(0.0)


@pytest.mark.unit
def test_parse_score_no_number_returns_zero() -> None:
    """数値を含まないテキストは 0.0。"""
    assert _parse_score("not a score") == pytest.approx(0.0)


@pytest.mark.unit
def test_parse_score_empty_returns_zero() -> None:
    """空文字は 0.0。"""
    assert _parse_score("") == pytest.approx(0.0)


@pytest.mark.unit
def test_parse_score_integer() -> None:
    """整数表記（"1"）も抽出してクランプ範囲内で返す。"""
    assert _parse_score("1") == pytest.approx(1.0)
    assert _parse_score("0") == pytest.approx(0.0)


# --- run_apo: 途中失敗時の部分成果保全（X1: slot ループ内 / X2: スコア再計算段） -----------


def _no_extra_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_require_agentlightning` を no-op 化する（cold-start import の時間を外すため）。"""
    monkeypatch.setattr(
        "oai_agentspec._adapters.lightning._require_agentlightning", lambda: None, raising=True
    )


def _entry(slot: str) -> dict[str, Any]:
    """最小の HistoryEntry dict を作る。"""
    return {"slot": slot, "best_score": 0.8, "best_version": 1, "placeholder_fallback": False}


async def test_run_apo_slot_failure_preserves_completed_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """slot 2 の失敗時、slot 1 の最良テキスト（vars 再注入済み）と履歴が partial に残る。

    完了済み slot の最適化は API コストを払って完了しており、後続 slot の失敗で
    ローカル変数ごと全損すると利用者は診断も救出もできない（pre-flight の案 B と同じ動機）。
    """
    from oai_agentspec.runtime.lightning import FailureKind, OptimizeError

    _no_extra_check(monkeypatch)

    async def _single(*, slot_name: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        if slot_name == "billing":
            raise RuntimeError("boom")
        return ("T-BEST ${company}", _entry(slot_name))

    monkeypatch.setattr(
        "oai_agentspec._adapters.lightning._run_apo_single_slot", _single, raising=True
    )

    with pytest.raises(OptimizeError) as exc_info:
        await run_apo(
            seeds={"triage": "t ${company}", "billing": "b"},
            train=[{"input": "x"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.5),
            config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
            vars_per_slot={"triage": {"company": "ACME"}},
        )

    exc = exc_info.value
    assert exc.kind is FailureKind.TRAINER_FAILED
    assert exc.partial is not None
    assert exc.partial.completed_slots == {"triage": "T-BEST ACME"}
    assert [e["slot"] for e in exc.partial.history] == ["triage"]
    assert exc.partial.failed_slot == "billing"
    assert "slot 2/2" in exc.message
    assert "'billing'" in exc.message
    assert "RuntimeError" in exc.message
    assert "boom" in exc.message
    assert "error.partial" in exc.message
    assert isinstance(exc.__cause__, RuntimeError)


async def test_run_apo_first_slot_failure_has_no_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """先頭 slot の失敗は保全対象がなく partial=None（非 None = 成果ありの契約）。

    本文が空の例外（`TimeoutError()`）でも型名がメッセージに残ることも併せて固定する。
    """
    from oai_agentspec.runtime.lightning import OptimizeError

    _no_extra_check(monkeypatch)

    async def _single(*, slot_name: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        raise TimeoutError()

    monkeypatch.setattr(
        "oai_agentspec._adapters.lightning._run_apo_single_slot", _single, raising=True
    )

    with pytest.raises(OptimizeError) as exc_info:
        await run_apo(
            seeds={"triage": "t"},
            train=[{"input": "x"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.5),
            config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        )

    exc = exc_info.value
    assert exc.partial is None
    assert "TimeoutError" in exc.message
    assert not exc.message.endswith(": ")
    assert "error.partial" not in exc.message


async def test_run_apo_slot_failure_optimize_error_passes_through_with_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """slot 途中の `OptimizeError` は kind / message を保ったまま partial だけ後付けされる。

    再ラップすると kind が TRAINER_FAILED へ変質し fail-closed の診断（NFR-8 の
    CONFIG_MISSING 等）が失われる。属性後付けなら既存契約を保ちながら成果を保全できる。
    """
    from oai_agentspec.runtime.lightning import FailureKind, OptimizeError

    _no_extra_check(monkeypatch)

    async def _single(*, slot_name: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        if slot_name == "billing":
            raise OptimizeError(FailureKind.CONFIG_MISSING, "承認済みツールがモック未差し替えです")
        return ("T-BEST", _entry(slot_name))

    monkeypatch.setattr(
        "oai_agentspec._adapters.lightning._run_apo_single_slot", _single, raising=True
    )

    with pytest.raises(OptimizeError) as exc_info:
        await run_apo(
            seeds={"triage": "t", "billing": "b"},
            train=[{"input": "x"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.5),
            config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        )

    exc = exc_info.value
    assert exc.kind is FailureKind.CONFIG_MISSING
    assert exc.message == "承認済みツールがモック未差し替えです"
    assert exc.partial is not None
    assert exc.partial.completed_slots == {"triage": "T-BEST"}
    assert exc.partial.failed_slot == "billing"


async def test_run_apo_score_failure_preserves_all_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全 slot 完了後のスコア再計算失敗では、全 slot の最良テキストが partial に残る。

    このケースが全損の本命: 最適化そのものは全部成功しているのに、再計算の失敗で
    成果ごと消えるのを防ぐ。`failed_slot=None` が「全 slot 完了」の標識。
    """
    from oai_agentspec.runtime.lightning import FailureKind, OptimizeError

    _no_extra_check(monkeypatch)

    async def _single(*, slot_name: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        return (f"{slot_name}-BEST", _entry(slot_name))

    async def _score_boom(*args: Any, **kwargs: Any) -> float:
        raise RuntimeError("score boom")

    monkeypatch.setattr(
        "oai_agentspec._adapters.lightning._run_apo_single_slot", _single, raising=True
    )
    monkeypatch.setattr(
        "oai_agentspec._adapters.lightning._score_candidate", _score_boom, raising=True
    )

    with pytest.raises(OptimizeError) as exc_info:
        await run_apo(
            seeds={"triage": "t", "billing": "b"},
            train=[{"input": "x"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.5),
            config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        )

    exc = exc_info.value
    assert exc.kind is FailureKind.TRAINER_FAILED
    assert exc.partial is not None
    assert exc.partial.completed_slots == {"triage": "triage-BEST", "billing": "billing-BEST"}
    assert exc.partial.failed_slot is None
    assert "RuntimeError" in exc.message
    assert "score boom" in exc.message


async def test_run_apo_partial_build_failure_falls_back_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """partial 組み立て自体が失敗しても元例外の構造化は行われる（fail-safe・partial=None）。

    `substitute_braced` は vars 値へ `str()` を適用するため、`__str__` が例外を投げる
    利用者オブジェクトで組み立てが失敗しうる。診断のための処理を新たな失敗源にしない。
    """
    from oai_agentspec.runtime.lightning import FailureKind, OptimizeError

    _no_extra_check(monkeypatch)

    class _BadStr:
        def __str__(self) -> str:
            raise ValueError("unstringable")

    async def _single(*, slot_name: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        if slot_name == "billing":
            raise RuntimeError("boom")
        return ("T-BEST ${company}", _entry(slot_name))

    monkeypatch.setattr(
        "oai_agentspec._adapters.lightning._run_apo_single_slot", _single, raising=True
    )

    with pytest.raises(OptimizeError) as exc_info:
        await run_apo(
            seeds={"triage": "t ${company}", "billing": "b"},
            train=[{"input": "x"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.5),
            config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
            vars_per_slot={"triage": {"company": _BadStr()}},  # type: ignore[dict-item]
        )

    exc = exc_info.value
    assert exc.kind is FailureKind.TRAINER_FAILED
    assert exc.partial is None
    assert "RuntimeError" in exc.message
    assert "boom" in exc.message


# --- _responses_complete_text: chat-only ゲートウェイへの自動 fallback ------------------------


def _not_found_error() -> Exception:
    """openai.NotFoundError（/responses 不在の 404）を最小構成で作る。"""
    import httpx
    import openai

    request = httpx.Request("POST", "https://gw.example.com/v1/responses")
    response = httpx.Response(404, request=request, json={"detail": "Not Found"})
    return openai.NotFoundError("Not Found", response=response, body={"detail": "Not Found"})


class _FakeGatewayClient:
    """responses が 404 を返し chat.completions は成功する chat-only ゲートウェイの疑似 client。"""

    def __init__(self) -> None:
        self.responses_calls = 0
        self.chat_calls: list[dict[str, Any]] = []

        outer = self

        class _Responses:
            async def create(self, **kwargs: Any) -> Any:
                outer.responses_calls += 1
                raise _not_found_error()

        class _Completions:
            async def create(self, **kwargs: Any) -> Any:
                outer.chat_calls.append(kwargs)
                from types import SimpleNamespace

                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="grad text"))]
                )

        class _Chat:
            completions = _Completions()

        self.responses = _Responses()
        self.chat = _Chat()


async def test_responses_complete_text_falls_back_to_chat_on_404() -> None:
    """/responses が 404 の client では chat.completions へ自動 fallback して完遂する。

    litellm 等の chat-only ゲートウェイを apo_client に渡した場合、Responses 固定だと
    APO gradient が必ず 404 で死ぬ（実 API で確認済み）。「client を渡せば動く」を保つため、
    エンドポイント不在（NotFoundError）のときだけ上流 agent-lightning 本来の chat 呼び出しへ
    倒す。messages は元の chat-style をそのまま渡す（system 分離は不要）。
    """
    from oai_agentspec._adapters.lightning import _responses_complete_text

    client = _FakeGatewayClient()
    messages = [
        {"role": "system", "content": "you are a critic"},
        {"role": "user", "content": "critique this"},
    ]

    text = await _responses_complete_text(
        client=client, model="gw-model", messages=messages, temperature=0.7
    )

    assert text == "grad text"
    assert client.responses_calls == 1
    assert len(client.chat_calls) == 1
    assert client.chat_calls[0]["model"] == "gw-model"
    assert client.chat_calls[0]["messages"] == messages
    assert client.chat_calls[0]["temperature"] == 0.7


async def test_responses_complete_text_fallback_is_memoized_via_provided_set() -> None:
    """呼び出し側が渡した memo set に fallback 済みモデルが記憶され、再試行しない。

    記憶の所有者は呼び出し側（`_build_apo` が APO インスタンスへ持たせる）。module-global に
    しないことで、id 再利用・GC 追従・unhashable 対策（旧 R1-R5 の 65 行）が構造ごと不要になり、
    一過性 404 の誤固定も 1 インスタンス（= 1 slot の APO 実行）に自動限定される。
    """
    from oai_agentspec._adapters.lightning import _responses_complete_text

    client = _FakeGatewayClient()
    memo: set[str] = set()
    messages = [{"role": "user", "content": "x"}]

    await _responses_complete_text(
        client=client, model="m", messages=messages, temperature=0.0, unsupported_models=memo
    )
    await _responses_complete_text(
        client=client, model="m", messages=messages, temperature=0.0, unsupported_models=memo
    )

    assert client.responses_calls == 1  # 1 回目のみ試行
    assert len(client.chat_calls) == 2
    assert memo == {"m"}


async def test_responses_fallback_memo_is_scoped_to_provided_set() -> None:
    """memo は渡された set の寿命に閉じる（別 set / 未渡しでは Responses を再試行する）。

    module-global 記憶だと一過性の誤分類 404 がプロセス寿命まで残る（外部レビュー指摘）。
    set 単位のスコープなら誤固定は最長でも 1 APO インスタンスの寿命で消える。
    """
    from oai_agentspec._adapters.lightning import _responses_complete_text

    client = _FakeGatewayClient()
    messages = [{"role": "user", "content": "x"}]

    await _responses_complete_text(
        client=client, model="m", messages=messages, temperature=0.0, unsupported_models=set()
    )
    # 別 set -> 記憶を共有せず Responses を再試行
    await _responses_complete_text(
        client=client, model="m", messages=messages, temperature=0.0, unsupported_models=set()
    )
    # memo 未渡し（None）-> 記憶なしで毎回試行
    await _responses_complete_text(client=client, model="m", messages=messages, temperature=0.0)

    assert client.responses_calls == 3


def test_responses_fallback_has_no_module_global_cache() -> None:
    """fallback の記憶に module-global を使わない（過剰設計の再発防止 pin）。"""
    from oai_agentspec._adapters import lightning as ln

    assert not hasattr(ln, "_responses_unsupported")
    assert not hasattr(ln, "_responses_known_unsupported")
    assert not hasattr(ln, "_remember_responses_unsupported")


async def test_responses_complete_text_non_404_propagates() -> None:
    """404 以外の失敗（認証エラー等）は fallback せず伝搬する（誤設定の隠蔽防止）。"""
    import httpx
    import openai

    from oai_agentspec._adapters.lightning import _responses_complete_text

    class _AuthFailClient:
        class responses:  # noqa: N801 - 疑似 client の属性名
            @staticmethod
            async def create(**kwargs: Any) -> Any:
                request = httpx.Request("POST", "https://gw.example.com/v1/responses")
                response = httpx.Response(401, request=request, json={})
                raise openai.AuthenticationError("bad key", response=response, body={})

    with pytest.raises(openai.AuthenticationError):
        await _responses_complete_text(
            client=_AuthFailClient(), model="m", messages=[], temperature=0.0
        )


def _not_found_error_with_code(code: str) -> Exception:
    """`error.code` を持つ 404（モデル / デプロイ不在）を作る。"""
    import httpx
    import openai

    body = {"error": {"code": code, "message": "not found"}}
    request = httpx.Request("POST", "https://api.example.com/v1/responses")
    response = httpx.Response(404, request=request, json=body)
    return openai.NotFoundError("not found", response=response, body=body["error"])


class _ModelNotFoundClient:
    """responses が model_not_found 404 を返す client（エンドポイントは存在する）。"""

    def __init__(self, code: str) -> None:
        self.code = code
        self.chat_calls = 0
        outer = self

        class _Responses:
            async def create(self, **kwargs: Any) -> Any:
                raise _not_found_error_with_code(outer.code)

        class _Completions:
            async def create(self, **kwargs: Any) -> Any:
                outer.chat_calls += 1
                from types import SimpleNamespace

                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="x"))]
                )

        class _Chat:
            completions = _Completions()

        self.responses = _Responses()
        self.chat = _Chat()


@pytest.mark.parametrize("code", ["model_not_found", "DeploymentNotFound"])
async def test_responses_complete_text_model_not_found_404_propagates(code: str) -> None:
    """モデル / デプロイ不在の 404 は fallback せず伝搬する（設定ミスを隠さない）。

    404 は「/responses が無い」だけでなく「model が無い」でも返る（OpenAI: model_not_found /
    Azure: DeploymentNotFound）。区別せず fallback すると、モデル名のタイポが
    「chat へ倒れて別のエラー」に化け、真因が隠れる。
    """
    import openai

    from oai_agentspec._adapters.lightning import _responses_complete_text

    client = _ModelNotFoundClient(code)
    with pytest.raises(openai.NotFoundError):
        await _responses_complete_text(
            client=client, model="typo-model", messages=[], temperature=0.0
        )
    assert client.chat_calls == 0


async def test_responses_complete_text_unhashable_client_still_falls_back() -> None:
    """hashable でない client でも fallback が成立する（記憶をスキップするだけ）。

    `__eq__` を定義して `__hash__` を失った wrapper（テストダブル / dataclass proxy 等）は
    weakref 可能でも unhashable。membership 判定が TypeError で落ちると、fallback へ
    到達する前に死ぬ（外部レビュー指摘・実測で TypeError を確認済み）。
    """
    from oai_agentspec._adapters.lightning import _responses_complete_text

    class _Unhashable(_FakeGatewayClient):
        __hash__ = None  # type: ignore[assignment]

        def __eq__(self, other: object) -> bool:
            return self is other

    client = _Unhashable()
    text = await _responses_complete_text(client=client, model="m", messages=[], temperature=0.0)
    assert text == "grad text"
    assert len(client.chat_calls) == 1


async def test_responses_complete_text_cache_registered_only_after_success() -> None:
    """chat fallback 自体が失敗したときは記憶しない（次回は Responses を再試行する）。

    成功前に登録すると、一過性の失敗でその (client, model) が永久に chat 固定になる。
    """
    from oai_agentspec._adapters.lightning import _responses_complete_text

    class _BothFailFirst:
        def __init__(self) -> None:
            self.responses_calls = 0
            self.chat_calls = 0
            outer = self

            class _Responses:
                async def create(self, **kwargs: Any) -> Any:
                    outer.responses_calls += 1
                    raise _not_found_error()

            class _Completions:
                async def create(self, **kwargs: Any) -> Any:
                    outer.chat_calls += 1
                    raise RuntimeError("chat down")

            class _Chat:
                completions = _Completions()

            self.responses = _Responses()
            self.chat = _Chat()

    client = _BothFailFirst()
    memo: set[str] = set()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="chat down"):
            await _responses_complete_text(
                client=client, model="m", messages=[], temperature=0.0, unsupported_models=memo
            )

    assert client.responses_calls == 2  # 記憶されていないので毎回 Responses を試す
    assert memo == set()


async def test_responses_complete_text_no_responses_attr_uses_chat_directly() -> None:
    """`responses` 属性を持たない chat-only client は 404 を待たず最初から chat を使う。

    最小構成の OpenAI 互換プロキシは `/responses` が 404 を返す以前に属性自体を持たない。
    `client.responses.create` へ触ると AttributeError で fallback に到達できず落ちる。
    """
    from types import SimpleNamespace

    from oai_agentspec._adapters.lightning import _responses_complete_text

    class _Completions:
        async def create(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="chat only"))]
            )

    class _Chat:
        completions = _Completions()

    class _ChatOnly:
        chat = _Chat()

    text = await _responses_complete_text(
        client=_ChatOnly(), model="m", messages=[], temperature=0.0
    )
    assert text == "chat only"


async def test_chat_complete_text_choice_without_message_returns_empty() -> None:
    """choice が message を持たない互換ゲートウェイ応答でも空文字へ degrade する。"""
    from types import SimpleNamespace

    from oai_agentspec._adapters.lightning import _chat_complete_text

    class _Completions:
        async def create(self, **kwargs: Any) -> Any:
            return SimpleNamespace(choices=[SimpleNamespace(delta="stream-like")])

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    assert (
        await _chat_complete_text(client=_Client(), model="m", messages=[], temperature=0.0) == ""
    )


async def test_run_apo_first_slot_import_error_maps_to_extra_missing_without_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """先頭 slot の ImportError は EXTRA_MISSING の OptimizeError になる（partial=None）。

    agentlightning 本体は導入済みでも `[apo]` 系サブ依存（poml 等）が欠けていると
    `_run_apo_single_slot` 内の遅延 import が ImportError を投げる。kind は EXTRA_MISSING
    契約を維持し（TRAINER_FAILED に化けない）、保全対象がない先頭 slot では partial=None。
    """
    from oai_agentspec.runtime.lightning import FailureKind, OptimizeError

    _no_extra_check(monkeypatch)

    async def _single(*, slot_name: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        raise ImportError("No module named 'poml'")

    monkeypatch.setattr(
        "oai_agentspec._adapters.lightning._run_apo_single_slot", _single, raising=True
    )

    with pytest.raises(OptimizeError) as exc_info:
        await run_apo(
            seeds={"triage": "t"},
            train=[{"input": "x"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.5),
            config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        )

    exc = exc_info.value
    assert exc.kind is FailureKind.EXTRA_MISSING
    assert "poml" in exc.message
    assert exc.partial is None
    assert isinstance(exc.__cause__, ImportError)


# --- apo_api 明示ノブ（_build_apo の bind 分岐 / allow_chat_fallback 配線） -------------------


def test_build_apo_chat_completions_does_not_bind_overrides(
    fake_apo_factory: list[_FakeAPO],
    monkeypatch: pytest.MonkeyPatch,  # noqa: ARG001 - fixture 副作用
) -> None:
    """`apo_api="chat_completions"` では Responses override を bind しない（上流 chat 実装）。

    上流 `agentlightning.APO` は素で chat.completions を使うため、非 bind = chat 動作。
    override 2 本は必ず同時に bind / 非 bind する（片方のみは gradient と apply-edit で
    API が食い違う不整合になる）。
    """
    from oai_agentspec._adapters.lightning import _build_apo
    from oai_agentspec.runtime.lightning import OptimizeConfig

    apo = _build_apo(OptimizeConfig(apo_client=_FAKE_APO_CLIENT, apo_api="chat_completions"))
    bound = vars(apo)
    assert "compute_textual_gradient" not in bound
    assert "textual_gradient_and_apply_edit" not in bound


@pytest.mark.parametrize(
    ("apo_api", "expected_fallback"),
    [(None, True), ("responses", False)],
)
def test_build_apo_binds_overrides_and_sets_fallback_flag(
    fake_apo_factory: list[_FakeAPO],
    monkeypatch: pytest.MonkeyPatch,  # noqa: ARG001 - fixture 副作用
    apo_api: str | None,
    expected_fallback: bool,
) -> None:
    """auto / responses 明示では override を 2 本とも bind し、fallback フラグを設定する。

    auto（None）= fallback 許可、"responses" 明示 = fallback 禁止（明示したのに黙って
    chat へ化けない fail-closed）。
    """
    from oai_agentspec._adapters.lightning import _build_apo
    from oai_agentspec.runtime.lightning import OptimizeConfig

    apo = _build_apo(OptimizeConfig(apo_client=_FAKE_APO_CLIENT, apo_api=apo_api))
    bound = vars(apo)
    assert "compute_textual_gradient" in bound
    assert "textual_gradient_and_apply_edit" in bound
    assert apo._oas_allow_chat_fallback is expected_fallback


async def test_responses_complete_text_strict_mode_propagates_404_without_chat() -> None:
    """`allow_chat_fallback=False` はエンドポイント不在 404 でも chat へ行かず伝搬・記憶なし。"""
    import openai

    from oai_agentspec._adapters import lightning as ln

    client = _FakeGatewayClient()
    memo: set[str] = set()
    with pytest.raises(openai.NotFoundError):
        await ln._responses_complete_text(
            client=client,
            model="m",
            messages=[],
            temperature=0.0,
            allow_chat_fallback=False,
            unsupported_models=memo,
        )
    assert client.chat_calls == []
    assert memo == set()


async def test_responses_complete_text_strict_mode_ignores_memo_and_attr_shortcut() -> None:
    """`allow_chat_fallback=False` は memo ヒットも属性ショートカットも使わず Responses を呼ぶ。

    同一インスタンスの先行 auto 呼び出しが memo へ記憶した後でも、明示 responses は
    Responses を再試行する（黙って chat へ行く経路をゼロにする fail-closed）。
    """
    from types import SimpleNamespace

    from oai_agentspec._adapters.lightning import _responses_complete_text

    class _ResponsesOk:
        def __init__(self) -> None:
            self.responses_called = 0
            outer = self

            class _Responses:
                async def create(self, **kwargs: Any) -> Any:
                    outer.responses_called += 1
                    return SimpleNamespace(output_text="resp")

            self.responses = _Responses()

    client = _ResponsesOk()
    memo = {"m"}  # 先行 auto の記憶を再現
    text = await _responses_complete_text(
        client=client,
        model="m",
        messages=[],
        temperature=0.0,
        allow_chat_fallback=False,
        unsupported_models=memo,
    )

    assert text == "resp"
    assert client.responses_called == 1


async def test_chat_complete_text_coerces_content_parts_list_to_str() -> None:
    """content が content-parts 形式（list）でも text を連結した str を返す（`-> str` 契約）。

    互換ゲートウェイは `[{"type": "text", "text": ...}]` 形式を返すことがある。list を
    そのまま返すと下流の APO 文字列処理が原因から遠い場所で TypeError になるか、
    list が「最適化済みプロンプト」として混入する。
    """
    from types import SimpleNamespace

    from oai_agentspec._adapters.lightning import _chat_complete_text

    class _Completions:
        async def create(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=[
                                {"type": "text", "text": "part1 "},
                                {"type": "text", "text": "part2"},
                            ]
                        )
                    )
                ]
            )

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    text = await _chat_complete_text(client=_Client(), model="m", messages=[], temperature=0.0)
    assert text == "part1 part2"


async def test_run_apo_slot_import_error_maps_to_extra_missing_with_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """slot 途中の ImportError は EXTRA_MISSING の OptimizeError に partial 付きで包まれる。

    生 raise（旧実装）では kind 契約は保てたが、(a) 完了済み slot の partial が捨てられ、
    (b) メッセージが原因の二面性（サブ依存欠落 or rollout 内 import 失敗）を説明できなかった。
    kind は EXTRA_MISSING を維持しつつ partial 保全と両立させる。
    """
    from oai_agentspec.runtime.lightning import FailureKind, OptimizeError

    _no_extra_check(monkeypatch)

    async def _single(*, slot_name: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        if slot_name == "billing":
            raise ImportError("No module named 'poml'")
        return ("T-BEST", _entry(slot_name))

    monkeypatch.setattr(
        "oai_agentspec._adapters.lightning._run_apo_single_slot", _single, raising=True
    )

    with pytest.raises(OptimizeError) as exc_info:
        await run_apo(
            seeds={"triage": "t", "billing": "b"},
            train=[{"input": "x"}],
            val=[{"input": "v"}],
            rollout=_const_rollout(0.5),
            config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        )

    exc = exc_info.value
    assert exc.kind is FailureKind.EXTRA_MISSING
    assert "poml" in exc.message
    assert exc.partial is not None
    assert exc.partial.completed_slots == {"triage": "T-BEST"}
    assert isinstance(exc.__cause__, ImportError)
