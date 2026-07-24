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


async def test_run_apo_seed_prompt_compose_with_fixed_and_diff(
    fake_apo_factory: list[_FakeAPO],  # noqa: ARG001
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """`fixed=` 経路: seed/prompt は base+parts を含む合成済み full テキストで、diff には tune の
    変更行のみ ± で出る（base/parts は同一行・unified diff フォーマット）。"""
    result = await run_apo(
        seeds={"bot": "hi ${var}"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.5),
        config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        fixed={"bot": "BASE TEXT\n\nSTYLE TEXT"},
    )
    # seed = fixed + tune（合成済み full）。
    assert result.seed == "BASE TEXT\n\nSTYLE TEXT\n\nhi ${var}"
    # prompt = fixed + 最適化済み tune（合成済み full）。
    assert result.prompt == "BASE TEXT\n\nSTYLE TEXT\n\n(optimized) hi ${var}"
    # diff: unified diff 形式・base/parts 行は不変（- / + 無し）、tune 行だけ ±。
    assert isinstance(result.diff, str)
    assert result.diff.startswith("--- before")
    assert "+++ after" in result.diff
    assert "-hi ${var}" in result.diff
    assert "+(optimized) hi ${var}" in result.diff
    # context として変更行近傍の不変行（STYLE TEXT 等）は ± なしで含まれる
    # （unified_diff の既定 context=3 で変更行の前後 3 行）。
    assert " STYLE TEXT" in result.diff


async def test_run_apo_compose_substitutes_vars_in_fixed(
    fake_apo_factory: list[_FakeAPO],  # noqa: ARG001
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """`vars_per_slot=` 経路: rollout 時 `_default_build`（fixed 側 vars 再注入）+ `_reinject_vars`
    （tune 側 vars 再注入）の両方が走るため、`OptimizeResult.seed` / `prompt` も rollout 実体と一致
    させるため fixed と tune の両方を substitute 済みの full テキストで返す（Codex 第4 round 指摘・
    "OptimizeResult.prompt は rollout 実体と一致" 公開契約）。"""
    result = await run_apo(
        seeds={"bot": "hi ${var}"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.5),
        config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        fixed={"bot": "company=${company}"},
        vars_per_slot={"bot": {"company": "AgentSpec", "var": "WORLD"}},
    )
    # fixed 側 (`${company}` -> AgentSpec) と tune 側 (`${var}` -> WORLD) の両方に
    # vars が注入される。
    assert result.seed == "company=AgentSpec\n\nhi WORLD"
    assert result.prompt == "company=AgentSpec\n\n(optimized) hi WORLD"


async def test_run_apo_compose_does_not_touch_bare_dollar_vars(
    fake_apo_factory: list[_FakeAPO],  # noqa: ARG001
    fake_trainer_factory: list[_FakeTrainer],  # noqa: ARG001
    fake_prompt_template: None,  # noqa: ARG001
    captured_rewards: list[float],  # noqa: ARG001
) -> None:
    """braced `${var}` のみ置換し、bare `$var` には触らない（Template.safe_substitute の
    bare `$var` 副作用回避・Codex 第3 round 指摘）。fixed 内の literal `$5` / `$PATH` は
    vars に同名キーがあっても置換されないこと。"""
    result = await run_apo(
        seeds={"bot": "hi ${var}"},
        train=[{"input": "t"}],
        val=[{"input": "v"}],
        rollout=_const_rollout(0.5),
        config=OptimizeConfig(apo_client=_FAKE_APO_CLIENT),
        # fixed: braced ${company} は置換、bare $5 と $company（中括弧なし）は維持。
        fixed={"bot": "company=${company}, price=$5, shell=$company"},
        # `"5": "FIVE"` は **意図的に未使用な dead key**（substitute_braced は `${5}` のような
        # 数字始まり identifier を PLACEHOLDER_RE が match しないため触らないことを確認する目的の
        # negative case）。`bare $5` も触られないことが load-bearing なアサーション対象。
        vars_per_slot={"bot": {"company": "AgentSpec", "5": "FIVE"}},
    )
    # ${company} → AgentSpec（braced・置換）。$5 と $company（bare）は触らない。
    assert "company=AgentSpec" in str(result.seed)
    assert "price=$5" in str(result.seed)
    assert "shell=$company" in str(result.seed)


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
