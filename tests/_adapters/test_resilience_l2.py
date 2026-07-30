"""L2: _adapters.resilience.build_model_retry の SDK 結線特性化テスト（#26 T4・RED 先行）。

`ModelRetryPolicy` の宣言メタデータが SDK `agents.ModelRetrySettings` へ正しく結線される
ことを検証する。具体的には (a) `max_retries` の転写、(b) backoff 系 4 フィールドの
None-omission と `ModelRetryBackoffSettings` への変換、(c) 生 `policy` 指定時の優先
（フラグ無視・is 比較）、(d) セマンティックフラグの `retry_policies.any(...)` 合成による
「まっとうな retry」の焼き込み（`RetryPolicyContext` を実際に評価して分岐を pin）を SDK
実型で検証する。

実装未完のため（`_adapters/resilience.py` の `build_model_retry` が未追加）、本モジュールの
import は `ImportError` となる（collection error = RED 状態が正しい）。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from unittest.mock import MagicMock

import pytest
from agents import (
    ModelRetryBackoffSettings,
    ModelRetryNormalizedError,
    ModelRetrySettings,
    RetryDecision,
    RetryPolicyContext,
    Usage,
)
from agents.lifecycle import RunHooksBase

import oai_agentspec._adapters.resilience as _resilience_adapter
from oai_agentspec._adapters.resilience import build_model_retry
from oai_agentspec.runtime.resilience._errors import RunBudgetExceeded
from oai_agentspec.runtime.resilience._types import ModelRetryPolicy, RunBudgetPolicy

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# ヘルパ
# ---------------------------------------------------------------------------
def _evaluate_policy(policy: object, ctx: RetryPolicyContext) -> bool:
    """SDK retry policy（sync/async・bool/RetryDecision 双方）を評価し retry 可否を bool で得る。

    `retry_policies.any(...)` の合成 policy は async かつ `RetryDecision` を返すため、
    coroutine なら `asyncio.run` で解決し、`RetryDecision` なら `.retry` を取り出す。
    """
    result = policy(ctx)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    if isinstance(result, RetryDecision):
        return result.retry
    return bool(result)


def _make_context(normalized: ModelRetryNormalizedError) -> RetryPolicyContext:
    """指定した normalized error facts を持つ最小の `RetryPolicyContext` を組む。"""
    return RetryPolicyContext(
        error=RuntimeError("dummy"),
        attempt=1,
        max_retries=3,
        stream=False,
        normalized=normalized,
    )


# ---------------------------------------------------------------------------
# 戻り値の型・max_retries 転写
# ---------------------------------------------------------------------------
def test_正常系_戻り値はSDK実型のModelRetrySettings() -> None:
    """`build_model_retry` は SDK 実型 `ModelRetrySettings` を返す。"""
    result = build_model_retry(ModelRetryPolicy(max_retries=3))
    assert isinstance(result, ModelRetrySettings)


def test_正常系_max_retriesがそのまま転写される() -> None:
    """`ModelRetryPolicy.max_retries` が `ModelRetrySettings.max_retries` に転写される。"""
    result = build_model_retry(ModelRetryPolicy(max_retries=3))
    assert result.max_retries == 3


# ---------------------------------------------------------------------------
# backoff の None-omission と変換
# ---------------------------------------------------------------------------
def test_正常系_backoff系全None時はbackoffがNone() -> None:
    """backoff 系 4 フィールド全 None のとき `.backoff` は None（SDK 既定へ委譲）。"""
    result = build_model_retry(ModelRetryPolicy(max_retries=3))
    assert result.backoff is None


def test_正常系_backoff部分指定は指定分のみ転写し他はNone() -> None:
    """`initial_delay_seconds` のみ指定時、`.backoff.initial_delay` のみ転写し他は None。"""
    result = build_model_retry(ModelRetryPolicy(max_retries=3, initial_delay_seconds=0.5))
    assert isinstance(result.backoff, ModelRetryBackoffSettings)
    assert result.backoff.initial_delay == 0.5
    assert result.backoff.max_delay is None
    assert result.backoff.multiplier is None
    assert result.backoff.jitter is None


def test_正常系_backoff全指定は各フィールドへ転写される() -> None:
    """backoff 系 4 フィールド全指定時、`ModelRetryBackoffSettings` の対応フィールドへ転写。"""
    result = build_model_retry(
        ModelRetryPolicy(
            max_retries=3,
            initial_delay_seconds=0.5,
            max_delay_seconds=10.0,
            backoff_multiplier=2.0,
            backoff_jitter=True,
        )
    )
    assert isinstance(result.backoff, ModelRetryBackoffSettings)
    assert result.backoff.initial_delay == 0.5
    assert result.backoff.max_delay == 10.0
    assert result.backoff.multiplier == 2.0
    assert result.backoff.jitter is True


def test_正常系_initial_delay_0_0はSDK側ge0で許容される() -> None:
    """`initial_delay_seconds=0.0` は SDK バリデーション（ge=0）を通り backoff に転写される。"""
    result = build_model_retry(ModelRetryPolicy(max_retries=3, initial_delay_seconds=0.0))
    assert isinstance(result.backoff, ModelRetryBackoffSettings)
    assert result.backoff.initial_delay == 0.0


# ---------------------------------------------------------------------------
# policy 合成: 既定フラグ・生 policy 優先
# ---------------------------------------------------------------------------
def test_正常系_policy未指定でも既定フラグからpolicyが必ず埋まる() -> None:
    """policy 未指定（全フラグ既定 True）でも `.policy` が callable で埋まる（no-op 排除）。"""
    result = build_model_retry(ModelRetryPolicy(max_retries=3))
    assert result.policy is not None
    assert callable(result.policy)


def test_正常系_生policy指定時はフラグ無視でそのまま優先される() -> None:
    """生 `policy` 指定時はフラグを無視し、その callable が `.policy` にそのまま渡る（is 比較）。"""

    def my_policy(ctx: RetryPolicyContext) -> bool:
        return True

    result = build_model_retry(ModelRetryPolicy(max_retries=3, policy=my_policy))
    assert result.policy is my_policy


# ---------------------------------------------------------------------------
# セマンティックフラグの合成挙動（RetryPolicyContext を実評価して分岐を pin）
# ---------------------------------------------------------------------------
def test_正常系_既定フラグでネットワークエラーはretry判定True() -> None:
    """全フラグ既定 True の合成 policy は network error を retry 対象と判定する。"""
    result = build_model_retry(ModelRetryPolicy(max_retries=3))
    ctx = _make_context(ModelRetryNormalizedError(is_network_error=True))
    assert _evaluate_policy(result.policy, ctx) is True


def test_正常系_既定フラグでタイムアウトはretry判定True() -> None:
    """全フラグ既定 True の合成 policy は timeout を retry 対象と判定する（network が両カバー）。"""
    result = build_model_retry(ModelRetryPolicy(max_retries=3))
    ctx = _make_context(ModelRetryNormalizedError(is_timeout=True))
    assert _evaluate_policy(result.policy, ctx) is True


def test_正常系_既定フラグでレート制限429はretry判定True() -> None:
    """全フラグ既定 True の合成 policy は HTTP 429（rate limit）を retry 対象と判定する。"""
    result = build_model_retry(ModelRetryPolicy(max_retries=3))
    ctx = _make_context(ModelRetryNormalizedError(status_code=429))
    assert _evaluate_policy(result.policy, ctx) is True


def test_正常系_既定フラグでサーバエラー500はretry判定True() -> None:
    """全フラグ既定 True の合成 policy は HTTP 500（server error）を retry 対象と判定する。"""
    result = build_model_retry(ModelRetryPolicy(max_retries=3))
    ctx = _make_context(ModelRetryNormalizedError(status_code=500))
    assert _evaluate_policy(result.policy, ctx) is True


def test_正常系_既定フラグでRetryAfterはretry判定True() -> None:
    """全フラグ既定 True の合成 policy は retry_after 提示時に retry 対象と判定する。"""
    result = build_model_retry(ModelRetryPolicy(max_retries=3))
    ctx = _make_context(ModelRetryNormalizedError(retry_after=1.0))
    assert _evaluate_policy(result.policy, ctx) is True


def test_正常系_既定フラグでもBadRequest400はretry判定False() -> None:
    """全フラグ既定 True の合成 policy でも HTTP 400（bad request）は retry しない。"""
    result = build_model_retry(ModelRetryPolicy(max_retries=3))
    ctx = _make_context(ModelRetryNormalizedError(status_code=400))
    assert _evaluate_policy(result.policy, ctx) is False


def test_正常系_extra_retry_statusesで追加ステータスがretry対象になる() -> None:
    """`extra_retry_statuses=(408,)` を指定すると HTTP 408 が retry 対象になる。"""
    result = build_model_retry(ModelRetryPolicy(max_retries=3, extra_retry_statuses=(408,)))
    ctx = _make_context(ModelRetryNormalizedError(status_code=408))
    assert _evaluate_policy(result.policy, ctx) is True


def test_正常系_ネットワークフラグ単独有効時に他エラーはretry対象外() -> None:
    """`retry_on_network_error` のみ有効時、network 系は retry・HTTP 500 は retry しない。"""
    result = build_model_retry(
        ModelRetryPolicy(
            max_retries=3,
            retry_on_network_error=True,
            retry_on_timeout=False,
            retry_on_rate_limit=False,
            retry_on_server_error=False,
            retry_on_retry_after=False,
        )
    )
    ctx_network = _make_context(ModelRetryNormalizedError(is_network_error=True))
    ctx_server = _make_context(ModelRetryNormalizedError(status_code=500))
    assert _evaluate_policy(result.policy, ctx_network) is True
    assert _evaluate_policy(result.policy, ctx_server) is False


# ===========================================================================
# T5: build_run_budget_hooks / _BudgetHooks（#26 T5・RED 先行）
#
# `RunHooksBase` を直接叩いて run 予算判定（on_llm_end のトークン/経過時間上限）を
# 単体検証する（Runner.run 経由の統合は T8）。実装未完のため、`build_run_budget_hooks`
# は `_adapters.resilience` に未追加で、下記テストは AttributeError で失敗する（RED）。
# 参照は属性アクセス経由にして T4 テスト群の collection を壊さない。
# ===========================================================================
def _build_budget_hooks(policy: RunBudgetPolicy) -> RunHooksBase:
    """未実装の `build_run_budget_hooks` を属性アクセス経由で解決して呼ぶ。

    トップレベル import にすると未実装時に collection error となり T4 テストまで
    巻き込むため、モジュール属性として遅延参照する（未実装時は AttributeError）。
    """
    return _resilience_adapter.build_run_budget_hooks(policy)


def _make_budget_context(total_tokens: int = 0, requests: int = 0) -> MagicMock:
    """`RunContextWrapper` 相当のスタブ。累積 usage を SDK 実型 `Usage` で持つ。"""
    ctx = MagicMock()
    ctx.usage = Usage(requests=requests, input_tokens=0, output_tokens=0, total_tokens=total_tokens)
    return ctx


def _make_agent(name: str = "test_agent") -> MagicMock:
    """`Agent` 相当のスタブ（`name` 属性のみ使用）。"""
    agent = MagicMock()
    agent.name = name
    return agent


def _make_response(usage: Usage) -> MagicMock:
    """`ModelResponse` 相当のスタブ（`usage` 属性のみ使用）。"""
    return MagicMock(usage=usage)


def _run_coro(coro: object) -> object:
    """フックの coroutine を同期テストから実行するヘルパ。"""
    return asyncio.run(coro)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 戻り値の型・インスタンス独立性
# ---------------------------------------------------------------------------
def test_正常系_build_run_budget_hooksはRunHooksBaseを返す() -> None:
    """戻り値は SDK 実型 `RunHooksBase` のサブクラスインスタンス。"""
    hooks = _build_budget_hooks(RunBudgetPolicy(max_total_tokens=100))
    assert isinstance(hooks, RunHooksBase)


def test_正常系_呼び出しごとに新インスタンスを返す() -> None:
    """1 run 1 インスタンス原則。呼び出しごとに別インスタンスを返す。"""
    policy = RunBudgetPolicy(max_total_tokens=100)
    assert _build_budget_hooks(policy) is not _build_budget_hooks(policy)


# ---------------------------------------------------------------------------
# no-op（両上限 None）
# ---------------------------------------------------------------------------
def test_正常系_両上限Noneはno_opで例外もwarningもない(caplog: pytest.LogCaptureFixture) -> None:
    """両上限 None は判定を行わず、usage 欠損でも warning を emit しない。"""
    hooks = _build_budget_hooks(RunBudgetPolicy())
    ctx = _make_budget_context(total_tokens=0, requests=0)
    agent = _make_agent()
    resp = _make_response(Usage(requests=0, total_tokens=0))
    with caplog.at_level(logging.WARNING, logger="oai_agentspec.resilience"):
        _run_coro(hooks.on_llm_start(ctx, agent, None, []))
        _run_coro(hooks.on_llm_end(ctx, agent, resp))
    assert caplog.records == []


# ---------------------------------------------------------------------------
# トークン上限
# ---------------------------------------------------------------------------
def test_異常系_トークン上限超過でRunBudgetExceededをraise() -> None:
    """累積 total_tokens が上限超過で `RunBudgetExceeded`。context 属性を検証する。"""
    hooks = _build_budget_hooks(RunBudgetPolicy(max_total_tokens=100))
    ctx = _make_budget_context(total_tokens=0, requests=1)
    agent = _make_agent("agent_x")
    _run_coro(hooks.on_llm_start(ctx, agent, None, []))
    ctx.usage.total_tokens = 200
    resp = _make_response(Usage(requests=1, total_tokens=200))
    with pytest.raises(RunBudgetExceeded) as exc_info:
        _run_coro(hooks.on_llm_end(ctx, agent, resp))
    exc = exc_info.value
    assert exc.usage is ctx.usage
    assert exc.context["exceeded"] == "max_total_tokens"
    assert exc.context["agent_name"] == "agent_x"
    assert exc.context["llm_calls"] >= 1
    assert exc.elapsed_seconds >= 0
    # Failsafe で RUNNING_AGENT を指定した際の解決元（run_data を持たない例外の
    # 読み取り先）。表示用の context["agent_name"] とは別に、実行中 agent の同一
    # オブジェクトを載せる。
    assert exc.last_agent is agent


def test_正常系_トークン上限と同値では超過しない() -> None:
    """total_tokens == max_total_tokens は境界内（strict greater での判定）。"""
    hooks = _build_budget_hooks(RunBudgetPolicy(max_total_tokens=100))
    ctx = _make_budget_context(total_tokens=100, requests=1)
    agent = _make_agent()
    _run_coro(hooks.on_llm_start(ctx, agent, None, []))
    resp = _make_response(Usage(requests=1, total_tokens=100))
    _run_coro(hooks.on_llm_end(ctx, agent, resp))


# ---------------------------------------------------------------------------
# 経過時間上限（monotonic を monkeypatch で制御）
# ---------------------------------------------------------------------------
def test_異常系_経過時間上限超過でRunBudgetExceededをraise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """初回 on_llm_start=0.0・on_llm_end=1.5 で 1.0 上限を超え `RunBudgetExceeded`。"""
    clock = {"t": 0.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["t"])
    hooks = _build_budget_hooks(RunBudgetPolicy(max_elapsed_seconds=1.0))
    ctx = _make_budget_context(total_tokens=10, requests=1)
    agent = _make_agent()
    clock["t"] = 0.0
    _run_coro(hooks.on_llm_start(ctx, agent, None, []))
    clock["t"] = 1.5
    resp = _make_response(Usage(requests=1, total_tokens=10))
    with pytest.raises(RunBudgetExceeded) as exc_info:
        _run_coro(hooks.on_llm_end(ctx, agent, resp))
    assert exc_info.value.context["exceeded"] == "max_elapsed_seconds"
    assert exc_info.value.elapsed_seconds == pytest.approx(1.5)
    # 時間超過経路でも実行中 agent の同一オブジェクトが last_agent に載る。
    assert exc_info.value.last_agent is agent


def test_正常系_RunBudgetExceededはlast_agent未指定ならNone() -> None:
    """`last_agent` 無しの構築も従来どおり通り、属性は None（後方互換の機械検証）。"""
    exc = RunBudgetExceeded("boom", usage=None, elapsed_seconds=0.0, context={})
    assert exc.last_agent is None


def test_正常系_経過時間が上限と同値では超過しない(monkeypatch: pytest.MonkeyPatch) -> None:
    """elapsed == max_elapsed_seconds は境界内（strict greater での判定）。"""
    clock = {"t": 0.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["t"])
    hooks = _build_budget_hooks(RunBudgetPolicy(max_elapsed_seconds=1.0))
    ctx = _make_budget_context(total_tokens=10, requests=1)
    agent = _make_agent()
    clock["t"] = 0.0
    _run_coro(hooks.on_llm_start(ctx, agent, None, []))
    clock["t"] = 1.0
    resp = _make_response(Usage(requests=1, total_tokens=10))
    _run_coro(hooks.on_llm_end(ctx, agent, resp))


# ---------------------------------------------------------------------------
# 開始時刻の遅延初期化
# ---------------------------------------------------------------------------
def test_正常系_構築直後は開始時刻が未初期化でon_llm_start初回で初期化される() -> None:
    """`_started` は構築時未初期化（None）で、初回 on_llm_start で初期化される。"""
    hooks = _build_budget_hooks(RunBudgetPolicy(max_elapsed_seconds=1.0))
    assert getattr(hooks, "_started", None) is None
    _run_coro(hooks.on_llm_start(_make_budget_context(), _make_agent(), None, []))
    assert getattr(hooks, "_started", None) is not None


def test_正常系_on_llm_start二回目は開始時刻を上書きしない(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """開始時刻は初回のみ確定する。2 回目の on_llm_start で上書きされない。"""
    clock = {"t": 0.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["t"])
    hooks = _build_budget_hooks(RunBudgetPolicy(max_elapsed_seconds=50.0))
    ctx = _make_budget_context(total_tokens=10, requests=1)
    agent = _make_agent()
    clock["t"] = 0.0
    _run_coro(hooks.on_llm_start(ctx, agent, None, []))
    clock["t"] = 10.0
    _run_coro(hooks.on_llm_start(ctx, agent, None, []))
    clock["t"] = 100.0
    resp = _make_response(Usage(requests=1, total_tokens=10))
    with pytest.raises(RunBudgetExceeded) as exc_info:
        _run_coro(hooks.on_llm_end(ctx, agent, resp))
    # 開始時刻が第 1 回目(0.0)から測られていれば elapsed は 100.0（10.0 上書きなら 90.0）
    assert exc_info.value.elapsed_seconds == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# usage 欠損の warning / 判定継続
# ---------------------------------------------------------------------------
def test_正常系_usage欠損時にwarningをemitし判定は継続する(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`response.usage` の requests==0 かつ total_tokens==0 で warning。例外は出さない。"""
    hooks = _build_budget_hooks(RunBudgetPolicy(max_total_tokens=100))
    ctx = _make_budget_context(total_tokens=0, requests=0)
    agent = _make_agent("agent_w")
    _run_coro(hooks.on_llm_start(ctx, agent, None, []))
    resp = _make_response(Usage(requests=0, total_tokens=0))
    with caplog.at_level(logging.WARNING, logger="oai_agentspec.resilience"):
        _run_coro(hooks.on_llm_end(ctx, agent, resp))
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_正常系_usage欠損ターン後も累積判定は継続して超過を検知する() -> None:
    """欠損ターンの後に valid usage を渡すと累積上限超過を発火できる。"""
    hooks = _build_budget_hooks(RunBudgetPolicy(max_total_tokens=100))
    ctx = _make_budget_context(total_tokens=0, requests=0)
    agent = _make_agent()
    _run_coro(hooks.on_llm_start(ctx, agent, None, []))
    _run_coro(hooks.on_llm_end(ctx, agent, _make_response(Usage(requests=0, total_tokens=0))))
    ctx.usage.total_tokens = 150
    with pytest.raises(RunBudgetExceeded):
        _run_coro(hooks.on_llm_end(ctx, agent, _make_response(Usage(requests=1, total_tokens=150))))


# ---------------------------------------------------------------------------
# 判定対象外フックの素通し
# ---------------------------------------------------------------------------
def test_正常系_判定対象外のフックは素通しする() -> None:
    """`on_agent_start` / `on_agent_end` は基底の既定実装で例外にならない。"""
    hooks = _build_budget_hooks(RunBudgetPolicy(max_total_tokens=100))
    ctx = _make_budget_context()
    agent = _make_agent()
    _run_coro(hooks.on_agent_start(ctx, agent))
    _run_coro(hooks.on_agent_end(ctx, agent, "output"))
