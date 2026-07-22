"""L1: Resilience 宣言型 (`ModelRetryPolicy` / `RunBudgetPolicy`) の純検証。

frozen dataclass の既定値・フィールド保持・frozen 性・`__post_init__` の build-time
検証（矛盾宣言の fail-fast）・境界の許容を pin する。外部依存 (agents / openai) なし。

T1 の RED 先行テスト。実装 `runtime/resilience/_types.py` は未作成のため、import が
ImportError となり本ファイル全体が collection error で失敗する = RED 状態が正しい。
実装完了後に緑化する（緑になるまで実装フェーズには進まない）。
"""

from __future__ import annotations

import dataclasses

import pytest
from oai_agentspec.runtime.resilience._types import ModelRetryPolicy, RunBudgetPolicy

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ModelRetryPolicy: 既定値・フィールド保持
# ---------------------------------------------------------------------------


def test_model_retry_policy_最小構成の既定値() -> None:
    """`ModelRetryPolicy()` が生成でき、既定値が仕様どおり。

    max_retries=None・backoff 系全 None（SDK 既定へ委譲）・全セマンティックフラグ True・
    extra_retry_statuses 空 tuple・policy None。
    """
    policy = ModelRetryPolicy()
    assert policy.max_retries is None
    assert policy.initial_delay_seconds is None
    assert policy.max_delay_seconds is None
    assert policy.backoff_multiplier is None
    assert policy.backoff_jitter is None
    assert policy.retry_on_network_error is True
    assert policy.retry_on_timeout is True
    assert policy.retry_on_rate_limit is True
    assert policy.retry_on_server_error is True
    assert policy.retry_on_retry_after is True
    assert policy.extra_retry_statuses == ()
    assert policy.policy is None


def test_model_retry_policy_フル指定でフィールド保持() -> None:
    """全フィールドを明示指定した値がそのまま保持される。"""

    def _sentinel_policy() -> None:  # 生 policy callable のダミー
        return None

    policy = ModelRetryPolicy(
        max_retries=5,
        initial_delay_seconds=0.5,
        max_delay_seconds=10.0,
        backoff_multiplier=2.0,
        backoff_jitter=True,
        retry_on_network_error=False,
        retry_on_timeout=False,
        retry_on_rate_limit=True,
        retry_on_server_error=False,
        retry_on_retry_after=False,
        extra_retry_statuses=(408, 409),
        policy=_sentinel_policy,
    )
    assert policy.max_retries == 5
    assert policy.initial_delay_seconds == 0.5
    assert policy.max_delay_seconds == 10.0
    assert policy.backoff_multiplier == 2.0
    assert policy.backoff_jitter is True
    assert policy.retry_on_network_error is False
    assert policy.retry_on_timeout is False
    assert policy.retry_on_rate_limit is True
    assert policy.retry_on_server_error is False
    assert policy.retry_on_retry_after is False
    assert policy.extra_retry_statuses == (408, 409)
    assert policy.policy is _sentinel_policy


def test_model_retry_policy_is_frozen() -> None:
    """frozen dataclass のため属性の書き換えは FrozenInstanceError。"""
    policy = ModelRetryPolicy(max_retries=3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.max_retries = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModelRetryPolicy: build-time ValueError（__post_init__ 検証）
# ---------------------------------------------------------------------------


def test_model_retry_policy_max_retries_負数は_ValueError() -> None:
    """max_retries が負数は build-time で ValueError。"""
    with pytest.raises(ValueError, match="max_retries"):
        ModelRetryPolicy(max_retries=-1)


def test_model_retry_policy_backoff_multiplier_1未満は_ValueError() -> None:
    """backoff_multiplier < 1（逆進 backoff）は ValueError。"""
    with pytest.raises(ValueError, match="backoff_multiplier"):
        ModelRetryPolicy(max_retries=3, backoff_multiplier=0.5)


def test_model_retry_policy_initial_delay_が_max_delay_超過は_ValueError() -> None:
    """initial_delay_seconds > max_delay_seconds は ValueError。"""
    with pytest.raises(ValueError, match="delay"):
        ModelRetryPolicy(
            max_retries=3,
            initial_delay_seconds=5.0,
            max_delay_seconds=1.0,
        )


def test_model_retry_policy_有効条件ゼロ_x_max_retries正は_ValueError() -> None:
    """全セマンティックフラグ False + extra 空 + policy None + max_retries 正は矛盾で ValueError。

    有効条件ゼロを許容すると `retry_policies.any()`（引数ゼロ）が never() を返し
    「max_retries=3 なのに一切 retry しない」silent no-op が再発するため fail-fast する。
    """
    with pytest.raises(ValueError):
        ModelRetryPolicy(
            max_retries=3,
            retry_on_network_error=False,
            retry_on_timeout=False,
            retry_on_rate_limit=False,
            retry_on_server_error=False,
            retry_on_retry_after=False,
            extra_retry_statuses=(),
            policy=None,
        )


# ---------------------------------------------------------------------------
# ModelRetryPolicy: 境界の許容（ValueError にしない）
# ---------------------------------------------------------------------------


def test_model_retry_policy_max_retries_0は許容() -> None:
    """max_retries=0（retry 無効の明示）は矛盾ではなく OK。"""
    policy = ModelRetryPolicy(max_retries=0)
    assert policy.max_retries == 0


def test_model_retry_policy_initial_delay_と_max_delay_同値は許容() -> None:
    """initial_delay_seconds == max_delay_seconds は OK。"""
    policy = ModelRetryPolicy(
        max_retries=3,
        initial_delay_seconds=1.0,
        max_delay_seconds=1.0,
    )
    assert policy.initial_delay_seconds == 1.0
    assert policy.max_delay_seconds == 1.0


def test_model_retry_policy_backoff_multiplier_1は許容() -> None:
    """backoff_multiplier == 1（べき等 backoff）は OK。"""
    policy = ModelRetryPolicy(max_retries=3, backoff_multiplier=1.0)
    assert policy.backoff_multiplier == 1.0


def test_model_retry_policy_有効条件ゼロでも_max_retries_None_は許容() -> None:
    """有効条件ゼロでも max_retries=None なら retry 無効なので矛盾なし。"""
    policy = ModelRetryPolicy(
        max_retries=None,
        retry_on_network_error=False,
        retry_on_timeout=False,
        retry_on_rate_limit=False,
        retry_on_server_error=False,
        retry_on_retry_after=False,
    )
    assert policy.max_retries is None


def test_model_retry_policy_有効条件ゼロでも_max_retries_0_は許容() -> None:
    """有効条件ゼロでも max_retries=0 なら retry 無効なので矛盾なし。"""
    policy = ModelRetryPolicy(
        max_retries=0,
        retry_on_network_error=False,
        retry_on_timeout=False,
        retry_on_rate_limit=False,
        retry_on_server_error=False,
        retry_on_retry_after=False,
    )
    assert policy.max_retries == 0


def test_model_retry_policy_有効条件ゼロでも_生policyがあれば許容() -> None:
    """有効条件ゼロでも生 policy（エスケープハッチ）があれば OK。"""

    def _raw_policy() -> None:
        return None

    policy = ModelRetryPolicy(
        max_retries=3,
        retry_on_network_error=False,
        retry_on_timeout=False,
        retry_on_rate_limit=False,
        retry_on_server_error=False,
        retry_on_retry_after=False,
        policy=_raw_policy,
    )
    assert policy.policy is _raw_policy


def test_model_retry_policy_有効条件ゼロでも_extra_statusesがあれば許容() -> None:
    """有効条件ゼロでも extra_retry_statuses が 1 つでもあれば OK。"""
    policy = ModelRetryPolicy(
        max_retries=3,
        retry_on_network_error=False,
        retry_on_timeout=False,
        retry_on_rate_limit=False,
        retry_on_server_error=False,
        retry_on_retry_after=False,
        extra_retry_statuses=(408,),
    )
    assert policy.extra_retry_statuses == (408,)


# ---------------------------------------------------------------------------
# RunBudgetPolicy: 既定値・フィールド保持
# ---------------------------------------------------------------------------


def test_run_budget_policy_最小構成の既定値() -> None:
    """`RunBudgetPolicy()` が生成でき、既定値は両 None（no-op）。"""
    policy = RunBudgetPolicy()
    assert policy.max_elapsed_seconds is None
    assert policy.max_total_tokens is None


def test_run_budget_policy_フル指定でフィールド保持() -> None:
    """max_elapsed_seconds / max_total_tokens の指定値がそのまま保持される。"""
    policy = RunBudgetPolicy(max_elapsed_seconds=60.0, max_total_tokens=200_000)
    assert policy.max_elapsed_seconds == 60.0
    assert policy.max_total_tokens == 200_000


def test_run_budget_policy_is_frozen() -> None:
    """frozen dataclass のため属性の書き換えは FrozenInstanceError。"""
    policy = RunBudgetPolicy(max_total_tokens=100)
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.max_total_tokens = 200  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RunBudgetPolicy: build-time ValueError / 境界の許容
# ---------------------------------------------------------------------------


def test_run_budget_policy_max_elapsed_seconds_負数は_ValueError() -> None:
    """max_elapsed_seconds が負数は build-time で ValueError。"""
    with pytest.raises(ValueError, match="max_elapsed_seconds"):
        RunBudgetPolicy(max_elapsed_seconds=-1.0)


def test_run_budget_policy_max_total_tokens_負数は_ValueError() -> None:
    """max_total_tokens が負数は build-time で ValueError。"""
    with pytest.raises(ValueError, match="max_total_tokens"):
        RunBudgetPolicy(max_total_tokens=-1)


def test_run_budget_policy_max_elapsed_seconds_0は許容() -> None:
    """max_elapsed_seconds == 0.0（意図的な即時上限）は OK。"""
    policy = RunBudgetPolicy(max_elapsed_seconds=0.0)
    assert policy.max_elapsed_seconds == 0.0


def test_run_budget_policy_max_total_tokens_0は許容() -> None:
    """max_total_tokens == 0（意図的な即時上限）は OK。"""
    policy = RunBudgetPolicy(max_total_tokens=0)
    assert policy.max_total_tokens == 0


def test_run_budget_policy_両None_は矛盾ではない() -> None:
    """両 None は no-op として許容し ValueError にしない。"""
    policy = RunBudgetPolicy(max_elapsed_seconds=None, max_total_tokens=None)
    assert policy.max_elapsed_seconds is None
    assert policy.max_total_tokens is None
