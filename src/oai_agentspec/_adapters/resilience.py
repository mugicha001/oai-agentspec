"""Resilience 系宣言型の SDK 結線（`build_model_retry` / `build_run_budget_hooks`）。

Model 呼び出しの一時失敗リトライ宣言（`ModelRetryPolicy`）と run 全体の予算超過制御
宣言（`RunBudgetPolicy`）を SDK ネイティブ機構へ変換する `_adapters` 窓口。前者は
本ファイルの `build_model_retry` で `agents.ModelRetrySettings` へコンパイルする。
後者（`build_run_budget_hooks` / `_BudgetHooks`）は本ファイルで実装済みで、
run 全体の累積 usage / 経過時間を `on_llm_end` のターン境界で判定する。

`from agents` の import は本ファイルに閉じる（NFR-1 の SDK 隔離）。上位層は plain な
`ModelRetryPolicy` / `RunBudgetPolicy` のみを扱い、SDK 実型には触れない。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agents import (
    ModelRetryBackoffSettings,
    ModelRetryNormalizedError,
    ModelRetrySettings,
    RetryDecision,
    RetryPolicyContext,
    RunErrorData,
    RunErrorHandlerInput,
    RunErrorHandlerResult,
    RunErrorHandlers,
    retry_policies,
)
from agents.lifecycle import RunHooksBase

from ..constants import RESILIENCE_LOGGER_NAME
from ..runtime.resilience._errors import RunBudgetExceeded
from ..runtime.resilience._types import ModelRetryPolicy, RunBudgetPolicy

# SDK 生型 10 種を本モジュール属性として集約再エクスポートする。
# `runtime/resilience/__init__.py` の PEP 562 `__getattr__` が本モジュール経由で遅延取得する
# ことで、公開窓口は `from agents` を書かず SDK 隔離（NFR-1）を維持する。
__all__ = [
    "ModelRetryBackoffSettings",
    "ModelRetryNormalizedError",
    "ModelRetrySettings",
    "RetryDecision",
    "RetryPolicyContext",
    "RunErrorData",
    "RunErrorHandlerInput",
    "RunErrorHandlerResult",
    "RunErrorHandlers",
    "build_model_retry",
    "build_run_budget_hooks",
    "retry_policies",
]

_logger = logging.getLogger(RESILIENCE_LOGGER_NAME)


def build_model_retry(policy: ModelRetryPolicy) -> ModelRetrySettings:
    """`ModelRetryPolicy` を SDK `ModelRetrySettings` へ結線する。

    backoff 系 4 フィールドは None-omission で `ModelRetryBackoffSettings` へ変換し、
    全 None なら backoff を None（SDK 既定へ委譲）にする。retry policy は生 `policy`
    指定を最優先し、未指定ならセマンティックフラグと `extra_retry_statuses` から
    `retry_policies.any(...)` を合成して必ず埋める（silent no-op の排除）。

    Args:
        policy: リトライ宣言（`__post_init__` で build-time 検証済み）。

    Returns:
        メタデータ適用済み `ModelRetrySettings`（SDK 実型）。
    """
    backoff = _build_backoff(policy)
    retry_policy = _build_retry_policy(policy)
    return ModelRetrySettings(
        max_retries=policy.max_retries,
        backoff=backoff,
        policy=retry_policy,
    )


def _build_backoff(policy: ModelRetryPolicy) -> ModelRetryBackoffSettings | None:
    """backoff 系 4 フィールドを None-omission で `ModelRetryBackoffSettings` へ変換する。

    Args:
        policy: リトライ宣言。

    Returns:
        指定フィールドのみ転写した `ModelRetryBackoffSettings`。全 None なら None。
    """
    kwargs: dict[str, Any] = {}
    if policy.initial_delay_seconds is not None:
        kwargs["initial_delay"] = policy.initial_delay_seconds
    if policy.max_delay_seconds is not None:
        kwargs["max_delay"] = policy.max_delay_seconds
    if policy.backoff_multiplier is not None:
        kwargs["multiplier"] = policy.backoff_multiplier
    if policy.backoff_jitter is not None:
        kwargs["jitter"] = policy.backoff_jitter

    if not kwargs:
        return None
    return ModelRetryBackoffSettings(**kwargs)


def _build_retry_policy(policy: ModelRetryPolicy) -> Any:
    """生 `policy` 優先・未指定ならセマンティックフラグから retry policy を合成する。

    `retry_policies.network_error()` は「is_network_error OR is_timeout」の両方を
    カバーするため、`retry_on_network_error` か `retry_on_timeout` のどちらかが True
    なら 1 回だけ追加する（両方 True でも重複しない）。

    Args:
        policy: リトライ宣言。

    Returns:
        生 `policy`（指定時）または `retry_policies.any(...)` の合成 callable。
        有効条件ゼロのときは `retry_policies.never()`（`any()` 引数ゼロ相当）。
    """
    if policy.policy is not None:
        return policy.policy

    collected: list[Any] = []
    if policy.retry_on_network_error or policy.retry_on_timeout:
        collected.append(retry_policies.network_error())
    if policy.retry_on_rate_limit:
        collected.append(retry_policies.http_status((429,)))
    if policy.retry_on_server_error:
        collected.append(retry_policies.http_status((500, 502, 503, 504)))
    if policy.retry_on_retry_after:
        collected.append(retry_policies.retry_after())
    if policy.extra_retry_statuses:
        collected.append(retry_policies.http_status(tuple(policy.extra_retry_statuses)))

    if not collected:
        return retry_policies.never()
    return retry_policies.any(*collected)


class _BudgetHooks(RunHooksBase[Any, Any]):
    """`RunBudgetPolicy` を `on_llm_end` のターン境界で判定する内部 hooks。

    設計原則:

    - **elapsed の遅延初期化**: 開始時刻はコンストラクタで確定させず、初回 `on_llm_start`
      で `time.monotonic()` を記録する（構築〜`Runner.run` 開始間の待機時間を予算に混入
      させない）。2 回目以降の `on_llm_start` は何もしない。
    - **usage は読むだけ**: SDK `run_loop` が `on_llm_end` 呼び出し直前に
      `context.usage.add(response.usage)` を済ませているため、hooks 内は
      `context.usage.total_tokens` を参照するだけ（自前加算は二重計上になる）。
    - **usage 欠損検知**: `response.usage.requests == 0` かつ `total_tokens == 0` の場合、
      `RESILIENCE_LOGGER_NAME` の logger に warning を emit する。判定自体は継続する。
    - **境界の扱い**: `>`（strict greater than）で比較する。`==` は超過扱いしない。
    - **上限 None は判定 skip**: `max_total_tokens` / `max_elapsed_seconds` それぞれ独立に
      判定するため、両 None なら実質 no-op になる。
    """

    def __init__(self, policy: RunBudgetPolicy) -> None:
        """RunBudgetPolicy を保持する新規 hooks インスタンスを初期化する。

        Args:
            policy: 累積上限宣言（両 None は許容 = 実質 no-op）。
        """
        super().__init__()
        self._policy = policy
        self._started: float | None = None
        self._llm_calls = 0

    async def on_llm_start(
        self,
        context: Any,
        agent: Any,
        system_prompt: Any,
        input_items: Any,
    ) -> None:
        """初回のみ `time.monotonic()` で開始時刻を遅延初期化する。

        Args:
            context: SDK `RunContextWrapper`。判定では未使用。
            agent: 呼び出し対象 Agent。判定では未使用。
            system_prompt: システムプロンプト。使用しない。
            input_items: 入力アイテム。使用しない。
        """
        if self._started is None:
            self._started = time.monotonic()

    async def on_llm_end(
        self,
        context: Any,
        agent: Any,
        response: Any,
    ) -> None:
        """累積 usage / 経過時間を判定し、超過時は `RunBudgetExceeded` を送出する。

        Args:
            context: SDK `RunContextWrapper`。`context.usage.total_tokens` を参照する。
            agent: 呼び出した Agent。エラー context の `agent_name` に載せる。
            response: SDK `ModelResponse`。usage 欠損検知に使用（判定には使わない）。

        Raises:
            RunBudgetExceeded: `max_total_tokens` / `max_elapsed_seconds` のいずれかを超過。
        """
        self._llm_calls += 1
        # 両上限 None（no-op）の場合は判定も warning も行わず素通し。片方でも上限が
        # 設定されているときのみ usage 欠損検知と累積判定を実行する（ログノイズ抑制）。
        if self._policy.max_total_tokens is None and self._policy.max_elapsed_seconds is None:
            return

        r_usage = getattr(response, "usage", None)
        if (
            r_usage is not None
            and getattr(r_usage, "requests", 0) == 0
            and getattr(r_usage, "total_tokens", 0) == 0
        ):
            _logger.warning(
                "resilience budget: usage missing for agent=%s (call #%s): "
                "response.usage.requests==0 and total_tokens==0",
                _agent_name(agent),
                self._llm_calls,
            )

        cum = context.usage
        elapsed = 0.0 if self._started is None else time.monotonic() - self._started

        if (
            self._policy.max_total_tokens is not None
            and cum.total_tokens > self._policy.max_total_tokens
        ):
            raise RunBudgetExceeded(
                f"run budget exceeded: total_tokens={cum.total_tokens} "
                f"> max={self._policy.max_total_tokens}",
                usage=cum,
                elapsed_seconds=elapsed,
                context={
                    "agent_name": _agent_name(agent),
                    "llm_calls": self._llm_calls,
                    "exceeded": "max_total_tokens",
                },
            )
        if (
            self._policy.max_elapsed_seconds is not None
            and elapsed > self._policy.max_elapsed_seconds
        ):
            raise RunBudgetExceeded(
                f"run budget exceeded: elapsed_seconds={elapsed:.3f} "
                f"> max={self._policy.max_elapsed_seconds}",
                usage=cum,
                elapsed_seconds=elapsed,
                context={
                    "agent_name": _agent_name(agent),
                    "llm_calls": self._llm_calls,
                    "exceeded": "max_elapsed_seconds",
                },
            )


def _agent_name(agent: Any) -> str:
    """agent オブジェクトから安全に name を取り出す（不透明値へのアクセス保護）。

    Args:
        agent: SDK Agent または類似オブジェクト。

    Returns:
        `agent.name` が存在すればその値、なければ ``"<unknown>"``。
    """
    return getattr(agent, "name", "<unknown>") if agent is not None else "<unknown>"


def build_run_budget_hooks(policy: RunBudgetPolicy) -> RunHooksBase[Any, Any]:
    """`RunBudgetPolicy` から新規の `_BudgetHooks` インスタンスを返す。

    「1 run 1 インスタンス」原則: `_started` 状態が run 間で汚染されないよう、呼び出し
    ごとに新インスタンスを返す。既存の他 `RunHooksBase` との合成（chain）機構は提供せず、
    複数 hooks の併用は利用者責務（利用者側で自作 `RunHooksBase` サブクラスに委譲する）。

    ハード timeout（tool 実行中も含めた即時中断）が必要な場合は、利用者側で
    ``asyncio.wait_for(Runner.run(...), timeout=...)`` を被せる（本 hooks はターン境界の
    graceful 判定のみを担う）。

    Args:
        policy: 累積上限宣言（両 None は許容 = 実質 no-op）。

    Returns:
        `Runner.run(agent, input, hooks=...)` に渡せる SDK `RunHooksBase` サブクラス
        インスタンス。
    """
    return _BudgetHooks(policy)
