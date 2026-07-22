"""Model Retry と Run Budget の宣言型（agents 非依存）。

Model 呼び出しの一時失敗リトライ（`ModelRetryPolicy`）と run 全体の予算超過制御
（`RunBudgetPolicy`）を、実行コードの分岐でなく frozen dataclass として宣言する。
どちらも `__post_init__` で build-time 検証を行い、矛盾した宣言を fail-fast する。
外部依存（agents / openai）を持たず、`_adapters` のコンパイル関数が SDK ネイティブ
機構（`ModelSettings.retry` / `Runner.run(hooks=...)`）へ変換する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelRetryPolicy:
    """Model 呼び出しの一時失敗リトライの宣言（`ModelRetrySettings` へコンパイル）。

    セマンティックフラグ（`retry_on_*`。既定全 True）と `extra_retry_statuses` を
    `retry_policies.any(...)` へ合成して必ず policy を埋めることで、SDK 生型の
    「`policy` 未指定 = silent no-op（`max_retries` だけでは retry しない）」を排除する。
    生 `policy` を指定した場合はフラグを無視して優先する（エスケープハッチ）。
    backoff 系を未指定（None）にすると SDK 既定へ委譲する。

    Note:
        `retry_on_network_error` と `retry_on_timeout` は SDK の
        `retry_policies.network_error()` にまとめてコンパイルされ、独立に無効化できない
        （どちらか片方でも True なら両方が retry 対象となる）。SDK に timeout 単独の
        プリミティブが存在しないための制約。timeout のみを retry したい場合は生
        `policy` にカスタム callable を渡すこと。

    Attributes:
        max_retries: 最大リトライ回数。None は SDK 既定へ委譲、0 は retry 無効の明示。
        initial_delay_seconds: 初回リトライ前の待機秒数。None は SDK 既定へ委譲。
        max_delay_seconds: リトライ待機秒数の上限。None は SDK 既定へ委譲。
        backoff_multiplier: 指数バックオフの倍率。1 未満は逆進のため不可。None は委譲。
        backoff_jitter: バックオフへのジッタ付与有無。None は SDK 既定へ委譲。
        retry_on_network_error: ネットワークエラーで retry するか。
        retry_on_timeout: タイムアウトで retry するか。
        retry_on_rate_limit: レート制限（429）で retry するか。
        retry_on_server_error: サーバエラー（5xx）で retry するか。
        retry_on_retry_after: Retry-After ヘッダで retry するか。
        extra_retry_statuses: 追加で retry 対象とする HTTP ステータスコード。
        policy: 生の SDK retry policy。指定時はフラグを無視して優先する。
    """

    max_retries: int | None = None
    initial_delay_seconds: float | None = None
    max_delay_seconds: float | None = None
    backoff_multiplier: float | None = None
    backoff_jitter: bool | None = None
    retry_on_network_error: bool = True
    retry_on_timeout: bool = True
    retry_on_rate_limit: bool = True
    retry_on_server_error: bool = True
    retry_on_retry_after: bool = True
    extra_retry_statuses: tuple[int, ...] = ()
    policy: Any = None

    def __post_init__(self) -> None:
        """build-time 検証を行い、矛盾した宣言を `ValueError` で fail-fast する。

        Raises:
            ValueError: `max_retries` 負数 / `backoff_multiplier` < 1 /
                `initial_delay_seconds` > `max_delay_seconds` / 有効な retry 条件が
                ゼロなのに `max_retries` が正、のいずれかに該当する場合。
        """
        if self.max_retries is not None and self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        if self.backoff_multiplier is not None and self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be >= 1")

        if (
            self.initial_delay_seconds is not None
            and self.max_delay_seconds is not None
            and self.initial_delay_seconds > self.max_delay_seconds
        ):
            raise ValueError("initial_delay_seconds must be <= max_delay_seconds")

        self._validate_effective_conditions()

    def _validate_effective_conditions(self) -> None:
        """有効な retry 条件ゼロ x `max_retries` 正の矛盾を検知する。

        全セマンティックフラグ False かつ `extra_retry_statuses` 空かつ生 `policy`
        なしのとき、`max_retries` が正だと `retry_policies.any()`（引数ゼロ）が
        never() を返し「retry 回数を指定したのに一切 retry しない」silent no-op に
        なるため fail-fast する。

        Raises:
            ValueError: 有効な retry 条件がゼロなのに `max_retries` が正の場合。
        """
        if self.max_retries is None or self.max_retries <= 0:
            return

        has_semantic_flag = (
            self.retry_on_network_error
            or self.retry_on_timeout
            or self.retry_on_rate_limit
            or self.retry_on_server_error
            or self.retry_on_retry_after
        )
        if has_semantic_flag or self.extra_retry_statuses or self.policy is not None:
            return

        raise ValueError(
            "max_retries > 0 but no effective retry condition is enabled "
            "(all retry_on_* are False, extra_retry_statuses is empty, and "
            "policy is None)"
        )


@dataclass(frozen=True)
class RunBudgetPolicy:
    """run 全体の予算上限の宣言（`_BudgetHooks` へコンパイル）。

    両上限 None は no-op として許容し（`ValueError` にしない）、0 は意図的な即時上限
    として許容する。負数のみ build-time で `ValueError` にする。

    Attributes:
        max_elapsed_seconds: run 全体の累積経過秒数の上限。None は上限なし。
        max_total_tokens: run 全体の累積トークン数の上限。None は上限なし。
    """

    max_elapsed_seconds: float | None = None
    max_total_tokens: int | None = None

    def __post_init__(self) -> None:
        """build-time 検証を行い、負数上限を `ValueError` で fail-fast する。

        Raises:
            ValueError: `max_elapsed_seconds` または `max_total_tokens` が負数の場合。
        """
        if self.max_elapsed_seconds is not None and self.max_elapsed_seconds < 0:
            raise ValueError("max_elapsed_seconds must be >= 0")

        if self.max_total_tokens is not None and self.max_total_tokens < 0:
            raise ValueError("max_total_tokens must be >= 0")
