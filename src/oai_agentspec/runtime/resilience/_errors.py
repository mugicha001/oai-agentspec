"""Resilience 系の lib 独自例外（agents 非依存）。

`RunBudgetPolicy` の累積上限（時間 / トークン）到達時に `_BudgetHooks`
（`_adapters.resilience`）から送出される。lib は agents 非依存を維持するため、
`usage` フィールドは SDK 型を直接参照せず不透明な `Any` として保持する。
"""

from __future__ import annotations

from typing import Any


class RunBudgetExceeded(Exception):
    """`RunBudgetPolicy` の累積上限に達した際に送出される例外。

    SDK の `RunErrorHandlers` は `MaxTurnsExceeded` / `ModelRefusalError` のみを
    isinstance dispatch するため、本例外は SDK に握り潰されず `Runner.run` の
    呼び出し元まで素通しで伝播する（streaming の場合は `stream_events()` 消費時に
    raise される）。

    Attributes:
        usage: 例外送出時点の累積 usage（SDK `Usage` インスタンス相当・不透明値）。
            `total_tokens` 等の属性は SDK 側のスキーマに従う。
        elapsed_seconds: `time.monotonic()` ベースの累積経過秒。
            hooks インスタンスが初回 `on_llm_start` で計測開始した基準時刻からの差分。
        context: 追加のコンテキスト情報（`agent_name` / `llm_calls` / `exceeded`）。
            `exceeded` は超過した上限名（`"max_total_tokens"` または `"max_elapsed_seconds"`）。
        last_agent: 例外送出時点で実行中だった agent（SDK `Agent` インスタンス相当・
            不透明値）。Failsafe で `RUNNING_AGENT` を指定した際の解決元になる。
            取得できない文脈では `None`。表示用の `context["agent_name"]`（文字列）とは役割が
            異なり、こちらは継続実行（再度 `Runner.run` へ渡す等）のためのオブジェクト
            参照を保持する。
    """

    def __init__(
        self,
        message: str,
        *,
        usage: Any,
        elapsed_seconds: float,
        context: dict[str, Any],
        last_agent: Any = None,
    ) -> None:
        """例外を初期化する。

        Args:
            message: 人間可読なエラー説明（super().__init__ へ渡される）。
            usage: 累積 usage（SDK `Usage` 相当の不透明値）。
            elapsed_seconds: 累積経過秒。
            context: agent 名・LLM 呼び出し回数・超過した上限名を含む辞書。
            last_agent: 例外送出時点で実行中だった agent（不透明値）。既定は `None`。
        """
        super().__init__(message)
        self.usage = usage
        self.elapsed_seconds = elapsed_seconds
        self.context = context
        self.last_agent = last_agent
