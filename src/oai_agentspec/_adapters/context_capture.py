"""経路C（`as_agent_spec`）向けの run スコープ実行コンテキスト捕捉（SDK 結合を閉じる・NFR-1）。

SDK の `Model.get_response` は run context を受け取らないため、`WorkflowModel` は自力では
`Runner.run(context=...)` の値を知れない。本モジュールは lib 所有の `AgentHooks.on_start` で
SDK から渡される `RunContextWrapper` を受け取り、現在 span（`agents.tracing.get_current_span()`）を
キーに保持する。`WorkflowModel.get_response` は同一 turn 内で同じ現在 span を観測するため、
`resolve()` で自 run の wrapper を取り出せる。

依拠する不変条件は「`on_start` と同一 turn の `get_response` / `stream_response` は同じ現在 span を
観測する」のみで、span の種類名には依拠しない（tracing 無効時は呼び出しごとに一意な `NoOpSpan`）。
保持は `WeakKeyDictionary` のため span の寿命が尽きるとエントリも自動消滅し、pop 忘れによる残留や
別 run への混線は起きない（tracing 有効時の span は exporter の送出キューが保持するため、エントリは
run 終了後も送出完了まで残る。`NoOpSpan` は run 終了で即解放される）。
1 `RunContextCapture` = 1 テーブル（1 `WorkflowModel` につき 1 つ生成し、共有レジストリは
持たない）。

`_adapters.hooks` / `runtime.hooks` はトップレベルで import しない（PEP 562 の遅延ロード probe を
維持するため。利用者側の合成は `chain_agent_hooks` に委ねる）。
"""

from __future__ import annotations

from typing import Any
from weakref import WeakKeyDictionary

from agents import AgentHooks
from agents.tracing import get_current_span


class _CaptureAgentHooks(AgentHooks[Any]):
    """`on_start` で受け取った `RunContextWrapper` を現在 span をキーに登録する lib 所有フック。

    `on_start` 以外は基底の no-op のまま（`chain_agent_hooks` の受理条件「`on_*` を 1 つ以上
    持つ」を `on_start` で満たす）。捕捉は agent 起動ごとに 1 回（NFR-5）。
    """

    def __init__(self, by_span: WeakKeyDictionary[Any, Any]) -> None:
        """捕捉テーブルを共有するフックを生成する。

        Args:
            by_span: `RunContextCapture` が所有する `現在 span -> RunContextWrapper` テーブル。
        """
        self._by_span = by_span

    async def on_start(self, context: Any, agent: Any) -> None:
        """現在 span をキーに wrapper を登録する（現在 span が無ければ何もしない）。

        Args:
            context: SDK から渡される `RunContextWrapper`（実型は SDK のサブクラス）。
            agent: 起動した Agent（未使用）。
        """
        span = get_current_span()
        if span is not None:
            self._by_span[span] = context


class RunContextCapture:
    """1 つの `WorkflowModel` に対応する run スコープ捕捉テーブル（現在 span -> wrapper）。

    `hooks` を `AgentSpec.hooks` に載せ、`resolve` を `WorkflowModel(context_resolver=)` へ
    渡すことで、同じテーブルを介して hook 側の登録と Model 側の解決を結ぶ。
    """

    def __init__(self) -> None:
        """空の捕捉テーブルと、それへ書き込む lib 所有フックを生成する。"""
        self._by_span: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()
        self._hooks = _CaptureAgentHooks(self._by_span)

    @property
    def hooks(self) -> AgentHooks[Any]:
        """`AgentSpec.hooks` に載せる lib 所有フック（毎回同一インスタンス）。"""
        return self._hooks

    def resolve(self) -> Any:
        """現在 span に登録された `RunContextWrapper` を返す（捕捉不能時は None）。

        現在 span が無い（`Runner.run` を経ない直接呼び出し）/ 現在 span はあるが未登録
        （`hooks` の上書きで `on_start` を経ていない）のいずれも、警告・例外なしに None を返す。
        読み取りでエントリを消費しないため、同一 turn 内の再試行でも同じ wrapper を返す。

        Returns:
            登録済みの `RunContextWrapper`（実型は SDK のサブクラス）または None。
        """
        span = get_current_span()
        if span is None:
            return None
        return self._by_span.get(span)
