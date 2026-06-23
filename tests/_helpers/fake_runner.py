"""L1（agents 非依存）テスト用の runner シーム fake。

`WorkflowGraph._interpret` が受け取る `RunnerSeam`（`async run(...)`）を満たすフェイク。
AGENT ステップ名 -> 出力（値 or callable）のテーブルで決定論的に応答し、各呼び出しの
引数（agent / input / context / session / max_turns 等）を記録する。

`agents.Runner` を構築せず、内部インタプリタの制御フロー（順次/並列/分岐/ループ/
context・session 受け渡し）を純粋に検証するためのフェイク（FakeAgentBuilder 方式に倣う）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _RunResult:
    """SDK RunResult 相当（`final_output` 属性のみ持つ）。"""

    final_output: Any


@dataclass
class _RunCall:
    """1 回の `run` 呼び出しの記録。

    Attributes:
        agent: 渡された AGENT ステップ名（registry None 時は名前が素通る）。
        input: ステップへの入力。
        context: 共有 context。
        kwargs: `Runner.run` へ素通しされた残りの kwarg（run_config / session / max_turns 等）。
    """

    agent: str
    input: Any
    context: Any
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def run_config(self) -> Any:
        """素通し kwarg から run_config を取り出す（未指定は None）。"""
        return self.kwargs.get("run_config")

    @property
    def session(self) -> Any:
        """素通し kwarg から session を取り出す（未指定は None）。"""
        return self.kwargs.get("session")

    @property
    def max_turns(self) -> int | None:
        """素通し kwarg から max_turns を取り出す（未指定は None）。"""
        return self.kwargs.get("max_turns")


class FakeRunnerAdapter:
    """`RunnerSeam` を満たす runner シーム fake（内部インタプリタへ注入する）。

    Attributes:
        outputs: AGENT 名 -> 出力。値が callable なら `fn(input, context)` を呼ぶ。
        calls: 各 `run` 呼び出しの記録。
    """

    def __init__(self, outputs: dict[str, Any] | None = None) -> None:
        """fake runner を生成する。

        Args:
            outputs: AGENT 名 -> 出力（値 or `(input, context) -> 出力` の callable）。
        """
        self.outputs: dict[str, Any] = dict(outputs or {})
        self.calls: list[_RunCall] = []

    def set(self, agent: str, output: Any) -> FakeRunnerAdapter:
        """AGENT 名の応答を登録する（自身を返す）。"""
        self.outputs[agent] = output
        return self

    async def run(
        self,
        agent: str,
        input: Any,
        *,
        context: Any = None,
        **runner_kwargs: Any,
    ) -> _RunResult:
        """AGENT ステップを fake 応答で実行し RunResult 相当を返す。

        Args:
            agent: AGENT ステップ名。
            input: ステップへの入力。
            context: 共有 context。
            **runner_kwargs: `Runner.run` へ素通しされる残りの kwarg（run_config /
                session / max_turns 等）。

        Returns:
            `final_output` を持つ RunResult 相当。

        Raises:
            KeyError: outputs に未登録の AGENT 名を実行しようとした場合。
        """
        self.calls.append(
            _RunCall(
                agent=agent,
                input=input,
                context=context,
                kwargs=dict(runner_kwargs),
            )
        )
        if agent not in self.outputs:
            raise KeyError(f"FakeRunnerAdapter: 未登録の agent 出力: {agent!r}")
        value = self.outputs[agent]
        if callable(value):
            return _RunResult(final_output=value(input, context))
        return _RunResult(final_output=value)


@dataclass
class FakeSession:
    """SDK Session 相当の空スタブ（parallel + session fail-fast 検証用）。"""

    items: list[Any] = field(default_factory=list)
