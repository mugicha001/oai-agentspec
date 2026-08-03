"""run 単位フックの合成（`chain_hooks`）と agent 単位との非対称の例。

`Runner.run(hooks=...)` は単数の `RunHooksBase` しか受け取らないため、複数のフックを併用する
には合成が必要になる。本例では以下を順に示す:

- run 単位の縮退仕様: 0 個なら全メソッド no-op の素 `RunHooksBase()`、1 個ならそのフック自身
  （`is` 一致・ラッパを作らない）、2 個以上なら合成ラッパ。
- agent 単位との非対称 2 点。同じ「合成ヘルパー」でも受理の緩さが違う:
  1. run 単位は `None` を除外しない（1 個なら素通しなので `None` がそのまま返る）。
     agent 単位は `None` を除外して no-op の素インスタンスを返す。
  2. run 単位は部分実装を許容しない（`getattr` ガードを持たず直接呼ぶ）。誤用は silent skip に
     ならず実行開始直後に `AttributeError` で顕在化する。agent 単位は欠損メソッドを skip する。
- メソッド名の違い: agent 単位の `on_start` / `on_end` に対応するのは run 単位の
  `on_agent_start` / `on_agent_end`。`on_handoff` は agent 単位が `(context, agent, source)`、
  run 単位が `(context, from_agent, to_agent)` と引数の意味も異なる。両者は互換ではない。
- 実 LLM 呼び出しで agent 単位と run 単位が同時に発火すること（最後のパターンのみ API を使う）。

最後のパターン以外は LLM を呼ばないため、環境変数を設定していなくても動く。実 API を使う
パターンまで通す場合は examples/_shared/_azure.py の環境変数を設定して実行する:

    uv run python examples/hooks/02_chain_hooks.py

agent 単位の合成（`chain_agent_hooks`）の詳細は `examples/hooks/01_chain_agent_hooks.py`、
run 単位フックと予算 hooks（`build_run_budget_hooks`）の併用は
`examples/resilience/01_retry_and_budget.py` が扱う。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from agents import Runner
from agents.lifecycle import AgentHooksBase, RunHooksBase

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.hooks import chain_agent_hooks, chain_hooks

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import build_model  # noqa: E402

# 実 API 呼び出しのハード上限（秒）。ターン境界の graceful 判定ではなく強制打ち切り。
_RUN_TIMEOUT_SECONDS = 60.0


class RunLoggingHooks(RunHooksBase[Any, Any]):
    """run 全体のライフサイクルを受ける run 単位フック。

    メソッド名が agent 単位と異なる点に注意する。agent 単位の `on_start` / `on_end` に
    対応するのは `on_agent_start` / `on_agent_end` で、run 内で複数のエージェントが動く場合は
    エージェントごとに発火する。
    """

    def __init__(self, label: str) -> None:
        """記録先ラベルを保持する新規インスタンスを初期化する。

        Args:
            label: print 出力の先頭に付けるラベル。
        """
        super().__init__()
        self.label = label
        self.events: list[str] = []

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        """run 内でエージェントが開始したことを記録する。"""
        self.events.append("on_agent_start")
        print(f"[{self.label}] on_agent_start agent={getattr(agent, 'name', '?')}")

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        """run 内でエージェントが終了したことを記録する。"""
        self.events.append("on_agent_end")
        print(f"[{self.label}] on_agent_end")


class AgentScopeHooks(AgentHooksBase[Any, Any]):
    """比較用の agent 単位フック（`on_start` / `on_end` を持つ）。"""

    def __init__(self, label: str) -> None:
        """記録先ラベルを保持する新規インスタンスを初期化する。

        Args:
            label: print 出力の先頭に付けるラベル。
        """
        super().__init__()
        self.label = label
        self.events: list[str] = []

    async def on_start(self, context: Any, agent: Any) -> None:
        """エージェント開始を記録する。"""
        self.events.append("on_start")
        print(f"[{self.label}] on_start agent={getattr(agent, 'name', '?')}")

    async def on_end(self, context: Any, agent: Any, output: Any) -> None:
        """エージェント終了を記録する。"""
        self.events.append("on_end")
        print(f"[{self.label}] on_end")


def _show_degenerate_forms() -> None:
    """run 単位の縮退（0 個は素インスタンス / 1 個は `is` 一致 / 2 個以上は合成）を示す。"""
    print("--- run 単位の縮退仕様 ---")

    print("  0 個 ->", type(chain_hooks()).__name__)

    single = RunLoggingHooks(label="single")
    print("  1 個は同一オブジェクトを返す ->", chain_hooks(single) is single)
    print("  2 個以上 ->", type(chain_hooks(single, RunLoggingHooks(label="b"))).__name__)


def _show_none_handling_asymmetry() -> None:
    """非対称 1: run 単位は `None` を除外しないことを示す。"""
    print("--- 非対称 1: None の扱い ---")

    # run 単位は 1 個なら素通しするため、`None` を渡すと `None` がそのまま返る。
    # `Runner.run(hooks=None)` は SDK 既定と同じ扱いだが、他のフックと併記すると
    # 合成ラッパが `None.on_agent_start` を呼んで AttributeError になる。
    print("  chain_hooks(None) ->", repr(chain_hooks(None)))
    print("  chain_agent_hooks(None) ->", type(chain_agent_hooks(None)).__name__)
    print("  run 単位で条件付き無効化を書くなら、呼び出し側で None を除いてから渡す")


async def _show_partial_implementation_asymmetry() -> None:
    """非対称 2: run 単位は部分実装を許容せず誤用が即座に顕在化することを示す。"""
    print("--- 非対称 2: 部分実装の扱い ---")

    class _PartialRunHooks:
        """`on_agent_start` だけを持つ run 単位フックの部分実装（許容されない）。"""

        async def on_agent_start(self, context: Any, agent: Any) -> None:
            """開始のみ受ける。"""

    chained = chain_hooks(_PartialRunHooks(), RunLoggingHooks(label="b"))  # type: ignore[arg-type]
    try:
        await chained.on_agent_end(None, None, None)
    except AttributeError as exc:
        print("  run 単位は欠損メソッドで即座に失敗 ->", exc)

    # agent 単位は欠損メソッドを skip するため、部分実装をそのまま並べられる。
    class _PartialAgentHooks:
        """`on_start` だけを持つ agent 単位フックの部分実装（許容される）。"""

        async def on_start(self, context: Any, agent: Any) -> None:
            """開始のみ受ける。"""

    agent_chained = chain_agent_hooks(_PartialAgentHooks(), AgentScopeHooks(label="b"))
    await agent_chained.on_end(None, None, None)  # 欠損は skip されるので例外にならない
    print("  agent 単位は欠損メソッドを skip -> 例外なし")


async def _run_both_scopes() -> None:
    """実 LLM 呼び出しで agent 単位と run 単位が同時に発火することを示す。

    2 つのスロットは独立している。agent 単位は `AgentSpec.hooks`（そのエージェントの
    ライフサイクル）、run 単位は `Runner.run(hooks=...)`（run 全体）で、どちらも単一スロット
    なので複数持たせるにはそれぞれの合成ヘルパーを使う。環境変数が必要。
    """
    print("--- 両スコープ同時（実 API）---")
    agent_hooks = AgentScopeHooks(label="agent")
    run_hooks = RunLoggingHooks(label="run")
    run_hooks_2 = RunLoggingHooks(label="run2")

    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="assistant",
            instructions="一文で簡潔に答えてください。",
            model=build_model(),
            # agent 単位: このエージェント固有のフック。
            hooks=chain_agent_hooks(agent_hooks),
        )
    )
    agent = registry.get("assistant")

    # run 単位: run 全体に効くフックを 2 つ合成する（宣言順に順次 await される）。
    # ハード上限は利用者側で被せる（lib はターン境界の graceful 判定のみを担う）。
    result = await asyncio.wait_for(
        Runner.run(agent, "こんにちは。", hooks=chain_hooks(run_hooks, run_hooks_2)),
        timeout=_RUN_TIMEOUT_SECONDS,
    )
    print("  final_output =", result.final_output)
    print("  agent 単位 ->", agent_hooks.events)
    print("  run 単位 ->", run_hooks.events, "/", run_hooks_2.events)


async def main() -> None:
    """API 不要のパターンを先に実行し、最後に実 API のパターンを実行する。"""
    _show_degenerate_forms()
    _show_none_handling_asymmetry()
    await _show_partial_implementation_asymmetry()
    await _run_both_scopes()


if __name__ == "__main__":
    asyncio.run(main())
