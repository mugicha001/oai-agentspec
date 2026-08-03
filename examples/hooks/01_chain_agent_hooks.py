"""エージェント単位フックの合成（`chain_agent_hooks`）の例。

`AgentSpec.hooks` は単一スロットのため、自作フックを複数持たせるには合成が必要になる。
本例では以下を順に示す:

- 宣言と build: `AgentSpec(hooks=chain_agent_hooks(...))` の合成結果が `registry.get()` で
  `agents.Agent.hooks` へ素通しされること。
- 受理する 3 形: `AgentHooksBase` インスタンス / `on_*` の一部だけを持つ部分実装
  （duck-typed・継承不要）/ `None`（条件分岐なしで無効化を表現できる）。
- 縮退仕様: `None` を除いた実効件数が 0 件なら全メソッド no-op の素インスタンス、1 件かつ
  `AgentHooksBase` インスタンスならそのフック自身（`is` 一致・ラッパを作らない）。
- fail-fast: 前段のフックが例外を送出したら後段は呼ばれず、例外がそのまま伝播する。
- run 単位フックの拒否: `RunHooksBase` インスタンスを渡すと build 時に `TypeError` になる
  （メソッド名が異なり `on_handoff` の引数意味も違うため）。
- 実 LLM 呼び出しでの発火順（最後のパターンのみ API を使う）。

最後のパターン以外は LLM を呼ばないため、環境変数を設定していなくても動く。実 API を使う
パターンまで通す場合は examples/_shared/_azure.py の環境変数を設定して実行する:

    uv run python examples/hooks/01_chain_agent_hooks.py

run 単位（`Runner.run(hooks=...)`）の合成 `chain_hooks` と agent 単位との非対称は
`examples/hooks/02_chain_hooks.py` が扱う。予算 hooks との併用は
`examples/resilience/01_retry_and_budget.py` を参照。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from agents import Runner
from agents.lifecycle import AgentHooksBase, RunHooksBase

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.hooks import chain_agent_hooks

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import build_model  # noqa: E402

# 実 API 呼び出しのハード上限（秒）。ターン境界の graceful 判定ではなく強制打ち切り。
_RUN_TIMEOUT_SECONDS = 60.0


class MetricsHooks(AgentHooksBase[Any, Any]):
    """全ライフサイクルを受ける通常の実装（`AgentHooksBase` を継承する形）。"""

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


class ToolLogger:
    """部分実装（`AgentHooksBase` を継承せず 2 メソッドのみ持つ duck-typed オブジェクト）。

    持たないメソッド（`on_start` 等）の呼び出しは合成側で skip されるため、継承の定型コードを
    書かずにツール境界だけを観測できる。
    """

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        """ツール開始を記録する。"""
        print(f"[tools] on_tool_start tool={getattr(tool, 'name', '?')}")

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        """ツール終了を記録する。"""
        print(f"[tools] on_tool_end tool={getattr(tool, 'name', '?')}")


class _RaisingHooks(AgentHooksBase[Any, Any]):
    """fail-fast の実演用に `on_start` で必ず例外を送出するフック。"""

    async def on_start(self, context: Any, agent: Any) -> None:
        """常に `RuntimeError` を送出する。"""
        raise RuntimeError("前段のフックが失敗しました")


def _show_declaration_and_build() -> None:
    """宣言した合成フックが build 後の `Agent.hooks` へ素通しされることを示す。"""
    print("--- 宣言と build（追加の変換は不要）---")
    enable_audit = False  # 条件付きで有効化するフックの例（無効時は None を渡す）

    chained = chain_agent_hooks(
        MetricsHooks(label="metrics"),
        MetricsHooks(label="audit") if enable_audit else None,
        ToolLogger(),
    )

    registry = AgentRegistry()
    registry.register(AgentSpec(name="triage", instructions="振り分け担当。", hooks=chained))
    agent = registry.get("triage")

    print("  agent.hooks is chained ->", agent.hooks is chained)
    print("  型 ->", type(agent.hooks).__name__)


def _show_degenerate_forms() -> None:
    """実効件数による縮退（0 件は素インスタンス / 1 件は `is` 一致）を示す。"""
    print("--- 縮退仕様 ---")

    empty = chain_agent_hooks()
    all_none = chain_agent_hooks(None, None)
    print("  0 件 ->", type(empty).__name__, "/ 全 None ->", type(all_none).__name__)

    single = MetricsHooks(label="single")
    print("  1 件（継承あり）は同一オブジェクトを返す ->", chain_agent_hooks(single) is single)
    print("  None 併記でも同じ ->", chain_agent_hooks(None, single) is single)

    # 部分実装は `AgentHooksBase` 非インスタンスなので、1 件でもラッパで包まれる。
    wrapped = chain_agent_hooks(ToolLogger())
    print("  1 件（部分実装）は包まれる ->", type(wrapped).__name__)
    print("  包んだ結果も AgentHooksBase 適合 ->", isinstance(wrapped, AgentHooksBase))


async def _show_fail_fast() -> None:
    """前段が例外を送出したら後段が呼ばれないことを示す（fail-fast）。"""
    print("--- fail-fast ---")
    later = MetricsHooks(label="later")
    hooks = chain_agent_hooks(_RaisingHooks(), later)

    try:
        await hooks.on_start(None, None)
    except RuntimeError as exc:
        print("  例外はそのまま伝播 ->", exc)

    print("  後段は呼ばれていない ->", later.events == [])


def _show_run_scope_rejection() -> None:
    """run 単位フックを渡すと build 時に `TypeError` で拒否されることを示す。"""
    print("--- run 単位フックの拒否 ---")

    class _RunScopeHooks(RunHooksBase[Any, Any]):
        """run 単位フック（`on_agent_start` 等・agent 単位とはメソッド名が異なる）。"""

        async def on_agent_start(self, context: Any, agent: Any) -> None:
            """run 単位の開始通知。"""

    try:
        chain_agent_hooks(MetricsHooks(label="metrics"), _RunScopeHooks())
    except TypeError as exc:
        print("  ", str(exc).splitlines()[0])


async def _run_with_real_model() -> None:
    """実 LLM 呼び出しで agent 単位のフックが宣言順に発火することを示す（環境変数が必要）。"""
    print("--- 実行時の発火順（実 API）---")
    first = MetricsHooks(label="first")
    second = MetricsHooks(label="second")

    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="assistant",
            instructions="一文で簡潔に答えてください。",
            model=build_model(),
            hooks=chain_agent_hooks(first, second),
        )
    )
    agent = registry.get("assistant")

    # ハード上限は利用者側で被せる（lib はターン境界の graceful 判定のみを担う）。
    result = await asyncio.wait_for(Runner.run(agent, "こんにちは。"), timeout=_RUN_TIMEOUT_SECONDS)
    print("  final_output =", result.final_output)
    print("  発火順 ->", "first:", first.events, "second:", second.events)


async def main() -> None:
    """API 不要のパターンを先に実行し、最後に実 API のパターンを実行する。"""
    _show_declaration_and_build()
    _show_degenerate_forms()
    await _show_fail_fast()
    _show_run_scope_rejection()
    await _run_with_real_model()


if __name__ == "__main__":
    asyncio.run(main())
