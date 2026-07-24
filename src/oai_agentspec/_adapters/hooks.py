"""複数 `RunHooksBase` を宣言順に合成する SDK 結線（`chain_hooks`）。

目的:
    `Runner.run(agent, input, hooks=...)` は単数の `RunHooks` しか受け取らないため、
    複数の hook（例: `build_run_budget_hooks` の予算 hooks とアプリ固有のログ hooks）を
    併用したい場合の合成窓口を提供する。`chain_hooks(*hooks)` は複数 hook を宣言順
    （引数の左から右）に順次呼び出す単一の `RunHooksBase` へまとめる。

合成仕様:
    - 全 7 メソッド（`on_llm_start` / `on_llm_end` / `on_agent_start` / `on_agent_end` /
      `on_handoff` / `on_tool_start` / `on_tool_end`）を宣言順に順次 `await` する。
    - fail-fast: 前段 hook が例外を raise したら後段はスキップし、例外はそのまま伝播する
      （`return_exceptions` 相当の集約は非対応。必要になれば別 Issue で扱う）。
    - 引数は無変更で全 hook に転送する。

SDK 追随手順:
    SDK（`agents.lifecycle.RunHooksBase`）に新規 hook メソッドが追加された場合、本モジュールの
    `_ChainedHooks` に同名のオーバーライドを追加すること（追加漏れは新メソッドが合成対象から
    抜ける silent gap になるため）。

配置理由:
    `RunHooksBase` の**サブクラス定義**が必要で `agents.lifecycle` の import が不可避なため、
    SDK 隔離（NFR-1）の窓口として `_adapters/` 配下に配置する。上位層は本モジュールの
    `chain_hooks` のみを介して合成し、SDK 実型には触れない。
"""

from __future__ import annotations

from typing import Any

from agents.lifecycle import RunHooksBase


class _ChainedHooks(RunHooksBase[Any, Any]):
    """複数の `RunHooksBase` を宣言順に順次 `await` する合成 hooks。

    設計原則:

    - **宣言順の順次実行**: 各メソッドは保持した hook 列を左から右へ順に `await` する。
    - **fail-fast**: 素の順次 `await` により、前段が raise すれば後段の `await` に到達せず
      例外がそのまま伝播する（try/except は使わない）。
    - **引数無変更転送**: 受け取った引数をそのまま各 hook の同名メソッドへ渡す。
    """

    def __init__(self, hooks: tuple[RunHooksBase[Any, Any], ...]) -> None:
        """合成対象の hook 列を保持する新規インスタンスを初期化する。

        Args:
            hooks: 宣言順（左から右）の `RunHooksBase` 列。2 個以上を想定する。
        """
        super().__init__()
        self._hooks = hooks

    async def on_llm_start(
        self,
        context: Any,
        agent: Any,
        system_prompt: Any,
        input_items: Any,
    ) -> None:
        """各 hook の `on_llm_start` を宣言順に順次 `await` する。

        Args:
            context: SDK `RunContextWrapper`。無変更で転送する。
            agent: 呼び出し対象 Agent。無変更で転送する。
            system_prompt: システムプロンプト。無変更で転送する。
            input_items: 入力アイテム。無変更で転送する。
        """
        for hook in self._hooks:
            await hook.on_llm_start(context, agent, system_prompt, input_items)

    async def on_llm_end(
        self,
        context: Any,
        agent: Any,
        response: Any,
    ) -> None:
        """各 hook の `on_llm_end` を宣言順に順次 `await` する。

        Args:
            context: SDK `RunContextWrapper`。無変更で転送する。
            agent: 呼び出した Agent。無変更で転送する。
            response: SDK `ModelResponse`。無変更で転送する。
        """
        for hook in self._hooks:
            await hook.on_llm_end(context, agent, response)

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        """各 hook の `on_agent_start` を宣言順に順次 `await` する。

        Args:
            context: SDK `RunContextWrapper`。無変更で転送する。
            agent: 開始する Agent。無変更で転送する。
        """
        for hook in self._hooks:
            await hook.on_agent_start(context, agent)

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        """各 hook の `on_agent_end` を宣言順に順次 `await` する。

        Args:
            context: SDK `RunContextWrapper`。無変更で転送する。
            agent: 終了した Agent。無変更で転送する。
            output: Agent の出力。無変更で転送する。
        """
        for hook in self._hooks:
            await hook.on_agent_end(context, agent, output)

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        """各 hook の `on_handoff` を宣言順に順次 `await` する。

        Args:
            context: SDK `RunContextWrapper`。無変更で転送する。
            from_agent: handoff 元 Agent。無変更で転送する。
            to_agent: handoff 先 Agent。無変更で転送する。
        """
        for hook in self._hooks:
            await hook.on_handoff(context, from_agent, to_agent)

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        """各 hook の `on_tool_start` を宣言順に順次 `await` する。

        Args:
            context: SDK `RunContextWrapper`。無変更で転送する。
            agent: ツールを呼ぶ Agent。無変更で転送する。
            tool: 実行するツール。無変更で転送する。
        """
        for hook in self._hooks:
            await hook.on_tool_start(context, agent, tool)

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        """各 hook の `on_tool_end` を宣言順に順次 `await` する。

        Args:
            context: SDK `RunContextWrapper`。無変更で転送する。
            agent: ツールを呼んだ Agent。無変更で転送する。
            tool: 実行したツール。無変更で転送する。
            result: ツールの結果。無変更で転送する。
        """
        for hook in self._hooks:
            await hook.on_tool_end(context, agent, tool, result)


def chain_hooks(*hooks: RunHooksBase[Any, Any]) -> RunHooksBase[Any, Any]:
    """複数の `RunHooksBase` を宣言順に合成した単一の `RunHooksBase` を返す。

    `Runner.run(hooks=...)` が単数の hook しか受け取らない制約に対し、複数 hook を宣言順
    （引数の左から右）に順次 `await` する合成インスタンスを提供する。引数個数に応じて
    軽量最適化する:

    - **0 個**: 素の `RunHooksBase()`（全メソッド no-op の SDK デフォルト）を返す。
    - **1 個**: その hook 自身をそのまま返す（`is` 一致・合成のオーバーヘッドを避ける）。
    - **2 個以上**: `_ChainedHooks` インスタンスを返す。

    合成メソッドは fail-fast（前段 raise で後段スキップ・例外はそのまま伝播）で、引数は
    無変更で全 hook に転送する。

    Args:
        hooks: 宣言順（左から右）の `RunHooksBase` 列（0 個以上）。

    Returns:
        `Runner.run(agent, input, hooks=...)` に渡せる合成済み `RunHooksBase`。
    """
    if not hooks:
        return RunHooksBase()
    if len(hooks) == 1:
        return hooks[0]
    return _ChainedHooks(hooks)
