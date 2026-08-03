"""複数 hook を宣言順に合成する SDK 結線（run 単位 / agent 単位の 2 ヘルパー）。

目的:
    `Runner.run(agent, input, hooks=...)` は単数の `RunHooks` しか受け取らず、`Agent.hooks`
    （`AgentSpec.hooks`）も単一スロットのため、複数 hook を併用したい場合の合成窓口を提供する。
    run 単位（例: `build_run_budget_hooks` の予算 hooks とアプリ固有のログ hooks）は
    `chain_hooks(*hooks)`、agent 単位（例: メトリクス hooks と監査 hooks）は
    `chain_agent_hooks(*hooks)` が、宣言順（引数の左から右）に順次呼び出す単一の hook へまとめる。

合成仕様（両ヘルパー共通）:
    - 対象基底クラスの全 7 メソッドを宣言順に順次 `await` する（run 単位は `on_llm_start` /
      `on_llm_end` / `on_agent_start` / `on_agent_end` / `on_handoff` / `on_tool_start` /
      `on_tool_end`、agent 単位は `on_start` / `on_end` / `on_handoff` / `on_tool_start` /
      `on_tool_end` / `on_llm_start` / `on_llm_end`）。
    - fail-fast: 前段 hook が例外を raise したら後段はスキップし、例外はそのまま伝播する
      （`return_exceptions` 相当の集約は非対応。必要になれば別 Issue で扱う）。
    - 引数は無変更で全 hook に転送する。

縮退仕様の非対称（run 単位と agent 単位で意図的に異なる点）:
    agent 単位 `chain_agent_hooks` のみ、`None` 要素を無視して実効件数を数え、実効 1 件が
    `AgentHooksBase` インスタンスでなければラッパで包む（`AgentHooksBase` の全 `on_*` を持たない
    部分実装も受理し、`getattr` で当該メソッドを持つ要素だけへ委譲する）。run 単位
    `chain_hooks` はこの緩さを持たない: 戻り値がそのまま `Runner.run(hooks=...)` の型契約
    （`RunHooksBase` の全メソッドを持つ）を満たす必要があり、`None` や部分実装を許容すると
    1 個だけ渡された場合にその要素自身を返す縮退が型契約を破る。したがって委譲ヘルパー
    `_delegate_agent_hook` は agent 単位専用とし、run 単位へは共通化しない。

SDK 追随手順:
    SDK（`agents.lifecycle`）の `RunHooksBase` / `AgentHooksBase` に新規 hook メソッドが
    追加された場合、本モジュールの対応する合成クラス（`_ChainedHooks` / `_ChainedAgentHooks`）
    に同名のオーバーライドを追加すること（追加漏れは新メソッドが合成対象から抜ける silent gap
    になるため）。

配置理由:
    `RunHooksBase` / `AgentHooksBase` の**サブクラス定義**が必要で `agents.lifecycle` の import
    が不可避なため、SDK 隔離（NFR-1）の窓口として `_adapters/` 配下に配置する。上位層は本
    モジュールの `chain_hooks` / `chain_agent_hooks` のみを介して合成し、SDK 実型には触れない。
"""

from __future__ import annotations

import inspect
from typing import Any

from agents.lifecycle import AgentHooksBase, RunHooksBase

# agent 単位フックのライフサイクルメソッド名。SDK から導出することで、SDK にメソッドが追加
# されたときに検証対象が自動で追随する（ハードコードするとドリフトし、新メソッドだけを持つ
# 要素が「1 つも持たない」と誤判定されうる）。
#
# `vars()` ではなく `dir()` を使う理由: `vars()` は当該クラスの `__dict__` のみを見るため、SDK が
# メソッドを中間基底クラスへ移すと導出が空になり、`on_*` を持つ正当な要素まで全て拒否される
# （実測: 空にすると 35 件のテストが一斉に落ち、原因の特定にノイズが乗る）。`dir()` は継承経由でも
# 拾うためこの失敗モードが生じない。SDK 0.17.4 では両者の結果は集合一致する。
_AGENT_HOOK_METHOD_NAMES: tuple[str, ...] = tuple(
    name for name in dir(AgentHooksBase) if name.startswith("on_")
)


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


async def _delegate_agent_hook(target: Any, method: str, *args: Any) -> None:
    """`target` の同名ライフサイクルメソッドへ委譲する（await 可能なら await する）。

    `target` が None、または当該メソッドを持たない場合は何もしない（部分実装を並べたまま
    合成できるようにするための薄い委譲）。callable 判定は行わないため、同名属性が callable
    でない場合は呼び出し時の `TypeError` がそのまま伝播する。

    属性解決の境界: 判定は `getattr(target, method, None)` で行うため、`property` や
    `__getattr__` 経由で当該メソッドを提供する要素がその内部で `AttributeError` を送出した
    場合も「メソッド未実装」と同じく skip される（fail-fast の対象外）。メソッド本体の実行中に
    生じた例外はそのまま伝播するため、両者の境界は属性解決が完了したかどうかで決まる。
    なお当該 `on_*` が要素の持つ唯一のライフサイクルメソッドである場合は、`chain_agent_hooks`
    の build 時検証（`on_*` を 1 つも持たない要素の拒否）で先に `TypeError` になる。

    Args:
        target: 委譲先の agent 単位フック（None 可・部分実装可）。
        method: 委譲先のライフサイクルメソッド名。
        *args: そのメソッドへ渡す引数。
    """
    if target is None:
        return
    fn = getattr(target, method, None)
    if fn is None:
        return
    result = fn(*args)
    if inspect.isawaitable(result):
        await result


class _ChainedAgentHooks(AgentHooksBase[Any, Any]):
    """複数の agent 単位フックを宣言順に順次呼び出す合成 hooks。

    設計原則:

    - **宣言順の順次実行**: 各メソッドは保持したフック列を左から右へ順に委譲する。
    - **fail-fast**: 素の順次 `await` により、前段が raise すれば後段の委譲に到達せず
      例外がそのまま伝播する（try/except は使わない）。
    - **引数無変更転送**: 受け取った引数をそのまま各フックの同名メソッドへ渡す。
    - **部分実装の受理**: 委譲は `_delegate_agent_hook` 経由で行うため、当該メソッドを
      持たない要素は skip される（`AgentHooksBase` の継承を要求しない）。
    """

    def __init__(self, hooks: tuple[Any, ...]) -> None:
        """合成対象のフック列を保持する新規インスタンスを初期化する。

        Args:
            hooks: 宣言順（左から右）の agent 単位フック列（`None` は除外済みを想定する）。
        """
        super().__init__()
        self._hooks = hooks

    async def on_start(self, context: Any, agent: Any) -> None:
        """各フックの `on_start` を宣言順に順次委譲する。

        Args:
            context: SDK の agent hook context。無変更で転送する。
            agent: 開始する Agent。無変更で転送する。
        """
        for hook in self._hooks:
            await _delegate_agent_hook(hook, "on_start", context, agent)

    async def on_end(self, context: Any, agent: Any, output: Any) -> None:
        """各フックの `on_end` を宣言順に順次委譲する。

        Args:
            context: SDK の agent hook context。無変更で転送する。
            agent: 終了した Agent。無変更で転送する。
            output: Agent の最終出力。無変更で転送する。
        """
        for hook in self._hooks:
            await _delegate_agent_hook(hook, "on_end", context, agent, output)

    async def on_handoff(self, context: Any, agent: Any, source: Any) -> None:
        """各フックの `on_handoff` を宣言順に順次委譲する。

        run 単位の `on_handoff(context, from_agent, to_agent)` とは引数の意味が異なる
        （agent 単位は handoff 先が `agent`・handoff 元が `source`）。

        Args:
            context: SDK `RunContextWrapper`。無変更で転送する。
            agent: handoff 先 Agent（本フックの対象 Agent）。無変更で転送する。
            source: handoff 元 Agent。無変更で転送する。
        """
        for hook in self._hooks:
            await _delegate_agent_hook(hook, "on_handoff", context, agent, source)

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        """各フックの `on_tool_start` を宣言順に順次委譲する。

        Args:
            context: SDK `RunContextWrapper` / `ToolContext`。無変更で転送する。
            agent: ツールを呼ぶ Agent。無変更で転送する。
            tool: 実行するツール。無変更で転送する。
        """
        for hook in self._hooks:
            await _delegate_agent_hook(hook, "on_tool_start", context, agent, tool)

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        """各フックの `on_tool_end` を宣言順に順次委譲する。

        Args:
            context: SDK `RunContextWrapper` / `ToolContext`。無変更で転送する。
            agent: ツールを呼んだ Agent。無変更で転送する。
            tool: 実行したツール。無変更で転送する。
            result: ツールの結果。無変更で転送する。
        """
        for hook in self._hooks:
            await _delegate_agent_hook(hook, "on_tool_end", context, agent, tool, result)

    async def on_llm_start(
        self,
        context: Any,
        agent: Any,
        system_prompt: Any,
        input_items: Any,
    ) -> None:
        """各フックの `on_llm_start` を宣言順に順次委譲する。

        Args:
            context: SDK `RunContextWrapper`。無変更で転送する。
            agent: LLM 呼び出しを行う Agent。無変更で転送する。
            system_prompt: システムプロンプト。無変更で転送する。
            input_items: 入力アイテム。無変更で転送する。
        """
        for hook in self._hooks:
            await _delegate_agent_hook(
                hook, "on_llm_start", context, agent, system_prompt, input_items
            )

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        """各フックの `on_llm_end` を宣言順に順次委譲する。

        Args:
            context: SDK `RunContextWrapper`。無変更で転送する。
            agent: LLM 応答を受けた Agent。無変更で転送する。
            response: SDK `ModelResponse`。無変更で転送する。
        """
        for hook in self._hooks:
            await _delegate_agent_hook(hook, "on_llm_end", context, agent, response)


def _validate_agent_hook_elements(hooks: tuple[Any, ...]) -> None:
    """各要素が agent 単位フックとして成立するかを build 時に検証する（fail-fast）。

    2 種類の誤用を宣言時点で落とす（いずれも従来は例外なく成立し、フックが黙って失われた）。
    詳細な理由は `_reject_run_scope_hooks` と `_reject_hooks_without_on_methods` の docstring に
    記述する。`None` は合成対象から除外される正当な値なので、どちらの検証もスキップする。

    Args:
        hooks: 利用者が渡した引数列（宣言順・`None` 除外前）。除外前を渡すのは、エラー
            メッセージの `hooks[i]` を利用者の引数位置と一致させるため。

    Raises:
        TypeError: 要素が run 単位フックの場合、または `on_*` を 1 つも持たない場合。
    """
    _reject_run_scope_hooks(hooks)
    _reject_hooks_without_on_methods(hooks)


def _reject_hooks_without_on_methods(hooks: tuple[Any, ...]) -> None:
    """`on_*` を 1 つも持たない要素があれば `TypeError` を送出する（build 時の fail-fast）。

    委譲は `getattr` で同名メソッドの有無を見て無ければ skip するため、`on_*` を 1 つも持たない
    要素を渡しても合成ラッパは成立し、全メソッドが no-op になる。`Agent.hooks` には正しく見える
    値が入るのに、フックが 1 度も発火しない状態が例外も警告もなく続く（silent gap）。実際に
    起きやすい誤用は `*` の付け忘れ（`chain_agent_hooks([h1, h2])` で list 自体が要素になる）と
    型違い・typo（文字列等）で、いずれもこの検証で検知できる（ADR-0017）。

    部分実装のサポートは壊さない。拒否条件は「7 つの `on_*` 名のいずれも持たない」であり、
    1 つでも持つ要素は従来どおり受理される。

    Args:
        hooks: 利用者が渡した引数列（宣言順・`None` 除外前）。`None` は除外対象なのでスキップする。

    Raises:
        TypeError: 要素が `None` でなく、かつ `on_*` を 1 つも持たない場合。
    """
    for index, hook in enumerate(hooks):
        if hook is None:
            continue
        if any(hasattr(hook, name) for name in _AGENT_HOOK_METHOD_NAMES):
            continue
        raise TypeError(
            f"chain_agent_hooks の要素は agent 単位フックのライフサイクルメソッドを"
            f"少なくとも 1 つ持つ必要があります: hooks[{index}] = {type(hook).__name__} は"
            f" {', '.join(_AGENT_HOOK_METHOD_NAMES)} のいずれも持ちません。包んでも全メソッドが"
            f"skip され、フックが 1 度も発火しない no-op になります。`*` の付け忘れ"
            f"（chain_agent_hooks([h1, h2])）ではないか確認してください。"
        )


def _reject_run_scope_hooks(hooks: tuple[Any, ...]) -> None:
    """要素に run 単位フックが含まれていれば `TypeError` を送出する（build 時の fail-fast）。

    run 単位フックを agent スロットへ渡すと、`AgentHooksBase` 非インスタンスとして合成ラッパへ
    包まれ SDK の型チェックも通過するため例外なく成立してしまう。その状態では `on_start` /
    `on_end` が run 単位の `on_agent_start` / `on_agent_end` と別名のため silent skip され、
    名前が衝突する `on_handoff` は agent 単位の `(context, agent, source)` が run 単位の
    `(context, from_agent, to_agent)` へ位置引数で渡って from/to が反転する。誤ったハンドオフ
    記録が例外なしで残るため、宣言時点で落とす（ADR-0017）。

    両基底（`AgentHooksBase` と `RunHooksBase`）を継承した要素も拒否する。`on_handoff` を
    1 メソッドしか持てず引数意味が一意に決まらないため、どちらとして解釈しても誤りうる。

    Args:
        hooks: 利用者が渡した引数列（宣言順・`None` 除外前）。`None` は `RunHooksBase` の
            インスタンスではないため判定はスキップされる。除外前を渡すのは、エラーメッセージの
            `hooks[i]` を利用者の引数位置と一致させるため（除外後の位置では `None` 混在時に
            誤った位置を案内してしまう）。

    Raises:
        TypeError: 要素に `RunHooksBase` インスタンスが含まれる場合。
    """
    for index, hook in enumerate(hooks):
        if isinstance(hook, RunHooksBase):
            raise TypeError(
                f"chain_agent_hooks は run 単位フック（agents.lifecycle.RunHooksBase）を"
                f"受け取れません: hooks[{index}] = {type(hook).__name__}。agent 単位は"
                f" on_start / on_end、run 単位は on_agent_start / on_agent_end とメソッド名が"
                f"異なり on_handoff の引数意味も異なるため、黙って無視されるメソッドと誤った"
                f"引数解釈が生じます。run 単位の合成には chain_hooks を使ってください。"
            )


def chain_agent_hooks(*hooks: Any) -> AgentHooksBase[Any, Any]:
    """複数の agent 単位フックを宣言順に合成した単一の `AgentHooksBase` を返す。

    `Agent.hooks`（`AgentSpec.hooks`）が単一スロットである制約に対し、複数フックを宣言順
    （引数の左から右）に順次呼び出す合成インスタンスを提供する。戻り値はそのまま
    `AgentSpec(hooks=...)` へ渡せる（追加の変換は不要）。

    受理する要素は次の 3 形:

    - **`AgentHooksBase` インスタンス**: 全 `on_*` を持つ通常の実装。
    - **部分実装（duck-typed）**: `AgentHooksBase` を継承せず `on_*` の一部だけを持つ
      オブジェクト。持たないメソッドの呼び出しは skip される。ただし `on_*` を**少なくとも
      1 つ**持つ必要がある（1 つも持たない要素は包んでも全メソッドが no-op になるため拒否する）。
    - **`None`**: 合成対象から除外される（条件分岐なしで無効化を表現できる）。

    `None` を除いた実効件数に応じて軽量最適化する:

    - **0 件**（`()` / `None` のみ）: 素の `AgentHooksBase()`（全メソッド no-op の SDK
      デフォルト）を返す。
    - **1 件かつ `AgentHooksBase` インスタンス**: そのフック自身をそのまま返す（`is` 一致・
      合成のオーバーヘッドを避ける）。
    - **それ以外**: `_ChainedAgentHooks` インスタンスを返す（実効 1 件の部分実装も
      `AgentHooksBase` 適合のため包む）。

    合成メソッドは fail-fast（前段 raise で後段スキップ・例外はそのまま伝播）で、引数は
    無変更で全フックに転送する。渡された引数列自体は変更しない。

    同期メソッドの許容範囲: 合成ラッパを経由する場合は `inspect.isawaitable` 判定により
    `async` でない `on_*` も呼び出せるが、**実効 1 件かつ `AgentHooksBase` インスタンスの
    経路はフック自身をそのまま返すため正規化を経ない**。この経路では SDK が戻り値を await
    するため、`on_*` を同期関数で定義したフックは `TypeError` になる。`AgentHooksBase` を
    継承するフックの `on_*` は `async` で定義すること（同期許容は部分実装のための緩さであり、
    フック件数に依存しない保証ではない）。

    run 単位フックは受理しない: 要素が `RunHooksBase` インスタンスの場合は `TypeError` で
    拒否する（両基底を継承した要素も拒否する。`on_handoff` の引数意味が一意に決まらないため）。
    要素型注釈が `Any` なのは部分実装（duck-typed）を受理するためであり、run 単位フックの
    受理を意味しない。run 単位の合成は `chain_hooks` を使う。

    Args:
        hooks: 宣言順（左から右）の agent 単位フック列（0 個以上・`None` 混在可）。

    Returns:
        `AgentSpec(hooks=...)` / `Agent(hooks=...)` に渡せる合成済み `AgentHooksBase`。

    Raises:
        TypeError: 要素に run 単位フック（`RunHooksBase` インスタンス）が含まれる場合、または
            要素が `None` でなく `on_*` を 1 つも持たない場合（`*` の付け忘れ・型違い等）。
    """
    _validate_agent_hook_elements(hooks)
    effective = tuple(hook for hook in hooks if hook is not None)
    if not effective:
        return AgentHooksBase()
    if len(effective) == 1 and isinstance(effective[0], AgentHooksBase):
        return effective[0]
    return _ChainedAgentHooks(effective)
