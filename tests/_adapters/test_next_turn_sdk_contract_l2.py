"""L2: Next-Turn Agent Override が依存する SDK 前提の構造契約トリップワイヤ。

到達時ハンドオフ禁止（FR-3）は、SDK の公式拡張点（`on_handoff` / `is_enabled`）への build 時
合成と、`RunContextWrapper` インスタンスをキーとする run 単位の到達記録で成り立つ。これらは
SDK 内部の振る舞い（run ごとの wrapper 生成・記録と参照へ渡る wrapper の同一性・handoff 有効性
のステップ毎評価・無効 handoff のモデル非提示）に依存しており、いずれかが SDK upgrade で
静かに変わると、**例外も型エラーも出ないまま禁止が効かなくなる / 到達記録が run を跨ぐ**。
本モジュールは SDK 実型・実関数を直接検査して、その退行を CI で fail させる（本機能のコードは
経由しない。合成側の挙動 pin は `test_next_turn_l2.py` / `test_next_turn_registry_l2.py` が担う）。

pin する前提:

1. run ごとの wrapper 生成: `ensure_context_wrapper` が plain な context / None を新しい
   `RunContextWrapper` へ包む（既存 wrapper はそのまま返す = 利用者が wrapper を自作して
   複数 run で再利用する形が非対応である構造的根拠）。
2. 記録と参照が同一 wrapper を受け取る: `get_handoffs(agent, context_wrapper)` は
   `is_enabled` を `(context_wrapper, agent)` で呼び、`execute_handoffs` は同じ
   `context_wrapper` を（fork や per-tool-call の派生インスタンスへ差し替えず）
   `Handoff.on_invoke_handoff` へ渡し、`on_invoke_handoff` はそれを `on_handoff` へ
   そのまま渡す。同一性は署名検査ではなく `execute_handoffs` の実駆動で確認する。
3. handoff 有効性のステップ毎評価: `get_handoffs` は callable の `is_enabled` を毎回評価する。
4. 無効 handoff のモデル非提示: `get_handoffs` の戻り値から除外される。
5. `handoff()` の on_handoff 署名検証: `input_type` の有無で要求される引数個数が 1 / 2 に
   固定される（合成 callable の arity をエッジ単位で選ぶ契約の根拠）。
6. agent-as-tool の fork は別インスタンス: `RunContextWrapper` の fork は新しい wrapper を
   返すため、親 run の到達記録が子 run へ漏れない。

`RunContextWrapper` が identity hash と weakref を持つこと（弱参照マップのキーにできること）も
併せて pin する。`agents` を import するため integration マーカー
（`tests/_adapters/test_resilience_exception_contract_l2.py` と同じ扱い）。
"""

from __future__ import annotations

import inspect
import weakref
from typing import Any

import pytest
from agents import Agent, Handoff, RunConfig, RunContextWrapper, RunHooks, Usage, handoff
from agents.exceptions import UserError
from agents.items import ModelResponse
from agents.run_internal.agent_runner_helpers import ensure_context_wrapper
from agents.run_internal.run_steps import NextStepHandoff, ToolRunHandoff
from agents.run_internal.turn_preparation import get_handoffs
from agents.run_internal.turn_resolution import execute_handoffs
from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel

pytestmark = pytest.mark.integration


class _HandoffPayload(BaseModel):
    """`input_type` あり経路の handoff 入力（最小の pydantic モデル）。"""

    reason: str


# ---------------------------------------------------------------------------
# 前提 1: run ごとに wrapper が新規生成される
# ---------------------------------------------------------------------------


def test_sdk_ensure_context_wrapperはplain_contextを新しいwrapperへ包む() -> None:
    """これが崩れると run ごとの記録分離が失われ、禁止がターンを越えて持続しうる。"""
    raw = {"tenant": "acme"}

    wrapped = ensure_context_wrapper(raw)

    assert isinstance(wrapped, RunContextWrapper)
    assert wrapped.context is raw


def test_sdk_ensure_context_wrapperは呼び出しごとに別インスタンスを返す() -> None:
    """run ごとに新しい wrapper が生成されるからこそ、並行 run の到達記録が構造的に分離される。"""
    first = ensure_context_wrapper(None)
    second = ensure_context_wrapper(None)

    assert first is not second


def test_sdk_ensure_context_wrapperは既存wrapperをそのまま返す() -> None:
    """利用者が wrapper を自作して複数 run で使い回す形を非対応とする根拠（記録が run を跨ぐ）。"""
    wrapper: RunContextWrapper[Any] = RunContextWrapper(context=None)

    assert ensure_context_wrapper(wrapper) is wrapper


def test_sdk_RunContextWrapperは弱参照マップのキーにできる() -> None:
    """identity hash と weakref が失われると、run 単位の到達記録ストア自体が成立しない。"""
    wrapper: RunContextWrapper[Any] = RunContextWrapper(context=None)

    assert weakref.ref(wrapper)() is wrapper
    assert hash(wrapper) == object.__hash__(wrapper)
    assert RunContextWrapper(context=None) != RunContextWrapper(context=None)


# ---------------------------------------------------------------------------
# 前提 2 / 3: 記録と参照へ同一 wrapper が渡り、有効性がステップ毎に評価される
# ---------------------------------------------------------------------------


async def test_sdk_get_handoffsはis_enabledをwrapperと所有agentで呼ぶ() -> None:
    """呼び出し引数が変わるとゲートが run を識別できず、禁止が効かない / 誤発動する。"""
    seen: list[tuple[Any, Any]] = []

    def _is_enabled(ctx: Any, agent: Any) -> bool:
        seen.append((ctx, agent))
        return True

    owner = Agent(name="triage")
    owner.handoffs = [handoff(Agent(name="billing"), is_enabled=_is_enabled)]
    wrapper: RunContextWrapper[Any] = RunContextWrapper(context=None)

    await get_handoffs(owner, wrapper)

    assert len(seen) == 1
    assert seen[0][0] is wrapper
    assert seen[0][1] is owner


async def test_sdk_get_handoffsは呼び出しのたびにis_enabledを評価する() -> None:
    """ステップ毎の再評価が無くなると、到達後に無効化しても同一ターン中に反映されない。"""
    calls: list[str] = []

    def _is_enabled(ctx: Any, agent: Any) -> bool:
        calls.append("evaluated")
        return True

    owner = Agent(name="triage")
    owner.handoffs = [handoff(Agent(name="billing"), is_enabled=_is_enabled)]
    wrapper: RunContextWrapper[Any] = RunContextWrapper(context=None)

    await get_handoffs(owner, wrapper)
    await get_handoffs(owner, wrapper)

    assert calls == ["evaluated", "evaluated"]


def test_sdk_execute_handoffsとget_handoffsのcontext_wrapper引数名を固定する() -> None:
    """引数名・引数順の pin（同一性そのものは次の実駆動テストが検証する）。

    キーワード引数名が変わると本機能の合成が渡す先を失うため、名前自体も検知対象に置く。
    """
    assert "context_wrapper" in inspect.signature(execute_handoffs).parameters
    assert list(inspect.signature(get_handoffs).parameters) == ["agent", "context_wrapper"]


async def test_sdk_execute_handoffsはis_enabledと同一wrapperをon_handoffへ渡す() -> None:
    """SDK 実関数を駆動し、ゲート参照側と到達記録側が受け取る wrapper の同一性（`is`）を pin。

    `no_handoff_on_arrival` は「`on_handoff` が書いたキー」を「`is_enabled` が読む」ことで
    成立する。SDK が `on_invoke_handoff` へ per-tool-call の派生 wrapper（`RunContextWrapper`
    のサブクラスを fork したもの等）を渡すようになると、型エラーも例外も出ないまま記録先と
    参照先のキーがずれて禁止が静かに機能停止する。署名検査では通り抜けるため、`execute_handoffs`
    を実際に呼び、`context_wrapper` に渡したインスタンスがそのまま `on_handoff` へ届くことを
    確認する。
    """
    gate_seen: list[Any] = []
    record_seen: list[Any] = []

    def _is_enabled(ctx: Any, agent: Any) -> bool:
        gate_seen.append(ctx)
        return True

    def _on_handoff(ctx: Any) -> None:
        record_seen.append(ctx)

    built = handoff(Agent(name="billing"), on_handoff=_on_handoff, is_enabled=_is_enabled)
    owner = Agent(name="triage")
    owner.handoffs = [built]
    wrapper: RunContextWrapper[Any] = RunContextWrapper(context=None)

    await get_handoffs(owner, wrapper)
    tool_call = ResponseFunctionToolCall(
        id="fc_next_turn",
        type="function_call",
        call_id="call_next_turn",
        name=built.tool_name,
        arguments="{}",
    )
    result = await execute_handoffs(
        public_agent=owner,
        original_input="hi",
        pre_step_items=[],
        new_step_items=[],
        new_response=ModelResponse(output=[tool_call], usage=Usage(), response_id=None),
        run_handoffs=[ToolRunHandoff(handoff=built, tool_call=tool_call)],
        hooks=RunHooks(),
        context_wrapper=wrapper,
        run_config=RunConfig(),
    )

    assert isinstance(result.next_step, NextStepHandoff)
    assert result.next_step.new_agent.name == "billing"
    assert len(gate_seen) == 1
    assert len(record_seen) == 1
    assert gate_seen[0] is wrapper
    assert record_seen[0] is wrapper


async def test_sdk_on_invoke_handoffは受け取ったwrapperをon_handoffへ渡す() -> None:
    """ここが別インスタンスになると、記録したキーとゲートが参照するキーが一致しなくなる。"""
    seen: list[Any] = []

    def _on_handoff(ctx: Any) -> None:
        seen.append(ctx)

    built = handoff(Agent(name="billing"), on_handoff=_on_handoff)
    wrapper: RunContextWrapper[Any] = RunContextWrapper(context=None)

    await built.on_invoke_handoff(wrapper, "{}")

    assert len(seen) == 1
    assert seen[0] is wrapper


# ---------------------------------------------------------------------------
# 前提 4: 無効 handoff はモデルへ提示されない
# ---------------------------------------------------------------------------


async def test_sdk_get_handoffsは無効なhandoffを戻り値から除外する() -> None:
    """除外されなくなると、ゲートが False を返してもモデルに handoff が提示され続ける。"""
    owner = Agent(name="triage")
    owner.handoffs = [
        handoff(Agent(name="billing"), is_enabled=lambda ctx, agent: True),
        handoff(Agent(name="tech"), is_enabled=lambda ctx, agent: False),
    ]
    wrapper: RunContextWrapper[Any] = RunContextWrapper(context=None)

    enabled = await get_handoffs(owner, wrapper)

    assert [h.agent_name for h in enabled] == ["billing"]


async def test_sdk_get_handoffsは素のAgent直appendを毎回内部でHandoff化する() -> None:
    """内部生成される handoff には合成の差し込み口が無く、build 時の昇格が必要である根拠。"""
    owner = Agent(name="triage")
    owner.handoffs = [Agent(name="billing")]
    wrapper: RunContextWrapper[Any] = RunContextWrapper(context=None)

    first = await get_handoffs(owner, wrapper)
    second = await get_handoffs(owner, wrapper)

    assert isinstance(first[0], Handoff)
    assert first[0] is not second[0]


# ---------------------------------------------------------------------------
# 前提 5: handoff() の on_handoff 署名検証（arity 契約の根拠）
# ---------------------------------------------------------------------------


def test_sdk_handoffはinput_typeなしで2引数のon_handoffを拒否する() -> None:
    """`input_type` なしのエッジへ 2 引数の合成を渡すと build が落ちる（arity 選択の根拠）。"""
    with pytest.raises(UserError):
        handoff(Agent(name="billing"), on_handoff=lambda ctx, payload: None)


def test_sdk_handoffはinput_typeありで1引数のon_handoffを拒否する() -> None:
    """`input_type` ありのエッジへ 1 引数の合成を渡すと build が落ちる（arity 選択の根拠）。"""
    with pytest.raises(UserError):
        handoff(Agent(name="billing"), on_handoff=lambda ctx: None, input_type=_HandoffPayload)


# ---------------------------------------------------------------------------
# 前提 6: agent-as-tool の fork は別 wrapper
# ---------------------------------------------------------------------------


def test_sdk_fork_with_tool_inputは新しいwrapperを返す() -> None:
    """fork が同一インスタンスになると、親 run の到達記録が子 run（agent-as-tool）へ漏れる。"""
    wrapper: RunContextWrapper[Any] = RunContextWrapper(context=None)

    forked = wrapper._fork_with_tool_input({"query": "x"})  # noqa: SLF001

    assert forked is not wrapper
    assert isinstance(forked, RunContextWrapper)


def test_sdk_fork_without_tool_inputは新しいwrapperを返す() -> None:
    """tool 入力なしの fork も別インスタンスであり、記録の共有経路にならない。"""
    wrapper: RunContextWrapper[Any] = RunContextWrapper(context=None)

    forked = wrapper._fork_without_tool_input()  # noqa: SLF001

    assert forked is not wrapper
