"""L2: 新 signature の意図予測アダプタ（`_adapters.intent.run_intent_prompt`）を pin する。

Issue #24 core-logic 再設計により、`run_intent_prompt` は
`(model, system, history_items, user_content, *, context) -> str` に変更される。
本テストは新 signature を前提とした RED 起点であり、実装（`_adapters/intent.py`）は
未変更のため import または呼び出しで fail する想定。

検証項目:

- 単一発話・履歴付き・RunContext ありの入力伝播
- `system=""` は Agent.instructions=None として扱われる
- 戻り値の str 型契約
- モデル応答 None は空文字に変換
- `agents` の module 属性を関数内遅延 import として保つ（module 属性に expose しない）
- `RunContextWrapper` を渡した場合、`.context` を lib 側で展開して `Runner.run` に forward
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from oai_agentspec._adapters import intent as intent_module
from oai_agentspec._adapters.intent import run_intent_prompt

from _helpers.intent_fakes import RecordingFakeModel

pytestmark = pytest.mark.integration


async def test_単一発話_system_instructions_と_input_が_モデルに渡る() -> None:
    """新 signature: history 空・単一 user 発話が input として渡る。"""
    model = RecordingFakeModel(text="RESP")
    result = await run_intent_prompt(
        model,
        system="sys",
        history_items=(),
        user_content="hi",
    )
    assert isinstance(result, str)
    assert result == "RESP"
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["system_instructions"] == "sys"
    assert call["input"] == [{"role": "user", "content": "hi"}]


async def test_履歴付き_history_items_と_user_content_が_連結される() -> None:
    """history_items の後ろに現在の user 発話が連結され input として渡る。"""
    model = RecordingFakeModel(text="OK")
    history = (
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    )
    await run_intent_prompt(
        model,
        system="sys",
        history_items=history,
        user_content="hi",
    )
    assert model.calls[0]["input"] == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "hi"},
    ]


async def test_RunContext_あり_例外なく実行できる() -> None:
    """plain dict の context を渡しても例外なく完了する（Runner.run へ forward）。"""
    model = RecordingFakeModel(text="OK")
    result = await run_intent_prompt(
        model,
        system="sys",
        history_items=(),
        user_content="hi",
        context={"tenant": "acme"},
    )
    assert isinstance(result, str)


async def test_system_空文字は_Agent_instructions_None_扱い() -> None:
    """`system=""` を渡すと SDK には `instructions=None` として扱われる。"""
    model = RecordingFakeModel(text="OK")
    await run_intent_prompt(
        model,
        system="",
        history_items=(),
        user_content="hi",
    )
    # Agent(instructions=None) の場合、Runner は system_instructions を渡さない or None。
    assert model.calls[0]["system_instructions"] is None


async def test_戻り値は_str_型() -> None:
    """`run_intent_prompt` の戻り値は必ず str（None を含む型でない）。"""
    model = RecordingFakeModel(text="hello")
    result = await run_intent_prompt(
        model,
        system="sys",
        history_items=(),
        user_content="hi",
    )
    assert isinstance(result, str)


async def test_モデル応答が_None_の場合は空文字を返す() -> None:
    """final_output が None（空出力）の場合、adapter は空文字列を返す。"""
    model = RecordingFakeModel(text=None)
    result = await run_intent_prompt(
        model,
        system="sys",
        history_items=(),
        user_content="hi",
    )
    assert result == ""


def test_agents_の_import_は関数内遅延_モジュール属性に_expose_しない() -> None:
    """`agents.Agent`/`Runner` はモジュール top-level 属性として expose しない（NFR-1）。"""
    # 関数内 import 契約: module 属性として Agent/Runner を持たない。
    assert not hasattr(intent_module, "Agent"), (
        "`agents.Agent` が module 属性として expose されている（関数内遅延 import 契約違反）"
    )
    assert not hasattr(intent_module, "Runner"), (
        "`agents.Runner` が module 属性として expose されている（関数内遅延 import 契約違反）"
    )


async def test_RunContextWrapper_を渡した場合_context_が展開される() -> None:
    """`RunContextWrapper` を context に渡した場合、lib 側で `.context` を開いて forward する。"""
    from agents import RunContextWrapper

    raw: dict[str, Any] = {"k": "v"}
    wrapper = RunContextWrapper(context=raw)
    model = RecordingFakeModel(text="OK")
    result = await run_intent_prompt(
        model,
        system="sys",
        history_items=(),
        user_content="hi",
        context=wrapper,
    )
    # 少なくとも例外なく実行完了する（Runner.run 側で二重 wrap を回避する契約）。
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 履歴のみでの意図分類 (Issue #24): 空 user_content の turn スキップ / fail-fast
# ---------------------------------------------------------------------------


async def test_empty_user_content_skips_user_turn() -> None:
    """`user_content=""` の場合、空 user turn を append せず履歴だけを input として送る。"""
    model = RecordingFakeModel(text="OK")
    await run_intent_prompt(
        model,
        system="sys",
        history_items=({"role": "user", "content": "a"},),
        user_content="",
    )
    assert len(model.calls) == 1
    assert model.calls[0]["input"] == [{"role": "user", "content": "a"}]


async def test_empty_user_content_and_empty_history_raises() -> None:
    """履歴も user_content も空の場合は ValueError で fail-fast する。

    メッセージには "utterance" と "history" の両方の語を含む。
    """
    model = RecordingFakeModel(text="OK")
    with pytest.raises(ValueError) as exc_info:
        await run_intent_prompt(
            model,
            system="sys",
            history_items=(),
            user_content="",
        )
    message = str(exc_info.value)
    assert "utterance" in message
    assert "history" in message
    # モデル呼び出しには到達しない
    assert model.calls == []


async def test_whitespace_user_content_is_sent() -> None:
    """空白のみの user_content (" ") は truthy なので user turn として送られる。

    スキップ対象は空文字 ("") のみであることを pin する。
    """
    model = RecordingFakeModel(text="OK")
    await run_intent_prompt(
        model,
        system="sys",
        history_items=(),
        user_content=" ",
    )
    assert model.calls[0]["input"] == [{"role": "user", "content": " "}]


def test_signature_に_history_items_と_user_content_と_context_が含まれる() -> None:
    """新 signature の pin: parameter 名で契約を確認する（呼び出し互換の保証）。"""
    sig = inspect.signature(run_intent_prompt)
    params = sig.parameters
    assert "model" in params
    assert "system" in params
    assert "history_items" in params
    assert "user_content" in params
    assert "context" in params
    # context は keyword-only
    assert params["context"].kind is inspect.Parameter.KEYWORD_ONLY
