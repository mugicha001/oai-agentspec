"""L2: context 対応 output guardrail ビルダ（`build_context_output_guardrail`）の SDK 結合検証。

SDK の出力 guardrail シグネチャ `(context, agent, agent_output)` のうち、既存
`build_output_guardrail` が捨てている `context` / `agent` を検知器へそのまま渡す接着を検証する。
SDK の `OutputGuardrail.run` 越しに駆動して `context` / `agent` の同一性（`is`）・テキスト化
（`_text_of`）・`Detection` → `GuardrailFunctionOutput` の写像・awaitable 正規化を pin し、既存の
1 引数検知契約（`build_output_guardrail`）が不変であることも最小限確認する。

`_adapters/__init__.py` 集約窓口からの再エクスポート（`__all__` 掲載・`is` 同一・既存 guardrails 系
メンバの非欠落）も本ファイルで pin する。
"""

from __future__ import annotations

from typing import Any

import pytest
from agents import OutputGuardrail, RunContextWrapper

from oai_agentspec import _adapters as adapters
from oai_agentspec._adapters.guardrails import (
    _text_of,
    build_context_output_guardrail,
    build_output_guardrail,
)
from oai_agentspec.runtime.guardrails._detectors import Detection

pytestmark = pytest.mark.integration


class _Agent:
    """SDK から渡る agent の代役（同一性照合のみに使う不透明スタブ）。"""


def _wrapper(payload: Any = None) -> RunContextWrapper[Any]:
    """`RunContextWrapper` を組む（検知器へ wrapper のまま届くことの照合用）。"""
    return RunContextWrapper(context=payload)


# ----------------------------------------------------------------------
# 戻り値型 / context・agent の透過
# ----------------------------------------------------------------------


def test_build_context_output_guardrailはSDK互換のOutputGuardrailを返す() -> None:
    """戻り値が SDK 互換 `OutputGuardrail` のインスタンスであること（注釈の実値整合）。"""
    guardrail = build_context_output_guardrail("cog", lambda c, a, t: Detection(triggered=False))
    assert isinstance(guardrail, OutputGuardrail)
    assert guardrail.get_name() == "cog"


@pytest.mark.asyncio
async def test_contextとagentが検知器へそのまま渡る() -> None:
    """SDK から渡った `context` / `agent` が破棄されず同一オブジェクトのまま detect に届く。

    呼び出し回数と戻り値だけを見るテストでは「context を捨てて別の値（None 等）で呼ぶ」変異が
    生存するため、引数の中身を `is` で照合する。
    """
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def detect(*args: Any, **kwargs: Any) -> Detection:
        calls.append((args, dict(kwargs)))
        return Detection(triggered=False)

    context = _wrapper({"canary": "T1"})
    agent = _Agent()
    guardrail = build_context_output_guardrail("cog", detect)

    await guardrail.run(context=context, agent=agent, agent_output="answer")

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs == {}
    assert len(args) == 3
    assert args[0] is context
    assert args[1] is agent
    assert args[2] == "answer"


@pytest.mark.asyncio
async def test_非str出力は_text_of相当でテキスト化されて渡る() -> None:
    """`agent_output` が非 str の場合 `_text_of` 相当（`str(...)`）でテキスト化して渡る。"""
    seen: list[str] = []

    def detect(context: Any, agent: Any, text: str) -> Detection:
        seen.append(text)
        return Detection(triggered=False)

    guardrail = build_context_output_guardrail("cog", detect)
    payload = [1, 2, 3]
    await guardrail.run(context=_wrapper(), agent=_Agent(), agent_output=payload)

    assert seen == [_text_of(payload)]
    assert seen == [str(payload)]


# ----------------------------------------------------------------------
# Detection → GuardrailFunctionOutput の写像（既定値と異なる sentinel を使う）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_Detectionのtriggered_reason_infoがguardrail出力へ写る() -> None:
    """`triggered` / `reason` / `info` が `GuardrailFunctionOutput` へ正しく写る。

    既定値（`triggered=False` / `reason=None` / `info=None`）と異なる sentinel を使い、
    「固定値を返す」変異と区別できる形で pin する。
    """
    detection = Detection(triggered=True, reason="canary-leak-sentinel", info={"matched": ["T1"]})
    guardrail = build_context_output_guardrail("cog", lambda c, a, t: detection)

    result = await guardrail.run(context=_wrapper(), agent=_Agent(), agent_output="oops")

    assert result.output.tripwire_triggered is True
    assert result.output.output_info == {
        "reason": "canary-leak-sentinel",
        "info": {"matched": ["T1"]},
    }


@pytest.mark.asyncio
async def test_非trip時はtripwireがFalseで理由と付帯情報が空のまま載る() -> None:
    """`triggered=False` は `tripwire_triggered=False`（reason / info は None のまま載る）。"""
    guardrail = build_context_output_guardrail("cog", lambda c, a, t: Detection(triggered=False))

    result = await guardrail.run(context=_wrapper(), agent=_Agent(), agent_output="clean")

    assert result.output.tripwire_triggered is False
    assert result.output.output_info == {"reason": None, "info": None}


# ----------------------------------------------------------------------
# awaitable 正規化（`_call_detect` と同型）
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_awaitableを返す検知器はawaitされて正規化される() -> None:
    """detect が awaitable を返す場合も await されて `Detection` に正規化される。"""

    async def detect(context: Any, agent: Any, text: str) -> Detection:
        return Detection(triggered="leak" in text, reason="async ctx detect")

    guardrail = build_context_output_guardrail("cog", detect)

    tripped = await guardrail.run(context=_wrapper(), agent=_Agent(), agent_output="a leak here")
    assert tripped.output.tripwire_triggered is True
    assert tripped.output.output_info["reason"] == "async ctx detect"

    passed = await guardrail.run(context=_wrapper(), agent=_Agent(), agent_output="clean")
    assert passed.output.tripwire_triggered is False


@pytest.mark.asyncio
async def test_async__call__を持つ検知器オブジェクトもawaitされる() -> None:
    """`async __call__` を持つ callable オブジェクトでも戻り値 awaitable 判定で await される。

    `inspect.iscoroutinefunction` は async `__call__` オブジェクトに False を返すため、型判定の
    実装では未 await の coroutine が後段へ流れて壊れる（戻り値判定なら取りこぼさない）。
    """

    class _AsyncDetector:
        async def __call__(self, context: Any, agent: Any, text: str) -> Detection:
            return Detection(triggered="leak" in text, reason="async obj")

    guardrail = build_context_output_guardrail("cog", _AsyncDetector())

    tripped = await guardrail.run(context=_wrapper(), agent=_Agent(), agent_output="a leak here")
    assert tripped.output.tripwire_triggered is True
    assert tripped.output.output_info["reason"] == "async obj"


# ----------------------------------------------------------------------
# 既存 1 引数検知契約（`build_output_guardrail`）の不変性
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_既存build_output_guardrailは1引数検知契約のまま動く() -> None:
    """context を受け取らない旧経路（`Callable[[str], Detection]`）が壊れていないこと。

    context 対応版の追加が既存契約へ波及していないことの最小限の確認（写像の詳細は
    `test_guardrails_l2.py` が担う）。
    """
    seen: list[tuple[Any, ...]] = []

    def detect(*args: Any) -> Detection:
        seen.append(args)
        return Detection(triggered=True, reason="legacy")

    guardrail = build_output_guardrail("og", detect)
    result = await guardrail.run(context=_wrapper(), agent=_Agent(), agent_output="answer")

    assert seen == [("answer",)]  # テキスト 1 引数のみで呼ばれる（context / agent は渡らない）
    assert result.output.tripwire_triggered is True
    assert result.output.output_info["reason"] == "legacy"


# ----------------------------------------------------------------------
# `_adapters/__init__.py` 集約窓口からの再エクスポート
# ----------------------------------------------------------------------

#: `_adapters/guardrails.py` 由来で集約窓口に載っているべき公開名（過小側の変異検知用）。
_GUARDRAIL_ADAPTER_EXPORTS = frozenset(
    {
        "OnTrip",
        "attach_tool_guardrails",
        "build_async_input_guardrail",
        "build_async_output_guardrail",
        "build_context_output_guardrail",
        "build_input_guardrail",
        "build_output_guardrail",
        "build_tool_input_guardrail",
        "build_tool_output_guardrail",
        "guardrail_boundary",
        "guardrail_visible_name",
        "run_judge_prompt",
    }
)


def test_build_context_output_guardrailは_adapters窓口からimportできる() -> None:
    """`from oai_agentspec._adapters import build_context_output_guardrail` が成功する。

    薄い `__init__` 集約窓口の原則に従い、`guardrails.py` の公開関数は他 11 件と同様に
    `_adapters/__init__.py` の import と `__all__` へ載せる（本関数だけ欠けている状態を弾く）。
    """
    from oai_agentspec._adapters import build_context_output_guardrail as reexported

    assert "build_context_output_guardrail" in adapters.__all__
    # 別実体を挟む変異（薄いラッパを作る等）を kill するため `is` 同一を要求する。
    assert reexported is build_context_output_guardrail
    assert adapters.build_context_output_guardrail is build_context_output_guardrail


def test_adapters窓口のguardrails系エクスポートが減っていない() -> None:
    """既存の guardrails 系 `__all__` メンバが 1 件も欠けていない（過小側の変異検知）。"""
    missing = _GUARDRAIL_ADAPTER_EXPORTS - set(adapters.__all__)
    assert missing == set()
    # 追加時の重複混入（同名 2 度載せ）も弾く。
    assert len(adapters.__all__) == len(set(adapters.__all__))
    for name in sorted(_GUARDRAIL_ADAPTER_EXPORTS):
        assert hasattr(adapters, name)
