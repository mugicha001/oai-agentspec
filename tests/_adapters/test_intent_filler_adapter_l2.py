"""L2: パラメータ予測の委譲窓口（`_adapters.intent.run_filler_prompt`）を pin する。

Issue #88 第 2 段タスク 2-1 の RED 起点。設計 §3.4（第 5 回レビュー WARN-2 で全面差し替え）に
従い、`run_filler_prompt(agent, history_items, user_content, *, context=None)` は
**構築済みの不透明な agent を 1 回だけ走らせ** `(応答テキスト, AgentRunUsage)` を返す。
予測エージェント専用 `AgentRegistry` の生成・`AgentSpec` の宣言・ガードレール登録名の解決は
上位層（`runtime/intent/_predict.py`・タスク 2-4）の責務であり、本ファイルは
「その registry が解決した実体を adapter がそのまま走らせる」ところを検証する。

検証項目:

- `Runner.run` が 1 回だけ呼ばれ、引数（args / kwargs）全体が期待テーブルと一致すること
- `max_turns` が内部定数 `FILLER_MAX_TURNS`（値 1）で渡され、超過時の SDK 例外が伝播すること
- `session` を渡す口を持たないこと（signature と実引数の双方で pin）
- ガードレール発火時に SDK 例外が後退なく伝播すること（入力 / 出力の両境界）
- `AgentRunUsage` への詰め替え（usage を取得できた場合 / 取得できない場合）
- 応答 None の空文字化と `ValueError` による fail-fast
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping
from typing import Any

import agents
import pytest
from agents import Agent, RunContextWrapper
from agents.exceptions import (
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    OutputGuardrailTripwireTriggered,
)
from agents.items import ModelResponse
from agents.usage import Usage

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec._adapters.intent import (
    FILLER_MAX_TURNS,
    AgentRunUsage,
    run_filler_prompt,
)
from oai_agentspec.runtime.guardrails import GuardrailRegistry

from _helpers.fake_model import FakeModel
from _helpers.intent_fakes import RecordingFakeModel

pytestmark = pytest.mark.integration

_FILLER_NAME = "oai-agentspec-param-filler"


# ----------------------------------------------------------------------
# helper: 予測エージェント相当の実体と Runner.run spy
# ----------------------------------------------------------------------


def _filler_agent(
    model: Any,
    *,
    guardrails: list[str] | None = None,
    guardrail_registry: GuardrailRegistry | None = None,
) -> Any:
    """予測エージェント専用 `AgentRegistry` から実体を解決して返す（上位層の代役）。

    Args:
        model: 予測に使う Model 相当（FakeModel 等）。
        guardrails: `AgentSpec.guardrails` に載せる登録名。
        guardrail_registry: 登録名の解決元。

    Returns:
        `AgentRegistry.get(_FILLER_NAME)` の戻り値（不透明な Agent 実体）。
    """
    registry = AgentRegistry(guardrail_registry=guardrail_registry)
    registry.register(
        AgentSpec(
            name=_FILLER_NAME,
            instructions="fill the parameters",
            model=model,
            guardrails=list(guardrails or []),
        )
    )
    return registry.get(_FILLER_NAME)


def _usage_response(text: str, usage: Usage) -> ModelResponse:
    """任意の `Usage` を載せたテキスト応答を作る（usage 詰め替えの検証用）。"""
    from oai_agentspec.runtime.deterministic import text_response

    return dataclasses.replace(text_response(text), usage=usage)


@dataclasses.dataclass(frozen=True)
class _StubRunResult:
    """`Runner.run` の戻り値のうち adapter が読む 2 属性だけを持つスタブ。"""

    final_output: Any
    raw_responses: list[ModelResponse]


def _stub_run(final_output: Any, responses: list[ModelResponse]) -> Any:
    """固定の `raw_responses` を返す `Runner.run` 差し替え関数を作る。

    `max_turns=1` では複数応答を実際に得られないため、usage 合算の検証に使う。

    Args:
        final_output: スタブ結果の最終出力。
        responses: `raw_responses` として返す応答のリスト。

    Returns:
        `Runner.run` と同じ呼び出し形で待機可能な差し替え関数。
    """

    async def _run(*args: Any, **kwargs: Any) -> _StubRunResult:
        return _StubRunResult(final_output=final_output, raw_responses=responses)

    return _run


@pytest.fixture
def runner_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """`Runner.run` の呼び出しを `(args, kwargs)` で記録し本体へ委譲する spy を仕掛ける。

    Returns:
        呼び出し記録のリスト（呼び出し順）。期待テーブルと `==` で全体照合する。
    """
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    original = agents.Runner.run

    async def _spy(*args: Any, **kwargs: Any) -> Any:
        calls.append((tuple(args), dict(kwargs)))
        return await original(*args, **kwargs)

    monkeypatch.setattr(agents.Runner, "run", _spy)
    return calls


# ----------------------------------------------------------------------
# 呼び出し回数と引数全体
# ----------------------------------------------------------------------


async def test_Runner_run_は_1_回だけ呼ばれ引数全体が期待テーブルと一致する(
    runner_calls: list[tuple[tuple[Any, ...], dict[str, Any]]],
) -> None:
    """履歴 + 現在発話を連結した input・context・`max_turns=1` で 1 回だけ走る。

    「正しい戻り値を返しつつ利用者の設定を捨てて別の値で呼ぶ」変異を検知するため、
    件数だけでなく `(args, kwargs)` 全体を期待テーブルと `==` で照合する。
    """
    agent = _filler_agent(FakeModel().queue_text("RESP"))
    history: tuple[Mapping[str, Any], ...] = (
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    )

    text, _usage = await run_filler_prompt(agent, history, "fill me")

    assert text == "RESP"
    assert runner_calls == [
        (
            (agent,),
            {
                "input": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                    {"role": "user", "content": "fill me"},
                ],
                "context": None,
                "max_turns": 1,
            },
        )
    ]
    # 複製ではなく渡された実体そのものが走ること（別 registry / session なしの構造的保証）。
    assert runner_calls[0][0][0] is agent


async def test_RunContextWrapper_は開封して_forward_される(
    runner_calls: list[tuple[tuple[Any, ...], dict[str, Any]]],
) -> None:
    """`RunContextWrapper` を渡すと `.context` を開いて `Runner.run` へ渡す。"""
    raw: dict[str, Any] = {"tenant": "acme"}
    agent = _filler_agent(FakeModel().queue_text("RESP"))

    await run_filler_prompt(agent, (), "fill me", context=RunContextWrapper(context=raw))

    assert len(runner_calls) == 1
    assert runner_calls[0][1]["context"] is raw


async def test_session_を渡す口を持たない(
    runner_calls: list[tuple[tuple[Any, ...], dict[str, Any]]],
) -> None:
    """予測エージェントに会話履歴を渡さない（`session` kwarg を一切渡さない）。"""
    agent = _filler_agent(FakeModel().queue_text("RESP"))

    await run_filler_prompt(agent, (), "fill me")

    assert "session" not in runner_calls[0][1]
    assert "session" not in inspect.signature(run_filler_prompt).parameters


# ----------------------------------------------------------------------
# max_turns（内部定数）と超過時の伝播
# ----------------------------------------------------------------------


def test_FILLER_MAX_TURNS_は内部定数_1_で公開引数に持たない() -> None:
    """`max_turns` は値 1 の module 定数であり、関数の引数として公開しない。"""
    assert FILLER_MAX_TURNS == 1
    assert isinstance(FILLER_MAX_TURNS, int)
    assert "max_turns" not in inspect.signature(run_filler_prompt).parameters


async def test_max_turns_は_FILLER_MAX_TURNS_で渡される(
    runner_calls: list[tuple[tuple[Any, ...], dict[str, Any]]],
) -> None:
    """`Runner.run` へ渡る `max_turns` が内部定数と一致する。"""
    agent = _filler_agent(FakeModel().queue_text("RESP"))

    await run_filler_prompt(agent, (), "fill me")

    assert runner_calls[0][1]["max_turns"] == FILLER_MAX_TURNS


async def test_MaxTurnsExceeded_は握り潰さず伝播する(monkeypatch: pytest.MonkeyPatch) -> None:
    """ターン上限超過の SDK 例外は catch せずそのまま呼び出し元へ抜ける。"""

    async def _raise(*args: Any, **kwargs: Any) -> Any:
        raise MaxTurnsExceeded("Max turns 1 exceeded")

    monkeypatch.setattr(agents.Runner, "run", _raise)
    agent = _filler_agent(FakeModel().queue_text("RESP"))

    with pytest.raises(MaxTurnsExceeded, match="Max turns 1 exceeded") as excinfo:
        await run_filler_prompt(agent, (), "fill me")
    assert type(excinfo.value) is MaxTurnsExceeded


# ----------------------------------------------------------------------
# ガードレール（登録名の解決結果を境界へ振り分け・発火時は伝播）
# ----------------------------------------------------------------------


async def test_guardrails_未宣言の構成では挙動が変わらない() -> None:
    """ガードレール未宣言（既定・opt-in）の agent は通常どおり応答を返す。

    登録名の境界振り分け（装着 0 件 / 実体同一性）そのものは上位層の責務であり、
    `tests/runtime/intent/test_predict_l1.py`（タスク 2-4）が pin する。
    """
    agent = _filler_agent(FakeModel().queue_text("RESP"))

    text, _usage = await run_filler_prompt(agent, (), "fill me")

    assert text == "RESP"


async def test_入力ガードレール発火の例外は後退せず伝播する() -> None:
    """入力境界の発火は安全事象として例外のまま抜ける（既定値への後退を適用しない）。"""
    guardrails = GuardrailRegistry()
    guardrails.predicate_guardrail(lambda text: "bad" in text, on="input", name="filler_in")
    agent = _filler_agent(
        FakeModel().queue_text("RESP"),
        guardrails=["filler_in"],
        guardrail_registry=guardrails,
    )

    with pytest.raises(InputGuardrailTripwireTriggered, match="triggered tripwire") as excinfo:
        await run_filler_prompt(agent, (), "bad request")
    assert type(excinfo.value) is InputGuardrailTripwireTriggered


async def test_出力ガードレール発火の例外は後退せず伝播する() -> None:
    """出力境界の発火も同様に例外のまま抜ける（応答テキストを返さない）。"""
    guardrails = GuardrailRegistry()
    guardrails.predicate_guardrail(lambda text: "danger" in text, on="output", name="filler_out")
    agent = _filler_agent(
        FakeModel().queue_text("danger zone"),
        guardrails=["filler_out"],
        guardrail_registry=guardrails,
    )

    with pytest.raises(OutputGuardrailTripwireTriggered, match="triggered tripwire") as excinfo:
        await run_filler_prompt(agent, (), "fill me")
    assert type(excinfo.value) is OutputGuardrailTripwireTriggered


# ----------------------------------------------------------------------
# AgentRunUsage への詰め替え
# ----------------------------------------------------------------------


async def test_usage_を取得できた場合は件数とトークンを詰め替える() -> None:
    """`raw_responses` の件数と全応答合計のトークン数を `AgentRunUsage` に載せる。"""
    model = FakeModel()
    model.responses.append(
        _usage_response(
            "RESP",
            Usage(requests=1, input_tokens=3, output_tokens=4, total_tokens=7),
        )
    )
    agent = _filler_agent(model)

    text, usage = await run_filler_prompt(agent, (), "fill me")

    assert text == "RESP"
    assert usage == AgentRunUsage(model_calls=1, input_tokens=3, output_tokens=4)


async def test_usage_未取得なら_tokens_は_None_で_model_calls_は残る() -> None:
    """`requests==0` かつ `total_tokens==0` は未取得と判定しトークンを None にする。

    SDK の `Usage` は全フィールド非 Optional・既定 0 で 0 と未取得を型で区別できないため、
    件数（`len(raw_responses)`）だけは usage の内容に依存せず実測値を残す。
    """
    agent = _filler_agent(FakeModel().queue_text("RESP"))

    _text, usage = await run_filler_prompt(agent, (), "fill me")

    assert usage == AgentRunUsage(model_calls=1, input_tokens=None, output_tokens=None)


async def test_複数応答の_tokens_は合算され_model_calls_は件数になる(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`raw_responses` が 2 件なら件数は 2、トークンは全応答の合計になる。

    `max_turns=1` では 2 応答を実際に得られないため `Runner.run` を差し替える。
    「先頭だけ」「最後だけ」を返す変異が生存しないよう 2 件の値は異なる数値にする。
    """
    responses = [
        _usage_response("A", Usage(requests=1, input_tokens=3, output_tokens=4, total_tokens=7)),
        _usage_response(
            "RESP", Usage(requests=1, input_tokens=10, output_tokens=20, total_tokens=30)
        ),
    ]
    monkeypatch.setattr(agents.Runner, "run", _stub_run("RESP", responses))
    agent = _filler_agent(FakeModel().queue_text("RESP"))

    text, usage = await run_filler_prompt(agent, (), "fill me")

    assert text == "RESP"
    assert usage == AgentRunUsage(model_calls=2, input_tokens=13, output_tokens=24)


async def test_一部の応答だけ_usage_を持つ場合は未取得としない(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未取得判定は「全応答が `requests==0` かつ `total_tokens==0`」なので、
    片方だけ 0 の場合は合算値を返す（None にしない）。
    """
    responses = [
        _usage_response("A", Usage()),
        _usage_response(
            "RESP", Usage(requests=1, input_tokens=5, output_tokens=6, total_tokens=11)
        ),
    ]
    monkeypatch.setattr(agents.Runner, "run", _stub_run("RESP", responses))
    agent = _filler_agent(FakeModel().queue_text("RESP"))

    _text, usage = await run_filler_prompt(agent, (), "fill me")

    assert usage == AgentRunUsage(model_calls=2, input_tokens=5, output_tokens=6)


def test_AgentRunUsage_は_frozen_な値型() -> None:
    """`AgentRunUsage` は agents 非依存の frozen dataclass（上位層が SDK 型に触れない）。"""
    assert dataclasses.is_dataclass(AgentRunUsage)
    assert [field.name for field in dataclasses.fields(AgentRunUsage)] == [
        "model_calls",
        "input_tokens",
        "output_tokens",
    ]
    usage = AgentRunUsage(model_calls=1, input_tokens=None, output_tokens=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        usage.model_calls = 2  # type: ignore[misc]


# ----------------------------------------------------------------------
# 戻り値契約と fail-fast
# ----------------------------------------------------------------------


async def test_応答が_None_なら空文字を返す() -> None:
    """`final_output` が None の場合、テキストは空文字（usage は返る）。"""
    agent = Agent(name=_FILLER_NAME, instructions="i", model=RecordingFakeModel(text=None))

    text, usage = await run_filler_prompt(agent, (), "fill me")

    assert text == ""
    assert isinstance(usage, AgentRunUsage)


async def test_発話も履歴も空なら_ValueError_で_fail_fast(
    runner_calls: list[tuple[tuple[Any, ...], dict[str, Any]]],
) -> None:
    """両方空の場合はモデル呼び出しに到達せず `ValueError` で落ちる。"""
    agent = _filler_agent(FakeModel().queue_text("RESP"))

    with pytest.raises(ValueError) as excinfo:
        await run_filler_prompt(agent, (), "")
    assert type(excinfo.value) is ValueError
    message = str(excinfo.value)
    assert "utterance" in message
    assert "history" in message
    assert runner_calls == []


def test_signature_は_agent_受け取り形で_context_のみ_keyword_only() -> None:
    """設計 §3.4 の signature を pin する（利用者から実体を受けない内部窓口）。"""
    params = inspect.signature(run_filler_prompt).parameters

    assert list(params) == ["agent", "history_items", "user_content", "context"]
    assert params["context"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["context"].default is None
