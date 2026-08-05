"""L2: `instructions_append` の build 時合成と run ごと評価を実 SDK 型で検証する。

`_adapters.build_agent` が生成する合成 instructions callable の契約を pin する:
SDK `Agent.get_system_prompt` が要求する厳密 2 名前付き引数・宣言順の `"\\n\\n"` 連結・
空断片のスキップ・async 断片の await・例外の伝播（fail-fast）・非 str 戻り値の `TypeError`・
callable `instructions` との併用拒否・容器型 / 要素型の検証（いずれも registry を迂回した
防御チェックで、register 経路と同一判定・同一文言）・追記が空のときの
現状経路の不変（`spec.instructions` を同一オブジェクトで素通し）・build 時に追記関数を
評価しないこと・`SandboxAgentSpec` では `base_instructions` を素通しすること。

run ごとの再評価は FakeModel + Runner でモデルへ渡る system prompt を実測して確認する
（`tests/test_dynamic_instructions.py` と同じ観測手法）。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import pytest
from agents import RunContextWrapper, Runner
from agents.sandbox import SandboxAgent

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec._adapters import build_agent
from oai_agentspec.spec import SandboxAgentSpec

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.integration


@dataclass
class Ctx:
    """追記関数が読む run context（`ctx.context.token` で開く）。"""

    token: str = "t"


async def _system_prompt(agent: Any, context: Any) -> str | None:
    """SDK と同じ経路（`get_system_prompt`）で合成結果を取得する。"""
    return await agent.get_system_prompt(RunContextWrapper(context))


# ---------------------------------------------------------------------------
# 追記が空: 現状経路が変わらない
# ---------------------------------------------------------------------------
def test_empty_append_passes_static_instructions_through_unchanged() -> None:
    """追記が空なら `Agent.instructions` は `spec.instructions` と同一オブジェクト。"""
    text = "".join(["static", " body"])  # 同一 str リテラルの interning に依存しない
    spec = AgentSpec(name="bot", instructions=text)
    agent = build_agent(spec)
    assert agent.instructions is text


def test_empty_append_keeps_callable_instructions_through_unchanged() -> None:
    """追記が空なら callable `instructions` もそのまま素通しされる（併用拒否は発火しない）。"""

    def instr(context: Any, agent: Any) -> str:
        return "dynamic"

    agent = build_agent(AgentSpec(name="bot", instructions=instr))
    assert agent.instructions is instr


# ---------------------------------------------------------------------------
# build 時未評価 + 合成 callable のシグネチャ
# ---------------------------------------------------------------------------
def test_append_functions_are_not_called_at_build() -> None:
    """非空の追記があっても build 時には追記関数を評価しない。"""

    def sentinel(context: Any, agent: Any) -> str:
        pytest.fail("instructions_append は build 時に評価されてはならない")

    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="static", instructions_append=[sentinel]))
    agent = reg.get("bot")
    assert callable(agent.instructions)


def test_composed_callable_has_exactly_two_named_parameters() -> None:
    """合成 callable は厳密に 2 つの名前付き引数（SDK のパラメータ数検査を満たす）。"""
    spec = AgentSpec(
        name="bot",
        instructions="static",
        instructions_append=[lambda ctx, agent: "frag"],
    )
    agent = build_agent(spec)
    params = list(inspect.signature(agent.instructions).parameters.values())
    assert len(params) == 2
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params)


# ---------------------------------------------------------------------------
# 連結順序・区切り・空断片
# ---------------------------------------------------------------------------
async def test_fragments_are_joined_in_declaration_order() -> None:
    """静的本文 + 各断片を宣言順に `"\\n\\n"` で連結する（並べ替えを検知する期待値）。"""
    spec = AgentSpec(
        name="bot",
        instructions="STATIC",
        instructions_append=[
            lambda ctx, agent: "mango",
            lambda ctx, agent: "apple",
            lambda ctx, agent: "zebra",
        ],
    )
    agent = build_agent(spec)
    assert await _system_prompt(agent, Ctx()) == "STATIC\n\nmango\n\napple\n\nzebra"


async def test_empty_string_fragments_are_skipped() -> None:
    """`""` を返す断片はスキップされ、余分な区切りが入らない。"""
    spec = AgentSpec(
        name="bot",
        instructions="STATIC",
        instructions_append=[
            lambda ctx, agent: "",
            lambda ctx, agent: "mango",
            lambda ctx, agent: "",
        ],
    )
    agent = build_agent(spec)
    assert await _system_prompt(agent, Ctx()) == "STATIC\n\nmango"


async def test_none_instructions_joins_fragments_only() -> None:
    """`instructions=None` なら追記のみを宣言順に連結する。"""
    spec = AgentSpec(
        name="bot",
        instructions=None,
        instructions_append=[
            lambda ctx, agent: "mango",
            lambda ctx, agent: "apple",
        ],
    )
    agent = build_agent(spec)
    assert await _system_prompt(agent, Ctx()) == "mango\n\napple"


async def test_empty_static_instructions_do_not_prepend_separator() -> None:
    """`instructions=""` は先頭断片として積まれず、区切りが先頭に入らない。

    `if static:` を `if static is not None:` へ変えると `"\\n\\nfrag"` になる（空本文の
    truthy 判定を falsy 判定へ退行させる変異を kill する）。
    """
    spec = AgentSpec(
        name="bot",
        instructions="",
        instructions_append=[lambda ctx, agent: "frag"],
    )
    agent = build_agent(spec)
    assert await _system_prompt(agent, Ctx()) == "frag"


async def test_append_list_is_snapshotted_at_build() -> None:
    """build 後に `spec.instructions_append` へ追加しても構築済み Agent の合成結果は変わらない。

    `list(spec.instructions_append)` のスナップショットを `spec.instructions_append` の共有参照へ
    変える変異では、build 済み Agent の system prompt に後付け断片が漏れ出す。
    """
    spec = AgentSpec(
        name="bot",
        instructions="STATIC",
        instructions_append=[lambda ctx, agent: "first"],
    )
    agent = build_agent(spec)
    spec.instructions_append.append(lambda ctx, agent: "later")

    assert await _system_prompt(agent, Ctx()) == "STATIC\n\nfirst"
    # 追加後に build し直せば新しい断片が反映される（スナップショットの取得時点の pin）。
    assert await _system_prompt(build_agent(spec), Ctx()) == "STATIC\n\nfirst\n\nlater"


async def test_none_instructions_with_all_empty_fragments_returns_empty_string() -> None:
    """`instructions=None` + 全断片 `""` の戻り値は `None` ではなく `""`。"""
    spec = AgentSpec(
        name="bot",
        instructions=None,
        instructions_append=[lambda ctx, agent: "", lambda ctx, agent: ""],
    )
    agent = build_agent(spec)
    result = await _system_prompt(agent, Ctx())
    assert result is not None
    assert result == ""


async def test_async_fragments_are_awaited_and_mixed_with_sync() -> None:
    """async 断片は await され、同期断片と混在しても宣言順に連結される。"""

    async def async_frag(context: Any, agent: Any) -> str:
        return "async-mango"

    spec = AgentSpec(
        name="bot",
        instructions="STATIC",
        instructions_append=[async_frag, lambda ctx, agent: "sync-apple"],
    )
    agent = build_agent(spec)
    assert await _system_prompt(agent, Ctx()) == "STATIC\n\nasync-mango\n\nsync-apple"


async def test_fragment_reads_run_context() -> None:
    """断片は `ctx.context.<attr>` で run context を読める（wrapper のまま渡す）。"""
    spec = AgentSpec(
        name="bot",
        instructions="STATIC",
        instructions_append=[lambda ctx, agent: f"token={ctx.context.token}"],
    )
    agent = build_agent(spec)
    assert await _system_prompt(agent, Ctx(token="abc")) == "STATIC\n\ntoken=abc"


# ---------------------------------------------------------------------------
# run ごとの再評価（Runner + FakeModel で system prompt を実測）
# ---------------------------------------------------------------------------
async def test_fragments_are_reevaluated_per_run() -> None:
    """run ごとに追記が再評価され、モデルへ渡る system prompt が run 単位で変わる。"""
    model = FakeModel().queue_text("one").queue_text("two")
    reg = AgentRegistry()
    reg.register(
        AgentSpec(
            name="bot",
            instructions="STATIC",
            instructions_append=[lambda ctx, agent: f"canary={ctx.context.token}"],
            model=model,
        )
    )
    agent = reg.get("bot")
    await Runner.run(agent, input="hi", context=Ctx(token="first"))
    await Runner.run(agent, input="hi", context=Ctx(token="second"))
    assert model.calls[0].system_instructions == "STATIC\n\ncanary=first"
    assert model.calls[1].system_instructions == "STATIC\n\ncanary=second"


async def test_fragment_exception_propagates_through_run() -> None:
    """断片が送出した例外は run へ伝播する（静的部分だけで縮退しない）。"""

    def boom(context: Any, agent: Any) -> str:
        raise RuntimeError("canary resolution failed")

    model = FakeModel().queue_text("one")
    reg = AgentRegistry()
    reg.register(
        AgentSpec(name="bot", instructions="STATIC", instructions_append=[boom], model=model)
    )
    agent = reg.get("bot")
    with pytest.raises(RuntimeError, match="canary resolution failed"):
        await Runner.run(agent, input="hi", context=Ctx())
    # 縮退して静的部分だけがモデルへ渡ることがない（そもそもモデル呼び出しに至らない）。
    assert model.calls == []


async def test_non_str_fragment_result_raises_type_error() -> None:
    """断片の戻り値が str 以外なら `TypeError`。"""
    spec = AgentSpec(
        name="bot",
        instructions="STATIC",
        instructions_append=[lambda ctx, agent: 42],
    )
    agent = build_agent(spec)
    with pytest.raises(TypeError):
        await _system_prompt(agent, Ctx())


# ---------------------------------------------------------------------------
# build_agent 側の防御チェック（registry 迂回）
# ---------------------------------------------------------------------------
def test_build_agent_rejects_callable_instructions_with_append() -> None:
    """registry を迂回して直接 build しても callable + 追記は `ValueError`。"""
    spec = AgentSpec(
        name="bot",
        instructions=lambda ctx, agent: "dynamic",
        instructions_append=[lambda ctx, agent: "frag"],
    )
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    message = str(excinfo.value)
    assert "'bot'" in message
    assert "instructions_append" in message


# ---------------------------------------------------------------------------
# build_agent 側の容器型・要素型検証（register と同一判定・同一文言）
# ---------------------------------------------------------------------------
def _spec_with_append(append: Any) -> AgentSpec:
    """`instructions_append` に任意の値を持つ spec を作る（型検証の入力生成用）。"""
    return AgentSpec(name="bot", instructions="STATIC", instructions_append=append)


def test_build_agent_rejects_bare_str_append() -> None:
    """素の `str` 容器は build 経路でも `ValueError`（1 文字ずつ callable 扱いされる取り違え）。

    `str` は反復可能なため build を通ると run で `TypeError: 'str' object is not callable` という
    原因の分かりにくい失敗になる。
    """
    with pytest.raises(ValueError) as excinfo:
        build_agent(_spec_with_append("abc"))
    message = str(excinfo.value)
    assert "instructions_append" in message
    assert "str" in message


def test_build_agent_rejects_non_callable_element() -> None:
    """非 callable 要素は build 経路でも `ValueError`（インデックス付きラベルで位置を示す）。"""
    with pytest.raises(ValueError) as excinfo:
        build_agent(_spec_with_append(["oops"]))
    message = str(excinfo.value)
    assert "instructions_append[0]" in message


def test_build_agent_rejects_generator_container() -> None:
    """使い切り iterable（generator）容器は build 経路でも `ValueError`（型名を報告する）。

    通すと 1 回目の build の `list(...)` で消費され、同一 spec の 2 回目の build で追記が silent に
    消える（build1 = `"STATIC\\n\\nfrag"` / build2 = `"STATIC"`）。
    """
    spec = _spec_with_append(f for f in [lambda ctx, agent: "frag"])
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    assert "generator" in str(excinfo.value)


@pytest.mark.parametrize("container", [list, tuple], ids=["list", "tuple"])
async def test_build_agent_accepts_list_and_tuple_containers(container: Any) -> None:
    """`list` / `tuple` 容器は従来どおり受理される（型検証追加による過剰拒否の検知）。"""
    spec = _spec_with_append(container([lambda ctx, agent: "frag"]))
    agent = build_agent(spec)
    assert await _system_prompt(agent, Ctx()) == "STATIC\n\nfrag"


@pytest.mark.parametrize(
    "append_factory",
    [
        lambda: "abc",
        lambda: ["oops"],
        lambda: (f for f in [lambda ctx, agent: "frag"]),
    ],
    ids=["bare-str", "non-callable-element", "generator"],
)
def test_register_and_build_share_identical_error_message(append_factory: Any) -> None:
    """register 経路と build 経路のエラーメッセージが完全に一致する（判定・文言の共有 pin）。

    片側だけを修正すると同じ不正宣言に対して別文言・別判定になり、利用者から見て「どちらの
    経路を通ったか」で診断が変わる。容器ごとに新しい spec を作る（generator の使い切りを
    経路間で共有しないため）。
    """
    with pytest.raises(ValueError) as build_exc:
        build_agent(_spec_with_append(append_factory()))
    with pytest.raises(ValueError) as register_exc:
        AgentRegistry().register(_spec_with_append(append_factory()))
    assert str(build_exc.value) == str(register_exc.value)


# ---------------------------------------------------------------------------
# SandboxAgentSpec: instructions 側にのみ効く
# ---------------------------------------------------------------------------
async def test_sandbox_spec_appends_to_instructions_only() -> None:
    """sandbox でも `instructions` 側の追記は効き、`base_instructions` は素通しされる。"""
    base = "".join(["BASE", " TEXT"])
    spec = SandboxAgentSpec(
        name="sbx",
        instructions="STATIC",
        base_instructions=base,
        instructions_append=[lambda ctx, agent: "mango"],
    )
    agent = build_agent(spec)
    assert isinstance(agent, SandboxAgent)
    assert agent.base_instructions is base
    assert await _system_prompt(agent, Ctx()) == "STATIC\n\nmango"


def test_sandbox_callable_base_instructions_is_unaffected_by_append() -> None:
    """`base_instructions` が callable でも `instructions_append` は合成対象にしない。"""

    def base(context: Any, agent: Any) -> str:
        return "BASE"

    spec = SandboxAgentSpec(
        name="sbx",
        instructions="STATIC",
        base_instructions=base,
        instructions_append=[lambda ctx, agent: "mango"],
    )
    agent = build_agent(spec)
    assert agent.base_instructions is base
