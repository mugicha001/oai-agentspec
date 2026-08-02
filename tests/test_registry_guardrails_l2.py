"""L2: コア `AgentRegistry` への guardrail 名前参照の結線検証（実 Agent + FakeModel + Runner）。

`AgentRegistry(guardrail_registry=...)` の DI 面（束縛不変・clone / freeze）、`_wire` の境界別
振り分けと連結順序、build 経路の例外 4 種、`validate()` の 2 群集約と問題行の書式、run 単位の
`RunConfig(**run_config_kwargs())` 経由の適用を、実 `agents.Agent` を構築して検証する（判定器は
常に非検知の plain 関数・モデルは `FakeModel` で実 API を呼ばない）。

`Boundary` の値域と `_adapters.guardrail_boundary` の戻り値集合一致は
`tests/_adapters/test_guardrails_l2.py` が pin しているため重複させない。
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from agents import (
    Agent,
    OutputGuardrailTripwireTriggered,
    RunConfig,
    Runner,
    function_tool,
)
from agents.sandbox import SandboxAgent

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec._adapters.guardrails import (
    build_input_guardrail,
    build_output_guardrail,
    build_tool_input_guardrail,
    build_tool_output_guardrail,
)
from oai_agentspec.protocols import GuardrailProvider
from oai_agentspec.runtime.guardrails._detectors import Detection
from oai_agentspec.runtime.guardrails.registry import GuardrailRegistry
from oai_agentspec.runtime.guardrails.types import Boundary, GuardrailSpec
from oai_agentspec.spec import SandboxAgentSpec

from _helpers.fake_model import FakeModel
from _helpers.responses import tool_call_response

pytestmark = pytest.mark.integration


# ----------------------------------------------------------------------
# テスト用ヘルパ（非検知の実体生成 / 本物 provider / fake provider）
# ----------------------------------------------------------------------


def _detect(text: str) -> Detection:
    """常に非検知を返す plain 検知器（実体生成用・実 LLM / 外部 I/O を持たない）。"""
    return Detection(triggered=False)


_BUILDERS = {
    Boundary.INPUT: build_input_guardrail,
    Boundary.OUTPUT: build_output_guardrail,
    Boundary.TOOL_INPUT: build_tool_input_guardrail,
    Boundary.TOOL_OUTPUT: build_tool_output_guardrail,
}


def _entity(name: str, boundary: Boundary = Boundary.INPUT) -> Any:
    """指定境界の上流 SDK guardrail 実体を可視名 `name` で作る。"""
    return _BUILDERS[boundary](name, _detect)


def _provider(*entries: tuple[str, Boundary]) -> GuardrailRegistry:
    """本物の登録簿（`GuardrailRegistry`）に指定名・指定境界の宣言を登録して返す。"""
    registry = GuardrailRegistry()
    for name, boundary in entries:
        registry.register(
            GuardrailSpec(name=name, boundary=boundary, guardrail=_entity(name, boundary))
        )
    return registry


class _FixedBoundaryProvider:
    """`get` / `boundary_of` の 2 照会のみを実装した最小 provider（境界を固定で答える）。

    `boundary_of` の戻り値を任意の文字列にできるため、値域外 str やツール境界を返す
    自作 provider の扱いを検証できる。`get` は名前ごとに同一の不透明値を返す。
    """

    def __init__(self, boundary: str) -> None:
        self._boundary = boundary
        self._entities: dict[str, object] = {}

    def get(self, name: str) -> object:
        """名前ごとに安定した不透明値を返す（identity 比較に使う）。"""
        return self._entities.setdefault(name, object())

    def boundary_of(self, name: str) -> str:
        """コンストラクタで受け取った境界文字列をそのまま返す。"""
        return self._boundary


def _registry(provider: Any | None = None) -> AgentRegistry:
    """既定 builder（`_adapters.build_agent`）で実 Agent を組む registry を返す。"""
    if provider is None:
        return AgentRegistry()
    return AgentRegistry(guardrail_registry=provider)


# ======================================================================
# 群 A: __init__ / clone / freeze
# ======================================================================


def test_init_signature_keeps_positional_agent_builder() -> None:
    """`agent_builder` は位置引数のまま・`guardrail_registry` は keyword-only 既定 None。"""
    params = inspect.signature(AgentRegistry.__init__).parameters
    assert list(params) == ["self", "agent_builder", "guardrail_registry"]
    assert params["agent_builder"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["guardrail_registry"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["guardrail_registry"].default is None


def test_registry_without_provider_builds_as_before() -> None:
    """provider 未注入・`guardrails` 未宣言の従来構成は従来どおり build できる。"""
    reg = _registry()
    reg.register(AgentSpec(name="a", instructions="i"))

    agent = reg.get("a")

    assert type(agent) is Agent
    assert agent.input_guardrails == []
    assert agent.output_guardrails == []


def test_provider_satisfies_guardrail_provider_protocol() -> None:
    """本物の登録簿と最小 fake provider の双方が `GuardrailProvider` を構造的に満たす。"""
    assert isinstance(_provider(("in_gr", Boundary.INPUT)), GuardrailProvider)
    assert isinstance(_FixedBoundaryProvider("input"), GuardrailProvider)


def test_clone_inherits_provider_reference() -> None:
    """`clone()` は provider 参照を引き継ぎ、clone 後も名前参照を解決できる。"""
    provider = _provider(("in_gr", Boundary.INPUT))
    reg = _registry(provider)
    reg.register(AgentSpec(name="a", instructions="i", guardrails=["in_gr"]))

    cloned = reg.clone()
    agent = cloned.get("a")

    assert agent.input_guardrails == [provider.get("in_gr")]
    assert agent.input_guardrails[0] is provider.get("in_gr")


def test_clone_copies_guardrails_list_instance() -> None:
    """`clone()` した spec の `guardrails` list は元 spec と別インスタンス（`_copy_spec`）。"""
    provider = _provider(("in_gr", Boundary.INPUT))
    reg = _registry(provider)
    spec = AgentSpec(name="a", instructions="i", guardrails=["in_gr"])
    reg.register(spec)

    cloned = reg.clone()
    cloned_spec = cloned._specs["a"]  # noqa: SLF001 - コンテナ複製の検証
    assert cloned_spec.guardrails == ["in_gr"]
    assert cloned_spec.guardrails is not spec.guardrails

    # 元 spec のリストを汚しても clone 側へ伝播しない（伝播すれば未登録名で KeyError になる）。
    spec.guardrails.append("leaked")
    assert cloned.get("a").input_guardrails == [provider.get("in_gr")]


def test_freeze_then_provider_addition_does_not_change_result() -> None:
    """`freeze()` 後に provider へ追加登録しても既存 build 結果は変わらない。"""
    provider = _provider(("in_gr", Boundary.INPUT))
    reg = _registry(provider)
    reg.register(AgentSpec(name="a", instructions="i", guardrails=["in_gr"]))
    before = reg.get("a")

    reg.freeze()
    provider.register(
        GuardrailSpec(
            name="extra_gr", boundary=Boundary.INPUT, guardrail=_entity("extra_gr", Boundary.INPUT)
        )
    )
    after = reg.get("a")

    assert before.input_guardrails == [provider.get("in_gr")]
    assert after.input_guardrails == [provider.get("in_gr")]
    assert after.input_guardrails[0] is provider.get("in_gr")


# ======================================================================
# 群 B: _wire の境界別振り分けと連結順序
# ======================================================================


def test_wire_dispatches_input_boundary_only_to_input_field() -> None:
    """input 境界の名前参照は `input_guardrails` にのみ入る。"""
    provider = _provider(("in_gr", Boundary.INPUT))
    reg = _registry(provider)
    reg.register(AgentSpec(name="a", instructions="i", guardrails=["in_gr"]))

    agent = reg.get("a")

    assert agent.input_guardrails == [provider.get("in_gr")]
    assert agent.output_guardrails == []


def test_wire_dispatches_output_boundary_only_to_output_field() -> None:
    """output 境界の名前参照は `output_guardrails` にのみ入る。"""
    provider = _provider(("out_gr", Boundary.OUTPUT))
    reg = _registry(provider)
    reg.register(AgentSpec(name="a", instructions="i", guardrails=["out_gr"]))

    agent = reg.get("a")

    assert agent.output_guardrails == [provider.get("out_gr")]
    assert agent.input_guardrails == []


def test_wire_appends_named_references_after_dedicated_field() -> None:
    """連結順序は「専用フィールド -> 名前参照」（宣言の可視順を保つ）。"""
    provider = _provider(("in_gr", Boundary.INPUT), ("out_gr", Boundary.OUTPUT))
    reg = _registry(provider)
    direct_in = _entity("direct_in", Boundary.INPUT)
    direct_out = _entity("direct_out", Boundary.OUTPUT)
    reg.register(
        AgentSpec(
            name="a",
            instructions="i",
            input_guardrails=[direct_in],
            output_guardrails=[direct_out],
            guardrails=["in_gr", "out_gr"],
        )
    )

    agent = reg.get("a")

    assert agent.input_guardrails == [direct_in, provider.get("in_gr")]
    assert agent.output_guardrails == [direct_out, provider.get("out_gr")]


def test_wire_does_not_deduplicate_named_references() -> None:
    """同じ名前を 2 回宣言すると同じ実体が 2 回 append される（重複排除しない）。"""
    provider = _provider(("gr", Boundary.INPUT))
    reg = _registry(provider)
    reg.register(AgentSpec(name="a", instructions="i", guardrails=["gr", "gr"]))

    agent = reg.get("a")

    assert agent.input_guardrails == [provider.get("gr"), provider.get("gr")]
    assert agent.input_guardrails[0] is agent.input_guardrails[1]


def test_wire_resolves_entity_identical_to_provider_get() -> None:
    """名前参照で解決した実体は `provider.get(name)` と `is` 一致する（copy しない）。"""
    provider = _provider(("gr", Boundary.INPUT))
    reg = _registry(provider)
    reg.register(AgentSpec(name="a", instructions="i", guardrails=["gr"]))

    agent = reg.get("a")

    assert agent.input_guardrails[0] is provider.get("gr")


def test_build_rejects_non_str_element_with_type_name() -> None:
    """非 str 要素（実体の直入れ）は型名を含む `ValueError`。"""
    provider = _provider(("gr", Boundary.INPUT))
    reg = _registry(provider)
    reg.register(AgentSpec(name="a", instructions="i", guardrails=[_entity("gr")]))

    with pytest.raises(ValueError) as exc:
        reg.get("a")
    assert "InputGuardrail" in str(exc.value)


def test_build_without_provider_raises_value_error_with_injection_hint() -> None:
    """provider 未注入で名前参照を宣言すると注入方法を含む `ValueError`。"""
    reg = _registry()
    reg.register(AgentSpec(name="a", instructions="i", guardrails=["gr"]))

    with pytest.raises(ValueError) as exc:
        reg.get("a")
    assert "guardrail_registry" in str(exc.value)


def test_build_propagates_key_error_from_provider_get() -> None:
    """未登録名は provider の `KeyError` がそのまま伝播する（`ValueError` に包まない）。"""
    provider = _provider(("gr", Boundary.INPUT))
    reg = _registry(provider)
    reg.register(AgentSpec(name="a", instructions="i", guardrails=["missing"]))

    with pytest.raises(KeyError) as exc:
        reg.get("a")
    assert "missing" in str(exc.value)


def test_build_rejects_tool_boundary_reference_with_boundary_value() -> None:
    """ツール境界の名前参照は境界値を含む `ValueError`（Agent へ振り分けできない）。"""
    provider = _provider(("tool_gr", Boundary.TOOL_OUTPUT))
    reg = _registry(provider)
    reg.register(AgentSpec(name="a", instructions="i", guardrails=["tool_gr"]))

    with pytest.raises(ValueError) as exc:
        reg.get("a")
    assert "tool_output" in str(exc.value)


def test_dependencies_excludes_guardrail_names() -> None:
    """`_dependencies` はエージェント名のみを返す（guardrail 名を混ぜない）。"""
    spec = AgentSpec(
        name="a", instructions="i", handoffs=["b"], sub_agents=["c"], guardrails=["gr"]
    )

    assert AgentRegistry._dependencies(spec) == ["b", "c"]  # noqa: SLF001 - 依存辺契約の検証


def test_invalidate_reverse_lookup_is_unaffected_by_guardrail_names() -> None:
    """guardrail 名を宣言した spec があっても `invalidate` の逆引きが壊れない。"""
    provider = _provider(("gr", Boundary.INPUT))
    reg = _registry(provider)
    reg.register(AgentSpec(name="a", instructions="i", handoffs=["b"], guardrails=["gr"]))
    reg.register(AgentSpec(name="b", instructions="i", guardrails=["gr"]))
    first = reg.get("a")

    reg.update(AgentSpec(name="b", instructions="updated", guardrails=["gr"]))
    rebuilt = reg.get("a")

    assert rebuilt is not first
    assert rebuilt.input_guardrails == [provider.get("gr")]
    assert reg.get("b").input_guardrails == [provider.get("gr")]


def test_sandbox_spec_named_reference_is_wired() -> None:
    """`SandboxAgentSpec` でも名前参照が同一に振り分けられる。"""
    provider = _provider(("in_gr", Boundary.INPUT), ("out_gr", Boundary.OUTPUT))
    reg = _registry(provider)
    reg.register(SandboxAgentSpec(name="s", instructions="i", guardrails=["in_gr", "out_gr"]))

    agent = reg.get("s")

    assert isinstance(agent, SandboxAgent)
    assert agent.input_guardrails == [provider.get("in_gr")]
    assert agent.output_guardrails == [provider.get("out_gr")]


# ======================================================================
# 群 C: provider 境界値の 3 分岐（自作 provider の注入）
# ======================================================================


@pytest.mark.parametrize(
    ("boundary", "target", "other"),
    [
        ("input", "input_guardrails", "output_guardrails"),
        ("output", "output_guardrails", "input_guardrails"),
    ],
)
def test_fake_provider_agent_boundary_dispatches(boundary: str, target: str, other: str) -> None:
    """agent 境界を返す自作 provider は build 成功で正しい側へ振り分けられる。"""
    provider = _FixedBoundaryProvider(boundary)
    reg = _registry(provider)
    reg.register(AgentSpec(name="a", instructions="i", guardrails=["gr"]))

    agent = reg.get("a")

    assert getattr(agent, target) == [provider.get("gr")]
    assert getattr(agent, other) == []


@pytest.mark.parametrize("boundary", ["tool_input", "tool_output"])
def test_fake_provider_tool_boundary_raises_value_error(boundary: str) -> None:
    """ツール境界を返す自作 provider は境界値を含む `ValueError`。"""
    reg = _registry(_FixedBoundaryProvider(boundary))
    reg.register(AgentSpec(name="a", instructions="i", guardrails=["gr"]))

    with pytest.raises(ValueError) as exc:
        reg.get("a")
    assert boundary in str(exc.value)


@pytest.mark.parametrize("boundary", ["bogus", "INPUT", ""])
def test_fake_provider_out_of_range_boundary_raises_value_error(boundary: str) -> None:
    """値域外 str を返す自作 provider は `ValueError`（黙って input 扱いにしない）。"""
    reg = _registry(_FixedBoundaryProvider(boundary))
    reg.register(AgentSpec(name="a", instructions="i", guardrails=["gr"]))

    with pytest.raises(ValueError):
        reg.get("a")


def test_two_query_provider_is_sufficient_for_build() -> None:
    """`get` / `boundary_of` の 2 照会だけを持つ provider で build が成功する。"""
    provider = _FixedBoundaryProvider("input")
    assert set(dir(provider)) >= {"get", "boundary_of"}
    reg = _registry(provider)
    reg.register(AgentSpec(name="a", instructions="i", guardrails=["gr"]))

    agent = reg.get("a")

    assert agent.input_guardrails == [provider.get("gr")]


# ======================================================================
# 群 D: validate() の集約
# ======================================================================

_UNKNOWN_LINE = "'triage' の guardrail 参照 'x' が未登録"
_UNINJECTED_LINE = (
    "'triage' の guardrail 参照 'x' を解決できません"
    "（AgentRegistry(guardrail_registry=...) で注入してください）"
)


def _non_str_line(position: int) -> str:
    """非 str 要素の問題行（宣言リスト内の位置で要素を区別する）。"""
    return (
        f"'triage' の guardrail 参照の {position} 番目に str 以外の要素が含まれます"
        "（型: InputGuardrail）"
    )


_TOOL_BOUNDARY_LINE = (
    "'triage' の guardrail 参照 'tool_pii' はツール境界（tool_output）のため "
    "Agent へ振り分けできません"
)


def test_validate_aggregates_agent_and_guardrail_groups_in_single_key_error() -> None:
    """エージェント参照と guardrail 参照の問題は単一 `KeyError` の 2 群へ集約される。

    未注入（provider が None のときのみ発生）は同時に成立しないため、guardrail 側は未登録 /
    非 str / ツール境界の 3 種を同時に置く。接頭辞 2 種と区切り `" | "` を pin する。
    """
    provider = _provider(("tool_pii", Boundary.TOOL_OUTPUT))
    reg = _registry(provider)
    reg.register(
        AgentSpec(
            name="triage",
            instructions="i",
            handoffs=["billing"],
            guardrails=["x", _entity("direct"), "tool_pii"],
        )
    )

    with pytest.raises(KeyError) as exc:
        reg.validate()

    message = exc.value.args[0]
    expected_guardrails = "; ".join([_UNKNOWN_LINE, _non_str_line(2), _TOOL_BOUNDARY_LINE])
    assert message == (
        "未解決のエージェント参照: 'triage' の handoff 参照 'billing' が未登録"
        " | guardrail 参照の問題: " + expected_guardrails
    )
    assert message.count(" | ") == 1


def test_validate_pins_unknown_guardrail_problem_line() -> None:
    """未登録名の問題行の書式を pin する。"""
    reg = _registry(_provider(("other", Boundary.INPUT)))
    reg.register(AgentSpec(name="triage", instructions="i", guardrails=["x"]))

    with pytest.raises(KeyError) as exc:
        reg.validate()
    assert exc.value.args[0] == "guardrail 参照の問題: " + _UNKNOWN_LINE


def test_validate_pins_uninjected_provider_problem_line() -> None:
    """provider 未注入の問題行の書式を pin する（注入方法を含む）。"""
    reg = _registry()
    reg.register(AgentSpec(name="triage", instructions="i", guardrails=["x"]))

    with pytest.raises(KeyError) as exc:
        reg.validate()
    assert exc.value.args[0] == "guardrail 参照の問題: " + _UNINJECTED_LINE


def test_validate_pins_non_str_problem_line() -> None:
    """非 str 要素の問題行の書式を pin する（型名を含む）。"""
    reg = _registry(_provider(("gr", Boundary.INPUT)))
    reg.register(AgentSpec(name="triage", instructions="i", guardrails=[_entity("gr")]))

    with pytest.raises(KeyError) as exc:
        reg.validate()
    assert exc.value.args[0] == "guardrail 参照の問題: " + _non_str_line(1)


def test_validate_pins_tool_boundary_problem_line() -> None:
    """ツール境界の問題行の書式を pin する（境界値を含む）。"""
    reg = _registry(_provider(("tool_pii", Boundary.TOOL_OUTPUT)))
    reg.register(AgentSpec(name="triage", instructions="i", guardrails=["tool_pii"]))

    with pytest.raises(KeyError) as exc:
        reg.validate()
    assert exc.value.args[0] == "guardrail 参照の問題: " + _TOOL_BOUNDARY_LINE


def test_validate_message_is_unchanged_for_legacy_configuration() -> None:
    """`guardrails` 未宣言の既存構成では例外型・接頭辞・区切りが完全に不変。"""
    reg = _registry()
    reg.register(AgentSpec(name="triage", instructions="i", handoffs=["billing"]))

    with pytest.raises(KeyError) as exc:
        reg.validate()

    message = exc.value.args[0]
    assert message == "未解決のエージェント参照: 'triage' の handoff 参照 'billing' が未登録"
    assert " | " not in message


def test_validate_succeeds_for_out_of_range_boundary_but_build_fails() -> None:
    """値域外 str を返す provider は `validate()` の集約対象外（build で `ValueError`）。"""
    reg = _registry(_FixedBoundaryProvider("bogus"))
    reg.register(AgentSpec(name="triage", instructions="i", guardrails=["gr"]))

    reg.validate()

    with pytest.raises(ValueError):
        reg.get("triage")


def test_validate_success_then_build_uses_same_entity() -> None:
    """`validate()` 成功後の `provider.get(name)` は build 結果の要素と `is` 一致する。"""
    provider = _provider(("gr", Boundary.INPUT))
    reg = _registry(provider)
    reg.register(AgentSpec(name="triage", instructions="i", guardrails=["gr"]))

    reg.validate()
    agent = reg.get("triage")

    assert agent.input_guardrails[0] is provider.get("gr")


# ======================================================================
# 群 E: run 単位（RunConfig + FakeModel + Runner）
# ======================================================================


def _never(text: str) -> bool:
    """常に False を返す述語（trip しない guardrail 用）。"""
    return False


async def test_run_config_kwargs_drives_runner_without_tripping() -> None:
    """`run_config_kwargs()` の戻り値で組んだ `RunConfig` で `Runner.run` が完走する。"""
    guardrails = GuardrailRegistry()
    guardrails.predicate_guardrail(_never, on="input", name="never_in")
    guardrails.predicate_guardrail(_never, on="output", name="never_out")
    reg = _registry(guardrails)
    reg.register(AgentSpec(name="a", instructions="i", model=FakeModel().queue_text("ok")))
    agent = reg.get("a")
    config = RunConfig(**guardrails.run_config_kwargs(["never_in", "never_out"]))

    result = await Runner.run(agent, "hi", run_config=config)

    assert result.final_output == "ok"
    assert config.input_guardrails == [guardrails.get("never_in")]
    assert config.output_guardrails == [guardrails.get("never_out")]


async def test_run_configの入力guardrailは初回ターンの入力にしか掛からない() -> None:
    """`RunConfig` に載せた入力 guardrail は 2 ターン目の入力では評価されない（上流結合点の pin）。

    上流 run loop は入力 guardrail の集合を「最初のターンかつ resume でないとき」に限って
    組み立てる。`docs/usage/safety/guardrails.md` の案内（毎ターン検査したいものは Agent 単位で
    宣言する）はこの前提に依存するため、上流が毎ターン評価する形へ変わったら気付く必要がある。

    ターン数が実際に 2 であることを併せて assert する。上流がツール呼び出しで 2 ターン目へ
    進まなくなると呼び出し回数 1 は自明に成立し、テストが緑のまま何も検証しなくなる（本テストは
    上流の変異を注入して RED を確認できないため、空振りの排除を assert で担保する）。

    照合は回数ではなく受け取った値の列（`["hi"]` = 初回入力のみ）で行う。上流が 2 ターン目でも
    評価する形へ変わると、ツール結果を含む 2 件目が現れて不一致になる。

    本テストが pin するのはライブラリ実装が依拠する結合点ではなく、docs の運用案内が依拠する
    上流の評価タイミングである（`tests/_adapters/test_guardrails_l2.py` の NFR-5 の 4 前提とは
    別カテゴリで、あの 4 件には数えない）。
    """
    seen: list[str] = []

    def _record(text: str) -> bool:
        seen.append(text)
        return False

    @function_tool
    def echo(value: str) -> str:
        """受け取った値をそのまま返す。"""
        return value

    guardrails = GuardrailRegistry()
    guardrails.predicate_guardrail(_record, on="input", name="count_in")
    model = FakeModel()
    model.queue_tool_call("echo", '{"value": "x"}')
    model.queue_text("done")
    reg = _registry(guardrails)
    reg.register(AgentSpec(name="a", instructions="i", model=model, tools=[echo]))
    config = RunConfig(**guardrails.run_config_kwargs(["count_in"]))

    result = await Runner.run(reg.get("a"), "hi", run_config=config)

    assert result.final_output == "done"
    assert len(model.calls) == 2, "2 ターン走っていないと呼び出し回数 1 が自明に成立する"
    assert seen == ["hi"]


async def test_run_configの出力guardrailはハンドオフ先の最終出力も検査する() -> None:
    """`RunConfig` に載せた出力 guardrail はハンドオフ先が出した最終出力でも trip する。

    上流 run loop は最終出力の検査で「そのターンのエージェント」の出力 guardrail と
    `RunConfig` の出力 guardrail を連結する（起点エージェントではない）。run 全体へ一律に
    掛けたい guardrail を run 単位で渡すという案内はこの前提に依存する。

    ハンドオフ先が実際に応答したことを併せて assert する。委譲が起きず起点エージェントが
    最終出力を出した場合も trip はするため、それだけでは「ハンドオフ先でも効く」ことの
    検証にならない（空振りの排除）。

    上のテストと同じく、pin する対象は docs の運用案内が依拠する上流の評価範囲であり、
    NFR-5 の 4 前提には数えない。
    """

    def _trips_on_delegate_output(text: str) -> bool:
        return "handled by b" in text

    guardrails = GuardrailRegistry()
    guardrails.predicate_guardrail(_trips_on_delegate_output, on="output", name="run_wide_out")
    model_a = FakeModel()
    model_a.responses.append(tool_call_response("transfer_to_b"))
    model_b = FakeModel().queue_text("handled by b")
    reg = _registry(guardrails)
    reg.register(AgentSpec(name="a", instructions="i", model=model_a, handoffs=["b"]))
    reg.register(AgentSpec(name="b", instructions="i", model=model_b))
    config = RunConfig(**guardrails.run_config_kwargs(["run_wide_out"]))

    with pytest.raises(OutputGuardrailTripwireTriggered):
        await Runner.run(reg.get("a"), "hi", run_config=config)

    assert model_b.calls, "ハンドオフ先が応答していないと起点側の出力で trip した可能性が残る"


def test_run_config_accepts_empty_kwargs() -> None:
    """対象が空でも 2 キーが揃い `RunConfig(**kwargs)` が構築できる。"""
    guardrails = GuardrailRegistry()
    guardrails.predicate_guardrail(_never, on="input", name="never_in")

    config = RunConfig(**guardrails.run_config_kwargs([]))

    assert config.input_guardrails == []
    assert config.output_guardrails == []


def test_コア側の境界定数はBoundaryの値域と一致する() -> None:
    """コア `registry.py` の境界定数 2 つの和が `Boundary` の値域と `==` で一致する。

    コア層は単方向依存を守るため `Boundary` を import できず、境界値域を文字列の定数として
    二重に持つ。乖離すると境界追加時に「'input' / 'output' のいずれでもない」という誤った説明の
    `ValueError` が出る（ツール系境界を足したのに `tool_boundary` の kind に落ちない）。両者を
    import できる本層で集合一致を pin し、乖離を機械的に検知する。
    """
    from oai_agentspec.registry import _AGENT_BOUNDARIES, _TOOL_BOUNDARIES

    assert _AGENT_BOUNDARIES | _TOOL_BOUNDARIES == {member.value for member in Boundary}
    assert _AGENT_BOUNDARIES.isdisjoint(_TOOL_BOUNDARIES)


def test_guardrail_problemは未知のkindでValueErrorになる() -> None:
    """`_guardrail_problem` の kind ディスパッチが fall-through しない（内部不整合の early fail）。

    純関数なので、将来 kind をタイポしても例外にならず「境界 '' は 'input' / 'output' の
    いずれでもない」という**無関係な文言**が静かに生成される。誰も気付く機会がないため raise で
    閉じる。
    """
    reg = AgentRegistry()
    assert "が未登録" in reg._guardrail_problem("a", "gr", "unknown")
    with pytest.raises(ValueError, match="unknown guardrail problem kind"):
        reg._guardrail_problem("a", "gr", "typo_kind")


def test_build時にproviderの実体と申告境界の不一致をValueErrorで落とす() -> None:
    """自作 provider が境界を偽った場合、build 時に診断可能な `ValueError` で落とす。

    cross-check が無いと build は通り、run 開始時に SDK 内部で `AttributeError`
    （`'OutputGuardrail' object has no attribute 'run_in_parallel'` 等）になる。利用者は自作
    provider の境界申告ミスに辿り着けないため、宣言と実体の整合をコア層で突き合わせる。
    """

    class _LyingProvider:
        """出力境界の実体を「入力境界」と申告する provider。"""

        def __init__(self) -> None:
            self._entity = build_output_guardrail("liar", _detect)

        def get(self, name: str) -> Any:
            """名前に関係なく出力境界の実体を返す。"""
            return self._entity

        def boundary_of(self, name: str) -> str:
            """実体は出力境界だが入力境界だと申告する。"""
            return "input"

    registry = AgentRegistry(guardrail_registry=_LyingProvider())
    registry.register(AgentSpec(name="a", instructions="x", guardrails=["liar"]))
    with pytest.raises(ValueError, match="liar"):
        registry.get("a")


def test_guardrailsにbare_strを渡すと登録時にValueErrorになる() -> None:
    """`AgentSpec.guardrails` に素の `str` を渡すと `register()` が `ValueError`（P5）。

    `guardrails=["a"]` と `guardrails="a"` の取り違えは起きやすく、現状は 1 文字ずつ名前参照と
    して解釈され `validate()` が偽の「未登録」を大量に報告する。`run_config_kwargs` は同じ
    fail-open を明示的に弾いているため、宣言側にも同じ守りを置く。
    """
    registry = AgentRegistry()
    with pytest.raises(ValueError, match="bare str"):
        registry.register(AgentSpec(name="a", instructions="x", guardrails="injection_baseline"))
    assert registry.names() == []

    # `update()` 経路でも同じ検証が働く（登録後の差し替えで穴が開かない）。
    registry.register(AgentSpec(name="b", instructions="x"))
    with pytest.raises(ValueError, match="bare str"):
        registry.update(AgentSpec(name="b", instructions="x", guardrails="gr"))


def test_providerがunhashableな境界を返してもValueErrorで落とす() -> None:
    """`boundary_of` が unhashable な値を返した場合に生の `TypeError` を漏らさない（P3）。

    境界判定は frozenset のメンバ検査なので、list 等を返す自作 provider は `TypeError:
    unhashable type` になる。値域外を `ValueError` で扱う既存方針と揃える。
    """

    class _UnhashableProvider:
        """境界に list を返す provider（unhashable）。"""

        def get(self, name: str) -> Any:
            """名前に関係なく不透明値を返す。"""
            return object()

        def boundary_of(self, name: str) -> Any:
            """unhashable な値を返す。"""
            return ["input"]

    registry = AgentRegistry(guardrail_registry=_UnhashableProvider())
    registry.register(AgentSpec(name="a", instructions="x", guardrails=["g"]))
    with pytest.raises(ValueError, match="g"):
        registry.get("a")
    # 値域外と同じ扱いで `validate()` の集約対象には含めない
    # （`TypeError` を漏らさないことが要点）。
    registry.validate()


def test_non_str問題行は値を含めて要素ごとに区別できる() -> None:
    """同一型の非 str 要素が複数あっても問題行が区別できる（P4）。

    型名だけでは `guardrails=[123, 456]` の 2 件が完全に同一文言になり、どの要素が問題なのか
    利用者が特定できない。値の repr を添えて要素ごとに一意にする。
    """
    registry = AgentRegistry()
    lines = [
        registry._guardrail_problem("a", 123, "non_str", "int", position=1),
        registry._guardrail_problem("a", 456, "non_str", "int", position=2),
    ]
    assert lines[0] != lines[1]
    assert "1 番目" in lines[0]
    assert "2 番目" in lines[1]


def test_validateはspec2件以上のguardrail問題行を登録順で全件集約する() -> None:
    """問題を持つ spec が複数あるとき、全 spec 分の問題行が spec 登録順で列挙される（R6）。

    集約は外側の spec ループと内側の宣言ループの 2 段で、単一 spec の構成だけでは外側ループを
    制約できない。「最初に問題が出た spec で打ち切る」「登録順を反転する」変異が素通りするため、
    2 spec に問題を置いて行順（spec 登録順・spec 内は宣言順）ごと完全一致で pin する。
    """
    provider = _provider(("tool_pii", Boundary.TOOL_OUTPUT))
    reg = _registry(provider)
    reg.register(AgentSpec(name="triage", instructions="i", guardrails=["x", _entity("direct")]))
    reg.register(AgentSpec(name="billing", instructions="i", guardrails=["tool_pii", "y"]))

    with pytest.raises(KeyError) as exc:
        reg.validate()

    billing_tool_line = (
        "'billing' の guardrail 参照 'tool_pii' はツール境界（tool_output）のため "
        "Agent へ振り分けできません"
    )
    billing_unknown_line = "'billing' の guardrail 参照 'y' が未登録"
    assert exc.value.args[0] == "guardrail 参照の問題: " + "; ".join(
        [_UNKNOWN_LINE, _non_str_line(2), billing_tool_line, billing_unknown_line]
    )


def test_validateはgetだけがKeyErrorを上げるproviderでも未登録を検出する() -> None:
    """`boundary_of` が正常に答えても `get` が `KeyError` なら未登録として報告する（R7）。

    集約器は `get` の戻り値を捨てるため、既存の provider ヘルパ（`get` が例外を上げない）では
    `provider.get(name)` の 1 行を削除しても全テストが通る。片側だけ解決できる非対称 provider
    を注入して、実体不在の検出が失われないことを pin する。
    """

    class _EntityMissingProvider:
        """`get` だけが `KeyError` を上げ、`boundary_of` は agent 境界を正常に返す provider。"""

        def get(self, name: str) -> Any:
            """実体を保持していないため常に `KeyError`。"""
            raise KeyError(name)

        def boundary_of(self, name: str) -> str:
            """境界だけは解決できると答える（`get` と非対称）。"""
            return "input"

    reg = _registry(_EntityMissingProvider())
    reg.register(AgentSpec(name="triage", instructions="i", guardrails=["x"]))

    with pytest.raises(KeyError) as exc:
        reg.validate()
    assert exc.value.args[0] == "guardrail 参照の問題: " + _UNKNOWN_LINE


@pytest.mark.parametrize("position", [1, 2])
def test_build経路の非str問題行は宣言リスト内の位置を示す(position: int) -> None:
    """build 経路の非 str `ValueError` も位置表示（`N 番目`）を含む（R8）。

    2 要素宣言の 1 番目 / 2 番目それぞれで完全一致を取るため、位置の受け渡し欠落（常に位置なし）
    と `enumerate` の開始値ずれの双方を検知できる。
    """
    provider = _provider(("gr", Boundary.INPUT))
    reg = _registry(provider)
    declared: list[Any] = ["gr", "gr"]
    declared[position - 1] = _entity("gr")
    reg.register(AgentSpec(name="triage", instructions="i", guardrails=declared))

    with pytest.raises(ValueError) as exc:
        reg.get("triage")
    assert str(exc.value) == _non_str_line(position)


def test_非strとprovider未注入が同時成立しても非strを先に報告する() -> None:
    """provider 未注入 + 非 str 要素では「非 str」が報告される（検証順の pin・R9）。

    順序が入れ替わると、実体を直入れした利用者へ「`AgentRegistry(guardrail_registry=...)` で
    注入してください」という誤った是正案（注入しても直らない）が出る。build 経路と `validate()`
    経路の双方で同時成立させ、`uninjected` 行にならないことを押さえる。
    """
    reg = _registry()
    reg.register(AgentSpec(name="triage", instructions="i", guardrails=[_entity("gr")]))

    with pytest.raises(ValueError) as build_exc:
        reg.get("triage")
    assert str(build_exc.value) == _non_str_line(1)
    assert "guardrail_registry" not in str(build_exc.value)

    with pytest.raises(KeyError) as validate_exc:
        reg.validate()
    message = validate_exc.value.args[0]
    assert message == "guardrail 参照の問題: " + _non_str_line(1)
    assert "guardrail_registry" not in message


def test_extraへguardrailsを渡すと専用フィールド衝突として拒否される() -> None:
    """`extra={"guardrails": ...}` が専用フィールドとの衝突として `ValueError`（P2）。

    `_DEDICATED_AGENT_KWARGS` に載せないと「SDK が受け付けないキー」という別理由で拒否され、
    上流に `Agent.guardrails` が追加された時点で検証を通って二重に渡ってしまう。
    `input_guardrails` / `output_guardrails` と同じ扱いに揃える。
    """
    registry = AgentRegistry()
    registry.register(
        AgentSpec(name="a", instructions="x", extra={"guardrails": ["injection_baseline"]})
    )
    with pytest.raises(ValueError, match="専用フィールド"):
        registry.get("a")
