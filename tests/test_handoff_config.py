"""L1/L2: HandoffConfig（型付きフィールド）と dynamic_edge（on_invoke ルーティング）の検証。

実 agents.Agent を構築するが Runner は起動しない（ハンドオフの結線と on_invoke_handoff の
直接呼び出しのみ検証するため、ネットワークは不要）。
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
from agents import Agent, Handoff
from pydantic import BaseModel, Field, ValidationError

from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph
from oai_agentspec.spec import DynamicHandoff, HandoffConfig


def reg_with(*names: str) -> AgentRegistry:
    reg = AgentRegistry()
    for name in names:
        reg.register(AgentSpec(name=name, instructions=name))
    return reg


def _find_handoff(agent: Agent, tool_name: str) -> Handoff:
    return next(h for h in agent.handoffs if getattr(h, "tool_name", "") == tool_name)


def test_edge_typed_fields_map_to_handoff() -> None:
    """edge の description / tool_name が SDK Handoff の対応値へマップされる。"""
    reg = reg_with("a", "b")
    graph = HandoffGraph(entry="a")
    graph.edge("a", "b", description="請求担当へ", tool_name="to_billing")
    graph.apply(reg)

    handoff = _find_handoff(reg.get("a"), "to_billing")
    assert isinstance(handoff, Handoff)
    assert handoff.tool_description == "請求担当へ"


def test_edge_without_options_appends_raw_agent() -> None:
    """設定なしエッジは生 Agent を handoffs に追加する（Handoff ラップしない）。"""
    reg = reg_with("a", "b")
    graph = HandoffGraph(entry="a")
    graph.edge("a", "b")
    graph.apply(reg)
    assert reg.get("a").handoffs[0] is reg.get("b")


def test_edge_options_reserved_key_guard() -> None:
    """options に型付きフィールドと重複する予約キーを入れると構築時に弾かれる。"""
    reg = reg_with("a", "b")
    graph = HandoffGraph(entry="a")
    graph.edge("a", "b", options={"on_handoff": lambda ctx: None})
    graph.apply(reg)
    with pytest.raises(ValueError, match="予約キー"):
        reg.get("a")


def test_apply_replace_overwrites_topology() -> None:
    """apply は replace で当該 src のトポロジを上書きする（再 apply で再構成）。"""
    reg = reg_with("a", "b", "c")
    HandoffGraph(entry="a").edge("a", "b").apply(reg)
    assert reg.get("a").handoffs[0] is reg.get("b")
    # 別グラフを再 apply すると b が c に置き換わる。
    HandoffGraph(entry="a").edge("a", "c").apply(reg)
    assert reg.get("a").handoffs[0] is reg.get("c")


def test_apply_clears_source_when_all_edges_removed() -> None:
    """前回 apply した src のエッジを全削除して再 apply すると、ハンドオフが空になる。"""
    reg = reg_with("triage", "billing")
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "billing")
    graph.apply(reg)
    assert reg.get("triage").handoffs  # billing あり

    graph.edges.clear()  # triage のエッジを全削除
    graph.apply(reg)
    assert reg.get("triage").handoffs == []  # 古い billing が残らない


def test_apply_skips_cleared_source_that_was_unregistered() -> None:
    """前回反映した src が unregister 済みでも、空クリア対象スキップで例外を出さない。"""
    reg = reg_with("triage", "billing")
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "billing")
    graph.apply(reg)

    reg.unregister("triage")
    graph.edges.clear()
    graph.apply(reg)  # triage は未登録 → スキップされ KeyError にならない
    assert "triage" not in reg.names()


@pytest.mark.asyncio
async def test_dynamic_edge_routes_to_candidate() -> None:
    """dynamic_edge の on_invoke_handoff が resolver の返す候補へ転送する。"""
    reg = reg_with("triage", "billing", "support")
    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage", ["billing", "support"], lambda ctx, inp: "support", tool_name="route"
    )
    graph.apply(reg)

    dyn = _find_handoff(reg.get("triage"), "route")
    chosen = await dyn.on_invoke_handoff(None, None)
    assert chosen is reg.get("support")


@pytest.mark.asyncio
async def test_dynamic_edge_rejects_out_of_candidate() -> None:
    """resolver が候補外の名前を返すと実行時 ValueError。"""
    reg = reg_with("triage", "billing")
    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge("triage", ["billing"], lambda ctx, inp: "ghost", tool_name="route")
    graph.apply(reg)

    dyn = _find_handoff(reg.get("triage"), "route")
    with pytest.raises(ValueError, match="候補外"):
        await dyn.on_invoke_handoff(None, None)


def test_dynamic_edge_validate_detects_missing_candidate() -> None:
    """validate が dynamic handoff の未登録候補を検出する。"""
    reg = reg_with("triage")
    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge("triage", ["ghost"], lambda ctx, inp: "ghost", tool_name="route")
    graph.apply(reg)
    with pytest.raises(KeyError, match="候補"):
        reg.validate()


def test_mermaid_includes_dynamic_edges() -> None:
    """mermaid に動的エッジが破線で含まれる。"""
    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge("triage", ["billing", "support"], lambda c, i: "billing", tool_name="route")
    out = graph.mermaid()
    assert "triage -.->|route| billing" in out
    assert "triage -.->|route| support" in out


# ---------------------------------------------------------------------------
# Issue #38: 動的エッジの on_handoff / input_type / input_filter / is_enabled / options
# ---------------------------------------------------------------------------


class _RouteInput(BaseModel):
    """動的ハンドオフの input_type に渡す pydantic モデル（テスト用）。"""

    reason: str = Field(description="転送理由")
    priority: int = 1


def _make_registry(*names: str) -> AgentRegistry:
    """名前順に AgentSpec を登録した AgentRegistry を返すヘルパ。"""
    reg = AgentRegistry()
    for name in names:
        reg.register(AgentSpec(name=name, instructions=name))
    return reg


@pytest.mark.asyncio
async def test_dynamic_edge_on_handoff_fires_without_input_type() -> None:
    """input_type 無しの場合、on_handoff(ctx) が転送先決定後に発火する（arity=1）。"""
    reg = _make_registry("triage", "billing", "support")
    called: list[Any] = []

    def on_handoff(ctx: Any) -> None:
        called.append(ctx)

    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing", "support"],
        lambda ctx, inp: "support",
        tool_name="route",
        on_handoff=on_handoff,
    )
    graph.apply(reg)

    dyn = _find_handoff(reg.get("triage"), "route")
    sentinel = object()
    chosen = await dyn.on_invoke_handoff(sentinel, None)
    assert chosen is reg.get("support")
    assert called == [sentinel]


@pytest.mark.asyncio
async def test_dynamic_edge_on_handoff_receives_parsed_input() -> None:
    """input_type 指定時、on_handoff(ctx, parsed) に parsed pydantic オブジェクトが届く。"""
    reg = _make_registry("triage", "billing", "support")
    captured: list[Any] = []

    def on_handoff(ctx: Any, parsed: _RouteInput) -> None:
        captured.append((ctx, parsed))

    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing", "support"],
        lambda ctx, inp: "billing",
        tool_name="route",
        on_handoff=on_handoff,
        input_type=_RouteInput,
    )
    graph.apply(reg)

    dyn = _find_handoff(reg.get("triage"), "route")
    sentinel = object()
    payload = json.dumps({"reason": "請求のため", "priority": 5})
    chosen = await dyn.on_invoke_handoff(sentinel, payload)
    assert chosen is reg.get("billing")
    assert len(captured) == 1
    ctx_arg, parsed_arg = captured[0]
    assert ctx_arg is sentinel
    assert isinstance(parsed_arg, _RouteInput)
    assert parsed_arg.reason == "請求のため"
    assert parsed_arg.priority == 5


@pytest.mark.asyncio
async def test_dynamic_edge_on_handoff_var_positional_dispatches_by_input_type() -> None:
    """*args の柔軟関数は `input_type` 有無に応じて `(ctx,)` / `(ctx, parsed)` で呼ばれる。

    現行 dispatch は `_call_on_handoff` が `has_input_type` を見て直接 1/2 引数を選ぶ方式
    （旧 arity 検出ロジックは Codex 指摘の P2 修正で撤去済み）。`*args` は両形を吸収するため
    観測結果は input_type 指定有無で素直に切り替わる。
    """
    reg = _make_registry("triage", "billing")
    calls: list[tuple[Any, ...]] = []

    def on_handoff(*args: Any) -> None:
        calls.append(args)

    # input_type 無し → arity=1（context のみ）
    graph_no_type = HandoffGraph(entry="triage")
    graph_no_type.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route_no_type",
        on_handoff=on_handoff,
    )
    graph_no_type.apply(reg)
    dyn_no_type = _find_handoff(reg.get("triage"), "route_no_type")
    await dyn_no_type.on_invoke_handoff("ctx-1", None)
    assert calls[-1] == ("ctx-1",)

    # input_type 有り → arity=2（context, parsed）
    reg2 = _make_registry("triage", "billing")
    graph_with_type = HandoffGraph(entry="triage")
    graph_with_type.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route_with_type",
        on_handoff=on_handoff,
        input_type=_RouteInput,
    )
    graph_with_type.apply(reg2)
    dyn_with_type = _find_handoff(reg2.get("triage"), "route_with_type")
    payload = json.dumps({"reason": "x", "priority": 2})
    await dyn_with_type.on_invoke_handoff("ctx-2", payload)
    last = calls[-1]
    assert last[0] == "ctx-2"
    assert isinstance(last[1], _RouteInput)
    assert last[1].reason == "x"


@pytest.mark.asyncio
async def test_dynamic_edge_on_handoff_optional_second_arg_receives_parsed() -> None:
    """`def cb(ctx, parsed=None)` のような optional 2 引数版にも parsed が届く。

    旧実装は「必須引数数」だけで arity を判定していたため、optional 2 引数版が arity=1 と
    誤分類されて parsed が落ちていた。SDK 同等の dispatch（input_type 有無で 1/2 引数を直接
    決める）に揃えた回帰テスト。
    """
    reg = _make_registry("triage", "billing")
    received: list[Any] = []

    def on_handoff(ctx: Any, parsed: Any = None) -> None:
        received.append((ctx, parsed))

    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route",
        on_handoff=on_handoff,
        input_type=_RouteInput,
    )
    graph.apply(reg)
    dyn = _find_handoff(reg.get("triage"), "route")
    payload = json.dumps({"reason": "x", "priority": 1})
    await dyn.on_invoke_handoff("ctx-A", payload)
    assert len(received) == 1
    ctx, parsed = received[0]
    assert ctx == "ctx-A"
    assert isinstance(parsed, _RouteInput)
    assert parsed.reason == "x"


@pytest.mark.asyncio
async def test_dynamic_edge_on_handoff_typeerror_propagates_without_retry() -> None:
    """callback 本体が `TypeError` を投げてもリトライ呼び出しせず、副作用は 1 回で例外伝播する。

    旧実装は callback 呼び出し時の `TypeError` を arity 誤判定と見做してもう一方の arity で
    リトライしていたため、副作用が 2 回走り、かつ元例外が「missing argument」に上書きされて
    本物のバグを隠していた。リトライ撤去の回帰テスト。
    """
    reg = _make_registry("triage", "billing")
    call_count = 0

    def on_handoff(ctx: Any, parsed: Any) -> None:
        nonlocal call_count
        call_count += 1
        raise TypeError("user-defined TypeError")

    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route",
        on_handoff=on_handoff,
        input_type=_RouteInput,
    )
    graph.apply(reg)
    dyn = _find_handoff(reg.get("triage"), "route")
    payload = json.dumps({"reason": "x", "priority": 1})
    with pytest.raises(TypeError, match="user-defined TypeError"):
        await dyn.on_invoke_handoff("ctx-B", payload)
    assert call_count == 1


@pytest.mark.asyncio
async def test_dynamic_edge_input_type_validates_even_without_on_handoff() -> None:
    """input_type 指定 + on_handoff 未指定でも LLM の構造化引数を pydantic で検証する。

    SDK 静的 handoff() は input_type 指定時に必ず on_handoff も要求するが、本ライブラリは
    resolver で構造化引数を解釈したい用途のため on_handoff なしを許容する。ただし silent な
    検証スキップは避け、_wrapped_on_invoke が input_type 指定時は parse / validate を実施し、
    不正入力は ValidationError として早期に表面化する。
    """
    reg = _make_registry("triage", "billing")
    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route",
        input_type=_RouteInput,
    )
    graph.apply(reg)
    dyn = _find_handoff(reg.get("triage"), "route")
    # 不正入力（必須フィールド reason 欠落）は ValidationError で弾かれる
    bad_payload = json.dumps({"priority": 1})
    with pytest.raises(ValidationError):
        await dyn.on_invoke_handoff("ctx", bad_payload)


@pytest.mark.asyncio
async def test_dynamic_edge_on_handoff_empty_input_skips_parse() -> None:
    """空文字列 / 空 bytes / None の input_json は parse をスキップし parsed=None で発火する。

    旧実装は `input_json != ""` だけで empty bytes (`b""`) を素通しさせ json.loads が
    JSONDecodeError を投げる経路があった。truthy 判定（`if input_json`）に統一して
    None / "" / b"" を一律に skip する回帰テスト。
    """
    reg = _make_registry("triage", "billing")
    received: list[Any] = []

    def on_handoff(ctx: Any, parsed: Any) -> None:
        received.append((ctx, parsed))

    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route",
        on_handoff=on_handoff,
        input_type=_RouteInput,
    )
    graph.apply(reg)
    dyn = _find_handoff(reg.get("triage"), "route")
    # 空 str / 空 bytes / None いずれも JSONDecodeError を起こさず parsed=None で発火
    for empty in ("", b"", None):
        received.clear()
        await dyn.on_invoke_handoff("ctx", empty)
        assert len(received) == 1
        assert received[0] == ("ctx", None)


@pytest.mark.asyncio
async def test_dynamic_edge_on_handoff_whitespace_surfaces_validation_error() -> None:
    """whitespace のみの input_json は validate_json が ValidationError を投げて伝播する。

    旧実装は json.loads が JSONDecodeError を投げて意味不明なエラーになっていた。
    `TypeAdapter.validate_json` 経由に統一し、pydantic の明示的な ValidationError が
    surface する。
    """
    reg = _make_registry("triage", "billing")
    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route",
        input_type=_RouteInput,
    )
    graph.apply(reg)
    dyn = _find_handoff(reg.get("triage"), "route")
    with pytest.raises(ValidationError):
        await dyn.on_invoke_handoff("ctx", "   ")


@pytest.mark.asyncio
async def test_dynamic_edge_on_handoff_bytearray_input_parses() -> None:
    """bytearray 入力も validate_json で正しくパースされる。

    旧実装は `isinstance(input_json, (str, bytes))` のみで bytearray は dict 経路に落ちて
    意味不明な ValidationError を起こしていた。bytes-like 全般を validate_json 経由に揃え、
    bytearray / bytes / str を同等に扱う。
    """
    reg = _make_registry("triage", "billing")
    received: list[Any] = []

    def on_handoff(ctx: Any, parsed: Any) -> None:
        received.append((ctx, parsed))

    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route",
        on_handoff=on_handoff,
        input_type=_RouteInput,
    )
    graph.apply(reg)
    dyn = _find_handoff(reg.get("triage"), "route")
    payload = bytearray(json.dumps({"reason": "ba", "priority": 2}).encode("utf-8"))
    await dyn.on_invoke_handoff("ctx", payload)
    assert len(received) == 1
    ctx, parsed = received[0]
    assert ctx == "ctx"
    assert isinstance(parsed, _RouteInput)
    assert parsed.reason == "ba"
    assert parsed.priority == 2


@pytest.mark.asyncio
async def test_dynamic_edge_on_handoff_empty_dict_input_validates() -> None:
    """空 dict `{}` を input_json として渡すと validate_python で必須フィールド不足の
    ValidationError として弾かれる（silent skip しない）。

    旧実装は `if input_json:` の truthy 判定で `{}` が falsy になり parse スキップして
    silently parsed=None を on_handoff に渡していた。defensive 経路（dict 直接渡し）でも
    入力検証を行うように変更したことの回帰テスト。
    """
    reg = _make_registry("triage", "billing")
    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route",
        input_type=_RouteInput,
    )
    graph.apply(reg)
    dyn = _find_handoff(reg.get("triage"), "route")
    with pytest.raises(ValidationError):
        await dyn.on_invoke_handoff("ctx", {})


def test_dynamic_edge_input_type_injects_non_empty_schema() -> None:
    """input_type 指定時、生 Handoff の input_json_schema が非空スキーマで差し替わる。"""
    reg = _make_registry("triage", "billing")
    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route",
        input_type=_RouteInput,
    )
    graph.apply(reg)

    dyn = _find_handoff(reg.get("triage"), "route")
    # _EMPTY_HANDOFF_SCHEMA は properties が空 dict。input_type 指定時は reason 等が現れる。
    assert dyn.input_json_schema != {
        "additionalProperties": False,
        "type": "object",
        "properties": {},
        "required": [],
    }
    assert "reason" in dyn.input_json_schema.get("properties", {})


def test_dynamic_edge_strict_json_schema_false_skips_strictification() -> None:
    """options={"strict_json_schema": False} 指定時、optional / default 値ありフィールドが
    `required` に勝手に含まれない（pydantic 生スキーマがそのまま渡る）。

    旧実装は schema を ensure_strict_json_schema で書き換え済みのまま `Handoff` に渡し、
    `strict_json_schema=False` の opt-out が事実上無視されていた（書き換え後は全 properties
    が required にされるため、利用者は default 値ありフィールドを LLM に省略させられない）。
    schema 生成段階でも strict フラグを反映する修正の回帰テスト。
    """

    class OptionalFields(BaseModel):
        priority: str
        reason: str = "未指定"  # default あり → 非 strict なら required から外れるはず

    reg = _make_registry("triage", "billing")
    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route",
        input_type=OptionalFields,
        options={"strict_json_schema": False},
    )
    graph.apply(reg)

    dyn = _find_handoff(reg.get("triage"), "route")
    required = set(dyn.input_json_schema.get("required", []))
    # 非 strict なら required は priority のみ（reason は default あり）
    assert "priority" in required
    assert "reason" not in required, (
        "strict_json_schema=False のとき default 値ありフィールドは required に含まれない"
    )
    # Handoff 側にも strict_json_schema=False が伝播している
    assert dyn.strict_json_schema is False


def test_dynamic_edge_strict_json_schema_true_default_keeps_required() -> None:
    """既定（strict_json_schema 未指定 = True 扱い）では default ありフィールドも required に
    含まれる（SDK strict mode の挙動）。"""

    class OptionalFields(BaseModel):
        priority: str
        reason: str = "未指定"

    reg = _make_registry("triage", "billing")
    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route",
        input_type=OptionalFields,
    )
    graph.apply(reg)

    dyn = _find_handoff(reg.get("triage"), "route")
    required = set(dyn.input_json_schema.get("required", []))
    # SDK strict_schema が利用可能なら default ありフィールドも required にされる
    assert "priority" in required
    assert "reason" in required


def test_dynamic_edge_input_filter_passes_through() -> None:
    """input_filter は生 Handoff.input_filter にそのまま渡る。"""
    reg = _make_registry("triage", "billing")

    def my_filter(data: Any) -> Any:
        return data

    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route",
        input_filter=my_filter,
    )
    graph.apply(reg)

    dyn = _find_handoff(reg.get("triage"), "route")
    assert dyn.input_filter is my_filter


def test_dynamic_edge_is_enabled_bool_false() -> None:
    """is_enabled=False が生 Handoff.is_enabled にそのまま反映される。"""
    reg = _make_registry("triage", "billing")
    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route",
        is_enabled=False,
    )
    graph.apply(reg)

    dyn = _find_handoff(reg.get("triage"), "route")
    assert dyn.is_enabled is False


def test_dynamic_edge_is_enabled_callable_passes_through() -> None:
    """is_enabled が callable のときは callable がそのまま渡る。"""
    reg = _make_registry("triage", "billing")

    def gate(ctx: Any, agent: Any) -> bool:
        return True

    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route",
        is_enabled=gate,
    )
    graph.apply(reg)

    dyn = _find_handoff(reg.get("triage"), "route")
    assert dyn.is_enabled is gate


def test_dynamic_edge_options_pass_through_extra_kwargs() -> None:
    """options 内の生 Handoff kwarg（型付きフィールド以外）は素通しされる。"""
    reg = _make_registry("triage", "billing")
    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route",
        options={"strict_json_schema": False},
    )
    graph.apply(reg)

    dyn = _find_handoff(reg.get("triage"), "route")
    assert dyn.strict_json_schema is False


def test_dynamic_edge_options_reserved_key_collision_raises() -> None:
    """options に型付きフィールドと重複する予約キーが入ると ValueError。

    エラーメッセージに「予約キー」を含むことも検証する。
    """
    reg = _make_registry("triage", "billing")

    def my_filter(data: Any) -> Any:
        return data

    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing"],
        lambda ctx, inp: "billing",
        tool_name="route",
        options={"input_filter": my_filter},
    )
    graph.apply(reg)
    with pytest.raises(ValueError, match="予約キー"):
        reg.get("triage")


def test_dynamic_edge_defaults_match_legacy_behavior() -> None:
    """新引数を一切渡さなければ生 Handoff の各フィールドは従来既定と一致する。"""
    reg = _make_registry("triage", "billing", "support")

    def resolver(ctx: Any, inp: Any) -> str:
        return "billing"

    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge("triage", ["billing", "support"], resolver, tool_name="route")
    graph.apply(reg)

    dyn = _find_handoff(reg.get("triage"), "route")
    # input_json_schema は空スキーマ（SDK の handoff(agent) と同一）
    assert dyn.input_json_schema == {
        "additionalProperties": False,
        "type": "object",
        "properties": {},
        "required": [],
    }
    assert dyn.input_filter is None
    assert dyn.is_enabled is True
    # on_handoff 未指定なら on_invoke_handoff は wrapped されず resolver クロージャそのまま。
    # 関数名は registry._build_dynamic_handoff 内の `on_invoke`。
    assert dyn.on_invoke_handoff.__name__ == "on_invoke"


def test_handoff_config_and_dynamic_handoff_share_field_names() -> None:
    """AC #7: 静的 HandoffConfig と動的 DynamicHandoff が共通フィールド名を持つ。"""
    static_fields = {f.name for f in dataclasses.fields(HandoffConfig)}
    dynamic_fields = {f.name for f in dataclasses.fields(DynamicHandoff)}
    common_required = {
        "description",
        "tool_name",
        "on_handoff",
        "input_type",
        "input_filter",
        "is_enabled",
        "options",
    }
    assert common_required.issubset(static_fields)
    assert common_required.issubset(dynamic_fields)


def test_dynamic_edge_apply_preserves_new_fields_on_spec() -> None:
    """HandoffGraph.apply 後、AgentSpec.dynamic_handoffs に新フィールドが保持される。"""
    reg = _make_registry("triage", "billing", "support")

    def on_handoff(ctx: Any, parsed: _RouteInput) -> None:
        return None

    def my_filter(data: Any) -> Any:
        return data

    def gate(ctx: Any, agent: Any) -> bool:
        return True

    graph = HandoffGraph(entry="triage")
    graph.dynamic_edge(
        "triage",
        ["billing", "support"],
        lambda ctx, inp: "billing",
        tool_name="route",
        description="動的ルーティング",
        on_handoff=on_handoff,
        input_type=_RouteInput,
        input_filter=my_filter,
        is_enabled=gate,
        options={"strict_json_schema": False},
    )
    graph.apply(reg)

    # apply は registry._update_handoffs 経由で spec.dynamic_handoffs に再パックする。
    spec = reg._specs["triage"]  # noqa: SLF001 - 内部状態のテスト的観測
    assert len(spec.dynamic_handoffs) == 1
    dh = spec.dynamic_handoffs[0]
    assert isinstance(dh, DynamicHandoff)
    assert dh.tool_name == "route"
    assert dh.candidates == ["billing", "support"]
    assert dh.description == "動的ルーティング"
    assert dh.on_handoff is on_handoff
    assert dh.input_type is _RouteInput
    assert dh.input_filter is my_filter
    assert dh.is_enabled is gate
    assert dh.options == {"strict_json_schema": False}
