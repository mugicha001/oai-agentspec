"""L2: _adapters.builders の標準ルート build_agent extra 検証の特性化テスト。

Issue #19（純リファクタ）の安全網として、標準ルート `build_agent` の extra 検証
（専用フィールド同名キー衝突 / agents.Agent 未知キー）の ValueError メッセージ原文を
`_adapters` レベルで直接ピン留めする。Realtime 側は `test_realtime_l2.py` の extra reject
テストで担保済みだが、標準ルートには `_adapters` レベルでメッセージ原文を固定するテストが
不在（既存 `tests/runtime/guardrails/test_factories_l2.py` は `match="input_guardrails"` の
部分一致のみ）。

本モジュールは現状で GREEN になる特性化テスト（characterization test）であり、将来の
リファクタでメッセージ文字列が変わったら失敗するよう原文を完全一致でピン留めする。
"""

from __future__ import annotations

import warnings

import pytest
from agents import Agent, function_tool
from agents.sandbox import SandboxAgent
from agents.tool import WebSearchTool, tool_namespace

from oai_agentspec import _adapters
from oai_agentspec._adapters import build_agent
from oai_agentspec._adapters.builders import (
    _AGENT_FIELD_NAMES,
    _DEDICATED_AGENT_KWARGS,
    _SANDBOX_FIELD_KWARGS,
)
from oai_agentspec.spec import AgentSpec, SandboxAgentSpec

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# build_agent: extra に専用フィールド同名キー → ValueError（衝突メッセージ原文）
# ---------------------------------------------------------------------------
def test_build_rejects_dedicated_field_collision_message() -> None:
    """extra に専用フィールド同名キー（name）を積むと衝突メッセージ原文で弾く。"""
    spec = AgentSpec(name="bot", instructions="i", extra={"name": "dup"})
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    message = str(excinfo.value)
    # メッセージ原文を完全一致でピン留め（agent 名 + 「専用フィールドと同名」+ キー一覧）。
    assert message == ("agent 'bot': extra に専用フィールドと同名のキーが含まれます: ['name']")


def test_build_collision_message_lists_keys_sorted() -> None:
    """複数の専用フィールド同名キーはソート済みリストで列挙される。"""
    spec = AgentSpec(name="bot", instructions="i", extra={"name": "dup", "model": object()})
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    # sorted() でキーが昇順（model, name）に整列することを含めてピン留めする。
    assert str(excinfo.value) == (
        "agent 'bot': extra に専用フィールドと同名のキーが含まれます: ['model', 'name']"
    )


def test_build_collision_takes_precedence_over_unknown() -> None:
    """衝突キーと未知キーが同時にある場合は衝突メッセージが優先される（検査順の固定）。"""
    spec = AgentSpec(name="bot", instructions="i", extra={"name": "dup", "bogus": 1})
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    # 衝突検査が未知検査より先に実行され、未知キー（bogus）はメッセージに現れない
    # （完全一致 assert が bogus の非出現も含めて固定する）。
    assert str(excinfo.value) == (
        "agent 'bot': extra に専用フィールドと同名のキーが含まれます: ['name']"
    )


# ---------------------------------------------------------------------------
# build_agent: extra に未知キー → ValueError（未知メッセージ原文）
# ---------------------------------------------------------------------------
def test_build_rejects_unknown_key_message() -> None:
    """extra に agents.Agent が受け付けない未知キーを積むと未知メッセージ原文で弾く。"""
    spec = AgentSpec(name="bot", instructions="i", extra={"nonexistent_kw": 1})
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    message = str(excinfo.value)
    # メッセージ原文を完全一致でピン留め（agent 名 + 「agents.Agent が受け付けない」+ キー一覧）。
    assert message == (
        "agent 'bot': extra に agents.Agent が受け付けないキーが含まれます: ['nonexistent_kw']"
    )


def test_build_unknown_message_lists_keys_sorted() -> None:
    """複数の未知キーはソート済みリストで列挙される。"""
    spec = AgentSpec(name="bot", instructions="i", extra={"zzz": 1, "aaa": 2})
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    # sorted() でキーが昇順（aaa, zzz）に整列することを含めてピン留めする。
    assert str(excinfo.value) == (
        "agent 'bot': extra に agents.Agent が受け付けないキーが含まれます: ['aaa', 'zzz']"
    )


# ---------------------------------------------------------------------------
# build_agent: 正常系（有効な素通し extra が反映される）
# ---------------------------------------------------------------------------
def test_build_passes_valid_extra_through() -> None:
    """agents.Agent が受け付ける有効な extra キーは構築された Agent へ素通しされる。"""
    spec = AgentSpec(name="bot", instructions="i", extra={"handoff_description": "desk"})
    agent = build_agent(spec)
    assert isinstance(agent, Agent)
    assert agent.handoff_description == "desk"


# ---------------------------------------------------------------------------
# build_agent: SandboxAgentSpec 分岐（Issue #21 T2・RED 先行）
# ---------------------------------------------------------------------------
def test_sandbox_spec_builds_sandbox_agent() -> None:
    """SandboxAgentSpec を渡すと agents.sandbox.SandboxAgent が構築される。"""
    spec = SandboxAgentSpec(name="sbx", instructions="i")
    agent = build_agent(spec)
    assert isinstance(agent, SandboxAgent)


def test_sandbox_spec_reflects_four_fields() -> None:
    """sandbox 4 フィールドの指定値が構築後の SandboxAgent へそのまま反映される。"""
    manifest = object()  # SandboxAgent は plain dataclass で manifest の実行時型検証をしない
    caps = [object(), object()]
    spec = SandboxAgentSpec(
        name="sbx",
        instructions="i",
        default_manifest=manifest,
        capabilities=caps,
        run_as="worker",
        base_instructions="base",
    )
    agent = build_agent(spec)
    assert isinstance(agent, SandboxAgent)
    assert agent.default_manifest is manifest
    assert list(agent.capabilities) == caps
    assert agent.run_as == "worker"
    assert agent.base_instructions == "base"


def test_sandbox_spec_none_fields_defer_to_sdk_defaults() -> None:
    """4 フィールド未指定（None）は kwargs へ積まれず SDK 既定に委ねられる。

    None-omission 規約の検証: capabilities は SDK 素の `SandboxAgent` を直構築したときの
    既定と同じ構成になり（既定値の具体形はハードコードしない）、default_manifest /
    run_as / base_instructions は None のまま。
    """
    spec = SandboxAgentSpec(name="sbx", instructions="i")
    agent = build_agent(spec)
    assert isinstance(agent, SandboxAgent)
    reference = SandboxAgent(name="ref", instructions="i")
    assert [type(c) for c in agent.capabilities] == [type(c) for c in reference.capabilities]
    assert agent.default_manifest is None
    assert agent.run_as is None
    assert agent.base_instructions is None


def test_plain_spec_still_builds_plain_agent() -> None:
    """通常の AgentSpec の build 結果は SandboxAgent ではない素の Agent のまま。"""
    spec = AgentSpec(name="bot", instructions="i")
    agent = build_agent(spec)
    assert type(agent) is Agent
    assert not isinstance(agent, SandboxAgent)


def test_sandbox_extra_dedicated_field_collision_is_rejected() -> None:
    """extra に sandbox 専用フィールドと同名のキーを積むと agent 名入り衝突 ValueError。"""
    spec = SandboxAgentSpec(name="sbx", instructions="i", extra={"default_manifest": object()})
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    message = str(excinfo.value)
    assert "'sbx'" in message
    assert "専用フィールドと同名" in message
    assert "default_manifest" in message


def test_sandbox_extra_non_init_field_is_rejected_as_unknown() -> None:
    """init=False の内部フィールド名は「受け付けないキー」の agent 名入り ValueError。

    `_sandbox_concurrency_guard` は SandboxAgent の有効 kwarg ではないため、
    生 TypeError へすり抜けず validate_extra_kwargs で早期に reject される。
    """
    spec = SandboxAgentSpec(
        name="sbx", instructions="i", extra={"_sandbox_concurrency_guard": object()}
    )
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    message = str(excinfo.value)
    assert "'sbx'" in message
    assert "受け付けないキー" in message
    assert "_sandbox_concurrency_guard" in message


def test_sandbox_extra_valid_inherited_kwarg_passes_through() -> None:
    """Agent 継承の有効 kwarg（handoff_description）は sandbox ルートでも素通しされる。"""
    spec = SandboxAgentSpec(name="sbx", instructions="i", extra={"handoff_description": "desk"})
    agent = build_agent(spec)
    assert isinstance(agent, SandboxAgent)
    assert agent.handoff_description == "desk"


def test_sandbox_base_instructions_callable_arity_is_validated() -> None:
    """1 引数 callable の base_instructions は build 時に agent 名入り ValueError。"""
    spec = SandboxAgentSpec(name="sbx", instructions="i", base_instructions=lambda ctx: None)
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    message = str(excinfo.value)
    assert "'sbx'" in message
    assert "base_instructions" in message


def test_sandbox_base_instructions_two_arg_callable_is_accepted() -> None:
    """(context, agent) の 2 引数 callable の base_instructions は build に成功する。"""

    def base(context: object, agent: object) -> str:
        return "x"

    spec = SandboxAgentSpec(name="sbx", instructions="i", base_instructions=base)
    agent = build_agent(spec)
    assert isinstance(agent, SandboxAgent)
    assert agent.base_instructions is base


def test_sandbox_field_kwargs_derives_expected_fields() -> None:
    """宣言から導出される _SANDBOX_FIELD_KWARGS が想定の 4 フィールドちょうどである。

    導出式（fields() 差分）が壊れて空集合や過剰集合になった場合の検知用に、
    独立に列挙した期待値と突き合わせる。
    """
    assert _SANDBOX_FIELD_KWARGS == {
        "default_manifest",
        "capabilities",
        "run_as",
        "base_instructions",
    }


def test_sandbox_capabilities_list_is_copied_at_build() -> None:
    """build 後に spec 由来の capabilities リストへ append しても構築済み agent に伝播しない。

    tools と同じ遮断挙動: 権限リストである capabilities は build 時にコピーされ、
    キャッシュ済み・稼働中の SandboxAgent が事後 mutation の影響を受けない。
    """
    caps: list[object] = [object()]
    spec = SandboxAgentSpec(name="sbx", instructions="i", capabilities=caps)
    agent = build_agent(spec)
    caps.append(object())
    assert len(list(agent.capabilities)) == 1


# ---------------------------------------------------------------------------
# build_agent: mcp_servers / mcp_config（Issue #83）
# ---------------------------------------------------------------------------
def test_build_passes_mcp_fields_through() -> None:
    """`mcp_servers` / `mcp_config` の宣言値が構築済み `Agent` へそのまま渡る。"""
    server = object()  # MCPServer 実体は不要（lib は素通しするだけ）
    spec = AgentSpec(
        name="bot",
        instructions="i",
        mcp_servers=[server],
        mcp_config={"include_server_in_tool_names": True},
    )
    agent = build_agent(spec)
    assert agent.mcp_servers == [server]
    assert agent.mcp_config == {"include_server_in_tool_names": True}


def test_build_mcp_servers_list_is_copied_at_build() -> None:
    """build 後に spec 由来の mcp_servers リストへ append しても構築済み agent に伝播しない。

    tools / sandbox capabilities と同じ遮断挙動。
    """
    spec = AgentSpec(name="bot", instructions="i", mcp_servers=[object()])
    agent = build_agent(spec)
    assert agent.mcp_servers is not spec.mcp_servers
    spec.mcp_servers.append(object())
    assert len(agent.mcp_servers) == 1


def test_build_mcp_fields_unset_defer_to_sdk_defaults() -> None:
    """`mcp_servers` / `mcp_config` 未指定は kwargs へ積まれず SDK 既定に委ねられる。

    `mcp_config` の SDK 既定は `{}`（`None` ではない）ことを pin する。`build_agent` が
    `None` を渡す実装へ退行すると、`Agent.__post_init__` の `isinstance(mcp_config, dict)`
    検証に引っかかり `TypeError` になる。
    """
    spec = AgentSpec(name="bot", instructions="i")
    agent = build_agent(spec)
    assert agent.mcp_servers == []
    assert agent.mcp_config == {}


def test_build_rejects_mcp_servers_extra_collision_message() -> None:
    """extra に `mcp_servers` と同名のキーを積むと衝突メッセージ原文で弾く。"""
    spec = AgentSpec(name="bot", instructions="i", extra={"mcp_servers": []})
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    assert str(excinfo.value) == (
        "agent 'bot': extra に専用フィールドと同名のキーが含まれます: ['mcp_servers']"
    )


def test_build_rejects_mcp_config_extra_collision_message() -> None:
    """extra に `mcp_config` と同名のキーを積むと衝突メッセージ原文で弾く。"""
    spec = AgentSpec(name="bot", instructions="i", extra={"mcp_config": {}})
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    assert str(excinfo.value) == (
        "agent 'bot': extra に専用フィールドと同名のキーが含まれます: ['mcp_config']"
    )


def test_build_rejects_both_mcp_extra_collisions_sorted() -> None:
    """extra に `mcp_servers` / `mcp_config` の両方を積むとソート済みキー一覧で弾く。"""
    spec = AgentSpec(name="bot", instructions="i", extra={"mcp_servers": [], "mcp_config": {}})
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    assert str(excinfo.value) == (
        "agent 'bot': extra に専用フィールドと同名のキーが含まれます: ['mcp_config', 'mcp_servers']"
    )


def test_sandbox_spec_passes_mcp_fields_through() -> None:
    """`SandboxAgentSpec` でも `mcp_servers` / `mcp_config` が構築済み `SandboxAgent` へ渡る。

    2 分岐（`if spec.mcp_servers:` / `if spec.mcp_config is not None:`）は `is_sandbox` 分岐
    より前に置かれているため sandbox 経路にも効くはずだが、これを個別に pin しないと
    「2 分岐を `return Agent(**kwargs)` の直前へ移す」変異が T4-a / T4-b を緑のまま通し、
    `SandboxAgentSpec` の MCP 宣言だけが無言で落ちる（silent capability loss）。
    """
    server = object()
    spec = SandboxAgentSpec(
        name="s",
        instructions="i",
        mcp_servers=[server],
        mcp_config={"convert_schemas_to_strict": True},
    )
    agent = build_agent(spec)
    assert isinstance(agent, SandboxAgent)
    assert agent.mcp_servers == [server]
    assert agent.mcp_config == {"convert_schemas_to_strict": True}


def test_dedicated_agent_kwargs_are_valid_agent_fields() -> None:
    """`_DEDICATED_AGENT_KWARGS` の各キーは（`guardrails` を除き）`Agent` の実在 kwarg である。

    実在しない kwarg 名を紛れ込ませると（例: `mcp_configs` の typo）、extra 検証の衝突判定が
    その名前とは一致しなくなり、`extra` へ同名キーを積んでも衝突として弾かれず素通りする。
    `guardrails` は `Agent` の kwarg ではない名前参照フィールドとして意図的に列挙されている
    唯一の例外（`builders.py` の定義直前コメントで明示）。
    """
    assert _DEDICATED_AGENT_KWARGS - {"guardrails"} <= _AGENT_FIELD_NAMES


# ---------------------------------------------------------------------------
# check_stop_at_tool_names_resolved: 停止対象名と実行時ツール名の突合（Issue #115 T3）
# ---------------------------------------------------------------------------
# 新関数は名前で import せず `_adapters.<関数名>` で属性参照する（未実装時に本モジュール
# 全体の収集を壊さず、新規テストだけを AttributeError で落とすため）。
@function_tool
def get_order(order_id: str) -> str:
    """注文を取得する。"""
    return order_id


@function_tool(name_override="refund")
def _refund_impl(order_id: str) -> str:
    """返金する（name_override で実行時名を refund にする）。"""
    return order_id


@function_tool(name_override="ask_researcher")
def _ask_researcher_impl(query: str) -> str:
    """調査担当へ問い合わせる（sub_agent の as_tool 名相当）。"""
    return query


@function_tool(name_override="disabled_tool", is_enabled=False)
def _disabled_impl(x: str) -> str:
    """is_enabled=False のツール。"""
    return x


def _router(tools: list[object], behavior: object, **kwargs: object) -> Agent:
    """停止指定を持つ router エージェントを直接構築する。"""
    return Agent(
        name="router",
        instructions="i",
        tools=tools,
        tool_use_behavior=behavior,
        **kwargs,
    )


def _assert_no_warning(agent: object) -> None:
    """突合ヘルパが警告を 1 件も出さないことを記録件数 0 で確認する。"""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        assert _adapters.check_stop_at_tool_names_resolved("router", agent) is None
    assert records == []


def test_stop_at_resolved_plain_tool_name_matches_without_warning() -> None:
    """通常の function tool 名（関数名由来）と一致すれば警告しない。"""
    agent = _router([get_order], {"stop_at_tool_names": ["get_order"]})
    _assert_no_warning(agent)


def test_stop_at_resolved_name_override_matches_without_warning() -> None:
    """name_override で付けた実行時名と一致すれば警告しない。"""
    agent = _router([_refund_impl], {"stop_at_tool_names": ["refund"]})
    _assert_no_warning(agent)


def test_stop_at_resolved_qualified_name_matches_without_warning() -> None:
    """tool_namespace で name と異なる qualified_name を持つツールは qualified_name 一致で通る。"""
    (namespaced,) = tool_namespace(name="billing", description="d", tools=[get_order])
    assert namespaced.qualified_name == "billing.get_order"
    assert namespaced.name != namespaced.qualified_name
    agent = _router([namespaced], {"stop_at_tool_names": ["billing.get_order"]})
    _assert_no_warning(agent)


def test_stop_at_resolved_namespaced_tool_bare_name_matches_without_warning() -> None:
    """name と qualified_name が異なるツールも素の name 一致で通る（SDK は name でも停止する）。"""
    (namespaced,) = tool_namespace(name="billing", description="d", tools=[get_order])
    assert namespaced.name != namespaced.qualified_name
    agent = _router([namespaced], {"stop_at_tool_names": ["get_order"]})
    _assert_no_warning(agent)


def test_stop_at_resolved_disabled_tool_is_still_candidate() -> None:
    """is_enabled=False のツール名も候補に含まれ、一致すれば警告しない。"""
    agent = _router([_disabled_impl], {"stop_at_tool_names": ["disabled_tool"]})
    _assert_no_warning(agent)


def test_stop_at_resolved_mismatch_warns_full_message() -> None:
    """不一致名があると RuntimeWarning（エージェント名・不一致名・sorted 済み候補の全文）。

    tools の定義順（refund, get_order, ask_researcher）は sorted 順と異なる構成にし、
    候補一覧が sorted されることを確かめる。
    """
    agent = _router(
        [_refund_impl, get_order, _ask_researcher_impl],
        {"stop_at_tool_names": ["refund", "researcher"]},
    )
    with pytest.warns(RuntimeWarning) as records:
        _adapters.check_stop_at_tool_names_resolved("router", agent)
    assert len(records) == 1
    assert str(records[0].message) == (
        "agent 'router': tool_use_behavior の stop_at_tool_names に、このエージェントの "
        "function tool の実行時名（name / qualified_name）と一致しない名前があります: "
        "['researcher']（候補: ['ask_researcher', 'get_order', 'refund']）。"
        "この名前では停止しません"
    )


def test_stop_at_resolved_mismatch_lists_candidates_sorted() -> None:
    """候補一覧は sorted 済みで列挙される。

    候補 3 件では set の反復順が偶然 sorted 順と一致するハッシュシードがあり、sorted の
    退行を見逃す。候補を 8 件にして偶然一致をほぼ起こらなくする。
    """

    def _noop(x: str) -> str:
        """ダミーのツール本体。"""
        return x

    names = ["tool_h", "tool_c", "tool_f", "tool_a", "tool_g", "tool_b", "tool_e", "tool_d"]
    tools = [function_tool(_noop, name_override=n) for n in names]
    agent = _router(tools, {"stop_at_tool_names": ["missing"]})
    with pytest.warns(RuntimeWarning) as records:
        _adapters.check_stop_at_tool_names_resolved("router", agent)
    assert len(records) == 1
    assert f"（候補: {sorted(names)}）" in str(records[0].message)


def test_stop_at_resolved_mismatch_lists_names_in_declared_order() -> None:
    """複数の不一致名は宣言順（sorted ではない）で列挙される。"""
    agent = _router([get_order], {"stop_at_tool_names": ["zeta", "get_order", "alpha"]})
    with pytest.warns(RuntimeWarning) as records:
        _adapters.check_stop_at_tool_names_resolved("router", agent)
    assert len(records) == 1
    assert "['zeta', 'alpha']（候補: ['get_order']）" in str(records[0].message)


def test_stop_at_resolved_hosted_tool_name_is_not_candidate() -> None:
    """hosted tool（WebSearchTool の name 'web_search'）は候補に含めず不一致として警告する。"""
    hosted = WebSearchTool()
    assert hosted.name == "web_search"
    agent = _router([hosted, get_order], {"stop_at_tool_names": ["web_search"]})
    with pytest.warns(RuntimeWarning) as records:
        _adapters.check_stop_at_tool_names_resolved("router", agent)
    assert len(records) == 1
    assert str(records[0].message) == (
        "agent 'router': tool_use_behavior の stop_at_tool_names に、このエージェントの "
        "function tool の実行時名（name / qualified_name）と一致しない名前があります: "
        "['web_search']（候補: ['get_order']）。この名前では停止しません"
    )


def test_stop_at_resolved_skips_agent_with_mcp_servers() -> None:
    """mcp_servers を持つエージェントは不一致でも突合せず警告しない。"""
    agent = _router([get_order], {"stop_at_tool_names": ["mcp_only_tool"]}, mcp_servers=[object()])
    _assert_no_warning(agent)


def test_stop_at_resolved_skips_sandbox_agent() -> None:
    """SandboxAgent（capabilities 未指定＝SDK 既定）は不一致でも突合せず警告しない。"""
    agent = SandboxAgent(
        name="router",
        instructions="i",
        tools=[get_order],
        tool_use_behavior={"stop_at_tool_names": ["sandbox_only_tool"]},
    )
    _assert_no_warning(agent)


def test_stop_at_resolved_ignores_non_dict_behavior() -> None:
    """tool_use_behavior が dict でない（文字列形・既定）なら何もしない。"""
    _assert_no_warning(_router([get_order], "stop_on_first_tool"))
    _assert_no_warning(Agent(name="router", instructions="i", tools=[get_order]))


def test_stop_at_resolved_ignores_agent_without_attributes() -> None:
    """tool_use_behavior 属性を持たないオブジェクト（フェイク Agent 等）は何もしない。"""
    _assert_no_warning(object())


def test_stop_at_resolved_reruns_shape_check_on_string_value() -> None:
    """文字列値の dict を持つ agent を直接渡すと、形状検査の再実行で ValueError。"""
    agent = _router([_refund_impl], {"stop_at_tool_names": "refund"})
    with pytest.raises(ValueError) as excinfo:
        _adapters.check_stop_at_tool_names_resolved("router", agent)
    assert str(excinfo.value) == (
        "agent 'router': tool_use_behavior の stop_at_tool_names は str の list / tuple "
        "である必要がありますが 'str' が渡されました: 'refund'"
    )


# ---------------------------------------------------------------------------
# build_agent: stop_at_tool_names の形状検査（Issue #115 T2・registry 非経由）
# ---------------------------------------------------------------------------
def test_build_rejects_stop_at_tool_names_string_value() -> None:
    """build_agent を直接呼んでも、dict 形の文字列値は agent 名と値を含む全文の ValueError。"""
    spec = AgentSpec(
        name="router",
        instructions="i",
        extra={"tool_use_behavior": {"stop_at_tool_names": "refund"}},
    )
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    assert str(excinfo.value) == (
        "agent 'router': tool_use_behavior の stop_at_tool_names は str の list / tuple "
        "である必要がありますが 'str' が渡されました: 'refund'"
    )


def test_sandbox_build_rejects_stop_at_tool_names_string_value() -> None:
    """SandboxAgentSpec でも同じ形状検査で agent 名入りの ValueError になる。"""
    spec = SandboxAgentSpec(
        name="sbx",
        instructions="i",
        extra={"tool_use_behavior": {"stop_at_tool_names": "refund"}},
    )
    with pytest.raises(ValueError) as excinfo:
        build_agent(spec)
    assert str(excinfo.value) == (
        "agent 'sbx': tool_use_behavior の stop_at_tool_names は str の list / tuple "
        "である必要がありますが 'str' が渡されました: 'refund'"
    )


def test_build_accepts_valid_stop_at_tool_names_and_string_behavior() -> None:
    """正しい dict 形と文字列形 "stop_on_first_tool" は構築でき、値が素通しされる。"""
    behavior = {"stop_at_tool_names": ["refund"]}
    agent = build_agent(
        AgentSpec(name="router", instructions="i", extra={"tool_use_behavior": behavior})
    )
    assert agent.tool_use_behavior == {"stop_at_tool_names": ["refund"]}
    agent = build_agent(
        AgentSpec(
            name="router", instructions="i", extra={"tool_use_behavior": "stop_on_first_tool"}
        )
    )
    assert agent.tool_use_behavior == "stop_on_first_tool"
