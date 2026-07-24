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

import pytest
from agents import Agent
from agents.sandbox import SandboxAgent

from oai_agentspec._adapters import build_agent
from oai_agentspec._adapters.builders import _SANDBOX_FIELD_KWARGS
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
