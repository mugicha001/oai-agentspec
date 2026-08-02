"""L1: `AgentSpec.guardrails`（名前参照による宣言的 guardrail 装着）の宣言面検証。

`guardrails` フィールドの存在・`kw_only`・既定値の非共有・フィールド集合の固定（既存 15 +
`guardrails` = 16 件）・既存フィールドの位置引数束縛が不変であること・宣言値が正規化されず
そのまま保持されること・`SandboxAgentSpec` への継承を pin する（解決・検証は build / validate の
責務なので本層では扱わない）。`spec.py` 自体の SDK 隔離は `tests/test_sandbox_spec.py` の
`test_spec_module_does_not_import_agents_sdk` が既に pin しているため重複させない。
"""

from __future__ import annotations

import dataclasses

import pytest

from oai_agentspec.spec import AgentSpec, SandboxAgentSpec

pytestmark = pytest.mark.unit

#: `AgentSpec` の宣言フィールド全件（既存 15 + `guardrails`）。
EXPECTED_FIELDS = {
    "name",
    "instructions",
    "prompt",
    "tools",
    "model",
    "model_settings",
    "hooks",
    "input_guardrails",
    "output_guardrails",
    "handoffs",
    "handoff_options",
    "sub_agents",
    "sub_agent_tools",
    "dynamic_handoffs",
    "extra",
    "guardrails",
}


def test_guardrails_field_is_kw_only() -> None:
    """`guardrails` フィールドが存在し `kw_only`（既存フィールドの位置引数束縛を壊さない）。"""
    fields_by_name = {f.name: f for f in dataclasses.fields(AgentSpec)}
    assert "guardrails" in fields_by_name
    assert fields_by_name["guardrails"].kw_only is True


def test_guardrails_defaults_to_empty_list_not_shared() -> None:
    """既定は空 list で、インスタンス間で共有されない（default_factory）。"""
    a = AgentSpec(name="a", instructions="i")
    b = AgentSpec(name="b", instructions="i")
    assert a.guardrails == []
    assert b.guardrails == []
    a.guardrails.append("pii")
    assert b.guardrails == []
    assert a.guardrails is not b.guardrails


def test_agent_spec_field_set_is_pinned() -> None:
    """フィールド集合を `==` で 16 件に固定する（追加・削除の両方向を検知する）。"""
    assert {f.name for f in dataclasses.fields(AgentSpec)} == EXPECTED_FIELDS


def test_positional_binding_is_unchanged() -> None:
    """既存の非 kw_only フィールドは従来どおり位置で渡せる（`guardrails` が割り込まない）。"""
    prompt = object()
    tool = object()
    model = object()
    settings = object()
    hooks = object()
    spec = AgentSpec("n", "instr", prompt, [tool], model, settings, hooks, ["billing"])
    assert spec.name == "n"
    assert spec.instructions == "instr"
    assert spec.prompt is prompt
    assert spec.tools == [tool]
    assert spec.model is model
    assert spec.model_settings is settings
    assert spec.hooks is hooks
    assert spec.handoffs == ["billing"]
    assert spec.guardrails == []


def test_guardrails_values_are_kept_as_declared() -> None:
    """宣言した名前はそのまま保持され、正規化・検証・重複排除をしない。"""
    names = ["pii", "pii", " Injection ", "未登録でもよい"]
    spec = AgentSpec(name="a", instructions="i", guardrails=names)
    assert spec.guardrails == ["pii", "pii", " Injection ", "未登録でもよい"]


def test_guardrails_accepts_non_str_without_validation() -> None:
    """非 str 要素も宣言時点では拒否しない（検証は build / validate の責務）。"""
    entity = object()
    spec = AgentSpec(name="a", instructions="i", guardrails=[entity])  # type: ignore[list-item]
    assert spec.guardrails == [entity]


def test_sandbox_agent_spec_inherits_guardrails() -> None:
    """`SandboxAgentSpec` も `guardrails` を継承する（既定空・宣言値を保持）。"""
    assert "guardrails" in {f.name for f in dataclasses.fields(SandboxAgentSpec)}
    assert SandboxAgentSpec(name="s").guardrails == []
    spec = SandboxAgentSpec(name="s", guardrails=["pii"])
    assert spec.guardrails == ["pii"]
