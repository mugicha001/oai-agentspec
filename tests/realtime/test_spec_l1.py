"""L1: RealtimeAgentSpec / RealtimeHandoffConfig の型レベル防御を introspection で固定する。

非対応フィールドが型（dataclass フィールド）から排除されていること、`RealtimeHandoffConfig` が
`input_filter` を型として持たず frozen であることを検証する（agents 非依存・FR-1 の第一防御）。
"""

from __future__ import annotations

import dataclasses

import pytest

from oai_agentspec.realtime.spec import RealtimeAgentSpec, RealtimeHandoffConfig

# RealtimeAgent が受け付けないため型レベルで排除すべきフィールド（FR-1 第一防御）。
_FORBIDDEN_SPEC_FIELDS = frozenset(
    {
        "model",
        "model_settings",
        "input_guardrails",
        "sub_agents",
        "sub_agent_tools",
        "dynamic_handoffs",
        "output_type",
        "tool_use_behavior",
    }
)


def _field_names(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


@pytest.mark.parametrize("forbidden", sorted(_FORBIDDEN_SPEC_FIELDS))
def test_spec_excludes_unsupported_field(forbidden: str) -> None:
    """RealtimeAgentSpec は非対応フィールドを dataclass フィールドとして持たない。"""
    assert forbidden not in _field_names(RealtimeAgentSpec)


def test_spec_field_set_is_exact() -> None:
    """RealtimeAgentSpec のフィールド集合を厳密に固定する（blacklist の網羅漏れ対策）。

    実行時 config 系（voice / modalities / model_name 等）を含む任意のフィールド追加は
    宣言型の契約変更であり、本テストの明示的な更新（= レビュー）を要求する。
    """
    assert _field_names(RealtimeAgentSpec) == {
        "name",
        "instructions",
        "prompt",
        "tools",
        "hooks",
        "handoff_description",
        "mcp_servers",
        "mcp_config",
        "output_guardrails",
        "handoffs",
        "handoff_options",
        "extra",
    }


def test_handoff_config_field_set_is_exact() -> None:
    """RealtimeHandoffConfig のフィールド集合を厳密に固定する（input_filter 等の混入防止）。"""
    assert _field_names(RealtimeHandoffConfig) == {
        "on_handoff",
        "input_type",
        "tool_name_override",
        "tool_description_override",
        "is_enabled",
    }


def test_handoff_config_excludes_input_filter() -> None:
    """RealtimeHandoffConfig は input_filter を型として持たない（realtime_handoff に無い引数）。"""
    assert "input_filter" not in _field_names(RealtimeHandoffConfig)


def test_handoff_config_has_no_passthrough_escape_hatch() -> None:
    """RealtimeHandoffConfig は options / extra の素通し口を持たない（input_filter 密輸を防ぐ）。"""
    names = _field_names(RealtimeHandoffConfig)
    assert "options" not in names
    assert "extra" not in names


def test_handoff_config_is_frozen() -> None:
    """RealtimeHandoffConfig は frozen dataclass で、フィールド代入は禁止される。"""
    config = RealtimeHandoffConfig(tool_name_override="go")
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.tool_name_override = "changed"  # type: ignore[misc]


def test_handoff_config_defaults() -> None:
    """RealtimeHandoffConfig の既定値（on_handoff/input_type=None, is_enabled=True）。"""
    config = RealtimeHandoffConfig()
    assert config.on_handoff is None
    assert config.input_type is None
    assert config.tool_name_override is None
    assert config.tool_description_override is None
    assert config.is_enabled is True
