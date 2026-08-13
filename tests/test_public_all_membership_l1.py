"""L1: コア `oai_agentspec.__all__` のメンバ集合を pin する（NFR-3 / ADR 0027）。

`CLAUDE.md` は「公開シンボルの振る舞いと `__all__` のメンバ集合は契約」と定めるが、既存
`tests/test_public_naming_l1.py` は禁止語彙（`fake` / `mock` / `dummy`）の照合とテーマ B
シンボルの非包含のみを見ており、メンバ集合そのものを固定していない。本ファイルが集合を
1 箇所で固定し、意図しない追加（過大側）と意図しない削除（過小側）の両方向を検知する。

期待値は「ADR 0027 適用前の 36 件 + `action_next_turn_agent`」の 37 件。集合の完全一致で
pin するため、余分な要素の混入も要素の欠落も同じ 1 本で落ちる。あわせて件数（重複混入の
検知）・挿入位置（大文字群 -> 小文字群のアルファベット順という既存の並び規則）・
`getattr` 可能性（NFR-3 の計測基準）を固定する。
"""

from __future__ import annotations

import pytest

import oai_agentspec

pytestmark = pytest.mark.unit

# ADR 0027 適用前のコア `__all__`（36 件）。差分を目視で追えるよう全件を literal で持つ。
_BEFORE: frozenset[str] = frozenset(
    {
        "END",
        "START",
        "AgentNames",
        "AgentRegistry",
        "AgentSpec",
        "FacadeMode",
        "HandoffConfig",
        "HandoffEdge",
        "HandoffGraph",
        "IntegrityCheck",
        "IntegrityError",
        "NextTurnPolicy",
        "NextTurnRule",
        "NodeFn",
        "NodeHook",
        "NodeResults",
        "PromptLayout",
        "PromptStore",
        "PromptTemplate",
        "PromptTemplateIntegrityError",
        "RegistryFrozenError",
        "Router",
        "SandboxAgentSpec",
        "ToolRegistry",
        "ToolSpec",
        "WorkflowFrozenError",
        "WorkflowGraph",
        "apply_next_turn_policy",
        "default_input_filter",
        "dynamic_prompt",
        "from_specs",
        "function_tool",
        "lockdown",
        "next_turn_agent",
        "resolve_next_agent",
        "validate_agent_names",
    }
)

# ADR 0027 で純追加する唯一のシンボル。
_ADDED: frozenset[str] = frozenset({"action_next_turn_agent"})

_EXPECTED: frozenset[str] = _BEFORE | _ADDED


def test_変更前のコアall__は36件である() -> None:
    """期待値の literal 自体が 36 件であることを固定する（期待値側の取り違えを検知する）。"""
    assert len(_BEFORE) == 36


def test_コアall__のメンバ集合が変更前36件と追加1件に完全一致する() -> None:
    """過大側（余分な要素）・過小側（要素の欠落）の両方向を 1 本の完全一致で検知する。"""
    actual = frozenset(oai_agentspec.__all__)

    assert actual - _EXPECTED == frozenset(), "コア __all__ に想定外のシンボルが増えている"
    assert _EXPECTED - actual == frozenset(), "コア __all__ から想定のシンボルが欠けている"
    assert actual == _EXPECTED


def test_コアall__の件数は37件で重複を含まない() -> None:
    """集合一致だけでは検知できない重複要素（同名の二重掲載）を件数で落とす。"""
    assert len(oai_agentspec.__all__) == 37
    assert len(set(oai_agentspec.__all__)) == len(oai_agentspec.__all__)


def test_action_next_turn_agentはWorkflowGraphとapply_next_turn_policyの間に並ぶ() -> None:
    """既存の並び規則（大文字群 -> 小文字群のアルファベット順）どおりの挿入位置を固定する。"""
    names = list(oai_agentspec.__all__)
    index = names.index("action_next_turn_agent")

    assert names[index - 1] == "WorkflowGraph"
    assert names[index + 1] == "apply_next_turn_policy"


def test_コアall__の全シンボルがモジュール属性として取得できる() -> None:
    """NFR-3 の計測基準（`__all__` 全件が `hasattr` で取得できる）を固定する。"""
    missing = [name for name in oai_agentspec.__all__ if not hasattr(oai_agentspec, name)]

    assert missing == []
