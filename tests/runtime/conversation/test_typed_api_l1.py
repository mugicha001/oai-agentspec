"""L1: 会話公開 API の型付け（SendStatus / ApprovalDecision）と公開契約を検証する。

`SendStatus`（StrEnum）と `ApprovalDecision`（型付き承認入力）の値・文字列互換・不変性、および
公開窓口 `oai_agentspec.runtime.conversation` の `__all__` 契約（新シンボルの公開・旧
`Conversation` エイリアスの撤去）を外部依存なしで確認する。
"""

from __future__ import annotations

import dataclasses

import pytest

from oai_agentspec.runtime import conversation
from oai_agentspec.runtime.conversation import ApprovalDecision, SendResult, SendStatus

pytestmark = pytest.mark.unit


def test_send_status_values_and_str_compat() -> None:
    """SendStatus は "final"/"pending" を値に持ち、StrEnum として文字列比較と互換。"""
    assert SendStatus.FINAL == "final"
    assert SendStatus.PENDING == "pending"
    # SendResult.status に enum を載せても従来の文字列比較が成立する（後方互換）。
    result = SendResult(status=SendStatus.FINAL, output="x")
    assert result.status == "final"


def test_approval_decision_fields_and_default() -> None:
    """ApprovalDecision は call_id/approve 必須・rejection_message 既定 None・frozen。"""
    decision = ApprovalDecision(call_id="c1", approve=True)
    assert decision.call_id == "c1"
    assert decision.approve is True
    assert decision.rejection_message is None
    assert dataclasses.is_dataclass(decision)
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.approve = False


def test_public_api_contract_adds_typed_symbols_and_drops_alias() -> None:
    """公開窓口は ApprovalDecision/SendStatus を公開し、旧 Conversation エイリアスを撤去済み。"""
    assert "ApprovalDecision" in conversation.__all__
    assert "SendStatus" in conversation.__all__
    assert "Conversation" not in conversation.__all__
    assert not hasattr(conversation, "Conversation")
