"""AGT ガバナンステスト用の共通フィクスチャ・ヘルパ（extra 未導入時 skip / fake 注入基盤）。

`agt_symbols` フィクスチャが AGT（agent-governance-toolkit）の実シンボルを `pytest.importorskip`
付きで提供し、governance extra 未導入環境では AGT 依存テストを skip する
（`tests/runtime/lightning/conftest.py` と同様のディレクトリ局所 conftest）。
fake（許可一辺倒 policy / 記録 sink）は AGT 非依存の plain オブジェクトで、
`GovernedAgentBuilder` の L1 検証（実 `govern_spec` への DI 注入）に使う。
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest


class AllowAllPolicy:
    """常に許可する fake policy（`check_tool` / `check_content` が None を返す・AGT 非依存）。"""

    def check_tool(self, tool_name: str) -> str | None:
        """ツール名照合（常に許可）。"""
        return None

    def check_content(self, content: str) -> str | None:
        """引数 JSON 照合（常に許可）。"""
        return None


class RecordingSink:
    """`record(...)` 呼び出しを記録する fake 監査 sink（AGT 非依存）。"""

    def __init__(self) -> None:
        """記録リストを初期化する。"""
        self.records: list[dict[str, Any]] = []

    def record(
        self,
        agent_id: str,
        action: str,
        decision: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """監査レコードをメモリに積む（AGT `AuditLog.record` と同シグネチャ）。"""
        self.records.append(
            {"agent_id": agent_id, "action": action, "decision": decision, "details": details}
        )


@pytest.fixture
def allow_all_policy() -> AllowAllPolicy:
    """許可一辺倒の fake policy を返す。"""
    return AllowAllPolicy()


@pytest.fixture
def recording_sink() -> RecordingSink:
    """記録 fake sink を返す。"""
    return RecordingSink()


@pytest.fixture
def agt_symbols() -> tuple[Any, Any, Any]:
    """AGT 実シンボル 3 つ組を返す（governance extra 未導入環境では skip）。

    Returns:
        `(GovernancePolicy, AuditLog, PolicyViolationError)`。
    """
    pytest.importorskip(
        "openai_agents_trust", reason="governance extra（agent-governance-toolkit）未導入"
    )
    from openai_agents_trust import AuditLog, GovernancePolicy

    with warnings.catch_warnings():
        # agent_os は legacy パッケージ名告知の DeprecationWarning を出すため抑制する。
        warnings.simplefilter("ignore", DeprecationWarning)
        from agent_os.exceptions import PolicyViolationError

    return GovernancePolicy, AuditLog, PolicyViolationError
