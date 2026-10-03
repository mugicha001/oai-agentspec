"""MCP サーバの統治ガードレール（`mcp_governance_guardrail`・agents 非依存の公開関数）。

統治（ツール単位の allow / deny と `tool:` 監査）を、MCPServer のツール入力ガードレールの位置で
行うための SDK ネイティブ guardrail を返す（ADR-0048）。本体は `_adapters/governance.py` に置き、
本モジュールは関数内の遅延 import で委譲するだけである（SDK 隔離・NFR-1。governance extra
未導入でも本モジュールと窓口の import は壊れない）。
"""

from __future__ import annotations


def mcp_governance_guardrail() -> object:
    """MCPServer の `tool_input_guardrails` の先頭に置く統治ガードレールを返す。

    `GovernedAgentBuilder` で構築したエージェントが MCP ツールを呼ぶと、ガードレールは当該
    エージェントに装着された監査フックを辿り、そのポリシー・監査 sink・`spec.name` で呼び出しを
    評価する。deny は `tool:` deny を記録してから `PolicyViolationError` を送出し（run は SDK の
    `UserError` で終了し `__cause__` に原例外が載る）、後続の内容検査ガードレールは呼ばれない。
    allow は `tool:` allow を記録し、`on_tool_start` での再評価を省く。統治していない
    エージェントの呼び出しは記録せずに素通しする。付け忘れた場合も `on_tool_start` が従来どおり
    評価する（安全網）。

    ```python
    server = MCPServerStdio(
        params=...,
        tool_input_guardrails=[mcp_governance_guardrail(), *content_guardrails],
    )
    ```

    内容検査より前に評価させるため、リストの先頭に置く。`spec.tools` には付けない（build 時の
    ラップと二重評価になる）。

    Returns:
        SDK の `ToolInputGuardrail`（不透明値）。`get_name()` は `"mcp_governance_guardrail"`。
    """
    from ..._adapters import mcp_governance_guardrail as _factory

    return _factory()
