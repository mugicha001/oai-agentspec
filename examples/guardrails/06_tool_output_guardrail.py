"""ツール境界 guardrail（中間ツール出力の検査・OWASP LLM02/05）の最小例（実 API）。

ツールガードレールは `tool_guardrail(detector, on="output")` で生成し、ツール定義時に
`function_tool(_func, tool_output_guardrails=[...])` で宣言する（SDK ネイティブ流儀）。ここでは
データを返すツールの**出力**を検知器で検査する。装着は実行本体・宣言メタ（name / description /
params_json_schema / needs_approval）を変えず、内容検査のみを足す（実行可否の allow / deny 制御は
新設しない＝AGT ガバナンスの責務）。

trip 時の挙動は `on_trip` で選ぶ:
  - 'reject'（既定）: ツール出力をモデルへ返さず注釈メッセージへ差し替えて会話を続行する。
  - 'raise': `ToolOutputGuardrailTripwireTriggered` を送出して実行を中断する。
  - 'allow': 検知しても通過する。

本例では同一関数から 'reject' 版と 'raise' 版の 2 ツールを `function_tool` で作り挙動の差を示す。
既存ツール（`as_tool` 等 `function_tool` で定義し直せないもの）へ後付けする場合は `guard_tool`
（コメント参照）。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/guardrails/06_tool_output_guardrail.py

導入: pip install 'oai-agentspec[guardrails]'（依存ゼロ opt-in extra）。
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

from agents import Runner, ToolOutputGuardrailTripwireTriggered

from oai_agentspec import AgentRegistry, AgentSpec, function_tool
from oai_agentspec.runtime.guardrails import Detection, tool_guardrail

# examples/ 共有ヘルパ _azure を解決するため _shared を import パスへ。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402


def _lookup_customer(customer_id: str) -> str:
    """顧客 ID から連絡先を返す（本例では PII を含むダミーデータを返す）。

    Args:
        customer_id: 顧客 ID。

    Returns:
        顧客の連絡先を含む文字列（メールアドレスを含む）。
    """
    return f"customer {customer_id}: contact email is taro.yamada@example.com"


def _email_detector(text: str) -> Detection:
    """ツール出力にメールアドレスらしき文字列が含まれるか検査する簡易検知器。

    Args:
        text: ツール出力テキスト。

    Returns:
        メールアドレスを検出したら triggered=True の `Detection`。
    """
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text):
        return Detection(triggered=True, reason="email address in tool output")
    return Detection(triggered=False)


def build_registry() -> AgentRegistry:
    """reject 版 / raise 版のツールを `function_tool` で定義した 2 エージェントを登録する。

    `function_tool(_func, tool_output_guardrails=[tool_guardrail(...)])` でツール定義時に
    guardrail を宣言する。同一関数から `on_trip` 違いで 2 ツールを作る。装着後も name /
    needs_approval 等の宣言メタは維持される（実行本体・スキーマは不変・内容検査のみ）。

    Returns:
        2 エージェントを登録済みの `AgentRegistry`。
    """
    instructions = (
        "あなたはサポート係です。顧客情報を聞かれたら必ず lookup_customer ツールを使って"
        "回答してください。"
    )

    # 既定 'reject': 検知してもツール出力を注釈へ差し替えて会話は続行する。
    reject_tool = function_tool(
        _lookup_customer,
        name_override="lookup_customer",
        tool_output_guardrails=[tool_guardrail(_email_detector, on="output", on_trip="reject")],
    )
    # 'raise': 検知したらツール出力 guardrail が例外を送出して実行を中断する。
    raise_tool = function_tool(
        _lookup_customer,
        name_override="lookup_customer",
        tool_output_guardrails=[tool_guardrail(_email_detector, on="output", on_trip="raise")],
    )

    # 既存ツール（function_tool で定義し直せない as_tool 等）への後付けは guard_tool:
    #   from oai_agentspec.runtime.guardrails import guard_tool
    #   guarded = guard_tool(existing_tool, output_detector=_email_detector, on_trip="reject")

    # guardrail を載せても name は元のまま（差し替えるのは guardrail だけ）。
    assert reject_tool.name == "lookup_customer"

    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="agent-reject",
            instructions=instructions,
            model=azure_model(),
            tools=[reject_tool],
        )
    )
    registry.register(
        AgentSpec(
            name="agent-raise",
            instructions=instructions,
            model=azure_model(),
            tools=[raise_tool],
        )
    )
    registry.validate()
    return registry


async def main() -> None:
    registry = build_registry()
    prompt = "顧客 C-001 の連絡先を教えて。"

    # reject: ツール出力の PII を検知すると、出力がモデルへ返らず注釈へ差し替わって続行する。
    print("--- on_trip='reject'（注釈付き返却で続行） ---")
    result = await Runner.run(registry.get("agent-reject"), input=prompt)
    print("output:", result.final_output[:120])

    # raise: 同じ検知で実行を中断する（例外送出）。
    print("\n--- on_trip='raise'（ブロックで中断） ---")
    try:
        await Runner.run(registry.get("agent-raise"), input=prompt)
        print("（中断されませんでした）")
    except ToolOutputGuardrailTripwireTriggered:
        print("ツール出力 guardrail が PII を検知して実行を中断しました")


if __name__ == "__main__":
    asyncio.run(main())
