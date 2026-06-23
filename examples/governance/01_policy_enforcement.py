"""AGT ガバナンス（ツール単位ポリシー強制）の最小例（実 API）。

`AgentRegistry(agent_builder=GovernedAgentBuilder(policy=...))` の builder 差し替え 1 行で、
登録済み spec の全 `FunctionTool` にポリシー強制（実行前 allow / deny）と監査記録が後付けされる。
`AgentSpec` / `tools` の宣言面は不変（ツール定義・spec 記述には一切手を入れない）。

ポリシーは YAML（`policies/support.yaml`）で宣言する。本統合で強制されるのは
`allowed_tools`（ツール名 allowlist）と `blocked_patterns`（ツール引数 JSON への正規表現照合・
生のワイヤ文字列と JSON 正規化文字列の両方）で、違反はツール実行前に AGT
`PolicyViolationError` で拒否される。SDK `Runner` 経由では SDK 例外にラップされ得るため、
捕捉時は `__cause__` も確認する。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/governance/01_policy_enforcement.py

導入: pip install 'oai-agentspec[governance]'（AGT を取り込む opt-in extra）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agents import Runner, function_tool

from oai_agentspec import AgentRegistry, AgentSpec

# 拒否例外は公開窓口から取得できる（AGT 内部パッケージ import や警告抑制は不要）。
from oai_agentspec.runtime.governance import GovernedAgentBuilder, PolicyViolationError

# examples/ 共有ヘルパ _azure を解決するため _shared を import パスへ。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402

POLICY_PATH = Path(__file__).resolve().parent / "policies" / "support.yaml"


@function_tool
def lookup_order(order_id: str) -> str:
    """注文の状況を返す（allowlist に載った許可ツール）。

    Args:
        order_id: 注文 ID。

    Returns:
        注文状況の文字列。
    """
    return f"注文 {order_id}: 発送準備中です"


@function_tool
def delete_order(order_id: str) -> str:
    """注文を削除する（allowlist 未掲載・ポリシーで実行前に拒否されるツール）。

    Args:
        order_id: 注文 ID。

    Returns:
        削除結果の文字列（ポリシー違反時はここに到達しない）。
    """
    return f"注文 {order_id} を削除しました"


def build_registry() -> tuple[AgentRegistry, GovernedAgentBuilder]:
    """ガバナンス builder を注入した registry を組む（宣言面は通常どおり）。

    Returns:
        `(registry, builder)`。builder は監査 sink の取得（`audit_sink`）用に返す。
    """
    # 追加はこの 1 行（builder 差し替え）と policy ファイルのみ。spec / tools は通常どおり書く。
    builder = GovernedAgentBuilder(policy=POLICY_PATH)
    registry = AgentRegistry(agent_builder=builder)
    registry.register(
        AgentSpec(
            name="support",
            instructions=(
                "あなたは注文サポート担当です。注文の照会には lookup_order を、"
                "削除依頼には delete_order を必ず使ってください。"
            ),
            model=azure_model(),
            tools=[lookup_order, delete_order],
        )
    )
    registry.validate()
    return registry, builder


async def _ask(registry: AgentRegistry, text: str) -> None:
    """1 入力を Runner で実行し、許可（実行）/ 拒否（実行前ブロック）を表示する。

    Args:
        registry: spec を登録済みの `AgentRegistry`（`get` で govern 済み Agent を遅延構築）。
        text: ユーザー入力。
    """
    agent = registry.get("support")  # 各 tool は govern ラップ済み・監査フック装着済み
    try:
        result = await Runner.run(agent, input=text)
        print(f"[allow] input={text!r}\n        output: {result.final_output[:80]}")
    except Exception as exc:
        # Runner は tool 実行例外を SDK 例外にラップし得るため __cause__ も確認する。
        cause = exc.__cause__
        if isinstance(exc, PolicyViolationError) or isinstance(cause, PolicyViolationError):
            reason = cause or exc
            print(f"[deny]  input={text!r}\n        -> ポリシー違反として実行前に拒否: {reason}")
        else:
            raise


async def main() -> None:
    registry, builder = build_registry()

    # allowlist に載った lookup_order は通常どおり実行される。
    await _ask(registry, "注文 A123 の状況を教えて")

    # allowlist 未掲載の delete_order は実関数を実行せず拒否される。
    await _ask(registry, "注文 A123 を削除して")

    # 許可ツールでも引数が blocked_patterns に合致すれば拒否される
    # （LLM が引数へ転記した場合に deny になるデモで、結果は入力の誘導に依存する）。
    await _ask(registry, "注文 'DROP TABLE users' の状況を教えて")

    # 監査ログ: 既定 sink は builder が共有保持し、audit_sink プロパティで取得できる。
    sink = builder.audit_sink
    print("\n監査ログ（allow / deny の決定記録・tamper-evident ハッシュチェーン）:")
    for entry in sink.get_entries():
        print(f"  {entry.agent_id}  {entry.action}  {entry.decision}")
    print(f"チェーン検証: verify_chain() = {sink.verify_chain()}")


if __name__ == "__main__":
    asyncio.run(main())
