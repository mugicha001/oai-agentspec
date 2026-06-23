"""HITL（ツール実行承認）をサーバ + CLI で確認する例（クライアント・サーバ型）。

承認必須ツール（`needs_approval=True`・ここでは擬似的なファイル削除）を持つエージェントで
会話サーバを起動する。別ターミナルから付属 CLI（`oai-agentspec chat`）で接続し、削除を依頼
すると、エージェントがツールを呼んだ時点で CLI に「承認待ち」が提示される。approve すると
ツールが実行され会話が再開し、reject するとツールを実行せず会話が継続する。承認待ちのまま
会話を閉じても、セッションを復元すれば承認待ちから続きを再開できる（永続化時）。

依存（extra）:
    pip install "oai-agentspec[serve]"   # サーバ（fastapi, uvicorn）
    pip install "oai-agentspec[cli]"     # 接続 CLI（httpx, websockets, rich）

サーバ起動（このスクリプト。127.0.0.1:8000・認証なし・ローカル開発専用）:
    uv run python examples/conversation/05_hitl_serve.py

別ターミナルから CLI で接続し、削除を依頼して承認/却下を試す:
    oai-agentspec chat
    # 例: 「古いログ /var/log/old.log を削除して」と入力 -> 承認待ちが提示される

CLI からサーバを起動する場合（registry ファクトリを module:callable で指定）:
    oai-agentspec serve --registry examples.conversation.05_hitl_serve:build_registry

HTTP/WS を直接叩く場合の例:
    # 承認待ちの会話に対して call_id 単位で承認/却下する
    curl localhost:8000/conversations/<id>/approvals                       # 承認待ち取得
    curl -X POST localhost:8000/conversations/<id>/approvals \\
      -H 'content-type: application/json' \\
      -d '{"decisions":[{"call_id":"<call_id>","decision":"approve"}]}'
"""

from __future__ import annotations

import sys
from pathlib import Path

from oai_agentspec import AgentRegistry, AgentSpec, function_tool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402

# 実際に実行されたファイル削除を記録する（承認前は空のまま＝承認前非実行の確認用）。
DELETED: list[str] = []


@function_tool(needs_approval=True)
def delete_file(path: str) -> str:
    """指定パスのファイルを削除する（承認必須・本例では擬似的に記録するのみ）。

    Args:
        path: 削除対象のファイルパス。

    Returns:
        削除結果のメッセージ。
    """
    DELETED.append(path)
    return f"削除しました: {path}"


def build_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="ops",
            instructions=(
                "あなたは運用アシスタントです。ファイル削除を依頼されたら、必ず delete_file "
                "ツールを呼んで実行してください。自分で確認を求めず、ツールを使うこと。"
            ),
            model=azure_model(),
            tools=[delete_file],
        )
    )
    registry.validate()
    return registry


def main() -> None:
    # serve extra（fastapi/uvicorn）が必要。未導入なら案内する。
    try:
        from oai_agentspec.runtime.serve import start_server
    except ImportError:
        print('serve extra が必要です: pip install "oai-agentspec[serve]"', file=sys.stderr)
        raise SystemExit(1) from None

    print("HITL 会話サーバを http://127.0.0.1:8000 で起動します（Ctrl-C で停止）。")
    print("別ターミナルから:  oai-agentspec chat")
    print("例: 「古いログ /var/log/old.log を削除して」と入力すると承認待ちになります。")
    start_server(build_registry(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
