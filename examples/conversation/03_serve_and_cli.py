"""会話サーバの起動と CLI クライアント接続の例（クライアント・サーバ型）。

会話 Helper は「ローカルサーバ + 接続クライアント」でも使える。本ファイルはサーバを
起動するスクリプトであり、別ターミナルから付属の CLI（または HTTP/WS）で接続する。

依存（extra）:
    pip install "oai-agentspec[serve]"   # サーバ（fastapi, uvicorn）
    pip install "oai-agentspec[cli]"     # 接続 CLI（httpx, websockets, rich）

サーバ起動（このスクリプト。127.0.0.1:8000・認証なし・ローカル開発専用）:
    uv run python examples/conversation/03_serve_and_cli.py

別ターミナルから CLI で接続（既定 http://localhost:8000）:
    oai-agentspec chat                       # 起動時にセッション選択画面を表示
    oai-agentspec chat --no-stream           # 完結応答（既定はストリーミング）

CLI はエントリ（登録順の先頭）エージェント起点で会話する。起動するとセッション選択
画面が出て、新規会話 / 過去 session の復元を選べる（会話画面では /back /quit /help）。

CLI からサーバを起動する場合（registry ファクトリを module:callable で指定。
任意 import のため信頼できるローカルコードのみ）:
    oai-agentspec serve --registry examples.conversation.03_serve_and_cli:build_registry
    oai-agentspec serve --registry ... --session-db ./my.db   # 永続化先を指定
    oai-agentspec serve --registry ... --ephemeral            # 揮発（永続化しない）
    oai-agentspec serve --registry ... --entry assistant      # 起点エージェントを明示

HTTP/WS を直接叩く場合の例:
    curl localhost:8000/agents
    curl -X POST localhost:8000/conversations -H 'content-type: application/json' -d '{}'
    curl localhost:8000/sessions
"""

from __future__ import annotations

import sys
from pathlib import Path

from oai_agentspec import AgentRegistry, AgentSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402


def build_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="assistant",
            instructions="あなたは簡潔に答える日本語アシスタントです。",
            model=azure_model(),
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

    print("会話サーバを http://127.0.0.1:8000 で起動します（Ctrl-C で停止）。")
    print("別ターミナルから:  oai-agentspec chat")
    start_server(build_registry(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
