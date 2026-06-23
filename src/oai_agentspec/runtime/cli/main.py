"""会話 CLI のエントリポイント（argparse・[project.scripts] から呼ばれる）。

`chat` サブコマンドで起動中の会話サーバへ接続して対話し、`serve` サブコマンドで会話
サーバを起動する。cli extra（httpx / websockets）・serve extra（fastapi / uvicorn）の
依存は各サブコマンド実行時に遅延 import し、未導入時は分かりやすい案内メッセージへ
変換する（本体 import で extra を強制しない）。SDK（`agents`）は import しない（NFR-1）。
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...registry import AgentRegistry

# cli extra（httpx / websockets）未導入時の案内。
_CLI_INSTALL_HINT = (
    "会話 CLI（chat）には httpx と websockets が必要です。"
    "次でインストールしてください: pip install 'oai-agentspec[cli]'"
)

# serve extra（fastapi / uvicorn）未導入時の案内。
_SERVE_INSTALL_HINT = (
    "会話サーバ（serve）には fastapi と uvicorn が必要です。"
    "次でインストールしてください: pip install 'oai-agentspec[serve]'"
)


def build_parser() -> argparse.ArgumentParser:
    """会話 CLI の argparse パーサを構築する。

    Returns:
        `chat` / `serve` サブコマンドを持つパーサ。
    """
    parser = argparse.ArgumentParser(
        prog="oai-agentspec",
        description="oai-agentspec 会話サーバの起動（serve）と接続（chat）を行う CLI。",
    )
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="起動中の会話サーバへ接続して対話する")
    chat.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="会話サーバのベース URL（既定 http://localhost:8000）",
    )
    chat.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="非ストリーミング（既定はストリーミング）",
    )
    chat.set_defaults(stream=True)

    serve = sub.add_parser(
        "serve",
        help="会話サーバ（FastAPI REST + WebSocket）を起動する",
    )
    serve.add_argument(
        "--registry",
        required=True,
        help=(
            "AgentRegistry を返す callable を 'module:callable' 形式で指定する。"
            " 指定モジュールを import して呼び出すため、信頼できるローカル開発専用。"
        ),
    )
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="バインド先ホスト（既定 127.0.0.1・localhost のみ）",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=8000,
        help="バインド先ポート（既定 8000）",
    )
    serve.add_argument(
        "--entry",
        default=None,
        help="エントリ（起点）エージェント名（未指定なら registry 登録順の先頭）",
    )
    serve.add_argument(
        "--session-db",
        default=None,
        help=(
            "会話履歴 SQLite db のファイルパス。未指定時は環境変数 XDG_DATA_HOME があれば"
            " $XDG_DATA_HOME/conversations.db、無ければ ./memory/conversations.db。"
            " 親ディレクトリは自動作成する。"
        ),
    )
    serve.add_argument(
        "--ephemeral",
        action="store_true",
        help="揮発モード（履歴を永続化せず in-memory のみ・--session-db を無視）",
    )
    return parser


def _resolve_registry(spec: str) -> AgentRegistry:
    """`module:callable` 形式の指定を import して AgentRegistry を得る。

    指定モジュールを import し callable を呼び出すため任意コード実行を伴う。信頼できる
    ローカル開発専用の機能であり、外部入力を渡してはならない。

    Args:
        spec: `module.path:callable` 形式の文字列。

    Returns:
        callable の戻り値（AgentRegistry 想定）。

    Raises:
        ValueError: 形式不正 / import 失敗 / 属性不在 / 非 callable の場合。
    """
    if ":" not in spec:
        raise ValueError(f"--registry は 'module:callable' 形式で指定してください: {spec!r}")
    module_path, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(f"--registry のモジュール import に失敗しました: {module_path!r}") from exc
    factory: Any = getattr(module, attr, None)
    if factory is None:
        raise ValueError(f"--registry の callable が見つかりません: {spec!r}")
    if not callable(factory):
        raise ValueError(f"--registry の指定対象が callable ではありません: {spec!r}")
    return factory()


def _run_chat(args: argparse.Namespace) -> int:
    """chat サブコマンドを実行する（cli extra 依存を遅延 import）。"""
    try:
        from .chat import run_chat
    except ImportError:
        print(_CLI_INSTALL_HINT, file=sys.stderr)
        return 1
    return asyncio.run(run_chat(base_url=args.base_url, stream=args.stream))


def _build_session_policy(session_db: str | None, ephemeral: bool) -> Any:
    """serve の永続化引数から `SessionPolicy` を組み立てる（CLI 境界で env を解決）。

    保存先の決定は次の優先順:
      1. `--session-db <path>`（明示フラグ）
      2. 環境変数 `XDG_DATA_HOME`（設定時はそのフォルダ直下 `$XDG_DATA_HOME/conversations.db`）
      3. 既定 `./memory/conversations.db`（プロジェクト直下の見えるフォルダ）

    env 参照はこの CLI 境界に閉じる（本体 `SessionPolicy` / `ConversationService` は環境変数
    に依存しない）。

    Args:
        session_db: 会話履歴 db のファイルパス（None で env / 既定にフォールバック）。
        ephemeral: True で揮発モード（永続化しない）。

    Returns:
        構築済み `SessionPolicy`。
    """
    import os
    from pathlib import Path

    from ..conversation import SessionPolicy

    persist = not ephemeral
    if session_db:
        path = Path(session_db).expanduser()
        return SessionPolicy(base_dir=path.parent, db_name=path.name, persist=persist)
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        # 指定フォルダの直下に置く（db 名は SessionPolicy の既定 conversations.db）。
        return SessionPolicy(base_dir=Path(xdg_data_home).expanduser(), persist=persist)
    return SessionPolicy(persist=persist)


def _run_serve(args: argparse.Namespace) -> int:
    """serve サブコマンドを実行する（serve extra 依存を遅延 import）。"""
    try:
        registry = _resolve_registry(args.registry)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        from ..serve.app import start_server
    except ImportError:
        print(_SERVE_INSTALL_HINT, file=sys.stderr)
        return 1
    policy = _build_session_policy(args.session_db, args.ephemeral)
    start_server(
        registry,
        host=args.host,
        port=args.port,
        session_policy=policy,
        entry_agent=args.entry,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI エントリポイント（`[project.scripts]` から呼ばれる）。

    Args:
        argv: コマンドライン引数（None で `sys.argv[1:]`）。

    Returns:
        プロセス終了コード。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "chat":
        return _run_chat(args)
    if args.command == "serve":
        return _run_serve(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - スクリプト直接実行用
    raise SystemExit(main())
