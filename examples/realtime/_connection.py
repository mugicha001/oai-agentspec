"""realtime examples 共通の接続 Config 補助（Azure 優先・OpenAI フォールバック）。

接続先（url / 認証ヘッダー）は宣言（RealtimeAgentSpec）ではなく実行時 Config の責務。
本モジュールは examples 専用の補助であり、ライブラリ本体（env 非依存）には含めない。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_api_version  # noqa: E402


def build_model_config() -> dict[str, Any] | None:
    """接続先の RealtimeModelConfig を組み立てる（Azure 優先・OpenAI フォールバック）。

    Returns:
        Azure OpenAI 設定時は `url` + `api-key` ヘッダーを持つ dict。
        未設定時は None（SDK 既定 = `OPENAI_API_KEY` で api.openai.com へ接続）。
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_REALTIME_DEPLOYMENT")
    if not (endpoint and api_key and deployment):
        return None
    host = endpoint.replace("https://", "wss://").rstrip("/")
    # API バージョンポリシー（既定 "preview"）は _azure.py と共有の単一ソースで解決する
    api_version = azure_api_version()
    if api_version == "preview":
        # Microsoft v1 preview エンドポイント（_azure.py の Responses 接続と同じ系統。
        # api-version=preview クエリがないと 401 になる）
        url = f"{host}/openai/v1/realtime?api-version=preview&model={deployment}"
    else:
        url = f"{host}/openai/realtime?api-version={api_version}&deployment={deployment}"
    # headers 指定時は SDK が Authorization ヘッダーを自動付与しないため api-key を明示する
    return {"url": url, "headers": {"api-key": api_key}}


def require_credentials() -> None:
    """必要 env の事前チェック（不足時はスタックトレースではなく短い案内で終了する）。"""
    has_azure = all(
        os.environ.get(k)
        for k in (
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_REALTIME_DEPLOYMENT",
        )
    )
    if not has_azure and not os.environ.get("OPENAI_API_KEY"):
        print(
            "実行には Azure OpenAI（AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / "
            "AZURE_OPENAI_REALTIME_DEPLOYMENT）または OPENAI_API_KEY の設定が必要です"
            "（.env.example 参照）"
        )
        raise SystemExit(1)


def scrub(text: str) -> str:
    """出力テキストから認証情報（環境変数の実値）を伏せる。

    エラーオブジェクトの repr に接続オプション等が含まれる可能性に備え、
    ターミナルへ鍵をエコーしないための最小のスクラブ。
    """
    for name in ("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, f"<{name}>")
    return text
