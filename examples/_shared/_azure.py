"""Azure OpenAI（Responses API）モデルを構築する共有ヘルパー（examples 用）。

oai-agentspec 本体は特定のモデルプロバイダに依存しない。各 example は
`AgentSpec.model` に Azure OpenAI の Responses API モデルを渡して実行する。

Responses API を使う。`AZURE_OPENAI_API_VERSION` の値で接続方式が変わる:
  - "preview"（既定）: Microsoft の v1 preview エンドポイント（/openai/v1/ パス）。
    `AsyncOpenAI(base_url=.../openai/v1/, default_query={"api-version": "preview"})`。
  - "2025-03-01-preview" 等の dated 版: legacy パス（`AsyncAzureOpenAI`）。

必要な環境変数（.env 等で設定）:
    AZURE_OPENAI_ENDPOINT      例: https://<resource>.openai.azure.com
    AZURE_OPENAI_API_KEY       Azure OpenAI の API キー
    AZURE_OPENAI_API_VERSION   既定 "preview"（v1 preview）。dated 版も可
    AZURE_OPENAI_DEPLOYMENT    デプロイ名（モデル名ではなくデプロイ名）
"""

from __future__ import annotations

import os
from pathlib import Path

from agents import OpenAIResponsesModel, set_tracing_disabled
from openai import AsyncAzureOpenAI, AsyncOpenAI


def _load_dotenv() -> None:
    """リポジトリ直下の .env を読み込む（examples 実行の利便。標準ライブラリのみ）。

    既に環境変数が設定されている場合は上書きしない。本ライブラリ本体は環境変数に
    依存せず、これは examples を `uv run python examples/...` で直接動かすための補助。
    """
    # examples/_shared/_azure.py -> リポジトリ直下（3 階層上）の .env
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def azure_api_version() -> str:
    """Azure OpenAI の API バージョンポリシーを解決する（examples 共通の単一ソース）。

    既定は "preview"（Microsoft v1 preview エンドポイント系統）。Responses クライアント
    （本モジュール）と Realtime WebSocket URL（examples/realtime/_connection.py）の両方が
    本関数を参照し、既定値・分岐条件のドリフトを防ぐ。

    Returns:
        環境変数 AZURE_OPENAI_API_VERSION の値（未設定なら "preview"）。
    """
    return os.environ.get("AZURE_OPENAI_API_VERSION", "preview")


def _azure_client() -> AsyncOpenAI:
    """Azure OpenAI 互換クライアントを生成する（preview は v1 エンドポイント）。"""
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    api_version = azure_api_version()

    if api_version == "preview":
        # api-version=preview は legacy /openai/ パスでは 404 になるため、
        # base_url を /openai/v1/ に切替えて Microsoft の v1 preview に送る。
        return AsyncOpenAI(
            base_url=f"{endpoint.rstrip('/')}/openai/v1/",
            api_key=api_key,
            default_query={"api-version": "preview"},
        )
    return AsyncAzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )


def azure_model() -> OpenAIResponsesModel:
    """環境変数から Azure OpenAI の Responses API モデルを構築する。

    Returns:
        AgentSpec.model に渡せる OpenAIResponsesModel。

    Raises:
        KeyError: 必須の環境変数が未設定の場合。
    """
    _load_dotenv()
    # トレーシングは OpenAI 本家へ送信されるため Azure 利用時は無効化する。
    set_tracing_disabled(True)
    return OpenAIResponsesModel(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        openai_client=_azure_client(),
    )


def azure_client() -> AsyncOpenAI:
    """compaction 等でクライアントを直接渡す用の Azure 互換クライアントを構築する。

    `azure_model()` がモデル実行用にクライアントを内包するのに対し、こちらは
    `SessionPolicy(compaction=CompactionConfig(client=...))` のようにクライアントを
    直接渡したい用途向け。dotenv 読み込み・トレーシング無効化も行う。

    Returns:
        Azure OpenAI の Responses API 互換 `AsyncOpenAI` クライアント。

    Raises:
        KeyError: 必須の環境変数が未設定の場合。
    """
    _load_dotenv()
    set_tracing_disabled(True)
    return _azure_client()


def azure_deployment() -> str:
    """compaction の model に渡すデプロイ名を環境変数から返す。

    Returns:
        `AZURE_OPENAI_DEPLOYMENT`（デプロイ名）。

    Raises:
        KeyError: 必須の環境変数が未設定の場合。
    """
    _load_dotenv()
    return os.environ["AZURE_OPENAI_DEPLOYMENT"]


def load_env() -> None:
    """リポジトリ直下の .env を環境変数へ読み込む（examples 実行の利便・公開窓口）。

    `azure_model()` 等も内部で同処理を行うが、Azure 以外の env（`LANGFUSE_*` 等）を読みたい
    example が、モデル構築前に明示的に呼べるよう公開する。既存の環境変数は上書きしない
    （本ライブラリ本体は環境変数に依存せず、これは examples を直接動かすための補助）。
    """
    _load_dotenv()
