"""LLM プロバイダ（Azure OpenAI / OpenAI 直接続）モデルを構築する共有ヘルパー（examples 用）。

oai-agentspec 本体は特定のモデルプロバイダに依存しない。各 example は
`AgentSpec.model` に Responses API モデルを渡して実行する。

プロバイダは環境変数 `EXAMPLES_LLM_PROVIDER` で切り替える（既定 "azure"）:
  - "azure": Azure OpenAI（従来どおり・既定）
  - "openai": OpenAI 直接続（`api.openai.com`）。課金を OpenAI 側へ分けたい場合に使う

歴史的経緯により関数名は `azure_model()` / `azure_client()` / `azure_deployment()` のままだが、
`EXAMPLES_LLM_PROVIDER=openai` のときは OpenAI 直接続のモデル / クライアント / モデル名を返す
（52 の既存 example の import を変えないための互換維持）。新規 example ではプロバイダ中立の
エイリアス `build_model()` / `build_client()` / `model_name()` を使うこと。

Azure は Responses API を使う。`AZURE_OPENAI_API_VERSION` の値で接続方式が変わる:
  - "preview"（既定）: Microsoft の v1 preview エンドポイント（/openai/v1/ パス）。
    `AsyncOpenAI(base_url=.../openai/v1/, default_query={"api-version": "preview"})`。
  - "2025-03-01-preview" 等の dated 版: legacy パス（`AsyncAzureOpenAI`）。

必要な環境変数（.env 等で設定）:
    EXAMPLES_LLM_PROVIDER      "azure"（既定）| "openai"
    [azure]
    AZURE_OPENAI_ENDPOINT      例: https://<resource>.openai.azure.com
    AZURE_OPENAI_API_KEY       Azure OpenAI の API キー
    AZURE_OPENAI_API_VERSION   既定 "preview"（v1 preview）。dated 版も可
    AZURE_OPENAI_DEPLOYMENT    デプロイ名（モデル名ではなくデプロイ名）
    [openai]
    OPENAI_API_KEY             OpenAI（または互換ゲートウェイ）の API キー
    OPENAI_MODEL               モデル名（例: gpt-5.4-mini）
    OPENAI_BASE_URL            任意（既定 https://api.openai.com/v1）。openai SDK 標準の
                               環境変数で、プロキシ / 互換ゲートウェイ利用時のみ設定する
    OPENAI_API_STYLE           任意。"responses"（既定・OpenAI 本家）| "chat_completions"
                               （litellm 等の Responses API 非対応ゲートウェイ向け。
                               `OpenAIChatCompletionsModel` でモデルを組む）
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


def _provider() -> str:
    """実行プロバイダを解決する（`EXAMPLES_LLM_PROVIDER`・既定 "azure"）。

    Returns:
        "azure" または "openai"。

    Raises:
        ValueError: 未知の値が設定されている場合。
    """
    provider = os.environ.get("EXAMPLES_LLM_PROVIDER", "azure").strip().lower()
    if provider not in ("azure", "openai"):
        raise ValueError(
            f"EXAMPLES_LLM_PROVIDER={provider!r} は未対応です（受理値: 'azure' | 'openai'）"
        )
    return provider


def _openai_api_style() -> str:
    """OpenAI 直接続時の API スタイルを解決する（`OPENAI_API_STYLE`・既定 "responses"）。

    Returns:
        "responses" または "chat_completions"。

    Raises:
        ValueError: 未知の値が設定されている場合。
    """
    style = os.environ.get("OPENAI_API_STYLE", "responses").strip().lower()
    if style not in ("responses", "chat_completions"):
        raise ValueError(
            f"OPENAI_API_STYLE={style!r} は未対応です（受理値: 'responses' | 'chat_completions'）"
        )
    return style


def _openai_client() -> AsyncOpenAI:
    """OpenAI 直接続クライアントを生成する（`OPENAI_API_KEY` 必須）。"""
    return AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


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
    """環境変数から Responses API モデルを構築する（プロバイダは env 切替）。

    `EXAMPLES_LLM_PROVIDER=openai` のときは OpenAI 直接続モデルを返す（関数名は互換維持）。

    Returns:
        AgentSpec.model に渡せる OpenAIResponsesModel。

    Raises:
        KeyError: 必須の環境変数が未設定の場合。
        ValueError: `EXAMPLES_LLM_PROVIDER` が未知の値の場合。
    """
    _load_dotenv()
    # トレーシング（OpenAI 本家へのアップロード）はプロバイダによらず無効化する
    # （Azure ではキー不一致で失敗し、OpenAI 直接続でも example 実行の副作用送信を避ける）。
    set_tracing_disabled(True)
    if _provider() == "openai":
        from agents import OpenAIChatCompletionsModel

        if _openai_api_style() == "chat_completions":
            # litellm 等の Responses API 非対応ゲートウェイ向け（/responses が 404 になる）。
            return OpenAIChatCompletionsModel(
                model=os.environ["OPENAI_MODEL"],
                openai_client=_openai_client(),
            )
        return OpenAIResponsesModel(
            model=os.environ["OPENAI_MODEL"],
            openai_client=_openai_client(),
        )
    return OpenAIResponsesModel(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        openai_client=_azure_client(),
    )


def azure_client() -> AsyncOpenAI:
    """compaction 等でクライアントを直接渡す用の Azure 互換クライアントを構築する。

    `azure_model()` がモデル実行用にクライアントを内包するのに対し、こちらは
    `SessionPolicy(compaction=CompactionConfig(client=...))` のようにクライアントを
    直接渡したい用途向け。dotenv 読み込み・トレーシング無効化も行う。

    `EXAMPLES_LLM_PROVIDER=openai` のときは OpenAI 直接続クライアントを返す（関数名は互換維持）。

    Returns:
        Responses API 互換 `AsyncOpenAI` クライアント。

    Raises:
        KeyError: 必須の環境変数が未設定の場合。
        ValueError: `EXAMPLES_LLM_PROVIDER` が未知の値の場合。
    """
    _load_dotenv()
    set_tracing_disabled(True)
    if _provider() == "openai":
        return _openai_client()
    return _azure_client()


def azure_deployment() -> str:
    """compaction の model に渡すデプロイ名 / モデル名を環境変数から返す。

    `EXAMPLES_LLM_PROVIDER=openai` のときは `OPENAI_MODEL` を返す（関数名は互換維持）。

    Returns:
        `AZURE_OPENAI_DEPLOYMENT`（デプロイ名）または `OPENAI_MODEL`（モデル名）。

    Raises:
        KeyError: 必須の環境変数が未設定の場合。
        ValueError: `EXAMPLES_LLM_PROVIDER` が未知の値の場合。
    """
    _load_dotenv()
    if _provider() == "openai":
        return os.environ["OPENAI_MODEL"]
    return os.environ["AZURE_OPENAI_DEPLOYMENT"]


def load_env() -> None:
    """リポジトリ直下の .env を環境変数へ読み込む（examples 実行の利便・公開窓口）。

    `azure_model()` 等も内部で同処理を行うが、Azure 以外の env（`LANGFUSE_*` 等）を読みたい
    example が、モデル構築前に明示的に呼べるよう公開する。既存の環境変数は上書きしない
    （本ライブラリ本体は環境変数に依存せず、これは examples を直接動かすための補助）。
    """
    _load_dotenv()


# プロバイダ中立のエイリアス（新規 example はこちらを使う。実体は上記の互換関数と同一）。
build_model = azure_model
build_client = azure_client
model_name = azure_deployment


def api_style() -> str:
    """`optimize(apo_api=...)` へ渡す API スタイルをプロバイダ設定から解決する。

    azure プロバイダは Responses 前提（`azure_model()` が `OpenAIResponsesModel` を返す）の
    ため "responses" 固定。openai プロバイダは `OPENAI_API_STYLE`（既定 "responses"）に従う。
    rollout のモデルクラス選択と gradient / apply-edit の API 選択が同一 env で揃う。

    Returns:
        "responses" または "chat_completions"。
    """
    _load_dotenv()
    if _provider() == "openai":
        return _openai_api_style()
    return "responses"
