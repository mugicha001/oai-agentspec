"""起動時ウォームアップの共有ヘルパー（examples 用）。

プロセス起動後の初回 LLM 呼び出しは、TCP/TLS 接続確立・接続プール初期化・サービス側の
推論経路初期化が乗るため、warm 時より数百 ms から 1 秒超遅くなる。極小の推論を 1 回
先払いすることで、以降の計測（分類レイテンシ等）を warm 状態で行えるようにする。

接続プールは OpenAI クライアント（＝ model インスタンス）ごとに独立なので、**温めたい
処理と同じ model インスタンスを渡すこと**（`azure_model()` を呼び直すと別クライアントに
なり接続は温まらない）。

使い方:
    from _warmup import warmup

    model = azure_model()
    await warmup(model)          # 起動時に 1 回
    classifier = intent_classifier_from_model(model=model, ...)
"""

from __future__ import annotations

from typing import Any

from agents import Agent, ModelSettings, Runner


async def warmup(model: Any, model_settings: Any | None = None) -> None:
    """極小の推論を 1 回実行して接続と推論経路を温める。

    Args:
        model: agents.Model 相当。温めたい処理と同じインスタンスを渡す。
        model_settings: agents.ModelSettings 相当。None なら `max_tokens=16`
            （Responses API の下限近く）の最小設定を使う。reasoning 系モデルで
            思考トークンを止めたい場合は利用側の設定をそのまま渡してよい。
    """
    settings = model_settings if model_settings is not None else ModelSettings(max_tokens=16)
    agent = Agent(name="warmup", model=model, model_settings=settings)
    await Runner.run(agent, input="ping")
