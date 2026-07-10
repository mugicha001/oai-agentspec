# realtime examples

Realtime 専用宣言ルート（`oai_agentspec.realtime`）の使用例。

- `basic_declaration.py`: 宣言 -> 登録 -> validate -> get で RealtimeAgent を構築するまで（実 API 不要）。
  実行: `uv run python examples/realtime/basic_declaration.py`
- `handoff_session.py`: triage <-> support の相互 handoff（循環）を宣言し RealtimeRunner で実行する最小例。
  実行時 Config は利用者が渡す（model_name / voice 等は `RealtimeRunner` 構築時、
  接続先はセッション開始 `run()` 時）。handoff の発生はモデルの判断に依存する。実 API が必要。
  他の examples と同様 Azure OpenAI を優先し（`.env` の `AZURE_OPENAI_ENDPOINT` /
  `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_REALTIME_DEPLOYMENT`）、未設定なら
  `OPENAI_API_KEY` で api.openai.com にフォールバックする。
  実行: `uv run python examples/realtime/handoff_session.py`
- `voice_chat.py`: マイク / スピーカーで音声会話する例（SDK 公式 realtime CLI デモの音声 I/O を
  踏襲し、エージェント構築を宣言ルートに差し替え）。barge-in（発話への割り込み）対応。
  sounddevice はライブラリ依存に含めないため `--with` で一時導入する。実 API が必要。Ctrl+C で終了。
  実行: `uv run --with sounddevice python examples/realtime/voice_chat.py`
- `_connection.py`: examples 共通の接続補助（Azure 優先・OpenAI フォールバック・
  認証情報の事前チェック・出力の認証情報スクラブ）。
