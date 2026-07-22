# 会話 Helper（ConversationService / Session / HITL / compaction）

## 何を解決するか

`ConversationService` は registry に登録済みのエージェントとマルチターン会話する上位ヘルパです。in-process から使えるほか、`[serve]` + `[cli]` のクライアント・サーバ型でも同じ API を共有します。履歴は SDK `Session`（SQLite）に委ね、session_id 連動で永続化・途中再開できます。

HITL（`function_tool(needs_approval=True)` / `ToolSpec(needs_approval=True)`）と compaction（履歴圧縮）を同一窓口で扱います。承認待ちは `SendResult(status=SendStatus.PENDING)` で返り、`resolve_approvals` で call_id 単位に approve / reject します。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| `send` | 完結応答（非ストリーミング） | 単発問い合わせ |
| `stream` | 逐次イベント（`StreamDelta` / `StreamDone` / `StreamError` / `ApprovalRequired`） | UI へ逐次表示 |
| compaction off（既定） | 履歴そのまま蓄積 | 短い会話・検証 |
| compaction on（`CompactionConfig(enabled=True, client=..., model=...)`） | Responses `compact` で要約置換 | 長い会話・トークン節約 |
| in-process | 同プロセスで完結 | dev・組込 |
| クライアント・サーバ型 | `serve` + `cli` を別プロセス | dev のマルチクライアント |

## 使い方

- import: `from oai_agentspec.runtime.conversation import (ConversationService, SessionPolicy, CompactionConfig, SendResult, SendStatus, StreamDelta, StreamDone, StreamError, StreamEvent, PendingApproval, ApprovalDecision, ApprovalRequired, SessionInfo, ConversationError, ConversationErrorCode)`
- extras: `pip install oai-agentspec[conversation]`（追加外部依存なし）
- 依存 env: なし（CLI 境界を除く）

```python
from oai_agentspec.runtime.conversation import (
    ApprovalDecision, ConversationService, SendStatus, StreamDelta,
)

chat = ConversationService(registry)
cid = await chat.create_conversation()
r = await chat.send("triage", "請求書ください", conversation_id=cid)
async for ev in chat.stream("triage", "続けて", conversation_id=cid):
    if isinstance(ev, StreamDelta):
        print(ev.text, end="")
```

HITL:

```python
r = await chat.send("triage", "ユーザー削除して", conversation_id=cid)
if r.status == SendStatus.PENDING:
    await chat.resolve_approvals(
        cid,
        [ApprovalDecision(call_id=r.pending[0].call_id, approve=True)],
    )
```

compaction:

```python
from openai import AsyncOpenAI
from oai_agentspec.runtime.conversation import CompactionConfig, SessionPolicy
policy = SessionPolicy(compaction=CompactionConfig(
    enabled=True, client=AsyncOpenAI(), model="gpt-4.1",
))
chat = ConversationService(registry, session_policy=policy)
```

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `ConversationService.__init__`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `registry` | `AgentRegistry` | 必須 | 名前解決に使う |
| `session_policy` | `SessionPolicy \| None` | `None` | session 生成方針 |
| `entry_agent` | `str \| None` | `None` | エントリ起点名。None で `registry.entry_name` |

### 主要メソッド（引数抜粋・15 個超えるため主要 6 個）

- `agents() -> list[str]`
- `entry_agent() -> str | None`
- `create_conversation(*, conversation_id=None, session_id=None) -> str`
- `send(agent_name, text, *, conversation_id) -> SendResult` — `agent_name` None でエントリ起点
- `stream(agent_name, text, *, conversation_id) -> AsyncIterator[StreamEvent | ApprovalRequired]`
- `resolve_approvals(conversation_id, decisions) -> SendResult` / `stream_resolve(...)` / `pending_approvals(conversation_id) -> list[PendingApproval]`
- `list_sessions() -> list[SessionInfo]` / `session_history(session_id, *, limit=10)`（既定 10 件・上書きは `int` を直接指定）

### `SessionPolicy`（frozen）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `base_dir` | `Path` | `Path("memory")` | ファイル永続化の基底 |
| `db_name` | `str` | `"conversations.db"` | 永続化 db ファイル名 |
| `persist` | `bool` | `True` | False で常に in-memory |
| `compaction` | `CompactionConfig \| None` | `None` | 圧縮設定 |

### `CompactionConfig`（frozen）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `enabled` | `bool` | `False` | 有効化フラグ |
| `client` | `AsyncOpenAI \| None` | `None` | `enabled=True` で必須 |
| `model` | `str \| None` | `None` | 圧縮モデル名 |
| `options` | `dict[str, Any]` | `{}` | `OpenAIResponsesCompactionSession` へ素通し |

### `SendResult`（frozen）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `status` | `SendStatus` | 必須 | `FINAL` or `PENDING`（StrEnum） |
| `output` | `str \| None` | `None` | FINAL 時のテキスト |
| `pending` | `list[PendingApproval]` | `[]` | PENDING 時の承認待ち |

### `ApprovalDecision`（frozen）

`call_id: str` / `approve: bool` / `rejection_message: str | None = None`。

### `PendingApproval`（frozen）

`tool_name: str` / `call_id: str`。

### `StreamDelta` / `StreamDone` / `StreamError` / `ApprovalRequired`

いずれも 1〜2 引数 frozen（詳細は docstring 参照）。

### `SendStatus`（StrEnum）: `FINAL` / `PENDING`

### `ConversationErrorCode`（StrEnum）

`UNKNOWN_AGENT` / `UNKNOWN_CONVERSATION` / `CONVERSATION_ALREADY_EXISTS` / `MODEL_NOT_CONFIGURED` / `EXECUTION_ERROR` / `UNKNOWN_APPROVAL` / `APPROVAL_ALREADY_RESOLVED` / `NO_PENDING_APPROVAL`。

## 判断軸

- UI へ逐次表示するなら **`stream`**、backend で完結処理なら **`send`**
- 会話が長期化するなら **compaction on**。ただし `client` / `model` を渡しただけでは有効化されない（`enabled=True` を明示）
- HITL 対象 tool は必ず `resolve_approvals` を呼ぶフローにする（承認なしでは会話が止まる）

## 落とし穴

- compaction は OpenAI Responses API 専用。`client` は `AsyncOpenAI` / `AsyncAzureOpenAI` で Responses を叩ければ動くが、`model` は OpenAI 形式名
- `enabled=True` かつ `client` 欠落は構築時 `ValueError`（暗黙有効化しない）
- 未解決の承認待ちがある間は `send` / `stream` が新ターンを開始しない（P1・安全性）

## 参照

- 詳細設計: `docs/architecture.md`（会話 Helper 節）
- 具体例: `examples/conversation/01_inprocess.py` 〜 `06_compaction.py`

## 次

[serve_and_cli.md](./serve_and_cli.md) — serve と cli の使い分け
