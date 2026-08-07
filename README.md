# oai-agentspec

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Built on openai-agents](https://img.shields.io/badge/built%20on-openai--agents-412991.svg)](https://github.com/openai/openai-agents-python)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#プロジェクトステータス)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://docs.astral.sh/ruff/)

> **Disclaimer**: `oai-agentspec` is an **unofficial, community-maintained** library built on top of [openai-agents](https://github.com/openai/openai-agents-python). **Not affiliated with, endorsed by, or sponsored by OpenAI.** OpenAI, the OpenAI logo, GPT, and ChatGPT are trademarks of OpenAI, Inc.
>
> 本ライブラリは [openai-agents](https://github.com/openai/openai-agents-python) を基盤としたコミュニティの非公式ライブラリで、**OpenAI 社による公式・推奨・スポンサー関係はありません**。OpenAI / GPT / ChatGPT 等は OpenAI, Inc. の商標です。

**OpenAI Agents SDK を、より宣言的に・ミスを防ぎながら扱うための薄いラッパーライブラリ。**

`oai-agentspec` は [openai-agents](https://github.com/openai/openai-agents-python) の `Agent` を置き換えません。
`Agent` と同じことができることを前提に、**プロンプト合成・名前ベースのハンドオフ/サブエージェント・遷移グラフ**を
宣言的にまとめ、実行前にタイポや未解決参照を検出できるようにします。SDK を知っている人ほど馴染みやすく、
かつ間違えにくい構成を目指しています。

---

## 目次

- [特徴](#特徴)
- [インストール](#インストール)
- [クイックスタート](#クイックスタート)
- [コアコンセプト](#コアコンセプト)（AgentSpec / Registry / プロンプト合成 / ハンドオフ / Realtime / ワークフロー / 会話 Helper / HITL / compaction）
- [プロンプトレイアウト](#プロンプトレイアウト)
- [サンプル](#サンプル)
- [開発](#開発)
- [プロジェクトステータス](#プロジェクトステータス)
- [ライセンス](#ライセンス)

## 特徴

| カテゴリ | できること |
|---|---|
| **Agent 宣言・編集** | `AgentSpec`（`Agent` の薄いラッパー）/ `AgentRegistry` で生成・`update`・`unregister`・差し替え / プロンプト合成（`base + parts + agent`・利用側 root）/ `RunContextWrapper` 経由の動的 instructions / サブエージェント（`sub_agents` で agent as tool） |
| **ハンドオフ** | 名前ベース宣言 + `validate()` で実行前タイポ検出 / 型付き設定（`on_handoff` / `input_type` / `input_filter` / `is_enabled`）/ 動的転送（`dynamic_edge`）/ 循環解決（`A⇄B`）/ `mermaid()` 可視化 |
| **Realtime（音声）** | `RealtimeAgentSpec` / 専用 `RealtimeAgentRegistry` による専用宣言ルート / 非対応フィールドの型レベル排除 / 名前ベース handoff の遅延構築・循環解決 / グラフ DSL（`RealtimeHandoffGraph`・`mermaid()` 可視化）/ `oai_agentspec.realtime` 窓口 |
| **ワークフロー（実験的）** | `WorkflowGraph` でノード（AGENT/FUNCTION）+ エッジ（通常 / 条件 / fan-in）を宣言、順次 / 並列 / 条件分岐 / 合流 / ループを表現 / build-time `validate()` / SDK tracing 自動配線（`workflow.*` span + AGENT 内側 `Runner.run` の親子接続・`set_tracing_disabled(True)` 時オーバーヘッド 0） |
| **会話 Helper** | `ConversationService`（in-process または `[serve]` + `[cli]` のクライアント・サーバ型）/ SDK `Session` で永続化・途中再開 / HITL 承認（`function_tool(needs_approval=True)` を call_id 単位で approve / reject）/ compaction（`CompactionConfig.enabled=True` で履歴圧縮を明示有効化） |
| **LLMOps（extras）** | `[llmops]` で観点別採点 + 統合 verdict（DeepEval ベース・任意で `[llmops-langfuse]` で Langfuse 観測）/ `[lightning]` で `AgentSpec` / `HandoffGraph` / `WorkflowGraph` のプロンプトを Agent Lightning へ委譲して自動改善（textual gradient + beam search） |
| **意図予測（extras）** | `[intent]` で発話 / 会話履歴からの意図分類基盤（`runtime/intent`）/ 信頼度 5 段階の候補列（`IntentPrediction`）/ `IntentPolicy` で意図集合・返却制約を宣言 / Protocol DI で全体・内部段を差し替え / `intent_classifier_from_model` / `intent_classifier_from_generator`（自作 generator 用）の 1 行ヘルパ |
| **Tool Registry** | `ToolRegistry` + `ToolSpec` で Tool メタデータ（`enabled` / `needs_approval` / `timeout` / `failure_error_function` / `name_override` / `description_override` / `strict_mode` / `extra`）を宣言的に一元管理 / 属性アクセス（`registry.<name>`）で `agents.function_tool()` を遅延構築・キャッシュ / `enabled` は closure で動的トグル（再構築なし） |
| **Resilience（extras）** | `[resilience]` で Model 呼び出し retry と run 全体の予算超過制御を宣言 / `ModelRetryPolicy` はセマンティックフラグ既定 True で SDK `ModelRetrySettings` の silent no-op を排除 / `RunBudgetPolicy` は `on_llm_end` ターン境界で累積時間 / トークン判定し `RunBudgetExceeded` を送出（SDK `error_handlers` を素通しで伝播）/ `FailsafePolicy` + `failsafe_call` で Runner の外へ漏れた例外を宣言 1 回で着地値（`FailsafeResult`）へ丸める（未宣言例外は素通し・`Exception` / `ExceptionGroup` 等 7 種は build-time `ValueError`）/ `last_agent` の 2 段解決（`FailsafeHandler.last_agent` -> `FailsafePolicy.fallback_last_agent`・`RUNNING_AGENT` を置いた段のみ opt-in で例外から実行中エージェントを解決）/ `FailsafeResult.from_exception` で `failsafe_call` 外側の except からも同じ結果型へ手動着地 / SDK 生型 10 種を窓口経由で再エクスポート |

詳細・サンプルは [コアコンセプト](#コアコンセプト) / [サンプル](#サンプル) を参照。

> 全機能の使い方ガイド（判断軸 + 最小コード + examples 誘導・トピック別）: [docs/usage/](docs/usage/index.md)

## インストール

現在 Alpha 段階のため PyPI には未公開です。GitHub から直接インストールしてください。

```bash
# 本体
uv add "oai-agentspec @ git+https://github.com/mugicha001/oai-agentspec.git"

# extras 付き
uv add "oai-agentspec[serve,cli] @ git+https://github.com/mugicha001/oai-agentspec.git"
uv add "oai-agentspec[llmops] @ git+https://github.com/mugicha001/oai-agentspec.git"
uv add "oai-agentspec[llmops,llmops-langfuse] @ git+https://github.com/mugicha001/oai-agentspec.git"
uv add "oai-agentspec[lightning] @ git+https://github.com/mugicha001/oai-agentspec.git"
uv add "oai-agentspec[intent] @ git+https://github.com/mugicha001/oai-agentspec.git"
uv add "oai-agentspec[resilience] @ git+https://github.com/mugicha001/oai-agentspec.git"

# ローカルクローンで開発する場合
git clone https://github.com/mugicha001/oai-agentspec.git
cd oai-agentspec
uv sync --all-extras
```

要件: Python 3.12+ / `openai-agents>=0.17.4`

extra は実行寄り層でのみ必要。コアの宣言 API（`AgentSpec` / `AgentRegistry` / `HandoffGraph` /
`WorkflowGraph` / `ConversationService` の in-process 利用）は extra なしで動く。`serve` = FastAPI
サーバ入口、`cli` = 接続 CLI（`oai-agentspec chat`）、`llmops` = 評価採点コア（DeepEval）、
`llmops-langfuse` = Langfuse 観測（`llmops` 前提・任意）、`lightning` = Agent Lightning APO
（プロンプト最適化）、`intent` = 意図予測（pydantic のみ）、`resilience` = 宣言的 Model retry と
run 予算（追加外部依存なし）。Tool Registry（`ToolRegistry` / `ToolSpec`）はコアのため extra 不要。

## クイックスタート

`triage` が `billing` / `support` にハンドオフする最小構成。プロンプトは `prompts/` 配下に置く（[後述](#プロンプトレイアウト)）。

```python
import asyncio
from pathlib import Path

from agents import Runner
from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph, PromptLayout, PromptStore

# プロンプトのルートとディレクトリ構成を明示する
store = PromptStore(Path("prompts"), PromptLayout(base="base", parts="parts", agents="agents"))
registry = AgentRegistry()

# AgentSpec は Agent のラッパー。instructions に合成済みプロンプトを渡す
for name in ("triage", "billing", "support"):
    registry.register(
        AgentSpec(
            name=name,
            instructions=store.compose(agent=name, base="main", parts=["style", "safety"]),
            # model=... 省略時は SDK のデフォルト。Azure 例は examples/_shared/_azure.py を参照
        )
    )

# 遷移グラフを宣言してレジストリに適用
graph = HandoffGraph(entry="triage")
graph.edge("triage", "billing", description="請求関連")
graph.edge("triage", "support", description="技術問い合わせ")
graph.apply(registry)

registry.validate()  # 未解決のハンドオフ参照を run 前に検出
print(graph.mermaid())  # description はエッジラベルとしても使われる


async def main() -> None:
    entry = graph.entry_agent(registry)
    result = await Runner.run(entry, input="先月の請求書のPDFが欲しいです")
    print(f"最終回答エージェント: {result.last_agent.name}")
    print(result.final_output)


asyncio.run(main())
```

## コアコンセプト

### AgentSpec

`Agent` のフィールド（`name` / `instructions` / `prompt` / `tools` / `model` / `model_settings` /
`hooks` など）を写した宣言的な仕様。加えて遷移グラフ連携のための `handoffs` / `sub_agents` を
**名前のリスト**で持つ。

専用フィールドを用意していない残りの `agents.Agent` kwarg（`output_type` / `input_guardrails` /
`output_guardrails` / `tool_use_behavior` / `handoff_description` など）は `extra` で素通しできる。

```python
AgentSpec(
    name="triage",
    instructions=...,
    extra={
        "output_type": TriageResult,
        "input_guardrails": [my_guardrail],
        "handoff_description": "振り分け担当",  # この Agent がハンドオフ先になる時の既定説明
    },
)
```

`extra` は構築時に検証され、ミスを早期に弾く。

- **衝突ガード** — `instructions` 等の専用フィールドと同名キーを入れると `ValueError`（二重指定の防止）
- **未知キーガード** — `agents.Agent` に存在しないキー（例 `output_typ` のタイポ）は黙殺せず `ValueError`

### AgentRegistry

`AgentSpec` を登録し、依存（`handoffs` ∪ `sub_agents` ∪ 動的ハンドオフ候補）を解決して
`Agent` を構築する。

```python
registry.register(spec)            # 仕様を登録
agent = registry.get("triage")     # 依存解決して Agent を構築（循環も解決）
registry.update(new_spec)          # 構築済みを再構成（依存先も無効化）
registry.unregister("billing")     # 削除（依存元も無効化）
registry.validate()                # 全 spec の参照健全性をまとめて検証
```

ハンドオフの編集は `HandoffGraph`（後述）で宣言し、`graph.apply(registry)` で反映する。
実行時にトポロジを変えたいときはグラフを編集して再 `apply` する（`apply` は当該 src を
replace で上書きする）。

> SDK 隔離点は `_adapters`（`from agents import ...` の単一窓口）であり、openai-agents が
> 更新されても直すのはここだけ。生成処理そのものを差し替える DI 拡張点 `AgentBuilder`
> （`oai_agentspec.protocols`）も用意しているが、テスト/上級用途向けでありトップレベル
> 公開 API には含めない。

### PromptStore / プロンプト合成

`PromptStore.compose` は `base -> parts -> agent` の順でプロンプトを連結する。`layout` で順序を
上書きでき、`vars` で変数を埋め込む。

```python
# デフォルト順（base -> parts -> agent）
store.compose(agent="triage", base="main", parts=["style", "safety"], vars={"company": "AgentSpec Inc."})

# 順序を明示的に上書き（agent -> base -> part:safety）
store.compose(layout=["agent:triage", "base:main", "part:safety"])
```

### 動的 instructions（RunContextWrapper）

`vars` に `RunContextWrapper -> dict` の callable を渡すと、run ごとに context から値を取り出して
instructions をレンダリングする。

```python
def extract_vars(context: RunContextWrapper[SupportContext]) -> dict[str, str]:
    ctx = context.context
    return {"tier": "VIP" if ctx.plan == "premium" else "標準", "user_name": ctx.user_name}

registry.register(AgentSpec(name="concierge", instructions=store.compose(agent="concierge", vars=extract_vars)))
```

### サブエージェント（agent as tool）

`sub_agents` に名前を渡すと、対象エージェントが `as_tool` でツールとして注入される
（ハンドオフと異なり制御がメインへ戻る）。

```python
registry.register(AgentSpec(name="orchestrator", sub_agents=["researcher", "writer"], instructions=...))
```

### ハンドオフ（HandoffGraph）

ハンドオフは `HandoffGraph` で宣言する。SDK `handoff()` の主要引数を**型付きフィールド**で
受け取り（`AgentSpec.extra` と同じ「専用フィールド + `options` で素通し」思想）、`apply()` で
registry に反映する。

```python
from dataclasses import dataclass
from agents import RunContextWrapper
from agents.extensions import handoff_filters

@dataclass
class EscalationInput:
    reason: str
    priority: int

async def on_escalate(ctx: RunContextWrapper, data: EscalationInput) -> None:
    ctx.context.audit.append(f"escalate: {data.reason}")

graph = HandoffGraph(entry="triage")
graph.edge("triage", "billing", description="請求関連")          # 最小
graph.edge(
    "triage", "support",
    description="技術エスカレーション",        # = tool_description_override
    on_handoff=on_escalate,                   # 転送時に発火
    input_type=EscalationInput,               # 転送時に LLM が構造化入力を埋める
    input_filter=handoff_filters.remove_all_tools,  # 次エージェントへ渡す履歴を変換
    is_enabled=lambda ctx, agent: ctx.context.allow_escalation,  # 動的有効化
)
graph.apply(registry)
```

`description` はハンドオフ tool（`transfer_to_<dst>` 等）の説明で、LLM の選択材料になり
`mermaid()` のラベルにもなる。ターゲット**エージェント側**の既定説明 `Agent.handoff_description`
とは別物（後者は `AgentSpec(..., extra={"handoff_description": "..."})`）。

### 動的ハンドオフ（dynamic_edge）

固定 1 ターゲットでなく、**転送先を実行時に選ぶ**ハンドオフ。resolver が候補から転送先名を
返し、ライブラリが SDK の `Handoff.on_invoke_handoff` を生成する。候補は `validate()` /
`mermaid()` の対象になり、resolver の戻り名は候補内に強制される。

```python
def route(ctx, input_json) -> str:
    return "billing" if needs_billing(ctx) else "support"   # 候補名を返す

graph.dynamic_edge(
    "triage", ["billing", "support"], route, tool_name="route", description="動的に担当を決定",
)
```

### Realtime エージェント（専用宣言ルート）

音声（Realtime）エージェントは `agents.realtime.RealtimeAgent` を対象とする専用宣言ルートで扱う。
`RealtimeAgentSpec` は RealtimeAgent が対応するフィールドのみを持ち、非対応フィールド（`model` /
`model_settings` / `input_guardrails` 等の実行時 Config）を型レベルで排除する。専用の
`RealtimeAgentRegistry` が名前ベース handoff を遅延構築（循環も解決）する。シンボルはコア
`__all__` に載せず `oai_agentspec.realtime` 窓口から取得する。

```python
from agents.realtime import RealtimeRunner
from oai_agentspec.realtime import RealtimeAgentRegistry, RealtimeAgentSpec

registry = RealtimeAgentRegistry()
registry.register(RealtimeAgentSpec(
    name="triage",
    instructions="受付担当。技術的な問い合わせはサポート担当へ引き継ぐ。",
    handoff_description="最初の受付・振り分け担当。",
    handoffs=["support"],  # 名前ベース handoff（registry が遅延構築で結線）
))
registry.register(RealtimeAgentSpec(name="support", instructions="技術サポート担当。"))
registry.validate()  # 未解決の handoff 参照を run 前に検出

entry = registry.get("triage")
# model_settings（model_name / voice / modalities 等）は宣言側が持たない実行時 Config。
# セッション開始時に利用者が RealtimeRunner へ渡す。
runner = RealtimeRunner(entry, config={"model_settings": {"model_name": "gpt-4o-realtime-preview"}})
```

ハンドオフのトポロジは `RealtimeHandoffGraph` によるノード・エッジ宣言でも構築できる
（`spec.handoffs` 直接宣言と同一の結線・`mermaid()` で可視化可能）:

```python
from oai_agentspec.realtime import RealtimeHandoffGraph

graph = RealtimeHandoffGraph(entry="triage")
graph.edge("triage", "support", tool_description="技術的な問い合わせを引き継ぐ")
graph.edge("support", "triage")  # 相互参照（循環）も可
graph.apply(specs)               # 検証付きで spec 群へ一括反映（build 前に行う）
print(graph.mermaid())           # flowchart TD ...
```

音声 I/O 込みの実行例は `examples/realtime/`（`handoff_session.py` / `voice_chat.py`）を参照。

### ワークフロー（WorkflowGraph）

> 実験的機能（experimental）。インターフェース・挙動は今後変わる可能性がある。

LangGraph / Microsoft Agent Framework に倣い、**ノードとエッジを明示宣言**してワークフローを
組む。ノードは AGENT（registry のエージェント実行）と FUNCTION（`(msg, ctx) -> 出力` の関数）の
2 種。エッジは通常（`add_edge`。同一ノードから複数張ると並列）/ 条件（`add_conditional_edges`）/
fan-in 合流（`add_fan_in_edge`）。`START` / `END` 番兵で入口・終端を示す。

```python
wf = WorkflowGraph("pipeline")
wf.add_agent_node("plan", agent="planner")
wf.add_function_node("format", fn=lambda msg, ctx: f"<{msg}>")
wf.add_edge(START, "plan")
wf.add_edge("plan", "format")
wf.add_edge("format", END)
wf.validate(registry)

# ワークフローを 1 つの Agent として登録し Runner.run で実行する
registry.register(wf.as_agent_spec("pipeline_agent", registry=registry))
result = await Runner.run(registry.get("pipeline_agent"), input="...")
```

実行口は SDK の `Runner.run` 一本（公開の実行 API は持たない）。ワークフローは Agent
（`as_agent_spec`・経路C）または tool ファサード（`as_facade_spec`・経路A/D）として消費する。
内部ノードで外側 context を使いたい場合は経路A/D（`as_facade_spec`）、決定論を保ったまま context
透過したい場合は `as_facade_spec(mode=FacadeMode.DETERMINISTIC)`（経路D・実 LLM 0 回）。詳細は
`docs/architecture.md` のワークフロー節を参照。

ワークフロー実行は SDK tracing に自動配線される。`workflow.run.<graph_name>` を親 span として
ノード / 条件分岐 / fan-out / fan-in が子 span として記録され、AGENT ノード内側の `Runner.run` は
外側 trace 配下に親子接続される。属性は OpenTelemetry 風 namespace `workflow.<key>`（`graph_name` /
`node_kind` / `node_name`）。tracing 無効時（`set_tracing_disabled(True)`）はオーバーヘッド 0。
詳細は `docs/architecture.md` の「ワークフロー tracing」節を参照。

### 会話 Helper（ConversationService / serve / CLI）

`ConversationService` は registry に登録済みのエージェントとマルチターン会話する上位ヘルパ。
Python から直接（in-process）使えるほか、ローカルサーバ（`oai-agentspec[serve]`）+ 接続 CLI
（`oai-agentspec[cli]`）のクライアント・サーバ型でも使える。

```python
from oai_agentspec.runtime.conversation import ConversationService, StreamDelta, StreamDone

chat = ConversationService(registry)
conversation_id = await chat.create_conversation()
result = await chat.send("triage", "請求書が欲しい", conversation_id=conversation_id)  # 完結応答
print(result.output)
async for event in chat.stream("triage", "続けて", conversation_id=conversation_id):    # 逐次
    if isinstance(event, StreamDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, StreamDone):
        print()
```

履歴は SDK `Session` に委ね、session_id 連動で既定はプロジェクト直下 `./memory/conversations.db`
に永続化して途中再開できる（保存先は `serve --session-db <path>` または環境変数 `XDG_DATA_HOME`
で変更可、`serve --ephemeral` で揮発）。CLI（`oai-agentspec chat`）はエントリ（登録順の先頭）
エージェント起点で、起動時のセッション選択画面から新規会話 / 過去セッションの復元を選べる。

クライアントの明示注入は 2 経路あり、別軸（モデル実行 / 履歴圧縮）として使い分ける。

- (a) モデル経由（応答生成）: `AgentSpec.model` に `OpenAIResponsesModel(openai_client=AsyncOpenAI(...))`
  を渡す（`examples/_shared/_azure.py` の `azure_model()` 参照）。
- (b) compaction 経由（履歴圧縮）: 下記 compaction を参照。

### HITL（承認必須ツール）

危険・不可逆なツールは `function_tool(needs_approval=True)` で宣言すると HITL（Human-In-The-Loop）
の対象になり、実行前に承認待ちになる。`send` は承認待ち（`SendResult(status="pending")`）を返し、
`resolve_approvals` で call_id 単位に approve / reject する（approve で実行・再開、reject で未実行・
継続）。CLI（`oai-agentspec chat`）でも承認待ちを画面で approve / reject でき、承認待ちのまま会話を
閉じてもセッション復元で続きから再開できる。

### compaction（履歴圧縮）

`SessionPolicy.compaction` に渡す `CompactionConfig` の `enabled` フラグで会話履歴の圧縮を明示的に
有効化する。有効化判定と client / model の受け渡しを分離しており、`compaction=None` または
`enabled=False` なら client を渡しても圧縮しない（暗黙有効化しない）。`enabled=True` かつ client
欠落は構築時に `ValueError`。

```python
from openai import AsyncOpenAI
from oai_agentspec.runtime.conversation import CompactionConfig, ConversationService, SessionPolicy

policy = SessionPolicy(
    compaction=CompactionConfig(enabled=True, client=AsyncOpenAI(), model="gpt-4.1"),
)
chat = ConversationService(registry, session_policy=policy)  # start_server(registry, session_policy=...) も可
```

compaction は SDK が会話履歴を圧縮する機構で、毎ターン後に判定フックが True なら OpenAI の
`responses.compact`（サーバ側＝モデルによる要約）を呼び、SQLite 履歴を要約版へ置換する。要約は
履歴を 1 件に潰すのではなく、ユーザー発話や直近の文脈を保持したまま古い中身を畳む（件数は会話が
伸びれば緩やかに増える）。発火タイミングは `options` 経由で渡す `should_trigger_compaction`
（`Callable[[dict], bool]`）で制御し、既定は候補アイテム（ユーザー発話を除く履歴）が 10 個以上で発火する。

```python
# 閾値や条件を変えるには判定フックを差し替える
CompactionConfig(
    enabled=True, client=AsyncOpenAI(), model="gpt-4.1",
    options={"should_trigger_compaction": lambda ctx: len(ctx["compaction_candidate_items"]) >= 20},
)
```

compaction は OpenAI Responses API 専用で、client は `AsyncOpenAI` / `AsyncAzureOpenAI` の
いずれでも Responses API を叩ければ動く。圧縮に使う `model` は OpenAI 形式の名前
（`gpt-*` / `o*` / `ft:gpt-*`）である必要がある。実行例は `examples/conversation/06_compaction.py`。

## プロンプトレイアウト

プロンプト本体はライブラリに同梱しない。利用側がルートディレクトリと構成を `PromptLayout` で明示する。

```
prompts/
├── base/        # 共通ベース（main.md, sub.md ...）
├── parts/       # 差し込みパーツ（style.md, safety.md ...）
└── agents/      # エージェント個別（triage.md, billing.md ...）
```

```python
PromptLayout(base="base", parts="parts", agents="agents")
# フラット構成なら "" を指定。各ディレクトリ配下は階層化しても再帰探索される。
```

## サンプル

`examples/` に実行可能なサンプルを用意している（AGENT を含む例は Azure OpenAI の Responses API を
利用。環境変数は `examples/_shared/_azure.py` 参照）。「offline」と記した例は API キー不要で動く。
カテゴリ別に `basic/`（基本・ハンドオフ）・`workflow/`（ワークフロー）・`conversation/`（会話 Helper）・
`tool_registry/`（Tool Registry）・`resilience/`（Model retry と run 予算）・
`llmops/`（LLMOps 評価）・`lightning/`（Agent Lightning APO）・`intent/`（意図予測）に整理し、共有ヘルパーは `_shared/`、
プロンプト素材は `prompts/` に置く。`examples/prompts/` はプロンプト記法のサンプル（`PromptStore`
レイアウト base/parts/agents・フロントマター・`${var}`・合成）で、詳細は `examples/prompts/README.md`
を参照。

| ファイル | 内容 |
|---|---|
| `examples/basic/basic.py` | triage -> billing/support のハンドオフ構成 |
| `examples/basic/composition.py` | プロンプト合成順序の比較（offline） |
| `examples/basic/custom_layout.py` | カスタム/フラットなプロンプトレイアウト |
| `examples/basic/dynamic_context.py` | RunContextWrapper による動的 instructions |
| `examples/basic/sub_agents.py` | サブエージェント（agent as tool）オーケストレーション |
| `examples/basic/cyclic_handoff.py` | 相互ハンドオフ（A⇄B 循環）の解決 |
| `examples/basic/dynamic_edge.py` | 動的ハンドオフ（候補から実行時選択。offline） |
| `examples/basic/runtime_update.py` | 実行時の再構成（from_specs / グラフ再 apply / update / unregister。offline） |
| `examples/workflow/workflow_01_sequential.py` | ワークフロー入門: 順次（offline） |
| `examples/workflow/workflow_02_parallel.py` | ワークフロー入門: 並列 fan-out + fan-in 合流（offline） |
| `examples/workflow/workflow_03_conditional.py` | ワークフロー入門: 条件分岐（offline） |
| `examples/workflow/workflow_04_loop.py` | ワークフロー入門: ループ（offline） |
| `examples/workflow/workflow_05_combined.py` | ワークフロー入門: 並列 + 合流 + 条件の組み合わせ（offline） |
| `examples/workflow/workflow_06_conditional_fanout.py` | ワークフロー入門: 条件 fan-out + 動的 fan-in（offline） |
| `examples/workflow/workflow_07_deterministic_context.py` | ワークフロー入門: 経路D（決定論ファサード・context 透過・実 LLM 0 回。offline） |
| `examples/workflow/workflow_handoff_paths.py` | ワークフロー流入 3 経路（C/A/B）の比較 |
| `examples/conversation/01_inprocess.py` | 会話 Helper を in-process で利用（send 完結 + stream 逐次） |
| `examples/conversation/02_session_resume.py` | session_id 連動の永続化と途中再開（resume） |
| `examples/conversation/03_serve_and_cli.py` | 会話サーバ起動 + CLI クライアント接続（クライアント・サーバ型） |
| `examples/conversation/04_hitl_approval.py` | HITL（承認必須ツールの approve / reject）を in-process で実演 |
| `examples/conversation/05_hitl_serve.py` | HITL をサーバ + CLI クライアント型で実演 |
| `examples/conversation/06_compaction.py` | 外部クライアント注入で compaction（履歴圧縮）を明示有効化 |
| `examples/llmops/01_agent_quality_eval.py` | Relevance / Conciseness 等の観点で agent 出力を採点 + 統合 verdict |
| `examples/llmops/02_tool_correctness_eval.py` | 期待ツール（`expected_tools`）と実行 tool_calls の recall 採点 |
| `examples/llmops/03_handoff_route_eval.py` | `HandoffRoute` の経路一致採点（`expected_route`） |
| `examples/llmops/04_langfuse_dataset.py` | Langfuse Datasets / Scores / Tracing 連携 |
| `examples/llmops/05_hitl_interrupted_eval.py` | HITL 中断状態での部分観測評価 |
| `examples/llmops/06_approval_gate_eval.py` | 承認ゲート発火（`ApprovalGate`）の採点 |
| `examples/llmops/07_mock_approve_eval.py` | mock-approve で承認自動解決し全 rollout を採点 |
| `examples/llmops/README.md` | LLMOps 評価の使い方（観点別採点・統合 verdict・Langfuse 連携） |
| `examples/lightning/01_single_agent_apo.py` | 単一 `AgentSpec` + `contains()` の最小 APO 例 |
| `examples/lightning/02_prompt_slot_apo.py` | `prompt_slot` で `PromptStore` 合成プロンプトを APO |
| `examples/lightning/03_graph_apo.py` | `prompt_slot_factory` + `HandoffGraph` でグラフ全体 APO |
| `examples/lightning/04_reward_and_safety.py` | `tool_match` + `tool_mocks` / `approvals` で危険ツールを安全に APO |
| `examples/lightning/05_failure_handling.py` | `OptimizeError` / `FailureKind` の判別（offline） |
| `examples/lightning/06_approval_match_apo.py` | `approval_match` で承認ゲート発火を APO 学習 |
| `examples/lightning/07_composite_reward_apo.py` | `OptimizeCase` 全観点 + 複合 reward + 系全体最適化 |
| `examples/lightning/README.md` | Agent Lightning APO の使い方（reward ファクトリ・`OptimizeResult`・`HistoryEntry` schema） |
| `examples/intent/01_basic_classification.py` | `intent_classifier_from_model` 1 行ヘルパの最小分類例 |
| `examples/intent/07_custom_candidate_generator.py` | 自作 `CandidateGenerator`（キーワードマッチ・LLM 不使用）を `intent_classifier_from_generator` で束ねる（offline） |
| `examples/intent/README.md` | 意図予測の使い方（例 01-07 一覧・信頼境界・レイテンシチューニング） |
| `examples/tool_registry/01_bootstrap_registration.py` | bootstrap で Tool を一元登録し `AgentSpec` に渡す最小例 |
| `examples/tool_registry/02_leaf_module_shared.py` | 分散した葉モジュールの Tool を共有 `ToolRegistry` に登録 |
| `examples/tool_registry/03_metadata_showcase.py` | 全メタデータフィールド（`needs_approval` / `timeout` / `failure_error_function` 等）の SDK 反映（offline） |
| `examples/tool_registry/04_error_handling.py` | 登録・照会エラー 5 種の挙動（offline） |
| `examples/tool_registry/05_runtime_feature_flag.py` | `enabled` の動的トグルで tool を実行時に切替（feature flag 用途） |
| `examples/resilience/01_retry_and_budget.py` | `ModelRetryPolicy` + `RunBudgetPolicy` の併用 + streaming 経路での例外観測 + `asyncio.wait_for` によるハード timeout パターン |
| `examples/resilience/02_failsafe.py` | `FailsafePolicy` の宣言 1 回 + `failsafe_call` による例外着地（正常時は透過・未宣言例外は素通し・`Exception` キーの build-time 拒否・`last_agent` の 2 段解決（`RUNNING_AGENT` / `fallback_last_agent`）） |
| `examples/hooks/01_chain_agent_hooks.py` | agent 単位フックの合成（`chain_agent_hooks`）: 受理 3 形（インスタンス / 部分実装 / `None`）・縮退（`is` 一致）・fail-fast・run 単位フックの build 時拒否 |
| `examples/hooks/02_chain_hooks.py` | run 単位フックの合成（`chain_hooks`）と agent 単位との非対称（`None` 非除外・部分実装非許容）+ 両スコープ同時使用 |
| `examples/mcp/01_declarative_mcp_servers.py` | `AgentSpec.mcp_servers` / `mcp_config` の宣言・run 時の MCP ツール解決・接続 lifecycle（同梱の最小 MCP サーバ `_server.py` を stdio で自動起動） |
| `examples/mcp/README.md` | MCP サーバ宣言の使い方（専用フィールド・run 時解決・lifecycle 責務・`mcp_config` の注意点） |

```bash
uv run python examples/basic/basic.py
uv run python examples/workflow/workflow_01_sequential.py   # offline（API キー不要）
uv run python examples/conversation/01_inprocess.py
uv run python examples/llmops/01_agent_quality_eval.py
uv run python examples/lightning/01_single_agent_apo.py
uv run python examples/lightning/05_failure_handling.py     # offline（API キー不要）
uv run python examples/intent/07_custom_candidate_generator.py  # offline（API キー不要）
```

会話 Helper（`ConversationService` / serve / CLI）・HITL・compaction の使い方は
[コアコンセプト](#コアコンセプト)の該当節を参照。

`final_output` は最終文しか表さないため、network 系のサンプルは共有ヘルパー
`examples/_shared/_run_path.py`（`print_run_path`）で「どのエージェントを経由し、どこでハンドオフ /
ツール呼び出しが起きたか」を `RunResult.new_items` から表示する。SDK の tracing とは別物。

## 開発

```bash
make test      # pytest（カバレッジ付き）
make lint      # ruff check + format --check
make format    # ruff format + check --fix
```

セキュリティスキャン（SAST: SonarQube / SCA: Trivy / Secrets: gitleaks）はローカルゲートとして運用する。
詳細は [docs/security-scanning.md](docs/security-scanning.md)、設計は [docs/architecture.md](docs/architecture.md) を参照。

```bash
make security-up   # SonarQube 起動（初回はトークンを発行し .env に設定）
make sast          # SAST（Quality Gate 合否まで判定）
make sca           # SCA（uv.lock 直スキャン）
make secrets       # gitleaks（git 履歴 + 作業ツリー）
```

## プロジェクトステータス

Alpha（0.2.x）。公開契約は `oai_agentspec.__all__` のシンボルのみで、バージョニングは SemVer に従う。
API は安定化に向け変更される可能性がある。

## ライセンス

[MIT License](./LICENSE)。`pyproject.toml` の `license` フィールドおよび classifier に反映済み。
