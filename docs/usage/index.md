# oai-agentspec 使い方ガイド

## このガイドについて

`oai-agentspec` の全機能を「使い分けの判断軸 + 最小コード + examples/ 誘導」で理解できるトピック別ガイドです。人間・AI 双方が読める前提で、1 ページ 1 テーマに集中しています。

詳細設計・不変条件の根拠は `docs/architecture.md`（現在仕様の SoT）、実行可能な具体コードは `examples/` を参照します。本ガイドは判断軸と最小コード、リンクの束ねに徹しています。

## 全体像

3 つの層で構成されます。

- **コア宣言層**（extras 不要）: 宣言的 API と build-time 検証。`docs/usage/core/`
- **Realtime 宣言層**（extras 不要）: `oai_agentspec.realtime` 窓口。`docs/usage/runtime/realtime.md`
- **runtime 実行寄り層**（extras 必要なものあり）: 実行 / 会話 / 安全網 / 評価 / 最適化のヘルパー群。`docs/usage/safety/` `docs/usage/ops/` `docs/usage/runtime/`

各層の機能列挙は下記「extras 一覧」表を参照（機能追加時はこの表が唯一の SoT）。

一貫原則は「build-don't-run」: lib は宣言・build-time 検証・薄い結線に徹し、実行は SDK `Runner.run` に委ねます。

## 推奨読み進め順

1. [quickstart.md](./quickstart.md)
2. [core/agents.md](./core/agents.md)
3. [core/prompts.md](./core/prompts.md)
4. [core/tools.md](./core/tools.md)
5. [core/handoffs.md](./core/handoffs.md)
6. [core/multi_agent.md](./core/multi_agent.md)
7. [core/next_turn.md](./core/next_turn.md)
8. [core/workflow.md](./core/workflow.md)
9. [safety/resilience.md](./safety/resilience.md)
10. [safety/guardrails.md](./safety/guardrails.md)
11. [safety/governance.md](./safety/governance.md)
12. [safety/integrity.md](./safety/integrity.md)
13. [ops/intent.md](./ops/intent.md)
14. [ops/llmops.md](./ops/llmops.md)
15. [ops/lightning.md](./ops/lightning.md)
16. [runtime/realtime.md](./runtime/realtime.md)
17. [runtime/conversation.md](./runtime/conversation.md)
18. [runtime/serve_and_cli.md](./runtime/serve_and_cli.md)
19. [runtime/deterministic.md](./runtime/deterministic.md)

## extras 一覧

| extra | 追加依存 | 有効化される機能 |
|---|---|---|
| （なし） | - | `AgentSpec` / `AgentRegistry` / `AgentNames`（エージェント名定数簿 + `validate_agent_names`）/ `HandoffGraph` / `NextTurnPolicy` / `WorkflowGraph` / `PromptStore` / `ToolRegistry` / `lockdown` / Realtime 宣言 / `runtime.deterministic`（決定的応答モデル + 応答ビルダ） |
| `conversation` | 追加依存なし | `ConversationService` 等の公開窓口分離（opt-in 表現） |
| `serve` | `fastapi` / `uvicorn` | `oai-agentspec serve`（FastAPI サーバ入口・dev 用途） |
| `cli` | `httpx` / `websockets` / `rich` | `oai-agentspec chat`（別プロセス CLI） |
| `llmops` | `deepeval` | `evaluate` / `Criterion` / `Verdict` 等 |
| `llmops-langfuse` | `[llmops]` + `langfuse` | LLMOps トレースを Langfuse へ送信 |
| `lightning` | `agentlightning[apo]` | Agent Lightning APO（プロンプト最適化） |
| `governance` | `agent-governance-toolkit[openai-agents]` | ツール単位ポリシー強制 + 監査ログ |
| `intent` | `pydantic` | `IntentClassifier` / `IntentPolicy` |
| `resilience` | 追加依存なし | `ModelRetryPolicy` / `RunBudgetPolicy` |
| `guardrails` | 追加依存なし | 入出力ガードレール helper |

一次情報は `pyproject.toml` の `[project.optional-dependencies]`（バージョン制約はこちらを参照）。

## 使い分け早見

- 静的 handoff で足りる → [core/handoffs](./core/handoffs.md)
- 実行時に振り分け先を決めたい → [core/multi_agent](./core/multi_agent.md)
- ハンドオフ後の次ターン開始エージェントを宣言で固定したい → [core/next_turn](./core/next_turn.md)
- 決定論的に順次実行したい → [core/workflow](./core/workflow.md)
- 会話履歴を扱いたい → [runtime/conversation](./runtime/conversation.md)
- レイテンシ最小の音声 → [runtime/realtime](./runtime/realtime.md)
- 実 API を呼ばずに決定的な応答で動かしたい → [runtime/deterministic](./runtime/deterministic.md)
- 実行の安全網（retry / budget / guardrails / policy） → [safety/](./safety/)
- 品質評価 → [ops/llmops](./ops/llmops.md)
- プロンプト自動最適化 → [ops/lightning](./ops/lightning.md)

## 横断原則

- **SDK 隔離（NFR-1）**: `from agents` / `from openai` の import は `_adapters/` 配下のみ。上位層は plain データと不透明型のみ扱う
- **build-don't-run**: 実行は SDK `Runner.run` に委譲。lib は公開の実行 API を持たない
- **環境変数は CLI 境界のみ**: `SessionPolicy` / `ConversationService` / `serve` / `_adapters` は env 非依存
- **プロンプトは lib 非同梱**: 利用側 root を `PromptStore` に渡す
- **`bool` フィールドは構築時に型検証**: 宣言 dataclass の `bool` / `bool | None` 注釈フィールドは構築時に型検証され、非 bool 値（`None` / 文字列 / int の `0` `1`）は `ValueError` で拒否される。`bool | None` は `None` を正当値として受理する（`docs/adr/0021-declarative-bool-field-validation.md`）

## 詳細設計 SoT

本ガイドは使い方に特化しています。詳細設計・不変条件の根拠は `docs/architecture.md`。

## 次

[quickstart.md](./quickstart.md) — 最速で動かす
