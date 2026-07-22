# クイックスタート

## 何を解決するか

初めて `oai-agentspec` を触る利用者が、宣言 → build → 実行の最短経路を把握するためのページです。ここでは `AgentSpec` + `AgentRegistry` + `HandoffGraph` + `Runner.run` の最小構成を示します。

プロンプト合成・サブエージェント・ワークフロー・会話 Helper 等の詳細は後続のトピックページで扱います。

## 使い方

- import: `from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph, PromptLayout, PromptStore`
- extras: なし（`Runner` は `openai-agents` に付属）
- 依存 env: モデル接続に必要な env のみ（例: Azure なら `examples/_shared/_azure.py` を参照）

```python
import asyncio
from pathlib import Path
from agents import Runner
from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph, PromptLayout, PromptStore

store = PromptStore(Path("prompts"), PromptLayout(base="base", parts="parts", agents="agents"))
registry = AgentRegistry()
for name in ("triage", "billing", "support"):
    registry.register(AgentSpec(
        name=name,
        instructions=store.compose(agent=name, base="main", parts=["style", "safety"]),
    ))

graph = HandoffGraph(entry="triage")
graph.edge("triage", "billing", description="請求関連")
graph.edge("triage", "support", description="技術問い合わせ")
graph.apply(registry)
registry.validate()  # 実行前タイポ検出

async def main() -> None:
    result = await Runner.run(graph.entry_agent(registry), input="請求書が欲しい")
    print(result.final_output)

asyncio.run(main())
```

このページで登場する主要 API はいずれも 3 個以下の引数で完結（`AgentRegistry()` 引数省略・`HandoffGraph(entry=...)` 単一引数）のため「パラメータ一覧」節は割愛します。詳細は次ページ以降を参照してください。

## 落とし穴

- `registry.validate()` を忘れると未解決 handoff がランタイムエラーになる。build 前に呼ぶ
- プロンプトは lib 非同梱。`PromptStore(Path("prompts"), ...)` の root は利用側で用意する

## 参照

- 詳細設計: `docs/architecture.md`
- 具体例: `examples/basic/basic.py`

## 次

[core/agents.md](./core/agents.md) — AgentSpec と AgentRegistry を深掘り
