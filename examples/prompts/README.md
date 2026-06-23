# プロンプト記法サンプル（PromptStore レイアウト）

`examples/` の各サンプルが読み込むプロンプト素材であると同時に、`PromptStore` /
`PromptLayout` / `compose` でプロンプトをどう書くかを示す**プロンプト authoring のリファレンス**。
コード側のサンプル（`examples/basic/`・`examples/workflow/`）と対になっている。

## ディレクトリレイアウト

`PromptLayout(base="base", parts="parts", agents="agents")` に対応する 3 区分。

```
prompts/
├── base/        # 土台プロンプト（全エージェント共通の前提）
│   ├── main.md  #   compose(base="main") で選ぶ
│   └── sub.md
├── parts/       # 差し込み部品（再利用する断片）
│   ├── style.md
│   └── safety.md
└── agents/      # 各エージェント固有の本文
    ├── triage.md / billing.md / support.md
    ├── concierge.md
    └── orchestrator.md / researcher.md / writer.md
```

## フロントマター（任意）

各 `.md` の先頭に `---` で囲んだメタデータを置ける（本文には含まれない）。

```markdown
---
version: 1
description: ユーザーの依頼を適切な担当エージェントに振り分ける
---
あなたは ${company} のトリアージ担当です。
```

## テンプレート変数

本文中の `${var}` は `compose(vars=...)` で差し込む。`vars` は dict か、`ctx -> dict` の
callable（動的 instructions。実行時に毎回 render）も渡せる。

```python
# 例: triage.md の ${company} を差し込む
store.compose(agent="triage", base="main", parts=["style", "safety"], vars={"company": "AgentSpec Inc."})
```

## 合成順

`compose` は **base -> parts -> agent** の順に連結する（順序は `layout` で上書き可）。
上の例なら `base/main.md` + `parts/style.md` + `parts/safety.md` + `agents/triage.md` を
合成した文字列が `AgentSpec.instructions` に渡るプロンプトになる。

## 使っているサンプル

- `examples/basic/basic.py`（base + parts + agent の合成 + ハンドオフ）
- `examples/basic/composition.py`（合成順の比較・offline）
- `examples/basic/custom_layout.py`（レイアウトの上書き）
- `examples/basic/dynamic_context.py`（`vars` を callable で動的差し込み）
- `examples/basic/sub_agents.py`（orchestrator / researcher / writer）
- `examples/workflow/workflow_handoff_paths.py`（ワークフロー内の AGENT ノード）
