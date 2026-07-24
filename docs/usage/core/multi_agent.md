# 複数エージェントのオーケストレーション

## 何を解決するか

複数の Agent に役割分担させ、入力や状態に応じて適切な Agent を実行するには、大きく分けて 4 つの手段があります。それぞれ「制御が誰に戻るか」「実行順序が誰に決まるか」で最適な場面が異なります。本ページはその使い分けを 1 箇所に集約し、他ページからはここへリンクする形にしています。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| 静的 handoff（`HandoffGraph.edge`） | src → dst を宣言時に固定 | 分岐が少なく設計時に決まる |
| 動的 handoff（`HandoffGraph.dynamic_edge`） | resolver 関数で候補から実行時に 1 つ選ぶ | 入力に応じて候補内で振り分けたい |
| agent as tool（`AgentSpec.sub_agents`） | サブ Agent を tool 呼び出し扱い（制御は親に戻る） | 親が結果を集約する必要がある |
| WorkflowGraph | 決定論的な DAG 実行（順次・並列・条件・ループ） | 実行順序・分岐条件をコードで表現したい |

## 使い方

各手段の詳細な宣言・build 手順は担当ページを参照:

- 静的 / 動的 handoff → [handoffs](./handoffs.md)
- agent as tool → [agents](./agents.md) の `sub_agents`
- WorkflowGraph → [workflow](./workflow.md)

本ページは使い分けの集約表のみを提供します（各手段のパラメータ表は担当ページに記載）。

## 判断軸

- LLM に振り分けを任せてよい → **handoff**（静的または動的）
- 常に親が結果を統合したい・複数サブ Agent の並列呼び出しを LLM に判断させたい → **agent as tool（`sub_agents`）**
- 実行フローを完全に決定論的にしたい・並列 fan-out + fan-in 合流が要る → **`WorkflowGraph`**
- 「振り分けは LLM でよいが実行時条件で候補を絞りたい」→ **dynamic_edge**

## 参照

- 詳細設計: `docs/architecture.md`（サブエージェント節 + ワークフロー節）
- 具体例: `examples/basic/sub_agents.py` / `examples/basic/cyclic_handoff.py` / `examples/basic/dynamic_edge.py` / `examples/workflow/`

## 次

[workflow.md](./workflow.md) — WorkflowGraph の経路パターン
