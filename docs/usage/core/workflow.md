# WorkflowGraph（決定論的な DAG 実行）

## 何を解決するか

LLM に振り分けを委ねると、順序・並列度・合流条件が非決定的になります。`WorkflowGraph` は LangGraph / Microsoft Agent Framework に倣い、**ノード（AGENT / FUNCTION）とエッジ（通常 / 条件 / fan-in）を明示宣言**して順次・並列・条件分岐・ループを表現します。

実行口は SDK `Runner.run` 一本。ワークフローは Agent として（`as_agent_spec`・経路 C）または tool ファサードとして（`as_facade_spec`・経路 A/D）消費します。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| 順次 | `add_edge(a, b)` 直列 | 直列パイプライン |
| 並列 fan-out | 同一 src から複数 `add_edge` | 独立処理を並列化 |
| 条件 | `add_conditional_edges(src, router)` | 分岐条件をコードで書く |
| fan-in 合流 | `add_fan_in_edge([...], dst)` | 並列結果を統合 |
| ループ | 条件エッジで戻り先を指す | retry / 反復精緻化 |
| 経路 C（`as_agent_spec`） | ワークフローを Agent として登録 | 外側 handoff から使う |
| 経路 A（`as_facade_spec`・既定 `LLM_INPUT`） | tool ファサード（LLM 起点） | 内部ノードで context 透過 |
| 経路 D（`as_facade_spec(mode=FacadeMode.DETERMINISTIC)`） | 決定論ファサード（LLM 0 回） | 完全決定論・context 透過 |

## 使い方

- import: `from oai_agentspec import WorkflowGraph, START, END, Router, NodeFn, NodeHook, NodeResults, FacadeMode, WorkflowFrozenError, default_input_filter`
  （`default_input_filter` は `as_agent_spec` / `as_facade_spec` の `input_filter` 引数で流入履歴を有界化するときに渡す既定関数。カスタム filter を書かない限り明示 import 不要）
- extras: なし
- 依存 env: なし

```python
from oai_agentspec import START, END, WorkflowGraph

wf = WorkflowGraph("pipeline")
wf.add_agent_node("plan", agent="planner")
wf.add_function_node("format", fn=lambda msg, ctx: f"<{msg}>")
wf.add_edge(START, "plan")
wf.add_edge("plan", "format")
wf.add_edge("format", END)
wf.validate(registry)

registry.register(wf.as_agent_spec("pipeline_agent", registry=registry))
result = await Runner.run(registry.get("pipeline_agent"), input="...")
```

tracing は自動配線されます（span 構造の詳細は `docs/architecture.md` ワークフロー節）。

## パラメータ一覧

（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `WorkflowGraph`（dataclass）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `name` | `str` | 必須 | ワークフロー名 |
| `nodes` | `dict[str, WorkflowNode]` | `{}` | ノード辞書 |
| `edges` | `dict[str, list[Any]]` | `{}` | 通常エッジ |
| `conditional_edges` | `dict[str, ConditionalEdge]` | `{}` | 条件エッジ |
| `fan_in_edges` | `dict[str, FanInEdge]` | `{}` | 合流エッジ |
| `entry` | `str \| None` | `None` | START の下流ノード名 |
| `recursion_limit` | `int` | `WORKFLOW_DEFAULT_RECURSION_LIMIT` | 1 run のノード実行数上限 |
| `run_defaults` | `dict[str, Any] \| None` | `None` | 全 AGENT ノードの `Runner.run` 既定 kwarg（`input` / `context` は予約） |

### 主要宣言メソッド

- `add_agent_node(name, *, agent, run_options=None)` — AGENT ノード追加
- `add_function_node(name, *, fn)` — FUNCTION ノード追加（`fn: NodeFn = (msg, ctx) -> 出力`）
- `add_edge(src, dst)` — 通常エッジ
- `add_conditional_edges(src, router, mapping=None, *, default=None, candidates=None)`
- `add_fan_in_edge(sources, dst)` — dst は FUNCTION 必須

### `as_agent_spec(name, *, ...)`（経路 C）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `name` | `str` | 必須 | 生成 AgentSpec 名 |
| `registry` | `AgentRegistry \| None` | `None` | AGENT ノード解決元 |
| `output_extractor` | `Callable[[Any], str] \| None` | `None` | 最終出力の文字列化 |
| `on_node_start` / `on_node_end` | `NodeHook \| None` | `None` | ノードフック |

### `as_facade_spec(name, *, ...)`（経路 A/D）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `name` | `str` | 必須 | 生成 AgentSpec 名 |
| `registry` | `AgentRegistry \| None` | `None` | AGENT ノード解決元 |
| `mode` | `FacadeMode` | `FacadeMode.LLM_INPUT` | 入口モデル種別 |
| `model` | `Any` | `None` | 入口モデル（`DETERMINISTIC` 指定時に非 None は `ValueError`） |
| `tool_name` | `str \| None` | `None` | ワークフロー tool 名 |
| `tool_description` | `str \| None` | `None` | tool 説明 |
| `output_extractor` | `Callable[[Any], str] \| None` | `None` | 最終出力の文字列化 |
| `on_node_start` / `on_node_end` | `NodeHook \| None` | `None` | ノードフック |

### `FacadeMode` メンバ

`DETERMINISTIC`（実 LLM 0 回）/ `LLM_INPUT`（既定・実 LLM 1 回）/ `LLM_INPUT_OUTPUT`（実 LLM 2 回）。

## 判断軸

- 実行順序を LLM に任せてよいなら **handoff** で足りるが、順序・条件を厳格に固定したいなら **WorkflowGraph**
- 外側 handoff の 1 ノードとして扱いたい → **経路 C（`as_agent_spec`）**
- 内部ノードで外側 context を参照したい → **経路 A（`as_facade_spec`）**
- 決定論を保ったまま context 透過（実 LLM 0 回） → **経路 D（`as_facade_spec(mode=FacadeMode.DETERMINISTIC)`）**

## 落とし穴

- `add_edge` 同一 src 複数張り = 並列 fan-out。順次のつもりで書くと想定外の並列実行になる
- ループは `add_conditional_edges` で戻り先を指す。無限ループにならないよう終了条件を必ず入れる
- `freeze()`（または `lockdown(root, workflow=wf)`）後の add_* は `WorkflowFrozenError`

## 参照

- 詳細設計: `docs/architecture.md`（ワークフロー節）
- 検討経緯: `docs/rationale/workflow-handoff-inflow.md`
- 具体例: `examples/workflow/workflow_01_sequential.py` 〜 `workflow_07_deterministic_context.py` / `workflow_handoff_paths.py`

## 次

[../safety/resilience.md](../safety/resilience.md) — Model retry と run 予算
