# LLMOps 評価

## 何を解決するか

エージェント出力の品質を観点別に採点し、統合 `Verdict` を返す評価基盤です。観点は `Criterion` オブジェクト（`Relevance` / `Safety` / `Conciseness` / `Faithfulness` / `GEval` / `ToolUse` / `HandoffRoute` / `ApprovalGate` の組込みファクトリ）で宣言し、`evaluate()` で一括採点します。任意で Langfuse Datasets / Scores / Tracing と連携します。

重い依存（`deepeval` / `langfuse`）はトップ import せず遅延 import。窓口 import は extra 未導入でも壊れず、採点時に必要 extra を案内します。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| `Relevance` / `Conciseness` / `Faithfulness` / `GEval` | LLM-as-judge 系 | 意味的品質 |
| `Safety` | 有害性判定 | 安全性チェック |
| `ToolUse` | 期待 tool 呼び出しとの一致 | ツール正しさ |
| `HandoffRoute` | 期待経路一致 | ハンドオフ挙動テスト |
| `ApprovalGate` | 承認ゲート発火 | HITL フロー検証 |
| Langfuse 連携 off | ローカル evaluate のみ | CI・オフライン |
| Langfuse 連携 on | dataset / scores / tracing 送信 | チーム運用 |

## 使い方

- import: `from oai_agentspec.runtime.llmops import (evaluate, EvalCase, Criterion, Relevance, Safety, Conciseness, Faithfulness, GEval, ToolUse, HandoffRoute, ApprovalGate, Verdict, CriterionStatus, CaseResult, CriterionResult, EvaluationResult, EvaluationConfig, JudgeConfig, LangfuseConfig, register_dataset, load_dataset)`
- extras: `pip install oai-agentspec[llmops]`（`deepeval`）+ 任意で `[llmops-langfuse]`
- 依存 env: judge Model の env・Langfuse 使用時は Langfuse env

```python
from oai_agentspec.runtime.llmops import EvalCase, Relevance, Safety, evaluate

cases = [EvalCase(input="請求書ください", expected_output="請求書リンクをお送りします")]
result = await evaluate(
    triage_spec,               # AgentSpec / HandoffGraph / WorkflowGraph
    cases,
    judge=my_judge_model,      # kw_only 必須（JudgeConfig でラップ可）
    criteria=[Relevance(), Safety()],
    registry=registry,         # HandoffGraph / AGENT ノード付き WorkflowGraph で必須
)
print(result.verdict)          # Verdict.PASS / Verdict.FAIL
```

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `evaluate(target, dataset, *, judge, criteria=None, registry=None, config=None, langfuse=None, approvals=None, tool_mocks=None)`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `target` | `AgentSpec \| WorkflowGraph \| HandoffGraph` | 必須 | 評価対象 |
| `dataset` | `Sequence[EvalCase]` | 必須 | 評価ケース列 |
| `judge` | `JudgeConfig \| Any` | 必須（kw_only） | 採点 LLM（生 model 可・内部で `JudgeConfig` にラップ） |
| `criteria` | `Sequence[Criterion] \| None` | `None` | None で `(Relevance(), Safety(), Conciseness(), Faithfulness())` |
| `registry` | `AgentRegistry \| None` | `None` | HandoffGraph 必須・AGENT 含む WorkflowGraph で必要 |
| `config` | `EvaluationConfig \| None` | `None` | 実行設定 |
| `langfuse` | `LangfuseConfig \| None` | `None` | Langfuse 送信設定 |
| `approvals` | `Callable[[dict], bool] \| None` | `None` | 承認自動解決 resolver |
| `tool_mocks` | `dict[str, dict[str, Any]] \| None` | `None` | `{agent_name: {tool_name: 値 or callable}}` |

### `EvalCase`（frozen）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `input` | `Any` | 必須 | 評価対象への入力 |
| `id` | `str \| None` | `None` | Langfuse dataset item 安定キー |
| `reference_context` | `list[str] \| None` | `None` | Faithfulness 用参照文脈 |
| `expected_route` | `list[str] \| None` | `None` | HandoffRoute 用期待経路 |
| `expected_tools` | `list[str] \| None` | `None` | ToolUse 用期待ツール |
| `expected_approvals` | `list[str] \| None` | `None` | ApprovalGate 用期待承認ゲート |
| `expected_output` | `str \| None` | `None` | 正解文（G-Eval が参照可） |

### `Criterion` ファクトリ（全て kw_only）

- `Relevance(*, knockout=False)`
- `Safety(*, rubric=None, knockout=True)`
- `Conciseness(*, rubric=None, knockout=False)`
- `Faithfulness(*, knockout=True)` — `reference_context` 必要
- `GEval(name, rubric, *, knockout=False)` — `name` / `rubric` は位置引数
- `ToolUse(*, knockout=False)` — `expected_tools` 必要
- `HandoffRoute(*, knockout=False)` — `expected_route` 必要
- `ApprovalGate(*, knockout=False)` — `expected_approvals` 必要

### `EvaluationConfig`（frozen）

`timeout_seconds: float | None = 60.0` / `concurrency: int = 1` / `required_criteria: frozenset[str] | None = None` / `fail_closed_status: CriterionStatus = FAIL` / `inconclusive_policy: Verdict = FAIL` / `deepeval_telemetry_opt_out: bool = True`。

### `JudgeConfig`（frozen・1 引数）

`model: Any`（kw 単一）。

### `LangfuseConfig`（frozen）

`public_key` / `secret_key` / `host` / `dataset_name` / `run_name` / `prompt_name` / `prompt_label`（すべて `str | None = None`）。

### `Verdict`（StrEnum）: `PASS` / `FAIL`

### `register_dataset(config, name, cases)` / `load_dataset(config, name)`

いずれも第 1 引数が `LangfuseConfig`、第 2 引数が dataset 名（3 個以下のため表省略）。

## 判断軸

- 意味的品質は **`Relevance` / `GEval`**、安全性は **`Safety`**、ツール正しさは **`ToolUse`**、経路検証は **`HandoffRoute`**、HITL は **`ApprovalGate`**
- ローカル CI 用途は **Langfuse 連携 off**、チーム運用・観測が要る場合のみ **on**
- judge Model のコストが気になるなら観点数を絞る（LLM-as-judge は API コスト源）

## 落とし穴

- judge Model は各 case × 各 criterion で呼ばれる。case 数と criterion 数を掛け合わせたコストを見積もる
- `approvals` で approve するツールは `tool_mocks` で必ずモック差し替え（本物の危険ツール実行を構造的に阻止）
- `dataset` は sequence（複数件）。単体で試す場合も `[EvalCase(...)]` で渡す

## 参照

- 詳細設計: `docs/architecture.md`（LLMOps 評価節）
- 具体例: `examples/llmops/01_agent_quality_eval.py` 〜 `07_mock_approve_eval.py`

## 次

[lightning.md](./lightning.md) — Agent Lightning APO
