# LLMOps 評価（oai-agentspec[llmops]）の使い方

利用者が宣言したエージェント（`AgentSpec` / `HandoffGraph` / `WorkflowGraph`）の出力品質を観点別に
採点し、観点別 pass/fail と統合 verdict（CI / リリースの品質ゲートに使える合否）を得るための
評価機能。採点は DeepEval、観測は任意で Langfuse（Tracing / Scores / Datasets / Prompt Management）。

評価対象の宣言物は read-only で扱い変更しない。プロンプトは lib に同梱せず、観点の判定基準文
（G-Eval rubric）や Judge 用モデルは利用者が渡す。

## インストール（extra）

```bash
pip install 'oai-agentspec[llmops]'              # 採点コア（DeepEval）
pip install 'oai-agentspec[llmops,llmops-langfuse]'  # + Langfuse 観測（任意）
```

`llmops`（DeepEval）が採点に必須。`llmops-langfuse`（Langfuse）は観測に使う任意の追加 extra。
Langfuse を使わない場合は `[llmops]` のみで評価できる（依存に langfuse を入れない）。

## 最小例（ローカルのみ・Langfuse なし）

```python
import asyncio
from oai_agentspec import AgentSpec
from oai_agentspec.runtime.llmops import evaluate, EvalCase, Relevance, Conciseness, Verdict

async def main() -> None:
    target = AgentSpec(name="assistant", instructions="簡潔に答える", model=my_model)
    result = await evaluate(
        target,
        [EvalCase("日本の首都は?")],
        judge=my_model,                       # 採点用 LLM（model 直接 or JudgeConfig）
        criteria=[Relevance(), Conciseness()],
    )
    print(result.verdict)                     # Verdict.PASS / FAIL（CI ゲート）
    for case in result.cases:
        for c in case.criteria:
            print(c.criterion, c.status.value, c.score, c.rationale)

asyncio.run(main())
```

`my_model` は SDK のモデル（例: examples では `examples/_shared/_azure.py` の `azure_model()`）。

## 観点（Criterion オブジェクト）

観点は自己完結のオブジェクトで宣言する（メトリクス・knockout・必要データ・rubric を1か所に集約）。
`criteria` を省略すると標準品質セット `[Relevance(), Safety(), Conciseness(), Faithfulness()]` を使う。

ファクトリ名（宣言時の API 名）と**結果キー**（`CriterionResult.criterion` / Langfuse score 名・
`#relevance` 等で表示）は別物。`Faithfulness()` / `ToolUse()` / `HandoffRoute()` は両者がズレる
（ファクトリ＝宣言しやすい名前・結果キー＝計測内容を表す名詞句）。下表で対応を確認すること。

| ファクトリ | 結果キー（score 名） | 採点 | 既定 knockout | 必要データ（EvalCase） |
|---|---|---|---|---|
| `Relevance()` | `relevance` | DeepEval Answer Relevancy | × | — |
| `Safety(rubric=None)` | `safety` | DeepEval G-Eval | ◯ | — |
| `Conciseness(rubric=None)` | `conciseness` | DeepEval G-Eval | × | — |
| `Faithfulness()` | `factual_grounding` | DeepEval Faithfulness | ◯ | `reference_context`（根拠） |
| `GEval(name, rubric)` | 指定した `name` | DeepEval G-Eval（任意観点） | × | — |
| `ToolUse()` | `tool_correctness` | DeepEval ToolCorrectness（決定的・recall） | × | `expected_tools` |
| `HandoffRoute()` | `handoff_correctness` | 経路の決定的比較 | × | `expected_route` |
| `ApprovalGate()` | `approval_gate` | 承認ゲートの決定的比較（実行ゼロ） | × | `expected_approvals` |

- `knockout=True` の観点が fail だと統合 verdict は即 fail（上書き不可）。各ファクトリで上書き可
  （例 `Safety(knockout=False)` / `Relevance(knockout=True)`）。
- G-Eval 系（safety / conciseness / 任意 `GEval`）の判定基準文は `rubric=` で利用者が渡す
  （lib にプロンプトを同梱しない）。`Safety(rubric="...")` / `GEval("politeness", "丁寧な敬語か")`。
- **criteria に挙げた観点だけ評価する**（自動付与しない）。ツール/ルーティングを見たいなら明示で
  `ToolUse()` / `HandoffRoute()` を入れる。
- not_applicable になるのは**必要データ（ground truth）が無いときだけ**（例: `reference_context`
  無で `Faithfulness` →「requires reference_context」/ `expected_tools` 無で `ToolUse` /
  `expected_route` 無で `HandoffRoute`）。対象がツールを持たない・単体対象という理由では NA に
  しない（明示した観点は評価され、期待を満たさなければ fail）。not_applicable / skip は verdict の
  母集合から除外される。
- `ToolUse()` は recall（期待ツールが全て呼ばれていれば pass・余分な呼び出しや handoff の
  `transfer_to_*` は無視・回数/順序は見ない）。handoff 後の下流エージェントが呼んだツールも観測対象。

## EvalCase（評価ケース）

```python
EvalCase(
    input,                       # 評価対象へ渡す入力
    id=None,                     # Langfuse dataset item の安定キー（未指定なら自動導出）
    reference_context=None,      # 根拠（参照文脈）。Faithfulness が忠実性を採点する源泉
    expected_output=None,        # 正解文（golden answer）。dataset item に反映・G-Eval が参照可
    expected_route=None,         # 期待ルーティング（HandoffRoute の ground truth・起点込みフルパス）
    expected_tools=None,         # 期待ツール（ToolUse の ground truth）
    expected_approvals=None,     # 期待承認ツール（ApprovalGate の ground truth・承認ゲート発火）
)
```

`reference_context`（根拠）と `expected_output`（模範解答）は役割が別:
- `reference_context` = 答えが矛盾してはいけない**根拠**（Faithfulness が「出力が根拠に忠実か」を見る）。
- `expected_output` = **模範解答**（dataset item に保存・提供時は G-Eval が参照）。

## verdict（統合合否）

観点別結果から CI ゲート用に1つの pass/fail を導出する:
1. `skip` / `not_applicable` は母集合から除外。
2. knockout 観点（既定 `safety` / `factual_grounding`）が fail なら即 fail。
3. 母集合に fail があれば fail（`inconclusive_policy=PASS` でも実在する fail は隠さない）。
4. inconclusive は `EvaluationConfig.inconclusive_policy`（既定 fail）で解決。
5. 母集合の全観点が pass なら pass。

評価対象が HITL / ツール承認で**中断**した場合（承認必須ツールを呼ぶ等）、既定ではそのケースを
採点せず全観点を inconclusive にして中断を顕在化する（出力非依存の観点が途中経路の一致で誤って
pass するのを防ぐ）。inconclusive は上記 3 の policy（既定 fail）で解決されるため、**中断は既定で
verdict=fail**。例: `05_hitl_interrupted_eval.py`。承認ゲートを評価したり承認を自動解決して完了まで
採点する方法は下記「HITL 評価」を参照。

## HITL 評価（承認ゲート / mock-approve）

承認必須ツール（`function_tool(needs_approval=True)`）を持つエージェントは 2 通りで評価できる。

- **ゲート評価（`ApprovalGate()`）**: 危険ツールを承認ゲートへ正しく回したかを判定する。
  `EvalCase(expected_approvals=[...])` と中断時の承認待ちを決定的比較（recall）。resume/approve せず
  **危険ツールは実行されない**。例 `06_approval_gate_eval.py`。
- **mock-approve（`evaluate(approvals=, tool_mocks=)`）**: 承認を自動解決して完了まで採点する。
  `tool_mocks={agent_name: {tool_name: 値 | callable}}`（**agent スコープのネスト dict**）でツールの
  **実行だけ**を安全なモックに差し替え（name / description / スキーマ / `needs_approval` は不変＝
  ゲートは発火）、`approvals=lambda pending: bool`（`{tool_name, call_id, agent_name}` を受ける）で
  承認/却下を決める。承認後に走るのは本物でなくモックなので、HITL 経路（中断→承認→再開）を安全に
  通して最終応答・ツール使用を採点できる。例 `07_mock_approve_eval.py`::

      tool_mocks={"account-agent": {"delete_account": "deleted (mock)"}}
      approvals=lambda pending: pending["tool_name"] == "delete_account"

  - **却下**: `approvals` が False を返すとツール非実行のまま再開し、拒否後の応答を採点できる。
  - **安全不変条件（approve 認可は `(agent_name, tool_name)` 単位）**: approve できるのは**当該
    agent で実際にモック差し替えされたツールだけ**。モック未登録 / 別 agent の同名ツール / 到達不能の
    ツールを approve しようとすると `ValueError`（同名ツールのすり抜けと本物の危険ツール実行を構造的に
    阻止）。
  - 横断（`HandoffGraph` / 動的 handoff）でも `tool_mocks` を渡すと registry をクローンして下流・
    動的候補のツールも各 agent 名のエントリで差し替える（利用者の registry は汚さない）。
- `approvals` / `tool_mocks` を渡さなければ従来どおり（中断→inconclusive→verdict fail・上記 verdict 節）。

## 実行設定（EvaluationConfig・任意）

```python
from oai_agentspec.runtime.llmops import EvaluationConfig
EvaluationConfig(
    timeout_seconds=60.0,        # 1 観点採点のタイムアウト（超過で fail-closed）
    concurrency=1,               # ケースの並列度（asyncio.Semaphore）
    inconclusive_policy=...,     # 母集合に inconclusive がある時の verdict（既定 fail）
    fail_closed_status=...,      # judge 失敗時に観点へ付与する状態（既定 fail）
    required_criteria=None,      # missing-pair fail-closed（既定 None=母集合由来）
    deepeval_telemetry_opt_out=True,  # DeepEval テレメトリ既定オフ
)
```

## 評価対象（単体 / 横断）

- **単体**: `AgentSpec` を渡す。`registry` 不要。
- **横断（handoff / workflow）**: `HandoffGraph` / `WorkflowGraph` を渡す。
  - `HandoffGraph` は specs を持たないため **`registry`（specs 登録済み）が必須**。
  - `WorkflowGraph` は AGENT ノードを含む場合のみ registry が必要（関数のみなら registry なしで可）。
  - 横断では実行経路を捕捉し、`HandoffRoute()` で `expected_route` と決定的比較する。`expected_route`
    は**起点を含むフルパス**で書く（順序・経由回数まで完全一致・例: triage で受け billing へ handoff
    なら `["triage", "billing"]`）。起点が自身で応答し handoff しなければ `["triage"]`。

```python
result = await evaluate(graph, dataset, judge=my_model,
                        criteria=[Relevance(), HandoffRoute()], registry=registry)
```

## Langfuse 連携（任意・3 モード）

| モード | 設定 | 挙動 |
|---|---|---|
| ローカルのみ | `langfuse` を渡さない | 採点 + verdict をローカルで返すだけ（送信なし） |
| Scores + Traces | `LangfuseConfig(...)`（dataset_name 無） | 観点別スコア・verdict・入出力・観測経路/ツールを Langfuse へ送信 |
| Datasets + Prompt | `dataset_name` / `prompt_name` も設定 | 下記 register→fetch→use + prompt version 集約 |

Langfuse は利用者が用意した稼働中インスタンス（self-host / cloud）へ**送信するだけ**（oai-agentspec は
サーバを立てない・サーバ側評価はさせない）。未設定 / 送信失敗でもローカル verdict は返る（best-effort）。

各ケースの trace metadata には `verdict` に加え観測した **`route`（起点込みフルパス）** と
**`tools_called`** を載せる（`HandoffRoute()` / `ToolUse()` を criteria に入れなくても、Langfuse の
trace で実際の経路・ツールが確認できる）。

### Datasets（register → fetch → use）

データセットは Langfuse を source とし「登録 → 呼び出して使う」運用:

```python
from oai_agentspec.runtime.llmops import register_dataset, load_dataset, LangfuseConfig

cfg = LangfuseConfig(public_key=..., secret_key=..., host=...,
                     dataset_name="qa-suite", run_name="v2", prompt_name="assistant")

register_dataset(cfg, "qa-suite", [EvalCase(...), ...])   # 一度きり（冪等・UI 登録でも可）
cases = load_dataset(cfg, "qa-suite")                     # fetch して EvalCase に復元
result = await evaluate(target, cases, judge=my_model, criteria=[...], langfuse=cfg)
```

- `evaluate` は既存 dataset item に run を**リンクするだけ**（毎回 item を作り直さない）。
- oai-agentspec 固有フィールド（`reference_context` / `expected_route` / `expected_tools`）は dataset item の
  metadata に保存し、`load_dataset` で復元される。

### Prompt Management（push・dedup）

`prompt_name` 設定時、評価対象プロンプト（AgentSpec の静的 instructions）を Langfuse へ登録し、評価
trace / scores を prompt version にリンクする（version ごとに judge 結果が集約される）。

- **内容が変わった時だけ新 version**（同一内容なら既存を再利用・無駄な version を増やさない）。
- Langfuse からプロンプトを取得して**エージェント実行に使う（配信）ことはしない**。`get_prompt` は
  dedup / link 目的に限定。プロンプトの source of truth は利用側（PromptStore 等）のまま。

### Langfuse 認証（環境変数）

`LangfuseConfig` に値で渡す（lib は env を読まない）。examples では `.env` を読み込む:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # self-host なら http://localhost:3000 等
```

## サンプル

| ファイル | 内容 |
|---|---|
| `01_agent_quality_eval.py` | エージェント単体の出力品質（relevance / safety / conciseness / factual_grounding + 任意 G-Eval） |
| `02_tool_correctness_eval.py` | ツール使用の正しさ（`ToolUse()` + `expected_tools`） |
| `03_handoff_route_eval.py` | ルーティングの正しさ（`HandoffGraph` + `HandoffRoute()` + `expected_route`） |
| `04_langfuse_dataset.py` | Langfuse 連携（register→fetch→use + Scores + push 専用 Prompt Management） |
| `05_hitl_interrupted_eval.py` | 承認必須ツールで中断する HITL ケース（中断→全観点 inconclusive→verdict fail・承認は自動解決しない安全設計） |
| `06_approval_gate_eval.py` | 承認ゲートの正しさ（`ApprovalGate()` + `expected_approvals`・危険ツールを承認に回したか・実行ゼロ） |
| `07_mock_approve_eval.py` | 承認を自動解決して完了採点（`approvals` + `tool_mocks`・mock で本物の危険ツールを実行せず HITL 経路を評価） |

実行（Azure OpenAI の環境変数が必要。`examples/_shared/_azure.py` 参照）:

```bash
uv run python examples/llmops/01_agent_quality_eval.py
```

04 は加えて `LANGFUSE_*` が必要。

## スコープ外

- プロンプト単体評価（Promptfoo）/ RAG 評価（RAGAS）/ Summarization 等の専用メトリクス（DeepEval の
  既製メトリクス選択や閾値調整は現状未露出。任意観点は `GEval` の rubric で表現する）。
- 報酬設計（reward modeling）。
