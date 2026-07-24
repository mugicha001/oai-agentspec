# Agent Lightning APO（プロンプト自動最適化）

## 何を解決するか

`AgentSpec` / `HandoffGraph` / `WorkflowGraph` のプロンプトを Agent Lightning に委譲し、reward 関数で採点しながら textual gradient + beam search で自動改善します。プロンプトを「slot」として抽出（`prompt_slot` / `prompt_slots`）し、`optimize()` に reward と評価ケースを渡すだけで学習ループが回ります。

本 extra は APO（プロンプト最適化）のみを提供します。`optimize()` に `algorithm="rl"` を渡すと未対応として明確なエラーで案内されます。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| `contains` / `exact` | 出力文字列マッチ | 決定的な期待出力 |
| `tool_match` | 期待 tool 呼び出しの一致 | ツール正しさで学習 |
| `approval_match` / `route_match` / `last_agent_match` | 承認ゲート / 経路 / 最終 agent 一致 | HITL / handoff 学習 |
| `judge(rubric, model)` | LLM-as-judge | 意味的品質 |
| 複合 reward | 上記の重み付き合成（利用者側で組む） | 系全体の総合最適化 |
| `prompt_slot` | 単一 spec の slot 抽出 | 単一 agent APO |
| `prompt_slots` | 複数 spec 一括 | グラフ全体 APO |

## 使い方

- import: `from oai_agentspec.runtime.lightning import (optimize, OptimizeConfig, OptimizeCase, OptimizeResult, Slot, RolloutResult, FailureKind, OptimizeError, contains, exact, tool_match, approval_match, route_match, last_agent_match, judge, prompt_slot, prompt_slots, train_val_split)`
- extras: `pip install oai-agentspec[lightning]`（`agentlightning[apo]`）
- 依存 env: 学習に使う Model の env

```python
from openai import AsyncOpenAI
from oai_agentspec.runtime.lightning import (
    OptimizeCase, contains, optimize, prompt_slot, train_val_split,
)

slot = prompt_slot(store, registry, tune="triage")
cases = [OptimizeCase(input="請求書ください", expected_output="請求")]
train, val = train_val_split(cases, val_ratio=0.2)

result = await optimize(
    triage_spec,
    train=train,
    val=val,
    reward=contains(),                 # `expected_output` を既定参照
    slot=slot,
    registry=registry,
    apo_client=AsyncOpenAI(),          # APO 必須
)
print(result.prompt)                   # 最適化済みテキスト（${var} 保持）
print(result.diff)                     # seed vs prompt の unified diff
```

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `optimize` の主要パラメータ（15 個超のため主要 10 個に絞る・残りは docstring 参照）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `target` | `Any` | 必須 | AgentSpec / WorkflowGraph / HandoffGraph |
| `algorithm` | `str` | `"apo"` | `"rl"` は本 extra で未対応 |
| `train` | `Sequence[Any]` | 必須（kw_only） | 最適化 / rollout に使う入力ケース |
| `val` | `Sequence[Any] \| None` | `None`（**必須**・空は CONFIG_MISSING） | 検証ケース |
| `reward` | `Callable[[RolloutResult], float \| Awaitable[float]]` | 必須（kw_only） | 報酬算出 |
| `registry` | `AgentRegistry \| None` | `None` | HandoffGraph 必須 |
| `slot` | `Slot \| str \| dict[str, Slot \| str] \| None` | `None` | 最適化対象スロット |
| `rebind` | `Callable[[Any], Any] \| None` | `None` | 生 seed 経路で必須 |
| `tool_mocks` | `dict[str, dict[str, Any]] \| None` | `None` | rollout 副作用の安全化 |
| `approvals` | `Callable[[dict], bool] \| None` | `None` | 承認自動解決 |
| `apo_client` | `Any` | `None`（**APO 必須**） | `AsyncOpenAI` 互換クライアント |

省略した追加 kwarg（`config` / `rounds` / `concurrency` / `timeout_seconds` / `store` / `apo_gradient_model` / `apo_apply_edit_model` / `apo_beam_width` / `apo_branch_factor` / `tracer`）は `OptimizeConfig` の各フィールドと同じ意味で、直接 kwargs か `config=OptimizeConfig(...)` の一方で渡す（同時指定は `CONFIG_MISSING`）。

### `OptimizeConfig`（frozen・主要 10 個）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `concurrency` | `int \| None` | `None` | rollout 並列度 |
| `rounds` | `int \| None` | `None` | 訓練ラウンド数（APO は `beam_rounds` にマップ） |
| `timeout_seconds` | `float \| None` | `None` | APO 1 batch タイムアウト（None で APO 既定 3600 秒） |
| `store` | `Any` | `None` | Agent Lightning Store（不透明値） |
| `apo_client` | `Any` | `None` | APO 必須 |
| `apo_gradient_model` | `str \| None` | `"gpt-5.4-mini"` | textual gradient 用モデル名 |
| `apo_apply_edit_model` | `str \| None` | `"gpt-5.4-mini"` | prompt edit 適用モデル名 |
| `apo_beam_width` | `int \| None` | `None` | beam 幅 |
| `apo_branch_factor` | `int \| None` | `None` | beam 分岐数 |
| `tracer` | `Any` | `None` | 独自 Tracer（escape hatch） |

### `OptimizeCase`（frozen）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `input` | `str` | 必須 | rollout への入力 |
| `id` | `str \| None` | `None` | ケース識別子 |
| `expected_output` | `str \| None` | `None` | `contains` / `exact` 既定参照 |
| `expected_tools` | `list[str]` | `[]` | `tool_match` 既定 |
| `expected_route` | `list[str]` | `[]` | `route_match` 既定 |
| `expected_last_agent` | `str \| None` | `None` | `last_agent_match` 既定 |
| `expected_approvals` | `list[str]` | `[]` | `approval_match` 既定 |
| `metadata` | `dict[str, Any]` | `{}` | 補助情報 |

### `Slot`（frozen）

`name` / `seed` / `build: Callable[[str], Any]` / `vars: dict[str, Any] = {}` / `fixed: str = ""`。

### `OptimizeResult`（frozen）

`prompt: str | dict[str, str]` / `train_score: float` / `val_score: float | None = None` / `history: list[HistoryEntry] = []` / `seed: str | dict[str, str] = ""` / `diff: str | dict[str, str] = ""`。`.to_dict()` / `.save(path)` を提供。

### `RolloutResult`（frozen・reward が受ける plain 観測）

`case: Any` / `output: str` / `tool_calls: list[str] = []` / `fired_approvals: list[str] = []` / `route_steps: list[str] = []` / `last_agent: str | None = None`。

### reward ファクトリ（すべて `field` 位置引数 1 個で既定は `OptimizeCase` の対応フィールド名）

- `contains(field="expected_output")`
- `exact(field="expected_output")`
- `tool_match(field="expected_tools")`
- `route_match(field="expected_route")`
- `last_agent_match(field="expected_last_agent")`
- `approval_match(field="expected_approvals")`
- `judge(rubric, model)` — 2 引数（rubric: str / model: Any）

### `prompt_slot(store, registry=None, *, tune, base=None, parts=(), vars=None, build=None)`

### `prompt_slots(store, registry, agents, *, base=None, parts=(), vars=None)`

### `train_val_split(data, *, val_ratio=0.2, seed=0, shuffle=True)`

### `FailureKind`（StrEnum）: `EXTRA_MISSING` / `CONFIG_MISSING` / `TRAINER_FAILED`

### `OptimizeError`

`OptimizeError(kind: FailureKind, message: str)`。

## 判断軸

- 期待出力が決定的 → **`contains` / `exact`**、意味的品質 → **`judge`**、tool / handoff / 承認は該当 reward
- 単一 agent の改善は **`prompt_slot`**、グラフ全体は **`prompt_slots`**
- 学習 rollout は API コスト源。`OptimizeConfig` で試行回数を制御し `train_val_split` で汎化確認

## 落とし穴

- `agentlightning` は private API 依存のため `pyproject.toml` で patch pin されている
- `OptimizeError` / `FailureKind` は必ず判別ハンドリング（rollout 失敗を握り潰さない）
- `val` は必須（APO の beam search 契約）。空は `CONFIG_MISSING`
- `apo_client` は APO 必須（未指定は `CONFIG_MISSING`）
- 結果テキストは `result.prompt`（`best_prompts` ではない）

## 参照

- 詳細設計: `docs/architecture.md`（Agent Lightning 節）
- 具体例: `examples/lightning/01_single_agent_apo.py` 〜 `07_composite_reward_apo.py`

## 次

[../runtime/realtime.md](../runtime/realtime.md) — Realtime エージェント
