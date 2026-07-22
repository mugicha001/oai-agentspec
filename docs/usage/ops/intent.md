# 意図予測（IntentClassifier / IntentPolicy）

## 何を解決するか

発話や会話履歴から「ユーザーの意図カテゴリ」を分類し、信頼度付きの候補列（`IntentPrediction`）を返します。ルーティング前段に挟んで dynamic_edge の resolver に使う、承認ゲートの判定に使う、といった用途を想定しています。

`IntentPolicy` で意図集合・返却制約を宣言し、Protocol DI で全体（`IntentClassifier`）・内部段（`ContextBuilder` / `CandidateGenerator`）を差し替えられます。1 行ヘルパ（`intent_classifier_from_model` / `intent_classifier_from_generator`）で最小構成も可能です。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| 挟まない | 従来通り LLM handoff に任せる | 意図分類が過剰・レイテンシ最優先 |
| `intent_classifier_from_model` | Model 1 本で分類 | 最小構成・PoC |
| `intent_classifier_from_generator` | 自作 `CandidateGenerator` を束ねる | LLM を使わないキーワード分類等 |
| `include_policy_in_system=True`（既定） | `IntentPolicy` を system prompt に含める | 意図集合を LLM に明示 |
| `include_policy_in_system=False` | 全無効化（prompt engineering を非同梱） | prompt を完全に利用者管理したい |

## 使い方

- import: `from oai_agentspec.runtime.intent import (IntentClassifier, IntentPolicy, IntentCategory, IntentQuery, IntentContext, IntentPrediction, IntentCandidate, ConsistencyReport, ConfidenceLevel, DefaultIntentClassifier, LLMCandidateGenerator, intent_classifier_from_model, intent_classifier_from_generator)`
- extras: `pip install oai-agentspec[intent]`（`pydantic`）
- 依存 env: 使う Model の env

```python
from oai_agentspec.runtime.intent import (
    IntentCategory, IntentPolicy, IntentQuery, intent_classifier_from_model,
)

policy = IntentPolicy(categories=(
    IntentCategory(name="billing", description="請求関連"),
    IntentCategory(name="support", description="技術問い合わせ"),
))
clf = intent_classifier_from_model(
    model=my_model,
    prompt=lambda ctx: ctx.utterance,
    policy=policy,
)
pred = await clf.classify(IntentQuery(utterance="請求書ください"))
```

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `IntentPolicy`（pydantic BaseModel・frozen・extra="forbid"）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `categories` | `tuple[IntentCategory, ...]` | 必須 | 非空・name 重複なし |
| `max_candidates` | `int` | `3` | 返却候補数上限（>= 1） |
| `extra_instructions` | `str` | `""` | `render_prompt` 先頭に差し込む信頼済み追加指示 |
| `include_rationale_in_prompt` | `bool` | `False` | 出力例 JSON に rationale を含めるか |

### `IntentCategory`（BaseModel・frozen・全 2 引数）

`name: str` / `description: str`。

### `intent_classifier_from_model(model, prompt, *, policy, history_limit=20, include_policy_in_system=True, model_settings=None)`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `model` | `Any` | 必須 | LLM（不透明値・DI） |
| `prompt` | `Callable[[IntentContext[Any]], str]` | 必須 | ctx → user 入力文字列 |
| `policy` | `IntentPolicy` | 必須（kw_only） | 分類契約 |
| `history_limit` | `int` | `20` | `DefaultContextBuilder` が history から取得する上限 |
| `include_policy_in_system` | `bool` | `True` | `policy.render_prompt()` を system に注入 |
| `model_settings` | `Any \| None` | `None` | `agents.ModelSettings` 相当 |

### `intent_classifier_from_generator(generator, *, history_limit=20)`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `generator` | `CandidateGenerator` | 必須 | 自作 Protocol 実装 |
| `history_limit` | `int` | `20` | 上と同じ |

### `IntentQuery`（BaseModel・generic `[TContext]`）

`utterance: str = ""` / `history: Any | None = None` / `run_context: TContext | None = None`。

### `IntentPrediction` / `IntentCandidate` / `ConsistencyReport` / `ConfidenceLevel`

`ConfidenceLevel` は `StrEnum`: `CERTAIN` / `HIGH` / `MEDIUM` / `LOW` / `SPECULATIVE`。他は data のみのため docstring 参照。

## 判断軸

- 従来の LLM handoff で精度が足りるなら **挟まない**（レイテンシ・コスト増を避ける）
- 意図集合を明示制約したい → **`IntentPolicy` + `intent_classifier_from_model`**
- LLM を使わずルールで分類したい → **`intent_classifier_from_generator` + 自作 `CandidateGenerator`**
- prompt を完全に自前管理したい → **`include_policy_in_system=False`**

## 落とし穴

- `IntentPolicy.render_prompt()` の固定文（タスク指示・出力形式）は lib 側 parser との出力契約の serialize。`include_policy_in_system=False` で全無効化可能
- 窓口は PEP 562 遅延 import。extra 未導入時は属性アクセスで `ImportError`
- `IntentPolicy.categories` は tuple（`list` を渡してもよいが frozen として保持される）

## 参照

- 詳細設計: `docs/architecture.md`（意図予測節）
- 具体例: `examples/intent/01_basic_classification.py` 〜 `07_custom_candidate_generator.py`

## 次

[llmops.md](./llmops.md) — 品質評価
