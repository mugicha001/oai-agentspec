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
| `intent_classifier_from_ml_inference` | sklearn 互換 ML 推論 callable を束ねる | 定型発話が多く低レイテンシ・低コストを優先したい |
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

### ML ベース分類器（sklearn 互換 estimator）

LLM を使わず、学習済みの sklearn 互換 estimator（`fit` / `predict_proba` / `classes_`）で分類する。
`fit_ml_estimator` は estimator の `fit()` を 1 回駆動するゼロコード fit ヘルパ（build-don't-run 不変
条件からの明示的逸脱・詳細は `docs/adr/0004-intent-ml-fit-deviation.md`）。学習を lib に任せない場合
は、学習済み estimator を `ml_inference_from_estimator` でラップする（`fit` を駆動しない）。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from oai_agentspec.runtime.intent import (
    IntentCategory, IntentPolicy, IntentQuery,
    fit_ml_estimator, intent_classifier_from_ml_inference,
)

policy = IntentPolicy(categories=(
    IntentCategory(name="billing", description="請求関連"),
    IntentCategory(name="support", description="技術問い合わせ"),
))
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))),
    ("clf", LogisticRegression(max_iter=1000)),
])
trained = fit_ml_estimator(
    pipeline, x_train=x_train_texts, y_train=y_train_labels, policy=policy,
)
clf = intent_classifier_from_ml_inference(
    trained, thresholds={"certain": 0.90, "high": 0.75, "medium": 0.50, "low": 0.25, "speculative": 0.0},
)
pred = await clf.classify(IntentQuery(utterance="請求書ください"))
```

- extras: `scikit-learn` は lib 本体・`[intent]` extra には含まれない（`[dependency-groups]` の `examples`（例の実行）と `dev`（end-to-end テスト）のみ）。利用側で個別インストールする
- 具体例: `examples/intent/08_ml_sklearn_pipeline.py`（生テキスト経路）/ `09_ml_pretrained_features.py`（事前ベクトル化）/ `10_ml_custom_trainer.py`（学習手段非依存の最小契約）/ `11_ml_persist_reload.py`（pickle 永続化・再接続）

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

### `intent_classifier_from_ml_inference(inference, *, policy=None, mapper=None, thresholds=None, history_limit=20)`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `inference` | `Callable \| TrainedIntentEstimator` | 必須 | ML 推論 callable、または `fit_ml_estimator` 等の成果物（サブクラスの `TunedIntentEstimator` も渡せる） |
| `policy` | `IntentPolicy \| None` | `None` | 省略時は成果物が保持する policy から自動解決（明示指定が優先・両方なしは `ValueError`） |
| `mapper` / `thresholds` | 排他 | `None` | スコア→`ConfidenceLevel` の変換（`thresholds` は5段階名のdict） |
| `history_limit` | `int` | `20` | 上と同じ |

### `fit_ml_estimator(estimator, *, x_train, y_train, policy, transform=None, label_encoding=None)`

sklearn 互換 estimator（`fit` 欠如は `AttributeError`）の `fit()` を 1 回駆動し `TrainedIntentEstimator`
を返す。`transform` 既定は `[ctx.utterance]`（単一サンプル列）。`label_encoding` は非単射（値の重複）
だと `ValueError`。`y_train` のラベルを被覆しない場合も `ValueError`（メッセージはラベル値を載せず
未被覆ラベルの件数のみ）。いずれも fit の駆動前に落ちる。

### `tune_ml_estimator(search, *, x_train, y_train, policy, transform=None, label_encoding=None)`

sklearn 互換の CV 探索器（`fit` 欠如は `AttributeError`）の `fit()` を 1 回駆動し、fit 後の
`best_estimator_` から推論 callable を組んで `TunedIntentEstimator`（`TrainedIntentEstimator` の
サブクラス・`best_params` / `best_score` / `cv_results` を保持）を返す。そのまま
`intent_classifier_from_ml_inference` へ渡せる（policy は成果物から自動解決）。`GridSearchCV` /
`RandomizedSearchCV` / `HalvingGridSearchCV` / 自作探索器のいずれも同じ入口で扱える（探索の設定は
探索器側に閉じる）。fit 後に `best_estimator_` / `best_params_` が無ければ `AttributeError`（再学習を
無効にした探索器は使用できない）、`best_score_` / `cv_results_` が無い場合は当該フィールドが `None`。
`transform` / `label_encoding` の扱いは `fit_ml_estimator` と同じ。

**評価指標（`scoring`）・分割数（`cv`）・探索空間（`param_grid` 等）は探索器インスタンスへ渡す**。
本関数のシグネチャには現れない。`best_score` / `cv_results` の値が何の指標かは探索器の `scoring` が
決め、lib は解釈も変換もせずそのまま成果物へ載せる（複数指標を指定すると `cv_results` のキー名も
`mean_test_<指標名>` へ変わる）。指標の意味を知りたい場合は探索器側（sklearn なら `scorer_`）を見る。

### `ml_inference_from_estimator(estimator, *, transform=None, decoder=None)`

学習済み estimator（`predict_proba` / `classes_` 欠如は `AttributeError`）から `fit` を駆動せず推論
callable を組み立てる。

### `IntentQuery`（BaseModel・generic `[TContext]`）

`utterance: str = ""` / `history: Any | None = None` / `run_context: TContext | None = None`。

### `IntentPrediction` / `IntentCandidate` / `ConsistencyReport` / `ConfidenceLevel`

`ConfidenceLevel` は `StrEnum`: `CERTAIN` / `HIGH` / `MEDIUM` / `LOW` / `SPECULATIVE`。他は data のみのため docstring 参照。

## 判断軸

- 従来の LLM handoff で精度が足りるなら **挟まない**（レイテンシ・コスト増を避ける）
- 意図集合を明示制約したい → **`IntentPolicy` + `intent_classifier_from_model`**
- LLM を使わずルールで分類したい → **`intent_classifier_from_generator` + 自作 `CandidateGenerator`**
- prompt を完全に自前管理したい → **`include_policy_in_system=False`**
- 定型発話が多くレイテンシ・コストを優先したい → **`fit_ml_estimator` + `intent_classifier_from_ml_inference`**

## 落とし穴

- `IntentPolicy.render_prompt()` の固定文（タスク指示・出力形式）は lib 側 parser との出力契約の serialize。`include_policy_in_system=False` で全無効化可能
- 窓口は PEP 562 遅延 import。extra 未導入時は属性アクセスで `ImportError`
- `IntentPolicy.categories` は tuple（`list` を渡してもよいが frozen として保持される）
- `fit_ml_estimator` / `tune_ml_estimator` は build-don't-run 不変条件からの逸脱（利用者が渡した学習器の `fit()` を lib が駆動）で、2 入口が private ヘルパ `_fit_once` 1 箇所を共有する。scikit-learn は lib 本体・`[intent]` extra に含まれず（`[dependency-groups]` の `examples` / `dev` のみ）利用側で個別インストールが必要

## 参照

- 詳細設計: `docs/architecture.md`（意図予測節）
- 検討経緯: `docs/adr/0004-intent-ml-fit-deviation.md`（ML 学習支援の build-don't-run 逸脱）
- 具体例: `examples/intent/01_basic_classification.py` 〜 `07_custom_candidate_generator.py`（LLM 版）/ `08_ml_sklearn_pipeline.py` 〜 `11_ml_persist_reload.py`（ML 版）

## 次

[llmops.md](./llmops.md) — 品質評価
