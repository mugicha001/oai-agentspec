# 意図予測基盤（runtime/intent）

## 1. 概要
発話・会話履歴からユーザー意図を分類する基盤を、実行寄り層 `runtime/intent/`（governance / llmops と同格・`oai-agentspec[intent]` extra・追加依存は `pydantic>=2` のみ）として提供する。Agent / Runner インスタンスに依存せず、呼び出しタイミングにも依存しない独立の分類ヘルパであり、入出力型と Protocol による DI で全体・内部段を差し替えられる。LLM 実行の SDK 結合は `_adapters/intent.py` に閉じ、build-don't-run 方針（実行は SDK `Runner.run` へ委譲）と SDK 隔離を維持する。

## 2. 機能要件

### FR-1: 独立した分類ヘルパ
- ユーザーストーリー: ライブラリ利用者として、既存の Agent / Runner の実行フローに縛られず任意のタイミングで意図分類を呼びたい。なぜなら、ルーティング前・応答後の分析などタイミングは利用側で決めたいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が分類器（`IntentClassifier` 実装）の `classify(query)` を呼ぶ THEN Agent / Runner インスタンスを引数に要求せず、`IntentQuery` のみで分類が完了する
  - [ ] IF 利用側が Agent 実行の前後いずれのタイミングで呼んでも THEN 分類器の挙動は呼び出しタイミングに依存しない

### FR-2: 入力型 `IntentQuery`
- ユーザーストーリー: ライブラリ利用者として、発話・履歴・実行コンテキストを 1 つの不変な入力型で渡したい。
- 受け入れ基準:
  - [ ] WHEN `IntentQuery(utterance=..., history=..., run_context=...)` を生成する THEN pydantic frozen BaseModel（`Generic[TContext]`）として不変に保持される
  - [ ] IF `utterance` を省略する（既定 `""`）THEN 履歴のみモード（history から分類）として扱われる
  - [ ] IF `history`（SDK `Session` 互換の不透明型）/ `run_context` を省略する THEN いずれも任意として `None` で成立する
  - [ ] IF `utterance` と `history` の両方が空 THEN `ValueError` で fail-fast する

### FR-3: 出力型 `IntentPrediction`
- ユーザーストーリー: ライブラリ利用者として、分類結果を信頼度付き候補列の固定契約で受け取りたい。
- 受け入れ基準:
  - [ ] WHEN 分類が成功する THEN `IntentPrediction.candidates` は `IntentCandidate`（`text` / `level` / 任意の `rationale`）の tuple であり、`ConfidenceLevel` 5 段階（`certain` / `high` / `medium` / `low` / `speculative`）の降順で並ぶ
  - [ ] IF 分類器が一貫性判定を行わない THEN `report`（`ConsistencyReport | None`）は `None` となる
  - [ ] IF 実装固有の付帯情報がない THEN `metadata`（`Mapping | None`）は `None` となる

### FR-4: Protocol による差し替え
- ユーザーストーリー: ライブラリ利用者として、分類器全体または内部段（コンテキスト整形・候補生成）を個別に差し替えたい。なぜなら、既定実装の一部だけ独自ロジックにしたいケースがあるから。
- 受け入れ基準:
  - [ ] WHEN `IntentClassifier` / `ContextBuilder` / `CandidateGenerator` の 3 Protocol（`@runtime_checkable`・async）を満たす実装を DI する THEN 全体・内部段のいずれの粒度でも差し替えられる
  - [ ] WHEN 既定実装 `DefaultIntentClassifier` を使う THEN `ContextBuilder` + `CandidateGenerator` の合成として動作し、generator の出力を素通しする（policy を強制しない）

### FR-5: `IntentPolicy` と policy 強制
- ユーザーストーリー: ライブラリ利用者として、分類器が返せる意図集合と返却制約を宣言的に定義し、既定の LLM 生成段で強制させたい。
- 受け入れ基準:
  - [ ] WHEN `IntentPolicy(categories=..., max_candidates=..., extra_instructions=..., include_rationale_in_prompt=...)` を生成する THEN `categories` は非空かつ `name` 一意で検証され、`max_candidates` は既定 3・`ge=1`、`extra_instructions` は既定 `""`（trusted developer text のみ・`render_prompt` 先頭に挿入）、`include_rationale_in_prompt` は既定 `False`（出力例の rationale 有無切替）となる
  - [ ] WHEN `LLMCandidateGenerator` が LLM 出力を受け取る THEN post-hoc 3 段（allowlist フィルタ / `ConfidenceLevel` 降順 sort / `max_candidates` truncate）を適用する（policy 強制は `LLMCandidateGenerator` の責務）
  - [ ] WHEN `render_prompt()` を呼ぶ THEN 固定タスク指示行 + 4 セクション（カテゴリ / 信頼度 / 出力形式 / 制約）+ JSON only 制約の最小プロンプトを返す
  - [ ] IF `include_policy_in_system=False` を指定する THEN `render_prompt()` の system 自動注入が全無効化される

### FR-6: LLM の DI と出力パース
- ユーザーストーリー: ライブラリ利用者として、LLM モデルを DI で渡し、env に依存せず動かしたい。
- 受け入れ基準:
  - [ ] WHEN `LLMCandidateGenerator` に `model` を渡す THEN `agents.Model` 相当を不透明型として受け、環境変数を参照しない
  - [ ] WHEN LLM が応答を返す THEN adapter は raw `str` を返し、`IntentPrediction.model_validate_json` で手動 parse する（strict structured output は生成速度への影響が大きいため採用しない）
  - [ ] IF LLM 出力が Markdown コードフェンスで包まれる THEN パース前に `_strip_code_fence` でフェンスを剥がして parse する
  - [ ] WHEN 履歴を LLM へ渡す THEN `Runner.run(input=list)` の multi-turn 形式で SDK に渡す（string へ flatten しない）

### FR-7: 1 行ヘルパ
- ユーザーストーリー: ライブラリ利用者として、model と prompt から既定構成の分類器を 1 行で組み立てたい。
- 受け入れ基準:
  - [ ] WHEN `intent_classifier_from_model(model, prompt, *, policy, history_limit=20, include_policy_in_system=True)` を呼ぶ THEN 既定構成（`DefaultContextBuilder` + `LLMCandidateGenerator`）の `DefaultIntentClassifier` が返る

### FR-8: 公開窓口
- ユーザーストーリー: ライブラリ利用者として、intent のシンボルを他の runtime extra と同型の公開窓口から import したい。
- 受け入れ基準:
  - [ ] WHEN `from oai_agentspec.runtime.intent import ...` を実行する THEN 14 シンボル（`ConfidenceLevel` / `IntentQuery` / `IntentContext` / `IntentCategory` / `IntentPolicy` / `IntentPrediction` / `IntentCandidate` / `ConsistencyReport` / `IntentClassifier` / `ContextBuilder` / `CandidateGenerator` / `DefaultIntentClassifier` / `LLMCandidateGenerator` / `intent_classifier_from_model`）が import できる
  - [ ] WHEN `import oai_agentspec` を intent extra 未導入環境で実行する THEN 壊れない（PEP 562 遅延再エクスポート・コア `__all__` は不変）

## 3. 非機能要件

### NFR-1: 保守性（SDK 隔離）
- 要件: `agents` への import は `_adapters/` 配下（`_adapters/intent.py`）に限定する。
- 計測基準: `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` の結果が空であること。

### NFR-2: 保守性（単方向依存）
- 要件: `runtime/intent` からコア（`_adapters` / 宣言層型）への上向き参照のみとし、コアから `runtime/intent` への依存辺を作らない。
- 計測基準: 依存方向が既存の runtime レイヤ規約（runtime -> コアの一方向）と一致していること。

### NFR-3: 保守性（コア公開契約の不変）
- 要件: コア `__all__`（27 件）を変更しない。
- 計測基準: `uv run python -c "import oai_agentspec as m; assert all(hasattr(m,s) for s in m.__all__)"` が成功し、`__all__` のメンバ集合が不変であること。

### NFR-4: 保守性（依存非膨張）
- 要件: intent extra の追加依存は `pydantic>=2` のみとする（openai-agents の推移依存として既に存在するが、直接依存の意図で明示宣言する）。
- 計測基準: `pyproject.toml` の `intent` extra が pydantic のみを列挙し、`uv.lock` に新規の外部パッケージが増えないこと。

### NFR-5: 保守性（テストカバレッジ）
- 要件: 新規追加コードを含めてカバレッジ基準を満たす。
- 計測基準: `uv run pytest` が `fail_under = 80` を満たして成功する。

## 4. 制約事項
- スコープ外: 分類結果に基づく実行分岐（PolicyEngine 相当）は本基盤に含めない。ルーティングは利用側が `IntentPrediction` を読んで行う。
- 技術的制約: プロンプト engineering は同梱しない。`render_prompt()` は最小固定文（タスク指示行・出力形式・制約）のみで、lib parser との出力契約の serialize として許容する（`./CLAUDE.md` の「プロンプト非同梱」原則の例外として明文化済み。`include_policy_in_system=False` で全無効化可能）。few-shot 例・ロール定義等は `extra_instructions` または prompt callable で利用側が持つ。
- 技術的制約: 環境変数を参照しない（env 参照は `runtime/cli` 境界に閉じる既存規約に従う）。

## 5. 影響範囲
- 関連コンポーネント: `src/oai_agentspec/runtime/intent/`（`__init__.py` / `types.py` / `protocols.py` / `factories.py` / `_default.py` / `_llm.py` の 6 ファイル）、`src/oai_agentspec/_adapters/intent.py`（LLM 実行の SDK 結合）、`src/oai_agentspec/_adapters/run_context.py`（`unwrap_run_context` 共有ヘルパ）、`examples/intent/`、`pyproject.toml` の `intent` extra
- 既存機能への影響: コアの宣言層・既存 runtime extra の挙動・公開契約は変更しない（純粋な追加）。

## 6. 用語定義
| 用語 | 定義 |
|------|------|
| 意図分類 | ユーザー発話・会話履歴から、事前定義したカテゴリ集合（`IntentPolicy.categories`）のどれに該当するかを信頼度付きで推定する処理 |
| ConfidenceLevel | 分類候補の信頼度 5 段階（`certain` / `high` / `medium` / `low` / `speculative`）。`IntentPrediction.candidates` はこの降順で並ぶ |
| 履歴のみモード | `IntentQuery.utterance=""` で history のみから分類するモード。utterance と history の両方が空の場合は `ValueError` |
| post-hoc 3 段 | `LLMCandidateGenerator` が LLM 出力へ適用する allowlist フィルタ / 降順 sort / truncate の 3 段加工。policy 強制の実体 |
| PEP 562 遅延再エクスポート | モジュールレベル `__getattr__` による遅延 import。extra 未導入環境でも `import oai_agentspec` を壊さないための機構 |
