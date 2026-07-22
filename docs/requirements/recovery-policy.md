# 宣言的 Model Retry と Run Budget（Resilience 系宣言型・初版）

## 命名注記

Issue #26 本文と本要件書のファイル名は「Recovery」を継続維持する（既存参照の破壊を避けるため）。ただし実装上の lib 名称は上位概念を「Resilience」（.NET Polly 等の業界前例）に統一し、下位の具体宣言を `ModelRetryPolicy` / `RunBudgetPolicy` の 2 種に分解する。Failsafe（任意例外の宣言的着地）は Issue #30 で別途扱う。

## 1. 概要

Model 呼び出しの一時失敗と、run 全体の予算超過に対する制御を、実行コードの分岐ではなく宣言型として一元管理できるようにする機能。openai-agents SDK 0.17.4 は `ModelRetrySettings`（Model 呼び出し retry）を提供するが、(a) `policy` 未指定時の silent no-op や `retry_policies.any(...)` の合成が煩雑で使いにくいこと、(b) run 全体の累積時間・累積トークン上限が SDK に存在しないこと、の 2 点が課題である。本機能は lib 独自の実行ループを持たず、宣言型を SDK ネイティブ機構（`ModelSettings.retry` / `Runner.run(hooks=...)`）へコンパイルする（build-don't-run の徹底）。任意例外の宣言的着地（Failsafe）は Issue #30 で別途扱い、本 Issue には含めない。

## 2. 機能要件

### 初版スコープ一覧

| 項目 | 初版に含む | 関連 FR | 備考 |
|------|:---:|------|------|
| Model retry 宣言（回数・backoff・条件） | 含む | FR-1 | SDK `ModelRetrySettings` にコンパイル |
| Model retry の Agent / Runner 両対応 | 含む | FR-1 | SDK の `_merge_retry_settings` に委譲（両方指定時は Runner 側が優先） |
| セマンティックフラグ（rate_limit / server_error / network_error / timeout / retry_after） | 含む | FR-1 | HTTP status 暗記を避ける |
| run 全体の累積時間 / トークン上限 | 含む | FR-2 | Runner scope のみ |
| 上限到達時の専用例外 (`RunBudgetExceeded`) | 含む | FR-2 | usage / elapsed_seconds / context を属性保持 |
| SDK 生型の再エクスポート窓口 | 含む | FR-3 | 上級者用エスケープ（`from agents` を書かせない） |
| SDK ネイティブ `RunErrorHandlers` との共存 | 含む | FR-4 | 併用可・SDK が先に飲む |
| `Runner.run_streamed` での動作 | 含む | FR-5 | streaming でも `ModelSettings.retry` と hooks は透過的に効くことを検証 |
| `Runner.run_sync` での動作 | 含む | FR-5 | sync 呼び出しでも動作すること |
| 任意例外の宣言的着地（`FailsafePolicy` / `failsafe_call`） | 含まない | - | Issue #30 として別途起票済み |
| `RealtimeRunner` / `RealtimeSession` 対応 | 含まない | - | websocket ベースで Runner とライフサイクル根本的に異なる |
| Model fallback（別 Model への切替） | 含まない | - | 将来スコープ（SDK ネイティブ機構待ち） |
| Tool 系の retry / fallback / 冪等性ゲート | 含まない | - | Tool Registry (#27) が SoT・SDK ネイティブ (`failure_error_function` / `timeout_behavior`) に委ねる |
| 別 Agent へのエスカレーション（Agent fallback） | 含まない | - | 将来スコープ |
| Workflow ノード単位 / Registry 全体単位の適用 | 含まない | - | 将来スコープ |
| HITL 承認機構との統合 | 含まない | - | 既存 conversation 承認機構との統合は個別 Issue |
| 累積コスト（金額）上限 | 含まない | - | トークン上限で代替。モデル別料金表の同梱はスコープ外 |
| 部分結果の構造化返却 | 含まない | - | 例外情報の充実で代替（利用者が try/except から組む） |

### FR-1: Model retry 宣言（`ModelRetryPolicy`）

- ユーザーストーリー: ライブラリ利用者として、Model 呼び出しのリトライを「よくある条件のセット」と「回数 / backoff」で宣言的に定義したい。なぜなら、SDK 生の `ModelRetrySettings` は `policy` を書き忘れると silent no-op になり、`retry_policies.any(...)` の合成もボイラープレートで冗長だから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `ModelRetryPolicy(max_retries=3)` を宣言する THEN セマンティックフラグ（`retry_on_network_error` / `retry_on_timeout` / `retry_on_rate_limit` / `retry_on_server_error` / `retry_on_retry_after`）の既定値により「まっとうな retry」がデフォルトで有効な不変オブジェクトが得られる（silent no-op を排除）
  - [ ] WHEN `build_model_retry(policy)` を呼ぶ THEN `agents.ModelRetrySettings` インスタンスが返り、`ModelSettings.retry` に埋め込める
  - [ ] WHEN `ModelSettings(retry=build_model_retry(policy))` を Agent 側 (`Agent.model_settings`) と Runner 側 (`RunConfig.model_settings`) の両方に渡す THEN SDK の `_merge_retry_settings` により両者がマージされ、Runner 側が Agent 側を上書きする（SDK ネイティブ挙動に完全委譲）
  - [ ] IF Policy のフィールドに矛盾がある（例: `max_retries` が負数、`backoff_multiplier` が 1 未満、`initial_delay > max_delay`）THEN build-time 検証で `ValueError` により fail-fast する
  - [ ] IF セマンティックフラグで表現できない条件を書きたい THEN `extra_retry_statuses=(408,)` で HTTP ステータスを追加でき、さらに生の `RetryPolicy` callable を `policy=...` で直接渡せる（エスケープハッチ）
  - [ ] IF 生の `policy` を指定した場合 THEN セマンティックフラグは無視され `policy` がそのまま SDK に渡る（生 policy を渡す上級者は自己責任で条件を組む）

### FR-2: run 全体の累積上限（`RunBudgetPolicy`）

- ユーザーストーリー: ライブラリ利用者として、Runner の 1 回の run に対して累積時間・累積トークンの上限を宣言したい。なぜなら、無限ループや想定外の課金・遅延を run 単位で防ぎたく、Model retry の回数上限だけでは run 全体を通した累積の抑制ができないから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `RunBudgetPolicy(max_elapsed_seconds=60.0, max_total_tokens=200_000)` を宣言する THEN Runner scope（run 全体）の上限を保持した不変オブジェクトが得られる
  - [ ] WHEN `build_run_budget_hooks(policy)` を呼ぶ THEN `agents.RunHooksBase` のサブクラスインスタンスが返り、`Runner.run(agent, input, hooks=...)` に渡せる
  - [ ] WHEN run 中の LLM 呼び出しの累積時間または累積トークンが上限を超える THEN `on_llm_end` のターン境界で `RunBudgetExceeded` 例外が送出される（tool 実行の途中で中断しない = graceful）
  - [ ] WHEN `RunBudgetExceeded` が送出される THEN 例外は `usage`（トークン内訳）・`elapsed_seconds`（累積秒）・`context`（トリガした agent 名・LLM 呼び出し回数）を属性として保持し、利用者が try/except で監査ログ・部分結果組立に利用できる
  - [ ] IF `max_elapsed_seconds` / `max_total_tokens` の両方を None にした THEN 実質何もしない no-op hooks が返る（build-time で ValueError にはしない・利用者の意図的な無効化を許容）
  - [ ] IF Policy のフィールドに矛盾がある（例: 上限が負数）THEN build-time 検証で `ValueError` により fail-fast する
  - [ ] IF SDK usage が取得できない LLM 応答があった場合（プロバイダ非対応等）THEN 該当ターンのトークンカウントは 0 として扱い、無音で無効化せず `logger.warning`（構造化: agent 名・ターン番号・理由）で通知する
  - [ ] IF 利用者が既に別の `RunHooksBase` サブクラスを渡している THEN それとの合成は利用者責務（本機能は単一の hooks インスタンスを返し、chain 機構は持たない）
  - [ ] IF ハード timeout（tool 実行中も含めて即中断）が必要な場合 THEN 利用者が `asyncio.wait_for(Runner.run(...), timeout=...)` を自前で被せることを docstring で案内する（本機能はターン境界のみで graceful）

### FR-3: SDK 生型の再エクスポート窓口

- ユーザーストーリー: ライブラリ利用者として、`ModelRetryPolicy` で表現しきれない上級用途で SDK の生型を使う場合も、`from agents import ...` を書かず `oai_agentspec.runtime.resilience` から一貫して import したい。なぜなら、NFR-1（SDK 隔離）の趣旨をユーザーコードにも波及させ、SDK のインポート窓口を lib に一本化したいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `from oai_agentspec.runtime.resilience import ModelRetrySettings, ModelRetryBackoffSettings, retry_policies, RetryDecision, RetryPolicyContext, ModelRetryNormalizedError` する THEN SDK `agents` パッケージの同名シンボルがそのまま取得できる
  - [ ] WHEN 利用者が `from oai_agentspec.runtime.resilience import RunErrorHandlers, RunErrorHandlerResult, RunErrorHandlerInput, RunErrorData` する THEN SDK `agents` パッケージの同名シンボルがそのまま取得できる（自作の error_handlers lambda を書くため）
  - [ ] WHEN `oai_agentspec.runtime.resilience` を import する THEN 追加の外部依存はゼロで（`resilience = []` extra）、`oai_agentspec` 本体だけで利用可能

### FR-4: SDK ネイティブ `RunErrorHandlers` との共存

- ユーザーストーリー: ライブラリ利用者として、SDK ネイティブの `RunErrorHandlers`（`max_turns` / `model_refusal` の run 内着地）と本機能を併用したい。なぜなら、SDK が run 内で着地できる例外は SDK に任せ、それ以外を lib で拾いたいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `Runner.run(..., hooks=build_run_budget_hooks(policy), error_handlers={"max_turns": handler}, run_config=RunConfig(model_settings=ModelSettings(retry=build_model_retry(policy))))` のように SDK ネイティブ機構と本機能を重ねがけする THEN それぞれが独立して動作し相互干渉しない（Model retry は SDK 内部で吸収・budget hooks は on_llm_end で判定・error_handlers は SDK 内部で着地）
  - [ ] IF `MaxTurnsExceeded` に対して SDK の `error_handlers["max_turns"]` を宣言した場合 THEN SDK が run 内で着地し `RunResult` が返り、budget hooks の判定は継続する
  - [ ] IF 本機能を利用せず既存の生 `RunErrorHandlers` のみを使う既存コード THEN 本機能追加により挙動が一切変わらない（純粋追加）

### FR-5: `Runner.run_streamed` / `Runner.run_sync` での動作

- ユーザーストーリー: ライブラリ利用者として、streaming モード（`Runner.run_streamed`）や同期呼び出し（`Runner.run_sync`）でも本機能を使いたい。なぜなら、UI 応答のリアルタイム表示や同期 API 統合等で streaming / sync を選ぶユースケースがあり、機能が非 async の主経路に限定されるのは受け入れがたいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `Runner.run_streamed(agent, input, hooks=build_run_budget_hooks(policy), run_config=RunConfig(model_settings=ModelSettings(retry=build_model_retry(policy))))` を呼ぶ THEN `ModelRetryPolicy` は透過的に効き、`RunBudgetPolicy` は各ターンの `on_llm_end` で判定される（architect 段階で SDK `run_streamed` の実行パスを追跡して `on_llm_end` の発火保証を確定させ、必要なら統合テストで実測担保する）
  - [ ] WHEN 利用者が `Runner.run_sync(...)` を呼ぶ THEN `Runner.run` と同じく本機能の 2 種 Policy が動作する（SDK 内部で `run_sync` は `run` を呼ぶ実装のため透過的に効くことを統合テストで担保する）
  - [ ] IF 上記いずれかで動作しないことが architect 段階で判明した THEN 該当モードをスコープ外に落とし、要件書と docs/architecture.md に「非対応モード」として明記する（未検証のまま実装しない）

## 3. 非機能要件

### NFR-1: SDK 隔離（既存 architecture.md の原則継承）
- 要件: `from agents` / `import agents` は `src/oai_agentspec/_adapters/` 配下のみで許容する。`runtime/resilience/` からは SDK を直接 import しない（生型の再エクスポートも `_adapters` 経由）。
- 計測基準: `grep -rnE "(from agents|import agents)" src/oai_agentspec/runtime/resilience/` の結果が空。

### NFR-2: 単方向依存
- 要件: `runtime/resilience/` は上向きにコア（`_adapters` / `constants`）を参照する単方向のみで、コアから `runtime/resilience` への依存辺を持たない。
- 計測基準: `src/oai_agentspec/` の spec.py / registry.py / handoffs.py / prompts.py / workflow/ から `runtime.resilience` への import 参照が存在しない。

### NFR-3: コア `__all__` 不変
- 要件: `src/oai_agentspec/__init__.py` の `__all__` メンバ集合は本機能追加により変化しない（`ModelRetryPolicy` 等は `oai_agentspec.runtime.resilience` 配下の公開窓口経由でのみ取得可能）。
- 計測基準: 実装前後で `python -c "import oai_agentspec as m; print(sorted(m.__all__))"` の出力が一致。

### NFR-4: build-don't-run（実行 API を持たない）
- 要件: lib は `Runner.run` を包む公開 API を持たない。宣言型は SDK ネイティブ機構（`ModelSettings.retry` / `Runner.run(hooks=...)`) にコンパイルされる。
- 計測基準: `oai_agentspec.runtime.resilience` の公開 API に `Runner.run` の代替関数（`resilience_run` 等）が存在しない。

### NFR-5: 純粋追加（既定挙動不変）
- 要件: 本機能を利用しない既存コードの挙動は一切変わらない。`AgentSpec` に新規フィールドを追加しない（要件が Runner scope 中心であり Agent フィールド追加を必要としないため）。
- 計測基準: 既存テストスイート (`uv run pytest`) が全緑を維持し、`AgentSpec` の dataclass フィールド集合が本機能追加前後で不変。

### NFR-6: extra
- 要件: 本機能は `oai-agentspec[resilience]` extra として提供する。extra は追加の外部依存を持たない（SDK 自体は本体依存）。
- 計測基準: `pyproject.toml` に `resilience = []` が追加され、`pip install oai-agentspec[resilience]` で追加パッケージが取得されない。

### NFR-7: テストカバレッジ
- 要件: 追加コードのカバレッジは既存 `fail_under = 80` を維持する。
- 計測基準: `uv run pytest --cov=src/oai_agentspec/runtime/resilience --cov=src/oai_agentspec/_adapters/resilience --cov-report=term` でモジュール別カバレッジ 80% 以上。

### NFR-8: SDK バージョン依存の明示
- 要件: 本機能は `agents>=0.17.4` の `ModelRetrySettings` / `retry_policies` / `RunHooksBase.on_llm_end` の存在を前提とする。SDK バージョンを `pyproject.toml` で明示する（既に本リポジトリの前提バージョン）。
- 計測基準: 依存最低バージョンで smoke test 通過。

## 4. 制約事項

- **技術的制約 1**: `RunBudgetPolicy` の enforcement は `on_llm_end` のターン境界のみ。tool 実行中の割り込みなし。ハード timeout が必要な場合、利用者が `asyncio.wait_for(Runner.run(...), timeout=...)` を自前で被せる（docstring / examples で明示）。
- **技術的制約 2**: `RunBudgetPolicy` は Runner scope 専用（`RunHooksBase` サブクラスを `Runner.run(hooks=...)` に渡す形）。Agent 単位の予算は本機能スコープ外（要件が Runner 共通の予算のため）。
- **技術的制約 3**: `Runner.run_streamed` での `on_llm_end` 発火保証は architect 段階で SDK 実装追跡と統合テストで確定する。未検証のまま「対応」と主張しない。
- **技術的制約 4**: `RealtimeRunner` / `RealtimeSession` は websocket ベースで Runner と根本的にライフサイクルが異なる（`error_handlers` 引数がない・`ModelSettings.retry` の意味論が成立しない）。本 Issue では完全にスコープ外とし、要件書と docs/architecture.md に「Realtime 非対応」を明記する。
- **技術的制約 5**: 任意例外の宣言的着地（Failsafe）は本 Issue に含めず、Issue #30 で別途扱う。本 Issue で提供する `RunBudgetExceeded` は Issue #30 のハンドラ対象例外の候補となる。
- **ビジネス制約**: Model fallback / 別 Agent fallback / HITL 統合は初版に含めない。Model fallback は SDK が対応した時点で本機能に追加検討する（将来スコープ）。

## 5. 影響範囲

- **関連コンポーネント（新規）**:
  - `src/oai_agentspec/runtime/resilience/`（新設）: 宣言型 (`ModelRetryPolicy` / `RunBudgetPolicy`)、例外 (`RunBudgetExceeded`)、公開窓口 (`__init__.py`)
  - `src/oai_agentspec/_adapters/resilience.py`（新設）: SDK 結線（`build_model_retry` / `build_run_budget_hooks` + 内部 `_BudgetHooks(RunHooksBase)`）
- **関連コンポーネント（変更）**:
  - `src/oai_agentspec/_adapters/__init__.py`: 新規結線関数の再エクスポート
  - `pyproject.toml`: `resilience = []` extra 追加
  - `docs/architecture.md`: Resilience 節を追加（Retry + Budget のみ・Failsafe は #30 参照）
  - `docs/adr/0002-*.md`: 設計判断の記録（0001 は Tool Registry で使用済み）
- **既存機能への影響**:
  - `AgentSpec` / `AgentRegistry` / `ToolRegistry` は変更なし（純粋追加）
  - 既存 extra (`conversation` / `serve` / `cli`) との相互作用なし
  - 既存テストの改修なし

## 6. 用語定義

| 用語 | 定義 |
|------|------|
| Resilience 系宣言型 | 本機能で提供する 2 種の宣言型 `ModelRetryPolicy` / `RunBudgetPolicy` の総称。上位概念名としてのみ使用し、実クラスとしての `ResiliencePolicy` は導入しない |
| ModelRetryPolicy | Model 呼び出しの retry 条件（何回・どんな条件で・どんな backoff で）を宣言する frozen dataclass。`build_model_retry` により SDK `ModelRetrySettings` へコンパイルされる |
| RunBudgetPolicy | run 全体の累積時間 / トークン上限を宣言する frozen dataclass。`build_run_budget_hooks` により `agents.RunHooksBase` サブクラスへコンパイルされる |
| RunBudgetExceeded | `RunBudgetPolicy` の上限到達時に送出される lib 独自例外。usage / elapsed_seconds / context を属性として持つ |
| build-don't-run | 本リポジトリの設計原則。宣言型を SDK ネイティブ機構にコンパイルし、実行は SDK `Runner.run` に委ねる。lib は公開の実行 API を持たない |
| Runner scope | 1 回の `Runner.run(...)` 呼び出しに閉じる範囲。run 内で発生する複数 Agent の LLM 呼び出しをまたいだ集計・制限を指す |
| セマンティックフラグ | HTTP status code の暗記を避け、意図（rate_limit / server_error / network_error / timeout / retry_after）で retry 条件を書けるようにするフラグ群。内部で `retry_policies.any(...)` にコンパイルされる |
| Failsafe | 任意例外を宣言的に着地値へ丸める機構。本 Issue には含めず Issue #30 で別途扱う |
