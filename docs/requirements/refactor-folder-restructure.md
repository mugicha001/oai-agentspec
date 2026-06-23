# src/oai_agentspec リファクタリングおよびフォルダ構成整理

## 1. 概要

`src/oai_agentspec`（openai-agents 上の宣言的エージェント管理ライブラリ）は、`workflow.py`（1335 行）や `conversation/service.py`（685 行）をはじめとする肥大化したモジュールが複数存在し、保守性とディレクトリ構成の一貫性に課題を抱えている。本要件は、特定の巨大ファイルのみを対象とするのではなく `src/oai_agentspec` 全体を対象に、外部から観測できる振る舞いを一切変えない純リファクタリングとして、肥大・責務過多なモジュールの分割、配置不整合の解消、serve（FastAPI）層の標準構成化、tests の src 構造ミラー化、フォルダ・命名・公開 API/インポートパスの再設計を行い、ライブラリの保守性と構造の見通しを向上させることを目的とする。

## 2. 機能要件

### FR-1: 肥大モジュールの責務分割

- ユーザーストーリー: ライブラリ保守者として、肥大化・責務過多なモジュールを責務単位の小さなモジュールへ分割したい。なぜなら、巨大ファイルは可読性・変更容易性・レビュー効率を著しく損なうため、分割により保守性を向上させたいから。
- 対象モジュール: `src/oai_agentspec` 配下の全モジュール。肥大化・責務過多と判断されるものは、ファイル名にかかわらず分割対象に含める（特定ファイルへの限定列挙ではない）。
- 優先的に着手する代表例（網羅リストではない・行数の多い順の例示）: `workflow.py`(1335)、`conversation/service.py`(685)、`serve/app.py`(509)、`cli/chat.py`(530)、`cli/client.py`(503)、`_adapters/session.py`(456)、`_adapters/runner.py`(420)。
- 受け入れ基準:
  - [ ] WHEN リファクタリング完了後に `src/oai_agentspec` 配下を点検した THEN 行数上限（400 行以下を目安）を超える、または責務過多と判断されるモジュールは、ファイル名にかかわらずすべて責務単位に分割されている。
  - [ ] WHEN あるモジュールを分割した後 THEN 分割前に存在した公開シンボルがすべて保持され、それらの振る舞いは分割前と同一である。
  - [ ] WHEN 分割により新規モジュールを作成した場合 THEN 各モジュールはプロジェクトのインポート順序規約（`.claude/rules/01-python.instructions.md` 5 節）に従い、循環インポートが発生しない。
  - [ ] IF 分割によってあるシンボルの定義位置（モジュール）が変わる場合 THEN そのシンボルへ到達するためのインポートパスは再設計後の構成に従って整合し、参照箇所はすべて更新されている。

### FR-2: serve（FastAPI）層の標準構成化

- ユーザーストーリー: ライブラリ保守者として、serve 配下を一般的な FastAPI プロジェクト構成（router / schemas / dependencies 等の層分け）に整理したい。なぜなら、FastAPI のベストプラクティスに沿った層分けにより、エンドポイント定義・スキーマ・依存解決の責務が明確になり保守しやすくなるから。
- 適用範囲: serve（FastAPI）層のみ。パッケージ全体へのドメイン層分けは強制しない。
- 受け入れ基準:
  - [ ] WHEN serve 層を整理した後 THEN serve が提供する HTTP/WebSocket エンドポイントのパス・メソッド・入出力契約（スキーマ）は整理前と同一である。
  - [ ] WHEN serve 層を整理した後 THEN serve 配下は一般的な FastAPI 構成（ルーティング定義 / リクエスト・レスポンススキーマ / 依存解決 等）の責務に分割されている。
  - [ ] IF 層分けを cli / conversation / _adapters 等の他サブパッケージに適用しようとする場合 THEN それは本要件のスコープ外とし、serve 層以外へドメイン層分けを強制しない。

### FR-3: tests の src 構造ミラー化

- ユーザーストーリー: 開発者として、tests のディレクトリ構成を src パッケージ構成と一致させたい。なぜなら、テストの所在が src 構造から自明になり、対応する実装とテストの対応関係を素早く特定できるから。
- 受け入れ基準:
  - [ ] WHEN tests を整理した後 THEN tests 配下のディレクトリ構成は src パッケージのサブパッケージ構成（`_adapters` / `cli` / `conversation` / `serve` 等）にミラーされている。
  - [ ] WHEN 対応する src モジュールがトップレベル（例: `handoffs.py`, `prompts.py`, `registry.py`）の場合 THEN 現状フラットに配置されているトップレベルテスト（`tests/test_handoffs.py`, `tests/test_prompts.py`, `tests/test_registry.py`, `tests/test_extra_isolation.py`, `tests/test_dynamic_instructions.py`, `tests/test_handoff_config.py`, `tests/test_handoffs_cyclic.py`）は、src 構造に従って tests 直下（または対応するサブパッケージ）へ配置される。
  - [ ] WHEN テストファイルを移動・再配置した場合 THEN 既存のテストファイルが持つ `_l1` / `_l2` / `_d5` 等のサフィックスや `test_hitl_*` 等の命名規則は保持される（命名規則そのものは本要件で変更しない）。
  - [ ] IF テスト対象モジュールの分割・移動が行われた場合 THEN 対応するテストの配置は分割後の src 構造に追従して整合する。
  - [ ] IF `tests/_helpers/`（共通フィクスチャ・Fake* 等）のような src にミラー対象を持たない補助ディレクトリがある場合 THEN それはミラー化対象外とする。

### FR-4: フォルダ構成・命名の一貫性整理

- ユーザーストーリー: ライブラリ保守者として、ディレクトリ構成とモジュール命名の一貫性を整えたい。なぜなら、配置・命名のルールが統一されることでパッケージ全体の見通しが良くなり、新規ファイルの配置判断が容易になるから。
- 対象範囲: `src/oai_agentspec` 配下全体（一部ファイルに限定しない）。
- 受け入れ基準:
  - [ ] WHEN フォルダ構成・命名を整理した後 THEN `src/oai_agentspec` 配下全体のモジュール命名・スタイルは `.claude/rules/01-python.instructions.md` の命名・スタイル規約に準拠し、`ruff`（import 整序の `I` を含む）の実行結果がエラー 0 件である。
  - [ ] IF `src/oai_agentspec` 配下のあるモジュールの配置が責務に対して不適切と判断される場合 THEN 適切なサブパッケージへ再配置し、参照箇所はすべて更新されている。

### FR-5: 公開 API（`__all__`）およびインポートパスの再設計

- ユーザーストーリー: ライブラリ保守者として、トップレベルの `__all__` および内部インポートパスを再設計したい。なぜなら、本ライブラリは未リリースで後方互換の制約がなく、構造整理に合わせて公開契約とインポート経路を最適な形へ整理できるから。
- 受け入れ基準:
  - [ ] WHEN 公開 API・インポートパスを再設計した後 THEN `import oai_agentspec` が成功し、`oai_agentspec.__all__` に掲載された全シンボルがトップレベルから import 可能である。
  - [ ] IF `__all__` の構成や個々のシンボル名・インポートパスを変更する場合 THEN 変更してよい（後方互換は不要）。ただし各シンボルが表す振る舞いは変更前と同一である。
  - [ ] WHEN 再設計後に公開 API を利用する内部箇所（`__init__.py`・各サブパッケージ・tests）を確認した THEN すべての参照が新しいインポートパスに整合し、未解決インポートが存在しない。

## 3. 非機能要件

### NFR-1: 保守性（振る舞い完全不変／純リファクタリング）

- 要件: 本作業は純リファクタリングであり、外部から観測できる振る舞い（公開 API の入出力契約、CLI の挙動、serve エンドポイントの入出力、例外種別・エラーコード、ストリーミング挙動等）を一切変更しない。公開 API のシンボル名やインポートパスが変わっても、各シンボルが表す振る舞いは同一であること。
- 計測基準: リファクタ前後で全テスト（pytest）が pass し、テストが検証する振る舞いに差分がないこと。振る舞いに関する仕様変更・新機能追加・バグ修正・ツール設定の追加を本作業に含めないこと（差分は構造変更・移動・分割のみ）。

### NFR-2: 可用性（全テスト緑の維持）

- 要件: リファクタリングの全工程を通じて、テストスイートが常に緑である状態を維持する。
- 計測基準: `pytest` の実行結果が exit code 0（全テスト pass、失敗・エラー 0 件）であること。テストの skip 件数がリファクタ前から増加しないこと。

### NFR-3: 保守性（カバレッジ維持）

- 要件: リファクタリングによりテストカバレッジを低下させない。
- 計測基準: リファクタ後のカバレッジ率がリファクタ前の計測値以上であり、かつプロジェクト既定の絶対基準 80%（`pyproject.toml` の `fail_under = 80` / pytest 規約 `--cov-fail-under=80`）を下回らないこと。計測対象・計測方法はリファクタ前と同一条件とする。

### NFR-4: 保守性（静的解析パス）

- 要件: 静的解析（ruff、プロジェクトで設定済みのツール）がエラーなくパスする。
- 計測基準: `ruff`（lint / format チェック、`select = ["E", "W", "F", "I", "B", "C4", "UP"]`、行長 100）の実行結果がエラー 0 件であること。なお、本プロジェクトには mypy が導入されていない（`pyproject.toml` に `[tool.mypy]` なし、dev 依存に mypy なし）ため、mypy の導入・実行は本作業の対象に含めない（ツール設定の追加は純リファクタの範囲外、NFR-1 参照）。

### NFR-5: 可用性（公開 API import スモークテスト）

- 要件: リファクタ後も公開 API がトップレベルから問題なく import できることをスモークテストで保証する。
- 計測基準: `import oai_agentspec` が例外なく成功し、`oai_agentspec.__all__` に列挙された全シンボル（END, START, AgentRegistry, AgentSpec, ApprovalRequired, Conversation, ConversationError, ConversationErrorCode, ConversationService, FacadeMode, HandoffConfig, HandoffEdge, HandoffGraph, NodeFn, NodeHook, NodeResults, PendingApproval, Router, PromptStore, PromptLayout, PromptTemplate, SessionInfo, SessionPolicy, StreamDelta, StreamDone, StreamError, StreamEvent, WorkflowGraph, default_input_filter, dynamic_prompt, from_specs, function_tool）の import が成功すること。

## 4. 制約事項

- 技術的制約:
  - 対象は `src/oai_agentspec` 全体であり、特定の巨大ファイルのみに限定しない。肥大・責務過多・配置不整合をファイル名にかかわらず解消する。
  - 振る舞い完全不変（純リファクタリング）であること。仕様変更・新機能追加・バグ修正・ツール設定の追加（例: mypy の導入）は本作業に含めない。
  - 全工程を通じて全テストが pass する状態を維持すること（NFR-2）。
  - FastAPI のベストプラクティス構成の適用範囲は serve 層のみとし、パッケージ全体へのドメイン層分けは強制しない。
  - tests の整理は src 構造へのミラーを方針とし、既存の `_l1` / `_l2` / `_d5` 等サフィックスや `test_hitl_*` 命名規則は維持する。`tests/_helpers/`（共通フィクスチャ・Fake* 等）は src にミラー対象を持たない補助ディレクトリのため、ミラー化対象外とする。
  - コーディング・命名・インポート順序は `.claude/rules/01-python.instructions.md` に準拠する。
- ビジネス制約:
  - 本ライブラリは未リリースのため後方互換は不要。トップレベル `__all__`・シンボル名・インポートパスは自由に変更してよい。
- スコープ外:
  - `sample_code/`（dev_tools 等）は本体と無関係な参考コードのため対象外。
  - examples / docs は本要件のスコープ外。対象を「絞らない」のはあくまで `src/oai_agentspec` 本体内のファイル選定の話であり、スコープ自体を examples / docs / sample_code へ広げるものではない。
  - 実施タイミング・ブランチ運用は本要件で規定しない（ブランチ運用はスコープ外）。

## 5. 影響範囲

- 関連コンポーネント（`src/oai_agentspec` 配下の全モジュールが対象。以下は限定列挙ではなく構成の例示）:
  - `src/oai_agentspec` 本体: トップレベル（`__init__.py`, `constants.py`, `handoffs.py`, `prompts.py`, `protocols.py`, `registry.py`, `spec.py`, `workflow.py`）、`_adapters/`（`builders.py`, `models.py`, `responses.py`, `runner.py`, `session.py`）、`cli/`（`chat.py`, `client.py`, `main.py`）、`conversation/`（`service.py`, `session.py`, `store.py`, `types.py`）、`serve/`（`app.py`, `protocol.py`, `schemas.py`）。上記に明示されないモジュールも含め、`src/oai_agentspec` 配下のすべてが分割・再配置・命名整理の対象となりうる。
  - `tests`: src 構造へのミラー化に伴い、ディレクトリ構成・テストファイルの配置が変更される（`tests/_helpers/` を除く）。
  - `pyproject.toml` の `[project.scripts]`（`oai-agentspec = "oai_agentspec.cli.main:main"`）: cli 配下の分割・再配置時にエントリポイント定義の追従が必要。
  - 公開 API（`__all__`）の利用箇所: 本リポジトリ内の `__init__.py`・各サブパッケージ・tests（リポジトリ外の利用者は未リリースのため存在しない前提）。
- 既存機能への影響:
  - 外部から観測できる振る舞いへの影響はなし（純リファクタリング、振る舞い完全不変）。
  - インポートパス・公開シンボル名の変更による影響は本リポジトリ内部に限定される（参照箇所はすべて新パスへ更新する）。
  - cli の分割・再配置を行う場合、`[project.scripts]` のエントリポイント解決先が変わるため、コンソールエントリの起動が維持されるよう定義を追従させる。
  - 未リリースのため、後方互換に関する外部利用者への影響はなし。

## 6. 用語定義

| 用語 | 定義 |
|------|------|
| 純リファクタリング | 外部から観測できる振る舞い（入出力契約・例外・エラーコード・ストリーミング挙動等）を一切変更せず、内部構造（分割・移動・命名・配置）のみを変更する作業。仕様変更・新機能・バグ修正・ツール設定の追加を含まない。 |
| 振る舞い完全不変 | リファクタ前後でテストが検証する観測可能な振る舞いに差分がない状態。公開 API のシンボル名やインポートパスが変わっても、各シンボルが表す動作は同一であることを指す。 |
| EARS | Easy Approach to Requirements Syntax。受け入れ基準を WHEN（事象発生時）/ IF（条件成立時）/ THEN（期待結果）の構文で記述する手法。 |
| 公開 API（`__all__`） | モジュールが外部に公開するシンボルを `__all__` リストで明示した契約。トップレベル `oai_agentspec.__all__` が本ライブラリの公開契約を表す。 |
| スモークテスト | システムの最も基本的な動作（ここでは `import oai_agentspec` と `__all__` 掲載シンボルの import 成功）を最小限で確認するテスト。 |
| L1 / L2 テスト | テストファイル名の `_l1` / `_l2` サフィックスが示すテスト層の区分（L1: 単体寄り、L2: 結合/統合寄り）。本要件では命名規則を変更せず維持する。 |
| `_d5` サフィックス | 既存テストファイルに付与されているテスト分類用サフィックス。本要件では命名規則を維持する。 |
| serve（FastAPI）層 | `src/oai_agentspec/serve/` 配下の HTTP/WebSocket サービング層。本要件で一般的な FastAPI プロジェクト構成（router / schemas / dependencies 等）へ整理する対象。 |
| src ミラー | tests のディレクトリ構成を src パッケージのサブパッケージ構成と一致させる配置方針。 |
| 肥大化／責務過多モジュール | 行数が多い（400 行以下を目安とする上限を超える）、または複数の独立した責務を 1 モジュールに抱えており、分割により可読性・変更容易性が向上すると判断されるモジュール。 |
