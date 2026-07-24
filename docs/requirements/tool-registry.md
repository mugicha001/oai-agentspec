# Tool Registry（Tool メタデータの中央集権管理基盤）

## 1. 概要

AgentRegistry と同列のコア公開 API として、Tool を中央集権的に一元管理する独立した Tool Registry を導入する。利用者は生の Python 関数とメタデータ（enabled / 権限等）を Registry に登録し、Registry が build 時に `_adapters` 経由で SDK `function_tool()` ラップとメタデータの SDK 引数（`is_enabled` / `needs_approval` / timeout 系等）への流し込みを行う。これにより Resilience 機能（ResiliencePolicy）の検討過程で顕在化した「Tool の性質と復旧方針の混在」を解消し、Tool メタデータの持ち場所を Registry に確定させる。

改訂注記: 当初 FR-2 / FR-7 に含めていた冪等性（idempotent）メタデータは、Resilience 機能のスコープ縮小（Tool 系ラッパー廃止 = 機械的な消費者の消滅）を受けたユーザー決定により**初版では導入しない**（投機的抽象の回避。将来必要になったら既定値付きフィールド追加で非破壊に再導入可能）。FR-7 は削除（欠番）とする。

## 2. 機能要件

### FR-1: 生 Python 関数とメタデータの登録
- ユーザーストーリー: ライブラリ利用者として、生の Python 関数を Tool メタデータ付きで Tool Registry に登録したい。なぜなら SDK デコレータを各定義箇所に散らばせず、Tool の性質を 1 箇所で一元管理できるからだ。
- 受け入れ基準:
  - [ ] WHEN 利用者が生の Python 関数（sync / async）とメタデータを登録する THEN Registry は宣言として保持し、この時点では SDK ラップ（`function_tool()` 呼び出し）を行わない（遅延ラップ）。
  - [ ] WHEN 同一 Tool 名で二重登録する THEN 文脈付き `ValueError` を送出する（AgentRegistry の重複登録エラーと同型）。
  - [ ] IF メタデータを一切指定しない THEN SDK `function_tool()` の既定値に委ねられる（Registry 側で SDK 既定値を再現・ハードコードしない）。

### FR-2: メタデータの宣言（初版スコープ）
- ユーザーストーリー: ライブラリ利用者として、Tool の有効/無効・承認要否・タイムアウト・失敗時エラー文言等のメタデータを型付きで宣言したい。なぜなら Tool 固有の性質を復旧方針（ResiliencePolicy）から分離し、宣言をレビュー・再利用しやすくできるからだ。
- 受け入れ基準:
  - [ ] WHEN メタデータを宣言する THEN 少なくとも `enabled`（有効/無効）・承認要否（SDK `needs_approval` 相当）・タイムアウト（SDK `timeout` / `timeout_behavior` / `timeout_error_function` 相当）・失敗時エラー文言関数（SDK `failure_error_function` 相当。Tool 失敗時にモデルへ返すエラー文字列の生成関数）・名前/説明の上書き（SDK `name_override` / `description_override` 相当）・厳格スキーマの有効/無効（SDK `strict_mode` 相当。未指定は SDK 既定に委ねる）を型付きフィールドで指定できる。
  - [ ] WHEN 失敗時エラー文言関数を宣言する THEN 「未指定（SDK 既定 formatter に委ねる）」「関数指定（当該関数で文言生成）」「None 明示（例外を文字列化せず素通しし、Runner 外へ生例外を出す）」の 3 値を区別して SDK `failure_error_function` へ委譲する（Registry 側で SDK の既定 formatter を再現・ハードコードしない）。
  - [ ] WHEN 型付きフィールドに無い `function_tool()` kwarg を渡したい THEN 素通し dict（`extra` 相当。`AgentSpec.extra` と同型思想）で指定できる。
  - [ ] IF `extra` に型付きフィールドと同名の予約キー、または `function_tool()` が受け付けない未知キーが含まれる THEN 構築時に `ValueError` を送出する（`AgentSpec.extra` の検証と同型）。
  - [ ] IF SDK にネイティブ機構が存在するメタデータ（enabled → `is_enabled`、承認要否 → `needs_approval`、タイムアウト → `timeout_*`、失敗時エラー文言 → `failure_error_function`）THEN 独自の実行時機構を作らず対応する SDK 引数へ委譲する。
  - [ ] IF レート制限を宣言しようとする THEN 初版では専用フィールドを提供しない（将来スコープ。制約事項参照）。

### FR-3: 属性アクセスによる SDK Tool の取得
- ユーザーストーリー: ライブラリ利用者として、`tool_registry.get_weather` のような属性アクセスでメタデータ適用済みの SDK Tool オブジェクトを取り出したい。なぜなら既存の `AgentSpec(tools=[...])` にそのまま渡せて、AgentSpec / AgentRegistry を一切変更せずに導入できるからだ。
- 受け入れ基準:
  - [ ] WHEN 登録済み Tool 名で属性アクセスする THEN メタデータを SDK 引数として適用済みの SDK Tool オブジェクト（`FunctionTool`）が返り、`AgentSpec.tools` にそのまま渡せる。
  - [ ] WHEN 未登録名で属性アクセスする THEN 文脈付きエラー（`AttributeError` 系。登録済み名の案内を含む）を送出する（無音失敗しない）。
  - [ ] IF 同一 Tool を複数回取得する THEN 構築済み Tool をキャッシュし同一インスタンスを返す（AgentRegistry の `_built` キャッシュと同型の遅延構築）。
  - [ ] WHEN SDK ラップ（`function_tool()` 呼び出し）が発生する THEN それは `_adapters` 経由でのみ行われる（NFR-1）。

### FR-4: enabled の SDK is_enabled 委譲と動的トグル
- ユーザーストーリー: ライブラリ利用者として、Registry 上で Tool の enabled を切り替えたら、構築済み Agent の再構築なしに次の run から LLM への Tool 提示有無に反映されてほしい。なぜなら運用中の Tool 無効化（障害時の一時停止等）を宣言の書き換えだけで行えるからだ。
- 受け入れ基準:
  - [ ] WHEN Registry が Tool を SDK ラップする THEN `enabled` は SDK `FunctionTool.is_enabled` へ Registry の現在値を参照する callable として結線される（bool の焼き込みではない）。
  - [ ] WHEN Registry 上で `enabled` を False に更新する THEN 構築済み Agent・構築済み Tool オブジェクトの再構築（invalidate 連鎖）なしに、次の実行時判定から当該 Tool が LLM から隠される（SDK `is_enabled` のネイティブ挙動に委譲）。
  - [ ] WHEN `enabled` を True へ戻す THEN 同様に再構築なしで Tool が再提示される。

### FR-5: メタデータの照会 API
- ユーザーストーリー: ライブラリ利用者・運用コードとして、Tool 名からメタデータ（enabled 等）を照会したい。なぜなら Tool の性質・状態を Registry から読み取って運用判断（一覧確認・トグル前の状態確認等）ができるからだ。
- 受け入れ基準:
  - [ ] WHEN 登録済み Tool 名でメタデータを照会する THEN 登録時に宣言したメタデータ（動的更新後は最新値）が plain な値として返る。
  - [ ] WHEN 未登録名で照会する THEN 文脈付きエラーを送出する。
  - [ ] WHEN 登録済み Tool 名の一覧を要求する THEN 昇順の名前リストが返る（`AgentRegistry.names()` と同型）。

### FR-6: メタデータの動的更新
- ユーザーストーリー: ライブラリ利用者として、登録後に Tool のメタデータを更新したい。なぜなら運用中に Tool の性質・可用性が変わった際、再登録・再構築なしで宣言を追随させられるからだ。
- 受け入れ基準:
  - [ ] WHEN 登録済み Tool のメタデータを更新する THEN 以後の照会（FR-5）は更新後の値を返す。
  - [ ] IF 更新対象が `enabled` THEN FR-4 の callable 結線により構築済み Tool へ再構築なしで反映される。
  - [ ] IF 更新対象が `enabled` 以外のメタデータ（承認要否・タイムアウト・失敗時エラー文言関数・名前/説明上書き・`extra` 等）THEN 構築済み（キャッシュ済み）SDK Tool オブジェクトへは反映されない（更新は照会 API（FR-5）の返却値にのみ反映される。SDK 引数への値は構築時に確定し、invalidate・再構築の機構は設けない）。
  - [ ] WHEN 未登録名を更新しようとする THEN 文脈付きエラーを送出する。

### FR-7: 削除（欠番）
- 当初は「Resilience 機能との接続契約（冪等性の Registry 一本化）」を定めていたが、Resilience 機能のスコープ縮小（Tool 系ラッパー廃止）により冪等性の機械的な消費者が消滅したため、冪等性メタデータごと初版から削除した（概要の改訂注記参照）。将来冪等性を必要とする機能が現れた場合は、「消費者 → Tool Registry の一方向照会・未登録は安全側既定（False）」の契約形で再導入する。

## 3. 非機能要件

### NFR-1: 保守性（SDK 隔離）
- 要件: Tool Registry のコア部分（宣言・保持・照会）は `agents` 非依存とし、`function_tool()` ラップ等の SDK 結合は `_adapters` 配下に閉じる。
- 計測基準: `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` の結果が空であること。

### NFR-2: 保守性（単方向依存）
- 要件: Tool Registry はコア層に配置し、既存の単方向依存（コア公開 API → 各層 → `_adapters` → `agents`。コアから `runtime/` への依存辺なし）を維持する。
- 計測基準: コアモジュールから `runtime/` 配下への import が存在しないこと（既存の依存方向検証テスト / コードレビューで確認）。

### NFR-3: 保守性（公開契約）
- 要件: Tool Registry のシンボルをコア `__all__` へ追加する（公開契約の拡張。要件ヒアリングでの回答をもって実装前のユーザー合意とする）。既存の `__all__` メンバ集合・既存シンボルの振る舞いは不変とする。
- 計測基準: `uv run python -c "import oai_agentspec as m; assert all(hasattr(m,s) for s in m.__all__)"` が成功し、既存シンボルのテストが全て緑のままであること。

### NFR-4: 保守性（純粋追加・既存挙動不変）
- 要件: Tool Registry を使用しない既存コード（`AgentSpec.tools` へ SDK Tool を直接渡す経路）の挙動は一切変わらない。`AgentSpec` / `AgentRegistry` の変更は行わない。
- 計測基準: 既存テストスイートが変更なしで全通過すること。`spec.py` / `registry.py` への差分が無いこと。

### NFR-5: 保守性（テスト品質）
- 要件: 新規モジュールを含めテストカバレッジを維持する。
- 計測基準: `uv run pytest` が `fail_under = 80` を満たして緑であること。

## 4. 制約事項

- 技術的制約:
  - build-don't-run: Registry は宣言・遅延ラップ・照会に徹し、独自の実行エンジン・実行 API を持たない。実行時挙動（有効判定・承認・タイムアウト）は SDK ネイティブ機構（`is_enabled` / `needs_approval` / `timeout_*`）への委譲で実現する。
  - レート制限は初版スコープ外とする。SDK 調査（`function_tool()` 全引数・`FunctionTool` 全フィールドの実測）でレート制限のネイティブ機構が存在しないことを確認済みであり、実現には `on_invoke_tool` ラップ内の独自実行制御が必要となって上記原則と緊張するため、将来スコープへ送る。将来導入時は実現方式の制約（独自実行制御の最小化）を別途要件化する。
  - Tool 識別子のキー設計（SDK `FunctionTool.name` を使うか、namespace 込みの `qualified_name` を使うか）は設計フェーズに委ねる。
  - Registry 名称（`ToolRegistry` 等）・API シンボル名・メタデータ dataclass 名は本要件書では仮称であり、設計フェーズで確定する。
  - プロンプト文字列のハードコード禁止・mypy 非導入等の既存リポジトリ方針に従う。
- ビジネス制約:
  - 本ライブラリは未リリース Alpha であり後方互換は必須でないが、公開契約（コア `__all__`）の拡張は要件ヒアリングでの合意を実装前合意とする。

## 5. 影響範囲

- 関連コンポーネント:
  - 新規: コア層の Tool Registry モジュール（`src/oai_agentspec/tool_registry.py` 等・仮）・メタデータ宣言 dataclass。
  - 新規: `_adapters` 配下の `function_tool()` ラップ結線（メタデータ → SDK 引数の流し込み・is_enabled callable 結線）。
  - 変更: `src/oai_agentspec/__init__.py`（コア `__all__` へのシンボル追加）。
  - 変更なし: `spec.py`（AgentSpec）・`registry.py`（AgentRegistry）。Tool Registry は完全独立であり、opt-in 注入点も設けない（橋渡しは利用者コードが `AgentSpec.tools=[tool_registry.<name>]` で行う）。
- 既存機能への影響:
  - Tool Registry 未使用時の既存挙動は不変（NFR-4）。
  - Resilience 機能（ResiliencePolicy・設計承認済み・実装前）はスコープ縮小済み（Tool 系ラッパー廃止・Agent/Runner 境界のリトライ戦略へ純化）。冪等性メタデータの廃止に伴い、本要件と Resilience 機能の間の接続契約は存在しない。Resilience 機能の設計方針ドキュメントの更新は同機能の再開時に行う。
  - `docs/architecture.md`（コア層の節・公開 API 表）への追記が必要。

## 6. 用語定義

| 用語 | 定義 |
|------|------|
| Tool Registry | Tool の宣言（生関数 + メタデータ）を一元管理し、遅延 SDK ラップ・照会・動的更新を提供するコア公開 API（名称は仮称） |
| メタデータ | Tool 固有の性質・設定の宣言。enabled / 承認要否 / タイムアウト / 失敗時エラー文言関数（failure_error_function）/ 名前・説明上書き / 厳格スキーマ（strict_mode）/ extra 素通し |
| failure_error_function | Tool 失敗時にモデルへ返すエラー文字列を生成する関数（SDK `function_tool()` 引数）。未指定 = SDK 既定 formatter・関数指定 = 当該関数で生成・None 明示 = 例外を文字列化せず Runner 外へ素通し、の 3 値を区別する。run 全体共有の `RunConfig.tool_error_formatter` に対する Tool 単位の個別指定にあたる |
| 遅延ラップ | 登録時ではなく取得時に `_adapters` 経由で `function_tool()` を呼び SDK Tool を構築すること（AgentRegistry の遅延構築と同型） |
| is_enabled 委譲 | Registry の enabled を SDK `FunctionTool.is_enabled` へ「Registry 現在値を参照する callable」として結線し、実行時の有効判定を SDK に委ねる方式 |
| qualified_name | SDK `FunctionTool` の namespace 込み公開識別名。Registry キー設計の設計フェーズ論点 |
