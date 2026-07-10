# Realtime エージェント専用の宣言型・registry・公開窓口

## 1. 概要
oai-agentspec に、OpenAI Agents SDK の `RealtimeAgent`（音声エージェント）を宣言的に扱うための**専用ルート**を追加する。通常の `AgentSpec` / `AgentRegistry` と共用せず、RealtimeAgent が非対応とするフィールドを**そもそも型として持たない**専用宣言型 `RealtimeAgentSpec` と、専用の registry・handoff 結線・専用公開窓口（`oai_agentspec.realtime`）を新設する。これにより「宣言と実行の分離」「宣言的定義」という agentspec の流儀を Realtime でも維持しつつ、非対応フィールドは型レベルで排除し、型で排除しきれない経路のみ build-time に reject する。

## 2. 機能要件

### FR-1: RealtimeAgent 専用の宣言型 `RealtimeAgentSpec` の提供
- ユーザーストーリー: 開発者として、RealtimeAgent が対応するフィールドだけを持つ専用宣言型で音声エージェントを宣言したい。なぜなら非対応フィールドを型レベルで排除でき、通常 Agent と混同せず宣言できるからだ。
- 受け入れ基準:
  - [ ] WHEN `RealtimeAgentSpec` を定義する THEN RealtimeAgent が対応するフィールド（`name` / `instructions` / `prompt` / `tools` / `hooks` / `output_guardrails`、および AgentBase 由来の `handoff_description` / `mcp_servers` / `mcp_config`）のみを持つ。
  - [ ] IF RealtimeAgent 非対応フィールド（`model` / `model_settings` / `input_guardrails` / `sub_agents` / `dynamic_handoffs`）を宣言型に指定しようとする THEN 型として存在しないため受け付けない（型レベル排除。`dataclasses.fields(RealtimeAgentSpec)` に非対応フィールド名が含まれないことをテストで確認する）。
  - [ ] WHEN `RealtimeAgentSpec` を定義する THEN `agents` へ依存しない純データ（dataclass）として保持する（SDK 隔離・NFR-1）。
  - [ ] WHEN `extra` に SDK 素通し用の kwarg を指定する THEN 専用フィールドと同名キー、または RealtimeAgent/AgentBase が受け付けないキーは build 時に reject する（FR-4 参照）。

### FR-2: `RealtimeAgentSpec` から `RealtimeAgent` を構築する専用アダプタ
- ユーザーストーリー: 開発者として、`RealtimeAgentSpec` から `RealtimeAgent` を構築したい。なぜなら宣言した設定を実行時に確実に反映させたいからだ。
- 受け入れ基準:
  - [ ] WHEN 有効な `RealtimeAgentSpec` を専用ビルダーに渡す THEN 対応フィールドをマップした `RealtimeAgent` が返る。
  - [ ] WHEN `instructions` が文字列または `(context, agent)` callable である THEN そのまま `RealtimeAgent` へ渡る（RealtimeAgent は両形式を受理）。
  - [ ] IF フィールドが未指定（None / 空）である THEN RealtimeAgent の既定値に委ね、明示的に None を渡さない。
  - [ ] WHEN `RealtimeAgent` を構築する THEN 実行（`RealtimeRunner` 起動）は行わず build のみに徹する（build-don't-run）。
  - [ ] IF アダプタが SDK 型（`RealtimeAgent` / `realtime_handoff`）を参照する THEN その import は `_adapters/` 配下に閉じる（NFR-1）。

### FR-3: RealtimeAgent 専用 registry と handoff 結線
- ユーザーストーリー: 開発者として、宣言した `handoffs`（エージェント名参照）を Realtime でも結線したい。なぜなら通常 Agent と同様にグラフ連携で音声エージェントを分割したいからだ。
- 受け入れ基準:
  - [ ] WHEN `RealtimeAgentSpec` を専用 registry に登録し get する THEN 遅延構築で `RealtimeAgent` を返す。
  - [ ] WHEN spec に `handoffs`（エージェント名リスト）が宣言される THEN 専用 registry が 2 パス遅延バインドで `realtime_handoff()` により結線する（既存 `AgentRegistry` を拡張せず RealtimeAgent 専用に実装する）。
  - [ ] WHEN 循環ハンドオフ（相互参照）が宣言される THEN 既存 registry と同様に遅延バインドで解決する。
  - [ ] IF 未登録のエージェント名が `handoffs` に含まれる THEN validate 時にエラーを返す。
  - [ ] IF 同名エージェントを専用 registry に重複登録しようとする THEN エラーを返す（既存 `AgentRegistry` と同等の挙動を専用 registry でも新規実装する）。
  - [ ] WHEN handoff 宣言（`RealtimeHandoffConfig`）を定義する THEN `on_handoff` / `input_type` / `tool_name_override` / `tool_description_override` / `is_enabled` を保持する（`input_filter` は型として持たない）。

### FR-4: 型で排除しきれない経路への reject バリデーション
- ユーザーストーリー: 開発者として、型レベルで排除しきれない不正指定は build 時にエラーを受け取りたい。なぜなら silently 無視されて実行時に設定が効かない事故を防ぎたいからだ。
- 受け入れ基準:
  - [ ] IF `extra` に専用フィールドと同名のキー、または RealtimeAgent/AgentBase が受け付けないキー（例: `model` / `model_settings` / `output_type` / `tool_use_behavior` / `input_guardrails`）が含まれる THEN `ValueError` を送出する。
  - [ ] IF handoff オプションに `realtime_handoff()` 非対応の `input_filter` が指定される THEN `ValueError` を送出する。
  - [ ] WHEN reject が発生する THEN エラーメッセージに agent 名と該当キー／フィールド名を含める。

### FR-5: RealtimeAgent 専用の公開窓口
- ユーザーストーリー: 開発者として、Realtime 関連シンボルを専用窓口から取得したい。なぜなら通常 Agent と一緒に使わない前提であり、コアの公開契約を汚さず専用ルートで扱いたいからだ。
- 受け入れ基準:
  - [ ] WHEN Realtime を使う THEN `oai_agentspec.realtime` サブモジュール公開窓口から `RealtimeAgentSpec` / 専用 registry 等を取得できる（`runtime.conversation` と同様の公開窓口方式）。
  - [ ] WHEN 本機能を追加する THEN コアの `oai_agentspec.__all__` は不変を保つ（Realtime シンボルはコア `__all__` に載せない）。
  - [ ] WHEN 本機能を追加する THEN 新規 extra は設けない（Realtime は既存 `agents` SDK に含まれるため追加依存なし。分離はコア `__all__` からの分離＝専用窓口経由で表現する）。
  - [ ] WHEN `oai_agentspec.realtime` を import しない THEN 既存の `import oai_agentspec` の挙動は不変（遅延 import 境界を維持）。

## 3. 非機能要件

### NFR-1: セキュリティ / 保守性（SDK 隔離）
- 要件: `RealtimeAgent` / `realtime_handoff` 等の `from agents ...` import はすべて `_adapters/` 配下に閉じる。`RealtimeAgentSpec` / 専用 registry / 専用公開窓口は plain データと不透明型のみ扱う。
- 計測基準: `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` の結果が空であること。

### NFR-2: 保守性（公開契約の不変）
- 要件: コアの `oai_agentspec.__all__` とコア公開シンボルの集合を変更しない。Realtime シンボルは `oai_agentspec.realtime` 公開窓口経由でのみ提供する。
- 計測基準: 公開 API スモーク（`__all__` 全件 import 可能）が緑で、`__all__` のメンバ集合が改訂前と一致すること。

### NFR-3: 保守性（単方向依存・薄い __init__・遅延 import 境界）
- 要件: 依存辺は既存アーキテクチャに整合させる。専用宣言型は最下層（`agents` 非依存）、専用 registry は `_adapters` / 宣言型を上向きに参照する単方向とし、コアから Realtime 層への依存辺は持たない。公開窓口の `__init__` は再エクスポート専用に保ち、extra 未導入耐性が必要な場合は関数内遅延 import を用いる。
- 計測基準: 単方向依存が保たれること（レビューで確認）、`import oai_agentspec` の挙動が改訂前と不変であること。

### NFR-4: 保守性（テスト）
- 要件: 追加コードにユニットテストを付与し、マッピング成功系・handoff 結線・各 reject 系・公開窓口の import を網羅する。テストは `tests/` の src ミラー構造に配置し、既存ルートと同様の 2 層構成とする:
  - L1: build 結果の introspection 検証（構築された `RealtimeAgent` のフィールド・`handoffs` の内容・循環解決）
  - L2: `RealtimeRunner` + フェイク `RealtimeModel` によるハンドオフ実委譲の検証。既存 `FakeModel`（`Model.get_response` 抽象）は Realtime に流用できないため、`RealtimeModel` 抽象（`connect` / `add_listener` / `send_event` / `close`）を実装する `FakeRealtimeModel` を `tests/_helpers/` に新設する
- 計測基準: `uv run pytest` が緑、カバレッジ 80% 以上（`fail_under = 80`）を維持。L1 / L2 の両層にテストが存在すること。

### NFR-5: 保守性（build-don't-run / 既存パターン踏襲）
- 要件: 実行エンジンや公開実行 API を追加せず、宣言・build-time 検証・薄い結線に徹する。既存 `AgentSpec` / `AgentRegistry` / `build_agent` の設計思想（専用フィールド + extra 素通し + 早期検証 + 2 パス遅延バインド）を踏襲する。
- 計測基準: 公開の実行 API を追加していないこと（レビューで確認）。

## 4. 制約事項
- 技術的制約:
  - SDK 隔離（NFR-1）により `RealtimeAgent` / `realtime_handoff` 参照は `_adapters/` のみ。
  - RealtimeAgent は `model` / `model_settings` / `output_type` / `tool_use_behavior` を非対応、`input_guardrails` フィールドを持たない（`output_guardrails` のみ。SDK docstring と `AgentBase` 定義で確認済み）。専用宣言型はこれらを型として持たない。
  - `realtime_handoff()` は `input_filter` 非対応（`on_handoff` / `input_type` / `tool_name_override` / `tool_description_override` / `is_enabled` のみ）。
  - `mypy` 非導入・静的解析は ruff（`S`/bandit なし）。
  - Realtime 専用の handoff 宣言型は通常ルートの `HandoffConfig` / `DynamicHandoff` とは別物として定義し（`input_filter` を型として持たない）、既存のフィールド対称性テストの対象外とする。
  - 新規 extra は設けない（Realtime は既存 `agents` SDK の `agents.realtime` に含まれ追加の外部依存がないため。`pyproject.toml` は変更しない）。
- ビジネス制約:
  - 成果物は本リポジトリ（oai-agentspec）への機能追加。
  - 通常 Agent と共用しない完全専用ルートとする。

## 5. 影響範囲
- 関連コンポーネント（新設・想定配置。最終配置は設計フェーズで確定）:
  - `src/oai_agentspec/realtime/`（Realtime 専用サブパッケージ・公開窓口。薄い `__init__.py` で再エクスポート）
    - 宣言型 `RealtimeAgentSpec` / handoff 宣言（例: `RealtimeHandoffConfig`。`input_filter` を持たない）
    - 専用 registry（`RealtimeAgentRegistry` 等・遅延構築・handoff 結線・validate）
  - `src/oai_agentspec/_adapters/`（`RealtimeAgent` / `realtime_handoff` を用いる専用ビルダー・SDK 結合をここに閉じる。既存 `builders.py` へ追加または `_adapters/realtime.py` を新設）
  - `docs/architecture.md`（Realtime 専用ルートの記述追記・`/spec-sync` 対象）
  - `tests/`（`realtime/` と `_adapters/` のミラーへ L1 / L2 テスト追加。`tests/_helpers/` に `FakeRealtimeModel` を新設）
- 既存機能への影響:
  - 既存 `AgentSpec` / `AgentRegistry` / `build_agent` / workflow の挙動は不変（純追加）。
  - コアの公開契約（`oai_agentspec.__all__`）は不変。

## 6. 用語定義
| 用語 | 定義 |
|------|------|
| RealtimeAgent | OpenAI Agents SDK の音声エージェント型（`agents.realtime.RealtimeAgent`）。`AgentBase` を継承し `model` 等を非対応とする |
| RealtimeAgentSpec | 本機能で追加する RealtimeAgent 専用の宣言型。非対応フィールドを型として持たない |
| 専用 registry | RealtimeAgentSpec 用の遅延構築・handoff 結線・validate を担う専用レジストリ（既存 AgentRegistry を拡張しない） |
| realtime_handoff | RealtimeAgent 間ハンドオフ用の SDK ヘルパ（`input_filter` 非対応） |
| 公開窓口 | `oai_agentspec.realtime` サブモジュールとして提供する再エクスポート専用の入口（`runtime.conversation` と同様） |
| SDK 隔離（NFR-1） | `from agents`/`import agents` を `_adapters/` 配下に限定する不変条件 |
| build-don't-run | lib は宣言・build 検証・薄い結線に徹し実行は SDK に委ねる方針 |
| FakeRealtimeModel | L2 テスト用に新設する `RealtimeModel` 抽象のフェイク実装（実 WebSocket 接続なしで `RealtimeRunner` を駆動する） |
</content>
</invoke>
