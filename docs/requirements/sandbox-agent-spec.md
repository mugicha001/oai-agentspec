# SandboxAgent 宣言的サポート（SandboxAgentSpec）

## 1. 概要
openai-agents SDK の `agents.sandbox.SandboxAgent`（`agents.Agent` の正式なサブクラスで、サンドボックス実行専用のフィールドを持つ）を、既存の `AgentSpec` / `AgentRegistry` の枠組みの中で宣言的に扱えるようにする。`SandboxAgent` は `RealtimeAgent`（`AgentBase` のみを共有する兄弟クラス）とは異なり `Agent` を継承した上位互換の形なので、独立した並列宣言ルート（`realtime/` 相当）を新設せず、`AgentSpec` を継承する `SandboxAgentSpec` を追加し、同一の `AgentRegistry` / `HandoffGraph` / `WorkflowGraph` を共用する。

## 2. 機能要件

### FR-1: SandboxAgentSpec の定義
- ユーザーストーリー: ライブラリ利用者として、`SandboxAgent` 固有フィールドを宣言的に指定したい。なぜなら、`AgentSpec` と同じ書き味でサンドボックス実行エージェントを定義したいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `SandboxAgentSpec(name=..., default_manifest=..., capabilities=..., run_as=..., base_instructions=..., <AgentSpec の既存引数>)` を生成する THEN `AgentSpec` の全フィールド（`tools` / `handoffs` / `model` / `model_settings` / `hooks` / `input_guardrails` / `output_guardrails` / `sub_agents` 等）をそのまま継承して保持する
  - [ ] IF `default_manifest` / `capabilities` / `run_as` / `base_instructions` がいずれも未指定（`None`）THEN 各フィールドは SDK 側の既定値に委ねられる（`SandboxAgentSpec` 側で SDK の既定値を再現・ハードコードしない）

### FR-2: build 時の構築先クラス分岐
- ユーザーストーリー: ライブラリ利用者として、registry から `SandboxAgentSpec` を build したら `agents.sandbox.SandboxAgent` インスタンスが得られるようにしたい。なぜなら `Runner.run` に渡す実体の型が合っていないとサンドボックス実行が機能しないから。
- 受け入れ基準:
  - [ ] WHEN `AgentRegistry.get(name)` が `SandboxAgentSpec` に対応する spec を解決する THEN ビルド処理が `agents.sandbox.SandboxAgent` を構築して返す
  - [ ] WHEN 通常の `AgentSpec` を解決する THEN 従来通り `agents.Agent` を構築する（既存の挙動を変えない）

### FR-3: 同一 AgentRegistry での混在
- ユーザーストーリー: ライブラリ利用者として、`AgentSpec` と `SandboxAgentSpec` を同じ `AgentRegistry` に混在登録し、両者間で `handoffs` / `sub_agents` を宣言したい。なぜなら `SandboxAgent` は `Agent` のサブクラスであり、通常エージェントとサンドボックスエージェントが同じハンドオフグラフに参加できて然るべきだから。
- 受け入れ基準:
  - [ ] WHEN `AgentSpec` と `SandboxAgentSpec` が A→B→A 型の循環参照を含む形で同一 registry に `handoffs`/`sub_agents` 混在登録される THEN 既存の循環ハンドオフ解決テストと同等のシナリオで例外なく解決される
  - [ ] WHEN `registry.validate()` を呼ぶ THEN `SandboxAgentSpec` 由来の `handoffs`/`sub_agents` 参照も通常の `AgentSpec` と同じ基準で検証される
  - [ ] WHEN `AgentRegistry.freeze()` / `clone()`（内部の `_copy_spec`）が `SandboxAgentSpec` インスタンスを複製する THEN `default_manifest` / `capabilities` / `run_as` / `base_instructions` を含む全フィールドが欠落せず複製後の同一クラスのインスタンスとして保持される

### FR-4: SDK 隔離の維持
- ユーザーストーリー: メンテナとして、`spec.py` に `agents.sandbox` の型を持ち込みたくない。なぜなら NFR-1（SDK 隔離）を維持し、`agents` 実体への import を `_adapters` 配下に閉じたいから。
- 受け入れ基準:
  - [ ] WHEN `SandboxAgentSpec.default_manifest` / `capabilities` / `run_as` を定義する THEN 型は `Any`（不透明型）として宣言し、`spec.py` は `agents.sandbox` を import しない
  - [ ] IF `SandboxAgent` を構築する処理を実装する THEN `_adapters` 配下でのみ `agents.sandbox` の型を import する

### FR-5: 公開 API
- ユーザーストーリー: ライブラリ利用者として、`SandboxAgentSpec` を他の宣言シンボル（`AgentSpec` 等）と同じ場所から import したい。
- 受け入れ基準:
  - [ ] WHEN `from oai_agentspec import SandboxAgentSpec` を実行する THEN import に成功する（コア `__all__` に追加する）

## 3. 非機能要件

### NFR-1: 保守性（SDK 隔離）
- 要件: `agents.sandbox` への import は `_adapters/` 配下に限定する。
- 計測基準: `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` の結果が空であること（既存の SDK 隔離チェックコマンドをそのまま満たす）。

### NFR-2: 保守性（後方互換性）
- 要件: `SandboxAgentSpec` を使わない既存の `AgentSpec` / `AgentRegistry` の振る舞いを変更しない。
- 計測基準: 既存テストスイートが全て green のまま（新規追加による既存テストの修正が発生しない）。

### NFR-3: 保守性（テストカバレッジ）
- 要件: 新規追加コードを含めてカバレッジ基準を満たす。
- 計測基準: `uv run pytest` が `fail_under = 80` を満たして成功する。

## 4. 制約事項
- 技術的制約: `pyproject.toml` は `openai-agents>=0.17.4` を既に指定しており、この最小バージョンで `agents.sandbox.SandboxAgent` が利用可能であることを確認済み（依存バージョンの引き上げは不要）。
- ビジネス制約: 実行時のサンドボックスクライアント/セッション/スナップショット設定（`RunConfig(sandbox=...)`）は本 Issue のスコープ外とする。本ライブラリの「build-don't-run」方針に従い、実行時設定は呼び出し側が SDK 標準の `RunConfig` で行う。
- スコープ外: `AgentRegistry.validate()` へのツール名重複検出（別途議論した候補）は本 Issue に含めない。必要であれば別 Issue とする。

## 5. 影響範囲
- 関連コンポーネント: `src/oai_agentspec/spec.py`（`SandboxAgentSpec` 追加）、`src/oai_agentspec/_adapters/builders.py`（構築先クラスの分岐）、`src/oai_agentspec/__init__.py`（公開 `__all__` 追加）、`src/oai_agentspec/registry.py`（`SandboxAgentSpec` 受け入れの要否確認）
- 既存機能への影響: 既存の `AgentSpec` / `AgentRegistry` の挙動・公開契約は変更しない（純粋な追加）。

## 6. 用語定義
| 用語 | 定義 |
|------|------|
| SandboxAgent | openai-agents SDK の `agents.sandbox.SandboxAgent`。`Agent` のサブクラスで、サンドボックス実行（ファイル操作・shell 等）向けの専用フィールドを持つ |
| Manifest | `agents.sandbox.manifest.Manifest`。サンドボックスのファイルシステム / env / マウント定義を持つ pydantic モデル |
| Capability | `agents.sandbox.capabilities.Capability`。サンドボックスが提供する機能（Filesystem / Shell / Compaction / Memory / Skills 等）のトグル |
| RunConfig(sandbox=...) | `Runner.run` 実行時にサンドボックスクライアント/セッション/マニフェスト上書き等を渡す SDK の実行時設定。本ライブラリの宣言層のスコープ外 |
