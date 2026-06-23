# compaction 明示有効化フローの整備と OpenAI Responses API クライアント明示注入の標準化

## 1. 概要

OpenAI Responses API クライアント（`AsyncOpenAI` / `AsyncAzureOpenAI`）の明示注入と、SDK の履歴圧縮（compaction）有効化フローを整備する。現状 compaction は「`make_session(compaction=dict)` に `client` を渡すと有効化」という暗黙の有効化方式であり、クライアントの受け渡しと有効化判定が分離されていない。本機能では「専用の明示フラグで compaction の有効化を制御し、client/model の受け渡しと有効化判定を分離する」ことを定義し、あわせて 2 つの注入経路（モデル経由 / compaction 経由）を docs・examples・API シグネチャの 3 面で標準化する。本 Issue は Issue #16（純リファクタ・振る舞い完全不変）完了後に実施する、振る舞い変更を伴う機能改善である。

## 2. 機能要件

### FR-1: compaction の明示フラグによる有効化制御

- ユーザーストーリー: ライブラリ利用者として、client/model を渡すこととは独立した明示フラグで compaction の有効化を制御したい。なぜなら client/model は compaction 以外の目的でも渡される可能性があり、「client/model を渡した = 自動で compaction ON」という暗黙の挙動を避け、有効化の意図を明示的に表現したいから。
- 補足（設計論点）: 現状の `compaction` dict（`client` 必須）と「明示フラグ」の関係整理が必要である。具体 API 形（例: `compaction=True/False` の bool フラグ + client/model を別引数で受ける、あるいは型付き設定オブジェクトに有効化フラグを持たせる等）は設計フェーズで詰める。要件としては「明示フラグで有効化を制御し、client/model の受け渡しと有効化判定を分離する」ことを定義する。
- 受け入れ基準:
  - [ ] WHEN 利用者が compaction 有効化フラグを ON にし、かつ client/model を渡した THEN `make_session` は `OpenAIResponsesCompactionSession` でラップした `Session` を返す
  - [ ] WHEN 利用者が compaction 有効化フラグを OFF（または未設定）にした THEN `make_session` は client/model の指定有無にかかわらず plain `SQLiteSession` を返す（現状どおり）
  - [ ] IF compaction 有効化フラグが ON だが compaction に必要な client が欠けている THEN `make_session` は `ValueError`（現行と同等の明示的な失敗）を送出する
  - [ ] WHEN client/model のみが渡され有効化フラグが指定されていない THEN compaction は ON にならない（暗黙有効化を行わない）
  - [ ] WHEN 設計フェーズで具体 API 形が確定する THEN client/model の受け渡し口と有効化判定が分離された契約として `make_session` / `SessionPolicy` のシグネチャに反映される（具体形は設計フェーズで詰める）

### FR-2: `make_session` の compaction 設定の型付き整理（API シグネチャ整理）

- ユーザーストーリー: ライブラリ利用者として、compaction 設定を任意キーの `dict[str, Any]` ではなく型付き引数 / dataclass 等で受け取りたい。なぜなら注入契約（必須値・任意値・有効化フラグ）が型レベルで明確になり、誤用を実行前に検知できるから。
- 補足: 現行は `compaction: dict[str, Any] | None`（`client` を pop し残りを `OpenAIResponsesCompactionSession` へ素通し）。型付き引数 / dataclass / 専用設定型のいずれを採るかは設計フェーズで詰める。本体の env 非依存（NFR-3）と SDK 隔離（NFR-1）を崩さない。
- 受け入れ基準:
  - [ ] WHEN compaction 設定を `make_session` / `SessionPolicy` に渡す THEN 型付きの契約（有効化フラグ・client・model 等）で受け取れる
  - [ ] IF 必須要素（有効化時の client）が欠けている THEN 型 / バリデーションで誤用を明示的に拒否する（`ValueError` 等）
  - [ ] WHEN 整理後のシグネチャを `_adapters/session.py` の docstring に反映する THEN Args / Returns / Raises（Google スタイル・日本語）に新契約が記述される
  - [ ] WHEN 型付き設定を導入する THEN `OpenAIResponsesCompactionSession` へ渡す追加オプション（model 等）の素通し経路が維持される

### FR-3: ConversationService / start_server レベルのクライアント受け渡し口の検討・整備

- ユーザーストーリー: ライブラリ利用者として、`ConversationService` / `start_server` のレベルで compaction 用クライアントを受け渡せる口があるかを確認し、必要なら整備したい。なぜなら現状 compaction 設定は `SessionPolicy.compaction` 経由でのみ伝播し、新しい明示フラグ方式に合わせて上位の受け渡し口を整える必要があるから。
- 補足（DI 方針整合の論点）: architecture.md の DI 拡張点は「生成 = `AgentBuilder`」「実行 = runner シーム（非公開）」の 2 点に限定されている。compaction 用クライアントの受け渡し口を `SessionPolicy` の枠内に収めるか、新たな注入口を設けるかは DI 方針との整合を要件レビューで問う。現状の伝播経路は `ConversationService(session_policy=SessionPolicy(compaction=...))` および `start_server(session_policy=...)`（registry 渡し時のみ適用）である。具体形は設計フェーズで詰める。
- 受け入れ基準:
  - [ ] WHEN 利用者が `ConversationService` を生成する THEN compaction の有効化フラグと client/model を `SessionPolicy`（または新設の受け渡し口）経由で渡せる
  - [ ] WHEN 利用者が `start_server` に `AgentRegistry` を渡す THEN compaction の有効化フラグと client/model が内部生成される `ConversationService` に適用される（現行の `session_policy` 適用範囲を踏襲）
  - [ ] IF 新たな注入口を設ける THEN architecture.md の DI 拡張点方針（生成 = `AgentBuilder` / 実行 = runner シーム）との整合が要件レビューで判断され、逸脱しないこと（具体形は設計フェーズで詰める）
  - [ ] WHEN `start_server` に構築済み `ConversationService` を直接渡す THEN session_policy 由来の compaction 設定は無視される（現行の `session_policy` 無視仕様を踏襲）

### FR-4: docs / README への注入パターンの明示

- ユーザーストーリー: ライブラリ利用者として、クライアント明示注入の推奨パターンを docs / README で参照したい。なぜなら現状 (a) モデル経由 / (b) compaction 経由の 2 経路があり、どちらをどう使うかが文書化されていないと利用者が判断できないから。
- 補足: (a) モデル経由 = `AgentSpec.model = OpenAIResponsesModel(openai_client=AsyncOpenAI(...))`（`examples/_shared/_azure.py` の `azure_model()` で実証済み）。(b) compaction 経由 = `make_session` / `SessionPolicy` 経由のクライアント注入。docs 規約（`05-docs.instructions.md`）に従い、現在の仕様のみを記述し、PR / Issue 番号・履歴記述は書かない。
- 受け入れ基準:
  - [ ] WHEN docs/architecture.md の compaction 記述（「会話 Helper / 履歴と session」節付近）を更新する THEN 明示フラグによる有効化方式が反映され、暗黙有効化の記述が現行仕様に置き換わる
  - [ ] WHEN docs / README を更新する THEN (a) モデル経由 / (b) compaction 経由の注入パターンが推奨例として明示される
  - [ ] WHEN docs を更新する THEN compaction が OpenAI Responses API 専用であり、クライアントは `AsyncOpenAI` / `AsyncAzureOpenAI` のいずれでも Responses API を叩ければ動く旨が記述される
  - [ ] IF docs 編集時に履歴記述（PR 番号 / 過去の挙動 / 旧方式比較）を書こうとする THEN docs 規約に従い記述しない

### FR-5: compaction を外部クライアントで有効化する conversation example の追加

- ユーザーストーリー: ライブラリ利用者として、外部クライアントで compaction を有効化する実行可能な example を参照したい。なぜなら現状 examples/conversation では compaction が未使用であり、(b) compaction 経由の注入を動く形で確認できないから。
- 補足: 既存の `examples/conversation/01_inprocess.py`（`ConversationService` 直接利用・`azure_model()` 注入）と `examples/_shared/_azure.py`（`_azure_client()` パターン）に倣う。compaction は Responses API 専用のため、example で使うクライアントも Responses API を叩ける構成とする。
- 受け入れ基準:
  - [ ] WHEN 新しい conversation example を追加する THEN 明示フラグ ON と外部クライアント注入で compaction を有効化する手順が示される
  - [ ] WHEN example を実行する THEN `ConversationService`（または `start_server`）経由で compaction 有効の会話が動作する
  - [ ] WHEN example が Azure 互換クライアントを使う THEN `examples/_shared/_azure.py` の既存パターン（v1 preview エンドポイント + `AsyncOpenAI`、または `AsyncAzureOpenAI`）を再利用する
  - [ ] WHEN example を追加する THEN モジュール docstring（日本語）に実行方法と前提環境変数が記述される

## 3. 非機能要件

### NFR-1: 保守性（SDK 隔離の維持）
- 要件: `from agents` / `from openai` / `import agents` の import は `_adapters/` 配下のみに閉じる。compaction の明示フラグ・型付き設定の追加でも、SDK 型（`OpenAIResponsesCompactionSession` / `Session` / `SQLiteSession`）と `openai` クライアント型への結合は `_adapters/session.py` に局在させる。上位層（`conversation/` / `serve/` / `cli/`）は plain なデータと不透明 `Session` のみを扱う。`SessionPolicy` が `openai` 型をランタイム import する必要が生じる場合は `TYPE_CHECKING` ブロックで扱う。
- 計測基準: `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` の結果が空。`openai` 直接 import についても同様に `_adapters/` 配下のみであることを grep で確認。

### NFR-2: 保守性（DI 拡張点方針との整合）
- 要件: 新たなクライアント受け渡し口を設ける場合も、architecture.md の DI 拡張点（生成 = `AgentBuilder` / 実行 = runner シーム（非公開）の 2 点限定）の方針と整合させる。compaction 用クライアント注入を新たな公開 DI 拡張点として無制限に増やさず、`SessionPolicy` の枠内に収めるか否かを設計フェーズで判断する。
- 計測基準: architecture.md の「SDK 隔離と依存性注入（DI）」節の方針と矛盾しないこと（要件レビューおよび architecture.md 更新で確認）。

### NFR-3: 保守性（本体の env 非依存維持）
- 要件: 本体（`SessionPolicy` / `ConversationService` / `serve` / `_adapters`）は環境変数に依存しない。env 解決は CLI 境界（`cli/main.py:_build_session_policy` の `XDG_DATA_HOME` 等）に閉じる。compaction クライアント / モデルの値解決を本体に env 依存として持ち込まない。
- 計測基準: `src/oai_agentspec/` 配下で `os.environ` / env 参照が CLI 境界（`cli/`）以外に新規導入されていないことを grep / レビューで確認。

### NFR-4: 保守性（コードスタイル・型）
- 要件: `01-python.instructions.md` に準拠する（行長 100、全関数に型注釈、`X | None`、`list` / `dict` 小文字、`from __future__ import annotations`、Google スタイル日本語 docstring、不変設定は `@dataclass(frozen=True)`、有効化フラグ等の列挙が必要なら `Enum`）。ruff（`select = ["E","W","F","I","B","C4","UP"]`）が通ること。
- 計測基準: `ruff check` / `ruff format --check` がエラー 0。

### NFR-5: 保守性（テスト緑・カバレッジ）
- 要件: 既存テストが緑のまま、本機能の振る舞い（明示フラグ ON/OFF・client 欠落時の `ValueError`・型付き設定の受理 / 拒否・上位層からの伝播）を新規テストでカバーする。テストは `FakeModel` / トレーシング無効化 / ネットワークガード方針に従い、実 API へ接続しない。
- 計測基準: 全テスト pass。`_adapters` を含む行カバレッジ 80% 以上を維持。

### NFR-6: 保守性（既存契約への影響明示）
- 要件: 本ライブラリは未リリースのため後方互換は不要。ただし、現行 `make_session(compaction=dict)` 利用箇所（`conversation/service.py` の `make_session` 呼び出し・`SessionPolicy.compaction`・`start_server` / `ConversationService` の `session_policy` 経由伝播）への影響範囲を要件・設計で明示し、移行を一括で行う。
- 計測基準: 影響を受ける既存呼び出し箇所（後述「5. 影響範囲」）が新シグネチャへ更新され、参照漏れがない（grep で旧 `compaction=dict` 形が残らない）こと。

### NFR-7: 可用性（Azure 互換の実 API 検証）
- 要件: compaction は OpenAI Responses API 専用を維持する。クライアントは `AsyncAzureOpenAI` でも `AsyncOpenAI`（v1 preview エンドポイント）でも、Responses API を叩ければ動くことを実装段階で実 API 検証する（`_azure.py` の既存パターン参照）。Azure 固有の追加実装検証はスコープに含めてよいが、「Responses API 専用」制約自体は変えない。
- 計測基準: Azure 互換クライアント（`_azure.py` のいずれかの方式）で compaction 有効の会話が実 API 上で成立することを実装段階で確認（手動 / example 実行記録）。

## 4. 制約事項

- 技術的制約:
  - compaction は OpenAI Responses API 専用を維持する。クライアント実体（`AsyncOpenAI` / `AsyncAzureOpenAI`）は問わないが、Responses API を叩ける前提を変更しない。
  - 本 Issue は Issue #16（純リファクタ・振る舞い完全不変）完了後に、#16 で整理された `_adapters/session.py`（compaction が `make_session` に局所化される構造）の上で実施する。
  - SDK 隔離（NFR-1）・本体 env 非依存（NFR-3）・DI 拡張点方針（NFR-2）を逸脱しない。
  - 未確定の具体 API 形（明示フラグの表現方法・型付き設定の形・上位受け渡し口の有無）は設計フェーズで詰める。確定済み要件（明示フラグで有効化を制御し client/model の受け渡しと有効化判定を分離する、注入標準化の 4 点、Responses API 専用維持）は変更しない。
- ビジネス制約:
  - 本ライブラリは未リリースのため後方互換は不要。ただし既存 examples・テスト・現行 `make_session(compaction=dict)` 利用箇所への影響範囲は要件で明示し、一括移行する。

## 5. 影響範囲

- 関連コンポーネント:
  - `src/oai_agentspec/_adapters/session.py`: `make_session` の compaction 実装。明示フラグ・型付き設定への変更主対象。
  - `src/oai_agentspec/conversation/session.py`: `SessionPolicy.compaction`（`dict[str, Any] | None`）の型・契約整理。
  - `src/oai_agentspec/conversation/service.py`: `make_session(..., compaction=self._policy.compaction)` 呼び出しの更新。
  - `src/oai_agentspec/serve/app.py`: `start_server(session_policy=...)` 経由の compaction 伝播。新たなクライアント受け渡し口を設ける場合の対象。
  - `src/oai_agentspec/cli/main.py`: `_build_session_policy` での `SessionPolicy` 組み立て（compaction を CLI から扱うか否かは設計フェーズ判断。env 非依存維持）。
  - `examples/conversation/`: compaction を外部クライアントで有効化する example の新規追加。
  - `examples/_shared/_azure.py`: Azure 互換クライアントパターンの再利用（変更は最小）。
  - `docs/architecture.md`: compaction 記述（「会話 Helper / 履歴と session」節・「SDK 隔離と依存性注入（DI）」節）の更新。
  - `README`: (a) モデル経由 / (b) compaction 経由の注入パターン推奨例の明示。
- 既存機能への影響:
  - 現行 `make_session(compaction=dict)` / `SessionPolicy.compaction=dict` を使う呼び出し箇所は新シグネチャへ移行する（後方互換は取らない）。
  - compaction 未設定 / 明示フラグ OFF 時の挙動（plain `SQLiteSession`）は現状どおり不変。
  - session 一覧 / 復元 / 履歴取得（`list_sessions` / `session_history`）・HITL 中断状態同居など compaction 以外の `_adapters/session.py` 機能には変更を与えない。

## 6. 用語定義

| 用語 | 定義 |
|------|------|
| compaction | SDK が提供する会話履歴の圧縮機構。`OpenAIResponsesCompactionSession` で `Session` をラップし、履歴を圧縮しながら会話を継続する。OpenAI Responses API 専用。 |
| OpenAIResponsesCompactionSession | compaction を実現する SDK の `Session` ラッパー。`client`（`AsyncOpenAI` 系）と `model` 等を受け、base `Session`（SQLite）を包む。`_adapters/session.py` で生成。 |
| 明示注入 | クライアント / モデルを内部生成・env 解決に頼らず、利用者が外から明示的に渡す注入方式。(a) モデル経由（`AgentSpec.model = OpenAIResponsesModel(openai_client=...)`）と (b) compaction 経由（`make_session` / `SessionPolicy` 経由）の 2 経路がある。 |
| 明示フラグ | compaction の有効化を client/model の受け渡しと分離して制御する専用フラグ。具体表現（bool 等）は設計フェーズで詰める。 |
| (a) モデル経由注入 | `AgentSpec.model` に `OpenAIResponsesModel(openai_client=AsyncOpenAI(...))` を渡す注入経路。`examples/_shared/_azure.py` の `azure_model()` で実証済み。 |
| (b) compaction 経由注入 | compaction 用クライアント / モデルを `make_session` / `SessionPolicy` 経由で渡す注入経路。examples では現状未使用。 |
| SessionPolicy | session 生成方針（永続化先・揮発・compaction 設定）を保持する `@dataclass(frozen=True)`。`ConversationService` / `start_server` が受け取り `make_session` へ素通しする。 |
| SDK 隔離（NFR-1） | `from agents` / `from openai` の import を `_adapters/` 配下に閉じる設計原則。上位層は plain データと不透明型のみ扱う。 |
| DI 拡張点 | architecture.md が定める 2 つの注入点（生成 = `AgentBuilder` / 実行 = runner シーム（非公開））。compaction クライアント注入口の設置はこの方針と整合させる。 |
| EARS | Easy Approach to Requirements Syntax。受け入れ基準を WHEN（イベント駆動）/ IF（状態・分岐駆動）/ THEN（期待結果）の形式で記述する記法。 |
| Issue #16 | 本 Issue の前提となる純リファクタ Issue（振る舞い完全不変。compaction を `make_session` に局所化）。本 Issue はその完了後に実施する別単位（振る舞い変更を伴う）。 |
| Azure 互換 | `AsyncAzureOpenAI`、または v1 preview エンドポイント設定の `AsyncOpenAI` を介して Azure OpenAI の Responses API を叩く構成。`examples/_shared/_azure.py` 参照。 |

## 7. 付録: 利用イメージ（参考・非規範・設計フェーズで確定）

本付録は要件の理解を助けるための利用イメージである。**非規範（normative ではない）**であり、具体的な API 形（型名・引数・フラグの表現）は設計フェーズで確定する。確定済みなのは「明示フラグで有効化を制御し、client/model の受け渡しと有効化判定を分離する」という方針のみ（FR-1）。以下のシンボル名（`CompactionConfig` 等）は確定を意味しない。

### 現状（暗黙有効化）

```python
from openai import AsyncOpenAI
from oai_agentspec import ConversationService, SessionPolicy

client = AsyncOpenAI()  # または Azure 互換クライアント

# compaction dict に client を入れた時点で暗黙に有効化される（明示フラグがない）
policy = SessionPolicy(compaction={"client": client, "model": "gpt-4.1"})
chat = ConversationService(registry, session_policy=policy)
# compaction=None なら plain SQLite。client が dict にないと ValueError。
```

課題: 「client を渡す」と「compaction を使う」が分離できない。client を別目的で持たせたいだけでも圧縮が始まる。

### 提案イメージ（FR-1 明示フラグ + FR-2 型付き設定）

候補A: 型付き設定オブジェクト（`enabled` フラグ内蔵）

```python
from openai import AsyncOpenAI
from oai_agentspec import ConversationService, SessionPolicy, CompactionConfig  # 新設候補

client = AsyncOpenAI()
policy = SessionPolicy(
    compaction=CompactionConfig(enabled=True, client=client, model="gpt-4.1"),
)
chat = ConversationService(registry, session_policy=policy)
# enabled=False / compaction=None -> 圧縮なし（plain SQLite・現状どおり）
# enabled=True かつ client=None     -> ValueError（明示的に失敗）
```

候補B: bool フラグ + client/model を分離した引数

```python
policy = SessionPolicy(
    compaction=True,            # 明示フラグ（有効化判定）
    openai_client=client,       # client は別目的でも渡しうる
    compaction_model="gpt-4.1",
)
# compaction=False / 未設定 -> client を渡しても圧縮しない（暗黙有効化しない）
```

候補A/B のどちらを採るかは設計フェーズの判断ポイント（型契約の締まりでは候補A が FR-2 と相性が良い）。

### (a) モデル経由のクライアント注入（既存・examples で実証済み）

```python
from openai import AsyncOpenAI
from agents import OpenAIResponsesModel  # SDK 型は agents から直接
from oai_agentspec import AgentSpec

client = AsyncOpenAI(
    base_url="https://<resource>.openai.azure.com/openai/v1/",
    api_key="...",
    default_query={"api-version": "preview"},
)  # Azure 互換例
spec = AgentSpec(
    name="assistant",
    instructions="...",
    model=OpenAIResponsesModel(model="gpt-4.1", openai_client=client),  # ここで client 注入
)
```

`examples/_shared/_azure.py:azure_model()` が採るパターン。compaction とは別軸（モデル実行のクライアント）。

### FR-3（会話サービス / サーバ起動経由）

`SessionPolicy` は `ConversationService(session_policy=...)` と `start_server(registry, session_policy=...)` の両方へ流れるため、新しい注入口を増やさず `SessionPolicy` の枠内で完結させるのが第一候補（DI 拡張点方針との衝突回避・NFR-2）。

```python
from oai_agentspec.serve import start_server

start_server(build_registry(), session_policy=policy, host="127.0.0.1", port=8000)
```

### FR-5（compaction を外部 client で有効化する example のイメージ）

```python
# examples/conversation/<番号>_compaction.py（新規候補）
"""外部クライアントで compaction を有効化する会話 example。"""
import asyncio

from openai import AsyncOpenAI

from oai_agentspec import (
    AgentRegistry,
    AgentSpec,
    CompactionConfig,
    ConversationService,
    SessionPolicy,
)


def build_registry() -> AgentRegistry:
    reg = AgentRegistry()
    reg.register(AgentSpec(name="assistant", instructions="簡潔に答える"))
    return reg


async def main() -> None:
    client = AsyncOpenAI()  # Responses API を叩ければ Azure / OpenAI どちらでも可
    policy = SessionPolicy(
        compaction=CompactionConfig(enabled=True, client=client, model="gpt-4.1"),
    )
    chat = ConversationService(build_registry(), session_policy=policy)
    # ... 会話を回すと履歴が compaction される ...


if __name__ == "__main__":
    asyncio.run(main())
```
