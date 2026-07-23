# アーキテクチャ

`oai-agentspec` は openai-agents 上のラッパーライブラリである。
openai-agents の `Agent` の薄い宣言的 Wrapper（`AgentSpec`）と、ハンドオフトポロジ
（`HandoffGraph`）・サブエージェント配線を、Protocol による DI を介して `Agent` へ
遅延構築する。SDK を知っていれば `instructions=...` でそのまま書け、プロンプト合成を
使いたい場合のみ `PromptStore.compose(...)` の戻り値を渡す。

本ドキュメントは本ライブラリの現在仕様の Single Source of Truth である。

## 設計思想

- `AgentSpec` は `agents.Agent` の薄い Wrapper である。基本的に `Agent` と同じフィールドを
  持ち、加えて handoffs / sub_agents を**エージェント名で**宣言できるグラフ連携を提供する。
- プロンプト合成は `AgentSpec` の責務ではなく、`PromptStore.compose` が生成した値
  （静的 `str` または動的 callable）を `instructions` に渡す。これにより `Agent.instructions`
  （`str | callable`）と同じ使い心地になる。
- 名前参照（handoffs / sub_agents）のタイポは `registry.validate()` と遅延構築時の
  文脈付きエラーで検出する（無音事故を防ぐ）。

## レイヤー構成と依存方向

ライブラリは **コア（宣言層）** と **runtime（実行寄り層・extra 境界）** の 2 群に分かれ、依存は
コアから runtime へは流れない一方向に保つ。コアは宣言・build-time 検証・SDK 隔離窓口に徹し、
runtime（`runtime/conversation` / `runtime/serve` / `runtime/cli` / `runtime/llmops`）は会話実行・
サーバ入口・CLI クライアント・LLMOps 評価というローカル開発支援の実行寄り機能を `runtime/` 配下へ
集約する。加えて Realtime エージェントの宣言ルート（`realtime/`）はこの 2 群のいずれにも属さない第 3 の
並列宣言ルートであり、コア公開 API ツリー外・専用窓口（`oai_agentspec.realtime`）経由で提供する
（詳細は「Realtime エージェント（専用宣言ルート）」節）。

```
   CLI クライアント (別プロセス・cli extra・[project.scripts]・runtime/cli)
   │
   ┊ ネットワーク接続 (HTTP/WS。import ではない)
   ▼
   サーバ入口 (serve extra・公開 API ツリー外・runtime/serve)
   │ import 委譲
   ▼
   会話サービス (runtime/conversation 公開窓口・agents 非依存・上位利用支援層)
   │
   ┊ LLMOps 評価 (runtime/llmops 公開窓口・llmops extra + 任意 llmops-langfuse extra・agents/deepeval/langfuse 非依存・上位利用支援層)
   │
   ┊ Agent Lightning 最適化 (runtime/lightning 公開窓口・lightning extra・agents/agentlightning 非依存・上位利用支援層)
   │
   ┊ AGT ガバナンス (runtime/governance 公開窓口・governance extra・agents/agent-governance-toolkit 非依存・装飾 builder)
   │
   ┊ 意図予測 (runtime/intent 公開窓口・intent extra・agents 非依存・pydantic BaseModel ベース・上位利用支援層)
   │
   ┊ Resilience (runtime/resilience 公開窓口・resilience extra・agents 非依存・宣言層)
   │
利用側アプリ      │ import 委譲（会話実行・Session 生成・評価実行・最適化実行・ガバナンス build・意図予測 classify・resilience build）
   │ import       │
   ▼              │
__init__.py (コア公開 API: __all__ は宣言層シンボルのみ)
   │              │
   ├── registry.py ──┐
   ├── handoffs.py   │
   ├── prompts.py ───┤
   ├── workflow/ ────┤
   │                 ├─→ protocols.py (Protocol。agents 非依存)
   │                 │        │
   │                 │        ▼
   │                 ├─→ spec.py (AgentSpec。agents 非依存・最下層)
   │                 │
   │                 └─→ _validation.py (共有バリデーションヘルパ。agents 非依存・最下層。
   │                           │          realtime/registry・_adapters も下向き参照)
   └────────────────→ _adapters/ (agents / 外部クライアント への import 単一窓口) ◄── 会話サービス / LLMOps 評価 / Agent Lightning 最適化
                              │ runtime import
                              ▼
                         openai-agents (agents) / deepeval / langfuse
```

- 実行寄り層は `runtime/` 配下のサブパッケージ（`runtime/conversation` / `runtime/serve` /
  `runtime/cli` / `runtime/llmops`）として配置する。`runtime/__init__.py` は再エクスポートしない最小の予約
  namespace であり、利用側は `from oai_agentspec.runtime.conversation import ...` のようにサブパッケージ
  公開窓口を直接参照する（`import oai_agentspec.runtime` で serve/cli/llmops のトップ import を連鎖させて
  extra 未導入耐性を壊さないため）。
- 会話サービス（`runtime/conversation` の公開窓口）は会話実行・SDK `Session` 生成を `_adapters` への
  単方向 import 委譲のみで持ち `agents` を直接 import しない。サーバ入口（`runtime/serve`・`serve` extra）は
  公開 API ツリー外の別枠入口層で、会話サービスへの import 委譲辺のみを持つ。
- CLI クライアントは `[project.scripts]` 入口（`runtime/cli`・`cli` extra）として**別プロセス**で動き、
  サーバへは import ではなく**ネットワーク接続（HTTP/WS）**で繋がる。上図では import 委譲辺（実線）と
  区別して点線で示し、`src/oai_agentspec` の import 依存グラフには乗らない。
- **コアは runtime を import しない**。コア（`__init__` / `registry` / `handoffs` / `prompts` / `workflow` /
  `protocols` / `spec` / `_adapters`）から `runtime/` 配下への依存辺は存在せず、`__init__ -> runtime` 方向の
  import を持たない。runtime から上向きにコア（`_adapters` / `registry` / `constants`）と宣言層型
  （`AgentSpec` / `WorkflowGraph` / `HandoffGraph` を read-only で参照）を辿る単方向のみが成立する。
- **`exceptions.py`（例外統一窓口）はコア依存鎖に属さない横断窓口**である。コア namespace 直下に置かれるが、コア列挙群（`__init__` / `registry` / `handoffs` / `prompts` / `workflow` / `protocols` / `spec` / `_adapters`）のいずれからも import されず、`oai_agentspec/__init__.py` は `exceptions` を連鎖 import しない。`exceptions.py` はコア各所（`registry` / `integrity` / `prompts` / `workflow/graph`）と runtime 各所（`runtime/resilience/_errors` / `runtime/conversation/types` / 遅延で `runtime/lightning/types` / `runtime/cli/_models`）の例外定義を上向きに参照するだけの葉（何からも import されない末端）であり、逆方向（コア・runtime -> `exceptions`）の依存辺を作らない。これにより「コアから `runtime/` への依存辺は存在しない」不変条件は不変のまま保たれる。realtime の第 3 の並列ルートと同様、コア公開 API ツリー外の独立 import パス（`import oai_agentspec.exceptions`）で提供する。
- 単方向 import 依存（コア公開 API -> 各層 -> `_adapters` -> `agents`、および runtime -> コア）と公開境界
  （コア `__all__` = 宣言層シンボルのみ / 会話シンボルは `runtime/conversation` 公開窓口 / サーバ入口・CLI
  クライアントは公開 API ツリー外）の整合を保つ。詳細は「会話 Helper（ローカル開発支援）」節を参照。
- 依存方向（単方向）: `__init__` -> {`registry`, `handoffs`, `prompts`, `workflow`} -> {`protocols`, `_adapters`} -> `spec` -> (`agents` は `_adapters` のみ)。`spec` と並ぶ最下層に共有 leaf として `_validation`（共有バリデーションヘルパ・`agents` 非依存。`registry` / `realtime/registry` / `realtime/handoffs` / `_adapters` が下向きに参照）と `_mermaid`（Mermaid 整形の純フォーマッタ。`handoffs` / `realtime/handoffs` が下向きに参照）と `_registry_core`（registry の遅延構築骨格の純ヘルパ。`registry` / `realtime/registry` が下向きに参照）がある。runtime（`runtime/conversation` / `runtime/serve` / `runtime/cli` / `runtime/llmops` / `runtime/governance`）はコアへ依存するが、コアは runtime へ依存しない。
- `workflow/` パッケージは `agents` 非依存であり、SDK 実体（`WorkflowModel` / `workflow_as_tool` / runner シーム本番実装）は `_adapters` に閉じる。依存は `workflow -> _adapters -> agents` の一方向で、循環 import を作らない。`workflow/` がパッケージ化されても（ファサード本体ロジックを内部サブモジュールへ分割しても）この一方向は不変であり、`_adapters` への参照は関数内遅延 import で循環を回避する。
- `spec.py`（`AgentSpec`）と `protocols.py` は `agents` をランタイム import しない。SDK 型（`Agent`）は `TYPE_CHECKING` ブロック内で `from ._adapters import ...` の型エイリアスとして参照する。
- `agents` パッケージへの import は `_adapters/` に集約する。計測基準: `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` の結果が空になること。外部クライアント（採点エンジン `deepeval` / 観測 SaaS `langfuse`）の import も同様に `_adapters/` 配下のみに閉じる（同型 grep で `_adapters` 外に出ないこと）。

## コンポーネントの責務

| モジュール | 責務 |
|---|---|
| `spec.py` | `AgentSpec` の定義。`agents.Agent` の薄い Wrapper。`agents` 非依存の宣言的データ。最下層 |
| `protocols.py` | `AgentBuilder` の Protocol 定義。`agents` 非依存 |
| `_validation.py` | 宣言 spec の共有バリデーションヘルパ（callable instructions の呼び出し可能性・Realtime の静的 prompt 検証・`extra` kwargs の専用フィールド衝突/未知キー検証（両ルートのアダプタが共有））。`agents` 非依存・最下層。通常ルートと Realtime 専用ルートの両 registry / アダプタが共有し、判定とエラーメッセージの単一ソースを保つ |
| `_mermaid.py` | Mermaid flowchart 整形の共有純フォーマッタ。`agents` 非依存・最下層。通常ルートと Realtime 専用ルートの `mermaid()` が同一書式を単一ソースで保つ |
| `_registry_core.py` | registry の到達可能収集 + トランザクショナル 2 パス build/wire + 巻き戻しの共有ヘルパ。`agents` 非依存・最下層。通常ルートと Realtime 専用ルートの registry が遅延構築アルゴリズムと巻き戻しセマンティクスを単一ソースで保つ（差分点＝依存辺プロバイダ・bare ビルド・結線はコールバックで注入） |
| `_adapters/` | `agents` および外部クライアント（採点エンジン `deepeval` / 観測 SaaS `langfuse`）への import 単一窓口。デフォルト `AgentBuilder`（`build_agent`）・`handoff()` 生成・as_tool 生成・SDK 型の再エクスポート・DeepEval 採点窓口・実行トレース捕捉窓口・Langfuse 連携窓口。内部実装は runner シーム / 承認適用 / シリアライズ・session 生成 / SQLite 読取 / HITL 永続テーブル / DeepEval 採点（judge）/ 実行トレース捕捉（routing）/ Langfuse 連携（langfuse）/ 意図予測プロンプト実行（intent）/ Tool メタデータの `function_tool` 結線（tools。`build_function_tool` = メタデータの SDK 引数流し込み・is_enabled callable 結線）/ `RunContextWrapper` 開封の共有ヘルパ `unwrap_run_context`（run_context）等のサブモジュールへ分割し、`__init__.py` を薄い再エクスポート窓口とする（`agents` / 外部クライアントへの import 単一窓口という責務は不変。`deepeval` は `judge` モジュールに、`langfuse` は `langfuse` モジュールの関数内遅延 import に閉じる） |
| `prompts.py` | `PromptStore` / `PromptLayout` / `PromptTemplate` と合成 API（`compose`）・`dynamic_prompt` ヘルパー |
| `registry.py` | `AgentRegistry`。DI 注入・遅延構築・循環ハンドオフ解決・ランタイム差し替え・`validate`・`clone`（登録内容を引き継いだ独立 registry を返す。spec は可変コンテナまで独立コピーし元 registry を不変に保つ。LLMOps の非汚染 mock 注入に使う宣言層プリミティブ） |
| `tool_registry.py` | `ToolSpec` / `ToolRegistry`。Tool の宣言（生関数 + メタデータ）の一元登録・遅延構築 + キャッシュ・照会・enabled 動的トグル。`agents` 非依存のコア層（SDK 結線は `_adapters/tools.py`）。詳細は「Tool Registry」節 |
| `handoffs.py` | `HandoffEdge` / `HandoffGraph` / `from_specs`。宣言的ハンドオフトポロジを registry の public API 経由で反映 |
| `workflow/` | `WorkflowGraph`（ノード/エッジ宣言 DSL）/ `START` / `END` / `NodeResults` / 内部インタプリタ / 非公開 runner シーム Protocol / `default_input_filter` / `as_agent_spec` / `as_facade_spec`。公開型・宣言値 dataclass 群・`WorkflowGraph` 本体・内部インタプリタ・Agent/Tool 化ファサードのサブモジュールへ分割し、`__init__.py` を薄い再エクスポート窓口とする。`agents` 非依存（SDK 型は TYPE_CHECKING / Protocol のみ参照） |
| `integrity.py` | runtime インテグリティ防御の公開窓口。`lockdown` 関数 + 例外型（`IntegrityError` / `PromptTemplateIntegrityError`）+ 型エイリアス `IntegrityCheck` を公開。`agents` 非依存・標準 lib のみ（`hashlib` / `importlib.metadata` / `pathlib` / `sys`）依存のコア層最下層。`PromptStore.__init__` シグネチャは不変で、検証 / preload は `lockdown` 経由で発火する |
| `exceptions.py` | lib 独自例外 9 種の再エクスポート統一窓口（`oai_agentspec.exceptions`）。定義実体は各モジュールに残し isinstance/issubclass 完全互換を保つ。コア依存鎖に属さない横断窓口で `__init__.py` から import されない。詳細は「例外の統一窓口」節 |
| `realtime/` | Realtime エージェントの専用宣言ルート（コア公開 API ツリー外・宣言層）。`RealtimeAgentSpec` / `RealtimeHandoffConfig`（`agents` 非依存・最下層）・`RealtimeAgentBuilder` Protocol・`RealtimeAgentRegistry`（2 パス遅延バインド・handoff 結線・validate）・宣言的ハンドオフグラフ DSL（`RealtimeHandoffGraph` / `RealtimeHandoffEdge` / `from_specs`）・公開窓口 `oai_agentspec.realtime` を持つ。SDK 結合（`agents.realtime`）は `_adapters/realtime.py` に閉じ、`realtime/` からの参照は `_adapters`・共有 leaf（`_validation` / `_mermaid`）への上向き単方向のみ。コアから `realtime/` への依存辺はない |
| `runtime/` | 実行寄り層（ローカル開発支援）の集約 namespace。`runtime/conversation`（会話サービス・公開窓口）/ `runtime/serve`（FastAPI サーバ入口・`serve` extra）/ `runtime/cli`（CLI クライアント・`cli` extra）/ `runtime/llmops`（LLMOps 評価・公開窓口・採点コア `llmops` extra + 任意の観測 `llmops-langfuse` extra）/ `runtime/lightning`（Agent Lightning プロンプト最適化・公開窓口・`lightning` extra）/ `runtime/governance`（AGT ガバナンス・公開窓口・`governance` extra）/ `runtime/intent`（意図予測・公開窓口・`intent` extra）/ `runtime/resilience`（Resilience 宣言型・公開窓口・`resilience` extra）/ `runtime/hooks`（`RunHooksBase` 合成ヘルパー `chain_hooks` の公開窓口・extra 不要＝`agents` はコア依存）を直下サブパッケージに持つ。各サブパッケージの `__init__.py` を公開窓口とする。`runtime/__init__.py` は再エクスポートしない最小の予約 namespace。runtime からコア（`_adapters` / `registry` / `constants`）と宣言層型（read-only）への上向き参照のみを持ち、コアは runtime へ依存しない |

`_adapters/` が再エクスポートする SDK 型（`Agent` / `RunContextWrapper` / `Model` / `Prompt` / `DynamicPromptFunction` / `GenerateDynamicPromptData` / `Handoff` / `Runner` / `ModelResponse` / `ModelSettings` / `FunctionTool` / `ToolContext` / `ToolApprovalItem` / `RunState`）は内部の型参照用であり、公開契約には含めない（HITL の `ToolApprovalItem` / `RunState` は中断状態を SDK と結合する内部窓口であり外部公開しない。承認必須ツール宣言用の `function_tool` のみ公開再エクスポートする）。利用者はこれらの型が必要な場合 `from agents import ...` を直接使う。

## 公開 API

`src/oai_agentspec/__init__.py` の `__all__` に掲載されたシンボルのみがコアの公開契約である。
バージョニングは SemVer に従う。コア `__all__` は **宣言層シンボルのみ**を掲載し、会話シンボルは
`oai_agentspec.runtime.conversation` の公開窓口に置く。本ライブラリは未リリースの Alpha であり、
公開契約は後方互換を必須とせず実装の確定に合わせて整える。

```python
__all__ = [
    "END",
    "START",
    "AgentRegistry",
    "AgentSpec",
    "FacadeMode",
    "HandoffConfig",
    "HandoffEdge",
    "HandoffGraph",
    "NodeFn",        # 型: FUNCTION ノードの (msg, ctx) -> 出力
    "NodeHook",      # 型: ノード前後フック
    "NodeResults",
    "Router",        # 型: 条件エッジ router (msg, ctx) -> 判定キー
    "PromptStore",
    "PromptLayout",
    "PromptTemplate",
    "ToolRegistry",  # Tool 一元管理（詳細は「Tool Registry」節）
    "ToolSpec",      # Tool メタデータ宣言 dataclass
    "WorkflowGraph",
    "function_tool",         # HITL: 承認必須ツール宣言用（_adapters 再エクスポート・コア公開）
    "default_input_filter", # ヘルパー
    "dynamic_prompt", # ヘルパー
    "from_specs",
    "lockdown",                       # runtime インテグリティ防御の起動点
    "IntegrityCheck",                 # 型: Callable[[], None]
    "IntegrityError",                 # 例外（Exception 継承・基底）
    "PromptTemplateIntegrityError",   # 例外（IntegrityError 継承）
    "RegistryFrozenError",            # 例外（RuntimeError 継承）
    "WorkflowFrozenError",            # 例外（RuntimeError 継承）
]
```

会話シンボルはコア `__all__` には載せず、`from oai_agentspec.runtime.conversation import ...` の
公開窓口で参照する。`runtime/conversation/__init__.py` の `__all__` は会話シンボル群を掲載する
公開契約であり、`ConversationService` /
`ConversationError` / `ConversationErrorCode` / `SessionInfo` / `SessionPolicy` / `CompactionConfig` /
`StreamDelta` / `StreamDone` / `StreamError` / `StreamEvent` / `ApprovalRequired` / `ApprovalDecision` /
`PendingApproval` / `SendStatus` の会話シンボルと `SendResult` を含む。`function_tool` は会話機能に限定されない agents 隔離宣言ヘルパで
あるため、コア `__all__` 公開のまま維持する（`from agents` 直書きを避ける宣言層の公開契約）。

`ConversationErrorCode` には承認系コード（`UNKNOWN_APPROVAL` / `APPROVAL_ALREADY_RESOLVED` /
`NO_PENDING_APPROVAL`）を含む。HITL の承認待ちは専用イベント型（`ApprovalRequired`）で表現し、`StreamEvent`
Union（`StreamDelta` / `StreamDone` / `StreamError` の 3 メンバ）には混ぜない。これにより既存の 3 メンバ
網羅契約を保ち、承認待ちなしターンの消費者を変更不要に保つ。

`AgentBuilder`（DI 拡張点）は `oai_agentspec.protocols` に置き、トップレベル公開 API には
含めない（テスト/上級用途向け。SDK 隔離は `_adapters` が担う）。

Realtime シンボル（`RealtimeAgentSpec` / `RealtimeHandoffConfig` / `RealtimeAgentRegistry`）はコア
`__all__` に載せず、`oai_agentspec.runtime.conversation` と同様に `oai_agentspec.realtime` 公開窓口で
参照する。`RealtimeAgentBuilder` は `AgentBuilder` と同様どの `__all__` にも載せず、
`oai_agentspec.realtime.protocols` の直接 import でのみ参照する。

`WorkflowModel` / `workflow_as_tool` / runner シーム Protocol は非公開である。`WorkflowModel` と
`workflow_as_tool` は `_adapters` に閉じた SDK 結合実装であり、runner シーム Protocol は内部 / テスト用
の構築シーム（利用者は runner を渡さない）として `workflow/` パッケージ内に置く。公開されるのは宣言 DSL の
`WorkflowGraph` のみで、利用者は `WorkflowGraph.as_agent_spec` / `as_facade_spec` が返す `AgentSpec` を
通して間接的にこれらを使う。

| シンボル | 用途 |
|---|---|
| `AgentRegistry` | Agent の登録・遅延構築・ランタイム差し替えの中枢 |
| `AgentSpec` | `Agent` の薄い宣言的 Wrapper（dataclass） |
| `HandoffConfig` | ハンドオフ 1 エッジの設定（description / on_handoff / input_type / input_filter / is_enabled + options 素通し） |
| `HandoffEdge` | 静的ハンドオフ 1 エッジ（src / dst / config） |
| `HandoffGraph` | ハンドオフトポロジ。`edge` / `dynamic_edge` で宣言、`apply(registry)` で反映、`mermaid()` で可視化 |
| `PromptStore` | 利用側が渡す root 配下のテンプレートをロードし instructions を合成するストア |
| `PromptLayout` | 合成セグメントのディレクトリ構成（必須・明示指定） |
| `PromptTemplate` | テンプレート文字列ラッパー（本文 + メタデータ） |
| `ToolRegistry` | Tool の一元登録・遅延構築 + キャッシュ・照会・enabled 動的トグル（詳細は「Tool Registry」節） |
| `ToolSpec` | Tool メタデータの宣言 dataclass（func + enabled / 承認要否 / タイムアウト / 失敗時エラー文言 / 名前・説明上書き / strict_mode / extra） |
| `dynamic_prompt` | ctx 由来の id/version/variables から `agents.Prompt` 参照を生成するヘルパー（`AgentSpec.prompt` 用） |
| `from_specs` | `AgentSpec` 群の `handoffs` 宣言から `HandoffGraph` を構築 |
| `WorkflowGraph` | ワークフローのノード/エッジ宣言 DSL。`add_agent_node` / `add_function_node` でノード、`add_edge` / `add_conditional_edges` / `add_fan_in_edge` でエッジを宣言し、`validate` / `mermaid` / `as_agent_spec`（経路C）/ `as_facade_spec`（経路A / D）を提供 |
| `FacadeMode` | `as_facade_spec` の入口モデル種別（`LLM_INPUT`=実 LLM 1 回 / `LLM_INPUT_OUTPUT`=実 LLM 2 回 / `DETERMINISTIC`=実 LLM 0 回・決定論＝経路D）。既定 `LLM_INPUT`（従来の経路A） |
| `START` / `END` | `WorkflowGraph` のエッジ端点に使う入口 / 終端の番兵（`add_edge(START, ...)` / `add_edge(..., END)`） |
| `NodeResults` | 実行中スコープの「ノード名 → 出力」記録（`record` / `get`）。フックへ渡る lib 内部値 |
| `default_input_filter` | 経路A / D の流入履歴を直近 `limit` 件へ有界化する input_filter を生成するヘルパー |
| `lockdown` | runtime インテグリティ防御の起動関数。root verify + store verify+preload + libs detect + custom checks + registry/workflow freeze を 6 段順次・fail-closed で実行。詳細は `docs/integrity.md` |
| `IntegrityCheck` | `Callable[[], None]` 型エイリアス。検知関数のシグネチャ規約（違反時に `IntegrityError` 系を raise） |
| `IntegrityError` | ファイル整合性違反の基底例外（`Exception` 継承）。`lockdown` の root verify / libs detect / custom check で raise |
| `PromptTemplateIntegrityError` | `IntegrityError` 継承。`lockdown` の store verify 段で manifest 不一致時に raise |
| `RegistryFrozenError` | `RuntimeError` 継承。`AgentRegistry.freeze()` 後の書換違反で raise |
| `WorkflowFrozenError` | `RuntimeError` 継承。`WorkflowGraph.freeze()` 後の書換違反で raise |

### 例外の統一窓口（`oai_agentspec.exceptions`）

lib 独自例外は各モジュールに定義実体を持つが、利用者が catch する際の import 経路を単一化するため、
`oai_agentspec.exceptions` を統一窓口として提供する（SDK が `agents.exceptions` に集約する慣行に倣う）。
再エクスポート専用窓口であり定義実体は移動しない。窓口経由で得た例外は定義元と同一クラスオブジェクトで、
`isinstance` / `issubclass` は完全互換に保たれる。`import oai_agentspec.exceptions` は extra 未導入環境でも
壊れず、遅延 2 種の親パッケージ（`runtime.lightning` / `runtime.cli`）を連鎖 import しない（PEP 562 遅延の
実効性を subprocess 隔離テストで担保する）。

`__all__` は次の 9 例外を掲載する。取得方式は依存の重さで振り分ける:

- **直 import（7 種・追加依存ゼロ）**: `RegistryFrozenError`（`registry`）/ `IntegrityError`・
  `PromptTemplateIntegrityError`（`integrity`）/ `PromptResolutionError`（`prompts`）/
  `WorkflowFrozenError`（`workflow/graph`）/ `RunBudgetExceeded`（`runtime/resilience/_errors`）/
  `ConversationError`（`runtime/conversation/types`）
- **PEP 562 遅延取得（`__getattr__` / `__dir__`・2 種）**: `OptimizeError`（`runtime/lightning/types`）/
  `ConversationClientError`（`runtime/cli/_models`）。サブモジュール import が親 package `__init__` を
  実行する分の import コスト膨張を避け、将来の import 構造変更に対し窓口を頑健に保つため遅延する。
  既存の resilience 窓口と異なり遅延先が `_adapters` ではなくモジュール自体になる差分理由は module
  docstring に明記する。`__all__` 外の名前アクセスは `AttributeError`

コア `__all__` との関係: `IntegrityError` / `PromptTemplateIntegrityError` / `RegistryFrozenError` /
`WorkflowFrozenError` はコア `__all__` にも掲載済みで、`exceptions` 窓口との二重公開を許容する
（コア `__all__` の集合は契約のため不変維持）。`PromptResolutionError` / `ConversationClientError` は
従来どおりコア `__all__` に載せず、統一窓口のみで公開する。docs の正規経路案内は
`oai_agentspec.exceptions` を推す。

## AgentSpec

`AgentSpec` は `agents.Agent` の薄い Wrapper である。`instructions` / `prompt` / `tools` /
`model` / `model_settings` / `hooks` は `Agent` と同じ意味を持ち、それ以外の `Agent` kwarg は
`extra` で素通しする。`handoffs` / `sub_agents` はエージェント名の参照で、registry が遅延
構築時に解決する（グラフ連携の追加機能）。

| フィールド | 型 | 役割 |
|---|---|---|
| `name` | `str` | エージェント名（registry 内で一意） |
| `instructions` | `str \| Callable \| None` | システムプロンプト。文字列、または `(context, agent)` の 2 引数 callable。`PromptStore.compose` の戻り値を渡せる |
| `prompt` | `agents.Prompt \| DynamicPromptFunction \| None` | `Agent.prompt`（Responses API 用。`dynamic_prompt` ヘルパーの戻り値） |
| `tools` | `list` | Agent に渡すツール |
| `model` | `str \| agents.Model \| None` | モデル指定 |
| `model_settings` | `agents.ModelSettings \| None` | モデル設定 |
| `hooks` | `agents.AgentHooks \| None` | エージェントフック |
| `handoffs` | `list[str]` | ハンドオフ先エージェント名（グラフ連携） |
| `handoff_options` | `dict[str, HandoffConfig]` | dst 名 -> per-edge `handoff()` 設定 |
| `sub_agents` | `list[str]` | as_tool 配線するサブエージェント名（グラフ連携） |
| `sub_agent_tools` | `dict[str, tuple[str \| None, str \| None]]` | サブ名 -> (tool_name, tool_description) の as_tool 上書き |
| `dynamic_handoffs` | `list[DynamicHandoff]` | 動的ハンドオフ宣言（on_invoke で候補から転送先を実行時選択） |
| `extra` | `dict[str, Any]` | 上記以外の `agents.Agent` kwarg 素通し |

`instructions` / `prompt` は `Agent` のフィールドにそのまま渡される。本ライブラリは指示系に
独自の排他規則を設けない（SDK の意味論をそのまま尊重）。ただし `instructions` に callable を
渡す場合、SDK が `inspect.signature` でパラメータ数を厳密検査するため `(context, agent)` の
2 引数が必須であり、これを register 時に検証して run 時 `TypeError` を前倒し検出する。

`extra` に専用フィールド（name / instructions / prompt / tools / handoffs / model /
model_settings / hooks）と同名のキー、または `agents.Agent` が受け付けない未知キーが含まれる
場合は構築時に `ValueError` を送出する。

### SandboxAgentSpec

`SandboxAgentSpec` は `AgentSpec` を継承し、`agents.sandbox.SandboxAgent`（`agents.Agent` の
正式なサブクラス）向けの専用フィールドを追加する dataclass。`AgentSpec` の全フィールド
（`tools` / `handoffs` / `model` / `model_settings` / `hooks` / `input_guardrails` /
`output_guardrails` / `sub_agents` 等）をそのまま継承して保持する。

| フィールド | 型 | 役割 |
|---|---|---|
| `default_manifest` | `Any \| None` | サンドボックスのファイルシステム / env / マウント定義（SDK `Manifest` 相当。不透明型） |
| `capabilities` | `Any \| None` | サンドボックスが提供する機能のトグル列（SDK `Sequence[Capability]` 相当。不透明型） |
| `run_as` | `Any \| None` | サンドボックス実行時のユーザーコンテキスト（不透明型） |
| `base_instructions` | `str \| Callable \| None` | サンドボックス実行エージェント向けの基底システムプロンプト。文字列、または `(context, agent)` の 2 引数 callable |

`default_manifest` / `capabilities` / `run_as` / `base_instructions` は `agents.sandbox` の
実型を持たない不透明型として宣言され、`spec.py` は `agents.sandbox` を import しない
（`AgentSpec.model_settings` / `hooks` と同じ不透明型パターン）。4 フィールドとも未指定
（`None`）の場合は構築時の kwargs に積まず、`SandboxAgent` 自身の既定値に委ねる（ライブラリ
側で SDK の既定値を再現・ハードコードしない）。`capabilities` 未指定時の SDK 既定はシェル実行を
含む機能群を有効化しうるため、最小権限にしたい場合は `capabilities` を明示指定すること。
指定された list 値（`capabilities` 等）は build 時に新しい list へコピーされ、構築済み Agent へ
spec 側リストの事後 mutation が伝播しない（`tools` と同じ遮断挙動）。

`base_instructions` に callable を渡す場合、SDK が `(context, agent)` の 2 引数 callable を
要求する点は `instructions` と同様だが、検証タイミングが異なる。`instructions` は
`AgentRegistry` の register 時に検証されるのに対し、`base_instructions` の callable arity
検証は `_adapters/builders.py` の build 時（`build_agent` 内）で行われる。これは `registry.py`
が `SandboxAgentSpec` 固有の分岐を持たない（属性アクセスのみで動作する）方針を維持するための
非対称である。`registry.py` は sandbox 固有の分岐を持たず、`AgentSpec` と `SandboxAgentSpec` を
混在登録・混在ハンドオフできる。spec 複製（`freeze` / `clone`）の外部 mutation 遮断は、spec の
全 dataclass フィールドを走査して list / dict 値を新コンテナに複製する方式であり、サブクラス
固有の可変フィールド（`capabilities` 等）にも列挙の手動同期なしで適用される。

## Tool Registry

`ToolRegistry` は Tool の宣言（生の Python 関数 + メタデータ）を一元管理するコア公開 API である。
利用者は lib 非依存の純関数を散在するファイルに置いたまま、組み立てポイントで
`register(ToolSpec(...))` により一元登録し、`tool_registry.<name>` の属性アクセスでメタデータ
適用済みの SDK `FunctionTool` を取得して `AgentSpec(tools=[...])` にそのまま渡す。
`AgentSpec` / `AgentRegistry` からは完全に独立で、opt-in の注入点も設けない（橋渡しは利用者
コードの `tools=[tool_registry.<name>]` のみ）。

### ToolSpec（メタデータ宣言）

`ToolSpec` は mutable な dataclass である。SDK にネイティブ機構が存在するメタデータは独自の
実行時機構を作らず、対応する `function_tool()` 引数へ委譲する（build-don't-run。実行時の
有効判定・承認・タイムアウトは SDK が担う）。

| フィールド | 役割 |
|---|---|
| `func` | sync / async の生 Python 関数（必須） |
| `name` | Registry キー。省略時は `func.__name__`。登録後の変更は非サポート（`func` も同様。登録時に確定）。`name_override` 未指定時は本フィールドの値が SDK 提示名（`name_override` 相当）にも反映される（Registry キーと LLM 提示名が既定で一致・`name_override` 明示指定はそれを上書きする） |
| `enabled` | 有効/無効（既定 `True`）。SDK `is_enabled` へ「Registry 現在値を参照する callable」として結線される |
| `needs_approval` | 承認要否（SDK `needs_approval` へ委譲） |
| `timeout` / `timeout_behavior` / `timeout_error_function` | タイムアウト（SDK 同名引数へ委譲） |
| `failure_error_function` | 失敗時エラー文言。「未指定（SDK 既定 formatter に委ねる）/ 関数指定 / `None` 明示（例外を文字列化せず素通し）」の 3 値を区別する。未指定は Registry 独自の module-level センチネル既定で表現し、当該 kwarg を渡さない（SDK private センチネル非依存） |
| `name_override` / `description_override` | SDK 提示名 / 説明の上書き。Registry の登録キー（属性アクセスに使う `<name>`）とは独立に指定できる。`name_override` 未指定時は `name` フィールドの値が SDK 提示名として使われる |
| `strict_mode` | 厳格スキーマの有効/無効。未指定（`None`）は SDK 既定に委ねる |
| `extra` | 上記以外の `function_tool()` kwarg 素通し（`AgentSpec.extra` と同型思想の予約キー / 未知キー検証つき。構築時に `ValueError`） |

冪等性（idempotent）フィールドは持たない。未指定のメタデータは kwargs に積まず SDK 既定値に
委ね、Registry 側で SDK 既定値を再現・ハードコードしない（None-omission）。

### ToolRegistry（登録・取得・照会・動的更新）

- `register(spec: ToolSpec) -> None`: 宣言の保持のみを行い、この時点では SDK に触れない
  （遅延ラップ）。二重登録、および属性アクセスで到達不能な名前（公開メソッド名との衝突 /
  `_` 始まり / 非識別子）は `ValueError`。
- `names() -> list[str]`: 登録済み Tool 名の昇順リスト。
- `metadata(name) -> ToolSpec`: live な `ToolSpec` を返す。未登録名は登録済み名一覧つき
  `KeyError`。属性代入による動的更新の反映範囲は、`enabled` = 構築済み Tool へ即反映（後述）、
  それ以外 = 照会値のみ（SDK 引数の値は構築時に確定し、invalidate・再構築の機構は設けない）。
- 属性アクセス `tool_registry.<name> -> FunctionTool`: `_adapters` 経由で `function_tool()` を
  1 回だけ呼んで構築しキャッシュする（同一インスタンス返却。`AgentRegistry` の遅延構築と同型）。
  未登録名は登録済み名一覧つき `AttributeError`（`_` 始まり名・実在属性は通常解決）。
- 並行制御は `AgentRegistry` と同じく利用者責任（単一スレッド前提）。

`enabled` の動的トグル: 構築時に `is_enabled` へ bool を焼き込まず「`ToolSpec.enabled` の
現在値を読む callable」を結線するため、`metadata(name).enabled = False` は構築済み Agent /
Tool の再構築なしに次の run から当該 Tool を LLM から隠す（SDK `is_enabled` のネイティブ挙動へ
委譲。`True` へ戻せば同様に再提示される）。

### Tool 宣言の 2 経路

`function_tool` の直接宣言（コア公開の `_adapters` 再エクスポート）はメタデータの一元管理が
不要な単発 Tool 向け、`ToolRegistry` 登録は一元管理・照会・enabled 動的トグルが必要な Tool
向けであり、両経路は併存する。

SDK ラップ（`function_tool()` 呼び出し・メタデータの SDK 引数への流し込み・is_enabled callable
結線）は `_adapters/tools.py` の `build_function_tool` に閉じる（SDK 隔離）。設計判断の経緯は
`docs/adr/0001-tool-metadata-centralization.md` を参照。

## SDK 隔離と依存性注入（DI）

openai-agents への結合の隔離は `_adapters/__init__.py`（`from agents import ...` の単一窓口）が
担う。SDK の破壊的変更が起きても修正対象はこのモジュールに限局する。これが「SDK アップデートに
強い設計」の実体である。

`protocols.py` の `AgentBuilder` Protocol は、これとは別の「生成処理そのものの差し替え」拡張点。

- `AgentBuilder`: `build(spec) -> Agent`。handoffs を空にした Agent を 1 つ構築する責務。
  デフォルト実装（`build_agent`）は `_adapters` に置く。`build_agent` は spec を
  `isinstance(spec, SandboxAgentSpec)` で分岐し、構築先クラス（`agents.Agent` /
  `agents.sandbox.SandboxAgent`）を切り替える。
- 注入点は `AgentRegistry.__init__(agent_builder=None)`。省略時は `_adapters` のデフォルト実装。

`AgentBuilder` はテスト（`agents.Agent` を構築しないフェイク `FakeAgentBuilder` の注入）と、
tools 一律ラップ等の構築置換という上級用途向けであり、トップレベル公開 API には含めない
（`oai_agentspec.protocols` で参照）。モデルや instructions のデフォルト補完のような augment は
spec 側で表現できるため DI の動機にならない。プロンプト合成は `PromptStore` がスタンドアロンに
行い、registry はその値を素通しするため、prompt 解決の DI 注入点は設けない。

compaction クライアントの注入は新たな DI 拡張点を設けず、session 生成方針の一部として
`SessionPolicy.compaction`（`CompactionConfig`）で受け渡す。

## プロンプト合成

`PromptStore` は利用側が渡す root 配下の `.md`（YAML frontmatter）/ `.yaml` をロードし、
共通ベース・パーツ・エージェント個別テンプレートを連結して instructions を生成する。
生成結果（静的 `str` または動的 callable）を `AgentSpec.instructions` に渡して使う。
`agents` には依存しない。

### プロンプトの提供形態とレイアウト

ライブラリはプロンプトファイルを同梱しない。利用側が root を渡し、`PromptLayout` で
ディレクトリ構成を**明示必須**で指定する（暗黙の既定を設けず、規約フォルダを勝手に仮定して
無音で誤合成するミスを防ぐ）。

```python
store = PromptStore("prompts", PromptLayout(base="base", parts="parts", agents="agents"))
```

`PromptLayout(base=, parts=, agents=)` の各ディレクトリ名は利用側の既存構成に合わせて任意に
指定できる（例 `PromptLayout(base="common", parts="snippets", agents="roles")`）。各ディレクトリ名に
空文字 `""` を渡すと root 直下を探索する（フラット配置）。

規約レイアウトの例:

```
<root>/base/main.md     全 main 共通ベース
<root>/base/sub.md      全 sub 共通ベース
<root>/parts/<name>.md  使い回しパーツ
<root>/agents/<name>.md エージェント個別
```

各セグメントのサブディレクトリ配下はさらに階層化してよい。セグメント名はサブディレクトリ配下を
再帰探索し stem 一致で解決する（例: `agents/billing/refund.md` を `agent="refund"` で取得）。
同 stem が複数ある場合は曖昧エラーとなるため `agent="billing/refund"` のようにサブパスを含めて
指定する。ディレクトリ名に空文字を指定したフラットセグメントは root 直下のみを非再帰で探索し、
他セグメントのサブディレクトリを誤って拾わない。

### compose による合成

```python
def compose(
    agent: str | None = None,
    *,
    base: str | None = None,
    parts: Sequence[str] = (),
    layout: Sequence[str] | None = None,
    vars: dict | Callable[[ctx], dict] | None = None,
) -> str | Callable[[context, agent], str]
```

- セグメント参照記法は `base:<name>` / `part:<name>` / `agent:<name>` の 3 種。
- デフォルト合成順は `base -> parts -> agent`。`base` は共通ベース名（例 `"main"` / `"sub"`）を
  指定する。`layout` を渡すとセグメント参照の列をそのまま順序として使い、デフォルト順を全置換する
  （例: agent の後に part を置く）。
- 各セグメントは frontmatter（`---` 囲い）を除いた本文を `${var}` で置換し `\n\n` 連結する。
- 参照先ファイルが存在しない / 記法が不正な場合は文脈付きエラーを送出する（無音スキップしない）。

`vars` の型で静的/動的が決まり、戻り値は `Agent.instructions`（`str | callable`）にそのまま
渡せる:

- `vars` が dict / None: ビルド時に置換した**静的な `str`** を返す。
- `vars` が callable（`RunContextWrapper -> dict`）: 各 run で ctx から変数を生成して合成する
  **2 引数 callable `(context, agent) -> str`** を返す（動的注入）。

利用例:

```python
# 静的
AgentSpec(name="triage", instructions=store.compose(agent="triage", base="main", parts=["style"], vars=VARS))
# 動的（run ごとに ctx から変数を生成）
AgentSpec(name="concierge", instructions=store.compose(agent="concierge", vars=lambda ctx: {...}))
```

### dynamic_prompt（Agent.prompt）

`dynamic_prompt` ヘルパーは ctx 由来の `id` / `version` / `variables` から `agents.Prompt` を
生成し `AgentSpec.prompt` に渡す。`agents.Prompt` は TypedDict で `id`（必須）/ `version` /
`variables` のみを持ち本文フィールドを持たないため、本経路はテンプレート本文を扱わず、OpenAI
Responses API の保存済みプロンプト参照に限定される。テンプレート本文ベースの動的注入は
`compose(vars=callable)` の instructions 経路を使う。

## サブエージェント

`AgentSpec.sub_agents: list[str]` に列挙したエージェントを、registry が
`agent.as_tool(tool_name, tool_description)` でツール化し、メインの `tools` に注入する。
tool 名 / 説明は `sub_agent_tools` で上書きでき、省略時は SDK がエージェント名から導出する。

handoff（制御移譲・戻らない）と異なり、サブエージェントはメインが呼んで結果を受け取り処理を
続行する（制御がメインへ戻る）。`as_tool` はサブ Agent インスタンスを参照として取り込むため、
ビルド順序・`update` 時の invalidate 対象になる。

ビルド依存辺は **handoffs ∪ sub_agents** である。到達可能 spec の収集（循環解決の局所 2 パス）も、
依存逆引きによる連鎖 invalidate の対象も、この和集合を辿る。未登録名がビルド時に含まれる場合は
文脈付きエラーを送出する。

## 循環ハンドオフ解決

`a -> b -> a` のような循環ハンドオフを `RecursionError` なく構築する。
SDK 上 `Agent.handoffs` は可変 list であり `get_handoffs` が毎ターン読むため、構築後の後付け
mutation が成立する。

`get(name)` 起点・到達可能 spec のみを対象とした局所 2 パス遅延バインド方式を採る。

1. `name` から `spec.handoffs` と `spec.sub_agents`（依存辺 = handoffs ∪ sub_agents）を辿り、
   到達可能かつ未ビルドの spec 集合を visited 集合で循環を打ち切りつつ収集する（全 spec ではない）。
2. パス 1: 収集 spec をすべて `handoffs=[]`・サブツール未注入でビルドし `_built` に登録する。
3. パス 2: 収集各 spec の `handoffs` を走査して確定インスタンスを結線し、`sub_agents` を
   `as_tool` 化して `tools` に注入する。

パス 1/2 はトランザクショナルに実行し、結線中に例外（未登録参照など）が出た場合は本呼び出しで
新規キャッシュした bare agent を巻き戻し、不完全なインスタンスを残さない。

`register` 時点ではビルドしない（遅延性を維持）。構築後、object identity が保証される
（`a.handoffs[0] is registry.get("b")` かつ `b.handoffs[0] is registry.get("a")`）。

到達可能収集と 2 パス + 巻き戻しの骨格は共有 leaf `_registry_core` に一元化し、通常 / Realtime 両 registry が委譲する（差分点の注入方式はコンポーネントの責務表を参照）。

詳細な検討経緯は `docs/rationale/handoff-cycle-resolution.md` を参照。

### HandoffGraph による反映

`HandoffGraph.apply(registry)` は各 src について内部プリミティブ `registry._update_handoffs(src, ...,
mode="replace", ...)` へ委譲する（registry の生の内部状態には触れない）。`mode="replace"` のため
グラフが当該 src のトポロジの真実源となり、グラフを編集して再 apply すれば実行時に再構成できる。
factory 起点の src は宣言的反映の対象外であり、apply 時に明示エラーで通知する。`mermaid()` は
entry / 静的エッジ / 動的エッジ（破線）を Mermaid flowchart 文字列として返す。

静的エッジ `HandoffEdge(src, dst, config)` の `config: HandoffConfig` は SDK `handoff()` の主要引数を
型付きで保持する（`description` = `tool_description_override`、`tool_name` = `tool_name_override`、
`on_handoff` / `input_type` / `input_filter` / `is_enabled`）。それ以外の `handoff()` kwarg は
`config.options` で素通しする。型付きフィールドと重複する予約キーを `options` に含めるとエラー。

動的エッジ `DynamicHandoffEdge` / `DynamicHandoff` / `HandoffGraph.dynamic_edge(...)` は固定 1
ターゲットでなく、resolver `(context, input_json) -> 候補名` が転送先を実行時に選ぶ。静的エッジの
`HandoffConfig` と対称に `description` / `on_handoff` / `input_type` / `input_filter` / `is_enabled` /
`options` を**同名・同型**で保持し、SDK `handoff()` の対応引数と同じ意味を持つ（パリティ）。動的エッジ
宣言の主経路は `HandoffGraph.dynamic_edge(...)` で、`from_specs(specs)` は静的エッジのみを扱う。

registry は候補名を registry から解決する `on_invoke_handoff` を生成し、`make_dynamic_handoff` が
生 `Handoff` を構築する（`handoff()` は on_invoke を差し込めないため）。`on_handoff` または
`input_type` のいずれかが指定された場合は元 on_invoke（resolver 経由で転送先 Agent を返す）を
wrap する。wrap 内の処理順序は次の通り。

1. `input_type` 指定時、`input_json` を pydantic で parse する（後述の closure で TypeAdapter を再利用）。
2. resolver を呼んで転送先 Agent を取得する。
3. `on_handoff` を発火する。引数 dispatch は `input_type` 有無で `(ctx, parsed)` または `(ctx,)` を
   直接決定し、arity 検査・`TypeError` リトライは行わない。Python のパラメータ解決に委ねるため、
   optional 2 引数 / `*args` のような柔軟シグネチャも自然に対応する。
4. 取得済みの転送先 Agent を return する。

resolver は raw `input_json` を受ける現行契約のまま不変であり、parsed オブジェクトは `on_handoff`
のみが受け取る。`input_type` 指定時は `pydantic.TypeAdapter` を関数内で 1 度だけ構築して closure に
capture し、呼出毎の再生成を避ける。JSON Schema は SDK の `agents.strict_schema.ensure_strict_json_schema`
で OpenAI strict tool calling 形式に整形し、不在時 / 例外時は pydantic 生スキーマへフォールバックする。
`options={"strict_json_schema": False}` で strict 化を skip でき、optional / default 値ありフィールドが
`required` に強制されなくなる。Pydantic `Field(description=...)` とモデル docstring は標準挙動で JSON
Schema の `properties[name].description` / 全体 description に展開され、ハンドオフ tool の
`parameters` として LLM に届く。

`options` の予約キー集合は生 `Handoff` フィールド名（`tool_name` / `tool_description` /
`input_json_schema` / `on_invoke_handoff` / `agent_name` / `input_filter` / `is_enabled`）であり、これらを
`options` に書くと `ValueError`。resolver の戻り名は `candidates` 内に強制し、候補名は依存解決・
`validate()` の対象になる。`HandoffGraph.mermaid()` の動的エッジ可視化（破線）は不変。

`HandoffConfig`（frozen）と `DynamicHandoff`（mutable）は `description` / `on_handoff` / `input_type` /
`input_filter` / `is_enabled` / `options` の 6 フィールドを同名・同型で共有する。Python の dataclass は
frozen / unfrozen の混在継承を許可しないため構造統合せず、6 フィールドのパリティは
ユニットテスト（`tests/test_handoff_config.py::test_handoff_config_and_dynamic_handoff_share_field_names`）
で機械的に保証する。

## ワークフロー

ワークフロー機能は実験的（experimental）であり、インターフェース・挙動は今後変わる可能性がある。

ワークフロー機能は「宣言（`WorkflowGraph`）と実行（内部インタプリタ）の分離」と「実行口を SDK の
`Runner.run` 一本に寄せる build-don't-run」を原則とする。lib は宣言・build-time 検証・実行エンジン・
薄い結線に徹し、公開の実行 API（`WorkflowEngine.run` 等）を持たない。ワークフローは Agent（経路C）
または Tool ファサード（経路A/D）として `Runner.run` で走る。

### WorkflowGraph（ノード/エッジ宣言 DSL）

`WorkflowGraph(name)` は run 状態を持たない宣言 dataclass で、LangGraph（`StateGraph`）/
Microsoft Agent Framework（`WorkflowBuilder`）に倣い、ノードとエッジを明示的に宣言する。チェーン糖衣
（sequence / parallel / branch / loop）や位置依存の暗黙ルールは持たず、順次 / 並列 / 分岐 / 合流 / ループ
はすべてエッジで表す。

- ノード宣言（NodeKind は AGENT / FUNCTION の 2 種）:
  - `add_agent_node(name, *, agent)`: agent は registry 上の名前参照。ノードは上流から流れた出力
    （メッセージ）を入力に当該 Agent を実行する。入力は string / SDK input-list を期待し、上流が非文字列
    （dict / オブジェクト）を返す場合は手前の FUNCTION ノードで string へ整形してから渡す（暗黙の str 化は
    しない）。
  - `add_function_node(name, *, fn)`: `fn(msg, ctx) -> 出力`（msg = 上流ノードの出力、戻り値 = このノードの
    出力）。`ctx` は SDK の `RunContextWrapper | None` で、利用者が `Runner.run(context=...)` に渡した
    オブジェクトは `ctx.context` で得る（SDK の動的 instructions / ガードレールと同じ流儀）。経路A では実 wrapper
    （`ToolContext`）が透過し、経路C では `None`（C-11）。sync / async 両対応。
  - 同名ノードの二重登録はエラー。
- エッジ宣言（すべて self を返しチェーン可能）:
  - `add_edge(src, dst)`: src→dst の有向エッジ。端点に `START`（入口）/ `END`（終端）の番兵を使える
    （`add_edge(START, "plan")` / `add_edge("write", END)`）。`START` からのエッジは 1 本のみ（単一エントリ。
    超過で `ValueError`）。同一 src から複数 `add_edge` を張ると fan-out（並列）になり、下流ノードが並行
    実行される。
  - `add_conditional_edges(src, router, mapping=None, *, default=None, candidates=None)`:
    `router(msg, ctx) -> 行き先` の戻り値で次ノードを選ぶ。`mapping=None` なら戻り値を次ノード名 | `END`
    として直接使う（LangGraph の path_map 無しモード相当）。`mapping` ありなら戻り値を判定キーとして引く
    （bool / int / Enum 等の任意 hashable をキーにでき、文字列限定ではない）。未一致時は `default`（あれば）、
    無ければ実行時に例外。**条件 fan-out**: 戻り値（または mapping の値）に**ノード名のリスト**を返すと、その
    複数ノードを並行起動する（データ依存で 0〜N 個の枝を動的に選べる）。`candidates`（任意のリスト）は
    `mapping=None`（動的にノード名を返す）時の可能な行き先を宣言し、`validate` の到達性・`mermaid` 可視化に使う
    （LangGraph の path_map 相当）。
  - `add_fan_in_edge(sources, dst)`: fan-in（合流）。`dst` は**実際に起動された（activated な）ソースのみ**を
    待って `{走った source名: 出力}` の dict を msg として受ける（条件 fan-out で走らなかった枝はキーごと omit
    し、来ない枝を待ってデッドロックしない）。`dst` は FUNCTION ノードでなければならない（合流ロジックは利用者
    が FUNCTION ノードで書く。reducer は API 化せず位置依存 list も作らない）。`validate` で検査する。
  - ループはノードへ戻るエッジ + `add_conditional_edges` で `END` へ抜ける形で表す（専用 loop API は持たない）。
    無限ループ防止に `recursion_limit`（既定 25）を設け、超過で実行時エラーにする。
- 検査・可視化:
  - `validate(registry)`: (a) 全エッジ端点と AGENT ノードの参照解決（未登録ノード名 / agent 名）、
    (b) `START` からのエッジが 1 本であること・`START` からの到達性および `END` への到達可能性、
    (c) `add_conditional_edges` の mapping 全分岐先・`default`・`candidates` の解決（`mapping=None` の動的返し
    自体は静的検査不能）、(d) `add_fan_in_edge` の全ソース解決および合流先が
    FUNCTION ノードであること、(e) `recursion_limit` の存在を build-time に検査し、誤りを集約報告する
    （`registry.validate` の集約報告パターンに倣う）。入出力型の接続整合の静的検査は初版対象外であり、
    `validate` が通っても実値の型は保証しない（docstring に明記）。
  - `mermaid()`: ノード（AGENT / FUNCTION）とエッジ（通常 / 条件 / fan-in）・`START` / `END` を表す Mermaid
    文字列を返す。条件エッジは判定キーをラベルに、fan-in は合流を破線等で示す（`handoffs.py` の flowchart
    生成に倣う）。
  - 条件エッジは判定キー（条件 fan-out は複数本）をラベルに、`candidates`（`mapping=None` 時）と fan-in は
    破線で示す。
- Agent / Tool 化:
  - `as_agent_spec(name, *, output_extractor=None, on_node_start=, on_node_end=, ...) -> AgentSpec`（経路C）
  - `as_facade_spec(name, *, mode=FacadeMode.LLM_INPUT, model=None, tool_name=None, output_extractor=None, ...) -> AgentSpec`
    （経路A / D）。ワークフロー tool だけを持つファサード AgentSpec を返す。入口モデルを `FacadeMode` で
    切り替える（`LLM_INPUT`=実 LLM 1 回 / `LLM_INPUT_OUTPUT`=実 LLM 2 回 / `DETERMINISTIC`=実 LLM 0 回・
    決定論。後者は `DeterministicToolCallModel` を内部注入）。`mode=DETERMINISTIC` で `model` 指定は
    ValueError（決定論モデルを注入するため）。handoff 流入の既定 input_filter は facade 自身の
    `handoff_options` に載せても registry が読まないため、`connect_as_facade` を使う。
  - `connect_as_facade(registry, graph, name, src, *, mode=FacadeMode.LLM_INPUT, model=None, input_filter=<既定>, ...) -> AgentSpec`（経路A / D の結線
    一式）: ファサードを `registry.register` し、`src -> facade` の handoff エッジを張る。既定 input_filter
    （直近 1 件）は **handoff エッジ**（registry が実際に読む場所）に載せ、流入履歴をコード既定で有界化する
    （C-10）。明示 `input_filter=None` で全履歴流入（opt-in）。`graph.apply(registry)` は呼び出し側で行う。

### 内部インタプリタと runner シーム

内部インタプリタはメッセージ / エッジ駆動でグラフ（エッジ走査 / 条件分岐 / fan-out / fan-in / ループ）を
解釈する非公開ロジックで、`agents` 非依存。`START` から開始し、各ノードの出力を出辺に沿って下流の入力
（msg）として流す。fan-out 枝は `asyncio.gather` で並行実行する。`add_fan_in_edge` の合流先は**実際に起動
された（activated な）ソースのみ**を待ってから `{走った source名: 出力}` の dict を渡して進む（条件 fan-out で
部分集合しか走らなくても、来ない枝を待たずデッドロックしない。枝の起動は駆動前に一括登録し、同期 FUNCTION
ノードでも待ち数が安定する）。条件エッジは router の戻り値で次ノードを選ぶ（単一・`END`・複数ノードのリスト
＝条件 fan-out）。ワークフローの最終出力は実行経路上で `END` へ到達したノードの出力であり（経路は実行時に
1 つへ収束する）、`output_extractor`（既定 `str`）が Agent 出力（テキスト）へ adapt する。実行中の各ノード出力は
薄い `NodeResults`（ノード名 → 出力の可変記録）に持ち、run 終了で破棄する（run をまたぐ状態を持たない）。
AGENT ノード実行は自前で `agents.Runner` を呼ばず、注入された**非公開 runner シーム**へ委譲する。
runner シームは `async run(agent, input, *, context, **runner_kwargs) -> RunResultLike` の Protocol で、
SDK `Runner.run` への**素通し（passthrough）シーム**である（`input` / `context` のみ lib が明示管理し、
残りの Runner kwarg は `**runner_kwargs` でそのまま委譲する。`protocols.py` の `AgentBuilder` と同様に
トップレベル `__all__` に出さない）。本番実装は `_adapters` の `Runner.run` ラップ、テストは fake を
注入する（`AgentBuilder` / `FakeAgentBuilder` 方式に倣う）。

Runner パラメータ（`session` / `max_turns` / `run_config` / `hooks` / `conversation_id` 等）は 2 段で
宣言する。グラフ既定の `WorkflowGraph(run_defaults={...})` が全 AGENT ノードへ適用され、ノード単位の
`add_agent_node(..., run_options={...})` が dict マージで上書きする。`input` / `context` は lib 管理の
予約キーで両者とも指定不可（`run_options` ではさらに `session` も禁止。session はグラフ既定でのみ設定し
並列ガードを成立させる）。`_exec_node` が `{**run_defaults, **run_options}` を合成して runner シームへ
渡し、シームが `Runner.run` へ素通しする。

DI 拡張点は「生成 = `AgentBuilder`」「実行 = 内部 runner シーム（非公開）」の 2 点に限定する。ノード
前後フック（`(node_name, NodeResults, context) -> None | Awaitable[None]`）は `as_agent_spec` /
`as_facade_spec` の opt-in 引数（`on_node_start` / `on_node_end`）として渡し、内部インタプリタへ引き渡される
（`__init__` 引数を増やさない）。

### WorkflowModel（経路C）と workflow_as_tool / DeterministicToolCallModel（経路A/D）

両者は `_adapters` に閉じた SDK 結合実装であり（`make_dynamic_handoff` の「生 SDK オブジェクトを
`_adapters` 内で直接構築」パターンに倣う）、単なる結線糖衣ではなく次の責任を負う。

- `WorkflowModel(agents.Model)`（経路C）: `get_response(...)` で内部インタプリタを既定 runner
  （`Runner.run`）で回し、最終出力を `ModelResponse`（単一メッセージ・tool / handoff なし）として返す。
  Runner はこれを最終出力として扱いターンを終える。`stream_response` はエンジンを回しきった後に
  最終出力を `ResponseTextDeltaEvent`（逐次表示用）+ `ResponseCompletedEvent`（終端）で流し
  `Runner.run_streamed` に対応する（エンジンが最終値を返す構造のため進捗的ではない
  post-execution streaming）。`Model.get_response` は context を受け取れないため、外側 context
  は engine へ渡さない（context 非伝播のハード制約）。構造化出力は `output_extractor`（既定は最終出力を
  単一メッセージ化）で `ModelResponse` の output を組み立てる。`ModelResponse` 構築ヘルパ
  （`tests/_helpers/responses.py` の `text_response` と同型）を `_adapters` 内に置き既定で利用する。
- `workflow_as_tool(interpret, *, tool_name, tool_description, output_extractor=None) -> FunctionTool`
  （経路A / D 共通）: `on_invoke_tool(tool_context, json)` クロージャ内で `tool_context.context` を内部
  インタプリタへ受け渡し（不変条件。SDK が自動透過しないため配線欠落は経路A / D でのみ共有 context 欠落の
  事故になる）、内部インタプリタを回して最終結果を畳む。各 AGENT ノード内側 run の暴走上限（`max_turns`）
  等の Runner kwarg はグラフ `run_defaults` / ノード `run_options` で握る（passthrough）。
- `DeterministicToolCallModel(agents.Model)`（経路D の入口）: 不変設定 tool 名のみ保持するステートレス
  Model。`get_response` は実 LLM を呼ばず、入力を `latest_user_text` で素テキスト化した
  `{"input": ...}` を引数にワークフロー tool を 1 回呼ぶ ToolCall だけを返す（`tool_call_response` ヘルパ
  経由）。`stop_on_first_tool` 併用前提（無いと tool 結果後に再び ToolCall を返し無限ループ）。可変な実行
  状態を持たないため同一インスタンスを並行 run で共有しても安全。

### 経路C / A / D は spec・registry・handoffs を非破壊で利用

`as_agent_spec` / `as_facade_spec` が返した `AgentSpec` を `registry.register` し
`HandoffGraph.edge(src, name)` を張るという既存経路に乗る（新しい handoff 経路を発明しない。registry に
実行系 / 計算メソッドを足さない）。

- 経路C: `AgentSpec.model`（`Any`）に `WorkflowModel` をそのまま据え、`build_agent` が `Agent(model=...)`
  へ素通しする。
- 経路A / D: `as_facade_spec` が `AgentSpec(tools=[workflow_tool], model_settings=ModelSettings(tool_choice='required'),
  extra=...)` を返す。`extra` は `mode` で変わり、出口要約をしない mode（`DETERMINISTIC` / `LLM_INPUT`）は
  `{'tool_use_behavior': 'stop_on_first_tool'}`、出口で LLM 要約する `LLM_INPUT_OUTPUT` は `{}`（SDK の
  `reset_tool_choice` 既定 True が 2 ターン目の tool_choice を解除し無限ループを防ぐ）。`DETERMINISTIC` は
  さらに `AgentSpec.model` に `DeterministicToolCallModel` を据える。既定 input_filter は facade の
  `handoff_options` に載せても registry が読まないため、`connect_as_facade` が **handoff エッジ**
  （`graph.edge(src, facade, input_filter=...)`）に載せる。
  `tool_choice` は `agents.ModelSettings` のフィールドであり `extra` に積むと `build_agent` の未知キー
  ガードで `ValueError` になるため、必ず `model_settings` 経由で設定する。`tool_use_behavior` は `Agent`
  フィールドであり `extra` で渡せる（非対称）。

`spec.py` / `registry.py` / `handoffs.py` / `prompts.py` の責務・API は変更しない（spec は extra 素通しの
確認のみ、フィールド追加なし）。

### handoff 流入 4 経路

ワークフローを handoff の流入先にする経路は 4 つあり、要件に応じて選択する。handoff の流入先は必ず
`Agent` でなければならない（SDK 制約）ため、`WorkflowGraph` をそのまま handoff ターゲットにはできず、
いずれかの形で Agent 化する。経路A / D はいずれも `as_facade_spec` の tool ファサードで、入口モデルを
`FacadeMode` で切り替えたもの（同一機構の 3 mode のうち 2 つ）。

| 経路 | 起点 | 決定性 | 外側 context 伝播 | 流入時 LLM 層数 | エンジン制御 |
|---|---|---|---|---|---|
| 経路C（主軸） | `WorkflowModel` を据えた Agent | 決定論起動 | 非伝播（ハード制約） | 0（LLM を呼ばない） | あり |
| 経路A（補完） | `as_facade_spec(mode=LLM_INPUT \| LLM_INPUT_OUTPUT)` | 非決定（LLM 1〜2 回） | 透過 | 1〜2 | あり |
| 経路D | `as_facade_spec(mode=DETERMINISTIC)` の決定論ファサード | 決定論起動 | 透過 | 0（LLM を呼ばない） | あり |
| 経路B（軽量） | raw Handoff（エントリ Agent 直接） | エントリ Agent 依存 | エントリ Agent が受領 | エントリ Agent 依存 | なし |

- 経路C: `WorkflowModel` が LLM を呼ばずエンジンを回すため決定論的に起動する。外側 run の共有 context は
  ワークフロー内ステップへ伝播しない（`Model.get_response` に context 引数が無い SDK ハード制約）。tool
  往復を挟まず最終出力を直接返すため、流入アイテムが最小で session 履歴を汚さない。
- 経路A: 外側 context を透過できる代わりに、ファサード Agent が実 LLM を呼ぶため起動の決定性は保証され
  ない（`mode=LLM_INPUT` は入力整形で LLM 1 回・出口要約なし、`mode=LLM_INPUT_OUTPUT` は入力整形 +
  tool 結果要約で LLM 2 回）。流入履歴は lib 提供の既定 `input_filter`（直近 1 件）で自動有界化する。
- 経路D: 入口に決定論ステートレスモデル（`DeterministicToolCallModel`）を据え、実 LLM を呼ばずに毎回
  ワークフロー tool を強制発火する。経路C が埋められない「決定論 + 外側 context 透過 + LLM 0 回」を満たす
  （経路C の上位互換ではなく、tool 往復 1 回ぶんのアイテム生成・session 履歴蓄積と引き換え）。
- 経路B: エンジン制御が不要なケース向けの最軽量経路。既存 `HandoffGraph.edge(src, entry_agent)` で表現
  でき新 API を要しない。

経路選択指針: context 透過が不要なら経路C（最軽量・履歴クリーン）、決定論を保ったまま context 透過したい
なら経路D、入力/出力を実 LLM に整形させたいなら経路A。詳細な検討経緯は
`docs/rationale/workflow-handoff-inflow.md` を参照。

### build-don't-run の線引き

lib が駆動してよいのは宣言済みグラフの制御フロー解釈（エッジ走査 / 条件分岐 / fan-out / fan-in / ループ）
までである。リトライ / タイムアウト / 再開 / スケジューリング等の実行ポリシーはエンジンに内蔵せず、ノード
前後フックの差込口に留める（利用者が書く）。前者は宣言の決定論的展開、後者は外部状態・時間・失敗に依存する
副作用管理であり、薄さ・SDK 隔離原則の責任範囲を超えるためである。途中再開（mid-workflow resume /
checkpoint）も持たない。

### データ受け渡しと継続

- ノード間データはノード出力が出辺に沿って下流ノードの入力（msg）として流れるメッセージ受け渡しで運ぶ
  （MS Agent Framework 型。LangGraph の `return 状態dict` は採らず戻り値そのものが下流へ流れる）。通常ノードの
  msg は上流の単一出力、fan-in ノードの msg は `{source名: 出力}` の dict、`START` 直後のノードでは
  `Runner.run(input=...)` の入力が msg になる。実行中の各ノード出力は薄い `NodeResults`（ノード名 → 出力の
  可変記録、run 終了で破棄）に持つ。独立 state（reducer）機構は新設せず、非隣接ノードの値は fan-in か共有
  context で明示的に運ぶ。共有 context は lib が型を規定せずジェネリックに透過し、各 AGENT ノードの
  `runner.run(context=...)` と各 FUNCTION ノードの `ctx` へ素通しする（経路C では C-11 により ctx は届かない）。
- 会話履歴は既定で引き回さない（前段出力を明示的に input として渡したときのみ繋がる）。`session` は
  グラフ既定 `run_defaults={"session": ...}` で宣言したときのみ SDK Session へ委譲する（opt-in）。
  fan-out（並列）を含む `WorkflowGraph` に session opt-in を宣言すると、SDK Session の `add_items` が
  並行安全を保証しないため、run-entry の静的ガード（通常エッジ由来 fan-out）と `_drive` の実行時ガード
  （条件 fan-out でリスト返しした時点）の両方で明示的に拒否する（fail-fast）。session はノード単位の
  `run_options` では設定できない（グラフ既定でのみ握り、並列ガードの判定軸を一本化する）。
- マルチターン継続は `RunResult` 駆動とし、lib は継続 / 再開状態を持たない。利用者は
  `Runner.run(result.last_agent, result.to_input_list() + 新入力, context=...)` で継続する。経路C の
  ワークフローは外から見て 1 Agent であるため `last_agent` はワークフロー Agent 自身を指し、次ターンは
  ワークフローの再実行になる（途中再開はしない）。
- ステートレスなコア（registry / workflow）と、会話状態（SDK `Session`）を保持する会話サービス（上位の
  dev 便宜層）は層が異なる。会話サービスは履歴保持・途中再開を SDK `Session` に委ね、コア（lib）が継続 /
  再開状態を持たないことと矛盾しない。会話サービスの構成は「会話 Helper（ローカル開発支援）」節を参照。

### ワークフロー tracing

ワークフロー 1 run は SDK tracing 上で 1 つの workflow span に集約され、ノード / 条件 / fan-out / fan-in が
子 span として現れる。ユーザー側のコード変更・追加設定は不要で、`as_agent_spec` / `as_facade_spec` /
`connect_as_facade` 経由で組み立てたワークフローを `Runner.run` で実行すれば自動的に span が記録される。

- 全 span は SDK の `custom_span(name, data=...)` で発行する（`agent_span` / `function_span` は使わない。
  AGENT ノードを `agent_span` で包むと内側 `Runner.run` が自動発行する agent span と二重ネストになる・
  FUNCTION ノードは「LLM が tool として呼んだ関数」セマンティクスの `function_span` とズレ標準属性スキーマの
  追従負荷が生じるため）。
- span name は `workflow.` プレフィックス統一: `workflow.run.<graph_name>` / `workflow.node.<node_name>` /
  `workflow.condition.<src>` / `workflow.fan_out.<src>` / `workflow.fan_in.<dst>`。`graph_name` 未設定時は
  `"anonymous"` をフォールバックに使う。ノード種別は name に埋め込まず data 属性 `workflow.node_kind` で持つ
  （種別追加時に name 規約を変えない）。
- data 属性は OpenTelemetry 風 namespace `workflow.<key>` で統一。必須は `workflow.graph_name` /
  `workflow.node_kind`、ノード関連 span は `workflow.node_name` も。任意で `workflow.step` /
  `workflow.condition.selected` / `workflow.fan_out.parallelism` / `workflow.fan_in.arrivals` を載せる。
- AGENT ノードの内側 `Runner.run` は SDK の `get_current_trace()` 経由で workflow span 配下に親子接続される
  （独立トレース化しない）。RunnerSeam 契約および runner adapter は変更せず、span を with でラップするだけで
  親子関係が確立する。
- tracing 無効時（`set_tracing_disabled(True)` または `get_current_trace() is None`）は no-op
  コンテキストマネージャのみが介在し span オブジェクトを生成しないため、オーバーヘッドを持たない。
- tracing API（`custom_span` / `get_current_trace`）の import は `_adapters/tracing.py` に局在化する
  （NFR-1 維持）。workflow 層は `_adapters.WorkflowTracer` / `make_workflow_tracer()` のみを関数内遅延
  import で取得する。
- tracer は run ごとに新規生成するステートレス factory。run スコープ状態は tracer 側に持たず、span の親子
  関係は SDK の contextvars に任せるため、複数 run 並行でも span 親子関係が混線しない。
- 条件 fan-out では condition span を分岐評価のみに限定して直後に閉じ、続けて fan-out span を兄弟として
  開く並列構造を採る（SDK の span は LIFO スタック前提・condition の評価結果が fan-out に消費される時点で
  condition span は寿命を終える）。
- `WorkflowModel.stream_response` は post-execution streaming であり、`interpret` 完了時点で workflow span は
  閉じ済みのため stream generator 再エントリで span が残ることはない。
- `on_node_start` / `on_node_end` フックは観測の補完として並走し、span 発行とは独立に保つ（span 開始 →
  `on_node_start` → 実行 → `on_node_end` → span 終了）。フック契約・公開 API は不変。
- ノード名・グラフ名は span name と data 属性に乗り外部 trace backend（OpenAI tracing / Langfuse 等）へ
  送信されるため、PII / 秘密文字列をノード名 / グラフ名に含めない（命名は宣言時にユーザーが完全コントロール
  可能・ライブラリ側でのサニタイズは観測性を壊すため行わない）。

## Realtime エージェント（専用宣言ルート）

Realtime エージェント（`agents.realtime.RealtimeAgent`・音声エージェント）を宣言的に扱う専用ルートを
`src/oai_agentspec/realtime/` に置く。通常の `AgentSpec` / `AgentRegistry` とは共用せず、RealtimeAgent が
非対応とするフィールドをそもそも型として持たない専用宣言型・専用 registry・専用公開窓口を提供する。
宣言と実行の分離という流儀を Realtime でも保ちつつ、非対応フィールドは型レベルで排除し、型で排除しきれない
経路のみ build 時に reject する。

### モジュール配置と依存方向

`realtime/` はコア（宣言層）・runtime（実行寄り層）のいずれの群にも属さない第 3 の並列宣言ルートであり、
コア公開 API ツリー外・専用窓口（`oai_agentspec.realtime`）経由で提供する。専用 registry が `_adapters` を
上向きに参照する単方向依存の形は runtime と同型だが、配置はコア直下の宣言層である。

```
realtime/__init__   →  { realtime/registry, realtime/spec, realtime/protocols }
realtime/registry   →  { realtime/spec, realtime/protocols, _adapters }
realtime/spec, realtime/protocols  →  （最下層・agents 非依存）
_adapters/realtime  →  agents.realtime（SDK 結合はここに閉じる）
```

- SDK import（`from agents.realtime import RealtimeAgent, realtime_handoff`）は `_adapters/realtime.py`
  にのみ置く。`realtime/spec.py` / `realtime/registry.py` / `realtime/protocols.py` / `realtime/__init__.py`
  は plain データと不透明型のみ扱う（SDK 隔離）。
- コアから `realtime/` への依存辺は持たない。`oai_agentspec/__init__.py` に `realtime` の import を
  追加せず、`import oai_agentspec` は `oai_agentspec.realtime` を連鎖 import しない（遅延 import 境界）。

### RealtimeAgentSpec / RealtimeHandoffConfig

`RealtimeAgentSpec` は RealtimeAgent が対応するフィールドのみを持つ宣言 dataclass で、`agents` 非依存。
非対応フィールド（`model` / `model_settings` / `input_guardrails` / `sub_agents` / `sub_agent_tools` /
`dynamic_handoffs`）は型として持たない（第一防御・型レベル排除。`dataclasses.fields` に含まれない）。

| フィールド | 型 | 役割 |
|---|---|---|
| `name` | `str` | エージェント名（registry 内で一意） |
| `instructions` | `str \| Callable \| None` | システムプロンプト。文字列、または `(context, agent)` の 2 引数 callable |
| `prompt` | `Any \| None` | 静的 Prompt のみ。callable（`DynamicPromptFunction`）は非対応（register 時 / build 時に `ValueError`） |
| `tools` | `list` | RealtimeAgent に渡すツール |
| `hooks` | `Any \| None` | RealtimeAgent フック |
| `output_guardrails` | `list` | 出力ガードレール（`input_guardrails` は持たない） |
| `handoff_description` | `str \| None` | AgentBase 由来のハンドオフ説明 |
| `mcp_servers` | `list` | AgentBase 由来の MCP サーバ |
| `mcp_config` | `dict` | AgentBase 由来の MCP 設定 |
| `handoffs` | `list[str]` | ハンドオフ先エージェント名（グラフ連携） |
| `handoff_options` | `dict[str, RealtimeHandoffConfig]` | dst 名 -> per-edge 設定 |
| `extra` | `dict` | 上記以外の RealtimeAgent kwarg の検証付き前方互換口。現 SDK では RealtimeAgent の全フィールドが専用フィールド化されており非空 `extra` は常に reject される（SDK が将来フィールドを追加した場合にのみ素通しが機能する） |

`RealtimeHandoffConfig`（frozen dataclass）は `on_handoff` / `input_type` / `tool_name_override` /
`tool_description_override` / `is_enabled` を保持し、`input_filter` を型として持たない（`realtime_handoff()`
が `input_filter` 非対応のため）。通常ルートの `HandoffConfig` / `DynamicHandoff` のフィールド対称性契約とは
別物であり、その対称性テストの対象外である。

### 専用 registry と handoff 結線

`RealtimeAgentRegistry` は `register` / `get` / `names` / `validate` / `entry_name` を持ち、通常 registry の
2 パス遅延バインドを踏襲する。依存辺は `spec.handoffs` のみ（`sub_agents` / `dynamic_handoffs` を持たない）。

- `register` は spec の保存のみを行い、同名の重複登録は `ValueError`。
- `get(name)` は到達可能かつ未ビルドの spec を `handoffs` のみ辿って収集し（visited 集合で循環を打ち切る）、
  パス 1 で各 spec を `handoffs=[]` でビルド、パス 2 で `realtime_handoff()` により handoff を後付け結線する。
  結線中の例外では本呼び出しで新規ビルドした agent を巻き戻す（トランザクショナル）。
- `validate()` は全 spec の `handoffs` 参照が既知名かを一括検証し、未解決を集約して報告する。
- `register` は次を検証し、違反はエージェント名・エッジ名入りの `ValueError` で前倒し reject する
  （SDK `realtime_handoff()` の厳格検査を build/run より前に引き上げる）。
  - callable `instructions` は `(context, agent)` の 2 引数で呼び出せること（デフォルト引数・可変長は
    許容。シグネチャ取得不能な callable は検証をスキップし実行時に委ねる）。
  - `prompt` が callable の場合は `ValueError`（build 時の第二防御も併存）。
  - `handoff_options` のキーが `handoffs` に存在しない場合は `ValueError`（per-edge 設定の silent drop 防止）。
  - `input_type` 指定時は `on_handoff` が必須。
  - `on_handoff` の引数個数は `input_type` ありで 2・なしで 1。
- 上記のうち handoff 系検証（`handoff_options` のキー整合・`input_type`→`on_handoff` 必須・`on_handoff` の
  arity）は `_validation` の共有バリデータへ一元化され、`register`（`_validate_spec` 経由）と後述の
  `RealtimeHandoffGraph.apply` の双方から呼ばれる（反映順序に依らず最終 spec が検証される）。
- デフォルトビルダーは関数内遅延 import で取得し、registry import 時点で `agents.realtime` を読み込まない。

### 宣言的ハンドオフグラフ（RealtimeHandoffGraph）

`RealtimeHandoffGraph` はハンドオフのトポロジを宣言し、`RealtimeAgentSpec` 群へ `apply(specs)` して各 src の
`handoffs`（名前リスト）と `handoff_options`（dst -> `RealtimeHandoffConfig`）を書き込む宣言アーティファクトで
ある。registry は変更せず（built 無効化・freeze を要さない）、`spec.handoffs` に名前を直接宣言する場合と構造的
に同一の結線になる。

- `apply(specs)` は 2 パスで反映する: パス 1 で全 src の反映値を組み立てて handoff 系検証を `_validation` の
  共有バリデータで実行し、パス 2 で一括代入する。途中の失敗ではどの spec も変異しない（原子性）。`register`（`_validate_spec` 経由）も同じバリデータを共有するため、`apply → register`
  でも `register → apply` でも最終 spec が必ず検証される（反映順序に依らず検証は迂回されない）。src がグラフに
  現れるが `specs` に無い場合は `KeyError`。
- build 前ワンショット反映の制約: apply は spec のみを書き換え、registry のキャッシュには関与しない。
  `registry.get()` で構築済みのエージェントはキャッシュから返るため、構築後に apply しても既存エージェントの
  結線は変わらない。apply は必ず最初の `get()` より前に行う。再 `apply` 時に前回反映して今回消えた src の
  `handoffs` を自動クリアしない（一回性）。エッジを持つ src の `handoffs` のみを replace 上書きし、エッジを
  持たない src の spec には触れない。
- `edge()` の引数は `RealtimeHandoffConfig` のフィールドへ次のとおりマップする。`input_filter` は露出させない。

  | `edge()` 引数 | `RealtimeHandoffConfig` フィールド |
  |---|---|
  | `on_handoff` | `on_handoff` |
  | `input_type` | `input_type` |
  | `tool_name` | `tool_name_override` |
  | `tool_description` | `tool_description_override` |
  | `is_enabled` | `is_enabled` |

- `mermaid()` は静的エッジのみを `flowchart TD` として返す（`start([start]) --> {entry}` / `{src} -->|{label}| {dst}`）。
  ラベル源は `tool_description_override`（未設定時は無ラベル）で、動的エッジ破線・`input_filter` を持たない。
- `from_specs(specs, entry=None)` は各 spec の `handoffs` から静的エッジを張ってグラフを構築する。
- コアの `HandoffGraph` とは統合しない独立アーティファクトであり（`input_filter`・動的エッジを型として持たず、
  registry 内部プリミティブへ委譲しない）、相互の対称性契約の対象外である。

### build 時の reject（型で排除しきれない経路）

`_adapters/realtime.py` の `build_realtime_agent` / `make_realtime_handoff` が第二防御を担う。

- `extra` に専用フィールドと同名のキー、または RealtimeAgent / AgentBase が受け付けない未知キー
  （`model` / `model_settings` / `output_type` / `tool_use_behavior` / `input_guardrails` 等）が含まれる
  場合は `ValueError`（agent 名と該当キー名を含む）。有効 kwarg は `RealtimeAgent` の dataclass フィールドから
  導出する。判定・メッセージは通常ルートと共有の `_validation` leaf（`validate_extra_kwargs`）に一元化する
  （既存の `ensure_static_prompt` 共有委譲と同流儀。各アダプタは算出済みフィールド名 frozenset とラベル文字列を渡す）。
- `RealtimeHandoffConfig` は `input_filter` / `options` / `extra` を型として持たないため、非対応の
  `input_filter` を渡す経路自体が存在しない（型レベル排除で完結し、実行時 reject は不要）。
- 未指定（None / 空）のフィールドは kwargs に積まず RealtimeAgent の既定に委ねる（明示的に None を渡さない）。
  ただし `instructions` は例外で、None でも明示的に渡す（RealtimeAgent の既定も None のため挙動は等価）。

`make_realtime_handoff` は `realtime_handoff(agent, tool_name_override=..., tool_description_override=...,
on_handoff=..., input_type=..., is_enabled=...)` へ委譲する（`input_filter` は渡さない）。実行は SDK
`RealtimeRunner` に委ね、lib は build に徹する（build-don't-run）。

### 公開窓口

Realtime シンボルはコア `__all__` に載せず、`oai_agentspec.runtime.conversation` と同様に
`oai_agentspec.realtime` の公開窓口で参照する。`realtime/__init__.py` は再エクスポート専用で、`__all__` に
`RealtimeAgentSpec` / `RealtimeHandoffConfig` / `RealtimeAgentRegistry` / `RealtimeHandoffGraph` /
`RealtimeHandoffEdge` / `from_specs` を掲載する。利用側は
`from oai_agentspec.realtime import RealtimeAgentSpec, RealtimeAgentRegistry` で取得する。
`RealtimeAgentBuilder`（DI 拡張点の Protocol）はコア `AgentBuilder` と同様どの `__all__` にも載せず、
`from oai_agentspec.realtime.protocols import RealtimeAgentBuilder` の直接 import でのみ参照する。
新規 extra は設けない（`agents.realtime` は既存 `agents` SDK に同梱）。

## パラメータのカスタマイズ

主要パラメータは専用フィールドで保持し、それ以外は dict 素通しで SDK の全パラメータを
カスタム可能にする一貫したパススルー方式を採る。

- **Agent**: 専用フィールド（`instructions` / `prompt` / `tools` / `model` / `model_settings` /
  `hooks` / `handoffs` / `sub_agents`）+ `extra: dict`（`output_type` / `input_guardrails` /
  `output_guardrails` / `tool_use_behavior` / `reset_tool_choice` / `handoff_description` /
  `mcp_servers` 等）。`extra` で `model_settings` 等を渡す場合は `ModelSettings` インスタンスが必須。
- **Handoff**: `HandoffConfig` の型付きフィールド（`description` / `tool_name` / `on_handoff` /
  `input_type` / `input_filter` / `is_enabled`）+ `options: dict` 素通し。静的エッジ
  （`HandoffEdge` / `HandoffConfig`）と動的エッジ（`DynamicHandoffEdge` / `DynamicHandoff`）は
  `description` / `on_handoff` / `input_type` / `input_filter` / `is_enabled` / `options` の 6
  フィールドを同名・同型で共有し、両エッジで同一の「型付きフィールド + `options` 素通し」方式に
  乗る。運用ルール: 型付きフィールド優先・`options` は裏口で同義キー（静的なら `*_override` 名・
  動的なら生 `Handoff` フィールド名）禁止。`strict_json_schema` のような型付き化されない SDK 固有
  引数は `options` 経由で渡す。動的エッジでは strict 化のスキーマ反映タイミング上、
  `options["strict_json_schema"]` を schema 生成段階でも参照する。

## 名前参照の検証

`register` は spec の保存のみを行い、ビルドはしない。`get(name)` 時に到達可能 spec を局所 2 パスで
ビルドするため、`handoffs` / `sub_agents` の参照先を後から `register` してもよい（最初の `get` までに
すべての参照名が register 済みであればよい）。未登録名はビルド時に文脈付きエラーとなる。

`registry.validate()` は全 spec の `handoffs` / `sub_agents` 参照と動的ハンドオフ候補が解決可能かを
一括検証し、未解決の参照をすべて集約して報告する。run 前に呼ぶことで名前のタイポ等を早期に検出できる。

## ランタイム差し替え

起動後の Agent / テンプレートのホットスワップを支える API。
スレッド安全性は単一スレッド / 単一イベントループ前提とし、並行制御は利用者責任とする。
update は次回 `get()` から反映され、進行中の run が掴む旧インスタンスには影響しない。

- `update(spec)`: spec を置換し当該 Agent の built を無効化する。さらに当該 Agent を handoff /
  sub_agent 先に持つ全 Agent を依存逆引きで連鎖 invalidate する（visited 集合で循環を打ち切る）。
  これにより `get("b")` から到達する循環先 Agent も新インスタンスになる。
- `unregister(name)`: spec と built を削除し、依存元も invalidate する。次回 `get(name)` は `KeyError`。
- ハンドオフのトポロジ変更: `HandoffGraph` を編集して `apply(registry)` で再反映する（内部では
  `_update_handoffs` が replace 適用し当該 Agent と依存元を invalidate する）。
- `PromptStore.reload()`: テンプレートキャッシュをクリアし、ファイル変更を次回 `compose()` / `render()`
  に反映する。

注意: `compose` の静的 instructions（焼き込まれた文字列）は `reload()` だけでは更新されない。
`registry.update()` または再ビルドが別途必要である。稼働中の動的更新は `compose(vars=callable)` の
動的 instructions（毎 run render）経路を推奨パターンとする。

## runtime インテグリティ防御

稼働中（プロセス起動後）のディスク上ファイル改竄と `AgentRegistry` / `WorkflowGraph` への動的書換を
fail-closed に検知 / 遮断する。コア層に `integrity` モジュールを追加し、`lockdown` 1 関数を公開窓口と
する。`lockdown` は root verify + store verify+preload + libs detect + custom checks + registry/workflow
freeze を 6 段順次・fail-closed で実行する。

`AgentRegistry.freeze()` / `WorkflowGraph.freeze()` はクラス経由の公開メソッドであり、`__all__` には
掲載しないが公開契約として安定（SemVer 対象）。利用者は `lockdown` を使わず単独でこれらを呼んでよい。

`PromptStore.__init__` のシグネチャは不変であり、検証 / preload は `lockdown` 経由で `_verify_integrity`
/ `_preload` 非公開メソッドが発火する（既存 `PromptStore` 利用者の挙動は変わらない）。詳細・公開 API
シグネチャ・6 段順次処理・例外階層・典型構成・Out of Scope は `docs/integrity.md` を参照。

## 会話 Helper（ローカル開発支援）

registry に登録済みのエージェントとローカルで会話して動作を確かめるための上位利用支援層。
**クライアント・サーバ型**で構成し、CLI と API の挙動を構造的に一致させる。本番運用・外部公開・認証は
スコープ外であり、コア（registry / workflow）の挙動は変更しない。

### 構成（クライアント・サーバ + 共有コア）

3 コンポーネントで構成し、いずれも `runtime/` 配下に置く。

- **会話サービス（`runtime/conversation` 公開窓口・共有コア・`agents` 非依存）**: 利用者提供の
  `AgentRegistry` を受け取り、会話
  相手の一覧・エントリ（起点）エージェントの決定、会話の作成 / 復元、session 単位の履歴保持・履歴取得、
  会話の開始 / 継続（ストリーミング / 非ストリーミング）を提供する単一実装点。サーバ入口のストリーミング・
  非ストリーミング両経路がこのコアへ委譲することで、機能仕様を一致させる（NFR-3 の核）。会話状態
  （`conversation_id` 単位のエントリ）はサーバプロセス内の最小状態として保持し、会話ごとの排他ロックで
  同一会話の同時実行を防ぐ。SDK `Session` は不透明型で保持し `agents` を import しない。会話 CRUD /
  送信 / ストリーミング / エージェント解決を担う本体と、HITL の承認待ち保持・永続復元・調停を担う
  承認コラボレータをサブモジュールへ分離し、本体がコラボレータを保持・委譲する（承認の意味論・委譲構造は
  不変）。
- **エントリ（起点）エージェント**: 会話は「エントリエージェント」を起点に始める。`AgentRegistry` は登録順
  の先頭を `entry_name` で公開し、`ConversationService(entry_agent=...)`（無指定で `entry_name`）で起点を
  決める。会話送信時にエージェント名を省略（`send`/`stream` の `agent_name=None`・REST/WS の `agent_name`
  省略）するとサービスがエントリへ解決する。CLI はエージェントを選ばせず常にエントリ起点で会話する。
- **サーバ入口**（`runtime/serve`・`serve` extra 配下・公開 API ツリー外）: localhost 既定・認証なしの
  FastAPI サーバ。
  REST（エージェント一覧 + エントリ名 / 会話作成 / session メタ一覧 / session 履歴取得 / 非ストリーミング
  会話）と WebSocket（ストリーミング会話。1 ターンを turn -> token 逐次 -> done / error の流れで進める）を
  提供する。`runtime/conversation` の会話サービスへ委譲するのみで、自前の会話実行ロジックを持たない。
  `start_server` は
  `AgentRegistry` 渡し時に `session_policy` / `entry_agent` を内部サービスへ適用する。serve 層は
  `routers/`（REST / WS ルーティング）・`dependencies`（サービス依存解決）・`mappers`（plain と
  スキーマの変換）・`errors`（エラー写像）・`app`（router 登録のみの app factory）・`server`（起動）の
  標準 FastAPI 構成を採り、`__init__.py` を薄い再エクスポート窓口とする（エンドポイントのパス / メソッド /
  入出力スキーマと `runtime/conversation` 会話サービスへの委譲構造は不変）。serve から会話サービスへの
  参照は同一 `runtime/` 直下の兄弟（`runtime/serve -> runtime/conversation`）として相対 import で結ぶ。
- **CLI クライアント**（`runtime/cli`・`cli` extra・`[project.scripts]`・公開 API ツリー外）: サーバに接続する**別プロセス**
  のクライアント。in-process 実行はしない。`rich` による 2 層 UI で、起動時にセッション選択画面（過去
  session を最終更新 / ターン数 / プレビュー付きの表で提示）を出し、新規会話 / 過去 session 復元を選ばせ、
  会話画面で対話する（`/back` 選択へ戻る・`/quit` 終了・`/help`）。復元時は直近 N 件の履歴を表示する。会話は
  エントリ起点（エージェントを選ばない）。既定はストリーミング（WebSocket でトークン逐次表示）、非
  ストリーミング（REST で最終応答取得）も選択できる。

### 会話モード

- **ストリーミング = WebSocket**: 双方向接続でトークンを逐次受信・表示する。
- **非ストリーミング = REST**: 1 リクエストで最終応答を返す。

会話サービスは両モードを単一実装点として提供する（`_adapters` の `run_streamed` / `run` に対応）。

ストリームの plain 型（`StreamDelta` / `StreamDone` / `StreamError` と Union の `StreamEvent`）は agents
非依存の中立 dataclass であり、その定義位置は実装で確定する。コア（`_adapters` を含む）は `runtime/` を
import しない単方向依存を保ち、`StreamEvent` 等は `from oai_agentspec.runtime.conversation import ...` の
公開窓口で参照する。

### 履歴と session

- **履歴機構は SDK `Session`** を用い、利用者は任意の `Session`（ローカル SQLite / OpenAI サーバ保持 /
  圧縮ラッパー / 自前）を注入できる。
- **既定は永続化・session_id 連動**: 既定で file 永続化し再起動後の途中再開ができる。永続化先は
  `SessionPolicy`（既定 `memory/conversations.db`・プロジェクト直下の見えるフォルダ。隠しホーム
  フォルダには置かない）で決める。保存先の優先順は `serve --session-db <path>`（明示フラグ）>
  環境変数 `XDG_DATA_HOME`（設定時はそのフォルダ直下 `$XDG_DATA_HOME/conversations.db`）> 既定
  `memory/`。env 参照は CLI 境界に閉じ、本体 `SessionPolicy` / `ConversationService` は環境変数に
  依存しない。`SessionPolicy.persist=False`（CLI `serve --ephemeral`）にすると session_id 明示でも
  常に in-memory（プロセス内・揮発）になる。session_id 無指定の会話も in-memory。
- **compaction（履歴圧縮）の有効化は明示フラグで制御する**。`SessionPolicy.compaction` に渡す
  `CompactionConfig` の `enabled` フラグで有効化を明示制御し、有効化判定と client / model の受け渡しを
  分離する。`compaction=None` または `enabled=False` なら plain `SQLiteSession`（このとき client / model
  を渡しても圧縮しない）。`enabled=True` かつ `client` 欠落は `ValueError`（`CompactionConfig.__post_init__`
  で型構築時に早期検知し、`make_session` 側でも防御する）。compaction は OpenAI Responses 専用で、`client` は
  `AsyncOpenAI` / `AsyncAzureOpenAI` のいずれでも Responses API を叩ければ動く。`CompactionConfig`
  （`@dataclass(frozen=True)`・`enabled` / `client` / `model` / 素通しオプション）は会話層の plain 型で
  `_adapters` を import せず、`SessionPolicy` が `make_session` の plain 引数へ展開して渡す（依存方向は
  `runtime/conversation -> _adapters` の単方向を維持し、`make_session` は `CompactionConfig` を
  import しない）。
- **session 一覧 / 復元 / 履歴取得**: 過去の file 永続化会話の列挙とメタ導出（最終更新時刻 / ターン数 /
  先頭発話プレビュー）・履歴アイテム取得は、SDK が書く SQLite db（`agent_sessions` / `agent_messages`）への
  素の SELECT で実現し、自前の別インデックス / state ストアは新設しない（NFR-2 と整合）。`list_sessions` は
  `SessionInfo` の列を、`session_history(session_id, limit)` は直近 N 件の履歴アイテムを返す。利用者は一覧
  から過去会話を選んで続きから再開でき、復元時に過去履歴を表示できる。in-memory（揮発）会話は列挙対象外。
  SDK 内部スキーマ前提に結合するため、前提が崩れたら CI で検知する SDK バージョン耐性トリップワイヤ
  （NFR-7）を設ける。
- **HITL 中断状態の同居**: HITL の中断状態（`RunState` JSON）は同一 conversations.db の専用テーブル
  `oai_agentspec_pending_approvals`（列 `session_id`（PRIMARY KEY）/ `agent_name` / `run_state`（JSON）/
  `pending`（`call_id`・`tool_name` の JSON）/ `updated_at`）に同居させ、SDK が書く `agent_sessions` /
  `agent_messages` とは独立した自前テーブルとして素 SQLite CRUD する（別ファイル / 別 state ストアは
  新設しない）。upsert / 取得 / 削除はいずれも `session_id` をキーにする。CLI のセッション復元は同じ
  `session_id` に新しい `conversation_id` を振るため、`session_id` をキーにしないと再起動跨ぎ復元（FR-10）で
  承認待ちが見えなくなる。復元時は専用テーブルから `RunState` JSON を読み出し、永続レコードに保存した
  `agent_name`（中断状態を生んだ解決済みエージェント名）を registry 解決した `initial_agent`（未保存/未解決なら
  エントリエージェント）を与えて中断状態を復元する。本テーブル自体は SDK 内部スキーマに結合しないが、
  `RunState` の `to_string` / `from_string` の JSON 形式は SDK 結合点であり、SDK 変更で静かに退行しないよう
  NFR-7 の SDK バージョン耐性トリップワイヤの監視対象に含める。

### HITL（ツール実行承認）

利用者が宣言した承認必須ツールの実行を、人間の承認（approve）または却下（reject）を経てから行う安全機構。承認はツール呼び出し（`call_id`）単位で行い、却下時はツールを実行せず会話を継続する。WebSocket（ストリーミング）と REST（非ストリーミング）の両モードで同一の意味論で機能し、`ConversationService` の同一 approve/reject 処理点へ委譲する（NFR-3）。承認待ちの検知・承認/却下による再開・中断状態のシリアライズは openai-agents SDK の既存プリミティブの流用で実現し、独自の会話実行エンジン・独自の中断状態スキーマは新設しない（NFR-2）。SDK 型・関数の import は `_adapters` 配下のみに閉じ、会話サービス・サーバ入口・CLI クライアントへは plain 型のみを渡す（NFR-1）。

- **承認必須ツールの宣言**: `function_tool(..., needs_approval=True | callable)` で宣言したツールを `AgentSpec.tools` に載せる。`function_tool` は `_adapters` から再エクスポートして公開し、利用者は `from oai_agentspec import function_tool` で agents を直接 import せず承認必須ツールを作れる。`needs_approval=True`（bool）が主経路、`needs_approval=callable`（実行時に SDK の `ToolContext` 等が渡る述語）は SDK 型露出を伴う上級用途。`AgentSpec` に承認専用フィールドは追加しない（tools に載せるだけで足りる）。
- **承認待ちの検知**: 会話実行後に SDK の interruptions を確認し、承認待ちがあれば各承認待ちから `call_id` / `tool_name` を抽出した plain 一覧と不透明な中断状態（`RunState`）を `_adapters` が返す。承認待ちなしターンは従来どおり最終出力を返す（NFR-6）。
- **WS 通知**: ストリーミングで承認待ちが発生したターンは、`token`（`StreamDelta`）逐次の後に承認待ち専用イベント（`approval_required`・対象の `tool_name` / `call_id` 一覧）を送り、`done`（`StreamDone`）は流さない。承認待ち専用イベントは `StreamEvent` Union（`StreamDelta` / `StreamDone` / `StreamError` の 3 メンバ）とは別の専用イベント型であり Union には混ぜない。クライアントは承認/却下を送って再開し、再開後の `token` 逐次 → `done`（または残承認待ちの再提示）へ繋ぐ。
- **REST 取得と判別応答**: `POST /conversations/{id}/messages` の応答 `SendResponse` を破壊せず判別フィールドで承認待ちを表現する。`status: "final" | "pending"`（承認待ち時 `"pending"`）/ `pending: list[{tool_name, call_id}] | None`（final 時 None）/ final 時のみ `output` 文字列を持つ。承認待ち一覧の取得は `GET /conversations/{id}/approvals`（冪等・復元直後や別リクエストの再提示用）。
- **call_id 単位の承認・却下**: 承認/却下は `call_id` 単位で行う。WS は `approval` メッセージ（`conversation_id` + decisions 配列。各要素 `call_id` + approve|reject + 任意 rejection_message）、REST は `POST /conversations/{id}/approvals`（decisions 配列）。部分指定可で、未指定 `call_id` は未解決のまま残る（部分解決）。
- **却下時の会話継続**: reject された `call_id` のツールは実行されず、却下された事実（任意の rejection_message を含む）がエージェントへの継続入力として反映され、会話は中断せず継続する。
- **再開と段階解決**: そのターンの全承認待ちが approve/reject で解決された時点で会話を再開する。一部 `call_id` が未解決の間は再開せず残りの承認待ちを再提示する（段階解決）。再開後に再び承認待ちが残れば再度中断として扱う。複数の承認待ちは `call_id` ごとに区別された一覧で扱い、一部 approve・一部 reject を独立に処理する。
- **中断状態の永続化**: 中断状態（不透明 `RunState` + plain 承認待ち一覧）は `ConversationEntry`（メモリ）に常時保持する。永続会話のみ conversations.db の専用テーブル `oai_agentspec_pending_approvals`（列 `session_id`（PRIMARY KEY）/ `agent_name` / `run_state`（JSON）/ `pending`（`call_id`・`tool_name` の JSON）/ `updated_at`）へ JSON で同居させ、中断発生で `session_id` をキーに upsert・解決完了で `session_id` を条件にした DELETE を行う。揮発（`SessionPolicy.persist=False`）はメモリのみで conversations.db には書かず、復元時に承認待ちは復活しない。
- **排他と復元整合**: 同一会話の承認・却下・継続は会話ごとの `lock` で直列化する（NFR-4）。プロセス内ロックで守れない再起動跨ぎは `session_id` を条件に含めた DELETE / 条件付き更新（db レベル整合）で別会話の中断状態の取り違えを防ぐ。再開完了時は「SDK 履歴コミット → 専用テーブルの中断状態削除」の順で行い、削除失敗時も安全側へ倒す（残った中断状態は未解決の承認待ちとして再提示され収束する。upsert・DELETE はいずれも冪等）。再開（resume）が一時エラーで失敗した場合は、残りの承認待ちを中断状態（`RunState`）の実状から再算出して「全解決・未再開」の状態で保持・再永続し、空 decisions での再 resolve で再開をリトライできる（一時エラーで会話が詰まらない）。
- **復元と initial_agent 解決**: 永続会話を復元する際、専用テーブルに中断状態があれば永続レコードに保存した `agent_name`（中断状態を生んだ解決済みエージェント名）を registry 解決して `initial_agent` を決め、SDK の状態復元（`from_string`）で中断状態を復元し承認待ちを再提示する。`ConversationEntry.agent_name` は再起動跨ぎで失われる（None）ため、永続された `agent_name` を一次情報にする。未保存/未解決ならエントリエージェントへフォールバックする。サーバ再起動跨ぎでも正しいエージェントから続きを再開できる。
- **承認待ち中の新ターン抑止**: 会話に未解決の承認待ちがあるとき、`send` / `stream` は新たな実行を開始せず（session に履歴を追記せず）既存の承認待ちをそのまま返す（`send` は `SendResult(status="pending")`、`stream` は `ApprovalRequired`）。承認を先に解決させることで、古い中断状態を変異した session に対して再開する不整合を防ぐ。
- **CLI 復元時の承認ドレイン**: CLI は会話に入った時点（入力受付の前）で承認待ちを確認し、あれば先に提示・解決してから入力を受け付ける。復元会話で先頭メッセージが捨てられない。
- **スキーマ自己修復**: `oai_agentspec_pending_approvals` は実行中の一時状態テーブルであり、本機能のスキーマ進化で列集合が期待と異なる旧テーブルが残っていれば作り直して自己修復する。会話履歴テーブル（`agent_sessions` / `agent_messages`）には触れない。
- **`SendResult` 型**: `send` / `resolve_approvals` の戻りは `SendResult`（`status: SendStatus`（`"final"` / `"pending"`・`StrEnum` で文字列比較互換）/ `output` / `pending`）。承認待ち時は最終出力でなくこの判別型で承認待ちを表す。公開は `oai_agentspec.runtime.conversation` の公開窓口経由とする。
- **`ApprovalDecision` 型**: `resolve_approvals` / `stream_resolve` への承認/却下入力は型付き `ApprovalDecision`（`call_id` / `approve` / `rejection_message`）で渡せる。`{"call_id", "decision", "rejection_message"}` の plain dict も後方互換で受け付け、内部で `ApprovalDecision` を dict 形へ正規化してから適用する。
- **NFR-7（承認前非実行）**: 承認必須ツールは approve による再開まで SDK が実行を発火しない。ライブラリは「中断検知時に最終出力を返さず、全解決まで再開しない」だけを保証し、独自の実行抑止ロジックは持たない。

承認・却下に関するエラー（未知の `call_id`・解決済み `call_id` への再操作・承認待ちが無い会話への承認操作）は `ConversationErrorCode` の承認系コードを持つ構造化エラーとして WS / REST 共通で返す（NFR-5）。SDK 例外は `_adapters` 内で捕捉し plain 結果へ変換し、SDK 例外型を境界に露出させない。

| エラーコード | 条件 | REST status |
|---|---|---|
| `UNKNOWN_APPROVAL` | 未知の `call_id` を approve/reject | 404 |
| `APPROVAL_ALREADY_RESOLVED` | 解決済み `call_id` への再操作 | 409 |
| `NO_PENDING_APPROVAL` | 承認待ちが無い会話への承認操作 | 409 |

### NFR の充足

- **SDK 隔離（NFR-1）**: 会話のストリーミング実行（`Runner.run_streamed` および streaming イベント型）と
  SDK `Session` 生成（SQLite / compaction ラッパー）、HITL の承認待ち検知・承認/却下適用・再開・中断状態
  シリアライズに触れるのは `_adapters` のみとする。`_adapters` は SDK 型を内部に閉じ、外へは plain なデータ
  （テキスト断片・最終出力・承認待ち一覧）と不透明 `Session` / `RunState` だけを渡す。会話サービス・サーバ
  入口・CLI クライアントは承認・却下・中断状態を扱う際も `agents` を直接 import しない（grep 計測
  `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` が空を維持する範囲に含む）。
- **独自エンジンを新設しない（NFR-2）**: 会話実行は SDK の `Runner`、履歴 / 永続化 / 圧縮は SDK `Session`
  を流用し、独自の会話実行エンジン・永続化スキーマ・state 機構を新設しない（継続様式の SoT は「マルチ
  ターン継続は `RunResult` 駆動」の記述を参照）。HITL の承認待ち検知・承認/却下による再開・中断状態の保持も
  SDK の実行・状態管理機構（interruptions / `RunState` の `to_string` / `from_string` / approve / reject）の
  流用で実現し、独自の中断状態スキーマは新設しない。会話サービスが持つ会話 store はサーバの最小状態管理で
  あり、SDK 標準様式に代わる独自エンジンではない。
- **API / CLI 挙動一致（NFR-3）**: CLI はサーバ経由で会話するため、CLI と API が物理的に同一コード経路
  （同一会話サービス）に到達し、機能仕様の一致が構造的に保証される（非決定的な出力テキストの完全一致は
  対象外）。HITL の承認・却下も CLI はサーバのエンドポイント / WS メッセージへ送って同一 approve/reject
  処理点へ到達し、CLI に独立した承認実行ロジックを持たない。
- **中断状態の保持と整合（NFR-4）**: 承認は元の会話メッセージとは別メッセージ / 別リクエストで到着しうる
  ため、中断状態を当該会話に紐づけて保持し、再開時に正しく対応付ける。同一会話に対する承認・却下・継続は
  会話ごとの `lock` で直列化し、永続テーブル操作は `session_id` を条件に含めた db レベル整合で取り違えを
  防ぐ。再開（resume）が一時エラーで失敗した場合は、残りの承認待ちを中断状態（`RunState`）の実状から再算出
  して「全解決・未再開」の状態で保持・再永続し、空 decisions での再 resolve で再開をリトライできる。
- **後方互換（NFR-6）**: 承認必須ツールを 1 つも宣言しない会話は承認待ちが発生せず、承認待ちイベント /
  フィールドは一切流れない（WS は `StreamDelta` 逐次 → `StreamDone`、REST は `status="final"`）。承認待ちを
  `StreamEvent` Union に混ぜず `SendResponse` を破壊せず判別フィールド追加にとどめるため、既存の WS / REST
  消費者は型・経路の両面で変更不要であり既存挙動が不変。
- **承認前非実行（NFR-7）**: 承認必須ツールは approve による再開まで実行されず、reject / 未解決の間も実行
  されない。ライブラリは「中断検知時に最終出力を返さず全解決まで再開しない」だけを保証し、実行抑止は SDK
  機構に委ねる。
- **依存の隔離**: 実行寄り層は `runtime/` 配下に集約し extra で opt-in する。サーバ依存（`fastapi` /
  `uvicorn`）は `serve` extra、CLI クライアント依存（`httpx` / `websockets` / `rich`）は `cli` extra に
  隔離し、本体の必須依存を増やさない。`conversation` extra は追加 PyPI 依存を持たない（会話コードは同一
  wheel に常時同梱され import 自体は可能）opt-in マーカーであり、その opt-in の実体は物理 import ブロック
  ではなく「コア `__all__` から会話シンボルを除外し `runtime/conversation` 公開窓口経由に限定する」公開
  窓口の分離で表現する。未導入時に必要 extra を案内するエラーは実依存を持つ `runtime/serve` /
  `runtime/cli` の入口に適用される（`conversation` は依存ゼロのため適用が限定される）。registry・モデル・
  プロンプトは利用者提供とし、ライブラリには同梱しない（非同梱方針）。

### スコープ外

bargein / stop・音声・WAF / FIM / audit・RAG・user_preferences は対象外とする。

## LLMOps 評価（ローカル品質ゲート支援）

利用者が宣言したエージェントを観点別に判定し、CI / リリースの品質ゲートで使える統合 verdict を得るための
上位利用支援層。`runtime/` 配下の実行寄り層の一員（`runtime/llmops`）であり、採点コアは
`oai-agentspec[llmops]` extra（DeepEval）で opt-in 導入する。観測連携（Langfuse）は別 extra
`oai-agentspec[llmops-langfuse]` に分離し、Langfuse を使わない利用者は採点コア `[llmops]` のみで評価できる。
公開窓口は `oai_agentspec.runtime.llmops` の `__init__.py` に集約し、評価実行 API（`evaluate`）・結果型・
設定型・観点オブジェクト（`Criterion` と組込みファクトリ `Relevance` / `Safety` / `Conciseness` /
`Faithfulness` / `GEval` / `ToolUse` / `HandoffRoute`）・dataset 連携（`register_dataset` / `load_dataset`）は
ここから参照する。コア `__all__` には載せない。コア（registry / workflow）の挙動は
変更せず、評価は宣言物を読み取りのみで参照する。

評価対象は宣言層の `AgentSpec` 単体に加え、`HandoffGraph` / `WorkflowGraph` の横断評価を含む。横断対象は
specs を register 済みの `AgentRegistry` を伴って受領し（`evaluate(registry=...)`）、`HandoffGraph` は
`graph.apply(registry)` でエッジを反映してから `entry_agent(registry)` で entry agent を取得し、
`WorkflowGraph` は `as_agent_spec(name, registry=...)` で AgentSpec 化して実行する。registry 未供給 / 必要
specs 未登録のまま横断対象が渡された場合は specs 入手元が無い旨の明確なエラーを送出する。

llmops は会話 Helper / 既存 extra と同じ整合方針に従う。SDK / 外部クライアント隔離（`_adapters` への
import 単一窓口）・extra 未導入耐性（公開窓口分離と遅延 import 境界）・単方向依存（runtime からコアへの
上向き参照のみ）の規約は既存節（「SDK 隔離と依存性注入（DI）」「会話 Helper（ローカル開発支援）」）が SoT で
あり、本節では再掲しない。

### 評価対象

評価対象は宣言層の `AgentSpec` / `WorkflowGraph` / `HandoffGraph` を read-only で受理し、宣言物自体を
mutate しない。評価対象の識別子（spec 名 / グラフ識別子）を結果に含める。未対応の型が渡された場合は許容型を
示す明確なエラー（`TypeError`）を送出し、暗黙のフォールバックをしない。評価データセット（入力ケース群と
期待観点）は利用者が渡したものをそのまま用い、ライブラリ側にケースをハードコードしない。

### 観点別採点（DeepEval を採点器として使用）

観点別の採点は DeepEval のメトリクスを採点器として使う。DeepEval 統合窓口は `_adapters/judge.py` に閉じ
（`import deepeval` をここに局在化し、隔離方針は「SDK 隔離と依存性注入（DI）」節を参照）、評価ロジック層は
plain データのみを扱う。DeepEval の LLM 呼び出しは custom model（`DeepEvalBaseLLM` 実装）で利用者 Judge
モデルをラップし `_adapters` 経由へ一本化する。DeepEval の Confident AI テレメトリは既定でオフにする
（利用者設定で有効化可）。DeepEval の結果オブジェクトは `_adapters/judge.py` 内で plain な
`CriterionResult`（観点名・ステータス・根拠・スコア）へ変換し、評価ロジック層が DeepEval 型を見ないようにする。

観点 → メトリクスの対応は `criteria.py` が抽象識別子のみ保持し DeepEval を import しない。識別子から DeepEval
metric クラスへの解決は `_adapters/judge.py` が担う（criteria は DeepEval 非依存）。Judge へ渡す untrusted
入力（評価対象の出力等）のマーキングは framework 非依存の domain 純ヘルパ `_spotlight`（Spotlighting）が担い、
`evaluator` が judge へ渡す前に適用する。`_adapters/judge.py` はマーキング済みの入力を受領する。観点判定の
構造（観点名・マーキング規則・出力スキーマ）はライブラリが持つが、Judge への rubric 本文（G-Eval 観点文）は
同梱せず利用者が渡す経路で扱う（プロンプト非同梱方針との整合）。Judge 用モデルは評価実行時に利用者が渡す。

対応観点は出力品質・ツール使用・ルーティング・承認ゲートの系からなる。

- 出力品質: `factual_grounding`（DeepEval Faithfulness / Hallucination・context ベースの事実整合性）/
  `safety`（G-Eval safety 観点・補助に Toxicity / Bias）/ `relevance`（Answer Relevancy）/ `conciseness`
  （G-Eval）。利用者は G-Eval rubric を渡して任意観点を追加できる。単体・横断のいずれにも適用する。
- `tool_correctness`: DeepEval ToolCorrectnessMetric により、捕捉した実行トレース中のツール呼び出し列
  （`tools_called`）と入力ケースの `expected_tools`（ground truth）を recall（threshold=1.0・期待ツールが
  全て呼ばれていれば pass・余分な呼び出しや handoff の `transfer_*` は無視）で決定的に比較する。単体・横断の
  いずれにも適用し、`expected_tools` 非在のときのみ `not_applicable` に倒す（評価対象がツールを持たない場合は
  NA にせず、期待ツールが呼ばれなければ recall=0 で fail とする）。
- `handoff_correctness`: 入力ケースの `expected_route`（ground truth）と捕捉した実行経路を oai-agentspec 側の決定的
  比較で判定する。`expected_route` は起点を含むフルパス（handoff の遷移先を順に並べ末尾に最終応答 agent を
  含む列・起点が自身で応答する場合は単一要素）で指定する。`expected_route` 非在のときのみ `not_applicable` に
  倒す（横断モードかどうかでは NA にしない。観点の適用可否は利用者の criteria 選択に委ねる）。
- `approval_gate`（`ApprovalGate()`）: 承認必須ツール（`needs_approval=True`）を持つ評価対象が、危険操作を
  実行する前に正しく承認ゲートへ回したかを判定する。入力ケースの `expected_approvals`（ground truth・期待
  承認ツール名）と、中断時の承認待ち（`RunOutcome.pending` のツール名）を decision 的に recall 比較する。
  resume も approve もせず**危険ツールは一切実行しない**。`expected_approvals` 非在のときのみ `not_applicable`。
  詳細は「HITL 評価（承認ゲート / mock-approve）」節を参照。

JSON / スキーマ適合・RAG 系・Summarization は対象外とする（`factual_grounding` 用の Faithfulness /
Hallucination は context ベースの事実整合性として使う範囲に限定する）。観点ごとに pass/fail（または順序尺度
スコアと閾値による pass/fail）と判定根拠を返す。観点別ステータスは `pass` / `fail` / `inconclusive` / `skip` /
`not_applicable` を取り、これと統合 verdict（`pass` / `fail`）の双方を構造化結果として返す。

- 入力ケースに参照文脈（`reference_context`）が与えられないとき、当該ケースの `factual_grounding` を
  `not_applicable` に倒し、knockout 判定および verdict 計算の母集合から除外する。
- DeepEval 採点がスキーマに適合しない、またはタイムアウトしたときは当該観点を fail もしくは inconclusive へ
  倒し（fail-closed）、未捕捉例外でプロセスを停止しない。タイムアウト（既定値あり・利用者設定可）と逐次 /
  並列の実行制御（並列度設定可）を持つ。
- 評価対象が HITL / ツール承認で中断（最終出力が無い）したケースは、既定では採点せず当該ケースの全観点を
  `inconclusive` に倒す（出力非依存の観点が中断時点までの経路一致で誤って pass するのを防ぐ）。inconclusive は
  verdict 計算で inconclusive ポリシー（既定 `fail`）により解決される。例外として `ApprovalGate` 観点は中断時
  でも `expected_approvals` と承認待ちを比較して採点する。`evaluate(approvals=, tool_mocks=)` を渡すと承認を
  自動解決して完了まで採点する（「HITL 評価」節）。承認・mock を渡さなければ本物の危険ツールは実行されない。

### 実行トレースの捕捉（横断 routing / ツール使用）

横断 routing もツール使用も、評価対象を実際に実行して得る生 `RunResult` から捕捉する。捕捉は `_adapters` 内で
完結し、`DefaultRunnerAdapter.run_with_observation` が生 `RunResult` を 1 回実行して `_adapters/routing.py`
が `new_items` を 1 パス走査し、plain な実行経路（`ObservedRoute` / `RouteStep`）とツール呼び出し列
（`ObservedToolCall`）を抽出する。生 `RunResult` は `_adapters` 外（評価ロジック層）へ出さない（SDK 型遮断・
「SDK 隔離と依存性注入（DI）」節を参照）。抽出した plain な `ObservedRun` を入力ケースの期待値
（`expected_route` / `expected_tools`）と決定的に比較し、`handoff_correctness` / `tool_correctness` の観点
結果を導く。単体・横断のいずれも同じ `run_with_observation` 経路で捕捉する。

### 統合 verdict 計算

観点別の結果から CI / リリースゲートで合否を一意に分岐させる統合 verdict（`pass` / `fail`）を導出する。計算は
副作用を持たない純粋関数に集約する。

- **母集合**: `skip` / `not_applicable` を母集合から除外する（母集合が空なら fail-closed）。
- **knockout 観点（ファクトリ既定 `safety` / `factual_grounding` が `knockout=True`）**: 当該観点が `fail` なら
  verdict を `fail` に確定し、他観点の結果で上書きしない（fail-closed）。ただし当該観点が `not_applicable` の
  ときは knockout 判定の対象外とする。
- **必須観点欠落**: 必須観点（既定で母集合に出現した観点集合）が母集合に存在しないときは欠落を検出して
  verdict を `fail` に倒す（missing-pair fail-closed）。
- **実 fail 優先**: 母集合に `fail` が 1 件でもあれば `fail`。`inconclusive_policy=PASS` でも実在する fail を
  inconclusive で隠さない（inconclusive 解決より先に評価する）。
- **inconclusive**: 残り（`pass` / `inconclusive`）に inconclusive があれば inconclusive ポリシー（既定 `fail`）
  に従って解決する。
- 上記の fail-closed 条件に該当せず母集合が全 pass のとき verdict を `pass` とする。

knockout は各 `Criterion` の `knockout` フラグで指定する（ファクトリ既定: `safety` / `factual_grounding` が
`knockout=True`）。inconclusive ポリシー・必須観点は評価設定（`EvaluationConfig`）で上書きできる。

### Langfuse 連携と graceful degradation

評価のスコア（観点別スコアと統合 verdict）とトレース（評価対象の入出力と判定）を Langfuse へ送信する。
Langfuse 連携は `_adapters/langfuse.py` を経由し、`langfuse` の import は関数内遅延 import に閉じる。Langfuse
連携全体が任意であり、`langfuse` は別 extra（`llmops-langfuse`）に分離されているため、観測を使わない利用者は
採点コア `[llmops]` のみで評価できる。送信はベストエフォートであり、外部依存不在時もローカル評価結果を返す
graceful degradation を保つ。Langfuse のうち使う機能は Tracing / Scores（常時）と、opt-in の Datasets 登録
および push 専用 Prompt Management に限定し、managed evaluator（サーバ側 LLM-as-judge）と Sessions・Users は
使わない。

- Tracing / Scores（常時）: 評価対象の入出力と判定をトレースとして、観点別スコアと統合 verdict（`verdict`）を
  NUMERIC Scores として送信する（verdict も Score 化することで Dataset Run 比較が pass/fail ゲートを集約できる）。
  trace metadata には verdict に加え観測した処理経路（`route`・起点込みフルパス）とツール呼び出し
  （`tools_called`）を載せ、`HandoffRoute` / `ToolUse` を criteria に含めなくても経路・ツールを trace で
  確認できる。
- Datasets（register → fetch → use・opt-in）: dataset への item 登録は `register_dataset`（一度きり・冪等
  upsert・安定キー `EvalCase.id`）が担い、`load_dataset` が fetch して `EvalCase` 列へ復元する（Langfuse が
  source）。`EvalCase.expected_output` は item の `expected_output` に、oai-agentspec 固有の `reference_context` /
  `expected_route` / `expected_tools` は item.metadata に写す。`evaluate` は `LangfuseConfig.dataset_name`
  設定時に **既存 dataset item へ run を link するだけ**（item の upsert / dataset 作成はしない）で、各ケースの
  trace / Scores を dataset item × run にリンクする。複数回の評価を同一データセット上で A/B・回帰比較するための
  連携で、`dataset_name` 未設定なら dataset 連携をスキップする（best-effort）。
- push 専用 Prompt Management（opt-in）: `LangfuseConfig.prompt_name` 設定時かつ評価対象プロンプトを抽出可能な
  ときのみ、評価対象プロンプトを Langfuse Prompt Management に register / upsert（push のみ）し、評価 trace を
  当該 prompt version にリンクする。プロンプト version ごとに judge 結果を集約するための観測記録であり、
  Langfuse からプロンプトを取得 / 配信しない（取得系 API は呼ばない）。PromptStore（利用者 root）が
  プロンプトの Single Source of Truth のままであり、プロンプトバージョニング管理機能（snapshot / list / diff /
  rollback）は作らない。`AgentSpec` の静的 `instructions` のときのみ抽出し、動的プロンプト（callable /
  PromptStore 合成）や横断（単一プロンプトが特定できない）は登録をスキップする（best-effort）。
- Langfuse 設定が渡されないときは送信をスキップし、ローカル評価結果（観点別結果と verdict）を返す。Langfuse
  設定は渡されたが送信（trace / Scores / dataset / prompt のいずれか）に失敗したときはローカル評価結果を優先して
  返し、送信失敗で評価全体を fail させない（warning ログ）。Langfuse 認証情報・`dataset_name` / `run_name` /
  `prompt_name` / `prompt_label` は評価実行の引数（`LangfuseConfig`）として利用者から受領し、コア・既存
  `_adapters` 契約に env 依存を波及させない。env 参照が必要な場合も `runtime/llmops` の評価境界に閉じる。

### HITL 評価（承認ゲート / mock-approve）

承認必須ツール（`needs_approval=True`）を持つ評価対象を、本物の危険な副作用を起こさずに評価する。2 系統を提供する。

- **承認ゲート評価（`ApprovalGate()`）**: 危険操作を実行前に正しく承認ゲートへ回したかを判定する（観点
  `approval_gate`・上記「観点別採点」）。resume も approve もせず、`expected_approvals` と中断時の承認待ちを
  決定的に recall 比較する。危険ツールは実行されない。
- **mock-approve（`evaluate(approvals=resolver, tool_mocks=...)`）**: 承認を自動解決して実行を完了させ、承認後
  の応答・経路・ツール使用を採点する。`tool_mocks` は **agent スコープのネスト dict**
  （`{agent_name: {tool_name: 値 | callable}}`）で、評価対象ツールの**実行本体（`on_invoke_tool`）だけ**を副作用
  のない mock へ差し替える（`name` / `description` / `params_json_schema` / `needs_approval` は不変＝ゲートは
  発火し HITL 経路を通る）。`approvals` resolver は承認待ち（`{tool_name, call_id, agent_name}`）を受けて
  approve / reject を返す。reject はツールを実行せず、拒否後の応答を採点できる。

mock の流入は**宣言層**で行う（ビルド済み実行グラフを走査しない）。`_target.normalize` が `_adapters` の
`mock_spec_tools` で spec の tools を mock 差し替えし、横断（`HandoffGraph` / `WorkflowGraph`）では
`AgentRegistry.clone(transform_spec=...)` で派生 registry を作って build する。動的 handoff の候補もクローン
registry 経由で解決されるため mock される。**利用者の registry / グラフは一切汚さない**（`HandoffGraph` は
`copy.deepcopy` してから apply・registry は spec まで独立コピー）。承認の自動解決は `_adapters` の resume ループ
（`apply_approvals` → observe 付き resume・segment マージ・上限ラウンドあり）で行い、SDK 型は `_adapters` に
閉じる。

**安全不変条件（fail-closed）**: approve を認可するのは、その承認待ちの `(agent_name, tool_name)` が**実際に
mock 差し替えされた**ものに限る。未差し替え（mock 未登録 / 到達不能 / 別 agent の同名ツール）や agent 不明の
approve は `ValueError` を送出し、本物の危険ツールが評価中に実行されるのを構造的に阻止する。

中断時の扱い: `ApprovalGate` は中断時も採点する。承認自動解決後も中断が残ったケースは発火済み承認を保持しつつ
全観点を `inconclusive` に倒す。Langfuse 連携時は trace metadata に承認待ち（`pending_approvals`）と中断有無
（`interrupted`）を反映する。

評価実行は独自実行エンジンを持たず、`_adapters` 経由で SDK `Runner.run` へ結線する（build-don't-run の維持）。
コア宣言層には評価実行 API を追加せず、公開の評価実行 API は `runtime/llmops` 公開窓口に集約する。

## Agent Lightning 最適化（プロンプト最適化支援）

利用者が宣言したエージェントのプロンプトを最適化（プロンプト改善）するための上位利用支援層。`runtime/` 配下の
実行寄り層の一員（`runtime/lightning`）であり、conversation / serve / cli / llmops と同型の責務分割・公開窓口・
extra 未導入契約・SDK 隔離方針に従う。プロンプト最適化（APO）を `oai-agentspec[lightning]` extra
（`agentlightning` の APO 機能）で opt-in 導入する。`agentlightning` の import は `_adapters/lightning`
に局在化する（SDK / 外部クライアント隔離の規約は「SDK 隔離と依存性注入（DI）」節が SoT・本節では再掲しない）。

公開窓口は `oai_agentspec.runtime.lightning` の `__init__.py` に集約し、最適化エントリ（`optimize`）・結果型
（`OptimizeResult`・`save(path)` / `to_dict()`）・設定型・reward ファクトリ（`contains` / `exact` / `tool_match` /
`route_match` / `last_agent_match` / `approval_match` / `judge`）・`prompt_slot` / `prompt_slots` /
`train_val_split` はここから参照する。コア `__all__` には載せない
（extra 未導入耐性・単方向依存の規約は既存節「会話 Helper（ローカル開発支援）」「LLMOps 評価」の方針に従い再掲
しない）。コア（registry / prompts / workflow）の挙動は変更せず、宣言物・`PromptStore` は読み取り / 複製経由で
参照する（依存方向は `runtime/lightning` からコア / `_adapters` への一方向）。

### 最適化対象とプロンプト最適化（APO）

最適化対象は宣言層の `AgentSpec` 単体に加え、`HandoffGraph` / `WorkflowGraph` を対象とするハンドオフを通る系全体
の end-to-end 最適化を含む。系全体対象は specs を register 済みの `AgentRegistry` を伴って受領し、対象は利用者が
明示列挙したエージェントのみとする（暗黙に全 agent を対象としない）。単一エージェント / 単一スロットは系全体最適化
の簡単版として併存する。`optimize` の `algorithm="apo"` で実行する。データ入口は `train`（必須）/ `val`（任意）に
統一し、暗黙の分割パラメータを持たない（分割は `train_val_split` または利用者自前）。

- **APO（`algorithm="apo"`・`[lightning]` のみで完結）**: 最適化対象は利用者指定のプロンプトスロット（vars 未展開
  のテンプレート文言・`${var}` プレースホルダ保持）+ rebind モデル。vars 値は最適化対象外（不変・確定）で、各
  rollout 時に再注入し、候補生成に含めない。候補が必要な `${var}` を失えば無効化 / 低評価に倒す（fail-closed）。
  単一スロットに加え `{名前: slot}` の mapping で系全体のプロンプトを同時最適化する。`prompt_slot` / `prompt_slots`
  は `PromptStore` の公開 `compose` / `get` を読み取り seed と固定部分を再合成し、`build`（候補テキスト →
  `AgentSpec`）を内包するため rebind を自動導出する（利用者は手書き rebind 不要・生 seed のパワーユーザー経路でのみ
  明示）。`build` 省略時の既定 build は registry 登録 `AgentSpec` を複製し `instructions` のみ候補で差し替える
  （tools / handoffs / model は複製で保持・registry 未解決かつ build 省略は fail-closed）。出力は `${var}` 保持の
  最適化済みテキスト（複数スロット時は名前付き mapping）。`PromptStore` は読み取りのみで内省・書き換えしない。

### データフロー

最適化ループ本体（系全体の複数プロンプト最適化を含む）は `_adapters/lightning` 経由で
agent-lightning の Trainer へ委譲する（build-don't-run の維持・コア宣言層に最適化 / 実行 API を追加しない）。lib は
rollout の結線と reward 算出への plain データ供給に徹する。

- target 正規化: `_target.normalize` が対象を実行可能 Agent + 実差し替え集合へ正規化する（系全体は
  `HandoffGraph` を `copy.deepcopy` してから `apply(registry)`・`WorkflowGraph` は registry を伴って Agent 化・
  利用者状態を汚さない）。
- 候補生成: APO の候補プロンプト生成は Trainer に委譲し、lib は beam-search / テキスト勾配 /
  最適化アルゴリズムを実装しない。
- rollout: `_adapters` の `DefaultRunnerAdapter.run_with_observation` で 1 回実行し、`observe_run_result` で plain な
  実行経路 / ツール呼び出し列へ変換して利用者供給の reward へ渡す（生 `RunResult` は `_adapters` 外へ出さない・
  実行トレース捕捉の流儀は「LLMOps 評価」節の実行トレース捕捉を再利用する）。
- APO 候補適用: 各 rollout で `registry.clone(transform_spec=候補で instructions 差し替え)` により候補適用済みの
  独立 registry を構築し、vars を rollout 直前に再注入する（利用者 registry / 登録 spec は不変）。
- 結果: 既定で plain な `OptimizeResult` を返すのみ（lib 自動書込なし・`PromptStore` 非書込）。`result.save(path)`
  は利用者指定パスへの opt-in 書込で、APO は `${var}` 保持テキストを書く。`OptimizeResult` は最適化済み
  `prompt`（after・rollout 時の合成済み full テキスト・`Slot.fixed` と tune を `\n\n` 連結し vars 再注入済み）に
  加え、`seed`（before・同じ shape の合成済み full）と `diff`（before / after の unified diff）を併せて返す
  （いずれも単一スロットは str・複数スロットは `{名前: str}` mapping）。利用者は `print(result.diff)` で 1 行
  記述で「どこが変わったか」を可視化でき、APO 結果の before/after 比較ボイラープレートが不要になる。
- 履歴: `OptimizeResult.history` は各スロット 1 件の `HistoryEntry`（TypedDict）の列で、`slot` / `best_score` /
  `best_version` / `placeholder_fallback` の 4 キーを持つ。APO 最良候補が seed の `${var}` プレースホルダを
  喪失した場合は seed にフォールバックして `placeholder_fallback=True` をマークし、`best_score` / `best_version`
  は破棄候補の値を指さないよう None に上書きする（公開契約「最適化済みテキストは `${var}` を保持する」を
  fail-closed で守り、利用者は warning 受信に依存せず history flag で programmatic に検出できる）。

### rollout 安全性

同一 rollout を多数回実行する最適化での危険ツール副作用の反復に対し、利用者は任意で `tool_mocks` / `approvals`
（mock-approve 相当）を rollout 実行へ適用できる。これは独自の安全機構を新設せず「LLMOps 評価」節の HITL 評価
（`tool_mocks` / `approvals`・`mock_spec_tools` による `on_invoke_tool` 差し替えと安全不変条件）を再利用する。
`tool_mocks` / `approvals` 未指定時は rollout を宣言どおり実行する。

Agent Lightning を LLMOps トラックへ振り分けた検討経緯は `docs/rationale/agt-governance-integration.md` を参照。

## 内容ガードレール（ローカル品質ゲート支援）

宣言したエージェントが「何を言うか」を入出力・中間ツール段で検査する上位利用支援層。`runtime/` 配下の
実行寄り層の一員（`runtime/guardrails`）であり、依存ゼロ opt-in extra `oai-agentspec[guardrails]` で
オプトイン境界を表現する（PyPI 依存を増やさず、公開窓口分離と意味を揃える）。重い専門検知（PII / モデレーション /
注入検知サービス）はライブラリに同梱せず利用者が外部 DI で渡す。

「何を言うか」を検査する内容ガードレールと、「何をできるか」を許可 / 拒否する AGT ガバナンス（ツール単位
ポリシー強制・別 Issue の AGT 統合）は直交する役割分担であり、相互に置き換えない。ガードレールの宣言経路は
新設しない。`input_guardrails` / `output_guardrails` は `agents.Agent` の正規フィールドであり、本ライブラリ
では既存の `AgentSpec.extra` 素通し（「パラメータのカスタマイズ」節）でそのまま宣言できる。本支援層はこの
既存経路へ渡せる SDK 互換 guardrail を生成するファクトリ群を提供するもので、`AgentSpec` にフィールドを
足さない。

### 検知 3 家族

検知器を性質で 3 家族に分け、いずれもファクトリへの DI で受ける。

- **外部検知器（A）**: Presidio（PII）・モデレーション / 注入検知サービス等。利用者の検知 callable を薄く
  包む接着のみを提供し、検知本体はライブラリ非同梱（外部 DI）。
- **prompt 駆動 LLM（B）**: 判定用 model と判定 prompt を DI で受け、LLM-as-judge で内容を判定する。判定
  model の呼び出しは `_adapters` 経由に寄せ、判定 prompt 本文・model はライブラリに同梱しない（プロンプト /
  モデル非同梱方針）。
- **決定的・ロジック系（C）**: カナリア（出力への漏洩トークン検知）・正規表現・長さ / サイズ閾値・allow / deny
  リスト・汎用 predicate（`Callable[[str], bool]`）・最低限の注入ベースライン（SQLi / コマンド注入 /
  パストラバーサルの代表パターン）。よく使う再利用 helper を同梱しつつ、すべて DI で上書き / 拡張できる。注入
  ベースラインは非網羅であり本丸はパラメータ化クエリ / 安全 API 利用であることを前提とした補助検知に留める
  （既定パターンは DI で上書き可）。

### 適用境界

- **agent 境界（会話入出力）**: ファクトリは SDK 互換 `InputGuardrail` / `OutputGuardrail` を返し、利用者はそれを
  `AgentSpec` の `input_guardrails` / `output_guardrails` フィールド（`agents.Agent` を鏡写し）へ渡す。`build_agent`
  がそれを `agents.Agent` へ転送し、評価は SDK `Runner` が会話入出力で行う。入力ガードレールは
  `run_in_parallel`（既定 True・SDK 既定）で並行実行（レイテンシ優先）か直列実行かを選べ、`on="input"` のときのみ効く
  （`OutputGuardrail` に該当フィールドはない）。既定 True は判定がエージェントのターンと並行に走るため、検査が trip する
  前にモデルがツールを呼びうる。`run_in_parallel=False` を指定すると検査完了を待ってからターンを開始し危険入力を実行前に
  ブロックできる。ツール実行の副作用はツール境界ガードレールが実行前にゲートする役割分担を前提とする。
- **ツール境界（中間ツール出力 / 引数）**: SDK ネイティブのツールガードレール（`ToolInputGuardrail` /
  `ToolOutputGuardrail`）を生成し、`FunctionTool` の `tool_input_guardrails` / `tool_output_guardrails` へ
  装着する。ツール引数（`ToolInputGuardrailData`）と中間ツール出力（`ToolOutputGuardrailData.output`）を
  検知器で検査し、ツールの実行本体・宣言メタ（`name` / `description` / `params_json_schema` /
  `needs_approval`）は変えない。ここで行うのは **内容検査のみ**であり、ポリシー強制（実行可否の allow / deny
  制御）は新設しない（それは AGT ガバナンスの責務）。trip 時の挙動は `ToolGuardrailFunctionOutput` が提供する
  `reject_content`（出力を差し替えて続行）/ `raise_exception`（実行を中断）/ `allow`（通過）から選べ、既定は
  `reject_content`（注釈付き返却）とする。ツールガードレールは検知器から
  `tool_guardrail(detector, on="input" | "output")` で生成し、`function_tool(tool_input_guardrails=[...],
  tool_output_guardrails=[...])` でツール定義時に宣言する（SDK ネイティブの流儀）。`function_tool` で定義できない
  既存ツール（`as_tool` / ワークフローツール / サードパーティ）へは `guard_tool(tool, input_detector=...,
  output_detector=...)` で後付け装着する。

### 設計方針（batteries-included かつ swappable）

helper はファクトリに徹し guardrail オブジェクトを返す。利用者はそれを `AgentSpec` の `input_guardrails` /
`output_guardrails` フィールド（`agents.Agent` と同型）と `tools`（`guard_tool` のラップ済みツール）へ宣言する。
よく使う再利用 helper（決定的検知・注入ベースライン等）を同梱しつつ、判定 model / prompt・カナリア値・
predicate・検知パターン・外部検知器はすべて利用者 DI で受け、上書き / 拡張できる。重い専門検知は外部 DI で
ライブラリ非同梱とする。

### 配置と隔離

公開窓口は `oai_agentspec.runtime.guardrails`（llmops / conversation と同型・コア `__all__` には載せない・
helper は実行寄りであり宣言層シンボルのみのコア `__all__` 原則に従う）。SDK 型（agent / tool 双方の guardrail
型・デコレータ・`ToolGuardrailFunctionOutput`）の import は `_adapters/guardrails.py` に閉じ、SDK 隔離 grep を
空に保つ。決定的検知の plain ロジックは agents 非依存層に分離し、SDK なしで単体検証できる。SDK 隔離・単方向
依存・extra 未導入耐性の規約は既存節（「SDK 隔離と依存性注入（DI）」「会話 Helper（ローカル開発支援）」）が
SoT であり、本節では再掲しない。

### 公開 API の使い勝手（DX）

利用者向けファクトリは次の方針に従う。

- **戻り値は SDK 互換の具体型**（`InputGuardrail` / `OutputGuardrail` / `FunctionTool`）で、IDE 補完・型読解が
  効く。型注釈は `_adapters/guardrails.py` 経由の `TYPE_CHECKING` import で与え（`from agents` を直接書かず
  SDK 隔離 grep を空に保つ）、入出力どちらも取りうる二境界ファクトリは `InputGuardrail | OutputGuardrail` を返す。
- **適用境界 `on` はキーワード必須**（既定値を置かない）。入出力どちらにも適用しうるファクトリ（regex /
  predicate / length / allow_deny / external_detector / prompt_llm）で `on` を毎回明示させ、既定の暗記負荷と
  取り違えをなくす。入力専用（注入ベースライン）/ 出力専用（カナリア）は `on` を取らない。
- **宣言面は `agents.Agent` を鏡写し**: guardrail は `AgentSpec.input_guardrails` / `AgentSpec.output_guardrails`
  フィールドへ渡す（型は `Any`・SDK 隔離のため `agents` を import しない）。`build_agent` がそれを
  `agents.Agent(input_guardrails=..., output_guardrails=...)` へ転送する。`extra` に同名キーを入れると専用フィールドと
  衝突する `ValueError`（宣言経路をフィールドへ一本化）。SDK ネイティブの `Agent(...)` と同じ流儀で書ける。

### OWASP LLM 対応表（運用上の現在仕様）

主軸とする OWASP LLM Top 10 項目への対応を、適用種別と主な検知家族で示す。

| OWASP LLM 項目 | 適用種別 | 主な検知家族 |
|---|---|---|
| LLM01 Prompt Injection | input | prompt 駆動 LLM（B）+ 外部検知器（A） |
| LLM07 System Prompt Leakage | output | カナリア（C）主 + prompt 駆動 LLM（B）の二層 |
| LLM02 Sensitive Information Disclosure | output | 外部検知器（A・Presidio）+ カナリア（C） |

### framework 非依存

同 helper 群は OWASP に限定されず、MITRE ATLAS / NIST AI RMF の関連項目や品質・ブランド系の内容検査にも適用
できる（検知家族は framework 中立）。framework 横断のカバレッジマトリクスと検知家族への振り分け根拠・選定
理由・トレードオフは `docs/rationale/content-guardrails-coverage.md` を参照する。カバレッジ項目の進捗追跡は
Issue 側で管理し、本 docs にチェックボックスや進捗表は置かない。

整合性 / 供給網インテグリティ（改竄検知・供給網の信頼）は内容検査では守れない直交領域であり別途扱う。

## AGT ガバナンス（ツール単位ポリシー強制と監査）

宣言したエージェントが「何をできるか」をツール単位のポリシーで許可 / 拒否し、許可 / 拒否を監査ログへ記録
する上位利用支援層。`runtime/` 配下の実行寄り層の一員（`runtime/governance`）であり、conversation /
serve / cli / llmops / lightning / guardrails と同型の責務分割・公開窓口・extra 未導入契約・SDK 隔離方針に
従う。`oai-agentspec[governance]` extra で opt-in 導入し、extra は MIT の `agent-governance-toolkit`（AGT）
に依存する。`agents` / AGT の import は `_adapters/governance.py` に局在化する（SDK / 外部クライアント隔離の
規約は「SDK 隔離と依存性注入（DI）」節が SoT・本節では再掲しない）。

他の runtime extra が別エントリ（評価 `evaluate` / 最適化 `optimize`）であるのに対し、governance は build 経路
（`AgentBuilder`）に差し込む**装飾 builder**として働く。`AgentRegistry(agent_builder=GovernedAgentBuilder(policy=...))`
で注入し、registry の遅延構築（`_builder().build`）を通るため、循環ハンドオフ解決後の到達可能 spec の
`FunctionTool` は govern 済みになる。`AgentSpec` / `tools` / コア `__all__` / `AgentBuilder` Protocol は変更せず、
利用者の追加記述は「builder を 1 つ差し替える」+ ポリシー指定のみ。

既知の境界（govern 対象外）: `sub_agents` の as_tool は registry が build 後に注入するため per-call の
allow / deny 評価・決定記録の対象外（監査フックの tool_start / tool_end 記録のみ。サブエージェント自身の
内部 `FunctionTool` は同 builder 経由で govern 済み）。`register_factory` 経路は builder を通らないため
govern 対象外。

### GovernedAgentBuilder（装飾 builder）

`GovernedAgentBuilder` は `AgentBuilder` Protocol（`build(spec) -> Agent`）を満たす別実装で、`inner`（既定は
`_adapters` のデフォルト builder）を装飾する。build 時に各ツールを govern ラップし、監査用 `AgentHooks` を装着
した新 `AgentSpec` を `inner.build` へ渡す。公開窓口は `oai_agentspec.runtime.governance` の `__init__.py` に
集約し、コア `__all__` には載せない。

```python
class GovernedAgentBuilder:
    def __init__(
        self,
        *,
        policy: str | os.PathLike[str] | object,  # 既定ポリシー（YAML パス、または AGT ポリシーオブジェクト）
        audit_sink: object | None = None,          # 監査ログ出力先。None は既定 sink を builder 内で共有生成
        inner: AgentBuilder | None = None,         # 装飾対象。None は既定 builder
        overrides: Mapping[str, str | os.PathLike[str] | object] | None = None,  # per-agent 上書き
    ) -> None: ...

    @classmethod
    def from_yaml(cls, path, *, audit_sink=None, inner=None) -> GovernedAgentBuilder: ...

    @property
    def audit_sink(self) -> object | None: ...          # 共有監査 sink（初回 build 前は指定値 or None）

    @property
    def unapplied_overrides(self) -> frozenset[str]: ...  # 未適用の overrides キー（typo 検知）

    def build(self, spec: AgentSpec) -> Agent: ...
```

- `policy` / `audit_sink` は `runtime/governance` では**不透明値として保持**し、評価・読込は
  `_adapters/governance` へ委譲する（`runtime/governance` は `agents` / AGT 型に触れず plain 値・不透明型のみ
  扱う）。`Agent` は型注釈上の不透明型（遅延参照）であり、`runtime/governance` は `agents` を実 import しない。
- `inner` 引数で装飾対象を差し替えられ、テスト時に fake builder を注入できる。
- **per-agent ポリシー（`overrides`）**: エージェント名 -> ポリシーの上書き。`build(spec)` 時に `spec.name`
  との完全一致（正規化なし）で引き当て、未掲載は `policy`（既定）へフォールバックする。値は `policy` と
  同形式で同一の fail-fast 検証を受ける（`None` は不正値）。一度も適用されなかったキーは
  `unapplied_overrides` で確認できる（適用済み記録は build 成功後・builder 単位）。
- **bundle YAML（`from_yaml`）**: `default`（必須）+ `agents`（任意・エージェント名 -> ポリシーフィールド）を
  1 ファイルに宣言する構築糖衣。呼び出し時に即時検証され、通常コンストラクタで同内容を組んだ場合と等価に
  動く。
- **YAML ポリシーの解決スナップショット**: YAML パスのポリシーは初回使用時に 1 度だけ読み込み・検証され、
  以降の build はスナップショットを共有する（build ごとの再読込によるエージェント間不整合を持たない）。
- **拒否例外の再エクスポート**: `PolicyViolationError`（AGT 由来・isinstance 互換）は公開窓口から遅延再
  エクスポートされる（PEP 562）。窓口 import 自体は extra 未導入でも壊れず、属性アクセス時に install hint
  付き `ImportError` となる（`hasattr` / `import *` にも同例外が伝播する fail-fast 仕様）。
- **状態のスコープ**: 既定監査 sink・解決済みポリシー・override 適用記録は builder インスタンス単位。
  `AgentRegistry.clone()` は builder を共有するため、系（本番 / 評価等）を分けたい場合は builder を分けて
  注入する。

### build 時結線

`GovernedAgentBuilder.build` は `_adapters/governance` へ govern ラップとフック合成を委譲し、得た新 spec を
`inner.build` で `Agent` 化する（**実行は SDK Runner に委ねる build-don't-run**。ポリシー評価・監査記録は
実行時に AGT 側で動き、lib は実行エンジンを持たない）。

- **全 `FunctionTool` を govern ラップ**: spec.tools の各 `FunctionTool` の `on_invoke_tool` を govern ラップ版へ
  差し替える（`dataclasses.replace` による非破壊置換・元実装の第 1 引数注釈は SDK のコンテキスト型選択の
  ため引き継ぐ）。実行時はツール呼び出し直前にポリシーを評価し、許可なら実関数を実行、違反なら実関数を
  実行せず拒否する。拒否例外は SDK `Runner` 経由では SDK 例外にラップされ得るため、利用者は
  `PolicyViolationError` を直接または `exc.__cause__` で捕捉する。引数照合は生ワイヤ JSON・JSON
  正規化文字列・デコード済み文字列スカラの 3 系統へ行い、
  エスケープ別表現によるパターン回避を塞ぐ（パース不能入力は生文字列照合へフォールバックし評価自体は
  失敗させない）。
- **非 `FunctionTool` は素通し**: hosted tool 等の関数ツール以外は走査時にそのまま通す（ポリシー強制境界は
  関数ツールの呼び出し）。SDK の HITL 承認（`needs_approval`）はツール実行前の承認フローとして govern
  ラップより先に走るため、ポリシーが拒否する呼び出しでも承認要求は先に発生し得る（承認後に deny・承認メタ
  は不変に維持）。
- **監査 `AgentHooks` を装着・`spec.hooks` は上書きでなく合成**: 監査フックを生成し、`spec.hooks` があれば
  各ライフサイクルメソッドで「監査記録 → 既存 `spec.hooks` の同名メソッドへ委譲」の順に呼ぶ合成 `AgentHooks`
  を作る（`spec.hooks is None` のときは監査フック単体）。これにより利用者のフックを失わずに監査を上乗せ
  する。合成は SDK 型を知る `_adapters/governance` に閉じる。
- `spec.handoffs` は変更しない（tools / hooks のみ非破壊置換）。`inner` へ委譲することで `AgentBuilder` の
  「handoffs 空で構築」契約をそのまま継承する。
- 監査ログの出力先は既定で AGT に委ね、`audit_sink` 引数で差し替えられる（env 参照は持たず引数 DI）。
  既定 sink は builder 内で生成・build 間で共有され、`audit_sink` プロパティで取得・検証できる。
- **ポリシー YAML の fail-fast 検証**: 空 / 非マッピング / 非文字列キー（YAML 1.1 暗黙型付けの
  `on:` 等）/ 未知キー / 強制対象フィールド（`allowed_tools` / `blocked_patterns`）の不正な値形状 /
  compile 不能な正規表現は、読み込み時に `ValueError` で拒否する。本統合で強制されないフィールドの
  指定は `RuntimeWarning` で警告する（強制対象は `allowed_tools` と `blocked_patterns` のみ）。

### 配置と隔離

公開窓口は `oai_agentspec.runtime.governance`（他 runtime extra と同型・コア `__all__` には載せない）。
`agents` / AGT の import は `_adapters/governance.py` に閉じ、SDK 隔離 grep を空に保つ。`runtime/governance` は
不透明値のみ扱い、`import oai_agentspec` は governance extra 未導入でも壊れない（AGT の import は関数内遅延で、
未導入時は install hint 付き `ImportError`）。SDK 隔離・単方向依存（`runtime/governance` からコア / `_adapters`
への一方向）・extra 未導入耐性の規約は既存節（「SDK 隔離と依存性注入（DI）」「会話 Helper（ローカル開発支援）」）が
SoT であり、本節では再掲しない。

詳細な検討経緯は `docs/rationale/agt-governance-integration.md` を参照する。

### integrity / 内容ガードレールとの住み分け

runtime インテグリティ防御（`lockdown` の `checks`・AGT を起動時 / ヘルスチェック時の一括ゲートとして発火）と
governance（AGT を各ツール呼び出しごとの実行時強制 + ライフサイクル監査として発火）は、ライフサイクル（起動時 vs
実行時）と粒度（一括ゲート vs tool 単位）が異なる補完層である。詳細は `docs/integrity.md` を参照する。
governance（「何をできるか」）は内容ガードレール（`guardrails` extra・「何を言うか」）とも直交する。

## 意図予測（`runtime/intent`）

LLM を用いた意図予測（分類）の汎用土台を `runtime/intent` に置く。分類器は `AgentSpec` / `Runner` に強制
結線されない独立サービスで、利用側が任意のタイミングで `IntentClassifier.classify(query)` を呼び、返却された
意図候補を後段のルーティング・分岐・UI・監視等で自由に扱う。実行分岐（PolicyEngine 相当）は本ライブラリの
スコープ外であり、返却フォーマットと分類 taxonomy の契約のみを本層が担う。`oai-agentspec[intent]` extra で
opt-in 導入し、extra は `pydantic>=2` に依存する（`openai-agents` の推移的依存で既に導入されるため実質増加
ゼロ）。SDK 隔離規約は「SDK 隔離と依存性注入（DI）」節が SoT・本節では再掲しない。

### 2 段構成と Protocol DI

分類器は 2 段の内部構造で、各段が Protocol で差し替え可能。

- `ContextBuilder.build(query) -> IntentContext`: `IntentQuery`（`utterance` / `history` / `run_context`。
  `utterance` は既定 `""` で省略可＝履歴のみで分類するモード。utterance と history のどちらか一方は必要）
  を受け、`query.history.get_items(limit=history_limit)` を duck-typed（SDK `Session` 互換）で呼び、戻り値を
  そのまま `tuple(...)` 化して `IntentContext.history_items: tuple[Mapping[str, Any], ...]` に pass-through
  する。lib 側では item の意味的検証（role の allowlist 等）を行わない。pydantic 側の型検証で各 item は
  `Mapping[str, Any]` として validate され、内部的に plain `dict` へ shallow copy される（同一性は保存
  されない）。SDK 由来の `TResponseInputItem`（TypedDict）はこの制約を満たすため今日の SDK では問題なく
  流通する。上位層は SDK `Session` 型に触れずに履歴を扱える。
- `CandidateGenerator.generate(context) -> IntentPrediction`: LLM を呼んで意図候補と（任意で）整合性
  レポートを返す責務。

`IntentClassifier.classify(query) -> IntentPrediction` は 2 段を束ねる上位 Protocol。既定実装は
`DefaultIntentClassifier`（`DefaultContextBuilder` + `LLMCandidateGenerator`）。3 Protocol は
`@runtime_checkable` で async 統一（LLM 実装が本質的 I/O バウンドのため）。

### `IntentPolicy` 契約（必須）

`IntentPolicy` は分類器が返せる意図集合と返却フォーマットの契約を型付きで表現する frozen BaseModel で、
`LLMCandidateGenerator` / `intent_classifier_from_model` の必須引数。

| フィールド | 型 | 役割 |
|---|---|---|
| `categories` | `tuple[IntentCategory, ...]` | 許容意図カテゴリ（非空・name 一意）。`IntentCandidate.text` はこの `name` 集合のいずれかのみ許容 |
| `max_candidates` | `int` | 返却候補件数の上限（既定 3・`ge=1`） |
| `extra_instructions` | `str` | 利用側がプロンプト先頭に注入する任意の追加指示（既定 `""`・空文字時は非出力・見出しなし） |
| `include_rationale_in_prompt` | `bool` | 出力例 JSON に `rationale` フィールドを載せて LLM に生成を促すか（既定 `False`＝rationale を促さず速度優先）。どちらでも parser は rationale を optional として受け入れる |

`IntentPolicy.render_prompt()` は `extra_instructions`（非空時のみ先頭に挿入・空白のみは非出力）+ 固定の
タスク指示 1 行（「ユーザー発話を以下のカテゴリに分類し、JSON のみを出力してください。」）+ 手書き
4 セクション（`# カテゴリ` / `# 信頼度 (level)` / `# 出力形式` / `# 制約`）を Markdown 見出しで区切って
組み立てる固定文字列を返す。タスク指示行と `# 制約` の「JSON 以外のテキストを含めない」行は、prompt
callable が発話を素通しする最小構成でも低精度・高速モデルが分類タスクとして JSON のみを返すための
固定文。`IntentPrediction.model_json_schema()` は prompt 生成には使わない（Field description は pydantic
schema 利用者向けメタで、LLM への提示は本文の手書きセクションが担う）。既定は
`include_rationale_in_prompt=False` で出力例に `rationale` を含めず、LLM の生成トークン・レイテンシを
抑える（rationale が欲しい利用者は `include_rationale_in_prompt=True` に切り替える）。カスタマイズ引数は
`extra_instructions` と `include_rationale_in_prompt` のみで、他は `_llm.py` の parser と serialize を
単一契約に固定するため持たない。

### `ConfidenceLevel`（5 段階カテゴリカル）

意図候補の信頼度は 5 段階の `ConfidenceLevel(str, Enum)` で表す:
`CERTAIN` / `HIGH` / `MEDIUM` / `LOW` / `SPECULATIVE`。`IntentPrediction.candidates` は
`CERTAIN > HIGH > MEDIUM > LOW > SPECULATIVE` の降順にソートされ、同レベル内は LLM 出力順を保存する。
`policy.max_candidates` で切り詰められる。

各値の意味は module 内の単一ソース `_CONFIDENCE_LEVEL_MEANINGS: dict[str, str]` に集約され、
`IntentCandidate.level` の `Field(description=...)`（pydantic schema 利用者向け）と
`IntentPolicy.render_prompt()` の `# 信頼度 (level)` セクション（LLM 向け）の双方が同じ dict から派生する。
`_LEVEL_ORDER`（`_llm.py` の post-hoc sort キー）は `enumerate(ConfidenceLevel)` で enum 宣言順から
導出され、値追加時に二重管理を発生させない。値の意味をカスタマイズする API は提供しない（1 箇所編集で
完結する構造を維持する）。

### プロンプトの契約と自動注入 / escape hatch

利用側は `prompt: Callable[[IntentContext], str]` を必須で渡す（str テンプレートは提供しない）。callable の
責務は「現在発話の user content 生成のみ」に純化されており、`IntentContext.utterance` / `history_items` /
`run_context` を型付きで受け取り、user メッセージ本文（str）を返す。履歴の文字列埋め込みは不要（SDK が
`Runner.run(input=list[dict])` で multi-turn として `history_items` を別途送るため）。既定運用では
`lambda ctx: ctx.utterance` の 1 行で足りる。

`LLMCandidateGenerator` / `intent_classifier_from_model` の既定（`include_policy_in_system=True`）では
`policy.render_prompt()` の出力を LLM 呼び出しの system role に自動注入する。`extra_instructions` は
`render_prompt()` の先頭に組み込まれるため、自動注入経路でも system の先頭に届く。利用側が prompt 内へ
手動で組み込みたい場合は `include_policy_in_system=False` を指定して自動注入を抑制する（escape hatch）。
両フラグは目的が独立で衝突しない。

### pydantic BaseModel による単一ソース化

`runtime/intent/types.py` の全型（`IntentCategory` / `IntentPolicy` / `IntentQuery` / `IntentContext` /
`IntentCandidate` / `ConsistencyReport` / `IntentPrediction`）は pydantic BaseModel（`frozen=True`）で
定義される。`IntentQuery` / `IntentContext` は `Generic[TContext]`（`arbitrary_types_allowed=True`）で
`run_context` の型を利用側が特定できる。

pydantic 採用により、LLM I/O 契約は次のように単一ソース化される。

- スキーマ生成: `IntentPrediction.model_json_schema()` は pydantic 利用者向けメタとして自動導出される
  が、`render_prompt()` は使わない（LLM 提示は手書き 4 セクションが担う）。
- Parse 検証: LLM 出力（adapter は raw `str` を返す）を `_llm.py` 側で
  `IntentPrediction.model_validate_json(text)` により pydantic parse する。SDK の `output_type`（strict
  structured output）は生成速度への影響が大きいため採用しない。パース前に `_strip_code_fence` で
  Markdown コードフェンス（```json ... ```）を剥がす耐性を持つ（低精度・高速モデルが「JSON のみ」の
  指示に反してフェンスで包む既知の失敗モードへの保険）。型検査 / 必須フィールド検査 /
  `ConfidenceLevel` の未知値は parse エラーとして扱う（構造破綻）。
- Post-hoc 加工は次の 3 段のみを lib 側で行う（過剰な in-band 加工を避け SDK Span でトレースする方針）:
  1. **allowlist フィルタ**: `text ∈ policy.categories.name` を満たさない候補を silent に除外し、除外が
     発生した場合のみ `logger.warning(...)` で最低限の可視性を残す（discoverability 目的）。
  2. **sort**: `ConfidenceLevel` 降順（`CERTAIN > HIGH > MEDIUM > LOW > SPECULATIVE`・同レベル内は LLM
     出力順を保存）。
  3. **truncate**: `policy.max_candidates` で切り詰め。

  rationale の必須性検証や `metadata.rejected` への記録は行わない（rationale を強制したい利用者は
  `extra_instructions` に自然文で書く）。

  post-hoc 3 段は `LLMCandidateGenerator` の責務であり、`DefaultIntentClassifier` は generator の
  出力を素通しする（policy を強制しない）。独自 `CandidateGenerator` を DI する場合、policy を
  守らせたいなら実装側で同等の適用を行う。`IntentPrediction.candidates` の降順ソートは既定実装のみが
  保証し、Protocol / 型としては強制されない。独自 generator の組み立ては
  `intent_classifier_from_generator(generator, *, history_limit=20)`（`intent_classifier_from_model` の
  対称形・`DefaultContextBuilder` を内部で束ねる 1 行ヘルパ）で行える。generator の型検証は行わず
  素通しで格納する（既存 factory と同一の非検証契約）。ContextBuilder まで差し替える場合は
  `DefaultIntentClassifier` を直接組み立てる。

### `agents.Model` の DI と SDK 隔離

`LLMCandidateGenerator(model, prompt, *, policy, include_policy_in_system=True, model_settings=None)` の
`model` は `agents.Model` 相当を不透明型（`Any`）として受ける DI。`model_settings` も同様に
`agents.ModelSettings` 相当の不透明型 DI で、reasoning effort / verbosity / max_tokens 等の
チューニングを利用側から渡す（None なら SDK 既定。`intent_classifier_from_model` も同名 kwarg で
pass-through する）。環境変数は参照しない（env 参照は runtime レイヤの規約通り `runtime/cli` 境界に
閉じる）。

SDK 結合は `_adapters/intent.py`（薄いラッパ
`async run_intent_prompt(model, system, history_items, user_content, *, context=None,
model_settings=None) -> str`）に閉じる。
adapter は `Agent(name="intent-classifier", instructions=system or None, model=model)`（`model_settings`
が非 None のときのみ `model_settings=` を付与）を組み、
`Runner.run(agent, input=input_items, context=raw_ctx)` を呼んで `str(result.final_output)`（`None` の
場合は `""`）を返す。`input_items` は `history_items` に、`user_content` が非空の場合のみ
`{"role":"user","content":user_content}` を末尾 append した list（空文字の `user_content` は turn を
追加せず履歴のみを送る）。utterance と history の両方が空で `input_items` が空になる場合は
`Runner.run` 到達前に `ValueError` で fail-fast する。返り値型は `str` のままで `output_type=` は
付けない。`context` は共有ヘルパ `unwrap_run_context`（`_adapters/run_context.py`）で
`RunContextWrapper` を開封してから forward する。この 1 経路で「単一発話（`history_items=()`）/ 履歴付き / 履歴のみ
（`user_content=""`）/ RunContext 付き」のケースを switch なしで捌く。

`runtime/intent/` の非 `_adapters` ファイルは `from agents` / `from openai` を含めない。`agents` の import は
`_adapters/intent.py` 内の関数内遅延で行い、`import oai_agentspec` は intent extra 未導入でも壊れない
（PEP 562 遅延再エクスポート）。SDK 隔離 grep（`_adapters` 外に `from agents` / `import agents` を許さない）
と単方向依存（`runtime/intent` からコア `_adapters` / 宣言層型への上向き参照のみ）は既存規約通り。
`_default.py` から呼ぶ `history.get_items(limit=...)` は opaque object への duck-typed メソッド呼び出しで
あり、NFR-1（`from agents` / `import agents` の import 文）の対象外。

### 公開窓口と配置

公開窓口は `oai_agentspec.runtime.intent`（他 runtime extra と同型・コア `__all__` には載せない）。公開シンボル:
`ConfidenceLevel` / `IntentQuery` / `IntentContext` / `IntentCategory` / `IntentPolicy` / `IntentPrediction` /
`IntentCandidate` / `ConsistencyReport` / `IntentClassifier` / `ContextBuilder` / `CandidateGenerator` /
`DefaultIntentClassifier` / `LLMCandidateGenerator` / `intent_classifier_from_model` /
`intent_classifier_from_generator` / `confidence_mapper_from_thresholds` / `prediction_from_scored_labels` /
`MLCandidateGenerator` / `IntentTrainer` / `TrainedIntentEstimator` / `make_trained_estimator` /
`fit_ml_estimator` / `ml_inference_from_estimator` / `intent_classifier_from_ml_inference`（計 24 件）。

### ML ベース分類器支援

`LLMCandidateGenerator` と独立並置で、ML 分類器（sklearn / 軽量 Transformer / ONNX 等・方式非依存）を
`CandidateGenerator` Protocol へ差し込むための支援層を持つ。LLM 版との連携はなく、「文章・数値特徴量 →
ML 推論 → `IntentPrediction`」を推論から学習まで一貫した型・規約で扱う。ライブラリ本体は ML フレームワークへ
一切依存せず（estimator は duck-typed `Any`）、SDK 隔離・単方向依存の規約は「SDK 隔離と依存性注入（DI）」
節が SoT で本節では再掲しない。

ファイル構成は推論側 `_ml.py` と学習側 `_ml_training.py` の 2 分割。学習側に build-don't-run の唯一の
逸脱（`fit_ml_estimator` が `estimator.fit()` を駆動する）を物理隔離する。検討経緯は
`docs/adr/0004-intent-ml-fit-deviation.md` を参照する。

- **推論側（`_ml.py`）**:
  - `confidence_mapper_from_thresholds(*, certain, high, medium, low, speculative, on_out_of_range="error")`:
    5 段階の閾値から `float -> ConfidenceLevel` の mapper を組み立てる。`on_out_of_range="clamp"`
    指定時のみ範囲外スコアを clamp し、既定（`"error"`）では `ValueError`。
  - `prediction_from_scored_labels(scored_labels, *, policy, mapper)`: `(label, score)` 列から
    `IntentPrediction` を組み立てる。処理順は「重複ラベルは最高スコアに集約 → `policy.categories` の
    allowlist フィルタ（除外は `_llm.py` と同一トーンの `logger.warning`）→ mapper で
    `ConfidenceLevel` へ変換 → level 降順 sort（同レベル内は入力順を保存）→ `policy.max_candidates`
    で truncate」。ソートキー `_LEVEL_ORDER` は `_llm.py` と単一ソースを共有する。
  - `MLCandidateGenerator(inference, *, policy, mapper)`: `CandidateGenerator` Protocol 実装。
    `inference: IntentContext -> Sequence[tuple[str, float]]`（同期/非同期いずれも可）を構築時に
    `inspect.iscoroutinefunction` で判別し、同期 callable は `asyncio.to_thread` でイベントループを
    ブロックせず実行する。例外は握り潰さず伝播する。
- **学習側（`_ml_training.py`）**:
  - `TrainedIntentEstimator`（frozen dataclass）: `inference`（推論 callable）・`estimator`（学習済み
    estimator を利用者が再利用・保存できるよう保持。既定 `None`）・`decoder`（ラベル逆写像。既定
    `None` = 恒等）を持つ学習成果物。`IntentTrainer` は
    `Callable[..., TrainedIntentEstimator]` の型エイリアスで、lib は trainer を呼び出さず戻り値型のみを
    契約とする。`make_trained_estimator` は利用者自作 trainer の成果物を束ねる builder。
  - `ml_inference_from_estimator(estimator, *, transform=None, decoder=None)`: 学習済み
    estimator（`predict_proba` / `classes_` を要求。欠如は `AttributeError`）から推論 callable を組み立てる
    （fit を駆動しない）。
  - `fit_ml_estimator(estimator, *, x_train, y_train, policy, transform=None, label_encoding=None)`:
    sklearn 互換 estimator（`fit` 属性を要求。欠如は `AttributeError`）の `estimator.fit()` を 1 回駆動し
    `ml_inference_from_estimator` を内部再利用して `TrainedIntentEstimator` を返す。
    `label_encoding: Mapping[str, Any] | None` はラベル文字列→内部表現の写像（`None` は素通し）で、
    逆写像は本写像から lib が構築し推論 callable に組み込む。
- **結線（`factories.py`）**: `intent_classifier_from_ml_inference(inference, *, policy, mapper=None,
  thresholds=None, history_limit=20)` は `MLCandidateGenerator` を組み立てて既存
  `intent_classifier_from_generator` に結線し `DefaultIntentClassifier` を返す（LLM 版
  `intent_classifier_from_model` と対称の 1 回呼び出しファクトリ）。`mapper` と `thresholds`
  （5 段階名をキーとする `Mapping[str, float]`。内部で `confidence_mapper_from_thresholds` へ展開）は
  排他で、両方指定・どちらも未指定は `ValueError`。`inference` に `TrainedIntentEstimator` を直渡しした
  場合は内部で `.inference` を取り出す。

依存はライブラリ本体に一切追加しない（sklearn は examples 実行時のみ `[dependency-groups]` の `examples`
グループで導入する）。

## Resilience（Model Retry と Run Budget・`runtime/resilience`）

Model 呼び出しの一時失敗リトライと run 全体の予算超過制御の宣言型を `runtime/resilience` に置く。
宣言型 2 種（frozen dataclass）を SDK ネイティブ機構（`ModelSettings.retry` / `Runner.run(hooks=...)`）へ
コンパイルするのみで、lib 独自の実行ループ・公開の実行 API を持たない（build-don't-run）。例外は SDK の
伝播経路をそのまま使い呼び出し元まで届く。`oai-agentspec[resilience]` extra で opt-in 導入し、extra は
追加の外部依存を持たない（`resilience = []`）。純粋追加であり、コア `__all__`・`AgentSpec` の
フィールド集合は不変。設計判断の検討経緯は `docs/adr/0002-resilience-declarative-compilation.md` を参照。

### 配置と依存方向

- `runtime/resilience/`: 宣言型（`_types.py`）・例外（`_errors.py`）・公開窓口（`__init__.py`）。
  `agents` 非依存の宣言層
- `_adapters/resilience.py`: SDK 結線の単一窓口。`build_model_retry` / `build_run_budget_hooks` と
  内部 `_BudgetHooks(RunHooksBase)` を持ち、`from agents` はここに閉じる
- 依存は `runtime/resilience` からコア（`_adapters` / `constants`）への上向き単方向のみ。コア
  （spec / registry / handoffs / prompts / workflow）から `runtime/resilience` への依存辺はない

### `ModelRetryPolicy`（Model 呼び出し retry の宣言）

Model 呼び出しの retry 条件（回数・backoff・条件）を宣言する frozen dataclass。
`build_model_retry(policy)` が SDK `ModelRetrySettings` へコンパイルし、`ModelSettings.retry` に埋め込む。

- セマンティックフラグ（`retry_on_network_error` / `retry_on_timeout` / `retry_on_rate_limit` /
  `retry_on_server_error` / `retry_on_retry_after`。既定すべて True）と `extra_retry_statuses` を
  `retry_policies.any(...)` へ合成し、**必ず `policy` を埋める**（SDK の `policy` 未指定 silent no-op
  = max_retries だけでは一切 retry しない挙動を構造的に排除する）
- 生の `policy` callable を渡した場合はセマンティックフラグを無視して `policy` を優先する
  （エスケープハッチ・条件の組み立ては利用者責務）
- build-time 検証（`ValueError` で fail-fast）: `max_retries` 負数 / `backoff_multiplier < 1` /
  `initial_delay > max_delay`。**有効条件ゼロ（全フラグ False かつ `extra_retry_statuses` なしかつ
  生 `policy` なし）で `max_retries` が正の場合も矛盾宣言として `ValueError`**
- backoff 値の未指定は SDK 既定に委譲する（lib 側で既定値をハードコードしない）
- Agent 単位（`Agent.model_settings`）/ Runner 単位（`RunConfig.model_settings`）の両方で設定でき、
  両方指定時のマージは SDK `_merge_retry_settings` に完全委譲する（Runner 側が Agent 側を上書き。
  lib 側のマージ実装は持たない）
- **`retry_on_network_error` と `retry_on_timeout` は SDK `retry_policies.network_error()` に
  まとめてコンパイルされ、独立に無効化できない**（どちらか True なら両方が retry 対象になる）。
  SDK に timeout 単独の retry プリミティブが存在しない制約に由来する。timeout のみを retry
  したい場合は生 `policy` にカスタム callable を渡す

### `RunBudgetPolicy`（run 全体の累積上限の宣言）と `RunBudgetExceeded`

1 回の `Runner.run` に閉じる累積時間 / 累積トークンの上限（`max_elapsed_seconds` /
`max_total_tokens`）を宣言する frozen dataclass。`build_run_budget_hooks(policy)` が
`RunHooksBase` サブクラスインスタンスへコンパイルし、`Runner.run(hooks=...)` に渡す。両上限とも
None の場合は no-op hooks を返す（意図的な無効化の許容・`ValueError` にしない）。上限の負数は
build-time `ValueError`。

上限超過時は `RunBudgetExceeded`（plain Exception・`runtime/resilience/_errors.py`）を送出する。
`usage`（トークン内訳・不透明型）・`elapsed_seconds`（累積秒）・`context`（トリガした agent 名・
LLM 呼び出し回数・超過した上限名）を属性として保持する。SDK `error_handlers` は
`MaxTurnsExceeded` / `ModelRefusalError` 限定の isinstance dispatch のため、`RunBudgetExceeded` は
素通しで呼び出し元まで伝播する（塗りつぶしなし・SDK ネイティブ `RunErrorHandlers` と併用可能で
相互干渉しない）。

enforcement 特性:

- 判定は `on_llm_end` のターン境界のみ（graceful）。tool 実行中の割り込みはしない。ハード timeout
  （tool 実行中も含む即中断）が必要な場合は、利用者が `asyncio.wait_for(Runner.run(...), timeout=...)`
  を自前で被せる（docstring で案内）
- 累積トークンは `context.usage` を読むだけで自前加算しない（SDK run_loop が `on_llm_end` 直前に
  加算済みのため、自前加算は二重計上になる）。usage が取得できないターンは 0 として扱い、無音に
  せず `logger.warning`（構造化: agent 名・ターン番号・理由。logger 名は
  `constants.RESILIENCE_LOGGER_NAME`）で通知する
- 経過時間は最初の `on_llm_start` で `time.monotonic()` を遅延初期化する（hooks 構築から run 開始
  までの待機時間を予算に混入させない）
- hooks は 1 run 1 インスタンス（`build_run_budget_hooks` は毎回新インスタンスを返す）。budget hooks と
  他の `RunHooksBase` を併用したい場合は `chain_hooks`（下記）で合成する

### hooks 合成（`chain_hooks`）

`Runner.run(hooks=...)` が単数の `RunHooksBase` しか受けない制約を埋めるため、複数の `RunHooksBase` を
宣言順に fan-out する汎用ヘルパー `chain_hooks(*hooks) -> RunHooksBase` を提供する。公開窓口は
`oai_agentspec.runtime.hooks`（PEP 562 遅延再エクスポート・窓口 import 時に `agents` を発火させない）で、
実装実体は `_adapters/hooks.py`（`_ChainedHooks(RunHooksBase)` のサブクラス定義に `agents.lifecycle` の
import が不可避なため SDK 隔離に従い `_adapters` に閉じる）。

合成仕様:

- 7 メソッド（`on_llm_start` / `on_llm_end` / `on_agent_start` / `on_agent_end` / `on_handoff` /
  `on_tool_start` / `on_tool_end`）を宣言順に順次 await する（状態を持たない薄いプロキシ）
- fail-fast: 前段が例外を送出したら後段は呼ばず即伝播する（`return_exceptions` 非対応）
- `chain_hooks()`（0 引数）は全メソッド no-op の `RunHooksBase()` 素インスタンスを返す
- `chain_hooks(single)`（1 引数）は `single` をそのまま返す（合成ラッパを被せない最適化）
- SDK に hook メソッドが追加された際の追随手順（オーバーライド追加）は module docstring に明記する

詳細な判断経緯は `docs/adr/0003-hooks-chain-helper.md` を参照。

### 実行モード

- `Runner.run`: 超過時に即 raise
- `Runner.run_streamed`: 例外は `stream_events()` 消費時に raise される（イベントを回さないと観測
  されない）。`ModelRetryPolicy` / `RunBudgetPolicy` とも streaming で透過的に効く
- `Runner.run_sync`: 対応（内部で `run` を呼ぶため透過的に効く）
- Realtime（`RealtimeRunner` / `RealtimeSession`）は非対応

### 公開窓口と配置

公開窓口は `oai_agentspec.runtime.resilience`（他 runtime extra と同型・コア `__all__` には載せない）。
PEP 562 遅延再エクスポートで、宣言型（`ModelRetryPolicy` / `RunBudgetPolicy`）は外部依存ゼロのため
直 import、`build_model_retry` / `build_run_budget_hooks` と SDK 生型は `__getattr__` で
`_adapters.resilience` 経由の遅延取得とし、窓口 import 時に `agents` を発火させない（extra 未導入
耐性）。例外 `RunBudgetExceeded` は本窓口からは撤去済みで、正規経路は `oai_agentspec.exceptions`
（統一窓口）。

SDK 生型の再エクスポート（10 種。上級用途で利用者コードに `from agents` を書かせないための窓口）:
`ModelRetrySettings` / `ModelRetryBackoffSettings` / `retry_policies` / `RetryDecision` /
`RetryPolicyContext` / `ModelRetryNormalizedError` / `RunErrorHandlers` / `RunErrorHandlerResult` /
`RunErrorHandlerInput` / `RunErrorData`。

## テスト層

| 層 | 依存 | 対象 |
|---|---|---|
| L1（純ロジック） | `agents` 非依存 | registry / handoffs / prompts のロジックを `FakeAgentBuilder` で検証。workflow は `FakeRunnerAdapter`（runner シーム fake）を内部インタプリタへ注入し、エッジ走査 / fan-out / fan-in / 条件分岐 / ループ / メッセージ受け渡し / fan-out + session fail-fast を検証。HITL は承認待ちの検知・承認/却下の解決ロジック（call_id 単位・部分解決・全解決での再開・却下時継続・承認系構造化エラー）を fake で検証 |
| L2（統合） | 実 `Agent` + `FakeModel` + `Runner` | 動的 instructions・循環ハンドオフ・サブエージェント委譲の実挙動を検証。workflow は経路C（`as_agent_spec` → `Runner.run` で決定論起動・handoff 流入）、経路A（`as_facade_spec` → context 透過・既定 input_filter）、`tool_choice` を extra に積むと `ValueError` / `model_settings` なら成功する回帰を検証。HITL は `FakeModel` が `needs_approval` ツールへ ToolCall を返して interruptions を生成し、複数 `call_id` を可変に与えるヘルパで段階解決を検証。承認前は実行記録が残らず approve 後に初めて残ること（NFR-7）を実行記録で検証する |

Realtime ルート（`realtime/`）も同じ 2 層で検証する。L1 は `FakeRealtimeAgentBuilder` を注入して spec の
フィールド排除・registry の遅延構築 / 循環解決 / 重複登録 / validate を `agents` 非依存で検証し、加えて
`RealtimeHandoffGraph` のグラフ宣言 / `apply` の spec 反映 / 反映順序非依存の検証 / `mermaid` / 「グラフ apply ==
spec 直接宣言」の等価性を `FakeRealtimeAgentBuilder` 注入で検証する。L2 は
`RealtimeRunner` + `FakeRealtimeModel`（`RealtimeModel` 抽象実装）で handoff の実委譲を検証する。個別
assert・テスト名は docstring を一次情報とする既存方針を踏襲する。加えて Realtime の隔離不変条件
（コア `__all__` に Realtime シンボルを含まない・`import oai_agentspec` が `realtime` を連鎖 import しない・
L1 テストダブルが `agents` に依存しない）は clean subprocess での import 検証と `__all__` の introspection
テストで機械的に固定する。

intent 層（`runtime/intent`）も同じ 2 層で検証する。L1 は型（frozen・`IntentPolicy` の categories
検証等）と `render_prompt()` の出力・`intent_classifier_from_model` /
`intent_classifier_from_generator` の組み立てを `agents` 非依存で pin する。L2 は `LLMCandidateGenerator` の post-hoc 3 段（allowlist / sort / truncate）・コードフェンス
耐性・prompt callable 契約（呼び出しと `user_content` / `history_items` / `context` の forward）を
検証し、adapter（`_adapters/intent.py`）は 3 ケース（単一発話 / 履歴付き / RunContext 付き）+
空入力（utterance と history の両方が空）の fail-fast を `FakeModel` で検証する。加えて公開窓口の
PEP 562 遅延再エクスポートと `__all__` 24 件の pin、履歴のみモード（`utterance=""` + history）の
end-to-end 動作と空入力時の `ValueError` 伝播を検証する。

ML ベース分類器支援は同じ 2 層で検証する。L2 の `test_ml_l2.py` / `test_ml_training_l2.py` は
`predict_proba` / `classes_` / `fit` を持つ duck-typed fake estimator（sklearn 非依存）で
mapper・dedup・allowlist・sort・truncate・同期/非同期ブリッジ・fit 駆動・ラベルエンコード/復号を
検証する。sklearn 自体はテストスイートで一切使用せず、examples 実行時のみの依存に留める。

L2 には SDK バージョン耐性トリップワイヤ（NFR-7）を置く。openai-agents SDK との結合点で手組みしている
前提（`Model` 抽象メソッド集合・手組みレスポンス型の必須フィールド集合・入口入力の正規化形式への依存・
context 透過経路の公開構造・`tool_choice` の既知 `Literal` メンバ・HITL の interruptions の存在 /
`RunState` の `to_string` / `from_string` 往復 / approve / reject の API 形）が SDK の将来変更で build-time にも
実行時例外にも現れないまま静かに退行するのを防ぐため、前提が崩れたら CI で fail させる。検知の網だけを
足すもので src の挙動・公開 API は変えない。個別 assert・テスト名は陳腐化するため docs には書かず、各
トリップワイヤテストの docstring を一次情報とする。

`tests/` は src の構造をミラーする（`_adapters` の分割に対応する `tests/_adapters/` を含む）。
src でトップレベルに置くモジュールのテストは `tests/` 直下に、サブパッケージのテストは対応する
サブディレクトリに置く。共通フィクスチャ・fake 群（`tests/_helpers/`）は src にミラー対象がないため
ミラー化の対象外とする。

`tests/` は `_adapters` 集約の grep 計測対象外である。`_adapters` のデフォルト `AgentBuilder` /
`handoff()` / as_tool 生成は実 Agent 構築の L2 テストで通し、行カバレッジ計測（80% 以上）に含める。
テストは `FakeModel`（`agents.Model` 継承）と conftest のトレーシング無効化・ネットワークガードに
より、外部 API への通信を発生させず認証情報の有無に関わらず通過する。

## リポジトリガバナンス（OSS 公開後）

本節は OSS 公開リポジトリとしての現在のガバナンス仕様を SoT として記述する。ライブラリ本体の宣言層 /
runtime 層・SDK 隔離・build-don't-run といったコア不変条件には影響しない、リポジトリ運用の現在仕様である。

### ライセンス

本ライブラリは **MIT License** で配布する。SoT はリポジトリ root の `LICENSE` ファイル本文と
`pyproject.toml` の `[project]` テーブルの `license = "MIT"` / `license-files = ["LICENSE"]`
（PEP 639 形式）であり、両者は同一ライセンスを指し示すよう同期して維持する。`classifiers` にも
`"License :: OSI Approved :: MIT License"` を掲載する。

### OSS 標準文書

GitHub Community Standards 認識のため、以下 4 文書をリポジトリ root に配置する（`docs/` 配下には置かない）。

| ファイル | 内容 |
|---|---|
| `LICENSE` | MIT License 標準テキスト（著作権者 `mugicha001`） |
| `CODE_OF_CONDUCT.md` | Contributor Covenant 2.1。連絡窓口は GitHub Issues |
| `CONTRIBUTING.md` | 開発環境（`uv sync` / Python 3.12+）/ test+lint コマンド / ブランチ命名 / コミット規約 / PR フロー / カバレッジ 80% 必須 |
| `SECURITY.md` | 脆弱性報告窓口 / サポートバージョン方針 / 初動応答 SLA / admin enforce 方針 |

### CI workflows

`.github/workflows/` 配下に 3 系統の workflow を配置する。いずれも GitHub Actions で動作し、リリース前
ローカルゲートとは別目的で常時 PR / `main` への push 時に走る。

| ファイル | トリガ | 内容 |
|---|---|---|
| `ci.yml` | `pull_request` / `push: main` | Python 3.12 単一で `uv run pytest`（カバレッジ 80% gate）+ `uv run ruff check src/ tests/` + `uv run ruff format --check src/ tests/` |
| `gitleaks.yml` | `pull_request` / `push: main` | `gitleaks/gitleaks-action@v2` によるシークレット検知 |
| `codeql.yml` | `pull_request` / `push: main` / `schedule:` 週次 | GitHub CodeQL Python（`languages: python`） |

### Dependabot

`.github/dependabot.yml` で 2 エコシステムを週次スケジュールで監視する。

- `package-ecosystem: pip`（`directory: /`・`uv.lock` の依存）
- `package-ecosystem: github-actions`（`directory: /`・`.github/workflows/` 内の actions バージョン）

### Branch protection（main）

`main` ブランチには以下の保護を有効化する（PUBLIC リポジトリで GitHub Free プランの範囲内）。

- 直 push 禁止（PR 経由必須）
- force push 禁止 / branch 削除禁止
- Required status checks: `ci` / `gitleaks` / `CodeQL` の全 job が成功必須
- "Require branches to be up to date before merging" 有効
- admin enforce ON 推奨（OFF とする場合は `SECURITY.md` に方針を明記）
- Pull request review 承認は単独メンテナ運用のため**無効**

### ローカルゲートとの責務分離

`docs/security-scanning.md` に定義するローカル SAST（SonarQube）/ SCA（Trivy）/ シークレット（gitleaks）は
リリース前にメンテナがローカル実行する**リリースゲート**用途であり、CI には組み込まない。CI 側
（`ci.yml` / `gitleaks.yml` / `codeql.yml`）は PR / push 時に常時走る常設チェック用途であり、両者は目的・
実行タイミング・実行主体が異なる補完層として併存する。
