# Rationale: AGT（Agent Governance Toolkit）統合可否の技術検証

本ファイルは AGT 統合可否の技術検証（Spike）の調査結果と判断を保持する archival ドキュメントである。
実装は本判断に基づく後続作業で行い、本ファイルは検討経緯として不変に保つ（実装変更に追随して更新しない）。
調査対象は Microsoft 製 OSS `agent-governance-toolkit`（AGT）である。

## 結論サマリ

- 判断: **条件付きで統合する**。
- 取り込む最小機能（MVP）: **Agent OS のツール単位ポリシー強制 + 監査ログ**の 2 つ。
- 結線手段: AGT の **`[openai-agents]` 連携アダプタ（first-tier・軽量）経由**で、本ライブラリの既存 DI（`AgentBuilder` 注入点 + `AgentSpec.hooks` 素通し）に乗せる。
- SDK/AGT 結合は `_adapters/` に局在させ、SDK 隔離 grep を空に保つ。追加依存は `governance` extra に隔離する。
- 条件: 後述の 3 条件（リポジトリ側ライセンス確定・MVP をポリシー強制+監査に限定・連携はアダプタ採用）を満たすこと。

## 1. AGT の実態（FR-1）

### 1.1 パッケージ構成と依存の重さ

AGT は 7 つのコアパッケージと補助コンポーネントからなる大きな体系で、Python のほか TypeScript / Rust / Go / .NET を提供する。MIT ライセンス。

ツール単位ポリシー強制に必要な依存は軽量で、本ライブラリへの限界増分は小さい。

| 区分 | 取得方法 | 主な依存 | 重さ |
|---|---|---|---|
| base（ツール単位 `govern`） | `pip install agent-governance-toolkit` | `pydantic` / `pyyaml` / `click` | 軽量（`pyyaml` は本ライブラリの既存依存・`pydantic` は `openai-agents` 経由で推移的に既存） |
| OpenAI Agents SDK 連携 | `agent-governance-toolkit[openai-agents]` | base + `agent-governance-toolkit-core` + `openai-agents`（本ライブラリのコア依存と同一） | 軽量（grpc/azure/opa/torch 等の重い推移依存なし） |
| フルスタック | `agent-governance-toolkit[full]` | core + integrations + cli + protocols | 重い（本 MVP では不要） |

### 1.2 ポリシー強制 API（`govern`）

代表 API はツール（callable）をラップする形をとる。ツール実行の直前にポリシーを評価し、許可なら実行・違反なら実行させず例外（`GovernanceDenied`）を送出する。ポリシー記述は YAML / OPA Rego / Cedar に対応し、ステートレスかつ低レイテンシのポリシーエンジンとして動く。

### 1.3 OpenAI Agents SDK 連携の位置づけ

OpenAI Agents SDK は AGT の **first-tier 連携**（shipped・PyPI 公開済み）で、Microsoft 自社フレームワークと同格の扱いである。連携アダプタは SDK のネイティブ拡張点（関数ツールのラップ・`RunHooks`/`AgentHooks` ライフサイクル）に governance を差し込み、エージェントコードの書き換えを不要にする。

### 1.4 スコープ境界（内容ガードレールは AGT のコア外）

AGT は「エージェントが何をできるか」を規則で許可/拒否するケーパビリティ/ポリシー制御 + 監査の層であり、「エージェントが何を言うか」を検査する**内容ガードレール（PII 検出・プロンプトインジェクション/ジェイルブレイク検出・モデレーション・出力フィルタ）はコア機能ではない**（プロンプトインジェクション検出のみ別 add-on モジュールとして存在）。内容ガードレールが必要な場合は OpenAI Agents SDK ネイティブの `input_guardrails`/`output_guardrails` や専用ライブラリ（`openai-guardrails` 等）を併用する。これらは AGT 由来ではなく本統合の対象外とする。SDK の guardrail 機構自体は run 中断（緊急停止）の実現手段としてのみ後段で参照する。

内容ガードレールは AGT 統合の前提ではない。守る対象が「何をできるか」（AGT ガバナンス）と「何を言うか」（内容ガードレール）で直交するため、導入順序に制約はない。さらに `input_guardrails`/`output_guardrails` は `agents.Agent` の正規フィールドであり、本ライブラリでは既存の `AgentSpec.extra` 素通し（`build_agent` が Agent 有効フィールドのみを許可して通す）でそのまま宣言できる。したがって内容ガードレールの宣言経路は追加実装・追加依存なしで既に存在し、AGT ガバナンス（ツールポリシー + 監査）と独立に導入できる。専用フィールド新設やガードレール実装を束ねた extra は任意の拡張であり、本統合の前提条件ではない。

## 2. 全機能の適合度マトリクス（FR-1 / FR-4）

各機能を「性質（in-process ライブラリか外部インフラか）」「OpenAI Agents SDK の結線シーム」「設計原則適合（build-don't-run / 隔離）」で評価し、振り分けを決めた。

| # | AGT パッケージ / コンポーネント | 機能 | 主な OWASP Agentic | 性質 | SDK 結線シーム | build-don't-run | 隔離 | 振り分け |
|---|---|---|---|---|---|---|---|---|
| 1 | Agent OS（ポリシー強制 `govern`） | 実行前にツール呼出をポリシー評価・違反で `GovernanceDenied` | Goal hijacking / Tool misuse | Lib | (1) tool-wrap | ◎ | ◎ | MVP |
| 2 | Agent OS（監査ログ / decision log） | 判定の tamper-evident 記録・ライフサイクル監査 | 横断（evidence） | Lib | (2) Hooks | ◎ | ◎ | MVP |
| 3 | Agent Mesh（ID・信頼） | DID/Ed25519 身元・IATP 暗号通信・信頼スコア | Identity abuse / Insecure comms / Rogue agents | Infra（証明書・A2A 網） | なし | △ | △ | 範囲外 |
| 4 | Agent Runtime（実行制御） | execution rings・saga・kill switch | Code execution / Rogue agents / Cascading | 実行ループ所有 | kill: (2) / rings・saga: なし | 中断: △ / rings・saga: × | × | 中断系=後段 / rings・saga=範囲外 |
| 5 | Agent SRE（信頼性運用） | SLO・circuit breaker・chaos・progressive delivery | Cascading failures | Ops/監視 | circuit breaker: (2) / 他: なし | × | × | LLMOps トラック |
| 6 | Agent Compliance（準拠検証） | 監査集約→準拠グレード・証跡 | 横断（evidence） | Lib（監査ログ前提） | なし（監査の上） | ○ | ○ | 後段 |
| 7 | Agent Marketplace（プラグイン流通） | プラグインライフサイクル・Ed25519 署名・capability gating | Supply chain / Tool misuse | Infra（署名・配布） | なし | × | × | 範囲外 |
| 8 | Agent Lightning（RL 訓練） | 報酬整形による学習時ガバナンス（訓練中の違反ゼロ） | Goal hijacking / Rogue agents | Train（学習時） | 別ライフサイクル | 実行時ガバナンス外 | - | LLMOps トラック |
| 9 | MCP Security Gateway | MCP ツール実行のセキュリティゲートウェイ | Tool misuse | Infra（sidecar/別プロセス） | なし | × | × | 範囲外 |
| 10 | CMVK（Cross-Model Verification Kernel） | 複数モデル多数決でメモリ整合 | Memory poisoning | マルチモデル＋合意（実行所有） | なし | × | × | 範囲外 |
| 11 | Framework Integrations（`[openai-agents]`） | AGT コアを SDK ネイティブ拡張点へ差し込むアダプタ | -（結線手段） | Lib（first-tier・軽量） | (1)+(2) | ◎ | ◎ | MVP（結線基盤） |

凡例:

- 性質: `Lib`=in-process ライブラリ / `Infra`=外部インフラ・別プロセス / `Train`=学習時 / `Ops`=運用層
- 結線シーム: `(1)`=ツールラップ（`govern`） / `(2)`=Hooks（`RunHooks`/`AgentHooks`） / `なし`=SDK 拡張点なし（`guardrail` は SDK ネイティブ機構で AGT のシームではない。run 中断の手段として後段でのみ参照）
- 適合: ◎ 良 / ○ 可 / △ 条件付き / × 不適
- 振り分け: `MVP` / `後段`（本検証スコープ内・MVP の後） / `LLMOps トラック`（学習・運用の別トラック） / `範囲外`

篩の要点:

- 本ライブラリが受け取れるのは「in-process・build 時結線・実行ループ非所有・宣言的ツール抽象に乗る」機能のみ。これを満たすのは実質 Agent OS のポリシー強制と監査ログで、AGT 自身も「`govern` から始めて層を足す」設計である。
- Runtime（rings・saga）/ Mesh / MCP Gateway / CMVK / Marketplace は外部インフラ・別プロセス・実行所有のいずれかを前提とし、build-don't-run・薄いラッパー・extra 隔離に収まらない。
- OWASP「Human-agent trust exploitation」に対応する承認ワークフロー（quorum）の領域は、本ライブラリが既に持つ HITL 承認と重なる（詳細は `docs/architecture.md` の HITL 節）。

## 3. 結線点候補の比較と推奨（FR-2）

| 候補 | 適用範囲 | SDK/AGT 隔離 | ラッパーの薄さ | build-don't-run | 評価 |
|---|---|---|---|---|---|
| (A) `GovernedAgentBuilder`（builder でツール一律 `govern` + 監査フック装着） | 全ツール一律 + ライフサイクル監査 | ○（`_adapters` に閉じる） | ○（builder 1 実装・既存 DI に乗る） | ◎（build 時結線・実行は SDK） | 推奨 |
| (B) SDK ネイティブ guardrail / hooks 単独 | 入出力内容チェック・run 中断 | ○ | △（agent ごと設定・内容ガードレール本体は別ライブラリ） | ◎ | 補完（AGT はツール単位ポリシーが主体で内容ガードレールは AGT 由来でない） |
| (C) runner シーム（`Runner.run` ラップ） | run 全体 | △（Runner を丸ごとラップ） | ×（公開実行口を新設） | ×（公開 run API を持たない方針に反する） | 却下 |

推奨は **(A)**。理由:

- 既存の DI 注入点に乗る。`AgentRegistry(agent_builder=...)` に `DefaultAgentBuilder` を装飾する `GovernedAgentBuilder` を注入するだけで、`AgentSpec` / `tools` の宣言面は不変のまま全ツールに `govern` を一律適用できる。
- 監査は同 builder が `AgentSpec.hooks`（→ `agents.AgentHooks`）相当のフックを装着する形で結線でき、実行は SDK Runner に委ねる（build-don't-run に一致）。
- ポリシー強制（ツール単位の許可/拒否）の粒度は (A) のツールラップが最も自然で、(B) の入出力 guardrail は内容チェック止まりのため補完的位置づけにとどまる。

緊急停止（Runtime の中断系）は将来 (B) の guardrail/hook で「条件成立時に run を止める」として後段で扱える。execution rings・saga は SDK に差込口が無く対象外。

## 4. 設計原則との整合性（FR-3 / NFR-1 / NFR-3）

| 原則 | 判定基準 | 統合方針での扱い |
|---|---|---|
| SDK 隔離（NFR-1） | `grep -rnE "(from agents\|import agents)" src/oai_agentspec/ \| grep -v _adapters` が空 | `agents` / AGT の import を新規モジュール `_adapters/governance.py` に閉じる。上位層（registry / spec / protocols / runtime 配下）は不透明な `AgentBuilder` のみを扱う。AGT アダプタ内部の `import agents` はサードパーティ側で grep 対象外。grep は空を維持できる（NFR-3） |
| extra 隔離 | 本体必須依存を増やさない | AGT 依存は `governance` extra（`agent-governance-toolkit[openai-agents]`）に隔離。`import oai_agentspec` は extra 未導入でも壊れない（governance の import は関数内遅延）。公開窓口は `runtime/governance/`（conversation/serve/cli と同型） |
| モデルの外部流入 | LLM モデル系を内部に持たない | governance 結線はツールラップとフック装着のみでモデルを一切埋め込まない。モデルは従来どおり `AgentSpec.model` 等で外部から DI 流入させる（他層と同様・env 非依存方針を維持） |
| build-don't-run | 公開の実行 API を持たない | ポリシー強制・監査の結線はすべて build 時。実行は SDK `Runner.run` に委ね、独自実行エンジンを新設しない |
| ライセンス整合 | MIT 整合 | AGT は MIT で copyleft 義務なし。ただし本リポジトリの `license` は現状 `pyproject.toml` で未確定（公開前にコメントアウト）。governance 統合の前提として確定が必要（後述の条件） |

## 5. 最小スコープと利用イメージ（FR-4）

MVP は「ポリシー強制 + 監査」の 2 機能。各機能が何に対して何をするかを以下に示す。

| 機能 | 何に対して | 何をする | いつ動く | 結線シーム |
|---|---|---|---|---|
| ポリシー強制（Agent OS `govern`） | エージェントが呼ぼうとする各ツール呼び出し（関数名 + 引数） | YAML ルール表に照合し、許可なら実行・違反なら実行させず `GovernanceDenied` を送出 | ツール実行直前（LLM の tool call 後・実関数の実行前） | (1) ツールラップ |
| 監査ログ（Agent OS decision log） | エージェントのライフサイクル事象（ツール開始/終了・ハンドオフ・ポリシー判定） | 「誰が・いつ・どのツールを・どの引数で呼び・許可/拒否されたか」を改ざん検知可能な記録として残す | 実行中（各事象発生時） | (2) Hooks |

後段（本検証スコープ内・MVP の後）に置く候補:

| 機能 | 何に対して | 何をする | 前提 |
|---|---|---|---|
| 準拠レポート（Agent Compliance） | 蓄積された監査ログ | 規制要件（SOC2 / EU AI Act 等）への適合度を集計・採点し証跡化 | 監査ログが存在すること |
| 緊急停止（Agent Runtime 中断系） | 実行中の run | 危険条件（連続違反等）で run を即時中断・以後のツール実行を遮断 | SDK ネイティブの guardrail/hook で表現（rings・saga は対象外） |

利用イメージ（必要記述量。API は後続実装で確定する暫定形）:

```python
# governance extra 導入時（oai-agentspec[governance]）
from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.governance import GovernedAgentBuilder

# 既存コードに対する追加は「builder を 1 つ差し替える」ことと「policy ファイル」のみ。
registry = AgentRegistry(
    agent_builder=GovernedAgentBuilder(policy="policies/support.yaml"),
)
registry.register(AgentSpec(name="support", tools=[refund, lookup_order, send_email]))
agent = registry.get("support")  # 各 tool が govern 済み・監査フック装着済み
```

`AgentSpec` / `tools` の宣言面は不変で、ガバナンスは builder の差し替え 1 行と policy ファイルで適用される。

## 6. 実現可否判断と後続作業（FR-5）

判断: **条件付きで統合する**。条件は次の 3 点。

1. 本リポジトリの `license` を確定する（`pyproject.toml` で未確定。MIT 想定。AGT は MIT で整合）。
2. MVP はツール単位ポリシー強制 + 監査ログに限定する（フルスタックや重い機能を一度に取り込まない）。
3. 連携は `[openai-agents]` アダプタを採用する（first-tier・軽量を確認済み。コアの `govern` 直接利用に対しアダプタ採用で結線が簡潔になる）。

後続 Issue の分割案（最小機能の実装単位）:

- `governance` extra の新設と `_adapters/governance.py`（`agents` / AGT 結合の単一窓口）。
- `GovernedAgentBuilder`（ツール一律 `govern` + 監査フック装着）と `runtime/governance/` 公開窓口。
- per-spec ポリシー宣言の使い勝手（registry 一括適用に加えた spec 単位の上書き経路）の検討。
- 後段機能（準拠レポート・緊急停止）の取り込み可否の再評価。

LLMOps トラックへの振り分け（本検証の対象外・相互参照のみ）:

- Agent Lightning（学習時 RL）と Agent SRE（信頼性運用）は実行時ガバナンスとはライフサイクルが異なるため、LLMOps 評価 extra の別トラックで扱う（`docs/requirements/llmops-evaluation-extra.md` 系の検討に接続）。

本検証は調査のみで、`src/oai_agentspec/` の挙動・公開 API（`__all__`）は変更していない（NFR-2）。
