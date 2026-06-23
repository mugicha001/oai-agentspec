# governance extra の使い勝手改善（per-agent ポリシーと PolicyViolationError 再エクスポート）

## 1. 概要

`oai-agentspec[governance]` extra（Issue #28 で実装済み・未リリース）の `GovernedAgentBuilder` は現状 registry 単位の一括ポリシーのみを受け付けるため、マルチエージェント構成でエージェントごとに許可ツールを出し分けられない。また拒否例外 `PolicyViolationError` の捕捉に AGT 内部パッケージ import と DeprecationWarning 抑制のボイラープレートが必要である。本機能は、builder の overrides 方式による per-agent ポリシーと、`oai_agentspec.runtime.governance` 公開窓口からの `PolicyViolationError` 再エクスポートにより、これら 2 点の使い勝手を改善する。

## 2. 機能要件

### FR-1: per-agent ポリシー（builder overrides 方式）

- ユーザーストーリー: マルチエージェント構成（triage / support / admin 等）の運用者として、エージェントごとに異なるツールポリシーを適用したい。なぜなら現状の registry 一括ポリシーでは、allowlist にツールを載せるとそのツールを tools に持つ全エージェントで許可となり、同一ツールのエージェント別出し分けができないから。
- 受け入れ基準:
  - [ ] WHEN `GovernedAgentBuilder(policy=既定ポリシー, overrides={"エージェント名": ポリシー})` を `AgentRegistry(agent_builder=...)` に注入し、overrides に掲載されたエージェントを build した THEN 当該エージェントの全 FunctionTool は overrides で指定されたポリシーで govern ラップされ強制される
  - [ ] WHEN overrides に掲載されていないエージェントを build した THEN 既定ポリシー（`policy` 引数）で govern ラップされ強制される（フォールバック）
  - [ ] WHEN 同一ツールを tools に持つ 2 つのエージェントに対し、overrides で一方は当該ツール allow・他方は deny となるポリシーを適用して実行した THEN 一方ではツールが実行され、他方では実行前に `PolicyViolationError` で拒否される（同一ツール名のエージェント別出し分けの実証）
  - [ ] WHEN overrides を指定せず既存どおり `GovernedAgentBuilder(policy=...)` のみで構築した THEN 既存の振る舞いと完全互換である（全エージェントに既定ポリシーが一括適用される）
  - [ ] IF overrides の値として YAML パス / AGT ポリシーオブジェクトのいずれを渡した場合でも THEN 既定の `policy` 引数と同じ形式が受理され同等に機能する
  - [ ] IF overrides の値が不正な場合（存在しない YAML パス / 未知キーを含む YAML / `check_tool`・`check_content` を欠くオブジェクト） THEN 既定の `policy` と同一の fail-fast 検証エラー（`FileNotFoundError` / `ValueError` / `TypeError`）で build 時に拒否される（異常系の同等性）
  - [ ] IF overrides の値に `None` を含む状態で当該エージェントを build した THEN `TypeError` で拒否される（既定ポリシーへ戻す意図はキーの削除で表現する・暗黙フォールバックの曖昧さを排除）
  - [ ] WHEN 登録済みの全エージェントを build した後に builder の未適用キー確認用プロパティ（名称は設計で確定）を参照した THEN overrides のうち一度も適用されなかったキーの集合が取得できる（typo 検知の opt-in 手段）。全キー適用済みなら空集合を返す
  - [ ] WHEN overrides を指定して build した THEN 監査 sink（`audit_sink`）は引き続き builder で 1 本共有される（per-agent 分割しない）
  - [ ] IF `policy` 引数を省略した場合 THEN 既存どおり `TypeError`（keyword-only 必須引数の欠落）となる（`policy` は引き続き必須・overrides は上書き専用）

### FR-2: PolicyViolationError の公開窓口からの再エクスポート

- ユーザーストーリー: governance extra の利用者として、`oai_agentspec.runtime.governance` から `PolicyViolationError` を import したい。なぜなら現状は AGT 内部パッケージ（`agent_os.exceptions`）からの import が必要で、import 時の DeprecationWarning 抑制ボイラープレートを書かないと拒否捕捉が実装できないから。
- 受け入れ基準:
  - [ ] WHEN governance extra 導入済み環境で `from oai_agentspec.runtime.governance import PolicyViolationError` を実行した THEN DeprecationWarning を発生させずに import が成功する
  - [ ] WHEN govern 済みツールがポリシー違反で拒否された THEN 再エクスポートされたシンボルで当該例外を捕捉できる（AGT が送出する例外クラスと isinstance 互換であること）
  - [ ] IF governance extra 未導入の環境である場合 THEN `import oai_agentspec.runtime.governance`（窓口 import 自体）は壊れない
  - [ ] IF governance extra 未導入の環境で `PolicyViolationError` 属性へアクセスした THEN 既存の遅延 import 方針と同型の install hint 付き `ImportError` が送出される（無言の失敗をしない）
  - [ ] WHEN 公開窓口の公開対象を確認した THEN 追加再エクスポートは `PolicyViolationError` のみである（GovernancePolicy / AuditLog 等は再エクスポートしない）

### FR-3: bundle YAML（制限の全量を 1 ファイルに宣言）

- ユーザーストーリー: governance extra の利用者として、既定 / per-agent の制限を単一の YAML ファイルにまとめて宣言したい。なぜなら制限定義が YAML（ポリシー本体）とコード（エージェント名との対応付け）に分離すると、「どのエージェントが何をできるか」の監査に両方を読む必要があるから。コードでポリシーオブジェクトを組む形式も引き続き使えること（既定 / per-agent とも YAML・コードの双方の形式を受理）。
- 受け入れ基準:
  - [ ] WHEN `default`（必須）と `agents`（任意・エージェント名 -> ポリシーフィールド）を持つ bundle YAML から builder を構築した THEN `default` が既定ポリシー・`agents` が per-agent 上書きとして機能し、通常コンストラクタ（`policy=` / `overrides=`）で同内容を組んだ場合と等価に強制される
  - [ ] IF bundle YAML に `default` セクションが無い / トップレベルに未知キーがある / セクションがマッピングでない / セクションに未知キーがある THEN `ValueError` で即時拒否される（各セクションは単一ポリシー YAML と同一の fail-fast 検証・非強制フィールドは警告）
  - [ ] WHEN bundle 構築後に `unapplied_overrides` / 監査 sink 共有 / フォールバックを利用した THEN 通常コンストラクタと同一の振る舞いをする（bundle は構築糖衣であり実行時の特別経路を持たない）
  - [ ] IF governance extra 未導入の環境で bundle 構築を呼んだ THEN install hint 付き `ImportError` が送出される（公開窓口の import 自体は壊れない）

## 3. 非機能要件

### NFR-1: 保守性（SDK / AGT 隔離）

- 要件: `agents` / AGT（`agent_os` 等）の import は `_adapters/governance.py` に閉じたままとする。`runtime/governance` はポリシー・例外を不透明値として扱い、AGT 型へ直接依存しない。
- 計測基準: `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` が空であること。AGT パッケージの import が `_adapters/governance.py` 以外の `src/oai_agentspec/` 配下に存在しないこと。

### NFR-2: 保守性（後方互換）

- 要件: overrides 未指定時の `GovernedAgentBuilder` の振る舞い・コア `__all__` のメンバ集合・AgentSpec / registry のインターフェースを不変に保つ。
- 計測基準: 既存テストスイートが無修正で green であること。公開 API スモーク（`__all__` 全件 import 可能チェック）が pass すること。

### NFR-3: 可用性（extra 未導入耐性）

- 要件: governance extra 未導入環境でも `import oai_agentspec` および `import oai_agentspec.runtime.governance` が成功する（再エクスポートは PEP 562 module `__getattr__` 等の遅延方式を想定。実現方式は設計に委ねる）。
- 計測基準: extra 未導入相当の条件下での import テストが green であること（既存の extra 未導入耐性テストと同型）。

### NFR-4: セキュリティ（強制の抜け穴を作らない）

- 要件: overrides 導入によりポリシー強制が無効化される経路を作らない。overrides 未掲載エージェントは必ず既定ポリシーへフォールバックし、「ポリシーなしで build される」状態を生まない。
- 計測基準: overrides 指定時に未掲載エージェントで既定ポリシーの deny が機能することを確認するテスト、および overrides の未適用キー（typo 相当）が確認用プロパティで検出できることを確認するテストが存在し green であること。

### NFR-5: 保守性（テスト品質）

- 要件: 追加実装はテストカバレッジ 80% 以上の維持を満たし、tests/ は src/ のミラー構造規約に従う。
- 計測基準: `uv run pytest`（`fail_under = 80`）が green であること。

## 4. 制約事項

- 技術的制約:
  - AgentSpec / tools の宣言面は不変とする。per-agent ポリシーは builder（`GovernedAgentBuilder`）内で完結させ、AgentSpec / registry は変更しない。
  - コア `__all__` は不変とする（governance シンボルは `oai_agentspec.runtime.governance` 公開窓口のみから取得する）。
  - SDK / AGT の import は `_adapters/governance.py` のみとする（SDK 隔離 grep を空に維持）。
  - 本体は env 非依存・build-don't-run（結線は build 時のみ・実行は SDK `Runner.run`）の方針を維持する。
  - overrides の引き当ては `build(spec)` 時に `spec.name` をキーとして行う。照合は完全一致のみとし、大文字小文字・前後空白等の正規化は行わない。
  - 設計（/architect）に委ねる事項: extra 未導入時の `PolicyViolationError` 属性アクセスで送出する `ImportError` の正確な文言（既存の install hint と同型とする）、および未適用キー確認用プロパティの名称。
- ビジネス制約:
  - 対象機能は Issue #28 実装済み・未リリースであり、`GovernedAgentBuilder` のコンストラクタへの keyword-only 引数追加は許容される。ただし既存引数（`policy` 必須・`audit_sink` / `inner` 任意）の意味は変えない。
- スコープ外（本要件で扱わない）:
  - soft-deny モード（拒否をツール結果として LLM に返し会話を継続する opt-in）
  - sub_agents の as_tool / register_factory 経路の govern 対象化（既知の境界のまま）
  - per-agent の監査 sink 分割
  - ポリシー強制フィールドの拡張（max_tool_calls 等の強制対応）
  - `PolicyViolationError` 以外のシンボル（GovernancePolicy / AuditLog 等）の再エクスポート

## 5. 影響範囲

- 関連コンポーネント:
  - `src/oai_agentspec/runtime/governance/`（`GovernedAgentBuilder`・公開窓口 `__init__.py`）
  - `src/oai_agentspec/_adapters/governance.py`（AGT 結合の単一窓口・例外型の取得元）
  - `tests/runtime/governance/`（ミラー構造のテスト追加）
  - `examples/governance/`（`01_policy_enforcement.py` の DeprecationWarning 抑制ボイラープレートの置換・per-agent ポリシー例の追加余地）
  - `docs/architecture.md`「AGT ガバナンス」節（per-agent overrides と再エクスポートの仕様反映）
- 既存機能への影響:
  - overrides 未指定時は完全互換であり、既存利用コードへの影響はない。
  - コア `__all__`・AgentSpec・AgentRegistry・他の runtime extra（conversation / serve / cli / llmops）への影響はない。
  - `docs/rationale/agt-governance-integration.md` §6 に後続検討として記載された「per-spec ポリシー宣言の使い勝手」が本要件で具体化される（rationale ファイル自体は immutable のため更新しない）。

## 6. 用語定義

| 用語 | 定義 |
|------|------|
| AGT | Microsoft 製 OSS `agent-governance-toolkit`。ツール単位ポリシー強制（`govern`）と監査を提供する（MIT） |
| govern ラップ | ツール（FunctionTool）実行直前にポリシー評価を行い、許可なら実行・違反なら例外送出する AGT のラップ機構 |
| GovernedAgentBuilder | `AgentBuilder` Protocol を満たす装飾 builder。build 時に spec の全 FunctionTool を govern ラップし監査フックを装着する |
| PolicyViolationError | AGT がポリシー違反時に送出する拒否例外（現状の取得元は `agent_os.exceptions`） |
| overrides | builder に渡すエージェント名 → ポリシーの辞書。掲載エージェントのみ既定ポリシーを上書きする |
| 既定ポリシー | `GovernedAgentBuilder` の `policy` 引数。overrides 未掲載の全エージェントに適用される（必須） |
| 公開窓口 | `oai_agentspec.runtime.governance` の `__init__.py`。governance シンボルの唯一の公開 import 経路 |
| extra 未導入耐性 | 該当 extra（ここでは governance）を install していなくても import 文自体は失敗しない性質 |
| 未適用キー確認用プロパティ | builder が保持する「overrides のうち一度も build で適用されていないキー集合」の読み出し口。全エージェント build 後に空でなければ typo の疑いを検知できる（名称は設計で確定） |
