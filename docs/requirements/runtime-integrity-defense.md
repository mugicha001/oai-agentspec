# runtime インテグリティ防御（lockdown 統合 API）

## 1. 概要

oai-agentspec の稼働中に発生し得るプロンプトテンプレート改竄・AgentRegistry / WorkflowGraph の動的差し替え・利用者コード（ディスク上の `.py` / リソースファイル）・配布物の事後改竄を、コア層 `integrity` モジュールが公開する `lockdown` 1 関数で一気通貫に検知・遮断する。`lockdown(root, store, registry, workflow, *, libs, checks)` は root verify + store verify+preload + libs detect + custom checks + registry/workflow freeze を 6 段順次・fail-closed で実行し、最初の違反で停止する。検証系（verify / detect / checks）は毎回再実行・状態遷移系（freeze）は冪等であり、利用者はヘルスチェック等から同関数を再発火することで擬似継続監視を構成できる。本要件は稼働中のコード／設定／宣言グラフの事後改竄に対する runtime 側防御に純化し、install 時の hash 検証・PEP 740 attestation 検証・リリース側の署名発行はスコープ外とする。

## 2. 機能要件

### FR-1: lockdown 関数による 6 段順次処理
- ユーザーストーリー: oai-agentspec の利用者（runtime 起動側）として、`lockdown(Path("src"))` 1 行で root 配下の sha256 検証 + `sys.modules` 配下配布物の RECORD 照合を済ませ、store / registry / workflow を渡せばさらに store verify + preload + freeze まで一気通貫で実行したい。なぜなら「全部守る or 何もしない」の二択 API で、利用者に守る対象を組み立てさせず、誤った組み合わせによる検証漏れを根本回避できるからだ。
- 受け入れ基準:
  - [ ] WHEN `lockdown(root, store=None, registry=None, workflow=None, *, libs=True, checks=None)` シグネチャが公開される THEN コア層 `src/oai_agentspec/integrity.py` から `__all__` 経由で `oai_agentspec.lockdown` として import 可能である
  - [ ] WHEN `lockdown` が呼ばれる THEN 以下の 6 段を順次実行する: (1) root verify、(2) store verify + preload（`store is not None` のとき）、(3) libs detect（`libs=True` のとき）、(4) custom checks（`checks is not None` のとき・順次発火）、(5) registry freeze（`registry is not None` のとき）、(6) workflow freeze（`workflow is not None` のとき）
  - [ ] WHEN 段 1 の root verify が `<root>/.integrity/sha256.manifest`（固定規約）と root 配下を sha256 照合する THEN 不一致 / manifest 不在 / 特殊ファイル（FIFO / デバイス / ソケット等）検出時に `IntegrityError` を raise し、以降の段（2〜6）はすべてスキップする（fail-closed）
  - [ ] WHEN 段 2 の store verify が `<store.root>/.integrity/sha256.manifest` と照合する THEN store.root が root 配下にあっても二重検証を行う。manifest 不一致時は `PromptTemplateIntegrityError` を raise し、以降の段（3〜6）はすべてスキップする
  - [ ] WHEN 段 2 の preload が manifest 記載の全テンプレを eager-load する THEN PromptStore の内部 `_cache` に格納し、以降の `PromptStore.get` / `compose` は cache のみを参照して disk アクセスを行わない
  - [ ] WHEN 段 3 の libs detect が `sys.modules` 全件 → `importlib.metadata.packages_distributions()` で配布物名にマップする THEN 各配布物の PEP 376 RECORD を site-packages 上の実ファイルと照合し、不一致 / md5・sha1 拒否 / その他サポート対象外アルゴリズム / RECORD 欠落 / 配布物未発見 時に `IntegrityError` を raise する
  - [ ] WHEN 段 4 の custom checks が `checks` リストを順次発火する THEN 最初の違反（`IntegrityError` 系の raise）で打ち切り、残りの check は実行しない（fail-closed）
  - [ ] WHEN 段 5 の registry freeze が `registry.freeze()` を呼ぶ THEN 状態遷移のみを行う（raise なし）。2 回目以降は冪等 no-op
  - [ ] WHEN 段 6 の workflow freeze が `workflow.freeze()` を呼ぶ THEN 状態遷移のみを行う（raise なし）。2 回目以降は冪等 no-op
  - [ ] WHEN 同じ引数で `lockdown` を再呼び出しする THEN 段 1〜4（検証系）は毎回再実行、段 5〜6（freeze）は冪等 no-op となり、ヘルスチェック再発火による擬似継続監視が成立する
  - [ ] IF `store=None` THEN 段 2 をスキップし、PromptStore の挙動は既存の lazy load 互換に保たれる
  - [ ] IF `libs=False` THEN 段 3 をスキップする（テストランタイム等の escape hatch）
  - [ ] IF `checks=None` または空リスト THEN 段 4 をスキップする
  - [ ] IF `registry=None` / `workflow=None` THEN それぞれ段 5 / 段 6 をスキップする

### FR-2: AgentRegistry.freeze() 公開クラスメソッド
- ユーザーストーリー: oai-agentspec の利用者（runtime 起動側）として、`registry.freeze()` を直接呼び `lockdown` を使わずレジストリのみ凍結したい。なぜならテストで freeze 状態を作る用途や、integrity 検証を外部で行いつつ宣言グラフだけ凍結したいケースに対応できるからだ。
- 受け入れ基準:
  - [ ] WHEN `AgentRegistry.freeze()` が引数なし公開クラスメソッドとして定義される THEN `__all__` 非掲載だが公開契約として安定（SemVer 対象）であり、`registry.freeze()` で直接呼べる
  - [ ] WHEN `freeze()` が呼ばれる THEN 以降の公開 API（`register` / `register_factory` / `update` / `unregister`）の呼び出しは `RegistryFrozenError` を raise する
  - [ ] WHEN `freeze()` 後にレジストリ状態を変更しうる全経路（公開 API に加え、内部経路 `_update_handoffs` および `HandoffGraph.apply` 経由）が呼ばれる THEN `RegistryFrozenError` を raise する。`HandoffGraph.apply` は唯一 `_update_handoffs` を経由するため、`_update_handoffs` 1 箇所のガードで全経路が塞がる
  - [ ] WHEN `freeze()` 後に `validate()` / `get` / `entry_name` 等の read-only API を実行する THEN 既存挙動と同一に成功する
  - [ ] WHEN `RegistryFrozenError` が定義される THEN `RuntimeError` を継承する（freeze 違反はファイル整合性違反とは別カテゴリのため `IntegrityError` は継承しない）。例外メッセージは違反操作名を含む
  - [ ] IF `freeze()` が複数回呼ばれる THEN 二回目以降は no-op として成功する（冪等）
  - [ ] WHEN `clone()` が呼ばれる THEN 戻り値の registry は freeze 状態を引き継がず、独立した unfrozen registry として返る（既存 `clone()` 実装で自然に成立）
  - [ ] IF 利用者が `freeze()` を一切呼ばない THEN 既存の登録 / 差し替え / 検証 API は完全互換で動作する

### FR-3: WorkflowGraph.freeze() 公開クラスメソッド
- ユーザーストーリー: oai-agentspec の利用者（runtime 起動側）として、`workflow.freeze()` を直接呼び `lockdown` を使わずワークフローのみ凍結したい。なぜなら宣言完了後のグラフ不変性を構造的に保証でき、稼働中にノード／エッジが動的に追加・書き換えされる経路を遮断できるからだ。
- 受け入れ基準:
  - [ ] WHEN `WorkflowGraph.freeze()` が引数なし公開クラスメソッドとして定義される THEN `__all__` 非掲載だが公開契約として安定（SemVer 対象）であり、`workflow.freeze()` で直接呼べる
  - [ ] WHEN `freeze()` が呼ばれる THEN 以降の `add_agent_node` / `add_function_node` / `add_edge` / `add_conditional_edges` / `add_fan_in_edge` の呼び出しは `WorkflowFrozenError` を raise する
  - [ ] WHEN `freeze()` 後に `validate()` / `mermaid()` / `_interpret()` / `as_agent_spec()` / `as_facade_spec()` / `connect_as_facade()` 等の read-only API を実行する THEN 既存挙動と同一に成功する
  - [ ] WHEN `WorkflowFrozenError` が定義される THEN `RuntimeError` を継承する（`IntegrityError` 系統と分離）。例外メッセージは違反操作名を含む
  - [ ] WHEN `_frozen` 状態が dataclass `field(default=False, init=False, compare=False, repr=False)` で表現される THEN 既存 `HandoffEdge._applied_srcs` パターンを踏襲し、等価性 / repr / pickle / diff に影響しない
  - [ ] IF `freeze()` が複数回呼ばれる THEN 二回目以降は no-op として成功する（冪等）
  - [ ] IF 利用者が `freeze()` を一切呼ばない THEN 既存のノード／エッジ追加 API は完全互換で動作する

### FR-4: 公開シンボル契約と libs auto-detection の正確性
- ユーザーストーリー: oai-agentspec の利用者として、公開シンボル 6 つ（`lockdown` / `IntegrityCheck` / `IntegrityError` / `PromptTemplateIntegrityError` / `RegistryFrozenError` / `WorkflowFrozenError`）の継承関係と libs auto-detection の検証規則を予測可能な形で規定したい。なぜなら例外捕捉の方針（`except IntegrityError` で freeze 違反を握り潰さない等）や、テストランタイムでの escape hatch 利用判断を明確にできるからだ。
- 受け入れ基準:
  - [ ] WHEN `__all__` に 6 シンボルが純増する THEN `lockdown` / `IntegrityCheck` / `IntegrityError` / `PromptTemplateIntegrityError` / `RegistryFrozenError` / `WorkflowFrozenError` の 6 つのみが追加され、既存 `__all__` メンバ集合は不変である
  - [ ] WHEN `IntegrityCheck` 型エイリアスが定義される THEN `Callable[[], None]` として公開され、違反時に `IntegrityError`（または継承例外）を raise する規約として docstring に明示される
  - [ ] WHEN 例外継承関係が定義される THEN `IntegrityError`(`Exception`) ← `PromptTemplateIntegrityError`、および `RegistryFrozenError`(`RuntimeError`) / `WorkflowFrozenError`(`RuntimeError`) の 2 系統に分離される
  - [ ] WHEN 各例外が raise される THEN メッセージに検知対象の識別子（テンプレート相対パス / 違反操作名 / 配布物名 / 不一致ファイルパス / 検出された hash アルゴリズム名）を含み、原因特定が可能である
  - [ ] WHEN `libs=True` で libs detect 段が走る THEN `sys.modules` 全件を `importlib.metadata.packages_distributions()` で配布物名にマップし、pytest / pip / oai-agentspec 自身を含むすべての import 済み配布物を検証対象にする（oai-agentspec 自身に special case を設けない）
  - [ ] WHEN PEP 376 RECORD エントリの hash フィールド `<alg>=<value>` がパースされる THEN `alg` が `hashlib.algorithms_guaranteed` に含まれ、かつ md5 / sha1 のいずれでもない場合のみ検証する。それ以外（md5 / sha1 / サポート対象外）は `IntegrityError` を raise する
  - [ ] WHEN RECORD エントリの hash フィールドが空である THEN そのエントリは検証対象外として skip する（PEP 376 で RECORD 自身などが空 hash となるケースに合わせる）
  - [ ] WHEN 対象配布物に RECORD（または `files`）が存在しない、または対象 distribution が `importlib.metadata.distribution(name)` で見つからない THEN `IntegrityError` を raise する
  - [ ] WHEN root 配下にシンボリックリンクが存在する THEN target ファイルの hash を計算して照合する（root verify / store verify 共通）
  - [ ] WHEN root 配下に通常ファイル・シンボリックリンク以外のエントリ（FIFO / デバイスファイル / ソケット等）が存在する THEN root verify は `IntegrityError`、store verify は `PromptTemplateIntegrityError` を raise する
  - [ ] WHEN manifest / RECORD に記載されているがファイルが存在しない、または対象 root に未記載ファイルが存在する THEN 対応する `IntegrityError` 系例外を raise する
  - [ ] IF 利用者が独自の検知関数を書く THEN `Callable[[], None]` シグネチャを満たし違反時に `IntegrityError`（または継承例外）を raise する規約に従えば、`lockdown(checks=[...])` の escape hatch として組み込める

## 3. 非機能要件

### NFR-1: セキュリティ（fail-closed）
- 要件: `lockdown` の 6 段順次処理は最初の違反で打ち切り、以降の段はスキップする（fail-closed）。テンプレ改竄検知（FR-1 段 2）・libs 配布物違反（FR-1 段 3）・custom check 違反（FR-1 段 4）・freeze 違反（FR-2 / FR-3）は専用例外を raise して処理を中断する。例外は静かに握り潰さず、呼び出し側で捕捉されない限り runtime を停止させる。検知後に同一インスタンス / 同一プロセスから対象操作が成功してはならない。
- 計測基準: `IntegrityError` / `PromptTemplateIntegrityError` / `RegistryFrozenError` / `WorkflowFrozenError` の raise 条件と、6 段順次処理が最初の違反で打ち切られる挙動が単体テストで網羅されること。改竄検知後にテンプレ取得・spec 変更・ノード追加が成功しないこと、および対象ファイルが部分 cache されないことをテストで確認する。

### NFR-2: 互換性（opt-in）
- 要件: `lockdown` を呼ばない既定状態では、`PromptStore` / `AgentRegistry` / `WorkflowGraph` の挙動を一切変更しない。`PromptStore.__init__` シグネチャは完全不変であり（`integrity_checks` 等の引数追加なし）、既存テストは無修正で緑のまま維持される。既存の公開 API シグネチャ・`__all__` 既存メンバ集合への破壊的変更を行わない。新規シンボル追加は `__all__` の純増 6 のみとする。
- 計測基準: 既存テストスイート（カバレッジ 80% 以上）が無修正で緑のまま維持されること。新規追加シンボル 6 以外は `__all__` のメンバ集合が不変であることをスモークテストで確認する。`PromptStore` を従来通り構築し `lockdown` を呼ばないケースのテンプレ参照挙動が既存と一致することをテストで確認する。

### NFR-3: 性能
- 要件: hash 計算と disk 参照は `lockdown` および `PromptStore._verify_integrity` / `_preload`（非公開メソッド・`lockdown` 内部からのみ呼ばれる）の発火境界にのみ閉じ、テンプレート取得・Agent 構築・ワークフロー実行のホットパスに hash 計算を持ち込まない。整合検証 1 回あたりの計算量は対象ファイル数 N に対して線形（O(N)）に留める。`PromptStore.get` / `compose` の disk 不参照化は `lockdown(store=...)` を呼んだケースに限定される（preload 段で `_cache` 充填が走る）。
- 計測基準: 本要件は性能ベンチマークを採らず、構造的計測（コードレビュー）で担保する。ホットパスを以下のとおり具体的に定義し、これらの本体経路上に `hashlib` 呼び出し・manifest / RECORD 読み込み・テンプレ disk 読み込みを含めないことをレビューで確認する: `PromptStore.get` / `PromptStore.compose` / `AgentRegistry.get` / `WorkflowGraph._interpret`。hash 計算と disk 参照は `lockdown` および非公開 helper（`PromptStore._verify_integrity` / `_preload`・`integrity` 内の `_prompt_manifest_check` / `_distribution_check` / `_path_manifest_check`）の本体にのみ閉じることをコードレベルで確認する。

### NFR-4: 保守性（SDK 隔離・構造）
- 要件: `integrity` モジュール追加によって `from agents` / `from openai` の import を `_adapters/` 配下以外に持ち込まない。単方向依存（`__init__` → `{registry, handoffs, prompts, workflow, integrity}` → `{protocols, _adapters}` → `spec`）を維持し、`runtime/` からコアへの参照のみとする。`integrity.py` は標準 lib（`hashlib` / `importlib.metadata` / `pathlib` / `sys` / `typing.Callable`）のみ依存とし、`_adapters` 経由を必要としないコア層最下層に配置する。`integrity` は他コアモジュールに依存しない（`prompts` / `registry` / `workflow` の公開シンボルは `TYPE_CHECKING` ブロック内 import で型ヒントのみ参照し、実装本体では duck typing で `_preload` / `freeze` を呼ぶ）。
- 計測基準: `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` が空であること。追加コードのテストカバレッジを含めプロジェクト全体で 80% 以上を維持すること。ruff lint / format が緑であること。

### NFR-5: 計測可能性・エラーモデル
- 要件: 改竄検知時のエラー型を専用例外として明示し、継承関係を以下のとおり規定する。
  - `IntegrityError` — ファイル整合性違反の基底例外（`Exception` を継承）。`lockdown` の root verify / libs detect / custom check で raise。
  - `PromptTemplateIntegrityError` — `IntegrityError` を継承。`lockdown` の store verify 段で raise。
  - `RegistryFrozenError` — `RuntimeError` を継承（`IntegrityError` 系統と分離）。`AgentRegistry.freeze()` 後の書換違反で raise。
  - `WorkflowFrozenError` — `RuntimeError` を継承（`IntegrityError` 系統と分離）。`WorkflowGraph.freeze()` 後の書換違反で raise。

  例外メッセージには検知対象の識別子（テンプレート相対パス／違反操作名／配布物名／不一致ファイルパス／検出された hash アルゴリズム名）を含め、原因の特定を可能にする。

  さらに `lockdown` は標準 `logging` モジュール経由で構造化イベントを emit する（logger 名は `oai_agentspec.integrity` 固定）。各段の `lockdown.start` / `lockdown.stage`（status=start/success）/ `lockdown.violation`（status=raise 直前）/ `lockdown.complete` を `extra` フィールド付きで出力し、OpenTelemetry / Datadog / Splunk / Microsoft Agent Governance Toolkit 等の observability 基盤に標準 logging 経由で接続できるようにする。lib 側は logging 統一とし、追加の外部依存（OpenTelemetry SDK 等）を持たない。
- 計測基準: 例外型が公開 `__all__` に追加され、それぞれの継承関係・raise 条件・メッセージ内容（識別子を含むこと）が単体テストで検証されていること。さらに `oai_agentspec.integrity` logger に対する単体テストで `lockdown.start` / `lockdown.stage` / `lockdown.violation` / `lockdown.complete` イベントが期待される `extra` フィールド付きで emit されることを検証する。`lockdown.violation` は対応する例外 raise の直前に必ず emit され、fail-closed 意味論を遅延させないことをテストで確認する。

## 4. 制約事項

- 技術的制約:
  - install 時防御（`uv.lock` 運用ガイダンス・`pip install --require-hashes`・PEP 740 attestation 検証手順）は本要件のスコープ外。本要件は稼働中アプリへの攻撃モデルに純化する。`docs/security-scanning.md` 側のローカル SCA（Trivy）/ リリース判定基準で対応する。
  - `sys.modules` ロック・`meta_path` 経由の import フック・`importlib.reload` 禁止フックは本要件のスコープ外。Python では `ctypes` / `__builtins__` 経由で迂回可能であり、見せかけの防御を導入しない。
  - 既に import 済みの code object に対する in-memory 改竄（`ctypes` / `gc.get_referents` 等）は Python 原理的に防げないためスコープ外。OS / プロセス分離 / 不変ファイルシステム / FIM の責務とする。
  - private 属性への直接書き換え（例: `registry._specs[x] = y` / `graph._nodes[x] = y` / `_frozen = False` 書換）および `object.__setattr__` 経由の迂回は Python 言語特性として防御不可能であり、本要件のスコープ外。freeze は公開 API および lib 内部の状態変更経路を遮断する範囲に限定する。
  - monkey-patch（lib のメソッドや module 属性の上書き・`__class__` 差し替え等）は Python 言語特性として防御不可能であり、本要件のスコープ外。
  - 継続監視（continuous monitoring）そのものは本要件のスコープ外。検知発火は明示呼び出し（`lockdown` の呼び出し）または freeze ゲート発火の瞬間に限る。利用者は健康チェックエンドポイントや定期タスクから `lockdown` を再発火することで擬似的な継続検証を構成可能（docs ガイダンスとして明示）。
  - **アプリ全体保護は Python の言語特性により原理的に不可能**。本機能はディスク上の事後改竄と公開 API / lib 内部の状態変更経路を網羅する範囲に限定する。
  - manifest ファイル自体および PEP 376 RECORD の真正性は本要件のスコープ外。利用者は manifest / RECORD ファイルを OS 権限（読み取り専用パーミッション）・読み取り専用ファイルシステム・別配置（不変ストレージ）・配布物への封入等で保護する前提とする。攻撃者が manifest／RECORD と対象ファイルの双方を同時に書き換え可能な環境では本機能の検知保証は失効する。
  - 並行性（スレッドセーフ性）は本要件のスコープ外。`AgentRegistry` / `PromptStore` / `WorkflowGraph` は単一スレッド / 単一イベントループ前提（既存 docstring の方針を踏襲）とし、複数スレッドからの同時アクセス時の 6 段順次処理発火順序・eager-load・freeze 状態遷移・例外 raise の振る舞いは規定しない。並行制御は利用者責任とする。
  - リリース成果物の署名発行側（sigstore / SLSA provenance 発行・PyPI Trusted Publishing の発行側設定）はスコープ外。
  - モデル / 学習データポイズニング（OWASP LLM04 の中核）は本ライブラリが学習を持たないためレイヤ外とする。
  - 既存の SDK 隔離（`from agents` / `from openai` は `_adapters/` のみ）・単方向依存・`__all__` 既存メンバの不変性を破壊しない。新規シンボル追加は `__all__` の純増 6 のみとする。
  - manifest フォーマットは GNU coreutils `sha256sum` 互換（`<sha256>  <relative-path>`）を採用し、独自バイナリ形式を導入しない。外部ツール（`sha256sum` 等）で manifest 生成可能であること。manifest の既定パスは `<root>/.integrity/sha256.manifest`（固定規約）とする。
  - libs detect 段の hash アルゴリズム選択は PEP 376 RECORD 記載に従う。サポート対象は Python 標準 `hashlib.algorithms_guaranteed` に含まれ、かつ md5 / sha1 を除いたもののみで、暗号学的に弱い md5 / sha1 は明示的に拒否する。
  - **`libs=True` のスコープ**: `sys.modules` 全件を `importlib.metadata.packages_distributions()` でマップし、pytest / pip / oai-agentspec 自身を含むすべての import 済み配布物を検証対象にする。oai-agentspec 自身に special case を設けない。テストランタイム等で邪魔な場合は `libs=False` の escape hatch を使う。
  - **3 ファクトリ helper（`_prompt_manifest_check` / `_distribution_check` / `_path_manifest_check`）は `integrity.py` 内の非公開実装**であり、`__all__` に掲載しない。利用者は `lockdown(checks=[...])` の escape hatch を通じて独自検知関数のみを供給する設計とし、ファクトリの誤った組み立てによる検証漏れを防ぐ。
  - `mypy` 等の型チェッカ導入はスコープ外（プロジェクト方針に従う）。`IntegrityCheck` 型エイリアスは `typing` 標準機能（`Callable`）の薄いエイリアスとして提供する。
- ビジネス制約:
  - 既存利用者の運用を破壊しないため、`lockdown` は opt-in（利用者が明示的に呼ぶ場合のみ動作）とする。
  - 本ライブラリにはプロンプト文字列を同梱しないため、PromptStore manifest は利用者側 root 配下に併置される前提とする。
  - 「利用者側コードの integrity が最重要」という前提に立ち、ライブラリ自身の self-check 専用 API は提供しない。`libs=True` で `sys.modules` 経由の自動検知に統合し、利用者コードとライブラリの双方をカバーする。
  - 検知関数のシグネチャ規約（`Callable[[], None]` + 違反時に `IntegrityError` 系を raise）は公開契約とし、`lockdown(checks=[...])` で利用者の独自検知関数を順次発火させる escape hatch を提供する。

## 5. 影響範囲

- 関連コンポーネント:
  - 新規 `src/oai_agentspec/integrity.py`（コア層最下層・単方向依存に従う・標準 lib のみ依存）:
    - 公開関数 `lockdown(root, store=None, registry=None, workflow=None, *, libs=True, checks=None) -> None`
    - 公開型エイリアス `IntegrityCheck = Callable[[], None]`
    - 公開例外 `IntegrityError`（`Exception` 継承・基底）
    - 公開例外 `PromptTemplateIntegrityError`（`IntegrityError` 継承）
    - 非公開ファクトリ helper（モジュール内 `_` プレフィクス）: `_prompt_manifest_check` / `_distribution_check` / `_path_manifest_check`
    - 非公開低レベル helper: `sha256sum` 互換 manifest パーサ / hash 計算 / シンボリックリンク target 解決 / 特殊ファイル（FIFO・デバイス・ソケット）検出 / RECORD `<alg>=<value>` パーサ / `sys.modules` ベース配布物検知（`importlib.metadata.packages_distributions()`）
  - `src/oai_agentspec/prompts.py`:
    - `PromptStore.__init__` シグネチャは完全不変（`integrity_checks` 等の引数追加なし）
    - 非公開メソッド追加: `_verify_integrity(checks: list[IntegrityCheck]) -> None`（checks 順次発火 fail-closed）/ `_preload() -> None`（manifest 由来テンプレを eager-load し `_cache` 充填し以降 disk 不参照化）
    - 公開メソッド（`get` / `compose` / `reload` / `render`）は完全互換（既存挙動・既存シグネチャ不変）
  - `src/oai_agentspec/registry.py`:
    - `_frozen: bool` を `__init__` で False 初期化
    - 公開クラスメソッド `freeze() -> None`（引数なし・冪等・状態遷移のみ）
    - 公開例外 `RegistryFrozenError`（`RuntimeError` 継承）定義
    - ガード配置: `register` / `register_factory` / `update` / `unregister` / `_update_handoffs`（`HandoffGraph.apply` 経由は `_update_handoffs` 1 箇所で塞がる）
    - `clone()` は既存実装で freeze 状態を自然に引き継がない（追加コード不要）
    - read-only API（`get` / `validate` / `entry_name` 等）は不変
  - `src/oai_agentspec/workflow/graph.py`:
    - `_frozen: bool = field(default=False, init=False, compare=False, repr=False)`（既存 `HandoffEdge._applied_srcs` パターン踏襲）
    - 公開クラスメソッド `freeze() -> None`（引数なし・冪等）
    - 公開例外 `WorkflowFrozenError`（`RuntimeError` 継承）定義
    - ガード配置: `add_agent_node` / `add_function_node` / `add_edge` / `add_conditional_edges` / `add_fan_in_edge`
    - read-only（`validate` / `mermaid` / `_interpret` / `as_agent_spec` / `as_facade_spec` / `connect_as_facade`）は不変
  - `src/oai_agentspec/__init__.py`:
    - `__all__` に純増 6 シンボル: `lockdown`, `IntegrityCheck`, `IntegrityError`, `PromptTemplateIntegrityError`, `RegistryFrozenError`, `WorkflowFrozenError`
    - `AgentRegistry.freeze()` / `WorkflowGraph.freeze()` はクラス経由公開メソッド（`__all__` 非掲載・SemVer 対象）
  - 新規 `docs/integrity.md` — 「lockdown 1 関数集約」設計の SoT。冒頭 5 行起動コード → 守れる範囲・守れない範囲の表 → `lockdown` シグネチャと 6 段順次処理 → 例外階層 → 典型構成 → manifest 信頼境界 → シンボリックリンク / 特殊ファイル方針 → ホットパス定義 → 冪等性ルール → Out of Scope を集約。install 時防御・in-memory 改竄・private 属性直接書換・monkey-patch・継続監視・アプリ全体保護不可・`libs=True` のスコープと escape hatch を Out of Scope セクションに現在仕様として記述する
  - 既存 `docs/security-scanning.md` — `docs/integrity.md` への cross-link を冒頭に追加。脅威領域の住み分け（ローカル SAST/SCA/Secrets vs runtime / 稼働中改竄）を明記
  - `docs/architecture.md` — 3 箇所の純増のみ: (1)「## ランタイム差し替え」直後に新規章「## runtime インテグリティ防御」を簡潔に追加（`integrity` モジュール導入 + `lockdown` 1 関数公開 + freeze はクラスメソッド + 詳細リンク）/ (2)「## コンポーネントの責務」表に `integrity.py` 行追加 / (3)「## 公開 API」`__all__` ブロックに 6 シンボル追記・公開 API 表に新規行追加。「## プロンプト合成」章は完全無変更
  - `docs/rationale/` への分離記録は行わない。install 時防御をスコープ外にした判断・private 属性／monkey-patch を防御不可能と整理した判断・「全部守る or 何もしない」二択 API として `lockdown` 1 関数に集約した判断などは `docs/integrity.md` 内の Out of Scope セクションおよび典型構成セクションで現在仕様として記述する
- 既存機能への影響:
  - opt-in 設計のため、`lockdown` を呼ばず `freeze()` も呼ばない既存利用者の挙動は不変。
  - 新規シンボル追加（公開関数 1 / 例外型 4 / 型エイリアス 1 = 純増 6）+ 既存クラスへのクラスメソッド追加（`AgentRegistry.freeze` / `WorkflowGraph.freeze`）のみで、既存公開 API シグネチャ・`__all__` 既存メンバには変更を加えない。
  - `PromptStore.__init__` シグネチャ完全不変のため、`PromptStore(root, layout)` で構築した既存利用者の `get` / `compose` 本体経路は従来の lazy load と一致する。`lockdown(store=...)` を呼んだ場合のみ eager-load 済み cache のみを参照する（disk アクセスが消えるが戻り値の意味論は変わらない）。
  - `AgentRegistry.clone()` の戻り値は引き続き独立した unfrozen registry であり、振る舞いは既存と互換。
  - ホットパス（`PromptStore.get` / `PromptStore.compose` / `AgentRegistry.get` / `WorkflowGraph._interpret`）の本体経路に hash 計算・disk 読み込みを追加しないため、テンプレ参照・Agent 構築・ワークフロー実行の性能特性は不変または改善（`lockdown(store=...)` 採用時）となる。

## 6. 用語定義

| 用語 | 定義 |
|------|------|
| integrity | 改竄されていないこと（同一性 / 完全性）。本要件ではテンプレ・spec 登録状態・ワークフロー宣言・利用者コード・配布物のディスク上同一性検証を指す。 |
| lockdown | runtime インテグリティ防御の起動関数。`lockdown(root, store, registry, workflow, *, libs, checks)` が root verify + store verify+preload + libs detect + custom checks + registry/workflow freeze を 6 段順次・fail-closed で実行する。本要件における「全部守る or 何もしない」二択 API の公開窓口。 |
| 6 段順次処理 | `lockdown` 内部の 6 段（root verify → store verify+preload → libs detect → custom checks → registry freeze → workflow freeze）。最初の違反で打ち切り、以降の段はスキップする。 |
| fail-closed | 検証失敗時に処理を継続せず停止する設計方針。本要件では専用例外を raise して 6 段順次処理を最初の違反で打ち切る。 |
| 固定（freeze） | AgentRegistry / WorkflowGraph を不変状態に遷移させる構造防御。以降の登録・更新・削除・内部ハンドオフ書き換え・ノード／エッジ追加を禁止し、参照・構築・read-only API のみを許可する。状態遷移のみを行い検知関数は伴わない。`lockdown` の段 5 / 段 6 でも発火するが、`registry.freeze()` / `workflow.freeze()` 単独呼び出しも公開契約として安定。 |
| ガードレール（IntegrityCheck） | 利用者が `lockdown(checks=[...])` 経由で渡す独自検知関数群。`IntegrityCheck = Callable[[], None]` シグネチャを満たし、違反時に `IntegrityError`（または継承例外）を raise する。`lockdown` の段 4 で順次発火（fail-closed・最初の違反で打ち切り）。 |
| IntegrityCheck（型エイリアス） | `Callable[[], None]` として公開される検知関数のシグネチャ規約。違反時は `IntegrityError`（または継承例外）を raise する契約を持つ。利用者は `lockdown(checks=[...])` の escape hatch でこの型を満たす関数を渡す。lib 同梱の 3 ファクトリ（`_prompt_manifest_check` / `_distribution_check` / `_path_manifest_check`）は非公開実装。 |
| eager-load + cache | `lockdown` の段 2（store verify + preload）で manifest 記載の全テンプレを一括で読み込み内部 `_cache` に格納する挙動。以降の `PromptStore.get` / `compose` は cache のみを参照し disk アクセスを行わないため、稼働中のテンプレ disk 改竄が cache に反映されない。 |
| manifest | 相対パスと sha256 を列挙したテキストファイル。本要件では `<root>/.integrity/sha256.manifest` を固定規約とし、`sha256sum` 互換フォーマット（`<sha256>  <relative-path>`）を採用する。 |
| manifest 信頼境界 | manifest ファイル（および PEP 376 RECORD）の真正性が確保される範囲。本要件は manifest が改竄されていない前提で動作し、manifest 自体の保護（OS 権限・読み取り専用 FS・配布物封入等）は利用者責任とする。攻撃者が manifest と対象ファイルの双方を同時に書き換え可能な環境では本機能の検知保証は失効する。 |
| ホットパス | runtime の通常稼働で頻繁に呼ばれる経路。本要件では `PromptStore.get` / `PromptStore.compose` / `AgentRegistry.get` / `WorkflowGraph._interpret` の本体経路を指し、ここに hash 計算・disk 読み込みを持ち込まないことを構造的計測で担保する。 |
| PEP 376 RECORD | Python パッケージ配布物に含まれるファイル一覧とその hash を記録するメタデータ。`importlib.metadata.distribution(name).files` 経由で参照可能。hash フィールドは `<alg>=<value>` 形式で、本要件では `hashlib.algorithms_guaranteed` に含まれ、かつ md5 / sha1 を除いたアルゴリズムのみサポートする。 |
| 稼働中防御 | プロセスが起動・継続している間に発生する改竄（ディスク上ファイルの事後書換・宣言データ構造の動的書換）に対する検知。install 時防御・リリース時防御とは別レイヤ。 |
| 継続監視（擬似構成） | 本要件は継続監視そのものを提供しないが、利用者がヘルスチェックエンドポイントや定期タスクから `lockdown` を同じ引数で再発火させることで構成可能な運用パターンを指す。段 1〜4（検証系）は毎回再実行、段 5〜6（freeze）は冪等 no-op となる。 |
| 冪等 | 同一引数で複数回呼んでも 1 回目と同じ結果になる性質。本要件では `registry.freeze()` / `workflow.freeze()` の状態遷移が冪等（2 回目以降 no-op）であり、`lockdown` 再呼び出し時の段 5 / 段 6 が冪等 no-op となる。 |
| opt-in | 既定では無効で、利用者が明示的に有効化したときのみ動作する設計方針。本要件のすべて（`lockdown` 呼び出し・`freeze()` 呼び出し）は opt-in とする。 |
| supply-chain（供給網） | ソース・ビルド・配布・導入・稼働の各経路を含む配布パイプライン全体。本要件は稼働経路における改竄検知に焦点を当てる。`docs/security-scanning.md` 側のローカル SAST/SCA/Secrets が install / リリース経路を担当する。 |
