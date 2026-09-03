# runtime インテグリティ防御

`oai-agentspec` のコア層に `integrity` モジュールを置き、稼働中のディスク上ファイル改竄
（プロンプトテンプレート / 配布物 / 任意パス）と `AgentRegistry` / `WorkflowGraph` への
動的書き換えを fail-closed に検知 / 遮断する。本ドキュメントは本機能の Single Source of Truth で
ある。`docs/security-scanning.md`（ローカル SAST/SCA/シークレットスキャン）が install / リリース
時の責務を担うのに対し、本ドキュメントは**稼働中**（プロセス起動後）の改竄に対する runtime 側
防御を扱う。

## 位置付け（lockdown ≠ full agent governance）

本機能は **runtime integrity 防御の最小コア**であり、enterprise agent governance 全体ではない。
oai-agentspec 内部データ構造（`AgentRegistry` / `WorkflowGraph` / `PromptStore`）の自衛と
sha256 / PEP 376 RECORD 整合性に責務を限定する。

| レイヤ | 本機能 (`lockdown`) | 汎用 agent governance（Microsoft Agent Governance Toolkit 等） |
|---|---|---|
| 守る対象 | lib 内部の宣言データ構造 + disk 上ファイル | agent ACTION / IDENTITY / BEHAVIOR |
| 検知方式 | hash 照合 + 構造 freeze | policy engine（OPA/Cedar/YAML）+ 署名 + execution rings |
| スコープ | `AgentRegistry._specs` 動的差替・テンプレ disk 改竄等 | agent 間通信なりすまし・tool 呼出 policy 違反・OWASP Agentic Top 10 等 |
| 関係 | 内部状態 / disk 整合性レイヤ | 行動 / 通信 / アイデンティティレイヤ |

OWASP Agentic Top 10 の包括カバー、agent 間 Ed25519 署名、execution rings、microVM isolation
等が必要な場合は Microsoft Agent Governance Toolkit / Proofpoint Agent Integrity Framework /
Blaxel 等の汎用ガバナンスレイヤと**併用**する設計が望ましい。両者は補完関係にあり、本機能の
`checks` escape hatch から外部ガバナンスのポリシーを発火させる統合パターンを「典型構成」節で示す。

## 最小起動コード

`lockdown(Path("src"))` 1 行で root verify + libs auto-detect が走り、これだけで src 配下の
sha256 検証と sys.modules 上の配布物（oai-agentspec 自身を含む）の PEP 376 RECORD 照合が
完了する。

```python
from pathlib import Path
from oai_agentspec import lockdown

lockdown(Path("src"))
```

## 守れる範囲・守れない範囲

| 範囲 | 守れる | 守れない |
|---|---|---|
| ディスク上ファイルの事後改竄 | root 配下 `.py` / リソース・PromptStore root 配下テンプレ・`sys.modules` 配下配布物の RECORD 対象ファイル・利用者が `checks` で渡す任意パス | manifest / RECORD ファイル自体の改竄（信頼境界・OS 権限 / RO FS で保護する前提） |
| 宣言データ構造の動的書換 | `AgentRegistry.register` / `register_factory` / `update` / `unregister` / `_update_handoffs` / `HandoffGraph.apply` 経由の書換、`WorkflowGraph.add_agent_node` / `add_function_node` / `add_edge` / `add_conditional_edges` / `add_fan_in_edge` 経由の書換、freeze 後の `WorkflowGraph` public field 経由の直接 dict mutation（`wf.nodes[x] = y` / `wf.edges[x].append(...)` 等は `MappingProxyType` + 値 tuple 化で遮断）、`register` 済 `AgentSpec` オブジェクトの外部書き換え（`spec.instructions = ...` / `spec.handoffs.append(...)` 等は freeze 時 snapshot により build 結果へ非伝播） | private 属性への直接書換（`registry._specs[x] = y` 等の内部 dict 直書き・`graph._frozen = False` / `registry._frozen = False` の freeze フラグ直書き）・`object.__setattr__` 経由の迂回 |
| import 経路 / メソッド差し替え | （スコープ外） | monkey-patch（lib のメソッド / module 属性の上書き・`__class__` 差し替え）・`sys.modules` 経由の偽 module 差し替え・`meta_path` / `importlib.reload` 経由の迂回 |
| in-memory のオブジェクト改竄 | （スコープ外） | 既に import 済み code object に対する `ctypes` / `gc.get_referents` 経由の改竄 |
| install / リリース時の供給網 | （スコープ外） | `uv.lock` / `pip install --require-hashes` / PEP 740 attestation / 署名発行（`docs/security-scanning.md` 側の責務） |
| 継続監視 | 利用者がヘルスチェック等から `lockdown` を再発火することで擬似構成可能（freeze は冪等 no-op・verify は毎回再実行） | OS / FIM レベルの自動継続監視 |
| データ源（取得系）の改竄 | 取得を `function_tool` として宣言した静的資産（ドキュメント本文 / スナップショット / 確定済みメモリ）の `Boundary.TOOL_OUTPUT` でのベースライン照合（本ドキュメント「データ源インテグリティ」節） | ライブレスポンス（呼び出しごとに値が変わる API / DB 応答。比較対象のベースラインが原理的に存在しない）・ツールとして宣言しない取得経路 |

**アプリ全体の防御は原理的に不可能**である。Python の言語特性（`ctypes` / monkey-patch / private
属性直書き）により、上記「守れない」列は本機能の保証範囲外となる。本機能は「ディスク上の事後改竄」と
「公開 API / lib 内部の状態変更経路」を網羅する範囲で fail-closed を成立させる。

## 公開 API

`__all__` に純増する公開シンボルは以下の 6 つである。

| シンボル | 種別 | 提供元 | 用途 |
|---|---|---|---|
| `lockdown` | 関数 | `integrity.py` | 起動時 / ヘルスチェック時に一気通貫で root verify + store verify+preload + libs detect + custom checks + registry/workflow freeze を実行 |
| `IntegrityCheck` | 型エイリアス `Callable[[], None]` | `integrity.py` | 検知関数のシグネチャ規約。違反時は `IntegrityError` 系を raise |
| `IntegrityError` | 例外（`Exception` 継承・基底） | `integrity.py` | ファイル整合性違反の基底例外 |
| `PromptTemplateIntegrityError` | 例外（`IntegrityError` 継承） | `integrity.py` | PromptStore manifest 不一致 |
| `RegistryFrozenError` | 例外（`RuntimeError` 継承） | `registry.py` | freeze 後の registry 書換違反 |
| `WorkflowFrozenError` | 例外（`RuntimeError` 継承） | `workflow/graph.py` | freeze 後の WorkflowGraph 書換違反 |

`AgentRegistry.freeze()` と `WorkflowGraph.freeze()` はクラス経由の公開メソッドであり、
`__all__` には掲載しないが**公開契約として安定（SemVer 対象）**である。`AgentRegistry` /
`WorkflowGraph` は既に公開シンボルであり、利用者は `registry.freeze()` / `workflow.freeze()` を
直接呼んでよい。

`lockdown` のシグネチャは次の通りである。

```python
def lockdown(
    root: Path,
    store: PromptStore | None = None,
    registry: AgentRegistry | None = None,
    workflow: WorkflowGraph | None = None,
    *,
    libs: bool = True,
    checks: list[IntegrityCheck] | None = None,
) -> None
```

## 6 段順次処理

`lockdown` は以下の 6 段を**順次・fail-closed**に実行する。最初の違反が発生した時点で残りの
段はスキップされ、対応する例外が呼び出し元に伝播する。

```
lockdown(root, store=, registry=, workflow=, libs=True, checks=)
  │
  ├─ 1. root verify
  │     <root>/.integrity/sha256.manifest と root 配下を sha256 照合
  │     不一致 → IntegrityError raise（以降のステップ全スキップ）
  │
  ├─ 2. store verify + preload（store is not None）
  │     <store.root>/.integrity/sha256.manifest と照合（store.root が root 配下にあっても二重検証）
  │     全テンプレを store._preload() で eager-load し _cache 充填
  │     manifest 不一致 → PromptTemplateIntegrityError raise
  │
  ├─ 3. libs detect（libs=True）
  │     sys.modules 全件 → importlib.metadata.packages_distributions() で配布物名にマップ
  │     各配布物の PEP 376 RECORD を site-packages 上の実ファイルと照合
  │     不一致 / md5・sha1 拒否 / RECORD 欠落 → IntegrityError raise
  │
  ├─ 4. custom checks（checks is not None）
  │     checks リストを順次発火（fail-closed・最初の違反で打ち切り）
  │
  ├─ 5. registry freeze（registry is not None）
  │     registry.freeze() 呼び出し（冪等）
  │
  └─ 6. workflow freeze（workflow is not None）
        workflow.freeze() 呼び出し（冪等）
```

各段の入力と raise 条件:

| 段 | 入力 | raise 条件 |
|---|---|---|
| 1. root verify | `root` | `<root>/.integrity/sha256.manifest` 不在 / 不一致 / root 配下に通常ファイル・シンボリックリンク以外の特殊ファイル存在 → `IntegrityError` |
| 2. store verify + preload | `store` | `<store.root>/.integrity/sha256.manifest` 不在 / 不一致 / 特殊ファイル存在 → `PromptTemplateIntegrityError`。違反なく完了した場合は全テンプレを eager-load して `_cache` に格納し、以降の `get` / `compose` は disk 不参照となる。preload 末尾で store は lockdown 状態に遷移し、以降 manifest 未掲載のテンプレ / セグメントを要求された場合でも disk アクセスを行わず `PromptTemplateIntegrityError` を raise する（cache only 契約の厳格化） |
| 3. libs detect | `libs=True` | `sys.modules` 配下配布物の RECORD 不一致 / 未存在 / md5・sha1 / その他サポート対象外のアルゴリズム / 配布物未発見 → `IntegrityError` |
| 4. custom checks | `checks` | 各 check 関数が `IntegrityError` 系を raise |
| 5. registry freeze | `registry` | （raise なし・状態遷移のみ）。冪等 no-op |
| 6. workflow freeze | `workflow` | （raise なし・状態遷移のみ）。冪等 no-op |

## 例外階層

```
Exception
├── IntegrityError              （基底）
│   └── PromptTemplateIntegrityError
│
RuntimeError
├── RegistryFrozenError
└── WorkflowFrozenError
```

二系統に分離するのは、freeze 違反が「ファイル整合性違反」ではなく「プログラム上の不変条件違反」
だからである。利用者の `except IntegrityError` で freeze 違反を握り潰さないよう、`RegistryFrozenError`
/ `WorkflowFrozenError` は `RuntimeError` 系統に置く。

例外メッセージは検知対象の識別子（テンプレート相対パス / 違反操作名 / 配布物名 / 不一致ファイル
パス / 検出された hash アルゴリズム名）を含め、原因特定を可能にする。

### 例外発生タイミング

| 例外 | 発生契機 |
|---|---|
| `IntegrityError` | `lockdown` の root verify / libs detect / custom check の不一致時。基底例外なので `except IntegrityError` で `PromptTemplateIntegrityError` も捕捉される |
| `PromptTemplateIntegrityError` | `lockdown` の store verify 段で manifest 不一致時。加えて lockdown 後（store が lockdown 状態）の `get` / `compose` で manifest 未掲載テンプレ / セグメントを要求した cache miss 時、および lockdown 後の `reload()` 呼び出し時 |
| `RegistryFrozenError` | `registry.freeze()` 後（`lockdown` で渡した場合を含む）に `register` / `register_factory` / `update` / `unregister` / `_update_handoffs` / `HandoffGraph.apply` を呼んだ時 |
| `WorkflowFrozenError` | `workflow.freeze()` 後（`lockdown` で渡した場合を含む）に `add_agent_node` / `add_function_node` / `add_edge` / `add_conditional_edges` / `add_fan_in_edge` を呼んだ時 |

## 典型構成

### 全部守る起動時バッチ

```python
from pathlib import Path
from oai_agentspec import lockdown, PromptStore, PromptLayout, AgentRegistry, WorkflowGraph

store = PromptStore("prompts", PromptLayout(base="base", parts="parts", agents="agents"))
registry = AgentRegistry()
# ... register specs ...
workflow = WorkflowGraph("main")
# ... add nodes / edges ...

lockdown(
    Path("src"),
    store=store,
    registry=registry,
    workflow=workflow,
)
# 以降 store.get / compose は cache のみ参照（disk 不参照）
# registry.register / workflow.add_* は RegistryFrozenError / WorkflowFrozenError
```

### ヘルスチェック再発火（擬似継続監視）

同じ引数で `lockdown` を再呼び出しすると、freeze 段は冪等 no-op、verify / detect / custom checks
は毎回再実行される。これによりヘルスチェックエンドポイントや定期タスクから稼働中の改竄を継続検知
可能。

```python
@app.get("/healthz/integrity")
async def integrity_check() -> dict:
    try:
        lockdown(Path("src"), store=store, registry=registry, workflow=workflow)
    except IntegrityError as exc:
        raise HTTPException(503, detail=str(exc)) from exc
    return {"ok": True}
```

### custom check escape hatch

`checks` 引数で利用者の独自検知関数を順次発火させられる。lib 同梱の 3 ファクトリ
（`_prompt_manifest_check` / `_distribution_check` / `_path_manifest_check`）は非公開実装であり、
利用者は `checks` 経由で任意の `IntegrityCheck`（`Callable[[], None]`・違反時に `IntegrityError`
系を raise）を渡す。

```python
from oai_agentspec import lockdown, IntegrityCheck, IntegrityError

def check_business_config() -> None:
    if not Path("config/business.yaml").exists():
        raise IntegrityError("business config missing")

lockdown(Path("src"), checks=[check_business_config])
```

### freeze のみ単体使用

`lockdown` を使わず `AgentRegistry.freeze()` / `WorkflowGraph.freeze()` を単独で呼ぶ用途
（テストで freeze 状態を作る、integrity 検証は外部で行うが宣言グラフだけ凍結したい等）にも
対応する。

```python
registry.freeze()
workflow.freeze()
```

#### freeze の不変性保証

`freeze()` は宣言データ構造を read-only に固定し、freeze 後の書き換えを以下の範囲で遮断する
（いずれも冪等で、同じ対象を 2 回 freeze しても 2 回目は no-op で成功する）。

- **`AgentRegistry`**: `freeze()` は freeze 時点で登録済 `AgentSpec` を独立 snapshot 化する
  （可変コンテナを含めて新インスタンスへ複製）。このため `register` に渡した spec オブジェクトの
  参照を利用者が保持し、freeze 後に `spec.instructions = ...` / `spec.handoffs.append(...)` 等で
  外部から書き換えても、registry の build 結果には伝播しない。`register` / `register_factory` /
  `update` / `unregister` / `_update_handoffs` / `HandoffGraph.apply` の公開経路は
  `RegistryFrozenError` を raise する。内部 dict 直書き（`_specs`）と `_frozen` 直書きは Python
  言語特性上スコープ外
- **`WorkflowGraph`**: `freeze()` は freeze 時点で `nodes` / `edges` / `conditional_edges` /
  `fan_in_edges` を `MappingProxyType`（`edges` の値 list は tuple 化）の read-only view に置換する。
  このため `add_agent_node` / `add_function_node` / `add_edge` / `add_conditional_edges` /
  `add_fan_in_edge` の公開経路（`WorkflowFrozenError`）に加え、`wf.nodes["evil"] = ...`（`TypeError`）/
  `wf.edges["x"].append(...)`（`AttributeError`）/ `wf.edges.clear()`（`AttributeError`）等の
  public field 経由の直接 dict mutation も遮断される。read-only API（`validate` / `mermaid` /
  `_interpret` / `as_agent_spec` / `as_facade_spec`）は freeze 後も動作する。`ConditionalEdge` /
  `FanInEdge` 内部 dataclass field の深い mutation はスコープ外
- **`PromptStore.reload()`**: store が lockdown 状態（preload 済）の場合は
  `PromptTemplateIntegrityError` を raise する。lockdown 後の継続検証は `reload` ではなく
  `lockdown` の再発火（冪等）で行う

### libs auto-detect の無効化

テストランタイムなど `sys.modules` 全件検査が邪魔な状況では `libs=False` で 3 段目をスキップする。

```python
lockdown(Path("src"), libs=False)
```

### manifest 自身の真正性検証（署名 escape hatch）

`<root>/.integrity/sha256.manifest` 自身の改竄は本機能の信頼境界外（manifest 信頼境界節を参照）
である。manifest 自身の真正性が必要な環境では、利用者が `checks` 引数で署名検証を差し込む。
代表的には Sigstore / PEP 740 attestation / 自前の Ed25519 / GPG 等を `IntegrityCheck`
（`Callable[[], None]`・違反時 `IntegrityError` 系を raise）でラップする。

```python
from pathlib import Path
from oai_agentspec import lockdown, IntegrityError

def verify_manifest_signature() -> None:
    """manifest 自身を Sigstore で検証する例（利用者実装）。"""
    if not _verify_sigstore_bundle(
        bundle=Path("src/.integrity/sha256.manifest.sigstore"),
        target=Path("src/.integrity/sha256.manifest"),
    ):
        raise IntegrityError("manifest signature invalid")

lockdown(Path("src"), checks=[verify_manifest_signature])
```

順次発火順序の都合で、署名検証は `checks` 段（4 段目）で動く。root verify（1 段目）/ store
verify（2 段目）より後に走るため、本パターンを用いる場合は manifest 改竄を OS 権限 / RO FS で
予防しつつ、署名検証を防御の最後の網として位置付ける運用を推奨する。

### Microsoft Agent Governance Toolkit との統合

汎用 agent governance（policy engine / OPA・Cedar 規則）を併用する場合、外部ポリシーを
`IntegrityCheck` でラップして `checks` に渡す。lockdown は本機能のスコープ（内部状態 / disk
整合性）を担保し、AGT は agent ACTION / IDENTITY 層を担う**補完関係**になる。

```python
from pathlib import Path
from oai_agentspec import lockdown, IntegrityError
# 仮想例: 利用者環境で AGT が import 可能であること
from agent_governance_toolkit import PolicyEngine

policy = PolicyEngine.load("policies/startup.yaml")

def agt_startup_policy() -> None:
    decision = policy.evaluate({"action": "lockdown", "root": "src"})
    if not decision.allowed:
        raise IntegrityError(f"AGT policy violation: {decision.reason}")

lockdown(Path("src"), checks=[agt_startup_policy])
```

詳細な統合パターン（runtime callback / saga orchestration / kill switch との連携）は外部
ガバナンスツール側のドキュメントに従う。lockdown 側は `checks` の fail-closed セマンティクスを
保証する範囲に責務を限定する。

## manifest 信頼境界

manifest フォーマットは GNU coreutils `sha256sum` 互換（`<sha256>  <relative-path>` 行・1 行 1 ファイル）
を採用する。独自バイナリ形式は導入せず、`sha256sum` で manifest を生成可能。

- 既定パスは `<root>/.integrity/sha256.manifest` 固定規約（root verify / store verify 共通）
- PEP 376 RECORD は `importlib.metadata.distribution(name).files` 経由で取得し、hash フィールド
  `<alg>=<value>` をパースする
- hash の形式は照合対象で異なる。`<root>/.integrity/sha256.manifest`（root verify / store verify）
  は `sha256sum` 互換の **hex** 形式。PEP 376 RECORD（libs detect）は `<alg>=<value>` の value が
  **base64-urlsafe-nopad** 形式であり、照合時の実ファイル hash も同形式で計算して突き合わせる
- サポートアルゴリズム: Python 標準 `hashlib.algorithms_guaranteed` に含まれ、かつ md5 / sha1
  以外。md5 / sha1 は明示的に拒否し `IntegrityError` を raise
- RECORD エントリの hash フィールドが空（PEP 376 で RECORD 自身などが空 hash となるケース）は
  検証対象外として skip

manifest ファイル自体および PEP 376 RECORD の真正性は本要件のスコープ外である。利用者は manifest
/ RECORD ファイルを OS 権限（読み取り専用パーミッション）・読み取り専用ファイルシステム・別配置
（不変ストレージ）・配布物への封入等で保護する前提とする。攻撃者が manifest／RECORD と対象
ファイルの双方を同時に書き換え可能な環境では本機能の検知保証は失効する。

## シンボリックリンク / 特殊ファイル

- root 配下のシンボリックリンクはリンク先（target）を解決して target ファイルの sha256 を計算
  して照合する
- 通常ファイル・シンボリックリンク以外のエントリ（FIFO / デバイスファイル / ソケット等）が root
  配下に存在すれば対応する例外（`IntegrityError` / `PromptTemplateIntegrityError`）を raise する
- manifest / RECORD に記載されているがファイルが存在しない、または対象 root に未記載ファイルが
  存在する場合も対応する例外を raise する

## ホットパス

runtime の通常稼働で頻繁に呼ばれる以下の本体経路には `hashlib` 呼び出し・manifest / RECORD
読み込み・テンプレート disk 読み込みを含めない。hash 計算と disk 参照は `lockdown` および
`PromptStore._verify_integrity` / `_preload` の発火境界にのみ閉じる。

- `PromptStore.get`
- `PromptStore.compose`
- `AgentRegistry.get`
- `WorkflowGraph._interpret`

`PromptStore.get` / `compose` の disk 不参照化は `store` を `lockdown` に渡した場合に限定される
（preload 段で `_cache` 充填が走るため）。`store` を渡さない / `lockdown` を呼ばないケースでは
既存の lazy load 挙動（`get` / `compose` 時の遅延読み込み）と完全互換である。

lockdown 後の cache only 保証は、ネスト stem 参照（`agents/billing/refund.md` を
`compose(agent="refund")` で解決する既存 API）でも成立する。preload は各テンプレを full path key
（`agent:billing/refund`）で `_cache` 充填する際、stem が一意なら stem alias key（`agent:refund`）も
登録するため、stem 参照の cache miss が disk へ落ちない（colon を含む stem は曖昧化回避のため alias
を登録しない）。

## 観測性（構造化ロギング）

`lockdown` は内部で標準 `logging` モジュールにイベントを emit する。logger 名は
`oai_agentspec.integrity` 固定で、各段の開始 / 成功 / 違反を構造化ログとして出す。利用者は
標準 `logging.config.dictConfig` や OpenTelemetry / Datadog / Splunk / Microsoft AGT 等の
observability 基盤に**そのまま接続できる**（lib 側は logging に統一し、外部依存を持たない）。

| イベント | level | 主な `extra` フィールド |
|---|---|---|
| `lockdown.start` | INFO | `root`, `libs`, `store`, `registry`, `workflow`, `checks_count` |
| `lockdown.stage` | DEBUG | `stage`（`root_verify` / `store_verify` / `libs_detect` / `custom_checks` / `registry_freeze` / `workflow_freeze`）, `status`（`start` / `success`） |
| `lockdown.violation` | WARNING | `stage`, `error_type`（例外型名）, `identifier`（テンプレート相対パス / 配布物名 / 不一致ファイルパス等） |
| `lockdown.complete` | INFO | `duration_ms` |

```python
import logging
logging.getLogger("oai_agentspec.integrity").setLevel(logging.DEBUG)
# あとは通常の logging handler / dictConfig / OpenTelemetry instrumentation でハンドリング
```

`lockdown.violation` イベントは raise 直前に必ず emit される（観測性確保のため）。fail-closed
意味論は維持され、ログ出力は呼び出し元への例外伝播を遅延させない。

## 冪等性ルール

| 段 | 性質 |
|---|---|
| root verify / store verify / libs detect / custom checks | 毎回再実行（同じ引数で `lockdown` を呼び直すと再検証される）。ヘルスチェック再発火用途と一致 |
| registry freeze / workflow freeze | 冪等。同じ registry / workflow を 2 回 lockdown しても 2 回目の freeze は no-op で成功 |

検証系（verify / detect / checks）は毎回再実行・状態遷移系（freeze）は冪等という使い分けにより、
擬似継続監視（ヘルスチェック再発火）が自然に成立する。

## データ源インテグリティ（RAG / メモリ / ツール出力）

エージェントが判断材料として読む取得系データ（RAG 検索結果・会話メモリ・静的資産を返すツール
出力）が、取得時点から改変されていないことを検知するパターンを扱う。検査は内容ガードレール層
（`oai-agentspec[guardrails]`）のツール出力 guardrail で行い、検知ロジック（ベースライン照合）は
利用者が注入する。`lockdown` とは対象も発火タイミングも異なる別機構であり、両者は併存する
（「lockdown との関係（併存）」節）。

### 適用境界と基本パターン

取得（RAG 検索 / メモリ読み出し / 静的資産の読み出し）を `function_tool` として宣言すると、
検査点は「ツールが値を返した直後・モデルが読む前」に一意に定まる。この位置が
`Boundary.TOOL_OUTPUT`（`runtime/guardrails/types.py` の `Boundary` メンバ）であり、ここへ
ベースライン照合の detector を装着するのが基本パターンである。agent 境界（`Boundary.INPUT` /
`Boundary.OUTPUT`）は会話の入出力を対象とするため、取得結果がモデルへ渡る瞬間を捉えられない。

基本パターンは「1 回のツール呼び出しが単一資産（1 ドキュメント / 1 メモリレコード）を返す形」を
指す。この形では返却テキスト全体のハッシュと資産単位のベースラインが 1 対 1 で照合できるため、
detector は「返却テキストの sha256 が既知ダイジェスト集合に含まれるか」の membership 判定で足りる
（集合は起動時に 1 回作る）。detector が受け取るのはテキストのみでツール引数（`doc_id` 等）は
渡らないため、識別子と本文の対応までは検証しない。資産が複数ある構成は「複数チャンク連結の
扱い」節の経路を使う。

### 装着経路

ツール定義時に宣言する場合は、`function_tool(_func, tool_output_guardrails=[tool_guardrail(detector,
on="output", on_trip="raise")])` のようにツール定義の引数へ渡す。`as_tool` 等 `function_tool` で
定義し直せない既存ツールへ後付けする場合は `guard_tool(tool, output_detector=detector,
on_trip="raise")` で包み、ガード版のみを `tools` へ入れる。

名前で一覧・照会したい場合は登録簿を併用する。登録は登録簿インスタンス経由の facade
`registry.tool_guardrail(detector, on="output", on_trip="raise", name=<登録名>,
severity=Severity.CRITICAL)` で行い、境界は `on` から導出される（`on="output"` で
`Boundary.TOOL_OUTPUT`）。装着は `registry.get(<登録名>)` で実体を取り出し、上記の
`function_tool(..., tool_output_guardrails=[...])` へ渡す。登録名を `AgentSpec.guardrails` へ
渡すことはできず、専用フィールド `input_guardrails` / `output_guardrails` は agent 境界専用で
ツール出力には効かない（`output_guardrails` は名称から誤解されやすい）。根拠は
`docs/architecture.md` の内容ガードレール節を参照する。

### detector の実装パターン

ベースラインと返却テキストのハッシュを突き合わせる述語を `predicate_detector` で包み、
`tool_guardrail` でツール出力 guardrail へ接着する。

```python
import hashlib

from oai_agentspec import function_tool
from oai_agentspec.runtime.guardrails import predicate_detector, tool_guardrail

# 取得時点の正解（doc_id -> sha256 hex）。取得時点に生成し読み取り専用配置へ保管する。
BASELINE: dict[str, str] = {"DOC-1042": "9f2c1e5b...", "MEM-42": "1b7e33c9..."}
KNOWN_DIGESTS = frozenset(BASELINE.values())


def _mismatches_baseline(text: str) -> bool:
    """返却テキスト全体の sha256 がベースラインに無ければ改竄とみなす（True で trip）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest() not in KNOWN_DIGESTS


integrity_detector = predicate_detector(
    _mismatches_baseline, reason="retrieved content does not match the acquisition-time baseline"
)

# _fetch_document は利用者の取得関数（RAG 検索 / メモリ読み出し / 静的資産の読み出し）。
fetch_document = function_tool(
    _fetch_document,
    name_override="fetch_document",
    tool_output_guardrails=[tool_guardrail(integrity_detector, on="output", on_trip="raise")],
)
```

`predicate_detector` は述語が True のとき trip するため、述語は「不一致 = True」で書く。向きを
反転させると不一致のときに trip せず、検知が素通りする。検知コストはテキスト長 n に対する
ハッシュ計算 1 回とベースライン照会 1 回に留まり、ネットワーク往復・モデル呼び出しを含めない。

### 複数チャンク連結の扱い

RAG 検索が複数ヒットを 1 つのテキストへ連結して返す形では、detector が受け取るのは連結後の
テキスト全体であり、資産単位のベースラインをそのまま突き合わせられない。この形では次の経路の
いずれかを使う。

**経路 (a) 連結後テキストに対するベースライン**。連結後テキストそのもののダイジェストを
ベースラインとして持つ。適用できるのは定型問い合わせ（クエリ集合が有限かつ事前に確定している
構成）に限る。固定 FAQ の定型クエリ集合に対する検索のように、複数ヒットの連結が前提でクエリが
確定している構成が該当する。成立には次が同時に必要となる。

- クエリ集合の事前確定: ベースラインは事前に用意するものであり、自由文入力の RAG はクエリ空間が
  非有界で原理的に用意できない
- 検索インデックスの不変: 文書の追加・削除・再インデックスは同一クエリでもヒット集合を変え、
  改竄でないのに全ベースラインを一斉に失効させる。`on_trip="raise"` と組み合わさると正常系が
  停止するため、ベースラインの失効と更新のコストと同じ論点として扱う
- 取得順序の決定性: 近似最近傍探索・スコア同点時のタイブレーク・シャード分割の並列取得は順序を
  揺らす

**経路 (b) doc_id とダイジェストを含む構造化出力**。ツールが資産単位の識別子と本文を含む JSON を
返し、detector がそれをパースして資産単位で照合する。

```json
{
  "query": "退職金の計算方法",
  "chunks": [
    {"doc_id": "DOC-1042", "digest": "9f2c1e5b...", "text": "退職金は基本給に..."},
    {"doc_id": "DOC-2087", "digest": "4a8d0c72...", "text": "勤続年数の算定は..."}
  ]
}
```

成立条件として、ツール本体は `json.dumps(...)` 済みの `str` を返す。`dict` を返すと detector には
Python の repr（シングルクォート）が渡り `json.loads` が失敗する。照合手順は次の通り。

1. detector が JSON をパースし `chunks` を走査する
2. 各 `doc_id` で利用者側のベースラインを引き、`text` を再ハッシュして突き合わせる。出力中の
   `digest` はツールが自称する値であり信頼の根拠にしない（再計算値と食い違えば trip 扱い）
3. ベースラインに存在しない `doc_id` が含まれていれば trip する（未知資産の混入）

経路 (b) は経路 (a) の条件を必要としないため、自由文クエリ・可変インデックスを扱う RAG の通常形
ではこちらを推奨形とする。chunk の走査は資産ごとのハッシュ計算 1 回と照会 1 回に留まる。

### 検知時の既定挙動（fail-closed）

改竄検知では `on_trip="raise"`（中断）を既定として使う。ファクトリ（`tool_guardrail` /
`guard_tool` / 登録簿の facade）の既定値は `"reject"`（注釈付き返却で続行）であるため、改竄検知
用途では明示指定が必要である。`"raise"` を指定すると trip 時に `ToolOutputGuardrailTripwireTriggered`
が送出され、当該ツール出力はモデルへ渡らないまま実行が中断する。

`"reject"` / `"allow"` は実行を中断せず会話が続くため、検知後もモデルが当該データを読み得る。
これらは fail-closed ではなく、監査ログ収集など検知結果を続行前提で扱う用途に限る。

登録簿経由で宣言する場合の推奨深刻度は `Severity.CRITICAL` である。

### lockdown との関係（併存）

| 機構 | 守る対象 | 発火タイミング |
|---|---|---|
| `lockdown` | ディスク上の静的資産と宣言グラフ（内訳は「守れる範囲・守れない範囲」節を参照） | 起動時、および利用者による明示の再発火（ヘルスチェック等） |
| ツール出力 guardrail（`Boundary.TOOL_OUTPUT`） | 1 ターンごとのツール出力テキスト | ツール実行のたび |

両者を統合せず併存とするのは層が異なるためである。`lockdown` はコア層、内容ガードレールは
opt-in extra にあり、統合すると extra 未導入環境でコア機能が壊れる。

`lockdown` の `checks` へ寄せる形も採らない。`checks` が受け取る `IntegrityCheck` は
`Callable[[], None]` で引数を取らずツール出力テキストを受け取れず、発火も `lockdown` の呼び出し時に
限られるため 1 ターンごとのツール出力に届かない（層の違いとは別の根拠）。

### 責務分界と適用条件

lib 側の責務は次に限る。

- 宣言面（境界 enum・ファクトリ・登録簿）の提供
- 本パターンと検知時の既定挙動の明文化
- example の提供

ベースラインの生成・保管・鍵管理・失効は利用者責務である。lib はベースラインの保存機構も署名
機構も持たない。

適用条件は次の通り。

- 取得時点のベースラインを保持しない場合、照合対象が存在しないため本パターンは適用できない
- RAG / メモリ取得をツールとして宣言しない構成（SDK 組み込みの検索機構をエージェントが直接使う
  等）では、検査点が定まらず本経路が成立しない
- ベースラインと対象データを双方同時に書き換えられる環境では検知保証が失効する（manifest 信頼
  境界の節と同じ前提）

実行できる最小例は
[examples/guardrails/09_data_integrity_detector.py](../examples/guardrails/09_data_integrity_detector.py)
を参照する。

## Out of Scope

以下は本機能のスコープ外であり、検知保証を提供しない。

- **install 時防御**: `uv.lock` 運用ガイダンス・`pip install --require-hashes`・PEP 740
  attestation 検証手順は本機能の範囲外。`docs/security-scanning.md` 側のローカル SCA（Trivy）/
  リリース判定基準で対応する
- **in-memory 改竄**: 既に import 済みの code object に対する `ctypes` / `gc.get_referents`
  等での改竄は Python 原理的に防げない。OS / プロセス分離 / 不変ファイルシステム / FIM の責務
- **private 属性への直接書き換え**: `registry._specs[x] = y` の内部 dict 直書き・
  `graph._frozen = False` / `registry._frozen = False` の freeze フラグ直書き・`object.__setattr__`
  経由の迂回は Python 言語特性として防御不可能。freeze は公開 API・lib 内部の状態変更経路・
  freeze 後の public field 経由の直接 dict mutation を遮断する範囲に限定する
- **monkey-patch**: lib のメソッドや module 属性の上書き・`__class__` 差し替えは Python 言語
  特性として防御不可能
- **sys.modules ロック / meta_path フック / `importlib.reload` 禁止**: `ctypes` /
  `__builtins__` 経由で迂回可能であり、見せかけの防御を導入しない
- **継続監視（continuous monitoring）そのもの**: 検知発火は明示呼び出し（`lockdown` の呼び出し）
  または freeze ゲート発火の瞬間に限る。継続監視は利用者がヘルスチェック等から `lockdown` を
  再発火することで擬似構成する
- **manifest / RECORD ファイルの真正性**: 利用者責任（OS 権限・RO FS・配布物封入で保護する前提）。
  manifest 自身の署名検証が必要な環境では、典型構成節の「manifest 自身の真正性検証（署名 escape
  hatch）」を参照し `checks` で Sigstore / Ed25519 / GPG 等の検証を差し込む。lib 側は署名方式を
  決め打ちしない
- **並行性（スレッドセーフ性）**: `AgentRegistry` / `PromptStore` / `WorkflowGraph` は単一スレッド
  / 単一イベントループ前提。複数スレッドからの同時アクセス時の検証発火順序・eager-load・freeze
  状態遷移・例外 raise の振る舞いは規定しない
- **リリース成果物の署名発行側**: sigstore / SLSA provenance 発行・PyPI Trusted Publishing の
  発行側設定はスコープ外
- **モデル / 学習データポイズニング**: 本ライブラリが学習を持たないためレイヤ外
- **アプリ全体保護**: Python の言語特性により**原理的に不可能**。本機能はディスク上の事後改竄と
  公開 API / lib 内部の状態変更経路を網羅する範囲に限定する
- **`libs=True` のスコープと escape hatch**: `libs=True` は `sys.modules` 全件を対象とし、
  pytest / pip / oai-agentspec 自身を含むすべての import 済み配布物を検証する。テストランタイム
  等で邪魔な場合は `libs=False` の escape hatch を使う

## 関連ドキュメント

- `docs/architecture.md` — レイヤー構成 / SDK 隔離 / コンポーネント責務 / 公開 API の SoT
- `docs/security-scanning.md` — ローカル SAST / SCA / シークレットスキャン（install / リリース時）
- `docs/requirements/runtime-integrity-defense.md` — 要件定義書
