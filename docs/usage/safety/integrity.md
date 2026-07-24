# Integrity（lockdown と manifest 検証）

## 何を解決するか

エージェント宣言・プロンプトテンプレート・tool 実体が実行時に想定と一致しているかを検証します。`lockdown` は 6 段順次（root verify → store verify+preload → libs detect → custom checks → registry freeze → workflow freeze）で fail-closed に整合性を守る単一関数です。`AgentRegistry` / `WorkflowGraph` の凍結（`freeze()`）、`PromptStore` の preload（cache only）、sha256 manifest 照合、PEP 376 RECORD による配布物照合を 1 呼び出しで行います。

プロンプトテンプレートの改ざんは `PromptTemplateIntegrityError`、それ以外の整合性違反は `IntegrityError` で検知します。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| `lockdown(root)` のみ | root manifest 照合 + libs 検出 | デプロイ直後の最小検証 |
| `lockdown(root, store=..., registry=..., workflow=...)` | 4 対象全部 | 本番相当の完全検証 |
| `checks=[...]` に custom | 独自 `IntegrityCheck` を差し込む | 追加検知（外部設定ハッシュ等） |
| `libs=False` | 配布物照合スキップ | dev / 依存が multi-source なとき |

## 使い方

- import: `from oai_agentspec import lockdown, IntegrityCheck, IntegrityError, PromptTemplateIntegrityError`
- extras: なし
- 依存 env: なし

```python
from pathlib import Path
from oai_agentspec import lockdown

# 例: root manifest + PromptStore + Registry + Workflow を一括固定
lockdown(
    Path("."),               # root（<root>/.integrity/sha256.manifest を照合）
    store=store,             # PromptStore（manifest 照合 + eager preload）
    registry=registry,       # AgentRegistry.freeze()
    workflow=wf,             # WorkflowGraph.freeze()
    libs=True,               # sys.modules 配布物の PEP 376 RECORD 照合
    checks=[my_custom_check],  # IntegrityCheck = Callable[[], None]
)
```

manifest 生成は `examples/integrity/gen_manifest.py` を参照。

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `lockdown(root, store=None, registry=None, workflow=None, *, libs=True, checks=None)`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `root` | `Path` | 必須 | `<root>/.integrity/sha256.manifest` と root 配下を sha256 照合 |
| `store` | `PromptStore \| None` | `None` | store 配下も照合 + eager preload。None でスキップ |
| `registry` | `AgentRegistry \| None` | `None` | `registry.freeze()` を呼ぶ。None でスキップ |
| `workflow` | `WorkflowGraph \| None` | `None` | `workflow.freeze()` を呼ぶ。None でスキップ |
| `libs` | `bool` | `True` | `sys.modules` 配下配布物の PEP 376 RECORD 照合 |
| `checks` | `list[IntegrityCheck] \| None` | `None` | 利用者の独自検知関数リスト（`IntegrityCheck = Callable[[], None]`） |

### 例外

| 例外 | 送出タイミング |
|---|---|
| `IntegrityError` | root verify / libs detect / custom check 違反 |
| `PromptTemplateIntegrityError` | `IntegrityError` サブクラス。store verify 違反 |
| `RegistryFrozenError` | registry freeze 後の変更操作 |
| `WorkflowFrozenError` | workflow freeze 後の変更操作 |

### `IntegrityCheck`

型エイリアス `Callable[[], None]`。違反時に `IntegrityError` 系例外を raise する契約。

## 判断軸

- registry の spec 差し替えを本番で禁止したいなら **`lockdown(..., registry=registry)`** を必ず呼ぶ
- プロンプト・tool の改ざん検知が要件なら **manifest 生成 + `lockdown(root, store=..., libs=True)`** をデプロイ後の healthcheck で走らせる
- プロンプトのみ検知したい場合は **`PromptTemplateIntegrityError`** をキャッチする粒度で運用

## 落とし穴

- `registry.freeze()` は不可逆。テスト用途では registry を都度作り直す
- manifest 生成はビルドパイプラインに組み込む。手動生成は運用ずれの原因になる
- `lockdown` 後の `PromptStore.reload()` は `PromptTemplateIntegrityError`（cache only 契約）

## 参照

- 詳細設計: `docs/integrity.md`
- 具体例: `examples/integrity/01_minimum.py` 〜 `04_custom_check.py` / `gen_manifest.py`

## 次

[../ops/intent.md](../ops/intent.md) — 意図予測をいつ挟むか
