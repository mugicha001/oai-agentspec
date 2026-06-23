# runtime インテグリティ防御（`lockdown`）の使い方

`oai_agentspec.lockdown` 1 関数で「ディスク上ファイルの改竄検知」と「`AgentRegistry` /
`WorkflowGraph` の動的書換遮断」を一括で行う。仕様の全体像は `docs/integrity.md` を参照。

本ディレクトリの example は **Azure / OpenAI API を呼ばずに動く**（lockdown は lib 内部の
構造防御 + ファイル整合性検証のみのため）。`uv run python examples/integrity/<file>.py` で
そのまま実行できる。

## ディレクトリ構成

```
examples/integrity/
├── README.md
├── _shared.py             共有 fixture helper（writable_copy 等）
├── gen_manifest.py        manifest 生成 helper（外部ツール sha256sum 互換）
├── sample_app/            git 同梱の read-only サンプル（.integrity/sha256.manifest 同梱）
│   ├── app.py
│   └── .integrity/sha256.manifest
├── sample_prompts/        プロンプトテンプレ用サンプル（.integrity/sha256.manifest 同梱）
│   ├── base/main.md
│   ├── parts/style.md
│   ├── agents/{triage,billing}.md
│   └── .integrity/sha256.manifest
├── 01_minimum.py          最小起動（lockdown 1 行）
├── 02_lockdown_all.py     全 6 段の動作確認（store / registry / workflow / freeze）
├── 03_healthcheck.py      冪等な再発火による擬似継続監視
└── 04_custom_check.py     checks= escape hatch（独自検知関数）
```

## 例の一覧（推奨読書順）

| ファイル | 学べること |
|---|---|
| `01_minimum.py` | `lockdown(<root>)` 1 行で root verify が走ること |
| `02_lockdown_all.py` | store / registry / workflow / custom checks の全 6 段順次処理。固定後の registry/workflow 変更が遮断されること |
| `03_healthcheck.py` | 同じ引数で `lockdown` を 2 回呼ぶ冪等性。disk 改竄を再発火で検知すること |
| `04_custom_check.py` | `checks=[...]` で利用者独自の検知関数を fail-closed フローに混ぜる方法 |

各 example は数十行以内にとどめ、**lockdown 呼び出し本体が一目で見える**ように
fixture コードを `_shared.py` / `sample_*/` に切り出している。

## manifest の作り方

`<root>/.integrity/sha256.manifest` を GNU coreutils `sha256sum` 互換フォーマットで配置する。

```sh
# 外部ツール（コマンドライン）で生成
cd src && mkdir -p .integrity && find . -type f \
  -not -path './.integrity/*' -not -path '*/__pycache__/*' \
  -print0 | xargs -0 sha256sum > .integrity/sha256.manifest
chmod 444 .integrity/sha256.manifest   # 読み取り専用（manifest 信頼境界）
```

または `gen_manifest.py` を使う（純 Python・CI 同梱用）:

```sh
uv run python examples/integrity/gen_manifest.py examples/integrity/sample_app
```

## sample_app / sample_prompts の更新

`sample_app/` と `sample_prompts/` 配下のファイルを編集した場合は、`.integrity/sha256.manifest`
を再生成する必要がある（古い manifest と新ファイルの hash が一致せず lockdown が IntegrityError
を raise する）。

```sh
uv run python examples/integrity/gen_manifest.py examples/integrity/sample_app
uv run python examples/integrity/gen_manifest.py examples/integrity/sample_prompts
```

## 守れる範囲と守れない範囲

`docs/integrity.md` の冒頭の表を参照。本機能は

- **守れる**: disk 上ファイルの事後改竄・registry / workflow の動的書換（公開 + 内部経路）
- **守れない**（原理的に不可能・Out of Scope）: in-memory 改竄・monkey-patch・private 属性
  直書き・継続監視そのもの・アプリ全体保護

汎用 agent governance（OPA / Cedar policy・Ed25519 署名・execution rings 等）は Microsoft
Agent Governance Toolkit / Proofpoint Agent Integrity Framework / Blaxel 等と**併用**する
設計（`docs/integrity.md` の位置付け節を参照）。

## libs 引数について

example はすべて `libs=False` で実行している。これは `sys.modules` 全件の PEP 376 RECORD
照合を**スキップ**する設定であり、`pytest` / `pip` 等を含むため重いことを避ける目的（example
が demo 実行で完結するように）。

**実運用では `libs=True` 既定が推奨**。本番アプリ起動時の lockdown は配布物すべての整合性
照合を含む方が安全。テストランタイムなどで邪魔な場合のみ `libs=False` の escape hatch を使う。
