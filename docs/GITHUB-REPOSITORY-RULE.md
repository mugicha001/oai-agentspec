# GitHub リポジトリルール

本ドキュメントは oai-agentspec のブランチ運用・コミット規約・PR 作成・Issue テンプレート、
および開発環境と規約を定める Single Source of Truth である。

集約先ブランチは `main`。`develop` は持たない。

---

## 開発環境セットアップ

前提:

- Python 3.12 以上
- [uv](https://docs.astral.sh/uv/)（パッケージ・仮想環境管理）

```bash
git clone https://github.com/mugicha001/oai-agentspec.git
cd oai-agentspec
uv sync --all-extras
```

`--all-extras` により `conversation` / `serve` / `cli` / `governance` / `llmops` /
`lightning` 等のオプション機能の依存も同時に解決される。

---

## テスト・lint

```bash
# テスト（カバレッジ 80% gate）
uv run pytest

# 単体テスト
uv run pytest tests/path/to/test_file.py -k "test_name"

# ruff lint（CI と同一の対象パス）
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# ruff format（自動整形）
uv run ruff format src/ tests/
```

カバレッジは `pyproject.toml` の `fail_under = 80` を下回ると失敗する。

`make test` / `make lint` も用意されているが、`make lint` は `examples/` を含む広い対象で
動く。CI の合否を再現する場合は上記の `src/ tests/` 指定を使う。

---

## コーディング規約

- PEP 8 準拠 / 行長 100 文字以内（ruff 設定準拠）
- 型ヒント必須（`from __future__ import annotations` + `X | None` 形式）
- docstring は日本語可（PEP 257 準拠）
- 詳細な設計原則・レイヤー構成は `docs/architecture.md` を参照

---

## ブランチ構成と役割

| 区分 | ブランチ/Prefix | 役割 | 起点 | PR宛先 | マージ方式 |
|---|---|---|---|---|---|
| 基幹 | `main` | リリースと同期 | - | - | - |
| 作業 | `feat/*` | 新機能追加 | `main` | `main` | squash |
| 作業 | `fix/*` | 不具合修正 | `main` | `main` | squash |
| 作業 | `refactor/*` | リファクタリング | `main` | `main` | squash |
| 作業 | `docs/*` | ドキュメント変更 | `main` | `main` | squash |
| 作業 | `test/*` | テスト追加・修正 | `main` | `main` | squash |
| 作業 | `chore/*` | メンテ・雑務 | `main` | `main` | squash |
| 作業 | `ci/*` | CI / CD 変更 | `main` | `main` | squash |
| 作業 | `perf/*` | パフォーマンス改善 | `main` | `main` | squash |
| 作業 | `security/*` | 脆弱性対応 | `main` | `main` | squash |
| 作業 | `deps/*` | 依存関係更新 | `main` | `main` | squash |

`main` への直 push・直コミットは禁止する。ドキュメント修正でもブランチを切り PR を経由する。

---

## ブランチ命名規則

基本形: `<type>/<issue>-<summary>`

- `<type>` は上表の作業ブランチ prefix（コミットの type と同じ語彙）
- summary は `<verb>-<object>(-<qualifier>)` の kebab-case
- 例: `feat/123-add-routing`
- 例: `fix/456-fix-timeout-handling`

推奨スラグ（verb の例）: `add`, `update`, `fix`, `remove`, `refactor`, `improve`

---

## Conventional Commits 規約

形式: `<type>[optional scope][!]: <description>`

```
<type>(<scope>): <要約（50 文字以内）>

<本文（任意・変更理由や背景）>

<Issue 参照（例: refs #123 / closes #123）>
```

### type 一覧

| type | 用途 | 対応ブランチ prefix |
|---|---|---|
| `feat` | 新機能追加 | `feat/*` |
| `fix` | バグ修正 | `fix/*` |
| `docs` | ドキュメント更新 | `docs/*` |
| `refactor` | リファクタリング（機能変更なし） | `refactor/*` |
| `perf` | パフォーマンス改善 | `perf/*` |
| `test` | テスト追加・修正 | `test/*` |
| `build` | ビルド設定・依存関係更新 | `chore/*` |
| `ci` | CI/CD 設定変更 | `ci/*` |
| `chore` | その他雑務（整形・lint 等） | `chore/*` |
| `revert` | コミット取り消し | `fix/*` |
| `security` | 脆弱性対応 | `security/*` |
| `deps` | 依存関係更新 | `deps/*` |

### scope 一覧

`src/oai_agentspec/` のモジュール構成に対応する。該当が無ければ省略してよい。

| 区分 | scope |
|---|---|
| 宣言層 | `spec` / `registry` / `handoffs` / `prompts` / `workflow` / `protocols` / `tool-registry` / `agent-names` / `validation` / `next-turn` / `integrity` / `exceptions` / `realtime` |
| runtime | `conversation` / `serve` / `cli` / `intent` / `finetune` / `resilience` / `observability` / `lightning` / `governance` / `guardrails` / `hooks` / `llmops` / `deterministic` |
| 横断 | `adapters` / `docs` / `examples` / `tests` / `ci` / `deps` / `security` / `requirements` / `usage` / `github` / `contributing` |

### ブレーキングチェンジ

- type に `!` を付与: `feat(serve)!: レスポンスフォーマットを変更`
- フッターに記載: `BREAKING CHANGE: <説明>`

### コミットメッセージ例

```
feat(workflow): 経路C で外側 context を内部ノードへ伝播

lib 所有フックで外側 context を捕捉し、内部ノードの実行時に引き回すようにした。
既存の経路 A / B の挙動は変えない。

Closes #109
```

---

## hunk 単位ステージング STEP

### STEP1: `git diff HEAD` で全変更を確認
### STEP2: `git add -p` で hunk 単位でステージング
- `y`: ステージング / `n`: スキップ / `s`: hunk をさらに分割 / `q`: 終了

---

## PR ルール

### PR タイトル

Conventional Commits に準拠する。

### PR テンプレート

`.github/pull_request_template.md` を使用する。構成:

| セクション | 内容 |
|---|---|
| 関連 Issue | `Closes #<Issue番号>` または PR の背景・動機 |
| 概要 | 変更の目的と内容を簡潔に |
| 変更内容 | 具体的な変更を箇条書き |
| 影響範囲 | 変更の影響を受けるコンポーネント |
| テスト | テスト方法・確認手順 |
| チェックリスト | スタイル準拠・テスト・ドキュメント・Breaking Changes |

### PR の作成手順

1. Issue 情報とコミットログを確認する
2. テンプレートの各セクションを埋める
3. base ブランチは `main`
4. squash マージを使用する
5. CI（pytest / ruff check / ruff format --check / gitleaks / CodeQL）が全て pass すること
   を確認する
6. カバレッジ 80% を維持する

メンテナ 1 人運用のため承認必須は無効としているが、最低限 status check が緑であることを
待ってからマージする。

---

## Issue テンプレート一覧

`.github/ISSUE_TEMPLATE/` 配下に以下のテンプレートを配置している:

| テンプレート | ファイル | 用途 | ラベル |
|---|---|---|---|
| バグ報告 | `bug-template.md` | 不具合・バグの報告 | `bug` |
| 新機能追加 | `feature-template.md` | 新規機能の追加要望 | `feature` |
| リファクタリング | `refactor-template.md` | コードのリファクタリング提案 | `refactor` |
| フィードバック | `feedback-template.md` | 一般的なフィードバック | `feedback` |
| トラッキング | `tracking.md` | 複数の子 Issue をまとめる親 Issue | `epic` |

テンプレート選択の設定は `.github/ISSUE_TEMPLATE/config.yml` で管理する。

---

## コミットテンプレート

`.github/commit_template.md` をコミットメッセージのテンプレートとして使用できる。

```bash
git config commit.template .github/commit_template.md
```

type と scope の詳細は本ドキュメントの「Conventional Commits 規約」を参照。

---

## コミット author email の設定

個人メールアドレスの公開を避けるため、GitHub の noreply email を使用する。

```bash
git config user.email "<userid>+<username>@users.noreply.github.com"
```

`<userid>` は GitHub Settings → Emails → "Keep my email addresses private" を有効化した
ときに表示される ID である。詳細は
[GitHub 公式ドキュメント](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/setting-your-commit-email-address)
を参照。

---

## 絵文字・AI 生成示唆の禁止

- コード / docs / コミットメッセージ / Issue / PR / コメントすべてで絵文字使用は禁止
- AI が生成したことを示唆する文言（`Co-Authored-By:` の AI 名義や生成ツール名の記載等）を
  含めない
