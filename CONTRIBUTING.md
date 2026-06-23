# コントリビューションガイド

oai-agentspec への貢献に興味を持っていただきありがとうございます。本書はバグ報告・機能提案・
コード貢献の流れと、開発環境セットアップおよび規約をまとめたものです。

## 開発環境セットアップ

前提:

- Python 3.12 以上
- [uv](https://docs.astral.sh/uv/) （パッケージ・仮想環境管理）

セットアップ手順:

```bash
git clone https://github.com/mugicha001/oai-agentspec.git
cd oai-agentspec
uv sync --all-extras
```

`--all-extras` を付けることで `conversation` / `serve` / `cli` / `governance` / `llmops` /
`lightning` 等のオプション機能の依存も同時に解決されます。

## テスト・lint 実行

```bash
# テスト（カバレッジ 80% gate）
uv run pytest

# 単体テスト
uv run pytest tests/path/to/test_file.py -k "test_name"

# ruff lint
uv run ruff check src/ tests/

# ruff format（自動整形）
uv run ruff format src/ tests/
```

カバレッジは `pyproject.toml` で `fail_under = 80` が設定されており、下回ると失敗します。

## コーディング規約

- PEP 8 準拠 / 行長 100 文字以内（ruff 設定準拠）
- 型ヒント必須（`from __future__ import annotations` + `X | None` 形式を推奨）
- docstring は日本語可（PEP 257 準拠）
- 詳細な設計原則・レイヤー構成は `docs/architecture.md` を参照
- 絵文字は使用しない（コード・ドキュメント・コミットメッセージ・Issue・PR すべて）

## ブランチ命名規約

`<type>/<issue>-<summary>` 形式（例: `feat/123-add-routing`）。

`type` の選択肢:

| type | 用途 |
|---|---|
| `feat` | 新機能追加 |
| `fix` | バグ修正 |
| `refactor` | リファクタリング |
| `docs` | ドキュメント更新 |
| `chore` | その他雑務 |
| `test` | テスト追加・修正 |
| `ci` | CI / CD 変更 |
| `perf` | パフォーマンス改善 |
| `security` | 脆弱性対応 |
| `deps` | 依存関係更新 |

`main` への直 push は禁止しています。ドキュメント修正でもブランチを切って PR を経由してください。

## コミットメッセージ規約

[Conventional Commits](https://www.conventionalcommits.org/) に従います。

```
<type>(<scope>): <要約（50 文字以内）>

<本文（任意・変更理由や背景）>

<Issue 参照（例: refs #123 / closes #123）>
```

詳細なテンプレートは `.github/commit_template.md` を参照してください。

## Pull Request フロー

1. base ブランチは `main`
2. PR を作成する際は `.github/pull_request_template.md` を使用
3. CI（pytest / ruff check / ruff format --check / gitleaks / CodeQL）が全て pass することを
   確認
4. カバレッジ 80% を維持
5. メンテナ 1 人運用のため承認必須は無効としていますが、最低限 status check が緑であることを
   待ってからマージしてください

## コミット author email の設定（重要）

個人メールアドレスの公開を避けるため、GitHub の noreply email を使用することを推奨します。

```bash
git config user.email "<userid>+<username>@users.noreply.github.com"
```

`<userid>` は GitHub Settings → Emails → "Keep my email addresses private" を有効化したときに
表示される ID です。詳細は
[GitHub 公式ドキュメント](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/setting-your-commit-email-address)
を参照してください。

## 絵文字・AI 生成示唆の禁止

- コード / docs / コミットメッセージ / Issue / PR / コメントすべてで絵文字使用は禁止
- AI（Claude / GPT 等）が生成したことを示唆する文言（`Co-Authored-By: Claude` や
  `Generated with Claude Code` 等）を含めない
