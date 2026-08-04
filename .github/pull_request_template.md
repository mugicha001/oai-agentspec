## 関連 Issue
<!-- Closes #123 / Refs #123 / 該当 Issue がない場合は理由を Why に書く -->

## Why
<!-- 背景・動機・解こうとしている問題を 1〜3 文で -->

## What
<!-- 変更の概要を 1〜3 文で。diff からは読み取れない要点に絞る -->

## レビューのポイント
<!-- diff だけでは伝わらない非自明な事実 (設計判断・トレードオフ等) -->
-

## テスト
<!--
実施した検証だけを書く。未実施は TBD と明記する。
例:
- `uv run pytest` (1166 passed / coverage 94.76%)
- `uv run ruff check src/ tests/` clean
- 手動確認: examples/workflow/workflow_01_sequential.py が動く
-->
-

## チェックリスト
- [ ] CI (ci / gitleaks / CodeQL) が全て green
- [ ] 変更に対するテストを追加・更新した（カバレッジ 80% 維持）
- [ ] `from agents` / `import agents` は `_adapters/` 配下のみ (NFR-1)
- [ ] ブランチ命名 (`<type>/<issue>-<summary>`) / コミットメッセージ (`<type>(<scope>): ...`) が規約通り（[CONTRIBUTING.md](../CONTRIBUTING.md) 参照）
- [ ] 絵文字・AI 生成示唆の文言（co-author 表記・生成ツール名の言及等）を含まない

<!-- ## Breaking Changes -->
<!-- 後方互換性に影響がある場合のみコメント解除して記載 -->
