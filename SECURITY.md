# セキュリティポリシー

oai-agentspec におけるセキュリティ脆弱性の報告手順、サポート対象バージョン、対応 SLA、および
ブランチ保護に関する方針をまとめます。

## 報告窓口

セキュリティ脆弱性は GitHub Issues 経由で報告してください:

- 報告窓口: https://github.com/mugicha001/oai-agentspec/issues
- 即座の対応が必要と判断される場合は Issue タイトルに `[SECURITY]` prefix を付けてください

**注記**: 一般的な OSS では GitHub Security Advisories（非公開報告）を推奨しますが、本リポジトリ
は Alpha 段階（0.x.x）かつメンテナ 1 人運用であり、公開窓口で報告を受け付ける運用判断としています。
報告者が公開を避けたい場合は、Issue 本文に詳細を書かず、最小限の連絡情報のみ記載してください。
追加の調整方法を別途連絡します。

## サポート対象バージョン

Alpha 期（0.x.x）は **最新マイナーバージョンのみサポート** します。

| Version | Supported |
|---|---|
| 0.3.x | サポート対象 |
| 0.2.x 以前 | サポート対象外 |

安定版（1.0.0）リリース以降のサポート方針は別途見直します。

## 対応 SLA

- **初動応答**: 報告受領から 7 営業日以内（メンテナ 1 人運用の目安）
- **修正版リリース**: 脆弱性の重大度（CVSS スコアまたは exploitability）に応じて判断
  - Critical / High: 可能な限り早期にパッチリリース
  - Medium / Low: 次回マイナー / パッチリリースに含める

## Branch Protection 方針

`main` ブランチには以下のブランチ保護が有効化されています:

- PR 経由必須（直 push 禁止）
- force push 禁止
- ブランチ削除禁止
- Required status checks: `ci` / `gitleaks` / `CodeQL`
- "Require branches to be up to date before merging": 有効
- "Require pull request reviews before merging": 無効（メンテナ 1 人運用のため）

### admin enforce 方針

**ON を推奨します**（メンテナを含むバイパスを禁止）。これは管理者であっても CI を通過していない
変更を `main` に直接反映できない運用を意味します。

緊急バイパスが必要な場合（例: CI 基盤障害により status check が永続的に失敗する状況）に限り、
一時的に admin enforce を OFF にすることがあり得ます。その場合は事後に PR + Issue で経緯（OFF
にした理由・影響範囲・再度 ON に戻した時刻）を文書化してください。
