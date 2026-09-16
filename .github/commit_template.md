# oai-agentspec のコミットテンプレート（git commit.template として使う）
# Conventional Commits の type / scope 定義。commit スキルが参照する。
# 注記: 本ファイルは git commit.template 用のため、# 始まりの行はコミット時に除去される。
#
# <type>(<scope>): <要約（50文字以内）>
#
# --- 本文（任意。72 文字で折り返す。何を・なぜ変えたか）---
#
#
# --- Issue 参照（任意）---
# refs #<番号>  /  closes #<番号>
#
# --- ブレーキングチェンジ（任意）---
# BREAKING CHANGE: <説明>
#
# ============================ 記入ガイド ============================
# type 一覧:
#   feat     新機能追加
#   fix      バグ修正
#   docs     ドキュメント更新
#   refactor リファクタリング（機能変更なし）
#   perf     パフォーマンス改善
#   test     テスト追加・修正
#   build    ビルド設定・依存関係更新
#   ci       CI/CD 設定変更
#   chore    その他雑務（整形・lint 等）
#   revert   コミット取り消し
#   security 脆弱性対応
#   deps     依存関係更新
#
# scope 一覧（src/oai_agentspec/ のモジュール構成に対応。該当が無ければ省略してよい）:
#   宣言層     spec / registry / handoffs / prompts / workflow / protocols /
#              tool-registry / agent-names / validation / next-turn /
#              integrity / exceptions / realtime
#   runtime    conversation / serve / cli / intent / finetune / resilience /
#              observability / lightning / governance / guardrails / hooks /
#              llmops / deterministic
#   横断       adapters / docs / examples / tests / ci / deps / security /
#              requirements / usage / github / contributing
#
# 破壊的変更は type に ! を付ける: feat(serve)!: レスポンスフォーマットを変更
# 詳細は docs/GITHUB-REPOSITORY-RULE.md（ブランチ命名規則・Conventional Commits 規約）を参照。
