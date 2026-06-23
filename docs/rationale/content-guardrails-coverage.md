# Rationale: 内容ガードレールのカバレッジマトリクスと検知家族の選定根拠

本ファイルは内容ガードレール（`runtime/guardrails`）の helper 群が、どのセキュリティ / 品質
framework のどの項目を、どの検知機構でカバーするかの選定根拠を保持する archival ドキュメントである。
検知家族への振り分け根拠・項目選別の根拠・トレードオフを検討経緯として不変に保つ（実装変更や
カバレッジ項目の進捗に追随して更新しない）。カバレッジ項目の進捗追跡は Issue 側で管理し、本ファイルは
選定根拠の archival に徹する。

現在の確定仕様（位置づけ・検知 3 家族・適用境界・配置・OWASP 対応表）は `docs/architecture.md` の
「内容ガードレール（ローカル品質ゲート支援）」節を Single Source of Truth とする。本ファイルはそれと
矛盾しない範囲で振り分けの根拠のみを記録する。

## 結論サマリ

- 内容ガードレールは「何を言うか」を入出力・中間ツール段で検査する層であり、検知器を 3 家族
  （外部検知器 A / prompt 駆動 LLM B / 決定的・ロジック系 C）に分けてファクトリへ DI する。
- 軸とする framework は OWASP LLM Top 10 を主軸とし、MITRE ATLAS / NIST AI RMF は内容検査で
  カバーできる関連項目に限定して併記する。検知家族は framework 中立であり、特定 framework に
  縛られない。
- 「何を言うか」を検査する内容ガードレールと「何をできるか」を許可 / 拒否する AGT ガバナンス
  （ツール単位ポリシー強制・監査）は直交する。改竄 / 供給網インテグリティ（integrity・供給網）は
  内容検査では守れない直交領域として別途扱う。

## 1. 検知 3 家族の定義と選定根拠

各 framework の項目を、検知の性質で 3 家族へ振り分ける。家族の分け方は「検知本体をライブラリが持つか
（同梱の決定的ロジック / 外部 DI）」と「判定が決定的か LLM 推論か」の 2 軸で決めた。

| 家族 | 検知の性質 | 同梱 / DI | 判定 | 代表ユースケース |
|---|---|---|---|---|
| A 外部検知器 | 専門検知サービス / モデルを薄く接着 | 検知本体は非同梱（外部 DI） | 外部実装依存 | PII 検出（Presidio）・モデレーション・注入検知サービス |
| B prompt 駆動 LLM | 判定 model + 判定 prompt で LLM-as-judge | model / prompt は DI（非同梱） | LLM 推論 | プロンプトインジェクション判定・文脈依存の漏洩判定 |
| C 決定的・ロジック系 | カナリア / 正規表現 / 長さ / allow-deny / predicate / 注入ベースライン | 再利用 helper を同梱（DI 上書き可） | 決定的 | システムプロンプト漏洩のカナリア検知・既知パターン照合 |

選定根拠:

- **A を外部 DI に寄せた理由**: PII 検出やモデレーションは専門 OSS / サービス（Presidio 等）が成熟して
  おり、本ライブラリが再実装する価値は低く、依存も重い。検知本体を同梱せず薄い接着のみ提供することで、
  依存ゼロ extra（`guardrails = []`）と非同梱方針（プロンプト / モデル / 重い検知をライブラリに持たない）を
  両立する。
- **B を prompt / model 全 DI にした理由**: LLM-as-judge の判定 prompt・model は利用者ドメイン・規制要件で
  大きく変わり、ライブラリにハードコードするとプロンプト非同梱方針・env 非依存方針に反する。判定 model の
  呼び出しは `_adapters` 経由へ寄せ、外部直叩きを避ける。
- **C を同梱 + DI 上書き可にした理由**: カナリア比較・正規表現・長さ・allow-deny は決定的で軽量・依存ゼロで
  あり、SDK なしで単体検証できる plain ロジックとして同梱できる。注入ベースライン（SQLi / コマンド注入 /
  パストラバーサルの代表パターン）も同梱するが、これは網羅的検知ではなく補助検知である。注入対策の本丸は
  パラメータ化クエリ / 安全 API 利用であり、ベースラインはあくまで早期の粗い網として置き、利用者が DI で
  上書き / 拡張できる前提とした。

## 2. カバレッジマトリクス（関連項目限定）

各 framework の項目のうち、内容ガードレール（入出力 / 中間ツール段の内容検査）で扱える関連項目に限定して
振り分ける。カバー状況は本支援層から見たもの（主カバー = 本支援層が主たる防御線 / 部分 = 補助的 / 対象外 =
内容検査では守れず他機構が担う）で表す。本支援層以外が主担当の項目は、その担当機構を併記する。

凡例:

- カバー機構: `A` 外部検知器 / `B` prompt 駆動 LLM / `C` 決定的・ロジック系 / `AGT` AGT ガバナンス
  （ツール単位ポリシー強制・監査・別 Issue） / `dep-scan` 依存スキャン（Trivy / gitleaks 等・`security-scanning.md`）/
  `llmops` LLMOps 評価（`runtime/llmops`）/ `HITL` ツール実行承認（`docs/architecture.md` の HITL 節）/ `-` 該当外
- カバー状況: 主カバー / 部分 / 対象外

### 2.1 OWASP LLM Top 10（主軸）

| 項目 | カバー機構 | カバー状況 |
|---|---|---|
| LLM01 Prompt Injection | B + A（input） | 主カバー |
| LLM02 Sensitive Information Disclosure | A（Presidio）+ C（カナリア）（output） | 主カバー |
| LLM03 Supply Chain | dep-scan / AGT | 対象外（内容検査では守れない・別 Issue） |
| LLM04 Data and Model Poisoning | dep-scan / llmops | 対象外（内容検査では守れない） |
| LLM05 Improper Output Handling | C（output・allow-deny / 正規表現）+ A | 部分 |
| LLM06 Excessive Agency | AGT / HITL | 対象外（「何をできるか」= AGT ・承認） |
| LLM07 System Prompt Leakage | C（カナリア）主 + B（output 二層） | 主カバー |
| LLM08 Vector and Embedding Weaknesses | -（RAG はスコープ外） | 対象外 |
| LLM09 Misinformation | B + llmops（factual_grounding） | 部分 |
| LLM10 Unbounded Consumption | C（長さ / サイズ閾値・input/output） | 部分 |

### 2.2 MITRE ATLAS（内容検査で扱える関連 technique に限定）

| 関連 technique | カバー機構 | カバー状況 |
|---|---|---|
| Prompt Injection（LLM Prompt Injection） | B + A（input） | 主カバー |
| Jailbreak（LLM Jailbreak） | B + A（input） | 部分 |
| LLM Meta Prompt Extraction（system prompt 抽出） | C（カナリア）+ B（output） | 主カバー |
| LLM Data Leakage（学習 / 文脈データ漏洩） | A（PII）+ C（カナリア）（output） | 部分 |
| ML Supply Chain Compromise | dep-scan | 対象外（内容検査では守れない） |
| 物理 / インフラ系 technique | - | 対象外 |

### 2.3 NIST AI RMF（内容検査で扱える関連 subcategory に限定）

NIST AI RMF は管理プロセスのフレームワークであり、内容ガードレールはその技術的統制の一手段として
MEASURE / MANAGE 機能の一部 subcategory に寄与する（ガバナンス全体を代替しない）。

| 関連 subcategory（機能） | カバー機構 | カバー状況 |
|---|---|---|
| MEASURE 2.6 系（安全性・有害出力の測定） | B + C + llmops | 部分 |
| MEASURE 2.7 系（セキュリティ・レジリエンスの測定） | B + A + C | 部分 |
| MEASURE 2.10 系（プライバシー・PII の測定） | A（Presidio）+ C | 部分 |
| MANAGE 2.x 系（特定リスクへの統制適用） | A / B / C（入出力検査統制として） | 部分 |
| GOVERN / MAP の組織・文書化 subcategory | - | 対象外（プロセス統制であり内容検査の範囲外） |

## 3. 項目選別の根拠とトレードオフ

- **主軸を OWASP LLM Top 10 にした理由**: LLM アプリケーション固有の脅威を入出力単位で具体的に列挙して
  おり、入出力 / ツール段の内容検査という本支援層の適用境界に最も自然に対応する。MITRE ATLAS / NIST AI RMF は
  内容検査で扱える関連項目に限定して併記し、家族が framework に縛られないことを示すに留めた（全項目の
  網羅は本支援層の責務ではない）。
- **対象外項目を明示した理由**: Supply Chain（LLM03）/ Data and Model Poisoning（LLM04）/ Excessive
  Agency（LLM06）/ Vector and Embedding（LLM08）は、内容検査では原理的に守れない（依存スキャン・AGT
  ガバナンス・承認・RAG スコープ外がそれぞれ担う）。これらを本支援層のカバー対象に含めると責務の境界が
  曖昧になり二重実装を招くため、担当機構を併記して対象外と明示した。
- **B（LLM-as-judge）の限界**: prompt 駆動判定は非決定的でコスト・レイテンシを伴い、判定 prompt / model の
  品質に依存する。決定的に確認できる箇所（C・カナリア / 正規表現 / 長さ）を優先し、B は決定的検知で捉え
  きれない文脈依存の判定（プロンプトインジェクション・文脈依存の漏洩判定）に限定して二層で用いる
  （LLM07 はカナリア主 + B 二層）。
- **注入ベースライン（C）を補助に留めた理由**: 正規表現ベースの注入検知は誤検知 / 検知漏れが不可避であり、
  網羅的防御を約束できない。注入対策の本丸はパラメータ化クエリ / 安全 API 利用であることを前提に、早期の
  粗い網（DI で上書き可）として位置づけることで、過剰な信頼を招かないようにした。
- **ツール境界を内容検査のみに限定した理由**: ツール境界ラッパーは中間ツール出力 / 引数の内容検査に徹し、
  実行可否の allow / deny 制御（ポリシー強制）は新設しない。ポリシー強制は AGT ガバナンスの責務であり
  （`docs/rationale/agt-governance-integration.md`）、両者の結線責務が重複するのを避けるための役割分担で
  ある。
