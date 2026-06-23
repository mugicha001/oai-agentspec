# /project-brief スキル

## 1. 概要
リポジトリを横断精査し「目的・背景・できること」を網羅した包括概観を、技術者・非技術者の双方が 1 資料で読める形で生成する新規スキル `/project-brief`（`.claude/skills/project-brief/`）を導入する。出力は `docs/` 配下の Markdown 正本（Why/What を担当し How は `docs/architecture.md` へ束ねる）と、その正本から単方向生成する自己完結型の consulting-style standalone HTML の 2 層構成とし、横断精査には既存 `code-explorer` を合成して用いる。

## 2. 機能要件

### FR-1: リポジトリ横断精査による包括概観の収集
- ユーザーストーリー: スキル利用者として、リポジトリ全体を横断精査して目的・背景・できることを網羅的に把握したい。なぜなら、散在する情報を 1 資料に集約するための一次入力を漏れなく揃えたいから。
- 受け入れ基準:
  - [ ] WHEN スキルが起動した THEN まず粗スキャンを実行し、対象リポジトリの全体像（ディレクトリ構成・主要モジュール・公開 API・ドキュメント所在）の見取り図を取得する。
  - [ ] WHEN 横断精査を行う THEN 探索の主力として `code-explorer` を合成して呼び出し、その出力レポートを後続の Markdown 合成の一次入力として扱う。
  - [ ] IF 精査範囲が未指定である THEN 既定の基線として `src` + `docs` + `README` を精査対象とする。
  - [ ] WHEN 精査結果を集約する THEN 推測ではなく実際に読んだ事実のみを採用し、確認できなかった領域は Markdown 正本上に「未確認」注記として明示する（正本に未確認の所在が残ることを確認軸とする）。

### FR-2: 対話的なスコープ・粒度・読み手比重の決定
- ユーザーストーリー: スキル利用者として、精査範囲・記述粒度・技術者/非技術者の読み手比重を対話で擦り合わせたい。なぜなら、概観の目的と読者層は実行ごとに異なり、固定既定では過不足が生じるから。
- 受け入れ基準:
  - [ ] WHEN 粗スキャンが完了した THEN 精査範囲（基線からの拡縮）・記述粒度・読み手比重をユーザーに提示し、合意を取ってから Markdown ドラフト作成へ進む。
  - [ ] IF ユーザーが基線（`src` + `docs` + `README`）の拡張または縮小を指示した THEN 指示に従って精査対象を再設定する。
  - [ ] WHEN 読み手比重が決定された THEN 技術者・非技術者の双方が 1 資料で読めるよう、用語補足と詳細度を比重に応じて調整する。
  - [ ] IF ユーザーがスコープに合意しない THEN 合意が得られるまで精査範囲・粒度・比重の擦り合わせを繰り返す。

### FR-3: Markdown 正本（SoT）の生成と docs 反映
- ユーザーストーリー: スキル利用者として、合意済みの概観を `docs/` 配下の Markdown 正本として反映したい。なぜなら、概観の Single Source of Truth を docs 規約準拠の現在仕様として保持したいから。
- 受け入れ基準:
  - [ ] WHEN Markdown ドラフトが合意された THEN `docs/` 配下の Markdown 正本（例: `docs/overview.md`）として反映する。
  - [ ] WHEN 正本を生成する THEN overview は Why/What（目的・背景・できること）を担当し、How（設計詳細）は「詳細は `docs/architecture.md` を参照」として束ね、設計詳細を再記述しない。
  - [ ] WHEN 正本と README の関係を扱う THEN README（導入・クイックスタート）と内容を重複させず、overview は包括概観として補完する位置づけにする。
  - [ ] IF 同一トピックの既存 docs ファイルが存在する THEN 新規作成の前に既存ファイルへの統合可否を確認し、二重記述を作らない。
  - [ ] WHEN docs ファイル名を決定する THEN Issue 番号・PBI 番号をファイル名に含めない。

### FR-4: 正本から consulting-style standalone HTML の生成（単方向）
- ユーザーストーリー: スキル利用者として、Markdown 正本から綺麗なコンサル資料品質の HTML 成果物を生成したい。なぜなら、非技術者にも訴求する配布可能な単一ファイルが必要だから。
- 受け入れ基準:
  - [ ] WHEN Markdown 正本が確定した THEN その正本を唯一の入力として consulting-style standalone HTML（例: `docs/overview.html`）を生成する。
  - [ ] WHEN HTML を生成する THEN `docs/` 配下へコミット対象として出力する。
  - [ ] WHEN HTML を生成する THEN HTML から Markdown 正本への逆方向の編集・反映は行わず、生成は正本 → HTML の単方向のみとする。
  - [ ] IF 正本が更新された THEN HTML は再生成（regenerate）して整合させ、HTML を手編集して差分を持たせない。

### FR-5: 自己完結型 HTML の品質要件
- ユーザーストーリー: 資料の閲覧者として、外部接続がなくても HTML をブラウザで開いて綺麗なコンサル資料として読みたい。なぜなら、オフライン環境や CDN 遮断下でも資料を確実に閲覧したいから。
- 受け入れ基準:
  - [ ] WHEN HTML を生成する THEN CSS・SVG をすべて inline で内蔵し、外部 CDN・外部スタイルシート・外部画像・外部フォントに依存しない自己完結ファイルとする。
  - [ ] WHEN HTML をブラウザで直接開いた THEN ネットワーク接続なしでレイアウト・配色・図が正しく描画される。
  - [ ] WHEN HTML を構成する THEN 表紙・エグゼクティブサマリ・章レイアウトと、editorial 視覚部品（統計コールアウト・「できること」グリッド・簡易の全体像シェマ）を含み、視覚言語を統一する。
  - [ ] WHEN 配色を決定する THEN 任意の配色でよいが、配色・タイポグラフィ・余白・コンポーネント様式が全章で統一されていること（「綺麗なコンサル資料」品質の検証は NFR-5 の計測基準に従う）。

### FR-6: 再利用方針（合成優先・コピー禁止）に基づく既存資産の利用
- ユーザーストーリー: スキル保守者として、既存スキル/テンプレを合成して利用しコードを複製しないようにしたい。なぜなら、重複実装を避け保守点を一元化したいから。
- 受け入れ基準:
  - [ ] WHEN 横断精査を行う THEN `code-explorer` を合成（compose）して用い、その実装を本スキルへ複製しない。
  - [ ] IF ユーザーが明示要求しない限り THEN `architecture-diagram`（ダーク/技術/インフラ寄り）は常用せず、明示要求時のみ ad hoc に利用する。
  - [ ] IF ユーザーが明示要求した場合 THEN `diagram-design` 由来の consulting 適性のある図型（quadrant / pyramid / layer-stack / timeline / venn 等）を ad hoc に利用してよく、その style-guide のカスタム作法は参考流用してよい。
  - [ ] WHEN 視覚部品や図エンジンが必要になった THEN 既存の図エンジンを再発明せず、必要な視覚部品のみを inline SVG/CSS で内蔵する。
  - [ ] WHEN 外部 OSS（`codebase-onboarding` / CodeWiki / DeepWiki 等）を参照する THEN 設計の参考に留め、コードをベンダリング（取り込み）しない。

### FR-7: 決定的手順の明示
- ユーザーストーリー: スキル保守者として、必ず実行されるべき工程を SKILL.md に明示の番号付き手順として書きたい。なぜなら、スキル間呼び出しは指示ベースで非決定的になりやすく、HTML 生成などの必須工程の取りこぼしを防ぎたいから。
- 受け入れ基準:
  - [ ] WHEN SKILL.md を記述する THEN 「粗スキャン → スコープ擦り合わせ → Markdown ドラフト → レビュー往復 → docs 反映 → HTML 生成」を明示の番号付き手順として記述する。
  - [ ] WHEN 必ず実行すべき工程（HTML 生成など）を扱う THEN その工程を番号付き手順内で省略不可と明記し、決定的に実行させる。
  - [ ] IF Markdown ドラフトのレビューで指摘が出た THEN 指摘反映のための往復を行い、合意が取れてから docs 反映へ進む。

### FR-8: 保守時の再生成フロー
- ユーザーストーリー: スキル保守者として、概観が陳腐化したときに正本と HTML を更新したい。なぜなら、概観を現在仕様に追従させ続けたいから。
- 受け入れ基準:
  - [ ] WHEN 正本 Markdown の更新が必要になった THEN `/spec-sync` での同期、またはスキルの再実行によって正本を更新する。
  - [ ] WHEN 正本が更新された THEN HTML を再生成して正本と整合させる。

## 3. 非機能要件

### NFR-1: 保守性（05-docs 準拠・二重記述ゼロ）
- 要件: 生成する Markdown 正本は `.claude/rules/05-docs.instructions.md` の Spec 駆動規約に従い、現在仕様のみを記述する。How（設計詳細）は `docs/architecture.md` へ参照で束ね、同一内容を複数ファイルに重複させない。
- 計測基準: 正本に PR 番号・Issue 番号・履歴記述（「以前は」「主な変更点」等）・AI モデル名・絵文字が含まれないこと（grep で 0 件）。`docs/architecture.md` および README と重複する記述ブロックが存在しないこと（レビューで二重記述指摘 0 件）。設計詳細は「詳細は `docs/architecture.md` を参照」の参照行で束ねられていること。

### NFR-2: 再利用性（合成優先・新規コードは薄い文書テンプレ＋視覚部品に限定）
- 要件: 横断精査・図生成の既存資産は合成して利用し、新規に持ち込むものは薄い consulting 文書テンプレ（表紙・エグゼクティブサマリ・章レイアウト）と editorial 視覚部品に限定する。図エンジンは再発明しない。
- 計測基準: 本スキルが `code-explorer` を呼び出し合成していること。`code-explorer` / `architecture-diagram` / `diagram-design` / 外部 OSS のコードを複製・ベンダリングした箇所が 0 件であること。本スキルが新規に内蔵するのは薄い文書テンプレと inline SVG/CSS 視覚部品のみであること（レビューで汎用図エンジン実装の指摘 0 件）。

### NFR-3: 整合性（MD 正本 → HTML 単方向生成の不変）
- 要件: HTML は Markdown 正本を唯一の入力として生成される派生物であり、生成方向は正本 → HTML の単方向に保つ。HTML から正本への逆流を持たない。
- 計測基準: HTML の内容が正本の内容に対応していること（章・できること項目・全体像が正本と矛盾しない）。正本更新後に HTML を再生成すると差分が反映されること。HTML を直接手編集した差分が正本に存在しないこと（HTML は regenerable な派生物として扱われていること）。

### NFR-4: 自己完結性（HTML が外部依存なしで開ける）
- 要件: HTML 成果物は外部 CDN・外部スタイルシート・外部画像・外部フォント・JavaScript 外部読み込みに依存せず、単一ファイルで完結する。
- 計測基準: HTML ファイル内に `http://` / `https://` で始まる外部リソース参照（link/script/img/font の外部 URL）が 0 件であること。ネットワーク遮断環境でブラウザに直接読み込んでレイアウト・配色・SVG 図が正しく描画されること。

### NFR-5: 視覚統一（consulting 資料品質）
- 要件: HTML は表紙・エグゼクティブサマリ・章レイアウトおよび editorial 視覚部品（統計コールアウト・できることグリッド・簡易全体像シェマ）を通じて一貫した視覚言語を持ち、綺麗なコンサル資料の品質を満たす。
- 計測基準: 全章で配色・タイポグラフィ・余白・コンポーネント様式が統一されていること（レビューで様式不統一の指摘 0 件）。技術者・非技術者の双方が 1 資料で目的・背景・できることを読み取れること（読み手比重の合意に沿うこと）。

## 4. 制約事項
- 技術的制約:
  - 生成する `docs/` 配下の Markdown は `.claude/rules/05-docs.instructions.md` に従い、SKILL.md は `.claude/rules/02-prompt.instructions.md` の skill 構造（`# 役割` / 番号付き STEP / `# 制約`、ケバブケースの一意 `name`、フロントマター規約）に従う。
  - HTML は inline CSS/SVG による自己完結ファイルとし、外部 CDN を参照しない。
  - スキル間呼び出しは指示ベースで非決定的なため、必須工程は SKILL.md に明示の番号付き手順として記述する。
  - `code-explorer` を合成の主力とし、`architecture-diagram` は常用せず明示要求時のみ ad hoc 利用、`diagram-design` 由来は降格・任意で明示要求時のみ ad hoc 利用とする。
  - `skill-creator` はビルド時に雛形を生成する用途に限り 1 回利用し、本スキルの実行時依存にはしない。
  - docs ファイル名に Issue 番号・PBI 番号を含めない。コード・ドキュメント・出力に絵文字、AI モデル名、履歴記述を含めない。
- ビジネス制約:
  - 概観の正本は `docs/` 配下に置き、README とは重複せず補完する位置づけを保つ。
  - 精査範囲は固定既定に固定せず、実行ごとに対話で決定する（既定基線は `src` + `docs` + `README`）。

## 5. 影響範囲
- 関連コンポーネント:
  - 新規: `.claude/skills/project-brief/`（SKILL.md と薄い consulting 文書テンプレ・視覚部品アセット）。
  - 新規生成物（出力先 docs）: Markdown 正本（例 `docs/overview.md`）と standalone HTML（例 `docs/overview.html`）。
  - 合成利用: `.claude/skills/code-explorer/`（横断精査の主力入力）。
  - 任意・ad hoc 利用: `.claude/skills/architecture-diagram/`（実在）、`diagram-design`（現リポジトリ未配置・外部/プラグイン提供の参照資産・style-guide 参考流用）。
  - ビルド時のみ利用: `skill-creator`（雛形生成）。
  - 参照のみ: `docs/architecture.md`（How の束ね先）、`README.md`（補完関係）、`.claude/rules/05-docs.instructions.md`、`.claude/rules/02-prompt.instructions.md`。
- 既存機能への影響:
  - 既存スキル・エージェント・ライブラリコードへの変更はなく、本スキルは新規追加のみ。
  - `/spec-sync` は正本 Markdown の同期対象として `docs/` の概観ファイルを扱い得る（保守フロー）。
  - 既存の `code-explorer` / `architecture-diagram` / `diagram-design` の挙動・契約は変更しない（合成・参照のみ）。

## 6. 用語定義
| 用語 | 定義 |
|------|------|
| 包括概観 | 目的・背景・できることを技術者・非技術者の双方が 1 資料で読める形に網羅した、リポジトリ横断の概観資料。 |
| SoT（Single Source of Truth） | ある情報について唯一正とみなす出所。本スキルでは `docs/` 配下の Markdown 正本を概観の SoT とする。 |
| 正本 | SoT として扱う Markdown ファイル（例 `docs/overview.md`）。Why/What を担当し、編集・更新の対象となる現在仕様。 |
| 派生物 | 正本から単方向で生成される regenerable な成果物。本スキルでは consulting-style standalone HTML を指す。 |
| 合成（compose） | 既存スキル/資産の出力や型を入力として組み合わせて用いること。実装の複製（コピー）を伴わない再利用。 |
| ベンダリング | 外部 OSS のコードを自リポジトリへ取り込むこと。本スキルでは禁止し、参考に留める。 |
| 05-docs | `.claude/rules/05-docs.instructions.md`。`docs/` を現在仕様の SoT として運用する Spec 駆動規約。 |
| 02-prompt | `.claude/rules/02-prompt.instructions.md`。skills/agents プロンプトの設計・記述・構造規約。 |
| code-explorer | 既存スキル `.claude/skills/code-explorer/`。コードベースを横断探索しコンテキストレポートを出力する。本スキルの精査主力。 |
| architecture-diagram | 既存スキル `.claude/skills/architecture-diagram/`。ダーク/技術/インフラ寄りの図を生成する。本スキルでは常用せず明示要求時のみ ad hoc 利用。 |
| diagram-design | consulting 適性のある図型（quadrant / pyramid / layer-stack / timeline / venn 等）と style-guide を持つ図設計資産。現リポジトリの `.claude/` 配下には未配置で、プラグイン/グローバル提供の外部参照資産として扱う。本スキルでは降格・任意で明示要求時のみ ad hoc 利用。 |
| consulting-style standalone HTML | 表紙・エグゼクティブサマリ・章レイアウトと editorial 視覚部品を持ち、inline CSS/SVG で自己完結する「綺麗なコンサル資料」品質の単一 HTML 成果物。 |
| editorial 視覚部品 | 統計コールアウト・できることグリッド・簡易の全体像シェマなど、視覚言語を統一するための内蔵パーツ。 |
| 基線 | 精査範囲の既定の出発点。本スキルでは `src` + `docs` + `README`。対話で拡縮する。 |
| 単方向生成 | 正本 → 派生物（Markdown → HTML）の一方向のみで生成し、派生物から正本へ逆流させない不変条件。 |
