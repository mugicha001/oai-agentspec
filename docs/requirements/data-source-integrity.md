# RAG / メモリ / ツールレスポンスの改竄防止方針（データ源インテグリティ）

## 1. 概要

エージェントが判断材料として読む取得系データ源（RAG 検索結果・会話メモリ・ツール返却値）の改竄・偽装に対する防御方針を、実装済みの内容ガードレール層（`oai-agentspec[guardrails]`）と既存の改竄防止機構 `lockdown` の責務境界の上で確定する。成果物は ADR 1 本・docs へのパターン記載・example の 3 点であり、`src/` へのコード追加は行わない。取得時点の正解（ベースライン）の生成・保管・保護は利用者責務とし、lib は既存の宣言面（`Boundary.TOOL_OUTPUT` 境界へ detector を装着する経路）と、適用パターン・検知時の既定挙動・スコープ境界の明文化のみを担う。

## 2. 機能要件

### FR-1: データ源インテグリティのスコープ判断を ADR として確定する
- ユーザーストーリー: oai-agentspec のメンテナとして、データ源インテグリティに対する lib の責務境界を ADR 1 本へ確定したい。なぜなら「どこまでを lib が受け持ち、どこからが利用者責務・レイヤ外か」が 1 箇所に定まらない限り、同種の要望が判断のやり直しを繰り返し要求するからだ。
- 受け入れ基準:
  - [ ] WHEN 設計判断が記録される THEN `docs/adr/0040-data-source-integrity-scope.md` が存在し、`- Status: accepted` 行と `## Context` / `## Decision` / `## Consequences` / `## Confirmation` の 4 節をすべて持つ（テンプレートは `.claude/skills/architect/reference/adr.md`）
  - [ ] WHEN 採番される THEN `0040` は既存 `docs/adr/` の最大番号の次であり、既存 ADR のファイル名・本文は変更されない（`git status --porcelain docs/adr/` の出力が新規 ADR 1 ファイルの untracked 行のみで、`M` / `D` / `R` で始まる行を含まないこと。untracked の新規ファイルは `git diff` に現れないため差分表示コマンドでは判定しない）
  - [ ] WHEN `## Decision` が書かれる THEN 次の 4 判断を各 1 段落以上で明記する: (a) `src/` への新規コードを追加せず既存 guardrails 層と利用者 DI で満たす、(b) 取得時点のベースライン（ハッシュ等）の生成・保管・保護は利用者責務とする、(c) RAG / メモリの検知は `Boundary.TOOL_OUTPUT` を基本の適用境界とする、(d) ライブなツールレスポンスの改竄検知はスコープ外とする
  - [ ] WHEN `## Decision` が Issue の受け入れ基準への着地を記述する THEN 以下の対応表を含み、各行の「着地」欄が「満たす」または「スコープ外として明記する」のいずれかで確定している（本表を Issue 受け入れ基準の着地の唯一の記載箇所とし、他の節・他ファイルへ複製しない）

    | Issue の受け入れ基準 | 着地 | 根拠 |
    |---|---|---|
    | RAG 検索結果・会話メモリの取得時点での想定外改変を検知できる方針 | 満たす | 取得をツールとして宣言し `Boundary.TOOL_OUTPUT` へベースライン照合 detector を装着するパターンを FR-2 で docs に記載する |
    | ツール（API/DB）レスポンスの改変・偽装を検知できる方針 | 一部スコープ外として明記する | 静的資産を返す取得系ツールは上段と同一パターンで検知できる。毎回値が変わるライブレスポンスは比較対象のベースラインが原理的に存在しないため FR-4 でスコープ外と明記する |
    | 検知時の既定挙動（fail-closed で拒否する等）の明文化 | 満たす | FR-3 で `on_trip="raise"` を改竄検知の既定として docs に明文化する |
    | 既存 lockdown との関係（統合 / 併存 / 独立）の設計判断 | 満たす | FR-5 で「併存」と結論し根拠を記述する |
    | 導入コスト（鍵管理・trust level 分割等）とトレードオフ | 満たす | FR-6 で ADR の `## Consequences` に整理する |

  - [ ] WHEN `## Confirmation` が書かれる THEN 本決定の強制手段が「docs 記載 + example の実在」であり自動テストによる強制手段を持たないことを明記し、`docs/QUALITY-GUARANTEES.md` へ行を追加しない理由（機械検証できる不変条件を新設しないため）を 1 文で述べる
  - [ ] IF ADR 本文に PR 番号・Issue 番号・AI モデル名・絵文字が含まれる THEN 受け入れない。検査は 2 段で行う: 語と番号は `grep -nE "#[0-9]+|Claude|Anthropic|GPT" docs/adr/0040-data-source-integrity-scope.md` が 0 件であること、絵文字は Unicode 範囲を検査する 1 行スクリプトが空リストを出力すること（grep のパターンは絵文字を検査しないため別手段を用いる）

### FR-2: RAG 検索結果・会話メモリの改竄検知パターンを docs に記載する
- ユーザーストーリー: oai-agentspec の利用者（RAG / メモリ を組み込む実装者）として、取得結果が取得時点から改変されていないことを検査する装着方法を docs から 1 本の手順として読み取りたい。なぜなら既存の guardrails 層に必要な部品が揃っていても、どの境界へ何を装着すれば改竄検知になるかが示されない限り利用者が自力で設計し直す必要があるからだ。
- 受け入れ基準:
  - [ ] WHEN パターンが記載される THEN `docs/integrity.md` に新規章「## データ源インテグリティ（RAG / メモリ / ツール出力）」が追加され、`grep -n "TOOL_OUTPUT" docs/integrity.md` が 1 件以上ヒットする
  - [ ] WHEN 当該章が適用境界を示す THEN RAG 検索・メモリ取得を `function_tool` として宣言し、その出力を `Boundary.TOOL_OUTPUT`（`runtime/guardrails/types.py` の `Boundary` メンバ）で検査することを基本パターンとして記述する
  - [ ] WHEN 当該章が装着経路を示す THEN 2 経路を区別して記述する: ツール定義時に宣言する場合は `function_tool(..., tool_output_guardrails=[tool_guardrail(detector, on="output", on_trip="raise")])`、既存ツールへ後付けする場合は `guard_tool(tool, output_detector=detector, on_trip="raise")`
  - [ ] WHEN 当該章が detector の実装パターンを示す THEN ベースライン（取得時点の sha256 等）と返却テキストのハッシュを突き合わせる述語を `predicate_detector` で包む形の最小コードを掲載する。掲載コードの import は NFR-2 の import 規律に従い、分類 (3)（`agents` の公開シンボル）を用いない
  - [ ] WHEN 当該章がベースラインの粒度を示す THEN 基本パターンを「1 回のツール呼び出しが単一資産（1 ドキュメント / 1 メモリレコード）を返す形」と定義し、その形では返却テキスト全体のハッシュと資産単位のベースラインが 1 対 1 で照合できることを明記する
  - [ ] WHEN 当該章が複数チャンク連結の形（RAG 検索が複数ヒットを 1 つのテキストへ連結して返す形）に触れる THEN detector が受け取るのは連結後のテキスト全体であり資産単位のベースラインをそのまま突き合わせられないことを明記し、次の 2 経路を成立条件つきで記述したうえで、経路 (b) を RAG における推奨形と位置づける 1 文を添える
    - 経路 (a) 連結後テキストそのものに対するベースライン。適用は定型問い合わせ（クエリ集合が有限かつ事前に確定している構成）に限る旨を明記し、成立に同時に必要な 3 条件を併記する: 第 1 にクエリ集合が有限かつ事前確定であること（ベースラインは事前に用意するものであり、自由文入力の RAG はクエリ空間が非有界で原理的に用意できない）、第 2 に検索インデックスが不変であること（文書の追加・削除・再インデックスは同一クエリでもヒット集合を変え、改竄でないのに全ベースラインを一斉に失効させる。`on_trip="raise"` と組み合わせると正常系が停止するため、FR-6 が挙げる「ベースラインの失効と更新」のコストと同じ論点である旨を参照で結ぶ）、第 3 に取得順序が決定的であること（近似最近傍探索・スコア同点時のタイブレーク・シャード分割の並列取得は順序を揺らす）
    - 経路 (b) ツールが doc_id とダイジェストを含む構造化出力（JSON 等）を返し、detector が出力をパースして資産単位で照合する構成。上記の条件を必要としないため、RAG の通常形（自由文クエリ・可変インデックス）にはこちらを用いる
  - [ ] WHEN 当該章が登録簿経由の宣言に触れる THEN 次の 3 点を記述する: (a) 登録は登録簿インスタンス経由の facade `registry.tool_guardrail(detector, on="output", on_trip="raise", name=<登録名>, severity=Severity.CRITICAL)` で行い、境界は `on` から導出される（`on="output"` で `Boundary.TOOL_OUTPUT`）、(b) 装着は `registry.get(<登録名>)` で実体を取り出し `function_tool(..., tool_output_guardrails=[...])` へ渡す、(c) 登録名を `AgentSpec.guardrails` へ渡せないこと、および専用フィールド `input_guardrails` / `output_guardrails` が agent 境界専用でツール出力には効かないこと（`output_guardrails` は名称からツール出力にも効くと誤解されやすい）を 1 行で述べ、根拠（ツール境界は Agent 実体へ振り分けられず `ValueError` になること）は `docs/architecture.md` の内容ガードレール節への参照に委ねる
  - [ ] IF 当該章がツール境界の装着を「専用フィールド経路と並ぶ選択肢」と表現する THEN 受け入れない
  - [ ] WHEN 当該章が lib 側と利用者側の責務分界を示す THEN lib 側の責務を「宣言面（境界 enum・ファクトリ・登録簿）の提供」「本パターンと既定挙動の明文化」「example の提供」の 3 点に限定して列挙し、ベースラインの生成・保管・鍵管理・失効を利用者責務として明記する
  - [ ] IF 本要件で `docs/integrity.md` と `docs/architecture.md` の双方へ同一内容を新たに書き足す THEN 二重記述として受け入れない。本基準が禁じるのは本要件で新たに両ファイルへ書き足す内容の重複であり、既存の `docs/architecture.md` 記述と重なる事実（ツール境界が名前参照の対象外であること）は対象外とする。当該事実の `docs/integrity.md` 側での書き方は登録簿経由の基準 (c) が定める。`docs/architecture.md` 側には `docs/integrity.md` の当該章への参照を 1 行だけ追加する
  - [ ] IF 利用者がベースラインを保持していない THEN 本パターンは適用できないことを制約として当該章に明記する

### FR-3: 検知時の既定挙動を fail-closed として明文化する
- ユーザーストーリー: oai-agentspec の利用者として、改竄を検知したときにエージェントが処理を続行しないことを宣言で保証したい。なぜなら改竄が疑われるデータを注釈付きでモデルへ返して続行すると、検知そのものが防御にならないからだ。
- 受け入れ基準:
  - [ ] WHEN FR-2 の章が既定挙動を記述する THEN 改竄検知用途では `on_trip="raise"`（中断）を既定として使うことを明記し、ファクトリの既定値が `"reject"`（注釈付き返却で続行）であるため明示指定が必要である旨を同じ段落に書く
  - [ ] WHEN 当該章が深刻度を記述する THEN 登録簿経由で宣言する場合の推奨値を `Severity.CRITICAL` と明記する
  - [ ] IF `on_trip="reject"` または `"allow"` を選ぶ THEN いずれも実行を中断しないため fail-closed ではないことを明記し、両者の差（`"reject"` は出力を注釈メッセージへ差し替えて続行するためモデルは元データを読まない / `"allow"` は出力をそのまま通すためモデルが当該データを読む）を書き分けたうえで、監査ログ収集など検知結果を続行前提で扱う用途に限る旨を記述する
  - [ ] WHEN example が提供される THEN FR-7 の example が `on_trip="raise"` を使い、改竄検出時に実行が中断することを実行結果として示す

### FR-4: ライブなツールレスポンスの改竄検知をスコープ外として根拠付きで明記する
- ユーザーストーリー: oai-agentspec のメンテナとして、API / DB のライブレスポンス改竄検知を lib が扱わない理由を明文化したい。なぜなら根拠を残さずに未対応とすると、同じ要望が繰り返し起票され、原理的に成立しない機構を実装する圧力になるからだ。
- 受け入れ基準:
  - [ ] WHEN ADR の `## Decision` がスコープ外を宣言する THEN ライブレスポンス（呼び出しごとに値が変わることが正常な応答）には比較すべきベースラインが原理的に存在しないことを根拠として記述する
  - [ ] WHEN ADR がスコープ外領域の残る防御を示す THEN トランスポート層の完全性（TLS 等・lib のレイヤ外）と内容ガードレール（実装済みの `Boundary.TOOL_OUTPUT` 検査・改竄検知ではなく内容の妥当性検査）の 2 つを列挙し、それぞれが lib のレイヤで表現できるか否かを明記する
  - [ ] WHEN `docs/integrity.md` の節「## 守れる範囲・守れない範囲」の表が更新される THEN 既存の 3 列形式（`| 範囲 | 守れる | 守れない |`。理由欄は存在しない）を保ったまま「データ源（取得系）の改竄」の行が 1 行追加され、「守れる」列に静的資産のベースライン照合（`Boundary.TOOL_OUTPUT` 経路）が、「守れない」列にライブレスポンス（比較対象のベースラインが原理的に存在しないため）が入る（表の位置は行番号ではなく節見出しで参照する）
  - [ ] IF 取得系ツールが静的資産（ドキュメント本文・スナップショット・確定済みメモリ）を返す THEN FR-2 のパターンが適用できることを、ライブレスポンスとの境界として同じ表または直後の段落に明記する

### FR-5: 既存 lockdown との関係を「併存」として整理する
- ユーザーストーリー: oai-agentspec の利用者（runtime 起動側）として、`lockdown` とデータ源インテグリティのどちらが何を守るのかを 1 つの表から判断したい。なぜなら両者を混同すると `lockdown` を呼んだだけで取得データまで守られていると誤認するからだ。
- 受け入れ基準:
  - [ ] WHEN ADR の `## Decision` が関係を結論づける THEN 「併存」（統合でも独立でもない）と明記し、根拠として次の 3 点を挙げる: (a) `lockdown` は起動時 / 明示発火時にディスク上の静的資産を検証する build-time 寄りの機構であるのに対し、データ源インテグリティは 1 ターンごとのツール出力に対する検査であり発火タイミングが異なる、(b) `lockdown` はコア層（`src/oai_agentspec/integrity.py`・標準 lib のみ依存）にあり、guardrails は opt-in extra の実行寄り層にあるため、統合するとコアから実行寄り層への依存辺ができ単方向依存とコア `__all__` の分離契約に反する、(c) 本要件は新規コードを追加しないため統合の実装面が存在しない
  - [ ] WHEN `docs/integrity.md` の当該章が両機構の関係を示す THEN 対比は 3 列の表 1 つ（機構名 / 守る対象 / 発火タイミング）に留め、各セルは 1 行以内とする。`lockdown` 行の「守る対象」セルは総称（ディスク上の静的資産と宣言グラフ）で書き、内訳の列挙は行わず、詳細は既存節への参照 1 行で置き換える
  - [ ] WHEN 当該章が対比表を置く THEN 表の直後に、両機構を統合せず併存とする理由（`lockdown` はコア層・guardrails は opt-in extra という層の違いにより、統合するとコアから実行寄り層への依存辺ができ単方向依存とコア `__all__` の分離契約に反すること）を 1 行で要約する。あわせて `lockdown` の `checks` へ寄せる案を採らない理由を別の 1 行で述べる（`checks` が受け取る `IntegrityCheck` は `Callable[[], None]` で引数を取らず、発火も `lockdown` の呼び出し時に限られるため、1 ターンごとのツール出力を受け取れない。層の違いとは別の根拠である）。この 2 行は表のセルではなく表の直後の地の文であり、基準 2 の「各セル 1 行以内」には抵触しない
  - [ ] IF `lockdown` の公開 API・処理段の内容・例外階層を変更する提案が含まれる THEN 本要件のスコープ外として受け入れない（`git status --porcelain src/oai_agentspec/integrity.py` が空）

### FR-6: 導入コストとトレードオフを ADR に整理する
- ユーザーストーリー: oai-agentspec の利用者（セキュリティ設計者）として、このパターンを採用したときに自分が負担する設計要素と、lib 側に残る制約を事前に把握したい。なぜならベースライン管理を利用者責務とする方針は利用者側に新たな設計課題を発生させ、適用境界をツール境界に限る方針は宣言的な装着面に制約を残すからだ。
- 受け入れ基準:
  - [ ] WHEN ADR の `## Consequences` が書かれる THEN コストとして少なくとも以下の項目を列挙し、各項目に「lib が肩代わりしない理由」または「解消しない理由」を 1 文添える（本リストを本要件におけるコスト列挙の唯一の記載箇所とし、件数を他所へ複製しない）: ベースラインの生成と保管、ベースライン自体の保護（読み取り専用配置・署名 / HMAC を用いる場合の鍵管理）、情報源ごとの trust level 分割（信頼できる取得元と untrusted 取得元の宣言分け）、ベースラインの失効と更新（元データが正当に更新された場合の運用。検索インデックスの更新が同一クエリのヒット集合を変える場合を含む）、宣言面のギャップ（下の基準で規定する）
  - [ ] WHEN `## Consequences` が宣言面のギャップを記述する THEN 次の内容を 1 項目として立てる: 適用境界をツール境界に限る帰結として、`AgentSpec` の専用フィールド（`input_guardrails` / `output_guardrails`）も名前参照（`guardrails`）もツール境界へ届かず、利用者は `function_tool` の引数へ guardrail を手で渡す必要がある。これは宣言でエージェントを構成するという本ライブラリの価値提案の外側に本パターンが位置することを意味し、本要件では新規コードを追加しない方針の下でこのギャップを解消しない
  - [ ] WHEN `## Consequences` がトレードオフを書く THEN 「lib へベースライン管理機構を持ち込む案」を却下案として記録し、却下理由（鍵管理と保存先の選択が利用者環境依存であること、build-don't-run の逸脱を新たに増やすこと）を記述する
  - [ ] WHEN ベースライン自体の信頼境界に触れる THEN 攻撃者がベースラインと対象データの双方を同時に書き換えられる環境では検知保証が失効することを明記し、`docs/integrity.md` の manifest 信頼境界の節と同じ前提に立つ旨を参照で示す

### FR-7: 実行可能な example を追加する
- ユーザーストーリー: oai-agentspec の利用者として、docs のパターンをそのまま動かせる形で確認したい。なぜなら guardrail の装着位置と trip 時の中断挙動は、コードを実行して初めて意図どおりか判断できるからだ。
- 受け入れ基準:
  - [ ] WHEN example が追加される THEN `examples/guardrails/09_data_integrity_detector.py` が存在し、既存 examples の命名・番号採番（`examples/guardrails/` 内の現在の最大番号 `08_canary_run_scoped.py` の次）に従う
  - [ ] WHEN example が構成される THEN ベースライン一致の正常系と、取得結果を改変した改竄系の 2 経路を同一スクリプト内で実行し、改竄系で `on_trip="raise"` により実行が中断することを標準出力に示す
  - [ ] WHEN example が import を書く THEN NFR-2 の import 規律に従い、分類 (3)（`agents` の公開シンボル `Runner` / `ToolOutputGuardrailTripwireTriggered`）を実行と中断の捕捉のために併用する（既存 `examples/guardrails/06_tool_output_guardrail.py` と同じ作法）
  - [ ] WHEN example が API へ接続する THEN 既存 examples と同じく `examples/_shared` 経由でモデルを構成し、スクリプト内に 60 秒の watchdog（経過時間で強制切断し非ゼロ終了する）を持つ
  - [ ] WHEN example を検証する THEN Bash の `timeout 90` を併用して 1 回実行し、正常系と改竄系の両経路が期待どおり終端することを確認する（スクリプト内 60 秒 < 実行側 90 秒の 2 層で上限を掛ける）
  - [ ] WHEN lint を実行する THEN 当該ファイルに対する `uv run ruff check` / `uv run ruff format --check` が緑である（`examples/` は CI の lint 対象パス外のため手動検証とし、実行コマンドと結果を検証記録へ残す。検証記録の置き場は実装 PR 本文の検証セクションとする）
  - [ ] IF docs から example を参照する THEN `docs/integrity.md` の当該章から相対パスで 1 箇所だけリンクし、example の内容を docs 本文へ再掲しない

## 3. 非機能要件

### NFR-1: 保守性（build-don't-run 不変条件と SDK 隔離の維持）
- 要件: 本要件は `src/` へコードを追加しないため、`./CLAUDE.md` および `docs/architecture.md` に列挙された build-don't-run 逸脱の一覧へ新たな項目を追加しない。ハッシュ再計算ループ・ベースライン照会の再試行・`Runner` 代行のいずれも lib へ持ち込まない。改竄検知の実行は SDK ネイティブの tool guardrail フックが担い、lib は既存ファクトリによる接着のみを提供する。`src/` を変更しないことの帰結として、SDK 隔離（`src/` 配下について、`from agents` / `from openai` の import を `_adapters/` 配下以外へ持ち込まない）も維持される。`examples/` は `src/` の外にあり本規律の適用対象外で、`agents` の公開シンボルを使ってよい（出所の規律は NFR-2 が定める）。
- 計測基準: `git status --porcelain` の全出力行が指すパスが `docs/` と `examples/` 配下に限られること（`src/` 配下のパスを含む行が 0 行。untracked の新規ファイルは `git diff` に現れないため差分表示コマンドでは判定しない）。`docs/architecture.md` が変更対象に含まれる場合は `git diff docs/architecture.md` を確認し、build-don't-run 逸脱の列挙節に差分が現れないこと。SDK 隔離は上記の `src/` 無変更から従属して成立するため、独立した計測は置かない。

### NFR-2: 保守性（掲載コードの import 規律）
- 要件: 本節を import 規律の唯一の定義箇所（SoT）とし、他の要件（FR-2 / FR-7）は本節を参照する。docs 掲載コードと example が使うシンボルの出所を次の 3 分類に限る。(1) lib のシンボルは `oai_agentspec` のコア `__all__` と `oai_agentspec.runtime.guardrails.__all__` のメンバ、(2) 標準ライブラリ（`hashlib` 等）、(3) `agents` の公開シンボル（実行と中断の捕捉に必要な `Runner` / `ToolOutputGuardrailTripwireTriggered`）。分類 (3) は example に限り用いてよく、docs 掲載コードには用いない。ツール定義は `agents` からではなくコア公開シンボルの `function_tool` を使う（既存 examples の作法に合わせる）。
- 計測基準: docs 掲載コードと example の import 行をすべて列挙し、各行が上記 3 分類のいずれに属するかを対応付けて確認する。分類 (3) を含む行が docs 掲載コードに現れないことを併せて確認する。lib シンボルについては、コア `__all__` と guardrails 窓口の `__all__` の和集合に含まれることを名前の突合で確認する。結果は実装 PR 本文の検証セクションへ記録する。

### NFR-3: セキュリティ（fail-closed 既定）
- 要件: docs が改竄検知用途で提示する既定は中断（`on_trip="raise"`）とし、続行系（`"reject"` / `"allow"`）を既定として提示しない。検知後に同一ターンでモデルが当該データを読み得る構成を、改竄検知の推奨形として記載しない。
- 計測基準: `docs/integrity.md` の当該章および `examples/guardrails/09_data_integrity_detector.py` のコード片における `on_trip=` の実指定がすべて `"raise"` であること（`grep -n "on_trip" docs/integrity.md examples/guardrails/09_data_integrity_detector.py` の結果から、`on_trip=` の形で値を与えている行のみを対象とする。FR-3 の受け入れ基準が要求する説明文中の `"reject"` / `"allow"` への言及は対象外）。

### NFR-4: 保守性（掲載内容と実装の一致）
- 要件: docs 掲載コードと example が参照する公開シンボル・シグネチャ・既定値が、実装の現在値と一致していること。文書だけが先に古くなる状態、および実装上成立しない経路を記載する状態を作らない。
- 計測基準: 次の対象について `inspect.signature` および enum メンバ集合と突き合わせ、記載と一致することを確認する（結果を実装 PR 本文の検証セクションへ記録する）。最終項目の後半のみ署名突合では確かめられないため、根拠を出典参照へ切り替える。
  - `oai_agentspec.runtime.guardrails.tool_guardrail(detector, *, on, on_trip="reject", name=None)`
  - `oai_agentspec.runtime.guardrails.guard_tool(tool, *, input_detector=None, output_detector=None, on_trip="reject")`
  - `GuardrailRegistry.tool_guardrail(detector, *, on, on_trip="reject", name, labels=None, severity=None)`（`self` を除く。未束縛のメソッドへ `inspect.signature` を適用すると第 1 引数に `self` が現れるため、突合時はこれを除いて比較する。`name` はキーワード必須・既定値なし）
  - `oai_agentspec.runtime.guardrails.predicate_detector(predicate, *, reason=None)`
  - `Detection(triggered, reason=None, info=None)` のフィールド構成
  - `Boundary` のメンバ集合（INPUT / OUTPUT / TOOL_INPUT / TOOL_OUTPUT）と `Severity` のメンバ集合（LOW / MEDIUM / HIGH / CRITICAL）
  - `AgentSpec.guardrails` の型が `list[str]` であること（登録名の参照であり実体を受け取らない）。ツール境界の登録名を渡すと `ValueError` になることは署名突合と enum 突合では確かめられない（実際に登録して build する必要がある）ため、本項目のみ出典参照で根拠を示す: `src/oai_agentspec/registry.py` の `_TOOL_BOUNDARIES` 判定と同ファイルの拒否メッセージ、および `docs/architecture.md` の内容ガードレール節

### NFR-5: 性能（検知コストの境界）
- 要件: docs が提示する detector はツール出力テキスト長 n に対して O(n) のハッシュ計算 1 回とベースライン照会 1 回に留め、ネットワーク往復・モデル呼び出しを含めない。ベースライン照会先は利用者が渡すインメモリのマッピングまたは利用者実装の同期関数とし、docs のパターンでは追加の外部サービス接続を前提としない。
- 計測基準: 本要件は性能ベンチマークを採らず構造的計測で担保する。実施者は実装 PR のレビュー担当（`/coding` の reviewer）とし、docs 掲載コードと example の detector 本体に、ハッシュ計算以外のループ・`await` を伴う外部呼び出し・モデル呼び出しが含まれないことを確認して、確認結果を実装 PR 本文の検証セクション（FR-7 の検証記録と同一の置き場）へ記録する。

## 4. 制約事項

- 技術的制約:
  - 成果物は ADR 1 本・`docs/` への記載・`examples/` への追加のみとし、`src/` への新規コードを追加しない。
  - ライブなツールレスポンス（呼び出しごとに値が変わることが正常な API / DB 応答）の改竄検知はスコープ外とする。比較すべきベースラインが原理的に存在しないため、lib のレイヤでは検知として表現できない。
  - トランスポート層の完全性（TLS 等）は lib のレイヤ外であり、本要件では利用者およびインフラの責務として扱う。
  - ベースライン（取得時点のハッシュ等）の生成・保管・保護・失効・鍵管理は利用者責務とする。lib はベースラインの保存機構・署名機構を提供しない。
  - 攻撃者がベースラインと対象データの双方を同時に書き換え可能な環境では検知保証が失効する（`docs/integrity.md` の manifest 信頼境界と同じ前提）。
  - ツール境界の guardrail は宣言フィールド（`AgentSpec.input_guardrails` / `output_guardrails`）にも名前参照（`AgentSpec.guardrails`）にも載らない。名前参照へ渡すと `ValueError` になるため、装着経路は `function_tool` の `tool_output_guardrails` 引数（および `guard_tool` による後付け）に限られる。
  - RAG / メモリ取得をツールとして宣言しない構成（SDK 組み込みの検索機構をエージェントが直接使う等）では `Boundary.TOOL_OUTPUT` 経路が成立しないため、本パターンの適用対象外とする。
  - 複数チャンクを連結して返す RAG ツールでは、資産単位のベースラインをそのまま連結後テキストへ突き合わせられない。推奨形は doc_id とダイジェストを含む構造化出力による資産単位の照合とする。連結後テキストに対するベースラインは、クエリ集合の事前確定・検索インデックスの不変・取得順序の決定性が同時に成立する定型問い合わせに限って適用できる。
  - `lockdown`（`src/oai_agentspec/integrity.py`）の公開 API・処理段・例外階層は変更しない。
  - モデル・学習データへのポイズニング対策は本ライブラリが学習を持たないためレイヤ外とする。
  - 依存パッケージの初回混入等のサプライチェーン防御はローカル SCA（`docs/security-scanning.md`）の責務としスコープ外とする。
  - エージェントの意思決定・実行そのもののガバナンス（何を実行してよいかの許可・拒否）は `runtime/governance` の責務でありスコープ外とする。
  - ADR は append-only で運用し、採番は既存 `docs/adr/` の最大番号 + 1（`0040`）とする。ファイル名に Issue 番号を含めない。
  - docs は現在仕様の SoT として記述し、履歴記述・PR / Issue 番号への言及・AI モデル名を含めない。
  - `examples/` は実 API（Azure / OpenAI）を使用し、従量課金への接続となるためスクリプト内 watchdog（60 秒）と実行側 timeout（90 秒）の二重上限を課す。
- ビジネス制約:
  - 既存利用者の運用を破壊しない。本要件の成果物は文書と example のみであり、既存構成の挙動を変更しない。
  - 重い専門検知（PII / モデレーション / 注入検知サービス）を lib 非同梱で利用者 DI とする guardrails 層の既存方針を踏襲し、ベースライン照合ロジックも同様に利用者 DI とする。
  - 宣言面のギャップ（ツール境界へ宣言フィールドが届かないこと）は本要件では解消せず、コストとして ADR に記録するに留める。解消は新規コードの追加を伴うため別途の設計判断を要する。
  - 機械検証できる不変条件を新設しないため `docs/QUALITY-GUARANTEES.md` へは行を追加しない（その判断理由を ADR の `## Confirmation` に記す）。

## 5. 影響範囲

- 関連コンポーネント:
  - 新規 `docs/adr/0040-data-source-integrity-scope.md` — データ源インテグリティのスコープ判断（追加コードなし / ベースラインは利用者責務 / 適用境界は `Boundary.TOOL_OUTPUT` / ライブレスポンスはスコープ外 / `lockdown` とは併存）と、Issue 受け入れ基準への着地表、導入コスト（宣言面のギャップを含む）とトレードオフ、却下案を記録する。
  - `docs/integrity.md` — 新規章「## データ源インテグリティ（RAG / メモリ / ツール出力）」を追加し、適用境界・装着経路（`tool_guardrail` / `guard_tool` / 登録簿 facade + `registry.get`）・detector の実装パターン・ベースラインの粒度（単一資産 / 複数チャンク連結の 2 経路と成立条件）・fail-closed 既定・責務分界・`lockdown` との対比表と統合しない理由の 1 行要約・example へのリンクを記載する。あわせて節「## 守れる範囲・守れない範囲」の表へ 1 行追加する。既存章の記述は変更しない。
  - `docs/architecture.md` — 内容ガードレール節へ `docs/integrity.md` の当該章を指す参照を 1 行追加する（二重記述を作らないため詳細は書かない）。
  - 新規 `examples/guardrails/09_data_integrity_detector.py` — ベースライン照合 detector を `Boundary.TOOL_OUTPUT` へ装着し、正常系と改竄系の 2 経路を実行して `on_trip="raise"` による中断を示す。
  - `src/oai_agentspec/runtime/guardrails/` — 参照のみ（変更なし）。`Boundary` / `Severity` / `GuardrailSpec` / `GuardrailRegistry`（`tool_guardrail` facade と `get`）/ `tool_guardrail` / `guard_tool` / `predicate_detector` / `Detection` を docs と example から使う。
  - `src/oai_agentspec/registry.py` — 変更なし（参照のみ）。ツール境界の名前参照が `ValueError` になる挙動を NFR-4 の出典参照として引用する。
  - `src/oai_agentspec/integrity.py` — 変更なし（`lockdown` とは併存）。
  - `src/oai_agentspec/runtime/governance/` — 変更なし（実行可否のポリシー強制は別レイヤ）。
- 既存機能への影響:
  - `src/` を変更しないため、既存の公開 API・`__all__`・既存利用者の挙動はすべて不変。
  - 既存テストスイートは無修正で緑のまま維持され、テストの追加・削除は行わない。
  - `docs/QUALITY-GUARANTEES.md` の台帳行は増減しない。
  - `docs/integrity.md` の既存章（最小起動コード・公開 API・例外階層・manifest 信頼境界・Out of Scope 等）の記述内容は変更せず、新規章の追加と「守れる範囲・守れない範囲」表への行追加のみを行う。

## 6. 用語定義

| 用語 | 定義 |
|------|------|
| データ源インテグリティ | エージェントが判断材料として読む取得系データ（RAG 検索結果・会話メモリ・ツール返却値）が、取得時点の内容から改変されていないこと。本要件の対象領域。 |
| ベースライン | 取得時点の正解を表す比較対象。照合の単位は検査対象の形に従い、単一資産を返すツールでは資産単位（doc_id 等の識別子とダイジェストの対応）、複数チャンクを連結して返すツールでは構造化出力中の資産単位（推奨形）または連結後テキスト単位（定型問い合わせに限る）となる。生成・保管・保護・失効は利用者責務。 |
| ライブレスポンス | 呼び出しごとに値が変わることが正常な API / DB の応答。比較すべきベースラインが原理的に存在しないため、本要件では改竄検知の対象外。 |
| 静的資産（取得系） | 取得時点から内容が変わらない前提のデータ（ドキュメント本文・スナップショット・確定済みメモリ等）。ベースライン照合が成立する対象。 |
| 定型問い合わせ | クエリ集合が有限かつ事前に確定している検索構成。連結後テキストへベースラインを持つ経路が適用できる唯一の形であり、成立には検索インデックスの不変と取得順序の決定性も同時に必要となる。 |
| 適用境界（Boundary） | ガードレールを適用する位置。`runtime/guardrails/types.py` の `Boundary` が agent 境界（INPUT / OUTPUT）とツール境界（TOOL_INPUT / TOOL_OUTPUT）を列挙する。本要件は `TOOL_OUTPUT` を基本とする。 |
| detector | テキストを受けて `Detection`（`triggered` / `reason` / `info`）を返す検知関数。本要件ではベースライン照合の述語を `predicate_detector` で包んだ純関数を指す。 |
| facade（登録簿） | `GuardrailRegistry` が持つ「生成 + 登録」を 1 呼び出しで行うインスタンスメソッド群。ツール境界では `GuardrailRegistry.tool_guardrail` が該当し、呼び出しは `registry.tool_guardrail(...)` の形で行い、境界を `on` から導出して登録し `GuardrailSpec` を返す。実体の取り出しは `registry.get(<登録名>)`。 |
| 宣言面のギャップ | ツール境界の guardrail が `AgentSpec` の専用フィールドにも名前参照にも載らず、`function_tool` の引数へ手で渡す必要がある非対称。本要件では解消せず ADR にコストとして記録する。 |
| on_trip | guardrail 発火時の挙動選択。`"reject"`（注釈付き返却で続行・ファクトリ既定）/ `"raise"`（中断）/ `"allow"`（通過）または `Detection` を受ける callable。本要件の改竄検知用途では `"raise"` を既定とする。 |
| fail-closed | 検証失敗時に処理を継続せず停止する設計方針。本要件では `on_trip="raise"` による中断を指す。 |
| lockdown | 稼働中のディスク上ファイル（プロンプトテンプレート / 配布物 / 任意パス）の sha256 照合・PEP 376 RECORD 照合と、`AgentRegistry` / `WorkflowGraph` の構造 freeze を行う既存の改竄防止機構。責務範囲の SoT は `docs/integrity.md`。 |
| 併存 | 2 つの機構が互いを呼び出さず、責務範囲と発火タイミングを分けたまま同一アプリ内で同時に使える関係。統合（一方が他方を内包する）でも独立（相互に無関係で対比も示さない）でもない。 |
| trust level 分割 | 情報源を信頼できる取得元と untrusted 取得元へ分けて宣言し、検査の強度を変える設計。本要件では利用者側の設計コストとして ADR に整理し、lib へは持ち込まない。 |
| 利用者 DI | 重い専門検知やベースライン照合など環境依存のロジックを lib へ同梱せず、利用者が関数・オブジェクトとして注入する既存方針。 |
| import 規律 | docs 掲載コードと example が使うシンボルの出所を NFR-2 が定める分類に限る規律。定義の SoT は NFR-2 であり、FR-2 と FR-7 は参照に留める。 |
| 検証記録 | 手動検証（example の実行・lint・シンボル突合・構造レビュー）の実施コマンドと結果の置き場。本要件では実装 PR 本文の検証セクションを指す。 |
| build-don't-run | lib は宣言・build-time 検証・薄い結線に徹し、実行は SDK に寄せるという本ライブラリの不変条件。本要件は逸脱を追加しない。 |
