# マネージド Fine-Tuning 統合（oai-agentspec[finetune] 公開 extra）

## 1. 概要

oai-agentspec に、OpenAI / Azure OpenAI のマネージド fine-tuning API（SFT / DPO）を利用者の宣言エージェント運用へ各操作を単一の公開関数呼び出しで組み込める Fine-Tuning 統合機能を、公開 optional extra（`oai-agentspec[finetune]`・`runtime/finetune`）として追加する。機能は (1) データセット整形（既存 dataset 資産 `EvalCase` / `OptimizeCase` / `DpoCase` や plain dict からの SFT chat 形式 / DPO preference 形式への純データ変換・持ち込み JSONL の検証・会話ログ（SDK `Session`）からの SFT データセット生成）、(2) ジョブ管理（学習ファイルのアップロード + ジョブ作成の submit・ステータス単発照会・完成 `model_ref` 取得・opt-in のタイムアウト必須待機）の 2 系統からなる。学習の実行エンジンは OpenAI / Azure プラットフォーム側にあり、lib は単発 API 呼び出しの薄い結線に徹する（build-don't-run 維持。唯一のポーリングループは opt-in の待機関数 1 つに隔離し ADR で正当化する。前例: ADR 0004 `fit_ml_estimator` / ADR 0012 `failsafe_call`）。

データセット整形は単一ターン（文字列 input）と複数ターン（messages 形式リスト input）の二形を受理し、ツール呼び出し（function calling）・assistant `weight`（loss masking）・content parts（vision）入りの学習例を非改変で透過する。`tools=` にはコア `ToolRegistry` 由来の `FunctionTool` 相当オブジェクトを直接渡せ、学習データと推論時のツール定義の一致を構造的に担保できる。

本機能は実行寄り層 `runtime/` 配下の一員（`runtime/finetune`）として追加し、llmops / lightning と同型の公開窓口・extra 未導入契約・SDK 隔離方針に従う。既存の `runtime/lightning` とは別トラックとして併存する（棲み分けの詳細は `docs/architecture.md` の「マネージド Fine-Tuning 統合」節を参照）。完成モデルは `model_ref`（モデル id・plain 文字列）として返し、利用者が従来どおり外部 DI（`AgentSpec` の model 流入）で使う。lib はモデル重み・学習データを保持せず、デプロイ・ホスティングを行わない。

## 2. 機能要件

### FR-1: SFT 用データ変換ヘルパ（既存 dataset 資産 / plain dict からの純データ変換）
- ユーザーストーリー: lib 利用者として、既存 dataset 資産（`EvalCase` / `OptimizeCase`）や plain dict のケース群を SFT 用 chat messages 形式へ 1 行で変換したい。なぜなら FT の最大摩擦である JSONL 整形を毎回手書きしたくないから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `to_sft_dataset(cases, ...)` へ `EvalCase` / `OptimizeCase` / plain dict の列を渡す THEN 各ケースの `input` と `expected_output`（plain dict の場合は `input_key` / `output_key` 引数でキー名を指定可）から `{"messages": [...]}` 形式の plain dict 列を含む結果（`DatasetBuildResult`）を返す。
  - [ ] WHEN ケースの `input` が文字列である THEN user 1 件のメッセージへ包む（単一ターン）。WHEN `input` が messages 形式のリストである THEN 各メッセージ dict を非改変で透過採用し（複数ターン）、`expected_output` を最終 assistant メッセージとして messages 末尾へ付す。
  - [ ] WHEN `expected_output` が文字列である THEN assistant 1 件のメッセージへ包む。WHEN `expected_output` が assistant メッセージ配列である THEN 非改変で透過採用する。
  - [ ] WHEN 利用者が system プロンプトを `system=` で付す THEN 利用者供給の system 文字列を messages 先頭へ付す。lib はプロンプト文字列・学習データを同梱しない（プロンプト非同梱原則と整合）。
  - [ ] IF `system=` と input リスト内 system メッセージの両方が存在する THEN 明確なエラーで報告し、暗黙マージ・暗黙置換をしない（skip オプションの対象外）。
  - [ ] WHEN input リストに `tool_calls` 付き assistant メッセージ（content 無しでも可）/ role `"tool"` メッセージが含まれる THEN 透過採用し、tool_calls・ツール応答の内容を解釈・改変しない。
  - [ ] WHEN input リストのメッセージに `weight` / parts 配列 content 等のフィールドが含まれる THEN 非改変で透過する（ヘルパはメッセージを改変しない）。`expected_output` から末尾付加する assistant メッセージには `weight` を付さない（既定の学習対象。weight を制御したい利用者は input リスト側で自メッセージに付す）。
  - [ ] WHEN 利用者が `tools=` / `parallel_tool_calls=` を指定する THEN 内容を解釈せずレコードレベル（レコード直下の `"tools"` / `"parallel_tool_calls"`）へ透過する。省略時（既定 None）はキー自体を出力しない。両引数は呼び出し単位で全レコード共通に付され、レコードごとに異なる tools を要するデータは持ち込み JSONL（FR-3 経由）で扱う。plain dict ケースのレコード別 `"tools"` キーはヘルパの読み取り対象外とする。
  - [ ] WHEN `tools=` に FunctionTool 相当オブジェクト（`name` / `params_json_schema` を持つ）が含まれる THEN FT の tools 定義形式（`{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}`）へ写像する（`description` 属性が無い、または値が `None` の場合は空文字として写像する）。plain dict の要素は解釈せず透過し、両者の混在を許す。IF dict でも FunctionTool 相当でもない要素が含まれる THEN 明確なエラーで報告する（skip オプションの対象外）。
  - [ ] IF ケースに `expected_output`（または指定キー）が欠落している、`input` が空リスト `[]` である、`input` / 出力側が文字列でもリストでもない型である、または出力側が空の assistant 配列 `[]` である THEN 当該ケースを明確なエラーで報告するか、明示指定の `skip_missing=True` 時のみ除外して除外件数を `DatasetBuildResult.skipped` に含める。欠落値を暗黙に補完しない。
  - [ ] IF `cases` が単一の dict / 文字列 / bytes である THEN 変換を開始せず明確なエラーを送出する（`skip_missing` の指定に依らない。1 件だけの場合も `[case]` のようにケースの列へ包む）。
  - [ ] WHEN input リストの各要素を検査する THEN 「dict であり `role` キーを持ち、かつ `content` または `tool_calls` のいずれかを持つ」ことのみを検査する（content の presence 判定は型非依存で parts 配列も content ありと数える）。role 値・構造の厳密検証は FR-3 の検証ヘルパに一元化する。
  - [ ] WHEN 変換を実行する THEN 純データ操作に徹し、SDK / 外部クライアント / ネットワークに触れない。
  - [ ] WHEN 利用者が結果型の `save(path)` を明示的に呼んだ場合のみ THEN JSONL ファイルとして当該パスへ書き出す（opt-in 書込）。既定は plain データ返却のみで lib は自動書き込みをしない。

### FR-2: DPO 用 preference 形式変換ヘルパ
- ユーザーストーリー: lib 利用者として、preferred / non_preferred の応答ペアを持つケース群を DPO の preference 形式へ 1 行で変換したい。なぜなら DPO 固有の JSONL 構造（input / preferred_output / non_preferred_output）を手書きしたくないから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `to_dpo_dataset(cases, ...)` へ入力・preferred 応答・non_preferred 応答を持つケース列（`DpoCase` または plain dict・キー名は `preferred_key` / `non_preferred_key` 引数で指定可）を渡す THEN OpenAI / Azure が受理する preference 形式（`input`（messages 構造）/ `preferred_output` / `non_preferred_output`）の plain dict 列を含む結果（`DatasetBuildResult`）を返す。
  - [ ] WHEN ケースの `input` が文字列である THEN user 1 件のメッセージとして `input.messages` へ包む。WHEN `input` が messages 形式のリストである THEN 非改変のまま `input.messages` へ透過する（複数ターン）。
  - [ ] WHEN `preferred_output` / `non_preferred_output` が文字列である THEN assistant 1 件のメッセージ配列へ包む。WHEN assistant メッセージ配列である THEN 非改変で透過採用する（preference 形式の出力側は assistant メッセージの配列である）。
  - [ ] WHEN system メッセージを含めたい THEN input の messages リスト先頭に含めて渡す（`to_dpo_dataset` は `system=` 引数を持たない）。
  - [ ] WHEN 利用者が `tools=` / `parallel_tool_calls=` を指定する THEN 内容を解釈せず `input.tools` / `input.parallel_tool_calls` へ透過する。FunctionTool 相当要素の写像・混在受理・不正要素のエラーは FR-1 と同一規則とする。
  - [ ] IF ケースに preferred / non_preferred のいずれかが欠落している、または FR-1 と同型の境界（空リスト・不正型・空配列）に該当する THEN 明確なエラーで報告するか、明示指定の `skip_missing=True` 時のみ除外して除外件数を結果に含める。lib は preference ペアを自動生成（評価スコアからの自動導出等）しない（ペアの供給は利用者責任）。
  - [ ] IF `cases` が単一の dict / 文字列 / bytes である THEN 変換を開始せず明確なエラーを送出する（`skip_missing` の指定に依らない。FR-1 と同一規則）。
  - [ ] WHEN 変換を実行する THEN 純データ操作に徹し、SDK / 外部クライアント / ネットワークに触れない。
  - [ ] WHEN 利用者が結果型の `save(path)` を明示的に呼んだ場合のみ THEN JSONL として書き出す（opt-in 書込・既定は返却のみ）。

### FR-3: 持ち込み JSONL の受理と検証
- ユーザーストーリー: lib 利用者として、自前で用意した学習用 JSONL を submit 前に検証したい。なぜならプラットフォームのジョブ失敗（課金・待ち時間発生後）より前に形式不備を検出したいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `validate_dataset(source, method=...)` へ JSONL（ファイルパスまたは plain dict 列）と method（`sft` / `dpo`）を渡す THEN 各行の JSON 妥当性・必須キーの存在・messages / preference 構造の形式を検証し、違反位置と理由の一覧（`DatasetViolation` の列）を含む plain な検証結果（`DatasetValidationReport`）を返す。`DatasetViolation.line` は、source がファイルパスの場合は 1 始まりの行番号、dict 列の場合は 1 始まりの要素位置を指す。検証仕様の準拠先は OpenAI 公式の SFT / DPO fine-tuning データ形式とする。
  - [ ] IF `method` が `sft` / `dpo` 以外である THEN 検証を開始せず明確なエラーを送出する（`raise_on_invalid` の指定に依らない）。
  - [ ] IF `source` が単一の dict である THEN 検証を開始せず明確なエラーを送出する（受けるのはレコードの列またはファイルパスであり、dict をキー文字列の列として誤読しない）。
  - [ ] WHEN ファイルパスの source を検証する THEN BOM を除去し、空行はスキップして `checked` に数えず、違反位置は空行を含む物理行番号（1 始まり）で報告する。IF ファイルを読めない THEN 読み取りエラーを呼び出し側へ伝播する（fail-closed）。
  - [ ] IF 違反が 1 件以上ある THEN 検証結果は不合格を示し、違反ゼロの場合のみ合格を示す（検証は fail-closed。合否判定に曖昧な中間状態を持たない）。
  - [ ] WHEN 検証項目を適用する THEN 単一ターン・複数ターンを区別せず同一規則で検証する。メッセージ必須条件は「`role` あり + `content` または `tool_calls` のいずれかあり」とする。
  - [ ] WHEN レコードにツール系フィールド（レコードレベル `tools` / `parallel_tool_calls`（DPO は `input` 内）・role `"tool"` メッセージ・content 無し + `tool_calls` 有りの assistant メッセージ）が含まれる THEN 合法として違反にしない。tools / tool_calls の内部構造（function スキーマ等）は解釈せず検証しない。
  - [ ] WHEN role 別制約を検証する THEN `tool_calls` キーを許容する role は `"assistant"` のみとし（他 role にあれば違反）、「content 無しでも合法」は tool_calls を持つ assistant に限る。role `"tool"` は `content` 必須 + `tool_call_id` キー存在必須（値の内容は非解釈）とする。
  - [ ] IF `method="sft"` のレコードの `messages` に role `"assistant"` のメッセージが 1 件も無い THEN 違反として報告する（メッセージ単位の違反とは独立に報告し、`weight`: 0 は判定に用いない）。IF レコードが JSON オブジェクトでない / `messages` キーが欠落している / `messages` が非リスト・空リストである THEN 当該の構造違反のみを報告し、本要件を重ねて報告しない。WHEN `method="dpo"` で検証する THEN 本要件を適用しない（assistant は `preferred_output` / `non_preferred_output` 側で必須とする）。
  - [ ] WHEN SFT レコードの assistant メッセージに `weight`: 0 または 1 が含まれる THEN 合法とする。IF `weight` が assistant 以外の role にある、または値が整数 0 / 1 以外（0.5・float の 1.0・true 等）である THEN 違反とする。`weight` の検証は `method="sft"` に限り適用し、DPO のメッセージ・出力配列に `weight` があっても違反にしない（受理可否はプラットフォームへ委ねる）。
  - [ ] WHEN messages の `content` が文字列または parts 配列である THEN 合法とし、parts の内部構造は検証しない。IF `content` が文字列でもリストでもない型である、または空リスト `[]` である THEN 違反とする。content の型規則は DPO の `input.messages`・出力配列にも同一に適用する。
  - [ ] WHEN メッセージに既知キー（`role` / `content` / `tool_calls` / `tool_call_id` / `weight`）以外の未知キーが含まれる THEN 違反にしない（非解釈で許容する）。レコードレベルの未知フィールドも同規則で許容する。
  - [ ] WHEN `method="dpo"` で検証する THEN `preferred_output` / `non_preferred_output` が assistant メッセージの配列であることを検証する（配列内に assistant 以外の role があれば違反）。
  - [ ] WHEN 利用者が `raise_on_invalid=True` を明示指定して検証する THEN 不合格時に FR-10 の構造化エラー（種別: `VALIDATION_FAILED`・検証結果を保持）を送出する。省略時（既定）は例外を送出せず検証結果の返却のみとする。FR-10 の「検証失敗」種別の発生経路は、本オプションおよび FR-1 / FR-2 / FR-4 のデータ不備エラーに限る。
  - [ ] WHEN `submit_job`（FR-5）を呼ぶ THEN submit は暗黙の事前検証を行わず、検証は本ヘルパの明示呼び出しに限る（検証仕様の二重管理を避け、最終判定はプラットフォームに委ねる）。
  - [ ] WHEN 検証を実行する THEN 純データ操作 + ローカルファイル読み取りに徹し、ネットワークに触れない。

### FR-4: 会話ログ（SDK Session）からの SFT データセット生成
- ユーザーストーリー: lib 利用者として、運用中の会話履歴（SDK `Session`）から SFT 用データセットを生成したい。なぜなら実運用の対話を（利用者供給の filter で選別して）学習データとして再利用したいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `dataset_from_session(session, ...)` へ SDK `Session`（不透明型）を渡す THEN `_adapters/` 経由で履歴 items を plain データとして抽出し、user / assistant ターンから SFT chat 形式のケース列を生成して返す（`Session` の内部型を `runtime/finetune` ロジック層へ出さない・SDK 隔離維持）。
  - [ ] WHEN 利用者が filter / transform callable（ケース単位の除外・整形・マスキング関数）を渡す THEN 生成前に各ケースへ適用する。省略時は抽出した全ターンを対象とし、lib は品質自動判定・個人情報の自動マスキングを内蔵しない（品質選別・マスキングロジックは利用者供給。設計論点として明記: 本 FR は `_adapters` への履歴抽出窓口の追加を要し、品質フィルタ / 個人情報の扱いが設計時の主要論点となる）。
  - [ ] IF 履歴が空、または抽出可能な user / assistant ターンが存在しない THEN 生成不能の理由を示す明確なエラーを返し、空データセットを暗黙に返さない。
  - [ ] WHEN `Session` へアクセスする THEN 読み取りのみとし、履歴の書込・削除・改変を行わない。
  - [ ] WHEN 生成対象の形式を定める THEN SFT 形式のみを対象とし、会話ログからの DPO preference 生成（preferred / non_preferred ペアの導出）はスコープ外とする（ペアは会話ログから機械的に決定できないため。DPO データは FR-2 経由で利用者がキュレーションする）。

### FR-5: FT ジョブの submit（アップロード + ジョブ作成・SFT / DPO 第一級・method passthrough）
- ユーザーストーリー: lib 利用者として、整形済みデータセットを渡して FT ジョブを 1 呼び出しで開始したい。なぜならファイルアップロードとジョブ作成の boilerplate を手書きしたくないから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `submit_job(client, train=..., model=..., method=...)`（`val` は任意）を呼ぶ THEN 学習ファイルのアップロードとジョブ作成をそれぞれ単発 API 呼び出しとして `_adapters/` 経由で実行し、`job_id` を含む plain なジョブ参照を返す。学習ループ・進捗監視は含まない（監視は FR-6 / FR-7 の明示呼び出しに限る）。
  - [ ] WHEN `method` に `sft` / `dpo` を指定する THEN 両プラットフォームの fine-tuning ジョブ API の method 指定形式（`method: {type: ..., ...}`）へマッピングして送信する（SFT / DPO を第一級サポート）。
  - [ ] WHEN 利用者が method 詳細設定（hyperparameters・beta 等）や将来メソッドの識別子を渡す THEN lib は解釈せずプラットフォームへ passthrough する（lib 側にメソッド仕様を抱え込まない）。
  - [ ] WHEN model と method の組み合わせ可否（例: DPO 対応モデルが SFT より狭い）を扱う THEN lib は対応モデル一覧を保持・検証せず、プラットフォームが返すエラーを FR-10 の明確なエラー（理由文言を保全）へ変換して返す（モデル一覧のハードコードは鮮度切れするため方針として非検証を確定する）。
  - [ ] WHEN 利用者がアップロード済みのファイル id を渡す THEN 再アップロードせず当該 id でジョブを作成する。
  - [ ] WHEN submit を実行する THEN 利用者の明示呼び出しに限り、lib が自動・暗黙にジョブを開始することはない（従量課金操作の明示性）。アップロードしたデータ・ジョブ設定を lib 内に保持しない。

### FR-6: ステータス照会と model_ref 取得（単発呼び出し）
- ユーザーストーリー: lib 利用者として、ジョブの進捗と完成モデルの参照を単発呼び出しで取得したい。なぜなら完成モデル id を `AgentSpec` の model にそのまま使いたいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `get_job(client, job_id)` を呼ぶ THEN 単発 API 照会で状態（queued / running / succeeded / failed / cancelled 等のプラットフォーム状態を plain な列挙へ写像）・完成時の `model_ref`（fine-tuned モデル id）・失敗時の理由を含む plain な結果を返す。
  - [ ] WHEN ジョブが succeeded である THEN 結果の `model_ref` は利用者がそのまま `AgentSpec` の model / 外部 DI 流入へ使える plain 文字列とする。lib はモデル重みを保持せず、デプロイ・ホスティングを行わない。
  - [ ] WHEN Azure OpenAI のジョブが succeeded である THEN `model_ref` はデプロイ前のモデル参照であり、推論利用には Azure 側のデプロイ操作が別途必要である旨を結果またはドキュメントで明示する（デプロイ管理（control plane 操作）はスコープ外・制約事項に記載）。
  - [ ] IF `job_id` が存在しない / アクセス不能 THEN プラットフォームエラーを FR-10 の明確なエラーへ変換して返す。

### FR-7: opt-in ブロッキング待機（タイムアウト必須・唯一のループ隔離）
- ユーザーストーリー: lib 利用者として、スクリプト内でジョブ完了まで待機したい。なぜならポーリングの手書きをせずに完成 `model_ref` を受け取りたいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `wait_job(client, job_id, timeout=..., poll_interval=...)`（`timeout` は必須引数・`poll_interval` は既定値あり）を呼ぶ THEN 単発照会（FR-6 相当）を `poll_interval` 間隔で繰り返し、終端状態（succeeded / failed / cancelled）に達したら FR-6 と同形の結果を返す。
  - [ ] IF 経過時間が `timeout` に達する THEN 明確なタイムアウトエラーを返す。ジョブ自体は取り消さず、利用者は同じ `job_id` で FR-6 / FR-7 を再実行できる。
  - [ ] WHEN `timeout` を省略しようとする THEN 受理しない（無限待機の既定を持たない・必須引数とする）。
  - [ ] WHEN 本関数を設計する THEN lib 内で唯一のポーリングループを本関数 1 つに隔離し、build-don't-run 原則の例外として ADR で正当化する（前例: ADR 0004 / ADR 0012 と同じ「宣言的な薄い結線 + 明示 opt-in + 上限必須」の型）。

### FR-8: OpenAI / Azure OpenAI 両対応（client 注入・env 非依存）
- ユーザーストーリー: lib 利用者として、同一の公開 API で OpenAI と Azure OpenAI のどちらの FT API も使いたい。なぜなら接続先の違いでコードを書き分けたくないから。
- 受け入れ基準:
  - [ ] WHEN 利用者が自ら構築した `AsyncOpenAI` client を渡す THEN FR-5〜FR-7 が OpenAI の fine-tuning API に対して動作する。
  - [ ] WHEN 利用者が自ら構築した `AsyncAzureOpenAI` client を渡す THEN 同一の公開関数群が Azure OpenAI の fine-tuning API に対して動作する（公開 API は接続先で分岐しない）。
  - [ ] WHEN client を注入する THEN lib は client を不透明値として扱い、client の内部構築・認証情報の組み立て・環境変数の読み取りを行わない（認証・エンドポイント・API バージョン等の構成は利用者が client 構築時に済ませる。env 非依存方針と整合）。
  - [ ] WHEN 両プラットフォームのエラーコード / レスポンス形式差分を吸収する THEN 差分の実装は `_adapters/` 配下（`_adapters/finetune.py`）に閉じ、`runtime/finetune` ロジック層には plain データのみを流す。

### FR-9: extra としての公開と未導入時に壊れない契約
- ユーザーストーリー: ライブラリ利用者として、FT 機能を任意導入したい。なぜなら FT を使わない利用者に追加の依存・API を意識させたくないから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `pip install oai-agentspec[finetune]` を行う THEN FT 機能（`runtime/finetune` の公開窓口）が利用可能になる。
  - [ ] WHEN finetune extra 未導入で `import oai_agentspec` を実行する THEN ImportError 等を起こさずコア宣言層の公開 API（コア `__all__`）が利用できる。
  - [ ] WHEN finetune extra 未導入で既存 extra（conversation / serve / cli / llmops / lightning）を import する THEN finetune 依存に起因する破綻なく従来どおり動作する。
  - [ ] WHEN finetune extra 未導入で FT 機能の import 経路へアクセスする THEN 必要 extra（`oai-agentspec[finetune]`）の導入を促す明確なエラーメッセージを返す。
  - [ ] WHEN FT 機能の公開 API を追加する THEN コア宣言層の公開契約（コア `__all__`）を変更せず、公開 API（変換 / 検証 / Session 生成ヘルパ・`submit_job` / `get_job` / `wait_job`・結果型・エラー型）は `runtime/finetune` の独立した公開窓口に集約する。

### FR-10: 失敗時の graceful degradation
- ユーザーストーリー: lib 利用者として、FT 操作の失敗時にプロセスが未捕捉例外で落ちないようにしたい。なぜなら extra 不在・設定不在・検証失敗・API エラー・タイムアウトを判別可能なエラーとして受け取り、運用を継続したいから。
- 失敗種別は `FineTuneFailureKind`（StrEnum）で判別する。メンバは `VALIDATION_FAILED`（検証失敗・FR-1/2/4 のデータ不備と FR-3 の raise opt-in）/ `EXTRA_MISSING`（extra 不在）/ `CONFIG_MISSING`（必須設定不在）/ `API_ERROR`（プラットフォーム API エラー）/ `TIMEOUT`（待機タイムアウト）の 5 種とする。
- 受け入れ基準:
  - [ ] IF finetune extra が不在の状態で FT 機能を要求する THEN 必要 extra の導入を促す明確なエラーを返し、未捕捉例外でプロセスを停止しない（FR-9 と整合）。
  - [ ] IF 必須設定（client / train / model / method 等）が不在 THEN 不足を示す明確なエラーを返し、暗黙のフォールバックをしない。
  - [ ] IF プラットフォーム API がエラー（認証失敗・モデル×メソッド非対応・ジョブ失敗等）を返す THEN プラットフォームの理由文言を保全した明確なエラーへ変換して返し、未捕捉例外でプロセスを停止しない。
  - [ ] WHEN 失敗を返す THEN 失敗の種別を判別可能な構造化されたエラー（`FineTuneError`・`kind` に `FineTuneFailureKind` を持つ）として返す。検証失敗時は検証結果（`DatasetValidationReport`）を keyword-only 属性として保持できる。

## 3. 非機能要件

### NFR-1: セキュリティ（SDK / 外部クライアント隔離）
- 要件: FT API / Files API / Session 履歴抽出の呼び出しは `_adapters/` 配下のみを窓口とし、`runtime/finetune` ロジック層は plain データと不透明型のみを扱う。`from agents` / `from openai` の直接 import を FT ロジック層に持ち込まない。データ変換・検証（FR-1/2/3）は SDK / openai を import しない純データ層とし、`tools=` の FunctionTool 相当オブジェクトは属性ダックタイピングで受ける（SDK 型を import しない）。
- 計測基準: `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` の結果が空であること（FT 機能追加後も空を維持）。`from openai` / `import openai` も同様に `_adapters/` 配下に閉じることを grep で確認する。

### NFR-2: 可用性（コア / 既存 extra が finetune 未導入で壊れない）
- 要件: finetune extra 未導入の環境で `import oai_agentspec` および既存 extra（conversation / serve / cli / llmops / lightning）の import が成功する。FT 依存は遅延 import 境界で隔離する。
- 計測基準: `uv run python -c "import oai_agentspec as m; assert all(hasattr(m,s) for s in m.__all__)"` が finetune extra 未導入でも成功すること。既存 extra の import スモークが finetune 未導入で緑であること。

### NFR-3: 保守性（env 参照の境界・build-don't-run 整合）
- 要件: 接続構成は利用者構築の client 注入とし、`runtime/finetune` / `_adapters` は環境変数に依存しない。lib は学習ループを実装せず、全操作を単発 API 呼び出しの薄い結線に限定する。唯一のポーリングループは `wait_job`（FR-7）に隔離し、ADR として記録する。
- 計測基準: `grep -rnE "os\.environ|getenv" src/oai_agentspec/runtime/finetune/` の結果が空であること、および本機能で追加する `_adapters` の FT 窓口ファイル（`src/oai_agentspec/_adapters/finetune.py`）を同 grep の対象に加えても結果が空であること（`_adapters/` 全体は対象にしない: 既存の `judge.py` / `lightning.py` に本機能と無関係な env 参照が存在するため、全体 grep は基準として成立しない）。ポーリングループが `wait_job` 実装以外に存在しないことをコードレビューで確認し、当該 ADR が `docs/adr/` に存在すること。

### NFR-4: 保守性（テストカバレッジ / リント・実通信なし検証）
- 要件: FT 機能追加後もプロジェクトのテストカバレッジ閾値とリント基準を維持する。実 API 通信・実課金に依存しない単体・統合テスト（fake クライアント / `_adapters` モック）で全 FR を検証する。
- 計測基準: `uv run pytest` がカバレッジ 80% 以上（`fail_under = 80`）で緑であること。`uv run ruff check src/ tests/` で本変更により新たに増える違反が 0 件であること。テストスイートが実ネットワーク通信なしで完結すること。

### NFR-5: セキュリティ（学習データの外部送信の明示性・個人情報）
- 要件: 学習データがプラットフォームへ送信されるのは `submit_job` の明示呼び出し時のみとし、変換・検証・Session 生成ヘルパはネットワークに触れない。会話ログ由来データの個人情報の除去は利用者供給の filter / transform callable の経路で行え、lib は履歴を submit 以外の外部先へ送信しない。
- 計測基準: 変換 / 検証 / Session 生成ヘルパの実装がネットワーク呼び出しを含まないことをテスト（fake クライアントの非呼び出し検証）で確認する。filter / transform callable がデータ送信前に適用されることをテストで確認する。

### NFR-6: 可用性（失敗の判別可能性・タイムアウト）
- 要件: extra 不在・設定不在・検証失敗・API エラー・タイムアウトのいずれでも未捕捉例外でプロセスを停止せず、種別判別可能なエラーへ倒す（FR-10）。`wait_job` は timeout 必須で無限待機の経路を持たない。
- 計測基準: 各失敗種別（fake / モックで再現）が対応する種別のエラーとして返ることをテストで検証する。`wait_job` の timeout 省略が受理されないこと・timeout 到達で明確なエラーになることをテストで検証する。

## 4. 制約事項

- 技術的制約:
  - SDK / 外部クライアント import は `_adapters/` 配下のみ（NFR-1）。単方向依存（`runtime/finetune` → コア（`_adapters` / `constants`）のみ・コアから finetune への依存辺なし）を維持する。データ変換・検証は `_adapters` / コアへの依存辺を持たない純データ層とする。
  - コア `__all__` は不変。FT 公開 API は `runtime/finetune` 窓口に集約する（FR-9）。
  - `finetune` extra は `pyproject.toml` の `[project.optional-dependencies]` に openai を明示宣言する（openai-agents の推移依存として既に入っているが、ジョブ管理が openai へ直接依存する意図で明示宣言する）。
  - lib はモデル重み・学習データを保持しない。Azure の fine-tuned モデルのデプロイ操作（control plane・推論利用の前提）はスコープ外で利用者責任とする（FR-6）。
  - DPO 対応モデルは SFT より狭い。lib は対応モデル一覧を保持・検証せず、プラットフォームエラーへ委ねる（FR-5・鮮度切れ回避）。DPO はベースモデル / SFT 済みモデルの双方に適用可能（SFT → DPO の 2 段構成は利用者が 2 回のジョブとして実行する）。
  - DPO の学習はプラットフォーム仕様上「1 例につき最後の assistant メッセージ 1 件を preferred / non_preferred として学習する」（1 ターン学習制約）。ヘルパ・`validate_dataset` は出力配列長 1 超を違反にせず形式検証に徹し、受理可否はプラットフォームへ委ねる。
  - `tools=` の FunctionTool 相当写像に `strict_json_schema` は含めない（strict 入りの tools 定義が必要な利用者は plain dict 経由で渡す。`validate_dataset` は未知フィールド許容規則により strict 入り dict も合法とする）。
  - 会話ログからの生成は SFT 形式のみ（FR-4）。品質自動判定・個人情報の自動マスキングは内蔵しない（filter / transform callable の経路を提供し最終責任は利用者）。
  - `Session` へのアクセスは読み取り専用。既存の `EvalCase` / `OptimizeCase` / `ToolRegistry` / `PromptStore` 等コア・既存 extra の型と契約を変更しない（`ToolRegistry` 構築物の受理は finetune 側の受理形であり、`ToolRegistry` 自体は不変）。
- ビジネス制約:
  - FT ジョブは従量課金操作である。lib は利用者の明示呼び出し以外でジョブ起動・ファイルアップロードを行わない（FR-5）。
  - examples で実 API を使う場合は既存前例（`examples/_shared/_azure.py`・`.env` 読み込み）に従う。

## 5. 影響範囲

- 関連コンポーネント:
  - `src/oai_agentspec/runtime/finetune/`（新規・公開窓口 + 変換 / 検証 / 生成 / ジョブ管理）
  - `src/oai_agentspec/_adapters/finetune.py`（FT ジョブ / Files API の窓口・`Session` 履歴抽出窓口）
  - `pyproject.toml`（`finetune` extra の宣言）
  - `tests/runtime/finetune/`（新規・fake / モックによる検証）
  - `docs/architecture.md`（runtime 層の追記）・`docs/adr/`（`wait_job` の build-don't-run 例外 ADR）
- 既存機能への影響:
  - コア宣言層・既存 extra（conversation / serve / cli / llmops / lightning）への変更なし（`EvalCase` / `OptimizeCase` / `ToolRegistry` 構築物は読み取りのみ）。
  - `runtime/lightning` とは別トラックとして併存する（棲み分けの詳細は `docs/architecture.md` の「マネージド Fine-Tuning 統合」節を参照）。
- 将来拡張（本要件のスコープ外）:
  - 会話ログからの DPO preference 生成・品質自動判定・個人情報自動マスキング・Azure デプロイ管理。

## 6. 用語定義

| 用語 | 定義 |
|------|------|
| マネージド fine-tuning API | OpenAI / Azure OpenAI がプラットフォーム側で学習を実行する fine-tuning ジョブ API（fine_tuning/jobs） |
| SFT | Supervised Fine-Tuning。入力と期待出力（chat messages 形式）のペアで学習する手法 |
| DPO | Direct Preference Optimization。同一入力に対する preferred / non_preferred 応答ペアで選好を学習する手法 |
| preference 形式 | DPO 用の JSONL 行形式（`input`（messages 構造）/ `preferred_output` / `non_preferred_output`。出力側は assistant メッセージの配列） |
| method passthrough | ジョブ作成時の method 指定・詳細設定を lib が解釈せずプラットフォームへそのまま渡す方針 |
| model_ref | 完成した fine-tuned モデルの参照（モデル id・plain 文字列）。lib は重みを保持せず参照のみ返す |
| JSONL | 1 行 1 JSON オブジェクトのテキスト形式。FT API の学習ファイル形式 |
| Session | OpenAI Agents SDK の会話履歴ストア（lib からは不透明型として扱う） |
| extra | pip の optional dependency 区分（例: `oai-agentspec[finetune]`）。未導入でもコアが壊れない契約を伴う |
| weight | SFT の assistant メッセージに付す loss masking フラグ（0 = 学習対象外 / 1 = 学習対象。整数のみ合法） |
| content parts | messages の `content` を文字列でなく parts 配列（text / image_url 等）で表す形式（vision fine-tuning）。lib は内部構造を解釈しない |
| FunctionTool 相当オブジェクト | `name` / `params_json_schema` 属性を持つオブジェクト（コア `ToolRegistry` の属性アクセスが返す SDK `FunctionTool` を含む）。`tools=` でダックタイピングにより検出し FT の tools 定義形式へ写像する |
| client 注入 | 利用者が構築した `AsyncOpenAI` / `AsyncAzureOpenAI` を不透明値として公開関数へ渡す接続方式。lib は client を内部構築せず環境変数も読まない |
