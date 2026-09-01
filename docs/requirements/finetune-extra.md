# マネージド Fine-Tuning 統合（oai-agentspec[finetune] 公開 extra）

## 1. 概要

oai-agentspec に、OpenAI / Azure OpenAI のマネージド fine-tuning API（SFT / DPO）を利用者の宣言エージェント運用へ各操作を単一の公開関数呼び出しで組み込める Fine-Tuning 統合機能を、公開 optional extra（`oai-agentspec[finetune]`・`runtime/finetune`）として追加する。機能は (1) データセット整形（既存 dataset 資産 `EvalCase` / `OptimizeCase` / `DpoCase` や plain dict からの SFT chat 形式 / DPO preference 形式への純データ変換・持ち込み JSONL の検証・会話ログ（SDK `Session`）からの SFT データセット生成・DPO preference データセット生成（pair_builder 供給 / 雛形整形 + CSV / JSONL 記入ワークフロー）・ツール往復の文脈保持）、(2) ジョブ管理（学習ファイルのアップロード + ジョブ作成の submit・ジョブ作成時の設定（base model / customization method / training type / データ / suffix / seed / hyperparameters）の通し道・ステータス単発照会・完成 `model_ref` 取得・opt-in のタイムアウト必須待機）の 2 系統からなる。学習の実行エンジンは OpenAI / Azure プラットフォーム側にあり、lib は単発 API 呼び出しの薄い結線に徹する（build-don't-run 維持。唯一のポーリングループは opt-in の待機関数 1 つに隔離し ADR で正当化する。前例: ADR 0004 `fit_ml_estimator` / ADR 0012 `failsafe_call`）。

データセット整形は単一ターン（文字列 input）と複数ターン（messages 形式リスト input）の二形を受理し、ツール呼び出し（function calling）・assistant `weight`（loss masking）・content parts（vision）入りの学習例を非改変で透過する。`tools=` にはコア `ToolRegistry` 由来の `FunctionTool` 相当オブジェクトを直接渡せ、学習データと推論時のツール定義の一致を構造的に担保できる。

ジョブ作成時の設定を網羅的に通せるようにするのは、Azure の training type（`GlobalStandard` / `Standard` / `Developer`）が学習コストに直結し、`suffix` / `seed` がモデル命名と再現性に必要でありながら、いずれも method 詳細設定（hyperparameters）とは別階層のトップレベル項目であるためである。lib はこれらを解釈せず、指定されたものだけを送信する。

本機能は実行寄り層 `runtime/` 配下の一員（`runtime/finetune`）として追加し、llmops / lightning と同型の公開窓口・extra 未導入契約・SDK 隔離方針に従う。既存の `runtime/lightning` とは別トラックとして併存する（棲み分けの詳細は `docs/architecture.md` の「マネージド Fine-Tuning 統合」節を参照）。完成モデルは `model_ref`（モデル id・plain 文字列）として返し、利用者が従来どおり外部 DI（`AgentSpec` の model 流入）で使う。lib はモデル重み・学習データを保持せず、デプロイ・ホスティングを行わない。

段階構成（実装単位。要件の分割ではなく提供順序の宣言）:

| 段階 | 範囲 | 状態 |
|---|---|---|
| 段階 1 | データ整形・検証（FR-1 / FR-2 / FR-3） | 実装済み |
| 段階 2 | 学習ジョブ管理 + ジョブ設定の通し道（FR-5 / FR-6 / FR-7） | 実装済み |
| 段階 3 | 会話ログ由来のデータセット生成（FR-4） | 実装済み |
| 段階 4 | 会話ログ由来の DPO preference データセット生成 + 記入ワークフロー + ツール往復の文脈保持（FR-11 / FR-12・FR-4 改訂） | 実装済み |

FR-8 / FR-9 / FR-10 および NFR-1〜NFR-7 は特定段階に閉じず全段階へ横断的に適用する（段階 1 の範囲で成立する項目は `[x]`、ジョブ管理を要する項目は段階 2 で満たす）。

## 2. 機能要件

### FR-1: SFT 用データ変換ヘルパ（既存 dataset 資産 / plain dict からの純データ変換）
- ユーザーストーリー: lib 利用者として、既存 dataset 資産（`EvalCase` / `OptimizeCase`）や plain dict のケース群を SFT 用 chat messages 形式へ 1 行で変換したい。なぜなら FT の最大摩擦である JSONL 整形を毎回手書きしたくないから。
- 受け入れ基準:
  - [x] WHEN 利用者が `to_sft_dataset(cases, ...)` へ `EvalCase` / `OptimizeCase` / plain dict の列を渡す THEN 各ケースの `input` と `expected_output`（plain dict の場合は `input_key` / `output_key` 引数でキー名を指定可）から `{"messages": [...]}` 形式の plain dict 列を含む結果（`DatasetBuildResult`）を返す。
  - [x] WHEN ケースの `input` が文字列である THEN user 1 件のメッセージへ包む（単一ターン）。WHEN `input` が messages 形式のリストである THEN 各メッセージ dict を非改変で透過採用し（複数ターン）、`expected_output` を最終 assistant メッセージとして messages 末尾へ付す。
  - [x] WHEN `expected_output` が文字列である THEN assistant 1 件のメッセージへ包む。WHEN `expected_output` が assistant メッセージ配列である THEN 非改変で透過採用する。
  - [x] WHEN 利用者が system プロンプトを `system=` で付す THEN 利用者供給の system 文字列を messages 先頭へ付す。lib はプロンプト文字列・学習データを同梱しない（プロンプト非同梱原則と整合）。
  - [x] IF `system=` と input リスト内 system メッセージの両方が存在する THEN 明確なエラーで報告し、暗黙マージ・暗黙置換をしない（skip オプションの対象外）。
  - [x] WHEN input リストに `tool_calls` 付き assistant メッセージ（content 無しでも可）/ role `"tool"` メッセージが含まれる THEN 透過採用し、tool_calls・ツール応答の内容を解釈・改変しない。
  - [x] WHEN input リストのメッセージに `weight` / parts 配列 content 等のフィールドが含まれる THEN 非改変で透過する（ヘルパはメッセージを改変しない）。`expected_output` から末尾付加する assistant メッセージには `weight` を付さない（既定の学習対象。weight を制御したい利用者は input リスト側で自メッセージに付す）。
  - [x] WHEN 利用者が `tools=` / `parallel_tool_calls=` を指定する THEN 内容を解釈せずレコードレベル（レコード直下の `"tools"` / `"parallel_tool_calls"`）へ透過する。省略時（既定 None）はキー自体を出力しない。両引数は呼び出し単位で全レコード共通に付され、レコードごとに異なる tools を要するデータは持ち込み JSONL（FR-3 経由）で扱う。plain dict ケースのレコード別 `"tools"` キーはヘルパの読み取り対象外とする。
  - [x] WHEN `tools=` に FunctionTool 相当オブジェクト（`name` / `params_json_schema` を持つ）が含まれる THEN FT の tools 定義形式（`{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}`）へ写像する（`description` 属性が無い、または値が `None` の場合は空文字として写像する）。plain dict の要素は解釈せず透過し、両者の混在を許す。IF dict でも FunctionTool 相当でもない要素が含まれる THEN 明確なエラーで報告する（skip オプションの対象外）。
  - [x] IF ケースに `expected_output`（または指定キー）が欠落している、`input` が空リスト `[]` である、`input` / 出力側が文字列でもリストでもない型である、または出力側が空の assistant 配列 `[]` である THEN 当該ケースを明確なエラーで報告するか、明示指定の `skip_missing=True` 時のみ除外して除外件数を `DatasetBuildResult.skipped` に含める。欠落値を暗黙に補完しない。
  - [x] IF `cases` が単一の dict / 文字列 / bytes である THEN 変換を開始せず明確なエラーを送出する（`skip_missing` の指定に依らない。1 件だけの場合も `[case]` のようにケースの列へ包む）。
  - [x] WHEN input リストの各要素を検査する THEN 「dict であり `role` キーを持ち、かつ `content` または `tool_calls` のいずれかを持つ」ことのみを検査する（content の presence 判定は型非依存で parts 配列も content ありと数える）。role 値・構造の厳密検証は FR-3 の検証ヘルパに一元化する。
  - [x] WHEN 変換を実行する THEN 純データ操作に徹し、SDK / 外部クライアント / ネットワークに触れない。
  - [x] WHEN 利用者が結果型の `save(path)` を明示的に呼んだ場合のみ THEN JSONL ファイルとして当該パスへ書き出す（opt-in 書込）。既定は plain データ返却のみで lib は自動書き込みをしない。

### FR-2: DPO 用 preference 形式変換ヘルパ
- ユーザーストーリー: lib 利用者として、preferred / non_preferred の応答ペアを持つケース群を DPO の preference 形式へ 1 行で変換したい。なぜなら DPO 固有の JSONL 構造（input / preferred_output / non_preferred_output）を手書きしたくないから。
- 受け入れ基準:
  - [x] WHEN 利用者が `to_dpo_dataset(cases, ...)` へ入力・preferred 応答・non_preferred 応答を持つケース列（`DpoCase` または plain dict・キー名は `preferred_key` / `non_preferred_key` 引数で指定可）を渡す THEN OpenAI / Azure が受理する preference 形式（`input`（messages 構造）/ `preferred_output` / `non_preferred_output`）の plain dict 列を含む結果（`DatasetBuildResult`）を返す。
  - [x] WHEN ケースの `input` が文字列である THEN user 1 件のメッセージとして `input.messages` へ包む。WHEN `input` が messages 形式のリストである THEN 非改変のまま `input.messages` へ透過する（複数ターン）。
  - [x] WHEN `preferred_output` / `non_preferred_output` が文字列である THEN assistant 1 件のメッセージ配列へ包む。WHEN assistant メッセージ配列である THEN 非改変で透過採用する（preference 形式の出力側は assistant メッセージの配列である）。
  - [x] WHEN system メッセージを含めたい THEN input の messages リスト先頭に含めて渡す（`to_dpo_dataset` は `system=` 引数を持たない）。
  - [x] WHEN 利用者が `tools=` / `parallel_tool_calls=` を指定する THEN 内容を解釈せず `input.tools` / `input.parallel_tool_calls` へ透過する。FunctionTool 相当要素の写像・混在受理・不正要素のエラーは FR-1 と同一規則とする。
  - [x] IF ケースに preferred / non_preferred のいずれかが欠落している、または FR-1 と同型の境界（空リスト・不正型・空配列）に該当する THEN 明確なエラーで報告するか、明示指定の `skip_missing=True` 時のみ除外して除外件数を結果に含める。lib は preference ペアを自動生成（評価スコアからの自動導出等）しない（ペアの供給は利用者責任）。
  - [x] IF `cases` が単一の dict / 文字列 / bytes である THEN 変換を開始せず明確なエラーを送出する（`skip_missing` の指定に依らない。FR-1 と同一規則）。
  - [x] WHEN 変換を実行する THEN 純データ操作に徹し、SDK / 外部クライアント / ネットワークに触れない。
  - [x] WHEN 利用者が結果型の `save(path)` を明示的に呼んだ場合のみ THEN JSONL として書き出す（opt-in 書込・既定は返却のみ）。

### FR-3: 持ち込み JSONL の受理と検証
- ユーザーストーリー: lib 利用者として、自前で用意した学習用 JSONL を submit 前に検証したい。なぜならプラットフォームのジョブ失敗（課金・待ち時間発生後）より前に形式不備を検出したいから。
- 受け入れ基準:
  - [x] WHEN 利用者が `validate_dataset(source, method=...)` へ JSONL（ファイルパスまたは plain dict 列）と method（`sft` / `dpo`）を渡す THEN 各行の JSON 妥当性・必須キーの存在・messages / preference 構造の形式を検証し、違反位置と理由の一覧（`DatasetViolation` の列）を含む plain な検証結果（`DatasetValidationReport`）を返す。`DatasetViolation.line` は、source がファイルパスの場合は 1 始まりの行番号、dict 列の場合は 1 始まりの要素位置を指す。検証仕様の準拠先は OpenAI 公式の SFT / DPO fine-tuning データ形式とする。
  - [x] IF `method` が `sft` / `dpo` 以外である THEN 検証を開始せず明確なエラーを送出する（`raise_on_invalid` の指定に依らない）。
  - [x] IF `source` が単一の dict である THEN 検証を開始せず明確なエラーを送出する（受けるのはレコードの列またはファイルパスであり、dict をキー文字列の列として誤読しない）。
  - [x] WHEN ファイルパスの source を検証する THEN BOM を除去し、空行はスキップして `checked` に数えず、違反位置は空行を含む物理行番号（1 始まり）で報告する。IF ファイルを読めない THEN 読み取りエラーを呼び出し側へ伝播する（fail-closed）。
  - [x] IF 違反が 1 件以上ある THEN 検証結果は不合格を示し、違反ゼロの場合のみ合格を示す（検証は fail-closed。合否判定に曖昧な中間状態を持たない）。
  - [x] WHEN 検証項目を適用する THEN 単一ターン・複数ターンを区別せず同一規則で検証する。メッセージ必須条件は「`role` あり + `content` または `tool_calls` のいずれかあり」とする。
  - [x] WHEN レコードにツール系フィールド（レコードレベル `tools` / `parallel_tool_calls`（DPO は `input` 内）・role `"tool"` メッセージ・content 無し + `tool_calls` 有りの assistant メッセージ）が含まれる THEN 合法として違反にしない。tools / tool_calls の内部構造（function スキーマ等）は解釈せず検証しない。
  - [x] WHEN role 別制約を検証する THEN `tool_calls` キーを許容する role は `"assistant"` のみとし（他 role にあれば違反）、「content 無しでも合法」は tool_calls を持つ assistant に限る。role `"tool"` は `content` 必須 + `tool_call_id` キー存在必須（値の内容は非解釈）とする。
  - [x] IF `method="sft"` のレコードの `messages` に role `"assistant"` のメッセージが 1 件も無い THEN 違反として報告する（メッセージ単位の違反とは独立に報告し、`weight`: 0 は判定に用いない）。IF レコードが JSON オブジェクトでない / `messages` キーが欠落している / `messages` が非リスト・空リストである THEN 当該の構造違反のみを報告し、本要件を重ねて報告しない。WHEN `method="dpo"` で検証する THEN 本要件を適用しない（assistant は `preferred_output` / `non_preferred_output` 側で必須とする）。
  - [x] WHEN SFT レコードの assistant メッセージに `weight`: 0 または 1 が含まれる THEN 合法とする。IF `weight` が assistant 以外の role にある、または値が整数 0 / 1 以外（0.5・float の 1.0・true 等）である THEN 違反とする。`weight` の検証は `method="sft"` に限り適用し、DPO のメッセージ・出力配列に `weight` があっても違反にしない（受理可否はプラットフォームへ委ねる）。
  - [x] WHEN messages の `content` が文字列または parts 配列である THEN 合法とし、parts の内部構造は検証しない。IF `content` が文字列でもリストでもない型である、または空リスト `[]` である THEN 違反とする。content の型規則は DPO の `input.messages`・出力配列にも同一に適用する。
  - [x] WHEN メッセージに既知キー（`role` / `content` / `tool_calls` / `tool_call_id` / `weight`）以外の未知キーが含まれる THEN 違反にしない（非解釈で許容する）。レコードレベルの未知フィールドも同規則で許容する。
  - [x] WHEN `method="dpo"` で検証する THEN `preferred_output` / `non_preferred_output` が assistant メッセージの配列であることを検証する（配列内に assistant 以外の role があれば違反）。
  - [x] WHEN 利用者が `raise_on_invalid=True` を明示指定して検証する THEN 不合格時に FR-10 の構造化エラー（種別: `VALIDATION_FAILED`・検証結果を保持）を送出する。省略時（既定）は例外を送出せず検証結果の返却のみとする。FR-10 の「検証失敗」種別の発生経路は、本オプションおよび FR-1 / FR-2 / FR-4 / FR-11 / FR-12 のデータ不備エラーに限る。
  - [x] WHEN `submit_job`（FR-5）を呼ぶ THEN submit は暗黙の事前検証を行わず、検証は本ヘルパの明示呼び出しに限る（検証仕様の二重管理を避け、最終判定はプラットフォームに委ねる）。
  - [x] WHEN 検証を実行する THEN 純データ操作 + ローカルファイル読み取りに徹し、ネットワークに触れない。

### FR-4: 会話ログ（SDK Session）からの SFT データセット生成
- ユーザーストーリー: lib 利用者として、運用中の会話履歴（SDK `Session`）から SFT 用データセットを生成したい。なぜなら実運用の対話を（利用者供給の filter で選別して）学習データとして再利用したいから。
- 受け入れ基準:
  - [x] WHEN 利用者が `dataset_from_session(session, ...)` へ SDK `Session`（不透明型）を渡す THEN `_adapters/` 経由で履歴 items を plain データとして抽出し、user / assistant ターンから SFT chat 形式のケース列を生成して返す（`Session` の内部型を `runtime/finetune` ロジック層へ出さない・SDK 隔離維持）。
  - [x] WHEN 利用者が filter / transform callable（ケース単位の除外・整形・マスキング関数）を渡す THEN 生成前に各ケースへ適用する。省略時は抽出した全ターンを対象とし、lib は品質自動判定・個人情報の自動マスキングを内蔵しない（品質選別・マスキングロジックは利用者供給。設計論点として明記: 本 FR は `_adapters` への履歴抽出窓口の追加を要し、品質フィルタ / 個人情報の扱いが設計時の主要論点となる）。
  - [x] IF 履歴が空、または抽出可能な user / assistant ターンが存在しない THEN 生成不能の理由を示す明確なエラーを返し、空データセットを暗黙に返さない。
  - [x] WHEN `Session` へアクセスする THEN 読み取りのみとし、履歴の書込・削除・改変を行わない。
  - [x] WHEN 利用者が `system=` を指定する THEN 利用者供給の system 文字列を生成レコードの messages 先頭へ付す（FR-1 の `system=` と同型・`to_sft_dataset` へ委譲）。履歴内の system / developer item は生成対象外（破棄）のため本引数と競合しない。
  - [x] WHEN 履歴 items を正規化する THEN FR-11 の共通正規化規則（ツール往復の chat 形式への決定的変換と文脈保持・射影列上で連続する function_call の併合・`output` の content 型への写像・孤児 item の破棄・非 function 系ツール item と生 role の system / developer / tool item の破棄。規則本体の SoT は FR-11）を適用する。ケース化の対象はテキスト応答の assistant ターンのみで不変であり、変換されたツールメッセージは文脈（input）にのみ現れる。本規則の適用によりツール往復を含む履歴では生成ケースの input に変換済みツールメッセージが追加される（ツール往復を含まない履歴では出力不変）。フラグ・オプションによる従来挙動の並存は設けない。
  - [x] WHEN 累積文脈を切り出す THEN FR-11 の切り出し規則を適用する（生成せず `skipped` に計上する対象は空文脈ケース・吸収後 content が空の応答ケース・文脈がツール往復の途中で切れたケースの 3 種。規則本体の SoT は FR-11 で、切り出しの実装は両経路で共通である）。
  - [x] WHEN `case_filter` / `case_transform` を適用する THEN それらが受けるケースの `input` には変換済みツールメッセージが含まれ得る。tool 出力（content へ写した文字列）に含まれる機密・個人情報の除去は本経路の利用者責務とする（NFR-5）。
  - [x] WHEN 利用者が `tools=` / `parallel_tool_calls=` を指定する THEN 両引数を keyword-only の省略可引数として受け、内容を解釈せず委譲先 `to_sft_dataset` の同名引数へ渡す（レコード直下の `"tools"` / `"parallel_tool_calls"` へ透過される）。省略時（既定 `None`）はレコードへ当該キーを出力しない。写像・混在受理・不正要素のエラーは FR-1 と同一規則を委譲先で適用する。`case_filter` が全ケースを除外した場合も委譲を省略せず、不正な `tools=` は `VALIDATION_FAILED` として表面化する（返却値は空 `DatasetBuildResult` の正常返却のまま）。
  - [x] WHEN 生成対象の形式を定める THEN `dataset_from_session` は SFT 形式のみを対象とする（会話ログからの DPO preference 生成は FR-11 / FR-12 が担い、ペアは機械的に決定せず pair_builder 供給または雛形記入とする）。

### FR-5: FT ジョブの submit（アップロード + ジョブ作成・SFT / DPO 第一級・ジョブ設定の通し道 + method passthrough）
- ユーザーストーリー: lib 利用者として、整形済みデータセットとジョブ設定（base model / customization method / training type / suffix / seed / hyperparameters）を渡して FT ジョブを 1 呼び出しで開始したい。なぜならファイルアップロードとジョブ作成の boilerplate を手書きせず、かつポータルで選ぶのと同じ設定項目を lib の窓口越しに欠落なく指定したいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `submit_job(client, train=..., model=..., method=...)`（`val` は任意）を呼ぶ THEN 学習ファイルのアップロードとジョブ作成をそれぞれ単発 API 呼び出しとして `_adapters/` 経由で実行し、`job_id` を含む plain なジョブ参照を返す。学習ループ・進捗監視は含まない（監視は FR-6 / FR-7 の明示呼び出しに限る）。
  - [ ] WHEN `method` に `sft` / `dpo` を指定する THEN `method.type` へそれぞれ `supervised` / `dpo` を設定して送信する（lib の `sft` は API の `supervised` に対応する。FR-3 の `validate_dataset` と語彙を揃えるためのエイリアスであり、他の値は写像せず passthrough 規則に従う）。
  - [ ] WHEN 利用者が method 詳細設定（hyperparameters・`beta` 等）や将来メソッドの識別子（例: `reinforcement`）を渡す THEN lib は解釈せずプラットフォームへ passthrough する。lib はメソッド別のハイパーパラメータ集合（`supervised` の batch_size / learning_rate_multiplier / n_epochs、`dpo` の beta 追加、`reinforcement` の compute_multiplier / eval_interval / eval_samples / reasoning_effort 等）の構造・許容値・既定値を保持せず、キー名の妥当性も検証しない（値の妥当性判定はプラットフォームに委ね、鮮度切れを避ける）。`"auto"` 等の特殊値も解釈せず透過する。
  - [ ] WHEN `val` を指定する THEN 検証ファイルもアップロードし、ジョブ作成リクエストの `validation_file` へ載せる。IF `val` が省略される THEN `validation_file` をリクエストに含めない。
  - [ ] WHEN 利用者が `train` / `val` にアップロード済みのファイル id を渡す THEN 当該引数について再アップロードせず当該 id をそのまま用いる（`train` / `val` は独立に判定し、一方のみファイル id・他方をデータという混在指定も受理する）。
  - [ ] WHEN 利用者が学習ジョブのトップレベル設定（`suffix` / `seed` / `metadata` / `integrations`）を指定する THEN それぞれをジョブ作成リクエストのトップレベルフィールドとして送信する。IF 各設定が省略される THEN 当該フィールドをリクエストに含めない（lib は既定値を発明せず `None` を明示送信しない）。
  - [ ] WHEN `seed` を指定する THEN `method` 内の hyperparameters ではなくリクエストのトップレベルへ載せる（プラットフォーム仕様に一致させる）。
  - [ ] WHEN `suffix` を指定する THEN lib は長さ・使用可能文字の検証を行わず、プラットフォームが返すエラーを FR-10 の明確なエラー（理由文言を保全）へ変換して返す（OpenAI は最大 64 文字、Azure は 18 文字かつドット不可というプラットフォーム差があり、lib が制約表を抱えると鮮度切れするため。制約差はドキュメントに記載する）。
  - [ ] WHEN 利用者が `training_type`（Azure の学習実行方式。`GlobalStandard` / `Standard` / `Developer` 等）を第一級引数として指定する THEN lib は値を検証・正規化せず、プラットフォームが受理する経路（リクエスト body 直下の該当フィールド）へ透過する。IF 省略される THEN 当該フィールドを送信しない（lib は既定値を選ばない）。第一級引数とする理由は学習コストに直結し発見性が要るためであり、値の妥当性・利用可否（リージョン・モデル依存）はプラットフォームへ委ねる。
  - [ ] WHEN Azure 固有の他の設定を指定したい THEN 汎用の追加フィールド透過引数 `extra_body=`（plain dict）で受け、lib は内容を解釈せずジョブ作成リクエストの body 直下へ合成する（設定項目ごとに引数を増やさない）。IF 省略される THEN 何も追加しない。
  - [ ] WHEN `submit_job` がジョブ作成リクエストを組み立てる THEN 利用者が明示指定していないフィールドを送信しない。特に学習完了後の自動デプロイを有効化するフィールドを lib の既定として付加しない（有効化したい利用者は `extra_body=` で明示指定する）。理由: 自動デプロイはホスティング課金を発生させるため、lib が暗黙に有効化してはならない。
  - [ ] WHEN 重複指定を判定する THEN 判定単位は「送信リクエストの同一階層における同一キー名」とする（`training_type=` は body 直下の `trainingType` を占有し、`suffix` / `seed` / `metadata` / `integrations` は body 直下の同名キーを占有する。`method` 内は別階層のため body 直下とは衝突しない）。占有は当該引数が実際に指定された場合に限り、省略された引数のキーは `extra_body=` から指定してよく衝突とはしない。IF `extra_body=` が既に占有されたキーを含む THEN 暗黙のマージ・上書きをせず、衝突したキー名を含む明確なエラー（`CONFIG_MISSING`）を返す。
  - [ ] WHEN model と method の組み合わせ可否（例: DPO 対応モデルが SFT より狭い）を扱う THEN lib は対応モデル一覧を保持・検証せず、プラットフォームが返すエラーを FR-10 の明確なエラー（理由文言を保全）へ変換して返す（モデル一覧のハードコードは鮮度切れするため方針として非検証を確定する）。
  - [ ] WHEN `submit_job` が `train` / `val` のデータをアップロードする THEN アップロードしたファイルがプラットフォーム側で処理完了するまで待ってからジョブ作成を行う（未処理ファイルでのジョブ作成をプラットフォームが拒否するため）。待機の上限は `file_wait_timeout`（既定値あり）で利用者が制御でき、IF 非正値が指定される THEN `CONFIG_MISSING` を返す。IF 処理が失敗状態で終わる THEN `API_ERROR`、IF 上限時間内に処理完了しない THEN `TIMEOUT` を、いずれもファイル id を含む明確なエラーとして返す。WHEN 利用者がアップロード済みファイル id（`str`）を渡す THEN lib は状態を確認・待機せず当該 id をそのまま用いる（当該ファイルが利用可能であることは利用者責任）。
  - [ ] WHEN submit を実行する THEN 利用者の明示呼び出しに限り、lib が自動・暗黙にジョブを開始することはない（従量課金操作の明示性）。アップロードしたデータ・ジョブ設定を lib 内に保持しない。

### FR-6: ステータス照会と model_ref 取得（単発呼び出し）
- ユーザーストーリー: lib 利用者として、ジョブの進捗と完成モデルの参照を単発呼び出しで取得したい。なぜなら完成モデル id を `AgentSpec` の model にそのまま使いたいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `get_job(client, job_id)` を呼ぶ THEN 単発 API 照会で状態（queued / running / succeeded / failed / cancelled 等のプラットフォーム状態を plain な列挙へ写像）・完成時の `model_ref`（fine-tuned モデル id）・失敗時の理由を含む plain な結果を返す。
  - [ ] IF プラットフォームが本要件の列挙に無い状態値を返す THEN 例外にせず「非終端の実行中」として扱い、生の状態文字列を結果に保全する（状態一覧のハードコードは鮮度切れするため、終端状態（succeeded / failed / cancelled）のみを判定対象とし、それ以外はすべて非終端とする）。
  - [ ] WHEN ジョブが succeeded である THEN 結果の `model_ref` は利用者がそのまま `AgentSpec` の model / 外部 DI 流入へ使える plain 文字列とする。lib はモデル重みを保持せず、デプロイ・ホスティングを行わない。
  - [ ] WHEN Azure OpenAI のジョブが succeeded である THEN `model_ref` はデプロイ前のモデル参照であり、推論利用には Azure 側のデプロイ操作が別途必要である旨を結果またはドキュメントで明示する（デプロイ管理（control plane 操作）はスコープ外・利用者責任。制約事項に記載）。
  - [ ] IF `job_id` が存在しない / アクセス不能 THEN プラットフォームエラーを FR-10 の明確なエラーへ変換して返す。

### FR-7: opt-in ブロッキング待機（タイムアウト必須・唯一のループ隔離）
- ユーザーストーリー: lib 利用者として、スクリプト内でジョブ完了まで待機したい。なぜならポーリングの手書きをせずに完成 `model_ref` を受け取りたいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `wait_job(client, job_id, timeout=..., poll_interval=...)`（`timeout` は必須引数・`poll_interval` は既定値あり）を呼ぶ THEN 単発照会（FR-6 相当）を `poll_interval` 間隔で繰り返し、終端状態（succeeded / failed / cancelled）に達したら FR-6 と同形の結果を返す。FR-6 の規則により未知の状態値は非終端として扱い、待機を継続する。
  - [ ] IF 経過時間が `timeout` に達する THEN 明確なタイムアウトエラーを返す。ジョブ自体は取り消さず、利用者は同じ `job_id` で FR-6 / FR-7 を再実行できる。
  - [ ] WHEN `timeout` を省略しようとする THEN 受理しない（無限待機の既定を持たない・必須引数とする）。
  - [ ] WHEN 本関数を設計する THEN lib 内で唯一のポーリングループを本関数 1 つに隔離し、build-don't-run 原則の例外として ADR で正当化する（前例: ADR 0004 / ADR 0012 と同じ「宣言的な薄い結線 + 明示 opt-in + 上限必須」の型）。

### FR-8: OpenAI / Azure OpenAI 両対応（client 注入・env 非依存）
- ユーザーストーリー: lib 利用者として、同一の公開 API で OpenAI と Azure OpenAI のどちらの FT API も使いたい。なぜなら接続先の違いでコードを書き分けたくないから。
- 受け入れ基準:
  - [ ] WHEN 利用者が自ら構築した `AsyncOpenAI` client を渡す THEN FR-5〜FR-7 が OpenAI の fine-tuning API に対して動作する。
  - [ ] WHEN 利用者が自ら構築した `AsyncAzureOpenAI` client を渡す THEN 同一の公開関数群が Azure OpenAI の fine-tuning API に対して動作する（公開 API は接続先で分岐しない）。
  - [ ] WHEN Azure 固有のジョブ設定（`training_type` / `extra_body` 等）を扱う THEN 接続先別の関数を分けず、共通関数の任意引数として表現する。IF OpenAI 向け client に対して Azure 固有設定が指定される THEN lib は接続先を判定してブロックすることをせず、指定どおり送信する。IF その結果プラットフォームがエラーを返す THEN FR-10 の明確なエラー（`API_ERROR`）へ変換して返す（無視された場合の挙動はプラットフォームの責任範囲であり、lib は関与しない）。
  - [ ] WHEN client を注入する THEN lib は client を不透明値として扱い、client の内部構築・認証情報の組み立て・環境変数の読み取りを行わない（認証・エンドポイント・API バージョン等の構成は利用者が client 構築時に済ませる。env 非依存方針と整合）。
  - [ ] WHEN 両プラットフォームのエラーコード / レスポンス形式差分を吸収する THEN 差分の実装は `_adapters/` 配下（`_adapters/finetune.py`）に閉じ、`runtime/finetune` ロジック層には plain データのみを流す。

### FR-9: extra としての公開と未導入時に壊れない契約
- ユーザーストーリー: ライブラリ利用者として、FT 機能を任意導入したい。なぜなら FT を使わない利用者に追加の依存・API を意識させたくないから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `pip install oai-agentspec[finetune]` を行う THEN FT 機能（`runtime/finetune` の公開窓口）が利用可能になる。
  - [x] WHEN finetune extra 未導入で `import oai_agentspec` を実行する THEN ImportError 等を起こさずコア宣言層の公開 API（コア `__all__`）が利用できる。
  - [x] WHEN finetune extra 未導入で既存 extra（conversation / serve / cli / llmops / lightning）を import する THEN finetune 依存に起因する破綻なく従来どおり動作する。
  - [ ] WHEN finetune extra 未導入で FT 機能の import 経路へアクセスする THEN 必要 extra（`oai-agentspec[finetune]`）の導入を促す明確なエラーメッセージを返す。
  - [ ] WHEN FT 機能の公開 API を追加する THEN コア宣言層の公開契約（コア `__all__`）を変更せず、公開 API（変換 / 検証 / Session 生成ヘルパ・`submit_job` / `get_job` / `wait_job`・結果型・エラー型）は `runtime/finetune` の独立した公開窓口に集約する。

### FR-10: 失敗時の graceful degradation
- ユーザーストーリー: lib 利用者として、FT 操作の失敗時にプロセスが未捕捉例外で落ちないようにしたい。なぜなら extra 不在・設定不在・検証失敗・API エラー・タイムアウトを判別可能なエラーとして受け取り、運用を継続したいから。
- 失敗種別は `FineTuneFailureKind`（StrEnum）で判別する。メンバは `VALIDATION_FAILED`（検証失敗・FR-1/2/4/11/12 のデータ不備と FR-3 の raise opt-in）/ `EXTRA_MISSING`（extra 不在）/ `CONFIG_MISSING`（必須設定の不在および設定の不整合。FR-5 の重複指定による衝突を含む）/ `API_ERROR`（プラットフォーム API エラー）/ `TIMEOUT`（待機タイムアウト）の 5 種とする。
  - 判断根拠: ジョブ設定の通し道を広げても新たに生じる失敗は「必須設定の不在・重複指定の衝突」と「プラットフォームが返す設定エラー」の 2 系統で、それぞれ `CONFIG_MISSING` / `API_ERROR` に収まる。種別は利用者の復旧行動を分ける単位であり（引数を直して呼び直す / プラットフォーム側の条件を見直す）、これ以上の細分は分岐価値を生まないため増やさない。プラットフォーム側の詳細（エラーコード・理由文言）は種別ではなくエラーの保全情報として持つ。`CONFIG_MISSING` という命名は段階 1 からの互換のため据え置き、意味は「設定の解決不能（不在・衝突の双方）」と読む。
- 受け入れ基準:
  - [ ] IF finetune extra が不在の状態で FT 機能を要求する THEN 必要 extra の導入を促す明確なエラー（`EXTRA_MISSING`）を返し、未捕捉例外でプロセスを停止しない（FR-9 と整合）。
  - [ ] IF 必須設定（client / train / model / method 等）が不在、または FR-5 の重複指定による衝突に該当する THEN 不足・衝突を示す明確なエラー（`CONFIG_MISSING`）を返し、暗黙のフォールバックをしない。
  - [ ] IF プラットフォーム API がエラー（認証失敗・モデル×メソッド非対応・training type 非対応・suffix 制約違反・ジョブ失敗等）を返す THEN プラットフォームの理由文言を保全した明確なエラー（`API_ERROR`）へ変換して返し、未捕捉例外でプロセスを停止しない。
  - [ ] WHEN 失敗を返す THEN 失敗の種別を判別可能な構造化されたエラー（`FineTuneError`・`kind` に `FineTuneFailureKind` を持つ）として返す。検証失敗時は検証結果（`DatasetValidationReport`）を keyword-only 属性として保持できる。

### FR-11: 会話ログ（SDK Session）からの DPO preference データセット生成（pair_builder 供給 / 雛形整形の 2 モード）
- ユーザーストーリー: lib 利用者として、運用中の会話履歴（SDK `Session`）から DPO 用 preference データセットを生成したい。なぜなら実運用の対話文脈（ツール往復を含む）を再利用しつつ、preferred / non_preferred の判断・調達（人手ラベル・別モデル出力の持ち込み・後からの記入）は自分の責任とペースで行いたいから。
- 受け入れ基準（共通）:
  - [x] WHEN ケース素材を切り出す THEN 両モード共通で FR-4 と同一の切り出し規則を適用する: 累積ペアリング（正規化後の各「テキスト応答の assistant ターン」ごとに 1 ケース・先行全採用ターンを文脈とする）・後述の正規化規則（ツール往復の変換保持を含む）・content parts 配列のテキスト str への吸収・空文脈ケース（input が空になるケース）と吸収後 content が空の assistant 応答のケースは生成せず `skipped` に計上する（skipped 意味論は FR-4 と同一）。
  - [x] WHEN 累積文脈を切り出す THEN 文脈プレフィックス内に「対応する role `"tool"` メッセージが後続しない `tool_calls` の id」を含むケースは生成せず `skipped` に計上する（`function_call` と対応する `function_call_output` の間に assistant テキストが入る履歴（HITL 承認で中断されたラン等）では、スライス境界が往復の途中を切り、推論時 API が拒否する dangling tool_call の並びになるため）。判定は切り出した文脈プレフィックス内で完結し（`tool_calls` の id 集合 − role `"tool"` メッセージの `tool_call_id` 集合が空でなければ skip）、正規化・併合規則には影響しない。skipped 意味論は空文脈・空応答の skip と同一で、往復が閉じた文脈のケースは skip しない。
  - [x] WHEN 履歴 items を正規化する THEN 次の規則を適用する（FR-4 と共通の正規化規則）:
    - `{"type": "function_call", "name": N, "arguments": A, "call_id": C}` item は `{"role": "assistant", "tool_calls": [{"id": C, "type": "function", "function": {"name": N, "arguments": A}}]}` へ、`{"type": "function_call_output", "call_id": C, "output": O}` item は `{"role": "tool", "tool_call_id": C, "content": <O を後述の写像規則で文字列へ写した値>}` へ、1:1 の決定的写像で変換して文脈に保持する（破棄しない）。変換後メッセージの合法性は FR-1 / FR-3 の既存規則（tool_calls 付き assistant・role `"tool"` の受理）に依拠し、新しい検証規則を発明しない。
    - WHEN 破棄対象 item を取り除いた列（以下「射影列」）の上で function_call item が連続する THEN 1 つの assistant メッセージの `tool_calls` 配列へ、生 item 列の出現順で併合する（Responses API の並列ツール呼び出しの表現）。IF 射影列上で 2 つの function_call の間に function_call_output・user / assistant テキスト item のいずれか（= 出力ターンを生む item）が存在する THEN 併合せず、それぞれ独立の assistant メッセージとする（逐次呼び出しの忠実表現）。function_call_output はいずれの場合もそれぞれ独立の role `"tool"` メッセージとする。
    - WHEN 射影列を定める THEN 破棄対象 item は「正規化で出力ターンを 1 件も生まない item」とし、dict でない item および本節の後続規則で破棄と定めた item（孤児 function 系 item・function 以外のツール系 / 補助 item・生の role が user / assistant 以外の item）に限る。対応が取れた function_call_output は出力ターンを生むため射影列に残る。
    - WHEN 決定性を保証する THEN 射影列は先行パス（`call_id` の相互突合）で確定する孤児集合と各 item 自身の `type` / `role` のみから一意に定まり、同じ履歴からは常に同じ出力を得る。
    - IF `call_id` の対応が取れない function_call / function_call_output（孤児）がある THEN 当該 item のみ破棄する（変換は往復の対応が取れたものに限る。ケース全体はエラー・除外にしない）。
    - WHEN function 以外のツール系・補助 item（web_search_call / file_search_call / reasoning / compaction 等）が履歴に含まれる THEN chat 形式に対応物がないため従来どおり破棄する（`skipped` に数えない）。
    - WHEN 履歴 item として生の role を持つ user / assistant 以外の item（system / developer / tool 等）を扱う THEN 従来どおり破棄する（変換で新たに生成される role `"tool"` メッセージとは区別する。破棄対象は履歴側の生 role item のみ）。
    - WHEN function_call_output を変換する THEN `output` を role `"tool"` メッセージの `content`（chat 形式が文字列を要求する欄）へ次の規則で写す: 文字列 THEN そのまま / 未指定または `None` THEN 空文字 / それ以外 THEN `json.dumps(..., ensure_ascii=False, default=str)` による JSON 文字列（JSON に対応しない値は当該値のみ文字列化し、外側の JSON 構造は保つ）。
    - IF `output` が JSON へ直列化できない（循環参照・文字列化できない dict キー等） THEN 劣化した文字列を silent に載せず、`call_id` と原因を含む明確なエラー（`VALIDATION_FAILED`）を送出する（当該経路は SDK が書いた履歴では発生せず、利用者が構築した値でのみ到達するため fail-closed とする）。
    - WHEN 本規則を「非改変透過」との関係で解釈する THEN 本写像規則および併合規則は FR-4 / FR-11 の履歴正規化にのみ適用され、FR-1 / FR-2 の利用者供給データ（`input` / `expected_output` / `preferred_output`）の非改変透過は不変である（型写像も行わない）。非改変透過とは「値の内容を解釈・要約・省略・再解釈しない」保証であり、chat 形式が要求する型への 1:1・決定的・可逆な写像は改変にあたらない。`name` / `arguments` は文字列欄のため写像を要さず従来どおり非改変で透過する。
    - WHEN 本規則の目的を確認する THEN 変換後の全メッセージが `validate_dataset`（FR-3）の合法集合に収まることを保証し、生成関数が成功したままプラットフォームが拒否するレコードを産まない。
  - [x] WHEN ケース化の対象を定める THEN 従来どおりテキスト応答の assistant ターンのみを `expected_output` / `response` の対象とし、変換されたツール往復メッセージ（tool_calls 付き assistant・role `"tool"`）は文脈（input）にのみ現れる（本規則の目的はツール文脈の欠落の解消であり、ツール呼び出し判断そのものの学習ケース化はスコープ外・将来拡張とする）。これにより DPO の `response` = 文字列の前提・累積ペアリングの骨格は両モードとも不変である。
  - [x] WHEN ツール定義（スキーマ）を扱う THEN 会話ログからのツール定義の**復元**は行わない（`Session` にはツール定義が記録されず、会話ログから得られるのはツール往復のみであるため）。一方、利用者が明示指定したツール定義の**透過**は行い、lib は内容を解釈せず委譲先の同名引数へ渡す。
  - [x] WHEN 利用者が `dpo_dataset_from_session(session, pair_builder=..., tools=..., parallel_tool_calls=...)` を呼ぶ THEN 両引数を keyword-only の省略可引数として受け、委譲先 `to_dpo_dataset` の同名引数へ渡す（レコードの `input` 内へ透過される）。省略時（既定 `None`）はレコードへ当該キーを出力しない。写像・混在受理・不正要素のエラーは FR-1 / FR-2 と同一規則を委譲先で適用する（規則を本 FR で二重に定義しない）。
  - [x] IF 雛形モード（`pair_builder` 省略時）で `tools=` / `parallel_tool_calls=` のいずれかが指定される THEN 履歴読み取り（`get_items`）より前に明確なエラー（`CONFIG_MISSING`。FR-10 の「設定の不整合」に該当する）を送出し、silent に無視しない（記入用ケース列は `to_dpo_dataset` へ委譲しないため反映先がない）。エラーメッセージには供給先が `finalize_dpo_draft` であることを含める。記入用ケースのキー集合は 4 キー（`input` / `preferred_output` / `non_preferred_output` / `response`）のまま不変であり、ツール定義を持ち回らない。
  - [x] WHEN 採用ケースが 0 件になる（callable モードで pair_builder が全件 `None` を返す等） THEN 委譲を省略せず、不正な `tools=` が `VALIDATION_FAILED` として表面化する（返却値は `DatasetBuildResult(records=(), skipped=全件)` の正常返却のまま）。
  - [x] IF `session` が `None` である THEN `CONFIG_MISSING` の明確なエラーを送出する（FR-4 と同型）。
  - [x] IF 履歴が空・抽出可能なターンが存在しない・テキスト応答の assistant ターンが 1 件も存在しない THEN 生成不能の理由を示す明確なエラー（`VALIDATION_FAILED`）を送出し、空データセットを暗黙に返さない（FR-4 と同一規則）。
  - [x] WHEN `Session` へアクセスする THEN 読み取りのみ（`get_items` の 1 回呼び出し）とし、履歴の書込・削除・改変を行わない（FR-4 と同一規則）。
  - [x] WHEN 本関数を公開する THEN 既存 `dataset_from_session`（FR-4・SFT 限定）とは独立した新関数として `runtime/finetune` の公開窓口へ追加する。`pair_builder` は keyword-only の省略可引数とし、指定時は callable モード・省略時（`None`）は雛形モードとして動作する。
  - [x] WHEN ペアの由来を定める THEN lib は preferred / non_preferred の品質自動判定・応答の自動生成（同一文脈での再生成・別モデル推論の駆動）・会話ログ内フィードバック信号（訂正・やり直し）からの機械導出をいずれも行わない（ペアの充足は callable モード = pair_builder・雛形モード = 事後記入のいずれも利用者責任。build-don't-run・品質判定非内蔵の既存方針を維持する）。
  - [x] WHEN 生成処理を実行する THEN ネットワーク・プラットフォーム API に触れない（SDK 接触は `Session.get_items` の履歴読み取りのみ。NFR-5 適用）。
- 受け入れ基準（callable モード・`pair_builder` 指定時）:
  - [x] WHEN 利用者が `dpo_dataset_from_session(session, pair_builder=...)` を呼ぶ THEN 各ケース素材へ pair_builder を適用して preference 形式（`input`（messages 構造）/ `preferred_output` / `non_preferred_output`）のレコード列を含む `DatasetBuildResult` を返す（最終変換は `to_dpo_dataset`（FR-2）へ委譲し、preference 形式の知識を二重管理しない）。
  - [x] WHEN pair_builder を各ケース素材へ適用する THEN 入力は `{"input": <累積文脈 messages リスト（変換済みツール往復メッセージを含み得る）>, "response": <実応答文字列>}` の plain dict とする（キー名 `response` は「ログ上の実応答」を指す。SFT 版の `expected_output` は「期待出力」の語義であり DPO 素材には不適合のため別名とする）。
  - [x] WHEN pair_builder が `preferred_output` / `non_preferred_output` の両キーを含む dict を返す THEN 当該ペアを採用してレコードを生成する。値の型規則（文字列 / assistant メッセージ配列・非改変透過）と境界（空リスト・不正型・空 assistant 配列のエラー化）は FR-2 と同一とし、委譲先 `to_dpo_dataset` の検証に一元化する。実応答（`response`）をペアのどちら側に置くか・置かないかは pair_builder の全権とし、lib は関与しない。
  - [x] WHEN pair_builder の戻り値 dict が任意キー `input` を含む THEN lib が組んだ累積文脈の代わりに当該値をレコードの input として採用する（マスキング・個人情報除去の経路。NFR-5 適用）。差し替え値の受理形（文字列 / messages 形式リスト）・型規則・境界（空リスト・不正型）も FR-2 の `input` 規則と同一とし、委譲先 `to_dpo_dataset` の検証に一元化する（違反は `VALIDATION_FAILED`）。IF `input` キーを含まない THEN lib が組んだ累積文脈をそのまま用いる。
  - [x] WHEN pair_builder が `None` を返す THEN 当該ケースを生成せず `skipped` に計上する（ペア不成立の明示 skip）。IF 全ケースが skip される THEN エラーにせず `DatasetBuildResult(records=(), skipped=全件)` を正常返却する（ADR 0033 Decision 6 と同型・skip は利用者の明示判断であり失敗ではない）。
  - [x] IF pair_builder の戻り値が `None` でも dict でもない、または dict だが `preferred_output` / `non_preferred_output` のいずれかを欠く THEN 明確なエラー（`VALIDATION_FAILED`・違反ケースの位置と理由を含む）を送出する（SFT 版 `case_transform` の戻り値検証と同型の fail-closed）。
- 受け入れ基準（雛形モード・`pair_builder` 省略時）:
  - [x] WHEN 利用者が `dpo_dataset_from_session(session)` を呼ぶ THEN 各ケース素材から `{"input": <累積文脈のフラットな messages リスト>, "preferred_output": "", "non_preferred_output": "", "response": <実応答文字列>}` の形の記入用ケース列を含む `DatasetBuildResult` を返す。この形は `to_dpo_dataset`（FR-2）の入力ケース形（既定キー名）そのものである。
  - [x] WHEN 記入用ケースを組み立てる THEN 実応答は `response` キーの参照欄として保持し、`preferred_output` / `non_preferred_output` へ仮置きしない（どちらに置くか・置かないかの判断は品質判定であり lib は内蔵しない）。記入は素の文字列（普通の文章）を空欄へ書くだけとし、messages 配列形式の記述を利用者に要求しない。
  - [x] WHEN 雛形モードの結果を解釈する THEN `DatasetBuildResult.records` の中身は最終 preference レコードではなく記入用ケースである（最終レコードは `finalize_dpo_draft`（FR-12）経由の `to_dpo_dataset` 出力）。この語義は既存の `records` フィールドのまま表現し（新フィールド・別結果型を追加しない）、docstring と本書用語定義で「雛形モードの records = 記入用ケース列」を明示する。記入用ケースは記入されるまで学習データとして不完全（意図的に空欄を含む）であり、「空データセットを暗黙に返さない」既存原則とは矛盾しない（雛形はレコード 0 件ではなく、ケースあり・未記入の状態である）。

### FR-12: DPO 雛形の記入ワークフロー（save_dpo_draft / finalize_dpo_draft）
- ユーザーストーリー: lib 利用者として、雛形をスプレッドシートで開いて素の文字列を記入し、1 関数で最終データセットへ取り込みたい。なぜなら preference ペアの記入は人手作業であり、JSON 編集や手書きフィルタなしで機能全量（生成→記入→取り込み→検証→submit）を使いたいから。
- 受け入れ基準（`save_dpo_draft(source, path)`・記入用ファイルの書き出し）:
  - [x] WHEN 利用者が `save_dpo_draft(source, path)` へ雛形モードの `DatasetBuildResult` または記入用ケース列を渡す THEN `path` の拡張子で形式を切り替えて書き出す: `.csv` はスプレッドシート向け CSV、`.jsonl` は従来形（`DatasetBuildResult.save()` と同内容の JSONL）。IF 拡張子が `.csv` / `.jsonl` のいずれでもない THEN `CONFIG_MISSING` の明確なエラーを送出する（lib は既定形式を発明しない）。
  - [x] WHEN CSV を書き出す THEN 列は `case_index`（1 始まり）/ `context`（累積文脈を人が読める形へ整形した読み取り専用の参照列。例: "user: ...\nassistant: ..." 形式・整形は非可逆でよい）/ `response`（実応答の参照列）/ `preferred_output`（記入列・初期値空）/ `non_preferred_output`（記入列・初期値空）/ `input_json`（累積文脈 messages の JSON 文字列。機械用・編集禁止を列名またはドキュメントで明示）とし、1 ファイルで自己完結させる（復元は `input_json` が担い、参照列の整形精度に依存しない）。
  - [x] WHEN セル内改行・引用符を含む値を書く THEN CSV の標準クオート規則（Python 標準 `csv` モジュールの範囲）に依拠し、新しいエスケープ形式を発明しない。エンコーディングは書き込み・読み取りとも `utf-8-sig` とする（日本語環境の Excel で文字化けせず開け、BOM なし UTF-8 の取り込みも壊れない）。
  - [x] WHEN 書き出しを実行する THEN 利用者の明示呼び出しに限り、`Session`・ネットワークに触れない（純データ + ローカルファイル書込のみ）。
  - [x] IF `source` に必須キー（`input` / `preferred_output` / `non_preferred_output` / `response`）を欠く要素・dict 以外の要素が含まれる、または `source` が dict 列でない（単一 dict / 文字列等） THEN 書き出し前に全件検証して明確なエラー（`VALIDATION_FAILED`・要素位置と欠落キー名を含む）を送出し、部分的に書かれたファイルを残さない。
  - [x] IF CSV へ書き出すいずれかのセルが、書き出し時点の `csv.field_size_limit()` を超える THEN 書き出す前に明確なエラー（`VALIDATION_FAILED`・ケース位置と超過した列名・実文字数と現在の上限を含む）を送出し、ファイルを作らない（CSV は書き出しがセルサイズ無制限・読み取りが `csv.field_size_limit()` 制限という非対称を持ち、無検査だと lib が書いた雛形を `finalize_dpo_draft` が読めない状態が人手記入後に発覚するため）。エラーメッセージには利用者側の対処（`csv.field_size_limit(<大きい値>)` を呼ぶ / `.jsonl` で書き出す）を含める。判定は実行時の現在値を読んで行い、lib はプロセスグローバルな上限そのものを変更しない。
- 受け入れ基準（`finalize_dpo_draft(source) -> DatasetBuildResult`・記入済みの取り込み）:
  - [x] WHEN 利用者が `finalize_dpo_draft(source)` へ記入済み CSV のパス / 記入済み JSONL のパス / メモリ上のケース列（iterable）のいずれかを渡す THEN ケースを読み取り、記入済みケースを既存 `to_dpo_dataset`（FR-2）へ委譲して preference 形式の最終レコード列を含む `DatasetBuildResult` を返す（文字列の assistant 配列への包み込み・`response` 欄の脱落は FR-2 の既存挙動の帰結のまま。FR-2 / FR-3 に変更を加えない）。パスは拡張子で `.csv` / `.jsonl` を判別し、いずれでもない場合は `CONFIG_MISSING`（`save_dpo_draft` と対称）とする。
  - [x] WHEN CSV の source を読む THEN 読み取りは列名（ヘッダ行）ベースとし、列の並び順に依存しない（スプレッドシートでの列の並べ替え・再保存に対して頑健）。`input_json` と記入 2 列（`preferred_output` / `non_preferred_output`）のみを読み、参照列（`case_index` / `context` / `response`）は無視する（参照列の編集は取り込みに影響しない）。IF `input_json` の値が空でなく JSON として不正である THEN 明確なエラー（`VALIDATION_FAILED`・ケース位置とパース理由つき）を送出する。IF `input_json` が空（strip 後に空）で記入 2 列のいずれかに値がある THEN 「JSON として不正」ではなく「文脈復元列が空である」ことを示す明確なエラー（`VALIDATION_FAILED`・ケース位置つき・当該列を削除・上書きした可能性と雛形の列をそのまま残す旨の案内を含む）を送出する（記入した内容の silent 喪失を防ぐ fail-closed）。IF 必須 3 列（`input_json` / 記入 2 列）がすべて空である THEN `input_json` のパースを試みず未記入ケースとして skip 経路（下記の未記入判定）へ合流させる。
  - [x] IF 記入済み CSV に必須列（`input_json` / `preferred_output` / `non_preferred_output`）のいずれかが存在しない THEN `VALIDATION_FAILED` の明確なエラー（欠落列名を含む）を送出する（列削除・別ファイル誤指定を silent な全件 skip にしない fail-closed）。
  - [x] WHEN 未記入を判定する THEN 記入欄の「空」は空文字列または空白文字（スペース・タブ・改行）のみのセル（strip 後に空）を指す。WHEN 両記入欄が空のケースを読む THEN 当該ケースを生成せず `skipped` に計上する（未記入 = 意図的な見送りとして既存 skip 意味論に合流。スプレッドシート編集で混入しやすい空白のみセルは未記入側へ倒し、片欄エラー・空白学習データの発生を防ぐ）。IF 全ケースが未記入である THEN エラーにせず `DatasetBuildResult(records=(), skipped=全件)` を正常返却する。
  - [x] WHEN 記入値を採用する THEN 未記入判定にのみ strip を用い、採用する値は非改変（strip しない）で `to_dpo_dataset` へ渡す（利用者が意図した前後空白・改行を lib が silent に削らない）。
  - [x] IF 片欄のみ記入（「空」の定義は上項と同一）のケースがある THEN 明確なエラー（`VALIDATION_FAILED`・ケース位置つき）を送出する（書きかけの silent 喪失を防ぐ fail-closed。skip にしない）。
  - [x] WHEN 戻り値の `skipped` を定める THEN 「finalize が未記入として skip した件数 + 委譲先 `to_dpo_dataset` が報告した skipped」とし、雛形生成時（`dpo_dataset_from_session`）の `skipped` は含めない（生成時と取り込み時は独立のカウントであり合算しない）。
  - [x] IF source のファイルを読めない THEN 読み取りエラーを呼び出し側へ伝播する（FR-3 と同型の fail-closed）。
  - [x] WHEN 利用者が `tools=` / `parallel_tool_calls=` を指定する THEN 両引数を keyword-only の省略可引数として受け、委譲先 `to_dpo_dataset` の同名引数へ渡す（レコードの `input` 内へ透過される。省略時はレコードへ当該キーを出力せず、写像・不正要素のエラーは FR-1 / FR-2 と同一規則を委譲先で適用する）。雛形ファイル（CSV / JSONL）はツール定義を保持せず CSV の 6 列構成も不変であり、雛形ワークフローにおけるツール定義の供給は本引数へ一本化する。
  - [x] WHEN 全ケースが未記入である THEN 委譲を省略せず、不正な `tools=` が `VALIDATION_FAILED` として表面化する（返却値は `DatasetBuildResult(records=(), skipped=全件)` の正常返却のまま）。
  - [x] WHEN 取り込みを実行する THEN `Session`・ネットワークに触れない（純データ + ローカルファイル読み取りのみ）。finalize は新しい検証規則を発明せず、「両欄空 = skip / 片欄のみ = エラー」という自関数の入力契約のみを持つ（レコードの形式検証は FR-2 委譲と FR-3 の明示呼び出しに委ねる）。

## 3. 非機能要件

### NFR-1: セキュリティ（SDK / 外部クライアント隔離）
- 要件: FT API / Files API / Session 履歴抽出の呼び出しは `_adapters/` 配下のみを窓口とし、`runtime/finetune` ロジック層は plain データと不透明型のみを扱う。`from agents` / `from openai` の直接 import を FT ロジック層に持ち込まない。データ変換・検証（FR-1/2/3）は SDK / openai を import しない純データ層とし、`tools=` の FunctionTool 相当オブジェクトは属性ダックタイピングで受ける（SDK 型を import しない）。
- 計測基準: `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` の結果が空であること（FT 機能追加後も空を維持）。`from openai` / `import openai` も同様に `_adapters/` 配下に閉じることを grep で確認する。

### NFR-2: 可用性（コア / 既存 extra が finetune 未導入で壊れない）
- 要件: finetune extra 未導入の環境で `import oai_agentspec` および既存 extra（conversation / serve / cli / llmops / lightning）の import が成功する。FT 依存は遅延 import 境界で隔離する。
- 計測基準: `uv run python -c "import oai_agentspec as m; assert all(hasattr(m,s) for s in m.__all__)"` が finetune extra 未導入でも成功すること。既存 extra の import スモークが finetune 未導入で緑であること。

### NFR-3: 保守性（env 参照の境界・build-don't-run 整合）
- 要件: 接続構成は利用者構築の client 注入とし、`runtime/finetune` / `_adapters` は環境変数に依存しない。lib は学習ループを実装せず、全操作を単発 API 呼び出しの薄い結線に限定する。lib が実装する唯一のポーリングループは `wait_job`（FR-7）に隔離し、ADR として記録する。
- 計測基準: `grep -rnE "os\.environ|getenv" src/oai_agentspec/runtime/finetune/` の結果が空であること、および本機能で追加する `_adapters` の FT 窓口ファイル（`src/oai_agentspec/_adapters/finetune.py`）を同 grep の対象に加えても結果が空であること（`_adapters/` 全体は対象にしない: 既存の `judge.py` / `lightning.py` に本機能と無関係な env 参照が存在するため、全体 grep は基準として成立しない）。lib が実装するポーリングループが `wait_job` 実装以外に存在しないことをコードレビューで確認し、当該 ADR が `docs/adr/` に存在すること。

### NFR-4: 保守性（テストカバレッジ / リント・実通信なし検証）
- 要件: FT 機能追加後もプロジェクトのテストカバレッジ閾値とリント基準を維持する。実 API 通信・実課金に依存しない単体・統合テスト（fake クライアント / `_adapters` モック）で全 FR を検証する。
- 計測基準: `uv run pytest` がカバレッジ 80% 以上（`fail_under = 80`）で緑であること。`uv run ruff check src/ tests/` で本変更により新たに増える違反が 0 件であること。テストスイートが実ネットワーク通信なしで完結すること。

### NFR-5: セキュリティ（学習データの外部送信の明示性・個人情報）
- 要件: 学習データがプラットフォームへ送信されるのは `submit_job` の明示呼び出し時のみとし、変換・検証・Session 生成ヘルパはネットワークに触れない。会話ログ由来データの個人情報の除去は利用者供給の filter / transform callable（SFT）・pair_builder の `input` 差し替え / 雛形ファイルの記入時編集（DPO）の経路で行え、lib は履歴を submit 以外の外部先へ送信しない。ファイル書込は `save()` / `save_dpo_draft` の明示呼び出しに限る。ツール往復の変換保持により tool 出力（`output` 文字列）が学習文脈・雛形ファイルへ含まれるため、その中の機密・個人情報の除去も同経路の利用者責務とする（lib は自動マスキングを内蔵しない）。
- 計測基準: 変換 / 検証 / Session 生成ヘルパの実装がネットワーク呼び出しを含まないことをテスト（fake クライアントの非呼び出し検証）で確認する。filter / transform callable がデータ送信前に適用されることをテストで確認する。

### NFR-6: 可用性（失敗の判別可能性・タイムアウト）
- 要件: extra 不在・設定不在（衝突含む）・検証失敗・API エラー・タイムアウトのいずれでも未捕捉例外でプロセスを停止せず、種別判別可能なエラーへ倒す（FR-10）。`wait_job` は timeout 必須で無限待機の経路を持たない。
- 計測基準: 各失敗種別（fake / モックで再現）が対応する種別のエラーとして返ることをテストで検証する。`EXTRA_MISSING` は実環境では発生し得ない状態（`finetune` extra が宣言する openai は openai-agents の推移依存として常に存在する）のため、モックで import 失敗を注入して再現する。`wait_job` の timeout 省略が受理されないこと・timeout 到達で明確なエラーになることをテストで検証する。

### NFR-7: セキュリティ（コスト影響操作の明示性）
- 要件: 課金を発生させる操作（ファイルアップロード・ジョブ作成・学習完了後の自動デプロイ）は利用者の明示指定に限る。lib は自動デプロイを有効化するフィールドを既定で付加せず、利用者が指定していないリクエストフィールドを送信しない。
- 計測基準: `submit_job(client, train=..., model=..., method="sft")`（最小引数）で fake クライアントが捕捉するジョブ作成リクエスト body のキー集合が `{"model", "training_file", "method"}` と完全一致すること（`suffix` / `seed` / `metadata` / `integrations` / training type の wire key / 自動デプロイ関連フィールドがいずれも含まれないことを、キー集合の完全一致 assert で同時に担保する）。任意引数を 1 つ指定するごとに、当該引数が担当するキー（`extra_body=` は渡した dict のキー集合）だけが増えることをテストで検証する。

## 4. 制約事項

- 技術的制約:
  - SDK / 外部クライアント import は `_adapters/` 配下のみ（NFR-1）。単方向依存（`runtime/finetune` → コア（`_adapters` / `constants`）のみ・コアから finetune への依存辺なし）を維持する。データ変換・検証は `_adapters` / コアへの依存辺を持たない純データ層とする。
  - コア `__all__` は不変。FT 公開 API は `runtime/finetune` 窓口に集約する（FR-9）。
  - `finetune` extra は `pyproject.toml` の `[project.optional-dependencies]` に openai を明示宣言する（openai-agents の推移依存として既に入っているが、ジョブ管理が openai へ直接依存する意図で明示宣言する）。extra は `finetune` の 1 つのみとし、本機能のために新たな外部依存を追加しない。
  - lib はモデル重み・学習データを保持しない。Azure の fine-tuned モデルのデプロイ操作（control plane・推論利用の前提）はスコープ外で利用者責任とする（FR-6）。ホスティングを伴う操作を lib は実行しない。
  - lib はメソッド別ハイパーパラメータの構造・許容値・既定値、対応モデル一覧、`training_type` の許容値、`suffix` の長さ制約、ジョブ状態の全列挙を保持しない（いずれも鮮度切れするため。判定はプラットフォームへ委ね、エラー・生の状態値を保全して返す）。
  - ジョブ作成リクエストのトップレベル `hyperparameters` はプラットフォーム側で deprecated のため第一級引数として扱わず、`method` 内 passthrough を正とする（どうしても必要な利用者は `extra_body=` で指定でき、lib は解釈しない）。
  - `suffix` の制約はプラットフォームで異なる（OpenAI: 最大 64 文字 / Azure: 18 文字・ドット不可）。lib は検証せず、差異はドキュメントに記載する（FR-5）。
  - DPO 対応モデルは SFT より狭い。lib は対応モデル一覧を保持・検証せず、プラットフォームエラーへ委ねる（FR-5）。DPO はベースモデル / SFT 済みモデルの双方に適用可能（SFT → DPO の 2 段構成は利用者が 2 回のジョブとして実行する）。
  - DPO の学習はプラットフォーム仕様上「1 例につき最後の assistant メッセージ 1 件を preferred / non_preferred として学習する」（1 ターン学習制約）。ヘルパ・`validate_dataset` は出力配列長 1 超を違反にせず形式検証に徹し、受理可否はプラットフォームへ委ねる。
  - `tools=` の FunctionTool 相当写像に `strict_json_schema` は含めない（strict 入りの tools 定義が必要な利用者は plain dict 経由で渡す。`validate_dataset` は未知フィールド許容規則により strict 入り dict も合法とする）。
  - 会話ログからの生成は SFT（FR-4）と DPO preference（FR-11 / FR-12）の 2 形式。DPO のペアは機械的に決定せず、pair_builder 供給または雛形記入で利用者が充足する。品質自動判定・個人情報の自動マスキングは内蔵しない（SFT は filter / transform callable、DPO は pair_builder の `input` 差し替え・記入時編集の経路を提供し、最終責任は利用者）。
  - 会話ログ正規化はツール往復（function_call / function_call_output）を chat 形式へ決定的に変換して文脈に保持する。変換は item のフィールド（`name` / `arguments` / `call_id` / `output`）に基づく写像であり、内容（arguments / output の中身）は解釈・改変しない。切り出しの骨格（累積ペアリング・skipped 意味論・parts→str 吸収・生 role の非 user/assistant item の破棄）は ADR 0033 を踏襲し、ツール往復の変換保持と DPO 2 モード・記入ワークフローの設計判断は ADR 0034 に記録する（ADR 0033 は partially superseded）。
  - DPO 生成・記入ワークフローの追加による公開 API の増分は `runtime/finetune` 窓口への 3 シンボル（`dpo_dataset_from_session` / `save_dpo_draft` / `finalize_dpo_draft`）のみで、コア `__all__` は不変。これに加えて既存 3 関数（`dataset_from_session` / `dpo_dataset_from_session` / `finalize_dpo_draft`）が keyword-only の省略可引数 `tools=` / `parallel_tool_calls=` を持つ（省略時の挙動は不変であり breaking change ではない。`save_dpo_draft` は対象外）。新たな外部依存・extra を追加しない（CSV は Python 標準 `csv` モジュールの範囲・`finetune` extra の範囲内）。`FineTuneFailureKind` も 5 種から増減させない（本機能の失敗は `CONFIG_MISSING` / `VALIDATION_FAILED` に収まる）。
  - `Session` へのアクセスは読み取り専用。既存の `EvalCase` / `OptimizeCase` / `ToolRegistry` / `PromptStore` 等コア・既存 extra の型と契約を変更しない（`ToolRegistry` 構築物の受理は finetune 側の受理形であり、`ToolRegistry` 自体は不変）。
- ビジネス制約:
  - FT ジョブは従量課金操作である。lib は利用者の明示呼び出し以外でジョブ起動・ファイルアップロードを行わない（FR-5）。
  - 学習完了後の自動デプロイはホスティング課金を発生させるため、lib は既定で有効化しない（FR-5 / NFR-7）。ただし経路自体は塞がず、利用者が `extra_body=` で当該フィールドを明示指定した場合に限り有効化され得る（lib は内容を解釈せず透過するため）。デプロイ運用（作成・削除・コスト管理）は利用者責任とする。
  - Azure の training type（`GlobalStandard` / `Standard` / `Developer`）は学習コストに直結する（Global は regional standard より低廉、Developer はさらに低廉だがスポット容量でプリエンプトがありデータレジデンシー保証がない）。lib は既定値を選ばず、指定がなければ当該フィールドを送信しない（FR-5）。
  - RFT（`reinforcement`）ジョブには per-job の課金上限が存在し、到達時にジョブが一時停止してチェックポイントを作る。lib は method passthrough に徹し、この挙動を検知・制御しない。
  - examples で実 API を使う場合は既存前例（`examples/_shared/_azure.py`・`.env` 読み込み）に従う。

## 5. 影響範囲

- 関連コンポーネント:
  - `src/oai_agentspec/runtime/finetune/`（公開窓口 + 変換 / 検証 / 生成 / ジョブ管理）。会話ログ由来の生成（SFT / DPO）は `session_dataset.py`、DPO 雛形の記入ワークフロー（Session 非接触の純データ + ローカルファイル I/O）は `dpo_draft.py`
  - `src/oai_agentspec/runtime/finetune/types.py`（エラー種別 `FineTuneFailureKind` の拡充・ジョブ結果型の追加）
  - `src/oai_agentspec/_adapters/finetune.py`（FT ジョブ / Files API の窓口・`Session` 履歴抽出窓口）
  - `pyproject.toml`（`finetune` extra の宣言）
  - `tests/runtime/finetune/`（fake / モックによる検証）
  - `docs/architecture.md`（runtime 層の追記）・`docs/adr/`（`wait_job` の build-don't-run 例外 ADR）
  - `CLAUDE.md`（build-don't-run 例外リストへ `wait_job` を 5 例目として追記）
- 既存機能への影響:
  - コア宣言層・既存 extra（conversation / serve / cli / llmops / lightning）への変更なし（`EvalCase` / `OptimizeCase` / `ToolRegistry` 構築物は読み取りのみ）。
  - 実装済みの段階 1（FR-1 / FR-2 / FR-3）の公開 API・挙動は変更しない。`to_dpo_dataset` / `validate_dataset` / `_adapters/finetune.py` の `fetch_session_items` は段階 4 でも再利用のみで変更しない（変換後メッセージの合法性は FR-1 / FR-3 の既存規則で受理済み）。
  - `dataset_from_session`（FR-4）は段階 4 で既定挙動が変わる（契約変更・ユーザー承認済み 2026-09-01）: ツール往復を含む履歴では生成ケースの input に変換済みツールメッセージが追加される。ツール往復を含まない履歴では出力不変。
  - `FineTuneFailureKind` は段階 2 で FR-10 の 5 種へ拡充する（現在は `VALIDATION_FAILED` のみ）。以後 5 種から増減させず、プラットフォーム側の詳細は種別ではなくエラーの保全情報で表す。
  - `runtime/lightning` とは別トラックとして併存する（棲み分けの詳細は `docs/architecture.md` の「マネージド Fine-Tuning 統合」節を参照）。
- 将来拡張（本要件のスコープ外）:
  - preference ペアの品質自動判定・応答の自動生成（同一文脈での再生成によるペア作成）・会話ログ内フィードバック信号からの機械導出・個人情報自動マスキング。
  - ツール呼び出し判断そのものの学習ケース化（tool_calls を `expected_output` にする形）・ツール定義（`tools=`）の会話ログからの復元・CSV / JSONL 以外の記入形式（xlsx 等）。
  - Azure のデプロイ管理（control plane 操作。デプロイの作成・照会・削除）およびホスティング運用・コスト監視。
  - ジョブのキャンセル・イベント / チェックポイント一覧の取得。

## 6. 用語定義

| 用語 | 定義 |
|------|------|
| マネージド fine-tuning API | OpenAI / Azure OpenAI がプラットフォーム側で学習を実行する fine-tuning ジョブ API（fine_tuning/jobs） |
| SFT | Supervised Fine-Tuning。入力と期待出力（chat messages 形式）のペアで学習する手法。lib の `method="sft"` は API の `method.type = "supervised"` に対応する |
| DPO | Direct Preference Optimization。同一入力に対する preferred / non_preferred 応答ペアで選好を学習する手法 |
| preference 形式 | DPO 用の JSONL 行形式（`input`（messages 構造）/ `preferred_output` / `non_preferred_output`。出力側は assistant メッセージの配列） |
| customization method | 学習手法の指定（API 側の語彙で `supervised` / `dpo` / `reinforcement`）。`method.type` に対応する |
| training type | Azure の学習実行方式（`GlobalStandard` / `Standard` / `Developer`）。学習コストに直結し、ジョブ作成リクエストの body 直下（wire key `trainingType`）で指定する。デプロイ方式（ホスティング）とは別概念 |
| トップレベル設定 | ジョブ作成リクエストの body 直下に置くフィールド（`suffix` / `seed` / `metadata` / `integrations` 等）。`method` 内の hyperparameters とは階層が異なる |
| method passthrough | ジョブ作成時の method 指定・詳細設定（hyperparameters 等）を lib が解釈せずプラットフォームへそのまま渡す方針 |
| `extra_body=` | Azure 固有・将来追加のジョブ設定を body 直下へ合成するための汎用透過引数（plain dict）。lib は内容を解釈せず、キーの重複のみ判定する |
| model_ref | 完成した fine-tuned モデルの参照（モデル id・plain 文字列）。lib は重みを保持せず参照のみ返す。Azure ではデプロイ前の参照であり、推論利用にはデプロイが別途必要（スコープ外） |
| 終端状態 | ジョブの最終状態（succeeded / failed / cancelled）。これ以外の状態値は未知のものも含め非終端として扱う |
| JSONL | 1 行 1 JSON オブジェクトのテキスト形式。FT API の学習ファイル形式 |
| Session | OpenAI Agents SDK の会話履歴ストア（lib からは不透明型として扱う） |
| extra | pip の optional dependency 区分（例: `oai-agentspec[finetune]`）。未導入でもコアが壊れない契約を伴う |
| weight | SFT の assistant メッセージに付す loss masking フラグ（0 = 学習対象外 / 1 = 学習対象。整数のみ合法） |
| content parts | messages の `content` を文字列でなく parts 配列（text / image_url 等）で表す形式（vision fine-tuning）。lib は内部構造を解釈しない |
| FunctionTool 相当オブジェクト | `name` / `params_json_schema` 属性を持つオブジェクト（コア `ToolRegistry` の属性アクセスが返す SDK `FunctionTool` を含む）。`tools=` でダックタイピングにより検出し FT の tools 定義形式へ写像する |
| client 注入 | 利用者が構築した `AsyncOpenAI` / `AsyncAzureOpenAI` を不透明値として公開関数へ渡す接続方式。lib は client を内部構築せず環境変数も読まない |
| pair_builder | 利用者供給の callable（keyword-only・省略可）。ケース素材 `{"input": messages, "response": str}` を受け、`{"preferred_output": ..., "non_preferred_output": ...}`（任意キー `input` で文脈差し替え可）または `None`（skip）を返す。ペアの判断・調達の全権を持つ |
| callable モード | `pair_builder` 指定時の `dpo_dataset_from_session` の動作。pair_builder が返したペアを `to_dpo_dataset` へ委譲して preference 形式レコードを生成する |
| 雛形モード | `pair_builder` 省略時の `dpo_dataset_from_session` の動作。記入用ケース列を `DatasetBuildResult` で返す。記入は素の文字列で行い、最終レコード化は `finalize_dpo_draft` で行う |
| 記入用ケース | 雛形モードが返す `{"input": <messages リスト>, "preferred_output": "", "non_preferred_output": "", "response": <実応答>}` の plain dict。`to_dpo_dataset` の入力ケース形（既定キー名）そのもの。空欄が記入されるまで学習データとして不完全 |
| response 欄 | ケース素材・記入用ケースが持つ実応答（ログ上の assistant 応答テキスト）の参照キー / CSV 参照列。取り込み（`to_dpo_dataset` 委譲）では指定キーのみを読む既存仕様により最終レコードへ透過されず自然に脱落する |
| 記入列 / 参照列 | CSV の列区分。記入列は `preferred_output` / `non_preferred_output`（初期値空・利用者が素の文字列を書く）。参照列は `case_index` / `context` / `response`（読み取り専用・finalize は無視する） |
| input_json | CSV の機械用列。累積文脈 messages の JSON 文字列で、finalize の復元源。編集禁止 |
| 未記入 / 片欄記入 | 記入欄の「空」= 空文字列または空白文字のみのセル（strip 後に空）。未記入 = 両記入欄が空（finalize が skip 計上）。片欄記入 = 一方のみ記入（書きかけとみなし `VALIDATION_FAILED`） |
| tools 透過 | 利用者が供給したツール定義を lib が解釈せず、委譲先（`to_sft_dataset` / `to_dpo_dataset`）の `tools=` へそのまま渡すこと。会話ログからのツール定義の復元（Session に記録がないため行わない）とは別概念 |
| ツール往復 | 履歴中の function_call item（ツール呼び出し）と対応する function_call_output item（ツール応答）の組。chat 形式（tool_calls 付き assistant + role `"tool"` メッセージ）へ決定的に変換して文脈に保持する |
| 孤児 | `call_id` の対応相手が履歴に存在しない function_call / function_call_output item。当該 item のみ破棄する |
| テキスト応答の assistant ターン | 吸収後 content が非空の assistant ターン。ケース化（`expected_output` / `response`）の対象。変換済みツールメッセージは対象外で文脈にのみ現れる |
| ケース素材 | 累積ペアリングで切り出した「累積文脈（input）+ 実応答（response）」の plain dict。pair_builder への入力・記入用ケースの素 |
| 累積ペアリング | 正規化後の各テキスト応答 assistant ターンごとに、先行全採用ターンを文脈として 1 ケースを生成する規則 |
