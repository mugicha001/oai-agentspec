# 0034: 会話ログのツール往復は chat 形式へ変換して文脈保持し、DPO ペアは pair_builder 供給 / 雛形記入の 2 モードで利用者に委ねる

- Status: accepted (partially superseded by 0036: Decision 1（output の非改変透過）と Decision 2（併合条件）)
- Date: 2026-09-01

## Context

会話ログ（SDK `Session`）由来のデータセット生成には 2 つの未解決点があった。

1. **ツール往復の文脈欠落**: ADR 0033 Decision 2 は `role` キーを持たない item を一律破棄すると
   決めており、`function_call` / `function_call_output` item もこれに含まれる。結果として
   ツールを使う実運用対話では「ツールを呼び、その結果を踏まえて答えた」という文脈が学習データから
   丸ごと落ち、応答だけが文脈なしで学習ケース化される。破棄の根拠は「Responses API の item 構造と
   FT chat 形式が非互換であり、透過すると `validate_dataset` 違反レコードを生む」ことだったが、
   非互換なのは**透過**の場合であり、chat 形式（tool_calls 付き assistant + role `"tool"`）への
   変換であれば `validate_dataset` が既に合法として受理する（FR-1 / FR-3 の既存規則）。
2. **DPO ペアの決定不能性**: 会話ログには実応答が 1 つしか存在せず、preferred / non_preferred の
   ペアを機械的に決定できない。ADR 0033 はこれを理由に会話ログからの DPO 生成をスコープ外と
   していた。一方で「文脈素材の組み立て」自体は SFT 経路と完全に共通であり、ペア充足だけを
   利用者へ委ねれば lib が品質判定・応答生成を内蔵せずに DPO 経路を提供できる。

ユーザー確定方針（2026-09-01）:

- ツール往復は破棄せず chat 形式へ変換して文脈に保持する。`dataset_from_session`（FR-4・
  リリース済み）の既定挙動が変わる契約変更を承認済み。フラグによる新旧併存は行わない。
- 記入ワークフローは「つかいやすく、かつ機能を全量使えること」を優先し、CSV 往復
  （スプレッドシート記入）と未記入ケースの自動 skip を採用する。JSON 手編集を要求しない。

制約（所与）:

- 新たな外部依存・extra を追加しない（CSV は Python 標準 `csv` モジュールの範囲）。
- `Session` アクセスは読み取り専用（`get_items` 1 回）。SDK 接触は `_adapters/finetune.py` の
  `fetch_session_items` に閉じる（NFR-1）。
- `to_dpo_dataset`（FR-2）/ `validate_dataset`（FR-3）/ `FineTuneFailureKind` 5 種は不変。

## Decision

確定した規則文言の SoT は `docs/requirements/finetune-extra.md`（FR-4 / FR-11 / FR-12）であり、
本 ADR は判断とその理由を記録する（規則の全文は転記しない）。

1. **ツール往復を chat 形式へ 1:1 の決定的写像で変換し、文脈に保持する**。`function_call` は
   tool_calls 付き assistant メッセージへ、`function_call_output` は role `"tool"` メッセージへ
   変換する。`arguments` / `output` の中身は解釈・改変せず透過する。これは ADR 0033 Decision 2 の
   うち function 系 item の破棄のみを覆す部分的な上書きであり、reasoning / compaction / 非 function
   ツール系 item（web_search_call / file_search_call 等）と生 role の system / developer / tool
   item の破棄、および Decision 1・3〜7 はそのまま存続する（変換後メッセージとして新たに生成される
   role `"tool"` は、履歴側の生 role item とは区別する）。
2. **併合は変換前の生 item 列上で直接隣接する function_call のみを対象とする**。間に他のいかなる
   item（function_call_output・破棄対象 item を含む）が挟まれば併合せず独立の assistant メッセージ
   とする。破棄対象 item を「透明」とみなして跨いで併合する特例は作らない。決定性（同じ履歴から
   常に同じ出力）と、逐次呼び出し / 並列呼び出しの忠実な表現を最優先するため。
3. **`call_id` の対応相手が存在しない孤児 item は当該 item のみ破棄する**。ケース全体をエラー・
   除外にはしない。変換は往復の対応が取れたものに限るという規則で、片側だけの不完全な
   tool_calls / tool メッセージを学習データへ混入させない。
4. **ケース化の対象はテキスト応答の assistant ターンのみとする**（不変）。変換されたツール
   メッセージは文脈（input）にのみ現れる。ツール呼び出し判断そのものの学習ケース化は将来拡張。
   これにより DPO の `response` = 文字列という前提と累積ペアリングの骨格が両モードで保たれる。
5. **DPO 生成は 2 モードとする**。`dpo_dataset_from_session(session, *, pair_builder=None)` は
   `pair_builder` 指定時に callable モード（ケース素材 `{"input", "response"}` を渡してペアを
   受け取る。`None` 返却で skip・任意キー `input` で文脈差し替え）、省略時に雛形モード（記入用
   ケース列を `DatasetBuildResult` で返す）として動作する。ケース素材のキー名は `response`（ログ上の
   実応答）とし、SFT の `expected_output`（期待出力）とは語義が異なるため別名を用いる。記入用
   ケースは `to_dpo_dataset` の入力ケース形そのものに `response` 参照欄を加えた plain dict とし、
   実応答を preferred / non_preferred のどちらにも仮置きしない（どちらに置くかは品質判定であり
   lib は内蔵しない）。新しい結果型は追加せず、雛形モードの `records` = 記入用ケース列という語義は
   docstring と要件書の用語定義で明示する。
6. **記入ワークフローは CSV / JSONL の往復とする**。`save_dpo_draft(source, path)` は拡張子で
   `.csv` / `.jsonl` を切り替え、他の拡張子は `CONFIG_MISSING`（既定形式を発明しない）。CSV の列は
   記入列（`preferred_output` / `non_preferred_output`）・参照列（`case_index` / `context` /
   `response`）・機械用列（`input_json`）で、復元は `input_json` のみが担う。読み取りは列名ベースで
   列順に依存せず、必須列の欠落は欠落列名つきの `VALIDATION_FAILED`（silent な全件 skip にしない
   fail-closed）。両記入欄が空のケースは skip、片欄のみ記入は `VALIDATION_FAILED`（書きかけの
   silent 喪失を防ぐ）。全件未記入は `DatasetBuildResult(records=(), skipped=全件)` の正常返却と
   する（ADR 0033 Decision 6 と同型・skip は失敗ではない）。
7. **CSV のエンコーディングは書き込み・読み取りとも `utf-8-sig` とする**。日本語環境の Excel が
   文字化けなく開ける実績のある形式であり、Python の `utf-8-sig` デコードは BOM なしファイルも
   受理するため、他ツールで再保存された BOM なし UTF-8 CSV の取り込みも壊れない。セル内改行・
   引用符の扱いは標準 `csv` モジュールのクオート規則に依拠し、独自エスケープを発明しない。
8. **`to_dpo_dataset` への委譲は両経路とも `skip_missing=False`（既定値）で行う**。finalize 経路は
   委譲前に「両欄記入済み + `input_json` 復元済み」を自ら確認するためキー欠落は構造上発生せず、
   `skip_missing=False` なら委譲先の値検証違反が「ケース {index}: ...」形式の `VALIDATION_FAILED`
   として表面化して fail-closed が保たれる。callable 経路は pair_builder 戻り値の形（`None` /
   dict / 両キー存在）を自関数が元ケース位置つきで検証し、値の型規則は委譲先へ一元化する
   （委譲先エラーの index は skip を含まない委譲リスト上の位置であることを docstring に明記する）。
9. **記入ヘルパ 2 つは新規モジュール `runtime/finetune/dpo_draft.py` へ置く**。Session 非接触の
   純データ + ローカルファイル I/O であり `session_dataset.py` は不適、`dataset.py` は「SDK 非接触の
   純データ層」に記入ワークフローという別責務と `csv` import を持ち込むため不適。ADR 0033
   Decision 5 と同じ「責務単位分割・公開窓口集約は不変」の原則を踏襲し、公開窓口
   `oai_agentspec.runtime.finetune` へ 3 シンボル（`dpo_dataset_from_session` / `save_dpo_draft` /
   `finalize_dpo_draft`）を追加する（コア `__all__` は不変）。
10. **未記入判定は strip した結果で行い、採用する値は非改変（strip しない）で委譲する**。
    判定と値加工を分離し、利用者が意図した前後空白・改行を lib が silent に削らないことを契約と
    する。空白のみのセルは未記入側へ倒し、スプレッドシート編集で混入しやすい空白による片欄
    エラー・空白学習データの発生を防ぐ。

## 却下案

- **フラグによる新旧挙動の並存**（`keep_tool_calls=False` 等で従来の破棄挙動を残す）: 同じ履歴から
  2 種類の出力が出る状態を恒久化し、テスト・ドキュメント・利用者の理解コストを二重化する。
  ツール往復の欠落は「正しくない出力」であり互換のために残す価値がないため却下。契約変更として
  ユーザー承認を得る方を選んだ。
- **破棄対象 item を透明扱いした併合**（reasoning 等を跨いで function_call を併合する）: 「並列
  呼び出しかどうか」の判定が破棄規則の詳細に依存し、破棄対象の集合が変わるたびに出力が変わる。
  決定性が下がるため却下。直接隣接のみという単純規則を採る。
- **`dataset.py` への記入ヘルパ同居**: 純データ層という宣言は満たすが、変換・検証の責務に記入
  ワークフローと `csv` import を混在させファイルを肥大させる。責務単位分割の原則に反するため却下。
- **`DpoCase` ベースの雛形**: 記入用ケースは `response` 参照欄を持ち `DpoCase` のフィールド集合と
  一致しない。plain dict の方が `to_dpo_dataset` への委譲と CSV / JSONL 直列化に素直であり、
  frozen dataclass の追加フィールド化は FR-2 の既存契約に触れるため却下。
- **JSONL 手編集のみのワークフロー**（`DatasetBuildResult.save()` + 利用者の手書きフィルタ）: 既存
  資産だけで実現できるが、人手記入のたびに JSON 構造の編集と未記入行の除外コードを利用者へ強いる。
  「生成→記入→取り込み→検証→submit を JSON 手編集なしで通す」という要件を満たさないため却下。

## Consequences

- + ツールを使う実運用対話が、ツール呼び出しとその結果を含む文脈のまま SFT / DPO の学習ケースへ
  変換される。tools 入り学習データの生成が持ち込み JSONL 経路だけの責務ではなくなる。
- + 会話ログから DPO preference データセットを生成できる。ペアの調達方法（callable による
  自動化 / スプレッドシートでの人手記入）を利用者が選べ、lib は品質判定・応答生成を内蔵しない
  build-don't-run の立場を保つ。
- - `dataset_from_session`（FR-4）の出力が変わる。ツール往復を含む履歴では生成ケースの input へ
  変換済みツールメッセージが加わる（ツール往復を含まない履歴では出力不変）。`case_filter` /
  `case_transform` が受けるケースにもツールメッセージが現れる。
- - tool 出力（`output` 文字列）が学習文脈・雛形ファイルへ含まれるようになるため、そこに含まれる
  機密・個人情報の除去が利用者責務として増える（除去経路は callable モードの `input` 差し替え・
  雛形モードの記入時編集・SFT 版の `case_filter` / `case_transform`。lib は自動マスキングを
  内蔵しない）。
- - 記入ワークフローが生成と取り込みの 2 呼び出しに分かれ、その間にファイルの人手編集という
  lib の外側の工程が入る。`skipped` は生成時と取り込み時で独立のカウントとなり合算されない。

## Confirmation

強制手段は次のテスト（いずれも `tests/runtime/finetune/`）。ツール往復の変換・併合・孤児破棄の分は
`docs/QUALITY-GUARANTEES.md` へ台帳行として登録済み（source = `docs/requirements/finetune-extra.md`
FR-11 / 本 ADR）。

- 変換写像（function_call → tool_calls 付き assistant / function_call_output → role `"tool"` の
  1:1 写像）と `output` の非改変透過:
  `test_session_dataset_l2.py::test_tool_roundtrip_items_are_converted_into_context_messages` /
  `::test_tool_output_payload_is_passed_through_without_text_absorption`
- 併合 / 非併合（生 item 列上で直接隣接する function_call のみ併合し、間に他 item が挟まれば非併合）:
  `test_session_dataset_l2.py::test_directly_adjacent_function_calls_are_merged_into_one_assistant` /
  `::test_function_calls_separated_by_output_are_not_merged` /
  `::test_function_calls_separated_by_dropped_item_are_not_merged`
- 孤児破棄（対応 `call_id` を欠く item のみが落ち、ケースは維持されること）:
  `test_session_dataset_l2.py::test_orphan_function_items_are_dropped_without_dropping_the_case`
- 非 function 系 item の破棄（`skipped` に数えられないこと）:
  `test_session_dataset_l2.py::test_non_function_tool_items_are_dropped_without_skipped_count`
- 文脈保持（生成ケースの input に変換済みツールメッセージが現れること）:
  `test_session_dataset_l2.py::test_tool_roundtrip_items_are_converted_into_context_messages` /
  `test_dpo_dataset_from_session_l2.py::test_draft_mode_context_contains_converted_tool_messages`
- 雛形モードの記入用ケース形（`input` / `preferred_output` = "" / `non_preferred_output` = "" /
  `response` の 4 キー）:
  `test_dpo_dataset_from_session_l2.py::test_draft_mode_returns_fillable_cases_with_four_keys` /
  `::test_draft_mode_cases_keep_exactly_four_keys`
- CSV / JSONL の round-trip（`save_dpo_draft` → 記入シミュレート → `finalize_dpo_draft`）と列順非依存:
  `test_dpo_draft_l1.py::test_csv_round_trip_produces_preference_records` /
  `::test_jsonl_round_trip_produces_preference_records` /
  `::test_finalize_reads_csv_by_column_name_regardless_of_order`
- 両欄空 skip・片欄エラー・必須列欠落エラー（欠落列名つき）:
  `test_dpo_draft_l1.py::test_blank_and_whitespace_only_cases_are_skipped` /
  `::test_all_unfilled_returns_empty_result_not_error` /
  `::test_partially_filled_case_raises_validation_failed` /
  `::test_missing_required_column_raises_validation_failed_with_column_name`
- utf-8-sig（BOM 付き書き出しと、BOM なしファイルの取り込み）:
  `test_dpo_draft_l1.py::test_save_csv_is_written_with_utf8_bom` / `::test_finalize_reads_csv_without_bom`
- 値非改変（strip は未記入判定にのみ使い、採用値の前後空白が保たれること）:
  `test_dpo_draft_l1.py::test_filled_values_are_passed_through_without_stripping`
- 読み取り専用（fake Session が `get_items` 以外を呼ばれないこと）:
  `test_session_dataset_l2.py::test_session_access_is_read_only` /
  `test_dpo_dataset_from_session_l2.py::test_session_access_is_read_only`。ネットワーク非接触は
  個別のテスト関数ではなく `tests/conftest.py` の autouse な `socket.connect` ガードが担保する
  （対応するテスト名は存在しない）。
