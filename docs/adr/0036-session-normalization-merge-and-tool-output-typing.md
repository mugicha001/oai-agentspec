# 0036: 会話ログ正規化の併合は射影列上の隣接で判定し、function_call_output の `output` は tool メッセージ content の型へ写す

- Status: accepted
- Date: 2026-09-01

## Context

ADR 0034 は会話ログ（SDK `Session`）のツール往復を chat 形式へ変換して文脈に保持する方針を
決め、その方針自体は妥当である。一方で、**変換後のレコードが実際にどう見えるかを実装・実測で
確認する前に Decision を確定させた**ため、次の 2 つの帰結を織り込めていなかった。

1. **併合の分裂（Decision 2 の帰結）**: Decision 2 は「変換前の生 item 列上で直接隣接する
   function_call のみ併合する」と定めた。しかし実際の Responses 履歴では、並列に呼ばれた
   function_call の間に reasoning item のような**変換後の出力に一切現れない補助 item** が挟まる。
   結果として、同時に発行された 2 つのツール呼び出しが 2 つの assistant メッセージへ分裂し、
   それぞれの `tool_calls` が 1 要素になる。この並びは「1 つ目の tool_call の直後に、対応する
   tool メッセージを挟まずに次の assistant が来る」形であり、推論時 API が拒否する並びである
   （FT のファイル検証が同じ拒否をするかは公式ドキュメントに明文がなく**未確定**）。少なくとも
   学習データとしては、逐次呼び出しでない対話を逐次の並びとして学習させる誤りになる。
2. **`output` の型 fail-open（Decision 1 の帰結）**: Decision 1 は `arguments` / `output` の
   中身を「解釈・改変せず透過する」と定めた。chat 形式の role `"tool"` メッセージは content が
   文字列であることを要求するため、`output` が dict / list / 数値の履歴、あるいは `output` キーを
   欠く履歴では、`dataset_from_session` が**成功したまま** `validate_dataset` 違反レコードを
   生成する。`validate_dataset` の要素検査はキーの**存在**しか見ないため、`content: None` を
   除けば検証も素通りし、誤りはアップロードとジョブ作成（= 課金発生）の後にプラットフォームの
   拒否としてしか露見しない。

**SDK 自身の変換規則という外部証拠**: OpenAI Agents SDK は Responses item 列 → chat messages の
変換を `agents.models.chatcmpl_converter.Converter.items_to_messages` に実装している。この実装は
`ensure_assistant_message()` で `tool_calls` を積み、**出力ターンを生む item に到達したときだけ**
`flush_assistant_message()` する。flush の呼び出しは user / system / developer メッセージ
（5 箇所）・function_call_output・走査終端の計 7 箇所にあり、**reasoning item の処理には
flush の呼び出しが無い（= reasoning は併合に対して透明）**。すなわち本 ADR が採る「射影列上の
隣接で判定する」セマンティクスは SDK 自身の変換規則と同一であり、**ADR 0034 Decision 2 は
SDK の変換規則と乖離していた**。この実装は実コードで確認済み。なお `items_to_messages` の
直接利用は採らない（NFR-1 により `session_dataset.py` は `agents` を import できず、`_adapters`
に新規窓口を設けるのは責務過大。また出力形が FT 用と異なる: `content: None` の付与・
placeholder 置換・`provider_data` 復元・非テキスト時の `UserError`）。

Decision 2 が挙げた懸念「破棄対象 item を透明扱いすると、破棄規則の詳細に出力が依存する」は、
決定性の問題ではなく**結合度**の問題だった。結合先を「その item が出力ターンを生むか否か」と
定義すれば、破棄集合が増減しても規則文言は変わらない。

`output` の型については、履歴内容を理由に生成を失敗させない方針（SDK が書いた履歴を利用者は
直せない）を維持しつつ、chat 形式が要求する型への写像だけを行う余地がある。ただし
`json.dumps` でなお直列化できない残余（循環参照・文字列化できない dict キー）は SDK 経由の
履歴では到達せず、利用者が自ら構築した値でのみ発生するため、上記方針の射程外である。

`dataset_from_session` は本ブランチの時点で公開窓口の配布物に含まれておらず、配布物経由の
外部利用者は存在しない。

ADR 0034 の Confirmation は append-only 規約により据え置く（本 ADR で覆す 2 領域についても
0034 本文は書き換えない）。当該 2 領域の現行の強制手段ポインタは、本 ADR の Confirmation と
`docs/QUALITY-GUARANTEES.md` の該当行が持つ。

制約（所与）:

- 新たな外部依存を追加しない（直列化は標準 `json` の範囲）。
- `Session` アクセスは読み取り専用のまま。SDK 接触は `_adapters/finetune.py` に閉じる（NFR-1）。
- ADR 0034 Decision 3（孤児 item のみ破棄・ケースは維持）と Decision 4（ケース化対象はテキスト
  応答の assistant ターンのみ）、および Decision 5 以降は不変。

## Decision

確定した規則文言の SoT は `docs/requirements/finetune-extra.md`（FR-4 / FR-11）であり、本 ADR は
判断とその理由を記録する（規則の全文は転記しない）。

1. **併合は「破棄対象 item を取り除いた列（射影列）の上での隣接」で判定する**。射影列上で連続
   する function_call を、生 item 列の出現順で 1 つの assistant メッセージの `tool_calls` 配列へ
   併合する。ADR 0034 Decision 2（生 item 列上の直接隣接のみ）を覆す。
2. **射影列に残るのは出力ターンを生む item である**。対応の取れた function_call_output と
   user / assistant テキスト item は射影列に残るため、それらを挟む function_call は併合せず
   独立の assistant メッセージとする（逐次呼び出しの忠実表現）。破棄対象 item = 正規化で出力
   ターンを 1 件も生まない item（dict でない item・孤児 function 系 item・function 系以外の
   ツール / 補助 item・生 role が user / assistant 以外の item）は透明として跨ぐ。
3. **決定性は保たれる**。孤児集合は `call_id` の相互突合という先行パスで履歴全体から確定し、
   破棄対象判定は「item 自身の `type` / `role` + 先行パスで確定した paired 集合」のみの関数で
   ある。射影列が一意に定まる以上、同じ履歴からは常に同じ出力が得られる。破棄集合が将来増減
   しても、結合先が「出力ターンを生むか否か」であるため規則文言は変わらない。
4. **`function_call_output` の `output` を role `"tool"` メッセージ content の型へ写す**。
   文字列はそのまま、未指定 / `None` は空文字、それ以外は
   `json.dumps(..., ensure_ascii=False, default=str)` による JSON 文字列とする。`ensure_ascii=False`
   により日本語は展開されたまま載り、`default=str` により JSON に対応しない値は**当該値のみ**
   文字列化されて外側の JSON 構造が保たれる（`json.loads` で構造を復元できる）。
5. **`json.dumps` の残余失敗は fail-closed とする**。`default=str` でもなお失敗する 2 系統
   （循環参照・文字列化できない dict キー）は、劣化した文字列を silent に載せず `call_id` と
   原因を含む `VALIDATION_FAILED` で失敗させる。「履歴内容で生成を落とさない」方針の根拠は
   「SDK が書いた履歴を利用者は直せない」ことにあり、この 2 系統は SDK 経由では到達せず利用者が
   構築した値でのみ発生するため、当該方針の射程外である。
6. **「非改変透過」を再定義する**。非改変透過とは「値の内容を解釈・要約・省略・再解釈しない」
   ことの保証であり、chat 形式が要求する型への 1:1・決定的・可逆な写像は改変にあたらない。
   `name` / `arguments` は文字列欄のため写像を要さず従来どおり非改変で透過する。本再定義は
   **FR-4 / FR-11 の履歴正規化写像にのみ適用**し、利用者供給の `input` / `expected_output` /
   `preferred_output`（FR-1 / FR-2）は型写像も行わず非改変で透過する（dict content が来たら
   `validate_dataset` が違反として検出するのが正しい）。
7. **ADR 0034 Decision 3 / 4 は不変とし、フラグによる新旧併存は設けない**。孤児 item のみ破棄・
   ケース化対象はテキスト応答の assistant ターンのみという骨格は本 ADR で変えない。

## 却下案

併合側:

- **生 item 列上の直接隣接のみで併合する（ADR 0034 Decision 2 の維持）**: 並列呼び出しが、
  出力に現れない補助 item の有無という無関係な要因で分裂する。学習データとして逐次呼び出しと
  区別がつかなくなるため却下。
- **reasoning item のみを特例で透明扱いする**: 透明にすべき item の集合を個別列挙する規則に
  なり、SDK が item 種別を追加するたびに規則と実装の両方を追記し続ける必要がある。「出力ターンを
  生まない item は透明」という一般則で同じ結果が得られるため却下。
- **function_call_output も透明として跨いで併合する**: `fc(A) → fco(A) → fc(B) → fco(B)` という
  逐次呼び出しを「同時に 2 つ呼んだ」という存在しなかった振る舞いとして学習させる。忠実表現を
  損なうため却下。
- **射影列を実体化してから 2 パスで走査する**: 追加のパスとリストを要するのに対し、走査中に
  「出力ターンを append する直前に併合状態をリセットする」だけで同値の結果が得られる。既存の
  `continue` へ落ちる構造をそのまま活かせるため、実体化は不要として却下。

`output` 側:

- **非改変で透過する（ADR 0034 Decision 1 の維持）**: dict / list / 数値の `output` や `output`
  キー欠落で、生成関数が成功したままプラットフォームが拒否するレコードを産む fail-open が残る。
  誤りは課金後にしか露見しないため却下。
- **`str()` で一律に文字列化する**: Python の repr（`{'count': 3}` / シングルクオート）は JSON
  として再パースできず、学習文脈に非 JSON の擬似構造が載る。可逆性を失うため却下。
- **`default=str` を使わず、直列化失敗時に `str()` で output 全体をフォールバックする**:
  直列化できない値が 1 つ混じるだけで、正常な兄弟キーまで含めた output 全体が Python repr へ
  劣化する。しかも `validate_dataset` もアップロードも通るため、利用者が気づく機会は学習後の
  モデル挙動（= 課金後）しかない。fail-open を作り直す方向であるため却下。
- **`_content_text`（parts 配列のテキスト吸収）を流用する**: parts 配列専用のロジックであり、
  素の配列 `[1, 2, 3]` を空文字へ潰す silent なデータ消失を起こすため却下。
- **非 str の `output` を持つ tool メッセージ自体を落とす**: 直前 assistant の `tool_calls` に
  対応する tool メッセージが消え、**dangling tool_call**（推論時 API が拒否する並び）を生む。
  片側だけを混入させないという ADR 0034 Decision 3 と正面から矛盾するため却下。
- **非 str の `output` を一律にエラーにする**: SDK が書いた履歴の内容は利用者が直せないため、
  正当な履歴で生成が不可能になる。型写像で解決できる問題を失敗へ倒すのは過剰であるため却下
  （直列化できない残余のみ Decision 5 でエラーとする）。
- **下流（`to_sft_dataset` / `validate_dataset`）で型を吸収する**: 純データ層へ Responses item の
  知識と型強制を持ち込み、利用者供給データの非改変透過（FR-1 / FR-2）まで巻き込む。責務の置き場
  として不適当なため却下。

## Consequences

- + 並列ツール呼び出しが、補助 item の有無に依らず 1 つの assistant メッセージの `tool_calls`
  として表現される。SDK 自身の変換規則と同一のセマンティクスになる。
- + 変換後の全メッセージが `validate_dataset` の合法集合に収まることを構造的に保証する
  （キーの存在検査に依存した fail-open が塞がれる）。
- + 併合規則の文言が破棄集合の増減に対して安定する（結合先が「出力ターンを生むか否か」に
  なったため、item 種別が増えても規則は変わらない）。
- - `dataset_from_session` / `dpo_dataset_from_session` の出力が変わる（併合結果の構造・非 str
  `output` の content 型）。これは**リリース前の契約改訂であり、互換措置（フラグによる新旧
  併存）は設けない**（論拠は ADR 0034 の却下案「フラグによる新旧挙動の並存」と同一）。逐次
  呼び出しのみ・`output` が str のみの履歴では出力は変わらない。
- - `float("nan")` / `float("inf")` は例外化されず、`NaN` / `Infinity` を含む**非厳密 JSON**
  として content に載る（`json.dumps` の `allow_nan` 既定に依拠する。Python の `json.loads` は
  受理するが一般の JSON パーサは拒否しうる）。到達は稀で実害が小さく、`allow_nan` を変える判断を
  縛らないため**コード対応はしない**。
- - 値および dict キーの**型情報は失われる**。非 str の `output` は JSON 文字列になり、int /
  float / bool / None のキーは `json` の既定どおり silent に文字列へ変換される（`{1: "a"}` →
  `'{"1": "a"}'`）。構造は `json.loads` で復元できるが、元の型そのものは復元できない。
- - 循環参照・文字列化できない dict キーを含む `output` では生成が `VALIDATION_FAILED` で失敗
  する。利用者が自作 Session / fake へ Python オブジェクトを入れた場合にのみ到達する。

## Confirmation

強制手段は次のテスト（いずれも `tests/runtime/finetune/`）。ツール往復の変換・併合・型写像の分は
`docs/QUALITY-GUARANTEES.md` へ台帳行として登録済み（source = `docs/requirements/finetune-extra.md`
FR-11 / ADR-0034 / 本 ADR）。ADR 0034 の Confirmation は append-only により当時の記述のまま
据え置かれており、本 ADR が覆した 2 領域（併合条件・`output` の型）の現行ポインタは本節と
上記台帳行が持つ。

- 併合（射影列上の隣接で判定し、出力に現れない item を跨いで併合すること）:
  `test_session_dataset_l2.py::test_function_calls_separated_by_dropped_item_are_merged`（ADR 0034
  時点の `..._are_not_merged` から意味を反転して改訂）/
  `::test_function_calls_separated_by_orphan_call_are_merged`（孤児 item と非 dict item を跨ぐ
  ケースを含む）/ `::test_directly_adjacent_function_calls_are_merged_into_one_assistant`
- 非併合（出力ターンを生む item は射影列に残ること・過大側の併合への退行検知）:
  `test_session_dataset_l2.py::test_function_calls_separated_by_output_are_not_merged` /
  `::test_function_calls_separated_by_text_turn_are_not_merged`
- `output` の型写像（非 str の JSON 直列化・`ensure_ascii=False`・str の素通し・欠落 / None の
  空文字化）:
  `test_session_dataset_l2.py::test_non_str_tool_output_is_serialized_into_json_string_content`
  （ADR 0034 時点の `::test_tool_output_payload_is_passed_through_without_text_absorption` から
  改訂）/ `::test_str_tool_output_is_passed_through_unchanged` /
  `::test_tool_output_key_missing_becomes_empty_string_content`
- `default=str` による部分劣化の限局と残余の fail-closed:
  `test_session_dataset_l2.py::test_non_serializable_tool_output_keeps_outer_json_structure` /
  `::test_circular_tool_output_raises_validation_failed`（kind と `call_id` を pin）
- fail-open 再発防止（構造化 `output` を含む履歴から生成したレコードが `validate_dataset` を
  違反 0 件で通ること）:
  `test_session_dataset_l2.py::test_generated_records_pass_validate_dataset_with_structured_tool_output` /
  `test_dpo_dataset_from_session_l2.py::test_generated_records_pass_validate_dataset_with_structured_tool_output`
- 変換写像そのもの（ADR 0034 Decision 1 のうち本 ADR が覆していない部分）:
  `test_session_dataset_l2.py::test_tool_roundtrip_items_are_converted_into_context_messages`
- 孤児破棄（ADR 0034 Decision 3・不変）:
  `test_session_dataset_l2.py::test_orphan_function_items_are_dropped_without_dropping_the_case`
