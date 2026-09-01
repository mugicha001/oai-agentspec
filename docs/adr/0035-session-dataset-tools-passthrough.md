# 0035: 会話ログ経路はツール定義を復元せず、利用者供給の tools を委譲先へ透過する（雛形モードでの指定は fail-closed）

- Status: accepted
- Date: 2026-09-01

確定した規則文言の SoT は `docs/requirements/finetune-extra.md`（FR-4 / FR-11 / FR-12）であり、
本 ADR は判断とその理由を記録する（規則の全文は転記しない）。

## Context

FR-11 は「`Session` にはツール定義（スキーマ）が記録されない」ことを理由に、会話ログ経路での
ツール定義の供給をスコープ外と決めていた。この判断は「会話ログからツール定義を**復元**する」
ことについては今も正しい（履歴に材料が無く実現不能）。

一方で ADR 0034 Decision 1 がツール往復（`function_call` / `function_call_output`）を chat 形式へ
変換して文脈に保持するようにしたため、会話ログ経路の出力は「`tool_calls` 付き assistant
メッセージを含むが、レコードには `tools` 定義が無い学習データ」になった。推論時に使うツール定義は
利用者の手元（コア `ToolRegistry` 由来の `FunctionTool` 相当オブジェクト、または plain dict）に
存在するため、これは材料不足ではなく**受け渡し口の不在**である。

要件書冒頭が掲げる価値「学習データと推論時のツール定義の一致を構造的に担保」は、既存の
`to_sft_dataset` / `to_dpo_dataset`（FR-1 / FR-2）が `tools=` / `parallel_tool_calls=` として
既に実装しており、会話ログ経路はその委譲先を呼んでいる。したがって不足しているのは上位 3 関数の
引数だけである。

回避策として利用者が「結果を捨てて `to_sft_dataset` / `to_dpo_dataset` を呼び直す」ことは、雛形
ワークフロー（FR-12）では成立しない。利用者の手元にあるのは finalize の戻り値（最終レコード）で
あり、委譲へ渡すケース列は CSV の `input_json` から自力で復元する必要があるため、FR-12 が掲げた
「JSON 手編集なしで機能全量を通す」が崩れる。

制約（所与）:

- 写像（`FunctionTool` 相当 → FT の tools 定義形式）と不正要素の検証は `dataset.py` に閉じており、
  この一元化を崩さない（ADR 0034 Decision 8 と同方向）。
- 雛形ファイルの CSV 列構成（ADR 0034 Decision 6）と記入用ケースの形（同 Decision 5）は不変。
- 公開窓口の増分シンボルを増やさず、コア `__all__` も不変に保つ。

## Decision

1. **`dataset_from_session` / `dpo_dataset_from_session` / `finalize_dpo_draft` の 3 関数へ
   keyword-only の省略可引数 `tools=` / `parallel_tool_calls=` を追加し、委譲先の同名引数へ
   そのまま透過する**。上位層は `_map_tools` を呼ばず、写像・検証・キー非出力（`None` 指定時）の
   規則はすべて `to_sft_dataset` / `to_dpo_dataset` 側に一元化したままとする。`parallel_tool_calls`
   は `False` を有意な指定として扱い `None`（未指定）と区別する。組み合わせ検証（tools と
   parallel_tool_calls の整合）はプラットフォーム仕様であり lib は行わない。
   `save_dpo_draft` には追加しない（書き出しは記入用ケースの直列化であり、反映先は最終変換側）。
   SFT 経路にも足すのは、ツール往復の変換保持が SFT にも入ったため「`tool_calls` はあるが tools
   定義が無い」不整合が DPO と同型に生じるからで、片側だけへ足す非対称は説明コストの方が高い。
2. **雛形モード（`pair_builder` 省略時）での `tools=` / `parallel_tool_calls=` の指定は
   `CONFIG_MISSING` で拒否する**。判定は `session is None` チェックの直後・履歴読み取り
   （`fetch_session_items`）より前に置き、反映先の無い指定で履歴を読ませない。記入用ケース列は
   `to_dpo_dataset` へ委譲しないため透過先が存在せず、silent に無視すると tools 無しの学習データが
   例外なく完成して誤りが学習後（課金済み）に露見する。エラーメッセージには供給先
   （`finalize_dpo_draft`）を含め、利用者が次の行動を取れる形にする。
3. **雛形ファイルはツール定義を持ち回らない**。記入用ケースのキー集合は 4 キーのまま、CSV の列
   構成も 6 列のまま不変とし、雛形ワークフローにおけるツール定義の供給は `finalize_dpo_draft` の
   引数へ一本化する。
4. **採用ケースが 0 件の経路でも委譲を省略しない**（早期 return を置かない）。委譲先は tools の
   写像・検証をケースのループより前に行うため、0 件でも不正な `tools=` は `VALIDATION_FAILED` と
   して表面化する。空リスト委譲の戻り値は `DatasetBuildResult(records=(), skipped=0)` であり、
   呼び出し側が `skipped + result.skipped` で合成する既存の集計と観測同値であるため、正常系の
   返却値は変わらない。

## 却下案

- **現状維持 + docstring での明記**（会話ログ経路では tools を渡せないと書くだけ）: 雛形
  ワークフローの利用者は委譲へ渡すケース列を手元に持たず、CSV の `input_json` からの自力復元を
  強いられる。FR-12 の「JSON 手編集なしで機能全量を通す」が成立しないため却下。
- **雛形の記入用ケース / CSV へ tools を持ち回る**: ADR 0034 Decision 6 の CSV 6 列契約と
  `input_json`（= 累積文脈 messages の JSON）の語義に触れる。定義は呼び出し単位で全レコード共通
  なのに全行へ複製することになり、行ごとの編集で破損しうる。さらに `FunctionTool` 相当
  オブジェクトは直列化できず、写像タイミングが経路ごとにずれる。得られる利便性は「finalize で
  引数を 1 つ渡す」の省略のみで割に合わないため却下。
- **雛形モードでの指定を黙って無視する**: silent failure。tools 無しの学習データが例外も警告も
  なく完成し、誤りはジョブ完了後に露見する（課金済み）。fail-closed を選び却下。
- **上位層で `_map_tools` を先取りして呼び、写像済み dict を委譲へ渡す**: `dataset.py` の private
  関数を跨いで呼ぶことになり、写像・検証規則の一元管理（ADR 0034 Decision 8 と同方向）を崩す。
  Decision 4（早期 return を置かない）で同じ fail-closed が得られるため却下。

## Consequences

- + 会話ログ経路だけで tools 入りの学習データが完成する。推論時のツール定義（`ToolRegistry` 由来）を
  同一ソースから学習データへ載せられ、ツール往復の文脈保持と定義の供給が同じ経路で揃う。
- - 雛形モードの利用者はツール定義を `finalize_dpo_draft` 側で渡す必要があり、ツール定義に関する
  知識が生成・取り込みの 2 呼び出しに分かれる（記入ワークフローが 2 呼び出しである既存の性質に
  従う）。
- - 「採用ケース 0 件 + 不正な `tools=`」の組み合わせが新たにエラーになる（従来は早期 return で
  素通りした）。tools を指定しない呼び出しと、正しい tools を渡す呼び出しの観測結果は変わらない。

## Confirmation

強制手段は次のテスト（いずれも `tests/runtime/finetune/`）。雛形モードの fail-closed 分は
`docs/QUALITY-GUARANTEES.md` へ台帳行として登録済み（source = `docs/requirements/finetune-extra.md`
FR-11）。

- 透過位置の階層差（SFT はレコード直下 / DPO は `input` 内）:
  `test_session_dataset_l2.py::test_tools_pass_through_to_record_top_level` /
  `test_dpo_dataset_from_session_l2.py::test_tools_pass_through_inside_input` /
  `test_dpo_draft_l1.py::test_tools_pass_through_inside_input` /
  `test_dpo_draft_l1.py::test_tools_pass_through_from_csv_source`（CSV source 経路）
- `FunctionTool` 相当オブジェクトの写像が委譲先で行われること:
  各ファイルの `::test_tools_function_tool_like_is_mapped_by_delegate`（3 ファイル同名）
- 不正要素の `VALIDATION_FAILED`:
  `test_session_dataset_l2.py::test_invalid_tools_element_raises_validation_failed` /
  `test_dpo_dataset_from_session_l2.py::test_invalid_tools_element_raises_validation_failed` /
  `test_dpo_draft_l1.py::test_invalid_tools_element_raises_validation_failed`
- 空結果経路でも委譲すること（早期 return の復活を検知）:
  `test_session_dataset_l2.py::test_invalid_tools_raises_even_when_filter_excludes_all_cases` /
  `test_dpo_dataset_from_session_l2.py::test_invalid_tools_raises_even_when_all_cases_are_skipped` /
  `test_dpo_draft_l1.py::test_invalid_tools_raises_even_when_all_cases_are_unfilled`
- `parallel_tool_calls=False` が透過されること（truthy 判定への退行を検知）:
  `test_session_dataset_l2.py::test_parallel_tool_calls_false_is_passed_through` /
  `test_dpo_dataset_from_session_l2.py::test_parallel_tool_calls_false_is_passed_through_inside_input` /
  `test_dpo_draft_l1.py::test_parallel_tool_calls_false_is_passed_through_inside_input`
- 省略時にキー自体が出ないこと（`None` → `False` 型の混入変異を検知）:
  各ファイルの `::test_tool_keys_are_absent_when_arguments_are_omitted`（3 ファイル同名）
- 雛形モードの fail-closed（履歴読み取り前に `CONFIG_MISSING`・`parallel_tool_calls=False` 単独でも
  拒否）: `test_dpo_dataset_from_session_l2.py::test_draft_mode_with_tools_raises_config_missing_before_reading_session` /
  `::test_draft_mode_with_parallel_tool_calls_false_raises_config_missing`
- 記入用ケースが 4 キーのまま（tools を持ち回る変異を検知）:
  `test_dpo_dataset_from_session_l2.py::test_draft_mode_cases_keep_exactly_four_keys`
- 引数が keyword-only かつ既定 `None`（実行時無症状の位置引数化・既定値変更を検知）:
  `test_session_dataset_l2.py::test_dataset_from_session_tool_arguments_are_keyword_only_with_none_default` /
  `test_dpo_dataset_from_session_l2.py::test_dpo_dataset_from_session_tool_arguments_are_keyword_only_with_none_default` /
  `test_dpo_draft_l1.py::test_finalize_dpo_draft_signature_is_source_positional_and_keyword_only_tools`
