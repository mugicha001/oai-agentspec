# 0037: ツール往復の順序制約は生成から分離し submit 前の明示ゲートで検査する

- Status: accepted
- Date: 2026-09-02

## Context

会話ログからのデータセット生成（FR-4 / FR-11）は、累積ペアリングで各 assistant 応答ごとに
文脈プレフィックス（`turns[:position]`）を切り出す。この切り出しは、`function_call` と対応する
`function_call_output` の間に assistant テキストが入る履歴（HITL 承認で中断されたラン等）では
**往復の途中を切る**。結果、文脈が「応答のない `tool_calls`」で終わるケースが生じる。この並びは
推論時 API が拒否する。なお FT のファイル検証が同じ並びを拒否するかは公式ドキュメントに明文が
なく**未確定**である（ADR 0036 の Context と同じ留保）。ただし仮にファイル検証が通したとしても、
応答のない `tool_calls` を含む文脈を学習させること自体が学習データの誤りであり、本 ADR の判断は
ファイル検証の挙動に依存しない。

この欠落へ最初に採った手当ては、生成側で当該ケースを `skipped` に計上して捨てるものだった。
実装後のレビューで、その判定が集合差（要求 id 集合 − 応答 id 集合が非空なら捨てる）であり、
**生き残る側に非隣接な並びが残る**ことが判明した。集合差は「文脈のどこかに応答があればよい」
としか言わないため、`assistant(tool_calls=[c1]) / assistant テキスト / tool(c1)` のように
呼び出しと応答の間に別のメッセージが挟まる並びを合格にする。判定を隣接へ強める修正を検討する
過程で、より上位の問題として次の 3 点が浮かんだ。

1. **判定の置き場が生成側だと、持ち込みデータを守れない**。同じ並びの誤りは手で作った JSONL・
   別ツールが出力したデータ・過去に生成したファイルにも起こりうるが、生成関数の内部判定は
   それらに一切届かない。
2. **生成の挙動が不透明になる**。生成側で捨てると、利用者からは `skipped` が増えたようにしか
   見えない。何が・なぜ落ちたのかは戻り値から復元できない。
3. **skip の意味論が混ざる**。生成側の skip は本来「学習ケースとして成立しない」もの（空文脈・
   空応答）と「利用者が明示的に外した」もの（`case_filter` / `pair_builder` が `None`）の 2 系統
   だった。ここへ「形式・品質の判定」という第 3 の種類を混ぜると、`skipped` の一語が指す事象が
   広がりすぎる。

`validate_dataset`（FR-3）は既に submit 前の検証ゲートとして存在するが、これはメッセージ**単位**の
合法性（role / content / 既知キーの値規則）を見るもので、メッセージ**間**の順序は見ない。実際、
往復が非隣接なレコードは `validate_dataset` を合格で通る。準拠先も異なる: `validate_dataset` の
準拠先は OpenAI 公式の SFT / DPO データ形式であり（FR-3）、順序制約はその明文とは別の根拠に立つ。

## Decision

1. **ツール往復の順序制約の検出を、生成から分離した独立の公開関数へ移す**。新規に
   `screen_tool_roundtrips(source, *, method="sft", raise_on_invalid=False)` を追加し、submit 前の
   明示ゲートとして利用者が呼ぶ。source は JSONL ファイルパスとレコード列の双方を受けるため、
   会話ログ由来のデータと持ち込みデータを同一のゲートで検査できる。

2. **生成（FR-4 / FR-11）はツール往復の並びを理由にケースを捨てない**。生成は履歴を忠実に
   変換することへ徹する。`skipped` に計上するのは構造的欠落（空文脈・空応答）と利用者の明示
   判断（`case_filter` / `pair_builder` の `None`）の 2 系統に純化する。

3. **判定は隣接規則とし、群内は順序非依存とする**。規則 (1) `tool_calls` を持つ assistant の
   直後に連続する role `"tool"` 群の `tool_call_id` 集合が、当該 assistant の `tool_calls` の
   id 集合と一致すること（過不足なし）。規則 (2) いずれの群にも属さない role `"tool"` が
   存在しないこと。群内の順序を問わないのは、並列ツール呼び出しの対応が id ベースであり
   順序に意味を持たないためである。

   規則 (1) は**末尾の群には適用しない**。対象 messages の末尾にある `tool_calls` 付き
   assistant は「ツール呼び出しそのものを学習させる」SFT レコードの学習ターゲット本体であり、
   応答が続かないのが正常だからである。この区別を欠くと `to_sft_dataset` が生成する正当な
   レコードを不合格にし、`partition_dataset` は例外を投げないため、ツール呼び出しを学習させる
   データセットが silent に脱落したまま投入される。規則 (1) が対象とするのは、あくまで文脈
   途中で切れた往復である。

   `tool_calls` がリストでない場合はツール呼び出しの対応そのものを検証できないため違反とする
   （`validate_dataset` はキーの存在しか見ず内部構造を解釈しないため、screening が素通しすると
   不正なレコードが両ゲートを通過する）。

4. **`validate_dataset` へ統合せず、別関数として並置する**。準拠先が異なり（公式データ形式 対
   推論時 API の順序要求）、`validate_dataset` の `ok` は「違反ゼロのときのみ合格」という
   fail-closed の一義な意味を持つ。opt-in 引数で判定範囲を切り替えると、同じ `ok=True` が
   引数依存で 2 つの意味を持つことになる。利用者は 2 つを並べて呼ぶ。

5. **既存の型・エラー種別・モジュールを再利用し、新設しない**。返却は
   `DatasetValidationReport`、違反は `DatasetViolation`、`raise_on_invalid=True` の送出は
   `FineTuneError(VALIDATION_FAILED)`。実装は純データ層 `dataset.py` へ置き、行の逐次読みと
   行番号保全は既存 private `_iter_source` を再利用する。公開シンボルの増加は関数 1 個のみ。

6. **構造違反は報告しない**。レコードが非 dict / 対象 messages が欠落・非リスト / 要素が非 dict
   の場合は違反を報告せず素通しする。これらは `validate_dataset` の責務であり、両ゲートで
   同じ誤りを二重に報告すると、利用者はどちらを直せばよいか判断できなくなる。

## Consequences

- 生成関数の戻り値が変わる（往復の途中で切れる履歴で `skipped` が減り `records` が増える）。
  この経路の利用者は submit 前に `screen_tool_roundtrips` を呼ぶ必要がある。lib は submit で暗黙の
  スクリーニングを行わない（FR-3 と同じ方針）。
- 検査の対象が会話ログ由来に限られなくなる。持ち込み JSONL も同じゲートを通せる。
- 違反は `line` と理由付きで返るため、何が・なぜ弾かれたかが利用者から見える（生成側で捨てて
  いたときは `skipped` の増分としてしか観測できなかった）。
- 2 つのゲートを並べて呼ぶ手数が増える。暗黙化しないのは、検証仕様の二重管理を避け最終判定を
  プラットフォームへ委ねるという FR-3 / FR-5 の既存方針と揃えるためである。
- ADR 0034 / ADR 0036 の Decision はいずれも覆さない（本 ADR は正規化・併合・型写像の規則に
  触れず、生成後の並びを誰がいつ検査するかだけを定める）。両 ADR の Confirmation も失効しない。

## Confirmation

- 順序制約の判定規則（規則 (1) の集合一致・群内の順序非依存・規則 (2)）と、`validate_dataset`
  と同型の契約（source の二形・`line` の意味・`method` 切り替え・`raise_on_invalid`・単一 dict の
  明示エラー）、構造違反を二重報告しないことは
  `tests/runtime/finetune/test_screening_l1.py` が固定する。
- 生成側が並びを理由にケースを捨てないこと、および生成結果を `screen_tool_roundtrips` へ流すと当該
  違反が検出されることは
  `tests/runtime/finetune/test_session_dataset_l2.py::test_context_with_dangling_tool_call_is_generated_not_skipped`
  ・`::test_generated_dangling_context_is_flagged_by_screening`
  ・`::test_closed_tool_roundtrip_passes_screening`、および DPO 経路の
  `tests/runtime/finetune/test_dpo_dataset_from_session_l2.py::test_context_with_dangling_tool_call_is_generated_not_skipped`
  が固定する。
- 公開シンボルの増加が関数 1 個であることは `tests/runtime/finetune/test_window_l1.py` の
  窓口 `__all__` の集合一致 pin が固定する。
