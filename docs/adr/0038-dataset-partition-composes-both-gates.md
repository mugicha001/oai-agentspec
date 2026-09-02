# 0038: 投入前の仕分けは両ゲートを合成し、合格側を DatasetBuildResult で返す

- Status: accepted
- Date: 2026-09-02

## Context

ADR 0037 で順序制約の検査を `screen_tool_roundtrips` として独立させ、生成は並びを理由にケースを
捨てないことにした。検出した違反を利用者がどう扱うかは利用者責務としたが、実運用で最も多い
「不合格分を除いた残りを投入し、除いた分は理由つきで手元に残して直す」という動線を、利用者が
毎回自前で組むことになっていた。

その動線は既存 API でも書けるが、2 つの摩擦がある。

1. **2 つのゲートを別々に呼び、結果を突き合わせる必要がある**。`validate_dataset` と
   `screen_tool_roundtrips` はどちらも「投入できるか」を判定するが、判定対象が異なるため両方を呼ぶ
   必要がある。片方だけで仕分けると、もう一方の違反を含むレコードが合格側へ混ざる。
2. **レポートと元データの結合を利用者が書く**。`DatasetValidationReport` は違反を `line` で
   返すため、不合格レコードの実体を得るには元データ側と `line` で join する。ファイル source
   では行番号から元レコードを引き直す必要さえある。

なお違反理由には、どのレコードかを示す `line` はあっても、レコード内のどのメッセージかを示す
情報が理由文へ入っていなかった（`validate_dataset` は `messages[3].role が不正` の形で位置を
理由文へ埋めているのに対し、`screen_tool_roundtrips` は埋めていなかった）。長い会話では `call_id` を
手掛かりに目視で探すことになる。

## Decision

1. **`partition_dataset(source, *, method="sft")` を追加し、両ゲートを合成する**。各レコードへ
   `validate_dataset` と `screen_tool_roundtrips` の判定を適用し、どちらにも違反しないレコードだけを
   合格側へ入れる。本関数は新しい判定規則を持たず、規則の SoT は FR-3 / FR-13 のままとする。
   片方だけで仕分ける案は採らない。「投入できる / できない」で分けるのが目的である以上、
   構造が壊れたレコードが合格側へ混ざる仕分けは名前と実態が食い違う。

2. **合格側は `DatasetBuildResult` で返す**。`submit_job(train=...)` と `save(path)` の既存の
   受け口へ詰め替えなしで渡せるため、利用者は新しい型の扱いを覚えずに済む。`skipped` には
   不合格件数を計上する（「除外したケース件数」という既存の語義と整合する）。plain な tuple を
   返す案は、`save()` が使えず利用者側で `DatasetBuildResult` へ包み直す手間が残る。

3. **不合格側は元レコードと理由を 1 件にまとめた `DatasetRejection` の列で返す**。`line` /
   `record` / `reasons` を持つ。レポートと元データを `line` で join する作業を利用者へ課さない
   ため、および直して再投入する動線を切らないため、レコード本体を持たせる。理由は両ゲートの
   ものを検出順に含め、「構造がおかしいのか並びがおかしいのか」を 1 箇所で読めるようにする。

4. **`raise_on_invalid` 相当を持たない**。不合格があっても例外を送出せず返却値で表す。
   fail-closed の送出は `validate_dataset` / `screen_tool_roundtrips` の同名オプションが担っており、
   仕分けの目的は「不合格があっても処理を続けて仕分けること」なので、送出は本関数の用途と
   矛盾する。

5. **`screen_tool_roundtrips` の違反理由へ位置を前置する**。`messages[N]:`（DPO は
   `input.messages[N]:`）の形式で、`validate_dataset` の既存書式に揃える。群の一致に関する
   違反は、当該群を開いた `tool_calls` 付き assistant の位置へ紐づける（違反の原因は群を
   開いた側にあり、応答が無いこと自体は位置を持たないため）。

## Consequences

- 公開シンボルが 3 つ増える（`partition_dataset` / `DatasetPartition` / `DatasetRejection`）。
  窓口 `__all__` は 20 件から 23 件になる。コア `__all__` は不変。
- 判定規則の重複は生まない。本関数は既存 2 関数の合成であり、規則を変更する場合の修正箇所は
  引き続き各ゲートの 1 箇所である。
- 合格側が `DatasetBuildResult` になることで、`skipped` の語義に「投入前の仕分けで不合格に
  なった件数」が加わる。`records` / `skipped` の意味は変えない。
- 位置の前置により `screen_tool_roundtrips` の理由文が変わる。理由文は人間可読の説明であり機械可読な
  契約ではないが、文字列で照合している利用者コードには影響しうる。
- ADR 0037 の Decision はいずれも覆さない（生成と精査の責務分離・両ゲートを統合しない判断は
  維持される。本 ADR は 2 つのゲートを呼び出し側で合成するヘルパを足すだけで、`ok` の意味論を
  引数依存にする案は引き続き採らない）。

## Confirmation

- 両ゲートの合成（片方だけでは仕分けられないこと）・合格側が `DatasetBuildResult` であること・
  不合格側が元レコードと理由を持つこと・解析不能行の扱い・source の二形と入口ガードは
  `tests/runtime/finetune/test_partition_l1.py` が固定する。screening を外す変異 / validate を
  外す変異 / 合格側を plain tuple へ変える変異 / 解析不能行を合格側へ混ぜる変異でいずれも
  RED になることを実行確認済み。
- 違反理由の位置表記は
  `tests/runtime/finetune/test_screening_l1.py::test_screen_reason_carries_message_position`
  ・`::test_screen_dpo_reason_uses_input_messages_label` が固定する（位置の前置を落とす変異で
  RED になることを実行確認済み）。
- 公開シンボルの増加が 3 件であることは `tests/runtime/finetune/test_window_l1.py` の窓口
  `__all__` の集合一致 pin が固定する。
