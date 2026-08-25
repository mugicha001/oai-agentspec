# 0032: submit_job のアップロード経路でファイル処理完了を待つ

- Status: accepted
- Date: 2026-08-23

## Context

`submit_job` は `train` / `val` にレコード列・`DatasetBuildResult`・`Path` を受けた場合、
`files.create(purpose="fine-tune")` でアップロードし、得たファイル id を `training_file` /
`validation_file` に載せてジョブを作成する。この経路は実 API に対して必ず失敗する。

実 API での観測:

- アップロード直後のファイルは `status="pending"` であり、その id でジョブを作成すると
  400 `The specified file reference must point to a completed file import.` が返る。
- ファイルの状態は `pending` -> `running` -> `processed` と遷移する。2.5 KB の学習データで
  `processed` 到達までおよそ 5 秒。処理完了後のファイル id ならジョブ作成は成功する。
- 同じ前提は OpenAI 本家でも成立する。ファイル処理完了がジョブ作成の前提条件であることは
  接続先によらないプラットフォーム共通の制約であり、client の種別で分岐する余地はない。

SDK 側のヘルパ `client.files.wait_for_processing(id, *, poll_interval, max_wait_seconds)` は
Azure OpenAI / OpenAI のいずれでも機能する。終端は `{processed, error, deleted}` で、
**`error` / `deleted` でも例外を送出せずファイルオブジェクトを返す**。上限時間の超過時のみ
`RuntimeError` を送出する。

ADR 0031 の Context は却下案 3（`client.files.wait_for_processing` 転用案）の理由として
「file 未処理はプラットフォームがジョブ側で扱う」と述べているが、この前提は実 API 検証により
誤りであることが判明した。却下案 3 が対象としていたのは「ジョブ完了待機の代替として file 待機を
転用すること」であり、その却下自体は妥当である一方、括弧内の前提は成立しない。

検討した選択肢:

1. **待たない案（却下）**: 現状の実装。アップロード経路が実 API で必ず失敗するため、要件
   （FR-5 のアップロード + ジョブ作成の 1 呼び出し）が満たせない。
2. **`upload_file` へ内包する案（却下）**: adapter の `upload_file` の内側で待機まで行う。
   `_adapters` は「1 関数 = 1 SDK 呼び出し」を核としており、待機を内包すると SDK 接触点と
   lib の制御が 1 関数へ混ざり、待機の有無・上限を jobs 層から制御できなくなる。
3. **lib で独自にポーリングを実装する案（却下）**: `files.retrieve` を lib のループで反復する。
   lib 内のポーリングループが 2 本に増え、ADR 0031 が確立した「lib が実装するループは 1 本」
   という隔離と、その grep 検証可能性が崩れる。
4. **上限を lib 固定にして引数化しない案（却下）**: 待機上限を lib 内定数のみで持つ。
   ADR 0031 が `wait_job` で確立した「待機の上限は利用者が明示的に制御する」という原則と
   非対称になり、大きなデータセットで上限不足に陥った利用者に回避手段がなくなる。
5. **上限超過時に `files.retrieve` を追加で打って最後の状態を得る案（却下）**: API 呼び出しと
   分岐が 1 段増えるのに対し、得られる情報（上限時点の中間状態）は利用者の次の行動
   （上限を伸ばして再実行する / ファイル id を確認する）を変えない。

## Decision

`submit_job` がデータからファイルをアップロードした場合に限り、そのファイルの処理完了を
待ってからジョブを作成する。

- **ADR 0031 Decision 第 1 項の適用範囲を再定義する**: 「`submit_job` / `get_job` は単発
  呼び出しのままで、暗黙の待機を一切持たない」は**ジョブ完了待機**についての決定であり、
  ジョブ作成の前提条件を満たすためのファイル処理完了待機は含まない。ジョブ完了待機は
  引き続き `wait_job` の明示呼び出しに限る。ADR 0031 は本項の適用範囲を除きすべて有効で
  あり、`wait_job` の設計・`timeout` 必須・`poll_interval` 既定・単発照会の反復・SDK 接触の
  隔離は変更しない。
- **待機は adapter に閉じる**: `_adapters/finetune.py` の `wait_file_processed(client,
  file_id, *, timeout)` が SDK ヘルパ `client.files.wait_for_processing` へ 1 回委譲する。
  待機ループの実体は SDK 側にあり、lib はループを持たない。`upload_file` は待機せず
  「1 関数 = 1 SDK 呼び出し」を維持し、呼び出しは `runtime/finetune/jobs.py` の
  ファイル id 解決の内側から行う。
- **上限は利用者が制御する**: `submit_job` の keyword-only 引数 `file_wait_timeout`
  （既定 300.0 秒）が `max_wait_seconds` に対応する。非正値はアップロード前に
  `FineTuneError(CONFIG_MISSING)` とし、API 呼び出しを発生させない。`poll_interval` は
  lib 定数 2.0 秒で固定する。SDK 既定値には依存せず両値を明示指定する。
- **終端の写像**: `processed` 以外の終端（`error` / `deleted`）は一律 `FineTuneError(API_ERROR)`
  へ倒す（fail-closed）。SDK の上限超過 `RuntimeError` は `FineTuneError(TIMEOUT)` へ写す。
  いずれのエラーも対象のファイル id をメッセージに含める。
- **ファイル id 経路では待たない**: 利用者が `train` / `val` にアップロード済みファイル id
  （`str`）を渡した場合、lib は状態を確認せず待機もせず当該 id をそのまま用いる。当該
  ファイルが利用可能であることは利用者責任とする。

## Consequences

- + `train` / `val` にデータを渡す経路が実 API に対して成功する（要件の中心経路が成立する）。
- + 待機ループの実装は SDK 側にあり、lib が実装するポーリングループは `wait_job` の 1 本の
  ままである（ADR 0031 の隔離と grep 検証可能性が保たれる）。
- + 上限は利用者が制御でき、失敗種別（`API_ERROR` / `TIMEOUT`）でファイル処理の失敗と
  上限超過を区別できる。
- - client の duck-type 契約が `files.wait_for_processing` まで広がる。データ経路を使う
  擬似 client（テスト・examples）は当該メソッドの実装が必要になる。
- - `train` と `val` の待機は直列に行われ、待ち時間は加算される。
- - ファイル id 経路には待機がないため、未処理のファイル id を渡した場合の 400 は
  プラットフォームエラーとして利用者へ返る。

## Confirmation

強制手段は以下のテストである。`tests/runtime/finetune/test_jobs_submit_l2.py` の行は
`docs/QUALITY-GUARANTEES.md` に登録済み（source = `docs/requirements/finetune-extra.md` FR-5）。

- `tests/runtime/finetune/test_jobs_submit_l2.py`: adapter を差し替えず client レベルの
  状態機械 fake で `submit_job` を最後まで通し、呼び出し順序
  `["files.create", "wait_for_processing", "files.create", "wait_for_processing", "jobs.create"]`
  を固定する。fake は未処理 id でのジョブ作成を実 API と同じ 400 相当で拒否するため、
  待機を削除すると RED になる（変異注入で確認済み）。`str` 経路で
  `files.create` / `wait_for_processing` のいずれも呼ばれないこと、`file_wait_timeout` が
  `max_wait_seconds` へ届くこと、非正値が API 呼び出しゼロで `CONFIG_MISSING` になることも
  同ファイルで固定する。
- `tests/_adapters/test_finetune_adapters_l2.py`: `wait_file_processed` の pin 7 件
  （`poll_interval=2.0` と `max_wait_seconds=<timeout>` の明示指定 / `processed` で id 返却 /
  `error`・`deleted` の `API_ERROR` 写像 / `RuntimeError` の `TIMEOUT` 写像とファイル id の
  同梱 / openai 例外の `API_ERROR` 写像 / `upload_file` が `wait_for_processing` を呼ばないこと）。
- `tests/runtime/finetune/test_jobs_l1.py`: jobs 層で、データ経路のアップロード -> 待機の
  順序と正しいファイル id / 上限値での呼び出し、および `str` 経路で両方が呼ばれないことを固定する。
