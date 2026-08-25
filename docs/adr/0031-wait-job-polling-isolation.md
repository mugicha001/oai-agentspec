# 0031: FT ジョブ完了待機のポーリングループを wait_job に隔離する

- Status: accepted
- Amended by: 0032（Decision 第 1 項「暗黙の待機を一切持たない」の適用範囲を再定義）
- Date: 2026-08-19

## Context

lib は build-don't-run（宣言・build-time 検証・薄い結線に徹し、実行ループを持たない）を原則とする。
一方、マネージド fine-tuning ジョブは分〜時間単位で完了する非同期リソースであり、投入後に終端状態
（succeeded / failed / cancelled）へ到達するまで照会を繰り返すポーリングは、FT を使う全利用者が
手書きすることになる定番の boilerplate である。openai-python の `fine_tuning.jobs` には
`create_and_poll` 相当の公開 poll ヘルパが確認できず、SDK 標準機能では代替できない。

検討した選択肢:

1. **wait 非提供（`get_job` のみ）案（却下）**: ポーリングを完全に利用者責務とする。全利用者が
   同一のループ（monotonic 計測・sleep・終端判定）を手書きすることになり、`timeout` 必須という
   無限待機を構造的に排除する安全枠も lib から提供できなくなる。
2. **`submit_job` への待機同梱案（却下）**: 投入と完了待機を 1 関数に結合する。従量課金操作
   （ジョブ投入）と長時間ブロックする待機が暗黙に結合され、明示 opt-in の原則に反する。投入だけ
   して後から照会する利用形態（ジョブ id の永続化・別プロセスでの待機）も表現できない。
3. **`client.files.wait_for_processing` 転用案（却下）**: SDK 既存の待機ヘルパだが、対象は
   アップロードファイルの処理完了でありジョブ完了ではない。submit 内で暗黙にファイル処理を待つ
   ことも build-don't-run と「唯一のポーリングループ」原則に反する（file 未処理はプラットフォームが
   ジョブ側で扱う）。

## Decision

`runtime/finetune` の `wait_job` を build-don't-run の例外として採用し、lib 内唯一のポーリング
ループをこの 1 関数に隔離する。

- **明示 opt-in**: 待機は利用者が `wait_job` を明示的に呼び出したときのみ行う。`submit_job` /
  `get_job` は単発呼び出しのままで、暗黙の待機を一切持たない。
- **`timeout` 必須**: `timeout` は既定値のない必須 keyword 引数とし、無限待機の経路を構造的に
  排除する。timeout 到達時は `FineTuneError(TIMEOUT)` を送出し、ジョブは取り消さない（message に
  job_id・timeout 値・最後に観測した `raw_status` を含め、同じ job_id で `get_job` / `wait_job` を
  再実行できる旨を明記する）。
- **`poll_interval` 既定 30 秒**: FT ジョブは分〜時間単位であり、30 秒間隔なら最大 120 req/時と
  API 負荷は無視できる。60 秒でなく 30 秒としたのは短時間ジョブでの終端検知の体感を優先したため。
  sleep は `min(poll_interval, 残時間)` で deadline を超過させず、計測は `time.monotonic()` 差分で
  行う。初回照会は sleep 前に即時実行する。
- **単発照会の反復のみ**: ループの中身は `get_job` 相当の単発照会だけとし、ジョブの起動・取消・
  再試行・モデル切替・Runner 代行を持たない。未知のジョブ状態は非終端として待機を継続する
  （終端 3 種のみ判定し、状態一覧をハードコードしない）。
- **SDK 接触の隔離**: openai への接触は `_adapters/finetune.py` に閉じ、`wait_job` 本体
  （`runtime/finetune/jobs.py`）は plain データと不透明 client のみ扱う。

## Consequences

- + FT 利用者全員が手書きするポーリング boilerplate を lib 側の 1 関数に集約でき、timeout 必須と
  いう安全枠を全利用箇所へ一律に効かせられる。
- + 無限待機の経路が存在しない（`timeout` 省略はシグネチャレベルで不可）。
- + ポーリングが 1 箇所に隔離されるため、「lib 内に他の実行ループが無い」ことを grep で機械検証
  できる。
- - streaming 進捗・イベント購読による完了通知は提供しない（ポーリング以外の待機手段はスコープ外）。
- - `poll_interval` より細かい終端検知はできない（間隔を縮めるのは利用者の明示指定に委ねる）。

## Confirmation

以下を強制手段として `tests/runtime/finetune/`（jobs のテスト）に追加する:

- `timeout` が必須であること（省略時に `TypeError` となること）。
- timeout 到達で `FineTuneError(TIMEOUT)` が送出されること。
- 未知のジョブ状態文字列で待機が継続すること（非終端扱い）。
- ポーリングが `wait_job` 実装以外に存在しないこと: `asyncio.sleep` の出現が
  `src/oai_agentspec/` 配下で `wait_job` の実装箇所のみであることの grep 検証
  （対象を `src/oai_agentspec/` に限定し、テストコードの sleep 使用による偽陽性を避ける）。
