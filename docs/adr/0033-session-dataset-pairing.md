# 0033: 会話ログ（Session）からの SFT データセット生成は累積ペアリング + 正規化破棄規則で行う

- Status: accepted
- Date: 2026-08-26

## Context

運用中の会話履歴（SDK `Session`・不透明型）から SFT 用データセットを生成する公開 API
`dataset_from_session` の生成規則（ターンの採否・ペアリング方式・配置）を定める必要があった。

前提（Decision ではなく所与の制約）:

- **SFT 限定は要件で確定済みのスコープ**である（`docs/requirements/finetune-extra.md` FR-4。
  会話ログからの DPO preference 生成はスコープ外。preferred / non_preferred ペアは会話ログから
  機械的に決定できない）。
- **Responses API の item 構造と FT chat 形式は非互換**である。`Session.get_items()` が返す
  履歴には `function_call` / `function_call_output` / `reasoning` 等、`role` キーを持たない
  item が混在し、これらは FT chat 形式の `tool_calls` 構造（assistant メッセージ内のキー）とは
  構造が異なる。透過すると `validate_dataset` 違反レコードを生む。
- **正規化後の履歴先頭が assistant となりうる**。例: assistant の挨拶で始まる履歴・外部で
  構築された Session・compaction 配下で先行 item が role なしの compaction item
  （`{"type": "compaction"}`）に置換された履歴。compaction item が role キーを持たないことは
  SDK 実ソース `agents/memory/openai_responses_compaction_session.py:41-49` で裏取り済み。
- `dataset.py` は docstring 冒頭で「SDK 非接触の純データ層」を宣言しており、async +
  `_adapters` 遅延 import を伴う関数の同居は宣言と矛盾する。既存の分業は
  純データ = `dataset.py` / SDK 接触あり = `jobs.py`。
- 前例: `_adapters/_session_store` の turn_count は assistant のみをカウントし function_call 系
  （role キーなし item）を除外する割り切りを既に採っている。

## Decision

1. **累積ペアリングを採用する**: 正規化後の各 assistant ターンを `expected_output` とし、
   それ以前の全 user / assistant ターンを `input`（messages リスト）とするケースを
   assistant ターンごとに 1 件生成する。SFT の学習単位は「文脈 → 応答」であり、実運用対話の
   再利用には文脈保全が本質であるため。
2. **正規化の破棄規則**: role キーを持たない item（function_call / function_call_output /
   reasoning / compaction 等）と、role が `system` / `developer` / `tool` 等
   user / assistant 以外の item は破棄する（無言破棄・`skipped` に数えない）。`skipped` は
   ケース単位の除外件数の語彙であり、ターン単位の破棄を混ぜると意味が濁るため計上しない。
3. **空 input ケースの skipped 計上**: input が空になるケース（正規化後の履歴先頭が
   assistant ターン）はケースを生成せず `DatasetBuildResult.skipped` に計上する。放置すると
   `to_sft_dataset` の空リスト検査が発火して呼び出し全体が失敗するため、先行文脈なしの応答は
   学習ケースとして成立しないと定義して個別除外する。
4. **content の parts 配列はテキスト str へ吸収する**: `output_text` 等の Responses parts
   形式は FT の vision parts 形式と別物のため透過せず、テキストへ吸収する
   （`_session_store._content_text` とロジック同型を新設。sqlite 層のプライベートヘルパは
   直接 import しない）。
5. **配置は `runtime/finetune/session_dataset.py` へ分離する**: `dataset.py` の純データ層
   宣言を維持し、公開窓口 `__init__` 集約は不変のまま責務単位で分割する。
6. **filter による全ケース除外は空 `DatasetBuildResult`（skipped = 全件）の正常返却とする**:
   FR-4 のエラー条件は「履歴が空・抽出可能ターンなし」（filter 適用前の抽出段階）のみで
   filter 後の 0 件を含まず、filter は利用者供給の明示的な除外であり「暗黙に空を返す」に
   当たらない。`skipped` が全件数を示すため 0 件の理由は判別可能。
7. **本経路では `to_sft_dataset` の system 競合検出が発火しない**（dead path）: 正規化規則 2
   により履歴内の system item は常に事前除外されるため、`system=` は本経路では常に競合しない
   引数になる。学習用 system は利用者が `system=` で明示供給する（実行時 instructions 由来の
   system と学習データの system は役割が異なる）。

## 却下案

- **隣接ペア方式**（直前 user 1 件のみを input とする）: 複数ターンの文脈を捨てる。実運用
  対話の再利用という FR-4 の目的（文脈込みの応答学習）を損なうため却下。
- **最終 assistant のみ方式**（履歴全体から 1 ケースのみ生成）: 中間 assistant ターンの
  学習データ量を捨てるため却下。
- **`dataset.py` 同居 + 宣言更新案**: docstring の「SDK 非接触の純データ層」宣言を書き換えて
  同居させる案。純データ層の検証しやすい性質（SDK / async 非依存でテスト可能）を壊すため却下。
- **filter 全滅の `VALIDATION_FAILED` エラー化**: 利用者の選別方針（明示供給した filter の
  結果）を lib が失敗扱いにする過剰反応として却下。

## Consequences

- + 実運用対話が文脈込みの SFT ケース列へ 1 呼び出しで変換され、`DatasetBuildResult` として
  そのまま `save(path)` / `submit_job(train=...)` へ渡せる。
- + role なし item・system 等の破棄により、生成レコードは `validate_dataset` の SFT 合法
  集合に収まる（tool_calls 非互換構造の透過による違反レコードを構造的に排除）。
- - ツール呼び出しを含む対話の tool 往復は学習データに現れない（tools 入り学習データは
  FR-1 / FR-3 経路の責務）。
- - compaction 済み履歴では畳まれたターンが学習ケースにならない（compaction は Session
  ストアの履歴自体を置換する不可逆操作で、compaction item は role なしのため正規化で除外
  される）。利用者向け制約として `docs/architecture.md` に明記する。

## Confirmation

強制手段として次のテストを追加する（実装は未着手。実装完了後に `/spec-sync` が変異注入で
検証する前提）:

- 累積ペアリングの pin（複数ターン履歴から assistant ターン数ぶんのケースが生成され、各
  input が先行全ターンを含むこと）— `tests/runtime/finetune/`（session_dataset 対象の新規テスト）
- 空 input ケースの skipped 計上（先頭 assistant 履歴で当該ケースが生成されず skipped に
  数えられること）— 同上
- 読み取り専用（fake Session の `get_items` 以外のメソッド非呼び出し）—
  `tests/_adapters/`（`fetch_session_items` 対象）
- filter 全滅の正常返却（空 `DatasetBuildResult`・skipped = 全件）— `tests/runtime/finetune/`
