# 0002: Resilience 宣言型は SDK ネイティブ機構へのコンパイルとして実装する

- Status: accepted
- Date: 2026-07-22

## Context

Model 呼び出しの一時失敗リトライと run 全体の予算超過制御を、実行コードの分岐でなく宣言型として
定義したいという要求があった。openai-agents SDK は Model 呼び出し retry の実行ループ本体
（`ModelRetrySettings` + `run_internal/model_retry.py`）を持つが、次の課題があった。

- `ModelRetrySettings` は `policy` 未指定時に silent no-op となる（`max_retries` を指定しただけでは
  一切 retry しない）。また retry 条件の `retry_policies.any(...)` 合成がボイラープレートで冗長
- run 全体の累積時間・累積トークン上限は SDK に存在しない（`max_turns` はターン回数のみで表現不可）
- 利用者要求として「エラーが塗りつぶされない」こと（SDK ネイティブの例外着地機構との共存で lib の
  例外が握り潰されない）が求められた

なお、任意例外の宣言的着地（Failsafe）は本機構に含めず別機能として扱う。

検討した選択肢:

1. **自前実行ループ案（却下）**: lib が `Runner.run` を包む実行 API を提供し、retry / budget を
   自前ループで制御する。build-don't-run（宣言・build-time 検証・薄い結線に徹し、実行は SDK
   `Runner.run` に寄せる）の設計原則に違反し、SDK 実行ループとの二重実装・追随保守が発生する。
2. **lib 側 hooks chain 機構案（却下）**: 利用者の既存 `RunHooksBase` と budget hooks を lib 側で
   合成する chain 機構を提供する。機械的な需要が確認できておらず YAGNI。合成は利用者責務とし、
   docstring での案内に留める。
3. **Agent 単位 budget 案（却下）**: budget を Agent 単位でも宣言できるようにする。要件が
   「Runner scope（1 回の run 全体）の累積上限」であり、Agent 単位の予算は要求に存在しない。
4. **Failsafe の同梱案（却下）**: 任意例外を宣言的に着地値へ丸める機構を本機構に同居させる。
   関心（retry / budget = 実行前の宣言コンパイル、Failsafe = 例外の着地）が異なるため別機能として
   分離する。本機構の `RunBudgetExceeded` はそのハンドラ対象例外の候補となる。
5. **pydantic での宣言型定義案（却下）**: 宣言型を pydantic BaseModel で定義する。コア宣言層
   （`AgentSpec` 等）は plain frozen dataclass が前例であり、SDK 側が pydantic_dataclass でも lib
   宣言層は plain に保つ。
6. **SDK ネイティブ機構へのコンパイル案（採用）**: 宣言型（frozen dataclass）を
   `ModelSettings.retry` / `Runner.run(hooks=...)` へコンパイルする薄い結線のみを提供する。

## Decision

Resilience 宣言型（`ModelRetryPolicy` / `RunBudgetPolicy`）を `runtime/resilience/`（agents 非依存の
宣言層）に置き、`_adapters/resilience.py` の build 関数で SDK ネイティブ機構へコンパイルする。
あわせて次の判断を採る。

- **silent no-op 排除のための policy 必須合成 + fail-fast**: `build_model_retry` はセマンティック
  フラグと `extra_retry_statuses` を `retry_policies.any(network_error(), http_status((429,)),
  http_status((500, 502, 503, 504)), retry_after(), http_status(extra))` へ合成し、必ず `policy` を
  埋める。有効条件ゼロ（全フラグ False かつ `extra_retry_statuses` なしかつ生 `policy` なし）で
  `max_retries` が正の宣言は、`retry_policies.any()`（引数ゼロ）が `never()` を返し「retry を宣言
  したのに一切 retry しない」silent no-op が再発するため（SDK 実測で確認）、矛盾宣言として
  build-time `ValueError` で fail-fast する。
- **budget は `RunHooksBase.on_llm_end` で実現する**: `on_llm_end` は streaming / 非 streaming の
  両実行経路で `context_wrapper.usage.add()` の直後に発火することを SDK 実装追跡と実測で確定した。
  hooks 内で送出した例外は SDK の gather（`return_exceptions` 未指定）により第一例外として再送出
  され握り潰されない。累積トークンは `context.usage` を読むだけとする（run_loop が `on_llm_end`
  直前に加算済みのため、lib 側の自前加算は二重計上になる）。経過時間は最初の `on_llm_start` で
  `time.monotonic()` を遅延初期化する（hooks 構築から run 開始までの待機時間を予算に混入させない・
  wall clock による時刻改変 / NTP 影響を排除する）。
- **lib 側マージ非実装**: Agent 単位 / Runner 単位の retry 設定のマージは SDK
  `_merge_retry_settings`（Runner 側が Agent 側の非 None フィールドを上書き）に完全委譲し、lib 側の
  マージ実装は作らない（SDK との二重実装・挙動乖離を避ける）。
- 例外 `RunBudgetExceeded` は plain Exception とし、SDK `error_handlers`（`MaxTurnsExceeded` /
  `ModelRefusalError` 限定の isinstance dispatch）を素通しで呼び出し元まで伝播させる
  （「塗りつぶし」なし。実測根拠あり）。

## Consequences

- + `ModelRetryPolicy(max_retries=3)` の 1 行で「まっとうな retry」が有効になり、SDK の silent
  no-op（policy 書き忘れ）が構造的に発生しない。矛盾宣言は build-time で検出される。
- + 実行は SDK 実行ループそのものであり、lib 独自ループの保守・SDK 追随コストが生じない。retry の
  マージセマンティクスも SDK と常に一致する。
- + budget 例外は SDK の例外伝播経路をそのまま通り、`RunErrorHandlers` との併用でも握り潰されない。
- - budget の enforcement は `on_llm_end` のターン境界のみ（graceful）で、tool 実行中の即中断は
  できない。ハード timeout は利用者が `asyncio.wait_for` を被せる形で補う。
- - streaming では例外が `stream_events()` 消費時まで観測されない（SDK の `_stored_exception`
  機構による）。docstring / example / 統合テストで明文化・担保する。
- - 複数 hooks の合成は利用者責務となる（chain 機構を提供しないトレードオフ）。

## Confirmation

- silent no-op 排除・fail-fast の強制手段: `tests/runtime/resilience/`（L1・宣言型検証）と
  `tests/_adapters/test_resilience_l2.py`（SDK 実型検証・有効条件ゼロ x `max_retries` 正の
  `ValueError` 確定挙動）。
- `on_llm_end` 発火保証・例外素通し・マージ委譲の強制手段: 同 L2 テスト（FakeModel + `Runner.run` /
  `run_streamed` / `run_sync` での `RunBudgetExceeded` 実測・`error_handlers` 併用の非干渉・
  Agent / Runner マージの実測）。
- SDK 隔離の強制手段: SDK 隔離 grep（`grep -rnE "(from agents|import agents)"
  src/oai_agentspec/ | grep -v _adapters` が空であること）。
