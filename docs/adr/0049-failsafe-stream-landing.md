# 0049: ストリーミング実行の例外を Failsafe の宣言で末尾 1 要素として着地させる（failsafe_stream）

- Status: accepted
- Date: 2026-10-03

## Context

### 出発点

ADR-0012 は Failsafe（`FailsafePolicy` で宣言した「例外型 -> 着地文言」で例外を着地させる機構）を
`failsafe_call`（単発 await 専用）として採用し、却下案 3「sync 版・streaming 版同梱案」で streaming を見送った。
見送りの理由は「streaming は `stream_events()` 消費時に例外が発生するため同一の着地機構では表現できない
（別途ユーザー側のイベントループでの捕捉が必要）」と YAGNI だった。

その後、ストリーミング実行（`Runner.run_streamed().stream_events()`）の途中で起きる例外も、非 streaming と
同じ宣言で着地させたいという要求が出た。利用者側のイベントループで捕捉して `FailsafeResult.from_exception` で
手動着地する現行の案内では、監査（`log_on_apply` の warning・`on_apply`）が発火せず、`last_agent` の決定モデルの
段 2（`FailsafePolicy.fallback_last_agent`）も使えない。宣言 1 回で着地するという Failsafe の目的を
streaming では満たせていなかった。

ADR-0012 の「同一の着地機構では表現できない」は、着地を「戻り値」として返す前提に立っていた。着地を
「ストリームの末尾 1 要素として yield して終了する」と定義すれば、0 件転送後でも N 件転送後でも同じ型で
表現でき、既配信要素も上書きされない。`runtime/conversation` の `approvals.stream_outcome` が `StreamError` を
1 件 yield して終端する形が先例である。これにより却下案 3 のうち streaming 版を翻す。sync 版
（`run_sync`）と Realtime は引き続き対象外とする。

### 依拠する SDK の前提

openai-agents 0.22.3 の `RunResultStreaming.stream_events()` は、run loop で起きた例外を `_stored_exception` に
保持し、イベント列の末尾で raise する（`agents/result.py:1076-1085`）。利用者側には `__anext__` の await 中の
例外として届く。`MaxTurnsExceeded` 等の drain 対象の例外は、キューに残ったイベントを流し切ってから raise する
（`:1000-1011`）。`asyncio.CancelledError` は `cancel()` を自ら呼んでから再送出する（`:1019-1022`）。

この経路により、`__anext__` の例外だけを捕捉対象にすれば SDK の streaming 例外を漏れなく着地対象にでき、
「N 件転送後の着地」は SDK の実経路でも起きる。`CancelledError` を素通しすれば SDK 側のキャンセル処理が
完結する。SDK がこの経路を変えた場合は L2 テストが検知する。

また `stream_events()` の `aclose()` は閉じるだけではない。`yield`（`:1040`）で `GeneratorExit` を受けると
`finally`（`:1043-1055`）に入り、`cancel()` が呼ばれていなければ run loop の完了を待つ（`:1055`）。
`cancel()` 済みなら即時に後始末して返る（`:1051`）。

### 既存資産・SDK 標準機能の棚卸し

| 既存資産 / 標準機能 | 採否 | 理由 |
|---|---|---|
| `failsafe_call(policy, lambda: consume_all(stream))`（ストリームを全消費する coroutine を thunk で包む） | 不採用 | 全消費後に 1 値を返す形になり逐次配信が消える。既配信イベントを呼び出し側へ届けられない |
| `FailsafeResult.from_exception` を利用者が `stream_events()` の except で使う | 不採用（併用は可） | 監査（`log_on_apply` / `on_apply`）が発火せず、段 2（`fallback_last_agent`）も使えない。宣言 1 回で着地する目的を満たさない |
| SDK `Runner.run_streamed(error_handlers=...)`（`RunErrorHandlers`） | 不採用 | `MaxTurnsExceeded` / `ModelRefusalError` の 2 種に SDK 内部でハードコードされた dispatch で、任意例外へ拡張できない（ADR-0012 却下案 5）。Failsafe と独立に併用でき、SDK が先に処理した例外は Failsafe に届かない関係は `failsafe_call` と同じ |
| SDK `RunResultStreaming.cancel()` | 不採用（案内のみ） | 利用者がストリームを止める手段であり、例外の着地ではない。利用者によるキャンセルを着地として扱わない方針と整合するため、早期終了の手順として docs で案内する |
| `approvals.stream_outcome`（`runtime/conversation`） | 不採用（パターンのみ踏襲） | `runtime/conversation` 専用型（`StreamEvent`）に依存し、`runtime/resilience` から参照すると単方向依存に反する。「着地を末尾 1 要素として yield して終端」のパターンだけを踏襲し、要素型は `FailsafeResult` にする |
| `contextlib.aclosing` | 不採用（同等処理を duck typing で） | `aclose` を持つ対象にしか使えない。`AsyncIterable[T]` の契約は `aclose` を保証しないため、`getattr` で有無を見て呼ぶ |
| `_adapters/serialization.py` の `run_streamed_outcome`（本体要素と終端要素を Union で 1 本の `AsyncIterator` に流す） | 不採用（型設計の先例として参照） | SDK 型に触れる adapter であり、Failsafe の SDK 非接触（ADR-0012）を崩す理由がない。Union で 1 本に流す型設計はこの先例と同型 |

### 却下した案

| # | 案 | 却下理由 |
|---|---|---|
| 1 | thunk（`Callable[[], AsyncIterable[T]]`）を受ける / thunk と iterable の両方を受ける | async generator object は最初の `__anext__` まで実行されないため thunk の遅延効果がない。両方受理は callable 判定の分岐と曖昧性（callable かつ async iterable）を生む。`aiter()` を try の外で 1 回呼ぶことで thunk 契約の目的（受理契約違反の fail-fast・1 回性）は満たせる |
| 2 | 0 件転送時は `FailsafeResult` を戻り値で、N 件転送後は追加イベントで返す（非対称 API） | async generator は戻り値を持てず、利用者に 2 つの受け取り方を強いる。0 件時も末尾 1 要素と見れば同じ型で表せ、利用者は受信済み件数で判別できる |
| 3 | `FailsafeResult` に転送済み件数等の属性を追加する | `FailsafeResult` の公開フィールド集合・`from_exception` との同一検証・`repr` マスクの保証に触れる。利用者側に既にある情報の複製である |
| 4 | 着地文言（`T` の値）をそのまま yield する | `str` のストリームでは通常のデルタと区別できない。`matched_type` / `exception` / `last_agent` を運べず、フォールバック agent への差し替えを表現できない |
| 5 | ストリーミング専用の Policy 型 / 専用イベント型を新設する | 宣言の二重化になる。`runtime/conversation` の `StreamEvent` の流用は単方向依存に反する |
| 6 | `_adapters/resilience.py` に SDK `stream_events()` 専用のラッパとして実装する | SDK 型に触れる必要がなく、汎用 `AsyncIterable[T]` で成立する。SDK 非接触を崩す理由がない |
| 7 | 着地後も source の消費を続ける（複数回着地） | 例外を送出した async generator は終了しており再開できない。再試行・継続は lib の責務外（build-don't-run） |
| 8 | ADR-0012 の却下案 3 を維持する（提供せず、利用者が `from_exception` で手動着地する） | 監査と段 2 が使えず、宣言 1 回で着地する目的を満たさない。ADR-0012 の「同一の着地機構では表現できない」は「末尾 1 要素として yield」で解消できる |

### 例外集合の名称

要求の記述で使われた `_EXCLUDED_EXCEPTION_TYPES`（`KeyboardInterrupt` / `SystemExit` / `asyncio.CancelledError` /
`GeneratorExit`）は、実装上の `_FORBIDDEN_HANDLER_TYPES`（`_failsafe.py`）の部分集合に当たる。新しい集合は作らない。

## Decision

1. **公開形**: `runtime/resilience` に
   `failsafe_stream[T](policy: FailsafePolicy, source: AsyncIterable[T]) -> AsyncIterator[T | FailsafeResult]` を
   追加する。公開関数は plain `def` で、`aiter(source)` を呼び出し時点・try の外で 1 回だけ実行し、private な
   async generator（`_relay`）を返す。受理契約は `AsyncIterable[T]` のみ（thunk は受けない）。非 async iterable は
   Python ランタイム由来の `TypeError` で呼び出し時点に fail-fast し、`handlers` に `TypeError` を宣言していても
   着地しない。lib 独自のメッセージは足さない。
2. **着地の表現**: 宣言済み例外を捕捉したら、0 件転送後でも N 件転送後でも `FailsafeResult` を末尾 1 要素として
   yield して終了する。既配信要素は上書きしない。正常要素は同一オブジェクトをそのまま yield し、正常完了時は
   `FailsafeResult` を出さない。判別は `isinstance(ev, FailsafeResult)` で、0 件か N 件かは利用者が受信済み件数で
   判別する。着地文言そのもの（`T` の値）は yield しない。
3. **捕捉範囲と協調キャンセル**: `await anext(iterator)` のみを try 内に置き、`except StopAsyncIteration` で正常
   終了し、`except Exception` で捕捉する。`yield` は try の外に置く。`KeyboardInterrupt` / `SystemExit` /
   `asyncio.CancelledError` / `GeneratorExit` は構造的に素通しする（policy 側の build-time 拒否との二重防御は
   ADR-0012 と同じ）。利用者の `async for` 本体の例外・`athrow()` で投げ込まれた例外・`Runner.run_streamed(...)`
   の呼び出し自体が同期送出する例外は着地対象外である。`StopAsyncIteration` を `handlers` に宣言しても正常
   終了として扱い着地させない。
4. **`aclose` の転送**: `finally` で source の `aclose` があれば await する。ラッパを挟んでも、1 要素以上を要求
   した後の明示 `aclose()`（`contextlib.aclosing` 経由を含む）が source へ届き、素の `stream_events()` を直接閉じた
   場合と同じ決定的な解放を保つ。反復を始める前の `aclose()` と GC では、async generator の本体が実行されない
   ため転送しない（`aiter(source)` は呼び出し時点で済んでいるため、source の `__aiter__` で確保した資源は利用者側
   で解放する）。`aclose()` 自身の例外は着地させずに伝播する。lib は `RunResultStreaming.cancel()` を代行
   しない。
5. **`handlers` が空のとき**: 専用分岐を置かず透過する。例外はそのまま伝播し、監査は発火せず、`__cause__` は
   付かない（`failsafe_call` の空分岐と観測上同一）。`aiter()` の fail-fast は `handlers` の有無に関わらず同じ
   位置で起きる。
6. **監査経路の共通化**: `failsafe_call` の捕捉後の処理を、挿入順 first-match の照合（`_match_handler`）と、
   fallback の解決・`last_agent` の決定・`FailsafeResult` の構築・warning・`on_apply` の実行（`_land`）の
   2 つの private 関数に抽出し、両関数から呼ぶ。抽出は文の移動のみで、`failsafe_call` の挙動は変えない。
   未一致時の bare `raise` は各関数の except 節内に置く（`from` を付けない）。
7. **例外集合**: 新しい集合は作らない。宣言の拒否は `FailsafePolicy.__post_init__`（`_FORBIDDEN_HANDLER_TYPES`）、
   捕捉の除外は `except Exception` が担い、`failsafe_stream` は集合を参照しない。
8. **配置と SDK 隔離**: `_failsafe.py` に置き（Failsafe と同一責務・同一の private 関数を共有する）、`agents` を
   import しない。`_adapters` への追加はしない。公開窓口 `oai_agentspec.runtime.resilience` の直 import 側に
   加える。コア `__all__` は変えない。

## Consequences

- + ストリーミング実行でも、非 streaming と同じ `FailsafePolicy` 1 つで宣言済み例外を着地させられる。
  既配信要素は保たれ、着地は末尾 1 要素として届く。
- + 監査（`log_on_apply` の warning・`on_apply`）と `last_agent` の決定モデル（段 1 / 段 2）が、`failsafe_call` と
  同じ private 関数を通るため 1 本の経路にまとまり、両関数で意味が揃う。
- + `failsafe_call` と `FailsafePolicy` / `FailsafeHandler` / `FailsafeResult` / `RUNNING_AGENT` の契約は
  変わらない（純粋追加）。新しい外部依存・例外型・Policy 型を持ち込まない。
- - sync（`run_sync`）・Realtime（`RealtimeRunner` / `RealtimeSession`）専用の着地ヘルパーは引き続き提供しない。
- - `aclose` の転送は run 本体を止めない。`cancel()` を呼ばずに `aclose()` すると、SDK の `stream_events()` は
  run loop の完了を待つ。途中で早期終了したい場合は、利用者が `RunResultStreaming.cancel()` を先に呼んでから
  `aclose()` する必要がある。`async for` の break は `aclose()` を呼ばず、決定的な解放の手段にならない。
- - 着地は `__anext__` の例外に限るため、利用者の `async for` 本体の例外は着地しない（利用者側の try/except の
  責務のまま）。
- - build-don't-run の逸脱は、利用者が渡すストリームを 1 回だけ消費して中継する薄い結線として、`failsafe_call` と
  同じ例外 (2) に並記する（逸脱の件数は増やさない）。

## Confirmation

保証する性質と強制手段（nodeid）の対応は次のとおり。`::` で始まる nodeid は
`tests/runtime/resilience/test_failsafe_stream_l1.py`（`agents` 非依存の L1）の関数を指す。

- 既配信を保ったまま末尾 1 要素で着地する: `::test_failsafe_stream_N件転送後の宣言例外は既配信の末尾にFailsafeResultが付く` /
  `::test_failsafe_stream_0件転送で宣言例外ならFailsafeResult1件のみを受け取る` /
  `::test_failsafe_stream_着地後は終了しsourceの続きを読まない`
- 未宣言の例外は着地せず伝播する: `::test_failsafe_stream_未宣言例外は既配信の後にそのまま伝播し監査しない`
- 捕捉は `except Exception` 限定（二重防御の 2 段目）: `::test_failsafe_stream_宣言検証を迂回したCancelledErrorキーでも着地せず伝播する` /
  `tests/runtime/resilience/test_failsafe_l1.py::test_failsafe_call_宣言検証を迂回したCancelledErrorキーでも着地せず伝播する`。
  協調キャンセルの素通しは `::test_failsafe_stream_sourceのCancelledErrorは着地せず伝播する`
- 正常終了は着地しない: `::test_failsafe_stream_StopAsyncIterationを宣言しても正常終了は着地しない`
- 受理契約は呼び出し時点・try の外で 1 回: `::test_failsafe_stream_非async_iterableは呼び出し時点でTypeErrorになり着地しない` /
  `::test_failsafe_stream_coroutineは呼び出し時点でTypeErrorになり着地しない` /
  `::test_failsafe_stream_aiterは呼び出し時点の1回だけ呼ばれる`
- `yield` は try の外: `::test_failsafe_stream_athrowで投げ込まれた宣言例外は着地せず伝播する`
- `aclose` の転送: `::test_failsafe_stream_明示acloseはsourceのfinallyへ転送され着地しない` /
  `::test_failsafe_stream_反復開始後の終了経路ではsourceのacloseへ1回転送する` /
  `::test_failsafe_stream_反復前のacloseはsourceのacloseへ転送しない`
- SDK の `stream_events()` との統合（上記「依拠する SDK の前提」の変化も検知する）:
  `tests/_adapters/test_resilience_integration_l2.py::test_B2_failsafe_streamはstream_events中の予算超過を既配信の末尾へ着地させる`
- `_match_handler` / `_land` を抽出しても `failsafe_call` の挙動が変わらないこと: 既存の
  `tests/runtime/resilience/test_failsafe_l1.py` が回帰で保証する

台帳との相互参照: `docs/QUALITY-GUARANTEES.md` に登録済み（source = ADR-0049）。
SDK 隔離は SDK 隔離 grep（`grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` が
空であること）が保証する。
