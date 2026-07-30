# 0012: 任意例外の宣言的着地（Failsafe）を独立機能として追加する

- Status: accepted
- Date: 2026-07-29

## Context

Runner の外側まで伝播する例外（Guardrail Tripwire 4 種・`RunBudgetExceeded`・`ToolTimeoutError` 等）を、
呼び出し箇所ごとの try/except でなく、宣言 1 回 + 単一関数で着地値へ丸めたいという要求があった。

`docs/adr/0002-resilience-declarative-compilation.md` の却下案 4「Failsafe の同梱案」は、retry / budget
（実行前の宣言コンパイル）と Failsafe（例外の着地）は関心が異なるため別機能として分離すると整理し、
`RunBudgetExceeded` を Failsafe のハンドラ対象例外の候補として予告していた。本 ADR はその分離方針に
基づき、Failsafe を `runtime/resilience` 配下の第 3 の宣言型として正式に採用する（0002 の判断を覆す
ものではなく補完するため、0002 の Status は変更しない）。

検討した選択肢:

1. **MRO 最 specific マッチ案（却下）**: `handlers` の複数キーが同時にマッチする場合、MRO 上最も
   specific な型を選ぶ。実装が複雑化し、宣言順に依存しない分マッチ結果が dict 構築順から独立せず
   直感的でない。挿入順 first-match の方が「宣言した順に試す」という単純なメンタルモデルで説明できる。
2. **`RunResult` との共通基底クラス導入案（却下）**: `FailsafeResult` と SDK `RunResult` に共通基底を
   設け多態的に扱う。SDK 型に lib 側の型階層を割り込ませることになり SDK 隔離（`agents` 非依存の
   宣言層）に反する。`final_output` の structural 互換（同名属性）で足りる。
3. **sync 版・streaming 版同梱案（却下）**: `failsafe_call` の同期版・`run_streamed` 対応版を同時に
   提供する。要件のスコープは `Runner.run`（非 streaming）のみで、streaming は `stream_events()` 消費時
   に例外が発生するため同一の着地機構では表現できない（別途ユーザー側のイベントループでの捕捉が必要）。
   YAGNI により見送り、`architecture.md` の「実行モード」節で対応範囲外を明示する。
4. **`Exception` キー許可案（却下）**: `handlers` のキーに `Exception` そのものを許可する。
   すべての例外を無差別に飲み込む宣言が容易に書け、意図しない例外の着地（本来伝播すべきバグの隠蔽）を
   誘発するため禁止する（`BaseException` / `KeyboardInterrupt` / `SystemExit` /
   `asyncio.CancelledError` / `GeneratorExit` と同様に build-time `ValueError` で拒否）。
5. **SDK `RunErrorHandlers` 拡張への相乗り案（却下）**: SDK ネイティブの `RunErrorHandlers` に
   Failsafe の機能を継ぎ足す。`RunErrorHandlers` は `MaxTurnsExceeded` / `ModelRefusalError` の 2 種に
   SDK 内部でハードコードされた dispatch であり、任意例外への拡張点を持たない。Failsafe は
   `RunErrorHandlers` と独立に、Runner 呼び出しの外側で `except Exception` により捕捉する構造とする
   （SDK が先に飲んだ例外は `RunErrorHandlers` 側で処理され、Failsafe には伝播しない関係となる）。

## Decision

`runtime/resilience` に第 3 の宣言型として Failsafe を追加する。`FailsafePolicy`（宣言）・
`FailsafeResult`（着地結果）・`failsafe_call`（単一 async 関数）の 3 シンボルで構成し、次の設計を採る。

- **SDK 非依存**: `_failsafe.py` は `agents` を import しない。例外型は利用者が `handlers` のキーと
  して持ち込む（lib 側は例外型を一切知らない）。
- **first-match（挿入順）**: `handlers` の dict 挿入順に `isinstance` で最初に一致したキーを採用する。
  MRO ベースの specificity 判定は行わない。
- **`except Exception` 限定**: 捕捉対象は `Exception` のサブクラスに限定する。`BaseException` 系
  （`KeyboardInterrupt` / `SystemExit` / `asyncio.CancelledError` / `GeneratorExit`）と `Exception` /
  `BaseException` 自体は `handlers` のキーとして宣言できず（build-time `ValueError`）、捕捉ロジック
  自体も `except Exception` に限定することで二重に防御する。
- **structural 互換の `.final_output`**: `FailsafeResult.final_output` は `RunResult.final_output`
  と同名属性でアクセス可能にし、共通基底クラスは導入しない。呼び出し元は戻り値が `RunResult` /
  `FailsafeResult` のどちらでも `.final_output` で一様にアクセスできる。
- **thunk 呼び出しは try の外**: `awaitable = thunk()` を try の外で実行し、`await awaitable` のみを
  try 内に置く。thunk 契約（`Callable[[], Awaitable[T]]`）違反（coroutine オブジェクト直渡し等）が
  送出する `TypeError` は、`handlers` に `TypeError` を宣言していても構造的に素通しし fail-fast する。

## Consequences

- + 呼び出し箇所ごとの try/except が「policy を 1 回宣言 + `failsafe_call` で包む」形に統一され、
  着地値・捕捉例外・マッチ型の監査（`FailsafeResult` / `on_apply`）が一貫して得られる。
- + `RunErrorHandlers`・retry・budget・guardrails のいずれとも独立して重畳できる（Failsafe は
  Runner 呼び出しの外側で最終防衛線として働く）。
- + 新規外部依存・新規例外型を導入しない（`resilience` extra は `[]` のまま）。
- - streaming（`run_streamed`）・sync（`run_sync`）専用の着地ヘルパーは提供しない。streaming 例外は
  従来どおり `stream_events()` 消費時に利用者側で捕捉する必要がある。
- - thunk 契約違反時の `TypeError` は着地せず素通しするため、宣言側で `TypeError` を握りつぶす目的
  では使えない（意図的な設計上の帰結）。

## Confirmation

- first-match・素通し・`except Exception` 限定・thunk 呼び出しが try 外であることの強制手段:
  `tests/runtime/resilience/test_failsafe_l1.py`（`agents` 非依存の純ロジック層検証）。
- `handlers` キーの build-time 検証（非例外値・禁止列挙・`BaseException` 系拒否）の強制手段: 同ファイル
  の `FailsafePolicy.__post_init__` 検証テスト。
- SDK 隔離の強制手段: SDK 隔離 grep（`grep -rnE "(from agents|import agents)"
  src/oai_agentspec/ | grep -v _adapters` が空であること）。
