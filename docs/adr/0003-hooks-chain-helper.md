# 0003: RunHooksBase 合成ヘルパー chain_hooks を提供する

- Status: accepted
- Date: 2026-07-22

## Context

ADR-0002 では「lib 側 hooks chain 機構案」を YAGNI として却下し、複数 `RunHooksBase` の合成は利用者責務
（docstring での案内に留める）とした。その後、`build_run_budget_hooks(policy)` が返す budget hooks と
利用者自作の hooks（ロギング・監査など）を併用したいケースが実需として確認された。

SDK `Runner.run(hooks=...)` は単数の `RunHooksBase` しか受け取らず、SDK 側に hooks 合成機構は存在しない。
このため併用のたびに利用者が `RunHooksBase` を継承した chain クラスを手書きする必要があり、
7 メソッド分のボイラープレートと合成順序・例外伝播の実装ミスを利用者側に負わせていた。

検討した選択肢:

1. **利用者責務のまま維持（ADR-0002 の判断継続・却下）**: budget hooks の併用が定型的に発生する以上、
   同一の chain クラスを各利用者が再実装する非効率が残り、UX を損なう。
2. **`return_exceptions` 方式で全 hooks を実行してから例外集約（却下）**: 各メソッドで全 hooks を
   `asyncio.gather(..., return_exceptions=True)` 的に実行し例外を集約する案。SDK の hooks は
   `Runner.run` の実行ループから逐次 await される前提で、budget hooks の例外は「即座に run を止める」
   ことに意味がある（超過後のさらなる Model 呼び出しを避ける）。全実行後集約では fail-fast が崩れ、
   前段が送出した中断意図が後段実行で希釈される。
3. **汎用 `chain_hooks(*hooks)` を提供（採用）**: 複数 `RunHooksBase` を宣言順に順次 await する薄い
   プロキシを lib で提供する。

## Decision

ADR-0002 の「lib 側 hooks chain 機構案（却下・YAGNI）」の判断を、汎用ヘルパー `chain_hooks(*hooks)` の
提供という形で **部分的に撤回**する。ADR-0002 本文は append-only のため書き換えず、Status に
`partially superseded by 0003` を追記する。撤回対象は「chain 機構を提供しない」判断のみで、ADR-0002 の
retry / budget コンパイル方針そのものは有効なまま残る。

- `chain_hooks(*hooks: RunHooksBase) -> RunHooksBase` を提供する。公開窓口は
  `oai_agentspec.runtime.hooks`、実装実体は `_adapters/hooks.py`（`_ChainedHooks(RunHooksBase)` の
  サブクラス定義に `agents.lifecycle` の import が不可避なため SDK 隔離に従い `_adapters` に閉じる）。
- 合成は 7 メソッドを宣言順に **順次 await** する。前段が例外を送出したら後段は呼ばず即伝播する
  （fail-fast）。budget hooks の中断意図を希釈しないため `return_exceptions` 方式は採らない（却下案）。
- `chain_hooks()`（0 引数）は全メソッド no-op の `RunHooksBase()` 素インスタンスを返す。
  `chain_hooks(single)`（1 引数）は `single` をそのまま返す（合成ラッパを被せない最適化）。

現在仕様の SoT は `docs/architecture.md`（Resilience 節「hooks 合成（`chain_hooks`）」）とし、本 ADR は
判断・却下案のみを記録して仕様詳細を重複させない。

## Consequences

- + budget hooks と自作 hooks の併用が `chain_hooks(budget_hooks, my_hooks)` の 1 行で書け、利用者が
  chain クラスを手書きする必要がなくなる。
- + 順次 await + fail-fast により、前段（budget）が送出した中断例外が後段実行で希釈されない。
- - `_ChainedHooks` は SDK の hook メソッド集合に追随する必要がある。SDK に hook メソッドが追加された
  際はオーバーライドを追加する保守が発生する（追随手順を module docstring に明記して担保する）。
- - `return_exceptions` 非対応のため、全 hooks を必ず実行したい用途には合致しない（意図的なトレードオフ）。

## Confirmation

- 合成挙動（7 メソッド順次 await・fail-fast 伝播・0 引数 no-op・1 引数素通し）の強制手段:
  `tests/runtime/hooks/`（`_l2` の `RunHooksBase` 実型を用いた合成挙動検証・`_l1` の窓口 import 時
  `agents` 非発火検証）。
- SDK 隔離の強制手段: SDK 隔離 grep（`grep -rnE "(from agents|import agents)" src/oai_agentspec/ |
  grep -v _adapters` が空であること）。
