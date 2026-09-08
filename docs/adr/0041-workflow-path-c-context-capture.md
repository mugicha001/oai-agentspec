# 0041: 経路C の実行コンテキストを lib 所有 AgentHooks で捕捉して内部インタプリタへ渡す

- Status: accepted
- Date: 2026-09-08

## Context

`WorkflowGraph.as_agent_spec`（経路C）は `WorkflowModel` を `AgentSpec.model` に据え、
`Model.get_response` の中で内部インタプリタを回す。SDK の `Model` プロトコルは
`get_response` / `stream_response` に実行コンテキストの引数を持たないため、利用者が
`Runner.run(context=...)` に渡したオブジェクトは経路C の内部ノード（FUNCTION ノードの `ctx` /
router / ノード前後フック / AGENT ノード内側 run）へ届かない状態だった。tool ファサード
（経路A / D）は `on_invoke_tool(tool_context, ...)` で `ToolContext` を受け取れるため、
context 透過は経路A / D のみで成立し、経路C を選ぶと `instructions_append` / hooks /
ガードレールが外側 context を読めないという非対称が利用者に見えていた。

SDK 0.17.4 の実装を実機検証して分かった制約:

- SDK は `AgentHooks` を `asyncio.gather` 経由の別 Task で呼ぶ。hook 内で set した
  `contextvars` の値は `get_response` を呼ぶ親 Task へ戻らない（contextvars 方式は不成立）。
- SDK は実行コンテキストのラッパーを contextvar として公開しておらず、`Model` 実装から
  自力で取得する手段がない。hook の引数として受け取るしかない。
- `AgentSpec` から agent 単位で載せられるライフサイクルのスロットは `hooks`
  （`Agent.hooks`）だけである。`Runner.run(hooks=...)` / `RunConfig` は run 単位で利用者
  所有のスロットであり、lib が宣言側から結線できず build-don't-run（`Runner` 引数へ lib が
  介入しない）にも反する。
- `agents.tracing.get_current_span()` は、同一 turn の `AgentHooks.on_start` と
  `get_response` / `stream_response` で同一の span オブジェクトを返す（SDK 0.17.4 では
  turn span。tracing 無効時は `NoOpSpan` で agent 呼び出しごとに一意）。span オブジェクトは
  weakref 可能で `__eq__` / `__hash__` は identity。

検討した受け渡し手段（本 ADR で却下したもの）:

- `WorkflowModel` へ context 引数を追加する / 公開シグネチャへ context を通す: SDK の
  `Model` プロトコルが呼び出し側であり lib が引数を増やせない。
- `Agent.instructions` を callable にして捕捉する: 捕捉自体は可能だが、利用者の
  instructions スロットを lib が占有し `PromptStore` / `instructions_append` と衝突する。
- `contextvars`（hook で set → `get_response` で get）: 上記のとおり別 Task のため不成立。
- `on_llm_start` の `input_items` と `get_response` の `input` の同一 list を run スコープの
  キーにする: 成立するが list は weakref 不可・unhashable のため `id()` キー + 手動削除が
  必要になる。入力ガードレールの tripwire 等で `get_response` に到達しない経路では削除の
  機会がなく残留し、`id` の再利用で別 run に旧コンテキストが当たりうる（run 間分離の破れ）。
- モジュールグローバルの共有捕捉レジストリ: プロセス内の全ワークフローが 1 テーブルを共有し、
  spec 破棄後も生存する。別ワークフローの hook が別 model へ配線される余地も生じる。
- 「最後に捕捉したコンテキスト」を単一スロットで持つ: 並行 run で混線する。
- 捕捉不能時に警告・例外を出す fail-fast: 直接 `get_response` を呼ぶ既存の利用形と
  `hooks` 上書きを壊す。run ごとの縮退ログは利用者の観測を汚す。
- build 時に利用者 hooks を lib 側で自動合成する: 利用者が渡していない合成を lib が行うと
  `spec.hooks` の所有関係が曖昧になり、合成順の期待も lib 側に埋まる。

## Decision

経路C の実行コンテキストを、lib 所有の `AgentHooks` で run 開始時に捕捉し、
`WorkflowModel` から内部インタプリタの `context` へ渡す。

- `as_agent_spec` は lib 所有の捕捉オブジェクトを 1 つ作り、その `AgentHooks` を
  戻り値の `AgentSpec.hooks` に載せ、同じ捕捉オブジェクトの解決関数を `WorkflowModel` へ
  注入する（1 `WorkflowModel` インスタンスにつき 1 捕捉テーブル。共有レジストリは持たない）。
- 捕捉は `AgentHooks.on_start` のみで行う。`on_start` は agent 起動ごとに 1 回であり、
  turn span が current になった後で `get_response` と同一の現在 span を観測する。
  `on_llm_start` には置かない（turn 数に比例して発火し、捕捉回数が run あたり 1 回でなくなる）。
- run スコープのキーは `agents.tracing.get_current_span()` の返す現在 span とし、
  捕捉テーブルは `WeakKeyDictionary`（span → 実行コンテキストのラッパー）で保持する。
  依拠する不変条件は「`on_start` と同一 turn の `get_response` / `stream_response` は同じ
  現在 span を観測する」であり、span の種類名（turn span / agent span 等）には依拠しない。
  span が GC されるとエントリは自動的に消える。
- `WorkflowModel.get_response` は解決関数を 1 回呼び、得られたラッパーを
  `interpret(..., context=...)` へ渡す。`stream_response` は `get_response` へ委譲する
  現行構造のため、streaming でも同じ経路で伝播する。
- 捕捉したラッパーは詰め替えずそのまま素通しする。FUNCTION ノード / router / ノード前後
  フックは `RunContextWrapper`（実型は SDK のサブクラス）を受け取り、AGENT ノードは既存の
  context 剥がし（`Runner.run(context=<生オブジェクト>)`）を経て生オブジェクトを受け取る。
  経路A / D が `ToolContext` を素通ししている現行と対称になる。
- 解決できない場合（現在 span が無い直接呼び出し・捕捉テーブル未登録）は、警告もログも
  例外も出さずに `context=None` で実行する。
- 利用者が独自の agent 単位フックを併用する場合は `chain_agent_hooks(spec.hooks, 独自フック)`
  で合成する。`spec.hooks` を上書きすると捕捉は失われる。
- SDK 接触（`AgentHooks` サブクラス定義・`get_current_span`）は `_adapters` 配下の専用
  モジュールに閉じ、`workflow/` 層は context を plain に素通しするのみとする（SDK 隔離）。

build-don't-run の例外リストには該当しない。追加するのは「SDK が駆動する hook 内で
テーブルへ 1 回書く」「`get_response` でテーブルを 1 回読む」だけであり、lib が新たに
`Runner.run` を呼ぶ・独自の実行ループや再試行を持つことはない。

## Consequences

- \+ 経路C でも外側 context が内部ノードへ届き、`instructions_append` / hooks /
  ガードレールが経路A / D と同じ形で機能する。経路の選択軸が「context 透過の要否」から
  「tool 往復の要否」へ移り、最軽量・履歴クリーンな経路C を既定として選べる。
- \+ 公開 `__all__` と `as_agent_spec` / `as_facade_spec` / `connect_as_facade` の
  シグネチャは不変（後方互換・SemVer minor）。伝播は宣言の追加なしに働く。
- \+ run スコープのキーが span の identity であるため、並行 run・連続 run・再試行
  （同一 turn 内の `get_response` 再呼び出し）で分離と再解決の双方が成り立つ。
- \- 観測可能な挙動が 3 点変わる: `as_agent_spec` 戻り値の `spec.hooks` が `None` から
  lib 所有のフックインスタンスになる（`hooks is None` 比較の利用者コードが影響を受ける）／
  経路C の FUNCTION ノード・router・ノード前後フックの `ctx` が `None` から
  `RunContextWrapper`（実型は SDK のサブクラス）になる／経路C の AGENT ノード内側
  `Runner.run` に生の context オブジェクトが渡る。
- \- `spec.hooks` を上書きすると捕捉が失われ、伝播は警告なしに止まる（`context=None` へ
  フォールバックする）。併用は合成ヘルパーで行う必要がある。
- \- `on_start` を経ずに `get_response` が呼ばれる経路（`WorkflowModel.get_response` の
  直接呼び出し、および同一 agent の turn 2 以降で turn span が別オブジェクトになる場合）は
  `context=None` になる。`WorkflowModel` は tool / handoff を返さないため後者は経路C 単体
  では発生しない。
- \- 捕捉エントリは span の寿命の間だけ残る（span の寿命は run 終了後、tracing exporter の
  送出キューが処理を終えるまで。tracing 無効時の `NoOpSpan` は run 終了で即解放される）。
  その間は値側のラッパー（利用者 context・そこに載るトークン等）もメモリに残るが、キーが
  span の identity であるため他 run へは漏れない。
- \- 依拠する不変条件は SDK の tracing の意味論（同一 turn 内で現在 span が不変）であり、
  SDK 側がこれを変えると伝播は静かに `context=None` へ縮退する。縮退は例外を出さないため、
  下記の強制手段テストで検知する。

## Confirmation

強制手段:

- `tests/workflow/test_specs_l2.py::test_path_c_concurrent_runs_isolate_context`
  （並行 run で各 run の内部ノードに自 run の context のみが届くこと）
- `tests/workflow/test_specs_l2.py::test_path_c_hooks_override_falls_back_to_none_context`
  （`spec.hooks` 上書き時に例外・警告なしで `context=None` へ落ちること）
- `tests/workflow/test_specs_l2.py::test_workflow_model_get_response_keyword_input`
  / `::test_workflow_model_get_response_positional_input`
  （`get_response` の直接呼び出しが `context=None` で成立し続けること）

これらのテストは本 ADR を受理した設計に基づき実装フェーズで追加する（本 ADR 受理時点では未実装）。

`docs/QUALITY-GUARANTEES.md` に登録済み（source = ADR-0041）。
