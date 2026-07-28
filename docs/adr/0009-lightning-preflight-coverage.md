# 0009: `optimize()` に seed 状態の pre-flight route coverage 検証を導入し、未到達 slot を fail-fast する

- Status: accepted
- Date: 2026-07-28

## Context

`optimize(target=HandoffGraph, slot=[...])` で複数エージェントの prompt を最適化する際、
`slot` に指定した agent が rollout の routing で 1 度も呼ばれない構成が silent no-op に
陥る問題があった。到達しない slot は reward が発火せず、textual gradient は route 未通過
場面の noise ベース観測でスコアが動き、最適化ループが「壊れていない」ように見えたまま
無意味な書き換えを繰り返す。silent 進行の危険性が高いため、fail-closed 側の初期挙動が
安全であり、opt-out は escape hatch として残す必要がある。

Phase 分割方針:

- **Phase 1（本 ADR）**: `run_apo` 委譲直前に seed 状態の pre-flight rollout を `train`
  全件について実行し、`ObservedRun.route.steps` の agent 名を集計する（pre-flight は
  `RolloutResult` を構築しない）。slot 集合と set 差分で
  突き合わせ、1 度も呼ばれていない slot があれば `OptimizeError(CONFIG_MISSING)` で
  fail-fast する。opt-out として `skip_coverage_check=True` を用意する。
- **Phase 2（別 ADR で扱う）**: `OptimizeCase.expected_route` を必須化し、pre-flight
  rollout に頼らず静的に coverage を検証する経路へ移行する。Phase 1 の pre-flight
  ヘルパは Phase 2 移行時に retire する。

既存 SDK / ライブラリで同等の pre-flight coverage を提供する機能はない
（SDK `agents` の handoff は動的 routing のため静的 API がなく、`agent-lightning`
側も per-slot 呼び出し数を報告しない）。ランタイム観測が唯一の検出手段である。

検討した却下案:

- **`route_match(field="expected_route")` reward による代替**: reward 側で 0.0 になるだけで
  未到達 slot の reward は発火しないため、silent no-op を解決できない。
- **candidate 状態を含む全候補で pre-flight**: 実装複雑度が `train × candidate × rounds`
  相当に膨らみ、Phase 1 スコープを逸脱する。動的 routing 下で seed と candidate で経路が
  変わるケースは docstring / docs の limitation として明記する。
- **並列 pre-flight**: `train` 件数は通常小規模（10-50 件想定）で逐次でも実測負担が
  小さく、並列化は overcomplicated 回避の観点から採用しない。将来は `OptimizeConfig.concurrency`
  に揃える余地を残す。
- **interrupted case の観測到達を union から破棄する**: coverage は「到達の union」であり、
  観測された到達は常に陽性証拠である。interrupted は「未完走で追加の到達が観測できなかった」
  ことを意味するだけで、観測済み到達を無効化する根拠にならない。破棄すると `covered` が
  過小になり、wireable な構成を fail 判定する偽陽性を作る方向にしか働かないため採用しない。
  なお空 route（`steps=[]`）の case は union に対して恒等（no-op）であり、特別な除外規則
  自体が不要である。
- **新規 module 分離（`_preflight.py` 等）**: 既存の observation 収集責務
  （`_apply_candidate` / `_run_one`）と同一系統のため `_rollout.py` に集約し、
  module 分離は行わない。
- **`OptimizeError` の kind を新設**: `CONFIG_MISSING` は既存 slot 名不一致・extra 未導入
  等の「構成が最適化を回すのに不足している」ケースと同カテゴリのため再利用する。
  診断情報（`covered` / `missing` / `per_case` / `interrupted_cases`）は新規公開型
  `CoverageReport` を `OptimizeError.coverage` 属性で受け渡す。

観測の時間上限について検討した却下案:

- **pre-flight 全体を 1 回の `asyncio.wait_for` で包む（総時間上限）**: どの case で詰まったかが
  判らず部分観測の診断価値が落ちる。1 case のハングと train 件数増による正当な長時間を
  区別できないため採用しない。
- **`OptimizeConfig.preflight_timeout_seconds` を新設**: 専用フィールドを要する実需要の実体が
  なく、公開 config を投機的に広げる。必要になった時点で非破壊追加できる。
- **上限未指定時に APO 既定に合わせて 3600 秒を適用する**: 既定値の暗黙導入は「未設定 = 無制限」
  という既存 `timeout_seconds` の意味と食い違い、既定構成の挙動を変えるため採用しない。
- **`RunBudgetPolicy.max_elapsed_seconds` + `build_run_budget_hooks` を流用する**: ターン境界の
  graceful 判定のみで、ツール実行中・単一の長時間 LLM 呼び出し中には止まらない（pre-flight の
  ハング源は主に後者）。加えて観測経路へ `hooks=` を配線する新規結線が必要になり、pre-flight の
  ためだけに実行経路の引数契約を広げることになる。`resilience` 側の docstring 自身が「ハード
  timeout が必要な場合は利用者側で `asyncio.wait_for` を被せる」と示しており、`wait_for` の
  採用と整合する。

## Decision

`optimize()` 冒頭の `_seeds_of` fail-fast 直後・`_make_rollout` 直後・`run_apo` 直前に
pre-flight coverage 検証を挿入する。既定は有効（`skip_coverage_check=False`）とし、
`OptimizeConfig.skip_coverage_check` および `optimize(skip_coverage_check=...)` の
kwarg で opt-out できる。

検証の骨子:

- `_normalize_slots(target, slot)` が `slots is None` を返す経路（生 seed + rebind）は
  pre-flight を skip する。slot 集合の SoT が API 上存在せず set 差分が意味を持たないため。
- `isinstance(target, AgentSpec)` の経路（既定スロット導出・`Slot` 単体・`[Slot]` 列
  いずれも）は pre-flight を skip する。起点 agent が必ず自身の slot をカバーするため
  trivial pass にしかならず、mock ベースの既存 rollout テストへ実 rollout を強制する
  副作用が上回るため。
- pre-flight の適用対象は `isinstance(target, HandoffGraph)` の allow-list とする。
  slot 名 = registry spec 名 = `route.steps` の agent 名が同一名前空間である経路のみが
  検証可能なため。将来の target 種追加時は allow-list により「skip 側（fail しない）」へ
  倒れる。
- `WorkflowGraph` は skip する。`_target.normalize` が workflow 全体を単一 agent
  （`"workflow"`）へ畳むため内部 agent の route が観測できず、rollout ベースの coverage
  検証が原理的に成立しない。Phase 2 の `expected_route` 宣言的検証に委ねる。
- pre-flight 実行前に extra（`agentlightning`）の可用性を検査し、未導入なら
  `OptimizeError(EXTRA_MISSING)` へ倒す（実 rollout コストを消費する前の fail-fast）。
- pre-flight の例外変換は 2 段に分ける。段 1 は extra 検査（`_require_agentlightning()`）
  のみを包み、`ImportError` を `EXTRA_MISSING` へ倒す。段 2 は観測本体を包み、
  `OptimizeError` は kind と診断情報（`coverage` 添付）を保つため raise-through、
  その他の `Exception` を `TRAINER_FAILED` へ変換する。段 2 では `ImportError` を
  特別扱いしない。rollout 中に利用者のツールが起こした import 失敗まで「extra 未導入」と
  誤診断させないためであり、発生源で分けられる情報を例外型だけで再構成しようとしない。
  段 2 のメッセージは pre-flight フェーズであることを識別可能にしたうえで
  `{type(exc).__name__}: {exc}` 形式にする（Trainer 本体の失敗とコスト構造・救済策が
  異なるため識別が要る。加えて `str(exc)` が空になる例外型でも型名から原因が読めるように
  するため）。
- seed 状態（現行 seed をそのまま候補として `_apply_candidate` で reify）で `train` 全件を
  逐次 `_run_one` で観測し、`observation.route.steps` のみ採取する。reward callable は
  経由せず、`_make_rollout` の返す rollout closure ではなく `_rollout` internals
  （`_apply_candidate` + `target_mod.normalize` + `_run_one`）を直接組む。
  `approvals` / `tool_mocks` / `context_factory` は本番 rollout と同値で素通しし、
  routing 挙動を同じ状態で観測する。
- `RunOutcome.interrupted=True`（`_run_one` の戻り値タプル第 1 要素）で途中打ち切りと
  なった case でも、観測できた `route.steps` は union に算入する（観測された到達は常に
  陽性証拠であり、破棄すると `covered` が過小になり偽陽性の fail を作る）。
  `interrupted_cases` は診断カウンタとしてのみ残し、判定には使わない。
  `_apply_candidate` が None を返した case（seed 状態で必要な `${var}` が失われた場合）と
  `_CandidateInvalid` が送出された case は**候補無効化**（rollout の観測が得られていない）で
  あり、「実行済みだが観測が空」（`steps=()`・route 構築の防御的経路）とは区別する。
  `_observe_route_steps` は無効化を `route_steps=None` の標識で返し、`per_case` にも `None`
  として記録する（3 値: `None` = 無効化 / `()` = 実行済み観測空 / 非空 = 到達観測）。
  いずれも union へは寄与しないが、無効化 case は `invalid_cases` カウンタ
  （`interrupted_cases` と同じく診断専用・判定に使わない）へ加算し、fail-fast メッセージを
  無効化の有無で 3 分岐させる: 無効化ゼロ = 未到達確定の全数主張 / 部分無効化 = 観測件数と
  無効化件数を分離し無効化原因を併記 / 全件無効化 = 「一度も routing されなかった」と
  主張せず無効化原因（`${var}` 喪失 / `vars=callable` の非 dict 戻り値 / 境界マーカー崩れ）
  のみ提示する。無効化を `()` へ畳むと「未到達確定」と誤診断され、効かない救済策
  （train 追加 / edge 見直し）へ利用者を誘導するため（実測で確認した欠陥の是正）。
  `missing` が確定であるのは `complete=True` かつ `invalid_cases == 0` のときに限る
  （`complete` は「観測ループを完走したか」に意味を限定する）。判定 pass でも無効化が
  あれば `logger.warning` で通知する（silent にしない・case 本文は出さない）。
- 集計後 `missing = set(slots.keys()) - covered` が非空なら
  `OptimizeError(FailureKind.CONFIG_MISSING, ..., coverage=CoverageReport(...))` を送出する。
  raise 前に `logger.warning` で集計行（`covered` / `missing` / `cases` / `interrupted`）を
  出力し、`train × 1 rollout` の観測情報を保全する。
- 観測が途中の case で失敗した場合も、そこまでの到達集計を保全する。`logger.warning` に
  集計行（何件目 / `covered` / 例外型名）を残したうえで、`complete=False` の部分
  `CoverageReport` を添付した `OptimizeError(FailureKind.TRAINER_FAILED)` へ変換する
  （原例外は `__cause__`）。ログは設定次第で失われ、かつ文字列であって except 節から
  プログラム的に取得できないため、ログのみの保全では「エラーで観測データがなくなる」
  という本 ADR の動機を満たさない。`missing` の意味が「未到達の確定」と「未到達 + 未観測」で
  二義になる問題は、`coverage` を諦めるのではなく `complete` フラグで意味論を分離して解く。
  ログには case 本文を出さず、`covered` の sorted リスト・件数・例外型名に限る
  （`per_case` の repr を抑止しているのと同じ PII 方針）。
- 観測ループ内で既に `OptimizeError` が送出された場合（承認安全違反の `CONFIG_MISSING`・
  NFR-8 fail-closed）は再ラップせずそのまま通す。無条件に `TRAINER_FAILED` へ包むと kind が
  変質し、fail-closed の診断が失われる。この経路の `coverage` は `None` のままであり、
  部分観測の構造化保全は本 ADR のスコープ外とする（kind と message の保存を優先する）。
- 失敗メッセージには case 位置（`case i/N`）・1 case 観測上限の適用状況・例外型名を含める。
  型名は本文が空でも常に出し、本文は非空のときだけ連結する（`str(TimeoutError())` は空で、
  無条件連結だと `'... TimeoutError: '` とコロンで終わり情報がゼロになる）。
- pre-flight の 1 case 観測は `OptimizeConfig.timeout_seconds` を上限として `asyncio.wait_for`
  で包む。`None`（既定）は上限なしとし、`TimeoutError` は `TRAINER_FAILED` へ倒す
  （ハード fail）。上限を pre-flight 全体でなく 1 case 単位に置くのは、どの case で詰まったかを
  部分観測ログから特定可能に保つため、および「1 case のハング」と「train 件数増による正当な
  長時間」を区別可能に保つためである。専用フィールドを新設せず既存の `timeout_seconds` を
  再利用するのは、APO 側の `rollout_batch_timeout`（1 まとまりの rollout 待ち合わせ上限）と
  意味論が同系であり、公開面を増やさずに表現できるためである。適用規則の詳細は
  `docs/usage/ops/lightning.md` を参照する。

新規公開型 `CoverageReport`（frozen dataclass・`covered: frozenset[str]` /
`missing: frozenset[str]` / `per_case: tuple[tuple[Any, tuple[str, ...]], ...]` /
`interrupted_cases: int` / `complete: bool = True`）を
`oai_agentspec.runtime.lightning.__all__` に追加する
（コア `oai_agentspec.__all__` には載せない。lightning の公開シンボルは extra 窓口経由で
取得する既存契約に揃える）。`per_case` の case 要素は
`RolloutResult.case: Any` と型を揃え、`train` が受理する `OptimizeCase` / dict / 利用者
定義任意型の多態性を保持する。`per_case` は `field(repr=False)` とし、`CoverageReport` の
`repr()` に raw case を展開しない（case 本文に含まれうる PII の accidental dump 防止）。
`report.per_case` への明示アクセスは不変。`OptimizeError.__init__` に optional keyword-only
`coverage: CoverageReport | None = None` を追加し、pre-flight 経路のみ非 None を注入する
（既存の他 `OptimizeError` 送出経路は `coverage=None` のまま。既存呼び出しには影響なし・
非破壊拡張）。

pre-flight 検証本体は `runtime/lightning/_rollout.py` 内 private
（`_check_route_coverage` / `_observe_route_steps` 等）として実装する。既存の
`_apply_candidate` / `_run_one` と同一 module に集約し、新規 module 分離しない。

rollout closure と `_observe_route_steps` に共通する実行手順（候補適用 → 正規化 →
`_run_one`）は `_execute_once` に集約し、candidate 無効時の degrade（reward `0.0` /
空観測）は各呼び出し側に残す。pre-flight と本番 rollout の観測条件が silent に乖離し、
coverage 判定が本番 routing を代表しなくなる経路を構造的に塞ぐためであり、degrade の
意味は経路ごとに異なるため共通化の対象に含めない。

## Consequences

- + Silent no-op が fail-fast に置き換わり、noise 由来の textual gradient で最適化が
  破壊される経路が構造的に遮断される。診断情報（未到達 slot 列挙・救済策 3 段・
  `CoverageReport`）が `OptimizeError` と `logger.warning` の 2 段で保全される。
- + 公開シグネチャは非破壊拡張（`OptimizeConfig` / `optimize()` / `OptimizeError` に
  optional field / kwarg / attribute を追加するのみ）。既存の except 分岐は
  `exc.coverage is None` のまま影響を受けない。
- + `timeout_seconds` を明示設定した利用者は 1 case あたりの無限ハングから保護され、
  承認 resume ループ（最大 6 rollout）も 1 case 上限の内側に入る。
- + 観測が途中で失敗しても、そこまでの到達集計が `logger.warning` に残るため、
  再実行前に「どの case で詰まったか」を診断できる。
- + Phase 2 への移行余地を残す（`expected_route` 静的検証で置き換わったら pre-flight の
  観測経路を retire する。台帳の retire 条件に明記）。retire の対象は
  `_check_route_coverage` / `_observe_route_steps` と pre-flight rollout コストに限る。
  `_execute_once` は本番 rollout closure と共有する実行経路であり retire 対象に含めない。
  `CoverageReport` / `skip_coverage_check` は公開契約であり retire しない（Phase 2 で
  これらを再利用するかは Phase 2 の ADR で決める）。
- - 既存 graph + slots 呼び出しで未到達 slot がある場合、silent no-op から fail-fast へ
  挙動が変わる（minor bump 相当・changelog で明示する）。
- - `train × 1 rollout` の LLM API コストが増える。既存の `run_apo` が
  `train × candidate × N rounds` 走るため相対増分は `1 / (candidate × rounds)` 程度で
  ある。動的 routing など pre-flight が判定に寄与しない graph 経路のユーザーは
  `skip_coverage_check=True` で回避できる。
- - 動的 routing 下で seed 状態と candidate 状態で経路が変わる構成は seed pre-flight のみ
  では検出しきれない。docstring / docs の limitation として明記し、Phase 2 の
  `expected_route` 静的検証に委ねる。
- - 上限は 1 case 単位のため、pre-flight 全体の壁時計は `timeout_seconds × len(train)` まで
  伸びる（全体上限ではない）。
- - 既定（`timeout_seconds` 未指定）では pre-flight に時間上限がかからない。上限保護は
  明示設定した利用者のみが受ける。
- - APO の `rollout_batch_timeout` が deadline 超過後も部分結果で続行するソフト degrade
  であるのに対し、pre-flight はハード fail であり非対称になる。これは「観測失敗を空観測へ
  degrade させると `covered` が過小になり偽陽性 fail を生む」という本 ADR の既存方針と
  整合させた意図的な選択である（タイムアウトも観測失敗の一種として扱う）。
- - 超過時のキャンセルは実行中の rollout を `await` 点で中断するため、既に開始した非モック
  ツール（`tool_mocks` 未差し替えかつ承認ゲートを持たないもの）の副作用は部分適用のまま
  残りえる（補償処理は行わない）。既存の「副作用の追加発火」limitation と同じ境界で、
  安全化は `tool_mocks` / `skip_coverage_check` に委ねる。
- - 却下: 並列 pre-flight（overcomplicated 回避）。将来 `OptimizeConfig.concurrency` に
  揃える余地は残す。
- - 却下: interrupted case の観測到達を union から破棄する（`covered` が過小になり
  偽陽性の fail を作るため。`interrupted_cases` は診断カウンタに限定する）。
- - 却下: `CoverageReport` の case 型を `OptimizeCase` に絞る（利用者任意型が失われる）。
  `Any` で `RolloutResult.case` と揃える。
- + 観測失敗経路でも部分 `CoverageReport`（`complete=False`）が `OptimizeError.coverage` から
  取得できる。支払い済み API コストで得た到達観測が except 節からプログラム的に扱える。
- - `missing` の意味が `complete` に依存する（`True` = 未到達の確定 / `False` = 未到達と
  未観測の和）。フラグを読まずに `missing` を対処判断へ使うと誤った是正（train の作り直し等）へ
  誘導しうる。型 docstring・usage docs・失敗メッセージ本文の 3 箇所で明示して緩和する。
- - pre-flight 標識文字列が `_rollout._check_route_coverage`（ループ内）と
  `optimizer`（ループ外の安全網）の 2 箇所に重複する。診断の入口が 2 つある事実を隠さない方を
  優先した。後者は `test_optimize_preflight_outer_safety_net_maps_to_trainer_failed` が pin する。

候補無効化の診断分類について検討した却下案:

- **`per_case` の `()` 据え置き（カウンタのみ追加）**: per_case から無効化 case を特定できず、
  「実行済み観測空」との区別が report 単体で読めない診断の穴が残る。`None` の型レベル排他
  表現を採る（`CoverageReport` は未リリースであり表現の確定に互換コストがない）。
- **無効化があるとき `complete=False` にする**: 「途中打ち切り（観測失敗）」との区別を失い、
  `CONFIG_MISSING` / `TRAINER_FAILED` と `complete` の対応も崩れる。無効化 case はループを
  止めないため「完走した」は事実として真。`complete` は「ループ完走」に意味を限定し、
  確定性は `complete and invalid_cases == 0` の組で表現する。
- **全件無効化に新しい `FailureKind` を割り当てる**: 全件無効化は「seed / vars / build 構成の
  不備で最適化を回せない」であり `CONFIG_MISSING` の既存カテゴリに合致。kind の増設は
  利用者の except 分岐を複雑化するだけで、判別は `coverage.invalid_cases` で足りる。
- **`_CandidateInvalid` の理由文字列を共有経路から運搬する（opt-in パラメータ）**: 系統
  (a)（`_reinject_vars` の None）には例外も理由文字列も構造的に存在せず完全性が得られない。
  共有 `_execute_once` の契約変更は本番 rollout の挙動不変制約に反する。メッセージの
  系統別原因リスト（3 項目・真因を必ず含む）で代替する。

観測失敗時の診断保全について検討した却下案:

- **`kind` で判別させる（`CONFIG_MISSING`=完走 / `TRAINER_FAILED`=部分）**: 公開型を広げずに
  済むが、「失敗種別」の軸に「観測完了度」という別関心を暗黙に相乗りさせる符号化で、利用者に
  対応表の記憶を強いる。加えて `CoverageReport` を例外から切り離してログ・保存した時点で
  完了度が失われる（`repr` にも出ない）。`complete` は未リリースの型への既定値付き末尾追加で
  コストが実質ゼロのため、自己記述的な表現を採る。
- **`observed_cases` / `total_cases` を持たせる**: `observed_cases` は `len(per_case)` と重複し、
  `total_cases` は呼び出し側が `len(train)` を保持している。公開型を 2 フィールド広げる価値がない。
- **フィールドを足さず docstring のみで注意喚起する**: 誤対処の防止が docstring 依存になる。
- **内部 sentinel 例外（`_PreflightObservationFailed`）を新設して optimizer で翻訳する**:
  raise 1 箇所・翻訳 1 箇所で分岐がなく純粋な間接化。メッセージ組み立てに要する `index` /
  `timeout_seconds` は `_check_route_coverage` にしか揃わないため、中継型は情報を運ぶだけになる。
- **`_check_route_coverage` の戻り値 / callback で report を渡す**: `-> None` の契約を変え、
  fail-fast を例外と戻り値検査へ二重化する。既存の spy helper の契約も壊れる。

なお本 ADR は `accepted` だが未マージであり、本 ADR を導入した PR 内での自己訂正にあたる。
規約（`.claude/rules/05-docs.instructions.md` §5 の append-only）に照らし、マージ済みなら
ADR 0010（supersedes 0009）を起こすべきところだが、同一 PR 内の未リリース仕様の是正のため
0009 を直接訂正する。

## Confirmation

- 強制手段: `tests/runtime/lightning/test_optimizer_l2.py` の `test_optimize_preflight_*`
  系（未到達 slot で `OptimizeError(CONFIG_MISSING, coverage=CoverageReport(...))` +
  メッセージが「原因 / 未到達 slot 名 / 救済策 3 段」を含む / 全 slot 到達で `run_apo` へ通す /
  `config=` 経路と kwarg 経路それぞれで `skip_coverage_check=True` opt-out・opt-out 時に
  pre-flight rollout コストを消費しない / `slots is None` 経路で skip / `AgentSpec` /
  `WorkflowGraph` target で skip（allow-list: `HandoffGraph` のみ）/ interrupted case の
  観測到達が union に算入され陽性証拠として働く / `approvals` / `tool_mocks` / `context_factory` の
  素通し / `logger.warning` の集計行出力 / `per_case` が観測空 case を空タプルで記録 /
  extra 不在時に rollout を 1 件も消費する前に `EXTRA_MISSING` で fail-fast する
  （`test_optimize_preflight_extra_missing_before_rollout`）/ pre-flight 観測中の実行時例外が
  pre-flight 標識付きの `TRAINER_FAILED` へ変換される
  （`test_optimize_preflight_runtime_error_maps_to_trainer_failed`）/ 観測中の `ImportError` が
  `EXTRA_MISSING` ではなく `TRAINER_FAILED` へ倒れる
  （`test_optimize_preflight_observation_import_error_maps_to_trainer_failed`）/ 観測失敗時に
  何件目・`covered`・例外型名の warning が出力され case 本文を含まない
  （`test_optimize_preflight_partial_observation_logged_on_failure`）/ `timeout_seconds` 超過が
  `TRAINER_FAILED` へ倒れる（`test_optimize_preflight_timeout_maps_to_trainer_failed`）/
  `timeout_seconds=None` では上限が適用されず完走する
  （`test_optimize_preflight_timeout_none_does_not_limit`）/ 観測失敗時の部分
  `CoverageReport` が呼び出し側まで届き case 本文を漏らさない
  （`test_optimize_preflight_partial_coverage_reaches_caller`）/ ループ外例外が安全網で
  `TRAINER_FAILED` へ倒れ `coverage=None` になる
  （`test_optimize_preflight_outer_safety_net_maps_to_trainer_failed`）/ rollout 実行時の
  `_CandidateInvalid` 全件無効化が「未到達確定」と診断されず invalid_cases / per_case=None /
  専用メッセージが届く
  （`test_optimize_preflight_all_invalid_reports_invalidation_not_unreached`））。
  候補無効化の診断分類（`_observe_route_steps` の `None` 標識 / 全件無効化で未到達を主張しない /
  部分無効化の件数分離 / 実行済み観測空 `()` との区別 / pass 時の warning / 部分 report への
  `invalid_cases` 反映と観測完了数の非 None 計上）は
  `tests/runtime/lightning/test_rollout_l2.py` の `test_observe_route_steps_*_invalid_signal` /
  `test_check_route_coverage_all_invalid_*` / `..._partial_invalid_*` / `..._pass_with_invalid_*` /
  `..._executed_empty_*` / `..._counts_invalid_*` が pin する。
  観測失敗時の診断保全そのもの（部分 report 添付 / `complete=False` / メッセージの case 位置・
  上限秒・型名 / 空本文で末尾コロンにしない / 観測中 `OptimizeError` の pass-through /
  PII 非露出）は `tests/runtime/lightning/test_rollout_l2.py` の
  `test_check_route_coverage_*` 6 本が直接 pin する。fail-closed 2 経路
  （candidate 適用 None・`_CandidateInvalid` で観測が空になり到達を誤申告しない）は
  `tests/runtime/lightning/test_rollout_l2.py` の `test_observe_route_steps_*`
  （`test_observe_route_steps_applied_none_returns_empty_no_reach_claim` /
  `test_observe_route_steps_candidate_invalid_returns_empty_no_reach_claim`）が担う。型と設定の契約は
  `tests/runtime/lightning/test_types_l1.py` の `test_coverage_report_*` /
  `test_optimize_error_*coverage*` と `tests/runtime/lightning/test_config_l1.py` の
  `test_optimize_config_skip_coverage_check_*` が担う。
  `docs/QUALITY-GUARANTEES.md` に登録済み。
- Phase 2 移行時（`expected_route` 静的検証への置き換え）に pre-flight の観測経路
  （`_check_route_coverage` / `_observe_route_steps`）を retire し、本 ADR を新 ADR で
  `superseded by NNNN` に変更する。`_execute_once` は本番 rollout closure と共有するため
  retire 対象外。`CoverageReport` / `skip_coverage_check` の公開契約もこの retire の
  対象に含めない。
