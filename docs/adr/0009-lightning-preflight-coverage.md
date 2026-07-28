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
- pre-flight 全体を専用の例外変換で包む: `OptimizeError` は raise-through、`ImportError`
  は `EXTRA_MISSING`、その他 `Exception` は `TRAINER_FAILED` へ変換し、メッセージで
  pre-flight フェーズであることを識別可能にする（Trainer 本体の失敗とコスト構造・
  救済策が異なるため）。
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
  `_CandidateInvalid` が送出された case は観測が空になる。空観測（`steps=()`）は union に
  対して恒等（no-op）であり、結果的に union へ寄与しない（特別な除外規則は持たない）。
- 集計後 `missing = set(slots.keys()) - covered` が非空なら
  `OptimizeError(FailureKind.CONFIG_MISSING, ..., coverage=CoverageReport(...))` を送出する。
  raise 前に `logger.warning` で集計行（`covered` / `missing` / `cases` / `interrupted`）を
  出力し、`train × 1 rollout` の観測情報を保全する。

新規公開型 `CoverageReport`（frozen dataclass・`covered: frozenset[str]` /
`missing: frozenset[str]` / `per_case: tuple[tuple[Any, tuple[str, ...]], ...]` /
`interrupted_cases: int`）を `oai_agentspec.runtime.lightning.__all__` に追加する
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

## Consequences

- + Silent no-op が fail-fast に置き換わり、noise 由来の textual gradient で最適化が
  破壊される経路が構造的に遮断される。診断情報（未到達 slot 列挙・救済策 3 段・
  `CoverageReport`）が `OptimizeError` と `logger.warning` の 2 段で保全される。
- + 公開シグネチャは非破壊拡張（`OptimizeConfig` / `optimize()` / `OptimizeError` に
  optional field / kwarg / attribute を追加するのみ）。既存の except 分岐は
  `exc.coverage is None` のまま影響を受けない。
- + Phase 2 への移行余地を残す（`expected_route` 静的検証で置き換わったら pre-flight
  ヘルパを retire する。台帳の retire 条件に明記）。
- - 既存 graph + slots 呼び出しで未到達 slot がある場合、silent no-op から fail-fast へ
  挙動が変わる（minor bump 相当・changelog で明示する）。
- - `train × 1 rollout` の LLM API コストが増える。既存の `run_apo` が
  `train × candidate × N rounds` 走るため相対増分は `1 / (candidate × rounds)` 程度で
  ある。動的 routing など pre-flight が判定に寄与しない graph 経路のユーザーは
  `skip_coverage_check=True` で回避できる。
- - 動的 routing 下で seed 状態と candidate 状態で経路が変わる構成は seed pre-flight のみ
  では検出しきれない。docstring / docs の limitation として明記し、Phase 2 の
  `expected_route` 静的検証に委ねる。
- - 却下: 並列 pre-flight（overcomplicated 回避）。将来 `OptimizeConfig.concurrency` に
  揃える余地は残す。
- - 却下: interrupted case の観測到達を union から破棄する（`covered` が過小になり
  偽陽性の fail を作るため。`interrupted_cases` は診断カウンタに限定する）。
- - 却下: `CoverageReport` の case 型を `OptimizeCase` に絞る（利用者任意型が失われる）。
  `Any` で `RolloutResult.case` と揃える。

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
  （`test_optimize_preflight_runtime_error_maps_to_trainer_failed`））。fail-closed 2 経路
  （candidate 適用 None・`_CandidateInvalid` で観測が空になり到達を誤申告しない）は
  `tests/runtime/lightning/test_rollout_l2.py` の `test_observe_route_steps_*`
  （`test_observe_route_steps_applied_none_returns_empty_no_reach_claim` /
  `test_observe_route_steps_candidate_invalid_returns_empty_no_reach_claim`）が担う。型と設定の契約は
  `tests/runtime/lightning/test_types_l1.py` の `test_coverage_report_*` /
  `test_optimize_error_*coverage*` と `tests/runtime/lightning/test_config_l1.py` の
  `test_optimize_config_skip_coverage_check_*` が担う。
  `docs/QUALITY-GUARANTEES.md` に登録済み。
- Phase 2 移行時（`expected_route` 静的検証への置き換え）に pre-flight ヘルパを retire
  し、本 ADR を新 ADR で `superseded by NNNN` に変更する。
