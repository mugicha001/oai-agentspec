# 品質保証台帳

「完了済みの状態を継続的に再検証したい」不変条件の意図・台帳。機械検証できる保証は
テストスイート（実体が強制手段・CI が毎回再検証）で表現し、その意図をここに 1 行登録する。

各行は次の列を持つ:

- **保証したい状態**: 何が満たされ続けるべきか
- **source**: 契機（要件等の安定アンカー。Issue 番号は書かない）
- **強制手段**: 再検証を担うテスト等のポインタ
- **retire 条件**: この保証を台帳から外してよくなる条件

| 保証したい状態 | source | 強制手段 | retire 条件 |
|---|---|---|---|
| extra 未導入環境でも `import oai_agentspec.exceptions` が壊れず、`agents` / 各 extra 依存および `runtime.lightning` / `runtime.cli` を `sys.modules` に載せない（PEP 562 遅延の実効性） | 独自例外の統一窓口（`oai_agentspec.exceptions`） | `tests/test_extra_isolation.py` の subprocess 隔離テスト | 例外統一窓口の遅延取得方式を廃止し全例外を直 import に統一した場合 |
| `_ChainedHooks` は SDK `RunHooksBase` の全 on_* メソッドをオーバーライドする（SDK 追随漏れによる silent gap 防止） | hooks 合成ヘルパー（`oai_agentspec.runtime.hooks.chain_hooks`） | `tests/runtime/hooks/test_chain_hooks_l2.py::test_chained_hooks_overrides_match_sdk_on_methods` | `chain_hooks` を廃止し利用者責務に戻した場合 |
| `prompt_slot(agent=None, layout=None, tune=<str>)` の旧 shape 呼び出しは `OptimizeError(CONFIG_MISSING)` で fail-closed される（新 shape 一本化・誤って旧経路が復活しないための regression guard） | prompt_slot の compose 一致化と旧経路削除（ADR 0007） | `tests/runtime/lightning/test_slots_l1.py::test_prompt_slot_rejects_legacy_shape` | ADR 0007 の either-or 契約（agent= か layout= 必須）が新 ADR で置き換えられた場合 |
| `examples/lightning/03_graph_apo.py` が import-safe で `build_registry` が構造的健全性を満たす | prompt_slot_factory 導入時の safety net 要件 | tests/examples/test_examples_smoke.py | 当該 example を publish から外した場合 |
| `optimize(target=HandoffGraph, slot=[...])` で graph routing 未到達 slot が pre-flight で `OptimizeError(CONFIG_MISSING, coverage=CoverageReport(...))` として fail-fast する（既定有効・`skip_coverage_check=True` で opt-out・silent no-op 防止） | graph routing 未到達 slot の silent no-op 防止（ADR 0009） | `tests/runtime/lightning/test_optimizer_l2.py::test_optimize_preflight_*`（未到達検知 / 全到達 pass / config・kwarg 両経路の opt-out と opt-out 時の pre-flight 未実行（rollout コスト回避）/ 生 seed 経路 skip / AgentSpec・WorkflowGraph target skip（allow-list: HandoffGraph のみ）/ interrupted 到達の union 算入・観測空 case の非寄与 / approvals・tool_mocks・context_factory の素通し / `logger.warning` 出力 / `per_case` 記録 / extra 不在は rollout 消費前に EXTRA_MISSING・観測中の実行時例外は pre-flight 標識付き TRAINER_FAILED）、`tests/runtime/lightning/test_rollout_l2.py::test_observe_route_steps_*`（fail-closed 2 経路: candidate 適用 None・`_CandidateInvalid` で空観測を返し到達を誤申告しない）、`tests/runtime/lightning/test_types_l1.py::test_coverage_report_*`（`per_case` の repr 抑止を含む）・`::test_optimize_error_*coverage*`、`tests/runtime/lightning/test_config_l1.py::test_optimize_config_skip_coverage_check_*` | Phase 2 の `expected_route` 必須化で pre-flight rollout を retire した場合 |
