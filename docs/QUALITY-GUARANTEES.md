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
| examples が import-safe で smoke 通過する | prompt_slot_factory 導入時の safety net 要件 | tests/examples/test_examples_smoke.py | examples を publish から外した場合 |
