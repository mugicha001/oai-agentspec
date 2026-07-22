# Resilience（Model retry と run 予算）

## 何を解決するか

長い会話・エージェント運用では、Model 呼び出しの一過性エラー（rate limit / timeout / 5xx）と run 全体の暴走（累積時間・トークン消費）の両方を制御する必要があります。`ModelRetryPolicy` は個別 Model 呼び出しの retry を宣言、`RunBudgetPolicy` は run 全体の累積予算を宣言し、超過時に `RunBudgetExceeded` を送出します。

セマンティックフラグ（`retry_on_*`）は既定 True で、SDK `ModelRetrySettings` の silent no-op（`policy` 未指定時 `max_retries` だけ指定しても retry しない）を排除します。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| `ModelRetryPolicy` 単独 | Model 呼び出しの retry のみ | 一過性エラーだけ吸収したい |
| `RunBudgetPolicy` 単独 | run 全体の累積時間 / トークン制御 | 暴走防止のみ必要 |
| 併用 | retry で吸収しつつ全体は budget で頭打ち | 本番想定・両方の安全網が要る |
| セマンティックフラグ off | SDK 素の retry 挙動へ戻す | 既存挙動と揃えたい |

## 使い方

- import: `from oai_agentspec.runtime.resilience import ModelRetryPolicy, RunBudgetPolicy, build_model_retry, build_run_budget_hooks`
- 例外の import: `from oai_agentspec.exceptions import RunBudgetExceeded`（lib 独自例外の統一窓口。extra 未導入でも import は壊れない）
- hooks 合成の import: `from oai_agentspec.runtime.hooks import chain_hooks`（budget hooks と自作 hooks の併用時）
- extras: `pip install oai-agentspec[resilience]`（追加外部依存なし）
- 依存 env: なし

```python
from oai_agentspec import AgentSpec
from oai_agentspec.runtime.resilience import (
    ModelRetryPolicy, RunBudgetPolicy,
    build_model_retry, build_run_budget_hooks,
)

retry_settings = build_model_retry(ModelRetryPolicy(max_retries=3))
budget_hooks = build_run_budget_hooks(
    RunBudgetPolicy(max_elapsed_seconds=60, max_total_tokens=100_000)
)

# AgentSpec.extra 経由で SDK ModelSettings.retry を素通し
spec = AgentSpec(
    name="assistant",
    instructions="...",
    extra={"model_settings": {"retry": retry_settings}},
)
# Runner.run(..., hooks=budget_hooks) は SDK 側で受け渡し
```

`Runner.run(hooks=...)` は単数の `RunHooksBase` しか受けないため、budget hooks と自作 hooks を併用する場合は `chain_hooks` で合成します。

```python
from oai_agentspec.runtime.hooks import chain_hooks

hooks = chain_hooks(budget_hooks, MyLoggingHooks())  # 宣言順に順次 await・前段 raise で後段スキップ
result = await Runner.run(agent, msg, hooks=hooks)
```

`RunBudgetExceeded` は `oai_agentspec.exceptions` から catch します。SDK `error_handlers` を素通しで伝播し、`on_llm_end` ターン境界で判定されます（詳細は architecture.md）。

```python
from oai_agentspec.exceptions import RunBudgetExceeded

try:
    result = await Runner.run(agent, msg, hooks=budget_hooks)
except RunBudgetExceeded as e:
    audit_log(usage=e.usage, elapsed=e.elapsed_seconds)
```

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `ModelRetryPolicy`（frozen）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `max_retries` | `int \| None` | `None` | 最大リトライ回数。None は SDK 既定、0 は retry 無効の明示 |
| `initial_delay_seconds` | `float \| None` | `None` | 初回リトライ前の待機秒 |
| `max_delay_seconds` | `float \| None` | `None` | リトライ待機秒上限 |
| `backoff_multiplier` | `float \| None` | `None` | 指数バックオフ倍率（1 未満は `ValueError`） |
| `backoff_jitter` | `bool \| None` | `None` | ジッタ付与 |
| `retry_on_network_error` | `bool` | `True` | ネットワークエラーで retry |
| `retry_on_timeout` | `bool` | `True` | タイムアウトで retry（`network_error` と同一プリミティブで独立無効化不可） |
| `retry_on_rate_limit` | `bool` | `True` | 429 で retry |
| `retry_on_server_error` | `bool` | `True` | 5xx で retry |
| `retry_on_retry_after` | `bool` | `True` | Retry-After ヘッダで retry |
| `extra_retry_statuses` | `tuple[int, ...]` | `()` | 追加 retry 対象ステータス |
| `policy` | `Any` | `None` | 生 SDK policy（指定時フラグを無視） |

### `RunBudgetPolicy`（frozen）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `max_elapsed_seconds` | `float \| None` | `None` | run 全体の累積経過秒上限（None で上限なし） |
| `max_total_tokens` | `int \| None` | `None` | run 全体の累積トークン上限（None で上限なし） |

### `RunBudgetExceeded`（例外）

`Exception(message, *, usage, elapsed_seconds, context)`。`context["exceeded"]` は `"max_total_tokens"` または `"max_elapsed_seconds"`。

### `build_model_retry(policy)` / `build_run_budget_hooks(policy)`

いずれも第 1 引数が対応する Policy インスタンス（1 個）で、SDK 生型（`ModelRetrySettings` / `RunErrorHandlers` 相当）を返す。

## 判断軸

- 一過性エラー吸収だけなら **`ModelRetryPolicy`** 単独。暴走防止だけなら **`RunBudgetPolicy`** 単独
- 本番運用では **併用**を既定に、streaming 経路のハード timeout は `asyncio.wait_for` で上位に別途噛ませる（budget は turn 境界判定のため）
- SDK 挙動を尊重したい場合のみ **セマンティックフラグ off**。既定 True 前提の設計を崩さないこと

## 落とし穴

- `RunBudgetPolicy` は `on_llm_end` 判定。stream 中の暴走には `asyncio.wait_for` を併用する
- `ModelRetrySettings` の silent no-op に頼らない。フラグは既定 True 前提で組む
- `max_retries > 0` かつ有効な retry 条件がゼロは build-time `ValueError`

## 参照

- 詳細設計: `docs/architecture.md`（Resilience 節）
- 設計判断: `docs/adr/0002-resilience-declarative-compilation.md` / `docs/adr/0003-hooks-chain-helper.md`（`chain_hooks`）
- 具体例: `examples/resilience/01_retry_and_budget.py`

## 次

[guardrails.md](./guardrails.md) — 入出力ガードレール 3 家族
