# Resilience（Model retry / run 予算 / Failsafe）

## 何を解決するか

長い会話・エージェント運用では、Model 呼び出しの一過性エラー（rate limit / timeout / 5xx）・run 全体の暴走（累積時間・トークン消費）・Runner の外へ漏れた例外の着地、という 3 種の失敗を制御する必要があります。`ModelRetryPolicy` は個別 Model 呼び出しの retry を宣言、`RunBudgetPolicy` は run 全体の累積予算を宣言し超過時に `RunBudgetExceeded` を送出、`FailsafePolicy` + `failsafe_call` は Guardrail Tripwire・`RunBudgetExceeded`・`ToolTimeoutError` 等 Runner の外側まで伝播した例外を、呼び出し箇所ごとの try/except でなく宣言 1 回で着地値へ丸めます。

セマンティックフラグ（`retry_on_*`）は既定 True で、SDK `ModelRetrySettings` の silent no-op（`policy` 未指定時 `max_retries` だけ指定しても retry しない）を排除します。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| `ModelRetryPolicy` 単独 | Model 呼び出しの retry のみ（SDK ネイティブ retry） | 一過性エラーだけ吸収したい |
| `RunBudgetPolicy` 単独 | run 全体の累積時間 / トークン制御（上限） | 暴走防止のみ必要 |
| 併用（retry + budget） | retry で吸収しつつ全体は budget で頭打ち | 本番想定・両方の安全網が要る |
| `FailsafePolicy` + `failsafe_call` | Runner の外へ漏れた例外の着地 | try/except を分散させず宣言的に着地値へ丸めたい |
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

エージェント単位（`agents.AgentHooks`）の合成は `chain_agent_hooks` を使います（[agents.md](../core/agents.md)）。

`RunBudgetExceeded` は `oai_agentspec.exceptions` から catch します。SDK `error_handlers` を素通しで伝播し、`on_llm_end` ターン境界で判定されます（詳細は architecture.md）。

```python
from oai_agentspec.exceptions import RunBudgetExceeded

try:
    result = await Runner.run(agent, msg, hooks=budget_hooks)
except RunBudgetExceeded as e:
    audit_log(usage=e.usage, elapsed=e.elapsed_seconds)
```

### Failsafe（`FailsafePolicy` / `FailsafeHandler` / `failsafe_call` / `FailsafeResult` / `RUNNING_AGENT`）

- import: `from oai_agentspec.runtime.resilience import FailsafeHandler, FailsafePolicy, FailsafeResult, RUNNING_AGENT, failsafe_call`
- `FailsafePolicy` はアプリ全体で 1 回だけ宣言し、`Runner.run(...)` を呼ぶ各箇所を `failsafe_call(policy, lambda: Runner.run(...))` で包みます。

```python
from oai_agentspec.exceptions import RunBudgetExceeded
from oai_agentspec.runtime.resilience import FailsafePolicy, FailsafeResult, failsafe_call

policy = FailsafePolicy(
    handlers={
        RunBudgetExceeded: lambda exc: "混雑のため回答を中断しました。",
        ValueError: "入力を処理できませんでした。",
    },
    log_on_apply=True,
    on_apply=lambda result: audit_log(matched_type=result.matched_type.__name__),
)

result = await failsafe_call(policy, lambda: Runner.run(agent, msg))
if isinstance(result, FailsafeResult):
    # 着地: result.final_output / result.exception / result.matched_type
    ...
else:
    # 正常完了: Runner.run の戻り値（RunResult）がそのまま返る
    ...
```

正常完了時は `thunk` の戻り値（`RunResult`）がそのまま返り、着地時のみ `FailsafeResult` が返ります。いずれも `.final_output` で一様にアクセスできますが（structural 互換）、共通基底クラスは持たないため判別は `isinstance(result, FailsafeResult)` で行います。

#### `last_agent`（着地時に実行中だったエージェントを参照する）

決定は 2 段です。`FailsafeHandler.last_agent`（例外ごとの指定）→ `FailsafePolicy.fallback_last_agent`（全体規定）の順に見て、どちらも無指定なら `None` です。どちらの段にも具体の agent（`AgentRegistry` から取得した `Agent` をそのまま渡せます）か `RUNNING_AGENT`（「実際に動いていた Agent を使う」ことを表す sentinel）を置けます。`RUNNING_AGENT` を置いた段でのみ例外から解決を試み（`exc.run_data.last_agent` -> `exc.last_agent`）、解決できなければ次の段へ落ちます。指定しない限り `last_agent` は常に `None` です（自動導出はしません）。

```python
from oai_agentspec.exceptions import RunBudgetExceeded
from oai_agentspec.runtime.resilience import (
    RUNNING_AGENT, FailsafeHandler, FailsafePolicy, FailsafeResult, failsafe_call,
)

policy = FailsafePolicy(
    handlers={
        # 段 1: この例外だけ、実際に実行していたエージェントを使う
        RunBudgetExceeded: FailsafeHandler(fallback=_budget_message, last_agent=RUNNING_AGENT),
        ValueError: "入力を処理できませんでした。",
    },
    # 段 2: 段 1 が無指定 / 解決不能なら registry の既定エージェントへ落とす
    fallback_last_agent=registry.get("triage"),
)

result = await failsafe_call(policy, lambda: Runner.run(agent, msg))
if isinstance(result, FailsafeResult) and result.last_agent is not None:
    # 継続実行: result.last_agent を次の Runner.run(...) の起点に使う
    ...
```

`failsafe_call` の外側で自前の except を書く場合は、`FailsafeResult.from_exception` で同じ結果型・同じ `last_agent` の意味へ手動着地できます。`policy` を経由しないため段 2 は持たず、`last_agent` は明示指定または `RUNNING_AGENT` の指定（段 1 相当）のみです。監査（`log_on_apply` の warning・`on_apply`）は発火しません。

```python
try:
    return await failsafe_call(policy, lambda: Runner.run(agent, msg))
except TimeoutError as exc:
    # 組み込み例外は run_data も last_agent 属性も持たないため、RUNNING_AGENT を
    # 指定しても解決できない（last_agent は None になる）。継続先を決めたい場合は
    # 具体の agent を渡す。
    return FailsafeResult.from_exception(
        exc, final_output="時間内に応答できませんでした。", last_agent=triage_agent,
    )
except MaxTurnsExceeded as exc:
    # SDK 例外は run_data.last_agent を持つため、RUNNING_AGENT で解決できる。
    return FailsafeResult.from_exception(
        exc, final_output="対話が長くなりすぎました。", last_agent=RUNNING_AGENT,
    )
```

#### 会話履歴の継続

lib に `to_input_list()` 相当のヘルパは提供しません。第一選択は SDK `Session`（会話全体をセッションに委ねる。`README.md` の会話 Helper 節を参照）です。`Session` を使わず着地直後の履歴だけを再構成したい場合は、`run_data` を持つ SDK 例外（`RunBudgetExceeded` は `run_data` を持たないため対象外）に限り、SDK の `ItemHelpers.input_to_new_input_list` / `item.to_input_item()` を利用者コード側で組み合わせます。

継続例には**安全制御に由来しない例外**（`MaxTurnsExceeded` 等）を選びます。ガードレール Tripwire を継続に使ってはいけない理由は下記「ガードレール Tripwire の着地」を参照してください。

```python
from agents import ItemHelpers, MaxTurnsExceeded

try:
    result = await failsafe_call(policy, lambda: Runner.run(agent, msg))
except MaxTurnsExceeded as exc:
    landed = FailsafeResult.from_exception(exc, final_output="対話が長くなりすぎました。",
                                           last_agent=RUNNING_AGENT)
    if exc.run_data is not None:
        history = ItemHelpers.input_to_new_input_list(exc.run_data.input)
        history += [item.to_input_item() for item in exc.run_data.new_items]
        # history + landed.last_agent で次の Runner.run(...) を再開する
        # 継続回数はアプリ側で上限を持つ（lib は再試行を提供しない）
```

#### ガードレール Tripwire の着地

入力ガードレール Tripwire（`InputGuardrailTripwireTriggered`）を着地させる場合、**`exc.run_data.input` をそのまま再投入しないでください**。これはガードレールが拒否した入力そのものです。再投入すると次の 2 つが起きます。

- 同一 agent へ再投入すると再び trip し、着地 -> 再投入のループになる（トークン・課金を消費し続ける）
- `last_agent` が段 2（`fallback_last_agent`）の別 agent へ落ちている場合、**そのガードレールを持たない agent へ拒否済み入力が到達し、安全制御を素通りする**

Tripwire を着地させるときは次の 3 点を守ってください。

- 拒否された入力を継続に使わない（継続しない、または拒否された item を除いた履歴だけを使う）
- 継続する設計にするなら、アプリ側で試行回数の上限を持つ（lib は再試行機構を提供しません）
- 着地を `matched_type` で分岐し、通常応答と区別して監査する（`log_on_apply` の warning か `on_apply` で必ず痕跡を残す）

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

`Exception(message, *, usage, elapsed_seconds, context, last_agent=None)`。`context["exceeded"]` は `"max_total_tokens"` または `"max_elapsed_seconds"`。`last_agent` は超過時点で実行中だった agent（不透明値）で、Failsafe で `RUNNING_AGENT` を指定した際の解決元になる。

### `build_model_retry(policy)` / `build_run_budget_hooks(policy)`

いずれも第 1 引数が対応する Policy インスタンス（1 個）で、SDK 生型（`ModelRetrySettings` / `RunErrorHandlers` 相当）を返す。

### `FailsafePolicy`（frozen）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `handlers` | `Mapping[type[Exception], Any]` | `{}` | 例外型 -> 着地値 / `Callable[[Exception], Any]`（sync/async 可）/ `FailsafeHandler` のマッピング。宣言順に first-match。build 時に `dict` へ正規化・検証したうえで `MappingProxyType` へ差し替えられ不変化する（事後注入・元 dict 変更は反映されない） |
| `log_on_apply` | `bool` | `True` | 着地時に `logger.warning(..., exc_info=True)` を出すか |
| `on_apply` | `Callable[[FailsafeResult], Any] \| None` | `None` | 着地時に呼ばれるコールバック（sync/async 可・戻り値は無視）。`None` でも callable でもなければ build-time `ValueError` |
| `fallback_last_agent` | `Any` | `None` | `last_agent` 決定モデルの段 2（全体規定）。具体の agent または `RUNNING_AGENT`。repr には出ない |

### `FailsafeHandler`（frozen）

`handlers` の値位置に置ける opt-in の宣言。`fallback` に `FailsafeHandler`（ネスト）または `RUNNING_AGENT` を渡すと build-time `ValueError`。

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `fallback` | `Any` | 必須 | 着地値そのもの、または例外を受け取り着地値を返す callable（sync/async 可） |
| `last_agent` | `Any` | `None` | `last_agent` 決定モデルの段 1（例外ごとの指定）。具体の agent または `RUNNING_AGENT`。repr には出ない |

### `RUNNING_AGENT`（sentinel）

「実際に動いていた Agent を使う」ことを表す指定値。`FailsafeHandler.last_agent` / `FailsafePolicy.fallback_last_agent` / `FailsafeResult.from_exception(last_agent=...)` に置ける。着地値位置（`FailsafeHandler.fallback` / `handlers` の値位置）に置くと build-time `ValueError`。

### `FailsafeResult`（frozen）

| パラメータ | 型 | 説明 |
|---|---|---|
| `final_output` | `Any` | 着地値（handlers の値、または fallback callable の戻り値） |
| `exception` | `Exception` | 捕捉した例外インスタンス |
| `matched_type` | `type[Exception]` | first-match した handlers のキー（`from_exception` では明示指定 or 送出型） |
| `last_agent` | `Any` | 決定モデルで確定した実行中エージェント（不透明値）。既定 `None`。repr には出ない |

### `failsafe_call(policy, thunk)`

`failsafe_call(policy: FailsafePolicy, thunk: Callable[[], Awaitable[T]]) -> T | FailsafeResult`。正常完了時は `thunk` の戻り値そのもの、着地時は `FailsafeResult` を返す。

### `FailsafeResult.from_exception(exception, *, final_output, matched_type=None, last_agent=None)`

`failsafe_call` の外側で捕捉した例外から手動で `FailsafeResult` を構築する classmethod。`matched_type` 既定は `type(exception)`。`last_agent` は決定モデルの段 1 相当のみ（policy を受け取らないため段 2 は無い）。監査（warning / `on_apply`）は発火しない。

## 判断軸

- 一過性エラー吸収だけなら **`ModelRetryPolicy`** 単独。暴走防止だけなら **`RunBudgetPolicy`** 単独
- 本番運用では **併用**を既定に、streaming 経路のハード timeout は `asyncio.wait_for` で上位に別途噛ませる（budget は turn 境界判定のため）
- SDK 挙動を尊重したい場合のみ **セマンティックフラグ off**。既定 True 前提の設計を崩さないこと
- Runner の外へ漏れた例外を宣言的に着地させたいなら **`FailsafePolicy` + `failsafe_call`**。streaming / sync / Realtime 専用のヘルパーは提供しない

## 落とし穴

- `RunBudgetPolicy` は `on_llm_end` 判定。stream 中の暴走には `asyncio.wait_for` を併用する
- `ModelRetrySettings` の silent no-op に頼らない。フラグは既定 True 前提で組む
- `max_retries > 0` かつ有効な retry 条件がゼロは build-time `ValueError`
- `FailsafePolicy.handlers` は宣言順 first-match。より specific な型を先に宣言する責務は利用者側にある
- `Exception` / `BaseException` / `ExceptionGroup` / `KeyboardInterrupt` / `SystemExit` / `asyncio.CancelledError` / `GeneratorExit` は `handlers` のキーにできない（build-time `ValueError`）。`ExceptionGroup` は `isinstance` マッチのため `TaskGroup` が束ねた無関係な例外まで丸ごと着地させる広すぎる捕捉になるため禁止する（利用者定義のサブクラスは捕捉範囲が限定されるので許容）
- `log_on_apply`（既定 True）の warning ログには例外メッセージとトレースバックがそのまま出る。機密を含みうる例外を扱う場合は `log_on_apply=False` にし `on_apply` でマスキングしたうえで記録する（`on_apply` 側でも result を丸ごと文字列化・シリアライズしない）
- `FailsafePolicy.handlers` は不変化されるため、policy 自体の `copy.deepcopy` / `dataclasses.asdict` は `TypeError` になる。複製が必要な場合は `FailsafePolicy(dict(policy.handlers), ...)` で再構築する
- `failsafe_call` は streaming（`run_streamed`）・sync（`run_sync`）・Realtime 用の専用ヘルパーを提供しない
- `repr` マスク（`FailsafeResult.last_agent` 等）は repr 限定。`dataclasses.asdict(result)` / `vars(result)` / 属性直参照は Agent 実体を返す。監査・メトリクスへ送るときは丸ごとシリアライズせず `getattr(agent, "name", None)` 等のメタデータへ落とす
- `handlers` の値位置には着地値 / callable / `FailsafeHandler` のみを置く。Agent 実体を置くと (a) それが `final_output` として利用者へ返り、(b) `handlers` は policy repr に出るため機微が露出する。`last_agent` を指定したいなら `FailsafeHandler` を使う
- `FailsafeResult` / `RunBudgetExceeded` を監査バッファ等で長期保持すると `last_agent` 経由で Agent と参照グラフが解放されない（`exc.__traceback__ = None` でも解放されない）。長期保持するならメタデータへ落としてから
- fallback callable の**戻り値**に `RUNNING_AGENT` / `FailsafeHandler` を返さない。build-time 検証は宣言位置のみを見るため、callable の戻り値経由では sentinel / handler がそのまま `final_output` に載る
- `RUNNING_AGENT` を指定したのに `last_agent` が常に `None` になる原因は 3 通りある。(a) 例外が実行文脈を運んでいない（`run_data` も `last_agent` 属性も持たない。組み込み例外・`Runner` 外で送出された例外が該当）。この場合は**ログが一切出ない**（読み出しが正常に「無い」と返るだけ）。(b) 読み出し自体が例外を送出した。この場合のみ `logger.debug`（`exc_info=True`）に記録される。(c) 例外側が運んでいた値が `RUNNING_AGENT` そのものだった（自作例外の `last_agent` 属性に sentinel を入れた場合）。`RUNNING_AGENT` は指定値であって解決結果ではないため、解決不能として次の読み取り先・次の段へ落ちる。`oai_agentspec.resilience` logger を debug にして何も出ないなら (a) か (c) であり、具体の agent を指定するか `fallback_last_agent` を使う
- thunk が**同期的に**送出した例外は着地しない（`awaitable` を await する前に伝播する）。`lambda: build_input_then_run(...)` のように thunk 内で同期バリデーションを走らせる形では、宣言済みの例外型でも素通りする。着地させたい処理は awaitable の内側（`async def` の中）に置く
- `FailsafePolicy` は `frozen=True` だが `hash()` できない（`handlers` が `mappingproxy` のため `TypeError`）。`set` の要素・`dict` のキー・`functools.lru_cache` の引数には使えない。`ModelRetryPolicy` / `RunBudgetPolicy` は hash 可能で、この点だけ非対称

## 参照

- 詳細設計: `docs/architecture.md`（Resilience 節）
- 設計判断: `docs/adr/0002-resilience-declarative-compilation.md` / `docs/adr/0003-hooks-chain-helper.md`（`chain_hooks`）/ `docs/adr/0012-failsafe-declarative-landing.md`（Failsafe）/ `docs/adr/0013-failsafe-last-agent-resolution.md`（`last_agent` の決定モデル）
- 具体例: `examples/resilience/01_retry_and_budget.py` / `examples/resilience/02_failsafe.py`

## 次

[guardrails.md](./guardrails.md) — 入出力ガードレール 3 家族
