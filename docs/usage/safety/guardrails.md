# 内容ガードレール

## 何を解決するか

エージェントが「何を言うか / 何を受け取るか」を入出力段で検査し、注入攻撃・PII 漏洩・不適切出力を防ぎます。本ライブラリはガードレールを 3 家族（LLM 判定系 / 静的パターン系 / tool ガード系）に整理し、`AgentSpec.input_guardrails` / `output_guardrails` へ渡せる SDK 互換オブジェクトを返す helper ファクトリで提供します。

重い専門検知（PII / モデレーション / 注入検知サービス）は lib 非同梱で利用者 DI、既定 helper（注入ベースライン等）は DI で上書き可能です。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| LLM 判定系（`prompt_llm_guardrail` / `canary_guardrail`） | 別 Model にプロンプトで判定 | 意味的判定・柔軟なルール |
| 静的パターン系（`regex_guardrail` / `length_guardrail` / `allow_deny_guardrail` / `injection_baseline_guardrail`） | 正規表現・長さ・語彙リスト | 高速・決定的な既知パターン検知 |
| 外部検知器連携（`external_detector_guardrail`） | 利用者 DI の検知器を接着 | PII / モデレーション等の専門サービス |
| tool ガード系（`tool_guardrail` / `guard_tool`） | ツール入出力を検査 | 危険 tool の副作用抑止 |
| input vs output | 入力段 or 出力段のどちらで検査するか | プロンプト注入は input、機密漏洩は output |

## 使い方

- import: `from oai_agentspec.runtime.guardrails import (prompt_llm_guardrail, canary_guardrail, regex_guardrail, length_guardrail, allow_deny_guardrail, injection_baseline_guardrail, external_detector_guardrail, tool_guardrail, guard_tool, Detection, INJECTION_BASELINE_PATTERNS)`
- extras: `pip install oai-agentspec[guardrails]`（追加外部依存なし）
- 依存 env: 外部検知器を使う場合はその env

```python
from oai_agentspec import AgentSpec
from oai_agentspec.runtime.guardrails import injection_baseline_guardrail, regex_guardrail

spec = AgentSpec(
    name="assistant", instructions="...",
    input_guardrails=[injection_baseline_guardrail()],
    output_guardrails=[regex_guardrail(r"\d{16}", on="output")],
)
```

### 名前で参照する（登録簿経由）

guardrail を登録簿へ登録すると、`handoffs` と同じ流儀で名前で宣言できます。登録簿のメソッドは生成と登録を 1 回の呼び出しで行い、登録名がそのまま上流 SDK 側の guardrail 名になります（トレース上の表示名と照合キーが食い違いません）。適用境界は `on` の指定から導出されます（helper 自体で境界が固定されるものは `on` を取りません）。framework ラベルと既定危険度が自動で付くのは、分類が helper 名で一意に定まる 2 件（`canary_guardrail` / `injection_baseline_guardrail`）に限ります。残りの helper は検知内容が利用者 DI で決まるため既定を持たず、`labels` / `severity` を渡さなければ未分類のままです（`min_severity` や labels での絞り込み対象に入りません）。いずれも引数で上書きできます。

```python
from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.guardrails import GuardrailRegistry, Severity

guardrails = GuardrailRegistry()
guardrails.injection_baseline_guardrail(name="injection_baseline")
guardrails.canary_guardrail(CANARY, name="system_prompt_canary", severity=Severity.CRITICAL)

registry = AgentRegistry(guardrail_registry=guardrails)
registry.register(AgentSpec(
    name="assistant", instructions="...",
    guardrails=["injection_baseline", "system_prompt_canary"],   # 境界は宣言から解決される
))
registry.validate()   # 未登録名・未注入を run 前に一括検出
```

生の SDK guardrail や自作の実体を名前参照へ載せたい場合は `register()` を使います。この経路では登録時に「上流 guardrail 型か」「名前が一致するか」「宣言境界と実体の境界が一致するか」を検証し、ラベルと危険度の自動付与は行いません。

### 適用範囲を選ぶ（Agent 単位 / run 単位）

| 適用範囲 | 書き方 | 評価される範囲 |
|---|---|---|
| Agent 単位 | `AgentSpec.guardrails` または専用フィールド | その Agent のみ。出力側は**ハンドオフ先に付いていなければ評価されない** |
| run 単位 | `RunConfig(**guardrails.run_config_kwargs([...]))` | run 全体。ハンドオフ先の出力も対象になる |

カナリアのようにシステム全体へ一律に掛けたい guardrail は run 単位を選びます。`run_config_kwargs()` が境界別に振り分けたマッピングを返すので、`**` 展開でそのまま渡せます。

```python
from agents import RunConfig, Runner

cfg = RunConfig(**guardrails.run_config_kwargs(["system_prompt_canary"]))
result = await Runner.run(registry.get("assistant"), input=text, run_config=cfg)
```

ツール境界の guardrail は `agents.Agent` にも `RunConfig` にもフィールドがないため名前参照の対象外です。登録簿から取り出してツール定義時に渡します（名前参照へ渡すと `ValueError` になります）。

```python
guardrails.tool_guardrail(my_detector, on="output", name="tool_pii")
guarded = function_tool(_my_func, tool_output_guardrails=[guardrails.get("tool_pii")])
```

### セッション単位で入れ直す（カナリア等）

カナリートークンのようにセッション毎に値を入れ替えるものは、共有の登録簿へ固定値として載せず、**セッション毎に登録簿を作り直して run 単位で渡します**。登録簿は宣言の保持に徹しており、登録済みの guardrail を差し替えるメソッドを持ちません（同名の再登録は `ValueError`）。名前は固定のまま、実体の生成だけをセッション境界へ寄せる形になります。

```python
def session_guardrails(canary: str) -> GuardrailRegistry:
    reg = GuardrailRegistry()
    reg.canary_guardrail([canary], name="session_canary")
    return reg

per_session = session_guardrails(issue_new_canary())
cfg = RunConfig(**per_session.run_config_kwargs())
result = await Runner.run(registry.get("assistant"), input=text, run_config=cfg)
```

`AgentSpec.guardrails` の名前参照は build 時に解決されて Agent へ焼き付くため、値が入れ替わるものには向きません。`AgentRegistry` は構築済み Agent をキャッシュするので、解決元が返す実体を差し替えても `update()` で invalidate するまで反映されません。セッションを跨いで固定のもの（注入検知のベースライン・固定の禁止語彙・長さ上限）は名前参照で宣言し、入れ替えるものは run 単位に置く、という切り分けになります。

解決元そのものを差し替える経路（`GuardrailProvider` の自作実装を `AgentRegistry(guardrail_registry=...)` へ注入する）も取れますが、構築済み Agent への反映に `update()` が必要になるため、値だけを入れ替えたい用途では run 単位のほうが単純です。

### run ごとにカナリアを解決する（resolver）

登録簿を作り直さずに run ごとの値を扱いたい場合は、`canary_guardrail` へ固定値ではなく resolver
（`(context, agent) -> str | Iterable[str] | None`）を渡します。resolver は登録時に評価されず、検知
呼び出しのたびに再解決されるため、登録簿・guardrail の実体を共有したまま run ごとのトークンを逐語
照合できます。

```python
guardrails = GuardrailRegistry()
guardrails.canary_guardrail(
    lambda ctx, agent: ctx.context.canary_token,   # run ごとに再解決される
    name="session_canary",
    severity=Severity.CRITICAL,
)
```

- `ctx.context.<attr>` で run context を開きます（`AgentSpec.instructions_append` と同じ引数規約）。
  resolver は同期関数として書いてください。`async def` の関数（および `async def __call__` を持つ
  オブジェクト）は登録時に拒否されます。
- resolver が返せるのは `str` / 文字列の iterable / `None` のみです。dict を返すとキーが照合対象に
  なってしまうため拒否されます（`TypeError`）。
- resolver が `None` や空を返した run では発火しません（「この run にはカナリアが無い」扱い）。
- トークンは lib の外で発行した**高エントロピーな値**にしてください。短い値はほぼ全出力に含まれて
  しまい、任意発火によるアラート疲弊を作れます。
- 発火時の情報にはマッチしたトークンが含まれ、SDK のトレーシング経由で観測基盤へ流れます。
  トークンを機密として扱う場合は送信先を確認してください。
- 固定値（`str` / `Iterable[str]`）を渡す既存の書き方は変わりません。セッション境界で値が固定される
  なら、登録簿をセッション毎に作り直す上記の形でも構いません。
- プロンプト側へ同じトークンを埋め込む書き方は
  [agents](../core/agents.md) の `instructions_append` を参照してください。

### 危険度と一覧

危険度は `low` < `medium` < `high` < `critical` の順序を持ちます（未宣言は順序比較の対象外）。監査や UI 表示のために登録済みの宣言を一覧で取り出せます。

```python
for spec in guardrails.specs():                       # 名前昇順
    # `boundary` は `str` 併用の列挙メンバ。表示には `.value` を使う
    #（そのまま埋めると `Boundary.INPUT` と出ます）。`severity` は IntEnum なので `.name`。
    print(spec.name, spec.boundary.value, spec.severity.name.lower() if spec.severity else "-")

guardrails.specs(min_severity=Severity.HIGH)          # 危険度 high 以上のみ
```

既定危険度はライブラリが付す出発点であり、運用ポリシーに応じて上書きする前提です（同じカナリア漏洩を `critical` と見るか `medium` と見るかは運用次第）。

トリップ時の例外から guardrail 名を引けるので、危険度に応じた着地を利用者側で組めます。

```python
except OutputGuardrailTripwireTriggered as exc:
    name = exc.guardrail_result.guardrail.get_name()   # 登録名と一致する
    severity = guardrails.metadata(name).severity
```

### 検知器を単独で使う

guardrail の中身にあたる検知器（テキストを受けて `Detection` を返す純関数）は単独で公開されています。上流 SDK のフックがない場所（webhook 応答・バッチ処理・自作パイプライン）で同じ検知を再利用できます。

```python
from oai_agentspec.runtime.guardrails import canary_detector, regex_detector

if canary_detector(CANARY)(webhook_body).triggered:
    ...
```

`Detection.info` にはマッチした値そのもの（カナリートークン等）が入ります。guardrail 経路ではこの値が SDK のトレースへ載るため、単独利用でも `Detection` を丸ごとログへ出さず `.triggered` /`.reason` を使ってください。

## パラメータ一覧（主要 factory 抜粋）
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


15 個超えるためすべてを網羅せず、代表 6 個を掲載します（残りは docstring 参照）。

### `prompt_llm_guardrail(model, prompt, *, on, verdict=None, name=None, run_in_parallel=True)`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `model` | `Any` | 必須 | 判定 LLM（不透明値・DI） |
| `prompt` | `str` | 必須 | 判定 prompt 本文 |
| `on` | `str` | 必須（kw_only） | `"input"` or `"output"` |
| `verdict` | `Callable[[str], Detection] \| None` | `None` | 既定は `UNSAFE` トークン照合 |
| `name` | `str \| None` | `None` | guardrail 名 |
| `run_in_parallel` | `bool` | `True` | 入力境界のみ有効 |

### `regex_guardrail(patterns, *, on, flags=0, name=None, run_in_parallel=True)`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `patterns` | `str \| Iterable[str]` | 必須 | 正規表現 |
| `on` | `str` | 必須（kw_only） | `"input"` or `"output"` |
| `flags` | `int` | `0` | `re.compile` フラグ |
| `name` / `run_in_parallel` | 上と同じ | — | — |

### `length_guardrail(*, max_length=None, min_length=None, on, name=None, run_in_parallel=True)`

`max_length` と `min_length` の少なくとも一方が必須（両方 None は `ValueError`）。

### `allow_deny_guardrail(*, deny=None, allow=None, case_sensitive=True, on, name=None, run_in_parallel=True)`

`deny` / `allow` は `Iterable[str] | None`。`deny` のいずれかを含むと trip、`allow` 指定時はいずれも含まなければ trip。

### `injection_baseline_guardrail(extra_patterns=None, *, name=None, run_in_parallel=True)`

`InputGuardrail` を返す（入力専用・on 引数を持たない）。

### `tool_guardrail(detector, *, on, on_trip="reject", name=None)` / `guard_tool(tool, *, input_detector=None, output_detector=None, on_trip="reject")`

`on_trip` は `"reject"` / `"raise"` / `"allow"` または `Callable[[Detection], Any]` DI。

### `Detection`（dataclass）

`triggered: bool`, `reason: str | None = None`, `info: Any = None`。

`triggered` は真の `bool` が必須で、truthy な非 bool（`re.Match` 等）を渡すと構築時 `ValueError` になる。`Detection` は検知関数が run 中に構築する型のため、この `ValueError` は build 時ではなく**実行時**に出る。`InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered` ではないので tripwire を捕捉するコードでは拾えない。自作の検知関数では `Detection(triggered=bool(...))` のように明示変換する。

## helper の framework 分類と既定危険度

登録簿が自動付与する既定値の一覧です。**本表は `HELPER_DEFAULTS`（コードが SoT）の投影です。**機械可読データが必要な場合は表を書き写さず `HELPER_DEFAULTS` を import してください。表の列構成（見出し行の 5 列）・framework ラベル列・「利用者が宣言」の行集合はテストでコード側と双方向に照合されます。既定危険度列・適用境界列・備考列のセル値は照合対象に含めません（既定危険度を除外する理由は `docs/adr/0015-declarative-guardrail-registry.md` を参照）。

| helper 識別子 | 適用境界 | framework ラベル | 既定危険度 | 備考 |
|---|---|---|---|---|
| `injection_baseline_guardrail` | input | `owasp_llm: LLM01` | medium | 非網羅の補助検知（本丸はパラメータ化クエリ・安全 API 利用） |
| `canary_guardrail` | output | `owasp_llm: LLM07` | high | 逐語一致。運用ポリシーに応じて `critical` へ上書きする |
| `regex_guardrail` | on 引数 | 利用者が宣言 | 利用者が宣言 | 分類は DI するパターン次第 |
| `predicate_guardrail` | on 引数 | 利用者が宣言 | 利用者が宣言 | 分類は DI する述語次第 |
| `allow_deny_guardrail` | on 引数 | 利用者が宣言 | 利用者が宣言 | 分類は DI する語彙次第 |
| `length_guardrail` | on 引数 | 利用者が宣言 | 利用者が宣言 | 分類は DI する閾値次第 |
| `external_detector_guardrail` | on 引数 | 利用者が宣言 | 利用者が宣言 | Presidio 接着なら LLM02、モデレーション接着なら LLM01 / LLM05 |
| `prompt_llm_guardrail` | on 引数 | 利用者が宣言 | 利用者が宣言 | 分類は DI する判定 prompt / model 次第 |
| `tool_guardrail` | on 引数（tool 境界へ写る） | 利用者が宣言 | 利用者が宣言 | 分類は DI する検知器次第 |

上段 2 件は helper 自体で適用境界と分類が固定されるため既定値を持ちます。下段 7 件は検知内容が利用者 DI で決まるため既定値を持たず、必要なら `labels` / `severity` を明示して宣言します（誤ったラベルを自動で付けないための切り分けです）。

## 判断軸

- 既知の注入パターン・PII は **静的パターン系 or 外部検知器**で高速に弾く。LLM 判定は最後の砦
- プロンプト注入対策は **input**、機密漏洩・出力ポリシー違反は **output** 段で検査
- tool の副作用を止めたいなら **tool_guardrail**（宣言時）or **guard_tool**（既存 tool を後付けラップ）

## 落とし穴

- LLM 判定系は追加レイテンシとコスト。頻度の低い最終段に限定する
- 既定 `INJECTION_BASELINE_PATTERNS` は最小構成。プロダクションでは DI で組織固有パターンを追加する
- `run_in_parallel=True`（既定）だと trip 前にモデルがツールを呼びうる。実行前ブロックが要るなら `False` または tool ガードを併用
- `RunConfig` へ渡した**入力** guardrail は初回ターンの入力にしか掛からない（上流が最初のターンに限定している）。毎ターンの入力を検査したいものは Agent 単位で宣言する。出力 guardrail にはこの制約がなく、ハンドオフ先が最終出力を出しても評価される
- `run_config_kwargs()` を引数なしで呼ぶと登録全件が対象になり、ツール境界の登録が 1 件でも混ざっていれば `ValueError` になる（静かに除外しない）。混在させる場合は対象名を明示する

## 参照

- 詳細設計: `docs/architecture.md`（内容ガードレール節）
- 検討経緯: `docs/rationale/content-guardrails-coverage.md`
- 設計判断: `docs/adr/0015-declarative-guardrail-registry.md`（登録簿）/ `docs/adr/0023-run-scoped-instructions-and-canary-resolver.md`（カナリアの run スコープ解決）
- 具体例: `examples/guardrails/01_injection_baseline.py` 〜 `06_tool_output_guardrail.py`

## 次

[governance.md](./governance.md) — ツール単位ポリシーと監査ログ
