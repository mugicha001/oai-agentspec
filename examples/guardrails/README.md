# 内容ガードレール（oai-agentspec[guardrails]）の使い方

宣言したエージェントが「何を言うか」を入出力・中間ツール段で検査する支援層。helper は SDK 互換
guardrail（`AgentSpec` の `input_guardrails` / `output_guardrails` フィールドへ渡す・`agents.Agent`
と同型）、または `FunctionTool` へツールガードレールを装着したラップ済みツールを返す**ファクトリ**に
徹する。

「何を言うか」を検査する内容ガードレールと、「何をできるか」を許可 / 拒否する AGT ガバナンス
（ツール単位ポリシー強制）は直交する役割分担であり、相互に置き換えない。ツール境界 helper は
内容検査のみを行い、実行可否の allow / deny 制御（ポリシー強制）は新設しない。

## インストール（extra）

```bash
pip install 'oai-agentspec[guardrails]'   # 依存ゼロ opt-in extra（公開窓口分離のための境界）
```

`guardrails` 自体は追加の PyPI 依存を持たない。重い専門検知（PII / モデレーション / 注入検知
サービス）は lib に同梱せず、利用者が外部 DI で渡す（各例の任意依存を参照）。

## 検知 3 家族

| 家族 | 検知 | 同梱 / DI | helper（例） |
|---|---|---|---|
| 外部検知器（A） | Presidio / モデレーション等を薄く接着 | 検知本体は非同梱（外部 DI） | `external_detector_guardrail` |
| prompt 駆動 LLM（B） | 判定 model + 判定 prompt で LLM-as-judge | model / prompt は DI（非同梱） | `prompt_llm_guardrail` |
| 決定的・ロジック系（C） | カナリア / 正規表現 / 長さ / allow-deny / predicate / 注入ベースライン | 再利用 helper を同梱（DI 上書き可） | `canary_guardrail` / `regex_guardrail` / `length_guardrail` / `allow_deny_guardrail` / `predicate_guardrail` / `injection_baseline_guardrail` |

## 適用境界

- **agent 境界（会話入出力）**: helper は SDK 互換 `InputGuardrail` / `OutputGuardrail` を返し、
  `AgentSpec(input_guardrails=[...], output_guardrails=[...])` 専用フィールドへ直接渡す
  （`agents.Agent` と同型）。評価は SDK `Runner` が会話入出力で行い、trip すると
  `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered` を送出する。
- **ツール境界（中間ツール出力 / 引数）**: ツールガードレールは `tool_guardrail(detector, on=...)`
  で `ToolInputGuardrail` / `ToolOutputGuardrail` を生成し、ツール定義時に
  `function_tool(_func, tool_input_guardrails=[...], tool_output_guardrails=[...])` で宣言する
  （SDK ネイティブ流儀・agent 境界と対称）。既存ツール（`as_tool` 等 `function_tool` で定義し直せ
  ないもの）へ**後付け**するときは `guard_tool(tool, input_detector=..., output_detector=...)` で
  装着する。いずれも name / description / params_json_schema / needs_approval は維持される（実行
  本体・宣言メタは不変・内容検査のみ追加）。trip 時の挙動は `on_trip`（'reject' 既定 = 注釈付き
  返却で続行 / 'raise' = 中断 / 'allow' = 通過、または `Detection` を受ける callable）で選ぶ。

二境界 factory（`regex_guardrail` / `predicate_guardrail` / `length_guardrail` /
`allow_deny_guardrail` / `external_detector_guardrail` / `prompt_llm_guardrail`）は `on` を
**キーワード必須**にしている（`on="input"` / `on="output"` を明示する）。`canary_guardrail`（output
専用）/ `injection_baseline_guardrail`（input 専用）は `on` を取らない。

入力ガードレール（`on="input"` で生成したもの・`injection_baseline_guardrail` を含む）は既定で
`run_in_parallel=True`（SDK 既定）であり、判定をエージェントのターンと**並行**に走らせる（レイテンシ
優先）。このため遅い / 非同期な検知が trip する前にモデルがツールを呼びうる。**危険入力を実行前に
ブロックしたい場合は `run_in_parallel=False`** を指定し、検査完了を待ってからターンを開始させる
（`on="output"` では無効・`OutputGuardrail` に該当フィールドがない）。ツール実行の副作用はツール境界
ガードレールが実行前にゲートする役割分担を前提とする。

```python
# 例: 注入ベースラインを実行前ブロックにする（並行ではなく直列で検査）
injection_baseline_guardrail(run_in_parallel=False)
external_detector_guardrail(moderation_detector, on="input", run_in_parallel=False)
```

## 宣言面（`agents.Agent` と同型）

agent 境界 guardrail は `AgentSpec` の専用フィールド `input_guardrails` / `output_guardrails` へ
直接渡す（`agents.Agent` と同じ宣言面）。ツールガードレールは `tool_guardrail(detector, on=...)` で
生成し、ツール定義時に `function_tool(_func, tool_*_guardrails=[...])` で宣言する（SDK 流儀）。

```python
from oai_agentspec import AgentSpec, function_tool
from oai_agentspec.runtime.guardrails import (
    regex_guardrail, canary_guardrail, tool_guardrail, Detection,
)

# ツール定義時にガードレールを宣言（SDK ネイティブ流儀）。
guarded_tool = function_tool(
    _my_func,
    tool_output_guardrails=[tool_guardrail(my_detector, on="output")],
)

spec = AgentSpec(
    name="bot",
    instructions="...",
    input_guardrails=[regex_guardrail(r"\d{3}-\d{4}", on="input")],
    output_guardrails=[canary_guardrail("CANARY-TOKEN")],
    tools=[guarded_tool],
)
```

guardrail を名前で参照したい場合（UI からの有効 / 無効切り替え・危険度つきの一覧・run 全体への一括適用）は
`GuardrailRegistry` へ登録して `AgentSpec(guardrails=[...])` で宣言する。登録簿のメソッドは生成と登録を 1 回の
呼び出しで行い、境界・framework ラベル・危険度の既定を入れる（使い方は `docs/usage/safety/guardrails.md`）。

既存ツール（`as_tool` 等 `function_tool` で定義し直せないもの）へ**後付け**するときは `guard_tool`
を使う。`guard_tool(my_tool, ...)` は `my_tool` と同名のガード済みコピーを返すので、**ガード版を
`tools` に入れ、元の無防備な `my_tool` を `tools` に残さないこと**（残すと無防備ツール経由で
ガードレールをバイパスできてしまう・利用者責任）。入力・出力の両検知器は 1 回の
`guard_tool(my_tool, input_detector=..., output_detector=...)` にまとめる（同一ツールを複数の
ガード版で渡さない）。

## サンプルと OWASP 対応

| ファイル | OWASP | 適用境界 / helper | 検知家族 | 任意依存 |
|---|---|---|---|---|
| `01_injection_baseline.py` | LLM01 | input / `injection_baseline_guardrail` | C（決定的・補助） | なし |
| `02_canary_system_prompt.py` | LLM07 | output / `canary_guardrail` | C（カナリア） | なし |
| `03_prompt_llm_guardrail.py` | LLM01 | input / `prompt_llm_guardrail` | B（LLM-as-judge） | なし（判定 model は DI） |
| `04_presidio_pii.py` | LLM02 | output / `external_detector_guardrail` | A（Presidio） | `presidio-analyzer` ほか |
| `05_moderation_external.py` | LLM01/05 | input / `external_detector_guardrail` | A（モデレーション） | なし（openai は本体同梱） |
| `06_tool_output_guardrail.py` | LLM02/05 | tool 出力 / `tool_guardrail` + `function_tool`（後付けは `guard_tool`） | C（regex）他 DI 可 | なし |
| `07_guardrail_registry.py` | LLM01/02/07 | `Boundary` 4 値すべて / `GuardrailRegistry`（facade 9 + register 経路 + 名前参照 + run 単位） | A（DI 検知器）/ B（判定 LLM）/ C（6 種）+ 自作 | なし |
| `08_canary_run_scoped.py` | LLM07 | output / `canary_guardrail`（resolver）+ `AgentSpec.instructions_append` | C（カナリア） | なし |

`07` は登録簿（`GuardrailRegistry`）で「名前の強制・適用境界の宣言・分類メタデータの宣言・名前
参照」を 1 本で通す例で、`oai_agentspec.runtime.guardrails` の公開シンボル 27 件・facade 9 件・
照会 6 件すべてを実際に使う（4 フェーズ構成: 宣言と一覧 / 名前参照と `validate()`・`clone()`・
`freeze()` / run 単位 / detector 単独利用と分類データとツール境界）。`01`〜`06` の**実体を
フィールドへ直接渡す経路は現行のまま有効**で、登録簿はそれを置き換えるものではない（名前で
照合したい・境界とメタデータを宣言として保持したい場合の追加経路）。`07` の各フェーズは
`docs/usage/safety/guardrails.md` の「名前で参照する」「適用範囲を選ぶ」「危険度と一覧」
「検知器を単独で使う」に対応する。

`08` は `02` の固定値カナリアを会話ごとに一意な値へ広げる例で、埋め込み
（`AgentSpec.instructions_append`）と検知（`canary_guardrail` の resolver）の両方が run スコープの
値を読む形を通す。どちらも構築・build の時点では評価されず、run / 検知呼び出しごとに再解決される
ため、同じ Agent 実体を使い回したまま会話ごとのトークンを逐語照合できる（近似照合へ劣化させない）。
カナリアの発行と会話単位の管理、埋め込み文言はいずれも利用側の責務。

facade は `on` から適用境界を導出する。framework ラベルと既定危険度が自動で付くのは、**分類が
helper 名で一意に定まる 2 件**（`canary_guardrail` / `injection_baseline_guardrail`）に限る。残りの
helper は検知の実体（パターン・述語・検知器・判定モデル）が利用者側にあるため分類が確定せず、
`labels` / `severity` を渡さなければ空 dict / 未宣言のままになる（`specs(min_severity=...)` や
labels フィルタの対象に入らない）。

helper は OWASP に限定されず MITRE ATLAS / NIST AI RMF / 品質・ブランド系の内容検査にも適用できる
（検知家族は framework 中立）。カバレッジマトリクスと選定根拠は
`docs/rationale/content-guardrails-coverage.md` を参照。

## 実行

Azure OpenAI の環境変数（`AZURE_OPENAI_*`・`examples/_shared/_azure.py` 参照）を `.env` に設定して
実行する:

```bash
uv run python examples/guardrails/01_injection_baseline.py
```

## 任意依存の導入（A 家族の外部検知器）

検知本体は lib 非同梱で、各 example 内で遅延 import する（未導入時は導入方法を案内して終了）。

```bash
# 04_presidio_pii.py（PII 検出）
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg

# 05_moderation_external.py は openai（本体同梱）のモデレーションを使う。
# Azure Content Safety / Llama Guard 等へは検知 callable を差し替えるだけで切替できる。
```

## 注意（補助検知の限界）

- **注入ベースライン（`injection_baseline_guardrail`）は補助検知**であり網羅的検知ではない。
  注入対策の本丸は**パラメータ化クエリ / 安全 API 利用**である。既定パターンは自然文入力で誤検知
  （false positive）しやすく検知漏れ（false negative）も前提のため、`extra_patterns` で利用者の
  入力分布に応じて調整する（完全な差し替えは `regex_guardrail` を直接使う）。
- **正規表現は untrusted テキストに適用される**。DI するパターンは利用者責任で ReDoS 安全なもの
  （壊滅的バックトラックを起こさない）を渡すこと。
- **prompt 駆動 LLM guardrail の既定 verdict は fail-open**（judge 出力が空 / 不正のとき trip
  しない）。fail-closed が必要なら `verdict=` に空応答を trip 扱いにするパーサを DI する。
- prompt 駆動・モデレーション系は非決定的でコスト / レイテンシを伴う。決定的に確認できる箇所
  （C 家族）を優先し、B / A は決定的検知で捉えきれない文脈依存の判定に限定して二層で用いる。

整合性 / 供給網インテグリティ（改竄検知・供給網の信頼）は内容検査では守れない直交領域であり
別途扱う。
