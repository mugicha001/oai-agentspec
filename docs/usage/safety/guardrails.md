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

`triggered: bool`, `reason: str | None = None`, `info: dict[str, Any] = {}`。

## 判断軸

- 既知の注入パターン・PII は **静的パターン系 or 外部検知器**で高速に弾く。LLM 判定は最後の砦
- プロンプト注入対策は **input**、機密漏洩・出力ポリシー違反は **output** 段で検査
- tool の副作用を止めたいなら **tool_guardrail**（宣言時）or **guard_tool**（既存 tool を後付けラップ）

## 落とし穴

- LLM 判定系は追加レイテンシとコスト。頻度の低い最終段に限定する
- 既定 `INJECTION_BASELINE_PATTERNS` は最小構成。プロダクションでは DI で組織固有パターンを追加する
- `run_in_parallel=True`（既定）だと trip 前にモデルがツールを呼びうる。実行前ブロックが要るなら `False` または tool ガードを併用

## 参照

- 詳細設計: `docs/architecture.md`（内容ガードレール節）
- 検討経緯: `docs/rationale/content-guardrails-coverage.md`
- 具体例: `examples/guardrails/01_injection_baseline.py` 〜 `06_tool_output_guardrail.py`

## 次

[governance.md](./governance.md) — ツール単位ポリシーと監査ログ
