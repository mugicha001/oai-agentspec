# 0028: 実行可能アクション宣言と決定的スロット確定の型・配置

- Status: accepted
- Date: 2026-08-11

## Context

実行可能アクションの宣言（`ActionSpec` / `ParameterSpec` / `param`）と、決定的なスロット確定
（`Slot` / `ActionPlan`）を `runtime/intent` へ追加するにあたり、実測でしか判明しない非自明な
判断がいくつも発生した。これらは diff からは読み取れず、後から「なぜこの形なのか」を復元できない
ため記録する。

### `ParameterSpec.default` の内部表現

`param()` の公開シグネチャは `default=PARAM_UNSET`（sentinel）を既定とするが、この sentinel を
`ParameterSpec` のフィールドとしてそのまま保持すると、`model_json_schema()` が
`PydanticJsonSchemaWarning: Default value ... is not JSON serializable` を毎回出す。
`ParameterSpec` の `model_json_schema()` 成立は宣言の契約であり、警告が出続ける状態は避けたい。

回避案を 4 つ実測した。

| 案 | 形 | 警告 | schema の `default` プロパティ |
|---|---|---|---|
| A | `Field(default=PARAM_UNSET, json_schema_extra={"default": None})` | **出る** | — |
| B | `default: Any = None` + `has_default: bool = False` | 出ない | 残る |
| C | `default: SkipJsonSchema[Any] = Field(default=PARAM_UNSET)` | 出ない | **消える** |
| D | `default: Any = Field(default_factory=lambda: PARAM_UNSET)` | 出ない | 残る |

**A では抑止できず、B / C / D では抑止できた。** C は `default` フィールドが
`model_json_schema()` から丸ごと消えるため、`ParameterSpec` のスキーマを UI フォーム生成へ
使い回す用途に対し宣言フィールドの存在自体が見えなくなる。D は 1 フィールドで済むが、
モジュール定数を返す lambda を `default_factory` に渡す形は「schema 警告を避けるため」以外の
意図が読み手に伝わらず、利用者が「未宣言か」を判定するには `PARAM_UNSET` を import して
identity 比較する必要がある。

### 予測段スキーマモデルの `max_length`

`max_suggestions` は「超える分は切り捨てる」契約である。実測で、`Field(max_length=N)` を付けた
parse 派生モデルは超過件数に対し切り捨てではなく `ValidationError` を送出することを確認した。
上限提示用のモデルと応答検証用の parse 派生を同一にすると、LLM が上限を 1 件超えただけで応答
全体が `ValidationError` になり、応答不正時の後退（`on_invalid_response`）の判断材料が失われる。

### `ExecutableSuggestion` が `IntentPrediction` を経由しない理由

`IntentPrediction.candidates` の宣言型は `tuple[IntentCandidate, ...]` である。実測の結果:

| 経路 | 結果 |
|---|---|
| インスタンス直渡し | `ExecutableIntent` のまま保持・追加フィールド保持 |
| `model_validate({"candidates": [ei]})` | 保持 |
| `model_validate({"candidates": [{...純 dict...}]})` | `IntentCandidate` へ coerce・**追加フィールド消失** |
| `model_dump()` | 親型のスキーマで直列化され `action_id` / `parameters` が現れない |

JSON から復元した `IntentPrediction` では追加フィールドが落ちるため、`ExecutableSuggestion` が
`IntentPrediction` を丸ごと保持する形は、経路によって候補の型が変わる不安定な契約になる。

### 配置と依存方向

`slots.py` はドメイン型を持つ層であり、モデル生成の汎用ビルダ（`_models.py`）がドメイン型を
import すると `slots.py -> _models.py -> slots.py` の循環が生じる。また `runtime/lightning` に
既に `slots.py`（`prompt_slot` = APO のプロンプト分割）が存在し、無 prefix の同名モジュールを
置くことによる概念の混同が懸念された（import 衝突はパッケージが異なるため起きないことを実測で
確認済み）。

### lockdown 環境での起動時検証

`PromptStore` の `lockdown()` 後に manifest 未掲載のセグメントを要求すると
`PromptTemplateIntegrityError` が送出される。この例外の MRO は
`PromptTemplateIntegrityError -> IntegrityError -> Exception` であり、`KeyError` 派生でも
`ValueError` 派生でもないため、`except PromptResolutionError` をすり抜けることを実測で確認した。
起動時検証がこれを捕捉して集約 `ValueError` へ変換すべきかを判断する必要があった。
なお「同 stem 複数一致（曖昧）はメッセージで区別できない」という当初の想定は誤りで、実測では
曖昧一致は `PromptResolutionError` として区別可能なメッセージを持つ。

### 既定マージ解決の所在

`prompt` / `prompt_vars` / `on_invalid_slot` は `ActionCatalog` 既定と `ActionSpec` 個別宣言の
両方に現れ、マージ（前 2 者）と上書き（後 1 者）で解決される。この解決を `ActionSpec` のメソッド
にすると catalog を引数に取ることになり、`ActionCatalog` のメソッドにすると起動時検証と予測段の
双方から同一実装を呼ぶ経路が作りにくい。

### 単方向依存の許可先

分離先の関数実行エージェント（`docs/requirements/function-execution-agent.md`）は
`AgentSpec(model=DeterministicResponseModel(rule), ...)` を返すため `runtime/deterministic` を
参照する。これは `runtime/intent` の許可先として列挙されていない sibling 参照である。

## Decision

1. **`ParameterSpec` は sentinel をフィールドに持たない**。`param()` が `PARAM_UNSET` を受け取り、
   `ParameterSpec(default=..., has_default=...)`（案 B）へ正規化する。B を選んだのは、「未宣言」と
   「明示的な `default=None`」を bool フィールドで型として分離できるという、schema 警告とは
   独立に成立する利点による。**他に警告を抑止する手段が無いからではない**（C / D でも抑止できる）。
   利用者から見た API（`param(..., default=PARAM_UNSET)`）は変わらない。sentinel の名は
   `PARAM_UNSET`（既存 `TOOL_UNSET` と同型の命名）とし、`runtime/intent/actions.py` に置く
   （コアへ intent 専用定数を持ち込まない）。
2. **parse 派生モデルには `max_length` を付けない**。上限提示用のモデルにのみ
   `Field(max_length=max_suggestions)` を付け、`model_json_schema()` の `maxItems` として
   LLM へ上限を伝える。超過分は lib 側が `ConfidenceLevel` 降順の stable sort 後にスライスで
   切り捨てる（ソートキーは `_llm.py` の `_LEVEL_ORDER` と単一ソースを共有する）。
3. **`ExecutableSuggestion` は `tuple[ExecutableIntent, ...]` を直接持つ**。`IntentPrediction` を
   丸ごと保持せず、`report` / `metadata` は素通しで分解して持つ。直列化で追加フィールドが落ちる
   ことは、`ExecutableSuggestion` が `run_context`（任意型）を含み直列化契約を持たないことと整合する。
4. **`runtime/intent/slots.py` は無 prefix とし、`_models.py` はドメイン型を import しない**。
   `_models.py` は呼び出し側が組み立てた annotation を引数で受ける汎用ビルダに徹し、
   `slots.py -> _models.py` の一方向を保つ。`runtime/lightning/slots.py` との概念の混同を避ける
   ため、`runtime/intent/slots.py` の module docstring 冒頭に「本モジュールのスロットは
   『1 パラメータの解決状態』であり、`runtime/lightning` の `prompt_slot`（APO のプロンプト分割）
   とは別概念である」を置く。
5. **起動時検証は `PromptTemplateIntegrityError` を捕捉せず素通しで伝播させる**。起動時検証は
   「宣言の不整合を洗い出す」機能であり、インテグリティ違反は別カテゴリの fail-closed 事象である。
   これを他の宣言不整合と同じ `ValueError` へ集約すると、fail-closed であるべき違反が「宣言ミスの
   一種」として埋もれる。前提条件を次のとおり明記する（省くと「lockdown 後は常に落ちる」と誤読
   されうる）。

   > `lockdown()`（の段 2 `_preload`）を通した `PromptStore` に対して、**manifest に未掲載の
   > セグメント**（= `_cache` に載っていないセグメント）を `prompt` が要求した場合に限り、
   > `catalog.validate()` は集約 `ValueError` ではなく `PromptTemplateIntegrityError` を
   > 送出する。manifest 掲載済みのセグメントは lockdown 後も `_cache` から解決され、検証は通常どおり
   > 完了する。

   運用上は `catalog.validate()` を `lockdown()` の**前**に呼ぶことを推奨とする。曖昧一致は
   `PromptResolutionError` として捕捉され、「解決できないセグメント」の集約 `ValueError` に含まれる。
6. **既定マージ解決は `actions.py` のモジュールレベル純関数 3 件へ一元化する**
   （`resolve_prompt` / `resolve_prompt_vars` / `resolve_on_invalid_slot`）。catalog と spec の
   双方を引数で受けることで、どちらの型にも解決ロジックを埋め込まずに起動時検証と予測段の双方から
   同一実装を呼べる。解決結果は決定的段で `ActionPlan` の `resolved_*` フィールド（いずれも
   `Field(exclude=True)`）へ載せて運び、予測段は `ActionCatalog` を受け取らない。`ActionPlan.spec`
   は as-declared のまま保持する（マージ済みで上書きすると「当該 `ActionSpec` 自身の宣言だけを
   判定対象とする」検査が原理的に書けなくなるため）。
7. **単方向依存の許可先へ `runtime/deterministic`（sibling・extra を持たない）を 1 件加える**。
   先例は `runtime/cli/main.py` の `from ..conversation import SessionPolicy` である。extra の
   追加コストは無く、SDK 隔離にも影響しない。なお当初は「SDK のロードを早めないよう遅延 import に
   すべき」と推測しかけたが、`import oai_agentspec` の時点で既に `agents` が `sys.modules` に
   載ることを実測で確認したため、遅延にする理由が無く撤回した。

現在仕様の SoT は `docs/architecture.md`（「意図予測（`runtime/intent`）」節の
「実行可能意図の宣言と決定的スロット確定」小節）とし、本 ADR は判断・却下案のみを記録する。

## Consequences

- + `ParameterSpec.model_json_schema()` が警告なしで成立し、かつ「未宣言」と「明示 `None`」を
  型で分離できる。利用者は sentinel の identity 比較を書かない。
- + LLM が上限を 1 件超えた応答も切り捨てで受け止められ、応答全体が失われない。
- + 候補の型が経路（インスタンス直渡し / JSON 復元）に依存せず安定する。
- + インテグリティ違反が宣言ミスに埋もれず fail-closed のまま届く。
- + 既定マージ解決の実装が 1 箇所に閉じ、起動時検証と予測段で再実装されない。
- - `ParameterSpec` のフィールドが 1 件増える（`has_default`）。`param()` のシグネチャは
  要件どおりで、内部表現のみが異なる。
- - 上限提示用モデルと parse 派生モデルの 2 つを持つため、スキーマ生成が 1 段複雑になる。
- - lockdown 済み環境では起動時検証が集約 `ValueError` 以外の例外型を送出しうるため、利用者は
  2 種の例外を意識する（前提条件を docstring と本 ADR に明記して緩和する）。

## Confirmation

強制手段（いずれも**新規作成**）:

- `tests/runtime/intent/test_actions_l1.py`: 宣言型の frozen・名前検証の 4 分岐・
  `on_invalid_slot` の値検証が宣言時に落ちること・`model_json_schema()` が警告なしで成立すること・
  既定マージ解決 3 関数。
- `tests/runtime/intent/test_slots_l1.py`: `ActionPlan.model_dump()` に `spec` / `agent` /
  `action_agent` / `resolved_*` / `run_context` 由来のキーが現れないこと・決定的段の決定性・
  `label` の render。
- `tests/runtime/intent/test_models_l1.py`: `parameters_model()` のキャッシュ同一性・`maxItems` の
  反映・parse 派生の欠落許容・`PydanticSchemaGenerationError` -> `ValueError` 変換。
- `tests/runtime/intent/test_validate_l1.py`: 起動時検証 9 種の検査と集約。
- `tests/runtime/intent/test_suggest_l1.py`: 非 `ExecutableIntent` 候補と未登録 `action_id` が
  同一 WARNING で除外されること。

`docs/QUALITY-GUARANTEES.md` に登録済み（source = ADR-0028）。
