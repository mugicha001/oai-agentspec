# 実行可能意図の宣言基盤（runtime/intent 拡張）

## 1. 概要
既存の意図予測基盤 `runtime/intent`（現在発話をカテゴリ名へ分類する）を、「次に実行したい意図をパラメータ付きで予測し、追加の文章入力なしで実行できる形に整える」用途まで拡張する。追加するのは実行可能アクションの宣言簿・宣言からの型生成・空欄確定の決定的な純関数・不足パラメータの予測（専用エージェントへ 1 ターン 1 回委譲）・起動時検証であり、候補の提示 UI・記憶の永続ストア・候補生成方式の実装・実行そのものは含まない（実行は利用側が SDK `Runner.run` で行う build-don't-run 方針を維持する）。

## 1.1 この要件書の読み方

以下は本文（第 2 章以降）を読む前の見取り図であり、契約は各 FR の受け入れ基準が正とする。

公開シンボルの件数は設計方針（ の「件数の Single Source of Truth」節）を実体とし、
本要件書に現れる件数はその転記である。件数を更新する場合は当該節を先に更新する。

### 追加する API を時間軸に並べた全体像

| 段階 | いつ | 呼ぶもの | LLM 呼び出し |
|---|---|---|---|
| 0 | アプリ起動時に 1 度 | `ActionCatalog` / `ActionSpec` / `param` で宣言し、`catalog.bind(registry=..., prompts=..., guardrail_registry=..., candidates=CandidateSource(...), llm_filler=LLMFiller(...))` で結線し、`catalog.validate(...)` で検証 | 0 |
| 1-2 | 各ターンの応答後 | `await catalog.plan(query)`（候補生成 -> 空欄の決定的確定 -> 不足があれば予測を 1 回） | 候補生成 0〜1 + 予測 0〜1 |
| 3 | 利用者がボタンを押した瞬間 | `plan.apply(answers)` -> `Runner.run(registry.get(plan.action_agent), input=plan.input_json)` | 0 |
| 4 | 実行後 | `action_next_turn_agent` で会話を窓口エージェントへ戻す | 0 |

段階 3 で LLM を呼ばないこと（押した瞬間の待ち時間と課金が発生しないこと）が本機能の狙いであり、
コストが乗るのは段階 1-2 の事前予測のみである。段階 1-2 は `catalog.plan()` の 1 呼び出しに
畳まれており、`predict=False` を渡せば予測を行わない決定的な段だけを取り出せる。

### 使い方の骨格（説明用の擬似コード）

```python
# 段階 0: 起動時
catalog = ActionCatalog(prompt=("intent/common",))
catalog.register(ActionSpec(
    "run_load_test", "負荷試験を実行する",
    action_agent="load_test_runner",
    label="${target} に ${seconds} 秒の負荷試験",
    parameters=[
        param("target", str, from_context=("current_env.host",)),        # 決定的に埋まる
        param("seconds", int, by_agent=True, default=30, confirm=True),  # 予測 + 確認
    ],
))
# action_agent が指す先は利用者が registry へ登録した任意のエージェントでよい
catalog.bind(
    registry=registry, prompts=prompts,
    guardrail_registry=my_guardrail_registry,           # ガードレールの解決簿
    candidates=CandidateSource(generator=my_generator), # 候補の出どころ
    llm_filler=LLMFiller(                               # 不足の埋め方（省略すると穴埋めしない）
        model=my_model,
        guardrails=("no_pii",),                         # 予測値の内容検査（opt-in）
    ),
)
catalog.validate(context=sample_ctx)

# 段階 1-2: 各ターン（1 呼び出し。予測は不足があるときだけ 1 回）
plans = await catalog.plan(query)

# 段階 3: 押下時
p = plans[i].apply({"seconds": 60})
# 利用者が決めた値だけを選り分けて自身のストアへ書き戻せる（書き戻しはアプリの責務）
remembered = {s.name: s.value for s in p.slots if s.from_user}
# 実行は明示的に書く（到達時ハンドオフ禁止を宣言しているなら apply_next_turn_policy の派生 registry から解決する）
result = await Runner.run(registry.get(p.action_agent), input=p.input_json)

# 段階 4: 実行後
next_agent = action_next_turn_agent(policy, result, registry)
```

### 機能要件の一覧

| FR | 追加するもの | 何のためか | 実装段 |
|---|---|---|---|
| FR-1 | `ActionSpec` / `ActionCatalog` / `param` | 実行できるアクションとパラメータの埋め方を 1 箇所へ宣言する | 第 1 段 |
| FR-2 | `parameters_model()` | 宣言した型を実行入力の検証・LLM スキーマ・UI フォームへ使い回す | 第 1 段 |
| FR-3 | `catalog.bind()` / `CandidateSource` / `LLMFiller` / `catalog.validate()` | 結線を関心事ごとの宣言型で 1 箇所へ集約し、宣言の不整合を起動時にまとめて落とす（押された瞬間に落とさない） | 第 1 段 |
| FR-4 | `ExecutableIntent` / `ExecutableSuggestion` | 候補生成方式（ルール / 学習 / LLM）に依存しない固定契約で候補を受ける | 第 1 段 |
| FR-5 | `catalog.plan()` / `ActionPlan` / `Slot` / `SlotState` / `Origin` | 何が埋まっていて何が足りないかを LLM 抜きで決定的に確定する | 第 1 段 |
| FR-6 | `catalog.plan(predict=True)` / `ParamUsage` / `PlanResult` | 不足パラメータを全候補まとめて 1 回の委譲で埋める | 第 2 段 |
| FR-7 | `on_invalid_response` / `on_invalid_slot` | 予測が失敗したとき「聞き直す」か「止める」かをアクション単位で選ぶ | 第 2 段 |
| FR-8 | `plan.apply` / `plan.input_json` / `plan.slots` / `plan.action_agent` | 確認結果と穴埋め入力を合流し、型検証済みの実行入力・実行先の名前・書き戻し材料を取り出す | 第 1 段 |
| FR-9 | `action_next_turn_agent` | 実行後の会話を判断能力のある窓口エージェントへ戻す | 第 2 段 |
| FR-10 | 公開窓口 | 既存 intent と同じ import 経路で使う | 各段 |

含まないもの: 候補の提示 UI・記憶の永続ストア・候補生成方式の実装・実行そのもの（実行は利用側の
`Runner.run`。build-don't-run 方針の維持）。

### 実装前の合意事項（すべて合意済み）

次の 4 項目は実装着手前のユーザー合意を要する判断であり、**すべて承認済み**である。
未決事項は残っていないため、実装者が再度確認する必要はない。

| # | 合意事項 | 状態 | 該当箇所 |
|---|---|---|---|
| 1 | FR-9 の `action_next_turn_agent` をコア直下 `next_turn.py` へ置くため、コア `__all__` が 36 件から 37 件へ増える（公開契約の変更） | **合意済み** | FR-9「配置」/ NFR-3 / 制約事項 |
| 2 | 実装を 2 段に分ける（第 1 段は LLM 呼び出し 0 で単体完結、第 2 段で予測・実行結線を追加） | **合意済み** | 制約事項（ビジネス制約） |
| 3 | `_adapters/intent.py` へ usage 取得用の新規関数を 1 件追加する（既存 `run_intent_prompt` は `str` のみ返し usage を扱えない） | **合意済み** | FR-6「委譲窓口」/ 影響範囲 |
| 4 | `runtime/intent` の公開窓口が 24 件から 40 件へ増えるため、`__all__` のメンバ集合を pin している既存テスト（`tests/runtime/intent/test_init_pep562_l1.py`）の期待値更新が不可避である。NFR-3 の計測基準「既存テストが 1 件も修正なしで通る」をこの 1 ファイルについて読み替える | **合意済み** | NFR-3 / FR-10 |

## 2. 機能要件

### FR-1: 実行可能アクションの宣言簿（`ActionSpec` / `ActionCatalog` / `param`）
- ユーザーストーリー: ライブラリ利用者として、システムが実行できるアクションと、そのパラメータの埋め方を 1 箇所へ宣言したい。なぜなら候補生成方式（ルール / 学習モデル / LLM）を差し替えても、下流のパラメータ契約と実行結線を変えずに済ませたいから。
- 受け入れ基準:
  - [ ] WHEN `ActionSpec(action_id, description, *, action_agent, label, parameters, prompt=(), prompt_vars={}, on_invalid_slot=None)` を宣言し `ActionCatalog.register(spec)` する THEN 宣言が保持され、同一 `action_id` の再登録は `ValueError` を送出する
  - [ ] WHEN `ActionSpec` / `ParameterSpec` / `ActionPlan` / `Slot` / `SlotSuggestion` / `ExecutableSuggestion` を生成する THEN いずれも frozen な pydantic `BaseModel` サブクラスとして成立する
  - [ ] WHEN `SlotSuggestion(value, level, rationale)` を `level` 省略で生成する THEN `level` は既存 `ConfidenceLevel.CERTAIN` となる（決定的に確定した値であることを示す既定）
  - [ ] WHEN 直列化（`model_dump()` / `model_json_schema()`）を行う THEN `ActionSpec` / `ParameterSpec` / `ActionPlan` / `Slot` / `SlotSuggestion` に限り成立を契約とする。`ExecutableSuggestion` は `IntentContext.run_context` に任意型が載る（`arbitrary_types_allowed=True`）ため直列化の成立を契約に含めない
  - [ ] WHEN `ActionCatalog` を生成する THEN plain な mutable クラスとして成立し（frozen 契約の対象外）、`register` / `names` / `get` / `bind` / `validate` / `plan` の 6 メソッドを公開する
  - [ ] WHEN `catalog.names()` を呼ぶ THEN 登録済み `action_id` を昇順のリストで返す
  - [ ] WHEN `catalog.get(action_id)` を未登録の `action_id` で呼ぶ THEN `KeyError` を送出する
  - [ ] IF `action_id` が空文字 / `str.isidentifier()` 偽 / `_` 始まり / Python 予約語 / `ActionCatalog` の公開メソッド名（`register` / `names` / `get` / `bind` / `validate` / `plan`）との衝突のいずれか THEN `ValueError` を送出する（規則は `agent_names._validate_attribute_name`（`agent_names.py:55-83`）と同型に 4 分岐をローカル再実装し、予約集合のみ差し替える。文書上の SoT は `ToolRegistry._validate_name` とし、`tool_registry.py` は変更しない）
  - [ ] WHEN `param(name, annotation, *, from_context=None, by_agent=False, prompt=None, description=None, default=PARAM_UNSET, max_suggestions=1, confirm=False, filled_by_candidate=False, extra={})` を呼ぶ THEN `ParameterSpec` を返す
  - [ ] IF `param` の `name` が `str.isidentifier()` 偽 THEN `ValueError` を送出する
  - [ ] IF 同一 `ActionSpec` 内に同名の `ParameterSpec` が 2 件以上ある THEN `ValueError` を送出する
  - [ ] IF `from_context` に `str` を渡す THEN 1 要素の tuple として正規化し、tuple を渡した場合は宣言順を保持する
  - [ ] IF `max_suggestions` が 1 未満 THEN `ValueError` を送出する
  - [ ] WHEN `ActionCatalog(*, prompt=(), prompt_vars={}, on_invalid_slot="skip")` を生成する THEN 全 `ActionSpec` 共通の既定として保持され、`ActionSpec` 側の同名フィールドは `prompt` / `prompt_vars` はマージ（アクション側を後に積む）、`on_invalid_slot` は上書きとして解決される
  - [ ] IF `on_invalid_slot` に `"error"` / `"skip"` 以外の値を `ActionCatalog` または `ActionSpec` の宣言時に渡す THEN `ValueError` を送出する（宣言時に落とし、予測段まで持ち越さない）
  - [ ] WHEN 宣言オブジェクトを生成する THEN `agents` / `openai` を import せず、`action_agent` はエージェント名の `str` として保持する

### FR-2: 宣言からの pydantic モデル生成（`parameters_model`）
- ユーザーストーリー: ライブラリ利用者として、宣言したパラメータの型を単一の出どころとして、実行入力の検証・LLM へのスキーマ提示・UI のフォーム生成に使い回したい。なぜなら型を複数箇所に書くと不一致が実行時まで発覚しないから。
- 受け入れ基準:
  - [ ] WHEN `spec.parameters_model()` を呼ぶ THEN 全 `ParameterSpec` を `pydantic.create_model` でフィールド化した frozen な `BaseModel` サブクラスを返し、フィールド型は `param` の第 2 引数、`Field(description=...)` は `param` の `description` を反映する
  - [ ] WHEN 同一 `ActionSpec` に対して `parameters_model()` を 2 回以上呼ぶ THEN 同一のクラスオブジェクトを返す（生成結果をキャッシュする）
  - [ ] WHEN 予測段が不足パラメータのスキーマモデルを組む THEN 当該 `ActionPlan` で `SlotState.NEEDS_AGENT` 状態にあるパラメータのみをフィールドに持つモデルとなり、既に解決済みのパラメータをフィールドに含めない（当該モデルの生成は `catalog.plan()` の内部実装であり公開 API として提供しない）
  - [ ] IF `param` の `max_suggestions` が 2 以上 THEN 当該スキーマモデルのフィールド型は `list[SlotSuggestion[T]]` となり、`max_suggestions` を上限として `model_json_schema()` に反映される
  - [ ] IF `param` の `max_suggestions` が 1 THEN 当該スキーマモデルのフィールド型は `SlotSuggestion[T]` となる
  - [ ] WHEN 当該スキーマモデルの parse 用派生を得る THEN 全フィールドが `X | None`（既定 `None`）となり、`max_suggestions` による `max_length` 制約を持たず、一部フィールドが欠落した JSON でも `model_validate_json` が成功する
  - [ ] IF `param` の第 2 引数に pydantic がフィールド型として扱えない型を渡す THEN `parameters_model()` の生成時に `pydantic.errors.PydanticSchemaGenerationError` を捕捉し、パラメータ名を添えた `ValueError` へ `raise ... from exc` で変換して送出する（当該例外は `RuntimeError` 派生であり素通しでは `ValueError` にならないため）

### FR-3: 結線と起動時検証（`ActionCatalog.bind` / `ActionCatalog.validate`）
- ユーザーストーリー: ライブラリ利用者として、実行に必要な結線を 1 箇所へ集約し、宣言の不整合をアプリ起動時にまとめて検出したい。なぜなら候補が押された瞬間に初めて落ちると、実行できないアクションを提示してしまうから。
- 実現手順（セグメント解決）: `PromptStore` のセグメント解決は private（`prompts.py:205` の `_load_segment`）で `all()` はコロン付きキーを除外するため、`catalog.validate()` は公開経路 `prompts.compose(layout=[<segment>], vars={})` を segment 単位で呼び、`PromptResolutionError`（`KeyError` 派生・`prompts.py:69`）を捕捉して未解決セグメント名を集約した `ValueError` へ `raise ... from exc` で変換する。プレースホルダ集合は `vars={}` の render 結果に残る `${x}` を `string.Template.get_identifiers()` で取得する（private API と `re` を使わない）。
- 受け入れ基準:
  - [ ] WHEN `catalog.bind(*, registry, prompts=None, guardrail_registry=None, candidates=None, llm_filler=None)` を呼ぶ THEN 結線対象を保持するだけで `Runner.run` を 1 回も呼ばず、`ActionSpec` の登録内容も変更しない
  - [ ] WHEN `CandidateSource(generator, *, context_builder=None, history_limit=None)` を生成する THEN frozen な pydantic `BaseModel` サブクラスとして成立し、`generator` / `context_builder` は不透明値として保持される
  - [ ] IF `CandidateSource` の `context_builder` と `history_limit` の双方に非 `None` を渡す THEN 生成時に `ValidationError` を送出する（`history_limit` は既定 builder 専用の便宜引数であることを固定する。既定値を `None` の sentinel とすることで「明示された 20」と「既定の 20」を区別できるようにする）
  - [ ] WHEN `LLMFiller(model, *, on_invalid_response="error", guardrails=())` を生成する THEN frozen な pydantic `BaseModel` サブクラスとして成立し、`model` は不透明値として保持される（利用者はエージェントの実体を渡さず `model` のみを渡す。ガードレールの解決簿は `bind(guardrail_registry=...)` 側が持つ）
  - [ ] IF `LLMFiller` の `guardrails` が非空で `catalog.bind()` の `guardrail_registry` が `None` THEN `catalog.validate()` が `ValueError` を送出する（登録名を解決する手段が無い結線を起動時に落とす）
  - [ ] IF `LLMFiller` の `on_invalid_response` に `"error"` / `"skip"` 以外の値を渡す THEN 生成時に `ValidationError` を送出する
  - [ ] IF `catalog.bind()` を呼ばずに `catalog.plan()` または `catalog.validate()` を呼ぶ THEN `RuntimeError` を送出する（結線漏れを無症状にしない）
  - [ ] IF `catalog.bind(candidates=None)` のまま `catalog.plan()` を呼ぶ THEN `RuntimeError` を送出する（候補生成は代替不能であり、候補 0 件を返すと「予測が効いていない」状態と区別できないため）
  - [ ] IF `catalog.bind(prompts=None)` のまま、いずれかの `ActionSpec` または `ActionCatalog` に `prompt` セグメント宣言がある状態で `catalog.validate()` を呼ぶ THEN `RuntimeError` を送出する（セグメント解決の検査が黙ってスキップされると起動時検証が空振りするため。セグメント宣言が 1 件も無ければ `prompts=None` でも正常に完了する）
  - [ ] WHEN `catalog.validate(*, context=None)` を呼び全宣言が整合している THEN 何も送出せず `None` を返す
  - [ ] WHEN `catalog.plan()` を `catalog.validate()` 未実行の状態で呼ぶ THEN 内部で `validate()` を 1 度だけ実行し、以降の `plan()` では再実行しない（検証は副作用を持たず冪等であり、LLM を 1 回も呼ばない）
  - [ ] IF いずれかの `ActionSpec.action_agent` が `registry.names()` に存在しない THEN 全違反を集約した単一の `KeyError` を送出する
  - [ ] IF `ActionSpec.label` のプレースホルダ集合（`string.Template` 構文 `${name}`）が宣言済みパラメータ名の集合に含まれない THEN 差分を列挙した `ValueError` を送出する
  - [ ] IF `prompt` に宣言したセグメント名が `prompts` で解決できない THEN 解決できないセグメント名を列挙した `ValueError` を送出する
  - [ ] IF `prompt` で解決したテンプレートのプレースホルダ集合が `prompt_vars` のキー集合に含まれない THEN 差分を列挙した `ValueError` を送出する（判定対象は `ActionCatalog` 由来分をマージ後の `prompt` / `prompt_vars` とする）
  - [ ] IF `prompt_vars` に宣言したキーがどのテンプレートのプレースホルダにも現れない THEN 効果のない宣言として `ValueError` を送出する（判定対象はカタログ全体のテンプレート集合とし、`ActionCatalog.prompt_vars` のキーはいずれかの `ActionSpec` のテンプレートで使われていれば足りる）
  - [ ] IF カタログ全体で同一の `prompt_vars` キーが異なるパスへ宣言されている THEN `ValueError` を送出する（同一キー・同一パスの重複は許容する）
  - [ ] WHEN `context` に代表インスタンスを渡す THEN 全 `from_context` のパスと全 `prompt_vars` のパスについて「mapping ならキー、それ以外は属性、`.` で分割して再帰」の規則で構造的に解決できるかを検査し、解決できないパスを列挙した `ValueError` を送出する（解決結果の値が `None` であることは違反として扱わない）
  - [ ] IF `context` を省略する THEN パスの構造検査を行わず、他の検査のみ実施する
  - [ ] IF あるパラメータが `from_context` / `by_agent` / `default` のいずれも宣言せず、かつ `filled_by_candidate=False`（既定）THEN 「候補の parameters または利用者入力のみで埋まるパラメータ」として WARNING ログを 1 行出力し、`ValueError` は送出しない（値の解決順の第 1 優先は候補の値であり、宣言からは導けないため）
  - [ ] IF `filled_by_candidate=True` を宣言する THEN 当該 WARNING を出力しない（候補が常に値を載せる設計であることの明示宣言。警告の常態化を避ける）
  - [ ] IF `ActionSpec` に `by_agent=True` のパラメータが 1 件も無いのに `prompt` / `prompt_vars` を宣言している THEN 効果のない宣言として `ValueError` を送出する（判定対象は当該 `ActionSpec` 自身の宣言に限り、`ActionCatalog` 由来のマージ分は対象としない）
  - [ ] IF `LLMFiller.guardrails` に宣言した登録名が `bind(guardrail_registry=...)` で解決できない THEN 解決できない名前を列挙した `KeyError` を送出する（押される前に検出するため、実行時ではなく `catalog.validate()` で落とす）

### FR-4: パラメータ付き候補型と候補生成の固定契約（`ExecutableIntent` / `ExecutableSuggestion`）
- ユーザーストーリー: ライブラリ利用者として、候補生成方式に依存しない固定契約で候補を受け取りたい。なぜならルールベース・学習モデル・LLM のどれで生成しても、下流の空欄確定と実行を作り直したくないから。
- 受け入れ基準:
  - [ ] WHEN `ExecutableIntent(action_id=..., parameters={...}, level=..., source=..., rationale=None)` を生成する THEN 既存 `IntentCandidate` のサブクラスとして frozen に保持され、必須フィールド `text` は `model_validator(mode="before")` で `action_id` から自動補完される（利用者は `text` を渡さない）
  - [ ] IF `ExecutableIntent` の生成時に `text` と `action_id` の双方を明示し両者が不一致 THEN `ValueError` を送出する
  - [ ] WHEN `ExecutableIntent` を `IntentPrediction.candidates` へ格納する THEN 検証後もサブクラスのインスタンスと追加フィールドが保持される
  - [ ] IF `parameters` を省略する THEN 空の Mapping として成立する
  - [ ] WHEN `level` を指定する THEN 既存 `ConfidenceLevel` 5 段階に限定される
  - [ ] WHEN `source` を指定する THEN 候補の生成系統を `str` として保持し、ライブラリは値の集合を検証しない
  - [ ] WHEN `await catalog.plan(query: IntentQuery[Any])` を呼ぶ THEN `bind(candidates=...)` の `CandidateSource.context_builder` が `None` なら `DefaultContextBuilder(history_limit=history_limit if history_limit is not None else 20)`、非 `None` なら当該 `ContextBuilder` を使って `IntentContext` を組み、`generator.generate(context)` を 1 回だけ呼ぶ
  - [ ] WHEN `await catalog.plan(query, detail=True)` を呼ぶ THEN `PlanResult(plans=..., suggestion=..., usage=...)` を返し、`suggestion` は `generator.generate()` が返した `IntentPrediction` の `report` / `metadata` を素通しで保持する `ExecutableSuggestion` である（情報を捨てない）
  - [ ] IF `generator` が返した候補に `catalog.names()` へ未登録の `action_id` が含まれる、または候補が `ExecutableIntent` のインスタンスでない THEN 当該候補を除外し、除外件数と `action_id`（または `repr(text)`）を含む WARNING ログを 1 行出力する
  - [ ] IF `generator` が LLM を使わない実装（既存 `CandidateGenerator` Protocol を満たすルールベース / ML 実装）である THEN `await catalog.plan(query, predict=False)` は LLM 実行アダプタを 1 回も呼ばない
  - [ ] IF `generator` が例外を送出する THEN 例外を握り潰さず呼び出し元へ伝播する

### FR-5: 空欄の決定的確定（`catalog.plan` / `ActionPlan` / `Slot`）
- ユーザーストーリー: ライブラリ利用者として、候補ごとに何が埋まっていて何が足りないかを LLM を呼ばずに確定させたい。なぜなら不足項目だけを選択式で埋めさせる UI を組みたく、判定が非決定的だと同じ状況で UI の形が変わるから。
- 受け入れ基準:
  - [ ] WHEN `await catalog.plan(query, predict=False)` を呼ぶ THEN 候補と同順・同数の `ActionPlan` の tuple を返し、空欄の確定段は LLM 実行アダプタ・ネットワーク・環境変数を一切参照せず、同一の候補列と同一の `run_context` に対し常に同一結果を返す
  - [ ] WHEN `ActionPlan` を生成する THEN 対応する `ActionSpec` を `spec`、実行先エージェント名を `action_agent`、マージ解決済みの既定を `resolved_prompt` / `resolved_prompt_vars` / `resolved_on_invalid_slot` として保持し、これら 5 フィールドはすべて `Field(exclude=True)` により `model_dump()` から除外される（実行先エージェントの**実体は保持しない**）
  - [ ] WHEN 各パラメータの値を決める THEN 「候補の `parameters` にある値」「`from_context` のパス解決（宣言順に試し最初の非 `None`）」「`by_agent` による予測（この段では未実施）」「`default`」「利用者入力」の優先順で解決する
  - [ ] IF あるパラメータが値を得られ `confirm=False` THEN 当該スロットは `Slot(name=..., state=SlotState.RESOLVED, value=..., origin=..., detail=...)` となる
  - [ ] IF あるパラメータが値を得られ `confirm=True` THEN 当該スロットは `Slot(name=..., state=SlotState.NEEDS_CONFIRMATION, suggestions=..., origin=..., detail=...)` となり、`suggestions` は解決した値 1 件の `SlotSuggestion` の tuple となる（`level` は宣言側の既定値）
  - [ ] IF あるパラメータが値を得られず `by_agent=True` THEN 当該スロットは `Slot(name=..., state=SlotState.NEEDS_AGENT)` となり、`origin` と `value` は `None` である
  - [ ] IF あるパラメータが値を得られず `by_agent=False` THEN 当該スロットは `Slot(name=..., state=SlotState.NEEDS_USER, suggestions=())` となり、`origin` と `value` は `None` である
  - [ ] WHEN スロットを生成する THEN 値の出どころを `origin: Origin | None`（`Origin` は `CANDIDATE` / `RUN_CONTEXT` / `DEFAULT` / `AGENT` / `USER_CONFIRMED` / `USER_INPUT` の 6 値を持つ `StrEnum`。値の文字列は接頭辞なしの snake_case で揃え、`model_dump(mode="json")` に現れる公開契約とする）と `detail: str | None`（`RUN_CONTEXT` なら解決に成功したパス、`AGENT` なら予測エージェント名、それ以外は `None`）の 2 フィールドで記録する
  - [ ] WHEN `Slot` を参照する THEN `from_user`（`origin` が `USER_INPUT` または `USER_CONFIRMED` か）が導出され、`origin` が `None` のスロットでも例外を送出せず `False` を返す（状態の判定は `state` が唯一の表現であり、`state` と同値の導出プロパティは持たない）
  - [ ] IF 次のいずれかに該当する THEN `Slot` の生成時に `ValidationError` を送出する（1 型化により構造的には作れてしまう不整合な組み合わせを、状態とフィールドの整合を検査する validator で拒否する）: (1) `state` が `NEEDS_AGENT` または `NEEDS_USER` で `origin` または `value` が非 `None`、(2) `state` が `NEEDS_AGENT` で `suggestions` が非空、(3) `state` が `NEEDS_CONFIRMATION` で `value` が非 `None`、(4) `state` が `NEEDS_CONFIRMATION` で `suggestions` が空（`apply` が `USER_CONFIRMED` を判別できず `USER_INPUT` へ誤分類するため）、(5) `state` が `RESOLVED` で `origin` が `None`（`from_user` が常に偽となり利用者が決めた値が書き戻し対象から漏れるため）、(6) `detail` が非 `None` で `origin` が `RUN_CONTEXT` / `AGENT` のいずれでもない
  - [ ] WHEN `state` が `RESOLVED` で `value` が `None` の `Slot` を生成する THEN `ValidationError` を送出しない（`param(..., default=None)` を明示宣言した場合に正当に発生する組み合わせであり、`value` の `None` は「未解決」ではなく「値が `None` であること」を意味する。未解決かどうかは `state` が持つ）
  - [ ] WHEN `ActionPlan` を生成する THEN `pending`（`NEEDS_CONFIRMATION` と `NEEDS_USER` のスロットを宣言順に並べた tuple）と `ready`（全スロットが `RESOLVED`）が導出される（「どの状態が確定か」「どの状態を利用者に聞くべきか」はライブラリ側の知識であるため導出して提供し、`slots` から一意に書ける判定は導出プロパティとして持たない）
  - [ ] WHEN `ActionPlan` を `model_dump()` する THEN `IntentContext` と `run_context` と `spec` と `action_agent` と `resolved_prompt` / `resolved_prompt_vars` / `resolved_on_invalid_slot` を一切含まない
  - [ ] WHEN `plan.label` を参照する THEN `ActionSpec.label` を `string.Template` 構文（`${name}`）とみなし、解決済みスロットの値と「未解決スロット名 -> `"…"`」を合わせた mapping で `Template(label).substitute(...)` により render した文字列を返す（`re` は使わない）

### FR-6: 不足パラメータの予測（`catalog.plan(predict=True)` / `ParamUsage`）
- ユーザーストーリー: ライブラリ利用者として、提示する全候補の不足パラメータを 1 回の呼び出しでまとめて埋めたい。なぜなら値が埋まっていないボタンでは「押すだけで実行」が成立せず、候補ごとに呼び出すと従量課金のコストが候補数に比例するから。
- 委譲窓口: 既存 `_adapters/intent.py` の `run_intent_prompt` は `model` を受け `str` のみを返し usage を扱えないため、`catalog.plan()` の予測段用に `_adapters/intent.py` へ「上位層が宣言した `AgentSpec` を受け、`AgentBuilder` で実体化して 1 回走らせ、応答 `str` と usage（モデル呼び出し回数・input/output トークン）を返す」新規関数を 1 件追加する。上位層は `agents` を import せず `AgentSpec(name=<lib の固定名>, instructions=<合成済み>, model=<LLMFiller.model>)` を宣言するだけであり、SDK `Agent` の属性へは触らない（NFR-1）。予測エージェントは利用者の `AgentRegistry` へ登録されず `plan()` の内部でのみ使われるため、「業務エージェントとは別に置き `session` を渡さない」制約が構造的に保証される。
- 受け入れ基準:
  - [ ] WHEN `await catalog.plan(query)`（`predict=True` が既定）を呼び `NEEDS_AGENT` のスロットが 1 件以上あり `llm_filler` が結線されている THEN `Runner.run` を **1 回だけ**呼び、`detail=True` のとき `PlanResult(plans=..., suggestion=..., usage=...)` を返す（`model` と `on_invalid_response` は `LLMFiller`、`prompts` は `bind` で注入する）
  - [ ] IF 全 `ActionPlan` に `NEEDS_AGENT` のスロットが 1 件も無い、または `predict=False` を渡した THEN `Runner.run` を 0 回呼び、決定的段の `plans` と `runs=0` / `model_calls=0` / `candidates=0` / `input_tokens=None` / `output_tokens=None` の `ParamUsage` を返す
  - [ ] IF `catalog.bind(llm_filler=None)` のまま `NEEDS_AGENT` のスロットがあり `predict=True`（既定）で `catalog.plan()` を呼ぶ THEN 例外を送出せず予測段をスキップし、決定的段の `plans` と `runs=0` の `ParamUsage` を返す（`LLMFiller` を渡さないことが「穴埋め経路を持たない」という利用者の明示的な意思表示であり、第 1 段を単体でリリースした構成でも既定引数のまま `plan(query)` を呼べるようにするため。予測結線の不在は `usage.runs == 0` かつ `plan.slots` に `SlotState.NEEDS_AGENT` のスロットが残るという観測可能な組み合わせで判別でき、`plan.ready` は `False` のままなので誤実行は起きない）
  - [ ] WHEN プロンプトを合成する THEN 各 `ActionPlan` の `resolved_prompt`（`ActionCatalog.prompt` と `ActionSpec.prompt` のマージ結果）と、`NEEDS_AGENT` 状態のパラメータの `param.prompt` のセグメントのみを積み、同一セグメント名が複数の候補・パラメータから要求された場合は 1 回だけ積む
  - [ ] WHEN プロンプトを合成する THEN `resolved_prompt_vars` の各キーを `context.run_context` からパス解決した値で `${var}` を置換し、`context.history_items` を会話部分として含め、複合応答モデルの `model_json_schema()` と JSON のみを返す旨の制約を出力形式として含める
  - [ ] WHEN 複合応答モデルを組む THEN `plans` の位置（0 始まり）から `candidate_<index>` をフィールド名とし、各フィールドの型は当該 `ActionPlan` のスキーマモデルの parse 用派生とする（同一 `action_id` の候補が複数含まれても衝突しない）。プロンプトには `candidate_<index>` と `action_id` / `label` の対応表を含める
  - [ ] IF `NEEDS_AGENT` 状態のパラメータが 1 つの候補に 5 件あり候補が 3 件ある THEN `Runner.run` の呼び出し回数は 1 回である（不足件数・候補数に比例しない）
  - [ ] WHEN `Runner.run` を呼ぶ THEN `session` を渡さず（渡す口を公開しない）、ライブラリ内部定数の `max_turns`（値 1）を指定し、`context.run_context` を `context` 引数として渡す
  - [ ] WHEN 応答を受け取る THEN raw `str` として受け、既存 `_strip_code_fence` でコードフェンスを剥がしたうえで全フィールド Optional の parse モデルで `model_validate_json` する（strict structured output は使用しない）
  - [ ] IF 予測で値を得たパラメータの `confirm` が `True` THEN 当該スロットの `state` は `SlotState.NEEDS_CONFIRMATION`、`False` THEN `SlotState.RESOLVED` となる
  - [ ] WHEN `origin` を記録する THEN `origin=Origin.AGENT` とし、`detail` にはライブラリが予測エージェントへ宣言した固定名を入れる（SDK `Agent` の属性を上位層から読まない。利用者はエージェント名を指定しない）
  - [ ] IF `param.max_suggestions` が 2 以上で複数の候補値が返る THEN `suggestions` は `SlotSuggestion(value, level, rationale)` の tuple となり、`ConfidenceLevel` 降順の stable sort で並び、`max_suggestions` を超える分は切り捨てられる
  - [ ] WHEN `usage` を返す THEN `ParamUsage(runs, model_calls, candidates, input_tokens, output_tokens)` を返し、`runs` は `Runner.run` の呼び出し回数、`model_calls` は SDK 応答の実測件数（usage の内容に依存しない）、`candidates` は予測対象として送った候補件数（`NEEDS_AGENT` を 1 件以上持つ `ActionPlan` の件数）となる（`runs` / `model_calls` は `_adapters/intent.py` の新規窓口が SDK `RunResult` から抽出した値であり、上位層は SDK の属性へ触らない）
  - [ ] IF 全応答の usage が `requests == 0` かつ `total_tokens == 0` THEN usage 未取得とみなし `input_tokens` / `output_tokens` は `None` を返す（`model_calls` は応答件数のまま。SDK の `usage` は非 Optional で既定値 0 のため、0 と「未取得」の区別はこの判定規則で行う）
  - [ ] WHEN `LLMFiller(guardrails=(...))` を宣言し `catalog.bind(guardrail_registry=...)` で解決簿を結線する THEN 予測エージェントの実行に当該ガードレールが適用され、`GuardrailProvider` が答える適用境界に応じて入力側 / 出力側へ装着される
  - [ ] IF `LLMFiller.guardrails` が空（既定）THEN 予測エージェントにガードレールは 1 件も装着されない
  - [ ] IF 予測エージェントの実行中にガードレールが発火する THEN SDK の例外（入力境界 / 出力境界で型が異なる）を握り潰さず呼び出し元へ伝播し、`on_invalid_response="skip"` による `default` / `NEEDS_USER` への後退を適用しない（ガードレール発火は「応答が壊れている」ではなく「危険な内容を検出した」という安全事象であり、既定値で実行を続けると利用者がガードレールを宣言した意図に反するため。後退が必要な場合は `runtime/resilience` の `failsafe_call` で `catalog.plan()` を包む）
  - [ ] WHEN 予測エージェントを構築する THEN 予測エージェント専用の `AgentRegistry` を用い、`catalog.bind()` の `guardrail_registry` を `GuardrailProvider` として注入して登録名を解決する（利用者の業務 `AgentRegistry` を参照しない）
  - [ ] IF `max_turns` を超過する THEN SDK の例外を握り潰さず呼び出し元へ伝播する
  - [ ] WHEN 予測エージェントを走らせる THEN `model_calls` は 1 となる（予測エージェントはツールを持たない構成でライブラリが宣言するため 1 ターンで完了する。`max_turns` は内部定数 1 であり公開しない。将来ツールを持たせる場合は内部で引き上げ、公開契約は変えない）

### FR-7: 応答不正・値欠落の扱い（`on_invalid_response` / `on_invalid_slot`）
- ユーザーストーリー: ライブラリ利用者として、予測が失敗したときに「利用者へ聞く側に落とす」か「その場で落とす」かをアクション単位で選びたい。なぜなら振込金額のように外したら止めたい項目と、負荷試験の秒数のように聞き直せば済む項目が同じアプリに混在するから。
- 受け入れ基準:
  - [ ] IF 応答が `_strip_code_fence` 適用後も JSON として parse できず `on_invalid_response="error"`（既定）THEN 例外を送出する
  - [ ] IF 応答が parse できず `on_invalid_response="skip"` THEN 全 `NEEDS_AGENT` スロットについて、`default` を宣言していれば `confirm=False` なら `Slot(state=SlotState.RESOLVED, origin=Origin.DEFAULT)` / `confirm=True` なら `Slot(state=SlotState.NEEDS_CONFIRMATION, origin=Origin.DEFAULT)`、`default` を宣言していなければ `Slot(state=SlotState.NEEDS_USER)` へ遷移させて処理を続行する（`NEEDS_CONFIRMATION` の `suggestions` は `default` 値 1 件の `SlotSuggestion` の tuple とし、`level` は宣言側の既定値を用いる）
  - [ ] IF parse は成功したが個別フィールドが欠落または宣言型に合わず `on_invalid_slot="skip"`（既定）THEN 当該スロットのみ、`default` を宣言していれば `confirm=False` なら `Slot(state=SlotState.RESOLVED, origin=Origin.DEFAULT)` / `confirm=True` なら `Slot(state=SlotState.NEEDS_CONFIRMATION, origin=Origin.DEFAULT)`、宣言していなければ `Slot(state=SlotState.NEEDS_USER)` へ遷移させ、同一応答内の他スロットの値は保持する（`NEEDS_CONFIRMATION` の `suggestions` は `default` 値 1 件の `SlotSuggestion` の tuple とする）
  - [ ] IF 個別フィールドが欠落または型不一致で `on_invalid_slot="error"` THEN 例外を送出する
  - [ ] WHEN `on_invalid_slot` を `ActionSpec` で宣言する THEN 当該アクションの全パラメータへ適用され、宣言しない場合は `ActionCatalog` の既定が適用される（`ActionPlan.resolved_on_invalid_slot` から読む）
  - [ ] WHEN `on_invalid_response` を指定する THEN `LLMFiller` のフィールドとして宣言時に検証される（FR-3 の `LLMFiller` の AC を参照。`on_invalid_slot` の値検証は FR-1 の宣言時に行う）

### FR-8: 利用者入力の合流と実行入力の組み立て（`plan.apply` / `plan.input_json` / `plan.slots` / `plan.action_agent`）
- ユーザーストーリー: ライブラリ利用者として、確認結果と穴埋め入力を計画へ合流させ、実行に渡す入力を型検証済みの形で取り出したい。なぜなら不完全なパラメータで実行してしまう事故を防ぎたいから。
- 受け入れ基準:
  - [ ] WHEN `plan.apply(answers)` を呼ぶ THEN 新しい `ActionPlan` を返し（元のインスタンスを変更しない）、`answers` のキーに対応するスロットを `SlotState.RESOLVED` へ遷移させる
  - [ ] WHEN `plan.apply(answers)` を呼ぶ THEN 対象スロットが `SlotState.NEEDS_CONFIRMATION` で `answers` の値がその `suggestions` のいずれかの値と等しい場合は `origin=Origin.USER_CONFIRMED`、それ以外（`NEEDS_USER` / 値が `suggestions` に無い）は `origin=Origin.USER_INPUT` とする
  - [ ] IF `answers` に宣言済みパラメータ名でないキーが含まれる THEN 未知キー名を列挙した `ValueError` を送出する
  - [ ] IF `answers` のキーが既に `SlotState.RESOLVED` のスロットを指す THEN 当該キーを列挙した `ValueError` を送出する（確定済み値の黙示的な上書きを禁止する）
  - [ ] IF `answers` の値が宣言型に合わない THEN 当該パラメータ単体を `TypeAdapter(<param の annotation>)` で検証し `ValidationError` を送出する（未解決スロットが残る段では全件モデルによる検証を行わない）
  - [ ] IF `plan.ready` が `False` の状態で `plan.input_json` を参照する THEN 未解決のスロット名を列挙した `ValueError` を送出する
  - [ ] WHEN `plan.ready` が `True` の状態で `plan.input_json` を参照する THEN 全スロットの値を `spec.parameters_model()` で型検証したうえで `model_dump_json()` した結果（`str`）を返す（型付きインスタンスが必要な利用者は公開されている `spec.parameters_model()` から自分で組める）
  - [ ] WHEN `plan.action_agent` を参照する THEN `ActionSpec.action_agent` と同じ実行先エージェント名（`str`）を返す（利用者が `plan.spec` を経由せずに実行先の名前を取れるようにするため）
  - [ ] WHEN `ActionPlan` を生成する THEN `AgentRegistry` を参照するフィールド・プロパティを一切持たず、実行先エージェントの**実体を解決しない**（実行は利用者が `Runner.run(registry.get(plan.action_agent), input=plan.input_json)` と明示的に書く。どの registry から解決したかが呼び出し箇所に現れることで、派生 registry の取り違えが隠れない）
  - [ ] WHEN 実行する THEN 到達時ハンドオフ禁止（`no_handoff_on_arrival`）を宣言している構成では `apply_next_turn_policy` が返す**派生 registry** から実行先を解決する（元 registry には到達記録の前置合成も `is_enabled` ゲートも設置されていないため、元 registry から解決すると宣言した禁止が無症状で効かない）
  - [ ] WHEN `plan.slots` を参照する THEN 宣言順のスロット列を返し、各スロットから `name` / `state` / 解決済みなら `value` と `origin` / `detail` を読み取れる（利用者が `slot.from_user` で値の出どころを判別し、利用者が入力・確認したスロットのみを自身のストアへ書き戻せるようにするため。書き戻し自体はライブラリの責務ではなく、次ターンでの再利用は `from_context` の宣言で表現する）

### FR-9: 実行後の次ターン開始エージェント解決（`action_next_turn_agent`）
- ユーザーストーリー: ライブラリ利用者として、アクション実行後に会話を設計上の窓口へ戻したい。なぜなら実行専用エージェントから次ターンが始まると、利用者が判断能力を持たないエージェントと会話し続けることになるから。
- 既存資産との関係: コアの `next_turn_agent(policy, result, registry)` は「ハンドオフ遷移が観測されるときの上書き」と「上書きなし時の `last_agent` 継続」を既に満たす（`next_turn.py:240-267`）。本 FR が追加するのは「ハンドオフ遷移が観測されないとき（アクションからの直接起動）に包括ルールを適用する」経路のみであり、他の分岐は `next_turn_agent` へ委譲する（挙動の二重実装を作らない）。
- 配置: ADR 0014 Decision 3（宣言 + 純関数はコア直下 `next_turn.py`）に従いコア直下へ追加する。`await` も `Runner` 参照も持たず intent 固有の型も扱わないため同 Decision の根拠にそのまま該当する。この場合コア `__all__` へ 1 件追加されるため、`__all__` のメンバ集合変更として実装前にユーザー合意を取る（NFR-3 の対象）。
- 受け入れ基準:
  - [ ] WHEN `action_next_turn_agent(policy, result, registry)` を呼び、当該ターンにハンドオフ遷移が観測される THEN 既存 `next_turn_agent(policy, result, registry)` へ委譲した結果を返す（選定されたルールが `next_agent` を持たない場合の `result.last_agent` への後退もこの委譲に含意される）
  - [ ] IF 当該ターンにハンドオフ遷移が観測されず（アクションからの直接起動で実行専用エージェントが完結した場合）最終回答者名が `policy.rules` のキーに一致し、当該エントリに包括ルール（`source` を持たないルール）があり、かつ当該ルールが `next_agent` を持つ THEN その `next_agent` を `registry.get()` で解決した値を返す（包括ルールの選定は `next_turn.py` の既存 `_select_rule` を `source` 不一致の値で呼び `source is None` のルールへ倒す形で再利用し、選定規則を二重実装しない。渡す値は registry へ登録され得ない文字列（例: 制御文字を含む固定 sentinel）とする。`NextTurnRule.source` の検証は非空 str のみ（`next_turn.py:34-47`）で識別子制約が無いため、宣言済みの任意の `source` と衝突しないことは型ではなくテストで固定する）
  - [ ] IF ハンドオフ遷移が観測されず、包括ルールはあるが `next_agent` を持たない（`no_handoff_on_arrival` のみのルール）THEN `result.last_agent` をそのまま返す（既存 `resolve_next_agent` の「上書きなし」と同じ帰結にする）
  - [ ] IF ハンドオフ遷移が観測されず、最終回答者名はキーに一致するが当該エントリに包括ルールが無い（`source` 限定ルールのみ）THEN `result.last_agent` をそのまま返す
  - [ ] IF 最終回答者名が `policy.rules` のキーに一致しない THEN `result.last_agent` をそのまま返す
  - [ ] IF `result.last_agent` も取得できない THEN `None` を返す
  - [ ] WHEN 本関数を呼ぶ THEN 既存 `next_turn.py` の公開シンボルの実装と挙動を変更しない（純追加）
  - [ ] WHEN 本関数を呼ぶ THEN 副作用を持たず、同一入力に対し常に同一結果を返す

### FR-10: 公開窓口
- ユーザーストーリー: ライブラリ利用者として、追加シンボルを既存 intent と同じ窓口から import したい。なぜなら import 経路を覚え直したくないから。
- 受け入れ基準:
  - [ ] WHEN `from oai_agentspec.runtime.intent import ...` を実行する THEN 本要件で追加した全公開シンボル（`action_next_turn_agent` を除く 16 件）が取得でき、取得値は module 属性へキャッシュされる（既存の PEP 562 遅延再エクスポート方式に従う）
  - [ ] WHEN `import oai_agentspec` を実行する THEN コア `__all__` は現行 36 件に `action_next_turn_agent` の 1 件を加えた 37 件となり、既存 36 件のメンバ集合は変更されない
  - [ ] WHEN `import oai_agentspec.runtime.intent` を intent extra 未導入環境で実行する THEN 窓口の import 自体は成功する
  - [ ] WHEN 既存 `runtime/intent` の 24 シンボルを使う THEN 振る舞いは変更前と一致する

## 3. 非機能要件

### NFR-1: 保守性（SDK 隔離）
- 要件: `agents` / `openai` への import は `_adapters/` 配下に限定し、本要件で追加するモジュールからは行わない。
- 計測基準: `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` の結果が空であること。

### NFR-2: 保守性（単方向依存）
- 要件: 追加モジュールからの参照は `runtime/intent` 内および上向きのコア（`_adapters` / `registry` / `tool_registry` / `next_turn` / `prompts` / `constants` / `_validation`）に限定し、コアから `runtime` への新規依存辺を作らない（`runtime/deterministic` への参照は関数実行エージェントの分離に伴い本要件の範囲から外れた）。「追加モジュール」には本要件で `_adapters/intent.py` へ追加する新規関数（FR-6 の委譲窓口）と `next_turn.py` の追加関数（FR-9）を含む。
- 計測基準: 本要件で追加するファイルの import 文が上記の範囲のみを指すこと（`grep -nE "^(from|import)" <追加ファイル>` を目視突合）。かつコア層のファイルへ `runtime` への新規 import を 1 件も追加しないこと（`git diff main -- 'src/oai_agentspec/*.py' 'src/oai_agentspec/_adapters/' | grep -E "^\+.*runtime\."` が空）。`exceptions.py:41-42` の既存 `runtime` 参照（例外階層の集約）は本要件のスコープ外である。

### NFR-3: 保守性（既存公開契約の不変）
- 要件: 既存 `runtime/intent` の 24 シンボルとコア既存 36 シンボルの振る舞いを変更しない（純追加）。`runtime/intent` の `__all__` へ 16 件、コア `__all__` へ `action_next_turn_agent` の 1 件のみ追加し、この追加は実装前にユーザー合意を取る。
- 計測基準: `uv run python -c "import oai_agentspec as m; assert all(hasattr(m,s) for s in m.__all__)"` が成功し、コア `__all__` のメンバ集合が「変更前の 36 件 + `action_next_turn_agent`」と完全一致すること。既存テストは、公開窓口の `__all__` メンバ集合と件数を pin している `tests/runtime/intent/test_init_pep562_l1.py` の期待値更新を除き、1 件も修正なしで通ること。

### NFR-4: 保守性（依存非膨張）
- 要件: `intent` extra の依存を `pydantic>=2` のみに保つ。
- 計測基準: `pyproject.toml` の `intent` extra の列挙が変更前と同一で、`uv.lock` に新規の外部パッケージが 0 件増えること（`StrEnum` は標準ライブラリ `enum`・Python 3.11+ のため依存を増やさない）。

### NFR-5: 性能（呼び出し回数と計算量）
- 要件: 1 ターンのパラメータ予測は `Runner.run` を 1 回に抑え、候補数・不足パラメータ数に比例させない。決定的な段はモデル呼び出しを行わない。プロンプト合成は同一セグメントを重複して積まない。
- 計測基準: 候補 3 件・不足 5 件の入力に対し `catalog.plan()` の `Runner.run` 呼び出しが 1 回であることをテストで固定する。不足 0 件で 0 回、`predict=False` で 0 回であることを固定する。`catalog.plan(predict=False)` / `plan.apply` / `plan.input_json` / `action_next_turn_agent` がモデル実行アダプタを 0 回呼ぶことを固定する。同一セグメントを 2 候補が要求したとき合成結果に本文が 1 回だけ現れることを固定する。

### NFR-6: セキュリティ（非信頼入力と個人情報の扱い）
- 要件: 予測結果は宣言済み `action_id` の allowlist を通らない候補を除外する。既に解決済みのパラメータは予測対象のスキーマに含めず、モデルが上書きできないようにする。`ActionPlan` は `run_context` を保持しない。エンドユーザー入力を system 指示として注入する経路を追加しない。**予測された値は型検証を通っても内容が危険でありうるため、利用者が内容検査を挟める口（`LLMFiller.guardrails`）を用意する**（既存 `runtime/guardrails` の登録名参照を再利用し、新規のガードレール機構を作らない）。
- 計測基準: 未登録 `action_id` を含む候補が除外され WARNING が 1 行出ることをテストで固定する。`from_context` で解決したスロットが予測段のスキーマモデルのフィールドに現れないことを固定する。`ActionPlan.model_dump()` の結果に `run_context` 由来のキーと `spec` が含まれないことを固定する。
- 計測基準（プロンプト注入経路）: プロンプト合成に渡る値の出どころが `ActionCatalog.prompt` / `ActionSpec.prompt` / `param.prompt`（開発者宣言）と `prompt_vars` のパス解決値・`context.history_items`（会話部分）のみであることをテストで固定する（`IntentContext.utterance` を system 指示部へ連結する経路を持たない）。
- 計測基準（予測値の内容検査）: 予測された値は型検証を通っても内容が危険でありうるため、`LLMFiller.guardrails` を宣言したときに予測エージェントの実行へガードレールが装着されること、宣言しないときに装着されないこと、発火時に例外が伝播すること（後退しないこと）をテストで固定する。
- 計測基準（`import re` の根拠）: LLM 由来の非信頼入力を線形時間で処理し ReDoS（CWE-1333）を避けるため、既存 `_strip_code_fence` と同じ方針で追加モジュールに `import re` を 0 件に保つ（`grep -rn "import re" <追加ファイル>` が空）。

### NFR-7: 保守性（テストとリント）
- 要件: 追加コードを含めてカバレッジ基準を満たす。
- 計測基準: `uv run pytest` が `fail_under = 80` を満たして成功し、`uv run ruff check src/ tests/` と `uv run ruff format src/ tests/` が差分なしで通ること。

## 4. 制約事項
- 技術的制約: build-don't-run を維持する。候補選択後のアクション実行（`Runner.run`）は利用者が呼ぶ。`catalog.plan()` の予測段はパラメータ予測エージェントを 1 回走らせる薄い結線であり、独自の実行ループ・再試行・`Runner` 代行を持たない。押下後の実行まで面倒を見る実行ヘルパーは提供しない。この逸脱は ADR で境界を明記する（既存の `fit_ml_estimator` / `failsafe_call` / observability の 3 例と同型の扱い。ADR 番号は現状の最大 0025 の次から採番する）。
- 技術的制約: 予測エージェントへツールを持たせる宣言口（`AgentSpec` を直接渡す逃げ道）と `max_turns` を公開しない。予測エージェントはツールを持たない構成で 1 ターン完了するため `max_turns` はライブラリ内部定数 1 とし、将来ツールを持たせる判断をした場合も内部で引き上げて公開契約を変えない。
- 技術的制約: 応答不正・値欠落に対する再試行・プロンプト再構成・モデル切り替えを行わない。例外は握り潰さず呼び出し元へ伝播する。再試行・タイムアウト・フォールバックが必要な場合は `runtime/resilience`（`ModelRetryPolicy` / `failsafe_call`）に委ねる。
- 技術的制約: strict structured output（`output_type` によるスキーマ強制）を採用しない。生成速度への影響が大きいため、スキーマはプロンプトへ提示し、応答は raw `str` を手動 parse する（既存 `runtime/intent` の同一判断に従う）。
- 技術的制約: プロンプト非同梱を維持する。プロンプト本文は利用側の `PromptStore` に置き、宣言はセグメント名と変数マッピングのみを持つ。合成が生成する固定文は出力形式・JSON のみ制約に限る。
- 技術的制約: 環境変数を参照しない（env 参照は `runtime/cli` 境界に閉じる既存規約に従う）。
- 技術的制約: 宣言は明示コンストラクタ（`ActionSpec` / `param`）で行い、関数のシグネチャからパラメータ宣言を導く decorator は提供しない。既存 `ToolSpec` / `AgentSpec` / `NextTurnRule` の宣言流儀を維持するためであり、この選択により「型と `description` を `param(...)` と実行関数の signature の 2 箇所へ書く」二重記述と、宣言パラメータ名と tool 引数名がずれうること（別 Issue「関数実行エージェントの組み立て」の build 時検証で検出する）を受け入れる。
- 技術的制約: パラメータの型は JSON 直列化可能なものに限る。`plan.input_json` は `model_dump_json()` の結果であり、ツール関数が受け取るのは直列化後の JSON 値である。`datetime` 等 pydantic が直列化できる型は宣言できるが、ツール関数側は文字列として受ける前提で書く（lib は逆変換を行わない）。任意の独自クラスは `parameters_model()` 生成時に `ValueError` となる。
- 技術的制約: アクション実行のパラメータは `Runner.run` の `input`（`plan.input_json`）で渡し、`context` はアプリ横断情報の受け渡しに限る。決定的応答モデルを用いる実行エージェント（別 Issue「関数実行エージェントの組み立て」）では `ModelRequest` に run context のフィールドが無く、パラメータを `context` 経由にすると当該エージェントだけ別経路になるため。
- 技術的制約: 穴埋めで確定した値の `run_context` への書き戻しはライブラリが行わない。`run_context` は利用者の任意型であり汎用の setter を書けないためで、`from_context` は読み取り専用の宣言である。lib は `plan.slots` と `slot.from_user` を「書き戻し材料」として提供するに留め、書き戻しはアプリの責務とする。
- 技術的制約: パラメータ予測エージェントの実体はライブラリが構築し、利用者は `LLMFiller(model=...)` で `model` のみを渡す（どのモデル・どこへ接続するかはライブラリが決められないため）。構築は既存の `AgentBuilder` 経由で行い、宣言（`AgentSpec`）と実体化を分離する本ライブラリの原則に従う。予測エージェントは利用者の `AgentRegistry` へ登録されず `plan()` の内部でのみ使われ、`session` を渡す口も公開しないため、「業務エージェントとは別に置き会話履歴に穴埋めのやりとりを混入させない」が構造的に保証される。
- 技術的制約: `NextTurnRule.no_handoff_on_arrival` はハンドオフ到達記録を契機に働くため（ADR 0014 Decision 1）、`ActionCatalog` からの直接起動では作動しない。アクション実行中の途中遷移を止めたい場合は、実行専用エージェントを `handoffs` を持たない `AgentSpec` として宣言する（宣言時点で出辺を持たせない）。
- 技術的制約（命名）: `ActionSpec.action_agent`（押した後に遷移して実行する先の名前）と `NextTurnRule.next_agent`（実行後に次ターンを開始する先）は別概念であり、シンボル名を衝突させない。`ActionPlan` 側では `action_agent` が同じ名前（`str`）、`agent` が registry で解決した実体を指す。`runtime/lightning` の `prompt_slot`（APO のプロンプト分割）も本要件のスロットとは無関係である。`runtime/intent` 配下で `agents` という識別子（変数・引数）を使わない（SDK パッケージ名との誤読を避ける）。
- ビジネス制約（スコープ外）: 候補生成方式の実装（ルールベース / retrieval / 学習モデル本体）。候補提示 UI・穴埋め UI。利用者記憶（Profile / Preferences / Tasks / Intent Memory / Entity Memory / Parameter Memory）と行動履歴の永続ストア。候補提示ログの記録と評価指標の集計。複数系統の候補を混合・ランキングする合成器。
- ビジネス制約（段階導入）: 実装対象は宣言・型・決定的段・1 回の予測委譲・起動時検証に限る。retrieval は既存 `ContextBuilder` の差し替え（`catalog.bind()` の `context_builder` 引数）、学習は既存 ML 系シンボルの再利用で表現し、本要件では本体を追加しない。
- ビジネス制約（段階導入・実装 2 段）: 第 1 段は FR-1 / FR-2 / FR-3 / FR-4 / FR-5 / FR-8（宣言・型生成・結線と起動時検証・候補契約・決定的な空欄確定・入力組み立て）。LLM 呼び出しを一切含まないため単体で価値が閉じる。第 2 段は FR-6 / FR-7 / FR-9（予測委譲・失敗方針・実行後の次ターン解決）。FR-10（公開窓口）は各段で更新する。
- ビジネス制約: 本ライブラリは未リリースの Alpha であり後方互換を必須としないが、既存公開シンボルの振る舞いと `__all__` のメンバ集合の変更は実装前にユーザー合意を取る。

## 5. 影響範囲
- 関連コンポーネント: `src/oai_agentspec/runtime/intent/`（`types.py` / `protocols.py` / `factories.py` / `__init__.py` への追加と、宣言簿・スロット・結線宣言型・型生成・計画・予測委譲・検証の新規モジュール）、`src/oai_agentspec/_adapters/intent.py`（FR-6 の委譲窓口となる新規関数の追加）、`src/oai_agentspec/next_turn.py`（`action_next_turn_agent` の追加）と `src/oai_agentspec/__init__.py`（コア `__all__` へ 1 件追加）、`tests/runtime/intent/` / `tests/_adapters/` および `tests/` の next_turn テスト（ミラー構造・`_l1` / `_l2` 命名規則）、`docs/architecture.md` の意図予測の節、`docs/adr/`（予測委譲の実行例外・アクション起動時の次ターン解決・宣言と決定的確定の型と配置・公開 API 形）、`docs/requirements/`、`docs/QUALITY-GUARANTEES.md`、`examples/intent/`。
- 既存機能への影響: コアの宣言層と他の runtime 配下の公開窓口（`conversation` / `serve` / `cli` / `llmops` / `lightning` / `governance` / `guardrails` / `resilience` / `observability` / `deterministic` / `hooks`。うち `deterministic` / `hooks` は extra を持たない）の挙動と公開契約は変更しない。`pyproject.toml` の依存宣言も変更しない。既存 `next_turn.py` の公開シンボルは実装・挙動とも変更せず、`action_next_turn_agent` を純追加する。
- 既存ドキュメントの追随: `runtime/intent` の公開シンボル数は実測 24 件、コア `__all__` は実測 36 件である。現在仕様の SoT である `docs/architecture.md` の意図予測の節へ本要件の追加後の件数（40 件）を反映し、`docs/QUALITY-GUARANTEES.md` に「コア `__all__` のメンバ集合の不変」を強制するテストへのポインタを 1 行登録する。既存の要件定義書（`docs/requirements/intent-prediction-foundation.md` の FR-8「14 シンボル」/ NFR-3「27 件」）は当時の合意記録として書き換えない。
- 入口の二重化: エージェントへの起動経路が `HandoffGraph`（会話の遷移）と `ActionCatalog`（候補からの直接起動）の 2 系統になる。`docs/architecture.md` に両者が同一の `AgentRegistry` を別観点から参照する関係として明記する。

## 6. 用語定義
| 用語 | 定義 |
|------|------|
| 実行可能意図 | 「次に何をしたいか」をアクション識別子とパラメータの組で表した候補。カテゴリ名のみを返す既存の意図分類とは異なり、選択後に追加の文章入力なしで実行できる形を目標とする |
| アクション宣言簿 | `ActionSpec` の集合と全アクション共通の既定を保持する `ActionCatalog`。既存の `ToolSpec` と `ToolRegistry` の関係に対応する |
| `action_agent` | 候補が選択された後に遷移して当該アクションを遂行するエージェント名。業務 `AgentRegistry` の住人を指す。`ActionPlan` では同名フィールドが名前を、`agent` が registry で解決した実体を返す |
| `bind` | `ActionCatalog` へ結線対象を起動時に 1 度だけ渡す操作。アプリ既存資産（`registry` / `tools` / `prompts`）と、関心事ごとの宣言型（`CandidateSource` / `LLMFiller`）を受ける。実行はせず、以降の `validate` / `plan` がこの結線を使う |
| `CandidateSource` | 候補の出どころを束ねる frozen 宣言型（`generator` / `context_builder` / `history_limit`）。`bind(candidates=...)` へ渡す |
| `LLMFiller` | 不足パラメータの埋め方を束ねる frozen 宣言型（`model` / `on_invalid_response` / `guardrails`）。ガードレールの解決簿は `bind(guardrail_registry=...)` 側が持つ。`bind(llm_filler=...)` へ渡し、**渡さなければ穴埋め経路が存在しない**（従量課金が発生しないことが呼び出し側から読める） |
| `ExecutableIntent.text` | 既存 `IntentCandidate` の必須フィールド。契約維持のため `action_id` を写した値が自動補完される（利用者は渡さない） |
| パラメータ予測エージェント | 不足パラメータの値を予測する専用エージェント。利用者は `LLMFiller(model=...)` で `model` のみを渡し、実体はライブラリが `AgentSpec` + `AgentBuilder` で構築する。利用者の `AgentRegistry` には登録されず、`session` を渡さず、1 ターンにつき 1 回だけ走る |
| スロット | 1 パラメータの解決状態。`Slot` 型の `state` フィールド（`SlotState`）が `RESOLVED` / `NEEDS_AGENT` / `NEEDS_CONFIRMATION` / `NEEDS_USER` の 4 値を取る |
| `origin` | スロットの値の出どころ。`Origin`（`StrEnum`・6 値）で候補・run context・既定値・予測エージェント・利用者確認・利用者入力を判別し、run context のパスと予測エージェント名は `detail` フィールドへ入る |
| `from_user` | `Slot` の導出プロパティ。`origin` が `USER_INPUT` または `USER_CONFIRMED` のときだけ真となり、アプリが「自分のストアへ書き戻すべき値」を選り分けるために使う |
| 値の解決順 | 候補の値 → `from_context` のパス解決 → `by_agent` による予測 → `default` → 利用者入力 の優先順。`by_agent` の予測が失敗した場合は `default` へ後退する |
| `from_context` | run context から値を取るパスの宣言。値の出どころを指し、run context への書き込みは行わない |
| `confirm` | 値が埋まっていても利用者へ提示して確認を得る宣言。「押すだけで実行」を意図的に無効化する安全弁。値を埋める経路ではないため起動時検証の免除条件にはならない |
| `SlotSuggestion` | 1 パラメータに対する候補値 1 件（値・`ConfidenceLevel`・任意の根拠）。`max_suggestions` の件数まで並ぶ |
| `ExecutableSuggestion` | 候補列と `IntentContext`・`report`・`metadata` を保持する型。`catalog.plan(detail=True)` が返す `PlanResult.suggestion` として取得する。`IntentContext` を含むため `run_context` が呼び出し元へ露出する（`ActionPlan` 側は保持しない）。`run_context` に任意型が載るため直列化の成立は契約に含めない |
| `filled_by_candidate` | 「このパラメータは候補が常に値を載せる」ことの明示宣言。起動時検証の「埋まる経路が宣言に無い」WARNING を抑止するためだけに使い、値を埋める経路そのものではない |
| `prompt_vars` | プロンプトの `${var}` へ差し込む値を run context のパスから宣言するマッピング。キーはカタログ全体で一意 |
| セグメント重複排除 | 同一のプロンプト断片が複数の候補・パラメータから要求された場合に本文を 1 回だけ積む合成処理。入力トークンを不足分に比例させるための機構 |
| 実行エージェント | `action_agent` が指す先。利用者が `AgentRegistry` へ登録した任意のエージェントでよい。判断を要さないアクションを LLM 呼び出し 0 回で関数へ落とす「関数実行エージェント」は別 Issue（`docs/requirements/function-execution-agent.md`）で扱い、本要件はその有無に依存しない |
| build-don't-run | 宣言・build 時検証・薄い結線に徹し、実行は SDK `Runner.run` に寄せる本ライブラリの原則。本要件が持つ唯一の実行はパラメータ予測エージェントの 1 回の run である |
