# 0043: openai-agents の宣言範囲を 1 つの minor に閉じ、finetune extra の openai を SDK の要求に揃えて追随する

- Status: accepted
- Date: 2026-09-30

## Context

### 決定時点の宣言範囲

- core: `openai-agents>=0.17.4`（上限なし。`pyproject.toml` の `[project] dependencies`）
- finetune extra: `openai>=2.36.0,<3`。SDK 0.17.4 の `Requires-Dist: openai<3,>=2.36.0` と一致させる運用
  （同ファイルの finetune extra のコメント）
- lock（`uv.lock`）: openai-agents 0.17.4 / openai 2.38.0 / mcp 1.28.1。CI は `uv sync --all-extras` で
  この 1 構成だけを検証する

### 2026-09-29〜30 時点の実測

SDK 各版の必須依存（PyPI の Requires-Dist と `uv pip compile` で確認）:

| openai-agents | openai 要求 | mcp 要求 |
|---|---|---|
| 0.17.4〜0.18.1 | `>=2.36,<3` | `<2` |
| 0.18.2〜0.19.x | `>=2.45,<3` | `<2` |
| 0.20.x | `>=2.45,<3` | `<3`（mcp 2 系に対応） |
| 0.21.0〜0.22.3 | `>=3,<4` | `<3` |

lib の全テストスイートでの実行時互換（`uv sync --all-extras` の上で pytest を実行）:

| 構成 | 結果 |
|---|---|
| 0.17.4 + openai 2.38（lock） | 全件緑 |
| 0.17.4 + openai 2.54 | 559 failed / 4191 passed |
| 0.17.8 / 0.18.0 + openai 2.54 | `tests/_adapters/test_governance_l2.py` で 34 failed |
| 0.18.3 / 0.19.0 + openai 2.54 | `tests/_adapters/test_governance_l2.py` が緑（全件は未実行） |
| 0.19.4 + openai 2.54 + mcp 1.28 | 4750 passed（全件緑） |
| 0.20.0 + openai 2.54 | 5 failed / 4745 passed |
| 0.22.3 + openai 3.20（finetune を `openai>=3,<4` に緩めた構成） | 5 failed / 4745 passed |
| 0.22.3 + openai 3.20 + `uv lock --upgrade` | 同じ 5 failed / 4745 passed |
| 0.22.3 + mcp 2.2.0（lightning / llmops-langfuse を除く extras） | 上の 5 件 + 観測系の分離テスト 3 件 |

破損点と原因:

- **openai 2.45 以降と SDK 0.17.4〜0.18.1**: SDK の usage 構築（0.17.4 の `agents/usage.py:112`）が
  `InputTokensDetails` を `cache_write_tokens` なしで組み、openai 2.45 以降で必須化されたこのフィールドの
  欠落で pydantic の ValidationError になる。実測したのは 0.17.4 / 0.17.8 / 0.18.0 で、0.18.1 は
  Requires-Dist からの推定、0.18.2 は未実測。決定時点の宣言範囲はこの組み合わせを許している。
  openai 2.44 以下と組めば 0.17.4 は緑である。
- **SDK 0.20.0 以降の call_id 再利用の拒否**: 同じ run の中で完了済みの call_id を別の呼び出しに使うと、
  SDK が `ModelBehaviorError` を送出する。決定論モデル（`runtime.deterministic`）の既定 call_id
  （`src/oai_agentspec/_adapters/deterministic.py:37` の `DEFAULT_CALL_ID`。ADR-0019 の「8. 命名と既定 id 値」）
  のまま 1 run で 2 回以上 tool call すると該当する。上表の 5 failed はすべてこれで、
  `tests/_adapters/test_governance_l2.py` の 5 件（`test_audit_record_sequence_unchanged_by_default` ほか）。
- **mcp 2 系での観測系分離テスト**: mcp 2.2.0 が import 時に opentelemetry-api を読み込むため、
  `import agents` の時点で opentelemetry が `sys.modules` に入る。失敗するのは次の 3 件で、いずれも
  ADR-0022 の Confirmation が定めた「窓口 import で観測系 SDK をロードしない」保証を検査する。
  - `tests/test_extra_isolation.py::test_importing_package_does_not_force_load_extra_deps`
  - `tests/runtime/observability/test_config_l1.py::test_config_module_does_not_load_observability_sdks`
  - `tests/runtime/observability/test_reexport_l1.py::test_importing_window_does_not_load_observability_sdks`

  lock で mcp が 1 系に留まるのは lightning extra の依存（litellm の proxy extra）が `mcp<2` を要求する
  ためで、lightning を入れない構成には mcp 2 系が入る。この構成は CI に現れない。
- **finetune extra と SDK 0.21 以降**: SDK 0.21 以降は openai>=3 を要求するため、finetune extra の
  `openai<3` と組むと依存を解決できない（`uv pip compile` で No solution）。

lock を使わない新規インストールの解決（`uv pip compile`・Python 3.12）:

- core のみ: openai-agents 0.22.3 + openai 3.20.0
- finetune extra 併用: openai-agents 0.20.0 + openai 2.54.0

どちらの解決も call_id 再利用の拒否の影響下にあり、CI が検証している版（0.17.4 + openai 2.38）とも
一致していない。

extras との共存（SDK 0.22.3 と組み合わせた `uv pip compile`）:

- agent-governance-toolkit 4.1.0（integrations は `openai-agents<1.0`）: 解決できる
- microsoft-agents-a365-observability-extensions-openai 1.0.0（`openai-agents>=0.2.6`）: 解決できる
- opentelemetry-sdk 1.45: 解決できる
- finetune extra の `openai<3`: 解決できない（上記）

`uv lock --upgrade` した all-extras 構成でも失敗は call_id の 5 件だけで、extras に起因する失敗は無い。

### SDK 0.22.3 に対する内部結合レイヤーの実測

SDK の非公開・内部挙動へ依存する箇所（`_adapters` 配下）と、openai への直接依存を、リリースノート・SDK
ソースの差分・SDK 0.22.3 + openai 3.20.0 の環境での実行で照合した。結果を 4 区分に分ける。

#### 変化なし・成立を確認したもの

- SDK ソースの差分で変化が無いことを確認した依存点: MCP 由来ツールの origin メタ、`dataclasses.replace`
  による `FunctionTool` のラップと同一性判定（ADR-0042 の前提。0.20.0 で `__copy__` が再実装されたが
  ラッパは素の関数のため保たれる）、`Agent.clone` の浅いコピー、handoff の署名検証と `is_enabled` 規則、
  SQLite セッションのスキーマ、トレーシングの無効判定と `NoOpTrace`、Agent 系の dataclass フィールド、
  `AgentHooksBase` の `on_*` 集合、`run_data.last_agent`、`Model.get_response` の引数順
- openai を直接呼ぶ 8 箇所（finetune 5 箇所・lightning 3 箇所）: openai 2.38 と 3.20 でシグネチャ・例外型・
  `files.wait_for_processing` のタイムアウト時の RuntimeError がすべて同一で、差分は 0 件だった。
  ADR-0032 の前提は保たれる。0.22.3 + openai 3.20 の環境で finetune / lightning のテストは 579 passed
- 利用者が自分で組むクライアント（`AsyncOpenAI` / `AsyncAzureOpenAI` / `OpenAIResponsesModel` /
  `OpenAIChatCompletionsModel`。examples の Azure 共有ヘルパの全分岐を含む）: 両版ともネットワーク無しで
  構築できた。examples の修正は要らない
- 承認の sticky（恒常）判断: 0.21.1 で個別の承認判断が sticky 判断より優先されるようになったが、lib は
  sticky 判断を使っていない（`src/oai_agentspec/_adapters/approvals.py:116,118` の承認・却下はいずれも
  恒常指定なしで、公開入力の承認判断にも sticky の口が無い）。lib の承認・再開フローは両版で同じ挙動だった
- strict schema の変換: 変換対象は利用者の型を `TypeAdapter` にかけたスキーマで、lib が独自に組むスキーマ
  ではない。dict 系の型（`dict[str, Any]`、`extra="allow"` 等）が `UserError` になるのは 0.17.4 でも同じで、
  lib の fallback（`src/oai_agentspec/_adapters/builders.py:325-329` の except）には試した 24 ケースのどれも
  両版で到達しなかった。lib のロジックは変えない（0.22.3 で新たに拒否される入力は下の告知対象に置く）
- ADR-0025 の Consequences の「deny 時も開始済みになり得る」: 0.19.4 で並行実行が `asyncio.gather` から
  取消付きの並行実行へ変わったが、新版でも開始は記録されるため帰結は成り立つ。違いは開始後の扱いで、
  旧版は run の失敗後も完了まで走り、新版は最初の await の時点で取り消される（await を含まないフックは
  新版でも完了する）。ADR-0025 の決定には影響しない。`asyncio.gather` という字面は ADR-0025 と
  `src/oai_agentspec/_adapters/governance.py:834` の docstring に残っている

#### 要修正

- **決定論モデルの既定 call_id**（上記の破損点）。0.22.3 で実測した 10 シナリオでは、既定の固定 ID のまま
  2 回以上 tool call する構成が 10 件すべて失敗した。同じ引数の繰り返しは例外にならず、`MaxTurnsExceeded`
  まで黙ってループする。0.17.4 でも固定 ID のままでは 10 件中 4 件が `MaxTurnsExceeded` になった（原因は
  未確認）。テストヘルパ `tests/_helpers/fake_model.py:68-79` の `queue_tool_call` も既定の call_id を使うため、
  lib 側の修正とは別に直す必要がある
- **govern ラップと条件付き承認**（セキュリティ）。0.22.3 は、実行本体が SDK 内部の失敗ハンドラ付き invoker
  のときだけ、callable な `needs_approval` の前に引数を検証する。検証で引数が変わる場合（緩い型変換・
  既定値の補完・予測できないスキーマ）は、判定関数を呼ばずに承認を必須にする。govern は実行本体を素の関数で
  置き換えるため、この強化を受けない。実測では、govern 済みツールの判定を引数の表現を変えるだけで
  すり抜けられた（すり抜けた入力は 4 件: 数値の文字列表現 `"5000"`・`5000.0`・真偽値の文字列表現、既定値なし
  ツールでの `"5000"`）。判定関数には文字列の `'5000'` が渡り、実行時は整数の `5000` で走る。govern なしの
  ツールはすり抜けないため、govern を足すと承認が弱くなる逆転になる。監査ログには生の引数が残り、実際に
  実行された値と食い違う。0.17.4 では govern の有無にかかわらずすり抜ける（lib の退行ではなく、0.22.3 で
  生まれた差）
- **usage 欠損の検知**。0.22.3 の SDK 組み込みモデル（`OpenAIChatCompletionsModel` / `OpenAIResponsesModel`）は、
  usage が欠けた応答でも requests を 1 と数える。このため、requests と total_tokens がともに 0 のときだけ
  欠損とみなす判定（`src/oai_agentspec/_adapters/resilience.py:212-223` と
  `src/oai_agentspec/_adapters/intent.py:104-111`）が働かなくなる。実測では、resilience の警告が 1 件から
  0 件になり、intent のトークン値が None から 0 になった。自作 Model が空の `Usage()` を返す場合は新版でも
  検知が働く。また、1 回 retry した後は requests が 1 以上になり検知が効かないという穴は、両版に共通して
  既にある
- **serve のエラー分類**。`src/oai_agentspec/runtime/conversation/service.py:504-511` は、エラー文言に
  `model` を含む例外を「モデル未構成」に分類する。このため、0.21.1 で新設された `ModelTimeoutError` が
  REST では 503 の `model_not_configured` と表示された（実測）。`ModelTimeoutError` が送出されるのは
  `ModelSettings.timeout`（0.17.4 には無いフィールド）を指定したときだけである
- **mcp 2 系での観測系分離保証**（上記の破損点）。ADR-0022 の Confirmation が定める保証の定義を見直す必要がある

#### 挙動の変化（利用者への告知対象）

lib のロジックを直さず、利用者へ告知する変化は次の 8 項目である。

1. SDK の既定モデルが変わった（0.20.0）。model を省略した `AgentSpec` が対象で、lib は既定モデルを持たない
2. openai 3 が必須になり、openai 2 に縛られた環境とは同居できない
3. finetune extra の利用者は openai 3 へ上がる
4. RunState の永続化形式は前方向にしか互換がない。旧版で保存したものは新版で読めるが、新版で保存した
   ものを旧版で読むと fail-fast になる
5. strict schema で新たに `UserError` になる入力が 3 系統ある。ルートが anyOf の型（`Optional[Model]` や
   `Union[A, B]` をルートにしたもの）、`json_schema_extra` で明示した `additionalProperties: {}`、制約の
   兄弟キーを持つ `$ref`。動的 handoff の経路は `strict_json_schema` を False にすれば build できる（両版で
   実測）。SDK の `handoff()` が無条件に strict 化する静的経路には回避策が無い（SDK ソースを読んだ結果で
   未実測）。`src/oai_agentspec/_adapters/builders.py:297-300` の docstring「想定外スキーマは pydantic 生成
   スキーマをそのまま返す」は `UserError` の場合と食い違っている
6. Responses API が failed / incomplete を返すと、`ModelBehaviorError` が送出される。旧版では空または部分的な
   出力で完了していた。serve では 500 の `execution_error` になり、安全側に倒れる
7. `ModelTimeoutError` は `AgentsException` の子で `TimeoutError` の子ではない。failsafe に `TimeoutError` を
   キーとして宣言しても着地しない。`AgentsException` か `ModelTimeoutError` をキーにすれば着地する。
   関連して、SDK の `RunErrorHandlers` に `invalid_final_output` キーが加わり、
   `src/oai_agentspec/runtime/resilience/_errors.py:16` の docstring の字面が古くなった
8. openai 3 の HTTP 層は httpx2 になった。旧 `httpx.AsyncClient` を `http_client=` に渡しても、現時点では
   警告なしで受理されて同じように動く（実測）。ただしこれは httpx が import 済みの場合に限る互換経路である。
   `APIStatusError` の `.response` / `.request` の型注釈も httpx2 の型になった（シグネチャを読んだ結果で
   未実測）

#### なお未実測のもの

- ストリーム経路での usage 欠損の扱い
- LiteLLM / AnyLLM 系のモデルでの usage の扱い
- WebSocket 経路で `ModelTimeoutError` が返すエラーコード（REST と同じ分類関数を通ることはコードからの推論）
- a365 への実送信（テストは fake）
- SDK の `handoff()` が strict 化する静的経路で、strict schema に回避策が無いこと（SDK ソースを読んだ結果）
- `APIStatusError` の `.response` / `.request` の型注釈の変化（シグネチャを読んだ結果）
- 0.17.4 で既定の固定 call_id のまま一部のシナリオが `MaxTurnsExceeded` になる原因
- 0.21.x での全テストスイートの実行

R4 との関係（R4 の対象は、引き上げ先の目標範囲の版での、lib の結合点、つまり `_adapters` が依存する
SDK / openai の挙動に関わる未実測点に限る）:

- ストリーム経路の usage と WebSocket 経路のエラーコードは、lib の欠損判定と serve のエラー分類が依存する
  挙動である。usage 欠損の検知と serve のエラー分類を直す引き上げ作業の中で、同じ判定をその経路でも検証する
  （R4 の対象）。
- 静的経路の回避策の有無は、利用者への告知の内容を左右する SDK の挙動である。告知を書くときに実測する
  （R4 の対象）。
- LiteLLM / AnyLLM は lib が提供するモデル実装ではない。直した後の欠損判定は requests に依存しないため、
  モデル実装によらず働く。R4 の対象外とする。
- a365 への実送信は従量課金の外部送信で、既存どおりテストは fake とする。R4 の対象外とする。
- `APIStatusError` の型注釈は告知の記述にとどまり、lib の結合点ではない。R4 の対象外とする。
- 0.17.4 の `MaxTurnsExceeded` の原因は引き上げ前の版の挙動で、R4 の対象外とする。
- 0.21.x は目標範囲の外で、R4 の対象外とする。

### Dependabot の実態

pip の週次更新で、uv.lock の bump PR と、pyproject の上限を広げる PR（finetune の openai の `<3` を `<4` へ
広げるもの等）は作られている。一方で openai-agents の更新 PR は過去に 1 度も作られていない。

2026-09-30 時点で open 中の Dependabot PR は 5 件（いずれも Python 依存）で、`.github/dependabot.yml` の
pip エントリの `open-pull-requests-limit: 5` に達している。上限に達している間は新しい PR が作られないため、
openai-agents の更新 PR が出ないこととこの上限到達は整合する（過去の期間に上限に達していたかは未確認）。
別の説明もありうる。lock の中で openai-agents を上げるには openai も同時に上げる必要があり、finetune extra の
`openai<3` と衝突して単独の更新を解決できない、というものである。いずれの原因でも、次に範囲を上げる契機を
現状の Dependabot の設定に頼れない。

### 段の計画: 単段（0.22.x へ直行）

範囲の引き上げは 0.22.x への 1 段で行い、0.19.x を経由しない。利用者の人数を計測する手段は無いため、
根拠は構成ごとの影響で示す。

- core のみの新規インストールは決定時点で 0.22.3 + openai 3.20 に解決されている。0.22.x へ直行しても、
  この構成の解決版は変わらない。
- finetune 併用の構成は 0.20.0 + openai 2.54 に解決されており、直行で 0.22.x + openai 3.x へ上がる
  （前方向のみ）。

0.19.x への是正を先に入れた場合に負担を負う構成:

| 構成 | 決定時点の解決 | 0.19.x 是正後 | 負担 |
|---|---|---|---|
| lock を使わない新規インストール（core のみ） | 0.22.3 + openai 3.20 | 0.19.x + openai 2.x | SDK がダウングレードされる |
| 上と同じで、他の依存が openai>=3 を要求する環境 | 0.22.3 + openai 3.x | 解決不能 | インストールが失敗する |
| finetune extra を併用する構成 | 0.20.0 + openai 2.54 | 0.19.x + openai 2.x | SDK がダウングレードされる |
| 0.20 以降で保存した承認待ちの RunState を持つ serve 利用者 | 復元できる | 0.19.x の RunState は旧 schema のため fail-fast | 承認待ちを復元できない |

0.19.x への是正で避けられる実害:

| 構成 | 実害 | 規模 |
|---|---|---|
| 決定論モデルを既定の call_id のまま使い、1 run で 2 回以上 tool call する（SDK 0.20 以降） | `ModelBehaviorError` で run が失敗する。同じ引数の繰り返しは例外にならず `MaxTurnsExceeded` まで黙ってループする | lib スイートで 5 failed。シナリオ実測の結果は「要修正」の call_id の項を参照（0.17.4 でも一部のシナリオが `MaxTurnsExceeded` になる）。一意な call_id を指定すれば回避できる（応答ビルダの call_id 引数） |
| SDK 0.17.4〜0.18.1 を自分で固定し、openai>=2.45 と組む | usage 構築で ValidationError | 0.17.4 + openai 2.54 で 559 failed。解決器の既定ではこの組み合わせにならない |
| SDK 組み込みの ChatCompletions / Responses モデルが usage の欠けた応答を返す（SDK 0.20 以降） | resilience の usage 欠損の警告が出ない。intent のトークン値が未取得の None ではなく 0 になる | 実測値は「要修正」の usage 欠損の検知の項を参照。run 自体は失敗しない |

是正が防げる実害は、回避策がある構成、既定の解決では起きない構成、または run を失敗させない観測値の劣化に
限られる。一方で是正は上表の構成にダウングレードまたは解決不能を強いる。mcp 2 系での分離テストの失敗は
lib のテストが守る不変条件の問題で、利用者の実行には影響しない（import 時に opentelemetry-api がロード
されるだけ）。strict schema の変化は 0.19.0 からのもので、0.19.x の範囲にも含まれる。

受け入れたコスト: 0.22.x への引き上げが完了するまで、宣言範囲が既知の破損組み合わせを許す状態と、
CI の検証版と利用者の解決版の不一致が続く。

### 範囲を上げる作業のスコープ

作業は次の 2 単位に分ける。

**引き上げ作業**（1 単位で行う）

1. 決定論モデルの既定 call_id を、SDK の同一 run での再利用拒否に対応させる。テストヘルパ
   （`tests/_helpers/fake_model.py:68-79`）も直す。方式は作業の冒頭で決める。実測で分かった制約は次のとおり。
   - SDK の拒否条件: 同じ run の中で、call_id が同じで指紋（tool または承認スコープと、正規化した引数）が
     違うと拒否する。指紋まで同じ場合は拒否せず、出力済みなら黙って実行を省く。範囲は run 単位で、Session を
     跨いだ run は対象外
   - ステートレス契約（ADR-0019 の「3. 入力は `ModelRequest` として渡し、多ターン判別フィールドを入力から
     導出する」、`docs/QUALITY-GUARANTEES.md` の決定的応答モデルのステートレス契約の行）: インスタンスに
     カウンタを持たせる採番はこの契約を壊す。要求から導く方式はいずれもこの契約のテストを緑に保った
   - 入力の刈り込み: ターン番号や tool 出力の件数から導く方式は、既定の構成では成立するが、入力が刈り込まれる
     3 構成（handoff の `input_filter`、`nest_handoff_history`、`call_model_input_filter`）で失敗した
   - 同一入力での衝突: instructions と入力全体のハッシュから導く方式は、10 シナリオすべてで成立した。ただし、
     instructions と入力が同じで応答が違うケース（tools などで分岐するルール、非純粋なルール）では衝突する
   - 利用者による指定の必須化: 刈り込み・共有・同一入力による衝突は原理的に起きないが、利用者のコード修正が要る
   - 既定値の維持と案内の拡張のみ: 0.22.3 では 2 回以上 tool call する構成がすべて失敗し、同じ引数の繰り返しは
     分かりにくい失敗になる

   Runner 経由で観測される call_id が変わる方式を採る場合は、ADR-0019 の「8. 命名と既定 id 値」の決定と、
   `docs/QUALITY-GUARANTEES.md` の既定 id 値の行を、新しい ADR で改める。
2. govern ラップを条件付き承認の強化に揃える（セキュリティ上の修正）。推奨する候補は、govern ラップを SDK
   内部の失敗ハンドラ付き invoker のサブクラスとして実装する方式である。試作では、試した 9 ケースすべてで
   govern なしと同じ挙動になり、governance 系テストの退行は 0 件だった（残った失敗は call_id の 5 件）。
   最終的な方式は、作業の冒頭で他の候補（`needs_approval` を包む方式、tool input guardrail へ移す方式）と
   比べて決める。推奨する候補の代償は次の 3 点である。
   - SDK の非公開実装への依存が増え、SDK バージョン耐性のトリップワイヤが要る
   - 統治済みの判定を、実行本体の同一性に加えてサブクラス判定と併用する必要がある
   - 既定値を省略した呼び出しで承認要求が増える（SDK の既定の挙動に揃う）
3. usage 欠損の検知（`src/oai_agentspec/_adapters/resilience.py:212-223` /
   `src/oai_agentspec/_adapters/intent.py:104-111`）を、requests の計上に頼らない判定へ直す。serve のエラー
   分類（`src/oai_agentspec/runtime/conversation/service.py:504-511`）が `ModelTimeoutError` をモデル未構成に
   分類しないよう直す。
4. 更新一式: pyproject の宣言範囲と finetune extra のコメント、uv.lock、README の要件表記と版表記、版番号
   （依存範囲の変更は利用者の解決結果を変えるため pre-1.0 の minor bump とする。先例: ADR-0007・ADR-0011）、
   古くなった docstring（`src/oai_agentspec/_adapters/builders.py:297-300` の strict schema、
   `src/oai_agentspec/_adapters/governance.py:834` と `src/oai_agentspec/runtime/resilience/_errors.py:16` の
   字面）、Dependabot の設定（`open-pull-requests-limit` の見直し、または手動の定期確認への切り替え）、
   利用者への告知（「挙動の変化」の各項目）。

**mcp 2 系での観測系分離保証の再定義**（引き上げ作業とは別の単位で行う）

- ADR-0022 の Confirmation の保証を「lib が強制ロードしない」へ定義し直す。SDK 経由で推移的に入る
  opentelemetry-api を対象外にし、観測系 SDK 本体は引き続き禁止する形を想定する。どこまでが SDK 経由の推移かは
  この作業の中で実測して確定する。保証を狭めるため新しい ADR を起こす。

ADR-0025 の決定（MCP 由来ツールを `AgentHooks.on_tool_start` で評価する）は本 ADR では変えない。同 ADR の
Context にある却下理由「MCP 由来 `FunctionTool` には `tool_input_guardrails` が付かない」は SDK 0.22.1 以降では
成り立たなくなるが、評価点を MCPServer 単位のガードレールへ移すかは別の判断として扱う。

### 検討し却下した案

| 案 | 内容 | 却下理由 |
|---|---|---|
| 据え置き | `>=0.17.4`・上限なしのまま | 宣言範囲が既知の破損組み合わせ（0.17.4〜0.18.1 と openai>=2.45）を許す。新規インストールは 0.20 以降に解決され、call_id の拒否で失敗しうる。CI の検証版と利用者の解決版が一致しない |
| 上限のみ追加 | `>=0.17.4,<0.20` | 下限 0.17.4 は openai>=2.45 と組むと壊れ、下限を全件緑の実測版にする規則に反する。core のみの構成も finetune 併用の構成も 0.20 未満へダウングレードされる |
| 下限 0.18.2 | `>=0.18.2,<0.20` | SDK の Requires-Dist が openai>=2.45 を強制するので破損組み合わせは除けるが、0.18.2 は未実測で規則に反する。構成ごとのダウングレード負担は 0.19.x 止めと同じ |
| 0.19.x 止め | `>=0.19.4,<0.20` で終える | 全件緑を実測済みでコード修正も要らない見込みだが、上の負担表の構成にダウングレードまたは解決不能を強いる。0.22.1 の MCPServer 単位のツールガードレールなど以降の改善も取り込めない |
| 段階追随 | 0.19.x を経て 0.22.x へ | 第 1 段で 0.19.x 止めと同じ負担を強いる一方、防げる実害は限られる。pin の変更が 2 回になる |

## Decision

- **R1**: openai-agents の宣言範囲は常に 1 つの minor に閉じ、範囲の引き上げは R2〜R4 を満たす単位で行う（段を踏むかどうかは
  Context の段の計画に置く）。
- **R2**: openai-agents の宣言範囲は、下限を「CI と同じ `uv sync --all-extras`（lock）構成で全件緑を実測した版」、上限を
  「次の minor 未満」とする（0.x の minor は破壊的変更を含むため）。
- **R3**: finetune extra の openai 範囲は、その時点の SDK の Requires-Dist と一致させる。
- **R4**: 引き上げ先の目標範囲の版での、lib の結合点（`_adapters` が依存する SDK / openai の挙動）に関わる
  未実測点は、範囲を上げる作業の中で実測してから上げる。lib が提供しないモデル実装での挙動と、従量課金の外部サービスへの実送信は対象外とする。
- **R5**: Dependabot が openai-agents や finetune の openai の上限を広げる PR を出しても、そのままマージしない。
  次段の追随作業の契機として扱い、R2〜R4 に従って範囲を変える。

## Issue の受け入れ基準への着地

本表を Issue 受け入れ基準の着地の唯一の記載箇所とし、他の節・他ファイルへ複製しない。

| Issue の受け入れ基準 | 着地 | 根拠 |
|---|---|---|
| 追随可能な openai-agents / openai の対応バージョン範囲が明確になっている | 満たす | 対応範囲の目標は `openai-agents>=0.22.3,<0.23`、finetune extra は `openai>=3.0.0,<4`（R2・R3 の適用）。決定時点で全件緑を実測したのは 0.19.4 + openai 2.54 である。全件を実行した 0.20.0 と 0.22.3 で残る失敗は call_id の 5 件で、0.22.3 + mcp 2.2.0 ではこれに加えて分離テスト 3 件がある。0.21.x の全件は実行していない（Context の実測表） |
| SDK 呼び出しを行う内部結合レイヤー全体が、対象バージョンでも破壊的変更なく動作する見込みがあるか（または要修正箇所）が洗い出されている | 満たす | Context の「SDK 0.22.3 に対する内部結合レイヤーの実測」に、変化なし・成立を確認したもの、要修正（call_id・govern と条件付き承認・usage 欠損の検知・serve のエラー分類・mcp 2 系の分離保証）、挙動の変化、なお未実測のものを分けて記載した |
| ファインチューニング関連機能のうち openai パッケージへ直接依存している箇所について、バージョン整合が保てるかが確認されている | 満たす | 決定時点の `openai<3` は SDK 0.21 以降と解決できない。`>=3.0.0,<4` へ変えれば解決できる。openai を直接呼ぶ箇所は、シグネチャ・例外型・`files.wait_for_processing` のタイムアウト時の RuntimeError（ADR-0032 の前提）が openai 2.38 と 3.20 で同一だった（Context の「変化なし・成立を確認したもの」） |
| 任意導入の拡張機能（ガバナンス連携・観測連携）が新しい依存バージョンと共存できるかが確認されている | 満たす | agent-governance-toolkit 4.1.0・a365 拡張 1.0.0・opentelemetry-sdk 1.45 はいずれも SDK 0.22.3 と解決でき、実行時の失敗は call_id の 5 件だけ。ガバナンス連携では govern 済みツールの条件付き承認の差分を実測し、引き上げ作業で直す。mcp 2 系での観測系分離保証は別の作業単位で定義し直す。a365 への実送信は未実測 |
| 追随する/しないの意思決定と、追随する場合の作業スコープが明文化されている | 満たす | 追随する。段は 0.22.x への単段（Context の「段の計画」）。作業スコープは Context の「範囲を上げる作業のスコープ」（引き上げ作業と、mcp 2 系での観測系分離保証の再定義の 2 単位） |

## Consequences

- + 範囲の変更後は、宣言範囲が既知の破損組み合わせを許さなくなる。
- + 範囲の変更後は、利用者の解決版と CI の検証版が同じ minor に揃う。
- + core のみの新規インストール構成は、決定時点の解決版（0.22.3）から版が変わらない。finetune 併用構成は
  前方向へ上がるだけで、ダウングレードは起きない。
- - 範囲の変更までは、宣言範囲が既知の破損組み合わせを許す状態と、CI の検証版と利用者の解決版の不一致が
  残る（単段を選んだことで受け入れたコスト）。
- - mcp 2 系での観測系分離保証の再定義が完了するまで、lightning を入れない構成（mcp 2 系が入る）では
  ADR-0022 の保証が成り立たない（再定義を引き上げ作業と別の単位にしたことで受け入れたコスト）。
- - 条件付き承認の差分を解消するまで、SDK を上げた構成では govern 済みツールだけ承認が弱い。このため
  差分の修正は引き上げと同時に行う。
- - openai 3 が必須になり、openai 2 に縛られた環境とは同居できない。
- - RunState は前方向にしか互換がないため、範囲の変更後に旧 SDK へ戻すと承認待ちを復元できない。
- - Dependabot の同時オープン上限に達している間は、openai-agents の更新 PR が作られず、次段の契機に
  ならない（設定の見直しか手動の定期確認への切り替えを引き上げ作業で行う）。
- - agent-governance-toolkit の integrations が `openai-agents<1.0` を要求するため、SDK 1.0 への追随は
  同ツールキット側の追随に縛られる。
- - lightning を入れない構成（mcp 2 系が入る）は CI で継続的には検証されない。

## Confirmation

- 強制手段は pyproject の宣言範囲そのものと、範囲を変えるときの CI 全件実行・`uv pip compile` による
  新規解決の確認である。R1〜R5 の遵守は、範囲を変える変更のレビューで担保する。
- `docs/QUALITY-GUARANTEES.md` へは行を追加しない。本決定の時点では `src/` と依存範囲を変えず、台帳が指す
  べき強制手段テストを新設しないためである。

本決定を覆す場合は、ADR の append-only 規約に従い新規 ADR を起こし、本 ADR の Status を
`superseded by NNNN` に変更する（本文そのものは書き換えない）。
