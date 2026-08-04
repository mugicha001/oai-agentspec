# 0019: ルール関数が応答を決めるステートレス Model と応答ビルダを runtime.deterministic で公開する

- Status: accepted
- Date: 2026-08-04

## Context

実 API を呼ばずに決定的な応答を返す `Model` 実装は、自動テストだけでなくオフライン開発・デモ実行・
決定的なシナリオ再生でも必要になる。リポジトリ内には決定的 Model が 2 つあるが、いずれも用途が
固定されている。`DeterministicToolCallModel` は「保持する固定 tool 名を 1 回呼ぶ ToolCall」に応答が
固定され、`WorkflowModel` は応答がワークフロー内部インタプリタの結果に固定される。どちらも
`_adapters` 配下の内部窓口であり利用者向け公開 API ではない。上流 openai-agents SDK にも `Model`
抽象のテストダブル実装は存在しない（配布パッケージ内は実プロバイダ実装と ABC のみで、
`agents/models/fake_id.py` の `FAKE_RESPONSES_ID` は id 穴埋め用の定数であって Model ではない）。

応答オブジェクトの構築には `agents` / `openai` の型知識が要るため、ルール関数を書く利用者が SDK 内部型を
直接扱わずに済むビルダも同時に必要になる。テスト用ヘルパ `tests/_helpers/responses.py` に同型のビルダが
4 種あるが、`tests/` 配下のため配布物に含まれず利用者は import できない。加えて「一言返してから tool を
呼ぶ」「発話しながらハンドオフする」ように 1 応答へアシスタントテキストとツール呼び出しを混在させる
需要があり、これは既存 4 種のいずれでも表現できない。

命名は「テスト専用を含意しない」ことが要件で、識別子（クラス名・関数名・モジュール名・公開窓口の
import パス）に加えて応答オブジェクトの既定 id 値も対象とする。`Fake` / `Mock` / `Dummy` は禁止語、
既定 id 値はこれに `workflow` / `wf` を加えた語を含まない固定値とする。

SDK 隔離の台帳行は強制手段（`src/` 走査の fitness test）が未整備のため本件では登録せず、別 Issue とする。

### 検討した選択肢

- **`DeterministicToolCallModel` を拡張してルール関数方式を取り込む（却下）**: 公開クラスに内部専用の
  `tool_name` 引数が露出し、既存経路D の挙動（毎回同一 tool を 1 回・`stop_on_first_tool` 前提・
  終端イベント 1 件の streaming）に条件分岐が混ざってクラスの契約が読めなくなる。
- **`_adapters/models.py` へ同居させ、ビルダを `_adapters/response_builders.py` へ別立てする（却下）**:
  既存の runtime 窓口は例外なく「1 窓口 ↔ 1 専用 `_adapters` モジュール（`_adapters/__init__.py` から
  import しない）」の形（`_adapters/hooks.py` / `guardrails.py` / `intent.py`）であり、この配置から
  外れる。`_adapters/models.py` の module docstring も責務を「ワークフロー結合の SDK Model / Tool
  実装」に限定しており、汎用の公開モデルを同居させると「Model 3 件のうち 2 件だけが
  `_adapters.__all__` に載る」非対称も生じる。加えて新規モジュールが 2 個になる。
- **既存 2 ビルダと公開ビルダを完全併存させる（却下）**: SDK Responses item の構築コードが 2 箇所へ
  重複し、SDK の必須フィールド追加時に片方だけ直る silent gap を生む。
- **既存 2 ビルダを公開ビルダで置き換え id 値も統一する（却下）**: ワークフロー経路の応答 id が変わり、
  stream 終端の `resp_workflow` / `oai-agentspec-workflow` との一貫性も崩れる。既存 2 ビルダの現行挙動を
  変更しない制約に反する。
- **共有 item ヘルパを新規モジュール側へ置く（却下）**: `responses.py -> deterministic.py`（ヘルパ）と
  `deterministic.py -> responses.py`（`latest_user_text`）で module レベルの循環になる。
- **`stream_response` を非対応（`NotImplementedError`）にする（却下）**: 実装は最小で済むが、
  `Runner.run_streamed` を使う利用者の経路が公開モデルだけ塞がれ、既存 2 モデルとの非対称が残る。
  ライブラリ内部が最終応答を保持している点は既存 2 モデルと同じであり、post-execution streaming を
  lib の責任で組めない技術的理由がない。
- **応答の item ビルダとコンポーザを公開して任意の組み合わせを利用者に組ませる（却下）**: 公開面が
  増える割に、現状必要な組み合わせは「テキスト + ツール呼び出し」の 1 種類しかない。専用ビルダ
  1 つで足り、将来別の組み合わせが必要になれば非破壊で追加できる。
- **PEP 562 遅延再エクスポート窓口（`runtime/hooks` 方式）（却下）**: `_adapters/__init__.py` が
  `.models` / `.responses` を eager import し、`oai_agentspec` の import で `agents` が必ず載るため
  遅延が成立しない。追加依存もゼロで extra 未導入耐性のための遅延も不要。
- **依存ゼロ extra を追加する（却下）**: 追加依存がなく、`runtime/hooks`（extra 宣言なし・コア依存のみ）
  という近い先例がある。extra を足しても利用者の import 経路は変わらず `pyproject.toml` の差分が
  増えるだけで、後から追加しても非破壊である。

## Decision

入力からルール関数が応答を決める純関数方式のステートレス `Model` 実装と応答ビルダ 5 種を、
`oai_agentspec.runtime.deterministic` 窓口から公開する。

### 1. 実体は新規 `_adapters/deterministic.py` の 1 モジュールへ一本化する

`ModelRequest`（frozen dataclass）・`DeterministicResponseModel`・公開ビルダ 5 種・既定 id 定数 5 件を
新規 `_adapters/deterministic.py` へ置く。`_adapters/models.py` と `_adapters/__init__.py` は無変更で、
新モジュールは `_adapters/__init__.py` から import しない（既存 3 モジュールと同じ配置パターン）。
既存 2 モデル（`WorkflowModel` / `DeterministicToolCallModel`）との責務差は「固定応答の内部専用モデル」
対「応答決定を利用者ルール関数へ委譲する公開モデル」であり、`docs/architecture.md` の
「決定的応答モデル」節に記す。

### 2. item 構築と streaming イベント生成は `_adapters/responses.py` の非公開共有ヘルパへ集約する

`_adapters/responses.py` に `_make_text_message` / `_make_function_call` を新設し、既存
`text_response` / `tool_call_response` はワークフロー用 id（`msg_workflow` / `wf_call` /
`item_id=None`）を渡す薄い委譲にする（挙動・id 値は不変）。公開ビルダは
`_adapters/deterministic.py` から同ヘルパを参照する。依存は `deterministic.py -> responses.py` の
1 方向のみで循環しない。

既存の streaming イベントヘルパ `_completed_event` / `_text_delta_events` は id 値をハードコード
している（`resp_workflow` / `oai-agentspec-workflow` / `item_id="msg_workflow"`）。このままでは公開
機能が生成するイベントに禁止語 `workflow` が乗るため、**両ヘルパも引数化する**
（`_completed_event(output_items, sequence_number, *, response_id, model)` /
`_text_delta_events(text, *, item_id)`）。既存呼び出しはワークフロー用の値を明示的に渡すため挙動は
不変である。

集約対象は必須フィールドを持つ item 構築（`ResponseOutputMessage` / `ResponseFunctionToolCall`）と
streaming イベント生成に限る。`ModelResponse` のラップは 1 行かつ SDK 追随リスクが item 側に集中して
いるため 2 箇所に残す。ヘルパはいずれも `_` 前置の非公開とし、`_adapters/__init__.py` への
再エクスポートは追加しない（`_text_of` と同じ扱い）。

### 3. 入力は `ModelRequest` として渡し、多ターン判別フィールドを入力から導出する

ルール関数へ渡す `ModelRequest` は `system_instructions` / `input` / `user_text` / `turn` /
`tool_outputs` / `model_settings` / `tools` / `handoffs` / `output_schema` を持つ frozen dataclass と
する。`tracing` / `previous_response_id` / `conversation_id` / `prompt` は応答決定に使う合理的理由が
なく公開面を広げるだけなので渡さない（後から増やすのは非破壊）。

`turn`（初回 0）と `tool_outputs`（入力中の `function_call_output` 列）はいずれも `input` から純粋に
導出する。内部カウンタを持たないためステートレス性は保たれる。`turn` は **1 モデル応答 = 1 ターン**
として数え、ターン境界は「user / tool 側が話したところ」（role が `user` / `system` / `developer`、
または `*_output` 型の item）で判定する。assistant 側の item 種別を列挙する allowlist にしないのは、
SDK が assistant 側の item 種別を増やす方向（hosted tool 系の `web_search_call` 等）であり、
allowlist では追随漏れした種別が 1 応答の途中でグループを分断して 1 応答が 2 ターンとして数えられる
ためである。この 2 つが必要なのは、`user_text` が載せるのは role が `user` の最新テキストであり、
tool 実行結果を受けた次のターンでも `user_text` が変わらないためである。`user_text` だけで分岐する
ルール関数は同じ ToolCall を返し続け `max_turns` まで回る。

`user_text` は user 由来のテキストだけを載せる `str` とし、抽出できない場合は空文字列にする。生の
入力が要る場合は `input` から取る。入力全体の文字列表現を載せると、tool 実行結果や system 文言と
いった非 user 由来のデータが `user_text` 経由で `if "..." in request.user_text` 形の遷移判定へ
流れ込む（信頼境界の越境）ためである。

### 4. ルール関数の契約

同期関数と async 関数の双方を受理し、戻り値が awaitable なら `inspect.isawaitable` で分岐して await
する（`AgentRegistry._build_dynamic_handoff` の resolver と同型）。`None` の返却は「応答を決定できない」
の通知手段で、空テキスト応答となり run は正常終了する。ルール関数が送出した例外は握り潰さず伝播
させ、空応答へ差し替えない。

`get_response` は SDK のバージョン差に耐えるためキーワード優先 + 位置フォールバックで引数を正規化する
（現行 SDK は `get_response` を全キーワードで、`stream_response` を 7 位置引数 + 3 キーワード専用で
呼ぶ）。SDK は retry ラッパ経由で呼ぶため、利用者が `ModelRetryPolicy` を併用した構成ではルール関数の
例外も再試行対象になりうる。これは利用者が明示的に有効化した SDK ネイティブ機構の作用であり、lib 側で
例外を握らない契約は変わらない。

### 5. `stream_response` は post-execution streaming で対応する

`stream_response` は `WorkflowModel.stream_response` と同型の **async generator** として実装する。
`get_response` を回して応答を確定させ、テキストが非空ならテキスト delta を流し、最後に終端イベント
（`ResponseCompletedEvent`）を yield する。ツール呼び出しのみの応答では delta が流れず終端イベントのみ
になる（`DeterministicToolCallModel` と同じ）。区切り規則（擬似トークン長）と `item_id` の採番は lib の
内部決定であり、利用者ルール関数へは課さない。

これにより lib 内の決定的 Model 3 種すべてが同じ post-execution streaming 方式で揃い、`Runner.run` と
`Runner.run_streamed` のどちらでも公開モデルが使える。

正直な制約として、これは**実際の逐次生成ではなく、完成済みの応答を一定長で区切って流す**方式である
（応答はルール関数が返した時点で確定しており、delta は進捗を表さない）。この性質は `WorkflowModel` と
同一で、docstring と利用者向けドキュメントに明記する。

### 6. 応答ビルダは 5 種（テキスト / usage 付き / 単一 ToolCall / 複数 ToolCall / 混在）

1 応答にアシスタントテキストとツール呼び出しを混在させる `mixed_response` を含めて 5 種を公開する。

```
mixed_response(
    text: str,
    calls: Sequence[tuple[str, str, str]],   # (tool_name, arguments, call_id)
    *, total_tokens: int = 0, requests: int = 0,
) -> ModelResponse
```

`output` はテキストメッセージ 1 件のあとに宣言順のツール呼び出しを並べる。既存 4 種は多数のテスト
モジュールが使うため残し、`mixed_response` はそれらと同じ共有 item ヘルパへ委譲する。

### 7. 公開窓口は `oai_agentspec.runtime.deterministic`（通常の再エクスポート）

`runtime/deterministic/__init__.py` は実装本体を持たず、直 import と `__all__` 宣言のみで構成する
（`runtime/guardrails` と同型）。`__all__` は `DeterministicResponseModel` / `ModelRequest` /
応答ビルダ 5 種の計 7 件で、run 実行関数・実行ループ・再試行機構は含めない。extra は宣言せず、
コア `__all__` には 1 件も載せない。`ModelRequest` を掲載するのは、ルール関数の引数へ型注釈を
付けられ、エディタの属性補完でフィールドを発見できるようにするためである。非掲載にすると
フィールドの発見手段がドキュメントだけになる。

### 8. 命名と既定 id 値

クラス名は `DeterministicResponseModel`（コンストラクタ引数名 `rule`）、ビルダは `text_response` /
`text_response_with_usage` / `tool_call_response` / `multi_tool_call_response` / `mixed_response`、
窓口名は `deterministic` とする。既定 id 値は 5 件で、SDK の id 接頭辞慣行を守りつつ由来がデバッグ時に
読める値にする。定数として宣言しつつ、テストではリテラルを pin する（定数のリネームで pin が空振り
しないため）。

| 対象 | 既定値 |
|---|---|
| `ResponseOutputMessage.id` | `msg_deterministic` |
| `ResponseFunctionToolCall.id` | `fc_deterministic`（複数指定できる版は `fc_<call_id>`） |
| `ResponseFunctionToolCall.call_id` | `call_deterministic` |
| stream 終端 `Response.id` | `resp_deterministic` |
| stream 終端 `Response.model` | `oai-agentspec-deterministic` |

却下した名称: `ScriptedModel`（キュー消費を含意）/ `RuleBasedModel`（ルールベース推論と誤読）/
`StubModel`（テスト用途の含意）/ `OfflineModel`（4 用途のうち 1 つに寄る）/ 窓口名 `responses`
（`_adapters/responses.py` と同名で層の混同）/ `models`（SDK `agents.models` と紛らわしい）/
`testing`・`stubs`（窓口パスがテスト専用を含意）/ 既定 id の `msg_agentspec` 系（dist 名変更で
陳腐化）/ `msg_stub`（テスト用途の含意）。

### 9. テストヘルパのビルダは公開版へ一本化する

`tests/_helpers/responses.py` を削除し、テスト側の import を公開窓口へ差し替える。公開版との差分は
第 1 引数の `name` -> `tool_name` 改名と `call_id` のキーワード専用化の 2 点で、既存呼び出しは全件
`call_id=` をキーワードで渡しており `name=` のキーワード呼び出しが無いため呼び出し行は書き換えない。
ステートフルな `FakeModel` / `ChoiceAwareModel` は公開せず残置する（import 元のみ差し替える）。

副次効果として公開ビルダが多数のテストモジュールで実使用され、公開 API の妥当性が dogfooding で
担保される。差し替え中に id 値・構造への assert 依存が見つかった場合に限り併存へ切り替える。

## Consequences

- + 利用者が実 API 抜きで `Runner.run` / `Runner.run_streamed` を完走させられる決定的モデルを、
  自作せずに公開シンボルとして使えるようになる。ステートレスのため同一インスタンスを複数 run・
  複数 Agent で共有でき、応答は呼び出し回数・順序に依存しない。
- + 応答オブジェクトの構築に `from agents` / `from openai` を書かずに済む（SDK 隔離の利益が利用者側へ
  も及ぶ）。
- + SDK item 構築と streaming イベント生成が 1 箇所へ集約され、SDK の必須フィールド追加時の追随点が
  1 つになる。ワークフロー経路の応答内容・固定 id 値・イベント列は不変のまま保たれる。
- + 追加依存ゼロ・extra 宣言なしで、コア `__all__` の宣言層限定という原則も保たれる。
- - 決定的 Model が 3 つ（内部 2 + 公開 1）並ぶ。責務差を `docs/architecture.md` に記述して区別する。
- - streaming は post-execution であり進捗的ではない（応答はルール関数が返した時点で確定している）。
  実 LLM の逐次生成と同じ体感を得る用途には使えない。
- - 既存の streaming ヘルパ 2 つの id 値が引数化され、呼び出し側が値を明示する責務を持つ。値の渡し
  忘れは既定値を持たせないことで防ぐ（ワークフロー用の値はハードコードのまま既存呼び出しが渡す）。
- - 「テスト専用を含意しない」命名の結果として、本番の LLM 呼び出し経路の代替として運用構成へ残る
  誤用リスクが上がる。利用者向けドキュメントに当該注意を明記して緩和する。
- - `ModelRequest` を公開するため、そのフィールド集合が公開契約になる。フィールドの削除・改名は
  破壊的変更になり、追加のみが非破壊である。
- - 既定 id 値の pin とテストヘルパ削除により、テスト側の import 経路が公開窓口へ結合する。

## Confirmation

強制手段として次のテストを置き、`docs/QUALITY-GUARANTEES.md` へ登録する（source = ADR-0019）。
個別 assert とテスト名の確定はテスト実装時に行い、一次情報は各テストの docstring とする。

- 同一インスタンスの再実行と複数 Agent 共有で応答が変わらないこと:
  `tests/_adapters/test_deterministic_model_l2.py`
- `Runner.run_streamed` でテキスト delta と終端イベントが流れ、ツール呼び出しのみの応答では終端
  イベントのみになること: `tests/_adapters/test_deterministic_model_l2.py`
- `turn` で分岐するルール関数で「tool 呼び出し -> 最終応答」の 2 ターンが完走し `max_turns` に達しない
  こと（無限ループ回帰の pin）: `tests/_adapters/test_deterministic_model_l2.py`
- 公開ビルダと公開 streaming イベントの既定 id 値 5 件が、いずれも禁止語を含まない固定値であること:
  `tests/_adapters/test_deterministic_builders_l1.py`
- 既存ワークフロー用ビルダの id 値と streaming イベントの id / model が不変であること
  （委譲統合・ヘルパ引数化の回帰 pin）: `tests/_adapters/test_deterministic_builders_l1.py`
- `mixed_response` が 1 応答へテキストメッセージと宣言順のツール呼び出しを載せること:
  `tests/_adapters/test_deterministic_builders_l1.py`
- 公開窓口の `__all__` が 7 件のみであること: `tests/runtime/deterministic/test_init_l1.py`
- 公開面（コア `__all__` と `runtime/` 全窓口の `__all__`）に `Fake` / `Mock` / `Dummy` を含む名前が
  無く、コア `__all__` に本機能のシンボルが無いこと: `tests/test_public_naming_l1.py`
