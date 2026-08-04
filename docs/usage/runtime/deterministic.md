# 決定的応答モデル（DeterministicResponseModel / 応答ビルダ）

## 何を解決するか

`DeterministicResponseModel` は、実 API を呼ばずに「入力からルール関数が応答を決める」SDK `Model` 実装です。`AgentSpec(model=...)` / `agents.Agent(model=...)` へ入れれば、ネットワークへ到達できない環境でも `Runner.run` / `Runner.run_streamed` が完走します。

内部キューを消費しないステートレス実装のため、同一インスタンスを複数 run・複数 Agent（handoff 元と handoff 先など）で共有しても応答は呼び出し回数・順序に依存しません。

応答ビルダ 5 種は、モデルが返す応答オブジェクトを `from agents` / `from openai` を書かずに組み立てるための純関数です。

想定する 4 用途:

1. 自動テスト
2. 実 API を呼ばないオフライン開発
3. デモ・サンプル実行
4. 決定的なシナリオ再生

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| `text_response` | 単一のアシスタントテキスト | 最終出力を返すだけ |
| `text_response_with_usage` | usage 付きテキスト | 累積使用量の判定（run 予算超過など）を動かしたい |
| `tool_call_response` | 単一 function ToolCall | tool 実行 / handoff を誘発したい |
| `multi_tool_call_response` | 複数 ToolCall（`call_id` 個別指定） | 1 応答で複数呼び出しを出したい |
| `mixed_response` | テキスト + ToolCall の混在（usage も指定可） | 「一言返してから tool を呼ぶ」「発話しながらハンドオフする」 |
| ルール関数が `None` を返す | 空テキスト応答 | 応答を決められないケースを正常終了させたい |
| 同期のルール関数 / async のルール関数 | どちらも受理（awaitable は await して解決） | 判定に I/O を挟むかどうかで選ぶ |

## 使い方

- import: `from oai_agentspec.runtime.deterministic import (DeterministicResponseModel, text_response, text_response_with_usage, tool_call_response, multi_tool_call_response, mixed_response)`
- extras: なし（追加依存なし）
- 依存 env: なし

```python
from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.deterministic import (
    DeterministicResponseModel,
    text_response,
    tool_call_response,
)


def rule(request):
    """入力から応答を決める純関数（同期 / async どちらでもよい）。"""
    # 多ターンは turn で分岐する（下記「落とし穴」を参照）
    if request.turn == 0 and "予約" in request.user_text:
        return tool_call_response("transfer_to_booking", "{}")
    return text_response(f"echo: {request.user_text}")


model = DeterministicResponseModel(rule)   # ステートレス: 複数 Agent / 複数 run で共有可

registry = AgentRegistry()
registry.register(AgentSpec("triage", "受付", model=model, handoffs=["booking"]))
registry.register(AgentSpec("booking", "予約", model=model))

# 実行は利用者責務（build-don't-run）
# result = await Runner.run(registry.get("triage"), "予約したい")
```

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `DeterministicResponseModel.__init__`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `rule` | `Callable[[ModelRequest], Any]` | 必須 | 応答を決めるルール関数。戻り値は応答オブジェクト / `None` / awaitable |

### `ModelRequest`（frozen・ルール関数の引数）

| 属性 | 型 | 説明 |
|---|---|---|
| `system_instructions` | `str \| None` | Agent の instructions |
| `input` | `Any` | SDK が渡した入力（str または input item のリスト） |
| `user_text` | `str` | 入力から抽出した直近の user テキスト。user 由来テキストのみを載せ、抽出できない場合は空文字列（生の入力は `input` から取得する） |
| `turn` | `int` | 入力（Session 併用時は履歴を含む）に含まれるモデル応答の件数（1 モデル応答 = 1 ターン）。`input` から導出。Session を使う構成では前ターンまでの応答も数に入るため、run ごとの初回は 0 になりません |
| `tool_outputs` | `tuple[Any, ...]` | 入力中の tool 実行結果アイテムの列。`input` から導出 |
| `model_settings` | `Any` | `agents.ModelSettings`（`tool_choice` 等で分岐したい場合に使う） |
| `tools` | `tuple[Any, ...]` | 提示されている Tool |
| `handoffs` | `tuple[Any, ...]` | 提示されている Handoff |
| `output_schema` | `Any` | 構造化出力スキーマ（未指定なら `None`） |

`ModelRequest` は公開窓口の `__all__` に載せているので、`def rule(request: ModelRequest) -> ModelResponse:` のように型注釈を付けられます。

### 応答ビルダ

- `text_response(text: str) -> ModelResponse`
- `text_response_with_usage(text: str, *, total_tokens: int, requests: int = 1) -> ModelResponse`
- `tool_call_response(tool_name: str, arguments: str = "{}", *, call_id: str = ...) -> ModelResponse`
- `multi_tool_call_response(calls: Sequence[tuple[str, str, str]]) -> ModelResponse` — 要素は `(tool 名, 引数 JSON, call_id)`
- `mixed_response(text: str, calls: Sequence[tuple[str, str, str]], *, total_tokens: int = 0, requests: int = 0) -> ModelResponse` — テキストメッセージ 1 件の後ろに宣言順の ToolCall を並べる

```python
# 「一言返してからハンドオフする」応答
return mixed_response("担当へおつなぎします", [("transfer_to_booking", "{}", "call_1")])
```

既定 id 値は 5 件です。

| 対象 | 既定値 |
|---|---|
| メッセージ id | `msg_deterministic` |
| ToolCall のアイテム id | `fc_deterministic`（`multi_tool_call_response` / `mixed_response` は `fc_<call_id>`） |
| `call_id` | `call_deterministic` |
| ストリーミング終端の response id | `resp_deterministic` |
| ストリーミング終端の model 名 | `oai-agentspec-deterministic` |

## 判断軸

- 実 API を呼ばずに動かしたいだけなら **`DeterministicResponseModel`**。ワークフローの決定論起動が目的なら `FacadeMode.DETERMINISTIC`（[core/workflow](../core/workflow.md)）
- 応答を「入力で分岐させたい」なら `turn` / `tool_outputs` / `model_settings` を見るルール関数、「常に同じ 1 応答でよい」なら定数を返すルール関数で足ります
- ルール関数は**副作用を持たない純関数**として書きます（同じ入力で常に同じ応答になることが、この機能の価値そのものです）

## 落とし穴

- **多ターンは `turn` で分岐します**。`user_text` だけで分岐すると tool 結果を受けた次のターンでも同じ ToolCall を返し続け、`max_turns` に達するまで無限ループになります。tool 実行結果は role が `user` のアイテムではないため、tool 呼び出しの前後で `user_text` は変わりません。戻り値や `call_id` で分岐したい場合は `tool_outputs` を見てください
- **`tool_outputs` に載るアイテムは tool 名を持ちません**。SDK が載せる `function_call_output` アイテムのフィールドは `call_id` / `output` / `type`（+ `id` / `status`）だけで、tool 名は `request.input` 側の `function_call` アイテムにしかありません。したがって `tool_outputs` の絞り込みは **`call_id` で行うのが正典**です。tool 名で絞りたい場合は 2 段階の手順を踏みます: (1) `request.input` を走査して `type == "function_call"` のアイテムから `name` -> `call_id` の対応を作る、(2) その `call_id` で `tool_outputs` を絞る
- **`tool_outputs` にはハンドオフ（`transfer_to_*`）の結果も載ります**。ハンドオフも SDK 上は関数呼び出しだからです。`if request.tool_outputs:` のように無条件で分岐すると、ハンドオフ後の応答まで tool 分岐が乗っ取ります。tool 実行で分岐するときは `call_id` で絞り込んでください（応答ビルダは `call_id` を指定できます）
- **tool 実行とハンドオフを併用する場合は、呼び出しごとに一意な `call_id` を指定してください**。`tool_call_response` の既定 `call_id` は `call_deterministic` という、全呼び出しで共有される単一の固定値です。1 つのルール関数が tool 呼び出しと `transfer_to_*` の双方を既定値のまま発行すると、両方の `function_call_output` が同じ `call_id` を持ち、`call_id` による絞り込みが判別能力を失います。`multi_tool_call_response` / `mixed_response` は `call_id` が必須引数なので、この穴はありません
- **`input` は必ず list で渡してください**。単体 dict（`Runner.run(agent, input={"role": ..., "content": ...})`）を渡すと list へ正規化できず、`user_text` が空文字列・`turn` が 0・`tool_outputs` が空になります。**例外も警告も出ない**ため、気づけるのはルール関数の分岐が想定と違う挙動をしたときだけです（`None` や非 iterable を渡した場合も同じ空値になります）
- Session（`runtime.conversation` / `Runner.run(session=...)`）併用時に run 単位で分岐したい場合は、`turn` の絶対値に依存せず `tool_outputs` または `input` を見てください。Session 併用時の `turn` にはセッション履歴中の応答も数に入るため、run ごとの初回でも 0 になりません
- ストリーミング（`Runner.run_streamed`）は **post-execution streaming** です。ルール関数が返した完成済みの応答を一定長で区切って delta として流すため、delta は進捗を表しません（実 LLM の逐次生成と同じ体感にはなりません）。ツール呼び出しのみの応答では delta が流れず終端イベントのみになります
- ルール関数が送出した例外は握り潰されず伝播します。ただし `ModelRetryPolicy`（[safety/resilience](../safety/resilience.md)）を併用した構成では、SDK が retry ラッパ経由でモデルを呼ぶためルール関数の例外も**再試行対象になりえます**。ルール関数を副作用のない純関数に保てば、再試行されても結果は変わりません
- **本番の LLM 呼び出し経路の代替として運用構成へ残さないでください**。名称がテスト専用を含意しないぶん、実 API を呼ぶべき経路に決定的モデルが残ったまま気付かれない誤用リスクがあります。運用構成では実モデルへ差し替える手順を用意してください
- ルール関数が `None` を返すと空テキスト応答になり、run は例外なく正常終了します（`result.final_output` が空文字列）。「決められなかった」を例外で知りたい場合はルール関数側で送出してください

## 参照

- 詳細設計: `docs/architecture.md`（決定的応答モデル節）
- 具体例: `examples/deterministic/01_rule_model.py`（基本・ステートレス性・`None`・async・例外伝播）/ `examples/deterministic/02_multi_turn_and_handoff.py`（`turn` 分岐による tool 実行と `mixed_response` ハンドオフ）/ `examples/deterministic/03_streaming.py`（`run_streamed` の差分イベント）/ `examples/deterministic/04_tool_and_handoff_in_one_rule.py`（tool とハンドオフ併用時の落とし穴と絞り込み）
- 設計判断: `docs/adr/0019-deterministic-response-model.md`

## 次

本ページでガイドは終わりです。より深い設計・不変条件は [docs/architecture.md](../../architecture.md) を参照してください。
