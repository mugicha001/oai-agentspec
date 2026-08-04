"""L2: `DeterministicResponseModel` を実 Agent + Runner で回す契約の pin（Issue #70・FR-4）。

入力からルール関数が応答を決めるステートレス Model が、実 API へ接続せずに
`Runner.run` / `Runner.run_streamed` を完走させることを、実 `agents.Agent` と実 `Runner`
で検証する（L2）。実行は `tests/conftest.py` の autouse fixture `_no_external_calls`
（`OPENAI_API_KEY` 削除・トレーシング無効化・ループバック以外の TCP 遮断）下で行われる
ため、外部接続が起きればテストは失敗する（NFR-7）。

pin する契約:

- ルール関数の応答が `final_output` に反映され、同一インスタンスの再実行・複数 Agent 共有で
  結果が変わらない（回数依存の状態が混入しても run は成功し続けるため、挙動差でしか
  検知できない）。
- ルール関数の戻り値: `ModelResponse` はそのまま / awaitable は await して解決 / `None` は
  空テキスト応答 / 例外は握り潰さず伝播。
- `ModelRequest` は入力から純粋に導出される（`user_text` は常に `str`、`turn` は入力中の
  モデル応答件数、`tool_outputs` は入力中の `function_call_output` 列）。`turn` で分岐する
  ルール関数が「tool 呼び出し -> 最終応答」の 2 ターンで完走することは無限ループ回帰の pin。
- SDK は `get_response` を全キーワードで、`stream_response` を 7 位置引数 + 3 kw-only で
  呼ぶため、`get_response` は位置引数・キーワード引数の双方で同じ正規化結果になる。
- streaming は post-execution streaming（テキストがあれば差分イベント列 -> 終端イベント、
  ToolCall のみなら終端イベントのみ）で、イベント識別子に禁止語を含まない。

実装（`_adapters/deterministic.py`）が存在しない間は、各テストが `ModuleNotFoundError` で
失敗する（RED）。
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest
from agents import Agent, ModelSettings, Runner, function_tool
from openai.types.responses import ResponseCompletedEvent, ResponseTextDeltaEvent

if TYPE_CHECKING:
    from oai_agentspec._adapters.deterministic import ModelRequest

pytestmark = pytest.mark.integration


def _deterministic() -> ModuleType:
    """決定的応答モデルのモジュールを取得する（未実装の間は ImportError で RED になる）。

    module レベル import にすると実装不在が collection error になり `uv run pytest` 全体が
    中断して他テストの回帰が見えなくなるため、各テストから遅延 import する。

    Returns:
        `oai_agentspec._adapters.deterministic` モジュール。
    """
    return importlib.import_module("oai_agentspec._adapters.deterministic")


class _RuleError(RuntimeError):
    """ルール関数が送出する例外（伝播の pin 用）。"""


class _RuleRecorder:
    """ルール関数へ渡された `ModelRequest` を記録しつつ応答を返す。

    Attributes:
        requests: 呼び出しごとの `ModelRequest`（呼び出し順）。
    """

    def __init__(self, responder: Callable[[ModelRequest], Any]) -> None:
        """記録付きルール関数を作る。

        Args:
            responder: `ModelRequest` から応答を決める関数。
        """
        self.requests: list[ModelRequest] = []
        self._responder = responder

    def __call__(self, request: ModelRequest) -> Any:
        """`ModelRequest` を記録して応答を返す。"""
        self.requests.append(request)
        return self._responder(request)

    @property
    def turns(self) -> list[int]:
        """記録した `ModelRequest` の `turn` 列。"""
        return [request.turn for request in self.requests]


@function_tool
def add_one(x: int) -> int:
    """1 を足す（tool 誘発の検証用）。

    Args:
        x: 入力値。

    Returns:
        `x + 1`。
    """
    return x + 1


def _echo_rule(request: ModelRequest) -> Any:
    """user テキストをそのまま返すルール関数。"""
    return _deterministic().text_response(f"echo: {request.user_text}")


def _tool_then_text(tool_name: str, arguments: str = "{}") -> Callable[[ModelRequest], Any]:
    """初回は ToolCall、以降はテキストを返すルール関数を作る。

    Args:
        tool_name: 初回に呼ぶ tool 名。
        arguments: tool 引数の JSON 文字列。

    Returns:
        `turn` で分岐するルール関数。
    """

    def rule(request: ModelRequest) -> Any:
        module = _deterministic()
        if request.turn == 0:
            return module.tool_call_response(tool_name, arguments)
        return module.text_response(f"turn{request.turn}: {request.user_text}")

    return rule


def _constant_text(text: str) -> Callable[[ModelRequest], Any]:
    """常に同じテキスト応答を返すルール関数を作る。

    Args:
        text: 返すテキスト。

    Returns:
        入力に依存しないルール関数。
    """
    return lambda request: _deterministic().text_response(text)


def _raw_events(events: list[Any]) -> list[Any]:
    """`stream_events` から raw イベントの data を取り出す。

    Args:
        events: `stream_events` が流したイベント列。

    Returns:
        `raw_response_event` の data 列。
    """
    return [event.data for event in events if event.type == "raw_response_event"]


def _output_of(item: Any) -> Any:
    """`function_call_output` アイテムから output 値を取り出す（dict / 属性の双方に対応）。"""
    if isinstance(item, dict):
        return item.get("output")
    return getattr(item, "output", None)


def _call_id_of(item: Any) -> Any:
    """`function_call_output` アイテムから call_id を取り出す（dict / 属性の双方に対応）。"""
    if isinstance(item, dict):
        return item.get("call_id")
    return getattr(item, "call_id", None)


# ---------------------------------------------------------------------------
# Runner.run 完走とステートレス性
# ---------------------------------------------------------------------------
async def test_ルール関数の応答が_Runner_run_の最終出力に反映される() -> None:
    """ルール関数を渡したモデルは実 API へ接続せず run を完走し、応答が final_output になる。"""
    model = _deterministic().DeterministicResponseModel(_echo_rule)
    agent = Agent(name="a", instructions="指示文", model=model)

    result = await Runner.run(agent, input="hello")

    assert result.final_output == "echo: hello"


async def test_同一インスタンスの再実行と複数_Agent_共有で応答が変わらない() -> None:
    """同一インスタンスを 2 回 run しても、複数 Agent で共有しても応答はルール関数だけで決まる。

    回数依存の状態（キュー消費・呼び出しカウンタ）が混入しても run は成功し続けるため、
    挙動差でしか検知できない（ADR 0019 / QUALITY-GUARANTEES）。
    """
    model = _deterministic().DeterministicResponseModel(_echo_rule)
    agent = Agent(name="a", instructions="指示文", model=model)

    first = await Runner.run(agent, input="hello")
    second = await Runner.run(agent, input="hello")

    assert first.final_output == second.final_output == "echo: hello"

    # 同一インスタンスを handoff 元 / 先の 2 Agent へ渡しても、応答は turn 判定だけで決まる
    # （呼び出し順序に依存しない）。
    shared = _deterministic().DeterministicResponseModel(_tool_then_text("transfer_to_b"))
    target = Agent(name="b", instructions="b エージェント", model=shared)
    source = Agent(name="a", instructions="a エージェント", model=shared, handoffs=[target])

    handoff_first = await Runner.run(source, input="ルーティングして", max_turns=2)
    handoff_second = await Runner.run(source, input="ルーティングして", max_turns=2)

    assert handoff_first.final_output == "turn1: ルーティングして"
    assert handoff_first.last_agent.name == "b"
    assert handoff_second.final_output == handoff_first.final_output
    assert handoff_second.last_agent.name == "b"


async def test_ルール関数が返した_ToolCall_が_tool_を誘発する() -> None:
    """ToolCall を返すと tool が実行され、その結果を受けた次ターンで最終応答になる。"""
    recorder = _RuleRecorder(_tool_then_text("add_one", '{"x": 1}'))
    model = _deterministic().DeterministicResponseModel(recorder)
    agent = Agent(name="a", instructions="指示文", model=model, tools=[add_one])

    result = await Runner.run(agent, input="足して", max_turns=2)

    assert result.final_output == "turn1: 足して"
    assert [_output_of(item) for item in recorder.requests[1].tool_outputs] == ["2"]


async def test_ルール関数が返した_ToolCall_が_handoff_を誘発する() -> None:
    """handoff tool 名の ToolCall を返すと遷移先エージェントが最終回答者になる。"""
    model = _deterministic().DeterministicResponseModel(_tool_then_text("transfer_to_b"))
    target = Agent(name="b", instructions="b エージェント", model=model)
    source = Agent(name="a", instructions="a エージェント", model=model, handoffs=[target])

    result = await Runner.run(source, input="任せる", max_turns=2)

    assert result.last_agent.name == "b"
    assert result.final_output == "turn1: 任せる"


async def test_turn_で分岐すると_tool_呼び出しから最終応答までの_2_ターンが完走する() -> None:
    """`turn` 分岐なら tool 結果を受けた 2 ターン目で終わり `max_turns` に達しない。

    `user_text` は tool 呼び出しの前後で変わらないため、`user_text` だけで分岐すると同じ
    ToolCall を返し続けて `max_turns` 例外まで回る。その無限ループの回帰 pin として
    `max_turns=2`（= モデル呼び出し 2 回）で完走することを固定する。
    """
    recorder = _RuleRecorder(_tool_then_text("add_one", '{"x": 41}'))
    model = _deterministic().DeterministicResponseModel(recorder)
    agent = Agent(name="a", instructions="指示文", model=model, tools=[add_one])

    result = await Runner.run(agent, input="計算して", max_turns=2)

    assert result.final_output == "turn1: 計算して"
    assert recorder.turns == [0, 1]


# ---------------------------------------------------------------------------
# ルール関数の戻り値と例外
# ---------------------------------------------------------------------------
async def test_ルール関数が_None_を返すと空テキストで正常終了する() -> None:
    """`None` は空テキスト応答として扱われ、run は例外なく終わり final_output は空文字。"""
    model = _deterministic().DeterministicResponseModel(lambda request: None)
    agent = Agent(name="a", instructions="指示文", model=model)

    result = await Runner.run(agent, input="hello")

    assert result.final_output == ""


async def test_ルール関数が_awaitable_を返すと_await_して解決される() -> None:
    """async 関数のルール関数でも await して `ModelResponse` を取り出す。"""

    async def rule(request: ModelRequest) -> Any:
        return _deterministic().text_response(f"async: {request.user_text}")

    model = _deterministic().DeterministicResponseModel(rule)
    agent = Agent(name="a", instructions="指示文", model=model)

    result = await Runner.run(agent, input="hello")

    assert result.final_output == "async: hello"


async def test_ルール関数の例外はそのまま伝播する() -> None:
    """ルール関数の例外は握り潰されず、空応答へ差し替えられない。

    `model_settings.retry` 未設定の既定構成で pin する（SDK は `get_response` を
    `get_response_with_retry` 経由で呼ぶため、`runtime/resilience` の再試行ポリシーを
    併用した構成では例外が再試行対象になりうる）。
    """

    def rule(request: ModelRequest) -> Any:
        raise _RuleError("ルール関数の失敗")

    model = _deterministic().DeterministicResponseModel(rule)
    agent = Agent(name="a", instructions="指示文", model=model)

    with pytest.raises(_RuleError, match="ルール関数の失敗"):
        await Runner.run(agent, input="hello")


# ---------------------------------------------------------------------------
# ModelRequest
# ---------------------------------------------------------------------------
def test_ModelRequest_のフィールド集合は_9_件で不変である() -> None:
    """ルール関数へ渡すフィールドを固定する（公開面なので増減は契約変更）。"""
    field_names = {field.name for field in dataclasses.fields(_deterministic().ModelRequest)}

    assert field_names == {
        "system_instructions",
        "input",
        "user_text",
        "turn",
        "tool_outputs",
        "model_settings",
        "tools",
        "handoffs",
        "output_schema",
    }


async def test_ModelRequest_は_frozen_で書き換えられない() -> None:
    """ルール関数からの書き換えを防ぐ frozen dataclass である。"""
    recorder = _RuleRecorder(_echo_rule)
    model = _deterministic().DeterministicResponseModel(recorder)
    await model.get_response(system_instructions=None, input="hello")

    request = recorder.requests[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.turn = 99  # type: ignore[misc]


async def test_run_時の_tools_handoffs_model_settings_が_ModelRequest_へ渡る() -> None:
    """SDK が渡す実行時パラメータが plain な tuple / 不透明値として載る。"""
    recorder = _RuleRecorder(_echo_rule)
    model = _deterministic().DeterministicResponseModel(recorder)
    target = Agent(name="b", instructions="b エージェント")
    source = Agent(name="a", instructions="指示文", model=model, tools=[add_one], handoffs=[target])

    await Runner.run(source, input="hello")

    request = recorder.requests[0]
    assert request.system_instructions == "指示文"
    assert request.input == [{"content": "hello", "role": "user"}]
    assert request.user_text == "hello"
    assert request.turn == 0
    assert request.tool_outputs == ()
    assert isinstance(request.tools, tuple)
    assert [getattr(tool, "name", None) for tool in request.tools] == ["add_one"]
    assert isinstance(request.handoffs, tuple)
    assert [getattr(item, "agent_name", None) for item in request.handoffs] == ["b"]
    assert request.model_settings is not None
    assert request.output_schema is None


async def test_user_text_は抽出できない入力では空文字列になる() -> None:
    """user テキストを抽出できない入力では `user_text` を空文字列にする。

    入力全体の文字列表現を載せると、tool 実行結果や system 文言といった非 user 由来の
    データが `user_text` 経由で遷移判定（`if "..." in request.user_text` 形）へ流れ込む。
    生の入力が要る場合は `input` から取得する。
    """
    recorder = _RuleRecorder(_echo_rule)
    model = _deterministic().DeterministicResponseModel(recorder)

    await model.get_response(system_instructions=None, input=[{"role": "system", "content": "x"}])

    request = recorder.requests[0]
    assert request.user_text == ""
    # `system` はターン境界（user / tool 側の発話）なのでモデル応答として数えない。
    assert request.turn == 0
    # 生の入力は input から取れる（情報は失われない）
    assert request.input == [{"role": "system", "content": "x"}]


async def test_developer_ロールの_item_はモデル応答として数えない() -> None:
    """`developer` ロールはターン境界（user / tool 側の発話）で `turn` を進めない。

    境界ロール集合から漏れると assistant 由来と見なされ、初回入力でも `turn` が 1 になり、
    `if request.turn == 0:` を前提にしたルール関数の初回分岐が静かに外れる。
    """
    recorder = _RuleRecorder(_echo_rule)
    model = _deterministic().DeterministicResponseModel(recorder)

    await model.get_response(
        system_instructions=None,
        input=[
            {"role": "developer", "content": "開発者向け指示"},
            {"role": "user", "content": "hello"},
        ],
    )

    request = recorder.requests[0]
    assert request.turn == 0
    assert request.user_text == "hello"


async def test_user_text_に_tool_実行結果が混入しない() -> None:
    """tool 実行結果（外部由来）が `user_text` へ載らないことを固定する。

    `user_text` は user 由来テキストのみを載せる契約。tool 出力が載ると、外部システムの
    戻り値でハンドオフ分岐が成立しうる（信頼境界の越境）。
    """
    recorder = _RuleRecorder(_echo_rule)
    model = _deterministic().DeterministicResponseModel(recorder)

    await model.get_response(
        system_instructions=None,
        input=[
            {"role": "user", "content": [{"type": "input_image", "image_url": "x"}]},
            {"type": "function_call_output", "call_id": "c1", "output": "TOOL-SECRET"},
        ],
    )

    assert "TOOL-SECRET" not in recorder.requests[0].user_text


async def test_user_text_は_content_のテキストパートを宣言順に連結する() -> None:
    """content が parts 形の user メッセージは、テキストパートを宣言順に連結して載せる。

    Responses API の user メッセージは `input_text` / `text` の 2 系統でテキストを運ぶ。
    片方だけを拾うと、その形で届いた発話が `user_text` から丸ごと欠落し、抽出不能
    （空文字列）と区別できなくなる。
    """
    recorder = _RuleRecorder(_echo_rule)
    model = _deterministic().DeterministicResponseModel(recorder)

    await model.get_response(
        system_instructions=None,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            }
        ],
    )

    assert recorder.requests[0].user_text == "ab"


async def test_turn_は入力中のモデル応答件数と一致し同じ入力なら同じ値になる() -> None:
    """`turn` は input から純粋に導出する（呼び出し回数カウンタではない）。"""
    recorder = _RuleRecorder(_echo_rule)
    model = _deterministic().DeterministicResponseModel(recorder)

    first_turn = [{"role": "user", "content": "1"}]
    second_turn = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "2"},
    ]
    after_tool = [
        {"role": "user", "content": "1"},
        {
            "type": "function_call",
            "call_id": "call_a",
            "name": "add_one",
            "arguments": "{}",
            "id": "fc_a",
        },
        {"type": "function_call_output", "call_id": "call_a", "output": "2"},
    ]

    await model.get_response(system_instructions=None, input="hello")
    await model.get_response(system_instructions=None, input=first_turn)
    await model.get_response(system_instructions=None, input=second_turn)
    await model.get_response(system_instructions=None, input=after_tool)
    # 同じ入力を再度渡しても値は変わらない（呼び出し回数に依存しない）。
    await model.get_response(system_instructions=None, input=first_turn)

    assert recorder.turns == [0, 0, 1, 1, 0]
    # `user_text` は入力中の**直近**の user メッセージ（先頭の "1" ではない）。
    assert recorder.requests[2].user_text == "2"


async def test_turn_は複数_item_の応答でも_1_ターンとして数える() -> None:
    """テキスト + ToolCall を 1 応答で返しても `turn` は 1 だけ進む（1 応答 = 1 ターン）。

    `mixed_response` は 1 応答へ message と function_call を並べる。item 単位で数えると
    ルール関数の `turn` 分岐が 1 ターンで 2 進み、利用者の分岐条件が静かに外れる。
    """

    def rule(request: ModelRequest) -> Any:
        module = _deterministic()
        if request.turn == 0:
            return module.mixed_response("少々お待ちください", [("add_one", '{"x": 1}', "call_a")])
        if request.turn == 1:
            return module.tool_call_response("add_one", '{"x": 2}', call_id="call_b")
        return module.text_response(f"turn{request.turn}")

    recorder = _RuleRecorder(rule)
    model = _deterministic().DeterministicResponseModel(recorder)
    agent = Agent(name="a", instructions="指示文", model=model, tools=[add_one])

    result = await Runner.run(agent, input="hello", max_turns=3)

    assert result.final_output == "turn2"
    assert recorder.turns == [0, 1, 2]


async def test_tool_outputs_は入力中の_function_call_output_を列挙する() -> None:
    """`tool_outputs` は入力中の tool 実行結果を宣言順の tuple で載せる。"""
    recorder = _RuleRecorder(_echo_rule)
    model = _deterministic().DeterministicResponseModel(recorder)
    model_input = [
        {"role": "user", "content": "hello"},
        {
            "type": "function_call",
            "call_id": "call_a",
            "name": "add_one",
            "arguments": "{}",
            "id": "fc_a",
        },
        {"type": "function_call_output", "call_id": "call_a", "output": "2"},
        {
            "type": "function_call",
            "call_id": "call_b",
            "name": "add_one",
            "arguments": "{}",
            "id": "fc_b",
        },
        {"type": "function_call_output", "call_id": "call_b", "output": "3"},
    ]

    await model.get_response(system_instructions=None, input=model_input)
    await model.get_response(system_instructions=None, input="hello")

    with_outputs, without_outputs = recorder.requests
    assert isinstance(with_outputs.tool_outputs, tuple)
    assert [_call_id_of(item) for item in with_outputs.tool_outputs] == ["call_a", "call_b"]
    assert [_output_of(item) for item in with_outputs.tool_outputs] == ["2", "3"]
    assert without_outputs.tool_outputs == ()


async def test_list_へ正規化できない_dict_単体入力では導出フィールドが空値になる() -> None:
    """input-list でない dict 単体を渡しても `turn` / `user_text` / `tool_outputs` は空値になる。

    `ItemHelpers.input_to_new_input_list` は dict をそのまま返すため、素朴に `list()` すると
    `TypeError` にはならずキー文字列の列（`["role", "content"]`）ができる。これが
    `_input_items` の戻り値 list 検査（`list()` を適用せず空列へ倒す）の根拠で、検査が無い
    まま item 列として扱うと `turn` がキー件数由来で 1 になり、`if request.turn == 0:` を
    前提にしたルール関数の初回分岐が静かに外れる。生の入力は `input` から取れるため情報は
    失われない。
    """
    recorder = _RuleRecorder(_echo_rule)
    model = _deterministic().DeterministicResponseModel(recorder)
    dict_input = {"role": "user", "content": "x"}

    await model.get_response(system_instructions=None, input=dict_input)
    # 正しい input-list を渡した場合との対比（従来どおりであることの pin）。
    await model.get_response(system_instructions=None, input=[{"role": "user", "content": "x"}])

    from_dict, from_list = recorder.requests
    assert from_dict.turn == 0
    assert from_dict.user_text == ""
    assert from_dict.tool_outputs == ()
    # 生の入力は input から取れる（情報は失われない）。
    assert from_dict.input == dict_input
    assert from_list.turn == 0
    assert from_list.user_text == "x"


async def test_Runner_経由の_dict_単体入力でも導出フィールドが空値になる() -> None:
    """`Runner.run(agent, input=<dict>)` という公開入口からも導出フィールドは空値になる。

    Runner は input を `list()` してからモデルへ渡すため、モデルには dict ではなく
    キー文字列の列（`["role", "content"]`）が「正しい list」として届く。item と見なせない
    要素（role も type も取れない）を assistant 由来として数えると `turn` が 1 になり、
    `if request.turn == 0:` を前提にしたルール関数の初回分岐が静かに外れる。
    """
    recorder = _RuleRecorder(_echo_rule)
    model = _deterministic().DeterministicResponseModel(recorder)
    agent = Agent(name="a", instructions="指示文", model=model)

    await Runner.run(agent, input={"role": "user", "content": "x"})
    # 正しい input-list を渡した場合との対比（従来どおりであることの pin）。
    await Runner.run(agent, input=[{"role": "user", "content": "x"}])

    from_dict, from_list = recorder.requests
    assert from_dict.turn == 0
    assert from_dict.user_text == ""
    assert from_dict.tool_outputs == ()
    assert from_list.turn == 0
    assert from_list.user_text == "x"


async def test_get_response_は位置引数とキーワード引数の双方で正規化される() -> None:
    """SDK の呼び出し形（全キーワード / 位置）の双方で同じ `ModelRequest` になる。

    SDK は `get_response` を全キーワードで、`stream_response` を 7 位置引数 + 3 kw-only で
    呼ぶ。将来の呼び出し形変更に備え、両様で同じ正規化結果になることを固定する。
    位置引数側は互いに区別できる番兵値を渡し、位置番号の割り当て（tools は 1・
    output_schema は 2・handoffs は 3）が入れ替わっていないことまで固定する。
    """
    recorder = _RuleRecorder(_echo_rule)
    model = _deterministic().DeterministicResponseModel(recorder)
    settings = ModelSettings()
    tools: list[Any] = ["T"]
    handoffs: list[Any] = ["H"]
    schema = object()

    await model.get_response("sys", "hello", settings, tools, schema, handoffs, None)
    await model.get_response(
        system_instructions="sys",
        input="hello",
        model_settings=settings,
        tools=tools,
        output_schema=schema,
        handoffs=handoffs,
        tracing=None,
    )

    positional, keyword = recorder.requests
    assert positional == keyword
    assert positional.system_instructions == "sys"
    assert positional.user_text == "hello"
    assert positional.model_settings is settings
    assert positional.tools == ("T",)
    assert positional.handoffs == ("H",)
    assert positional.output_schema is schema


async def test_run_streamed_の位置引数呼び出しで_tools_と_handoffs_が取り違えられない() -> None:
    """SDK が `stream_response` を 7 位置引数で呼ぶ実経路で、位置番号の割り当てが保たれる。

    `stream_response` は `*args` をそのまま `get_response` へ渡すため、tools（位置 1）と
    handoffs（位置 3）の割り当てを取り違えても run は成功し続け、ルール関数だけが
    入れ替わった列を受け取る。実 Agent に tool と handoff を両方持たせて固定する。
    """
    recorder = _RuleRecorder(_constant_text("done"))
    model = _deterministic().DeterministicResponseModel(recorder)
    target = Agent(name="b", instructions="b エージェント")
    source = Agent(name="a", instructions="指示文", model=model, tools=[add_one], handoffs=[target])

    streamed = Runner.run_streamed(source, input="hello")
    async for _event in streamed.stream_events():
        pass

    request = recorder.requests[0]
    assert streamed.final_output == "done"
    assert [getattr(tool, "name", None) for tool in request.tools] == ["add_one"]
    assert [getattr(item, "agent_name", None) for item in request.handoffs] == ["b"]
    assert request.model_settings is not None
    assert request.output_schema is None


# ---------------------------------------------------------------------------
# streaming（post-execution streaming）
# ---------------------------------------------------------------------------
async def test_run_streamed_で差分イベントと終端イベントが流れる() -> None:
    """`Runner.run_streamed` で完走し、テキスト差分イベント列 -> 終端イベントが流れる。"""
    model = _deterministic().DeterministicResponseModel(_constant_text("ストリーミング応答"))
    agent = Agent(name="a", instructions="指示文", model=model)

    streamed = Runner.run_streamed(agent, input="hello")
    events = [event async for event in streamed.stream_events()]

    raw = _raw_events(events)
    deltas = [event for event in raw if isinstance(event, ResponseTextDeltaEvent)]
    completed = [event for event in raw if isinstance(event, ResponseCompletedEvent)]
    assert deltas, "テキスト応答では差分イベントが 1 件以上流れる"
    assert "".join(event.delta for event in deltas) == "ストリーミング応答"
    assert len(completed) == 1
    assert streamed.final_output == "ストリーミング応答"


async def test_ToolCall_のみの応答では差分が流れず終端イベントのみになる() -> None:
    """ツール呼び出しだけの応答はテキストを持たないため差分イベントを流さない。"""
    module = _deterministic()
    model = module.DeterministicResponseModel(
        lambda request: module.tool_call_response("add_one", '{"x": 1}')
    )

    events = [event async for event in model.stream_response(None, "hello")]

    assert len(events) == 1
    assert isinstance(events[0], ResponseCompletedEvent)
    assert events[0].response.output[0].name == "add_one"


async def test_ストリーミングイベントの識別子が禁止語を含まない() -> None:
    """イベント識別子は `*_deterministic` 系で、`workflow` / `wf` を含まない（FR-6）。"""
    model = _deterministic().DeterministicResponseModel(_constant_text("hello streaming text"))

    events = [event async for event in model.stream_response(None, "hello")]

    deltas, completed = events[:-1], events[-1]
    assert deltas, "テキスト応答では差分イベントが 1 件以上流れる"
    assert {event.item_id for event in deltas} == {"msg_deterministic"}
    assert completed.type == "response.completed"
    assert completed.response.id == "resp_deterministic"
    assert completed.response.model == "oai-agentspec-deterministic"

    identifiers = [event.item_id for event in deltas] + [
        completed.response.id,
        completed.response.model,
    ]
    for identifier in identifiers:
        lowered = identifier.lower()
        assert "workflow" not in lowered
        assert "wf" not in lowered


async def test_ルール関数が応答オブジェクト以外を返すと_TypeError_になる() -> None:
    """契約境界で fail-fast させる（原因から遠い SDK 内部エラーにしない）。

    素の `str` を返す誤用は他ライブラリの fake model では一般的なため、SDK 内部で
    `'str' object has no attribute 'output'` になる前に明示メッセージで弾く。
    """
    model = _deterministic().DeterministicResponseModel(lambda request: "plain string")

    with pytest.raises(TypeError) as excinfo:
        await model.get_response(system_instructions=None, input="hi")

    assert "str" in str(excinfo.value)


async def test_async_ルール関数が応答オブジェクト以外を返しても_TypeError_になる() -> None:
    """awaitable の解決後に型検証する（async ルール関数も対象）。"""

    async def rule(request: Any) -> Any:
        return 42

    model = _deterministic().DeterministicResponseModel(rule)

    with pytest.raises(TypeError) as excinfo:
        await model.get_response(system_instructions=None, input="hi")

    assert "int" in str(excinfo.value)


async def test_hosted_tool_の_item_が挟まっても_1_応答は_1_ターンと数える() -> None:
    """ターン境界は user / tool 側の発話であり、assistant 側 item 種別を列挙しない。

    assistant 由来の item 種別を allowlist で列挙すると、hosted tool 系 item
    （`web_search_call` 等）が 1 応答の途中に挟まったときに連続グループが分断され、
    1 応答が 2 ターンとして数えられる。公開窓口が掲げる用途「決定的なシナリオ再生」は
    実 API の履歴を input として渡す使い方なので、看板の用途で壊れる。
    """
    recorder = _RuleRecorder(_echo_rule)
    model = _deterministic().DeterministicResponseModel(recorder)

    await model.get_response(
        system_instructions=None,
        input=[
            {"role": "user", "content": "調べて"},
            # ここから下は 1 回のモデル応答が展開されたもの。hosted tool の item が
            # assistant 由来 item 群の**途中**に挟まる形でなければ allowlist 実装との
            # 差が出ない（境界の数が変わらないため）。
            {"role": "assistant", "content": "調べます"},
            {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            {"role": "assistant", "content": "調べました"},
        ],
    )

    # allowlist（assistant 由来を function_call / reasoning / role==assistant で列挙）だと
    # web_search_call が境界と誤判定されてグループが 2 つに割れ、turn == 2 になる。
    assert recorder.requests[0].turn == 1
