"""決定的応答モデルと公開応答ビルダ（SDK 結合を閉じる・NFR-1）。

入力からルール関数が応答を決める純関数方式のステートレス SDK `Model` 実装
（`DeterministicResponseModel`）・ルール関数へ渡す入力ペイロード（`ModelRequest`）・
ルール関数が返す `ModelResponse` を組み立てる公開ビルダ 5 種（`text_response` /
`text_response_with_usage` / `tool_call_response` / `multi_tool_call_response` /
`mixed_response`）を提供する。公開窓口は `oai_agentspec.runtime.deterministic`。

SDK Responses item とストリームイベントの構築は `responses` モジュールの共有ヘルパへ委譲する
（上向き 1 方向の依存）。既定 id は本モジュールの公開用と `responses` の内部ワークフロー用の
2 系統を持ち、いずれもヘルパ呼び出し時に明示する。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from agents import ItemHelpers, Model, Usage
from agents.items import ModelResponse

from .responses import (
    _completed_event,
    _make_function_call,
    _make_text_message,
    _text_delta_events,
    _text_of,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

# 公開経路の既定 id（値そのものが公開契約。SDK の id 接頭辞慣行に従う）。
DEFAULT_MESSAGE_ID: Final[str] = "msg_deterministic"
DEFAULT_TOOL_CALL_ITEM_ID: Final[str] = "fc_deterministic"
DEFAULT_CALL_ID: Final[str] = "call_deterministic"
DEFAULT_RESPONSE_ID: Final[str] = "resp_deterministic"
DEFAULT_STREAM_MODEL: Final[str] = "oai-agentspec-deterministic"

# ターン境界となる input item の role（user / tool 側の発話。`_count_turns` の判定条件）。
_BOUNDARY_ROLES: Final[frozenset[str]] = frozenset({"user", "system", "developer"})

# ターン境界となる input item の type 接尾辞（`function_call_output` 等の tool 実行結果）。
_BOUNDARY_ITEM_TYPE_SUFFIX: Final[str] = "_output"

# tool 実行結果アイテムの type（`ModelRequest.tool_outputs` の抽出条件）。
_TOOL_OUTPUT_ITEM_TYPE: Final[str] = "function_call_output"

# user メッセージの content パートのうちテキストとして扱う type。
_USER_TEXT_PART_TYPES: Final[frozenset[str]] = frozenset({"input_text", "text"})


def text_response(text: str) -> ModelResponse:
    """単一のアシスタントテキストメッセージを返す ModelResponse を作る。

    Args:
        text: メッセージ本文。

    Returns:
        単一テキストメッセージ・usage 全 0 の ModelResponse。
    """
    message = _make_text_message(text, message_id=DEFAULT_MESSAGE_ID)
    return ModelResponse(output=[message], usage=Usage(), response_id=None)


def text_response_with_usage(text: str, *, total_tokens: int, requests: int = 1) -> ModelResponse:
    """usage（トークン数・requests）付きのテキスト ModelResponse を作る。

    SDK run loop は応答ごとに `context.usage` へ加算するため、累積使用量で分岐する挙動
    （run 予算超過など）を実 API 抜きで動かすには usage を載せた応答が要る。

    Args:
        text: メッセージ本文。
        total_tokens: この応答の累積対象トークン数。
        requests: この応答の requests 数（usage 欠損検知の回避に 1 以上を既定とする）。

    Returns:
        usage を持つ単一テキストメッセージの ModelResponse。
    """
    message = _make_text_message(text, message_id=DEFAULT_MESSAGE_ID)
    usage = Usage(requests=requests, total_tokens=total_tokens)
    return ModelResponse(output=[message], usage=usage, response_id=None)


def tool_call_response(
    tool_name: str,
    arguments: str = "{}",
    *,
    call_id: str = DEFAULT_CALL_ID,
) -> ModelResponse:
    """単一の function ToolCall を返す ModelResponse を作る（tool 実行 / handoff の誘発）。

    item id は既定値で固定し、`call_id` のみ指定できる（tool_call と tool_result の対応を
    利用者が制御するための最小の口）。

    Args:
        tool_name: 呼び出す tool 名（handoff なら `transfer_to_<エージェント名>`）。
        arguments: tool 引数の JSON 文字列。`json.dumps()` で生成すること。
            `ModelRequest` 由来の値を文字列連結・f-string で埋め込むと、入力に含まれる
            引用符で JSON を脱出して実 tool へ想定外のキーを渡せる。
        call_id: tool_call と tool_result を対応づける id。

    Returns:
        単一の function ToolCall を持つ ModelResponse。
    """
    call = _make_function_call(
        tool_name, arguments, call_id=call_id, item_id=DEFAULT_TOOL_CALL_ITEM_ID
    )
    return ModelResponse(output=[call], usage=Usage(), response_id=None)


def multi_tool_call_response(calls: Sequence[tuple[str, str, str]]) -> ModelResponse:
    """複数の function ToolCall を 1 応答で返す ModelResponse を作る。

    1 ターンで複数の tool を呼ぶ（複数の承認待ちを同時に生む等）シナリオの再現に使う。
    item id は `call_id` から個別化する（`fc_<call_id>`）。

    Args:
        calls: `(tool 名, 引数の JSON 文字列, call_id)` のタプル列。`call_id` は一意にすること。
            引数の JSON は `json.dumps()` で生成すること（`ModelRequest` 由来の値を文字列連結・
            f-string で埋め込むと引用符で JSON を脱出できる）。

    Returns:
        宣言順に ToolCall を並べた ModelResponse。
    """
    output = [
        _make_function_call(tool_name, arguments, call_id=call_id, item_id=f"fc_{call_id}")
        for tool_name, arguments, call_id in calls
    ]
    return ModelResponse(output=output, usage=Usage(), response_id=None)


def mixed_response(
    text: str,
    calls: Sequence[tuple[str, str, str]],
    *,
    total_tokens: int = 0,
    requests: int = 0,
) -> ModelResponse:
    """テキストメッセージ 1 件と function ToolCall N 件を 1 応答へ載せる ModelResponse を作る。

    「一言返してから tool を呼ぶ」「発話しながらハンドオフする」を 1 応答で表現する。
    `calls` が空なら出力はテキストメッセージ 1 件のみになる。

    Args:
        text: 先頭に置くアシスタントテキスト。
        calls: `(tool 名, 引数の JSON 文字列, call_id)` のタプル列（宣言順に並ぶ）。引数の
            JSON は `json.dumps()` で生成すること（`ModelRequest` 由来の値を文字列連結・
            f-string で埋め込むと引用符で JSON を脱出できる）。
        total_tokens: この応答の累積対象トークン数。
        requests: この応答の requests 数。既定の 0 は「usage を指定していない」を意味する
            （`text_response` と整合）。usage を指定する場合は `requests` も 1 以上を
            指定すること（`requests=0` かつ `total_tokens>0` は usage 欠損検知の対象になる）。

    Returns:
        テキストメッセージの後ろに ToolCall を並べた ModelResponse。
    """
    output: list[Any] = [_make_text_message(text, message_id=DEFAULT_MESSAGE_ID)]
    output.extend(
        _make_function_call(tool_name, arguments, call_id=call_id, item_id=f"fc_{call_id}")
        for tool_name, arguments, call_id in calls
    )
    usage = Usage(requests=requests, total_tokens=total_tokens)
    return ModelResponse(output=output, usage=usage, response_id=None)


@dataclass(frozen=True)
class ModelRequest:
    """ルール関数へ渡す 1 回分のモデル呼び出し入力（すべて `get_response` 引数から導出）。

    SDK 型（`ModelSettings` / Tool / Handoff / 出力スキーマ）は不透明値として運ぶだけで、
    lib はフィールドの属性へ触らない（SDK 隔離・NFR-1）。

    Attributes:
        system_instructions: Agent の instructions。
        input: SDK が渡した入力そのもの（文字列 / input-list）。
        user_text: 入力から抽出した直近の user テキスト。抽出できない場合（画像のみの
            user メッセージ・user ロールのアイテムが無い履歴等）は空文字列になる
            （生の入力は `input` から取得する）。
        turn: 入力に含まれるモデル応答の件数（初回 0）。`input` から純粋に導出する。
        tool_outputs: 入力中の tool 実行結果アイテムの列。`input` から純粋に導出する。
        model_settings: SDK の `ModelSettings`。
        tools: 提示されている Tool の列。
        handoffs: 提示されている Handoff の列。
        output_schema: 構造化出力スキーマ（未指定なら `None`）。
    """

    system_instructions: str | None
    input: Any
    user_text: str
    turn: int
    tool_outputs: tuple[Any, ...]
    model_settings: Any
    tools: tuple[Any, ...]
    handoffs: tuple[Any, ...]
    output_schema: Any


def _positional(args: tuple[Any, ...], index: int) -> Any:
    """位置引数を安全に取り出す（不足時は None）。

    Args:
        args: `get_response` が `*args` で吸収した位置引数。
        index: 取り出す位置。

    Returns:
        当該位置の値、または不足していれば None。
    """
    return args[index] if len(args) > index else None


def _as_tuple(value: Any) -> tuple[Any, ...]:
    """SDK が渡す列（tools / handoffs）を plain な tuple へ正規化する。

    Args:
        value: 列または None。

    Returns:
        tuple 化した列（None なら空 tuple）。
    """
    if value is None:
        return ()
    return tuple(value)


def _input_items(model_input: Any) -> list[Any]:
    """input を SDK の input-list へ正規化する（`ModelRequest` 導出フィールドの唯一の起点）。

    `user_text` / `turn` / `tool_outputs` はすべて本関数の戻り値から導出する。正規化を
    1 箇所に集約することで、同じ正規化と同じ例外処理が複数箇所へ散らないようにする。

    `ItemHelpers.input_to_new_input_list` は文字列を user メッセージ 1 件へ展開し、
    それ以外は要素を dict 化して返すだけで、`None` や非 iterable はそのまま返す。
    その値を `list()` する時点で `TypeError` になるため、捕捉対象はこれに限定する
    （SDK 由来の想定外例外を無言で握り潰さない）。

    Args:
        model_input: `Model.get_response` が受ける input（文字列 / input-list）。

    Returns:
        input item のリスト。input が `None` や非 iterable で正規化できない場合は空列。
        このとき `user_text` は空文字列・`turn` は 0・`tool_outputs` は空になる。
    """
    try:
        return list(ItemHelpers.input_to_new_input_list(model_input))
    except TypeError:
        return []


def _latest_user_text(items: list[Any]) -> str | None:
    """正規化済み input item 列から直近の user テキストを取り出す（見つからなければ None）。

    `responses.latest_user_text` は WorkflowModel 向けに「抽出できなければ元の input を
    そのまま返す」安全フォールバックを持つが、`ModelRequest.user_text: str` は別契約
    （user 由来のテキストのみを載せる）なので、抽出できないことを `None` で表す。
    入力全体を文字列化して埋めることはしない（tool 実行結果や system 文言が user 由来の
    値として遷移判定へ流れ込むのを防ぐため）。

    Args:
        items: `_input_items` が返した正規化済みの input item 列。

    Returns:
        直近の user メッセージのテキスト。user テキストが無ければ None。
    """
    for item in reversed(items):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in _USER_TEXT_PART_TYPES
            ]
            if texts:
                return "".join(texts)
    return None


def _item_fields(item: Any) -> tuple[Any, Any]:
    """input item から role / type を取り出す（dict / オブジェクトの双方に対応）。

    Args:
        item: 正規化済みの input item。

    Returns:
        `(role, type)`。取得できない要素は None になる。
    """
    if isinstance(item, dict):
        return item.get("role"), item.get("type")
    return getattr(item, "role", None), getattr(item, "type", None)


def _is_turn_boundary(item: Any) -> bool:
    """input item がターン境界（user / tool 側の発話）かどうかを返す。

    ターン境界は本来「user / tool 側が話したところ」であり、assistant 側の item 種別を
    列挙する allowlist にはしない。SDK は assistant 側の item 種別を増やす方向のため
    （hosted tool 系の `web_search_call` / `file_search_call` / `computer_call` 等）、
    allowlist だと追随漏れした種別が 1 応答の途中でグループを分断し、1 応答が 2 ターンとして
    数えられる。境界側（有限で安定した user / system / developer と `*_output`）を列挙して
    残りを assistant 由来として扱えば、SDK の item 種別追加へ追随漏れしない。

    Args:
        item: 正規化済みの input item。

    Returns:
        ターン境界なら True（assistant 由来なら False）。
    """
    role, item_type = _item_fields(item)
    if role in _BOUNDARY_ROLES:
        return True
    return isinstance(item_type, str) and item_type.endswith(_BOUNDARY_ITEM_TYPE_SUFFIX)


def _count_turns(items: list[Any]) -> int:
    """入力中のモデル応答件数を数える（1 モデル応答 = 1 ターン）。

    1 回のモデル応答は input-list 上で複数 item（テキストメッセージ + ToolCall 等）へ展開され、
    tool 結果ターンには `role == "assistant"` の item が 1 つも無いことがある。item 単位で
    数えると 1 応答で 2 ターン進み、ルール関数の `turn` 分岐が静かに外れるため、assistant 由来
    item（= ターン境界でない item）の**連続グループ数**を 1 ターンとして数える。

    Args:
        items: 正規化済みの input item 列。

    Returns:
        入力に含まれるモデル応答の件数（初回 0）。
    """
    turns = 0
    previous_is_assistant = False
    for item in items:
        current_is_assistant = not _is_turn_boundary(item)
        if current_is_assistant and not previous_is_assistant:
            turns += 1
        previous_is_assistant = current_is_assistant
    return turns


def _collect_tool_outputs(items: list[Any]) -> tuple[Any, ...]:
    """入力中の tool 実行結果アイテムを宣言順に集める。

    Args:
        items: 正規化済みの input item 列。

    Returns:
        `function_call_output` アイテムの tuple。
    """
    return tuple(item for item in items if _item_fields(item)[1] == _TOOL_OUTPUT_ITEM_TYPE)


class DeterministicResponseModel(Model):
    """入力からルール関数が応答を決めるステートレス Model（実 API を呼ばない）。

    保持するのはルール関数（不変設定）のみで、可変な実行状態を一切持たない。そのため同一
    インスタンスを複数 run・複数 Agent（handoff 元と handoff 先など）で共有しても、応答は
    呼び出し回数・順序に依存しない。`get_response` は SDK 引数を `ModelRequest` へ正規化して
    ルール関数へ渡し、戻り値が awaitable なら await して解決する（同期 / async の双方を受理）。
    `None` は空テキスト応答として扱い、`ModelResponse` でも `None` でもない戻り値は
    `TypeError` で弾く。ルール関数自身の例外は握り潰さず伝播させる。
    `stream_response` は応答を確定させてから流す post-execution streaming
    （`WorkflowModel` と同型）。SDK `Model` ABC（get_response / stream_response）へ結合する。

    応答は利用者のルール関数が決め、モデル提供側の安全機構を一切経由しない。信頼できない
    入力を扱う本番経路の Model として運用構成へ残さないこと（想定用途は自動テスト・実 API を
    呼ばないオフライン開発・デモ実行・決定的なシナリオ再生）。
    """

    def __init__(self, rule: Callable[[ModelRequest], Any]) -> None:
        """決定的応答モデルを生成する。

        Args:
            rule: `ModelRequest` から応答を決める関数（不変設定）。戻り値は `ModelResponse` /
                `None` / それらの awaitable。
        """
        self._rule = rule

    async def get_response(
        self,
        system_instructions: Any = None,
        input: Any = None,  # noqa: A002 - SDK Model.get_response の引数名に追従
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        """ルール関数が決めた応答を返す（実 LLM 呼び出しは 0 回）。

        SDK は `get_response` を全キーワードで呼ぶが、位置引数でも受かるようキーワード優先 +
        位置フォールバックで正規化する（位置は `model_settings` / `tools` / `output_schema` /
        `handoffs` / `tracing` の順）。`tracing` 以降は `ModelRequest` へ渡さない。

        Returns:
            ルール関数が返した ModelResponse（`None` を返した場合は空テキスト応答）。

        Raises:
            TypeError: ルール関数（awaitable なら解決後）の戻り値が `ModelResponse` でも
                `None` でもない場合。SDK 内部まで持ち越すと原因から遠い属性エラーになるため、
                契約境界で fail-fast させる。
        """
        request = self._build_request(system_instructions, input, args, kwargs)
        response = self._rule(request)
        if inspect.isawaitable(response):
            response = await response
        if response is None:
            return text_response("")
        if not isinstance(response, ModelResponse):
            raise TypeError(
                "ルール関数は応答ビルダ（text_response 等）の戻り値か None を返すこと: "
                f"{type(response).__name__} が返されました"
            )
        return response

    async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """応答を確定させてから text-delta + completed イベントで流す（run_streamed 対応）。

        `get_response` でルール関数の応答を確定させ、テキストが非空なら
        `ResponseTextDeltaEvent` に区切って流し、最後に `ResponseCompletedEvent`（Runner が
        最終出力として取り出す終端）を yield する。ツール呼び出しのみの応答はテキストを持た
        ないため差分が流れず終端イベントのみになる。応答は先に確定しているため、これは
        進捗を表さない post-execution streaming である。

        Yields:
            ResponseTextDeltaEvent / ResponseCompletedEvent。
        """
        response = await self.get_response(*args, **kwargs)
        text = _text_of(response)
        seq = 0
        for event in _text_delta_events(text, item_id=DEFAULT_MESSAGE_ID):
            yield event
            seq = event.sequence_number + 1
        yield _completed_event(
            response.output,
            seq,
            response_id=DEFAULT_RESPONSE_ID,
            model=DEFAULT_STREAM_MODEL,
        )

    @staticmethod
    def _build_request(
        system_instructions: Any,
        model_input: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ModelRequest:
        """SDK 引数を `ModelRequest` へ正規化する（キーワード優先 + 位置フォールバック）。

        input の正規化は `_input_items` の 1 回のみで、`user_text` / `turn` / `tool_outputs`
        はすべてその item 列から導出する。

        Args:
            system_instructions: Agent の instructions。
            model_input: SDK が渡した input。
            args: `*args` で吸収した位置引数。
            kwargs: `**kwargs` で吸収したキーワード引数。

        Returns:
            ルール関数へ渡す ModelRequest。
        """
        items = _input_items(model_input)
        user_text = _latest_user_text(items)
        return ModelRequest(
            system_instructions=system_instructions,
            input=model_input,
            user_text=user_text if user_text is not None else "",
            turn=_count_turns(items),
            tool_outputs=_collect_tool_outputs(items),
            model_settings=kwargs.get("model_settings", _positional(args, 0)),
            tools=_as_tuple(kwargs.get("tools", _positional(args, 1))),
            handoffs=_as_tuple(kwargs.get("handoffs", _positional(args, 3))),
            output_schema=kwargs.get("output_schema", _positional(args, 2)),
        )
