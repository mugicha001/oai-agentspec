"""L2: `dpo_dataset_from_session`（FR-11）の 2 モード・切り出し規則・エラー方針を固定する。

fake Session（`_helpers.fake_session.FakeSession`・呼び出しメソッド記録付き）を通して
「`dpo_dataset_from_session` -> `_adapters.finetune.fetch_session_items` -> Session」の鎖を測る。

固定する契約（要件書 FR-11 / ADR 0034 Decision 5）:
    - 共通: ケース素材の切り出しは FR-4 と同一（累積ペアリング・ツール往復の変換保持・
      parts の str 吸収・空文脈 / 空応答ケースは生成せず skipped へ計上）
    - callable モード（`pair_builder` 指定）: 入力は `{"input", "response"}` の plain dict。
      両キー（preferred_output / non_preferred_output）を含む dict で採用、任意キー `input`
      で文脈差し替え、`None` で skip、全 skip は空結果の正常返却、それ以外の戻り値は
      ケース位置つき VALIDATION_FAILED
    - 雛形モード（`pair_builder` 省略）: `records` は最終レコードではなく
      `{"input", "preferred_output", "non_preferred_output", "response"}` の記入用ケース列
      （記入 2 欄は空文字・実応答は `response` 参照欄にのみ置く）
    - 読み取り専用: Session へは `get_items` のみ（書込系メソッドを一切呼ばない）
    - tools 透過: `tools=` / `parallel_tool_calls=` は写像も検証もせず `to_dpo_dataset` へ
      素通しし、`input` 内へ載る（採用 0 件でも委譲する）。雛形モードでの指定は反映先が
      無いため履歴を読む前に CONFIG_MISSING で拒否する
    - エラー: session=None は CONFIG_MISSING、空履歴 / 抽出可能ターンなし / テキスト応答の
      assistant なしは VALIDATION_FAILED

Session Protocol への duck typing 接触を含むため層は L2（`@pytest.mark.integration`）。
ネットワーク非接触は conftest の autouse ガードが担保する。
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from oai_agentspec.runtime.finetune import (
    FineTuneError,
    FineTuneFailureKind,
    dpo_dataset_from_session,
    validate_dataset,
)

from _helpers.fake_session import FakeSession

pytestmark = pytest.mark.integration

# ツール往復を 1 組含む履歴 items（Responses API 形式・plain dict）。
_SAMPLE_ITEMS: list[dict[str, Any]] = [
    {"role": "user", "content": "会員登録の手順を教えて"},
    {
        "type": "function_call",
        "name": "lookup_faq",
        "arguments": '{"q":"register"}',
        "call_id": "call_1",
    },
    {"type": "function_call_output", "call_id": "call_1", "output": "{...}"},
    {"role": "assistant", "content": [{"type": "output_text", "text": "手順は次の通りです: ..."}]},
    {"role": "user", "content": "料金は?"},
    {"role": "assistant", "content": [{"type": "output_text", "text": "月額 500 円です"}]},
]

_TOOL_CALL_MESSAGE = {
    "role": "assistant",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup_faq", "arguments": '{"q":"register"}'},
        }
    ],
}
_TOOL_OUTPUT_MESSAGE = {"role": "tool", "tool_call_id": "call_1", "content": "{...}"}

# `_SAMPLE_ITEMS` から切り出される 2 件のケース素材の累積文脈（変換済みツール往復を含む）。
_CONTEXT_1 = [
    {"role": "user", "content": "会員登録の手順を教えて"},
    _TOOL_CALL_MESSAGE,
    _TOOL_OUTPUT_MESSAGE,
]
_CONTEXT_2 = [
    *_CONTEXT_1,
    {"role": "assistant", "content": "手順は次の通りです: ..."},
    {"role": "user", "content": "料金は?"},
]
_RESPONSE_1 = "手順は次の通りです: ..."
_RESPONSE_2 = "月額 500 円です"


# ----------------------------------------------------------------------
# callable モード（pair_builder 指定）
# ----------------------------------------------------------------------


async def test_pair_builder_receives_case_material_with_input_and_response() -> None:
    """pair_builder には `{"input", "response"}` の 2 キーだけを持つ plain dict が渡る。

    キー名 `response`（ログ上の実応答）と、累積文脈へ変換済みツール往復メッセージが
    含まれること（文脈保持）を同時に固定する。SFT 版の `expected_output` へ戻す変異・
    余分なキーを混ぜる変異が RED になる。
    """
    session = FakeSession(_SAMPLE_ITEMS)
    seen: list[dict[str, Any]] = []

    def builder(material: dict[str, Any]) -> dict[str, Any]:
        seen.append(material)
        return {"preferred_output": "よい応答", "non_preferred_output": "わるい応答"}

    await dpo_dataset_from_session(session, pair_builder=builder)

    assert [sorted(material) for material in seen] == [["input", "response"], ["input", "response"]]
    assert seen == [
        {"input": _CONTEXT_1, "response": _RESPONSE_1},
        {"input": _CONTEXT_2, "response": _RESPONSE_2},
    ]


async def test_pair_builder_result_is_converted_into_preference_records() -> None:
    """両キーを含む dict を返したケースは preference レコードへ変換される。

    最終形（`{"input": {"messages": [...]}, "preferred_output": [...],
    "non_preferred_output": [...]}`）は `to_dpo_dataset` 委譲の帰結であり、文字列は
    assistant 1 件へ包まれる。`records` は全体を `==` で照合する。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    def builder(material: dict[str, Any]) -> dict[str, Any]:
        return {
            "preferred_output": material["response"],
            "non_preferred_output": "そっけない返答",
        }

    result = await dpo_dataset_from_session(session, pair_builder=builder)

    assert result.records == (
        {
            "input": {"messages": _CONTEXT_1},
            "preferred_output": [{"role": "assistant", "content": _RESPONSE_1}],
            "non_preferred_output": [{"role": "assistant", "content": "そっけない返答"}],
        },
        {
            "input": {"messages": _CONTEXT_2},
            "preferred_output": [{"role": "assistant", "content": _RESPONSE_2}],
            "non_preferred_output": [{"role": "assistant", "content": "そっけない返答"}],
        },
    )
    assert result.skipped == 0


async def test_pair_builder_input_key_replaces_generated_context() -> None:
    """戻り値の任意キー `input` は lib が組んだ累積文脈を置き換える（マスキング経路）。"""
    session = FakeSession(_SAMPLE_ITEMS)

    def builder(material: dict[str, Any]) -> dict[str, Any]:
        masked = [
            {**message, "content": str(message["content"]).replace("会員登録", "[MASKED]")}
            if "content" in message
            else message
            for message in material["input"]
        ]
        return {
            "input": masked,
            "preferred_output": "よい応答",
            "non_preferred_output": "わるい応答",
        }

    result = await dpo_dataset_from_session(session, pair_builder=builder)

    assert result.records[0]["input"]["messages"][0] == {
        "role": "user",
        "content": "[MASKED]の手順を教えて",
    }
    assert result.records[1]["input"]["messages"][0] == {
        "role": "user",
        "content": "[MASKED]の手順を教えて",
    }


async def test_pair_builder_empty_input_key_replaces_context_and_fails_closed() -> None:
    """`input` は「キーの有無」で差し替える（空リストも指定として尊重し fail-closed）。

    `test_pair_builder_input_key_replaces_generated_context` と対の pin で、truthy 判定
    （`built.get("input") or context`）へ倒す変異が RED になる。退行すると、利用者が
    マスキング目的で明示的に置いた空文脈が silent に破棄され、マスキング前の生ログが
    そのまま学習データへ載る。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    def builder(_material: dict[str, Any]) -> dict[str, Any]:
        return {
            "input": [],
            "preferred_output": "よい応答",
            "non_preferred_output": "わるい応答",
        }

    with pytest.raises(FineTuneError) as exc_info:
        await dpo_dataset_from_session(session, pair_builder=builder)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース 1" in exc_info.value.message
    assert "input が空リストである" in exc_info.value.message


async def test_pair_builder_returning_none_skips_case() -> None:
    """pair_builder が None を返したケースは生成せず skipped に計上する。"""
    session = FakeSession(_SAMPLE_ITEMS)

    def builder(material: dict[str, Any]) -> dict[str, Any] | None:
        if material["response"] == _RESPONSE_1:
            return None
        return {"preferred_output": "よい応答", "non_preferred_output": "わるい応答"}

    result = await dpo_dataset_from_session(session, pair_builder=builder)

    assert len(result.records) == 1
    assert result.records[0]["input"] == {"messages": _CONTEXT_2}
    assert result.skipped == 1


async def test_pair_builder_skipping_all_cases_returns_empty_result_not_error() -> None:
    """全ケース skip はエラーにせず `records == ()` / `skipped == 全件` を正常返却する。

    skip は利用者の明示判断であり失敗ではない（ADR 0033 Decision 6 と同型）。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    result = await dpo_dataset_from_session(session, pair_builder=lambda _material: None)

    assert result.records == ()
    assert result.skipped == 2


async def test_pair_builder_returning_non_dict_raises_validation_failed() -> None:
    """None でも dict でもない戻り値はケース位置つき VALIDATION_FAILED で失敗する。

    位置は元ケースの並び（1 始まり）であり、2 件目の違反は「ケース 2」と報告される
    （暗黙 skip にしない fail-closed）。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    def builder(material: dict[str, Any]) -> Any:
        if material["response"] == _RESPONSE_1:
            return {"preferred_output": "よい応答", "non_preferred_output": "わるい応答"}
        return "not-a-dict"

    with pytest.raises(FineTuneError) as exc_info:
        await dpo_dataset_from_session(session, pair_builder=builder)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース 2" in exc_info.value.message
    assert "pair_builder" in exc_info.value.message


async def test_pair_builder_missing_required_key_raises_validation_failed() -> None:
    """必須キーを欠く dict はケース位置と欠落キー名つきの VALIDATION_FAILED で失敗する。"""
    session = FakeSession(_SAMPLE_ITEMS)

    def builder(material: dict[str, Any]) -> dict[str, Any]:
        if material["response"] == _RESPONSE_1:
            return {"preferred_output": "よい応答", "non_preferred_output": "わるい応答"}
        return {"preferred_output": "よい応答"}

    with pytest.raises(FineTuneError) as exc_info:
        await dpo_dataset_from_session(session, pair_builder=builder)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース 2" in exc_info.value.message
    assert "non_preferred_output" in exc_info.value.message


async def test_delegated_validation_error_is_not_silently_skipped() -> None:
    """委譲先が受理しないペア（型不正）は skip されず VALIDATION_FAILED で失敗する。

    最終変換は `to_dpo_dataset(skip_missing=False)` へ委譲する決定の pin（`skip_missing=True`
    へ変える変異が RED になる）。不備ケースを silent に落として件数だけ減らさない。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    def builder(material: dict[str, Any]) -> dict[str, Any]:
        if material["response"] == _RESPONSE_1:
            return {"preferred_output": "よい応答", "non_preferred_output": "わるい応答"}
        return {"preferred_output": 42, "non_preferred_output": "わるい応答"}

    with pytest.raises(FineTuneError) as exc_info:
        await dpo_dataset_from_session(session, pair_builder=builder)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース 2" in exc_info.value.message


# ----------------------------------------------------------------------
# 雛形モード（pair_builder 省略）
# ----------------------------------------------------------------------


async def test_draft_mode_returns_fillable_cases_with_four_keys() -> None:
    """雛形モードは 4 キーの記入用ケース列を返す（最終レコードではない）。

    記入 2 欄は空文字で、実応答は `response` 参照欄にのみ保持する（どちらの欄へ置くかは
    品質判定であり lib は内蔵しない・ADR 0034 Decision 5）。形は `to_dpo_dataset` の
    入力ケース形 + `response` 欄そのもの。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    result = await dpo_dataset_from_session(session)

    assert result.records == (
        {
            "input": _CONTEXT_1,
            "preferred_output": "",
            "non_preferred_output": "",
            "response": _RESPONSE_1,
        },
        {
            "input": _CONTEXT_2,
            "preferred_output": "",
            "non_preferred_output": "",
            "response": _RESPONSE_2,
        },
    )
    assert result.skipped == 0


async def test_draft_mode_context_contains_converted_tool_messages() -> None:
    """雛形モードのケース文脈にも変換済みツール往復メッセージが現れる（文脈保持）。"""
    session = FakeSession(_SAMPLE_ITEMS)

    result = await dpo_dataset_from_session(session)

    assert result.records[0]["input"][1] == _TOOL_CALL_MESSAGE
    assert result.records[0]["input"][2] == _TOOL_OUTPUT_MESSAGE


# ----------------------------------------------------------------------
# 共通（切り出し規則・読み取り専用・エラー方針）
# ----------------------------------------------------------------------


async def test_session_access_is_read_only() -> None:
    """Session へは `get_items` のみ呼ばれ、書込系メソッドは 1 回も呼ばれない（FR-11）。"""
    session = FakeSession(_SAMPLE_ITEMS)

    await dpo_dataset_from_session(session)

    assert session.calls == ["get_items"]


async def test_leading_assistant_and_empty_response_cases_are_skipped() -> None:
    """先頭 assistant（空文脈）と空へ吸収される応答のケースは skipped へ計上する（FR-4 と同一）。"""
    session = FakeSession(
        [
            {"role": "assistant", "content": "こんにちは"},
            {"role": "user", "content": "これはできますか?"},
            {"role": "assistant", "content": [{"type": "refusal", "refusal": "お断りします"}]},
            {"role": "user", "content": "では料金は?"},
            {"role": "assistant", "content": "月額 500 円です"},
        ]
    )

    result = await dpo_dataset_from_session(session)

    assert len(result.records) == 1
    assert result.records[0]["response"] == "月額 500 円です"
    assert result.skipped == 2


async def test_none_session_raises_config_missing_before_get_items() -> None:
    """session=None は CONFIG_MISSING で失敗する（get_items を呼ぶ前・fake 不要）。"""
    with pytest.raises(FineTuneError) as exc_info:
        await dpo_dataset_from_session(None)

    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING


async def test_empty_history_raises_validation_failed() -> None:
    """空履歴は VALIDATION_FAILED で失敗する（空データセットを暗黙に返さない）。"""
    session = FakeSession([])

    with pytest.raises(FineTuneError) as exc_info:
        await dpo_dataset_from_session(session)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


async def test_history_without_extractable_turns_raises_validation_failed() -> None:
    """孤児 function_call と system item のみの履歴は VALIDATION_FAILED で失敗する。"""
    session = FakeSession(
        [
            {"type": "function_call", "name": "f", "arguments": "{}", "call_id": "call_1"},
            {"role": "system", "content": "指示"},
        ]
    )

    with pytest.raises(FineTuneError) as exc_info:
        await dpo_dataset_from_session(session)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


async def test_history_without_text_assistant_raises_validation_failed() -> None:
    """テキスト応答の assistant が無い履歴（ツール往復のみ）は VALIDATION_FAILED で失敗する。"""
    session = FakeSession(
        [
            {"role": "user", "content": "調べて"},
            {"type": "function_call", "name": "f", "arguments": "{}", "call_id": "call_1"},
            {"type": "function_call_output", "call_id": "call_1", "output": "A"},
        ]
    )

    with pytest.raises(FineTuneError) as exc_info:
        await dpo_dataset_from_session(session)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


# ----------------------------------------------------------------------
# tools= / parallel_tool_calls= の透過（委譲先 `to_dpo_dataset` へ素通し）
# ----------------------------------------------------------------------


class _FakeFunctionTool:
    """SDK `FunctionTool` 相当の属性のみを持つ fake（`agents` を import しない）。

    `test_dataset_l1.py` の同名 fake と同型（写像規則の担保は委譲先テストの責務であり、
    ここでは「渡した値が委譲先へ届くか」だけを測る）。
    """

    def __init__(self, name: str, params_json_schema: dict[str, Any], description: str) -> None:
        self.name = name
        self.params_json_schema = params_json_schema
        self.description = description


# 透過確認に使う plain dict の tools（写像では非改変で載る形）。
_PLAIN_TOOLS: list[Any] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_faq",
            "description": "FAQ を検索する",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
]


def _pair(_material: dict[str, Any]) -> dict[str, Any]:
    """全ケースを採用する最小の pair_builder（透過テスト用）。"""
    return {"preferred_output": "よい応答", "non_preferred_output": "わるい応答"}


async def test_tools_pass_through_inside_input() -> None:
    """`tools=` は委譲先 `to_dpo_dataset` へ素通しされ、`record["input"]["tools"]` に載る（P1）。

    透過位置は `input` 内（SFT のレコード直下との差）であり、階層の取り違えを固定する。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    result = await dpo_dataset_from_session(session, pair_builder=_pair, tools=_PLAIN_TOOLS)

    assert [record["input"]["tools"] for record in result.records] == [
        _PLAIN_TOOLS,
        _PLAIN_TOOLS,
    ]
    assert "tools" not in result.records[0]


async def test_tools_function_tool_like_is_mapped_by_delegate() -> None:
    """`FunctionTool` 相当オブジェクトは委譲先の写像を経て dict で `input` へ載る（P2）。"""
    session = FakeSession(_SAMPLE_ITEMS)
    tool = _FakeFunctionTool(
        name="lookup_faq",
        params_json_schema={"type": "object"},
        description="FAQ を検索する",
    )

    result = await dpo_dataset_from_session(session, pair_builder=_pair, tools=[tool])

    assert result.records[0]["input"]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup_faq",
                "description": "FAQ を検索する",
                "parameters": {"type": "object"},
            },
        }
    ]


async def test_invalid_tools_element_raises_validation_failed() -> None:
    """不正要素を含む `tools=` は委譲先の検証で VALIDATION_FAILED になる（P3）。"""
    session = FakeSession(_SAMPLE_ITEMS)

    with pytest.raises(FineTuneError) as exc_info:
        await dpo_dataset_from_session(session, pair_builder=_pair, tools=[42])

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "tools[0]" in exc_info.value.message


async def test_invalid_tools_raises_even_when_all_cases_are_skipped() -> None:
    """pair_builder が全件 skip しても不正 `tools=` は VALIDATION_FAILED になる（P4(b)）。

    採用 0 件で委譲せず早期 return する変異（不正 tools が素通りする経路）が RED になる。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    with pytest.raises(FineTuneError) as exc_info:
        await dpo_dataset_from_session(session, pair_builder=lambda _material: None, tools=[42])

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


async def test_parallel_tool_calls_false_is_passed_through_inside_input() -> None:
    """`parallel_tool_calls=False` は `input` 内へ載る（P5・truthy 判定への退行を検知）。"""
    session = FakeSession(_SAMPLE_ITEMS)

    result = await dpo_dataset_from_session(session, pair_builder=_pair, parallel_tool_calls=False)

    assert [record["input"]["parallel_tool_calls"] for record in result.records] == [False, False]


async def test_tool_keys_are_absent_when_arguments_are_omitted() -> None:
    """未指定なら `input` 内に tools / parallel_tool_calls のキー自体が出ない（P12）。

    `bool(...)` 型の変異（None -> False）はキーが混入する側に倒れるため P5 では
    捉えられない。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    result = await dpo_dataset_from_session(session, pair_builder=_pair)

    for record in result.records:
        assert "tools" not in record["input"]
        assert "parallel_tool_calls" not in record["input"]


async def test_draft_mode_with_tools_raises_config_missing_before_reading_session() -> None:
    """雛形モードでの `tools=` 指定は CONFIG_MISSING で、履歴を読む前に落ちる（P6）。

    記入用ケース列は `to_dpo_dataset` へ委譲しないため反映先が無く、silent に無視しない。
    ガードを削除する変異・`fetch_session_items` の後ろへ移す変異が RED になる
    （`get_items` に到達していないことを `calls == []` で測る）。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    with pytest.raises(FineTuneError) as exc_info:
        await dpo_dataset_from_session(session, tools=_PLAIN_TOOLS)

    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert session.calls == []


async def test_draft_mode_with_parallel_tool_calls_false_raises_config_missing() -> None:
    """雛形モードでは `parallel_tool_calls=False` 単独の指定でも CONFIG_MISSING（P7）。

    `if tools or parallel_tool_calls:` のような truthy 判定への退行（False を未指定と
    みなす）が RED になる。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    with pytest.raises(FineTuneError) as exc_info:
        await dpo_dataset_from_session(session, parallel_tool_calls=False)

    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert session.calls == []


async def test_draft_mode_cases_keep_exactly_four_keys() -> None:
    """雛形モードの記入用ケースのキー集合は 4 キーと完全一致する（P8）。

    tools を記入用ケースへ持ち回る変異（CSV 6 列契約・`input_json` の語義に触れる案）が
    RED になる。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    result = await dpo_dataset_from_session(session)

    for record in result.records:
        assert set(record) == {"input", "preferred_output", "non_preferred_output", "response"}


async def test_context_with_dangling_tool_call_is_generated_not_skipped() -> None:
    """DPO 側でも、ツール往復の並びを理由にケースを捨てない（形式判定は screening の責務）。

    素材の切り出しは SFT と共通のため、DPO だけに判定を戻す変異も RED になる。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "調べて"},
            {"type": "function_call", "name": "f", "arguments": "{}", "call_id": "c1"},
            {"role": "assistant", "content": "少々お待ちください"},
            {"type": "function_call_output", "call_id": "c1", "output": "結果"},
            {"role": "assistant", "content": "結果はこうです"},
        ]
    )

    result = await dpo_dataset_from_session(session)

    assert result.skipped == 0
    assert len(result.records) == 2
    assert result.records[-1]["response"] == "結果はこうです"


async def test_closed_tool_roundtrip_context_is_not_skipped_in_dpo() -> None:
    """往復が閉じている履歴では skip が起きない（何でも skip する変異の過大側を検知）。"""
    session = FakeSession(_SAMPLE_ITEMS)

    result = await dpo_dataset_from_session(session)

    assert result.skipped == 0
    assert len(result.records) == 2


def test_dpo_dataset_from_session_tool_arguments_are_keyword_only_with_none_default() -> None:
    """`tools` / `parallel_tool_calls` は keyword-only かつ既定 None（P10）。"""
    parameters = inspect.signature(dpo_dataset_from_session).parameters

    for name in ("tools", "parallel_tool_calls"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is None


async def test_generated_records_pass_validate_dataset_with_structured_tool_output() -> None:
    """構造化 `output` を含む履歴の DPO レコードが validate_dataset に違反 0 件で通る。

    tool メッセージの content の型写像が退行すると、生成は成功したまま validate 違反レコード
    （プラットフォームが拒否するレコード）を産む fail-open が再発する（ADR 0036）。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "在庫と ID を教えて"},
            {"type": "function_call", "name": "stock", "arguments": "{}", "call_id": "call_1"},
            {"type": "function_call_output", "call_id": "call_1", "output": {"count": 3}},
            {"type": "function_call", "name": "ids", "arguments": "{}", "call_id": "call_2"},
            {"type": "function_call_output", "call_id": "call_2", "output": [1, 2, 3]},
            {"type": "function_call", "name": "note", "arguments": "{}", "call_id": "call_3"},
            {"type": "function_call_output", "call_id": "call_3"},
            {"role": "assistant", "content": "3 個あります"},
        ]
    )

    result = await dpo_dataset_from_session(
        session,
        pair_builder=lambda case: {
            "preferred_output": case["response"],
            "non_preferred_output": "わかりません",
        },
    )

    report = validate_dataset(list(result.records), method="dpo")
    assert report.violations == ()
    assert report.ok is True
