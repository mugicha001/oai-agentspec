"""L2: `dataset_from_session`（FR-4）の生成規則・読み取り専用契約・エラー方針を固定する。

fake Session（`_helpers.fake_session.FakeSession`・呼び出しメソッド記録付き）を通して
「`dataset_from_session` -> `_adapters.finetune.fetch_session_items` -> Session」の鎖を測る。

固定する契約（要件書 FR-4 の正規化規則・ADR 0033 / ADR 0034）:
    - 正規化（採用）: role が user / assistant の item はテキストターンへ。parts 配列 content は
      str へ吸収する
    - 正規化（変換保持）: 対応済み function_call は tool_calls 付き assistant メッセージへ、
      対応済み function_call_output は role "tool" メッセージへ 1:1 の決定的写像で変換して
      文脈へ残す（`arguments` は解釈・改変せず透過。`output` は内容を解釈せず content の型
      へ写す = str はそのまま / 非 str は `json.dumps(ensure_ascii=False, default=str)` の
      JSON 文字列 / キー無し・None は空文字。直列化不能な残余は VALIDATION_FAILED）
    - 併合: 破棄対象 item を取り除いた列（射影列）の上で連続する function_call を 1 つの
      assistant の tool_calls 配列へ併合する。出力ターンを生まない item（reasoning・孤児・
      非 dict 等）は透明として跨ぐが、出力ターンを生む item（function_call_output・
      テキストターン）が挟まれば併合せず独立の assistant メッセージにする
    - 正規化（破棄）: 孤児 function_call / function_call_output（call_id の対応相手が無い）は
      当該 item のみ破棄。非 function 系の補助 item（reasoning / compaction /
      web_search_call 等）と生 role の system / developer / tool item も破棄
      （いずれも skipped に数えない・ケースは維持する）
    - 累積ペアリング: ケース化対象はテキスト応答の assistant ターンのみ（変換済みツール
      メッセージは文脈にのみ現れる）。各テキスト assistant ターンを expected_output とし、
      それ以前の全採用ターンを input とするケースを 1 件生成。input が空になるケース
      （先頭 assistant）は skipped へ計上
    - filter -> transform の順で適用。filter 除外は skipped へ計上。filter 全滅は
      `DatasetBuildResult(records=(), skipped=全件)` の正常返却（エラーにしない）
    - 読み取り専用: Session へは `get_items` のみ（書込系メソッドを一切呼ばない）
    - tools 透過: `tools=` / `parallel_tool_calls=` は写像も検証もせず `to_sft_dataset` へ
      素通しし、レコード直下へ載る（採用 0 件でも委譲するため不正 tools は必ず表面化する）
    - エラー: session=None は CONFIG_MISSING、空履歴 / 抽出可能ターンなし / assistant なし /
      transform の不正戻り値は VALIDATION_FAILED

Session Protocol への duck typing 接触を含むため層は L2（`@pytest.mark.integration`）。
ネットワーク非接触は conftest の autouse ガードが担保する。
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import pytest

from oai_agentspec.runtime.finetune import (
    FineTuneError,
    FineTuneFailureKind,
    dataset_from_session,
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

# `_SAMPLE_ITEMS` の function_call が変換される tool_calls 付き assistant メッセージ。
_SAMPLE_TOOL_CALL_MESSAGE = {
    "role": "assistant",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup_faq", "arguments": '{"q":"register"}'},
        }
    ],
}

# `_SAMPLE_ITEMS` の function_call_output が変換される role "tool" メッセージ。
_SAMPLE_TOOL_OUTPUT_MESSAGE = {"role": "tool", "tool_call_id": "call_1", "content": "{...}"}

# 上記から生成される最終レコード列（parts は str へ吸収済み・ツール往復は文脈へ変換保持）。
_SAMPLE_RECORDS = (
    {
        "messages": [
            {"role": "user", "content": "会員登録の手順を教えて"},
            _SAMPLE_TOOL_CALL_MESSAGE,
            _SAMPLE_TOOL_OUTPUT_MESSAGE,
            {"role": "assistant", "content": "手順は次の通りです: ..."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "会員登録の手順を教えて"},
            _SAMPLE_TOOL_CALL_MESSAGE,
            _SAMPLE_TOOL_OUTPUT_MESSAGE,
            {"role": "assistant", "content": "手順は次の通りです: ..."},
            {"role": "user", "content": "料金は?"},
            {"role": "assistant", "content": "月額 500 円です"},
        ]
    },
)


# ----------------------------------------------------------------------
# 正常系（累積ペアリング・正規化・読み取り専用）
# ----------------------------------------------------------------------


async def test_multiturn_history_yields_cumulative_records() -> None:
    """複数ターン履歴 (a) から累積ペアリングで 2 レコードが生成される。

    parts 配列 content は str へ吸収され、function_call / function_call_output は
    それぞれ tool_calls 付き assistant メッセージ / role "tool" メッセージへ変換されて
    両レコードの文脈に現れる（ツール往復の文脈保持・ADR 0034 Decision 1）。ケース化の
    対象はテキスト応答の assistant ターンのみで、tool_calls 付き assistant はケースを
    生まず skipped にも数えない。`records` は全体を `==` で照合する（部分照合にしない）。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    result = await dataset_from_session(session)

    assert result.records == _SAMPLE_RECORDS
    assert result.skipped == 0


async def test_session_access_is_read_only() -> None:
    """Session へは `get_items` のみ呼ばれ、書込系メソッドは 1 回も呼ばれない（FR-4）。"""
    session = FakeSession(_SAMPLE_ITEMS)

    await dataset_from_session(session)

    assert session.calls == ["get_items"]
    assert "add_items" not in session.calls
    assert "pop_item" not in session.calls
    assert "clear_session" not in session.calls


async def test_leading_assistant_case_is_skipped_not_failed() -> None:
    """先頭 assistant のケース（空 input）は生成せず skipped へ計上する。

    [assistant, user, assistant] の履歴では 2 件目の assistant のケース 1 件のみが
    生成され、input には先頭 assistant ターンも文脈として含まれる。
    """
    session = FakeSession(
        [
            {"role": "assistant", "content": "こんにちは、何かお手伝いできますか?"},
            {"role": "user", "content": "料金は?"},
            {"role": "assistant", "content": "月額 500 円です"},
        ]
    )

    result = await dataset_from_session(session)

    assert result.records == (
        {
            "messages": [
                {"role": "assistant", "content": "こんにちは、何かお手伝いできますか?"},
                {"role": "user", "content": "料金は?"},
                {"role": "assistant", "content": "月額 500 円です"},
            ]
        },
    )
    assert result.skipped == 1


async def test_single_assistant_only_history_returns_empty_result_not_error() -> None:
    """assistant 1 件のみの履歴は例外にせず `records == ()` かつ `skipped == 1` を正常返却する。

    assistant 存在チェック（抽出段階）は通過し、唯一のケースが空 input で skip された
    結果の 0 件は、filter 全滅と同じく正常返却になる境界の pin（S3）。
    """
    session = FakeSession([{"role": "assistant", "content": "こんにちは"}])

    result = await dataset_from_session(session)

    assert result.records == ()
    assert result.skipped == 1


async def test_system_items_are_dropped_without_skipped_count() -> None:
    """履歴中の system / developer item はレコードへ現れず、skipped にも数えない。

    破棄はターン単位・skipped はケース単位の除外件数の語彙であり、混ぜない
    （設計方針 [WARN]-1）。
    """
    session = FakeSession(
        [
            {"role": "system", "content": "あなたはサポート担当です"},
            {"role": "user", "content": "料金は?"},
            {"role": "developer", "content": "内部指示"},
            {"role": "assistant", "content": "月額 500 円です"},
        ]
    )

    result = await dataset_from_session(session)

    assert result.records == (
        {
            "messages": [
                {"role": "user", "content": "料金は?"},
                {"role": "assistant", "content": "月額 500 円です"},
            ]
        },
    )
    assert result.skipped == 0


async def test_empty_expected_output_case_is_skipped_not_recorded() -> None:
    """吸収後の content が空になる assistant ターンは学習ケースにせず skipped へ計上する。

    text フィールドを持たない parts のみの assistant 応答（refusal 等）は `_content_text`
    で空文字になる。これを expected_output にすると「空出力を教える」レコードが silent に
    混入するため、ケースを生成せず skipped に数える（空 input ケースと同じ意味論）。
    文脈（input）側には空 content のターンとして残ることも合わせて固定する。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "これはできますか?"},
            {"role": "assistant", "content": [{"type": "refusal", "refusal": "お断りします"}]},
            {"role": "user", "content": "では別の質問です"},
            {"role": "assistant", "content": "はい、それは可能です"},
        ]
    )

    result = await dataset_from_session(session)

    assert result.records == (
        {
            "messages": [
                {"role": "user", "content": "これはできますか?"},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "では別の質問です"},
                {"role": "assistant", "content": "はい、それは可能です"},
            ]
        },
    )
    assert result.skipped == 1


async def test_all_assistant_outputs_empty_returns_empty_result() -> None:
    """全 assistant 応答が空へ吸収される履歴は、エラーにせず空の結果（skipped=件数）を返す。"""
    session = FakeSession(
        [
            {"role": "user", "content": "質問"},
            {"role": "assistant", "content": [{"type": "refusal", "refusal": "no"}]},
        ]
    )

    result = await dataset_from_session(session)

    assert result.records == ()
    assert result.skipped == 1


async def test_normalization_boundaries_for_malformed_items_and_content() -> None:
    """content=None / 未知型 content / 非 dict item の正規化境界を pin する。

    - content=None のターンは空文字 `""` としてレコードへ載る（`"None"` 文字列の
      silent 混入を防ぐ）
    - 非 str・非 list の未知型 content は `str()` 変換結果（`42` -> `"42"`）が載る
    - 履歴中の非 dict 要素（文字列等）は例外にせず無言で読み飛ばす（ターン破棄と同じく
      skipped に数えない）
    - role の採用判定は厳密な文字列一致（小文字限定・ADR 0033 Decision 2）。非小文字の
      role バリアント（"User" / "ASSISTANT"）はターン破棄され、レコードへ現れず skipped
      にも数えない（判定を大文字小文字非区別へ緩和する変異を検知する pin）
    """
    session = FakeSession(
        [
            "壊れた文字列 item",  # type: ignore[list-item]
            {"role": "User", "content": "大文字始まり role は破棄される"},
            {"role": "user", "content": None},
            {"role": "assistant", "content": "承知しました"},
            {"role": "ASSISTANT", "content": "全大文字 role も破棄される"},
            {"role": "user", "content": 42},
            {"role": "assistant", "content": "回答です"},
        ]
    )

    result = await dataset_from_session(session)

    assert result.records == (
        {
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "承知しました"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "承知しました"},
                {"role": "user", "content": "42"},
                {"role": "assistant", "content": "回答です"},
            ]
        },
    )
    assert result.skipped == 0


async def test_multiple_text_parts_are_concatenated_without_separator() -> None:
    """複数の text parts は separator なしで連結される（ADR 0033 Decision 4 の連結規則）。

    原型（`_session_store._content_text`）と同型の `"".join` であることを pin する
    （separator を混入させる変異 = 学習データ本文の silent 汚染を検知する）。text を
    持たない part は無視されることも合わせて固定する。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "質問"},
            {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "A"},
                    {"type": "refusal", "refusal": "text キーなしは無視"},
                    {"type": "output_text", "text": "B"},
                ],
            },
        ]
    )

    result = await dataset_from_session(session)

    assert result.records == (
        {
            "messages": [
                {"role": "user", "content": "質問"},
                {"role": "assistant", "content": "AB"},
            ]
        },
    )
    assert result.skipped == 0


async def test_system_argument_prepends_to_all_records() -> None:
    """`system=` 指定時は全レコードの messages 先頭へ system メッセージが載る。

    履歴に system item があっても事前破棄されるため `to_sft_dataset` の競合検出は
    発火しない（本経路で dead path・ADR 0033）。
    """
    session = FakeSession(
        [
            {"role": "system", "content": "実行時 instructions"},
            *_SAMPLE_ITEMS,
        ]
    )

    result = await dataset_from_session(session, system="サポート担当として答える")

    assert len(result.records) == 2
    for record in result.records:
        assert record["messages"][0] == {
            "role": "system",
            "content": "サポート担当として答える",
        }
    assert result.records[1]["messages"] == [
        {"role": "system", "content": "サポート担当として答える"},
        {"role": "user", "content": "会員登録の手順を教えて"},
        _SAMPLE_TOOL_CALL_MESSAGE,
        _SAMPLE_TOOL_OUTPUT_MESSAGE,
        {"role": "assistant", "content": "手順は次の通りです: ..."},
        {"role": "user", "content": "料金は?"},
        {"role": "assistant", "content": "月額 500 円です"},
    ]


# ----------------------------------------------------------------------
# エラー方針（CONFIG_MISSING / VALIDATION_FAILED）
# ----------------------------------------------------------------------


async def test_none_session_raises_config_missing_before_get_items() -> None:
    """session=None は CONFIG_MISSING で失敗する（get_items を呼ぶ前・fake 不要）。"""
    with pytest.raises(FineTuneError) as exc_info:
        await dataset_from_session(None)

    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING


async def test_empty_history_raises_validation_failed() -> None:
    """空履歴は VALIDATION_FAILED で失敗する（空データセットを暗黙に返さない）。"""
    session = FakeSession([])

    with pytest.raises(FineTuneError) as exc_info:
        await dataset_from_session(session)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


async def test_history_without_extractable_turns_raises_validation_failed() -> None:
    """孤児 function_call と system item のみの履歴は VALIDATION_FAILED で失敗する。

    対応する function_call_output が無い function_call は孤児として当該 item のみ破棄される
    ため（ADR 0034 Decision 3）、採用ターンが 1 件も残らない。
    """
    session = FakeSession(
        [
            {"type": "function_call", "name": "f", "arguments": "{}", "call_id": "c1"},
            {"role": "system", "content": "指示"},
        ]
    )

    with pytest.raises(FineTuneError) as exc_info:
        await dataset_from_session(session)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


async def test_history_without_assistant_turns_raises_validation_failed() -> None:
    """user のみの履歴（assistant ターン 0 件）は VALIDATION_FAILED で失敗する。"""
    session = FakeSession([{"role": "user", "content": "料金は?"}])

    with pytest.raises(FineTuneError) as exc_info:
        await dataset_from_session(session)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


# ----------------------------------------------------------------------
# case_filter / case_transform
# ----------------------------------------------------------------------


async def test_case_filter_excludes_cases_into_skipped() -> None:
    """filter が False を返したケースは records から除外され skipped に計上される。"""
    session = FakeSession(_SAMPLE_ITEMS)

    result = await dataset_from_session(
        session,
        case_filter=lambda case: case["expected_output"] != "手順は次の通りです: ...",
    )

    assert result.records == (_SAMPLE_RECORDS[1],)
    assert result.skipped == 1


async def test_filter_rejecting_all_cases_returns_empty_result_not_error() -> None:
    """filter 全滅は例外にせず `records == ()` かつ `skipped == 全ケース数` を正常返却する。

    FR-4 のエラー条件は filter 適用前の抽出段階のみ（確定案 (a)・2026-08-26 承認）。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    result = await dataset_from_session(session, case_filter=lambda _case: False)

    assert result.records == ()
    assert result.skipped == 2


async def test_case_transform_is_applied_to_records() -> None:
    """transform の戻り dict が採用される（マスキング例: input の文言を置換）。"""
    session = FakeSession(_SAMPLE_ITEMS)

    def mask(case: dict[str, Any]) -> dict[str, Any]:
        # 変換済みツールメッセージ（tool_calls 付き assistant）は content キーを持たないため、
        # 存在するものだけを書き換える（マスキング実装側の前提の pin も兼ねる）。
        masked_input = [
            {**message, "content": str(message["content"]).replace("会員登録", "[MASKED]")}
            if "content" in message
            else message
            for message in case["input"]
        ]
        return {**case, "input": masked_input}

    result = await dataset_from_session(session, case_transform=mask)

    assert result.records[0]["messages"][0] == {
        "role": "user",
        "content": "[MASKED]の手順を教えて",
    }
    assert result.records[1]["messages"][0] == {
        "role": "user",
        "content": "[MASKED]の手順を教えて",
    }
    assert result.skipped == 0


async def test_transform_is_not_called_for_filtered_out_cases() -> None:
    """適用順は filter -> transform（filter 除外済みケースへ transform を呼ばない）。"""
    session = FakeSession(_SAMPLE_ITEMS)
    filter_seen: list[str] = []
    transform_seen: list[str] = []

    def keep_only_second(case: dict[str, Any]) -> bool:
        filter_seen.append(case["expected_output"])
        return case["expected_output"] == "月額 500 円です"

    def record_transform(case: dict[str, Any]) -> dict[str, Any]:
        transform_seen.append(case["expected_output"])
        return case

    await dataset_from_session(
        session, case_filter=keep_only_second, case_transform=record_transform
    )

    assert filter_seen == ["手順は次の通りです: ...", "月額 500 円です"]
    assert transform_seen == ["月額 500 円です"]


async def test_transform_returning_non_dict_raises_validation_failed() -> None:
    """transform が dict 以外を返した場合は VALIDATION_FAILED で失敗する（暗黙スキップしない）。

    メッセージが `case_transform` を名指しすることまで固定する（発生源が
    session_dataset 側の検証であることの pin。この検証を外す変異は、非 dict ケースが
    下流 `to_sft_dataset` の別文言へ落ちるため RED になる）。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    with pytest.raises(FineTuneError) as exc_info:
        await dataset_from_session(session, case_transform=lambda _case: "not-a-dict")  # type: ignore[arg-type, return-value]

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "case_transform" in exc_info.value.message


async def test_transform_breaking_case_propagates_dataset_validation_error() -> None:
    """transform がケースを破壊（input を空リスト化）すると to_sft_dataset 側の検証が伝播する。

    伝播する VALIDATION_FAILED のメッセージにはケース位置（`ケース N`）が含まれる。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    with pytest.raises(FineTuneError) as exc_info:
        await dataset_from_session(session, case_transform=lambda case: {**case, "input": []})

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース" in exc_info.value.message


# ----------------------------------------------------------------------
# ツール往復の変換保持（ADR 0034 Decision 1-4 / FR-11 の正規化規則）
# ----------------------------------------------------------------------


async def test_tool_roundtrip_items_are_converted_into_context_messages() -> None:
    """function_call / function_call_output が chat 形式へ 1:1 で変換され文脈に残る。

    変換写像の pin: function_call は `tool_calls`（`id` / `type` / `function.name` /
    `function.arguments`）付きの assistant メッセージへ、function_call_output は role
    `"tool"`（`tool_call_id` / `content`）へ写す。`arguments` / `output` の中身は解釈・
    改変せず透過する（JSON 文字列がそのまま載る）。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "為替を調べて"},
            {
                "type": "function_call",
                "name": "get_rate",
                "arguments": '{"pair":"USDJPY"}',
                "call_id": "call_x",
            },
            {"type": "function_call_output", "call_id": "call_x", "output": '{"rate":150.5}'},
            {"role": "assistant", "content": "150.5 円です"},
        ]
    )

    result = await dataset_from_session(session)

    assert result.records == (
        {
            "messages": [
                {"role": "user", "content": "為替を調べて"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_x",
                            "type": "function",
                            "function": {"name": "get_rate", "arguments": '{"pair":"USDJPY"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_x", "content": '{"rate":150.5}'},
                {"role": "assistant", "content": "150.5 円です"},
            ]
        },
    )
    assert result.skipped == 0


async def test_non_str_tool_output_is_serialized_into_json_string_content() -> None:
    """非文字列の `output` は `json.dumps(ensure_ascii=False, default=str)` で JSON 文字列へ写す。

    `_content_text` の parts 吸収（素の配列が空文字へ潰れる）と `str()`（Python repr で
    再パース不能）を挟む変異を検知する pin（ADR 0036）。日本語値が展開されたまま載ること
    （`ensure_ascii=False` で `\\uXXXX` へエスケープしない）と、素の配列が空文字に
    ならないことも同時に固定する。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "在庫を確認して"},
            {"type": "function_call", "name": "stock", "arguments": "{}", "call_id": "call_s"},
            {"type": "function_call_output", "call_id": "call_s", "output": {"count": 3}},
            {"type": "function_call", "name": "shop", "arguments": "{}", "call_id": "call_j"},
            {"type": "function_call_output", "call_id": "call_j", "output": {"shop": "新宿店"}},
            {"type": "function_call", "name": "ids", "arguments": "{}", "call_id": "call_l"},
            {"type": "function_call_output", "call_id": "call_l", "output": [1, 2, 3]},
            {"role": "assistant", "content": "3 個あります"},
        ]
    )

    result = await dataset_from_session(session)

    messages = result.records[0]["messages"]
    tool_messages = [message for message in messages if message["role"] == "tool"]
    assert tool_messages == [
        {"role": "tool", "tool_call_id": "call_s", "content": '{"count": 3}'},
        {"role": "tool", "tool_call_id": "call_j", "content": '{"shop": "新宿店"}'},
        {"role": "tool", "tool_call_id": "call_l", "content": "[1, 2, 3]"},
    ]


async def test_str_tool_output_is_passed_through_unchanged() -> None:
    """str の `output` は写像を挟まずそのまま content へ載る（二重直列化しない）。

    既存 `test_tool_roundtrip_items_are_converted_into_context_messages` と検知範囲が重複する
    ことを承知の上で、str 分岐の明示 pin として置く（ADR 0036）。str 分岐を削除して常に
    `json.dumps` する変異では `'"晴れ"'` / `'"{\\"rate\\":150.5}"'` になり RED。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "天気とレートは?"},
            {"type": "function_call", "name": "weather", "arguments": "{}", "call_id": "call_w"},
            {"type": "function_call_output", "call_id": "call_w", "output": "晴れ"},
            {"type": "function_call", "name": "rate", "arguments": "{}", "call_id": "call_r"},
            {"type": "function_call_output", "call_id": "call_r", "output": '{"rate":150.5}'},
            {"role": "assistant", "content": "晴れ、150.5 円です"},
        ]
    )

    result = await dataset_from_session(session)

    tool_messages = [
        message for message in result.records[0]["messages"] if message["role"] == "tool"
    ]
    assert tool_messages == [
        {"role": "tool", "tool_call_id": "call_w", "content": "晴れ"},
        {"role": "tool", "tool_call_id": "call_r", "content": '{"rate":150.5}'},
    ]


async def test_tool_output_key_missing_becomes_empty_string_content() -> None:
    """`output` キー欠落 / `output: None` はいずれも空文字の content になる。

    `json.dumps(None)` の `"null"` を載せる変異と、`content` キー自体を落とす変異を検知する
    pin（ADR 0036。role "tool" は content が文字列必須）。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "確認して"},
            {"type": "function_call", "name": "f1", "arguments": "{}", "call_id": "call_m"},
            {"type": "function_call_output", "call_id": "call_m"},
            {"type": "function_call", "name": "f2", "arguments": "{}", "call_id": "call_n"},
            {"type": "function_call_output", "call_id": "call_n", "output": None},
            {"role": "assistant", "content": "確認しました"},
        ]
    )

    result = await dataset_from_session(session)

    tool_messages = [
        message for message in result.records[0]["messages"] if message["role"] == "tool"
    ]
    assert tool_messages == [
        {"role": "tool", "tool_call_id": "call_m", "content": ""},
        {"role": "tool", "tool_call_id": "call_n", "content": ""},
    ]
    assert all("content" in message for message in tool_messages)


async def test_non_serializable_tool_output_keeps_outer_json_structure() -> None:
    """直列化できない値だけが `default=str` で文字列へ落ち、外側の JSON 構造は保たれる。

    `str()` 全体フォールバックへ戻す変異（正常な兄弟キーまで Python repr になる）と
    `default=str` を落とす変異を検知する pin（ADR 0036）。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "ID を教えて"},
            {"type": "function_call", "name": "ids", "arguments": "{}", "call_id": "call_x"},
            {
                "type": "function_call_output",
                "call_id": "call_x",
                "output": {"ids": {1}, "shop": "新宿店"},
            },
            {"role": "assistant", "content": "1 件です"},
        ]
    )

    result = await dataset_from_session(session)

    content = result.records[0]["messages"][2]["content"]
    assert content == '{"ids": "{1}", "shop": "新宿店"}'
    assert json.loads(content) == {"ids": "{1}", "shop": "新宿店"}


async def test_circular_tool_output_raises_validation_failed() -> None:
    """循環参照を含む `output` は silent 劣化させず VALIDATION_FAILED で失敗する。

    `str()` フォールバックを復活させる変異（再パース不能な文字列が silent 混入する）を
    検知する pin（ADR 0036）。エラーメッセージは当該 call_id を示す。
    """
    circular: dict[str, Any] = {"self": None}
    circular["self"] = circular
    session = FakeSession(
        [
            {"role": "user", "content": "循環を返して"},
            {"type": "function_call", "name": "loop", "arguments": "{}", "call_id": "call_c"},
            {"type": "function_call_output", "call_id": "call_c", "output": circular},
            {"role": "assistant", "content": "返しました"},
        ]
    )

    with pytest.raises(FineTuneError) as exc_info:
        await dataset_from_session(session)

    assert exc_info.value.kind is FineTuneFailureKind.VALIDATION_FAILED
    assert "call_c" in str(exc_info.value)


async def test_generated_records_pass_validate_dataset_with_structured_tool_output() -> None:
    """構造化 `output` を含む履歴から生成したレコードが validate_dataset に違反 0 件で通る。

    型写像が退行するとビルドは成功したまま validate 違反レコード（プラットフォームが拒否する
    レコード）を産む fail-open が再発するため、上位で機械的に固定する（ADR 0036）。
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

    result = await dataset_from_session(session)

    report = validate_dataset(list(result.records), method="sft")
    assert report.violations == ()
    assert report.ok is True


async def test_directly_adjacent_function_calls_are_merged_into_one_assistant() -> None:
    """直接隣接する function_call は 1 つの assistant の tool_calls 配列へ併合される。

    並列ツール呼び出しの表現の pin。`tool_calls` が 2 要素になることまで固定するため、
    併合結果を先頭 1 件へ切り詰める変異も RED になる（ADR 0034 Decision 2）。
    function_call_output は併合せず、それぞれ独立の tool メッセージになる。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "東京と大阪の天気は?"},
            {
                "type": "function_call",
                "name": "weather",
                "arguments": '{"city":"tokyo"}',
                "call_id": "call_a",
            },
            {
                "type": "function_call",
                "name": "weather",
                "arguments": '{"city":"osaka"}',
                "call_id": "call_b",
            },
            {"type": "function_call_output", "call_id": "call_a", "output": "晴れ"},
            {"type": "function_call_output", "call_id": "call_b", "output": "曇り"},
            {"role": "assistant", "content": "東京は晴れ、大阪は曇りです"},
        ]
    )

    result = await dataset_from_session(session)

    assert result.records == (
        {
            "messages": [
                {"role": "user", "content": "東京と大阪の天気は?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "weather", "arguments": '{"city":"tokyo"}'},
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "weather", "arguments": '{"city":"osaka"}'},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_a", "content": "晴れ"},
                {"role": "tool", "tool_call_id": "call_b", "content": "曇り"},
                {"role": "assistant", "content": "東京は晴れ、大阪は曇りです"},
            ]
        },
    )
    assert result.skipped == 0


async def test_function_calls_separated_by_output_are_not_merged() -> None:
    """間に function_call_output を挟む function_call は併合されない（逐次呼び出し）。

    併合対象を広げる変異（生 item 列の隣接を見ずに連続する function_call をまとめる等）を
    検知する過大側の pin。assistant メッセージは 2 件・各 tool_calls は 1 要素になる。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "順に調べて"},
            {"type": "function_call", "name": "f1", "arguments": "{}", "call_id": "call_a"},
            {"type": "function_call_output", "call_id": "call_a", "output": "A"},
            {"type": "function_call", "name": "f2", "arguments": "{}", "call_id": "call_b"},
            {"type": "function_call_output", "call_id": "call_b", "output": "B"},
            {"role": "assistant", "content": "A と B でした"},
        ]
    )

    result = await dataset_from_session(session)

    assert result.records == (
        {
            "messages": [
                {"role": "user", "content": "順に調べて"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "f1", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_a", "content": "A"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "f2", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_b", "content": "B"},
                {"role": "assistant", "content": "A と B でした"},
            ]
        },
    )
    assert result.skipped == 0


async def test_function_calls_separated_by_dropped_item_are_merged() -> None:
    """間に破棄対象 item（reasoning）を挟む function_call は併合される。

    破棄対象（reasoning）は出力ターンを 1 件も生まないため射影列に現れず、跨いで併合する
    （ADR 0036）。生 item 列の隣接で判定する変異が RED になる。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "調べて"},
            {"type": "function_call", "name": "f1", "arguments": "{}", "call_id": "call_a"},
            {"type": "reasoning", "summary": []},
            {"type": "function_call", "name": "f2", "arguments": "{}", "call_id": "call_b"},
            {"type": "function_call_output", "call_id": "call_a", "output": "A"},
            {"type": "function_call_output", "call_id": "call_b", "output": "B"},
            {"role": "assistant", "content": "A と B でした"},
        ]
    )

    result = await dataset_from_session(session)

    assert result.records[0]["messages"][1] == {
        "role": "assistant",
        "tool_calls": [
            {"id": "call_a", "type": "function", "function": {"name": "f1", "arguments": "{}"}},
            {"id": "call_b", "type": "function", "function": {"name": "f2", "arguments": "{}"}},
        ],
    }
    assert len(result.records[0]["messages"]) == 5


async def test_function_calls_separated_by_orphan_call_are_merged() -> None:
    """孤児 function_call / 非 dict item を挟む function_call も射影列上で隣接し併合される。

    孤児と非 dict item はいずれも出力ターンを生まないため射影列に現れない（ADR 0036）。
    孤児自体は破棄されたまま出力に現れないことも同時に固定する。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "調べて"},
            {"type": "function_call", "name": "f1", "arguments": "{}", "call_id": "call_a"},
            {"type": "function_call", "name": "ghost", "arguments": "{}", "call_id": "call_x"},
            "not a dict",
            {"type": "function_call", "name": "f2", "arguments": "{}", "call_id": "call_b"},
            {"type": "function_call_output", "call_id": "call_a", "output": "A"},
            {"type": "function_call_output", "call_id": "call_b", "output": "B"},
            {"role": "assistant", "content": "A と B でした"},
        ]
    )

    result = await dataset_from_session(session)

    messages = result.records[0]["messages"]
    assert messages[1] == {
        "role": "assistant",
        "tool_calls": [
            {"id": "call_a", "type": "function", "function": {"name": "f1", "arguments": "{}"}},
            {"id": "call_b", "type": "function", "function": {"name": "f2", "arguments": "{}"}},
        ],
    }
    assert len(messages) == 5
    assert "call_x" not in str(messages)


async def test_function_calls_separated_by_text_turn_are_not_merged() -> None:
    """間にテキストターンを挟む function_call は併合されない（過大側の pin）。

    テキストターン（user / assistant）は出力ターンを生むため射影列に残り、併合を切る
    （ADR 0036）。テキストターン側の併合状態リセットを落とす変異が RED になる。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "順に調べて"},
            {"type": "function_call", "name": "f1", "arguments": "{}", "call_id": "call_a"},
            {"role": "assistant", "content": "少々お待ちください"},
            {"type": "function_call", "name": "f2", "arguments": "{}", "call_id": "call_b"},
            {"type": "function_call_output", "call_id": "call_a", "output": "A"},
            {"type": "function_call_output", "call_id": "call_b", "output": "B"},
            {"role": "assistant", "content": "A と B でした"},
        ]
    )

    result = await dataset_from_session(session)

    messages = result.records[-1]["messages"]
    tool_call_messages = [message for message in messages if "tool_calls" in message]
    assert tool_call_messages == [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "f1", "arguments": "{}"}}
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_b", "type": "function", "function": {"name": "f2", "arguments": "{}"}}
            ],
        },
    ]


async def test_orphan_function_items_are_dropped_without_dropping_the_case() -> None:
    """call_id の対応相手を欠く function_call / function_call_output は当該 item のみ落ちる。

    孤児の片側だけを学習データへ混入させず、ケース自体はエラー・除外にしない
    （ADR 0034 Decision 3）。skipped はターン破棄で増えない。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "在庫は?"},
            {"type": "function_call", "name": "f", "arguments": "{}", "call_id": "call_missing"},
            {"type": "function_call_output", "call_id": "call_ghost", "output": "捨てられる"},
            {"role": "assistant", "content": "3 個です"},
        ]
    )

    result = await dataset_from_session(session)

    assert result.records == (
        {
            "messages": [
                {"role": "user", "content": "在庫は?"},
                {"role": "assistant", "content": "3 個です"},
            ]
        },
    )
    assert result.skipped == 0


async def test_non_function_tool_items_are_dropped_without_skipped_count() -> None:
    """function 系以外の補助 item（reasoning / compaction / web_search_call）は破棄される。

    chat 形式に対応物が無いため従来どおり破棄し、`skipped`（ケース単位の除外件数）には
    数えない（ADR 0034 Decision 1・破棄規則の存続部分）。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "最新のニュースは?"},
            {"type": "reasoning", "summary": []},
            {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            {"type": "compaction", "content": "要約"},
            {"role": "assistant", "content": "本日の主要ニュースです"},
        ]
    )

    result = await dataset_from_session(session)

    assert result.records == (
        {
            "messages": [
                {"role": "user", "content": "最新のニュースは?"},
                {"role": "assistant", "content": "本日の主要ニュースです"},
            ]
        },
    )
    assert result.skipped == 0


async def test_history_with_only_tool_roundtrip_raises_validation_failed() -> None:
    """テキスト応答の assistant が無くツール往復のみの履歴は VALIDATION_FAILED で失敗する。

    ケース化対象はテキスト応答の assistant ターンのみであり（ADR 0034 Decision 4）、
    変換で生成された tool_calls 付き assistant を「assistant ターンあり」と数えて
    空データセットを返す変異を検知する pin。
    """
    session = FakeSession(
        [
            {"role": "user", "content": "調べて"},
            {"type": "function_call", "name": "f", "arguments": "{}", "call_id": "call_a"},
            {"type": "function_call_output", "call_id": "call_a", "output": "A"},
        ]
    )

    with pytest.raises(FineTuneError) as exc_info:
        await dataset_from_session(session)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


# ----------------------------------------------------------------------
# ツール往復の途中で切れる文脈の skip（dangling tool_calls の抑止）
# ----------------------------------------------------------------------

# function_call と function_call_output の間へ assistant テキストが挟まる履歴
# （HITL 承認で中断されたラン等）。累積ペアリングは往復の途中でも切り出す。
_DANGLING_ITEMS: list[dict[str, Any]] = [
    {"role": "user", "content": "調べて"},
    {"type": "function_call", "name": "f", "arguments": "{}", "call_id": "c1"},
    {"role": "assistant", "content": "少々お待ちください"},
    {"type": "function_call_output", "call_id": "c1", "output": "結果"},
    {"role": "assistant", "content": "結果はこうです"},
]


async def test_context_with_dangling_tool_call_is_skipped() -> None:
    """対応する tool メッセージを欠く tool_calls を含む文脈のケースは skip される。

    推論時 API が拒否する並び（tool_calls に応答が無い）を silent に産む fail-open の pin。
    判定を削除する変異が RED になる。
    """
    session = FakeSession(_DANGLING_ITEMS)

    result = await dataset_from_session(session)

    assert result.skipped == 1
    assert len(result.records) == 1
    # 残るのは往復が閉じた文脈のケース（最後の assistant を expected_output とするもの）。
    assert result.records[0]["messages"][-1] == {
        "role": "assistant",
        "content": "結果はこうです",
    }
    assert any(message.get("role") == "tool" for message in result.records[0]["messages"])


async def test_closed_tool_roundtrip_context_is_not_skipped() -> None:
    """往復が閉じている履歴では skip が起きない（何でも skip する変異の過大側を検知）。"""
    session = FakeSession(_SAMPLE_ITEMS)

    result = await dataset_from_session(session)

    assert result.skipped == 0
    assert len(result.records) == 2


# ----------------------------------------------------------------------
# tools= / parallel_tool_calls= の透過（委譲先 `to_sft_dataset` へ素通し）
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


async def test_tools_pass_through_to_record_top_level() -> None:
    """`tools=` は委譲先 `to_sft_dataset` へ素通しされ、レコード直下 "tools" に載る（P1）。

    透過位置は SFT だけレコード直下（DPO は `input` 内）であり、階層の取り違えを固定する。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    result = await dataset_from_session(session, tools=_PLAIN_TOOLS)

    assert [record["tools"] for record in result.records] == [_PLAIN_TOOLS, _PLAIN_TOOLS]


async def test_tools_function_tool_like_is_mapped_by_delegate() -> None:
    """`FunctionTool` 相当オブジェクトは委譲先の写像を経て dict でレコードへ載る（P2）。

    上位層で raw 透過する（`_map_tools` を経由しない）変異が RED になる。
    """
    session = FakeSession(_SAMPLE_ITEMS)
    tool = _FakeFunctionTool(
        name="lookup_faq",
        params_json_schema={"type": "object"},
        description="FAQ を検索する",
    )

    result = await dataset_from_session(session, tools=[tool])

    assert result.records[0]["tools"] == [
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
    """不正要素を含む `tools=` は委譲先の検証で VALIDATION_FAILED になる（P3）。

    検証を上位層でバイパスして透過する変異が RED になる。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    with pytest.raises(FineTuneError) as exc_info:
        await dataset_from_session(session, tools=[42])

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "tools[0]" in exc_info.value.message


async def test_invalid_tools_raises_even_when_filter_excludes_all_cases() -> None:
    """filter が全件除外しても不正 `tools=` は VALIDATION_FAILED になる（P4(a)）。

    採用 0 件で委譲せず早期 return する変異（不正 tools が素通りする経路）が RED になる。
    返却値そのものは委譲しても同値のため、この状況だけが差を捉えられる。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    with pytest.raises(FineTuneError) as exc_info:
        await dataset_from_session(session, case_filter=lambda _case: False, tools=[42])

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


async def test_parallel_tool_calls_false_is_passed_through() -> None:
    """`parallel_tool_calls=False` はレコードへ載る（P5・truthy 判定への退行を検知）。"""
    session = FakeSession(_SAMPLE_ITEMS)

    result = await dataset_from_session(session, parallel_tool_calls=False)

    assert [record["parallel_tool_calls"] for record in result.records] == [False, False]


async def test_tool_keys_are_absent_when_arguments_are_omitted() -> None:
    """未指定なら tools / parallel_tool_calls のキー自体がレコードへ出ない（P12）。

    `bool(...)` 型の変異（None -> False）はキーが混入する側に倒れるため、P5 では
    捉えられない。要件「省略時はキー自体を出力しない」の pin。
    """
    session = FakeSession(_SAMPLE_ITEMS)

    result = await dataset_from_session(session)

    for record in result.records:
        assert "tools" not in record
        assert "parallel_tool_calls" not in record


def test_dataset_from_session_tool_arguments_are_keyword_only_with_none_default() -> None:
    """`tools` / `parallel_tool_calls` は keyword-only かつ既定 None（P10）。"""
    parameters = inspect.signature(dataset_from_session).parameters

    for name in ("tools", "parallel_tool_calls"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is None
