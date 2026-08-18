"""L1: Fine-Tuning データ変換・検証ヘルパ（純データ層）を検証する。

`to_sft_dataset`（単一 / 複数ターン二形受理・`system=` 先頭挿入と system 競合エラー・出力側の
文字列 / assistant 配列二形・ツール入り messages の透過・`weight` / parts の非改変保全・
`tools=` / `parallel_tool_calls=` のレコードレベル透過・FunctionTool 相当からの FT 形式写像
（strict 非写像・description 既定空文字）・欠落 / 境界の既定エラーと `skip_missing`）、
`to_dpo_dataset`（preference 形式・出力側は assistant 配列・`input.tools` 透過・`system=` を
持たない）、`validate_dataset`（合法集合と role 別制約・weight の SFT 限定検証・content 二形と
空リスト違反・未知キー許容・preference 出力の role 制約・fail-closed・`raise_on_invalid`・
ファイル / dict 列の `line` 意味）を網羅する。すべて純データ操作で外部依存なし
（`@pytest.mark.unit`）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from oai_agentspec.runtime.finetune import (
    DpoCase,
    FineTuneError,
    FineTuneFailureKind,
    to_dpo_dataset,
    to_sft_dataset,
    validate_dataset,
)

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# テスト用 fake（SDK 非依存・ダックタイピング受理の検証用）
# ----------------------------------------------------------------------


class _FakeFunctionTool:
    """SDK `FunctionTool` 相当の属性のみを持つ fake（`agents` を import しない）。"""

    def __init__(
        self,
        name: str,
        params_json_schema: dict[str, Any],
        description: str | None = None,
        strict_json_schema: bool = True,
    ) -> None:
        self.name = name
        self.params_json_schema = params_json_schema
        if description is not None:
            self.description = description
        self.strict_json_schema = strict_json_schema


class _FakeEvalCase:
    """`EvalCase` / `OptimizeCase` 相当の属性アクセス経路を持つ fake。"""

    def __init__(self, input: Any, expected_output: Any) -> None:
        self.input = input
        self.expected_output = expected_output


def _schema() -> dict[str, Any]:
    """FunctionTool 相当 fake が持つ params_json_schema の共通値。"""
    return {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    }


# ----------------------------------------------------------------------
# to_sft_dataset: 単一ターン変換 / system
# ----------------------------------------------------------------------


def test_sft_single_turn_wraps_strings_into_user_and_assistant() -> None:
    """文字列 input は user 1 件・expected_output は assistant 1 件へ包まれる。"""
    result = to_sft_dataset(
        [{"input": "返品ポリシーを教えて", "expected_output": "30 日以内です。"}]
    )

    assert result.skipped == 0
    assert list(result.records) == [
        {
            "messages": [
                {"role": "user", "content": "返品ポリシーを教えて"},
                {"role": "assistant", "content": "30 日以内です。"},
            ]
        }
    ]


def test_sft_accepts_attribute_style_cases() -> None:
    """属性アクセス型のケース（EvalCase / OptimizeCase 相当）も dict と同様に受理する。"""
    result = to_sft_dataset([_FakeEvalCase(input="質問", expected_output="回答")])

    assert list(result.records) == [
        {
            "messages": [
                {"role": "user", "content": "質問"},
                {"role": "assistant", "content": "回答"},
            ]
        }
    ]


def test_sft_custom_input_and_output_keys() -> None:
    """plain dict のキー名は input_key / output_key で指定できる。"""
    result = to_sft_dataset([{"q": "質問", "a": "回答"}], input_key="q", output_key="a")

    assert list(result.records) == [
        {
            "messages": [
                {"role": "user", "content": "質問"},
                {"role": "assistant", "content": "回答"},
            ]
        }
    ]


def test_sft_system_argument_is_prepended() -> None:
    """`system=` は messages 先頭へ挿入される。"""
    result = to_sft_dataset(
        [{"input": "質問", "expected_output": "回答"}], system="サポート担当として答える"
    )

    assert list(result.records)[0]["messages"][0] == {
        "role": "system",
        "content": "サポート担当として答える",
    }


def test_sft_system_inside_input_list_passes_through() -> None:
    """input リスト内の system メッセージのみの場合はそのまま透過する。"""
    messages = [
        {"role": "system", "content": "サポート担当として答える"},
        {"role": "user", "content": "注文をキャンセルしたい"},
    ]
    result = to_sft_dataset([{"input": messages, "expected_output": "承りました。"}])

    assert list(result.records)[0]["messages"][0] == {
        "role": "system",
        "content": "サポート担当として答える",
    }


def test_sft_system_conflict_raises_validation_failed() -> None:
    """`system=` と input リスト内 system の併存はエラー（暗黙マージ・置換をしない）。"""
    messages = [
        {"role": "system", "content": "リスト側"},
        {"role": "user", "content": "質問"},
    ]
    with pytest.raises(FineTuneError) as exc_info:
        to_sft_dataset([{"input": messages, "expected_output": "回答"}], system="引数側")

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


def test_sft_system_conflict_raises_even_with_skip_missing() -> None:
    """system 競合はデータ矛盾であり `skip_missing=True` でも常時エラー（skip 対象外）。"""
    messages = [
        {"role": "system", "content": "リスト側"},
        {"role": "user", "content": "質問"},
    ]
    with pytest.raises(FineTuneError) as exc_info:
        to_sft_dataset(
            [{"input": messages, "expected_output": "回答"}], system="引数側", skip_missing=True
        )

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


# ----------------------------------------------------------------------
# to_sft_dataset: 複数ターン透過 / 出力側二形
# ----------------------------------------------------------------------


def test_sft_multi_turn_messages_pass_through_and_append_assistant() -> None:
    """messages リスト input は非改変透過され、expected_output が末尾 assistant として付く。"""
    messages = [
        {"role": "system", "content": "サポート担当として答える"},
        {"role": "user", "content": "注文をキャンセルしたい"},
        {"role": "assistant", "content": "注文番号をお知らせください。"},
        {"role": "user", "content": "A-1234 です"},
    ]
    result = to_sft_dataset([{"input": messages, "expected_output": "承りました。"}])

    assert list(result.records)[0]["messages"] == [
        *messages,
        {"role": "assistant", "content": "承りました。"},
    ]


def test_sft_output_accepts_assistant_message_array() -> None:
    """expected_output が assistant メッセージ配列なら非改変で透過採用される。"""
    outputs = [
        {"role": "assistant", "content": "回答 1"},
        {"role": "assistant", "content": "回答 2"},
    ]
    result = to_sft_dataset([{"input": "質問", "expected_output": outputs}])

    assert list(result.records)[0]["messages"] == [
        {"role": "user", "content": "質問"},
        *outputs,
    ]


def test_sft_tool_call_messages_round_trip_unchanged() -> None:
    """tool_calls 付き assistant / role "tool" のメッセージは非改変で透過する。"""
    messages = [
        {"role": "user", "content": "A-1234 の配送状況を教えて"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_order_status",
                        "arguments": '{"order_id": "A-1234"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "in_transit"}'},
    ]
    result = to_sft_dataset([{"input": messages, "expected_output": "配送中です。"}])

    built = list(result.records)[0]["messages"]
    assert built[:3] == messages
    assert built[3] == {"role": "assistant", "content": "配送中です。"}


def test_sft_preserves_weight_and_does_not_add_weight_to_appended_assistant() -> None:
    """`weight` は非改変で保全され、末尾付加 assistant には weight を付さない。"""
    messages = [
        {"role": "user", "content": "解約方法を教えて"},
        {"role": "assistant", "content": "マイページから手続きできます。", "weight": 0},
        {"role": "user", "content": "マイページに入れません"},
    ]
    result = to_sft_dataset([{"input": messages, "expected_output": "再設定をご案内します。"}])

    built = list(result.records)[0]["messages"]
    assert built[1] == {
        "role": "assistant",
        "content": "マイページから手続きできます。",
        "weight": 0,
    }
    assert built[-1] == {"role": "assistant", "content": "再設定をご案内します。"}
    assert "weight" not in built[-1]


def test_sft_preserves_content_parts_array() -> None:
    """parts 配列 content（vision）は非改変で透過する（内部構造を解釈しない）。"""
    parts = [
        {"type": "text", "text": "この画像の商品に傷はありますか"},
        {"type": "image_url", "image_url": {"url": "https://example.com/item.jpg"}},
    ]
    messages = [{"role": "user", "content": parts}]
    result = to_sft_dataset([{"input": messages, "expected_output": "擦り傷があります。"}])

    assert list(result.records)[0]["messages"][0] == {"role": "user", "content": parts}


# ----------------------------------------------------------------------
# to_sft_dataset: tools / parallel_tool_calls の透過
# ----------------------------------------------------------------------


def test_sft_tools_dict_passes_through_at_record_level() -> None:
    """plain dict の `tools=` はレコード直下 "tools" へ非解釈で透過する。"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_order_status",
                "description": "注文番号から配送状況を取得する",
                "parameters": _schema(),
            },
        }
    ]
    result = to_sft_dataset([{"input": "質問", "expected_output": "回答"}], tools=tools)

    assert list(result.records)[0]["tools"] == tools


def test_sft_parallel_tool_calls_passes_through() -> None:
    """`parallel_tool_calls=` はレコード直下へ透過する。"""
    result = to_sft_dataset(
        [{"input": "質問", "expected_output": "回答"}], parallel_tool_calls=False
    )

    assert list(result.records)[0]["parallel_tool_calls"] is False


def test_sft_omits_tool_keys_when_arguments_are_none() -> None:
    """既定（None）では tools / parallel_tool_calls キー自体を出力しない。"""
    record = list(to_sft_dataset([{"input": "質問", "expected_output": "回答"}]).records)[0]

    assert "tools" not in record
    assert "parallel_tool_calls" not in record


def test_sft_maps_function_tool_like_object_without_strict() -> None:
    """FunctionTool 相当 fake は FT 形式へ写像され、strict は出力に含まれない。"""
    tool = _FakeFunctionTool(
        name="get_order_status",
        params_json_schema=_schema(),
        description="注文番号から配送状況を取得する",
        strict_json_schema=True,
    )
    result = to_sft_dataset([{"input": "質問", "expected_output": "回答"}], tools=[tool])

    mapped = list(result.records)[0]["tools"]
    assert mapped == [
        {
            "type": "function",
            "function": {
                "name": "get_order_status",
                "description": "注文番号から配送状況を取得する",
                "parameters": _schema(),
            },
        }
    ]
    assert "strict" not in json.dumps(mapped)


def test_sft_maps_function_tool_like_without_description_to_empty_string() -> None:
    """`description` 属性を持たない FunctionTool 相当は description "" で写像する。"""
    tool = _FakeFunctionTool(name="ping", params_json_schema={"type": "object"})
    result = to_sft_dataset([{"input": "質問", "expected_output": "回答"}], tools=[tool])

    assert list(result.records)[0]["tools"][0]["function"]["description"] == ""


def test_sft_tools_accepts_mixed_dict_and_function_tool_like() -> None:
    """dict と FunctionTool 相当の混在リストを両形で正しく処理する。"""
    plain = {"type": "function", "function": {"name": "plain_tool", "parameters": {}}}
    tool = _FakeFunctionTool(
        name="mapped_tool", params_json_schema={"type": "object"}, description="d"
    )
    result = to_sft_dataset([{"input": "質問", "expected_output": "回答"}], tools=[plain, tool])

    mapped = list(result.records)[0]["tools"]
    assert mapped[0] == plain
    assert mapped[1] == {
        "type": "function",
        "function": {"name": "mapped_tool", "description": "d", "parameters": {"type": "object"}},
    }


def test_sft_tools_invalid_element_raises_validation_failed() -> None:
    """dict でも FunctionTool 相当でもない要素はエラー。"""
    with pytest.raises(FineTuneError) as exc_info:
        to_sft_dataset([{"input": "質問", "expected_output": "回答"}], tools=[42])

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


def test_sft_tools_invalid_element_raises_even_with_skip_missing() -> None:
    """`tools=` は呼び出し単位の引数のため `skip_missing=True` でも常時エラー。"""
    with pytest.raises(FineTuneError) as exc_info:
        to_sft_dataset(
            [{"input": "質問", "expected_output": "回答"}], tools=[42], skip_missing=True
        )

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


# ----------------------------------------------------------------------
# to_sft_dataset: 欠落・境界（SHOULD-5）
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        pytest.param({"input": "質問"}, id="expected_output-missing"),
        pytest.param({"input": [], "expected_output": "回答"}, id="input-empty-list"),
        pytest.param({"input": 123, "expected_output": "回答"}, id="input-invalid-type"),
        pytest.param({"input": "質問", "expected_output": []}, id="output-empty-array"),
        pytest.param({"input": "質問", "expected_output": 123}, id="output-invalid-type"),
        pytest.param(
            {"input": [{"role": "user"}], "expected_output": "回答"},
            id="message-without-content-or-tool-calls",
        ),
        pytest.param({"input": [42], "expected_output": "回答"}, id="message-element-not-dict"),
        pytest.param(
            {"input": [{"content": "x"}], "expected_output": "回答"},
            id="message-element-without-role",
        ),
    ],
)
def test_sft_missing_or_boundary_cases_raise_by_default(case: dict[str, Any]) -> None:
    """欠落・境界ケースは既定でエラー（暗黙補完しない）。"""
    with pytest.raises(FineTuneError) as exc_info:
        to_sft_dataset([case])

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


@pytest.mark.parametrize(
    "case",
    [
        pytest.param({"input": "質問"}, id="expected_output-missing"),
        pytest.param({"input": [], "expected_output": "回答"}, id="input-empty-list"),
        pytest.param({"input": 123, "expected_output": "回答"}, id="input-invalid-type"),
        pytest.param({"input": "質問", "expected_output": []}, id="output-empty-array"),
        pytest.param(
            {"input": [{"role": "user"}], "expected_output": "回答"},
            id="message-without-content-or-tool-calls",
        ),
        pytest.param({"input": [42], "expected_output": "回答"}, id="message-element-not-dict"),
        pytest.param(
            {"input": [{"content": "x"}], "expected_output": "回答"},
            id="message-element-without-role",
        ),
    ],
)
def test_sft_missing_or_boundary_cases_are_skipped_when_opted_in(case: dict[str, Any]) -> None:
    """`skip_missing=True` では除外され、skipped に件数が入る（正常ケースは残る）。"""
    result = to_sft_dataset([case, {"input": "質問", "expected_output": "回答"}], skip_missing=True)

    assert result.skipped == 1
    assert len(list(result.records)) == 1


# ----------------------------------------------------------------------
# to_dpo_dataset
# ----------------------------------------------------------------------


def test_dpo_single_turn_builds_preference_record() -> None:
    """DpoCase の文字列 input / 出力は preference 形式へ変換される。"""
    case = DpoCase(
        input="配送状況を確認したい",
        preferred_output="注文番号をお知らせください。",
        non_preferred_output="自分で調べてください。",
    )
    result = to_dpo_dataset([case])

    assert list(result.records) == [
        {
            "input": {"messages": [{"role": "user", "content": "配送状況を確認したい"}]},
            "preferred_output": [{"role": "assistant", "content": "注文番号をお知らせください。"}],
            "non_preferred_output": [{"role": "assistant", "content": "自分で調べてください。"}],
        }
    ]


def test_dpo_outputs_are_always_assistant_arrays() -> None:
    """出力側は単一 dict ではなく assistant メッセージ配列で固定される（SHOULD-1）。"""
    result = to_dpo_dataset([DpoCase(input="q", preferred_output="p", non_preferred_output="n")])

    record = list(result.records)[0]
    assert isinstance(record["preferred_output"], list)
    assert isinstance(record["non_preferred_output"], list)
    assert record["preferred_output"][0]["role"] == "assistant"
    assert record["non_preferred_output"][0]["role"] == "assistant"


def test_dpo_accepts_plain_dict_cases_with_custom_keys() -> None:
    """plain dict ケースも preferred_key / non_preferred_key 指定で受理する。"""
    result = to_dpo_dataset(
        [{"input": "q", "good": "p", "bad": "n"}],
        preferred_key="good",
        non_preferred_key="bad",
    )

    record = list(result.records)[0]
    assert record["preferred_output"] == [{"role": "assistant", "content": "p"}]
    assert record["non_preferred_output"] == [{"role": "assistant", "content": "n"}]


def test_dpo_multi_turn_input_passes_through() -> None:
    """messages リスト input は非改変で input.messages へ透過する。"""
    messages = [
        {"role": "user", "content": "解約したい"},
        {"role": "assistant", "content": "理由をお聞かせください。"},
        {"role": "user", "content": "料金が高いため"},
    ]
    preferred = [{"role": "assistant", "content": "割引プランもございます。"}]
    non_preferred = [{"role": "assistant", "content": "解約はできません。"}]
    result = to_dpo_dataset(
        [DpoCase(input=messages, preferred_output=preferred, non_preferred_output=non_preferred)]
    )

    record = list(result.records)[0]
    assert record["input"]["messages"] == messages
    assert record["preferred_output"] == preferred
    assert record["non_preferred_output"] == non_preferred


def test_dpo_tools_pass_through_inside_input() -> None:
    """`tools=` / `parallel_tool_calls=` は input 内へ透過する（SFT 直下との差）。"""
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    result = to_dpo_dataset(
        [DpoCase(input="q", preferred_output="p", non_preferred_output="n")],
        tools=tools,
        parallel_tool_calls=True,
    )

    record = list(result.records)[0]
    assert record["input"]["tools"] == tools
    assert record["input"]["parallel_tool_calls"] is True
    assert "tools" not in record
    assert "parallel_tool_calls" not in record


def test_dpo_maps_function_tool_like_object() -> None:
    """FunctionTool 相当は SFT と同一規則で FT 形式へ写像される。"""
    tool = _FakeFunctionTool(name="t", params_json_schema={"type": "object"}, description="d")
    result = to_dpo_dataset(
        [DpoCase(input="q", preferred_output="p", non_preferred_output="n")], tools=[tool]
    )

    assert list(result.records)[0]["input"]["tools"] == [
        {
            "type": "function",
            "function": {"name": "t", "description": "d", "parameters": {"type": "object"}},
        }
    ]


def test_dpo_omits_tool_keys_when_arguments_are_none() -> None:
    """既定（None）では input 内に tools / parallel_tool_calls キーを出力しない。"""
    record = list(
        to_dpo_dataset([DpoCase(input="q", preferred_output="p", non_preferred_output="n")]).records
    )[0]

    assert record["input"] == {"messages": [{"role": "user", "content": "q"}]}


@pytest.mark.parametrize(
    "case",
    [
        pytest.param({"input": "q", "non_preferred_output": "n"}, id="preferred-missing"),
        pytest.param({"input": "q", "preferred_output": "p"}, id="non-preferred-missing"),
        pytest.param(
            {"input": "q", "preferred_output": [], "non_preferred_output": "n"},
            id="preferred-empty-array",
        ),
        pytest.param(
            {"input": [], "preferred_output": "p", "non_preferred_output": "n"},
            id="input-empty-list",
        ),
        pytest.param(
            {"input": 1, "preferred_output": "p", "non_preferred_output": "n"},
            id="input-invalid-type",
        ),
    ],
)
def test_dpo_missing_or_boundary_cases_raise_by_default(case: dict[str, Any]) -> None:
    """preferred / non_preferred の欠落と境界ケースは既定でエラー。"""
    with pytest.raises(FineTuneError) as exc_info:
        to_dpo_dataset([case])

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


def test_dpo_skip_missing_excludes_and_counts() -> None:
    """`skip_missing=True` では欠落ケースを除外し skipped に件数が入る。"""
    result = to_dpo_dataset(
        [
            {"input": "q", "non_preferred_output": "n"},
            {"input": "q", "preferred_output": "p", "non_preferred_output": "n"},
        ],
        skip_missing=True,
    )

    assert result.skipped == 1
    assert len(list(result.records)) == 1


def test_dpo_does_not_accept_system_argument() -> None:
    """`to_dpo_dataset` は `system=` を持たない（INFO-1・system は input リストで渡す）。"""
    with pytest.raises(TypeError):
        to_dpo_dataset(
            [DpoCase(input="q", preferred_output="p", non_preferred_output="n")],
            system="サポート担当",  # type: ignore[call-arg]
        )


# ----------------------------------------------------------------------
# validate_dataset: 合格ケース（誤検知を出さない）
# ----------------------------------------------------------------------


def _write_jsonl(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    """レコード列を JSONL ファイルへ書き出してパスを返す（検証テストの共通ヘルパ）。"""
    target = tmp_path / "data.jsonl"
    target.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return target


def test_validate_accepts_valid_tool_record(tmp_path: Path) -> None:
    """正しいツール入り SFT JSONL は合格する（誤検知を出さない）。"""
    record = {
        "messages": [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_order_status", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "in_transit"}'},
            {"role": "assistant", "content": "配送中です。"},
        ],
        "tools": [{"type": "function", "function": {"name": "get_order_status", "parameters": {}}}],
        "parallel_tool_calls": False,
    }
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="sft")

    assert report.ok is True
    assert report.checked == 1
    assert report.violations == ()


def test_validate_accepts_weight_and_content_parts(tmp_path: Path) -> None:
    """weight 入り・parts 入りの正しい SFT JSONL は合格する。"""
    record = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "この画像の商品に傷はありますか"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/i.jpg"}},
                ],
            },
            {"role": "assistant", "content": "傷があります。", "weight": 0},
            {"role": "assistant", "content": "左下の角です。", "weight": 1},
        ]
    }
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="sft")

    assert report.ok is True


def test_validate_accepts_unknown_keys(tmp_path: Path) -> None:
    """メッセージの未知キー・レコードレベルの未知フィールドは違反にしない（INFO-10）。"""
    record = {
        "messages": [
            {"role": "user", "content": "質問", "custom_field": 1},
            {"role": "assistant", "content": "回答"},
        ],
        "record_level_extra": {"a": 1},
    }
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="sft")

    assert report.ok is True


def test_validate_accepts_valid_dpo_record(tmp_path: Path) -> None:
    """正しい DPO preference レコードは合格する。"""
    record = {
        "input": {"messages": [{"role": "user", "content": "配送状況を確認したい"}]},
        "preferred_output": [{"role": "assistant", "content": "注文番号をお知らせください。"}],
        "non_preferred_output": [{"role": "assistant", "content": "自分で調べてください。"}],
    }
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="dpo")

    assert report.ok is True


# ----------------------------------------------------------------------
# validate_dataset: 違反ケース（SFT）
# ----------------------------------------------------------------------


def test_validate_flags_message_without_content_and_tool_calls(tmp_path: Path) -> None:
    """content と tool_calls の両方が無いメッセージは違反。"""
    record = {"messages": [{"role": "user"}, {"role": "assistant", "content": "回答"}]}
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="sft")

    assert report.ok is False
    assert [v.line for v in report.violations] == [1]


def test_validate_flags_tool_calls_on_user_message(tmp_path: Path) -> None:
    """tool_calls を許容する role は assistant のみ（user にあれば違反・SHOULD-7）。"""
    record = {
        "messages": [
            {"role": "user", "content": "質問", "tool_calls": [{"id": "c", "type": "function"}]},
            {"role": "assistant", "content": "回答"},
        ]
    }
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="sft")

    assert report.ok is False


def test_validate_flags_tool_message_without_tool_call_id(tmp_path: Path) -> None:
    """role "tool" は tool_call_id キー必須（欠落は違反・SHOULD-7）。"""
    record = {
        "messages": [
            {"role": "user", "content": "質問"},
            {"role": "tool", "content": "{}"},
            {"role": "assistant", "content": "回答"},
        ]
    }
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="sft")

    assert report.ok is False


def test_validate_flags_tool_message_without_content(tmp_path: Path) -> None:
    """role "tool" は content 必須（content 無し合法は tool_calls 付き assistant に限る）。"""
    record = {
        "messages": [
            {"role": "user", "content": "質問"},
            {"role": "tool", "tool_call_id": "call_1"},
            {"role": "assistant", "content": "回答"},
        ]
    }
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="sft")

    assert report.ok is False


@pytest.mark.parametrize(
    "message",
    [
        pytest.param({"role": "user", "content": "質問", "weight": 1}, id="weight-on-user"),
        pytest.param({"role": "assistant", "content": "回答", "weight": 0.5}, id="weight-0.5"),
        pytest.param(
            {"role": "assistant", "content": "回答", "weight": 1.0}, id="weight-float-1.0"
        ),
        pytest.param({"role": "assistant", "content": "回答", "weight": True}, id="weight-bool"),
        pytest.param({"role": "assistant", "content": "回答", "weight": 2}, id="weight-int-2"),
        pytest.param(
            {"role": "assistant", "content": "回答", "weight": -1}, id="weight-int-minus1"
        ),
    ],
)
def test_validate_flags_invalid_weight(tmp_path: Path, message: dict[str, Any]) -> None:
    """weight は SFT の assistant のみ・整数 0 / 1 のみ合法（他は違反・INFO-9）。"""
    record = {"messages": [message, {"role": "assistant", "content": "末尾"}]}
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="sft")

    assert report.ok is False


@pytest.mark.parametrize("weight", [pytest.param(0, id="weight-0"), pytest.param(1, id="weight-1")])
def test_validate_accepts_legal_weight_on_assistant(tmp_path: Path, weight: int) -> None:
    """assistant の weight 0 / 1 は合法値として合格する（過大検知の退行 pin）。"""
    record = {
        "messages": [
            {"role": "user", "content": "質問"},
            {"role": "assistant", "content": "回答", "weight": weight},
        ]
    }
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="sft")

    assert report.ok is True
    assert report.violations == ()


def test_validate_does_not_check_weight_for_dpo(tmp_path: Path) -> None:
    """weight の role / 値検証は SFT のみ（DPO レコードに weight があっても合格）。"""
    record = {
        "input": {"messages": [{"role": "user", "content": "質問", "weight": 0.5}]},
        "preferred_output": [{"role": "assistant", "content": "良", "weight": 7}],
        "non_preferred_output": [{"role": "assistant", "content": "悪"}],
    }
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="dpo")

    assert report.ok is True


@pytest.mark.parametrize(
    "content",
    [pytest.param(123, id="content-int"), pytest.param([], id="content-empty-list")],
)
def test_validate_flags_invalid_content_type(tmp_path: Path, content: Any) -> None:
    """content の合法型は文字列 / 非空 parts 配列のみ（int・空リストは違反・SHOULD-8）。"""
    record = {
        "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": "回答"}]
    }
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="sft")

    assert report.ok is False


def test_validate_flags_unparsable_json_line(tmp_path: Path) -> None:
    """JSON として解析できない行は違反（行番号付き）。"""
    target = tmp_path / "data.jsonl"
    target.write_text(
        '{"messages": [{"role": "user", "content": "q"},'
        ' {"role": "assistant", "content": "a"}]}\nnot-json\n',
        encoding="utf-8",
    )
    report = validate_dataset(target, method="sft")

    assert report.ok is False
    assert [v.line for v in report.violations] == [2]


def test_validate_flags_non_dict_message_in_sft_messages() -> None:
    """messages 要素が dict でない場合は違反として報告する。"""
    record = {"messages": ["user: hi", {"role": "assistant", "content": "a"}]}
    report = validate_dataset([record], method="sft")

    assert report.ok is False
    assert any("dict でない" in violation.reason for violation in report.violations)
    assert any("messages[0]" in violation.reason for violation in report.violations)


def test_validate_flags_non_dict_message_in_dpo_preferred_output() -> None:
    """DPO の出力配列要素が dict でない場合も違反として報告する。"""
    record = {
        "input": {"messages": [{"role": "user", "content": "質問"}]},
        "preferred_output": ["p"],
        "non_preferred_output": [{"role": "assistant", "content": "悪"}],
    }
    report = validate_dataset([record], method="dpo")

    assert report.ok is False
    assert any("dict でない" in violation.reason for violation in report.violations)
    assert any("preferred_output[0]" in violation.reason for violation in report.violations)


# ----------------------------------------------------------------------
# validate_dataset: 違反ケース（DPO）
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "expected_reason_fragment"),
    [
        pytest.param(123, "JSON オブジェクトでない", id="record-not-dict"),
        pytest.param(
            {
                "input": "質問",
                "preferred_output": [{"role": "assistant", "content": "良"}],
                "non_preferred_output": [{"role": "assistant", "content": "悪"}],
            },
            "'input' が JSON オブジェクトでない",
            id="input-not-dict",
        ),
        pytest.param(
            {
                "input": {"tools": []},
                "preferred_output": [{"role": "assistant", "content": "良"}],
                "non_preferred_output": [{"role": "assistant", "content": "悪"}],
            },
            "'input.messages' が存在しない",
            id="input-messages-key-missing",
        ),
        pytest.param(
            {
                "input": {"messages": [{"role": "user", "content": "質問"}]},
                "non_preferred_output": [{"role": "assistant", "content": "悪"}],
            },
            "'preferred_output'",
            id="preferred-output-missing",
        ),
        pytest.param(
            {
                "input": {"messages": [{"role": "user", "content": "質問"}]},
                "preferred_output": [],
                "non_preferred_output": [{"role": "assistant", "content": "悪"}],
            },
            "'preferred_output'",
            id="preferred-output-empty-list",
        ),
        pytest.param(
            {
                "input": {"messages": [{"role": "user", "content": "質問"}]},
                "preferred_output": "良",
                "non_preferred_output": [{"role": "assistant", "content": "悪"}],
            },
            "'preferred_output'",
            id="preferred-output-not-list",
        ),
        pytest.param(
            {
                "input": {"messages": [{"role": "user", "content": "質問"}]},
                "preferred_output": [{"role": "assistant", "content": "良"}],
            },
            "'non_preferred_output'",
            id="non-preferred-output-missing",
        ),
        pytest.param(
            {
                "input": {"messages": [{"role": "user", "content": "質問"}]},
                "preferred_output": [{"role": "assistant", "content": "良"}],
                "non_preferred_output": [],
            },
            "'non_preferred_output'",
            id="non-preferred-output-empty-list",
        ),
    ],
)
def test_validate_flags_structural_violations_for_dpo(
    record: Any, expected_reason_fragment: str
) -> None:
    """DPO レコードの構造違反はバリデータ経路で ok=False と理由付きで報告される。"""
    report = validate_dataset([record], method="dpo")

    assert report.ok is False
    assert any(expected_reason_fragment in violation.reason for violation in report.violations), [
        violation.reason for violation in report.violations
    ]


def test_validate_flags_non_assistant_role_in_preference_output(tmp_path: Path) -> None:
    """preference 出力配列に role "user" が混入すれば違反（INFO-5）。"""
    record = {
        "input": {"messages": [{"role": "user", "content": "質問"}]},
        "preferred_output": [{"role": "user", "content": "良い応答のつもり"}],
        "non_preferred_output": [{"role": "assistant", "content": "悪い応答"}],
    }
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="dpo")

    assert report.ok is False


def test_validate_allows_multiple_assistant_messages_in_preference_output(tmp_path: Path) -> None:
    """DPO の出力配列長 1 超は違反にしない（受理可否はプラットフォーム・INFO-4）。"""
    record = {
        "input": {"messages": [{"role": "user", "content": "質問"}]},
        "preferred_output": [
            {"role": "assistant", "content": "良 1"},
            {"role": "assistant", "content": "良 2"},
        ],
        "non_preferred_output": [{"role": "assistant", "content": "悪"}],
    }
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="dpo")

    assert report.ok is True


# ----------------------------------------------------------------------
# validate_dataset: source の二形 / fail-closed / raise_on_invalid
# ----------------------------------------------------------------------


def test_validate_accepts_str_path(tmp_path: Path) -> None:
    """source はファイルパスの str も受け取れる。"""
    record = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    report = validate_dataset(str(_write_jsonl(tmp_path, [record])), method="sft")

    assert report.ok is True
    assert report.checked == 1


def test_validate_file_violation_line_is_one_based_line_number(tmp_path: Path) -> None:
    """ファイル source の `line` は 1 始まりの行番号（SHOULD-3）。"""
    ok_record = {
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    }
    bad_record = {"messages": [{"role": "user"}, {"role": "assistant", "content": "a"}]}
    report = validate_dataset(
        _write_jsonl(tmp_path, [ok_record, ok_record, bad_record]), method="sft"
    )

    assert report.checked == 3
    assert [v.line for v in report.violations] == [3]


def test_validate_accepts_dict_sequence_with_one_based_positions() -> None:
    """dict 列 source の `line` は 1 始まりの要素位置（SHOULD-3）。"""
    ok_record = {
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    }
    bad_record = {"messages": [{"role": "user"}, {"role": "assistant", "content": "a"}]}
    report = validate_dataset([ok_record, bad_record], method="sft")

    assert report.checked == 2
    assert [v.line for v in report.violations] == [2]


def test_validate_is_fail_closed() -> None:
    """違反ゼロのときのみ ok=True（違反が 1 件でもあれば不合格）。"""
    ok_record = {
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    }
    assert validate_dataset([ok_record], method="sft").ok is True
    assert validate_dataset([ok_record, {"no_messages": 1}], method="sft").ok is False


def test_validate_does_not_raise_by_default() -> None:
    """既定（raise_on_invalid=False）は例外を送出せずレポートを返すのみ。"""
    report = validate_dataset([{"no_messages": 1}], method="sft")

    assert report.ok is False
    assert len(report.violations) >= 1


def test_validate_raise_on_invalid_raises_with_report() -> None:
    """`raise_on_invalid=True` は不合格時に FineTuneError を送出し report を保持する。"""
    with pytest.raises(FineTuneError) as exc_info:
        validate_dataset([{"no_messages": 1}], method="sft", raise_on_invalid=True)

    err = exc_info.value
    assert err.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert err.report is not None
    assert err.report.ok is False
    assert len(err.report.violations) >= 1


def test_validate_raise_on_invalid_returns_report_when_valid() -> None:
    """`raise_on_invalid=True` でも合格時は例外を送出せずレポートを返す。"""
    ok_record = {
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    }
    report = validate_dataset([ok_record], method="sft", raise_on_invalid=True)

    assert report.ok is True


# ----------------------------------------------------------------------
# validate_dataset: 入力エンコーディング / 入力の頑健性
# ----------------------------------------------------------------------


def test_validate_accepts_bom_prefixed_jsonl(tmp_path: Path) -> None:
    """BOM 付き（utf-8-sig）の正しい SFT JSONL は合格する（1 行目を誤検知しない）。"""
    record = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    target = tmp_path / "bom.jsonl"
    target.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8-sig")

    report = validate_dataset(target, method="sft")

    assert report.ok is True
    assert report.checked == 1
    assert report.violations == ()


def test_validate_reports_deeply_nested_line_as_violation(tmp_path: Path) -> None:
    """深い入れ子の行は例外送出でなくレポート上の違反として報告される（fail-closed）。"""
    target = tmp_path / "deep.jsonl"
    target.write_text("[" * 200_000 + "]" * 200_000 + "\n", encoding="utf-8")

    report = validate_dataset(target, method="sft")

    assert report.ok is False
    assert [v.line for v in report.violations] == [1]


# ----------------------------------------------------------------------
# validate_dataset: method 引数の検証
# ----------------------------------------------------------------------


@pytest.mark.parametrize("method", ["DPO", "dop", "SFT", ""])
def test_validate_rejects_unknown_method(method: str) -> None:
    """未知の `method` 値は FineTuneError（VALIDATION_FAILED）で拒否する。"""
    record = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    with pytest.raises(FineTuneError) as exc_info:
        validate_dataset([record], method=method)

    err = exc_info.value
    assert err.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert repr(method) in err.message or method in err.message


@pytest.mark.parametrize("method", ["sft", "dpo"])
def test_validate_accepts_known_methods(method: str) -> None:
    """既知の `method` 値（sft / dpo）は従来どおり検証を実行する。"""
    if method == "sft":
        record: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
    else:
        record = {
            "input": {"messages": [{"role": "user", "content": "q"}]},
            "preferred_output": [{"role": "assistant", "content": "p"}],
            "non_preferred_output": [{"role": "assistant", "content": "n"}],
        }
    report = validate_dataset([record], method=method)

    assert report.ok is True
    assert report.checked == 1


def test_validate_rejects_single_dict_source() -> None:
    """`source` に単一 dict を渡すとキー文字列の列として誤読せず明示エラーにする。"""
    source = {"messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(FineTuneError) as exc_info:
        validate_dataset(source, method="sft")

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


def test_validate_default_method_is_sft() -> None:
    """`method` 省略時は既定で SFT 規則が適用される（DPO では違反になるレコード）。"""
    record = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}

    report = validate_dataset([record])

    assert report.ok is True


# ----------------------------------------------------------------------
# 非改変透過: 参照 identity の固定
# ----------------------------------------------------------------------


def test_sft_message_dicts_are_passed_by_reference() -> None:
    """SFT の messages 要素は copy されず入力 dict の参照がそのまま載る。"""
    message = {"role": "user", "content": "q", "custom": {"nested": 1}}
    result = to_sft_dataset([{"input": [message], "expected_output": "a"}])

    assert result.records[0]["messages"][0] is message


def test_dpo_input_message_dicts_are_passed_by_reference() -> None:
    """DPO の `input.messages` 要素は copy されず入力 dict の参照がそのまま載る。"""
    message = {"role": "user", "content": "q", "custom": {"nested": 1}}
    result = to_dpo_dataset(
        [
            {
                "input": [message],
                "preferred_output": "p",
                "non_preferred_output": "n",
            }
        ]
    )

    assert result.records[0]["input"]["messages"][0] is message


# ----------------------------------------------------------------------
# validate_dataset: role 合法集合の両側 pin
# ----------------------------------------------------------------------


def test_validate_flags_unknown_role(tmp_path: Path) -> None:
    """合法集合に無い role（bot）は違反になり、違反理由から role 値を特定できる。"""
    record = {
        "messages": [
            {"role": "bot", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    }
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="sft")

    assert report.ok is False
    assert any("bot" in violation.reason for violation in report.violations)


def test_validate_accepts_all_legal_roles(tmp_path: Path) -> None:
    """合法 role すべて（system / developer / user / assistant / tool）を含む記録は合格する。"""
    record = {
        "messages": [
            {"role": "system", "content": "あなたはサポート担当です。"},
            {"role": "developer", "content": "簡潔に答えること。"},
            {"role": "user", "content": "A-1234 の配送状況は?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_order_status", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "in_transit"}'},
            {"role": "assistant", "content": "配送中です。"},
        ]
    }
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="sft")

    assert report.ok is True
    assert report.violations == ()


def test_validate_accepts_explicit_null_content_with_tool_calls(tmp_path: Path) -> None:
    """`content: null` を明示した tool_calls 付き assistant は違反にならない。"""
    record = {
        "messages": [
            {"role": "user", "content": "A-1234 の配送状況は?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_order_status", "arguments": "{}"},
                    }
                ],
            },
        ]
    }
    report = validate_dataset(_write_jsonl(tmp_path, [record]), method="sft")

    assert report.ok is True
    assert report.violations == ()


# ----------------------------------------------------------------------
# validate_dataset: 空行入りファイルの line / checked
# ----------------------------------------------------------------------


def test_validate_keeps_physical_line_numbers_with_blank_lines(tmp_path: Path) -> None:
    """空行を挟んだファイルでも `line` は実ファイルの行番号で、空行は checked に数えない。"""
    ok_record = {
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    }
    bad_record = {"messages": [{"role": "user"}, {"role": "assistant", "content": "a"}]}
    target = tmp_path / "blank.jsonl"
    target.write_text(
        json.dumps(ok_record, ensure_ascii=False)
        + "\n\n"
        + json.dumps(bad_record, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    report = validate_dataset(target, method="sft")

    assert report.checked == 2
    assert [v.line for v in report.violations] == [3]


# ----------------------------------------------------------------------
# tools 写像 / skip_missing の境界 pin
# ----------------------------------------------------------------------


class _FakeToolWithNoneDescription:
    """`description` 属性が存在して値が None の FunctionTool 相当 fake。"""

    def __init__(self, name: str, params_json_schema: dict[str, Any]) -> None:
        self.name = name
        self.params_json_schema = params_json_schema
        self.description = None


def test_sft_maps_none_description_to_empty_string() -> None:
    """`description` が None の FunctionTool 相当も description "" で写像する（None 不可）。"""
    tool = _FakeToolWithNoneDescription(name="ping", params_json_schema={"type": "object"})
    result = to_sft_dataset([{"input": "質問", "expected_output": "回答"}], tools=[tool])

    assert result.records[0]["tools"][0]["function"]["description"] == ""


def test_sft_skip_missing_counts_every_skipped_case() -> None:
    """`skip_missing=True` の skipped は除外件数ぶん加算される（2 件以上でも正しい）。"""
    result = to_sft_dataset(
        [
            {"input": "q1"},
            {"input": "q2", "expected_output": "a2"},
            {"expected_output": "a3"},
        ],
        skip_missing=True,
    )

    assert result.skipped == 2
    assert len(result.records) == 1


def test_sft_empty_cases_builds_empty_result() -> None:
    """空のケース列は records 空・skipped 0 を返す。"""
    result = to_sft_dataset([])

    assert result.records == ()
    assert result.skipped == 0


def test_dpo_empty_cases_builds_empty_result() -> None:
    """空のケース列は DPO でも records 空・skipped 0 を返す。"""
    result = to_dpo_dataset([])

    assert result.records == ()
    assert result.skipped == 0


# ----------------------------------------------------------------------
# validate_dataset: hashable でない role の頑健性
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "role",
    [pytest.param(["user"], id="role-list"), pytest.param({"a": 1}, id="role-dict")],
)
def test_validate_reports_unhashable_role_as_violation(role: Any) -> None:
    """hashable でない role（リスト / dict）は例外送出でなく違反として報告される。"""
    report = validate_dataset([{"messages": [{"role": role, "content": "hi"}]}], method="sft")

    assert report.ok is False
    assert any("role" in violation.reason for violation in report.violations)


def test_validate_unhashable_role_in_dpo_output_is_violation() -> None:
    """DPO の preference 出力に hashable でない role があっても例外送出せず違反にする。"""
    record = {
        "input": {"messages": [{"role": "user", "content": "q"}]},
        "preferred_output": [{"role": ["assistant"], "content": "p"}],
        "non_preferred_output": [{"role": "assistant", "content": "n"}],
    }
    report = validate_dataset([record], method="dpo")

    assert report.ok is False
    assert any("role" in violation.reason for violation in report.violations)


# ----------------------------------------------------------------------
# validate_dataset: SFT の assistant メッセージ必須（公式 cookbook 準拠）
# ----------------------------------------------------------------------


def test_validate_flags_sft_record_without_assistant_message() -> None:
    """assistant メッセージが 1 件も無い SFT レコードは違反（公式 cookbook 準拠）。"""
    report = validate_dataset([{"messages": [{"role": "user", "content": "hi"}]}], method="sft")

    assert report.ok is False
    assert any("assistant" in violation.reason for violation in report.violations)


def test_validate_flags_sft_record_with_only_system_and_user() -> None:
    """system + user のみの SFT レコードも assistant 欠落として違反になる。"""
    record = {
        "messages": [
            {"role": "system", "content": "あなたはサポート担当です。"},
            {"role": "user", "content": "配送状況は?"},
        ]
    }
    report = validate_dataset([record], method="sft")

    assert report.ok is False
    assert any("assistant" in violation.reason for violation in report.violations)


def test_validate_accepts_sft_record_with_single_assistant_message() -> None:
    """assistant メッセージが 1 件でもあれば assistant 欠落の違反は出ない（正常系の保全）。"""
    record = {
        "messages": [
            {"role": "user", "content": "配送状況は?"},
            {"role": "assistant", "content": "配送中です。"},
        ]
    }
    report = validate_dataset([record], method="sft")

    assert report.ok is True
    assert report.violations == ()


def test_validate_does_not_flag_sft_record_whose_assistants_are_all_weight_zero() -> None:
    """全 assistant が weight 0 でも assistant 欠落にはしない（公式検証に無い項目）。"""
    record = {
        "messages": [
            {"role": "user", "content": "配送状況は?"},
            {"role": "assistant", "content": "配送中です。", "weight": 0},
        ]
    }
    report = validate_dataset([record], method="sft")

    assert report.ok is True
    assert report.violations == ()


def test_validate_reports_assistant_absence_independently_of_message_violations() -> None:
    """メッセージ単位の違反があっても assistant 欠落は短絡されず独立に報告される。"""
    record = {"messages": [{"role": "user"}, {"role": "user", "content": "追記"}]}
    report = validate_dataset([record], method="sft")

    assert report.ok is False
    reasons = [violation.reason for violation in report.violations]
    assert sum(1 for reason in reasons if "messages[0]" in reason) >= 1
    assert sum(1 for reason in reasons if "assistant" in reason) >= 1
    assert all(violation.line == 1 for violation in report.violations)
    assert "messages[0]" in reasons[0]
    assert "assistant" in reasons[-1] and "messages[" not in reasons[-1]


def test_validate_reports_assistant_absence_with_illegal_role_message() -> None:
    """不正 role の違反と assistant 欠落の違反が同一レコードで併記される。"""
    record = {"messages": [{"role": "bot", "content": "x"}]}
    report = validate_dataset([record], method="sft")

    assert report.ok is False
    reasons = [violation.reason for violation in report.violations]
    assert sum(1 for reason in reasons if "bot" in reason) >= 1
    assert sum(1 for reason in reasons if "assistant" in reason and "messages[" not in reason) >= 1


@pytest.mark.parametrize(
    ("record", "expected_reason_fragment"),
    [
        pytest.param(123, "JSON オブジェクトでない", id="record-not-dict"),
        pytest.param({"no_messages": 1}, "'messages'", id="messages-key-missing"),
        pytest.param({"messages": []}, "messages", id="messages-empty-list"),
        pytest.param({"messages": "user: hi"}, "messages", id="messages-not-list"),
    ],
)
def test_validate_does_not_add_assistant_absence_to_structural_violations(
    record: Any, expected_reason_fragment: str
) -> None:
    """messages 自体が無い / 壊れている経路では assistant 欠落を重ねて報告しない。"""
    report = validate_dataset([record], method="sft")

    assert report.ok is False
    assert len(report.violations) == 1
    assert expected_reason_fragment in report.violations[0].reason
    assert "1 件も存在しない" not in report.violations[0].reason


def test_validate_does_not_require_assistant_in_dpo_input_messages() -> None:
    """DPO の `input.messages` が user のみでも合格する（assistant 必須は SFT 限定）。"""
    record = {
        "input": {"messages": [{"role": "user", "content": "配送状況は?"}]},
        "preferred_output": [{"role": "assistant", "content": "注文番号をどうぞ。"}],
        "non_preferred_output": [{"role": "assistant", "content": "自分で調べて。"}],
    }
    report = validate_dataset([record], method="dpo")

    assert report.ok is True
    assert report.violations == ()


def test_validate_accepts_records_built_by_to_sft_dataset() -> None:
    """`to_sft_dataset` の出力は assistant 必須の新規則でも合格する（変換経路の回帰 pin）。"""
    result = to_sft_dataset(
        [
            {"input": "配送状況は?", "expected_output": "配送中です。"},
            {
                "input": [{"role": "user", "content": "在庫は?"}],
                "expected_output": [{"role": "assistant", "content": "あります。"}],
            },
        ],
        system="あなたはサポート担当です。",
    )
    report = validate_dataset(list(result.records), method="sft")

    assert report.ok is True
    assert report.checked == 2


# ----------------------------------------------------------------------
# 変換ヘルパ: エラーメッセージの位置情報
# ----------------------------------------------------------------------


def test_sft_case_error_message_reports_one_based_case_index() -> None:
    """ケース位置は 1 始まりで載る（2 件目の不備なら「ケース 2」）。"""
    with pytest.raises(FineTuneError) as exc_info:
        to_sft_dataset([{"input": "質問", "expected_output": "回答"}, {"input": "質問のみ"}])

    assert "ケース 2" in exc_info.value.message


def test_dpo_case_error_message_reports_one_based_case_index() -> None:
    """DPO も同様にケース位置を 1 始まりで載せる。"""
    with pytest.raises(FineTuneError) as exc_info:
        to_dpo_dataset(
            [
                {"input": "質問", "preferred_output": "良", "non_preferred_output": "悪"},
                {"input": "質問", "preferred_output": "良"},
            ]
        )

    assert "ケース 2" in exc_info.value.message


def test_tools_error_message_reports_zero_based_element_position() -> None:
    """`tools=` の不正要素は 0 始まりの位置表記で載る（2 要素目なら tools[1]）。"""
    with pytest.raises(FineTuneError) as exc_info:
        to_sft_dataset([{"input": "質問", "expected_output": "回答"}], tools=[{}, 42])

    assert "tools[1]" in exc_info.value.message


# ----------------------------------------------------------------------
# 変換ヘルパ: 単一 dict を直接渡す誤用のガード
# ----------------------------------------------------------------------


@pytest.mark.parametrize("skip_missing", [False, True])
def test_sft_rejects_single_dict_cases(skip_missing: bool) -> None:
    """`to_sft_dataset` に単一 dict を渡すと `skip_missing` によらず FineTuneError。"""
    with pytest.raises(FineTuneError) as exc_info:
        to_sft_dataset({"input": "x", "expected_output": "y"}, skip_missing=skip_missing)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


@pytest.mark.parametrize("skip_missing", [False, True])
def test_dpo_rejects_single_dict_cases(skip_missing: bool) -> None:
    """`to_dpo_dataset` に単一 dict を渡すと `skip_missing` によらず FineTuneError。"""
    with pytest.raises(FineTuneError) as exc_info:
        to_dpo_dataset(
            {"input": "x", "preferred_output": "p", "non_preferred_output": "n"},
            skip_missing=skip_missing,
        )

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


# ----------------------------------------------------------------------
# 変換ヘルパ: 文字列 / bytes を直接渡す誤用のガード
# ----------------------------------------------------------------------


@pytest.mark.parametrize("skip_missing", [False, True])
@pytest.mark.parametrize(
    "cases", [pytest.param("abcd", id="str"), pytest.param(b"abcd", id="bytes")]
)
def test_sft_rejects_str_or_bytes_cases(cases: Any, skip_missing: bool) -> None:
    """`to_sft_dataset` に str / bytes を渡すと 1 文字 1 ケースと誤読せず FineTuneError。"""
    with pytest.raises(FineTuneError) as exc_info:
        to_sft_dataset(cases, skip_missing=skip_missing)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


@pytest.mark.parametrize("skip_missing", [False, True])
@pytest.mark.parametrize(
    "cases", [pytest.param("abcd", id="str"), pytest.param(b"abcd", id="bytes")]
)
def test_dpo_rejects_str_or_bytes_cases(cases: Any, skip_missing: bool) -> None:
    """`to_dpo_dataset` に str / bytes を渡しても `skip_missing` によらず FineTuneError。"""
    with pytest.raises(FineTuneError) as exc_info:
        to_dpo_dataset(cases, skip_missing=skip_missing)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


# ----------------------------------------------------------------------
# validate_dataset: 逐次読み込み化に対する回帰 pin
# ----------------------------------------------------------------------


def test_validate_keeps_input_order_of_violations() -> None:
    """複数レコード・複数違反でも violations は入力順（レコード順・メッセージ順）で並ぶ。"""
    first = {
        "messages": [
            {"role": "user"},
            {"role": "bot", "content": "x"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    second = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant"}]}
    report = validate_dataset([first, second], method="sft")

    assert report.checked == 2
    assert [violation.line for violation in report.violations] == [1, 1, 2]
    assert "messages[0]" in report.violations[0].reason
    assert "messages[1]" in report.violations[1].reason


def test_validate_raise_on_invalid_report_matches_non_raising_report() -> None:
    """`raise_on_invalid=True` の `report` は非送出時のレポートと同じ内容を持つ。"""
    records = [
        {"messages": [{"role": "user"}, {"role": "assistant", "content": "a"}]},
        {"no_messages": 1},
    ]
    expected = validate_dataset(records, method="sft")
    with pytest.raises(FineTuneError) as exc_info:
        validate_dataset(records, method="sft", raise_on_invalid=True)

    actual = exc_info.value.report
    assert actual is not None
    assert actual.ok == expected.ok
    assert actual.checked == expected.checked
    assert actual.violations == expected.violations
