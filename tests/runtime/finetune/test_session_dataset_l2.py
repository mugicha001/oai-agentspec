"""L2: `dataset_from_session`（FR-4）の生成規則・読み取り専用契約・エラー方針を固定する。

fake Session（`_helpers.fake_session.FakeSession`・呼び出しメソッド記録付き）を通して
「`dataset_from_session` -> `_adapters.finetune.fetch_session_items` -> Session」の鎖を測る。

固定する契約（設計方針 /tmp/architecture/55_policy.md）:
    - 正規化: role が user / assistant の item のみ採用。role 無し item（function_call 等）と
      system / developer item は破棄（skipped に数えない）。parts 配列 content は str へ吸収
    - 累積ペアリング: 各 assistant ターンを expected_output とし、それ以前の全ターンを input
      とするケースを 1 件生成。input が空になるケース（先頭 assistant）は skipped へ計上
    - filter -> transform の順で適用。filter 除外は skipped へ計上。filter 全滅は
      `DatasetBuildResult(records=(), skipped=全件)` の正常返却（エラーにしない）
    - 読み取り専用: Session へは `get_items` のみ（書込系メソッドを一切呼ばない）
    - エラー: session=None は CONFIG_MISSING、空履歴 / 抽出可能ターンなし / assistant なし /
      transform の不正戻り値は VALIDATION_FAILED

Session Protocol への duck typing 接触を含むため層は L2（`@pytest.mark.integration`）。
ネットワーク非接触は conftest の autouse ガードが担保する。
"""

from __future__ import annotations

from typing import Any

import pytest

from oai_agentspec.runtime.finetune import (
    FineTuneError,
    FineTuneFailureKind,
    dataset_from_session,
)

from _helpers.fake_session import FakeSession

pytestmark = pytest.mark.integration

# 設計方針のデータサンプル (a) と同一構成の履歴 items（Responses API 形式・plain dict）。
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

# 上記から生成される最終レコード列（parts は str へ吸収済み・function_call 系は不在）。
_SAMPLE_RECORDS = (
    {
        "messages": [
            {"role": "user", "content": "会員登録の手順を教えて"},
            {"role": "assistant", "content": "手順は次の通りです: ..."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "会員登録の手順を教えて"},
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
    レコードへ現れない。`records` は全体を `==` で照合する（部分照合にしない）。
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
    """
    session = FakeSession(
        [
            "壊れた文字列 item",  # type: ignore[list-item]
            {"role": "user", "content": None},
            {"role": "assistant", "content": "承知しました"},
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
    """role なし item と system item のみの履歴は VALIDATION_FAILED で失敗する。"""
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
        masked_input = [
            {**message, "content": str(message["content"]).replace("会員登録", "[MASKED]")}
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
