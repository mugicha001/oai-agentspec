"""L1: ツール往復のスクリーニング（`screen_tool_roundtrips`）を検証する。

`screen_tool_roundtrips` は submit 前の明示ゲートとして「メッセージ間の順序制約」だけを判定する
（メッセージ単位の合法性は `validate_dataset` の責務）。規則 (1) `tool_calls` を持つ
assistant の直後に続く連続した role `"tool"` 群の `tool_call_id` 集合が当該 assistant の
`tool_calls` の id 集合と一致すること（過不足なし・群内の順序は問わない）、規則 (2) いずれの
群にも属さない role `"tool"` が存在しないこと、を固定する。あわせて `validate_dataset` と
同型の契約（source の二形と `line` の意味・`method` の `"sft"` / `"dpo"` 切り替え・
`raise_on_invalid`・単一 dict の明示エラー）と、構造違反を二重報告しないことを網羅する。
すべて純データ操作で外部依存なし（`@pytest.mark.unit`）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from oai_agentspec.runtime.finetune import (
    FineTuneError,
    FineTuneFailureKind,
    screen_tool_roundtrips,
)

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# テスト用ヘルパ（メッセージ組み立て / JSONL 書き出し）
# ----------------------------------------------------------------------


def _assistant_calls(*call_ids: str) -> dict[str, Any]:
    """指定 id の `tool_calls` を持つ assistant メッセージを組む。"""
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "get_order_status", "arguments": "{}"},
            }
            for call_id in call_ids
        ],
    }


def _tool(call_id: str) -> dict[str, Any]:
    """指定 id へ応答する role `"tool"` メッセージを組む。"""
    return {"role": "tool", "tool_call_id": call_id, "content": '{"status": "in_transit"}'}


def _sft(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """messages を SFT レコードへ包む。"""
    return {"messages": messages}


def _dpo(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """messages を DPO（preference）レコードの `input.messages` へ包む。"""
    return {
        "input": {"messages": messages},
        "preferred_output": [{"role": "assistant", "content": "配送中です。"}],
        "non_preferred_output": [{"role": "assistant", "content": "知りません。"}],
    }


def _write_jsonl(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    """レコード列を JSONL ファイルへ書き出してパスを返す（`test_dataset_l1.py` と同形）。"""
    target = tmp_path / "data.jsonl"
    target.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return target


# ----------------------------------------------------------------------
# 合格ケース（誤検知を出さない・過大側の変異を kill する）
# ----------------------------------------------------------------------


def test_screen_accepts_sequential_tool_roundtrip() -> None:
    """N1: 正常な逐次呼び出し（呼び出しごとに直後で応答）は合格する。"""
    record = _sft(
        [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            _assistant_calls("call_a"),
            _tool("call_a"),
            _assistant_calls("call_b"),
            _tool("call_b"),
            {"role": "assistant", "content": "配送中です。"},
        ]
    )
    report = screen_tool_roundtrips([record], method="sft")

    assert report.ok is True
    assert report.checked == 1
    assert report.violations == ()


def test_screen_accepts_parallel_tool_outputs_in_any_order() -> None:
    """N2: 並列呼び出しの出力順が呼び出し順と逆でも合格する（群内の順序は問わない）。"""
    record = _sft(
        [
            {"role": "user", "content": "2 件まとめて調べて"},
            _assistant_calls("call_a", "call_b"),
            _tool("call_b"),
            _tool("call_a"),
            {"role": "assistant", "content": "どちらも配送中です。"},
        ]
    )
    report = screen_tool_roundtrips([record], method="sft")

    assert report.ok is True
    assert report.violations == ()


def test_screen_accepts_record_without_tool_messages() -> None:
    """N10: ツールを含まないレコードは合格する（過大側の誤検知を出さない）。"""
    record = _sft(
        [
            {"role": "user", "content": "こんにちは"},
            {"role": "assistant", "content": "こんにちは。"},
        ]
    )
    report = screen_tool_roundtrips([record], method="sft")

    assert report.ok is True
    assert report.checked == 1
    assert report.violations == ()


# ----------------------------------------------------------------------
# 違反ケース（規則 (1): 群の一致 / 規則 (2): 先行群なしの tool）
# ----------------------------------------------------------------------


def test_screen_flags_tool_output_separated_by_assistant_text() -> None:
    """N3: 呼び出しと応答の間に assistant テキストが挟まる非隣接は違反（規則 (1)）。"""
    record = _sft(
        [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            _assistant_calls("call_c1"),
            {"role": "assistant", "content": "確認します。少々お待ちください。"},
            _tool("call_c1"),
            {"role": "assistant", "content": "配送中です。"},
        ]
    )
    report = screen_tool_roundtrips([record], method="sft")

    assert report.ok is False
    assert report.checked == 1
    assert [violation.line for violation in report.violations] == [1, 1]
    reasons = " / ".join(violation.reason for violation in report.violations)
    assert "call_c1" in reasons
    assert "tool" in reasons


def test_screen_flags_partially_answered_tool_call_group() -> None:
    """N4: 群の一部だけ応答（要求 2 件・直後の群 1 件）は違反 1 件（規則 (1)）。"""
    record = _sft(
        [
            {"role": "user", "content": "2 件まとめて調べて"},
            _assistant_calls("call_a", "call_b"),
            _tool("call_a"),
            {"role": "assistant", "content": "片方だけ分かりました。"},
        ]
    )
    report = screen_tool_roundtrips([record], method="sft")

    assert report.ok is False
    assert len(report.violations) == 1
    assert report.violations[0].line == 1
    assert "call_b" in report.violations[0].reason


def test_screen_flags_unrequested_tool_id_inside_group() -> None:
    """N11: 群内に未要求の `tool_call_id` が混じる場合は違反（規則 (1) を一致で判定する pin）。

    規則 (1) を「要求 id ⊆ 群」の部分集合判定へ弱める等価変異は、余剰側を見逃すため本 pin
    でのみ RED になる（N5 は規則 (2) 単独で違反が立つため弱めても緑のまま通る）。
    """
    record = _sft(
        [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            _assistant_calls("call_a"),
            _tool("call_a"),
            _tool("call_b"),
            {"role": "assistant", "content": "配送中です。"},
        ]
    )
    report = screen_tool_roundtrips([record], method="sft")

    assert report.ok is False
    assert len(report.violations) == 1
    assert "call_b" in report.violations[0].reason


def test_screen_flags_tool_message_without_preceding_tool_calls() -> None:
    """N5: いずれの群にも属さない tool は違反（規則 (2) 単独・群の一致は満たす形）。"""
    record = _sft(
        [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            _tool("call_c1"),
            _assistant_calls("call_c1"),
            _tool("call_c1"),
            {"role": "assistant", "content": "配送中です。"},
        ]
    )
    report = screen_tool_roundtrips([record], method="sft")

    assert report.ok is False
    assert len(report.violations) == 1
    assert report.violations[0].line == 1
    assert "tool" in report.violations[0].reason


# ----------------------------------------------------------------------
# method の切り替え / source の二形 / raise_on_invalid / 二重報告しない
# ----------------------------------------------------------------------


def test_screen_accepts_trailing_tool_calls_as_learning_target() -> None:
    """N20: 末尾の `tool_calls` 付き assistant は違反にしない（学習ターゲット本体）。

    「ツール呼び出しそのものを学習させる」SFT レコードでは、`expected_output` が
    `tool_calls` 付き assistant になり応答が続かないのが正常である。末尾の群へ規則 (1) を
    適用すると、lib 自身の `to_sft_dataset` が生成した正当なレコードを不合格にし、
    `partition_dataset` 経由では silent に全件脱落する。
    """
    record = _sft(
        [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            _assistant_calls("call_a"),
        ]
    )
    report = screen_tool_roundtrips([record], method="sft")

    assert report.ok is True
    assert report.violations == ()


def test_screen_accepts_trailing_tool_calls_after_closed_roundtrip() -> None:
    """N21: 閉じた往復の後に続く末尾の呼び出しも違反にしない（文脈と末尾の共存）。"""
    record = _sft(
        [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            _assistant_calls("call_a"),
            _tool("call_a"),
            _assistant_calls("call_b"),
        ]
    )
    report = screen_tool_roundtrips([record], method="sft")

    assert report.ok is True
    assert report.violations == ()


def test_screen_still_flags_unanswered_group_in_the_middle() -> None:
    """N22: 末尾でない群の未応答は従来どおり違反（末尾例外を全群へ広げる過大側を検知）。"""
    record = _sft(
        [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            _assistant_calls("call_a"),
            {"role": "assistant", "content": "確認します。"},
            _tool("call_a"),
            {"role": "assistant", "content": "配送中です。"},
        ]
    )
    report = screen_tool_roundtrips([record], method="sft")

    assert report.ok is False
    assert any("call_a" in violation.reason for violation in report.violations)


def test_screen_flags_non_list_tool_calls() -> None:
    """N23: `tool_calls` が非リストなら違反（往復の対応を検証できない）。

    `validate_dataset` はキーの存在しか見ないため（FR-3 は内部構造を解釈しない）、
    screening が素通しすると両ゲートを通過して `passed` へ混ざる。
    """
    for bad in ({}, "call_1", 123):
        record = _sft(
            [
                {"role": "user", "content": "A-1234 の配送状況を教えて"},
                {"role": "assistant", "tool_calls": bad},
                {"role": "assistant", "content": "配送中です。"},
            ]
        )
        report = screen_tool_roundtrips([record], method="sft")

        assert report.ok is False, f"tool_calls={bad!r} が素通しされた"
        assert report.violations[0].reason.startswith("messages[1]:")


def test_screen_flags_tool_call_without_string_id() -> None:
    """N15: `tool_calls` の要素が str の `id` を欠くと違反（往復の対応を検証できないため）。

    id が無い呼び出しを要求集合から黙って落とすと、後続 tool が無い構成では「要求 0 件 /
    応答 0 件」で一致してしまい合格する。`validate_dataset` は `tool_calls` の内部構造を
    解釈しない（FR-3）ため、どのゲートも捕らえない fail-open になる。
    """
    record = _sft(
        [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_order_status", "arguments": "{}"},
                    }
                ],
            },
            {"role": "assistant", "content": "配送中です。"},
        ]
    )
    report = screen_tool_roundtrips([record], method="sft")

    assert report.ok is False
    assert len(report.violations) == 1
    assert report.violations[0].line == 1


def test_screen_flags_non_string_tool_call_id() -> None:
    """N16: `tool_calls[].id` が非 str（数値等）でも違反として報告する。"""
    record = _sft(
        [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": 123,
                        "type": "function",
                        "function": {"name": "get_order_status", "arguments": "{}"},
                    }
                ],
            },
            {"role": "assistant", "content": "配送中です。"},
        ]
    )
    report = screen_tool_roundtrips([record], method="sft")

    assert report.ok is False
    assert len(report.violations) == 1


def test_screen_flags_id_less_tool_call_mixed_with_answered_call() -> None:
    """N17: 正常な呼び出しと id 欠落が混在する群でも、欠落側が違反として現れる。

    正常側の id が応答されて集合が一致するため、id 欠落を落とす実装では合格してしまう。
    """
    record = _sft(
        [
            {"role": "user", "content": "2 件まとめて調べて"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "get_order_status", "arguments": "{}"},
                    },
                    {
                        "type": "function",
                        "function": {"name": "get_order_status", "arguments": "{}"},
                    },
                ],
            },
            _tool("call_a"),
            {"role": "assistant", "content": "配送中です。"},
        ]
    )
    report = screen_tool_roundtrips([record], method="sft")

    assert report.ok is False
    assert len(report.violations) == 1


def test_screen_reason_carries_message_position() -> None:
    """N18: 違反理由は `messages[N]:` 形式で位置を示す（`validate_dataset` と同書式）。

    1 レコード内の離れた 2 箇所が別々に違反するとき、どのメッセージが原因かを call id 頼りに
    探さずに特定できる。位置の前置を落とす変異が RED になる。
    """
    record = _sft(
        [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            _assistant_calls("call_a"),
            {"role": "assistant", "content": "確認します。"},
            _tool("call_a"),
            {"role": "user", "content": "もう 1 件"},
            _assistant_calls("call_b"),
            {"role": "assistant", "content": "終わりです。"},
        ]
    )
    report = screen_tool_roundtrips([record], method="sft")

    reasons = [violation.reason for violation in report.violations]
    assert [reason.split(":")[0] for reason in reasons] == [
        "messages[1]",
        "messages[3]",
        "messages[5]",
    ]


def test_screen_dpo_reason_uses_input_messages_label() -> None:
    """N19: DPO では位置表記が `input.messages[N]:` になる（走査範囲と一致した表記）。"""
    record = _dpo(
        [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            _assistant_calls("call_c1"),
            {"role": "assistant", "content": "確認します。"},
            _tool("call_c1"),
        ]
    )
    report = screen_tool_roundtrips([record], method="dpo")

    assert report.violations[0].reason.startswith("input.messages[1]:")


def test_screen_dpo_reads_input_messages() -> None:
    """N6: `method="dpo"` は `input.messages` を見る（SFT の `messages` は見ない）。"""
    bad = _dpo(
        [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            _assistant_calls("call_c1"),
            {"role": "assistant", "content": "確認します。"},
            _tool("call_c1"),
        ]
    )
    good = _dpo(
        [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            _assistant_calls("call_c1"),
            _tool("call_c1"),
        ]
    )

    assert screen_tool_roundtrips([good], method="dpo").ok is True

    report = screen_tool_roundtrips([bad], method="dpo")
    assert report.ok is False
    assert report.checked == 1
    assert "call_c1" in " / ".join(violation.reason for violation in report.violations)


def test_screen_file_source_line_is_one_based_line_number(tmp_path: Path) -> None:
    """N7: ファイル source でも判定は同一で、`line` は 1 始まりの物理行番号になる。"""
    ok_record = _sft(
        [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            _assistant_calls("call_a"),
            _tool("call_a"),
            {"role": "assistant", "content": "配送中です。"},
        ]
    )
    bad_record = _sft(
        [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            _assistant_calls("call_a"),
            {"role": "assistant", "content": "確認します。"},
            _tool("call_a"),
        ]
    )
    report = screen_tool_roundtrips(
        _write_jsonl(tmp_path, [ok_record, ok_record, bad_record]), method="sft"
    )

    assert report.ok is False
    assert report.checked == 3
    assert {violation.line for violation in report.violations} == {3}


def test_screen_raise_on_invalid_raises_with_report() -> None:
    """N8: `raise_on_invalid=True` は不合格時に FineTuneError を送出し report を保持する。"""
    record = _sft(
        [
            {"role": "user", "content": "A-1234 の配送状況を教えて"},
            _assistant_calls("call_c1"),
            {"role": "assistant", "content": "確認します。"},
            _tool("call_c1"),
        ]
    )
    with pytest.raises(FineTuneError) as exc_info:
        screen_tool_roundtrips([record], method="sft", raise_on_invalid=True)

    err = exc_info.value
    assert err.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert err.report is not None
    assert err.report.ok is False
    assert len(err.report.violations) >= 1


def test_screen_does_not_report_structural_violations() -> None:
    """N9: 構造違反（messages 非リスト・要素が非 dict）は screening では報告しない。"""
    report = screen_tool_roundtrips(
        [
            {"messages": "not-a-list"},
            {"messages": ["not-a-dict", {"role": "assistant", "content": "a"}]},
            {"no_messages": 1},
        ],
        method="sft",
    )

    assert report.ok is True
    assert report.checked == 3
    assert report.violations == ()


def test_screen_rejects_single_dict_source() -> None:
    """`validate_dataset` と同型に、単一 dict の source は明示エラーにする。"""
    with pytest.raises(FineTuneError) as exc_info:
        screen_tool_roundtrips(_sft([{"role": "user", "content": "q"}]), method="sft")

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


def test_screen_rejects_unknown_method() -> None:
    """`validate_dataset` と同型に、`method` は `"sft"` / `"dpo"` のみ受け付ける。"""
    with pytest.raises(FineTuneError) as exc_info:
        screen_tool_roundtrips([], method="preference")

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
