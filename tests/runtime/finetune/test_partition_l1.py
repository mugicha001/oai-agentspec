"""L1: 投入前の仕分け（`partition_dataset`）を検証する。

`partition_dataset` は `validate_dataset`（メッセージ単位の合法性）と `screen_tool_roundtrips`
（メッセージ間の順序制約）の両ゲートを各レコードへ適用し、どちらにも違反しないレコードだけを
`passed` へ、違反したレコードを理由つきで `rejected` へ仕分ける。`passed` は
`DatasetBuildResult` として返し、`submit_job(train=...)` / `.save()` の既存の受け口へ
そのまま渡せることを固定する。すべて純データ操作で外部依存なし（`@pytest.mark.unit`）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from oai_agentspec.runtime.finetune import (
    DatasetBuildResult,
    FineTuneError,
    FineTuneFailureKind,
    partition_dataset,
    to_sft_dataset,
)

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# テスト用ヘルパ
# ----------------------------------------------------------------------


def _assistant_calls(*call_ids: str) -> dict[str, Any]:
    """指定 id の `tool_calls` を持つ assistant メッセージを組む。"""
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "check_stock", "arguments": "{}"},
            }
            for call_id in call_ids
        ],
    }


def _tool(call_id: str) -> dict[str, Any]:
    """指定 id へ応答する role `"tool"` メッセージを組む。"""
    return {"role": "tool", "tool_call_id": call_id, "content": '{"count": 3}'}


# 合格レコード（メッセージ単位も並びも正しい）。
_GOOD: dict[str, Any] = {
    "messages": [
        {"role": "user", "content": "A-100 の在庫を教えて"},
        _assistant_calls("call_a"),
        _tool("call_a"),
        {"role": "assistant", "content": "在庫は 3 個です。"},
    ]
}

# 並びだけが不正（screening は違反・validate は合格）。
_BAD_ORDER: dict[str, Any] = {
    "messages": [
        {"role": "user", "content": "A-100 の在庫を教えて"},
        _assistant_calls("call_b"),
        {"role": "assistant", "content": "確認します。"},
        _tool("call_b"),
        {"role": "assistant", "content": "在庫は 3 個です。"},
    ]
}

# 構造だけが不正（validate は違反・screening は素通し）。
_BAD_STRUCTURE: dict[str, Any] = {
    "messages": [
        {"role": "customer", "content": "役割が不正"},
        {"role": "assistant", "content": "応答"},
    ]
}


def _write_jsonl(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    """レコード列を JSONL ファイルへ書き出してパスを返す。"""
    target = tmp_path / "data.jsonl"
    target.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return target


# ----------------------------------------------------------------------
# 仕分けの基本契約
# ----------------------------------------------------------------------


def test_partition_splits_passed_and_rejected() -> None:
    """P1: 合格レコードは passed へ、違反レコードは rejected へ仕分けられる。"""
    part = partition_dataset([_GOOD, _BAD_ORDER, _GOOD], method="sft")

    assert part.checked == 3
    assert part.ok is False
    assert part.passed.records == (_GOOD, _GOOD)
    assert len(part.rejected) == 1
    assert part.rejected[0].line == 2


def test_partition_passed_is_dataset_build_result_ready_for_submit() -> None:
    """P2: `passed` は `DatasetBuildResult` で、`skipped` に不合格件数が載る。

    `submit_job(train=...)` / `.save()` の既存の受け口へ詰め替えなしで渡せることを固定する
    （plain な tuple / list を返す変異が RED になる）。
    """
    part = partition_dataset([_GOOD, _BAD_ORDER], method="sft")

    assert isinstance(part.passed, DatasetBuildResult)
    assert part.passed.skipped == 1
    assert part.passed.records == (_GOOD,)


def test_partition_passed_records_are_the_original_objects() -> None:
    """P3: `passed` のレコードは入力そのもの（複製せず参照を載せる・非改変透過）。"""
    part = partition_dataset([_GOOD], method="sft")

    assert part.passed.records[0] is _GOOD


def test_partition_rejected_carries_record_and_reasons() -> None:
    """P4: 不合格側は元レコードと理由を抱えて返る（report との突き合わせを要さない）。"""
    part = partition_dataset([_BAD_ORDER], method="sft")

    rejected = part.rejected[0]
    assert rejected.record is _BAD_ORDER
    assert rejected.line == 1
    assert len(rejected.reasons) >= 1
    assert "call_b" in " / ".join(rejected.reasons)


def test_partition_all_passed_reports_ok() -> None:
    """P5: 違反ゼロなら `ok=True` で rejected は空（過大側の誤検知を出さない）。"""
    part = partition_dataset([_GOOD, _GOOD], method="sft")

    assert part.ok is True
    assert part.rejected == ()
    assert part.passed.skipped == 0


# ----------------------------------------------------------------------
# 両ゲートの合成（片方だけでは仕分けられないことの pin）
# ----------------------------------------------------------------------


def test_partition_rejects_structural_violation_detected_only_by_validate() -> None:
    """P6: 構造違反（screening は素通しする）も rejected へ入る（validate 側の合成）。

    screening だけで仕分ける変異が RED になる。
    """
    part = partition_dataset([_BAD_STRUCTURE], method="sft")

    assert part.ok is False
    assert part.passed.records == ()
    assert len(part.rejected) == 1


def test_partition_rejects_order_violation_detected_only_by_screening() -> None:
    """P7: 並びの違反（validate は合格にする）も rejected へ入る（screening 側の合成）。

    validate だけで仕分ける変異が RED になる。
    """
    part = partition_dataset([_BAD_ORDER], method="sft")

    assert part.ok is False
    assert part.passed.records == ()
    assert len(part.rejected) == 1


def test_partition_merges_reasons_from_both_gates() -> None:
    """P8: 両ゲートに違反するレコードは、両方の理由が 1 件の rejected に集まる。"""
    both_bad: dict[str, Any] = {
        "messages": [
            {"role": "customer", "content": "役割が不正"},
            _assistant_calls("call_c"),
            {"role": "assistant", "content": "確認します。"},
            _tool("call_c"),
        ]
    }
    part = partition_dataset([both_bad], method="sft")

    assert len(part.rejected) == 1
    reasons = " / ".join(part.rejected[0].reasons)
    assert "customer" in reasons
    assert "call_c" in reasons


# ----------------------------------------------------------------------
# source の二形 / method / 入口ガード
# ----------------------------------------------------------------------


def test_partition_file_source_line_is_physical_line_number(tmp_path: Path) -> None:
    """P9: ファイル source でも仕分けは同一で、`line` は 1 始まりの物理行番号になる。"""
    part = partition_dataset(_write_jsonl(tmp_path, [_GOOD, _GOOD, _BAD_ORDER]), method="sft")

    assert part.checked == 3
    assert len(part.passed.records) == 2
    assert [rejected.line for rejected in part.rejected] == [3]


def test_partition_dpo_reads_input_messages() -> None:
    """P10: `method="dpo"` は DPO レコードとして両ゲートを適用する。"""
    good_dpo: dict[str, Any] = {
        "input": {"messages": [{"role": "user", "content": "q"}]},
        "preferred_output": [{"role": "assistant", "content": "よい"}],
        "non_preferred_output": [{"role": "assistant", "content": "わるい"}],
    }
    bad_dpo: dict[str, Any] = {
        "input": {
            "messages": [
                {"role": "user", "content": "q"},
                _assistant_calls("call_d"),
                {"role": "assistant", "content": "確認します。"},
                _tool("call_d"),
            ]
        },
        "preferred_output": [{"role": "assistant", "content": "よい"}],
        "non_preferred_output": [{"role": "assistant", "content": "わるい"}],
    }
    part = partition_dataset([good_dpo, bad_dpo], method="dpo")

    assert part.passed.records == (good_dpo,)
    assert [rejected.line for rejected in part.rejected] == [2]


def test_partition_passes_tool_call_learning_target_from_to_sft_dataset() -> None:
    """P16: lib 自身の `to_sft_dataset` が作るツール呼び出し学習レコードは合格する。

    末尾の `tool_calls` を未応答として弾くと、ツール呼び出しを学習させるデータセットが
    `partition_dataset` で silent に全件脱落し、利用者は気づかないまま投入する
    （`partition_dataset` は例外を投げない）。lib の生成物が lib のゲートを通ることを固定する。
    """
    built = to_sft_dataset(
        [
            {
                "input": "A-1234 の配送状況を教えて",
                "expected_output": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_order_status", "arguments": "{}"},
                            }
                        ],
                    }
                ],
            }
        ]
    )
    part = partition_dataset(built.records, method="sft")

    assert part.ok is True
    assert len(part.passed.records) == 1


def test_partition_rejects_non_list_tool_calls() -> None:
    """P17: 非リストの `tool_calls` を持つレコードは合格側へ入らない（両ゲートの穴）。"""
    record: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": {}},
        ]
    }
    part = partition_dataset([record], method="sft")

    assert part.ok is False
    assert part.passed.records == ()


def test_partition_rejects_unknown_method() -> None:
    """P11: `method` は `"sft"` / `"dpo"` のみ受け付ける（両ゲートと同型の入口ガード）。"""
    with pytest.raises(FineTuneError) as exc_info:
        partition_dataset([], method="preference")

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


def test_partition_rejects_single_dict_source() -> None:
    """P12: 単一 dict の source は明示エラーにする（キー文字列の列として誤読しない）。"""
    with pytest.raises(FineTuneError) as exc_info:
        partition_dataset(_GOOD, method="sft")

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


def test_partition_treats_unparsable_line_as_rejected(tmp_path: Path) -> None:
    """P13: JSON として解析できない行は rejected 扱いにする（合格側へ混ぜない）。"""
    target = tmp_path / "broken.jsonl"
    target.write_text(
        json.dumps(_GOOD, ensure_ascii=False) + "\n{壊れた行\n",
        encoding="utf-8",
    )
    part = partition_dataset(target, method="sft")

    assert part.checked == 2
    assert len(part.passed.records) == 1
    assert [rejected.line for rejected in part.rejected] == [2]


def test_partition_keeps_raw_text_of_unparsable_line(tmp_path: Path) -> None:
    """P14: 解析できない行は原文を `raw` に保持する（元ファイルを開き直させない）。

    `record` は組み立てられないため None になるが、壊れた行こそ中身を見たいケースであり、
    原文を落とすと `DatasetRejection` の存在理由（元データを引き直さずに直せる）が
    その行だけ失われる。原文を捨てる変異が RED になる。
    """
    target = tmp_path / "broken.jsonl"
    target.write_text(
        json.dumps(_GOOD, ensure_ascii=False) + '\n{"messages": [壊れた\n',
        encoding="utf-8",
    )
    part = partition_dataset(target, method="sft")

    rejected = part.rejected[0]
    assert rejected.record is None
    assert rejected.raw == '{"messages": [壊れた'


def test_partition_does_not_keep_raw_for_parsed_records() -> None:
    """P15: 解析できたレコードの `raw` は None（原文を二重に抱えない）。

    `record` があれば原文は不要であり、両方を保持すると全量保持のメモリ特性がさらに悪化する。
    """
    part = partition_dataset([_BAD_ORDER], method="sft")

    assert part.rejected[0].record is _BAD_ORDER
    assert part.rejected[0].raw is None
