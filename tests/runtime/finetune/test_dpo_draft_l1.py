"""L1: `save_dpo_draft` / `finalize_dpo_draft`（FR-12）の記入ワークフロー契約を固定する。

雛形モードの記入用ケース列を CSV / JSONL へ書き出し、人手記入後に最終 preference
データセットへ取り込む往復を測る。純データ + ローカルファイル I/O のみで Session・
ネットワークには触れない（層は L1）。

固定する契約（要件書 FR-12 / ADR 0034 Decision 6-8・10）:
    - `save_dpo_draft`: 拡張子で `.csv` / `.jsonl` を切替（他は CONFIG_MISSING）。CSV 列は
      `case_index` / `context` / `response` / `preferred_output` / `non_preferred_output` /
      `input_json` の 6 列で 1 ファイル自己完結。必須キーを欠く source は全件検証して
      エラーにし、部分的に書かれたファイルを残さない
    - `finalize_dpo_draft`: CSV パス / JSONL パス / ケース列（iterable）の 3 形を受理し
      `DatasetBuildResult` を返す。CSV 読み取りは列名ベース（列順非依存）で参照列は無視
    - 記入判定: strip 後に空の 2 欄は skip（skipped 計上）、片欄のみはケース位置つき
      VALIDATION_FAILED、全件未記入は `records=()` の正常返却。採用値は非改変（strip しない）
    - エンコーディング: 書き込み・読み取りとも `utf-8-sig`（BOM なしファイルも取り込める）
    - `skipped` は finalize 分 + 委譲先分（雛形生成時のカウントは含めない）
    - tools 透過: `finalize_dpo_draft` の `tools=` / `parallel_tool_calls=` は写像も検証も
      せず `to_dpo_dataset` へ素通しし、`input` 内へ載る（採用 0 件でも委譲する）
"""

from __future__ import annotations

import codecs
import csv
import inspect
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from oai_agentspec.runtime.finetune import (
    DatasetBuildResult,
    FineTuneError,
    FineTuneFailureKind,
    finalize_dpo_draft,
    save_dpo_draft,
)

pytestmark = pytest.mark.unit

_CSV_COLUMNS = [
    "case_index",
    "context",
    "response",
    "preferred_output",
    "non_preferred_output",
    "input_json",
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

_CONTEXT_1: list[dict[str, Any]] = [
    {"role": "user", "content": "会員登録の手順を教えて"},
    _TOOL_CALL_MESSAGE,
    _TOOL_OUTPUT_MESSAGE,
]
_CONTEXT_2: list[dict[str, Any]] = [
    *_CONTEXT_1,
    {"role": "assistant", "content": "手順は次の通りです: ..."},
    {"role": "user", "content": "料金は?"},
]

# 雛形モード（`dpo_dataset_from_session(session)`）が返す形の記入用ケース列。
_DRAFT_CASES: list[dict[str, Any]] = [
    {
        "input": _CONTEXT_1,
        "preferred_output": "",
        "non_preferred_output": "",
        "response": "手順は次の通りです: ...",
    },
    {
        "input": _CONTEXT_2,
        "preferred_output": "",
        "non_preferred_output": "",
        "response": "月額 500 円です",
    },
]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """記入用 CSV を列名ベースで読み出す（テスト側の記入シミュレーション用）。"""
    with path.open(encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def _write_csv_rows(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """列名を指定して CSV を書き戻す（スプレッドシートでの再保存を模す）。"""
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _fill_csv(path: Path, fills: list[tuple[str, str]]) -> None:
    """CSV の記入 2 列へ値を書き込んで保存し直す（人手記入のシミュレーション）。"""
    rows = _read_csv_rows(path)
    for row, (preferred, non_preferred) in zip(rows, fills, strict=True):
        row["preferred_output"] = preferred
        row["non_preferred_output"] = non_preferred
    _write_csv_rows(path, list(rows[0]), rows)


def _fill_jsonl(path: Path, fills: list[tuple[str, str]]) -> None:
    """JSONL の記入 2 キーへ値を書き込んで保存し直す（人手記入のシミュレーション）。"""
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for case, (preferred, non_preferred) in zip(cases, fills, strict=True):
        case["preferred_output"] = preferred
        case["non_preferred_output"] = non_preferred
    path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )


def _expected_records(*fills: tuple[list[dict[str, Any]], str, str]) -> tuple[dict[str, Any], ...]:
    """文脈と記入値から `to_dpo_dataset` 委譲後の最終レコード列を組む（期待値の組み立て）。"""
    return tuple(
        {
            "input": {"messages": context},
            "preferred_output": [{"role": "assistant", "content": preferred}],
            "non_preferred_output": [{"role": "assistant", "content": non_preferred}],
        }
        for context, preferred, non_preferred in fills
    )


# ----------------------------------------------------------------------
# save_dpo_draft（書き出し）
# ----------------------------------------------------------------------


def test_save_csv_writes_six_self_contained_columns(tmp_path: Path) -> None:
    """CSV は 6 列（参照 3 列 + 記入 2 列 + 機械用 1 列）で 1 ファイル自己完結する。

    `case_index` は 1 始まり、`response` は実応答の参照列、記入 2 列は空、`input_json` は
    累積文脈 messages の JSON 文字列（復元はこの列のみが担う）。`context` は人が読むための
    非可逆な整形列であり、文脈の本文が含まれることのみを固定する。
    """
    target = tmp_path / "draft.csv"

    save_dpo_draft(_DRAFT_CASES, target)

    with target.open(encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        assert reader.fieldnames == _CSV_COLUMNS
        rows = list(reader)
    assert [row["case_index"] for row in rows] == ["1", "2"]
    assert [row["response"] for row in rows] == ["手順は次の通りです: ...", "月額 500 円です"]
    assert [row["preferred_output"] for row in rows] == ["", ""]
    assert [row["non_preferred_output"] for row in rows] == ["", ""]
    assert [json.loads(row["input_json"]) for row in rows] == [_CONTEXT_1, _CONTEXT_2]
    assert "会員登録の手順を教えて" in rows[0]["context"]


def test_save_csv_formats_non_list_input_into_context_column(tmp_path: Path) -> None:
    """`input` が messages リストでない（文字列）ケースも書き出せ、context 列は素の文字列になる。

    文字列 `input` は委譲先 `to_dpo_dataset` が受理する正当な形であり、`save_dpo_draft` は
    型を検査せず通す。参照列の整形が空文字・固定文へ退行する変異を検知する pin
    （復元は `input_json` 列が担うため、そちらは JSON としての往復を固定する）。
    """
    target = tmp_path / "draft.csv"
    cases = [
        {
            "input": "料金は?",
            "preferred_output": "",
            "non_preferred_output": "",
            "response": "月額 500 円です",
        }
    ]

    save_dpo_draft(cases, target)

    rows = _read_csv_rows(target)
    assert rows[0]["context"] == "料金は?"
    assert json.loads(rows[0]["input_json"]) == "料金は?"


def test_save_csv_accepts_dataset_build_result(tmp_path: Path) -> None:
    """source には雛形モードの `DatasetBuildResult` をそのまま渡せる（records を読む）。"""
    target = tmp_path / "draft.csv"

    save_dpo_draft(DatasetBuildResult(records=tuple(_DRAFT_CASES), skipped=3), target)

    rows = _read_csv_rows(target)
    assert [json.loads(row["input_json"]) for row in rows] == [_CONTEXT_1, _CONTEXT_2]


def test_save_csv_is_written_with_utf8_bom(tmp_path: Path) -> None:
    """CSV は BOM 付き（utf-8-sig）で書かれる（日本語環境の Excel で文字化けしない）。"""
    target = tmp_path / "draft.csv"

    save_dpo_draft(_DRAFT_CASES, target)

    assert target.read_bytes().startswith(codecs.BOM_UTF8)


def test_save_jsonl_writes_one_case_per_line(tmp_path: Path) -> None:
    """`.jsonl` は記入用ケースを 1 行 1 件で書き出す（`DatasetBuildResult.save()` と同内容）。"""
    target = tmp_path / "draft.jsonl"

    save_dpo_draft(_DRAFT_CASES, target)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == _DRAFT_CASES


def test_save_with_unknown_extension_raises_config_missing(tmp_path: Path) -> None:
    """`.csv` / `.jsonl` 以外の拡張子は CONFIG_MISSING（lib は既定形式を発明しない）。"""
    target = tmp_path / "draft.txt"

    with pytest.raises(FineTuneError) as exc_info:
        save_dpo_draft(_DRAFT_CASES, target)

    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING
    assert not target.exists()


def test_save_rejects_case_missing_required_key_without_writing_file(tmp_path: Path) -> None:
    """必須キーを欠く要素があると書き出し前に全件検証して失敗し、ファイルを残さない。

    エラーは要素位置（1 始まり）と欠落キー名を含む。部分的に書かれたファイルが残ると
    利用者が壊れた雛形へ記入してしまうため、検証は全件先行で行う。
    """
    target = tmp_path / "draft.csv"
    broken = [_DRAFT_CASES[0], {"input": _CONTEXT_2, "preferred_output": ""}]

    with pytest.raises(FineTuneError) as exc_info:
        save_dpo_draft(broken, target)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース 2" in exc_info.value.message
    assert "non_preferred_output" in exc_info.value.message
    assert not target.exists()


def test_save_rejects_non_dict_case_without_writing_file(tmp_path: Path) -> None:
    """要素が dict でない source は要素位置つき VALIDATION_FAILED で失敗し、ファイルを残さない。

    必須キー欠落と同じく全件先行検証の対象であり、壊れた雛形を部分的に書き残さない
    （ADR 0034 Decision 6）。
    """
    target = tmp_path / "draft.csv"
    broken = [_DRAFT_CASES[0], "not-a-case"]

    with pytest.raises(FineTuneError) as exc_info:
        save_dpo_draft(broken, target)  # type: ignore[list-item]

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース 2" in exc_info.value.message
    assert not target.exists()


def test_save_rejects_case_missing_response_key_without_writing_file(tmp_path: Path) -> None:
    """参照欄 `response` を欠くケースも欠落キー名つき VALIDATION_FAILED で失敗する（FR-12）。

    書き出す雛形は 4 キー（`input` / 記入 2 欄 / `response`）を必須とする。実応答の参照欄が
    空の雛形を書くと、記入者が何と比較して良し悪しを判断するかの手がかりを失うため、空文字で
    埋めず fail-closed にする。取り込み側（`finalize_dpo_draft`）は参照欄を使わないため
    `response` を要求せず、必須キーは経路間で非対称である。
    """
    target = tmp_path / "draft.csv"
    broken = [
        _DRAFT_CASES[0],
        {"input": _CONTEXT_2, "preferred_output": "", "non_preferred_output": ""},
    ]

    with pytest.raises(FineTuneError) as exc_info:
        save_dpo_draft(broken, target)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース 2" in exc_info.value.message
    assert "response" in exc_info.value.message
    assert not target.exists()


def test_save_rejects_unserializable_input_without_writing_file(tmp_path: Path) -> None:
    """`input` を JSON へ直列化できないケースは位置つき VALIDATION_FAILED で失敗する。

    `input_json` 列の直列化は全件を組んでからファイルを開くため、途中の失敗で部分的に
    書かれた雛形が残らない（直列化を書き込みループ内で行う変異が RED になる）。
    """
    target = tmp_path / "draft.csv"
    broken = [
        _DRAFT_CASES[0],
        {
            "input": [{"role": "user", "content": {1, 2}}],
            "preferred_output": "",
            "non_preferred_output": "",
            "response": "応答",
        },
    ]

    with pytest.raises(FineTuneError) as exc_info:
        save_dpo_draft(broken, target)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース 2" in exc_info.value.message
    assert not target.exists()


def test_save_rejects_single_dict_source_without_writing_file(tmp_path: Path) -> None:
    """source が単一 dict / 文字列（ケースの列でない）場合は VALIDATION_FAILED で失敗する。"""
    target = tmp_path / "draft.csv"

    with pytest.raises(FineTuneError) as exc_info:
        save_dpo_draft(_DRAFT_CASES[0], target)  # type: ignore[arg-type]

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert not target.exists()

    with pytest.raises(FineTuneError):
        save_dpo_draft("draft", target)  # type: ignore[arg-type]

    assert not target.exists()


# ----------------------------------------------------------------------
# finalize_dpo_draft（取り込み・round-trip）
# ----------------------------------------------------------------------


def test_csv_round_trip_produces_preference_records(tmp_path: Path) -> None:
    """save -> 記入 -> finalize の CSV 往復で最終 preference レコードが得られる。

    文脈は `input_json` から復元され（ツール往復メッセージを含む）、記入した素の文字列は
    `to_dpo_dataset` 委譲により assistant 1 件のメッセージへ包まれる。
    """
    target = tmp_path / "draft.csv"
    save_dpo_draft(_DRAFT_CASES, target)

    _fill_csv(target, [("よい応答 1", "わるい応答 1"), ("よい応答 2", "わるい応答 2")])
    result = finalize_dpo_draft(target)

    assert result.records == _expected_records(
        (_CONTEXT_1, "よい応答 1", "わるい応答 1"),
        (_CONTEXT_2, "よい応答 2", "わるい応答 2"),
    )
    assert result.skipped == 0


def test_jsonl_round_trip_produces_preference_records(tmp_path: Path) -> None:
    """save -> 記入 -> finalize の JSONL 往復でも同じ最終レコードが得られる。"""
    target = tmp_path / "draft.jsonl"
    save_dpo_draft(_DRAFT_CASES, target)

    _fill_jsonl(target, [("よい応答 1", "わるい応答 1"), ("よい応答 2", "わるい応答 2")])
    result = finalize_dpo_draft(target)

    assert result.records == _expected_records(
        (_CONTEXT_1, "よい応答 1", "わるい応答 1"),
        (_CONTEXT_2, "よい応答 2", "わるい応答 2"),
    )
    assert result.skipped == 0


def test_jsonl_round_trip_survives_unicode_line_separators(tmp_path: Path) -> None:
    """U+2028 / U+2029 / U+0085 を含む値でも JSONL 往復が壊れない（行分割は改行のみ）。

    `json.dumps(ensure_ascii=False)` はこれらを素通しするため 1 行として書けるが、読み取りを
    `str.splitlines()` で行うと Unicode 行境界でも分割され、1 件が 2 断片になって
    `VALIDATION_FAILED` になる（書けたのに読めない非対称。CSV 経路は往復できる）。
    web や文書からのコピーで混入しやすい文字である。
    """
    marks = "\u2028\u2029\u0085"
    case = {
        "input": [{"role": "user", "content": f"質問{marks}続き"}],
        "preferred_output": f"よい{marks}応答",
        "non_preferred_output": "わるい応答",
        "response": f"実応答{marks}続き",
    }
    target = tmp_path / "draft.jsonl"
    save_dpo_draft([case], target)

    result = finalize_dpo_draft(target)

    assert result.skipped == 0
    assert result.records == _expected_records(
        ([{"role": "user", "content": f"質問{marks}続き"}], f"よい{marks}応答", "わるい応答")
    )


def test_finalize_reads_csv_by_column_name_regardless_of_order(tmp_path: Path) -> None:
    """列を並べ替えて再保存した CSV でも列名ベースで正しく読み取る（列順非依存）。"""
    target = tmp_path / "draft.csv"
    save_dpo_draft(_DRAFT_CASES[:1], target)
    _fill_csv(target, [("よい応答", "わるい応答")])

    rows = _read_csv_rows(target)
    _write_csv_rows(target, list(reversed(_CSV_COLUMNS)), rows)
    result = finalize_dpo_draft(target)

    assert result.records == _expected_records((_CONTEXT_1, "よい応答", "わるい応答"))


def test_finalize_reads_csv_without_bom(tmp_path: Path) -> None:
    """BOM なし UTF-8 で再保存された CSV も取り込める（utf-8-sig デコードの受理範囲）。"""
    target = tmp_path / "draft.csv"
    with target.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "case_index": "1",
                "context": "user: 会員登録の手順を教えて",
                "response": "手順は次の通りです: ...",
                "preferred_output": "よい応答",
                "non_preferred_output": "わるい応答",
                "input_json": json.dumps(_CONTEXT_1, ensure_ascii=False),
            }
        )

    result = finalize_dpo_draft(target)

    assert result.records == _expected_records((_CONTEXT_1, "よい応答", "わるい応答"))


def test_finalize_rejects_cp932_saved_csv_as_validation_failed(tmp_path: Path) -> None:
    """cp932 等で保存し直された CSV は生の `UnicodeDecodeError` を漏らさず案内つきで失敗する。

    `test_finalize_reads_csv_without_bom`（受理範囲）と対の pin で、デコード失敗を
    `VALIDATION_FAILED` へ変換する経路を落とす変異が RED になる（lib は文字化けの silent な
    取り込みを避けるため自動判定しない契約）。
    """
    target = tmp_path / "draft.csv"
    save_dpo_draft(_DRAFT_CASES[:1], target)
    target.write_bytes(target.read_text(encoding="utf-8-sig").encode("cp932"))

    with pytest.raises(FineTuneError) as exc_info:
        finalize_dpo_draft(target)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "utf-8-sig" in exc_info.value.message
    assert "cp932" in exc_info.value.message
    assert "保存し直" in exc_info.value.message
    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


def test_finalize_ignores_reference_columns(tmp_path: Path) -> None:
    """参照列（case_index / context / response）は取り込みに影響せず記録にも現れない。

    参照列を編集・破壊しても結果は変わらず、`response` キーは最終レコードへ現れない
    （`to_dpo_dataset` が指定キーのみを読む帰結）。
    """
    target = tmp_path / "draft.csv"
    save_dpo_draft(_DRAFT_CASES[:1], target)
    _fill_csv(target, [("よい応答", "わるい応答")])

    rows = _read_csv_rows(target)
    rows[0]["case_index"] = "書き換えた"
    rows[0]["context"] = ""
    rows[0]["response"] = "編集された参照列"
    _write_csv_rows(target, _CSV_COLUMNS, rows)
    result = finalize_dpo_draft(target)

    assert result.records == _expected_records((_CONTEXT_1, "よい応答", "わるい応答"))
    assert "response" not in result.records[0]


def test_finalize_accepts_iterable_of_cases() -> None:
    """メモリ上の記入済みケース列（iterable）もそのまま取り込める（ファイル不要）。"""
    cases = [
        {**_DRAFT_CASES[0], "preferred_output": "よい応答", "non_preferred_output": "わるい応答"}
    ]

    result = finalize_dpo_draft(cases)

    assert result.records == _expected_records((_CONTEXT_1, "よい応答", "わるい応答"))
    assert result.skipped == 0


def test_finalize_with_unknown_extension_raises_config_missing(tmp_path: Path) -> None:
    """パスの拡張子が `.csv` / `.jsonl` 以外なら CONFIG_MISSING（save と対称）。"""
    target = tmp_path / "draft.txt"
    target.write_text("dummy", encoding="utf-8")

    with pytest.raises(FineTuneError) as exc_info:
        finalize_dpo_draft(target)

    assert exc_info.value.kind == FineTuneFailureKind.CONFIG_MISSING


def test_finalize_missing_file_propagates_read_error(tmp_path: Path) -> None:
    """存在しないパスの読み取りエラーは握り潰さず呼び出し側へ伝播する（fail-closed）。"""
    with pytest.raises(OSError):
        finalize_dpo_draft(tmp_path / "absent.csv")


# ----------------------------------------------------------------------
# 記入判定（両欄空 skip / 片欄エラー / 値非改変）
# ----------------------------------------------------------------------


def test_blank_and_whitespace_only_cases_are_skipped(tmp_path: Path) -> None:
    """両記入欄が空（strip 後に空・空白のみセルを含む）のケースは skip して skipped へ計上する。"""
    target = tmp_path / "draft.csv"
    save_dpo_draft(_DRAFT_CASES, target)

    _fill_csv(target, [("よい応答", "わるい応答"), ("   ", " \n ")])
    result = finalize_dpo_draft(target)

    assert result.records == _expected_records((_CONTEXT_1, "よい応答", "わるい応答"))
    assert result.skipped == 1


def test_all_unfilled_returns_empty_result_not_error(tmp_path: Path) -> None:
    """全ケース未記入はエラーにせず `records == ()` / `skipped == 全件` を正常返却する。"""
    target = tmp_path / "draft.csv"
    save_dpo_draft(_DRAFT_CASES, target)

    result = finalize_dpo_draft(target)

    assert result.records == ()
    assert result.skipped == 2


def test_partially_filled_case_raises_validation_failed(tmp_path: Path) -> None:
    """片欄のみ記入のケースはケース位置つき VALIDATION_FAILED で失敗する（skip にしない）。"""
    target = tmp_path / "draft.csv"
    save_dpo_draft(_DRAFT_CASES, target)

    _fill_csv(target, [("よい応答", "わるい応答"), ("よい応答だけ書いた", "  ")])

    with pytest.raises(FineTuneError) as exc_info:
        finalize_dpo_draft(target)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース 2" in exc_info.value.message


def test_filled_values_are_passed_through_without_stripping(tmp_path: Path) -> None:
    """strip は未記入判定にのみ使い、採用値は前後空白・改行を含めて非改変で委譲する。"""
    target = tmp_path / "draft.csv"
    save_dpo_draft(_DRAFT_CASES[:1], target)

    _fill_csv(target, [("  よい\n応答  ", "\tわるい応答 ")])
    result = finalize_dpo_draft(target)

    assert result.records == _expected_records((_CONTEXT_1, "  よい\n応答  ", "\tわるい応答 "))


def test_finalize_skipped_excludes_draft_generation_count(tmp_path: Path) -> None:
    """`skipped` は finalize が数えた分のみで、雛形生成時のカウントは合算しない。"""
    target = tmp_path / "draft.csv"
    save_dpo_draft(DatasetBuildResult(records=tuple(_DRAFT_CASES), skipped=5), target)

    _fill_csv(target, [("よい応答", "わるい応答"), ("", "")])
    result = finalize_dpo_draft(target)

    assert result.skipped == 1


# ----------------------------------------------------------------------
# 取り込み時のエラー方針（必須列欠落 / 不正 JSON）
# ----------------------------------------------------------------------


def test_missing_required_column_raises_validation_failed_with_column_name(tmp_path: Path) -> None:
    """必須列を欠く CSV は欠落列名だけを挙げた VALIDATION_FAILED で失敗する。

    列削除・別ファイルの誤指定を silent な全件 skip にしない fail-closed の pin。欠落列名の
    集合は過小側（列名を挙げない）・過大側（存在する必須列まで挙げる）の両方向で固定する
    （メッセージ末尾の「必須列: ...」は列の案内であり欠落報告ではないため判定から除く）。
    """
    target = tmp_path / "draft.csv"
    save_dpo_draft(_DRAFT_CASES[:1], target)
    _fill_csv(target, [("よい応答", "わるい応答")])

    rows = _read_csv_rows(target)
    columns = [column for column in _CSV_COLUMNS if column != "preferred_output"]
    _write_csv_rows(target, columns, rows)

    with pytest.raises(FineTuneError) as exc_info:
        finalize_dpo_draft(target)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    reported = exc_info.value.message.split("（", 1)[0]
    assert "preferred_output" in reported
    assert "non_preferred_output" not in reported
    assert "input_json" not in reported


def test_invalid_input_json_raises_validation_failed_with_position(tmp_path: Path) -> None:
    """`input_json` が JSON として不正ならケース位置つき VALIDATION_FAILED で失敗する。"""
    target = tmp_path / "draft.csv"
    save_dpo_draft(_DRAFT_CASES, target)
    _fill_csv(target, [("よい応答 1", "わるい応答 1"), ("よい応答 2", "わるい応答 2")])

    rows = _read_csv_rows(target)
    rows[1]["input_json"] = "{壊れた JSON"
    _write_csv_rows(target, _CSV_COLUMNS, rows)

    with pytest.raises(FineTuneError) as exc_info:
        finalize_dpo_draft(target)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース 2" in exc_info.value.message


def test_invalid_jsonl_line_raises_validation_failed_with_position(tmp_path: Path) -> None:
    """JSONL のいずれかの行が JSON として不正ならケース位置つき VALIDATION_FAILED で失敗する。

    CSV の `input_json` 不正と対称の fail-closed の pin（壊れた行を silent に読み飛ばさない）。
    """
    target = tmp_path / "draft.jsonl"
    target.write_text(
        json.dumps(
            {
                "input": _CONTEXT_1,
                "preferred_output": "よい応答",
                "non_preferred_output": "わるい応答",
            },
            ensure_ascii=False,
        )
        + "\n{壊れた JSON\n",
        encoding="utf-8",
    )

    with pytest.raises(FineTuneError) as exc_info:
        finalize_dpo_draft(target)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース 2" in exc_info.value.message


def test_null_fill_values_are_treated_as_unfilled(tmp_path: Path) -> None:
    """記入欄が `null`（None）のケースは未記入として skip する（空文字と同じ扱い）。

    JSONL の null・CSV の欠損セルを「記入済み」と誤判定して委譲先の検証違反へ落とす変異を
    検知する pin。
    """
    target = tmp_path / "draft.jsonl"
    target.write_text(
        json.dumps(
            {"input": _CONTEXT_1, "preferred_output": None, "non_preferred_output": None},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = finalize_dpo_draft(target)

    assert result.records == ()
    assert result.skipped == 1


def test_non_string_fill_values_are_delegated_without_modification(tmp_path: Path) -> None:
    """str 以外の記入値は「未記入でない」と判定し、非改変のまま委譲する。

    メッセージ配列で記入した値は assistant へ包み直されず透過し（`to_dpo_dataset` の
    非改変透過）、委譲先が受け付けない型（数値）は skip せず VALIDATION_FAILED で失敗する
    （非記入扱いの silent skip にしない fail-closed）。
    """
    preferred_messages = [{"role": "assistant", "content": "よい応答", "weight": 1}]
    passthrough = finalize_dpo_draft(
        [
            {
                "input": _CONTEXT_1,
                "preferred_output": preferred_messages,
                "non_preferred_output": "わるい応答",
            }
        ]
    )

    assert passthrough.records[0]["preferred_output"] == preferred_messages
    assert passthrough.skipped == 0

    with pytest.raises(FineTuneError) as exc_info:
        finalize_dpo_draft(
            [{"input": _CONTEXT_1, "preferred_output": 42, "non_preferred_output": 7}]
        )

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


def test_finalize_rejects_case_missing_required_key_in_jsonl(tmp_path: Path) -> None:
    """JSONL 経路も必須キーを全件検証し、欠落があればケース位置つきで失敗する。

    CSV の必須列欠落エラーと対称の fail-closed の pin。記入 2 キーをともに欠くケースは、
    検証を通さないと「両欄未記入 = skip」へ落ちて silent に件数だけ減るため、キー欠落と
    未記入 skip を取り違えない形（両キー欠落で失敗すること）で固定する。
    """
    target = tmp_path / "draft.jsonl"
    target.write_text(
        json.dumps(
            {
                "input": _CONTEXT_1,
                "preferred_output": "よい応答",
                "non_preferred_output": "わるい応答",
            },
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps({"input": _CONTEXT_2}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FineTuneError) as exc_info:
        finalize_dpo_draft(target)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース 2" in exc_info.value.message
    assert "preferred_output" in exc_info.value.message
    assert "non_preferred_output" in exc_info.value.message


def test_finalize_rejects_case_missing_required_key_in_iterable() -> None:
    """メモリ上のケース列も必須キーを全件検証する（JSONL 経路と同じ fail-closed）。"""
    cases = [
        {
            "input": _CONTEXT_1,
            "preferred_output": "よい応答",
            "non_preferred_output": "わるい応答",
        },
        {"input": _CONTEXT_2},
    ]

    with pytest.raises(FineTuneError) as exc_info:
        finalize_dpo_draft(cases)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース 2" in exc_info.value.message
    assert "preferred_output" in exc_info.value.message
    assert "non_preferred_output" in exc_info.value.message


def test_finalize_accepts_cases_without_response_reference_key(tmp_path: Path) -> None:
    """取り込みは参照欄 `response` を要求しない（save と必須キーが非対称であることの pin）。

    JSONL 経路・iterable 経路のいずれも、記入 2 欄と `input` があれば `response` の有無に
    かかわらず同じ最終レコードを返す。取り込み側にも `response` を要求する変異（非対称性の
    過大側）が RED になる。
    """
    filled = {
        "input": _CONTEXT_1,
        "preferred_output": "よい応答",
        "non_preferred_output": "わるい応答",
    }
    target = tmp_path / "filled.jsonl"
    target.write_text(json.dumps(filled, ensure_ascii=False) + "\n", encoding="utf-8")

    from_jsonl = finalize_dpo_draft(target)
    from_iterable = finalize_dpo_draft([filled])

    expected = _expected_records((_CONTEXT_1, "よい応答", "わるい応答"))
    assert from_jsonl.records == expected
    assert from_iterable.records == expected
    assert from_jsonl.skipped == 0


# ----------------------------------------------------------------------
# 巨大セル / 直列化不能 / 値非改変（レビュー指摘 WARNING の pin）
# ----------------------------------------------------------------------


@pytest.fixture
def restore_csv_field_size_limit() -> Iterator[None]:
    """`csv.field_size_limit` はプロセスグローバルなので、テスト後に必ず元へ戻す。"""
    original = csv.field_size_limit()
    try:
        yield
    finally:
        csv.field_size_limit(original)


@pytest.mark.usefixtures("restore_csv_field_size_limit")
def test_finalize_reports_oversized_cell_as_validation_failed(tmp_path: Path) -> None:
    """`csv.field_size_limit` 超過のセルは生の `csv.Error` を漏らさず VALIDATION_FAILED になる。

    メッセージには利用者が自力で対処できるよう `field_size_limit` の案内を含める。
    捕捉を外す変異（`csv.Error` が素通りする）が RED になる。なお `save_dpo_draft` は
    上限超過のセルを書き出す前に弾くため（`test_save_rejects_oversized_cell_*`）、本 pin は
    上限を引き上げて書いた雛形を既定上限で読む状況（外部生成 CSV 相当）を作って測る。
    """
    csv.field_size_limit(10_000_000)
    target = tmp_path / "draft.csv"
    oversized = "あ" * 140_000
    save_dpo_draft(
        [
            {
                "input": _CONTEXT_1,
                "preferred_output": "",
                "non_preferred_output": "",
                "response": oversized,
            }
        ],
        target,
    )
    csv.field_size_limit(131072)

    with pytest.raises(FineTuneError) as exc_info:
        finalize_dpo_draft(target)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "field_size_limit" in exc_info.value.message
    assert isinstance(exc_info.value.__cause__, csv.Error)


@pytest.mark.usefixtures("restore_csv_field_size_limit")
def test_save_rejects_oversized_cell_without_writing_file(tmp_path: Path) -> None:
    """`csv.field_size_limit` を超えるセルは書き出す前に弾き、ファイルを作らない。

    save はセルサイズ無制限・read は上限ありという非対称（人手記入後に発覚する）を
    書き出し側で閉じる pin。事前検査を削除する変異が RED になる。上限はプロセス
    グローバルのため fixture で必ず復元する。
    """
    csv.field_size_limit(131072)
    target = tmp_path / "draft.csv"
    oversized = "あ" * 140_000

    with pytest.raises(FineTuneError) as exc_info:
        save_dpo_draft(
            [
                {
                    "input": _CONTEXT_1,
                    "preferred_output": "",
                    "non_preferred_output": "",
                    "response": oversized,
                }
            ],
            target,
        )

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース 1" in exc_info.value.message
    assert "response" in exc_info.value.message
    assert "field_size_limit" in exc_info.value.message
    assert not target.exists()


@pytest.mark.usefixtures("restore_csv_field_size_limit")
def test_save_cell_size_check_follows_current_limit(tmp_path: Path) -> None:
    """事前検査は実行時の `csv.field_size_limit()` を読む（上限を上げれば書ける）。

    上限をハードコードする変異・常に失敗させる変異（過大側）が RED になる。
    """
    csv.field_size_limit(10_000_000)
    target = tmp_path / "draft.csv"
    case = {
        "input": _CONTEXT_1,
        "preferred_output": "",
        "non_preferred_output": "",
        "response": "あ" * 140_000,
    }

    save_dpo_draft([case], target)

    assert target.exists()
    assert len(_read_csv_rows(target)) == 1


def test_save_tolerates_non_dict_function_in_tool_calls(tmp_path: Path) -> None:
    """`tool_calls` の `function` が非 dict でも参照列の整形で AttributeError を出さない。

    保存前に `records` を加工してマスキングする経路（docstring 推奨）で到達する形。
    フォールバックを削除する変異（生の `AttributeError` が漏れる）が RED になる。
    """
    target = tmp_path / "draft.csv"
    context = [
        {"role": "user", "content": "会員登録の手順を教えて"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_1", "type": "function", "function": None}],
        },
    ]

    save_dpo_draft(
        [
            {
                "input": context,
                "preferred_output": "",
                "non_preferred_output": "",
                "response": "手順は次の通りです: ...",
            }
        ],
        target,
    )

    rows = _read_csv_rows(target)
    assert len(rows) == 1
    assert "[tool_calls]" in rows[0]["context"]
    assert json.loads(rows[0]["input_json"]) == context


def test_blank_input_json_with_filled_columns_reports_empty_column(tmp_path: Path) -> None:
    """`input_json` 空 + 記入欄ありの行は「列が空」と分かるエラーにする（不正 JSON ではない）。

    存在しない構文エラーを利用者に探させないための pin。メッセージを
    「JSON として不正」へ戻す変異が RED になる。
    """
    target = tmp_path / "draft.csv"
    save_dpo_draft(_DRAFT_CASES, target)
    _fill_csv(target, [("よい応答", "わるい応答"), ("", "")])
    rows = _read_csv_rows(target)
    rows[0]["input_json"] = ""
    _write_csv_rows(target, _CSV_COLUMNS, rows)

    with pytest.raises(FineTuneError) as exc_info:
        finalize_dpo_draft(target)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース 1" in exc_info.value.message
    assert "input_json" in exc_info.value.message
    assert "空" in exc_info.value.message
    assert "JSON として不正" not in exc_info.value.message


def test_separator_only_row_is_skipped_without_failing_the_whole_import(tmp_path: Path) -> None:
    """必須 3 列がすべて空の行（区切り文字だけの行）は取り込み全体を失敗させず skip する。

    表計算ソフトが末尾へ出力する空行を想定した pin。`input_json` のパースより前に判定する
    経路を落とす変異が RED になる（退行時は `input_json` 列が空です で全体が失敗する）。
    過大側（記入欄に値がある行まで skip する変異）は
    `test_blank_input_json_with_filled_columns_reports_empty_column` と対で測る。
    """
    target = tmp_path / "draft.csv"
    save_dpo_draft(_DRAFT_CASES[:1], target)
    _fill_csv(target, [("よい応答", "わるい応答")])
    baseline = finalize_dpo_draft(target)

    with target.open("a", encoding="utf-8-sig", newline="") as fp:
        fp.write("," * (len(_CSV_COLUMNS) - 1) + "\r\n")
    result = finalize_dpo_draft(target)

    assert result.records == baseline.records
    assert result.records == _expected_records((_CONTEXT_1, "よい応答", "わるい応答"))
    assert result.skipped == baseline.skipped + 1


def test_save_jsonl_rejects_unserializable_case_without_writing_file(tmp_path: Path) -> None:
    """JSONL 経路も直列化不能ケースを位置つき VALIDATION_FAILED にし、ファイルを残さない。

    CSV 経路（`test_save_rejects_unserializable_input_without_writing_file`）と対の pin で、
    JSONL だけ生の `TypeError` が漏れる / 部分ファイルが残る変異が RED になる。
    """
    target = tmp_path / "draft.jsonl"
    broken = [
        _DRAFT_CASES[0],
        {
            "input": [{"role": "user", "content": {1, 2}}],
            "preferred_output": "",
            "non_preferred_output": "",
            "response": "応答",
        },
    ]

    with pytest.raises(FineTuneError) as exc_info:
        save_dpo_draft(broken, target)

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "ケース 2" in exc_info.value.message
    assert not target.exists()


def test_save_writes_formula_like_values_without_modification(tmp_path: Path) -> None:
    """数式風の値（`=` / `+` / `-` / `@` 始まり）を無害化せず非改変で書く（Decision 10）。

    CSV インジェクション対策の無害化（先頭への `'` 前置等）は lib では行わない契約。
    参照列の値へ無害化を入れる変異が RED になる。
    """
    target = tmp_path / "draft.csv"
    formula = "=SUM(A1)+HYPERLINK(B1)"
    save_dpo_draft(
        [
            {
                "input": [{"role": "user", "content": formula}],
                "preferred_output": "",
                "non_preferred_output": "",
                "response": formula,
            }
        ],
        target,
    )

    row = _read_csv_rows(target)[0]

    assert row["response"] == formula
    assert row["context"] == f"user: {formula}"
    assert json.loads(row["input_json"]) == [{"role": "user", "content": formula}]


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

# 記入済みのケース列（iterable 経路の透過テスト用）。
_FILLED_CASES: list[dict[str, Any]] = [
    {
        "input": _CONTEXT_1,
        "preferred_output": "よい応答",
        "non_preferred_output": "わるい応答",
    }
]


def test_tools_pass_through_inside_input() -> None:
    """`tools=` は委譲先 `to_dpo_dataset` へ素通しされ、`record["input"]["tools"]` に載る（P1）。"""
    result = finalize_dpo_draft(_FILLED_CASES, tools=_PLAIN_TOOLS)

    assert result.records[0]["input"]["tools"] == _PLAIN_TOOLS
    assert "tools" not in result.records[0]


def test_tools_pass_through_from_csv_source(tmp_path: Path) -> None:
    """CSV パス経路でも `tools=` は `input` 内へ透過する（`_cases_from_csv` 分岐の回帰 pin）。"""
    target = tmp_path / "draft.csv"
    save_dpo_draft(_DRAFT_CASES[:1], target)
    _fill_csv(target, [("よい応答", "わるい応答")])

    result = finalize_dpo_draft(target, tools=_PLAIN_TOOLS)

    assert result.records[0]["input"]["messages"] == _CONTEXT_1
    assert result.records[0]["input"]["tools"] == _PLAIN_TOOLS


def test_tools_function_tool_like_is_mapped_by_delegate() -> None:
    """`FunctionTool` 相当オブジェクトは委譲先の写像を経て dict で `input` へ載る（P2）。"""
    tool = _FakeFunctionTool(
        name="lookup_faq",
        params_json_schema={"type": "object"},
        description="FAQ を検索する",
    )

    result = finalize_dpo_draft(_FILLED_CASES, tools=[tool])

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


def test_invalid_tools_element_raises_validation_failed() -> None:
    """不正要素を含む `tools=` は委譲先の検証で VALIDATION_FAILED になる（P3）。"""
    with pytest.raises(FineTuneError) as exc_info:
        finalize_dpo_draft(_FILLED_CASES, tools=[42])

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED
    assert "tools[0]" in exc_info.value.message


def test_invalid_tools_raises_even_when_all_cases_are_unfilled(tmp_path: Path) -> None:
    """全ケース未記入でも不正 `tools=` は VALIDATION_FAILED になる（P4(c)）。

    採用 0 件で委譲せず早期 return する変異（不正 tools が素通りする経路）が RED になる。
    """
    target = tmp_path / "draft.csv"
    save_dpo_draft(_DRAFT_CASES, target)

    with pytest.raises(FineTuneError) as exc_info:
        finalize_dpo_draft(target, tools=[42])

    assert exc_info.value.kind == FineTuneFailureKind.VALIDATION_FAILED


def test_parallel_tool_calls_false_is_passed_through_inside_input() -> None:
    """`parallel_tool_calls=False` は `input` 内へ載る（P5・truthy 判定への退行を検知）。"""
    result = finalize_dpo_draft(_FILLED_CASES, parallel_tool_calls=False)

    assert result.records[0]["input"]["parallel_tool_calls"] is False


def test_tool_keys_are_absent_when_arguments_are_omitted() -> None:
    """未指定なら `input` 内に tools / parallel_tool_calls のキー自体が出ない（P12）。"""
    result = finalize_dpo_draft(_FILLED_CASES)

    assert "tools" not in result.records[0]["input"]
    assert "parallel_tool_calls" not in result.records[0]["input"]


def test_finalize_dpo_draft_signature_is_source_positional_and_keyword_only_tools() -> None:
    """位置引数は `source` のみで、追加 2 引数は keyword-only かつ既定 None（P10）。"""
    parameters = inspect.signature(finalize_dpo_draft).parameters
    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    assert positional == ["source"]
    for name in ("tools", "parallel_tool_calls"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is None
