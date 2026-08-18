"""Fine-Tuning データセットの変換・検証ヘルパ（純データ層・SDK 非接触）。

`to_sft_dataset` / `to_dpo_dataset` は EvalCase / OptimizeCase / `DpoCase` / plain dict の列を
OpenAI 公式 SFT / DPO（preference）形式のレコード列へ変換し、`validate_dataset` は持ち込み
JSONL（またはレコード列）を同形式に照らして検証する。`agents` / `openai` を import せず、
ネットワークにも触れない（ローカルファイルの読み書きのみ）。

非改変透過の実装形: 入力 messages リストの各要素 dict は **copy せず参照のまま** 出力レコードへ
載せる（リスト自体は新規に組み直す）。したがって `weight` / parts 配列 `content` /
`tool_call_id` / 未知キーはすべて保全される（契約）。呼び出し側が入力 dict を後から変更すると
結果にも波及する点に留意すること。

ケースからの値取り出しは `input_key` / `output_key`（DPO は `preferred_key` /
`non_preferred_key`）で指定したキー（属性）のみを読む。plain dict ケースにレコード別の
`"tools"` キー等が含まれていても読まない（レコード別 tools は持ち込み JSONL の責務）。

`tools=` / `parallel_tool_calls=` は全レコード共通の値として載る。`tools=` 由来のリストは
1 度だけ組み立てて全レコードで **同一オブジェクトを共有参照** するため、変換後に 1 レコードの
tools リストを変更すると全レコードへ波及する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .types import (
    DatasetBuildResult,
    DatasetValidationReport,
    DatasetViolation,
    FineTuneError,
    FineTuneFailureKind,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

_LEGAL_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
_LEGAL_ROLE_HINT = "system / developer / user / assistant / tool"


class _CaseDataError(Exception):
    """ケース単位のデータ不備（`skip_missing=True` のとき除外対象になる内部例外）。"""


def _case_value(case: Any, field_name: str) -> Any:
    """入力ケースからフィールド値を取り出す（dict / 属性アクセス両対応）。

    Args:
        case: 入力ケース（dict または属性を持つオブジェクト）。
        field_name: 取り出すフィールド名。

    Returns:
        フィールド値。dict なら `case[field_name]`、それ以外は `getattr(case, field_name, None)`。
    """
    if isinstance(case, dict):
        return case.get(field_name)
    return getattr(case, field_name, None)


def _check_message_element(value: Any, position: int, label: str) -> None:
    """messages リスト要素の境界検査（presence のみ・型の妥当性は検証しない）。

    Args:
        value: 検査対象の要素。
        position: 要素位置（0 始まり）。
        label: エラーメッセージに載せるフィールド名。

    Raises:
        _CaseDataError: dict でない / `role` が無い / `content` と `tool_calls` の両方が無い場合。
    """
    if not isinstance(value, dict):
        raise _CaseDataError(f"{label}[{position}] が dict でない")
    if "role" not in value:
        raise _CaseDataError(f"{label}[{position}] に 'role' が存在しない")
    if "content" not in value and "tool_calls" not in value:
        raise _CaseDataError(f"{label}[{position}] に content / tool_calls のいずれも存在しない")


def _to_messages(value: Any, label: str, *, wrap_role: str) -> list[dict[str, Any]]:
    """文字列 / messages リストの二形を messages リストへ正規化する（非改変透過）。

    Args:
        value: 文字列（1 件のメッセージへ包む）または messages 形式のリスト（透過）。
        label: エラーメッセージに載せるフィールド名。
        wrap_role: 文字列を包むときの role。

    Returns:
        messages のリスト（要素 dict は入力の参照をそのまま載せる）。

    Raises:
        _CaseDataError: 値が欠落 / 空リスト / 文字列でもリストでもない型 / 要素が境界検査に
            適合しない場合。
    """
    if isinstance(value, str):
        return [{"role": wrap_role, "content": value}]
    if isinstance(value, list):
        if not value:
            raise _CaseDataError(f"{label} が空リストである")
        for position, element in enumerate(value):
            _check_message_element(element, position, label)
        return list(value)
    if value is None:
        raise _CaseDataError(f"{label} が存在しない")
    raise _CaseDataError(f"{label} が文字列でもリストでもない型である: {type(value).__name__}")


def _map_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """`tools=` の各要素を FT の tools 定義形式へ正規化する（dict は非解釈透過）。

    dict はそのまま透過し、dict でない要素は `name` / `params_json_schema` 属性の
    ダックタイピングで FunctionTool 相当と判別して写像する（`strict_json_schema` は写像に
    含めない・`description` が無い場合は空文字）。呼び出し時点の属性値の静的スナップショットで
    あり `is_enabled` は評価しない。

    Args:
        tools: ツール定義の列（plain dict と FunctionTool 相当の混在可）。

    Returns:
        FT の tools 定義形式のリスト。

    Raises:
        FineTuneError: dict でも FunctionTool 相当でもない要素が含まれる場合
            （`skip_missing` の対象外・常時エラー）。
    """
    mapped: list[dict[str, Any]] = []
    for position, tool in enumerate(tools):
        if isinstance(tool, dict):
            mapped.append(tool)
            continue
        if hasattr(tool, "name") and hasattr(tool, "params_json_schema"):
            description = getattr(tool, "description", None)
            mapped.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": description if description is not None else "",
                        "parameters": tool.params_json_schema,
                    },
                }
            )
            continue
        raise FineTuneError(
            FineTuneFailureKind.VALIDATION_FAILED,
            f"tools[{position}] が dict でも FunctionTool 相当"
            f"（name / params_json_schema を持つ）でもない: {type(tool).__name__}",
        )
    return mapped


def _tool_fields(tools: list[Any] | None, parallel_tool_calls: bool | None) -> dict[str, Any]:
    """`tools=` / `parallel_tool_calls=` をレコードへ載せるフィールド dict を組む。

    Args:
        tools: ツール定義の列（None ならキー自体を出さない）。
        parallel_tool_calls: 並列ツール呼び出しの可否（None ならキー自体を出さない）。

    Returns:
        レコードへ合成するフィールド dict（指定が無ければ空 dict）。

    Raises:
        FineTuneError: `tools` に不正要素が含まれる場合。
    """
    fields: dict[str, Any] = {}
    if tools is not None:
        fields["tools"] = _map_tools(tools)
    if parallel_tool_calls is not None:
        fields["parallel_tool_calls"] = parallel_tool_calls
    return fields


def _case_error(index: int, exc: _CaseDataError) -> FineTuneError:
    """ケース位置と理由を載せた `FineTuneError` を組む。

    Args:
        index: ケース位置（1 始まり）。
        exc: 元のケース不備。

    Returns:
        構造化エラー。
    """
    return FineTuneError(FineTuneFailureKind.VALIDATION_FAILED, f"ケース {index}: {exc}")


def _reject_non_sequence_cases(cases: Any) -> None:
    """変換ヘルパの `cases` に単一 dict / 文字列が渡された場合を明示エラーにする。

    dict をそのまま反復するとキー文字列の列として、`str` / `bytes` は 1 文字（1 バイト）
    ずつのケース列として誤読され、`skip_missing=True` では空データセットを無言で返して
    しまうため、`skip_missing` によらず送出する。

    Args:
        cases: 変換対象として渡された値。

    Raises:
        FineTuneError: `cases` が単一の dict / `str` / `bytes` の場合。
    """
    if isinstance(cases, (dict, str, bytes)):
        raise FineTuneError(
            FineTuneFailureKind.VALIDATION_FAILED,
            "cases が単一の dict / 文字列である（ケースの列を渡す。1 件だけの場合も [case] の"
            "ように列へ包むこと）",
        )


def to_sft_dataset(
    cases: Iterable[Any],
    *,
    system: str | None = None,
    tools: list[Any] | None = None,
    parallel_tool_calls: bool | None = None,
    input_key: str = "input",
    output_key: str = "expected_output",
    skip_missing: bool = False,
) -> DatasetBuildResult:
    """ケース列を SFT（chat 形式）のレコード列へ変換する。

    `input` は文字列（user 1 件へ包む）または messages リスト（非改変透過）を受け、出力側は
    文字列（assistant 1 件へ包む）または assistant メッセージ配列（非改変透過）を受ける。
    出力側のメッセージは messages 末尾へ付す（末尾付加分に `weight` は付さない）。
    メッセージ dict は copy せず参照のまま載せるため `weight` / parts 配列 content /
    `tool_call_id` / 未知キーは保全される。ケースは `input_key` / `output_key` のみを読み、
    レコード別 `"tools"` キー等は読まない（持ち込み JSONL の責務）。

    Args:
        cases: ケース列（属性アクセス型 / plain dict の混在可）。
        system: 全レコードの先頭へ挿入する system メッセージ本文（None なら挿入しない）。
        tools: 全レコードへ透過するツール定義（plain dict / FunctionTool 相当の混在可）。
            None ならレコードへキー自体を出さない。
        parallel_tool_calls: 全レコードへ透過する並列ツール呼び出しの可否。None ならキー自体を
            出さない。
        input_key: plain dict ケースの入力キー名。
        output_key: plain dict ケースの出力キー名。
        skip_missing: True のとき、必須フィールド欠落・境界不備のケースを除外して
            `DatasetBuildResult.skipped` に件数を報告する。

    Returns:
        変換結果（`records` / `skipped`）。書き出しは `save(path)` の明示呼び出しのみ。

    Raises:
        FineTuneError: `skip_missing=False` でデータ不備がある場合、`cases` が単一の dict /
            `str` / `bytes` の場合、`tools=` に不正要素がある場合、`system=` と input リスト内
            system メッセージが競合する場合（後 3 者は `skip_missing` の対象外）。
    """
    _reject_non_sequence_cases(cases)
    record_fields = _tool_fields(tools, parallel_tool_calls)
    records: list[dict[str, Any]] = []
    skipped = 0
    for index, case in enumerate(cases, start=1):
        try:
            messages = _to_messages(_case_value(case, input_key), input_key, wrap_role="user")
            outputs = _to_messages(_case_value(case, output_key), output_key, wrap_role="assistant")
        except _CaseDataError as exc:
            if skip_missing:
                skipped += 1
                continue
            raise _case_error(index, exc) from exc
        if system is not None and any(
            isinstance(message, dict) and message.get("role") == "system" for message in messages
        ):
            raise FineTuneError(
                FineTuneFailureKind.VALIDATION_FAILED,
                f"ケース {index}: `system=` と {input_key} リスト内の system メッセージが"
                "競合する（暗黙マージ・暗黙置換はしない）",
            )
        head = [{"role": "system", "content": system}] if system is not None else []
        records.append({"messages": [*head, *messages, *outputs], **record_fields})
    return DatasetBuildResult(records=tuple(records), skipped=skipped)


def to_dpo_dataset(
    cases: Iterable[Any],
    *,
    tools: list[Any] | None = None,
    parallel_tool_calls: bool | None = None,
    input_key: str = "input",
    preferred_key: str = "preferred_output",
    non_preferred_key: str = "non_preferred_output",
    skip_missing: bool = False,
) -> DatasetBuildResult:
    """ケース列を DPO（preference 形式）のレコード列へ変換する。

    `input` は文字列（user 1 件へ包む）または messages リスト（非改変透過）を受け、出力側は
    文字列（assistant 1 件へ包む）または assistant メッセージ配列（非改変透過）を受ける。
    出力側は文字列で渡した場合のみ role `"assistant"` を付与して包み、リストで渡した場合は
    非改変透過のため role は強制しない（role の妥当性は `validate_dataset(method="dpo")` が
    検証する）。`tools` /
    `parallel_tool_calls` は `input` 内へ透過する（SFT はレコード直下）。system メッセージは
    `input` リスト内に含めて渡す（`system=` 引数は持たない）。メッセージ dict は copy せず
    参照のまま載せる。ケースは指定キーのみを読み、レコード別 `"tools"` キー等は読まない。

    Args:
        cases: ケース列（`DpoCase` / plain dict の混在可）。
        tools: `input` 内へ透過するツール定義（plain dict / FunctionTool 相当の混在可）。
        parallel_tool_calls: `input` 内へ透過する並列ツール呼び出しの可否。
        input_key: plain dict ケースの入力キー名。
        preferred_key: plain dict ケースの preferred 出力キー名。
        non_preferred_key: plain dict ケースの non-preferred 出力キー名。
        skip_missing: True のとき、必須フィールド欠落・境界不備のケースを除外して
            `DatasetBuildResult.skipped` に件数を報告する。

    Returns:
        変換結果（`records` / `skipped`）。

    Raises:
        FineTuneError: `skip_missing=False` でデータ不備がある場合、`cases` が単一の dict /
            `str` / `bytes` の場合、`tools=` に不正要素がある場合（後 2 者は `skip_missing` の
            対象外）。
    """
    _reject_non_sequence_cases(cases)
    input_fields = _tool_fields(tools, parallel_tool_calls)
    records: list[dict[str, Any]] = []
    skipped = 0
    for index, case in enumerate(cases, start=1):
        try:
            messages = _to_messages(_case_value(case, input_key), input_key, wrap_role="user")
            preferred = _to_messages(
                _case_value(case, preferred_key), preferred_key, wrap_role="assistant"
            )
            non_preferred = _to_messages(
                _case_value(case, non_preferred_key), non_preferred_key, wrap_role="assistant"
            )
        except _CaseDataError as exc:
            if skip_missing:
                skipped += 1
                continue
            raise _case_error(index, exc) from exc
        records.append(
            {
                "input": {"messages": messages, **input_fields},
                "preferred_output": preferred,
                "non_preferred_output": non_preferred,
            }
        )
    return DatasetBuildResult(records=tuple(records), skipped=skipped)


def _validate_message(
    message: Any, label: str, *, check_weight: bool, require_assistant: bool = False
) -> list[str]:
    """メッセージ 1 件を検証して違反理由の列を返す。

    既知キー（`role` / `content` / `tool_calls` / `tool_call_id` / `weight`）にのみ規則を
    適用し、未知キーは違反にしない（プラットフォームのフィールド追加で誤検知しないため）。

    Args:
        message: 検証対象のメッセージ。
        label: 違反理由に載せる位置表記。
        check_weight: `weight` の role / 値制約を検証するか（SFT のみ True）。
        require_assistant: role が `"assistant"` であることを要求するか（preference 出力）。

    Returns:
        違反理由の列（違反が無ければ空）。
    """
    if not isinstance(message, dict):
        return [f"{label} が dict でない"]

    reasons: list[str] = []
    role = message.get("role")
    if not isinstance(role, str):
        return [
            f"{label}.role が文字列でない: {type(role).__name__} 型（合法値: {_LEGAL_ROLE_HINT}）"
        ]
    if role not in _LEGAL_ROLES:
        return [f"{label}.role が不正: {role!r}（合法値: {_LEGAL_ROLE_HINT}）"]
    if require_assistant and role != "assistant":
        reasons.append(f"{label}.role は 'assistant' のみ許容: {role!r}")

    has_content = message.get("content") is not None
    has_tool_calls = message.get("tool_calls") is not None
    if has_tool_calls and role != "assistant":
        reasons.append(f"{label}.tool_calls は assistant メッセージのみ許容（role: {role!r}）")
    if not has_content and not has_tool_calls:
        reasons.append(f"{label} に content / tool_calls のいずれも存在しない")
    if has_content:
        content = message["content"]
        if isinstance(content, list):
            if not content:
                reasons.append(f"{label}.content が空リストである")
        elif not isinstance(content, str):
            reasons.append(
                f"{label}.content が不正な型: {type(content).__name__}"
                "（文字列または parts 配列のみ合法）"
            )
    if role == "tool":
        if not has_content:
            reasons.append(f"{label} は role 'tool' のため content が必須")
        if "tool_call_id" not in message:
            reasons.append(f"{label} は role 'tool' のため tool_call_id が必須")
    if check_weight and "weight" in message:
        weight = message["weight"]
        if role != "assistant":
            reasons.append(f"{label}.weight は assistant メッセージのみ許容（role: {role!r}）")
        elif isinstance(weight, bool) or not isinstance(weight, int) or weight not in (0, 1):
            reasons.append(f"{label}.weight が不正: {weight!r}（整数 0 / 1 のみ合法）")
    return reasons


def _validate_messages(messages: Any, label: str, *, check_weight: bool) -> list[str]:
    """messages リストを検証して違反理由の列を返す。

    Args:
        messages: 検証対象の messages。
        label: 違反理由に載せる位置表記。
        check_weight: `weight` の role / 値制約を検証するか。

    Returns:
        違反理由の列。
    """
    if not isinstance(messages, list) or not messages:
        return [f"{label} が非空のリストでない"]
    reasons: list[str] = []
    for position, message in enumerate(messages):
        reasons.extend(
            _validate_message(message, f"{label}[{position}]", check_weight=check_weight)
        )
    return reasons


def _validate_sft_record(record: Any) -> list[str]:
    """SFT レコード 1 件を検証して違反理由の列を返す。

    メッセージ単位の規則に加えて、レコード単位の必須要件として「`messages` に role
    `"assistant"` のメッセージが 1 件以上あること」を課す（OpenAI 公式 cookbook
    `Chat_finetuning_data_prep.ipynb` の検証項目 `example_missing_assistant_message`
    に準拠）。これは合法キー・合法値の集合を広げる変更ではなくレコード単位の必須要件で
    あり、メッセージ単位の違反とは独立に報告する（メッセージ単位の違反の後、末尾へ付す）。
    ただし `messages` 自体が無い / 非リスト / 空リストの経路では重ねて報告しない（構造が
    壊れたレコードに assistant の有無を重ねない）。`weight` の値は判定に用いない
    （全 assistant が `weight: 0` でも欠落とはしない・公式検証に該当項目が無いため）。
    DPO では `input.messages` が prompt であり assistant は `preferred_output` /
    `non_preferred_output` 側で必須化されるため、本要件は適用しない。

    Args:
        record: 検証対象のレコード。

    Returns:
        違反理由の列。
    """
    if not isinstance(record, dict):
        return ["レコードが JSON オブジェクトでない"]
    if "messages" not in record:
        return ["必須キー 'messages' が存在しない"]
    messages = record["messages"]
    reasons = _validate_messages(messages, "messages", check_weight=True)
    if not isinstance(messages, list) or not messages:
        return reasons
    if not any(
        isinstance(message, dict) and message.get("role") == "assistant" for message in messages
    ):
        reasons.append("messages に role 'assistant' のメッセージが 1 件も存在しない")
    return reasons


def _validate_dpo_record(record: Any) -> list[str]:
    """DPO（preference）レコード 1 件を検証して違反理由の列を返す。

    `weight` は SFT のみ検証対象のため、DPO では role / 値制約を適用しない。

    Args:
        record: 検証対象のレコード。

    Returns:
        違反理由の列。
    """
    if not isinstance(record, dict):
        return ["レコードが JSON オブジェクトでない"]

    reasons: list[str] = []
    record_input = record.get("input")
    if not isinstance(record_input, dict):
        reasons.append(f"'input' が JSON オブジェクトでない: {type(record_input).__name__}")
    elif "messages" not in record_input:
        reasons.append("必須キー 'input.messages' が存在しない")
    else:
        reasons.extend(
            _validate_messages(record_input["messages"], "input.messages", check_weight=False)
        )
    for key in ("preferred_output", "non_preferred_output"):
        outputs = record.get(key)
        if not isinstance(outputs, list) or not outputs:
            reasons.append(f"必須キー '{key}' が非空の assistant メッセージ配列でない")
            continue
        for position, message in enumerate(outputs):
            reasons.extend(
                _validate_message(
                    message, f"{key}[{position}]", check_weight=False, require_assistant=True
                )
            )
    return reasons


def _iter_source(source: str | Path | Iterable[Any]) -> Iterator[tuple[int, Any, str | None]]:
    """検証対象を `(line, record, parse_error)` のジェネレータへ正規化する。

    ファイル source は全量を読み込まず行単位で逐次 yield する（BOM は `utf-8-sig` で
    除去する・空行はスキップし行番号は原文の 1 始まりを保つ）。データセット全量を
    メモリへ載せないため、呼び出し側は逐次消費すること。ファイルは `with` 節で開くため、
    ジェネレータが途中で破棄された場合も close 時にハンドルは解放される。

    Args:
        source: JSONL ファイルパス（str / Path）またはレコード（dict）の列。

    Yields:
        `(line, record, parse_error)`。`parse_error` が非 None のとき `record` は None。

    Raises:
        OSError: ファイルを読めない場合（fail-closed・呼び出し側へ伝播）。
        UnicodeDecodeError: ファイルが UTF-8 として解釈できない場合。
    """
    if not isinstance(source, (str, Path)):
        for position, record in enumerate(source, start=1):
            yield (position, record, None)
        return

    with Path(source).open(encoding="utf-8-sig") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except (json.JSONDecodeError, RecursionError):
                yield (line_number, None, "JSON として解析できない（入れ子が深すぎる等）")
                continue
            yield (line_number, parsed, None)


def validate_dataset(
    source: str | Path | Iterable[Any],
    *,
    method: str = "sft",
    raise_on_invalid: bool = False,
) -> DatasetValidationReport:
    """SFT / DPO データセットを OpenAI 公式形式に照らして検証する（ローカル読み取りのみ）。

    メッセージの既知キー（`role` / `content` / `tool_calls` / `tool_call_id` / `weight`）に
    規則を適用し、未知キー・レコードレベルの未知フィールドは違反にしない。ツール定義や
    content parts の内部構造は解釈しない。`weight` の role / 値制約は `method="sft"` のみ
    適用する。違反ゼロのときのみ `ok=True`（fail-closed）。

    Args:
        source: JSONL ファイルパス（str / Path）またはレコード（dict）の列。単一の dict を
            渡すことはできない（キー文字列の列として誤読しないよう明示エラーにする）。
        method: 検証する形式（`"sft"` または `"dpo"`）。他の値はエラーにする。
        raise_on_invalid: True のとき不合格で `FineTuneError` を送出する（既定は返却のみ）。

    Returns:
        検証レポート。`DatasetViolation.line` は source がファイルパスなら 1 始まりの行番号、
        dict 列なら 1 始まりの要素位置を表す。

    Raises:
        FineTuneError: `method` が `"sft"` / `"dpo"` 以外の場合、`source` が単一の dict の
            場合、`raise_on_invalid=True` かつ不合格の場合（最後のみ `report` を載せる）。
        OSError: ファイル source を読めない場合。
        UnicodeDecodeError: ファイル source が UTF-8 として解釈できない場合。
    """
    if method not in ("sft", "dpo"):
        raise FineTuneError(
            FineTuneFailureKind.VALIDATION_FAILED,
            f"method が不正: {method!r}（'sft' / 'dpo' のみ）",
        )
    if isinstance(source, dict):
        raise FineTuneError(
            FineTuneFailureKind.VALIDATION_FAILED,
            "source が単一の dict である（レコードの列またはファイルパスを渡す）",
        )
    validate_record = _validate_dpo_record if method == "dpo" else _validate_sft_record
    violations: list[DatasetViolation] = []
    checked = 0
    for line, record, parse_error in _iter_source(source):
        checked += 1
        if parse_error is not None:
            violations.append(DatasetViolation(line=line, reason=parse_error))
            continue
        violations.extend(
            DatasetViolation(line=line, reason=reason) for reason in validate_record(record)
        )
    report = DatasetValidationReport(
        ok=not violations, checked=checked, violations=tuple(violations)
    )
    if raise_on_invalid and not report.ok:
        raise FineTuneError(
            FineTuneFailureKind.VALIDATION_FAILED,
            f"データセット検証に失敗しました（違反 {len(report.violations)} 件 / "
            f"検証 {report.checked} 件）",
            report=report,
        )
    return report
