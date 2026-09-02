"""Fine-Tuning データセットの変換・検証ヘルパ（純データ層・SDK 非接触）。

`to_sft_dataset` / `to_dpo_dataset` は EvalCase / OptimizeCase / `DpoCase` / plain dict の列を
OpenAI 公式 SFT / DPO（preference）形式のレコード列へ変換し、`validate_dataset` は持ち込み
JSONL（またはレコード列）を同形式に照らして検証する。`screen_tool_roundtrips` は submit 前の形式
ゲートとして、メッセージ間の順序制約（ツール往復の並び）だけを検査する。
`agents` / `openai` を import せず、
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
    DatasetPartition,
    DatasetRejection,
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


def _iter_source(
    source: str | Path | Iterable[Any],
) -> Iterator[tuple[int, Any, str | None, str | None]]:
    """検証対象を `(line, record, parse_error, raw)` のジェネレータへ正規化する。

    ファイル source は全量を読み込まず行単位で逐次 yield する（BOM は `utf-8-sig` で
    除去する・空行はスキップし行番号は原文の 1 始まりを保つ）。データセット全量を
    メモリへ載せないため、呼び出し側は逐次消費すること。ファイルは `with` 節で開くため、
    ジェネレータが途中で破棄された場合も close 時にハンドルは解放される。

    Args:
        source: JSONL ファイルパス（str / Path）またはレコード（dict）の列。

    Yields:
        `(line, record, parse_error, raw)`。`parse_error` が非 None のとき `record` は None で、
        `raw` が当該行の原文（改行を除く）を持つ。解析できた行とレコード列の要素では `raw` は
        None（`record` があれば原文は不要であり、両方を持つとメモリ使用量が二重になる）。

    Raises:
        OSError: ファイルを読めない場合（fail-closed・呼び出し側へ伝播）。
        UnicodeDecodeError: ファイルが UTF-8 として解釈できない場合。
    """
    if not isinstance(source, (str, Path)):
        for position, record in enumerate(source, start=1):
            yield (position, record, None, None)
        return

    with Path(source).open(encoding="utf-8-sig") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except (json.JSONDecodeError, RecursionError):
                yield (
                    line_number,
                    None,
                    "JSON として解析できない（入れ子が深すぎる等）",
                    line.rstrip("\n"),
                )
                continue
            yield (line_number, parsed, None, None)


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

    本関数だけでは投入可否を判定できない（メッセージ間の順序制約は `screen_tool_roundtrips` の
    責務）。両方をまとめて適用し合格・不合格へ仕分けるなら `partition_dataset` を使う。

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
    for line, record, parse_error, _raw in _iter_source(source):
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


def _tool_group_reasons(requested: set[str], answered: set[str], label: str) -> list[str]:
    """tool_calls 群と直後の role `"tool"` 群の集合比較から違反理由の列を返す。

    Args:
        requested: assistant の `tool_calls` が持つ id の集合。
        answered: 直後に連続する role `"tool"` メッセージの `tool_call_id` の集合。
        label: 違反理由の先頭へ付す位置表記（群を開いた assistant の位置）。

    Returns:
        違反理由の列（過不足が無ければ空）。
    """
    if requested == answered:
        return []
    details: list[str] = []
    missing = sorted(requested - answered)
    extra = sorted(answered - requested)
    if missing:
        details.append(f"応答が無い call id: {', '.join(missing)}")
    if extra:
        details.append(f"呼び出しに無い call id: {', '.join(extra)}")
    return [
        f"{label}: tool_calls 付き assistant の直後に続く role 'tool' 群が tool_calls の"
        f" id 集合と一致しない（{' / '.join(details)}）"
    ]


def _screen_messages(messages: Any, label: str) -> list[str]:
    """messages のツール往復の順序制約を検査して違反理由の列を返す。

    メッセージ単位の合法性（role / content / 必須キー等）は `validate_dataset` の責務のため
    判定しない。`messages` が非リストの場合や非 dict 要素は違反にせず素通しする（構造違反の
    二重報告を避ける）。非 dict 要素の位置で群の連続性は途切れる扱いにする。

    判定は id の集合比較で行うため、群内に同じ `tool_call_id` が重複しても判定に影響しない
    （重複そのものの是非はメッセージ単位の問題であり本関数の責務外）。集合へ入れるのは str の
    id のみで、`tool_calls` の要素が文字列 `id` を欠く場合は集合比較の対象にできない（後続
    tool が無ければ空集合同士で一致してしまう）ため、その時点で件数付きの違反として報告する。
    `tool_calls` 自体がリストでない場合も対応を検証できないため違反として報告する。
    role `"tool"` 側の非 str・キー欠落の `tool_call_id` は集合へ入らず、群の不一致として現れる。

    末尾に開いたままの群は違反にしない。対象 messages の末尾にある `tool_calls` 付き assistant は
    「ツール呼び出しそのものを学習させる」SFT レコードの学習ターゲット本体であり、応答が続かない
    のが正常だからである（この区別を欠くと `to_sft_dataset` の正当な生成物を不合格にする）。

    違反理由には `messages[N]:` 形式で messages 内の位置を前置する（`validate_dataset` と
    同書式）。群の不一致は、群を開いた `tool_calls` 付き assistant の位置に紐づける。

    Args:
        messages: 検査対象の messages（非リストなら判定しない）。
        label: 位置表記の見出し（`"messages"` / `"input.messages"`）。

    Returns:
        違反理由の列（違反が無ければ空）。
    """
    if not isinstance(messages, list):
        return []
    reasons: list[str] = []
    pending: tuple[set[str], set[str], str] | None = None
    for position, message in enumerate(messages):
        at = f"{label}[{position}]"
        if isinstance(message, dict) and message.get("role") == "tool":
            if pending is None:
                call_id = message.get("tool_call_id")
                reasons.append(
                    f"{at}: role 'tool' メッセージが直前の tool_calls 群に属さない"
                    f"（tool_call_id: {call_id!r}）"
                )
                continue
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str):
                pending[1].add(tool_call_id)
            continue
        if pending is not None:
            reasons.extend(_tool_group_reasons(*pending))
            pending = None
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if "tool_calls" not in message:
            continue
        tool_calls = message["tool_calls"]
        if not isinstance(tool_calls, list):
            # `validate_dataset` はキーの存在しか見ない（FR-3 は内部構造を解釈しない）ため、
            # ここで素通しすると不正なレコードが両ゲートを通過する。
            reasons.append(
                f"{at}: tool_calls がリストでない: {type(tool_calls).__name__}"
                "（ツール呼び出しの対応を検証できない）"
            )
            continue
        identified = [
            call
            for call in tool_calls
            if isinstance(call, dict) and isinstance(call.get("id"), str)
        ]
        requested = {call["id"] for call in identified}
        # 重複 id は判定に影響しないため、集合の要素数ではなく要素の件数差で数える。
        unidentified = len(tool_calls) - len(identified)
        if unidentified > 0:
            # id を欠く（または非 str の）呼び出しは要求集合へ入らないため、集合比較だけ
            # では「応答が無い」ことすら判定できない（後続 tool が無ければ空集合同士で
            # 一致して合格する）。往復の対応を検証できない時点で違反として報告する。
            reasons.append(
                f"{at}: tool_calls に文字列 'id' を持たない呼び出しがある"
                f"（{unidentified} 件・対応する role 'tool' メッセージを特定できない）"
            )
        pending = (requested, set(), at)
    # 末尾に開いたままの群は学習ターゲット本体（ツール呼び出しそのものを学習させる形）であり、
    # 応答が続かないのが正常なため違反にしない。文脈途中の未応答だけが規則 (1) の対象である。
    return reasons


def _screen_record(record: Any, *, method: str) -> list[str]:
    """レコード 1 件のツール往復を検査して違反理由の列を返す。

    Args:
        record: 検査対象のレコード（非 dict なら判定しない）。
        method: 検査する形式（`"sft"` は `messages`、`"dpo"` は `input.messages` を見る）。

    Returns:
        違反理由の列。
    """
    if not isinstance(record, dict):
        return []
    if method == "dpo":
        record_input = record.get("input")
        if not isinstance(record_input, dict):
            return []
        return _screen_messages(record_input.get("messages"), "input.messages")
    return _screen_messages(record.get("messages"), "messages")


def screen_tool_roundtrips(
    source: str | Path | Iterable[Any],
    *,
    method: str = "sft",
    raise_on_invalid: bool = False,
) -> DatasetValidationReport:
    """データセットのツール往復の並びを検査する（submit 前の形式ゲート・ローカル読み取りのみ）。

    判定対象は「メッセージ間の順序制約」のみで、メッセージ単位の合法性は
    `validate_dataset` の責務として二重報告しない。規則は 2 つある。

    - `tool_calls` を持つ assistant メッセージの直後に連続する role `"tool"` メッセージ群の
      `tool_call_id` 集合が、当該 assistant の `tool_calls` の id 集合と一致すること
      （過不足なし・群内の順序は問わない）。ただし**末尾**の `tool_calls` 付き assistant には
      適用しない（ツール呼び出しそのものを学習させる SFT の学習ターゲット本体であり、応答が
      続かないのが正常なため）
    - いずれの群にも属さない role `"tool"` メッセージが存在しないこと

    `messages` が取り出せない / 非リストなど構造が壊れたレコードは違反を報告せず素通しする
    （構造違反は `validate_dataset` が報告する）。違反ゼロのときのみ `ok=True`（fail-closed）。

    判定の準拠先は推論時 API の順序要求であり、FT のファイル検証が同じ並びを拒否するかは
    未確定である（ADR 0036 の Context）。ただし応答のない `tool_calls` を含む文脈を学習させる
    こと自体が学習データの誤りのため、本関数の価値はファイル検証の挙動に依存しない。

    本関数だけでは投入可否を判定できない（メッセージ単位の合法性は `validate_dataset` の
    責務）。両方をまとめて適用し合格・不合格へ仕分けるなら `partition_dataset` を使う。

    Args:
        source: JSONL ファイルパス（str / Path）またはレコード（dict）の列。単一の dict を
            渡すことはできない（キー文字列の列として誤読しないよう明示エラーにする）。
        method: 検査する形式（`"sft"` は `messages`、`"dpo"` は `input.messages` を対象に
            する。DPO の `preferred_output` / `non_preferred_output` は対象外）。他の値は
            エラーにする。
        raise_on_invalid: True のとき不合格で `FineTuneError` を送出する（既定は返却のみ）。

    Returns:
        検査レポート。`DatasetViolation.line` は source がファイルパスなら 1 始まりの行番号、
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
    violations: list[DatasetViolation] = []
    checked = 0
    for line, record, parse_error, _raw in _iter_source(source):
        checked += 1
        if parse_error is not None:
            violations.append(DatasetViolation(line=line, reason=parse_error))
            continue
        violations.extend(
            DatasetViolation(line=line, reason=reason)
            for reason in _screen_record(record, method=method)
        )
    report = DatasetValidationReport(
        ok=not violations, checked=checked, violations=tuple(violations)
    )
    if raise_on_invalid and not report.ok:
        raise FineTuneError(
            FineTuneFailureKind.VALIDATION_FAILED,
            f"ツール往復のスクリーニングに失敗しました（違反 {len(report.violations)} 件 / "
            f"検査 {report.checked} 件）",
            report=report,
        )
    return report


def partition_dataset(
    source: str | Path | Iterable[Any],
    *,
    method: str = "sft",
) -> DatasetPartition:
    """投入できるレコードとできないレコードへ仕分ける（submit 前の仕分けヘルパ）。

    各レコードへ `validate_dataset`（メッセージ単位の合法性）と `screen_tool_roundtrips`
    （メッセージ間の順序制約）の両方を適用し、どちらにも違反しないレコードだけを `passed` へ
    入れる。判定規則は両関数へ委譲するため本関数は新しい規則を持たない。JSON として解析できない
    行も不合格として扱う（合格側へ混ぜない）。

    `passed` は `DatasetBuildResult` で返すため、`submit_job(train=...)` と `save(path)` へ
    詰め替えなしで渡せる。不合格側は元レコードと理由を `DatasetRejection` にまとめて返すので、
    レポートと元データを位置で突き合わせる必要がない。

    仕分けの性質上、合格・不合格の双方をメモリへ保持する（`_iter_source` の逐次読みの利点は
    本関数では活きない）。逐次処理が必要な規模では、レコードを保持せず違反だけを溜める
    `validate_dataset` / `screen_tool_roundtrips` を直接使うこと。

    Args:
        source: JSONL ファイルパス（str / Path）またはレコード（dict）の列。単一の dict を
            渡すことはできない（キー文字列の列として誤読しないよう明示エラーにする）。
        method: 仕分ける形式（`"sft"` または `"dpo"`）。他の値はエラーにする。

    Returns:
        仕分け結果。`DatasetRejection.line` は source がファイルパスなら 1 始まりの行番号、
        レコード列なら 1 始まりの要素位置を表す。

    Raises:
        FineTuneError: `method` が `"sft"` / `"dpo"` 以外の場合、`source` が単一の dict の
            場合（`VALIDATION_FAILED`）。仕分け自体は例外を送出せず、不合格は返却値で表す。
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
    passed: list[dict[str, Any]] = []
    rejected: list[DatasetRejection] = []
    checked = 0
    for line, record, parse_error, raw in _iter_source(source):
        checked += 1
        if parse_error is not None:
            rejected.append(
                DatasetRejection(line=line, record=None, reasons=(parse_error,), raw=raw)
            )
            continue
        reasons = [*validate_record(record), *_screen_record(record, method=method)]
        if reasons:
            rejected.append(DatasetRejection(line=line, record=record, reasons=tuple(reasons)))
            continue
        passed.append(record)
    return DatasetPartition(
        passed=DatasetBuildResult(records=tuple(passed), skipped=len(rejected)),
        rejected=tuple(rejected),
        checked=checked,
    )
