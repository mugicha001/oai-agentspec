"""DPO 雛形の書き出し・取り込みヘルパ（純データ + ローカルファイル I/O・FR-12）。

`dpo_dataset_from_session` の雛形モードが返す記入用ケース列を CSV / JSONL へ書き出し
（`save_dpo_draft`）、人手記入後に最終 preference データセットへ取り込む
（`finalize_dpo_draft`）記入ワークフローを担う。`Session` にもネットワークにも触れず、
標準ライブラリの `csv` / `json` のみを使う（`_adapters` へ依存しない）。

設計判断（列構成・列名ベース読み取り・両欄空 skip / 片欄エラー・エンコーディング・
strip の適用範囲）は ADR 0034（`docs/adr/0034-session-tool-roundtrip-and-dpo-draft.md`）
Decision 6-8・10 を参照。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from .dataset import to_dpo_dataset
from .types import DatasetBuildResult, FineTuneError, FineTuneFailureKind

if TYPE_CHECKING:
    from collections.abc import Iterable

# 記入用 CSV の列（参照 3 列 + 記入 2 列 + 機械用 1 列で 1 ファイル自己完結）。
_CASE_INDEX_COLUMN: Final[str] = "case_index"
_CONTEXT_COLUMN: Final[str] = "context"
_RESPONSE_COLUMN: Final[str] = "response"
_PREFERRED_COLUMN: Final[str] = "preferred_output"
_NON_PREFERRED_COLUMN: Final[str] = "non_preferred_output"
_INPUT_JSON_COLUMN: Final[str] = "input_json"

_CSV_COLUMNS: Final[tuple[str, ...]] = (
    _CASE_INDEX_COLUMN,
    _CONTEXT_COLUMN,
    _RESPONSE_COLUMN,
    _PREFERRED_COLUMN,
    _NON_PREFERRED_COLUMN,
    _INPUT_JSON_COLUMN,
)

# 取り込みに必須の列（参照列は無視する）。
_REQUIRED_CSV_COLUMNS: Final[tuple[str, ...]] = (
    _INPUT_JSON_COLUMN,
    _PREFERRED_COLUMN,
    _NON_PREFERRED_COLUMN,
)

# 書き出す記入用ケースが持つ必須キー（`response` 参照欄を欠く雛形を書かせない・FR-12）。
_SAVE_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "input",
    _PREFERRED_COLUMN,
    _NON_PREFERRED_COLUMN,
    _RESPONSE_COLUMN,
)

# 取り込む記入済みケースが持つ必須キー（参照欄は取り込みに使わないため要求しない）。
_FINALIZE_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "input",
    _PREFERRED_COLUMN,
    _NON_PREFERRED_COLUMN,
)

# 拡張子ごとの書式（lib は既定形式を発明せず、拡張子が不明なら失敗する）。
_CSV_SUFFIX: Final[str] = ".csv"
_JSONL_SUFFIX: Final[str] = ".jsonl"

# CSV は日本語環境の表計算ソフトで文字化けしないよう BOM 付きで読み書きし、JSONL は BOM なし
# で書く。読み取りは BOM 付き / なしの双方を受理する `utf-8-sig` でデコードする
# （ADR 0034 Decision 7）。
_CSV_ENCODING: Final[str] = "utf-8-sig"


def _validation_error(message: str) -> FineTuneError:
    """`VALIDATION_FAILED` の構造化エラーを組み立てる。

    Args:
        message: 人間可読のエラーメッセージ。

    Returns:
        `FineTuneError`（kind は `VALIDATION_FAILED`）。
    """
    return FineTuneError(FineTuneFailureKind.VALIDATION_FAILED, message)


def _resolve_suffix(path: Path) -> str:
    """パスの拡張子から書式を判別する。

    Args:
        path: 対象ファイルパス。

    Returns:
        `".csv"` または `".jsonl"`。

    Raises:
        FineTuneError: 拡張子が `.csv` / `.jsonl` のいずれでもない場合（`CONFIG_MISSING`）。
    """
    suffix = path.suffix.lower()
    if suffix not in (_CSV_SUFFIX, _JSONL_SUFFIX):
        raise FineTuneError(
            FineTuneFailureKind.CONFIG_MISSING,
            f"対応していない拡張子です: {path.suffix!r}"
            f"（{_CSV_SUFFIX} / {_JSONL_SUFFIX} のいずれかを指定すること）",
        )
    return suffix


def _tool_call_name(call: dict[str, Any]) -> str:
    """tool_calls 要素からツール名を取り出す（参照列の整形用・非可逆）。

    `function` が dict でない不正な要素（利用者が `records` を加工した場合等）でも参照列の
    整形で例外にせず、代替表現へフォールバックする。

    Args:
        call: tool_calls の要素（dict）。

    Returns:
        `function.name` の文字列。取り出せない場合は要素全体の文字列表現。
    """
    function = call.get("function")
    if isinstance(function, dict):
        return str(function.get("name"))
    return str(call)


def _format_context(messages: Any) -> str:
    """累積文脈を人が読むための 1 セル文字列へ整形する（非可逆・参照専用）。

    復元は `input_json` 列のみが担うため、本列は読みやすさを優先した非可逆な表現である。

    Args:
        messages: 累積文脈（messages のリストを想定）。

    Returns:
        `"role: 本文"` を改行で連ねた文字列（tool_calls はツール名を示す）。
    """
    if not isinstance(messages, list):
        return str(messages)
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            lines.append(str(message))
            continue
        role = message.get("role", "")
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            names = ", ".join(
                _tool_call_name(call) for call in tool_calls if isinstance(call, dict)
            )
            lines.append(f"{role}[tool_calls]: {names}")
            continue
        lines.append(f"{role}: {message.get('content', '')}")
    return "\n".join(lines)


def _draft_cases(
    source: DatasetBuildResult | Iterable[dict[str, Any]],
    *,
    required: tuple[str, ...],
) -> list[dict[str, Any]]:
    """ケース列を取り出して全件検証する（書き出し前の部分書き込み防止・取り込み前の検証）。

    要求する必須キーは経路ごとに異なる。書き出し（`save_dpo_draft`）は参照欄
    `response` を含む `_SAVE_REQUIRED_KEYS`、取り込み（`finalize_dpo_draft`）は参照欄を
    使わないため `_FINALIZE_REQUIRED_KEYS` を渡す。

    Args:
        source: 雛形モードの `DatasetBuildResult` またはケースの列。
        required: 各ケースへ要求する必須キー（keyword-only）。

    Returns:
        検証済みのケースのリスト。

    Raises:
        FineTuneError: source が単一の dict / 文字列 / バイト列の場合、要素が dict でない
            場合、`required` のキーを欠く場合（`VALIDATION_FAILED`）。
    """
    records: Any = source.records if isinstance(source, DatasetBuildResult) else source
    if isinstance(records, (dict, str, bytes)):
        raise _validation_error(
            "source が単一の dict / 文字列である（記入用ケースの列を渡す。1 件だけの場合も "
            "[case] のように列へ包むこと）"
        )
    cases: list[dict[str, Any]] = []
    for index, case in enumerate(records, start=1):
        if not isinstance(case, dict):
            raise _validation_error(f"ケース {index}: 記入用ケースが dict でない")
        missing = [key for key in required if key not in case]
        if missing:
            raise _validation_error(
                f"ケース {index}: 記入用ケースに必須キーがありません: {', '.join(missing)}"
            )
        cases.append(case)
    return cases


def _check_csv_cell_sizes(rows: list[dict[str, Any]]) -> None:
    """書き出す各セルが `csv.field_size_limit()` に収まるかを事前検査する。

    `csv` の書き出しはセルサイズ無制限だが読み取りは `csv.field_size_limit()` で制限される
    ため、無検査だと lib が書いた雛形を lib が読めない非対称が生じ、人手記入後に発覚する。
    書き出し前に失敗させて記入の手戻りを防ぐ（上限はプロセスグローバルのため lib からは
    変更せず、実行時の現在値を読んで判定する）。

    Args:
        rows: 書き出す CSV 行（列名 -> 値）のリスト（行順はケース順）。

    Raises:
        FineTuneError: いずれかのセルが現在の上限を超える場合（`VALIDATION_FAILED`）。
    """
    limit = csv.field_size_limit()
    for index, row in enumerate(rows, start=1):
        for column, value in row.items():
            if isinstance(value, str) and len(value) > limit:
                raise _validation_error(
                    f"ケース {index}: {column} 列のセルが CSV の読み取り上限を超えます"
                    f"（{len(value)} 文字 > 上限 {limit} 文字）。このまま書き出すと"
                    " finalize_dpo_draft で読み取れないため中止しました。"
                    " 利用者側で csv.field_size_limit(<大きい値>) を呼ぶか、"
                    f"{_JSONL_SUFFIX} で書き出すこと"
                )


def save_dpo_draft(
    source: DatasetBuildResult | Iterable[dict[str, Any]],
    path: str | Path,
) -> None:
    """記入用ケース列を CSV / JSONL の雛形ファイルへ書き出す（FR-12）。

    書式は拡張子で切り替える。`.csv` は `case_index` / `context` / `response`（参照列）+
    `preferred_output` / `non_preferred_output`（記入列）+ `input_json`（機械用の文脈復元列）
    の 6 列で 1 ファイル自己完結し、`utf-8-sig`（BOM 付き）で書く。`.jsonl` は記入用ケースを
    1 行 1 件でそのまま書く（`DatasetBuildResult.save()` と同内容）。

    書き出し前に全件を検証し、必須キー（`input` / `preferred_output` /
    `non_preferred_output` / `response`）を欠く要素や JSON へ直列化できない要素が
    あれば、ファイルを作らずに失敗する（壊れた雛形・実応答の参照欄が空の雛形へ記入させない
    ため・FR-12 / ADR 0034 Decision 6）。`.csv` では各セルが現在の `csv.field_size_limit()`
    に収まるかも事前検査し、超える場合はファイルを作らずに失敗する（読み取れない雛形へ人手で
    記入させないため。上限はプロセスグローバルのため lib は書き換えない）。

    Args:
        source: 雛形モード（`dpo_dataset_from_session(session)`）の `DatasetBuildResult`
            または同形の記入用ケースの列。
        path: 書き出し先パス（拡張子は `.csv` / `.jsonl` のいずれか）。

    Raises:
        FineTuneError: 拡張子が `.csv` / `.jsonl` でない場合（`CONFIG_MISSING`）。source が
            ケースの列でない / 必須キー（`input` / `preferred_output` /
            `non_preferred_output` / `response`）を欠く要素を含む / `input`（`.jsonl` では
            ケース全体）を JSON へ直列化できない場合、`.csv` でいずれかのセルが
            `csv.field_size_limit()` を超える場合（`VALIDATION_FAILED`）。
        OSError: 書込先が書込不能 / 不正な場合（fail-closed・呼び出し側へ伝播）。

    Note:
        既存ファイルは上書きする（追記しない）。シンボリックリンクはリンク先へ追従し、
        作成されるファイルのパーミッションはプロセスの umask に依存する。

        ツール出力を含む会話文脈がローカルの平文ファイルへ書かれる。機密情報・個人情報の
        除去は利用者の責務で、lib は自動マスキングを内蔵しない（雛形モードには
        `case_transform` 相当の hook が無いため、必要なら本関数へ渡す前に `records` を
        加工すること・NFR-5）。

        雛形 CSV の参照列（`context` / `response`）には会話ログ由来の文字列がそのまま入る。
        `=` / `+` / `-` / `@` などで始まる値は Excel / Google Sheets が数式として解釈しうる
        ため、信頼できないログを扱う場合はテキストとしてインポートすること。lib は値を
        改変しない（ADR 0034 Decision 10）。
    """
    target = Path(path)
    suffix = _resolve_suffix(target)
    cases = _draft_cases(source, required=_SAVE_REQUIRED_KEYS)

    if suffix == _JSONL_SUFFIX:
        # CSV 経路と同じく、直列化の失敗で部分的なファイルを残さないよう全行を組んでから書く。
        lines: list[str] = []
        for index, case in enumerate(cases, start=1):
            try:
                lines.append(json.dumps(case, ensure_ascii=False) + "\n")
            except (TypeError, ValueError) as exc:
                raise _validation_error(
                    f"ケース {index}: ケースを JSON へ直列化できません: {exc}"
                ) from exc
        target.write_text("".join(lines), encoding="utf-8")
        return

    # JSON 直列化の失敗で部分的に書かれたファイルが残らないよう、全行を組んでから開く。
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        try:
            input_json = json.dumps(case["input"], ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise _validation_error(
                f"ケース {index}: input を JSON へ直列化できません: {exc}"
            ) from exc
        rows.append(
            {
                _CASE_INDEX_COLUMN: index,
                _CONTEXT_COLUMN: _format_context(case["input"]),
                _RESPONSE_COLUMN: case[_RESPONSE_COLUMN],
                _PREFERRED_COLUMN: case[_PREFERRED_COLUMN],
                _NON_PREFERRED_COLUMN: case[_NON_PREFERRED_COLUMN],
                _INPUT_JSON_COLUMN: input_json,
            }
        )

    _check_csv_cell_sizes(rows)

    with target.open("w", encoding=_CSV_ENCODING, newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(_CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _cases_from_csv(path: Path) -> list[dict[str, Any]]:
    """記入済み CSV を列名ベース（列順非依存）で読み、ケース列へ復元する。

    参照列（`case_index` / `context` / `response`）は取り込みに使わない。

    必須 3 列（`input_json` / `preferred_output` / `non_preferred_output`）がすべて空の行は、
    記入も文脈も無い空行（表計算ソフトが出力する区切り文字だけの行を含む）として
    `finalize_dpo_draft` の未記入 skip 経路へ合流させる（`input_json` のパースは行わない）。
    `input_json` のみが空で記入欄に値がある行は、記入したのに文脈が失われている状態のため
    silent skip にせずエラーにする。

    Args:
        path: 記入済み CSV のパス。

    Returns:
        `{"input", "preferred_output", "non_preferred_output"}` を持つケースのリスト
        （必須 3 列がすべて空の行は `input` が None の未記入ケースとして返す）。

    Raises:
        FineTuneError: 必須列を欠く場合、`input_json` が JSON として不正な場合（必須 3 列が
            すべて空の行を除く）、CSV を読み取れない場合（セルが `csv.field_size_limit()` を
            超える等）、`utf-8-sig` としてデコードできない場合（`VALIDATION_FAILED`）。
        OSError: ファイルを読めない場合（呼び出し側へ伝播）。
    """
    rows: list[dict[str, Any]] = []
    with path.open(encoding=_CSV_ENCODING, newline="") as fp:
        reader = csv.DictReader(fp)
        try:
            fieldnames = reader.fieldnames or []
            missing = [column for column in _REQUIRED_CSV_COLUMNS if column not in fieldnames]
            if missing:
                raise _validation_error(
                    f"必須列がありません: {', '.join(missing)}"
                    f"（必須列: {', '.join(_REQUIRED_CSV_COLUMNS)}）"
                )
            for row in reader:
                rows.append(row)
        except csv.Error as exc:
            raise _validation_error(
                f"ケース {len(rows) + 1}: CSV を読み取れません: {exc}"
                f"（セルが大きすぎる可能性があります。現在の上限は "
                f"{csv.field_size_limit()} 文字で、引き上げるには利用者側で "
                f"csv.field_size_limit(<大きい値>) を呼ぶこと）"
            ) from exc
        except UnicodeDecodeError as exc:
            raise _validation_error(
                f"ケース {len(rows) + 1}: CSV を {_CSV_ENCODING} としてデコードできません: {exc}"
                "（日本語環境の表計算ソフトが cp932 等で保存した可能性があります。"
                "UTF-8（BOM 付き可）で保存し直してから取り込むこと。"
                "lib は文字化けの silent な取り込みを避けるため自動判定しません）"
            ) from exc

    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if all(_is_blank(row.get(column)) for column in _REQUIRED_CSV_COLUMNS):
            # 必須 3 列がすべて空の行（区切り文字だけの行等）は文脈の復元を試みず、
            # 未記入ケースとして skip 経路へ渡す（input_json のパースより前に判定する）。
            cases.append({"input": None, _PREFERRED_COLUMN: "", _NON_PREFERRED_COLUMN: ""})
            continue
        raw_input = row.get(_INPUT_JSON_COLUMN) or ""
        if not raw_input.strip():
            raise _validation_error(
                f"ケース {index}: {_INPUT_JSON_COLUMN} 列が空です"
                "（記入欄に値があるため未記入 skip の対象ではありません。"
                f"{_INPUT_JSON_COLUMN} 列を削除・上書きした可能性があります。"
                "雛形が書き出した文脈復元列をそのまま残して取り込むこと）"
            )
        try:
            context = json.loads(raw_input)
        except json.JSONDecodeError as exc:
            raise _validation_error(
                f"ケース {index}: {_INPUT_JSON_COLUMN} が JSON として不正です: {exc}"
            ) from exc
        cases.append(
            {
                "input": context,
                _PREFERRED_COLUMN: row.get(_PREFERRED_COLUMN) or "",
                _NON_PREFERRED_COLUMN: row.get(_NON_PREFERRED_COLUMN) or "",
            }
        )
    return cases


def _cases_from_jsonl(path: Path) -> list[dict[str, Any]]:
    """記入済み JSONL を読み、ケース列へ復元する。

    Args:
        path: 記入済み JSONL のパス。

    Returns:
        1 行 1 件の JSON をデコードしたケースのリスト（空行は読み飛ばす）。

    Raises:
        FineTuneError: いずれかの行が JSON として不正な場合（`VALIDATION_FAILED`）。
        OSError: ファイルを読めない場合（呼び出し側へ伝播）。
    """
    lines = [line for line in path.read_text(encoding=_CSV_ENCODING).splitlines() if line.strip()]
    cases: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise _validation_error(f"ケース {index}: JSON として不正な行です: {exc}") from exc
    return cases


def _is_blank(value: Any) -> bool:
    """記入欄が未記入（strip 後に空）かを判定する。

    strip は判定にのみ使い、採用値は非改変で委譲する（ADR 0034 Decision 10）。

    Args:
        value: 記入欄の値。

    Returns:
        None / 空文字 / 空白のみの文字列なら True。
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def finalize_dpo_draft(
    source: str | Path | Iterable[dict[str, Any]],
    *,
    tools: list[Any] | None = None,
    parallel_tool_calls: bool | None = None,
) -> DatasetBuildResult:
    """記入済みの雛形を最終 DPO（preference）データセットへ取り込む（FR-12）。

    source は CSV パス / JSONL パス（拡張子で判別・`save_dpo_draft` と対称）／メモリ上の
    記入済みケース列の 3 形を受ける。CSV は列名ベースで読むため列順の入れ替えに依存せず、
    参照列（`case_index` / `context` / `response`）は取り込みに影響しない。文脈は
    `input_json` 列（JSONL / ケース列では `input` キー）から復元する。

    記入判定は strip 後の空文字で行い、両記入欄が未記入のケースは skip して `skipped` へ
    計上する（全件未記入はエラーにせず `records=()` を正常返却する）。CSV では必須 3 列
    （`input_json` / 記入 2 列）がすべて空の行も同じ skip 経路へ合流する（表計算ソフトが
    出力する区切り文字だけの行を取り込み全体の失敗にしない）。`input_json` のみが空で記入欄
    に値がある行は、文脈を失った記入としてエラーにする。片欄のみ記入の
    ケースはケース位置つきの `VALIDATION_FAILED` で失敗する。採用した値は strip せず
    前後空白・改行を含めて**非改変**で `to_dpo_dataset(skip_missing=False)` へ委譲する
    （ADR 0034 Decision 8・10）。

    Args:
        source: 記入済み CSV / JSONL のパス、または記入済みケースの列。
        tools: 各レコードの `input` 内へ透過するツール定義（plain dict / FunctionTool 相当の
            混在可・SFT のレコード直下と透過位置が異なる）。利用者が供給した定義をそのまま
            `to_dpo_dataset` へ渡すのみで、lib は内容を解釈しない（写像・検証の規則は
            FR-1 / FR-2 と同一で委譲先に一元化する）。None（既定）なら `input` へキー自体を
            出さない。雛形は tools を持ち回らないため、ツール定義は本引数で供給する
            （ADR 0035）。
        parallel_tool_calls: 各レコードの `input` 内へ透過する並列ツール呼び出しの可否。
            `False` も指定として扱い透過する。None（既定）なら `input` へキー自体を出さない。

    Returns:
        変換結果（`records` / `skipped`）。`skipped` は本関数が数えた未記入 skip と委譲先の
        skip の合計で、雛形生成時（`dpo_dataset_from_session`）のカウントは含まない。

    Raises:
        FineTuneError: パスの拡張子が `.csv` / `.jsonl` でない場合（`CONFIG_MISSING`）。
            必須列 / 必須キーの欠落、`input_json` の不正 JSON（必須 3 列がすべて空の行を
            除く）、CSV を読み取れない場合（セルが `csv.field_size_limit()` を超える等）、
            CSV を `utf-8-sig` としてデコードできない場合（cp932 等で保存された CSV。
            UTF-8 で保存し直すこと）、片欄のみ記入のケースがある
            場合、および委譲先 `to_dpo_dataset` の検証違反（`tools=` の不正要素を含む。
            記入済みケースが 0 件でも委譲するため空結果の経路でも発火する・
            `VALIDATION_FAILED`）。委譲先
            エラーに載る `ケース {index}` は skip を除いた委譲リスト上の位置である。
        OSError: source のファイルを読めない場合（握り潰さず伝播する）。
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        suffix = _resolve_suffix(path)
        cases = (
            _cases_from_csv(path)
            if suffix == _CSV_SUFFIX
            else _draft_cases(_cases_from_jsonl(path), required=_FINALIZE_REQUIRED_KEYS)
        )
    else:
        cases = _draft_cases(source, required=_FINALIZE_REQUIRED_KEYS)

    skipped = 0
    filled: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        preferred = case.get(_PREFERRED_COLUMN)
        non_preferred = case.get(_NON_PREFERRED_COLUMN)
        preferred_blank = _is_blank(preferred)
        non_preferred_blank = _is_blank(non_preferred)
        if preferred_blank and non_preferred_blank:
            skipped += 1
            continue
        if preferred_blank or non_preferred_blank:
            blank_column = _PREFERRED_COLUMN if preferred_blank else _NON_PREFERRED_COLUMN
            raise _validation_error(
                f"ケース {index}: 記入欄が片方のみ埋まっています（未記入: {blank_column}）"
                "。両方を記入するか、両方を空にして skip すること"
            )
        filled.append(
            {
                "input": case.get("input"),
                _PREFERRED_COLUMN: preferred,
                _NON_PREFERRED_COLUMN: non_preferred,
            }
        )

    result = to_dpo_dataset(
        filled, skip_missing=False, tools=tools, parallel_tool_calls=parallel_tool_calls
    )
    return DatasetBuildResult(records=result.records, skipped=skipped + result.skipped)
