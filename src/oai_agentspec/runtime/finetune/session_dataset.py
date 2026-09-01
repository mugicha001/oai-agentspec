"""SDK Session からの SFT データセット生成（adapter 結合層・FR-4）。

会話履歴（SDK `Session`・不透明型）を累積ペアリングで SFT ケース列へ変換し、
`to_sft_dataset` へ委譲して `DatasetBuildResult` を返す。SDK 接触（`Session.get_items`）を
伴うため純データ層 `dataset.py` とは分離し、`_adapters/finetune.py` の
`fetch_session_items` を関数内遅延 import で呼ぶ（NFR-1）。

生成規則（累積ペアリング + 正規化破棄規則）の設計判断は ADR 0033
（`docs/adr/0033-session-dataset-pairing.md`）を参照。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .dataset import to_sft_dataset
from .types import DatasetBuildResult, FineTuneError, FineTuneFailureKind

if TYPE_CHECKING:
    from collections.abc import Callable

# 正規化で採用する role（それ以外の item はターン単位で破棄・skipped に数えない）。
_ADOPTED_ROLES = frozenset({"user", "assistant"})


def _content_text(content: Any) -> str:
    """履歴 item の content フィールドをテキスト文字列へ吸収する。

    Responses API の content は `str` または `[{"text": "..."}, ...]` 形式の parts 配列
    （`output_text` 等）を取り得る。parts 形式は FT の vision parts 形式とは別物のため
    透過せず、text 系フィールドを連結した文字列へ吸収する（ADR 0033 Decision 4。
    `_session_store._content_text` と同型のロジックの新設であり直接 import はしない）。

    Args:
        content: 履歴 item の content（str / list / その他）。

    Returns:
        テキスト化した文字列。None は空文字、未知型は str() 変換結果。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _normalize_turns(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """履歴 items を user / assistant ターンの列へ正規化する。

    `role` キーを持ち値が `"user"` / `"assistant"` の item のみ採用し、role なし item
    （function_call / function_call_output / reasoning / compaction 等）と system /
    developer / tool 等の item はターン単位で破棄する（`skipped` に数えない・ADR 0033
    Decision 2）。content は `_content_text` で str へ吸収する。

    Args:
        items: `Session.get_items()` が返した履歴 items。

    Returns:
        `{"role", "content"}` のみを持つターン dict の列（履歴順）。
    """
    turns: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in _ADOPTED_ROLES:
            continue
        turns.append({"role": role, "content": _content_text(item.get("content"))})
    return turns


def _validation_error(message: str) -> FineTuneError:
    """`VALIDATION_FAILED` の構造化エラーを組み立てる。

    Args:
        message: 人間可読のエラーメッセージ。

    Returns:
        `FineTuneError`（kind は `VALIDATION_FAILED`）。
    """
    return FineTuneError(FineTuneFailureKind.VALIDATION_FAILED, message)


async def dataset_from_session(
    session: Any,
    *,
    system: str | None = None,
    case_filter: Callable[[dict[str, Any]], bool] | None = None,
    case_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> DatasetBuildResult:
    """会話履歴（SDK `Session`）から SFT データセットを生成する（読み取り専用）。

    `session.get_items()` を 1 回だけ呼び（adapter 経由・書込系メソッドには触れない）、
    正規化後の各 assistant ターンを `expected_output`、それ以前の全採用ターンを `input`
    とするケースを累積ペアリングで生成し、`to_sft_dataset` へ委譲する。input が空になる
    ケース（正規化後の履歴先頭が assistant）と、吸収後の content が空になる assistant 応答
    （text フィールドを持たない parts のみ = refusal 等）のケースは生成せず `skipped` に
    計上する（空の expected_output を学習させるレコードを silent に混入させない）。

    `case_filter` で除外されたケースも `skipped` に計上する。filter が全ケースを除外した
    場合はエラーにせず `DatasetBuildResult(records=(), skipped=全件)` を正常返却する
    （ADR 0033 Decision 6）。`case_transform` は filter 通過ケースのみに適用する。

    Args:
        session: SDK `Session` Protocol 相当のオブジェクト（`get_items` を持つこと）。
        system: 全レコードの先頭へ挿入する system メッセージ本文（None なら挿入しない）。
            履歴由来の system item は正規化で事前破棄されるため競合しない（ADR 0033）。
            `case_transform` が input へ system メッセージを注入した場合はこの限りでなく、
            委譲先 `to_sft_dataset` の競合検出が発火する（fail-closed）。
        case_filter: ケースの採否を返す述語（履歴順に全ケースへ 1 回ずつ適用。False で
            除外し `skipped` に計上）。None なら全ケース採用。`case_transform` と同じく
            turn dict は全ケース間で共有参照のため、判定中に in-place 変更しないこと。
        case_transform: filter 通過ケースへ適用する変換（マスキング等）。戻り値の dict を
            ケースとして採用する。None なら無変換。累積ペアリングの `input` 内の turn dict
            は copy せず参照のまま全ケース間で共有するため、受け取った dict を in-place
            変更せず新しい dict を組んで返すこと（in-place 変更は他ケースの `input` へ
            波及する）。

    Returns:
        変換結果（`records` / `skipped`）。`skipped` は空 input ケース数 + 空応答ケース数 +
        filter 除外数。

    Raises:
        FineTuneError: `session` が None の場合（`CONFIG_MISSING`）。履歴が空・抽出可能な
            ターンが無い・assistant ターンが無い・`case_transform` が dict 以外を返した
            場合、および委譲先 `to_sft_dataset` の検証違反（`VALIDATION_FAILED`）。
    """
    if session is None:
        raise FineTuneError(
            FineTuneFailureKind.CONFIG_MISSING,
            "session が指定されていません（SDK Session オブジェクトは必須です）",
        )

    from ..._adapters import finetune as _ft

    items = await _ft.fetch_session_items(session)
    if not items:
        raise _validation_error("履歴が空です（Session に items が 1 件もありません）")

    turns = _normalize_turns(items)
    if not turns:
        raise _validation_error(
            "抽出可能なターンがありません（user / assistant の item が履歴に存在しません）"
        )
    if not any(turn["role"] == "assistant" for turn in turns):
        raise _validation_error(
            "assistant ターンがありません（expected_output にできる応答が履歴に存在しません）"
        )

    skipped = 0
    cases: list[dict[str, Any]] = []
    for position, turn in enumerate(turns):
        if turn["role"] != "assistant":
            continue
        if position == 0 or not turn["content"]:
            # 先行文脈なし（先頭 assistant）と、吸収後の content が空の応答（text を持たない
            # parts のみ = refusal 等）は学習ケースとして成立しないため個別除外する
            # （空 expected_output のレコードは「空出力を教える」silent 汚染になる）。
            skipped += 1
            continue
        cases.append({"input": turns[:position], "expected_output": turn["content"]})

    accepted: list[dict[str, Any]] = []
    for case in cases:
        if case_filter is not None and not case_filter(case):
            skipped += 1
            continue
        if case_transform is not None:
            transformed = case_transform(case)
            if not isinstance(transformed, dict):
                raise _validation_error(
                    "case_transform が dict 以外を返しました: "
                    f"{type(transformed).__name__}（ケースは dict で返すこと）"
                )
            case = transformed
        accepted.append(case)

    if not accepted:
        return DatasetBuildResult(records=(), skipped=skipped)
    result = to_sft_dataset(accepted, system=system)
    return DatasetBuildResult(records=result.records, skipped=skipped + result.skipped)
