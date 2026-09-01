"""SDK Session からの SFT / DPO データセット生成（adapter 結合層・FR-4 / FR-11）。

会話履歴（SDK `Session`・不透明型）を累積ペアリングでケース列へ変換し、`to_sft_dataset` /
`to_dpo_dataset` へ委譲して `DatasetBuildResult` を返す。SDK 接触（`Session.get_items`）を
伴うため純データ層 `dataset.py` とは分離し、`_adapters/finetune.py` の
`fetch_session_items` を関数内遅延 import で呼ぶ（NFR-1）。

生成規則（累積ペアリング + 正規化破棄規則）の設計判断は ADR 0033
（`docs/adr/0033-session-dataset-pairing.md`）、ツール往復の変換保持と DPO の 2 モードは
ADR 0034（`docs/adr/0034-session-tool-roundtrip-and-dpo-draft.md`）を参照。併合条件（射影列上の
連続）と `output` の型写像は ADR 0036
（`docs/adr/0036-session-normalization-merge-and-tool-output-typing.md`）で改訂した。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .dataset import to_dpo_dataset, to_sft_dataset
from .types import DatasetBuildResult, FineTuneError, FineTuneFailureKind

if TYPE_CHECKING:
    from collections.abc import Callable

# 正規化で採用する role（それ以外の item はターン単位で破棄・skipped に数えない）。
_ADOPTED_ROLES = frozenset({"user", "assistant"})

# 変換保持の対象となるツール往復 item の type（ADR 0034 Decision 1）。
_FUNCTION_CALL_TYPE = "function_call"
_FUNCTION_CALL_OUTPUT_TYPE = "function_call_output"

# DPO のケース素材が持つキー（`pair_builder` へ渡す plain dict の形）。
_DPO_REQUIRED_KEYS = ("preferred_output", "non_preferred_output")


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


def _tool_output_text(output: Any, call_id: Any) -> str:
    """function_call_output の `output` を chat 形式の tool メッセージ content へ写す。

    role `"tool"` のメッセージは content が文字列であることを要求されるため（`validate_dataset`
    の合法集合・プラットフォーム側の受理条件）、非文字列の `output` は JSON 文字列へ直列化して
    載せる。値の中身は解釈・要約・省略せず保つ（ADR 0036 の「内容非改変の型写像」。chat 形式が
    要求する型への 1:1・決定的・可逆な写像のみを行う）。`_content_text` は parts 配列専用で
    素の配列を空文字へ潰すため使わない。

    直列化できない値は `default=str` で当該値のみ文字列へ落とし、外側の JSON 構造は保つ。
    それでも失敗する残余（循環参照・文字列化できない dict キー）は SDK 経由の履歴では到達せず
    利用者が直せる入力のため、silent 劣化を残さず `VALIDATION_FAILED` で失敗させる。
    なお `float("nan")` / `inf` は `allow_nan` の既定に依拠して例外化せず、`NaN` / `Infinity`
    を含む非厳密 JSON として載る（int / float 等のキーも `json` 既定のまま文字列へ変換される）。

    Args:
        output: 履歴 item の `output` フィールド（未指定なら None）。
        call_id: エラーメッセージへ載せる当該 item の `call_id`。

    Returns:
        `output` が str ならそのまま、None（キー無しを含む）なら空文字、その他は
        `json.dumps(..., ensure_ascii=False, default=str)` の結果。

    Raises:
        FineTuneError: `output` を JSON へ直列化できない場合（`VALIDATION_FAILED`）。
    """
    if isinstance(output, str):
        return output
    if output is None:
        return ""
    try:
        return json.dumps(output, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        raise _validation_error(
            f"function_call_output（call_id: {call_id!r}）の output を JSON へ直列化できません"
            f": {exc}（循環参照または文字列化できないキーを含む値は tool メッセージの"
            " content へ載せられません）"
        ) from exc


def _paired_call_ids(items: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """function_call / function_call_output の `call_id` を相互突合する先行パス。

    対応相手を持たない孤児 item（片側のみの function_call / function_call_output）を本パスで
    破棄できるよう、両側に現れた `call_id` の集合を求める（ADR 0034 Decision 3）。

    Args:
        items: `Session.get_items()` が返した履歴 items。

    Returns:
        `(function_call 側で対応が取れた call_id 集合, function_call_output 側で対応が取れた
        call_id 集合)`。両者は同一集合だが、呼び出し側の判定を明示にするため 2 値で返す。
    """
    call_ids: set[str] = set()
    output_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        call_id = item.get("call_id")
        if not isinstance(call_id, str):
            continue
        if item.get("type") == _FUNCTION_CALL_TYPE:
            call_ids.add(call_id)
        elif item.get("type") == _FUNCTION_CALL_OUTPUT_TYPE:
            output_ids.add(call_id)
    paired = call_ids & output_ids
    return paired, paired


def _tool_call_entry(item: dict[str, Any]) -> dict[str, Any]:
    """function_call item を chat 形式の tool_calls 要素へ写す（非改変透過）。

    Args:
        item: `type` が `"function_call"` の履歴 item。

    Returns:
        `{"id", "type", "function": {"name", "arguments"}}` 形式の tool_calls 要素。
        `arguments` は解釈・改変せずそのまま載せる。
    """
    return {
        "id": item.get("call_id"),
        "type": "function",
        "function": {"name": item.get("name"), "arguments": item.get("arguments")},
    }


def _normalize_turns(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """履歴 items を chat 形式のターン列へ正規化する（ADR 0034 Decision 1-3 / ADR 0036）。

    採用・変換規則は次の 3 形で、生成されるターン dict は異種列になる。

    - role が `"user"` / `"assistant"` の item: `{"role", "content"}` のテキストターン
      （content は `_content_text` で str へ吸収する）
    - 対応済み function_call: `{"role": "assistant", "tool_calls": [...]}`（`arguments` は
      非改変透過）。**破棄対象 item を取り除いた列（射影列）の上で連続する** function_call を
      1 つの assistant の tool_calls 配列へ併合する（出力ターンを 1 件も生まない item は
      透明として跨ぐ。間に出力ターンを生む item = function_call_output / テキストターンが
      あれば併合しない・ADR 0036）
    - 対応済み function_call_output: `{"role": "tool", "tool_call_id", "content"}`
      （`output` は `_tool_output_text` で文字列へ写す。str はそのまま、非 str は JSON 文字列、
      キー無し / None は空文字。role `"tool"` は content が文字列必須のため）

    `call_id` の対応相手を欠く孤児 function_call / function_call_output は当該 item のみ
    破棄する（ADR 0034 Decision 3）。非 function 系の補助 item（reasoning / compaction /
    web_search_call 等）と生 role の system / developer / tool item は従来どおりターン単位で
    破棄する（`skipped` に数えない・ADR 0033 Decision 2 の存続部分）。

    Args:
        items: `Session.get_items()` が返した履歴 items。

    Returns:
        テキストターン / tool_calls 付き assistant / role `"tool"` の 3 形が混在する
        ターン dict の列（履歴順）。

    Raises:
        FineTuneError: function_call_output の `output` を JSON へ直列化できない場合
            （委譲先 `_tool_output_text` が送出する・`VALIDATION_FAILED`）。
    """
    paired_calls, paired_outputs = _paired_call_ids(items)
    turns: list[dict[str, Any]] = []
    pending_tool_calls: dict[str, Any] | None = None
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        call_id = item.get("call_id")
        if item_type == _FUNCTION_CALL_TYPE and call_id in paired_calls:
            if pending_tool_calls is not None:
                pending_tool_calls["tool_calls"].append(_tool_call_entry(item))
                continue
            pending_tool_calls = {"role": "assistant", "tool_calls": [_tool_call_entry(item)]}
            turns.append(pending_tool_calls)
            continue
        if item_type == _FUNCTION_CALL_OUTPUT_TYPE and call_id in paired_outputs:
            content = _tool_output_text(item.get("output"), call_id)
            pending_tool_calls = None
            turns.append({"role": "tool", "tool_call_id": call_id, "content": content})
            continue
        role = item.get("role")
        if role not in _ADOPTED_ROLES:
            continue
        pending_tool_calls = None
        turns.append({"role": role, "content": _content_text(item.get("content"))})
    return turns


def _is_text_assistant(turn: dict[str, Any]) -> bool:
    """ターンがケース化対象（テキスト応答の assistant）かを判定する。

    変換で生成した tool_calls 付き assistant は文脈にのみ現れ、ケースを生まない
    （ADR 0034 Decision 4）。

    Args:
        turn: `_normalize_turns` が生成したターン dict。

    Returns:
        テキスト応答の assistant ターンなら True。
    """
    return turn["role"] == "assistant" and "tool_calls" not in turn


def _has_dangling_tool_call(context: list[dict[str, Any]]) -> bool:
    """文脈プレフィックス内に、対応する tool メッセージを欠く tool_calls があるかを判定する。

    累積ペアリングはツール往復の途中（function_call と function_call_output の間に assistant
    テキストが挟まる履歴など）でも切り出しうるため、切り出した文脈だけを見ると `tool_calls`
    に対応する role `"tool"` メッセージが含まれないことがある。この並びは推論時 API が拒否
    するため、当該ケースは生成せず skip する（判定は文脈プレフィックス内で完結させ、併合
    ロジックには手を入れない）。

    Args:
        context: `_normalize_turns` が生成したターン列のプレフィックス（切り出した文脈）。

    Returns:
        対応する tool メッセージが文脈内に無い `tool_calls` の id が 1 つでもあれば True。
    """
    requested: set[str] = set()
    answered: set[str] = set()
    for turn in context:
        tool_calls = turn.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, dict) and isinstance(call.get("id"), str):
                    requested.add(call["id"])
            continue
        if turn.get("role") == "tool":
            tool_call_id = turn.get("tool_call_id")
            if isinstance(tool_call_id, str):
                answered.add(tool_call_id)
    return bool(requested - answered)


def _validation_error(message: str) -> FineTuneError:
    """`VALIDATION_FAILED` の構造化エラーを組み立てる。

    Args:
        message: 人間可読のエラーメッセージ。

    Returns:
        `FineTuneError`（kind は `VALIDATION_FAILED`）。
    """
    return FineTuneError(FineTuneFailureKind.VALIDATION_FAILED, message)


def _collect_case_materials(
    items: list[dict[str, Any]],
) -> tuple[list[tuple[list[dict[str, Any]], str]], int]:
    """履歴 items を正規化し、累積ペアリングでケース素材へ切り出す（SFT / DPO 共通）。

    ケース化対象はテキスト応答の assistant ターンのみで、各ターンの直前までの全採用ターン
    （変換済みツールメッセージを含む）を文脈とする（ADR 0034 Decision 4）。文脈が空になる
    ケース（正規化後の履歴先頭が assistant）と、吸収後の応答が空になるケース（text
    フィールドを持たない parts のみ = refusal 等）は生成せず `skipped` に計上する。文脈が
    ツール往復の途中で切れるケース（対応する tool メッセージを欠く `tool_calls` を含む）も
    同じ skip 経路へ合流させる。

    Args:
        items: `Session.get_items()` が返した履歴 items。

    Returns:
        `((文脈ターン列, 実応答文字列) の列, skipped 件数)`。

    Raises:
        FineTuneError: 履歴が空 / 抽出可能なターンが無い / テキスト応答の assistant ターンが
            無い場合、および function_call_output の `output` を JSON へ直列化できない場合
            （`VALIDATION_FAILED`）。
    """
    if not items:
        raise _validation_error("履歴が空です（Session に items が 1 件もありません）")

    turns = _normalize_turns(items)
    if not turns:
        raise _validation_error(
            "抽出可能なターンがありません（user / assistant の item が履歴に存在しません）"
        )
    if not any(_is_text_assistant(turn) for turn in turns):
        raise _validation_error(
            "assistant ターンがありません（expected_output にできる応答が履歴に存在しません）"
        )

    skipped = 0
    materials: list[tuple[list[dict[str, Any]], str]] = []
    for position, turn in enumerate(turns):
        if not _is_text_assistant(turn):
            continue
        if position == 0 or not turn["content"]:
            # 先行文脈なし（先頭 assistant）と、吸収後の content が空の応答（text を持たない
            # parts のみ = refusal 等）は学習ケースとして成立しないため個別除外する
            # （空 expected_output のレコードは「空出力を教える」silent 汚染になる）。
            skipped += 1
            continue
        context = turns[:position]
        if _has_dangling_tool_call(context):
            # ツール往復の途中で切れた文脈（対応する tool メッセージを欠く tool_calls を含む）
            # は推論時 API が拒否する並びのため、空文脈 / 空応答と同じ skip 経路へ合流させる。
            skipped += 1
            continue
        materials.append((context, turn["content"]))
    return materials, skipped


async def dataset_from_session(
    session: Any,
    *,
    system: str | None = None,
    tools: list[Any] | None = None,
    parallel_tool_calls: bool | None = None,
    case_filter: Callable[[dict[str, Any]], bool] | None = None,
    case_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> DatasetBuildResult:
    """会話履歴（SDK `Session`）から SFT データセットを生成する（読み取り専用）。

    `session.get_items()` を 1 回だけ呼び（adapter 経由・書込系メソッドには触れない）、
    正規化後の各 assistant ターンを `expected_output`、それ以前の全採用ターンを `input`
    とするケースを累積ペアリングで生成し、`to_sft_dataset` へ委譲する。input が空になる
    ケース（正規化後の履歴先頭が assistant）と、吸収後の content が空になる assistant 応答
    （text フィールドを持たない parts のみ = refusal 等）のケースは生成せず `skipped` に
    計上する（空の expected_output を学習させるレコードを silent に混入させない）。文脈が
    ツール往復の途中で切れるケース（`tool_calls` に対応する role `"tool"` メッセージが文脈に
    含まれない = function_call と function_call_output の間に assistant テキストが挟まる履歴
    等）も、推論時 API が拒否する並びのため生成せず `skipped` に計上する。

    `case_filter` で除外されたケースも `skipped` に計上する。filter が全ケースを除外した
    場合はエラーにせず `DatasetBuildResult(records=(), skipped=全件)` を正常返却する
    （ADR 0033 Decision 6）。`case_transform` は filter 通過ケースのみに適用する。

    履歴中のツール往復は破棄せず chat 形式へ変換して文脈に残す（function_call →
    `tool_calls` 付き assistant メッセージ / function_call_output → role `"tool"`
    メッセージ・ADR 0034 Decision 1）。したがって `case_filter` / `case_transform` が
    受けるケースの `input` には変換済みツールメッセージが混在し、tool_calls 付き
    assistant は `content` キーを持たない（`case["input"]` の各要素へ `content` 前提で
    アクセスしないこと）。tool メッセージの `content` は文字列へ正規化する（`output` が str
    ならそのまま、非 str は `json.dumps` の JSON 文字列、キー無し / None は空文字。role
    `"tool"` は content が文字列必須のため）。値の中身は解釈・要約せず保つため、機密情報の
    除去は利用者の責務である（NFR-5・`case_transform` で行う）。履歴が function_call から
    始まる断片（compaction 後のスライス等）では、`input` が変換済みツールメッセージのみで
    user メッセージを 1 件も含まないケースも生成され得る（空でない限り skip しない）。

    Args:
        session: SDK `Session` Protocol 相当のオブジェクト（`get_items` を持つこと）。
        system: 全レコードの先頭へ挿入する system メッセージ本文（None なら挿入しない）。
            履歴由来の system item は正規化で事前破棄されるため競合しない（ADR 0033）。
            `case_transform` が input へ system メッセージを注入した場合はこの限りでなく、
            委譲先 `to_sft_dataset` の競合検出が発火する（fail-closed）。
        tools: 全レコードの直下へ透過するツール定義（plain dict / FunctionTool 相当の混在可）。
            利用者が供給した定義をそのまま `to_sft_dataset` へ渡すのみで、lib は内容を解釈
            しない（写像・検証の規則は FR-1 / FR-2 と同一で委譲先に一元化する）。None（既定）
            ならレコードへキー自体を出さない。会話ログにツール定義は記録されないため履歴から
            の復元は行わない（利用者供給の定義の透過のみ・ADR 0035）。
        parallel_tool_calls: 全レコードの直下へ透過する並列ツール呼び出しの可否。`False` も
            指定として扱い透過する。None（既定）ならレコードへキー自体を出さない。
        case_filter: ケースの採否を返す述語（履歴順に全ケースへ 1 回ずつ適用。False で
            除外し `skipped` に計上）。None なら全ケース採用。`case_transform` と同じく
            turn dict は全ケース間で共有参照のため、判定中に in-place 変更しないこと。
        case_transform: filter 通過ケースへ適用する変換（マスキング等）。戻り値の dict を
            ケースとして採用する。None なら無変換。累積ペアリングの `input` 内の turn dict
            は copy せず参照のまま全ケース間で共有するため、受け取った dict を in-place
            変更せず新しい dict を組んで返すこと（in-place 変更は他ケースの `input` へ
            波及する。変換済みツールメッセージも同じ共有参照であり同様に扱うこと）。

    Returns:
        変換結果（`records` / `skipped`）。`skipped` は空 input ケース数 + 空応答ケース数 +
        ツール往復の途中で切れた文脈のケース数 + filter 除外数。

    Raises:
        FineTuneError: `session` が None の場合（`CONFIG_MISSING`）。履歴が空・抽出可能な
            ターンが無い・assistant ターンが無い・`case_transform` が dict 以外を返した
            場合、履歴中の function_call_output の `output` を JSON へ直列化できない場合
            （循環参照・文字列化できない dict キー）、および委譲先 `to_sft_dataset` の検証違反
            （`tools=` の不正要素を含む。採用ケースが 0 件でも委譲するため空結果の経路でも
            発火する・`VALIDATION_FAILED`）。
    """
    if session is None:
        raise FineTuneError(
            FineTuneFailureKind.CONFIG_MISSING,
            "session が指定されていません（SDK Session オブジェクトは必須です）",
        )

    from ..._adapters import finetune as _ft

    items = await _ft.fetch_session_items(session)
    materials, skipped = _collect_case_materials(items)
    cases: list[dict[str, Any]] = [
        {"input": context, "expected_output": response} for context, response in materials
    ]

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

    result = to_sft_dataset(
        accepted, system=system, tools=tools, parallel_tool_calls=parallel_tool_calls
    )
    return DatasetBuildResult(records=result.records, skipped=skipped + result.skipped)


async def dpo_dataset_from_session(
    session: Any,
    *,
    pair_builder: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    tools: list[Any] | None = None,
    parallel_tool_calls: bool | None = None,
) -> DatasetBuildResult:
    """会話履歴（SDK `Session`）から DPO（preference）データセットを生成する（読み取り専用）。

    ケース素材の切り出しは `dataset_from_session` と同一（`session.get_items()` を 1 回だけ
    呼び、累積ペアリング・ツール往復の変換保持・空文脈 / 空応答 / ツール往復の途中で切れた
    文脈のケースの skip を行う）で、
    素材は `{"input": <累積文脈>, "response": <ログ上の実応答>}` の 2 キーの plain dict で
    ある。どちらの応答を preferred / non-preferred とするかは品質判定であり lib は内蔵
    しないため、次の 2 モードのいずれかで利用者が決める（ADR 0034 Decision 5）。

    - callable モード（`pair_builder` 指定）: 各ケース素材へ適用し、`preferred_output` /
      `non_preferred_output` を含む dict で採用する（任意キー `input` を含めれば lib が
      組んだ累積文脈を差し替えられる）。`None` を返したケースは生成せず `skipped` へ計上し、
      全ケース skip はエラーにせず空結果を正常返却する。最終変換は
      `to_dpo_dataset(skip_missing=False)` へ委譲する
    - 雛形モード（`pair_builder` 省略）: `{"input", "preferred_output",
      "non_preferred_output", "response"}` の 4 キーからなる**記入用ケース列**を
      `records` として返す（記入 2 欄は空文字）。この `records` は最終レコードではなく、
      `save_dpo_draft` / `finalize_dpo_draft`（FR-12）で記入・取り込みを経て最終
      データセットになる

    `input` の turn dict は copy せず全ケース間で共有参照するため、`pair_builder` 内で
    in-place 変更せず新しい dict を組むこと。ツール出力は文脈の tool メッセージの `content`
    へ文字列として載る（非 str の `output` は `json.dumps` で JSON 文字列化し、キー無し /
    None は空文字にする）。値の中身は解釈・要約しないため、機密情報の除去は利用者の責務で
    ある（NFR-5）。履歴が function_call から始まる断片
    （compaction 後のスライス等）では、`input` が変換済みツールメッセージのみで user
    メッセージを 1 件も含まないケースも生成され得る（空でない限り skip しない）。

    Args:
        session: SDK `Session` Protocol 相当のオブジェクト（`get_items` を持つこと）。
        pair_builder: ケース素材から preference ペアを組む callable。`preferred_output` /
            `non_preferred_output` の両キーを含む dict（任意キー `input` で文脈差し替え）
            または skip を表す `None` を返すこと。None（既定）なら雛形モードで動作する。
        tools: 各レコードの `input` 内へ透過するツール定義（plain dict / FunctionTool 相当の
            混在可・SFT のレコード直下と透過位置が異なる）。利用者が供給した定義をそのまま
            `to_dpo_dataset` へ渡すのみで、lib は内容を解釈しない（写像・検証の規則は
            FR-1 / FR-2 と同一で委譲先に一元化する）。None（既定）なら `input` へキー自体を
            出さない。callable モード専用で、雛形モードで指定すると `CONFIG_MISSING` で失敗
            する（ADR 0035）。会話ログにツール定義は記録されないため履歴からの復元は行わない
            （利用者供給の定義の透過のみ）。
        parallel_tool_calls: 各レコードの `input` 内へ透過する並列ツール呼び出しの可否。
            `False` も指定として扱い透過する。None（既定）なら `input` へキー自体を出さない。
            `tools` と同じく雛形モードでは指定できない。

    Returns:
        変換結果（`records` / `skipped`）。雛形モードの `records` は記入用ケース列であり
        最終レコードではない。`skipped` は空文脈ケース + 空応答ケース + ツール往復の途中で
        切れた文脈のケース + `pair_builder` が `None` を返したケースの合計。

    Raises:
        FineTuneError: `session` が None の場合、および雛形モード（`pair_builder` 省略）で
            `tools=` / `parallel_tool_calls=` を指定した場合（いずれも `CONFIG_MISSING`。
            後者は反映先が無いため履歴を読む前に失敗する。ツール定義は
            `finalize_dpo_draft(source, tools=...)` へ渡すこと）。履歴が空・抽出可能な
            ターンが無い・テキスト応答の assistant ターンが無い・`pair_builder` が None でも
            dict でもない値を返した・戻り値が必須キーを欠く場合、および委譲先
            `to_dpo_dataset` の検証違反（`tools=` の不正要素を含む。採用ケースが 0 件でも
            委譲するため空結果の経路でも発火する・`VALIDATION_FAILED`）。履歴中の
            function_call_output の `output` を JSON へ直列化できない場合（循環参照・
            文字列化できない dict キー）も `VALIDATION_FAILED` になる。委譲先エラーに載る
            `ケース {index}` は skip を除いた委譲リスト上の位置であり、元ケースの位置とは
            一致しないことがある。
    """
    if session is None:
        raise FineTuneError(
            FineTuneFailureKind.CONFIG_MISSING,
            "session が指定されていません（SDK Session オブジェクトは必須です）",
        )
    if pair_builder is None and (tools is not None or parallel_tool_calls is not None):
        raise FineTuneError(
            FineTuneFailureKind.CONFIG_MISSING,
            "雛形モードでは tools= / parallel_tool_calls= を指定できません（記入用ケース列は"
            " to_dpo_dataset へ委譲しないため反映先がありません）。ツール定義は"
            " finalize_dpo_draft(source, tools=...) で渡すこと",
        )

    from ..._adapters import finetune as _ft

    items = await _ft.fetch_session_items(session)
    materials, skipped = _collect_case_materials(items)

    if pair_builder is None:
        records = tuple(
            {
                "input": context,
                "preferred_output": "",
                "non_preferred_output": "",
                "response": response,
            }
            for context, response in materials
        )
        return DatasetBuildResult(records=records, skipped=skipped)

    cases: list[dict[str, Any]] = []
    for index, (context, response) in enumerate(materials, start=1):
        built = pair_builder({"input": context, "response": response})
        if built is None:
            skipped += 1
            continue
        if not isinstance(built, dict):
            raise _validation_error(
                f"ケース {index}: pair_builder が dict でも None でもない値を返しました: "
                f"{type(built).__name__}（ペアは dict、skip は None で返すこと）"
            )
        missing = [key for key in _DPO_REQUIRED_KEYS if key not in built]
        if missing:
            raise _validation_error(
                f"ケース {index}: pair_builder の戻り値に必須キーがありません: {', '.join(missing)}"
            )
        cases.append(
            {
                "input": built["input"] if "input" in built else context,
                "preferred_output": built["preferred_output"],
                "non_preferred_output": built["non_preferred_output"],
            }
        )

    result = to_dpo_dataset(
        cases, skip_missing=False, tools=tools, parallel_tool_calls=parallel_tool_calls
    )
    return DatasetBuildResult(records=result.records, skipped=skipped + result.skipped)
