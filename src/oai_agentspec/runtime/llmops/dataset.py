"""評価入力ケースの plain 型（`EvalCase`）・安定キー導出・Langfuse dataset 連携ヘルパ。

利用者が提供する評価データのラッパー（lib にケースを同梱しない）。`id` は Langfuse dataset
item の安定キー（run 横断で同一ケースを対応づける）として使い、それ以外の評価ロジックでは
使わない。`id` 未指定時は index + 入力ハッシュから安定キーを導出する。

Langfuse Datasets は **register → fetch → use** モデル（Langfuse が source）。`register_dataset`
で一度きり登録し、`load_dataset` で fetch して使う。両ヘルパは sync（Langfuse SDK は sync・eval の
async 外で setup / fetch する想定）。EvalCase ↔ plain dict のマッピングは本モジュールが担い、
`_adapters/langfuse.py` には plain dict のみ渡す（単方向依存維持・`import langfuse` は `_adapters`
の関数内遅延に閉じる）。Langfuse dataset item は `input` / `expected_output` / `metadata` / `id` の
みのため、oai-agentspec 固有の `reference_context` / `expected_route` / `expected_tools` は
item.metadata に格納する。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .config import LangfuseConfig


@dataclass(frozen=True)
class EvalCase:
    """評価 1 ケースの入力（利用者提供・plain）。

    Attributes:
        input: 評価対象へ渡す入力（文字列 / input-list 等の任意型）。
        id: Langfuse dataset item の安定キー（run 横断の対応づけ用）。None なら
            `stable_id` が index + 入力ハッシュから導出する。dataset 連携以外では使わない。
        reference_context: factual_grounding 用の参照文脈（根拠）。criteria に Faithfulness を含めた
            場合、None ならその観点は not_applicable（criteria に無ければ評価行自体が出ない）。
        expected_route: 横断ルーティングの ground truth（期待エージェント名の系列）。criteria に
            HandoffRoute を含めた場合、None なら（または mode=single なら）not_applicable。
        expected_tools: ツール使用の ground truth（期待ツール名の集合）。criteria に ToolUse を
            含めた場合、None なら（または評価対象がツール非保有なら）not_applicable。
        expected_approvals: 承認ゲート発火の ground truth（中断時に承認待ちへ出るべきツール名の
            集合）。criteria に ApprovalGate を含めた場合、None ならその観点は not_applicable。
        expected_output: 正解文（golden answer）。Langfuse dataset item の expected_output に
            反映され、提供時は G-Eval が参照可能（`LLMTestCaseParams.EXPECTED_OUTPUT`）。観点
            ベース評価では必須でない（None 可）。利用者提供の信頼入力であり Spotlighting しない。
    """

    input: Any
    id: str | None = None
    reference_context: list[str] | None = None
    expected_route: list[str] | None = None
    expected_tools: list[str] | None = None
    expected_approvals: list[str] | None = None
    expected_output: str | None = None


def stable_id(case: EvalCase, index: int) -> str:
    """`EvalCase` の安定キーを返す（`id` 優先・未指定時は index + 入力ハッシュ）。

    利用者が `id` を渡していればそれを使う。未指定なら `case-{index}-{hash8}` を導出する
    （`hash8` は入力の文字列表現の SHA-256 先頭 8 桁）。run 横断で同一ケースを対応づける用途に
    用い、それ以外の評価ロジックでは使わない。

    Args:
        case: 安定キーを導出する評価ケース。
        index: データセット内の 0 始まりインデックス。

    Returns:
        安定キー文字列。
    """
    if case.id is not None:
        return case.id
    digest = hashlib.sha256(repr(case.input).encode("utf-8")).hexdigest()[:8]
    return f"case-{index}-{digest}"


# oai-agentspec 固有フィールドを格納する item.metadata のキー（EvalCase フィールド名と一致）。
_META_REFERENCE_CONTEXT = "reference_context"
_META_EXPECTED_ROUTE = "expected_route"
_META_EXPECTED_TOOLS = "expected_tools"
_META_EXPECTED_APPROVALS = "expected_approvals"


def _case_to_item(case: EvalCase, index: int) -> dict[str, Any]:
    """`EvalCase` を Langfuse dataset item の plain dict へ変換する。

    Langfuse item の `input` / `expected_output` / `id` へそのまま写し、oai-agentspec 固有の
    `reference_context` / `expected_route` / `expected_tools` / `expected_approvals` は
    `metadata` に格納する（None のフィールドは metadata から省略する）。

    Args:
        case: 変換する評価ケース。
        index: データセット内の 0 始まりインデックス（id 未指定時の stable_id 導出に使う）。

    Returns:
        `{"id", "input", "expected_output", "metadata"}` の plain dict。
    """
    metadata: dict[str, Any] = {}
    if case.reference_context is not None:
        metadata[_META_REFERENCE_CONTEXT] = case.reference_context
    if case.expected_route is not None:
        metadata[_META_EXPECTED_ROUTE] = case.expected_route
    if case.expected_tools is not None:
        metadata[_META_EXPECTED_TOOLS] = case.expected_tools
    if case.expected_approvals is not None:
        metadata[_META_EXPECTED_APPROVALS] = case.expected_approvals
    return {
        "id": stable_id(case, index),
        "input": case.input,
        "expected_output": case.expected_output,
        "metadata": metadata or None,
    }


def _item_to_case(item: dict[str, Any]) -> EvalCase:
    """Langfuse dataset item の plain dict を `EvalCase` へ復元する。

    `input` / `id` / `expected_output` を直接、`metadata` から `reference_context` /
    `expected_route` / `expected_tools` / `expected_approvals` を取り出す（metadata 非在キーは
    None）。

    Args:
        item: `{"id", "input", "expected_output", "metadata"}` の plain dict。

    Returns:
        復元した `EvalCase`。
    """
    metadata = item.get("metadata") or {}
    return EvalCase(
        input=item.get("input"),
        id=item.get("id"),
        reference_context=metadata.get(_META_REFERENCE_CONTEXT),
        expected_route=metadata.get(_META_EXPECTED_ROUTE),
        expected_tools=metadata.get(_META_EXPECTED_TOOLS),
        expected_approvals=metadata.get(_META_EXPECTED_APPROVALS),
        expected_output=item.get("expected_output"),
    )


def register_dataset(config: LangfuseConfig, name: str, cases: Sequence[EvalCase]) -> None:
    """評価ケース群を Langfuse dataset へ一度きり register/upsert する（冪等・sync）。

    各 `EvalCase` を plain dict（`_case_to_item`）へ変換して `_adapters` の `register_dataset_items`
    へ渡す（oai-agentspec 固有フィールドは item.metadata に格納）。register は冪等のため
    再実行可（同一
    `id` で upsert）。`import langfuse` は `_adapters` の関数内遅延に閉じる。

    Args:
        config: Langfuse 設定（認証・接続先）。`dataset_name` 等の評価専用設定は使わない。
        name: 登録先 dataset 名。
        cases: 登録する評価ケース列。
    """
    from ..._adapters import register_dataset_items

    items = [_case_to_item(case, index) for index, case in enumerate(cases)]
    register_dataset_items(config, name, items)


def load_dataset(config: LangfuseConfig, name: str) -> list[EvalCase]:
    """Langfuse dataset を fetch し `EvalCase` 列へ復元して返す（sync・「呼び出して使う」本体）。

    `_adapters` の `fetch_dataset_items`（plain dict 列）を `_item_to_case` で `EvalCase` 化する。
    dataset は Langfuse が source のため取得系 `get_dataset` を使う（push 専用制約は Prompt
    Management のみ）。`import langfuse` は `_adapters` の関数内遅延に閉じる。

    Args:
        config: Langfuse 設定（認証・接続先）。
        name: 取得元 dataset 名。

    Returns:
        復元した `EvalCase` 列。
    """
    from ..._adapters import fetch_dataset_items

    return [_item_to_case(item) for item in fetch_dataset_items(config, name)]
