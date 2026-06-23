"""reward ファクトリ群（`contains` / `exact` / `tool_match` / `route_match` / `last_agent_match` /
`approval_match` / `judge`・callable 生成のみ）。

よくある目的関数を 1 行で記述するための callable 生成ヘルパ。dataset の各観点フィールドや rubric
を受けて `reward` callable（`(RolloutResult) -> float`）を返すだけで、報酬データ・プロンプト・
データを lib に内蔵しない（FR-5 / FR-9）。手書き `reward` callable も従来どおり `optimize` に
渡せる（併存）。

各 reward ファクトリの `field` 引数は省略可能で、既定値は `OptimizeCase`（`dataset` モジュール）
の標準フィールド名に揃っている:

    - `contains` / `exact`        → `"expected_output"`
    - `tool_match`                → `"expected_tools"`
    - `route_match`               → `"expected_route"`
    - `last_agent_match`          → `"expected_last_agent"`
    - `approval_match`            → `"expected_approvals"`

`OptimizeCase` を使うときはフィールド名を渡さずに `reward=contains()` と書ける。dict ケースで
自由なフィールド名を使うときは従来どおり `contains("my_expected")` のように明示する（後方互換・
`_case_value` が dict / 属性アクセス両対応）。

`RolloutResult` は plain な観測（`case` / `output` / `tool_calls` / `fired_approvals` /
`route_steps` / `last_agent`）を `types` モジュールで定義する。本モジュールは型を import するのみ。

`judge` の Judge モデル呼び出しは `_adapters/lightning.judge_score` 経由（SDK 結合を `_adapters`
に閉じる・NFR-1）。本モジュールは `agents` / `agentlightning` を import しない（plain データのみ
扱う）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .types import RolloutResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


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


def contains(field: str = "expected_output") -> Callable[[RolloutResult], float]:  # noqa: A002 - 公開引数名
    """出力に `case[field]`（期待文字列）が含まれれば 1.0 を返す reward を生成する（FR-9）。

    Args:
        field: 期待文字列を持つ dataset フィールド名。既定 `"expected_output"`（`OptimizeCase` の
            標準フィールド）。dict ケースで自由なフィールド名を使う場合は明示する。

    Returns:
        `(RolloutResult) -> float` の reward callable。
    """

    def reward(result: RolloutResult) -> float:
        expected = _case_value(result.case, field)
        if expected is None:
            return 0.0
        return 1.0 if str(expected) in result.output else 0.0

    return reward


def exact(field: str = "expected_output") -> Callable[[RolloutResult], float]:  # noqa: A002 - 公開引数名
    """出力が `case[field]`（期待文字列）と完全一致すれば 1.0 を返す reward を生成する（FR-9）。

    Args:
        field: 期待文字列を持つ dataset フィールド名。既定 `"expected_output"`（`OptimizeCase` の
            標準フィールド）。

    Returns:
        `(RolloutResult) -> float` の reward callable。
    """

    def reward(result: RolloutResult) -> float:
        expected = _case_value(result.case, field)
        if expected is None:
            return 0.0
        return 1.0 if result.output.strip() == str(expected).strip() else 0.0

    return reward


def tool_match(field: str = "expected_tools") -> Callable[[RolloutResult], float]:  # noqa: A002 - 公開引数名
    """期待ツール（`case[field]`）が全て呼ばれていれば 1.0 を返す reward を生成する（FR-9）。

    `case[field]` は期待ツール名の列（list / tuple 等）。観測ツール呼び出し集合が期待集合を包含
    （recall）すれば 1.0、欠落があれば 0.0（余分な呼び出しは無視）。

    Args:
        field: 期待ツール名の列を持つ dataset フィールド名。既定 `"expected_tools"`（`OptimizeCase`
            の標準フィールド）。

    Returns:
        `(RolloutResult) -> float` の reward callable。
    """

    def reward(result: RolloutResult) -> float:
        expected = _case_value(result.case, field)
        if not expected:
            return 0.0
        observed = set(result.tool_calls)
        return 1.0 if all(tool in observed for tool in expected) else 0.0

    return reward


def route_match(field: str = "expected_route") -> Callable[[RolloutResult], float]:  # noqa: A002 - 公開引数名
    """期待ルート（`case[field]`）と観測ルート（route_steps）が完全一致すれば 1.0 を返す（FR-9）。

    `case[field]` は期待される経路（agent 名の列・**起点を含むフルパス**）。観測 `route_steps`
    （`RolloutResult.route_steps`）と順序・経由回数まで含めて完全一致比較する（llmops
    `HandoffRoute` と同型）。「triage で受けて billing へ handoff」を期待するなら
    `["triage", "billing"]`。単体応答（handoff なし）なら `["triage"]`。途中で中断した場合は
    `route_steps` が短くなり一致しない（0.0）。

    Args:
        field: 期待経路（agent 名の列）を持つ dataset フィールド名。既定 `"expected_route"`
            （`OptimizeCase` の標準フィールド）。

    Returns:
        `(RolloutResult) -> float` の reward callable。
    """

    def reward(result: RolloutResult) -> float:
        expected = _case_value(result.case, field)
        if not expected:
            return 0.0
        return 1.0 if list(result.route_steps) == list(expected) else 0.0

    return reward


def last_agent_match(field: str = "expected_last_agent") -> Callable[[RolloutResult], float]:  # noqa: A002 - 公開引数名
    """期待最終 agent（`case[field]`）と観測 `last_agent` が一致すれば 1.0 を返す（FR-9）。

    `case[field]` は期待される最終応答 agent 名（経路の終端）。「請求関連は最終的に billing に
    届くべき」のように、handoff 経路の細部に関わらず**最終応答した agent だけ**を採点したいとき
    に使う（経路全体を厳密採点したい場合は `route_match`）。`last_agent` が None（rollout が応答
    せず中断した場合）は常に 0.0。

    Args:
        field: 期待最終 agent 名を持つ dataset フィールド名。既定 `"expected_last_agent"`
            （`OptimizeCase` の標準フィールド）。

    Returns:
        `(RolloutResult) -> float` の reward callable。
    """

    def reward(result: RolloutResult) -> float:
        expected = _case_value(result.case, field)
        if not expected or result.last_agent is None:
            return 0.0
        return 1.0 if result.last_agent == str(expected) else 0.0

    return reward


def approval_match(field: str = "expected_approvals") -> Callable[[RolloutResult], float]:  # noqa: A002 - 公開引数名
    """期待承認ゲート（`case[field]`）が全て発火していれば 1.0 を返す reward を生成する（FR-9）。

    `case[field]` は期待される承認ゲート（`needs_approval=True` のツール）名の列（list / tuple
    等）。観測した発火承認集合（`RolloutResult.fired_approvals`）が期待集合を包含（recall）すれば
    1.0、欠落があれば 0.0（余分な発火は無視）。「危険ツールを正しく承認ゲートへ回せたか」を APO
    で学習させたいときに使う（tool_match は承認後に実行されたツールを見るのに対し、本ファクトリは
    承認待ちに**出たか**自体を見る）。

    Args:
        field: 期待承認ゲート名の列を持つ dataset フィールド名。既定 `"expected_approvals"`
            （`OptimizeCase` の標準フィールド）。

    Returns:
        `(RolloutResult) -> float` の reward callable。
    """

    def reward(result: RolloutResult) -> float:
        expected = _case_value(result.case, field)
        if not expected:
            return 0.0
        observed = set(result.fired_approvals)
        return 1.0 if all(tool in observed for tool in expected) else 0.0

    return reward


def judge(rubric: str, model: Any) -> Callable[[RolloutResult], Awaitable[float]]:
    """利用者 Judge モデルで rollout 出力を 0.0..1.0 で採点する reward を生成する（FR-9）。

    生成した reward は async callable（`(RolloutResult) -> Awaitable[float]`）で、`_adapters` 経由
    （`judge_score`）で最小エージェントを 1 ターン実行して採点する（SDK 結合を `_adapters` に
    閉じる・NFR-1）。`rubric` / `model` は利用者供給で lib に判定プロンプト・モデルを内蔵しない。

    Args:
        rubric: 採点観点文（利用者供給）。
        model: 採点に使う LLM（SDK `Model` / モデル名文字列等の不透明値）。

    Returns:
        `(RolloutResult) -> Awaitable[float]` の async reward callable。
    """

    async def reward(result: RolloutResult) -> float:
        from ..._adapters import judge_score

        return await judge_score(rubric=rubric, model=model, output=result.output, case=result.case)

    return reward
