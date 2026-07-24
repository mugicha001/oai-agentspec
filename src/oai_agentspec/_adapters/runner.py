"""Runner 委譲アダプタと plain 結果型（SDK 結合を `_adapters` に閉じる・NFR-1）。

`DefaultRunnerAdapter`（runner シーム本番実装）/ plain 結果型 `RunOutcome` / `ApplyResult` /
SDK 結果を plain 統一表現へ変換する `_extract_pending` / `_outcome_from_result` を提供する。SDK
結合（`agents` の `Runner` / `RunContextWrapper`）は本モジュール内に閉じ、外へは plain な値のみを
渡す。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agents import Runner

from .run_context import unwrap_run_context

if TYPE_CHECKING:
    from ..runtime.llmops.types import ObservedRun


@dataclass(frozen=True)
class RunOutcome:
    """`Runner` 実行結果の plain 統一表現（中断 or 完了・NFR-1）。

    共有コアが SDK 型（RunResult / RunState / ToolApprovalItem）を直接見ないよう、
    中断有無を bool で、承認待ち一覧を plain dict のリストで、再開用 `RunState` を不透明
    `Any`（共有コアからは中身を覗かない）で受け渡す。

    Attributes:
        final_output: 完了時の最終出力テキスト。中断時は None。
        interrupted: 承認待ち（中断）が 1 件以上あるか。
        pending: 承認待ち一覧（`{"tool_name": str, "call_id": str, "agent_name": str}` の plain
            dict のリスト）。`agent_name` は承認待ちを発生させた Agent 名（approve 認可を
            `(agent_name, tool_name)` 単位で行うため）。中断なしなら空リスト。
        state: 再開に使う SDK `RunState`（不透明 `Any`）。中断なしなら None。
    """

    final_output: str | None
    interrupted: bool
    pending: list[dict[str, str]] = field(default_factory=list)
    state: Any = None


@dataclass(frozen=True)
class ApplyResult:
    """承認/却下適用の plain 結果（SDK 例外を漏らさない構造化結果）。

    Attributes:
        applied: 適用できた call_id のリスト。
        unknown: 引き当てられなかった（未知の）call_id のリスト。
        already_resolved: 既に解決済みで再操作された call_id のリスト。
    """

    applied: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    already_resolved: list[str] = field(default_factory=list)


def _pending_agent_name(item: Any) -> str:
    """`ToolApprovalItem` から、その承認待ちを発生させた agent 名を防御的に取り出す。

    SDK の `ToolApprovalItem.agent`（当該ツール呼び出しを行った Agent）の `name` を返す。属性
    消失（SDK 退行）や agent 不明時は空文字を返す（共有コアの approve 認可は agent 不明を安全側
    ＝非認可として扱う）。SDK 内部結合は本 `_adapters` に閉じる（NFR-1）。

    Args:
        item: 対象の `ToolApprovalItem`。

    Returns:
        承認待ちを発生させた agent 名。取得不能なら空文字。
    """
    agent = getattr(item, "agent", None)
    name = getattr(agent, "name", None)
    return "" if name is None else str(name)


def _extract_pending(interruptions: list[Any]) -> list[dict[str, str]]:
    """SDK の interruptions（ToolApprovalItem 列）を plain dict 列へ変換する。

    各 `ToolApprovalItem` から `call_id`（function_call の call_id）・`tool_name`・`agent_name`
    （承認待ちを発生させた Agent 名）を取り出し `{"tool_name", "call_id", "agent_name"}` の plain
    dict にする。`call_id` / `tool_name` / agent 名が None の場合は空文字で補完する（共有コアは
    call_id で引き当て、approve 認可は `(agent_name, tool_name)` で行う）。

    Args:
        interruptions: SDK `ToolApprovalItem` のリスト（`result.interruptions`）。

    Returns:
        承認待ち一覧の plain dict のリスト（`{"tool_name", "call_id", "agent_name"}`）。
    """
    pending: list[dict[str, str]] = []
    for item in interruptions:
        call_id = getattr(item, "call_id", None)
        tool_name = getattr(item, "tool_name", None)
        pending.append(
            {
                "tool_name": "" if tool_name is None else str(tool_name),
                "call_id": "" if call_id is None else str(call_id),
                "agent_name": _pending_agent_name(item),
            }
        )
    return pending


def _outcome_from_result(result: Any) -> RunOutcome:
    """SDK の RunResult / RunResultStreaming を `RunOutcome`（plain 統一表現）へ変換する。

    `result.interruptions` を確認し、1 件以上あれば中断として承認待ち一覧と
    `result.to_state()` の `RunState`（不透明）を載せた `RunOutcome` を返す。空なら完了として
    最終出力のみを載せた `RunOutcome` を返す（従来の最終出力経路と同義）。

    Args:
        result: SDK `RunResult` または `RunResultStreaming`。

    Returns:
        中断 or 完了を表す `RunOutcome`。
    """
    # 前提: SDK RunResult は `interruptions` 属性を必ず持つ。`getattr(..., None) or []` は属性
    # 消失（SDK 退行）と「中断ゼロ」を同一視するが、属性存在前提は L2 SDK 耐性トリップワイヤで
    # 別途検証する想定のため、ここでは安全側（中断なし扱い）に倒すだけに留める。
    interruptions = list(getattr(result, "interruptions", None) or [])
    if interruptions:
        return RunOutcome(
            final_output=None,
            interrupted=True,
            pending=_extract_pending(interruptions),
            state=result.to_state(),
        )
    final = result.final_output
    return RunOutcome(
        final_output="" if final is None else str(final),
        interrupted=False,
    )


class DefaultRunnerAdapter:
    """runner シーム（`RunnerSeam`）の本番実装。SDK `Runner.run` をラップする。

    AGENT ステップ実行を委譲される。`agent` は registry 上の名前ではなく解決済みの
    SDK Agent を受け取る前提だが、内部インタプリタからはステップ宣言の agent 名が渡る。
    本実装は名前から Agent を解決する責務を持たないため、利用者は `as_agent_spec` /
    `as_facade_spec` 経由で利用する（registry 解決は呼び出し側で完結する）。

    SDK `Runner.run` への素通し（passthrough）に結合する（NFR-7）。`input` / `context`
    のみ明示管理し、残りの kwarg は `**runner_kwargs` で `Runner.run` へ委譲する。SDK
    シグネチャ進化時の追従対象を本 `_adapters` に局在化する。
    """

    def __init__(self, registry: Any = None) -> None:
        """runner を生成する。

        Args:
            registry: AGENT ステップ名を SDK Agent へ解決する AgentRegistry（任意）。
                None の場合は agent をそのまま SDK Agent として扱う。
        """
        self._registry = registry

    async def run(
        self,
        agent: Any,
        input: Any,
        *,
        context: Any = None,
        **runner_kwargs: Any,
    ) -> Any:
        """AGENT ステップを `Runner.run` で実行し RunResult を返す。

        Args:
            agent: registry 上のエージェント名、または SDK Agent。
            input: ステップへの入力。
            context: 各ステップへ素通しする共有 context。`RunContextWrapper`（経路A の
                ToolContext 等）の場合は `.context`（生オブジェクト）を取り出して `Runner.run`
                へ渡す（SDK が再ラップするため）。生オブジェクト / None はそのまま渡す。
            **runner_kwargs: `Runner.run` へ素通しする残りの kwarg（run_config / session /
                max_turns / hooks 等）。グラフ既定 run_defaults + ノード run_options のマージ結果。

        Returns:
            SDK RunResult（`final_output` を持つ）。
        """
        resolved = self._registry.get(agent) if self._registry is not None else agent
        raw_context = unwrap_run_context(context)
        return await Runner.run(resolved, input, context=raw_context, **runner_kwargs)

    async def run_outcome(
        self,
        agent: Any,
        input: Any,  # noqa: A002 - Runner.run の引数名に追従
        *,
        context: Any = None,
        **runner_kwargs: Any,
    ) -> RunOutcome:
        """AGENT ステップを `Runner.run` で実行し中断 or 完了の `RunOutcome` を返す。

        `run` と同じく `Runner.run` へ素通しするが、戻りを `result.interruptions` に基づく
        plain な `RunOutcome`（中断時は承認待ち一覧 + 不透明 `RunState`、完了時は最終出力）へ
        変換する。共有コアが SDK 型を見ずに HITL を扱えるようにする（NFR-1）。

        Args:
            agent: registry 上のエージェント名、または SDK Agent。
            input: ステップへの入力（文字列 / input-list / 再開時は `RunState`）。
            context: 各実行へ素通しする共有 context（`RunContextWrapper` は `.context` を展開）。
            **runner_kwargs: `Runner.run` へ素通しする残りの kwarg（session 等）。

        Returns:
            中断 or 完了を表す `RunOutcome`。
        """
        result = await self.run(agent, input, context=context, **runner_kwargs)
        return _outcome_from_result(result)

    async def run_with_observation(
        self,
        agent: Any,
        input: Any,  # noqa: A002 - Runner.run の引数名に追従
        *,
        context: Any = None,
        **runner_kwargs: Any,
    ) -> tuple[RunOutcome, ObservedRun]:
        """AGENT ステップを 1 回実行し plain な `RunOutcome` + `ObservedRun` を返す（LLMOps）。

        `self.run(...)` で生 `RunResult` を 1 回だけ取得し、`_outcome_from_result`（最終出力 /
        中断）と `routing.observe_run_result`（routing 経路 + ツール呼び出し）の plain 抽出を
        その場（`_adapters` 内）で行う。**生 `RunResult` は `_adapters` 外へ一切出さない**
        （NFR-1）。LLMOps 評価が横断 routing / ツール使用を採点するために使う。既存
        `run` / `run_outcome` は不変。

        Args:
            agent: registry 上のエージェント名、または SDK Agent。
            input: ステップへの入力。
            context: 各実行へ素通しする共有 context（`RunContextWrapper` は `.context` を展開）。
            **runner_kwargs: `Runner.run` へ素通しする残りの kwarg（session 等）。

        Returns:
            plain な `RunOutcome`（最終出力 / 中断）と `ObservedRun`（route + tool_calls）のタプル。
        """
        from .routing import observe_run_result

        result = await self.run(agent, input, context=context, **runner_kwargs)
        return _outcome_from_result(result), observe_run_result(result)
