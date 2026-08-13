"""不足パラメータの予測（`LLMFiller`）と、実行後の会話継続を通す例（実 API）。

例 12（オフライン）との差は `bind(llm_filler=LLMFiller(model=...))` を渡す 1 点である。
渡すと `await planner.plan(query)` が「不足のある候補がある場合に限り」予測段を駆動する。
駆動は **1 ターンあたり `Runner.run` 1 回**であり、候補件数・不足件数に比例しない
（本例は候補 2 件・不足 3 件を 1 回の予測でまとめて埋める。`detail=True` で得られる
`PlanResult.usage`（`ParamUsage`）に `runs=1` / `candidates=2` として観測できる）。

`LLMFiller(model=...)` に渡すのは**モデル**であってエージェント実体ではない。予測エージェント
はライブラリが内部で組み立てるため、利用者はプロンプトもエージェントも書かない。

スロットの読み方（例 12 の 3 種類に加えて予測由来の 2 種類を混ぜてある）:

- `by_llm=True` + `confirm=False`: 予測値がそのまま入る -> `resolved`（origin=llm）
- `by_llm=True` + `confirm=True`: 押す前に確認させる -> `needs_confirmation`
  （`max_suggestions=2` なら信頼度の降順に最大 2 件が `suggestions` として提示される）

**予測段に会話を届ける経路は `IntentContext.history_items` だけである**。`IntentQuery.utterance`
は system 指示部へ連結されない（プロンプト注入経路を増やさない設計判断・NFR-6）ため、
`IntentQuery.history` にセッションを渡して `ContextBuilder` に `history_items` を組ませないと、
予測エージェントは発話の内容（「90 秒」「staging」）を知る手段がない。本例は
`agents.SQLiteSession` に窓口とのやり取りと現在発話を積み、`IntentQuery(history=session)` で
渡している。

実行と会話継続:

- 押下後の実行は利用者側が `Runner.run(registry.get(plan.action_agent), input=plan.input_json)`
  と書く（build-don't-run。lib は実行 API を持たない）。
- そのターンはハンドオフ遷移を経ないため `resolve_next_agent` / `next_turn_agent` の発動条件
  （ハンドオフ観測）が成立しない。アクション直接起動の後始末は `action_next_turn_agent` が
  担い、包括ルールの `next_agent` へ会話を戻す。
- **到達時ハンドオフ禁止（`no_handoff_on_arrival=True`）を宣言しているため、実行先の解決は
  `apply_next_turn_policy` が返す派生 registry から行う**。元の registry には到達記録の前置
  合成も `is_enabled` ゲートも設置されていないため、元から解決すると宣言した禁止が無症状で
  効かない。

Azure OpenAI の環境変数（AZURE_OPENAI_*・examples/_shared/_azure.py 参照。`.env` は
`azure_model()` が読み込む）を設定して実行:

    uv run python examples/intent/13_action_param_prediction.py

本例は実 API を呼ぶ（warmup 1 回 + パラメータ予測 1 回 + アクション実行 1 回）。滞留したまま
走り続けないよう、スクリプト側にも絶対上限（`TIMEOUT_SECONDS`）を掛けてある。
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from agents import Runner, SQLiteSession

from oai_agentspec import (
    AgentRegistry,
    AgentSpec,
    HandoffGraph,
    NextTurnPolicy,
    NextTurnRule,
    action_next_turn_agent,
    apply_next_turn_policy,
    resolve_next_agent,
)
from oai_agentspec.runtime.intent import (
    ActionCatalog,
    ActionPlan,
    ActionSpec,
    CandidateSource,
    ConfidenceLevel,
    ExecutableIntent,
    IntentContext,
    IntentPrediction,
    IntentQuery,
    LLMFiller,
    PlanResult,
    param,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _warmup import warmup  # noqa: E402

from _azure import azure_model  # noqa: E402

#: スクリプト自身に掛ける絶対上限（秒）。実 API 呼び出しが滞留したまま走り続けないよう、
#: 実行側の timeout とあわせて多層で上限を持たせる。
TIMEOUT_SECONDS: Final[float] = 120.0

#: 現在発話。候補生成（キーワードマッチ）は `IntentContext.utterance` を読むが、予測段へは
#: `history_items` 経由でしか届かないため、セッションにも同じ文を user item として積む。
UTTERANCE: Final[str] = (
    "api.example.com の staging に 90 秒の負荷テストをかけたい。だめなら切り戻しも"
)

# action_id -> 候補生成のキーワードと、候補自身が供給するパラメータ。候補生成は LLM を
# 使わないキーワードマッチに固定し、この例で駆動される LLM を予測段の 1 回だけに絞る。
KEYWORDS: Final[dict[str, frozenset[str]]] = {
    "run_load_test": frozenset({"負荷", "ロードテスト"}),
    "rollback_deployment": frozenset({"ロールバック", "切り戻し"}),
}
CANDIDATE_PARAMETERS: Final[dict[str, dict[str, Any]]] = {
    "run_load_test": {},
    "rollback_deployment": {"service": "checkout"},
}

# 実行エージェント X が回答を終えたら会話を窓口（reception）へ戻し、あわせて X へ
# ハンドオフで到達したターンでは X の全 handoff を無効化する（たらい回しの遮断）。
POLICY: Final[NextTurnPolicy] = NextTurnPolicy(
    rules={
        "load_test_runner": NextTurnRule(next_agent="reception", no_handoff_on_arrival=True),
        "deploy_operator": NextTurnRule(next_agent="reception", no_handoff_on_arrival=True),
    }
)


@dataclass(frozen=True)
class OpsContext:
    """`IntentQuery.run_context` に渡す run context（`from_context` の解決先）。"""

    host: str
    operator: str
    ticket_id: str | None = None


def build_registry() -> AgentRegistry:
    """窓口 1 件とアクションの実行先 2 件を登録した registry を組む。

    Returns:
        ハンドオフ結線済みの `AgentRegistry`（`apply_next_turn_policy` へ渡す前の素の状態）。
    """
    model = azure_model()
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="reception",
            instructions="運用作業の受付窓口です。依頼内容を確認し、必要なら担当へ引き継ぎます。",
            model=model,
        )
    )
    registry.register(
        AgentSpec(
            name="load_test_runner",
            instructions=(
                "負荷テストの実行担当です。渡された JSON の条件で実行したものとして、"
                "結果の要約を 2 文以内で日本語で報告してください。"
            ),
            model=model,
        )
    )
    registry.register(
        AgentSpec(
            name="deploy_operator",
            instructions="デプロイの切り戻し担当です。作業内容を 2 文以内で日本語で報告します。",
            model=model,
        )
    )

    graph = HandoffGraph(entry="reception")
    graph.edge("reception", "load_test_runner", description="負荷テストの実行は負荷担当へ")
    graph.edge("reception", "deploy_operator", description="切り戻しはデプロイ担当へ")
    # 実行担当どうしの出辺。到達時ハンドオフ禁止が無効化する対象になる。
    graph.edge("load_test_runner", "deploy_operator", description="切り戻しが要るなら引き継ぐ")
    graph.apply(registry)
    registry.validate()
    return registry


def build_catalog() -> ActionCatalog:
    """実行可能アクション 2 件を宣言する（`by_llm=True` を各 1 件以上含む）。

    Returns:
        アクション 2 件を登録済みの `ActionCatalog`。
    """
    catalog = ActionCatalog()
    catalog.register(
        ActionSpec(
            action_id="run_load_test",
            description="対象ホストへ負荷テストを実行する",
            action_agent="load_test_runner",
            label=(
                "${target} へ ${duration_seconds} 秒の負荷テスト（${environment} / ${ticket_id}）"
            ),
            parameters=(
                # run context から解決する（-> resolved / origin=run_context）。
                param("target", str, from_context="host", description="負荷をかける対象ホスト"),
                # 予測段が埋める（-> resolved / origin=llm）。
                param(
                    "duration_seconds",
                    int,
                    by_llm=True,
                    description="負荷をかける秒数。発話に無ければ 60 と見積もる",
                ),
                # 予測段が埋めるが押す前に確認させる（-> needs_confirmation）。
                # max_suggestions=2 なので信頼度の降順に最大 2 件が提示される。
                param(
                    "environment",
                    str,
                    by_llm=True,
                    confirm=True,
                    max_suggestions=2,
                    description="実行環境（production / staging のいずれか。誤爆防止のため確認）",
                ),
                # run context に無い（ticket_id=None）ので利用者に聞く（-> needs_user）。
                param("ticket_id", str, from_context="ticket_id", description="起票済みの番号"),
            ),
        )
    )
    catalog.register(
        ActionSpec(
            action_id="rollback_deployment",
            description="デプロイを直前のリビジョンへ切り戻す",
            action_agent="deploy_operator",
            label="${service} を ${revision} へ切り戻し（承認 ${approved_by}）",
            parameters=(
                param("service", str, filled_by_candidate=True, description="対象サービス"),
                param(
                    "revision",
                    str,
                    by_llm=True,
                    description="戻す先のリビジョン。発話に無ければ 'previous' と答える",
                ),
                param(
                    "approved_by",
                    str,
                    from_context="operator",
                    confirm=True,
                    description="承認者（確認する）",
                ),
            ),
        )
    )
    return catalog


class KeywordActionGenerator:
    """キーワードマッチで `ExecutableIntent` を返す `CandidateGenerator`（LLM 不使用）。"""

    async def generate(self, context: IntentContext[Any]) -> IntentPrediction:
        """発話に含まれるキーワードから候補を組み立てる。

        Args:
            context: `ContextBuilder` が組んだ入力。ここでは `utterance` だけを見る。

        Returns:
            マッチしたアクションぶんの `ExecutableIntent` を載せた `IntentPrediction`。
        """
        candidates = [
            ExecutableIntent(
                action_id=action_id,
                level=ConfidenceLevel.HIGH,
                source="keyword",
                parameters=CANDIDATE_PARAMETERS[action_id],
            )
            for action_id, keywords in KEYWORDS.items()
            if any(keyword in context.utterance for keyword in keywords)
        ]
        return IntentPrediction(candidates=tuple(candidates), metadata={"generator": "keyword"})


def print_plan(plan: ActionPlan, title: str) -> None:
    """計画 1 件のスロット状態を表示する。

    Args:
        plan: 表示する `ActionPlan`。
        title: 見出し。
    """
    print(f"\n[{title}] action_id={plan.action_id} agent={plan.action_agent}")
    print(f"[LABEL]     {plan.label}")
    for slot in plan.slots:
        origin = slot.origin.value if slot.origin is not None else "-"
        suggested = (
            " suggestions="
            + str([(s.value, s.level.value) for s in slot.suggestions])
            + f" (max={_max_suggestions(plan, slot.name)})"
            if slot.suggestions
            else ""
        )
        print(
            f"[SLOT]      {slot.name}: state={slot.state.value} value={slot.value!r} "
            f"origin={origin}{suggested}"
        )
    pending = ", ".join(slot.name for slot in plan.pending) or "なし"
    print(f"[PENDING]   {pending}")
    print(f"[READY]     {plan.ready}")


def _max_suggestions(plan: ActionPlan, name: str) -> int:
    """当該パラメータの宣言済み `max_suggestions` を返す（提示件数の上限の表示用）。

    Args:
        plan: 対象の計画。
        name: パラメータ名。

    Returns:
        宣言済みの `max_suggestions`。
    """
    return next(p.max_suggestions for p in plan.spec.parameters if p.name == name)


def print_usage(result: PlanResult) -> None:
    """予測段の実行量を表示する（`Runner.run` が 1 ターン 1 回であることの観測点）。

    Args:
        result: `planner.plan(detail=True)` の戻り。
    """
    usage = result.usage
    print(
        f"[USAGE]     runs={usage.runs} model_calls={usage.model_calls} "
        f"candidates={usage.candidates} input_tokens={usage.input_tokens} "
        f"output_tokens={usage.output_tokens}"
    )


async def main() -> None:
    """宣言 -> 予測 -> 確認 -> 実行 -> 会話継続までを 1 本で通す。"""
    run_context = OpsContext(host="api.example.com", operator="suzuki")

    # 到達時ハンドオフ禁止を宣言しているため、以降は派生 registry だけを使う
    # （元の registry から解決すると禁止の合成が載っていない実体が返る）。
    runtime_registry = apply_next_turn_policy(POLICY, build_registry())

    model = azure_model()
    await warmup(model)

    planner = build_catalog().bind(
        registry=runtime_registry,
        candidates=CandidateSource(generator=KeywordActionGenerator()),
        # 穴埋めの結線。渡すのはモデルであり、予測エージェントは lib が内部で組む。
        llm_filler=LLMFiller(model=model),
    )
    planner.validate(context=run_context)
    print("=" * 72)
    print("[VALIDATE]  宣言と結線の整合を確認（LLM 0 回）")

    with tempfile.TemporaryDirectory() as tmp:
        # 窓口とのやり取りと現在発話をセッションへ積む（実利用では ConversationService や
        # Runner が積む）。utterance は system 指示部へ連結されないため、発話の内容が
        # 予測エージェントへ届く経路は history_items だけである。
        session = SQLiteSession(session_id="ops-console", db_path=str(Path(tmp) / "conv.db"))
        await session.add_items(
            [
                {"role": "user", "content": "リリース前の確認をしたい"},
                {
                    "role": "assistant",
                    "content": "承知しました。負荷テストの実行や切り戻しの手配ができます。",
                },
                {"role": "user", "content": UTTERANCE},
            ]
        )
        query = IntentQuery(utterance=UTTERANCE, history=session, run_context=run_context)
        result = await planner.plan(query, detail=True)

    print(f"[QUERY]     {query.utterance}")
    print("[HISTORY]   予測段へ会話を届ける唯一の経路（IntentContext.history_items）:")
    for item in result.suggestion.context.history_items:
        print(f"[HISTORY]     {item}")
    print(f"[PLANS]     候補と同順・同数: {len(result.plans)} 件")
    print_usage(result)
    for plan in result.plans:
        print_plan(plan, "PLAN")

    # 押下: 確認スロットは提示された候補値をそのまま採用し（-> user_confirmed）、
    # 残りは利用者入力で埋める（-> user_input）。元の計画は変更されない。
    plan = next(p for p in result.plans if p.action_id == "run_load_test")
    environment = next(slot for slot in plan.slots if slot.name == "environment")
    answers = {
        # 提示の先頭は信頼度が最も高い候補値（`suggestions` は降順に並ぶ）。
        "environment": environment.suggestions[0].value,
        "ticket_id": "OPS-1234",
    }
    applied = plan.apply(answers)
    print("\n" + "=" * 72)
    print(f"[APPLY]     answers={answers}")
    print_plan(applied, "APPLIED")

    # 実行（build-don't-run: この 1 行は lib ではなく利用者が書く）。
    print("\n" + "=" * 72)
    print(f"[INPUT]     {applied.input_json}")
    run_result = await Runner.run(
        runtime_registry.get(applied.action_agent), input=applied.input_json
    )
    print(f"[OUTPUT]    {run_result.final_output}")

    # 会話継続: ハンドオフを経ていないため通常の解決は発動しない（None）。
    # アクション直接起動の後始末は action_next_turn_agent が担う。
    print(
        f"[RESOLVE]   resolve_next_agent -> "
        f"{resolve_next_agent(POLICY, run_result)!r}（上書きなし）"
    )
    agent = action_next_turn_agent(POLICY, run_result, runtime_registry)
    if agent is None:
        print("[NEXT]      次ターンの開始エージェントを決定できませんでした")
        return
    print(f"[NEXT]      次ターンの開始エージェント: {agent.name}")


async def _main_with_timeout() -> None:
    """絶対上限（`TIMEOUT_SECONDS`）を掛けて `main()` を走らせる。

    Raises:
        TimeoutError: 上限を超えた場合。実 API が滞留したまま走り続けないようにする。
    """
    async with asyncio.timeout(TIMEOUT_SECONDS):
        await main()


if __name__ == "__main__":
    asyncio.run(_main_with_timeout())
