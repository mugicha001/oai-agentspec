"""実行可能アクションの宣言から押下までを LLM なしで通す例（オフライン実行）。

`ActionCatalog` / `ActionSpec` / `param` でアクションとパラメータを宣言し、
`planner = catalog.bind(...)` で結線済みの `ActionPlanner` を得て、
`planner.validate(...)` の起動時検証 -> `await planner.plan(query, predict=False)` の
決定的なスロット確定 -> `plan.apply(answers)` -> `plan.input_json` までを 1 本で示す。

要点は `bind(llm_filler=...)` を**渡さない**ことである。渡さなければ穴埋め経路そのものが
存在せず、従量課金も環境変数も発生しない（`llm_filler` を渡さないことが「LLM に埋めさせ
ない」という利用者の明示的な意思表示になる）。候補も LLM ではなく自作のキーワード
マッチ generator が返すため、この例は API キーなしで完全にオフラインで動く。

スロットの状態遷移が読めるよう、値の出どころを 3 種類混ぜてある:

- `filled_by_candidate=True`: 候補（`ExecutableIntent.parameters`）が供給する -> `resolved`
- `from_context=...`: run context のパスから解決する -> `resolved`（`confirm=True` なら
  `needs_confirmation`）
- `from_context` が解決できない: 利用者に聞く -> `needs_user`（`plan.apply(answers)` で確定）

アクションの実行（`Runner.run`）はこの例の範囲外である（build-don't-run）。ここでは
実行入力（`plan.input_json`）を組み立てるところまでを示す。

環境変数不要で実行:
    uv run python examples/intent/12_executable_actions.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from oai_agentspec import AgentRegistry, AgentSpec
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
    param,
)

# action_id -> 候補生成のキーワードと、候補自身が供給するパラメータ。
KEYWORDS: dict[str, frozenset[str]] = {
    "run_load_test": frozenset({"負荷", "ロードテスト"}),
    "rollback_deployment": frozenset({"ロールバック", "切り戻し"}),
}
CANDIDATE_PARAMETERS: dict[str, dict[str, Any]] = {
    "run_load_test": {"duration_seconds": 60},
    "rollback_deployment": {"service": "checkout", "revision": "v1.4.2"},
}


@dataclass(frozen=True)
class OpsContext:
    """`Runner.run(context=...)` に渡す想定の run context（`from_context` の解決先）。"""

    host: str
    environment: str
    operator: str
    ticket_id: str | None = None


def build_registry() -> AgentRegistry:
    """アクションの実行先エージェントを宣言する（この例では実行しない）。

    `ActionSpec.action_agent` はエージェント名の str であり、実体の解決は利用者が
    `registry.get()` で行う。起動時検証は「その名前が registry にあるか」だけを見る。

    Returns:
        実行先エージェントを登録済みの `AgentRegistry`。
    """
    registry = AgentRegistry()
    registry.register(
        AgentSpec(name="load_test_runner", instructions="負荷テストを実行して結果を報告します。")
    )
    registry.register(
        AgentSpec(name="deploy_operator", instructions="デプロイの切り戻しを実行します。")
    )
    return registry


def build_catalog() -> ActionCatalog:
    """実行可能アクション 2 件を宣言する。

    `prompt` / `prompt_vars` / `by_llm` は 1 つも宣言しない。穴埋め段が存在しない構成で
    これらを宣言すると「効かない宣言」として起動時検証に落とされる。

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
                # 候補（ExecutableIntent.parameters）が供給する（-> resolved / origin=candidate）。
                param(
                    "duration_seconds",
                    int,
                    filled_by_candidate=True,
                    description="負荷をかける秒数",
                ),
                # 解決できるが確認を要する（-> needs_confirmation）。
                param(
                    "environment",
                    str,
                    from_context="environment",
                    confirm=True,
                    description="実行環境（誤爆防止のため確認する）",
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
                param("revision", str, filled_by_candidate=True, description="戻す先のリビジョン"),
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
    """キーワードマッチで `ExecutableIntent` を返す自作 `CandidateGenerator`（LLM 不使用）。

    候補は必ず `ExecutableIntent` として返す。素の `IntentCandidate` を返すと
    `planner.plan()` の allowlist 除外で落ちる（`action_id` を持たないため）。
    """

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


def show_validation_catches_mistakes(registry: AgentRegistry) -> None:
    """宣言の誤りが `planner.validate()` で落ちることを見せる。

    ここで落とすのは、候補が押された瞬間ではなくアプリ起動時に不整合を出すためである。

    Args:
        registry: 実行先エージェントの登録簿。
    """
    broken = ActionCatalog()
    broken.register(
        ActionSpec(
            action_id="typo_action",
            description="action_agent の名前を打ち間違えた宣言",
            action_agent="load_test_ruuner",  # 打ち間違い（registry に無い）
            label="typo",
            parameters=(param("target", str, from_context="host"),),
        )
    )
    try:
        broken.bind(registry=registry).validate()
    except KeyError as exc:
        print(f"[VALIDATE] 宣言の誤りを起動時に検出: {exc}")


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
        detail = f" path={slot.detail}" if slot.detail is not None else ""
        suggested = (
            f" suggestions={[s.value for s in slot.suggestions]}" if slot.suggestions else ""
        )
        print(
            f"[SLOT]      {slot.name}: state={slot.state.value} value={slot.value!r} "
            f"origin={origin}{detail}{suggested}"
        )
    pending = ", ".join(slot.name for slot in plan.pending) or "なし"
    print(f"[PENDING]   {pending}")
    print(f"[READY]     {plan.ready}")


async def main() -> None:
    """宣言 -> 起動時検証 -> 計画 -> 押下（実行入力の組み立て）までを通す。"""
    registry = build_registry()
    catalog = build_catalog()
    run_context = OpsContext(host="api.example.com", environment="staging", operator="suzuki")

    print("=" * 72)
    print(f"[CATALOG]   宣言済み action_id: {catalog.names()}")

    # bind: 結線済みの ActionPlanner を得る（宣言簿はスナップショットされる）。
    # llm_filler を渡さない = 穴埋め経路そのものが存在しない（従量課金なし）。
    planner = catalog.bind(
        registry=registry,
        candidates=CandidateSource(generator=KeywordActionGenerator()),
    )

    # 起動時検証: run context の代表インスタンスを渡すとパスの構造検査まで行う。
    planner.validate(context=run_context)
    print("[VALIDATE]  宣言と結線の整合を確認（LLM 0 回・network なし）")
    show_validation_catches_mistakes(registry)

    # 毎ターンの窓口。predict=False で予測段を明示的に使わない。
    query = IntentQuery(
        utterance="staging に負荷テストをかけたい。だめなら切り戻しも検討したい",
        run_context=run_context,
    )
    plans = await planner.plan(query, predict=False)
    print("\n" + "=" * 72)
    print(f"[QUERY]     {query.utterance}")
    print(f"[PLANS]     候補と同順・同数: {len(plans)} 件")
    for plan in plans:
        print_plan(plan, "PLAN")

    # 押下: 確認済みの値と利用者入力を合流させる（元の計画は変更されない）。
    plan = plans[0]
    answers = {"environment": "staging", "ticket_id": "OPS-1234"}
    applied = plan.apply(answers)
    print("\n" + "=" * 72)
    print(f"[APPLY]     answers={answers}")
    print_plan(applied, "APPLIED")
    print(f"[IMMUTABLE] 元の計画は不変: ready={plan.ready}")

    # 実行入力の組み立て（実行は行わない = build-don't-run）。
    print("\n" + "=" * 72)
    print(f"[INPUT]     {applied.input_json}")
    print(f"[NEXT]      利用者側で Runner.run(registry.get({applied.action_agent!r}), input=...)")


if __name__ == "__main__":
    asyncio.run(main())
