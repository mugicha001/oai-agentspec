"""全部守る: store + registry + workflow + custom checks を 1 呼び出しで固定。

`lockdown` の 6 段順次処理を全段動かす。
git 同梱の `sample_app/` と `sample_prompts/` を tmp にコピーして使う（コピー後に固定）。

実行:
    uv run python examples/integrity/02_lockdown_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import SAMPLE_APP, SAMPLE_PROMPTS, writable_copy  # noqa: E402

from oai_agentspec import (  # noqa: E402
    END,
    START,
    AgentRegistry,
    AgentSpec,
    PromptLayout,
    PromptStore,
    RegistryFrozenError,
    WorkflowFrozenError,
    WorkflowGraph,
    lockdown,
)

LAYOUT = PromptLayout(base="base", parts="parts", agents="agents")


def my_business_check() -> None:
    """利用者独自検知の雛形。違反時は IntegrityError を raise する。"""
    return None


def main() -> int:
    with writable_copy(SAMPLE_APP) as src_root, writable_copy(SAMPLE_PROMPTS) as prompt_root:
        store = PromptStore(prompt_root, LAYOUT)
        registry = AgentRegistry()
        for name in ("triage", "billing"):
            registry.register(
                AgentSpec(
                    name=name,
                    instructions=store.compose(agent=name, base="main", parts=["style"]),
                )
            )
        workflow = WorkflowGraph("main")
        workflow.add_agent_node("triage", agent="triage")
        workflow.add_agent_node("billing", agent="billing")
        workflow.add_edge(START, "triage")
        workflow.add_edge("triage", "billing")
        workflow.add_edge("billing", END)

        # 1 呼び出しで 6 段順次・fail-closed
        lockdown(
            src_root,
            store=store,
            registry=registry,
            workflow=workflow,
            libs=False,
            checks=[my_business_check],
        )
        print("[OK] lockdown 成功（6 段順次完了）")

        try:
            registry.register(AgentSpec(name="evil", instructions="evil"))
        except RegistryFrozenError as exc:
            print(f"[OK] registry 変更が遮断: {exc}")
        else:
            return 1

        try:
            workflow.add_edge("triage", END)
        except WorkflowFrozenError as exc:
            print(f"[OK] workflow 変更が遮断: {exc}")
        else:
            return 1

        registry.validate()
        print(f"[OK] read-only API 維持: {registry.names()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
