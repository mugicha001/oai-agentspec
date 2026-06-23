"""プロンプト合成の順序を比較する例。

デフォルト順（base -> parts -> agent）と `layout` による順序上書きを、合成された
instructions を表示して比較する。モデル呼び出しは行わない。

実行:
    uv run python examples/composition.py
"""

from __future__ import annotations

from pathlib import Path

from oai_agentspec import PromptLayout, PromptStore

PROMPT_VARS = {"company": "AgentSpec Inc."}
LAYOUT = PromptLayout(base="base", parts="parts", agents="agents")


def main() -> None:
    store = PromptStore(Path(__file__).resolve().parent.parent / "prompts", LAYOUT)

    print("=== デフォルト順（base -> parts -> agent）===")
    print(store.compose(agent="triage", base="main", parts=["style", "safety"], vars=PROMPT_VARS))

    print("\n=== layout で順序上書き（agent -> base -> part:safety）===")
    print(store.compose(layout=["agent:triage", "base:main", "part:safety"], vars=PROMPT_VARS))


if __name__ == "__main__":
    main()
