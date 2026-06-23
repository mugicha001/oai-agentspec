"""プロンプトのディレクトリ構成を利用側に合わせる例。

PromptStore のディレクトリ構成は PromptLayout で明示する。既存フォルダ構成
（例 common / snippets / roles）に合わせられる。各 dir に "" を渡すと root 直下になる。

ここでは一時ディレクトリに common/ snippets/ roles/ という独自構成を作って合成を示す
（モデル呼び出しは行わない）。

実行:
    uv run python examples/custom_layout.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from oai_agentspec import PromptLayout, PromptStore


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # 既定とは異なるディレクトリ名で配置する。
        (root / "common").mkdir()
        (root / "snippets").mkdir()
        (root / "roles").mkdir()
        (root / "common" / "main.md").write_text(
            "あなたは ${company} のサポート担当です。", encoding="utf-8"
        )
        (root / "snippets" / "style.md").write_text("回答は3文以内で簡潔に。", encoding="utf-8")
        (root / "roles" / "triage.md").write_text(
            "依頼を適切な担当に振り分ける。", encoding="utf-8"
        )

        # 独自ディレクトリ名を PromptLayout で明示する。
        store = PromptStore(root, PromptLayout(base="common", parts="snippets", agents="roles"))

        print("=== 独自ディレクトリ構成（common / snippets / roles）での合成 ===")
        print(
            store.compose(
                agent="triage", base="main", parts=["style"], vars={"company": "AgentSpec Inc."}
            )
        )


if __name__ == "__main__":
    main()
