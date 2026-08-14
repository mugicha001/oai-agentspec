"""SDK の Skills capability でスキルをサンドボックスエージェントへ与える例。

スキルを使う本線はこの経路（SandboxAgentSpec + `Skills` capability）。SDK が
スキルフォルダの発見・システムプロンプトへの一覧注入・遅延ロード用ツールの生成まで
面倒を見るため、利用側にローダー実装は要らない:

    SandboxAgentSpec(capabilities=[Shell(), Skills(lazy_from=...)])

`LocalDirLazySkillSource(source=LocalDir(src=...))` は **ホスト上の**スキルフォルダを
指す。スキル本文の置き場所（ホストのローカルディスク）と実行環境（サンドボックス
セッション）は分離されており、`load_skill` が呼ばれた時点で当該スキルだけが
ワークスペース配下（既定 `.agents/<スキル名>/`）へ materialize される。実行環境を
Docker バックエンドへ差し替えてもスキルの置き場所はホストのままでよい。

スキルの実体は `examples/skills/<スキル名>/SKILL.md`（frontmatter に name /
description、本文に手順）。同じフォルダを通常 Agent の ShellTool 経由で使う例は
`examples/basic/shell_tool_skills.py` にあるが、そちらは SDK がフォルダ発見を
提供しないため自作ローダーが必要になる（本例との対比）。

本例のスキル本文には「回答へ固有マーカーを含める」という指示が入っている。実行後に
(1) ワークスペースへスキルが materialize されたこと、(2) 最終回答にマーカーが
含まれること、の 2 点で「スキルが本当に読まれて適用された」ことを機械的に確認する。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/sandbox/skills.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

from agents import RunConfig, Runner
from agents.sandbox import Manifest, SandboxRunConfig
from agents.sandbox.capabilities import Shell, Skills
from agents.sandbox.capabilities.skills import LocalDirLazySkillSource
from agents.sandbox.entries import LocalDir
from agents.sandbox.sandboxes import UnixLocalSandboxClient

from oai_agentspec import AgentRegistry, SandboxAgentSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402

# 従量課金 API に接続するため、開始からの絶対上限を script 内 watchdog で強制する
# （想定所要 60s + マージン。macOS 標準に timeout コマンドが無いため script 内で完結させる）。
WATCHDOG_SECONDS = 90

SKILLS_ROOT = Path(__file__).resolve().parent.parent / "skills"
SKILL_MARKER = "SKILLMARK-42"


async def run() -> None:
    skill_source = LocalDirLazySkillSource(source=LocalDir(src=SKILLS_ROOT))
    print(f"--- discovered skills ({SKILLS_ROOT}) ---")
    for meta in skill_source.list_skill_metadata(skills_path=".agents"):
        print(f"{meta.name}: {meta.description} -> {meta.path}")

    workspace = tempfile.mkdtemp(prefix="oai-agentspec-skills-")
    try:
        registry = AgentRegistry()
        registry.register(
            SandboxAgentSpec(
                name="release_writer",
                instructions=(
                    "あなたはリリースノート担当。スキルが提供されている場合は、"
                    "必ずスキルを読み込んでから、その手順に従って回答する。"
                ),
                model=azure_model(),
                default_manifest=Manifest(root=workspace),
                # Shell は materialize 済み SKILL.md をモデルが読むために必要
                # （Filesystem が提供するのは view_image / apply_patch で、読み取り手段は
                # Shell の exec_command）。
                capabilities=[Shell(), Skills(lazy_from=skill_source)],
            )
        )
        agent = registry.get("release_writer")
        print(f"capabilities: {[type(c).__name__ for c in agent.capabilities]}")

        result = await Runner.run(
            agent,
            "release-note スキルに従って、『サンドボックスエージェントの宣言的サポート追加』の"
            "リリースノートを書いてください。",
            run_config=RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient())),
        )
        out = str(result.final_output)
        print("--- final output ---")
        print(out)

        # 検証: 遅延ロードでスキルがワークスペースへ materialize され、手順が適用されたか。
        # UnixLocal バックエンドのワークスペースはホストのディレクトリそのものなので、
        # ホスト側から materialize 結果を直接確認できる。
        staged = Path(workspace) / ".agents" / "release-note" / "SKILL.md"
        print("--- verification ---")
        print(f"skill materialized in workspace: {staged.exists()} ({staged})")
        print(f"skill marker in final output   : {SKILL_MARKER in out}")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


async def main() -> None:
    try:
        await asyncio.wait_for(run(), timeout=WATCHDOG_SECONDS)
    except TimeoutError:
        print(f"watchdog: {WATCHDOG_SECONDS}s を超過したため強制終了します", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
