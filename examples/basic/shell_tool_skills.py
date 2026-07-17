"""通常の AgentSpec で ShellTool のスキル（environment.skills）を使う例。

スキル機構はサンドボックス専用ではない。SDK の `ShellTool`（次世代 shell ツール）は
通常の `Agent` に渡せる Tool であり、`environment` にスキル（名前・説明・置き場所）を
宣言すると、モデルはシェル経由でスキル本文（SKILL.md）を読んで手順に従う:

    AgentSpec(tools=[ShellTool(executor=..., environment={"type": "local", "skills": [...]})])

スキルの実体は `examples/skills/<スキル名>/SKILL.md`（frontmatter に name / description、
本文に手順）。この `<フォルダ>/SKILL.md` レイアウトは SDK の sandbox `Skills` capability が
実行時発見に使う形式と同一で、`LocalDirLazySkillSource` で SandboxAgentSpec からも同じ
フォルダを共有できる。

本例のスキル本文には「回答へ固有マーカーを含める」という指示が入っている。実行後に
(1) executor が記録したコマンドログに SKILL.md の読み取りが現れること、(2) 最終回答に
マーカーが含まれること、の 2 点で「スキルが本当に読まれて適用された」ことを機械的に
確認する。

local 環境の executor はホスト上でコマンドを直接実行する（隔離なし）。本例は実行時の
作業ディレクトリを一時ディレクトリに限定した最小実装とする。隔離が必要な用途は
ホスト型コンテナ環境（environment type: container_auto）または SandboxAgentSpec
（examples/sandbox/）を使う。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/basic/shell_tool_skills.py
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from agents import (
    Runner,
    ShellCallOutcome,
    ShellCommandOutput,
    ShellCommandRequest,
    ShellResult,
    ShellTool,
)

from oai_agentspec import AgentRegistry, AgentSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402

# 従量課金 API に接続するため、開始からの絶対上限を script 内 watchdog で強制する
# （想定所要 60s + マージン。macOS 標準に timeout コマンドが無いため script 内で完結させる）。
WATCHDOG_SECONDS = 90

SKILLS_ROOT = Path(__file__).resolve().parent.parent / "skills"
SKILL_MARKER = "SKILLMARK-42"


def load_local_skills(root: Path) -> list[dict[str, str]]:
    """`<root>/<スキル名>/SKILL.md` を走査し ShellTool の skills メタデータへ変換する。

    frontmatter（`---` 区切り）の `name:` / `description:` を読み取り、
    `{"name", "description", "path"}` のリストを返す。frontmatter が無い・不完全な
    フォルダはスキップする。

    Args:
        root: スキルフォルダ群のルートディレクトリ。

    Returns:
        ShellTool の `environment["skills"]` にそのまま渡せる dict のリスト。
    """
    skills: list[dict[str, str]] = []
    for skill_md in sorted(root.glob("*/SKILL.md")):
        meta: dict[str, str] = {}
        lines = skill_md.read_text(encoding="utf-8").splitlines()
        if lines and lines[0].strip() == "---":
            for line in lines[1:]:
                if line.strip() == "---":
                    break
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
        if "name" in meta and "description" in meta:
            skills.append(
                {
                    "name": meta["name"],
                    "description": meta["description"],
                    "path": str(skill_md.parent),
                }
            )
    return skills


def make_executor(workdir: Path, command_log: list[str]):
    """一時ディレクトリを cwd にコマンドを実行し、実行内容を記録する executor を作る。

    Args:
        workdir: コマンド実行の作業ディレクトリ。
        command_log: 実行されたコマンド文字列を追記するリスト（スキル読取の検証用）。

    Returns:
        ShellTool に渡す executor callable。
    """

    def executor(request: ShellCommandRequest) -> ShellResult:
        outputs: list[ShellCommandOutput] = []
        for command in request.data.action.commands:
            command_log.append(command)
            proc = subprocess.run(  # noqa: S603 - 例示用の最小 local executor
                command,
                shell=True,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            outputs.append(
                ShellCommandOutput(
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    outcome=ShellCallOutcome(type="exit", exit_code=proc.returncode),
                    command=command,
                )
            )
        return ShellResult(output=outputs)

    return executor


async def run() -> None:
    skills = load_local_skills(SKILLS_ROOT)
    print(f"--- discovered skills ({SKILLS_ROOT}) ---")
    for s in skills:
        print(f"{s['name']}: {s['description']}")

    workdir = Path(tempfile.mkdtemp(prefix="oai-agentspec-skills-"))
    command_log: list[str] = []
    try:
        environment: dict[str, Any] = {"type": "local", "skills": skills}
        registry = AgentRegistry()
        registry.register(
            AgentSpec(
                name="release_writer",
                instructions=(
                    "あなたはリリースノート担当。スキルが提供されている場合は、"
                    "必ずスキルの SKILL.md を読んでから、その手順に従って回答する。"
                ),
                model=azure_model(),
                tools=[
                    ShellTool(executor=make_executor(workdir, command_log), environment=environment)
                ],
            )
        )
        agent = registry.get("release_writer")

        result = await Runner.run(
            agent,
            "release-note スキルに従って、『サンドボックスエージェントの宣言的サポート追加』の"
            "リリースノートを書いてください。",
        )
        out = str(result.final_output)
        print("--- final output ---")
        print(out)

        # スキルが本当に使われたかの機械的検証。一次証拠は run の実行トランスクリプト
        # （result.new_items）そのもの:
        # (1) tool_call_item に SKILL.md を読むシェル呼び出しが記録されている
        # (2) tool_call_output_item（ツール実行結果としてモデルへ返った内容）に
        #     SKILL.md にだけ書いたマーカーが含まれる = スキル本文がモデルに届いた
        # (3) 最終回答がマーカーで始まる = 届いた手順が適用された
        # executor 側のコマンドログは補助証拠（ツールが呼ばれない限り 1 行も記録されない）。
        calls = [str(i.raw_item) for i in result.new_items if i.type == "tool_call_item"]
        tool_outputs = [
            str(i.output) for i in result.new_items if i.type == "tool_call_output_item"
        ]
        print("--- verification (from result.new_items) ---")
        print(f"shell call reads SKILL.md   : {any('SKILL.md' in c for c in calls)}")
        print(f"skill body in tool output   : {any(SKILL_MARKER in o for o in tool_outputs)}")
        print(f"skill marker in final output: {SKILL_MARKER in out}")
        print("--- verification (executor log) ---")
        print(f"executed commands           : {[shlex.split(c)[0] for c in command_log]}")
        print(f"skill file read via shell   : {any('SKILL.md' in c for c in command_log)}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def main() -> None:
    try:
        await asyncio.wait_for(run(), timeout=WATCHDOG_SECONDS)
    except TimeoutError:
        print(f"watchdog: {WATCHDOG_SECONDS}s を超過したため強制終了します", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
