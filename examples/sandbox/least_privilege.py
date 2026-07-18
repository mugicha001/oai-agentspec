"""capabilities を明示指定して最小権限のサンドボックスエージェントを実行する例。

`SandboxAgentSpec.capabilities` を未指定（None）にすると SDK 既定
（Filesystem / Shell / Compaction 等）に委ねられ、シェル実行が黙って有効になる。
最小権限にしたい場合は capabilities を明示指定する。本例は Filesystem のみを与え、
シェル実行なしでファイル操作タスクを完了させる。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/sandbox/least_privilege.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

from agents import RunConfig, Runner
from agents.sandbox import Manifest, SandboxRunConfig
from agents.sandbox.capabilities import Filesystem
from agents.sandbox.sandboxes import UnixLocalSandboxClient

from oai_agentspec import AgentRegistry, SandboxAgentSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402

# 従量課金 API に接続するため、開始からの絶対上限を script 内 watchdog で強制する
# （想定所要 60s + マージン。macOS 標準に timeout コマンドが無いため script 内で完結させる）。
WATCHDOG_SECONDS = 90


async def run() -> None:
    workspace = tempfile.mkdtemp(prefix="oai-agentspec-sandbox-")
    try:
        registry = AgentRegistry()
        registry.register(
            SandboxAgentSpec(
                name="file_worker",
                instructions=(
                    "あなたはワークスペース内のファイル操作だけを行うアシスタント。"
                    "指示されたファイルを作成・確認して結果を簡潔に報告する。"
                ),
                model=azure_model(),
                default_manifest=Manifest(root=workspace),
                # 最小権限: ファイル操作のみ許可（Shell を与えない）。
                capabilities=[Filesystem()],
            )
        )
        agent = registry.get("file_worker")
        print(f"capabilities: {[type(c).__name__ for c in agent.capabilities]}")

        result = await Runner.run(
            agent,
            "ワークスペース直下に notes.txt を作り、中身を 'least privilege' にして、"
            "作成できたことを確認して報告してください。シェルコマンドは使わないでください。",
            run_config=RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient())),
        )
        print("--- final output ---")
        print(result.final_output)

        notes = Path(workspace) / "notes.txt"
        print("--- host-side verification ---")
        print(f"notes.txt : exists={notes.exists()}")
        if notes.exists():
            print(f"content   : {notes.read_text(encoding='utf-8').strip()}")
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
