"""SandboxAgentSpec で宣言したエージェントを UnixLocal サンドボックスで実行する例。

宣言（lib の責務）と実行時設定（SDK `RunConfig` の責務）の分離を示す:

    SandboxAgentSpec 宣言（default_manifest でワークスペースの場所を指定）
    -> registry.get で SandboxAgent を構築
    -> Runner.run(..., run_config=RunConfig(sandbox=SandboxRunConfig(client=...)))

ワークスペースの場所は `Manifest.root` が決める（SDK 既定は "/workspace"）。
`UnixLocalSandboxClient` は **ホスト上でコマンドを直接実行する**バックエンドで、
ワークスペースはホストのディレクトリそのもの（隔離ではなくルート制限 + capabilities に
よる操作制限）。本例は使い捨ての一時ディレクトリを root に指定する。真の隔離が必要な
場合は Docker バックエンド（`DockerSandboxClient`。コンテナ内 /workspace）を使う。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/sandbox/local_run.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

from agents import RunConfig, Runner
from agents.sandbox import Manifest, SandboxRunConfig
from agents.sandbox.sandboxes import UnixLocalSandboxClient

from oai_agentspec import AgentRegistry, SandboxAgentSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402

# 従量課金 API に接続するため、開始からの絶対上限を script 内 watchdog で強制する
# （想定所要 60s + マージン。macOS 標準に timeout コマンドが無いため script 内で完結させる）。
WATCHDOG_SECONDS = 90


def build_agent(workspace_root: str):
    """一時ディレクトリをワークスペースにする SandboxAgent を宣言・構築する。

    Args:
        workspace_root: サンドボックスのワークスペースにするホスト上のパス。

    Returns:
        構築済みの SandboxAgent。
    """
    registry = AgentRegistry()
    registry.register(
        SandboxAgentSpec(
            name="code_runner",
            instructions=(
                "あなたはサンドボックス内で作業するアシスタント。"
                "指示されたファイル操作やコマンド実行をワークスペース内で行い、"
                "結果を簡潔に報告する。"
            ),
            model=azure_model(),
            # 宣言側でワークスペースの場所を指定する（実行時の RunConfig でも上書き可能）。
            default_manifest=Manifest(root=workspace_root),
        )
    )
    return registry.get("code_runner")


async def run() -> None:
    workspace = tempfile.mkdtemp(prefix="oai-agentspec-sandbox-")
    try:
        agent = build_agent(workspace)
        result = await Runner.run(
            agent,
            "ワークスペース直下に hello.txt を作り、中身を 'hello from sandbox' にして、"
            "その後ファイル一覧と hello.txt の中身を確認して報告してください。",
            run_config=RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient())),
        )
        print("--- final output ---")
        print(result.final_output)

        # ホスト側からワークスペースの実体を検証する（UnixLocal はホストのディレクトリ）。
        hello = Path(workspace) / "hello.txt"
        print("--- host-side verification ---")
        print(f"workspace : {workspace}")
        print(f"hello.txt : exists={hello.exists()}")
        if hello.exists():
            print(f"content   : {hello.read_text(encoding='utf-8').strip()}")
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
