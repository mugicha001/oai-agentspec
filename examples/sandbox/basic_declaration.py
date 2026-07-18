"""SandboxAgentSpec を宣言し AgentRegistry で SandboxAgent を構築する最小例。

SandboxAgent は `agents.Agent` の正式なサブクラスであるため、Realtime のような
専用宣言ルートは不要で、通常の `AgentRegistry` / `HandoffGraph` をそのまま共用する:

    SandboxAgentSpec 宣言 -> 通常 AgentSpec と同一 registry へ混在登録
    -> HandoffGraph でトポロジ宣言 -> validate
    -> get で SandboxAgent を遅延構築（handoffs は 2 パスで後付け結線）

サンドボックス固有の 4 フィールド（default_manifest / capabilities / run_as /
base_instructions）は未指定（None）なら SDK 既定に委ねられる。`capabilities` の
SDK 既定はシェル実行を含む機能群を有効化しうるため、最小権限にしたい場合は
明示指定する（least_privilege.py 参照）。

実 API へは接続しない。宣言と build-time 検証・構築までで完結し、構築された
SandboxAgent のフィールドを print で確認するだけの例。

サンドボックスの実行時設定（クライアント・セッション・スナップショット）は spec の
責務ではなく、実行時に `RunConfig(sandbox=...)` へ渡す（local_run.py 参照）。

実行:
    uv run python examples/sandbox/basic_declaration.py
"""

from __future__ import annotations

from agents.sandbox import Manifest, SandboxAgent

from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph, SandboxAgentSpec


def build_registry() -> tuple[AgentRegistry, HandoffGraph]:
    """通常 AgentSpec と SandboxAgentSpec を混在登録した registry を返す。

    Returns:
        validate 済みの (AgentRegistry, HandoffGraph)。entry は triage。
    """
    registry = AgentRegistry()

    # 通常のエージェント（受付）。
    registry.register(
        AgentSpec(
            name="triage",
            instructions="依頼を聞き取り、コード実行が必要なら code_runner へ引き継ぐ受付担当。",
        )
    )
    # サンドボックス実行エージェント。default_manifest でワークスペースの場所を宣言する。
    # root="/workspace" は Docker バックエンド（コンテナ内パス）前提の SDK 既定値。
    # UnixLocal（ホスト実行）で動かす場合はホスト上の書き込み可能なディレクトリを
    # 指定する必要がある（local_run.py は一時ディレクトリを渡している）。
    registry.register(
        SandboxAgentSpec(
            name="code_runner",
            instructions="サンドボックス内でコードやコマンドを実行し、結果を報告する担当。",
            default_manifest=Manifest(root="/workspace"),
            run_as="worker",
            base_instructions=None,  # None なら SDK のサンドボックス既定プロンプトに委ねる
        )
    )

    # 相互ハンドオフ（triage <-> code_runner の循環）。registry の 2 パス遅延バインドが解決する。
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "code_runner", description="コード実行の依頼を引き継ぐ")
    graph.edge("code_runner", "triage")
    graph.apply(registry)
    registry.validate()  # handoffs 参照のタイポを run 前に検出

    return registry, graph


def main() -> None:
    registry, graph = build_registry()

    print("--- handoff topology (mermaid) ---")
    print(graph.mermaid())

    triage = registry.get("triage")
    runner = registry.get("code_runner")

    print("--- built agents ---")
    print(f"triage      : {type(triage).__name__}")
    print(f"code_runner : {type(runner).__name__}")
    assert isinstance(runner, SandboxAgent)

    print("--- sandbox fields (code_runner) ---")
    print(f"default_manifest.root : {runner.default_manifest.root}")
    print(f"run_as                : {runner.run_as}")
    # capabilities は未指定なので SDK 既定（Filesystem / Shell / Compaction 等）が入る。
    print(f"capabilities (SDK 既定): {[type(c).__name__ for c in runner.capabilities]}")
    # handoffs には Handoff（agent_name あり）と Agent 直参照が混在しうる
    names = [getattr(h, "agent_name", None) or getattr(h, "name", "?") for h in runner.handoffs]
    print(f"handoffs              : {names}")


if __name__ == "__main__":
    main()
