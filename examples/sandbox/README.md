# sandbox examples

`SandboxAgentSpec`（`AgentSpec` のサブクラス）でサンドボックス実行エージェントを宣言的に扱う例。
SandboxAgent は `agents.Agent` の正式なサブクラスのため、通常の `AgentRegistry` / `HandoffGraph` を
そのまま共用する（Realtime のような専用宣言ルートは不要）。

- `basic_declaration.py`: 宣言 -> 通常 AgentSpec との混在登録 -> ハンドオフグラフ -> validate ->
  get で SandboxAgent を構築するまで（実 API 不要）。
  実行: `uv run python examples/sandbox/basic_declaration.py`
- `local_run.py`: `UnixLocalSandboxClient` + `RunConfig(sandbox=...)` で実際にサンドボックス
  タスク（ファイル作成・確認）を実行する最小例。宣言（`default_manifest` でワークスペースの
  場所を指定）と実行時設定（client 等は `RunConfig`）の分離を示す。実 API が必要。
  実行: `uv run python examples/sandbox/local_run.py`
- `least_privilege.py`: `capabilities=[Filesystem()]` を明示指定してシェル実行なしの最小権限で
  動かす例。実 API が必要。
  実行: `uv run python examples/sandbox/least_privilege.py`

いずれの実 API 例も他の examples と同様 Azure OpenAI を使う（`.env` の `AZURE_OPENAI_ENDPOINT` /
`AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_DEPLOYMENT`。`examples/_shared/_azure.py` 参照）。

## 実行環境（ワークスペース）の場所

ワークスペースの場所は `Manifest.root` が決める（SDK 既定は `"/workspace"`）。実体は
バックエンドで異なる:

| バックエンド | コードが走る場所 | `Manifest.root` の実体 |
|---|---|---|
| `UnixLocalSandboxClient` | ホスト上（サブプロセス実行・**隔離なし**） | ホストのディレクトリそのもの（無ければ作成される） |
| `DockerSandboxClient` | Docker コンテナ内 | コンテナ内パス |

`UnixLocalSandboxClient` は手軽な開発・検証用で、隔離ではなく「ワークスペースのルート制限 +
capabilities による操作制限」である。SDK 既定の `root="/workspace"` のまま macOS 等で動かすと
ルート直下への mkdir で失敗するため、examples は一時ディレクトリを `Manifest(root=...)` で
明示指定している。真の隔離が必要な場合は Docker バックエンドを使う。

## capabilities と最小権限

`capabilities` を未指定（None）にすると SDK 既定（Filesystem / Shell / Compaction 等）に
委ねられ、シェル実行が有効になる。最小権限にしたい場合は `least_privilege.py` のように
必要な capability だけを明示指定する。
