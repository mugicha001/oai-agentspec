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
- `skills.py`: `capabilities=[Shell(), Skills(lazy_from=...)]` でスキル（`SKILL.md`）を
  与える例。実 API が必要。
  実行: `uv run python examples/sandbox/skills.py`

いずれの実 API 例も他の examples と同様 Azure OpenAI を使う（`.env` の `AZURE_OPENAI_ENDPOINT` /
`AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_DEPLOYMENT`。`examples/_shared/_azure.py` 参照）。

sandbox の capabilities は SDK の hosted tool（`apply_patch` 等）を伴うため、**Responses API に
対応したプロバイダが必要**。Chat Completions 系のゲートウェイ（`OPENAI_API_STYLE=chat_completions`）
では `Hosted tools are not supported with the ChatCompletions API` で実行できない。

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

## スキル（SKILL.md）を使う

スキルを使う本線は SandboxAgent + `Skills` capability（`skills.py`）。SDK がスキルフォルダの
発見・システムプロンプトへの一覧注入・遅延ロード用ツール（`load_skill`）の生成まで担うため、
利用側にローダー実装は要らない。`LocalDirLazySkillSource(source=LocalDir(src=...))` が指すのは
**ホスト上の**スキルフォルダで、スキル本文の置き場所と実行環境（サンドボックスセッション）は
分離されている。実行環境を Docker バックエンドへ差し替えてもスキルはホスト側に置いたままでよい。

`Skills` 単体ではスキルを読めない。`Skills` が提供するのは `load_skill`（materialize）だけで、
materialize 済み `SKILL.md` を読む手段は別 capability が要る。`Filesystem` のツールは
`view_image` / `apply_patch` で読み取りを含まないため、`Shell`（`exec_command`）を併せて
与える。

通常 `Agent` でも `ShellTool` の `environment["skills"]` 経由でスキルを渡せる
（`examples/basic/shell_tool_skills.py`）が、そちらは SDK がフォルダ発見もプロンプト注入も
提供しないため、メタデータを組み立てるローダーを利用側で自作する必要がある。
