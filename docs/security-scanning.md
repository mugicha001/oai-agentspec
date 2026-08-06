# ローカル SAST / SCA / シークレットスキャン

本ライブラリは SonarQube による SAST、Trivy による SCA、gitleaks によるシークレット
スキャンをローカルで実行する。CI（GitHub Actions）には組み込まず、コミット時フック
（gitleaks）とリリース前のローカルゲートとして運用する。

本ドキュメントはローカル SAST/SCA/シークレットスキャン手順の Single Source of Truth である。

## 脅威領域の住み分け

本ドキュメントは **install / リリース時** の供給網（コード品質 / 依存 CVE / 漏出シークレット）に対する
責務を負う。**稼働中（プロセス起動後）のディスク上ファイル改竄および宣言グラフへの動的書換** に対する
runtime 側防御は `docs/integrity.md` を参照。両者は別レイヤであり、ローカル SAST/SCA/Secrets と runtime
インテグリティ防御は補完関係にある（install 時はローカルスキャンで防ぎ、稼働中は `lockdown` で検知 /
遮断する）。

## 構成

リポジトリルートの `docker-compose.yml` は次のサービスのみで構成する最小構成である。

| サービス | 役割 |
|---|---|
| `sonarqube` | SonarQube サーバ（SAST 解析結果の登録・Quality Gate 判定） |
| `postgres` | SonarQube のバックエンド DB |
| `sonar-scanner` | `sonar-project.properties` に基づきリポジトリを解析し SonarQube に送信 |
| `trivy-fs` | ファイルシステム + 依存パッケージ（`uv.lock` 直スキャン）の CVE / シークレット / 設定不備スキャン |
| `gitleaks` | git 履歴 + 作業ツリーのシークレットスキャン |

named volume は 5 つ（`sonarqube_data` / `sonarqube_extensions` / `sonarqube_logs` / `postgresql_data` / `trivy-cache`）。
DAST / fuzzing / pentest / observability / trivy-image は含めない。

`sonar-scanner` の image は SonarQube 9.9 LTS と互換の `sonarsource/sonar-scanner-cli:5.0` に固定する
（`latest`=8.x は 9.9 サーバと非互換で `/api/server/version` が 404 になる）。

Trivy のスキャナ名は `vuln,secret,misconfig` に統一する。

## 前提（ホスト設定・一度きり）

SonarQube は内蔵 Elasticsearch を持ち、メモリが不足すると長時間 GC ストールで ES がダウン判定となり
サーバが再起動ループに陥る（スキャンが完走しない）。Docker / Colima のVM割当メモリは
**12GB 以上**を前提とする（SonarQube 単体で約 4GB 推奨、scanner + Postgres 同時稼働を見込む）。

Colima の例:

```sh
colima stop && colima start --memory 12 --cpu 6
```

SonarQube は「サーバ」であり毎回作り直さない。一度起動したら起動したままにし、
解析時は `sonar-scanner` を実行するだけにする。

## 環境変数

ルート `.env.example` に次を記載する。

| 変数 | 用途 |
|---|---|
| `SONAR_TOKEN` | sonar-scanner が SonarQube に認証するためのトークン |
| `SONARQUBE_PORT` | SonarQube サーバの公開ポート |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | SonarQube バックエンド DB 接続情報 |

## sonar-project.properties

| キー | 値 |
|---|---|
| `sonar.projectKey` | プロジェクト識別子 |
| `sonar.sources` | `src/oai_agentspec` |
| `sonar.tests` | `tests` |
| `sonar.exclusions` | `examples/` |
| `sonar.python.version` | `3.12` |
| `sonar.python.coverage.reportPaths` | `coverage.xml` |

New Code 定義は Previous version を用いる。初回解析は overall code を許容する。

## 実行手順

### 1. SonarQube 起動と SONAR_TOKEN 発行（一度きり）

```sh
make security-up   # = docker compose up -d
```

SonarQube の Web UI（`SONARQUBE_PORT`、初期 admin/admin）にアクセスし、My Account → Security
からユーザートークンを発行して `.env` の `SONAR_TOKEN` に設定する。トークンは Postgres ボリュームに
永続するため発行は一度きりでよい。SonarQube サーバは起動したままにする。

### 2. SAST 実行

```sh
make sast
```

`make sast` は coverage.xml 生成（`uv run pytest --cov=oai_agentspec --cov-report=xml`）と
`sonar-scanner` 実行を一本化する。coverage.xml は `sonar.python.coverage.reportPaths` で SonarQube に連携される。
`sonar-project.properties` の `sonar.qualitygate.wait=true` により、スキャナが Quality Gate 確定まで
待機し、NG なら非ゼロ終了する（合否判定の自前ポーリングは不要）。
サーバ起動待ちは compose の `depends_on: sonarqube (service_healthy)` が担保する。

### 3. SCA 実行

```sh
make sca
```

`make sca` は `trivy-fs` で `uv.lock` を直接スキャンする（Trivy のネイティブ uv 対応。requirements.txt の
事前生成は不要）。JSON 出力 + `--exit-code 1 --severity HIGH,CRITICAL` で、HIGH/CRITICAL 検出時に
終了コードで失敗判定する。

#### 期限付き ignore（`.trivyignore.yaml`）

上流依存の制約により当該リポジトリ側では解消できない検出は、`.trivyignore.yaml` に**期限と解除条件を
明記して**登録する（`--ignorefile` でコンテナへ渡す）。ゲートを緑に保ちながら、解消できない債務を
可視化された形で残すための仕組みである。

- **登録してよいもの**: 依存の版を上げれば解消するが、他の依存が課す上限により到達できないもの。
  上限を課している依存鎖と、解除できる条件（どの上流がどう変わればよいか）を `statement` と
  コメントに書く。
- **登録してはならないもの**: 自分たちで版を上げれば解消するもの。これは ignore せず依存を更新する。
- **期限（`expired_at`）**: 期限を過ぎると Trivy は当該 ignore を無効化し、検出が再浮上して
  `make sca` が再び失敗する。これが棚卸しの契機になる。期限到来時は解除条件の成否を確認し、
  解消できるなら依存を更新して ignore を削除、まだ無理なら理由を更新して期限を延ばす。
  フィールド名は snake_case の `expired_at` である。camelCase（`expiredAt`）は未知フィールドとして
  **黙って無視され、ignore が無期限に効いてしまう**。エントリを追加・編集したら、期限を過去日に
  変えて `make sca` が非ゼロ終了することを実測し、期限が実際に効くことを確認する（書式ミスは
  警告が出ないため、実測が唯一の検証手段）。
- **experimental 扱い**: YAML 形式の ignore は Trivy 側で experimental であり、後方互換なく変わり
  うる。`--ignorefile` での明示指定が必須（自動読み込みはされない）。Trivy のバージョンを上げた
  ときは、上記の期限テストで本ファイルが依然有効かを確認する。
- **検知経路の限界**: 上流の新版公開を Dependabot が拾えるかは `pyproject.toml` の pin 幅に依存する。
  major 上限で pin している依存（例: `agent-governance-toolkit[openai-agents]>=4.0,<5`）は、上流が
  次の major を出しても更新候補にならず PR は作られない。この場合、解除の契機は `expiredAt` による
  再浮上のみ（pull 型）であり、期限到来時の手動確認が唯一の検知手段になる。ignore を登録するときは
  この限界が当てはまるかを判断し、当てはまるならその旨をコメントに残す。

## シークレットスキャン（gitleaks）

シークレットの流出を二段構えで防ぐ。

### コミット時フック（予防）

`.pre-commit-config.yaml` で gitleaks フックを定義する。次でフックを有効化する。

```sh
make secrets-install   # = uv run pre-commit install
```

以降、コミット時にステージ差分が自動スキャンされ、シークレットを検出するとコミットが
ブロックされる。

### 全体スキャン（履歴 + 作業ツリー）

```sh
make secrets
```

`make secrets` は docker の gitleaks イメージで git 履歴 + 作業ツリー全体を `detect`
スキャンし、検出時に終了コードで失敗する（`--redact` で値はマスクされる）。

## 判定基準

リリース前に次を満たすこと。

- Trivy の HIGH / CRITICAL 検出が 0 件（`make sca` が終了コード 0 で完了）。
  `.trivyignore.yaml` に期限付きで登録した例外を除く。例外は「上流制約により解消不能」なものに
  限り、登録時点で解除条件と期限を明記する（上記「期限付き ignore」を参照）。期限切れで再浮上した
  検出は未対応の債務として扱い、リリース前に解除条件の成否を確認する。
- SonarQube の Quality Gate が `Passed`。
- gitleaks のシークレット検出が 0 件（`make secrets` が終了コード 0 で完了）。

## 生成物

`coverage.xml` / レポート出力先（`./.security/reports` 等）は `.gitignore` に含め、リポジトリへの混入を防ぐ。レポート出力先は `.gitkeep` でディレクトリを保持する。
