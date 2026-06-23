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
- SonarQube の Quality Gate が `Passed`。
- gitleaks のシークレット検出が 0 件（`make secrets` が終了コード 0 で完了）。

## 生成物

`coverage.xml` / レポート出力先（`./.security/reports` 等）は `.gitignore` に含め、リポジトリへの混入を防ぐ。レポート出力先は `.gitkeep` でディレクトリを保持する。
