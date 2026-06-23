# oai-agentspec 開発タスク
#
# テスト・lint・ローカル SAST/SCA を一本化する。
# SAST/SCA は CI には組み込まず、リリース前にローカルで実行する運用とする。
#
# docker-compose.yml はリポジトリルートに置き、相対バインドマウント（./src ./uv.lock 等）が
# ルート基準で解決されるようにする（docker compose が自動検出する）。

COMPOSE := docker compose

.PHONY: test lint format sast sca secrets secrets-install security-up security-down clean

test:
	uv run pytest

lint:
	uv run ruff check src tests examples
	uv run ruff format --check src tests examples

format:
	uv run ruff format src tests examples
	uv run ruff check --fix src tests examples

# SonarQube 起動（初回は数分。http://localhost:9000）
security-up:
	$(COMPOSE) up -d

security-down:
	$(COMPOSE) down

# SAST: SonarScanner（要 SONAR_TOKEN。SonarQube 起動 + カバレッジ生成が前提）
sast:
	uv run pytest --cov=oai_agentspec --cov-report=xml
	$(COMPOSE) --profile scanner run --rm sonar-scanner

# SCA: Trivy（uv.lock を直接スキャン。HIGH/CRITICAL 検出で非ゼロ終了）
sca:
	$(COMPOSE) --profile security run --rm trivy-fs

# シークレットスキャン: gitleaks（git 履歴 + 作業ツリー。検出で非ゼロ終了）
secrets:
	$(COMPOSE) --profile secrets run --rm gitleaks

# pre-commit フック（gitleaks）を有効化する。コミット時にステージ差分を自動スキャン
secrets-install:
	uv run pre-commit install

clean:
	rm -f coverage.xml
	rm -rf htmlcov .coverage .security/reports/*.json
