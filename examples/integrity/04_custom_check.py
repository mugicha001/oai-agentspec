"""`checks=` escape hatch: 独自検知関数を差し込む。

`IntegrityCheck = Callable[[], None]` 規約に従い、違反時に `IntegrityError` を raise する
関数を `checks=[...]` に渡す。lib 同梱の標準検証と独自検知を同じ fail-closed フローに混ぜる。

代表ユースケース:
    - manifest 自身の真正性検証（Sigstore / PEP 740 / Ed25519 / GPG）
    - OS の FIM ログ取込（AIDE / Tripwire / OSSEC）
    - Microsoft Agent Governance Toolkit のポリシー発火
    - 業務固有の前提条件チェック

実行:
    uv run python examples/integrity/04_custom_check.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import SAMPLE_APP  # noqa: E402

from oai_agentspec import IntegrityCheck, IntegrityError, lockdown  # noqa: E402


def check_business_config(config_path: Path) -> IntegrityCheck:
    """設定ファイルの存在と必須フィールドを検証する独自 check を返す。"""

    def check() -> None:
        if not config_path.exists():
            raise IntegrityError(f"必須設定が見つかりません: {config_path}")
        text = config_path.read_text(encoding="utf-8")
        if "compliance_version" not in text:
            raise IntegrityError(f"compliance_version が未設定: {config_path}")

    return check


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "config.yaml"
        config.write_text("compliance_version: 1\n", encoding="utf-8")

        # 1. 通る組み合わせ
        lockdown(SAMPLE_APP, libs=False, checks=[check_business_config(config)])
        print("[OK] 独自 check 全件通過")

        # 2. 違反で fail-closed
        config.unlink()
        try:
            lockdown(SAMPLE_APP, libs=False, checks=[check_business_config(config)])
        except IntegrityError as exc:
            print(f"[OK] 独自 check 違反で fail-closed: {exc}")
            return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
