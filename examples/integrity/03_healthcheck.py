"""ヘルスチェック再発火: 同じ引数で `lockdown` を 2 回呼んで擬似的な継続監視を構成。

`lockdown` は冪等。freeze 段は no-op、verify / detect / checks は毎回再実行される。
FastAPI 等の /healthz から `lockdown(...)` を呼べば「呼び出し時点のスナップショット検証」
として機能する。

実行:
    uv run python examples/integrity/03_healthcheck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import SAMPLE_APP, writable_copy  # noqa: E402

from oai_agentspec import (  # noqa: E402
    AgentRegistry,
    AgentSpec,
    IntegrityError,
    RegistryFrozenError,
    lockdown,
)


def main() -> int:
    with writable_copy(SAMPLE_APP) as root:
        registry = AgentRegistry()
        registry.register(AgentSpec(name="triage", instructions="triage"))

        lockdown(root, registry=registry, libs=False)
        print("[OK] 起動時 lockdown")

        lockdown(root, registry=registry, libs=False)
        print("[OK] 2 回目 lockdown（冪等・registry は固定済み）")

        try:
            registry.register(AgentSpec(name="late", instructions="late"))
        except RegistryFrozenError:
            print("[OK] 再 lockdown 後も固定状態を維持")
        else:
            return 1

        # 稼働中に disk 改竄が起きた状況をシミュレート
        (root / "app.py").write_text("# tampered\n", encoding="utf-8")
        try:
            lockdown(root, registry=registry, libs=False)
        except IntegrityError as exc:
            msg = str(exc).split("（")[0]  # noqa: RUF001  全角括弧で分割
            print(f"[OK] ヘルスチェックで改竄検知: {msg}")
            return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
