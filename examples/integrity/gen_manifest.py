"""`<root>/.integrity/sha256.manifest` を生成する helper。

GNU coreutils `sha256sum` 互換フォーマット（`<sha256>  <relative-path>` 1 行 / 1 ファイル）
を出力する。実運用では `sha256sum` を直接使ってもよいが、CI で純 Python で生成したい場合や
本 example 群を実行する前提として利用する。

使い方:
    uv run python examples/integrity/gen_manifest.py <root>

`<root>` 直下に `.integrity/sha256.manifest` を作成する（既存があれば上書き）。
通常ファイル・シンボリックリンクのみが対象。`__pycache__` / `.integrity/` は除外。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def _excluded(path: Path, root: Path) -> bool:
    """root 配下の relative path で `__pycache__` / `.integrity/` を除外。"""
    rel_parts = path.relative_to(root).parts
    return "__pycache__" in rel_parts or rel_parts[0] == ".integrity"


def generate_manifest(root: Path) -> Path:
    """root 配下の通常ファイルを sha256 で manifest 化し、`.integrity/sha256.manifest` に書く。"""
    integrity_dir = root / ".integrity"
    integrity_dir.mkdir(exist_ok=True)
    manifest_path = integrity_dir / "sha256.manifest"

    lines: list[str] = []
    for entry in sorted(root.rglob("*")):
        if not entry.is_file():
            continue
        if _excluded(entry, root):
            continue
        digest = hashlib.sha256(entry.read_bytes()).hexdigest()
        rel = entry.relative_to(root).as_posix()
        lines.append(f"{digest}  {rel}")

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python gen_manifest.py <root>", file=sys.stderr)
        sys.exit(2)
    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"root が存在しないかディレクトリではない: {root}", file=sys.stderr)
        sys.exit(2)
    manifest = generate_manifest(root)
    print(f"manifest を生成: {manifest}")


if __name__ == "__main__":
    main()
