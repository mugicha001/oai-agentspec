"""integrity example 共有ヘルパ: tmp ディレクトリへの書込可能コピー。

example が `lockdown` を呼ぶ前に「サンプルディレクトリを tmp にコピーして書込可能にし、
改竄シミュレートをする」というパターンを共通化する。`sample_app/` や `sample_prompts/` 自体は
git 同梱の read-only サンプル（manifest 同梱済み）であり、example はこれをコピーして使う。
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent
SAMPLE_APP = SAMPLES_DIR / "sample_app"
SAMPLE_PROMPTS = SAMPLES_DIR / "sample_prompts"


@contextmanager
def writable_copy(source: Path) -> Iterator[Path]:
    """source ディレクトリを tmp dir にコピーして書込可能な Path を yield する。

    example で「サンプルを改竄して fail-closed を確認する」用途に使う。
    `with writable_copy(SAMPLE_APP) as root:` の形で利用。
    """
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / source.name
        shutil.copytree(source, dst)
        yield dst
