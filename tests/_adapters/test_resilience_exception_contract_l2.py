"""L2: SDK 例外の構造契約トリップワイヤ（`AgentsException` / `RunErrorDetails`）。

`runtime/resilience/_failsafe.py` の `_read_run_data_last_agent` が duck typing
（`getattr(exc, "run_data")` -> `getattr(run_data, "last_agent")`）で読む SDK 側の契約を、
SDK upgrade で silent に壊れないよう構造として pin する。duck typing は読み取り先が
消えても例外にせず None を返して着地を成立させるため、`last_agent` が常に解決不能に
なる退行が build-time にも実行時例外にも現れない。ここで SDK 実型を直接検査して CI で
fail させる（L1 側の `tests/runtime/resilience/test_failsafe_l1.py` は fake で解決規則を
pin しており、実型の構造は本モジュールが担う）。

pin する契約:

- `AgentsException.__init__` が `run_data` を None 初期化する（Runner 外で構築した例外
  でも `run_data` 属性が存在し、1 つ目の読み取り先が None を返せる）。
- `RunErrorDetails` が `last_agent` フィールドを持つ（1 つ目の読み取り先の解決対象）。
- `AgentsException` が `Exception` サブクラスである（`FailsafePolicy.handlers` のキー
  として宣言できる = 着地対象にできる）。
"""

from __future__ import annotations

import dataclasses

import pytest
from agents.exceptions import AgentsException, RunErrorDetails

pytestmark = pytest.mark.integration


def test_sdk_AgentsExceptionのrun_dataは既定でNone() -> None:
    """`AgentsException` は `run_data` 属性を持ち、未設定時は None で初期化される。"""
    exc = AgentsException("x")

    assert hasattr(exc, "run_data")
    assert exc.run_data is None


def test_sdk_RunErrorDetailsはlast_agentフィールドを持つ() -> None:
    """`RunErrorDetails` は dataclass で、`last_agent` フィールドを持つ。"""
    assert dataclasses.is_dataclass(RunErrorDetails)

    field_names = {f.name for f in dataclasses.fields(RunErrorDetails)}

    assert "last_agent" in field_names


def test_sdk_AgentsExceptionはException派生である() -> None:
    """SDK 例外は `Exception` 派生のため handlers のキーとして宣言できる。"""
    assert issubclass(AgentsException, Exception)
