"""examples の import-safety smoke test（Issue #46 #5、#70 F-23）。

`examples/lightning/03_graph_apo.py` を実際に import し、`build_registry()` が正常に
`(AgentRegistry, HandoffGraph)` を返すことを検証する。`main()` は呼ばない（実 API 呼び出しを
避ける）。`tests/conftest.py` の autouse ネットワークガードが二重の安全網として働く。

対象ファイル名が `03_graph_apo.py` のように数字始まりで通常の `from examples...` import が
`SyntaxError` になるため、`importlib.import_module` 経由で読み込む。

`examples/agent_names/` `examples/deterministic/` は `DeterministicResponseModel` のみを
使い実 API へ到達しないため、`main()` の実行まで完走することを検証する
（Issue #70 で追加された 5 本の smoke カバレッジ）。
"""

from __future__ import annotations

import asyncio
import importlib

import pytest

from oai_agentspec import AgentRegistry, HandoffGraph

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.unit

_SYNC_MAIN_EXAMPLES = ["examples.agent_names.01_declarative_names"]
_ASYNC_MAIN_EXAMPLES = [
    "examples.deterministic.01_rule_model",
    "examples.deterministic.02_multi_turn_and_handoff",
    "examples.deterministic.03_streaming",
    "examples.deterministic.04_tool_and_handoff_in_one_rule",
]


def test_example_builds_registry_without_network() -> None:
    """`build_registry(FakeModel())` は実 API を触らずに完結する。

    `main()` は呼ばないため `azure_model` / `azure_client` は起動されない。import-safety と
    `build_registry` の構造的健全性のみを pin する。
    """
    module = importlib.import_module("examples.lightning.03_graph_apo")
    registry, graph = module.build_registry(FakeModel())

    assert isinstance(registry, AgentRegistry)
    assert isinstance(graph, HandoffGraph)


@pytest.mark.parametrize("module_name", _SYNC_MAIN_EXAMPLES)
def test_同期_main_の_example_が_実_API_無しで完走する(
    module_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`examples/agent_names/01_declarative_names.py` は同期 `main()` を持つ。

    import から `main()` の完走まで通し、公開ビルダ・公開シンボルの参照が壊れていないことを
    pin する（例: `AgentNames` / `validate_agent_names` の参照先が壊れると `main()` が
    `AttributeError`/`ImportError` で失敗する）。
    """
    module = importlib.import_module(module_name)

    module.main()

    captured = capsys.readouterr()
    assert captured.out != ""


@pytest.mark.parametrize("module_name", _ASYNC_MAIN_EXAMPLES)
def test_非同期_main_の_example_が_実_API_無しで完走する(
    module_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`examples/deterministic/01〜04` は `async def main()` を持つ。

    `DeterministicResponseModel` のみを使うため実 API へ到達せず、`asyncio.run` 相当で
    `main()` を完走できる。応答ビルダ（`text_response` 等）の参照が壊れると
    `ImportError`/`AttributeError` で本テストが RED になる。
    """
    module = importlib.import_module(module_name)

    asyncio.run(module.main())

    captured = capsys.readouterr()
    assert captured.out != ""
