"""examples の import-safety smoke test（Issue #46 #5）。

`examples/lightning/03_graph_apo.py` を実際に import し、`build_registry()` が正常に
`(AgentRegistry, HandoffGraph)` を返すことを検証する。`main()` は呼ばない（実 API 呼び出しを
避ける）。`tests/conftest.py` の autouse ネットワークガードが二重の安全網として働く。

対象ファイル名が `03_graph_apo.py` のように数字始まりで通常の `from examples...` import が
`SyntaxError` になるため、`importlib.import_module` 経由で読み込む。
"""

from __future__ import annotations

import importlib

from oai_agentspec import AgentRegistry, HandoffGraph

from _helpers.fake_model import FakeModel


def test_example_builds_registry_without_network() -> None:
    """`build_registry(FakeModel())` は実 API を触らずに完結する。

    `main()` は呼ばないため `azure_model` / `azure_client` は起動されない。import-safety と
    `build_registry` の構造的健全性のみを pin する。
    """
    module = importlib.import_module("examples.lightning.03_graph_apo")
    registry, graph = module.build_registry(FakeModel())

    assert isinstance(registry, AgentRegistry)
    assert isinstance(graph, HandoffGraph)
