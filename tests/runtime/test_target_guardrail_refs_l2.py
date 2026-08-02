"""L2: 評価 / 最適化 target の正規化が guardrail 名前参照を silent に落とさないことを検証する。

`llmops` / `lightning` の `normalize()` は target が単体 `AgentSpec` のとき registry を経由せず
`_adapters.build_agent(spec)` を直呼びする。この経路では `AgentSpec.guardrails`（名前参照）を
解決する `GuardrailProvider` が存在しないため、宣言が丸ごと無視されうる。専用フィールド
（`input_guardrails` / `output_guardrails`）は builder が転送するため適用される非対称があり、
「宣言したのに何も検査されない」状態を例外も警告もなく作らないことを pin する。
"""

from __future__ import annotations

import pytest

from oai_agentspec import AgentSpec
from oai_agentspec.runtime.guardrails import canary_guardrail
from oai_agentspec.runtime.lightning import _target as lightning_target
from oai_agentspec.runtime.llmops import _target as llmops_target

pytestmark = pytest.mark.integration

_MODULES = [llmops_target, lightning_target]
_IDS = ["llmops", "lightning"]


@pytest.mark.parametrize("module", _MODULES, ids=_IDS)
def test_単体AgentSpec_targetは名前参照宣言をValueErrorで拒否する(module: object) -> None:
    """`guardrails` を宣言した単体 `AgentSpec` を target にすると `ValueError` になる。

    silent に落とすと、評価・最適化の測定対象に安全制御が存在しないまま結果を信頼してしまう
    （OWASP LLM09 の過剰依存）。registry 経由へ誘導する文言を含めて fail-closed にする。
    """
    spec = AgentSpec(name="a", instructions="x", guardrails=["canary"])
    with pytest.raises(ValueError, match="AgentRegistry"):
        module.normalize(spec, None)


@pytest.mark.parametrize("module", _MODULES, ids=_IDS)
def test_単体AgentSpec_targetは専用フィールドならそのまま通る(module: object) -> None:
    """実体を専用フィールドへ渡す従来経路は非破壊（builder が転送するため適用される）。"""
    spec = AgentSpec(
        name="a", instructions="x", output_guardrails=[canary_guardrail("LEAK", name="canary")]
    )
    agent, _ = module.normalize(spec, None)
    assert [g.get_name() for g in agent.output_guardrails] == ["canary"]


@pytest.mark.parametrize("module", _MODULES, ids=_IDS)
def test_単体AgentSpec_targetは名前参照が空なら通る(module: object) -> None:
    """`guardrails` が空（既定）なら従来どおり build できる（既存経路の非破壊）。"""
    agent, _ = module.normalize(AgentSpec(name="a", instructions="x"), None)
    assert agent.name == "a"
