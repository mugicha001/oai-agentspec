"""L1: 最適化実行設定 `OptimizeConfig`（passthrough 設定）を検証する。

すべて Trainer への passthrough で、None は「未指定（Trainer 既定）」。`store` は不透明値を
覗かずに保持する。純データ操作で外部依存なし（`@pytest.mark.unit`）。
"""

from __future__ import annotations

import pytest

from oai_agentspec.runtime.lightning import OptimizeConfig

pytestmark = pytest.mark.unit


def test_optimize_config_defaults_are_none() -> None:
    """passthrough 系は既定 None（未指定 = Trainer 既定に委ねる）。APO モデル名のみ oai-agentspec
    の標準 `gpt-5.4-mini` を既定とする（agent-lightning APO の既定 `gpt-5-mini` / `gpt-4.1-mini`
    を上書き）。"""
    config = OptimizeConfig()
    assert config.concurrency is None
    assert config.rounds is None
    assert config.timeout_seconds is None
    assert config.store is None
    # APO モデル名は oai-agentspec 標準（agent-lightning APO 既定の上書き）。
    assert config.apo_gradient_model == "gpt-5.4-mini"
    assert config.apo_apply_edit_model == "gpt-5.4-mini"
    # tracer 既定は None（_adapters 側で AgentOpsTracer(agentops_managed=True, ...) を構築する）。
    assert config.tracer is None


def test_optimize_config_holds_passthrough_values() -> None:
    """指定値（並列度 / ラウンド数 / タイムアウト）はそのまま保持する。"""
    config = OptimizeConfig(concurrency=4, rounds=3, timeout_seconds=12.5)
    assert config.concurrency == 4
    assert config.rounds == 3
    assert config.timeout_seconds == pytest.approx(12.5)


def test_optimize_config_store_is_opaque_passthrough() -> None:
    """store は中身を覗かない不透明値としてそのまま保持する。"""
    sentinel = object()
    config = OptimizeConfig(store=sentinel)
    assert config.store is sentinel


def test_optimize_config_is_frozen() -> None:
    """OptimizeConfig は frozen dataclass で属性再代入できない。"""
    config = OptimizeConfig(concurrency=1)
    with pytest.raises((AttributeError, TypeError)):
        config.concurrency = 2  # type: ignore[misc]


# ----------------------------------------------------------------------
# skip_coverage_check（Issue #47 Phase 1: pre-flight opt-out フラグ）
# ----------------------------------------------------------------------


def test_optimize_config_skip_coverage_check_default_false() -> None:
    """OptimizeConfig の skip_coverage_check は既定 False（pre-flight 有効）。"""
    assert OptimizeConfig().skip_coverage_check is False


def test_optimize_config_skip_coverage_check_can_be_true() -> None:
    """skip_coverage_check=True で opt-out できる。"""
    assert OptimizeConfig(skip_coverage_check=True).skip_coverage_check is True


def test_optimize_config_apo_api_defaults_none() -> None:
    """`apo_api` は既定 None（auto = Responses 優先 + 404 fallback・現行互換）。"""
    from oai_agentspec.runtime.lightning import OptimizeConfig

    assert OptimizeConfig().apo_api is None
    assert OptimizeConfig(apo_api="chat_completions").apo_api == "chat_completions"
