"""L1: 評価実行設定 `EvaluationConfig` の純検証（外部依存なし）。

既定値（実行系設定 / fail-closed ポリシー / テレメトリ opt-out）の保持と、bool フィールド
`deepeval_telemetry_opt_out` の構築時型検証を pin する。DeepEval / agents 非依存。
"""

from __future__ import annotations

import re

import pytest

from oai_agentspec.runtime.llmops import EvaluationConfig

pytestmark = pytest.mark.unit


def test_evaluation_config_defaults() -> None:
    """既定は timeout 60 秒 / 逐次 / required 未指定 / テレメトリ opt-out True。"""
    config = EvaluationConfig()
    assert config.timeout_seconds == pytest.approx(60.0)
    assert config.concurrency == 1
    assert config.required_criteria is None
    assert config.deepeval_telemetry_opt_out is True


# ----------------------------------------------------------------------
# deepeval_telemetry_opt_out の構築時 bool 型検証
# ----------------------------------------------------------------------


def test_evaluation_config_telemetry_opt_out_none_raises() -> None:
    """deepeval_telemetry_opt_out=None は bool でないため ValueError（メッセージ全文を pin）。

    テレメトリ抑止フラグが黙って falsy になる（外部送信が有効化される）silent failure を排除する。
    """
    with pytest.raises(
        ValueError,
        match=re.escape("deepeval_telemetry_opt_out must be a bool, got 'NoneType'"),
    ):
        EvaluationConfig(deepeval_telemetry_opt_out=None)  # type: ignore[arg-type]


def test_evaluation_config_telemetry_opt_out_str_raises() -> None:
    """deepeval_telemetry_opt_out="no" は truthy な文字列だが ValueError で弾く。"""
    with pytest.raises(
        ValueError, match=re.escape("deepeval_telemetry_opt_out must be a bool, got 'str'")
    ):
        EvaluationConfig(deepeval_telemetry_opt_out="no")  # type: ignore[arg-type]


def test_evaluation_config_telemetry_opt_out_int_zero_raises() -> None:
    """deepeval_telemetry_opt_out=0（int）は bool でないため ValueError。"""
    with pytest.raises(ValueError, match="deepeval_telemetry_opt_out"):
        EvaluationConfig(deepeval_telemetry_opt_out=0)  # type: ignore[arg-type]


def test_evaluation_config_telemetry_opt_out_bool_constructs() -> None:
    """True / False を渡した構築は従来どおり成功する（正常系の維持）。"""
    assert EvaluationConfig(deepeval_telemetry_opt_out=True).deepeval_telemetry_opt_out is True
    assert EvaluationConfig(deepeval_telemetry_opt_out=False).deepeval_telemetry_opt_out is False
