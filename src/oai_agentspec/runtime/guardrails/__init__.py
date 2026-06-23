"""内容ガードレールの公開窓口（`oai-agentspec[guardrails]` extra・agents 非依存・公開 API）。

宣言したエージェントが「何を言うか」を入出力・中間ツール段で検査する helper ファクトリ群を再
エクスポートする。helper は `AgentSpec.input_guardrails` / `output_guardrails` 専用フィールドへ
直接渡せる SDK 互換 `InputGuardrail` / `OutputGuardrail`、または `FunctionTool` へ tool guardrail を
装着したラップ済みツールを返す（`agents.Agent` と同型の宣言面）。

SDK 型（agent / tool 双方の guardrail 型・デコレータ）の import は `_adapters/guardrails.py` に
閉じ、本窓口は plain な検知結果（`Detection`）と接着 helper のみ扱う（SDK 隔離・NFR-1）。重い
専門検知（PII / モデレーション / 注入検知サービス）は lib 非同梱で利用者 DI、既定 helper（注入
ベースライン等）は DI で上書き / 拡張できる。コア `__init__` の `__all__` には載せない（公開 API は
本窓口に集約・helper は実行寄りであり宣言層シンボルのみのコア `__all__` 原則に従う）。

依存ゼロ opt-in extra（`guardrails = []`）のため、本窓口からの import は extra 未導入でも
壊れない。
"""

from __future__ import annotations

from ._detectors import (
    COMMAND_INJECTION_PATTERNS,
    INJECTION_BASELINE_PATTERNS,
    PATH_TRAVERSAL_PATTERNS,
    SQLI_PATTERNS,
    Detection,
)
from .factories import (
    allow_deny_guardrail,
    canary_guardrail,
    external_detector_guardrail,
    guard_tool,
    injection_baseline_guardrail,
    length_guardrail,
    predicate_guardrail,
    prompt_llm_guardrail,
    regex_guardrail,
    tool_guardrail,
)

__all__ = [
    # agent 境界 helper ファクトリ
    "prompt_llm_guardrail",
    "canary_guardrail",
    "predicate_guardrail",
    "regex_guardrail",
    "length_guardrail",
    "allow_deny_guardrail",
    "injection_baseline_guardrail",
    "external_detector_guardrail",
    # ツール境界 helper ファクトリ（tool_guardrail = function_tool 流儀 / guard_tool = 後付け）
    "tool_guardrail",
    "guard_tool",
    # plain 検知結果型
    "Detection",
    # 注入ベースライン既定パターン（DI 上書きの基点・補助検知）
    "INJECTION_BASELINE_PATTERNS",
    "SQLI_PATTERNS",
    "COMMAND_INJECTION_PATTERNS",
    "PATH_TRAVERSAL_PATTERNS",
]
