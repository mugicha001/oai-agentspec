"""内容ガードレールの公開窓口（`oai-agentspec[guardrails]` extra・agents 非依存・公開 API）。

宣言したエージェントが「何を言うか」を入出力・中間ツール段で検査する helper ファクトリ群を再
エクスポートする。helper は `AgentSpec.input_guardrails` / `output_guardrails` 専用フィールドへ
直接渡せる SDK 互換 `InputGuardrail` / `OutputGuardrail`、または `FunctionTool` へ tool guardrail を
装着したラップ済みツールを返す（`agents.Agent` と同型の宣言面）。

あわせて宣言的登録の一式を再エクスポートする: 宣言型と値域型（`GuardrailSpec` / `Boundary` /
`Severity`）・登録簿（`GuardrailRegistry`。名前の強制・境界とメタデータの宣言・`AgentSpec.guardrails`
からの名前参照解決を担う）・同梱 helper の既定分類（`HelperDefaults` / `HELPER_DEFAULTS`）・
guardrail フック外でも使える detector ファクトリ 6 件。実体をフィールドへ直接渡す従来経路は現行の
まま有効で、登録簿はそれを置き換えず並存する追加経路である。

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
    allow_deny_detector,
    canary_detector,
    injection_baseline_detector,
    length_detector,
    predicate_detector,
    regex_detector,
)
from .catalog import HELPER_DEFAULTS, HelperDefaults
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
from .registry import GuardrailRegistry
from .types import Boundary, GuardrailSpec, Severity

__all__ = [
    # 宣言型と値域型（宣言層）
    "GuardrailSpec",
    "Boundary",
    "Severity",
    # 登録簿（facade 経路 / register 経路 / 照会）
    "GuardrailRegistry",
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
    # detector ファクトリ（guardrail フック外でも使える純関数）
    "canary_detector",
    "regex_detector",
    "length_detector",
    "allow_deny_detector",
    "predicate_detector",
    "injection_baseline_detector",
    # plain 検知結果型
    "Detection",
    # 同梱 helper の既定分類（framework ラベル + 既定危険度）
    "HelperDefaults",
    "HELPER_DEFAULTS",
    # 注入ベースライン既定パターン（DI 上書きの基点・補助検知）
    "INJECTION_BASELINE_PATTERNS",
    "SQLI_PATTERNS",
    "COMMAND_INJECTION_PATTERNS",
    "PATH_TRAVERSAL_PATTERNS",
]
