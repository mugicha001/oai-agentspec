"""同梱 guardrail helper の分類カタログ（agents 非依存・plain データ・不変）。

同梱 helper のうち framework 分類（OWASP 等）が helper 名から一意に定まるものの既定値
（`HELPER_DEFAULTS`）と、公開窓口 `runtime/guardrails/__init__.__all__` のシンボルを
「facade 化対象の helper ファクトリ」「対象外」に切り分けるための識別子集合（`frozenset` 4
種）を提供する。本ファイルはデータ定義のみを持ち、それらを用いた集合演算・整合性検証は
テスト側の責務とする。

本層は agents パッケージを一切 import せず、SDK なしで単体検証できる（NFR-1）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from .types import Severity

# 公開 helper だが guardrail 実体を返さないもの（`guard_tool` は既存 `FunctionTool` へ
# ツール境界 guardrail を後付けしてラップ済みツールを返す）。facade 化の対象外。
NON_GUARDRAIL_HELPERS: Final[frozenset[str]] = frozenset({"guard_tool"})

# guardrail フックの外でも単独利用できる純粋な検知器ファクトリ（`Callable[[str], Detection]`
# を返す）。guardrail を返さないので facade 化の対象外。
DETECTOR_FACTORIES: Final[frozenset[str]] = frozenset(
    {
        "canary_detector",
        "regex_detector",
        "length_detector",
        "allow_deny_detector",
        "predicate_detector",
        "injection_baseline_detector",
    }
)

# 公開窓口の `__all__` に載るが helper ファクトリでないシンボル（宣言型・値域型・登録簿・
# plain 検知結果型・分類データ・パターン定数）。
# 内訳は宣言型 3 件 / 登録簿 1 件 / plain 検知結果型 1 件 / 分類データ 2 件 / パターン定数 4 件。
NON_FACTORY_SYMBOLS: Final[frozenset[str]] = frozenset(
    {
        "GuardrailSpec",
        "Boundary",
        "Severity",
        "GuardrailRegistry",
        "Detection",
        "HelperDefaults",
        "HELPER_DEFAULTS",
        "INJECTION_BASELINE_PATTERNS",
        "SQLI_PATTERNS",
        "COMMAND_INJECTION_PATTERNS",
        "PATH_TRAVERSAL_PATTERNS",
    }
)

# 同梱 helper のうち、framework 分類（OWASP 等）が利用者が注入する設定次第で変わるもの。
# 検知の実体（パターン・述語・検知器・判定モデル）が利用者側にあるため、helper 名から
# 分類を確定できない。ゆえに `HELPER_DEFAULTS` のキーに含めない。
DI_DEPENDENT_HELPERS: Final[frozenset[str]] = frozenset(
    {
        "prompt_llm_guardrail",
        "predicate_guardrail",
        "regex_guardrail",
        "length_guardrail",
        "allow_deny_guardrail",
        "external_detector_guardrail",
        "tool_guardrail",
    }
)


@dataclass(frozen=True)
class HelperDefaults:
    """同梱 helper 1 件分の分類既定値（plain 表現・値を解釈せず保持する）。

    Attributes:
        labels: 既定ラベル（フィルタ / 分類用）。`HELPER_DEFAULTS` に載せる既定値は
            `MappingProxyType` で保持する（注釈は `Mapping` で、構築時の coerce はしない）。
        severity: 既定の深刻度。
    """

    labels: Mapping[str, Any]
    severity: Severity


# 同梱 helper のうち、helper 名から framework 分類が一意に定まるものの既定値。
# `HELPER_DEFAULTS` のキー集合は「同梱 helper 識別子 9 件 − `DI_DEPENDENT_HELPERS` 7 件 =
# 2 件」と一致する（この算術が閉じることは別タスクのテストが検証する）。
# 本マッピングがコードの SoT であり、docs の分類表はその投影である。
# labels は「その helper がどの framework 項目に属する検知家族か」の分類であり、当該項目を
# 網羅的にカバーする主張ではない（例: `injection_baseline_guardrail` の LLM01 は非網羅の補助検知で、
# 本丸はパラメータ化クエリ・安全 API の利用）。監査集計へ機械転記する場合は
# `docs/rationale/content-guardrails-coverage.md` のカバレッジマトリクスを併せて参照する。
HELPER_DEFAULTS: Final[Mapping[str, HelperDefaults]] = MappingProxyType(
    {
        "injection_baseline_guardrail": HelperDefaults(
            labels=MappingProxyType({"owasp_llm": "LLM01"}),
            severity=Severity.MEDIUM,
        ),
        "canary_guardrail": HelperDefaults(
            labels=MappingProxyType({"owasp_llm": "LLM07"}),
            severity=Severity.HIGH,
        ),
    }
)
