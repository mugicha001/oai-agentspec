"""L2: 内容ガードレール公開窓口（`runtime.guardrails.__init__`）の `__all__` 契約を検証する。

`__all__` のメンバ集合（27 件・ブロック構成）・全メンバが窓口から import 可能であること
（公開 API スモーク）・コア `__all__` へ会話 / 実行寄りシンボルが漏れていないこと（opt-in の
表現をコア `__all__` からの分離で行う方針の pin）を検証する。

`__all__` を差集合の入力に使う FR-6 の導出規則（除外 3 集合との関係）は
`test_facade_sync_l2.py` が担うため、本モジュールでは集合そのものを固定する。
"""

from __future__ import annotations

import pytest

import oai_agentspec
import oai_agentspec.runtime.guardrails as guardrails

pytestmark = pytest.mark.integration

# 公開窓口の `__all__` メンバ集合（設計のブロック構成どおり 27 件）。
_EXPECTED_ALL = {
    # 宣言型と値域型
    "GuardrailSpec",
    "Boundary",
    "Severity",
    # 登録簿
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
    # ツール境界 helper ファクトリ
    "tool_guardrail",
    "guard_tool",
    # detector ファクトリ
    "canary_detector",
    "regex_detector",
    "length_detector",
    "allow_deny_detector",
    "predicate_detector",
    "injection_baseline_detector",
    # plain 検知結果型
    "Detection",
    # 同梱 helper の既定分類
    "HelperDefaults",
    "HELPER_DEFAULTS",
    # 注入ベースライン既定パターン
    "INJECTION_BASELINE_PATTERNS",
    "SQLI_PATTERNS",
    "COMMAND_INJECTION_PATTERNS",
    "PATH_TRAVERSAL_PATTERNS",
}


def test_公開窓口のall集合は27件に固定される() -> None:
    """`__all__` のメンバ集合を `==` で pin する（追加漏れ・削除の両方向を検知する）。

    集合一致で書くのは、シンボルの追加（宣言したが窓口へ載せ忘れ）と削除（公開 API の破壊）を
    1 本で同時に検知するため。
    """
    assert set(guardrails.__all__) == _EXPECTED_ALL
    assert len(guardrails.__all__) == 27


def test_公開窓口のallに重複がない() -> None:
    """`__all__` は同一シンボルを 2 度載せない（ブロック追加時のコピペ重複の検知）。"""
    assert len(guardrails.__all__) == len(set(guardrails.__all__))


@pytest.mark.parametrize("symbol", sorted(_EXPECTED_ALL))
def test_公開窓口のall全件が窓口からimportできる(symbol: str) -> None:
    """公開 API スモーク: `__all__` の全メンバが窓口の属性として解決できる。"""
    assert hasattr(guardrails, symbol), f"{symbol} が公開窓口から解決できない"


def test_detector6件は呼び出すと検知器を返す() -> None:
    """detector ファクトリ 6 件が guardrail フックの外でも単独で使える（`Detection` を返す）。"""
    detectors = [
        guardrails.canary_detector("S3CR3T"),
        guardrails.regex_detector(r"\d{3}"),
        guardrails.length_detector(max_length=3),
        guardrails.allow_deny_detector(deny=["bad"]),
        guardrails.predicate_detector(lambda text: "x" in text),
        guardrails.injection_baseline_detector(),
    ]
    for detect in detectors:
        result = detect("some text")
        assert isinstance(result, guardrails.Detection)
        assert isinstance(result.triggered, bool)


def test_ガードレールシンボルはコアのallに載らない() -> None:
    """opt-in はコア `__all__` からの分離で表現する（宣言層シンボルのみをコアに置く方針）。

    公開窓口の 27 件がコア `__all__` へ 1 件も混入していないことを pin する（混入すると
    `import oai_agentspec` の公開面が実行寄り層へ広がり、extra の境界が意味を失う）。
    """
    assert set(oai_agentspec.__all__).isdisjoint(_EXPECTED_ALL)
