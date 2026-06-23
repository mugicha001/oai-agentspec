"""内容ガードレールの plain 検知ロジック層（agents 非依存・決定的・SDK を見ない）。

「テキスト → 検知結果（`Detection`）」を返す純関数 / 純述語のファクトリ群を提供する。canary
（漏洩トークンの逐語照合）・regex（DI パターン）・length（長さ閾値）・allow_deny（許可 / 拒否
リスト）・predicate（DI callable の薄い包み）・injection_baseline（SQLi / コマンド注入 /
パストラバーサルの代表パターン）を含む。注入ベースラインの既定パターンはモジュール定数として
持ち、DI で上書き / 拡張できる（補助検知であり網羅的検知ではない）。

本層は agents パッケージを一切 import せず、SDK なしで単体検証できる（SDK 隔離 grep の対象外）。
SDK 互換 guardrail / ラップ済み `FunctionTool` への接着は `factories.py`（+ `_adapters/guardrails`）
が担う。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

# 注入ベースラインの既定パターン（補助検知・非網羅・DI 上書き可）。注入対策の本丸はパラメータ化
# クエリ / 安全 API 利用であり、本パターンは早期の粗い網に留める（rationale 参照）。
SQLI_PATTERNS: Final[tuple[str, ...]] = (
    r"(?i)\bunion\b\s+\bselect\b",
    r"(?i)\bor\b\s+1\s*=\s*1",
    r"(?i)\bdrop\b\s+\btable\b",
    r"(?i);\s*--",
    r"(?i)'\s*or\s*'",
)

COMMAND_INJECTION_PATTERNS: Final[tuple[str, ...]] = (
    r"[;&|`]\s*\w",
    r"\$\(",
    r"(?i)\b(rm|curl|wget|nc|bash|sh)\b\s+-",
)

PATH_TRAVERSAL_PATTERNS: Final[tuple[str, ...]] = (
    r"\.\./",
    r"\.\.\\",
    r"(?i)%2e%2e%2f",
    r"(?i)/etc/passwd",
)

# 注入ベースラインの既定パターン全集合（SQLi + コマンド注入 + パストラバーサル）。
INJECTION_BASELINE_PATTERNS: Final[tuple[str, ...]] = (
    *SQLI_PATTERNS,
    *COMMAND_INJECTION_PATTERNS,
    *PATH_TRAVERSAL_PATTERNS,
)


@dataclass(frozen=True)
class Detection:
    """検知結果の plain 表現（agents / SDK 非依存）。

    Attributes:
        triggered: 検知が発火したか（True なら guardrail の tripwire を立てる）。
        reason: 検知理由（任意・トレース / 注釈メッセージに使う）。
        info: 付帯情報（任意・マッチしたパターンや件数等の granular な結果）。
    """

    triggered: bool
    reason: str | None = None
    info: Any = field(default=None)


def canary_detector(canary: str | Iterable[str]) -> Callable[[str], Detection]:
    """canary（漏洩トークン）の逐語照合検知関数を作る（システムプロンプト漏洩検知）。

    出力テキストに canary 文字列が逐語で含まれていれば検知する（部分文字列照合）。canary は
    単一文字列または文字列の iterable（複数トークン）を受ける。

    Args:
        canary: 照合する canary 値（単一 or 複数）。

    Returns:
        テキストを受けて `Detection` を返す検知関数。
    """
    canaries = (canary,) if isinstance(canary, str) else tuple(canary)

    def _detect(text: str) -> Detection:
        hits = [c for c in canaries if c and c in text]
        if hits:
            return Detection(
                triggered=True,
                reason="canary token leaked",
                info={"matched": hits},
            )
        return Detection(triggered=False)

    return _detect


def regex_detector(patterns: str | Iterable[str], *, flags: int = 0) -> Callable[[str], Detection]:
    """正規表現パターンへのマッチで検知する関数を作る（DI パターン）。

    いずれかのパターンがテキスト内に出現すれば検知する（`re.search`）。パターンは単一文字列
    または文字列の iterable を受ける。guardrail は untrusted テキストに適用されるため、DI する
    パターンは利用者責任で ReDoS 安全なもの（壊滅的バックトラックを起こさない）を渡すこと。

    Args:
        patterns: 検知に使う正規表現（単一 or 複数）。
        flags: `re.compile` に渡すフラグ（既定 0）。

    Returns:
        テキストを受けて `Detection` を返す検知関数。
    """
    raw = (patterns,) if isinstance(patterns, str) else tuple(patterns)
    compiled = [re.compile(p, flags) for p in raw]

    def _detect(text: str) -> Detection:
        matched = [c.pattern for c in compiled if c.search(text)]
        if matched:
            return Detection(
                triggered=True,
                reason="regex pattern matched",
                info={"matched": matched},
            )
        return Detection(triggered=False)

    return _detect


def length_detector(
    *, max_length: int | None = None, min_length: int | None = None
) -> Callable[[str], Detection]:
    """長さ / サイズ閾値で検知する関数を作る（無制限消費の粗い網）。

    テキスト長が `max_length` 超過、または `min_length` 未満なら検知する（いずれか / 両方指定可）。

    Args:
        max_length: 上限文字数（超過で検知）。None で上限なし。
        min_length: 下限文字数（未満で検知）。None で下限なし。

    Returns:
        テキストを受けて `Detection` を返す検知関数。
    """

    def _detect(text: str) -> Detection:
        length = len(text)
        if max_length is not None and length > max_length:
            return Detection(
                triggered=True,
                reason=f"length {length} exceeds max {max_length}",
                info={"length": length, "max_length": max_length},
            )
        if min_length is not None and length < min_length:
            return Detection(
                triggered=True,
                reason=f"length {length} below min {min_length}",
                info={"length": length, "min_length": min_length},
            )
        return Detection(triggered=False)

    return _detect


def allow_deny_detector(
    *,
    deny: Iterable[str] | None = None,
    allow: Iterable[str] | None = None,
    case_sensitive: bool = True,
) -> Callable[[str], Detection]:
    """allow / deny リスト照合で検知する関数を作る（部分文字列ベース）。

    `deny` のいずれかがテキストに含まれれば検知する。`allow` を指定した場合、`allow` のいずれも
    含まれなければ検知する（許可リスト外を検知）。両方指定時は deny 優先（deny ヒットなら即検知）。

    Args:
        deny: 含まれていたら検知する拒否語の集合（任意）。
        allow: いずれも含まれなければ検知する許可語の集合（任意）。
        case_sensitive: 大文字小文字を区別するか（既定 True）。

    Returns:
        テキストを受けて `Detection` を返す検知関数。
    """
    deny_list = tuple(deny or ())
    allow_list = tuple(allow or ())

    def _normalize(value: str) -> str:
        return value if case_sensitive else value.lower()

    def _detect(text: str) -> Detection:
        haystack = _normalize(text)
        denied = [d for d in deny_list if _normalize(d) in haystack]
        if denied:
            return Detection(
                triggered=True,
                reason="deny term matched",
                info={"matched": denied},
            )
        if allow_list and not any(_normalize(a) in haystack for a in allow_list):
            return Detection(
                triggered=True,
                reason="no allow term matched",
                info={"allow": list(allow_list)},
            )
        return Detection(triggered=False)

    return _detect


def predicate_detector(
    predicate: Callable[[str], bool], *, reason: str | None = None
) -> Callable[[str], Detection]:
    """汎用 predicate（`Callable[[str], bool]`）を検知関数へ薄く包む（DI 述語）。

    `predicate(text)` が True を返したら検知する。任意ロジックを利用者 DI で差し込む拡張点。

    Args:
        predicate: テキストを受けて検知有無（bool）を返す述語。
        reason: 検知時の理由（任意）。

    Returns:
        テキストを受けて `Detection` を返す検知関数。
    """

    def _detect(text: str) -> Detection:
        if predicate(text):
            return Detection(triggered=True, reason=reason or "predicate matched")
        return Detection(triggered=False)

    return _detect


def injection_baseline_detector(
    extra_patterns: Iterable[str] | None = None,
) -> Callable[[str], Detection]:
    """注入ベースライン（SQLi / コマンド注入 / パストラバーサル）の検知関数を作る（補助検知）。

    既定パターン（`INJECTION_BASELINE_PATTERNS`）に `extra_patterns` を追記して照合する。本検知は
    **網羅的検知ではなく補助検知**であり、注入対策の本丸はパラメータ化クエリ / 安全 API 利用で
    ある（rationale 参照）。既定パターンは DI（`extra_patterns`）で拡張でき、完全な差し替えが必要
    なら `regex_detector` を直接使う。既定パターンは自然文入力で誤検知（false positive）しやすく
    （特にコマンド注入パターン）検知漏れ（false negative）も前提のため、利用者の入力分布に応じて
    DI で調整すること。

    Args:
        extra_patterns: 既定に追記する正規表現パターン（任意）。

    Returns:
        テキストを受けて `Detection` を返す検知関数。
    """
    patterns = (*INJECTION_BASELINE_PATTERNS, *tuple(extra_patterns or ()))
    return regex_detector(patterns)
