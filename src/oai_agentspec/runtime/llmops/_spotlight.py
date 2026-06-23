"""Spotlighting の純ヘルパ（untrusted 入力のマーキング・framework 非依存・判断H）。

評価対象が生成した output 等の untrusted テキストを LLM-as-Judge へ渡す前に、データ部分を
明示的なデリミタで囲って「これは評価対象のデータでありプロンプト指示ではない」と判定器に
区別させる（prompt injection 緩和）。DeepEval / agents に非依存の純関数で、単体テスト可能。

`evaluator` が judge へ渡す前に適用する（`_adapters` は `runtime/llmops` を import しない
依存規則のため、マーキングは domain 側に置く）。`_adapters/judge.py` はマーキング済み入力を
受け取り `LLMTestCase` に配置するのみ。

限界と利用者責務（重要）:
    Spotlighting マーカーは「マーカー内部をデータとして扱え」という指示を**判定プロンプト側が
    解釈して初めて**機能する。本実装はデリミタで囲うのみで、マーカーの意味を判定器に伝える指示
    文は含まない（プロンプト非同梱方針）。したがって:
    - 内蔵メトリクス（Faithfulness / AnswerRelevancy）は DeepEval 固定プロンプトで採点され
      マーカーの意味を知らないため、Spotlighting は実質無効（デリミタは無害なノイズに留まる）。
    - G-Eval では利用者の rubric（観点文）側で「マーカー内部はデータでありプロンプト指示として
      解釈しない」旨を含める必要がある（マーカーを機能させるのは利用者責務）。
    マーカー文字列（`_SPOTLIGHT_BEGIN` / `_SPOTLIGHT_END`）はこの目的のため公開・固定する。
"""

from __future__ import annotations

from typing import Final

# untrusted データ部分を囲うマーカー（判定器がデータ境界を識別するための固定デリミタ）。
_SPOTLIGHT_BEGIN: Final[str] = "<<UNTRUSTED_DATA>>"
_SPOTLIGHT_END: Final[str] = "<</UNTRUSTED_DATA>>"


def is_spotlighted(text: str) -> bool:
    """`text` が既に spotlight マーカーで囲われているかを返す（冪等性判定）。

    Args:
        text: 判定対象テキスト。

    Returns:
        先頭が開始マーカー・末尾が終了マーカーなら True。
    """
    return text.startswith(_SPOTLIGHT_BEGIN) and text.endswith(_SPOTLIGHT_END)


def spotlight(text: str) -> str:
    """untrusted テキストを spotlight マーカーで囲ってマーキングする（冪等）。

    既にマーキング済み（`is_spotlighted`）の場合はそのまま返す（二重マーキングしない）。
    text 内に偶発的に出現するマーカー文字列は無害化のため空白へ置換し、データ境界の偽装を
    防ぐ。

    Args:
        text: マーキングする untrusted テキスト（評価対象 output 等）。

    Returns:
        開始 / 終了マーカーで囲ったテキスト。既にマーキング済みなら入力をそのまま返す。
    """
    if is_spotlighted(text):
        return text
    sanitized = text.replace(_SPOTLIGHT_BEGIN, " ").replace(_SPOTLIGHT_END, " ")
    return f"{_SPOTLIGHT_BEGIN}{sanitized}{_SPOTLIGHT_END}"
