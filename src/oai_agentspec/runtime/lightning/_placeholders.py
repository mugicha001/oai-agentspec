"""`${var}` プレースホルダの抽出・置換・合成ヘルパ（Single Source of Truth）。

本モジュールは APO 関連で 3 箇所に重複していた `${var}` regex / 合成規則を一本化する。
`_default_build`（rollout 時 `agent.instructions` 生成）と `_compose_full`（OptimizeResult
構築時）の双方が本ヘルパを呼ぶことで、合成規則 drift（公開契約 "OptimizeResult.prompt は
rollout 実体と一致" の違反）を不可能にする。

`Template.safe_substitute` は `${var}` と bare `$var` の両方にマッチするため、seed や APO
候補に含まれる literal `$5` / `$PATH` を vars キーと衝突して silent rewrite するリスクが
あった（Codex 第3 round 指摘）。本モジュールの `substitute_braced` は braced 形式
（`${name}`）のみを置換し、bare `$var` は一切触らない。

NFR-1: agentlightning / agents への依存を一切持たない（plain regex / Python str のみ・
`runtime/lightning/_placeholders` は core 層）。`_adapters/lightning.py` からは関数内遅延
import で参照する（既存の types / config 参照と同じパターン）。
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# `${var}` プレースホルダ（識別子は Python 識別子相当）。bare `$var` には**マッチしない**
# ことが本ヘルパの load-bearing な性質（Template.safe_substitute との差分）。
PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def extract_placeholders(text: str) -> set[str]:
    """`text` 内の `${var}` プレースホルダ識別子集合を返す（重複は集約）。

    Args:
        text: 対象テキスト。

    Returns:
        識別子 `set[str]`（テキスト中に `${var}` が無ければ空集合）。
    """
    return set(PLACEHOLDER_RE.findall(text))


def substitute_braced(text: str, vars_dict: dict[str, Any] | None) -> str:
    """`${var}` 形式のみを `vars_dict` で置換する（bare `$var` は触らない）。

    `string.Template.safe_substitute` は `${var}` と `$var` の両方を置換するため、seed や
    APO 候補に含まれる literal `$X`（価格表記 `$5`, shell 風 `$PATH`, モデル emit の
    `$name` 等）を vars キーと衝突して silent rewrite するリスクがあった
    （Codex 第3 round 指摘）。本関数は `${var}` 形式のみを置換し、`vars_dict` に未指定の
    placeholder は `${name}` のまま保持する（safe_substitute の "未指定は維持" セマンティ
    クスを継承）。

    Args:
        text: 置換対象テキスト。
        vars_dict: 置換値（None / 空 dict は no-op で text をそのまま返す）。

    Returns:
        `${var}` を `vars_dict` で置換したテキスト（`vars_dict` 未指定キーは `${name}` 維持）。
    """
    if not vars_dict or not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in vars_dict:
            value = vars_dict[name]
            if not isinstance(value, str):
                # 非 str 値は `str(value)` で強制変換するが、利用者が dict / list / dataclass 等を
                # 誤って渡したまま気付かないと、prompt に `{'k': 'v'}` のような repr 形式の文字列が
                # 埋め込まれて APO スコアが silent に劣化する。warn で利用者へ通知する
                # （Codex 第4 round 指摘・fail-open 維持で既存挙動と互換）。
                logger.warning(
                    "vars[%r]=%r は str ではありません（%s）。str(value) で変換しますが、"
                    "意図せず repr 形式が prompt に埋め込まれる場合は事前に str 化してください",
                    name,
                    value,
                    type(value).__name__,
                )
            return str(value)
        return match.group(0)

    return PLACEHOLDER_RE.sub(_replace, text)


def compose_with_vars(fixed: str, tune: str, vars_dict: dict[str, Any] | None = None) -> str:
    """`Slot.fixed`（base + parts）と tune を rollout 時 `agent.instructions` と同じ形で合成。

    `_default_build`（rollout 時）と `_compose_full`（OptimizeResult 構築時）の両方が本関数
    を呼ぶことで、合成規則の drift を不可能にする（Single Source of Truth）。fixed 側にのみ
    `vars_dict` を再注入する（`_default_build` の規則）。tune 側は APO 候補本体で `${var}`
    温存契約のため substitute しない（rollout 直前の `_reinject_vars` で別途注入される）。

    合成規則:
        - `fixed` が空文字: `tune` をそのまま返す。
        - `fixed` が非空: `f"{fixed_substituted}\\n\\n{tune}"` を返す
          （`fixed_substituted` は `substitute_braced(fixed, vars_dict)`）。

    `fixed` の空判定は **substitution 前** の `fixed` で行う（`fixed_substituted` ではなく）。
    fixed が `"${role}"` のみで vars が `{"role": ""}` のとき、後者で判定すると "\\n\\n" 区切りが
    silent に脱落して `tune` 単体に落ちるが、利用者が fixed を渡している以上は空 vars 値でも
    "\\n\\n" 区切りを保つ方が rollout 実体（_default_build の合成結果）と整合する
    （Codex 第4 round 指摘）。

    Args:
        fixed: 固定部分テキスト（base + parts の合成済み・`${var}` 保持・空文字可）。
        tune: APO 最適化対象テキスト（候補プロンプト・`${var}` 温存）。
        vars_dict: `${var}` 置換値（None / 空 dict は no-op）。

    Returns:
        合成済み full テキスト（`fixed` が空なら `tune` 単体）。
    """
    if not fixed:
        return tune
    fixed_substituted = substitute_braced(fixed, vars_dict)
    return f"{fixed_substituted}\n\n{tune}"
