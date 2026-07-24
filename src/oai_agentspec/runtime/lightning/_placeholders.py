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

import difflib
import logging
import re
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from oai_agentspec.runtime.lightning.types import SlotSegment

logger = logging.getLogger(__name__)

# `${var}` プレースホルダ（識別子は Python 識別子相当）。bare `$var` には**マッチしない**
# ことが本ヘルパの load-bearing な性質（Template.safe_substitute との差分）。
PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# 境界マーカーの予約接頭辞。tune セグメントが 2 個以上のとき seed 連結時に
# `${oas_boundary_1}` .. `${oas_boundary_(n-1)}` を境界へ挟み、rollout / OptimizeResult 合成で
# `split_marked` が分割消費する（成果物には現れない）。`_adapters/lightning` の post-fit
# フォールバック判定もこの接頭辞を関数内遅延 import で参照する（SSoT・予約接頭辞の衝突検査）。
BOUNDARY_PREFIX: Final = "oas_boundary_"

# `${oas_boundary_N}` 形式の境界マーカー（N は 1 始まりの構成順連番）。接頭辞から組み立てて
# `BOUNDARY_PREFIX` を Single Source of Truth に保つ。
_BOUNDARY_MARKER_RE = re.compile(r"\$\{" + re.escape(BOUNDARY_PREFIX) + r"(\d+)\}")


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


def split_marked(candidate: str, n_tune: int) -> list[str] | None:
    """境界マーカーで候補テキストを `n_tune` 個の tune セグメントへ分割する（exact-once 検査付き）。

    tune セグメントが 2 個以上のとき、候補テキストは `${oas_boundary_1}` ..
    `${oas_boundary_(n_tune-1)}` を境界に挟んだ 1 テキストとして最適化される。本関数は
    マーカーが **各々ちょうど 1 回**・**構成順（1, 2, 3, ...）どおり**に出現していることを
    検査し、満たす場合のみそれらの位置で分割した `n_tune` 個の文字列リストを返す。
    マーカーの欠落・重複（2 回以上）・順序不整合・想定外の連番は `None`（候補無効）で
    表現し、例外は送出しない。呼び出し側（`_apply_candidate`）はこの `None` を
    `_reinject_vars` の None と同一の per-candidate 無効化経路（候補 reward 0.0）で扱う。

    Args:
        candidate: 境界マーカーを含みうる候補テキスト。
        n_tune: 期待する tune セグメント数（`1` はマーカー不要・`0` は防御的に None）。

    Returns:
        分割済みの `n_tune` 個の文字列リスト。マーカー検査に失敗した場合や `n_tune == 0` は None。
    """
    if n_tune <= 0:
        return None

    matches = list(_BOUNDARY_MARKER_RE.finditer(candidate))
    numbers = [int(match.group(1)) for match in matches]
    # 期待列（構成順・各 1 回）と完全一致しなければ無効。欠落・重複・順序不整合・想定外連番を
    # 一括で弾く。`n_tune == 1` は期待列が空になり、マーカー混入があれば None に倒れる。
    if numbers != list(range(1, n_tune)):
        return None

    parts: list[str] = []
    prev = 0
    for match in matches:
        parts.append(candidate[prev : match.start()])
        prev = match.end()
    parts.append(candidate[prev:])
    return parts


def boundary_intact(seed_text: str, best_text: str) -> bool:
    """seed と best の境界マーカー列（連番の順序込み）が完全一致するか判定する（C2 対応）。

    APO best 候補の post-fit 検査に用いる。従来の `str.count` ベースの検査は各マーカー名の
    出現回数だけを比較しており、seed と best で count は一致するが順序が入れ替わっている
    ケース（例: seed=`${oas_boundary_1}...${oas_boundary_2}` に対し best は 2 と 1 が swap）を
    素通ししていた。順序不整合は `split_marked` 側で None を返すが、post-fit fallback が
    発火しないと `_recompose_new_shape_results` で silent に `continue` され、`OptimizeResult`
    に literal `${oas_boundary_N}` が漏出する。本関数は連番の順序も含めた完全一致で判定する
    ため、post-fit で fallback を確実に発火させる
    （`split_marked` と同じ order-check セマンティクス）。

    Args:
        seed_text: seed 側のテキスト（構造の基準）。
        best_text: APO 最良候補のテキスト。

    Returns:
        seed と best の境界マーカー連番列（出現順）が完全一致すれば True、それ以外は False。
    """
    seed_numbers = [int(m.group(1)) for m in _BOUNDARY_MARKER_RE.finditer(seed_text)]
    best_numbers = [int(m.group(1)) for m in _BOUNDARY_MARKER_RE.finditer(best_text)]
    return seed_numbers == best_numbers


def compose_segments(
    segments: tuple[SlotSegment, ...],
    tune_texts: list[str],
    vars_dict: dict[str, Any],
) -> str:
    """`segments` を構成順に走査し、tune テキストを再インターリーブして full テキストへ合成する。

    `tune=True` のセグメントは `tune_texts` を先頭から順に消費する（構成順とテキスト順が対応）。
    `tune=False` の固定セグメントは `text` をそのまま使い、その `text` にのみ `vars_dict` を
    `substitute_braced` で注入する（tune 側は `${var}` 温存契約のため注入しない・rollout 直前の
    `_reinject_vars` で別途注入される）。既定 build（rollout 時）と optimizer（OptimizeResult 合成）
    の双方が本関数を呼ぶことで、合成規則の drift を不可能にする（`compose_with_vars` と同じ SSoT）。

    Args:
        segments: 構成順の `SlotSegment` タプル（空なら空文字を返す）。
        tune_texts: `tune=True` セグメントへ順に割り当てるテキスト列。
        vars_dict: 固定セグメントへ注入する `${var}` 置換値（未指定キーは `${name}` 保持）。

    Returns:
        各セグメントを `"\\n\\n"` で連結した合成済み full テキスト（`segments` が空なら空文字）。

    Raises:
        ValueError: `tune_texts` の長さが `tune=True` のセグメント数と一致しない場合
            （実装者ミス検出）。
    """
    tune_count = sum(1 for segment in segments if segment.tune)
    if len(tune_texts) != tune_count:
        raise ValueError(
            f"tune_texts の長さ ({len(tune_texts)}) が "
            f"tune セグメント数 ({tune_count}) と一致しません"
        )

    rendered: list[str] = []
    tune_index = 0
    for segment in segments:
        if segment.tune:
            rendered.append(tune_texts[tune_index])
            tune_index += 1
        else:
            rendered.append(substitute_braced(segment.text, vars_dict))
    return "\n\n".join(rendered)


def compose_from_marked(
    segments: tuple[SlotSegment, ...],
    candidate: str,
    vars_dict: dict[str, Any],
) -> str | None:
    """境界マーカー入り候補を分割・再インターリーブして full テキストへ合成する（SSoT）。

    `segments` 内の `tune=True` セグメント数から `n_tune` を算出し、`split_marked` で候補を
    `n_tune` 個の tune 断片へ分割する。各断片は `.strip()` して境界由来の前後空白を除去し
    （`compose_segments` が改めて `"\\n\\n"` 連結するため二重化を防ぐ）、`compose_segments` で
    固定セグメントと構成順に再インターリーブする。

    Args:
        segments: 構成順の `SlotSegment` タプル（tune / 固定の別を保持）。
        candidate: 境界マーカーを含みうる候補テキスト。
        vars_dict: 固定セグメントへ注入する `${var}` 置換値（未指定キーは `${name}` 保持）。

    Returns:
        合成済み full テキスト。`split_marked` がマーカー崩れ（欠落・重複・順序不整合）で
        `None` を返した場合は `None`（候補無効・呼び出し側で扱う）。本関数は rollout 実体
        （`slots._new_default_build` の build）と OptimizeResult 合成の両方から呼ばれる SSoT
        契約であり、両経路の合成規則 drift（"OptimizeResult.prompt == rollout instructions"）
        を不可能にする。
    """
    n_tune = sum(1 for segment in segments if segment.tune)
    tune_texts = split_marked(candidate, n_tune)
    if tune_texts is None:
        return None
    stripped = [text.strip() for text in tune_texts]
    return compose_segments(segments, stripped, vars_dict)


def unified_diff_labeled(seed: str, prompt: str) -> str:
    """合成済み full の seed / prompt から unified diff 文字列を生成する（SSoT 契約）。

    rollout build（`_recompose_new_shape_results` が組む OptimizeResult）と `_adapters.lightning`
    の `run_apo` 合成の両方から呼ばれる Single Source of Truth。`fromfile="before"` /
    `tofile="after"` にラベルを統一することで、diff 生成の二重実装によるラベル drift を不可能に
    する。`splitlines()` した seed / prompt を `difflib.unified_diff` に渡し、`lineterm=""` で
    連結時の余計な改行を防ぐ。差分がなければ空文字を返す。

    Args:
        seed: 最適化前テキスト（before・合成済み full）。
        prompt: 最適化後テキスト（after・合成済み full）。

    Returns:
        `fromfile=before` / `tofile=after` で統一した unified diff 文字列（差分なしは空文字）。
    """
    return "\n".join(
        difflib.unified_diff(
            seed.splitlines(),
            prompt.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )
