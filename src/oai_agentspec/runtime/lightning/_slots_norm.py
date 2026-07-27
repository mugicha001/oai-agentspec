"""スロット / seed / 入力ケース正規化（`optimizer` 内部の private ヘルパ集）。

`optimize` 公開エントリから呼び出される正規化系を集約する。`Slot` / 生 seed / dict 混在の判別、
seed テキストの抽出、`${var}` 再注入、`OptimizeCase` / dict ケースからの入力テキスト抽出を担う。
SDK / `agentlightning` を import せず、宣言層の `AgentSpec` のみを参照する（NFR-1）。

公開窓口は `optimizer.optimize` 経由のみで、本モジュールは `_` 接頭辞のとおり外向き API を持た
ない。テストからの直接 import は `from .._slots_norm import _normalize_slots` のように private
として参照する（後方互換のため `optimizer` モジュールからも再エクスポートする）。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ._placeholders import extract_placeholders, substitute_braced
from .types import FailureKind, OptimizeError, Slot

if TYPE_CHECKING:
    pass


def _extract_case_input(case: Any) -> Any:
    """rollout 入力（プロンプト本文）をケースから取り出す（dict / 属性両対応）。

    dict なら `case["input"]`、`OptimizeCase` 等の属性ケースなら `case.input`、いずれでも無ければ
    `case` 自身を返す（生 str / 自由ケースとの後方互換）。`RolloutResult.case` には常に元ケースが
    そのまま渡るため、reward callable は dict / `OptimizeCase` どちらでも採点できる。

    Args:
        case: rollout への入力ケース。

    Returns:
        rollout に渡す入力テキスト（または case 自身）。
    """
    if isinstance(case, dict):
        return case.get("input")
    return getattr(case, "input", case)


def _normalize_slots(
    target: Any, slot: Slot | str | Iterable[Slot] | dict[str, Slot | str] | None
) -> dict[str, Slot] | None:
    """`slot` 引数を `{名前: Slot}` か None（生 seed 経路）へ正規化する。

    `Slot` / `{名前: Slot}` / `Iterable[Slot]` は `Slot` mapping へ、生 seed（str / `{名前: str}`）
    は None（rebind 必須）へ倒す。`slot=None` で target が静的 `AgentSpec`（instructions が str）
    のときは instructions を seed とする既定 `Slot`（既定 build = instructions 差し替え）を 1 件
    生成する。`str` も `Iterable` だが生 seed 経路として先に判別するため、列（`list[Slot]` 等）
    と混同しない。

    Args:
        target: 最適化対象（既定スロット導出に使う）。
        slot: 利用者指定のスロット。

    Returns:
        `{名前: Slot}` の mapping（自動 rebind 経路）、または None（生 seed = rebind 必須経路）。

    Raises:
        OptimizeError: `slot` の dict に `Slot` と生 seed(str) が混在する場合、`slot` の列が空、
            列に `Slot` 以外の要素が混在する場合、列内の `Slot.name` が重複する場合、または
            `target` が `AgentSpec` のときに `slot.name`（列は各要素名）が `target.name` と
            不一致の場合（`FailureKind.CONFIG_MISSING`・fail-closed）。
    """
    from ...spec import AgentSpec

    if slot is None:
        if isinstance(target, AgentSpec) and isinstance(target.instructions, str):
            seed = target.instructions

            def _build(candidate: str) -> AgentSpec:
                import dataclasses

                return dataclasses.replace(target, instructions=candidate)

            return {target.name: Slot(name=target.name, seed=seed, build=_build)}
        return None

    if isinstance(slot, Slot):
        _ensure_slot_target_name_match(target, [slot.name])
        return {slot.name: slot}

    if isinstance(slot, dict):
        if not slot:
            # 空 dict は最適化対象が無い不正設定（`prompt_slots(agents=[])` 等で起きうる・
            # 空のまま最適化を進めると prompt={} で返り誤った成功になるため fail-closed）。
            raise OptimizeError(
                FailureKind.CONFIG_MISSING,
                "slot の dict が空です（最適化対象スロットがありません）。"
                "prompt_slots(agents=[...]) に少なくとも 1 つのエージェント名を指定するか、"
                "slot=prompt_slot(...) で単一スロットを渡してください",
            )
        slot_values = [isinstance(v, Slot) for v in slot.values()]
        if all(slot_values):
            # dict キーと `Slot.name` の不一致は黙って受理すると、`_apply_candidate` が
            # `reinjected[slot.name]` で KeyError を起こすか、graph 経路で dict キーで参照される
            # agent に違う Slot が当たって wrong agent を build する原因になる（Codex P2）。
            mismatched = [
                (k, v.name) for k, v in slot.items() if isinstance(v, Slot) and k != v.name
            ]
            if mismatched:
                pairs = ", ".join(f"{k!r}->Slot(name={n!r})" for k, n in mismatched)
                raise OptimizeError(
                    FailureKind.CONFIG_MISSING,
                    f"slot の dict キーと Slot.name が一致しません ({pairs})。"
                    "mapping のキーは各 Slot の name と同一にしてください "
                    "（不一致だと wrong agent が最適化されたり KeyError で落ちます）",
                )
            _ensure_slot_target_name_match(target, list(slot.keys()))
            return {k: v for k, v in slot.items() if isinstance(v, Slot)}
        if any(slot_values):
            # Slot と 生 seed の混在は rebind 経路が曖昧（fail-closed）。
            raise OptimizeError(
                FailureKind.CONFIG_MISSING,
                "slot の dict に Slot と生 seed(str) が混在しています。"
                "全て Slot（自動 rebind）か全て生 seed(str)（rebind 明示）に揃えてください",
            )
        return None

    if isinstance(slot, str):
        # 生 seed（str）は rebind 必須経路。`str` は `Iterable` でもあるため、以降の
        # Iterable[Slot] 分岐より前に処理して silent な列解釈を防ぐ（ADR 0008）。
        return None

    if isinstance(slot, Iterable):
        items = list(slot)
        if not items:
            raise OptimizeError(
                FailureKind.CONFIG_MISSING,
                "slot の列が空です（最適化対象スロットがありません）。"
                "少なくとも 1 つの Slot を含めるか、"
                "slot=prompt_slot(...) で単一スロットを渡してください",
            )
        if not all(isinstance(item, Slot) for item in items):
            raise OptimizeError(
                FailureKind.CONFIG_MISSING,
                "slot の列に Slot 以外の要素が含まれます（列経路は自動 rebind 専用・"
                "生 seed の列は不可）。全要素を Slot にするか、"
                "生 seed の列は使わずに個別 rebind 経路を選んでください",
            )
        names = [item.name for item in items]
        if len(set(names)) != len(names):
            raise OptimizeError(
                FailureKind.CONFIG_MISSING,
                f"slot の列に Slot.name の重複があります: {names}。"
                "各 Slot に一意な name を割り当ててください",
            )
        _ensure_slot_target_name_match(target, names)
        return {item.name: item for item in items}

    # 保険（防御的フォールバック）: 上記いずれの型にも該当しない場合。
    return None


def _seeds_of(
    slots: dict[str, Slot] | None,
    slot: Slot | str | Iterable[Slot] | dict[str, Slot | str] | None,
) -> dict[str, str]:
    """最適化対象スロットの seed テキスト（`{名前: seed}`・`${var}` 保持）を導出する。

    `slots`（自動 rebind 経路）があれば各 `Slot.seed` を、無ければ生 seed（str / `{名前: str}`）を
    そのまま seed とする。

    Args:
        slots: 正規化済み `{名前: Slot}`（自動 rebind 経路）または None。
        slot: 利用者指定のスロット（生 seed 経路の seed 取得元）。

    Returns:
        `{名前: seed}` の mapping（`${var}` 保持）。
    """
    if slots is not None:
        return {name: s.seed for name, s in slots.items()}
    if isinstance(slot, str):
        return {"prompt": slot}
    if isinstance(slot, dict):
        return {k: str(v) for k, v in slot.items()}
    return {}


def _ensure_slot_target_name_match(target: Any, slot_names: list[str]) -> None:
    """`target=AgentSpec` のとき slot 名がちょうど 1 つで `target.name` と一致するか検証する。

    `target=AgentSpec(name='A')` に対して `slot=Slot(name='B', ...)` を渡すと、`_apply_candidate`
    は registry から 'B' の spec を resolve してビルドするため、利用者が指定した `target=A` では
    なく別の agent が黙って最適化される。さらに mapping `{'B': ..., 'A': ...}` のように余剰スロット
    を含むケースでは `_apply_candidate` が `next(iter(slots.values()))` で 1 個目を取るため、辞書
    順序によって wrong agent を最適化したり余剰スロットが silent に無視されたりする
    （Codex P2）。`AgentSpec` は単一 agent なのでスロットは 1 個（target.name と同一）に限定して
    fail-closed する。

    Args:
        target: 最適化対象（`AgentSpec` 以外は対象外で no-op）。
        slot_names: 利用者指定スロット名一覧。

    Raises:
        OptimizeError: `target=AgentSpec` で `slot_names` が空、複数、または `target.name` と
            一致しない場合（`FailureKind.CONFIG_MISSING`）。
    """
    from ...spec import AgentSpec

    if not isinstance(target, AgentSpec):
        return
    if len(slot_names) == 1 and slot_names[0] == target.name:
        return
    names = ", ".join(repr(n) for n in slot_names)
    raise OptimizeError(
        FailureKind.CONFIG_MISSING,
        f"target=AgentSpec(name={target.name!r}) には slot をちょうど 1 つ "
        f"（target.name と同一）渡す必要があります（受領: {names}）。"
        "AgentSpec は単一 agent のため、複数スロット mapping は wrong agent を最適化する原因に "
        "なります（HandoffGraph / WorkflowGraph 経由で系全体最適化したい場合は target に graph を "
        "渡してください）",
    )


def _reinject_vars(slot: Slot, candidate: str) -> str | None:
    """候補テキストに `Slot.vars` を再注入する（必要 `${var}` 喪失は None で fail-closed）。

    seed に**実在する全 `${var}` プレースホルダ**を検査対象とし、いずれかが候補から失われている
    場合は無効化（None を返す）。`slot.vars` のキーに限定しないのは、`vars` が optional な
    APO 公開契約で「最適化済みテンプレートは `${var}` を保持する」と約束しているため、利用者が
    vars に値を渡していない placeholder（例: `${role}` のみ seed に存在）でも保持を検証する必要が
    あるため（Codex P2 回帰防止）。検査をパスした候補は `substitute_braced` で vars を再注入する
    （braced `${name}` のみ置換し bare `$var` には触らない・未定義 key は `${...}` のまま保持・
    `Template.safe_substitute` の bare `$var` 副作用を回避・Codex 第3 round 指摘）。

    Args:
        slot: 対象スロット（seed / vars を持つ）。
        candidate: Trainer が生成した候補テキスト（`${var}` 保持想定）。

    Returns:
        vars 再注入済みテキスト。必要 `${var}` を喪失した候補なら None（fail-closed）。
    """
    seed_placeholders = extract_placeholders(slot.seed)
    for name in seed_placeholders:
        placeholder = "${" + name + "}"
        if placeholder not in candidate:
            return None
    return substitute_braced(candidate, slot.vars)
