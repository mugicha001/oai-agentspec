"""APO の使いやすさヘルパ（`prompt_slot` / `prompt_slots`・seed 取得 + 既定 build 内包）。

`prompt_slot` は `PromptStore` の公開メソッド（`get` / `compose`）を**読み取るのみ**で seed
（vars 未展開・`${var}` 保持）と固定部分（base / parts）を取得し、候補テキストから `AgentSpec` を
構築する `build` を内包した `Slot` を返す。`build` 省略時の既定 build は registry 登録 `AgentSpec`
を複製して `instructions` のみ候補で差し替える（tools / handoffs / model 等は登録 spec から複製・
利用者は再宣言不要）。registry 未解決かつ build 省略は fail-closed エラー。`prompt_slots` は列挙
エージェント分の `Slot` を一括生成する。

seed の解決は `PromptLayout` を尊重する: 既定で `store.compose(agent=tune, vars=None)`（`agents`
サブディレクトリの `agent:<tune>` セグメントとして解決）を優先し、見つからなければ `store.get(tune)`
（root 直下の flat 配置）にフォールバックする。これにより `PromptStore("prompts",
PromptLayout(base="base", parts="parts", agents="agents"))` の標準レイアウト（`agents/<name>.md`）
でも、`PromptLayout(base="", parts="", agents="")` の flat レイアウトでも seed を解決できる。

vars は seed に展開せず `${var}` プレースホルダを保持し、最適化対象外（不変）として `Slot` に保持
する。rollout 時の vars 再注入は `optimizer` が `Slot.vars` を使って
`_placeholders.substitute_braced`（braced `${name}` のみ・bare `$var` 不変）で行う（本モジュール
は vars を展開しない）。`PromptStore` / `AgentRegistry` は読み取り / 複製経由のみで一切改変しない
（依存方向 `runtime/lightning → core(prompts/registry)` の一方向・FR-9）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...prompts import PromptResolutionError
from ...registry import _copy_spec
from ._placeholders import compose_with_vars, extract_placeholders
from .types import FailureKind, OptimizeError, Slot

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ...prompts import PromptStore
    from ...registry import AgentRegistry
    from ...spec import AgentSpec


def _ensure_fixed_vars_present(name: str, fixed: str, vars_dict: dict[str, Any]) -> None:
    """`fixed`（base + parts）に含まれる `${var}` がすべて `vars_dict` に存在することを検査する。

    `_default_build` は rollout 時 `compose_with_vars(fixed, candidate, vars_dict)` で fixed の
    vars を再注入する（内部で `substitute_braced` が braced `${name}` のみを置換し、未知の
    placeholder は `${name}` のまま残す・fail-open）。利用者が `vars` の指定を漏らすと `${role}`
    等が literal なまま `agent.instructions` に残り、APO スコアが silent に劣化する。slot 構築
    段階で fail-closed に倒し、利用者に即時通知する。

    Args:
        name: slot 名（エラーメッセージ用）。
        fixed: 固定部分テキスト（`${var}` 保持）。
        vars_dict: 利用者指定の vars（`{name: value}`）。

    Raises:
        OptimizeError: `fixed` 内 placeholder のいずれかが `vars_dict` に欠落している場合
            （`FailureKind.CONFIG_MISSING`・fail-closed）。
    """
    if not fixed:
        return
    placeholders = extract_placeholders(fixed)
    missing = sorted(placeholders - set(vars_dict.keys()))
    if not missing:
        return
    names = ", ".join(repr(n) for n in missing)
    raise OptimizeError(
        FailureKind.CONFIG_MISSING,
        f"slot {name!r} の base/parts に含まれる `${{var}}` のうち vars に未指定: {names}。"
        "vars に値を渡すか、対象 placeholder を base/parts から削除してください "
        "（未指定だと rollout 時に literal な `${{var}}` が agent.instructions に残ります）",
    )


def _compose_fixed(store: PromptStore, *, base: str | None, parts: Sequence[str]) -> str:
    """固定部分（base / parts）を vars 未展開（`${var}` 保持）で合成する。

    `PromptStore.compose(vars=None)` は各セグメントを `${var}` プレースホルダ保持のまま `\\n\\n`
    連結した静的 str を返す（読み取りのみ・`PromptStore` 非改変）。base / parts いずれも未指定の
    場合は空文字を返す。

    Args:
        store: プロンプトストア（読み取り専用）。
        base: base:<name> セグメント（共通ベース名）。None で base なし。
        parts: part:<name> セグメント名の列。

    Returns:
        固定部分の合成済みテキスト（`${var}` 保持・base/parts 非指定なら空文字）。
    """
    if base is None and not parts:
        return ""
    composed = store.compose(base=base, parts=tuple(parts), vars=None)
    return composed if isinstance(composed, str) else ""


def _default_build(
    registry: AgentRegistry | None,
    name: str,
    fixed: str,
    vars: dict[str, Any] | None = None,  # noqa: A002 - PromptStore.compose の引数名に追従
) -> Callable[[str], AgentSpec]:
    """既定 build（registry 登録 spec を複製し instructions のみ候補で差し替え）を作る。

    候補テキストは固定部分（`fixed`）と `\\n\\n` 連結して instructions にする（固定部分が空なら
    候補テキストのみ）。registry から対象 spec を解決して `_copy_spec` で独立コピーし、
    `instructions` だけ差し替えた新 `AgentSpec` を返す（tools / handoffs / model 等は複製で保持・
    利用者 registry の登録 spec は不変）。registry 未供給 / 対象 spec 未登録は fail-closed エラー。

    base/parts に `${var}` プレースホルダが含まれる場合、`vars` を再注入する（`_reinject_vars`
    は候補テキストにのみ適用されるため、固定部分にも同じ vars 再注入を行わないと rollout 時に
    literal な `${var}` が instructions に残ってしまう）。合成は `_placeholders.compose_with_vars`
    に委譲（`_compose_full` と Single Source of Truth・braced のみ置換で bare `$var` 副作用なし）。
    `vars` 未指定または空なら `fixed` はそのまま使う（既存挙動と互換）。

    Args:
        registry: 既定 build の spec 解決元。None なら fail-closed（呼び出し時に build できない）。
        name: 対象エージェント名（registry から解決する spec 名）。
        fixed: 固定部分の合成済みテキスト（`${var}` 保持・空文字可）。
        vars: `${var}` 置換値（None / 空 dict なら fixed を素通し）。

    Returns:
        候補テキスト → `AgentSpec` の build 関数。

    Raises:
        ValueError: registry 未供給で既定 build が spec を解決できない場合（fail-closed）。
    """
    if registry is None:
        raise ValueError(
            f"prompt_slot の既定 build には registry が必須です（slot {name!r}）。"
            "optimize / prompt_slots に registry を渡すか build= を明示してください"
        )

    vars_dict = dict(vars or {})

    def build(candidate: str) -> AgentSpec:
        spec = _resolve_spec(registry, name)
        # 合成規則は `_compose_full` と共通（compose_with_vars は fixed 側にのみ braced `${var}`
        # を再注入し、tune 側は APO 候補本体で温存・bare `$var` には触らない）。
        instructions = compose_with_vars(fixed, candidate, vars_dict)
        import dataclasses

        return dataclasses.replace(_copy_spec(spec), instructions=instructions)

    return build


def _resolve_spec(registry: AgentRegistry, name: str) -> AgentSpec:
    """registry から対象 `AgentSpec` を解決する（spec ベース登録のみ・fail-closed）。

    Args:
        registry: spec 解決元の registry。
        name: 対象エージェント名。

    Returns:
        登録済み `AgentSpec`。

    Raises:
        ValueError: 対象 spec が registry に未登録（または factory 登録で spec 実体を持たない）
            場合。
    """
    spec = registry._specs.get(name)
    if spec is None:
        raise ValueError(
            f"既定 build が登録 spec を解決できません: {name!r} が registry に未登録です"
            "（build= を明示するか spec を register してください・fail-closed）"
        )
    return spec


def _resolve_seed(store: PromptStore, tune: str) -> str:
    """`tune` の seed テキスト（`${var}` 保持）を `PromptLayout` 優先で解決する。

    `store.compose(agent=tune, vars=None)` で `agents/<tune>.md` を優先し、`PromptResolutionError`
    のときは `store.get(tune).body`（root 直下の flat 配置）にフォールバックする。両者で見つから
    なければ `KeyError`（`PromptResolutionError` は `KeyError` のサブクラス）を伝搬する。`vars=None`
    のとき compose は静的 str を返すため、戻り値型は str に確定する（callable 経路に入らない）。

    Args:
        store: プロンプトストア（読み取り専用）。
        tune: チューニング対象セグメント名（agent / 直下ファイル名）。

    Returns:
        seed テキスト（`${var}` 保持）。

    Raises:
        KeyError: `tune` が `agents/<tune>` にも root 直下にも存在しない場合
            （`PromptResolutionError` の伝搬を含む）。
    """
    try:
        composed = store.compose(agent=tune, vars=None)
    except PromptResolutionError:
        # `agents/<tune>` に未在 → root 直下の flat 配置にフォールバック（後方互換）。
        return store.get(tune).body
    if isinstance(composed, str):
        return composed
    # vars=None のとき compose は静的 str を返す前提。callable 経路は到達しない。
    return store.get(tune).body  # pragma: no cover - 防御的フォールバック


def prompt_slot(
    store: PromptStore,
    registry: AgentRegistry | None = None,
    *,
    tune: str,
    base: str | None = None,
    parts: Sequence[str] = (),
    vars: dict[str, Any] | None = None,  # noqa: A002 - PromptStore.compose の引数名に追従
    build: Callable[[str], AgentSpec] | None = None,
) -> Slot:
    """合成プロンプト最適化の定型を畳む `Slot` を生成する（FR-9）。

    `store.get(tune).body`（公開・frozen・`${var}` 保持）から vars 未展開の seed を読み取り、
    固定部分（base / parts）を `store.compose(vars=None)` で読み取る（いずれも `PromptStore`
    非改変）。`build` 省略時の既定 build は registry 登録 `AgentSpec` を複製し `instructions` のみ
    候補で差し替える（registry 必須・未解決は fail-closed）。`vars` は seed に展開せず `Slot.vars`
    に保持し、rollout 時に `optimizer` が再注入する（vars は最適化対象外・不変）。

    Args:
        store: プロンプトストア（読み取り専用・`get` / `compose` のみ参照）。
        registry: 既定 build の spec 解決元（build= 省略時に使用）。build= 明示時は不要。
        tune: チューニング対象セグメント名（seed = `store.get(tune).body`・`${var}` 保持）。
        base: 固定部分の base:<name> セグメント（共通ベース名）。
        parts: 固定部分の part:<name> セグメント名の列。
        vars: `${var}` 置換値（最適化対象外・rollout 再注入）。
        build: 候補テキスト → `AgentSpec` の build 関数（明示時は registry 不要）。

    Returns:
        seed / build / vars を保持した `Slot`（`optimizer` が rebind を自動導出する）。

    Raises:
        KeyError: `tune` セグメントが `agents/<tune>` でも root 直下でも解決できない場合
            （`PromptResolutionError` は `KeyError` のサブクラスとして伝搬する）。
        ValueError: build 省略かつ registry 未供給の場合（既定 build が spec を解決できない）。
    """
    seed = _resolve_seed(store, tune)
    fixed = _compose_fixed(store, base=base, parts=parts)
    effective_build = build
    # 既定 build のときだけ `Slot.fixed` を埋める（その build が `fixed + candidate` を agent の
    # instructions として組み立てるため、合成済み full テキストの concept が成立する）。利用者が
    # custom `build` を渡した場合は build がどう組み立てるかライブラリ側で保証できないので
    # `fixed` は空のままにし、`OptimizeResult.seed` / `prompt` も tune そのものを返す
    # （誤った合成済みテキストを見せない）。
    slot_fixed = ""
    if effective_build is None:
        # base/parts に含まれる `${var}` プレースホルダがすべて vars に存在することを slot 構築時
        # に検証する（不足を許すと rollout 時に literal `${role}` が agent.instructions に残り
        # APO スコアが silent に劣化する）。fail-closed で構築段階で利用者へ通知する。
        _ensure_fixed_vars_present(tune, fixed, dict(vars or {}))
        effective_build = _default_build(registry, tune, fixed, vars=vars)
        slot_fixed = fixed
    return Slot(
        name=tune,
        seed=seed,
        build=effective_build,
        vars=dict(vars or {}),
        fixed=slot_fixed,
    )


def prompt_slots(
    store: PromptStore,
    registry: AgentRegistry,
    agents: Sequence[str],
    *,
    base: str | None = None,
    parts: Sequence[str] = (),
    vars: dict[str, Any] | None = None,  # noqa: A002 - prompt_slot の引数名に追従
) -> dict[str, Slot]:
    """列挙エージェント分の `Slot` を一括生成し `{名前: Slot}` の mapping を返す（FR-9）。

    各エージェントについて `prompt_slot` 相当（seed = 対象セグメントの vars 未展開テンプレート・
    `${var}` 保持・既定 build = 登録 spec 複製で instructions 差し替え）を生成する。生成 mapping を
    `optimize(graph, slot=slots, ...)` に渡せば rebind 自動導出と合わせてグラフ全体 APO が実質 2 行
    で書ける。最適化対象は列挙したエージェントのみ（未掲載のプロンプトは固定）。`PromptStore` は
    公開 `compose` / `get` を読み取るのみ・registry は読み取り複製のみ（非改変）。

    Args:
        store: プロンプトストア（読み取り専用）。
        registry: 既定 build の spec 解決元（必須・各 slot 共通）。
        agents: 最適化対象とするエージェント名の列。
        base: 固定部分の base:<name> セグメント（全 slot 共通）。
        parts: 固定部分の part:<name> セグメント名の列（全 slot 共通）。
        vars: `${var}` 置換値（全 slot 共通・最適化対象外・rollout 再注入）。

    Returns:
        `{エージェント名: Slot}` の mapping。

    Raises:
        KeyError: いずれかの対象セグメントが `store.get` で解決できない場合。
        ValueError: いずれかの対象 spec が registry に未登録の場合（既定 build の fail-closed）。
    """
    return {
        name: prompt_slot(store, registry, tune=name, base=base, parts=parts, vars=vars)
        for name in agents
    }
