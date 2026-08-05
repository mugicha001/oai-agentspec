"""APO の使いやすさヘルパ（`prompt_slot` / `prompt_slot_factory`・seed 取得 + 既定 build 内包）。

`prompt_slot` は `PromptStore` の公開メソッド（`get` / `compose`）を**読み取るのみ**で seed
（vars 未展開・`${var}` 保持）と固定部分（base / parts）を取得し、候補テキストから `AgentSpec` を
構築する `build` を内包した `Slot` を返す。`build` 省略時の既定 build は registry 登録 `AgentSpec`
を複製して `instructions` のみ候補で差し替える（tools / handoffs / model 等は登録 spec から複製・
利用者は再宣言不要）。registry 未解決かつ build 省略は fail-closed エラー。`prompt_slot_factory` は
共通既定値を束ねて agent ごとの差分だけで `Slot` を生成する callable を返す。

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
from ._placeholders import (
    BOUNDARY_PREFIX,
    compose_from_marked,
    extract_placeholders,
)
from .types import FailureKind, OptimizeError, Slot, SlotSegment, _CandidateInvalid

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ...prompts import PromptStore
    from ...registry import AgentRegistry
    from ...spec import AgentSpec


def _ensure_fixed_vars_present(name: str, fixed: str, vars_dict: dict[str, Any]) -> None:
    """`fixed`（base + parts）に含まれる `${var}` がすべて `vars_dict` に存在することを検査する。

    `_new_default_build` は rollout 時に全非 tune セグメント（base / parts / 非 tune の agent 含む）
    へ `${var}` を注入する（`substitute_braced` が braced `${name}` のみを置換し、未知の placeholder
    は `${name}` のまま残す・fail-open）。利用者が `vars` の指定を漏らすと `${role}` 等が literal
    なまま `agent.instructions` に残り、APO スコアが silent に劣化する。slot 構築段階で fail-closed
    に倒し、利用者に即時通知する。

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


def _new_default_build(
    registry: AgentRegistry | None,
    name: str,
    segments: tuple[SlotSegment, ...],
    vars_dict: dict[str, Any],
    *,
    vars_fn: Callable[[Any], dict[str, Any]] | None = None,
) -> Callable[[str], AgentSpec]:
    """新 shape 用の既定 build（候補を境界マーカーで分割し構成順に再インターリーブする）。

    候補テキストの分割・strip・構成順の再インターリーブは `_placeholders.compose_from_marked`
    （SSoT ヘルパ・内部で `split_marked` -> `compose_segments`）へ委譲して `instructions` を
    組み立てる（固定セグメントには `vars_dict` を注入・tune 側は `${var}` 温存）。同ヘルパは
    OptimizeResult 合成側からも呼ばれ、"OptimizeResult.prompt == rollout instructions" 契約の
    drift を防ぐ。`compose_from_marked` が `None`（マーカー崩れ）を返した場合は `ValueError` に
    倒す（`_apply_candidate` が catch し reward 0.0 の per-candidate 無効化経路へ）。registry から
    対象 spec を解決して `_copy_spec` で独立コピーし、`instructions` だけ差し替えた新 `AgentSpec`
    を返す（tools / handoffs / model 等は複製で保持）。

    Args:
        registry: 既定 build の spec 解決元。None なら fail-closed（呼び出し時に build できない）。
        name: 対象エージェント名（registry から解決する spec 名）。
        segments: 構成順の `SlotSegment` タプル（tune / 固定の別を保持）。
        vars_dict: 固定セグメントへ注入する `${var}` 置換値（未指定キーは `${name}` 保持）。
        vars_fn: vars=callable のときの動的 vars 生成関数（`context -> dict`）。非 None のとき
            build は静的合成でなく `(context, agent) -> str` の動的 instructions を据え、rollout
            時に `vars_fn(context)` を評価して固定セグメントへ注入する（None なら従来の静的経路）。

    Returns:
        候補テキスト → `AgentSpec` の build 関数。

    対象 spec が `instructions_append`（run スコープの追記）を宣言している場合、その spec の APO は
    未サポートであり本関数の呼び出し時点（rollout を待たず）に `ValueError` で拒否する。

    Raises:
        ValueError: registry 未供給で既定 build が spec を解決できない場合（fail-closed）、または
            対象 spec が `instructions_append` を宣言している場合（APO 未サポート）。
    """
    if registry is None:
        raise ValueError(
            f"prompt_slot の既定 build には registry が必須です（slot {name!r}）。"
            "optimize / prompt_slot / prompt_slot_factory に registry を渡すか build= を明示"
            "してください"
        )
    _reject_instructions_append(registry, name)

    def build(candidate: str) -> AgentSpec:
        import dataclasses

        if vars_fn is not None:
            # vars=callable 経路: rollout 時に context から vars を生成する動的 instructions を
            # 据える。SDK 規約 `(context, agent) -> str` の callable を instructions にし、
            # `_agent` は未使用（compose(vars=callable) と同一パターン）。context は duck typing
            # で扱い SDK 型の import に依存しない（NFR-1）。
            def instructions(context: Any, _agent: Any) -> str:
                dynamic_vars = vars_fn(context)
                if not isinstance(dynamic_vars, dict):
                    # vars callable の戻り値が dict でないと `substitute_braced` 深部で cryptic な
                    # TypeError になる。ここで明快なメッセージ付き _CandidateInvalid に先取りして
                    # 倒す（rollout closure が catch し reward 0.0 経路へ・C1 対応）。
                    raise _CandidateInvalid(
                        f"slot {name!r}: vars callable の戻り値は dict である必要があります"
                        f"（実際の型: {type(dynamic_vars).__name__}）。"
                        "compose(vars=callable) と同一の契約で `Callable[[Any], dict[str, Any]]` を"
                        "返すよう修正してください"
                    )
                # ADR 0005 契約: rollout 時に tune 側の `${var}` も dynamic_vars で
                # substitute する。`substitute_tune=True` を渡し compose_from_marked 内で
                # tune テキストにも `substitute_braced` を適用させる（固定側は既に substitute
                # 済み・二重 pass すると fixed vars の値内 `${...}` が意図せず再解釈される
                # ため full 文字列に対する post-substitute は使わない）。
                full = compose_from_marked(segments, candidate, dynamic_vars, substitute_tune=True)
                if full is None:
                    raise _CandidateInvalid(
                        f"slot {name!r}: 境界マーカー `${{oas_boundary_N}}` の欠落・重複・"
                        "順序不整合を rollout 時に検出（動的 instructions 経路）。"
                        "候補テキストは無効化されます"
                    )
                return full

            spec = _resolve_spec(registry, name)
            return dataclasses.replace(_copy_spec(spec), instructions=instructions)

        # 静的 instructions（既存経路）: 候補の分割・strip・再インターリーブは
        # `compose_from_marked`（SSoT）へ委譲する。rollout 実体（本 build）と OptimizeResult
        # 合成（optimizer）の両経路が同一ヘルパを呼ぶことで "OptimizeResult.prompt == rollout
        # instructions" 契約の drift を防ぐ。
        full_text = compose_from_marked(segments, candidate, vars_dict)
        if full_text is None:
            # 境界マーカーの欠落・重複・順序不整合を検出。候補テキストは無効化される
            # （`_apply_candidate` が `_CandidateInvalid` を catch し reward 0.0 の per-candidate
            # 無効化経路へ倒す・FU C3 対応で内部 sentinel に一本化）。
            raise _CandidateInvalid(
                f"slot {name!r}: 境界マーカー `${{oas_boundary_N}}` の欠落・重複・順序不整合を"
                "検出。候補テキストは無効化されます（rollout の候補単位 fail-closed 経路）"
            )
        spec = _resolve_spec(registry, name)
        return dataclasses.replace(_copy_spec(spec), instructions=full_text)

    return build


def _build_marked_seed(segments: tuple[SlotSegment, ...]) -> str:
    """tune セグメントの本文を構成順に連結し、2 個以上なら境界マーカーを挟んだ seed を作る。

    `tune=True` のセグメント本文を構成順に取り出し、`n_tune >= 2` のとき隣接本文の間に
    `${oas_boundary_1}` .. `${oas_boundary_(n_tune-1)}` を（前後 `\\n\\n` で囲んで）挿入する。
    `n_tune == 1` はマーカーなしで本文そのもの・`n_tune == 0` は空文字（防御的）。

    Args:
        segments: 構成順の `SlotSegment` タプル。

    Returns:
        マーカー入りの tune 連結 seed（`${var}` 保持・tune が空なら空文字）。
    """
    tune_texts = [segment.text for segment in segments if segment.tune]
    if not tune_texts:
        return ""
    joined = tune_texts[0]
    for index, text in enumerate(tune_texts[1:], start=1):
        joined += f"\n\n${{{BOUNDARY_PREFIX}{index}}}\n\n{text}"
    return joined


def _reject_instructions_append(registry: AgentRegistry, name: str) -> None:
    """対象 spec が `instructions_append` を宣言していれば早期に fail-closed で拒否する。

    既定 build は候補テキストを `instructions` に据える（`vars=callable` 経路では lib が生成した
    動的 callable を据える）ため、`instructions_append` を持つ spec を対象にすると rollout 中の
    `build_agent` で「callable instructions と instructions_append は併用不可」エラーになる。その
    エラーは利用者が制御できない lib 生成 callable を指しており原因が分からず、失敗が rollout まで
    遅延する。原因の分かるメッセージで構築時に倒す。

    spec が registry に未登録の場合は何もしない（未登録の診断は build 呼び出し時の
    `_resolve_spec` が担う契約を変えないため）。

    Args:
        registry: spec 解決元の registry。
        name: 対象エージェント名。

    Raises:
        ValueError: 対象 spec が `instructions_append` を宣言している場合。
    """
    spec = registry._specs.get(name)
    if spec is None or not spec.instructions_append:
        return
    raise ValueError(
        f"AgentSpec {name!r} は instructions_append（run スコープの追記）を宣言しており、"
        "その spec の APO（prompt_slot による最適化）は未サポートです。既定 build は候補テキストを"
        " instructions に据えるため追記との併用が成立しません。追記を宣言しない spec を"
        "対象にするか、追記の合成を含めて自前で組み立てる build= を明示してください"
    )


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
    agent: str | None = None,
    *,
    base: str | None = None,
    parts: Sequence[str] = (),
    layout: Sequence[str] | None = None,
    tune: str | Sequence[str] | None = None,
    vars: dict[str, Any]  # noqa: A002 - PromptStore.compose の引数名に追従
    | Callable[[Any], dict[str, Any]]
    | None = None,
    build: Callable[[str], AgentSpec] | None = None,
) -> Slot:
    """合成プロンプト最適化の定型を畳む `Slot` を生成する（FR-9・compose 一致の新 shape）。

    `PromptStore.compose` と同じ使い方で構成を組み立て、最適化対象セグメントを `tune` セレクタで
    選ぶ。`agent=` または `layout=` のいずれかが必須（詳細は ADR 0007）。

    ディスパッチ規則:
        - `agent=None` + `layout=None`: `OptimizeError(CONFIG_MISSING)`（fail-closed）。
        - `agent` 指定 or `layout` 指定: 構成順のセグメント列を確定し `tune` セレクタで最適化対象を
          選ぶ（`Slot.segments` を設定する）。

    Args:
        store: プロンプトストア（読み取り専用・`get` / `compose` のみ参照）。
        registry: 既定 build の spec 解決元（build= 省略時に使用）。build= 明示時は不要。
        agent: agent:<name> セグメント（spec 解決名を兼ねる・compose と同名同位置）。
        base: 構成セグメントの base:<name>。
        parts: 構成セグメントの part:<name> の列。
        layout: セグメント参照の明示列（compose と同一意味論・指定時は agent/base/parts を無視）。
        tune: 最適化対象セレクタ（plain 名または qualified 参照・None は agent セグメントのみ）。
        vars: `${var}` 置換値（最適化対象外・rollout 再注入）。dict / None のほか
            `Callable[[context], dict]` を渡すと動的 vars 生成となり、既定 build が rollout 時に
            `(context, agent) -> str` の動的 instructions を組み立てる（`vars_fn` に保持・
            build= との併用は fail-closed）。
        build: 候補テキスト → `AgentSpec` の build 関数（明示時は registry 不要）。

    Returns:
        seed / build / vars / segments を保持した `Slot`。

    Raises:
        OptimizeError: `agent=` / `layout=` のいずれも未指定、または fail-closed 検証違反
            （`FailureKind.CONFIG_MISSING`）。
        KeyError: セグメントが store で解決できない場合（`PromptResolutionError` の伝搬を含む）。
        ValueError: build 省略かつ registry 未供給の場合（既定 build が spec を解決できない）、
            または対象 spec が `instructions_append` を宣言している場合（追記を持つ spec の APO は
            未サポート・rollout を待たず構築時に拒否する）。
    """
    if agent is None and layout is None:
        raise OptimizeError(
            FailureKind.CONFIG_MISSING,
            "prompt_slot は agent= または layout= のいずれかが必須です（詳細は ADR 0007）。"
            "旧: prompt_slot(store, reg, tune='bot')  "
            "新: prompt_slot(store, reg, agent='bot')",
        )
    return _new_shape_slot(
        store,
        registry,
        agent,
        base=base,
        parts=parts,
        layout=layout,
        tune=tune,
        vars=vars,
        build=build,
    )


def _segment_text(store: PromptStore, ref: str) -> str:
    """qualified 参照 `kind:name`（`base:main` / `part:style` / `agent:triage`）の本文を読み取る。

    `agent:<name>` は `_resolve_seed`（agents/<name> 優先 + root flat フォールバック）で解決する。
    `base` / `part` は `store.compose` を単一セグメントで呼んで `${var}` 保持のまま取得する
    （読み取りのみ・`PromptStore` 非改変）。

    Args:
        store: プロンプトストア（読み取り専用）。
        ref: qualified セグメント参照（`base:main` / `part:style` / `agent:triage`）。

    Returns:
        セグメント本文（`${var}` 保持）。

    Raises:
        PromptResolutionError: `ref` の記法が不正、またはセグメントが解決できない場合。
    """
    kind, _, name = ref.partition(":")
    if not name:
        raise PromptResolutionError(
            f"不正なセグメント参照: {ref!r}（base:<name> / part:<name> / agent:<name>）"
        )
    if kind == "agent":
        return _resolve_seed(store, name)
    if kind == "base":
        composed = store.compose(base=name, vars=None)
    elif kind == "part":
        composed = store.compose(parts=[name], vars=None)
    else:
        raise PromptResolutionError(
            f"不正なセグメント参照: {ref!r}（base:<name> / part:<name> / agent:<name>）"
        )
    return composed if isinstance(composed, str) else ""


def _construction_refs(
    agent: str | None,
    *,
    base: str | None,
    parts: Sequence[str],
    layout: Sequence[str] | None,
) -> list[str]:
    """新 shape の構成セグメント参照列を確定する（compose の `_segments` と同一規則）。

    `layout` があればその並びをそのまま構成順とする（compose 同様 base/parts/agent の構成指定を
    無視）。無ければ base -> parts -> agent の順で qualified 参照列を組み立てる（`agent` は
    新 shape 経路では非 None が保証される）。

    Args:
        agent: agent:<name> の名前（layout 無し経路では非 None）。
        base: base:<name> の名前（None で base なし）。
        parts: part:<name> の名前の列。
        layout: セグメント参照の明示列（指定時は agent/base/parts を無視）。

    Returns:
        構成順の qualified 参照列。

    Raises:
        OptimizeError: `layout` が空、または重複参照を含む場合（`FailureKind.CONFIG_MISSING`）。
    """
    if layout is not None:
        refs = list(layout)
        if not refs:
            raise OptimizeError(
                FailureKind.CONFIG_MISSING, "layout が空です（参照を 1 つ以上指定）"
            )
        if len(set(refs)) != len(refs):
            raise OptimizeError(
                FailureKind.CONFIG_MISSING, f"layout に重複した参照があります: {refs}"
            )
        return refs
    refs: list[str] = []
    if base:
        refs.append(f"base:{base}")
    refs += [f"part:{p}" for p in parts]
    refs.append(f"agent:{agent}")
    return refs


def _resolve_slot_name(agent: str | None, refs: Sequence[str]) -> str:
    """新 shape の `Slot.name`（spec 解決名）を確定する。

    `agent` 明示があればそのまま採用する。`agent=None`（layout 経路）のときは refs 内の
    `agent:X` 参照がちょうど 1 つある場合のみ X を採用する（0 個 / 複数は fail-closed）。

    Args:
        agent: `agent=` 引数（None なら layout 内 `agent:X` から暗黙解決）。
        refs: 構成セグメント参照列。

    Returns:
        spec 解決名（`Slot.name`）。

    Raises:
        OptimizeError: `agent=None` かつ layout 内の `agent:` 参照が 0 個または複数個の場合
            （`FailureKind.CONFIG_MISSING`）。
    """
    if agent is not None:
        return agent
    agent_names = [ref.partition(":")[2] for ref in refs if ref.partition(":")[0] == "agent"]
    if len(agent_names) != 1:
        raise OptimizeError(
            FailureKind.CONFIG_MISSING,
            f"agent=None のとき layout 内の agent: 参照はちょうど 1 つ必要です"
            f"（検出 {len(agent_names)} 個: {agent_names}）。agent= を明示してください",
        )
    return agent_names[0]


def _resolve_tune_refs(
    name: str,
    refs: Sequence[str],
    tune: str | Sequence[str] | None,
) -> set[str]:
    """`tune` セレクタを構成参照へ照合し、最適化対象の参照集合を返す（論点 A・fail-closed）。

    各セレクタは plain 名（`main` / `style` / `triage`）と qualified 参照（`base:main` 等）の
    両形式を受理する。plain 名が複数セグメントに一致する場合は qualified 参照を要求する
    （silent な優先順位を持たせない）。`tune=None` は agent セグメントのみを既定選択する。

    Args:
        name: spec 解決名（`tune=None` 既定選択の `agent:<name>` に使う）。
        refs: 構成セグメント参照列。
        tune: 最適化対象セレクタ（None / str / Sequence）。

    Returns:
        最適化対象（`tune=True`）とする参照の集合。

    Raises:
        OptimizeError: セレクタ空 / 未解決 / plain 名衝突 / 名寄せ重複 / `tune=None` 既定の
            agent セグメント不在（いずれも `FailureKind.CONFIG_MISSING`）。
    """
    ref_set = set(refs)
    by_plain: dict[str, list[str]] = {}
    for ref in refs:
        by_plain.setdefault(ref.partition(":")[2], []).append(ref)

    if tune is None:
        agent_ref = f"agent:{name}"
        if agent_ref not in ref_set:
            raise OptimizeError(
                FailureKind.CONFIG_MISSING,
                f"tune=None の既定（agent セグメントのみ最適化）は {agent_ref!r} が構成に必要です。"
                "tune= で最適化対象を明示してください",
            )
        return {agent_ref}

    selectors = [tune] if isinstance(tune, str) else list(tune)
    if not selectors:
        raise OptimizeError(FailureKind.CONFIG_MISSING, "tune が空です（対象を 1 つ以上指定）")

    resolved: list[str] = []
    for sel in selectors:
        if ":" in sel:
            if sel not in ref_set:
                raise OptimizeError(
                    FailureKind.CONFIG_MISSING, f"tune 参照 {sel!r} が構成に存在しません: {refs}"
                )
            resolved.append(sel)
            continue
        matches = by_plain.get(sel, [])
        if not matches:
            raise OptimizeError(
                FailureKind.CONFIG_MISSING, f"tune 名 {sel!r} が構成に存在しません: {refs}"
            )
        if len(matches) > 1:
            raise OptimizeError(
                FailureKind.CONFIG_MISSING,
                f"tune 名 {sel!r} が複数セグメントに一致します: {matches}。"
                "qualified 参照（base:/part:/agent:）で一意に指定してください",
            )
        resolved.append(matches[0])

    if len(set(resolved)) != len(resolved):
        raise OptimizeError(
            FailureKind.CONFIG_MISSING,
            f"tune に同一セグメントを指す重複参照があります（名寄せ後）: {resolved}",
        )
    return set(resolved)


def _check_reserved_prefix(
    refs: Sequence[str],
    texts: dict[str, str],
    vars_dict: dict[str, Any],
) -> None:
    """予約接頭辞 `oas_boundary_` の衝突を検査する（`_ensure_fixed_vars_present` より前）。

    セグメント本文（tune / 固定の両方）に literal `${oas_boundary_...}` が含まれる、または dict
    vars のキーが `oas_boundary_` で始まる場合は fail-closed に倒す（境界マーカー予約と衝突する
    ため・既存検査の「vars に値を渡してください」への誤誘導を防ぐ）。

    Args:
        refs: 構成セグメント参照列。
        texts: 参照 -> 本文の mapping。
        vars_dict: 利用者指定の vars（dict・callable は対象外）。

    Raises:
        OptimizeError: セグメント本文 / vars キーが予約接頭辞と衝突する場合
            （`FailureKind.CONFIG_MISSING`）。
    """
    reserved_literal = "${" + BOUNDARY_PREFIX
    for ref in refs:
        if reserved_literal in texts[ref]:
            raise OptimizeError(
                FailureKind.CONFIG_MISSING,
                f"セグメント {ref!r} の本文に予約接頭辞 `${{{BOUNDARY_PREFIX}...}}` が含まれます。"
                "境界マーカー予約のため使用できません",
            )
    for key in vars_dict:
        if key.startswith(BOUNDARY_PREFIX):
            raise OptimizeError(
                FailureKind.CONFIG_MISSING,
                f"vars キー {key!r} は予約接頭辞 `{BOUNDARY_PREFIX}` で始まります。"
                "境界マーカー予約のため使用できません",
            )


def _new_shape_slot(
    store: PromptStore,
    registry: AgentRegistry | None,
    agent: str | None,
    *,
    base: str | None,
    parts: Sequence[str],
    layout: Sequence[str] | None,
    tune: str | Sequence[str] | None,
    vars: dict[str, Any]  # noqa: A002 - PromptStore.compose の引数名に追従
    | Callable[[Any], dict[str, Any]]
    | None,
    build: Callable[[str], AgentSpec] | None,
) -> Slot:
    """新 shape の `Slot` を構築する（compose 一致・tune セレクタ・論点 A/A2/E）。

    構成順のセグメント列を確定し（`_construction_refs`）、`tune` セレクタで最適化対象を選び
    （`_resolve_tune_refs`）、`SlotSegment` 列として `Slot.segments` に設定する。予約接頭辞衝突を
    fail-closed に検査する。seed は tune セグメント本文の構成順連結（2 個以上なら境界マーカーを
    挿入）、既定 build は候補を `split_marked` -> `compose_segments` で再インターリーブする。
    新 shape では構造情報を `segments` が保持するため `Slot.fixed` は常に空に統一する。

    Args:
        store: プロンプトストア（読み取り専用）。
        registry: 既定 build の spec 解決元（build= 省略時に使用）。
        agent: agent:<name> の名前（spec 解決名を兼ねる・None なら layout から暗黙解決）。
        base: base:<name> セグメント名（layout 指定時は無視）。
        parts: part:<name> セグメント名の列（layout 指定時は無視）。
        layout: セグメント参照の明示列（指定時は agent/base/parts の構成指定を無視）。
        tune: 最適化対象セレクタ（None は agent セグメントのみ）。
        vars: `${var}` 置換値（最適化対象外・rollout 再注入）。
        build: 候補テキスト → `AgentSpec` の build 関数（明示時は registry 不要）。

    Returns:
        `segments` を設定した `Slot`（build は `_new_default_build` で候補を境界マーカー分割し
        構成順に再インターリーブする）。

    Raises:
        OptimizeError: 構成 / tune 照合 / 予約接頭辞のいずれかで fail-closed 条件に該当する場合。
        KeyError: セグメントが store で解決できない場合（`PromptResolutionError` の伝搬）。
        ValueError: build 省略かつ registry 未供給の場合、または対象 spec が
            `instructions_append` を宣言している場合（追記を持つ spec の APO は未サポート）。
    """
    refs = _construction_refs(agent, base=base, parts=parts, layout=layout)
    name = _resolve_slot_name(agent, refs)
    texts = {ref: _segment_text(store, ref) for ref in refs}

    # vars=callable（動的 vars 生成）の分岐。callable は「rollout 時に context から vars を
    # 生成する」意味論で、静的 vars を持たない（`vars_dict = {}`）。dict の isinstance 判定を
    # 併用して「callable かつ非 dict」のみを callable 経路とする。
    vars_is_callable = callable(vars) and not isinstance(vars, dict)
    if vars_is_callable and build is not None:
        # vars=callable の評価は既定 build の動的 instructions だけが担う契約。custom build は
        # どう組み立てるかライブラリ側で保証できないため、両者の同時指定を fail-closed に倒す。
        raise OptimizeError(
            FailureKind.CONFIG_MISSING,
            f"slot {name!r}: vars=callable と build= を同時に指定することはできません "
            "（vars callable の評価は既定 build のみが担う契約）",
        )
    if vars_is_callable:
        vars_fn: Callable[[Any], dict[str, Any]] | None = vars  # type: ignore[assignment]
        vars_dict: dict[str, Any] = {}  # 静的 vars なし
    else:
        vars_fn = None
        vars_dict = dict(vars or {})  # type: ignore[arg-type]

    # 予約接頭辞検査: セグメント本文は常に検査する。vars キー検査は dict/None のときのみ対象
    # （callable のときは `vars_dict = {}` で自然に空検査になる・返す辞書キーは実行時まで不明）。
    _check_reserved_prefix(refs, texts, vars_dict)
    tuned_refs = _resolve_tune_refs(name, refs, tune)

    segments = tuple(
        SlotSegment(ref=ref, text=texts[ref], tune=(ref in tuned_refs)) for ref in refs
    )

    # custom build + multi-tune の組み合わせは fail-closed で拒否する。境界マーカー入り seed の
    # 再分解は既定 build (`_new_default_build`) の compose_from_marked が担う契約で、custom build
    # はマーカーを解釈しないため OptimizeResult.seed / prompt / diff に literal で漏出する
    # （"予約接頭辞は成果物に一切現れない" 契約の違反）。
    n_tune = sum(1 for s in segments if s.tune)
    if build is not None and n_tune > 1:
        raise OptimizeError(
            FailureKind.CONFIG_MISSING,
            f"prompt_slot: custom build と multi-tune（tune=[...] で 2 個以上）は併用不可"
            f"（slot {name!r}・境界マーカー再合成は既定 build が担うため）",
        )

    # seed は tune セグメント本文の構成順連結（n_tune >= 2 なら境界マーカーを挟む）。build は
    # 候補を `split_marked` -> `compose_segments` で構成順に再インターリーブする。
    seed = _build_marked_seed(segments)
    # 固定セグメント（tune=False）の本文は `_ensure_fixed_vars_present` の検査対象として組み立てる。
    fixed_text = "\n\n".join(texts[ref] for ref in refs if ref not in tuned_refs)

    effective_build = build
    if effective_build is None:
        # 全非 tune セグメント（base / parts / 非 tune の agent 含む）の `${var}` がすべて vars に
        # 存在することを検証する（不足を許すと既定 build 経由で rollout 時に literal な `${org}` が
        # agent.instructions に残り APO スコアが silent 劣化する）。custom build 経路は組み立てを
        # ライブラリ側で保証できないため検査しない。
        # vars=callable のときは検査を免除する（compose(vars=callable) と同じ fail-open 意味論・
        # callable が返す辞書キーは実行時まで不明のため構築段階では検査できない）。
        if not vars_is_callable:
            _ensure_fixed_vars_present(name, fixed_text, vars_dict)
        effective_build = _new_default_build(registry, name, segments, vars_dict, vars_fn=vars_fn)
    # custom build 経路では `Slot.segments` を空に保つ。segments が非空だと
    # `optimizer._recompose_new_shape_results` が既定 build による segments 合成前提で
    # `OptimizeResult.prompt/seed/diff` を full 再合成で上書きしてしまい、custom build が実際に
    # 組み立てた rollout instructions と乖離する（"OptimizeResult.prompt == rollout instructions"
    # 契約の drift）。custom build は候補テキストをどう扱うか自由なため、segments ベースの再合成
    # 対象から外し、生 seed 経路と同じ「run_apo 返却をそのまま尊重する」挙動に統一する
    # （`_new_default_build` を使うときのみ segments を保持し contract を成立させる）。
    slot_segments = segments if build is None else ()
    return Slot(
        name=name,
        seed=seed,
        build=effective_build,
        vars=vars_dict,
        segments=slot_segments,
        vars_fn=vars_fn,
    )


def _merge_slot_kwargs(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """`prompt_slot_factory` の defaults / overrides を合成する（vars 双方 dict のみマージ）。

    per-agent 上書きの意味論を最小限に定義する。基本方針は「overrides で置換」であり、
    `vars` のみ双方が dict のときにマージする（同一キーは per-agent 優先・新しい dict を作る
    ため defaults 側は非破壊）。`None` を「未指定」として除去するフィルタは入れない
    （`base=None` / `vars=None` による共通指定の打ち消しを成立させるため）。callable が絡む
    組み合わせ（dict + callable / callable + dict / callable + callable）はマージせず
    置換（callable の合成意味論を `prompt_slot` に持たせないため）。

    Args:
        defaults: `prompt_slot_factory` の共通既定値。
        overrides: `make()` 呼び出しごとの per-agent 上書き。

    Returns:
        合成済み kwargs（`prompt_slot` に素通しできる形）。
    """
    merged = {**defaults, **overrides}
    default_vars, override_vars = defaults.get("vars"), overrides.get("vars")
    if isinstance(default_vars, dict) and isinstance(override_vars, dict):
        merged["vars"] = {**default_vars, **override_vars}
    return merged


def _snapshot_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    """`prompt_slot_factory` 構築時に defaults の可変コンテナを浅く複製する。

    呼び出し元が factory 生成後に元 dict / list / dict-vars を mutate しても、
    以降の `make()` 呼び出しに漏れないよう独立コピーを保持する。callable の `vars` は
    そのまま素通す（置換意味論のため）。
    """
    snapshot: dict[str, Any] = {}
    for key, value in defaults.items():
        if key in ("parts", "layout") and isinstance(value, list):
            snapshot[key] = list(value)
        elif key == "vars" and isinstance(value, dict):
            snapshot[key] = dict(value)
        else:
            snapshot[key] = value
    return snapshot


def prompt_slot_factory(
    store: PromptStore,
    registry: AgentRegistry | None = None,
    **defaults: Any,
) -> Callable[..., Slot]:
    """共通既定値を束ね、agent ごとの差分だけで `Slot` を作る callable を返す。

    per-agent 差分（`base` / `parts` / `layout` / `tune` / `vars` / `build`）を本物の kwargs で
    受けるため、キー名の typo・`agent` の二重指定は Python が `TypeError` で弾く。許可キーリスト /
    `Mapping` 型検査 / 内側値の型契約 / 未指定時のフォールバック規約は設けない。返り値 callable の
    シグネチャは `make(agent: str, **overrides) -> Slot` で、`prompt_slot` の全 kwarg を素通す。
    `vars` のみ defaults / overrides の双方が dict のときにマージし、それ以外の kwarg は置換とする
    （合成規則の詳細は `_merge_slot_kwargs`・ADR 0008）。

    Args:
        store: プロンプトストア（読み取り専用・`prompt_slot` へ素通し）。
        registry: 既定 build の spec 解決元（`prompt_slot` へ素通し・`build=` 明示時は不要）。
        **defaults: 共通既定値（`prompt_slot` の全 kwarg 名を許可・検査なし）。

    Returns:
        `make(agent: str, **overrides) -> Slot` の callable。呼び出しごとに defaults と overrides を
        合成して `prompt_slot(store, registry, agent=agent, **合成 kwargs)` を呼ぶ。
    """

    snapshot = _snapshot_defaults(defaults)

    def make(agent: str, **overrides: Any) -> Slot:
        merged = _merge_slot_kwargs(snapshot, overrides)
        return prompt_slot(store, registry, agent=agent, **merged)

    return make
