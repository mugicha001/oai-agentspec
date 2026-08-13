"""起動時検証（`ActionPlanner.validate()` の実装・設計 §4a-2 の 9 種）。

宣言簿と結線の不整合を、候補が押された瞬間ではなくアプリ起動時にまとめて落とす（FR-3）。
LLM / network / env を一切参照しないため同期関数である。

方針:
- **検査ごとに全違反を集約して 1 例外**にする（最初の 1 件で止めない）。検査「間」は集約せず、
  ある検査が違反を見つけた時点で送出する。1 回の起動で全種別を直したい要求はここまでは
  求めておらず、種別をまたいで集約すると例外の型（`KeyError` / `ValueError`）を選べない。
- 例外の型は検査 1 / 9 が `KeyError`、検査 2-8 が `ValueError`（§4a-2 の表）。
  **`PromptResolutionError` は `KeyError` 派生**であるため、検査 3 は当該例外だけを捕まえて
  `ValueError` へ `raise ... from exc` で変換する。`except Exception` にすると lockdown 時の
  `PromptTemplateIntegrityError` まで飲み込み、fail-closed が弱まる（設計 §3.12）。
- セグメント解決は公開経路 `prompts.compose(layout=[seg], vars={})` のみを使い、
  プレースホルダは `string.Template.get_identifiers()` で取る（private API と `re` を使わない）。
- 判定対象のスコープは検査ごとに異なる。検査 4 は**マージ後**、検査 5 は**カタログ全体**、
  検査 8 は**当該 `ActionSpec` 自身の宣言のみ**（マージ結果を見ると、カタログ既定を置いた
  だけで `by_llm` を持たない全アクションが違反になる）。
- 「埋まる経路が宣言に無いパラメータ」は WARNING 1 行のみで例外にしない（検査種別に数えない）。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from string import Template
from typing import Any

from ...prompts import PromptResolutionError
from .actions import ActionCatalog, ActionSpec, resolve_prompt, resolve_prompt_vars
from .binding import LLMFiller

logger = logging.getLogger(__name__)


def _validate_catalog(
    catalog: ActionCatalog,
    *,
    registry: Any,
    prompts: Any = None,
    guardrail_registry: Any = None,
    llm_filler: LLMFiller | None = None,
    context: Any = None,
) -> None:
    """宣言簿と結線の整合を検査する。違反があれば送出し、無ければ何も返さない。

    Args:
        catalog: 検査対象の宣言簿。
        registry: 実行先エージェントを解決する `AgentRegistry`（`names()` を使う）。
        prompts: セグメントを解決する `PromptStore`。`None` でもセグメント宣言が 1 件も
            無ければ正常に完了する。
        guardrail_registry: ガードレール登録名を解決する簿冊（`names()` を使う）。
        llm_filler: 不足パラメータの埋め方。`None` なら検査 9 の対象が無い。
        context: run context の代表インスタンス。渡したときだけパスの構造検査（検査 7）を行う。

    Returns:
        `None`。全検査を通過したことを表す。

    Raises:
        KeyError: 未登録の `action_agent`（検査 1）または解決できないガードレール登録名
            （検査 9）がある場合。いずれも全違反を集約した 1 件を送出する。
        ValueError: 検査 2-8 のいずれかに違反がある場合、および `guardrails` が非空なのに
            解決簿が未結線の場合（検査 9・名前の不在ではなく結線の欠落）。
        RuntimeError: セグメント宣言があるのに `prompts` が未結線の場合。黙って検査 3-5 を
            飛ばすと起動時検証が空振りする（設計 §3.4a）。
        PromptTemplateIntegrityError: lockdown 済みの `PromptStore` が manifest 未掲載の
            セグメントを拒否した場合。捕捉せずそのまま伝播させる（設計 §3.12）。
    """
    specs = [catalog.get(action_id) for action_id in catalog.names()]

    _check_action_agents(specs, registry)
    _check_label_placeholders(specs)
    bodies = _load_segment_bodies(catalog, specs, prompts)
    _check_template_placeholders(catalog, specs, bodies)
    _check_unused_prompt_vars(catalog, specs, bodies)
    _check_prompt_vars_conflicts(catalog, specs)
    _check_context_paths(catalog, specs, context)
    _check_effective_prompt_declarations(specs)
    _check_guardrail_names(llm_filler, guardrail_registry)
    _warn_parameters_without_fill_path(specs)
    return None


# ---- 検査 1: action_agent が registry に存在するか ----


def _check_action_agents(specs: Sequence[ActionSpec], registry: Any) -> None:
    """全 `action_agent` が `registry.names()` に存在することを検査する（検査 1）。

    Args:
        specs: 検査対象の宣言。
        registry: 実行先エージェントの登録簿。

    Raises:
        KeyError: 未登録の `action_agent` がある場合。全違反を集約した 1 件を送出する。
    """
    known = set(registry.names())
    missing = sorted({spec.action_agent for spec in specs if spec.action_agent not in known})
    if missing:
        raise KeyError(f"unknown action_agent (not registered in the AgentRegistry): {missing}")


# ---- 検査 2: label のプレースホルダ ⊆ 宣言パラメータ名 ----


def _check_label_placeholders(specs: Sequence[ActionSpec]) -> None:
    """`label` が宣言パラメータだけで render できることを検査する（検査 2）。

    2 つを見る。(a) `${...}` が宣言パラメータ名に含まれること（等しさではなく包含）、
    (b) `label` 全体が `Template.substitute` を通ること。(b) を足すのは
    `Template.get_identifiers()` が `"100$ "` のような不正なプレースホルダを黙って無視する
    ためで、(a) だけでは render 不能な `label` が起動時検証を通過してしまう。render 側
    （`slots.ActionPlan.label`）は宣言の取りこぼしを黙って通さないよう意図的に
    `safe_substitute` ではなく `substitute` を使っており、その前提をここで成立させる。

    (b) の置換値には宣言パラメータ名に加えて (a) で挙がった未宣言の名前も入れる。入れないと
    `substitute` が `KeyError` になり、(a) で報告済みの違反が別種の例外へすり替わる。

    Args:
        specs: 検査対象の宣言。

    Raises:
        ValueError: 宣言されていない名前を `label` が参照している場合、または `label` が
            render 不能な場合。検査ごとに全違反を 1 つの例外へ集約する。
    """
    violations: list[str] = []
    for spec in specs:
        declared = {parameter.name for parameter in spec.parameters}
        identifiers = set(Template(spec.label).get_identifiers())
        missing = sorted(identifiers - declared)
        if missing:
            violations.append(f"{spec.action_id}: undeclared placeholders {missing}")
        try:
            Template(spec.label).substitute(dict.fromkeys(declared | identifiers, ""))
        except ValueError as exc:
            violations.append(f"{spec.action_id}: {exc}")
    if violations:
        raise ValueError(f"label declarations cannot be rendered as declared: {violations}")


# ---- 検査 3: prompt セグメントが prompts で解決できるか ----


def _segment_names(catalog: ActionCatalog, specs: Sequence[ActionSpec]) -> list[str]:
    """カタログ全体で宣言されたセグメント名を重複なく宣言順に集める。

    `param(prompt=...)` のセグメントも含める。穴埋め時に実際に合成へ積まれる（FR-6）ため、
    外すとその分だけ検査 3 が空振りする。

    Args:
        catalog: カタログ既定を持つ宣言簿。
        specs: 検査対象の宣言。

    Returns:
        セグメント名のリスト（重複排除済み・初出順）。
    """
    names: dict[str, None] = dict.fromkeys(catalog.prompt)
    for spec in specs:
        for segment in spec.prompt:
            names.setdefault(segment)
        for parameter in spec.parameters:
            if parameter.prompt is not None:
                names.setdefault(parameter.prompt)
    return list(names)


def _load_segment_bodies(
    catalog: ActionCatalog, specs: Sequence[ActionSpec], prompts: Any
) -> dict[str, str]:
    """全セグメントを解決し、本文を返す（検査 3 + `prompts` 未結線の規則）。

    本文は検査 4 / 5 のプレースホルダ集合の材料でもあるため、ここで 1 度だけ合成して配る。

    Args:
        catalog: カタログ既定を持つ宣言簿。
        specs: 検査対象の宣言。
        prompts: セグメントを解決する `PromptStore`。

    Returns:
        セグメント名から本文（`${...}` を残したまま）への辞書。セグメント宣言が 1 件も
        無い場合は空の辞書。

    Raises:
        RuntimeError: セグメント宣言があるのに `prompts` が未結線の場合（設計 §3.4a）。
        ValueError: 解決できないセグメントがある場合。全違反を集約し、原因の
            `PromptResolutionError` を `__cause__` に残す。
    """
    names = _segment_names(catalog, specs)
    if not names:
        return {}
    if prompts is None:
        raise RuntimeError(
            "prompt segments are declared but no PromptStore is wired; pass "
            f"bind(prompts=...) to validate them: {names}"
        )

    bodies: dict[str, str] = {}
    unresolved: list[str] = []
    cause: PromptResolutionError | None = None
    for name in names:
        try:
            # PromptResolutionError だけを捕まえる。lockdown の PromptTemplateIntegrityError は
            # 捕捉せず伝播させる（設計 §3.12）。
            bodies[name] = prompts.compose(layout=[name], vars={})
        except PromptResolutionError as exc:
            unresolved.append(name)
            if cause is None:
                cause = exc
    if unresolved:
        # PromptResolutionError は KeyError 派生。素通しさせると検査 1 / 9 と型で区別できない。
        raise ValueError(f"prompt segments cannot be resolved: {unresolved}") from cause
    return bodies


# ---- 検査 4: テンプレのプレースホルダ ⊆ prompt_vars のキー（マージ後） ----


def _placeholders(bodies: Mapping[str, str], names: Sequence[str]) -> set[str]:
    """指定セグメントの本文に残る `${...}` の名前を集める。

    Args:
        bodies: セグメント名から本文への辞書。
        names: 対象のセグメント名。

    Returns:
        プレースホルダ名の集合。
    """
    found: set[str] = set()
    for name in names:
        found.update(Template(bodies[name]).get_identifiers())
    return found


def _scanned_segments(catalog: ActionCatalog, spec: ActionSpec) -> tuple[str, ...]:
    """当該アクションの穴埋めで合成へ積まれるセグメント名を集める（検査 4 / 5 の走査対象）。

    マージ後のプロンプト（カタログ既定 + アクション宣言）に加えて、`param(prompt=...)` の
    セグメントも含める。検査 3 が既に本文を解決している範囲と一致させることで、パラメータ側
    テンプレのプレースホルダの見落とし（過小検知）と、そこでのみ使うキーの未使用判定
    （過剰検知）の双方を塞ぐ。

    Args:
        catalog: カタログ既定を持つ宣言簿。
        spec: 対象の宣言。

    Returns:
        セグメント名の tuple（重複排除済み・カタログ / アクション / パラメータの宣言順）。
    """
    names: dict[str, None] = dict.fromkeys(resolve_prompt(catalog, spec))
    for parameter in spec.parameters:
        if parameter.prompt is not None:
            names.setdefault(parameter.prompt)
    return tuple(names)


def _check_template_placeholders(
    catalog: ActionCatalog, specs: Sequence[ActionSpec], bodies: Mapping[str, str]
) -> None:
    """テンプレの `${...}` が `prompt_vars` のキーで供給されることを検査する（検査 4）。

    判定対象は**マージ後**（`_scanned_segments` / `resolve_prompt_vars`）。カタログ既定が
    供給する変数で足りているなら、アクション側が宣言していなくても違反ではない。
    走査するテンプレートはカタログ既定・アクション宣言・`param(prompt=...)` の全セグメント。

    Args:
        catalog: カタログ既定を持つ宣言簿。
        specs: 検査対象の宣言。
        bodies: セグメント名から本文への辞書。

    Raises:
        ValueError: 供給されないプレースホルダがある場合。差分だけを挙げる。
    """
    violations: list[str] = []
    for spec in specs:
        supplied = set(resolve_prompt_vars(catalog, spec))
        missing = sorted(_placeholders(bodies, _scanned_segments(catalog, spec)) - supplied)
        if missing:
            violations.append(f"{spec.action_id}: {missing}")
    if violations:
        raise ValueError(f"template placeholders are not supplied by prompt_vars: {violations}")


# ---- 検査 5: prompt_vars のキーがどのテンプレでも未使用でないか（カタログ全体） ----


def _check_unused_prompt_vars(
    catalog: ActionCatalog, specs: Sequence[ActionSpec], bodies: Mapping[str, str]
) -> None:
    """宣言した `prompt_vars` のキーがどこかで使われることを検査する（検査 5）。

    判定対象は**カタログ全体**のテンプレート集合（カタログ既定・アクション宣言・
    `param(prompt=...)` の全セグメント）。`ActionCatalog.prompt_vars` のキーは
    いずれか 1 つの `ActionSpec` のテンプレートで使われていれば足りる。

    Args:
        catalog: カタログ既定を持つ宣言簿。
        specs: 検査対象の宣言。
        bodies: セグメント名から本文への辞書。

    Raises:
        ValueError: どのテンプレートにも現れないキーがある場合。効果のない宣言であり、
            変数名の打ち間違いがそのまま素通りする経路になる。
    """
    used: set[str] = set()
    for spec in specs:
        used |= _placeholders(bodies, _scanned_segments(catalog, spec))
    declared: set[str] = set(catalog.prompt_vars)
    for spec in specs:
        declared |= set(spec.prompt_vars)
    unused = sorted(declared - used)
    if unused:
        raise ValueError(f"prompt_vars keys are not used by any template: {unused}")


# ---- 検査 6: 同一 prompt_vars キーが異なるパスへ宣言されていないか ----


def _check_prompt_vars_conflicts(catalog: ActionCatalog, specs: Sequence[ActionSpec]) -> None:
    """同一キーが異なるパスへ宣言されていないことを検査する（検査 6）。

    判定対象は宣言そのもの（マージ結果ではない）。マージ後を見るとアクション側の宣言が
    カタログ側を上書きして矛盾が消えるため、食い違いを検出できない。

    Args:
        catalog: カタログ既定を持つ宣言簿。
        specs: 検査対象の宣言。

    Raises:
        ValueError: 同一キーに 2 通り以上のパスが宣言されている場合。同一キー・同一パスの
            重複は許容する。
    """
    paths: dict[str, set[str]] = {}
    sources: list[Mapping[str, str]] = [catalog.prompt_vars, *(spec.prompt_vars for spec in specs)]
    for source in sources:
        for key, path in source.items():
            paths.setdefault(key, set()).add(path)
    conflicts = sorted(f"{key}: {sorted(found)}" for key, found in paths.items() if len(found) > 1)
    if conflicts:
        raise ValueError(f"prompt_vars keys are declared with different paths: {conflicts}")


# ---- 検査 7: context 指定時のパス構造解決 ----


def _is_resolvable(obj: Any, path: str) -> bool:
    """`.` 区切りのパスを**構造的に**辿れるかどうかを返す。

    辿り方（mapping ならキー・それ以外は属性・`.` で分割して再帰）は
    `actions._resolve_path` と同一に保つこと。ただし当該関数は「解決できない」と
    「解決できた値が `None`」の双方に `None` を返すため、この検査には使えない
    （値が `None` であることは違反ではない・FR-3）。

    中間セグメントが `None` になった時点で `True` を返し、以降は辿らない。
    `_resolve_path` はこの状態を「解決できないので次のパスを試す」正常系として扱うため
    （FR-3 L152 / FR-5 L179）、ここだけが違反にすると代表 context の任意項目が未設定なだけで
    アプリが起動できなくなる。緩めるのはこの 1 点のみで、**非 `None` のオブジェクトに無い
    属性 / キーは引き続き違反**である（宣言の打ち間違いは起動時に落とす）。

    Args:
        obj: 起点のオブジェクト（run context の代表インスタンス）。
        path: `.` 区切りのパス。

    Returns:
        全セグメントを辿れたか、途中で `None` に行き当たったなら `True`。
    """
    current = obj
    for segment in path.split("."):
        if current is None:
            return True
        if isinstance(current, Mapping):
            if segment not in current:
                return False
            current = current[segment]
        else:
            if not hasattr(current, segment):
                return False
            current = getattr(current, segment)
    return True


def _check_context_paths(catalog: ActionCatalog, specs: Sequence[ActionSpec], context: Any) -> None:
    """`from_context` と `prompt_vars` のパスが構造的に解決できることを検査する（検査 7）。

    Args:
        catalog: カタログ既定を持つ宣言簿。
        specs: 検査対象の宣言。
        context: run context の代表インスタンス。`None` なら検査そのものを行わない。

    Raises:
        ValueError: 辿れないパスがある場合。`from_context` 由来と `prompt_vars` 由来を
            1 つの例外へ集約する。
    """
    if context is None:
        return
    declared: dict[str, None] = {}
    for spec in specs:
        for parameter in spec.parameters:
            declared.update(dict.fromkeys(parameter.from_context))
        declared.update(dict.fromkeys(resolve_prompt_vars(catalog, spec).values()))
    unresolvable = sorted(path for path in declared if not _is_resolvable(context, path))
    if unresolvable:
        raise ValueError(
            f"declared paths cannot be resolved against the given context: {unresolvable}"
        )


# ---- 検査 8: by_llm=True が 0 件なのに prompt / prompt_vars を宣言していないか ----


def _check_effective_prompt_declarations(specs: Sequence[ActionSpec]) -> None:
    """穴埋め経路が無いのにプロンプト宣言を持たないことを検査する（検査 8）。

    判定対象は**当該 `ActionSpec` 自身の宣言のみ**。マージ結果を見ると、カタログ既定を
    置いただけで `by_llm` を持たない全アクションが違反になってしまう。

    宣言と数えるのはアクション側の `prompt` / `prompt_vars` に加えて、穴埋め段でしか効かない
    パラメータ側の 2 つ（`param(prompt=...)` と `max_suggestions>1`）である。検査 4 / 5 の
    走査範囲が `param(prompt=...)` を含むよう拡張されたのと同じ取りこぼしであり、片方だけ
    見ると「効かないのに宣言されている」状態が残る。`confirm=True` は穴埋め段に限らず
    ユーザー確認一般に関わるため対象にしない。`max_suggestions` は既定値（1）を宣言と
    数えない（数えると `by_llm=True` を持たない全アクションが違反になる）。

    Args:
        specs: 検査対象の宣言。

    Raises:
        ValueError: `by_llm=True` のパラメータが 1 件も無いのに、穴埋め段でしか効かない宣言を
            持つ場合。効果のない宣言であり、宣言したのに効かない状態を残さない。
    """
    violations = sorted(
        spec.action_id
        for spec in specs
        if _has_fill_only_declaration(spec)
        and not any(parameter.by_llm for parameter in spec.parameters)
    )
    if violations:
        raise ValueError(
            "prompt / prompt_vars / param(prompt=...) / max_suggestions are declared but no "
            f"parameter is filled by the prediction stage (by_llm=True): {violations}"
        )


def _has_fill_only_declaration(spec: ActionSpec) -> bool:
    """穴埋め段でしか効かない宣言を 1 つでも持つかを返す（検査 8 の判定材料）。

    Args:
        spec: 判定対象の宣言。

    Returns:
        アクション側の `prompt` / `prompt_vars`、またはパラメータ側の `prompt` /
        `max_suggestions>1` を持つなら `True`。
    """
    if spec.prompt or spec.prompt_vars:
        return True
    return any(parameter.prompt or parameter.max_suggestions > 1 for parameter in spec.parameters)


# ---- 検査 9: LLMFiller.guardrails の登録名が解決できるか ----


def _check_guardrail_names(llm_filler: LLMFiller | None, guardrail_registry: Any) -> None:
    """`guardrails` の登録名が解決できることを検査する（検査 9）。

    Args:
        llm_filler: 不足パラメータの埋め方。`None` なら対象が無い。
        guardrail_registry: ガードレール登録名の解決簿。

    Raises:
        ValueError: `guardrails` が非空なのに解決簿が未結線の場合。名前が見つからないの
            ではなく結線が欠けているため、`KeyError` とは区別する。
        KeyError: 解決できない登録名がある場合。全違反を集約した 1 件を送出する。
    """
    if llm_filler is None or not llm_filler.guardrails:
        return
    if guardrail_registry is None:
        raise ValueError(
            "LLMFiller.guardrails is declared but no guardrail_registry is wired: "
            f"{list(llm_filler.guardrails)}"
        )
    known = set(guardrail_registry.names())
    missing = sorted({name for name in llm_filler.guardrails if name not in known})
    if missing:
        raise KeyError(f"unknown guardrail name (not registered in the registry): {missing}")


# ---- WARNING: 埋まる経路が宣言に無いパラメータ（検査種別には数えない） ----


def _warn_parameters_without_fill_path(specs: Sequence[ActionSpec]) -> None:
    """埋まる経路が宣言に無いパラメータを WARNING 1 行で報告する。

    値の解決順の第 1 優先は候補が載せた値であり、それが来るかどうかは宣言から導けない。
    したがって例外にはせず、`filled_by_candidate=True` の明示宣言があれば報告もしない
    （警告の常態化を避ける）。

    Args:
        specs: 検査対象の宣言。
    """
    without_path = [
        f"{spec.action_id}.{parameter.name}"
        for spec in specs
        for parameter in spec.parameters
        if not (
            parameter.from_context
            or parameter.by_llm
            or parameter.has_default
            or parameter.filled_by_candidate
        )
    ]
    if without_path:
        logger.warning(
            "%d parameters have no declared fill path (they are filled only by the candidate "
            "or by user input): %s",
            len(without_path),
            without_path,
        )
