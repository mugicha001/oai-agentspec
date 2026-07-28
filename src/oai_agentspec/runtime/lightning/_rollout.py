"""rollout オーケストレーション（`optimizer` 内部の private ヘルパ集）。

`optimize` の各 rollout で実行される (1) 候補適用 + vars 再注入 (`_apply_candidate`)、(2) target /
registry 組み直し → SDK 経由実行 + 承認自動解決 (`_make_rollout` / `_run_one`)、(3) 承認 decision
構築 + 安全不変条件チェック (`_build_decisions`) を集約する。SDK / `agentlightning` を import せず、
`_adapters` 経由で実行する（NFR-1）。

公開窓口は `optimizer.optimize` 経由のみ。テストからは `from .._rollout import _build_decisions`
等で private として参照する（`optimizer` モジュールからの再エクスポートも維持）。
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any

from ._slots_norm import _extract_case_input, _reinject_vars
from .types import (
    CoverageReport,
    FailureKind,
    OptimizeError,
    RolloutResult,
    Slot,
    _CandidateInvalid,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from ...registry import AgentRegistry

logger = logging.getLogger(__name__)


async def _observe_route_steps(
    *,
    target: Any,
    registry: AgentRegistry | None,
    slots: dict[str, Slot],
    seeds: dict[str, str],
    case: Any,
    approvals: Callable[[dict], bool] | None,
    tool_mocks: dict[str, dict[str, Any]] | None,
    context_factory: Callable[[], Any] | None,
) -> tuple[tuple[str, ...], bool]:
    """seed 状態で 1 case を rollout し `(route_steps, interrupted)` を返す（pre-flight 用）。

    `_make_rollout` の rollout closure（reward 呼び出しを含む）を経由せず、`_apply_candidate`
    + `_target.normalize` + `_run_one` を直接組んで observation.route.steps のみ採取する。
    approvals / tool_mocks / context_factory は本番 rollout と同値で素通しし routing 挙動を
    同じ状態で観測する。observation で到達が確認できた agent 名は `RunOutcome.interrupted=True`
    （承認保留などで未完走）でもそのまま返す — 観測された到達は常に陽性証拠であり、破棄すると
    coverage が過小になり偽陽性 fail を作る。`interrupted` フラグは診断カウンタ用に返すのみで、
    union 集計の可否には使わない。

    Args:
        target: 最適化対象（graph / AgentSpec）。
        registry: 現行 registry（graph 経路必須）。
        slots: 正規化済み Slot mapping。
        seeds: seed テキスト mapping。
        case: train ケース（`OptimizeCase` / dict / 利用者定義任意型）。
        approvals: 承認ガード（本番同値素通し）。
        tool_mocks: ツールモック（本番同値素通し）。
        context_factory: context 生成 factory（引数なし・本番同値素通し）。

    Returns:
        `(route_steps: tuple[str, ...], interrupted: bool)`。candidate 無効（`_apply_candidate`
        が None / `_CandidateInvalid`）の場合のみ `((), False)`（観測なし）。interrupted の
        場合も観測できた route_steps を返す。

    Note:
        candidate 無効以外の例外（API のレート制限・タイムアウト等）は捕捉せず伝播させ、
        呼び出し側の `optimize()` が `TRAINER_FAILED` へ変換する。観測失敗を空観測へ
        degrade させると `covered` が過小になり偽陽性の fail-fast を生むため、意図的に
        握らない。
    """
    from ..._adapters import DefaultRunnerAdapter
    from . import _target as target_mod

    applied = _apply_candidate(
        target=target,
        registry=registry,
        slots=slots,
        rebind=None,
        candidate=dict(seeds),
    )
    if applied is None:
        # 必要 ${var} 喪失 / _CandidateInvalid で無効化 → 空観測（union 集計から除外）
        return (), False
    opt_target, opt_registry = applied

    context = context_factory() if context_factory is not None else None
    agent, replaced = target_mod.normalize(opt_target, opt_registry, tool_mocks=tool_mocks)
    try:
        outcome, observation, _fired = await _run_one(
            agent=agent,
            case=case,
            replaced=replaced,
            approvals=approvals,
            runner=DefaultRunnerAdapter(),
            context=context,
        )
    except _CandidateInvalid:
        return (), False

    return tuple(step.agent for step in observation.route.steps), bool(outcome.interrupted)


async def _check_route_coverage(
    *,
    target: Any,
    registry: AgentRegistry | None,
    slots: dict[str, Slot],
    seeds: dict[str, str],
    train: Sequence[Any],
    approvals: Callable[[dict], bool] | None,
    tool_mocks: dict[str, dict[str, Any]] | None,
    context_factory: Callable[[], Any] | None,
) -> None:
    """train 全件で seed 状態 rollout を観測し未到達 slot があれば fail-fast する（pre-flight）。

    `slots` の名前集合と `train` 各 case の route_steps union を突き合わせ、差分（未到達 slot）
    が非空なら `OptimizeError(FailureKind.CONFIG_MISSING, ..., coverage=CoverageReport(...))`
    を送出する。raise 前に `logger.warning` に集計行を出力し、rollout で費やした API コストが
    診断可能な形で保全されるようにする（`OptimizeError.coverage` と log の 2 段保全）。

    union 算入規則: 観測が空（`route_steps=()`）の case は union に寄与しない（空集合の union は
    恒等）。interrupted になった case も、そこまでに観測できた到達は算入する（到達観測は常に
    陽性証拠で、破棄は wireable な構成を fail させる偽陽性にしかならない）。`interrupted_cases`
    は診断カウンタとしてのみ集計し、判定には使わない。

    Args:
        target: 最適化対象。
        registry: 現行 registry。
        slots: 正規化済み Slot mapping。
        seeds: seed テキスト mapping。
        train: 学習ケース列。
        approvals: 承認ガード（本番同値素通し）。
        tool_mocks: ツールモック（本番同値素通し）。
        context_factory: context 生成 factory（引数なし・本番同値素通し）。

    Raises:
        OptimizeError: 未到達 slot 集合が非空のとき `FailureKind.CONFIG_MISSING`
            （`coverage` 添付・`logger.warning` 出力）。
    """
    covered: set[str] = set()
    per_case: list[tuple[Any, tuple[str, ...]]] = []
    interrupted = 0
    for case in train:
        steps, was_interrupted = await _observe_route_steps(
            target=target,
            registry=registry,
            slots=slots,
            seeds=seeds,
            case=case,
            approvals=approvals,
            tool_mocks=tool_mocks,
            context_factory=context_factory,
        )
        per_case.append((case, steps))
        if was_interrupted:
            interrupted += 1
        if steps:
            covered |= set(steps)

    missing = set(slots.keys()) - covered
    if not missing:
        return

    report = CoverageReport(
        covered=frozenset(covered),
        missing=frozenset(missing),
        per_case=tuple(per_case),
        interrupted_cases=interrupted,
    )
    logger.warning(
        "[preflight] coverage insufficient: covered=%s, missing=%s, cases=%d, interrupted=%d",
        sorted(covered),
        sorted(missing),
        len(per_case),
        interrupted,
    )
    raise OptimizeError(
        FailureKind.CONFIG_MISSING,
        f"train 全 {len(per_case)} 件の seed 状態 rollout で slot {sorted(missing)!r} が"
        "一度も routing されませんでした（route coverage 不足）。"
        "以下のいずれかで対処してください:\n"
        "  1. train ケースに未到達 slot を経由する入力を追加する\n"
        "     （例: OptimizeCase(input=..., expected_route=[..., '<slot>']) を train へ）\n"
        "  2. HandoffGraph の handoff edge / trigger 条件を見直し seed prompt から到達可能にする\n"
        "  3. 意図的に到達不能な slot を最適化対象へ含めている場合は "
        "optimize(..., skip_coverage_check=True) で本検査を無効化する\n"
        f"  検出 slot（到達済み）: {sorted(covered)}\n"
        f"  未到達 slot: {sorted(missing)}\n"
        "  診断情報は error.coverage（CoverageReport）から取得できます。",
        coverage=report,
    )


def _merge_observation(base: Any, nxt: Any) -> Any:
    """承認 resume の前後 segment の `ObservedRun` を 1 つへマージする（route / tool 連結）。

    承認自動解決ループでは 1 ケースが複数 segment（中断 → resume → ...）に分かれて実行される。
    各 segment の `route.steps` と `tool_calls` を順に連結し、`route.last_agent` は後段（`nxt`）の
    値で更新する。各 segment の route 末尾は last_agent を含むため、segment 境界では前段末尾と後段
    先頭が同一 agent になりうる（例: 単体 agent の resume で `['bot']` + `['bot']`）。境界の連続
    重複だけを 1 件畳み（後段先頭ステップの agent が累積末尾の agent と一致ならその先頭を捨て）、
    別 agent への正当な handoff-back は畳まない（連続が同一 agent のときのみ）。

    llmops `evaluator._merge_observation` と同じ規則（横断検証済み）。Codex P2 回帰防止: 承認後の
    handoff / 最終応答が segment 側で起きるケースで `route_match` / `last_agent_match` reward が
    pre-resume の古い route で誤採点されないようにする。

    Args:
        base: これまでに蓄積した `ObservedRun`（`route` / `tool_calls` を持つ不透明型）。
        nxt: 直近 segment の `ObservedRun`。

    Returns:
        route ステップとツール呼び出しを連結した新しい `ObservedRun`。
    """
    import dataclasses

    base_steps = list(base.route.steps)
    nxt_steps = list(nxt.route.steps)
    if base_steps and nxt_steps and base_steps[-1].agent == nxt_steps[0].agent:
        nxt_steps = nxt_steps[1:]
    merged_route = dataclasses.replace(
        base.route,
        steps=base_steps + nxt_steps,
        last_agent=nxt.route.last_agent or base.route.last_agent,
    )
    return dataclasses.replace(
        base,
        route=merged_route,
        tool_calls=list(base.tool_calls) + list(nxt.tool_calls),
    )


def _make_rollout(
    *,
    target: Any,
    registry: AgentRegistry | None,
    slots: dict[str, Slot] | None,
    rebind: Callable[[Any], Any] | None,
    reward: Callable[[RolloutResult], float | Awaitable[float]],
    tool_mocks: dict[str, dict[str, Any]] | None,
    approvals: Callable[[dict], bool] | None,
    context_factory: Callable[[], Any] | None = None,
) -> Callable[[dict[str, str], Any], Awaitable[float]]:
    """候補スロット mapping + 1 ケースから報酬を返す rollout callable を組む。

    各 rollout で (1) 候補に vars を再注入し（必要 `${var}` 喪失は 0.0 で fail-closed）、(2)
    `Slot.build` から自動導出した rebind（または利用者供給 rebind）で target を組み直し、(3)
    `_target.normalize` → `_adapters` `run_with_observation` で実行し、(4) plain な `RolloutResult`
    を利用者 reward へ渡す。

    Args:
        target: 最適化対象。
        registry: specs 供給経路 / 既定 build の spec 解決元。
        slots: 自動 rebind 経路の `{名前: Slot}`（None で生 seed + rebind 経路）。
        rebind: 生 seed 経路の rebind（候補 → 宣言物 or registry）。
        reward: rollout の `RolloutResult` から報酬を返す callable（同期 / async）。
        tool_mocks: rollout 安全化のモック dict（llmops 経路を再利用）。
        approvals: 承認自動解決ポリシー（llmops 経路を再利用）。
        context_factory: rollout ごとに新鮮な context を生成する引数なし callable（None で
            context=None）。戻り値は初回 `run_with_observation` の `context=` へ素通しする（FR-2）。

    Returns:
        `(候補スロット mapping, ケース) -> 報酬`（non-blocking）。
    """

    async def rollout(candidate: dict[str, str], case: Any) -> float:
        applied = _apply_candidate(
            target=target, registry=registry, slots=slots, rebind=rebind, candidate=candidate
        )
        if applied is None:
            # 必要 ${var} を喪失した候補は無効化（fail-closed・低評価）。
            return 0.0
        opt_target, opt_registry = applied

        from ..._adapters import DefaultRunnerAdapter
        from . import _target as target_mod

        # rollout ごとに新鮮な context を生成する（1 rollout = 1 context・承認 resume ループ内は
        # SDK RunState 内包 context を再利用するため `_run_one` へは初回分のみ渡す）。
        context = context_factory() if context_factory is not None else None
        agent, replaced = target_mod.normalize(opt_target, opt_registry, tool_mocks=tool_mocks)
        try:
            outcome, observation, fired_approvals = await _run_one(
                agent=agent,
                case=case,
                replaced=replaced,
                approvals=approvals,
                runner=DefaultRunnerAdapter(),
                context=context,
            )
        except _CandidateInvalid:
            # vars=callable 経路: dynamic instructions closure が SDK Runner.run 実行時に
            # `_CandidateInvalid` を投げるケース（境界マーカー崩れ or vars_fn 非 dict 戻り値）。
            # C1 対応: build 時ではなく rollout 時に発火する例外を per-candidate 無効化経路
            # （reward 0.0）で吸収する。他の例外（TypeError / RuntimeError 等）は
            # 暴走防止のため伝搬させる。
            return 0.0
        result = RolloutResult(
            case=case,
            output=outcome.final_output or "",
            tool_calls=[tc.tool for tc in observation.tool_calls],
            fired_approvals=fired_approvals,
            route_steps=[step.agent for step in observation.route.steps],
            # ObservedRoute.last_agent は str（空文字なら応答なし）→ 空は None に正規化する。
            last_agent=observation.route.last_agent or None,
        )
        scored = reward(result)
        if inspect.isawaitable(scored):
            scored = await scored
        return float(scored)

    return rollout


def _apply_candidate(
    *,
    target: Any,
    registry: AgentRegistry | None,
    slots: dict[str, Slot] | None,
    rebind: Callable[[Any], Any] | None,
    candidate: dict[str, str],
) -> tuple[Any, AgentRegistry | None] | None:
    """候補を vars 再注入の上で適用し `(target', registry')` を返す（fail-closed は None）。

    自動 rebind 経路（`slots` あり）では各スロットの `build` から target / registry を組み直す。
    生 seed 経路（`slots` なし）では利用者 `rebind`（単一候補 / 候補 mapping）へ委譲する。必要
    `${var}` を喪失した候補は None（無効化・低評価）。

    Args:
        target: 最適化対象。
        registry: specs 供給経路 / 既定 build の spec 解決元。
        slots: 自動 rebind 経路の `{名前: Slot}`（None で生 seed + rebind 経路）。
        rebind: 生 seed 経路の rebind。
        candidate: Trainer が生成した候補スロット mapping（`{名前: 候補テキスト}`）。

    Returns:
        `(組み直した target, 組み直した registry)`。必要 `${var}` 喪失で無効化なら None。
    """
    from ...spec import AgentSpec

    if slots is None:
        # 生 seed 経路: rebind へ委譲（単一候補なら値、複数なら mapping を渡す）。
        payload: Any = next(iter(candidate.values())) if len(candidate) == 1 else dict(candidate)
        return rebind(payload), registry  # type: ignore[misc]

    # 自動 rebind 経路: 各スロットの build で組み直す。
    reinjected: dict[str, str] = {}
    for name, slot in slots.items():
        text = candidate.get(name, slot.seed)
        applied = _reinject_vars(slot, text)
        if applied is None:
            return None
        reinjected[name] = applied

    if isinstance(target, AgentSpec):
        # 単一スロット = target 自身の build で AgentSpec を組み直す。
        slot = next(iter(slots.values()))
        try:
            built = slot.build(reinjected[slot.name])
        except _CandidateInvalid:
            # per-candidate 無効化経路（reward 0.0・`_reinject_vars` の None と同一扱い）。
            # C3 対応: 内部の `_CandidateInvalid` sentinel のみを catch する。旧 shape の
            # 利用者 `build=` が raise する generic な `ValueError` は silent 化せず伝搬させる
            # （fail-closed 診断性の維持）。境界マーカー崩れは `_new_default_build` が
            # `_CandidateInvalid` で signal する。
            return None
        return built, registry

    # 横断（グラフ）: registry をクローンし各スロットの名前の spec を build 済み spec へ差し替える。
    if registry is None:
        raise OptimizeError(
            FailureKind.CONFIG_MISSING,
            "グラフ最適化には registry が必須です（optimize(registry=...) で渡してください）",
        )

    def _transform(spec: AgentSpec) -> AgentSpec:
        slot = slots.get(spec.name)
        if slot is None:
            return spec
        return slot.build(reinjected[spec.name])

    try:
        cloned = registry.clone(transform_spec=_transform)
    except _CandidateInvalid:
        # per-candidate 無効化経路（reward 0.0・`_reinject_vars` の None と同一扱い）。
        # C3 対応: 内部の `_CandidateInvalid` sentinel のみを catch する。旧 shape の
        # 利用者 `build=` が raise する generic な `ValueError` や `_resolve_spec` の
        # config 不整合 `ValueError` は silent 化せず伝搬させる（fail-closed 診断性の維持）。
        return None
    return target, cloned


async def _run_one(
    *,
    agent: Any,
    case: Any,
    replaced: frozenset[tuple[str, str]],
    approvals: Callable[[dict], bool] | None,
    runner: Any,
    context: Any = None,
) -> tuple[Any, Any, list[str]]:
    """1 rollout を実行し `(RunOutcome, ObservedRun, fired_approvals)` を返す（承認自動解決対応）。

    `run_with_observation` で 1 回実行し、`approvals` 指定かつ中断した場合は承認自動解決ループ
    （llmops 経路の `apply_approvals` / `resume_with_observation`・安全不変条件は `replaced` で
    担保）で完了まで再開する。承認後の segment で観測されたツール呼び出しは初回 observation の
    `tool_calls` に追記してマージし、reward callable が `tool_match` 等の recall 評価で正しく
    score できるようにする。中断時の `pending`（承認ゲート発火）は各ラウンドで `fired_approvals`
    にツール名で連結し、`approval_match` 等の recall reward が承認ゲート挙動を判定できるように
    する（approve / reject を問わず収集・llmops `ObservedApproval` と同型）。`approvals` 未指定で
    中断した場合も初回 pending の tool_name は `fired_approvals` に含める（reward 側で参照可）。

    Args:
        agent: 正規化済み実行 Agent（不透明型）。
        case: rollout への入力ケース。
        replaced: 実差し替えした `(agent, tool)` ペア集合（approve 認可の安全不変条件）。
        approvals: 承認自動解決ポリシー（None で自動解決しない）。
        runner: `DefaultRunnerAdapter` インスタンス。
        context: 初回実行へ素通しする共有 context（FR-2・None で従来どおり）。承認 resume は
            SDK `RunState` 内包の context を再利用するため `resume_with_observation` へは渡さない。

    Returns:
        `(RunOutcome, マージ済み ObservedRun, fired_approvals ツール名列)` の plain タプル。
    """
    from ..._adapters import apply_approvals, resume_with_observation

    case_input = _extract_case_input(case)
    outcome, observation = await runner.run_with_observation(agent, case_input, context=context)

    # 初回中断時の pending を fired に積む（approvals 有無に関わらず承認ゲート発火を観測する）。
    fired_approvals: list[str] = []
    if outcome.interrupted:
        fired_approvals.extend(p.get("tool_name", "") for p in outcome.pending)

    if approvals is None or not outcome.interrupted:
        return outcome, observation, fired_approvals

    _MAX_ROUNDS = 5
    merged_observation = observation
    current = outcome
    for _round in range(_MAX_ROUNDS):
        if not current.interrupted:
            break
        decisions = _build_decisions(
            list(current.pending), resolver=approvals, replaced_tools=replaced
        )
        applied = apply_approvals(current.state, decisions)
        if not applied.applied:
            break
        current, segment = await resume_with_observation(agent, current.state)
        # resume 後の segment は tool_calls だけでなく **route_steps / last_agent も**
        # 反映する必要がある（Codex P2 回帰防止）。承認後にハンドオフ / 最終応答が segment 側で
        # 起きるケースで、`route_match` / `last_agent_match` reward が pre-resume の古い経路で
        # 採点されないよう、全観測をマージする（境界重複 = 累積末尾 agent と segment 先頭 agent
        # が同一なら 1 件だけ畳む・llmops `_merge_observation` と同等規則）。
        merged_observation = _merge_observation(merged_observation, segment)
        if current.interrupted:
            # resume 後に新たに発火した承認ゲートを fired へ追記する（後段ラウンドの recall 用）。
            fired_approvals.extend(p.get("tool_name", "") for p in current.pending)
    return current, merged_observation, fired_approvals


def _build_decisions(
    pending: list[dict[str, str]],
    *,
    resolver: Callable[[dict], bool],
    replaced_tools: frozenset[tuple[str, str]],
) -> list[dict[str, Any]]:
    """承認待ち列に resolver を適用し `apply_approvals` 用 decisions を構築する（安全不変条件）。

    approve を返した `(agent_name, tool_name)` が `replaced_tools`（実際にモックへ差し替えた集合）に
    含まれない場合は `OptimizeError(FailureKind.CONFIG_MISSING)`（本物の危険ツールを構造的に
    実行させない・fail-closed）。reject は安全（ツール非実行）。llmops `evaluator._build_decisions`
    と同じ安全不変条件を踏襲しつつ、APO の LitAgent 経路で握り潰されず構造化失敗種別へ昇格する
    よう `OptimizeError` を使う（FR-8 / NFR-8 整合）。

    Args:
        pending: 中断時点の承認待ち一覧（plain dict 列・`agent_name` を含む）。
        resolver: `(pending_dict) -> bool`。approve(True) / reject(False) を返す。
        replaced_tools: 実際にモックへ差し替えた `(agent, tool)` ペアの集合。

    Returns:
        `apply_approvals` 用 decisions（`{"call_id", "decision", "rejection_message"}` 列）。

    Raises:
        OptimizeError: approve を返した `(agent, tool)` が `replaced_tools` に無い場合
            （`FailureKind.CONFIG_MISSING`）。
    """
    decisions: list[dict[str, Any]] = []
    for item in pending:
        tool_name = item.get("tool_name", "")
        call_id = item.get("call_id", "")
        agent_name = item.get("agent_name", "")
        approved = bool(resolver(dict(item)))
        if approved and (agent_name, tool_name) not in replaced_tools:
            # NFR-8 安全不変条件: 本物の危険ツールを構造的に実行させない。FR-8 整合のため
            # `OptimizeError(CONFIG_MISSING)` に倒し、APO LitAgent 経路でも握り潰されず利用者へ
            # 明確な失敗種別として伝える。
            raise OptimizeError(
                FailureKind.CONFIG_MISSING,
                f"approval resolver が approve を返したツール {tool_name!r}（agent "
                f"{agent_name!r}）がモックへ差し替えられていません。本物の危険ツールの実行を防ぐ"
                "ため、approve するツールは optimize(tool_mocks={agent: {tool: 値}}) で当該 agent "
                "のモック実装を指定し、かつ実際に差し替え可能（spec ベース登録の FunctionTool）"
                "である必要があります",
            )
        if approved:
            decisions.append({"call_id": call_id, "decision": "approve"})
        else:
            decisions.append(
                {
                    "call_id": call_id,
                    "decision": "reject",
                    "rejection_message": "rejected by optimization resolver",
                }
            )
    return decisions
