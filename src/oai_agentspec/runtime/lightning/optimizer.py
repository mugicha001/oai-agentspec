"""最適化エントリ `optimize`（公開窓口の本体）。

`optimize(宣言物, train=..., val=..., reward=..., apo_client=...)` で APO（プロンプト最適化）を
回す。データフロー: スロットから seed を取り出し → 候補生成は `_adapters/lightning.run_apo`（Agent
Lightning Trainer へ委譲）→ 各 rollout で候補に vars を再注入して target を組み直し（`Slot.build`
から rebind を自動導出・生 seed 経路は利用者 rebind）→ `_adapters` `run_with_observation` → 利用者
reward → Trainer へフィードバック → `${var}` 保持の最適化済みテキスト返却。

最適化ループ本体は `_adapters/lightning` 経由で Agent Lightning Trainer へ委譲する
（build-don't-run・独自エンジンを実装しない）。rollout 安全性は llmops の `tool_mocks` /
`approvals` 経路を `_target` / `_adapters` 経由で再利用する。`agents` / `agentlightning` は本
モジュールで import しない（NFR-1）。結果は既定で plain `OptimizeResult` 戻り値のみ（自動書込なし・
`PromptStore` 非書込）。

正規化系（`_normalize_slots` / `_seeds_of` / `_reinject_vars` / `_extract_case_input`）は
`_slots_norm` モジュールに、rollout 系（`_make_rollout` / `_apply_candidate` / `_run_one` /
`_build_decisions`）は `_rollout` モジュールに分離する。本モジュールからも後方互換のため再エクス
ポートする（テスト互換）。
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any

# 後方互換: 既存テストの `from .optimizer import _build_decisions` 等の import を維持するため、
# 内部 helper を本モジュールから再エクスポートする（`_rollout` / `_slots_norm` への移動は内部実装の
# 再編であって公開契約ではない）。
from ._placeholders import compose_from_marked, unified_diff_labeled
from ._rollout import _apply_candidate, _build_decisions, _make_rollout, _run_one  # noqa: F401
from ._slots_norm import (
    _extract_case_input,  # noqa: F401
    _normalize_slots,
    _reinject_vars,  # noqa: F401
    _seeds_of,
)
from .types import FailureKind, OptimizeError, OptimizeResult, RolloutResult, Slot

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Sequence

    from ...registry import AgentRegistry
    from .config import OptimizeConfig

# 受理する algorithm 値（APO のみ・RL は別 Issue / 別 extra）。
_ALGORITHM_APO = "apo"
_ALGORITHM_RL = "rl"

# `OptimizeConfig` で受ける passthrough 系のキー（`optimize` の直接渡し kwargs と一致）。
_DIRECT_CONFIG_KEYS = (
    "apo_client",
    "rounds",
    "concurrency",
    "timeout_seconds",
    "skip_coverage_check",
    "store",
    "apo_gradient_model",
    "apo_apply_edit_model",
    "apo_beam_width",
    "apo_branch_factor",
    "tracer",
)


async def optimize(
    target: Any,
    *,
    algorithm: str = _ALGORITHM_APO,
    train: Sequence[Any],
    val: Sequence[Any] | None = None,
    reward: Callable[[RolloutResult], float | Awaitable[float]],
    registry: AgentRegistry | None = None,
    slot: Slot | str | Iterable[Slot] | dict[str, Slot | str] | None = None,
    rebind: Callable[[Any], Any] | None = None,
    tool_mocks: dict[str, dict[str, Any]] | None = None,
    approvals: Callable[[dict], bool] | None = None,
    context_factory: Callable[[], Any] | None = None,
    config: OptimizeConfig | None = None,
    apo_client: Any = None,
    rounds: int | None = None,
    concurrency: int | None = None,
    timeout_seconds: float | None = None,
    store: Any = None,
    apo_gradient_model: str | None = None,
    apo_apply_edit_model: str | None = None,
    apo_beam_width: int | None = None,
    apo_branch_factor: int | None = None,
    tracer: Any = None,
    skip_coverage_check: bool | None = None,
) -> OptimizeResult:
    """宣言物を APO で最適化し `${var}` 保持の最適化済みテキストを返す（公開窓口・FR-2）。

    第 1 引数は最適化対象の宣言物（`AgentSpec` / `WorkflowGraph` / `HandoffGraph`）。スロットは
    常に `slot=` キーワードで渡す（FR-9）。`slot` が `prompt_slot` / `prompt_slots` の戻り値
    （`Slot` / `{名前: Slot}`）のときは各スロットの `build` から rebind を自動導出する（手書き
    rebind 不要）。生 seed（str / `{名前: str}`）のときは `rebind`（単一候補 / 候補 mapping を
    受けて宣言物を組み直す関数）を明示する。`val` は agent-lightning APO の beam search が
    validation セット必須のため本 extra でも必須（省略 / 空は `OptimizeError(CONFIG_MISSING)`・
    利用者は `train_val_split` 等で明示分割する）。

    APO 設定は 2 経路で渡せる:
        - **直接 kwargs**（推奨・最小ケース）: `optimize(..., apo_client=client, rounds=2)`
        - **`config=` 経由**（パワーユーザー）: `optimize(..., config=OptimizeConfig(...))`

    両方を同時に渡すのは曖昧で禁止（`OptimizeError(CONFIG_MISSING)`）。`apo_client` は APO の
    textual gradient 計算 / prompt edit 適用に使う `AsyncOpenAI` 互換クライアントで、APO 利用時は
    必須（未指定は `OptimizeError(CONFIG_MISSING)`・fail-closed）。

    Args:
        target: 最適化対象（AgentSpec / WorkflowGraph / HandoffGraph）。
        algorithm: 最適化系統セレクタ。既定 `"apo"`。`"rl"` は別 extra `[lightning-rl]`。
        train: 最適化 / rollout に使う入力ケース群（必須・利用者供給）。
        val: 最良候補の選定と汎化スコア確認に使う入力ケース群（**必須**・省略 / 空は
            `OptimizeError(CONFIG_MISSING)`）。
        reward: rollout の plain な観測（`RolloutResult`）から報酬を返す callable（同期 / async）。
        registry: 横断対象 / 既定 build の specs 供給経路（HandoffGraph 必須・既定 build の解決
            元）。
        slot: APO の最適化対象スロット（`Slot` / 生 seed str / `Iterable[Slot]` の列 /
            `{名前: Slot | str}` mapping）。列経路（`prompt_slot_factory` の返す `make()` を並べる
            等）は `Slot.name` をキーとする mapping へ正規化される（空 / name 重複 / 非 Slot 混在は
            `OptimizeError(CONFIG_MISSING)`・列は自動 rebind 専用）。None で target が静的
            `AgentSpec` のときのみ既定スロット（instructions 文字列）を使う。
        rebind: 生 seed 経路で候補から宣言物を組み直す関数（`Slot` 利用時は build から自動導出
            のため不要）。
        tool_mocks: agent スコープのモック dict（rollout 副作用の安全化・llmops 経路を再利用）。
        approvals: 承認自動解決ポリシー（mock-approve 相当・llmops 経路を再利用）。
        context_factory: rollout ごとに呼び出し新鮮な context を生成する引数なし callable（FR-2）。
            戻り値は各 rollout の初回 `run_with_observation` の `context=` へ素通しされ、SDK
            `Runner.run(context=...)` から動的 Instructions / ツールへ届く（`vars=callable` の
            動的 instructions も本 context を受ける）。承認 resume ループ内は SDK `RunState` 内包の
            context が再利用されるため再生成しない（1 rollout = 1 context）。None で従来どおり
            `context=None`。
        config: 実行制御設定（並列度 / ラウンド数 / タイムアウト / Store の passthrough・パワー
            ユーザー経路）。直接 kwargs と同時指定はエラー。
        apo_client: APO の textual gradient / edit 用 `AsyncOpenAI` 互換クライアント（直接渡し・
            最小ケースの推奨経路）。
        rounds: 最適化の訓練ラウンド数（APO は `beam_rounds` にマップ）。
        concurrency: rollout の並列実行数。
        timeout_seconds: APO の 1 batch のタイムアウト秒（`rollout_batch_timeout`）。
        store: Agent Lightning の Store 設定（不透明値・passthrough）。
        apo_gradient_model: APO の textual gradient 用モデル名。
        apo_apply_edit_model: APO の prompt edit 適用用モデル名。
        apo_beam_width: APO beam search の幅。
        apo_branch_factor: APO beam search の分岐数。
        tracer: agent-lightning Tracer 派生を直接渡す escape hatch（不透明・上級者向け・通常不要）。
            未指定で agent-lightning 既定の `AgentOpsTracer(agentops_managed=True,
            instrument_managed=True)` を構築する。AgentOps クラウドアップロードを抑止したい場合は
            `AGENTOPS_API_KEY` を本物のキーに設定しないこと（dummy キーで silent fail する）。
        skip_coverage_check: True で pre-flight route coverage 検証を skip する
            （既定 None は `OptimizeConfig.skip_coverage_check`（既定 False）を尊重する。
            kwarg は `bool | None` で明示 True/False を渡せる）。動的 routing 下で seed 状態のみ
            では判定できない構成の escape hatch。

    Returns:
        `${var}` 保持の最適化済みテキストを含む plain `OptimizeResult`。

    Raises:
        OptimizeError: 失敗種別 `kind` を伴う構造化エラー（FR-8）。設定不在（algorithm 不正 /
            train・reward 未供給 / slot・rebind 解決不能 / registry 不在 / 直接 kwargs と config
            の二重指定 / 未到達 slot（pre-flight route coverage 不足・
            `coverage=CoverageReport(...)` 添付））は `FailureKind.CONFIG_MISSING`、extra 不在は
            `FailureKind.EXTRA_MISSING`、Trainer / rollout / reward 実行中の失敗は
            `FailureKind.TRAINER_FAILED` で送出する。

    Note:
        pre-flight route coverage 検証（Phase 1）は seed 状態のみで実行するため、
        動的 routing 下で seed 状態と candidate 状態で経路が変わる構成は完全には
        カバーできません。API コストは `train × 1 rollout` の追加消費が発生します。
        詳細は `docs/adr/0009-lightning-preflight-coverage.md` を参照。
    """
    direct_kwargs = {
        "apo_client": apo_client,
        "rounds": rounds,
        "concurrency": concurrency,
        "timeout_seconds": timeout_seconds,
        "skip_coverage_check": skip_coverage_check,
        "store": store,
        "apo_gradient_model": apo_gradient_model,
        "apo_apply_edit_model": apo_apply_edit_model,
        "apo_beam_width": apo_beam_width,
        "apo_branch_factor": apo_branch_factor,
        "tracer": tracer,
    }
    effective_config = _resolve_config(config, direct_kwargs)

    if algorithm == _ALGORITHM_RL:
        raise OptimizeError(
            FailureKind.CONFIG_MISSING,
            "algorithm='rl'（RL によるモデル更新）は本 extra では未対応です。"
            "RL は oai-agentspec[lightning-rl]（別 Issue）で提供されます",
        )
    if algorithm != _ALGORITHM_APO:
        raise OptimizeError(
            FailureKind.CONFIG_MISSING, f"未対応の algorithm です: {algorithm!r}（受理値: 'apo'）"
        )
    if not train:
        raise OptimizeError(
            FailureKind.CONFIG_MISSING,
            "train（最適化 / rollout に使う入力ケース群）は必須です・空にできません",
        )
    if reward is None:
        raise OptimizeError(FailureKind.CONFIG_MISSING, "reward（報酬算出ロジック）は必須です")
    if not val:
        raise OptimizeError(
            FailureKind.CONFIG_MISSING,
            "APO は検証用ケース列 val が必須です（agent-lightning APO の "
            "val_dataset 必須に従う）。train_val_split(seed=0) などで分割してください",
        )
    if effective_config.apo_client is None:
        raise OptimizeError(
            FailureKind.CONFIG_MISSING,
            "APO には apo_client（AsyncOpenAI 互換クライアント）が必須です。"
            "optimize(..., apo_client=<AsyncOpenAI>) を渡すか、"
            "config=OptimizeConfig(apo_client=...) で指定してください "
            "（textual gradient 計算と prompt 編集に使用）",
        )

    slots = _normalize_slots(target, slot)
    if slots is None and rebind is None:
        raise OptimizeError(
            FailureKind.CONFIG_MISSING,
            "生 seed 経路では rebind の明示が必要です（slot が prompt_slot の戻り値なら不要）",
        )

    seeds = _seeds_of(slots, slot)
    if not seeds:
        # 自動 rebind 経路でも生 seed 経路でも、最適化対象スロットの seed が 1 件も解決でき
        # なければ APO は走らせない。`rebind` だけ与えて `slot=` を忘れた場合（HandoffGraph /
        # WorkflowGraph で `_normalize_slots` が None を返し `_seeds_of` も空 dict）に、
        # Trainer 側で seeds 不在のまま落ちる前に CONFIG_MISSING で fail-closed する
        # （Codex P2 回帰防止）。
        raise OptimizeError(
            FailureKind.CONFIG_MISSING,
            "最適化対象スロットの seed が解決できません（slot= を渡してください）。"
            "AgentSpec の既定スロットは target が静的 AgentSpec のときのみ自動導出されます。"
            "HandoffGraph / WorkflowGraph では prompt_slot / prompt_slots の戻り値か、"
            "生 seed（str / {名前: str}）を slot= に明示してください",
        )

    rollout = _make_rollout(
        target=target,
        registry=registry,
        slots=slots,
        rebind=rebind,
        reward=reward,
        tool_mocks=tool_mocks,
        approvals=approvals,
        context_factory=context_factory,
    )

    from ...spec import AgentSpec

    if (
        slots is not None
        and not effective_config.skip_coverage_check
        and not isinstance(target, AgentSpec)
    ):
        from ._rollout import _check_route_coverage

        await _check_route_coverage(
            target=target,
            registry=registry,
            slots=slots,
            seeds=seeds,
            train=train,
            approvals=approvals,
            tool_mocks=tool_mocks,
            context_factory=context_factory,
        )

    from ..._adapters import run_apo

    # `Slot.vars` を `run_apo` に渡し、`OptimizeResult.seed` / `prompt` の tune 側 `${var}` を
    # rollout 実体と同じ規則で再注入する（新 shape 再合成 `_recompose_new_shape_results` は
    # `Slot.segments` 側の SoT で固定セグメントを扱う）。生 seed + rebind 経路（slots is None）や
    # custom build 経路（segments 空）は run_apo 返却をそのまま OptimizeResult にする。
    vars_map = _build_vars_map(slots)

    # FR-8: 最適化実行（Trainer / rollout / reward）の失敗を構造化エラーへ倒す。extra 不在は
    # EXTRA_MISSING、設定不在（rollout 内で遅延検知される registry 不在等の OptimizeError）は
    # その kind を保ち、その他の実行時例外は TRAINER_FAILED へ変換して未捕捉例外で止めない。
    try:
        result = await run_apo(
            seeds=seeds,
            train=train,
            val=val,
            rollout=rollout,
            config=effective_config,
            vars_per_slot=vars_map,
        )
    except OptimizeError:
        raise
    except ImportError as exc:
        raise OptimizeError(FailureKind.EXTRA_MISSING, str(exc)) from exc
    except Exception as exc:
        raise OptimizeError(
            FailureKind.TRAINER_FAILED, f"最適化の実行に失敗しました: {exc}"
        ) from exc

    # 新 shape slot（`Slot.segments` 非空）は run_apo が tune-only テキストを返すため、返却後に
    # `compose_from_marked` で固定セグメントを含む full テキストへ再合成して prompt / seed / diff を
    # 上書きする（"OptimizeResult.prompt == rollout instructions" 契約・論点 G）。旧 shape は不変。
    return _recompose_new_shape_results(result, slots) if slots else result


def _build_vars_map(slots: dict[str, Slot] | None) -> dict[str, dict[str, Any]] | None:
    """`run_apo` へ渡す `vars_per_slot` を構築する（静的 dict は値渡し・callable は空 dict 明示）。

    旧 shape の空 vars を除外する既存契約（`if s.vars`）は維持しつつ、`vars=callable`（`vars_fn`
    を持つ）slot は静的 vars を持たない（`Slot.vars` は空 dict）ため上の filter で漏れる。当該 slot
    は最適化ループへ vars を非伝搬（rollout 時に context から動的生成）とする意味論なので、`run_apo`
    へは空 dict を明示的に渡す（論点 G）。

    Args:
        slots: 正規化済みスロット mapping（生 seed + rebind 経路では None）。

    Returns:
        `{名前: vars dict}` の mapping。`slots` が None のときは None。
    """
    if slots is None:
        return None
    vars_map = {name: dict(s.vars) for name, s in slots.items() if s.vars}
    for name, s in slots.items():
        if s.vars_fn is not None:
            vars_map[name] = {}
    return vars_map


def _recompose_new_shape_results(result: OptimizeResult, slots: dict[str, Slot]) -> OptimizeResult:
    """新 shape slot の prompt / seed / diff を full 再合成した `OptimizeResult` を返す（論点 G）。

    `Slot.segments` が非空の slot について、`run_apo` が返した tune-only の prompt / seed を
    `compose_from_marked` で固定セグメントを含む full テキストへ再合成し、diff は full 合成後の
    seed / prompt から `difflib.unified_diff` で再計算する。`vars=callable` の slot は `Slot.vars`
    が空 dict のため値を注入せず `${var}` プレースホルダを保持する（静的注入なし）。旧 shape
    （`Slot.segments` 空）の slot はそのまま維持する。`OptimizeResult` の shape（単一 str /
    複数 dict）も維持する。

    Args:
        result: `run_apo` の返却値（tune-only テキストを含む）。
        slots: 正規化済みスロット mapping。

    Returns:
        新 shape slot を full 再合成した `OptimizeResult`（frozen のため `replace` で再構築）。
    """
    for name, slot in slots.items():
        if not slot.segments:
            continue  # 旧 shape は run_apo 返却をそのまま使う。
        vars_dict = dict(slot.vars)
        if isinstance(result.prompt, dict):
            seed_map = result.seed if isinstance(result.seed, dict) else {}
            full_prompt = compose_from_marked(slot.segments, result.prompt[name], vars_dict)
            full_seed = compose_from_marked(slot.segments, seed_map.get(name, ""), vars_dict)
            if full_prompt is None or full_seed is None:
                # SSoT 契約違反（予約接頭辞が成果物に漏出しうる）を silent 化しない。上流
                # `_adapters/lightning::_run_apo_single_slot` の post-fit fallback が seed に倒す
                # のが正常経路で、ここに到達するのは fallback すり抜けを意味する。
                warnings.warn(
                    f"[_recompose_new_shape_results] slot {name!r}: 境界マーカー再合成に失敗"
                    "（compose_from_marked が None を返却）。run_apo の返却をそのまま維持しますが、"
                    "OptimizeResult に literal `${oas_boundary_N}` が漏出する可能性があります。"
                    "_adapters の post-fit fallback を確認してください",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            new_prompt = dict(result.prompt)
            new_prompt[name] = full_prompt
            new_seed = dict(seed_map)
            new_seed[name] = full_seed
            new_diff = dict(result.diff) if isinstance(result.diff, dict) else {}
            new_diff[name] = unified_diff_labeled(full_seed, full_prompt)
            result = dataclasses.replace(result, prompt=new_prompt, seed=new_seed, diff=new_diff)
        else:
            # 単一 slot 経路。run_apo の shape 契約により seed も str のはず・非 str は契約違反。
            if not isinstance(result.seed, str):
                raise OptimizeError(
                    FailureKind.TRAINER_FAILED,
                    f"run_apo 返却の shape 契約違反（slot {name!r}: prompt は str だが seed が "
                    f"{type(result.seed).__name__}）。upstream の返却実装を確認してください",
                )
            full_prompt = compose_from_marked(slot.segments, result.prompt, vars_dict)
            full_seed = compose_from_marked(slot.segments, result.seed, vars_dict)
            if full_prompt is None or full_seed is None:
                warnings.warn(
                    f"[_recompose_new_shape_results] slot {name!r}: 境界マーカー再合成に失敗。"
                    "run_apo 返却をそのまま維持しますが literal marker 漏出可能性あり",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            result = dataclasses.replace(
                result,
                prompt=full_prompt,
                seed=full_seed,
                diff=unified_diff_labeled(full_seed, full_prompt),
            )
    return result


def _resolve_config(config: OptimizeConfig | None, direct_kwargs: dict[str, Any]) -> OptimizeConfig:
    """`config=` と直接 kwargs（`apo_client=` 等）の併用を解決し最終 `OptimizeConfig` を返す。

    どちらか片方のみ指定可（同時指定は CONFIG_MISSING で曖昧禁止）。両方未指定なら全フィールドが
    None の `OptimizeConfig`（実行直前の `apo_client` 必須チェックで CONFIG_MISSING になる）。

    Args:
        config: パワーユーザー経路の `OptimizeConfig`（None で未指定）。
        direct_kwargs: 直接渡しの kwargs（`apo_client` / `rounds` 等・None は未指定）。

    Returns:
        最終 `OptimizeConfig`（直接 kwargs の経路では新規構築・config 経路ではそのまま返す）。

    Raises:
        OptimizeError: 直接 kwargs と config を同時指定した場合（`FailureKind.CONFIG_MISSING`）。
    """
    from .config import OptimizeConfig

    has_direct = any(direct_kwargs[k] is not None for k in _DIRECT_CONFIG_KEYS)
    if config is not None and has_direct:
        raise OptimizeError(
            FailureKind.CONFIG_MISSING,
            "config= と直接 kwargs（apo_client= 等）を同時に指定することはできません。"
            "どちらか一方のみを使ってください（最小ケースは直接 kwargs を推奨）",
        )
    if config is not None:
        return config
    # 利用者が渡さなかった直接 kwargs（None）は OptimizeConfig 既定値を尊重するため除外する。
    # `dataclasses.replace(..., field=None)` は既定値を None で上書きしてしまうため、明示値のみ
    # 渡す（例: `apo_gradient_model` の既定 "gpt-5.4-mini" を None で潰さない）。
    overrides = {k: v for k, v in direct_kwargs.items() if v is not None}
    return dataclasses.replace(OptimizeConfig(), **overrides)
