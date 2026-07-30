"""Resilience 系宣言型の 1 本例: Failsafe（宣言的な例外着地）。

Runner の外側まで伝播した例外を、呼び出し箇所ごとの try/except でなく
`FailsafePolicy` の宣言 1 回 + `failsafe_call` で着地値へ丸める。本例では以下を示す:

- `FailsafePolicy(handlers={例外型: 着地値 or callable or FailsafeHandler})` を 1 回宣言し、
  各呼び出しを `failsafe_call(policy, lambda: Runner.run(...))` で包む。
- 正常完了時は `Runner.run` の戻り値（`RunResult`）がそのまま返り、着地時のみ
  `FailsafeResult` が返る。どちらも `.final_output` で一様にアクセスできる
  （structural 互換。共通基底クラスは持たない）。
- 着地の有無は `isinstance(result, FailsafeResult)` で判別し、`exception` /
  `matched_type` / `last_agent` で原因と実行文脈を参照できる。
- `last_agent`（「もともと実行中だったエージェント」）は 2 段で決まる:
  (1) 例外ごとの指定（`FailsafeHandler.last_agent`）、
  (2) 全体規定（`FailsafePolicy.fallback_last_agent`）。どちらにも具体の agent
  （`AgentRegistry.get(name)` の戻り値をそのまま置ける）か公開 sentinel
  `RUNNING_AGENT` を置け、`RUNNING_AGENT` を置いた段でのみ例外からの解決（opt-in）が
  走る。どちらの段も無指定なら `last_agent` は None のまま。本例では段 1 に
  `RUNNING_AGENT`（実行中の agent を継続）と registry 由来の agent（例外型ごとの
  エスカレーション）の両形、段 2 に registry 由来の agent を宣言している。
- `FailsafeResult.from_exception` を使うと、`failsafe_call` の外側の except でも
  同じ結果型・同じ `last_agent` の意味へ手動で着地させられる（監査は発火しない）。
- 宣言していない例外は着地せずそのまま伝播する（塗りつぶしを防ぐため `Exception`
  そのものは handlers のキーにできず、宣言すると build-time `ValueError`。同様に
  `RUNNING_AGENT` を着地値の位置（`fallback` / handlers の値）に直接置く宣言も
  build-time `ValueError`）。
- 監査は既定 on の `logger.warning`（logger 名 `oai_agentspec.resilience`）と、
  メトリクス連携用の `on_apply` callback の 2 経路。

retry は Failsafe の責務ではなく SDK ネイティブ機構（`ModelRetryPolicy`）に委ねる。
Failsafe は「Runner の外へ漏れた例外を着地させる」ことだけを行う。

Azure OpenAI の環境変数（AZURE_OPENAI_*・examples/_shared/_azure.py 参照）を設定して実行:

    uv run python examples/resilience/02_failsafe.py

本例は実 API を呼ぶ（合計 2 回）。予算超過の実演では `max_total_tokens` を
極小値にして 1 ターン目で確実に上限へ到達させている。
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from agents import Agent, Runner

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.exceptions import RunBudgetExceeded
from oai_agentspec.runtime.resilience import (
    RUNNING_AGENT,
    FailsafeHandler,
    FailsafePolicy,
    FailsafeResult,
    RunBudgetPolicy,
    build_run_budget_hooks,
    failsafe_call,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402


def _on_failsafe_applied(result: FailsafeResult) -> None:
    """`on_apply` の実演: 実運用ではメトリクス送信・監査ログ連携に使う。

    ここで送出した例外は `logger.error` に記録されたうえで握り潰され、着地結果の
    返却は継続する（監査経路の失敗が本処理を壊さない）。
    """
    print(f"[on_apply] failsafe applied: matched_type={result.matched_type.__name__}")


def _budget_message(exc: BaseException) -> str:
    """捕捉した例外から着地文言を組み立てる fallback（callable 形式）。

    handlers の値は固定値でもよいが、callable にすると例外の属性を使って文脈のある
    文言を返せる（sync / async のどちらでもよい）。
    """
    total = getattr(getattr(exc, "usage", None), "total_tokens", "unknown")
    return f"混雑のため回答を中断しました（累積トークン: {total}）。"


def _build_registry() -> AgentRegistry:
    """着地方針が参照するエージェント群の registry（宣言と着地方針の接続点）。

    `AgentRegistry` は `AgentSpec` から `Agent` を遅延構築するため、着地方針側は
    `registry.get(name)` の戻り値をそのまま `last_agent` に置ける
    （registry の差し替え・遅延構築とそのまま組み合わせられる）。本例では
    `support` を段 1（例外ごとの指定）に、`triage` を段 2（全体規定）に使う。
    """
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="triage",
            instructions="You are a triage agent. Route the user to the right place.",
            model=azure_model(),
        )
    )
    registry.register(
        AgentSpec(
            name="support",
            instructions="You are a support agent. Handle escalated requests.",
            model=azure_model(),
        )
    )
    return registry


def _build_policy(registry: AgentRegistry) -> FailsafePolicy:
    """アプリ全体で 1 回だけ宣言する着地方針。

    宣言順に first-match するため、より specific な型を先に宣言する責務は利用者側にある。
    `Exception` / `BaseException` / `KeyboardInterrupt` / `SystemExit` /
    `asyncio.CancelledError` / `GeneratorExit` / `ExceptionGroup` はキーにできない
    （build-time `ValueError`）。

    Args:
        registry: `last_agent` に置くエージェントの取得元（段 1 の `support`・
            段 2 の `triage`）。
    """
    return FailsafePolicy(
        handlers={
            # 予算超過: 例外の属性を使って文言を組み立てる（callable fallback）。
            # last_agent は段 1（例外ごとの指定）に RUNNING_AGENT を置き、
            # RunBudgetExceeded.last_agent から実行中だった agent を解決する。
            RunBudgetExceeded: FailsafeHandler(fallback=_budget_message, last_agent=RUNNING_AGENT),
            # タイムアウト: 段 1 に registry 由来の agent を置き、この例外型だけ別の
            # エージェント（support）へエスカレーションする。RUNNING_AGENT と違い
            # 「実行中だった agent」ではなく宣言した固定の agent が入る。
            TimeoutError: FailsafeHandler(
                fallback="時間内に応答できませんでした。担当者へ引き継ぎます。",
                last_agent=registry.get("support"),
            ),
            # 想定外の値エラー: 固定文言へ丸める（非 callable fallback・従来形のまま）。
            # last_agent は例外ごとの指定を持たないため、段 2（fallback_last_agent）へ落ちる。
            ValueError: "入力を処理できませんでした。",
        },
        # log_on_apply=True が既定。ログには例外メッセージとトレースバックが
        # そのまま出るため、機密を含みうる例外を扱う場合は False にして
        # on_apply 側でマスキングする。
        log_on_apply=True,
        on_apply=_on_failsafe_applied,
        # 全体規定（段 2）: 例外ごとの指定が無い / 解決できない場合の落とし先。
        # AgentRegistry から取得した Agent をそのまま置ける（RUNNING_AGENT を置いて
        # 「実行中だった agent を使う」ことも可能で、その場合は解決できなければ None）。
        fallback_last_agent=registry.get("triage"),
    )


def _show(label: str, result: object) -> None:
    """戻り値を `.final_output` で一様に扱いつつ、着地の有無を判別して表示する。"""
    final_output = result.final_output  # type: ignore[attr-defined]
    if isinstance(result, FailsafeResult):
        print(f"[{label}] landed  : {final_output}")
        print(f"[{label}]   cause : {type(result.exception).__name__}: {result.exception}")
        print(f"[{label}]   matched: {result.matched_type.__name__}")
        # Agent 実体（result.last_agent）をそのまま print しない: システムプロンプト等の
        # 機微情報が出力に載りうるため、getattr で name のみを安全に取り出す
        # （last_agent が None、または name を持たない不透明値でも例外にならない）。
        print(f"[{label}]   last_agent: {getattr(result.last_agent, 'name', None)}")
    else:
        print(f"[{label}] normal  : {final_output}")


async def _run_normal_completion(policy: FailsafePolicy) -> None:
    """正常完了: `Runner.run` の戻り値がそのまま返る（`FailsafeResult` でラップしない）。"""
    agent = Agent(
        name="assistant",
        instructions="You are a helpful assistant. Reply in one sentence.",
        model=azure_model(),
    )

    result = await failsafe_call(policy, lambda: Runner.run(agent, "What is 2 + 2?"))
    _show("normal", result)


async def _run_landing_on_budget_exceeded(policy: FailsafePolicy) -> None:
    """着地: 予算超過で `RunBudgetExceeded` が Runner の外へ伝播し、着地値へ丸まる。

    `max_total_tokens=1` により 1 ターン目の `on_llm_end` 境界で確実に上限へ到達する。
    従来はこの箇所ごとに try/except を書く必要があったが、policy の宣言 1 回で足りる。

    予算判定は `on_llm_end`（LLM 呼び出しの完了時）に行うため、上限を極小にしても
    1 回目の生成そのものは最後まで走る。実演を短時間で終わらせるため、出力が短くなる
    プロンプトと instructions にしている。
    """
    agent = Agent(
        name="assistant",
        instructions="Reply with a single short sentence.",
        model=azure_model(),
    )
    hooks = build_run_budget_hooks(RunBudgetPolicy(max_total_tokens=1))

    result = await failsafe_call(
        policy,
        lambda: Runner.run(agent, "Say hello.", hooks=hooks),
    )
    _show("budget", result)


async def _run_passthrough_for_undeclared(policy: FailsafePolicy) -> None:
    """素通し: 宣言していない例外型は着地せず、そのまま呼び出し元へ伝播する。

    着地させたい失敗だけを宣言する設計のため、宣言漏れの例外が「正常値」へ
    すり替わることはない（例外チェーンの差し替えも行わない）。捕捉した側は
    `FailsafeResult.from_exception` を使えば、`failsafe_call` の着地結果と同じ結果型・
    同じ `last_agent` の意味へ手動で着地させられる（監査（warning / on_apply）は
    発火しない点が `failsafe_call` の着地との違い）。
    API 呼び出しは不要なため、thunk 内で直接送出して示す。
    """

    async def _raises_undeclared() -> str:
        raise KeyError("undeclared-key")

    try:
        await failsafe_call(policy, _raises_undeclared)
    except KeyError as exc:
        print(f"[passthrough] propagated as-is: {type(exc).__name__}: {exc}")
        manual = FailsafeResult.from_exception(
            exc, final_output="不明なキーのため処理を中断しました。"
        )
        _show("manual", manual)


async def _run_per_exception_registry_agent(policy: FailsafePolicy) -> None:
    """段 1（例外ごとの指定）に registry 由来の agent を置く: 例外型ごとの倒し先を変える。

    `TimeoutError` の宣言は `FailsafeHandler(last_agent=registry.get("support"))` を持つため、
    実行中だった agent ではなく宣言した support agent が `last_agent` に入る
    （`RUNNING_AGENT` を置いた `RunBudgetExceeded` との違い）。エスカレーション先を
    例外の種類ごとに固定したい場合の形。API 呼び出しは不要なため thunk 内で直接送出する。
    """

    async def _raises_timeout() -> str:
        raise TimeoutError("upstream-timeout")

    result = await failsafe_call(policy, _raises_timeout)
    _show("escalation", result)


async def _run_fallback_stage(policy: FailsafePolicy) -> None:
    """段 2（全体規定）への降格: 例外ごとの指定が無い宣言は registry 由来の既定へ落ちる。

    `ValueError` の宣言は素の着地値のみで `last_agent` の指定を持たないため、決定は段 2
    （`fallback_last_agent`）へ落ち、`AgentRegistry` から取得した agent が入る。
    API 呼び出しは不要なため、thunk 内で直接送出して示す。
    """

    async def _raises_value_error() -> str:
        raise ValueError("bad-input")

    result = await failsafe_call(policy, _raises_value_error)
    _show("fallback-stage", result)


async def _run_forbidden_declaration() -> None:
    """build-time 拒否: 捕捉範囲が広すぎる宣言・sentinel の誤配置は構築時点で `ValueError`。

    `Exception` を宣言できるとバグ由来の例外まで正常値へ丸めてしまうため、
    プロセス制御例外（`KeyboardInterrupt` 等）とあわせて禁止している。また
    `RUNNING_AGENT` は `last_agent` の指定値であり着地値ではないため、`fallback` の
    位置（`FailsafeHandler(fallback=...)` や handlers の値）に直接置く宣言も拒否する。
    """
    try:
        FailsafePolicy(handlers={Exception: "swallow everything"})
    except ValueError as exc:
        print(f"[forbidden] rejected at build time: {exc}")

    try:
        FailsafeHandler(fallback=RUNNING_AGENT)
    except ValueError as exc:
        print(f"[forbidden] rejected at build time: {exc}")


async def main() -> None:
    """6 パターンを順に実行する。"""
    # 監査ログ（既定 on の warning）を標準エラーへ表示する
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    # 着地方針は registry と組み合わせて 1 回だけ宣言する（段 2 の落とし先を registry
    # から取得する形。registry 側の差し替え・遅延構築はそのまま効く）。
    policy = _build_policy(_build_registry())

    await _run_normal_completion(policy)
    await _run_landing_on_budget_exceeded(policy)
    await _run_per_exception_registry_agent(policy)
    await _run_fallback_stage(policy)
    await _run_passthrough_for_undeclared(policy)
    await _run_forbidden_declaration()


if __name__ == "__main__":
    asyncio.run(main())
