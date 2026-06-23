"""失敗種別を判別して扱う例（`OptimizeError` / `FailureKind`・オフラインで動く）。

最適化の失敗は未捕捉例外でプロセスを止めず、`OptimizeError` に統一して送出される。`error.kind`
（`FailureKind`）で失敗の種別を機械的に判別できる:

    FailureKind.EXTRA_MISSING   [lightning] extra（agentlightning）未導入。
    FailureKind.CONFIG_MISSING  必須設定（algorithm / train / reward / val / apo_client / slot /
                                 rebind / registry）不在 / 直接 kwargs と config の二重指定。
    FailureKind.TRAINER_FAILED  最適化実行（Trainer / rollout / reward）中の失敗。

本例は設定不在（CONFIG_MISSING）の代表ケースを意図的に起こして種別を表示する。これらは rollout
到達前に送出されるため、LLM もネットワークも不要でそのまま実行できる:
    uv run python examples/lightning/05_failure_handling.py

EXTRA_MISSING は agentlightning 未導入時、TRAINER_FAILED は Trainer / rollout / reward が実行中に
例外を送出したときに、同じ `except OptimizeError` で `kind` を見るだけで分岐できる。

導入: pip install 'oai-agentspec[lightning]'
"""

from __future__ import annotations

import asyncio

from oai_agentspec import AgentSpec
from oai_agentspec.runtime.lightning import (
    FailureKind,
    OptimizeCase,
    OptimizeError,
    contains,
    optimize,
)


async def _run(label: str, **kwargs: object) -> None:
    """1 ケースを実行し、`OptimizeError` の種別とメッセージを表示する。"""
    target = AgentSpec(name="demo", instructions="ユーザーの依頼を1語で分類する。", model=None)
    try:
        await optimize(target, **kwargs)  # type: ignore[arg-type]
        print(f"[{label}] 成功（このデモでは到達しない想定）")
    except OptimizeError as exc:
        # kind で分岐できる（ここでは表示のみ）。
        recoverable = exc.kind is FailureKind.CONFIG_MISSING
        print(f"[{label}] kind={exc.kind.value} recoverable={recoverable}")
        print(f"        message: {exc.message[:80]}")


async def main() -> None:
    train_case = [OptimizeCase(input="請求の件", expected_output="billing")]
    val_case = [OptimizeCase(input="検証", expected_output="billing")]
    # apo_client の sentinel（オフラインデモのため本物の AsyncOpenAI でなくてよい・
    # config_missing 経路は client が None かのみ見る）。
    fake_client = object()

    # 1) 未対応 algorithm（RL は別 extra）-> CONFIG_MISSING。
    await _run(
        "rl-unsupported",
        algorithm="rl",
        train=train_case,
        val=val_case,
        reward=contains(),
        apo_client=fake_client,
    )

    # 2) 学習データ空 -> CONFIG_MISSING。
    await _run(
        "empty-train",
        train=[],
        val=val_case,
        reward=contains(),
        apo_client=fake_client,
    )

    # 3) val 未指定 -> CONFIG_MISSING（APO は検証用ケース列必須）。
    await _run(
        "val-missing",
        train=train_case,
        val=None,
        reward=contains(),
        apo_client=fake_client,
    )

    # 4) apo_client 未指定 -> CONFIG_MISSING（APO は AsyncOpenAI 互換クライアント必須）。
    await _run(
        "apo-client-missing",
        train=train_case,
        val=val_case,
        reward=contains(),
        # apo_client / config の両方未指定で CONFIG_MISSING。
    )

    # 5) 生 seed を渡したが rebind 未指定 -> CONFIG_MISSING（prompt_slot を使えば rebind は不要）。
    await _run(
        "raw-seed-without-rebind",
        train=train_case,
        val=val_case,
        reward=contains(),
        slot="あなたは分類器です。",
        apo_client=fake_client,
    )


if __name__ == "__main__":
    asyncio.run(main())
