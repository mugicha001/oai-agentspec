"""失敗種別を判別して扱う例（`OptimizeError` / `FailureKind`・オフラインで動く）。

最適化の失敗は未捕捉例外でプロセスを止めず、`OptimizeError` に統一して送出される。`error.kind`
（`FailureKind`）で失敗の種別を機械的に判別できる:

    FailureKind.EXTRA_MISSING   [lightning] extra（agentlightning）未導入。
    FailureKind.CONFIG_MISSING  必須設定（algorithm / train / reward / val / apo_client / slot /
                                 rebind / registry）不在 / 直接 kwargs と config の二重指定 /
                                 pre-flight route coverage の未到達 slot 検出。
    FailureKind.TRAINER_FAILED  最適化実行（Trainer / rollout / reward）中の失敗。pre-flight
                                 観測中の失敗（timeout 等）もここに入る。

本例は設定不在（CONFIG_MISSING）の代表ケースを意図的に起こして種別を表示する。これらは rollout
到達前に送出されるため、LLM もネットワークも不要でそのまま実行できる:
    uv run python examples/lightning/05_failure_handling.py

最後の 1 ケースだけは TRAINER_FAILED 側で、graph target の **pre-flight route coverage の
タイムアウト**を扱う。`optimize(target=HandoffGraph, slot={...})` は APO へ委譲する前に seed 状態で
`train` 全件を 1 巡 rollout して未到達 slot を検出するが、この観測は `timeout_seconds` を
**1 case あたりの上限**として `asyncio.wait_for` で守られており、超過は `TimeoutError` を
チェーンした `TRAINER_FAILED` になる（`CONFIG_MISSING` ではない点に注意。設定は正しく、実行が
時間内に終わらなかったという区別）。

`timeout_seconds` は **APO の `rollout_batch_timeout` と pre-flight の 1 case 上限の両方**に効き、
`None`（未指定）の意味が適用先で異なる — APO 側は APO 既定（3600 秒）、pre-flight 側は上限なし。
pre-flight の時間上限保護が要るなら明示設定する。

このケースは応答しない疑似モデル（`_HangingModel`）で rollout を止めるため実 LLM もネットワークも
不要だが、pre-flight は extra 可用性を先に確定するので **agentlightning の import が走る**
（初回は数秒かかる）。

EXTRA_MISSING は agentlightning 未導入時に、同じ `except OptimizeError` で `kind` を見るだけで
分岐できる。

補足: 複数 slot APO の**実行段**（pre-flight 通過後）の途中失敗では、`exc.partial`
（`OptimizePartial`）に完了済み slot の最良テキストと履歴が保全される（`failed_slot=None` は
「全 slot 完了・スコア再計算段の失敗」）。この経路は実 APO の起動を要するためオフライン例では
実演しない（詳細は `docs/usage/ops/lightning.md` の「APO 途中失敗時の部分成果」を参照）。

導入: pip install 'oai-agentspec[lightning]'
"""

from __future__ import annotations

import asyncio
import dataclasses

from agents import set_tracing_disabled
from agents.models.interface import Model

from oai_agentspec import AgentRegistry, AgentSpec, HandoffGraph
from oai_agentspec.runtime.lightning import (
    FailureKind,
    OptimizeCase,
    OptimizeError,
    Slot,
    contains,
    optimize,
)


class _HangingModel(Model):
    """`get_response` が決して返らない疑似モデル（pre-flight のタイムアウト発火用）。

    実 LLM もネットワークも使わずに「観測が時間内に終わらない」状況だけを再現する。
    """

    async def get_response(self, *args: object, **kwargs: object) -> object:
        """呼ばれたら待ち続ける（`asyncio.wait_for` にキャンセルされる）。"""
        await asyncio.sleep(3600)
        raise AssertionError("到達しない")

    def stream_response(self, *args: object, **kwargs: object) -> object:
        """本例では未使用。"""
        raise NotImplementedError


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


async def _run_preflight_timeout(
    train_case: list[OptimizeCase], val_case: list[OptimizeCase]
) -> None:
    """graph target の pre-flight 観測をタイムアウトさせ TRAINER_FAILED を表示する。"""
    # ケース 6 は実 rollout（`Runner.run`）を回すため SDK のトレース送信が発火する。
    # 本例は「LLM もネットワークも不要」を契約にしているので明示的に無効化する
    # （他の lightning example は `azure_model()` 内で無効化済み。本例は疑似モデルを使い
    # `azure_model()` を通らないため、ここで自前に無効化する必要がある）。
    set_tracing_disabled(True)
    model = _HangingModel()
    registry = AgentRegistry()
    for name in ("triage", "billing"):
        registry.register(AgentSpec(name=name, instructions="(seed)", model=model))
    graph = HandoffGraph(entry="triage")
    graph.edge("triage", "billing", description="請求関連")
    graph.apply(registry)
    registry.validate()

    # 手書き `Slot`（07 と同じミニマル構成）。build は apply 後の登録 spec を複製して
    # instructions だけ候補で差し替える（handoffs / model は複製で保持される）。
    slots = {
        name: Slot(
            name=name,
            seed=f"{name} の seed",
            build=lambda candidate, _n=name: dataclasses.replace(
                registry._specs[_n],  # noqa: SLF001 - example の最小化のため
                instructions=candidate,
            ),
        )
        for name in ("triage", "billing")
    }

    try:
        await optimize(
            graph,
            train=train_case,
            val=val_case,
            reward=contains(),
            slot=slots,
            registry=registry,
            apo_client=object(),  # pre-flight まで到達すればよいので sentinel で足りる。
            # pre-flight の 1 case あたり観測上限
            # （同じ値が APO の rollout_batch_timeout にも効く）。
            timeout_seconds=0.5,
        )
        print("[preflight-timeout] 成功（このデモでは到達しない想定）")
    except OptimizeError as exc:
        recoverable = exc.kind is FailureKind.CONFIG_MISSING
        print(f"[preflight-timeout] kind={exc.kind.value} recoverable={recoverable}")
        print(f"        message: {exc.message.splitlines()[0]}")
        print(f"        cause:   {type(exc.__cause__).__name__}")
        # 観測が途中で終わっても、そこまでの到達観測は coverage から取得できる
        # （complete=False = 部分レポート。missing は未観測を含むため確定扱いしない）。
        if exc.coverage is not None:
            print(f"        coverage.complete: {exc.coverage.complete}")
            print(f"        coverage.covered:  {sorted(exc.coverage.covered)}")
            # 観測完了数は per_case の非 None エントリで数える（None = 候補無効化・観測なし）。
            observed = sum(1 for _, s in exc.coverage.per_case if s is not None)
            print(f"        観測完了 case 数:   {observed}")


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

    # 6) graph target の pre-flight 観測がタイムアウト -> TRAINER_FAILED（設定不備ではない）。
    await _run_preflight_timeout(train_case, val_case)


if __name__ == "__main__":
    asyncio.run(main())
