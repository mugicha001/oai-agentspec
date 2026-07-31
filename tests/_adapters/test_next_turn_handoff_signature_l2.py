"""L2: 記録 `on_handoff` 合成が実 SDK `handoff()` の署名検証を通過することの pin。

到達時ハンドオフ禁止（FR-3）の記録は、利用者宣言の `on_handoff` へ前置合成する形で
SDK の公式拡張点に載せる。SDK は `handoff()` の構築時に `on_handoff` の引数個数を
`inspect.signature` で検証し（`input_type` なし = 1 引数 / あり = 2 引数）、外れると
`UserError` を送出する。合成 callable の arity をエッジの `input_type` 有無へ合わせる
契約が実型に対して成立することを、実 SDK の `handoff()` 構築で固定する。

`agents` を import するため integration マーカー（`tests/_adapters/test_builders_l2.py`
と同じ扱い）。SDK 型に依存しない記録・ゲートの挙動 pin は `test_next_turn_l2.py` が担う。
"""

from __future__ import annotations

from typing import Any

import pytest
from agents import Agent, Handoff, handoff
from pydantic import BaseModel

from oai_agentspec._adapters.next_turn import ArrivalStore, make_arrival_recorder

pytestmark = pytest.mark.integration


class _EscalationInput(BaseModel):
    """`input_type` あり経路の handoff 入力（最小の pydantic モデル）。"""

    reason: str


def test_記録合成はinput_typeなしのhandoff構築を通過する() -> None:
    """`input_type` なしのエッジでは 1 引数として受理され、`Handoff` が構築できる。"""
    recorder = make_arrival_recorder(ArrivalStore(), "billing", None, False)

    built = handoff(Agent(name="billing"), on_handoff=recorder)

    assert isinstance(built, Handoff)


def test_記録合成はinput_typeありのhandoff構築を通過する() -> None:
    """`input_type` ありのエッジでは 2 引数として受理され、`Handoff` が構築できる。"""
    recorder = make_arrival_recorder(ArrivalStore(), "billing", None, True)

    built = handoff(Agent(name="billing"), on_handoff=recorder, input_type=_EscalationInput)

    assert isinstance(built, Handoff)


def test_利用者on_handoffをchainした合成もhandoff構築を通過する() -> None:
    """利用者宣言の `on_handoff` を chain しても arity は保たれ、署名検証を通過する。"""
    store = ArrivalStore()

    def _user_one(ctx: Any) -> None:
        return None

    def _user_two(ctx: Any, payload: _EscalationInput) -> None:
        return None

    without_input = handoff(
        Agent(name="billing"),
        on_handoff=make_arrival_recorder(store, "billing", _user_one, False),
    )
    with_input = handoff(
        Agent(name="billing"),
        on_handoff=make_arrival_recorder(store, "billing", _user_two, True),
        input_type=_EscalationInput,
    )

    assert isinstance(without_input, Handoff)
    assert isinstance(with_input, Handoff)
