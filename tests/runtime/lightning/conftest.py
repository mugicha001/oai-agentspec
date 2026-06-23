"""Agent Lightning 最適化テスト用の共通フィクスチャ・ヘルパ（FakeModel ベース spec ファクトリ）。

最適化対象 Agent は FakeModel（実 LLM を呼ばない）。`optimize` は関数内で
`from ..._adapters import run_apo` するため、monkeypatch 対象は使用箇所パス
`oai_agentspec._adapters.run_apo` とする（test_optimizer_l2 で適用）。Judge 採点は
`_adapters.judge_score` 経由（test_rewards_l1 でモック差し替え）。
"""

from __future__ import annotations

from typing import Any

import pytest

from oai_agentspec import AgentSpec

from _helpers.fake_model import FakeModel


def make_spec(
    name: str = "bot",
    *,
    instructions: str = "be helpful",
    tools: list[Any] | None = None,
    model: Any = None,
    output_text: str = "hello world",
) -> AgentSpec:
    """FakeModel を据えた `AgentSpec` を作る（既定はテキスト応答 1 件）。

    Args:
        name: エージェント名。
        instructions: システムプロンプト（静的 str）。
        tools: ツール一覧。
        model: 明示モデル（省略時は output_text を返す FakeModel）。
        output_text: FakeModel が返す既定の応答テキスト。

    Returns:
        構築済み `AgentSpec`。
    """
    fake = model if model is not None else FakeModel().queue_text(output_text)
    return AgentSpec(name=name, instructions=instructions, model=fake, tools=list(tools or []))


@pytest.fixture
def spec_factory() -> Any:
    """`make_spec` を返すフィクスチャ（ミュータブルな FakeModel をテストごとに新規生成する）。"""
    return make_spec
