"""L1: usage 欠損の共通述語 `_adapters.runner.usage_is_missing` の判定表を pin する（ADR-0046）。

usage が欠損であるとは `input_tokens == 0` かつ `output_tokens == 0` かつ `total_tokens == 0`
であることをいい、`requests` は判定に使わない。resilience の予算フックと intent の usage 詰め替えの
両方がこの述語を使う。
"""

from __future__ import annotations

import pytest
from agents.usage import Usage

from oai_agentspec._adapters.runner import usage_is_missing

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        pytest.param(Usage(requests=1), True, id="requests1_tokens0_builtin_model_missing"),
        pytest.param(Usage(requests=0), True, id="requests0_tokens0"),
        pytest.param(
            Usage(requests=1, input_tokens=30, output_tokens=12, total_tokens=42),
            False,
            id="all_tokens_present",
        ),
        pytest.param(
            Usage(requests=0, input_tokens=0, output_tokens=0, total_tokens=10),
            False,
            id="total_only",
        ),
        pytest.param(
            Usage(requests=0, input_tokens=5, output_tokens=3, total_tokens=0),
            False,
            id="in_out_without_total",
        ),
        pytest.param(
            Usage(requests=0, input_tokens=5, output_tokens=0, total_tokens=0),
            False,
            id="input_only",
        ),
        pytest.param(
            Usage(requests=0, input_tokens=0, output_tokens=5, total_tokens=0),
            False,
            id="output_only",
        ),
        pytest.param(Usage(requests=3), True, id="requests3_tokens0_after_retry"),
    ],
)
def test_トークン3項目がすべて0のときだけ欠損と判定しrequestsは見ない(
    usage: Usage, expected: bool
) -> None:
    """欠損判定はトークン 3 項目（in / out / total）がすべて 0 かどうかだけで決まる。

    - requests が 1 以上でもトークンがすべて 0 なら欠損（組み込みモデル・retry 後の穴）
    - total を埋めず in / out だけを持つ usage は欠損ではない
    - in / out / total のいずれか 1 項目でも非 0 なら欠損ではない（3 条件それぞれを独立に pin する）
    """
    assert usage_is_missing(usage) is expected
