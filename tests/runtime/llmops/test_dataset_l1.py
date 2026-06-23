"""L1: `EvalCase` / `stable_id` / EvalCase↔item マッピングの純ロジック検証（外部依存なし）。"""

from __future__ import annotations

import pytest

from oai_agentspec.runtime.llmops import EvalCase
from oai_agentspec.runtime.llmops.dataset import _case_to_item, _item_to_case, stable_id

pytestmark = pytest.mark.unit


def test_stable_id_prefers_explicit_id() -> None:
    """id 指定時はそれをそのまま返す（index に依存しない）。"""
    case = EvalCase(input="q", id="my-key")
    assert stable_id(case, 3) == "my-key"


def test_stable_id_derives_from_index_and_hash_when_missing() -> None:
    """id 未指定時は `case-{index}-{hash8}` を導出する。"""
    case = EvalCase(input="hello")
    sid = stable_id(case, 0)
    assert sid.startswith("case-0-")
    # hash8 は 8 桁の hex。
    suffix = sid.rsplit("-", 1)[1]
    assert len(suffix) == 8
    int(suffix, 16)  # hex parse できる（失敗すれば例外で test fail）


def test_stable_id_is_deterministic_for_same_input() -> None:
    """同一 input + index で導出キーは決定的。"""
    a = stable_id(EvalCase(input="same"), 1)
    b = stable_id(EvalCase(input="same"), 1)
    assert a == b


def test_stable_id_differs_for_different_input() -> None:
    """input が異なれば導出キーのハッシュ部が異なる。"""
    a = stable_id(EvalCase(input="x"), 0)
    b = stable_id(EvalCase(input="y"), 0)
    assert a != b


def test_eval_case_defaults() -> None:
    """EvalCase の任意フィールドは既定 None。"""
    case = EvalCase(input="q")
    assert case.id is None
    assert case.reference_context is None
    assert case.expected_route is None
    assert case.expected_tools is None
    assert case.expected_approvals is None
    assert case.expected_output is None


def test_eval_case_has_no_criteria_field() -> None:
    """criteria フィールドは観点オブジェクト化に伴い廃止された（criteria= 引数へ移動）。"""
    assert not hasattr(EvalCase(input="q"), "criteria")


def test_eval_case_expected_output_provided() -> None:
    """expected_output（正解文）を任意で保持できる。"""
    case = EvalCase(input="q", expected_output="正解です")
    assert case.expected_output == "正解です"


# ----------------------------------------------------------------------
# EvalCase ↔ Langfuse dataset item（plain dict）マッピング
# ----------------------------------------------------------------------


def test_case_to_item_maps_fields_and_metadata() -> None:
    """oai-agentspec 固有フィールドは item.metadata に格納し
    input/expected_output/id を直接写す。"""
    case = EvalCase(
        input="q",
        id="c1",
        reference_context=["r"],
        expected_route=["a"],
        expected_tools=["t"],
        expected_approvals=["danger"],
        expected_output="o",
    )
    item = _case_to_item(case, 0)
    assert item["id"] == "c1"
    assert item["input"] == "q"
    assert item["expected_output"] == "o"
    assert item["metadata"] == {
        "reference_context": ["r"],
        "expected_route": ["a"],
        "expected_tools": ["t"],
        "expected_approvals": ["danger"],
    }


def test_case_item_roundtrip_expected_approvals() -> None:
    """expected_approvals は metadata 経由で EvalCase へ往復する。"""
    original = EvalCase(input="q", id="c1", expected_approvals=["danger", "wire"])
    restored = _item_to_case(_case_to_item(original, 0))
    assert restored.expected_approvals == ["danger", "wire"]
    assert restored == original


def test_case_to_item_omits_none_metadata() -> None:
    """oai-agentspec 固有フィールドが全て None なら metadata は None（空キーを残さない）。"""
    item = _case_to_item(EvalCase(input="x"), 2)
    assert item["metadata"] is None
    assert item["id"].startswith("case-2-")  # id 未指定なら stable_id 導出


def test_item_to_case_restores_from_metadata() -> None:
    """item の plain dict を EvalCase へ復元する（metadata から固有フィールド）。"""
    item = {
        "id": "c1",
        "input": "q",
        "expected_output": "o",
        "metadata": {"reference_context": ["r"], "expected_tools": ["t"]},
    }
    case = _item_to_case(item)
    assert case == EvalCase(
        input="q", id="c1", reference_context=["r"], expected_tools=["t"], expected_output="o"
    )


def test_item_to_case_handles_missing_metadata() -> None:
    """metadata 非在（None）でも復元でき固有フィールドは None。"""
    case = _item_to_case({"id": "c2", "input": "q2", "expected_output": None, "metadata": None})
    assert case == EvalCase(input="q2", id="c2")


def test_case_item_roundtrip() -> None:
    """EvalCase → item → EvalCase で同値に戻る（id 明示時）。"""
    original = EvalCase(
        input="q", id="c1", reference_context=["r"], expected_route=["a"], expected_output="o"
    )
    assert _item_to_case(_case_to_item(original, 0)) == original
