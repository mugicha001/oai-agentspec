"""L2: `DeterministicResponseModel` が依存する SDK `Model` ABC の構造契約トリップワイヤ。

`DeterministicResponseModel.get_response` / `stream_response` は SDK が `*args, **kwargs`
で吸収した呼び出しを `ModelRequest` へ正規化する際、キーワード優先 + 位置フォールバックで
`model_settings` / `tools` / `output_schema` / `handoffs` の位置番号を固定値（0 / 1 / 2 / 3）
として決め打ちしている（`_adapters/deterministic.py` の `_build_request` / `_positional`）。
`agents.models.interface.Model` の抽象メソッド定義が SDK upgrade で位置引数順・kw-only 集合を
変えると、この決め打ちが静かにずれて誤った値を拾う（例外は出ない）。本モジュールは SDK 実型を
直接検査してその退行を CI で fail させる（本機能のコードは経由しない。合成側の挙動 pin は
`tests/_adapters/test_deterministic_model_l2.py` が担う。`agents` を import するため integration
マーカー、`tests/_adapters/test_next_turn_sdk_contract_l2.py` と同じ扱い）。
"""

from __future__ import annotations

import inspect

import pytest
from agents.models.interface import Model

pytestmark = pytest.mark.integration

# 位置引数の並び（先頭 self を含む）。`_positional(args, 0..3)` が model_settings / tools /
# output_schema / handoffs の順で拾う前提の根拠。
_EXPECTED_POSITIONAL_PARAMS = [
    "self",
    "system_instructions",
    "input",
    "model_settings",
    "tools",
    "output_schema",
    "handoffs",
    "tracing",
]

# kw-only 集合（`get_response` / `stream_response` へ渡っても `ModelRequest` へ載せない引数）。
_EXPECTED_KEYWORD_ONLY_PARAMS = {"previous_response_id", "conversation_id", "prompt"}


def _positional_and_keyword_only(func: object) -> tuple[list[str], set[str]]:
    """関数シグネチャから位置引数順（`self` 含む）と kw-only 集合を取り出す。"""
    signature = inspect.signature(func)
    positional = [
        name
        for name, param in signature.parameters.items()
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    keyword_only = {
        name
        for name, param in signature.parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    }
    return positional, keyword_only


def test_sdk_get_responseの位置引数順とkw_only集合を固定する() -> None:
    """ここがずれると `_positional` の決め打ち番号が誤った引数を拾う。"""
    positional, keyword_only = _positional_and_keyword_only(Model.get_response)

    assert positional == _EXPECTED_POSITIONAL_PARAMS
    assert keyword_only == _EXPECTED_KEYWORD_ONLY_PARAMS


def test_sdk_stream_responseの位置引数順とkw_only集合を固定する() -> None:
    """`get_response` と同じ正規化ロジックを共有するため、位置引数順・kw-only 集合も同一である。"""
    positional, keyword_only = _positional_and_keyword_only(Model.stream_response)

    assert positional == _EXPECTED_POSITIONAL_PARAMS
    assert keyword_only == _EXPECTED_KEYWORD_ONLY_PARAMS


def test_sdk_get_responseはコルーチン関数である() -> None:
    """`get_response` が `async def` でなくなると `await` 前提の呼び出し側実装が壊れる。"""
    assert inspect.iscoroutinefunction(Model.get_response)


def test_sdk_stream_responseは非asyncのdefでAsyncIteratorを返す() -> None:
    """`stream_response` は `async def`（コルーチン関数）ではなく、`AsyncIterator` を返す
    非 async の `def` である。`DeterministicResponseModel.stream_response` は `async def` の
    async generator として実装しており、SDK 側がコルーチン関数へ変わると `await` してから
    iterate する形へ SDK 呼び出し側の期待が変わりうる。
    """
    assert not inspect.iscoroutinefunction(Model.stream_response)
    assert not inspect.isasyncgenfunction(Model.stream_response)

    signature = inspect.signature(Model.stream_response)
    assert signature.return_annotation == "AsyncIterator[TResponseStreamEvent]"
