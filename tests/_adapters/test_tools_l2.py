"""L2: _adapters.tools.build_function_tool の SDK 結線特性化テスト（Issue #27 Task 2・RED 先行）。

`ToolSpec` の宣言メタデータが SDK `agents.function_tool()` を経由して `FunctionTool`
の各属性へ正しく結線されること・extra 検証エラーメッセージ原文（`subject_label="tool"` /
`agent_label="agents.function_tool"`）・failure_error_function 3 値（未指定 = _UNSET /
callable / None 明示）・enabled の closure 動的解決（FR-4・再構築なし）を SDK 実型で検証する。

実装未完のため（`_adapters/tools.py` および `_adapters/__init__` 再エクスポートが未追加）、
本モジュールの import は `ImportError` となる（collection error = RED 状態が正しい）。
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import MagicMock

import pytest
from agents import FunctionTool

from oai_agentspec import ToolSpec
from oai_agentspec._adapters import build_function_tool

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# ヘルパ
# ---------------------------------------------------------------------------
async def _sample_fn(query: str) -> str:
    """テスト用の素朴な async tool 関数。"""
    return f"answer:{query}"


async def _other_fn(x: int) -> int:
    """別名テスト用の第二関数。"""
    return x + 1


def _resolve_is_enabled(tool: FunctionTool, ctx: object, agent: object) -> bool:
    """SDK `is_enabled` が MaybeAwaitable[bool] のため sync/async 双方に対応して bool を得る。"""
    if isinstance(tool.is_enabled, bool):
        return tool.is_enabled
    result = tool.is_enabled(ctx, agent)
    if inspect.iscoroutine(result):
        return asyncio.get_event_loop().run_until_complete(result)
    return bool(result)


# ---------------------------------------------------------------------------
# 正常系: 最小 ToolSpec で FunctionTool が返る
# ---------------------------------------------------------------------------
def test_正常系_最小ToolSpecでFunctionToolが返る() -> None:
    """最小構成の `ToolSpec` から SDK 実型の `FunctionTool` が組み立てられる。"""
    spec = ToolSpec(func=_sample_fn)
    tool = build_function_tool(spec, lambda: True)
    assert isinstance(tool, FunctionTool)


# ---------------------------------------------------------------------------
# 正常系: name / name_override / description_override の反映
# ---------------------------------------------------------------------------
def test_正常系_toolspec_nameがname_overrideとしてSDKに反映される() -> None:
    """`ToolSpec.name`（Registry 登録キー）が SDK `name_override` として tool.name に反映される。"""
    spec = ToolSpec(func=_sample_fn, name="alias")
    tool = build_function_tool(spec, lambda: True)
    assert tool.name == "alias"


def test_正常系_name_overrideとdescription_overrideがSDKに反映() -> None:
    """`name_override` / `description_override` の明示指定が SDK 側属性に反映される。"""
    spec = ToolSpec(
        func=_sample_fn,
        name="reg_key",
        name_override="ov",
        description_override="d",
    )
    tool = build_function_tool(spec, lambda: True)
    assert tool.name == "ov"
    assert tool.description == "d"


# ---------------------------------------------------------------------------
# 正常系: enabled は closure で動的に bool() を返す（FR-4・再構築なし）
# ---------------------------------------------------------------------------
def test_正常系_enabled_supplierがis_enabledとして焼き込まれず動的にbool_を返す() -> None:
    """`enabled_supplier` が SDK `is_enabled` に closure として渡り、後から値変更が即反映される。"""
    spec = ToolSpec(func=_sample_fn, enabled=True)
    tool = build_function_tool(spec, lambda: spec.enabled)
    ctx = MagicMock()
    agent = MagicMock()
    assert _resolve_is_enabled(tool, ctx, agent) is True
    spec.enabled = False
    assert _resolve_is_enabled(tool, ctx, agent) is False


# ---------------------------------------------------------------------------
# 正常系: timeout の None-omission と明示指定
# ---------------------------------------------------------------------------
def test_正常系_timeout_None_omissionで未指定() -> None:
    """`timeout=None` は kwarg を渡さず SDK 既定（None）に委ねる。明示指定は反映される。"""
    tool_default = build_function_tool(ToolSpec(func=_sample_fn, timeout=None), lambda: True)
    assert tool_default.timeout_seconds is None
    tool_explicit = build_function_tool(ToolSpec(func=_sample_fn, timeout=10.0), lambda: True)
    assert tool_explicit.timeout_seconds == 10.0


def test_正常系_timeout_behavior_None_omissionで未指定() -> None:
    """`timeout_behavior=None` は kwarg を渡さず SDK 既定（"error_as_result"）に委ねる。

    明示指定（`"raise_exception"`）が FunctionTool.timeout_behavior に反映される。
    """
    tool_default = build_function_tool(
        ToolSpec(func=_sample_fn, timeout_behavior=None), lambda: True
    )
    assert tool_default.timeout_behavior == "error_as_result"
    tool_explicit = build_function_tool(
        ToolSpec(func=_sample_fn, timeout_behavior="raise_exception"), lambda: True
    )
    assert tool_explicit.timeout_behavior == "raise_exception"


def test_正常系_timeout_error_function_None_omissionで未指定() -> None:
    """`timeout_error_function=None` は kwarg を渡さず SDK 既定（None）に委ねる。

    明示指定した callable が FunctionTool.timeout_error_function に格納される。
    """

    def my_timeout_fmt(ctx: object, err: Exception) -> str:
        return f"custom timeout: {err}"

    tool_default = build_function_tool(
        ToolSpec(func=_sample_fn, timeout_error_function=None), lambda: True
    )
    assert tool_default.timeout_error_function is None
    tool_explicit = build_function_tool(
        ToolSpec(func=_sample_fn, timeout_error_function=my_timeout_fmt), lambda: True
    )
    assert tool_explicit.timeout_error_function is my_timeout_fmt


# ---------------------------------------------------------------------------
# 正常系: failure_error_function 3 値（_UNSET / callable / None 明示）
# ---------------------------------------------------------------------------
def test_正常系_failure_error_function_3値_未指定はSDK既定に委ねる() -> None:
    """`failure_error_function` 未指定（_UNSET）時は SDK 既定 formatter が使われる。"""
    spec = ToolSpec(func=_sample_fn)
    tool = build_function_tool(spec, lambda: True)
    # SDK は _UNSET 相当なら _use_default_failure_error_function=True を立てる。
    assert tool._use_default_failure_error_function is True


def test_正常系_failure_error_function_3値_関数指定はそのまま渡る() -> None:
    """明示 callable が SDK `_failure_error_function` にそのまま格納される。"""

    def my_fmt(ctx: object, err: Exception) -> str:
        return f"failed: {err}"

    spec = ToolSpec(func=_sample_fn, failure_error_function=my_fmt)
    tool = build_function_tool(spec, lambda: True)
    assert tool._failure_error_function is my_fmt
    assert tool._use_default_failure_error_function is False


def test_正常系_failure_error_function_3値_None明示は生例外を通す() -> None:
    """`failure_error_function=None` 明示は SDK 側で「既定 formatter を使わない」設定になる。"""
    spec = ToolSpec(func=_sample_fn, failure_error_function=None)
    tool = build_function_tool(spec, lambda: True)
    assert tool._failure_error_function is None
    assert tool._use_default_failure_error_function is False


# ---------------------------------------------------------------------------
# 異常系: extra の予約キー衝突・未知キー（subject_label="tool" プレフィクス）
# ---------------------------------------------------------------------------
def test_異常系_extra_予約キー衝突でValueError() -> None:
    """extra に専用フィールド同名キー（is_enabled）を積むと tool prefix の ValueError で弾く。"""
    spec = ToolSpec(func=_sample_fn, name="mytool", extra={"is_enabled": True})
    with pytest.raises(
        ValueError,
        match=r"tool 'mytool': extra に専用フィールドと同名のキーが含まれます",
    ):
        build_function_tool(spec, lambda: True)


def test_異常系_extra_未知キーでValueError() -> None:
    """extra に SDK function_tool が受け付けない未知キーを積むと未知メッセージで弾く。"""
    spec = ToolSpec(func=_sample_fn, name="mytool", extra={"unknown_kw": 1})
    with pytest.raises(
        ValueError,
        match=r"tool 'mytool': extra に agents\.function_tool が受け付けないキーが含まれます",
    ):
        build_function_tool(spec, lambda: True)


# ---------------------------------------------------------------------------
# 正常系: extra 素通し（defer_loading）
# ---------------------------------------------------------------------------
def test_正常系_extra_有効な素通しキー_defer_loadingが渡る() -> None:
    """extra 経由で SDK 有効 kwarg（defer_loading）が FunctionTool 属性に反映される。"""
    spec = ToolSpec(func=_sample_fn, extra={"defer_loading": True})
    tool = build_function_tool(spec, lambda: True)
    assert tool.defer_loading is True


# ---------------------------------------------------------------------------
# 正常系: needs_approval / strict_mode の None-omission と明示指定
# ---------------------------------------------------------------------------
def test_正常系_needs_approval_None_omission() -> None:
    """`needs_approval=None` は kwarg を渡さず SDK 既定 False、明示 True は反映される。"""
    tool_default = build_function_tool(ToolSpec(func=_sample_fn), lambda: True)
    assert tool_default.needs_approval is False
    tool_true = build_function_tool(ToolSpec(func=_sample_fn, needs_approval=True), lambda: True)
    assert tool_true.needs_approval is True


def test_正常系_strict_mode_None_omission() -> None:
    """`strict_mode=None` は kwarg を渡さず SDK 既定 True、明示 False は反映される。"""
    tool_default = build_function_tool(ToolSpec(func=_sample_fn), lambda: True)
    assert tool_default.strict_json_schema is True
    tool_false = build_function_tool(ToolSpec(func=_other_fn, strict_mode=False), lambda: True)
    assert tool_false.strict_json_schema is False
