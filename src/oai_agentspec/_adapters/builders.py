"""Agent / handoff / sub-tool 構築アダプタ（SDK 結合を `_adapters` に閉じる・NFR-1）。

`build_agent`（デフォルト AgentBuilder）/ `make_handoff` / `make_dynamic_handoff` /
`DefaultAgentBuilder` / `make_agent_tool` / `mock_spec_tools`（LLMOps の HITL 完了採点で
宣言層の AgentSpec.tools の実行だけをモック差し替え）を提供する。`from agents import ...` の
SDK 結合は本モジュール内に閉じる。
"""

from __future__ import annotations

import inspect
import json
from dataclasses import fields as _dataclass_fields
from dataclasses import replace as _dataclass_replace
from typing import TYPE_CHECKING, Any

from agents import (
    Agent,
    FunctionTool,
    Handoff,
    handoff,
)

from .._validation import validate_extra_kwargs

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ..spec import AgentSpec, HandoffConfig

# 専用フィールド名（AgentSpec 側で別扱いするため extra から除外する Agent kwarg）。
_DEDICATED_AGENT_KWARGS = frozenset(
    {
        "name",
        "instructions",
        "prompt",
        "tools",
        "handoffs",
        "model",
        "model_settings",
        "hooks",
        "input_guardrails",
        "output_guardrails",
    }
)

# handoff() の無入力時スキーマ（SDK の handoff(agent) が生成するものと同一）。
_EMPTY_HANDOFF_SCHEMA: dict[str, Any] = {
    "additionalProperties": False,
    "type": "object",
    "properties": {},
    "required": [],
}

# HandoffConfig が型付きで扱う handoff() 引数（options 素通しでの二重指定を禁止する）。
_HANDOFF_RESERVED_KEYS = frozenset(
    {
        "agent",
        "tool_description_override",
        "tool_name_override",
        "on_handoff",
        "input_type",
        "input_filter",
        "is_enabled",
    }
)

# DynamicHandoff が型付きで扱う生 Handoff フィールド（options 素通しでの重複を禁止する）。
# 静的版（SDK `handoff()` ヘルパ引数名）と物理キー名が異なるのは、動的版が `Handoff(...)` を
# 直接構築するため。AC #7（学習統一性）は「型付きフィールドを使う」運用ルールで担保する。
_DYNAMIC_HANDOFF_RESERVED_KEYS = frozenset(
    {
        "tool_name",
        "tool_description",
        "input_json_schema",
        "on_invoke_handoff",
        "agent_name",
        "input_filter",
        "is_enabled",
    }
)

# Agent が受け付ける有効な kwarg 名（extra の早期検証に使う）。
_AGENT_FIELD_NAMES = frozenset(f.name for f in _dataclass_fields(Agent))


def build_agent(spec: AgentSpec) -> Agent:
    """spec から handoffs 空の Agent を 1 つ構築する（デフォルト AgentBuilder 実装）。

    `instructions` / `prompt` / `tools` / `model` / `model_settings` / `hooks` /
    `input_guardrails` / `output_guardrails` を `Agent` にそのまま渡す。handoffs は空
    （registry が後付け結線）。

    Args:
        spec: 構築対象の AgentSpec。

    Returns:
        agents.Agent（handoffs は空。サブツール未注入）。

    Raises:
        ValueError: extra に専用フィールド名と同名のキー、または Agent が受け付けない
            未知のキーが含まれる場合。
    """
    extra = dict(spec.extra)
    validate_extra_kwargs(
        spec.name,
        extra,
        dedicated=_DEDICATED_AGENT_KWARGS,
        field_names=_AGENT_FIELD_NAMES,
        agent_label="agents.Agent",
    )

    kwargs: dict[str, Any] = {
        "name": spec.name,
        "instructions": spec.instructions,
        "tools": list(spec.tools),
        "handoffs": [],
        "input_guardrails": list(spec.input_guardrails),
        "output_guardrails": list(spec.output_guardrails),
        **extra,
    }
    if spec.prompt is not None:
        kwargs["prompt"] = spec.prompt
    if spec.model is not None:
        kwargs["model"] = spec.model
    if spec.model_settings is not None:
        kwargs["model_settings"] = spec.model_settings
    if spec.hooks is not None:
        kwargs["hooks"] = spec.hooks
    return Agent(**kwargs)


def make_handoff(agent: Agent, config: HandoffConfig) -> Handoff:
    """SDK の handoff() を HandoffConfig（型付きフィールド + options 素通し）で生成する。

    型付きフィールド（description / tool_name / on_handoff / input_type / input_filter /
    is_enabled）を handoff() の対応引数へマップし、残りは config.options で素通しする。

    Args:
        agent: ハンドオフ先の Agent インスタンス。
        config: ハンドオフ設定。

    Returns:
        agents.Handoff。

    Raises:
        ValueError: options に型付きフィールドと重複する予約キーが含まれる場合。
    """
    opts = dict(config.options)
    reserved = _HANDOFF_RESERVED_KEYS & opts.keys()
    if reserved:
        raise ValueError(
            f"handoff options に予約キーが含まれます: {sorted(reserved)}"
            "（HandoffConfig の型付きフィールドを使ってください）"
        )
    kwargs: dict[str, Any] = {}
    if config.description is not None:
        kwargs["tool_description_override"] = config.description
    if config.tool_name is not None:
        kwargs["tool_name_override"] = config.tool_name
    if config.on_handoff is not None:
        kwargs["on_handoff"] = config.on_handoff
    if config.input_type is not None:
        kwargs["input_type"] = config.input_type
    if config.input_filter is not None:
        kwargs["input_filter"] = config.input_filter
    if config.is_enabled is not True:
        kwargs["is_enabled"] = config.is_enabled
    return handoff(agent, **kwargs, **opts)


def _build_input_json_schema(type_adapter: Any, *, strict: bool = True) -> dict[str, Any]:
    """構築済み `TypeAdapter` から JSON Schema を生成する。

    canonical path は `pydantic.TypeAdapter(input_type).json_schema()`。フィールド
    `Field(description=...)` とモデル docstring は pydantic 標準挙動で
    `properties[name].description` / 全体 `description` に展開される（素通し）。

    `strict=True`（既定）の場合、SDK の `agents.strict_schema.ensure_strict_json_schema` が
    利用できれば生成スキーマを OpenAI strict tool calling 形式に整形する
    （`additionalProperties: false` の付与・全 properties を `required` に含める等）。
    SDK 内部 API のため、不在時 / 想定外スキーマ時は pydantic 生成スキーマをそのまま返す。

    `strict=False` の場合は strict 化を skip し、pydantic 生成スキーマ（optional / default 値
    ありフィールドは `required` に含まれない）をそのまま返す。`Handoff.strict_json_schema=False`
    と整合させるためのフラグ。

    Args:
        type_adapter: `pydantic.TypeAdapter(input_type)`。`make_dynamic_handoff` 内で 1 度だけ
            構築し、毎ハンドオフ呼出で再生成しない（キャッシュ用途）。
        strict: True で strict 化を試みる（既定）。False で pydantic 生スキーマを返す。

    Returns:
        生成された JSON Schema。
    """
    schema = type_adapter.json_schema()
    if not strict:
        return schema
    try:
        from agents.strict_schema import (  # type: ignore[import-not-found]
            ensure_strict_json_schema,
        )
    except ImportError:
        return schema
    try:
        return ensure_strict_json_schema(schema)
    except (TypeError, ValueError, KeyError):
        # 厳格化が想定するスキーマ形状から外れた場合のみ素スキーマで返却する。
        # NameError / AttributeError / 等の未知例外は SDK 側リグレッションを示唆するため
        # 黙って握りつぶさず伝播させる（本層は防御フォールバックの過剰拡張をしない）。
        return schema


async def _call_on_handoff(
    on_handoff: Callable[..., Any],
    context: Any,
    parsed: Any,
    *,
    has_input_type: bool,
) -> None:
    """`on_handoff` を `has_input_type` に応じて呼び分け、await 可能なら await する。

    `has_input_type` が True なら `(context, parsed)` で、False なら `(context,)` で呼ぶ
    （SDK `handoff()` ヘルパの実行時挙動と同等）。シグネチャ検査・arity ディスパッチ・
    `TypeError` リトライは行わない。`*args` や optional な 2 引数目を持つ柔軟シグネチャでも
    Python のパラメータ解決に委ね、形が合わなければ `TypeError` をそのまま伝播させる
    （リトライによる副作用の二重実行と、callback 本体由来の `TypeError` の握りつぶしを避ける）。

    Args:
        on_handoff: ユーザー指定のコールバック。
        context: SDK の `RunContextWrapper` 等。
        parsed: `input_type` 指定時のパース済み引数（`has_input_type` が False のとき未使用）。
        has_input_type: `input_type` が指定されているか。
    """
    if has_input_type:
        result = on_handoff(context, parsed)
    else:
        result = on_handoff(context)
    if inspect.isawaitable(result):
        await result


def make_dynamic_handoff(
    *,
    tool_name: str,
    description: str | None,
    on_invoke: Callable[..., Awaitable[Agent]],
    on_handoff: Any = None,
    input_type: Any = None,
    input_filter: Any = None,
    is_enabled: Any = True,
    options: dict[str, Any] | None = None,
) -> Handoff:
    """生 Handoff を組み、on_invoke_handoff で転送先を実行時決定する動的ハンドオフを作る。

    handoff() は固定ターゲット専用で on_invoke_handoff を差し込めないため、Handoff を
    直接構築する。入力スキーマは `input_type` 指定時に `pydantic.TypeAdapter` から生成し、
    未指定時は無入力（SDK の handoff(agent) と同一）。`input_type` 指定時は `TypeAdapter` を
    本関数内で 1 度だけ構築し、呼出毎の再生成を避ける。

    `input_type` または `on_handoff` のいずれかが指定された場合に元 on_invoke（resolver で
    転送先 Agent を返すクロージャ）に処理を合成する。順序は input_json 検証 → 転送先決定
    → on_handoff 発火 → Agent return。`input_type` 指定時は on_handoff 有無に関わらず
    pydantic で入力検証を行い、LLM の不正な構造化引数を `ValidationError` として早期に
    表面化する（SDK 静的 handoff() と同等の挙動）。

    SDK 静的 handoff() は agent が a priori 既知のため「parse → on_handoff → return」だが、
    動的版では転送先を resolver で決定する必要があるため「parse → resolver → on_handoff →
    return」の順序を取る。これにより resolver が失敗した場合は on_handoff が発火せず、
    resolver の副作用と on_handoff の副作用の不整合を避ける（resolver は転送先名のみ返す
    純関数寄りの選択器という設計分離に従う）。on_handoff 発火中の例外は target 返却前に
    伝播し SDK 側で tool execution error として扱われる。

    `input_filter` / `is_enabled` / `options` は生 `Handoff` のフィールドへ素通しする。

    Args:
        tool_name: ハンドオフ tool 名。
        description: ハンドオフ tool の説明。
        on_invoke: `(context, input_json) -> Awaitable[Agent]`。転送先 Agent を返す。
        on_handoff: 転送先決定後・Agent 返却前に発火するコールバック（任意）。
        input_type: 転送時に LLM が埋める構造化入力の型（任意）。
        input_filter: 次エージェントへ渡す履歴の変換（任意）。
        is_enabled: 動的有効化（bool または `(context, agent) -> bool` callable）。
        options: 上記以外の生 `Handoff` kwarg 素通し用 dict（任意）。`registry` 側で既に
            防御コピー済みのため本関数では再コピーしない。

    Returns:
        agents.Handoff。

    Raises:
        ValueError: options に型付きフィールドと重複する予約キーが含まれる場合。
    """
    opts: dict[str, Any] = options if options is not None else {}
    reserved = _DYNAMIC_HANDOFF_RESERVED_KEYS & opts.keys()
    if reserved:
        raise ValueError(
            f"dynamic handoff options に予約キーが含まれます: {sorted(reserved)}"
            "（DynamicHandoff の型付きフィールドを使ってください）"
        )

    # options で strict_json_schema=False が指定されたら strict 化を skip する。
    # 生 `Handoff` の `strict_json_schema` フィールドへの素通し（handoff_kwargs.update(opts)）
    # だけでは LLM 提示スキーマが既に strict 化済みになる時間順序の問題があるため、
    # opt-out フラグはスキーマ生成段階でも反映する必要がある（optional / default 値ありフィールド
    # が `required` に勝手に含まれる挙動を回避）。
    strict_json_schema = opts.get("strict_json_schema", True)

    type_adapter: Any = None
    input_json_schema: dict[str, Any] = dict(_EMPTY_HANDOFF_SCHEMA)
    has_input_type = input_type is not None
    if has_input_type:
        from pydantic import TypeAdapter

        type_adapter = TypeAdapter(input_type)
        input_json_schema = _build_input_json_schema(type_adapter, strict=strict_json_schema)

    effective_on_invoke: Callable[..., Awaitable[Agent]] = on_invoke
    if has_input_type or on_handoff is not None:

        async def _wrapped_on_invoke(context: Any, input_json: Any = None) -> Agent:
            parsed: Any = None
            if has_input_type and input_json is not None:
                if isinstance(input_json, (str, bytes, bytearray)):
                    # 空 str / bytes / bytearray は SDK 文脈で「入力なし」を意味するため
                    # parse を skip し parsed=None で発火する（None と同等扱い）。空白のみの
                    # 文字列は意図的に validate_json へ送って pydantic の ValidationError として
                    # 表面化させる（silent succeed しない）。
                    if input_json:
                        parsed = type_adapter.validate_json(input_json)
                else:
                    # SDK の `on_invoke_handoff` は型ヒント上 str を渡す契約だが、テストや
                    # 将来 SDK 拡張で既にパース済みの dict / list が直接渡る defensive 経路。
                    # 空 dict / 空 list も意味のある「入力なし」状態なので validate_python に
                    # 委ね、必須フィールド欠落は ValidationError として表面化させる。
                    parsed = type_adapter.validate_python(input_json)
            target = await on_invoke(context, input_json)
            if on_handoff is not None:
                await _call_on_handoff(on_handoff, context, parsed, has_input_type=has_input_type)
            return target

        effective_on_invoke = _wrapped_on_invoke

    handoff_kwargs: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_description": description or "",
        "input_json_schema": input_json_schema,
        "on_invoke_handoff": effective_on_invoke,
        "agent_name": tool_name,
    }
    if input_filter is not None:
        handoff_kwargs["input_filter"] = input_filter
    if is_enabled is not True:
        handoff_kwargs["is_enabled"] = is_enabled
    handoff_kwargs.update(opts)
    return Handoff(**handoff_kwargs)


class DefaultAgentBuilder:
    """`build_agent` をラップするデフォルト AgentBuilder 実装。"""

    def build(self, spec: AgentSpec) -> Agent:
        return build_agent(spec)


def make_agent_tool(agent: Agent, *, tool_name: str | None, tool_description: str | None) -> Any:
    """サブエージェントを as_tool でツール化する（keyword 渡し）。

    tool_name 省略時は SDK がエージェント名から導出する（独自実装しない）。

    Args:
        agent: サブエージェントの（ビルド済み）Agent インスタンス。
        tool_name: ツール名。None で SDK 既定（エージェント名由来）。
        tool_description: ツール説明。None で SDK 既定（空文字）。

    Returns:
        FunctionTool（agents.Tool として tools に追加可能）。
    """
    return agent.as_tool(tool_name=tool_name, tool_description=tool_description)


def _mock_invoker(value: Any) -> Any:
    """tool_mocks の値（静的値 or callable）を `on_invoke_tool` シグネチャの実装に包む。

    `(ctx, input_json: str) -> str` を満たす coroutine を返す（`models.py` の on_invoke_tool が
    手本）。値が callable のとき `json.loads(input_json or "{}")` で引数 dict を復元して渡し、
    その戻り値を str 化する。静的値のときは入力に関わらず `str(値)` を返す。`needs_approval` は
    差し替えないため（呼び出し側で維持）、本実装は**ツール実行だけ**を置換する。

    callable mock が送出した例外は握りつぶさず SDK のツール実行エラー経路へ委ねる（JSON パース
    失敗のみ空 dict にフォールバックし、それ以外の例外は呼び出し元へ伝播させる）。

    Args:
        value: 静的値、または `callable(args: dict) -> Any`。

    Returns:
        `on_invoke_tool` 互換の async 関数。
    """

    async def _on_invoke_tool(ctx: Any, input_json: str) -> str:  # noqa: ARG001 - SDK シグネチャ
        if callable(value):
            try:
                args = json.loads(input_json) if input_json else {}
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            return str(value(args))
        return str(value)

    return _on_invoke_tool


def mock_spec_tools(
    spec: AgentSpec, tool_mocks: dict[str, Any]
) -> tuple[AgentSpec, set[tuple[str, str]]]:
    """宣言（`AgentSpec`）層で当該 agent の `tool_mocks` 指定ツールの**実行だけ**を差し替える。

    `tool_mocks` は **その agent スコープの `{tool_name: 値 | callable}`**（呼び出し側が
    `全体tool_mocks.get(spec.name, {})` を渡す）。`spec.tools` 内の同名 `FunctionTool` を
    `dataclasses.replace(tool, on_invoke_tool=モック実装)` で差し替えた**新しい tools リスト**を
    作り、`dataclasses.replace(spec, tools=...)` で新 spec を返す（元 spec / 元 tool は不変）。
    **name / description / params_json_schema / needs_approval は維持する**（承認ゲートを発火させ
    HITL 経路を通すのが #29 の核・差し替えるのは実行本体だけ）。差し替え対象が無ければ元 spec を
    そのまま返す。

    **実際に差し替えた `(spec.name, tool_name)` ペア集合**も返す（plain `set[tuple[str, str]]`）。
    呼び出し側はこの集合で「approve してよいツール（= 当該 agent でモックに差し替え済み）」を
    判定する（fail-closed の根拠を tool_mocks のキーではなく実差し替えに置く・同名ツールでも別
    agent の approve を認可しない・Codex P1）。

    `tool_mocks` の各値は静的値（`str(値)` を返す）または `callable(args: dict)`（JSON 引数を
    渡し戻りを str 化）。FunctionTool 以外（as_tool / MCP 等）や同名でないツールは触らない。

    SDK 型操作（`FunctionTool` の `dataclasses.replace`）を `_adapters` に閉じるためのヘルパで、
    呼び出し側（`_target`）は plain な `AgentSpec` / `tool_mocks` dict のみ扱う（NFR-1）。

    Args:
        spec: 変換対象の `AgentSpec`（plain・core 型）。
        tool_mocks: 当該 agent スコープの「ツール名 -> モック実装（静的値 or callable）」の dict。

    Returns:
        `(差し替え済み AgentSpec, 実差し替えした (agent, tool) ペア集合)`。空なら `(spec, set())`。
    """
    if not tool_mocks:
        return spec, set()
    replaced: set[tuple[str, str]] = set()
    new_tools: list[Any] = []
    for tool in spec.tools:
        if isinstance(tool, FunctionTool) and tool.name in tool_mocks:
            new_tools.append(
                _dataclass_replace(tool, on_invoke_tool=_mock_invoker(tool_mocks[tool.name]))
            )
            replaced.add((spec.name, tool.name))
        else:
            new_tools.append(tool)
    if not replaced:
        return spec, set()
    return _dataclass_replace(spec, tools=new_tools), replaced
