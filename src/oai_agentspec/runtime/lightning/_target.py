"""最適化対象（AgentSpec / HandoffGraph / WorkflowGraph）の実行可能 Agent への正規化。

型で build 経路を分岐する:
    - `AgentSpec`    -> `_adapters.build_agent(spec)` 直接（registry 不要・handoffs 空）。
    - `HandoffGraph` -> registry 必須（spec 実体を持たないため）。`graph.apply(registry)` で
      エッジ反映 → `entry_agent(registry)`。
    - `WorkflowGraph`-> `as_agent_spec(name, registry=registry)` -> `build_agent`。
      registry は **AGENT ノードを含む場合のみ必要**で、関数ノードのみの workflow は registry=None
      でも最適化できる（registry はそのまま素通しし、AGENT ノードがあれば as_agent_spec / runner
      側で自然にエラーになる）。

`tool_mocks`（**agent スコープのネスト dict** `{agent_name: {tool_name: 値 | callable}}`）指定時は
**build 前の宣言（spec / registry）層**でツール実行をモック化する（build 済み Agent を mutate
しない＝利用者の registry / キャッシュ済み Agent を汚さない）:
    - `AgentSpec`    -> `mock_spec_tools(spec, tool_mocks.get(spec.name, {}))` で spec を差し替え。
    - 横断（HandoffGraph / WorkflowGraph）-> 利用者 registry を `clone(transform_spec=...)` で
      クローンし、各 spec を自分の名前のエントリでモック化した派生 registry を作って build する。
      dynamic handoff の候補もクローン経由で解決されるためモック済みになる（利用者 registry は
      不変）。HandoffGraph の mock 経路では、`apply` がグラフの `_applied_srcs`（差分クリア用
      bookkeeping）を書き換えるため、利用者グラフを直接 apply せず deepcopy したグラフを apply
      する（利用者グラフも無傷に保つ）。

未対応型は許容型を列挙した明示 `TypeError`、HandoffGraph の registry 不足は明示 `ValueError`。
SDK 型操作（FunctionTool の差し替え）は `_adapters.mock_spec_tools` に閉じ、本モジュールは plain な
`AgentSpec` / ネスト `tool_mocks` dict のみ扱う（NFR-1）。`_adapters` は関数内遅延 import に閉じる。

NOTE: 本モジュールは llmops `_target` を最適化文脈へ調整したものである（評価文脈固定のメッセージ
を最適化文脈へ置換）。コア層への共有ユーティリティ昇格は本スコープ外。
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from ...handoffs import HandoffGraph
from ...spec import AgentSpec
from ...workflow import WorkflowGraph

if TYPE_CHECKING:
    from ...registry import AgentRegistry


def target_id(target: Any) -> str:
    """最適化対象の識別子を導出する。

    Args:
        target: 最適化対象（AgentSpec / HandoffGraph / WorkflowGraph）。

    Returns:
        AgentSpec は `name`、HandoffGraph は entry 名（または "handoff_graph"）、
        WorkflowGraph は "workflow"。
    """
    if isinstance(target, AgentSpec):
        return target.name
    if isinstance(target, HandoffGraph):
        return target.entry or "handoff_graph"
    if isinstance(target, WorkflowGraph):
        return "workflow"
    return str(getattr(target, "name", "target"))


def normalize(
    target: Any,
    registry: AgentRegistry | None,
    *,
    tool_mocks: dict[str, dict[str, Any]] | None = None,
) -> tuple[Any, frozenset[tuple[str, str]]]:
    """最適化対象を実行可能 Agent へ正規化し、`(agent, 実差し替え (agent, tool) 集合)` を返す。

    `tool_mocks` は **agent スコープのネスト dict**（`{agent_name: {tool_name: 値 | callable}}`）。
    指定時は **build 前の宣言層**でツール実行をモック化する（build 済み Agent を mutate しない）。
    AgentSpec は `tool_mocks.get(spec.name, {})` で spec を差し替えて build、横断（HandoffGraph /
    WorkflowGraph）は利用者 registry をクローンし各 spec を自分の名前のエントリでモック化した派生
    registry から build する（利用者 registry は不変。dynamic handoff 候補もクローン経由でモック
    済み）。`replaced` は全 spec 横断で実際にモックへ差し替えた `(agent, tool)` ペアの集合で、
    呼び出し側が approve 認可（同名ツールでも別 agent を認可しない fail-closed）に使う。

    Args:
        target: 最適化対象（AgentSpec / HandoffGraph / WorkflowGraph）。
        registry: 横断対象の specs 供給経路。HandoffGraph は必須（spec 実体を持たないため）。
            WorkflowGraph は **AGENT ノードを含む場合のみ必要**（関数ノードのみなら None 可）。
            AgentSpec 単体最適化では不要。
        tool_mocks: agent スコープのモック dict（`{agent: {tool: 値 | callable}}`）。None / 空で
            モック化しない（`replaced` は空集合）。

    Returns:
        `(構築済み Agent（不透明型）, 実差し替えした (agent, tool) ペアの frozenset)`。

    Raises:
        TypeError: 未対応の target 型の場合（許容型を列挙）。
        ValueError: HandoffGraph に registry が供給されていない場合。
    """
    from ..._adapters import build_agent, mock_spec_tools

    mocks = tool_mocks or {}

    if isinstance(target, AgentSpec):
        spec, replaced = mock_spec_tools(target, mocks.get(target.name, {}))
        return build_agent(spec), frozenset(replaced)

    if isinstance(target, HandoffGraph):
        if registry is None:
            raise ValueError(
                "HandoffGraph の最適化には specs 登録済みの registry が必須です"
                "（optimize(registry=...) で渡してください）"
            )
        # 最適化フローは非破壊契約のため、グラフ・registry の両方を独立コピーしてから apply する。
        # グラフは `_applied_srcs`（差分クリア用 bookkeeping）が書き換わり、`apply` は registry 内
        # spec の `handoffs` も書き換えうるため、利用者の双方を無傷に保つ。resolver は関数のため
        # deepcopy 後も同一関数を参照（問題なし）。
        opt_graph = copy.deepcopy(target)
        if mocks:
            opt_registry, replaced = _mocked_registry(registry, mocks)
        else:
            opt_registry, replaced = registry.clone(), set()
        opt_graph.apply(opt_registry)
        return opt_graph.entry_agent(opt_registry), frozenset(replaced)

    if isinstance(target, WorkflowGraph):
        # registry は素通し（None 可）。AGENT ノードを含む場合のみ as_agent_spec / runner 側で
        # 自然にエラーになる。関数ノードのみの workflow は registry=None でも最適化できる。
        replaced: set[tuple[str, str]] = set()
        opt_registry = registry
        if registry is not None and mocks:
            opt_registry, replaced = _mocked_registry(registry, mocks)
        spec = target.as_agent_spec(target_id(target), registry=opt_registry)
        return build_agent(spec), frozenset(replaced)

    raise TypeError(
        "未対応の最適化対象型です: "
        f"{type(target).__name__}"
        "（AgentSpec / HandoffGraph / WorkflowGraph を渡してください）"
    )


def _mocked_registry(
    registry: AgentRegistry, tool_mocks: dict[str, dict[str, Any]]
) -> tuple[AgentRegistry, set[tuple[str, str]]]:
    """利用者 registry をクローンし、各 spec の tools を自分の名前のエントリでモック化して返す。

    呼び出し側は **非空の `tool_mocks` でのみ**本関数を呼ぶ（空ならクローン不要なので呼ばない）。
    `registry.clone(transform_spec=...)` で、各 spec を `mock_spec_tools(spec,
    tool_mocks.get(spec.name, {}))` で変換しながら独立した新 registry を組み立てる（利用者
    registry は不変）。全 spec 横断で実際に差し替えた `(agent, tool)` ペアを集約して返す。factory
    登録の agent は spec 実体を持たず変換対象外（その上の承認ツールは差し替えられず `replaced` に
    入らない＝approve 時に fail-closed のエラー）。

    SDK 型操作（FunctionTool 差し替え）は `mock_spec_tools`（`_adapters`）に閉じ、本関数は core 型
    （AgentSpec / AgentRegistry）と plain なネスト dict のみ扱う（NFR-1）。

    Args:
        registry: 利用者が渡した元 registry（不変）。
        tool_mocks: agent スコープのモック dict（`{agent: {tool: 値 | callable}}`・非空）。

    Returns:
        `(派生 registry, 実差し替えした (agent, tool) ペア集合)`。
    """
    from ..._adapters import mock_spec_tools

    replaced: set[tuple[str, str]] = set()

    def _transform(spec: AgentSpec) -> AgentSpec:
        mocked, pairs = mock_spec_tools(spec, tool_mocks.get(spec.name, {}))
        replaced.update(pairs)
        return mocked

    cloned = registry.clone(transform_spec=_transform)
    return cloned, replaced
