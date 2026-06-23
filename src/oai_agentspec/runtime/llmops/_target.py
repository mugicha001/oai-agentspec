"""評価対象（AgentSpec / HandoffGraph / WorkflowGraph）の実行可能 Agent への正規化。

型で build 経路を分岐する:
    - `AgentSpec`    -> `_adapters.build_agent(spec)` 直接（registry 不要・handoffs 空）。
    - `HandoffGraph` -> registry 必須（spec 実体を持たないため）。`graph.apply(registry)` で
      エッジ反映 → `entry_agent(registry)`。
    - `WorkflowGraph`-> `as_agent_spec(name, registry=registry)` -> `build_agent`。
      registry は **AGENT ノードを含む場合のみ必要**で、関数ノードのみの workflow は registry=None
      でも評価できる（registry はそのまま素通しし、AGENT ノードがあれば as_agent_spec / runner 側で
      自然にエラーになる）。

`tool_mocks`（**agent スコープのネスト dict** `{agent_name: {tool_name: 値 | callable}}`）指定時は
**build 前の宣言（spec / registry）層**でツール実行をモック化する（#29・build 済み Agent を mutate
しない＝利用者の registry / キャッシュ済み Agent を汚さない）:
    - `AgentSpec`    -> `mock_spec_tools(spec, tool_mocks.get(spec.name, {}))` で spec を差し替え。
    - 横断（HandoffGraph / WorkflowGraph）-> 利用者 registry を `clone(transform_spec=...)` で
      クローンし、各 spec を自分の名前のエントリでモック化した派生 registry を作って build する。
      dynamic handoff の候補もクローン経由で解決されるためモック済みになる（利用者 registry は
      不変）。HandoffGraph の mock 経路では、`apply` がグラフの `_applied_srcs`（差分クリア用
      bookkeeping）を書き換えるため、利用者グラフを直接 apply せず deepcopy したグラフを apply
      する（利用者グラフも無傷に保つ・Codex P2）。

未対応型は許容型を列挙した明示 `TypeError`、HandoffGraph の registry 不足は明示 `ValueError`。
SDK 型操作（FunctionTool の差し替え）は `_adapters.mock_spec_tools` に閉じ、本モジュールは plain な
`AgentSpec` / ネスト `tool_mocks` dict のみ扱う（NFR-1）。`_adapters` は関数内遅延 import に閉じる。
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
    """評価対象の識別子（`EvaluationResult.target_id`）を導出する。

    Args:
        target: 評価対象（AgentSpec / HandoffGraph / WorkflowGraph）。

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


def extract_prompt(target: Any) -> str | None:
    """評価対象から静的プロンプト（`AgentSpec.instructions` が文字列）を抽出する。

    `AgentSpec` で `instructions` が静的文字列のときのみ抽出する。callable / 動的 prompt
    （`AgentSpec.prompt` 設定）/ 横断（HandoffGraph / WorkflowGraph・単一プロンプト不特定）は
    None を返す（Langfuse PM 連携でスキップ・push 専用・§18・§15-7）。

    Args:
        target: 評価対象。

    Returns:
        抽出できた静的プロンプト文字列。抽出不可なら None。
    """
    if (
        isinstance(target, AgentSpec)
        and target.prompt is None
        and isinstance(target.instructions, str)
    ):
        return target.instructions
    return None


def normalize(
    target: Any,
    registry: AgentRegistry | None,
    *,
    tool_mocks: dict[str, dict[str, Any]] | None = None,
) -> tuple[Any, frozenset[tuple[str, str]]]:
    """評価対象を実行可能 Agent へ正規化し、`(agent, 実差し替え (agent, tool) 集合)` を返す。

    `tool_mocks` は **agent スコープのネスト dict**（`{agent_name: {tool_name: 値 | callable}}`）。
    指定時は **build 前の宣言層**でツール実行をモック化する（build 済み Agent を mutate しない）。
    AgentSpec は `tool_mocks.get(spec.name, {})` で spec を差し替えて build、横断（HandoffGraph /
    WorkflowGraph）は利用者 registry をクローンし各 spec を自分の名前のエントリでモック化した派生
    registry から build する（利用者 registry は不変・P2-1 解消。dynamic handoff 候補もクローン
    経由でモック済み・P2-2 解消）。`replaced` は全 spec 横断で実際にモックへ差し替えた
    `(agent, tool)` ペアの集合で、呼び出し側が approve 認可（同名ツールでも別 agent を認可しない
    fail-closed・Codex P1）に使う。

    実行モードや対象のツール保有有無は返さない（NA 判定は ground truth 非在のみが根拠で、
    対象の能力には依存しないため・観点の適用可否は利用者の criteria 選択に委ねる）。

    Args:
        target: 評価対象（AgentSpec / HandoffGraph / WorkflowGraph）。
        registry: 横断対象の specs 供給経路。HandoffGraph は必須（spec 実体を持たないため）。
            WorkflowGraph は **AGENT ノードを含む場合のみ必要**（関数ノードのみなら None 可）。
            AgentSpec 単体評価では不要。
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
                "HandoffGraph の評価には specs 登録済みの registry が必須です"
                "（evaluate(registry=...) で渡してください）"
            )
        if mocks:
            # mock（クローン）経路: 利用者グラフを直接 apply せず deepcopy したグラフを apply する。
            # `apply` はグラフの `_applied_srcs`（差分クリア用 bookkeeping）を書き換えるため、直接
            # apply すると利用者グラフの状態が汚れ、後の `graph.apply(real_registry)` が stale な
            # handoffs を正しくクリアできなくなる（Codex P2）。deepcopy で edges / dynamic /
            # _applied_srcs を複製し（resolver は関数のため参照維持・問題なし）、利用者グラフを
            # 無傷に保つ。registry も `_mocked_registry` でクローン済みで利用者 registry も不変。
            eval_registry, replaced = _mocked_registry(registry, mocks)
            eval_graph = copy.deepcopy(target)
            eval_graph.apply(eval_registry)
            return eval_graph.entry_agent(eval_registry), frozenset(replaced)
        # 非 mock 経路は #24 既存挙動のまま（利用者 registry / グラフを直接使う前提）。
        target.apply(registry)
        return target.entry_agent(registry), frozenset()

    if isinstance(target, WorkflowGraph):
        # registry は素通し（None 可）。AGENT ノードを含む場合のみ as_agent_spec / runner 側で
        # 自然にエラーになる。関数ノードのみの workflow は registry=None でも評価できる。
        replaced: set[tuple[str, str]] = set()
        eval_registry = registry
        if registry is not None and mocks:
            eval_registry, replaced = _mocked_registry(registry, mocks)
        spec = target.as_agent_spec(target_id(target), registry=eval_registry)
        return build_agent(spec), frozenset(replaced)

    raise TypeError(
        "未対応の評価対象型です: "
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
