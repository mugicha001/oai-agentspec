"""ワークフロー実行を SDK tracing（custom_span）へ載せる adapter。

SDK 隔離（NFR-1）: `agents` の tracing API（`custom_span` / `get_current_trace`）の import
は本ファイルに局在化する。workflow 層は `_adapters.WorkflowTracer` / `make_workflow_tracer()`
のみを関数内遅延 import で取得する。

設計の核:

- 全 span は `custom_span(name, data=...)` で発行（`agent_span` / `function_span` は使わない）。
- span name は `workflow.` プレフィックス統一（`workflow.run.<graph_name>` /
  `workflow.node.<name>` 等）。
- data 属性は OpenTelemetry 風 namespace `workflow.<key>`。
- tracing 無効時（`set_tracing_disabled(True)` または `get_current_trace() is None`）は
  no-op CM を返すファクトリ構造（span オブジェクトを生成しない）。
- tracer は run ごとに新規生成するステートレス factory。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol

from agents import custom_span, get_current_trace

__all__ = [
    "WorkflowTracer",
    "make_workflow_tracer",
]


# ----------------------------------------------------------------------
# 命名関数（テストから直接 assert する用に分離）
# ----------------------------------------------------------------------
def _workflow_span_name(graph_name: str | None) -> str:
    """workflow span の name を返す（graph_name 未設定時は anonymous フォールバック）。

    Args:
        graph_name: グラフ名（空文字 / None 時は `anonymous` を使う）。

    Returns:
        `workflow.run.<graph_name>` 形式の span name。
    """
    return f"workflow.run.{graph_name or 'anonymous'}"


def _node_span_name(node_name: str) -> str:
    """ノード span の name を返す（種別は data 属性側で持つ）。

    Args:
        node_name: ノード名。

    Returns:
        `workflow.node.<node_name>` 形式の span name。
    """
    return f"workflow.node.{node_name}"


def _condition_span_name(src: str) -> str:
    """条件分岐 span の name を返す（分岐評価のみを span 化する）。

    Args:
        src: 分岐元ノード名。

    Returns:
        `workflow.condition.<src>` 形式の span name。
    """
    return f"workflow.condition.{src}"


def _fan_out_span_name(src: str) -> str:
    """fan-out span の name を返す（並列起動の親）。

    Args:
        src: fan-out 起点のノード名。

    Returns:
        `workflow.fan_out.<src>` 形式の span name。
    """
    return f"workflow.fan_out.{src}"


def _fan_in_span_name(dst: str) -> str:
    """fan-in span の name を返す（合流先）。

    Args:
        dst: 合流先（fan-in dst）のノード名。

    Returns:
        `workflow.fan_in.<dst>` 形式の span name。
    """
    return f"workflow.fan_in.{dst}"


# ----------------------------------------------------------------------
# WorkflowTracer Protocol（workflow 層の依存対象）
# ----------------------------------------------------------------------
class WorkflowTracer(Protocol):
    """ワークフロー run スコープで span を発行する tracer Protocol。

    各メソッドは contextmanager を返し、`with tracer.xxx_span(...)` で span を開閉する。
    実装は SDK の `custom_span` を呼ぶ `_SdkWorkflowTracer` か、span を発行しない
    `_NoopWorkflowTracer` のいずれか（`make_workflow_tracer` が状況に応じて選ぶ）。

    workflow 層は本 Protocol のみに依存し、SDK 型（Span / Trace）を直接参照しない。
    """

    def workflow_span(self, graph_name: str) -> Any:
        """ワークフロー全体を包む span を開く（run の親 span）。

        Args:
            graph_name: グラフ名（空文字なら `anonymous` フォールバック）。

        Returns:
            contextmanager（with でブロックを囲む）。
        """
        ...

    def node_span(self, node_name: str, kind: str) -> Any:
        """単一ノード実行を包む span を開く（AGENT / FUNCTION）。

        Args:
            node_name: ノード名。
            kind: ノード種別（`agent` / `function`）。

        Returns:
            contextmanager。
        """
        ...

    def condition_span(self, src: str) -> Any:
        """条件分岐の評価を包む span を開く（分岐評価のみ・直後に閉じる）。

        Args:
            src: 分岐元ノード名。

        Returns:
            contextmanager。
        """
        ...

    def fan_out_span(self, src: str) -> Any:
        """fan-out の並列起動を包む span を開く（兄弟として並ぶ子 span の親）。

        Args:
            src: fan-out 起点のノード名。

        Returns:
            contextmanager。
        """
        ...

    def fan_in_span(self, dst: str) -> Any:
        """fan-in の合流を包む span を開く（合流先 FUNCTION の実行を内包）。

        Args:
            dst: 合流先ノード名。

        Returns:
            contextmanager。
        """
        ...


# ----------------------------------------------------------------------
# SDK 実装（custom_span でラップ）
# ----------------------------------------------------------------------
class _SdkWorkflowTracer:
    """SDK の `custom_span` を使う `WorkflowTracer` 実装。

    span は親 trace / 親 span を SDK の contextvars から自動継承する（明示 parent 渡しなし）。
    `graph_name` は tracer インスタンス内に保持し、全 span の data に乗せる。

    Attributes:
        graph_name: 全 span の data に乗せるグラフ名（空文字なら `anonymous` フォールバック）。
    """

    def __init__(self, graph_name: str | None) -> None:
        """tracer を生成する。

        Args:
            graph_name: グラフ名（空文字なら `anonymous` を data へ乗せる）。
        """
        self.graph_name: str = graph_name or "anonymous"

    @contextmanager
    def workflow_span(self, graph_name: str) -> Iterator[Any]:
        """ワークフロー全体を包む `workflow.run.<graph>` span を発行する。"""
        with custom_span(
            _workflow_span_name(graph_name),
            data={
                "workflow.graph_name": self.graph_name,
                "workflow.node_kind": "workflow",
            },
        ) as span:
            yield span

    @contextmanager
    def node_span(self, node_name: str, kind: str) -> Iterator[Any]:
        """単一ノードを包む `workflow.node.<name>` span を発行する。"""
        with custom_span(
            _node_span_name(node_name),
            data={
                "workflow.graph_name": self.graph_name,
                "workflow.node_name": node_name,
                "workflow.node_kind": kind,
            },
        ) as span:
            yield span

    @contextmanager
    def condition_span(self, src: str) -> Iterator[Any]:
        """条件分岐評価を包む `workflow.condition.<src>` span を発行する。"""
        with custom_span(
            _condition_span_name(src),
            data={
                "workflow.graph_name": self.graph_name,
                "workflow.node_name": src,
                "workflow.node_kind": "condition",
            },
        ) as span:
            yield span

    @contextmanager
    def fan_out_span(self, src: str) -> Iterator[Any]:
        """fan-out 並列起動を包む `workflow.fan_out.<src>` span を発行する。"""
        with custom_span(
            _fan_out_span_name(src),
            data={
                "workflow.graph_name": self.graph_name,
                "workflow.node_name": src,
                "workflow.node_kind": "fan_out",
            },
        ) as span:
            yield span

    @contextmanager
    def fan_in_span(self, dst: str) -> Iterator[Any]:
        """fan-in 合流を包む `workflow.fan_in.<dst>` span を発行する。"""
        with custom_span(
            _fan_in_span_name(dst),
            data={
                "workflow.graph_name": self.graph_name,
                "workflow.node_name": dst,
                "workflow.node_kind": "fan_in",
            },
        ) as span:
            yield span


# ----------------------------------------------------------------------
# no-op 実装（tracing 無効時のショートカット）
# ----------------------------------------------------------------------
class _NoopWorkflowTracer:
    """span を一切発行しない `WorkflowTracer` 実装（tracing 無効時のショートカット）。

    全メソッドが `@contextmanager` で yield のみを返し、span オブジェクトを生成しない。
    SDK の no-op 経路に乗らず自前で短絡することで、SDK 実装変更に依らない 0 コストを保証する。
    """

    @contextmanager
    def workflow_span(self, graph_name: str) -> Iterator[None]:
        """no-op（span を発行しない）。"""
        yield None

    @contextmanager
    def node_span(self, node_name: str, kind: str) -> Iterator[None]:
        """no-op（span を発行しない）。"""
        yield None

    @contextmanager
    def condition_span(self, src: str) -> Iterator[None]:
        """no-op（span を発行しない）。"""
        yield None

    @contextmanager
    def fan_out_span(self, src: str) -> Iterator[None]:
        """no-op（span を発行しない）。"""
        yield None

    @contextmanager
    def fan_in_span(self, dst: str) -> Iterator[None]:
        """no-op（span を発行しない）。"""
        yield None


# ----------------------------------------------------------------------
# factory（run ごとに新規生成・ステートレス）
# ----------------------------------------------------------------------
def make_workflow_tracer(graph_name: str | None) -> WorkflowTracer:
    """run スコープの `WorkflowTracer` を新規生成する（ステートレス factory）。

    現在の trace を見て tracing 有効時は `_SdkWorkflowTracer`、無効時は
    `_NoopWorkflowTracer` を返す。tracer 状態は run スコープを跨がない（span 親子関係は
    SDK の contextvars に任せる）ため、並行 run でも混線しない。

    Args:
        graph_name: グラフ名（None / 空文字なら span name / data で `anonymous` フォールバック）。

    Returns:
        WorkflowTracer 実装。tracing が完全に無効（`get_current_trace()` が None）か、
        `set_tracing_disabled(True)` 状態（SDK が `NoOpTrace` を current trace として
        セットしている場合）は `_NoopWorkflowTracer` を返し、span 発行を完全にショートカット
        する（SDK の no-op 経路に乗せず、`custom_span()` 呼び出し自体を回避してオーバー
        ヘッド 0 を担保する）。
    """
    current = get_current_trace()
    # `set_tracing_disabled(True)` 配下では SDK が `NoOpTrace` を current にセットするため
    # `is None` だけでは捕捉できない。SDK 内部型への直接 import を避け、type 名比較で判定する
    # （`agents.tracing` 公開窓口から `NoOpTrace` は import できない）。
    if current is None or type(current).__name__ == "NoOpTrace":
        return _NoopWorkflowTracer()
    return _SdkWorkflowTracer(graph_name)
