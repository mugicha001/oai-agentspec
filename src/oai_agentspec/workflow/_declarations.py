"""ワークフローの宣言値 dataclass 群と非公開 runner シーム（Protocol）。

`WorkflowNode` / `ConditionalEdge` / `FanInEdge`（トポロジ宣言値）、`NodeResults` / `WorkflowResult`
（実行スコープの記録・結果）、`RunnerSeam`（AGENT ノード実行を委譲する DI 注入点 Protocol）を
提供する。SDK には依存しない（NFR-1）。
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ._types import NodeFn, NodeKind, Router

__all__ = [
    "ConditionalEdge",
    "FanInEdge",
    "NodeResults",
    "WorkflowNode",
    "WorkflowResult",
]


@dataclass
class WorkflowNode:
    """ワークフローの 1 ノード（宣言値）。

    Attributes:
        name: ノード名（WorkflowGraph 内で一意）。
        kind: ノード種別（AGENT / FUNCTION）。
        agent: AGENT ノードの registry 上のエージェント名（FUNCTION では None）。
        fn: FUNCTION ノードの callable（AGENT では None）。
        run_options: AGENT ノードの Runner.run へ素通しする kwarg（グラフ既定 run_defaults を
            上書き）。`input` / `context` / `session` は予約キーで指定不可。
    """

    name: str
    kind: NodeKind
    agent: str | None = None
    fn: NodeFn | None = None
    run_options: dict[str, Any] | None = None


@dataclass
class ConditionalEdge:
    """条件エッジの宣言（router の戻り値で 1 経路を選ぶ・FR-2）。

    Attributes:
        src: 分岐元ノード名。
        router: `(msg, ctx) -> ノード名 | END | 判定キー`。mapping=None なら戻り値を
            次ノード名 | END として直接使う。mapping ありなら戻り値をキーとして引く。
        mapping: 判定キー -> 次ノード名 | END の有限写像（None で戻り値を直接使用）。
        default: mapping に解決しない場合の既定の行き先（ノード名 | END）。None で未一致は例外。
        candidates: mapping=None（router が動的にノード名を返す）時に、可能な行き先を宣言する
            ためのリスト（検証・到達性・可視化用。LangGraph の path_map 相当）。
    """

    src: str
    router: Router
    mapping: dict[Hashable, Any] | None
    default: Any = None
    candidates: list[Any] | None = None


@dataclass
class FanInEdge:
    """fan-in（合流）エッジの宣言（合流先は FUNCTION 必須・C-4/FR-1）。

    Attributes:
        sources: 合流するソースノード名のリスト（全完了後に dst へ進む）。
        dst: 合流先ノード名（FUNCTION ノードのみ。`{source名: 出力}` の dict を受ける）。
    """

    sources: list[str]
    dst: str


@dataclass
class NodeResults:
    """実行中スコープのみ保持する薄い可変記録（run 終了で破棄・NFR-3）。

    「ノード名 -> 出力」を運ぶ。会話履歴は保持しない（SDK の RunResult / Session が
    真実源・FR-11）。フックへ第 2 引数として渡され、実行経緯の観測に使う（FR-13）。

    Attributes:
        outputs: ノード名 -> そのノードの出力。
    """

    outputs: dict[str, Any] = field(default_factory=dict)

    def record(self, name: str, output: Any) -> None:
        """ノード出力を記録する。

        Args:
            name: ノード名。
            output: そのノードの出力。
        """
        self.outputs[name] = output

    def get(self, name: str, default: Any = None) -> Any:
        """記録済みノード出力を取得する（観測用）。

        Args:
            name: ノード名。
            default: 未記録時に返す既定値。

        Returns:
            記録済み出力、または default。
        """
        return self.outputs.get(name, default)


@dataclass
class WorkflowResult:
    """内部インタプリタの実行結果（lib 内部値）。

    Attributes:
        final_output: ワークフローの最終出力（END へ到達したノードの出力）。
        results: 実行スコープの NodeResults（観測用）。
    """

    final_output: Any
    results: NodeResults


class RunnerSeam(Protocol):
    """AGENT ノード実行を委譲する非公開の runner シーム（DI 注入点）。

    SDK `Runner.run` への**素通し（passthrough）シーム**である（NFR-7）。`input` /
    `context` のみ lib が明示管理し、それ以外の Runner kwarg（run_config / session /
    max_turns / hooks / conversation_id / previous_response_id 等）は `**runner_kwargs`
    として `Runner.run` へそのまま委譲する。本番実装は `_adapters` の既定 runner
    （`Runner.run` ラップ）、テストは fake を注入する。トップレベル `__all__` には出さない
    （公開 API ではない・利用者は runner を渡さない）。
    """

    async def run(
        self,
        agent: str,
        input: Any,
        *,
        context: Any = None,
        **runner_kwargs: Any,
    ) -> Any:
        """AGENT ノードを実行し最終出力を持つ RunResult 相当を返す。

        Args:
            agent: 実行する registry 上のエージェント名。
            input: ノードへの入力（上流出力。string / SDK input-list を期待）。
            context: 各ノードへ素通しする共有 context（経路A 時のみ非 None・C-11）。
            **runner_kwargs: `Runner.run` へ素通しする残りの kwarg（run_config / session /
                max_turns 等。グラフ既定 run_defaults + ノード run_options のマージ結果）。

        Returns:
            `final_output` 属性を持つ RunResult 相当オブジェクト。
        """
        ...
