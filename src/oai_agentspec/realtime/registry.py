"""RealtimeAgent の登録・遅延構築・循環 handoff 解決の中枢（MVP）。

`agents` には依存せず（SDK 型は `TYPE_CHECKING` + `_adapters` 経由）、DI で注入された
`RealtimeAgentBuilder` を用いて RealtimeAgent を遅延構築する。循環ハンドオフは `get(name)`
起点・到達可能 spec のみの局所 2 パス遅延バインドで解決する（通常ルートの `AgentRegistry` の
handoffs のみ版）。

公開 API は `register` / `get` / `names` / `validate` / `entry_name` に限定する（`freeze` /
`clone` / `update` 等は本 MVP の要件外）。依存辺は `spec.handoffs` のみ（sub_agents /
dynamic_handoffs は Realtime ルートに存在しない）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._validation import (
    ensure_static_prompt,
    validate_instructions_callable,
    validate_realtime_handoff_options,
)
from .spec import RealtimeAgentSpec

if TYPE_CHECKING:
    from .protocols import RealtimeAgentBuilder


class RealtimeAgentRegistry:
    """RealtimeAgentSpec を宣言的に登録し、遅延構築を管理する。

    エージェントは初回 `get()` 時に局所 2 パス（パス 1: handoffs 空でビルド、パス 2: handoffs を
    後付け結線）で構築されるため、handoffs の循環を許容する。単一スレッド / 単一イベントループ
    前提（並行制御は利用者責任）。
    """

    def __init__(self, agent_builder: RealtimeAgentBuilder | None = None):
        """レジストリを生成する。

        Args:
            agent_builder: RealtimeAgent 構築の Protocol 実装。省略時は `_adapters` の
                デフォルト実装を関数内遅延生成で使う。テストでフェイクを注入できる。
        """
        self._agent_builder = agent_builder
        self._specs: dict[str, RealtimeAgentSpec] = {}
        self._built: dict[str, Any] = {}
        # 登録順を保持する（entry_name で登録順の先頭を引くため。names() の昇順とは別）。
        self._order: list[str] = []

    # ------------------------------------------------------------------
    # 登録
    # ------------------------------------------------------------------
    def register(self, spec: RealtimeAgentSpec) -> RealtimeAgentSpec:
        """RealtimeAgentSpec を登録する（ビルドは遅延）。

        Args:
            spec: 登録する RealtimeAgentSpec。

        Returns:
            登録した spec。

        Raises:
            ValueError: 名前重複、または callable instructions の引数数が不正な場合。
        """
        if spec.name in self._specs:
            raise ValueError(f"agent already registered: {spec.name}")
        self._validate_spec(spec)
        self._specs[spec.name] = spec
        self._order.append(spec.name)
        return spec

    def names(self) -> list[str]:
        """登録済みエージェント名を昇順で返す。"""
        return sorted(self._specs)

    @property
    def entry_name(self) -> str | None:
        """最初に登録されたエージェント名（エントリエージェント）を返す。

        Returns:
            登録順で最初のエージェント名。未登録なら None。
        """
        return self._order[0] if self._order else None

    # ------------------------------------------------------------------
    # 取得・遅延構築（局所 2 パス遅延バインド）
    # ------------------------------------------------------------------
    def get(self, name: str) -> Any:
        """エージェントを取得する。未構築なら到達可能 spec を局所 2 パスで構築する。

        Args:
            name: 取得するエージェント名。

        Returns:
            構築済みの RealtimeAgent。

        Raises:
            KeyError: 未登録名、または handoff 参照先が未登録の場合。
        """
        if name in self._built:
            return self._built[name]
        if name not in self._specs:
            raise KeyError(f"unknown agent: {name}")

        reachable = self._collect_reachable(name)
        # パス 1/2 はトランザクショナルに実行する。途中で例外が出たら本呼び出しで
        # 新規キャッシュした bare agent を巻き戻し、不完全なインスタンスを残さない。
        newly_built: list[str] = []
        try:
            # パス 1: handoffs 空でビルドして登録
            for target in reachable:
                if target not in self._built:
                    self._built[target] = self._builder().build(self._specs[target])
                    newly_built.append(target)
            # パス 2: handoffs を後付け結線
            for target in reachable:
                self._wire(self._specs[target], self._built[target])
        except Exception:
            for target in newly_built:
                self._built.pop(target, None)
            raise
        return self._built[name]

    def _collect_reachable(self, name: str) -> list[str]:
        """name から handoffs を辿り未ビルドの spec 名を集める（visited で循環を打ち切る）。"""
        collected: list[str] = []
        visited: set[str] = set()
        stack = [name]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if current not in self._specs:
                continue
            if current not in self._built:
                collected.append(current)
            for dep in self._specs[current].handoffs:
                if dep not in visited:
                    stack.append(dep)
        return collected

    def _wire(self, spec: RealtimeAgentSpec, agent: Any) -> None:
        """ビルド済み RealtimeAgent に handoffs を後付け結線する。"""
        for dst in spec.handoffs:
            target = self._require(dst, spec.name)
            config = spec.handoff_options.get(dst)
            agent.handoffs.append(self._builder().make_handoff(target, config))

    def _require(self, name: str, src: str) -> Any:
        try:
            return self.get(name)
        except KeyError as exc:
            raise KeyError(f"agent {src!r} の handoff 参照 {name!r} が未登録です") from exc

    # ------------------------------------------------------------------
    # 検証
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """全 spec の handoffs 参照が解決可能かを一括検証する。

        未登録名をすべて集約して報告する。run 前に呼ぶことでタイポ等の参照ミスを
        早期に検出できる（遅延構築の build 時エラーより前倒し）。

        Raises:
            KeyError: 解決できない参照が 1 つ以上ある場合（全件を列挙）。
        """
        known = set(self._specs)
        problems: list[str] = []
        for name, spec in self._specs.items():
            for dst in spec.handoffs:
                if dst not in known:
                    problems.append(f"{name!r} の handoff 参照 {dst!r} が未登録")
        if problems:
            raise KeyError("未解決のエージェント参照: " + "; ".join(problems))

    # ------------------------------------------------------------------
    # 内部ヘルパ
    # ------------------------------------------------------------------
    def _builder(self) -> RealtimeAgentBuilder:
        if self._agent_builder is None:
            # 遅延 import 境界（NFR-1）: `agents.realtime`（websockets を連鎖 import する）の
            # 読込を build 時まで遅らせるため、`_adapters` パッケージ __init__ の再エクスポート
            # 経由ではなく submodule を直接 import する（`import oai_agentspec` で extra を強制
            # ロードしない）。
            from .._adapters.realtime import DefaultRealtimeAgentBuilder

            self._agent_builder = DefaultRealtimeAgentBuilder()
        return self._agent_builder

    @staticmethod
    def _validate_spec(spec: RealtimeAgentSpec) -> None:
        """spec の register 時検証を行う。

        - callable instructions が (context, agent) の 2 引数で呼び出せること
        - prompt が callable（DynamicPromptFunction）でないこと（RealtimeAgent は
          静的 Prompt のみ対応のため。build 時の第二防御と同じ制約を register 時に前倒し）
        - handoff_options のキーが handoffs に存在すること（タイポによる per-edge 設定の
          silent drop を防ぐ）
        - handoff_options の `input_type` 指定時に `on_handoff` が伴うこと、および
          `on_handoff` の引数個数（input_type ありで 2・なしで 1。SDK `realtime_handoff()`
          の必須制約）。get() 時の文脈なし UserError を register 時のエージェント名・
          エッジ名入り ValueError に前倒しする
        """
        # SDK 側に instructions の引数検査はないため bind で呼び出し可能性のみ確認する
        # （on_handoff の厳格 len 検査とは前提が異なる）。検証本体は両宣言ルート共有の
        # _validation ヘルパに一元化している（handoff_options は RealtimeHandoffGraph.apply
        # とも共有し、apply -> register / register -> apply の順序に依らず同一規則で検証する）。
        validate_instructions_callable(spec.name, spec.instructions)
        ensure_static_prompt(spec.name, spec.prompt)
        validate_realtime_handoff_options(spec.name, spec.handoffs, spec.handoff_options)
