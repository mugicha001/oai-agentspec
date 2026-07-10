"""エージェントの登録・遅延構築・循環解決・ランタイム差し替えの中枢。

`agents` には依存せず（SDK 型は `TYPE_CHECKING` + `_adapters` 経由）、DI で注入された
`AgentBuilder` を用いて Agent を遅延構築する。循環ハンドオフは `get(name)` 起点・
到達可能 spec のみの局所 2 パス遅延バインドで解決する（詳細は docs/architecture.md）。
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ._validation import validate_instructions_callable
from .spec import AgentSpec

if TYPE_CHECKING:
    from ._adapters import Agent
    from .protocols import AgentBuilder
    from .spec import HandoffConfig

AgentFactory = Callable[["AgentRegistry"], "Agent"]


class RegistryFrozenError(RuntimeError):
    """凍結後の ``AgentRegistry`` に対する変更操作で raise される例外。

    ``IntegrityError`` 系統とは別で ``RuntimeError`` を継承するため、利用者の
    ``except IntegrityError`` で握り潰されない。例外メッセージは違反操作名を含む。
    """


class AgentRegistry:
    """Agent を宣言的に登録し、遅延構築・差し替えを管理する。

    エージェントは初回 `get()` 時に局所 2 パスで構築されるため、handoffs / sub_agents の
    循環を許容する。単一スレッド / 単一イベントループ前提（並行制御は利用者責任）。
    """

    def __init__(self, agent_builder: AgentBuilder | None = None):
        """レジストリを生成する。

        Args:
            agent_builder: Agent 構築の Protocol 実装。省略時は `_adapters` の
                デフォルト実装を使う。テストでフェイクを注入できる。
        """
        self._agent_builder = agent_builder
        self._specs: dict[str, AgentSpec] = {}
        self._factories: dict[str, AgentFactory] = {}
        self._built: dict[str, Agent] = {}
        # 登録順を保持する（spec / factory をまたいだ通し順）。names() の昇順とは別に、
        # 「最初に登録されたエージェント」を entry_name で引けるようにするため。
        self._order: list[str] = []
        # freeze 後は変更操作（register / register_factory / update / unregister /
        # _update_handoffs）を遮断する。clone() で得た新 registry はこのフラグを引き継がない。
        self._frozen: bool = False

    # ------------------------------------------------------------------
    # 登録
    # ------------------------------------------------------------------
    def register(self, spec: AgentSpec) -> AgentSpec:
        """AgentSpec を登録する（ビルドは遅延）。

        Args:
            spec: 登録する AgentSpec。

        Returns:
            登録した spec。

        Raises:
            ValueError: 名前重複、または指示系フィールドの組み合わせが不正な場合。
            RegistryFrozenError: ``freeze()`` 後の呼び出し。
        """
        self._ensure_unfrozen("register")
        if spec.name in self._specs or spec.name in self._factories:
            raise ValueError(f"agent already registered: {spec.name}")
        self._validate_spec(spec)
        self._specs[spec.name] = spec
        self._order.append(spec.name)
        return spec

    def register_factory(self, name: str, factory: AgentFactory) -> None:
        """ファクトリ関数で構築する Agent を登録する。

        Raises:
            ValueError: 名前が既に登録済みの場合。
            RegistryFrozenError: ``freeze()`` 後の呼び出し。
        """
        self._ensure_unfrozen("register_factory")
        if name in self._specs or name in self._factories:
            raise ValueError(f"agent already registered: {name}")
        self._factories[name] = factory
        self._order.append(name)

    def names(self) -> list[str]:
        """登録済みエージェント名を昇順で返す。"""
        return sorted({*self._specs, *self._factories})

    def clone(
        self, *, transform_spec: Callable[[AgentSpec], AgentSpec] | None = None
    ) -> AgentRegistry:
        """登録内容を引き継いだ独立した新 registry を返す（元 registry は不変）。

        spec ベースの登録は `transform_spec`（任意）で各 spec を変換してから新 registry へ
        登録する（未指定なら元 spec をそのまま再登録）。**登録する spec は必ず独立コピー**にする
        （新 `AgentSpec` オブジェクト + 可変コンテナ list/dict を新インスタンスに複製）。これにより
        identity 共有・可変コンテナの参照共有が無くなり、クローン側で `apply`（`spec.handoffs` の
        書き換え）等を行っても元 registry の spec を汚さない（`AgentSpec` はミュータブル
        dataclass のため・Codex P2）。list/dict の**中身**（FunctionTool / handoff 名等）は共有で
        よい（apply が触るのはコンテナと spec オブジェクト自体）。factory 登録は spec 実体を持た
        ないため変換・コピー対象外で、ファクトリ関数をそのまま引き継ぐ。登録順（`entry_name` の
        基準）と builder は引き継ぐ。`_built` キャッシュは引き継がない（新 registry で再構築）。

        評価（LLMOps）で「利用者 registry を一切汚さずに tools をモック化した派生 registry」を
        作るための宣言層プリミティブ。`transform_spec` には plain な `AgentSpec -> AgentSpec` を
        渡し、SDK 型操作（FunctionTool 差し替え等）は呼び出し側の `_adapters` ヘルパに委ねる。

        Args:
            transform_spec: 各 spec を登録前に変換する関数（任意）。None で素通し。

        Returns:
            独立した新 `AgentRegistry`。
        """
        cloned = AgentRegistry(self._agent_builder)
        for name in self._order:
            if name in self._specs:
                # 先に独立コピーを作って transform へ渡す。transform が mutate した結果を
                # そのまま register に渡しても、元 registry の spec は無傷（既に copy 済み）。
                # transform が新しい AgentSpec を返した場合も同様（元 spec は触られない）。
                spec = _copy_spec(self._specs[name])
                if transform_spec is not None:
                    spec = transform_spec(spec)
                cloned.register(spec)
            else:
                cloned.register_factory(name, self._factories[name])
        return cloned

    @property
    def entry_name(self) -> str | None:
        """最初に登録されたエージェント名（エントリエージェント）を返す。

        spec / factory をまたいだ登録順の先頭を返す。会話 CLI が「エントリ起点」で
        会話を始めるための既定エージェントに使う。1 つも登録が無ければ None。

        Returns:
            登録順で最初のエージェント名。未登録なら None。
        """
        return self._order[0] if self._order else None

    # ------------------------------------------------------------------
    # 取得・遅延構築（局所 2 パス遅延バインド）
    # ------------------------------------------------------------------
    def get(self, name: str) -> Agent:
        """エージェントを取得する。未構築なら到達可能 spec を局所 2 パスで構築する。

        Raises:
            KeyError: 未登録名の場合。
        """
        if name in self._built:
            return self._built[name]
        if name in self._factories:
            agent = self._factories[name](self)
            self._built[name] = agent
            return agent
        if name not in self._specs:
            raise KeyError(f"unknown agent: {name}")

        reachable = self._collect_reachable(name)
        # パス 1/2 はトランザクショナルに実行する。途中で例外が出たら本呼び出しで
        # 新規キャッシュした bare agent を巻き戻し、不完全なインスタンスを残さない。
        newly_built: list[str] = []
        try:
            # パス 1: handoffs 空・サブツール未注入でビルドして登録
            for target in reachable:
                if target not in self._built:
                    self._built[target] = self._build_bare(self._specs[target])
                    newly_built.append(target)
            # パス 2: handoffs / sub_agents を後付け結線
            for target in reachable:
                self._wire(self._specs[target], self._built[target])
        except Exception:
            for target in newly_built:
                self._built.pop(target, None)
            raise
        return self._built[name]

    def _collect_reachable(self, name: str) -> list[str]:
        """name から依存辺（handoffs ∪ sub_agents）を辿り未ビルドの spec 名を集める。

        visited 集合で循環を打ち切る。到達不能 spec は含めない。spec でない依存名
        （factory / 未登録）は収集対象外（factory は get() 時に自前構築、未登録は
        結線フェーズでエラーになる）。
        """
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
            for dep in self._dependencies(self._specs[current]):
                if dep not in visited:
                    stack.append(dep)
        return collected

    def _build_bare(self, spec: AgentSpec) -> Agent:
        return self._builder().build(spec)

    def _wire(self, spec: AgentSpec, agent: Agent) -> None:
        """ビルド済み Agent に handoffs / dynamic_handoffs / sub_agents を結線する。"""
        from . import _adapters

        for dst in spec.handoffs:
            target = self._require(dst, spec.name, "handoff")
            config = spec.handoff_options.get(dst)
            if config is not None:
                agent.handoffs.append(_adapters.make_handoff(target, config))
            else:
                agent.handoffs.append(target)
        for dyn in spec.dynamic_handoffs:
            agent.handoffs.append(self._build_dynamic_handoff(spec, dyn))
        for sub in spec.sub_agents:
            sub_agent = self._require(sub, spec.name, "sub_agent")
            tool_name, tool_description = spec.sub_agent_tools.get(sub, (None, None))
            agent.tools.append(
                _adapters.make_agent_tool(
                    sub_agent, tool_name=tool_name, tool_description=tool_description
                )
            )

    def _build_dynamic_handoff(self, spec: AgentSpec, dyn: Any) -> Any:
        """DynamicHandoff から、候補名を解決する on_invoke_handoff 付き Handoff を作る。

        resolver の戻り名は candidates 内に限る（外れたら実行時 ValueError）。転送先 Agent
        は registry から名前解決する（registry をクロージャに閉じ込める）。
        """
        from . import _adapters

        candidates = set(dyn.candidates)

        async def on_invoke(context: Any, input_json: Any = None) -> Any:
            chosen = dyn.resolver(context, input_json)
            if inspect.isawaitable(chosen):
                chosen = await chosen
            if chosen not in candidates:
                raise ValueError(
                    f"agent {spec.name!r} の dynamic handoff resolver が候補外の名前を"
                    f"返しました: {chosen!r}（候補: {sorted(candidates)}）"
                )
            return self._require(chosen, spec.name, "dynamic handoff")

        return _adapters.make_dynamic_handoff(
            tool_name=dyn.tool_name,
            description=dyn.description,
            on_invoke=on_invoke,
            on_handoff=dyn.on_handoff,
            input_type=dyn.input_type,
            input_filter=dyn.input_filter,
            is_enabled=dyn.is_enabled,
            options=dyn.options,
        )

    def _require(self, name: str, src: str, kind: str) -> Agent:
        try:
            return self.get(name)
        except KeyError as exc:
            raise KeyError(f"agent {src!r} の {kind} 参照 {name!r} が未登録です") from exc

    # ------------------------------------------------------------------
    # ランタイム差し替え
    # ------------------------------------------------------------------
    def update(self, spec: AgentSpec) -> None:
        """同名 spec を置換し、当該 Agent と依存元を連鎖 invalidate する。

        次回 `get()` から反映され、進行中の run には影響しない。

        Raises:
            KeyError: 未登録名の場合。
            ValueError: 指示系フィールドの組み合わせが不正な場合。
            RegistryFrozenError: ``freeze()`` 後の呼び出し。
        """
        self._ensure_unfrozen("update")
        if spec.name not in self._specs:
            raise KeyError(f"unknown agent: {spec.name}")
        self._validate_spec(spec)
        self._specs[spec.name] = spec
        self._invalidate(spec.name)

    def unregister(self, name: str) -> None:
        """spec と built を削除し、依存元を連鎖 invalidate する。

        Raises:
            KeyError: 未登録名の場合。
            RegistryFrozenError: ``freeze()`` 後の呼び出し。
        """
        self._ensure_unfrozen("unregister")
        if name not in self._specs:
            raise KeyError(f"unknown agent: {name}")
        dependents = self._dependents(name)
        del self._specs[name]
        self._built.pop(name, None)
        if name in self._order:
            self._order.remove(name)
        for dep in dependents:
            self._built.pop(dep, None)

    def _update_handoffs(
        self,
        name: str,
        handoffs: list[str],
        *,
        mode: str = "replace",
        handoff_options: dict[str, HandoffConfig] | None = None,
        dynamic_handoffs: list[Any] | None = None,
    ) -> None:
        """spec のハンドオフを更新し、当該 Agent と依存元を invalidate する内部プリミティブ。

        ユーザーは `HandoffGraph`（`edge` / `dynamic_edge` + `apply`）経由で利用する。
        `apply()` が本メソッドへ委譲する（registry の生の内部状態には触らせない）。

        Args:
            name: 対象エージェント名。
            handoffs: ハンドオフ先名リスト。
            mode: "replace"（既存を置換）または "append"（追記 + 重複排除）。
            handoff_options: dst 名 -> HandoffConfig の per-edge 設定。
            dynamic_handoffs: DynamicHandoff のリスト（replace 時のみ反映）。

        Raises:
            KeyError: 未登録名（または spec ベースでない）場合。
            ValueError: mode が不正な場合。
            RegistryFrozenError: ``freeze()`` 後の呼び出し。
                ``HandoffGraph.apply`` は唯一本メソッドを経由するため、freeze 後の apply
                経路もここで遮断される。
        """
        self._ensure_unfrozen("_update_handoffs")
        if name not in self._specs:
            raise KeyError(f"unknown spec-based agent: {name}")
        if mode not in ("replace", "append"):
            raise ValueError(f"invalid mode: {mode!r}")
        spec = self._specs[name]
        if mode == "replace":
            spec.handoffs = list(handoffs)
            spec.handoff_options = dict(handoff_options or {})
            spec.dynamic_handoffs = list(dynamic_handoffs or [])
        else:
            for dst in handoffs:
                if dst not in spec.handoffs:
                    spec.handoffs.append(dst)
            if handoff_options:
                spec.handoff_options.update(handoff_options)
            if dynamic_handoffs:
                spec.dynamic_handoffs.extend(dynamic_handoffs)
        self._invalidate(name)

    # ------------------------------------------------------------------
    # 凍結
    # ------------------------------------------------------------------
    def freeze(self) -> None:
        """以降の登録・更新・削除・内部ハンドオフ書き換えを禁止する。

        ``register`` / ``register_factory`` / ``update`` / ``unregister`` /
        ``_update_handoffs`` の各経路で ``RegistryFrozenError`` を raise するようになる。
        ``HandoffGraph.apply`` は ``_update_handoffs`` 経由で書き込むため、apply 経路も
        本メソッド 1 つで遮断される。

        freeze 時点で登録済 ``AgentSpec`` を独立コピー（``_copy_spec`` で可変コンテナを
        新インスタンスに複製）に置き換え、外部参照経由の spec mutation（``spec.instructions
        = ...`` / ``spec.handoffs.append(...)`` / ``spec.tools.append(...)`` 等）が
        registry の build 結果に伝播しないようにする。コピーに伴い ``_built`` キャッシュも
        invalidate し、次回 ``get()`` でコピー後の spec から再構築する。

        ``get`` / ``validate`` / ``entry_name`` / ``names`` 等の read-only API は影響を
        受けない。本メソッドは冪等で、複数回呼んでも 2 回目以降は no-op として成功する
        （snapshot は 1 回目のみ実行する。2 回目以降は外部 mutation が既にコピーに反映
        されない状態のため再 snapshot 不要）。``clone()`` で得られた新 registry は本フラグを
        引き継がず unfrozen 状態で返る。
        """
        if self._frozen:
            return
        # 外部からの spec mutation を遮断するため独立コピーに置き換える（_copy_spec は
        # tools / handoffs / handoff_options / sub_agents / sub_agent_tools / dynamic_handoffs
        # / input_guardrails / output_guardrails / extra を新 list/dict にコピーする）。
        self._specs = {name: _copy_spec(spec) for name, spec in self._specs.items()}
        # コピー前の spec から組まれた Agent は無効化（次回 get() で snapshot から再構築）。
        self._built.clear()
        self._frozen = True

    def _ensure_unfrozen(self, operation: str) -> None:
        """frozen registry に対する変更操作なら ``RegistryFrozenError`` を raise する。

        Args:
            operation: 違反した変更操作名（``register`` / ``_update_handoffs`` 等）。

        Raises:
            RegistryFrozenError: ``freeze()`` 後の場合。
        """
        if self._frozen:
            raise RegistryFrozenError(f"frozen registry に対する変更操作: {operation}")

    def _invalidate(self, name: str) -> None:
        """name と、name に（推移的に）依存する全 Agent の built を破棄する。"""
        for target in {name, *self._dependents(name)}:
            self._built.pop(target, None)

    def _dependents(self, name: str) -> set[str]:
        """name を依存辺（handoffs ∪ sub_agents）に持つ spec 名を推移的に集める。

        spec を真実源として逆引きを導出する（visited で循環を打ち切る）。
        """
        result: set[str] = set()
        stack = [name]
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for spec_name, spec in self._specs.items():
                if current in self._dependencies(spec) and spec_name not in result:
                    result.add(spec_name)
                    stack.append(spec_name)
        return result

    # ------------------------------------------------------------------
    # 内部ヘルパ
    # ------------------------------------------------------------------
    @staticmethod
    def _dependencies(spec: AgentSpec) -> list[str]:
        """ビルド依存辺（handoffs ∪ sub_agents ∪ dynamic_handoffs の候補）。"""
        dynamic = [c for dyn in spec.dynamic_handoffs for c in dyn.candidates]
        return [*spec.handoffs, *spec.sub_agents, *dynamic]

    def _builder(self) -> AgentBuilder:
        if self._agent_builder is None:
            from ._adapters import DefaultAgentBuilder

            self._agent_builder = DefaultAgentBuilder()
        return self._agent_builder

    def validate(self) -> None:
        """全 spec の handoffs / sub_agents 参照が解決可能かを一括検証する。

        未登録名をすべて集約して報告する。run 前に呼ぶことでタイポ等の参照ミスを
        早期に検出できる（遅延構築の build 時エラーより前倒し）。

        Raises:
            KeyError: 解決できない参照が 1 つ以上ある場合（全件を列挙）。
        """
        known = {*self._specs, *self._factories}
        problems: list[str] = []
        for name, spec in self._specs.items():
            for dst in spec.handoffs:
                if dst not in known:
                    problems.append(f"{name!r} の handoff 参照 {dst!r} が未登録")
            for sub in spec.sub_agents:
                if sub not in known:
                    problems.append(f"{name!r} の sub_agent 参照 {sub!r} が未登録")
            for dyn in spec.dynamic_handoffs:
                for cand in dyn.candidates:
                    if cand not in known:
                        problems.append(
                            f"{name!r} の dynamic handoff {dyn.tool_name!r} の候補 "
                            f"{cand!r} が未登録"
                        )
        if problems:
            raise KeyError("未解決のエージェント参照: " + "; ".join(problems))

    @staticmethod
    def _validate_spec(spec: AgentSpec) -> None:
        """callable instructions が (context, agent) の 2 引数で呼び出せることを検証する。"""
        validate_instructions_callable(spec.name, spec.instructions)


def _copy_spec(spec: AgentSpec) -> AgentSpec:
    """`AgentSpec` を独立コピーする（新オブジェクト + 可変コンテナを新 list/dict に複製）。

    `clone` が登録する spec を元 registry と identity / 可変コンテナ共有しないようにするための
    ヘルパ。`AgentSpec` のミュータブル list/dict フィールド（`tools` / `input_guardrails` /
    `output_guardrails` / `handoffs` / `handoff_options` / `sub_agents` / `sub_agent_tools` /
    `dynamic_handoffs` / `extra`）を全て新インスタンスに浅くコピーする（中身の要素 = FunctionTool /
    guardrail / handoff 名 / DynamicHandoff 等は共有でよい。`apply` 等が触るのはコンテナと spec
    オブジェクト自体のため）。スカラー / 不変フィールド（`name` / `instructions` / `model` 等）は
    そのまま引き継ぐ。

    Args:
        spec: コピー元の `AgentSpec`。

    Returns:
        可変コンテナを共有しない独立した `AgentSpec`。
    """
    return dataclasses.replace(
        spec,
        tools=list(spec.tools),
        input_guardrails=list(spec.input_guardrails),
        output_guardrails=list(spec.output_guardrails),
        handoffs=list(spec.handoffs),
        handoff_options=dict(spec.handoff_options),
        sub_agents=list(spec.sub_agents),
        sub_agent_tools=dict(spec.sub_agent_tools),
        dynamic_handoffs=list(spec.dynamic_handoffs),
        extra=dict(spec.extra),
    )
