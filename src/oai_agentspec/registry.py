"""エージェントの登録・遅延構築・循環解決・ランタイム差し替えの中枢。

`agents` には依存せず（SDK 型は `TYPE_CHECKING` + `_adapters` 経由）、DI で注入された
`AgentBuilder` を用いて Agent を遅延構築する。循環ハンドオフは `get(name)` 起点・
到達可能 spec のみの局所 2 パス遅延バインドで解決する（詳細は docs/architecture.md）。
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final

from ._registry_core import build_two_pass, collect_reachable
from ._validation import (
    ensure_appendable_instructions,
    validate_instructions_append_shape,
    validate_instructions_callable,
)
from .spec import AgentSpec, HandoffConfig

if TYPE_CHECKING:
    from ._adapters import Agent, NextTurnWiring
    from .protocols import AgentBuilder, GuardrailProvider

AgentFactory = Callable[["AgentRegistry"], "Agent"]

#: Agent 実体へ振り分けられる guardrail 境界（`GuardrailProvider.boundary_of` の戻り値）。
_AGENT_BOUNDARIES: Final[frozenset[str]] = frozenset({"input", "output"})
#: ツール呼び出しに適用する境界（Agent 実体の guardrail フィールドへは振り分けられない）。
_TOOL_BOUNDARIES: Final[frozenset[str]] = frozenset({"tool_input", "tool_output"})


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

    def __init__(
        self,
        agent_builder: AgentBuilder | None = None,
        *,
        guardrail_registry: GuardrailProvider | None = None,
    ):
        """レジストリを生成する。

        Args:
            agent_builder: Agent 構築の Protocol 実装。省略時は `_adapters` の
                デフォルト実装を使う。テストでフェイクを注入できる。
            guardrail_registry: `AgentSpec.guardrails` の名前参照を解決する
                `GuardrailProvider` 実装（`runtime.guardrails` の `GuardrailRegistry` 等）。
                省略時は名前参照を解決できず、宣言があれば build / validate で報告する。
        """
        self._agent_builder = agent_builder
        self._guardrail_registry = guardrail_registry
        self._specs: dict[str, AgentSpec] = {}
        self._factories: dict[str, AgentFactory] = {}
        self._built: dict[str, Agent] = {}
        # 登録順を保持する（spec / factory をまたいだ通し順）。names() の昇順とは別に、
        # 「最初に登録されたエージェント」を entry_name で引けるようにするため。
        self._order: list[str] = []
        # freeze 後は変更操作（register / register_factory / update / unregister /
        # _update_handoffs）を遮断する。clone() で得た新 registry はこのフラグを引き継がない。
        self._frozen: bool = False
        # Next-Turn Agent Override（到達時ハンドオフ禁止）の結線一式（判定表 + 記録ストア）。
        # `_install_next_turn_state`（`apply_next_turn_policy` 専用の内部プリミティブ）を
        # 通した registry だけが値を持ち、既定は「合成なし＝従来経路と同一」。
        self._next_turn: NextTurnWiring | None = None

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
        Next-Turn Agent Override の結線一式（判定表 + 到達記録ストア）は共有継承する（記録は
        run 単位に分離されるため共有で安全・継承しないと clone 経由で禁止が静かに脱落する）。
        分離は SDK が run ごとに context wrapper を生成する前提で成立する。利用者が同一
        `RunContextWrapper` を複数 run へ渡した場合は記録が残るが、禁止が残る方向（安全側）。
        guardrail 名前参照の解決元（`GuardrailProvider`）は参照を共有継承する（登録簿は
        宣言の保持に徹し build 結果を汚さないため共有で安全・継承しないと clone 側で名前
        参照が解決不能になる）。

        評価（LLMOps）で「利用者 registry を一切汚さずに tools をモック化した派生 registry」を
        作るための宣言層プリミティブ。`transform_spec` には plain な `AgentSpec -> AgentSpec` を
        渡し、SDK 型操作（FunctionTool 差し替え等）は呼び出し側の `_adapters` ヘルパに委ねる。

        Args:
            transform_spec: 各 spec を登録前に変換する関数（任意）。None で素通し。

        Returns:
            独立した新 `AgentRegistry`。
        """
        cloned = AgentRegistry(self._agent_builder, guardrail_registry=self._guardrail_registry)
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
        # Next-Turn Agent Override の結線一式（判定表 + 到達記録ストア）は共有継承する
        # （記録は wrapper キーで run 分離されるため共有で安全。継承しないと clone 経由で
        # 到達時ハンドオフ禁止が静かに脱落する）。
        if self._next_turn is not None:
            cloned._install_next_turn_state(self._next_turn)
        return cloned

    def _install_next_turn_state(self, wiring: NextTurnWiring) -> None:
        """到達時ハンドオフ禁止の結線一式を設置する内部プリミティブ。

        ユーザーは `apply_next_turn_policy`（`next_turn.py`）経由で利用する。設置後は
        `_wire` / `_build_dynamic_handoff` が判定表に載るエッジにだけ合成を行う
        （判定表に載らないエッジは従来経路のまま）。

        Args:
            wiring: 判定表（流入エッジ集合 / ゲート対象名集合）と到達記録ストアの一式
                （`_adapters` の `NextTurnWiring`）。

        Raises:
            RegistryFrozenError: ``freeze()`` 後の呼び出し。
        """
        self._ensure_unfrozen("_install_next_turn_state")
        self._next_turn = wiring
        # 設置前に構築済みの Agent は合成前の結線を持つため破棄する（次回 get() で再構築）。
        # 現行の唯一の呼び出し元（`apply_next_turn_policy` -> clone 直後 / clone の継承）では
        # `_built` は常に空のため no-op だが、プリミティブ単体の整合性のため残す。
        self._built.clear()

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

        # 到達可能収集とトランザクショナルな 2 パス build/wire + 巻き戻しは共有 leaf
        # `_registry_core` に委譲する（差分点＝依存辺・bare ビルド・結線はコールバックで注入）。
        reachable = collect_reachable(name, self._specs, self._built, self._dependencies)
        build_two_pass(reachable, self._specs, self._built, self._build_bare, self._wire)
        return self._built[name]

    def _build_bare(self, spec: AgentSpec) -> Agent:
        return self._builder().build(spec)

    def _wire(self, spec: AgentSpec, agent: Agent) -> None:
        """ビルド済み Agent に handoffs / dynamic_handoffs / sub_agents / guardrails を結線する。

        `spec.guardrails` の名前参照は `GuardrailProvider` で実体へ解決し、宣言境界に応じて
        Agent の `input_guardrails` / `output_guardrails` へ append する（builder が専用
        フィールドのコピーを渡すため spec を汚さない）。append 順により連結順序は
        「専用フィールド -> 名前参照」になり、同名の重複宣言は排除しない。
        """
        from . import _adapters

        for dst in spec.handoffs:
            target = self._require(dst, spec.name, "handoff")
            # 判定表に載るエッジだけ合成済み config へ差し替える（載らないエッジは素通し＝
            # per-edge 設定が無ければ従来どおり Agent 実体の直 append を維持する）。
            config = self._next_turn_config(spec.name, dst, spec.handoff_options.get(dst))
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
        for index, gname in enumerate(spec.guardrails, start=1):
            guardrail, boundary = self._require_guardrail(spec.name, gname, position=index)
            if boundary == "input":
                agent.input_guardrails.append(guardrail)
            else:
                agent.output_guardrails.append(guardrail)

    def _next_turn_config(
        self, src: str, dst: str, config: HandoffConfig | None
    ) -> HandoffConfig | None:
        """静的エッジ 1 本に到達時ハンドオフ禁止の合成を適用した config を返す。

        判定表に載らないエッジでは引数の config をそのまま返すため、per-edge 設定を持た
        ないエッジは従来どおり Agent 実体の直 append 経路に残る。載るエッジでは
        （per-edge 設定が無くても）`HandoffConfig` を組んで `make_handoff` 経由の
        `Handoff` へ昇格させる（SDK は直 append 経路の handoff を毎ステップ内部生成する
        ため、合成の差し込み口が無い）。

        合成は元 config を書き換えず `dataclasses.replace` で新インスタンスを作る
        （config は元 registry の spec と共有されうるため）。

        記録を前置するエッジの利用者宣言は `_adapters` の
        `validate_recorded_edge_declaration` で build 時に検証する（SDK 固有の規則と例外型の
        知識は `_adapters` に閉じる。コア層は `agents` を直接参照しない）。

        Args:
            src: エッジの所有側（遷移元）エージェント名。
            dst: エッジの遷移先エージェント名。
            config: 利用者宣言の per-edge 設定（未宣言なら None）。

        Returns:
            合成済みの `HandoffConfig`。判定表に載らないエッジでは引数の config のまま。

        Raises:
            UserError: 記録を前置するエッジで `input_type` があるのに利用者宣言の
                `on_handoff` が無い場合（SDK の `handoff()` と同じ例外型）。
        """
        wiring = self._next_turn
        if wiring is None:
            return config
        record = (src, dst) in wiring.arrivals
        gate = src in wiring.gated
        if not record and not gate:
            return config

        from . import _adapters

        base = config if config is not None else HandoffConfig()
        changes: dict[str, Any] = {}
        if record:
            _adapters.validate_recorded_edge_declaration(src, dst, base)
            changes["on_handoff"] = _adapters.make_arrival_recorder(
                wiring.store, dst, base.on_handoff, base.input_type is not None
            )
        if gate:
            changes["is_enabled"] = _adapters.make_arrival_gate(wiring.store, src, base.is_enabled)
        return dataclasses.replace(base, **changes)

    def _record_next_turn_arrival(self, context: Any, src: str, dst: str) -> None:
        """動的エッジの到達を、判定表に載る `(src, dst)` のときだけ記録する。

        静的エッジの記録は `on_handoff` への前置合成で行うが、動的エッジは遷移先が実行時に
        決まるため、遷移先が確定する `on_invoke` の内側で記録する（到達の意味論は同一）。

        Args:
            context: SDK が渡す run のコンテキスト wrapper（記録キー）。
            src: エッジの所有側（遷移元）エージェント名。
            dst: resolver が選んだ遷移先エージェント名。
        """
        wiring = self._next_turn
        if wiring is None:
            return
        if (src, dst) in wiring.arrivals:
            wiring.store.record(context, dst)

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
            target = self._require(chosen, spec.name, "dynamic handoff")
            # 到達記録は遷移先が確定してから行う（利用者 `on_handoff` は make_dynamic_handoff
            # 側で on_invoke の後に発火するため、静的エッジと同じ「記録 -> 利用者」順になる）。
            self._record_next_turn_arrival(context, spec.name, chosen)
            return target

        # X の全出辺にゲートを AND 合成する（静的エッジと同一の判定表・同一の意味論）。
        is_enabled = dyn.is_enabled
        wiring = self._next_turn
        if wiring is not None and spec.name in wiring.gated:
            is_enabled = _adapters.make_arrival_gate(wiring.store, spec.name, dyn.is_enabled)

        return _adapters.make_dynamic_handoff(
            tool_name=dyn.tool_name,
            description=dyn.description,
            on_invoke=on_invoke,
            on_handoff=dyn.on_handoff,
            input_type=dyn.input_type,
            input_filter=dyn.input_filter,
            is_enabled=is_enabled,
            options=dyn.options,
        )

    def _require(self, name: str, src: str, kind: str) -> Agent:
        try:
            return self.get(name)
        except KeyError as exc:
            raise KeyError(f"agent {src!r} の {kind} 参照 {name!r} が未登録です") from exc

    # ------------------------------------------------------------------
    # guardrail 名前参照の解決（build 用）と問題行の文言（build / validate 共有）
    # ------------------------------------------------------------------
    @staticmethod
    def _guardrail_problem(
        src: str, name: Any, kind: str, detail: str = "", *, position: int | None = None
    ) -> str:
        """guardrail 名前参照の問題 1 件を表す 1 行を組み立てる（純関数・raise しない）。

        build 時（`_require_guardrail`）と `validate()` 時（`_collect_guardrail_problems`）で
        同一文言を使うため、書式をここ 1 箇所に寄せる。

        Args:
            src: 参照を宣言しているエージェント名。
            name: 宣言された参照値（非 str の場合もそのまま受け取る）。
            kind: 問題の種類（`"non_str"` / `"uninjected"` / `"unknown"` /
                `"tool_boundary"` / `"boundary"` / `"mismatch"` の 6 種）。
            position: 宣言リスト内の位置（1 始まり）。`"non_str"` の問題行で要素を区別する
                ために使う（同一型の不正要素が複数あると型名だけでは同一文言になる）。
            detail: 種類ごとの補足（`"non_str"` は型名、`"tool_boundary"` / `"boundary"` は
                境界値の文字列、`"mismatch"` は `申告=… , 実体=…` 形式の突合結果）。

        Raises:
            ValueError: `kind` が上記 6 種のいずれでもない場合（内部不整合の early fail）。

        Returns:
            問題 1 件を表す 1 行の文字列（末尾に区切り文字を含まない）。
        """
        ref = f"{src!r} の guardrail 参照"
        # `boundary_of` は `str` 契約だが `str` 併用 Enum のメンバが渡されうる。f-string は
        # その場合 Enum の `__str__`（`Boundary.TOOL_OUTPUT` 形式）を経由するため、宣言された
        # 境界値そのもの（`tool_output`）になるよう str 側の変換を明示する。
        text = str.__str__(detail) if isinstance(detail, str) else str(detail)
        if kind == "non_str":
            where = "" if position is None else f"の {position} 番目"
            return f"{ref}{where}に str 以外の要素が含まれます（型: {text}）"
        if kind == "uninjected":
            return (
                f"{ref} {name!r} を解決できません"
                "（AgentRegistry(guardrail_registry=...) で注入してください）"
            )
        if kind == "unknown":
            return f"{ref} {name!r} が未登録"
        if kind == "tool_boundary":
            return f"{ref} {name!r} はツール境界（{text}）のため Agent へ振り分けできません"
        if kind == "boundary":
            return (
                f"{ref} {name!r} の境界 {text!r} は 'input' / 'output' のいずれでもないため "
                "Agent へ振り分けできません"
            )
        if kind == "mismatch":
            return (
                f"{ref} {name!r} は解決元が申告した境界と実体の境界が一致しません（{text}）。"
                "guardrail_registry の boundary_of と get の整合を確認してください"
            )
        raise ValueError(f"unknown guardrail problem kind: {kind!r}")

    def _require_guardrail(
        self, src: str, name: Any, *, position: int | None = None
    ) -> tuple[Any, str]:
        """guardrail 名前参照 1 件を実体と境界へ解決する（build 用・解決できなければ raise）。

        検証順は「非 str -> provider 未注入 -> 名前解決 -> 境界の受け入れ可否」。`boundary_of`
        が agent 境界（`"input"` / `"output"`）以外を返した場合は既定境界へフォールバックせず
        `ValueError` にする（ツール境界・値域外の文字列いずれも Agent へ振り分けできない）。

        Args:
            src: 参照を宣言しているエージェント名。
            name: 宣言された参照値。

        Returns:
            `(guardrail 実体, 境界)`。境界は `"input"` / `"output"` と比較できる文字列。

        Raises:
            ValueError: 参照が str でない、provider が未注入、または境界が agent 境界でない場合。
            KeyError: provider が未登録名として `KeyError` を上げた場合（包まずそのまま伝播）。
        """
        if not isinstance(name, str):
            raise ValueError(
                self._guardrail_problem(
                    src, name, "non_str", type(name).__name__, position=position
                )
            )
        provider = self._guardrail_registry
        if provider is None:
            raise ValueError(self._guardrail_problem(src, name, "uninjected"))
        guardrail = provider.get(name)
        boundary = provider.boundary_of(name)
        # `boundary_of` は duck-typed なので str 以外（list 等の unhashable も）を返しうる。
        # frozenset のメンバ検査に素で通すと生の `TypeError` になるため、先に値域外へ寄せる。
        if not isinstance(boundary, str):
            raise ValueError(self._guardrail_problem(src, name, "boundary", repr(boundary)))
        if boundary not in _AGENT_BOUNDARIES:
            kind = "tool_boundary" if boundary in _TOOL_BOUNDARIES else "boundary"
            raise ValueError(self._guardrail_problem(src, name, kind, boundary))
        # 自作 provider は境界を偽れる（Protocol は duck-typed で値域も検証しない）。cross-check が
        # ないと build は通り、run 開始時に SDK 内部の `AttributeError` になって利用者が原因へ
        # 辿り着けない。実体側の境界判定は `_adapters` に閉じているので関数内遅延 import で引く。
        from . import _adapters

        actual = _adapters.guardrail_boundary(guardrail)
        # 上流 4 型と判定できたときのみ突き合わせる。`GuardrailProvider` の契約は「不透明型を
        # 返す」であり、SDK 実体でない値（テスト用フェイク等）を返す provider も適合するため、
        # 判定不能（`None`）を不一致として扱うと Protocol の契約を実質狭めてしまう。
        if actual is not None and actual != boundary:
            raise ValueError(
                self._guardrail_problem(src, name, "mismatch", f"申告={boundary}, 実体={actual}")
            )
        return guardrail, boundary

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

        guardrail 名前参照の**解決元は凍結対象外**である。``guardrail_registry`` は参照を共有
        したまま保持され、凍結が保証するのは spec の独立性で、外部 provider の内容までは凍結
        しない。ただし解決は build 時にしか走らないため、provider 側の登録変更が反映されるのは
        「本メソッドが ``_built`` を invalidate した後の初回 ``get()``」の 1 回だけである。以降は
        構築済み Agent がキャッシュから返り、凍結後は ``update`` / ``unregister`` も遮断される
        ため invalidate 経路が無い（frozen registry 上で guardrail を差し替えることはできない）。

        ``get`` / ``validate`` / ``entry_name`` / ``names`` 等の read-only API は影響を
        受けない。本メソッドは冪等で、複数回呼んでも 2 回目以降は no-op として成功する
        （snapshot は 1 回目のみ実行する。2 回目以降は外部 mutation が既にコピーに反映
        されない状態のため再 snapshot 不要）。``clone()`` で得られた新 registry は本フラグを
        引き継がず unfrozen 状態で返る。
        """
        if self._frozen:
            return
        # 外部からの spec mutation を遮断するため独立コピーに置き換える（_copy_spec は
        # spec の全 dataclass フィールドを走査し list/dict 値を新コンテナにコピーする。
        # サブクラスの可変フィールド（SandboxAgentSpec.capabilities 等）も対象）。
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

    def _collect_guardrail_problems(self) -> list[str]:
        """全 spec の guardrail 名前参照を走査し、問題行を宣言順に集める（raise しない）。

        文言は `_guardrail_problem` に単一ソース化する。`boundary_of` が値域外の文字列を返す
        ケースは集約対象に含めない（provider の値域は登録簿側の責務で、`validate()` は成功しうる。
        この不整合は build 時に `_require_guardrail` が `ValueError` で落とす）。ツール境界は
        「宣言としては正しいが Agent へ振り分けられない」不整合なので集約対象に含める。

        Returns:
            問題行のリスト（spec 登録順・spec 内は `guardrails` の宣言順。問題なしは空リスト）。
        """
        problems: list[str] = []
        for src, spec in self._specs.items():
            for index, name in enumerate(spec.guardrails, start=1):
                if not isinstance(name, str):
                    problems.append(
                        self._guardrail_problem(
                            src, name, "non_str", type(name).__name__, position=index
                        )
                    )
                    continue
                provider = self._guardrail_registry
                if provider is None:
                    problems.append(self._guardrail_problem(src, name, "uninjected"))
                    continue
                try:
                    # `get` の戻り値は捨てるが呼び出しは必要。`boundary_of` 単体でも未登録は
                    # `KeyError` になる契約なので、両方を照会するのは「片側だけ解決できる
                    # 非対称な自作 provider」を弾くため（削除すると実体不在の検出が失われる）。
                    provider.get(name)
                    boundary = provider.boundary_of(name)
                except KeyError:
                    problems.append(self._guardrail_problem(src, name, "unknown"))
                    continue
                if isinstance(boundary, str) and boundary in _TOOL_BOUNDARIES:
                    problems.append(self._guardrail_problem(src, name, "tool_boundary", boundary))
        return problems

    def validate(self) -> None:
        """全 spec の handoffs / sub_agents / guardrails 参照が解決可能かを一括検証する。

        未登録名をすべて集約して報告する。run 前に呼ぶことでタイポ等の参照ミスを
        早期に検出できる（遅延構築の build 時エラーより前倒し）。

        報告はエージェント参照と guardrail 参照の 2 群を接頭辞（`未解決のエージェント参照: ` /
        `guardrail 参照の問題: `）で分け、単一の `KeyError` へ集約する。群が 2 つ揃うときだけ
        `" | "` で連結するため、`guardrails` 未宣言の構成では従来と同一の文言になる。
        guardrail 側の集約対象は `_collect_guardrail_problems` の契約に従う（`boundary_of` の
        値域外は build 時の `ValueError` に委ねる）。

        Raises:
            KeyError: 解決できない参照・Agent へ振り分けできない guardrail 参照が 1 つ以上ある
                場合（全件を列挙）。
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
        messages: list[str] = []
        if problems:
            messages.append("未解決のエージェント参照: " + "; ".join(problems))
        guardrail_problems = self._collect_guardrail_problems()
        if guardrail_problems:
            messages.append("guardrail 参照の問題: " + "; ".join(guardrail_problems))
        if messages:
            raise KeyError(" | ".join(messages))

    @staticmethod
    def _validate_spec(spec: AgentSpec) -> None:
        """callable instructions / 追記関数の arity と `guardrails` のコンテナ型を検証する。

        `guardrails` に素の `str` を渡す取り違え（`["a"]` と `"a"`）は、`str` が反復可能なため
        1 文字ずつ名前参照として解釈され、`validate()` が偽の「未登録」を大量に報告する。
        `GuardrailRegistry.run_config_kwargs` と同じ守りを宣言側にも置く。

        `instructions_append` の容器型・要素型（素の `str` / `Sequence` 以外の容器 / 非 callable
        要素の拒否）は `validate_instructions_append_shape` へ委譲する（`build_agent` 側の第二
        防御と判定・文言を共有するため）。本メソッドはさらに callable `instructions` との併用
        拒否（`ensure_appendable_instructions`）と、各要素が (context, agent) の 2 引数で
        呼び出せることを `instructions_append[i]` ラベル付きで検証する。追記関数は本検証で
        評価しない（評価は run ごとの SDK 側責務）。

        Args:
            spec: 検証対象の `AgentSpec`。

        Raises:
            ValueError: callable instructions / 追記要素の arity が不正な場合、callable
                `instructions` に `instructions_append` を併用した場合、`instructions_append`
                が素の `str`・`Sequence` 以外の容器・非 callable 要素を含む場合、または
                `guardrails` が素の `str` の場合。
        """
        validate_instructions_callable(spec.name, spec.instructions)
        validate_instructions_append_shape(spec.name, spec.instructions_append)
        ensure_appendable_instructions(spec.name, spec.instructions, spec.instructions_append)
        for i, fn in enumerate(spec.instructions_append):
            validate_instructions_callable(spec.name, fn, field_label=f"instructions_append[{i}]")
        if isinstance(spec.guardrails, str):
            raise ValueError(
                f"agent {spec.name!r} の guardrails must be a sequence of str, "
                f"not a bare str: {spec.guardrails!r}"
            )


def _copy_spec(spec: AgentSpec) -> AgentSpec:
    """`AgentSpec` を独立コピーする（新オブジェクト + 可変コンテナを新 list/dict に複製）。

    `clone` / `freeze` が保持する spec を元 registry と identity / 可変コンテナ共有しない
    ようにするためのヘルパ。spec の全 dataclass フィールドを走査し、値が list / dict の
    フィールドを全て新インスタンスに浅くコピーする（中身の要素 = FunctionTool / guardrail /
    handoff 名 / DynamicHandoff 等は共有でよい。`apply` 等が触るのはコンテナと spec
    オブジェクト自体のため）。スカラー / 不変フィールド（`name` / `instructions` / `model` 等）は
    そのまま引き継ぐ。フィールド列挙を宣言（`dataclasses.fields`）から導出するため、
    `AgentSpec` のサブクラス（`SandboxAgentSpec` の `capabilities` 等）の可変コンテナも
    列挙の手動同期なしで複製対象になる。

    Args:
        spec: コピー元の `AgentSpec`（サブクラス可。戻り値は同一クラス）。

    Returns:
        可変コンテナを共有しない独立した `AgentSpec`。
    """
    copies: dict[str, Any] = {}
    for f in dataclasses.fields(spec):
        if not f.init:
            continue
        value = getattr(spec, f.name)
        if isinstance(value, list):
            copies[f.name] = list(value)
        elif isinstance(value, dict):
            copies[f.name] = dict(value)
    return dataclasses.replace(spec, **copies)
