"""`GovernedAgentBuilder`（AGT ガバナンスの装飾 builder）。

`AgentBuilder` Protocol（`build(spec) -> Agent`）を満たす別実装で、`inner`（既定は `_adapters` の
`DefaultAgentBuilder`）を装飾する。build 時に各ツールを govern ラップし、監査 `AgentHooks` を装着し
た新 `AgentSpec` を `inner.build` へ渡す（ポリシー評価・監査記録は実行時に AGT 側で動く・
build-don't-run）。

`policy` / `audit_sink` は本層では不透明値として保持し、評価・読込・SDK/AGT 結合は
`_adapters.governance` へ委譲する（本層は `agents` / AGT を実 import せず plain 値・不透明型のみ扱
う・NFR-1）。`_adapters` への import は関数内遅延（extra 未導入耐性・AGT は `govern_spec` 内で初めて
遅延 import される）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os
    from collections.abc import Mapping

    from ..._adapters import Agent
    from ...protocols import AgentBuilder
    from ...spec import AgentSpec


class GovernedAgentBuilder:
    """`AgentBuilder` を満たす装飾 builder。tools を govern ラップし監査 / 強制フックを装着する。

    `AgentRegistry(agent_builder=GovernedAgentBuilder(policy=...))` で注入すると、registry の遅延
    構築が唯一の構築経路（`_builder().build`）を通るため、循環ハンドオフ解決後の到達可能 spec も
    govern 済みになる。`AgentSpec` / `tools` / コア `__all__` / `AgentBuilder` Protocol は変えない。

    強制点は 2 つある。`spec.tools` の `FunctionTool` は build 時に実行本体
    （`on_invoke_tool`）をラップして評価する。`spec.mcp_servers` 経由の MCP ツールは SDK が
    **run 時**に解決するため build 時のラップ対象が存在せず、装着した `AgentHooks.on_tool_start`
    で評価する（宣言は同じ `allowed_tools` / `blocked_patterns` で足り、規約は 1 本のまま）。

    既知の境界（govern 対象外）:
        - `sub_agents` の as_tool は registry が build 後に注入するため per-call の allow/deny
          評価・監査レコードを持たない（監査フックの tool_start / tool_end 記録のみ）。サブ
          エージェント自身が同 builder で build されていれば内部 `FunctionTool` は govern 済み。
        - `register_factory` 経路は builder（`build`）を通らないため govern 対象外。
        - SDK の HITL 承認（`needs_approval`）はツール実行前の承認フローとして govern ラップより
          先に走るため、ポリシーが拒否する呼び出しでも承認要求は先に発生し得る（承認後に deny）。
        - hosted MCP（Responses API のサーバ側 MCP・`HostedMCPTool`）はモデルプロバイダ側で実行
          されるため評価も監査も発生しない。統治されるのは client-side MCP（`spec.mcp_servers`）
          のみ。`RealtimeAgentSpec` の `mcp_servers` も別 builder 経路のため対象外。MCP について
          統治するのはツール**呼び出し**のみで、`get_prompt` / resources 経由でサーバから取得した
          文面は対象外。
        - 評価対象はツール名と引数のみで、**ツールの戻り値は評価されない**（許可した呼び出しの
          結果は素通しでモデル文脈へ入る）。第三者の MCP サーバを使う場合、戻り値が間接プロンプト
          インジェクションの経路になるため SDK の出力ガードレールを併用する。

    MCP 経路と `spec.tools` 経路の非対称（利用者が観測しうる差）:
        - MCP の deny は `on_tool_start` からの送出で合成チェーンを中断するため、利用者の
          `spec.hooks.on_tool_start` へ**到達しない**（`spec.tools` の deny は実行本体のラップで
          弾くため到達する）。利用者フックで監査・計測している場合は観測が欠ける。
        - `tool:` レコードの `agent_id` は宣言時の `spec.name`、`tool_start:` は runtime の
          `agent.name` で取得元が違う（`Agent.clone(name=...)` すると食い違う）。
        - build 後に `Agent.hooks` を差し替える（`clone(hooks=...)` 含む）と **MCP 経路の強制と
          監査がともに失われる**（`spec.tools` 経路はラップが tool 自身へ焼き込まれるため強制と
          per-call の `tool:` レコードは残り、失われるのはフック由来のライフサイクル記録のみ）。
          利用者フックを足したい場合は差し替えでなく `spec.hooks` へ宣言する（本 builder が
          合成するため既存フックは失われない）。

    実装に近い粒度の境界（評価をスキップする全経路・照合名の詳細）は `_adapters/governance.py` の
    `govern_spec` docstring を参照する。

    状態のスコープ（builder 単位）:
        - 既定監査 sink・解決済みポリシー・override 適用記録は **builder インスタンス単位** の
          状態である。`AgentRegistry.clone()` は builder を共有するため、clone 先（llmops /
          lightning の内部 clone を含む）の build も同じ監査チェーン・同じ適用記録に混ざる。
          系（本番 / 評価等）を分けたい場合は builder を registry ごとに分けて注入する。
    """

    def __init__(
        self,
        *,
        policy: str | os.PathLike[str] | object,
        audit_sink: object | None = None,
        inner: AgentBuilder | None = None,
        overrides: Mapping[str, str | os.PathLike[str] | object] | None = None,
    ) -> None:
        """装飾 builder を初期化する。

        Args:
            policy: 既定のポリシー定義（YAML ファイルパス、または AGT ポリシーオブジェクト）。
                `overrides` に掲載されていない全エージェントへ適用される。本層では不透明値として
                保持し、読込・評価は `_adapters.governance` へ委譲する。
            audit_sink: 監査ログ出力先（不透明値）。None のときは初回 `build` で AGT 既定 sink を
                生成し、以降の build で共有する（マルチエージェントでハッシュチェーンが連続する）。
                `__init__` では AGT を import しない（extra 未導入でもコンストラクト可能）。
                `overrides` を使う場合も sink は builder で 1 本共有される（per-agent 分割しない）。
            inner: 装飾対象の `AgentBuilder`。None で `_adapters` の `DefaultAgentBuilder`。
            overrides: エージェント名 -> ポリシー定義の per-agent 上書き。`build(spec)` 時に
                `spec.name` との完全一致（正規化なし）で引き当て、未掲載は `policy`（既定）へ
                フォールバックする。値は `policy` と同形式（YAML パス / ポリシーオブジェクト）で、
                既定と同一の fail-fast 検証を受ける（None は不正値・既定へ戻す意図はキーの削除で
                表現する）。未適用キーは `unapplied_overrides` で確認できる（typo 検知）。
        """
        self._policy = policy
        self._audit_sink = audit_sink
        self._inner = inner
        self._overrides: dict[str, object] = dict(overrides) if overrides is not None else {}
        self._applied_overrides: set[str] = set()

    @classmethod
    def from_yaml(
        cls,
        path: str | os.PathLike[str],
        *,
        audit_sink: object | None = None,
        inner: AgentBuilder | None = None,
    ) -> GovernedAgentBuilder:
        """bundle YAML（`default` + `agents`）から builder を構築する（制限の全量を 1 ファイルへ）。

        既定 / per-agent の制限をコード側に分離せず、単一の宣言ファイルにまとめたい場合の
        入り口。読み込み・検証（各セクションとも単一ポリシー YAML と同一の fail-fast）は
        呼び出し時に即時実行され、設定ミスは起動時に顕在化する。コード側でポリシー
        オブジェクトを組みたい場合は通常のコンストラクタ（`policy=` / `overrides=`）を使う
        （両形式は等価で、どちらでも同じ builder になる）。

        ```yaml
        default:
          allowed_tools: [lookup_order]
        agents:
          support:
            allowed_tools: [lookup_order, refund]
        ```

        Args:
            path: bundle YAML のファイルパス。
            audit_sink: 監査ログ出力先（コンストラクタと同じ・None で既定 sink を共有生成）。
            inner: 装飾対象の `AgentBuilder`（コンストラクタと同じ）。

        Returns:
            bundle の `default` を既定ポリシー・`agents` を overrides として構成した builder。

        Raises:
            ImportError: governance extra（agent-governance-toolkit）が未導入の場合（案内付き）。
            FileNotFoundError: パスが存在しない場合。
            ValueError: bundle YAML の構造・キーが不正な場合（未知キー / `default` 欠落等）。
        """
        from ..._adapters import load_policy_bundle

        default_policy, agent_policies = load_policy_bundle(path)
        return cls(
            policy=default_policy,
            audit_sink=audit_sink,
            inner=inner,
            overrides=agent_policies,
        )

    @property
    def unapplied_overrides(self) -> frozenset[str]:
        """`overrides` のうち一度も build で適用されていないキー集合を返す（typo 検知用）。

        登録済みの全エージェントを build した後に空でなければ、registry のエージェント名と
        一致しないキー（typo の疑い）が含まれている。利用者・テストは本プロパティが空集合で
        あることを確認することで、意図しない既定ポリシーへのフォールバックを検知できる。

        「適用済み」は **builder 単位の build 成功**を意味する（override の読込・検証や build に
        失敗したキーは未適用のまま残る）。registry の解決（`AgentRegistry.get`）は途中失敗時に
        構築済みエージェントをロールバックするが、本プロパティはそのトランザクションを観測しない
        ため、ロールバックされた build のキーも適用済みのままになる（次回の解決成功後に確認する
        運用を前提とする）。

        Returns:
            未適用の overrides キーの frozenset。全キー適用済み（または overrides 未指定）なら空。
        """
        return frozenset(self._overrides) - self._applied_overrides

    @property
    def audit_sink(self) -> object | None:
        """既定 / 指定の監査 sink を返す（初回 build 前は利用者指定値 or None）。

        既定 sink（`audit_sink=None` 指定時）は builder 内で生成・build 間で共有され、本プロパティで
        取得・検証できる（記録の `verify_chain` 等）。実運用では
        `GovernedAgentBuilder(audit_sink=...)` で明示指定もできる。

        Returns:
            監査 sink オブジェクト。未指定かつ初回 build 前なら None。
        """
        return self._audit_sink

    def build(self, spec: AgentSpec) -> Agent:
        """spec を govern 化して `inner` で Agent 化する（`AgentBuilder` Protocol 実装）。

        `_adapters.governance.govern_spec` で tools を govern ラップ + 監査フックを `spec.hooks` と
        合成した新 spec を得て、`inner`（None なら `DefaultAgentBuilder`）の `build` へ委譲する。
        `inner` への委譲で `AgentBuilder` の「handoffs 空で構築」契約をそのまま継承する。
        `audit_sink` 未指定時は初回 build で既定 sink を生成して保持し、以降の build で共有する
        （sink の分断を防ぎハッシュチェーンを連続させる）。

        適用ポリシーは `overrides` に `spec.name` が掲載されていればその値、なければ既定
        （`policy`）を使う。掲載キーは **build が成功した後に** 適用済みとして記録され
        `unapplied_overrides` から除かれる（override の読込・検証に失敗した build ではキーが
        未適用のまま残り、失敗した override を診断できる）。

        `sub_agents` の as_tool（registry が build 後に注入）と `register_factory` 経路は govern
        対象外（クラス docstring の「既知の境界」を参照）。

        Args:
            spec: 構築対象の `AgentSpec`。

        Returns:
            govern 済み tools / 監査フックを装着した `Agent`（handoffs は空・サブツール未注入）。

        Raises:
            ImportError: governance extra（agent-governance-toolkit）が未導入の場合（案内付き）。
        """
        import os

        from ..._adapters import DefaultAgentBuilder, govern_spec, new_audit_sink, resolve_policy

        if self._audit_sink is None:
            self._audit_sink = new_audit_sink()
        # YAML パスのポリシーは初回使用時に 1 度だけ読み込み・検証し、解決済みオブジェクトで
        # 保持する（build ごとの再読込は、同一 registry 解決内でのエージェント間ポリシー不整合
        # （ファイル更新タイミング差）や非強制フィールド警告の重複発火を生むため。以降の build
        # はスナップショットを共有する）。オブジェクト形はそのまま渡す（govern_spec 内で検証）。
        used_override = spec.name in self._overrides
        if used_override:
            policy = self._overrides[spec.name]
            if isinstance(policy, (str, os.PathLike)):
                policy = resolve_policy(policy)
                self._overrides[spec.name] = policy
        else:
            policy = self._policy
            if isinstance(policy, (str, os.PathLike)):
                policy = resolve_policy(policy)
                self._policy = policy
        governed = govern_spec(spec, policy=policy, audit_sink=self._audit_sink)
        builder = self._inner if self._inner is not None else DefaultAgentBuilder()
        agent = builder.build(governed)
        # 適用済み記録は build 成功後（失敗した override を unapplied_overrides に残すため）。
        if used_override:
            self._applied_overrides.add(spec.name)
        return agent
