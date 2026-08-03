# 0016: AgentHooks 合成ヘルパー chain_agent_hooks を agent 単位専用クラスで提供する

- Status: accepted (partially superseded by 0017)
- Date: 2026-08-03

## Context

ADR-0003 で run 単位の合成ヘルパー `chain_hooks(*hooks: RunHooksBase) -> RunHooksBase` を提供したが、
エージェント単位（`agents.AgentHooks`）には同等の公開窓口がない。`AgentSpec.hooks` は単一スロット
（`Any` 型）であり、利用者が自作フックを複数宣言するには「どれか 1 つを捨てる」か「委譲クラスを
自作する」しかなく、run 単位との非対称が残っていた。

一方で等価な duck-typed 委譲は `_adapters/governance.py` の `_delegate` / `_make_audit_hooks`
（監査記録 -> 既存フックへ委譲する合成 `AgentHooks` を作る）として内部に既に存在した。すなわち
不足していたのは公開窓口であり、委譲ロジック自体は lib 内に二重に持たせない形で再利用できる。

SDK 側の制約（openai-agents 0.17.4 で実行確認）:

- `agents/lifecycle.py` は基底 2 クラス（`RunHooksBase` / `AgentHooksBase`）と添字付きエイリアス
  2 個（`RunHooks` / `AgentHooks`）のみで、合成 / combinator / list 受理の API を持たない。
  `Agent.hooks` も単一スロットである。
- `AgentHooksBase` のメソッドは `on_start` / `on_end` / `on_handoff` / `on_llm_start` /
  `on_llm_end` / `on_tool_start` / `on_tool_end` の 7 件で、run 単位（`on_agent_start` /
  `on_agent_end`）とメソッド名が異なり、`on_handoff` の引数意味も agent 単位は
  `(context, agent, source)`、run 単位は `(context, from_agent, to_agent)` で異なる。
- `AgentHooks` は `AgentHooksBase[TContext, Agent]` の添字付きジェネリックエイリアスであり
  `isinstance` の第 2 引数に使えない（`TypeError: Subscripted generics cannot be used with
  class and instance checks`）。型判定には非ジェネリック基底 `AgentHooksBase` を使う。

検討した選択肢:

1. **既存 `_ChainedHooks`（run 単位）を継承・共用（却下）**: `RunHooksBase` サブクラスのため
   `Agent(hooks=...)` の型契約に載らない。メソッド名と `on_handoff` の引数意味が異なるため、
   共用すると使われない run 単位メソッドを抱え、`on_handoff` の意味を型で区別できなくなる。
2. **`_delegate` をそのまま公開（却下）**: `_delegate(inner, method, *args)` はメソッド名文字列を
   呼び側が渡す coroutine 関数で、`AgentSpec.hooks` へ代入できるオブジェクトを返さない。
3. **`AgentSpec.hooks` を list 化し registry 側で合成（却下）**: 公開データクラスの契約変更に
   なり、registry に実行時の型分岐が入って「宣言と薄い結線」の役割分担が崩れる。利用者側の
   factory 呼び出し 1 回で同じ目的を満たせる。
4. **agent 単位専用の合成クラス + ファクトリを `_adapters/hooks.py` に追加（採用）**。

## Decision

- `chain_agent_hooks(*hooks: Any) -> AgentHooksBase[Any, Any]` を提供する。公開窓口は
  `oai_agentspec.runtime.hooks`（`chain_hooks` と同じ PEP 562 遅延再エクスポート窓口）、実装実体は
  `_adapters/hooks.py`（`AgentHooksBase` のサブクラス定義に `agents.lifecycle` の import が
  不可避なため SDK 隔離に従う）。コア `oai_agentspec.__all__` には載せない（`chain_hooks` と同じ
  実行寄り層の独立窓口扱い）。
- 合成実体は agent 単位専用クラス `_ChainedAgentHooks` とし、run 単位 `_ChainedHooks` を継承も
  共用もしない（MRO に含めない）。メソッド名集合と `on_handoff` の引数意味が異なるため、
  1 クラスに束ねると型で区別できない誤用（run 単位フックを agent スロットへ渡す等）を招く。
  run 単位側は実装・仕様・戻り値とも不変に保つ。
- 要素として `None` と部分実装（`AgentHooksBase` 非継承で一部の `on_*` のみを持つ duck-typed
  オブジェクト）を許容する。要素型注釈を `AgentHooksBase[Any, Any] | None` ではなく `Any` と
  するのは、受理集合を偽って狭く宣言しないため。受理する 3 形はファクトリの docstring に列挙する。
- 縮退は `None` 除外後の実効件数で決める。0 件なら全 `on_*` が no-op の素 `AgentHooksBase()`、
  1 件かつ `isinstance(x, AgentHooksBase)` ならその要素自身（`is` 一致・ラッパ非生成）、それ以外は
  `_ChainedAgentHooks` の新インスタンス。run 単位との非対称（`None` 無視・非インスタンスは包む）は
  意図的で、run 単位は戻り値が `Runner.run(hooks=...)` の型契約を満たす必要があり緩さを持ち込めない。
  非対称の理由は `_adapters/hooks.py` の docstring に明記する。
- 委譲実体は `_adapters/hooks.py` のモジュール private ヘルパー 1 個へ一元化し、
  `governance._delegate` は削除する。`_make_audit_hooks(sink, inner)` は「監査記録のみを行う
  `AgentHooks` サブクラス」と `chain_agent_hooks(audit, inner)` の組み合わせへ再構成する。
  宣言順が `(audit, inner)` であることで「監査記録 -> 既存フック委譲」の順序が保たれ、
  `govern_spec` の外部から観測可能な振る舞いは不変。
- 監査専用クラスは `AgentHooks[Any]`（= `AgentHooksBase`）の継承を維持する。委譲を持たなくなっても
  基底を外すと `isinstance(audit, AgentHooksBase)` が偽になり、`inner is None` のときに合成が
  audit 自身を `is` 一致で返す縮退が壊れて不要なラッパが生成される。
- `governance` 側からの `chain_agent_hooks` 参照は `_make_audit_hooks` の**関数内遅延 import**とする。
  `governance` は `_adapters/__init__.py` 経由で常時ロードされるため、トップレベル import にすると
  `_adapters.hooks` も常時ロードされ、`runtime/hooks` 窓口の「import だけでは実装実体を発火させない」
  遅延 import 契約が破れる。build 時 1 回の呼び出しであり遅延化のコストはない。

## Consequences

- + 複数の `AgentHooks` を `AgentSpec(hooks=chain_agent_hooks(a, b))` の 1 行で宣言でき、run 単位
  （`chain_hooks`）との非対称が解消される。
- + `None` と部分実装を許容するため、条件付きで有効化するフックを分岐なしに列挙でき、継承の
  定型コードも不要になる。
- + duck-typed 委譲ロジックが lib 内 1 箇所（`_adapters/hooks.py`）のみになり、governance の
  監査フックは監査記録の責務だけを持つ。
- - agent 単位・run 単位の 2 クラスが SDK のメソッド集合に個別に追随する必要がある。追随漏れは
  「そのフックだけ合成されない」silent gap になるため、各クラスにオーバーライド網羅テストを置く。
- - 属性が callable でない場合（`on_start = 1` 等）は callable 判定を挟まないため、呼び出し時の
  `TypeError` がそのまま伝播する（既存 `_delegate` と同一挙動を維持する意図的なトレードオフ）。
- - `fail-fast`（前段の例外で後段を呼ばず即伝播）のため、全要素を必ず実行したい用途には合致しない
  （run 単位と同一方針）。

## Confirmation

強制手段は `docs/QUALITY-GUARANTEES.md` に登録した 1 行（agent 単位 hooks 合成ヘルパー）が指す
テストである。

- `_ChainedAgentHooks` が SDK `AgentHooksBase` の `on_*` を覆い漏れなくオーバーライドしていること
  （`vars()` 上のオーバーライド集合との差集合が空）

これに加えて次を lib 内のテスト・ゲートで担保する。個別の assert とテスト名はテストの docstring を
一次情報とし、本 ADR には列挙しない。

- 縮退仕様（0 件 / 全 `None` / 1 件の `is` 一致 / 非インスタンスの包み込み）・duck-typed 委譲・
  fail-fast・引数の `is` 一致転送・MRO に run 単位クラスを含まないこと: `tests/runtime/hooks/`
- 公開窓口の `__all__` メンバ集合・遅延解決とキャッシュ・窓口 import 時に実装実体を発火させない
  こと（`governance` のトップレベル import 回帰もこの probe が検知する）: `tests/runtime/hooks/`
- `govern_spec` の観測可能な振る舞い不変（監査記録 -> 委譲の順序・`spec.hooks is None` 時の監査
  単体動作・元 spec の非破壊）: `tests/_adapters/test_governance_l2.py`
- SDK 隔離: `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` が空
