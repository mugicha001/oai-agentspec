# 0017: chain_agent_hooks は run 単位フックを build 時に拒否する

- Status: accepted
- Date: 2026-08-03

## Context

ADR-0016 は agent 単位の合成ヘルパー `chain_agent_hooks(*hooks: Any)` を提供し、要素として
`None` と部分実装（`AgentHooksBase` 非継承で一部の `on_*` のみを持つ duck-typed オブジェクト）を
許容すると決めた。要素型注釈を `Any` としたのは、受理集合を偽って狭く宣言しないためである。
同 ADR は agent 単位専用クラスを run 単位 `_ChainedHooks` と分ける根拠として「1 クラスに束ねると
型で区別できない誤用（run 単位フックを agent スロットへ渡す等）を招く」ことも挙げていた。

しかしクラスを分けただけでは当該誤用は実行時に区別されない。run 単位フック
（`agents.lifecycle.RunHooksBase` インスタンス）を `chain_agent_hooks` へ渡すと次が起きる。

- `isinstance(hook, AgentHooksBase)` が偽なので縮退の「1 件かつインスタンス」経路に入らず、
  合成ラッパ `_ChainedAgentHooks` へ包まれる。ラッパは `AgentHooksBase` サブクラスなので
  SDK `Agent.__post_init__` の型チェックも通過し、宣言は例外なく成立する。
- `on_start` / `on_end` は run 単位では `on_agent_start` / `on_agent_end` という別名のため
  `getattr` が None を返し、duck-typed 委譲の skip 経路で**黙って捨てられる**。
- 名前が衝突する `on_handoff` は発火するが、agent 単位の `(context, agent=遷移先,
  source=遷移元)` が run 単位の `(context, from_agent, to_agent)` へ位置引数で渡るため
  **from/to が反転**する。誤ったハンドオフ記録が例外なしで残る。

同じ「包んでも黙って no-op になる」問題は、`on_*` を 1 つも持たない要素にも当てはまる。委譲は
`getattr` で有無を見て無ければ skip するため、そのような要素は合成ラッパに包まれて全メソッドが
no-op になり、`Agent.hooks` には正しく見える値が入るのにフックが 1 度も発火しない。実際に
起きやすい誤用は `*` の付け忘れ（`chain_agent_hooks([h1, h2])` で list 自体が要素になる）と
型違い・typo で、どちらも例外も警告も出ない。

`govern_spec` は内部で `chain_agent_hooks(audit, inner)` を通るため、`spec.hooks` に run 単位
フックや `on_*` を持たない値を入れた宣言も同じ経路で壊れる。mypy は非導入のため静的な歯止めも
ない。

検討した代替案:

1. **docs への明記のみ（却下）**: docs は実行されない。型でも実行時でも検出されず、
   `on_handoff` の from/to 反転による誤データがそのまま残る。
2. **`on_agent_start` 等の属性を持つ要素を検知して `warnings.warn`（却下）**: 警告は既定で
   1 回のみ表示され容易に抑制でき、誤った記録を止められない。加えて属性名ベースの判定は
   独自命名の duck-typed オブジェクトへ偽陽性を出す（`isinstance` なら偽陽性はない）。
3. **型注釈の厳格化（`*hooks: AgentHooksBase[Any, Any] | None`）（却下）**: mypy 非導入のため
   実行時効果がゼロで、部分実装の受理を偽って狭く宣言することになり ADR-0016 が `Any` を
   選んだ理由に反する。
4. **`on_handoff` で引数名を introspect して意味を適応（却下）**: 実行時イントロスペクションで
   宣言意図を推測する仕組みであり、build-don't-run と「薄い結線」に反する。誤用を成立させる
   方向でもある。

## Decision

ADR-0016 の「受理集合を狭く宣言しない」判断を、run 単位フックの拒否という形で**部分的に撤回**
する。ADR-0016 本文は append-only のため書き換えず、Status に `partially superseded by 0017` を
追記する。撤回対象は次の 2 点で、`None` と部分実装の受理および `Any` 注釈そのものは有効なまま
残る。

1. 「`RunHooksBase` インスタンスも実行時に受理される」点。
2. ADR-0016 が主張した「`govern_spec` の外部から観測可能な振る舞いは不変」という性質。
   `spec.hooks` に run 単位フックを置いた宣言は build 時に `TypeError` へ変わるため、この
   不変性は成り立たなくなる（詳細は Consequences）。

- `chain_agent_hooks` は要素に `RunHooksBase` インスタンスが含まれる場合 `TypeError` を送出する。
  メッセージには引数位置・クラス名・メソッド名が異なる理由・`chain_hooks` への誘導を含める。
- 検証は `None` 除外の**前**（利用者が渡した引数列そのもの）に対して行い、縮退分岐
  （0 件 / 1 件 / 2 件以上）より前に置く。除外前を対象にするのは、エラーメッセージの
  `hooks[i]` を利用者の引数位置と一致させるためである（除外後の位置で数えると
  `chain_agent_hooks(None, None, run_hook)` が `hooks[0]` という誤った位置を案内する）。
  `None` は `RunHooksBase` インスタンスではないため判定はスキップされ、`None` のみの
  呼び出しは従来どおり素の `AgentHooksBase()` を返す。この順序はテストで pin する。
- 検証は `chain_agent_hooks` の 1 箇所に置き、`_ChainedAgentHooks.__init__` には入れない。
  実効 1 件でフック自身を返す縮退経路はコンストラクタを通らないため、そこに置くと検証漏れになる。
- 両基底（`AgentHooksBase` と `RunHooksBase`）を継承した要素も拒否する。`on_handoff` を
  1 メソッドしか持てず引数意味が一意に決まらないため、どちらとして解釈しても誤りうる。
  agent 単位専用クラスへ分けて渡すことを利用者に要求する。
- **`on_*` を 1 つも持たない要素も拒否する**。委譲は `getattr` で有無を見て無ければ skip するため、
  そのような要素は包んでも全メソッドが no-op になり、`Agent.hooks` には正しく見える値が入るのに
  フックが 1 度も発火しない silent gap を生む。起きやすい誤用は `*` の付け忘れ
  （`chain_agent_hooks([h1, h2])` で list 自体が要素になる）と型違い・typo。拒否条件は「7 つの
  `on_*` 名のいずれも持たない」に限定し、1 つでも持つ部分実装は従来どおり受理する（ADR-0016 の
  部分実装サポートは維持する）。判定に使うメソッド名は `AgentHooksBase` から導出し、SDK に
  メソッドが追加されたときに自動で追随させる（ハードコードすると新メソッドだけを持つ要素を
  誤って拒否しうる）。
- 例外型は `TypeError` とする。本ライブラリは宣言内容の不正に `ValueError`、渡されたオブジェクトの
  型・形状が契約と違う場合に `TypeError` を使う（`_adapters/governance.py` の policy メソッド
  検証と同じ二分）。新規の独自例外型は追加しない。
- **run 単位 `chain_hooks` には対称の検証を入れない**。ADR-0016 の「run 単位側は実装・仕様・
  戻り値とも不変に保つ」を維持する。被害度が非対称であり、`_ChainedHooks` は `getattr` ガードを
  持たず `hook.on_agent_start(...)` を直接呼ぶため、agent 単位フックを渡した誤用は実行開始直後に
  `AttributeError` で顕在化する。実行時拒否が必要なのは「存在しないメソッドを黙って skip する」
  duck-typed 委譲を持つ agent 単位側のみである。

## Consequences

- + `RunHooksBase` を継承した要素については、`on_start` / `on_end` の silent skip と
  `on_handoff` の from/to 反転が起こりえなくなる。誤ったハンドオフ記録という観測しにくい失敗が、
  宣言時点の例外へ前倒しされる。
- - **残る穴**: 判定は `isinstance` のため、`RunHooksBase` を継承せずに run 単位の形
  （`on_agent_start` や `(context, from_agent, to_agent)` 署名の `on_handoff`）だけを持つ
  duck-typed オブジェクトは拒否されず、同じ from/to 反転が起こりうる。属性名ベースの検知
  （却下案 2）は独自命名の部分実装へ偽陽性を出すため採らなかった。本 ADR の保証範囲は
  `RunHooksBase` のインスタンスに限られる。
- + ADR-0016 がクラス分離の根拠として挙げた「型で区別できない誤用」が、実際に区別されるように
  なる。論拠と実挙動のギャップが解消される。
- + `on_*` を 1 つも持たない要素の拒否により、`*` の付け忘れ・型違い・typo で全フックが無音で
  失われる状態が build 時に検知される。これは SDK パリティ tripwire が防いでいる silent gap と
  同じクラスの欠陥で、対策の一貫性が取れる。
- - `chain_agent_hooks` の実行時の受理集合が `Any` 注釈より狭くなる。注釈と実挙動の差は
  docstring の `Raises` と受理形の記述で埋める必要がある。
- - `govern_spec` は既にリリース済みの公開シンボルであり、`spec.hooks` に run 単位フックまたは
  `on_*` を 1 つも持たない値（`*` の付け忘れで渡した list 等）を入れていた宣言は build 時に
  `TypeError` へ変わる（既存シンボルの振る舞い変更）。ただしどちらの宣言も変更前からすでに
  壊れており（前者は監査後の委譲が silent skip され `on_handoff` が反転、後者は既存フックが
  1 度も発火しない）、fail-fast 化は正当と判断した。
- - run 単位側と非対称な設計が残る。対称化の要求が出た場合は、上記の被害度の非対称を根拠に
  判断すること。

## Confirmation

強制手段は `docs/QUALITY-GUARANTEES.md` に登録した行が指すテストである。個別の assert と
テスト名はテストの docstring を一次情報とし、本 ADR には列挙しない。

- 単体 / agent 単位との混在 / `None` 併記 / 両基底継承の各ケースで `TypeError` が送出され、
  メッセージに引数位置とクラス名と `chain_hooks` への誘導が含まれること: `tests/runtime/hooks/`
- `None` のみの呼び出しが従来どおり素の `AgentHooksBase()` を返すこと（回帰）:
  `tests/runtime/hooks/`
- `spec.hooks` に run 単位フックを置いた宣言が `govern_spec` の build 時に `TypeError` になる
  こと: `tests/_adapters/test_governance_l2.py`
