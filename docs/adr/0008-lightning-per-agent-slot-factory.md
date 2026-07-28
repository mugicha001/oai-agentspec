# 0008: per-agent のプロンプトスロット指定をファクトリで畳み、`optimize(slot=)` に `Slot` の列を受理させる

- Status: superseded by 0011 (partially: prompt_slots 併存決定のみ)
- Date: 2026-07-27

## Context

`prompt_slots` は `base` / `parts` / `vars` が全エージェント共通で、per-agent に指定できるのは `tune` のみである。
これに対し、エージェントごとに `base` / `parts` / `vars` を変えたいという要求があった。

`prompt_slots` のシグネチャを拡張する案は 2 度検討し、いずれも却下された。

- **flat 形**（`base_overrides=` / `parts_overrides=` / `vars_overrides=` の 3 kwarg を追加する案）: 上書き対象ごとに
  kwarg が増え、1 エージェントの構成が複数の mapping に分散するため call site の可読性が改善しない。
- **集約形**（`overrides={エージェント名: {"base": ..., "parts": ...}}` の 1 kwarg を追加する案）: 内側 dict が
  実 kwarg ではないため、許可キーリスト検査・`Mapping` 型検査・内側値の型契約・未指定時のフォールバック規約を
  ライブラリ側で新設する必要があり、それでも typo は実行時まで検出できない。

さらに `prompt_slots` は `layout` 非対応（ADR 0006 の決定）であるため、どちらの案も**セグメント種別の順序
（base -> parts -> agent）を per-agent に変える手段を提供できない**。custom `build` の per-agent 指定も同様に届かない。

一方、既存の `prompt_slot` をエージェントごとに呼ぶ書き方（`functools.partial` で `store` / `registry` / 共通既定値を
束ねる形を含む）は、要求される per-agent 指定・`parts` 内の順序・`layout`・`build` をすべて満たすことを実機で確認した。
残る不足は次の 3 点である。

1. **`vars` の silent 劣化**: 共通 `vars` を束ねた上で per-agent に `vars={"tone": "formal"}` と書くと `vars` は
   まるごと置換され、共通キーが消える。`_ensure_fixed_vars_present` は固定セグメント（base / parts / 非 tune の agent）
   しか検査しないため、失われたキーの `${var}` が tune セグメント側にある場合は検査を素通りし、literal `${var}` を
   含むプロンプトで最適化が回って APO スコアが黙って劣化する。利用者は毎回 `vars={**共通, ...}` と書く規律を要求される。
2. **発見可能性**: この書き方は公開 API として提示されておらず、`__all__` にも docstring にも usage docs にも現れない。
3. **エージェント名の二重記述**: 返り値は `{名前: Slot}` の dict 手組みになるため、エージェント名を dict キーと
   `agent=` の 2 箇所に書く。

## Decision

`prompt_slot` へ委譲するだけのファクトリ `prompt_slot_factory(store, registry=None, **defaults) -> Callable[..., Slot]`
を `runtime/lightning/slots.py` に追加し、per-agent の上書きを**本物の kwargs** で受ける。返り値は
`make(agent: str, **overrides) -> Slot` で、`prompt_slot` の全 kwarg（`base` / `parts` / `layout` / `tune` / `vars` /
`build`）がそのまま素通しされる。

`vars` は **双方が dict のときのみマージ**（同一キーは per-agent 優先・新しい dict を作るため既定側は非破壊）とし、
それ以外の kwarg は置換とする。このマージが本決定で新規に定義する唯一の挙動であり、上記 Context の 1 を塞ぐ。
`base=None` / `parts=[]` / `vars=None` のような明示的な打ち消しを成立させるため、実装では `None` を「未指定」として
除去するフィルタを入れない。`vars` に callable が絡む組み合わせは合成せず置換に倒す（callable は「rollout 時に
context から全 vars を生成する」契約であり、dict との合成意味論は `prompt_slot` に存在しないため、ここで発明しない）。

許可キーリスト検査・`agent` 衝突の専用検査は**入れない**。typo（`part=` / `varz=`）も `defaults` への `agent` 混入も、
委譲先 `prompt_slot` の呼び出しで Python が `TypeError` を送出するため、最初の `make()` 呼び出しで必ず露見する。

あわせて `optimize(slot=)` の受理型を `Slot | str | Iterable[Slot] | dict[str, Slot | str] | None` に広げ、`Slot` の列を
`Slot.name` をキーとする mapping へ正規化する（Context の 3 を塞ぐ）。判別は `_normalize_slots` 内で
`None` -> `Slot` -> `dict` -> `str` -> `Iterable` の順に行う。`str` / `dict` も `Iterable` であるため、両者が
`Iterable` 分岐より前に処理されることがこの設計の前提条件である。列経路は空列 / `Slot.name` の重複 / `Slot` 以外の
要素の混在をいずれも `OptimizeError(CONFIG_MISSING)` で fail-closed し、通過後に
`_ensure_slot_target_name_match` を dict 経路と同じ位置（`_normalize_slots` 内部）で適用する。列は自動 rebind 経路
専用と定義し、名前を持たない生 seed の列は受理しない。

`Sequence` ではなく `Iterable` を採る。`str` / `dict` は上位分岐で処理済みのため受理範囲を広げてもコード量は増えず、
generator や `set` / `dict_values` が「生 seed 経路では rebind の明示が必要です」という無関係なメッセージで落ちる
誤誘導が消える。`Iterable` は `__iter__` の有無で判定されるため、`__getitem__` のみを持つ旧式シーケンスは従来どおり
末尾の fallback に落ち、受理範囲が意図せず膨らむことはない。generator の 1 度きり消費は、新経路が先頭で `list(slot)`
して以降その列だけを使い、かつ列経路では `_normalize_slots` が必ず非空 mapping を返すか送出するため `_seeds_of` が
`slot` を再走査しないことで閉じる。

`prompt_slots` は変更しない。全エージェントが同一構成で per-agent 差分が `tune` だけのケースでは今も最短であるため、
両者は使い分けとして併存させる。

- 却下: `prompt_slots` のシグネチャ拡張案（flat 形の 3 kwarg / 集約形の overrides mapping）。Context のとおり
  call site の可読性が改善せず、`layout` / `build` の per-agent 指定にも届かないため不採用。
- 却下: 命名 `slot_factory` / `slot_defaults`。前者は `Slot` の生成一般（生 seed からの `Slot` 構築）と読める余地が
  あり `prompt_` family から外れる。後者は返り値が callable であることが読み取れない。`prompt_slot` / `prompt_slots`
  と同一接頭辞で family を作り `__all__` 上で隣接させることを優先し `prompt_slot_factory` を採用した。
- 却下: 返り値 callable への `Protocol` 型注釈。`**kwargs` の型は結局 `Any` になり、mypy 非導入の本リポジトリでは
  得られる情報がゼロのため `Callable[..., Slot]` に留める。
- 却下: `parts` の追記セマンティクス。挿入位置の規則を新設することになり `layout` と機能が重なるため、
  `vars` 以外の kwarg は一律置換とする。

## Consequences

- + per-agent 上書きが本物の kwargs であるため、キー名の typo は Python が `TypeError` で弾く。許可キーリスト検査 /
  `Mapping` 型検査 / 内側値の型契約 / フォールバック規約 / 型注釈の `Any` 許容という、集約形が必要としていた設計要素が
  丸ごと不要になる。
- + 実装は `prompt_slot` への委譲のみで、`layout` / `build` を含む全機能が素通しできる。`prompt_slots` では
  提供できない per-agent の `layout` / `build` がファクトリ経由で書けるため、シグネチャ拡張案の上位互換になる。
- + `optimize(slot=)` が `Slot` の列を受けるため、エージェント名を dict キーと `agent=` に二重記述する必要がなくなる。
- + `vars` のマージにより、共通 vars のキーを書き落として tune セグメント側の `${var}` が literal 残留する
  silent 劣化の経路が塞がれる。
- - per-agent 差分がある場合の入口が 2 つ（`prompt_slots` / `prompt_slot_factory`）になるため、usage docs で
  使い分けを明示する必要がある。
- - ファクトリは常に `agent=` を渡すため、`layout` のみを渡して `agent:X` 参照から `Slot.name` を暗黙解決する経路
  （ADR 0006 で定義）はファクトリ経由では使えない。この経路が必要な場合は `prompt_slot` を直接呼ぶ。
- - `vars=None` / `vars={}` による共通 vars の打ち消しは、固定セグメントに `${var}` が残っていない場合に限り成立する。
  残っている場合は既定 build 経路の `_ensure_fixed_vars_present` が `OptimizeError(CONFIG_MISSING)` を送出する
  （既存契約であり本決定はこの挙動に触らない）。
- - `optimize(slot=)` の受理型が 1 つ増える（既存の `None` / `Slot` / dict / str の 4 経路は挙動不変）。
  **将来さらに入力型を追加する場合は、本 ADR の判別順序（`Slot` -> `dict` -> `str` -> `Iterable`）を崩さないこと。**
  `str` / `dict` を `Iterable` 分岐より後ろに置くと、生 seed（`str`）と `{名前: Slot}` mapping が列として解釈され、
  既存経路が silent に壊れる。

## Confirmation

- ファクトリの契約（既定値の適用 / `parts` の置換 / `vars` のマージと per-agent 優先 / 既定 vars の非破壊 /
  `vars=None` と `base=None` の打ち消し / `layout` と `build` の素通し / callable vars の非マージ /
  未知キーと `agent` 混入が `make()` 呼び出し時の `TypeError` になること）の強制手段:
  `tests/runtime/lightning/test_slots_l1.py` に追加する 10 件。
- `optimize(slot=)` の列受理（list / tuple / generator の `{Slot.name: Slot}` 正規化 / 空列・name 重複・非 Slot 混在の
  fail-closed / `AgentSpec` 対象時の名前一致検査）の強制手段:
  `tests/runtime/lightning/test_optimizer_l2.py` に追加する 3 件。
- 既存経路の不変性は、`test_optimizer_l2.py` の `_normalize_slots` 既存群（`Slot` 単体 / dict / 全生 seed dict /
  空 dict / 混在 / キー不一致 / `AgentSpec` 名検査 / 生 str）と `test_slots_l1.py` の `prompt_slots` 10 件が
  **無修正で緑であること**を回帰 anchor とする。とくに `str` 判定の前倒しの非退行は既存の
  `test_normalize_slots_raw_str_returns_none` が pin する。
- `docs/QUALITY-GUARANTEES.md` には行を登録しない。列受理の fail-closed は `_normalize_slots` 内の純粋な分岐であり、
  崩れれば上記テストが即座に赤になるため、台帳が対象とする「テスト外の要因で silent に緩む」性質を持たないため。
  将来の入力型追加に対する注意は本節および Consequences の判別順序で担保する。
