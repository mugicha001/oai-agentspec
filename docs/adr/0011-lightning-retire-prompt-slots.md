# 0011: `prompt_slots` を廃止し複数スロット生成を `prompt_slot_factory` へ一本化する

- Status: accepted
- Date: 2026-07-29

## Context

ADR 0008 は `prompt_slot_factory` を導入した際、`prompt_slots` を「全エージェントが同一構成で
per-agent 差分が `tune` だけのケースでは今も最短」という限定的な根拠で併存させた。同 ADR は
同時に、ファクトリが `prompt_slots` では提供できない per-agent の `layout` / `build` まで素通し
できる **上位互換** であることも明記している。すなわち `prompt_slots` にしかできないことは存在せず、
併存の根拠は「短く書ける」という書き味の 1 点だけである。

その後の実態は次のとおり。

- グラフ全体の APO を扱う examples（`examples/lightning/03_graph_apo.py` / `08_*`）は既に
  ファクトリを使用しており、usage docs の「グラフ全体は `prompt_slots`」という記述は
  コード実態と一致していない。この不一致は併存を維持する場合でも修正が不可避である。
- 利用者側には「単体か複数か」に加えて「複数のとき `prompt_slots` かファクトリか」という
  第 2 の使い分け判断が残り続ける。

本決定の要件は「API 認知負荷の削減（1 経路に絞ることで使い分けの判断が不要になる）」である。

判断の前提として、本ライブラリに外部利用者（配布先）は存在しないことを確認済みである。
バージョンは 0.3.0（pre-1.0 SemVer）であり、minor bump での破壊的変更が許容される。

検討した選択肢は次の 3 案。

| 軸 | A: 併存維持（docs 修正のみ） | B: soft deprecation | C: hard removal |
|---|---|---|---|
| API 認知負荷 | 使い分け判断が残り続ける（要件未達） | 1 バージョン間は 3 関数併存で最悪 | 即座に 1 経路へ削減 |
| 後方互換 | 完全維持 | 警告のみ・0.5.0 で breaking | 0.4.0 で breaking（pre-1.0 SemVer で許容範囲） |
| 実装コスト | docs 不一致修正のみ | 警告実装 + 既存テスト 10 件の警告対応 + 後日削除の 2 段階 | 削除 1 段階 |
| docs 単純化 | 使い分け表 3 行維持 | deprecated 注記が現在仕様の docs に載る | 使い分け表から行削除・誤誘導を解消 |
| リポジトリ既存パターン整合 | - | deprecation 前例ゼロ（新規パターン確立が必要） | 削除 + 要件撤回宣言の前例あり（ADR 0007） |

実現手段として既存資産・標準機能も先に列挙した。

| 手段 | 採否 | 理由 |
|---|---|---|
| `prompt_slot_factory` + dict comprehension（既存資産のみ） | 採用 | `prompt_slots` の全機能を上位互換で提供済み。新規コード不要 |
| `warnings.warn(DeprecationWarning)` | 不採用 | soft 案の実現手段としては可だが、hard removal 採択のため不要 |
| `warnings.deprecated`（PEP 702） | 不採用 | Python 3.13+ 専用。本プロジェクトは `requires-python >= 3.12` |
| `typing_extensions.deprecated` | 不採用 | 新規直接依存の追加。mypy 非導入で静的検知の利得も無い |
| ファクトリ拡張（`agents=[...]` 一括生成ラッパーの追加） | 不採用 | dict comprehension（標準構文）で同じ目的を満たせる。新規メカニズムを増やさない |

## Decision

案 C（hard removal）を採用する。次マイナー **0.4.0** で `prompt_slots` を削除し、deprecation
期間は設けない。複数スロットの生成は `prompt_slot_factory` + dict comprehension に一本化し、
使い分けの軸を「単体（`prompt_slot`）か複数（`prompt_slot_factory`）か」の 1 つに縮約する。

### 要件 FR-9 の読み替え宣言

要件定義書 `docs/requirements/lightning-optimization-extra.md` の FR-9 は AC レベルで
`prompt_slots` を名指しで要求している。本決定はこれと正面から矛盾するため、ADR 0007 が
要件 NFR-3 の撤回を ADR 本文で宣言した前例と同型で対処する。

FR-9 の `prompt_slots` 名指し AC は、**`prompt_slot_factory` + dict comprehension で充足する**
読み替えとする。要件の意図（グラフ全体の APO が単一の `optimize` 呼び出しで成立すること）は
不変であり、変更するのは充足手段の指定のみである。

### ADR 0008 からの引き継ぎ範囲

ADR 0008 の Status は `superseded by 0011 (partially: prompt_slots 併存決定のみ)` とする。
0008 のうち本 ADR が覆すのは `prompt_slots` を併存させるという決定の 1 点に限られる。
`prompt_slot_factory` の導入（既定値の適用 / `vars` の dict マージ / その他 kwarg の置換 /
許可キーリスト検査を置かない方針）と `optimize(slot=)` の `Iterable[Slot]` 受理（判別順序
`Slot` -> `dict` -> `str` -> `Iterable` を含む）は **有効なまま本 ADR が引き継ぐ**。

### 移行パターン

機械変換 1 パターンのみで、`optimize` 以降は無修正である。

```python
# Before
slots = prompt_slots(store, registry, ["triage", "billing"], base="main", tune={"triage": "x"})

# After
make_slot = prompt_slot_factory(store, registry, base="main")
tune_map = {"triage": "x"}
slots = {name: make_slot(name, tune=tune_map.get(name)) for name in ["triage", "billing"]}
# 後続の optimize(slot=slots) は無修正
```

## 却下案

### 案 A: 併存維持（docs の不一致修正のみ）

`prompt_slots` を残し、usage docs の「グラフ全体は `prompt_slots`」という実態不一致だけを直す。
コストは最小だが、「複数のときどちらを使うか」という使い分け判断が残るため、本決定の要件
（API 認知負荷の削減）を満たさない。上位互換の関数が併存する状態そのものが認知負荷の源で
あるため却下。

### 案 B: soft deprecation（0.4.0 で `DeprecationWarning`・0.5.0 で削除）

段階的移行として一般には妥当だが、本リポジトリでは次の理由で却下する。

- **deprecation の前例がゼロ**である。ADR 0007 は legacy shape を deprecation 期間なしで削除し、
  要件（NFR-3 後方互換）の撤回を ADR 本文で宣言した。案 B の方がむしろ新規パターンの導入になる。
- **2 段階分のコストが上乗せされる**。警告の実装に加え、後日の削除作業が別途必要になる。
- **検証事実**: pyproject の pytest 設定に `filterwarnings` は無く、警告でテストが赤になることは
  ない。しかし ADR 0008 が回帰 anchor と明記する `test_prompt_slots_*` 10 件を「警告を出しつつ
  緑」に保つ扱いの設計が別途必要になる。
- 外部利用者が存在しないため、警告期間が保護する対象が実在しない。
- 移行が 1 パターンの機械変換で済むため、段階的猶予の価値が小さい。
- deprecated 注記を現在仕様の docs に載せることになり、docs 規約の「一時的な移行措置を書かない」
  方針と緊張する。

なお、外部配布済み・外部利用者ありという前提であれば案 B が推奨に切り替わる。本決定は
「外部利用者なし」を前提条件として成立する。

## Consequences

- **破壊的変更（あり）**: `oai_agentspec.runtime.lightning` の `__all__` から `prompt_slots` を
  削除する。既存の `prompt_slots(...)` 呼び出しは `ImportError` / `AttributeError` になる
  （pre-1.0 のため許容範囲）。
- 付随する observable 変更: 利用者向けエラーメッセージ（`optimizer.py` の slot 種別エラー・
  `_slots_norm.py` の空 agents エラー）の文言から `prompt_slots` の言及が消える。
- + 使い分け表が「単体 or 複数」の 1 軸に単純化され、usage docs の実態不一致（「グラフ全体は
  `prompt_slots`」）も同時に解消される。
- + 公開 API の関数数が 1 つ減り、`Slot` 生成の入口が 1 つ減る。データフローは変更なく、
  `optimize(slot=)` 以降の正規化・rollout・APO ループは無修正である。
- - 移行が必要な呼び出し側は上記 1 パターンの書き換えを要する。
- **実装は別 Issue**: 本 ADR の成果物は設計判断の記録のみである。コード削除・テスト更新・
  usage docs / architecture.md / examples / README の同期は別 Issue（0.4.0 向け実装）で行う。
- **要件定義書の書き換えも実装 Issue のスコープ**: `docs/requirements/lightning-optimization-extra.md`
  の FR-9 の名指し箇所は、コード削除と同一 PR で書き換える。docs が現在仕様の SoT である以上、
  コードと要件定義書の名指しを同時に更新する。
- **ADR 0005 / 0006 は不変更**: 両 ADR の `prompt_slots` 言及は当時の記録として正であり、
  append-only 規約により実装 Issue でも書き換えない。
- usage docs には「移行ガイド」という節を設けず、削除実装と同一 PR で「複数エージェントの一括
  生成は `prompt_slot_factory` + dict comprehension」を現在仕様として記載する。移行の before/after
  は本 ADR に記録済みである。

## Confirmation

本決定の遵守は実装 Issue 側の以下で検証する。

- テストスイート緑（`uv run pytest`・カバレッジ 80% 以上）。`tests/runtime/lightning/test_slots_l1.py`
  の `test_prompt_slots_*` 10 件を削除し、`tests/runtime/lightning/test_optimizer_l2.py` の
  `prompt_slots` 参照をファクトリ経路へ書き換える（エラーメッセージ変更に伴う assert 文言の追随を含む）。
- 公開 API スモーク（`__all__` 全件が import 可能であること）で、`prompt_slots` 削除後の
  `__all__` 整合を機械確認する。
- 残存確認: `grep -rn prompt_slots src/ tests/ docs/ examples/ README.md` の結果が `docs/adr/`
  配下（0005 / 0006 / 0008 / 本 ADR の記録）のみになること。
- `docs/QUALITY-GUARANTEES.md` に `prompt_slots` の参照は無い（grep 確認済み）ため、台帳行の
  追加・変更は不要である。
