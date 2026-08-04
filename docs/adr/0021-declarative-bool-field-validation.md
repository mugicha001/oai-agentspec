# 0021: 宣言 dataclass の bool フィールドに構築時型検証を課す

- Status: accepted
- Date: 2026-08-04

## Context

宣言 dataclass の `bool` フィールドは型注釈が `bool` でも実行時に検証されない。値は truthiness で
評価されるだけのため、注釈に反する値を渡した宣言が例外もログも出さずに受理され、意図と異なる
挙動で確定する。実測した silent failure は次の 2 形である（発見契機は
`NextTurnRule.no_handoff_on_arrival`。到達時ハンドオフ禁止という安全制御フラグでも同じ特性だった）。

```python
NextTurnRule(next_agent="planner", no_handoff_on_arrival=None)  # 黙って OFF（結線ゼロ）
NextTurnRule(next_agent="planner", no_handoff_on_arrival="no")  # "no" が truthy で ON
```

前者は「宣言したのに効かない」、後者は env / 設定ファイル由来の `"false"` / `"no"` が意図と逆に
効く。いずれも例外・警告が出ないため、利用者は誤りに気付く手段を持たない。

同種のフィールドは lib 全体に散在する（安全・回復性・HITL・評価 fail-closed に効くものを含む）。
1 機能だけを是正すると既存フィールドとの間に新たな非対称が生まれるため、lib 全体としての方針を
決める必要があった。検討した選択肢は次の 2 つである。

1. **(a) 宣言 dataclass の全 bool フィールドに構築時型検証を入れる（採用）**: 型注釈を実行時契約
   として機能させ、silent failure を構造的に排除する。実装は共有ヘルパ 2 関数 + 各クラス 1〜2 行。
2. **(b) 検証は入れず docs で「bool 以外を渡さないこと」を明文化するのみ（却下）**: 安全制御フラグの
   silent failure が残る。「宣言・build-time 検証・薄い結線」というライブラリの原則、および
   silent gap / silent trap を構造的に排除してきた既存の設計判断（ADR-0017 / ADR-0018）と矛盾する。

実現手段として検討し却下した案:

- **Pydantic への移行（却下）**: Pydantic は型強制（coercion）を行うため、`"no"` を黙って変換する
  方向に働き「注釈違反を fail-fast させる」という目的に反する。重量依存の追加も最小依存方針に
  反する。
- **静的型チェッカの導入（却下）**: 型チェッカは非導入方針であり、かつ静的検査は env / 設定由来の
  実行時値を捕捉できない（上記の `"no"` は静的には検出されない）。
- **attrs の validator（却下）**: 新規外部依存の追加になる。stdlib dataclass の `__post_init__` で
  同等を達成できる。
- **実行時リフレクションによる自動検証（`fields()` 走査で bool 注釈を自動検証する汎用デコレータ。
  却下）**: `from __future__ import annotations` により注釈は文字列化され、`TYPE_CHECKING` 限定
  import を含む型（`CompactionConfig.client` 等）の解決が実行時に失敗しうる。暗黙の機構より明示
  1 行のほうが既存パターンと整合する。ただし**テスト時**のリフレクション（メタテスト）は網羅性の
  担保手段として採用する。
- **SDK 側の検証機構への委譲（却下）**: 対象は lib の宣言層 dataclass であり SDK は関与しない。SDK
  passthrough フィールドは `_adapters` が None-omission で素通しするのみで型検証しない。SDK 到達後の
  失敗は build 時ではなく実行時になり fail-fast 要件を満たさない。

## Decision

### 1. 対象スコープの規則

次の 3 条件をすべて満たす dataclass フィールドを構築時型検証の対象とする。対象フィールドの一覧と
メタテストの走査は**この同一規則**で閉じる（規則の外に個別の例外を作らない）。

1. `src/oai_agentspec/` 配下（**`_adapters/` を除く**）で定義される dataclass のフィールドである
   - `_adapters/` を除く理由: SDK 隔離境界の内側の実装詳細であり、lib 自身のみが構築して利用者の
     宣言が届かないため。宣言層の内部型（`SlotSegment` / `ConversationEntry` 等）は場所規則により
     対象に含む
2. `init=True` である（構築時に値を渡せる）
   - 除外例: `WorkflowGraph` の `_frozen`（`init=False`）。構築時に外部入力が届かず、構築時検証の
     対象になりえない
3. 型注釈が `bool` または `bool | None` である
   - 除外例: `HandoffConfig.is_enabled`（注釈 `Any`。bool / callable の二値ディスパッチとして
     ADR-0014 で確立済み）/ `ToolSpec.needs_approval`（注釈 `Any`。HITL 承認ゲートの
     bool / callable 二値ディスパッチで SDK へ委譲する）

規則の外にあるもの（本 ADR の対象外）:

- **関数 / メソッドの bool 引数**: 「宣言の受理」という適用点の意味論（宣言型は値を保持し後から
  効く / 関数は即時実行される）が異なるため対象に含めない。観点ファクトリの bool kwarg のように
  dataclass フィールドへ流入するものは、フィールド側の検証で間接的にカバーされる。一方
  `run_in_parallel` のように dataclass を経由せず SDK へ直行する bool kwarg は本規則の対象外で、
  無検証（truthiness 評価）のまま残る
- **TypedDict のキー**: dataclass ではなく `__post_init__` に相当する検証フックを持たない
- **Pydantic BaseModel のフィールド**: Pydantic 自身の検証規則に委ねる
- **通常クラスのインスタンス属性**: 宣言の受理点ではない
- **mutable 型の構築後の動的代入**: `metadata(name).enabled = False` のような動的トグルは確立済みの
  契約であり、これを検証するには `__setattr__` フックという新規機構が必要になる。構築時検証と
  本 ADR での限界明記に留める（決定 5 を参照）

### 2. 実装形 = 共有ヘルパ + 各 `__post_init__` からの明示呼び出し

`_validation.py`（stdlib のみ依存の最下層・共有バリデーションヘルパ）に `validate_bool` /
`validate_optional_bool` を追加し、対象クラスの `__post_init__` 冒頭（既存の整合検証より前）から
呼ぶ。`__post_init__` を持たないクラスには新設する。既存の build-time 検証パターン
（`__post_init__` + `isinstance` + `ValueError`）の踏襲であり、新規メカニズムはヘルパ 2 関数のみ。

両ヘルパは `_validation` 内部ヘルパであり公開しない（`__all__` 非掲載）。依存方向は「コア宣言層 /
runtime 各 extra -> `_validation`」の上向き参照のみで、単方向依存・SDK 隔離・extra 未導入耐性は
変わらない。検証は構築時に閉じるため、run 中のオーバーヘッドはゼロである。

### 3. 例外型は `ValueError`、メッセージは英語書式

- 例外型は `ValueError` とする。宣言 dataclass の build-time 検証は、型検査を含めて既に
  `ValueError` で統一されている。型の意味論としては `TypeError` も候補だが、同一クラス内で
  型起因 / 値起因が例外型で割れると利用者の捕捉が複雑になる。`TypeError` の前例
  （`chain_agent_hooks`・ADR-0017）は hooks 合成という別文脈であり、宣言 dataclass 系の規約は
  `ValueError` で維持する。
- メッセージ書式は `"{field} must be a bool, got {型名!r}"` /
  `"{field} must be a bool or None, got {型名!r}"` とする。`next_turn.py` の `_validate_name`
  （`"{label} must be a str, got ..."`）の前例を踏襲した英語書式であり、`_validation.py` 内の
  日本語メッセージのヘルパと言語が併存する。エラーメッセージ契約は各呼び出し文脈の既存前例に
  合わせる方針を採り、既存メッセージの一括変更は行わない。

### 4. 受理集合

- `bool` 注釈のフィールドは `True` / `False` のみ受理する。`0` / `1`（int）も拒否する
  （`isinstance(1, bool)` は False。bool は int のサブクラスだがその逆は成立しないため、strict な
  `isinstance` 判定で意図どおり弾ける）。
- `bool | None` 注釈のフィールドは `True` / `False` / `None` を受理する。None は「kwarg を渡さない
  = SDK 既定に委ねる」という確立済みの None-omission 意味論を持つ正当値である。

### 5. 適用点は構築時のみ

mutable な宣言型の構築後の属性代入（`ToolSpec.enabled` の動的トグル等）は検証しない。`__setattr__`
フックの追加は新規メカニズムであり、得られる保証に対して機構が重い。構築時検証と本節の限界明記に
留める。

## Consequences

- + 型注釈が実行時契約として機能し、注釈違反の宣言が構築時に fail-fast する。安全制御フラグの
  「宣言したのに効かない」silent failure が構造的に排除される。
- + 対象を機械的規則（場所 + `init=True` + 注釈）で定義したため、公開 / 内部の別による例外がなく、
  新しいフィールドが規則へ自動的に載る。規則からの脱落はメタテストで機械検知できる。
- - **破壊的変更**: 従来 `None` / 文字列 / int を渡して silent に truthiness 動作していた呼び出しが
  `ValueError` になる。影響を受けるのは「型注釈違反の入力で、現状すでに意図と異なる silent 動作を
  している」呼び出しのみで、正しい bool を渡している利用者には影響しない。0.x 系につき minor バンプ
  （0.4.0）とし、リリースノートに破壊的変更として明記する。
- - 対象クラスに `__post_init__` と 1〜2 行の検証呼び出しが増える（維持コスト）。
- - 構築後の動的代入は検証されないため、「宣言が型検証される」という説明に「構築時に限る」という
  限定が付く。
- 対象には build 時の宣言だけでなく、run 中に構築される型（`Detection` / `ObservedRun` /
  `CoverageReport`）も場所規則により含まれる。これらは検証の発火点が run 中になるため、truthy な
  非 bool（`re.Match` 等）を返す自作検知関数は実行時に `ValueError` を受ける。guardrail の
  tripwire 例外ではないため tripwire を捕捉するコードでは拾えないが、誤検知の握り潰しではなく
  停止させる fail-closed 方向であり、`bool(...)` の明示変換で解消できる。

## Confirmation

強制手段は次の 3 層で成立する。

- **網羅性の強制手段（メタテスト）**: `tests/test_bool_fields_l1.py`。Decision 1 の対象スコープの
  規則と同一の走査（`src/oai_agentspec/` から `_adapters/` を除外・`init=True`・注釈が `bool` /
  `bool | None`）で対象フィールドを抽出し、「非 bool 値の構築が `ValueError` になる」
  ことを parametrize 検証する。注釈は正規形の文字列へ落として判定し、文字列注釈と型オブジェクトの
  双方を同一に扱う（`get_type_hints` は `TYPE_CHECKING` 限定 import の解決に失敗しうるため
  使わない）。各クラスの正当 kwargs は fixture map で供給し、
  走査で発見されたクラスが map に未登録ならテスト自体が失敗する（新クラス追加時の検証漏れ =
  新たな非対称の機械検知）。
- **ヘルパ単体の強制手段**: `tests/test_validation.py`（`validate_bool` /
  `validate_optional_bool` の受理集合とメッセージ書式の pin）。
- **各クラスの強制手段**: 対象クラスのミラー位置のテストが拒否 / 受理ケースを持つ。
- 上記のうち網羅性の保証は `docs/QUALITY-GUARANTEES.md` に source = ADR-0021 として登録する。
