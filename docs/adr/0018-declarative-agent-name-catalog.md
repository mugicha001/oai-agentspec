# 0018: エージェント名をクラス属性の宣言として持ち整合検査を独立関数で行う

- Status: accepted
- Date: 2026-08-04

## Context

`AgentSpec` はハンドオフ先・サブエージェント・動的ハンドオフ候補・次ターン規則の到達元 / 遷移先を
すべて「エージェント名の str」で宣言する。同じ文字列リテラルを複数箇所へ書き写す運用ではタイポが
`AgentRegistry.validate()` / `get()` の実行時検出まで発覚せず、mypy 非導入のため静的型チェックによる
前倒しも効かない。したがって「記述時に気付ける状態」を型チェッカ以外の 3 点
（宣言済み名の静的属性解決と `dir()` 掲載 / 未宣言名アクセスの即時 `AttributeError` /
到達不能名のクラス定義時 `ValueError`）で作る必要がある。

宣言形式は「静的解析可能なクラス属性宣言」であること、値が `str` として既存の名前参照フィールドへ
変換なしに渡せること、追加依存を持たず `agents` / `openai` 非依存のコア最下層に置けることが前提となる。

### 検討した選択肢

- **`enum.StrEnum`（却下）**: 値が `str`・静的属性解決・`dir()` 掲載・追加依存なしは満たすが、
  (1) 未定義メンバアクセスが既定文言の `AttributeError` になり「宣言済み名の一覧」を出すには
  `EnumMeta` 派生の独自メタクラスが要るため「標準機能で済む」前提が崩れる、(2) **同一値のメンバが
  暗黙のエイリアスへ静かに畳まれる**（`A = "x"` / `B = "x"` で `B` が `A` の別名になり反復・
  `_member_map_` から消える）ため、名前の取り違えを検出したい目的と逆行する。
- **`typing.Final` + モジュール定数（却下）**: 宣言集合を機械的に取得する境界がなく（モジュール属性
  走査では import 済みの他シンボルが混ざる）、未宣言名アクセスで一覧を出す拡張点も、定義時検査の
  置き場も持たない。名前空間がフラットで「どのアプリのエージェント名か」を束ねられない。
- **`types.SimpleNamespace`（却下）**: 名前を実行時のコンストラクタ引数で持つ形であり、宣言形式の
  前提（静的なクラス属性宣言）から外れる。定義時検査・未宣言名メッセージの拡張点も持たない。
- **既存 `ToolRegistry` の流用（部分採用）**: 全体は却下。宣言が実行時 `register(ToolSpec)` であり、
  `__getattr__` が宣言済み名の**解決**にも使われる（本 ADR は `__getattr__` を未宣言検出専用と
  規定する）。到達不能名の判定規則 4 分岐・未登録名メッセージの単一ソース体裁・予約集合の明示宣言形・
  `_` 始まり属性の素通しは同型で流用する。
- **既存 `GuardrailRegistry` の流用（却下）**: 名前検証が「非空白 str のみ」で、属性アクセスを
  提供しない前提のため識別子制約を課していない。定義時検査・静的属性解決のいずれも満たさない。
- **整合検査を `AgentRegistry.validate()` の第 3 群にする / 定数簿の classmethod にする（却下）**:
  前者は `AgentRegistry` から名前定数簿への依存辺を生み、定数簿を使わない利用者の経路に定数簿の
  概念が漏れる。後者は利用者が宣言したクラスに registry 依存の検証 API が生えて「宣言のみ」の
  性質が薄れ、予約属性名が 1 つ増える。

### 同一値の重複宣言を拒否する理由

`PLANNER = "planner"` と `PLANER = "planner"` の併存は、本機構が防ごうとしているタイポの一類型で
ありながら、拒否しなければ**どの検出網にも掛からない**。属性アクセスは両方成功し、整合検査も宣言側は
値の集合になるため差分 0 件で通る。加えて `StrEnum` を「同値メンバが暗黙エイリアスへ畳まれる」ことを
理由に却下した以上、自前実装で同じ状態を許すと却下理由の一貫性が崩れる。`names()` が「宣言された値の
昇順リスト」を返す設計とも噛み合わない（重複を許すと宣言属性数と一覧件数が対応しなくなる）。

## Decision

エージェント名の宣言を、メタクラス付き基底クラス `AgentNames`（`src/oai_agentspec/agent_names.py`）
の**クラス属性宣言**として持つ。実行時 import は `keyword` と `typing` のみで、`agents` / `openai` に
依存しないコア最下層のリーフとする。

### 1. 宣言はクラス属性・検査はクラス定義時

利用者は `class Names(AgentNames): PLANNER = "planner"` の形で宣言する。メタクラスの `__new__` が
クラス定義時に namespace を走査し、次を `ValueError` で拒否する。

| namespace のエントリ | 扱い |
|---|---|
| dunder（`__module__` / `__qualname__` / `__doc__` / `__annotations__` 等） | 検査・宣言集合とも除外（除外しないと docstring 付きクラスが定義できない） |
| 基底 `AgentNames` 自身の生成 | 検査を skip（`bases` に メタクラスのインスタンスを含まない呼び出し） |
| 単一 `_` 始まりの非 dunder | `ValueError`（到達不能名規則に揃える） |
| callable / classmethod / staticmethod / property / 型オブジェクト | `ValueError`（宣言専用の性質を守る） |
| 注釈のみの宣言（`PLANNER: str`） | `ValueError`（属性が生えず参照が `AttributeError` になる silent trap を防ぐ） |
| 非空の str 値 | 採用。属性名に到達不能名規則 4 分岐、値に「非空 str」を課す |

到達不能名規則の 4 分岐（非空 + `str.isidentifier()` / `_` 始まり禁止 / `keyword.iskeyword()` 禁止 /
予約属性名との衝突禁止）は `ToolRegistry._validate_name` と同一規則とし、規則の SoT は
`ToolRegistry._validate_name` 側に置いて docstring から相互参照する（共通化のために既存 private
メソッドを動かすリスクは取らない）。

予約属性名は `frozenset({"names", "mro"})` の明示集合とする。`mro` を含めるのは、`type.mro` が
非データ記述子でクラス属性に隠され、`cls.mro()` による introspection が壊れるためである。

### 2. `__getattr__` は未宣言検出専用

宣言済み名はクラス属性として静的に解決され、`__getattr__` を通らない。未宣言名のアクセスは
宣言済み**属性名**の一覧を含む `AttributeError` にする（属性アクセスの失敗なので、利用者が書いた
識別子で照合できる形が有用）。`_` 始まりの名前は素の `AttributeError` で素通しし、`inspect` /
`copy` / `pickle` / pytest の内部プロトコル探索へ誤誘導メッセージを返さない。

### 3. 継承は許可し宣言集合を MRO 集約する

多段継承（`class Sub(Names): ...`）を許可し、`names()` は MRO 集約後の宣言値の昇順リストを返す
（`dir()` が親の属性を含むため、集約しないと `names()` と `dir()` が食い違う）。

### 4. 同一値の重複宣言はクラス定義時に `ValueError`

MRO 集約後の `{属性名: 値}` において、同一値が 2 つ以上の**異なる属性名**へ割り当てられていたら
`ValueError` にする（同一属性名の override は 1 名前として扱うため通る）。理由は Context に記した
3 点。属性名と値が異なる宣言（`PLANNER_V2 = "planner-v2"`）は引き続き許容し、拒否対象が「値の重複」
だけであることをエラーメッセージで明示する。

トレードオフとして「2 つの論理ロールを同一エージェントへ意図的に別名付けする」用途は塞がれる。
定数簿は opt-in であり、必要な場合は当該箇所を生 str へ落とせるため許容範囲と判断する。

### 5. 整合検査は独立関数 `validate_agent_names(names, registry)`

定数簿と同一モジュールにモジュール関数として置き、コア `__all__` へ載せる。`AgentRegistry` からは
既存公開 API の `registry.names()`（spec と factory の和集合）のみを呼び、型注釈は `if TYPE_CHECKING:`
配下でのみ import する。これにより `AgentRegistry` は名前定数簿を知らず、定数簿を使わない利用者の
経路は不変に保たれる。

報告は「定数簿に宣言済みだが registry へ未登録」「registry へ登録済みだが定数簿に未宣言」の
両方向の差分を全件集約し、単一の `KeyError` で送出する（群ごとに接頭辞を付け `"; "` で連結し、
群が 2 つ揃うときのみ `" | "` で連結する `AgentRegistry.validate()` と同型の体裁）。差分 0 件なら
例外を送出しない。

### 6. 例外メッセージの言語は踏襲元のトピックに揃える

属性アクセス・識別子規則に属する `ValueError` / `AttributeError` は英語（`ToolRegistry` 踏襲）、
集約報告の `KeyError` は日本語（`AgentRegistry.validate()` 踏襲）とする。モジュール単位で統一する
のではなく、同一トピックの既存メッセージと揃えることを基準にする。

### 7. 定数簿は opt-in の追加手段

生 str による宣言、`AgentRegistry.validate()` / `get()` の実行時検出、`next_turn` の集約報告は
一切変更しない。`spec.py` / `handoffs.py` / `next_turn.py` / `registry.py` は無変更で、定数の値が
`str` であることによって既存フィールドがそのまま受ける。

## Consequences

- + エージェント名の SoT が 1 箇所のクラス本体に集約され、以降の参照が定数経由になる。タイポは
  未宣言名アクセスの即時 `AttributeError`、到達不能名・値の重複のクラス定義時 `ValueError`、
  定数簿と registry の両方向差分の単一 `KeyError` の 3 点で捕捉できる。
- + 定数の値は `str` のため、既存の `list[str]` / `dict[str, ...]` フィールドは無変更で受ける。
  `AgentSpec` の位置引数束縛契約にも影響しない。
- + 整合検査が独立関数のため、`AgentRegistry` にも定数簿にも相手側への依存辺が生まれない。
- - メタクラス方式はスタックトレースが読みづらく、クラス定義時の失敗はモジュール import 時に起きる。
  緩和として、エラーメッセージへ必ず「どの属性名 / どの値が問題か」を含め、`_` 始まりを素通しして
  introspection への誤誘導を避ける。
- - 到達不能名規則の 4 分岐が `ToolRegistry._validate_name` と本モジュールの 2 箇所に存在する。
  既存 private メソッドを動かす共通化はリスクが利益を上回るため、docstring の相互参照で運用する。
- - 同一エージェントへ意図的に複数の別名を付ける用途は定数簿では表現できない（生 str へ落とす）。
- - 定数簿は opt-in のため、定数簿を通さない登録が混ざれば「1 箇所宣言」の前提は崩れる。この検知が
  整合検査の役割であり、run 前に呼ぶかどうかは利用者の判断に委ねる。

## Confirmation

強制手段として次のテストを置き、`docs/QUALITY-GUARANTEES.md` へ登録する（source = ADR-0018）。
個別 assert とテスト名の確定はテスト実装時に行い、一次情報は各テストの docstring とする。

- 未宣言名アクセスが宣言済み名の一覧つき `AttributeError` になること: `tests/test_agent_names_l1.py`
- 到達不能な宣言（4 分岐 / 非 str / 注釈のみ / 値の重複）がクラス定義時に `ValueError` になること:
  `tests/test_agent_names_l1.py`
- 整合検査が両方向の差分を単一 `KeyError` で全件報告し、`register_factory` 登録名を未登録と
  誤検知しないこと: `tests/test_agent_names_l1.py`
- 定数と生 str で同一の `Agent` 構成（`handoffs` の並び・`tools` の並び）になること:
  `tests/test_agent_names_refs_l1.py`
