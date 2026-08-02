# 0015: ガードレールを名前・境界・分類・危険度の宣言として登録し名前で参照する

- Status: accepted
- Date: 2026-08-02

## Context

内容ガードレールの生成 helper（`runtime/guardrails` の factory 群）は SDK 互換の guardrail オブジェクトを返し、利用者はそれを `AgentSpec` の `input_guardrails` / `output_guardrails` 専用フィールドへ渡す。この経路では guardrail の識別子・適用境界・分類メタデータのいずれも宣言として保持されないため、利用側プロジェクトが次の 3 つを各自で再実装する状況が生じていた。

- **識別子の照合キーの自作**: 上流 SDK の guardrail 型は `name` が省略可能で、省略時は `get_name()` が guardrail 関数名を返す。表示名と照合キーが食い違うと、guardrail の指定・無効化がエラーも警告もなく無効化される（silent no-op）。同梱 factory は既定名を必ず渡すため `name` が `None` になることはないが、既定名は factory 単位の共有定数であり、同一 factory を複数回呼ぶと同名が重複する（一意性が強制されない）。
- **実体型からの境界推論**: 適用境界が宣言として存在しないため、消費側が `isinstance` で実体型を覗いて input / output を判別する。
- **framework 分類表の書き写し**: helper と OWASP LLM Top 10 等の対応が docs の Markdown 表にしか存在せず、機械可読データがない。

加えて同一の `AgentSpec` 内で参照方式が非対称だった。`handoffs` / `sub_agents` はエージェント名（str）で宣言できるのに対し、guardrail はオブジェクト参照しかできず、import していないものは宣言できない。

### 検討した選択肢

- **登録簿（入れ物）のみを追加する**: `dict[str, InputGuardrail | OutputGuardrail]` 相当を公開する案。却下。名前の一意性・境界・分類のいずれも強制されないため上記 3 つの再実装が 1 つも消えず、登録簿が 3 つ（agent / tool / guardrail）になる代償だけが残る。
- **利用者が生成した実体を登録簿へ後から載せる**: 登録キーと上流 SDK 可視名の一致を、登録簿が `dataclasses.replace` で事後に名前を差し替えることで担保する案。却下。上流 SDK の guardrail 型が非 frozen dataclass であることに依存し、上流の内部構造への結合が増える。
- **`AgentSpec.extra` 素通しで宣言する**: 却下。`extra` は `agents.Agent` の kwarg 素通しであり、str 名を実体へ解決する主体が存在しない。
- **上流 SDK の機構で代替する**: 却下。上流 SDK に guardrail の registry・名前解決・メタデータ機構は存在しない。guardrail 4 型は `guardrail_function` + `name` + `get_name()` のみを持ち、分類・危険度に相当するフィールドがない。`Agent` と `RunConfig` はいずれも `input_guardrails` / `output_guardrails` の 2 フィールドのみで、ツール境界のフィールドを持たない。
- **既存 `ToolRegistry` を流用する**: 却下（パターンの踏襲のみ採用）。`ToolSpec.func` が必須で `function_tool` 専用の build 経路を持ち、属性アクセス（`registry.<name>`）を前提とした識別子検証を課している。

## Decision

### 1. 登録簿を `runtime/guardrails` に置き、コアへは Protocol 1 個で結線する

宣言型（`GuardrailSpec` / `Boundary` / `Severity`）・分類データ・登録簿（`GuardrailRegistry`）はいずれも `runtime/guardrails` 配下に置く。登録簿が同梱 factory を代理呼び出しするため、コア層に置くと「コア層 → `runtime/guardrails`」の依存辺が生じ単方向依存に反する。

`AgentRegistry` は登録簿の実装型を import せず、`GuardrailProvider` Protocol（`protocols.py`・`AgentBuilder` と同一モジュール）でのみ受け取る。Protocol は「名前 → guardrail 実体」と「名前 → 宣言境界の str」の 2 照会のみを宣言し、戻り値に `runtime/guardrails` の実装型を用いない。

**Protocol の配置基準**: Protocol は、それを型注釈で受け取る側の層に置く。`GuardrailProvider` を受け取るのはコア層の `AgentRegistry` なのでコア `protocols.py` に置く。`runtime/intent/protocols.py` が層内にあるのは、その Protocol をコアが参照せず intent 層内でのみ受け取るためであり、基準は一貫している。

### 2. 生成と登録を同一呼び出しで行う（facade 経路）

登録簿は同梱 guardrail factory 9 個（agent 境界 8 + ツール境界 1）に対応する facade メソッドを持ち、実体の生成と登録を同一呼び出しで行う。登録キーを `name` として factory へ注入するため、**登録キーと上流 SDK 可視名（`get_name()` の戻り値）の一致が構造的に成立**し、事後の名前差し替えを持たない。同時に、登録簿がどの helper 由来かを知るため framework ラベルと既定危険度の自動付与が成立する。

利用者が用意した実体（生の上流 SDK guardrail・自作のもの）を名前参照へ載せるための逃げ道として `register(GuardrailSpec(...))` を残す。この経路では登録時に「上流 4 型のいずれかか」「可視名が登録キーと一致するか」「宣言境界と実体の境界が一致するか」を検証する。

`guard_tool` は facade の対象外とする。`FunctionTool` を受けて包んだ `FunctionTool` を返し guardrail を生成しないため、登録対象そのものが存在しない。

### 3. facade は対応 factory のシグネチャを写し、同期をテストで強制する

facade メソッドは `**kwargs` で受け流さず、対応 factory のシグネチャを写して型補完と引数エラーの即時検出を保つ。写しの同期は `inspect.signature(...).parameters` の突合テストで機械的に強制し、許容差分を `name` の必須化・`labels` の追加・`severity` の追加の 3 点のみに限定する。

### 4. 名前参照は単一フィールドで宣言し、専用フィールドと併存させる

`AgentSpec.guardrails: list[str]`（キーワード専用）に guardrail 名を宣言し、`AgentRegistry` が provider 経由で解決して宣言境界に従って `input_guardrails` / `output_guardrails` へ振り分ける。利用者は境界を再指定しない。

専用フィールド（オブジェクト直接指定）と名前参照は併存し、各境界のリストは「専用フィールドの既存要素の順序 → 名前参照由来の解決結果の宣言順」で連結する。重複排除は行わない（重複の是非は利用者判断）。

ツール境界の登録名を `AgentSpec.guardrails` および run 単位の境界別マッピングへ渡した場合は例外とする。`agents.Agent` にも `RunConfig` にもツール境界 guardrail のフィールドが存在せず振り分け先がないため、無言で落とさない。

### 5. `Severity` は `IntEnum`、`Boundary` は `str, Enum`（表現形の非対称）

危険度は順序比較が要件のため `IntEnum`（`LOW=1` / `MEDIUM=2` / `HIGH=3` / `CRITICAL=4`）とする。比較演算子による順序連鎖が言語標準で成立し、比較用の dunder を書かない。

境界は文字列リテラル互換が要件のため `str, Enum` とする。`str` を継承することで `Boundary.INPUT == "input"` が成立して文字列指定も等価に受理でき、かつ Protocol の 2 照会目の戻り値を `str` と注釈できる（コアが実装型を参照しない）。

却下した表現形:

- `str, Enum` で危険度を表す案 — 順序が辞書順（`critical < high < low < medium`）になり要件を満たさない。
- `str, Enum` + `__lt__` 系の自作 — `str` 継承のため混在比較（`"high" < Severity.CRITICAL`）が `str.__lt__` の辞書順になり非対称。また `functools.total_ordering` は補完手段にならない（`str` 継承により 4 つの比較 dunder すべてが `object` 既定と異なるため roots が全埋まりになり、補完対象 0 件でデコレータが実質 no-op になる）。
- 順位を返す関数・dict を経由する案 — 順序の観測手段を比較演算子に固定する要件に反する。

`IntEnum` の副作用への対処: 素の int との比較が成功するため、値域検証は `isinstance(value, Severity)` ガードで行い、登録時だけでなく危険度による絞り込み照会の引数にも適用する。また Python 3.11 以降の `IntEnum` は `__str__` が int 由来のため、docs・examples・例外メッセージでは `.name.lower()`（`medium` / `high`）で表記し、メンバを素で文字列へ埋め込まない。

### 6. 規準名と実装名の対応

要件定義の規準名に対し、実装では次の名前を採る。

| 規準名 | 実装名 | 理由 |
|---|---|---|
| `list()` | `specs()` | `ToolRegistry` の語彙（`names()` = 名前 / `metadata(name)` = 宣言 1 件）と揃え、全件宣言は `specs()` とする。あわせて `list` がクラス本文の名前空間で組み込みを遮蔽する形を避ける |
| `boundary` | `boundary`（変更なし） | facade 引数 `on` は 2 値、宣言が保持する境界は 4 値（ツール境界を含む）で値域が異なる。同名にすると `on` から境界への写像の存在が隠れる |

### 7. framework 分類データはコードを SoT とし、docs 表はその投影とする

helper 識別子 → framework ラベル + 既定危険度の対応データをコードに置き、docs の分類表はその投影とする。乖離はテストで双方向に検知する（対応データのキー集合が「同梱 helper 識別子 − 分類が DI 依存の helper」と一致すること・分類表の framework ラベルが対応データと一致すること）。

既定危険度は運用組織のポリシー判断の色が強いため、ライブラリが付す値は出発点であり、一覧照会による可視化と facade 引数による上書きを前提とする。この性質から既定危険度は docs 表との照合対象に含めない（照合するのは framework ラベルのみ）。

対応データに載せるのは helper 自体で適用境界と分類が固定される helper に限る。検知内容が利用者 DI で決まる helper（正規表現・述語・語彙・閾値・外部検知器・判定 prompt / model・任意の検知 callable を受けるもの）は分類が DI 内容で変わるため載せない。

### 8. 「内容ガードレール」節の記述の是正

`docs/architecture.md` の「内容ガードレール」節には「ガードレールの宣言経路は新設しない」「`AgentSpec.extra` 素通しでそのまま宣言できる」「`AgentSpec` にフィールドを足さない」という記述があったが、これは同じ節の他の箇所（「`AgentSpec` の `input_guardrails` / `output_guardrails` フィールドへ渡す」）と実装（当該専用フィールドが実在する）の双方に反していた。両者は同一のコミットに同居しており、当該判断を主題とする ADR も存在しない。したがって本 ADR は確立した判断の覆しではなく、実装と整合しない記述の是正（現在仕様の記述への差し替え）を含む。

### 9. 要件定義の受け入れ基準の解釈

- 受け入れ基準が登録時の拒否対象として agent 境界 2 型を挙げているのは「上流 guardrail 型でない実体（duck-typed オブジェクト・テスト用 Mock）」の例示であり、受理型の上限を定めるものではないと解釈する。境界の値域が上流 SDK の guardrail 4 型と 1 対 1 に対応する要件と整合させ、逃げ道の登録経路は 4 型すべてを受理する。
- 全件照会の名称は規準名 `list()` に対し実装名 `specs()` を採る（上記 6）。

### 10. `GuardrailSpec` は frozen とする

宣言 1 件を `@dataclass(frozen=True)` とし、フィールドの再代入を禁じる。登録時の検証（宣言境界と実体境界の一致・可視名の一致）を通った宣言が後から書き換えられると、検証済みの不変条件が失われる。境界を書き換えられた場合、出力境界の宣言が入力側へ結線され、対象を一度も検査しないまま一覧上は「登録済み」に見える。

frozen が禁じるのは属性の再代入のみで、`labels` は `dict` のためキー単位の更新は通る（宣言後のラベル追記を許す意図的な設計）。監査ラベルの完全性まで求める場合は不変マッピングでの保持が必要になるが、実利用の要求が確認できていないため採らない。

### 11. 登録時の例外型は `ValueError` に一本化する

上流 4 型は `name=None` を許し、可視名の取得は `guardrail_function.__name__` へフォールバックする。`functools.partial` や `__call__` を持つオブジェクトを guardrail 関数にした実体は型・境界の検証を通るが可視名を取得できず `AttributeError` になる。これを包まずに漏らすと「登録時の検証は必ず `ValueError`」という契約が崩れ、利用者が宣言不備を一様に処理できない。したがって登録キーを含む `ValueError` へ包む。

### 12. 解決元との突合は「実体が上流 4 型と判定できたときのみ」に限る

`GuardrailProvider` は duck-typed で、申告された境界の値域も実体の型も検証しない。build 時に実体から判定した境界と申告境界を突き合わせるが、突合は実体が上流 4 型と判定できたときに限る。Protocol の契約は「不透明型を返す」であり、判定不能を不一致として扱うと契約を実質「上流 4 型を返す」へ狭め、2 照会のみを実装した自作解決元を排除してしまう。申告境界が agent 境界でない場合は既定境界へフォールバックせず例外にする（fail-closed）。

### 13. 名前参照の解決は `AgentRegistry` 経由の build に限る

解決元は `AgentRegistry` が保持するため、registry を経由しない build 経路では名前参照を解決できない。評価・最適化が単体の宣言を対象に取る経路は registry を通らないため、名前参照を宣言した spec は無言で落とさず明示的な例外で拒否する。silent に落とすと「宣言した検査が存在しない対象」を測って結果を信頼してしまう（過剰依存）。専用フィールドへ実体を渡す経路は builder が転送するため同経路でも適用される。

### 14. `extra` 経由の宣言は専用フィールド衝突として拒否する

`guardrails` は上流 `Agent` の kwarg ではないが、専用フィールド名の集合へ加える。加えない場合は「SDK が受け付けないキー」という別理由で拒否されるため、拒否理由が `input_guardrails` / `output_guardrails` と非対称になり、上流に同名フィールドが追加された時点で検証を通って二重に渡る余地が生じる。

### 15. 誤宣言の問題行は位置で要素を区別する

`guardrails` に str 以外の要素が含まれる場合の問題行には、宣言リスト内の位置（1 始まり）を添える。型名だけでは同一型の不正要素が複数あるとき完全に同一の文言になり、どの要素が問題か特定できない。値そのものの repr を添える案は採らない（guardrail 実体の repr には関数オブジェクトのアドレスが含まれ、文言が実行ごとに変わってログもテストの pin も安定しない）。

### 16. 値域外の入力は fail-open にしない

宣言・照会の各入口で、値域外の入力を「静かに無視して既定へ寄せる」形を採らない。具体的には、照会引数の値域外は空リストではなく例外、run 単位へ束ねる照会でツール境界が含まれる場合は静かな除外ではなく例外、`guardrails` へ素の str を渡した場合は 1 文字ずつの反復ではなく例外、解決元が返す境界が str でない場合は型検査の前に値域外として例外にする。いずれも「掛けたつもりの検査が無い」状態を無言で成立させないための統一方針である。

## Consequences

### 得られるもの

- 登録キーと上流 SDK 可視名の一致が構造的に成立し、表示名と照合キーの食い違いによる silent no-op が生成時点で排除される。上流 SDK の dataclass 構造への結合も持たない。
- 適用境界が宣言として保持されるため、消費側が実体型を `isinstance` で覗く必要がなくなる。
- framework 分類が機械可読データになり、docs 表の書き写しが不要になる（helper 自体で分類が定まるものについて）。
- `handoffs` と同じ流儀で guardrail を名前参照できるようになり、同一宣言内の参照方式の非対称が解消される。
- 危険度が宣言として保持され、トリップ例外から guardrail 名を引いて危険度を取得できる。危険度に応じた着地の実装は利用者側に委ね、ライブラリは分岐機構を持たない。
- 既存のオブジェクト直接渡しは変更なしで動作し続ける。コア `__all__` のメンバ集合は増減しない。

### 払うコスト

- 登録簿が 3 つ（agent / tool / guardrail）になる。ただし guardrail 登録簿は runtime 層に置くため、コア層の登録簿は 2 つのままである。
- facade メソッド 9 個と対応 factory のシグネチャが二重に存在する。写しのずれは `inspect.signature` 突合テストで機械的に検知する。
- 名前参照は import 時の静的検査が効かないため、誤りの検出がオブジェクト参照より遅れる。この後退は build 時の明示例外と build-time の一括検証で埋め、無言の無視を行わない。
- 上流 SDK の結合点（`name` 注入が `get_name()` に反映されること・`get_name()` の存在・guardrail 4 型の型分離・run 単位設定の引数名）への追随責務が増える。前提はバージョン耐性トリップワイヤで pin する。
- 表現形の非対称（`IntEnum` と `str, Enum`）を利用者が学ぶ必要がある。順序と文字列互換という要件の違いに対応する。
- 自作の provider を注入した場合、実体の整合はライブラリが検証しない（provider 実装側の責務）。ただし境界値の扱いは振り分け側の責務として例外化する。

## Confirmation

強制手段は `docs/QUALITY-GUARANTEES.md` に登録した 5 行（source = ADR-0015）が指すテストである。

- facade メソッド集合が同梱 factory と一致し、シグネチャが許容差分 3 点のみで同期していること
- 分類データのキー集合が「同梱 helper 識別子 − 分類が DI 依存の helper」と一致していること
- facade 経路で宣言される適用境界が、生成された実体から判定した境界と一致していること
- 名前参照が `AgentRegistry` 経由の build でのみ解決され、registry を経由しない build 経路は例外で拒否すること
- docs 分類表の列構成・framework ラベル列・「利用者が宣言」の行集合がコード側の分類データと一致していること

上流 SDK 結合点の 4 前提はバージョン耐性トリップワイヤのテストで pin する。個別の assert とテスト名はテストの docstring を一次情報とし、本 ADR には列挙しない。
