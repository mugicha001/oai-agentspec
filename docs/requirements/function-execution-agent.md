# 関数実行エージェントの組み立て（runtime/intent 拡張）

## 1. 概要
判断を要さないアクションを、LLM を 1 回も呼ばずに確実な関数呼び出しへ落とすためのエージェント組み立てを提供する。既存の `DeterministicResponseModel`（`runtime/deterministic`）と `ToolRegistry` を結線し、入力 JSON をそのままツール引数へ渡す `AgentSpec` を宣言から自動で用意する。関数を直接呼ぶ実装と違い、SDK の HITL 承認・トレース・ツール統治（`metadata(name).enabled`）の経路に乗る点が本機能の価値である。実行そのものは利用側が `Runner.run` で行う（build-don't-run 方針の維持）。

本要件は「実行可能意図の宣言基盤」（`docs/requirements/executable-intent-declaration.md`）から分離したものである。分離の理由は、関数実行エージェントが意図予測とは独立した関心事であり、意図予測を伴わない文脈（ワークフローの終端・単発の確認付き操作など）でも単体で有用なためである。

## 1.1 この要件書の読み方

### 「実行可能意図の宣言基盤」との依存関係

| 方向 | 関係 |
|---|---|
| 本要件 -> 意図基盤 | **依存する**。`ActionSpec` / `ActionCatalog` / `param` の宣言を入力として実行エージェントを組み立てる |
| 意図基盤 -> 本要件 | **依存しない**。意図基盤の `ActionSpec.action_agent` は「利用者が `AgentRegistry` へ登録した任意のエージェント名」でよく、本要件が無くても成立する |

本要件は意図基盤の `action_agent` が指す先を**自動で用意する**位置づけである。利用者が判断能力のあるエージェント（LLM を使う業務エージェント）を `action_agent` に指定する構成は、本要件なしでそのまま動く。

### 解決したい使いにくさ

分離前の設計では、利用者が 1 アクションにつき次を手で書く必要があった。

```python
# アクション 1 件ごとに繰り返す（20 アクションなら 20 回）
registry.register(function_action_agent(
    "load_test_runner",                       # (1) ActionSpec.action_agent と同じ名前を再度書く
    spec=catalog.get("run_load_test"),        # (2) 自分で登録したものを取り出し直す
    tool=ToolSpec(func=run_load_test),        # (3) tool を組む
    tool_registry=tools))                     # (4) registry へ手で登録する
```

同じ名前を二度書き、自分で登録した宣言を取り出し直し、結線を手で組む。本要件はこれを宣言からの自動化で解消する（FR-2）。

### 機能要件の一覧

| FR | 追加するもの | 何のためか |
|---|---|---|
| FR-1 | 関数実行エージェントの組み立て | 判断不要のアクションを LLM 抜きで関数へ落としつつ、HITL 承認・トレース・ツール統治に乗せる |
| FR-2 | 宣言からの自動登録 | 利用者が結線を手で組まず、アクション 1 件あたりの結線行数を 0 にする |
| FR-3 | 宣言とツール関数の整合検証 | 宣言パラメータ名とツール引数名のずれを起動時に落とす |
| FR-4 | 公開窓口 | 既存の意図予測と同じ import 経路で使う |

含まないもの: アクション実行そのもの（利用側の `Runner.run`）。ツール関数の実装。意図予測・候補生成（意図基盤の担当）。

## 2. 機能要件

### FR-1: 関数実行エージェントの組み立て
- ユーザーストーリー: ライブラリ利用者として、判断を要さないアクションを LLM を介さず確実に関数へ落としたい。なぜならパラメータが確定しているのに LLM に引数を書き写させると誤記の余地が残り、かつ関数を直接呼ぶと HITL 承認・トレース・ツール統治をすべてバイパスするから。
- 受け入れ基準:
  - [ ] WHEN 対応する `ActionSpec` と `ToolSpec` とアプリ本体の `ToolRegistry` を与えて関数実行エージェントを組み立てる THEN `tool` を `tool_registry.register(tool)` で登録し（登録名は `tool.name`、省略時は `tool.func.__name__`）、`getattr(tool_registry, <登録名>)` で構築した SDK Tool 1 件を `tools` に持ち、`model` に `DeterministicResponseModel` を持つ `AgentSpec` を返す
  - [ ] IF 同名の tool が既に `tool_registry` に登録済み THEN 再登録せず既存の構築結果を使う（アプリ本体と別インスタンスの Tool を生まないことで `metadata(name).enabled` によるツール統治の経路に乗せる）
  - [ ] IF 登録名の重複を判定する THEN `tool_registry.register` を try/except せず `name in tool_registry.names()` で事前判定する（`register` は重複と不正名の双方で同じ `ValueError` を送出するため、捕捉すると不正名の違反を握り潰す）
  - [ ] WHEN 返された `AgentSpec` を registry へ登録し `Runner.run(agent, input=<実行入力 JSON>)` を **`session` を渡さずに**実行する THEN ルール関数は `ModelRequest.tool_outputs` が空なら `tool_call_response` で入力 JSON を引数として当該 tool を 1 回呼び、非空なら tool 実行結果を `text_response` として最終出力に載せる
  - [ ] IF 会話履歴つきで実行する必要がある THEN 関数実行エージェントを別 run として分離する（`ModelRequest.turn` / `tool_outputs` はいずれも入力全体から導出され run 単位ではないため、Session 併用時は前ターンのモデル応答・tool 実行結果が分岐条件を汚染し、ツールを 1 度も呼ばずに終わる）
  - [ ] WHEN 実行する THEN LLM 実行アダプタを 1 回も呼ばない
  - [ ] IF `tool` の `ToolSpec.needs_approval` が真値 THEN SDK の HITL 承認機構が作動する
  - [ ] WHEN ツール引数を組み立てる THEN 実行入力の JSON 文字列をそのまま `tool_call_response` の `arguments` へ渡し、f-string 等による組み立てを行わない（JSON 脱出事故を構造的に起こさないため）

### FR-2: 宣言からの自動登録
- ユーザーストーリー: ライブラリ利用者として、アクションを宣言したら実行エージェントが自動で用意されてほしい。なぜならアクションごとに同じ名前を二度書き、自分で登録した宣言を取り出し直し、結線を手で組むのは、アクション件数に比例した定型作業であり書き間違いの余地も残るから。
- 受け入れ基準:
  - [ ] WHEN アクション宣言に対応するツール関数を宣言する THEN 実行エージェントの登録に必要な利用者コードの行数はアクション 1 件あたり 0 行となる（宣言そのもの以外に結線コードを書かない）
  - [ ] WHEN アクションを宣言する THEN 実行先エージェント名を利用者が二度書く必要がない（宣言済みの名前を再入力しない）
  - [ ] WHEN 自動登録を行う THEN 対象は「ツール関数が対応づけられたアクション」に限り、`action_agent` が利用者の登録した別のエージェントを指すアクションは自動登録の対象外として素通しする
  - [ ] IF 自動登録先の名前が `AgentRegistry` に既に登録済み THEN 上書きせず `ValueError` を送出する（利用者が同名で登録した業務エージェントを黙って置き換えない）
  - [ ] WHEN 自動登録を行う THEN 登録は build 時に完結し、`Runner.run` を 1 回も呼ばない
  - [ ] WHEN 自動登録された実行エージェントを取り出す THEN 既存の `AgentRegistry.get(name)` で取得でき、専用の取り出し口を新設しない

### FR-3: 宣言とツール関数の整合検証
- ユーザーストーリー: ライブラリ利用者として、宣言したパラメータ名とツール関数の引数名がずれていることを起動時に知りたい。なぜなら実行時に初めて落ちると、利用者がボタンを押した瞬間に失敗するから。
- 受け入れ基準:
  - [ ] IF アクション宣言のパラメータ名の集合が `tool.func` のシグネチャの引数名の集合と一致しない THEN 差分を列挙した `ValueError` を送出する
  - [ ] WHEN 検証を行う THEN 標準ライブラリ `inspect.signature` で引数名を取得し、独自のシグネチャ解析を実装しない
  - [ ] WHEN 検証が失敗する THEN 実行時ではなく組み立て時（build 時）に送出する

### FR-4: 公開窓口
- ユーザーストーリー: ライブラリ利用者として、追加シンボルを既存の意図予測と同じ窓口から import したい。なぜなら import 経路を覚え直したくないから。
- 受け入れ基準:
  - [ ] WHEN `from oai_agentspec.runtime.intent import ...` を実行する THEN 本要件で追加した公開シンボルが取得でき、取得値は module 属性へキャッシュされる（既存の PEP 562 遅延再エクスポート方式に従う）
  - [ ] WHEN `import oai_agentspec.runtime.intent` を intent extra 未導入環境で実行する THEN 窓口の import 自体は成功する
  - [ ] WHEN 既存の公開シンボルを使う THEN 振る舞いは変更前と一致する（純追加）

## 3. 非機能要件

### NFR-1: 保守性（SDK 隔離）
- 要件: `agents` / `openai` への import は `_adapters/` 配下に限定し、本要件で追加するモジュールからは行わない。
- 計測基準: `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` の結果が空であること。

### NFR-2: 保守性（単方向依存）
- 要件: 追加モジュールからの参照は `runtime/intent` 内および上向きのコア（`registry` / `tool_registry` / `spec` / `constants`）と、同じ `runtime` 配下の `deterministic` 公開窓口（extra を持たないため intent extra の依存を増やさない。先例: `runtime/cli` -> `runtime/conversation`）に限定し、コアから `runtime` への新規依存辺を作らない。
- 計測基準: 本要件で追加するファイルの import 文が上記の範囲のみを指すこと。かつコア層のファイルへ `runtime` への新規 import を 1 件も追加しないこと。

### NFR-3: 保守性（既存公開契約の不変）
- 要件: 既存の公開シンボルの振る舞いを変更しない（純追加）。`tool_registry.py` / `spec.py` / `runtime/deterministic` は変更しない。
- 計測基準: `uv run python -c "import oai_agentspec as m; assert all(hasattr(m,s) for s in m.__all__)"` が成功し、コア `__all__` のメンバ集合が変更前と一致すること。既存テストは、公開窓口の `__all__` メンバ集合と件数を pin しているテストの期待値更新を除き、1 件も修正なしで通ること。

### NFR-4: 保守性（依存非膨張）
- 要件: `intent` extra の依存を `pydantic>=2` のみに保つ。
- 計測基準: `pyproject.toml` の `intent` extra の列挙が変更前と同一で、`uv.lock` に新規の外部パッケージが 0 件増えること（`inspect` は標準ライブラリ）。

### NFR-5: 性能（呼び出し回数）
- 要件: 関数実行エージェントの実行はモデル呼び出しを行わない。組み立てと自動登録は build 時に完結する。
- 計測基準: 関数実行エージェントの `Runner.run` について、履歴に `function_call_output` を含む入力に対してもツール呼び出しが 1 回起きること（`session` なし実行で分岐が汚染されないこと）をテストで固定する。組み立て・自動登録が LLM 実行アダプタを 0 回呼ぶことを固定する。

### NFR-6: セキュリティ（ツール統治と承認の経路）
- 要件: アプリ本体の `ToolRegistry` と別インスタンスの Tool を生まない。`ToolSpec.needs_approval` / `enabled` の宣言が実行時に効く経路を維持する。
- 計測基準: 同名 tool が既に登録済みのとき、`getattr(tool_registry, name)` が返す Tool が同一インスタンスであることをテストで固定する。`metadata(name).enabled` を偽へトグルすると当該 tool が無効化されることを固定する。`needs_approval` が真値のとき SDK の承認機構が作動することを固定する。

### NFR-7: 保守性（テストとリント）
- 要件: 追加コードを含めてカバレッジ基準を満たす。
- 計測基準: `uv run pytest` が `fail_under = 80` を満たして成功し、`uv run ruff check src/ tests/` と `uv run ruff format src/ tests/` が差分なしで通ること。

## 4. 制約事項
- 技術的制約: build-don't-run を維持する。本要件が提供するのは `AgentSpec` の組み立てと registry への登録までで、実行（`Runner.run`）は利用者が呼ぶ。実行を代行するヘルパーを提供しない。
- 技術的制約: 関数実行エージェントは `session` を渡さずに実行する。`ModelRequest` の `turn` / `tool_outputs` は**入力全体から導出され run 単位ではない**ため、履歴を伴うと前ターンのモデル応答・tool 実行結果でルール関数の分岐が汚染され、ツールを 1 度も呼ばずに終わる。この制約は実装読解で確認済みである（`_adapters/deterministic.py` の `_collect_tool_outputs` が `_input_items(model_input)`、すなわち SDK が渡した入力全体を対象とする。`ModelRequest.turn` の docstring も「Session を使う構成では前ターンまでの応答も数に入る」と明記する）。
- 技術的制約: 実行パラメータは `Runner.run` の `input` で渡し、`context` 経由にしない。`DeterministicResponseModel` のルール関数へ渡る `ModelRequest` に **run context のフィールドが無い**ため（フィールドは `system_instructions` / `input` / `user_text` / `turn` / `tool_outputs` / `model_settings` / `tools` / `handoffs` / `output_schema` の 9 件のみ）、`context` 経由にすると関数実行エージェントだけ別経路になる。
- 技術的制約: ツール引数は実行入力の JSON 文字列をそのまま `tool_call_response` の `arguments` へ渡す。f-string 等による組み立ては JSON 脱出事故を招くため行わない（`_adapters/deterministic.py` の docstring が同趣旨の警告を持つ）。
- 技術的制約: `ToolRegistry.register` は重複登録時に `ValueError` を送出するが、名前が不正な場合（`_` 始まり / 非識別子 / Python 予約語 / 公開メソッド名衝突）も**同じ `ValueError`** を送出する。したがって重複判定を try/except で行わず、`name in tool_registry.names()` の事前判定で行う。登録名の解決規則（`spec.name` があればそれ、無ければ `spec.func.__name__`）は `ToolRegistry` 側と同一にする。
- 技術的制約: 新規のツール統治機構・承認機構を作らない。`ToolSpec.needs_approval` / `enabled` の既存宣言と `ToolRegistry.__getattr__` のキャッシュ機構をそのまま使う。
- 技術的制約: 環境変数を参照しない（env 参照は `runtime/cli` 境界に閉じる既存規約に従う）。
- ビジネス制約（スコープ外）: ツール関数の実装。アクションの宣言そのもの（「実行可能意図の宣言基盤」の担当）。候補生成・意図予測・パラメータ予測。実行結果の後処理・整形。
- ビジネス制約: 本ライブラリは未リリースの Alpha であり後方互換を必須としないが、既存公開シンボルの振る舞いと `__all__` のメンバ集合の変更は実装前にユーザー合意を取る。

## 5. 影響範囲
- 関連コンポーネント: `src/oai_agentspec/runtime/intent/`（関数実行エージェントの組み立てと自動登録の新規モジュール、`__init__.py` の公開窓口への追加）、`src/oai_agentspec/runtime/deterministic`（参照のみ・変更なし）、`src/oai_agentspec/tool_registry.py`（参照のみ・変更なし）、`src/oai_agentspec/spec.py`（参照のみ・変更なし）、`tests/runtime/intent/`、`docs/architecture.md` の意図予測の節、`docs/adr/`、`docs/requirements/`、`docs/QUALITY-GUARANTEES.md`、`examples/intent/`。
- 既存機能への影響: 純追加であり既存の公開契約・挙動を変更しない。`ToolRegistry` / `AgentSpec` / `runtime/deterministic` はいずれも参照するのみで変更しない。`pyproject.toml` の依存宣言も変更しない。
- 依存する要件: 「実行可能意図の宣言基盤」（`docs/requirements/executable-intent-declaration.md`）の `ActionSpec` / `ActionCatalog` / `param` と、実行入力の JSON（`plan.input_json`）。本要件は当該基盤の第 1 段が完了していることを前提とする。
- 依存されない関係: 当該基盤は本要件なしで成立する（`ActionSpec.action_agent` に利用者が登録した任意のエージェント名を指定すればよい）。したがって本要件は当該基盤のリリースを妨げない。
- 再利用する既存資産: `DeterministicResponseModel` / `ModelRequest` / `tool_call_response` / `text_response`（`runtime/deterministic`・extra 不要）、`ToolRegistry` / `ToolSpec`（登録・遅延ラップ・`enabled` トグル・`needs_approval`）、`AgentSpec`（`model` / `tools` / `handoffs`）、`AgentRegistry`（登録・遅延構築）、標準 `inspect.signature`。新規機構は追加しない。

## 6. 用語定義
| 用語 | 定義 |
|------|------|
| 関数実行エージェント | `DeterministicResponseModel` により入力 JSON をそのままツール引数へ落とすエージェント。LLM 呼び出し 0 回で、SDK の承認・トレース・ツール統治の経路に乗る |
| ルール関数 | `DeterministicResponseModel` に渡す関数。`ModelRequest` を受けて `ModelResponse` を返し、`tool_outputs` の空 / 非空で「ツールを呼ぶ」「結果を最終出力に載せる」を分岐する |
| 自動登録 | アクション宣言とツール関数の対応から、実行エージェントの `AgentSpec` を組み立てて `AgentRegistry` へ登録するまでをライブラリが行うこと。利用者はアクション 1 件あたり結線コードを書かない |
| 登録名の解決規則 | `ToolSpec.name` があればその値、無ければ `ToolSpec.func.__name__` を `ToolRegistry` の登録キーとする規則。本要件は同一規則を用いて重複判定を行う |
| ツール統治 | `ToolRegistry.metadata(name).enabled` の動的トグルが次回 run から SDK ネイティブの `is_enabled` 経由で効く経路。関数実行エージェントがアプリ本体と同一の Tool インスタンスを使うことで成立する |
| build-don't-run | 宣言・build 時検証・薄い結線に徹し、実行は SDK `Runner.run` に寄せる本ライブラリの原則。本要件は実行を 1 回も駆動しない |
