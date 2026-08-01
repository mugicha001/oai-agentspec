# 0014: 次ターン開始エージェントの宣言的上書きと到達時ハンドオフ禁止の実現方式

- Status: accepted
- Date: 2026-07-31

## Context

SDK のマルチターン継続は `Runner.run(result.last_agent, ...)` の last_agent 継続が標準パターンである。
ハンドオフで専門エージェントへ遷移すると次ターンも専門エージェント起点になるため、「会話を設計上の窓口
（トリアージ等）へ戻す」判断がターン毎の分岐として呼び出しコードに散らばる。これを「ハンドオフ遷移を経て
エージェント X が回答を終えたら次ターンは Y から開始する」という宣言としてグラフ宣言と同じレベルで固定し、
run 完了結果から副作用なく解決できるようにしたい。あわせて「X にハンドオフで到達したターンでは X の全
handoff を無効化し、X 自身に回答を終えさせる」到達時ハンドオフ禁止をルール単位の opt-in で提供する。

満たすべき制約:

- build-don't-run（宣言・build-time 検証・薄い結線に徹し、次ターンの `Runner.run` は利用者コードが呼ぶ）
- SDK 隔離（`from agents` は `_adapters/` 配下のみ）
- エージェント実体の複製・差し替え・事後書き換えを行わない（registry 上の実体を単一に保ち、トレース上の
  管理対象を増やさない）
- 利用者 context を汚染しない・並行 run で干渉しない
- last_agent 継続の復元（禁止は当該ターンを越えて持続しない）

### SDK 読解で確認した前提

実装読解の対象は `agents.run` / `agents.run_internal`（`turn_preparation` / `turn_resolution` /
`run_loop` / `agent_runner_helpers`）/ `agents.handoffs` / `agents.run_context`。

1. `RunContextWrapper` は run 開始時に 1 回だけ生成され、同一インスタンスが handoff 実行
   （`on_invoke_handoff`）と handoff 有効性評価（`get_handoffs`）の双方へ渡る。plain な context / `None`
   を渡した場合は run ごとに新規生成される（並行 run が構造的に分離される）
2. `get_handoffs` はステップごとに「その時点の実行エージェント」に対して呼ばれる。handoff 実行は
   ステップ N の応答処理内で起き、到達先の最初の `get_handoffs` はステップ N+1 のため、到達した
   ターン内で無効化が効く
3. streaming 経路（`run_streamed`）も同一の `get_handoffs` を使う
4. `get_handoffs` は `is_enabled` を評価して無効な handoff を除外した列を返し、その列がモデルへの
   handoff 提示と agent span の `handoffs` 名になる（無効化はモデルへの非提示として観測できる）
5. `RunContextWrapper` は `eq=False` の dataclass であり identity hash・weakref が成立する
6. `handoff()` は `input_type` なしの `on_handoff` に 1 引数 `(ctx)`、`input_type` ありに 2 引数
   `(ctx, input)` を署名検証で強制する
7. 1 ステップで複数の handoff が要求されても実行されるのは先頭の 1 件のみである
8. `agent.handoffs` に Agent 実体を直接 append した場合、SDK は `get_handoffs` 内で毎ステップ
   `handoff(agent)` を新規生成する。この経路には `on_handoff` / `is_enabled` を合成できない
9. agent-as-tool 用の子 wrapper は新しいインスタンスであり、親 run の記録を共有しない

### 検討し却下した選択肢

1. **per-edge の遷移先差し替え（却下）**: 禁止対象の到達エッジについて、handoff の遷移先を
   `handoffs=[]` の clone インスタンスへ差し替えた構成を build 時に確定する。実行時状態を一切持たない
   利点があるが、同名の変形インスタンスが実行系に存在してシングルトン性が崩れ、トレース・管理対象が
   増える。`result.last_agent` が変形インスタンスを指すため組み立てヘルパ側に同名正規化が必要になり、利用者が
   生の `last_agent` で継続した場合の「元構成への復元」は「同名の別実体」という解釈に依存する。
2. **`RunResult.last_agent` の書き換え（却下）**: `last_agent` は setter を持たない読み取り専用
   property であり、書き換えは「実際に誰が回答したか」という観測事実を失わせる（route 採点・監査の
   材料が壊れる）。
3. **到達記録を利用者 context に置く（却下）**: 利用者定義の context オブジェクトに到達フラグを書く。
   利用者の名前空間を汚染し、同一 context を複数 run で使い回す形では記録が次の run へ漏れる。
4. **arm 式ワンショット無効化（却下）**: モジュールグローバルな「次の 1 回だけ無効」フラグを立てる。
   グローバル状態を持ち、並行 run では別の run の到達で誤発火する。
5. **`model_settings.tool_choice="none"`（却下）**: handoff 以外のツールまで殺すため「handoff 以外の
   ツール・応答生成は不変」という要求に反する。
6. **session 履歴の改変・ハンドオフ痕跡の除去（却下）**: 会話履歴は SDK `Session` の関心事であり
   lib は改変しない。
7. **`HandoffGraph.apply` / `_update_handoffs` への相乗り（却下）**: src 単位の replace 粒度であり、
   「同じ X への到達でも遷移元によって挙動を変える」per-edge の合成を表現できない。
8. **`observe_run_result`（`_adapters/routing.py`）の再利用（却下）**: 戻り値が `runtime.llmops` の型
   であり、コア層が参照すると「コアは runtime を import しない」単方向依存に抵触する。加えて属性
   読み出し自体が送出する例外を捕捉する契約を満たさない。防御的読み取りの idiom のみ踏襲する。
9. **1 ターン限りの handoff 無効化プリミティブの提供（却下）**: 下記 Decision 10 を参照。

## Decision

### 1. 到達時ハンドオフ禁止 = build 時判定表 + `is_enabled` ゲート

到達時ハンドオフ禁止は、(1) build 時に確定する判定表（どの `(遷移元, X)` 到達で X の handoff を
無効にするか）と、(2) SDK 公式拡張点への build 時合成で実現する。エージェント実体の複製・差し替え・
事後書き換えは行わない。

- **記録**: 判定表に載る流入エッジ（`src -> X`）の `HandoffConfig.on_handoff` に「到達記録の追記」を
  前置合成する。利用者宣言の `on_handoff` があれば「記録 -> 利用者 `on_handoff`」の chain とする
  （hooks chain helper と同型の前例 = ADR 0003）。SDK が生成した `Handoff` オブジェクトの事後
  書き換えは行わない。
- **ゲート**: X の全出辺の `is_enabled` に「記録参照ゲート」を AND 合成する
  （`X が記録済みなら False`、そうでなければ既存 `is_enabled`（bool / callable / async）の評価へ委譲）。
- ゲートは closure が捕捉した X の名前で判定し、`is_enabled` の第 2 引数に依存しない。

per-edge 差し替え方式（Context の却下案 1）は採らない。却下理由は「シングルトン性が崩れる」
「トレース・管理対象が増える」「生の last_agent 継続での復元が解釈依存になる」の 3 点。

### 2. 到達記録ストア（ArrivalStore）の設計

- 実体は `WeakKeyDictionary[RunContextWrapper, set[str]]`（`_adapters` 内部・公開しない）。キーは
  run ごとに新規生成される wrapper インスタンス（Context 前提 1・5）で、run 終了後は弱参照により
  解放される。次 run には記録が存在しないため元構成へ自動復元される。
- ストアは `apply_next_turn_policy` の呼び出しごとに独立生成し、派生 registry の合成 closure が共有
  する。モジュールグローバル状態を持たない。
- **arity 契約**: 記録用 callable は per-edge に arity を合わせて生成する。`input_type` なしのエッジは
  1 引数形 `(ctx)`、`input_type` ありのエッジは 2 引数形 `(ctx, input)`（Context 前提 6）。利用者宣言の
  `on_handoff` を chain する場合も同じ arity で透過する。動的エッジは lib が `on_invoke` を直接構築する
  ため SDK の署名検証の制約外だが、同じ 2 形へ揃える。
- 記録は**実際に実行された到達のみ**に発火する（Context 前提 7。1 ステップで複数 handoff が要求されても
  実行は先頭 1 件）。「到達」の意味論と一致する。
- **並行性**: `get_handoffs` は複数 handoff の有効性を並行評価するが、ゲート部は await を挟まない
  sync 参照のみでレースが発生しない。マルチスレッドから同一 registry / ストアを共有する形は
  registry と同一の「利用者責任」前提とする。
- **caveat（非対応）**: 利用者が `RunContextWrapper` インスタンスを自作して複数 run で再利用すると、
  同一キーのまま記録が run を跨ぐ。この形は非対応として docs に明記する。HITL 承認再開（resume）は
  本機能のスコープ外。agent-as-tool の子 wrapper は記録を共有しないため、サブ run へ禁止は漏れない
  （Context 前提 9）。

### 3. 配置 = コア直下 `next_turn.py` + `_adapters/next_turn.py`

宣言型・解決関数・組み立てヘルパ・結線関数はコア直下の `next_turn.py`（`agents` 非依存）に置く。
`await` も `Runner` 参照も持たない宣言 + 純関数であり、handoff 宣言と同じ宣言層の関心事のため、
extra 境界を持つ `runtime/` 配下には置かない。SDK 結合（観測抽出・記録合成・ゲート合成）は
`_adapters/next_turn.py` に閉じる。`next_turn.py` からの `_adapters` 参照は `registry.py` と同じく
関数内遅延 import とし、参照箇所を限定する。

### 4. 解決の判定材料を抽出する関数を `_adapters` に新設

run 完了結果からの観測抽出は `extract_turn_observation` として `_adapters/next_turn.py` に新設し、
plain な frozen dataclass（最終回答者名と `(遷移元, 遷移先)` の順序列）を返す。既存
`observe_run_result` は再利用しない（Context の却下案 8）。読み取りは属性の有無で判定し `type`
リテラルに依存しない防御的読み取りとし、属性欠落と属性アクセス時例外の双方を安全側（上書きなし）へ
倒して `logger.debug` に記録する。

### 5. 素の Agent 直 append エッジは Handoff オブジェクトへ昇格させる

判定表に載る（記録またはゲートを要する）エッジは、build 時に `make_handoff` 経由の `Handoff`
オブジェクトへ昇格させる（Context 前提 8）。`make_handoff` は指定フィールドのみを SDK `handoff()` へ
渡すため、昇格後の挙動は SDK が内部生成する `handoff(agent)` と同一で、既定挙動は変わらない。
昇格の対象は判定表に載るエッジに限る。

### 6. 規準名と実名の対応

要件書は規準名（意味論の識別子）を定め、実名の確定を設計に委ねている。対応は次の通り。

| 要件書の規準名 / 仮表記 | 実名 |
|---|---|
| `NextTurnPolicy` | `NextTurnPolicy`（同一） |
| 上書きルールの宣言単位 | `NextTurnRule` |
| 次ターン指定（仮表記 `force_next`） | `NextTurnRule.next_agent` |
| 到達時ハンドオフ禁止の opt-in | `NextTurnRule.no_handoff_on_arrival`（仮表記と同一） |
| 到達元条件（仮表記 `from_`） | `NextTurnRule.source` |
| `resolve_next_agent` / `next_turn_agent` | 同一 |
| 結線関数（要件書に規準名なし） | `apply_next_turn_policy` |

公開シンボルは `NextTurnRule` / `NextTurnPolicy` / `resolve_next_agent` / `next_turn_agent` /
`apply_next_turn_policy` の 5 つ。`next_agent` / `source` は Python 予約語との衝突（`from`）と
語義の明確さ（「強制」ではなく「次ターンの開始先」）を優先して仮表記から変更する。

### 7. registry フックの周辺契約

per-edge の合成は registry の内部プリミティブとして 4 点構造で持つ（すべて private）。

1. 静的エッジ結線（`_wire`）フック: 判定表に基づき流入エッジへ記録を前置合成し、X の出辺へゲートを
   AND 合成する
2. 動的エッジ生成（`_build_dynamic_handoff`）フック: 同一の判定表を同一の意味論で適用する
3. `clone` 継承: 判定表とストアは clone 先へ共有継承する。記録は wrapper キーで run 単位に分離される
   ため共有で安全であり、継承しないと clone 経由で禁止が静かに脱落する
4. 設置プリミティブの freeze ガード: 合成の設置は `_ensure_unfrozen` ガード付きのプリミティブ経由と
   する。frozen な元 registry に対しても `apply_next_turn_policy` は適用でき（派生 registry は
   unfrozen として生成される）、frozen registry 自身への事後設置は拒否される

既定（policy 未適用）では合成は一切設置されず、従来経路と同一である。

### 8. 「動的介入」の再定義

「ターン実行中の動的介入・SDK 内部 dispatch への割り込みを行わない」という制約が禁じるのは、
**宣言・構成そのものの実行時変更**と **SDK 内部 dispatch への割り込み**である。本方式が実行中に行うのは
「run 単位に分離された lib 内部ストアへの到達記録の追記」と「build 時に確定した判定表 + 記録の参照に
よる有効 / 無効の返却」のみで、判定表・宣言は実行中に変化しない。SDK 公式拡張点が評価する述語が
run 内の観測に依存すること自体は、禁じられた動的介入には当たらない。

### 9. build-don't-run 台帳上の位置づけ（逸脱一覧に追加しない）

ArrivalStore とゲートは実行を駆動しない（`await` しない・`Runner` を参照しない・独自の実行ループや
再試行を持たない）。SDK が呼ぶ公式 callback の内側で記録・参照するだけであり、hooks 合成（ADR 0003）と
同種の薄い結線である。したがって `fit_ml_estimator`（ADR 0004）/ `failsafe_call`（ADR 0012）のような
build-don't-run の逸脱には当たらず、逸脱一覧（現行 2 関数）には追加しない。逸脱の列挙数を
「実行を 1 回駆動する関数の数」として不変に保つための整理である。

### 10. 1 ターン限りの handoff 無効化プリミティブは提供しない

「エージェント実体を複製しない」「registry 上は単一実体」「run 開始前に run を識別する手段がない」の
3 制約下では、ターン開始エージェントに対する 1 ターン限りの無効化を lib の公開 API として実現する
手段が存在しない。代替 3 案はいずれも本 ADR が守る性質を壊すため却下する。

- 派生 registry を「開始時から無効」の構成で作る: 同名の 2 実体が並ぶ（シングルトン性が崩れる）
- 利用者 context にマーカーを置く: context 汚染 + 同一 context 再利用時に次 run へ漏れる
- arm 式ワンショット: グローバル状態 + 並行 run での誤発火

必要な利用者には SDK 標準の `Agent.clone(handoffs=[])` を直接使う案内を docs（`docs/usage/core/next_turn.md`）
に置く。これにより lib からの Agent 複製はゼロになる。

## Consequences

- + registry 上の実体は 1 つのままで、トレース・管理対象が増えない。無効化は agent span の `handoffs`
  名から当該 handoff が消える形で観測できる。
- + 次 run は新しい wrapper で記録が存在しないため、`next_turn_agent` のフォールバック経由でも利用者に
  よる生の `last_agent` 継続でも元の handoff 構成へ自動復元される。同名正規化が不要になり、
  `next_turn_agent` の非発動時は `result.last_agent` をそのまま返す簡素な仕様になった。
- + 純粋追加であり、宣言しない限り既存挙動は不変（判定表が空なら合成は設置されない）。
- - run 単位の内部状態（ArrivalStore）を持つため、「lib は状態を持たない」という説明に「run 内一時状態
  であり、ターン間・run 間には持ち越さない」という限定が付く。
- - SDK 前提への依存が増える（wrapper のインスタンス同一性・ステップ毎評価・無効 handoff のモデル
  非提示・`on_handoff` の署名検証）。SDK バージョン耐性トリップワイヤの監視対象を拡張する必要がある。
- - 判定表に載るエッジの内部表現が「素の Agent 直 append」から `Handoff` オブジェクトへ変わる
  （外部から見た挙動は同一）。
- - 利用者が `RunContextWrapper` を自作して複数 run で再利用する形は非対応になる。
- - registry に per-edge 合成の内部プリミティブ（4 点構造）が増え、その維持コストを負う。

## Confirmation

- 決定モデル（ハンドオフ経由 AND 回答者一致の発動条件・発動ルール選定 = 一致 `source` -> 包括 ->
  なし・禁止のみルールの「上書きなし」・防御的解決）の強制手段: `tests/test_next_turn.py`
  （`agents` 非依存の決定表）。
- 到達記録の record / lookup、`is_enabled` ゲートの AND 合成、`on_handoff` の arity（1 引数 / 2 引数の
  per-edge 選択）の強制手段: `tests/_adapters/` 配下の該当テスト。
- SDK 前提（run ごとの wrapper 新規生成・記録と参照の間のインスタンス同一性・到達ターン内での
  ステップ毎再評価・streaming 同経路・無効 handoff のモデル非提示・agent-as-tool 子 wrapper の非干渉）の
  強制手段: `tests/_adapters/` 配下の SDK 前提 pin テスト（バージョン耐性トリップワイヤ）。
- registry フック（判定表に基づく per-edge 合成・素 append の Handoff 昇格・`clone` 継承・
  設置プリミティブの freeze ガードと `_built` 破棄）の強制手段:
  `tests/_adapters/test_next_turn_registry_l2.py`（合成を設置しない既定経路は `tests/test_registry.py`）。
- SDK 隔離の強制手段: SDK 隔離 grep（`grep -rnE "(from agents|import agents)" src/oai_agentspec/ |
  grep -v _adapters` が空であること）。
- 上記のうちテストで表現する保証は `docs/QUALITY-GUARANTEES.md` に source = ADR-0014 として登録する。
