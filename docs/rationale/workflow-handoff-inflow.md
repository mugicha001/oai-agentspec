# Rationale: ワークフロー handoff 流入経路の設計経緯

本ファイルは不変な検討経緯（immutable）を保持する archival ドキュメントである。
現在の確定仕様は `docs/architecture.md` を参照すること。本ファイルは実装変更に追随して更新しない。

## ノード/エッジ明示宣言を採る理由

WorkflowGraph は LangGraph（`StateGraph`）/ Microsoft Agent Framework（`WorkflowBuilder`）に倣い、
`add_agent_node` / `add_function_node` でノードを、`add_edge` / `add_conditional_edges` /
`add_fan_in_edge`（+ `START` / `END` 番兵）でエッジを明示宣言する。チェーン糖衣（sequence / parallel /
branch / loop）や位置依存の暗黙ルールは採らない。

採用理由:

- 制御フロー（順次 / 並列 / 分岐 / 合流 / ループ）をすべて明示エッジで表すことで、宣言の位置や順序に依存
  する暗黙ルールを排除する。fan-out は同一 src からの複数エッジ、fan-in は `add_fan_in_edge`、分岐は
  `add_conditional_edges`、ループは戻りエッジ + 条件エッジ（`recursion_limit` で上限）として、糖衣 API を
  介さず一様に表現できる。
- 2 つの主要グラフフレームワーク（LangGraph / MS Agent Framework）の語彙・形に準拠することで、利用者の
  既存知識を流用でき学習コストを下げる。データ運搬は MS Agent Framework のメッセージ受け渡し型（ノード出力
  が出辺に沿って下流へ流れる）に倣い、LangGraph の reducer 付き共有 State は採らない（独立 state 機構を
  新設しない）。合流は `{source名: 出力}` の dict を受ける FUNCTION ノードで利用者が書き、reducer を API 化
  しない・位置依存 list を作らない。

## 背景となる SDK 制約

ワークフローを handoff の流入先にするには、次の SDK 仕様が前提となる。

- handoff の流入先は必ず `Agent` でなければならない（SDK: `on_invoke_handoff -> Awaitable[TAgent]`、
  "Can't handoff to a non-Agent"）。宣言 dataclass の `WorkflowGraph` をそのまま handoff ターゲットに
  はできない。流入には「Agent 化したワークフロー」が要る。
- ファサード（tool だけを持つ Agent）に handoff した場合、ファサード Agent は SDK 仕様上 LLM を必ず
  1 回呼ぶ。tools だけで LLM をバイパスする正攻法は存在しない。`tool_choice='required'` は逸脱頻度を
  下げるのみであり（`reset_tool_choice=True` のため初回ターンにしか効かない）、起動の決定性は保証
  されない。
- `agents.Model.get_response` は run context を引数に取らず、SDK は run context を contextvar で公開
  していない（実機検証済み）。よってカスタム Model はワークフロー内ステップへ外側 context を渡せない。

## 流入経路の選定（経路C 主軸・経路A/D 補完）

決定性・外側 context 伝播・LLM 層数・エンジン制御の 4 軸で 3 経路を比較し、次のトレードオフから
経路C を主軸、経路A をその補完（context 透過の逃げ道）と位置づけた。

### 経路C: WorkflowModel を据えた Agent（採用・主軸）

カスタム Model（`WorkflowModel`）は LLM を呼ばず内部インタプリタを回し、最終出力を `ModelResponse`
（単一メッセージ・tool/handoff なし）として返す。Runner はこれを最終出力として扱いターンを終える。

採用理由:

- ワークフローが handoff 流入時に**決定論的に起動する**（ファサードの「LLM 必ず 1 回」非決定性を
  負わない）。
- 外から見て 1 つの「本物の Agent」であり、`HandoffGraph.edge(src, name)` の既存経路にそのまま乗る
  （新しい handoff 経路を発明しない）。
- 継続は外から見てアトミックな 1 Agent の再実行（`RunResult.last_agent`）に閉じ、途中再開を持たない
  ことが build-don't-run の線引きと一致する。

トレードオフ（受容したハード制約）:

- `Model.get_response` が context を受け取れないため、外側 run の共有 context はワークフロー内ステップ
  へ伝播しない。これは SDK 由来のハード制約であり回避策はない。context が必要な場合は経路A を使う。
- `stream_response` はエンジン完了後に最終出力をイベント化して流す post-execution streaming
  （`Runner.run_streamed` 対応。エンジンが最終値を返す構造のため進捗的ではない）。

### 経路A: as_facade_spec（採用・補完）

ワークフロー tool だけを持つファサード Agent に handoff し、ファサードの LLM が tool を呼ぶことで
`on_invoke_tool` クロージャ内でエンジンが回る。

採用理由:

- `on_invoke_tool(tool_context, json)` から `tool_context.context` を内部インタプリタへ受け渡せるため、
  外側の共有 context がワークフロー内ステップへ伝播する（経路C との差別化点）。これが経路C の context
  非伝播に対する唯一の正攻法の逃げ道である。

トレードオフ（受容した非決定性）:

- ファサード Agent が SDK 仕様上 LLM を必ず 1 回呼ぶため、起動の決定性は保証されない。

### 経路B: raw Handoff（既存機能で表現）

エントリ Agent へ直接 handoff する最軽量経路。エンジン制御は得られないが、既存
`HandoffGraph.edge(src, entry_agent)` で表現でき新 API を要しない。

### 経路D: 決定論ファサード（経路C の context 制約の補完）

経路C は決定論起動できるが、`Model.get_response` が context を受け取れない SDK 制約により
ワークフロー内ステップへ外側 context を透過できない（上記「背景となる SDK 制約」）。経路A は
tool 経由で context を透過できるが、ファサード Agent が実 LLM を呼ぶため起動が非決定になる。
「決定論を保ったまま context を透過したい」象限はどちらも埋められなかった。

経路D は経路A と同じ tool ファサード機構のまま、入口モデルを実 LLM ではなく決定論ステートレス
モデル（`DeterministicToolCallModel`）に差し替える。毎回ワークフロー tool を強制発火するため実
LLM 0 回・決定論で起動し、tool 経由なので `tool_context` から context が内部ステップへ透過する。
入口モデルの可変化は `FacadeMode`（`LLM_INPUT` / `LLM_INPUT_OUTPUT` / `DETERMINISTIC`）で表す。

トレードオフ: 経路C と異なり tool 往復 1 回ぶんのアイテム生成（session 履歴蓄積）が増えるため、
context 不要なら経路C（最軽量・履歴クリーン）、context が要るなら経路D（決定論を維持）という
補完関係に置く。経路C の上位互換ではない。

## tool_choice を ModelSettings に置く理由

経路A のファサードは `tool_choice='required'` を必要とするが、`tool_choice` は `agents.Agent` の
フィールドではなく `agents.ModelSettings` のフィールドである。`AgentSpec.extra` に積むと `build_agent`
の未知キーガードで `ValueError` になるため、必ず `model_settings=ModelSettings(tool_choice='required')`
経由で設定する。一方 `tool_use_behavior='stop_on_first_tool'` は `Agent` フィールドであるため `extra`
で渡せる（両者は非対称）。この配置は SDK のフィールド帰属に従った帰結であり選択の余地はない。

## context 非伝播の SDK 根拠

経路C で外側 context が伝播しない根拠は `Model.get_response` のシグネチャに context 引数が無く、SDK が
run context を contextvar で公開していないことにある（実機検証済み）。docs/examples では「context 要否
で A/C を選ぶ」という経路選択指針を明示し、利用者の誤用（経路C でステップに context が届くと誤認する
事故）を防ぐ。

## Runner.run 一本化の理由

実行口を SDK の `Runner.run` 一本に寄せ、公開の `WorkflowEngine.run()` を提供しない。

採用理由:

- 実行口を増やさず、ワークフローも他の Agent と同じ `Runner.run` で走る（使用負荷の最小化）。
- lib の責務を「宣言 + build-time 検証 + 実行エンジン + 薄い結線」に純化し、継続・再開・履歴管理を
  `RunResult.last_agent` / `to_input_list()` という SDK 標準様式に委ねる（build-don't-run・薄さに一致）。
- 内部インタプリタを `agents.Runner` 非依存とし、Agent ステップ実行を注入 runner シーム（本番は
  `_adapters` の `Runner.run` 実装、テストは fake）へ委譲することで、SDK 結合を `_adapters` に局在化
  しつつ `agents` 非依存テストを成立させる。
