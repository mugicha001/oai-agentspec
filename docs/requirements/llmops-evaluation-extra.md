# LLMOps 評価機能（oai-agentspec[llmops] 公開 extra）

## 1. 概要

oai-agentspec に、利用者が宣言した任意のエージェント（`AgentSpec` / `WorkflowGraph` / `HandoffGraph`）を評価できる LLMOps 評価機能を、公開 optional extra として追加する。採点コアは `oai-agentspec[llmops]`（DeepEval）として導入し、観測連携（Langfuse）は任意の別 extra `oai-agentspec[llmops-langfuse]` に分離する。評価は (1) エージェント単体（`AgentSpec` の入力 → 出力を観点別判定）、(2) エージェント横断（`HandoffGraph` / `WorkflowGraph` の end-to-end 評価）からなり、観点別の採点は DeepEval のメトリクスを採点器として行う。観点別 pass/fail と統合 verdict（pass/fail）を返すとともに、スコア / トレースを Langfuse に送信する（任意）。

本機能は要件 (B)「実行寄り層の runtime/ 集約と extra 化」が新設する `src/oai_agentspec/runtime/` 配下の実行寄り層の一員（`runtime/llmops`）として追加する。llmops は conversation / serve / cli と同じく `_adapters/` 経由で SDK 実行（`Runner.run`）へ薄く結線する公開の実行寄り層であり、build-don't-run の原則（コア宣言層は独自実行エンジン・公開の実行 API を持たない）はコア側で維持される。すなわち、評価を実行する公開 API は `runtime/llmops` の独立した公開窓口に集約し、コア宣言層の公開契約を汚さない。コア（`import oai_agentspec`）および既存 extra（conversation / serve / cli）は本 extra 未導入でも壊れないことを契約として維持する。本要件は (B) の構造再編を前提（依存）とする。

## 2. 機能要件

### FR-1: 評価対象の指定（任意の宣言物を入力にできる）
- ユーザーストーリー: lib 利用者として、自分が宣言した任意の `AgentSpec` / `WorkflowGraph` / `HandoffGraph` を評価対象として指定したい。なぜなら examples 固有でなく自分のエージェント構成をそのまま品質評価したいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `AgentSpec` を評価 API に渡す THEN 評価機能は当該 spec を対象として受理し、評価対象の識別子（名前等）を結果に含める。
  - [ ] WHEN 利用者が `WorkflowGraph` または `HandoffGraph` を評価 API に渡す THEN 評価機能はそのグラフ構造を横断評価の対象として受理する。
  - [ ] WHEN 利用者が `HandoffGraph` / `WorkflowGraph`（AGENT ノードを含む）を横断評価の対象として渡す THEN 評価機能は当該グラフに必要な spec を register 済みの `AgentRegistry` を伴って受領する（specs 供給経路を registry に一本化する）。IF registry が供給されない、または必要 specs が未登録 THEN specs 入手元が無い旨の明確なエラーを送出し、暗黙のフォールバックをしない。
  - [ ] IF 評価対象として未対応の型が渡された THEN 評価機能は明確なエラー（型・許容対象を示すメッセージ）を送出し、暗黙のフォールバックをしない。
  - [ ] WHEN 評価対象を指定する THEN 評価対象の宣言物自体を変更（mutate）せず、読み取り専用で扱う。
  - [ ] WHEN 評価データセット（入力ケース群と期待観点）を指定する THEN 利用者が渡したデータをそのまま評価入力に用い、lib 側にハードコードしたケースを混入させない。

### FR-2: プロンプト単体評価（Promptfoo harness）— スコープ外
- 本要件のスコープ外とする。理由: 採点エンジンとして採用する DeepEval が出力評価（観点別採点）をカバーし、プロンプト単体の評価は最小の `AgentSpec`（当該プロンプトを `instructions` とするエージェント）として FR-3 / FR-1 の経路で評価できるため、Promptfoo（Node.js / npx）を実行 harness として持ち込む構成は採らない。子プロセス起動を伴わないことに付随して NFR-4（子プロセス env 最小化）も該当なしとなる（NFR-4 参照）。

### FR-3: エージェント評価（DeepEval を採点器とする観点判定）
- ユーザーストーリー: エージェント開発者として、エージェントの入力に対する出力を観点別に判定したい。なぜなら事実整合性・安全性・関連性・簡潔性・ツール使用の正しさといった品質観点を定量的に把握したいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `AgentSpec` と入力ケースを指定して評価を実行する THEN 評価機能はエージェントの input → output を取得し、観点別に採点する。
  - [ ] WHEN 観点別の採点を行う THEN 採点は DeepEval のメトリクスを採点器として行う。DeepEval の LLM 呼び出しは custom model で利用者 Judge モデルをラップし `_adapters/` 経由に一本化し、DeepEval の結果を plain な観点結果へ変換して評価ロジック層が DeepEval 型を扱わないようにする。DeepEval のテレメトリは既定でオフにする（利用者設定で有効化可）。
  - [ ] WHEN 観点判定を行う THEN 少なくとも `factual_grounding` / `safety` / `relevance` / `conciseness`（出力品質）に対応し、加えて `tool_correctness`（ツール使用の正しさ・捕捉したツール呼び出し列と入力ケースの `expected_tools`（ground truth）の決定的比較・単体/横断のいずれにも適用可）に対応する。観点ごとに pass/fail（または順序尺度スコアと閾値による pass/fail）と判定根拠を返す。利用者は G-Eval rubric を渡して任意観点を追加できる。
  - [ ] WHEN Judge へ untrusted な入力（評価対象の出力等）を渡す THEN Spotlighting 相当のマーキングを施し、プロンプトインジェクションの影響を低減する。マーキングは framework 非依存の純ヘルパが担い、採点器へ渡す前に適用する。
  - [ ] IF 採点が所定スキーマに適合しない、またはタイムアウトする THEN 評価機能は当該観点を fail もしくは inconclusive に倒し（fail-closed）、未捕捉例外でプロセスを停止しない。
  - [ ] IF 入力ケースに参照文脈（ground truth / context）が与えられない THEN 評価機能は当該ケースの `factual_grounding` 観点を `not_applicable` に倒し、knockout fail-closed の対象および FR-5 の verdict 計算対象から除外する。
  - [ ] IF 入力ケースに `expected_tools` が与えられない THEN `tool_correctness` 観点を `not_applicable` に倒し、verdict 計算対象から除外する。WHEN `expected_tools` が与えられた THEN 評価対象がツールを持たない場合も NA にせず、期待ツールが呼ばれなければ recall=0 で fail とする（比較は recall・threshold=1.0・余分な呼び出しや handoff の `transfer_*` は無視）。
  - [ ] WHEN 採点器（DeepEval）/ Judge モデルを呼び出す THEN LLM / SDK 呼び出しは `_adapters/` 配下を経由して行う（SDK / 外部クライアント隔離との整合）。
  - [ ] WHEN 露出する観点を定める THEN JSON / スキーマ適合・RAG 系（retrieval 評価）・Summarization は対象外とする（`factual_grounding` 用のメトリクスは context ベースの事実整合性として使う範囲に限定する）。

### FR-4: エージェント横断評価（handoff / WorkflowGraph E2E）
- ユーザーストーリー: エージェント設計者として、複数エージェントの連携（handoff）や WorkflowGraph の end-to-end 動作を評価したい。なぜなら単体では正しくてもルーティングや連携の誤りで全体品質が劣化しないか確認したいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `HandoffGraph` または `WorkflowGraph` を指定して横断評価を実行する THEN 評価機能は end-to-end の入力から最終出力までを評価対象とする。横断評価は当該グラフに必要な spec を register 済みの `AgentRegistry` を伴って受領し（FR-1）、`HandoffGraph` は `apply(registry)` でエッジを反映してから entry agent を実行し、`WorkflowGraph` は registry を伴って Agent 化して実行する。
  - [ ] WHEN 横断評価を行う THEN `handoff_correctness` 観点（意図したエージェントへ正しくルーティングされたか）を評価し、pass/fail と根拠を返す。判定は入力ケースの `expected_route`（ground truth）と捕捉した実行経路の決定的比較で行う。
  - [ ] WHEN 横断評価で実行トレースを捕捉する THEN 生の実行結果は `_adapters/` 内で消費し、実行経路（ルーティング）とツール呼び出し列を plain なデータとして 1 パスで抽出し、評価ロジック層へは SDK 型を出さない（SDK 隔離との整合）。捕捉した実行経路 / ツール列を入力ケースの期待値（`expected_route` / `expected_tools`）と決定的に比較する。
  - [ ] WHEN `WorkflowGraph` を評価する THEN グラフのエントリから終端（END）までの実行結果に対して観点判定を行う。
  - [ ] IF 入力ケースに `expected_route` が与えられない THEN `handoff_correctness` 観点を `not_applicable` として記録し、verdict 計算から除外する（横断モードかどうかでは NA にしない。`expected_route` は起点を含むフルパスで指定する）。

### FR-5: pass/fail 判定と統合 verdict
- ユーザーストーリー: 品質ゲート運用者として、観点別の結果から 1 つの統合判定（pass/fail）を得たい。なぜなら CI / リリースゲートで合否を一意に分岐させたいから。
- 受け入れ基準:
  - [ ] WHEN verdict 計算対象観点（`skip` / `not_applicable` を除く観点）がすべて pass THEN 統合 verdict を `pass` とする（`skip` / `not_applicable` は母集合に含めない。FR-3 の参照文脈非在時 not_applicable / FR-4 の構造的 not_applicable 除外 / NFR-3 の skip と整合）。
  - [ ] IF いずれかの knockout 観点（既定で `safety` / `factual_grounding`）が fail THEN 統合 verdict を `fail` とし、他観点の結果で上書きしない（fail-closed）。ただし当該観点が `not_applicable` の場合は knockout 判定の対象外とする。
  - [ ] IF 判定が保留（inconclusive）となる観点がある THEN 設定された inconclusive ポリシー（既定 `fail`）に従って verdict を確定する。
  - [ ] WHEN 統合 verdict を導出する THEN 観点別の pass/fail/inconclusive/skip/not_applicable と統合 verdict の両方を構造化された結果として返す。
  - [ ] WHEN 評価対象に必須観点が欠落している THEN 欠落を検出し、verdict を fail に倒す（missing-pair fail-closed）。

### FR-6: Langfuse へのスコア / トレース送信（任意）
- ユーザーストーリー: LLMOps 運用者として、評価のスコアとトレースを Langfuse に送信したい。なぜなら評価結果を時系列で観測・比較し、品質回帰を検知したいから。
- 使う Langfuse 機能: Tracing / Scores（常時）、Datasets 登録 + dataset run リンク（opt-in）、push 専用 Prompt Management 登録 + 結果リンク（opt-in）。使わない Langfuse 機能: managed evaluator（サーバ側 LLM-as-judge）、Sessions・Users（評価の関心事でないためスコープ外）。
- 受け入れ基準:
  - [ ] WHEN Langfuse 設定（認証情報・接続先）が評価実行に渡される THEN 評価機能は観点別スコアと統合 verdict を Langfuse の Scores として送信し、評価対象の入出力と判定をトレースとして送信する。
  - [ ] WHEN dataset へケースを登録する THEN `register_dataset`（一度きり・冪等 upsert・安定キーは入力ケースの `id`）が item を登録し、`load_dataset` が fetch して `EvalCase` 列へ復元する（Langfuse が source の register → fetch → use）。WHEN Langfuse 設定に `dataset_name` が指定され `evaluate` を実行する THEN 既存 dataset item へ run を link するだけ（item の upsert / dataset 作成はしない）で、各ケースの trace / Scores を dataset item × run にリンクする（A/B・回帰比較用）。IF `dataset_name` が指定されない THEN dataset 連携をスキップし、Scores / Traces のみ送信する。
  - [ ] WHEN Langfuse 設定に `prompt_name` が指定され、かつ評価対象プロンプトが抽出可能（`AgentSpec` の静的 `instructions`） THEN 評価機能は当該プロンプトを Langfuse Prompt Management に register / upsert（push のみ）し、評価 trace を当該 prompt version にリンクする。Langfuse からプロンプトを取得 / 配信しない（取得系 API は呼ばない）。IF `prompt_name` が指定されない、または評価対象プロンプトが抽出不可（動的 `instructions` / PromptStore 合成 / 横断で単一プロンプト不特定） THEN prompt 連携をスキップする。
  - [ ] IF Langfuse 設定（認証情報）が渡されない THEN 評価機能は送信をスキップし、ローカルの評価結果（pass/fail / verdict）を返す（NFR-3 の graceful degradation と整合）。
  - [ ] IF Langfuse 設定は渡されたが送信（trace / Scores / dataset / prompt のいずれか）に失敗する THEN 評価機能はローカルの評価結果を返すことを優先し、送信失敗で評価全体を fail させない（ベストエフォート、warning ログ）。
  - [ ] WHEN Langfuse へ送信する THEN 外部 SaaS 通信は `_adapters/` 配下の窓口を経由する（SDK / 外部クライアント隔離との整合）。
- Prompt Management 連携とプロンプトバージョニングの線引き: 本連携は評価対象プロンプトの「登録 + 評価結果リンク」の観測記録であり、プロンプトバージョニング管理機能（snapshot / list / diff / rollback / push as feature）ではない。PromptStore（利用者 root）がプロンプトの Single Source of Truth のままであり（プロンプト非同梱方針との整合）、プロンプトバージョニング管理機能は制約事項のとおりスコープ外を維持する。

### FR-7: extra としての公開と未導入時に壊れない契約
- ユーザーストーリー: ライブラリ利用者として、評価機能を任意導入したい。なぜなら評価が不要な利用者に採点エンジン（DeepEval）や観測クライアント（Langfuse）の依存を強制したくないから。
- extra 分割: 採点コアは `oai-agentspec[llmops]`（DeepEval・採点に必須）、観測連携は `oai-agentspec[llmops-langfuse]`（Langfuse・任意）に分ける。観測込みは `oai-agentspec[llmops,llmops-langfuse]` で導入する。Langfuse を使わない利用者は `[llmops]` のみで評価でき、langfuse を導入しない（「使わないなら入れない」opt-in extra 哲学）。
- 受け入れ基準:
  - [ ] WHEN 利用者が `pip install oai-agentspec[llmops]` を行う THEN 採点コアとその依存（DeepEval）が導入され、Langfuse 設定を渡さない評価がローカルで完結する（採点 + verdict）。
  - [ ] WHEN 利用者が `pip install oai-agentspec[llmops,llmops-langfuse]` を行う THEN 観測連携（Langfuse クライアント）が追加導入され、Langfuse への送信が利用できる。
  - [ ] WHEN llmops extra 未導入で `import oai_agentspec` を実行する THEN ImportError 等を起こさずコア宣言層の公開 API（(B) 再設計後の宣言層 `__all__`）が利用できる。
  - [ ] WHEN llmops extra 未導入で既存 extra（`oai_agentspec.runtime.cli` 等）を import する THEN llmops 依存に起因する破綻なく従来どおり動作する。
  - [ ] WHEN llmops extra 未導入で評価機能の import 経路へアクセスする THEN 必要 extra（`oai-agentspec[llmops]`）の導入を促す明確なエラーメッセージを返す。
  - [ ] WHEN llmops-langfuse extra 未導入で Langfuse 連携を要求する（Langfuse 設定を渡す） THEN 必要 extra（`oai-agentspec[llmops-langfuse]`）の導入を促す明確なエラーメッセージを返す。Langfuse 設定を渡さない場合は該当せず、評価は採点コアのみでローカルに完結する。
  - [ ] WHEN 評価機能の公開 API を追加する THEN コア宣言層の公開契約（(B) 再設計後の宣言層 `__all__`）を壊さず、llmops 公開 API は `runtime/llmops` の独立した公開窓口に集約する。

### FR-8: 承認ゲートの正しさ評価（HITL ゲート評価）
- ユーザーストーリー: エージェント開発者として、承認必須ツール（人間承認ゲート）を持つエージェントが、危険操作を実行する前に正しく承認ゲートへ回したかを評価したい。なぜなら危険な副作用を無断実行しない安全設計が効いているかを定量的に確認したいから。
- 受け入れ基準:
  - [ ] WHEN 承認必須ツールを持つ評価対象と `ApprovalGate` 観点・期待承認ツール（`expected_approvals`）を指定して評価する THEN 評価機能は実行中に承認待ちとなったツール名を捕捉し、`expected_approvals` と決定的に比較して pass/fail と根拠を返す。
  - [ ] WHEN 承認ゲート評価を行う THEN resume も承認もせず、評価対象の危険ツールを一切実行しない。
  - [ ] IF 入力ケースに `expected_approvals` が与えられない THEN `ApprovalGate` 観点を `not_applicable` に倒し、verdict 計算対象から除外する。
  - [ ] WHEN 評価対象が承認待ちで中断する THEN `ApprovalGate` は中断時でも採点し、その他の観点は inconclusive に倒す（FR-5 の inconclusive ポリシーで verdict 解決）。

### FR-9: 承認の自動解決による完了採点（mock-approve・安全な HITL 評価）
- ユーザーストーリー: エージェント開発者として、承認必須ツールを持つエージェントを、本物の危険な副作用を起こさずに完了まで実行して承認後の応答・経路・ツール使用を評価したい。なぜなら HITL 経路（中断→承認→再開）を通したエンドツーエンド品質を、危険操作を実行せずに測りたいから。
- 受け入れ基準:
  - [ ] WHEN 評価実行に承認ポリシー（`approvals`・承認待ちを受けて承認/却下を返す）とツールモック（`tool_mocks`・agent スコープの `{agent_name: {tool_name: 値 | callable}}`）を渡す THEN 評価機能は承認を自動解決して実行を完了させ、完了後の出力・経路・ツール使用を採点する。`approvals` / `tool_mocks` を渡さなければ FR-5 の中断既定挙動（inconclusive→fail）を維持する。
  - [ ] WHEN ツールをモック差し替えする THEN ツールの実行本体だけを副作用のない代替へ差し替え、ツールの宣言メタデータ（名前・説明・引数スキーマ・承認要否）は変更しない（評価対象の「ツールを呼ぶか」の判断を本番と同一に保つ）。
  - [ ] WHEN モック差し替え・registry のクローン・グラフの複製を行う THEN 利用者が渡した registry / グラフを一切変更しない（評価は利用者状態を汚さない）。動的ハンドオフの候補もクローン経由でモックされる。
  - [ ] IF 承認ポリシーが承認を返したツールが実際にはモック差し替えされていない（モック未登録 / 到達不能 / 別 agent の同名ツール / agent 不明） THEN 明確なエラーを送出し、本物の危険ツールを実行しない（fail-closed・安全不変条件）。
  - [ ] WHEN 承認ポリシーが却下を返す THEN ツールを実行せず実行を継続し、拒否後の応答を評価できる。
  - [ ] WHEN Langfuse 連携時に承認・中断が発生する THEN 観測記録（trace metadata）に承認待ち・中断状況を反映する。

## 3. 非機能要件

### NFR-1: セキュリティ（SDK / 外部クライアント隔離）
- 要件: 評価機能による LLM / 採点器（DeepEval）/ Langfuse / SDK 呼び出しは `_adapters/` 配下のみを窓口とし、評価ロジック層は plain データと不透明型のみを扱う。`from agents` / `from openai` / `import deepeval` / `import langfuse` の直接 import を評価ロジック層に持ち込まない。観点 → メトリクスの対応は抽象識別子として保持し、DeepEval クラスへの解決は `_adapters/` 配下に閉じる。
- 計測基準: `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` の結果が空であること。評価機能追加後も同コマンドが空を維持すること。採点エンジン（`deepeval`）/ 外部 SaaS クライアント（`langfuse`）の import も `_adapters/` 配下に閉じること（同様の grep で確認）。

### NFR-2: 可用性（コア / 既存 extra が llmops 未導入で壊れない）
- 要件: llmops extra 未導入の環境で `import oai_agentspec` および既存 extra（conversation / serve / cli）の import が成功する。評価依存は遅延 import 境界で隔離する。
- 計測基準: `uv run python -c "import oai_agentspec as m; assert all(hasattr(m,s) for s in m.__all__)"` が llmops extra 未導入でも成功すること（(B) 再設計後の宣言層 `__all__` を対象とする）。llmops 依存をアンインストールした状態のテスト（または依存非導入を模した環境）でコア公開 API スモークが緑であること。

### NFR-3: 可用性（外部依存非在時の graceful degradation）
- 要件: 評価を採点コア（ローカル完結）と観測連携（Langfuse・任意）に分離する。Langfuse 認証情報・`llmops-langfuse` extra・LLM API のいずれかが不在でも、観測連携を skip / 明確なエラーに倒し、不在に起因する未捕捉例外でプロセスを停止させない。Langfuse 設定を渡さない評価は採点コア `[llmops]` のみでローカルに完結する。
- 計測基準: Langfuse 未設定時に評価がローカル verdict を返すこと、`dataset_name` / `prompt_name` 未設定時に当該連携をスキップして Scores / Traces のみ送信すること、Langfuse 送信（trace / Scores / dataset / prompt）失敗時にローカル結果を返すこと、`llmops-langfuse` extra 未導入で Langfuse 設定を渡したときに明示エラーを返すことを、それぞれテストで検証する（外部実通信なしの fake / モック層で再現）。

### NFR-4: セキュリティ（外部子プロセスへの env 最小化）— 該当なし
- 本要件では外部子プロセス（npx 等）を起動しないため該当なし（FR-2 をスコープ外としたことに伴う）。評価機能に `os.environ.copy()` 相当の全量受け渡しや子プロセス起動を持ち込まない。

### NFR-5: 保守性（env 参照の境界）
- 要件: 本体（コア宣言層）の env 非依存方針と整合させ、評価実行に必要な env 参照は評価機能の実行境界（`runtime/llmops` の評価エントリ / `_adapters` の設定ヘルパ）に閉じる。コアの宣言層および `_adapters` の既存契約に env 依存を波及させない。
- 計測基準: 評価機能の env 参照箇所が `runtime/llmops` サブパッケージの境界に限定されることをコードレビューで確認する。コア層の既存 env 非依存テストが緑を維持すること。

### NFR-6: 保守性（テストカバレッジ / リント）
- 要件: 評価機能追加後もプロジェクトのテストカバレッジ閾値とリント基準を維持する。外部実通信に依存しない単体・統合テスト（fake / モデルモック）でレイヤを検証する。
- 計測基準: `uv run pytest` がカバレッジ 80% 以上（`fail_under = 80`）で緑であること。`uv run ruff check src/ tests/` において、`runtime/llmops` サブパッケージおよび追加テストファイルが lint をパスし、既存ファイルへ新規の lint 違反を持ち込まないこと（本変更で新たに増える違反が 0 件であること）。

### NFR-7: 保守性（実行は _adapters 経由・runtime 実行寄り層への整合）
- 要件: llmops は独自実行エンジンを持たず、実行は `_adapters/` 経由で SDK `Runner.run` へ結線する。公開の評価実行 API は `runtime/llmops` サブパッケージ側に置き、コア宣言層には実行 API を追加しない。llmops は (B) が新設する runtime/ 配下の実行寄り層の一員として、conversation / serve / cli と同じ整合方針に従う。
- 計測基準: コア宣言層の公開契約（(B) 再設計後の `__all__`）に評価実行 API が含まれないこと、評価実行 API が `runtime/llmops` 配下のみに存在することをコードレビュー + 公開 API スモークで確認する。

### NFR-8: 性能（評価実行のタイムアウトと実行制御）
- 要件: 評価実行（特に採点器（DeepEval）/ LLM 呼び出し）にタイムアウトを設定可能とし、未指定時は既定値を適用する。複数件の入力ケースに対して逐次実行と並列実行（並列度を設定可能）の実行制御を持ち、未指定時は既定の実行モードを適用する。
- 計測基準: 採点呼び出しにタイムアウト（既定値あり・利用者が設定可能）が適用されることをテストで検証する。タイムアウト到達時は FR-3 の fail / inconclusive 倒し（fail-closed）に従って観点結果が確定すること（タイムアウト基準は FR-3 と矛盾させない）。逐次 / 並列の実行制御が設定で切り替わり、複数ケースが指定どおりに処理されることをテストで検証する。

## 4. 制約事項

- 技術的制約:
  - 本要件は (B)「実行寄り層の runtime/ 集約と extra 化」を前提（依存）とする。llmops は (B) が新設する `runtime/` 配下に `runtime/llmops` として追加されるため、(B) の構造を前提に設計する。
  - llmops は独自実行エンジンを持たず、実行は `_adapters/` 経由で SDK `Runner.run` へ薄く結線する公開の実行寄り層である。build-don't-run の原則（コア宣言層は独自実行エンジン・公開の実行 API を持たない）はコア側で維持する。緩和はしない。
  - SDK / 外部クライアント呼び出しは `_adapters/` 経由（NFR-1）。評価ロジック層は plain データ / 不透明型のみを扱う。
  - プロンプトは lib に同梱しない方針を維持する。評価対象プロンプト・評価データセット・G-Eval rubric 本文は利用者が渡す（lib にケース・プロンプト文字列をハードコードしない）。
  - 外部依存（採点エンジン DeepEval / 観測 SaaS Langfuse / Azure・OpenAI）は optional extra として許容する。採点コア（`llmops` = DeepEval）と観測（`llmops-langfuse` = Langfuse）は別 extra に分割する。ただしコア `import oai_agentspec` と既存 extra（conversation / serve / cli）は llmops extra 未導入でも壊れない（NFR-2）。
  - extra の宣言は `pyproject.toml` の `[project.optional-dependencies]` に (B) で追加される conversation / serve / cli extra と同じ階に並べて追加する。遅延 import 境界の運用は runtime 実行寄り層の方式を踏襲し、`deepeval` / `langfuse` のトップ import を本体に持ち込まない。
  - dist 名は `oai-agentspec`、パッケージ名は `oai_agentspec`。型チェッカ（mypy）は導入しない（既存方針）。
- スコープ外（本要件に含めない）:
  - (B) 構造再編そのものの実施（runtime/ の新設、conversation / serve / cli の移動・extra 化、コア `__all__` の再設計）。これらは別要件 (B) が担当する。本要件は (B) の runtime/ 配下に乗る前提で `runtime/llmops` を追加する。
  - プロンプトバージョニング（snapshot save / list / diff / rollback / push）。
  - RAGAS / RAG 評価（`retrieval_quality` 観点・RAGAS observability metrics を含む）。
  - 報酬設計（reward modeling / Lightning 相当）。別途相談とする。
- ビジネス制約:
  - 出力（コード・ドキュメント・コミット・Issue・PR 等）に絵文字を含めない。AI が生成したことを示唆する文言を含めない。

## 5. 影響範囲

- 関連コンポーネント:
  - `pyproject.toml`: `[project.optional-dependencies]` に採点コア `llmops`（DeepEval）と観測 `llmops-langfuse`（Langfuse）の 2 extra を追加する。これは (B) で追加される conversation / serve / cli extra と同じ階に並ぶ。
  - `src/oai_agentspec/runtime/llmops`: 評価機能サブパッケージを (B) の runtime/ 配下に新設し、独立した公開窓口（サブパッケージ `__init__`）に評価 API（`evaluate`）・結果型・設定型・観点オブジェクト（`Criterion` と組込みファクトリ）・dataset 連携（`register_dataset` / `load_dataset`）を集約する。コアの宣言層公開契約（(B) 再設計後の `__all__`）を汚さない。公開関数名は `evaluate` とし、`eval`（Python 組み込み関数）をシンボル・モジュール名に採用しない。
  - `src/oai_agentspec/_adapters/`: DeepEval 採点窓口（`import deepeval` を局在化）・実行トレース捕捉窓口（生実行結果を `_adapters` 内で消費し plain な実行経路 / ツール呼び出し列を抽出）・Langfuse 連携窓口（`langfuse` を関数内遅延 import）を追加（既存 `models` / `responses` / `runner` 等と同列の adapter モジュール）。
  - 評価対象となる宣言物の型（`spec.AgentSpec` / `handoffs.HandoffGraph` / `workflow.WorkflowGraph` / `prompts.PromptStore`）は読み取りのみで参照し、改変しない。
  - `docs/`: `docs/architecture.md`（評価レイヤを runtime 実行寄り層の一員として、extra 構成と整合する現在仕様として記述）、`docs/requirements/`（本要件の反映）。子プロセス起動を持ち込まないため `docs/security-scanning.md` への追記は不要。Spec 駆動規約（現在仕様の SoT・履歴記述や Issue 番号入りファイル名の禁止）に従う。
  - `tests/`: src ミラー構造（(B) 再設計後の runtime 構造に追従）に評価機能テストを追加（fake / モデルモックで外部実通信を排した unit / integration）。
- 既存機能への影響:
  - コア宣言層の公開契約（(B) 再設計後の `__all__`）と既存シンボルの振る舞いは不変に保つ（契約）。llmops の公開 API はコア `__all__` に追加せず `runtime/llmops` の公開窓口に集約する。
  - 既存 extra（conversation / serve / cli）の import 契約・遅延 import 境界に回帰を与えない。
  - 単方向依存（コア宣言層は runtime / llmops へ依存しない。依存方向は runtime/llmops → コア宣言層 / `_adapters` の一方向）を壊さない。評価サブパッケージは宣言物を読み取り依存し、コア層を逆依存させない。

## 6. 用語定義

| 用語 | 定義 |
|------|------|
| LLMOps | LLM を用いたアプリケーションの評価・観測・品質管理を運用する実践領域 |
| extra | Python パッケージの optional dependency 群。本機能の採点コアは `[llmops]`（DeepEval）、観測連携は `[llmops-langfuse]`（Langfuse）に分割する。未導入でもコアが動く前提 |
| build-don't-run | 本ライブラリの核方針。コアは独自実行エンジンを持たず、宣言・build-time 検証に徹し、実行は `_adapters/` 経由で SDK `Runner.run` への薄い結線に寄せる。実行寄り層は runtime に属する |
| 実行寄り層（runtime） | 公開の実行サービス API を提供する層。conversation / serve / cli / llmops が該当し、`src/oai_agentspec/runtime/` 配下に集約される（(B) で新設） |
| runtime/ | (B) が新設する中間ディレクトリ。実行寄り層を配下に集約する。本要件の llmops は `runtime/llmops` として追加する |
| AgentSpec | `agents.Agent` の薄い宣言的 Wrapper。評価対象となる単一エージェントの宣言 |
| HandoffGraph | エージェント間のハンドオフ関係を表す宣言グラフ。横断評価の対象 |
| WorkflowGraph | ワークフロー DSL の宣言グラフ（START / END / ノード）。end-to-end 評価の対象 |
| PromptStore | 利用側 root 配下のプロンプトテンプレートを合成する仕組み。lib はプロンプト非同梱 |
| DeepEval | LLM アプリ評価 OSS。本要件では観点別採点の採点器として使用する（Faithfulness / Answer Relevancy / G-Eval / ToolCorrectnessMetric 等）。`import deepeval` は `_adapters/` 配下に閉じる |
| 観点（criterion） | 評価の軸。本要件では `factual_grounding` / `safety` / `relevance` / `conciseness` / `tool_correctness` / `handoff_correctness` / `approval_gate` を対象とする。観点の適用可否は利用者の criteria 選択と入力ケースの ground truth 充足で決まる |
| factual_grounding | 出力が与えられた参照文脈・事実に整合しているかの観点（knockout 観点）。参照文脈が与えられないケースでは not_applicable に倒す |
| safety | 出力が安全（有害・危険でない）かの観点（knockout 観点） |
| relevance | 出力が入力・意図に関連しているかの観点 |
| conciseness | 出力が冗長でなく簡潔かの観点 |
| tool_correctness | 意図したツールが正しく呼び出されたかの観点。捕捉したツール呼び出し列と入力ケースの `expected_tools`（ground truth）を recall（期待ツールが全て呼ばれていれば pass・余分な呼び出しや handoff の `transfer_*` は無視）で決定的に比較する。`expected_tools` 非在のとき not_applicable（ツール非保有では NA にせず recall=0 で fail） |
| handoff_correctness | 意図したエージェントへ正しくルーティング／ハンドオフされたかの観点。`expected_route`（ground truth・起点を含むフルパス）と捕捉した実行経路を決定的に比較する。`expected_route` 非在のとき not_applicable |
| approval_gate | 承認必須ツールを実行前に正しく人間承認ゲートへ回したかの観点（`ApprovalGate()`）。`expected_approvals`（ground truth・期待承認ツール名）と中断時の承認待ちツール名を決定的に recall 比較する。resume / approve せず危険ツールを実行しない。`expected_approvals` 非在のとき not_applicable |
| tool_mocks | mock-approve でツールの実行本体を副作用のない代替へ差し替える宣言。agent スコープのネスト dict（`{agent_name: {tool_name: 値 \| callable}}`）。`on_invoke_tool` だけ差し替え、名前・説明・引数スキーマ・承認要否は不変（評価対象の判断を本番と同一に保つ） |
| approvals | mock-approve の承認ポリシー。承認待ち（`{tool_name, call_id, agent_name}`）を受けて承認/却下（bool）を返す。承認は実際にモック差し替えされた `(agent_name, tool_name)` のみ認可し、未差し替えの承認は fail-closed エラー |
| expected_approvals | `ApprovalGate` の ground truth。実行前に承認ゲートへ回るべきツール名の集合 |
| 安全不変条件（mock-approve） | mock-approve で本物の危険ツールが評価中に実行されないための不変条件。approve を認可するのは実際にモック差し替えされた `(agent_name, tool_name)` に限り、未差し替え / 到達不能 / 別 agent 同名 / agent 不明は ValueError（fail-closed） |
| Langfuse | LLM アプリ向け観測性 SaaS。評価スコア（Scores）/ トレース / opt-in の Datasets / opt-in の push 専用 Prompt Management の送信先（観測連携・任意） |
| Quality Gate | 評価結果に基づき CI / リリースの合否を分岐させる品質ゲート |
| verdict | 統合判定。本要件では `pass` / `fail`（および中間状態 `inconclusive`）を取りうる |
| knockout 観点 | fail した時点で統合 verdict を fail に確定させ上書きされない観点（既定 `safety` / `factual_grounding`）。当該観点が not_applicable のときは判定対象外 |
| verdict 計算対象観点 | 統合 verdict の母集合となる観点。skip / not_applicable は母集合から除外する |
| inconclusive | 判定保留状態。設定ポリシー（既定 fail）に従って verdict 確定時に解決される |
| not_applicable | 評価対象・入力ケースに構造的に適用できない観点の状態。verdict 計算から除外する |
| skip | 評価を実施しなかった観点の状態。verdict 計算の母集合から除外する |
| Spotlighting | untrusted 入力をマーキングし、プロンプトインジェクションの影響を低減する防御手法 |
| fail-closed | 不確実・欠落・失敗時に安全側（fail）へ倒す方針 |
| graceful degradation | 外部依存不在時にクラッシュせず skip / 明示エラー等で機能縮退する挙動 |
| SDK 隔離（NFR） | `from agents` / `from openai` 等の外部 SDK import および外部クライアント（`deepeval` / `langfuse`）の import を `_adapters/` 配下のみに閉じる本体の不変条件 |
| EARS | Easy Approach to Requirements Syntax。WHEN / IF / THEN による受け入れ基準記述形式 |
| _adapters | 外部 SDK / 外部クライアントへの import 単一窓口（SDK / 外部クライアント隔離を担うサブパッケージ） |
