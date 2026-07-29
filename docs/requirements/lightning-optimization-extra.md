# Agent Lightning 統合（oai-agentspec[lightning] / [lightning-rl] 公開 extra）

## 1. 概要

oai-agentspec に、利用者が宣言した任意のエージェント（`AgentSpec` / `WorkflowGraph` / `HandoffGraph`）を最適化できる Agent Lightning 統合機能を、公開 optional extra として追加する。最適化は (1) APO（Automatic Prompt Optimization・プロンプトテキストの自動改善・GPU 不要の軽量系）、(2) RL（LightningRL on VERL・モデル重みの更新・GPU を伴う重量系）の 2 系統からなり、いずれも Agent Lightning（pip: `agentlightning`）を単一の最適化エンジンとして委譲する。利用者が rollout（宣言エージェントの実行）・reward（報酬）・データ（`train` / 任意の `val`）を供給し、`algorithm` 指定で APO / RL を選択する統一エントリ `optimize` を通じて最適化を回す。単一エージェントの最適化に加え、グラフ（`HandoffGraph` / `WorkflowGraph`）を対象とするハンドオフを通る系全体の end-to-end 最適化（APO は複数スロットによるプロンプト同時最適化、RL は credit assignment による系全体ポリシー最適化）に対応する。APO の出力は最適化済みプロンプト文字列、RL の出力は checkpoint パス / OpenAI 互換エンドポイント等の参照であり、いずれも plain データとして返す。

本機能は実行寄り層 `runtime/` 配下の一員（`runtime/lightning`）として追加し、`runtime/llmops` の `evaluate(宣言物, ...)` と同型の入力パターン・公開窓口・extra 未導入契約・SDK 隔離方針に従う。lightning は conversation / serve / cli / llmops と同じく `_adapters/` 経由で SDK 実行（`Runner.run`）へ薄く結線する公開の実行寄り層であり、最適化ループ本体は agent-lightning の Trainer へ委譲する（build-don't-run の原則は維持され、コア宣言層は独自の最適化／実行エンジンを持たない）。すなわち、最適化を実行する公開 API は `runtime/lightning` の独立した公開窓口に集約し、コア宣言層の公開契約（コア `__all__`）を汚さない。コア（`import oai_agentspec`）および既存 extra（conversation / serve / cli / llmops）は本 extra 未導入でも壊れないことを契約として維持する。

APO は最適化対象を利用者指定の「チューナブルなプロンプトスロット（seed テキスト）+ rebind（候補スロット値からエージェントを組み直す関数）」モデルで扱い、静的 str / 動的 callable / Responses `prompt`（id 参照）や複数テンプレートの合成結果と両立する。単一スロット（1 エージェント / 1 セグメント調整の簡単版）に加え、複数スロットの mapping を受けて系全体のプロンプトを同時最適化できる。`prompt_slot` ヘルパは各スロットに `build`（候補テキスト → `AgentSpec`）を内包するため、スロットが `prompt_slot` の戻り値である限りフレームワークが rebind を自動導出し、利用者は手書きの rebind を渡す必要がない（手書き rebind は生 seed 文字列等のパワーユーザー経路でのみ用いる）。`build` は省略でき、対象エージェントが registry に登録済みなら既定 build は登録 `AgentSpec` を複製して `instructions` だけ候補で差し替える（tools / handoffs / model 等を再宣言しなくてよい）。ファクトリヘルパ `prompt_slot_factory` は共通既定値を束ねた slot 生成 callable を返すため、dict comprehension と組み合わせれば列挙したエージェントのスロット mapping を組み立てられ、グラフ全体 APO が単一の生成経路と `optimize` の単一呼び出しで完結する。APO の最適化対象は vars（`${var}` 置換値）を展開していないテンプレート文言（`${var}` プレースホルダ保持）であり、vars 値は最適化対象外（不変・確定）で各 rollout で再注入される。lib は `PromptStore` の合成内部に踏み込まず（テンプレートを内省・書き換えしない）、`PromptStore` は利用者の Single Source of Truth のままで、APO 出力は `${var}` を保持した plain なテンプレートとして返し利用者が永続化する。

使いやすさのため、`runtime/lightning` 側に opt-in な薄いヘルパ（よくある目的関数を 1 行で記述する reward ファクトリ・合成プロンプト最適化の定型を畳む `prompt_slot` / 共通既定値を束ねる `prompt_slot_factory`・データを決定的に分割する `train_val_split`）を併設する。これらのヘルパは既存 `PromptStore`（`src/oai_agentspec/prompts.py`）の公開メソッド（`compose` / `get`）を読み取るだけで（分割ヘルパは純データ操作）、`PromptStore` のクラス・メソッド・役割（利用者 SoT・lib 非書込）を一切変更しない。生の `slot` / `rebind` / `reward` callable も併存して受け、パワーユーザーの拡張余地を残す。データ入口は `train`（必須）/ `val`（任意）に統一し、`optimize` 内に暗黙の分割パラメータを持たない（データの渡し方を 2 通りにしない）。最適化結果は既定で戻り値（plain データ）として返すのみで lib は自動書き込みをせず、利用者が opt-in で `result.save(path)` を呼んだときだけ利用者指定パスへ書き出す。

プロンプト自動改善（APO）の責務は Lightning 側に集約する。プロンプト自動改善 APO を含むエージェント自動作成支援機能は、本機能が提供する Lightning APO を内部利用する上位 UX として位置づけ、独自の APO エンジンを別実装しない棲み分けとする（影響範囲・制約事項に明記）。Agent Lightning を LLMOps トラックへ振り分けた検討経緯は `docs/rationale/agt-governance-integration.md` を参照。

## 2. 機能要件

### FR-1: 最適化対象の指定（任意の宣言物を入力にできる・系全体も対象）
- ユーザーストーリー: lib 利用者として、自分が宣言した任意の `AgentSpec` / `WorkflowGraph` / `HandoffGraph` を最適化対象として指定したい。なぜなら examples 固有でなく自分のエージェント構成（単一エージェントからハンドオフを含む系全体まで）をそのまま最適化（プロンプト改善 / モデル更新）したいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `AgentSpec` を `optimize` に渡す THEN 最適化機能は当該 spec を rollout 対象として受理し、対象の識別子（名前等）を結果に含める。
  - [ ] WHEN 利用者が `WorkflowGraph` または `HandoffGraph` を `optimize` に渡す THEN 最適化機能はそのグラフ構造を横断 rollout の対象として受理する。
  - [ ] WHEN 利用者がグラフ（`HandoffGraph` / `WorkflowGraph`）を対象に渡す THEN rollout はハンドオフを通る系全体を end-to-end に実行し、系全体を最適化対象にできる（系全体最適化）。単一エージェント（`AgentSpec`）の最適化は系全体最適化の簡単版として併存する。
  - [ ] WHEN 利用者が `HandoffGraph` / `WorkflowGraph`（AGENT ノードを含む）を rollout 対象として渡す THEN 最適化機能は当該グラフに必要な spec を register 済みの `AgentRegistry` を伴って受領する（specs 供給経路を registry に一本化する）。IF registry が供給されない、または必要 specs が未登録 THEN specs 入手元が無い旨の明確なエラーを送出し、暗黙のフォールバックをしない。
  - [ ] IF 最適化対象として未対応の型が渡された THEN 最適化機能は明確なエラー（型・許容対象を示すメッセージ）を送出し、暗黙のフォールバックをしない。
  - [ ] WHEN 最適化対象を指定する THEN 最適化対象の宣言物自体を変更（mutate）せず、読み取り専用で扱う。最適化結果（プロンプト文字列 / モデル参照）は新規の plain データとして返し、利用者が渡した registry / グラフ / spec を一切変更しない。

### FR-2: 統一エントリ `optimize` と algorithm 選択（APO / RL）
- ユーザーストーリー: lib 利用者として、単一の公開エントリから algorithm を指定して APO（プロンプト最適化）または RL（モデル更新）を選びたい。なぜなら llmops の `evaluate(宣言物, ...)` と同じ入力パターンで一貫して最適化を扱いたいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `optimize(宣言物, algorithm="apo", ...)` を呼ぶ THEN 最適化機能は APO 経路（プロンプトテキスト最適化）を実行し、最適化済みプロンプト文字列を含む結果を返す。
  - [ ] WHEN 利用者が `optimize(宣言物, algorithm="rl", ...)` を呼ぶ THEN 最適化機能は RL 経路（モデル更新）を実行し、checkpoint パス / OpenAI 互換エンドポイント等の参照を含む結果を返す。
  - [ ] WHEN 最短ケースを実行する THEN `optimize(宣言物, algorithm=..., train=..., reward=...)` の単一呼び出しで最適化が完結する（NFR-9 と整合）。
  - [ ] WHEN 結果を返す THEN 結果に train 上のスコア / 履歴と `val_score`（`val` 上の汎化スコア・`val` 省略時は None）を含める。APO の結果は最適化済みスロットテキスト（複数スロット時は名前付き mapping）、RL の結果はモデル参照（`model_ref`・target agents ごとに識別可能な参照を含む）を併せて含める。
  - [ ] WHEN `optimize` が結果を返す THEN 既定では result（plain データ）を返すのみで、lib はファイル・`PromptStore`・外部ストアへ自動書き込みをしない（PromptStore 非書込・モデル外部流入・build-don't-run と整合）。結果の永続化は利用者の明示操作（FR-9 の `result.save(path)` または利用者自前の書き出し）に限る。
  - [ ] WHEN RL の物理出力（checkpoint 等）を扱う THEN その出力先は agent-lightning の Trainer / Store 設定の passthrough であり、lib は保管を所有せず参照（`model_ref`）を返すのみとする（NFR-7 と整合・lib はモデル重みを保持しない）。
  - [ ] IF `algorithm` に未対応の値が渡された THEN 最適化機能は対応値（`apo` / `rl`）を示す明確なエラーを送出し、暗黙のフォールバックをしない。
  - [ ] WHEN 公開エントリ名を定める THEN Python 組込みと衝突する名（`eval` 等）をシンボル・モジュール名に採用せず、`optimize` を採用する。
  - [ ] WHEN APO / RL いずれの経路でも THEN reward・データ（train / val）・rollout 設定は利用者供給とし、lib 側にプロンプト・データ・報酬関数をハードコードしない。

### FR-3: APO（プロンプト最適化・スロット+rebind モデル・vars 不変・複数スロット系全体最適化・GPU 不要・[lightning] のみで完結）
- ユーザーストーリー: プロンプト設計者として、エージェント（単一 / ハンドオフを含む系全体）のプロンプトを Agent Lightning の APO で自動改善したい。なぜなら beam-search + LLM テキスト勾配でプロンプト品質を向上させたいが、GPU や重い訓練インフラは導入したくないから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `algorithm="apo"` で `optimize` を実行する THEN 最適化機能は APO ループを agent-lightning の Trainer へ委譲し、最適化済みプロンプト（スロット）テキストを結果として返す。
  - [ ] WHEN APO を実行する THEN 最適化対象は利用者が指定するチューナブルなプロンプトスロット（seed テキスト）とする。スロットの粒度（agent テンプレート本文 / part / 合成結果全体など）は利用者が選び、lib が agent 全体を暗黙に最適化対象としない。
  - [ ] WHEN 利用者が vars（`${var}` 置換値）を指定する THEN vars は最適化対象外（不変・確定）とする。最適化対象（slot）は vars を展開していないテンプレート文言（`${var}` プレースホルダ保持）であり、APO は vars 値を最適化対象（候補生成）に含めない。
  - [ ] WHEN APO が各 rollout を実行する THEN vars は rollout 時に再注入（substitute）する（`prompt_slot` 利用時はヘルパが内部で再注入し、生 callable 経路では rebind / build が再注入する）。vars 値は候補生成の対象にしない。
  - [ ] WHEN APO の出力（最適化済みスロットテキスト）を返す THEN `${var}` プレースホルダを保持したテンプレートとして返し、vars 値を埋め込まない（利用者が自分のテンプレートへそのまま貼り戻せる・複数スロット時は名前付き mapping）。lib は `PromptStore` へ書き込まない（永続化は利用者責任・FR-9 の `result.save` で opt-in 可）。
  - [ ] IF 候補プロンプトが rollout に必要な `${var}` プレースホルダを失う THEN 当該候補を無効化 / 低評価に倒し（fail-closed）、vars 値が最適化で改変・欠落しない不変条件を保つ。
  - [ ] WHEN 利用者がスロットを渡す THEN 単一スロット（seed テキストまたは `prompt_slot` の戻り値）に加え、複数スロットの mapping `{名前: seed/slot}` を受けられる。複数スロット時は各エージェント / セグメントのプロンプトを同時に最適化する（系全体のプロンプト最適化）。単一スロットは 1 エージェント / 1 セグメントだけを調整する簡単版として併存する。
  - [ ] WHEN グラフ全体を APO で最適化する THEN 利用者は `prompt_slot_factory`（FR-9）+ dict comprehension で slot mapping を生成し `optimize(graph, slot=slots, ...)` の単一呼び出しで完結できる。最適化対象は slot mapping に掲載したエージェントのみとし、lib が暗黙に全 agent を対象とせず、利用者が mapping への掲載で明示選択する（未掲載のエージェントのプロンプトは固定）。
  - [ ] WHEN スロットが `prompt_slot` の戻り値（単一 / mapping）である THEN 各スロットの `build`（候補テキスト → `AgentSpec`）から rebind を自動導出し、registry / グラフ全体を組み直す。利用者は手書きの rebind を渡さなくてよい（系全体でも `optimize(graph, slot={名前: prompt_slot(...)}, ...)` の単一呼び出しで完結する）。
  - [ ] WHEN スロットが生 seed（`build` を持たない生 seed テキスト等）である THEN 利用者が rebind（単一候補、または候補 mapping `{名前: 候補テキスト}` を受けて registry / グラフ全体を組み直す関数）を渡す。この生 callable 経路でのみ rebind の明示が必要となり、rebind / build が vars 再注入を担う。
  - [ ] WHEN 複数スロットを最適化する THEN それらを agent-lightning の Trainer へ複数プロンプトリソースとして委譲し、lib は最適化ループを実装しない（build-don't-run と整合）。
  - [ ] WHEN APO が候補プロンプトを各 rollout に適用する THEN rebind（自動導出されたもの、または利用者供給のもの）を通じてエージェントを構築する。rebind 内（`prompt_slot` の `build` を含む）で利用者が自分の `PromptStore` 合成を適用してよく、lib は `PromptStore` のテンプレートを内省・書き換えしない（合成内部に踏み込まない）。
  - [ ] WHEN `AgentSpec.instructions` が静的 str のみの単純ケース THEN スロットの既定値を当該文字列とし、rebind 省略時は instructions を差し替える既定 rebind を用いてよい。
  - [ ] WHEN `AgentSpec.instructions` が動的 callable `(context, agent) -> str` / Responses `prompt`（id 参照）である THEN 利用者がスロット + rebind（`prompt_slot` の `build` 経由でもよい）を与えれば一様に APO 対象にできる（「合成 / 動的だから対象外」を解消する）。
  - [ ] IF チューナブルなスロットが設計不能（スロット未指定かつ instructions が内省不能な動的 callable 等で seed テキストを定められない） THEN 明確なエラー（スロット / rebind の指定を促す理由を示す）を送出し、暗黙のフォールバックをしない。
  - [ ] WHEN APO の rollout を実行する THEN 候補スロット（単一 / mapping）を rebind で適用し vars を再注入した宣言エージェント（系全体ではハンドオフを通る系全体）を `_adapters/` → `Runner.run` で実行し、実行結果から reward 算出に必要な plain データ（出力・実行経路・ツール列）を抽出して利用者供給の reward へ渡す。
  - [ ] WHEN APO の rollout / テキスト勾配 LLM 呼び出しを行う THEN LLM / SDK 呼び出しは `_adapters/` 配下を経由する（SDK / 外部クライアント隔離との整合）。`[lightning]` extra（`agentlightning` クライアント + LLM 呼び出し）のみで完結し、`[lightning-rl]`（VERL / torch / vLLM）を要求しない。

### FR-4: RL（LightningRL on VERL・モデル更新・系全体 end-to-end + 対象エージェント選択・[lightning-rl] が必要）
- ユーザーストーリー: モデル最適化担当者として、エージェント（単一 / ハンドオフを含む系全体）のモデル重みを Agent Lightning の RL（LightningRL on VERL）で更新したい。なぜなら報酬に基づき系全体のポリシーを最適化し、より良いルーティング・応答を得たいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `algorithm="rl"` で `optimize` を実行する THEN 最適化機能は RL ループを agent-lightning の Trainer（VERL）へ委譲し、出力（checkpoint パス / OpenAI 互換エンドポイント等）を `model_ref` として参照で返す。結果には train 上のスコア / 履歴と `val_score`（`val` 省略時は None）を併せて含める。
  - [ ] WHEN 利用者がグラフ（`HandoffGraph` / `WorkflowGraph`）を対象に RL を実行する THEN rollout は系全体（ハンドオフ経由）を実行し、LightningRL が軌跡全体の各エージェントの LLM 呼び出しに報酬を credit assignment して系全体のポリシーを最適化する。
  - [ ] WHEN 利用者が学習対象エージェント（`target_agents`）を指定する THEN 当該エージェント集合のモデルを重み更新の対象とする。既定は系内の全エージェント、またはサブセット選択が可能であり、対象外エージェントのモデルは凍結 / 現状のまま rollout に用いる。共有モデル / エージェント別モデルの別は利用者設定（passthrough）とする。
  - [ ] WHEN RL の出力（更新済みモデル）を返す THEN lib 内にモデル重みを保持せず、checkpoint パス / エンドポイント等の参照を plain データとして返す（target agents ごとに識別可能な参照を含む）。モデルは従来どおり外部 DI 流入とし、lib がモデルをホスティング / サービングしない。checkpoint 等の物理出力先は Trainer / Store 設定の passthrough である（FR-2・NFR-7 と整合）。
  - [ ] IF `[lightning-rl]` extra（VERL / torch / vLLM）が未導入のまま RL を要求する THEN 最適化機能は必要 extra（`oai-agentspec[lightning-rl]`）の導入を促す明確なエラーメッセージを返し、未捕捉例外でプロセスを停止しない。
  - [ ] WHEN RL の rollout を実行する THEN 宣言エージェント（系全体ではハンドオフを通る系全体）を `_adapters/` → `Runner.run` で実行し、実行結果から reward 算出に必要な plain データを抽出して利用者供給の reward へ渡す（Training-Agent Disaggregation の agent 側 rollout に相当）。
  - [ ] WHEN RL の rollout を実行する THEN 合成済み instructions（静的 / 動的いずれも `AgentSpec` 構築結果）をそのまま実行コンテキストとして用い、RL はモデル重みを最適化対象とする（プロンプトテキストを最適化しない）。プロンプト合成に対する特別扱い（スロット / rebind / vars）を要しない。
  - [ ] WHEN RL の重依存（verl / torch / vLLM 等）を参照する THEN それらの import は `_adapters/` 配下に閉じ、最適化ロジック層へ重依存型を出さない（SDK / 外部クライアント隔離との整合）。系全体最適化・target agents 選択も agent-lightning の Trainer / VERL へ委譲し、lib は独自の credit assignment や最適化ループを実装しない。

### FR-5: rollout と reward / train・val の利用者供給
- ユーザーストーリー: lib 利用者として、rollout（タスク実行）・reward（報酬）・データ（train / val）を自分で供給したい。なぜなら最適化の目的関数と入力ケースは自分のドメインに固有であり、lib にハードコードされたくないから。
- 受け入れ基準:
  - [ ] WHEN 利用者が `train`（最適化 / rollout に使う入力ケース群・必須）を渡す THEN 最適化機能は当該データをそのまま rollout 入力に用い、lib 側にハードコードしたケースを混入させない。`optimize` のデータ入口は `train` / `val` の 1 通りに統一し、`dataset=` + `val_split=` のような暗黙分割パラメータを持たない。
  - [ ] WHEN 利用者が `val`（最良候補の選定と汎化スコア確認に使う入力ケース群・**必須**）を渡す THEN 最適化機能は `val` 上で候補を選定・評価し、結果に `val_score` を含める。APO（agent-lightning 0.3 系）の beam search は validation セットを必須要件とするため、本 extra は `val` を必須とする。
  - [ ] IF `val` が省略 / 空である THEN 最適化機能は `OptimizeError(FailureKind.CONFIG_MISSING)` で fail-closed する（自動分割や `train` 流用といった暗黙のフォールバックをしない）。利用者は `train_val_split` 等で明示的に分割する。
  - [ ] WHEN dataset の各ケースのフィールド（例: `expected`）を扱う THEN それらはフレームワーク予約キーではなく利用者定義のフィールドであり、`reward`（または reward ファクトリ）が解釈する。lib は `expected` 等の固有キーを予約せず、特定のスキーマを強制しない。
  - [ ] WHEN 利用者が reward（報酬算出ロジック）を渡す THEN 最適化機能は rollout の実行結果から抽出した plain データを reward へ渡して報酬を得る。lib は報酬関数を内蔵しない（reward ファクトリは FR-9 のとおり callable を生成するヘルパであり、報酬データを内蔵しない）。
  - [ ] WHEN rollout を実行する THEN 宣言エージェントを `_adapters/` → `Runner.run` で実行し、生の実行結果は `_adapters/` 内で消費して実行経路（ルーティング）とツール呼び出し列を plain データとして 1 パスで抽出し、reward / 最適化ロジック層へは SDK 型を出さない（SDK 隔離との整合・llmops の実行トレース捕捉の流儀を再利用）。
  - [ ] IF `train` または reward が供給されない THEN 最適化機能は供給元が無い旨の明確なエラーを送出し、暗黙のフォールバックをしない。

### FR-6: rollout 安全性（任意の mock-approve / tool_mocks 再利用）
- ユーザーストーリー: lib 利用者として、RL のように同一 rollout を多数回実行する最適化で、危険ツールの副作用が反復しないようにしたい。なぜなら本物の副作用を多数回起こさずに最適化を回したいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が任意で rollout 実行に `tool_mocks`（agent スコープのツール実行本体の代替）/ `approvals`（承認自動解決ポリシー・mock-approve 相当）を渡す THEN 最適化機能はそれらを rollout 実行に適用し、副作用のない代替で rollout を完了させる。`tool_mocks` / `approvals` を渡さなければ rollout は宣言どおりに実行される（既定挙動）。
  - [ ] WHEN ツールをモック差し替えする THEN ツールの実行本体だけを副作用のない代替へ差し替え、ツールの宣言メタデータ（名前・説明・引数スキーマ・承認要否）は変更しない。registry / グラフ / spec は複製経由で扱い、利用者が渡した状態を一切変更しない。
  - [ ] IF 承認ポリシーが承認を返したツールが実際にはモック差し替えされていない THEN 明確なエラーを送出し、本物の危険ツールを実行しない（fail-closed・安全不変条件）。
  - [ ] WHEN rollout 安全性ヘルパを提供する THEN llmops の `tool_mocks` / `approvals` 実装を再利用する経路とし、MVP を過剰に拡張する独自の安全機構を新設しない（rollout 副作用の安全性は最終的に利用者責任とし、本機能は任意の補助経路を提供するに留める）。

### FR-7: extra としての公開と未導入時に壊れない契約
- ユーザーストーリー: ライブラリ利用者として、最適化機能を任意導入したい。なぜなら最適化が不要な利用者に `agentlightning` クライアントや RL 重依存（VERL / torch / vLLM）を強制したくないから。
- extra 分割: 軽量コアは `oai-agentspec[lightning]`（`agentlightning` クライアント + LLM 呼び出し・APO はこの extra のみで完結）、RL 重依存は `oai-agentspec[lightning-rl]`（VERL / torch / vLLM・任意）に分ける。RL 込みは `oai-agentspec[lightning,lightning-rl]` で導入する。APO のみを使う利用者は `[lightning]` のみで完結し、RL 重依存を導入しない（「使わないなら入れない」opt-in extra 哲学・llmops / llmops-langfuse と同型分割）。
- 受け入れ基準:
  - [ ] WHEN 利用者が `pip install oai-agentspec[lightning]` を行う THEN 軽量コアとその依存（`agentlightning`）が導入され、APO（`algorithm="apo"`）がローカルで完結する。
  - [ ] WHEN 利用者が `pip install oai-agentspec[lightning,lightning-rl]` を行う THEN RL 重依存（VERL / torch / vLLM）が追加導入され、RL（`algorithm="rl"`）が利用できる。
  - [ ] WHEN lightning extra 未導入で `import oai_agentspec` を実行する THEN ImportError 等を起こさずコア宣言層の公開 API（コア `__all__`）が利用できる。
  - [ ] WHEN lightning extra 未導入で既存 extra（`oai_agentspec.runtime.cli` / `runtime.llmops` 等）を import する THEN lightning 依存に起因する破綻なく従来どおり動作する。
  - [ ] WHEN lightning extra 未導入で最適化機能の import 経路へアクセスする THEN 必要 extra（`oai-agentspec[lightning]`）の導入を促す明確なエラーメッセージを返す。
  - [ ] WHEN `[lightning-rl]` extra 未導入で RL を要求する THEN 必要 extra（`oai-agentspec[lightning-rl]`）の導入を促す明確なエラーメッセージを返す。APO 要求時は該当せず、`[lightning]` のみでローカル完結する。
  - [ ] WHEN 最適化機能の公開 API を追加する THEN コア宣言層の公開契約（コア `__all__`）を壊さず、lightning 公開 API（`optimize`・reward ファクトリ群・`prompt_slot`・`prompt_slot_factory`・`train_val_split`・結果型の `save` / `to_dict` を含む）は `runtime/lightning` の独立した公開窓口に集約する。

### FR-8: 失敗時の graceful degradation
- ユーザーストーリー: lib 利用者として、最適化の失敗時にプロセスが未捕捉例外で落ちないようにしたい。なぜなら extra 不在・設定不在・Trainer 例外を明確なエラーまたは skip として受け取り、運用を継続したいから。
- 受け入れ基準:
  - [ ] IF `[lightning]` / `[lightning-rl]` extra が不在の状態で最適化を要求する THEN 必要 extra の導入を促す明確なエラーを返し、未捕捉例外でプロセスを停止しない（FR-7 と整合）。
  - [ ] IF 最適化に必須の設定（train / reward / registry / algorithm 等）が不在 THEN 不足を示す明確なエラーを返し、暗黙のフォールバックをしない。
  - [ ] IF agent-lightning の Trainer が実行中に例外を送出する THEN 最適化機能は当該例外を捕捉して明確なエラー（最適化失敗の理由を含む）に変換するか、明示的に skip 結果へ倒し、未捕捉例外でプロセスを停止しない。
  - [ ] WHEN 失敗を返す THEN 失敗の種別（extra 不在 / 設定不在 / Trainer 失敗）を判別可能な構造化された結果またはエラーとして返す。

### FR-9: 使いやすさヘルパ（reward ファクトリ / prompt_slot / prompt_slot_factory / train_val_split / 結果保存・薄い opt-in 補助）
- ユーザーストーリー: lib 利用者として、よくある目的関数・合成プロンプトの最適化・グラフ全体のスロット生成・データ分割・結果の保存を 1 行で記述したい。なぜなら定型的な reward callable や seed 取得・固定部分との再合成・候補適用 rebind・build 宣言・train/val 分割・結果の永続化の手書きを毎回書かずに、最短経路で `optimize` を呼び結果を扱いたいから。
- 受け入れ基準:
  - [ ] WHEN 利用者が reward ファクトリ（例: `contains(field=...)` / `exact(field=...)` / `tool_match(field=...)` / `approval_match(field=...)` / `route_match(field=...)` / `last_agent_match(field=...)` / `judge(rubric, model)`）を呼ぶ THEN 当該ファクトリは利用者の dataset フィールド名や rubric を受け取って `reward` callable を生成して返す。`field` 引数は省略可能で、既定値は `OptimizeCase` の標準フィールド名（`expected_output` / `expected_tools` / `expected_route` / `expected_last_agent` / `expected_approvals`）に揃える（`OptimizeCase` 利用時はフィールド名を渡さずに呼べる）。lib にプロンプト・データ・報酬データをハードコードしない（FR-5 と整合）。手書き `reward` callable も従来どおり受ける。`approval_match` は `RolloutResult.fired_approvals`（中断時の承認ゲート発火集合・approve / reject を問わず収集）の recall を採点し、`route_match` は `RolloutResult.route_steps`（実行経路の agent 名フルパス）と期待経路の完全一致、`last_agent_match` は `RolloutResult.last_agent`（最終応答 agent）と期待 agent の一致を採点する（llmops `EvalCase` の `expected_tools` / `expected_route` / `expected_approvals` を APO データセットでも同じ発想で 1 ケースに集約し、複合 reward の合成基盤を提供する）。
  - [ ] WHEN 利用者が `OptimizeCase`（typed なケース型・`input` 必須 + `id` / `expected_output` / `expected_tools` / `expected_route` / `expected_last_agent` / `expected_approvals` / `metadata`）でデータセットを記述する THEN reward ファクトリは `OptimizeCase` の標準フィールド名を既定 `field` として参照し、フィールド名を渡さずに `contains()` / `tool_match()` / `route_match()` / `last_agent_match()` / `approval_match()` を呼べる。`optimizer` は `case` から `input` を `OptimizeCase.input` 属性 / dict `case["input"]` のいずれでも抽出する（dict ケースとの併存・後方互換）。`OptimizeCase` は plain `@dataclass(frozen=True)` で SDK 非依存（NFR-1 と整合）。
  - [ ] WHEN reward ファクトリ群・`prompt_slot`・`prompt_slot_factory`・`train_val_split` を公開する THEN これらは `runtime/lightning` の公開窓口に含め、コア宣言層の公開契約（コア `__all__`）には追加しない。
  - [ ] WHEN 利用者が `prompt_slot` ヘルパを呼ぶ THEN 当該ヘルパは `PromptStore` の公開 `compose`（必要に応じ `get`）を読み取り、seed・固定部分との再合成・候補適用 rebind を内包し、`build`（候補 instructions → `AgentSpec`）を受けて slot を構成する。`prompt_slot` は `PromptStore` を内省・書き換えせず、公開メソッドの戻り値を読むのみである。
  - [ ] WHEN `prompt_slot` で対象が registry 登録済みかつ `build` が省略される THEN 既定 build は登録 `AgentSpec` を複製し `instructions` のみ候補で差し替える（tools / handoffs / model 等は登録 spec から複製され、利用者は再宣言しなくてよい）。既定 build は `optimize` / `prompt_slot_factory` に渡る registry から対象 spec を解決する。IF 登録 spec が見つからず `build` も省略される THEN 明確なエラー（解決不能の理由を示す・fail-closed）を送出する。利用者は従来どおり `build=` を明示して動的構築もできる（パワーユーザー経路・併存）。
  - [ ] WHEN `prompt_slot` が vars を扱う THEN vars を seed に展開せず `${var}` プレースホルダを保持し、rollout 時に内部で vars を再注入する。利用者は `vars` を `prompt_slot` に渡すだけでよく、`build`（既定 build を含む）内で vars を再注入する必要はない（vars は最適化対象外・rollout 再注入をヘルパが担保）。
  - [ ] WHEN 利用者が `prompt_slot_factory(store, registry, **defaults)` を呼ぶ THEN 共通既定値（`prompt_slot` の全 kwarg）を束ねた `make(agent, **overrides) -> Slot` の callable を返し、利用者は dict comprehension（例: `{name: make(name) for name in [...]}`）で `{名前: slot}` の mapping を組み立てられる。各 slot は `prompt_slot` 相当（seed = 対象セグメントの vars 未展開テンプレート・`${var}` 保持・既定 build = 登録 spec 複製で instructions 差し替え）である。`runtime/lightning` 公開窓口に含め、`PromptStore` は公開 `compose` / `get` を読み取るのみ（非改変）。
  - [ ] WHEN 組み立てた mapping を `optimize(graph, slot=slots, ...)` に渡す THEN rebind 自動導出（本 FR の既存基準）と合わせて、手書きの rebind / build なしでグラフ全体 APO が単一呼び出しで成立する。最適化対象は mapping に掲載したエージェントのみであり、未掲載のエージェントのプロンプトは固定とする。
  - [ ] WHEN スロットが `prompt_slot`（または `prompt_slot_factory` 生成）の戻り値（単一 / mapping）である THEN `prompt_slot` は `build`（既定 build を含む）を内包するため、フレームワークが各スロットの `build` から rebind を自動導出し、利用者は手書きの rebind を渡さなくてよい（単一 / 複数スロット mapping いずれでも自動導出）。手書き rebind が必要なのはスロットが生 seed（`build` を持たない）であるパワーユーザー経路のときのみである。
  - [ ] WHEN 複数エージェントのプロンプトを系全体で最適化する THEN 利用者は `prompt_slot_factory` + dict comprehension（または各エージェントに `prompt_slot`）を使い、`{名前: slot}` の mapping として `optimize` に渡せる（FR-3 の複数スロットと整合）。`build` から rebind が自動導出されるため `optimize(graph, slot=slots, ...)` の単一呼び出しで完結する。単一スロットの簡単版も併存する。
  - [ ] WHEN 利用者が `train_val_split(data, *, val_ratio=0.2, seed=0, shuffle=True)` を呼ぶ THEN ヘルパは `(train, val)` のタプルを返す（引数名・既定は暫定）。`seed` 固定で決定的に分割する。利用者は自前分割（例: スライス・層化・時系列）の結果も同じく `train` / `val` として `optimize` に渡せる。
  - [ ] WHEN `train_val_split` を実行する THEN 純データ操作に徹し、SDK / `PromptStore` / 外部クライアントに触れない（依存方向・隔離方針に影響しない）。
  - [ ] WHEN 利用者が結果に対して `result.save(path)` を明示的に呼ぶ THEN 当該結果を利用者指定パスへ書き出す。APO の場合は `result.prompt`（最適化済みスロットテキスト・`${var}` プレースホルダ保持・複数スロット時は名前付き mapping）を当該パスへ書き、RL の場合はモデル重みを書かずメタデータ / サマリ（`model_ref`・train スコア / 履歴・`val_score` 等）を当該パスへ書く。`save` を呼ばない限り何も書かない（既定は戻り値のみ・opt-in 書込）。
  - [ ] WHEN `result.save(path)` が書き込む THEN 利用者が渡したパスにのみ書き、`PromptStore` のテンプレートやライブラリ管理領域を一切書き換えない（PromptStore 非書込・モデル重み非保持と整合）。
  - [ ] IF `result.save(path)` のパスが書込不能 / 不正である THEN 明確なエラー（書込先の問題を示す）を送出する（fail-closed）。
  - [ ] WHEN 利用者が結果を plain dict として扱う THEN 任意で `result.to_dict()` 相当（結果を plain dict として取得しログ / 外部保存に使える・シグネチャは暫定）を提供してよい。
  - [ ] WHEN 利用者が slot（`prompt_slot` / `prompt_slot_factory` で生成した `Slot`、単一の静的 str、または `{名前: slot}` の mapping）を渡す THEN slot は常に `slot=` キーワードで渡し、`optimize` の第1引数は最適化対象（`AgentSpec` / `WorkflowGraph` / `HandoffGraph`）とする（FR-1 と整合・slot を第1引数にしない）。`prompt_slot` 利用時は `build` から rebind が自動導出されるため `rebind` を別途渡す必要はない（rebind の冗長排除・生 seed 経路でのみ rebind を明示）。
  - [ ] WHEN 生の `slot` / `rebind` / `reward` callable を直接渡す THEN ヘルパを経由せずに従来どおり受理し、ヘルパとパワーユーザー経路が併存する。
  - [ ] WHEN ヘルパ（reward ファクトリ / `prompt_slot` / `prompt_slot_factory`）が `PromptStore` に触れる THEN 既存 `PromptStore`（`src/oai_agentspec/prompts.py`）のクラス・メソッド・役割を一切変更せず、公開メソッドの読み取りに限定する（依存方向 `runtime/lightning → core(prompts)` の一方向を守り、core は lightning を逆参照しない）。

## 3. 非機能要件

### NFR-1: セキュリティ（SDK / 外部クライアント隔離）
- 要件: 最適化機能による LLM / agent-lightning / RL 重依存（verl / torch / vLLM）/ SDK 呼び出しは `_adapters/` 配下のみを窓口とし、最適化ロジック層は plain データと不透明型のみを扱う。`from agents` / `from openai` / `import agentlightning` / `import verl` / `import torch`（RL 系）の直接 import を最適化ロジック層に持ち込まない。algorithm → Trainer / 最適化器の対応は抽象識別子として保持し、agent-lightning クラスへの解決は `_adapters/` 配下に閉じる。
- 計測基準: `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` の結果が空であること（最適化機能追加後も空を維持）。外部クライアント（`agentlightning`）および RL 重依存（`verl` / `torch` / `vllm`）の import も `_adapters/` 配下に閉じること（同様の grep で確認）。

### NFR-2: 可用性（コア / 既存 extra が lightning 未導入で壊れない）
- 要件: lightning / lightning-rl extra 未導入の環境で `import oai_agentspec` および既存 extra（conversation / serve / cli / llmops）の import が成功する。最適化依存は遅延 import 境界で隔離する。
- 計測基準: `uv run python -c "import oai_agentspec as m; assert all(hasattr(m,s) for s in m.__all__)"` が lightning extra 未導入でも成功すること。lightning 依存をアンインストールした状態（または非導入を模した環境）でコア公開 API スモークが緑であること。

### NFR-3: 可用性（外部依存非在時の graceful degradation）
- 要件: 最適化を軽量コア（APO・`[lightning]` でローカル完結）と RL 重依存（`[lightning-rl]`・任意）に分離する。`[lightning]` extra・`[lightning-rl]` extra・LLM API・必須設定（train / reward / registry）のいずれかが不在でも、明確なエラー / skip に倒し、不在や Trainer 例外に起因する未捕捉例外でプロセスを停止させない。APO は `[lightning]` のみでローカルに完結する。
- 計測基準: `[lightning-rl]` 未導入で RL を要求したときに明示エラーを返すこと、APO が `[lightning]` のみで完結すること、必須設定不在時に明示エラーを返すこと、Trainer 例外時に明確なエラー / skip へ倒してプロセスを落とさないことを、それぞれテストで検証する（実 GPU / 実訓練 / 実通信なしの fake / モック層で再現）。

### NFR-4: 保守性（env 参照の境界）
- 要件: 本体（コア宣言層）の env 非依存方針と整合させ、最適化実行に必要な env 参照は最適化機能の実行境界（`runtime/lightning` の `optimize` エントリ / `_adapters` の設定ヘルパ）に閉じる。コアの宣言層および `_adapters` の既存契約に env 依存を波及させない。
- 計測基準: 最適化機能の env 参照箇所が `runtime/lightning` サブパッケージの境界に限定されることをコードレビューで確認する。コア層の既存 env 非依存テストが緑を維持すること。

### NFR-5: 保守性（テストカバレッジ / リント）
- 要件: 最適化機能追加後もプロジェクトのテストカバレッジ閾値とリント基準を維持する。実 GPU / 実訓練 / 実通信に依存しない単体・統合テスト（fake / Trainer モック）でレイヤを検証する。
- 計測基準: `uv run pytest` がカバレッジ 80% 以上（`fail_under = 80`）で緑であること。`uv run ruff check src/ tests/` において、`runtime/lightning` サブパッケージおよび追加テストファイルが lint をパスし、既存ファイルへ新規の lint 違反を持ち込まないこと（本変更で新たに増える違反が 0 件であること）。

### NFR-6: 保守性（実行は _adapters 経由・build-don't-run / runtime 実行寄り層への整合）
- 要件: lightning は独自の最適化エンジン / 実行エンジンを持たず、最適化ループ本体（系全体の credit assignment・複数プロンプト最適化を含む）は agent-lightning の Trainer へ委譲し、rollout 実行は `_adapters/` 経由で SDK `Runner.run` へ結線する。公開の最適化実行 API は `runtime/lightning` サブパッケージ側に置き、コア宣言層には最適化 / 実行 API を追加しない。lightning は runtime/ 配下の実行寄り層の一員として、conversation / serve / cli / llmops と同じ整合方針に従う。
- 計測基準: コア宣言層の公開契約（コア `__all__`）に最適化実行 API（`optimize`・reward ファクトリ・`prompt_slot`・`prompt_slot_factory`・`train_val_split`・結果型の `save` を含む）が含まれないこと、最適化実行 API が `runtime/lightning` 配下のみに存在すること、最適化ループ本体・credit assignment を独自実装せず agent-lightning Trainer へ委譲していることをコードレビュー + 公開 API スモークで確認する。

### NFR-7: 性能（実行制御・タイムアウト・ストレージ passthrough）
- 要件: 最適化実行（rollout / Trainer）の並列度・訓練ラウンド数・タイムアウト・Store（agent-lightning の InMemory / Sqlite / Mongo）・target agents 選択・共有 / エージェント別モデルの別等の実行制御設定を利用者設定の passthrough として受け、未指定時は agent-lightning の既定を適用する。RL の checkpoint 等の物理出力先も Trainer / Store 設定の passthrough であり、lib は保管を所有しない。rollout / 採点 LLM 呼び出しにタイムアウトを設定可能とし、タイムアウト到達時は FR-8 の失敗倒し（明確なエラー / skip）に従う。
- 計測基準: 並列度・訓練ラウンド数・タイムアウト・Store・target agents 等の設定が agent-lightning Trainer へ passthrough されること、未指定時に既定が適用されること、タイムアウト到達時に NFR-3 / FR-8 の graceful degradation に従うことを、fake / Trainer モックで検証する（タイムアウト基準は FR-8 と矛盾させない）。

### NFR-8: セキュリティ（rollout 副作用の反復に対する安全方針）
- 要件: RL のように同一 rollout を多数回実行する最適化において、危険ツールの副作用が反復しないよう、利用者が任意で `tool_mocks` / `approvals`（mock-approve 相当）を rollout 実行へ適用できる経路を提供する。承認を認可するのは実際にモック差し替えされた `(agent_name, tool_name)` に限り、未差し替えの承認は fail-closed エラーとする（安全不変条件）。rollout 副作用の安全性は最終的に利用者責任とし、本機能は補助経路の提供に留める。
- 計測基準: `tool_mocks` / `approvals` を渡した rollout が副作用のない代替で完了すること、未差し替えツールの承認が fail-closed エラーになること、`tool_mocks` / `approvals` 未指定時は宣言どおりに rollout が実行されることを、fake ツール / モック層で検証する。

### NFR-9: 保守性 / 使いやすさ（最短経路と段階的な拡張余地）
- 要件: 最短ケースは `optimize(宣言物, algorithm=..., train=..., reward=...)` の単一呼び出しで完結する。よくある目的関数は reward ファクトリ（FR-9）で 1 行記述でき、合成プロンプト（`PromptStore`）の最適化は `prompt_slot` ヘルパ（FR-9）で定型（seed 取得・固定部分との再合成・候補適用 rebind・既定 build による spec 複製・vars 再注入・冗長な rebind 受け渡し）を畳み、グラフ全体 APO は `prompt_slot_factory` + dict comprehension による mapping 生成 + 自動 rebind で最短化（単一の生成経路 + `optimize` の単一呼び出しで完結）でき、train / val 分割は `train_val_split`（FR-9・決定的・opt-in）で、結果の永続化は `result.save(path)`（FR-9・opt-in）でそれぞれ 1 行記述できる。系全体のプロンプト最適化では `prompt_slot_factory` + dict comprehension（または各エージェントの `prompt_slot`）で `{名前: slot}` の mapping を生成して渡せ、`prompt_slot` は `build`（既定 build を含む）を内包するため手書き build / rebind なしで完結する。vars は最適化対象外（不変）で rollout 時に再注入され、APO 出力は `${var}` プレースホルダを保持する。1 エージェント / 1 セグメントだけ調整する単一スロットの簡単版も併存する。`optimize` のデータ入口は `train` / `val` の 1 通りに統一し、暗黙の分割パラメータを持たない。結果は既定で戻り値のみ・lib 自動書込なし。生の `slot` / `rebind` / `reward` callable も併存して受け、パワーユーザーの拡張余地を残す。ヘルパは `PromptStore` の公開メソッドを読み取るのみで、`PromptStore` を改変しない。
- 計測基準: 静的 instructions の APO / RL が `optimize(..., train=..., reward=...)` 単一呼び出し + 既製 reward ファクトリで記述できること、`PromptStore` 合成ケースが `prompt_slot` ヘルパ + `optimize` 単一呼び出しで記述できること、`prompt_slot_factory` + dict comprehension で生成した mapping + `optimize` でグラフ全体 APO が手書き build / rebind なしで成立すること、`prompt_slot` の build 省略時に登録 spec を複製して tools / handoffs を保持すること、registry 未解決 + build 省略で fail-closed になること、`prompt_slot` を使う系全体最適化が手書き rebind なしで `optimize(graph, slot={名前: prompt_slot(...)}, ...)` の単一呼び出しで成立すること、生 seed 経路では rebind の明示が必要なこと、vars が最適化対象に入らず APO 出力が `${var}` プレースホルダを保持すること、`train_val_split` が決定的に `(train, val)` を返すこと、自前分割の `train` / `val` も受理されること、`val` 省略時に warning を出し自動分割しないこと、生 callable 経路（手書き `reward` / `rebind` / `slot`）も併存して動作することを、テスト / 利用イメージ（第 7 章）で確認する。`prompt_slot` / `prompt_slot_factory` / reward ファクトリが `PromptStore` の公開メソッド読み取りに限定され書き込みをしないこと、`train_val_split` が純データ操作であること、`result.save(path)` が利用者指定パスへ書き未呼び出し時は何も書かず `PromptStore` を触らないことをテストで確認する。

## 4. 制約事項

- 技術的制約:
  - 本要件は実行寄り層 `runtime/` 集約を前提とし、lightning は `runtime/` 配下に `runtime/lightning` として追加する。
  - lightning は独自の最適化エンジン / 実行エンジンを持たず、最適化ループ本体（系全体の credit assignment・複数プロンプト最適化を含む）は agent-lightning の Trainer へ委譲し、rollout 実行は `_adapters/` 経由で SDK `Runner.run` へ薄く結線する公開の実行寄り層である。build-don't-run の原則（コア宣言層は独自の最適化 / 実行エンジン・公開の実行 API を持たない）はコア側で維持し、緩和しない。
  - SDK / 外部クライアント呼び出しは `_adapters/` 経由（NFR-1）。最適化ロジック層は plain データ / 不透明型のみを扱う。
  - モデル外部流入を維持する。RL の出力（更新済みモデル）は lib 内に重みを保持せず checkpoint パス / OpenAI 互換エンドポイント等の参照を plain データで返し、モデルは従来どおり外部 DI 流入とする。RL の checkpoint 等の物理出力は Trainer / Store 設定の passthrough であり、lib は保管を所有しない。系全体 RL の target agents 選択・共有 / エージェント別モデルの別も利用者設定の passthrough とする。
  - プロンプト非同梱方針を維持する。APO の最適化対象は利用者指定のプロンプトスロット（単一 / 複数 mapping）+ rebind とし、`prompt_slot` 利用時は各スロットの `build`（既定 build を含む）から rebind を自動導出する（手書き rebind は生 seed 経路でのみ）。`build` 省略時の既定 build は registry 登録 `AgentSpec` を複製して `instructions` のみ差し替えるものであり、lib が新規にプロンプト・ツール・モデルを内蔵するものではない（複製元は利用者宣言の登録 spec）。vars 値は APO の最適化対象外（不変）で、最適化対象は vars 未展開のテンプレート文言（`${var}` プレースホルダ保持）であり、vars は rollout 時に再注入する。lib は `PromptStore` の合成内部に踏み込まない（テンプレートを内省・書き換えしない）。`PromptStore` は利用者の Single Source of Truth のままとする。APO の出力（最適化済みスロットテキスト）は `${var}` を保持した plain な文字列 / 名前付き mapping として返し、lib が `PromptStore` へ書き込まない（永続化は利用者責任）。最適化対象プロンプト（スロット）・rebind・vars・データ（train / val）・reward は利用者が渡す（lib にケース・プロンプト文字列・報酬関数をハードコードしない）。
  - 結果の出力は既定で戻り値（plain データ）のみとし、lib はファイル・`PromptStore`・外部ストアへ自動書き込みをしない。`result.save(path)` は利用者指定パスへの opt-in 書込であり、`PromptStore` のテンプレートやライブラリ管理領域を触らない。APO はプロンプトテキスト（`${var}` 保持）を、RL はメタデータ / 参照（重みは書かない）を書く。RL の checkpoint の物理出力は Trainer / Store 設定の passthrough で lib は重みを保持しない。
  - 既存 `PromptStore`（`src/oai_agentspec/prompts.py`）は本要件の変更対象外とする。`runtime/lightning` 側に足す使いやすさヘルパ（reward ファクトリ・`prompt_slot`・`prompt_slot_factory`・`train_val_split`・結果型の `save` / `to_dict`）のうち、`PromptStore` に関わるもの（`prompt_slot` / `prompt_slot_factory`）は既存公開メソッド（`compose` / `get`）を読み取るのみで `PromptStore` のクラス・メソッド・役割を改変しない（`train_val_split` は純データ操作・`save` は利用者指定パスへ書くのみで、いずれも `PromptStore` に触れない）。既定 build は registry（利用者宣言の `AgentSpec`）を複製するのみで `PromptStore` を書き換えない。依存方向は `runtime/lightning → core(prompts)` の一方向を守り、core は lightning を逆参照しない。
  - データ入口は `train`（必須）/ `val`（任意）の 1 通りに統一する。`optimize` 内に `dataset=` + `val_split=` のような暗黙分割パラメータを設けない。分割は独立ヘルパ `train_val_split` または利用者自前で行い、結果の `train` / `val` を `optimize` に渡す。
  - reward ファクトリ（`contains` / `exact` / `tool_match` / `judge` 等）は利用者の dataset フィールド名や rubric から `reward` callable を生成するヘルパであり、報酬データ・プロンプト・データを lib に内蔵しない。dataset の固有フィールド（例: `expected`）は利用者定義であり、lib は予約キーを設けない。
  - プロンプト最適化責務は Lightning 側に集約する。プロンプト自動改善 APO を含むエージェント自動作成支援機能は本機能の Lightning APO を内部利用する上位 UX として位置づけ、独自 APO エンジンを別実装しない棲み分けとする。
  - 外部依存（`agentlightning` / VERL / torch / vLLM / Azure・OpenAI）は optional extra として許容する。軽量コア（`[lightning]` = `agentlightning` クライアント + LLM 呼び出し）と RL 重依存（`[lightning-rl]` = VERL / torch / vLLM）は別 extra に分割する。コア `import oai_agentspec` と既存 extra（conversation / serve / cli / llmops）は lightning extra 未導入でも壊れない（NFR-2）。
  - extra の宣言は `pyproject.toml` の `[project.optional-dependencies]` に既存 extra（conversation / serve / cli / llmops / llmops-langfuse）と同じ階に並べて追加する。遅延 import 境界の運用は runtime 実行寄り層の方式を踏襲し、`agentlightning` / RL 重依存のトップ import を本体に持ち込まない。
  - 公開関数名は `optimize` とし、Python 組込みと衝突する `eval` 等をシンボル・モジュール名に採用しない。
  - dist 名は `oai-agentspec`、パッケージ名は `oai_agentspec`。型チェッカ（mypy）は導入しない（既存方針）。
- スコープ外（本要件に含めない）:
  - 独自 RL アルゴリズムの実装（最適化アルゴリズム・credit assignment は agent-lightning に委譲する）。
  - モデル重みのホスティング / サービング基盤（RL 出力は参照のみを返す）。
  - 訓練サーバインフラ自体（Training-Agent Disaggregation の server 側・agent-lightning へ委譲する）。
  - GPU / クラスタの運用・プロビジョニング。
  - プロンプト自動改善 APO を含むエージェント自動作成支援機能の上位 UX 実装（本機能は Lightning APO のエンジンを提供し、上位 UX はそれを内部利用する別作業）。
  - APO 出力プロンプトの永続化・プロンプトバージョニング（snapshot / list / diff / rollback / push）。利用者が `result.save(path)` または `PromptStore` 等で永続化する（本機能の出力はローカル戻り値 + opt-in `save` に限る）。
  - vars 値（`${var}` 置換値）の最適化。vars は APO の最適化対象外（不変）で、各 rollout で再注入することで確定し、最適化対象は vars 未展開のテンプレート文言に限る。
  - 観測連携（Langfuse 等の外部観測 SaaS への送信）。これは llmops 側の関心事であり本要件のスコープ外とする。本機能の結果出力はローカルの戻り値と opt-in `result.save(path)` に限る。
  - `optimize` 内蔵の暗黙データ分割（`dataset=` + `val_split=` 等）。分割は独立ヘルパ `train_val_split` または利用者自前で行う。高度な分割戦略（層化・時系列・交差検証）の内蔵もスコープ外で、利用者が自前で `train` / `val` を構成する。
  - `PromptStore` クラス自体の改変。使いやすさヘルパ（reward ファクトリ・`prompt_slot`・`prompt_slot_factory`・`result.save`）は `PromptStore` を内省・書き換え・合成介入しない（前項のスコープ外と整合）。
- ビジネス制約:
  - 出力（コード・ドキュメント・コミット・Issue・PR 等）に絵文字を含めない。AI が生成したことを示唆する文言を含めない。

## 5. 影響範囲

- 関連コンポーネント:
  - `pyproject.toml`: `[project.optional-dependencies]` に軽量コア `lightning`（`agentlightning`）と RL 重依存 `lightning-rl`（VERL / torch / vLLM）の 2 extra を追加する。既存の conversation / serve / cli / llmops / llmops-langfuse extra と同じ階に並ぶ。
  - `src/oai_agentspec/runtime/lightning`: 最適化機能サブパッケージを runtime/ 配下に新設し、独立した公開窓口（サブパッケージ `__init__`）に最適化 API（`optimize`）・結果型（train スコア / 履歴・`val_score`・APO の最適化済みスロットテキスト（`${var}` 保持・単一 / 名前付き mapping）・RL の `model_ref`（target agents ごと識別可能）・opt-in 保存ヘルパ `save(path)` と任意の `to_dict()`）・設定型（実行制御 / Store / タイムアウト / 並列度 / target agents 選択 / 共有・エージェント別モデルの passthrough）・APO のスロット（単一 / mapping）・vars（不変・rollout 再注入）・rebind（`prompt_slot` 利用時は `build` から自動導出・生 seed 経路では単一 / 候補 mapping を受ける）受け口・使いやすさヘルパ（reward ファクトリ群 `contains` / `exact` / `tool_match` / `judge` 等・`prompt_slot`（`build` 省略時は registry 登録 spec 複製で instructions 差し替えが既定）・`prompt_slot_factory`・`train_val_split`）・rollout 安全性経路（`tool_mocks` / `approvals` 受け口）を集約する。コアの宣言層公開契約（コア `__all__`）を汚さない。公開関数名は `optimize` とし、`eval` をシンボル・モジュール名に採用しない。
  - `src/oai_agentspec/_adapters/`: agent-lightning 窓口（`import agentlightning` を局在化・Trainer / APO（単一・複数プロンプトリソース）・RL アルゴリズム（系全体 credit assignment・target agents）への結線）・RL 重依存窓口（`verl` / `torch` / `vllm` を関数内遅延 import）・実行トレース捕捉窓口（生の rollout 実行結果を `_adapters` 内で消費し plain な実行経路 / ツール呼び出し列を抽出・llmops の捕捉実装を再利用）を追加（既存 `models` / `responses` / `runner` 等と同列の adapter モジュール）。
  - `src/oai_agentspec/prompts.py`（`PromptStore`）: 本要件の変更対象外。`prompt_slot` / `prompt_slot_factory` 等の使いやすさヘルパは公開 `compose`（必要に応じ `get`）を読み取り参照するのみで、`PromptStore` のクラス・メソッド・役割を改変しない。`result.save(path)` も利用者指定パスへ書くのみで `PromptStore` を触らない。
  - `AgentRegistry`（`registry`）: 既定 build はここから対象 `AgentSpec` を解決して複製し（`instructions` のみ候補で差し替え・tools / handoffs / model 等を保持）、`prompt_slot_factory` も同じ registry を参照する。registry は読み取り・複製経由で扱い、利用者が渡した registry / 登録 spec を一切変更しない。
  - 最適化対象となる宣言物の型（`spec.AgentSpec` / `handoffs.HandoffGraph` / `workflow.WorkflowGraph` / `prompts.PromptStore`）は読み取りのみで参照し、改変しない。とりわけ `AgentSpec.instructions`（静的 str / `(context, agent) -> str` callable）・`prompts.PromptStore.compose` は読み取り参照のみとし、APO の最適化対象は vars 未展開のテンプレート文言（`${var}` 保持）で vars は rollout 再注入する。候補スロットの適用は `prompt_slot` の `build`（既定 build を含む）から自動導出される rebind（または生 seed 経路の利用者供給 rebind・系全体は候補 mapping）を通じて行う（lib は `PromptStore` テンプレートを書き換えず、合成内部に踏み込まない）。
  - rollout 安全性は `runtime/llmops` の `tool_mocks` / `approvals` 実装を再利用する（重複実装をしない）。
  - `docs/`: `docs/architecture.md`（最適化レイヤを runtime 実行寄り層の一員として、extra 構成と整合する現在仕様として記述）、`docs/requirements/`（本要件の反映）。`docs/rationale/agt-governance-integration.md`（Agent Lightning が LLMOps トラックへ振り分けられた経緯）への相互参照。Spec 駆動規約（現在仕様の SoT・履歴記述や Issue 番号入りファイル名の禁止）に従う。
  - `tests/`: src ミラー構造（runtime 構造に追従）に最適化機能テストを追加（fake / Trainer モックで実 GPU / 実訓練 / 実通信を排した unit / integration）。使いやすさヘルパ（reward ファクトリ・`prompt_slot`・`prompt_slot_factory`・`train_val_split`）が `PromptStore` の公開メソッド読み取りに限定され書き込みをしないこと（`train_val_split` は純データ操作で決定的であること）、既定 build の spec 複製・tools / handoffs 保持・registry 未解決 + build 省略の fail-closed・`prompt_slot_factory` による複数 slot 生成、`result.save(path)` が利用者指定パスへ書き未呼び出し時は何も書かず `PromptStore` を触らないこと、vars が最適化対象に入らず APO 出力が `${var}` プレースホルダを保持すること・候補が必要プレースホルダを失った場合に無効化 / 低評価（fail-closed）になること、系全体最適化（`prompt_slot_factory` + dict comprehension で生成した複数スロット APO は手書き build / rebind なしで成立・target agents RL）・単一スロットの簡単版・生 seed 経路の rebind 明示・最短経路 / 生 callable 経路の併存、`val` 省略時の warning も検証する。
- 既存機能への影響:
  - コア宣言層の公開契約（コア `__all__`）と既存シンボルの振る舞いは不変に保つ（契約）。lightning の公開 API（`optimize`・reward ファクトリ・`prompt_slot`・`prompt_slot_factory`・`train_val_split`・結果型の `save` / `to_dict` を含む）はコア `__all__` に追加せず `runtime/lightning` の公開窓口に集約する。
  - 既存 extra（conversation / serve / cli / llmops / llmops-langfuse）の import 契約・遅延 import 境界に回帰を与えない。
  - 単方向依存（コア宣言層は runtime / lightning へ依存しない。依存方向は runtime/lightning → コア宣言層 / `_adapters` の一方向）を壊さない。最適化サブパッケージは宣言物を読み取り依存し、コア層を逆依存させない。使いやすさヘルパも `runtime/lightning → core(prompts)` / `runtime/lightning → core(registry)` の一方向で `PromptStore` 公開メソッド読み取り・登録 spec 複製のみとする（`train_val_split` は core 非依存の純データ操作・`result.save` は利用者指定パスへ書くのみ）。
  - `prompts.PromptStore` の役割・クラス・メソッドは不変に保つ（プロンプトの利用者 Single Source of Truth・lib 非書込）。reward ファクトリ・`prompt_slot` / `prompt_slot_factory`・`result.save` は `PromptStore` の合成・書き込み契約を変更しない。`AgentRegistry` も読み取り・複製経由で扱い、登録 spec を変更しない。
  - 横断評価 / 横断最適化での registry / グラフの扱い（`HandoffGraph` は `apply(registry)` でエッジ反映・`WorkflowGraph` は registry を伴う Agent 化）は読み取り・複製経由とし、利用者状態を汚さない。
  - プロンプト自動改善 APO との関係: プロンプト最適化責務は本機能の Lightning APO に集約し、エージェント自動作成支援機能はその上位 UX として Lightning APO を内部利用する棲み分けとする（上位 UX 実装は本要件のスコープ外）。

## 6. 用語定義

| 用語 | 定義 |
|------|------|
| Agent Lightning | 任意フレームワークのエージェントを最適化する OSS（pip: `agentlightning` / dist `agent-lightning`）。APO / RL の両系統を担う単一の最適化エンジン。`import agentlightning` は `_adapters/` 配下に閉じる |
| LitAgent | Agent Lightning がエージェントをラップする抽象。rollout（タスク実行 + 報酬返却）を担う |
| rollout | 宣言エージェント（単一 / ハンドオフを通る系全体）を実行してタスクを遂行し、最適化のための実行結果（出力 / 経路 / ツール列）を得る単位。本機能では `_adapters/` → `Runner.run` で実行し plain データへ変換する。APO では各 rollout で vars を再注入する |
| 系全体最適化 | グラフ（`HandoffGraph` / `WorkflowGraph`）を対象に、ハンドオフを通る系全体を end-to-end に最適化すること。RL は credit assignment で軌跡全体の各エージェントの LLM 呼び出しに報酬を割り当て系全体のポリシーを最適化し、APO は複数スロット（`{名前: slot}` mapping・`prompt_slot_factory` + dict comprehension で一括生成可）で系全体のプロンプトを同時最適化する。最適化対象は利用者が明示列挙したエージェントのみ。単一エージェント / 単一スロットは簡単版として併存する |
| target agents（学習対象エージェント） | RL で重み更新の対象とするエージェント集合。既定は系内の全エージェント、サブセット選択も可能。対象外エージェントのモデルは凍結 / 現状のまま rollout に用いる。共有モデル / エージェント別モデルの別は利用者設定の passthrough |
| reward | rollout の良し悪しを表す報酬。利用者が供給し、lib は報酬関数を内蔵しない |
| reward ファクトリ | よくある目的関数を 1 行で記述するための callable 生成ヘルパ（例: `contains(field)` / `exact(field)` / `tool_match(field)` / `judge(rubric, model)`）。利用者の dataset フィールド名や rubric を受けて `reward` callable を返すだけで、報酬データ・プロンプト・データを lib に内蔵しない。`runtime/lightning` 公開窓口に含める |
| dataset | 入力ケース群の総称（概念）。`optimize` へは `train` / `val` に分けて渡す（`optimize` に `dataset=` 引数は持たない）。各ケースのフィールド（例: `expected`）は利用者定義で `reward`（または reward ファクトリ）が解釈し、lib は予約キーを設けない |
| train | 最適化 / rollout に使う入力ケース群（`optimize` の必須引数）。lib にハードコードしない利用者供給データ |
| val | 最良候補の選定と汎化スコア確認に使う入力ケース群（`optimize` の任意引数）。省略時は `train` で候補選定を行う旨を warning で通知し、`train` を黙って自動分割しない。`val_score` は省略時 None |
| train_val_split | sklearn 風の決定的なデータ分割ヘルパ（opt-in・`runtime/lightning` 公開窓口）。`train_val_split(data, *, val_ratio=0.2, seed=0, shuffle=True) -> (train, val)`（引数名・既定は暫定）。`seed` 固定で決定的。純データ操作で SDK / `PromptStore` / 外部クライアントに触れない。利用者自前分割（スライス・層化・時系列等）も同じく `train` / `val` として渡せる |
| val_score | `val` 上で測った最適化結果の汎化スコア。結果に含め、`val` 省略時は None。結果には train 上のスコア / 履歴も併せて含める |
| result.save | 最適化結果を利用者指定パスへ書き出す opt-in ヘルパ。APO はプロンプトテキスト（`result.prompt`・`${var}` プレースホルダ保持・複数スロット時は名前付き mapping）を、RL はメタデータ / 参照（`model_ref`・train スコア / 履歴・`val_score` 等。モデル重みは書かない）を書く。`PromptStore` のテンプレートやライブラリ管理領域は書き換えず、未呼び出し時は何も書かない（既定は戻り値のみ）。書込不能 / 不正パスは明確なエラー（fail-closed）。任意で `to_dict()` 相当（結果を plain dict として取得）を併せて提供してよい |
| APO | Automatic Prompt Optimization。beam-search + LLM テキスト勾配でプロンプトテキストを最適化する軽量系（GPU 不要）。最適化対象は利用者指定のプロンプトスロット（vars 未展開のテンプレート文言・`${var}` プレースホルダ保持）で、単一スロット（1 エージェント / 1 セグメント）に加え複数スロット mapping（`prompt_slot_factory` + dict comprehension で一括生成可）による系全体のプロンプト同時最適化に対応する。vars 値は最適化対象外（不変・確定）で各 rollout で再注入する。各 rollout への適用は rebind を通じて行い、`prompt_slot` 利用時は各スロットの `build`（既定 build を含む）から rebind を自動導出する（手書き rebind は生 seed 経路でのみ）。lib は `PromptStore` の合成内部に踏み込まない。出力は最適化済みスロットテキスト（`${var}` 保持の plain な文字列 / 名前付き mapping・永続化は利用者責任・`result.save` で opt-in 可）。`[lightning]` extra のみで完結する |
| vars | プロンプトの `${var}` 置換値。APO の最適化対象外（不変・確定）であり、各 rollout で再注入（substitute）される。最適化対象はプレースホルダを保持したテンプレート文言で、vars 値は候補生成に含めない。`prompt_slot` 利用時はヘルパが内部で再注入し、生 callable 経路では rebind / build が再注入する。候補が必要プレースホルダを失った場合は無効化 / 低評価（fail-closed） |
| プロンプトスロット（slot） | APO の最適化対象として利用者が指定するチューナブルなプロンプトテキスト（vars 未展開・`${var}` プレースホルダ保持）。粒度（agent テンプレート本文 / part / 合成結果全体など）は利用者が選ぶ。単一 seed（または `prompt_slot` の戻り値）に加え、`{名前: seed/slot}` の mapping を取りうる（複数スロット = 系全体のプロンプト同時最適化・`prompt_slot_factory` + dict comprehension で一括生成可）。vars は最適化対象に含めず rollout で再注入する。静的 str のみの単純ケースでは当該文字列を既定値にできる |
| build | 候補テキストから `AgentSpec` を構築する関数（`prompt_slot` が受ける）。省略でき、対象エージェントが registry に登録済みなら既定 build は登録 `AgentSpec` を複製して `instructions` のみ候補で差し替える（tools / handoffs / model 等は複製で保持・利用者は再宣言不要）。既定 build は `optimize` / `prompt_slot_factory` に渡る registry から対象 spec を解決し、解決不能かつ build 省略時は fail-closed エラー。利用者は `build=` を明示して動的構築もできる（併存）。既定 build でも候補テンプレートに vars を rollout 再注入する |
| rebind | 候補スロット値からエージェントを組み直す関数。`prompt_slot` を使う場合は各スロットの `build`（既定 build を含む）からフレームワークが自動導出するため利用者は渡さなくてよい（単一 / mapping いずれも省略可）。スロットが生 seed（`build` を持たない）であるパワーユーザー経路でのみ、利用者が単一候補、または候補 mapping `{名前: 候補テキスト}` を受けて registry / グラフ全体を組み直す rebind を明示する。rebind / build は rollout で vars を再注入する。rebind 内で利用者の `PromptStore` 合成を適用してよく、lib は `PromptStore` テンプレートを内省・書き換えしない |
| prompt_slot | 合成プロンプト最適化の定型を畳む使いやすさヘルパ。`PromptStore` の公開 `compose`（必要に応じ `get`）を読み取り、seed・固定部分との再合成・候補適用 rebind を内包し、`build`（候補 instructions → `AgentSpec`）を受けて slot を構成する。`build` は省略でき、既定 build は registry 登録 spec を複製して `instructions` を差し替える（registry 必要・未解決は fail-closed）。vars を seed に展開せず `${var}` プレースホルダを保持し、rollout 時に内部で vars を再注入する（利用者は `vars` を渡すだけで build 内の再注入は不要）。`build`（既定含む）を内包するため単一 / mapping いずれでもフレームワークが rebind を自動導出し、利用者は手書き rebind が不要（生 callable 経路でのみ rebind を渡す）。系全体では各エージェントに使い `{名前: slot}` の mapping として渡せる。`PromptStore` を内省・書き換えせず公開メソッドの戻り値を読むのみ。`runtime/lightning` 公開窓口に含める |
| prompt_slot_factory | `prompt_slot` の共通既定値を束ねるファクトリヘルパ。`prompt_slot_factory(store, registry, **defaults) -> make(agent, **overrides) -> Slot`。返り値 callable は `prompt_slot` の全 kwarg を素通しし、per-agent 差分（`base` / `parts` / `layout` / `tune` / `vars` / `build`）だけを上書きできる（`vars` は双方 dict のときマージ・それ以外は置換）。複数エージェントの一括生成は dict comprehension（例: `{name: make(name) for name in [...]}`）で `{名前: slot}` mapping を組み立て、`optimize(graph, slot=slots, ...)` に渡せば rebind 自動導出と合わせてグラフ全体 APO が単一呼び出しで完結する。最適化対象は mapping に掲載したエージェントのみ。`PromptStore` は公開 `compose` / `get` を読み取るのみ（非改変）・registry は読み取り複製のみ。`runtime/lightning` 公開窓口に含める |
| RL（LightningRL） | LightningRL on VERL によるモデル重みの強化学習更新。torch / vLLM / GPU を伴う重量系。系全体（ハンドオフ経由）の rollout に対し credit assignment で系全体のポリシーを最適化し、`target_agents` で学習対象を選択できる。合成済み instructions をそのまま実行コンテキストに用い、最適化対象はモデル重みでプロンプトテキスト / vars ではない。出力は checkpoint パス / OpenAI 互換エンドポイント等の参照（`model_ref`）。checkpoint 等の物理出力は Trainer / Store 設定の passthrough で lib は重みを保持しない。`[lightning-rl]` extra が必要 |
| algorithm | `optimize` の最適化系統セレクタ。`apo`（プロンプト最適化）/ `rl`（モデル更新）を取りうる |
| optimize | 最適化を実行する公開エントリ。`algorithm` 指定で APO / RL を選択し、単一エージェント / 系全体（グラフ）いずれも対象にできる。データ入口は `train`（必須）/ `val`（任意）に統一し、暗黙の分割パラメータを持たない。最短ケースは `optimize(宣言物, algorithm=..., train=..., reward=...)` の単一呼び出しで完結する。結果は既定で戻り値のみ・lib 自動書込なし。`runtime/lightning` の公開窓口に集約し、Python 組込みと衝突する `eval` 等は採用しない |
| Trainer | agent-lightning が最適化ループ本体（系全体の credit assignment・複数プロンプト最適化を含む）を回す実体。本機能は最適化ループを Trainer へ委譲する（独自実装しない） |
| Training-Agent Disaggregation | agent を回す client と訓練 / 配信を担う server を分離する Agent Lightning のアーキテクチャ。本機能は agent 側 rollout を結線し、訓練サーバインフラ自体は agent-lightning へ委譲する |
| Store | agent-lightning のストレージ抽象（InMemory / Sqlite / Mongo）。並列度 / 訓練ラウンド数 / タイムアウト / target agents 等と同じく利用者設定の passthrough として扱う。RL の checkpoint 物理出力先もこの設定の passthrough であり、lib は保管を所有しない |
| tool_mocks | rollout 副作用反復を避けるため、ツールの実行本体を副作用のない代替へ差し替える宣言（agent スコープのネスト dict）。名前・説明・引数スキーマ・承認要否は不変。llmops 実装を再利用する |
| approvals | rollout で承認待ちを自動解決する mock-approve 相当のポリシー。承認を認可するのは実際にモック差し替えされた `(agent_name, tool_name)` のみ（安全不変条件）。llmops 実装を再利用する |
| 安全不変条件（mock-approve） | mock-approve で本物の危険ツールが rollout 中に実行されないための不変条件。未差し替え / 到達不能 / 別 agent 同名 / agent 不明の承認は fail-closed エラー |
| credit assignment | RL で軌跡全体（系全体の rollout）の各ステップ / 各エージェントの LLM 呼び出しに報酬を割り当てる処理。本機能では agent-lightning（LightningRL on VERL）へ委譲し、lib は独自実装しない |
| extra | Python パッケージの optional dependency 群。本機能の軽量コアは `[lightning]`（`agentlightning`）、RL 重依存は `[lightning-rl]`（VERL / torch / vLLM）に分割する。未導入でもコアが動く前提 |
| build-don't-run | 本ライブラリの核方針。コアは独自実行 / 最適化エンジンを持たず、宣言・build-time 検証に徹し、実行は `_adapters/` 経由で SDK `Runner.run`、最適化ループは agent-lightning Trainer へ委譲する。実行寄り層は runtime に属する |
| 実行寄り層（runtime） | 公開の実行サービス API を提供する層。conversation / serve / cli / llmops / lightning が該当し、`src/oai_agentspec/runtime/` 配下に集約される |
| runtime/ | 実行寄り層を配下に集約する中間ディレクトリ。本要件の lightning は `runtime/lightning` として追加する |
| AgentSpec | `agents.Agent` の薄い宣言的 Wrapper。最適化対象となる単一エージェントの宣言。`instructions` は `PromptStore.compose` の戻り値で、静的 str / 動的 callable `(context, agent) -> str` / 別経路の `prompt`（Responses API の id 参照型）の形態を取る。既定 build は登録 `AgentSpec` を複製し `instructions` のみ差し替える |
| AgentRegistry | `AgentSpec` を登録 / 遅延構築する registry。横断 rollout / 系全体最適化で必要 spec の供給元として `optimize` / `prompt_slot_factory` に渡す。既定 build はここから対象 spec を解決して複製する。lib は読み取り・複製経由で扱い、登録 spec を変更しない |
| HandoffGraph | エージェント間のハンドオフ関係を表す宣言グラフ。横断 rollout・系全体最適化の対象 |
| WorkflowGraph | ワークフロー DSL の宣言グラフ（START / END / ノード）。end-to-end rollout・系全体最適化の対象 |
| PromptStore | 利用側 root 配下のプロンプトテンプレート（base / part / agent セグメント）を `${var}` 置換し連結して `compose()` で instructions を合成する仕組み。lib はプロンプト非同梱で、合成結果（プロンプト）の Single Source of Truth は利用者にある。本要件では変更対象外で、使いやすさヘルパ（`prompt_slot` / `prompt_slot_factory` 等）は公開 `compose` / `get` を読み取り参照するのみ。lib は `PromptStore` のテンプレートを内省・書き換えしない（`result.save` も `PromptStore` を触らない）。APO 出力の永続化は利用者責任 |
| PromptTemplate | プロンプトテキストのテンプレート単位（`${var}` プレースホルダを含みうる）。APO の最適化対象は本テンプレート本文そのものではなく、利用者がスロットとして切り出した vars 未展開の seed テキスト（`${var}` 保持）で、合成への反映は rebind（`prompt_slot` の `build`（既定 build を含む）から自動導出、または生 seed 経路の利用者供給 rebind）が担い、vars は rollout 時に再注入する |
| モデル外部流入 | LLM モデルを lib 内に保持せず外部から DI 流入させる方針。RL の更新済みモデルも checkpoint パス / エンドポイント等の参照を plain データで返し、lib に重みを保持しない |
| SDK 隔離（NFR） | `from agents` / `from openai` 等の外部 SDK import および外部クライアント（`agentlightning`）・RL 重依存（`verl` / `torch` / `vllm`）の import を `_adapters/` 配下のみに閉じる本体の不変条件 |
| fail-closed | 不確実・欠落・失敗時に安全側（明確なエラー / fail / skip）へ倒す方針 |
| graceful degradation | 外部依存（extra / LLM API / 設定）不在や Trainer 例外時にクラッシュせず明確なエラー / skip で機能縮退する挙動 |
| EARS | Easy Approach to Requirements Syntax。WHEN / IF / THEN による受け入れ基準記述形式 |
| _adapters | 外部 SDK / 外部クライアント（`agentlightning` / RL 重依存）への import 単一窓口（SDK / 外部クライアント隔離を担うサブパッケージ） |

## 7. 利用イメージ（暫定 API・確定は設計フェーズ）

以下のフィールド名・シグネチャは暫定であり、確定は設計フェーズで行う。データ入口は `train`（必須）/ `val`（任意）に統一し、`optimize` 内に暗黙分割を持たない。結果は既定で戻り値のみ・lib 自動書込なしで、永続化は `result.save(path)`（opt-in）または利用者自前の書き出しに限る。いずれの例も、lib は `PromptStore` を変更せず公開 `compose` / `get` を読み取るのみである。

APO（静的 instructions・最短・既製 reward・val 省略）:

```python
from oai_agentspec.runtime.lightning import optimize, contains

# spec.instructions は静的 str。スロット既定値は当該文字列、rebind は既定 instructions 差し替え。
# val 省略は warning（train で選定・自動分割はしない）。
result = optimize(spec, algorithm="apo", train=data, reward=contains("expected"))
# result.prompt は plain な最適化済みテキスト。result.val_score は None。既定では何も書かれない。
# 注記: vars を保護したいなら（vars 展開済みの静的 str でなく）${var} プレースホルダ + prompt_slot を使う。
```

APO（単一エージェント・`PromptStore` 合成・`prompt_slot`・build 省略で登録 spec 複製・vars 不変・opt-in 保存）:

```python
from pathlib import Path
from oai_agentspec.runtime.lightning import optimize, prompt_slot, train_val_split, contains

train, val = train_val_split(data, val_ratio=0.2, seed=0)

# build を省略すると、registry 登録済みの "triage" spec を複製し instructions のみ候補で差し替える
# （tools / handoffs / model は登録 spec から複製・再宣言不要）。registry は optimize に渡すものを使う。
# vars は最適化対象外（不変）・rollout 再注入。出力は ${var} プレースホルダを保持。
slot = prompt_slot(
    store,
    registry,                 # 既定 build の spec 解決元（build= 省略時に使用）
    tune="triage",            # チューニング対象セグメント（${var} を展開せず保持）
    base="main",
    parts=["style"],
    vars=VARS,                # ${var} 置換値（最適化対象外・rollout 再注入）
)
# 第1引数は最適化対象の宣言物（spec は triage の AgentSpec）。slot は slot= キーワードで渡す。
result = optimize(spec, algorithm="apo", registry=registry, slot=slot, train=train, val=val, reward=contains("expected"))

# result.prompt は ${var} プレースホルダを保持したテンプレート。永続化は利用者の opt-in。
result.save("prompts/agents/triage.md")
# パワーユーザー注記: 動的構築が必要なら prompt_slot(..., build=lambda text: AgentSpec(...)) を明示できる（併存）。
```

APO（系全体・`prompt_slot_factory` + dict comprehension でグラフ全体のスロットを一括生成・手書き build / rebind 不要・vars 不変）:

```python
from oai_agentspec.runtime.lightning import optimize, prompt_slot_factory, train_val_split, contains

train, val = train_val_split(data, val_ratio=0.2, seed=0)

# ファクトリが共通既定値を束ね、dict comprehension で列挙エージェント分の slot を一括生成
# （各 slot は登録 spec 複製の既定 build・${var} 保持）。
# build から rebind が自動導出されるため、手書き build / rebind は不要。
make_slot = prompt_slot_factory(
    store, registry,
    base="main", parts=["style"], vars=VARS,
)
slots = {name: make_slot(name) for name in ["triage", "billing"]}

result = optimize(handoff_graph, algorithm="apo", registry=registry, slot=slots, train=train, val=val, reward=contains("expected"))
# 各エージェントの最適化プロンプトを名前付きで取り出す（result.prompt は {名前: 最適化テキスト} の mapping・${var} 保持）。
triage_prompt = result.prompt["triage"]
billing_prompt = result.prompt["billing"]
# 最適化対象は slots に入れたエージェント（triage / billing）のみ。未掲載のエージェントのプロンプトは固定。
# パワーユーザー注記: 個別生成 / registry 外の動的構築は prompt_slot(..., build=...) を手書きで併存できる。
```

自前分割（スライス・層化・時系列など）を `train` / `val` として渡す:

```python
from oai_agentspec.runtime.lightning import optimize, contains

train, val = data[:80], data[80:]   # 利用者が自前で分割（時系列・層化等も同様）
# 第1引数は最適化対象（spec）。slot は slot= キーワードで渡す。
result = optimize(spec, algorithm="apo", registry=registry, slot=slot, train=train, val=val, reward=contains("expected"))
```

RL（単一・既製 reward + tool_mocks による安全化・メタデータの opt-in 保存）:

```python
from oai_agentspec.runtime.lightning import optimize, train_val_split, judge

train, val = train_val_split(data, val_ratio=0.2, seed=0)
result = optimize(
    spec,
    algorithm="rl",
    train=train,
    val=val,
    reward=judge(rubric, model=azure_model),   # rubric / model は利用者供給
    tool_mocks={"triage": {"issue_refund": lambda **kw: {"ok": True}}},  # 危険ツールを副作用なし代替へ
)
# result.model_ref は checkpoint パス / OpenAI 互換エンドポイント等の参照（plain データ）。lib は重みを保持しない。
result.save("runs/summary.json")   # メタデータ/参照のみ保存（重みは書かない・PromptStore は触らない）。
# result.val_score は val 上の汎化スコア。
```

RL（系全体 end-to-end + 学習対象エージェント選択）:

```python
from oai_agentspec.runtime.lightning import optimize, train_val_split, judge

train, val = train_val_split(data, val_ratio=0.2, seed=0)

# 系全体（ハンドオフ経由）を rollout し、credit assignment で系全体のポリシーを最適化。
# target_agents で学習対象を選択（既定は全エージェント・対象外は凍結）。
result = optimize(
    handoff_graph,
    algorithm="rl",
    registry=registry,
    target_agents=["triage", "billing"],
    train=train,
    val=val,
    reward=judge(rubric, model=azure_model),
)
# result.model_ref は target agents ごとに識別可能な参照（plain データ）。lib は重みを保持しない。
```

横断（registry 同伴・WorkflowGraph）:

```python
from oai_agentspec.runtime.lightning import optimize, contains

result = optimize(
    workflow_graph,
    algorithm="apo",
    registry=registry,        # 必要 spec を register 済みの AgentRegistry を同伴（FR-1・既定 build の spec 解決元）
    slot=slots,               # prompt_slot_factory / prompt_slot で生成した Slot（build から rebind 自動導出・vars 再注入）、単一 str、または {名前: slot}
    train=train,
    val=val,
    reward=contains("expected"),
)
# lib は registry / グラフ / PromptStore を変更せず、出力は plain データ。永続化は利用者の opt-in。
```

生 callable 経路（パワーユーザー・ヘルパ非経由）も併存する。スロットを生 seed（`build` を持たない文字列等）で渡す場合は `rebind`（単一候補 / 候補 mapping）を明示し、rebind / build が vars を再注入する。`prompt_slot(..., build=...)` で動的構築する経路も併存する。`slot`（単一 / mapping）/ `rebind` / `reward` を手書き callable として直接 `optimize` に渡せ、`train` / `val` も利用者が任意に構成して渡せる（FR-9）。
