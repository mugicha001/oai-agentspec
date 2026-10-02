# 0046: usage 欠損はトークン 3 フィールドがすべて 0 で判定し、requests を見ない

- Status: accepted
- Date: 2026-09-30

## Context

### 決定時点の判定

- lib は SDK の `Usage` を 2 箇所で「欠損（未取得）」かどうか判定する。
  - resilience の予算フック（`src/oai_agentspec/_adapters/resilience.py` の `_BudgetHooks`）: 1 応答の usage が
    欠損なら warning を出す（トークン上限をどちらも設定しない場合は判定しない）。
  - intent の予測エージェント（`src/oai_agentspec/_adapters/intent.py` の `_collect_usage`）: 全応答の usage が
    欠損なら `input_tokens` / `output_tokens` を None にする。
- どちらも「`requests == 0` かつ `total_tokens == 0`」を欠損とする。intent 側のこの判定規則は、要件書
  `docs/requirements/executable-intent-declaration.md` の `ParamUsage` の受け入れ基準
  （「全応答の usage が `requests == 0` かつ `total_tokens == 0` なら usage 未取得とみなす」）が定めている。
- SDK の `Usage` は非 Optional で既定値が 0 のため、0 と未取得を型で区別できない。欠損を示すフラグも無い。

### requests に依存した判定が働かない経路

- SDK 0.22.3 の組み込みモデル（`OpenAIChatCompletionsModel` / `OpenAIResponsesModel`）は、usage が欠けた応答でも
  requests を 1 と数える。このため決定時点の判定では欠損と判定されない（ADR-0043 の「要修正」の usage 欠損の
  検知の項の実測）。
- SDK 0.17.4 のストリーム経路も、終端イベントの usage がトークン 0 で存在すれば requests を 1 と数える（SDK
  ソースを読んだ結果）。
- 1 回 retry した後は requests が 1 以上になり、欠損が検知されない（両版に共通する既存の穴。ADR-0043）。

### 検討した判定条件

| 判定条件 | 判定と理由 |
|---|---|
| `requests == 0` かつ `total_tokens == 0`（決定時点の判定） | 却下。requests の数え方はモデル実装と SDK の版に依存し、上の経路で働かない |
| `total_tokens == 0` のみ | 却下。`total_tokens` を埋めずに `input_tokens` / `output_tokens` だけを返すモデル実装を欠損と誤判定する（推論） |
| **`input_tokens == 0` かつ `output_tokens == 0` かつ `total_tokens == 0`** | **採用**（Decision） |
| モデル実装ごとに欠損の表現を判別する | 却下。lib も SDK 本体も提供しない拡張のモデル実装（LiteLLM / AnyLLM 等）まで追随が要り、SDK の内部表現に結合する |

### 要件書の判定規則との関係と先例

採用する条件は要件書の判定規則を置き換える。要件書を supersede した先例は ADR-0005（要件書 D-2 の方式を
ユーザー指示により supersede した）である。先例は未実装・未コミット段階の supersede で互換影響が無かった。
本件は実装済みの判定を変えるため互換影響がある（Consequences）。本件の supersede はユーザーが承認した設計
方針に基づく。

## Decision

### 1. 判定条件

usage が欠損であるとは、`input_tokens == 0` かつ `output_tokens == 0` かつ `total_tokens == 0` であることをいう。
`requests` は判定に使わない。

- requests を見ないのは、requests の数え方がモデル実装・SDK の版・経路（ストリーム / 非ストリーム・retry の
  有無）で変わり、トークンの有無と対応しないためである。
- 実 API の応答でトークンがすべて 0 になることは無いという前提に立つ（推論。未実測）。
- total だけでなく in / out も見るのは、total を埋めないモデル実装で誤判定しないためである。

### 2. 共通の述語を 1 つ置く

- 述語 `usage_is_missing(usage) -> bool` を `_adapters/runner.py` に置き、resilience と intent の両方から使う。
- 置き場の理由: `_adapters/runner.py` の `_adapters` 内の依存は `run_context` のみで、`intent.py` と
  `resilience.py` のどちらから import しても循環しない。intent から resilience を import する形（機能間の結合）を
  避ける。新規モジュールは作らない。
- import の作法は各モジュールの既存の作法に合わせる。`intent.py` は `_adapters` 内の依存も関数内で遅延 import
  しているので、`_collect_usage` の関数内で import する。`resilience.py` はモジュール先頭で import する。

### 3. 各判定箇所

- resilience: 1 応答の `response.usage` に述語を適用する。トークン上限をどちらも設定しない場合は判定しない
  こと、logger 名（`constants.py` の定数）は変えない。warning の文言と docstring は新しい条件に合わせる。
- intent: いずれかの応答で述語が偽なら取得済み、全応答で真なら `input_tokens` / `output_tokens` を None にする。
  `model_calls` は応答件数のまま。型と関数の docstring は新しい条件に合わせる。

### 4. 要件書の判定規則を supersede する

`docs/requirements/executable-intent-declaration.md` の `ParamUsage` の受け入れ基準のうち、usage 未取得の判定
規則（「全応答の usage が `requests == 0` かつ `total_tokens == 0`」）を、本 ADR の判定条件（全応答で
`input_tokens` / `output_tokens` / `total_tokens` がすべて 0）へ置き換える。要件書は Issue の原本として編集しない。同じ受け入れ基準の他の部分
（未取得なら None を返す・`model_calls` は応答件数のまま・0 と未取得の区別を判定規則で行う）は変えない。

### 5. 周辺の記述

- 応答ビルダの `requests` の既定値 1 は変えない（公開シグネチャ）。docstring 上の理由は「SDK の累積 usage の
  requests と整合させるため」とする（欠損判定は requests を見ない）。
- usage を指定するビルダの docstring は「usage を指定する場合はトークン数を 1 以上にする（トークンがすべて 0 の
  応答は欠損として扱われる）」の趣旨とする。
- `text_response` / `tool_call_response` の `Usage()`（すべて 0）は欠損として扱われる。決定論モデルと
  `RunBudgetPolicy` を併用すると warning が出る。

## Consequences

- + SDK 組み込みモデルが usage の欠けた応答を返した場合も欠損を検知する。resilience は warning を出し、intent は
  トークン値を未取得（None）として返す。
- + ストリーム経路と非ストリーム経路が同じ判定になる。retry 後に requests が 1 以上になっても、トークンが
  すべて 0 なら欠損と判定する（テストで固定する）。
- + モデル実装ごとの requests の数え方に依存しない。拡張のモデル実装（LiteLLM / AnyLLM 等）もトークンの表現が
  同じなら同じ判定になる見込みである（未実測）。
- - 互換影響: requests が 1 以上でトークンがすべて 0 の usage は、要件書の判定規則の下では取得済み、本 ADR の
  判定の下では欠損になる。該当する入力:
  - `text_response_with_usage(text, total_tokens=0)`（`requests` の既定値は 1）
  - requests だけを数え、トークンを 0 のまま返す自作 Model（例: `Usage(requests=1)` を返すもの）
- - 上の入力での観測の差: intent の `input_tokens` / `output_tokens` は 0 ではなく None になる。resilience
  （トークン上限を設定した場合）は warning が出る。
- - 実 API がトークンのすべて 0 の応答を返す場合があれば、その応答も欠損として扱われる（warning と None。
  run 自体は失敗しない）。

## Confirmation

本 ADR は設計フェーズで受理したものであり、以下の強制手段は実装フェーズで追加する対象である。追加した
テストは `docs/QUALITY-GUARANTEES.md` へ登録する（source = ADR-0046）。個別 assert とテスト名の確定はテスト
実装時に行い、一次情報は各テストの docstring とする。

- 非ストリームで、Model が requests を数えトークンがすべて 0 の usage を返すと、resilience の warning が 1 件出て
  intent のトークン値が None になることを pin するテスト
- ストリームで、終端イベントにトークンがすべて 0 の usage を載せると warning が 1 件出ることを pin するテスト
- ストリームで、終端イベントの usage が無い場合も warning が 1 件出ることを pin する回帰テスト
- SDK 組み込みの ChatCompletions / Responses モデルに、usage を欠いた応答を返すスタブのクライアント（ネット
  ワーク・課金なし）を渡し、warning が 1 件出て intent のトークン値が None になることを pin するテスト。
  スタブのクライアントを受理するか、HTTP 層のモックトランスポートが要るかは実装時に確認する
- in / out を持ち total を持たない usage を欠損と判定しないことを pin するテスト
