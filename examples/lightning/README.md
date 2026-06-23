# Agent Lightning 最適化（oai-agentspec[lightning]）の使い方

利用者が宣言したエージェント（単一 `AgentSpec`、またはハンドオフを含む系全体の `HandoffGraph` /
`WorkflowGraph`）のプロンプトを、**実行時の振る舞いと利用者供給の報酬**に基づいて自動改善（APO:
Automatic Prompt Optimization）する。最適化ループ本体は Agent Lightning の Trainer へ委譲し、本
ライブラリは宣言物・rollout・報酬の薄い結線に徹する（build-don't-run）。最適化対象の宣言物・
registry は read-only で扱い変更しない。プロンプト・評価ケース・報酬は lib に同梱せず、すべて
利用者が渡す。

## 目次

1. [インストール](#インストール)
2. [30 秒で動く最小例](#30-秒で動く最小例)
3. [使い方の流れ（4 ステップ）](#使い方の流れ4-ステップ)
4. [ユースケース別ガイド（どの example を見るか）](#ユースケース別ガイドどの-example-を見るか)
5. [リファレンス](#リファレンス)
   - [`optimize` 引数](#optimize-引数)
   - [APO 設定（直接 kwargs / `OptimizeConfig`）](#apo-設定直接-kwargs--optimizeconfig)
   - [`OptimizeCase`（推奨データ型）](#optimizecase推奨データ型)
   - [報酬ファクトリ](#報酬ファクトリ)
   - [スロットヘルパ（`prompt_slot` / `prompt_slots`）](#スロットヘルパprompt_slot--prompt_slots)
   - [置換変数（`${var}`）の扱い](#置換変数varの扱い)
   - [データ分割（`train_val_split`）](#データ分割train_val_split)
   - [結果（`OptimizeResult`）と保存](#結果optimizeresultと保存)
6. [HITL・安全性（rollout 副作用の反復）](#hitl安全性rollout-副作用の反復)
7. [失敗種別（`OptimizeError` / `FailureKind`）](#失敗種別optimizeerror--failurekind)
8. [AsyncOpenAI クライアントの作り方（Azure / OpenAI 直接）](#asyncopenai-クライアントの作り方azure--openai-直接)
9. [仕組み（裏側）](#仕組み裏側)
10. [スコープ外](#スコープ外)

---

## インストール

```bash
pip install 'oai-agentspec[lightning]'      # APO（agentlightning[apo]）
```

RL（モデル重み更新）は重依存のため別 extra `oai-agentspec[lightning-rl]`（後続）で提供する。
`optimize(algorithm="rl", ...)` は本 extra では明確なエラーで案内する。

## 30 秒で動く最小例

```python
import asyncio
from oai_agentspec import AgentSpec
from oai_agentspec.runtime.lightning import OptimizeCase, contains, optimize, train_val_split

async def main() -> None:
    target = AgentSpec(name="router", instructions="依頼を1語で分類する。", model=my_model)
    data = [
        OptimizeCase(input="請求の件",   expected_output="billing"),
        OptimizeCase(input="アプリが落ちる", expected_output="support"),
        OptimizeCase(input="営業時間は？",  expected_output="other"),
        OptimizeCase(input="二重請求された", expected_output="billing"),
    ]
    train, val = train_val_split(data, val_ratio=0.25, seed=0)

    result = await optimize(
        target,                       # algorithm は省略可（既定 "apo"）
        train=train, val=val,         # val は必須（APO の beam search 用）
        reward=contains(),            # 既定 field=expected_output を読む
        apo_client=my_apo_client,     # APO 計算用クライアント（直接渡し・必須）
        rounds=1, apo_beam_width=1, apo_branch_factor=1,  # 最小設定（後述）
    )
    print(result.train_score, result.val_score)
    print(result.prompt)              # ${var} 保持の最適化済みテキスト

asyncio.run(main())
```

`my_model` は SDK のモデル、`my_apo_client` は AsyncOpenAI 互換クライアント
（→ [作り方](#asyncopenai-クライアントの作り方azure--openai-直接)）。

> **最小設定 vs 本番設定**: 上記 `rounds=1 / apo_beam_width=1 / apo_branch_factor=1` は **E2E 動作
> 確認用の最小コスト**（1 ラウンドで 1 候補だけ生成 → 評価）。最適化の効果を実際に得たいときは、
> ラウンドを増やし beam を広げる（例: `rounds=3, apo_beam_width=4, apo_branch_factor=4` で 1 ラウンド
> あたり最大 16 候補から最良を選ぶ）。`examples/lightning/*.py` は前者の最小設定で動くようにして
> あるため、利用者は `python examples/lightning/01_single_agent_apo.py` 程度ですぐ E2E が回る。

---

## 使い方の流れ（4 ステップ）

1. **最適化対象を宣言** — `AgentSpec`（単体）/ `HandoffGraph`（系全体）/ `WorkflowGraph`。registry は
   `apply` 済みのものを `optimize(..., registry=...)` に渡せばよい（lib が clone して扱う）。
2. **データセットを `OptimizeCase` で書く** — `input` 必須 + 期待観点（`expected_output` /
   `expected_tools` / `expected_route` / `expected_last_agent` / `expected_approvals`）を
   採点したい観点だけ埋める。
3. **reward を選ぶ** — 単観点なら `contains()` / `tool_match()` 等のファクトリを引数なしで呼ぶ
   （`OptimizeCase` の標準フィールドを既定 `field` で読む）。複合は手書きで AND / 重み付き平均。
4. **`optimize` を呼ぶ** — 第 1 引数に target、`train` / `val` / `reward` / `apo_client` を渡すだけ。
   `${var}` 保持の最適化済みテキストが `result.prompt` に返る。

> **後方互換**: dict ケース（`{"input": ..., "expected": ...}`）も併存し、その場合は reward 側で
> `contains("expected")` のように自由フィールド名を明示する。

---

## ユースケース別ガイド（どの example を見るか）

| あなたの状況 | 見るべき example | 主な道具 |
|---|---|---|
| まずは単一 agent の prompt を最適化したい | `01_single_agent_apo.py` | `OptimizeCase` + `contains()` + `apo_client` |
| `PromptStore` 合成プロンプトを最適化したい（既定 build / `${var}` 保持） | `02_prompt_slot_apo.py` | `prompt_slot` |
| ハンドオフ含む系全体（`HandoffGraph`）を最適化したい | `03_graph_apo.py` | `prompt_slots` |
| 期待ツールを ground truth にしつつ承認必須の危険ツールを安全に回したい | `04_reward_and_safety.py` | `tool_match()` + `tool_mocks` + `approvals` |
| 危険操作を必ず承認ゲートへ回すプロンプトを学習させたい（HITL） | `06_approval_match_apo.py` | `approval_match()` |
| 出力 / ツール / 経路 / 最終 agent を 1 ケースに集約して複合 reward で学習 | `07_composite_reward_apo.py` | `OptimizeCase` 全観点 + 複合 reward |
| 失敗種別の判別だけ確認したい（オフライン・LLM 不要） | `05_failure_handling.py` | `OptimizeError` / `FailureKind` |

実行（Azure OpenAI の環境変数が必要。`examples/_shared/_azure.py` 参照）:

```bash
uv run python examples/lightning/01_single_agent_apo.py
uv run python examples/lightning/05_failure_handling.py   # これは環境変数なしで動く
```

---

## リファレンス

### `optimize` 引数

第 1 引数は常に最適化対象（`AgentSpec` / `HandoffGraph` / `WorkflowGraph`）。スロットは `slot=`
キーワードで渡す。

| 引数 | 役割 |
|---|---|
| `target`（位置） | 最適化対象の宣言物 |
| `algorithm` | 省略可（既定 `"apo"`）。`"rl"` は別 extra `[lightning-rl]` で明確なエラーで案内 |
| `train` | 最適化 / rollout に使う入力ケース群（**必須**・`OptimizeCase` / dict / 任意型） |
| `val` | 最良候補の選定 / 汎化スコアに使う入力ケース群（**必須**・APO の beam search に必要） |
| `reward` | rollout の `RolloutResult` から報酬を返す callable（同期 / async・ファクトリ可） |
| `slot` | 最適化対象スロット（`Slot` / 生 seed str / `{名前: Slot}` mapping）。省略で静的 `AgentSpec` の `instructions` を既定スロットにする |
| `rebind` | 生 seed 経路で候補から宣言物を組み直す関数（`prompt_slot` / `prompt_slots` 利用時は `build` から自動導出のため不要） |
| `registry` | 横断対象 / 既定 build の specs 供給経路（`HandoffGraph` 必須） |
| `tool_mocks` | rollout 副作用を安全化する agent スコープのモック dict |
| `approvals` | 承認自動解決ポリシー（`tool_mocks` と併用） |
| `apo_client` 等 | APO 計算用クライアント等の直接渡し kwargs（**最小ケース推奨**・[後述](#apo-設定直接-kwargs--optimizeconfig)） |
| `config` | `OptimizeConfig`（パワーユーザー経路・直接 kwargs と同時指定はエラー） |

### APO 設定（直接 kwargs / `OptimizeConfig`）

最小ケースは `apo_client=` 直接渡しで足りる。`rounds` / `concurrency` 等も同列で渡せる。
パワーユーザーは `OptimizeConfig` を組み立てて `config=` で渡す（**両方同時指定は
`CONFIG_MISSING`**）。

| 項目 | 直接 kwargs 名 | `OptimizeConfig` 同名 | 役割 |
|---|---|---|---|
| APO 計算用クライアント | `apo_client` | `apo_client` | `AsyncOpenAI` 互換（**必須**） |
| 訓練ラウンド数 | `rounds` | `rounds` | APO の `beam_rounds` |
| 並列度 | `concurrency` | `concurrency` | `Trainer.n_runners` |
| タイムアウト | `timeout_seconds` | `timeout_seconds` | 1 batch の `rollout_batch_timeout` 秒 |
| Store 抽象 | `store` | `store` | InMemory / Sqlite / Mongo（不透明値・passthrough） |
| gradient モデル | `apo_gradient_model` | `apo_gradient_model` | 既定 `gpt-5.4-mini` |
| edit 適用モデル | `apo_apply_edit_model` | `apo_apply_edit_model` | 既定 `gpt-5.4-mini` |
| beam 幅 / 分岐数 | `apo_beam_width` / `apo_branch_factor` | 同名 | APO beam search |

```python
# 推奨（最小ケース）
result = await optimize(target, train=train, val=val, reward=contains(),
                        apo_client=my_apo_client, rounds=2)

# パワーユーザー（OptimizeConfig 経由）
from oai_agentspec.runtime.lightning import OptimizeConfig
result = await optimize(target, train=train, val=val, reward=contains(),
                        config=OptimizeConfig(apo_client=my_apo_client, rounds=2,
                                              apo_beam_width=4, store=my_store))
```

#### コスト最小化（E2E 動作確認向け）

APO は 1 ラウンドあたり `apo_beam_width × apo_branch_factor` 個の候補プロンプトを生成 / 評価する
ため、既定（beam=4 / branch=4）だと 1 example あたり **30+ 件の LLM 呼び出し**になる。手元での
動作確認を最短で済ませたいときは:

```python
result = await optimize(
    target, train=train, val=val, reward=contains(),
    apo_client=my_apo_client,
    rounds=1,              # ラウンド数（候補生成 → 評価のサイクル数）
    apo_beam_width=1,      # 親候補の保持数（1 = 直前ラウンドの最良 1 つだけ）
    apo_branch_factor=1,   # 親候補からの分岐数（1 = 1 ラウンドあたり 1 候補だけ生成）
)
```

この設定で「seed → 1 候補生成 → 評価 → 採用 / 棄却」の最小フローを 1 巡だけ回す（実 LLM 呼び出しは
gradient + apply_edit + train rollout × 件数 + val rollout × 件数 程度）。本番運用では `rounds` を
増やし beam を広げて多候補比較を有効化する。`examples/lightning/*.py` はすべてこの最小設定で動く。

#### Trace 計測と AgentOps

APO は rollout で発生する OpenAI 呼び出しの `gen_ai.*` span を読んで textual gradient を計算する
ため、内部で agent-lightning の `AgentOpsTracer(agentops_managed=True, instrument_managed=True)`
（agent-lightning 既定）を使う。利用者は通常**何も指定しなくてよい**（既定で動く）。AgentOps SDK
への依存は agent-lightning 0.3 のトレーサ構造上 unavoidable（`OtelTracer` / `DummyTracer` は OpenAI
計測を持たないため APO 互換性なし）。

##### `AGENTOPS_API_KEY` 未設定時の挙動（既定）

利用者が `AGENTOPS_API_KEY` を設定していない場合（oai-agentspec の標準ケース）:

- agent-lightning の `agentops_managed=True` 経路が `os.environ.setdefault("AGENTOPS_API_KEY", "dummy")` を行い、AgentOps SDK は dummy キーで初期化される
- **クラウドアップロードは silent fail**（dummy キーは認証で弾かれる・実際の送信は起きない）
- 本ライブラリは `_adapters/lightning._build_trainer` で `AGENTOPS_API_KEY` 未設定を検知すると、`AGENTOPS_LOG_LEVEL=ERROR` と `AGENTOPS_LOGGING_TO_FILE=False` を `setdefault` し、agentops の console 出力（`[OPENAI INSTRUMENTOR] Error ...` warning や `Session Replay https://...` URL）と `agentops.log` ファイル生成を抑制する
- 利用者が `AGENTOPS_LOG_LEVEL=DEBUG` 等を明示していれば、`setdefault` は上書きしないので unmute できる

##### AgentOps クラウドを使いたい場合

環境変数 `AGENTOPS_API_KEY` を本物の値に設定するだけ。本ライブラリは `AGENTOPS_API_KEY` が設定済みのときは agentops 関連の env を一切触らず、agentops の通常動作（INFO レベルログ・`agentops.log` ファイル生成・cloud upload）に任せる。

```bash
export AGENTOPS_API_KEY=ao-xxxxxxxxxxxxx
uv run python examples/lightning/01_single_agent_apo.py
# Session Replay URL を辿れば実際にクラウドへ上がっている
```

##### 上級者向け escape hatch

agent-lightning の `Tracer` 派生インスタンスを直接渡したい場合は `OptimizeConfig.tracer=` / `optimize(..., tracer=)` も使える（既定 tracer を捨てて明示値を使う）。`OtelTracer` / `DummyTracer` は OpenAI 計測を持たないため APO 互換性なし（gradient 計算用 span が空になる）。本 README ではサンプルを載せない。

### `OptimizeCase`（推奨データ型）

llmops `EvalCase` 相当の typed なケース型。標準フィールド名が reward ファクトリの既定 `field` と
一致するため、**reward は引数なしで呼べる**。

```python
OptimizeCase(
    input="...",                   # 必須
    id="case-1",                   # 任意・ログ / 失敗解析用
    expected_output="...",         # contains() / exact() の既定
    expected_tools=["..."],        # tool_match() の既定
    expected_route=["a", "b"],     # route_match() の既定（起点を含む agent 名フルパス）
    expected_last_agent="b",       # last_agent_match() の既定
    expected_approvals=["..."],    # approval_match() の既定
    metadata={...},                # 採点に使わない補助情報（reward は参照しない）
)
```

dict ケース（`[{"input": ..., "expected": ...}, ...]`）も併存し、自由なフィールド名で
`contains("expected")` のように明示すれば従来どおり動く（後方互換）。

### 報酬ファクトリ

dataset のフィールド名 / rubric を受けて `reward` callable を生成する。lib に報酬・データ・
プロンプトを同梱しない。手書き `(RolloutResult) -> float` もそのまま渡せる。

各ファクトリの `field` 引数は省略可能で、既定値は `OptimizeCase` の標準フィールド名に揃う。

| ファクトリ | 報酬 1.0 の条件 | 既定 `field` | 必要データ |
|---|---|---|---|
| `contains(field=...)` | 出力に `case[field]` が含まれる | `"expected_output"` | 期待文字列 |
| `exact(field=...)` | 出力が `case[field]` と完全一致 | `"expected_output"` | 期待文字列 |
| `tool_match(field=...)` | 期待ツールが全て呼ばれている（recall） | `"expected_tools"` | 期待ツール名の列 |
| `approval_match(field=...)` | 期待承認ゲートが全て発火（recall） | `"expected_approvals"` | 期待承認ツール名の列 |
| `route_match(field=...)` | 期待経路と観測経路が完全一致（順序・経由回数含む） | `"expected_route"` | 期待経路（起点込み agent 名の列） |
| `last_agent_match(field=...)` | 期待最終 agent と観測 `last_agent` が一致 | `"expected_last_agent"` | 期待最終 agent 名 |
| `judge(rubric, model)` | 利用者 Judge モデルが 0.0..1.0 で採点 | (rubric は宣言時に渡す) | - |

`RolloutResult` は `case` / `output` / `tool_calls` / `fired_approvals` / `route_steps` /
`last_agent` を持つ plain な観測。`fired_approvals` は中断時に発火した承認ゲート、`route_steps`
は実行経路（起点を含む agent 名の列・llmops `HandoffRoute` と同型）、`last_agent` は最終応答した
agent 名（中断時は None）。

**複合 reward の例**（`07_composite_reward_apo.py`）:

```python
def composite_reward(r: RolloutResult) -> float:
    return (contains()(r)
            + tool_match()(r)
            + route_match()(r)
            + last_agent_match()(r)) / 4
```

### スロットヘルパ（`prompt_slot` / `prompt_slots`）

合成プロンプトの seed 取得・固定部分との再合成・候補適用 rebind・build 宣言を畳む。`PromptStore`
の公開メソッドを**読み取るのみ**で一切改変しない。

- `prompt_slot(store, registry, *, tune, base=None, parts=(), vars=None, build=None)` -> `Slot`
  - seed は `PromptLayout` を尊重して解決する: `store.compose(agent=tune, vars=None)`（標準の
    `agents/<tune>.md`）を優先し、見つからなければ `store.get(tune).body`（root 直下の flat 配置）に
    フォールバックする。`${var}` プレースホルダは保持されたまま seed として保存される。
  - `build` 省略時の既定 build は registry 登録 `AgentSpec` を複製し `instructions` だけ候補で
    差し替える（registry 必須・未解決は fail-closed）。tools / handoffs / model 等は登録 spec から
    複製され、利用者は再宣言不要。
  - `vars` は最適化対象外で seed に保持し、rollout 時に内部で再注入する。
- `prompt_slots(store, registry, agents=[...], *, base=None, parts=(), vars=None)` -> `{名前: Slot}`
  - 列挙したエージェント分を一括生成。`optimize(graph, slot=slots, registry=registry)` に渡すと
    rebind 自動導出と合わせてグラフ全体 APO が実質 2 行で書ける。**未掲載のエージェントは固定**。

`Slot`（`prompt_slot` / `prompt_slots` の戻り値）は `build` を内包するため、フレームワークが各
スロットの `build` から rebind を自動導出する（手書き rebind は生 seed 経路のときだけ必要）。

### 置換変数（`${var}`）の扱い

`${var}` は最適化対象外（不変）。`Slot.seed` には未展開のまま保持され、APO 内部の候補生成 / 評価
（textual gradient + beam search）も `${var}` 保持のまま扱う。rollout 時に `vars` が再注入されて
agent の `instructions` になる。`OptimizeResult.prompt` / `seed` は**この rollout 実体と一致する
合成済み full テキスト**を返すため、`vars` を指定した場合は **vars 展開済み**で返る
（`vars` を渡さなければ `${var}` のまま）。

**候補がプレースホルダを喪失した場合は当該候補を seed にフォールバックし** `RuntimeWarning` で利用者
へ通知する（fail-closed・公開契約「最適化済みテキストは `${var}` を保持する」を守る）。利用者は
`history[i]["placeholder_fallback"]` フラグで warning 受信に依存せず programmatic に検出できる
（→ [`OptimizeResult.history`](#結果optimizeresultと保存)）。

### データ分割（`train_val_split`）

```python
train, val = train_val_split(data, val_ratio=0.2, seed=0, shuffle=True)
```

`seed` 固定で決定的に分割する純データ操作（SDK / `PromptStore` に触れない）。`OptimizeCase` /
dict / 任意型のいずれも分割できる。自前分割（スライス / 層化 / 時系列）の結果も同じく `train` /
`val` として渡せる。**`optimize` は内部で自動分割しない**（利用者が明示的に分けて渡す契約）。

### 結果（`OptimizeResult`）と保存

| 属性 / メソッド | 内容 |
|---|---|
| `result.prompt` | 最適化済みプロンプト（**rollout 時に agent が見る合成済み full** ・単一は str・複数は `{名前: str}` mapping・`vars` 指定時は展開済み・未指定時は `${var}` のまま） |
| `result.seed` | 最適化前のプロンプト（`prompt` と同じ shape の合成済み full・vars 展開ルールも同一） |
| `result.diff` | `seed` と `prompt` の **unified diff** 表記（`prompt` と同じ shape・差分なしは空文字）。複数パーツ合成中の変更箇所が一目で分かる |
| `result.train_score` / `result.val_score` | train / val 平均スコア（`val` 省略時は `val_score=None`） |
| `result.history` | 各スロット 1 件の `HistoryEntry` の列（`slot` / `best_score` / `best_version` / `placeholder_fallback` の 4 キー・[後述](#historyentry-schema)） |
| `result.to_dict()` | 結果を plain dict で取得（ログ / 外部保存用・history entry は deep copy されるため戻り値の書き換えは `result.history` に伝播しない） |
| `result.save(path)` | 利用者指定パスへ書き出す（**opt-in**・`save` を呼ばない限り何も書かない） |

#### `HistoryEntry` schema

`OptimizeResult.history` は各スロット 1 件の TypedDict の列。

| キー | 型 | 内容 |
|---|---|---|
| `slot` | `str` | 当該ラウンドで最適化したスロット名 |
| `best_score` | `float \| None` | APO 内部の最良スコア（`placeholder_fallback=True` のときは None） |
| `best_version` | `int \| None` | APO 内部の最良 version（`placeholder_fallback=True` のときは None） |
| `placeholder_fallback` | `bool` | APO 最良候補が seed の `${var}` を喪失し seed にフォールバックしたら True |

```python
result = await optimize(...)
for entry in result.history:
    if entry["placeholder_fallback"]:
        # 当該スロットは APO の最良候補が `${var}` を喪失したため seed にフォールバック。
        # rounds / beam を増やすか reward 設計で placeholder 保持を強化するシグナル。
        print(f"slot {entry['slot']!r}: APO fell back to seed (placeholder lost)")
    else:
        print(f"slot {entry['slot']!r}: best_score={entry['best_score']:.3f}")
```

```python
result = await optimize(...)
print("=== before ===")
print(result.seed)        # 合成済み full（base + parts + tune）
print("=== after ===")
print(result.prompt)      # 最適化済みの合成済み full
print("=== diff ===")
print(result.diff)        # unified diff（base/parts は context 行・tune だけ ±）
```

> **「合成済み full」の意味**: `prompt_slot(store, registry, tune=..., base=..., parts=...)` で base
> / parts が指定された場合、APO は tune 部分だけを最適化するが、`result.seed` / `result.prompt` は
> `base + parts + tune` を `\n\n` で連結した **rollout 時に agent が実際に instructions として
> 受け取る形** で返る。base / parts が無い構成（生 seed や `prompt_slot(tune=..., base=None,
> parts=())`）では tune そのものになる。

`save` は `PromptStore` のテンプレートやライブラリ管理領域を一切書き換えない（利用者が渡したパスに
のみ書く）。**出力フォルダに注意**: サンプルはリポジトリを汚さないよう一時ディレクトリ
（`tempfile.TemporaryDirectory()`）へ書き出している（`01_single_agent_apo.py`）。実運用では任意の
パスを渡してよいが、保存先は利用者の管理下とする。

---

## HITL・安全性（rollout 副作用の反復）

APO は同一 rollout を多数回実行する。承認必須ツール（`function_tool(needs_approval=True)`）を持つ
エージェントをそのまま回すと本物の副作用が反復する恐れがある。`optimize(tool_mocks=, approvals=)`
で承認を自動解決しつつツール実行だけを安全なモックへ差し替えられる（llmops の評価経路と同じ安全
機構を再利用）。

```python
result = await optimize(
    target, train=train, val=val,
    reward=tool_match(),                # OptimizeCase.expected_tools の recall
    approvals=lambda pending: pending["tool_name"] == "delete_account",
    tool_mocks={"account-agent": {"delete_account": "deleted (mock)"}},
    apo_client=my_apo_client,
)
```

**安全不変条件**: `approvals` が approve を返す `(agent_name, tool_name)` は、`tool_mocks` で実際に
モック差し替えされていなければならない。未登録ツールの approve は `OptimizeError(CONFIG_MISSING)`
で最適化失敗に倒れる（**本物の危険ツール実行を構造的に阻止**）。詳細は `04_reward_and_safety.py`
（`tool_match`）と `06_approval_match_apo.py`（`approval_match` で承認ゲート発火自体を採点）。

---

## 失敗種別（`OptimizeError` / `FailureKind`）

最適化の失敗は未捕捉例外でプロセスを止めず、`OptimizeError` に統一して送出する。`error.kind` で
種別を判別できる。

| `FailureKind` | 意味 |
|---|---|
| `EXTRA_MISSING` | `[lightning]` extra（agentlightning）未導入 |
| `CONFIG_MISSING` | 必須設定不在: `algorithm` 不正 / `train` 空 / `reward=None` / `val` 空 / `apo_client` 不在 / `slot` と `rebind` の解決不能 / `registry` 不在（グラフ最適化）/ 直接 kwargs と `config=` の同時指定 / 承認の安全違反 |
| `TRAINER_FAILED` | 最適化実行（Trainer / rollout / reward）中の失敗 |

```python
try:
    result = await optimize(...)
except OptimizeError as exc:
    if exc.kind is FailureKind.EXTRA_MISSING:
        ...   # pip install 'oai-agentspec[lightning]' を案内
    elif exc.kind is FailureKind.CONFIG_MISSING:
        ...   # 設定ミスを修正（例外メッセージに修正方法が記載される）
    elif exc.kind is FailureKind.TRAINER_FAILED:
        ...   # exc.__cause__ に元の例外がチェーンされる（ログ / リトライ判断）
```

詳細は `05_failure_handling.py`（オフラインで動く・LLM 不要）。

---

## AsyncOpenAI クライアントの作り方（Azure / OpenAI 直接）

APO は内部の textual gradient 計算 / prompt 編集に `AsyncOpenAI` 互換クライアントを必要とする。
採点用の SDK モデル（`AgentSpec.model`）とは別の関心。

**Azure OpenAI**（`examples/_shared/_azure.py` 参照）:

```python
from openai import AsyncAzureOpenAI

client = AsyncAzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
result = await optimize(target, ..., apo_client=client)
```

**OpenAI 直接**:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
result = await optimize(target, ..., apo_client=client)
```

`apo_gradient_model` / `apo_apply_edit_model` で gradient / edit 用モデル名を上書きできる
（既定: 両方 `gpt-5.4-mini`）。Azure の場合は当該デプロイ名を渡す（同名のデプロイがあれば既定値で
そのまま動く）。

---

## 仕組み（裏側）

最適化ループは agent-lightning の Trainer + APO アルゴリズム（textual gradient + beam search）に
委譲する。テンプレートエンジンは内部で jinja2 を採用し、oai-agentspec 側の `${var}` プレースホルダを
境界で `{{ var }}` に相互変換する（lightning ロジック層は `${var}` のみ扱う）。`agents` /
`agentlightning` の import は `_adapters/` 配下に閉じ、最適化ロジック層は plain データのみ扱う
（NFR-1: SDK 隔離）。

最適化対象の `HandoffGraph` / `WorkflowGraph` は内部で deepcopy + `registry.clone()` され、利用者
の宣言物・registry は最適化前後で不変（read-only 契約）。

---

## スコープ外

- RL（モデル重み更新・LightningRL / VERL・系全体の報酬信用割当・学習対象エージェント選択）は別
  extra `oai-agentspec[lightning-rl]`（後続）。
- 独自の最適化アルゴリズムの実装（外部トレーナへ委譲する）。
- 置換変数（`${var}`）の最適化（不変・実行時再注入で確定）。
- 観測連携（外部観測 SaaS へのスコア / トレース送信）は LLMOps トラックの関心事。
