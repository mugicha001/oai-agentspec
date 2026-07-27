# Agent Lightning APO（プロンプト自動最適化）

## 何を解決するか

`AgentSpec` / `HandoffGraph` / `WorkflowGraph` のプロンプトを Agent Lightning に委譲し、reward 関数で採点しながら textual gradient + beam search で自動改善します。プロンプトを「slot」として抽出（`prompt_slot` / `prompt_slots`）し、`optimize()` に reward と評価ケースを渡すだけで学習ループが回ります。

本 extra は APO（プロンプト最適化）のみを提供します。`optimize()` に `algorithm="rl"` を渡すと未対応として明確なエラーで案内されます。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| `contains` / `exact` | 出力文字列マッチ | 決定的な期待出力 |
| `tool_match` | 期待 tool 呼び出しの一致 | ツール正しさで学習 |
| `approval_match` / `route_match` / `last_agent_match` | 承認ゲート / 経路 / 最終 agent 一致 | HITL / handoff 学習 |
| `judge(rubric, model)` | LLM-as-judge | 意味的品質 |
| 複合 reward | 上記の重み付き合成（利用者側で組む） | 系全体の総合最適化 |
| `prompt_slot` | 単一 spec の slot 抽出 | 単一 agent APO |
| `prompt_slots` | 複数 spec 一括 | グラフ全体 APO・全 agent 同一構成で per-agent 差分が `tune` だけ |
| `prompt_slot_factory` | 共通既定値を束ねた slot 生成 callable | per-agent で `base` / `parts` / `vars` が違う、または `layout` / `build` が要る |

## 使い方

- import: `from oai_agentspec.runtime.lightning import (optimize, OptimizeConfig, OptimizeCase, OptimizeResult, Slot, RolloutResult, FailureKind, OptimizeError, contains, exact, tool_match, approval_match, route_match, last_agent_match, judge, prompt_slot, prompt_slots, prompt_slot_factory, train_val_split)`
- extras: `pip install oai-agentspec[lightning]`（`agentlightning[apo]`）
- 依存 env: 学習に使う Model の env

```python
from openai import AsyncOpenAI
from oai_agentspec.runtime.lightning import (
    OptimizeCase, contains, optimize, prompt_slot, train_val_split,
)

slot = prompt_slot(store, registry, agent="triage")
cases = [OptimizeCase(input="請求書ください", expected_output="請求")]
train, val = train_val_split(cases, val_ratio=0.2)

result = await optimize(
    triage_spec,
    train=train,
    val=val,
    reward=contains(),                 # `expected_output` を既定参照
    slot=slot,
    registry=registry,
    apo_client=AsyncOpenAI(),          # APO 必須
)
print(result.prompt)                   # 最適化済みテキスト（${var} 保持）
print(result.diff)                     # seed vs prompt の unified diff
```

### compose 一致 shape（tune セレクタ + vars=callable + context_factory）

`prompt_slot` の使い方は `PromptStore.compose` と一致する。`agent` / `base` / `parts`（または `layout`）でプロンプト全体を組み立て、最適化対象セグメントを `tune` セレクタで選ぶ。`tune` に含まれないセグメント（base を含む）は固定される。複数セグメントは構成順（base -> parts -> agent）に連結した 1 テキストとして単一 APO ループで最適化され、rollout 時は固定セグメントと構成順どおり再合成される。実行時 context 由来の値は `vars` に callable を渡して注入し、成果物では `${var}` のまま保持する。

```python
slot = prompt_slot(
    store, registry,
    agent="triage",                       # compose と同名同位置・spec 解決名を兼ねる
    base="main",
    parts=["style", "safety"],
    tune=["main", "triage"],              # base も選択対象・非選択（style/safety）は固定
    vars=lambda ctx: {                    # compose(vars=callable) と同一型
        "tone": "丁寧語",                  # 静的値も callable の返す dict に含めてよい
        "triage_result": ctx.context.triage_result,
    },
)
result = await optimize(
    graph,
    slot={"triage": slot},
    registry=registry,
    train=train, val=val,
    reward=reward,
    apo_client=AsyncOpenAI(),
    context_factory=lambda: TriageContext(),   # rollout ごとに新鮮な context
)
print(result.prompt)   # "<最適化後 main>\n\n<style>\n\n<safety>\n\n... ${triage_result} ..."
```

`vars` に callable を渡すと、値は最適化ループへ一切伝搬せず、既定 build が SDK 動的 Instructions 規約 `(context, agent) -> str` の instructions を据え、rollout ごとに `vars_fn(context)` を評価して `${var}` 位置へ注入する（`compose(vars=callable)` と同一の評価タイミング・引数・未解決キーの `${var}` 温存）。最適化成果物では `${tone}` / `${triage_result}` は具体値がベイクされず保持される。`vars` に dict を渡した場合は従来どおり静的注入（未知キーのみ `${var}` 保持）で、callable 経路との差は受理型のみで二分される。承認 resume ループ内は同一 context（SDK `RunState` 内包 context の再利用）で継続する。この判断の詳細は `docs/adr/0005-lightning-vars-callable-runtime-injection.md`・`docs/adr/0006-lightning-tune-selector-boundary-markers.md` を参照。

### layout で構成順を明示指定する

compose と同じく `layout` に qualified 参照列を渡すと、その並びがそのまま構成順になり `agent` / `base` / `parts` の構成指定は無視される（compose と完全一致の意味論）。`tune` の照合先は layout 列になる。

```python
slot = prompt_slot(
    store, registry,
    layout=["base:main", "part:style", "agent:triage"],  # 並びが構成順
    tune=["base:main", "agent:triage"],                   # 照合先は layout 列
)
```

`agent` を省略した場合、layout 内に `agent:X` 参照がちょうど 1 つあれば X が spec 解決名（`Slot.name`）になる（0 個または複数は `OptimizeError(CONFIG_MISSING)`）。`prompt_slots` は `layout` 非対応で、layout が必要なエージェントは `prompt_slot` を個別に呼んで slot mapping を組み立てる。

### per-agent の差分だけを上書きする

`prompt_slot_factory` は `store` / `registry` と共通既定値（`base` / `parts` / `vars` など `prompt_slot` の全 kwarg）を
束ね、エージェントごとの差分だけを本物の kwargs で受け取る callable を返す。返り値の callable が返すのは `Slot` で、
`optimize(slot=)` は `Slot` の列を受理して `Slot.name` をキーとする mapping へ正規化するため、エージェント名は
1 度書けばよい。

```python
from oai_agentspec.runtime.lightning import optimize, prompt_slot_factory

make_slot = prompt_slot_factory(
    store, registry,
    base="main", parts=["style"], vars={"org": "AgentSpec"},
)

result = await optimize(
    graph,
    slot=[
        make_slot("triage",  tune=["main", "triage"]),
        make_slot("billing", base="common", parts=["style", "billing_rules"],
                             vars={"tone": "formal"}),     # org は自動マージ
        make_slot("support", base=None),                   # 共通 base の打ち消し
    ],
    registry=registry,
    train=train, val=val, reward=reward, apo_client=client,
)
```

共通 `vars` と per-agent `vars` はマージされる（同一キーは per-agent 優先）。`vars` 以外の kwarg は置換で、`parts` は
列全体の差し替えになる。マージ規則の詳細は `docs/adr/0008-lightning-per-agent-slot-factory.md` を参照。キー名の typo や
`agent` の二重指定は委譲先 `prompt_slot` の呼び出しで `TypeError` になる。

- `vars=None` / `vars={}` による共通 `vars` の打ち消しは、固定セグメント（base / parts / 非 tune の agent）に `${var}` が
  残っていない場合に限り成立する（残っていると `_ensure_fixed_vars_present` が `OptimizeError(CONFIG_MISSING)` を送出する）。
- ファクトリは常に `agent=` を渡すため、`layout` のみを指定して `agent:X` 参照から `Slot.name` を暗黙解決する経路は
  使えない（必要なら `prompt_slot` を直接呼ぶ）。

### 境界マーカーについて

複数セグメントを連結するとき、内部的に境界へ予約 braced placeholder（`${oas_boundary_N}`）を挟んで固定・最適化セグメントの再インターリーブ位置を保全する。マーカーは rollout 合成・`OptimizeResult` の full 合成で分割消費され、成果物（`prompt` / `seed` / `diff`）には一切現れない。`oas_boundary_` 接頭辞は境界マーカー予約のため、seed 本文・固定セグメント本文・dict vars のキーで使用できない（使用すると slot 構築時に専用メッセージで `OptimizeError(CONFIG_MISSING)`）。

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `optimize` の主要パラメータ（15 個超のため主要 10 個に絞る・残りは docstring 参照）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `target` | `Any` | 必須 | AgentSpec / WorkflowGraph / HandoffGraph |
| `algorithm` | `str` | `"apo"` | `"rl"` は本 extra で未対応 |
| `train` | `Sequence[Any]` | 必須（kw_only） | 最適化 / rollout に使う入力ケース |
| `val` | `Sequence[Any] \| None` | `None`（**必須**・空は CONFIG_MISSING） | 検証ケース |
| `reward` | `Callable[[RolloutResult], float \| Awaitable[float]]` | 必須（kw_only） | 報酬算出 |
| `registry` | `AgentRegistry \| None` | `None` | HandoffGraph 必須 |
| `slot` | `Slot \| str \| Iterable[Slot] \| dict[str, Slot \| str] \| None` | `None` | 最適化対象スロット（`Slot` の列は `Slot.name` をキーとする mapping へ正規化する） |
| `rebind` | `Callable[[Any], Any] \| None` | `None` | 生 seed 経路で必須 |
| `tool_mocks` | `dict[str, dict[str, Any]] \| None` | `None` | rollout 副作用の安全化 |
| `approvals` | `Callable[[dict], bool] \| None` | `None` | 承認自動解決 |
| `apo_client` | `Any` | `None`（**APO 必須**） | `AsyncOpenAI` 互換クライアント |
| `context_factory` | `Callable[[], Any] \| None` | `None` | rollout ごとに新鮮な実行時 context を生成する factory。`run_with_observation(context=...)` 経由で SDK `Runner.run(context=...)` へ素通しする（rollout 間で context を共有しない） |

省略した追加 kwarg（`config` / `rounds` / `concurrency` / `timeout_seconds` / `store` / `apo_gradient_model` / `apo_apply_edit_model` / `apo_beam_width` / `apo_branch_factor` / `tracer`）は `OptimizeConfig` の各フィールドと同じ意味で、直接 kwargs か `config=OptimizeConfig(...)` の一方で渡す（同時指定は `CONFIG_MISSING`）。

### `OptimizeConfig`（frozen・主要 10 個）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `concurrency` | `int \| None` | `None` | rollout 並列度 |
| `rounds` | `int \| None` | `None` | 訓練ラウンド数（APO は `beam_rounds` にマップ） |
| `timeout_seconds` | `float \| None` | `None` | APO 1 batch タイムアウト（None で APO 既定 3600 秒） |
| `store` | `Any` | `None` | Agent Lightning Store（不透明値） |
| `apo_client` | `Any` | `None` | APO 必須 |
| `apo_gradient_model` | `str \| None` | `"gpt-5.4-mini"` | textual gradient 用モデル名 |
| `apo_apply_edit_model` | `str \| None` | `"gpt-5.4-mini"` | prompt edit 適用モデル名 |
| `apo_beam_width` | `int \| None` | `None` | beam 幅 |
| `apo_branch_factor` | `int \| None` | `None` | beam 分岐数 |
| `tracer` | `Any` | `None` | 独自 Tracer（escape hatch） |

### `OptimizeCase`（frozen）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `input` | `str` | 必須 | rollout への入力 |
| `id` | `str \| None` | `None` | ケース識別子 |
| `expected_output` | `str \| None` | `None` | `contains` / `exact` 既定参照 |
| `expected_tools` | `list[str]` | `[]` | `tool_match` 既定 |
| `expected_route` | `list[str]` | `[]` | `route_match` 既定 |
| `expected_last_agent` | `str \| None` | `None` | `last_agent_match` 既定 |
| `expected_approvals` | `list[str]` | `[]` | `approval_match` 既定 |
| `metadata` | `dict[str, Any]` | `{}` | 補助情報 |

### `Slot`（frozen）

`name` / `seed` / `build: Callable[[str], Any]` / `vars: dict[str, Any] = {}` / `fixed: str = ""` / `segments: tuple[SlotSegment, ...] = ()` / `vars_fn: Callable[[Any], dict[str, Any]] | None = None`。

- `vars` は静的注入用の dict（既存契約不変）。`prompt_slot(vars=<dict>)` はここに入る。
- `vars_fn` は `prompt_slot(vars=<callable>)` を渡したときの保持先。callable のとき `vars` は空 dict になり、既定 build が生成する動的 instructions が rollout ごとに `vars_fn(context)` を評価して注入する。
- `segments` は `prompt_slot` / `prompt_slots` が自動設定する構成順の構造情報（`SlotSegment` 要素の列）で、rollout 合成と `OptimizeResult` の full 合成が同一 SSoT ヘルパで参照する。custom build・手書き `Slot` では空のまま「run_apo 返却をそのまま尊重する」経路を通る。
- `SlotSegment`（frozen: `ref`（`base:main` 等の qualified 参照）/ `text`（`${var}` 保持の本文）/ `tune`（最適化対象フラグ））は内部構造で、公開契約（`__all__`）には含めない（利用者が手書きする対象ではない）。

### `OptimizeResult`（frozen）

`prompt: str | dict[str, str]` / `train_score: float` / `val_score: float | None = None` / `history: list[HistoryEntry] = []` / `seed: str | dict[str, str] = ""` / `diff: str | dict[str, str] = ""`。`.to_dict()` / `.save(path)` を提供。

### `RolloutResult`（frozen・reward が受ける plain 観測）

`case: Any` / `output: str` / `tool_calls: list[str] = []` / `fired_approvals: list[str] = []` / `route_steps: list[str] = []` / `last_agent: str | None = None`。

### reward ファクトリ（すべて `field` 位置引数 1 個で既定は `OptimizeCase` の対応フィールド名）

- `contains(field="expected_output")`
- `exact(field="expected_output")`
- `tool_match(field="expected_tools")`
- `route_match(field="expected_route")`
- `last_agent_match(field="expected_last_agent")`
- `approval_match(field="expected_approvals")`
- `judge(rubric, model)` — 2 引数（rubric: str / model: Any）

### `prompt_slot(store, registry=None, agent=None, *, base=None, parts=(), layout=None, tune=None, vars=None, build=None)`

- `agent: str | None` — 最適化対象エージェント名（compose と同名同位置・`Slot.name` = registry の spec 解決名）。`agent` または `layout` のいずれかが必須。両方未指定は `OptimizeError(CONFIG_MISSING)`（詳細は ADR 0007）。
- `base: str | None` / `parts: Sequence[str]` — compose と同じ構成指定。新 shape では `tune` の選択対象セグメントになる。
- `layout: Sequence[str] | None` — compose と同一意味論の qualified 参照列。指定時は並びがそのまま構成順になり `agent` / `base` / `parts` の構成指定は無視され、`tune` の照合先は layout 列になる。
- `tune: str | Sequence[str] | None` — 最適化対象セグメントのセレクタ。要素はセグメント名（base 名 / part 名 / agent 名）または qualified 参照（`base:main` / `part:style` / `agent:triage`）で照合する。`Sequence` は構成順（base -> parts -> agent）に連結し 1 候補テキストとして単一 APO ループで最適化する（列挙順は順序に使わない）。`agent` 指定時に省略すると agent セグメントのみを最適化する。
- `vars: dict | Callable[[Any], dict] | None` — compose と同一型。dict は静的注入、callable は `Slot.vars_fn` に保持され rollout ごとに評価注入される（成果物は `${var}` 保持）。
- `build: Callable[[str], AgentSpec] | None` — 候補テキストから `AgentSpec` を組む関数。省略時は既定 build（registry 登録 spec を複製し instructions を差し替え・vars=callable のとき動的 instructions 生成）。

### `prompt_slots(store, registry, agents, *, base=None, parts=(), tune=None, vars=None)`

- `agents: Sequence[str]` — 各名前を新 shape の `agent=` として slot を生成する（base / parts / vars は全 slot 共通）。
- `tune: dict[str, str | Sequence[str]] | None` — agent 名ごとのセレクタ。`None` のときは各 slot で agent セグメントのみ最適化（従来動作）。`mapping` のキーが `agents` に含まれない場合は fail-closed。
- `vars: dict | Callable[[Any], dict] | None` — 全 slot 共通（`prompt_slot` と同一型）。
- `layout` は非対応（layout が必要なら `prompt_slot` を個別に呼ぶ）。

fail-closed 検証（`OptimizeError(FailureKind.CONFIG_MISSING)`）:

- `tune` の `Sequence` が空 / 重複要素を含む（plain と qualified の表記違いで同一セグメントを指す場合を含む）/ 構成に存在しない名前を含む / plain 名が複数のセグメント名前空間に一致して一意に定まらない（FR-1）
- `agent=None` かつ `layout=None`（`agent=` または `layout=` のいずれかが必須・詳細は ADR 0007・FR-1）
- `vars` に callable を渡し、かつ `build=` を明示する（callable の評価は既定 build のみが担う・FR-3）
- `oas_boundary_` 予約接頭辞が seed 本文・固定セグメント本文・dict vars のキーに現れる（境界マーカー予約・専用メッセージ）
- `layout` が空 / 重複参照を含む / `agent=None` かつ layout 内の `agent:` 参照が 0 個または複数 / `layout` 指定 + `tune=None` かつ layout 内に agent セグメント不在（FR-4）

既存経路のエラー型（registry 不在の `ValueError`・セグメント解決不能の `KeyError`）は後方互換のため型不変で伝搬する。

### `prompt_slot_factory(store, registry=None, **defaults)`

- 返り値: `Callable[..., Slot]`。`make(agent, **overrides) -> Slot` として呼び、`agent` は位置引数 1 個。
- `**defaults` — `prompt_slot` の全 kwarg（`base` / `parts` / `layout` / `tune` / `vars` / `build`）を共通既定値として
  束ねる。許可キーは制限せず、`prompt_slot` にそのまま素通しする。
- `**overrides` — エージェントごとの差分。`vars` は双方が dict のときマージされ（同一キーは per-agent 優先・既定側の
  dict は非破壊）、それ以外の kwarg は置換になる。`base=None` / `parts=[]` / `tune=None` / `vars=None` は明示的な
  打ち消しとして働く。
- 未知キー（typo）や `defaults` への `agent` 混入は、ファクトリ生成時ではなく `make()` 呼び出し時に Python の
  `TypeError` になる（ライブラリ側の許可キーリスト検査は持たない）。
- ファクトリは常に `agent=` を渡すため、`layout` のみによる `Slot.name` の暗黙解決は使えない。

### `train_val_split(data, *, val_ratio=0.2, seed=0, shuffle=True)`

### `FailureKind`（StrEnum）: `EXTRA_MISSING` / `CONFIG_MISSING` / `TRAINER_FAILED`

### `OptimizeError`

`OptimizeError(kind: FailureKind, message: str)`。

## 判断軸

- 期待出力が決定的 → **`contains` / `exact`**、意味的品質 → **`judge`**、tool / handoff / 承認は該当 reward
- 単一 agent の改善は **`prompt_slot`**、グラフ全体は **`prompt_slots`**
- 学習 rollout は API コスト源。`OptimizeConfig` で試行回数を制御し `train_val_split` で汎化確認

## 落とし穴

- `agentlightning` は private API 依存のため `pyproject.toml` で patch pin されている
- `OptimizeError` / `FailureKind` は必ず判別ハンドリング（rollout 失敗を握り潰さない）
- `val` は必須（APO の beam search 契約）。空は `CONFIG_MISSING`
- `apo_client` は APO 必須（未指定は `CONFIG_MISSING`）
- 結果テキストは `result.prompt`（`best_prompts` ではない）

## 参照

- 詳細設計: `docs/architecture.md`（Agent Lightning 節）
- 具体例: `examples/lightning/01_single_agent_apo.py` 〜 `07_composite_reward_apo.py`

## 次

[../runtime/realtime.md](../runtime/realtime.md) — Realtime エージェント
