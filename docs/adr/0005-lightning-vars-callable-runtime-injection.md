# 0005: vars=callable（compose 一致）を既定 build が動的 instructions として毎 run 評価し、保持は vars_map 非伝搬 + 未知キー温存で成立させる

- Status: accepted
- Date: 2026-07-23

## Context

`runtime/lightning` の APO プロンプト最適化に、実行時 context 由来の値（triage 判定結果など）を
最適化対象プロンプト内で `${var}` プレースホルダとして扱う機能を追加するにあたり、context 由来値の
宣言方法・保持・rollout 時の注入をどの層でどう実現するかを検討した。

要件は 2 つある。(1) context 由来値は最適化成果物（`OptimizeResult.prompt`）に具体値としてベイクされず
`${var}` のまま保持されること。(2) rollout では実行時 context から取得した値がプレースホルダ位置に注入された
instructions でエージェントが動作すること。(1) は「context 由来キーを vars_map に含めない」だけで
`substitute_braced` の未知キー保持（fail-open）により成立する。論点は宣言 API の形と (2) の注入をどこが担うか
である。

さらに `prompt_slot` の使い方は `PromptStore.compose` と一致させることがユーザー指示で確定した
（原文: 「store.composeにて、context_vars=["triage_result"],になっていないの、差分があるのはなぜ？」「同じにしてください」）。
`compose(vars=...)` は `dict | Callable[[ctx], dict] | None` を受理し、callable のとき render / run ごとに
context を受けて dict を返す。したがって prompt_slot も同一型・同一意味論で `vars` を受理する必要がある。

lib の不変条件は SDK 隔離（NFR-1: `from agents` は `_adapters/` 配下のみ）と build-don't-run（宣言・
build-time 検証・薄い結線に徹する）である。

検討した選択肢:

1. **`context_vars` 別チャネル方式（却下）**: context 由来値を `prompt_slot` / `prompt_slots` の
   `context_vars=` kwarg と `Slot.context_vars` フィールドで宣言し、既定 build がその名前群を
   `substitute_braced` の対象キーにする。保持要件 (1) と注入要件 (2) は満たすが、`compose` には
   `context_vars` チャネルが存在せず、ユーザー指示「compose と使い方を同じにする」に反する。
   `vars`（compose と同一型）で表現できるため別チャネルは重複であり不採用。
2. **lookup 規則のライブラリ側発明（却下）**: `context_vars` の名前から値を取り出す規則
   （mapping の `__getitem__` -> 属性 `getattr` の duck typing）をライブラリが持つ方式。
   compose では値の取り出し方は利用者 callable が全権を持つため、ライブラリが lookup 規則を発明すると
   compose と乖離する。値の取り出しは利用者 callable に全権委譲し、ライブラリは規則を持たない。
3. **`_apply_candidate` 段階での事前 evaluate（却下）**: callable を rollout の候補適用時に 1 回評価して
   dict 化し、静的 instructions にベイクする。特定 rollout の context 値が静的 instructions に固定され、
   compose(vars=callable) の「render ごとに評価」意味論と乖離するため不採用。
4. **sentinel 方式（却下）**: context 由来値を特別な sentinel オブジェクトで表現し注入点で判別する。
   `${var}` プレースホルダ + vars_map 除外という既存機構で保持が成立するのに新しい表現を導入するのは
   overcomplicated であり、`substitute_braced` の既存セマンティクスと二重管理になるため却下。
5. **固定 `context=` 引数（却下）**: `optimize(context=固定オブジェクト)` で 1 つの context を全 rollout に
   共有する。rollout 間で context オブジェクトが共有されて状態汚染が起きるため却下（`context_factory=` で
   rollout ごとに新鮮な context を生成する方針を採用・D-1）。

## Decision

`prompt_slot`（および `prompt_slots`）の `vars` を `PromptStore.compose(vars=...)` と同一型
（`dict[str, Any] | Callable[[Any], dict[str, Any]] | None`）で受理する。

- **dict**: 従来どおり静的注入。`Slot.vars` に入り、`_reinject_vars` / `run_apo` の `vars_per_slot` として
  最適化ループへ伝搬する（既存経路・無変更）。
- **callable**: `Slot.vars_fn`（新設・default None・`Slot.vars` の dict 契約とは分離）に保持し、`Slot.vars` は
  空 dict にする。callable は最適化ループへ一切伝搬せず（`vars_per_slot` は空 dict）、既定 build が SDK 動的
  Instructions 規約 `(context, agent) -> str` の callable を instructions に据える。callable の中身は
  `substitute_braced(合成済みテキスト, vars_fn(context))` とし、rollout ごと（render ごと）に評価する
  （`compose(vars=callable)` と同一の評価タイミング・第 1 引数に context をそのまま渡す・未解決キーの
  `${var}` 温存）。値の取り出しは利用者 callable の全権とし、ライブラリは lookup 規則を持たない。

保持（context 値をベイクしない）は「context 由来キーを vars_map に絶対含めない」ことで `substitute_braced` の
未知キー保持により全注入点で自動的に成立させ、新しい置換規則を作らない。`Slot.vars = {}` のとき
`_reinject_vars` の `substitute_braced` は no-op となり候補内の全 `${var}` が温存され、seed 内全 placeholder の
存在検査（境界マーカーを含む）はそのまま機能する（`_slots_norm.py` は無変更）。

付随規則:

- `vars=callable` のとき `_ensure_fixed_vars_present` の構築時検査は免除する（キー集合が実行時まで不明で
  静的検査が不可能・`compose(vars=callable)` と同一の fail-open 意味論）。
- `vars=callable` かつ `build=` 明示は `OptimizeError(CONFIG_MISSING)` で fail-closed（既定 build だけが
  `vars_fn` を評価するため）。
- SDK 隔離: 生成 callable は context を利用者 callable へ素通しするだけで `from agents` を要さない（NFR-1 維持）。
- context の rollout への配線は `context_factory=`（D-1）で行い、粒度は「1 rollout = 1 context」・承認 resume
  ループ内は SDK `RunState` 内包 context を再利用する（`resume` へ `context=` を渡さない）。

現在仕様の SoT は `docs/architecture.md`（「Agent Lightning 最適化」節のデータフロー・`${var}` 保持規則）と
`docs/usage/ops/lightning.md`（使い方・パラメータ）とし、本 ADR は判断・却下案のみを記録して仕様詳細を
重複させない。tune セレクタ・境界マーカーの判断は `docs/adr/0006-lightning-tune-selector-boundary-markers.md` を参照。

## Consequences

- + `prompt_slot` の `vars` が `compose` と完全一致の使い方になり、compose 利用者が追加の学習コストなく
  context 由来値を扱える（ユーザー指示「同じにしてください」を満たす）。
- + 既定 build（主経路）の利用者が custom build を書かずに context 由来値の保持と rollout 時注入の
  両方を得られる。
- + `Slot.vars`（dict・既存契約）と `Slot.vars_fn`（callable・新設）の分離により、`_reinject_vars` /
  `run_apo` の `vars_per_slot`（dict 前提）を無変更のまま成立させる（NFR-3 後方互換）。`vars=dict` は
  型拡張のみで全て従来コードパスを通り、既存の呼び出し・テストは無修正で通過する。
- + 保持は既存 `substitute_braced` の fail-open と vars_map 除外の再利用のみで成立し、新規メカニズム・
  新しい置換規則・lookup 規則を導入しない（build-don't-run / surgical 方針の維持）。
- + callable が context を利用者 callable へ素通しするだけのため lightning 層に `from agents` が増えず
  SDK 隔離（NFR-1）を維持する。
- - context 値の取得不能時に `${var}` が残る（fail-open）ため、注入漏れが実行時に silent に通過しうる。
  これは triage が run 途中で値を確定させるユースケースを成立させるための意図的なトレードオフであり、
  最適化成果物のプレースホルダ喪失は既存 `_reinject_vars` の fail-closed で別途検出する。
- - 静的値と動的値を併用したい利用者は callable の返す dict に静的値も含める必要がある（混在受理はしない）。
  これは compose 利用者と同一の作法であり、受理型で dict / callable が二分される単純さを優先した。
- D-2 の supersede: 要件書 D-2 の `context_vars` フィールド追加方式は本ユーザー指示により supersede され、
  D-2'（`vars` の compose 同一型受理・`vars_fn` 分離）へ置き換わった。`context_vars` は kwarg・フィールドとも
  設けない（内部専用属性としても残さない）。未実装・未コミット段階のため互換影響はない。sentinel 不採用の
  判断は D-2' でも維持する。

## Confirmation

- `${var}` 保持（context 値の非ベイク）・rollout 時注入・resume 内 context 共有・SDK 隔離維持の強制手段:
  `tests/runtime/lightning/`（`test_slots_l1` が FR-3 の保持と既定 build の callable 注入・`vars_fn` 分離を、
  `test_optimizer_l2` が context_factory の rollout ごと新鮮・resume 内共有を検証）。
- SDK 隔離（lightning 層に `from agents` / `from openai` を増やさない）は
  `grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` が空であること（既存スモークと同一基準）で担保する。
