# 0007: prompt_slot の旧 shape（tune=<str> の縮退経路）を削除し新 shape 一本化する

- Status: accepted
- Date: 2026-07-24

## Context

ADR 0006 で `prompt_slot` を `PromptStore.compose` 一致の新 shape（`agent=` / `layout=`
+ `tune=` セレクタ）へ拡張した際、既存呼び出し（`prompt_slot(store, reg, tune="triage")`）と
の後方互換のため、`agent=None` かつ `layout=None` かつ `tune` が単一 str のときに旧経路
（`_legacy_prompt_slot` / `_default_build` / `_compose_fixed` の従来合成・`Slot.segments = ()`）
へディスパッチする分岐を残していた（要件 NFR-3・後方互換）。

その後 Codex review が旧経路 × callable vars の穴を指摘した:

> [P2] Handle callable vars before legacy dispatch — src/oai_agentspec/runtime/lightning/slots.py:371-374
> When callers use the newly documented `vars=callable` form with the legacy-compatible call shape
> (`prompt_slot(..., tune="triage")` without `agent`/`layout`), this branch forwards the callable into
> `_legacy_prompt_slot`, which then executes `dict(vars or {})` and raises a raw `TypeError`.

旧経路は callable vars を受け入れない設計にもかかわらず、公開シグネチャは受理してしまい即
`TypeError` で落ちる。fail-closed に倒すか、旧経路も callable 対応にするか、あるいは旧経路自体を
削除するかの選択となった。

利用者判断:
- 本 lib は pre-1.0（v0.3.x）で SemVer 上 minor bump で破壊的変更が可能
- 新 shape で `prompt_slot(store, reg, agent="bot")` と書けば旧 shape `prompt_slot(store, reg, tune="bot")`
  と同等の意味論が得られる（呼び出しコストほぼ同じ）
- 旧経路存続に構造的意義は薄く、Codex 指摘のような組み合わせの穴を今後も抱え続けるコストが上回る

## Decision

`prompt_slot` から旧 shape（`agent=None` + `layout=None` + `tune=<str>`）の縮退経路を **削除**し、
`agent=` または `layout=` のいずれかを必須とする新 shape 一本化に統一する。要件 NFR-3
（後方互換）は本 ADR により **撤回** する。同時に旧経路とともに dead になる関連インフラ
（`Slot.fixed` フィールド / `run_apo` の `fixed=` パラメータ / `_compose_full` / `compose_with_vars`）
も削除する。

- `_legacy_prompt_slot` / `_default_build` / `_compose_fixed` を削除（旧経路専用の 3 関数）
- `prompt_slot` の dispatch を単純化: `agent is None and layout is None` を無条件で
  `OptimizeError(FailureKind.CONFIG_MISSING)` に倒す
- `Slot.fixed` フィールドを削除（新 shape では `Slot.segments` が構成情報 SoT・fixed は常に空で
  唯一の consumer が旧経路依存の plumbing だった）
- `_adapters/lightning.py::run_apo` の `fixed=` パラメータと `_compose_full` を削除（新 shape の
  full 再合成は optimizer の `_recompose_new_shape_results` が `Slot.segments` を SoT に担う・
  run_apo は tune 側 `${var}` 再注入のみ担う）
- `_placeholders.compose_with_vars` を削除（`_compose_full` が唯一の consumer だった）
- 追加の regression guard として custom build + multi-tune の組み合わせを `OptimizeError`
  （`CONFIG_MISSING`）で fail-closed（境界マーカー入り seed が custom build 経路で
  `OptimizeResult` に literal 漏出するのを防ぐ）
- ADR 0006 の Status は `accepted` のまま維持する（0006 は旧経路の存続を義務化していないため
  supersede は不要・本 ADR は NFR-3 の撤回と関連インフラの整理を宣言する）

## Consequences

**Pro**:
- Codex 指摘の callable vars × 旧 shape の穴が消滅（moot 化）
- 分岐削減により slots.py の実装表面が減り、`Slot.fixed` を SoT とする旧経路と `segments` を SoT
  とする新経路の二重管理も解消
- 公開シグネチャに残る一貫性: `agent=` / `layout=` のいずれかが必須という単純な契約

**Con**:
- 破壊的変更: 既存の `prompt_slot(store, reg, tune=<str>)` 呼び出しは `OptimizeError` になる
  （pre-1.0 のため許容範囲）
- `Slot` dataclass の shape 変更: `fixed` フィールド削除により `Slot(name=..., seed=..., build=...,
  vars=..., fixed=...)` を positional / keyword で書いていた利用者は要修正（pre-1.0 のため許容）
- custom build + multi-tune の組み合わせ利用者がいた場合は fail-closed に変更（従来は境界マーカー
  漏出のため動作しても壊れていた）

## 却下案

### 案 A: 旧経路を残しつつ callable vars 対応を追加

`_legacy_prompt_slot` を callable vars に対応させる。互換を維持できるが、旧経路の位置づけが
「新 shape の縮退形」から「新 shape と同等機能を持つ並行経路」に変質し、二重実装の維持コスト
が増える。却下。

### 案 B: 旧経路を残しつつ callable vars 組み合わせのみ fail-closed

旧経路 × dict vars は動くが旧経路 × callable vars は `OptimizeError(CONFIG_MISSING)` に倒す。
穴は塞げるが「特定の組み合わせだけエラー」という説明の複雑さが残り、公開契約として不自然。
却下。

### 案 C: 旧経路を deprecate warning 付きで残す

`DeprecationWarning` を発火させて v0.4.0 で削除予告する。段階的移行として妥当だが、pre-1.0 で
かつ利用者判断が「即削除で良い」であるため、段階導入は複雑化の割にリターンが小さい。却下。

## 影響

- テスト: 旧 shape を使っていた既存テスト（`tests/runtime/lightning/test_slots_l1.py` /
  `test_optimizer_l2.py`）を新 shape に書き換え。旧 shape 専用テストと `Slot.fixed` /
  `fixed=` / `compose_with_vars` 経路の専用テストを削除
- regression guard 追加:
  - `test_prompt_slot_rejects_legacy_shape` / `test_prompt_slot_rejects_missing_agent_and_layout`
    （旧 shape 呼び出しが `OptimizeError(CONFIG_MISSING)` を出すことを固定）
  - `test_prompt_slot_new_shape_custom_build_multi_tune_fail_closed`（custom build + multi-tune が
    fail-closed される・境界マーカー漏出防止）
  - `test_optimize_custom_build_slot_result_unchanged`（segments 空 slot は run_apo 返却を素通し）
- docs: `docs/usage/ops/lightning.md`（「旧経路の互換」節を削除・パラメータ記述の更新）・
  `docs/architecture.md`（dispatch 記述の更新）・`examples/lightning/README.md`（旧経路記述の
  除去）・`examples/lightning/07_composite_reward_apo.py` の comment を更新
- 品質保証台帳（`docs/QUALITY-GUARANTEES.md`）に「旧 shape 呼び出しは fail-closed される」を
  1 行追加
