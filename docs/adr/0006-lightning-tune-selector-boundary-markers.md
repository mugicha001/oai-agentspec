# 0006: prompt_slot を compose と一致させ tune をセレクタ化し、境界は予約 placeholder マーカーで保全して full 合成する

- Status: accepted
- Date: 2026-07-23

## Context

`runtime/lightning` の APO プロンプト最適化を、本番構成（複数セグメント instructions・動的 Instructions・
handoff 間 context 受け渡し）をそのまま再現できる形に拡張するにあたり、`prompt_slot` の構成規則・
最適化対象の選び方・複数セグメントの連結と rollout / 成果物合成の境界保全を検討した。

ユーザー指示により `prompt_slot` の使い方は `PromptStore.compose` と一致させることが確定した
（原文: 「使い方ちゃんと一致させたい、compose同様、 agent=None, *, base=None, parts=(),で組み立てた上で、
修正するターゲットをtuneとかに入れ」「あ、やりたいのは個別最適化でなく、指定した物以外を固定するということ」）。
compose は `agent` / `base` / `parts`（または qualified 参照列 `layout`）でプロンプト全体を構成順
（base -> parts -> agent）に組み立てる。したがって prompt_slot も同一構成規則で全体を組み立て、最適化対象を
セレクタで選び、非選択セグメント（base を含む）を固定する形にする必要がある。

論点は (A) tune の照合規則、(A2) layout の受理、(B) 不連続 tune の連結順と rollout 合成順序、
(C) 連結時の境界保全、(D/H) compose(vars=callable) との統合、(E) 後方互換と spec 解決名の担い手、
(G) `OptimizeResult` の合成経路と構造情報の持ち場所である。制約は build-don't-run（upstream APO 0.3.x は
単一プロンプトの beam search・plain テキスト編集のみ）と NFR-4（最適化ループ回数を増やさない）と
NFR-3（後方互換・`__all__` 不変）。vars=callable の判断は `docs/adr/0005-lightning-vars-callable-runtime-injection.md` を参照。

## Decision

### tune のセレクタ化と照合規則（論点 A）

`tune` の各要素を、当該呼び出しの構成セグメント（base 名・各 part 名・agent 名）と照合する。qualified 参照
（`base:main` / `part:style` / `agent:triage`・compose の layout 記法の再利用）も常に受理する。plain 名が複数の
セグメント名前空間に一致して一意に定まらない場合は `OptimizeError(CONFIG_MISSING)` で fail-closed し qualified
参照を要求する（silent な優先順位を持たせない）。構成不在・空・重複（plain と qualified の表記違いで同一
セグメントを指す場合を含む）も fail-closed。

- 却下: 役割語 shorthand（`tune=["base","agent"]` の "base" で base セグメントを指す）。part 名が literal に
  "base" のとき第 3 の名前空間衝突を生み、compose に無い新記法になるため不採用。

### layout の受理（論点 A2）

`prompt_slot(layout=...)` を compose と同一意味論で受理する（qualified 参照列をそのままの順序で構成に使い、
`agent` / `base` / `parts` の構成指定を無視・`base`/`parts` 併用時は layout 優先）。`tune` の照合先は layout 列。
spec 解決名は `agent=` 明示、または layout 内 `agent:X` がちょうど 1 つのとき X で暗黙解決する（0 個または
複数は `OptimizeError(CONFIG_MISSING)`）。`prompt_slots` は layout 非対応（共有 layout に各エージェント固有の
`agent:<name>` 参照を書く手段が compose に無く、role トークン等の新記法は論点 A の却下と矛盾するため）。

### 連結順と rollout 合成（論点 B）

候補テキストは tune セグメントを常に構成順（base -> parts -> agent）に連結する（`tune` の列挙順は順序に
使わず重複検査にのみ使う）。rollout 時は候補を分割し、`Slot.segments`（構成順の構造情報）に沿って固定
セグメントと再インターリーブして `\n\n` 連結する。旧 shape（`agent=None` + `tune=str`）の
「fixed + "\n\n" + candidate」前置合成は従来経路のまま（完全互換）。

- 却下: セグメントごとに独立 APO ループ。「個別最適化でなく指定した物以外を固定」のユーザー確定に反し
  NFR-4（ループ回数非増加）にも違反するため不採用。

### 境界保全 = 予約 placeholder マーカー（論点 C）

tune セグメントが 2 個以上のとき、seed 連結時に `${oas_boundary_1}` .. `${oas_boundary_(n-1)}` を境界に挟む。
境界マーカーは新記法を作らず braced placeholder の予約名で表現する。これにより (1) `_reinject_vars` の既存
fail-closed（seed 内全 placeholder の候補内存在検査）がマーカー喪失検出を無変更でカバーし、(2)
`substitute_braced` の未知キー保持でマーカーが全注入点を素通りする。

- exact-once（重複・増殖）検査は `_apply_candidate` が build 呼び出し前に `split_marked`（違反 = None）で行い、
  違反候補は `_reinject_vars` の None と同一の per-candidate 無効化経路（候補 reward 0.0）で処理する。build 内で
  `OptimizeError` を raise する方式は不採用（LitAgent の `except OptimizeError` が critical_error sentinel 経由で
  最適化全体を abort するため）。build は検証済み候補のみ受け取り同じ `split_marked` を呼ぶ（決定的・SSoT）。
- post-fit フォールバック: `run_apo` の post-fit placeholder 検査を「`oas_boundary_` 接頭辞の placeholder は
  seed と出現回数一致」まで拡張し、exact-once 違反 best は既存 placeholder_fallback と同一の seed フォールバック
  対象に含める（seed は構築時に必ず split 可能で optimizer の再合成が常に成功する）。予約接頭辞定数は
  `runtime/lightning` 側に置き `_adapters` からは関数内遅延 import で参照する。
- 予約接頭辞の衝突検査: slot 構築時に tune seed 本文・固定セグメント本文の literal `${oas_boundary_*}`・dict vars
  のキーのいずれかに `oas_boundary_` 接頭辞が現れたら、`_ensure_fixed_vars_present` より前に専用メッセージで
  `OptimizeError(CONFIG_MISSING)` に倒す（callable の返すキーは実行時まで不明だが、substitute はマーカー分割
  消費後のテキストに適用されるため実害経路なし）。

- 却下: (1) マーカーなし `\n\n` 連結のみ。不連続 tune の再インターリーブが不可能・境界崩れを検出できないため
  却下。(2) Trainer に JSON 等の構造化出力を求める。APO 0.3.x は plain テキスト編集のみで独自エンジン化は
  build-don't-run 違反のため却下。(3) マーカー文字列の正規化工程。full 合成でマーカーは分割消費されるため
  独立工程は不要・廃止。

### compose(vars=callable) との完全一致統合（論点 D/H）

`prompt_slot(vars=...)` の受理型・意味論を `compose(vars=...)` と完全一致させる。詳細と却下案は
`docs/adr/0005-lightning-vars-callable-runtime-injection.md`（vars=callable / vars_fn 分離）に記録する。

- 却下: context_vars 別チャネル方式（ユーザー指示により compose との使い方一致を優先して不採用・0005 参照）。

### `name=` 不採用と後方互換（論点 E）

spec 解決名は `agent=` 引数が担い、`tune` はセレクタとして役割分離する。`agent=None` + `tune` 単一 str は旧経路
完全互換（tune が seed 解決名かつ `Slot.name`）。`agent=None` + `tune` Sequence（layout 未指定）は spec 解決名が
定まらないため `OptimizeError(CONFIG_MISSING)`。

- 却下: `name=` kwarg の新設。旧設計は spec 解決名を補うため `name=` を検討したが、新 shape では `agent=` が
  解決名を担うため不要と結論し新設しない（未実装・未コミット段階のため互換影響なし）。

### OptimizeResult の合成経路と構造情報の持ち場所（論点 G）

`OptimizeResult.prompt` / `seed` / `diff` は rollout 実体一致の full 合成とする（既存 SSoT 契約「prompt は
rollout 実体と一致」を維持）。構造情報は `Slot.segments: tuple[SlotSegment, ...] = ()`（`SlotSegment` は frozen:
`ref` / `text` / `tune`）に持つ。既定 build の closure と optimizer の結果整形の双方がこの同一構造を参照し、合成は
双方とも `_placeholders.compose_segments` を呼ぶ（rollout と成果物の drift を SSoT で不可能にする）。`segments` を
設定するのは新 shape の `prompt_slot` / `prompt_slots` のみで、旧 shape・custom build・手書き `Slot` は
`segments=()` のまま従来経路（default 付き追加のため NFR-3 成立）。`SlotSegment` は `__all__` に追加しない。

optimizer は run_apo へ渡す `fixed` を「segments 非空 slot は `""`、それ以外は従来どおり `Slot.fixed`」とし、
run_apo（`_compose_full` 含む）は無変更のまま、segments 非空 slot の seed / best を tune 連結（マーカー入り）の
まま返す。optimizer が返却後に当該 slot の seed / best を `split_marked` -> `compose_segments`（fixed 側 vars
注入込み）で full 再合成し `prompt` / `seed` を上書きし、diff を full 再合成後テキストから `_unified_diff` 同一
規則で再計算する（run_apo の unified diff 文字列への置換は行構造を壊すため行わない）。

- 却下: (1) tune-only 連結返却（D-3「rollout 実体一致 full 合成」に反する）。(2) マーカー文字列正規化を独立工程に
  する案（full 合成に吸収され不要）。(3) run_apo の diff 文字列への直接置換（行構造を壊す）。

## Consequences

- + `prompt_slot` の使い方が `compose` と一致し、構成規則・qualified 参照記法を新規発明せず踏襲する
  （学習コスト最小・ユーザー指示の充足）。
- + 境界保全が既存 placeholder 機構（`_reinject_vars` / `substitute_braced` の未知キー保持）の予約名運用のみで
  成立し、新規メカニズムは境界マーカー（予約接頭辞）と SSoT ヘルパ（`split_marked` / `compose_segments`）に
  限定される。金銭コスト・レイテンシ増なし。
- + rollout 実体と `OptimizeResult` の合成が同一 SSoT ヘルパを共用するため合成規則の drift が構造的に不可能。
- + `Slot.segments` / `Slot.vars_fn` は default 付き追加で `__all__` 不変・既存呼び出し / テスト無修正
  （NFR-3）。複数 tune は 1 候補テキストとして単一 APO ループで最適化するため NFR-4（ループ回数非増加）を満たす。
- - 境界マーカーという予約接頭辞（`oas_boundary_`）を利用者の seed 本文・固定セグメント本文・dict vars キーで
  使用禁止にする制約が増える（slot 構築時に fail-closed で通知）。予約領域を明示することで実害経路を塞ぐ
  意図的なトレードオフ。
- - `_adapters/lightning.py` に post-fit フォールバック判定の exact-once 拡張という 1 点の変更が入る
  （合成経路 `_compose_full` / `run_apo` 自体は無変更・新 shape は `fixed=""` で素通し）。

## Confirmation

- tune セレクタの照合・fail-closed・複数セグメント連結の単一 APO ループ・境界マーカー保全・full 合成
  （マーカー非出現）・後方互換の強制手段: `tests/runtime/lightning/`（`test_slots_l1` が tune 照合 / layout /
  fail-closed / マーカー連結を、`test_placeholders_l1` が `split_marked` / `compose_segments` の exact-once と
  再インターリーブを、`test_optimizer_l2` が full 再合成 / diff 再計算 / post-fit フォールバックを、NFR-4 の
  Trainer 呼び出し回数一致テストがループ回数非増加を検証）。
- 後方互換（`__all__` メンバ集合不変・旧 shape 無修正通過）は既存の公開 API テストが無修正で通過することで担保する。
