# 0010: APO 逐次実行の途中失敗で完了済み slot の成果を `OptimizeError.partial` に保全する

- Status: accepted
- Date: 2026-07-28

## Context

`optimize()` の複数 slot APO は agent-lightning 0.3 の単一プロンプト最適化制約により逐次実行で
ある（slot を順に APO へ通し、最良候補で次 slot の rollout コンテキストを更新する）。この構造で
slot i の最適化が失敗すると、slot 1..i-1 の最適化は既に完了している（数ラウンドの実 API コスト
を払って最良テキストが確定している）にもかかわらず、`run_apo` のローカル変数（`current` /
`history`）ごと破棄され、利用者には何も残らなかった。全 slot の最適化が成功した後の合成スコア
再計算（`_score_candidate`）で失敗した場合は、**全 slot の成果が揃っているのに全損**していた。

あわせて、`optimize()` の `run_apo` 境界 catch-all のメッセージが `f"...: {exc}"` 形式で、
`str(TimeoutError())` のように本文が空の例外では `'最適化の実行に失敗しました: '` とコロンで
終わる情報ゼロの文字列になっていた（ADR 0009 の pre-flight 側で修正したものと同型の穴）。

動機はユーザー要件「終了した時に、エラーで各データがなくなることを危惧している」（ADR 0009 の
案 B と同じ）。ADR 0009 は pre-flight route coverage の観測失敗に部分 `CoverageReport` を添付
する保全を導入したが、その適用範囲は pre-flight に限られ、APO 実行段には同型の穴が残っていた。

ADR 0009 は「観測ループ内で送出済み `OptimizeError` の pass-through では `coverage=None` の
ままとし、部分観測の構造化保全はスコープ外」と判断した。本 ADR は隣接スコープ（`run_apo` 段）で
pass-through 例外へ `partial` を属性後付けする拡張を行うが、これは矛盾ではない: pre-flight では
観測失敗時の部分情報（`covered` 等）が catch 位置と raise 位置で分かれており添付には情報が
揃わなかったのに対し、`run_apo` 段では partial 組み立てに要する情報（`current` / `history` /
`vars_per_slot`）がすべて catch 位置（`run_apo` ループ内・ループ後）に揃うため、kind / message
を保ったまま属性後付けだけで成立する。

## Decision

### 部分成果の器: 新公開型 `OptimizePartial` + `OptimizeError.partial`

frozen dataclass `OptimizePartial` を新設し、`OptimizeError.__init__` に keyword-only
`partial: OptimizePartial | None = None` を追加する（`coverage` と同型の非破壊拡張・
`oai_agentspec.runtime.lightning.__all__` にのみ追加しコア `__all__` には載せない既存契約）。

- `completed_slots: dict[str, str] = field(repr=False)` — 完了済み slot の最良テキスト
  （`${var}` 再注入済み）。prompt 本文の accidental dump を防ぐため repr 抑止
  （`CoverageReport.per_case` と同方針・明示アクセスは可）
- `history: list[HistoryEntry]` — 完了済み slot の履歴。stdlib に frozen mapping が無く
  `OptimizeResult.history` と型を揃えるため list（`CoverageReport` の frozenset / tuple とは
  前例が分かれるが、schema 一致を優先）
- `failed_slot: str | None` — 失敗 slot 名。None は「全 slot 完了・スコア再計算段の失敗」

「非 None = 保全された成果がある」を契約とし、先頭 slot 失敗等の保全対象がない失敗では
`partial=None` のまま（空 partial は作らない）。

検討した却下案:

- **`coverage` 属性への相乗り / 統一属性化**: `CoverageReport` は pre-flight 観測の型で意味が
  別。統合は既存公開契約の破壊。
- **`OptimizeResult` の流用（`partial_result: OptimizeResult`）**: `train_score: float` が
  算出不能（ダミー値 0.0 は「スコア 0」と区別不能）。`prompt` の「合成済み full」契約も
  満たせず型の意味が汚染される。
- **`logger.warning` + `__cause__` のみ**: ログは設定次第で消え、文字列であって except 節から
  プログラム的に取得できない（ADR 0009 で却下済みと同理由）。
- **`failed_at`（index）フィールド**: slot 位置はメッセージ本文（`slot i/N`）にあり、
  `failed_slot` 名で一意に特定できる。
- **X2 で成功済み `train_score` を載せる**: 「train 成功後 val 失敗」時のみ部分的に存在する
  条件依存フィールドになり、`missing` 二義性問題（ADR 0009）の再生産。
- **内部 sentinel 例外（run_apo → optimizer 翻訳）**: 表現レベルを「vars 再注入まで」に
  したことで optimizer の再合成知識が不要になり、sentinel は純粋な間接化（ADR 0009 の
  却下理由と同根）。

### catch 位置: `run_apo` 内 2 箇所の発生源 raise

- **X1（slot ループ内）**: `_run_apo_single_slot` を try で包む。失敗時、完了済み slot から
  partial を組み立て `OptimizeError(TRAINER_FAILED, "slot {i}/{N} {name!r} の APO 実行に
  失敗しました: ...", partial=...)` を送出する。
- **X2（ループ後）**: `_score_candidate` 2 呼び出しと `${var}` 再注入・diff 算出を try で包む。
  失敗時は全 slot を `completed_slots` に載せ `failed_slot=None` で送出する。
- **既に `OptimizeError` の失敗**（NFR-8 fail-closed sentinel の `critical_error` 等）は
  kind / message / coverage を保つため再ラップせず、`exc.partial` が None のときのみ属性
  後付けして re-raise する。
- **fail-safe**: partial 組み立て（`_build_partial`）自体を try で包み、失敗時は
  `partial=None` で元例外を優先する。`substitute_braced` は vars 値へ `str()` を適用する
  ため、`__str__` が例外を投げる利用者オブジェクトで組み立てが失敗しうる（実測確認済み）。
  診断のための処理を新たな失敗源にしない。

### 部分成果の表現レベル: `${var}` 再注入まで・full 再合成なし

再注入は run_apo 正常経路と同一規則（`substitute_braced`・braced のみ）。新 shape slot の
固定セグメント込み full 再合成（`compose_from_marked`）は optimizer 側の `Slot.segments`
知識が必要で、それ自体が失敗しうる。成果保全の目的に対し過剰であり行わない。帰結として
`completed_slots` は**診断・救出用の中間表現**（新 shape では tune 側テキストのまま）で、
そのまま instructions に使う値ではない（正本: `OptimizePartial` の型 docstring）。

### メッセージ整形の共有ヘルパ `_format_exception_message`

「型名は常に・本文は非空のときだけ連結」の整形規則を `types.py` の private 純関数へ集約し、
3 箇所（optimizer の catch-all / `_rollout` の pre-flight 観測 / `_adapters` の X1・X2）で
共有する（複製すると空本文でコロン終わりになるバグが drift で再発するため）。optimizer の
catch-all は run_apo が X1/X2 を構造化した後も死枝にならない（`_require_agentlightning` の
非 ImportError 素通り・ループ外の準備段階失敗が残る）ことをテストで直接 pin する。

## Consequences

- + 複数 slot APO の途中失敗・スコア再計算失敗で、支払い済み API コストの成果（最良テキスト・
  履歴）が `error.partial` から取得できる。
- + 失敗メッセージが slot 位置（`slot i/N '名前'`）・例外型名を常に含み、本文空の例外でも
  情報が残る。
- - `completed_slots` は中間表現であり、新 shape slot ではそのまま使えない（docstring と
  usage docs に明記して緩和）。
- - `OptimizeError` の診断属性が `coverage` / `partial` の 2 系統になる（前者 = pre-flight・
  後者 = APO 実行段。適用経路が重ならないため同時に非 None にはならない）。
- 公開契約: `OptimizePartial` の `runtime.lightning.__all__` 追加と `OptimizeError.partial`
  は非破壊拡張（keyword-only・既定 None）。SemVer minor 相当。

## Confirmation

- 強制手段: `tests/_adapters/test_lightning_adapters_l2.py::test_run_apo_*`（slot 途中失敗で
  完了済み slot の保全 + vars 再注入 + メッセージの slot 位置・型名・本文
  （`test_run_apo_slot_failure_preserves_completed_slots`）/ 先頭 slot 失敗は `partial=None`
  かつ空本文例外で型名が残る（`test_run_apo_first_slot_failure_has_no_partial`）/
  `OptimizeError` の kind / message を保ったまま partial 後付け
  （`test_run_apo_slot_failure_optimize_error_passes_through_with_partial`）/ スコア再計算
  失敗で全 slot 保全・`failed_slot=None`（`test_run_apo_score_failure_preserves_all_slots`）/
  組み立て失敗の fail-safe（`test_run_apo_partial_build_failure_falls_back_to_none`））。
  型契約は `tests/runtime/lightning/test_types_l1.py::test_optimize_partial_*` /
  `::test_optimize_error_partial_*` / `::test_format_exception_message_*`。optimizer 境界の
  catch-all 生存と型名整形は
  `tests/runtime/lightning/test_optimizer_l2.py::test_optimize_run_apo_boundary_message_carries_type_name`。
- pin 感度は変異注入で実証済み（後付け削除 / `failed_slot` 固定 / fail-safe 除去の 3 種で
  対応テストが RED になることを確認・注入は復元済み）。
- retire 条件: agent-lightning が複数プロンプト同時最適化を提供し逐次実行を廃止した場合、
  X1 の保全は再設計する（X2 と型契約は残る）。

## 関連判断: gradient / apply-edit の API 選択（Responses 優先 + chat fallback）

APO の textual gradient / apply-edit は Azure の Responses-only deployment（gpt-5 系）対応の
ため Responses API を優先する。`/responses` エンドポイント不在（404）の chat-only ゲートウェイ
（litellm 等）では、上流 agent-lightning 本来の `chat.completions` へ自動 fallback する
（「client を渡せばそのまま動く」を公開ノブなしで維持する）。

- fallback 済みモデルの記憶は **APO インスタンス属性の `set[str]`**（`_build_apo` が持たせ、
  bound override が渡す）。寿命が 1 APO インスタンス（= 1 slot の実行）に閉じるため、
  (a) 一過性の誤分類 404 による chat 固定が最長でも 1 slot 実行で消える、(b) module-global
  記憶に必要だった id 再利用ガード・GC 追従・unhashable 対策（約 65 行の不変条件群）が
  構造ごと不要になる。記憶は fallback 成功後のみ（一過性の失敗で固定化しない）。
- モデル / デプロイ不在を示す 404（`model_not_found` / `DeploymentNotFound`）は fallback せず
  伝搬する（モデル名のタイポが chat 側の別エラーに化けて真因が隠れるため）。
- `responses` 属性を持たない最小構成 client は 404 を待たず最初から chat を使う。
- slot ループ内の `ImportError`（`[apo]` 系サブ依存（poml 等）の欠落、または rollout 内の
  利用者コード import 失敗）は、X1/X2 共有ヘルパ `_raise_run_apo_failure` が
  `OptimizeError(EXTRA_MISSING, ..., partial=partial)` に包んで送出する（kind 契約と
  partial 保全の両立。生 raise は完了済み slot の成果を捨てる欠陥だった）。X1/X2 の
  振り分けポリシーは同ヘルパの単一定義（重複 2 箇所の乖離リスク排除）。
- chat fallback の応答 `content` が content-parts 形式（list）の場合は text 連結で str へ
  強制する（`-> str` 契約の維持）。
- `apo_api` の受理値は `config.py` の共有定数（`APO_API_VALUES` 等）を optimizer の検証と
  `_build_apo` のディスパッチが共に参照する（2 層のリテラル drift 防止）。

**訂正（同一 PR 内・未マージのため本節を直接更新）**: 当初「設定フィールドによる明示選択」を
却下したが、エコシステムの主流が明示設定（openai-agents SDK 自身がモデルクラスを明示選択させる）
であり、rollout 側（`OPENAI_API_STYLE`）と gradient 側の対称性を欠くことから、
**`OptimizeConfig.apo_api`（None = auto / "responses" / "chat_completions"・既定 None）を追加し、
404 自動 fallback は「未指定時の安全網」へ格下げ**する。

- `"chat_completions"` は override を bind しない（上流 agent-lightning 本来の chat 実装。
  override 2 本は必ず同時に bind / 非 bind する）。
- `"responses"` は fallback を無効化する（明示したのに黙って chat へ化けない fail-closed。
  属性ショートカット・(client, model) 記憶の読みも skip し、`responses` 属性を持たない client は
  optimize() が pre-flight 前に `CONFIG_MISSING` で fail-fast する）。
- 配線は `_build_apo` がインスタンス属性 `_oas_allow_chat_fallback = (apo_api is None)` を set し、
  bound override が `getattr(self, ..., True)` で読む（既定 True により属性未設定でも auto 挙動）。

その他の却下案: module-global の記憶（`WeakSet` / 素の `id()` キー / weakref + 実体一致） —
それぞれ unhashable client での membership 落ち・id 再利用の誤爆・約 65 行の防御的機構、と
段階的に欠陥と複雑さを生んだ（3 実装の失敗を経てインスタンス属性方式へ収束）。
`Literal` 型による静的検証 — mypy 非導入のため実行時検証（`algorithm` の前例）に揃える。
ImportError の生 raise — kind 契約は保てるが partial 喪失と原因二面性の不説明が残る。

強制手段: `tests/_adapters/test_lightning_adapters_l2.py::test_responses_complete_text_*` /
`::test_responses_fallback_memo_*`・`::test_responses_fallback_has_no_module_global_cache` /
`::test_chat_complete_text_*`（content-parts 強制含む） /
`::test_run_apo_slot_import_error_maps_to_extra_missing_with_partial`・
`::test_run_apo_first_slot_import_error_maps_to_extra_missing_without_partial` /
`::test_build_apo_*`（bind 分岐と
fallback フラグ）/ `::test_responses_complete_text_strict_mode_*`（明示 responses の fail-closed）、
`tests/runtime/lightning/test_optimizer_l2.py::test_optimize_apo_api_*`（受理値検証・事前
不整合検出・mutex）、`tests/runtime/lightning/test_config_l1.py::test_optimize_config_apo_api_*`。
