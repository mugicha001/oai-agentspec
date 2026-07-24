# 0001: Tool メタデータの持ち場所を独立した ToolRegistry へ一元化する

- Status: accepted
- Date: 2026-07-21

## Context

Tool の性質（有効/無効・承認要否・タイムアウト・失敗時エラー文言・名前/説明上書き等）を宣言する
場所が定まっておらず、Resilience 機能（復旧方針 `ResiliencePolicy` の宣言）の検討過程で「Tool
固有の性質と復旧方針の混在」が顕在化した。Agent 数は少なく Tool 数は多い運用を見込むため、多数の
Tool を散在するファイルに置いた生の Python 関数のまま登録しても、一元的に管理・照会できることが
求められた。

検討した選択肢:

1. **AgentSpec 拡張案（却下）**: `AgentSpec` に Tool メタデータのフィールドを追加する。Tool は
   複数 Agent から共有されうるため Agent 単位の宣言はメタデータの重複・不整合を生み、
   「`agents.Agent` の薄い Wrapper」という `AgentSpec` の責務も超える。
2. **Tool 定義側分散（デコレータ）案（却下）**: 各 Tool 定義箇所にデコレータでメタデータを付す。
   デコレータ適用時点で SDK ラップが確定するため中央集権的な照会・enabled 動的トグルを満たせず、
   Tool 定義ファイルが lib / SDK 依存になる（利用者は生の Python 関数を lib 非依存のまま散在
   ファイルに置きたい）。既存の `oai_agentspec.function_tool` 直接宣言は「メタデータ管理不要の
   単発 Tool を宣言層で agents 非依存に書く」用途で引き続き有効なため、削除・deprecate せず
   併存させる。
3. **ResiliencePolicy への混在案（却下）**: Resilience 機能の宣言に Tool メタデータを同居させる。
   Tool 固有の性質（宣言）と復旧方針（実行時戦略）は変更理由が異なり、混在は関心の分離に反する。
4. **冪等性メタデータ導入案（見送り）**: `ToolSpec` に `idempotent` フィールドを設ける。
   Resilience 機能のスコープ純化（Tool 系ラッパー廃止 = Agent / Runner 境界のリトライ戦略への
   限定）により冪等性の機械的な消費者が存在しなくなったため、投機的抽象を避けて設けない。将来
   必要になった場合は既定値付きフィールドの追加（非破壊・純粋追加）で再導入でき、その際の参照
   契約は「消費者 → ToolRegistry の一方向照会・未登録は安全側既定」の形とする。
5. **独立 ToolRegistry 案（採用）**: `AgentRegistry` と同列のコア公開 API として、Tool の宣言
   （生関数 + メタデータ）を一元管理する独立 Registry を新設する。

## Decision

Tool メタデータの持ち場所を、独立したコア公開 API `ToolRegistry` / `ToolSpec`
（`tool_registry.py`）へ一元化する。あわせて次の委譲原則を採る:

- **SDK ネイティブ機構への委譲原則（独自実行時機構を作らない）**: SDK に対応機構が存在する
  メタデータ（enabled → `is_enabled`、承認要否 → `needs_approval`、タイムアウト → `timeout` 系、
  失敗時エラー文言 → `failure_error_function`）は、独自の実行時機構を作らず対応する
  `function_tool()` 引数へ委譲する（build-don't-run）。
- Registry は宣言・遅延ラップ・照会・動的更新に徹し、独自の実行エンジン・実行 API を持たない。
  SDK 結合（`function_tool()` 呼び出し）は `_adapters/tools.py` に閉じる（SDK 隔離）。
- `AgentSpec` / `AgentRegistry` は変更しない純粋追加とし、opt-in の注入点も設けない。橋渡しは
  利用者コードの `AgentSpec(tools=[tool_registry.<name>])` で行う。

## Consequences

- + Tool メタデータの宣言場所が一意に定まり、多数の Tool を散在ファイルの生関数のまま一元管理・
  照会・enabled 動的トグルできる。
- + 実行時挙動（有効判定・承認・タイムアウト・エラー文言）を SDK ネイティブ機構へ委譲するため、
  独自実行制御の保守負担と SDK との二重実装が生じない。
- + 純粋追加のため、Tool Registry 未使用の既存コード・既存公開契約への影響がない。
- - Tool 宣言の経路が 2 つになる（`function_tool` 直接宣言 / `ToolRegistry` 登録）。使い分けは
  `docs/architecture.md` の「Tool Registry」節に明記して緩和する。
- - `enabled` 以外のメタデータの登録後更新は、構築済み（キャッシュ済み）SDK Tool へ反映されない
  （invalidate・再構築の機構を持たないトレードオフ。更新は照会値にのみ反映される）。

## Confirmation

- 委譲原則・SDK 隔離の強制手段: SDK 隔離 grep
  （`grep -rnE "(from agents|import agents)" src/oai_agentspec/ | grep -v _adapters` が空である
  こと）と、`tests/test_tool_registry.py`（コア層）/ `tests/_adapters/test_tools_l2.py`
  （SDK 結線）のテストスイート。
- 純粋追加の強制手段: 既存テストスイートが変更なしで全通過すること（`spec.py` / `registry.py`
  への差分なし）。
