# 0030: ツール拒否例外に構造化 payload（details）を付与する

- Status: accepted
- Date: 2026-08-14

## Context

MCP 由来ツールを含むツール統治（ADR-0025）の拒否は `_deny_tool_call` を唯一の送出点として
集約されている。監査 sink レコードは `tool:{name}` / `deny` / `details` で構造化されている
一方、送出される例外は人間可読のメッセージ文字列しか持たない。利用側は拒否ツール名を
メッセージ文言のパースでしか取得できず、lib のメッセージ書式変更で無警告に壊れる。

AGT には既に規約があり、`PolicyViolationError.from_check_result` は `details["tool_name"]` を
組み立てる。`_deny_tool_call` だけがこの規約から外れ、監査と例外の非対称が生じている。

却下した案:

| 案 | 却下理由 |
|---|---|
| (a) `PolicyViolationError.from_check_result` をそのまま使う | `from_check_result` は `PolicyCheckResult`（`category` / `matched_rule` / `scope` 等）を要求するが、`_evaluate_tool` の戻りは `str \| None` のみで渡せる実体が無い |
| (b) `PolicyCheckResult` を lib 側で組み立てて渡す | `category` / `matched_rule` を lib が捏造することになり、`_adapters` が AGT の追加型へ依存する。fail-closed 経路は評価自体が成立していないため `PolicyCheckResult` の意味論に合わない |
| (c) 例外に載せず監査 sink の購読で届ける | 例外を捕捉した箇所と sink レコードの突合は並行ツール呼び出しで対応付けが不定になり、catch ハンドラで即座にツール名が要る用途を満たさない。sink は監査永続先であって制御フローの戻り値ではない |

## Decision

`_deny_tool_call` の送出に `details={"tool_name": ..., "reason": ...}` を**キーワード引数で**
付与する（`AgentOSError.__init__` の第 2 位置引数は `error_code` のため、位置渡しは
`error_code` を静かに潰す）。

- ツール引数は `details` に載せない。例外は画面・ログ・エラーレスポンスまで運ばれうる一方、
  引数は LLM 出力そのもので機微になりうる。引数の置き場は監査 sink とする。
- キー集合は `tool_name` / `reason` に限定し、`category` 等の None 埋めをしない。値が
  永遠に None のキーは利用側に必ず偽になる分岐の罠を配る。キー不在からの将来の追加は
  `.get()` の戻りが変わるだけで後方互換である。
- 例外メッセージ文字列・監査 sink レコードの形は変更しない。
- テストの dict `==` は公開契約ではなく、payload の拡張を意図的な判断として顕在化させる
  検知手段である（公開契約は「`tool_name` / `reason` を必ず含む・`.get()` で読む」）。
- `reason` は人間可読の説明であり機械判別のキーではないため、AGT 生成文言をリテラル固定する
  テストは書かない（実 AGT を使う層では構造と `tool_name` のみを固定する）。

## Consequences

- **+** 利用側は文言パースなしで拒否ツール名を取得でき、`from_check_result` 由来の例外と
  `details.get("tool_name")` の 1 本のコードで扱える。
- **+** 監査と例外の非対称が解消する。新規メカニズムはゼロで、`AgentOSError.details`
  （既定 None の任意引数）へ値を渡すだけである。
- **-** `reason` は人間可読で機械判別に使えない。機械判別可能な識別子が要る場合は
  `_evaluate_tool` の戻り値型を `PolicyCheckResult` 化する別の意思決定が前提になる。
- **-** payload を拡張する場合は台帳行と docs の記述の同時更新が必要になる。
- **-** 本経路の例外は `check_result` が `None` のままである（実体が入るのは
  `from_check_result` 経由のみ）。`check_result` の有無で由来を判別している利用側コードは
  従来どおり動く。

## Confirmation

強制手段として次のテストを追加する（本 ADR の受理時点では未実装であり、追加をもって
本決定の遵守が機械的に検証される状態になる）。

- `tests/_adapters/test_governance_l1.py::test_deny_exception_details_carries_tool_name_and_reason`
  （dict `==` + メッセージ書式の完全一致）
- `tests/_adapters/test_governance_l1.py::test_deny_exception_details_excludes_arguments`
- `tests/_adapters/test_governance_l1.py::test_deny_exception_details_on_fail_closed_path`
- `tests/_adapters/test_governance_l1.py::test_deny_exception_details_on_govern_tool_path`
- `tests/_adapters/test_governance_l2.py::test_real_sdk_mcp_function_tool_is_evaluated_by_audit_hooks`
  （実 AGT 経路でのキー集合 / `tool_name` / `error_code`）
- `tests/_adapters/test_governance_l2.py::test_agent_hooks_replacement_drops_mcp_enforcement_not_spec_tools`
  （docs が案内する `exc.__cause__.details` の取得形が公開経路で成立することの実証）

不変条件と強制手段の対応は `docs/QUALITY-GUARANTEES.md` に登録する（source = ADR-0030）。
