# 0027: アクション直接起動時の次ターン開始エージェント解決をコア `next_turn.py` へ純追加する

- Status: accepted
- Date: 2026-08-11

## Context

実行可能アクションの直接起動（候補ボタンの押下 → `Runner.run(registry.get(plan.action_agent), ...)`）
では、会話のハンドオフ遷移を経ずに実行エージェントが起動する。既存の `next_turn_agent` は
「ハンドオフ遷移を観測したときに次ターン開始エージェントを上書きする」という発動条件を持つため、
遷移が観測されない直接起動では発動せず、実行後の会話が窓口エージェントへ戻らない。

この解決関数をどこへ置くかは、コア `__all__` のメンバ集合（36 件）という公開契約に触れる論点で
あるため、「新シンボルを増やさない案」を含めて比較した。

検討した選択肢:

1. **`runtime/intent/_next_turn.py` へ置き `runtime/intent` の窓口へ載せる（却下）**:
   コア `__all__` は 36 のままで済むが、ADR 0014 Decision 3 が「宣言型・解決関数・組み立てヘルパ・
   結線関数はコア直下の `next_turn.py`（`agents` 非依存）に置く。`runtime/` 配下には置かない」と
   明文で定めた配置基準を破り、同一の関数群内で配置基準が割れる。加えて発動ルールの選定は
   `next_turn.py` の module private な `_select_rule` にあり、パッケージ外から参照すると
   private 越境になる。本関数は intent 固有の型を 1 つも扱わないため、intent extra の境界に
   置く理由もない。
2. **既存 `next_turn_agent`（または `resolve_next_agent`）へ keyword-only 引数
   （例: `allow_direct_start: bool = False`）を追加して `__all__` を増やさない（却下）**:
   本論点の核心（コア `__all__` を増やさない）を満たす唯一の案だが、既定値付きでも公開関数の
   シグネチャは契約であり、「既存 36 シンボルの振る舞いを変更しない」という要件の主文に抵触する。
   `__all__` のメンバ集合を守るために既存関数のシグネチャ契約を変えるのは、守る対象を取り違えた
   交換である。実装上も 1 関数が 2 つの発動条件を持つことになり、「発動条件は … の AND」と
   言い切っている既存 docstring を条件付きへ書き換えることになる。
3. **コア `next_turn.py` へ `action_next_turn_agent` を新規公開シンボルとして純追加する（採用）**:
   コア `__all__` は 36 -> 37 になる。ADR 0014 Decision 3 が挙げる 4 根拠（`agents` 非依存 /
   `await` を持たない / `Runner` 参照を持たない / 宣言層の関心事）に本関数は 4 点とも該当し、
   扱う型も `NextTurnPolicy` / `RunResult` / `AgentRegistry` のみで intent 固有型を含まない。
   `next_turn_agent` の隣に並ぶため使い分けが 1 箇所で読める。

`__all__` のメンバ集合変更は契約変更であるため、実装着手前にユーザー合意を取ることを前提とした。

## Decision

`action_next_turn_agent` をコア直下 `next_turn.py` へ純追加し、コア `__all__` を 36 から 37 へ
増やす。

- 配置根拠は ADR 0014 Decision 3 と同一の 4 根拠（`agents` 非依存・`await` なし・`Runner` 参照
  なし・宣言層の関心事）であり、既存 5 公開シンボルの実装・挙動は変更しない（純追加）。
- 解決規則: ハンドオフ遷移を観測した場合は既存 `next_turn_agent` へ丸ごと委譲する。遷移が
  観測されない場合は `last_agent` に対応する規則束から**包括ルール**（`source is None`）を選び、
  `next_agent` があれば `registry.get(next_agent)`、それ以外は `last_agent` へフォールバックする。
- 包括ルールの選定は既存 `_select_rule` を再利用する。`registry` へ登録され得ない sentinel を
  `source` として渡すことで、第 1 ループ（`source` 一致）が必ず外れ、第 2 ループ（包括ルール）へ
  倒れる。選定規則を二重実装しない。
- `NextTurnRule.source` の検証は非空 `str` のみで識別子制約が無いため、sentinel が利用者の宣言
  `source` と衝突しないことは型ではなくテストで固定する。

現在仕様の SoT は `docs/architecture.md`（「Next-Turn Agent Override」節および
「意図予測（`runtime/intent`）」節）とし、本 ADR は判断・却下案のみを記録する。

## Consequences

- + 直接起動の実行後も、宣言済みの `NextTurnPolicy` に従って会話が窓口エージェントへ戻る。
- + 発動条件が「ハンドオフ観測あり / なし」で 2 関数に分かれ、各関数が 1 つの発動条件だけを持つ。
- + 既存 `next_turn_agent` / `resolve_next_agent` のシグネチャ・挙動・docstring は無変更である。
- - コア `__all__` のメンバ集合が変わる（36 -> 37）。公開契約の変更であり、合意と追随テストの
  更新を要する。
- - sentinel の非衝突が型で保証されないため、テストによる固定が必須になる。

## Confirmation

強制手段:

- `tests/test_next_turn.py`（既存ファイルへ追記）: `agents` 非依存の決定表として 5 分岐
  （ハンドオフあり = 既存経路へ委譲 / ハンドオフなし = 包括ルール適用 / 包括ルールが
  `next_agent` を持たない / 規則束のキー不一致 / `last_agent` なし）を pin し、あわせて
  sentinel `source` が宣言済み `source` と衝突しないことを 1 件で固定する。
- `tests/test_public_all_membership_l1.py`（**新規作成**）: コア `__all__` のメンバ集合が
  「変更前 36 件 + `action_next_turn_agent`」と完全一致することを pin する。既存
  `tests/test_public_naming_l1.py` は禁止語彙の照合のみでメンバ集合を pin していないため、
  既存ファイルへの相乗りではなく新規ファイルとする。

`docs/QUALITY-GUARANTEES.md` に登録済み（source = ADR-0027）。
