# 0020: 遅延窓口の不変条件を「実装実体モジュールの非ロード」と定め resilience 窓口で成立させる

- Status: accepted
- Date: 2026-08-04

## Context

ADR-0003 の Confirmation は hooks 窓口の `_l1` テストを「窓口 import 時 `agents` 非発火検証」と
記述しているが、この表現は成立しない。`agents` はコア依存であり、
`oai_agentspec/__init__.py` -> `_adapters/__init__.py` のトップレベル import 連鎖で、どの窓口の
import よりも前に必ずロード済みになる。窓口の PEP 562 遅延が守れるのは「実装実体モジュール
（`_adapters.hooks` / `_adapters.resilience` 等）を属性アクセスまでロードしない」ことであって、
`agents` の非発火ではない。実際に `tests/runtime/hooks/test_init_pep562_l1.py` の probe が検査
しているのも `_adapters.hooks` の非ロードである。

さらに `runtime/resilience` 窓口は module docstring / `__getattr__` docstring で同じ誤表現
（「窓口 import 自体は `agents` を発火させない」）を主張していたうえ、実測では遅延そのものが
効いていなかった。原因は `_adapters/__init__.py` がトップレベルで
`from .resilience import build_model_retry, build_run_budget_hooks` を行っていたことで、コア
import 連鎖により `_adapters.resilience` が常時ロードされ、窓口の `__getattr__` 遅延は対象が
ロード済みの空振りになっていた。この再エクスポートを `_adapters` パッケージ属性経由で参照する
消費者は src / tests に存在しない（唯一の利用経路は窓口 `__getattr__` からのサブモジュール直接
import）。

検討した選択肢:

1. **記述のみ訂正し「resilience の遅延は効いていない」事実を明記（却下）**: 窓口の
   `__getattr__` 遅延機構が「無意味だが存在する」状態を仕様として文書化することになり、
   hooks 窓口（遅延成立）との非対称が恒久化する。機構を残すなら成立させるべき。
2. **`agents` をコア依存から外して遅延化し、「`agents` 非発火」を成立させる（却下）**:
   公開 API の import 構造の変更であり影響が広い。別途の設計判断が必要（本 ADR のスコープ外）。
3. **不変条件を「実装実体モジュールの非ロード」と定義し直し、`_adapters/__init__.py` の
   トップレベル再エクスポートを撤去して resilience 窓口でも成立させる（採用）**: 消費者ゼロが
   確認済みの 1 import + `__all__` 2 エントリの削除で、hooks 窓口と同じ不変条件に統一できる。

## Decision

- 遅延窓口が守る不変条件を「窓口 import 時点で実装実体モジュール（`_adapters.<name>`）を
  ロードしない」と定める。「窓口 import 時に `agents` を発火させない」は不変条件ではない
  （`agents` はコア依存で窓口 import より前にロード済み）。ADR-0003 の Confirmation にある
  「`agents` 非発火検証」はこの誤表現であり、本 ADR で訂正する。ADR-0003 本文は append-only の
  ため書き換えず、Status に訂正参照を追記する。
- `_adapters/__init__.py` のトップレベル `from .resilience import ...` と `__all__` の
  `build_model_retry` / `build_run_budget_hooks` 2 エントリを撤去し、`_adapters.resilience` の
  ロード経路を resilience 窓口の `__getattr__` からのサブモジュール import のみに閉じる。
  これにより resilience 窓口の遅延が実際に成立する。

## Consequences

- + resilience 窓口の PEP 562 遅延が実挙動として成立し、docstring / `docs/architecture.md` の
  主張と一致する。hooks 窓口と同型の不変条件に統一され、probe テストの書き方も揃う。
- + 不変条件の定義が「検査可能な形」（`sys.modules` の非ロード）に固定され、以後の遅延窓口の
  docstring / テストが同じ表現を使える。
- - `build_model_retry` / `build_run_budget_hooks` は `_adapters` パッケージ属性としては参照
  できなくなる（`_adapters` は内部窓口で公開契約外・消費者ゼロを確認済み。必要時は
  `_adapters.resilience` サブモジュールから直接 import する）。

## Confirmation

- resilience 窓口の不変条件の強制手段:
  `tests/runtime/resilience/test_init_pep562_l1.py::test_importing_window_does_not_load_adapter_module` /
  `::test_dir_call_does_not_load_adapter_module`（クリーン subprocess での `sys.modules` probe）。
  `docs/QUALITY-GUARANTEES.md` に登録済み（source = ADR-0020）。
- hooks 窓口の同型不変条件の強制手段: `tests/runtime/hooks/test_init_pep562_l1.py` の同名 probe。
