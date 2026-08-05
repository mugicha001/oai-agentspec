"""オブザーバビリティ連携の公開窓口（`oai-agentspec[observability]` extra・agents 非依存）。

設定 dataclass（`Agent365TracingConfig` / `OtelLoggingConfig`）と有効化関数
（`enable_agent365_tracing` / `enable_otel_logging`）を再エクスポートする。設定型は外部依存を
持たない plain dataclass で、有効化関数の重い依存（`microsoft_agents_a365` / `opentelemetry`）は
`_adapters/observability.py` の関数内遅延 import に閉じる。よって
`from oai_agentspec.runtime.observability import enable_otel_logging` は extra 未導入でも壊れず、
実際の有効化時に初めて必要 extra を案内する（FR-7 / NFR-2）。

有効化関数はプロセス全体へ効くグローバル結線（SDK トレーシングへの計装適用 / root logger への
ハンドラ付与）を行うため、import 副作用では一切実行されず利用者の明示呼び出しでのみ作用する
（build-don't-run の例外 3 例目・ADR 0022）。コア `__init__` の `__all__` には載せない
（公開 API は本窓口に集約・FR-7）。
"""

from __future__ import annotations

from ..._adapters import enable_agent365_tracing, enable_otel_logging
from .config import Agent365TracingConfig, OtelLoggingConfig

__all__ = [
    # 有効化エントリ（グローバル結線・利用者が明示的に 1 回呼ぶ）
    "enable_agent365_tracing",
    "enable_otel_logging",
    # 設定型
    "Agent365TracingConfig",
    "OtelLoggingConfig",
]
