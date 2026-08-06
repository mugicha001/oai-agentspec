"""オブザーバビリティ連携の設定 plain dataclass（`Agent365TracingConfig` / `OtelLoggingConfig`）。

いずれも外部 SDK 非依存の frozen dataclass。有効化（`enable_agent365_tracing` /
`enable_otel_logging`）はグローバル結線を伴うため `_adapters/observability.py` に分離し、
本モジュールは宣言のみを担う（`microsoft_agents_a365` / `opentelemetry` を import しない）。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..._validation import validate_bool


@dataclass(frozen=True)
class Agent365TracingConfig:
    """Agent 365 トレース連携（MS 拡張 `configure()`）へのパススルー設定。

    エクスポータ選択の列挙型は持たず、切替仕様の SoT は MS 拡張側に置く（ADR 0022）。

    Attributes:
        service_name: MS 拡張 `configure()` の必須引数。
        service_namespace: MS 拡張 `configure()` の必須引数。
        logger_name: 内部ログ出力に使う logger 名。None で MS 拡張既定に委ねる。
        token_resolver: 実 Agent365 API 向けの `(scope, tenant) -> token | None` callable。
            None でトークン解決を行わない。`exporter_options` を渡した場合、MS 拡張は options の
            値のみを使うため本フィールドは参照されない（`exporter_options` 側に設定する）。
        cluster_category: MS 拡張のクラスタ区分（既定 "prod"）。
        exporter_options: sidecar 向け `SpectraExporterOptions` 等の不透明値。lib は素通しを原則と
            する（SDK 隔離）。例外として `token_resolver` 属性の有無と値のみを警告判定に読む
            （型判別もエクスポータ選択も行わない。ADR 0024）。
        suppress_invoke_agent_input: True で InvokeAgent スパンの `gen_ai.input.messages` 属性
            （完全一致キー）のみを送出対象から除く。**本文送出の全面的な抑止手段ではない**:
            system instructions は接頭辞付きキー（`gen_ai.input.messages.0.*`）で送出され、
            chat スパンの入出力とツールの入出力も抑止されない。機微情報を送出したくない場合は
            エクスポート先の選択そのもので制御する。
    """

    service_name: str
    service_namespace: str
    logger_name: str | None = None
    # 認証情報を束縛した callable / 接続設定が repr 経由でログへ流出しないよう表示から外す。
    token_resolver: Callable[[str, str], str | None] | None = field(default=None, repr=False)
    cluster_category: str = "prod"
    exporter_options: Any = field(default=None, repr=False)
    suppress_invoke_agent_input: bool = False

    def __post_init__(self) -> None:
        """`suppress_invoke_agent_input` が bool であることを構築時に検証する。

        Raises:
            ValueError: `suppress_invoke_agent_input` が bool でない場合。
        """
        validate_bool(self.suppress_invoke_agent_input, "suppress_invoke_agent_input")


@dataclass(frozen=True)
class OtelLoggingConfig:
    """標準 `logging` の OTel Logs 連携設定（root logger への `LoggingHandler` 付与用）。

    Attributes:
        service_name: OTel Resource の `service.name`。None で未設定。
        level: root logger に付与する `LoggingHandler` のレベル（既定 `logging.INFO`）。
        otlp_enabled: True で既定コンソール出力に加え `OTLPLogExporter` を併用追加する
            （置換ではない）。
        console_json_lines: True でコンソール出力を 1 行 JSON（JSON Lines）にする。既定の
            整形済み出力は 1 レコードが複数行に分かれるため、コンテナの標準出力を「1 行 =
            1 レコード」で取り込むログ収集基盤では 1 レコードが分割されてしまう。その構成で
            使う場合に有効化する。コンソール出力のみに影響し、OTLP 側の形式は変わらない。
    """

    service_name: str | None = None
    level: int = logging.INFO
    otlp_enabled: bool = False
    console_json_lines: bool = False

    def __post_init__(self) -> None:
        """bool フィールドが bool であることを構築時に検証する。

        Raises:
            ValueError: `otlp_enabled` / `console_json_lines` が bool でない場合。
        """
        validate_bool(self.otlp_enabled, "otlp_enabled")
        validate_bool(self.console_json_lines, "console_json_lines")
