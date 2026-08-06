"""オブザーバビリティ連携窓口（Agent 365 拡張 / OpenTelemetry Logs を `_adapters` に閉じる）。

`microsoft_agents_a365` / `opentelemetry` / `agents` の import を本モジュールの関数内遅延 import に
閉じる（NFR-1）。提供するのは次の 2 つの薄い結線のみで、lib 独自の実行ループ・`Runner` 参照は
持たない（build-don't-run の例外 3 例目・ADR 0022）。

- `enable_agent365_tracing`: SDK トレーシングへ Agent 365 拡張のプロセッサを結線する。
  エクスポート先（既定コンソール / sidecar / 実 Agent 365 API / 汎用 OTLP）の選択は Agent 365 の
  構成関数へパススルー委譲し、lib 独自の選択方式は新設しない。
- `enable_otel_logging`: 標準 `logging` の root logger へ OTel `LoggingHandler` を冪等に付与する。
  既定はコンソール出力のみで、OTLP は置換せず**併用追加**する。

いずれもプロセス全体へ効くグローバル結線のため、import 副作用としては一切実行せず、利用者が
明示的に呼んだときだけ作用する（本モジュールの import は root logger に触れない）。
"""

from __future__ import annotations

import logging
import threading
import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..runtime.observability.config import Agent365TracingConfig, OtelLoggingConfig

# observability extra（Agent 365 拡張 / opentelemetry-sdk）未導入時の案内。
_OBSERVABILITY_INSTALL_HINT = (
    "オブザーバビリティ連携には microsoft-agents-a365-observability-extensions-openai と "
    "opentelemetry-sdk が必要です。"
    "次でインストールしてください: pip install 'oai-agentspec[observability]'"
)

# OTLP 併用（`otlp_enabled=True`）時のみ必要になる別配布物の未導入案内。
_OTLP_EXPORTER_INSTALL_HINT = (
    "ログの OTLP 併用には opentelemetry-exporter-otlp-proto-http が必要です。"
    "次でインストールしてください: pip install 'oai-agentspec[observability]'"
)

# `exporter_options.token_resolver` の「属性が無い」と「属性はあるが None」を区別する番兵。
# 前者は sidecar 構成 / options 未指定で警告不要、後者は認証手段が失われる構成で警告対象。
# `None` を既定値にすると両者が同じ値へ潰れて sidecar 構成を誤検知するため専用の番兵を使う。
_MISSING = object()

# root logger への `LoggingHandler` 重複付与を防ぐフラグ（プロセス内 1 回だけ付与する）。
_OTEL_LOG_HANDLER_ATTACHED = False

# 付与時に適用した設定（`(service_name, level, otlp_enabled, console_json_lines)`）。2 回目以降に
# 異なる設定で呼ばれたことを検知して警告するためだけに保持する（適用済み設定は変更しない）。
_OTEL_LOG_APPLIED_SETTINGS: tuple[str | None, int, bool, bool] | None = None

# 上記 2 つの判定〜更新を不可分にするロック（同時呼び出しでの二重付与を防ぐ）。
_OTEL_LOG_LOCK = threading.Lock()

# `enable_otel_logging(config=None)` で用いる既定値。値の Single Source of Truth は
# `runtime/observability/config.OtelLoggingConfig` のフィールド既定値で、ここはコア層から
# `runtime` を実行時 import しない（単方向依存）ための写しである。整合は
# `tests/_adapters/test_observability_l1.py` の既定値 pin が守る。
_DEFAULT_LOG_SERVICE_NAME: str | None = None
_DEFAULT_LOG_LEVEL = logging.INFO
_DEFAULT_LOG_OTLP_ENABLED = False
_DEFAULT_LOG_CONSOLE_JSON_LINES = False


def _require_agent365_tracing() -> tuple[Any, Any, Any]:
    """Agent 365 のトレース連携シンボルを遅延 import する（未導入時は案内付き ImportError）。

    Returns:
        `(configure, is_configured, OpenAIAgentsTraceInstrumentor)` の 3 つ組。

    Raises:
        ImportError: observability extra が未導入の場合（案内文字列付き）。
    """
    try:
        from microsoft_agents_a365.observability.core import configure, is_configured
        from microsoft_agents_a365.observability.extensions.openai import (
            OpenAIAgentsTraceInstrumentor,
        )
    except ImportError as exc:
        raise ImportError(_OBSERVABILITY_INSTALL_HINT) from exc
    return configure, is_configured, OpenAIAgentsTraceInstrumentor


def _require_opentelemetry() -> SimpleNamespace:
    """OpenTelemetry Logs の構築シンボルを遅延 import する（未導入時は案内付き ImportError）。

    `ConsoleLogExporter` は後継の `ConsoleLogRecordExporter` へ改名され、旧名は将来削除予定の
    非推奨エイリアス（構築時に `DeprecationWarning` を出す）になっている。新名称があればそちらを
    使い、無い版では旧名へフォールバックする（extra の下限を上げずに警告と将来の破壊を避ける）。
    旧名が削除された版でも解決できるよう、フォールバック先は新名称の不在を確認してから参照する
    （既定値を先行評価する形では旧名削除時に `AttributeError` になる）。

    Returns:
        `LoggerProvider` / `LoggingHandler` / `ConsoleLogExporter` / `LogRecordProcessor`
        （`BatchLogRecordProcessor`）/ `Resource` を属性に持つ名前空間。OTLP エクスポータは
        別配布物のため含めない（`_require_otlp_log_exporter` が担う）。

    Raises:
        ImportError: observability extra が未導入の場合（案内文字列付き）。
    """
    try:
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs import export as log_export
        from opentelemetry.sdk.resources import Resource
    except ImportError as exc:
        raise ImportError(_OBSERVABILITY_INSTALL_HINT) from exc
    # 既定値を先行評価させないため 2 段で解決する（`getattr(..., 旧名)` 形は旧名が削除された
    # 将来版で新名称の有無に関わらず `AttributeError` になり、フォールバックが成立しない）。
    console_exporter = getattr(log_export, "ConsoleLogRecordExporter", None)
    if console_exporter is None:  # 新名称が無い旧 SDK 版のみ
        console_exporter = log_export.ConsoleLogExporter
    return SimpleNamespace(
        LoggerProvider=LoggerProvider,
        LoggingHandler=LoggingHandler,
        ConsoleLogExporter=console_exporter,
        LogRecordProcessor=log_export.BatchLogRecordProcessor,
        Resource=Resource,
    )


def _require_otlp_log_exporter() -> Any:
    """OTLP のログエクスポータを遅延 import する（未導入時は案内付き ImportError）。

    `opentelemetry-sdk` とは別配布物（`opentelemetry-exporter-otlp-proto-http`）のため
    `_require_opentelemetry` から分離し、OTLP 併用を要求されたときにだけ解決する。既定の
    コンソール出力しか使わない利用者は本配布物に依存しない。

    Returns:
        `OTLPLogExporter` クラス。

    Raises:
        ImportError: 当該配布物が未導入の場合（案内文字列付き）。
    """
    try:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    except ImportError as exc:
        raise ImportError(_OTLP_EXPORTER_INSTALL_HINT) from exc
    return OTLPLogExporter


def _json_lines_formatter(record: Any) -> str:
    """ログレコードを 1 行の JSON（JSON Lines）へ整形する。

    SDK 既定の整形（`indent=4`）は 1 レコードが複数行に分かれるため、標準出力を「1 行 =
    1 レコード」で取り込むログ収集基盤では 1 レコードが分割されてしまう。改行を末尾の 1 つだけに
    することでそのまま 1 レコードとして扱えるようにする。

    Args:
        record: OpenTelemetry のログレコード（`to_json` を持つ不透明値）。

    Returns:
        末尾に改行を 1 つだけ持つ 1 行の JSON 文字列。
    """
    return record.to_json(indent=None) + "\n"


def _tracing_disabled() -> bool:
    """SDK トレーシングが `set_tracing_disabled(True)` で無効化されているかを判定する。

    `agents` は無効状態を返す公開 API を持たないため、トレースプロバイダの内部フラグを
    防御的に参照する（属性が失われた将来版では「無効ではない」と見なして警告を出さない）。

    限界: 内部フラグは `set_tracing_disabled()` の呼び出しか最初のスパン生成時にしか環境変数
    （`OPENAI_AGENTS_DISABLE_TRACING`）から更新されない。有効化 API は通常プロセス起動直後に
    呼ばれるため、**環境変数による無効化はここでは検知できない**（False を返す）。

    Returns:
        トレーシングが無効化されている場合 True。
    """
    from agents.tracing import get_trace_provider

    return bool(getattr(get_trace_provider(), "_disabled", False))


def enable_agent365_tracing(config: Agent365TracingConfig) -> None:
    """Agent 365 オブザーバビリティへのトレース連携を有効化する。

    Agent 365 を構成してから SDK トレーシングへ計装を適用する（構成前に計装を生成すると
    Agent 365 側が `RuntimeError` を送出するため、この順序が契約）。計装は SDK のトレース
    プロセッサ列を Agent 365 のプロセッサへ差し替えるため、SDK 既定のエクスポート先
    （OpenAI プラットフォーム）への送信は行われなくなる。

    冪等化は Agent 365 側へ委譲する（再構成は警告付きで無視され、再計装も no-op になる）ため、
    本関数は独自の状態を持たない。

    構成に失敗した場合は `RuntimeWarning` で通知して計装を行わずに戻る（観測の失敗で利用者の
    アプリを停止させない）。計装しないため SDK 既定のトレースプロセッサ列はそのまま残り、
    既定のエクスポート先への送信は継続する。Agent 365 の計装経路にのみ効く本文抑止
    （`suppress_invoke_agent_input`）や span enricher はこの経路には適用されない。

    `token_resolver` を渡していても、`token_resolver` 属性を持つ `exporter_options`（Agent365
    形式）を併用した場合は Agent 365 側が options の値のみを使うため、渡した `token_resolver` は
    参照されない（上流の合成規則）。**`exporter_options` 側にも `token_resolver` が設定されて
    いない場合**（実効の認証手段がどこにも残らない構成）に限り `RuntimeWarning` で通知するが、
    構成失敗時とは異なり処理は継続する（計装まで到達する）。options 側に設定済みの場合はそちらが
    使われるため通知しない。検知範囲には限界があり、環境変数による有効化フラグ未設定に起因する
    未達は検知しない（本ライブラリは環境変数を読まないため）。詳細は ADR 0024 を参照する。

    Args:
        config: トレース連携の宣言的設定（接続先・認証手段は本設定経由でのみ受領する）。

    Raises:
        ImportError: observability extra が未導入の場合（案内文字列付き）。
    """
    configure, is_configured, instrumentor_cls = _require_agent365_tracing()

    if _tracing_disabled():
        warnings.warn(
            "SDK トレーシングが無効化されているため Agent 365 へトレースは送信されません。"
            "`agents.set_tracing_disabled(False)` で有効化してください。",
            RuntimeWarning,
            stacklevel=2,
        )

    # 上流は exporter_options が渡された場合その値のみを使い、トップレベル token_resolver を
    # 参照しない（合成規則は上流 core/config.py の `exporter_options is None` 分岐）。Agent365 形式
    # options かの判別に isinstance を使わないのは、_adapters へ上流型を持ち込まないため。
    # SpectraExporterOptions は当該属性を持たないので、属性の不在（sentinel）で sidecar 構成を
    # 除外できる（exporter_options 未指定 = None も属性を持たないため同じ経路で除外される）。
    # 属性の読み取りは 1 回だけ行い、あらゆる例外を吸収する: 利用者が渡す不透明値は property や
    # `__getattr__` を持ちうるため、`hasattr` では `AttributeError` 以外が呼び出し元へ伝播して
    # 観測の構成ミスがアプリを停止させてしまう（ベストエフォート方針に反する）。
    # 断定形を避けるのは、2 回目以降の configure() が渡した設定を丸ごと無視して真を返すため
    # （上流 core/config.py の再構成ガード）。その経路では実際の exporter は初回構成のものになる。
    if config.token_resolver is not None:
        try:
            options_resolver: Any = getattr(config.exporter_options, "token_resolver", _MISSING)
        except Exception:  # noqa: BLE001 - 観測の判定失敗で利用者のアプリを停止させない
            options_resolver = _MISSING
    else:
        options_resolver = _MISSING
    if options_resolver is None:
        warnings.warn(
            "exporter_options を指定したため token_resolver は Agent 365 側で参照されません"
            "（exporter_options が渡された場合、Agent 365 はその値のみを使います）。"
            "この設定が適用される場合、Agent 365 実サービスへは送信されずコンソール出力へ"
            "フォールバックします（configure() は成功を返すため戻り値では検知できません）。"
            "実 Agent 365 API へ送る場合は exporter_options の token_resolver を設定してください。",
            RuntimeWarning,
            stacklevel=2,
        )

    kwargs: dict[str, Any] = {
        "service_name": config.service_name,
        "service_namespace": config.service_namespace,
        "token_resolver": config.token_resolver,
        "cluster_category": config.cluster_category,
        "exporter_options": config.exporter_options,
        "suppress_invoke_agent_input": config.suppress_invoke_agent_input,
    }
    # None は「委譲先の既定に委ねる」意味なので引数自体を渡さない（既定値の上書きを避ける）。
    if config.logger_name is not None:
        kwargs["logger_name"] = config.logger_name

    configure_ok = configure(**kwargs)
    configured = is_configured()
    if not configure_ok or not configured:
        # 構成失敗は送信不能を意味するが、エージェント実行自体は妨げない（ベストエフォート）。
        warnings.warn(
            "Agent 365 の構成に失敗したため Agent 365 へトレースは送信されません。"
            "service_name / service_namespace と接続設定を確認してください。"
            "計装しないため SDK 既定の送信先への送信は継続します"
            "（Agent 365 の計装経路にのみ効く本文抑止・span enricher は適用されません）。",
            RuntimeWarning,
            stacklevel=2,
        )
        # 計装は行わない。構成失敗のまま計装すると `set_trace_processors` が SDK 既定の
        # トレースプロセッサ列を置換して既定の送信先まで失われ（`_uninstrument` は既定列を
        # 復元しない）、未構成のまま生成すると Agent 365 側が RuntimeError を送出する。
        return

    instrumentor_cls().instrument()


def enable_otel_logging(config: OtelLoggingConfig | None = None) -> None:
    """標準 `logging` のログを OpenTelemetry Logs として送出する結線を有効化する。

    root logger へ `LoggingHandler` を 1 つ追加する。既存のハンドラ・フォーマッタ・登録順および
    root logger のレベルは変更しない（追加のみ）。したがって root のレベルが既定のままだと
    `config.level` より詳細なログはハンドラへ届かないため、必要に応じて利用者側で調整する。

    宛先は既定でコンソールのみで、`config.otlp_enabled` が真のときはコンソールを置換せず OTLP
    エクスポータを併用追加する。プロセス内で複数回呼ばれても付与は 1 度だけ行う（冪等）。2 回目
    以降に初回と異なる設定を渡した場合は適用されないため `RuntimeWarning` で通知する（より
    制限的な設定へ変えたつもりが効いていない、という誤解を避けるため）。

    Args:
        config: ログ連携の宣言的設定。None なら既定値（コンソールのみ・INFO）で結線する。

    Raises:
        ImportError: observability extra が未導入の場合（案内文字列付き）。
    """
    global _OTEL_LOG_HANDLER_ATTACHED, _OTEL_LOG_APPLIED_SETTINGS  # noqa: PLW0603 - 1 回 setup
    if config is None:
        service_name = _DEFAULT_LOG_SERVICE_NAME
        level = _DEFAULT_LOG_LEVEL
        otlp_enabled = _DEFAULT_LOG_OTLP_ENABLED
        console_json_lines = _DEFAULT_LOG_CONSOLE_JSON_LINES
    else:
        service_name = config.service_name
        level = config.level
        otlp_enabled = config.otlp_enabled
        console_json_lines = config.console_json_lines
    settings = (service_name, level, otlp_enabled, console_json_lines)

    # 判定〜付与〜状態更新を不可分にする（同時呼び出しでの二重付与を防ぐ）。
    with _OTEL_LOG_LOCK:
        if _OTEL_LOG_HANDLER_ATTACHED:
            if settings != _OTEL_LOG_APPLIED_SETTINGS:
                warnings.warn(
                    "ログ連携は既に有効化済みのため、今回渡した設定は適用されません。"
                    "設定を変える場合はプロセスの起動時に 1 度だけ有効化してください。",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return

        otel = _require_opentelemetry()
        otlp_exporter_cls = _require_otlp_log_exporter() if otlp_enabled else None

        provider_kwargs: dict[str, Any] = {}
        if service_name is not None:
            provider_kwargs["resource"] = otel.Resource.create({"service.name": service_name})
        provider = otel.LoggerProvider(**provider_kwargs)

        # 1 行 JSON を要求されたときだけ formatter を差し替える（既定は SDK の整形済み出力）。
        console_kwargs: dict[str, Any] = {}
        if console_json_lines:
            console_kwargs["formatter"] = _json_lines_formatter
        provider.add_log_record_processor(
            otel.LogRecordProcessor(otel.ConsoleLogExporter(**console_kwargs))
        )
        if otlp_exporter_cls is not None:
            # 置換ではなく追加（コンソール確認を維持したまま外部バックエンドへも送る）。
            provider.add_log_record_processor(otel.LogRecordProcessor(otlp_exporter_cls()))

        logging.getLogger().addHandler(otel.LoggingHandler(level=level, logger_provider=provider))
        _OTEL_LOG_HANDLER_ATTACHED = True
        _OTEL_LOG_APPLIED_SETTINGS = settings
