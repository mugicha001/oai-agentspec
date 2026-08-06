"""L1: オブザーバビリティ設定型の純検証（外部依存なし）。

`Agent365TracingConfig`（トレース連携）と `OtelLoggingConfig`（logging -> OTel Logs 連携）の
不変性（frozen）・必須フィールド・既定値・bool フィールドの構築時型検証・任意フィールドの値保持を
pin する。設定型は宣言のみを担い、有効化（グローバル結線）は行わないため、本モジュールの import と
dataclass 構築が観測系 SDK（`opentelemetry` / `microsoft_agents_a365`）や `agents` をロードしない
ことも subprocess 隔離で固定する（NFR-1 / ADR 0022）。
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from oai_agentspec.runtime.observability.config import Agent365TracingConfig, OtelLoggingConfig

pytestmark = pytest.mark.unit

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"


def _make_tracing_config(**overrides: object) -> Agent365TracingConfig:
    """必須フィールドを埋めた `Agent365TracingConfig` を生成する（任意フィールドは上書き可）。"""
    kwargs: dict[str, object] = {"service_name": "svc", "service_namespace": "ns"}
    kwargs.update(overrides)
    return Agent365TracingConfig(**kwargs)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Agent365TracingConfig: 既定値・必須フィールド・不変性
# ----------------------------------------------------------------------


def test_agent365_tracing_config_defaults() -> None:
    """必須 2 件のみ指定した場合、任意フィールドは既定値（prod / None / False）になる。"""
    config = _make_tracing_config()
    assert config.service_name == "svc"
    assert config.service_namespace == "ns"
    assert config.logger_name is None
    assert config.token_resolver is None
    assert config.cluster_category == "prod"
    assert config.exporter_options is None
    assert config.suppress_invoke_agent_input is False


def test_agent365_tracing_config_accepts_positional_required_fields() -> None:
    """`service_name` / `service_namespace` は宣言順の位置引数で渡せる（フィールド順の pin）。"""
    config = Agent365TracingConfig("svc", "ns")
    assert config.service_name == "svc"
    assert config.service_namespace == "ns"


def test_agent365_tracing_config_requires_service_name() -> None:
    """`service_name` を省略すると TypeError（Agent 365 configure() の必須引数）。"""
    with pytest.raises(TypeError, match="service_name"):
        Agent365TracingConfig(service_namespace="ns")  # type: ignore[call-arg]


def test_agent365_tracing_config_requires_service_namespace() -> None:
    """`service_namespace` を省略すると TypeError（Agent 365 configure() の必須引数）。"""
    with pytest.raises(TypeError, match="service_namespace"):
        Agent365TracingConfig(service_name="svc")  # type: ignore[call-arg]


def test_agent365_tracing_config_is_frozen() -> None:
    """構築後の属性代入は FrozenInstanceError（宣言は不変に保つ）。"""
    config = _make_tracing_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.service_name = "other"  # type: ignore[misc]


def test_agent365_tracing_config_optional_fields_are_retained() -> None:
    """任意フィールドは渡した値をそのまま保持する（logger_name / cluster_category）。"""
    config = _make_tracing_config(logger_name="my.logger", cluster_category="dev")
    assert config.logger_name == "my.logger"
    assert config.cluster_category == "dev"


def test_agent365_tracing_config_retains_token_resolver_callable() -> None:
    """`token_resolver` は callable を同一オブジェクトのまま保持する（呼び出しはしない）。"""

    def resolver(scope: str, tenant: str) -> str | None:
        return f"{scope}:{tenant}"

    config = _make_tracing_config(token_resolver=resolver)
    assert config.token_resolver is resolver
    assert config.token_resolver("s", "t") == "s:t"


def test_agent365_tracing_config_retains_opaque_exporter_options() -> None:
    """`exporter_options` は SDK 型を解釈せず不透明に保持する（SDK 隔離）。"""
    sentinel = object()
    config = _make_tracing_config(exporter_options=sentinel)
    assert config.exporter_options is sentinel


# ----------------------------------------------------------------------
# Agent365TracingConfig: 機微フィールドの repr 抑止（多層防御）
# ----------------------------------------------------------------------


class _ExporterOptionsSentinel:
    """repr に現れたら検出できるよう、目印を返す `exporter_options` のダミー。"""

    def __repr__(self) -> str:
        """接続設定の流出を検出するための目印。"""
        return "<EXPORTER-OPTIONS-SENTINEL>"


def _secret_bearing_token_resolver(scope: str, tenant: str) -> str | None:
    """認証情報を束縛した callable を模した resolver（関数名自体を目印に使う）。"""
    return "SECRET-TOKEN-VALUE"


def test_agent365_tracing_config_repr_hides_credential_fields() -> None:
    """`token_resolver` / `exporter_options` は repr に名前も値も現れない。

    利用者が `logger.info("config=%s", cfg)` のように設定をログへ出した際、認証情報を束縛した
    callable や接続設定が repr 経由で流出する経路を閉じる（多層防御）。値そのものは属性として
    保持され続ける（`repr=False` が `init=False` や値の破棄へ変異したら他の保持テストが落ちる）。
    """
    options = _ExporterOptionsSentinel()
    config = _make_tracing_config(
        token_resolver=_secret_bearing_token_resolver, exporter_options=options
    )

    text = repr(config)
    assert "token_resolver" not in text
    assert "exporter_options" not in text
    assert "_secret_bearing_token_resolver" not in text
    assert "EXPORTER-OPTIONS-SENTINEL" not in text
    # 表示から外れるだけで値は保持される。
    assert config.token_resolver is _secret_bearing_token_resolver
    assert config.exporter_options is options


def test_agent365_tracing_config_repr_shows_non_credential_fields() -> None:
    """機微でないフィールドは repr に残る（過剰に隠す変異を検出する）。

    repr は設定内容を確認するための主要な手段であり、抑止対象は認証情報・接続設定に限定する。
    """
    config = _make_tracing_config(
        service_name="svc-x",
        service_namespace="ns-x",
        logger_name="lg-x",
        cluster_category="dev",
        suppress_invoke_agent_input=True,
    )

    text = repr(config)
    for field_name in (
        "service_name",
        "service_namespace",
        "logger_name",
        "cluster_category",
        "suppress_invoke_agent_input",
    ):
        assert field_name in text, f"{field_name} が repr から欠落しています: {text}"
    assert "'svc-x'" in text
    assert "'ns-x'" in text
    assert "'lg-x'" in text
    assert "'dev'" in text
    assert "True" in text


# ----------------------------------------------------------------------
# Agent365TracingConfig: suppress_invoke_agent_input の構築時 bool 型検証
# ----------------------------------------------------------------------


def test_agent365_tracing_config_suppress_none_raises() -> None:
    """`suppress_invoke_agent_input=None` は bool でないため ValueError（メッセージ全文を pin）。

    入力抑止フラグが黙って falsy になる（プロンプト本文がトレースへ送出される）silent failure を
    排除する。
    """
    with pytest.raises(
        ValueError,
        match=re.escape("suppress_invoke_agent_input must be a bool, got 'NoneType'"),
    ):
        _make_tracing_config(suppress_invoke_agent_input=None)


def test_agent365_tracing_config_suppress_str_raises() -> None:
    """`suppress_invoke_agent_input="true"` は truthy な文字列だが ValueError で弾く。"""
    with pytest.raises(
        ValueError,
        match=re.escape("suppress_invoke_agent_input must be a bool, got 'str'"),
    ):
        _make_tracing_config(suppress_invoke_agent_input="true")


def test_agent365_tracing_config_suppress_int_one_raises() -> None:
    """`suppress_invoke_agent_input=1`（int）は bool でないため ValueError。"""
    with pytest.raises(
        ValueError,
        match=re.escape("suppress_invoke_agent_input must be a bool, got 'int'"),
    ):
        _make_tracing_config(suppress_invoke_agent_input=1)


def test_agent365_tracing_config_suppress_bool_constructs() -> None:
    """True / False を渡した構築は成功し、値がそのまま保持される（正常系の維持）。"""
    assert (
        _make_tracing_config(suppress_invoke_agent_input=True).suppress_invoke_agent_input is True
    )
    assert (
        _make_tracing_config(suppress_invoke_agent_input=False).suppress_invoke_agent_input is False
    )


# ----------------------------------------------------------------------
# OtelLoggingConfig: 既定値・不変性・値保持
# ----------------------------------------------------------------------


def test_otel_logging_config_defaults() -> None:
    """全フィールド省略時は service_name 未指定 / INFO / OTLP 併用なし / 整形済みコンソール出力。

    `console_json_lines` の既定 False は「現行のコンソール出力（`indent=4` の整形済み JSON）を
    維持する」という後方互換の約束であり、既定値が True へ反転すると目視確認前提の開発体験が
    黙って変わる。
    """
    config = OtelLoggingConfig()
    assert config.service_name is None
    assert config.level == logging.INFO
    assert config.otlp_enabled is False
    assert config.console_json_lines is False


def test_otel_logging_config_is_frozen() -> None:
    """構築後の属性代入は FrozenInstanceError（root logger への結線設定は不変）。"""
    config = OtelLoggingConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.level = logging.DEBUG  # type: ignore[misc]


def test_otel_logging_config_retains_values() -> None:
    """指定した service_name / level はそのまま保持される。"""
    config = OtelLoggingConfig(service_name="svc", level=logging.DEBUG)
    assert config.service_name == "svc"
    assert config.level == logging.DEBUG


def test_otel_logging_config_accepts_positional_fields() -> None:
    """全フィールドを宣言順の位置引数で渡せる（フィールド順の pin）。

    `console_json_lines` は**末尾**に足す（既存の位置引数呼び出しの意味を変えないため）。
    途中へ挿入する変異はこの位置引数呼び出しで RED になる。
    """
    config = OtelLoggingConfig("svc", logging.WARNING, True, True)
    assert config.service_name == "svc"
    assert config.level == logging.WARNING
    assert config.otlp_enabled is True
    assert config.console_json_lines is True


# ----------------------------------------------------------------------
# OtelLoggingConfig: otlp_enabled の構築時 bool 型検証
# ----------------------------------------------------------------------


def test_otel_logging_config_otlp_enabled_none_raises() -> None:
    """`otlp_enabled=None` は bool でないため ValueError（メッセージ全文を pin）。

    OTLP 併用フラグが黙って falsy になり、宛先が既定コンソールのみへ縮退する silent failure を
    排除する。
    """
    with pytest.raises(ValueError, match=re.escape("otlp_enabled must be a bool, got 'NoneType'")):
        OtelLoggingConfig(otlp_enabled=None)  # type: ignore[arg-type]


def test_otel_logging_config_otlp_enabled_str_raises() -> None:
    """`otlp_enabled="true"` は truthy な文字列だが ValueError で弾く。"""
    with pytest.raises(ValueError, match=re.escape("otlp_enabled must be a bool, got 'str'")):
        OtelLoggingConfig(otlp_enabled="true")  # type: ignore[arg-type]


def test_otel_logging_config_otlp_enabled_int_one_raises() -> None:
    """`otlp_enabled=1`（int）は bool でないため ValueError。"""
    with pytest.raises(ValueError, match=re.escape("otlp_enabled must be a bool, got 'int'")):
        OtelLoggingConfig(otlp_enabled=1)  # type: ignore[arg-type]


def test_otel_logging_config_otlp_enabled_bool_constructs() -> None:
    """True / False を渡した構築は成功し、値がそのまま保持される（正常系の維持）。"""
    assert OtelLoggingConfig(otlp_enabled=True).otlp_enabled is True
    assert OtelLoggingConfig(otlp_enabled=False).otlp_enabled is False


# ----------------------------------------------------------------------
# OtelLoggingConfig: console_json_lines の構築時 bool 型検証
# ----------------------------------------------------------------------


def test_otel_logging_config_console_json_lines_none_raises() -> None:
    """`console_json_lines=None` は bool でないため ValueError（メッセージ全文を pin）。

    1 行 JSON 指定が黙って falsy になると、ログ収集基盤が 1 行 = 1 レコードとして取り込めない
    整形済み出力へ縮退する（有効化したつもりで効いていない silent failure）。
    """
    with pytest.raises(
        ValueError, match=re.escape("console_json_lines must be a bool, got 'NoneType'")
    ):
        OtelLoggingConfig(console_json_lines=None)  # type: ignore[arg-type]


def test_otel_logging_config_console_json_lines_str_raises() -> None:
    """`console_json_lines="true"` は truthy な文字列だが ValueError で弾く。"""
    with pytest.raises(ValueError, match=re.escape("console_json_lines must be a bool, got 'str'")):
        OtelLoggingConfig(console_json_lines="true")  # type: ignore[arg-type]


def test_otel_logging_config_console_json_lines_int_one_raises() -> None:
    """`console_json_lines=1`（int）は bool でないため ValueError。"""
    with pytest.raises(ValueError, match=re.escape("console_json_lines must be a bool, got 'int'")):
        OtelLoggingConfig(console_json_lines=1)  # type: ignore[arg-type]


def test_otel_logging_config_console_json_lines_bool_constructs() -> None:
    """True / False を渡した構築は成功し、値がそのまま保持される（正常系の維持）。"""
    assert OtelLoggingConfig(console_json_lines=True).console_json_lines is True
    assert OtelLoggingConfig(console_json_lines=False).console_json_lines is False


# ----------------------------------------------------------------------
# SDK 非依存（subprocess 隔離）
# ----------------------------------------------------------------------


def test_config_module_does_not_load_observability_sdks() -> None:
    """config モジュールの import と dataclass 構築で観測系 SDK / agents をロードしない。

    設定型は宣言のみで有効化（グローバル結線）を行わないため、`opentelemetry` /
    `microsoft_agents_a365` は一切ロードされてはならない。`agents` はコア依存として
    `import oai_agentspec` の時点で既に載るため、ベースライン（本体 import 直後）からの差分に
    `agents.*` が現れないことで「config 自身が SDK を import しない」ことを検証する。
    他テストの副作用を排除するためクリーンな子プロセスで確認する。
    """
    probe = (
        "import sys\n"
        "import oai_agentspec\n"
        "baseline = set(sys.modules)\n"
        "from oai_agentspec.runtime.observability.config import (\n"
        "    Agent365TracingConfig,\n"
        "    OtelLoggingConfig,\n"
        ")\n"
        "Agent365TracingConfig(service_name='svc', service_namespace='ns')\n"
        "OtelLoggingConfig()\n"
        "added = sorted(m for m in set(sys.modules) - baseline "
        "if m == 'agents' or m.startswith('agents.'))\n"
        "forbidden = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m == 'opentelemetry' or m.startswith('opentelemetry.')\n"
        "    or m == 'microsoft_agents_a365' or m.startswith('microsoft_agents_a365')\n"
        ")\n"
        "print(','.join(added + forbidden))\n"
    )
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_SRC_DIR) + (os.pathsep + existing if existing else "")
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    loaded = [m for m in result.stdout.strip().split(",") if m]
    assert loaded == [], f"observability 設定型の import / 構築で SDK がロードされました: {loaded}"
