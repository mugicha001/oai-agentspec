"""L1: `_adapters.observability` の純ロジック検証（spy 注入・実 SDK 非依存）。

`enable_agent365_tracing`（Agent 365 トレース連携）と `enable_otel_logging`（標準 logging ->
OTel Logs 連携）の結線を、依存取得関数（`_require_agent365_tracing` / `_require_opentelemetry` /
`_require_otlp_log_exporter`）を monkeypatch で spy へ差し替えて検証する。実 SDK は一切呼ばない
（実 SDK 側の契約 -- `Agent365ExporterOptions` が `token_resolver` を持ち
`SpectraExporterOptions` が持たないこと・上流の引数合成規則 -- は `test_observability_l2.py` が
別途 pin する）。

pin する不変条件（ADR 0022 / ADR 0024 Confirmation）:

- 順序契約: `configure()` は `OpenAIAgentsTraceInstrumentor` 生成より先に呼ばれる。
- パススルー: `Agent365TracingConfig` の各フィールドが `configure()` へそのまま渡る
  （`logger_name=None` のときだけ引数を渡さず SDK 既定に委ねる）。
- 構成失敗: 警告のみで例外にせず、かつ計装へ進まない（`configure()` が偽・`is_configured()` が
  偽のいずれでも同じ着地。計装は既定のトレースプロセッサ列を置換するため、半端な構成のまま
  進むと既定の送信先まで失われる）。正常系で計装へ到達することは別テストが pin する。
- resolver ドロップ検知: `token_resolver` と Agent365 形式 `exporter_options`（resolver 未設定）の
  併用は `RuntimeWarning` で通知するが処理は継続する。属性を持たない値（sidecar 構成・未指定）は
  番兵で除外して警告しない。警告文面は原因・断定を避ける条件節・是正指示の 3 点を個別に固定する。
- 観測の失敗でアプリを止めない: `exporter_options` の属性取得が例外を投げても伝播させず、
  判定不能として番兵へ倒して計装まで到達する。
- 判定不能の痕跡: 吸収した例外は警告にしないが DEBUG ログへトレースバック付きで記録する
  （無音化すると番兵へ倒れた事実が「判定対象外」と外形上区別できなくなる）。
- 属性読み取りは 1 回だけ: 利用者が渡す計算プロパティを二重評価しない（ADR 0024）。
- 警告の帰属: すべての `RuntimeWarning` は `stacklevel=2` で利用者の呼び出し位置へ帰属する
  （`stacklevel=1` への退行は警告を出し続けるためメッセージ照合では検知できない）。
- 冪等: `enable_otel_logging` を複数回呼んでも root logger への `LoggingHandler` 付与は 1 回だけ。
- 再設定検知: 初回と異なる設定での 2 回目は `RuntimeWarning` のみで、適用済みの結線は変えない。
- 非接触: 既存 handler オブジェクト・フォーマッタ・登録順・root logger の level は変更しない
  （追加のみ）。
- OTLP 併用: `otlp_enabled=True` は既定の Console を置換せず processor を 2 本構成にする。
- tracing 無効時: `RuntimeWarning` で通知しつつ処理は継続する（例外にしない）。
- 未導入時: 案内付き `ImportError`。

`_require_opentelemetry()` の戻り値は属性名で参照されるため `SimpleNamespace` のスタブで模す
（属性名が実 SDK と一致することは L2 が実パッケージに対して pin する）。instrumentor のスタブは
実 SDK と同じく「未構成なら生成時に `RuntimeError`」という前提条件を再現する（スタブが実 SDK
より緩いと、実環境で例外になる実装が L1 だけ緑で通過するため）。
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from oai_agentspec._adapters import observability as obs
from oai_agentspec.runtime.observability.config import Agent365TracingConfig, OtelLoggingConfig

pytestmark = pytest.mark.unit

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"


def _assert_warnings_attributed_to_caller(caught: Any) -> None:
    """記録された `RuntimeWarning` が利用者の呼び出し位置（本ファイル）へ帰属することを検証する。

    `warnings.warn(..., stacklevel=2)` の 2 を 1 へ落とす退行は、警告の帰属を lib 内部の行へ
    移す。利用者は `filterwarnings` の module 指定や IDE の飛び先で自分の呼び出し箇所を特定
    できなくなるが、警告そのものは出続けるためメッセージ照合では検知できない。

    `RuntimeWarning` に絞るのは、pin の対象が「lib が出す警告の帰属」であるため。`pytest.warns`
    は指定カテゴリ以外も記録するので、母集団を広げると将来ブロック内で依存ライブラリが出す
    `DeprecationWarning` 等によって、`stacklevel` の退行が無いのに赤くなる。

    Args:
        caught: `pytest.warns(...)` / `warnings.catch_warnings(record=True)` の記録列。
    """
    filenames = [item.filename for item in caught if item.category is RuntimeWarning]
    assert filenames, "RuntimeWarning が 1 件も記録されていません"
    assert set(filenames) == {__file__}, filenames


# ----------------------------------------------------------------------
# spy / fake
# ----------------------------------------------------------------------


class _TracingSpy:
    """Agent 365 側 3 シンボル（configure / is_configured / instrumentor）の spy 束。

    `configure` / instrumentor 生成 / `instrument` の呼び出しを **1 本の列**（`calls`）へ記録し、
    順序契約を列全体の等値比較で検証できるようにする。
    """

    # `configure()` の位置引数を名前へ正規化するための宣言順（実 SDK のシグネチャ順）。
    _CONFIGURE_PARAMS = (
        "service_name",
        "service_namespace",
        "logger_name",
        "token_resolver",
        "cluster_category",
        "exporter_options",
        "suppress_invoke_agent_input",
    )

    def __init__(self, *, configured: bool = True, configure_result: bool = True) -> None:
        """spy を初期化する。

        Args:
            configured: `is_configured()` の戻り値（構成済み判定）。
            configure_result: `configure()` の戻り値（実 SDK は失敗時に例外でなく False を返す）。
        """
        self.calls: list[str] = []
        self.configure_kwargs: dict[str, Any] | None = None
        self.is_configured_calls = 0
        self._configured = configured
        self._configure_result = configure_result
        spy = self

        class _Instrumentor:
            """`OpenAIAgentsTraceInstrumentor` のスタブ（生成 / instrument を記録する）。

            実 SDK は `__init__` で `is_configured()` を確認し、未構成なら `RuntimeError` を
            送出する（`test_observability_l2.py::test_instrumentor_requires_configure_first`
            が実 SDK に対して pin している前提条件）。スタブがこの前提を模さないと「未構成の
            まま計装を生成する」実装が L1 だけ緑で通過してしまうため、同じ条件を再現する。
            """

            def __init__(self) -> None:
                if not spy._configured:
                    raise RuntimeError("Microsoft Agent 365 is not configured yet.")
                spy.calls.append("instrumentor_init")

            def instrument(self, **kwargs: Any) -> None:
                spy.calls.append("instrument")

        self.instrumentor_cls = _Instrumentor

    def configure(self, *args: Any, **kwargs: Any) -> bool:
        """`configure()` のスタブ（位置引数も名前へ正規化して記録する）。"""
        self.calls.append("configure")
        merged = dict(zip(self._CONFIGURE_PARAMS, args, strict=False))
        merged.update(kwargs)
        self.configure_kwargs = merged
        return self._configure_result

    def is_configured(self) -> bool:
        """`is_configured()` のスタブ。"""
        self.is_configured_calls += 1
        return self._configured

    def as_tuple(self) -> tuple[Any, Any, Any]:
        """`_require_agent365_tracing()` の戻り値形（3 要素タプル）へ整形する。"""
        return (self.configure, self.is_configured, self.instrumentor_cls)


class _FakeExporter:
    """LogExporter のスタブ（種別の判別だけに使う）。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """引数は保持のみ（送信は行わない）。"""
        self.args = args
        self.kwargs = kwargs


class _FakeConsoleExporter(_FakeExporter):
    """`ConsoleLogExporter` のスタブ。"""


class _FakeOtlpExporter(_FakeExporter):
    """`OTLPLogExporter` のスタブ。"""


class _FakeReadableLogRecord:
    """`ReadableLogRecord` のスタブ（`to_json()` の既定 indent を実 SDK と同形に模す）。

    実 SDK の `ReadableLogRecord.to_json(indent: int | None = 4)` は `json.dumps()` の結果を
    返すため、既定（`indent=4`）では複数行・`indent=None` では 1 行になる（実 SDK 側の契約は
    `test_observability_l2.py` の `..._to_json_indent_contract` が pin する）。
    スタブが indent を無視すると「整形済み出力を返す formatter」を渡す実装まで緑になるため、
    ここでは実 SDK と同じく indent を反映させる。
    """

    _PAYLOAD = {
        "body": "single-line test",
        "severity_text": "INFO",
        "attributes": {"code.function": "f"},
        "resource": {"service.name": "svc-x"},
    }

    def to_json(self, indent: int | None = 4) -> str:
        """ペイロードを JSON 文字列にする（`indent` の指定をそのまま反映する）。"""
        return json.dumps(self._PAYLOAD, indent=indent)


class _FakeProcessor:
    """LogRecordProcessor のスタブ（ラップした exporter を保持する）。"""

    def __init__(self, exporter: Any = None, *args: Any, **kwargs: Any) -> None:
        """exporter を保持する（Batch / Simple のどちらの実装でも成立する形）。"""
        self.exporter = exporter


class _OtelSpy:
    """OTel Logs 側スタブ一式（生成された provider / handler / resource を記録する）。

    `_require_opentelemetry()` が返す名前空間（5 属性）と、別配布物のため分離されている
    `_require_otlp_log_exporter()` の戻り値（OTLP エクスポータクラス）の両方を差し替える。
    """

    def __init__(self) -> None:
        """provider / handler の生成記録を初期化し、スタブ名前空間を組み立てる。"""
        self.providers: list[Any] = []
        self.handlers: list[Any] = []
        self.resources: list[Any] = []
        self.otlp_exporter_cls: Any = _FakeOtlpExporter
        spy = self

        class _Resource:
            """`Resource` のスタブ（`create()` に渡された属性を記録する）。"""

            def __init__(self, attributes: Any = None) -> None:
                self.attributes = attributes

            @classmethod
            def create(cls, attributes: Any = None) -> _Resource:
                """`Resource.create()` のスタブ（生成物を spy に記録する）。"""
                instance = cls(attributes)
                spy.resources.append(instance)
                return instance

        class _LoggerProvider:
            """`LoggerProvider` のスタブ（登録された processor を保持する）。"""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs
                self.processors: list[Any] = []
                spy.providers.append(self)

            def add_log_record_processor(self, processor: Any) -> None:
                """processor を登録順に保持する（置換ではなく追加であることの観測点）。"""
                self.processors.append(processor)

        class _LoggingHandler(logging.Handler):
            """`LoggingHandler` のスタブ（root へ実際に addHandler されるため実 Handler 派生）。"""

            def __init__(
                self, level: int = logging.NOTSET, logger_provider: Any = None, **kwargs: Any
            ) -> None:
                super().__init__(level=level)
                self.logger_provider = logger_provider
                self.extra_kwargs = kwargs
                spy.handlers.append(self)

            def emit(self, record: logging.LogRecord) -> None:
                """no-op（送出しない）。"""

        # 属性名は `_require_opentelemetry()` の契約（`test_observability_l2.py` が実 SDK で pin）。
        self.namespace = SimpleNamespace(
            LoggerProvider=_LoggerProvider,
            LoggingHandler=_LoggingHandler,
            ConsoleLogExporter=_FakeConsoleExporter,
            LogRecordProcessor=_FakeProcessor,
            Resource=_Resource,
        )

    @property
    def provider(self) -> Any:
        """最後に生成された `LoggerProvider` スタブ。"""
        assert self.providers, "LoggerProvider が 1 度も生成されていません"
        return self.providers[-1]

    def exporter_types(self) -> list[str]:
        """最後の provider に登録された processor が包む exporter の型名列。"""
        return [type(p.exporter).__name__ for p in self.provider.processors]

    def exporter_of(self, exporter_cls: type) -> Any:
        """最後の provider に登録された指定種別の exporter スタブ（ちょうど 1 本を要求する）。"""
        found = [
            p.exporter for p in self.provider.processors if isinstance(p.exporter, exporter_cls)
        ]
        assert len(found) == 1, f"{exporter_cls.__name__} が 1 本ではありません: {len(found)}"
        return found[0]


# ----------------------------------------------------------------------
# fixture
# ----------------------------------------------------------------------


@pytest.fixture
def root_handlers() -> Iterator[list[logging.Handler]]:
    """root logger の handler 列と level を snapshot し、teardown で必ず復元する。

    Yields:
        呼び出し前の handler 列（`is` 比較用のスナップショット）。
    """
    root = logging.getLogger()
    before = list(root.handlers)
    before_level = root.level
    try:
        yield before
    finally:
        root.handlers[:] = before
        root.setLevel(before_level)


@pytest.fixture
def otel(monkeypatch: pytest.MonkeyPatch) -> _OtelSpy:
    """`_require_opentelemetry` をスタブへ差し替え、付与状態（フラグと適用済み設定）を初期化する。

    冪等フラグと適用済み設定はモジュール状態のため、両方を戻さないとテスト間で漏れる
    （片方だけ戻すと「付与済みでないのに適用済み設定が残る」不整合な初期状態になる）。
    """
    spy = _OtelSpy()
    monkeypatch.setattr(obs, "_OTEL_LOG_HANDLER_ATTACHED", False)
    monkeypatch.setattr(obs, "_OTEL_LOG_APPLIED_SETTINGS", None)
    monkeypatch.setattr(obs, "_require_opentelemetry", lambda: spy.namespace)
    # OTLP エクスポータは別配布物のため取得関数が分かれている（併用時だけ解決される）。
    monkeypatch.setattr(obs, "_require_otlp_log_exporter", lambda: spy.otlp_exporter_cls)
    return spy


@pytest.fixture
def tracing() -> Iterator[None]:
    """SDK トレーシングを局所的に有効化する（teardown で root conftest の既定へ戻す）。

    root conftest の autouse fixture が全テストで `set_tracing_disabled(True)` を立てているため、
    「トレーシング有効」状態の検証には明示的な上書きが必要になる。
    """
    from agents import set_tracing_disabled

    set_tracing_disabled(False)
    try:
        yield
    finally:
        set_tracing_disabled(True)


@pytest.fixture
def tracing_spy(monkeypatch: pytest.MonkeyPatch) -> _TracingSpy:
    """`_require_agent365_tracing` を spy へ差し替える。"""
    spy = _TracingSpy()
    monkeypatch.setattr(obs, "_require_agent365_tracing", spy.as_tuple)
    return spy


def _tracing_config(**overrides: Any) -> Agent365TracingConfig:
    """必須フィールドを埋めた `Agent365TracingConfig` を生成する。"""
    kwargs: dict[str, Any] = {"service_name": "svc", "service_namespace": "ns"}
    kwargs.update(overrides)
    return Agent365TracingConfig(**kwargs)


# ----------------------------------------------------------------------
# enable_agent365_tracing: 順序契約・パススルー
# ----------------------------------------------------------------------


def test_enable_agent365_tracing_configures_before_instrumentor(
    tracing_spy: _TracingSpy, tracing: None
) -> None:
    """`configure()` -> instrumentor 生成 -> `instrument()` の順で呼ばれる（列全体を照合）。

    instrumentor は `configure()` 前に生成すると実 SDK が `RuntimeError` を送出するため、
    順序そのものが契約になる。列全体の等値比較にすることで順序入れ替えの変異を検出する。
    """
    obs.enable_agent365_tracing(_tracing_config())

    assert tracing_spy.calls == ["configure", "instrumentor_init", "instrument"]


def test_enable_agent365_tracing_passes_all_config_fields(
    tracing_spy: _TracingSpy, tracing: None
) -> None:
    """config の全フィールドが `configure()` へパススルーされる（期待 dict と全体照合）。

    payload はすべて既定値と異なる sentinel にしてあるため、配線が抜けた変異が必ず差分になる。
    """

    def resolver(scope: str, tenant: str) -> str | None:
        return None

    options = object()
    config = _tracing_config(
        service_name="svc-x",
        service_namespace="ns-x",
        logger_name="lg-x",
        token_resolver=resolver,
        cluster_category="dev",
        exporter_options=options,
        suppress_invoke_agent_input=True,
    )

    obs.enable_agent365_tracing(config)

    assert tracing_spy.configure_kwargs == {
        "service_name": "svc-x",
        "service_namespace": "ns-x",
        "logger_name": "lg-x",
        "token_resolver": resolver,
        "cluster_category": "dev",
        "exporter_options": options,
        "suppress_invoke_agent_input": True,
    }


def test_enable_agent365_tracing_omits_logger_name_when_none(
    tracing_spy: _TracingSpy, tracing: None
) -> None:
    """`logger_name=None` のときだけ引数を渡さない（SDK 既定の上書きを避ける）。

    「キーが無いこと」だけを見ると全項目が渡っていない実装でも通ってしまうため、
    残り 6 フィールドが渡っていることを同時に検証する。
    """
    obs.enable_agent365_tracing(_tracing_config(logger_name=None, cluster_category="dev"))

    kwargs = tracing_spy.configure_kwargs
    assert kwargs is not None
    assert "logger_name" not in kwargs
    assert set(kwargs) == {
        "service_name",
        "service_namespace",
        "token_resolver",
        "cluster_category",
        "exporter_options",
        "suppress_invoke_agent_input",
    }
    assert kwargs["cluster_category"] == "dev"


# ----------------------------------------------------------------------
# enable_agent365_tracing: tracing 無効検知
# ----------------------------------------------------------------------


def test_enable_agent365_tracing_warns_when_tracing_disabled(tracing_spy: _TracingSpy) -> None:
    """`set_tracing_disabled(True)` 配下では RuntimeWarning を出しつつ結線は継続する。

    トレースが送信されない状態の黙殺を禁止する一方、例外にはしない（ADR 0022）。
    """
    from agents import set_tracing_disabled

    set_tracing_disabled(True)
    with pytest.warns(RuntimeWarning) as caught:
        obs.enable_agent365_tracing(_tracing_config())

    _assert_warnings_attributed_to_caller(caught)
    # 警告後も中断せず結線が完了している（例外化・early return への変異を検出する）。
    assert tracing_spy.calls == ["configure", "instrumentor_init", "instrument"]


def test_enable_agent365_tracing_does_not_warn_when_tracing_enabled(
    tracing_spy: _TracingSpy, tracing: None
) -> None:
    """トレーシング有効時は警告を出さない（無条件 warn への変異を検出する）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obs.enable_agent365_tracing(_tracing_config())

    assert tracing_spy.calls == ["configure", "instrumentor_init", "instrument"]


def test_enable_agent365_tracing_twice_does_not_raise(
    tracing_spy: _TracingSpy, tracing: None
) -> None:
    """2 回呼んでも例外にならない（重複有効化の下限契約）。"""
    obs.enable_agent365_tracing(_tracing_config())
    obs.enable_agent365_tracing(_tracing_config())


# ----------------------------------------------------------------------
# enable_agent365_tracing: 構成失敗の検知（例外でなく戻り値で失敗する SDK 契約）
# ----------------------------------------------------------------------


def test_enable_agent365_tracing_warns_when_configure_returns_false(
    monkeypatch: pytest.MonkeyPatch, tracing: None
) -> None:
    """`configure()` が False なら `is_configured()` が True でも計装へ進まない。

    実 SDK の `configure()` は内部例外を握り潰して False を返すため、戻り値を見なければ
    「構成に失敗したのに無警告で進む」silent failure になる。さらに構成失敗のまま計装すると
    実害が出る: 計装は `set_trace_processors` で SDK のトレースプロセッサ列を**置換**するため、
    Agent 365 側が半端な構成（span processor 未登録）の場合、SDK 既定の送信先まで失われて
    トレースがどこにも到達しなくなる。`_uninstrument` は既定列を復元しないため事後修復も
    できない。したがって構成失敗と未構成は**同じ扱い**（警告して計装せず return）とする。

    - `configure()` が False かつ `is_configured()` が True の組: 本テスト（分岐自体は
      `is_configured()` の真偽を問わず同じ着地になる）。
    - `is_configured()` が False: `..._warns_when_not_configured`（同じ着地）。

    正常系で計装へ到達すること（= 本条件を過剰に塞ぐ退行の検知）は
    `..._configures_before_instrumentor` / `..._does_not_warn_when_tracing_enabled` が pin する。
    """
    spy = _TracingSpy(configure_result=False)
    monkeypatch.setattr(obs, "_require_agent365_tracing", spy.as_tuple)

    with pytest.warns(RuntimeWarning, match="構成に失敗") as caught:
        obs.enable_agent365_tracing(_tracing_config())

    _assert_warnings_attributed_to_caller(caught)
    # instrumentor は生成しない（既定のトレースプロセッサ列を置換させないため）。
    assert spy.calls == ["configure"]


def test_enable_agent365_tracing_warns_when_not_configured(
    monkeypatch: pytest.MonkeyPatch, tracing: None
) -> None:
    """`is_configured()` が False なら警告し、計装へは進まない（RuntimeError を避ける）。

    構成の成否は `is_configured()` が最終的な真実であり、戻り値だけを見る実装では「True を
    返したが構成されていない」状態を取りこぼす。さらに未構成のまま instrumentor を生成すると
    実 SDK は `RuntimeError` を送出する（`test_observability_l2.py` が実 SDK で pin）ため、
    警告のみに留めて中断する必要がある。
    """
    spy = _TracingSpy(configured=False)
    monkeypatch.setattr(obs, "_require_agent365_tracing", spy.as_tuple)

    with pytest.warns(RuntimeWarning, match="構成に失敗"):
        obs.enable_agent365_tracing(_tracing_config())

    # instrumentor は生成しない（生成した時点で RuntimeError になるため）。
    assert spy.calls == ["configure"]


def test_enable_agent365_tracing_does_not_warn_when_configure_succeeds(
    tracing_spy: _TracingSpy, tracing: None
) -> None:
    """構成成功時は構成失敗の警告を出さない（無条件 warn への変異を検出する）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obs.enable_agent365_tracing(_tracing_config())


# ----------------------------------------------------------------------
# enable_agent365_tracing: exporter_options による token_resolver ドロップ検知
# ----------------------------------------------------------------------


def _resolver(scope: str, tenant: str) -> str | None:
    """`token_resolver` として渡す callable のスタブ（値は使わない）。"""
    return None


# 到達先未達警告の原因部分。両方の発火テストが正方向で照合するため、綴りの誤りは必ず RED になる。
_DROPPED_RESOLVER_PHRASE = "token_resolver は Agent 365 側で参照されません"


def test_enable_agent365_tracing_warns_when_exporter_options_drops_token_resolver(
    tracing_spy: _TracingSpy, tracing: None
) -> None:
    """resolver を渡したのに Agent365 形式 options 併用で捨てられる構成を警告する。

    上流は `exporter_options` が渡された場合トップレベルの `token_resolver` を参照しないため、
    認証手段を渡したつもりでコンソール出力へフォールバックする。`configure()` /
    `is_configured()` はこの構成でも真を返すため、警告が唯一の検知手段になる。

    検知する退行: 警告ブロックの削除（M5）・例外化（M6）・カテゴリ変更（M7）・照合語の削除（M10）。
    加えて警告後も `configure` -> 計装へ到達することを列全体で検査し、警告ブロック末尾へ
    `return` を挿入して無音で計装が止まる退行（M8）を検知する。

    文面は 3 点を個別に固定する: (a) 原因（渡した resolver が参照されないこと）、(b) 断定を
    避ける条件節（同一プロセスの 2 回目以降の `configure()` は渡した設定を丸ごと無視して真を
    返すため、結果を断定すると偽になる。ADR 0024 Decision 5）、(c) 是正指示（利用者が次に取る
    行動）。単一の `match=` では (b) (c) が消えても緑のまま通る。
    """
    config = _tracing_config(
        token_resolver=_resolver,
        exporter_options=SimpleNamespace(token_resolver=None),
    )

    with pytest.warns(RuntimeWarning, match=_DROPPED_RESOLVER_PHRASE) as caught:
        obs.enable_agent365_tracing(config)

    assert len(caught) == 1, [str(item.message) for item in caught]
    message = str(caught[0].message)
    # (b) 結果を断定しない条件節。
    assert "この設定が適用される場合" in message, message
    # (c) 是正指示。
    assert "exporter_options の token_resolver を設定してください" in message, message
    _assert_warnings_attributed_to_caller(caught)

    # 警告のみで中断しない（警告後に return を挿入する変異を検出する）。
    assert tracing_spy.calls == ["configure", "instrumentor_init", "instrument"]


def test_enable_agent365_tracing_warns_about_dropped_resolver_even_when_configure_fails(
    monkeypatch: pytest.MonkeyPatch, tracing: None
) -> None:
    """構成失敗（`configure()` が偽）でも resolver ドロップの警告は発出される。

    判定材料は config のみで構成の成否と独立しているため、警告は `configure()` 呼び出しの
    **前**に置く。構成失敗は early return するので、警告ブロックを構成失敗判定の後ろへ移すと
    この経路で警告が失われる（順序退行 M9）。

    構成失敗の警告も同時に出るため、全警告を記録して「resolver ドロップの警告が含まれること」を
    直接検査する（`pytest.warns(match=...)` 単独では他方の警告で偽に緑化しうる）。
    """
    spy = _TracingSpy(configure_result=False)
    monkeypatch.setattr(obs, "_require_agent365_tracing", spy.as_tuple)
    config = _tracing_config(
        token_resolver=_resolver,
        exporter_options=SimpleNamespace(token_resolver=None),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obs.enable_agent365_tracing(config)

    messages = [str(item.message) for item in caught]
    dropped = [msg for msg in messages if _DROPPED_RESOLVER_PHRASE in msg]
    assert dropped, f"resolver ドロップの警告が出ていません: {messages}"
    assert all(item.category is RuntimeWarning for item in caught)
    _assert_warnings_attributed_to_caller(caught)
    # 構成失敗側の着地（計装しない）は既存テストが pin する。ここでは共存のみ確認する。
    assert any("構成に失敗" in msg for msg in messages), messages


def test_enable_agent365_tracing_does_not_warn_when_exporter_options_carries_token_resolver(
    tracing_spy: _TracingSpy, tracing: None
) -> None:
    """options 側に resolver が設定済みなら警告しない（実効 resolver があるため）。

    検知する退行: 条件から `options.token_resolver is None` を落として resolver 設定済み構成へ
    誤警告する変異（M1）。
    """
    config = _tracing_config(
        token_resolver=_resolver,
        exporter_options=SimpleNamespace(token_resolver=_resolver),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obs.enable_agent365_tracing(config)

    assert tracing_spy.calls == ["configure", "instrumentor_init", "instrument"]


class _NoResolverAttrOptions:
    """`token_resolver` 属性をまったく持たない options のスタブ（sidecar 構成に相当）。

    実 `SpectraExporterOptions` が当該属性を持たないことは `test_observability_l2.py` が実 SDK に
    対して pin する。L1 側で実クラスを構築すると observability extra 未導入環境では
    `pytest.importorskip` で skip され、番兵分岐に対する変異（M2）の検知が黙って失われるため、
    ここでは「属性を持たない値」であることだけをスタブで模す。
    """


def test_enable_agent365_tracing_does_not_warn_for_options_without_resolver_attribute(
    tracing_spy: _TracingSpy, tracing: None
) -> None:
    """`token_resolver` 属性を持たない options（sidecar 構成）では resolver 併用でも警告しない。

    sidecar 構成は OTLP コレクタへ送るため resolver を必要とせず、上流の options 型は当該属性を
    持たない。実装は属性の不在（番兵）でこの構成を除外する。

    検知する退行: `getattr(options, "token_resolver", _MISSING)` の既定値を `None` へ置換し、
    属性を持たない値を「resolver が捨てられた」と誤検知する変異（M2）。トップレベル
    `token_resolver` も同時に渡さないとこの変異を検知できないため必ず渡す。
    """
    config = _tracing_config(
        token_resolver=_resolver,
        exporter_options=_NoResolverAttrOptions(),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obs.enable_agent365_tracing(config)

    assert tracing_spy.calls == ["configure", "instrumentor_init", "instrument"]


def test_enable_agent365_tracing_does_not_warn_without_options(
    tracing_spy: _TracingSpy, tracing: None
) -> None:
    """`exporter_options` 未指定なら警告しない（resolver は上流で合成される）。

    検知する退行: 条件を無条件化する変異（M4）。
    """
    config = _tracing_config(token_resolver=_resolver)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obs.enable_agent365_tracing(config)

    assert tracing_spy.calls == ["configure", "instrumentor_init", "instrument"]


def test_enable_agent365_tracing_does_not_warn_for_options_without_top_level_resolver(
    tracing_spy: _TracingSpy, tracing: None
) -> None:
    """resolver をどこにも渡していない options 単独指定では警告しない。

    上流はバッチ処理パラメータを全エクスポータ分岐で使うため、コンソール出力のままバッチ挙動を
    調整する目的で `Agent365ExporterOptions(max_queue_size=...)` を渡すのは正当な用途であり、
    ここで警告すると「動いている構成へのノイズ」になる。

    検知する退行: 条件から `config.token_resolver is not None` を落とし、バッチ調整目的の
    options 単独指定へ誤警告する変異（M3）。
    """
    config = _tracing_config(exporter_options=SimpleNamespace(token_resolver=None))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obs.enable_agent365_tracing(config)

    assert tracing_spy.calls == ["configure", "instrumentor_init", "instrument"]


class _CountingOptions:
    """`token_resolver` の読み取り回数を数える options のスタブ（値は `None` を返す）。"""

    def __init__(self) -> None:
        """読み取り回数カウンタを初期化する。"""
        self.reads = 0

    @property
    def token_resolver(self) -> Any:
        """読み取りのたびに回数を数え、resolver 未設定を表す `None` を返す。"""
        self.reads += 1
        return None


def test_enable_agent365_tracing_reads_options_token_resolver_once(
    tracing_spy: _TracingSpy, tracing: None
) -> None:
    """判定は `exporter_options.token_resolver` を 1 回だけ読む（ADR 0024 Decision 2）。

    利用者が渡す options は遅延解決の計算プロパティを持ちうるため、二重評価は副作用・
    レイテンシ・外部呼び出しを二重化する。上流より広い経路で評価する実装であることから、
    回数の下限契約は lib 側で固定しておく必要がある。

    検知する退行: `getattr` の前に `hasattr` 相当の判定を足して二重評価にする変異
    （警告は出続けるためメッセージ照合では検知できない）。
    """
    options = _CountingOptions()
    config = _tracing_config(token_resolver=_resolver, exporter_options=options)

    with pytest.warns(RuntimeWarning, match=_DROPPED_RESOLVER_PHRASE):
        obs.enable_agent365_tracing(config)

    assert options.reads == 1


class _ExplodingOptions:
    """`token_resolver` の読み取りが `AttributeError` 以外で失敗する options のスタブ。

    `__getattr__` ではなく property で失敗させるのは、実装が想定する現実のケース
    （利用者が渡す不透明値が遅延解決の計算プロパティを持ち、認証情報の不備等で例外になる）に
    最も近い形であり、かつ属性は「宣言上は存在する」ため、`AttributeError` のみを吸収する
    実装（却下した `hasattr` 案・ADR 0024 Decision 2）では例外が呼び出し元へ伝播することを
    示せるため。

    読み取り回数を数えるのは、例外経路でも属性を 1 回だけ読むことを固定するため。
    """

    def __init__(self) -> None:
        """読み取り回数カウンタを初期化する。"""
        self.reads = 0

    @property
    def token_resolver(self) -> Any:
        """読み取りのたびに回数を数え、`AttributeError` 以外の例外を送出する。"""
        self.reads += 1
        raise ValueError("token_resolver の解決に失敗しました")


def test_enable_agent365_tracing_does_not_raise_when_options_attribute_access_fails(
    tracing_spy: _TracingSpy, tracing: None
) -> None:
    """options の属性取得が例外を投げても伝播させず、無警告で計装まで到達する。

    観測の構成判定の失敗で利用者のアプリを停止させない（AC2）。属性値が取れない構成は
    「判定不能」であり、resolver がドロップされたとは断定できないため警告も出さず、番兵へ
    倒して処理を継続する（却下した `hasattr` 案は `AttributeError` のみを吸収するため、この形の
    値では例外が呼び出し元へ伝播していた）。

    検知する退行: 属性取得を囲む `try` / `except Exception` の削除（素の
    `getattr(config.exporter_options, "token_resolver", _MISSING)` へ戻す変異）。この変異では
    `ValueError` が `enable_agent365_tracing` から伝播して本テストが RED になる。
    """
    options = _ExplodingOptions()
    config = _tracing_config(token_resolver=_resolver, exporter_options=options)

    with warnings.catch_warnings():
        # 例外の伝播に加え、判定不能な構成での誤警告（本警告の発出）も同時に RED にする。
        warnings.simplefilter("error")
        obs.enable_agent365_tracing(config)

    # 例外経路でも読み取りは 1 回だけ（例外を握って再試行する退行を検知する）。
    assert options.reads == 1
    # 例外吸収後も処理が継続する（`configure` -> 計装まで到達する）。
    assert tracing_spy.calls == ["configure", "instrumentor_init", "instrument"]


def test_enable_agent365_tracing_logs_debug_when_options_attribute_access_fails(
    tracing_spy: _TracingSpy, tracing: None, caplog: pytest.LogCaptureFixture
) -> None:
    """属性取得の失敗は警告せず、痕跡を DEBUG ログ + トレースバックとして残す。

    警告にしないのは判定不能な構成へ誤警告しないため（ADR 0024）だが、無音で捨てると
    「到達先判定が行われなかった」事実がどこにも残らず、番兵へ倒れた結果が「判定対象外
    （sidecar 構成・options 未指定）」と外形上区別できない。既定レベルでは出力されない
    `DEBUG` に留めることで、誤警告を増やさずに追跡可能性だけを確保する。

    検知する退行: ログ記録の削除（無音化への逆戻り）・`exc_info=True` の欠落
    （原因の例外が失われる）・レベルの引き上げ（既定レベルへのノイズ混入）・型情報の欠落
    （複数の options 実装を持つ利用者がどの値で失敗したかを 1 行で特定できなくなる）。
    """
    config = _tracing_config(token_resolver=_resolver, exporter_options=_ExplodingOptions())

    with caplog.at_level(logging.DEBUG, logger=obs.__name__):
        obs.enable_agent365_tracing(config)

    records = [
        item
        for item in caplog.records
        if item.name == obs.__name__ and item.levelno == logging.DEBUG
    ]
    assert len(records) == 1, [item.getMessage() for item in caplog.records]
    # 原因の例外を追跡できること（`exc_info=True` の欠落を検知する）。
    assert records[0].exc_info is not None
    assert records[0].exc_info[0] is ValueError
    message = records[0].getMessage()
    assert "token_resolver" in message, message
    # どの options で失敗したかが 1 行で分かること（型情報が落ちる退行を検知する）。
    assert _ExplodingOptions.__name__ in message, message


# ----------------------------------------------------------------------
# enable_otel_logging: root logger への付与
# ----------------------------------------------------------------------


def test_enable_otel_logging_attaches_handler_to_root(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """root logger へ `LoggingHandler` を 1 個だけ付与し、生成した provider を結びつける。"""
    root = logging.getLogger()

    obs.enable_otel_logging(OtelLoggingConfig())

    added = [h for h in root.handlers if h not in root_handlers]
    assert len(added) == 1
    assert added[0] is otel.handlers[-1]
    assert added[0].logger_provider is otel.provider


def test_enable_otel_logging_passes_level_to_handler(otel: _OtelSpy, root_handlers: list) -> None:
    """`config.level` は付与する handler の level に反映される（既定 INFO 以外で検証）。"""
    obs.enable_otel_logging(OtelLoggingConfig(level=logging.DEBUG))

    assert otel.handlers[-1].level == logging.DEBUG


def test_enable_otel_logging_defaults_to_default_config(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """config 省略時は `OtelLoggingConfig()` 相当（INFO / OTLP なし / 整形済み出力）で結線する。"""
    obs.enable_otel_logging()

    assert otel.handlers[-1].level == logging.INFO
    assert otel.exporter_types() == ["_FakeConsoleExporter"]
    # config=None 経路も既定値定数（`_DEFAULT_LOG_CONSOLE_JSON_LINES=False`）を読み、
    # SDK 既定の整形済み出力のまま（formatter を渡さない）であることを固定する。
    assert "formatter" not in otel.exporter_of(_FakeConsoleExporter).kwargs


def test_adapter_default_constants_match_config_defaults() -> None:
    """`config=None` 用の既定値定数が `OtelLoggingConfig` の既定値と一致する。

    adapter 側の定数は単方向依存（コアから `runtime` を実行時 import しない）を保つための写しで
    あり、宣言側の既定値が変わったときに黙って乖離する drift を防ぐ。値の一致に加えて
    **フィールド集合**も等値で pin する: 値だけを見ていると、新フィールドが追加されたときに
    `config=None` 経路だけがその設定を無視する（写しに対応する定数が無い）drift を見逃す。
    集合比較なのでフィールドの追加・削除の両方向を同時に守る。
    """
    defaults = OtelLoggingConfig()

    assert obs._DEFAULT_LOG_SERVICE_NAME == defaults.service_name
    assert obs._DEFAULT_LOG_LEVEL == defaults.level
    assert obs._DEFAULT_LOG_OTLP_ENABLED == defaults.otlp_enabled
    assert obs._DEFAULT_LOG_CONSOLE_JSON_LINES == defaults.console_json_lines
    assert {f.name for f in dataclasses.fields(OtelLoggingConfig)} == {
        "service_name",
        "level",
        "otlp_enabled",
        "console_json_lines",
    }, "OtelLoggingConfig にフィールドを足したら _adapters 側の既定値定数も更新する"


# ----------------------------------------------------------------------
# enable_otel_logging: Resource（service.name）の載せ方
# ----------------------------------------------------------------------


def test_enable_otel_logging_builds_resource_from_service_name(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """`service_name` 指定時は `service.name` 属性の Resource を provider へ渡す。"""
    obs.enable_otel_logging(OtelLoggingConfig(service_name="svc-x"))

    assert len(otel.resources) == 1
    assert otel.resources[0].attributes == {"service.name": "svc-x"}
    assert otel.provider.kwargs["resource"] is otel.resources[0]


def test_enable_otel_logging_omits_resource_when_service_name_none(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """`service_name=None` では Resource を組まず provider へも渡さない（OTel 既定に委ねる）。"""
    obs.enable_otel_logging(OtelLoggingConfig(service_name=None))

    assert otel.resources == []
    assert "resource" not in otel.provider.kwargs


# ----------------------------------------------------------------------
# enable_otel_logging: OTLP 併用（置換ではなく追加）
# ----------------------------------------------------------------------


def test_enable_otel_logging_uses_console_only_by_default(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """`otlp_enabled=False` では processor が Console 1 本のみ。"""
    obs.enable_otel_logging(OtelLoggingConfig(otlp_enabled=False))

    assert otel.exporter_types() == ["_FakeConsoleExporter"]


def test_enable_otel_logging_adds_otlp_processor_additively(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """`otlp_enabled=True` は Console を置換せず OTLP を追加して 2 本構成にする。"""
    obs.enable_otel_logging(OtelLoggingConfig(otlp_enabled=True))

    types_ = otel.exporter_types()
    assert len(types_) == 2
    assert "_FakeConsoleExporter" in types_  # 置換されていない
    assert "_FakeOtlpExporter" in types_


# ----------------------------------------------------------------------
# enable_otel_logging: コンソール出力の 1 行 JSON 化（console_json_lines）
# ----------------------------------------------------------------------


def test_enable_otel_logging_console_exporter_has_no_formatter_by_default(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """`console_json_lines=False`（既定）では `ConsoleLogExporter` へ formatter を渡さない。

    既定の出力形式（`indent=4` の整形済み JSON）は SDK 側の既定に委ねる。lib が常に formatter を
    渡す変異は、SDK 既定の変更に追随できなくなるためここで RED にする。位置引数も検査するのは、
    `ConsoleLogExporter(out, formatter)` の第 1 引数が出力先（`out`）であり、位置で渡す実装は
    出力先を壊すため。
    """
    obs.enable_otel_logging(OtelLoggingConfig(console_json_lines=False))

    console = otel.exporter_of(_FakeConsoleExporter)
    assert console.args == ()
    assert "formatter" not in console.kwargs


def test_enable_otel_logging_console_json_lines_emits_single_line(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """`console_json_lines=True` では 1 レコードを 1 行の JSON へ落とす formatter を渡す。

    「formatter が渡されたか」だけを見ると、整形済み JSON を返す formatter でも通ってしまうため、
    受け取った formatter を実際に呼んで出力の**行数**を検証する。1 行 = 1 レコードが崩れると
    ログ収集基盤側で `trace_id` と `body` が別レコードに分かれて相関できなくなる。
    """
    obs.enable_otel_logging(OtelLoggingConfig(console_json_lines=True))

    formatter = otel.exporter_of(_FakeConsoleExporter).kwargs["formatter"]
    record = _FakeReadableLogRecord()
    output = formatter(record)

    # 末尾の改行 1 個だけを許し、レコード本体には改行を含まない（1 行 = 1 レコード）。
    assert output.endswith("\n")
    assert len(output.splitlines()) == 1
    assert "\n" not in output[:-1]
    # SDK 既定（indent=4）の整形済み出力ではないことを、同一レコードの既定出力と対比して固定する。
    assert len(record.to_json().splitlines()) > 1
    assert output != record.to_json() + "\n"
    # 1 行化しても JSON としての内容は失われない（改行除去等の力技への変異を検出する）。
    assert json.loads(output) == json.loads(record.to_json())


def test_enable_otel_logging_console_json_lines_does_not_affect_otlp_exporter(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """`console_json_lines=True` でも OTLP エクスポータへは formatter を渡さない。

    formatter はコンソール出力専用の整形手段であり、OTLP は独自のプロトコルで送出する
    （渡すと実 SDK では未知キーワードとして `TypeError` になる）。
    """
    obs.enable_otel_logging(OtelLoggingConfig(console_json_lines=True, otlp_enabled=True))

    otlp = otel.exporter_of(_FakeOtlpExporter)
    assert otlp.args == ()
    assert otlp.kwargs == {}
    # console 側には渡っている（両方に渡さない変異と区別する）。
    assert "formatter" in otel.exporter_of(_FakeConsoleExporter).kwargs


# ----------------------------------------------------------------------
# enable_otel_logging: 冪等・非接触
# ----------------------------------------------------------------------


def test_enable_otel_logging_is_idempotent(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """2 回呼んでも root への付与は 1 回だけ（重複付与・二重出力を防ぐ）。

    過大側（フラグ検査を外して毎回付与する変異）は handler 数 / provider 生成数の増加で、
    過小側（常に早期 return して一度も付与しない変異）は 1 回目の付与が消えることで RED になる。
    """
    root = logging.getLogger()

    obs.enable_otel_logging(OtelLoggingConfig())
    after_first = [h for h in root.handlers if h not in root_handlers]

    obs.enable_otel_logging(OtelLoggingConfig())
    after_second = [h for h in root.handlers if h not in root_handlers]

    assert len(after_first) == 1
    assert after_second == after_first  # 同一オブジェクトのまま増えていない
    assert len(otel.providers) == 1  # LoggerProvider も 1 度しか組まない


def test_enable_otel_logging_records_applied_settings(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """付与時に適用した設定 4 要素を記録する（再設定検知の比較対象）。

    要素数と並びを等値で pin する。フィールドを足したのに記録へ含め忘れると、その設定だけ
    再設定検知をすり抜ける（違う設定で呼んでも無警告になる）。
    """
    obs.enable_otel_logging(
        OtelLoggingConfig(
            service_name="svc-x", level=logging.DEBUG, otlp_enabled=True, console_json_lines=True
        )
    )

    assert obs._OTEL_LOG_APPLIED_SETTINGS == ("svc-x", logging.DEBUG, True, True)


def test_enable_otel_logging_same_settings_second_call_does_not_warn(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """同一設定での 2 回目は無警告で黙って return する（無条件 warn への変異を検出）。"""
    config = OtelLoggingConfig(service_name="svc-x", level=logging.DEBUG, otlp_enabled=True)
    obs.enable_otel_logging(config)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obs.enable_otel_logging(
            OtelLoggingConfig(service_name="svc-x", level=logging.DEBUG, otlp_enabled=True)
        )


def test_enable_otel_logging_omitted_config_matches_explicit_defaults(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """既定値で有効化した後に config 省略で呼んでも「設定が違う」と誤検知しない。

    `config=None` の既定（モジュール定数の写し）と `OtelLoggingConfig()` の既定が同じ設定として
    比較されることを固定する（写しの drift は再設定警告の誤発火として現れる）。
    """
    obs.enable_otel_logging(OtelLoggingConfig())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obs.enable_otel_logging()


@pytest.mark.parametrize(
    ("field_name", "changed"),
    [
        ("service_name", OtelLoggingConfig(service_name="other", level=logging.INFO)),
        ("level", OtelLoggingConfig(service_name="svc-x", level=logging.DEBUG)),
        ("otlp_enabled", OtelLoggingConfig(service_name="svc-x", otlp_enabled=True)),
        (
            "console_json_lines",
            OtelLoggingConfig(service_name="svc-x", console_json_lines=True),
        ),
    ],
)
def test_enable_otel_logging_different_settings_second_call_warns(
    otel: _OtelSpy,
    root_handlers: list[logging.Handler],
    field_name: str,
    changed: OtelLoggingConfig,
) -> None:
    """初回と異なる設定での 2 回目は RuntimeWarning で「適用されない」ことを通知する。

    4 フィールドを 1 つずつ変えて独立に検証するため、比較対象から 1 要素を落とす変異
    （例: `service_name` を比較しない）は該当ケースだけが RED になり、どのフィールドの
    検知が壊れたのかが特定できる。
    """
    obs.enable_otel_logging(OtelLoggingConfig(service_name="svc-x", level=logging.INFO))

    with pytest.warns(RuntimeWarning, match="既に有効化済み") as caught:
        obs.enable_otel_logging(changed)

    _assert_warnings_attributed_to_caller(caught)


def test_enable_otel_logging_different_settings_does_not_change_wiring(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """異なる設定で呼び直しても適用済みの結線は変わらない（後勝ちで上書きしない）。

    警告だけを出して return するため、handler の追加・provider の再生成・handler level の
    変更のいずれも起きてはならない。
    """
    root = logging.getLogger()
    obs.enable_otel_logging(OtelLoggingConfig(service_name="svc-x", level=logging.INFO))
    after_first = [h for h in root.handlers if h not in root_handlers]

    with pytest.warns(RuntimeWarning):
        obs.enable_otel_logging(
            OtelLoggingConfig(
                service_name="other",
                level=logging.DEBUG,
                otlp_enabled=True,
                console_json_lines=True,
            )
        )

    assert [h for h in root.handlers if h not in root_handlers] == after_first
    assert len(otel.providers) == 1
    assert otel.handlers[-1].level == logging.INFO  # 初回の level のまま
    assert otel.exporter_types() == ["_FakeConsoleExporter"]  # OTLP は追加されない
    # コンソールの出力形式も初回のまま（後から 1 行 JSON へ切り替わらない）。
    assert "formatter" not in otel.exporter_of(_FakeConsoleExporter).kwargs
    assert obs._OTEL_LOG_APPLIED_SETTINGS == ("svc-x", logging.INFO, False, False)


def test_enable_otel_logging_sets_attached_flag(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """付与後はモジュールフラグが True になる（冪等判定の状態が実際に立つ）。"""
    assert obs._OTEL_LOG_HANDLER_ATTACHED is False

    obs.enable_otel_logging(OtelLoggingConfig())

    assert obs._OTEL_LOG_HANDLER_ATTACHED is True


def test_enable_otel_logging_does_not_touch_existing_handlers(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """既存 handler・フォーマッタ・登録順は変更しない（追加のみ）。"""
    root = logging.getLogger()
    existing = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    existing.setFormatter(formatter)
    root.addHandler(existing)
    before = list(root.handlers)

    obs.enable_otel_logging(OtelLoggingConfig())

    # 既存 handler は同一オブジェクトのまま先頭側の順序を保って残る。
    assert root.handlers[: len(before)] == before
    assert existing in root.handlers
    assert existing.formatter is formatter


def test_enable_otel_logging_does_not_change_root_level(
    otel: _OtelSpy, root_handlers: list[logging.Handler]
) -> None:
    """root logger の level は呼び出し前後で不変（`config.level` は handler にのみ効く）。

    root へ `setLevel(config.level)` を足すと利用者の logging 構成を無断で書き換えることになる
    （FR-3 の「追加のみ」に反する）。config.level と異なる level を明示的に立ててから呼び、
    root 側が書き換わらないことを固定する（`root_handlers` fixture が teardown で復元する）。
    """
    root = logging.getLogger()
    root.setLevel(logging.ERROR)

    obs.enable_otel_logging(OtelLoggingConfig(level=logging.DEBUG))

    assert root.level == logging.ERROR
    # handler 側にだけ config.level が反映されている（結線自体は成立している）。
    assert otel.handlers[-1].level == logging.DEBUG


def test_importing_adapter_does_not_touch_root_logger() -> None:
    """モジュール import 自体は root logger へ触れない（import 副作用ゼロ・ADR 0022）。

    有効化はあくまで明示的な関数呼び出しでのみ起き、import では handler も level も変わらない。
    現プロセスは他テストの影響を受けるうえ、`importlib.reload` は再エクスポート先が保持する
    関数オブジェクトとの同一性を壊す（窓口テストを汚染する）ため、クリーンな子プロセスで
    import 前後を比較する。
    """
    probe = (
        "import logging\n"
        "root = logging.getLogger()\n"
        "before = (len(root.handlers), root.level)\n"
        "import oai_agentspec._adapters.observability  # noqa: F401\n"
        "after = (len(root.handlers), root.level)\n"
        "print('same' if before == after else f'{before} -> {after}')\n"
    )
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_SRC_DIR) + (os.pathsep + existing if existing else "")
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env
    )

    assert result.stdout.strip() == "same"


# ----------------------------------------------------------------------
# 未導入時の案内付き ImportError
# ----------------------------------------------------------------------


def test_require_agent365_tracing_missing_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent 365 拡張が未導入相当なら extra 案内付き ImportError を送出する。"""
    monkeypatch.setitem(sys.modules, "microsoft_agents_a365.observability.core", None)
    monkeypatch.setitem(sys.modules, "microsoft_agents_a365.observability.extensions.openai", None)

    with pytest.raises(ImportError, match=r"oai-agentspec\[observability\]"):
        obs._require_agent365_tracing()


def test_require_opentelemetry_missing_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """opentelemetry-sdk が未導入相当なら extra 案内付き ImportError を送出する。"""
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk._logs", None)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk._logs.export", None)

    with pytest.raises(ImportError, match=r"oai-agentspec\[observability\]"):
        obs._require_opentelemetry()


def test_require_otlp_log_exporter_missing_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OTLP エクスポータ配布物が未導入相当なら extra 案内付き ImportError を送出する。

    `opentelemetry-sdk` とは別配布物のため取得関数が分かれており、案内も独立している
    （どちらの経路でも extra のインストール手順に到達できることを固定する）。
    """
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http._log_exporter", None)

    with pytest.raises(ImportError, match=r"oai-agentspec\[observability\]"):
        obs._require_otlp_log_exporter()
