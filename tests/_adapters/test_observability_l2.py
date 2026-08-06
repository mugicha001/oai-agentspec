"""L2: observability 連携が依存する実 SDK 側の契約を pin する（有効化は行わない）。

`_adapters/observability` が前提にしている外部シンボルの形（`configure()` のキーワード名・
instrumentor の順序契約・OTel Logs の構築 API）を、実パッケージに対して検証する。L1 の spy は
これらの形を模しているため、SDK 側が変わったときに L1 が「模造品にだけ通る」状態へ静かに退行
するのを防ぐ。

**本プロセスでは有効化（`configure()` / `instrument()` の実呼び出し）を行わない**。Agent 365 の
`TelemetryManager` はプロセス singleton で一度 configure すると元に戻せず、`instrument()` は
`set_trace_processors([...])` で SDK の processor 列を置換するため、実行するとテストセッション
全体（`tests/workflow/test_tracing_l2.py` 等）を汚染する。本ファイルはシグネチャ・属性・
未設定時の例外に加え、グローバル状態へ触れない純粋な整形（コンソール出力の 1 行 JSON 化に使う
`_json_lines_formatter` と実レコードの `to_json()`）のみを実呼び出しで検査する。例外は
`..._ignores_top_level_token_resolver_when_exporter_options_given` のみで、上流の引数合成規則は
`configure()` を実際に呼ばないと確かめられないため**子プロセスへ隔離して**実行する（span は
生成しないので外部通信は発生しない）。observability extra 未導入環境では skip する。
"""

from __future__ import annotations

import inspect
import io
import json
import subprocess
import sys
from typing import Any

import pytest

from oai_agentspec._adapters import observability as obs

pytestmark = pytest.mark.unit

# 実装が `configure()` へパススルーするキーワード名（L1 の passthrough テストと同一集合）。
_CONFIGURE_KWARGS = (
    "service_name",
    "service_namespace",
    "logger_name",
    "token_resolver",
    "cluster_category",
    "exporter_options",
    "suppress_invoke_agent_input",
)


def _a365_core() -> object:
    """Agent 365 observability core モジュールを返す（extra 未導入なら skip）。"""
    return pytest.importorskip(
        "microsoft_agents_a365.observability.core",
        reason="observability extra（microsoft-agents-a365-observability-extensions-openai）未導入",
    )


def test_agent365_configure_accepts_documented_kwargs() -> None:
    """`configure()` が実装のパススルー対象 7 キーワードを受け付ける（名前 drift の検知）。"""
    core = _a365_core()
    params = inspect.signature(core.configure).parameters  # type: ignore[attr-defined]

    missing = [name for name in _CONFIGURE_KWARGS if name not in params]
    assert missing == [], f"configure() が受け付けないキーワードがあります: {missing}"


def test_agent365_core_exposes_is_configured() -> None:
    """`is_configured()` が公開されている（tracing 有効化前の状態判定に使う）。"""
    core = _a365_core()
    assert callable(core.is_configured)  # type: ignore[attr-defined]


def test_instrumentor_requires_configure_first() -> None:
    """未 configure 状態の instrumentor 生成は RuntimeError（順序契約は SDK 側にある）。

    L1 で pin している「configure が先」の順序が、実 SDK 側の強制に裏付けられていることを示す。
    本テストは configure を呼ばないため副作用はない。
    """
    core = _a365_core()
    if core.is_configured():  # type: ignore[attr-defined]  # pragma: no cover - 実行順依存
        pytest.skip("既に configure 済みのため未設定状態を検証できない")

    ext = pytest.importorskip(
        "microsoft_agents_a365.observability.extensions.openai",
        reason="observability extra 未導入",
    )
    with pytest.raises(RuntimeError):
        ext.OpenAIAgentsTraceInstrumentor()


def test_require_agent365_tracing_returns_sdk_symbols() -> None:
    """`_require_agent365_tracing()` が実 SDK の 3 シンボルを宣言順で返す（有効化はしない）。

    L1 の spy はこの 3 つ組の形を模しているため、実 SDK 側の名前・並びが変わったときに
    spy だけが通る状態へ退行しないよう、成功経路そのものを固定する。
    """
    _a365_core()  # extra 未導入環境では skip
    configure, is_configured, instrumentor_cls = obs._require_agent365_tracing()

    assert callable(configure)
    assert callable(is_configured)
    assert instrumentor_cls.__name__ == "OpenAIAgentsTraceInstrumentor"


def test_agent365_exporter_options_exposes_token_resolver() -> None:
    """`Agent365ExporterOptions` は公開属性 `token_resolver` を持ち、既定は `None`。

    実装の到達先判定は番兵付き `getattr(options, "token_resolver", _MISSING)` と値の照合だけに
    依存する（`isinstance` による型判別はしない）。属性の消失・改名は判定を無音で成立しなく
    するため、存在・既定値・コンストラクタ経由の設定の 3 点を実 SDK で固定する。

    検知する退行: 実装が読む属性名の drift、および pin が実際に属性存在を見ているか（M11）。
    """
    core = _a365_core()

    default_options = core.Agent365ExporterOptions()  # type: ignore[attr-defined]
    assert hasattr(default_options, "token_resolver")
    assert default_options.token_resolver is None

    async def resolver(scope: str, tenant: str) -> str | None:
        return None

    configured = core.Agent365ExporterOptions(token_resolver=resolver)  # type: ignore[attr-defined]
    assert configured.token_resolver is resolver


def test_spectra_exporter_options_has_no_token_resolver() -> None:
    """`SpectraExporterOptions` は `token_resolver` 属性を持たない（sidecar 構成の除外根拠）。

    実装は番兵付き `getattr` でこの差を見て Spectra を警告対象から外す。将来上流が Spectra 側へ
    当該属性を追加すると sidecar 構成で誤警告が始まるため、その前提をここで pin する。

    「存在しない属性の不在」だけを主張すると属性名 typo でも真になり無音成立するため、同じ
    リテラルを当該属性を持つ `Agent365ExporterOptions` へ正方向で当てる（M12）。
    """
    core = _a365_core()

    options = core.SpectraExporterOptions()  # type: ignore[attr-defined]

    # 同一リテラルを Agent365 形式へ正方向で当てる（属性名を typo すると此処が落ちる）。
    assert hasattr(core.Agent365ExporterOptions(), "token_resolver")  # type: ignore[attr-defined]
    assert not hasattr(options, "token_resolver")
    # 正の assert（Spectra 側の公開属性が実在することの担保）。
    assert hasattr(options, "endpoint")
    assert hasattr(options, "protocol")


# 上流の引数合成規則を確かめる子プロセス probe。`TelemetryManager` はプロセス singleton で一度
# configure すると元に戻せないため親プロセスでは実行しない。有効化フラグは子プロセス内で立てる
# （本体は環境変数を読まないため、テスト側も親プロセスの環境を汚さない）。span は生成しない。
_COMPOSITION_PROBE = """
import os
import sys

os.environ["ENABLE_A365_OBSERVABILITY_EXPORTER"] = "true"
os.environ.pop("ENABLE_OTLP_EXPORTER", None)

from microsoft_agents_a365.observability.core import configure
from microsoft_agents_a365.observability.core.config import TelemetryManager
from microsoft_agents_a365.observability.core.exporters import Agent365ExporterOptions


def resolver(scope, tenant):
    return "token"


kwargs = {"service_name": "svc", "service_namespace": "ns", "token_resolver": resolver}
if sys.argv[1] == "with_options":
    kwargs["exporter_options"] = Agent365ExporterOptions()

configure_ok = configure(**kwargs)
exporter = TelemetryManager()._span_processors["batch"].span_exporter
print(f"VERDICT:{configure_ok}:{type(exporter).__name__}")
"""


def _composition_probe(case: str) -> str:
    """子プロセスで `configure()` を 1 回呼び、`"<戻り値>:<採用 exporter のクラス名>"` を返す。

    上流のログや `ConsoleSpanExporter` の出力が混ざりうるため、標準出力から `VERDICT:` 行だけを
    取り出す（現行の上流のフォールバック WARNING は stderr へ出る）。

    Args:
        case: `"with_options"`（`Agent365ExporterOptions()` を併用）または `"without_options"`。

    Returns:
        `"True:ConsoleSpanExporter"` のような 1 行（`VERDICT:` 接頭辞を除いたもの）。
    """
    result = subprocess.run(
        [sys.executable, "-c", _COMPOSITION_PROBE, case],
        capture_output=True,
        text=True,
        check=False,
        # 上流が構築時のトークン取得・shutdown 時の flush を行う実装へ変わった場合に、
        # 「CI が止まる」ではなく「テストが落ちる」へ着地させる（drift 検知が目的の probe）。
        timeout=60,
    )
    assert result.returncode == 0, f"probe が失敗しました: {result.stderr}"
    verdicts = [
        line[len("VERDICT:") :]
        for line in result.stdout.splitlines()
        if line.startswith("VERDICT:")
    ]
    assert len(verdicts) == 1, result.stdout
    return verdicts[0]


def test_upstream_ignores_top_level_token_resolver_when_exporter_options_given() -> None:
    """上流は `exporter_options` を渡されるとトップレベル `token_resolver` を参照しない。

    これは `enable_agent365_tracing` の到達先未達警告が成立する唯一の前提（ADR 0024）である。
    上流依存はバージョン上限を持たないため、上流が両者を合成する実装へ変わった時点で本警告は
    「実サービスへ届いている構成に対する誤警告」へ反転する。その退行は lib 側のテストを緑のまま
    通過してしまうため、前提そのものをここで pin する。

    `configure()` の戻り値が両ケースで真であることも同時に固定する（戻り値は到達の証拠に
    ならない = 警告が唯一の検知手段である、という設計全体の前提）。

    上流内部（`TelemetryManager._span_processors`）を参照するのは、採用された exporter を知る
    公開 API が上流にも OpenTelemetry にも無いためで、上流 drift の検知を目的とした意図的な
    内部依存である。参照先が失われた場合は probe が非ゼロ終了して本テストが落ちる（無音で
    緑化しない）。
    """
    _a365_core()  # observability extra 未導入なら skip する

    with_options = _composition_probe("with_options")
    without_options = _composition_probe("without_options")

    # 正の対照: options を渡さなければトップレベル resolver は使われ実 exporter が選ばれる。
    # この assert が無いと、有効化フラグが効いていないだけの環境でも本テストが緑になる。
    assert without_options.startswith("True:"), without_options
    assert "Console" not in without_options, without_options
    # 本題: options を渡すとトップレベル resolver は捨てられ、コンソールへフォールバックする。
    assert with_options == "True:ConsoleSpanExporter", with_options


def test_require_opentelemetry_returns_documented_namespace() -> None:
    """`_require_opentelemetry()` が文書化された属性名で実クラスを返す。

    `LogRecordProcessor` の実体は `BatchLogRecordProcessor` に確定しているため、型同一性で
    照合する（Simple へ差し替える変異が RED になる）。`ConsoleLogExporter` 属性は SDK の
    新旧名称（`ConsoleLogRecordExporter` / 非推奨の `ConsoleLogExporter`）のどちらを採用するかが
    仕様のため、ここでは「実 SDK のどちらかの実体であること」に留め、採用順は後続の 2 件
    （新名称優先 / 旧名称フォールバック）が pin する。
    """
    export = pytest.importorskip(
        "opentelemetry.sdk._logs.export", reason="observability extra 未導入"
    )
    logs = pytest.importorskip("opentelemetry.sdk._logs", reason="observability extra 未導入")
    resources = pytest.importorskip(
        "opentelemetry.sdk.resources", reason="observability extra 未導入"
    )

    otel = obs._require_opentelemetry()

    assert otel.LoggerProvider is logs.LoggerProvider
    assert otel.LoggingHandler is logs.LoggingHandler
    assert otel.ConsoleLogExporter in {
        getattr(export, "ConsoleLogRecordExporter", None),
        export.ConsoleLogExporter,
    }
    assert otel.LogRecordProcessor is export.BatchLogRecordProcessor
    assert otel.Resource is resources.Resource


def test_require_opentelemetry_prefers_console_log_record_exporter() -> None:
    """後継の `ConsoleLogRecordExporter` がある版では、非推奨の旧名ではなく新名称を採用する。

    旧名 `ConsoleLogExporter` は将来削除予定の非推奨エイリアスで、構築のたびに
    `DeprecationWarning` を出す。利用者が `enable_otel_logging()` を呼ぶだけで警告が出る状態を
    避けるため、新名称が存在する環境では必ずそちらを使う（「常に旧名を返す」変異が RED になる）。
    """
    export = pytest.importorskip(
        "opentelemetry.sdk._logs.export", reason="observability extra 未導入"
    )
    if not hasattr(export, "ConsoleLogRecordExporter"):  # pragma: no cover - 旧 SDK 版でのみ通る
        pytest.skip("この opentelemetry-sdk には後継の ConsoleLogRecordExporter が無い")

    otel = obs._require_opentelemetry()

    assert otel.ConsoleLogExporter is export.ConsoleLogRecordExporter
    # 非推奨エイリアスは別クラスであり、取り違えると警告が復活する。
    assert otel.ConsoleLogExporter is not export.ConsoleLogExporter


def test_require_opentelemetry_falls_back_to_legacy_console_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """後継が無い版（extra 下限 `opentelemetry-sdk>=1.36` 側）では旧名へフォールバックする。

    新名称の不在を `delattr` で模す。フォールバックを失った実装（新名称を直接参照する変異）は
    ここで `AttributeError` になるため、下限バージョンの環境を用意せずに退行を検知できる。
    """
    export = pytest.importorskip(
        "opentelemetry.sdk._logs.export", reason="observability extra 未導入"
    )
    monkeypatch.delattr(export, "ConsoleLogRecordExporter", raising=False)

    otel = obs._require_opentelemetry()

    assert otel.ConsoleLogExporter is export.ConsoleLogExporter


def test_require_opentelemetry_survives_legacy_console_exporter_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧名が削除された将来版でも、新名称があれば解決できる（`AttributeError` にしない）。

    上流は旧名 `ConsoleLogExporter` を「将来のリリースで削除する」と明言しており、extra の
    上限を張らない（`opentelemetry-sdk>=1.36`）ため削除版が入りうる。旧名の不在を `delattr` で
    模す。`getattr(export, "新名称", export.ConsoleLogExporter)` のように**既定値を先行評価**する
    実装はこの時点で `AttributeError` になる（`try` の外なので案内付き `ImportError` にも
    変換されない）ため、削除版を用意せずに退行を検知できる。

    前掲の `..._falls_back_to_legacy_console_exporter` が「新名称なし」方向を pin するのに対し、
    本テストは「旧名なし」方向を pin する（両方向が揃って初めてフォールバックの契約が成立する）。
    """
    export = pytest.importorskip(
        "opentelemetry.sdk._logs.export", reason="observability extra 未導入"
    )
    if not hasattr(export, "ConsoleLogRecordExporter"):  # pragma: no cover - 旧 SDK 版でのみ通る
        pytest.skip("この opentelemetry-sdk には後継の ConsoleLogRecordExporter が無い")
    monkeypatch.delattr(export, "ConsoleLogExporter", raising=False)

    otel = obs._require_opentelemetry()

    assert otel.ConsoleLogExporter is export.ConsoleLogRecordExporter


def test_require_otlp_log_exporter_returns_sdk_class() -> None:
    """`_require_otlp_log_exporter()` が OTLP ログエクスポータの実クラスを返す。

    `opentelemetry-sdk` とは別配布物のため取得関数が分かれている。L1 のスタブが模している形
    （クラスを 1 つ返す）が実 SDK と一致することを固定する。
    """
    module = pytest.importorskip(
        "opentelemetry.exporter.otlp.proto.http._log_exporter",
        reason="opentelemetry-exporter-otlp-proto-http 未導入",
    )

    assert obs._require_otlp_log_exporter() is module.OTLPLogExporter


def test_otel_logs_sdk_exposes_construction_api() -> None:
    """OTel Logs 側の構築 API（provider / handler / console exporter）が揃っている。"""
    logs = pytest.importorskip("opentelemetry.sdk._logs", reason="observability extra 未導入")
    export = pytest.importorskip(
        "opentelemetry.sdk._logs.export", reason="observability extra 未導入"
    )

    assert hasattr(logs.LoggerProvider, "add_log_record_processor")
    assert hasattr(export, "ConsoleLogExporter")
    # Batch / Simple のどちらを使うかは実装の選択に委ねるが、少なくとも一方は存在する。
    assert hasattr(export, "BatchLogRecordProcessor") or hasattr(export, "SimpleLogRecordProcessor")


def test_otel_logging_handler_accepts_documented_kwargs() -> None:
    """`LoggingHandler` が `level` / `logger_provider` を受け付ける（L1 スタブと同形）。"""
    logs = pytest.importorskip("opentelemetry.sdk._logs", reason="observability extra 未導入")
    params = inspect.signature(logs.LoggingHandler).parameters

    assert "level" in params
    assert "logger_provider" in params


# ----------------------------------------------------------------------
# コンソール出力の 1 行 JSON 化（console_json_lines）が前提にする実 SDK 契約
# ----------------------------------------------------------------------


def _make_log_record(body: str) -> Any:
    """コンソール formatter が受け取るレコードを実 SDK の型で 1 件組む（未導入なら skip）。

    レコード型は SDK 1.4x で `LogRecord` -> `ReadableLogRecord`（API 側 `LogRecord` を包む形）へ
    再編されたため、対応 range（`opentelemetry-sdk>=1.36`）の両形をコンストラクタのシグネチャで
    判別して組む。型名・構築形が両方とも失われたら本ヘルパが失敗し、drift として検知される。

    Args:
        body: レコードの本文（出力の同一性確認に使う）。

    Returns:
        `to_json()` を持つ実 SDK のログレコード。
    """
    logs = pytest.importorskip("opentelemetry.sdk._logs", reason="observability extra 未導入")
    resources = pytest.importorskip(
        "opentelemetry.sdk.resources", reason="observability extra 未導入"
    )
    record_cls = getattr(logs, "ReadableLogRecord", None) or getattr(logs, "LogRecord", None)
    assert record_cls is not None, "SDK にログレコード型が見つかりません（名前 drift）"

    resource = resources.Resource.create({"service.name": "svc-x"})
    if "log_record" in inspect.signature(record_cls).parameters:
        api_logs = pytest.importorskip("opentelemetry._logs", reason="observability extra 未導入")
        return record_cls(log_record=api_logs.LogRecord(body=body), resource=resource)
    return record_cls(body=body, resource=resource)


def test_console_log_exporter_writes_one_line_per_record_with_json_lines_formatter() -> None:
    """`formatter=` で渡した 1 行化関数が、実 `ConsoleLogExporter` の出力に反映される。

    実装が `console_json_lines=True` のときに使う差し込み口（`formatter` キーワード）と、その
    結果として「1 レコード = 1 行」になることを実 SDK で固定する。L1 は exporter がスタブで
    `**kwargs` を受けるため、キーワード名の消失・formatter を使わなくなる変更に気づけない。
    出力先は `out=` で手元のバッファへ向けるためグローバル状態には触れない。

    検証対象は**実装が実際に採用する実体**（`_require_opentelemetry().ConsoleLogExporter`）とする。
    旧名を直接構築すると、実装が新名称へ移った後も旧名の契約だけを見続けることになるため。
    """
    pytest.importorskip("opentelemetry.sdk._logs.export", reason="observability extra 未導入")
    console_exporter_cls = obs._require_opentelemetry().ConsoleLogExporter
    record = _make_log_record("single-line test")

    buffer = io.StringIO()
    console_exporter_cls(out=buffer, formatter=obs._json_lines_formatter).export([record, record])

    lines = buffer.getvalue().splitlines()
    assert len(lines) == 2, f"1 レコード = 1 行になっていません: {buffer.getvalue()!r}"
    assert json.loads(lines[0])["body"] == "single-line test"


def test_console_log_exporter_default_output_is_indented() -> None:
    """既定（formatter 未指定）の出力は 1 レコードが複数行になる（本機能の動機の裏取り）。

    「既定では 1 行 = 1 レコードにならない」という前提が崩れたら `console_json_lines` の存在意義が
    変わるため、SDK 側の既定をここで観測しておく（`console_json_lines=False` が現行出力を維持する
    という後方互換の約束もこの前提の上に立つ）。検証対象は上と同じく実装が採用する実体とする。
    """
    pytest.importorskip("opentelemetry.sdk._logs.export", reason="observability extra 未導入")
    console_exporter_cls = obs._require_opentelemetry().ConsoleLogExporter
    record = _make_log_record("single-line test")

    buffer = io.StringIO()
    console_exporter_cls(out=buffer).export([record])

    assert len(buffer.getvalue().splitlines()) > 1


def test_readable_log_record_to_json_indent_contract() -> None:
    """`to_json(indent=None)` は 1 行・既定（`indent=4`）は複数行（L1 スタブが模す前提）。

    「SDK 既定のコンソール出力は複数行」という前提が本機能の動機そのものであり、L1 の
    `_FakeReadableLogRecord` はこの indent 反映を模している。SDK 側が既定を 1 行へ変えたり
    `indent` を無視するようになったら、ここが RED になって L1 スタブの更新を促す。
    """
    record = _make_log_record("single-line test")

    single = record.to_json(indent=None)
    default = record.to_json()

    assert len(single.splitlines()) == 1
    assert len(default.splitlines()) > 1
    # 整形の有無だけの違いで、JSON としての内容は同一。
    assert json.loads(single) == json.loads(default)


def test_json_lines_formatter_emits_single_line_for_real_record() -> None:
    """実装の `_json_lines_formatter` が実 SDK のレコードに対して 1 行 JSON を返す。

    L1 は formatter をスタブレコードで呼ぶため、実レコードの属性・メソッド名が変わっても
    気づけない。1 行 = 1 レコードで取り込むログ収集基盤向けの契約を実物で固定する
    （整形のみでグローバル状態には触れないため、L2 でも実呼び出しできる）。
    """
    pytest.importorskip("opentelemetry.sdk._logs", reason="observability extra 未導入")
    record = _make_log_record("single-line test")

    output = obs._json_lines_formatter(record)

    assert output.endswith("\n")
    assert len(output.splitlines()) == 1
    assert json.loads(output)["body"] == "single-line test"
