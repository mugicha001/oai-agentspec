"""L2: 宣言的ガードレール登録簿（`registry`）の SDK 結合検証（上流 4 型 + `RunConfig`）。

facade 経路（生成 + 登録 + 境界導出）・`factories` への代理呼び出し・`HELPER_DEFAULTS` 由来の
labels / severity 自動付与・`register` 経路の検証順 6 段・照会 6 メソッド・`run_config_kwargs()`
を検証する。実体は `_adapters.guardrails` の builder で作った上流 SDK guardrail 4 型を使い、
登録キーと上流可視名の一致（`guardrail_visible_name`）と `RunConfig(**kwargs)` の構築可能性まで
突き合わせる（実 LLM は呼ばず検知器は常に非検知の plain 関数）。
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

import pytest
from agents import (
    InputGuardrail,
    OutputGuardrail,
    RunConfig,
    ToolInputGuardrail,
    ToolOutputGuardrail,
)

from oai_agentspec._adapters.guardrails import (
    build_input_guardrail,
    build_output_guardrail,
    build_tool_input_guardrail,
    build_tool_output_guardrail,
    guardrail_boundary,
    guardrail_visible_name,
)
from oai_agentspec.runtime.guardrails import factories
from oai_agentspec.runtime.guardrails._detectors import Detection
from oai_agentspec.runtime.guardrails.catalog import DI_DEPENDENT_HELPERS, HELPER_DEFAULTS
from oai_agentspec.runtime.guardrails.registry import GuardrailRegistry
from oai_agentspec.runtime.guardrails.types import Boundary, GuardrailSpec, Severity

pytestmark = pytest.mark.integration


# ----------------------------------------------------------------------
# テスト用ヘルパ（検知器 / facade ディスパッチャ / 登録可能な宣言の生成）
# ----------------------------------------------------------------------

#: `on` を取る agent 境界 facade（`Boundary(on)` で INPUT / OUTPUT を導出する 6 件）。
ON_METHODS = (
    "prompt_llm_guardrail",
    "predicate_guardrail",
    "regex_guardrail",
    "length_guardrail",
    "allow_deny_guardrail",
    "external_detector_guardrail",
)

#: 境界が helper 自体で固定される agent 境界 facade（2 件）。
FIXED_BOUNDARY_METHODS = (
    ("injection_baseline_guardrail", Boundary.INPUT),
    ("canary_guardrail", Boundary.OUTPUT),
)

#: agent 境界 guardrail を返す facade（8 件）。
AGENT_BOUNDARY_METHODS = (*ON_METHODS, "injection_baseline_guardrail", "canary_guardrail")

#: facade 全件（agent 境界 8 + ツール境界 1）。
FACADE_METHODS = (*AGENT_BOUNDARY_METHODS, "tool_guardrail")


def _detect(text: str) -> Detection:
    """常に非検知を返す plain 検知器（実体生成用・実 LLM / 外部 I/O を持たない）。"""
    return Detection(triggered=False)


def _never(text: str) -> bool:
    """常に False を返す述語（`predicate_guardrail` 用）。"""
    return False


class _DuckGuardrail:
    """`get_name()` だけを持つ上流 4 型でないオブジェクト（登録が拒否されるべき実体）。"""

    def get_name(self) -> str:
        """上流 4 型と同じ可視名 API だけを模倣する。"""
        return "duck"


_BUILDERS = {
    Boundary.INPUT: build_input_guardrail,
    Boundary.OUTPUT: build_output_guardrail,
    Boundary.TOOL_INPUT: build_tool_input_guardrail,
    Boundary.TOOL_OUTPUT: build_tool_output_guardrail,
}


def _entity(name: str, boundary: Boundary | str) -> Any:
    """指定境界の上流 SDK guardrail 実体を可視名 `name` で作る。"""
    return _BUILDERS[Boundary(boundary)](name, _detect)


def _spec(name: str, boundary: Boundary | str = Boundary.INPUT, **fields: Any) -> GuardrailSpec:
    """実体と宣言（可視名 / 境界）が整合する `GuardrailSpec` を組む。

    Args:
        name: 登録名（実体の可視名にも同じ値を使う）。
        boundary: 宣言境界（`Boundary` メンバまたは値域内の文字列）。
        **fields: `labels` / `severity` の上書き指定。

    Returns:
        `register()` が受理できる `GuardrailSpec`。
    """
    return GuardrailSpec(name=name, boundary=boundary, guardrail=_entity(name, boundary), **fields)


def _invoke_facade(
    reg: GuardrailRegistry, method: str, name: str, *, on: str = "input", **extra: Any
) -> GuardrailSpec:
    """facade メソッドを対応 factory の必須引数つきで呼ぶ（テスト用ディスパッチャ）。

    境界が固定の 2 件（`injection_baseline_guardrail` / `canary_guardrail`）では `on` を
    渡さない（facade のシグネチャに存在しないため）。

    Args:
        reg: 対象の登録簿。
        method: facade メソッド名。
        name: 登録名（キーワード必須引数として渡す）。
        on: 適用境界（`on` を取る facade のみ使う）。
        **extra: `labels` / `severity` の指定を素通しする。

    Returns:
        facade の戻り値（`GuardrailSpec`）。
    """
    if method == "prompt_llm_guardrail":
        return reg.prompt_llm_guardrail("judge-model", "judge prompt", on=on, name=name, **extra)
    if method == "predicate_guardrail":
        return reg.predicate_guardrail(_never, on=on, name=name, **extra)
    if method == "regex_guardrail":
        return reg.regex_guardrail(r"\d+", on=on, name=name, **extra)
    if method == "length_guardrail":
        return reg.length_guardrail(max_length=10, on=on, name=name, **extra)
    if method == "allow_deny_guardrail":
        return reg.allow_deny_guardrail(deny=["bad"], on=on, name=name, **extra)
    if method == "external_detector_guardrail":
        return reg.external_detector_guardrail(_detect, on=on, name=name, **extra)
    if method == "injection_baseline_guardrail":
        return reg.injection_baseline_guardrail(name=name, **extra)
    if method == "canary_guardrail":
        return reg.canary_guardrail("LEAK", name=name, **extra)
    if method == "tool_guardrail":
        return reg.tool_guardrail(_detect, on=on, name=name, **extra)
    raise AssertionError(f"未知の facade メソッド: {method}")


# ======================================================================
# A. facade 経路（生成 + 登録 + 境界導出）
# ======================================================================


def test_facade9メソッドはGuardrailSpecを返し登録名がnamesに現れる() -> None:
    """facade 9 件はいずれも `GuardrailSpec` を返し、登録名が `names()` に現れる。

    `names()` を集合ではなく昇順リストの `==` で pin する（登録漏れ = 過小と余分な登録 =
    過大の両方向を同時に検知するため）。
    """
    reg = GuardrailRegistry()
    for method in FACADE_METHODS:
        spec = _invoke_facade(reg, method, f"g_{method}")
        assert isinstance(spec, GuardrailSpec)
        assert spec.name == f"g_{method}"
    assert reg.names() == sorted(f"g_{method}" for method in FACADE_METHODS)


@pytest.mark.parametrize(
    ("on", "expected"), [("input", Boundary.INPUT), ("output", Boundary.OUTPUT)]
)
@pytest.mark.parametrize("method", ON_METHODS)
def test_on付きagent境界facadeはonから境界を導出する(
    method: str, on: str, expected: Boundary
) -> None:
    """`on` を取る agent 境界 facade 6 件は `on` から INPUT / OUTPUT を導出する。

    `metadata()` の `boundary` が `Boundary` メンバへ正規化されていることを `is` で pin する
    （素の文字列が保持されていると `str, Enum` の等価比較で素通りするため）。
    """
    reg = GuardrailRegistry()
    _invoke_facade(reg, method, "g", on=on)
    assert reg.metadata("g").boundary is expected


@pytest.mark.parametrize(("method", "expected"), FIXED_BOUNDARY_METHODS)
def test_固定境界facadeは境界をhelperごとに固定する(method: str, expected: Boundary) -> None:
    """`injection_baseline_guardrail` は INPUT、`canary_guardrail` は OUTPUT に固定される。"""
    reg = GuardrailRegistry()
    _invoke_facade(reg, method, "g")
    assert reg.metadata("g").boundary is expected


@pytest.mark.parametrize(
    ("on", "expected"),
    [("input", Boundary.TOOL_INPUT), ("output", Boundary.TOOL_OUTPUT)],
)
def test_tool_guardrailはonからツール境界を導出する(on: str, expected: Boundary) -> None:
    """`tool_guardrail` は `on="input"` で TOOL_INPUT、`on="output"` で TOOL_OUTPUT。"""
    reg = GuardrailRegistry()
    reg.tool_guardrail(_detect, on=on, name="g")
    assert reg.metadata("g").boundary is expected


@pytest.mark.parametrize("on", ["input", "output"])
@pytest.mark.parametrize("method", FACADE_METHODS)
def test_facadeの宣言境界は実体の境界と一致する(method: str, on: str) -> None:
    """facade 9 件すべてで宣言 `boundary` が実体から判定した境界と一致する。

    `register` 経路はこの一致を検証するが、facade 経路は factory の `on` 契約に依拠した導出を
    信頼しており cross-check がない。導出（固定値 / `Boundary(on)` / ツール境界の三項）と factory
    側の `on` 値域が将来食い違うと、出力を検査するはずの宣言が入力側へ結線される silent な
    取り違えが成立するため、register 経路と同じ不変条件を facade 経路にも機械的に課す。
    """
    reg = GuardrailRegistry()
    _invoke_facade(reg, method, "g", on=on)
    spec = reg.metadata("g")
    assert guardrail_boundary(spec.guardrail) == spec.boundary.value


@pytest.mark.parametrize("on", ["input", "output"])
@pytest.mark.parametrize("method", FACADE_METHODS)
def test_facadeは登録キーと上流可視名が一致する(method: str, on: str) -> None:
    """facade 9 件は登録キーを factory の `name` へ注入し可視名を一致させる。

    可視名と照合キーの食い違いによる silent no-op を構造的に排除する不変条件の pin。ツール境界
    （`tool_guardrail`）を対象から外すと、当該 facade だけ `name` 注入を落とす変異が生存する
    （実体の可視名が factory の既定名になり、trip イベントの `get_name()` から `metadata()` を
    引く利用が `KeyError` になる）。facade が `register()` の実体整合突合を再利用しない設計の
    前提は、9 件すべてを機械的に押さえて初めて成立する。
    """
    reg = GuardrailRegistry()
    _invoke_facade(reg, method, "declared", on=on)
    assert guardrail_visible_name(reg.get("declared")) == "declared"


def test_facadeはfactory由来のValueErrorを伝播し登録を残さない() -> None:
    """対応 factory が `ValueError` を上げる引数（`on` 不正）では例外が伝播し登録が増えない。

    facade が「実体生成 → 登録」の順で動き、生成失敗時に部分適用（名前だけの登録）を
    残さないことを pin する。
    """
    reg = GuardrailRegistry()
    _invoke_facade(reg, "regex_guardrail", "keep")
    before = reg.names()
    with pytest.raises(ValueError):
        reg.regex_guardrail(r"\d+", on="invalid", name="broken")
    assert reg.names() == before


def test_facadeはfactoryの引数不備でも部分適用を残さない() -> None:
    """`length_guardrail` の閾値未指定（factory の `ValueError`）でも登録は空のまま。"""
    reg = GuardrailRegistry()
    with pytest.raises(ValueError):
        reg.length_guardrail(on="input", name="broken")
    assert reg.names() == []


def test_同名facadeの2回目はValueErrorで登録は1件のまま() -> None:
    """同じ `name` で 2 回呼ぶと `ValueError` になり、登録は 1 件のまま（一意性の強制）。"""
    reg = GuardrailRegistry()
    reg.regex_guardrail(r"\d+", on="input", name="dup")
    with pytest.raises(ValueError):
        reg.canary_guardrail("LEAK", name="dup")
    assert reg.names() == ["dup"]


@pytest.mark.parametrize("name", [123, "", "   "])
def test_facadeのname値域外はValueErrorで登録されない(name: Any) -> None:
    """非 str / 空文字 / 空白文字のみの `name` は `ValueError` で、登録は空のまま。"""
    reg = GuardrailRegistry()
    with pytest.raises(ValueError):
        reg.regex_guardrail(r"\d+", on="input", name=name)
    assert reg.names() == []


@pytest.mark.parametrize("severity", ["high", 3, object()])
def test_facadeのseverity値域外はValueErrorで登録されない(severity: Any) -> None:
    """`Severity` メンバでも None でもない `severity` は `ValueError`（登録は空のまま）。

    `Severity` は `IntEnum` のため素の int（`3`）が比較で素通りしうる。文字列 / int /
    任意オブジェクトの 3 種を明示的に拒否対象として pin する。
    """
    reg = GuardrailRegistry()
    with pytest.raises(ValueError):
        reg.regex_guardrail(r"\d+", on="input", name="g", severity=severity)
    assert reg.names() == []


# ======================================================================
# B. NFR-4 代理呼び出しの spy 計測
# ======================================================================


def _install_spy(monkeypatch: pytest.MonkeyPatch, method: str) -> list[Any]:
    """`factories.<method>` を戻り値を記録する同期 spy へ差し替え、記録先を返す。

    実体は退避した本物の factory の戻り値を返すため、facade の後段（登録・検証）は
    本番と同じ経路を通る。

    Args:
        monkeypatch: pytest の monkeypatch フィクスチャ。
        method: 差し替える factory 名。

    Returns:
        spy が返した値を呼び出し順に積むリスト（長さが呼び出し回数）。
    """
    original = getattr(factories, method)
    returned: list[Any] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        returned.append(result)
        return result

    monkeypatch.setattr(factories, method, spy)
    return returned


@pytest.mark.parametrize("method", ["canary_guardrail", "regex_guardrail"])
def test_facadeはfactoriesのモジュール属性経由で1回だけ代理呼び出しする(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """facade は `factories` のモジュール属性を 1 回だけ呼び、その戻り値を登録する。

    `from ... import <factory>` で束縛済みのローカル名を使う実装は monkeypatch が効かず
    呼び出し回数 0 になるため、モジュール属性経由の代理呼び出しという実装制約を pin できる。
    """
    returned = _install_spy(monkeypatch, method)
    reg = GuardrailRegistry()
    _invoke_facade(reg, method, "g")
    assert len(returned) == 1
    assert reg.get("g") is returned[0]


# ======================================================================
# C. labels / severity の自動付与（HELPER_DEFAULTS）
# ======================================================================


@pytest.mark.parametrize(
    ("method", "labels", "severity"),
    [
        ("canary_guardrail", {"owasp_llm": "LLM07"}, Severity.HIGH),
        ("injection_baseline_guardrail", {"owasp_llm": "LLM01"}, Severity.MEDIUM),
    ],
)
def test_既定分類を持つfacadeはlabelsとseverityを自動付与する(
    method: str, labels: dict[str, str], severity: Severity
) -> None:
    """`HELPER_DEFAULTS` にキーを持つ facade は labels / severity が自動付与される。"""
    reg = GuardrailRegistry()
    spec = _invoke_facade(reg, method, "g")
    assert spec.labels == labels
    assert spec.severity is severity


def test_labelsの自動付与はキー単位マージで利用者宣言が未指定キーを上書きしない() -> None:
    """利用者 `labels` は既定 labels とキー単位でマージされ、既定キーは残る。"""
    reg = GuardrailRegistry()
    spec = reg.canary_guardrail("LEAK", name="g", labels={"team": "sec"})
    assert spec.labels == {"owasp_llm": "LLM07", "team": "sec"}


def test_labelsの同一キーは利用者宣言が既定に優先する() -> None:
    """既定と同じキーを利用者が宣言した場合は利用者値が勝つ。"""
    reg = GuardrailRegistry()
    spec = reg.canary_guardrail("LEAK", name="g", labels={"owasp_llm": "X"})
    assert spec.labels == {"owasp_llm": "X"}


def test_自動付与されたlabelsは毎回新しいdictでHELPER_DEFAULTSを共有しない() -> None:
    """自動付与 labels は登録ごとに新規 dict で、既定データと実体を共有しない。

    `GuardrailSpec` は frozen でも `labels` のキー更新は通るため、共有していると 1 件の
    宣言へのラベル追記が既定データと他の宣言へ波及する。
    """
    reg = GuardrailRegistry()
    first = reg.canary_guardrail("LEAK", name="g1")
    second = reg.canary_guardrail("LEAK", name="g2")
    assert first.labels is not second.labels
    first.labels["owasp_llm"] = "mutated"
    first.labels["team"] = "sec"
    assert dict(HELPER_DEFAULTS["canary_guardrail"].labels) == {"owasp_llm": "LLM07"}
    assert second.labels == {"owasp_llm": "LLM07"}


def test_severityのNone明示は未指定と同一で既定が自動付与される() -> None:
    """`severity=None` の明示は未指定と同じ扱いで、既定 severity が付与される。"""
    reg = GuardrailRegistry()
    explicit = reg.canary_guardrail("LEAK", name="g1", severity=None)
    implicit = reg.canary_guardrail("LEAK", name="g2")
    assert explicit.severity is Severity.HIGH
    assert implicit.severity is Severity.HIGH


def test_severityのメンバ明示は既定の自動付与に優先する() -> None:
    """`Severity` メンバを明示した場合は既定 severity を上書きする。"""
    reg = GuardrailRegistry()
    spec = reg.canary_guardrail("LEAK", name="g", severity=Severity.LOW)
    assert spec.severity is Severity.LOW


@pytest.mark.parametrize("method", sorted(DI_DEPENDENT_HELPERS))
def test_DI依存facadeはlabels空でseverity未設定になる(method: str) -> None:
    """`HELPER_DEFAULTS` にキーを持たない DI 依存 helper 7 件は自動付与を受けない。"""
    reg = GuardrailRegistry()
    spec = _invoke_facade(reg, method, "g")
    assert spec.labels == {}
    assert spec.severity is None


def test_register経路ではlabelsとseverityを自動付与しない() -> None:
    """`register()` は helper 名と一致する登録名でも自動付与せず宣言をそのまま保持する。

    自動付与が「facade がどの helper 由来かを知っている」ことに基づく機構であり、登録名の
    文字列照合ではないことを pin する。
    """
    reg = GuardrailRegistry()
    spec = reg.register(_spec("canary_guardrail", Boundary.OUTPUT))
    assert spec.labels == {}
    assert spec.severity is None


# ======================================================================
# D. register 経路（検証順 6 段）
# ======================================================================


def test_register検証順はnameがboundaryより先() -> None:
    """`name` 値域違反と `boundary` 値域違反が混在したら `name` 由来の例外になる。"""
    reg = GuardrailRegistry()
    bad = GuardrailSpec(name="", boundary="bogus", guardrail=_entity("x", Boundary.INPUT))
    with pytest.raises(ValueError) as exc:
        reg.register(bad)
    message = str(exc.value).lower()
    assert "name" in message
    assert "boundary" not in message


def test_register検証順はboundaryがseverityより先() -> None:
    """`boundary` 値域違反と `severity` 値域違反が混在したら `boundary` 由来の例外になる。"""
    reg = GuardrailRegistry()
    bad = GuardrailSpec(
        name="g", boundary="bogus", guardrail=_entity("g", Boundary.INPUT), severity="high"
    )
    with pytest.raises(ValueError) as exc:
        reg.register(bad)
    message = str(exc.value).lower()
    assert "boundary" in message
    assert "severity" not in message


def test_register検証順はseverityがguardrail非Noneより先() -> None:
    """`severity` 値域違反と `guardrail=None` が混在したら `severity` 由来の例外になる。"""
    reg = GuardrailRegistry()
    bad = GuardrailSpec(name="g", boundary=Boundary.INPUT, guardrail=None, severity="high")
    with pytest.raises(ValueError) as exc:
        reg.register(bad)
    assert "severity" in str(exc.value).lower()


def test_register検証順はguardrail非Noneが重複名より先() -> None:
    """`guardrail=None` と重複名が混在したら `guardrail` 由来の例外になる。

    重複名の文言（`already` を含む）が現れないことで、後段の重複検査より前に落ちたことを
    判定する（両段の文言がともに `guardrail` を含みうるため）。
    """
    reg = GuardrailRegistry()
    reg.register(_spec("dup", Boundary.INPUT))
    bad = GuardrailSpec(name="dup", boundary=Boundary.INPUT, guardrail=None)
    with pytest.raises(ValueError) as exc:
        reg.register(bad)
    assert "already" not in str(exc.value).lower()


def test_register検証順は重複名が実体整合より先() -> None:
    """重複名と実体整合違反（可視名不一致）が混在したら重複名由来の例外になる。"""
    reg = GuardrailRegistry()
    reg.register(_spec("dup", Boundary.INPUT))
    bad = GuardrailSpec(
        name="dup", boundary=Boundary.INPUT, guardrail=_entity("actual", Boundary.INPUT)
    )
    with pytest.raises(ValueError) as exc:
        reg.register(bad)
    assert "already" in str(exc.value).lower()


@pytest.mark.parametrize("boundary", ["bogus", "", 123, None, object()])
def test_registerのboundary値域外はValueError(boundary: Any) -> None:
    """`Boundary` メンバでも 4 値の文字列でもない `boundary` は `ValueError`。"""
    reg = GuardrailRegistry()
    bad = GuardrailSpec(name="g", boundary=boundary, guardrail=_entity("g", Boundary.INPUT))
    with pytest.raises(ValueError):
        reg.register(bad)


def test_registerのguardrailがNoneならValueError() -> None:
    """`guardrail=None` は `ValueError`（実体のない宣言を登録しない）。"""
    reg = GuardrailRegistry()
    with pytest.raises(ValueError):
        reg.register(GuardrailSpec(name="g", boundary=Boundary.INPUT, guardrail=None))


@pytest.mark.parametrize("entity", [object(), _DuckGuardrail()])
def test_registerは上流4型でない実体を拒否する(entity: Any) -> None:
    """上流 4 型のインスタンスでない実体（duck-typed を含む）は `ValueError`。"""
    reg = GuardrailRegistry()
    with pytest.raises(ValueError):
        reg.register(GuardrailSpec(name="g", boundary=Boundary.INPUT, guardrail=entity))


def test_registerは可視名が登録キーと不一致な実体を拒否する() -> None:
    """実体の可視名が登録キーと違う宣言は `ValueError`（silent no-op の排除）。"""
    reg = GuardrailRegistry()
    bad = GuardrailSpec(
        name="declared", boundary=Boundary.INPUT, guardrail=_entity("actual", Boundary.INPUT)
    )
    with pytest.raises(ValueError):
        reg.register(bad)


def test_registerは宣言境界と実体境界の不一致を拒否する() -> None:
    """宣言 `boundary` と実体から判定した境界が食い違う宣言は `ValueError`。"""
    reg = GuardrailRegistry()
    bad = GuardrailSpec(name="g", boundary=Boundary.INPUT, guardrail=_entity("g", Boundary.OUTPUT))
    with pytest.raises(ValueError):
        reg.register(bad)


@pytest.mark.parametrize(
    ("boundary", "expected_type"),
    [
        (Boundary.INPUT, InputGuardrail),
        (Boundary.OUTPUT, OutputGuardrail),
        (Boundary.TOOL_INPUT, ToolInputGuardrail),
        (Boundary.TOOL_OUTPUT, ToolOutputGuardrail),
    ],
)
def test_registerは上流4型すべてを受理する(boundary: Boundary, expected_type: type) -> None:
    """宣言境界と実体が一致する限り上流 4 型すべてを登録できる（逃げ道の受理型）。"""
    reg = GuardrailRegistry()
    reg.register(_spec("g", boundary))
    assert isinstance(reg.get("g"), expected_type)
    assert reg.metadata("g").boundary is boundary


@pytest.mark.parametrize(
    ("boundary", "expected"),
    [
        ("input", Boundary.INPUT),
        ("output", Boundary.OUTPUT),
        ("tool_input", Boundary.TOOL_INPUT),
        ("tool_output", Boundary.TOOL_OUTPUT),
    ],
)
def test_registerのboundary文字列はBoundaryメンバへ正規化される(
    boundary: str, expected: Boundary
) -> None:
    """値域内の文字列で宣言しても登録でき、`metadata()` は `Boundary` メンバを返す。"""
    reg = GuardrailRegistry()
    reg.register(_spec("g", boundary))
    assert reg.metadata("g").boundary is expected


def test_registerの戻り値はmetadataと同一インスタンス() -> None:
    """`register()` は登録された `GuardrailSpec` を返し `metadata()` と `is` 一致する。"""
    reg = GuardrailRegistry()
    returned = reg.register(_spec("g", Boundary.OUTPUT))
    assert returned is reg.metadata("g")


@pytest.mark.parametrize(
    "bad",
    [
        GuardrailSpec(name="", boundary=Boundary.INPUT, guardrail=None),
        GuardrailSpec(name="g", boundary="bogus", guardrail=None),
        GuardrailSpec(name="g", boundary=Boundary.INPUT, guardrail=None, severity="high"),
        GuardrailSpec(name="g", boundary=Boundary.INPUT, guardrail=None),
        GuardrailSpec(name="g", boundary=Boundary.INPUT, guardrail=object()),
    ],
)
def test_register失敗時はnamesが変化しない(bad: GuardrailSpec) -> None:
    """検証のどの段で落ちても登録簿の状態は変化しない（部分適用なし）。"""
    reg = GuardrailRegistry()
    reg.register(_spec("existing", Boundary.INPUT))
    before = reg.names()
    with pytest.raises(ValueError):
        reg.register(bad)
    assert reg.names() == before


# ======================================================================
# E. 照会 6（get / names / metadata / boundary_of / specs）
# ======================================================================


def test_getは登録した実体を返し未登録はKeyError() -> None:
    """`get()` は登録実体と `is` 一致し、未登録名は名前を含む `KeyError`。"""
    reg = GuardrailRegistry()
    spec = reg.register(_spec("g", Boundary.INPUT))
    assert reg.get("g") is spec.guardrail
    with pytest.raises(KeyError) as exc:
        reg.get("missing")
    assert "missing" in str(exc.value)


def test_namesは昇順で境界フィルタは列挙型と文字列で一致する() -> None:
    """`names()` は昇順で、`boundary` フィルタは `Boundary` メンバと文字列で同じ結果。"""
    reg = GuardrailRegistry()
    reg.register(_spec("b_in", Boundary.INPUT))
    reg.register(_spec("a_in", Boundary.INPUT))
    reg.register(_spec("c_out", Boundary.OUTPUT))
    assert reg.names() == ["a_in", "b_in", "c_out"]
    assert reg.names(boundary=Boundary.INPUT) == ["a_in", "b_in"]
    assert reg.names(boundary="input") == ["a_in", "b_in"]


def test_登録が空ならnamesは空リストを返す() -> None:
    """登録が 1 件もなければ `names()` は空リスト（None ではない）。"""
    assert GuardrailRegistry().names() == []


@pytest.mark.parametrize("boundary", ["bogus", 123, object()])
def test_namesの境界値域外はValueError(boundary: Any) -> None:
    """`names(boundary=...)` の値域外は空リストを返さず `ValueError`（無言の空振り排除）。"""
    reg = GuardrailRegistry()
    reg.register(_spec("g", Boundary.INPUT))
    with pytest.raises(ValueError):
        reg.names(boundary=boundary)


def test_metadataは宣言を返し未登録はKeyError() -> None:
    """`metadata()` は `GuardrailSpec` を返し、未登録名は名前を含む `KeyError`。"""
    reg = GuardrailRegistry()
    reg.register(_spec("g", Boundary.INPUT))
    assert isinstance(reg.metadata("g"), GuardrailSpec)
    with pytest.raises(KeyError) as exc:
        reg.metadata("missing")
    assert "missing" in str(exc.value)


def test_boundary_ofは文字列として比較でき未登録はKeyError() -> None:
    """`boundary_of()` は str として比較可能な値を返し、未登録名は `KeyError`。"""
    reg = GuardrailRegistry()
    reg.register(_spec("g", Boundary.OUTPUT))
    assert reg.boundary_of("g") == "output"
    with pytest.raises(KeyError) as exc:
        reg.boundary_of("missing")
    assert "missing" in str(exc.value)


def test_specsは名前昇順で要素はmetadataと同一インスタンス() -> None:
    """`specs()` は名前昇順で、各要素は `metadata()` と `is` 一致する。"""
    reg = GuardrailRegistry()
    reg.register(_spec("b", Boundary.INPUT))
    reg.register(_spec("a", Boundary.OUTPUT))
    specs = reg.specs()
    assert [spec.name for spec in specs] == ["a", "b"]
    assert specs[0] is reg.metadata("a")
    assert specs[1] is reg.metadata("b")


def test_specsの戻り値は毎回新規リストで内部状態に波及しない() -> None:
    """返された list を破壊しても次回の `specs()` に波及しない（内部 list を露出しない）。

    非波及だけでなく**毎回新しい list であること**も pin する（呼び出しごとに同一 list を
    clear して詰め直す実装は非波及を満たしてしまい、内部 list の露出を検知できない）。
    """
    reg = GuardrailRegistry()
    reg.register(_spec("a", Boundary.INPUT))
    reg.register(_spec("b", Boundary.INPUT))
    first = reg.specs()
    first.clear()
    assert [spec.name for spec in reg.specs()] == ["a", "b"]
    assert reg.specs() is not reg.specs()


def test_登録が空ならspecsは空リストを返す() -> None:
    """登録が 1 件もなければ `specs()` は空リスト。"""
    assert GuardrailRegistry().specs() == []


def test_specsは境界でフィルタでき値域外はValueError() -> None:
    """`specs(boundary=...)` は境界で絞り込み、値域外は `ValueError`。"""
    reg = GuardrailRegistry()
    reg.register(_spec("a_in", Boundary.INPUT))
    reg.register(_spec("b_out", Boundary.OUTPUT))
    assert [spec.name for spec in reg.specs(boundary=Boundary.OUTPUT)] == ["b_out"]
    assert [spec.name for spec in reg.specs(boundary="output")] == ["b_out"]
    with pytest.raises(ValueError):
        reg.specs(boundary="bogus")


def test_specsのmin_severityは閾値以上のみ返しseverity未設定を除外する() -> None:
    """`min_severity` は閾値以上のみ返し、`severity is None` の登録は例外にせず除外する。"""
    reg = GuardrailRegistry()
    reg.register(_spec("a_low", Boundary.INPUT, severity=Severity.LOW))
    reg.register(_spec("b_high", Boundary.INPUT, severity=Severity.HIGH))
    reg.register(_spec("c_critical", Boundary.INPUT, severity=Severity.CRITICAL))
    reg.register(_spec("d_none", Boundary.INPUT))
    assert [spec.name for spec in reg.specs(min_severity=Severity.HIGH)] == [
        "b_high",
        "c_critical",
    ]


@pytest.mark.parametrize("min_severity", ["high", 3, object()])
def test_specsのmin_severity値域外はValueError(min_severity: Any) -> None:
    """`min_severity` が `Severity` メンバでなければ `ValueError`（素の int も拒否）。"""
    reg = GuardrailRegistry()
    reg.register(_spec("g", Boundary.INPUT, severity=Severity.HIGH))
    with pytest.raises(ValueError):
        reg.specs(min_severity=min_severity)


def test_specsのboundaryとmin_severityはAND条件() -> None:
    """`boundary` と `min_severity` を同時指定すると両条件を満たす宣言のみ返る。"""
    reg = GuardrailRegistry()
    reg.register(_spec("a_in_high", Boundary.INPUT, severity=Severity.HIGH))
    reg.register(_spec("b_in_low", Boundary.INPUT, severity=Severity.LOW))
    reg.register(_spec("c_out_high", Boundary.OUTPUT, severity=Severity.HIGH))
    specs = reg.specs(boundary=Boundary.INPUT, min_severity=Severity.HIGH)
    assert [spec.name for spec in specs] == ["a_in_high"]


# ======================================================================
# F. run_config_kwargs()
# ======================================================================


def test_run_config_kwargsは2キーを返しRunConfigを構築できる() -> None:
    """戻り値は境界別 2 キーのみで、そのまま `RunConfig(**kwargs)` に渡せる。"""
    reg = GuardrailRegistry()
    reg.register(_spec("a_in", Boundary.INPUT))
    reg.register(_spec("b_out", Boundary.OUTPUT))
    kwargs = reg.run_config_kwargs()
    assert set(kwargs) == {"input_guardrails", "output_guardrails"}
    config = RunConfig(**kwargs)
    assert config.input_guardrails == [reg.get("a_in")]
    assert config.output_guardrails == [reg.get("b_out")]


def test_run_config_kwargsは境界別に振り分け渡した宣言順を保つ() -> None:
    """明示した名前は境界別に振り分けられ、各境界内では渡した順序が保たれる。"""
    reg = GuardrailRegistry()
    for name in ("i1", "i2", "i3"):
        reg.register(_spec(name, Boundary.INPUT))
    for name in ("o1", "o2"):
        reg.register(_spec(name, Boundary.OUTPUT))
    kwargs = reg.run_config_kwargs(["i3", "o2", "i1", "o1", "i2"])
    assert kwargs["input_guardrails"] == [reg.get("i3"), reg.get("i1"), reg.get("i2")]
    assert kwargs["output_guardrails"] == [reg.get("o2"), reg.get("o1")]


def test_run_config_kwargsの引数なしは登録全件を名前昇順で返す() -> None:
    """`names=None` は登録全件を対象とし、各境界内は名前昇順になる。"""
    reg = GuardrailRegistry()
    for name in ("i_b", "i_a"):
        reg.register(_spec(name, Boundary.INPUT))
    for name in ("o_b", "o_a"):
        reg.register(_spec(name, Boundary.OUTPUT))
    kwargs = reg.run_config_kwargs()
    assert kwargs["input_guardrails"] == [reg.get("i_a"), reg.get("i_b")]
    assert kwargs["output_guardrails"] == [reg.get("o_a"), reg.get("o_b")]


def test_run_config_kwargsは重複名を排除しない() -> None:
    """同じ名前を 2 回渡すと同じ実体が 2 回入る（重複の是非は利用者判断）。"""
    reg = GuardrailRegistry()
    reg.register(_spec("g", Boundary.INPUT))
    kwargs = reg.run_config_kwargs(["g", "g"])
    assert kwargs["input_guardrails"] == [reg.get("g"), reg.get("g")]


def test_run_config_kwargsは空列でもキーを欠落させない() -> None:
    """空列を渡しても 2 キーは存在し、値は空リストになる（キー欠落で SDK 既定に落ちない）。"""
    reg = GuardrailRegistry()
    reg.register(_spec("g", Boundary.INPUT))
    kwargs = reg.run_config_kwargs([])
    assert kwargs == {"input_guardrails": [], "output_guardrails": []}


def test_run_config_kwargsの戻り値は毎回新規で要素はgetと同一() -> None:
    """返された list を破壊しても次回に波及せず、要素は `get()` と `is` 一致する。

    非波及だけでなく**dict と内側 list が毎回新規であること**も pin する（同一オブジェクトを
    clear して詰め直す実装は非波及を満たしてしまう）。
    """
    reg = GuardrailRegistry()
    reg.register(_spec("g", Boundary.INPUT))
    first = reg.run_config_kwargs()
    first["input_guardrails"].clear()
    second = reg.run_config_kwargs()
    assert second["input_guardrails"] == [reg.get("g")]
    assert second["input_guardrails"][0] is reg.get("g")
    assert reg.run_config_kwargs() is not reg.run_config_kwargs()
    assert (
        reg.run_config_kwargs()["input_guardrails"]
        is not reg.run_config_kwargs()["input_guardrails"]
    )


def test_run_config_kwargsの検証順は要素の型が名前解決より先() -> None:
    """非 str 要素と未登録名が混在したら型検査由来の `ValueError` になる。"""
    reg = GuardrailRegistry()
    with pytest.raises(ValueError):
        reg.run_config_kwargs(["missing", 123])


def test_run_config_kwargsの検証順は名前解決が境界検査より先() -> None:
    """未登録名とツール境界名が混在したら名前解決由来の `KeyError` になる。"""
    reg = GuardrailRegistry()
    reg.register(_spec("t_out", Boundary.TOOL_OUTPUT))
    with pytest.raises(KeyError):
        reg.run_config_kwargs(["missing", "t_out"])


def test_run_config_kwargsはツール境界名の明示をValueErrorで拒否する() -> None:
    """ツール境界の登録名を明示で渡すと境界値を含む `ValueError`（無言で落とさない）。"""
    reg = GuardrailRegistry()
    reg.register(_spec("t_out", Boundary.TOOL_OUTPUT))
    with pytest.raises(ValueError) as exc:
        reg.run_config_kwargs(["t_out"])
    assert "tool_output" in str(exc.value)


def test_run_config_kwargsは引数なしでもツール境界の登録を静かに除外しない() -> None:
    """`names=None` でも登録にツール境界が含まれれば `ValueError`（暗黙の除外をしない）。"""
    reg = GuardrailRegistry()
    reg.register(_spec("a_in", Boundary.INPUT))
    reg.register(_spec("t_in", Boundary.TOOL_INPUT))
    with pytest.raises(ValueError):
        reg.run_config_kwargs()


# ----------------------------------------------------------------------
# レビュー指摘の反映（堅牢化・fail-closed）
# ----------------------------------------------------------------------


def test_run_config_kwargsはbare_strを拒否する() -> None:
    """`names` に素の `str` を渡すと `ValueError`（1 文字ずつ分解された fail-open を閉じる）。

    `str` は `Sequence[str]` を満たすため型注釈では防げない。空文字を渡すと guardrail 0 件の
    `RunConfig` が静かに生成されるため、検証順 3 段の前段で弾く。
    """
    reg = GuardrailRegistry()
    reg.predicate_guardrail(_never, on="input", name="ib")
    for names in ("", "ib"):
        with pytest.raises(ValueError, match="bare str"):
            reg.run_config_kwargs(names)
    assert reg.run_config_kwargs(["ib"])["input_guardrails"] == [reg.get("ib")]


def test_boundary_ofの戻り値注釈はBoundaryである() -> None:
    """`boundary_of` の戻り値注釈が実返却（`Boundary` メンバ）と一致する。

    注釈が `str` だと、利用者が `f"{reg.boundary_of(n)}"` と素直に書いたときログ / 表示へ
    `Boundary.OUTPUT` が出る落とし穴が型から見えない（`Boundary` は `str` 部分型なので
    既存の文字列比較契約は不変）。
    """
    hints = inspect.get_annotations(GuardrailRegistry.boundary_of)
    assert hints["return"] == "Boundary"


def test_例外文言は登録名をreprで埋め込む() -> None:
    """重複名・未登録名の文言が `{name!r}` を使う（生の制御文字がログへ流れない）。

    登録名が設定ファイル / 管理 UI 由来の構成で改行入りの名前が渡ると、`str(exc)` をそのまま
    ログへ出すと偽の監査行を注入できる（CWE-117）。repr エスケープで閉じる。
    """
    reg = GuardrailRegistry()
    with pytest.raises(KeyError) as unknown:
        reg.get("a\nb: FAKE")
    assert "\\n" in str(unknown.value)

    reg.predicate_guardrail(_never, on="input", name="dup")
    with pytest.raises(ValueError) as dup:
        reg.predicate_guardrail(_never, on="input", name="dup")
    assert "'dup'" in str(dup.value)


def test_registerは可視名取得の失敗をValueErrorへ包む() -> None:
    """可視名を取得できない上流型の実体を `register()` が `ValueError` にする（P1）。

    上流 4 型は `name=None` を許し、`get_name()` は `guardrail_function.__name__` へフォール
    バックする。`functools.partial` や `__call__` オブジェクトを guardrail 関数にすると
    `AttributeError` になるため、包まないと「登録時の検証は必ず `ValueError`」という契約が崩れ、
    利用者は宣言不備を一様に処理できない。文言には登録キーを含める。
    """
    entity = InputGuardrail(guardrail_function=functools.partial(_never))
    spec = GuardrailSpec(name="partial_gr", boundary=Boundary.INPUT, guardrail=entity)
    reg = GuardrailRegistry()
    with pytest.raises(ValueError, match="partial_gr"):
        reg.register(spec)
    assert reg.names() == []
