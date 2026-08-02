"""L1: 宣言的ガードレール登録の plain 型層（`types`）の純検証（agents / SDK 非依存）。

`Boundary`（4 境界の値域・`str` 互換・文字列 coerce・agent 境界 2 / ツール境界 2 の分割）、
`Severity`（比較演算子による全順序・1 始まりの値域・全メンバが真値）、`GuardrailSpec`
（与えたフィールドの保持・`labels` / `severity` の既定・`labels` 既定のインスタンス非共有・
値を解釈せず保持に徹すること・値域外を受けても構築が成功すること）を検証する。あわせて
`types` モジュール単体が `agents` をロードしないこと（NFR-1）を固定する。SDK を一切 import しない。
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from oai_agentspec.runtime.guardrails.types import Boundary, GuardrailSpec, Severity

pytestmark = pytest.mark.unit


_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
_TYPES_PATH = _SRC_DIR / "oai_agentspec" / "runtime" / "guardrails" / "types.py"


# ----------------------------------------------------------------------
# Boundary
# ----------------------------------------------------------------------


def test_boundary_のメンバ集合は4境界に固定される() -> None:
    """値域は input / output / tool_input / tool_output の 4 種のみ。

    集合の `==` で pin する（過大 = 境界追加と過小 = 境界切り詰めの両方向を同時に検知する
    ため。`<=` や個別メンバの存在確認へ弱めると切り詰め変異を見逃す）。
    """
    assert {b.value for b in Boundary} == {"input", "output", "tool_input", "tool_output"}
    assert len(Boundary) == 4


def test_boundary_は文字列と相互運用できる() -> None:
    """`str` 併用 Enum として素の文字列と等価比較・isinstance・dict キー互換が成立する。"""
    assert Boundary.INPUT == "input"
    assert Boundary.OUTPUT == "output"
    assert Boundary.TOOL_INPUT == "tool_input"
    assert Boundary.TOOL_OUTPUT == "tool_output"
    assert isinstance(Boundary.INPUT, str)
    # ハッシュも str と一致するため、素の文字列キーの dict をそのまま引ける。
    assert {"input": "hit"}[Boundary.INPUT] == "hit"


def test_boundary_は値域内の文字列からcoerceできる() -> None:
    """`Boundary("tool_output")` のように文字列から対応メンバへ復元できる。"""
    assert Boundary("input") is Boundary.INPUT
    assert Boundary("output") is Boundary.OUTPUT
    assert Boundary("tool_input") is Boundary.TOOL_INPUT
    assert Boundary("tool_output") is Boundary.TOOL_OUTPUT


def test_boundary_は値域外の文字列でValueErrorになる() -> None:
    """値域外の文字列 coerce は Enum 標準どおり `ValueError`（黙って通さない）。"""
    with pytest.raises(ValueError):
        Boundary("tool")
    with pytest.raises(ValueError):
        Boundary("INPUT")
    with pytest.raises(ValueError):
        Boundary("")


def test_boundary_はagent境界2とツール境界2に分割される() -> None:
    """4 境界は agent 境界 2 / ツール境界 2 に排他かつ網羅的に分かれる。

    メンバ集合の一致で pin する（値の接頭辞など表現に依存した判別は書かない。値表記を
    変えても壊れず、境界の増減だけを検知したいため）。
    """
    agent_boundaries = {Boundary.INPUT, Boundary.OUTPUT}
    tool_boundaries = {Boundary.TOOL_INPUT, Boundary.TOOL_OUTPUT}
    assert len(agent_boundaries) == 2
    assert len(tool_boundaries) == 2
    # 排他（重複なし）かつ網羅（この 2 群で全メンバを覆う）を両方向で固定する。
    assert agent_boundaries & tool_boundaries == set()
    assert agent_boundaries | tool_boundaries == set(Boundary)


# ----------------------------------------------------------------------
# Severity
# ----------------------------------------------------------------------


def test_severity_は比較演算子で昇順の全順序を持つ() -> None:
    """LOW < MEDIUM < HIGH < CRITICAL が比較演算子で成立する（観測手段は比較演算子に固定）。"""
    assert Severity.LOW < Severity.MEDIUM < Severity.HIGH < Severity.CRITICAL
    assert Severity.CRITICAL > Severity.HIGH > Severity.MEDIUM > Severity.LOW


def test_severity_は逆向きの比較が偽になる() -> None:
    """逆順の比較は偽（順序の向きを pin し、比較の反転変異を検知する）。"""
    assert not Severity.MEDIUM < Severity.LOW
    assert not Severity.HIGH < Severity.MEDIUM
    assert not Severity.CRITICAL < Severity.HIGH
    assert not Severity.LOW > Severity.CRITICAL


def test_severity_のメンバ集合は4段階に固定される() -> None:
    """値域は 1 / 2 / 3 / 4 の 4 段階のみ。

    集合の `==` で pin する（段階追加 = 過大と段階削除・値ずらし = 過小の両方向を同時に
    検知するため。片方向の包含だけでは切り詰め変異を見逃す）。
    """
    assert {s.value for s in Severity} == {1, 2, 3, 4}
    assert len(Severity) == 4
    assert [s.name for s in Severity] == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def test_severity_は1始まりで全メンバが真値() -> None:
    """LOW が 1 始まりで、どのメンバも `bool()` が真（0 始まりの真偽値の落とし穴がない）。"""
    assert Severity.LOW == 1
    assert min(Severity) == Severity.LOW
    assert all(bool(s) for s in Severity)


# ----------------------------------------------------------------------
# GuardrailSpec
# ----------------------------------------------------------------------


def test_guardrailspec_のフィールド集合は5件に固定される() -> None:
    """宣言フィールドは name / boundary / guardrail / labels / severity の 5 件のみ。

    集合の `==` で pin する（フィールド追加と削除の両方向を同時に検知するため）。
    """
    assert {f.name for f in fields(GuardrailSpec)} == {
        "name",
        "boundary",
        "guardrail",
        "labels",
        "severity",
    }


def test_guardrailspec_は与えたフィールドをそのまま保持する() -> None:
    """5 フィールドを明示指定すると、値（guardrail は同一オブジェクト）をそのまま保持する。"""
    sentinel = object()
    spec = GuardrailSpec(
        name="pii-input",
        boundary=Boundary.INPUT,
        guardrail=sentinel,
        labels={"stage": "pre", "owner": "safety"},
        severity=Severity.HIGH,
    )
    assert spec.name == "pii-input"
    assert spec.boundary is Boundary.INPUT
    assert spec.guardrail is sentinel
    assert spec.labels == {"stage": "pre", "owner": "safety"}
    assert spec.severity is Severity.HIGH


def test_guardrailspec_の位置引数は宣言順で受け取れる() -> None:
    """位置引数の順序は name / boundary / guardrail / labels / severity。"""
    sentinel = object()
    spec = GuardrailSpec("n", Boundary.OUTPUT, sentinel, {"k": "v"}, Severity.LOW)
    assert spec.name == "n"
    assert spec.boundary is Boundary.OUTPUT
    assert spec.guardrail is sentinel
    assert spec.labels == {"k": "v"}
    assert spec.severity is Severity.LOW


def test_guardrailspec_の省略時既定は空dictとNone() -> None:
    """`labels` 未指定は空 dict、`severity` 未指定は None（既定で severity を推定しない）。"""
    spec = GuardrailSpec(name="n", boundary=Boundary.INPUT, guardrail=object())
    assert spec.labels == {}
    assert spec.severity is None


def test_guardrailspec_のlabels既定はインスタンス間で共有されない() -> None:
    """`labels` の既定はインスタンスごとに独立（`field(default_factory=dict)` の pin）。"""
    first = GuardrailSpec(name="a", boundary=Boundary.INPUT, guardrail=object())
    second = GuardrailSpec(name="b", boundary=Boundary.INPUT, guardrail=object())
    first.labels["added"] = "yes"
    assert second.labels == {}
    assert first.labels is not second.labels


def test_guardrailspec_はフィールドの再代入でFrozenInstanceErrorになる() -> None:
    """`frozen=True` の pin（登録時検証を通った宣言が後から書き換わる経路を閉じる）。

    境界を書き換えられると、出力境界の宣言が入力側へ結線されて対象を一度も検査しないまま
    一覧上は「登録済み」に見える。5 フィールドすべてを回して抜けを作らない。
    """
    spec = GuardrailSpec(name="a", boundary=Boundary.OUTPUT, guardrail=object())
    for attr, value in (
        ("name", "b"),
        ("boundary", Boundary.INPUT),
        ("guardrail", object()),
        ("labels", {"k": "v"}),
        ("severity", Severity.LOW),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(spec, attr, value)
    assert spec.boundary is Boundary.OUTPUT


def test_guardrailspec_のlabelsはfrozenでもキー単位で更新できる() -> None:
    """`frozen=True` が禁止するのは属性の再代入のみで、`labels` のキー更新は通る（意図的）。

    宣言後のラベル追記を許す設計なので、`labels` 自体を不変マッピングへ差し替えると壊れる。
    """
    spec = GuardrailSpec(name="a", boundary=Boundary.INPUT, guardrail=object())
    spec.labels["team"] = "sec"
    assert spec.labels == {"team": "sec"}


def test_guardrailspec_はlabelsの値を正規化しない() -> None:
    """`labels` の値は単一 str / シーケンスのいずれも畳み込み・展開せずそのまま保持する。"""
    tags = ["pii", "prod"]
    spec = GuardrailSpec(
        name="n",
        boundary=Boundary.INPUT,
        guardrail=object(),
        labels={"tag": "pii", "tags": tags, "tuple": ("a", "b")},
    )
    assert spec.labels["tag"] == "pii"
    assert spec.labels["tags"] is tags
    assert spec.labels["tuple"] == ("a", "b")


def test_guardrailspec_は値域外のboundaryとseverityでも構築できる() -> None:
    """値域外の boundary / severity でも構築は成功する（検証は登録簿の責務・保持に徹する）。"""
    spec = GuardrailSpec(
        name="n",
        boundary="not-a-boundary",
        guardrail=object(),
        severity=99,  # type: ignore[arg-type]
    )
    assert spec.boundary == "not-a-boundary"
    assert spec.severity == 99


def test_guardrailspec_はboundaryを文字列のまま保持する() -> None:
    """値域内の文字列を渡しても Enum へ coerce せずそのまま保持する（変換は登録簿の責務）。"""
    spec = GuardrailSpec(name="n", boundary="tool_output", guardrail=object())
    assert spec.boundary == "tool_output"
    assert type(spec.boundary) is str


# ----------------------------------------------------------------------
# SDK 隔離（NFR-1）
# ----------------------------------------------------------------------


def _run_in_clean_subprocess(probe: str) -> str:
    """`src` を path に通したクリーンな子プロセスで probe スクリプトを実行し標準出力を返す。"""
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
    return result.stdout.strip()


def test_types_モジュール単体のimportでagentsがロードされない() -> None:
    """`types.py` 単体を実行しても `sys.modules` に `agents` が現れない（NFR-1）。

    パッケージ窓口（`runtime.guardrails.__init__`）は `factories` 経由で SDK を載せるため、
    ファイルパス指定で当該モジュールのみをクリーンな子プロセスで実行して切り分ける。
    """
    probe = (
        "import importlib.util\n"
        "import sys\n"
        f"spec = importlib.util.spec_from_file_location('_gr_types_probe', r'{_TYPES_PATH}')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['_gr_types_probe'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "assert mod.Boundary is not None\n"
        "loaded = sorted(\n"
        "    m for m in sys.modules if m == 'agents' or m.startswith('agents.')\n"
        ")\n"
        "print(','.join(loaded))\n"
    )
    out = _run_in_clean_subprocess(probe)
    loaded = [m for m in out.split(",") if m]
    assert loaded == [], f"types 単体 import で agents がロードされました: {loaded}"


def test_types_モジュールはagentsへの参照を含まない() -> None:
    """`types.py` のソースに `agents` / `_adapters` への import 文が現れない（NFR-1）。"""
    source = _TYPES_PATH.read_text(encoding="utf-8")
    assert "from agents" not in source
    assert "import agents" not in source
    assert "_adapters" not in source
