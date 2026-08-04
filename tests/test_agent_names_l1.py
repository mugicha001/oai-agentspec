"""L1: エージェント名定数簿（`AgentNames` / `validate_agent_names`）の検証（agents 非依存）。

タスク A1 の RED テスト。`src/oai_agentspec/agent_names.py` は未実装のため、本ファイルは
import 時点で失敗する（実装後に緑へ変わる）。

検証範囲は設計方針 §3-1（`docs/architecture.md` の「エージェント名定数簿」節 /
`docs/adr/0018-declarative-agent-name-catalog.md`）の 3 点:

1. 宣言済み名がクラス属性として静的に解決でき、`dir()` に載り、`names()` が宣言値の昇順を返す
2. 未宣言名アクセスが宣言済み**属性名**の一覧つき `AttributeError` になる（`_` 始まりは素通し）
3. 到達不能な宣言・不正な値・注釈のみ宣言・値の重複がクラス定義時に `ValueError` になる

加えて整合検査（`validate_agent_names`）が両方向の差分を単一 `KeyError` で全件報告し、
`register_factory` 登録名を未登録と誤検知しないことを pin する。

例外メッセージの言語は「踏襲元の同一トピックに揃える」方針のため、`ValueError` /
`AttributeError` は英語（`ToolRegistry` 踏襲）、集約報告の `KeyError` は日本語
（`AgentRegistry.validate()` 踏襲）で照合する。
"""

from __future__ import annotations

from typing import Any

import pytest

import oai_agentspec
from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.agent_names import AgentNames, validate_agent_names

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# コア公開窓口（`oai_agentspec` トップレベル）経由の解決
# ---------------------------------------------------------------------------
def test_コア公開窓口から_AgentNames_と_validate_agent_names_を_import_できる() -> None:
    """docs が案内する `from oai_agentspec import AgentNames, validate_agent_names` を pin する。

    `src/oai_agentspec/__init__.py` の再エクスポート行が消えると `ImportError` になる
    （内部モジュール直接 import のテストだけでは検知できない）。
    """
    from oai_agentspec import AgentNames as PublicAgentNames
    from oai_agentspec import validate_agent_names as public_validate_agent_names

    assert PublicAgentNames is AgentNames
    assert public_validate_agent_names is validate_agent_names


def test_AgentNames_と_validate_agent_names_は_all_に載り_getattr_で解決できる() -> None:
    """`__all__` への掲載と `getattr` 解決可能性の両方を positive に確認する。"""
    assert "AgentNames" in oai_agentspec.__all__
    assert "validate_agent_names" in oai_agentspec.__all__
    assert oai_agentspec.AgentNames is AgentNames
    assert oai_agentspec.validate_agent_names is validate_agent_names


class _Names(AgentNames):
    """テスト共通の定数簿（宣言はこの 1 箇所）。"""

    PLANNER = "planner"
    WRITER = "writer"


class _Empty(AgentNames):
    """宣言を 1 件も持たない定数簿（空一覧の文言を確認するため）。"""


def _make_names_class(namespace: dict[str, Any], *, name: str = "_Dynamic") -> type:
    """`type()` 経由で `AgentNames` のサブクラスを生成する。

    class 文では書けない属性名（非識別子 / Python 予約語）も検査対象に含めるため、
    namespace を直接渡す形で生成する。メタクラスは基底から継承されるため、
    class 文と同じ定義時検査が走る。

    Args:
        namespace: クラス body 相当の名前空間。
        name: 生成するクラス名。

    Returns:
        生成した `AgentNames` サブクラス。
    """
    return type(name, (AgentNames,), namespace)


def _plain_function() -> str:
    """宣言値として拒否されるべき callable（テスト用）。"""
    return "planner"


# ---------------------------------------------------------------------------
# 定義できること（dunder 除外 / 基底クラス自身の skip）
# ---------------------------------------------------------------------------
def test_docstring付きのサブクラスを定義できる() -> None:
    """`__module__` / `__qualname__` / `__doc__` は検査対象・宣言集合の双方から除外される。

    除外しないと docstring 付きクラスが「値が非 str」で定義時に落ちる（NG-3）。
    """

    class _WithDoc(AgentNames):
        """docstring を持つ定数簿。"""

        PLANNER = "planner"

    assert _WithDoc.__doc__ is not None
    assert _WithDoc.__qualname__.endswith("_WithDoc")
    # 宣言集合に dunder が混ざっていないこと（値は宣言した 1 件のみ）。
    assert _WithDoc.names() == ["planner"]


def test_基底クラス自身の生成と空のサブクラス定義が通る() -> None:
    """基底 `AgentNames` の生成では検査を skip し、宣言 0 件のサブクラスも定義できる。"""
    assert isinstance(AgentNames, type)
    assert AgentNames.names() == []
    assert _Empty.names() == []


# ---------------------------------------------------------------------------
# 参照（静的属性解決 / dir() 掲載 / names() の昇順）
# ---------------------------------------------------------------------------
def test_宣言済み名は属性解決でき_dir_と_names_に載る() -> None:
    """宣言済み名は str 値として解決され、`dir()` に全宣言名が含まれる。"""
    assert _Names.PLANNER == "planner"
    assert _Names.WRITER == "writer"
    assert isinstance(_Names.PLANNER, str)
    assert isinstance(_Names.WRITER, str)
    assert {"PLANNER", "WRITER"} <= set(dir(_Names))
    assert _Names.names() == ["planner", "writer"]


def test_names_は宣言値の昇順リストを返す() -> None:
    """宣言順ではなく**値の昇順**で返す（属性名の順序に依存しない）。"""

    class _Unsorted(AgentNames):
        """宣言順と昇順が一致しない定数簿。"""

        ZETA = "zeta"
        ALPHA = "alpha"
        MIDDLE = "middle"

    assert _Unsorted.names() == ["alpha", "middle", "zeta"]


# ---------------------------------------------------------------------------
# 未宣言名アクセス（一覧つき AttributeError / `_` 始まりの素通し）
# ---------------------------------------------------------------------------
def test_未宣言名アクセスは宣言名一覧つき_AttributeError() -> None:
    """`ToolRegistry._unknown_tool_message` と同一体裁の文言を pin する。

    列挙するのは**属性名**（利用者が書いた識別子で照合できる形が有用なため）。
    宣言が 0 件のときは一覧を `(none)` と表示する（先例と同一）。
    """
    with pytest.raises(AttributeError) as exc:
        _ = _Names.PLANER
    assert str(exc.value) == "unknown agent name: PLANER. declared agent names: PLANNER, WRITER"

    with pytest.raises(AttributeError) as empty_exc:
        _ = _Empty.PLANNER
    assert str(empty_exc.value) == "unknown agent name: PLANNER. declared agent names: (none)"


def test_アンダースコア始まり属性のアクセスは素の_AttributeError() -> None:
    """`_` 始まり（dunder を含む）は一覧つき文言を出さない（内部プロトコル探索保護）。

    `inspect` / `copy` / `pickle` / pytest の introspection が誤誘導メッセージを受け取らない
    ことを担保する（`ToolRegistry.__getattr__` の `_` 始まり素通しと同型）。
    """
    # `object` が実体を持つ dunder（`__getstate__` 等）は `__getattr__` を通らないため、
    # 実体を持たない探索名だけを対象にする。
    for attr in ("_missing", "__deepcopy__", "__copy__", "__wrapped__"):
        with pytest.raises(AttributeError) as exc:
            getattr(_Names, attr)
        message = str(exc.value)
        assert "unknown agent name" not in message
        assert "declared agent names" not in message


# ---------------------------------------------------------------------------
# クラス定義時の ValueError（到達不能名 4 分岐 / 値 / 注釈のみ / 値の重複）
# ---------------------------------------------------------------------------
def test_到達不能な宣言はクラス定義時に_ValueError() -> None:
    """4 分岐（非識別子 / `_` 始まり / Python 予約語 / 予約属性名）を定義時に拒否する。

    規則の SoT は `ToolRegistry._validate_name`（`tool_registry.py:176-199`）。予約属性名は
    `{"names", "mro"}`（`mro` は `cls.mro()` による introspection を守るため）。
    """
    with pytest.raises(ValueError, match="not-an-identifier"):
        _make_names_class({"not-an-identifier": "planner"})
    with pytest.raises(ValueError, match="_PLANNER"):
        _make_names_class({"_PLANNER": "planner"})
    with pytest.raises(ValueError, match="class"):
        _make_names_class({"class": "planner"})
    with pytest.raises(ValueError, match="names"):
        _make_names_class({"names": "planner"})
    with pytest.raises(ValueError, match="mro"):
        _make_names_class({"mro": "planner"})


def test_アンダースコア始まりの宣言はクラス文でも定義時に_ValueError() -> None:
    """利用者が実際に書く class 文の形でも同じ規則が働く。"""
    with pytest.raises(ValueError, match="_PLANNER"):

        class _Bad(AgentNames):
            """`_` 始まりの宣言を持つ定数簿。"""

            _PLANNER = "planner"


def test_値が非_str_または空文字の宣言はクラス定義時に_ValueError() -> None:
    """値は非空の str のみを受け付ける。"""
    with pytest.raises(ValueError, match="PLANNER"):
        _make_names_class({"PLANNER": 1})
    with pytest.raises(ValueError, match="PLANNER"):
        _make_names_class({"PLANNER": None})
    with pytest.raises(ValueError, match="PLANNER"):
        _make_names_class({"PLANNER": b"planner"})
    with pytest.raises(ValueError, match="PLANNER"):
        _make_names_class({"PLANNER": ""})


def test_callable_や_descriptor_の宣言はクラス定義時に_ValueError() -> None:
    """定数簿の subclass body に振る舞いを持たせない（宣言専用の性質を守る）。"""
    with pytest.raises(ValueError, match="PLANNER"):
        _make_names_class({"PLANNER": _plain_function})
    with pytest.raises(ValueError, match="PLANNER"):
        _make_names_class({"PLANNER": classmethod(_plain_function)})
    with pytest.raises(ValueError, match="PLANNER"):
        _make_names_class({"PLANNER": staticmethod(_plain_function)})
    with pytest.raises(ValueError, match="PLANNER"):
        _make_names_class({"PLANNER": property(_plain_function)})
    with pytest.raises(ValueError, match="PLANNER"):
        _make_names_class({"PLANNER": str})


def test_メソッドを宣言したサブクラスはクラス定義時に_ValueError() -> None:
    """class 文でメソッドを持たせた場合も同じ規則で拒否する。"""
    with pytest.raises(ValueError, match="planner_name"):

        class _Bad(AgentNames):
            """メソッドを持つ定数簿。"""

            def planner_name(self) -> str:
                """振る舞いは宣言専用の性質に反する。"""
                return "planner"


def test_注釈のみの宣言はクラス定義時に_ValueError() -> None:
    """属性が生えず参照が `AttributeError` になる silent trap を定義時に拒否する。"""
    with pytest.raises(ValueError, match="PLANNER"):

        class _Bad(AgentNames):
            """注釈だけで値を代入していない定数簿。"""

            PLANNER: str


def test_注釈と値をともに書いた宣言は許容される() -> None:
    """`PLANNER: str = "planner"` は属性が生えるため通る（注釈のみとの差分を pin）。"""

    class _Annotated(AgentNames):
        """注釈つきで値も代入した定数簿。"""

        PLANNER: str = "planner"

    assert _Annotated.PLANNER == "planner"
    assert _Annotated.names() == ["planner"]


def test_同一値を異なる属性名へ宣言するとクラス定義時に_ValueError() -> None:
    """`PLANNER` / `PLANER` の併存はどの検出網にも掛からないため定義時に拒否する（決定 7）。"""
    with pytest.raises(ValueError) as exc:
        _make_names_class({"PLANNER": "planner", "PLANER": "planner"})
    message = str(exc.value)
    assert "planner" in message
    assert "PLANNER" in message
    assert "PLANER" in message


def test_属性名と値が異なる宣言は許容される() -> None:
    """拒否対象は「値の重複」だけで、属性名と値の不一致は通る。"""

    class _Versioned(AgentNames):
        """属性名と値が異なる定数簿。"""

        PLANNER = "planner"
        PLANNER_V2 = "planner-v2"

    assert _Versioned.names() == ["planner", "planner-v2"]


# ---------------------------------------------------------------------------
# 継承（MRO 集約）
# ---------------------------------------------------------------------------
def test_多段継承で_names_は_MRO_集約され_dir_と対応する() -> None:
    """`dir()` が親の属性を含むため、`names()` も MRO 集約しないと食い違う。"""

    class _Base(AgentNames):
        """親の定数簿。"""

        PLANNER = "planner"

    class _Sub(_Base):
        """子の定数簿（親の宣言を引き継ぐ）。"""

        WRITER = "writer"

    assert _Base.names() == ["planner"]
    assert _Sub.names() == ["planner", "writer"]
    assert _Sub.PLANNER == "planner"
    assert {"PLANNER", "WRITER"} <= set(dir(_Sub))
    # `dir()` に載る宣言名（公開の str 属性）と `names()` の内容が対応する。
    declared = {
        attr
        for attr in dir(_Sub)
        if not attr.startswith("_") and isinstance(getattr(_Sub, attr, None), str)
    }
    assert declared == {"PLANNER", "WRITER"}
    assert {getattr(_Sub, attr) for attr in declared} == set(_Sub.names())


def test_同一属性名の_override_は重複エラーにならず_1_名前として扱われる() -> None:
    """override は「異なる属性名への同一値の割り当て」ではないため通る。"""

    class _Base(AgentNames):
        """親の定数簿。"""

        PLANNER = "planner"

    class _Override(_Base):
        """親と同じ属性名を上書きする定数簿。"""

        PLANNER = "planner-v2"

    assert _Override.PLANNER == "planner-v2"
    assert _Override.names() == ["planner-v2"]


def test_継承をまたいだ値の重複もクラス定義時に_ValueError() -> None:
    """値の重複検査は MRO 集約後の `{属性名: 値}` に対して行う。"""

    class _Base(AgentNames):
        """親の定数簿。"""

        PLANNER = "planner"

    with pytest.raises(ValueError, match="planner"):

        class _Sub(_Base):
            """親と同じ値を別の属性名で宣言する定数簿。"""

            PLANER = "planner"


# ---------------------------------------------------------------------------
# 整合検査（validate_agent_names）
# ---------------------------------------------------------------------------
def test_整合検査は両方向の差分を単一_KeyError_で全件報告する() -> None:
    """「宣言済みだが未登録」「登録済みだが未宣言」を 2 群へまとめ単一 `KeyError` にする。

    体裁は `AgentRegistry.validate()` 踏襲（群ごとに接頭辞 + `"; "` 連結、群が 2 つ揃うときのみ
    `" | "` 連結）。片方向だけの報告は「1 箇所宣言」の前提を静かに崩すため、両方向が同時に
    出ることを 1 件ずつの厳密一致で pin する。
    """

    class _Catalog(AgentNames):
        """planner / writer を宣言した定数簿。"""

        PLANNER = "planner"
        WRITER = "writer"

    registry = AgentRegistry()
    registry.register(AgentSpec(name="planner", instructions="計画を立てる"))
    registry.register(AgentSpec(name="reviewer", instructions="レビューする"))

    with pytest.raises(KeyError) as exc:
        validate_agent_names(_Catalog, registry)
    assert exc.value.args[0] == (
        "定数簿に宣言済みだが registry へ未登録: 'writer'"
        " | registry へ登録済みだが定数簿に未宣言: 'reviewer'"
    )


def test_整合検査は各群の差分を全件列挙する() -> None:
    """1 群に複数の差分があっても最初の 1 件で打ち切らない（全件が 1 つの例外に載る）。"""

    class _Catalog(AgentNames):
        """3 件を宣言した定数簿。"""

        PLANNER = "planner"
        WRITER = "writer"
        ZEBRA = "zebra"

    registry = AgentRegistry()
    registry.register(AgentSpec(name="planner", instructions="計画を立てる"))
    registry.register(AgentSpec(name="alpha", instructions="alpha"))
    registry.register(AgentSpec(name="reviewer", instructions="レビューする"))

    with pytest.raises(KeyError) as exc:
        validate_agent_names(_Catalog, registry)
    message = exc.value.args[0]
    for name in ("writer", "zebra", "alpha", "reviewer"):
        assert repr(name) in message
    # 群内は "; "、群間は " | " で 1 回だけ連結する。
    assert "; " in message
    assert message.count(" | ") == 1


def test_整合検査は片方向のみの差分では群を連結しない() -> None:
    """群が 1 つのときは `" | "` を挟まない（`AgentRegistry.validate()` と同型）。"""

    class _Catalog(AgentNames):
        """planner / writer を宣言した定数簿。"""

        PLANNER = "planner"
        WRITER = "writer"

    registry = AgentRegistry()
    registry.register(AgentSpec(name="planner", instructions="計画を立てる"))

    with pytest.raises(KeyError) as exc:
        validate_agent_names(_Catalog, registry)
    assert exc.value.args[0] == "定数簿に宣言済みだが registry へ未登録: 'writer'"


def test_整合検査は_register_factory_登録名を未登録と誤検知しない() -> None:
    """registry 側の既知名は `registry.names()`（spec と factory の和集合）で読む。"""

    class _Catalog(AgentNames):
        """planner / writer を宣言した定数簿。"""

        PLANNER = "planner"
        WRITER = "writer"

    registry = AgentRegistry()
    registry.register(AgentSpec(name="planner", instructions="計画を立てる"))
    registry.register_factory("writer", lambda _registry: object())

    validate_agent_names(_Catalog, registry)  # 例外が出なければ OK


def test_整合検査は差分_0_件なら例外を送出しない() -> None:
    """宣言集合と登録名集合が一致する構成では何も起きない。"""
    registry = AgentRegistry()
    registry.register(AgentSpec(name="planner", instructions="計画を立てる"))
    registry.register(AgentSpec(name="writer", instructions="本文を書く"))

    validate_agent_names(_Names, registry)  # 例外が出なければ OK


def test_整合検査は空の定数簿と空の_registry_でも例外を送出しない() -> None:
    """双方が空なら差分 0 件（境界条件）。"""
    validate_agent_names(_Empty, AgentRegistry())
