"""L1: ToolRegistry / ToolSpec のロジック検証（agents 非依存・RED 先行）。

Task 1（`tool_registry.py`）のスコープに絞る。属性アクセス経由の SDK 結線
（`_adapters.build_function_tool`）は Task 2 依存のため monkeypatch でスタブ化し、
Registry 側のキャッシュ挙動・エラー文言・登録名検証のみを検証する。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from oai_agentspec import ToolRegistry, ToolSpec


# ----------------------------------------------------------------------
# ToolSpec dataclass
# ----------------------------------------------------------------------
def _dummy_fn() -> str:
    """テスト用の同期関数。"""
    return "ok"


async def _dummy_async_fn() -> str:
    """テスト用の非同期関数。"""
    return "ok"


def test_正常系_toolspec_最小引数で生成できる() -> None:
    """`func` のみ指定で他フィールドは既定値（enabled=True 等）。"""
    spec = ToolSpec(func=_dummy_fn)
    assert spec.func is _dummy_fn
    assert spec.name is None
    assert spec.enabled is True
    assert spec.needs_approval is None
    assert spec.timeout is None
    assert spec.timeout_behavior is None
    assert spec.timeout_error_function is None
    assert spec.name_override is None
    assert spec.description_override is None
    assert spec.strict_mode is None
    assert spec.extra == {}


def test_正常系_toolspec_全メタデータフィールドが型付きで指定できる() -> None:
    """全フィールドを指定して属性で取り出せる（characterization）。"""

    def _timeout_err(_ctx: object, _err: object) -> str:
        return "timeout"

    def _failure_err(_ctx: object, _err: object) -> str:
        return "failure"

    spec = ToolSpec(
        func=_dummy_fn,
        name="alias",
        enabled=False,
        needs_approval=True,
        timeout=10.0,
        timeout_behavior="raise",
        timeout_error_function=_timeout_err,
        failure_error_function=_failure_err,
        name_override="ov_name",
        description_override="ov_desc",
        strict_mode=False,
        extra={"defer_loading": True},
    )
    assert spec.name == "alias"
    assert spec.enabled is False
    assert spec.needs_approval is True
    assert spec.timeout == 10.0
    assert spec.timeout_behavior == "raise"
    assert spec.timeout_error_function is _timeout_err
    assert spec.failure_error_function is _failure_err
    assert spec.name_override == "ov_name"
    assert spec.description_override == "ov_desc"
    assert spec.strict_mode is False
    assert spec.extra == {"defer_loading": True}


def test_正常系_toolspec_extraは独立のdictである() -> None:
    """`extra` は `field(default_factory=dict)` で各インスタンス独立。"""
    s1 = ToolSpec(func=_dummy_fn)
    s2 = ToolSpec(func=_dummy_fn)
    s1.extra["k"] = "v"
    assert s2.extra == {}
    assert s1.extra is not s2.extra


def test_異常系_toolspec_idempotentフィールドは存在しない() -> None:
    """設計判断 8 のガード: `idempotent` は初版では持たない。"""
    with pytest.raises(TypeError):
        ToolSpec(func=_dummy_fn, idempotent=True)  # type: ignore[call-arg]
    spec = ToolSpec(func=_dummy_fn)
    assert not hasattr(spec, "idempotent")


# ----------------------------------------------------------------------
# ToolRegistry.register
# ----------------------------------------------------------------------
def test_正常系_register_単一Tool登録できる() -> None:
    reg = ToolRegistry()
    reg.register(ToolSpec(func=_dummy_fn, name="get_weather"))
    assert reg.names() == ["get_weather"]


def test_異常系_register_二重登録でValueError() -> None:
    reg = ToolRegistry()
    reg.register(ToolSpec(func=_dummy_fn, name="get_weather"))
    with pytest.raises(ValueError, match="tool already registered"):
        reg.register(ToolSpec(func=_dummy_fn, name="get_weather"))


def test_異常系_register_名前がアンダースコア始まりでValueError() -> None:
    reg = ToolRegistry()
    with pytest.raises(ValueError):
        reg.register(ToolSpec(func=_dummy_fn, name="_private"))


@pytest.mark.parametrize("bad_name", ["has space", "1number", "has-dash", ""])
def test_異常系_register_名前が非識別子でValueError(bad_name: str) -> None:
    reg = ToolRegistry()
    with pytest.raises(ValueError):
        reg.register(ToolSpec(func=_dummy_fn, name=bad_name))


@pytest.mark.parametrize("collide", ["names", "register", "metadata"])
def test_異常系_register_名前がToolRegistryの公開メソッド名と衝突するとValueError(
    collide: str,
) -> None:
    """属性アクセスで到達不能な名前を無音で許さない（設計判断 3）。"""
    reg = ToolRegistry()
    with pytest.raises(ValueError):
        reg.register(ToolSpec(func=_dummy_fn, name=collide))


@pytest.mark.parametrize("kw", ["class", "from", "return", "if", "None", "True", "False"])
def test_異常系_register_名前がPython予約語でValueError(kw: str) -> None:
    """Python 予約語は `isidentifier()` が True を返すが `registry.class` 等の

    属性アクセスは SyntaxError で到達不能。設計判断 3「到達不能な名前を無音で許さない」の
    ガード漏れ（Codex review [P2] 指摘）。
    """
    reg = ToolRegistry()
    with pytest.raises(ValueError, match="keyword"):
        reg.register(ToolSpec(func=_dummy_fn, name=kw))


def test_正常系_register_名前指定なしでfunc名が使われる() -> None:
    reg = ToolRegistry()

    def my_fn() -> str:
        return "x"

    reg.register(ToolSpec(func=my_fn))
    assert reg.names() == ["my_fn"]


def test_正常系_register_ToolSpec_name明示指定が優先される() -> None:
    reg = ToolRegistry()

    def my_fn() -> str:
        return "x"

    reg.register(ToolSpec(func=my_fn, name="alias"))
    assert reg.names() == ["alias"]


# ----------------------------------------------------------------------
# ToolRegistry.names
# ----------------------------------------------------------------------
def test_正常系_names_昇順で返る() -> None:
    reg = ToolRegistry()
    reg.register(ToolSpec(func=_dummy_fn, name="zeta"))
    reg.register(ToolSpec(func=_dummy_fn, name="alpha"))
    reg.register(ToolSpec(func=_dummy_fn, name="mango"))
    assert reg.names() == ["alpha", "mango", "zeta"]


def test_正常系_names_空の場合空リスト() -> None:
    reg = ToolRegistry()
    assert reg.names() == []


# ----------------------------------------------------------------------
# ToolRegistry.metadata
# ----------------------------------------------------------------------
def test_正常系_metadata_登録したToolSpecのliveインスタンスが返る() -> None:
    """属性代入で状態更新→再照会で反映（is 同一性）。"""
    reg = ToolRegistry()
    spec = ToolSpec(func=_dummy_fn, name="tool_a", enabled=True)
    reg.register(spec)
    got = reg.metadata("tool_a")
    assert got is spec
    got.enabled = False
    assert reg.metadata("tool_a").enabled is False


def test_異常系_metadata_未登録名でKeyError() -> None:
    """`match="unknown tool"` かつ登録済み名の案内が含まれる。"""
    reg = ToolRegistry()
    reg.register(ToolSpec(func=_dummy_fn, name="get_weather"))
    reg.register(ToolSpec(func=_dummy_fn, name="search_docs"))
    with pytest.raises(KeyError) as exc:
        reg.metadata("get_wether")
    msg = str(exc.value)
    assert "unknown tool" in msg
    assert "get_weather" in msg
    assert "search_docs" in msg


def test_異常系_metadata_未登録名で登録なし時に_none_表示() -> None:
    reg = ToolRegistry()
    with pytest.raises(KeyError) as exc:
        reg.metadata("missing")
    msg = str(exc.value)
    assert "unknown tool" in msg
    assert "(none)" in msg


# ----------------------------------------------------------------------
# ToolRegistry.__getattr__ (build 呼び出しは Task 2 依存のためスタブ)
# ----------------------------------------------------------------------
def test_異常系_getattr_未登録名でAttributeError() -> None:
    """`unknown tool` + 登録済み名一覧付き（`metadata()` と同一文言）。"""
    reg = ToolRegistry()
    reg.register(ToolSpec(func=_dummy_fn, name="get_weather"))
    with pytest.raises(AttributeError) as exc:
        _ = reg.get_wether
    msg = str(exc.value)
    assert "unknown tool" in msg
    assert "get_weather" in msg


def test_異常系_getattr_アンダースコア始まりは通常AttributeError() -> None:
    """`_` 始まりは登録済み名一覧を出さず Python 標準の未定義属性エラー。"""
    reg = ToolRegistry()
    reg.register(ToolSpec(func=_dummy_fn, name="get_weather"))
    with pytest.raises(AttributeError) as exc:
        _ = reg._something
    msg = str(exc.value)
    assert "unknown tool" not in msg
    assert "get_weather" not in msg


def test_異常系_getattr_specs未初期化時は素のAttributeErrorで無限再帰しない() -> None:
    """`__new__` バイパスで `_specs` 未初期化のインスタンスへの属性アクセスが

    `_unknown_tool_message` → `self.names()` → `self._specs` → `__getattr__` の無限再帰
    （RecursionError）にならず、素の `AttributeError`（`unknown tool` 文言を含まない）で
    早期終了する二段防御パスを検証する（unpickle 等の想定シナリオ）。
    """
    # `__init__` をバイパスして `_specs` / `_built` 未初期化状態を作る。
    reg = ToolRegistry.__new__(ToolRegistry)
    with pytest.raises(AttributeError) as exc:
        _ = reg.anything
    msg = str(exc.value)
    # 登録済み名一覧付きの分かりやすいメッセージには到達しない（防御パスは素の
    # AttributeError で早期終了する）。
    assert "unknown tool" not in msg
    # `RecursionError` にはならない（`pytest.raises` を通過している時点で確認）。


def test_正常系_getattr_登録済み名アクセス時にbuild_function_toolが1回だけ呼ばれキャッシュされる(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_built` キャッシュ（FR-3）: 同名アクセス 2 回で build は 1 回のみ。"""
    calls: list[ToolSpec] = []

    def _fake_build(spec: ToolSpec, enabled_supplier: object) -> object:
        calls.append(spec)
        return SimpleNamespace(name=spec.name, _stub=True)

    # Task 2 で実装される _adapters.build_function_tool をスタブ化。
    import oai_agentspec._adapters as adapters_module

    monkeypatch.setattr(adapters_module, "build_function_tool", _fake_build, raising=False)

    reg = ToolRegistry()
    reg.register(ToolSpec(func=_dummy_fn, name="get_weather"))
    t1 = reg.get_weather
    t2 = reg.get_weather
    assert t1 is t2
    assert len(calls) == 1


# ----------------------------------------------------------------------
# テスト意図サマリー
# ----------------------------------------------------------------------
# 1. ToolSpec の既定値・全フィールド指定・extra 独立性・冪等性フィールド不在をピン留め。
# 2. ToolRegistry.register の到達不能名（`_`/非識別子/公開メソッド衝突）と二重登録の拒否を検証。
# 3. metadata の live 参照・__getattr__ のキャッシュ + エラー文言
#    （_unknown_tool_message 単一ソース）を検証。
