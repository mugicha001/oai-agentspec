"""L2: serve サブコマンド（argparse + --registry 解決 + start_server 呼び出し）を検証する。

実 uvicorn は起動せず `start_server` をパッチする。`--registry` の `module:callable` 解決は
`sys.modules` へ注入したフェイクモジュールで検証する。httpx / websockets 未導入環境では
importorskip でスキップする（cli extra の入口モジュールを import するため）。
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

pytest.importorskip("httpx")
pytest.importorskip("websockets")

from pathlib import Path  # noqa: E402

from oai_agentspec.runtime.cli.main import (  # noqa: E402
    _build_session_policy,
    _resolve_registry,
    build_parser,
    main,
)

pytestmark = pytest.mark.integration


def test_build_parser_serve_defaults() -> None:
    """serve サブコマンドの既定（host / port）と --registry 必須を確認する。"""
    parser = build_parser()
    args = parser.parse_args(["serve", "--registry", "mymod:make"])
    assert args.command == "serve"
    assert args.registry == "mymod:make"
    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_build_parser_serve_requires_registry() -> None:
    """--registry 未指定は argparse がエラー終了する（SystemExit）。"""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve"])


def _install_fake_registry_module(monkeypatch: pytest.MonkeyPatch, name: str, factory: Any) -> None:
    """`name` モジュールに `make_registry` 属性を持つフェイクを sys.modules へ注入する。"""
    module = types.ModuleType(name)
    module.make_registry = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)


def test_resolve_registry_calls_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """--registry の module:callable を import して呼び出し、戻り値を返す。"""
    sentinel = object()
    _install_fake_registry_module(monkeypatch, "fake_reg_mod", lambda: sentinel)
    assert _resolve_registry("fake_reg_mod:make_registry") is sentinel


def test_resolve_registry_bad_format() -> None:
    """':' を含まない指定は ValueError。"""
    with pytest.raises(ValueError, match="module:callable"):
        _resolve_registry("no_colon")


def test_resolve_registry_missing_attr(monkeypatch: pytest.MonkeyPatch) -> None:
    """存在しない callable 名は ValueError。"""
    _install_fake_registry_module(monkeypatch, "fake_reg_mod2", lambda: None)
    with pytest.raises(ValueError, match="callable が見つかりません"):
        _resolve_registry("fake_reg_mod2:absent")


def test_resolve_registry_not_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    """callable でない属性は ValueError。"""
    module = types.ModuleType("fake_reg_mod3")
    module.make_registry = 123  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_reg_mod3", module)
    with pytest.raises(ValueError, match="callable ではありません"):
        _resolve_registry("fake_reg_mod3:make_registry")


def test_session_policy_flag_takes_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--session-db フラグは XDG_DATA_HOME より優先される。"""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    policy = _build_session_policy(str(tmp_path / "custom" / "my.db"), ephemeral=False)
    assert policy.base_dir == tmp_path / "custom"
    assert policy.db_name == "my.db"
    assert policy.persist is True


def test_session_policy_uses_xdg_data_home_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--session-db 未指定時は XDG_DATA_HOME 直下（サブフォルダなし）を使う。"""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    policy = _build_session_policy(None, ephemeral=False)
    assert policy.base_dir == tmp_path / "xdg"
    assert policy.db_name == "conversations.db"


def test_session_policy_default_is_project_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """--session-db も XDG_DATA_HOME も無ければ既定はプロジェクト直下 ./memory/。"""
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    policy = _build_session_policy(None, ephemeral=False)
    assert policy.base_dir == Path("memory")
    assert policy.db_name == "conversations.db"


def test_session_policy_ephemeral_disables_persist(monkeypatch: pytest.MonkeyPatch) -> None:
    """--ephemeral 指定で persist=False になる。"""
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    policy = _build_session_policy(None, ephemeral=True)
    assert policy.persist is False


def test_main_serve_invokes_start_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """main serve が registry を解決し start_server を呼ぶ（実起動しない）。"""
    registry_obj = object()
    _install_fake_registry_module(monkeypatch, "fake_reg_mod4", lambda: registry_obj)

    captured: dict[str, Any] = {}

    def _fake_start_server(
        registry: Any,
        *,
        host: str,
        port: int,
        session_policy: Any = None,
        entry_agent: str | None = None,
    ) -> None:
        captured["registry"] = registry
        captured["host"] = host
        captured["port"] = port
        captured["session_policy"] = session_policy
        captured["entry_agent"] = entry_agent

    monkeypatch.setattr("oai_agentspec.runtime.serve.app.start_server", _fake_start_server)
    rc = main(
        [
            "serve",
            "--registry",
            "fake_reg_mod4:make_registry",
            "--host",
            "0.0.0.0",
            "--port",
            "9999",
            "--entry",
            "triage",
        ]
    )
    assert rc == 0
    assert captured["registry"] is registry_obj
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9999
    assert captured["entry_agent"] == "triage"
    # 既定（--ephemeral 無し）は永続化方針 persist=True。
    assert captured["session_policy"].persist is True


def test_main_serve_ephemeral_sets_non_persistent_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--ephemeral 指定で session_policy.persist=False が渡る。"""
    registry_obj = object()
    _install_fake_registry_module(monkeypatch, "fake_reg_mod5", lambda: registry_obj)
    captured: dict[str, Any] = {}

    def _fake_start_server(registry: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("oai_agentspec.runtime.serve.app.start_server", _fake_start_server)
    rc = main(["serve", "--registry", "fake_reg_mod5:make_registry", "--ephemeral"])
    assert rc == 0
    assert captured["session_policy"].persist is False


def test_main_serve_bad_registry_returns_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--registry 解決失敗は 1 を返し案内を表示する（start_server を呼ばない）。"""
    called = False

    def _fake_start_server(*args: Any, **kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("oai_agentspec.runtime.serve.app.start_server", _fake_start_server)
    rc = main(["serve", "--registry", "no_colon"])
    assert rc == 1
    assert called is False
