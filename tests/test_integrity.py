"""L1: ``lockdown`` ・ ``IntegrityError`` 階層・manifest 照合・配布物照合の検証。

``oai_agentspec.integrity`` の public/private 双方を ``agents`` 非依存（標準ライブラリ
のみ）でテストする。FR-1〜FR-4 / NFR-1, NFR-3, NFR-5 を網羅する。
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from oai_agentspec import (
    AgentRegistry,
    AgentSpec,
    IntegrityCheck,
    IntegrityError,
    PromptLayout,
    PromptStore,
    PromptTemplateIntegrityError,
    RegistryFrozenError,
    WorkflowFrozenError,
    WorkflowGraph,
    lockdown,
)
from oai_agentspec import integrity as integrity_mod
from oai_agentspec.constants import INTEGRITY_LOGGER_NAME
from oai_agentspec.integrity import (
    _detect_used_distributions,
    _distribution_check,
    _verify_directory_against_manifest,
)
from oai_agentspec.workflow import END, START

from _helpers.fake_builder import FakeAgentBuilder

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# 共通ヘルパー
# ----------------------------------------------------------------------
def _write_manifest(root: Path, files: dict[str, str]) -> Path:
    """root 配下のファイル群に対応する sha256 manifest を書き込む。

    Args:
        root: 検証対象ディレクトリ。
        files: 相対パス -> ファイル本文の dict。

    Returns:
        書き込んだ manifest の絶対パス。
    """
    manifest_dir = root / ".integrity"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "sha256.manifest"
    lines: list[str] = []
    for relative, body in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        lines.append(f"{digest}  {relative}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def _make_root(tmp_path: Path, name: str = "root") -> Path:
    """root ディレクトリ + manifest（hello.txt 1 件）を用意して返す。"""
    root = tmp_path / name
    root.mkdir()
    _write_manifest(root, {"hello.txt": "hi\n"})
    return root


def _make_store(tmp_path: Path, name: str = "store") -> PromptStore:
    """sha256 manifest 同梱済みの ``PromptStore`` を返す（agents 直下に 1 ファイル）。"""
    store_root = tmp_path / name
    store_root.mkdir()
    _write_manifest(
        store_root,
        {"agents/triage.md": "Route requests.\n"},
    )
    (store_root / "agents").mkdir(exist_ok=True)
    return PromptStore(store_root, PromptLayout(base="base", parts="parts", agents="agents"))


def _make_registry() -> AgentRegistry:
    reg = AgentRegistry(agent_builder=FakeAgentBuilder())
    reg.register(AgentSpec(name="a", instructions="a"))
    return reg


def _make_workflow() -> WorkflowGraph:
    wf = WorkflowGraph(name="wf")
    wf.add_function_node("f", fn=lambda msg, ctx: msg)
    wf.add_edge(START, "f")
    wf.add_edge("f", END)
    return wf


@pytest.fixture
def isolated_distribution_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """libs detect の検証対象を空集合に固定して 3 段目を高速・決定論にする。"""
    monkeypatch.setattr(integrity_mod, "_detect_used_distributions", lambda: set())
    yield


# ----------------------------------------------------------------------
# 例外階層
# ----------------------------------------------------------------------
def test_integrity_error_is_exception() -> None:
    """``IntegrityError`` は ``Exception`` を継承する。"""
    assert issubclass(IntegrityError, Exception)


def test_prompt_template_integrity_error_is_integrity_error() -> None:
    """``PromptTemplateIntegrityError`` は ``IntegrityError`` を継承する。"""
    assert issubclass(PromptTemplateIntegrityError, IntegrityError)


def test_registry_frozen_error_is_runtime_error_not_integrity_error() -> None:
    """``RegistryFrozenError`` は ``RuntimeError`` を継承し ``IntegrityError`` 系統と分離。"""
    assert issubclass(RegistryFrozenError, RuntimeError)
    assert not issubclass(RegistryFrozenError, IntegrityError)


def test_workflow_frozen_error_is_runtime_error_not_integrity_error() -> None:
    """``WorkflowFrozenError`` は ``RuntimeError`` を継承し ``IntegrityError`` 系統と分離。"""
    assert issubclass(WorkflowFrozenError, RuntimeError)
    assert not issubclass(WorkflowFrozenError, IntegrityError)


def test_except_integrity_catches_prompt_template_error() -> None:
    """``except IntegrityError`` で ``PromptTemplateIntegrityError`` を捕捉できる。"""
    with pytest.raises(IntegrityError):
        raise PromptTemplateIntegrityError("test")


def test_except_integrity_does_not_catch_registry_frozen() -> None:
    """``except IntegrityError`` で ``RegistryFrozenError`` は捕捉**されない**。"""
    with pytest.raises(RegistryFrozenError):
        try:
            raise RegistryFrozenError("test")
        except IntegrityError:  # pragma: no cover - 捕捉されないはず
            pytest.fail("RegistryFrozenError should not be caught by IntegrityError")


def test_except_integrity_does_not_catch_workflow_frozen() -> None:
    """``except IntegrityError`` で ``WorkflowFrozenError`` は捕捉**されない**。"""
    with pytest.raises(WorkflowFrozenError):
        try:
            raise WorkflowFrozenError("test")
        except IntegrityError:  # pragma: no cover - 捕捉されないはず
            pytest.fail("WorkflowFrozenError should not be caught by IntegrityError")


# ----------------------------------------------------------------------
# 公開 API スモーク
# ----------------------------------------------------------------------
def test_public_api_imports() -> None:
    """``oai_agentspec`` から integrity 系シンボル一式を直接 import できる。"""
    from oai_agentspec import (
        IntegrityCheck as _IC,
    )
    from oai_agentspec import (
        IntegrityError as _IE,
    )
    from oai_agentspec import (
        PromptTemplateIntegrityError as _PTE,
    )
    from oai_agentspec import (
        RegistryFrozenError as _RFE,
    )
    from oai_agentspec import (
        WorkflowFrozenError as _WFE,
    )
    from oai_agentspec import (
        lockdown as _ld,
    )

    assert _IC is IntegrityCheck
    assert _IE is IntegrityError
    assert _PTE is PromptTemplateIntegrityError
    assert _RFE is RegistryFrozenError
    assert _WFE is WorkflowFrozenError
    assert _ld is lockdown


# ----------------------------------------------------------------------
# lockdown 6 段順次処理
# ----------------------------------------------------------------------
def test_lockdown_root_only_succeeds(tmp_path: Path, isolated_distribution_detection: None) -> None:
    """root のみ渡しで 1 段目のみ実行され正常終了する。"""
    root = _make_root(tmp_path)
    lockdown(root, libs=False)  # libs=False で 3 段目スキップ。


def test_lockdown_all_stages_run_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_distribution_detection: None,
) -> None:
    """root / store / registry / workflow / checks 全部渡し時、6 段がすべて順番に実行される。"""
    root = _make_root(tmp_path, "root")
    store = _make_store(tmp_path, "store")
    registry = _make_registry()
    workflow = _make_workflow()

    call_order: list[str] = []

    real_verify = integrity_mod._verify_directory_against_manifest

    def spy_verify(target_root: Path, manifest: Path, *, exception_factory=IntegrityError) -> None:
        # root 検証 / store 検証を区別して記録（store は exception_factory が差し替わる）。
        if exception_factory is IntegrityError:
            call_order.append("verify_root")
        else:
            call_order.append("verify_store")
        real_verify(target_root, manifest, exception_factory=exception_factory)

    monkeypatch.setattr(integrity_mod, "_verify_directory_against_manifest", spy_verify)

    def custom_check() -> None:
        call_order.append("custom")

    original_freeze_reg = registry.freeze
    original_freeze_wf = workflow.freeze

    def spy_reg_freeze() -> None:
        call_order.append("freeze_registry")
        original_freeze_reg()

    def spy_wf_freeze() -> None:
        call_order.append("freeze_workflow")
        original_freeze_wf()

    monkeypatch.setattr(registry, "freeze", spy_reg_freeze)
    monkeypatch.setattr(workflow, "freeze", spy_wf_freeze)

    lockdown(root, store=store, registry=registry, workflow=workflow, checks=[custom_check])

    # 期待順序: root -> store -> (libs skip) -> custom -> reg-freeze -> wf-freeze
    assert call_order == [
        "verify_root",
        "verify_store",
        "custom",
        "freeze_registry",
        "freeze_workflow",
    ]


def test_lockdown_libs_false_skips_libs_detect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``libs=False`` で 3 段目 libs detect がスキップされる。"""
    root = _make_root(tmp_path)
    called = {"n": 0}

    def spy() -> set[str]:
        called["n"] += 1
        return set()

    monkeypatch.setattr(integrity_mod, "_detect_used_distributions", spy)
    lockdown(root, libs=False)
    assert called["n"] == 0


def test_lockdown_libs_true_invokes_detect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``libs=True``（既定）で 3 段目 libs detect が呼ばれる。"""
    root = _make_root(tmp_path)
    called = {"n": 0}

    def spy() -> set[str]:
        called["n"] += 1
        return set()

    monkeypatch.setattr(integrity_mod, "_detect_used_distributions", spy)
    lockdown(root)
    assert called["n"] == 1


def test_lockdown_checks_only_with_root(
    tmp_path: Path, isolated_distribution_detection: None
) -> None:
    """checks のみ追加渡し時、root verify と custom checks 段が実行される。"""
    root = _make_root(tmp_path)
    bag: list[str] = []

    def chk() -> None:
        bag.append("ran")

    lockdown(root, checks=[chk], libs=False)
    assert bag == ["ran"]


def test_lockdown_registry_freeze_only(
    tmp_path: Path, isolated_distribution_detection: None
) -> None:
    """registry のみ追加渡しで段 5 が実行され freeze 状態になる。"""
    root = _make_root(tmp_path)
    registry = _make_registry()
    lockdown(root, registry=registry, libs=False)
    with pytest.raises(RegistryFrozenError):
        registry.register(AgentSpec(name="late", instructions="x"))


def test_lockdown_workflow_freeze_only(
    tmp_path: Path, isolated_distribution_detection: None
) -> None:
    """workflow のみ追加渡しで段 6 が実行され freeze 状態になる。"""
    root = _make_root(tmp_path)
    wf = _make_workflow()
    lockdown(root, workflow=wf, libs=False)
    with pytest.raises(WorkflowFrozenError):
        wf.add_function_node("g", fn=lambda msg, ctx: msg)


# ----------------------------------------------------------------------
# fail-closed セマンティクス
# ----------------------------------------------------------------------
def test_root_verify_mismatch_does_not_freeze_downstream(
    tmp_path: Path, isolated_distribution_detection: None
) -> None:
    """root manifest 不一致なら ``IntegrityError`` raise + registry/workflow が未 freeze。"""
    root = _make_root(tmp_path)
    # ファイル内容を改竄して hash 不一致を発生させる。
    (root / "hello.txt").write_text("tampered\n", encoding="utf-8")

    registry = _make_registry()
    workflow = _make_workflow()

    with pytest.raises(IntegrityError):
        lockdown(root, registry=registry, workflow=workflow, libs=False)

    # 後段スキップにより freeze されていないことを add_* で確認。
    registry.register(AgentSpec(name="z", instructions="z"))
    workflow.add_function_node("g", fn=lambda msg, ctx: msg)


def test_store_verify_mismatch_skips_subsequent_stages(
    tmp_path: Path, isolated_distribution_detection: None
) -> None:
    """store manifest 不一致なら ``PromptTemplateIntegrityError`` + 以降スキップ。"""
    root = _make_root(tmp_path, "r")
    store = _make_store(tmp_path, "s")
    # store 配下のテンプレ本文を改竄。
    (store.root / "agents" / "triage.md").write_text("evil\n", encoding="utf-8")

    registry = _make_registry()
    workflow = _make_workflow()

    custom_called = {"n": 0}

    def chk() -> None:
        custom_called["n"] += 1

    with pytest.raises(PromptTemplateIntegrityError):
        lockdown(
            root,
            store=store,
            registry=registry,
            workflow=workflow,
            checks=[chk],
            libs=False,
        )
    # store verify 失敗後の custom check / registry freeze / workflow freeze は走らない。
    assert custom_called["n"] == 0
    registry.register(AgentSpec(name="z", instructions="z"))
    workflow.add_function_node("g", fn=lambda msg, ctx: msg)


def test_lockdown_store_missing_manifest_raises_prompt_error(
    tmp_path: Path, isolated_distribution_detection: None
) -> None:
    """store.root の manifest 不在は ``PromptTemplateIntegrityError`` を raise する。

    公開契約「store integrity 失敗は ``PromptTemplateIntegrityError``」を E2E で確認する。
    manifest 不在という最も一般的な失敗が基底 ``IntegrityError`` に潰れないこと。
    """
    root = _make_root(tmp_path, "r")
    # store.root にテンプレだけ置いて manifest を作らない。
    store_root = tmp_path / "s"
    store_root.mkdir()
    (store_root / "agents").mkdir()
    (store_root / "agents" / "triage.md").write_text("Route.\n", encoding="utf-8")
    store = PromptStore(store_root, PromptLayout(base="base", parts="parts", agents="agents"))

    with pytest.raises(PromptTemplateIntegrityError, match="manifest が見つかりません"):
        lockdown(root, store=store, libs=False)


def test_libs_detect_failure_skips_subsequent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """libs detect 失敗で ``IntegrityError`` + 以降の custom/freeze はスキップ。"""
    root = _make_root(tmp_path)
    registry = _make_registry()

    def boom() -> set[str]:
        raise IntegrityError("packages_distributions が壊れています")

    monkeypatch.setattr(integrity_mod, "_detect_used_distributions", boom)

    custom_called = {"n": 0}

    def chk() -> None:
        custom_called["n"] += 1

    with pytest.raises(IntegrityError, match="packages_distributions"):
        lockdown(root, registry=registry, checks=[chk])
    assert custom_called["n"] == 0
    # registry もまだ freeze されていない。
    registry.register(AgentSpec(name="z", instructions="z"))


def test_custom_check_violation_propagates_and_skips_remaining(
    tmp_path: Path, isolated_distribution_detection: None
) -> None:
    """custom check 違反が伝播し、違反より後の check と freeze はスキップ。"""
    root = _make_root(tmp_path)
    registry = _make_registry()
    workflow = _make_workflow()

    calls: list[str] = []

    def good() -> None:
        calls.append("good")

    def bad() -> None:
        calls.append("bad")
        raise IntegrityError("user violation")

    def never() -> None:  # pragma: no cover - 到達しないはず
        calls.append("never")

    with pytest.raises(IntegrityError, match="user violation"):
        lockdown(
            root,
            registry=registry,
            workflow=workflow,
            checks=[good, bad, never],
            libs=False,
        )
    # 違反した check より後の check は呼ばれない。
    assert calls == ["good", "bad"]
    # 段 5/6 もスキップ → registry / workflow は未 freeze。
    registry.register(AgentSpec(name="z", instructions="z"))
    workflow.add_function_node("g", fn=lambda msg, ctx: msg)


# ----------------------------------------------------------------------
# 冪等性
# ----------------------------------------------------------------------
def test_lockdown_is_idempotent_for_same_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同一引数で 2 回呼んでも 2 回目も成功し、root verify / libs detect は再実行される。"""
    root = _make_root(tmp_path)
    store = _make_store(tmp_path, "s")
    registry = _make_registry()
    workflow = _make_workflow()

    counter = {"verify": 0, "libs": 0}
    real_verify = integrity_mod._verify_directory_against_manifest

    def spy_verify(target_root, manifest, *, exception_factory=IntegrityError):
        if exception_factory is IntegrityError:
            counter["verify"] += 1
        real_verify(target_root, manifest, exception_factory=exception_factory)

    def spy_libs() -> set[str]:
        counter["libs"] += 1
        return set()

    monkeypatch.setattr(integrity_mod, "_verify_directory_against_manifest", spy_verify)
    monkeypatch.setattr(integrity_mod, "_detect_used_distributions", spy_libs)

    lockdown(root, store=store, registry=registry, workflow=workflow)
    lockdown(root, store=store, registry=registry, workflow=workflow)

    # 各回で root verify が再実行されている（冪等性検証のため）。
    assert counter["verify"] == 2
    assert counter["libs"] == 2


# ----------------------------------------------------------------------
# manifest 検証（_verify_directory_against_manifest）
# ----------------------------------------------------------------------
def test_verify_passes_on_well_formed_manifest(tmp_path: Path) -> None:
    """正常な manifest（sha256sum 互換）と内容一致なら例外を出さない。"""
    root = tmp_path / "r"
    root.mkdir()
    manifest = _write_manifest(root, {"a.txt": "A\n", "sub/b.txt": "B\n"})
    _verify_directory_against_manifest(root, manifest)  # 例外なし。


def test_verify_detects_unlisted_file(tmp_path: Path) -> None:
    """manifest 未掲載のファイルが root 配下にあれば ``IntegrityError``。"""
    root = tmp_path / "r"
    root.mkdir()
    manifest = _write_manifest(root, {"a.txt": "A\n"})
    (root / "extra.txt").write_text("uninvited\n", encoding="utf-8")
    with pytest.raises(IntegrityError, match="manifest 未掲載"):
        _verify_directory_against_manifest(root, manifest)


def test_verify_detects_missing_file(tmp_path: Path) -> None:
    """manifest 記載のファイルが存在しなければ ``IntegrityError``。"""
    root = tmp_path / "r"
    root.mkdir()
    manifest = _write_manifest(root, {"a.txt": "A\n", "b.txt": "B\n"})
    (root / "b.txt").unlink()
    with pytest.raises(IntegrityError, match="manifest 記載のファイルが存在しません"):
        _verify_directory_against_manifest(root, manifest)


def test_verify_detects_hash_mismatch_with_relative_path(tmp_path: Path) -> None:
    """hash 不一致時、メッセージに相対パスと sha256 が含まれる。"""
    root = tmp_path / "r"
    root.mkdir()
    manifest = _write_manifest(root, {"a.txt": "A\n"})
    (root / "a.txt").write_text("TAMPERED\n", encoding="utf-8")
    with pytest.raises(IntegrityError) as exc:
        _verify_directory_against_manifest(root, manifest)
    msg = str(exc.value)
    assert "a.txt" in msg
    assert "sha256" in msg


def test_verify_resolves_symlink_target(tmp_path: Path) -> None:
    """シンボリックリンクは target を解決して照合される。"""
    root = tmp_path / "r"
    root.mkdir()
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target_file = target_dir / "real.txt"
    target_file.write_text("LINKED\n", encoding="utf-8")
    link_path = root / "alias.txt"
    link_path.symlink_to(target_file)
    digest = hashlib.sha256(b"LINKED\n").hexdigest()
    manifest_dir = root / ".integrity"
    manifest_dir.mkdir()
    manifest = manifest_dir / "sha256.manifest"
    manifest.write_text(f"{digest}  alias.txt\n", encoding="utf-8")
    _verify_directory_against_manifest(root, manifest)


def test_verify_detects_symlink_loop(tmp_path: Path) -> None:
    """シンボリックリンクループは ``IntegrityError``（resolve(strict=True) 失敗）。"""
    root = tmp_path / "r"
    root.mkdir()
    link_a = root / "a.txt"
    link_b = root / "b.txt"
    link_a.symlink_to(link_b)
    link_b.symlink_to(link_a)
    # 適当な hash を入れた manifest を用意（実際にはリンク解決で死ぬ）。
    manifest_dir = root / ".integrity"
    manifest_dir.mkdir()
    manifest = manifest_dir / "sha256.manifest"
    manifest.write_text("0" * 64 + "  a.txt\n" + "0" * 64 + "  b.txt\n", encoding="utf-8")
    with pytest.raises(IntegrityError):
        _verify_directory_against_manifest(root, manifest)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO 未対応 OS")
def test_verify_rejects_special_file(tmp_path: Path) -> None:
    """FIFO 等の特殊ファイルが root 配下に存在すれば ``IntegrityError``。"""
    root = tmp_path / "r"
    root.mkdir()
    manifest_dir = root / ".integrity"
    manifest_dir.mkdir()
    manifest = manifest_dir / "sha256.manifest"
    manifest.write_text("0" * 64 + "  fifo.pipe\n", encoding="utf-8")
    os.mkfifo(root / "fifo.pipe")  # type: ignore[attr-defined]
    with pytest.raises(IntegrityError, match="特殊ファイル"):
        _verify_directory_against_manifest(root, manifest)


def test_verify_skips_comment_and_blank_lines(tmp_path: Path) -> None:
    """``#`` 始まりのコメント行と空行は manifest パーサがスキップする。"""
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.txt").write_text("A\n", encoding="utf-8")
    digest = hashlib.sha256(b"A\n").hexdigest()
    manifest_dir = root / ".integrity"
    manifest_dir.mkdir()
    manifest = manifest_dir / "sha256.manifest"
    manifest.write_text(
        f"# header comment\n\n{digest}  a.txt\n# trailing comment\n",
        encoding="utf-8",
    )
    _verify_directory_against_manifest(root, manifest)


def test_verify_detects_corrupt_manifest_line(tmp_path: Path) -> None:
    """パース不能な manifest 行は ``IntegrityError`` を raise する。"""
    root = tmp_path / "r"
    root.mkdir()
    manifest_dir = root / ".integrity"
    manifest_dir.mkdir()
    manifest = manifest_dir / "sha256.manifest"
    # スペースを含まない 1 トークン行 → パース失敗。
    manifest.write_text("garbage_line_without_path\n", encoding="utf-8")
    with pytest.raises(IntegrityError):
        _verify_directory_against_manifest(root, manifest)


def test_verify_missing_manifest_file_raises(tmp_path: Path) -> None:
    """manifest 自体が存在しないとき ``IntegrityError`` を raise する。"""
    root = tmp_path / "r"
    root.mkdir()
    with pytest.raises(IntegrityError, match="manifest が見つかりません"):
        _verify_directory_against_manifest(root, root / ".integrity" / "sha256.manifest")


def test_verify_nonexistent_root_raises(tmp_path: Path) -> None:
    """存在しない root を渡すと ``IntegrityError`` を raise する。"""
    missing = tmp_path / "absent"
    manifest_dir = tmp_path / ".integrity"
    manifest_dir.mkdir()
    manifest = manifest_dir / "sha256.manifest"
    manifest.write_text("\n", encoding="utf-8")
    with pytest.raises(IntegrityError, match="検証対象のディレクトリ"):
        _verify_directory_against_manifest(missing, manifest)


def test_verify_directory_missing_manifest_uses_exception_factory(tmp_path: Path) -> None:
    """manifest 不在時、``exception_factory`` 系統（派生例外）が raise される。

    store verify では ``PromptTemplateIntegrityError`` を渡すため、manifest 不在という
    最も一般的な失敗も派生例外として届く（基底 ``IntegrityError`` に潰さない）。
    """
    root = tmp_path / "r"
    root.mkdir()
    manifest = root / ".integrity" / "sha256.manifest"  # 作らない。
    with pytest.raises(PromptTemplateIntegrityError, match="manifest が見つかりません"):
        _verify_directory_against_manifest(
            root, manifest, exception_factory=PromptTemplateIntegrityError
        )


def test_verify_directory_malformed_manifest_uses_exception_factory(tmp_path: Path) -> None:
    """manifest 破損（不正な行）時も ``exception_factory`` 系統が raise される。"""
    root = tmp_path / "r"
    root.mkdir()
    manifest_dir = root / ".integrity"
    manifest_dir.mkdir()
    manifest = manifest_dir / "sha256.manifest"
    manifest.write_text("garbage_line_without_path\n", encoding="utf-8")
    with pytest.raises(PromptTemplateIntegrityError):
        _verify_directory_against_manifest(
            root, manifest, exception_factory=PromptTemplateIntegrityError
        )


# ----------------------------------------------------------------------
# PEP 376 RECORD 配布物検証（_distribution_check）
# ----------------------------------------------------------------------
def test_distribution_check_succeeds_when_files_meta_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``files`` が空（hash 確認すべきエントリ無し）なら例外無く完了する。"""

    class _StubDist:
        files: list = []

        def locate_file(self, _entry):
            return Path("/tmp")

    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _StubDist())
    _distribution_check("stub-dist")()  # 例外無し。


def test_distribution_check_missing_distribution_raises() -> None:
    """存在しない配布物名は ``IntegrityError`` を raise する。"""
    with pytest.raises(IntegrityError, match="配布物が見つかりません"):
        _distribution_check("this_distribution_does_not_exist_xyz")()


def test_distribution_check_raises_when_files_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``dist.files`` が None（RECORD 欠落）なら ``IntegrityError`` を raise する。"""

    class _NoRecordDist:
        files = None

        def locate_file(self, _entry):  # pragma: no cover - 到達しない
            return Path("/tmp")

    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _NoRecordDist())
    with pytest.raises(IntegrityError, match="RECORD"):
        _distribution_check("no-record-dist")()


def test_distribution_check_rejects_md5_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RECORD に md5 が含まれる配布物は ``IntegrityError`` を raise（アルゴリズム名を含む）。"""

    target = tmp_path / "f.txt"
    target.write_text("x", encoding="utf-8")

    class _Hash:
        mode = "md5"
        value = "9e107d9d372bb6826bd81d3542a419d6"

    class _Entry:
        hash = _Hash()

        def __str__(self) -> str:
            return "f.txt"

    class _Md5Dist:
        files = [_Entry()]

        def locate_file(self, _entry):
            return target

    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _Md5Dist())
    with pytest.raises(IntegrityError, match="md5"):
        _distribution_check("md5-dist")()


def test_distribution_check_skips_empty_hash_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``mode`` や ``value`` が空のエントリ（RECORD 自身など）は skip される。"""

    class _EmptyHash:
        mode = ""
        value = ""

    class _Entry:
        hash = _EmptyHash()

    class _StubDist:
        files = [_Entry()]

        def locate_file(self, _entry):  # pragma: no cover - skip されるはず
            return tmp_path / "should-not-be-read"

    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _StubDist())
    _distribution_check("empty-hash-dist")()  # 例外無く完了。


def _b64_nopad(data: bytes, algorithm: str = "sha256") -> str:
    """PEP 376 RECORD 互換の urlsafe-base64-nopad sha256 を計算する（テスト用）。"""
    digest = hashlib.new(algorithm, data).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def test_distribution_check_accepts_pep376_b64_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PEP 376 RECORD の urlsafe-base64-nopad sha256 と一致する場合に成功する。"""
    body = b"matched content\n"
    target = tmp_path / "real.py"
    target.write_bytes(body)

    class _Hash:
        mode = "sha256"
        value = _b64_nopad(body)

    class _Entry:
        hash = _Hash()

        def __str__(self) -> str:
            return "real.py"

    class _OkDist:
        files = [_Entry()]

        def locate_file(self, entry):
            if entry == "":
                return tmp_path
            return target

    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _OkDist())
    _distribution_check("ok-dist")()  # 例外なし。


def test_distribution_check_detects_b64_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PEP 376 b64 hash が一致しない場合に ``IntegrityError`` を raise する。"""
    target = tmp_path / "real.py"
    target.write_bytes(b"actual content\n")

    class _Hash:
        mode = "sha256"
        # 別内容の hash を入れて不一致を発生させる。
        value = _b64_nopad(b"different content\n")

    class _Entry:
        hash = _Hash()

        def __str__(self) -> str:
            return "real.py"

    class _MismatchDist:
        files = [_Entry()]

        def locate_file(self, entry):
            if entry == "":
                return tmp_path
            return target

    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _MismatchDist())
    with pytest.raises(IntegrityError, match="hash 不一致"):
        _distribution_check("mismatch-dist")()


def test_distribution_check_rejects_hex_hash_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """hex 値（旧 lib バグ）を RECORD に入れた配布物は b64 比較で必ず不一致になる。

    リグレッション防止: 過去に hex で比較していた実装は hex 値の RECORD を **誤って成功**
    と判定してしまうため、本テストで b64 比較に切り替わったことを保証する。
    """
    body = b"hello world\n"
    target = tmp_path / "real.py"
    target.write_bytes(body)

    class _Hash:
        mode = "sha256"
        # hex（旧仕様）を入れる。b64 比較なら一致しない。
        value = hashlib.sha256(body).hexdigest()

    class _Entry:
        hash = _Hash()

        def __str__(self) -> str:
            return "real.py"

    class _HexDist:
        files = [_Entry()]

        def locate_file(self, entry):
            if entry == "":
                return tmp_path
            return target

    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _HexDist())
    with pytest.raises(IntegrityError, match="hash 不一致"):
        _distribution_check("hex-dist")()


def test_supported_hash_algorithm_rejects_md5() -> None:
    """``md5`` は明示的に拒否される（暗号学的に弱い）。"""
    with pytest.raises(IntegrityError, match="md5"):
        integrity_mod._supported_hash_algorithm("md5")


def test_supported_hash_algorithm_rejects_sha1() -> None:
    """``sha1`` は明示的に拒否される。"""
    with pytest.raises(IntegrityError, match="sha1"):
        integrity_mod._supported_hash_algorithm("sha1")


def test_supported_hash_algorithm_rejects_uppercase_md5() -> None:
    """``MD5`` のように大文字でも拒否される（正規化後判定）。"""
    with pytest.raises(IntegrityError, match="md5"):
        integrity_mod._supported_hash_algorithm("MD5")


def test_supported_hash_algorithm_rejects_padded_sha1() -> None:
    """前後空白付き ``" sha1 "`` も拒否される。"""
    with pytest.raises(IntegrityError, match="sha1"):
        integrity_mod._supported_hash_algorithm(" sha1 ")


def test_supported_hash_algorithm_rejects_empty() -> None:
    """空文字のアルゴリズム名は拒否される。"""
    with pytest.raises(IntegrityError, match="空"):
        integrity_mod._supported_hash_algorithm("")


def test_supported_hash_algorithm_accepts_sha256() -> None:
    """sha256 は正規化されて返る。"""
    assert integrity_mod._supported_hash_algorithm("SHA256") == "sha256"


def test_supported_hash_algorithm_rejects_unknown() -> None:
    """guaranteed に無いアルゴリズムは拒否される。"""
    with pytest.raises(IntegrityError, match="未サポート"):
        integrity_mod._supported_hash_algorithm("not_a_real_hash")


# ----------------------------------------------------------------------
# _detect_used_distributions
# ----------------------------------------------------------------------
def test_detect_used_distributions_returns_imported_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sys.modules`` のトップレベルが ``packages_distributions()`` で配布物名にマップされる。"""

    # 制御された mapping と sys.modules で結果集合を厳密に検証する。
    monkeypatch.setattr(
        importlib.metadata,
        "packages_distributions",
        lambda: {"yaml": ["pyyaml"], "agents": ["openai-agents"]},
    )
    found = _detect_used_distributions()
    # yaml と agents は両方 sys.modules にあるため、配布物名が両方拾われる。
    assert "pyyaml" in found
    assert "openai-agents" in found


def test_detect_used_distributions_returns_set_for_real_env() -> None:
    """実環境呼び出しでも例外無く ``set`` を返す（fail-closed 失敗時を除く）。"""
    found = _detect_used_distributions()
    assert isinstance(found, set)


def test_detect_used_distributions_wraps_os_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``packages_distributions()`` の ``OSError`` は ``IntegrityError`` で wrap される。

    環境依存の I/O 失敗（読み取り権限不足など）は fail-closed して空セット偽陰性を防ぐ。
    """

    def boom() -> dict[str, list[str]]:
        raise OSError("metadata unreadable")

    monkeypatch.setattr(importlib.metadata, "packages_distributions", boom)
    with pytest.raises(IntegrityError, match="配布物メタデータ取得に失敗"):
        _detect_used_distributions()


def test_detect_used_distributions_propagates_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bug 由来例外（``AttributeError`` 等）は捕捉せず propagate する。

    stdlib / 依存ライブラリ側のバグを「環境依存の I/O 失敗」に紛れ込ませない設計。
    """

    def boom() -> dict[str, list[str]]:
        raise AttributeError("unexpected attribute error")

    monkeypatch.setattr(importlib.metadata, "packages_distributions", boom)
    # IntegrityError ではなく AttributeError がそのまま propagate されるはず。
    with pytest.raises(AttributeError, match="unexpected"):
        _detect_used_distributions()


# ----------------------------------------------------------------------
# 構造化ロギング（NFR-5）
# ----------------------------------------------------------------------
def test_lockdown_emits_start_and_complete_on_success(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    isolated_distribution_detection: None,
) -> None:
    """成功時は ``lockdown.start`` と ``lockdown.complete`` の両方が emit される。"""
    root = _make_root(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=INTEGRITY_LOGGER_NAME):
        lockdown(root, libs=False)
    messages = [r.message for r in caplog.records if r.name == INTEGRITY_LOGGER_NAME]
    assert "lockdown.start" in messages
    assert "lockdown.complete" in messages


def test_lockdown_start_extra_contains_inputs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    isolated_distribution_detection: None,
) -> None:
    """``lockdown.start`` の ``extra`` に root / libs / checks_count を含む。"""
    root = _make_root(tmp_path)

    def chk() -> None:
        return None

    with caplog.at_level(logging.INFO, logger=INTEGRITY_LOGGER_NAME):
        lockdown(root, checks=[chk, chk], libs=False)
    start = next(r for r in caplog.records if r.message == "lockdown.start")
    assert start.__dict__["root"] == str(root)
    assert start.__dict__["libs"] is False
    assert start.__dict__["checks_count"] == 2


def test_lockdown_complete_extra_contains_duration(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    isolated_distribution_detection: None,
) -> None:
    """``lockdown.complete`` の ``extra`` に ``duration_ms`` を含む。"""
    root = _make_root(tmp_path)
    with caplog.at_level(logging.INFO, logger=INTEGRITY_LOGGER_NAME):
        lockdown(root, libs=False)
    complete = next(r for r in caplog.records if r.message == "lockdown.complete")
    assert "duration_ms" in complete.__dict__
    assert isinstance(complete.__dict__["duration_ms"], int)


def test_lockdown_violation_emitted_before_raise(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    isolated_distribution_detection: None,
) -> None:
    """違反発生時、``lockdown.violation`` が raise の直前に emit される（complete は出ない）。"""
    root = _make_root(tmp_path)
    (root / "hello.txt").write_text("evil\n", encoding="utf-8")  # 改竄。
    with caplog.at_level(logging.DEBUG, logger=INTEGRITY_LOGGER_NAME):
        with pytest.raises(IntegrityError):
            lockdown(root, libs=False)
    names = [r.message for r in caplog.records if r.name == INTEGRITY_LOGGER_NAME]
    assert "lockdown.start" in names
    assert "lockdown.violation" in names
    assert "lockdown.complete" not in names  # 例外時は出ない。

    violation = next(r for r in caplog.records if r.message == "lockdown.violation")
    assert violation.__dict__["stage"] == "root_verify"
    assert "IntegrityError" in violation.__dict__["error_type"]


def test_lockdown_stage_logs_emit_debug(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    isolated_distribution_detection: None,
) -> None:
    """``lockdown.stage`` が DEBUG レベルで stage / status を運ぶ。"""
    root = _make_root(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=INTEGRITY_LOGGER_NAME):
        lockdown(root, libs=False)
    stage_records = [r for r in caplog.records if r.message == "lockdown.stage"]
    assert stage_records, "lockdown.stage は少なくとも 1 件 emit される"
    statuses = {r.__dict__["status"] for r in stage_records}
    assert {"start", "success"} <= statuses


def test_lockdown_logger_name_is_configurable_namespace() -> None:
    """logger 名が ``oai_agentspec.integrity`` 名前空間で固定されている。"""
    assert INTEGRITY_LOGGER_NAME == "oai_agentspec.integrity"
    logger = logging.getLogger(INTEGRITY_LOGGER_NAME)
    assert logger.name == "oai_agentspec.integrity"


# ----------------------------------------------------------------------
# 雑多な内部 helper（_iter_files / _ensure_regular_file_or_symlink）
# ----------------------------------------------------------------------
def test_iter_files_excludes_integrity_dir(tmp_path: Path) -> None:
    """``.integrity/`` 配下は manifest 自身として走査対象から除外される。"""
    root = tmp_path / "r"
    root.mkdir()
    _write_manifest(root, {"a.txt": "A\n"})  # manifest は .integrity 配下に置かれる。
    files = integrity_mod._iter_files(root)
    rels = sorted(p.relative_to(root).as_posix() for p in files)
    assert rels == ["a.txt"]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO 未対応 OS")
def test_iter_files_raises_on_special_file(tmp_path: Path) -> None:
    """FIFO を走査時に検出すると ``IntegrityError`` を raise する。"""
    root = tmp_path / "r"
    root.mkdir()
    os.mkfifo(root / "pipe")  # type: ignore[attr-defined]
    with pytest.raises(IntegrityError, match="特殊ファイル"):
        integrity_mod._iter_files(root)


# ----------------------------------------------------------------------
# IntegrityCheck 型エイリアス
# ----------------------------------------------------------------------
def test_integrity_check_alias_callable() -> None:
    """``IntegrityCheck`` 型エイリアスに合致する callable を ``checks`` へ渡せる。"""

    def chk() -> None:
        return None

    # mypy 等の静的検証では到達しないが、ランタイムで callable であることのみ確認。
    check: IntegrityCheck = chk
    check()
