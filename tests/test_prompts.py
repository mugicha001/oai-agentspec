"""L1: PromptStore のロード・合成・ヘルパー検証。"""

from __future__ import annotations

from pathlib import Path

import pytest

from oai_agentspec import (
    PromptLayout,
    PromptStore,
    PromptTemplateIntegrityError,
    dynamic_prompt,
)
from oai_agentspec.prompts import PromptResolutionError

CONVENTION = PromptLayout(base="base", parts="parts", agents="agents")


def build_root(tmp_path: Path) -> Path:
    (tmp_path / "base").mkdir()
    (tmp_path / "parts").mkdir()
    (tmp_path / "agents").mkdir()
    (tmp_path / "base" / "main.md").write_text(
        "---\nversion: 1\n---\nYou are ${company} staff.", encoding="utf-8"
    )
    (tmp_path / "base" / "sub.md").write_text("Sub base.", encoding="utf-8")
    (tmp_path / "parts" / "safety.md").write_text("Be safe.", encoding="utf-8")
    (tmp_path / "agents" / "triage.md").write_text("Route requests.", encoding="utf-8")
    return tmp_path


def test_default_composition_order(tmp_path: Path) -> None:
    store = PromptStore(build_root(tmp_path), CONVENTION)
    result = store.compose(
        agent="triage", base="main", parts=["safety"], vars={"company": "AgentSpec"}
    )
    assert result == "You are AgentSpec staff.\n\nBe safe.\n\nRoute requests."


def test_base_selects_sub(tmp_path: Path) -> None:
    store = PromptStore(build_root(tmp_path), CONVENTION)
    assert store.compose(agent="triage", base="sub").startswith("Sub base.")


def test_layout_override(tmp_path: Path) -> None:
    store = PromptStore(build_root(tmp_path), CONVENTION)
    assert store.compose(layout=["agent:triage", "part:safety"]) == "Route requests.\n\nBe safe."


def test_agent_only(tmp_path: Path) -> None:
    store = PromptStore(build_root(tmp_path), CONVENTION)
    assert store.compose(agent="triage") == "Route requests."


def test_empty_compose_raises(tmp_path: Path) -> None:
    store = PromptStore(build_root(tmp_path), CONVENTION)
    with pytest.raises(PromptResolutionError, match="合成対象がありません"):
        store.compose()


def test_missing_segment_raises(tmp_path: Path) -> None:
    store = PromptStore(build_root(tmp_path), CONVENTION)
    with pytest.raises(PromptResolutionError, match="part:ghost"):
        store.compose(agent="triage", parts=["ghost"])


def test_missing_base_raises(tmp_path: Path) -> None:
    # base を明示指定したのにファイルが無ければエラー（無音スキップしない）。
    store = PromptStore(build_root(tmp_path), CONVENTION)
    with pytest.raises(PromptResolutionError, match="base:other"):
        store.compose(agent="triage", base="other")


def test_invalid_segment_ref(tmp_path: Path) -> None:
    store = PromptStore(build_root(tmp_path), CONVENTION)
    with pytest.raises(PromptResolutionError):
        store.compose(layout=["bogus:foo"])


def test_reload_cache(tmp_path: Path) -> None:
    (tmp_path / "agents").mkdir()
    f = tmp_path / "agents" / "x.md"
    f.write_text("v1", encoding="utf-8")
    store = PromptStore(tmp_path, CONVENTION)
    assert store.compose(agent="x") == "v1"
    f.write_text("v2", encoding="utf-8")
    assert store.compose(agent="x") == "v1"  # キャッシュ有効
    store.reload()
    assert store.compose(agent="x") == "v2"


def test_compose_with_callable_vars_renders_per_ctx(tmp_path: Path) -> None:
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "greeter.md").write_text("Hello ${user} (${plan})", encoding="utf-8")
    store = PromptStore(tmp_path, CONVENTION)
    fn = store.compose(agent="greeter", vars=lambda ctx: {"user": ctx.user, "plan": ctx.plan})

    import inspect

    assert len(inspect.signature(fn).parameters) == 2

    class Ctx:
        user = "Mugi"
        plan = "premium"

    assert fn(Ctx(), None) == "Hello Mugi (premium)"


def test_custom_layout_dir_names(tmp_path: Path) -> None:
    (tmp_path / "common").mkdir()
    (tmp_path / "roles").mkdir()
    (tmp_path / "common" / "main.md").write_text("共通。", encoding="utf-8")
    (tmp_path / "roles" / "triage.md").write_text("個別。", encoding="utf-8")
    store = PromptStore(tmp_path, PromptLayout(base="common", parts="snippets", agents="roles"))
    assert store.compose(agent="triage", base="main") == "共通。\n\n個別。"


def test_flat_layout(tmp_path: Path) -> None:
    (tmp_path / "main.md").write_text("base。", encoding="utf-8")
    (tmp_path / "triage.md").write_text("agent。", encoding="utf-8")
    store = PromptStore(tmp_path, PromptLayout(base="", parts="", agents=""))
    assert store.compose(agent="triage", base="main") == "base。\n\nagent。"


def test_flat_does_not_recurse_into_other_dirs(tmp_path: Path) -> None:
    # base="" のとき agents/ 配下の同名 main.md を誤って拾わない。
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "main.md").write_text("WRONG", encoding="utf-8")
    (tmp_path / "agents" / "triage.md").write_text("triage 本体", encoding="utf-8")
    store = PromptStore(tmp_path, PromptLayout(base="", parts="parts", agents="agents"))
    with pytest.raises(PromptResolutionError, match="base:main"):
        store.compose(agent="triage", base="main")


def test_nested_segment_resolved_recursively(tmp_path: Path) -> None:
    (tmp_path / "agents" / "billing").mkdir(parents=True)
    (tmp_path / "agents" / "billing" / "refund.md").write_text("返金担当。", encoding="utf-8")
    store = PromptStore(tmp_path, CONVENTION)
    assert store.compose(agent="refund") == "返金担当。"


def test_nested_ambiguous_stem_raises(tmp_path: Path) -> None:
    (tmp_path / "agents" / "a").mkdir(parents=True)
    (tmp_path / "agents" / "b").mkdir(parents=True)
    (tmp_path / "agents" / "a" / "dup.md").write_text("A", encoding="utf-8")
    (tmp_path / "agents" / "b" / "dup.md").write_text("B", encoding="utf-8")
    store = PromptStore(tmp_path, CONVENTION)
    with pytest.raises(PromptResolutionError, match="曖昧"):
        store.compose(agent="dup")


def test_nested_explicit_subpath(tmp_path: Path) -> None:
    (tmp_path / "agents" / "a").mkdir(parents=True)
    (tmp_path / "agents" / "b").mkdir(parents=True)
    (tmp_path / "agents" / "a" / "dup.md").write_text("A", encoding="utf-8")
    (tmp_path / "agents" / "b" / "dup.md").write_text("B", encoding="utf-8")
    store = PromptStore(tmp_path, CONVENTION)
    assert store.compose(agent="a/dup") == "A"


def test_all_excludes_segment_cache_entries(tmp_path: Path) -> None:
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "triage.md").write_text("agent body", encoding="utf-8")
    (tmp_path / "shared.md").write_text("flat body", encoding="utf-8")
    store = PromptStore(tmp_path, CONVENTION)
    store.compose(agent="triage")  # agent:triage をキャッシュ
    flat = store.all()
    assert "shared" in flat
    assert all(":" not in k for k in flat)


def test_dynamic_prompt_returns_id_ref() -> None:
    fn = dynamic_prompt(lambda ctx: {"id": "pmpt_1", "version": "2"})

    class Data:
        context = object()

    assert fn(Data()) == {"id": "pmpt_1", "version": "2"}


# ----------------------------------------------------------------------
# 非公開 _verify_integrity / _preload（lockdown 内部用）
# ----------------------------------------------------------------------
def test_verify_integrity_runs_checks_sequentially(tmp_path: Path) -> None:
    """``_verify_integrity`` は checks を順次発火し、全件成功でも 1 回ずつ呼ばれる。"""
    store = PromptStore(build_root(tmp_path), CONVENTION)
    calls: list[str] = []

    def c1() -> None:
        calls.append("c1")

    def c2() -> None:
        calls.append("c2")

    store._verify_integrity([c1, c2])  # noqa: SLF001 - 非公開ヘルパの単体検証
    assert calls == ["c1", "c2"]


def test_verify_integrity_is_fail_closed(tmp_path: Path) -> None:
    """``_verify_integrity`` は最初の違反で残りの check をスキップする（fail-closed）。"""
    store = PromptStore(build_root(tmp_path), CONVENTION)
    calls: list[str] = []

    def good() -> None:
        calls.append("good")

    def bad() -> None:
        calls.append("bad")
        raise ValueError("violation")

    def never() -> None:  # pragma: no cover - 到達しないはず
        calls.append("never")

    with pytest.raises(ValueError, match="violation"):
        store._verify_integrity([good, bad, never])  # noqa: SLF001
    assert calls == ["good", "bad"]


def test_preload_populates_cache_for_segments(tmp_path: Path) -> None:
    """``_preload()`` 後の ``compose`` は disk を参照せず cache のみで動く。"""
    store = PromptStore(build_root(tmp_path), CONVENTION)
    store._preload()  # noqa: SLF001 - eager-load

    # _load_file が呼ばれたら違反する monkeypatch を仕込み、cache hit のみで動くことを検証。
    def explode(self, path):  # noqa: ANN001 - test stub
        raise AssertionError(f"disk アクセスが発生しました: {path}")

    # 直接 monkeypatch（pytest fixture が無いコンテキストで愚直に書き換え）。
    original = PromptStore._load_file
    PromptStore._load_file = explode  # type: ignore[assignment]
    try:
        result = store.compose(
            agent="triage", base="main", parts=["safety"], vars={"company": "AgentSpec"}
        )
        assert result == "You are AgentSpec staff.\n\nBe safe.\n\nRoute requests."
    finally:
        PromptStore._load_file = original  # type: ignore[assignment]


def test_preload_populates_flat_cache(tmp_path: Path) -> None:
    """root 直下の flat 配置ファイルも ``_preload`` で cache 充填される。"""
    (tmp_path / "shared.md").write_text("shared body", encoding="utf-8")
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "triage.md").write_text("triage body", encoding="utf-8")
    store = PromptStore(tmp_path, CONVENTION)
    store._preload()  # noqa: SLF001
    # cache キーが flat stem と segment key の両方を含む。
    assert "shared" in store._cache  # noqa: SLF001 - flat key
    assert "agent:triage" in store._cache  # noqa: SLF001 - segment key


def test_lazy_load_still_works_when_no_integrity_used(tmp_path: Path) -> None:
    """``_preload`` を呼ばなければ既存挙動（lazy load）と完全互換。"""
    store = PromptStore(build_root(tmp_path), CONVENTION)
    # cache は空（compose で lazy にロードされる）。
    assert store._cache == {}  # noqa: SLF001
    store.compose(agent="triage")
    assert "agent:triage" in store._cache  # noqa: SLF001


# ----------------------------------------------------------------------
# reload guard（lockdown 後の cache only 契約維持）
# ----------------------------------------------------------------------
def test_reload_after_lockdown_raises(tmp_path: Path) -> None:
    """``_preload()`` 後（``_locked=True``）の ``reload()`` は禁止され例外を raise する。

    lockdown 後の disk 改竄 → reload → 改竄プロンプト流入経路を遮断する。
    """
    store = PromptStore(build_root(tmp_path), CONVENTION)
    store._preload()  # noqa: SLF001 - lockdown 内部相当
    assert store._locked is True  # noqa: SLF001
    with pytest.raises(PromptTemplateIntegrityError, match="reload は禁止"):
        store.reload()


def test_reload_without_lockdown_still_works(tmp_path: Path) -> None:
    """``_preload`` を呼んでいない場合の ``reload()`` は既存挙動（cache clear のみ）。"""
    store = PromptStore(build_root(tmp_path), CONVENTION)
    store.compose(agent="triage")  # lazy load で cache に入る。
    assert "agent:triage" in store._cache  # noqa: SLF001
    store.reload()
    assert store._cache == {}  # noqa: SLF001


def test_lockdown_reverify_via_lockdown_call(tmp_path: Path) -> None:
    """``_preload()`` を 2 回呼んでも ``_locked`` は True のまま、 reload は禁止のまま。

    lockdown の冪等性に対応した擬似シナリオ（healthcheck 再発火相当）。
    """
    store = PromptStore(build_root(tmp_path), CONVENTION)
    store._preload()  # noqa: SLF001
    store._preload()  # noqa: SLF001 - 冪等で動く
    assert store._locked is True  # noqa: SLF001
    with pytest.raises(PromptTemplateIntegrityError):
        store.reload()


# ----------------------------------------------------------------------
# preload の stem alias（ネスト agent 等が cache hit する）
# ----------------------------------------------------------------------
def test_preload_caches_nested_stem_alias(tmp_path: Path) -> None:
    """``agents/billing/refund.md`` を preload した後、``agent="refund"`` が cache hit する。

    ネスト stem alias を _preload が登録しなければ、cache miss → disk 再読込となり
    「lockdown 後 disk 不参照」契約が崩れる。リグレッション防止。
    """
    (tmp_path / "base").mkdir()
    (tmp_path / "parts").mkdir()
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "billing").mkdir()
    (tmp_path / "agents" / "billing" / "refund.md").write_text("refund body", encoding="utf-8")

    store = PromptStore(tmp_path, CONVENTION)
    store._preload()  # noqa: SLF001
    # 両方のキーが cache に存在する。
    assert "agent:billing/refund" in store._cache  # noqa: SLF001
    assert "agent:refund" in store._cache  # noqa: SLF001 - stem alias

    # disk アクセス禁止 monkeypatch で cache only を保証。
    original = PromptStore._load_file

    def explode(self, path):  # noqa: ANN001 - test stub
        raise AssertionError(f"disk アクセスが発生しました: {path}")

    PromptStore._load_file = explode  # type: ignore[assignment]
    try:
        result = store.compose(agent="refund")
        assert "refund body" in result
    finally:
        PromptStore._load_file = original  # type: ignore[assignment]


def test_preload_does_not_alias_ambiguous_stem(tmp_path: Path) -> None:
    """同 stem が複数ある場合は alias を登録しない。

    曖昧 stem は cache 充填されない。``_preload`` 後は ``_locked=True`` のため、
    cache miss は cache only 契約により ``PromptTemplateIntegrityError`` で遮断される。
    """
    (tmp_path / "base").mkdir()
    (tmp_path / "parts").mkdir()
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "billing").mkdir()
    (tmp_path / "agents" / "support").mkdir()
    (tmp_path / "agents" / "billing" / "refund.md").write_text("b", encoding="utf-8")
    (tmp_path / "agents" / "support" / "refund.md").write_text("s", encoding="utf-8")

    store = PromptStore(tmp_path, CONVENTION)
    store._preload()  # noqa: SLF001
    # 各フルパスは cache に入るが alias は入らない（曖昧 stem）。
    assert "agent:billing/refund" in store._cache  # noqa: SLF001
    assert "agent:support/refund" in store._cache  # noqa: SLF001
    assert "agent:refund" not in store._cache  # noqa: SLF001 - alias を作らない
    # _locked=True 状態なので cache miss は cache only 契約で遮断される。
    with pytest.raises(PromptTemplateIntegrityError, match="segment cache miss"):
        store.compose(agent="refund")


def test_ambiguous_stem_without_preload_raises_resolution_error(tmp_path: Path) -> None:
    """``_preload`` を呼ばない場合の曖昧 stem は既存どおり ``PromptResolutionError``。

    既存挙動の保証（lockdown 不使用時は disk 走査して曖昧エラーを返す）。
    """
    (tmp_path / "base").mkdir()
    (tmp_path / "parts").mkdir()
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "billing").mkdir()
    (tmp_path / "agents" / "support").mkdir()
    (tmp_path / "agents" / "billing" / "refund.md").write_text("b", encoding="utf-8")
    (tmp_path / "agents" / "support" / "refund.md").write_text("s", encoding="utf-8")

    store = PromptStore(tmp_path, CONVENTION)
    # _preload を呼ばない → _locked=False のまま disk 走査して PromptResolutionError。
    assert store._locked is False  # noqa: SLF001
    with pytest.raises(PromptResolutionError):
        store.compose(agent="refund")


# ----------------------------------------------------------------------
# cache only 契約: lockdown 後の disk アクセス禁止
# ----------------------------------------------------------------------
def test_get_after_lockdown_unknown_name_raises_integrity_error(tmp_path: Path) -> None:
    """``_preload()`` 後の ``get("unknown")`` は ``PromptTemplateIntegrityError``。

    cache miss 時に disk アクセスせず即時 raise する（cache only 契約維持）。
    """
    store = PromptStore(build_root(tmp_path), CONVENTION)
    store._preload()  # noqa: SLF001

    # disk アクセス禁止 monkeypatch を仕込み、disk が触られないことを保証。
    original = PromptStore._load_file

    def explode(self, path):  # noqa: ANN001 - test stub
        raise AssertionError(f"disk アクセスが発生しました: {path}")

    PromptStore._load_file = explode  # type: ignore[assignment]
    try:
        with pytest.raises(PromptTemplateIntegrityError, match="cache miss"):
            store.get("unknown_template")
    finally:
        PromptStore._load_file = original  # type: ignore[assignment]


def test_load_segment_after_lockdown_unknown_segment_raises(tmp_path: Path) -> None:
    """``_preload()`` 後の ``compose(agent="undefined")`` は ``PromptTemplateIntegrityError``。

    segment cache miss 時に disk アクセスせず即時 raise する（cache only 契約維持）。
    """
    store = PromptStore(build_root(tmp_path), CONVENTION)
    store._preload()  # noqa: SLF001

    original = PromptStore._load_file

    def explode(self, path):  # noqa: ANN001 - test stub
        raise AssertionError(f"disk アクセスが発生しました: {path}")

    PromptStore._load_file = explode  # type: ignore[assignment]
    try:
        with pytest.raises(PromptTemplateIntegrityError, match="segment cache miss"):
            store.compose(agent="undefined_agent")
    finally:
        PromptStore._load_file = original  # type: ignore[assignment]


def test_load_segment_after_lockdown_known_alias_hits_cache(tmp_path: Path) -> None:
    """``_preload()`` 後の既知 alias（ネスト stem）は cache hit で disk アクセスなし。

    既存 P2 alias 機能が cache only 契約下でも引き続き動作することを保証する。
    """
    (tmp_path / "base").mkdir()
    (tmp_path / "parts").mkdir()
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "billing").mkdir()
    (tmp_path / "agents" / "billing" / "refund.md").write_text("refund body", encoding="utf-8")

    store = PromptStore(tmp_path, CONVENTION)
    store._preload()  # noqa: SLF001

    original = PromptStore._load_file

    def explode(self, path):  # noqa: ANN001 - test stub
        raise AssertionError(f"disk アクセスが発生しました: {path}")

    PromptStore._load_file = explode  # type: ignore[assignment]
    try:
        result = store.compose(agent="refund")  # stem alias 経由で cache hit。
        assert "refund body" in result
    finally:
        PromptStore._load_file = original  # type: ignore[assignment]


def test_preload_skips_alias_for_colon_in_stem(tmp_path: Path) -> None:
    """colon を含む stem は alias 登録されない（``kind:name`` 解釈の曖昧化回避）。

    例: ``agents/foo:bar.md`` で ``agent:foo:bar`` という alias を作ると、cache キーの
    解釈が ``partition(":")`` で曖昧化するためスキップする。フルパスキーは通常登録。
    """
    (tmp_path / "base").mkdir()
    (tmp_path / "parts").mkdir()
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "nested").mkdir()
    # ファイル名に colon を含む（OS 依存だが macOS/Linux では可）。
    (tmp_path / "agents" / "nested" / "foo:bar.md").write_text("colon body", encoding="utf-8")

    store = PromptStore(tmp_path, CONVENTION)
    store._preload()  # noqa: SLF001
    # fullpath キーは通常通り登録される（with_suffix で拡張子除去後の name）。
    assert "agent:nested/foo:bar" in store._cache  # noqa: SLF001
    # colon を含む stem の alias は登録しない（曖昧化回避）。
    assert "agent:foo:bar" not in store._cache  # noqa: SLF001


# ----------------------------------------------------------------------
# flat layout（空 subdir）の preload segment key 登録
# ----------------------------------------------------------------------
def test_preload_flat_layout_registers_segment_keys(tmp_path: Path) -> None:
    """flat layout（空 subdir）で root 直下ファイルが segment key でも cache される。

    ``PromptLayout(base="", parts="", agents="")`` では root 直下の ``main.md`` が
    ``compose(base="main")`` / ``compose(agent="main")`` / ``compose(parts=["main"])``
    で解決される。``_preload`` 後（``_locked=True``）でも disk 不参照で cache hit する。
    """
    (tmp_path / "main.md").write_text("main body", encoding="utf-8")
    (tmp_path / "style.md").write_text("style body", encoding="utf-8")
    flat = PromptLayout(base="", parts="", agents="")
    store = PromptStore(tmp_path, flat)
    store._preload()  # noqa: SLF001

    # flat key と 3 種の segment key が登録される。
    assert "main" in store._cache  # noqa: SLF001 - flat key
    assert "base:main" in store._cache  # noqa: SLF001
    assert "agent:main" in store._cache  # noqa: SLF001
    assert "part:main" in store._cache  # noqa: SLF001

    # disk アクセス禁止 monkeypatch で cache only を保証。
    original = PromptStore._load_file

    def explode(self, path):  # noqa: ANN001 - test stub
        raise AssertionError(f"disk アクセスが発生しました: {path}")

    PromptStore._load_file = explode  # type: ignore[assignment]
    try:
        assert store.compose(base="main") == "main body"
        assert store.compose(agent="main") == "main body"
        assert store.compose(parts=["main"]) == "main body"
        assert store.compose(base="main", parts=["style"]) == "main body\n\nstyle body"
    finally:
        PromptStore._load_file = original  # type: ignore[assignment]


def test_lockdown_flat_layout_segment_lookup_works(tmp_path: Path) -> None:
    """``lockdown(store=...)`` 後の flat layout segment 参照が回帰しない。

    flat layout を使う利用者が lockdown しても segment 参照が
    ``PromptTemplateIntegrityError`` を出さずに動く（回帰防止）。
    """
    import hashlib

    from oai_agentspec import lockdown

    # prompts root（flat layout）+ manifest。
    proot = tmp_path / "prompts"
    proot.mkdir()
    (proot / "main.md").write_text("# main\n", encoding="utf-8")
    (proot / "style.md").write_text("# style\n", encoding="utf-8")
    integ = proot / ".integrity"
    integ.mkdir()
    lines = []
    for f in sorted(proot.rglob("*.md")):
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        lines.append(f"{digest}  {f.relative_to(proot).as_posix()}")
    (integ / "sha256.manifest").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # src root + manifest。
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("# app\n", encoding="utf-8")
    src_integ = src / ".integrity"
    src_integ.mkdir()
    src_digest = hashlib.sha256((src / "app.py").read_bytes()).hexdigest()
    (src_integ / "sha256.manifest").write_text(f"{src_digest}  app.py\n", encoding="utf-8")

    store = PromptStore(proot, PromptLayout(base="", parts="", agents=""))
    lockdown(src, store=store, libs=False)
    # lockdown 後の segment 参照が成功する（cache hit・disk 不参照）。
    result = store.compose(base="main", parts=["style"])
    assert "main" in result
    assert "style" in result
