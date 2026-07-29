"""L1: スロットヘルパ `prompt_slot`（PromptStore / AgentRegistry 読み取り）。

`prompt_slot` の seed 取得（`${var}` 保持）・既定 build（registry 登録 spec 複製で instructions
差し替え・tools 等保持・元 spec 不変）・固定部分（base / parts）合成・vars 保持（seed 非展開）・
registry 未解決の fail-closed ValueError・`build=` 明示経路（registry 不要）を網羅する。
PromptStore は実ファイル（tmp_path）から読み取る（実 LLM 非依存）。
ファイル I/O を伴うため `@pytest.mark.unit` だが外部実通信はしない。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oai_agentspec import AgentRegistry, AgentSpec, function_tool
from oai_agentspec.prompts import PromptLayout, PromptStore
from oai_agentspec.runtime.lightning import (
    FailureKind,
    OptimizeError,
    Slot,
    prompt_slot,
    prompt_slot_factory,
)

from _helpers.fake_model import FakeModel

pytestmark = pytest.mark.unit


def _store(tmp_path: Path) -> PromptStore:
    """tmp_path 配下に flat な tune セグメント・base/parts サブディレクトリを置いた PromptStore。"""
    # flat 配置（store.get(tune) 用・seed として `${var}` を保持）。
    (tmp_path / "bot.md").write_text("You are ${role}. Be helpful.", encoding="utf-8")
    (tmp_path / "triage.md").write_text("Triage prompt ${tone}", encoding="utf-8")
    (tmp_path / "billing.md").write_text("Billing prompt", encoding="utf-8")
    # base / parts セグメント。
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "main.md").write_text("BASE ${org}", encoding="utf-8")
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    (parts_dir / "style.md").write_text("STYLE part", encoding="utf-8")
    return PromptStore(tmp_path, PromptLayout(base="base", parts="parts", agents="agents"))


def _registry() -> AgentRegistry:
    """tune セグメント名と同名の spec を登録した registry。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="orig", model=FakeModel(), tools=[]))
    reg.register(AgentSpec(name="triage", instructions="orig", model=FakeModel()))
    reg.register(AgentSpec(name="billing", instructions="orig", model=FakeModel()))
    return reg


# ----------------------------------------------------------------------
# prompt_slot: seed 取得・vars 保持
# ----------------------------------------------------------------------


def test_prompt_slot_reads_seed_with_placeholders(tmp_path: Path) -> None:
    """seed は `store.get(tune).body`（`${var}` プレースホルダ保持）から読み取る。"""
    slot = prompt_slot(_store(tmp_path), _registry(), agent="bot")
    assert isinstance(slot, Slot)
    assert slot.name == "bot"
    assert slot.seed == "You are ${role}. Be helpful."


def test_prompt_slot_keeps_vars_without_expanding_seed(tmp_path: Path) -> None:
    """vars は Slot.vars に保持され seed には展開しない（最適化対象外・rollout 再注入）。"""
    slot = prompt_slot(_store(tmp_path), _registry(), agent="bot", vars={"role": "agent"})
    # seed は `${role}` のまま（展開しない）。
    assert "${role}" in slot.seed
    assert slot.vars == {"role": "agent"}


# ----------------------------------------------------------------------
# prompt_slot: 既定 build（registry 登録 spec 複製で instructions 差し替え）
# ----------------------------------------------------------------------


def test_default_build_replaces_instructions_only(tmp_path: Path) -> None:
    """既定 build は登録 spec を複製し instructions のみ候補で差し替える（tools 等は保持）。"""

    @function_tool(name_override="search")
    def _search(q: str) -> str:
        """ダミーツール。"""
        return "r"

    reg = AgentRegistry()
    orig = AgentSpec(name="bot", instructions="orig", model=FakeModel(), tools=[_search])
    reg.register(orig)

    slot = prompt_slot(_store(tmp_path), reg, agent="bot")
    built = slot.build("NEW INSTRUCTIONS")

    assert isinstance(built, AgentSpec)
    assert built.instructions == "NEW INSTRUCTIONS"
    # tools / model は登録 spec から複製される（再宣言不要）。
    assert [t.name for t in built.tools] == ["search"]
    # 元 registry の spec は不変（複製であり instructions は "orig" のまま）。
    assert reg._specs["bot"].instructions == "orig"  # noqa: SLF001


def test_default_build_prepends_fixed_part(tmp_path: Path) -> None:
    """固定部分（base / parts）を指定すると候補テキストに `\\n\\n` で前置する。"""
    slot = prompt_slot(
        _store(tmp_path),
        _registry(),
        agent="bot",
        base="main",
        parts=["style"],
        vars={"org": "AgentSpec"},
    )
    built = slot.build("CANDIDATE")
    # fixed の `${org}` は build 時に再注入され、candidate を後置。
    assert built.instructions == "BASE AgentSpec\n\nSTYLE part\n\nCANDIDATE"


def test_default_build_without_fixed_uses_candidate_only(tmp_path: Path) -> None:
    """固定部分なしのとき instructions は候補テキストのみ（前置の空連結を入れない）。"""
    slot = prompt_slot(_store(tmp_path), _registry(), agent="bot")
    built = slot.build("ONLY CANDIDATE")
    assert built.instructions == "ONLY CANDIDATE"


def test_prompt_slot_segments_carry_fixed_content_in_new_shape(tmp_path: Path) -> None:
    """新 shape では base+parts の固定内容は `Slot.segments` が SoT として保持する。

    `_recompose_new_shape_results` / `_new_default_build` の双方が segments 経由で参照するため、
    fixed 側の redundant な合成文字列は保持しない（compose_from_marked が二重合成しない）。"""
    slot = prompt_slot(
        _store(tmp_path),
        _registry(),
        agent="bot",
        base="main",
        parts=["style"],
        vars={"org": "AgentSpec"},
    )
    # 固定セグメントは `${var}` 保持のまま segments に格納される。
    fixed_segments = [s for s in slot.segments if not s.tune]
    assert [s.ref for s in fixed_segments] == ["base:main", "part:style"]
    assert fixed_segments[0].text == "BASE ${org}"
    assert fixed_segments[1].text == "STYLE part"


def test_prompt_slot_custom_build_leaves_segments_empty(tmp_path: Path) -> None:
    """custom `build` 経路では `Slot.segments` が空（optimizer の再合成対象から外す）。"""
    sentinel = AgentSpec(name="custom", instructions="x", model=FakeModel())

    def _build(_candidate: str) -> AgentSpec:
        return sentinel

    slot = prompt_slot(_store(tmp_path), agent="bot", base="main", parts=["style"], build=_build)
    assert slot.segments == ()


def test_default_build_substitutes_vars_into_fixed(tmp_path: Path) -> None:
    """base/parts の `${var}` は build 時に `Slot.vars` で再注入される（Codex P2 回帰防止）。

    `_reinject_vars` は候補テキストにのみ vars を注入するため、固定部分（base/parts）にも同じ
    vars 再注入を適用しないと literal な `${var}` が rollout 時の instructions に残る。
    """
    slot = prompt_slot(
        _store(tmp_path),
        _registry(),
        agent="bot",
        base="main",
        parts=["style"],
        vars={"org": "AgentSpec"},
    )
    built = slot.build("CANDIDATE")
    # fixed の `${org}` が `AgentSpec` に置換され、`STYLE part` はそのまま、candidate は後置。
    assert built.instructions == "BASE AgentSpec\n\nSTYLE part\n\nCANDIDATE"


# ----------------------------------------------------------------------
# prompt_slot: fail-closed（registry 未解決）と build= 明示経路
# ----------------------------------------------------------------------


def test_default_build_without_registry_raises(tmp_path: Path) -> None:
    """build 省略かつ registry 未供給は即 fail-closed の ValueError（slot 生成時）。"""
    with pytest.raises(ValueError, match="registry"):
        prompt_slot(_store(tmp_path), None, agent="bot")


def test_default_build_unregistered_spec_raises(tmp_path: Path) -> None:
    """build 省略で対象 spec が registry に未登録なら build 呼び出し時に ValueError。"""
    # store には bot.md があるが registry には bot を登録しない。
    reg = AgentRegistry()
    reg.register(AgentSpec(name="other", instructions="o", model=FakeModel()))
    slot = prompt_slot(_store(tmp_path), reg, agent="bot")
    with pytest.raises(ValueError, match="未登録"):
        slot.build("x")


def test_explicit_build_bypasses_registry(tmp_path: Path) -> None:
    """build= 明示時は registry 不要で、その build がそのまま Slot.build になる。"""
    sentinel = AgentSpec(name="custom", instructions="from-build", model=FakeModel())

    def _build(candidate: str) -> AgentSpec:
        return sentinel

    slot = prompt_slot(_store(tmp_path), agent="bot", build=_build)
    assert slot.build is _build
    assert slot.build("anything") is sentinel
    # seed は依然 store から取得される。
    assert slot.seed == "You are ${role}. Be helpful."


def test_prompt_slot_missing_tune_segment_raises(tmp_path: Path) -> None:
    """tune セグメントが store で解決できなければ KeyError。"""
    with pytest.raises(KeyError):
        prompt_slot(_store(tmp_path), _registry(), agent="nonexistent")


# ----------------------------------------------------------------------
# PromptLayout（標準 agents/<name>.md レイアウト）の seed 解決
# ----------------------------------------------------------------------


def test_prompt_slot_resolves_seed_from_agents_layout(tmp_path: Path) -> None:
    """`PromptLayout(agents="agents")` の標準レイアウト（`agents/<name>.md`）から seed を取得する。

    `store.get(tune)` は root 直下のみを見るため、標準 agents/ 配下のテンプレートを解決できない
    （Codex P2）。`prompt_slot` は `store.compose(agent=tune, vars=None)` 優先で agents/ を読む。
    """
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "billing.md").write_text("Billing agent ${tone}", encoding="utf-8")
    store = PromptStore(tmp_path, PromptLayout(base="base", parts="parts", agents="agents"))
    reg = AgentRegistry()
    reg.register(AgentSpec(name="billing", instructions="orig", model=FakeModel()))

    slot = prompt_slot(store, reg, agent="billing")

    assert slot.seed == "Billing agent ${tone}"


def test_prompt_slot_falls_back_to_flat_when_agents_dir_misses(tmp_path: Path) -> None:
    """`agents/<tune>.md` が無いときは root 直下の flat 配置にフォールバックする（後方互換）。

    既存の flat レイアウト利用者（root 直下に `<tune>.md`）でも `prompt_slot` が壊れない。
    """
    # root 直下に flat 配置（agents/ ディレクトリは未作成）。
    (tmp_path / "billing.md").write_text("Flat billing prompt", encoding="utf-8")
    store = PromptStore(tmp_path, PromptLayout(base="base", parts="parts", agents="agents"))
    reg = AgentRegistry()
    reg.register(AgentSpec(name="billing", instructions="orig", model=FakeModel()))

    slot = prompt_slot(store, reg, agent="billing")

    assert slot.seed == "Flat billing prompt"


def test_prompt_slot_prefers_agents_layout_over_flat(tmp_path: Path) -> None:
    """同名が両方あれば agents/ レイアウトを優先する（layout 尊重・標準を信頼する）。"""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "billing.md").write_text("LAYOUT BODY", encoding="utf-8")
    (tmp_path / "billing.md").write_text("FLAT BODY", encoding="utf-8")
    store = PromptStore(tmp_path, PromptLayout(base="base", parts="parts", agents="agents"))
    reg = AgentRegistry()
    reg.register(AgentSpec(name="billing", instructions="orig", model=FakeModel()))

    slot = prompt_slot(store, reg, agent="billing")

    assert slot.seed == "LAYOUT BODY"


# ----------------------------------------------------------------------
# prompt_slot: fixed var 検査（既定 build / custom build）
# ----------------------------------------------------------------------


def test_prompt_slot_missing_fixed_var_raises_config_missing(tmp_path: Path) -> None:
    """既定 build 経路で base/parts に含まれる `${var}` が vars に未指定なら CONFIG_MISSING で
    fail-closed（literal な `${var}` が agent.instructions に残るのを防ぐ）。"""

    # _store fixture: base/main.md = "BASE ${org}" だが vars に 'org' を渡さない。
    with pytest.raises(OptimizeError) as exc:
        prompt_slot(_store(tmp_path), _registry(), agent="bot", base="main", parts=["style"])
    assert exc.value.kind == FailureKind.CONFIG_MISSING
    assert "org" in str(exc.value)


def test_prompt_slot_custom_build_skips_fixed_var_check(tmp_path: Path) -> None:
    """custom build 経路は fixed の vars 不足を検査しない（custom build がどう組み立てるかを
    ライブラリ側で保証できないため）。"""
    sentinel = AgentSpec(name="custom", instructions="x", model=FakeModel())

    def _build(_candidate: str) -> AgentSpec:
        return sentinel

    # vars に 'org' を渡さないが custom build なので CONFIG_MISSING にならない。
    slot = prompt_slot(_store(tmp_path), agent="bot", base="main", parts=["style"], build=_build)
    assert slot.segments == ()


# ----------------------------------------------------------------------
# prompt_slot: 新 shape（RED: Issue #40 T3・compose 一致・tune セレクタ + layout + fail-closed）
#
# `agent=` / `layout=` は本テスト作成時点で未実装（実装は後段）。新 shape 系はすべて RED
# （TypeError 等で失敗）になる想定。旧経路互換の 2 件は現行実装のまま緑を維持する。
# ----------------------------------------------------------------------


def _store_new_shape(tmp_path: Path) -> PromptStore:
    """新 shape（agent= / layout=）テスト用ストア（agents/ ディレクトリにセグメントを配置）。"""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "triage.md").write_text("Triage seed ${tone}", encoding="utf-8")
    (agents_dir / "billing.md").write_text("Billing seed", encoding="utf-8")
    (agents_dir / "reserved.md").write_text("${oas_boundary_1} reserved seed", encoding="utf-8")
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "main.md").write_text("BASE ${org}", encoding="utf-8")
    (base_dir / "reserved.md").write_text("${oas_boundary_1} reserved base", encoding="utf-8")
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    (parts_dir / "style.md").write_text("STYLE part", encoding="utf-8")
    # part 名 "triage" は agent 名 "triage" と衝突させ、plain 名の名前空間衝突検査に使う。
    (parts_dir / "triage.md").write_text("PART TRIAGE", encoding="utf-8")
    return PromptStore(tmp_path, PromptLayout(base="base", parts="parts", agents="agents"))


# --- 旧 shape 削除の regression guard（ADR 0007） ---


def test_prompt_slot_rejects_legacy_shape(tmp_path: Path) -> None:
    """v0.3.x で旧 shape (`agent=None` + `layout=None` + `tune=<str>`) を削除した regression guard。

    ADR 0007 で NFR-3 撤回・旧経路削除を決定。誤って旧 shape 呼び出しが復活しないよう、
    `agent=` / `layout=` のいずれも指定しない呼び出しは `OptimizeError(CONFIG_MISSING)` で
    fail-closed することを固定する。
    """
    with pytest.raises(OptimizeError) as exc:
        prompt_slot(_store(tmp_path), _registry(), tune="bot")
    assert exc.value.kind == FailureKind.CONFIG_MISSING
    assert "agent=" in str(exc.value) or "layout=" in str(exc.value)


def test_prompt_slot_rejects_missing_agent_and_layout(tmp_path: Path) -> None:
    """`agent=None` + `layout=None` + `tune=None` も同様に fail-closed（regression guard）。"""
    with pytest.raises(OptimizeError) as exc:
        prompt_slot(_store(tmp_path), _registry())
    assert exc.value.kind == FailureKind.CONFIG_MISSING


# --- 新 shape（agent= 指定） ---


def test_prompt_slot_new_shape_agent_only(tmp_path: Path) -> None:
    """`agent="triage"` のみ指定（`tune=None` 既定）は agent セグメントのみが `tune=True`。"""
    slot = prompt_slot(_store_new_shape(tmp_path), _registry(), agent="triage")
    assert slot.name == "triage"
    assert len(slot.segments) == 1
    seg = slot.segments[0]
    assert seg.ref == "agent:triage"
    assert seg.tune is True


def test_prompt_slot_new_shape_base_parts_agent_config(tmp_path: Path) -> None:
    """base/parts/agent を指定すると構成順（base -> parts -> agent）で segments が構築され、
    `tune=None` 既定では agent セグメントのみが `tune=True` になる。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        parts=["style"],
        vars={"org": "AgentSpec"},
    )
    refs = [seg.ref for seg in slot.segments]
    assert refs == ["base:main", "part:style", "agent:triage"]
    tuned = {seg.ref: seg.tune for seg in slot.segments}
    assert tuned == {"base:main": False, "part:style": False, "agent:triage": True}


def test_prompt_slot_new_shape_multi_tune_plain_names(tmp_path: Path) -> None:
    """`tune=["main", "triage"]` は plain 名で base:main / agent:triage を選択する
    （構成順維持）。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        parts=["style"],
        tune=["main", "triage"],
    )
    refs = [seg.ref for seg in slot.segments]
    assert refs == ["base:main", "part:style", "agent:triage"]
    tuned = {seg.ref: seg.tune for seg in slot.segments}
    assert tuned == {"base:main": True, "part:style": False, "agent:triage": True}


def test_prompt_slot_new_shape_multi_tune_qualified_refs(tmp_path: Path) -> None:
    """`tune=["base:main", "agent:triage"]`（qualified 参照）は plain 名指定と同一の選択結果。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        parts=["style"],
        tune=["base:main", "agent:triage"],
    )
    tuned = {seg.ref: seg.tune for seg in slot.segments}
    assert tuned == {"base:main": True, "part:style": False, "agent:triage": True}


# --- layout（compose と同一意味論） ---


def test_prompt_slot_layout_explicit(tmp_path: Path) -> None:
    """`layout=` 指定時は構成順が layout の並びどおりになる
    （base/parts の自然順と異なる並びでも）。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        layout=["part:style", "base:main", "agent:triage"],
        vars={"org": "AgentSpec"},
    )
    refs = [seg.ref for seg in slot.segments]
    assert refs == ["part:style", "base:main", "agent:triage"]


def test_prompt_slot_layout_implicit_agent_resolution(tmp_path: Path) -> None:
    """`agent=None` + layout 内に `agent:X` がちょうど 1 つあれば X が `Slot.name` になる。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        layout=["base:main", "agent:triage"],
        vars={"org": "AgentSpec"},
    )
    assert slot.name == "triage"


def test_prompt_slot_layout_overrides_base_parts(tmp_path: Path) -> None:
    """layout 指定時は `base` / `parts` kwarg の構成指定が無視される。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        parts=["style"],
        layout=["agent:triage"],
    )
    assert [seg.ref for seg in slot.segments] == ["agent:triage"]


# --- fail-closed（OptimizeError CONFIG_MISSING） ---


def test_prompt_slot_fail_agent_none_with_sequence_tune(tmp_path: Path) -> None:
    """`agent=None` + layout 未指定のとき、`tune` が Sequence（spec 解決名不定）でも
    `tune=None`（新経路と判定・旧経路は `tune=str` 必須）でも CONFIG_MISSING。"""
    store = _store_new_shape(tmp_path)
    reg = _registry()
    with pytest.raises(OptimizeError) as exc_seq:
        prompt_slot(store, reg, base="main", tune=["main", "triage"])
    assert exc_seq.value.kind == FailureKind.CONFIG_MISSING

    with pytest.raises(OptimizeError) as exc_none:
        prompt_slot(store, reg, base="main", parts=["style"])
    assert exc_none.value.kind == FailureKind.CONFIG_MISSING


def test_prompt_slot_fail_empty_tune_sequence(tmp_path: Path) -> None:
    """`tune=[]` は CONFIG_MISSING（`tune=None` の「既定 = agent のみ」とは区別される）。"""
    with pytest.raises(OptimizeError) as exc:
        prompt_slot(_store_new_shape(tmp_path), _registry(), agent="triage", tune=[])
    assert exc.value.kind == FailureKind.CONFIG_MISSING


def test_prompt_slot_fail_duplicate_tune(tmp_path: Path) -> None:
    """`tune` の重複要素（plain / qualified 表記違いの同一セグメント参照を含む）は
    CONFIG_MISSING。"""
    store = _store_new_shape(tmp_path)
    reg = _registry()
    with pytest.raises(OptimizeError) as exc1:
        prompt_slot(store, reg, agent="triage", base="main", tune=["main", "main"])
    assert exc1.value.kind == FailureKind.CONFIG_MISSING

    with pytest.raises(OptimizeError) as exc2:
        prompt_slot(store, reg, agent="triage", base="main", tune=["main", "base:main"])
    assert exc2.value.kind == FailureKind.CONFIG_MISSING


def test_prompt_slot_fail_unresolved_tune_name(tmp_path: Path) -> None:
    """`tune` に構成に存在しない名前を含むと CONFIG_MISSING。"""
    with pytest.raises(OptimizeError) as exc:
        prompt_slot(_store_new_shape(tmp_path), _registry(), agent="triage", tune=["nonexistent"])
    assert exc.value.kind == FailureKind.CONFIG_MISSING


def test_prompt_slot_fail_plain_name_collision(tmp_path: Path) -> None:
    """plain 名が複数のセグメント名前空間に一致（part:triage と agent:triage）すると一意に定まらず
    CONFIG_MISSING（qualified 参照を要求する）。"""
    with pytest.raises(OptimizeError) as exc:
        prompt_slot(
            _store_new_shape(tmp_path),
            _registry(),
            agent="triage",
            parts=["triage"],
            tune=["triage"],
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


def test_prompt_slot_fail_layout_empty(tmp_path: Path) -> None:
    """`layout=[]` は CONFIG_MISSING。"""
    with pytest.raises(OptimizeError) as exc:
        prompt_slot(_store_new_shape(tmp_path), _registry(), agent="triage", layout=[])
    assert exc.value.kind == FailureKind.CONFIG_MISSING


def test_prompt_slot_fail_layout_duplicate_ref(tmp_path: Path) -> None:
    """layout の重複参照は CONFIG_MISSING。"""
    with pytest.raises(OptimizeError) as exc:
        prompt_slot(
            _store_new_shape(tmp_path),
            _registry(),
            agent="triage",
            layout=["base:main", "base:main", "agent:triage"],
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


def test_prompt_slot_fail_layout_agent_zero_or_multiple(tmp_path: Path) -> None:
    """`agent=None` かつ layout 内の `agent:` 参照が 0 個または複数個なら CONFIG_MISSING。"""
    store = _store_new_shape(tmp_path)
    reg = _registry()
    with pytest.raises(OptimizeError) as exc_zero:
        prompt_slot(store, reg, layout=["base:main", "part:style"])
    assert exc_zero.value.kind == FailureKind.CONFIG_MISSING

    with pytest.raises(OptimizeError) as exc_multi:
        prompt_slot(store, reg, layout=["agent:triage", "agent:billing"])
    assert exc_multi.value.kind == FailureKind.CONFIG_MISSING


def test_prompt_slot_fail_layout_tune_none_missing_agent_segment(tmp_path: Path) -> None:
    """layout 指定 + `tune=None`（既定）かつ layout 内に agent セグメントが無ければ CONFIG_MISSING
    （既定の「agent セグメントのみ最適化」が成立しないため）。"""
    with pytest.raises(OptimizeError) as exc:
        prompt_slot(
            _store_new_shape(tmp_path),
            _registry(),
            agent="triage",
            layout=["base:main", "part:style"],
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


def test_prompt_slot_fail_reserved_prefix_in_vars_key(tmp_path: Path) -> None:
    """予約接頭辞 `oas_boundary_` を含む vars キーは CONFIG_MISSING（境界マーカー予約）。"""
    with pytest.raises(OptimizeError) as exc:
        prompt_slot(
            _store_new_shape(tmp_path),
            _registry(),
            agent="triage",
            vars={"oas_boundary_1": "x"},
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


def test_prompt_slot_fail_reserved_prefix_in_seed_body(tmp_path: Path) -> None:
    """seed 本文に予約接頭辞 `${oas_boundary_N}` が含まれる場合は CONFIG_MISSING。"""
    with pytest.raises(OptimizeError) as exc:
        prompt_slot(_store_new_shape(tmp_path), _registry(), agent="reserved")
    assert exc.value.kind == FailureKind.CONFIG_MISSING


def test_prompt_slot_fail_reserved_prefix_in_fixed_body(tmp_path: Path) -> None:
    """固定セグメント（base）本文に予約接頭辞 `${oas_boundary_N}` が含まれる場合は
    CONFIG_MISSING。"""
    with pytest.raises(OptimizeError) as exc:
        prompt_slot(
            _store_new_shape(tmp_path),
            _registry(),
            agent="triage",
            base="reserved",
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


def test_prompt_slot_new_shape_fail_fixed_var_missing(tmp_path: Path) -> None:
    """新 shape の既定 build 経路で、固定セグメント（tune=False）に含まれる `${var}` が vars に
    未指定なら CONFIG_MISSING で fail-closed（旧経路 `_legacy_prompt_slot` と対称・literal な
    `${org}` が rollout 時に agent.instructions へ残るのを防ぐ）。"""
    # base:main = "BASE ${org}" が固定側（tune=None 既定で agent:triage のみ tune）に入り、
    # vars に 'org' を渡さないため欠落を検出する。
    with pytest.raises(OptimizeError) as exc:
        prompt_slot(
            _store_new_shape(tmp_path),
            _registry(),
            agent="triage",
            base="main",
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING
    assert "org" in str(exc.value)


def test_prompt_slot_new_shape_tune_var_not_required(tmp_path: Path) -> None:
    """tune セグメント（tune=True）の `${var}` は vars 欠落でも失敗しない（APO 最適化対象なので
    温存が正当・固定 vars 検査は fixed 側だけを対象にする）。"""
    # agent:triage = "Triage seed ${tone}" は tune=None 既定で tune=True（固定側ではない）。
    # base:main の `${org}` は tune=["main", "triage"] で tune 側へ回すため固定側は空になる。
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        tune=["main", "triage"],
    )
    tuned = {seg.ref: seg.tune for seg in slot.segments}
    assert tuned == {"base:main": True, "agent:triage": True}
    # tune 側の `${tone}` / `${org}` は vars 欠落でも保持され seed に残る。
    assert "${tone}" in slot.seed
    assert "${org}" in slot.seed


# ----------------------------------------------------------------------
# prompt_slot: 新 shape の seed マーカー連結 + 既定 build 再インターリーブ
# （RED: Issue #40 T4・_new_shape_slot の seed 生成とマーカー連結・compose_segments 経由の
# 既定 build は本テスト作成時点で未実装。実装は後段。）
# ----------------------------------------------------------------------


def test_prompt_slot_new_shape_seed_single_tune_no_marker(tmp_path: Path) -> None:
    """tune セグメントが 1 個（agent のみ）のとき seed にマーカーは挟まらない。"""
    slot = prompt_slot(_store_new_shape(tmp_path), _registry(), agent="triage")
    assert slot.seed == "Triage seed ${tone}"
    assert "${oas_boundary_" not in slot.seed


def test_prompt_slot_new_shape_seed_two_tune_with_marker(tmp_path: Path) -> None:
    """tune セグメントが 2 個のとき seed は `${oas_boundary_1}` を挟んで構成順に連結される。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        tune=["main", "triage"],
        vars={"org": "AgentSpec"},
    )
    assert "${oas_boundary_1}" in slot.seed
    assert "\n\n${oas_boundary_1}\n\n" in slot.seed
    parts_around_marker = slot.seed.split("${oas_boundary_1}")
    assert "BASE" in parts_around_marker[0]
    assert "Triage seed" in parts_around_marker[1]


def test_prompt_slot_new_shape_seed_three_tune_with_two_markers(tmp_path: Path) -> None:
    """tune セグメントが 3 個のとき seed は `${oas_boundary_1}` と `${oas_boundary_2}` を
    構成順（base -> part -> agent）に挟んで連結される。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        parts=["style"],
        tune=["main", "style", "triage"],
    )
    assert "\n\n${oas_boundary_1}\n\n" in slot.seed
    assert "\n\n${oas_boundary_2}\n\n" in slot.seed
    first, rest = slot.seed.split("${oas_boundary_1}", 1)
    second, third = rest.split("${oas_boundary_2}", 1)
    assert "BASE" in first
    assert "STYLE" in second
    assert "Triage seed" in third


def test_prompt_slot_new_shape_seed_construction_order(tmp_path: Path) -> None:
    """seed の連結順は `tune` 引数の列挙順ではなく構成順（base -> parts -> agent）に従う。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        # 列挙順は agent が先・base が後だが、構成順（base -> agent）で連結される想定。
        tune=["triage", "main"],
    )
    assert "\n\n${oas_boundary_1}\n\n" in slot.seed
    first, second = slot.seed.split("${oas_boundary_1}")
    assert "BASE" in first
    assert "Triage seed" in second


def test_prompt_slot_new_shape_build_single_tune(tmp_path: Path) -> None:
    """n_tune=1 のとき既定 build は候補を構成順どおりに再インターリーブする（tune が固定
    セグメントより前に来る構成でも順序が保たれる・旧実装は常に fixed を先頭に置くため RED）。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        layout=["agent:triage", "base:main"],
        vars={"org": "AgentSpec"},
    )
    built = slot.build("CANDIDATE")
    assert built.instructions == "CANDIDATE\n\nBASE AgentSpec"


def test_prompt_slot_new_shape_build_multi_tune_reinterleaves(tmp_path: Path) -> None:
    """n_tune=2 のとき既定 build は候補を境界マーカーで分割し、構成順に再インターリーブする。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        parts=["style"],
        tune=["main", "triage"],
        vars={"org": "AgentSpec"},
    )
    candidate = "OPTIMIZED_MAIN\n\n${oas_boundary_1}\n\nOPTIMIZED_TRIAGE"
    built = slot.build(candidate)
    instructions = built.instructions
    assert instructions == "OPTIMIZED_MAIN\n\nSTYLE part\n\nOPTIMIZED_TRIAGE"
    assert "${oas_boundary_" not in instructions


def test_prompt_slot_new_shape_build_fixed_vars_substituted(tmp_path: Path) -> None:
    """既定 build は固定セグメントの `${var}` を rollout 時に注入し、tune 側は候補テキストの
    プレースホルダを温存する（`_reinject_vars` が別途注入する契約）。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        vars={"org": "AgentSpec"},
    )
    built = slot.build("CANDIDATE ${tone}")
    assert built.instructions == "BASE AgentSpec\n\nCANDIDATE ${tone}"
    assert "${org}" not in built.instructions


def test_prompt_slot_new_shape_build_invalid_marker_raises(tmp_path: Path) -> None:
    """境界マーカーが欠落・崩れた候補は既定 build で `_CandidateInvalid` を送出する
    （`_apply_candidate` が catch して候補無効化する経路の土台・FU C3 対応で内部 sentinel 化）。"""
    from oai_agentspec.runtime.lightning.types import _CandidateInvalid

    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        tune=["main", "triage"],
        vars={"org": "AgentSpec"},
    )
    with pytest.raises(_CandidateInvalid):
        slot.build("NO MARKER HERE")


def test_prompt_slot_new_shape_custom_build_leaves_segments_empty(tmp_path: Path) -> None:
    """新 shape で `build=` が明示されたら `Slot.segments` は空に保つ（Codex P2 修正）。

    segments が非空だと `optimizer._recompose_new_shape_results` が既定 build による segments
    合成前提で `OptimizeResult.prompt/seed/diff` を full 再合成で上書きしてしまい、custom build
    が実際に組み立てた rollout instructions と乖離する。custom build 経路は `Slot.segments = ()`
    にして _recompose 対象外にし、「run_apo 返却をそのまま尊重する」挙動に統一する。
    """

    def _custom_build(candidate: str) -> AgentSpec:
        return AgentSpec(name="triage", instructions=candidate, model=FakeModel())

    slot_custom = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        vars={"org": "AgentSpec"},
        build=_custom_build,
    )
    assert slot_custom.segments == ()


def test_prompt_slot_new_shape_custom_build_multi_tune_fail_closed(tmp_path: Path) -> None:
    """新 shape で custom `build=` + multi-tune の併用は fail-closed（境界マーカー漏出防止）。

    2 個以上の tune セグメントは境界マーカー `${oas_boundary_N}` 入り seed になり、custom build は
    マーカーを解釈しないため `OptimizeResult.seed / prompt / diff` に literal で漏出する。
    「予約接頭辞は成果物に一切現れない」契約を守るため slot 構築時に `OptimizeError` で拒否する。
    """

    def _custom_build(candidate: str) -> AgentSpec:
        return AgentSpec(name="triage", instructions=candidate, model=FakeModel())

    with pytest.raises(OptimizeError) as exc:
        prompt_slot(
            _store_new_shape(tmp_path),
            _registry(),
            agent="triage",
            base="main",
            tune=["main", "triage"],
            vars={"org": "AgentSpec"},
            build=_custom_build,
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING
    assert "custom build" in str(exc.value)
    assert "multi-tune" in str(exc.value)


# ----------------------------------------------------------------------
# prompt_slot: vars=callable（動的 instructions 生成・Issue #40 T5）
# （RED: `vars_fn` 保持・既定 build の動的 instructions 化・_ensure_fixed_vars_present 免除は
# 本テスト作成時点で未実装。実装は後段。）
# ----------------------------------------------------------------------


def test_prompt_slot_vars_callable_stores_in_vars_fn(tmp_path: Path) -> None:
    """`vars=callable` を渡すと `Slot.vars_fn` に保持され `Slot.vars` は空 dict になる。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        vars=lambda ctx: {"tone": "polite"},
    )
    assert slot.vars_fn is not None
    assert slot.vars == {}


def test_prompt_slot_vars_dict_uses_vars_only(tmp_path: Path) -> None:
    """`vars=dict` は従来どおり `Slot.vars` に保持され `Slot.vars_fn` は None のまま。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        vars={"org": "AgentSpec"},
    )
    assert slot.vars_fn is None
    assert slot.vars == {"org": "AgentSpec"}


def test_prompt_slot_vars_none_uses_neither(tmp_path: Path) -> None:
    """`vars=None`（既定）は `Slot.vars` が空 dict・`Slot.vars_fn` は None。"""
    slot = prompt_slot(_store_new_shape(tmp_path), _registry(), agent="triage")
    assert slot.vars == {}
    assert slot.vars_fn is None


def test_prompt_slot_vars_callable_build_returns_dynamic_instructions(tmp_path: Path) -> None:
    """`vars=callable` の既定 build は `(context, agent) -> str` の callable を instructions に
    据え、rollout 時に `vars_fn(context)` を評価して fixed 側へ注入する。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        vars=lambda ctx: {"org": ctx.company},
    )
    agent_spec = slot.build("CANDIDATE_TEXT")
    assert callable(agent_spec.instructions)

    class FakeCtx:
        company = "AgentSpec"

    result = agent_spec.instructions(FakeCtx(), agent_spec)
    assert "AgentSpec" in result
    assert "CANDIDATE_TEXT" in result


def test_prompt_slot_vars_callable_multi_tune_still_reinterleaves(tmp_path: Path) -> None:
    """`tune=["main", "triage"]` + `vars=callable` でも構成順の再インターリーブと `vars_fn`
    評価が両立する。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        tune=["main", "triage"],
        vars=lambda ctx: {"org": ctx.company},
    )
    candidate = "OPTIMIZED_MAIN\n\n${oas_boundary_1}\n\nOPTIMIZED_TRIAGE"
    agent_spec = slot.build(candidate)
    assert callable(agent_spec.instructions)

    class FakeCtx:
        company = "AgentSpec"

    result = agent_spec.instructions(FakeCtx(), agent_spec)
    assert result == "OPTIMIZED_MAIN\n\nOPTIMIZED_TRIAGE"


def test_prompt_slot_vars_callable_substitutes_tune_side_var(tmp_path: Path) -> None:
    """`vars=callable` は rollout 時に tune 側の `${var}` も substitute する（ADR 0005 契約）。

    ADR 0005 は「callable の中身は substitute_braced(合成済みテキスト, vars_fn(context))」
    と規定。旧実装は `compose_from_marked` だけを呼び tune セグメントを raw のまま返していたため
    tune 側 `${var}` が literal で SDK に渡っていた（context 由来値注入の主用途が機能せず）。
    修正: 動的 instructions closure は full 合成後に `substitute_braced(full, dynamic_vars)` を
    追加適用する。
    """
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",  # agent セグメント 1 本を tune 対象（既定）
        vars=lambda ctx: {"tone": ctx.tone},
    )
    # candidate は tune 側の `${tone}` を保持したまま（`_reinject_vars` は Slot.vars={} で no-op）。
    agent_spec = slot.build("${tone} で応答してください")
    assert callable(agent_spec.instructions)

    class FakeCtx:
        tone = "polite"

    result = agent_spec.instructions(FakeCtx(), agent_spec)
    # 旧実装: "${tone} で応答してください" (literal 残り・BUG)
    # 修正後: "polite で応答してください" (dynamic_vars で substitute される)
    assert result == "polite で応答してください"
    assert "${tone}" not in result


def test_prompt_slot_vars_callable_no_double_substitute_on_fixed_side(tmp_path: Path) -> None:
    """`vars=callable` の rollout closure は fixed 側の値内 `${...}` を二重 substitute しない
    （Codex P2 regression guard・compose(vars=callable) と同じ 1-pass セマンティクス）。

    シナリオ: 固定 base の `${org}` に対して callable が値 `"${tone}"` を返す。二重 pass だと
    fixed の `${org}` → `"${tone}"` → `"polite"` と再解釈されるが、正しい 1-pass 動作では
    fixed に `${tone}` が literal で残る（tune 側 callable と compose の意味論を保つ）。
    """
    from pathlib import Path as _P

    # base に `${org}` 固定・tune=agent 単独の store を組む。
    root = tmp_path
    _P(root / "agents").mkdir()
    _P(root / "agents" / "triage.md").write_text("TUNE ${tone}", encoding="utf-8")
    _P(root / "base").mkdir()
    _P(root / "base" / "main.md").write_text("org=${org}", encoding="utf-8")
    from oai_agentspec.prompts import PromptLayout, PromptStore

    store = PromptStore(root, PromptLayout(base="base", parts="parts", agents="agents"))

    slot = prompt_slot(
        store,
        _registry(),
        agent="triage",
        base="main",
        vars=lambda ctx: {"org": "${tone}", "tone": "polite"},
    )
    agent_spec = slot.build("TUNE ${tone}")

    class FakeCtx:
        pass

    result = agent_spec.instructions(FakeCtx(), agent_spec)
    # 固定側: ${org} → "${tone}" (1-pass で literal 保持・二重 pass だと "polite" になる)
    # tune 側: ${tone} → "polite" (rollout 時 substitute される)
    assert result == "org=${tone}\n\nTUNE polite"


def test_prompt_slot_vars_callable_optimize_result_keeps_placeholder(tmp_path: Path) -> None:
    """`vars=callable` の slot は `OptimizeResult` 側では tune 側 `${var}` を literal で保持する
    （具体値をベイクしない・compose(vars=callable) と同一契約）。

    `_recompose_new_shape_results` は Slot.vars = {} を使うため tune / 固定いずれの `${var}` も
    substitute されない。context 由来値の実行時注入は rollout closure だけが担い、成果物では
    placeholder を温存する（rollout 実体と OptimizeResult の意味的分離）。
    """
    from oai_agentspec.runtime.lightning._placeholders import compose_from_marked

    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        vars=lambda ctx: {"tone": ctx.tone},
    )
    # OptimizeResult 合成側は Slot.vars（空 dict）で compose_from_marked を呼ぶ。
    optimize_result_prompt = compose_from_marked(slot.segments, "${tone} で応答", dict(slot.vars))
    assert optimize_result_prompt == "${tone} で応答"  # literal 保持（ベイクなし）


def test_prompt_slot_vars_callable_returns_non_dict_raises_candidate_invalid(
    tmp_path: Path,
) -> None:
    """`vars=callable` の戻り値が dict でないと、動的 instructions 経路で `_CandidateInvalid` に
    倒す（FU C1 fix: TypeError から内部 sentinel に変更し rollout closure で catch・reward 0.0）。

    旧実装は TypeError を投げていたが、SDK Runner.run が rollout 中に invoke するため
    `_apply_candidate` の catch を通らず optimize 全体が abort する経路になっていた。
    現在は `_CandidateInvalid` を投げ、`_make_rollout` の rollout closure が catch する。"""
    from oai_agentspec.runtime.lightning.types import _CandidateInvalid

    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        vars=lambda ctx: ["not", "a", "dict"],
    )
    agent_spec = slot.build("CANDIDATE_TEXT")

    class FakeCtx:
        pass

    with pytest.raises(_CandidateInvalid) as exc:
        agent_spec.instructions(FakeCtx(), agent_spec)
    assert "dict" in str(exc.value)
    assert "vars callable" in str(exc.value)


def test_prompt_slot_vars_callable_with_build_fail(tmp_path: Path) -> None:
    """`vars=callable` かつ `build=` 明示は既定 build のみが `vars_fn` を評価する契約のため
    CONFIG_MISSING で fail-closed。"""
    with pytest.raises(OptimizeError) as exc:
        prompt_slot(
            _store_new_shape(tmp_path),
            _registry(),
            agent="triage",
            vars=lambda ctx: {},
            build=lambda cand: None,
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


def test_prompt_slot_vars_callable_skips_fixed_vars_check(tmp_path: Path) -> None:
    """`vars=callable` のときは `_ensure_fixed_vars_present` の構築時検査が免除される（同じ
    `base` で `vars=dict` のときは CONFIG_MISSING になるケースでも構築が成功する）。"""
    slot = prompt_slot(
        _store_new_shape(tmp_path),
        _registry(),
        agent="triage",
        base="main",
        vars=lambda ctx: {"org": ctx.company},
    )
    assert slot.vars_fn is not None


def test_prompt_slot_vars_dict_still_calls_fixed_vars_check(tmp_path: Path) -> None:
    """既存挙動: `vars=dict` で `${var}` 欠落は引き続き CONFIG_MISSING（後方互換）。"""
    with pytest.raises(OptimizeError) as exc:
        prompt_slot(
            _store_new_shape(tmp_path),
            _registry(),
            agent="triage",
            base="main",
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


# ----------------------------------------------------------------------
# prompt_slot_factory: 共通既定値を束ねた per-agent Slot 生成（RED: Issue #41 T3・本テスト
# 作成時点で未実装。実装は後段。）
# ----------------------------------------------------------------------


def test_slot_factory_applies_defaults(tmp_path: Path) -> None:
    """既定値のみの生成が `prompt_slot(agent=...)` 直呼びと同一構造になる。"""
    factory = prompt_slot_factory(
        _store(tmp_path),
        _registry(),
        base="main",
        parts=["style"],
        vars={"org": "AgentSpec"},
    )
    slot = factory("bot")
    assert slot.name == "bot"
    assert [seg.ref for seg in slot.segments] == ["base:main", "part:style", "agent:bot"]


def test_slot_factory_override_replaces_parts(tmp_path: Path) -> None:
    """`parts` の上書きは追記でなく置換になる。"""
    factory = prompt_slot_factory(
        _store(tmp_path),
        _registry(),
        base="main",
        parts=["style"],
        vars={"org": "AgentSpec"},
    )
    slot = factory("bot", parts=[])
    assert [seg.ref for seg in slot.segments] == ["base:main", "agent:bot"]


def test_slot_factory_merges_vars(tmp_path: Path) -> None:
    """`vars` は defaults と override が双方 dict のときのみマージされ、per-agent が優先する。"""
    factory = prompt_slot_factory(
        _store(tmp_path),
        _registry(),
        vars={"org": "AgentSpec", "tone": "casual"},
    )
    slot = factory("bot", vars={"tone": "formal", "extra": "z"})
    assert slot.vars == {"org": "AgentSpec", "tone": "formal", "extra": "z"}


def test_slot_factory_does_not_mutate_defaults(tmp_path: Path) -> None:
    """同一ファクトリで 2 回 `make()` しても、defaults の `vars` に per-agent キーが混入しない。"""
    factory = prompt_slot_factory(_store(tmp_path), _registry(), vars={"org": "shared"})
    factory("bot", vars={"role": "assistant"})
    slot2 = factory("triage", vars={"tone": "formal"})
    assert "role" not in slot2.vars


def test_slot_factory_vars_none_clears(tmp_path: Path) -> None:
    """`vars=None` 明示で `Slot.vars` が空 dict になる（打ち消しフィルタを入れないことの pin）。"""
    factory = prompt_slot_factory(_store(tmp_path), _registry(), vars={"org": "AgentSpec"})
    slot = factory("billing", vars=None)
    assert slot.vars == {}


def test_slot_factory_base_none_clears(tmp_path: Path) -> None:
    """`base=None` 明示で `base:` セグメントが消える（None 除去フィルタを入れないことの pin）。"""
    factory = prompt_slot_factory(_store(tmp_path), _registry(), base="main")
    slot = factory("billing", base=None)
    assert "base:main" not in [seg.ref for seg in slot.segments]


def test_slot_factory_passes_layout_through(tmp_path: Path) -> None:
    """`layout` の並びがそのまま `Slot.segments` の ref 順になり、`Slot.name` は常に `agent=`
    から決まる（layout の暗黙解決経路には到達しない）。"""
    factory = prompt_slot_factory(_store(tmp_path), _registry())
    slot = factory(
        "billing",
        layout=["part:style", "base:main", "agent:billing"],
        vars={"org": "AgentSpec"},
    )
    assert [seg.ref for seg in slot.segments] == ["part:style", "base:main", "agent:billing"]
    assert slot.name == "billing"


def test_slot_factory_passes_build_through(tmp_path: Path) -> None:
    """`build=` 素通し時に `Slot.segments` が空になる（custom build 経路）。"""
    sentinel = AgentSpec(name="custom", instructions="x", model=FakeModel())

    def _build(_candidate: str) -> AgentSpec:
        return sentinel

    factory = prompt_slot_factory(_store(tmp_path))
    slot = factory("bot", build=_build)
    assert slot.segments == ()


def test_slot_factory_vars_callable_passthrough(tmp_path: Path) -> None:
    """defaults dict + override callable は置換になり、`Slot.vars_fn` が設定される。"""
    factory = prompt_slot_factory(_store(tmp_path), _registry(), vars={"org": "x"})
    slot = factory("billing", vars=lambda ctx: {"org": "dyn"})
    assert slot.vars_fn is not None
    assert slot.vars == {}


@pytest.mark.parametrize(
    "key,defaults",
    [
        ("parts", {"parts": ["style"]}),
        ("layout", {"layout": ["agent:bot"]}),
    ],
)
def test_slot_factory_defaults_containers_are_not_shared(
    tmp_path: Path, key: str, defaults: dict[str, list[str]]
) -> None:
    """defaults の `parts` / `layout` の list を factory 生成後に変更しても、既に生成済みの
    `Slot.segments` には反映されない（Issue #46 #1・defaults 参照共有 pin）。

    `prompt_slot` は `layout=` 指定時に `agent`/`base`/`parts` を無視するため、
    parts と layout は独立した defaults で pin する必要がある。真に警戒すべきは
    「factory を保持したまま defaults を書き換え → 2 回目の factory() で MUTATED が
    漏れる」経路のため、生成済み Slot と mutation 後の再生成 Slot の両方を検査する。"""
    factory = prompt_slot_factory(_store(tmp_path), _registry(), **defaults)
    slot_before = factory("bot")
    defaults[key].append("MUTATED")
    slot_after = factory("bot")
    assert "MUTATED" not in [seg.ref for seg in slot_before.segments]
    assert "MUTATED" not in [seg.ref for seg in slot_after.segments]


@pytest.mark.parametrize(
    "override",
    [{"k": "v"}, lambda ctx: {"k": "v"}],
    ids=["dict-override", "callable-override"],
)
def test_slot_factory_vars_callable_defaults_replaced_by_override(
    tmp_path: Path, override: object
) -> None:
    """defaults=callable の `vars` に override（dict / callable のいずれか）を渡すと、
    callable 側のマージ意味論を持たず「置換」される（Issue #46 #2・vars callable マージ経路 pin）。

    `vars_fn is not None` だけでは defaults 側 callable が残っても満たされるため、実際に
    `vars_fn(ctx)` を評価して override 側の結果のみが返り defaults 側キー（`"base"`）が
    含まれないことを検証する。
    """
    factory = prompt_slot_factory(_store(tmp_path), _registry(), vars=lambda ctx: {"base": "b"})
    slot = factory("bot", vars=override)
    if callable(override):
        assert slot.vars_fn is not None
        assert slot.vars == {}
        resolved = slot.vars_fn(None)
        assert resolved == {"k": "v"}
        assert "base" not in resolved
    else:
        assert slot.vars == override
        assert slot.vars_fn is None


def test_slot_factory_passes_tune_kwarg(tmp_path: Path) -> None:
    """`tune=` が factory 経由でも `prompt_slot` にそのまま伝わり、対象セグメントのみ
    `tune=True` になる（Issue #46 #3・tune 素通し pin）。"""
    factory = prompt_slot_factory(_store(tmp_path), _registry(), base="main", parts=["style"])
    slot = factory("bot", tune=["main"], vars={"role": "assistant"})
    tuned = {seg.ref: seg.tune for seg in slot.segments}
    assert tuned == {"base:main": True, "part:style": False, "agent:bot": False}


@pytest.mark.parametrize(
    "bad_defaults",
    [
        {"agent": "conflict"},  # agent の二重指定
        {"part": ["style"]},  # typo（parts の間違い）
    ],
    ids=["agent-in-defaults", "typo-part"],
)
def test_slot_factory_invalid_kwarg_raises_type_error(tmp_path: Path, bad_defaults: dict) -> None:
    """defaults に `agent` / 未知キーを含むと `make()` 呼び出し時（factory 生成時ではなく）に
    `TypeError`（許可キーリスト・`agent` 衝突検査を追加していないことの pin）。"""
    factory = prompt_slot_factory(_store(tmp_path), _registry(), **bad_defaults)
    with pytest.raises(TypeError):
        factory("bot")
