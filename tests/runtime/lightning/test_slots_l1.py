"""L1: スロットヘルパ `prompt_slot` / `prompt_slots`（PromptStore / AgentRegistry 読み取り）。

`prompt_slot` の seed 取得（`${var}` 保持）・既定 build（registry 登録 spec 複製で instructions
差し替え・tools 等保持・元 spec 不変）・固定部分（base / parts）合成・vars 保持（seed 非展開）・
registry 未解決の fail-closed ValueError・`build=` 明示経路（registry 不要）・`prompt_slots` の
一括生成を網羅する。PromptStore は実ファイル（tmp_path）から読み取る（実 LLM 非依存）。
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
    prompt_slots,
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
    slot = prompt_slot(_store(tmp_path), _registry(), tune="bot")
    assert isinstance(slot, Slot)
    assert slot.name == "bot"
    assert slot.seed == "You are ${role}. Be helpful."


def test_prompt_slot_keeps_vars_without_expanding_seed(tmp_path: Path) -> None:
    """vars は Slot.vars に保持され seed には展開しない（最適化対象外・rollout 再注入）。"""
    slot = prompt_slot(_store(tmp_path), _registry(), tune="bot", vars={"role": "agent"})
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

    slot = prompt_slot(_store(tmp_path), reg, tune="bot")
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
        tune="bot",
        base="main",
        parts=["style"],
        vars={"org": "AgentSpec"},
    )
    built = slot.build("CANDIDATE")
    # fixed の `${org}` は build 時に再注入され、candidate を後置。
    assert built.instructions == "BASE AgentSpec\n\nSTYLE part\n\nCANDIDATE"


def test_default_build_without_fixed_uses_candidate_only(tmp_path: Path) -> None:
    """固定部分なしのとき instructions は候補テキストのみ（前置の空連結を入れない）。"""
    slot = prompt_slot(_store(tmp_path), _registry(), tune="bot")
    built = slot.build("ONLY CANDIDATE")
    assert built.instructions == "ONLY CANDIDATE"


def test_prompt_slot_populates_fixed_when_default_build(tmp_path: Path) -> None:
    """既定 build 経路では `Slot.fixed` に base+parts の合成済み固定部分（`${var}` 保持）を持つ
    （`OptimizeResult.seed` / `prompt` を rollout 時の合成済み full テキストで返すため）。"""
    slot = prompt_slot(
        _store(tmp_path),
        _registry(),
        tune="bot",
        base="main",
        parts=["style"],
        vars={"org": "AgentSpec"},
    )
    # _store fixture: base/main.md = "BASE ${org}", parts/style.md = "STYLE part"。
    # Slot.fixed 自体は `${var}` を保持（rollout 時 / run_apo の compose で再注入）。
    assert slot.fixed == "BASE ${org}\n\nSTYLE part"


def test_prompt_slot_fixed_empty_when_custom_build(tmp_path: Path) -> None:
    """custom `build` 経路では `Slot.fixed` は空文字（custom build の組み立てが不明のため）。"""
    sentinel = AgentSpec(name="custom", instructions="x", model=FakeModel())

    def _build(_candidate: str) -> AgentSpec:
        return sentinel

    slot = prompt_slot(_store(tmp_path), tune="bot", base="main", parts=["style"], build=_build)
    assert slot.fixed == ""


def test_default_build_substitutes_vars_into_fixed(tmp_path: Path) -> None:
    """base/parts の `${var}` は build 時に `Slot.vars` で再注入される（Codex P2 回帰防止）。

    `_reinject_vars` は候補テキストにのみ vars を注入するため、固定部分（base/parts）にも同じ
    vars 再注入を適用しないと literal な `${var}` が rollout 時の instructions に残る。
    """
    slot = prompt_slot(
        _store(tmp_path),
        _registry(),
        tune="bot",
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
        prompt_slot(_store(tmp_path), None, tune="bot")


def test_default_build_unregistered_spec_raises(tmp_path: Path) -> None:
    """build 省略で対象 spec が registry に未登録なら build 呼び出し時に ValueError。"""
    # store には bot.md があるが registry には bot を登録しない。
    reg = AgentRegistry()
    reg.register(AgentSpec(name="other", instructions="o", model=FakeModel()))
    slot = prompt_slot(_store(tmp_path), reg, tune="bot")
    with pytest.raises(ValueError, match="未登録"):
        slot.build("x")


def test_explicit_build_bypasses_registry(tmp_path: Path) -> None:
    """build= 明示時は registry 不要で、その build がそのまま Slot.build になる。"""
    sentinel = AgentSpec(name="custom", instructions="from-build", model=FakeModel())

    def _build(candidate: str) -> AgentSpec:
        return sentinel

    slot = prompt_slot(_store(tmp_path), tune="bot", build=_build)
    assert slot.build is _build
    assert slot.build("anything") is sentinel
    # seed は依然 store から取得される。
    assert slot.seed == "You are ${role}. Be helpful."


def test_prompt_slot_missing_tune_segment_raises(tmp_path: Path) -> None:
    """tune セグメントが store で解決できなければ KeyError。"""
    with pytest.raises(KeyError):
        prompt_slot(_store(tmp_path), _registry(), tune="nonexistent")


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

    slot = prompt_slot(store, reg, tune="billing")

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

    slot = prompt_slot(store, reg, tune="billing")

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

    slot = prompt_slot(store, reg, tune="billing")

    assert slot.seed == "LAYOUT BODY"


# ----------------------------------------------------------------------
# prompt_slots: 一括生成
# ----------------------------------------------------------------------


def test_prompt_slots_generates_mapping(tmp_path: Path) -> None:
    """prompt_slots は列挙エージェント分の Slot を `{名前: Slot}` mapping で返す。"""
    slots = prompt_slots(_store(tmp_path), _registry(), ["triage", "billing"])
    assert set(slots) == {"triage", "billing"}
    assert all(isinstance(s, Slot) for s in slots.values())
    assert slots["triage"].seed == "Triage prompt ${tone}"
    assert slots["billing"].seed == "Billing prompt"


def test_prompt_slots_applies_common_vars_and_fixed(tmp_path: Path) -> None:
    """prompt_slots は base / parts / vars を全 slot 共通で適用する。"""
    slots = prompt_slots(
        _store(tmp_path),
        _registry(),
        ["triage"],
        base="main",
        parts=["style"],
        vars={"tone": "x", "org": "AgentSpec"},
    )
    slot = slots["triage"]
    assert slot.vars == {"tone": "x", "org": "AgentSpec"}
    built = slot.build("CAND")
    # base の `${org}` は build 時に再注入される。
    assert built.instructions == "BASE AgentSpec\n\nSTYLE part\n\nCAND"


def test_prompt_slot_missing_fixed_var_raises_config_missing(tmp_path: Path) -> None:
    """既定 build 経路で base/parts に含まれる `${var}` が vars に未指定なら CONFIG_MISSING で
    fail-closed（literal な `${var}` が agent.instructions に残るのを防ぐ）。"""

    # _store fixture: base/main.md = "BASE ${org}" だが vars に 'org' を渡さない。
    with pytest.raises(OptimizeError) as exc:
        prompt_slot(_store(tmp_path), _registry(), tune="bot", base="main", parts=["style"])
    assert exc.value.kind == FailureKind.CONFIG_MISSING
    assert "org" in str(exc.value)


def test_prompt_slot_custom_build_skips_fixed_var_check(tmp_path: Path) -> None:
    """custom build 経路は `Slot.fixed` を空にし、fixed の vars 不足を検査しない（custom build が
    どう組み立てるかライブラリ側で保証できないため）。"""
    sentinel = AgentSpec(name="custom", instructions="x", model=FakeModel())

    def _build(_candidate: str) -> AgentSpec:
        return sentinel

    # vars に 'org' を渡さないが custom build なので CONFIG_MISSING にならない。
    slot = prompt_slot(_store(tmp_path), tune="bot", base="main", parts=["style"], build=_build)
    assert slot.fixed == ""


def test_prompt_slots_unregistered_spec_raises_on_build(tmp_path: Path) -> None:
    """prompt_slots の既定 build も未登録 spec は build 時に fail-closed の ValueError。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="triage", instructions="o", model=FakeModel()))
    # billing は store には在るが registry には未登録。
    slots = prompt_slots(_store(tmp_path), reg, ["triage", "billing"])
    with pytest.raises(ValueError, match="未登録"):
        slots["billing"].build("x")


# ----------------------------------------------------------------------
# prompt_slots: 新 shape 追随（RED: Issue #40 T9・agent= 経由の一括生成 + tune mapping +
# vars=callable・layout 非対応・本テスト作成時点で未実装。実装は後段。）
# ----------------------------------------------------------------------


def test_prompt_slots_new_shape_returns_dict_of_slots(tmp_path: Path) -> None:
    """`prompt_slots` は各 agent 名を新 shape の `agent=` として `prompt_slot` を呼び、
    `{名前: Slot}` の mapping を返す（各 Slot は `segments` 非空・`vars_fn` は None）。"""
    slots = prompt_slots(
        _store_new_shape(tmp_path),
        _registry(),
        ["triage", "billing"],
        base="main",
        vars={"org": "AgentSpec"},
    )
    assert set(slots) == {"triage", "billing"}
    for name, slot in slots.items():
        assert isinstance(slot, Slot)
        assert slot.name == name
        assert slot.segments != ()
        assert slot.vars_fn is None


def test_prompt_slots_new_shape_each_slot_has_agent_segment_tune_by_default(
    tmp_path: Path,
) -> None:
    """`tune` 省略時は各 slot で agent セグメントのみ `tune=True`（他は `tune=False`）。"""
    slots = prompt_slots(
        _store_new_shape(tmp_path),
        _registry(),
        ["triage", "billing"],
        base="main",
        vars={"org": "AgentSpec"},
    )
    for name, slot in slots.items():
        tuned = {seg.ref: seg.tune for seg in slot.segments}
        assert tuned[f"agent:{name}"] is True
        assert tuned["base:main"] is False


def test_prompt_slots_with_tune_mapping(tmp_path: Path) -> None:
    """`tune=dict` を渡すと agent ごとに個別の tune セレクタが適用される。"""
    slots = prompt_slots(
        _store_new_shape(tmp_path),
        _registry(),
        ["triage", "billing"],
        base="main",
        vars={"org": "AgentSpec"},
        tune={"triage": ["main", "triage"], "billing": "billing"},
    )
    triage_tuned = {seg.ref: seg.tune for seg in slots["triage"].segments}
    assert triage_tuned == {"base:main": True, "agent:triage": True}
    billing_tuned = {seg.ref: seg.tune for seg in slots["billing"].segments}
    assert billing_tuned == {"base:main": False, "agent:billing": True}


def test_prompt_slots_with_tune_partial_mapping(tmp_path: Path) -> None:
    """`tune=dict` に一部の agent しか指定しない場合、未指定の agent は既定
    （agent セグメントのみ tune）にフォールバックする。"""
    slots = prompt_slots(
        _store_new_shape(tmp_path),
        _registry(),
        ["triage", "billing"],
        base="main",
        vars={"org": "AgentSpec"},
        tune={"triage": ["main", "triage"]},
    )
    billing_tuned = {seg.ref: seg.tune for seg in slots["billing"].segments}
    assert billing_tuned == {"base:main": False, "agent:billing": True}


def test_prompt_slots_fail_tune_key_not_in_agents(tmp_path: Path) -> None:
    """`tune` mapping のキーが `agents` に含まれない場合は fail-closed の CONFIG_MISSING。"""
    with pytest.raises(OptimizeError) as exc:
        prompt_slots(
            _store_new_shape(tmp_path),
            _registry(),
            ["triage"],
            tune={"unknown": "x"},
        )
    assert exc.value.kind == FailureKind.CONFIG_MISSING


def test_prompt_slots_with_vars_callable(tmp_path: Path) -> None:
    """`vars=callable` を渡すと全 slot で `vars_fn` が設定され `vars` は空 dict に統一される。"""
    slots = prompt_slots(
        _store_new_shape(tmp_path),
        _registry(),
        ["triage", "billing"],
        base="main",
        vars=lambda ctx: {"org": ctx.company},
    )
    for slot in slots.values():
        assert slot.vars_fn is not None
        assert slot.vars == {}


def test_prompt_slots_legacy_call_unchanged(tmp_path: Path) -> None:
    """既存呼び出し `prompt_slots(store, registry, ["triage"])` が現状動作を維持する。"""
    slots = prompt_slots(_store(tmp_path), _registry(), ["triage"])
    assert set(slots) == {"triage"}
    slot = slots["triage"]
    assert slot.name == "triage"
    assert slot.seed == "Triage prompt ${tone}"


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


# --- 旧経路の完全互換（agent=None・tune=単一 str） ---


def test_prompt_slot_legacy_shape_still_works(tmp_path: Path) -> None:
    """`agent=None` + `tune` が単一 str の旧経路は既存挙動を完全互換で維持する。"""
    slot = prompt_slot(_store(tmp_path), _registry(), tune="triage")
    assert slot.name == "triage"
    assert slot.seed == "Triage prompt ${tone}"
    assert slot.segments == ()


def test_prompt_slot_legacy_with_base_and_parts(tmp_path: Path) -> None:
    """旧経路で base/parts を渡した固定部分合成が新 shape 追加後も変わらない。"""
    slot = prompt_slot(
        _store(tmp_path),
        _registry(),
        tune="bot",
        base="main",
        parts=["style"],
        vars={"org": "AgentSpec"},
    )
    assert slot.fixed == "BASE ${org}\n\nSTYLE part"
    assert slot.segments == ()


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
    """FU（Codex P2 修正）: 新 shape でも `build=` が明示されたら `Slot.segments` は空に保つ。

    segments が非空だと `optimizer._recompose_new_shape_results` が「既定 build による segments
    合成が使われた」前提で `OptimizeResult.prompt/seed/diff` を full 再合成で上書きしてしまい、
    custom build が実際に組み立てた rollout instructions と乖離する（"OptimizeResult.prompt ==
    rollout instructions" 契約の drift）。custom build 経路は `Slot.segments = ()` にして
    _recompose 対象外にし、旧 shape / 生 seed 経路と同じ「run_apo 返却をそのまま尊重する」
    挙動に統一する。既定 build（build=None）のときは従来どおり segments を保持する。
    """

    def _custom_build(candidate: str) -> AgentSpec:
        return AgentSpec(name="triage", instructions=candidate, model=FakeModel())

    dir_custom = tmp_path / "custom"
    dir_custom.mkdir()
    dir_default = tmp_path / "default"
    dir_default.mkdir()

    slot_custom = prompt_slot(
        _store_new_shape(dir_custom),
        _registry(),
        agent="triage",
        base="main",
        tune=["main", "triage"],
        vars={"org": "AgentSpec"},
        build=_custom_build,
    )
    assert slot_custom.segments == ()

    slot_default = prompt_slot(
        _store_new_shape(dir_default),
        _registry(),
        agent="triage",
        base="main",
        tune=["main", "triage"],
        vars={"org": "AgentSpec"},
    )
    assert len(slot_default.segments) > 0


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
