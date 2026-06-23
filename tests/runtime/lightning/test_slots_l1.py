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
from oai_agentspec.runtime.lightning import Slot, prompt_slot, prompt_slots

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
    from oai_agentspec.runtime.lightning import FailureKind, OptimizeError

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
