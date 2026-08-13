"""L1: 決定的なスロット確定（タスク 1-6・FR-5）と `actions._resolve_path` の純検証。

対象は次の 4 つ。

1. 値の解決順（候補の `parameters` -> `from_context` -> `by_llm`（この段では未実施）->
   `default` -> 利用者入力）と、各状態への遷移（FR-5 L179-L183）
2. `origin` / `detail` の記録（`RUN_CONTEXT` なら解決に成功したパス・それ以外は `None`。
   FR-5 L184）
3. 決定性（LLM 実行アダプタ・ネットワーク・環境変数を参照せず、同一の候補列と同一の
   `run_context` に対し常に同一結果。FR-5 L177）
4. 既定マージ解決 3 関数（`resolve_prompt` / `resolve_prompt_vars` /
   `resolve_on_invalid_slot`）を呼んで `ActionPlan.resolved_*` へ格納すること（設計 §3.4a）

あわせてパス解決ヘルパ `_resolve_path` が **`actions.py` に置かれること**を pin する
（設計 §3.4b。利用者は `_validate.py` / `slots.py` / `_predict.py` の 3 つあり、別々に
書かれると一元化の設計意図が崩れる）。

決定的段の関数名は設計に明示が無いため、案 1 の `plan_slots` を非公開化した
`slots._plan_slots(candidates, catalog, context)` として固定する（設計 §3.13 の
「実装は各モジュールに残し公開シンボルから外す」に従う）。外部依存 (agents / openai) なし。
"""

from __future__ import annotations

import socket
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from oai_agentspec.runtime.intent import actions as actions_module
from oai_agentspec.runtime.intent import slots as slots_module
from oai_agentspec.runtime.intent.actions import (
    ActionCatalog,
    ActionSpec,
    param,
    resolve_on_invalid_slot,
    resolve_prompt,
    resolve_prompt_vars,
)
from oai_agentspec.runtime.intent.slots import ActionPlan, Origin, SlotState
from oai_agentspec.runtime.intent.types import (
    ConfidenceLevel,
    ExecutableIntent,
    IntentContext,
)

pytestmark = pytest.mark.unit

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
_SLOTS_PATH = _SRC_DIR / "oai_agentspec" / "runtime" / "intent" / "slots.py"


class _Env:
    """属性アクセスで辿る run context の内側。"""

    def __init__(self, host: str | None) -> None:
        self.host = host


class _Ctx:
    """属性アクセスで辿る run context。"""

    def __init__(self, host: str | None = "api.example.com") -> None:
        self.current_env = _Env(host)
        self.profile = {"default_seconds": 45}


def _spec(**overrides: Any) -> ActionSpec:
    """テスト用の ActionSpec を組む。"""
    fields: dict[str, Any] = {
        "action_id": "run_load_test",
        "description": "負荷試験を実行する",
        "action_agent": "load_test_runner",
        "label": "${target} に ${seconds} 秒の負荷試験",
        "parameters": (param("target", str), param("seconds", int)),
    }
    fields.update(overrides)
    return ActionSpec(**fields)


def _catalog(spec: ActionSpec, **overrides: Any) -> ActionCatalog:
    """1 件の宣言だけを載せた ActionCatalog を組む。"""
    catalog = ActionCatalog(**overrides)
    catalog.register(spec)
    return catalog


def _candidate(**overrides: Any) -> ExecutableIntent:
    """テスト用の候補 1 件を組む。"""
    fields: dict[str, Any] = {
        "action_id": "run_load_test",
        "parameters": {},
        "level": ConfidenceLevel.HIGH,
        "source": "rule",
    }
    fields.update(overrides)
    return ExecutableIntent(**fields)


def _context(run_context: Any = None) -> IntentContext[Any]:
    """テスト用の IntentContext を組む。"""
    return IntentContext(utterance="負荷試験して", run_context=run_context)


def _plan_one(spec: ActionSpec, candidate: ExecutableIntent, run_context: Any = None) -> ActionPlan:
    """候補 1 件ぶんの ActionPlan を得る。"""
    plans = slots_module._plan_slots((candidate,), _catalog(spec), _context(run_context))
    return plans[0]


def _slot(plan: ActionPlan, name: str) -> Any:
    """名前でスロットを引く。"""
    return next(slot for slot in plan.slots if slot.name == name)


# ---------------------------------------------------------------------------
# `_resolve_path` の所在と規則 (設計 §3.4b / FR-3 L152)
# ---------------------------------------------------------------------------


def test_resolve_path_lives_in_actions_module() -> None:
    """パス解決ヘルパは `actions.py` に置かれる (設計 §3.4b)。

    利用者は `_validate.py` / `slots.py` / `_predict.py` の 3 つあり、`actions.py` は
    3 者すべてが既に import する最下層である。別モジュールへ置くと一元化が崩れ、
    `integrity.py` の同名関数（`Path` を扱う無関係の実装）との取り違えも起きる。
    """
    resolve_path = actions_module._resolve_path
    assert callable(resolve_path)
    assert resolve_path.__module__ == "oai_agentspec.runtime.intent.actions"


def test_slots_does_not_define_its_own_path_resolver() -> None:
    """`slots.py` は自前のパス解決を持たず `actions._resolve_path` を使う (設計 §3.4b)。

    同じ規則が 2 箇所に書かれると、片方だけが直される drift が起きる。
    """
    own = vars(slots_module).get("_resolve_path")
    assert own is None or own is actions_module._resolve_path


def test_resolve_path_reads_mapping_by_key() -> None:
    """mapping はキーで辿る (FR-3 L152)。"""
    assert actions_module._resolve_path({"host": "api.example.com"}, "host") == "api.example.com"


def test_resolve_path_reads_attribute_for_non_mapping() -> None:
    """mapping 以外は属性で辿る (FR-3 L152)。"""
    assert actions_module._resolve_path(_Env("api.example.com"), "host") == "api.example.com"


def test_resolve_path_recurses_on_dots() -> None:
    """`.` で分割して再帰する (FR-3 L152)。"""
    assert actions_module._resolve_path(_Ctx(), "current_env.host") == "api.example.com"


def test_resolve_path_mixes_attribute_and_mapping_segments() -> None:
    """属性と mapping が混在するパスも辿れる (FR-3 L152)。"""
    assert actions_module._resolve_path(_Ctx(), "profile.default_seconds") == 45


def test_resolve_path_returns_none_for_missing_key() -> None:
    """解決できないキーは `None` を返す（例外にしない）(FR-3 L152)。"""
    assert actions_module._resolve_path({"host": "x"}, "missing") is None


def test_resolve_path_returns_none_for_missing_attribute() -> None:
    """解決できない属性は `None` を返す (FR-3 L152)。"""
    assert actions_module._resolve_path(_Env("x"), "missing") is None


def test_resolve_path_returns_none_when_an_intermediate_segment_is_missing() -> None:
    """途中のセグメントが解決できなければ `None` を返す (FR-3 L152)。"""
    assert actions_module._resolve_path(_Ctx(), "missing.host") is None


def test_resolve_path_returns_none_for_none_object() -> None:
    """`run_context` が `None` のときも例外にせず `None` を返す (FR-5 L179)。"""
    assert actions_module._resolve_path(None, "current_env.host") is None


def test_resolve_path_prefers_mapping_key_over_attribute() -> None:
    """mapping ならキーを優先する（`dict.items` のような属性へ落ちない）(FR-3 L152)。"""
    assert actions_module._resolve_path({"items": 7}, "items") == 7


def test_resolve_path_returns_none_for_a_str_object() -> None:
    """`str` を起点にしても例外を出さず `None` を返す (FR-3 L152)。

    `hasattr(obj, "__getitem__")` で mapping 判定する実装は `"abcdef"["abc"]` を評価して
    `TypeError` になる。`run_context` の途中セグメントが文字列であることは正常系であり、
    宣言順に次のパスを試せなくなる。
    """
    assert actions_module._resolve_path("abcdef", "abc") is None


def test_resolve_path_returns_none_for_a_list_object() -> None:
    """`list` を起点にしても例外を出さず `None` を返す (FR-3 L152)。"""
    assert actions_module._resolve_path(["a"], "a") is None


def test_resolve_path_reads_a_non_dict_mapping_by_key() -> None:
    """`dict` でない `Mapping` もキーで辿る (FR-3 L152)。

    判定を `isinstance(current, dict)` へ狭めた実装は `MappingProxyType` を属性アクセス
    へ落として `None` を返す。`ActionCatalog.prompt_vars` が読み取り専用ビューであるよう
    に、`run_context` にも `dict` でない `Mapping` が現れる。
    """
    assert actions_module._resolve_path(MappingProxyType({"a": 1}), "a") == 1


def test_resolve_path_returns_a_falsy_value_as_is() -> None:
    """解決できた偽値はそのまま返す（`None` へ潰さない）(FR-3 L152 / FR-5 L179)。

    `if not current: return None` のような実装は `0` / `""` を「解決できなかった」と
    区別できなくなり、宣言した `from_context` が黙って無視される。
    """
    assert actions_module._resolve_path({"cfg": {"seconds": 0}}, "cfg.seconds") == 0
    assert actions_module._resolve_path({"a": ""}, "a") == ""


# ---------------------------------------------------------------------------
# 戻り値の形 (FR-5 L177 / L178)
# ---------------------------------------------------------------------------


def test_plan_slots_returns_one_plan_per_candidate_in_order() -> None:
    """候補と同順・同数の `ActionPlan` の tuple を返す (FR-5 L177)。"""
    spec = _spec()
    other = _spec(action_id="send_notice", label="${target}", parameters=(param("target", str),))
    catalog = _catalog(spec)
    catalog.register(other)
    candidates = (
        _candidate(parameters={"target": "a", "seconds": 1}),
        _candidate(action_id="send_notice", parameters={"target": "b"}),
        _candidate(parameters={"target": "c", "seconds": 3}),
    )
    plans = slots_module._plan_slots(candidates, catalog, _context())
    assert tuple(plan.action_id for plan in plans) == (
        "run_load_test",
        "send_notice",
        "run_load_test",
    )


def test_plan_slots_returns_exactly_one_plan_per_candidate() -> None:
    """候補と**同数**の `ActionPlan` を返す (FR-5 L177)。

    候補を絞り込んだり重複排除したりしない。件数が変わると、呼び出し側が握っている
    候補列との添字の対応が黙って崩れる。
    """
    spec = _spec()
    catalog = _catalog(spec)
    candidates = (
        _candidate(parameters={"target": "a", "seconds": 1}),
        _candidate(parameters={"target": "a", "seconds": 1}),
        _candidate(parameters={"target": "b", "seconds": 2}),
    )
    plans = slots_module._plan_slots(candidates, catalog, _context())
    assert isinstance(plans, tuple)
    assert len(plans) == len(candidates)


def test_plan_slots_raises_key_error_for_an_unregistered_action_id() -> None:
    """候補の `action_id` が宣言簿に無ければ `KeyError` (FR-5 L177 の Raises)。

    段 (1) が allowlist 除外を済ませている前提であり、ここで `None` を返したり候補を
    読み飛ばしたりして防御的に受け入れると、未登録候補が下流へ黙って流れる。
    """
    catalog = _catalog(_spec())
    candidate = _candidate(action_id="not_registered")
    with pytest.raises(KeyError):
        slots_module._plan_slots((candidate,), catalog, _context())


def test_plan_slots_returns_empty_tuple_for_no_candidates() -> None:
    """候補 0 件なら空の tuple を返す (FR-5 L177)。"""
    assert slots_module._plan_slots((), _catalog(_spec()), _context()) == ()


def test_plan_slots_keeps_declaration_order_of_slots() -> None:
    """スロットは `ActionSpec.parameters` の宣言順で並ぶ (FR-5 L188 / 設計 §3.5a)。"""
    plan = _plan_one(_spec(), _candidate(parameters={"seconds": 30, "target": "api"}))
    assert tuple(slot.name for slot in plan.slots) == ("target", "seconds")


def test_plan_slots_carries_spec_and_action_agent() -> None:
    """`spec` と `action_agent` を宣言のまま載せる (FR-5 L178)。"""
    spec = _spec()
    plan = _plan_one(spec, _candidate(parameters={"target": "api", "seconds": 30}))
    assert plan.spec == spec
    assert plan.action_agent == "load_test_runner"


def test_plan_slots_is_synchronous() -> None:
    """決定的段は `await` を要さない（LLM を呼ばないことの構造的表現）(FR-5 L177)。"""
    import inspect

    assert not inspect.iscoroutinefunction(slots_module._plan_slots)


# ---------------------------------------------------------------------------
# 値の解決順 (FR-5 L179)
# ---------------------------------------------------------------------------


def test_candidate_parameters_win_over_from_context() -> None:
    """候補の `parameters` は `from_context` より優先される (FR-5 L179)。"""
    spec = _spec(
        parameters=(
            param("target", str, from_context=("current_env.host",)),
            param("seconds", int, default=30),
        ),
    )
    plan = _plan_one(spec, _candidate(parameters={"target": "from-candidate"}), _Ctx())
    slot = _slot(plan, "target")
    assert slot.value == "from-candidate"
    assert slot.origin is Origin.CANDIDATE


def test_from_context_wins_over_default() -> None:
    """`from_context` は `default` より優先される (FR-5 L179)。"""
    spec = _spec(
        parameters=(
            param("target", str, from_context=("current_env.host",), default="fallback"),
            param("seconds", int, default=30),
        ),
    )
    slot = _slot(_plan_one(spec, _candidate(), _Ctx()), "target")
    assert slot.value == "api.example.com"
    assert slot.origin is Origin.RUN_CONTEXT


def test_by_llm_wins_over_default() -> None:
    """`by_llm` は `default` より優先され、この段では未実施のまま `NEEDS_LLM` になる (FR-5 L179)。

    解決順が「`by_llm` -> `default`」であるため、`default` を宣言していても
    決定的段で `DEFAULT` へ倒してはならない（倒すと予測が一度も走らない）。
    `default` は FR-7 の後退先として予測段が使う。
    """
    spec = _spec(
        parameters=(
            param("target", str, default="fallback", by_llm=True),
            param("seconds", int, default=30),
        ),
    )
    slot = _slot(_plan_one(spec, _candidate()), "target")
    assert slot.state is SlotState.NEEDS_LLM
    assert slot.value is None
    assert slot.origin is None


def test_default_is_used_when_nothing_else_resolves() -> None:
    """候補にも `from_context` にも無く `by_llm` でなければ `default` を採る (FR-5 L179)。"""
    spec = _spec(
        parameters=(param("target", str, default="fallback"), param("seconds", int, default=30)),
    )
    slot = _slot(_plan_one(spec, _candidate()), "target")
    assert slot.state is SlotState.RESOLVED
    assert slot.value == "fallback"
    assert slot.origin is Origin.DEFAULT


def test_from_context_tries_paths_in_declaration_order() -> None:
    """`from_context` は宣言順に試し、最初の非 `None` を採る (FR-5 L179)。"""
    spec = _spec(
        parameters=(
            param("target", str, from_context=("current_env.missing", "current_env.host")),
            param("seconds", int, default=30),
        ),
    )
    slot = _slot(_plan_one(spec, _candidate(), _Ctx()), "target")
    assert slot.value == "api.example.com"
    assert slot.detail == "current_env.host"


def test_from_context_stops_at_the_first_non_none_path() -> None:
    """先に宣言したパスが解決すれば後続のパスは採らない (FR-5 L179)。"""
    spec = _spec(
        parameters=(
            param("target", str, from_context=("profile.default_host", "current_env.host")),
            param("seconds", int, default=30),
        ),
    )
    run_context = _Ctx()
    run_context.profile = {"default_host": "first.example.com"}
    slot = _slot(_plan_one(spec, _candidate(), run_context), "target")
    assert slot.value == "first.example.com"
    assert slot.detail == "profile.default_host"


def test_from_context_falls_through_when_every_path_is_none() -> None:
    """全パスが `None` なら次の優先順位へ落ちる (FR-5 L179)。"""
    spec = _spec(
        parameters=(
            param("target", str, from_context=("current_env.host",), default="fallback"),
            param("seconds", int, default=30),
        ),
    )
    slot = _slot(_plan_one(spec, _candidate(), _Ctx(host=None)), "target")
    assert slot.value == "fallback"
    assert slot.origin is Origin.DEFAULT


def test_from_context_keeps_a_falsy_zero_value() -> None:
    """`from_context` が `0` を解決したら `default` へ倒さない (FR-5 L179)。

    パス解決の成否は `is not None` で判定する。`if value:` と書いた実装は `0` を
    「解決できなかった」と扱い、宣言した `default` を利用者に無断で採る。
    """
    spec = _spec(
        parameters=(
            param("target", str, default="fallback"),
            param("seconds", int, from_context=("cfg.seconds",), default=30),
        ),
    )
    slot = _slot(_plan_one(spec, _candidate(), {"cfg": {"seconds": 0}}), "seconds")
    assert slot.state is SlotState.RESOLVED
    assert slot.value == 0
    assert slot.origin is Origin.RUN_CONTEXT
    assert slot.detail == "cfg.seconds"


def test_from_context_keeps_an_empty_string_value() -> None:
    """`from_context` が空文字を解決したら `default` へ倒さない (FR-5 L179)。"""
    spec = _spec(
        parameters=(
            param("target", str, from_context=("cfg.host",), default="fallback"),
            param("seconds", int, default=30),
        ),
    )
    slot = _slot(_plan_one(spec, _candidate(), {"cfg": {"host": ""}}), "target")
    assert slot.state is SlotState.RESOLVED
    assert slot.value == ""
    assert slot.origin is Origin.RUN_CONTEXT


# ---------------------------------------------------------------------------
# 状態遷移 (FR-5 L180-L183)
# ---------------------------------------------------------------------------


def test_resolved_slot_when_value_is_found_and_confirm_is_false() -> None:
    """値を得て `confirm=False` なら `RESOLVED` になる (FR-5 L180)。"""
    plan = _plan_one(_spec(), _candidate(parameters={"target": "api", "seconds": 30}))
    slot = _slot(plan, "seconds")
    assert slot.state is SlotState.RESOLVED
    assert slot.value == 30
    assert slot.origin is Origin.CANDIDATE
    assert slot.suggestions == ()


def test_needs_confirmation_slot_when_confirm_is_true() -> None:
    """値を得て `confirm=True` なら `NEEDS_CONFIRMATION` になる (FR-5 L181)。"""
    spec = _spec(parameters=(param("target", str), param("seconds", int, confirm=True)))
    plan = _plan_one(spec, _candidate(parameters={"target": "api", "seconds": 30}))
    slot = _slot(plan, "seconds")
    assert slot.state is SlotState.NEEDS_CONFIRMATION
    assert slot.value is None
    assert slot.origin is Origin.CANDIDATE


def test_needs_confirmation_carries_exactly_one_suggestion() -> None:
    """`NEEDS_CONFIRMATION` の `suggestions` は解決した値 1 件である (FR-5 L181)。"""
    spec = _spec(parameters=(param("target", str), param("seconds", int, confirm=True)))
    plan = _plan_one(spec, _candidate(parameters={"target": "api", "seconds": 30}))
    slot = _slot(plan, "seconds")
    assert len(slot.suggestions) == 1
    assert slot.suggestions[0].value == 30


def test_needs_confirmation_suggestion_uses_the_declared_default_level() -> None:
    """`suggestions` の `level` は宣言側の既定値である (FR-5 L181)。

    決定的に解決した値は推測ではないため `CERTAIN` が既定になる。
    """
    spec = _spec(parameters=(param("target", str), param("seconds", int, confirm=True)))
    plan = _plan_one(spec, _candidate(parameters={"target": "api", "seconds": 30}))
    assert _slot(plan, "seconds").suggestions[0].level is ConfidenceLevel.CERTAIN


def test_confirm_applies_to_from_context_values_too() -> None:
    """`confirm=True` は `from_context` 由来の値にも効き `detail` を保つ (FR-5 L181 / L184)。"""
    spec = _spec(
        parameters=(
            param("target", str, from_context=("current_env.host",), confirm=True),
            param("seconds", int, default=30),
        ),
    )
    slot = _slot(_plan_one(spec, _candidate(), _Ctx()), "target")
    assert slot.state is SlotState.NEEDS_CONFIRMATION
    assert slot.origin is Origin.RUN_CONTEXT
    assert slot.detail == "current_env.host"
    assert slot.suggestions[0].value == "api.example.com"


def test_needs_llm_slot_when_value_is_missing_and_by_llm_is_true() -> None:
    """値を得られず `by_llm=True` なら `NEEDS_LLM` になる (FR-5 L182)。"""
    spec = _spec(parameters=(param("target", str, by_llm=True), param("seconds", int, default=30)))
    slot = _slot(_plan_one(spec, _candidate()), "target")
    assert slot.state is SlotState.NEEDS_LLM
    assert slot.origin is None
    assert slot.value is None
    assert slot.suggestions == ()


def test_needs_user_slot_when_value_is_missing_and_by_llm_is_false() -> None:
    """値を得られず `by_llm=False` なら `NEEDS_USER` になる (FR-5 L183)。"""
    slot = _slot(_plan_one(_spec(), _candidate()), "target")
    assert slot.state is SlotState.NEEDS_USER
    assert slot.origin is None
    assert slot.value is None
    assert slot.suggestions == ()


def test_explicit_none_default_resolves_to_a_resolved_slot() -> None:
    """明示的な `default=None` は `RESOLVED` + `value=None` になる (FR-5 L187 / 設計 §3.7)。

    `value` の `None` は「未解決」ではなく「値が `None` であること」を意味する。
    """
    spec = _spec(
        label="${target}",
        parameters=(param("target", str | None, default=None),),
    )
    slot = _slot(_plan_one(spec, _candidate()), "target")
    assert slot.state is SlotState.RESOLVED
    assert slot.value is None
    assert slot.origin is Origin.DEFAULT


def test_plan_is_ready_only_when_every_slot_is_resolved() -> None:
    """全スロットが `RESOLVED` のときだけ `ready` になる (FR-5 L188)。"""
    resolved = _plan_one(_spec(), _candidate(parameters={"target": "api", "seconds": 30}))
    partial = _plan_one(_spec(), _candidate(parameters={"target": "api"}))
    assert resolved.ready
    assert not partial.ready


# ---------------------------------------------------------------------------
# origin / detail の記録 (FR-5 L184)
# ---------------------------------------------------------------------------


def test_run_context_origin_records_the_resolved_path_in_detail() -> None:
    """`RUN_CONTEXT` の `detail` は解決に成功したパスである (FR-5 L184)。"""
    spec = _spec(
        parameters=(
            param("target", str, from_context=("current_env.host",)),
            param("seconds", int, default=30),
        ),
    )
    slot = _slot(_plan_one(spec, _candidate(), _Ctx()), "target")
    assert slot.origin is Origin.RUN_CONTEXT
    assert slot.detail == "current_env.host"


def test_candidate_origin_has_no_detail() -> None:
    """`CANDIDATE` の `detail` は `None` である (FR-5 L184)。"""
    plan = _plan_one(_spec(), _candidate(parameters={"target": "api", "seconds": 30}))
    assert _slot(plan, "target").detail is None


def test_default_origin_has_no_detail() -> None:
    """`DEFAULT` の `detail` は `None` である (FR-5 L184)。"""
    spec = _spec(
        parameters=(param("target", str, default="fallback"), param("seconds", int, default=30)),
    )
    assert _slot(_plan_one(spec, _candidate()), "target").detail is None


def test_deterministic_stage_never_records_llm_or_user_origins() -> None:
    """決定的段は `LLM` / `USER_*` の `origin` を作らない (FR-5 L179 / L184)。

    予測段（FR-6）と `apply`（FR-8）だけが作れる出どころであり、ここで作られると
    「利用者が決めた値」の選り分け（`from_user`）が壊れる。
    """
    spec = _spec(
        parameters=(
            param("target", str, from_context=("current_env.host",)),
            param("seconds", int, default=30),
        ),
    )
    plan = _plan_one(spec, _candidate(), _Ctx())
    origins = {slot.origin for slot in plan.slots}
    assert origins <= {Origin.CANDIDATE, Origin.RUN_CONTEXT, Origin.DEFAULT}


def test_deterministic_slots_never_report_from_user() -> None:
    """決定的段のスロットは `from_user` が偽である (設計 §3.5a)。"""
    spec = _spec(
        parameters=(param("target", str, default="fallback"), param("seconds", int, default=30)),
    )
    plan = _plan_one(spec, _candidate())
    assert not any(slot.from_user for slot in plan.slots)


# ---------------------------------------------------------------------------
# 既定マージ解決 3 関数の呼び出し (設計 §3.4a / §5 タスク 1-6)
# ---------------------------------------------------------------------------


def test_plan_carries_resolved_prompt_from_the_merge_function() -> None:
    """`resolved_prompt` は `resolve_prompt` の結果と一致する (設計 §3.4a)。"""
    spec = _spec(prompt=("action_seg",))
    catalog = _catalog(spec, prompt=("catalog_seg",))
    plan = slots_module._plan_slots((_candidate(),), catalog, _context())[0]
    assert plan.resolved_prompt == resolve_prompt(catalog, spec)
    assert plan.resolved_prompt == ("catalog_seg", "action_seg")


def test_plan_carries_resolved_prompt_vars_from_the_merge_function() -> None:
    """`resolved_prompt_vars` は `resolve_prompt_vars` の結果と一致する (設計 §3.4a)。"""
    spec = _spec(prompt_vars={"host": "current_env.host", "who": "profile.name"})
    catalog = _catalog(spec, prompt_vars={"who": "catalog.name", "when": "clock.now"})
    plan = slots_module._plan_slots((_candidate(),), catalog, _context())[0]
    assert dict(plan.resolved_prompt_vars) == dict(resolve_prompt_vars(catalog, spec))
    assert dict(plan.resolved_prompt_vars) == {
        "who": "profile.name",
        "when": "clock.now",
        "host": "current_env.host",
    }


def test_plan_carries_resolved_on_invalid_slot_from_the_merge_function() -> None:
    """`resolved_on_invalid_slot` は `resolve_on_invalid_slot` の結果と一致する (設計 §3.4a)。"""
    spec = _spec(on_invalid_slot="error")
    catalog = _catalog(spec, on_invalid_slot="skip")
    plan = slots_module._plan_slots((_candidate(),), catalog, _context())[0]
    assert plan.resolved_on_invalid_slot == resolve_on_invalid_slot(catalog, spec)
    assert plan.resolved_on_invalid_slot == "error"


def test_plan_falls_back_to_catalog_on_invalid_slot() -> None:
    """`ActionSpec` 側が未宣言ならカタログ既定が載る (設計 §3.4a)。"""
    spec = _spec()
    catalog = _catalog(spec, on_invalid_slot="error")
    plan = slots_module._plan_slots((_candidate(),), catalog, _context())[0]
    assert plan.resolved_on_invalid_slot == "error"


def test_plan_keeps_spec_as_declared_after_merging() -> None:
    """マージ結果を `spec` へ書き戻さない（`spec` は as-declared・設計 §3.4a）。

    起動時検証の「当該 `ActionSpec` 自身の宣言に限る」検査（FR-3 L156）が、
    マージ済みの値を見てしまうと成立しなくなる。
    """
    spec = _spec()
    catalog = _catalog(spec, prompt=("catalog_seg",))
    plan = slots_module._plan_slots((_candidate(),), catalog, _context())[0]
    assert plan.spec.prompt == ()


# ---------------------------------------------------------------------------
# 決定性 (FR-5 L177 / NFR-1)
# ---------------------------------------------------------------------------


def test_same_inputs_produce_equal_plans() -> None:
    """同一の候補列と同一の `run_context` に対し常に同一結果を返す (FR-5 L177)。"""
    spec = _spec(
        parameters=(
            param("target", str, from_context=("current_env.host",)),
            param("seconds", int, default=30),
        ),
    )
    catalog = _catalog(spec)
    run_context = _Ctx()
    candidates = (_candidate(),)
    first = slots_module._plan_slots(candidates, catalog, _context(run_context))
    second = slots_module._plan_slots(candidates, catalog, _context(run_context))
    assert first == second


def test_environment_variables_do_not_change_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """環境変数を参照しない (FR-5 L177)。"""
    spec = _spec(
        parameters=(
            param("target", str, from_context=("current_env.host",)),
            param("seconds", int, default=30),
        ),
    )
    catalog = _catalog(spec)
    run_context = _Ctx()
    monkeypatch.setenv("OPENAI_API_KEY", "first")
    monkeypatch.setenv("OAI_AGENTSPEC_TEST_KNOB", "on")
    first = slots_module._plan_slots((_candidate(),), catalog, _context(run_context))
    monkeypatch.delenv("OAI_AGENTSPEC_TEST_KNOB")
    monkeypatch.setenv("OPENAI_API_KEY", "second")
    second = slots_module._plan_slots((_candidate(),), catalog, _context(run_context))
    assert first == second


def test_no_network_access_during_slot_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """ネットワークを参照しない (FR-5 L177)。

    ソケット生成そのものを禁止して、決定的段が通信を行わないことを構造的に固定する。
    """

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the deterministic slot stage must not open a socket")

    monkeypatch.setattr(socket, "socket", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)
    spec = _spec(
        parameters=(
            param("target", str, from_context=("current_env.host",)),
            param("seconds", int, by_llm=True),
        ),
    )
    plans = slots_module._plan_slots((_candidate(),), _catalog(spec), _context(_Ctx()))
    assert _slot(plans[0], "seconds").state is SlotState.NEEDS_LLM


def test_slots_module_does_not_import_the_sdk() -> None:
    """`slots.py` は `agents` / `openai` を import しない (NFR-1 / FR-5 L177)。"""
    source = _SLOTS_PATH.read_text(encoding="utf-8")
    assert "import agents" not in source
    assert "from agents" not in source
    assert "import openai" not in source
    assert "from openai" not in source


def test_slots_module_does_not_import_the_adapter_layer() -> None:
    """`slots.py` は LLM 実行アダプタを import しない (FR-5 L177)。"""
    source = _SLOTS_PATH.read_text(encoding="utf-8")
    import_lines = [
        line
        for line in source.splitlines()
        if line.lstrip().startswith(("import ", "from ")) and "import" in line
    ]
    assert not [line for line in import_lines if "_adapters" in line]


def test_plan_slots_does_not_mutate_the_candidate() -> None:
    """候補を変更しない（同じ候補列で 2 回呼べる）(FR-5 L177)。"""
    candidate = _candidate(parameters={"target": "api", "seconds": 30})
    before = dict(candidate.parameters)
    slots_module._plan_slots((candidate,), _catalog(_spec()), _context())
    assert dict(candidate.parameters) == before


def test_plan_slots_does_not_mutate_the_catalog() -> None:
    """宣言簿を変更しない (FR-5 L177)。"""
    catalog = _catalog(_spec())
    slots_module._plan_slots((_candidate(),), catalog, _context())
    assert catalog.names() == ["run_load_test"]


# ---------------------------------------------------------------------------
# 非公開性 (設計 §3.13 / §3.5b)
# ---------------------------------------------------------------------------


def test_plan_slots_is_not_a_public_symbol() -> None:
    """決定的段の関数は公開しない (設計 §3.13)。

    案 1 の 3 呼び出し（`suggest_executable_intents` / `plan_slots` / `predict_params`）は
    `ActionPlanner.plan()` へ畳まれ、実装だけが各モジュールへ残る。
    """
    from oai_agentspec.runtime import intent as intent_package

    assert slots_module._plan_slots.__name__.startswith("_")
    assert "plan_slots" not in intent_package.__all__
    assert "_plan_slots" not in intent_package.__all__


def test_plan_spec_is_the_registered_instance() -> None:
    """`plan.spec` は宣言簿の登録インスタンスそのもの（複製ではない）(FR-5 L178)。

    複製を挟むと、カタログ側で `parameters_model()` を呼ぶ前に作られた計画ごとに
    別のモデルクラスが生成され、計画をまたいだ検証型が一致しなくなる。
    """
    spec = _spec()
    catalog = _catalog(spec)
    plans = slots_module._plan_slots((_candidate(),), catalog, _context())
    assert plans[0].spec is spec
    assert plans[0].spec is catalog.get("run_load_test")


# ---------------------------------------------------------------------------
# 組み立てた計画の resolved_prompt_vars は読み取り専用 (レビュー 2 巡目・指摘 #88-W2 系)
# ---------------------------------------------------------------------------
#
# `_plan_slots` は `resolve_prompt_vars()` が返した素の dict をそのまま `ActionPlan` へ
# 載せるため、宣言側 (`ActionSpec.prompt_vars`) を読み取り専用にしても、マージ後の値は
# プロンプト展開の直前まで書き換えられる。読み取り専用化する境界は「マージ関数の戻り値」
# ではなく「計画のフィールドとして保持された後」であることを、両方向から pin する。


def test_plan_slots_returns_a_read_only_resolved_prompt_vars() -> None:
    """`_plan_slots` が組んだ計画の resolved_prompt_vars は書き換えられない (指摘 #88-W2 系)。"""
    spec = _spec(prompt_vars={"host": "current_env.host"})
    catalog = _catalog(spec, prompt_vars={"who": "catalog.name"})
    plan = slots_module._plan_slots((_candidate(),), catalog, _context())[0]
    with pytest.raises(TypeError) as assignment:
        plan.resolved_prompt_vars["secret"] = "credentials.api_key"
    assert type(assignment.value) is TypeError
    with pytest.raises(TypeError) as deletion:
        del plan.resolved_prompt_vars["host"]
    assert type(deletion.value) is TypeError
    assert dict(plan.resolved_prompt_vars) == {
        "host": "current_env.host",
        "who": "catalog.name",
    }


def test_resolve_prompt_vars_still_returns_a_mutable_dict_for_the_caller() -> None:
    """マージ関数の戻り値は可変 dict のままで、計画へ載った後だけ読み取り専用になる。

    読み取り専用化するのは `ActionPlan` が保持した後である（既存の pin
    `test_resolve_prompt_vars_still_returns_a_plain_mutable_dict` と矛盾しないこと）。
    戻り値を書き換えても、既に組み立て済みの計画には影響しない。
    """
    spec = _spec(prompt_vars={"host": "current_env.host"})
    catalog = _catalog(spec, prompt_vars={"who": "catalog.name"})
    plan = slots_module._plan_slots((_candidate(),), catalog, _context())[0]
    merged = resolve_prompt_vars(catalog, spec)
    assert type(merged) is dict
    merged["secret"] = "credentials.api_key"
    assert dict(plan.resolved_prompt_vars) == {
        "host": "current_env.host",
        "who": "catalog.name",
    }
