"""L1: `runtime.intent.actions` の宣言簿層 (`ActionSpec` / `ParameterSpec` / `ActionCatalog` /
`param` / `PARAM_UNSET`) の純検証。

FR-1 の受け入れ基準を pin する。`ActionSpec` / `ParameterSpec` の frozen 性と直列化契約、
`ActionCatalog` が plain な mutable クラスであること (公開メソッドは register / names / get /
bind の 4 つ)、`action_id` の 4 分岐 + 予約語衝突検証、`param()` の引数正規化と検証、
既定 (prompt / prompt_vars / on_invalid_slot) のマージ・上書き解決、`ParameterSpec` が
sentinel をフィールドに持たないこと (設計 §3.7) を対象とする。外部依存 (agents / openai) なし。
"""

from __future__ import annotations

import copy
import os
import pickle
import subprocess
import sys
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any, ForwardRef, Literal, Optional

import pytest
from pydantic import BaseModel, ValidationError

from oai_agentspec.runtime.intent.actions import (
    PARAM_UNSET,
    ActionCatalog,
    ActionPlanner,
    ActionSpec,
    ParameterSpec,
    param,
    resolve_on_invalid_slot,
    resolve_prompt,
    resolve_prompt_vars,
)

pytestmark = pytest.mark.unit

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
_ACTIONS_PATH = _SRC_DIR / "oai_agentspec" / "runtime" / "intent" / "actions.py"


def _make_spec(
    action_id: str = "send_message",
    **overrides: Any,
) -> ActionSpec:
    """テスト用の最小 ActionSpec を組む。"""
    fields: dict[str, Any] = {
        "action_id": action_id,
        "description": "メッセージを送る",
        "action_agent": "messenger",
        "label": "${target} へ送信",
        "parameters": (param("target", str),),
    }
    fields.update(overrides)
    return ActionSpec(**fields)


# ---------------------------------------------------------------------------
# ActionSpec の宣言と frozen 性 (FR-1 L102 / L103)
# ---------------------------------------------------------------------------


def test_action_spec_declares_all_fields() -> None:
    """ActionSpec は宣言した全フィールドを保持する (FR-1 L102)。"""
    p = param("target", str)
    spec = ActionSpec(
        action_id="send_message",
        description="メッセージを送る",
        action_agent="messenger",
        label="${target} へ送信",
        parameters=(p,),
        prompt=("actions/send",),
        prompt_vars={"tone": "polite"},
        on_invalid_slot="error",
    )
    assert spec.action_id == "send_message"
    assert spec.description == "メッセージを送る"
    assert spec.action_agent == "messenger"
    assert spec.label == "${target} へ送信"
    assert spec.parameters == (p,)
    assert spec.prompt == ("actions/send",)
    assert spec.prompt_vars == {"tone": "polite"}
    assert spec.on_invalid_slot == "error"


def test_action_spec_optional_fields_have_declared_defaults() -> None:
    """prompt / prompt_vars / on_invalid_slot の既定は () / {} / None (FR-1 L102)。"""
    spec = _make_spec()
    assert spec.prompt == ()
    assert spec.prompt_vars == {}
    assert spec.on_invalid_slot is None


def test_action_spec_is_frozen() -> None:
    """ActionSpec は frozen な pydantic BaseModel (再代入は ValidationError) (FR-1 L103)。"""
    spec = _make_spec()
    assert isinstance(spec, BaseModel)
    with pytest.raises(ValidationError):
        spec.action_id = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        spec.description = "other"  # type: ignore[misc]


def test_parameter_spec_is_frozen() -> None:
    """ParameterSpec は frozen な pydantic BaseModel (再代入は ValidationError) (FR-1 L103)。"""
    p = param("target", str)
    assert isinstance(p, BaseModel)
    with pytest.raises(ValidationError):
        p.name = "other"  # type: ignore[misc]


def test_action_spec_and_parameter_spec_are_serializable() -> None:
    """ActionSpec / ParameterSpec は model_dump / model_json_schema が成立する (FR-1 L105)。"""
    spec = _make_spec()
    assert isinstance(spec.model_dump(), dict)
    assert isinstance(spec.model_json_schema(), dict)
    p = spec.parameters[0]
    assert isinstance(p.model_dump(), dict)
    assert isinstance(p.model_json_schema(), dict)


# ---------------------------------------------------------------------------
# ActionCatalog の型と 4 メソッド (FR-1 L106)
# ---------------------------------------------------------------------------


def test_action_catalog_is_plain_mutable_class() -> None:
    """ActionCatalog は frozen 契約の対象外の plain クラスである (FR-1 L106)。"""
    catalog = ActionCatalog()
    assert not isinstance(catalog, BaseModel)


def test_action_catalog_exposes_exactly_four_public_methods() -> None:
    """公開メソッドは register / names / get / bind の 4 つだけ (FR-1 L106)。"""
    catalog = ActionCatalog()
    methods = {
        name
        for name in dir(catalog)
        if not name.startswith("_") and callable(getattr(catalog, name))
    }
    assert methods == {"register", "names", "get", "bind"}


def test_action_catalog_has_no_validate_or_plan() -> None:
    """validate / plan は ActionPlanner 側であり ActionCatalog には無い (FR-1 L106)。"""
    catalog = ActionCatalog()
    assert not hasattr(catalog, "validate")
    assert not hasattr(catalog, "plan")


def test_action_catalog_register_keeps_declaration() -> None:
    """register した宣言は get で同一オブジェクトとして取り出せる (FR-1 L102)。"""
    catalog = ActionCatalog()
    spec = _make_spec()
    catalog.register(spec)
    assert catalog.get("send_message") is spec


def test_action_catalog_rejects_duplicate_action_id() -> None:
    """同一 action_id の再登録は ValueError (FR-1 L102)。"""
    catalog = ActionCatalog()
    catalog.register(_make_spec())
    with pytest.raises(ValueError):
        catalog.register(_make_spec())


def test_action_catalog_names_returns_sorted_list() -> None:
    """names() は登録済み action_id を昇順の list で返す (FR-1 L107)。"""
    catalog = ActionCatalog()
    for action_id in ("send_message", "archive", "notify"):
        catalog.register(_make_spec(action_id=action_id))
    names = catalog.names()
    assert isinstance(names, list)
    assert names == ["archive", "notify", "send_message"]


def test_action_catalog_names_is_empty_list_when_unregistered() -> None:
    """未登録の ActionCatalog の names() は空 list (FR-1 L107)。"""
    assert ActionCatalog().names() == []


def test_action_catalog_get_raises_key_error_for_unknown_id() -> None:
    """未登録 action_id の get() は KeyError (FR-1 L108)。"""
    catalog = ActionCatalog()
    catalog.register(_make_spec())
    with pytest.raises(KeyError):
        catalog.get("unknown_action")


# ---------------------------------------------------------------------------
# action_id の 4 分岐検証 + 予約語衝突 (FR-1 L109)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action_id", "branch"),
    [
        ("", "空文字"),
        ("send-message", "非識別子"),
        ("send message", "非識別子"),
        ("_send", "アンダースコア始まり"),
        ("class", "Python 予約語"),
        ("None", "Python 予約語"),
    ],
)
def test_action_spec_rejects_invalid_action_id(action_id: str, branch: str) -> None:
    """action_id の 4 分岐 (空文字 / 非識別子 / _ 始まり / 予約語) は ValueError (FR-1 L109)。"""
    with pytest.raises(ValueError):
        _make_spec(action_id=action_id)


@pytest.mark.parametrize("reserved", ["register", "names", "get", "bind"])
def test_action_spec_rejects_catalog_method_names(reserved: str) -> None:
    """ActionCatalog の公開メソッド名 4 件と衝突する action_id は ValueError (FR-1 L109)。"""
    with pytest.raises(ValueError):
        _make_spec(action_id=reserved)


@pytest.mark.parametrize("allowed", ["validate", "plan"])
def test_action_spec_accepts_planner_method_names(allowed: str) -> None:
    """validate / plan は ActionCatalog の公開メソッドでないため合法 (FR-1 L109)。"""
    spec = _make_spec(action_id=allowed)
    assert spec.action_id == allowed


def test_action_catalog_accepts_planner_method_names_on_register() -> None:
    """validate / plan を action_id とする宣言は register でき names に現れる (FR-1 L109)。"""
    catalog = ActionCatalog()
    catalog.register(_make_spec(action_id="validate"))
    catalog.register(_make_spec(action_id="plan"))
    assert catalog.names() == ["plan", "validate"]


# ---------------------------------------------------------------------------
# param() のシグネチャと正規化 (FR-1 L110 / L113)
# ---------------------------------------------------------------------------


def test_param_returns_parameter_spec_with_defaults() -> None:
    """param() は ParameterSpec を返し、省略時の既定を持つ (FR-1 L110)。"""
    p = param("target", str)
    assert isinstance(p, ParameterSpec)
    assert p.name == "target"
    assert p.annotation is str
    assert p.from_context == ()
    assert p.by_llm is False
    assert p.prompt is None
    assert p.description is None
    assert p.max_suggestions == 1
    assert p.confirm is False
    assert p.filled_by_candidate is False


def test_param_keeps_all_keyword_arguments() -> None:
    """param() は全キーワード引数を ParameterSpec へ載せる (FR-1 L110)。"""
    p = param(
        "seconds",
        int,
        from_context=("run_context.seconds",),
        by_llm=True,
        prompt="actions/seconds",
        description="待機秒数",
        default=60,
        max_suggestions=3,
        confirm=True,
        filled_by_candidate=True,
    )
    assert p.name == "seconds"
    assert p.annotation is int
    assert p.from_context == ("run_context.seconds",)
    assert p.by_llm is True
    assert p.prompt == "actions/seconds"
    assert p.description == "待機秒数"
    assert p.max_suggestions == 3
    assert p.confirm is True
    assert p.filled_by_candidate is True


def test_param_has_no_extra_argument() -> None:
    """param() に extra 引数は存在しない (FR-1 L110)。"""
    with pytest.raises(TypeError):
        param("target", str, extra={"k": "v"})  # type: ignore[call-arg]


def test_param_normalizes_str_from_context_to_single_tuple() -> None:
    """from_context に str を渡すと 1 要素 tuple へ正規化される (FR-1 L113)。"""
    p = param("target", str, from_context="run_context.target")
    assert p.from_context == ("run_context.target",)


def test_param_preserves_from_context_tuple_order() -> None:
    """from_context に tuple を渡すと宣言順を保持する (FR-1 L113)。"""
    p = param("target", str, from_context=("a.b", "c.d", "e"))
    assert p.from_context == ("a.b", "c.d", "e")


def test_param_rejects_non_identifier_name() -> None:
    """param の name が str.isidentifier() 偽なら ValueError (FR-1 L111)。"""
    for bad in ("", "target-1", "1target", "with space"):
        with pytest.raises(ValueError):
            param(bad, str)


@pytest.mark.parametrize("max_suggestions", [0, -1])
def test_param_rejects_max_suggestions_below_one(max_suggestions: int) -> None:
    """max_suggestions が 1 未満なら ValueError (FR-1 L114)。"""
    with pytest.raises(ValueError):
        param("target", str, max_suggestions=max_suggestions)


def test_action_spec_rejects_duplicate_parameter_names() -> None:
    """同一 ActionSpec 内に同名の ParameterSpec が 2 件以上あれば ValueError (FR-1 L112)。"""
    with pytest.raises(ValueError):
        _make_spec(parameters=(param("target", str), param("target", int)))


def test_action_spec_accepts_distinct_parameter_names() -> None:
    """名前が異なるパラメータは宣言順のまま保持される (FR-1 L112)。"""
    spec = _make_spec(parameters=(param("target", str), param("seconds", int)))
    assert [p.name for p in spec.parameters] == ["target", "seconds"]


# ---------------------------------------------------------------------------
# PARAM_UNSET と ParameterSpec の内部表現 (設計 §3.6 / §3.7)
# ---------------------------------------------------------------------------


def test_param_unset_is_a_dedicated_sentinel() -> None:
    """PARAM_UNSET は None ではない専用 sentinel であり identity が安定する (設計 §3.6)。"""
    assert PARAM_UNSET is not None
    assert PARAM_UNSET is PARAM_UNSET


def test_parameter_spec_does_not_hold_the_sentinel_as_a_field() -> None:
    """default 未宣言でも ParameterSpec のどのフィールドにも PARAM_UNSET は載らない (設計 §3.7)。"""
    p = param("target", str)
    values = [getattr(p, name) for name in type(p).model_fields]
    assert all(value is not PARAM_UNSET for value in values)


def test_parameter_spec_separates_unset_from_explicit_none() -> None:
    """未宣言と明示的な default=None を has_default で区別する (設計 §3.7 案 B)。"""
    unset = param("target", str)
    assert unset.has_default is False
    assert unset.default is None

    explicit_none = param("target", str, default=None)
    assert explicit_none.has_default is True
    assert explicit_none.default is None

    explicit_value = param("seconds", int, default=60)
    assert explicit_value.has_default is True
    assert explicit_value.default == 60


def test_parameter_spec_json_schema_emits_no_warning() -> None:
    """sentinel をフィールドに持たないため model_json_schema() が警告を出さない (設計 §3.7)。"""
    p = param("target", str)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        schema = p.model_json_schema()
    assert isinstance(schema, dict)


# ---------------------------------------------------------------------------
# ActionCatalog の既定と解決規則 (FR-1 L115)
# ---------------------------------------------------------------------------


def test_action_catalog_holds_declared_defaults() -> None:
    """ActionCatalog は prompt / prompt_vars / on_invalid_slot の既定を保持する (FR-1 L115)。"""
    catalog = ActionCatalog()
    assert catalog.prompt == ()
    assert catalog.prompt_vars == {}
    assert catalog.on_invalid_slot == "skip"


def test_action_catalog_holds_explicit_defaults() -> None:
    """明示した既定値がそのまま保持される (FR-1 L115)。"""
    catalog = ActionCatalog(
        prompt=("common/base",),
        prompt_vars={"tone": "polite"},
        on_invalid_slot="error",
    )
    assert catalog.prompt == ("common/base",)
    assert catalog.prompt_vars == {"tone": "polite"}
    assert catalog.on_invalid_slot == "error"


def test_action_catalog_copies_the_given_prompt_vars() -> None:
    """渡した dict を防御的にコピーし、呼び出し側の後からの変更に影響されない (FR-1 L115)。"""
    source = {"tone": "polite"}
    catalog = ActionCatalog(prompt_vars=source)
    source["tone"] = "casual"
    assert catalog.prompt_vars == {"tone": "polite"}


def test_resolve_prompt_merges_with_action_last() -> None:
    """prompt はマージされアクション側を後に積む (FR-1 L115)。"""
    catalog = ActionCatalog(prompt=("common/base",))
    spec = _make_spec(prompt=("actions/send",))
    assert resolve_prompt(catalog, spec) == ("common/base", "actions/send")


def test_resolve_prompt_vars_merges_with_action_overriding() -> None:
    """prompt_vars はマージされ、同名キーはアクション側が勝つ (FR-1 L115)。"""
    catalog = ActionCatalog(prompt_vars={"tone": "polite", "locale": "ja"})
    spec = _make_spec(prompt_vars={"tone": "casual"})
    assert resolve_prompt_vars(catalog, spec) == {"tone": "casual", "locale": "ja"}


def test_resolve_on_invalid_slot_is_overridden_by_action() -> None:
    """on_invalid_slot はマージではなく上書きで解決される (FR-1 L115)。"""
    catalog = ActionCatalog(on_invalid_slot="error")
    assert resolve_on_invalid_slot(catalog, _make_spec(on_invalid_slot="skip")) == "skip"


def test_resolve_on_invalid_slot_falls_back_to_catalog() -> None:
    """ActionSpec 側が None なら ActionCatalog の既定が採られる (FR-1 L115)。"""
    catalog = ActionCatalog(on_invalid_slot="error")
    assert resolve_on_invalid_slot(catalog, _make_spec()) == "error"


# ---------------------------------------------------------------------------
# on_invalid_slot の値検証は宣言時に落とす (FR-1 L116)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["halt", "ERROR", "", "raise", None])
def test_action_catalog_rejects_invalid_on_invalid_slot(bad: Any) -> None:
    """ActionCatalog の宣言時に error / skip 以外を渡すと ValueError (FR-1 L116)。"""
    with pytest.raises(ValueError):
        ActionCatalog(on_invalid_slot=bad)


@pytest.mark.parametrize("bad", ["halt", "ERROR", "", "raise"])
def test_action_spec_rejects_invalid_on_invalid_slot(bad: str) -> None:
    """ActionSpec の宣言時に error / skip 以外を渡すと ValueError (FR-1 L116)。"""
    with pytest.raises(ValueError):
        _make_spec(on_invalid_slot=bad)


@pytest.mark.parametrize("value", ["error", "skip"])
def test_declaration_accepts_valid_on_invalid_slot(value: str) -> None:
    """error / skip は ActionCatalog / ActionSpec の双方で合法 (FR-1 L116)。"""
    assert ActionCatalog(on_invalid_slot=value).on_invalid_slot == value
    assert _make_spec(on_invalid_slot=value).on_invalid_slot == value


# ---------------------------------------------------------------------------
# SDK 隔離 (FR-1 L117 / NFR-1)
# ---------------------------------------------------------------------------


def test_action_spec_holds_action_agent_as_plain_str() -> None:
    """action_agent はエージェント名の str として保持される (FR-1 L117)。"""
    spec = _make_spec()
    assert type(spec.action_agent) is str
    assert spec.action_agent == "messenger"


def test_actions_module_source_has_no_sdk_import() -> None:
    """actions.py のソースに agents / openai / _adapters への import が現れない (FR-1 L117)。"""
    source = _ACTIONS_PATH.read_text(encoding="utf-8")
    assert "from agents" not in source
    assert "import agents" not in source
    assert "import openai" not in source
    assert "_adapters" not in source


def test_actions_module_import_does_not_load_sdk() -> None:
    """actions.py を単体 import しても agents / openai が sys.modules に載らない (FR-1 L117)。

    パッケージ窓口 (`oai_agentspec.__init__`) は SDK を載せるため、親パッケージを空の
    スタブへ差し替えたクリーンな子プロセスで当該モジュールのみを import して切り分ける。
    """
    probe = (
        "import importlib\n"
        "import sys\n"
        "import types\n"
        "from pathlib import Path\n"
        f"root = Path(r'{_SRC_DIR}') / 'oai_agentspec'\n"
        "for name, path in (\n"
        "    ('oai_agentspec', root),\n"
        "    ('oai_agentspec.runtime', root / 'runtime'),\n"
        "    ('oai_agentspec.runtime.intent', root / 'runtime' / 'intent'),\n"
        "):\n"
        "    stub = types.ModuleType(name)\n"
        "    stub.__path__ = [str(path)]\n"
        "    sys.modules[name] = stub\n"
        "mod = importlib.import_module('oai_agentspec.runtime.intent.actions')\n"
        "assert mod.ActionSpec is not None\n"
        "loaded = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m in ('agents', 'openai') or m.startswith(('agents.', 'openai.'))\n"
        ")\n"
        "print(','.join(loaded))\n"
    )
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
    loaded = [m for m in result.stdout.strip().split(",") if m]
    assert loaded == [], f"actions 単体 import で SDK がロードされました: {loaded}"


# ---------------------------------------------------------------------------
# param() の name は action_id と同型に _ 始まりを弾く (セキュリティレビュー指摘 #88-1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["_secret", "_", "_x", "_from_context"])
def test_param_rejects_underscore_prefixed_name(name: str) -> None:
    """_ 始まりの name は ValueError (指摘 #88-1)。

    `str.isidentifier()` は `_secret` に対して True を返すため現行の `_validate_name` を
    通過するが、生成モデルのフィールド名にすると pydantic が
    「Fields must not use names with leading underscores」で `NameError` を投げる。
    `ValueError` を契約する `param()` の Raises 節に対する契約違反であり、宣言時に落とす。
    `action_id` 側の 4 分岐 (`actions.py:210-220`) と同型に揃える。
    """
    with pytest.raises(ValueError):
        param(name, str)


@pytest.mark.parametrize("name", ["__module__", "__config__", "__base__", "__validators__"])
def test_param_rejects_dunder_name(name: str) -> None:
    """dunder の name も _ 始まりと同一経路で ValueError (指摘 #88-1)。

    これらは `create_model()` が予約する引数名であり、フィールドとして渡しても例外も警告も
    なく黙って消える (`__module__` を宣言したモデルはフィールド 0 件になる)。宣言した
    パラメータが痕跡なく失われる経路であるため、宣言時に落とす。
    """
    with pytest.raises(ValueError):
        param(name, str)


@pytest.mark.parametrize("name", ["class", "return", "None", "lambda"])
def test_param_rejects_python_keyword_name(name: str) -> None:
    """Python 予約語の name も ValueError (指摘 #88-1)。

    `str.isidentifier()` は予約語に対して True を返すが、フィールド名にすると
    `instance.class` が SyntaxError となり属性アクセスで到達できない。`_` 始まりや
    dunder と同じ「宣言が黙って失われる」系統であるため、宣言時に落とす。
    `action_id` 側の 4 分岐と同型に揃える。
    """
    with pytest.raises(ValueError):
        param(name, str)


@pytest.mark.parametrize("name", ["target", "seconds", "a1", "ターゲット"])
def test_param_still_accepts_ordinary_names(name: str) -> None:
    """_ 始まりでない識別子は従来どおり受理される (指摘 #88-1 の回帰防止)。"""
    assert param(name, str).name == name


# ---------------------------------------------------------------------------
# param() の annotation は str の前方参照を受理しない (セキュリティレビュー指摘 #88-2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("annotation", ["int", "str", "list[int]", "SomeUndefinedType"])
def test_param_rejects_str_annotation(annotation: str) -> None:
    """annotation に str を渡すと ValueError (指摘 #88-2)。

    pydantic は str の annotation を前方参照として `eval` するため、宣言層で受理すると
    任意式の実行経路になる。型そのものだけを受け取る契約を宣言時に pin する。
    """
    with pytest.raises(ValueError):
        param("target", annotation)


def test_param_does_not_evaluate_a_str_annotation() -> None:
    """str の annotation を渡しても副作用が起きない (指摘 #88-2)。

    拒否は「受け取ってから評価して型でないと判定する」形であってはならない。評価された
    場合にのみ環境変数が増えるプローブ式を渡し、増えていないことで未評価を検知する。
    テストが環境を汚さないよう前後で当該キーを取り除く。
    """
    key = "OAI_AGENTSPEC_PARAM_ANNOTATION_PROBE"
    probe = f"__import__('os').environ.setdefault({key!r}, '1')"
    os.environ.pop(key, None)
    try:
        with pytest.raises(ValueError):
            param("target", probe)
        assert key not in os.environ, "param() が str の annotation を評価しました"
    finally:
        os.environ.pop(key, None)


def test_param_still_accepts_real_types_as_annotation() -> None:
    """型オブジェクトの annotation は従来どおり受理される (指摘 #88-2 の回帰防止)。"""
    assert param("target", str).annotation is str
    assert param("seconds", int).annotation is int
    assert param("tags", list[str]).annotation == list[str]


# ---------------------------------------------------------------------------
# ActionCatalog の設定属性は読み取り専用 (セキュリティレビュー指摘 #88-3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("prompt", ("other/base",)),
        ("prompt_vars", {"tone": "casual"}),
        ("on_invalid_slot", "bogus"),
    ],
)
def test_action_catalog_settings_are_read_only(attribute: str, value: Any) -> None:
    """prompt / prompt_vars / on_invalid_slot への再代入は AttributeError (指摘 #88-3)。

    現状はいずれも plain な属性であるため、`catalog.on_invalid_slot = "bogus"` が
    `__init__` の値検証 (FR-1 L116) を迂回して不正値を予測段まで持ち越せる。既定の
    解決規則 (`resolve_on_invalid_slot`) が検証済みの 2 値を前提にしているため、宣言後に
    書き換わらないことを契約とする。
    """
    catalog = ActionCatalog()
    with pytest.raises(AttributeError):
        setattr(catalog, attribute, value)


def test_action_catalog_settings_keep_their_values_after_a_rejected_assignment() -> None:
    """再代入が弾かれた後も既定値は宣言時のまま (指摘 #88-3)。"""
    catalog = ActionCatalog(
        prompt=("common/base",),
        prompt_vars={"tone": "polite"},
        on_invalid_slot="error",
    )
    for attribute, value in (
        ("prompt", ("other/base",)),
        ("prompt_vars", {"tone": "casual"}),
        ("on_invalid_slot", "bogus"),
    ):
        with pytest.raises(AttributeError):
            setattr(catalog, attribute, value)
    assert catalog.prompt == ("common/base",)
    assert catalog.prompt_vars == {"tone": "polite"}
    assert catalog.on_invalid_slot == "error"


def test_action_catalog_remains_registerable_when_settings_are_read_only() -> None:
    """設定属性が読み取り専用でも register / names / get は従来どおり動く (指摘 #88-3)。

    ActionCatalog は plain な mutable クラス (FR-1 L106) であり、frozen 化するのではなく
    設定 3 件だけを読み取り専用にする。宣言簿としての可変性は維持する。
    """
    catalog = ActionCatalog()
    spec = _make_spec()
    catalog.register(spec)
    catalog.register(_make_spec(action_id="archive"))
    assert catalog.names() == ["archive", "send_message"]
    assert catalog.get("send_message") is spec
    assert not isinstance(catalog, BaseModel)


# ---------------------------------------------------------------------------
# ActionCatalog.prompt_vars は中身も書き換えられない (セキュリティレビュー指摘 #88-10)
# ---------------------------------------------------------------------------
#
# 指摘 #88-3 は属性の再代入 (`catalog.prompt_vars = {...}`) を塞いだが、property が返す
# 実体は plain な dict のままであり `catalog.prompt_vars["b"] = "2"` で中身を差し替えられる。
# prompt_vars は LLM へ渡るプロンプトへ展開される値であるため、宣言後に外から書き換わると
# 宣言簿を読んでも実際に送られる内容が分からなくなる。読み取り専用ビュー
# (`MappingProxyType`) を返し、書き込みを `TypeError` にする。


def test_action_catalog_prompt_vars_rejects_item_assignment() -> None:
    """prompt_vars の要素代入は TypeError (指摘 #88-10)。"""
    catalog = ActionCatalog(prompt_vars={"a": "1"})
    with pytest.raises(TypeError):
        catalog.prompt_vars["b"] = "2"


def test_action_catalog_prompt_vars_rejects_item_overwrite_and_deletion() -> None:
    """既存キーの上書きと削除も TypeError (指摘 #88-10)。"""
    catalog = ActionCatalog(prompt_vars={"tone": "polite"})
    with pytest.raises(TypeError):
        catalog.prompt_vars["tone"] = "casual"
    with pytest.raises(TypeError):
        del catalog.prompt_vars["tone"]


def test_action_catalog_prompt_vars_keeps_its_values_after_a_rejected_write() -> None:
    """書き込みが弾かれた後も prompt_vars は宣言時のまま (指摘 #88-10)。"""
    catalog = ActionCatalog(prompt_vars={"tone": "polite"})
    with pytest.raises(TypeError):
        catalog.prompt_vars["tone"] = "casual"
    with pytest.raises(TypeError):
        catalog.prompt_vars["locale"] = "ja"
    assert catalog.prompt_vars == {"tone": "polite"}


def test_action_catalog_prompt_vars_is_still_readable_as_a_mapping() -> None:
    """読み取り専用にしても Mapping としての読み取りは従来どおり (指摘 #88-10 の回帰防止)。"""
    catalog = ActionCatalog(prompt_vars={"tone": "polite", "locale": "ja"})
    assert catalog.prompt_vars["tone"] == "polite"
    assert sorted(catalog.prompt_vars) == ["locale", "tone"]
    assert len(catalog.prompt_vars) == 2
    assert dict(catalog.prompt_vars) == {"tone": "polite", "locale": "ja"}


def test_resolve_prompt_vars_still_returns_a_plain_mutable_dict() -> None:
    """resolve_prompt_vars は従来どおり書き換え可能な dict を返す (指摘 #88-10 の回帰防止)。

    読み取り専用にするのはカタログが保持する既定であり、マージ結果は呼び出し側が
    自由に扱える新しい dict である。
    """
    catalog = ActionCatalog(prompt_vars={"tone": "polite"})
    resolved = resolve_prompt_vars(catalog, _make_spec())
    assert type(resolved) is dict
    resolved["extra"] = "ok"
    assert catalog.prompt_vars == {"tone": "polite"}


def test_resolve_prompt_vars_still_merges_with_action_overriding_when_read_only() -> None:
    """prompt_vars が読み取り専用でもマージ規則は変わらない (指摘 #88-10 の回帰防止)。"""
    catalog = ActionCatalog(prompt_vars={"tone": "polite", "locale": "ja"})
    spec = _make_spec(prompt_vars={"tone": "casual"})
    assert resolve_prompt_vars(catalog, spec) == {"tone": "casual", "locale": "ja"}


def test_action_catalog_still_copies_the_given_prompt_vars_when_read_only() -> None:
    """読み取り専用ビューでも渡した dict の後からの変更に影響されない (指摘 #88-10 の回帰防止)。

    `MappingProxyType(source)` をそのまま返すと元 dict への変更が透けるため、
    コピーの上に被せる必要がある。
    """
    source = {"tone": "polite"}
    catalog = ActionCatalog(prompt_vars=source)
    source["tone"] = "casual"
    source["locale"] = "ja"
    assert catalog.prompt_vars == {"tone": "polite"}


def test_action_catalog_remains_registerable_when_prompt_vars_is_read_only() -> None:
    """prompt_vars が読み取り専用でも register / names / get は従来どおり動く (指摘 #88-10)。"""
    catalog = ActionCatalog(prompt_vars={"tone": "polite"})
    spec = _make_spec()
    catalog.register(spec)
    catalog.register(_make_spec(action_id="archive"))
    assert catalog.names() == ["archive", "send_message"]
    assert catalog.get("send_message") is spec
    with pytest.raises(ValueError):
        catalog.register(_make_spec())


# ---------------------------------------------------------------------------
# param() の annotation はネストした前方参照も受理しない (指摘 #88-11)
# ---------------------------------------------------------------------------
#
# `_validate_annotation` は annotation 全体が str かどうかしか見ないため、`list["..."]` /
# `Optional["..."]` / `ForwardRef("...")` のように 1 段でも包むと素通しする。宣言時は
# 保持するだけで評価されないが、`parameters_model()` が pydantic へ渡した時点で前方参照
# として eval される。宣言層で受理してしまうと、危険な宣言が実行段まで運ばれる。
# 生成層 (`_models.build_frozen_model`) にも同じ再帰検査があり多層防御になる。

#: 評価されると `_evil` の NameError になるプローブ式。評価しない限り無害な str である。
_FORWARD_REF_PROBE = "_evil()"


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(list[_FORWARD_REF_PROBE], id="list"),
        pytest.param(ForwardRef(_FORWARD_REF_PROBE), id="forward-ref"),
        pytest.param(Optional[_FORWARD_REF_PROBE], id="optional"),  # noqa: UP045
        pytest.param(dict[str, _FORWARD_REF_PROBE], id="dict-value"),
        pytest.param(tuple[_FORWARD_REF_PROBE, ...], id="tuple-ellipsis"),
        pytest.param(list[list[_FORWARD_REF_PROBE]], id="list-of-list"),
    ],
)
def test_param_rejects_a_nested_forward_ref_annotation(annotation: Any) -> None:
    """annotation の内側に前方参照が 1 つでもあれば ValueError (指摘 #88-11)。

    最上位の str しか見ない検査は 1 段包むだけで迂回でき、pydantic が任意式として eval
    する宣言をそのまま宣言簿へ載せられてしまう。
    """
    with pytest.raises(ValueError):
        param("target", annotation)


def test_param_does_not_evaluate_a_nested_forward_ref_annotation() -> None:
    """ネストした前方参照を渡しても副作用が起きない (指摘 #88-11)。

    拒否は「受け取ってから評価して型でないと判定する」形であってはならない。評価される
    たびに 1 ずつ増えるカウンタ式を渡し、増分が 0 であることで未評価を検知する。
    テストが環境を汚さないよう前後で当該キーを取り除く。
    """
    key = "OAI_AGENTSPEC_PARAM_NESTED_ANNOTATION_PROBE"
    expression = (
        "__import__('os').environ.__setitem__("
        f"{key!r}, str(int(__import__('os').environ.get({key!r}, '0')) + 1))"
    )
    os.environ.pop(key, None)
    try:
        with pytest.raises(ValueError):
            param("target", list[expression])
        evaluations = int(os.environ.get(key, "0"))
        assert evaluations == 0, f"param() がネストした前方参照を {evaluations} 回評価しました"
    finally:
        os.environ.pop(key, None)


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(list[str], id="list-str"),
        pytest.param(dict[str, int], id="dict"),
        pytest.param(int | None, id="optional-int"),
        pytest.param(Literal["a", "b"], id="literal"),
        pytest.param(Literal["a"] | None, id="literal-optional"),
        pytest.param(list[Literal["a", "b"]], id="list-literal"),
        pytest.param(Annotated[int, "note"], id="annotated-str-metadata"),
        pytest.param(Annotated[list[str], "note"], id="annotated-list-metadata"),
    ],
)
def test_param_still_accepts_nested_annotations_without_forward_refs(annotation: Any) -> None:
    """前方参照を含まない合成 annotation は従来どおり受理される (指摘 #88-11 の回帰防止)。

    再帰検査が Literal の値や Annotated のメタデータを前方参照と誤認すると、正当な宣言が
    宣言時に落ちる。誤検知しないことを拒否側と対で pin する。
    """
    assert param("target", annotation).annotation == annotation


# ---------------------------------------------------------------------------
# ActionSpec / ActionPlanner の prompt_vars も中身を書き換えられない (指摘 #88-W2)
# ---------------------------------------------------------------------------
#
# 指摘 #88-10 は `ActionCatalog.prompt_vars` を読み取り専用ビューにしたが、同じ値を運ぶ
# pydantic 側の 2 フィールド (`ActionSpec.prompt_vars` / `ActionPlanner.prompt_vars`) は
# `Mapping[str, str]` を素の dict として保持するため、`model_config = {"frozen": True}` でも
# 中身が書き換わる (frozen は属性の再束縛のみを禁じる)。`resolve_prompt_vars` は plan 時に
# `spec.prompt_vars` を読むため、起動時検証 (検査 6 / 7) を通過した後に run context のパスを
# 機微な属性へ向け直せる。解決値は LLM へ渡るプロンプトへ展開される。


def test_action_spec_prompt_vars_rejects_item_assignment() -> None:
    """ActionSpec.prompt_vars への要素追加は TypeError (指摘 #88-W2)。"""
    spec = _make_spec(prompt_vars={"tone": "polite"})
    with pytest.raises(TypeError):
        spec.prompt_vars["secret"] = "credentials.api_key"


def test_action_spec_prompt_vars_rejects_item_overwrite_and_deletion() -> None:
    """既存キーの上書きと削除も TypeError (指摘 #88-W2)。

    起動時検証を通過したパスを、実際にプロンプトへ展開される直前に別のパスへ差し替えられる
    経路であるため、追加だけでなく上書き・削除も塞ぐ。
    """
    spec = _make_spec(prompt_vars={"tone": "tenant.tone"})
    with pytest.raises(TypeError):
        spec.prompt_vars["tone"] = "credentials.api_key"
    with pytest.raises(TypeError):
        del spec.prompt_vars["tone"]


def test_action_spec_prompt_vars_keeps_its_values_after_a_rejected_write() -> None:
    """書き込みが弾かれた後も ActionSpec.prompt_vars は宣言時のまま (指摘 #88-W2)。"""
    spec = _make_spec(prompt_vars={"tone": "tenant.tone"})
    with pytest.raises(TypeError):
        spec.prompt_vars["tone"] = "credentials.api_key"
    with pytest.raises(TypeError):
        spec.prompt_vars["extra"] = "credentials.api_key"
    assert dict(spec.prompt_vars) == {"tone": "tenant.tone"}


def test_action_spec_prompt_vars_is_still_readable_as_a_mapping() -> None:
    """読み取り専用にしても Mapping としての読み取りは従来どおり (指摘 #88-W2 の回帰防止)。"""
    spec = _make_spec(prompt_vars={"tone": "tenant.tone", "locale": "tenant.locale"})
    assert spec.prompt_vars["tone"] == "tenant.tone"
    assert sorted(spec.prompt_vars) == ["locale", "tone"]
    assert len(spec.prompt_vars) == 2
    assert dict(spec.prompt_vars) == {"tone": "tenant.tone", "locale": "tenant.locale"}


def test_action_spec_prompt_vars_repr_shows_its_contents() -> None:
    """読み取り専用ビューの repr は中身が読める形のまま (指摘 #88-W2 の回帰防止)。

    `collections.abc.Mapping` は `__repr__` を提供しないため、実装が省くと
    `<..._ReadOnlyMapping object at 0x...>` になり、公開型 (`ActionSpec` /
    `ExecutableIntent`) の repr から宣言内容が読めなくなる。差し替え前の
    `mappingproxy({...})` と同等の可読性を pin する。
    """
    spec = _make_spec(prompt_vars={"tone": "tenant.tone"})
    rendered = repr(spec.prompt_vars)
    assert rendered == "_ReadOnlyMapping({'tone': 'tenant.tone'})"
    assert rendered in repr(spec)


def test_action_spec_still_copies_the_given_prompt_vars() -> None:
    """宣言時に渡した dict を後から書き換えても中身が透けない (指摘 #88-W2)。

    読み取り専用ビューを被せるだけでは、呼び出し側が握ったままの元 dict への変更が透ける。
    コピーの上にビューを被せる必要がある。
    """
    source = {"tone": "tenant.tone"}
    spec = _make_spec(prompt_vars=source)
    source["tone"] = "credentials.api_key"
    source["extra"] = "credentials.api_key"
    assert dict(spec.prompt_vars) == {"tone": "tenant.tone"}


def test_action_spec_prompt_vars_default_is_still_empty_and_read_only() -> None:
    """既定の prompt_vars も空のまま書き換えられない (指摘 #88-W2)。"""
    spec = _make_spec()
    assert dict(spec.prompt_vars) == {}
    with pytest.raises(TypeError):
        spec.prompt_vars["secret"] = "credentials.api_key"


def test_action_planner_prompt_vars_rejects_item_assignment() -> None:
    """bind() が返す ActionPlanner.prompt_vars も書き換えられない (指摘 #88-W2)。

    `registry` は不透明値として保持されるだけなので、bind には素のオブジェクトで足りる。
    """
    catalog = ActionCatalog(prompt_vars={"tone": "tenant.tone"})
    catalog.register(_make_spec())
    planner = catalog.bind(registry=object())
    with pytest.raises(TypeError):
        planner.prompt_vars["tone"] = "credentials.api_key"
    assert dict(planner.prompt_vars) == {"tone": "tenant.tone"}


def test_action_planner_prompt_vars_rejects_item_deletion() -> None:
    """ActionPlanner.prompt_vars の削除も TypeError (指摘 #88-W2)。"""
    catalog = ActionCatalog(prompt_vars={"tone": "tenant.tone"})
    planner = catalog.bind(registry=object())
    with pytest.raises(TypeError):
        del planner.prompt_vars["tone"]


def test_action_planner_still_copies_the_given_prompt_vars() -> None:
    """直接構築した ActionPlanner でも渡した dict の後からの変更が透けない (指摘 #88-W2)。"""
    source = {"tone": "tenant.tone"}
    planner = ActionPlanner(
        specs=(),
        prompt=(),
        prompt_vars=source,
        on_invalid_slot="skip",
        registry=object(),
    )
    source["tone"] = "credentials.api_key"
    assert dict(planner.prompt_vars) == {"tone": "tenant.tone"}


def test_action_planner_prompt_vars_is_still_readable_as_a_mapping() -> None:
    """ActionPlanner.prompt_vars の読み取りは従来どおり (指摘 #88-W2 の回帰防止)。"""
    catalog = ActionCatalog(prompt_vars={"tone": "tenant.tone", "locale": "tenant.locale"})
    planner = catalog.bind(registry=object())
    assert planner.prompt_vars["locale"] == "tenant.locale"
    assert sorted(planner.prompt_vars) == ["locale", "tone"]
    assert dict(planner.prompt_vars) == {"tone": "tenant.tone", "locale": "tenant.locale"}


def test_resolve_prompt_vars_still_merges_when_spec_prompt_vars_is_read_only() -> None:
    """spec 側が読み取り専用でもマージ規則と戻り値の可変性は変わらない (指摘 #88-W2 の回帰防止)。"""
    catalog = ActionCatalog(prompt_vars={"tone": "tenant.tone", "locale": "tenant.locale"})
    spec = _make_spec(prompt_vars={"tone": "tenant.style"})
    resolved = resolve_prompt_vars(catalog, spec)
    assert resolved == {"tone": "tenant.style", "locale": "tenant.locale"}
    assert type(resolved) is dict
    resolved["extra"] = "ok"
    assert dict(spec.prompt_vars) == {"tone": "tenant.style"}


def test_action_spec_is_still_serializable_when_prompt_vars_is_read_only() -> None:
    """読み取り専用ビューでも model_dump / model_json_schema が成立する (指摘 #88-W2 の回帰防止)。

    `MappingProxyType` をそのままフィールドに置くと直列化が落ちうるため、既存の直列化契約
    (FR-1 L105) が保たれることを対で pin する。`model_dump_json` は `annotation` が型
    オブジェクトを保持するため元から契約外であり、ここでも検査しない。
    """
    spec = _make_spec(prompt_vars={"tone": "tenant.tone"})
    assert spec.model_dump()["prompt_vars"] == {"tone": "tenant.tone"}
    assert isinstance(spec.model_json_schema(), dict)


# ---------------------------------------------------------------------------
# 読み取り専用にした prompt_vars でも複製・永続化できる (レビュー 2 巡目・指摘 #88-W2 の退行)
# ---------------------------------------------------------------------------
#
# 指摘 #88-W2 / #88-10 の修正で prompt_vars を `MappingProxyType` へ正規化したところ、
# `mappingproxy` が pickle 不可であるため `copy.deepcopy` / `model_copy(deep=True)` /
# `pickle` の 3 経路が `TypeError` で落ちるようになった (修正前は素の dict で 3 経路とも
# 成立していた)。宣言簿はプロセスを跨いで運ばれうるため、読み取り専用性と複製可能性は
# 両立させる。読み取り専用のまま複製できることを対で pin し、「pickle を通すために素の
# dict へ戻す」修正で #88-W2 が無言で巻き戻ることを防ぐ。

#: 複製・永続化の 3 経路。どれか 1 つでも落ちれば宣言をプロセス外へ運べない。
_CLONE_ROUTES: list[Any] = [
    pytest.param(copy.deepcopy, id="deepcopy"),
    pytest.param(lambda obj: obj.model_copy(deep=True), id="model-copy-deep"),
    pytest.param(lambda obj: pickle.loads(pickle.dumps(obj)), id="pickle"),
]


def _assert_rejects_item_assignment(mapping: Mapping[str, str]) -> None:
    """Mapping への要素代入が TypeError で弾かれることを確かめる。

    Args:
        mapping: 読み取り専用であることを期待する Mapping。
    """
    with pytest.raises(TypeError) as excinfo:
        mapping["injected"] = "credentials.api_key"  # type: ignore[index]
    assert type(excinfo.value) is TypeError


@pytest.mark.parametrize("clone", _CLONE_ROUTES)
def test_action_spec_survives_cloning_with_its_prompt_vars(clone: Callable[[Any], Any]) -> None:
    """ActionSpec は 3 経路で複製でき prompt_vars の値と実体の別が保たれる (#88-W2 の退行)。"""
    spec = _make_spec(prompt_vars={"tone": "tenant.tone", "locale": "tenant.locale"})
    restored = clone(spec)
    assert dict(restored.prompt_vars) == {"tone": "tenant.tone", "locale": "tenant.locale"}
    assert restored.action_id == spec.action_id
    assert restored.prompt_vars is not spec.prompt_vars


@pytest.mark.parametrize("clone", _CLONE_ROUTES)
def test_action_spec_prompt_vars_stay_read_only_after_cloning(clone: Callable[[Any], Any]) -> None:
    """複製後の ActionSpec.prompt_vars も読み取り専用のまま (#88-W2 の退行)。

    複製を通すために素の dict へ戻す修正だと値の一致だけは緑になるため、書き込みが
    `TypeError` であることを複製の先でも確かめる。
    """
    spec = _make_spec(prompt_vars={"tone": "tenant.tone"})
    restored = clone(spec)
    _assert_rejects_item_assignment(restored.prompt_vars)
    assert dict(restored.prompt_vars) == {"tone": "tenant.tone"}


@pytest.mark.parametrize("clone", _CLONE_ROUTES)
def test_action_planner_survives_cloning_with_its_prompt_vars(clone: Callable[[Any], Any]) -> None:
    """bind() が返す ActionPlanner も 3 経路で複製できる (#88-W2 の退行)。

    `bind()` が返す planner は内部に再構築したカタログを抱えるため、pydantic フィールド
    側だけでなくカタログが保持する既定も複製可能である必要がある。
    """
    catalog = ActionCatalog(prompt_vars={"tone": "tenant.tone"})
    catalog.register(_make_spec())
    planner = catalog.bind(registry=object())
    restored = clone(planner)
    assert dict(restored.prompt_vars) == {"tone": "tenant.tone"}
    assert restored.specs[0].action_id == "send_message"
    assert restored.prompt_vars is not planner.prompt_vars


@pytest.mark.parametrize("clone", _CLONE_ROUTES)
def test_action_planner_prompt_vars_stay_read_only_after_cloning(
    clone: Callable[[Any], Any],
) -> None:
    """複製後の ActionPlanner.prompt_vars も読み取り専用のまま (#88-W2 の退行)。"""
    planner = ActionPlanner(
        specs=(),
        prompt=(),
        prompt_vars={"tone": "tenant.tone"},
        on_invalid_slot="skip",
        registry=object(),
    )
    restored = clone(planner)
    _assert_rejects_item_assignment(restored.prompt_vars)
    assert dict(restored.prompt_vars) == {"tone": "tenant.tone"}


# ---------------------------------------------------------------------------
# ActionCatalog(prompt=...) は bare str を宣言時に弾く (レビュー指摘 #88-R1)
# ---------------------------------------------------------------------------
#
# `self._prompt = tuple(prompt)` は bare str を 1 文字ずつ分解し、`prompt="billing"` を
# `("b", "i", "l", "l", "i", "n", "g")` として黙って通す。同じ `prompt` 契約を持つ
# `ActionSpec(prompt="billing")` は `ValidationError` で弾かれるため、カタログ側だけが
# silent misconfiguration になる。分解されたセグメント名は起動時検証（検査 3）で
# 「未解決セグメント」として初めて表面化するか、あるいは空文字なら何事もなく通る。


@pytest.mark.parametrize(
    "bad",
    ["billing", "common/base", "a", ""],
    ids=["word", "segment-path", "single-char", "empty"],
)
def test_action_catalog_rejects_a_bare_str_prompt(bad: str) -> None:
    """prompt に bare str を渡すと宣言時に ValueError (指摘 #88-R1)。

    1 文字ずつ分解された結果は「セグメント名の列」として型的には成立してしまうため、
    分解される前の宣言時に落とす。単一文字・空文字も同じ誤りの一部であり、通してはならない。
    """
    with pytest.raises(ValueError, match="prompt") as excinfo:
        ActionCatalog(prompt=bad)  # type: ignore[arg-type]
    assert type(excinfo.value) is ValueError


def test_action_catalog_bare_str_prompt_error_names_the_offending_value() -> None:
    """拒否メッセージに渡された値が現れる (指摘 #88-R1)。

    `on_invalid_slot` の既存メッセージと同じく、どの宣言が落ちたかを読み取れる形にする
    （カタログは 1 プロセスに複数ありうるため、値が出ないと宣言の特定に手がかりが無い）。
    """
    with pytest.raises(ValueError, match="billing"):
        ActionCatalog(prompt="billing")  # type: ignore[arg-type]


def test_action_catalog_accepts_an_empty_prompt() -> None:
    """既定（省略）と明示の空 tuple はいずれも成立する (指摘 #88-R1 の誤検知防止)。"""
    assert ActionCatalog().prompt == ()
    assert ActionCatalog(prompt=()).prompt == ()


def test_action_catalog_accepts_a_tuple_prompt_and_keeps_the_segments() -> None:
    """tuple の prompt は宣言順のままセグメント名として保持される (指摘 #88-R1 の誤検知防止)。"""
    catalog = ActionCatalog(prompt=("seg1", "seg2"))
    assert catalog.prompt == ("seg1", "seg2")


def test_action_catalog_accepts_a_list_prompt_and_normalizes_it_to_a_tuple() -> None:
    """list の prompt は tuple へ正規化され中身が保たれる (指摘 #88-R1 の誤検知防止)。

    セグメント名の列であれば列の型は問わない。`str` だけを特別扱いして落とす。
    """
    catalog = ActionCatalog(prompt=["seg"])
    assert catalog.prompt == ("seg",)


def test_action_catalog_bare_str_prompt_rejection_does_not_break_other_defaults() -> None:
    """prompt の検証が prompt_vars / on_invalid_slot の既定解決を巻き込まない (指摘 #88-R1)。"""
    catalog = ActionCatalog(
        prompt=["common/base"], prompt_vars={"tone": "polite"}, on_invalid_slot="error"
    )
    assert catalog.prompt == ("common/base",)
    assert dict(catalog.prompt_vars) == {"tone": "polite"}
    assert catalog.on_invalid_slot == "error"
