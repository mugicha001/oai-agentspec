"""L1: 意図推定の型 (`ConfidenceLevel` / `IntentPolicy` / `IntentQuery` ほか) の純検証。

pydantic BaseModel 群の frozen 性・バリデーション・デフォルト値・`render_prompt` 出力・
`_CONFIDENCE_LEVEL_DESCRIPTION` の schema 反映を検証する。外部依存 (agents / openai) なし。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from oai_agentspec.runtime.intent.types import (
    ConfidenceLevel,
    ConsistencyReport,
    IntentCandidate,
    IntentCategory,
    IntentContext,
    IntentPolicy,
    IntentPrediction,
    IntentQuery,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ConfidenceLevel
# ---------------------------------------------------------------------------


def test_confidence_level_has_five_values() -> None:
    """ConfidenceLevel は 5 値 (certain / high / medium / low / speculative) を持つ。"""
    values = {level.value for level in ConfidenceLevel}
    assert values == {"certain", "high", "medium", "low", "speculative"}


def test_confidence_level_is_str_enum() -> None:
    """str 継承のため文字列と等価比較できる。"""
    assert ConfidenceLevel.CERTAIN == "certain"
    assert ConfidenceLevel.HIGH == "high"
    assert ConfidenceLevel.MEDIUM == "medium"
    assert ConfidenceLevel.LOW == "low"
    assert ConfidenceLevel.SPECULATIVE == "speculative"
    assert isinstance(ConfidenceLevel.CERTAIN, str)


def test_confidence_level_ordering() -> None:
    """宣言順序が CERTAIN → HIGH → MEDIUM → LOW → SPECULATIVE。"""
    assert list(ConfidenceLevel) == [
        ConfidenceLevel.CERTAIN,
        ConfidenceLevel.HIGH,
        ConfidenceLevel.MEDIUM,
        ConfidenceLevel.LOW,
        ConfidenceLevel.SPECULATIVE,
    ]


# ---------------------------------------------------------------------------
# IntentCategory / IntentPolicy
# ---------------------------------------------------------------------------


def _make_category(name: str = "greet", description: str = "挨拶") -> IntentCategory:
    return IntentCategory(name=name, description=description)


def test_intent_category_construction_and_frozen() -> None:
    """IntentCategory は生成でき、frozen (再代入は ValidationError)。"""
    cat = _make_category(name="greet", description="挨拶")
    assert cat.name == "greet"
    assert cat.description == "挨拶"
    with pytest.raises(ValidationError):
        cat.name = "other"  # type: ignore[misc]


def test_intent_policy_rejects_empty_categories() -> None:
    """categories 空 tuple は ValidationError (非空必須)。"""
    with pytest.raises(ValidationError):
        IntentPolicy(categories=())


def test_intent_policy_rejects_duplicate_category_names() -> None:
    """重複 name の categories は ValidationError。"""
    with pytest.raises(ValidationError):
        IntentPolicy(
            categories=(
                _make_category(name="a", description="d1"),
                _make_category(name="a", description="d2"),
            )
        )


def test_intent_policy_defaults() -> None:
    """max_candidates=3 / extra_instructions='' / include_rationale_in_prompt=False がデフォルト値。

    require_rationale は非存在。
    """
    policy = IntentPolicy(
        categories=(
            _make_category(name="a", description="d1"),
            _make_category(name="b", description="d2"),
        )
    )
    assert policy.max_candidates == 3
    assert policy.extra_instructions == ""
    assert policy.include_rationale_in_prompt is False
    assert len(policy.categories) == 2
    # require_rationale フィールドは存在しない
    assert "require_rationale" not in type(policy).model_fields


def test_intent_policy_rejects_require_rationale_kwarg() -> None:
    """require_rationale フィールドは削除済み。指定すると ValidationError。"""
    with pytest.raises(ValidationError):
        IntentPolicy(
            categories=(_make_category(),),
            require_rationale=True,  # type: ignore[call-arg]
        )


def test_intent_policy_accepts_extra_instructions() -> None:
    """extra_instructions を指定して生成できる。"""
    policy = IntentPolicy(
        categories=(_make_category(),),
        extra_instructions="Focus on X",
    )
    assert policy.extra_instructions == "Focus on X"


def test_intent_policy_render_prompt_contains_expected_content() -> None:
    """render_prompt は categories / schema / 5 レベル / max_candidates を含む。"""
    policy = IntentPolicy(
        categories=(
            _make_category(name="greet", description="挨拶を検出"),
            _make_category(name="ask", description="質問を検出"),
        ),
        max_candidates=5,
    )
    prompt = policy.render_prompt()
    assert isinstance(prompt, str)
    # 各 category の name / description
    assert "greet" in prompt
    assert "挨拶を検出" in prompt
    assert "ask" in prompt
    assert "質問を検出" in prompt
    # 5 レベル
    for level in ("certain", "high", "medium", "low", "speculative"):
        assert level in prompt
    # max_candidates の反映
    assert "5" in prompt


def test_intent_policy_render_prompt_omits_require_rationale_constraint() -> None:
    """render_prompt 出力に「rationale は必須」制約行は含まれない。"""
    policy = IntentPolicy(categories=(_make_category(),))
    prompt = policy.render_prompt()
    assert "rationale は必須" not in prompt


def test_intent_policy_render_prompt_places_extra_instructions_at_head() -> None:
    """extra_instructions は render_prompt 出力の先頭 200 バイトに配置される。"""
    policy = IntentPolicy(
        categories=(_make_category(),),
        extra_instructions="Focus on X",
    )
    prompt = policy.render_prompt()
    assert "Focus on X" in prompt[:200]


def test_intent_policy_render_prompt_empty_extra_instructions_no_change() -> None:
    """extra_instructions='' の場合、追加行なしで既定と同一出力。"""
    categories = (_make_category(),)
    default_prompt = IntentPolicy(categories=categories).render_prompt()
    empty_extra_prompt = IntentPolicy(categories=categories, extra_instructions="").render_prompt()
    assert default_prompt == empty_extra_prompt


def test_intent_policy_render_prompt_whitespace_only_extra_instructions_no_change() -> None:
    """空白のみの extra_instructions は空扱いで、既定と同一出力（先頭に空行を挿入しない）。"""
    categories = (_make_category(),)
    default_prompt = IntentPolicy(categories=categories).render_prompt()
    for ws in ("   ", "\n", " \n\n "):
        ws_prompt = IntentPolicy(categories=categories, extra_instructions=ws).render_prompt()
        assert ws_prompt == default_prompt, f"extra_instructions={ws!r} で出力が変わった"


def test_intent_policy_render_prompt_rationale_marked_optional_in_example() -> None:
    """include_rationale_in_prompt=True の出力例 JSON では rationale フィールドを含む。"""
    policy = IntentPolicy(categories=(_make_category(),), include_rationale_in_prompt=True)
    prompt = policy.render_prompt()
    assert '"rationale"' in prompt
    assert "<理由>" in prompt


def test_include_rationale_in_prompt_default_false() -> None:
    """include_rationale_in_prompt の既定値は False。"""
    policy = IntentPolicy(categories=(_make_category(),))
    assert policy.include_rationale_in_prompt is False


def test_render_prompt_default_omits_rationale_from_output_example() -> None:
    """既定 (include_rationale_in_prompt=False) では出力例に rationale フィールドが載らない。"""
    policy = IntentPolicy(categories=(_make_category(),))
    prompt = policy.render_prompt()
    assert '"rationale"' not in prompt


def test_render_prompt_with_include_rationale_true_shows_rationale() -> None:
    """include_rationale_in_prompt=True では出力例に rationale フィールドが `<理由>` 表記で載る。"""
    policy = IntentPolicy(categories=(_make_category(),), include_rationale_in_prompt=True)
    prompt = policy.render_prompt()
    assert '"rationale"' in prompt
    assert "<理由>" in prompt
    assert "<理由 (任意)>" not in prompt


def test_render_prompt_with_include_rationale_true_field_order() -> None:
    """include_rationale_in_prompt=True の出力例フィールド並びは text -> level -> rationale。"""
    policy = IntentPolicy(categories=(_make_category(),), include_rationale_in_prompt=True)
    prompt = policy.render_prompt()
    example_line = next(
        line for line in prompt.splitlines() if '"text":' in line and '"level":' in line
    )
    assert example_line.index('"text"') < example_line.index('"level"')
    assert example_line.index('"level"') < example_line.index('"rationale"')


def test_include_rationale_in_prompt_rejects_non_bool() -> None:
    """include_rationale_in_prompt に bool へ変換不能な値を渡すと ValidationError。

    pydantic の bool 検証は "yes"/"1" 等の一部文字列は lax モードで bool 変換
    (coerce) されるため、変換不能な list を渡して弾かれることを確認する。
    """
    with pytest.raises(ValidationError):
        IntentPolicy(categories=(_make_category(),), include_rationale_in_prompt=[1, 2])


def test_intent_policy_rejects_zero_max_candidates() -> None:
    """max_candidates=0 は ge=1 制約に反し ValidationError。"""
    with pytest.raises(ValidationError):
        IntentPolicy(categories=(_make_category(),), max_candidates=0)


def test_intent_policy_rejects_negative_max_candidates() -> None:
    """max_candidates=-1 は ge=1 制約に反し ValidationError。"""
    with pytest.raises(ValidationError):
        IntentPolicy(categories=(_make_category(),), max_candidates=-1)


def test_intent_policy_accepts_max_candidates_one() -> None:
    """max_candidates=1 は下限値として正常に受け入れられる。"""
    policy = IntentPolicy(categories=(_make_category(),), max_candidates=1)
    assert policy.max_candidates == 1


def test_intent_policy_is_frozen() -> None:
    """IntentPolicy は frozen (再代入は ValidationError)。"""
    policy = IntentPolicy(categories=(_make_category(),))
    with pytest.raises(ValidationError):
        policy.max_candidates = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# IntentQuery / IntentContext
# ---------------------------------------------------------------------------


def test_intent_query_construction_defaults() -> None:
    """IntentQuery は utterance のみ必須。history / run_context は None デフォルト。"""
    q = IntentQuery(utterance="hello")
    assert q.utterance == "hello"
    assert q.history is None
    assert q.run_context is None


def test_intent_query_utterance_defaults_to_empty() -> None:
    """utterance 未指定でも正常生成でき、デフォルトは空文字（履歴のみ分類対応）。"""
    q = IntentQuery()
    assert q.utterance == ""
    assert q.history is None
    assert q.run_context is None


def test_intent_query_history_only_construction() -> None:
    """history のみ指定（utterance 省略）で正常生成できる。"""
    history = object()
    q = IntentQuery(history=history)
    assert q.utterance == ""
    assert q.history is history


def test_intent_context_construction_defaults() -> None:
    """IntentContext は utterance のみ必須。history_items は空 tuple デフォルト。"""
    ctx = IntentContext(utterance="hi")
    assert ctx.utterance == "hi"
    assert ctx.history_items == ()
    assert ctx.run_context is None


def test_intent_context_accepts_history_items() -> None:
    """history_items に Mapping tuple を渡して生成できる。"""
    items = ({"role": "user", "content": "a"},)
    ctx = IntentContext(utterance="hi", history_items=items)
    assert ctx.utterance == "hi"
    assert ctx.history_items == items


def test_intent_context_history_text_field_removed() -> None:
    """history_text フィールドは削除済み。属性アクセスは AttributeError。"""
    ctx = IntentContext(utterance="hi")
    with pytest.raises(AttributeError):
        _ = ctx.history_text  # type: ignore[attr-defined]
    assert "history_text" not in type(ctx).model_fields


def test_intent_context_history_items_is_tuple_and_frozen() -> None:
    """history_items は tuple 型で pydantic frozen (再代入不可)。"""
    ctx = IntentContext(
        utterance="hi",
        history_items=({"role": "user", "content": "a"},),
    )
    assert isinstance(ctx.history_items, tuple)
    with pytest.raises(ValidationError):
        ctx.history_items = ()  # type: ignore[misc]


def test_intent_query_is_frozen() -> None:
    """IntentQuery は frozen。"""
    q = IntentQuery(utterance="hi")
    with pytest.raises(ValidationError):
        q.utterance = "changed"  # type: ignore[misc]


def test_intent_context_is_frozen() -> None:
    """IntentContext は frozen。"""
    ctx = IntentContext(utterance="x")
    with pytest.raises(ValidationError):
        ctx.utterance = "z"  # type: ignore[misc]


def test_intent_query_and_context_accept_type_parameter() -> None:
    """Generic[TContext] として型パラメータを受け付ける (実行時エラーがない)。"""
    _ = IntentQuery[dict]
    _ = IntentContext[dict]

    class _MyContext:
        pass

    _ = IntentQuery[_MyContext]
    _ = IntentContext[_MyContext]
    q: IntentQuery[dict] = IntentQuery(utterance="hi", run_context={"key": "v"})
    assert q.run_context == {"key": "v"}


# ---------------------------------------------------------------------------
# IntentCandidate / ConsistencyReport / IntentPrediction
# ---------------------------------------------------------------------------


def test_intent_candidate_construction_defaults() -> None:
    """IntentCandidate は text / level 必須・rationale は None デフォルト。"""
    c = IntentCandidate(text="greet", level=ConfidenceLevel.HIGH)
    assert c.text == "greet"
    assert c.level is ConfidenceLevel.HIGH
    assert c.rationale is None


def test_intent_candidate_is_frozen() -> None:
    """IntentCandidate は frozen。"""
    c = IntentCandidate(text="x", level=ConfidenceLevel.LOW)
    with pytest.raises(ValidationError):
        c.text = "y"  # type: ignore[misc]


def test_consistency_report_defaults_empty_tuples() -> None:
    """ConsistencyReport は全空 tuple デフォルトで成功する。"""
    r = ConsistencyReport()
    assert r.conflicts == ()
    assert r.stale_context == ()
    assert r.over_inference == ()


def test_consistency_report_is_frozen() -> None:
    """ConsistencyReport は frozen。"""
    r = ConsistencyReport()
    with pytest.raises(ValidationError):
        r.conflicts = ("x",)  # type: ignore[misc]


def test_intent_prediction_construction_defaults() -> None:
    """IntentPrediction は candidates 必須・report / metadata は None デフォルト。空 tuple 可。"""
    p = IntentPrediction(candidates=())
    assert p.candidates == ()
    assert p.report is None
    assert p.metadata is None


def test_intent_prediction_with_candidates() -> None:
    """IntentPrediction は複数 candidate を保持できる。"""
    c1 = IntentCandidate(text="a", level=ConfidenceLevel.CERTAIN)
    c2 = IntentCandidate(text="b", level=ConfidenceLevel.HIGH, rationale="根拠")
    p = IntentPrediction(candidates=(c1, c2))
    assert p.candidates == (c1, c2)


def test_intent_prediction_is_frozen() -> None:
    """IntentPrediction は frozen。"""
    p = IntentPrediction(candidates=())
    with pytest.raises(ValidationError):
        p.candidates = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _CONFIDENCE_LEVEL_DESCRIPTION の schema 反映
# ---------------------------------------------------------------------------


def test_confidence_level_description_pinned_in_schema() -> None:
    """IntentPrediction.model_json_schema() に 5 レベル全語が description として含まれる。"""
    schema = IntentPrediction.model_json_schema()
    schema_str = str(schema)
    for token in ("certain", "high", "medium", "low", "speculative"):
        assert token in schema_str, f"'{token}' が schema description に含まれていない"


def test_confidence_level_description_pinned_in_intent_candidate_level() -> None:
    """`IntentCandidate.level.description` に 5 レベル全語が含まれる（$defs 経由）。

    IntentPrediction.model_json_schema() の $defs.IntentCandidate.properties.level に
    description が付与されており、5 語すべてを含むことをピン留めする。
    """
    schema = IntentPrediction.model_json_schema()
    defs = schema.get("$defs") or schema.get("definitions") or {}
    candidate_def = defs.get("IntentCandidate")
    assert candidate_def is not None, "$defs に IntentCandidate 定義が無い"
    level_prop = candidate_def["properties"]["level"]
    description = str(level_prop.get("description", ""))
    for token in ("certain", "high", "medium", "low", "speculative"):
        assert token in description, f"'{token}' が level.description に含まれていない"


def test_intent_policy_render_prompt_pins_minimal_structure() -> None:
    """render_prompt の 4 セクション最小構造を pin する（extra_instructions なし版）。

    `#` 見出し 4 種 + 事前定義値からの引用 (categories / 5 levels / max_candidates) のみを
    含み、ロール定義や injection defense・「rationale は必須」制約等の追加テキストは持たない
    ことを検証する。
    """
    policy = IntentPolicy(
        categories=(
            _make_category(name="greet", description="挨拶を検出"),
            _make_category(name="ask", description="質問を検出"),
        ),
        max_candidates=3,
    )
    prompt = policy.render_prompt()

    # Markdown 見出しによる 4 セクション区切り
    for heading in ("# カテゴリ", "# 信頼度 (level)", "# 出力形式", "# 制約"):
        assert heading in prompt

    # 各 category の name: description 行
    assert "- greet: 挨拶を検出" in prompt
    assert "- ask: 質問を検出" in prompt

    # 5 レベルと意味（_CONFIDENCE_LEVEL_MEANINGS 単一ソース経由）
    for level in ("certain", "high", "medium", "low", "speculative"):
        assert f"- {level}: " in prompt

    # 出力例のフィールド並びは pydantic モデル順 (text -> level)。
    # 既定 (include_rationale_in_prompt=False) では rationale は出力例に含まれない。
    example_line = next(
        line for line in prompt.splitlines() if '"text":' in line and '"level":' in line
    )
    assert example_line.index('"text"') < example_line.index('"level"')
    assert '"rationale"' not in prompt

    # 出力例が「複数候補可」を示唆する ... を含む
    assert "..." in prompt

    # max_candidates の反映
    assert "最大 3 件" in prompt

    # 「rationale は必須」制約行は含まれない
    assert "rationale は必須" not in prompt

    # 実装内部のノイズや自己判断で足したテキストが混入していないこと
    for noise in (
        "IntentPrediction",
        "model_json_schema",
        "$defs",
        "pydantic",
        "分類器",
        "untrusted",
        "指示として解釈せず",
        "結論より先に自問",
    ):
        assert noise not in prompt


# ---------------------------------------------------------------------------
# 低精度・高速モデル対応: 固定タスク指示行 / JSON 制約行の pin (Issue #24)
# ---------------------------------------------------------------------------


def test_render_prompt_contains_task_instruction_line() -> None:
    """render_prompt に固定タスク指示行が含まれ、extra_instructions 未指定時は先頭行になる。"""
    policy = IntentPolicy(categories=(_make_category(),))
    prompt = policy.render_prompt()
    assert "ユーザー発話を以下のカテゴリに分類し、JSON のみを出力してください。" in prompt
    assert (
        prompt.splitlines()[0]
        == "ユーザー発話を以下のカテゴリに分類し、JSON のみを出力してください。"
    )


def test_render_prompt_task_line_comes_after_extra_instructions() -> None:
    """extra_instructions 指定時、その内容はタスク指示行より前に位置する。"""
    policy = IntentPolicy(
        categories=(_make_category(),),
        extra_instructions="Focus on X",
    )
    prompt = policy.render_prompt()
    assert prompt.index("Focus on X") < prompt.index("ユーザー発話を以下のカテゴリ")


def test_render_prompt_contains_json_only_constraint() -> None:
    """render_prompt の # 制約 セクションに JSON 以外のテキストを含めない旨の行が含まれる。"""
    policy = IntentPolicy(categories=(_make_category(),))
    prompt = policy.render_prompt()
    assert "- JSON 以外のテキスト（説明文・コードフェンス）を含めない" in prompt
