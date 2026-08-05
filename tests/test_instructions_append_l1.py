"""L1: `AgentSpec.instructions_append` の宣言層検証（registry 登録時の検証）を pin する。

`AgentRegistry.register` が `_validate_spec` で行う 2 件の前倒し検証を扱う:

- `instructions` が callable のときの追記併用を `ValueError` で拒否すること（ADR 0023 判断 2）。
- 各追記要素が `(context, agent)` の 2 引数で bind できることを `instructions_append[i]` の
  `field_label` 付きで検証すること（`validate_instructions_callable` の再利用）。

追記関数が登録時点で評価されないこと（run ごと評価の前提）も本層で pin する。build / run 側の
検証（合成 callable の生成・連結順序・await・例外伝播）は `tests/_adapters/` の `_l2` の責務。
"""

from __future__ import annotations

from typing import Any

import pytest

from oai_agentspec import AgentRegistry, AgentSpec

pytestmark = pytest.mark.unit


def _fragment(context: Any, agent: Any) -> str:
    """`(context, agent)` の 2 引数で bind できる正常な追記関数。"""
    return "fragment"


# ---------------------------------------------------------------------------
# callable instructions との併用拒否
# ---------------------------------------------------------------------------
def test_callable_instructions_with_append_is_rejected_at_register() -> None:
    """`instructions` が callable + 追記非空は register 時に `ValueError`。"""
    reg = AgentRegistry()
    spec = AgentSpec(
        name="bot",
        instructions=lambda ctx, agent: "dynamic",
        instructions_append=[_fragment],
    )
    with pytest.raises(ValueError) as excinfo:
        reg.register(spec)
    message = str(excinfo.value)
    # エージェント名と両フィールド名が分かるメッセージであること。
    assert "'bot'" in message
    assert "instructions" in message
    assert "instructions_append" in message


def test_callable_instructions_without_append_is_accepted() -> None:
    """追記が空なら callable `instructions` は従来どおり登録できる（既存経路の不変）。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions=lambda ctx, agent: "dynamic"))
    assert reg.names() == ["bot"]


def test_none_instructions_with_append_is_accepted() -> None:
    """`instructions=None` + 追記のみは許容される（追記だけを連結する構成）。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions=None, instructions_append=[_fragment]))
    assert reg.names() == ["bot"]


def test_str_instructions_with_append_is_accepted() -> None:
    """静的 str `instructions` + 追記非空は登録できる（本機能の主用途）。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="static", instructions_append=[_fragment]))
    assert reg.names() == ["bot"]


# ---------------------------------------------------------------------------
# 各要素の arity 検証（validate_instructions_callable の再利用）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(lambda: "x", id="zero-arg"),
        pytest.param(lambda ctx: "x", id="one-arg"),
        pytest.param(lambda ctx, agent, extra: "x", id="three-required-args"),
    ],
)
def test_append_element_arity_is_validated_at_register(bad: Any) -> None:
    """2 引数で bind できない追記要素は register 時に `ValueError`（arity 検証の再利用）。"""
    reg = AgentRegistry()
    with pytest.raises(ValueError) as excinfo:
        reg.register(AgentSpec(name="bot", instructions="static", instructions_append=[bad]))
    message = str(excinfo.value)
    assert "'bot'" in message
    # `field_label` にインデックスが入る形（`instructions_append[0]`）を含めて pin する。
    assert "instructions_append[0]" in message


def test_append_element_arity_error_reports_offending_index() -> None:
    """複数要素のうち不正な要素のインデックスがラベルに現れる（[1] を pin）。"""
    reg = AgentRegistry()
    with pytest.raises(ValueError) as excinfo:
        reg.register(
            AgentSpec(
                name="bot",
                instructions="static",
                instructions_append=[_fragment, lambda ctx: "x"],
            )
        )
    assert "instructions_append[1]" in str(excinfo.value)


@pytest.mark.parametrize(
    "good",
    [
        pytest.param(lambda ctx, agent: "x", id="two-positional"),
        pytest.param(lambda ctx, agent=None: "x", id="default-second-arg"),
        pytest.param(lambda *args: "x", id="var-positional"),
    ],
)
def test_append_element_bindable_forms_are_accepted(good: Any) -> None:
    """2 引数で bind できる形（デフォルト引数・`*args` を含む）は登録できる。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="static", instructions_append=[good]))
    assert reg.names() == ["bot"]


# ---------------------------------------------------------------------------
# コンテナ型・要素型の検証（素の str / 非 callable 要素の拒否）
# ---------------------------------------------------------------------------
def test_bare_str_instructions_append_is_rejected_at_register() -> None:
    """`instructions_append="abc"`（list の取り違え）は register 時に `ValueError`。

    素の `str` は反復可能なため素通りし、run で 1 文字ずつ callable として呼ばれて
    `TypeError: 'str' object is not callable` という cryptic な失敗になる。`guardrails` の
    素 str 拒否と同じ失敗類型なので同じ文言（`bare str`）で弾く。
    """
    reg = AgentRegistry()
    with pytest.raises(ValueError, match="bare str") as excinfo:
        reg.register(AgentSpec(name="bot", instructions="static", instructions_append="abc"))  # type: ignore[arg-type]
    assert "instructions_append" in str(excinfo.value)
    assert reg.names() == []


def test_bare_str_instructions_append_is_rejected_at_update() -> None:
    """`update()` 経路でも素の `str` 追記は弾く（登録後の差し替えで穴が開かない）。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="static"))
    with pytest.raises(ValueError, match="bare str"):
        reg.update(AgentSpec(name="bot", instructions="static", instructions_append="abc"))  # type: ignore[arg-type]


def test_non_callable_append_element_is_rejected_at_register() -> None:
    """callable でない追記要素は register 時に `ValueError`（インデックス付きラベル）。"""
    reg = AgentRegistry()
    with pytest.raises(ValueError) as excinfo:
        reg.register(AgentSpec(name="bot", instructions="static", instructions_append=["oops"]))  # type: ignore[list-item]
    message = str(excinfo.value)
    assert "'bot'" in message
    assert "instructions_append[0]" in message


def test_non_callable_append_element_reports_offending_index() -> None:
    """非 callable 要素が 2 番目なら `instructions_append[1]` が現れる（インデックスの pin）。"""
    reg = AgentRegistry()
    with pytest.raises(ValueError) as excinfo:
        reg.register(
            AgentSpec(
                name="bot",
                instructions="static",
                instructions_append=[_fragment, 42],  # type: ignore[list-item]
            )
        )
    message = str(excinfo.value)
    assert "instructions_append[1]" in message
    assert "instructions_append[0]" not in message


def test_explicit_empty_append_list_is_accepted() -> None:
    """明示的な空 list の追記はコンテナ型検証を追加しても登録できる（非退行）。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="static", instructions_append=[]))
    assert reg.names() == ["bot"]


# ---------------------------------------------------------------------------
# コンテナは Sequence に限る（使い切り iterable / 無順序容器の拒否）
# ---------------------------------------------------------------------------
def test_generator_container_is_rejected_at_register() -> None:
    """generator 容器は register 時に `ValueError`（検証ループが消費して断片が消えるため）。

    使い切り iterable を渡すと `_validate_spec` の走査がそれを消費し、build 側の
    `list(spec.instructions_append)` が空になって追記断片が silent に消える（fail-open）。
    宣言時に型名付きで弾くことで、カナリア埋め込みの恒久的な非発火を防ぐ。
    """
    reg = AgentRegistry()
    with pytest.raises(ValueError) as excinfo:
        reg.register(
            AgentSpec(
                name="bot",
                instructions="static",
                instructions_append=(_fragment for _ in range(2)),  # type: ignore[arg-type]
            )
        )
    message = str(excinfo.value)
    assert "'bot'" in message
    assert "instructions_append" in message
    # 原因の型が分かるメッセージであること（`type(...).__name__` 相当）。
    assert "generator" in message
    assert reg.names() == []


def test_set_container_is_rejected_at_register() -> None:
    """set 容器は register 時に `ValueError`（宣言順の連結契約が非決定的に破れるため）。"""
    reg = AgentRegistry()

    def other(context: Any, agent: Any) -> str:
        return "other"

    with pytest.raises(ValueError) as excinfo:
        reg.register(
            AgentSpec(
                name="bot",
                instructions="static",
                instructions_append={_fragment, other},  # type: ignore[arg-type]
            )
        )
    message = str(excinfo.value)
    assert "instructions_append" in message
    assert "set" in message
    assert reg.names() == []


def test_iterator_container_is_rejected_at_register() -> None:
    """`iter([...])` の iterator 容器も register 時に `ValueError`（使い切り iterable）。"""
    reg = AgentRegistry()
    with pytest.raises(ValueError) as excinfo:
        reg.register(
            AgentSpec(
                name="bot",
                instructions="static",
                instructions_append=iter([_fragment]),  # type: ignore[arg-type]
            )
        )
    message = str(excinfo.value)
    assert "instructions_append" in message
    assert reg.names() == []


def test_list_container_is_accepted() -> None:
    """`list` 容器は従来どおり受理される（Sequence 検証追加後も非退行）。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="static", instructions_append=[_fragment]))
    assert reg.names() == ["bot"]


def test_tuple_container_is_accepted() -> None:
    """`tuple` 容器も受理される（Sequence 検証で過剰に拒否しないことの pin）。"""
    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="static", instructions_append=(_fragment,)))
    assert reg.names() == ["bot"]


# ---------------------------------------------------------------------------
# エラーメッセージに値を載せない（カナリア文字列の漏洩防止）
# ---------------------------------------------------------------------------
def test_bare_str_error_message_does_not_leak_the_value() -> None:
    """素の `str` 追記のエラーメッセージに渡した文字列の値を含めない。

    `instructions_append` にカナリア埋め込み文そのものを取り違えて渡した場合、値を
    メッセージへ載せるとトークンがログ・トレースへ流出する。型名のみを報告する。
    """
    reg = AgentRegistry()
    secret = "CT-SECRET-TOKEN"
    with pytest.raises(ValueError) as excinfo:
        reg.register(AgentSpec(name="bot", instructions="static", instructions_append=secret))  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert secret not in message
    # 値を落としてもメッセージが空洞化していない（原因の型が分かる）こと。
    assert "instructions_append" in message
    assert "str" in message


# ---------------------------------------------------------------------------
# 検証ループが容器を消費しない
# ---------------------------------------------------------------------------
def test_register_does_not_consume_the_append_container() -> None:
    """register の検証ループが追記容器を消費しない（消費バグの再発検知）。"""

    def second(context: Any, agent: Any) -> str:
        return "second"

    reg = AgentRegistry()
    spec = AgentSpec(name="bot", instructions="static", instructions_append=[_fragment, second])
    reg.register(spec)
    assert list(spec.instructions_append) == [_fragment, second]


def test_iterator_container_is_rejected_before_any_element_is_consumed() -> None:
    """使い切り容器の拒否は要素走査より**前**に行う（容器を消費してから落ちない）。

    `list` 容器では反復が非破壊なため「消費しないこと」を観測できない。消費可能な
    `iter([...])` を渡し、`ValueError` 送出後も容器が未消費であることで検証順序（容器型判定 →
    要素ループ）を pin する（ADR 0023 判断 11）。順序を入れ替える変異では、拒否の前に要素が
    読み出されて容器が空になる。
    """
    reg = AgentRegistry()
    container = iter([_fragment])
    with pytest.raises(ValueError) as excinfo:
        reg.register(
            AgentSpec(
                name="bot",
                instructions="static",
                instructions_append=container,  # type: ignore[arg-type]
            )
        )
    assert "instructions_append" in str(excinfo.value)
    # 拒否時点で 1 要素も読み出されていない（未消費のまま残っている）。
    assert list(container) == [_fragment]
    assert reg.names() == []


# ---------------------------------------------------------------------------
# 登録・validate の時点では追記関数を評価しない
# ---------------------------------------------------------------------------
def test_append_functions_are_not_called_at_register_or_validate() -> None:
    """register / validate は追記関数を呼ばない（評価は run ごとの SDK 側責務）。"""

    def sentinel(context: Any, agent: Any) -> str:
        pytest.fail("instructions_append は登録・validate 時に評価されてはならない")

    reg = AgentRegistry()
    reg.register(AgentSpec(name="bot", instructions="static", instructions_append=[sentinel]))
    reg.validate()
