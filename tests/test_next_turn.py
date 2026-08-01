"""L1: Next-Turn Agent Override の宣言型と解決（宣言 / `resolve_next_agent` / `next_turn_agent`）。

FR-1（次ターン開始エージェントの宣言）の受け入れ基準を決定表として pin する:

- `NextTurnRule`: frozen dataclass の既定値（`next_agent=None` / `no_handoff_on_arrival=False` /
  `source=None`）・フィールド保持・frozen 性（`dataclasses.FrozenInstanceError`）と、
  「次ターン指定と到達時ハンドオフ禁止のいずれも持たない宣言」（`source` のみ・全未指定）の
  build-time 拒否（効果のない宣言の拒否）。
- `NextTurnPolicy` の値位置多態: 次ターン指定のみの str 略記 / 単一 `NextTurnRule` /
  `NextTurnRule` の列（tuple・list の双方）。
- 正規化と不変化: 構築後の `rules` は `MappingProxyType` で事後注入できず、呼び出し側が渡した
  元 dict を後から変更しても反映されない（`FailsafePolicy.handlers` と同一方針）。値位置は
  ルールの列として読める形へ正規化され、列の順序は宣言順のまま保たれる。
- build-time `ValueError`: キー / `next_agent` / `source` が str でない・空文字、同一 X の
  ルール列内の `source` 重複、到達元条件なしの包括ルールが 2 件以上、空列。
- 許容: 空 `rules`（no-op 宣言）・X == `next_agent`（継続の明示固定）・`source` == X
  （発動しないだけの無害宣言）・禁止のみルール（`next_agent` なし）。

FR-2（`resolve_next_agent`）/ FR-4（`next_turn_agent`）の受け入れ基準も同じ形で pin する:

- 発動条件の AND: 「ターン内にハンドオフ遷移が 1 件以上」かつ「最終回答者名が X」。
- 発動ルールの選定規則:「X への**最後の到達**の遷移元と一致する `source` を持つルール ->
  `source` なしの包括ルール -> 発動ルールなし」。選ばれたルールが `next_agent` を持たない
  （禁止のみ）なら「上書きなし」（`None`）。
- 防御的解決: `last_agent` / `new_items` の属性欠落・アクセス例外でも例外を送出せず
  「上書きなし」へ倒す。純粋性（入力不変・決定的）。
- `next_turn_agent`: 発動時は `registry.get(Y)` の戻り値、非発動時は `result.last_agent` を
  **そのまま**（同一性まで）返す。`last_agent` も取得できない場合のみ `None`
  （「開始エージェント決定不能」であり `resolve_next_agent` の `None` とは意味が異なる）。
  Y が未登録なら registry の `KeyError` を握り潰さず伝播する。

`agents` 非依存（宣言層は SDK 型を一切扱わない）。run 完了結果と registry は SDK 型ではなく
フェイク（`last_agent` / `new_items` を持つ simple object・`get(name)` を持つ simple object）で
与える。frozen dataclass の等価性・repr は契約に含めないため pin しない（フィールド単位で
読み取って検証する）。
"""

from __future__ import annotations

import dataclasses
import logging
from types import MappingProxyType
from typing import Any

import pytest

from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.constants import NEXT_TURN_LOGGER_NAME
from oai_agentspec.next_turn import (
    NextTurnPolicy,
    NextTurnRule,
    apply_next_turn_policy,
    next_turn_agent,
    resolve_next_agent,
)
from oai_agentspec.registry import RegistryFrozenError

from _helpers.fake_builder import FakeAgent, FakeAgentBuilder

pytestmark = pytest.mark.unit


def _rules_of(policy: NextTurnPolicy, key: str) -> tuple[NextTurnRule, ...]:
    """X のエントリを「ルールの列」として読む（単一ルール正規化の形に依存しない）。

    Args:
        policy: 検証対象の宣言。
        key: 回答者名 X。

    Returns:
        X に宣言されたルールを宣言順に並べた tuple。
    """
    entry: Any = policy.rules[key]
    if isinstance(entry, NextTurnRule):
        return (entry,)
    return tuple(entry)


# ---------------------------------------------------------------------------
# NextTurnRule: 既定値・フィールド保持・frozen 性
# ---------------------------------------------------------------------------


def test_next_turn_rule_次ターン指定のみの既定値() -> None:
    """`next_agent` だけを指定した場合、禁止は False・到達元条件は None が既定になる。"""
    rule = NextTurnRule(next_agent="triage")

    assert rule.next_agent == "triage"
    assert rule.no_handoff_on_arrival is False
    assert rule.source is None


def test_next_turn_rule_禁止のみの宣言は既定でnext_agentを持たない() -> None:
    """禁止のみルール（`no_handoff_on_arrival=True`）は次ターン指定を持たずに構築できる。"""
    rule = NextTurnRule(no_handoff_on_arrival=True)

    assert rule.next_agent is None
    assert rule.no_handoff_on_arrival is True
    assert rule.source is None


def test_next_turn_rule_フル指定でフィールド保持() -> None:
    """3 フィールドを明示指定した値がそのまま保持される。"""
    rule = NextTurnRule(next_agent="triage", no_handoff_on_arrival=True, source="server")

    assert rule.next_agent == "triage"
    assert rule.no_handoff_on_arrival is True
    assert rule.source == "server"


def test_next_turn_rule_is_frozen() -> None:
    """frozen dataclass のため属性の書き換えは FrozenInstanceError。"""
    rule = NextTurnRule(next_agent="triage")

    with pytest.raises(dataclasses.FrozenInstanceError):
        rule.next_agent = "billing"  # type: ignore[misc]

    assert rule.next_agent == "triage"


def test_next_turn_rule_no_handoff_on_arrivalもfrozenで書き換え不可() -> None:
    """禁止フラグも frozen の対象で、宣言後の opt-in 追加はできない。"""
    rule = NextTurnRule(next_agent="triage")

    with pytest.raises(dataclasses.FrozenInstanceError):
        rule.no_handoff_on_arrival = True  # type: ignore[misc]

    assert rule.no_handoff_on_arrival is False


# ---------------------------------------------------------------------------
# NextTurnRule: build-time ValueError（効果のない宣言の拒否）
# ---------------------------------------------------------------------------


def test_next_turn_rule_全未指定は_ValueError() -> None:
    """次ターン指定も禁止も持たない宣言は効果が無いため build-time で ValueError。"""
    with pytest.raises(ValueError):
        NextTurnRule()


def test_next_turn_rule_sourceのみの宣言は_ValueError() -> None:
    """到達元条件のみのルールは効果が無いため build-time で ValueError。"""
    with pytest.raises(ValueError):
        NextTurnRule(source="triage")


def test_next_turn_rule_禁止Falseの明示だけでも_ValueError() -> None:
    """`no_handoff_on_arrival=False` の明示は「禁止を持たない」であり効果のない宣言。"""
    with pytest.raises(ValueError):
        NextTurnRule(no_handoff_on_arrival=False, source="triage")


# ---------------------------------------------------------------------------
# NextTurnPolicy: 既定値・値位置多態
# ---------------------------------------------------------------------------


def test_next_turn_policy_既定のrulesは空() -> None:
    """`NextTurnPolicy()` は生成でき、既定の rules は空（no-op 宣言）。"""
    policy = NextTurnPolicy()

    assert len(policy.rules) == 0
    assert dict(policy.rules) == {}


def test_next_turn_policy_空rulesは許容される() -> None:
    """空 dict の明示指定も no-op 宣言として許容される（矛盾ではない）。"""
    policy = NextTurnPolicy(rules={})

    assert len(policy.rules) == 0


def test_next_turn_policy_str略記は次ターン指定のみのルールへ正規化される() -> None:
    """値位置の str は `NextTurnRule(next_agent=<str>)` 相当へ正規化される（禁止は付かない）。"""
    policy = NextTurnPolicy(rules={"tech": "triage"})

    rules = _rules_of(policy, "tech")
    assert len(rules) == 1
    assert rules[0].next_agent == "triage"
    assert rules[0].no_handoff_on_arrival is False
    assert rules[0].source is None


def test_next_turn_policy_単一ルール値はそのまま保持される() -> None:
    """値位置の単一 `NextTurnRule` は宣言した 3 フィールドのまま読める。"""
    policy = NextTurnPolicy(
        rules={"billing": NextTurnRule(next_agent="tech", no_handoff_on_arrival=True)}
    )

    rules = _rules_of(policy, "billing")
    assert len(rules) == 1
    assert rules[0].next_agent == "tech"
    assert rules[0].no_handoff_on_arrival is True
    assert rules[0].source is None


def test_next_turn_policy_tuple列は宣言順のまま保持される() -> None:
    """tuple のルール列は宣言順（source 付き -> 包括）のまま読める。"""
    policy = NextTurnPolicy(
        rules={
            "billing": (
                NextTurnRule(next_agent="triage", no_handoff_on_arrival=True, source="server"),
                NextTurnRule(no_handoff_on_arrival=True),
            )
        }
    )

    rules = _rules_of(policy, "billing")
    assert [r.source for r in rules] == ["server", None]
    assert [r.next_agent for r in rules] == ["triage", None]
    assert [r.no_handoff_on_arrival for r in rules] == [True, True]


def test_next_turn_policy_list列も同じ形へ正規化される() -> None:
    """list のルール列も tuple と同じく宣言順で読める（Sequence 多態）。"""
    policy = NextTurnPolicy(
        rules={
            "billing": [
                NextTurnRule(next_agent="server", source="tech"),
                NextTurnRule(next_agent="triage", source="server"),
            ]
        }
    )

    rules = _rules_of(policy, "billing")
    assert [r.source for r in rules] == ["tech", "server"]
    assert [r.next_agent for r in rules] == ["server", "triage"]


def test_next_turn_policy_複数エントリを同時に宣言できる() -> None:
    """複数の回答者 X のエントリを 1 つの宣言で保持できる。"""
    policy = NextTurnPolicy(
        rules={
            "billing": NextTurnRule(next_agent="triage"),
            "tech": "server",
        }
    )

    assert set(policy.rules) == {"billing", "tech"}
    assert _rules_of(policy, "billing")[0].next_agent == "triage"
    assert _rules_of(policy, "tech")[0].next_agent == "server"


# ---------------------------------------------------------------------------
# NextTurnPolicy: 正規化と不変化
# ---------------------------------------------------------------------------


def test_next_turn_policy_is_frozen() -> None:
    """frozen dataclass のため属性の差し替えは FrozenInstanceError。"""
    policy = NextTurnPolicy(rules={"tech": "triage"})

    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.rules = {}  # type: ignore[misc]

    assert set(policy.rules) == {"tech"}


def test_next_turn_policy_rulesはMappingProxyTypeへ差し替えられる() -> None:
    """構築後の rules は `MappingProxyType`（読み取り専用ビュー）になる。"""
    policy = NextTurnPolicy(rules={"tech": "triage"})

    assert isinstance(policy.rules, MappingProxyType)


def test_next_turn_policy_rulesへの事後注入はできない() -> None:
    """構築後の rules は書き込み不可で、検証を迂回したエントリの事後注入ができない。"""
    policy = NextTurnPolicy(rules={"tech": "triage"})

    with pytest.raises(TypeError):
        policy.rules["billing"] = NextTurnRule(next_agent="tech")  # type: ignore[index]

    assert set(policy.rules) == {"tech"}


def test_next_turn_policy_元dictの変更は反映されない() -> None:
    """rules はコピーしてから不変化するため、呼び出し側の元 dict 経由の事後注入も効かない。"""
    source: dict[str, Any] = {"tech": "triage"}
    policy = NextTurnPolicy(rules=source)

    source["billing"] = NextTurnRule(next_agent="tech")
    source["tech"] = "server"

    assert set(policy.rules) == {"tech"}
    assert _rules_of(policy, "tech")[0].next_agent == "triage"


def test_next_turn_policy_元listの変更は反映されない() -> None:
    """値位置の list も列として正規化されるため、元 list への追記は反映されない。"""
    entry = [NextTurnRule(next_agent="triage", source="server")]
    policy = NextTurnPolicy(rules={"billing": entry})

    entry.append(NextTurnRule(no_handoff_on_arrival=True))

    rules = _rules_of(policy, "billing")
    assert len(rules) == 1
    assert rules[0].source == "server"


# ---------------------------------------------------------------------------
# NextTurnPolicy: build-time ValueError（名前の型・空文字）
# ---------------------------------------------------------------------------


def test_next_turn_policy_非strキーは_ValueError() -> None:
    """キーが str でなければ build-time で ValueError。"""
    with pytest.raises(ValueError):
        NextTurnPolicy(rules={1: NextTurnRule(next_agent="triage")})  # type: ignore[dict-item]


def test_next_turn_policy_空文字キーは_ValueError() -> None:
    """キーが空文字なら build-time で ValueError。"""
    with pytest.raises(ValueError):
        NextTurnPolicy(rules={"": NextTurnRule(next_agent="triage")})


def test_next_turn_policy_非strのnext_agentは_ValueError() -> None:
    """次ターン指定が str でなければ build-time で ValueError。"""
    with pytest.raises(ValueError):
        NextTurnPolicy(rules={"billing": NextTurnRule(next_agent=42)})  # type: ignore[arg-type]


def test_next_turn_policy_空文字のnext_agentは_ValueError() -> None:
    """次ターン指定が空文字なら build-time で ValueError。"""
    with pytest.raises(ValueError):
        NextTurnPolicy(rules={"billing": NextTurnRule(next_agent="")})


def test_next_turn_policy_str略記の空文字は_ValueError() -> None:
    """str 略記も次ターン指定として同じ検証を受ける（空文字は ValueError）。"""
    with pytest.raises(ValueError):
        NextTurnPolicy(rules={"billing": ""})


def test_next_turn_policy_非strのsourceは_ValueError() -> None:
    """到達元条件が str でなければ build-time で ValueError。"""
    with pytest.raises(ValueError):
        NextTurnPolicy(
            rules={"billing": NextTurnRule(next_agent="triage", source=42)}  # type: ignore[arg-type]
        )


def test_next_turn_policy_空文字のsourceは_ValueError() -> None:
    """到達元条件が空文字なら build-time で ValueError。"""
    with pytest.raises(ValueError):
        NextTurnPolicy(rules={"billing": NextTurnRule(next_agent="triage", source="")})


# ---------------------------------------------------------------------------
# NextTurnPolicy: build-time ValueError（発動ルールの選定が一意にならない宣言）
# ---------------------------------------------------------------------------


def test_next_turn_policy_同一sourceの重複は_ValueError() -> None:
    """同一 X のルール列に同じ到達元条件が 2 件あると選定が一意にならず ValueError。"""
    with pytest.raises(ValueError):
        NextTurnPolicy(
            rules={
                "billing": (
                    NextTurnRule(next_agent="triage", source="server"),
                    NextTurnRule(no_handoff_on_arrival=True, source="server"),
                )
            }
        )


def test_next_turn_policy_包括ルール2件は_ValueError() -> None:
    """到達元条件なしの包括ルールが 2 件あると選定が一意にならず ValueError。"""
    with pytest.raises(ValueError):
        NextTurnPolicy(
            rules={
                "billing": (
                    NextTurnRule(next_agent="triage"),
                    NextTurnRule(no_handoff_on_arrival=True),
                )
            }
        )


def test_next_turn_policy_空tuple列は_ValueError() -> None:
    """ルール 0 件の空列は効果のない宣言として ValueError。"""
    with pytest.raises(ValueError):
        NextTurnPolicy(rules={"billing": ()})


def test_next_turn_policy_空list列は_ValueError() -> None:
    """list の空列も同じく ValueError。"""
    with pytest.raises(ValueError):
        NextTurnPolicy(rules={"billing": []})


# ---------------------------------------------------------------------------
# NextTurnPolicy: build-time ValueError（値位置の型・列の要素）
# ---------------------------------------------------------------------------


def test_next_turn_policy_列内に非NextTurnRuleの要素があると_ValueError() -> None:
    """ルール列の要素に `NextTurnRule` でないものが混ざると、要素の型名を含む ValueError。"""
    with pytest.raises(ValueError, match="NextTurnRule instances"):
        NextTurnPolicy(
            rules={
                "billing": (
                    NextTurnRule(next_agent="triage"),
                    "not-a-rule",
                )
            }  # type: ignore[dict-item]
        )


def test_next_turn_policy_値位置がintなら_ValueError() -> None:
    """値位置が str / `NextTurnRule` / 列のいずれでもなければ、型名を含む ValueError。"""
    with pytest.raises(ValueError, match="int"):
        NextTurnPolicy(rules={"billing": 42})  # type: ignore[dict-item]


def test_next_turn_policy_値位置がdictなら_ValueError() -> None:
    """dict は `Sequence` でないため、非対応型として同じ経路で ValueError になる。"""
    with pytest.raises(ValueError, match="dict"):
        NextTurnPolicy(rules={"billing": {"next_agent": "triage"}})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# NextTurnPolicy: 許容される宣言（拒否しすぎないこと）
# ---------------------------------------------------------------------------


def test_next_turn_policy_source付きと包括の混在は許容される() -> None:
    """到達元条件付き 1 件 + 包括 1 件は選定が一意に決まるため許容される。"""
    policy = NextTurnPolicy(
        rules={
            "billing": (
                NextTurnRule(next_agent="triage", no_handoff_on_arrival=True, source="triage"),
                NextTurnRule(no_handoff_on_arrival=True),
            )
        }
    )

    rules = _rules_of(policy, "billing")
    assert [r.source for r in rules] == ["triage", None]


def test_next_turn_policy_異なるsourceの複数ルールは許容される() -> None:
    """到達元条件が互いに異なるルールは何件でも許容される。"""
    policy = NextTurnPolicy(
        rules={
            "billing": (
                NextTurnRule(next_agent="server", source="tech"),
                NextTurnRule(next_agent="triage", source="server"),
                NextTurnRule(no_handoff_on_arrival=True, source="triage"),
            )
        }
    )

    rules = _rules_of(policy, "billing")
    assert [r.source for r in rules] == ["tech", "server", "triage"]


def test_next_turn_policy_next_agentがキーと同名でも許容される() -> None:
    """X == Y（継続の明示固定）は有効な宣言として構築できる。"""
    policy = NextTurnPolicy(rules={"billing": NextTurnRule(next_agent="billing")})

    assert _rules_of(policy, "billing")[0].next_agent == "billing"


def test_next_turn_policy_sourceがキーと同名でも許容される() -> None:
    """source == X（自分からの到達）は発動しないだけの無害な宣言として許容される。"""
    policy = NextTurnPolicy(rules={"billing": NextTurnRule(next_agent="triage", source="billing")})

    rule = _rules_of(policy, "billing")[0]
    assert rule.source == "billing"
    assert rule.next_agent == "triage"


def test_next_turn_policy_禁止のみルールは許容される() -> None:
    """次ターン指定なし・禁止のみのルールは有効な宣言として構築できる。"""
    policy = NextTurnPolicy(rules={"billing": NextTurnRule(no_handoff_on_arrival=True)})

    rule = _rules_of(policy, "billing")[0]
    assert rule.next_agent is None
    assert rule.no_handoff_on_arrival is True


# ---------------------------------------------------------------------------
# 解決（FR-2 / FR-4）用のフェイク（SDK 型に依存しない）
# ---------------------------------------------------------------------------


class _FakeAgent:
    """SDK Agent の代役（判定に必要な `name` のみを持つ最小形）。"""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeHandoffItem:
    """handoff アイテムの代役（`source_agent` / `target_agent` を持つ）。"""

    def __init__(self, source: str, target: str) -> None:
        self.source_agent = _FakeAgent(source)
        self.target_agent = _FakeAgent(target)


class _FakeResult:
    """run 完了結果の代役（`last_agent` / `new_items` を持つ）。"""

    def __init__(self, last_agent: Any, new_items: Any) -> None:
        self.last_agent = last_agent
        self.new_items = new_items


class _ResultWithoutLastAgent:
    """`last_agent` 属性を持たない結果（属性欠落の防御読みの検証用）。"""

    def __init__(self, new_items: Any) -> None:
        self.new_items = new_items


class _ResultRaisingLastAgent:
    """`last_agent` の読み出し自体が例外を送出する結果（release 後アクセス相当）。"""

    def __init__(self, new_items: Any) -> None:
        self.new_items = new_items

    @property
    def last_agent(self) -> Any:
        """アクセスのたびに例外を送出する。

        Raises:
            RuntimeError: 常に送出する。
        """
        raise RuntimeError("last_agent is no longer available")


class _ResultRaisingNewItems:
    """`new_items` の読み出し自体が例外を送出する結果。"""

    def __init__(self, last_agent: Any) -> None:
        self.last_agent = last_agent

    @property
    def new_items(self) -> Any:
        """アクセスのたびに例外を送出する。

        Raises:
            RuntimeError: 常に送出する。
        """
        raise RuntimeError("new_items is no longer available")


class _FakeRegistry:
    """`AgentRegistry` の代役（`get(name)` のみを持ち、未登録は `KeyError`）。"""

    def __init__(self, agents: dict[str, Any]) -> None:
        self.agents = dict(agents)
        self.calls: list[str] = []

    def get(self, name: str) -> Any:
        """登録済みの Agent 代役を返す（呼び出し名を記録する）。

        Args:
            name: 解決するエージェント名。

        Returns:
            登録済みの Agent 代役。

        Raises:
            KeyError: 未登録の名前を渡された場合。
        """
        self.calls.append(name)
        return self.agents[name]


def _turn(last_agent: Any, *handoffs: tuple[str, str]) -> _FakeResult:
    """ターンの run 完了結果を組み立てる。

    Args:
        last_agent: 最終回答者（Agent 代役 / None）。
        *handoffs: 観測順に並べた `(遷移元, 遷移先)` の組。

    Returns:
        フェイクの run 完了結果。
    """
    return _FakeResult(last_agent, [_FakeHandoffItem(src, dst) for src, dst in handoffs])


# 一致 source ルール・包括ルール・到達元のいずれとも重ならない名前を使い、どの経路で
# 決まった値かを戻り値だけで識別できるようにする（選定規則の無効化を検知するため）。
_SOURCE_MATCHED_POLICY = NextTurnPolicy(
    rules={
        "billing": (
            NextTurnRule(next_agent="frontdesk", source="triage"),
            NextTurnRule(next_agent="lobby"),
        )
    }
)


# ---------------------------------------------------------------------------
# resolve_next_agent: 発動ルールの選定規則（FR-2）
# ---------------------------------------------------------------------------


def test_resolve_next_agent_一致sourceのルールが包括より優先される() -> None:
    """到達元と一致する `source` を持つルールが第一候補として選ばれる（包括は選ばれない）。"""
    result = _turn(_FakeAgent("billing"), ("triage", "billing"))

    assert resolve_next_agent(_SOURCE_MATCHED_POLICY, result) == "frontdesk"


def test_resolve_next_agent_不一致sourceでは包括ルールが選ばれる() -> None:
    """一致する `source` が無ければ第二候補の包括ルールが選ばれる。"""
    result = _turn(_FakeAgent("billing"), ("server", "billing"))

    assert resolve_next_agent(_SOURCE_MATCHED_POLICY, result) == "lobby"


def test_resolve_next_agent_包括が禁止のみなら上書きなし() -> None:
    """選ばれた包括ルールが次ターン指定を持たなければ「上書きなし」（None）。"""
    policy = NextTurnPolicy(
        rules={
            "billing": (
                NextTurnRule(next_agent="frontdesk", source="triage"),
                NextTurnRule(no_handoff_on_arrival=True),
            )
        }
    )
    result = _turn(_FakeAgent("billing"), ("server", "billing"))

    assert resolve_next_agent(policy, result) is None


def test_resolve_next_agent_一致ルールが禁止のみなら包括へ落ちない() -> None:
    """選定は先に行い、選ばれたルールが禁止のみなら None（次ターン指定を持つ包括へ落ちない）。"""
    policy = NextTurnPolicy(
        rules={
            "billing": (
                NextTurnRule(no_handoff_on_arrival=True, source="triage"),
                NextTurnRule(next_agent="lobby"),
            )
        }
    )
    result = _turn(_FakeAgent("billing"), ("triage", "billing"))

    assert resolve_next_agent(policy, result) is None


def test_resolve_next_agent_単一ルールのsourceが一致しなければ上書きなし() -> None:
    """包括ルールが無く `source` も一致しなければ発動ルールなし（None）。"""
    policy = NextTurnPolicy(
        rules={"billing": NextTurnRule(next_agent="frontdesk", source="triage")}
    )
    result = _turn(_FakeAgent("billing"), ("server", "billing"))

    assert resolve_next_agent(policy, result) is None


def test_resolve_next_agent_str略記エントリは次ターン名を返す() -> None:
    """str 略記のエントリは包括ルール（次ターン指定のみ）として発動する。"""
    policy = NextTurnPolicy(rules={"tech": "triage"})
    result = _turn(_FakeAgent("tech"), ("server", "tech"))

    assert resolve_next_agent(policy, result) == "triage"


# ---------------------------------------------------------------------------
# resolve_next_agent: 発動条件の AND（ハンドオフ経由 + 回答者一致）
# ---------------------------------------------------------------------------


def test_resolve_next_agent_ハンドオフが無ければ上書きなし() -> None:
    """開始エージェントがそのまま回答したターンは、宣言があっても非発動（None）。"""
    result = _turn(_FakeAgent("billing"))

    assert resolve_next_agent(_SOURCE_MATCHED_POLICY, result) is None


def test_resolve_next_agent_last_agentがキーと不一致なら上書きなし() -> None:
    """最終回答者がどのキーとも一致しなければ「上書きなし」（None）。"""
    result = _turn(_FakeAgent("triage"), ("triage", "billing"), ("billing", "triage"))

    assert resolve_next_agent(_SOURCE_MATCHED_POLICY, result) is None


def test_resolve_next_agent_空のpolicyは常に上書きなし() -> None:
    """no-op 宣言（空 rules）では何が観測されても None。"""
    result = _turn(_FakeAgent("billing"), ("triage", "billing"))

    assert resolve_next_agent(NextTurnPolicy(), result) is None


def test_resolve_next_agent_last_agentへの到達エッジが無ければ上書きなし() -> None:
    """ハンドオフは観測され last_agent も X と一致するが、X 自身への到達エッジが無ければ None。

    triage は開始エージェントとして回答を終えており、観測されたハンドオフは
    `triage -> billing` のみ（triage への到達ではない）。
    """
    policy = NextTurnPolicy(rules={"triage": NextTurnRule(next_agent="tech")})
    result = _turn(_FakeAgent("triage"), ("triage", "billing"))

    assert resolve_next_agent(policy, result) is None


# ---------------------------------------------------------------------------
# resolve_next_agent: 「X への最後の到達」で選定する
# ---------------------------------------------------------------------------


def test_resolve_next_agent_最後の到達の遷移元で選定する() -> None:
    """複数回 X へ到達した場合、最後の到達（server -> billing）の遷移元で選ぶ。"""
    policy = NextTurnPolicy(
        rules={
            "billing": (
                NextTurnRule(next_agent="frontdesk", source="triage"),
                NextTurnRule(next_agent="lobby", source="server"),
            )
        }
    )
    result = _turn(
        _FakeAgent("billing"),
        ("triage", "billing"),
        ("billing", "server"),
        ("server", "billing"),
    )

    assert resolve_next_agent(policy, result) == "lobby"


def test_resolve_next_agent_到達順を入れ替えると選定ルールも入れ替わる() -> None:
    """到達順を逆にすると最後の到達も変わり、選ばれるルールが入れ替わる（最初の到達ではない）。"""
    policy = NextTurnPolicy(
        rules={
            "billing": (
                NextTurnRule(next_agent="frontdesk", source="triage"),
                NextTurnRule(next_agent="lobby", source="server"),
            )
        }
    )
    result = _turn(
        _FakeAgent("billing"),
        ("server", "billing"),
        ("billing", "triage"),
        ("triage", "billing"),
    )

    assert resolve_next_agent(policy, result) == "frontdesk"


def test_resolve_next_agent_X起点の循環でも最後の到達で選定する() -> None:
    """X 起点で他へ handoff し X へ再到達したターンも、その到達の遷移元で選ぶ。"""
    policy = NextTurnPolicy(
        rules={
            "billing": (
                NextTurnRule(next_agent="frontdesk", source="tech"),
                NextTurnRule(next_agent="lobby"),
            )
        }
    )
    result = _turn(_FakeAgent("billing"), ("billing", "tech"), ("tech", "billing"))

    assert resolve_next_agent(policy, result) == "frontdesk"


def test_resolve_next_agent_X以外への到達は選定に影響しない() -> None:
    """選定に使うのは「X への」最後の到達であり、他エージェントへの到達は無視される。"""
    result = _turn(_FakeAgent("billing"), ("triage", "billing"), ("server", "tech"))

    assert resolve_next_agent(_SOURCE_MATCHED_POLICY, result) == "frontdesk"


# ---------------------------------------------------------------------------
# resolve_next_agent: 防御的解決（NFR-5）と純粋性
# ---------------------------------------------------------------------------


def test_resolve_next_agent_last_agent属性の欠落は上書きなし() -> None:
    """判定材料が取れないときは例外を送出せず安全側（None）へ倒す。"""
    result = _ResultWithoutLastAgent([_FakeHandoffItem("triage", "billing")])

    assert resolve_next_agent(_SOURCE_MATCHED_POLICY, result) is None


def test_resolve_next_agent_last_agentのアクセス例外も上書きなし() -> None:
    """属性アクセス自体が例外を送出しても伝播させず None を返す。"""
    result = _ResultRaisingLastAgent([_FakeHandoffItem("triage", "billing")])

    assert resolve_next_agent(_SOURCE_MATCHED_POLICY, result) is None


def test_resolve_next_agent_new_itemsのアクセス例外も上書きなし() -> None:
    """ハンドオフ観測が取れないときは AND 条件が成立せず None（例外にしない）。"""
    result = _ResultRaisingNewItems(_FakeAgent("billing"))

    assert resolve_next_agent(_SOURCE_MATCHED_POLICY, result) is None


def test_resolve_next_agent_入力を変更しない() -> None:
    """解決は読み取りのみで、宣言と結果オブジェクトを変更しない。"""
    agent = _FakeAgent("billing")
    items = [_FakeHandoffItem("triage", "billing")]
    snapshot = list(items)
    result = _FakeResult(agent, items)
    rules_before = {
        key: tuple(_rules_of(_SOURCE_MATCHED_POLICY, key)) for key in _SOURCE_MATCHED_POLICY.rules
    }

    resolve_next_agent(_SOURCE_MATCHED_POLICY, result)

    assert result.last_agent is agent
    assert result.new_items is items
    assert items == snapshot
    assert {
        key: tuple(_rules_of(_SOURCE_MATCHED_POLICY, key)) for key in _SOURCE_MATCHED_POLICY.rules
    } == rules_before


def test_resolve_next_agent_同一入力で同一結果を返す() -> None:
    """同じ宣言・結果から何度解決しても同じ名前が返る（決定的）。"""
    result = _turn(_FakeAgent("billing"), ("triage", "billing"))

    assert resolve_next_agent(_SOURCE_MATCHED_POLICY, result) == "frontdesk"
    assert resolve_next_agent(_SOURCE_MATCHED_POLICY, result) == "frontdesk"


# ---------------------------------------------------------------------------
# next_turn_agent: 組み立てヘルパ（FR-4）
# ---------------------------------------------------------------------------


def test_next_turn_agent_発動時はregistryの解決結果をそのまま返す() -> None:
    """上書き発動時は `registry.get(Y)` の戻り値が同一性を保ったまま返る。"""
    override = _FakeAgent("frontdesk")
    last = _FakeAgent("billing")
    registry = _FakeRegistry({"frontdesk": override, "lobby": _FakeAgent("lobby")})
    result = _FakeResult(last, [_FakeHandoffItem("triage", "billing")])

    returned = next_turn_agent(_SOURCE_MATCHED_POLICY, result, registry)

    assert returned is override
    assert returned is not last
    assert registry.calls == ["frontdesk"]


def test_next_turn_agent_非発動時はlast_agentをそのまま返す() -> None:
    """非発動時は `result.last_agent` を registry で正規化せずそのまま返す。"""
    last = _FakeAgent("billing")
    registry = _FakeRegistry({"billing": _FakeAgent("billing"), "frontdesk": _FakeAgent("front")})
    result = _FakeResult(last, [])

    returned = next_turn_agent(_SOURCE_MATCHED_POLICY, result, registry)

    assert returned is last
    assert returned is not registry.agents["billing"]
    assert registry.calls == []


def test_next_turn_agent_禁止のみルール発動時もlast_agentを返す() -> None:
    """禁止のみルールが発動したターンは「上書きなし」のため last_agent へフォールバックする。"""
    policy = NextTurnPolicy(rules={"billing": NextTurnRule(no_handoff_on_arrival=True)})
    last = _FakeAgent("billing")
    registry = _FakeRegistry({"billing": _FakeAgent("billing")})
    result = _FakeResult(last, [_FakeHandoffItem("triage", "billing")])

    returned = next_turn_agent(policy, result, registry)

    assert returned is last
    assert registry.calls == []


def test_next_turn_agent_last_agent属性の欠落はNoneを返す() -> None:
    """フォールバック先の `last_agent` も取得できなければ「開始エージェント決定不能」（None）。"""
    registry = _FakeRegistry({"frontdesk": _FakeAgent("frontdesk")})
    result = _ResultWithoutLastAgent([])

    assert next_turn_agent(_SOURCE_MATCHED_POLICY, result, registry) is None


def test_next_turn_agent_last_agentのアクセス例外でもNoneを返す() -> None:
    """`last_agent` の読み出しが例外を送出しても伝播させず None を返す。"""
    registry = _FakeRegistry({"frontdesk": _FakeAgent("frontdesk")})
    result = _ResultRaisingLastAgent([_FakeHandoffItem("triage", "billing")])

    assert next_turn_agent(_SOURCE_MATCHED_POLICY, result, registry) is None


def test_next_turn_agent_未登録のYはKeyErrorが伝播する() -> None:
    """上書き先 Y が registry に無ければ `KeyError` を握り潰さずそのまま伝播する。"""
    registry = _FakeRegistry({"billing": _FakeAgent("billing")})
    result = _FakeResult(_FakeAgent("billing"), [_FakeHandoffItem("triage", "billing")])

    with pytest.raises(KeyError):
        next_turn_agent(_SOURCE_MATCHED_POLICY, result, registry)


def test_next_turn_agent_registryと結果を変更しない() -> None:
    """ヘルパは読み取りと `registry.get` のみで、登録内容も結果オブジェクトも変更しない。"""
    override = _FakeAgent("frontdesk")
    last = _FakeAgent("billing")
    registry = _FakeRegistry({"frontdesk": override})
    registered_before = dict(registry.agents)
    items = [_FakeHandoffItem("triage", "billing")]
    snapshot = list(items)
    result = _FakeResult(last, items)

    next_turn_agent(_SOURCE_MATCHED_POLICY, result, registry)

    assert registry.agents == registered_before
    assert result.last_agent is last
    assert result.new_items is items
    assert items == snapshot


# ---------------------------------------------------------------------------
# apply_next_turn_policy（FR-1 名前整合検証 + 派生 registry）用のヘルパ
# ---------------------------------------------------------------------------


def _make_registry() -> AgentRegistry:
    """triage -> billing -> tech のハンドオフ構成を持つ registry を作る。

    Returns:
        フェイク builder（`agents` 非依存）を注入した `AgentRegistry`。
    """
    registry = AgentRegistry(agent_builder=FakeAgentBuilder())
    registry.register(AgentSpec(name="triage", instructions="t", handoffs=["billing"]))
    registry.register(AgentSpec(name="billing", instructions="b", handoffs=["tech"]))
    registry.register(AgentSpec(name="tech", instructions="x"))
    return registry


def _rules_snapshot(policy: NextTurnPolicy) -> dict[str, tuple[tuple[Any, ...], ...]]:
    """宣言の内容を比較可能な素の値へ写し取る（適用前後の同一性検証用）。

    Args:
        policy: 写し取る宣言。

    Returns:
        回答者名 -> 各ルールの `(next_agent, no_handoff_on_arrival, source)` の tuple。
    """
    return {
        key: tuple(
            (rule.next_agent, rule.no_handoff_on_arrival, rule.source)
            for rule in _rules_of(policy, key)
        )
        for key in policy.rules
    }


# ---------------------------------------------------------------------------
# apply_next_turn_policy: 名前整合検証（build-time fail-fast）
# ---------------------------------------------------------------------------


def test_apply_next_turn_policy_キーが未登録なら_ValueError() -> None:
    """回答者名 X が registry に無ければ、不在名を含む `ValueError` で fail-fast する。"""
    policy = NextTurnPolicy(rules={"sales": NextTurnRule(next_agent="triage")})

    with pytest.raises(ValueError, match="sales"):
        apply_next_turn_policy(policy, _make_registry())


def test_apply_next_turn_policy_next_agentが未登録なら_ValueError() -> None:
    """次ターン指定 Y が registry に無ければ、不在名を含む `ValueError` で fail-fast する。"""
    policy = NextTurnPolicy(rules={"billing": NextTurnRule(next_agent="sales")})

    with pytest.raises(ValueError, match="sales"):
        apply_next_turn_policy(policy, _make_registry())


def test_apply_next_turn_policy_sourceが未登録なら_ValueError() -> None:
    """到達元条件が registry に無ければ、不在名を含む `ValueError` で fail-fast する。"""
    policy = NextTurnPolicy(rules={"billing": NextTurnRule(next_agent="triage", source="sales")})

    with pytest.raises(ValueError, match="sales"):
        apply_next_turn_policy(policy, _make_registry())


def test_apply_next_turn_policy_登録済みの名前だけなら成功する() -> None:
    """キー / `next_agent` / `source` がすべて登録済みなら検証を通過する。"""
    policy = NextTurnPolicy(rules={"billing": NextTurnRule(next_agent="triage", source="triage")})
    registry = _make_registry()

    derived = apply_next_turn_policy(policy, registry)

    assert derived.names() == registry.names()


# ---------------------------------------------------------------------------
# apply_next_turn_policy: 派生 registry と元 registry の不変性
# ---------------------------------------------------------------------------


def test_apply_next_turn_policy_派生registryは元と別オブジェクト() -> None:
    """戻り値は `registry.clone()` 由来の派生であり、元 registry そのものではない。"""
    registry = _make_registry()

    derived = apply_next_turn_policy(
        NextTurnPolicy(rules={"billing": NextTurnRule(no_handoff_on_arrival=True)}), registry
    )

    assert derived is not registry


def test_apply_next_turn_policy_派生registryは独立にビルドする() -> None:
    """派生は元と独立した registry のため、同名でも別の Agent インスタンスを構築する。"""
    registry = _make_registry()

    derived = apply_next_turn_policy(
        NextTurnPolicy(rules={"billing": NextTurnRule(next_agent="triage")}), registry
    )

    assert derived.get("tech") is not registry.get("tech")


def test_apply_next_turn_policy_空policyでも派生を返す() -> None:
    """no-op 宣言（空 rules）でも検証は通り、同じ登録内容の派生 registry が返る。"""
    registry = _make_registry()

    derived = apply_next_turn_policy(NextTurnPolicy(), registry)

    assert derived is not registry
    assert derived.names() == registry.names()


def test_apply_next_turn_policy_元registryの登録内容を変更しない() -> None:
    """禁止を宣言しても元 registry の登録名・結線は変わらない（禁止が元へ漏れない）。"""
    registry = _make_registry()
    names_before = registry.names()

    apply_next_turn_policy(
        NextTurnPolicy(rules={"billing": NextTurnRule(no_handoff_on_arrival=True)}), registry
    )

    assert registry.names() == names_before
    assert registry.get("triage").handoffs == [registry.get("billing")]
    assert registry.get("billing").handoffs == [registry.get("tech")]


def test_apply_next_turn_policy_宣言を変更しない() -> None:
    """適用は宣言を読むだけで、`rules` の内容を書き換えない。"""
    policy = NextTurnPolicy(
        rules={
            "billing": (
                NextTurnRule(next_agent="triage", no_handoff_on_arrival=True, source="triage"),
                NextTurnRule(no_handoff_on_arrival=True),
            )
        }
    )
    snapshot = _rules_snapshot(policy)

    apply_next_turn_policy(policy, _make_registry())

    assert _rules_snapshot(policy) == snapshot


# ---------------------------------------------------------------------------
# apply_next_turn_policy: freeze 状態との関係
# ---------------------------------------------------------------------------


def test_apply_next_turn_policy_frozenな元registryにも適用できる() -> None:
    """freeze 済みの registry でも合成は派生側へ設置されるため適用できる。"""
    registry = _make_registry()
    registry.freeze()

    derived = apply_next_turn_policy(
        NextTurnPolicy(rules={"billing": NextTurnRule(no_handoff_on_arrival=True)}), registry
    )

    assert derived is not registry
    assert derived.names() == registry.names()


def test_apply_next_turn_policy_元がunfrozenなら派生もunfrozen() -> None:
    """unfrozen な元 registry の派生は unfrozen で返り、追加登録などの変更操作ができる。"""
    registry = _make_registry()

    derived = apply_next_turn_policy(
        NextTurnPolicy(rules={"billing": NextTurnRule(no_handoff_on_arrival=True)}), registry
    )
    derived.register(AgentSpec(name="sales", instructions="s"))

    assert "sales" in derived.names()
    assert "sales" not in registry.names()


def test_apply_next_turn_policy_元がfrozenなら派生もfrozen() -> None:
    """完全性（freeze）は派生へ引き継ぐ（推奨フローの終点が変更可能にならない）。"""
    registry = _make_registry()
    registry.freeze()

    derived = apply_next_turn_policy(
        NextTurnPolicy(rules={"billing": NextTurnRule(no_handoff_on_arrival=True)}), registry
    )

    with pytest.raises(RegistryFrozenError):
        derived.register(AgentSpec(name="evil", instructions="e"))
    with pytest.raises(RegistryFrozenError):
        derived.update(AgentSpec(name="triage", instructions="t", handoffs=["billing"]))


def test_apply_next_turn_policy_frozenな派生でも禁止の結線は残る() -> None:
    """freeze は read-only 経路に影響しないため、frozen な派生でも合成は設置済みのまま。"""
    registry = _make_registry()
    registry.freeze()

    derived = apply_next_turn_policy(
        NextTurnPolicy(rules={"billing": NextTurnRule(no_handoff_on_arrival=True)}), registry
    )

    # 判定表と記録ストアは freeze 後も保持される（実結線の振る舞いは L2 で pin する）。
    wiring = derived._next_turn
    assert wiring is not None
    assert wiring.gated == frozenset({"billing"})
    assert ("triage", "billing") in wiring.arrivals


# ---------------------------------------------------------------------------
# apply_next_turn_policy: ファクトリ登録との関係（禁止の結線可否）
# ---------------------------------------------------------------------------


def _make_registry_with_factory(name: str) -> AgentRegistry:
    """triage -> billing -> tech 構成のうち 1 つをファクトリ登録にした registry を作る。

    Args:
        name: ファクトリ登録に置き換えるエージェント名。

    Returns:
        指定名だけがファクトリ登録の `AgentRegistry`。

    Raises:
        ValueError: 構成に無い名前を指定した場合。
    """
    specs = {
        "triage": AgentSpec(name="triage", instructions="t", handoffs=["billing"]),
        "billing": AgentSpec(name="billing", instructions="b", handoffs=["tech"]),
        "tech": AgentSpec(name="tech", instructions="x"),
    }
    if name not in specs:
        raise ValueError(f"unknown agent for factory registration: {name}")

    registry = AgentRegistry(agent_builder=FakeAgentBuilder())
    for spec_name, spec in specs.items():
        if spec_name == name:
            registry.register_factory(spec_name, lambda _r, n=spec_name: FakeAgent(name=n))
        else:
            registry.register(spec)
    return registry


def test_apply_next_turn_policy_禁止対象がファクトリ登録なら_ValueError() -> None:
    """X がファクトリ登録だと出辺へゲートを合成できないため build 時に fail-fast する。"""
    registry = _make_registry_with_factory("billing")
    policy = NextTurnPolicy(
        rules={"billing": NextTurnRule(no_handoff_on_arrival=True, source="triage")}
    )

    with pytest.raises(ValueError, match="billing") as exc:
        apply_next_turn_policy(policy, registry)

    assert "禁止対象" in str(exc.value)


def test_apply_next_turn_policy_sourceで明示指定した到達元がファクトリ登録なら_ValueError() -> None:
    """`source` の明示指定は「その遷移元の記録が必須」の宣言のため、結線できなければ拒否する。"""
    registry = _make_registry_with_factory("triage")
    policy = NextTurnPolicy(
        rules={"billing": NextTurnRule(no_handoff_on_arrival=True, source="triage")}
    )

    with pytest.raises(ValueError, match="triage") as exc:
        apply_next_turn_policy(policy, registry)

    assert "明示指定" in str(exc.value)


def test_apply_next_turn_policy_包括禁止は無関係なファクトリ登録を拒否しない(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """候補の展開は over-approximation のため、包括ルールでは警告に留めて適用を通す。

    ファクトリ登録が禁止対象へ handoff を持つかは build するまで分からず、無関係な
    ファクトリ登録があるだけで宣言全体を落とすのは過剰なため。
    """
    registry = _make_registry_with_factory("tech")
    policy = NextTurnPolicy(rules={"billing": NextTurnRule(no_handoff_on_arrival=True)})

    with caplog.at_level(logging.WARNING, logger=NEXT_TURN_LOGGER_NAME):
        derived = apply_next_turn_policy(policy, registry)

    # spec 登録の遷移元（triage）経由の禁止は従来どおり結線される。
    wiring = derived._next_turn
    assert wiring is not None
    assert wiring.gated == frozenset({"billing"})
    assert ("triage", "billing") in wiring.arrivals
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "tech" in warnings[0].getMessage()
    assert "billing" in warnings[0].getMessage()


def test_apply_next_turn_policy_ファクトリ登録が無ければ警告しない(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """spec 登録のみの registry では、包括禁止でも警告は出ない（ノイズを出さない）。"""
    policy = NextTurnPolicy(rules={"billing": NextTurnRule(no_handoff_on_arrival=True)})

    with caplog.at_level(logging.WARNING, logger=NEXT_TURN_LOGGER_NAME):
        apply_next_turn_policy(policy, _make_registry())

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_apply_next_turn_policy_禁止なしならファクトリ登録でも適用できる() -> None:
    """次ターン指定のみのルールは `registry.get` で解決するだけのためファクトリ登録でも通す。"""
    registry = _make_registry_with_factory("billing")
    policy = NextTurnPolicy(rules={"billing": NextTurnRule(next_agent="triage")})

    derived = apply_next_turn_policy(policy, registry)

    assert derived.names() == registry.names()
    result = _turn(_FakeAgent("billing"), ("triage", "billing"))
    assert resolve_next_agent(policy, result) == "triage"
    assert next_turn_agent(policy, result, derived).name == "triage"


def test_apply_next_turn_policy_ファクトリ登録があっても禁止対象が_spec登録なら通る() -> None:
    """禁止に関与しないファクトリ登録（`source` 指定で候補外）は拒否しない。"""
    registry = _make_registry_with_factory("tech")
    policy = NextTurnPolicy(
        rules={"billing": NextTurnRule(no_handoff_on_arrival=True, source="triage")}
    )

    derived = apply_next_turn_policy(policy, registry)

    wiring = derived._next_turn
    assert wiring is not None
    assert wiring.gated == frozenset({"billing"})


def test_apply_next_turn_policy_spec登録のみなら禁止を宣言できる() -> None:
    """ファクトリ登録が無い registry では従来どおり禁止を結線できる（回帰ガード）。"""
    registry = _make_registry()
    policy = NextTurnPolicy(rules={"billing": NextTurnRule(no_handoff_on_arrival=True)})

    derived = apply_next_turn_policy(policy, registry)

    wiring = derived._next_turn
    assert wiring is not None
    assert wiring.gated == frozenset({"billing"})


def test_apply_next_turn_policy_警告はその到達元から効かない禁止対象だけを名指しする(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """警告が名指しするのは、当該ファクトリ登録を遷移元とする流入エッジの禁止対象だけ。

    ファクトリ登録 tech から流入するエッジは `(tech, billing)` のみで、`(tech, triage)` は
    存在しない（triage への到達元は billing / triage のみ）。tech の到達記録を合成できない
    ことで実際に禁止が効かなくなりうるのは billing だけであり、triage の禁止は tech とは
    無関係に spec 登録の遷移元経由で効く。したがって警告に triage を含めてはならない
    （禁止対象の全体 = gated をそのまま列挙すると、無関係な triage まで名指しして
    利用者に不要な調査を強いる）。
    """
    registry = _make_registry_with_factory("tech")
    policy = NextTurnPolicy(
        rules={
            "billing": NextTurnRule(no_handoff_on_arrival=True),
            "triage": [
                NextTurnRule(next_agent="billing", source="tech"),
                NextTurnRule(no_handoff_on_arrival=True),
            ],
        }
    )

    with caplog.at_level(logging.WARNING, logger=NEXT_TURN_LOGGER_NAME):
        derived = apply_next_turn_policy(policy, registry)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "tech" in message
    assert "billing" in message
    assert "triage" not in message
    # 引数レベルでも固定する（部分一致だけだと余分な要素の混入を「triage が無い」に
    # 依存して検出する形になるため）。
    assert warnings[0].args == (["tech"], ["billing"])

    # 警告の対象を絞っても、禁止対象そのもの（gated）は狭めない。
    wiring = derived._next_turn
    assert wiring is not None
    assert wiring.gated == frozenset({"billing", "triage"})


def test_apply_next_turn_policy_警告はその到達元から効かない禁止対象をすべて名指しする(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """名指しは「余分を含まない」だけでなく「もれなく挙げる」方向も満たす。

    ファクトリ登録 tech を遷移元候補に持つ到達エッジが 2 本ある構成（billing / triage の
    どちらも到達元を限定しない禁止ルール）では、tech の到達記録を合成できないことで
    禁止が効かなくなりうる対象は billing と triage の 2 つになる。1 つでも落とすと
    利用者は残りを見落とすため、列そのものを完全一致で固定する。

    上の「だけを名指しする」テストは期待値が 1 要素のため、切り詰め変異（`[:1]` /
    `[min(...)]` / `sorted(...)[-1:]` 等）は 1 要素リストを不変に写して素通りする。
    本テストが過小側の変異を kill する担当である（2 方向で 1 本ずつ）。
    """
    registry = _make_registry_with_factory("tech")
    policy = NextTurnPolicy(
        rules={
            "billing": NextTurnRule(no_handoff_on_arrival=True),
            "triage": NextTurnRule(no_handoff_on_arrival=True),
        }
    )

    with caplog.at_level(logging.WARNING, logger=NEXT_TURN_LOGGER_NAME):
        derived = apply_next_turn_policy(policy, registry)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    # 完全一致で固定する（部分一致では「1 件だけ挙げる」退行を検出できない）。
    # 順序も含めて固定するのは、実装が `sorted` で決定的な列を作る契約のため。
    assert warnings[0].args == (["tech"], ["billing", "triage"])

    wiring = derived._next_turn
    assert wiring is not None
    assert wiring.gated == frozenset({"billing", "triage"})
