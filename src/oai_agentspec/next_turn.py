"""次ターン開始エージェントの宣言的上書き（`NextTurnRule` / `NextTurnPolicy`・agents 非依存）。

「ハンドオフ遷移を経てエージェント X が回答を終えたら、次ターンは Y から開始する」という
上書きをハンドオフグラフと同じ宣言レベルで固定する。あわせて「ハンドオフで X に到達した
ターンでは X の全 handoff を無効化する」到達時ハンドオフ禁止をルール単位で opt-in できる。

`NextTurnPolicy.rules` は「回答者名 X -> 単一ルール / ルールの列 / 次ターン指定のみの str 略記」
の Mapping で、`__post_init__` が `dict(...)` へ正規化・検証したうえで値位置をルールの tuple へ
揃え、`MappingProxyType` で不変化する（`FailsafePolicy.handlers` と同一方針）。効果のない宣言・
発動ルールの選定が一意にならない宣言は build-time `ValueError` で fail-fast する。

X のエントリから発動ルールを選ぶ規則は「到達の遷移元と一致する `source` を持つルール ->
`source` を持たない包括ルール -> 発動ルールなし」であり、この一意性を build 時に保証するために
同一 `source` の重複と包括ルール 2 件以上を拒否する。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .constants import NEXT_TURN_LOGGER_NAME

if TYPE_CHECKING:
    from .registry import AgentRegistry

logger = logging.getLogger(NEXT_TURN_LOGGER_NAME)


def _validate_name(value: Any, label: str) -> None:
    """エージェント名として使う値を検証する（str かつ非空）。

    Args:
        value: 検証対象の値。
        label: エラーメッセージに載せる項目名。

    Raises:
        ValueError: str でない、または空文字の場合。
    """
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a str, got {type(value).__name__!r}")
    if not value:
        raise ValueError(f"{label} must not be empty")


@dataclass(frozen=True)
class NextTurnRule:
    """次ターン開始エージェントの上書きルール 1 件（frozen）。

    次ターン指定（`next_agent`）と到達時ハンドオフ禁止（`no_handoff_on_arrival`）のいずれも
    持たない宣言は効果が無いため build-time `ValueError` で拒否する。

    Attributes:
        next_agent: 次ターンの開始エージェント名（Y）。X と同名も可（継続の明示固定）。
            None は「次ターン指定を持たない」（禁止のみのルール）。
        no_handoff_on_arrival: 到達時ハンドオフ禁止の opt-in。True で、ハンドオフによる
            X 到達以降そのターン中は X の全 handoff が無効化される。
        source: 到達元条件。指定した遷移元からのハンドオフ到達に限定する（1 ルールに 1 名）。
            None は到達元不問の包括ルール。
    """

    next_agent: str | None = None
    no_handoff_on_arrival: bool = False
    source: str | None = None

    def __post_init__(self) -> None:
        """build-time 検証を行い、不正・無効果な宣言を `ValueError` で fail-fast する。

        Raises:
            ValueError: `next_agent` / `source` が str でない / 空文字の場合、または
                次ターン指定も到達時ハンドオフ禁止も持たない（効果のない）宣言の場合。
        """
        if self.next_agent is not None:
            _validate_name(self.next_agent, "next_agent")
        if self.source is not None:
            _validate_name(self.source, "source")

        if self.next_agent is None and not self.no_handoff_on_arrival:
            raise ValueError(
                "NextTurnRule must declare next_agent or no_handoff_on_arrival=True "
                "(a rule with neither has no effect)"
            )


@dataclass(frozen=True)
class NextTurnPolicy:
    """回答者名ごとの次ターン上書きルールの宣言（frozen）。

    Attributes:
        rules: 回答者名 X から「単一 `NextTurnRule` / `NextTurnRule` の列 / 次ターン名の
            str 略記」へのマッピング。`__post_init__` の検証後に値位置は宣言順の
            `tuple[NextTurnRule, ...]` へ正規化され、全体が `MappingProxyType` へ
            差し替えられて不変化する（事後注入・呼び出し側が渡した元 dict / 元 list の
            事後変更は反映されない）。空の `rules` は no-op 宣言として許容する。
    """

    rules: Mapping[str, NextTurnRule | Sequence[NextTurnRule] | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """build-time 検証と正規化を行い、`rules` を不変化する。

        `rules` は先に `dict` へ正規化してから検証し、**検証したその dict を**
        `MappingProxyType` で包んで格納する（検証対象と格納対象が食い違わない）。

        Raises:
            ValueError: キーが str でない / 空文字の場合、値位置が対応しない型の場合、
                空列の場合、同一 X のルール列内で `source` が重複する場合、または
                到達元条件なしの包括ルールが 2 件以上ある場合。
        """
        normalized: dict[str, tuple[NextTurnRule, ...]] = {}
        for key, value in dict(self.rules).items():
            _validate_name(key, "rules key")
            normalized[key] = self._normalize_entry(key, value)

        object.__setattr__(self, "rules", MappingProxyType(normalized))

    @staticmethod
    def _normalize_entry(key: str, value: Any) -> tuple[NextTurnRule, ...]:
        """X のエントリ 1 件をルールの tuple へ正規化し、選定の一意性を検証する。

        Args:
            key: 回答者名 X。
            value: 値位置の宣言（str 略記 / 単一 `NextTurnRule` / `NextTurnRule` の列）。

        Returns:
            宣言順に並べたルールの tuple。

        Raises:
            ValueError: 値位置が対応しない型 / 空列 / `source` の重複 /
                包括ルールが 2 件以上の場合。
        """
        if isinstance(value, str):
            rules: tuple[NextTurnRule, ...] = (NextTurnRule(next_agent=value),)
        elif isinstance(value, NextTurnRule):
            rules = (value,)
        elif isinstance(value, Sequence):
            rules = tuple(value)
            if not rules:
                raise ValueError(
                    f"rules entry for {key!r} must declare at least one NextTurnRule "
                    "(an empty sequence has no effect)"
                )
            for rule in rules:
                if not isinstance(rule, NextTurnRule):
                    raise ValueError(
                        f"rules entry for {key!r} must contain NextTurnRule instances, "
                        f"got {type(rule).__name__!r}"
                    )
        else:
            raise ValueError(
                f"rules value for {key!r} must be a NextTurnRule, a sequence of NextTurnRule, "
                f"or a next-agent name (str), got {type(value).__name__!r}"
            )

        seen: set[str | None] = set()
        for rule in rules:
            if rule.source in seen:
                where = f"source {rule.source!r}" if rule.source is not None else "no source"
                raise ValueError(
                    f"rules entry for {key!r} declares duplicate rules with {where} "
                    "(the triggering rule would not be uniquely selected)"
                )
            seen.add(rule.source)

        return rules


def _select_rule(rules: Sequence[NextTurnRule], source: str) -> NextTurnRule | None:
    """到達の遷移元から発動ルールを 1 件選ぶ。

    選定規則は「遷移元と一致する `source` を持つルール -> `source` を持たない包括ルール ->
    発動ルールなし」の順。候補の一意性は `NextTurnPolicy.__post_init__` が build 時に
    保証しているため、各段の最初の一致をそのまま採用する。

    Args:
        rules: X に宣言されたルールの列（宣言順）。
        source: X への到達の遷移元エージェント名。

    Returns:
        発動ルール。どの段でも選べなければ None。
    """
    for rule in rules:
        if rule.source == source:
            return rule
    for rule in rules:
        if rule.source is None:
            return rule
    return None


def resolve_next_agent(policy: NextTurnPolicy, result: Any) -> str | None:
    """run 完了結果から次ターン開始エージェント名を解決する（副作用なし・決定的）。

    発動条件は「ターン内にハンドオフ遷移が 1 件以上観測される」かつ「最終回答者名が
    `policy.rules` のキー X と一致する」の AND で、成立したときのみ **X への最後の到達**の
    遷移元で発動ルールを選ぶ。発動ルールが次ターン指定（`next_agent`）を持たない
    （禁止のみの）場合は「上書きなし」として None を返し、次ターン指定を持つ包括ルールへ
    フォールバックすることはない（選定を先に行い、その結果だけを見る）。

    判定材料の読み取りは `_adapters` の防御的な観測抽出に委ねるため、`last_agent` /
    `new_items` の属性欠落・アクセス例外はいずれも例外を送出せず「上書きなし」へ倒れる。

    Args:
        policy: 次ターン上書きの宣言。
        result: run 完了結果（`last_agent` / `new_items` を持つ前提）。

    Returns:
        上書き先のエージェント名（Y）。上書きが発動しない場合は None。
    """
    from . import _adapters

    observation = _adapters.extract_turn_observation(result)
    last_agent = observation.last_agent
    if last_agent is None or not observation.handoffs:
        return None

    entry = policy.rules.get(last_agent)
    if entry is None:
        return None

    # X への到達だけを取り出し、その最後の遷移元で選定する（他エージェントへの到達や、
    # X 到達より後に起きた別の遷移は選定に影響しない）。
    arrivals = [src for src, dst in observation.handoffs if dst == last_agent]
    if not arrivals:
        return None

    # `__post_init__` が値位置をルールの tuple へ正規化済みのため、そのまま列として読む。
    rule = _select_rule(tuple(entry), arrivals[-1])
    if rule is None:
        return None
    return rule.next_agent


def next_turn_agent(policy: NextTurnPolicy, result: Any, registry: AgentRegistry) -> Any | None:
    """次ターンの開始エージェントを 1 回の呼び出しで決める（解決 + registry 解決 + 継続）。

    上書きが発動した場合は `registry.get(Y)` の戻り値をそのまま返し、Y が未登録なら
    registry の `KeyError` を握り潰さず伝播する。上書きが発動しない場合は
    `result.last_agent` をそのまま返す（registry を経由した正規化はしないため、SDK が返した
    実体の同一性が保たれる）。`last_agent` も取得できない場合のみ None を返す。

    戻り値の None は「開始エージェント決定不能」であり、`resolve_next_agent` の
    None（上書きなし＝正常系）とは意味が異なる。

    Args:
        policy: 次ターン上書きの宣言。
        result: run 完了結果（`last_agent` / `new_items` を持つ前提）。
        registry: 上書き先 Y を解決する registry（`apply_next_turn_policy` の派生 registry）。

    Returns:
        次ターンの開始エージェント（不透明値）。決定できない場合は None。

    Raises:
        KeyError: 上書き先 Y が registry に登録されていない場合（registry から伝播する）。
    """
    from . import _adapters

    name = resolve_next_agent(policy, result)
    if name is not None:
        return registry.get(name)
    return _adapters.read_last_agent(result)


def _declared_names(policy: NextTurnPolicy) -> set[str]:
    """宣言中に現れる全エージェント名（キー / `next_agent` / `source`）を集める。

    Args:
        policy: 次ターン上書きの宣言。

    Returns:
        宣言に現れるエージェント名の集合。
    """
    names: set[str] = set()
    for key, entry in policy.rules.items():
        names.add(key)
        for rule in tuple(entry):
            if rule.next_agent is not None:
                names.add(rule.next_agent)
            if rule.source is not None:
                names.add(rule.source)
    return names


def _expand_prohibition_table(
    policy: NextTurnPolicy, known: list[str]
) -> tuple[frozenset[tuple[str, str]], frozenset[str]]:
    """到達時ハンドオフ禁止の判定表を build 時に静的展開する。

    登録済みの全エージェント名を到達元候補として、`resolve_next_agent` と同一の選定規則
    （一致 `source` -> 包括 -> なし）で発動ルールを決め、禁止を持つ組み合わせだけを残す。
    これにより実行時に必要なのは記録の追記と参照だけになる。

    Args:
        policy: 次ターン上書きの宣言。
        known: registry に登録済みのエージェント名（到達元の候補）。

    Returns:
        記録を前置する流入エッジ `(遷移元, X)` の集合と、出辺へゲートを合成する X の集合。
    """
    arrivals: set[tuple[str, str]] = set()
    gated: set[str] = set()
    for key, entry in policy.rules.items():
        rules = tuple(entry)
        for source in known:
            rule = _select_rule(rules, source)
            if rule is not None and rule.no_handoff_on_arrival:
                arrivals.add((source, key))
                gated.add(key)
    return frozenset(arrivals), frozenset(gated)


def _explicit_prohibition_sources(policy: NextTurnPolicy) -> set[str]:
    """禁止を宣言したルールが `source` で明示指定した到達元名を集める。

    包括ルール（`source` なし）は「登録済みの全名を到達元候補として展開する」だけで、
    利用者がその遷移元を意図して指定したわけではない。両者を区別するため、明示指定だけを
    集めて検証の対象にする。

    Args:
        policy: 次ターン上書きの宣言。

    Returns:
        禁止を宣言したルールの `source` に現れる名前の集合。
    """
    return {
        rule.source
        for entry in policy.rules.values()
        for rule in tuple(entry)
        if rule.no_handoff_on_arrival and rule.source is not None
    }


def _validate_prohibition_wiring(
    policy: NextTurnPolicy,
    registry: AgentRegistry,
    arrivals: frozenset[tuple[str, str]],
    gated: frozenset[str],
) -> None:
    """到達時ハンドオフ禁止の合成を載せられない宣言を build 時に拒否する。

    factory 登録のエージェントは `registry.get()` が factory の戻り値をそのまま返し、
    spec からの結線（`_wire`）を通らない。そのため禁止の合成（流入エッジへの到達記録・
    出辺へのゲート）を差し込む口が無く、宣言だけが受理されて禁止が効かない silent failure に
    なる。次の 2 つは禁止が確実に効かないため `ValueError` で fail-fast する。

    1. 禁止対象 X 自身が factory 登録（X の出辺へゲートを載せられない）。
    2. 禁止を宣言したルールが `source` で factory 登録名を**明示指定**している
       （その遷移元からの到達を記録する必要があるのに記録を載せられない）。

    包括ルールの到達元候補に factory 登録が含まれるだけの場合は拒否しない。候補の展開は
    登録済みの全名に対する静的な over-approximation であり、その factory 登録が X への
    handoff を実際に持つとは限らない（factory の中身は build するまで分からないため、
    エッジの有無を宣言から判定できない）。無関係な factory 登録があるだけで禁止の宣言全体を
    落とすのは過剰なため、`logger.warning` で 1 回警告して適用は通す。この警告が名指しする
    禁止対象は、当該 factory 登録を遷移元に持つ流入エッジの遷移先だけに絞る（到達元限定の
    ルールで禁止が付かない X は、他の遷移元経由で禁止が効いていても名指ししない）。

    検証対象は禁止の実現に spec 登録が必要な名前だけで、次ターン指定のみのルール
    （`registry.get(Y)` で解決するだけ）は factory 登録でも従来どおり通す。

    Args:
        policy: 次ターン上書きの宣言。
        registry: 適用元の registry。
        arrivals: 記録を前置する流入エッジ `(遷移元, X)` の集合。
        gated: 出辺へゲートを合成する X の集合。

    Raises:
        ValueError: 禁止対象 X が factory 登録の場合、または禁止を宣言したルールが
            `source` で factory 登録名を明示指定している場合。
    """
    factories = set(registry._factories)  # noqa: SLF001 - registry の内部状態の参照
    if not factories or not arrivals:
        return

    explicit = _explicit_prohibition_sources(policy) & factories
    problems = [
        f"{name!r}（禁止対象: 出辺へゲートを合成できません）" for name in sorted(gated & factories)
    ]
    problems += [
        f"{name!r}（source で明示指定された到達元: 到達記録を合成できません）"
        for name in sorted(explicit)
    ]
    if problems:
        raise ValueError(
            "到達時ハンドオフ禁止はファクトリ登録のエージェントには結線できません: "
            + "、".join(problems)
            + "（ファクトリ登録は registry.get() がファクトリの戻り値をそのまま返し spec からの"
            "結線を通らないため、宣言だけが受理されて禁止が効きません。"
            "該当エージェントを AgentSpec 登録へ変更するか、"
            "当該ルールの no_handoff_on_arrival を外してください）"
        )

    # explicit は上の raise で既に落ちているため、ここでは常に空（差集合は不要）。
    implicit = {src for src, _ in arrivals} & factories
    if implicit:
        affected = sorted({dst for src, dst in arrivals if src in implicit})
        logger.warning(
            "next-turn policy: 包括ルールの到達元候補にファクトリ登録のエージェント %s が"
            "含まれます（当該ファクトリ登録からの到達で効かなくなりうる禁止対象: %s）。"
            "これらのエージェントが実際に禁止対象へ handoff する場合、到達記録を合成できない"
            "ためそのターンの禁止は効きません。確実に効かせるには当該エージェントを "
            "AgentSpec 登録へ変更するか、ルールの source で到達元を spec 登録のエージェントに"
            "限定してください",
            sorted(implicit),
            affected,
        )


def apply_next_turn_policy(policy: NextTurnPolicy, registry: AgentRegistry) -> AgentRegistry:
    """宣言の名前整合を検証し、到達時ハンドオフ禁止を結線した派生 registry を返す。

    宣言中の全エージェント名（キー X / `next_agent` / `source`）を registry の登録名と
    突合し、不在があれば不在名を挙げて `ValueError` で fail-fast する。検証後は
    `registry.clone()` の派生に対してのみ判定表と到達記録ストアを設置するため、**元 registry は
    一切変更されない**（frozen な registry にも適用できる）。

    **派生 registry の freeze 状態は元 registry から引き継ぐ**（元が frozen なら派生も frozen で
    返る）。docs / examples は「実行には派生 registry を使う」ことを前提にするため、引き継がないと
    完全性を固めたデプロイでも推奨フローの終点が変更可能な registry になる。判定表と記録ストアの
    設置は freeze ガード付きの内部プリミティブを通るため、設置を終えてから freeze する。
    freeze は read-only 経路（`get` 等）に影響しないため、禁止の結線は frozen な派生でも働く。

    禁止を 1 件も宣言しない policy（次ターン指定のみ・空 rules）では合成を設置しないため、
    派生 registry の結線は従来経路と同一になる。

    **到達時ハンドオフ禁止が確実に効かない宣言は `ValueError` で拒否する**。factory 登録は
    spec からの結線を通らず合成の差し込み口が無いため、(1) 禁止対象 X が factory 登録の場合と
    (2) 禁止を宣言したルールが `source` で factory 登録名を明示指定している場合は、宣言だけが
    受理されて禁止が効かない状態になる。包括ルールの到達元候補に factory 登録が含まれるだけの
    場合は（候補の展開が over-approximation で実エッジの有無を判定できないため）拒否せず
    `logger.warning` で警告する。次ターン指定のみのルールは factory 登録でも通す。

    **判定表を持つ registry への適用は `ValueError` で拒否する**。拒否の条件は「適用先が
    到達時ハンドオフ禁止を結線済みであること」であり、本関数の戻り値かどうかではない
    （禁止を 1 件も宣言しない policy の派生は判定表を持たないため、その派生への適用は通る）。
    判定表を持つ registry へ重ねると、禁止を含む policy では判定表の上書きで先行 policy の禁止が
    黙って消え、禁止を含まない policy では clone 継承で先行 policy の禁止が残るという非対称な
    食い違いが生じる（宣言と結線が一致しなくなる）。禁止を含む複数の宣言を合成したい場合は
    1 つの `NextTurnPolicy` にまとめ、判定表を持たない元 registry から適用し直す。

    Args:
        policy: 次ターン上書きの宣言。
        registry: 適用元の registry（変更されない）。

    Returns:
        到達時ハンドオフ禁止を結線した派生 `AgentRegistry`。

    Raises:
        ValueError: 宣言中のエージェント名が registry に登録されていない場合、禁止対象 X が
            factory 登録 / 禁止を宣言したルールが `source` で factory 登録名を明示指定して
            いる場合、または registry が既に判定表（到達時ハンドオフ禁止の結線）を保持して
            いる場合。
    """
    from . import _adapters

    if registry._next_turn is not None:  # noqa: SLF001 - registry の内部状態の参照
        raise ValueError(
            "next-turn policy が適用済みの registry には再適用できません"
            "（重ねると先行 policy の禁止が消える / 残るという食い違いが生じます。"
            "複数の宣言は 1 つの NextTurnPolicy にまとめ、"
            "適用元の registry から適用し直してください）"
        )

    known = registry.names()
    missing = sorted(_declared_names(policy) - set(known))
    if missing:
        raise ValueError(
            f"next-turn policy が参照する未登録のエージェント名: {missing} （登録済み: {known}）"
        )

    arrivals, gated = _expand_prohibition_table(policy, known)
    _validate_prohibition_wiring(policy, registry, arrivals, gated)

    derived = registry.clone()
    if arrivals:
        derived._install_next_turn_state(  # noqa: SLF001 - registry の内部プリミティブへの委譲
            _adapters.NextTurnWiring(arrivals=arrivals, gated=gated, store=_adapters.ArrivalStore())
        )
    # 完全性（freeze）は派生へ引き継ぐ。設置は freeze ガード付きプリミティブを通るため、
    # 設置を終えてから freeze する。
    if registry._frozen:  # noqa: SLF001 - registry の内部状態の参照
        derived.freeze()
    return derived
