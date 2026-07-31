"""Next-Turn Agent Override の SDK 結合窓口（観測抽出・到達記録・合成の単一窓口・NFR-1）。

責務は 2 つある。

1. **観測抽出**: 生の run 完了結果（`RunResult` 等）を `_adapters` 内で消費し、次ターン開始
   エージェントの解決に必要な判定材料だけを plain な `TurnObservation`（最終回答者名 /
   `(遷移元, 遷移先)` 列）として取り出す。SDK 型を `_adapters` 外へ出さず、宣言層
   （`next_turn.py`）は本モジュールの戻り値のみを扱う。
2. **到達時ハンドオフ禁止の実現形**: 到達記録ストア（`ArrivalStore`）と、SDK 公式拡張点へ
   前置 / AND 合成する callable のファクトリ（`make_arrival_recorder` / `make_arrival_gate`）。
   エージェント実体の複製・書き換えは行わず、`on_handoff` への記録の前置合成と `is_enabled`
   への記録参照ゲートの合成だけで構成する。

抽出は `observe_run_result`（`routing.py`）と同型のダックタイピングで行い、handoff アイテムの
判定は `source_agent` / `target_agent` **属性の有無**で行う（`type` 文字列のリテラルに依存
しない）。加えて属性欠落だけでなく **属性アクセス自体が送出する例外**も捕捉し、読み取り先ごとに
独立して安全側（`last_agent=None` / 観測列は空 / 当該アイテムのみスキップ）へ倒して
`logger.debug(exc_info=True)` に記録する（Failsafe の `_derive_last_agent` と同一方針）。

記録とゲートは SDK が呼ぶ公式 callback の内側で行う同期の読み書きのみであり、`await` を挟む
独自の実行ループ・`Runner` 参照は持たない（build-don't-run の逸脱に当たらない）。
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final
from weakref import WeakKeyDictionary

from agents.exceptions import UserError

from ..constants import NEXT_TURN_LOGGER_NAME

if TYPE_CHECKING:
    from ..spec import HandoffConfig

logger = logging.getLogger(NEXT_TURN_LOGGER_NAME)

_MISSING: Final[object] = object()
"""属性が欠落している / 読み出しに失敗したことを表す内部センチネル。

`None` は「属性はあるが値が None」という正当な観測値と区別できないため、判定は
`is _MISSING` の同一性で行う。
"""


@dataclass(frozen=True)
class TurnObservation:
    """1 ターンの run 完了結果から抽出した plain 観測（SDK 型を含まない）。

    Attributes:
        last_agent: 最終回答者のエージェント名。取得できない場合は None。
        handoffs: 観測されたハンドオフ遷移の `(遷移元, 遷移先)` 列（観測順・並べ替えなし）。
    """

    last_agent: str | None
    handoffs: tuple[tuple[str, str], ...]


def _read_attr(obj: Any, name: str) -> Any:
    """属性 1 つを防御的に読む（欠落・読み出し例外の双方を吸収する）。

    `getattr` の既定値は `AttributeError` しか吸収しないため、property が任意の例外を
    送出する実装（release 後アクセス等）に備えて `try/except` で包む。読み出しに失敗した
    場合は `logger.debug(exc_info=True)` に記録して `_MISSING` を返す。

    Args:
        obj: 読み取り対象のオブジェクト。
        name: 読み取る属性名。

    Returns:
        読み取れた属性値。属性が無い / 読み出しが例外を送出した場合は `_MISSING`。
    """
    try:
        return getattr(obj, name, _MISSING)
    except Exception:
        logger.debug(
            "next-turn observation: reading %s from %s failed",
            name,
            type(obj).__name__,
            exc_info=True,
        )
        return _MISSING


def _agent_name(agent: Any) -> str | None:
    """エージェント代表値から名前を防御的に取り出す。

    Args:
        agent: `name` 属性を持つ前提のエージェント（防御的に読む）。

    Returns:
        エージェント名。取得できない場合は None。
    """
    name = _read_attr(agent, "name")
    if name is _MISSING or name is None:
        return None
    if isinstance(name, str):
        return name
    # str 以外の name も受けるが（`observe_run_result` と同じ寛容さ）、`__str__` が例外を
    # 送出する値で解決全体を落とさないよう変換も防御の内側で行う。
    try:
        return str(name)
    except Exception:
        logger.debug(
            "next-turn observation: converting name of %s to str failed",
            type(name).__name__,
            exc_info=True,
        )
        return None


def _read_items(result: Any) -> tuple[Any, ...]:
    """run 完了結果からアイテム列を防御的に取り出す。

    Args:
        result: run 完了結果（`new_items` を持つ前提）。

    Returns:
        アイテムの tuple。属性欠落・読み出し例外・反復不能のいずれの場合も空 tuple。
    """
    items = _read_attr(result, "new_items")
    if items is _MISSING or items is None:
        return ()
    try:
        return tuple(items)
    except Exception:
        logger.debug(
            "next-turn observation: iterating new_items of %s failed",
            type(result).__name__,
            exc_info=True,
        )
        return ()


def _read_handoff(item: Any) -> tuple[str, str] | None:
    """アイテム 1 件をハンドオフ遷移として読む。

    判定は `source_agent` / `target_agent` 属性の有無で行い、`type` 文字列のリテラルには
    依存しない。どちらかの読み出しが失敗した / 名前を解決できないアイテムは
    ハンドオフとみなさずスキップする。

    Args:
        item: run 完了結果の `new_items` の 1 要素。

    Returns:
        `(遷移元, 遷移先)` の組。ハンドオフとして読めない場合は None。
    """
    source = _read_attr(item, "source_agent")
    if source is _MISSING:
        return None
    target = _read_attr(item, "target_agent")
    if target is _MISSING:
        return None

    src = _agent_name(source)
    dst = _agent_name(target)
    if src is None or dst is None:
        return None
    return (src, dst)


def read_last_agent(result: Any) -> Any | None:
    """run 完了結果から最終回答者の**実体**を防御的に読む（名前ではない）。

    `extract_turn_observation` が返すのは名前（str）であり、次ターンの `Runner.run` へ
    そのまま渡せる Agent 実体は得られない。`next_turn_agent` の非発動時フォールバックが
    「`result.last_agent` をそのまま返す（registry で正規化しない）」ため、実体を読む窓口を
    別に設ける。属性欠落・読み出し例外はいずれも None へ倒す（`logger.debug` に記録）。

    Args:
        result: run 完了結果（`last_agent` を持つ前提・ダックタイピングで読む）。

    Returns:
        最終回答者の実体（不透明値）。属性が無い / 読み出しが例外を送出した / 値が None の
        場合は None。
    """
    agent = _read_attr(result, "last_agent")
    return None if agent is _MISSING else agent


def extract_turn_observation(result: Any) -> TurnObservation:
    """run 完了結果から次ターン解決用の plain 観測を抽出する（副作用なし・決定的）。

    `last_agent`（最終回答者の名前）と `new_items` 由来のハンドオフ遷移列を 1 パスで読む。
    入力オブジェクトは一切変更せず、同一入力に対して常に同一の観測を返す。読み取り先ごとに
    独立して防御するため、`last_agent` が読めなくてもハンドオフ観測は継続し、特定アイテムの
    読み取り失敗は当該アイテムのスキップに留まる。

    Args:
        result: run 完了結果（`last_agent` / `new_items` を持つ前提・ダックタイピングで読む）。

    Returns:
        plain `TurnObservation`（SDK 型を含まない）。判定材料が一切読めない場合は
        `last_agent=None` / `handoffs=()`。
    """
    raw_last_agent = _read_attr(result, "last_agent")
    last_agent = None if raw_last_agent is _MISSING else _agent_name(raw_last_agent)

    handoffs: list[tuple[str, str]] = []
    for item in _read_items(result):
        edge = _read_handoff(item)
        if edge is not None:
            handoffs.append(edge)

    return TurnObservation(last_agent=last_agent, handoffs=tuple(handoffs))


class ArrivalStore:
    """run 単位のハンドオフ到達記録（`RunContextWrapper` をキーとする弱参照マップ）。

    SDK は run 開始時に context wrapper を 1 回だけ生成し、handoff 実行（`on_handoff`）と
    handoff 有効性評価（`is_enabled`）の双方へ同一インスタンスを渡す。そのインスタンスを
    キーにすることで、記録と参照がキーで一致し、並行 run は構造的に分離される。run 終了後は
    弱参照により記録も解放されるため、記録は **run 内の一時状態**でありターン間・run 間の
    継続状態にはならない。

    ストアはインスタンスごとに独立でモジュールグローバル状態を持たない
    （`apply_next_turn_policy` の呼び出しごとに 1 つ生成される）。記録・参照はいずれも
    同期処理で `await` を挟まないため、記録と参照の間に割り込みは入らない。
    """

    def __init__(self) -> None:
        """空の到達記録を持つストアを生成する。"""
        self._arrivals: WeakKeyDictionary[Any, set[str]] = WeakKeyDictionary()

    def record(self, ctx: Any, agent_name: str) -> None:
        """当該 run でエージェント X へ到達したことを記録する（二重記録は冪等）。

        Args:
            ctx: run の context wrapper（記録キー）。
            agent_name: 到達したエージェント名 X。
        """
        arrived = self._arrivals.get(ctx)
        if arrived is None:
            arrived = set()
            self._arrivals[ctx] = arrived
        arrived.add(agent_name)

    def has_arrived(self, ctx: Any, agent_name: str) -> bool:
        """当該 run でエージェント X へ到達済みかを返す。

        Args:
            ctx: run の context wrapper（記録キー）。
            agent_name: 判定するエージェント名 X。

        Returns:
            到達済みなら True。未到達（当該 run の記録が無い場合を含む）なら False。
        """
        arrived = self._arrivals.get(ctx)
        return arrived is not None and agent_name in arrived


@dataclass(frozen=True)
class NextTurnWiring:
    """到達時ハンドオフ禁止の結線一式（判定表 + 到達記録ストア）。

    3 要素は常に揃って設置・継承・参照される単位であり、部分的な設置はない。

    Attributes:
        arrivals: 記録を前置する流入エッジ `(遷移元, X)` の集合（build 時に静的展開）。
        gated: ゲートを AND 合成する出辺の所有側エージェント名 X の集合。
        store: 到達記録ストア（記録側とゲート側で共有する）。
    """

    arrivals: frozenset[tuple[str, str]]
    gated: frozenset[str]
    store: ArrivalStore


def validate_recorded_edge_declaration(src: str, dst: str, config: HandoffConfig) -> None:
    """記録を前置するエッジの利用者宣言を SDK の `handoff()` と同じ規則で検証する。

    記録を前置するエッジでは合成が常に `on_handoff` を埋めるため、SDK の
    「`input_type` あり -> `on_handoff` 必須」チェックが発火しなくなる。同じ規則をここで
    利用者宣言に対して適用し、禁止を宣言しないエッジ（従来どおり SDK が送出する）と同じ
    build 時点・同じ例外型（`agents.exceptions.UserError`）で失敗させる。

    SDK 固有の規則と例外型の知識を `_adapters` に閉じるため、判定自体を本関数へ集約する
    （コア層は `agents` を直接参照しない・NFR-1）。

    Args:
        src: エッジの所有側（遷移元）エージェント名。
        dst: エッジの遷移先エージェント名。
        config: エッジの実効 `HandoffConfig`（利用者宣言が無い場合は既定値のインスタンス）。

    Raises:
        UserError: `input_type` を宣言しているのに `on_handoff` が無い場合
            （SDK の `handoff()` と同じ例外型）。
    """
    if config.input_type is not None and config.on_handoff is None:
        raise UserError(
            f"agent {src!r} から {dst!r} への handoff は input_type を宣言している"
            "ため on_handoff が必要です（SDK の handoff() と同じ要件: "
            "You must provide on_handoff when input_type is provided）"
        )


def _validate_user_on_handoff(
    user_on_handoff: Any, agent_name: str, *, has_input_type: bool
) -> None:
    """利用者宣言の `on_handoff` を SDK の `handoff()` と同じ規則で build 時に検証する。

    SDK の `handoff()` は合成後の callable しか見ないため、合成すると利用者宣言の誤りが
    build 時をすり抜けて run 中の失敗（`TypeError` 等）になる。SDK と同じ規則・同じ順序で
    ここでも検証し、禁止を宣言しないエッジと同じ build 時点で失敗させる。

    検証は SDK の分岐構造をそのまま写す。`input_type` あり経路だけが `callable()` を
    先行チェックして `UserError`（SDK の `on_handoff must be callable` 相当）とし、なし経路は
    `callable()` を見ずに `inspect.signature` へ渡す（非 callable は SDK と同じく
    `inspect.signature` 由来の `TypeError` になる）。いずれの経路でも、引数個数が期待値
    （`input_type` なし = 1 / あり = 2）と違えば `UserError`。

    例外型は SDK の `handoff()` と同じ `agents.exceptions.UserError` に揃える。禁止を宣言
    しないエッジでは同じ誤宣言に対して SDK が `UserError` を送出するため、型を変えると
    「禁止を宣言したエッジだけ例外型が変わる」非対称が生じ、利用者の `except UserError` を
    すり抜けてしまう。`callable()` チェックを SDK と同じ経路に限るのも同じ理由である。

    署名を取得できない callable（C 実装のビルトイン等）は検証をスキップせず、
    `inspect.signature` の例外をそのまま伝播させる（SDK の `handoff()` も `inspect.signature`
    を無防備に呼ぶため、スキップすると「SDK なら build 時に落ちる宣言が禁止対象エッジでだけ
    通る」非対称になる）。

    利用者宣言が無い（None）場合は記録のみを合成するため検証対象外。SDK の
    「`input_type` あり -> `on_handoff` 必須」の規則は**利用者宣言に対する**要件であり、
    合成の差し込み口を持つ registry 側（`_next_turn_config`）で判定する。

    Args:
        user_on_handoff: 利用者宣言の `on_handoff`。None なら検証しない。
        agent_name: 到達先エージェント名 X（エラーメッセージ用）。
        has_input_type: エッジが `input_type` を持つか。

    Raises:
        UserError: `input_type` ありのエッジで `on_handoff` が callable でない場合 /
            引数個数が期待値と一致しない場合。
        Exception: `inspect.signature` が送出した例外（SDK と同じくそのまま伝播する。
            `input_type` なしのエッジの非 callable は `TypeError` になる）。
    """
    if user_on_handoff is None:
        return

    if has_input_type and not callable(user_on_handoff):
        raise UserError(
            f"agent {agent_name!r} への handoff に宣言された on_handoff が callable では"
            f"ありません（{type(user_on_handoff).__name__}。SDK の handoff() と同じ要件: "
            "on_handoff must be callable）"
        )

    parameters = len(inspect.signature(user_on_handoff).parameters)

    expected = 2 if has_input_type else 1
    if parameters != expected:
        detail = "two arguments: context and input" if has_input_type else "one argument: context"
        edge = "input_type あり" if has_input_type else "input_type なし"
        raise UserError(
            f"agent {agent_name!r} への handoff に宣言された on_handoff の引数個数が "
            f"{parameters} 件です（{edge} のエッジでは on_handoff must take {detail}。"
            "SDK の handoff() と同じ要件です）"
        )


def make_arrival_recorder(
    store: ArrivalStore,
    agent_name: str,
    user_on_handoff: Any,
    has_input_type: bool,
) -> Callable[..., Any]:
    """到達記録を前置合成した `on_handoff` を組む（利用者宣言があれば chain する）。

    合成 callable の引数個数はエッジの `input_type` 有無へ合わせる。SDK の `handoff()` は
    `len(inspect.signature(on_handoff).parameters)` を厳密に検証し（`input_type` なし = 1 /
    あり = 2）、外れると `UserError` を送出するため、可変長引数や既定値で兼用せず
    **arity ごとに別の関数を返す**。

    呼び出し順は「記録 -> 利用者 `on_handoff`」で、利用者側からは自分の到達が記録済みに
    見える。利用者 `on_handoff` は同期・async のいずれでもよく、戻り値が awaitable なら
    await する（SDK の `_invoke_handoff` と同じ扱い）。引数は同一オブジェクトのまま透過する。

    合成すると SDK の検証は合成 callable だけを見るため、利用者宣言の誤り（非 callable /
    引数個数違い）は build 時に落ちず run 中の失敗に後ろ倒しになる。これを避けるため
    `_validate_user_on_handoff` で SDK と同じ規則・同じ分岐構造を build 時に適用する
    （禁止を宣言しないエッジで SDK が落とすのと同じタイミング・同じ例外型で失敗させる）。

    Args:
        store: 到達記録ストア（ゲート側と共有する）。
        agent_name: 到達先エージェント名 X（記録に使う名前）。
        user_on_handoff: 利用者宣言の `on_handoff`。None なら記録のみを行う。
        has_input_type: エッジが `input_type` を持つか（True で 2 引数形を返す）。

    Returns:
        SDK の `on_handoff` として渡せる合成 callable（`input_type` 有無に応じた arity）。

    Raises:
        UserError: `input_type` ありのエッジで利用者 `on_handoff` が callable でない場合 /
            引数個数がエッジの `input_type` 有無と合わない場合（SDK と同じ例外型）。
        Exception: `inspect.signature` が送出した例外（SDK と同じくそのまま伝播する。
            `input_type` なしのエッジの非 callable は `TypeError` になる）。
    """
    _validate_user_on_handoff(user_on_handoff, agent_name, has_input_type=has_input_type)

    if has_input_type:

        async def _record_with_input(ctx: Any, handoff_input: Any) -> None:
            store.record(ctx, agent_name)
            if user_on_handoff is not None:
                outcome = user_on_handoff(ctx, handoff_input)
                if inspect.isawaitable(outcome):
                    await outcome

        return _record_with_input

    async def _record(ctx: Any) -> None:
        store.record(ctx, agent_name)
        if user_on_handoff is not None:
            outcome = user_on_handoff(ctx)
            if inspect.isawaitable(outcome):
                await outcome

    return _record


def make_arrival_gate(
    store: ArrivalStore,
    agent_name: str,
    existing_is_enabled: Any,
) -> Callable[..., Any]:
    """到達記録を参照するゲートを `is_enabled` へ AND 合成する。

    合成 callable は SDK が bool 以外の `is_enabled` を呼ぶ形に合わせて **ちょうど 2 引数
    `(ctx, agent)`** を取る。判定は closure が捕捉した X の名前と当該 run の記録だけで行い、
    第 2 引数の内容には依存しない（所有側 Agent が渡る実装に依存しないため）。

    当該 run で X へ到達済みなら既存 `is_enabled` を**評価せずに** False を返す（短絡）。
    未到達なら既存へ委譲し、既存が bool ならその値、callable なら `(ctx, agent)` を同一
    オブジェクトのまま渡した評価結果（awaitable なら await した結果）を返す。

    Args:
        store: 到達記録ストア（記録側と共有する）。
        agent_name: 出辺の所有側エージェント名 X（禁止の判定に使う名前）。
        existing_is_enabled: 既存の `is_enabled` 宣言（bool または `(ctx, agent)` callable）。

    Returns:
        SDK の `is_enabled` として渡せる合成 callable（2 引数）。
    """

    async def _gate(ctx: Any, agent: Any) -> bool:
        if store.has_arrived(ctx, agent_name):
            return False
        if not callable(existing_is_enabled):
            return bool(existing_is_enabled)
        outcome = existing_is_enabled(ctx, agent)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        return bool(outcome)

    return _gate
