"""Failsafe（宣言的な例外着地）の宣言型（agents 非依存）。

Runner 外へ伝播した例外を、呼び出し側に分散した try/except でなく
`FailsafePolicy`（例外型 -> 着地値 / fallback のマッピング）として宣言する。
`__post_init__` で handlers のキーと `on_apply` を build-time 検証し、捕捉範囲が
広すぎる宣言やプロセス制御例外の握り潰しを fail-fast したうえで、handlers を
不変化して構築後の書き換えを防ぐ。適用結果は `FailsafeResult` として
返り、SDK `RunResult` とは共通基底を持たず `.final_output` のみ structural に一致する。

着地時には「もともと実行中だったエージェント」を `FailsafeResult.last_agent` として
参照できる。決定は 2 段で、(1) 例外ごとの指定（`FailsafeHandler.last_agent`）、
(2) 全体規定（`FailsafePolicy.fallback_last_agent`）の順に見る。どちらの段にも
具体の agent か `RUNNING_AGENT` を置け、どちらも無指定なら None になる。
`RUNNING_AGENT` は「実際に動いていた Agent を使う」ことを表す公開 sentinel で、
これを置いた段でのみ例外からの解決（`exc.run_data.last_agent` -> `exc.last_agent`）を
試みる opt-in の合図であり、解決できなければ次の段へ落ちる。解決は `getattr` の
duck typing のみで行うため、外部依存（agents / openai）を持たない。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..._validation import validate_bool
from ...constants import RESILIENCE_LOGGER_NAME

logger = logging.getLogger(RESILIENCE_LOGGER_NAME)

_FORBIDDEN_HANDLER_TYPES: frozenset[type[BaseException]] = frozenset(
    {
        Exception,
        BaseException,
        ExceptionGroup,
        KeyboardInterrupt,
        SystemExit,
        asyncio.CancelledError,
        GeneratorExit,
    }
)
"""handlers のキーとして禁止する例外型の列挙。

`Exception` / `BaseException` は「あらゆる失敗を着地させる」塗りつぶしとなり、
バグ由来の例外まで正常値へすり替えて障害を不可視化するため禁止する。
`ExceptionGroup` は「無関係な例外の束」を表す集約例外で、マッチが `isinstance` である以上
`TaskGroup` 等が束ねた中身によらず丸ごと着地させる広すぎる捕捉になるため同じ理由で禁止する
（`BaseExceptionGroup` は `Exception` 非派生として別の段で拒否される）。
`KeyboardInterrupt` / `SystemExit` / `asyncio.CancelledError` / `GeneratorExit` は
プロセス制御・協調キャンセルのための例外であり、握り潰すと停止要求やキャンセルが
効かなくなるため禁止する。

禁止は列挙メンバーそのものに限る。利用者定義のサブクラス（例: `ExceptionGroup` を継承した
独自の集約例外）は捕捉範囲が限定されるため、従来どおりキーとして許容する。
"""


class _RunningAgentSentinel:
    """`RUNNING_AGENT` の型（単一インスタンスのみを持つ内部 sentinel 型）。

    利用者が直接構築することは想定せず、モジュール変数 `RUNNING_AGENT` を
    そのまま指定値として使う。判定は `is RUNNING_AGENT` の同一性で行う。
    `__new__` が唯一のインスタンスを返すため、直接構築した `_RunningAgentSentinel()` も
    `RUNNING_AGENT` と同一オブジェクトになる。別インスタンスを作れてしまうと
    `is RUNNING_AGENT` を前提とした誤宣言のガード（`FailsafeHandler.fallback` と
    handlers 値位置の拒否）を迂回でき、指定用 sentinel がそのまま `final_output` に
    載るため、単一性はガードの前提となる不変条件である。
    `__reduce__` により複製（`copy` / `deepcopy`）や pickle 往復でも同一
    インスタンスのまま復元される。宣言（`FailsafeHandler` や handlers dict）ごと
    複製されても `_resolve_last_agent` の `is RUNNING_AGENT` 判定が外れない、という
    不変条件を保つためで、これが崩れると複製後の sentinel が「具体の agent」と
    誤認され、そのまま `FailsafeResult.last_agent` に載ってしまう。
    """

    __slots__ = ()

    _instance: ClassVar[_RunningAgentSentinel | None] = None
    """唯一のインスタンス（`__new__` が生成して保持する）。"""

    def __new__(cls) -> _RunningAgentSentinel:
        """唯一のインスタンスを返す（未生成なら生成して保持する）。

        生成済み判定はモジュール変数 `RUNNING_AGENT` ではなくクラス属性で行う
        （`RUNNING_AGENT` 自身の初期化もこの経路を通るため、モジュール変数を参照すると
        初期化が循環する）。

        Returns:
            `RUNNING_AGENT` と同一の `_RunningAgentSentinel` インスタンス。
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        """デバッグ時に素性が読めるよう、sentinel 名そのものを返す。

        Returns:
            文字列 `"RUNNING_AGENT"`。
        """
        return "RUNNING_AGENT"

    def __reduce__(self) -> str:
        """複製・pickle 往復でも同一インスタンスを保つ（`is` 判定の前提）。

        Returns:
            モジュール変数名 `"RUNNING_AGENT"`（`copy` / `deepcopy` はこの文字列
            返却により元オブジェクトをそのまま返し、pickle は同名の global 参照
            として復元する）。
        """
        return "RUNNING_AGENT"


RUNNING_AGENT: Final = _RunningAgentSentinel()
"""`last_agent` の指定値として「実際に動いていた Agent を使う」ことを表す sentinel。

`FailsafeHandler.last_agent`（例外ごとの指定）と `FailsafePolicy.fallback_last_agent`
（全体規定）のどちらにも置ける。この値を置いた段でのみ例外からの解決
（`exc.run_data.last_agent` -> `exc.last_agent`）を試みる opt-in の合図であり、
指定しなければ解決は一切走らない（`last_agent` は None のまま）。解決できなかった
場合は次の段へ落ちる（最後まで解決できなければ None）。`failsafe_call` /
`FailsafeResult.from_exception` が生成する結果には sentinel そのものが載ることはない
（`FailsafeResult` を直接構築する場合は `_resolve_last_agent` を経由しないため、
渡した値がそのまま載る）。
"""


def _type_name_of(value: Any) -> str:
    """検証エラーのメッセージに載せる型名を返す。

    クラスを渡すべき位置に「クラス」と「インスタンス等の値」のどちらが来ても、原因が
    特定できる名前を出すためのヘルパ。

    Args:
        value: 型名を求める対象。

    Returns:
        `value` がクラスならその `__name__`、それ以外なら `type(value).__name__`。
    """
    return value.__name__ if isinstance(value, type) else type(value).__name__


def _read_run_data_last_agent(exc: Exception) -> Any:
    """1 つ目の読み取り先（`exc.run_data.last_agent`）を読む。

    属性を持たないオブジェクトに備えて `getattr` の防御読みとするが、属性が property で
    実装され読み出し自体が失敗する場合は例外がそのまま伝播する（段ごとの防御は
    呼び出し側の `_derive_last_agent` が担う）。

    Args:
        exc: 着地対象の例外インスタンス。

    Returns:
        読み取れた `last_agent`（不透明値）。`run_data` を持たない場合、または
        `run_data` が `last_agent` を持たない場合は None。
    """
    run_data = getattr(exc, "run_data", None)
    if run_data is None:
        return None
    return getattr(run_data, "last_agent", None)


def _read_last_agent_attr(exc: Exception) -> Any:
    """2 つ目の読み取り先（`exc.last_agent`）を読む。

    Args:
        exc: 着地対象の例外インスタンス。

    Returns:
        読み取れた `last_agent`（不透明値）。属性を持たない場合は None。
    """
    return getattr(exc, "last_agent", None)


def _derive_last_agent(exc: Exception) -> Any:
    """例外から `last_agent`（実行中だったエージェント）を解決する。

    読み取り先は `exc.run_data.last_agent` -> `exc.last_agent` の順。いずれも
    `getattr` の duck typing で読むため、SDK 例外・lib 独自例外・利用者定義例外の
    いずれも同じ規約（どちらかの属性を持てば解決に参加できる）で扱える。
    解決できない場合は「不明」を正直に表現して None を返す（例外にしない）。

    防御は読み取り先ごとに独立して行う（読み出し全体を 1 つの `try` で囲わない）。
    属性が property で実装され読み出し自体が失敗する例外であっても、着地
    （`FailsafeResult` の返却・warning・`on_apply`）を壊してはならず、かつ一方の
    読み取り先の失敗でもう一方を飛ばしてはならないためで、失敗した段は
    `logger.debug(exc_info=True)` に記録して次の段へ進む。すべての段が失敗
    （または None）なら解決不能として None を返す。

    各読み取り先の採否は `is None` / `is RUNNING_AGENT` の同一性のみで判定するため、
    空文字や 0 のような falsy な値も正当な `last_agent` として採用される。`RUNNING_AGENT`
    は「例外から解決せよ」という指定用の sentinel であって解決結果ではないため、例外側が
    それを運んでいた場合は解決不能として次の読み取り先へ進む（sentinel が着地結果の
    `last_agent` に載ることはない）。

    Args:
        exc: 着地対象の例外インスタンス。

    Returns:
        解決した `last_agent`（不透明値）。解決できない場合は None
        （`RUNNING_AGENT` そのものは決して返らない）。
    """
    for read in (_read_run_data_last_agent, _read_last_agent_attr):
        try:
            candidate = read(exc)
        except Exception:
            logger.debug(
                "failsafe last_agent resolution failed: exception_type=%s",
                type(exc).__name__,
                exc_info=True,
            )
            continue
        if candidate is not None and candidate is not RUNNING_AGENT:
            return candidate

    return None


def _resolve_last_agent(exc: Exception, per_exception: Any, policy_default: Any) -> Any:
    """2 段の指定から `last_agent` を確定する。

    段は「例外ごとの指定」->「全体規定」の順に見る。各段の値は具体の agent または
    `RUNNING_AGENT` で、None は「その段の指定なし」を意味する。`RUNNING_AGENT` を
    置いた段でのみ例外からの解決を試み、解決できなければ次の段へ落ちる。

    判定は `is None` / `is RUNNING_AGENT` の同一性のみで行う（`or` 連鎖や truthiness
    判定は空文字や 0 のような falsy な具体値を黙って次段へ飛ばすため使わない）。

    Args:
        exc: 着地対象の例外インスタンス。
        per_exception: 段 1（`FailsafeHandler.last_agent`）の指定。
        policy_default: 段 2（`FailsafePolicy.fallback_last_agent`）の指定。

    Returns:
        確定した `last_agent`（不透明値）。どの段でも決まらない場合は None
        （`RUNNING_AGENT` そのものは決して返らない）。
    """
    for spec in (per_exception, policy_default):
        if spec is None:
            continue
        if spec is RUNNING_AGENT:
            resolved = _derive_last_agent(exc)
            if resolved is not None:
                return resolved
            continue
        return spec

    return None


@dataclass(frozen=True)
class FailsafeHandler:
    """handlers の値位置に置ける opt-in の宣言（着地値 + 例外ごとの `last_agent` 指定）。

    `FailsafePolicy.handlers` の値は素の着地値・fallback callable のままでよく、
    「この例外型のときだけ `last_agent` を指定したい」場合にのみ本型で包む
    （既定形の宣言は無変更で従来どおり動く）。`fallback` の解決規則は包まない場合と
    完全に同一で、素の値ならそれ自体が着地値、callable なら捕捉例外を単一引数に
    呼んだ戻り値（sync / async 可）が着地値になる。

    Attributes:
        fallback: 着地値そのもの、または例外を受け取り着地値を返す callable。
        last_agent: 決定モデルの段 1（例外ごとの指定）。具体の agent（`AgentRegistry`
            から取得した `Agent` をそのまま渡せる）または `RUNNING_AGENT`。既定 None は
            「この段の指定なし」を意味し、段 2（`FailsafePolicy.fallback_last_agent`）へ
            落ちる。値は機微（システムプロンプト・資格情報）を含みうるため `repr` には
            出さない（属性としては従来どおり参照できる）。

    Raises:
        ValueError: `fallback` に `FailsafeHandler` を入れ子で渡した場合（内側の
            宣言がそのまま着地値になる silent な誤宣言を build-time で拒否する）、
            または `fallback` に `RUNNING_AGENT` を渡した場合（`last_agent` の指定用
            sentinel が着地値として `final_output` に載る silent な誤宣言を拒否する。
            `last_agent=RUNNING_AGENT` は正当な指定であり拒否しない）。
    """

    fallback: Any
    last_agent: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """`fallback` 位置の誤宣言（ネスト宣言・指定用 sentinel）を build-time で拒否する。

        Raises:
            ValueError: `fallback` が `FailsafeHandler` インスタンスの場合、または
                `fallback` が `RUNNING_AGENT` の場合。
        """
        if isinstance(self.fallback, FailsafeHandler):
            raise ValueError(
                "fallback must not be a FailsafeHandler instance "
                "(nested declaration would land the inner FailsafeHandler as final_output; "
                "pass the landing value or a callable instead)"
            )

        if self.fallback is RUNNING_AGENT:
            raise ValueError(
                "fallback must not be RUNNING_AGENT "
                "(the sentinel specifies last_agent and would land as final_output; "
                "pass the landing value or a callable instead, "
                "and use last_agent=RUNNING_AGENT to resolve the running agent)"
            )


@dataclass(frozen=True)
class FailsafePolicy:
    """例外型ごとの着地（failsafe）の宣言。

    `handlers` のキーは `Exception` のサブクラス（`Exception` 自身は不可）に限定する。
    値は着地値そのもの、例外を受け取り着地値を返す callable（sync / async 可）、または
    `FailsafeHandler`。空 `handlers` は「一切捕捉しない」no-op として許容し、
    矛盾ではないため `ValueError` にしない。

    Attributes:
        handlers: 例外型から着地値 / fallback へのマッピング。宣言順に first-match する。
            値は着地値そのもの、callable、または `FailsafeHandler`（例外ごとの
            `last_agent` 指定を伴う opt-in 宣言）。値位置に直接 `RUNNING_AGENT` を
            置いた宣言（`handlers={E: RUNNING_AGENT}`）は build-time `ValueError` で
            拒否する（`FailsafeHandler(fallback=RUNNING_AGENT)` の拒否と対称。sentinel は
            `last_agent` の指定値であり、着地値として利用者へ返るのは誤宣言のため）。
            `__post_init__` の検証後に `MappingProxyType` へ差し替えられ、構築後の
            書き換え（検証を迂回した禁止キーの事後注入）はできない。この差し替えにより
            policy 自体の `copy.deepcopy` / `dataclasses.asdict` は `TypeError` になる
            （`mappingproxy` は pickle 不可）。複製が必要な場合は
            `FailsafePolicy(dict(policy.handlers), ...)` で再構築する。
        log_on_apply: failsafe 適用時に warning ログを出すか（既定 True）。出力には
            例外メッセージとトレースバックがそのまま含まれるため、プロンプト・
            ユーザー入力・資格情報を含みうる例外を扱う場合は `False` とし、
            `on_apply` でマスキングしたうえで記録する。
        on_apply: 適用時に呼ばれるコールバック（`FailsafeResult` を受け取る・
            sync / async 可・戻り値は無視）。None は未指定。コールバックが送出した
            例外は `logger.error(exc_info=True)` で記録して握り潰す（監査経路の失敗は
            着地結果の返却を妨げない）。
        fallback_last_agent: 決定モデルの段 2（全体規定）。例外ごとの指定
            （`FailsafeHandler.last_agent`）が無い / 解決できなかったときに使う
            `last_agent` で、具体の agent または `RUNNING_AGENT` を置ける。
            既定 None は「全体規定なし」を意味し、段 1 でも決まらなければ
            `FailsafeResult.last_agent` は None になる。値は機微（システム
            プロンプト・資格情報）を含みうるため `repr` には出さない（属性としては
            従来どおり参照でき、`==` にも従来どおり含まれる）。
    """

    handlers: Mapping[type[Exception], Any] = field(default_factory=dict)
    log_on_apply: bool = True
    on_apply: Any = None
    fallback_last_agent: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """build-time 検証を行い、不正な宣言を `ValueError` で fail-fast する。

        `handlers` は先に `dict` へ正規化してから検証し、**検証したその dict を**
        `MappingProxyType` で包んで格納する。反復（`__iter__`）と添字アクセス
        （`keys()` / `__getitem__`）で別の内容を返す独自 `Mapping` を渡されても、
        検証対象と格納対象が食い違わない（検証の迂回を構造的に防ぐ）。

        Raises:
            ValueError: `log_on_apply` が bool でない、`handlers` のキーが例外クラスでない /
                禁止列挙のいずれかそのもの / `Exception` を継承しない `BaseException` 系、
                `handlers` の値が `RUNNING_AGENT` そのもの、または `on_apply` が None でも
                callable でもない場合。
        """
        validate_bool(self.log_on_apply, "log_on_apply")

        normalized = dict(self.handlers)
        for key, value in normalized.items():
            self._validate_handler_key(key)
            if value is RUNNING_AGENT:
                raise ValueError(
                    f"handlers value for {key.__name__!r} must not be RUNNING_AGENT "
                    "(the sentinel specifies last_agent and would land as final_output; "
                    "use FailsafeHandler(fallback=..., last_agent=RUNNING_AGENT) instead)"
                )

        if self.on_apply is not None and not callable(self.on_apply):
            raise ValueError(
                f"on_apply must be callable or None, got {type(self.on_apply).__name__!r}"
            )

        object.__setattr__(self, "handlers", MappingProxyType(normalized))

    @staticmethod
    def _validate_handler_key(key: Any) -> None:
        """handlers のキー 1 件を 3 段で検証する。

        Args:
            key: 検証対象の handlers キー。

        Raises:
            ValueError: 例外クラスでない / 禁止列挙そのもの / `Exception` 非派生の
                `BaseException` 系のいずれかに該当する場合（メッセージに違反キー名を含む）。
        """
        if not (isinstance(key, type) and issubclass(key, BaseException)):
            name = key.__name__ if isinstance(key, type) else type(key).__name__
            raise ValueError(
                f"handlers key must be an exception class, got {name!r} "
                "(pass the exception type itself, not an instance or other value)"
            )

        if key in _FORBIDDEN_HANDLER_TYPES:
            raise ValueError(
                f"handlers key {key.__name__!r} is forbidden "
                "(too broad, or a process-control exception that must not be swallowed)"
            )

        if not issubclass(key, Exception):
            raise ValueError(
                f"handlers key {key.__name__!r} must be a subclass of Exception "
                "(BaseException-derived types outside Exception are not caught)"
            )


@dataclass(frozen=True)
class FailsafeResult:
    """failsafe が適用されたことを表す結果。

    SDK `RunResult` と共通基底は持たず、`.final_output` のみ structural に一致させる。
    呼び出し側は `isinstance(x, FailsafeResult)` で failsafe 適用を判別できる。

    Attributes:
        final_output: 着地値（handlers の値、または fallback callable の戻り値）。
        exception: 捕捉した例外インスタンス。
        matched_type: 結果に載せる例外型。`failsafe_call` の着地では first-match した
            handlers のキー、`from_exception` では明示指定した値（未指定なら送出型
            `type(exception)`）。
        last_agent: 決定モデル（例外ごとの指定 -> 全体規定）で確定した、実行中だった
            エージェント（不透明値）。どの段でも決まらない場合は None。`failsafe_call` /
            `from_exception` が生成する結果には指定用の sentinel `RUNNING_AGENT` が
            そのまま載ることはない（本型を直接構築する場合は `_resolve_last_agent` を
            経由しないため、
            渡した値がそのまま載る）。値は機微（システム
            プロンプト・資格情報）を含みうるため `repr` には出さない（属性としては
            従来どおり参照でき、`==` にも従来どおり含まれる）。

    Raises:
        ValueError: `exception` が `Exception` インスタンスでない場合、または
            `matched_type` が `Exception` サブクラスでない場合（`__post_init__` の
            build-time 検証。直接構築と `from_exception` で同一型に 2 つの契約を
            作らないため、両経路が同じ検証を通る）。
    """

    final_output: Any
    exception: Exception
    matched_type: type[Exception]
    last_agent: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """入力（例外インスタンス / 例外クラス）を build-time 検証する。

        直接構築と `from_exception` の双方がこの検証を通るため、同一型で契約が分岐しない
        （`from_exception` は `matched_type=None`（送出型を採る）の許容だけを固有に持つ）。
        `last_agent` は任意の不透明値（Agent・その代役・falsy 値）を受け取る契約のため
        検証しない。

        Raises:
            ValueError: `exception` が `Exception` インスタンスでない場合、または
                `matched_type` が `Exception` サブクラスでない場合（いずれもメッセージに
                受け取った値の型名を含む）。
        """
        if not isinstance(self.exception, Exception):
            raise ValueError(
                f"exception must be an Exception instance, "
                f"got {type(self.exception).__name__!r} "
                "(pass the caught exception instance, not a class or other value)"
            )

        if not (isinstance(self.matched_type, type) and issubclass(self.matched_type, Exception)):
            raise ValueError(
                f"matched_type must be a subclass of Exception, "
                f"got {_type_name_of(self.matched_type)!r} "
                "(pass the exception type itself, not an instance or other value)"
            )

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        *,
        final_output: Any,
        matched_type: type[Exception] | None = None,
        last_agent: Any = None,
    ) -> FailsafeResult:
        """例外から手動で `FailsafeResult` を構築する（`failsafe_call` の外側用）。

        `failsafe_call` に宣言していない例外を利用者が自前の except で捕捉する場合でも、
        同じ結果型・同じ `last_agent` の意味へ着地させるためのファクトリ。解決ロジックを
        利用者側へ複製させないことが目的で、`failsafe_call` 内外で `last_agent` の
        意味が揃う。policy を受け取らないため決定モデルの段 2（全体規定）は持たず、
        本メソッドの `last_agent` 引数（段 1 相当）だけで決まる。

        監査は発火しない（warning ログも `policy.on_apply` も呼ばれない）。`failsafe_call`
        の着地とは異なり、記録するかどうかは呼び出し側の except 節の責務になる。

        Example:
            ```python
            try:
                return await failsafe_call(policy, lambda: Runner.run(agent, msg))
            except TimeoutError as exc:
                return FailsafeResult.from_exception(
                    exc, final_output="時間内に応答できませんでした。"
                )
            ```

        Args:
            exception: 捕捉した例外インスタンス。
            final_output: 着地値。
            matched_type: 結果に載せる例外型。None（既定）なら `type(exception)` を採る
                （手動着地では宣言キーが存在しないため送出型が自然な既定）。
            last_agent: `last_agent` の指定。具体の agent ならその値がそのまま載り、
                `RUNNING_AGENT` なら `failsafe_call` の着地時と同一の解決
                （`run_data.last_agent` -> `last_agent`）を試みる。None（既定）は
                「指定なし」で、解決は走らず結果も None になる。

        Returns:
            構築した `FailsafeResult`。

        Raises:
            ValueError: `matched_type` が None でも `Exception` サブクラスでもない場合
                （None 許容は本メソッド固有のため、ここで固有のメッセージを出す）、または
                `exception` が `Exception` インスタンスでない場合（`__post_init__` の
                共通検証が直接構築と同一のメッセージで送出する）。いずれもメッセージに
                受け取った値の型名を含む。
        """
        if matched_type is not None and not (
            isinstance(matched_type, type) and issubclass(matched_type, Exception)
        ):
            raise ValueError(
                f"matched_type must be a subclass of Exception or None, "
                f"got {_type_name_of(matched_type)!r} "
                "(pass the exception type itself, not an instance or other value)"
            )

        return cls(
            final_output=final_output,
            exception=exception,
            matched_type=matched_type if matched_type is not None else type(exception),
            last_agent=_resolve_last_agent(exception, last_agent, None),
        )


async def failsafe_call[T](
    policy: FailsafePolicy, thunk: Callable[[], Awaitable[T]]
) -> T | FailsafeResult:
    """`thunk` を 1 回 await し、`policy` の宣言に従って例外を着地させる。

    `policy.handlers` が空なら（`thunk` の受理契約検査は行ったうえで）一切捕捉せず
    完全透過する。例外が送出された場合は
    handlers を宣言順（挿入順）に走査し、最初に `isinstance` マッチしたキーの値を
    着地値として `FailsafeResult` を返す（first-match。より specific な型を先に
    宣言する責務は利用者側にある）。どのキーにもマッチしない例外は握り潰さず
    そのまま再送出する（例外チェーンも付けない）。

    handlers の値が callable なら fallback とみなして捕捉例外を単一引数に呼ぶ
    （sync / async 可）。callable そのものを着地値にしたい場合は
    `lambda exc: my_func` のようにラップして返す。値が `FailsafeHandler` の場合は
    `fallback` を取り出して同じ規則で解決し、`last_agent` を例外ごとの指定として扱う。

    着地結果の `last_agent` は 2 段で確定する。段 1 は例外ごとの指定
    （`FailsafeHandler.last_agent`）、段 2 は全体規定（`policy.fallback_last_agent`）で、
    どちらにも具体の agent か `RUNNING_AGENT` を置ける。`RUNNING_AGENT` を置いた段では
    例外からの解決（`exc.run_data.last_agent` -> `exc.last_agent`）を試み、解決できなければ
    次の段へ落ちる。どちらの段も無指定なら解決は走らず None になる（各段の採否は
    `is None` / `is RUNNING_AGENT` の同一性判定のため、空文字や 0 のような falsy な
    具体値も正当な指定として採用される）。確定値は `FailsafeResult.last_agent` に載り、
    `on_apply` へ渡る結果からも参照できる。

    `thunk` の受理契約は `Callable[[], Awaitable[T]]` のみで、coroutine オブジェクトの
    直渡し（呼び出せない）や、呼び出せても awaitable 以外を返す callable は
    `TypeError` で fail-fast する。この検査は handlers の宣言有無より前・try の外で
    1 度だけ行う（`thunk` の呼び出しも経路によらず 1 回）。handlers が空でも同じ
    メッセージで fail-fast し、handlers に `TypeError` を宣言していても受理契約違反は
    着地対象にならない。また `thunk` 本体が同期的に送出する例外も着地対象外で、
    await 中に発生した例外のみが着地対象となる。

    `policy.on_apply` が送出した例外は `logger.error(exc_info=True)` で記録して
    握り潰し、`FailsafeResult` の返却は継続する（監査経路の失敗は呼び出し側へ
    伝播しない）。

    streaming（`run_streamed`）・sync（`run_sync`）専用のヘルパーは提供しない
    （sync 文脈は `asyncio.run(failsafe_call(...))` で代替する）。Realtime は非対応。

    Args:
        policy: 着地の宣言（handlers / log_on_apply / on_apply / fallback_last_agent）。
        thunk: 引数なしで awaitable を返す callable（例: `lambda: Runner.run(...)`）。

    Returns:
        正常完了時は `thunk` の戻り値そのもの（ラップしない）。着地時は
        `FailsafeResult`。

    Raises:
        TypeError: `thunk` が受理契約を満たさない場合（coroutine の直渡し・
            awaitable 以外を返す callable）。
        Exception: handlers のどのキーにもマッチしない例外、および fallback callable
            自身が送出した例外（いずれも素通しする）。
    """
    awaitable = thunk()
    if not inspect.isawaitable(awaitable):
        raise TypeError(f"thunk must return an awaitable, got {type(awaitable).__name__!r}")

    if not policy.handlers:
        return await awaitable

    try:
        result = await awaitable
    except Exception as exc:
        for key in policy.handlers:
            if isinstance(exc, key):
                break
        else:
            raise

        declared = policy.handlers[key]
        # `FailsafeHandler` 判定を callable 判定より先に行う（実装契約）。dataclass は
        # callable() が False のため現状は順序に依存しないが、契約として固定する。
        if isinstance(declared, FailsafeHandler):
            fallback = declared.fallback
            per_exception_last_agent = declared.last_agent
        else:
            fallback = declared
            per_exception_last_agent = None

        if callable(fallback):
            value = fallback(exc)
            if inspect.isawaitable(value):
                value = await value
        else:
            value = fallback

        last_agent = _resolve_last_agent(exc, per_exception_last_agent, policy.fallback_last_agent)

        failsafe_result = FailsafeResult(
            final_output=value, exception=exc, matched_type=key, last_agent=last_agent
        )
        if policy.log_on_apply:
            logger.warning(
                "failsafe applied: matched_type=%s exception_type=%s exception=%s",
                key.__name__,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
        if policy.on_apply is not None:
            try:
                outcome = policy.on_apply(failsafe_result)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:
                logger.error(
                    "failsafe on_apply callback failed: matched_type=%s",
                    key.__name__,
                    exc_info=True,
                )
        return failsafe_result
    else:
        return result
