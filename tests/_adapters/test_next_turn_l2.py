"""L2: `_adapters.next_turn.extract_turn_observation`（run 完了結果からの観測抽出）の pin。

FR-2 / NFR-5 の受け入れ基準のうち、SDK 結合点の防御的読み取りに関わる部分を決定表として
pin する。判定材料は run 完了結果の `last_agent`（最終回答者）と `new_items`（handoff
アイテム）で、抽出結果は plain frozen dataclass（`last_agent: str | None` /
`handoffs: tuple[tuple[str, str], ...]`）である。

- アイテム判定は `source_agent` / `target_agent` 属性の**有無**で行い、`type` 文字列の
  リテラルに依存しない（`type` が別物の handoff アイテムも拾う / 属性を持たないアイテムは
  `type` が handoff 風でも無視する）。`observe_run_result`（`_adapters/routing.py`）と同型。
- `last_agent` はエージェントの `.name` を読む。
- 複数のハンドオフ到達は観測順（`(遷移元, 遷移先)` の順序列）で保たれる。
- 防御性: `last_agent` / `new_items` の属性欠落に加え、**属性アクセス自体が例外を送出する**
  場合（SDK の `last_agent` は release 後アクセスで例外を送出しうる）も例外を送出せず
  安全側（`last_agent=None` / `handoffs=()`）へ倒す。個々のアイテムの `source_agent` /
  `target_agent` アクセスが例外を送出した場合は、そのアイテムだけをスキップして他は拾う。
- 純粋性: 入力の結果オブジェクトを変更せず、同一入力に対して常に同一の結果を返す。

あわせて FR-3（到達時ハンドオフ禁止）の実現形である到達記録ストアと 2 つの合成
（記録 `on_handoff` / `is_enabled` ゲート）の契約を pin する:

- 到達記録ストア: run（`RunContextWrapper` インスタンス）単位で記録が分離され、記録は
  エージェント名で識別される。内部は `WeakKeyDictionary` で、wrapper への参照が消えれば
  エントリも解放される（run 内一時状態でありターン間に持ち越さない）。ストアは
  `apply_next_turn_policy` 呼び出しごとに独立生成されるためインスタンス間でも独立する。
- 記録 `on_handoff` 合成: SDK の署名検証（`input_type` なし = 1 引数 / あり = 2 引数）を
  通過する arity で生成され、呼ぶと到達が記録される。利用者宣言の `on_handoff` がある
  場合は「記録 -> 利用者 `on_handoff`」の順で、同じ引数を透過して chain する（sync /
  async の双方）。
- `is_enabled` ゲート合成: SDK が渡す 2 引数 `(ctx, agent)` を受け、当該 run で X へ到達
  済みなら `False`、未到達なら既存 `is_enabled`（bool / callable）の評価へ委譲する。
  判定は closure に閉じた X 名で行い、第 2 引数には依存しない。

SDK 型に依存しないフェイク（`name` 属性のみを持つエージェント代役・`source_agent` /
`target_agent` を持つアイテム代役・identity hash を持つ wrapper 代役）で検証する。実 SDK
`handoff()` の署名検証を通過することの pin は `test_next_turn_handoff_signature_l2.py`
（integration）が担う。
"""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import inspect
import time
import weakref
from typing import Any

import pytest
from agents.exceptions import UserError

from oai_agentspec._adapters.next_turn import (
    ArrivalStore,
    extract_turn_observation,
    make_arrival_gate,
    make_arrival_recorder,
)

pytestmark = pytest.mark.unit


class _FakeAgent:
    """SDK Agent の代役（観測に必要な `name` のみを持つ最小形）。"""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeHandoffItem:
    """handoff アイテムの代役（`source_agent` / `target_agent` を持つ）。

    `type` は既定で SDK リテラルと異なる値にし、判定が `type` 文字列に依存していないことを
    示す（依存していればこのアイテムは取りこぼされる）。
    """

    def __init__(self, source: str, target: str, type_: str = "not-a-handoff-type") -> None:
        self.source_agent = _FakeAgent(source)
        self.target_agent = _FakeAgent(target)
        self.type = type_


class _FakeNonHandoffItem:
    """handoff でないアイテムの代役（属性を持たず `type` だけが handoff 風）。"""

    def __init__(self, type_: str = "handoff_output_item") -> None:
        self.type = type_


class _FakeMessageItem:
    """メッセージ相当のアイテム代役（handoff 判定に使う属性を一切持たない）。"""

    def __init__(self, text: str = "answer") -> None:
        self.text = text


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
        """アクセスのたびに例外を送出する（`getattr` の既定値では吸収できない）。

        Raises:
            RuntimeError: 常に送出する。
        """
        raise RuntimeError("last_agent is no longer available")


class _ResultWithoutNewItems:
    """`new_items` 属性を持たない結果（属性欠落の防御読みの検証用）。"""

    def __init__(self, last_agent: Any) -> None:
        self.last_agent = last_agent


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


class _RaisingSourceItem:
    """`source_agent` の読み出しが例外を送出する handoff 風アイテム。"""

    def __init__(self, target: str) -> None:
        self.target_agent = _FakeAgent(target)

    @property
    def source_agent(self) -> Any:
        """アクセスのたびに例外を送出する（`hasattr` でも吸収されない）。

        Raises:
            RuntimeError: 常に送出する。
        """
        raise RuntimeError("source_agent is broken")


class _RaisingTargetItem:
    """`target_agent` の読み出しが例外を送出する handoff 風アイテム。"""

    def __init__(self, source: str) -> None:
        self.source_agent = _FakeAgent(source)

    @property
    def target_agent(self) -> Any:
        """アクセスのたびに例外を送出する。

        Raises:
            RuntimeError: 常に送出する。
        """
        raise RuntimeError("target_agent is broken")


# ---------------------------------------------------------------------------
# 基本の抽出（last_agent / handoffs）
# ---------------------------------------------------------------------------


def test_extract_turn_observation_last_agentは名前として読まれる() -> None:
    """`result.last_agent` の `.name` が最終回答者名として観測される。"""
    result = _FakeResult(_FakeAgent("billing"), [_FakeHandoffItem("triage", "billing")])

    observation = extract_turn_observation(result)

    assert observation.last_agent == "billing"


def test_extract_turn_observation_handoffは遷移元と遷移先の組になる() -> None:
    """handoff アイテムは `(遷移元, 遷移先)` の組として観測される。"""
    result = _FakeResult(_FakeAgent("billing"), [_FakeHandoffItem("triage", "billing")])

    observation = extract_turn_observation(result)

    assert observation.handoffs == (("triage", "billing"),)


def test_extract_turn_observation_handoffsはtupleで返る() -> None:
    """観測列は tuple（不変な順序列）で返る。"""
    result = _FakeResult(_FakeAgent("billing"), [_FakeHandoffItem("triage", "billing")])

    observation = extract_turn_observation(result)

    assert isinstance(observation.handoffs, tuple)
    assert isinstance(observation.handoffs[0], tuple)


def test_extract_turn_observation_複数到達は観測順で保たれる() -> None:
    """2 段のハンドオフは宣言順ではなく観測順（triage -> billing -> tech）で並ぶ。"""
    result = _FakeResult(
        _FakeAgent("tech"),
        [
            _FakeHandoffItem("triage", "billing"),
            _FakeHandoffItem("billing", "tech"),
        ],
    )

    observation = extract_turn_observation(result)

    assert observation.handoffs == (("triage", "billing"), ("billing", "tech"))


def test_extract_turn_observation_観測順は入力順に追随する() -> None:
    """入力アイテムの順序を入れ替えると観測列の順序も入れ替わる（並べ替えをしない）。"""
    result = _FakeResult(
        _FakeAgent("billing"),
        [
            _FakeHandoffItem("triage", "tech"),
            _FakeHandoffItem("tech", "billing"),
        ],
    )

    observation = extract_turn_observation(result)

    assert observation.handoffs == (("triage", "tech"), ("tech", "billing"))


def test_extract_turn_observation_handoffが無いターンは空列() -> None:
    """ハンドオフ遷移が 1 件も無ければ観測列は空（最終回答者名のみが得られる）。"""
    result = _FakeResult(_FakeAgent("triage"), [_FakeMessageItem()])

    observation = extract_turn_observation(result)

    assert observation.handoffs == ()
    assert observation.last_agent == "triage"


def test_extract_turn_observation_new_itemsが空でも安全に返る() -> None:
    """`new_items` が空列でも例外にならず、観測列は空になる。"""
    result = _FakeResult(_FakeAgent("triage"), [])

    observation = extract_turn_observation(result)

    assert observation.handoffs == ()
    assert observation.last_agent == "triage"


def test_extract_turn_observation_戻り値はfrozenで書き換え不可() -> None:
    """観測は plain frozen dataclass であり、抽出後の改竄はできない。"""
    result = _FakeResult(_FakeAgent("billing"), [_FakeHandoffItem("triage", "billing")])

    observation = extract_turn_observation(result)

    assert dataclasses.is_dataclass(observation)
    with pytest.raises(dataclasses.FrozenInstanceError):
        observation.last_agent = "tech"  # type: ignore[misc]

    assert observation.last_agent == "billing"


# ---------------------------------------------------------------------------
# アイテム判定は属性の有無で行う（type リテラル非依存）
# ---------------------------------------------------------------------------


def test_extract_turn_observation_type文字列が別物でもhandoffとして拾う() -> None:
    """`type` が SDK リテラルと異なっても、属性を持つアイテムは handoff として拾われる。"""
    result = _FakeResult(
        _FakeAgent("billing"),
        [_FakeHandoffItem("triage", "billing", type_="totally-unknown-type")],
    )

    observation = extract_turn_observation(result)

    assert observation.handoffs == (("triage", "billing"),)


def test_extract_turn_observation_属性を持たないアイテムはtypeがhandoff風でも無視する() -> None:
    """判定は属性の有無で行うため、`type` だけ handoff 風のアイテムは観測されない。"""
    result = _FakeResult(_FakeAgent("triage"), [_FakeNonHandoffItem()])

    observation = extract_turn_observation(result)

    assert observation.handoffs == ()


def test_extract_turn_observation_片方の属性しか持たないアイテムは無視する() -> None:
    """`source_agent` / `target_agent` の双方が揃うアイテムのみを handoff とみなす。"""

    class _HalfItem:
        """遷移先だけを持つアイテム（handoff としては不完全）。"""

        def __init__(self) -> None:
            self.target_agent = _FakeAgent("billing")

    result = _FakeResult(_FakeAgent("billing"), [_HalfItem()])

    observation = extract_turn_observation(result)

    assert observation.handoffs == ()


def test_extract_turn_observation_handoff以外のアイテムが混在しても順序が保たれる() -> None:
    """メッセージ等が挟まっても handoff のみが観測順で抽出される。"""
    result = _FakeResult(
        _FakeAgent("tech"),
        [
            _FakeMessageItem("hello"),
            _FakeHandoffItem("triage", "billing"),
            _FakeNonHandoffItem(),
            _FakeHandoffItem("billing", "tech"),
            _FakeMessageItem("done"),
        ],
    )

    observation = extract_turn_observation(result)

    assert observation.handoffs == (("triage", "billing"), ("billing", "tech"))


# ---------------------------------------------------------------------------
# 防御的読み取り（属性欠落 / 属性アクセス時の例外）
# ---------------------------------------------------------------------------


def test_extract_turn_observation_last_agent属性の欠落はNoneへ倒す() -> None:
    """`last_agent` 属性が無くても例外を送出せず None を返す（handoff の抽出は継続する）。"""
    result = _ResultWithoutLastAgent([_FakeHandoffItem("triage", "billing")])

    observation = extract_turn_observation(result)

    assert observation.last_agent is None
    assert observation.handoffs == (("triage", "billing"),)


def test_extract_turn_observation_last_agentがNoneならNoneへ倒す() -> None:
    """`last_agent` が None（未確定）でも例外にならず None を返す。"""
    result = _FakeResult(None, [_FakeHandoffItem("triage", "billing")])

    observation = extract_turn_observation(result)

    assert observation.last_agent is None


def test_extract_turn_observation_last_agentのアクセス例外もNoneへ倒す() -> None:
    """属性アクセス自体が例外を送出しても、例外を伝播させず None を返す（NFR-5）。"""
    result = _ResultRaisingLastAgent([_FakeHandoffItem("triage", "billing")])

    observation = extract_turn_observation(result)

    assert observation.last_agent is None
    assert observation.handoffs == (("triage", "billing"),)


def test_extract_turn_observation_new_items属性の欠落は空列へ倒す() -> None:
    """`new_items` 属性が無くても例外を送出せず観測列は空になる（last_agent は取れる）。"""
    result = _ResultWithoutNewItems(_FakeAgent("billing"))

    observation = extract_turn_observation(result)

    assert observation.handoffs == ()
    assert observation.last_agent == "billing"


def test_extract_turn_observation_new_itemsのアクセス例外も空列へ倒す() -> None:
    """`new_items` の読み出しが例外を送出しても、例外を伝播させず観測列は空になる。"""
    result = _ResultRaisingNewItems(_FakeAgent("billing"))

    observation = extract_turn_observation(result)

    assert observation.handoffs == ()
    assert observation.last_agent == "billing"


def test_extract_turn_observation_new_itemsがNoneでも空列へ倒す() -> None:
    """`new_items` が None でも例外にならず観測列は空になる。"""
    result = _FakeResult(_FakeAgent("billing"), None)

    observation = extract_turn_observation(result)

    assert observation.handoffs == ()


def test_extract_turn_observation_new_itemsが非iterableでも空列へ倒す() -> None:
    """`new_items` が反復不能な値（int）でも `tuple()` 変換の例外を吸収し観測列は空になる。"""
    result = _FakeResult(_FakeAgent("billing"), 42)

    observation = extract_turn_observation(result)

    assert observation.handoffs == ()
    assert observation.last_agent == "billing"


class _RaisingIterItems:
    """`__iter__` 自体が例外を送出するオブジェクト（`tuple()` 変換失敗の代表例）。"""

    def __iter__(self) -> Any:
        """反復開始のたびに例外を送出する（`getattr` の既定値では吸収できない）。

        Raises:
            RuntimeError: 常に送出する。
        """
        raise RuntimeError("new_items is not iterable")


def test_extract_turn_observation_new_itemsのiter自体が例外でも空列へ倒す() -> None:
    """`__iter__` が例外を送出するオブジェクトでも `tuple()` 変換の例外を吸収し観測列は空になる。"""
    result = _FakeResult(_FakeAgent("billing"), _RaisingIterItems())

    observation = extract_turn_observation(result)

    assert observation.handoffs == ()
    assert observation.last_agent == "billing"


def test_extract_turn_observation_判定材料が両方取れなくても例外にしない() -> None:
    """`last_agent` も `new_items` も持たない結果でも安全側（None / 空列）へ倒す。"""

    class _Empty:
        """判定材料を一切持たない結果。"""

    observation = extract_turn_observation(_Empty())

    assert observation.last_agent is None
    assert observation.handoffs == ()


def test_extract_turn_observation_遷移元のアクセス例外は当該アイテムのみスキップする() -> None:
    """`source_agent` の読み出しが失敗するアイテムだけを飛ばし、他の到達は拾う。"""
    result = _FakeResult(
        _FakeAgent("tech"),
        [
            _RaisingSourceItem("billing"),
            _FakeHandoffItem("billing", "tech"),
        ],
    )

    observation = extract_turn_observation(result)

    assert observation.handoffs == (("billing", "tech"),)
    assert observation.last_agent == "tech"


def test_extract_turn_observation_遷移先のアクセス例外は当該アイテムのみスキップする() -> None:
    """`target_agent` の読み出しが失敗するアイテムだけを飛ばし、前後の到達は拾う。"""
    result = _FakeResult(
        _FakeAgent("tech"),
        [
            _FakeHandoffItem("triage", "billing"),
            _RaisingTargetItem("billing"),
            _FakeHandoffItem("billing", "tech"),
        ],
    )

    observation = extract_turn_observation(result)

    assert observation.handoffs == (("triage", "billing"), ("billing", "tech"))


# ---------------------------------------------------------------------------
# 純粋性（入力不変・決定的）
# ---------------------------------------------------------------------------


def test_extract_turn_observation_入力を変更しない() -> None:
    """抽出は読み取りのみで、結果オブジェクトと `new_items` を変更しない。"""
    agent = _FakeAgent("tech")
    items = [_FakeHandoffItem("triage", "billing"), _FakeHandoffItem("billing", "tech")]
    snapshot = list(items)
    result = _FakeResult(agent, items)

    extract_turn_observation(result)

    assert result.last_agent is agent
    assert result.new_items is items
    assert items == snapshot


def test_extract_turn_observation_同一入力で同一結果を返す() -> None:
    """同じ結果オブジェクトから何度抽出しても同じ観測になる（決定的）。"""
    result = _FakeResult(
        _FakeAgent("tech"),
        [_FakeHandoffItem("triage", "billing"), _FakeHandoffItem("billing", "tech")],
    )

    first = extract_turn_observation(result)
    second = extract_turn_observation(result)

    assert first.last_agent == second.last_agent == "tech"
    assert first.handoffs == second.handoffs == (("triage", "billing"), ("billing", "tech"))


# ---------------------------------------------------------------------------
# 到達記録ストア・合成（FR-3）用のフェイクとヘルパ
# ---------------------------------------------------------------------------


class _FakeWrapper:
    """`RunContextWrapper` の代役（identity hash を持ち弱参照できる普通のオブジェクト）。

    SDK は run ごとに wrapper を 1 つ生成し、handoff 実行と handoff 有効性評価の双方へ
    同一インスタンスを渡す。記録と参照のキーがこのインスタンスであることを模す。
    """

    def __init__(self, label: str = "run") -> None:
        self.label = label


async def _invoke(func: Any, *args: Any) -> Any:
    """SDK と同じ呼び方（同期戻り値ならそのまま・awaitable なら await）で呼ぶ。

    Args:
        func: 呼び出す callable（合成された `on_handoff` / `is_enabled`）。
        *args: 渡す引数。

    Returns:
        呼び出し結果（awaitable なら await した結果）。
    """
    result = func(*args)
    if inspect.isawaitable(result):
        result = await result
    return result


def _weak_map_of(store: Any) -> weakref.WeakKeyDictionary[Any, Any]:
    """ストアが内包する弱参照マップを取り出す（属性名には依存しない）。

    Args:
        store: 到達記録ストア。

    Returns:
        ストアが保持する `WeakKeyDictionary`。

    Raises:
        AssertionError: `WeakKeyDictionary` を保持していない場合（run 終了後に記録が
            解放される契約が満たせないため）。
    """
    for value in vars(store).values():
        if isinstance(value, weakref.WeakKeyDictionary):
            return value
    raise AssertionError("arrival store must hold a WeakKeyDictionary keyed by the run wrapper")


# ---------------------------------------------------------------------------
# 到達記録ストア: run 単位の分離・名前識別・弱参照
# ---------------------------------------------------------------------------


def test_arrival_store_記録前は到達していない() -> None:
    """何も記録していないストアはどの名前についても False を返す。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()

    assert store.has_arrived(ctx, "billing") is False


def test_arrival_store_記録した名前だけがTrueになる() -> None:
    """記録した名前は True・記録していない名前は False（名前で識別する）。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()

    store.record(ctx, "billing")

    assert store.has_arrived(ctx, "billing") is True
    assert store.has_arrived(ctx, "tech") is False


def test_arrival_store_同一wrapperに複数の名前を記録できる() -> None:
    """1 つの run で複数のエージェントに到達しても、それぞれ独立に記録される。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()

    store.record(ctx, "billing")
    store.record(ctx, "tech")

    assert store.has_arrived(ctx, "billing") is True
    assert store.has_arrived(ctx, "tech") is True
    assert store.has_arrived(ctx, "triage") is False


def test_arrival_store_同じ名前の二重記録は冪等() -> None:
    """同一ターンで同じ到達が 2 回記録されても状態は変わらない（エラーにしない）。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()

    store.record(ctx, "billing")
    store.record(ctx, "billing")

    assert store.has_arrived(ctx, "billing") is True


def test_arrival_store_別のwrapperでは記録が分離される() -> None:
    """run ごとに wrapper が異なるため、run A の記録は run B から見えない（並行 run 分離）。"""
    store = ArrivalStore()
    run_a = _FakeWrapper("a")
    run_b = _FakeWrapper("b")

    store.record(run_a, "billing")

    assert store.has_arrived(run_a, "billing") is True
    assert store.has_arrived(run_b, "billing") is False


def test_arrival_store_インスタンスは互いに独立() -> None:
    """ストアはモジュールグローバルではなく、別インスタンスの記録は共有されない。"""
    first = ArrivalStore()
    second = ArrivalStore()
    ctx = _FakeWrapper()

    first.record(ctx, "billing")

    assert first.has_arrived(ctx, "billing") is True
    assert second.has_arrived(ctx, "billing") is False


def test_arrival_store_wrapperの解放で記録も解放される() -> None:
    """記録は run 内一時状態であり、wrapper が回収されるとエントリも消える（弱参照）。"""
    store = ArrivalStore()
    ctx: Any = _FakeWrapper()
    store.record(ctx, "billing")
    weak_map = _weak_map_of(store)
    assert len(weak_map) == 1

    del ctx
    gc.collect()

    assert len(weak_map) == 0


# ---------------------------------------------------------------------------
# 記録 on_handoff 合成: arity（SDK の署名検証を通過する形）
# ---------------------------------------------------------------------------


def test_arrival_recorder_input_typeなしはちょうど1引数() -> None:
    """`input_type` なしのエッジ用は `(ctx)` の 1 引数（SDK の署名検証は != 1 を拒否する）。"""
    recorder = make_arrival_recorder(ArrivalStore(), "billing", None, False)

    assert len(inspect.signature(recorder).parameters) == 1


def test_arrival_recorder_input_typeありはちょうど2引数() -> None:
    """`input_type` ありのエッジ用は `(ctx, input)` の 2 引数（SDK は != 2 を拒否する）。"""
    recorder = make_arrival_recorder(ArrivalStore(), "billing", None, True)

    assert len(inspect.signature(recorder).parameters) == 2


def test_arrival_recorder_利用者on_handoff付きでも引数個数は変わらない() -> None:
    """chain しても arity はエッジの `input_type` 有無だけで決まる（署名検証を壊さない）。"""
    store = ArrivalStore()

    def _user_one(ctx: Any) -> None:
        return None

    def _user_two(ctx: Any, payload: Any) -> None:
        return None

    one = make_arrival_recorder(store, "billing", _user_one, False)
    two = make_arrival_recorder(store, "billing", _user_two, True)

    assert len(inspect.signature(one).parameters) == 1
    assert len(inspect.signature(two).parameters) == 2


# ---------------------------------------------------------------------------
# 記録 on_handoff 合成: 記録と利用者 on_handoff の chain
# ---------------------------------------------------------------------------


async def test_arrival_recorder_1引数形の呼び出しで記録される() -> None:
    """利用者 `on_handoff` が無くても、呼び出しで当該 run の到達が記録される。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()
    recorder = make_arrival_recorder(store, "billing", None, False)

    await _invoke(recorder, ctx)

    assert store.has_arrived(ctx, "billing") is True
    assert store.has_arrived(ctx, "tech") is False


async def test_arrival_recorder_2引数形の呼び出しで記録される() -> None:
    """`input_type` ありの 2 引数形でも同じく到達が記録される。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()
    recorder = make_arrival_recorder(store, "billing", None, True)

    await _invoke(recorder, ctx, {"reason": "escalation"})

    assert store.has_arrived(ctx, "billing") is True


async def test_arrival_recorder_記録は呼び出した_run_にだけ入る() -> None:
    """記録キーは呼び出しで渡された wrapper であり、他の run へは波及しない。"""
    store = ArrivalStore()
    run_a = _FakeWrapper("a")
    run_b = _FakeWrapper("b")
    recorder = make_arrival_recorder(store, "billing", None, False)

    await _invoke(recorder, run_a)

    assert store.has_arrived(run_a, "billing") is True
    assert store.has_arrived(run_b, "billing") is False


async def test_arrival_recorder_記録は利用者on_handoffより先に行われる() -> None:
    """chain の順序は「記録 -> 利用者 `on_handoff`」（利用者側から到達済みが見える）。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()
    calls: list[tuple[str, bool]] = []

    def _user(received: Any) -> None:
        calls.append(("user", store.has_arrived(received, "billing")))

    recorder = make_arrival_recorder(store, "billing", _user, False)

    await _invoke(recorder, ctx)

    assert calls == [("user", True)]


async def test_arrival_recorder_1引数形は利用者on_handoffへctxを透過する() -> None:
    """利用者 `on_handoff` は SDK から渡ったのと同一の ctx を受け取る。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()
    seen: list[Any] = []

    def _user(received: Any) -> None:
        seen.append(received)

    recorder = make_arrival_recorder(store, "billing", _user, False)

    await _invoke(recorder, ctx)

    assert len(seen) == 1
    assert seen[0] is ctx


async def test_arrival_recorder_2引数形は利用者on_handoffへinputも透過する() -> None:
    """2 引数形では ctx と input の双方が同一性を保って透過される。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()
    payload = object()
    seen: list[tuple[Any, Any]] = []

    def _user(received_ctx: Any, received_input: Any) -> None:
        seen.append((received_ctx, received_input))

    recorder = make_arrival_recorder(store, "billing", _user, True)

    await _invoke(recorder, ctx, payload)

    assert len(seen) == 1
    assert seen[0][0] is ctx
    assert seen[0][1] is payload


async def test_arrival_recorder_async利用者on_handoffもawaitされる() -> None:
    """利用者 `on_handoff` が async の場合、本体が最後まで実行される（await される）。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()
    done: list[str] = []

    async def _user(received: Any) -> None:
        await asyncio.sleep(0)
        done.append("user")

    recorder = make_arrival_recorder(store, "billing", _user, False)

    await _invoke(recorder, ctx)

    assert done == ["user"]
    assert store.has_arrived(ctx, "billing") is True


async def test_arrival_recorder_2引数形のasync利用者on_handoffもawaitされる() -> None:
    """2 引数形でも async の利用者 `on_handoff` は await され、input が透過される。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()
    payload = object()
    seen: list[Any] = []

    async def _user(received_ctx: Any, received_input: Any) -> None:
        await asyncio.sleep(0)
        seen.append(received_input)

    recorder = make_arrival_recorder(store, "billing", _user, True)

    await _invoke(recorder, ctx, payload)

    assert len(seen) == 1
    assert seen[0] is payload


# ---------------------------------------------------------------------------
# is_enabled ゲート合成: シグネチャと記録済み / 未記録の分岐
# ---------------------------------------------------------------------------


def test_arrival_gate_シグネチャはちょうど2引数() -> None:
    """SDK は bool 以外の `is_enabled` を必ず `(ctx, agent)` の 2 引数で呼ぶ。"""
    gate = make_arrival_gate(ArrivalStore(), "billing", True)

    assert len(inspect.signature(gate).parameters) == 2


async def test_arrival_gate_記録済みならFalse() -> None:
    """当該 run で X へ到達済みなら、既存が True でもゲートは False（禁止の発動）。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()
    agent = _FakeAgent("billing")
    gate = make_arrival_gate(store, "billing", True)

    store.record(ctx, "billing")

    assert await _invoke(gate, ctx, agent) is False


async def test_arrival_gate_未記録なら既存Trueへ委譲する() -> None:
    """未到達なら既存 `is_enabled`（True）の評価へ委譲する（常に False にしない）。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()
    gate = make_arrival_gate(store, "billing", True)

    assert await _invoke(gate, ctx, _FakeAgent("billing")) is True


async def test_arrival_gate_未記録で既存Falseなら_False() -> None:
    """既存宣言が False なら、未到達でも False のまま（既存宣言を無視して True にしない）。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()
    gate = make_arrival_gate(store, "billing", False)

    assert await _invoke(gate, ctx, _FakeAgent("billing")) is False


async def test_arrival_gate_未記録なら同期callableの戻り値へ委譲する() -> None:
    """既存が callable なら、その戻り値（True / False）がそのままゲートの結果になる。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()

    def _enabled(received_ctx: Any, received_agent: Any) -> bool:
        return True

    def _disabled(received_ctx: Any, received_agent: Any) -> bool:
        return False

    enabled_gate = make_arrival_gate(store, "billing", _enabled)
    disabled_gate = make_arrival_gate(store, "billing", _disabled)

    assert await _invoke(enabled_gate, ctx, _FakeAgent("billing")) is True
    assert await _invoke(disabled_gate, ctx, _FakeAgent("billing")) is False


async def test_arrival_gate_委譲時は既存callableへ同じ引数が渡る() -> None:
    """既存 callable には SDK から渡った `(ctx, agent)` がそのまま透過される。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()
    agent = _FakeAgent("billing")
    seen: list[tuple[Any, Any]] = []

    def _existing(received_ctx: Any, received_agent: Any) -> bool:
        seen.append((received_ctx, received_agent))
        return True

    gate = make_arrival_gate(store, "billing", _existing)

    await _invoke(gate, ctx, agent)

    assert len(seen) == 1
    assert seen[0][0] is ctx
    assert seen[0][1] is agent


async def test_arrival_gate_未記録ならasync_callableの戻り値へ委譲する() -> None:
    """既存が async callable でも await され、その戻り値がゲートの結果になる。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()

    async def _existing(received_ctx: Any, received_agent: Any) -> bool:
        await asyncio.sleep(0)
        return False

    gate = make_arrival_gate(store, "billing", _existing)

    assert await _invoke(gate, ctx, _FakeAgent("billing")) is False


async def test_arrival_gate_記録済みなら既存callableを評価しない() -> None:
    """到達済みの判定が先で、既存 `is_enabled` の評価には進まない（短絡）。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()
    calls: list[str] = []

    def _existing(received_ctx: Any, received_agent: Any) -> bool:
        calls.append("existing")
        return True

    gate = make_arrival_gate(store, "billing", _existing)
    store.record(ctx, "billing")

    assert await _invoke(gate, ctx, _FakeAgent("billing")) is False
    assert calls == []


# ---------------------------------------------------------------------------
# is_enabled ゲート合成: 判定は closure の X 名と run で行う
# ---------------------------------------------------------------------------


async def test_arrival_gate_第2引数がNoneでも判定は変わらない() -> None:
    """判定は closure に閉じた X 名で行うため、第 2 引数の内容に依存しない。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()
    gate = make_arrival_gate(store, "billing", True)

    assert await _invoke(gate, ctx, None) is True

    store.record(ctx, "billing")

    assert await _invoke(gate, ctx, None) is False


async def test_arrival_gate_第2引数が無関係なagentでも判定は変わらない() -> None:
    """所有側でない Agent を渡されても、記録済みなら False・未記録なら委譲のまま。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()
    unrelated = _FakeAgent("unrelated")
    gate = make_arrival_gate(store, "billing", True)

    assert await _invoke(gate, ctx, unrelated) is True

    store.record(ctx, "billing")

    assert await _invoke(gate, ctx, unrelated) is False


async def test_arrival_gate_別のrunでは委譲側に倒れる() -> None:
    """記録は run 単位のため、別 wrapper の評価では禁止が働かない（ターンを越えない）。"""
    store = ArrivalStore()
    run_a = _FakeWrapper("a")
    run_b = _FakeWrapper("b")
    gate = make_arrival_gate(store, "billing", True)

    store.record(run_a, "billing")

    assert await _invoke(gate, run_a, _FakeAgent("billing")) is False
    assert await _invoke(gate, run_b, _FakeAgent("billing")) is True


async def test_arrival_gate_別の名前の記録では委譲側に倒れる() -> None:
    """他エージェントへの到達記録では発動しない（名前で識別する）。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()
    gate = make_arrival_gate(store, "billing", True)

    store.record(ctx, "tech")

    assert await _invoke(gate, ctx, _FakeAgent("billing")) is True


async def test_arrival_gate_記録と参照は同一storeを共有する() -> None:
    """記録 `on_handoff` の合成とゲートの合成は同じストアを見る（結線の一貫性）。"""
    store = ArrivalStore()
    ctx = _FakeWrapper()
    recorder = make_arrival_recorder(store, "billing", None, False)
    gate = make_arrival_gate(store, "billing", True)

    assert await _invoke(gate, ctx, _FakeAgent("billing")) is True

    await _invoke(recorder, ctx)

    assert await _invoke(gate, ctx, _FakeAgent("billing")) is False


# ---------------------------------------------------------------------------
# 記録 on_handoff 合成: 2 引数形の記録順序と利用者宣言の arity 検証
# ---------------------------------------------------------------------------


async def test_arrival_recorder_2引数形でも記録は利用者on_handoffより先に行われる() -> None:
    """`input_type` あり経路でも chain の順序は「記録 -> 利用者 `on_handoff`」。

    1 引数形と対で pin する（片側だけだと 2 引数形の順序反転を検知できない）。
    """
    store = ArrivalStore()
    ctx = _FakeWrapper()
    calls: list[tuple[str, bool]] = []

    def _user(received: Any, payload: Any) -> None:
        calls.append(("user", store.has_arrived(received, "billing")))

    recorder = make_arrival_recorder(store, "billing", _user, True)

    await _invoke(recorder, ctx, {"reason": "escalation"})

    assert calls == [("user", True)]


def test_arrival_recorder_利用者on_handoffのarity不一致はbuild時に_UserError() -> None:
    """`input_type` なしのエッジに 2 引数の `on_handoff` を宣言したら build 時に落ちる。

    合成すると SDK の署名検証は合成 callable しか見ないため、利用者宣言の arity 誤りは
    run 中の `TypeError` へ後ろ倒しになる。合成側で同じ期待値を検証して build 時に戻す。
    例外型も SDK の `handoff()` と同じ `UserError` に揃える（禁止を宣言したエッジだけ
    型が変わると利用者の `except UserError` をすり抜けるため）。
    """

    def _user_two(ctx: Any, payload: Any) -> None:
        return None

    with pytest.raises(UserError, match="one argument"):
        make_arrival_recorder(ArrivalStore(), "billing", _user_two, False)


def test_arrival_recorder_input_typeありに1引数のon_handoffもbuild時に_UserError() -> None:
    """`input_type` ありのエッジに 1 引数の `on_handoff` を宣言した場合も build 時に落ちる。"""

    def _user_one(ctx: Any) -> None:
        return None

    with pytest.raises(UserError, match="two arguments"):
        make_arrival_recorder(ArrivalStore(), "billing", _user_one, True)


def test_arrival_recorder_arityが一致する宣言は受理される() -> None:
    """期待どおりの arity（なし = 1 / あり = 2）なら検証を通過する（拒否しすぎない）。"""

    def _user_one(ctx: Any) -> None:
        return None

    def _user_two(ctx: Any, payload: Any) -> None:
        return None

    assert make_arrival_recorder(ArrivalStore(), "billing", _user_one, False) is not None
    assert make_arrival_recorder(ArrivalStore(), "billing", _user_two, True) is not None


def test_arrival_recorder_署名を取得できないon_handoffは例外をそのまま伝播する() -> None:
    """`inspect.signature` が失敗する callable はスキップせず SDK と同じ例外を伝播する。

    SDK の `handoff()` は `inspect.signature` を無防備に呼ぶため、同じ宣言は禁止を宣言しない
    エッジでは build 時に落ちる。合成側でスキップすると「SDK なら落ちる宣言が禁止対象エッジで
    だけ通り、arity 誤りが run 中の `TypeError` へ後退する」非対称になる。C 実装のビルトイン
    （`time.time` は `inspect.signature` が `ValueError` を送出する）を代表例として使う。
    """
    with pytest.raises(ValueError, match="no signature found for builtin"):
        make_arrival_recorder(ArrivalStore(), "billing", time.time, False)


# ---------------------------------------------------------------------------
# 防御的読み取り: 名前の str 変換も防御の内側で行う
# ---------------------------------------------------------------------------


class _RaisingStrName:
    """`__str__` が例外を送出する名前オブジェクト（防御の外に変換が漏れていないかの検証用）。"""

    def __str__(self) -> str:
        """変換のたびに例外を送出する。

        Raises:
            RuntimeError: 常に送出する。
        """
        raise RuntimeError("name is not renderable")


class _NameRaisingAgent:
    """`name` の値が `str()` 変換で例外を送出するエージェント代役。"""

    def __init__(self) -> None:
        self.name = _RaisingStrName()


def test_extract_turn_observation_last_agentのstr変換例外もNoneへ倒す() -> None:
    """`name` の `__str__` が例外を送出しても抽出は例外を伝播させない（NFR-5）。"""
    result = _FakeResult(_NameRaisingAgent(), [])

    observation = extract_turn_observation(result)

    assert observation.last_agent is None


def test_extract_turn_observation_遷移元のstr変換例外は当該アイテムのみスキップする() -> None:
    """handoff アイテム側の `name` 変換が失敗しても、他の到達は観測される。"""

    class _BrokenNameItem:
        """遷移元の名前が変換できないアイテム。"""

        def __init__(self) -> None:
            self.source_agent = _NameRaisingAgent()
            self.target_agent = _FakeAgent("billing")

    result = _FakeResult(
        _FakeAgent("tech"), [_BrokenNameItem(), _FakeHandoffItem("billing", "tech")]
    )

    observation = extract_turn_observation(result)

    assert observation.handoffs == (("billing", "tech"),)


def test_arrival_recorder_非callableのon_handoffはinput_typeなしなら_TypeError() -> None:
    """`input_type` なしの経路は SDK と同じく `callable()` を見ずに署名検査へ渡す。

    SDK の `handoff()` は `callable()` チェックを `input_type` あり経路にしか持たず、なし経路の
    非 callable は `inspect.signature` 由来の `TypeError` になる。lib 側で先行チェックを足すと
    禁止対象エッジだけ例外型が変わるため、分岐構造ごと SDK に合わせる（build 時に落ちること
    自体は署名検査の例外伝播で担保される）。
    """
    with pytest.raises(TypeError, match="is not a callable object"):
        make_arrival_recorder(ArrivalStore(), "billing", "not-callable", False)


def test_arrival_recorder_非callableのon_handoffはinput_typeありなら_UserError() -> None:
    """`input_type` あり経路は SDK と同じく `callable()` を署名検査より前に見る。"""
    with pytest.raises(UserError, match="must be callable"):
        make_arrival_recorder(ArrivalStore(), "billing", "not-callable", True)


def test_arrival_recorder_on_handoff未宣言は検証対象外() -> None:
    """利用者宣言が無ければ記録のみを合成する（`input_type` 有無によらず factory は受理する）。

    SDK の「`input_type` あり -> `on_handoff` 必須」は**利用者宣言に対する**要件であり、
    合成の差し込み口を持つ registry 側で判定する（pin は
    `test_next_turn_registry_l2.py`）。ここは合成 callable の arity 契約のみを見る。
    """
    assert (
        len(inspect.signature(make_arrival_recorder(ArrivalStore(), "b", None, False)).parameters)
        == 1
    )
    assert (
        len(inspect.signature(make_arrival_recorder(ArrivalStore(), "b", None, True)).parameters)
        == 2
    )
