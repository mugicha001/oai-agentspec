"""L1: ADR-0021（宣言 dataclass の bool フィールド構築時型検証）の網羅性メタテスト。

ADR-0021 Confirmation 節が名指す強制手段であり、`docs/QUALITY-GUARANTEES.md` の登録行が
指すファイルである。目的は「新しく追加された bool フィールドが規則から静かに漏れる退行」を
機械検知することであり、個々のクラスの拒否 / 受理はミラー位置のテストが別途担保する。

走査規則（ADR-0021 Decision 1 の対象スコープと同一。規則の外に個別の例外を作らない）:

1. `src/oai_agentspec/` 配下（ルートの `__init__.py` を含む）で定義される dataclass の
   フィールド（`oai_agentspec._adapters` 配下は SDK 隔離境界の内側の実装詳細のため除外する）
2. `dataclasses.fields()` の `field.init is True`（`init=False` は構築時に値を渡せないため
   対象外。実例: `workflow/graph.py` の `_frozen`）
3. 型注釈が `bool` または `bool | None`

実装上の前提（崩れると走査が静かに空振りするため明記する）:

- 注釈は `field.type` の**文字列**で判定する。`typing.get_type_hints` は使わない
  （`from __future__ import annotations` により注釈が文字列化されており、`TYPE_CHECKING`
  限定 import を含む型（例: `CompactionConfig.client: AsyncOpenAI`）の実行時解決に失敗する）。
- ただし `field.type` は常に文字列とは限らない。モジュールが `from __future__ import
  annotations` を持たない場合、`field.type` は型オブジェクト（`bool` クラス /
  `types.UnionType`）になる。文字列一致だけで判定すると、そうしたモジュールのクラスが
  **静かに走査から落ちる**（下限チェックでも 1 クラスの欠落は検知できない）。そのため
  `_annotation_key()` で文字列・型オブジェクトの双方を同じ正規形へ落としてから判定する。
- 正規形は「`|` で分割 -> 各要素を strip -> ソートして `" | "` で連結」であり、`ruff format`
  の整形表記（`|` の前後のスペース有無）と Union の要素順に依存しない。`bool | None` は
  `"None | bool"` に正規化される（`_OPTIONAL_BOOL` 参照）。
- 走査は**全 extras 導入済みのテスト環境**を前提とする（extra 未導入環境では対象モジュールを
  import できず、走査対象が黙って縮む）。import エラーは握りつぶさず伝播させる。
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
import re
from typing import Any

import pytest

import oai_agentspec

pytestmark = pytest.mark.unit

# SDK 隔離境界の内側（利用者の宣言が届かない実装詳細）は ADR-0021 の対象外。
_EXCLUDED_PREFIX = "oai_agentspec._adapters"

# 注釈の正規形（`_annotation_key()` の出力。モジュール docstring 参照）。
_BOOL = "bool"
_OPTIONAL_BOOL = "None | bool"
_BOOL_ANNOTATIONS = frozenset({_BOOL, _OPTIONAL_BOOL})

# 走査の空振り検知に使う下限（現時点の実装は 15 クラス / 22 フィールド）。件数の直書きは
# 新フィールド追加のたびに壊れる脆い pin になるため、下限のみを置く。
_MIN_TARGET_CLASSES = 15
_MIN_TARGET_FIELDS = 22


def _annotation_key(raw: object) -> str:
    """注釈（文字列 / 型オブジェクト）を比較用の正規形へ落とす。

    `from __future__ import annotations` の有無で `field.type` は文字列にも型オブジェクトにも
    なる。型オブジェクトは `bool` クラスなら `__name__`（`"bool"`）、`bool | None`
    （`types.UnionType`）なら `str()`（`"bool | None"`）を経由し、いずれも文字列注釈と同じ
    正規形へ収束する。

    Args:
        raw: `dataclasses.Field.type` の値。

    Returns:
        `|` 区切りの要素を strip・ソートして連結した正規形の文字列。
    """
    text = raw if isinstance(raw, str) else getattr(raw, "__name__", str(raw))
    return " | ".join(sorted(part.strip() for part in text.split("|")))


def _sample_tool() -> str:
    """`ToolSpec.func` に渡すダミーの関数（構築が通れば十分で呼び出しはしない）。"""
    return "ok"


def _valid_kwargs_map() -> dict[str, dict[str, Any]]:
    """対象クラスごとの「構築が成功する最小 kwargs」を返す（毎回新しい値を組む）。

    キーは `f"{module}.{qualname}"`。走査で見つかったクラスがこの map に無ければ
    網羅検査が失敗し、新クラス追加時の検証漏れ（新たな非対称）を機械検知する。

    Returns:
        クラス識別子から正当 kwargs への map。
    """
    from oai_agentspec.runtime.llmops.types import ObservedRoute

    return {
        "oai_agentspec.next_turn.NextTurnRule": {"next_agent": "planner"},
        "oai_agentspec.tool_registry.ToolSpec": {"func": _sample_tool},
        # client は enabled=True の整合検証（client 必須）を通すために渡す。
        "oai_agentspec.runtime.conversation.session.CompactionConfig": {"client": object()},
        "oai_agentspec.runtime.conversation.session.SessionPolicy": {},
        "oai_agentspec.runtime.conversation.store.ConversationEntry": {
            "conversation_id": "conv-1",
            "session_id": "sess-1",
            "session": object(),
        },
        "oai_agentspec.runtime.conversation.types.ApprovalDecision": {
            "call_id": "call-1",
            "approve": True,
        },
        "oai_agentspec.runtime.guardrails._detectors.Detection": {"triggered": True},
        "oai_agentspec.runtime.lightning.config.OptimizeConfig": {},
        "oai_agentspec.runtime.lightning.types.CoverageReport": {
            "covered": frozenset({"a"}),
            "missing": frozenset(),
            "per_case": (),
            "interrupted_cases": 0,
        },
        "oai_agentspec.runtime.lightning.types.SlotSegment": {
            "ref": "base:main",
            "text": "seed",
            "tune": True,
        },
        "oai_agentspec.runtime.llmops.config.EvaluationConfig": {},
        "oai_agentspec.runtime.llmops.criteria.Criterion": {"name": "accuracy"},
        "oai_agentspec.runtime.llmops.types.ObservedRun": {
            "route": ObservedRoute(steps=[], last_agent="bot")
        },
        "oai_agentspec.runtime.resilience._failsafe.FailsafePolicy": {},
        "oai_agentspec.runtime.resilience._types.ModelRetryPolicy": {},
    }


def _class_key(cls: type) -> str:
    """クラスの正規化キー（再エクスポート由来の重複を 1 件に畳む）。"""
    return f"{cls.__module__}.{cls.__qualname__}"


def _iter_modules() -> list[str]:
    """`oai_agentspec` 配下の全モジュール名を返す（`_adapters` 配下は除外）。

    `walk_packages` はルートパッケージ自身を yield しないため、場所規則（`src/oai_agentspec/`
    配下 = ルートの `__init__.py` を含む）と一致させるべく `oai_agentspec` を明示的に加える。

    import エラーは握りつぶさない（走査対象が静かに減るのを防ぐため `onerror` で再送出する）。
    """

    def _reraise(name: str) -> None:
        # pkgutil は import 失敗の except 節から onerror を呼ぶため、bare raise で伝播できる。
        raise

    names = [oai_agentspec.__name__] + [
        info.name
        for info in pkgutil.walk_packages(
            oai_agentspec.__path__, prefix="oai_agentspec.", onerror=_reraise
        )
        if not info.name.startswith(_EXCLUDED_PREFIX)
    ]
    return sorted(names)


def _scan_bool_fields() -> dict[str, tuple[type, list[tuple[str, str]]]]:
    """走査規則を満たす「bool 注釈フィールドを持つ dataclass」を収集する。

    Returns:
        クラスキー -> (クラス, [(フィールド名, 注釈文字列), ...]) の map。
    """
    found: dict[str, tuple[type, list[tuple[str, str]]]] = {}
    for module_name in _iter_modules():
        module = importlib.import_module(module_name)
        for obj in vars(module).values():
            if not inspect.isclass(obj) or not dataclasses.is_dataclass(obj):
                continue
            defining_module = getattr(obj, "__module__", "")
            # 定義元が `oai_agentspec` 配下（ルート自身を含み `_adapters` を除く）のみ対象。
            if not (
                defining_module == oai_agentspec.__name__
                or defining_module.startswith(f"{oai_agentspec.__name__}.")
            ):
                continue
            if defining_module.startswith(_EXCLUDED_PREFIX):
                continue
            key = _class_key(obj)
            if key in found:
                continue
            bool_fields = [
                (f.name, _annotation_key(f.type))
                for f in dataclasses.fields(obj)
                if f.init and _annotation_key(f.type) in _BOOL_ANNOTATIONS
            ]
            if bool_fields:
                found[key] = (obj, bool_fields)
    return found


_SCANNED = _scan_bool_fields()


def _field_cases() -> list[tuple[str, type, str, str]]:
    """`(クラスキー, クラス, フィールド名, 注釈)` の平坦なリストを返す（parametrize 用）。"""
    return [
        (key, cls, field_name, annotation)
        for key, (cls, fields) in sorted(_SCANNED.items())
        for field_name, annotation in fields
    ]


_FIELD_CASES = _field_cases()
_FIELD_IDS = [f"{key.rsplit('.', 1)[-1]}.{name}" for key, _cls, name, _ann in _FIELD_CASES]


def _build(cls: type, key: str, field_name: str, value: Any) -> Any:
    """正当 kwargs の当該フィールドだけを `value` で上書きして構築する。"""
    kwargs = dict(_valid_kwargs_map()[key])
    kwargs[field_name] = value
    return cls(**kwargs)


def test_走査が空振りしていないこと() -> None:
    """走査規則が対象を 1 件も拾えていない（表記変更・extra 未導入等）状態を検知する。"""
    assert _SCANNED, "bool 注釈フィールドを持つ dataclass が 1 件も見つからない（走査の空振り）"
    assert len(_SCANNED) >= _MIN_TARGET_CLASSES, sorted(_SCANNED)
    # フィールド総数も下限で見る（クラス数の下限だけでは、既存クラスから bool フィールドが
    # 落ちた退行を検知できない）。等値 pin ではなく下限のため新規フィールド追加では壊れない。
    assert len(_FIELD_CASES) >= _MIN_TARGET_FIELDS, _FIELD_IDS


def test_走査で見つかった全クラスが正当kwargsマップに登録されている() -> None:
    """新規 bool フィールドを持つクラスの検証漏れ（新たな非対称）を機械検知する。"""
    unregistered = sorted(set(_SCANNED) - set(_valid_kwargs_map()))

    assert unregistered == [], (
        "bool 注釈フィールドを持つ次のクラスが _valid_kwargs_map() に未登録です"
        f"（ADR-0021 の対象スコープに入るため拒否 / 受理の検証が必要）: {unregistered}"
    )


def test_正当kwargsマップに走査対象外のクラスが残っていない() -> None:
    """クラス削除・改名時に map の死んだエントリが残らないようにする。"""
    stale = sorted(set(_valid_kwargs_map()) - set(_SCANNED))

    assert stale == [], f"_valid_kwargs_map() のエントリが走査結果に存在しません: {stale}"


@pytest.mark.parametrize(("key", "cls", "field_name", "annotation"), _FIELD_CASES, ids=_FIELD_IDS)
def test_bool注釈フィールドは非bool値を構築時に拒否する(
    key: str, cls: type, field_name: str, annotation: str
) -> None:
    """全対象フィールドで非 bool 値の構築が `ValueError` になることを検証する。

    `bool` 注釈は `None` / `"no"` / `0`（int）を、`bool | None` 注釈は `"no"` / `0` を拒否する
    （None は None-omission の正当値のため代わりに文字列で試す）。
    """
    invalid_values: list[Any] = ["no", 0]
    if annotation == _BOOL:
        invalid_values.append(None)

    for value in invalid_values:
        with pytest.raises(ValueError, match=re.escape(f"{field_name} must be a bool")):
            _build(cls, key, field_name, value)


@pytest.mark.parametrize(("key", "cls", "field_name", "annotation"), _FIELD_CASES, ids=_FIELD_IDS)
def test_bool注釈フィールドはboolを構築時に受理する(
    key: str, cls: type, field_name: str, annotation: str
) -> None:
    """全対象フィールドで `True` / `False`（`bool | None` は加えて `None`）が受理される。"""
    valid_values: list[Any] = [True, False]
    if annotation == _OPTIONAL_BOOL:
        valid_values.append(None)

    for value in valid_values:
        instance = _build(cls, key, field_name, value)
        assert getattr(instance, field_name) is value
