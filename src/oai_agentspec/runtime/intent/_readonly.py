"""frozen な宣言型が持つ `Mapping` フィールドを読み取り専用へ正規化する型エイリアス。

`model_config = {"frozen": True}` は属性の**再束縛**だけを禁じ、フィールドが保持する
`Mapping` の中身は守らない。`spec.prompt_vars["tone"] = "..."` のような書き込みが通ると、
起動時検証（検査 6 / 7）を通過した宣言を、実際に LLM プロンプトへ展開される直前へ差し替え
られる。同じ規則を「防御的コピー + 読み取り専用ビュー」の 2 段で適用するための置き場である。

2 段が両方必要である理由:

- コピーを挟まないと、呼び出し側が握ったままの dict への変更が宣言へ透ける。
- ビューを被せないと、フィールド経由で中身を書き換えられる。

ビューに `types.MappingProxyType` を使わないのは、`mappingproxy` が pickle 不可であり
`copy.deepcopy` / `model_copy(deep=True)` / `pickle` の 3 経路が `TypeError` になるためで
ある。宣言と候補は利用者コードやセッション層を渡り歩く公開型であり、複製・永続化は起こる。
代わりに `__deepcopy__` / `__reduce__` を持つ薄い `Mapping` 実装（`_ReadOnlyMapping`）を
置き、3 経路すべてで「値が保たれ・読み取り専用のまま・元とは別実体」を満たす。

同じ規則を要するフィールドは `ActionSpec.prompt_vars` / `ActionPlanner.prompt_vars` /
`ActionCatalog.prompt_vars`（`actions.py`）、`ExecutableIntent.parameters` /
`ExecutableSuggestion.metadata`（`types.py`）、`ActionPlan.resolved_prompt_vars`
（`slots.py`）の 6 件ある。validator を各所へ複製すると
片方だけが直される drift になるため、正規化と直列化を束ねた `Annotated` エイリアスをここへ
置いて共有する（pydantic を経由しない `ActionCatalog` は `_to_read_only_mapping` を直接使う）。

正規化は validator であるため、`model_copy(update=...)` では効かない（pydantic の仕様として
`model_copy` は validator を通らない）。派生を作る場合は `model_validate` 経由にすること。

本モジュールはドメイン型を 1 つも import しない葉であり、`actions.py` / `types.py` /
`slots.py` のどれから読んでも循環しない。
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from typing import Annotated, Any

from pydantic.functional_serializers import PlainSerializer
from pydantic.functional_validators import AfterValidator


class _ReadOnlyMapping(Mapping[str, Any]):
    """複製・pickle を通せる読み取り専用の `Mapping`（`MappingProxyType` の代替）。

    書き込み系メソッド（`__setitem__` / `__delitem__` / `update` 等）を一切持たないため、
    `m["k"] = v` は `TypeError` になる。読み取りは `Mapping` の既定実装（`get` / `items` /
    `==` など）がそのまま使える。

    `__deepcopy__` / `__reduce__` を持つのは、`mappingproxy` が pickle 不可で
    `copy.deepcopy` / `model_copy(deep=True)` / `pickle` を落としてしまうためである
    （`_failsafe._RunningAgentSentinel.__reduce__` と同じ体裁で、複製の意味論を型の側に
    持たせる）。どちらも**新しい実体**を返す。`self` を返す実装にすると、値の型が `Any` で
    可変値（list / dict）を載せられるフィールドで片方の変更が他方へ透ける。
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        """防御的コピーを取り込む。

        Args:
            data: 取り込む Mapping。呼び出し側が握ったままの実体は保持しない。
        """
        self._data: dict[str, Any] = dict(data)

    def __getitem__(self, key: str) -> Any:
        """キーに対応する値を返す。

        Args:
            key: 参照するキー。

        Returns:
            対応する値。

        Raises:
            KeyError: 当該キーが無い場合。
        """
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        """キーを挿入順に反復する。

        Returns:
            キーのイテレータ。
        """
        return iter(self._data)

    def __len__(self) -> int:
        """要素数を返す。

        Returns:
            保持している要素数。
        """
        return len(self._data)

    def __repr__(self) -> str:
        """中身が読める形の repr を返す。

        `Mapping` は `__repr__` を提供せず、既定のままだと `<_ReadOnlyMapping object at
        0x...>` になる。本型は `ExecutableIntent.parameters` など公開型のフィールド値として
        `repr()` へ現れるため、`MappingProxyType` と同等に中身が読める形を保つ。

        Returns:
            `_ReadOnlyMapping({...})` 形式の文字列。
        """
        return f"{type(self).__name__}({self._data!r})"

    def __deepcopy__(self, memo: dict[int, Any]) -> _ReadOnlyMapping:
        """中身まで複製した**別実体**の読み取り専用 Mapping を返す。

        Args:
            memo: `copy.deepcopy` が渡す循環参照の記録。

        Returns:
            入れ子の値まで deepcopy した新しい `_ReadOnlyMapping`。
        """
        clone = type(self)(copy.deepcopy(self._data, memo))
        memo[id(self)] = clone
        return clone

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        """pickle 往復でも読み取り専用のまま復元されるようにする。

        Returns:
            再構築のための `(呼び出し可能, 引数 tuple)`。復元結果は読み取り専用の
            `_ReadOnlyMapping` であり、素の dict へは戻らない。
        """
        return (type(self), (self._data,))


def _to_read_only_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """検証済みの Mapping を、防御的コピーの上に読み取り専用ビューを被せた形へ正規化する。

    Args:
        value: pydantic が検証した Mapping（pydantic を経由しない呼び出しでは素の Mapping）。

    Returns:
        コピーを取り込んだ読み取り専用の `_ReadOnlyMapping`。要素代入・削除は `TypeError`
        になり、`copy.deepcopy` / `pickle` を通しても読み取り専用のまま復元される。
    """
    return _ReadOnlyMapping(value)


#: 検証の最後段で読み取り専用ビューへ正規化する。`mode="after"` 相当の位置であり、
#: `Mapping[str, X]` としての型検証が済んだ値だけを受け取る。
_TO_READ_ONLY = AfterValidator(_to_read_only_mapping)

#: 直列化時は素の dict へ戻す。読み取り専用 Mapping をそのまま流すと `model_dump()` が
#: 「宣言型と違う値」の `UserWarning` を出し、`model_dump_json()` は
#: `PydanticSerializationError` で落ちる（既存の直列化契約 FR-1 L105 を壊す）。
#: `return_type` はエイリアスごとに値型を保つ（`dict` だけを渡すと
#: `model_json_schema(mode="serialization")` の値型が `additionalProperties: true` へ潰れる）。
_AS_PLAIN_STR_DICT = PlainSerializer(dict, return_type=dict[str, str])
_AS_PLAIN_ANY_DICT = PlainSerializer(dict, return_type=dict[str, Any])

#: プロンプト変数のように値が str の読み取り専用 Mapping。
ReadOnlyStrMapping = Annotated[Mapping[str, str], _TO_READ_ONLY, _AS_PLAIN_STR_DICT]

#: パラメータ値のように値の型を縛らない読み取り専用 Mapping。
ReadOnlyAnyMapping = Annotated[Mapping[str, Any], _TO_READ_ONLY, _AS_PLAIN_ANY_DICT]
