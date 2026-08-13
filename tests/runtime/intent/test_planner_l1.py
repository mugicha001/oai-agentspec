"""L1: `ActionPlanner` の公開面と宣言簿スナップショットの中身（設計 §5 タスク 1-12）。

`test_catalog_l1.py`（`bind()` の構造と `validate()`）と `test_catalog_plan_l2.py`
（`plan()` の毎ターン契約）が pin していない 2 系統だけを対象にする。重複して同じ契約を
2 ファイルへ書くと、片方だけが直される drift が起きるため範囲を分ける。

1. **公開面の形**: 公開メソッドがちょうど `validate` / `plan` の 2 つであること（§3.4a の表）と、
   両者の任意引数（`context` / `predict` / `detail`）がキーワード専用であること。`bind` の
   引数がキーワード専用であるのと同じ理由で、呼び出し側から「どれがどの目的か」が読める形を
   契約とする。
2. **スナップショットがカタログ既定を運ぶこと**: `bind()` が持つ宣言簿のスナップショットは
   `ActionSpec` の集合だけでなく `ActionCatalog` 側の既定（`prompt` / `prompt_vars` /
   `on_invalid_slot`）も含む。ここが落ちると既定マージ解決（§3.4a の純関数 3 件）の入力が
   欠け、`plan.resolved_*` からカタログ由来分が黙って消え、起動時検証もカタログ由来の
   セグメントを見なくなる。**タスク 1-11 で生存した変異 `C4_spec_prompt_only` と同じ死角**で
   あり、`ActionPlanner` 経由でも塞ぐ。

外部依存（agents / openai）なし。`AgentRegistry` は duck-typed な Fake、`PromptStore` は
tmp_path 上の実物を使う。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from oai_agentspec.prompts import PromptLayout, PromptStore
from oai_agentspec.runtime.intent.actions import ActionCatalog, ActionSpec, param
from oai_agentspec.runtime.intent.binding import CandidateSource
from oai_agentspec.runtime.intent.types import (
    ConfidenceLevel,
    ExecutableIntent,
    IntentContext,
    IntentPrediction,
    IntentQuery,
)

pytestmark = pytest.mark.unit


_LAYOUT = PromptLayout(base="base", parts="parts", agents="agents")
#: `ActionPlanner` の公開面から除く pydantic 由来の名前。`validate` は `BaseModel` の
#: 非推奨 classmethod と同名だが本設計の公開メソッドであるため除外集合から外す
#: （実測 30 の追試で、実名 `validate` でも警告なしに成立することを確認済み）。
_BASE_MODEL_NAMES = set(dir(BaseModel)) - {"validate"}


# ---- Fake ----


class _FakeAgentRegistry:
    """`AgentRegistry` の最小 Fake。`names()` / `get()` の双方を提供する。"""

    def __init__(self, *names: str) -> None:
        self._names = sorted(names)

    def names(self) -> list[str]:
        """登録済みエージェント名を昇順で返す。"""
        return list(self._names)

    def get(self, name: str) -> object:
        """未登録名なら `KeyError`（本物と同じ契約）。"""
        if name not in self._names:
            raise KeyError(name)
        return object()


class _FixedGenerator:
    """`CandidateGenerator` の Fake。固定の予測を返す。"""

    def __init__(self, prediction: IntentPrediction) -> None:
        self._prediction = prediction

    async def generate(self, context: IntentContext[Any]) -> IntentPrediction:
        """固定の予測を返す。"""
        return self._prediction


class _Ctx:
    """`validate(context=...)` へ渡す代表インスタンス。"""

    region = "jp"


# ---- ヘルパ ----


def _spec(
    action_id: str = "run_load_test",
    *,
    action_agent: str = "load_test_agent",
    parameters: tuple[Any, ...] | None = None,
) -> ActionSpec:
    """他の検査に触れない健全な `ActionSpec` を組む。"""
    return ActionSpec(
        action_id=action_id,
        description="負荷試験を実行する",
        action_agent=action_agent,
        label="負荷試験",
        parameters=parameters if parameters is not None else (param("seconds", int, default=30),),
    )


def _catalog(*specs: ActionSpec, **kwargs: Any) -> ActionCatalog:
    """宣言簿を組んで `specs` を登録した `ActionCatalog` を返す。"""
    catalog = ActionCatalog(**kwargs)
    for spec in specs:
        catalog.register(spec)
    return catalog


def _store(tmp_path: Path, **bodies: str) -> PromptStore:
    """`parts/<name>.md` を書き出した実物の `PromptStore` を返す。"""
    parts = tmp_path / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for name, body in bodies.items():
        (parts / f"{name}.md").write_text(body, encoding="utf-8")
    return PromptStore(tmp_path, _LAYOUT)


def _bind(catalog: ActionCatalog, **kwargs: Any) -> Any:
    """`registry` の既定を補って `catalog.bind()` を呼ぶ。"""
    kwargs.setdefault("registry", _FakeAgentRegistry("load_test_agent"))
    return catalog.bind(**kwargs)


def _query() -> IntentQuery[Any]:
    """最小の入力クエリ。"""
    return IntentQuery(utterance="負荷試験を回して")


def _source(*action_ids: str) -> CandidateSource:
    """指定 `action_id` の候補を返す候補源を組む。"""
    candidates = tuple(
        ExecutableIntent(action_id=action_id, level=ConfidenceLevel.HIGH, source="rule")
        for action_id in action_ids
    )
    return CandidateSource(generator=_FixedGenerator(IntentPrediction(candidates=candidates)))


def _public_names(planner: Any) -> set[str]:
    """`ActionPlanner` 自身が足した公開名を返す。

    `dir()` の結果を先に名前で絞ってから `callable()` を評価する。`model_fields` などへ
    インスタンス経由で `getattr` すると pydantic 2.11 の非推奨警告が出るため、pydantic 由来の
    名前には触れない。
    """
    names = {name for name in dir(planner) if not name.startswith("_")} - _BASE_MODEL_NAMES
    return {name for name in names if callable(getattr(planner, name))}


# ---- 公開面の形（§3.4a の表・§4a-3） ----


def test_planner_public_methods_are_exactly_validate_and_plan() -> None:
    """公開メソッドはちょうど 2 つ（宣言簿の `register` / `names` / `get` を生やさない）。

    `callable(planner.validate)` だけを見る形では、`ActionPlanner` へ第 3 のメソッドを足す
    変異が素通りする。`ActionCatalog` と `ActionPlanner` の責務分離は公開面の件数そのものが
    契約であるため、集合として固定する。
    """
    planner = _bind(_catalog(_spec()))

    assert _public_names(planner) == {"validate", "plan"}


def test_validate_context_is_keyword_only() -> None:
    """`validate(context=...)` はキーワード専用。"""
    planner = _bind(_catalog(_spec()))

    with pytest.raises(TypeError):
        planner.validate(_Ctx())


async def test_plan_predict_and_detail_are_keyword_only() -> None:
    """`plan(query, *, predict, detail)` の 2 つのフラグはキーワード専用。

    位置引数で渡せると `plan(query, False)` がどちらのフラグを指すか読めなくなり、
    `predict` / `detail` の入れ替わりが型では検出できない（どちらも `bool`）。
    """
    planner = _bind(_catalog(_spec()), candidates=_source("run_load_test"))

    with pytest.raises(TypeError):
        await planner.plan(_query(), False)


# ---- スナップショットがカタログ既定を運ぶ（§3.4a の既定マージ解決） ----


async def test_snapshot_carries_catalog_wide_defaults_into_resolved_fields(
    tmp_path: Path,
) -> None:
    """`plan()` の `resolved_*` にカタログ既定が現れる（3 件すべて）。

    `bind()` のスナップショットが `ActionSpec` の集合だけを写して `ActionCatalog` 側の既定を
    落とすと、`resolve_prompt` / `resolve_prompt_vars` / `resolve_on_invalid_slot` の入力が
    欠ける。`spec` 側が同じ宣言を持たない構成で確かめるため、カタログ由来分が消えたことが
    そのまま観測できる。
    """
    prompts = _store(tmp_path, hint="テナント ${tenant} を対象にする")
    catalog = _catalog(
        _spec(parameters=(param("note", str, by_llm=True),)),
        prompt=("part:hint",),
        prompt_vars={"tenant": "tenant.id"},
        on_invalid_slot="error",
    )
    planner = _bind(catalog, prompts=prompts, candidates=_source("run_load_test"))

    plans = await planner.plan(_query())

    assert plans[0].resolved_prompt == ("part:hint",)
    assert dict(plans[0].resolved_prompt_vars) == {"tenant": "tenant.id"}
    assert plans[0].resolved_on_invalid_slot == "error"


def test_snapshot_carries_catalog_wide_segments_into_validate() -> None:
    """カタログ側だけにセグメント宣言がある場合も `prompts` 未結線を検出する（規則 2）。

    スナップショットがカタログ既定を落とすと「セグメント宣言 0 件」に見え、`prompts=None`
    でも `validate()` が通ってしまう（起動時検証の空振り）。タスク 1-11 で生存した変異
    `C4_spec_prompt_only` と同じ死角を `ActionPlanner` 経由で塞ぐ。
    """
    catalog = _catalog(
        _spec(parameters=(param("note", str, by_llm=True),)),
        prompt=("part:hint",),
    )
    planner = _bind(catalog)

    with pytest.raises(RuntimeError) as excinfo:
        planner.validate()

    assert type(excinfo.value) is RuntimeError


def test_snapshot_carries_catalog_wide_prompt_vars_into_validate(tmp_path: Path) -> None:
    """カタログ側の `prompt_vars` がプレースホルダの供給元として数えられる（検査 4）。

    スナップショットが `prompt_vars` を落とすと、供給されているはずの `${tenant}` が
    未供給と判定され、正常な宣言が `ValueError` で落ちる。「落ちないこと」を固定する。
    """
    prompts = _store(tmp_path, hint="テナント ${tenant} を対象にする")
    catalog = _catalog(
        _spec(parameters=(param("note", str, by_llm=True),)),
        prompt=("part:hint",),
        prompt_vars={"tenant": "tenant.id"},
    )

    assert _bind(catalog, prompts=prompts).validate() is None


# ---- plan() は宣言簿を変異させない ----


async def test_plan_does_not_mutate_the_catalog(tmp_path: Path) -> None:
    """`plan()` の前後で宣言簿の状態は変わらない（`bind()` と同じ契約）。

    既定マージ解決の結果を宣言簿へ書き戻す実装（`spec` を as-declared に保つ契約の破り方）
    を検出する。カタログ既定を 3 件とも宣言した構成で確かめる。
    """
    prompts = _store(tmp_path, hint="テナント ${tenant} を対象にする")
    spec = _spec(parameters=(param("note", str, by_llm=True),))
    catalog = _catalog(
        spec,
        prompt=("part:hint",),
        prompt_vars={"tenant": "tenant.id"},
        on_invalid_slot="error",
    )
    before = (catalog.names(), catalog.prompt, dict(catalog.prompt_vars), catalog.on_invalid_slot)
    planner = _bind(catalog, prompts=prompts, candidates=_source("run_load_test"))

    await planner.plan(_query())

    assert catalog.names() == before[0]
    assert catalog.prompt == before[1]
    assert dict(catalog.prompt_vars) == before[2]
    assert catalog.on_invalid_slot == before[3]
    assert catalog.get("run_load_test") is spec
