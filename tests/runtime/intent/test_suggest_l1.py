"""L1: `runtime.intent._suggest` の候補取得と allowlist 除外（設計 §5 タスク 1-10）。

FR-4（候補型と候補生成の固定契約）と NFR-6（allowlist 除外と WARNING）を pin する。
`ContextBuilder` の選択（`CandidateSource.context_builder` / `history_limit`）、
未登録 `action_id` の除外、**非 `ExecutableIntent` 候補を同一経路・同一 WARNING で除外する
こと**（設計 §3.4c）、除外 0 件で WARNING を出さないこと、全件除外でも例外にしないこと、
`report` / `metadata` の素通し、非公開であること、SDK 非依存であることを対象とする。
外部依存（agents / openai）なし。

WARNING の形は既存 `_llm.py:148-154` に揃える（除外件数 + `repr` 化した名前一覧。LLM 由来の
非信頼テキストが制御文字を含みうるためのログフォージング対策 = CWE-117）。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

# ruff の isort は存在しないモジュールを第一者と判定できないため、_suggest.py が未実装の
# 間だけ I001 を報告する（実装後は既存ファイルと同じ並びで警告なしになる）。
from oai_agentspec.runtime.intent._suggest import _suggest_intents
from oai_agentspec.runtime.intent.binding import CandidateSource
from oai_agentspec.runtime.intent.types import (
    ConfidenceLevel,
    ConsistencyReport,
    ExecutableIntent,
    ExecutableSuggestion,
    IntentCandidate,
    IntentContext,
    IntentPrediction,
    IntentQuery,
)

pytestmark = pytest.mark.unit


_LOGGER_NAME = "oai_agentspec.runtime.intent._suggest"
_PRIVATE_SYMBOL = "_suggest_intents"
_ALLOWED = ["open_dashboard", "run_load_test"]


# ---- Fake（SDK / LLM を使わない） ----


class _FakeHistory:
    """`agents.Session` 互換の最小 Fake。`get_items(limit=...)` の引数を記録する。"""

    def __init__(self, items: list[dict[str, Any]] | None = None) -> None:
        self._items = items if items is not None else []
        self.limit_calls: list[int | None] = []

    async def get_items(self, limit: int | None = None) -> list[dict[str, Any]] | None:
        """記録して固定アイテムを返す。"""
        self.limit_calls.append(limit)
        return self._items


class _RecordingGenerator:
    """`CandidateGenerator` Protocol の Fake。渡された context を記録する。"""

    def __init__(self, prediction: IntentPrediction) -> None:
        self._prediction = prediction
        self.contexts: list[IntentContext[Any]] = []

    async def generate(self, context: IntentContext[Any]) -> IntentPrediction:
        """記録して固定の予測を返す。"""
        self.contexts.append(context)
        return self._prediction


class _MarkerContextBuilder:
    """`ContextBuilder` Protocol の Fake。差し替えが効いたことを識別できる context を返す。"""

    def __init__(self, utterance: str = "marker-utterance") -> None:
        self._utterance = utterance
        self.queries: list[IntentQuery[Any]] = []

    async def build(self, query: IntentQuery[Any]) -> IntentContext[Any]:
        """記録してマーカー付きの context を返す。"""
        self.queries.append(query)
        return IntentContext(
            utterance=self._utterance,
            history_items=({"role": "user", "content": "marker"},),
            run_context=query.run_context,
        )


class _BoomGenerator:
    """`generate()` が必ず例外を送出する Fake。"""

    async def generate(self, context: IntentContext[Any]) -> IntentPrediction:
        """常に送出する。"""
        raise ZeroDivisionError("generator exploded")


class _BoomContextBuilder:
    """`build()` が必ず例外を送出する Fake。"""

    async def build(self, query: IntentQuery[Any]) -> IntentContext[Any]:
        """常に送出する。"""
        raise ZeroDivisionError("builder exploded")


class _SubclassedIntent(ExecutableIntent):
    """`ExecutableIntent` の派生。除外判定が `isinstance` であることを pin するために使う。"""


# ---- ヘルパ ----


def _intent(
    action_id: str,
    *,
    level: ConfidenceLevel = ConfidenceLevel.HIGH,
    parameters: Mapping[str, Any] | None = None,
    source: str = "rule",
) -> ExecutableIntent:
    """テスト用の `ExecutableIntent` を組み立てる。"""
    return ExecutableIntent(
        action_id=action_id,
        level=level,
        parameters=dict(parameters or {}),
        source=source,
    )


def _prediction(
    *candidates: IntentCandidate,
    report: ConsistencyReport | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> IntentPrediction:
    """テスト用の `IntentPrediction` を組み立てる。"""
    return IntentPrediction(candidates=tuple(candidates), report=report, metadata=metadata)


def _source(prediction: IntentPrediction, **kwargs: Any) -> CandidateSource:
    """記録用 generator を載せた `CandidateSource` を返す。"""
    return CandidateSource(generator=_RecordingGenerator(prediction), **kwargs)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """`_suggest` の logger が出した WARNING レコードだけを取り出す。"""
    return [r for r in caplog.records if r.levelno == logging.WARNING and r.name == _LOGGER_NAME]


# ---- ContextBuilder の選択（FR-4 / 設計 §3.13 段 (1)） ----


async def test_default_context_builder_used_when_context_builder_is_none() -> None:
    """`context_builder=None` なら `DefaultContextBuilder` が使われ history を素通しする。"""
    items = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    history = _FakeHistory(items)
    source = _source(_prediction(_intent("run_load_test")))
    query = IntentQuery(utterance="発話", history=history, run_context={"tenant": "t1"})

    result = await _suggest_intents(query, source, _ALLOWED)

    assert result.context.utterance == "発話"
    assert result.context.history_items == tuple(items)
    assert result.context.run_context == {"tenant": "t1"}


async def test_default_context_builder_uses_history_limit_20_when_unset() -> None:
    """`history_limit=None` のとき既定 builder の 20 が使われる（FR-4 の既定値）。"""
    history = _FakeHistory()
    source = _source(_prediction())
    await _suggest_intents(IntentQuery(utterance="hi", history=history), source, _ALLOWED)

    assert history.limit_calls == [20]


async def test_default_context_builder_receives_declared_history_limit() -> None:
    """`history_limit=5` は既定 builder へ渡る（20 の既定にすり替わらない）。"""
    history = _FakeHistory()
    source = _source(_prediction(), history_limit=5)
    await _suggest_intents(IntentQuery(utterance="hi", history=history), source, _ALLOWED)

    assert history.limit_calls == [5]


async def test_explicit_context_builder_is_used_instead_of_default() -> None:
    """`context_builder` を宣言したらそれが使われ、既定 builder は動かない。"""
    builder = _MarkerContextBuilder()
    history = _FakeHistory()
    source = _source(_prediction(), context_builder=builder)
    query = IntentQuery(utterance="original", history=history)

    result = await _suggest_intents(query, source, _ALLOWED)

    assert result.context.utterance == "marker-utterance"
    assert history.limit_calls == []
    assert builder.queries == [query]


async def test_generator_receives_the_built_context_exactly_once() -> None:
    """`generate()` は builder が組んだ context で 1 回だけ呼ばれる（FR-4 / NFR-5）。"""
    builder = _MarkerContextBuilder()
    source = _source(_prediction(_intent("run_load_test")), context_builder=builder)
    generator: _RecordingGenerator = source.generator

    result = await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert len(generator.contexts) == 1
    assert generator.contexts[0].utterance == "marker-utterance"
    assert result.context == generator.contexts[0]


async def test_result_type_is_executable_suggestion() -> None:
    """戻り値は `ExecutableSuggestion` そのもの（別型・派生へすり替えない）。"""
    source = _source(_prediction(_intent("run_load_test")))
    result = await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert type(result) is ExecutableSuggestion


# ---- allowlist 除外（NFR-6 / FR-4 / 設計 §3.4c） ----


async def test_registered_candidates_pass_through_in_generator_order() -> None:
    """登録済み候補は generator の順序のまま残る（level で並べ替えない）。"""
    source = _source(
        _prediction(
            _intent("run_load_test", level=ConfidenceLevel.LOW),
            _intent("open_dashboard", level=ConfidenceLevel.CERTAIN),
            _intent("run_load_test", level=ConfidenceLevel.MEDIUM),
        )
    )

    result = await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert [c.action_id for c in result.candidates] == [
        "run_load_test",
        "open_dashboard",
        "run_load_test",
    ]
    assert [c.level for c in result.candidates] == [
        ConfidenceLevel.LOW,
        ConfidenceLevel.CERTAIN,
        ConfidenceLevel.MEDIUM,
    ]


async def test_candidate_fields_are_not_rewritten() -> None:
    """残った候補の `parameters` / `source` / `level` は加工されない。"""
    source = _source(_prediction(_intent("run_load_test", parameters={"seconds": 30}, source="ml")))

    result = await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert result.candidates[0].parameters == {"seconds": 30}
    assert result.candidates[0].source == "ml"
    assert result.candidates[0].text == "run_load_test"


async def test_unregistered_action_id_is_excluded(caplog: pytest.LogCaptureFixture) -> None:
    """未登録 `action_id` の候補は除外される（NFR-6）。"""
    source = _source(_prediction(_intent("run_load_test"), _intent("delete_everything")))
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert [c.action_id for c in result.candidates] == ["run_load_test"]


async def test_non_executable_intent_candidate_is_excluded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """親型 `IntentCandidate` がそのまま来ても除外される（設計 §3.4c）。"""
    plain = IntentCandidate(text="run_load_test", level=ConfidenceLevel.HIGH)
    source = _source(_prediction(plain, _intent("open_dashboard")))
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    # text が allowlist に載っていても ExecutableIntent でなければ通さない。
    assert [c.action_id for c in result.candidates] == ["open_dashboard"]


async def test_executable_intent_subclass_is_kept() -> None:
    """除外判定は `isinstance` であり `ExecutableIntent` の派生は残る。"""
    sub = _SubclassedIntent(action_id="run_load_test", level=ConfidenceLevel.HIGH, source="rule")
    source = _source(_prediction(sub))

    result = await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert [c.action_id for c in result.candidates] == ["run_load_test"]


async def test_both_exclusion_kinds_share_one_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """未登録 `action_id` と非 `ExecutableIntent` は同一の WARNING 1 行に含まれる（§3.4c）。"""
    plain = IntentCandidate(text="plain_candidate", level=ConfidenceLevel.MEDIUM)
    source = _source(
        _prediction(
            _intent("run_load_test"),
            _intent("delete_everything"),
            plain,
            _intent("drop_tables"),
        )
    )
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert [c.action_id for c in result.candidates] == ["run_load_test"]
    records = _warnings(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert "3" in message
    assert "delete_everything" in message
    assert "plain_candidate" in message
    assert "drop_tables" in message


async def test_warning_is_a_single_line(caplog: pytest.LogCaptureFixture) -> None:
    """WARNING は 1 レコード・1 行（除外件数ぶん行が増えない）。"""
    source = _source(_prediction(_intent("a"), _intent("b"), _intent("c"), _intent("d")))
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    records = _warnings(caplog)
    assert len(records) == 1
    assert "\n" not in records[0].getMessage()


async def test_warning_escapes_control_characters(caplog: pytest.LogCaptureFixture) -> None:
    """除外名は `repr` 化して載せる（CWE-117 ログフォージング対策・`_llm.py` と同形）。"""
    forged = "evil\nWARNING  intent classifier accepted everything"
    source = _source(_prediction(_intent(forged), IntentCandidate(text=forged, level="low")))
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    records = _warnings(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert "\n" not in message
    assert "\\n" in message
    assert "evil" in message


async def test_warning_comes_from_module_logger(caplog: pytest.LogCaptureFixture) -> None:
    """WARNING の logger 名は `_suggest` モジュール（`logging.getLogger(__name__)`）。"""
    source = _source(_prediction(_intent("delete_everything")))
    with caplog.at_level(logging.WARNING):
        await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    names = {r.name for r in caplog.records if r.levelno == logging.WARNING}
    assert names == {_LOGGER_NAME}


async def test_no_warning_when_nothing_is_excluded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """除外 0 件なら WARNING を 1 件も出さない（警告の常態化を避ける）。"""
    source = _source(_prediction(_intent("run_load_test"), _intent("open_dashboard")))
    with caplog.at_level(logging.WARNING):
        result = await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert len(result.candidates) == 2
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


async def test_no_warning_when_generator_returns_no_candidates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """候補 0 件は除外 0 件であり WARNING を出さない。"""
    source = _source(_prediction())
    with caplog.at_level(logging.WARNING):
        result = await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert result.candidates == ()
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


async def test_all_candidates_excluded_is_not_an_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """全件除外でも例外にせず空 tuple を返す（ただし WARNING は出る）。"""
    source = _source(_prediction(_intent("delete_everything"), _intent("drop_tables")))
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert result.candidates == ()
    assert len(_warnings(caplog)) == 1


async def test_empty_allowlist_excludes_every_candidate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """宣言簿が空なら全候補が除外される（allowlist は「未登録なら通さない」）。"""
    source = _source(_prediction(_intent("run_load_test")))
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = await _suggest_intents(IntentQuery(utterance="hi"), source, [])

    assert result.candidates == ()
    assert len(_warnings(caplog)) == 1


# ---- report / metadata の素通し（FR-4） ----


async def test_report_is_passed_through() -> None:
    """`IntentPrediction.report` は加工せず `ExecutableSuggestion` へ載る。"""
    report = ConsistencyReport(conflicts=("c1",), stale_context=("s1",), over_inference=())
    source = _source(_prediction(_intent("run_load_test"), report=report))

    result = await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert result.report == report


async def test_metadata_is_passed_through_without_injection() -> None:
    """`metadata` は素通しで、lib 側が予約キーを差し込まない。"""
    metadata = {"generator": "rule-v3", "elapsed_ms": 12}
    source = _source(_prediction(_intent("run_load_test"), metadata=metadata))

    result = await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert result.metadata == metadata
    assert result.metadata is not None
    assert set(result.metadata) == {"generator", "elapsed_ms"}


async def test_report_and_metadata_stay_none_when_absent() -> None:
    """generator が返さなかった `report` / `metadata` を捏造しない。"""
    source = _source(_prediction(_intent("run_load_test")))

    result = await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert result.report is None
    assert result.metadata is None


async def test_report_and_metadata_survive_full_exclusion() -> None:
    """全候補が除外されても `report` / `metadata` は捨てない（情報を失わせない）。"""
    report = ConsistencyReport(conflicts=("c1",))
    source = _source(_prediction(_intent("delete_everything"), report=report, metadata={"k": 1}))

    result = await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert result.candidates == ()
    assert result.report == report
    assert result.metadata == {"k": 1}


# ---- 例外の扱い（FR-4「握り潰さず伝播」） ----


async def test_generator_exception_propagates() -> None:
    """`generator` の例外は握り潰さず呼び出し元へ伝播する（FR-4）。"""
    source = CandidateSource(generator=_BoomGenerator())
    with pytest.raises(ZeroDivisionError, match="generator exploded") as excinfo:
        await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert type(excinfo.value) is ZeroDivisionError


async def test_context_builder_exception_propagates() -> None:
    """`context_builder` の例外も同様に伝播する（設計未明示のため固定）。"""
    source = CandidateSource(
        generator=_RecordingGenerator(_prediction()), context_builder=_BoomContextBuilder()
    )
    with pytest.raises(ZeroDivisionError, match="builder exploded") as excinfo:
        await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert type(excinfo.value) is ZeroDivisionError


async def test_generator_is_not_called_when_builder_fails() -> None:
    """builder が落ちた時点で止まり `generate()` は呼ばれない。"""
    generator = _RecordingGenerator(_prediction())
    source = CandidateSource(generator=generator, context_builder=_BoomContextBuilder())
    with pytest.raises(ZeroDivisionError):
        await _suggest_intents(IntentQuery(utterance="hi"), source, _ALLOWED)

    assert generator.contexts == []


# ---- 非公開性と SDK 隔離（設計 §3.13 / NFR-1 / NFR-6） ----


def test_suggest_symbol_is_not_publicly_exported() -> None:
    """段 (1) の関数は公開シンボルではない（`plan()` へ畳む契約・設計 §3.13）。"""
    import oai_agentspec.runtime.intent as intent_mod

    assert _PRIVATE_SYMBOL not in intent_mod.__all__
    assert not any(name.startswith("_") for name in intent_mod.__all__)
    with pytest.raises(AttributeError):
        getattr(intent_mod, _PRIVATE_SYMBOL)


def test_suggest_module_is_private() -> None:
    """モジュール名は `_` 始まり（`_suggest.py`）。"""
    from oai_agentspec.runtime.intent import _suggest as suggest_mod

    assert suggest_mod.__name__ == _LOGGER_NAME
    assert Path(suggest_mod.__file__ or "").name == "_suggest.py"


def test_suggest_module_does_not_import_sdk_or_re() -> None:
    """`agents` / `openai` を import せず `import re` も持たない（NFR-1 / NFR-6）。"""
    from oai_agentspec.runtime.intent import _suggest as suggest_mod

    text = Path(suggest_mod.__file__ or "").read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]
    assert not any(line.startswith(("import agents", "from agents")) for line in lines)
    assert not any(line.startswith(("import openai", "from openai")) for line in lines)
    assert not any(line.startswith("import re") or line.startswith("from re ") for line in lines)
