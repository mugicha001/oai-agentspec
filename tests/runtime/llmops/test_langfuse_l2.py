"""L2: `_adapters` の Langfuse 連携（register/fetch/send）の分岐網羅（クライアントをモック）。

`langfuse.Langfuse` を fake クライアントに patch し、register_dataset_items（ensure + item upsert）/
fetch_dataset_items（get_dataset → plain dict）/ langfuse_send（Tracing + Scores は trace のみ紐づけ
+ dataset_name 設定時は既存 item へ link 専用 + prompt_name 設定時は dedup 付き register（内容不変
なら既存再利用・変更時のみ新 version）を generation 種別 trace にリンク）を検証する。get_prompt は
dedup/link 目的でのみ使い配信には使わないこと・送信失敗を warning で吸収し評価を落とさないことも
確認する（実通信なし）。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from oai_agentspec._adapters import (
    fetch_dataset_items,
    langfuse_send,
    register_dataset_items,
)
from oai_agentspec.runtime.llmops import (
    CaseResult,
    CriterionResult,
    CriterionStatus,
    EvalCase,
    EvaluationResult,
    LangfuseConfig,
    ObservedApproval,
    ObservedRoute,
    ObservedRun,
    ObservedToolCall,
    RouteStep,
    Verdict,
)

pytestmark = pytest.mark.integration

# 観点名（CriterionResult.criterion の値）。langfuse_send を直接叩く本テストでは名前文字列を使う。
RELEVANCE = "relevance"
SAFETY = "safety"
TOOL_CORRECTNESS = "tool_correctness"


class _FakeSpan:
    """trace_id を持つ fake span（start_as_current_observation の戻り）。"""

    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id


class _FakeRunItem:
    """dataset_run_items.create の戻り（dataset_run_id を持つ）。"""

    def __init__(self, dataset_run_id: str) -> None:
        self.dataset_run_id = dataset_run_id


class _FakeDatasetRunItems:
    """client.api.dataset_run_items.create を記録する fake。"""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeRunItem:
        self.created.append(kwargs)
        return _FakeRunItem(dataset_run_id=f"run-{len(self.created)}")


class _FakeApi:
    """client.api.dataset_run_items を提供する fake。"""

    def __init__(self) -> None:
        self.dataset_run_items = _FakeDatasetRunItems()


class _FakeDatasetItem:
    """get_dataset().items の要素（DatasetItem 相当・id/input/expected_output/metadata）。"""

    def __init__(
        self,
        *,
        id: str,  # noqa: A002 - DatasetItem の属性名に追従
        input: Any = None,  # noqa: A002 - DatasetItem の属性名に追従
        expected_output: Any = None,
        metadata: Any = None,
    ) -> None:
        self.id = id
        self.input = input
        self.expected_output = expected_output
        self.metadata = metadata


class _FakeDataset:
    """get_dataset の戻り（items を持つ DatasetClient 相当）。"""

    def __init__(self, items: list[_FakeDatasetItem]) -> None:
        self.items = items


class _FakePrompt:
    """get_prompt / create_prompt が返す prompt client 相当（本文 `prompt` で dedup 比較）。"""

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt


class _FakeLangfuseClient:
    """Langfuse クライアントの fake（送信系を記録・dataset/prompt の取得系も提供）。

    `get_prompt` は dedup/link 目的の取得として提供する（prompt の配信＝実行 source 使用は
    しないのが真の不変条件）。`get_prompt_existing` に既存 prompt を仕込むと dedup 一致を再現でき、
    `get_prompt_error=True` で読み取り失敗を再現できる。
    """

    def __init__(self) -> None:
        self.scores: list[dict[str, Any]] = []
        self.datasets_created: list[str] = []
        self.dataset_items: list[dict[str, Any]] = []
        self.prompts_created: list[dict[str, Any]] = []
        self.spans: list[dict[str, Any]] = []
        self.flushed = False
        self.api = _FakeApi()
        self._trace_seq = 0
        # create_prompt が返す sentinel（新 version の prompt client 相当）。
        self.registered_prompt = object()
        # get_dataset が返す items（fetch_dataset_items テストで差し替える）。
        self.dataset_to_fetch: list[_FakeDatasetItem] = []
        self.get_dataset_calls: list[str] = []
        # get_prompt の dedup シナリオ制御。
        self.get_prompt_calls: list[dict[str, Any]] = []
        self.get_prompt_existing: _FakePrompt | None = None
        self.get_prompt_error = False

    def create_score(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)

    def create_dataset(self, *, name: str) -> None:
        self.datasets_created.append(name)

    def create_dataset_item(self, **kwargs: Any) -> None:
        self.dataset_items.append(kwargs)

    def get_dataset(self, name: str) -> _FakeDataset:
        self.get_dataset_calls.append(name)
        return _FakeDataset(self.dataset_to_fetch)

    def get_prompt(self, name: str, **kwargs: Any) -> Any:
        self.get_prompt_calls.append({"name": name, **kwargs})
        if self.get_prompt_error:
            raise RuntimeError("get_prompt down")
        if self.get_prompt_existing is None:
            raise RuntimeError("NotFound: prompt does not exist")
        return self.get_prompt_existing

    def create_prompt(self, **kwargs: Any) -> Any:
        self.prompts_created.append(kwargs)
        return self.registered_prompt

    @contextmanager
    def start_as_current_observation(self, **kwargs: Any) -> Any:
        self._trace_seq += 1
        self.spans.append(kwargs)
        yield _FakeSpan(trace_id=f"trace-{self._trace_seq}")

    def flush(self) -> None:
        self.flushed = True


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: Any) -> None:
    """`langfuse.Langfuse` を渡した fake client を返すコンストラクタへ patch する。"""
    import langfuse

    def _ctor(**kwargs: Any) -> Any:
        return client

    monkeypatch.setattr(langfuse, "Langfuse", _ctor, raising=True)


def _result(cases: list[CaseResult], verdict: Verdict = Verdict.PASS) -> EvaluationResult:
    """plain EvaluationResult を組む。"""
    return EvaluationResult(target_id="bot", cases=cases, verdict=verdict)


def _case(*criteria: CriterionResult) -> CaseResult:
    """plain CaseResult を組む。"""
    return CaseResult(case_input="in", output="out", criteria=list(criteria))


# ----------------------------------------------------------------------
# Tracing / Scores（常時送信）
# ----------------------------------------------------------------------


def test_sends_trace_and_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """各ケースを trace + Scores として送信する（dataset / prompt 未設定）。"""
    client = _FakeLangfuseClient()
    _patch_client(monkeypatch, client)

    case = _case(
        CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale="ok", score=0.9)
    )
    langfuse_send(
        _result([case]),
        LangfuseConfig(),
        cases=[EvalCase(input="in")],
    )
    assert len(client.spans) == 1
    # 観点別 score(relevance) + 統合 verdict score の 2 件。
    assert len(client.scores) == 2
    assert client.scores[0]["name"] == RELEVANCE
    assert client.scores[0]["value"] == pytest.approx(0.9)
    assert client.scores[0]["trace_id"] == "trace-1"
    # score は trace のみに紐づく（dataset_run_id を渡さない・両 id 指定は 400 になる）。
    assert "dataset_run_id" not in client.scores[0]
    # 統合 verdict も NUMERIC score（pass=1.0）として trace に送る（run 比較で集約可能）。
    verdict_scores = [s for s in client.scores if s["name"] == "verdict"]
    assert len(verdict_scores) == 1
    assert verdict_scores[0]["value"] == pytest.approx(1.0)
    assert verdict_scores[0]["trace_id"] == "trace-1"
    assert "dataset_run_id" not in verdict_scores[0]
    assert client.flushed is True
    # observation 非在のケースは metadata に verdict のみ（route/tools は省略）。
    assert client.spans[0]["metadata"] == {"verdict": Verdict.PASS.value}


def test_trace_metadata_includes_observed_route_and_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """observation があれば trace metadata に起点込み route とツールを載せる（観点不問）。"""
    client = _FakeLangfuseClient()
    _patch_client(monkeypatch, client)

    observation = ObservedRun(
        route=ObservedRoute(
            steps=[
                RouteStep(agent="triage", handoff_from=None),
                RouteStep(agent="billing", handoff_from="triage"),
            ],
            last_agent="billing",
        ),
        tool_calls=[ObservedToolCall(tool="get_invoice")],
    )
    case = CaseResult(
        case_input="in",
        output="out",
        criteria=[CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale="")],
        observation=observation,
    )
    langfuse_send(_result([case]), LangfuseConfig(), cases=[EvalCase(input="in")])
    metadata = client.spans[0]["metadata"]
    assert metadata["route"] == ["triage", "billing"]
    assert metadata["tools_called"] == ["get_invoice"]
    assert metadata["verdict"] == Verdict.PASS.value
    # 承認を通らない実行では pending_approvals 空 / interrupted False。
    assert metadata["pending_approvals"] == []
    assert metadata["interrupted"] is False


def test_trace_metadata_includes_pending_approvals_and_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """observation に承認情報があれば metadata に pending_approvals / interrupted を載せる。"""
    client = _FakeLangfuseClient()
    _patch_client(monkeypatch, client)

    observation = ObservedRun(
        route=ObservedRoute(steps=[RouteStep(agent="bot")], last_agent="bot"),
        tool_calls=[],
        pending_approvals=[ObservedApproval(tool="danger", call_id="c1")],
        interrupted=True,
    )
    case = CaseResult(
        case_input="in",
        output="",
        criteria=[
            CriterionResult(criterion=RELEVANCE, status=CriterionStatus.INCONCLUSIVE, rationale="")
        ],
        observation=observation,
    )
    langfuse_send(_result([case]), LangfuseConfig(), cases=[EvalCase(input="in")])
    metadata = client.spans[0]["metadata"]
    assert metadata["pending_approvals"] == ["danger"]
    assert metadata["interrupted"] is True


def test_score_value_falls_back_to_status_when_no_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """score 非在の観点は pass=1.0 / fail=0.0 に写す。"""
    client = _FakeLangfuseClient()
    _patch_client(monkeypatch, client)

    case = _case(
        CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""),
        CriterionResult(criterion=SAFETY, status=CriterionStatus.FAIL, rationale=""),
    )
    langfuse_send(_result([case]), LangfuseConfig(), cases=[EvalCase(input="in")])
    by_name = {s["name"]: s["value"] for s in client.scores}
    assert by_name[RELEVANCE] == pytest.approx(1.0)
    assert by_name[SAFETY] == pytest.approx(0.0)


def test_overall_verdict_sent_as_score_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """統合 verdict=FAIL も NUMERIC score（verdict=0.0・comment に verdict 値）として送る。"""
    client = _FakeLangfuseClient()
    _patch_client(monkeypatch, client)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.FAIL, rationale=""))
    langfuse_send(
        _result([case], verdict=Verdict.FAIL),
        LangfuseConfig(),
        cases=[EvalCase(input="in")],
    )
    verdict_scores = [s for s in client.scores if s["name"] == "verdict"]
    assert len(verdict_scores) == 1
    assert verdict_scores[0]["value"] == pytest.approx(0.0)
    assert verdict_scores[0]["comment"] == Verdict.FAIL.value
    assert verdict_scores[0]["data_type"] == "NUMERIC"


def test_skip_and_not_applicable_criteria_not_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """skip / not_applicable 観点は Scores 送信から除外される。"""
    client = _FakeLangfuseClient()
    _patch_client(monkeypatch, client)

    case = _case(
        CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""),
        CriterionResult(criterion=SAFETY, status=CriterionStatus.SKIP, rationale=""),
        CriterionResult(
            criterion=TOOL_CORRECTNESS, status=CriterionStatus.NOT_APPLICABLE, rationale=""
        ),
    )
    langfuse_send(_result([case]), LangfuseConfig(), cases=[EvalCase(input="in")])
    names = [s["name"] for s in client.scores]
    # skip / not_applicable 観点は score を送らない（verdict score は別途付く）。
    assert RELEVANCE in names
    assert SAFETY not in names
    assert TOOL_CORRECTNESS not in names


# ----------------------------------------------------------------------
# Datasets（register → fetch → use）
# ----------------------------------------------------------------------


def test_register_dataset_items_ensures_and_upserts(monkeypatch: pytest.MonkeyPatch) -> None:
    """register_dataset_items は dataset を ensure し plain dict item を upsert する。"""
    client = _FakeLangfuseClient()
    _patch_client(monkeypatch, client)

    register_dataset_items(
        LangfuseConfig(),
        "ds",
        [
            {
                "id": "c1",
                "input": "q",
                "expected_output": "a",
                "metadata": {"reference_context": ["r"]},
            },
            {"id": "c2", "input": "q2", "expected_output": None, "metadata": None},
        ],
    )
    assert client.datasets_created == ["ds"]
    assert [it["id"] for it in client.dataset_items] == ["c1", "c2"]
    assert client.dataset_items[0]["dataset_name"] == "ds"
    assert client.dataset_items[0]["input"] == "q"
    assert client.dataset_items[0]["expected_output"] == "a"
    assert client.dataset_items[0]["metadata"] == {"reference_context": ["r"]}
    assert client.flushed is True


def test_register_dataset_items_conflict_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """既存 dataset の create 例外（conflict 相当）を握りつぶし item upsert を継続する（冪等）。"""
    client = _FakeLangfuseClient()

    def _conflict(*, name: str) -> None:
        raise RuntimeError("dataset already exists: conflict")

    client.create_dataset = _conflict  # type: ignore[method-assign]
    _patch_client(monkeypatch, client)

    register_dataset_items(LangfuseConfig(), "ds", [{"id": "c1", "input": "q"}])
    # create_dataset が例外でも item upsert は継続する。
    assert client.dataset_items[0]["id"] == "c1"


def test_register_dataset_items_flush_failure_absorbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """register の flush 失敗は warning で吸収する（item upsert は完了済み・best-effort）。"""
    client = _FakeLangfuseClient()

    def _boom() -> None:
        raise RuntimeError("flush down")

    client.flush = _boom  # type: ignore[method-assign]
    _patch_client(monkeypatch, client)

    # 例外を送出せず完了する（item は upsert 済み）。
    register_dataset_items(LangfuseConfig(), "ds", [{"id": "c1", "input": "q"}])
    assert client.dataset_items[0]["id"] == "c1"


def test_fetch_dataset_items_returns_plain_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_dataset_items は get_dataset の items を plain dict 列へ変換する。"""
    client = _FakeLangfuseClient()
    client.dataset_to_fetch = [
        _FakeDatasetItem(
            id="c1", input="q", expected_output="a", metadata={"reference_context": ["r"]}
        ),
        _FakeDatasetItem(id="c2", input="q2"),
    ]
    _patch_client(monkeypatch, client)

    items = fetch_dataset_items(LangfuseConfig(), "ds")
    assert client.get_dataset_calls == ["ds"]
    assert items == [
        {
            "id": "c1",
            "input": "q",
            "expected_output": "a",
            "metadata": {"reference_context": ["r"]},
        },
        {"id": "c2", "input": "q2", "expected_output": None, "metadata": None},
    ]
    # 戻りは plain dict（langfuse 型を外に出さない）。
    assert all(isinstance(it, dict) for it in items)


def test_register_dataset_helper_converts_cases_to_items(monkeypatch: pytest.MonkeyPatch) -> None:
    """公開 register_dataset は EvalCase を plain dict に変換し register_dataset_items へ渡す。"""
    from oai_agentspec.runtime.llmops import register_dataset

    captured: dict[str, Any] = {}

    def _fake_register(config: Any, name: str, items: list[dict[str, Any]]) -> None:
        captured["name"] = name
        captured["items"] = items

    monkeypatch.setattr(
        "oai_agentspec._adapters.register_dataset_items", _fake_register, raising=True
    )

    register_dataset(
        LangfuseConfig(),
        "ds",
        [EvalCase("q", id="c1", reference_context=["r"], expected_output="o")],
    )
    assert captured["name"] == "ds"
    item = captured["items"][0]
    assert item["id"] == "c1"
    assert item["input"] == "q"
    assert item["expected_output"] == "o"
    assert item["metadata"] == {"reference_context": ["r"]}


def test_load_dataset_helper_restores_eval_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """公開 load_dataset は fetch_dataset_items の plain dict を EvalCase へ復元する。"""
    from oai_agentspec.runtime.llmops import load_dataset

    def _fake_fetch(config: Any, name: str) -> list[dict[str, Any]]:
        return [
            {
                "id": "c1",
                "input": "q",
                "expected_output": "o",
                "metadata": {"expected_tools": ["t"]},
            }
        ]

    monkeypatch.setattr("oai_agentspec._adapters.fetch_dataset_items", _fake_fetch, raising=True)

    cases = load_dataset(LangfuseConfig(), "ds")
    assert cases == [EvalCase("q", id="c1", expected_tools=["t"], expected_output="o")]


def test_dataset_link_only_does_not_upsert_items(monkeypatch: pytest.MonkeyPatch) -> None:
    """evaluate（langfuse_send）は dataset_name 設定でも item upsert / dataset 作成をしない。"""
    client = _FakeLangfuseClient()
    _patch_client(monkeypatch, client)

    case = _case(
        CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale="", score=1.0)
    )
    langfuse_send(
        _result([case]),
        LangfuseConfig(dataset_name="ds", run_name="run-A"),
        cases=[EvalCase(input="in", id="case-1", expected_output="正解文")],
    )
    # register が担うので evaluate 経路では dataset / item を作らない。
    assert client.datasets_created == []
    assert client.dataset_items == []
    # 既存 item（dataset_item_id=EvalCase.id）へ run を link するだけ。
    created = client.api.dataset_run_items.created[0]
    assert created["run_name"] == "run-A"
    assert created["dataset_item_id"] == "case-1"
    assert created["trace_id"] == "trace-1"
    # score は trace のみに紐づく（run id は渡さない・両 id 指定は 400）。
    assert client.scores[0]["trace_id"] == "trace-1"
    assert "dataset_run_id" not in client.scores[0]


def test_dataset_link_uses_stable_id_when_case_id_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """EvalCase.id 未指定時は stable_id 導出した item id へ link する。"""
    client = _FakeLangfuseClient()
    _patch_client(monkeypatch, client)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    langfuse_send(
        _result([case]),
        LangfuseConfig(dataset_name="ds"),
        cases=[EvalCase(input="in")],  # id 未指定
    )
    linked_id = client.api.dataset_run_items.created[0]["dataset_item_id"]
    assert linked_id.startswith("case-0-")


def test_dataset_not_configured_skips_link(monkeypatch: pytest.MonkeyPatch) -> None:
    """dataset_name 未設定なら dataset 系 API を呼ばず Scores は trace のみに紐づく。"""
    client = _FakeLangfuseClient()
    _patch_client(monkeypatch, client)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    langfuse_send(_result([case]), LangfuseConfig(), cases=[EvalCase(input="in")])
    assert client.datasets_created == []
    assert client.dataset_items == []
    assert client.api.dataset_run_items.created == []
    assert "dataset_run_id" not in client.scores[0]


def test_dataset_run_link_failure_keeps_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """dataset run リンク失敗時も Scores（trace のみ紐づけ）を送る（best-effort）。"""
    client = _FakeLangfuseClient()

    def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("run link down")

    client.api.dataset_run_items.create = _boom  # type: ignore[method-assign]
    _patch_client(monkeypatch, client)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    langfuse_send(
        _result([case]),
        LangfuseConfig(dataset_name="ds"),
        cases=[EvalCase(input="in")],
    )
    assert len(client.scores) == 2  # 観点 score + 統合 verdict score
    assert "dataset_run_id" not in client.scores[0]


# ----------------------------------------------------------------------
# Prompt Management（dedup 付き register/link・opt-in）
# ----------------------------------------------------------------------


def test_prompt_register_creates_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """初回（get_prompt が NotFound）は create_prompt で新 version を作る（push）。"""
    client = _FakeLangfuseClient()  # 既定: get_prompt は NotFound 相当を送出
    _patch_client(monkeypatch, client)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    langfuse_send(
        _result([case]),
        LangfuseConfig(prompt_name="agent-prompt", prompt_label="prod"),
        cases=[EvalCase(input="in")],
        prompt_text="static instructions body",
    )
    # dedup のため get_prompt を読み（NotFound）→ create_prompt で新規作成。
    assert len(client.get_prompt_calls) == 1
    assert len(client.prompts_created) == 1
    created = client.prompts_created[0]
    assert created["name"] == "agent-prompt"
    assert created["prompt"] == "static instructions body"
    assert created["labels"] == ["prod"]
    assert created["type"] == "text"


def test_prompt_dedup_reuses_existing_on_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """既存 version の本文が一致すれば create_prompt せず既存を再利用し trace に link する。"""
    client = _FakeLangfuseClient()
    existing = _FakePrompt(prompt="static instructions body")
    client.get_prompt_existing = existing
    _patch_client(monkeypatch, client)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    langfuse_send(
        _result([case]),
        LangfuseConfig(prompt_name="agent-prompt", prompt_label="prod"),
        cases=[EvalCase(input="in")],
        prompt_text="static instructions body",
    )
    # 本文一致 → 新 version を作らない（version が増えない）。
    assert client.prompts_created == []
    # dedup 比較は最新を読む（cache_ttl_seconds=0）+ ラベル指定。
    assert client.get_prompt_calls[0]["cache_ttl_seconds"] == 0
    assert client.get_prompt_calls[0]["label"] == "prod"
    # trace は既存 prompt に link される。
    assert client.spans[0]["prompt"] is existing


def test_prompt_dedup_creates_new_on_content_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """既存 version の本文が異なれば create_prompt で新 version を作る。"""
    client = _FakeLangfuseClient()
    client.get_prompt_existing = _FakePrompt(prompt="old body")
    _patch_client(monkeypatch, client)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    langfuse_send(
        _result([case]),
        LangfuseConfig(prompt_name="agent-prompt"),
        cases=[EvalCase(input="in")],
        prompt_text="new body",
    )
    assert len(client.prompts_created) == 1
    assert client.prompts_created[0]["prompt"] == "new body"
    # 新規 prompt（create_prompt の戻り）が trace に link される。
    assert client.spans[0]["prompt"] is client.registered_prompt


def test_prompt_dedup_get_read_failure_falls_back_to_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_prompt 読み取り失敗は best-effort で create にフォールバックし評価を継続する。"""
    client = _FakeLangfuseClient()
    client.get_prompt_error = True
    _patch_client(monkeypatch, client)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    langfuse_send(
        _result([case]),
        LangfuseConfig(prompt_name="agent-prompt"),
        cases=[EvalCase(input="in")],
        prompt_text="body",
    )
    # 読み取り失敗でも create にフォールバックし trace に link、Scores も送る。
    assert len(client.prompts_created) == 1
    assert client.spans[0]["prompt"] is client.registered_prompt
    assert len(client.scores) == 2  # 観点 score + 統合 verdict score


def test_prompt_linked_to_trace_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    """登録した prompt を各 trace の observation へ prompt= で渡す（prompt version 紐づけ）。"""
    client = _FakeLangfuseClient()
    _patch_client(monkeypatch, client)

    cases = [
        _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale="")),
        _case(CriterionResult(criterion=SAFETY, status=CriterionStatus.PASS, rationale="")),
    ]
    langfuse_send(
        _result(cases),
        LangfuseConfig(prompt_name="agent-prompt"),
        cases=[EvalCase(input="a"), EvalCase(input="b")],
        prompt_text="static instructions body",
    )
    # create_prompt を 1 回だけ呼び、その戻り（registered_prompt）を各 trace に prompt= でリンク。
    assert len(client.prompts_created) == 1
    assert len(client.spans) == 2
    for span in client.spans:
        assert span["prompt"] is client.registered_prompt
        # prompt= は generation / embedding 種別でのみ有効。evaluator 種別に戻ると prompt
        # リンクが無効化する退行を防ぐため as_type=generation を assert する。
        assert span["as_type"] == "generation"


def test_prompt_not_linked_to_trace_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """prompt_name 未設定なら start_as_current_observation に prompt= を渡さない。"""
    client = _FakeLangfuseClient()
    _patch_client(monkeypatch, client)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    langfuse_send(_result([case]), LangfuseConfig(), cases=[EvalCase(input="a")])
    assert client.prompts_created == []
    assert "prompt" not in client.spans[0]


def test_prompt_register_failure_does_not_link(monkeypatch: pytest.MonkeyPatch) -> None:
    """prompt 登録が失敗したら trace へのリンクもしない（prompt= を渡さない）・評価は継続。"""
    client = _FakeLangfuseClient()

    def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("prompt api down")

    client.create_prompt = _boom  # type: ignore[method-assign]
    _patch_client(monkeypatch, client)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    langfuse_send(
        _result([case]),
        LangfuseConfig(prompt_name="p"),
        cases=[EvalCase(input="a")],
        prompt_text="body",
    )
    assert "prompt" not in client.spans[0]
    assert len(client.scores) == 2  # 観点 score + 統合 verdict score


def test_prompt_register_without_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """prompt_label 未指定なら labels は空リスト。"""
    client = _FakeLangfuseClient()
    _patch_client(monkeypatch, client)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    langfuse_send(
        _result([case]),
        LangfuseConfig(prompt_name="p"),
        cases=[EvalCase(input="in")],
        prompt_text="body",
    )
    assert client.prompts_created[0]["labels"] == []


def test_prompt_skipped_when_name_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """prompt_name 未設定なら prompt_text があっても登録しない。"""
    client = _FakeLangfuseClient()
    _patch_client(monkeypatch, client)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    langfuse_send(
        _result([case]),
        LangfuseConfig(),
        cases=[EvalCase(input="in")],
        prompt_text="body",
    )
    assert client.prompts_created == []


def test_prompt_skipped_when_text_unextractable(monkeypatch: pytest.MonkeyPatch) -> None:
    """prompt_text=None（抽出不可・動的/横断）なら prompt_name 設定でも登録しない。"""
    client = _FakeLangfuseClient()
    _patch_client(monkeypatch, client)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    langfuse_send(
        _result([case]),
        LangfuseConfig(prompt_name="p"),
        cases=[EvalCase(input="in")],
        prompt_text=None,
    )
    assert client.prompts_created == []


def test_prompt_register_failure_absorbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """prompt 登録失敗は warning で吸収し評価（Scores 送信）を続ける。"""
    client = _FakeLangfuseClient()

    def _boom(**kwargs: Any) -> None:
        raise RuntimeError("prompt api down")

    client.create_prompt = _boom  # type: ignore[method-assign]
    _patch_client(monkeypatch, client)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    langfuse_send(
        _result([case]),
        LangfuseConfig(prompt_name="p"),
        cases=[EvalCase(input="in")],
        prompt_text="body",
    )
    assert len(client.scores) == 2  # 観点 score + 統合 verdict score


def test_prompt_get_used_only_for_dedup_not_delivery(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_prompt は dedup/link 目的でのみ使う（読み取りはするが配信＝実行 source にしない）。

    真の不変条件は「Langfuse をプロンプトの配信元（実行 source）にしない」。`langfuse_send` は
    エージェントを実行せず、get_prompt の戻りは dedup 比較と trace への link にのみ使う（評価対象の
    実行プロンプトには使わない）。本テストは dedup のための読み取りが行われることを確認する。
    """
    client = _FakeLangfuseClient()
    client.get_prompt_existing = _FakePrompt(prompt="body")
    _patch_client(monkeypatch, client)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    langfuse_send(
        _result([case]),
        LangfuseConfig(prompt_name="p", dataset_name="ds"),
        cases=[EvalCase(input="in")],
        prompt_text="body",
    )
    # dedup のため get_prompt を読む（本文一致で再利用・新 version は作らない）。
    assert len(client.get_prompt_calls) == 1
    assert client.prompts_created == []


# ----------------------------------------------------------------------
# best-effort: trace/score 送信失敗・flush 失敗の吸収
# ----------------------------------------------------------------------


def test_trace_send_failure_absorbed_per_case(monkeypatch: pytest.MonkeyPatch) -> None:
    """1 ケースの trace 送信失敗は warning で吸収し他ケース処理 + flush を継続する。"""
    client = _FakeLangfuseClient()

    calls = {"n": 0}
    real_cm = client.start_as_current_observation

    @contextmanager
    def _flaky(**kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("span down")
        with real_cm(**kwargs) as span:
            yield span

    client.start_as_current_observation = _flaky  # type: ignore[method-assign]
    _patch_client(monkeypatch, client)

    case_a = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    case_b = _case(CriterionResult(criterion=SAFETY, status=CriterionStatus.PASS, rationale=""))
    langfuse_send(
        _result([case_a, case_b]),
        LangfuseConfig(),
        cases=[EvalCase(input="a"), EvalCase(input="b")],
    )
    # 1 件目は失敗・2 件目は成功して Scores が送られる。評価は落ちない。
    names = [s["name"] for s in client.scores]
    assert SAFETY in names  # 2 件目（成功）の観点 score
    assert RELEVANCE not in names  # 1 件目（失敗）の score は送られない
    assert "verdict" in names  # 成功ケースの統合 verdict score
    assert client.flushed is True


def test_flush_failure_absorbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """flush 失敗も warning で吸収し例外を送出しない（評価を落とさない）。"""
    client = _FakeLangfuseClient()

    def _boom() -> None:
        raise RuntimeError("flush down")

    client.flush = _boom  # type: ignore[method-assign]
    _patch_client(monkeypatch, client)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    # 例外が外へ漏れないこと（送出されれば test 失敗）。
    langfuse_send(_result([case]), LangfuseConfig(), cases=[EvalCase(input="in")])
    assert len(client.scores) == 2  # 観点 score + 統合 verdict score


def test_make_client_passes_auth_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """LangfuseConfig の認証・接続先が Langfuse コンストラクタへ渡る（env 非依存）。"""
    import langfuse

    captured: dict[str, Any] = {}

    def _ctor(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _FakeLangfuseClient()

    monkeypatch.setattr(langfuse, "Langfuse", _ctor, raising=True)

    case = _case(CriterionResult(criterion=RELEVANCE, status=CriterionStatus.PASS, rationale=""))
    langfuse_send(
        _result([case]),
        LangfuseConfig(public_key="pk", secret_key="sk", host="https://lf.local"),
        cases=[EvalCase(input="in")],
    )
    assert captured["public_key"] == "pk"
    assert captured["secret_key"] == "sk"
    assert captured["host"] == "https://lf.local"
