"""Langfuse 連携窓口（観測クライアント結合を `_adapters` に閉じる・NFR-1）。

`import langfuse` を本モジュールの関数内遅延 import に閉じる。plain 評価結果
（観点別スコア + verdict + 入出力）を Langfuse Tracing / Scores へ送信し（常時）、opt-in で
Datasets と Prompt Management（評価対象プロンプトの dedup 付き register/upsert + 評価 trace を当該
prompt version にリンク）を行う。

Datasets は **register → fetch → use** モデル（Langfuse が source）:
`register_dataset_items` で一度きり item を upsert（dataset は冪等 ensure）し、`fetch_dataset_items`
で plain dict 列として取得する。evaluate（`langfuse_send`）は **既存 item へ run を link するだけ**
で毎回 upsert しない。register/fetch は EvalCase 等の runtime/llmops 型を import せず **plain dict**
のみ扱う（単方向依存維持）。

観点スコアは **trace に紐づける**（Langfuse の score は trace / dataset_run / observation / session
の **いずれか 1 つ** にのみ紐づき、trace_id + dataset_run_id の同時指定は 400 になる）。run の比較
UI はリンクされた trace のスコアを集約するため、score 側に run id は渡さない。

真の不変条件は **「Langfuse をプロンプトの配信元（実行 source）にしない」**（PromptStore が SoT）。
取得系 API（`get_dataset` / `get_prompt`）は dataset の fetch、および prompt の dedup（内容不変なら
新 version を作らない）/ link 目的でのみ使い、**取得したプロンプトをエージェント実行に使う（配信）
ことはしない**。

全送信は best-effort（`except Exception: logger.warning(..., exc_info=True)`）でローカル評価結果を
fail させない（NFR-3）。langfuse 未導入で `LangfuseConfig` が渡されたときのみ明示 ImportError +
案内（`_LLMOPS_LANGFUSE_INSTALL_HINT`）。env 直読はしない（設定は引数で受領・NFR-5）。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..runtime.llmops.config import LangfuseConfig
    from ..runtime.llmops.dataset import EvalCase
    from ..runtime.llmops.types import CriterionStatus, EvaluationResult

logger = logging.getLogger(__name__)

# llmops-langfuse extra（langfuse）未導入時の案内。
_LLMOPS_LANGFUSE_INSTALL_HINT = (
    "LLMOps の Langfuse 連携には langfuse が必要です。"
    "次でインストールしてください: pip install 'oai-agentspec[llmops-langfuse]'"
)


def _require_langfuse() -> Any:
    """langfuse を遅延 import する（未導入時は案内付き ImportError）。

    Returns:
        langfuse モジュール。

    Raises:
        ImportError: langfuse が未導入の場合（案内文字列付き）。
    """
    try:
        import langfuse  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise ImportError(_LLMOPS_LANGFUSE_INSTALL_HINT) from exc
    return langfuse


def _make_client(config: LangfuseConfig) -> Any:
    """`LangfuseConfig` から Langfuse クライアントを構築する（env 非依存）。

    Args:
        config: Langfuse 設定（認証・接続先）。

    Returns:
        Langfuse クライアントインスタンス。
    """
    from langfuse import Langfuse

    kwargs: dict[str, Any] = {}
    if config.public_key is not None:
        kwargs["public_key"] = config.public_key
    if config.secret_key is not None:
        kwargs["secret_key"] = config.secret_key
    if config.host is not None:
        kwargs["host"] = config.host
    return Langfuse(**kwargs)


def _score_value(status: CriterionStatus) -> float:
    """観点状態を数値スコア（pass=1.0 / それ以外=0.0）へ写す。

    Args:
        status: 観点状態。

    Returns:
        pass のみ 1.0、それ以外 0.0。
    """
    from ..runtime.llmops.types import CriterionStatus

    return 1.0 if status == CriterionStatus.PASS else 0.0


def _send_scores(client: Any, *, trace_id: str, case: Any) -> None:
    """1 ケースの観点別スコアを Langfuse Scores へ送る（trace に紐づける）。

    Langfuse の score は target（trace / dataset_run / observation / session）の **いずれか 1 つ**
    にのみ紐づく（trace_id + dataset_run_id の同時指定は 400 Bad request）。per-case の観点スコアは
    trace に付け、dataset run への対応づけは `_link_dataset_run_item`（trace を run にリンク）に
    委ねる。run の比較 UI はリンクされた trace のスコアを集約するため、score 側に run id は不要。

    Args:
        client: Langfuse クライアント。
        trace_id: 当該ケースの trace id（score の唯一の紐づけ先）。
        case: plain `CaseResult`。
    """
    from ..runtime.llmops.types import CriterionStatus

    for criterion in case.criteria:
        if criterion.status in (CriterionStatus.SKIP, CriterionStatus.NOT_APPLICABLE):
            continue
        client.create_score(
            name=criterion.criterion,
            value=(
                criterion.score if criterion.score is not None else _score_value(criterion.status)
            ),
            trace_id=trace_id,
            comment=criterion.rationale or None,
            data_type="NUMERIC",
        )


def _send_verdict_score(client: Any, *, trace_id: str, verdict: Any) -> None:
    """統合 verdict を NUMERIC score（`verdict` = pass:1.0 / fail:0.0）として trace に送る。

    観点別 Score に加え統合 verdict を score 化することで、Langfuse の Dataset Run 比較・フィルタが
    pass/fail ゲートを集約・参照できる（trace metadata だけだと run 比較の Score 集計に出ない）。
    各ケースの trace に同じ overall verdict を付ける（run 全体で同値に集約される）。score 名は固定
    `verdict`。

    Args:
        client: Langfuse クライアント。
        trace_id: 当該ケースの trace id（score の紐づけ先）。
        verdict: 統合 verdict（`Verdict`）。
    """
    from ..runtime.llmops.types import Verdict

    client.create_score(
        name="verdict",
        value=1.0 if verdict == Verdict.PASS else 0.0,
        trace_id=trace_id,
        comment=verdict.value,
        data_type="NUMERIC",
    )


def _ensure_dataset(client: Any, dataset_name: str) -> None:
    """dataset を冪等に ensure する（既存なら no-op・後続の item upsert を止めない）。

    `create_dataset` は create 専用で同名 dataset 既存時に conflict 例外を投げうる。これを握りつぶし
    正常扱いにすることで、既存 dataset でも後続の item upsert を継続させる。conflict 判定は例外型/
    メッセージに依存せず広めに catch し、デバッグ用に warning を残す（register 専用のセットアップ
    経路で使う・evaluate 経路からは呼ばない）。

    Args:
        client: Langfuse クライアント。
        dataset_name: dataset 名。
    """
    try:
        client.create_dataset(name=dataset_name)
    except Exception:  # noqa: BLE001 - 既存 dataset 等は冪等扱いで継続（NFR-3）
        logger.warning(
            "langfuse dataset create をスキップしました（既存の可能性・継続します）", exc_info=True
        )


def register_dataset_items(config: LangfuseConfig, name: str, items: list[dict[str, Any]]) -> None:
    """plain dict の item 列を Langfuse dataset へ register/upsert する（一度きりのセットアップ）。

    各 item dict は `{"id", "input", "expected_output", "metadata"}` を持つ（`id` 必須・他は任意）。
    dataset は冪等に ensure（既存握りつぶし）してから item を `create_dataset_item`（同一 id で
    upsert）する。EvalCase 等の runtime/llmops 型は import しない（plain dict のみ受領）。

    Args:
        config: Langfuse 設定（認証・接続先）。
        name: dataset 名。
        items: 登録する item の plain dict 列（id / input / expected_output / metadata）。
    """
    _require_langfuse()
    client = _make_client(config)
    _ensure_dataset(client, name)
    for item in items:
        client.create_dataset_item(
            dataset_name=name,
            id=item["id"],
            input=item.get("input"),
            expected_output=item.get("expected_output"),
            metadata=item.get("metadata"),
        )
    try:
        client.flush()
    except Exception:  # noqa: BLE001 - best-effort（NFR-3）
        logger.warning("langfuse flush に失敗しました", exc_info=True)


def fetch_dataset_items(config: LangfuseConfig, name: str) -> list[dict[str, Any]]:
    """Langfuse dataset を fetch し item を plain dict 列へ変換して返す（langfuse 型を外に出さず）。

    `get_dataset(name).items`（DatasetItem 列）を `{"id", "input", "expected_output", "metadata"}`
    の plain dict 列へ防御的に変換する（`getattr` で属性消失耐性）。dataset は Langfuse が source の
    ため取得系 `get_dataset` を使ってよい（push 専用制約は Prompt Management のみ）。

    Args:
        config: Langfuse 設定（認証・接続先）。
        name: dataset 名。

    Returns:
        item の plain dict 列（id / input / expected_output / metadata）。
    """
    _require_langfuse()
    client = _make_client(config)
    dataset = client.get_dataset(name)
    items: list[dict[str, Any]] = []
    for item in getattr(dataset, "items", None) or []:
        items.append(
            {
                "id": getattr(item, "id", None),
                "input": getattr(item, "input", None),
                "expected_output": getattr(item, "expected_output", None),
                "metadata": getattr(item, "metadata", None),
            }
        )
    return items


def _link_dataset_run_item(
    client: Any,
    *,
    run_name: str,
    dataset_item_id: str,
    trace_id: str,
) -> str | None:
    """評価 trace を dataset item × dataset run にリンクし dataset_run_id を返す。

    langfuse 4.x の低レベル API `client.api.dataset_run_items.create(...)` を使う。`run_name` から
    dataset run が（無ければ）作成され、当該 item × trace がその run にリンクされる。run の比較 UI
    はリンクされた trace のスコアを集約するため、観点スコア自体は trace に紐づければよい（score 側
    に run id は渡さない・`_send_scores` 参照）。戻りの `dataset_run_id` は確立した run の識別子で、
    呼び出し側ではログ用途等に留める（score の target には使わない）。リンクのため dataset / item を
    参照する API を使うのは push 専用制約の対象外（Datasets は対象外・push 専用は Prompt Management
    のみ・設計 §17 / 判断I）。

    Args:
        client: Langfuse クライアント。
        run_name: dataset run 名（A/B・回帰比較の run 識別）。
        dataset_item_id: リンク対象 dataset item の安定 id。
        trace_id: 当該ケースの評価 trace id。

    Returns:
        確立された dataset run id（ログ用途）。取得できなければ None。
    """
    run_item = client.api.dataset_run_items.create(
        run_name=run_name,
        dataset_item_id=dataset_item_id,
        trace_id=trace_id,
    )
    return getattr(run_item, "dataset_run_id", None)


def _register_prompt(client: Any, config: LangfuseConfig, prompt_text: str) -> Any:
    """評価対象プロンプトを Langfuse PM へ dedup 付きで register/upsert し、prompt を返す。

    内容が変わった時だけ新 version を作る（dedup）。まず `get_prompt` で既存 version を読み、本文が
    今回の `prompt_text` と一致すれば既存 prompt を再利用して `create_prompt` を呼ばない（評価の
    たびに version が増えるのを防ぐ）。一致しない / 既存無し（NotFound 等）なら `create_prompt` で
    新 version を作る。いずれも得た prompt オブジェクト（既存再利用 or 新規）を返し、呼び出し側が
    `start_as_current_observation(prompt=...)` で評価 trace を当該 version に link する。

    `get_prompt` は **dedup / link のためにのみ使う**読み取りで、取得したプロンプトを
    **エージェント実行に使う（配信）ことはしない**（PromptStore が SoT のまま・真の不変条件は
    「Langfuse をプロンプトの配信元＝実行 source にしない」）。dedup 比較はキャッシュで stale に
    ならないよう `cache_ttl_seconds=0` で最新を読む。

    Args:
        client: Langfuse クライアント。
        config: Langfuse 設定（prompt_name / prompt_label）。
        prompt_text: 登録するプロンプト本文（静的 instructions・plain 文字列）。

    Returns:
        prompt client（既存再利用 or 新規・trace へのリンク用）。
    """
    name = str(config.prompt_name)
    labels = [config.prompt_label] if config.prompt_label else []

    # dedup: 既存 version の本文が一致すれば再利用（create_prompt しない）。読み取り失敗 / 既存無し
    # は create にフォールバックする（best-effort・評価は落とさない）。
    get_kwargs: dict[str, Any] = {"cache_ttl_seconds": 0}
    if config.prompt_label:
        get_kwargs["label"] = config.prompt_label
    try:
        existing = client.get_prompt(name, **get_kwargs)
    except Exception:  # noqa: BLE001 - 既存無し / 読み取り失敗は create フォールバック（NFR-3）
        existing = None
    if existing is not None and getattr(existing, "prompt", None) == prompt_text:
        return existing

    return client.create_prompt(
        name=name,
        prompt=prompt_text,
        labels=labels,
        type="text",
    )


def langfuse_send(
    result: EvaluationResult,
    config: LangfuseConfig,
    *,
    cases: list[EvalCase],
    prompt_text: str | None = None,
) -> None:
    """plain 評価結果を Langfuse へ送信する（Tracing / Scores + opt-in Datasets / Prompt・NFR-3）。

    常時: 各ケースを generation 種別の observation として送信し、観点スコアと統合 verdict score
    （`verdict` = pass:1.0 / fail:0.0）を当該 trace に紐づける（score は trace のみに紐づけ・run id
    は渡さない）。verdict を score 化するのは Dataset Run 比較・フィルタが Score を集約するためで、
    trace metadata だけだと pass/fail ゲートが run 比較に出ない。trace metadata には verdict に加え
    観測した経路（`route` = 起点込みフルパス）とツール（`tools_called`）を載せる（criteria に
    HandoffRoute / ToolUse を含めなくても観測経路・ツールが Langfuse trace で確認できる・観測不能な
    ケースは省略）。observation を generation 種別にするのは、
    Langfuse SDK が `prompt=`（model 等も）を generation / embedding 種別でのみ有効化するためで、
    evaluator 種別では prompt= が無視され prompt-version リンクが成立しない。`config.dataset_name`
    設定時は **既存 dataset item（dataset_item_id = EvalCase.id・未指定なら stable_id 導出）へ
    run を link するだけ**（item upsert / dataset 作成は `register_dataset_items` が担う・evaluate
    は毎回 push しない）。各ケースの評価 trace を dataset item × dataset run（`run_name`・無指定時は
    自動採番名）へ `dataset_run_items.create` でリンクし、run の比較 UI がリンクされた trace の
    スコアを集約する（A/B・回帰比較が UI で成立）。item が存在しない等の link 失敗は best-effort
    warning で吸収し、Scores（trace 紐づけ）は継続する。`config.prompt_name` 設定 + `prompt_text`
    抽出可能時のみ Prompt Management へ dedup 付きで register/upsert し（内容不変なら新 version を
    作らず既存を再利用）、その prompt version を各ケースの trace に
    `start_as_current_observation(prompt=...)` でリンクする（version 単位で judge 結果が集約される・
    取得は dedup/link 限定で配信には使わない）。

    全工程 best-effort（例外吸収 + warning ログ）。送信失敗で評価を fail させない。
    langfuse 未導入時のみ `_require_langfuse` が ImportError を送出する。

    Args:
        result: plain `EvaluationResult`（ローカル採点済み・必ず返る前提のもの）。
        config: Langfuse 設定。
        cases: 評価ケース列（既存 dataset item への link 対応づけ用・`result.cases` と同順）。
        prompt_text: 抽出済み評価対象プロンプト本文（静的 instructions・抽出不可なら None）。
    """
    import uuid

    from ..runtime.llmops.dataset import stable_id

    _require_langfuse()
    client = _make_client(config)

    # dataset 連携は「既存 item へ run を link するだけ」（item upsert / dataset 作成は register
    # が担う）。各ケースの link 対象 item id（EvalCase.id・未指定なら stable_id）をケース順に。
    dataset_enabled = config.dataset_name is not None
    item_ids: list[str] = (
        [stable_id(case, index) for index, case in enumerate(cases)] if dataset_enabled else []
    )

    # dataset run 名（API は run_name 必須・無指定時は安定でない採番名を生成）。
    run_name = config.run_name or f"oai-agentspec-eval-{uuid.uuid4().hex[:8]}"

    # prompt_name 設定 + 抽出可能時のみ 1 回登録し、prompt client を各 trace に紐づける。
    registered_prompt: Any = None
    if config.prompt_name is not None and prompt_text is not None:
        try:
            registered_prompt = _register_prompt(client, config, prompt_text)
        except Exception:  # noqa: BLE001 - best-effort（NFR-3）
            logger.warning("langfuse prompt 登録に失敗しました", exc_info=True)

    for index, case in enumerate(result.cases):
        try:
            # observation は generation 種別（評価対象の input→output = 生成を記録）。Langfuse SDK
            # は prompt=（model 等も）を generation / embedding 種別でのみ有効化するため、
            # prompt-version への judge 結果集約には generation が必須（evaluator 種別では prompt=
            # が無視されリンクされない）。
            metadata: dict[str, Any] = {"verdict": result.verdict.value}
            # 観測経路（起点込みフルパス）とツールを metadata に載せる（観点の有無に関わらず
            # Langfuse trace で確認できる）。捕捉不能（observation=None）なら省略する。
            observation = case.observation
            if observation is not None:
                metadata["route"] = [step.agent for step in observation.route.steps]
                metadata["tools_called"] = [tc.tool for tc in observation.tool_calls]
                # 承認ゲートの発火 / 中断（HITL）を metadata に載せる（観測のみ・観点の有無に
                # 依存しない）。承認を通らない実行では空 / False。
                metadata["pending_approvals"] = [a.tool for a in observation.pending_approvals]
                metadata["interrupted"] = observation.interrupted
            observation_kwargs: dict[str, Any] = {
                "name": f"eval-{result.target_id}",
                "as_type": "generation",
                "input": case.case_input,
                "output": case.output,
                "metadata": metadata,
            }
            # 登録できた prompt version に trace（と付随する Scores）をリンクする。
            if registered_prompt is not None:
                observation_kwargs["prompt"] = registered_prompt
            with client.start_as_current_observation(**observation_kwargs) as span:
                trace_id = span.trace_id
                # dataset 連携時のみ trace を item × run にリンクする（run 比較 UI は当該 trace の
                # スコアを集約する）。score は trace のみに紐づけるため戻りの run id は使わない。
                if dataset_enabled and index < len(item_ids):
                    try:
                        _link_dataset_run_item(
                            client,
                            run_name=run_name,
                            dataset_item_id=item_ids[index],
                            trace_id=trace_id,
                        )
                    except Exception:  # noqa: BLE001 - best-effort（NFR-3）
                        logger.warning(
                            "langfuse dataset run リンクに失敗しました（case=%s）",
                            index,
                            exc_info=True,
                        )
                _send_scores(client, trace_id=trace_id, case=case)
                # 統合 verdict も score 化（run 比較で pass/fail ゲートを集約・参照可能にする）。
                _send_verdict_score(client, trace_id=trace_id, verdict=result.verdict)
        except Exception:  # noqa: BLE001 - best-effort（NFR-3）
            logger.warning(
                "langfuse trace/score 送信に失敗しました（case=%s）", index, exc_info=True
            )

    try:
        client.flush()
    except Exception:  # noqa: BLE001 - best-effort（NFR-3）
        logger.warning("langfuse flush に失敗しました", exc_info=True)
