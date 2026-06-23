"""評価の設定 plain dataclass（`EvaluationConfig` / `JudgeConfig` / `LangfuseConfig`）。

すべて外部 SDK 非依存の plain dataclass（Pydantic 非導入）。`model`（Judge）や認証情報等の
値は利用者から `evaluate` 引数で受領し、env 直読をコア層へ波及させない（NFR-5）。env 参照が
必要な場合も利用者が値を解決して本設定型へ詰める想定（境界は利用側）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import CriterionStatus, Verdict


@dataclass(frozen=True)
class JudgeConfig:
    """LLM-as-Judge の採点用モデル設定。

    `model` は DeepEval の custom model（`DeepEvalBaseLLM` 実装）でラップされ、採点時の LLM
    呼び出しに使われる。判定プロンプト本文（G-Eval rubric 等）は lib に同梱せず観点文は
    `Criterion.rubric` 経由で利用者が渡す（プロンプト非同梱方針）。`evaluate(judge=...)` には
    model を直接渡せる（内部で本型にラップ）ため、本型は将来の judge パラメータ拡張余地として
    `model` のみを保持する。

    Attributes:
        model: 採点に使う LLM（SDK `Model` / モデル名文字列 等の不透明値）。
    """

    model: Any


@dataclass(frozen=True)
class EvaluationConfig:
    """評価実行の挙動設定（並列度 / タイムアウト / verdict ポリシー / テレメトリ）。

    観点集合・knockout は `Criterion` オブジェクト側に集約したため本型は実行系設定のみを持つ。
    fail-closed（judge 層・観点 status）と inconclusive ポリシー（verdict 層・統合合否）は別概念
    として 2 フィールドに分離する: `fail_closed_status` は judge がタイムアウト / スキーマ不適合時に
    当該観点へ付与する状態、`inconclusive_policy` は母集合に INCONCLUSIVE があるとき
    `compute_verdict` が解決する verdict。

    Attributes:
        timeout_seconds: 1 観点採点呼び出しのタイムアウト秒。超過時 fail-closed。None で無制限。
        concurrency: ケースの並列実行数（1 = 逐次）。`asyncio.Semaphore` で制御。
        required_criteria: verdict 母集合に存在を要求する観点集合（missing-pair fail-closed）。
            None なら評価した観点集合をそのまま要求集合とする（既定・推奨）。
        fail_closed_status: judge がタイムアウト / スキーマ不適合時に当該観点へ付与する状態
            （judge 層・既定 `CriterionStatus.FAIL`）。
        inconclusive_policy: 母集合に `INCONCLUSIVE` があるときに `compute_verdict` が解決する
            verdict（verdict 層・既定 `Verdict.FAIL`）。
        deepeval_telemetry_opt_out: DeepEval の Confident AI テレメトリを無効化するか（既定 True）。
    """

    timeout_seconds: float | None = 60.0
    concurrency: int = 1
    required_criteria: frozenset[str] | None = None
    fail_closed_status: CriterionStatus = CriterionStatus.FAIL
    inconclusive_policy: Verdict = Verdict.FAIL
    deepeval_telemetry_opt_out: bool = True


@dataclass(frozen=True)
class LangfuseConfig:
    """Langfuse 観測シンクの設定（任意・`evaluate(langfuse=...)` で渡す）。

    認証・接続先・dataset / prompt 連携設定をすべて引数で受領し env 非依存に保つ（NFR-5）。
    `dataset_name` 未設定なら Datasets 連携をスキップ・`prompt_name` 未設定なら push 専用
    Prompt Management 連携をスキップする（Scores + Traces は常時送信・best-effort）。

    Attributes:
        public_key: Langfuse public key。
        secret_key: Langfuse secret key。
        host: Langfuse 接続先 URL（self-host / cloud）。None で SDK 既定。
        dataset_name: Datasets 連携を有効化する dataset 名。None でスキップ。
        run_name: dataset run 名（A/B・回帰比較の run 識別）。None なら自動採番に委ねる。
        prompt_name: push 専用 Prompt Management 連携を有効化する prompt 名。None でスキップ。
        prompt_label: 登録 prompt に付与するラベル（任意）。None でラベルなし。
    """

    public_key: str | None = None
    secret_key: str | None = None
    host: str | None = None
    dataset_name: str | None = None
    run_name: str | None = None
    prompt_name: str | None = None
    prompt_label: str | None = None
