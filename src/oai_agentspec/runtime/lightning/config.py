"""最適化実行の設定 plain dataclass（`OptimizeConfig`）。

外部 SDK 非依存の plain dataclass（Pydantic 非導入）。並列度 / 訓練ラウンド数 / タイムアウト /
Store 等の実行制御は Agent Lightning Trainer への passthrough であり、未指定時は Trainer の既定を
適用する（NFR-7）。env 直読をコア層へ波及させない（NFR-4・env 参照が必要な場合も利用者が値を
解決して本設定型へ詰める想定）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..._validation import validate_bool

# `apo_api` の受理値（optimizer の検証と `_adapters._build_apo` のディスパッチが共有する
# 単一定義。2 層で独立にリテラルを持つと drift 時に「検証は通るが誤った API で実行される」
# silent 事故になるため）。
APO_API_RESPONSES = "responses"
APO_API_CHAT_COMPLETIONS = "chat_completions"
APO_API_VALUES = (APO_API_RESPONSES, APO_API_CHAT_COMPLETIONS)


@dataclass(frozen=True)
class OptimizeConfig:
    """最適化実行の挙動設定（並列度 / ラウンド数 / タイムアウト / Store / APO 設定の passthrough）。

    すべて Agent Lightning Trainer / APO アルゴリズムへの passthrough で、None は「未指定（既定に
    委ねる）」を意味する（NFR-7）。`store` は Agent Lightning の Store 抽象（InMemory / Sqlite /
    Mongo）の不透明値で、`_adapters/lightning` 経由で Trainer へ素通しする（本型は中身を覗かない）。
    `apo_client` は APO の textual gradient 計算 / prompt edit 適用で agent-lightning が呼ぶ
    `AsyncOpenAI` 互換クライアントで、APO 利用時は必須（未指定は CONFIG_MISSING・fail-closed）。

    Attributes:
        concurrency: rollout の並列実行数。None で Trainer 既定。
        rounds: 最適化の訓練ラウンド数（APO は `beam_rounds` にマップ）。None で Trainer 既定。
        timeout_seconds: APO の 1 batch（候補評価バッチ）の rollout 待ち合わせタイムアウト秒。
            agent-lightning APO の `rollout_batch_timeout` に passthrough する。あわせて
            pre-flight route coverage の **1 case あたりの観測上限**（lib ローカル適用・
            `asyncio.wait_for`）としても使う。**2 つの適用先で None の意味が異なる**点に注意:
            APO 側は None で APO 既定（3600 秒）が適用されるが、pre-flight 側は None で
            `wait_for` を適用しない（= 上限なし）。pre-flight の時間上限保護が必要な場合は
            本フィールドを明示設定する。
        store: Agent Lightning の Store 設定（不透明値・passthrough）。None で Trainer 既定。
        apo_client: APO の textual gradient / edit 用 `AsyncOpenAI` 互換クライアント（APO 必須・
            未指定は `OptimizeError(FailureKind.CONFIG_MISSING)`）。
        apo_gradient_model: APO の textual gradient 用モデル名。既定 `"gpt-5.4-mini"`
            （oai-agentspec の標準モデル名・agent-lightning APO 既定 `gpt-5-mini` を上書き）。
            Azure 利用時は当該デプロイ名を渡す（例: `apo_gradient_model="my-gpt5-deployment"`）。
        apo_apply_edit_model: APO の prompt edit 適用用モデル名。既定 `"gpt-5.4-mini"`
            （oai-agentspec の標準モデル名・agent-lightning APO 既定 `gpt-4.1-mini` を上書き）。
            Azure 利用時は当該デプロイ名を渡す。
        apo_api: APO の gradient / apply-edit で使う API の明示選択（None / "responses" /
            "chat_completions"）。None（既定）は auto: Responses API を優先し、`/responses`
            エンドポイント不在（404）の chat-only ゲートウェイでは chat.completions へ自動
            fallback する（安全網）。"responses" は Responses 固定で fallback しない（明示した
            のに黙って別 API へ化けない fail-closed）。"chat_completions" は最初から上流
            agent-lightning 本来の chat 実装を使う。rollout 側のモデルクラス選択
            （`OpenAIResponsesModel` / `OpenAIChatCompletionsModel`）とは独立。
        apo_beam_width: APO beam search の幅。None で APO 既定。
        apo_branch_factor: APO beam search の分岐数。None で APO 既定。
        tracer: rollout の trace を採取する Agent Lightning Tracer（不透明値・上級者向け escape
            hatch・通常は不要）。明示時は `_adapters/lightning` が構築する既定 tracer
            （`AgentOpsTracer(agentops_managed=True, instrument_managed=True)`・agent-lightning
            既定）を使わずこの値を Trainer へそのまま渡す。`OtelTracer` / `DummyTracer` は OpenAI
            計測を持たないため APO とは互換性がない（gradient 計算用 span が空になる）。AgentOps の
            クラウドアップロードを抑止したい場合は `AGENTOPS_API_KEY` を本物のキーに設定しないこと
            （SDK は `os.environ.setdefault("AGENTOPS_API_KEY", "dummy")` で初期化するため、本物の
            キーが入っていない限り送信は silent fail する）。
        skip_coverage_check: True で pre-flight route coverage 検証を skip する（既定 False で
            有効）。動的 routing 下で seed 状態のみでは判定できない構成や、単一 slot 経路で
            train × 1 rollout の pre-flight コストを回避したい場合の escape hatch。詳細は
            `docs/adr/0009-lightning-preflight-coverage.md` を参照。
    """

    concurrency: int | None = None
    rounds: int | None = None
    timeout_seconds: float | None = None
    store: Any = None
    apo_client: Any = None
    apo_gradient_model: str | None = "gpt-5.4-mini"
    apo_apply_edit_model: str | None = "gpt-5.4-mini"
    apo_api: str | None = None
    apo_beam_width: int | None = None
    apo_branch_factor: int | None = None
    tracer: Any = None
    skip_coverage_check: bool = False

    def __post_init__(self) -> None:
        """`skip_coverage_check` が bool であることを構築時に検証する。

        Raises:
            ValueError: `skip_coverage_check` が bool でない場合。
        """
        validate_bool(self.skip_coverage_check, "skip_coverage_check")
