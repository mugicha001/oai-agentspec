"""Agent Lightning 統合窓口（最適化エンジン結合を `_adapters` に閉じる・NFR-1）。

`import agentlightning` を本モジュールの関数内遅延 import に閉じる。APO（Automatic Prompt
Optimization）の最適化ループを Agent Lightning の Trainer / APO アルゴリズム / LitAgent へ委譲する
薄い結線を提供する。`run_apo` は、利用者が供給した rollout / reward / 学習データ・検証データを
束ね、`${var}` プレースホルダを保持した最適化済みテキスト（単一 / 名前付き mapping）と train /
val スコア・履歴を plain な戻り値（`runtime/lightning/types.OptimizeResult`）として返す。
Agent Lightning の型はロジック層へ一切出さない（plain で返す）。

設計の核:
    - APO 0.3.0 は単一プロンプトの beam search 最適化。複数スロット mapping は順次最適化する
      （各スロットを順に APO に通し、最良候補で他のスロットの seed を順次更新）。
    - テンプレートエンジン: agent-lightning は `f-string` / `jinja` / `poml` を持つが、
      oai-agentspec 側は `string.Template` の `${var}` のみ扱う。
      境界で `${var}` <-> `{{ var }}` を相互変換（jinja）
      して lightning ロジック層へは `${var}` のみ流す。
    - スコア: APO は内部の `_history_best_score`（val 基準）を持つ。本モジュールは Trainer.fit
      完了後に最適候補で train / val を改めて rollout し、合成候補（複数スロット時は全スロット
      最良）の train_score / val_score を再計算して plain `OptimizeResult` に詰める。
    - Trainer.fit は同期関数で内部に asyncio ループを抱える。本関数は `asyncio.to_thread` で
      別スレッドへ退避させる（呼び出し側 async ループをブロックしない）。
    - 候補プロンプトは APO の `initial_resources={name: PromptTemplate(...)}`（engine="jinja"）で
      渡し、LitAgent サブクラス内で `resources[name].template` から取り出して oai-agentspec 側
      rollout callable へ渡す。reward は `emit_reward(float)` で span として emit する。

agentlightning 未導入時は明示 ImportError + 案内（`_LIGHTNING_INSTALL_HINT`）。`apo_client` 未供給
（APO は `AsyncOpenAI` 互換クライアント必須）/ val 不在は呼び出し側で
`OptimizeError(CONFIG_MISSING)` へ倒す（本モジュールは plain な前提違反検知のみ）。
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from ..runtime.lightning.config import OptimizeConfig
    from ..runtime.lightning.types import OptimizeResult

logger = logging.getLogger(__name__)

# `_setup_agentlightning_console_logging` の重複登録防止フラグ（プロセス内 1 回だけ handler 追加）。
_AGENTLIGHTNING_CONSOLE_HANDLER_ATTACHED = False


def _setup_agentlightning_console_logging() -> None:
    """`agentlightning` ロガーに INFO レベルの console handler を 1 度だけ取り付ける。

    `AGENTOPS_API_KEY` 未設定時は agentops の console 出力を抑止する代わりに、agent-lightning が
    出している progress ログ（rollout 完了 / round スコア / candidate 評価）が利用者の console に
    流れるようにする。`_build_trainer` から呼ばれる前提で、プロセス内 1 回のみ実行される
    （複数回 `optimize` を呼んでも handler が重複登録されない）。

    ロガーレベルが `NOTSET`（=未設定・effective レベルは親 root の WARNING に落ちる）のときだけ
    INFO へ引き下げる。利用者が `logging.getLogger("agentlightning").setLevel(logging.WARNING)` の
    ように**明示的に**設定している場合（NOTSET 以外）はそれを尊重し、本関数は level を触らない
    （progress を mute したい / DEBUG を見たい等の利用者意図を優先・Codex P3 回帰防止）。
    """
    global _AGENTLIGHTNING_CONSOLE_HANDLER_ATTACHED  # noqa: PLW0603 - プロセス単位の 1 回 setup
    if _AGENTLIGHTNING_CONSOLE_HANDLER_ATTACHED:
        return
    al_logger = logging.getLogger("agentlightning")
    if al_logger.level == logging.NOTSET:
        al_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    al_logger.addHandler(handler)
    _AGENTLIGHTNING_CONSOLE_HANDLER_ATTACHED = True


# lightning extra（agentlightning）未導入時の案内。
_LIGHTNING_INSTALL_HINT = (
    "Agent Lightning による最適化には agentlightning が必要です。"
    "次でインストールしてください: pip install 'oai-agentspec[lightning]'"
)

# `${var}` ↔ `{{ var }}` 相互変換用の安全な識別子マッチ（Python 識別子相当）。
# `${var}` 側は `runtime.lightning._placeholders.PLACEHOLDER_RE` を Single Source of Truth として
# 共有する（合成規則 / 抽出 / 置換と同じ regex を使うことで drift を不可能にする）。
_JINJA_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def _require_agentlightning() -> Any:
    """agentlightning を遅延 import する（未導入時は案内付き ImportError）。

    Returns:
        agentlightning モジュール。

    Raises:
        ImportError: agentlightning が未導入の場合（案内文字列付き）。
    """
    try:
        import agentlightning  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise ImportError(_LIGHTNING_INSTALL_HINT) from exc
    return agentlightning


def _to_jinja(text: str) -> str:
    """`${var}` を `{{ var }}` へ変換する（oai-agentspec → agent-lightning 境界）。

    `${var}` プレースホルダのみ変換する（他のリテラルは触らない）。識別子は Python 識別子相当に
    限定（`${1abc}` のような不正名は変換せず、結果として fail-closed の placeholder 喪失検査に
    引っかかる）。

    Args:
        text: oai-agentspec 側のテンプレート（`${var}` プレースホルダ）。

    Returns:
        jinja エンジン互換のテンプレート（`{{ var }}` プレースホルダ）。
    """
    from ..runtime.lightning._placeholders import PLACEHOLDER_RE

    return PLACEHOLDER_RE.sub(r"{{ \1 }}", text)


def _from_jinja(text: str) -> str:
    """`{{ var }}` を `${var}` へ変換する（agent-lightning → oai-agentspec 境界）。

    `{{ var }}`（任意の空白許容）プレースホルダのみ変換する。`{{ var | filter }}` のような jinja
    固有構文は本変換では維持される（その場合は後段の `_reinject_vars` の placeholder 喪失検査で
    fail-closed に倒れる・oai-agentspec は `${var}` のみ扱う前提）。

    Args:
        text: agent-lightning が返した最適化済みテンプレート（jinja エンジン）。

    Returns:
        oai-agentspec 側のテンプレート（`${var}` プレースホルダ）。
    """
    return _JINJA_VAR_RE.sub(r"${\1}", text)


def _make_litagent(
    *,
    target_slot: str,
    seeds: dict[str, str],
    rollout: Callable[[dict[str, str], Any], Awaitable[float]],
) -> Any:
    """oai-agentspec の rollout callable を Agent Lightning LitAgent へ橋渡しする
    サブクラスを生成する。

    LitAgent.rollout_async は `(task, resources, rollout_obj)` を受ける。本関数で動的生成するサブ
    クラスは:
        1. `resources[target_slot].template` から **jinja 候補** を取り出し `${var}` へ復元
        2. `seeds` を母体に target_slot のみ候補で上書きした `candidate_dict` を組み立てる
        3. oai-agentspec 側 `rollout(candidate_dict, task)` を呼び float の報酬を得る
        4. `emit_reward(score)` で span として emit（agent-lightning が拾う）
    rollout 例外は捕捉し reward=0.0 で emit して継続する（fail-closed・最適化全体は止めない）。

    Args:
        target_slot: 本ラウンドで最適化対象のスロット名（resources のキー）。
        seeds: 全スロットの seed（target_slot 以外はそのまま使用）。
        rollout: oai-agentspec 側 rollout callable（候補スロット mapping + ケース → 報酬）。

    Returns:
        Agent Lightning LitAgent のサブクラスインスタンス。
    """
    from agentlightning import LitAgent, emit_reward

    seeds_snapshot = dict(seeds)

    # `OptimizeError`（CONFIG_MISSING の構造化失敗・FR-8 / NFR-8）は握り潰さず外へ伝搬させる。
    # rollout 内の一般的失敗（外部 API 一時障害等）のみ reward=0.0 で継続させ最適化全体は止めない。
    from ..runtime.lightning.types import OptimizeError

    class _OaiAgentSpecLitAgent(LitAgent):  # type: ignore[misc, valid-type]
        """oai-agentspec の rollout 契約を agent-lightning の
        rollout_async / emit_reward に橋渡し。"""

        # `OptimizeError` を agent-lightning worker thread が握り潰した場合の sentinel。
        # `run_apo` が `Trainer.fit` 後にこれを check し、None でなければ re-raise する
        # （NFR-8 安全違反が fail-open に堕ちないことを保証）。
        critical_error: OptimizeError | None = None

        async def rollout_async(self, task: Any, resources: Any, rollout_obj: Any) -> None:
            """1 rollout: 候補抽出 -> oai-agentspec rollout -> reward emit。"""
            try:
                cand_template = resources[target_slot]
                cand_text = _from_jinja(cand_template.template)
                cand_dict = dict(seeds_snapshot)
                cand_dict[target_slot] = cand_text
                score = await rollout(cand_dict, task)
            except OptimizeError as exc:
                # 構造化失敗（特に NFR-8 安全違反）は Trainer.fit を通じて呼び出し側へ伝搬させたい
                # が、agent-lightning の worker thread が catch-all でログに落として続行する可能性が
                # ある。確実に escape させるため sentinel に保持し、re-raise も併用する（worker が
                # 捕捉しても run_apo 側が post-fit check で検出可能）。
                if self.critical_error is None:
                    self.critical_error = exc
                raise
            except Exception:
                logger.exception("rollout failed in LitAgent.rollout_async (reward=0.0 で継続)")
                emit_reward(0.0)
                return
            emit_reward(float(score))

    return _OaiAgentSpecLitAgent()


def _messages_to_responses_input(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    """OpenAI chat-style messages を Responses API の `(input, instructions)` 形式へ純変換する。

    Responses API は system / developer ロールの指示を `instructions=` で受け、`input=` には
    user / assistant のターンを並べる（chat-style messages list を直接受け付ける）。本関数は
    `role == "system"` を `instructions` に集約し、それ以外を `input` に詰め替える純変換で、SDK 型を
    一切公開しない（plain dict / str のみ・NFR-1）。

    Args:
        messages: poml が `format="openai_chat"` で生成する chat-style messages（dict 列）。

    Returns:
        `(input_messages, instructions)`。`instructions` は system 統合文字列（なければ None）。
        `input_messages` は user / assistant ターンの dict 列。
    """
    instructions_parts: list[str] = []
    input_messages: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in {"system", "developer"} and content:
            instructions_parts.append(str(content))
        elif role and content:
            input_messages.append({"role": role, "content": content})
    instructions = "\n\n".join(instructions_parts) if instructions_parts else None
    return input_messages, instructions


# モデル / デプロイ不在を示す 404 の error code（これらは fallback せず伝搬させる。
# chat へ倒すと「モデル名のタイポ」が「chat 側の別エラー」に化けて真因が隠れるため）。
_MODEL_NOT_FOUND_CODES = frozenset({"model_not_found", "deploymentnotfound"})


def _is_endpoint_missing_404(exc: Any) -> bool:
    """404 が「/responses エンドポイント不在」かを判定する（モデル不在なら False）。

    OpenAI は未知モデルに `error.code='model_not_found'`、Azure は `DeploymentNotFound` を
    返す。いずれもエンドポイント自体は存在するため chat への fallback は不適切
    （設定ミスの隠蔽になる）。判別できない 404（`{'detail': 'Not Found'}` 形式の
    ゲートウェイ応答等）はエンドポイント不在とみなす。

    Args:
        exc: 捕捉した `openai.NotFoundError`。

    Returns:
        エンドポイント不在（fallback してよい）なら True。
    """
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code.lower() in _MODEL_NOT_FOUND_CODES:
        return False
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        body_code = body.get("code")
        if isinstance(body_code, str) and body_code.lower() in _MODEL_NOT_FOUND_CODES:
            return False
    return True


async def _chat_complete_text(
    *, client: Any, model: str, messages: list[dict[str, Any]], temperature: float
) -> str:
    """chat.completions で chat-style messages を 1 ターン実行し最終テキストを返す。

    上流 agent-lightning APO 本来の呼び出し形（`apo.py` は chat.completions を使う）。
    Responses API を持たない chat-only ゲートウェイ（litellm 等）向けの fallback 経路。

    Args:
        client: AsyncOpenAI 互換クライアント（`chat.completions` を持つ）。
        model: 呼び出すモデル名。
        messages: chat-style messages（system 分離は不要・そのまま渡す）。
        temperature: 生成温度。

    Returns:
        モデル出力テキスト（空 / None なら空文字列）。
    """
    response = await client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 互換ゲートウェイの content-parts 形式（[{"type": "text", "text": ...}, ...]）を
        # text 連結で str へ強制する（`-> str` 契約の維持・list 混入で下流の文字列処理が
        # 原因から遠い場所で落ちるのを防ぐ）。
        return "".join(
            part.get("text", "") if isinstance(part, dict) else (getattr(part, "text", "") or "")
            for part in content
        )
    return str(content)


async def _responses_complete_text(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    allow_chat_fallback: bool = True,
    unsupported_models: set[str] | None = None,
) -> str:
    """chat-style messages を 1 ターン実行し最終テキストを返す（Responses 優先・chat fallback）。

    `chat.completions.create` の代替として使う（agent-lightning APO のサブクラス内部で利用）。
    まず `messages` を `(input, instructions)` に分解し `client.responses.create(...)` を呼ぶ
    （Azure の Responses-only deployment（gpt-5 系等）でも動作する経路）。エンドポイント不在
    （`openai.NotFoundError` = 404）の場合のみ、上流 agent-lightning 本来の
    `chat.completions.create` へ自動 fallback する（litellm 等の chat-only ゲートウェイ対応・
    「client を渡せばそのまま動く」の維持）。fallback は client 単位で記憶し、同一 client の
    2 回目以降は Responses を再試行しない。記憶は呼び出し側が渡す `unsupported_models`
    （モデル名の set）に閉じる — `_build_apo` が APO インスタンス属性として持たせるため、
    寿命は 1 APO インスタンス（= 1 slot の実行）に限定され、一過性の誤分類 404 が
    プロセス寿命まで残らない。モデル名単位なので他モデルの正常な Responses 経路を
    巻き込まない。None なら記憶なしで毎回試行する。
    モデル / デプロイ不在を示す 404（`model_not_found` / `DeploymentNotFound`）および
    404 以外の失敗（認証エラー等）は fallback せず伝搬する（誤設定の隠蔽防止）。

    Args:
        client: AsyncOpenAI 互換クライアント（`responses` または `chat.completions` を持つ）。
        model: 呼び出すモデル名（Azure ではデプロイ名）。
        messages: chat-style messages（poml 生成・Responses 経路では system を instructions に
            分離し、chat fallback 経路ではそのまま渡す）。
        temperature: 生成温度。

    Returns:
        モデル出力テキスト（空 / None なら空文字列）。
    """
    from openai import NotFoundError

    if allow_chat_fallback:
        # auto（apo_api 未指定）のみの経路。明示 "responses"（allow_chat_fallback=False）では
        # 属性ショートカットも memo ヒットも使わず必ず Responses を呼ぶ — 「明示したのに黙って
        # chat へ行く」経路をゼロにする fail-closed（先行 auto 実行の記憶にも影響されない）。
        # `responses` 属性を持たない最小構成の chat-only client / プロキシは、404 を待つ以前に
        # 属性アクセスで落ちるため最初から chat を使う（memo 不関与・getattr は毎回で十分安い）。
        if getattr(client, "responses", None) is None:
            return await _chat_complete_text(
                client=client, model=model, messages=messages, temperature=temperature
            )

        if unsupported_models is not None and model in unsupported_models:
            return await _chat_complete_text(
                client=client, model=model, messages=messages, temperature=temperature
            )

    input_messages, instructions = _messages_to_responses_input(messages)
    kwargs: dict[str, Any] = {
        "model": model,
        "input": input_messages if input_messages else "",
        "temperature": temperature,
    }
    if instructions is not None:
        kwargs["instructions"] = instructions
    try:
        response = await client.responses.create(**kwargs)
    except NotFoundError as exc:
        if not allow_chat_fallback:
            # 明示 "responses": 404 でも fallback せず伝搬（記憶にも書かない）。
            raise
        if not _is_endpoint_missing_404(exc):
            # モデル / デプロイ不在の 404。chat へ倒すと設定ミスが別のエラーに化けるため伝搬。
            raise
        # /responses エンドポイント不在（chat-only ゲートウェイ）。上流本来の chat へ倒す。
        logger.warning(
            "[apo] client が Responses API 非対応（404）のため chat.completions へ "
            "fallback します（model=%s・以降この (client, model) では chat を使用）",
            model,
        )
        text = await _chat_complete_text(
            client=client, model=model, messages=messages, temperature=temperature
        )
        # 記憶は fallback が成功してから（一過性の失敗で chat 固定化しないため）。
        if unsupported_models is not None:
            unsupported_models.add(model)
        return text
    return getattr(response, "output_text", "") or ""


async def _responses_compute_textual_gradient(
    self: Any, current_prompt: Any, rollout_results: Any, *, prefix: Any = None
) -> str:
    """親 `agentlightning.APO.compute_textual_gradient` の Responses API 版実装。

    親実装は `chat.completions.create` で critique を取得するが、本実装は同じ poml メッセージ列を
    Responses API（`responses.create`）へ送って output text を返す。最適化アルゴリズム本体（beam
    search / 履歴管理）は親の制御フローのまま、LLM 呼び出しだけ差し替える（build-don't-run）。

    Args:
        self: APO インスタンス（メソッドバインド先・`async_openai_client` / `gradient_model` /
            `gradient_batch_size` / `diversity_temperature` を参照する）。
        current_prompt: 現候補プロンプト（`prompt_template.template` を持つ）。
        rollout_results: 候補に対する rollout 結果列。
        prefix: 親実装互換の log プレフィックス（本実装では未使用）。

    Returns:
        textual gradient（critique）のテキスト。
    """
    import random as _random

    import poml  # type: ignore[import-not-found]
    from agentlightning.algorithm.apo.apo import GRADIENT_PROMPT_FILES

    tg_template = _random.choice(GRADIENT_PROMPT_FILES)
    if len(rollout_results) < self.gradient_batch_size:
        sampled = rollout_results
    else:
        sampled = _random.sample(rollout_results, self.gradient_batch_size)

    tg_msg = poml.poml(
        tg_template,
        context={
            "experiments": sampled,
            "prompt_template": current_prompt.prompt_template.template,
        },
        format="openai_chat",
    )
    return await _responses_complete_text(
        client=self.async_openai_client,
        allow_chat_fallback=getattr(self, "_oas_allow_chat_fallback", True),
        unsupported_models=getattr(self, "_oas_responses_unsupported", None),
        model=self.gradient_model,
        messages=tg_msg["messages"],
        temperature=self.diversity_temperature,
    )


async def _responses_textual_gradient_and_apply_edit(
    self: Any, current_prompt: Any, rollout: Any, *, prefix: Any = None
) -> str:
    """親 `agentlightning.APO.textual_gradient_and_apply_edit` の Responses API 版実装。

    critique → apply_edit の 2 段呼び出しを Responses API で実行する。critique が空 / None なら親
    実装と同様に元プロンプトを返す（fail-safe）。

    Args:
        self: APO インスタンス（メソッドバインド先）。
        current_prompt: 現候補プロンプト。
        rollout: 候補に対する rollout 結果列。
        prefix: 親実装互換の log プレフィックス（本実装では `compute_textual_gradient` へ渡す）。

    Returns:
        edit を適用した新プロンプトテキスト。critique 取得失敗時は元プロンプトのまま。
    """
    import random as _random

    import poml  # type: ignore[import-not-found]
    from agentlightning.algorithm.apo.apo import APPLY_EDIT_PROMPT_FILES

    critique_text = await self.compute_textual_gradient(current_prompt, rollout, prefix=prefix)
    if not critique_text:
        return current_prompt.prompt_template.template

    ae_template = _random.choice(APPLY_EDIT_PROMPT_FILES)
    ae_msg = poml.poml(
        ae_template,
        context={
            "prompt_template": current_prompt.prompt_template.template,
            "critique": critique_text,
        },
        format="openai_chat",
    )
    return await _responses_complete_text(
        client=self.async_openai_client,
        allow_chat_fallback=getattr(self, "_oas_allow_chat_fallback", True),
        unsupported_models=getattr(self, "_oas_responses_unsupported", None),
        model=self.apply_edit_model,
        messages=ae_msg["messages"],
        temperature=self.diversity_temperature,
    )


def _build_apo(config: OptimizeConfig | None) -> Any:
    """`OptimizeConfig` から Responses API 版 APO インスタンスを構築する。

    APO 本体（`agentlightning.APO`）は内部で `chat.completions.create` を 2 箇所（textual gradient /
    apply_edit）でハードコードして呼ぶ。Azure の Responses-only deployment（`gpt-5` 系等の新モデル）
    では同じデプロイ名でも `chat.completions` 経路が `DeploymentNotFound (404)` で落ちる。本関数は
    `agentlightning.APO(**kwargs)` で生成したインスタンスに対し、Responses API 版の override
    メソッド
    （`_responses_compute_textual_gradient` / `_responses_textual_gradient_and_apply_edit`）を bound
    method として動的に差し替え、agent / APO の両方を Responses API 一本に揃える（NFR-1: SDK 結合は
    `_adapters` に閉じる）。

    最適化ロジック本体（beam search / rollout 評価 / プロンプト履歴）は親 APO の実装をそのまま使う
    （build-don't-run）。インスタンスレベルでメソッドを差し替える（subclass しない）のは、テスト時の
    `monkeypatch.setattr("agentlightning.APO", _factory)` で APO が**関数ファクトリ**へ差し替えら
    れるケース（class subclass 不可）にも追従するため。

    Args:
        config: 実行制御設定（None で全項目既定）。`config.apo_client` は必須前提で呼び出し側が
            検証済み。

    Returns:
        Responses API 版 override 済みの `agentlightning.APO` インスタンス。
    """
    import types

    from agentlightning import APO

    assert config is not None and config.apo_client is not None  # 呼び出し側で検証済み

    kwargs: dict[str, Any] = {"async_openai_client": config.apo_client}
    if config.apo_gradient_model is not None:
        kwargs["gradient_model"] = config.apo_gradient_model
    if config.apo_apply_edit_model is not None:
        kwargs["apply_edit_model"] = config.apo_apply_edit_model
    if config.apo_beam_width is not None:
        kwargs["beam_width"] = config.apo_beam_width
    if config.apo_branch_factor is not None:
        kwargs["branch_factor"] = config.apo_branch_factor
    if config.rounds is not None:
        kwargs["beam_rounds"] = config.rounds
    if config.timeout_seconds is not None:
        # oai-agentspec 側 `timeout_seconds` は agentlightning APO の `rollout_batch_timeout`
        # （1 batch の rollout 待ち合わせタイムアウト・既定 3600 秒）にマップする。
        kwargs["rollout_batch_timeout"] = config.timeout_seconds

    instance = APO(**kwargs)
    from ..runtime.lightning.config import APO_API_CHAT_COMPLETIONS

    apo_api = getattr(config, "apo_api", None)
    if apo_api == APO_API_CHAT_COMPLETIONS:
        # 明示 chat: override を bind しない = 上流 agent-lightning 本来の chat.completions
        # 実装がそのまま動く（gradient / apply-edit とも）。
        return instance
    # Responses API 版へインスタンスメソッド差し替え（dataclass instance / pydantic model どちらでも
    # 動く・bound method として束ねるため `types.MethodType` を使う）。override 2 本は必ず同時に
    # bind する（片方のみだと gradient と apply-edit で API が食い違う不整合になる）。
    instance.compute_textual_gradient = types.MethodType(  # type: ignore[method-assign]
        _responses_compute_textual_gradient, instance
    )
    instance.textual_gradient_and_apply_edit = types.MethodType(  # type: ignore[method-assign]
        _responses_textual_gradient_and_apply_edit, instance
    )
    # 明示 "responses" は fallback 禁止（明示したのに黙って chat へ化けない）。None（auto）は許可。
    instance._oas_allow_chat_fallback = apo_api is None
    # fallback 済みモデルの memo（本インスタンスの寿命に閉じる。gradient / apply-edit の
    # 呼び出し間で 404 再試行を省く目的だけの最小状態）。
    instance._oas_responses_unsupported = set()
    return instance


def _build_trainer(
    *, algorithm: Any, initial_resources: dict[str, Any], config: OptimizeConfig | None
) -> Any:
    """`OptimizeConfig` から `agentlightning.Trainer` インスタンスを構築する。

    実行戦略は `SharedMemoryExecutionStrategy`（単一プロセス + 協調 worker thread）を**既定**に
    する。
    agent-lightning の既定は `ClientServerExecutionStrategy`（multiprocessing spawn）だが、本ライブ
    ラリは `LitAgent.rollout_async` を closure で構築する都合で macOS spawn-mode の pickling に
    失敗する（`Can't get local object` エラー）。SharedMemory 戦略は単一プロセスで thread 並列に
    rollout を回すため pickling 不要・I/O bound な APO rollout の並列度を Trainer.n_runners で制御
    できる（API 互換）。

    tracer は既定で `AgentOpsTracer(agentops_managed=True, instrument_managed=True)`
    （agent-lightning の既定構成）を構築する。`agentops_managed=True` で
    `agentops.init(auto_start_session=False)` を呼び AgentOps の OTel TracerProvider を初期化し、
    `instrument_managed=True` で OpenAI 計測
    （`gen_ai.*` span・APO の textual gradient 計算に必須）を仕込む。`agentops_managed=False` の
    組み合わせは AgentOps を別途初期化していないとランタイムエラー
    （`AgentOps TracerProvider is not initialized`）になるため使わない。`config.tracer` を明示した
    場合のみ既定 tracer を捨ててその値を Trainer へそのまま渡す（上級者向け escape hatch）。
    AgentOps クラウドへのアップロードは `AGENTOPS_API_KEY` を本物のキーに設定していない限り
    silent fail する（dummy キーで初期化される）。

    Args:
        algorithm: `agentlightning.APO` インスタンス。
        initial_resources: `{resource_name: PromptTemplate}` の dict（APO の seed）。
        config: 実行制御設定（None で全項目既定・concurrency=None なら main_thread="runner" の
            n_runners=1）。

    Returns:
        `agentlightning.Trainer` インスタンス。
    """
    from agentlightning import Trainer
    from agentlightning.adapter import TraceToMessages
    from agentlightning.execution import SharedMemoryExecutionStrategy
    from agentlightning.tracer import AgentOpsTracer

    # `AGENTOPS_API_KEY` 未設定時は AgentOps クラウドへ実際にアップロードできない（dummy キーで
    # silent fail）。その状態で `[OPENAI INSTRUMENTOR] Error ...` warning や `Session Replay
    # https://app.agentops.ai/...` の console 出力 + `agentops.log` ファイル生成は単なるノイズ
    # （利用者の手元 example 実行を汚す）なので、本物のキーが入っていない時だけ agentops の出力を
    # 抑制する。`setdefault` を使うため、利用者が明示的に `AGENTOPS_LOG_LEVEL=DEBUG` 等を設定して
    # いれば unmute できる。
    if not os.environ.get("AGENTOPS_API_KEY"):
        os.environ.setdefault("AGENTOPS_LOG_LEVEL", "ERROR")
        os.environ.setdefault("AGENTOPS_LOGGING_TO_FILE", "False")
        # agentops の console / file 出力を切る代わりに、agent-lightning の進捗ログ（round /
        # rollout / candidate 評価）を console に流して利用者から見えるようにする。
        _setup_agentlightning_console_logging()

    n_runners = config.concurrency if config is not None and config.concurrency is not None else 1
    # main_thread は常に "algorithm" を使う。"runner" だと algorithm 完了後も runner thread が
    # stop_evt を受け取らず無限に rollout 待機して終了しない（agent-lightning 0.3 の挙動）。
    # "algorithm" なら algorithm 完了時に stop_evt を立てて runner を協調停止できる
    # （n_runners 制約なし）。
    strategy = SharedMemoryExecutionStrategy(n_runners=n_runners, main_thread="algorithm")

    # tracer 既定: agent-lightning の既定構成（agentops_managed=True / instrument_managed=True）。
    # agentops_managed=False は AgentOps の TracerProvider を別途初期化していない限り
    # `AgentOps TracerProvider is not initialized` で落ちるため使わない。`config.tracer` 明示時のみ
    # 既定を捨ててその値をそのまま使う（上級者向け escape hatch）。
    if config is not None and config.tracer is not None:
        tracer = config.tracer
    else:
        tracer = AgentOpsTracer(agentops_managed=True, instrument_managed=True)

    # APO は `TraceToMessages` adapter を要求する（既定の `TracerTraceToTriplet` だと
    # `Adapter must be a TraceToMessages for APO algorithm` で実行時失敗する）。Trainer に明示。
    kwargs: dict[str, Any] = {
        "algorithm": algorithm,
        "initial_resources": initial_resources,
        "strategy": strategy,
        "adapter": TraceToMessages(),
        "tracer": tracer,
    }
    if config is not None and config.store is not None:
        kwargs["store"] = config.store
    return Trainer(**kwargs)


async def _run_apo_single_slot(
    *,
    slot_name: str,
    other_candidates: dict[str, str],
    train: Sequence[Any],
    val: Sequence[Any],
    rollout: Callable[[dict[str, str], Any], Awaitable[float]],
    config: OptimizeConfig | None,
) -> tuple[str, dict[str, Any]]:
    """1 スロット分の APO を実行し `(最良テキスト, 履歴 1 エントリ)` を返す（plain）。

    Trainer.fit は sync のため `asyncio.to_thread` で別スレッドへ退避させる（呼び出し側 async
    ループをブロックしない）。

    Args:
        slot_name: 本ラウンドで最適化対象のスロット名。
        other_candidates: 全スロットの候補テキスト（`${var}` 保持）。target スロット以外はこのまま
            rollout で使い、本ラウンド完了後に target スロットの最良候補で上書きされる。
        train: 学習用ケース列。
        val: 検証用ケース列。
        rollout: oai-agentspec 側 rollout callable。
        config: 実行制御設定。

    Returns:
        `(最良テキスト・${var} 保持, history エントリ dict)`。
    """
    from agentlightning import PromptTemplate

    seed_text = other_candidates[slot_name]
    jinja_seed = _to_jinja(seed_text)

    litagent = _make_litagent(target_slot=slot_name, seeds=other_candidates, rollout=rollout)
    apo = _build_apo(config)
    trainer = _build_trainer(
        algorithm=apo,
        initial_resources={slot_name: PromptTemplate(template=jinja_seed, engine="jinja")},
        config=config,
    )

    # Trainer.fit は sync 関数で内部に asyncio ループを抱える。別スレッドへ退避してから呼ぶ。
    await asyncio.to_thread(trainer.fit, litagent, list(train), val_dataset=list(val))

    # NFR-8 fail-closed: agent-lightning worker thread が `OptimizeError` を握り潰しても、
    # LitAgent.critical_error sentinel に保持されているため Trainer.fit 完了後に再 raise する。
    critical = getattr(litagent, "critical_error", None)
    if critical is not None:
        raise critical

    best_template = apo.get_best_prompt()
    best_text = _from_jinja(best_template.template)

    # APO の最良候補が seed の `${var}` プレースホルダを喪失していた場合、`OptimizeResult.prompt`
    # に壊れたテキスト（dynamic 変数を受け取れない）が残ってしまう。`_reinject_vars` は rollout 時の
    # 候補についてはこれを検出して reward=0.0 で fail-closed するが、APO の内部スコアが seed と
    # tied のときに edited candidate が best として残るケースがあり、最終結果が contract に反する
    # （契約: 「最適化済みテキストは `${var}` を保持する」）。ここで seed と同じ placeholder 集合
    # を満たさない最良候補は seed にフォールバックし、warnings で利用者へ知らせる（silent failure
    # を避ける）。
    from ..runtime.lightning._placeholders import BOUNDARY_PREFIX, extract_placeholders

    placeholder_fallback = False
    seed_placeholders = extract_placeholders(seed_text)
    if seed_placeholders:
        best_placeholders = extract_placeholders(best_text)
        # 通常 placeholder は存在検査（set 差分）で欠落を検出する。予約接頭辞
        # `oas_boundary_` を持つ境界マーカーは、slot 境界の再構成に出現順まで一致する
        # 必要があるため、`boundary_intact`（順序込みの連番列比較）で判定する。存在検査
        # のみでは best 側で count 一致・順序不整合の swap ケースを取りこぼし
        # （`_recompose_new_shape_results` が silent continue して literal マーカーが
        # OptimizeResult に漏出する契約穴）。C2 対応で `boundary_intact` に一本化する。
        from ..runtime.lightning._placeholders import boundary_intact

        if any(name.startswith(BOUNDARY_PREFIX) for name in seed_placeholders) and not (
            boundary_intact(seed_text, best_text)
        ):
            boundary_mismatched = {
                name for name in seed_placeholders if name.startswith(BOUNDARY_PREFIX)
            }
        else:
            boundary_mismatched = set()
        missing = sorted((seed_placeholders - best_placeholders) | boundary_mismatched)
        if missing:
            names = ", ".join(repr(n) for n in missing)
            warnings.warn(
                f"slot {slot_name!r} の APO 最良候補が seed の `${{var}}` を喪失しています "
                f"（不足: {names}）。OptimizeResult.prompt が dynamic 変数を保持しなくなる "
                "契約違反を避けるため seed にフォールバックします（rounds / beam を増やすか、 "
                "reward 設計で placeholder 保持を強化してください）。",
                RuntimeWarning,
                stacklevel=2,
            )
            best_text = seed_text
            placeholder_fallback = True

    # APO 内部の最良スコア / version を履歴に拾う。`_history_best_score` は agent-lightning 0.3
    # 内部実装に依存（version pin は `>=0.3,<0.3.1`・lightning extra）。`-inf`（APO 初期値で
    # 全ラウンド未更新時）と非 finite は JSON 互換 / 利用者誤読防止のため None に正規化する。
    # 属性自体が欠落していた場合は agent-lightning の API rename を疑い、silent failure にしない
    # よう warnings で顕在化させる（version pin で予防しているが、ローカル overrides 等を想定）。
    if not hasattr(apo, "_history_best_score"):
        warnings.warn(
            "agent-lightning APO に `_history_best_score` 属性がありません。"
            "OptimizeResult.history.best_score は None になります。"
            "agentlightning のバージョン互換性を確認してください "
            "（lightning extra の pin: agentlightning[apo]>=0.3,<0.3.1）。",
            RuntimeWarning,
            stacklevel=2,
        )
    raw_best = getattr(apo, "_history_best_score", None)
    best_score: float | None = (
        float(raw_best) if isinstance(raw_best, int | float) and math.isfinite(raw_best) else None
    )
    # placeholder fallback が発火した場合、best_score / best_version は破棄された候補の値を
    # 指してしまうため、利用者の誤読防止のため None に上書きし、`placeholder_fallback: True`
    # フラグで programmatic に検出可能にする（warnings filter に依存しない経路・Codex 第3 round）。
    from ..runtime.lightning.types import HistoryEntry

    history_entry: HistoryEntry = {
        "slot": slot_name,
        "best_score": None if placeholder_fallback else best_score,
        "best_version": None
        if placeholder_fallback
        else getattr(apo, "_history_best_version", None),
        "placeholder_fallback": placeholder_fallback,
    }
    return best_text, history_entry


def _build_partial(
    *,
    current: dict[str, str],
    history: list[Any],
    vars_map: dict[str, dict[str, str]],
    failed_slot: str | None,
) -> Any:
    """完了済み slot の部分成果 `OptimizePartial` を組み立てる（best-effort・失敗時 None）。

    `${var}` 再注入は run_apo 正常経路と同一規則（`substitute_braced`・braced のみ）。
    `substitute_braced` は vars 値へ `str()` を適用するため、`__str__` が例外を投げる利用者
    オブジェクトで組み立て自体が失敗しうる。診断のための処理を新たな失敗源にしないため、
    失敗時は None を返して元例外を優先する（fail-safe）。

    Args:
        current: 現在の候補 mapping（完了済み slot は最良テキスト・未処理 slot は seed のまま）。
        history: 完了済み slot の `HistoryEntry` 列（この `slot` キーの集合が「完了済み」の SoT）。
        vars_map: `{slot 名: vars}` の再注入 mapping。
        failed_slot: 失敗した slot 名（None は全 slot 完了・スコア再計算段の失敗）。

    Returns:
        `OptimizePartial`。保全対象がない（history 空）または組み立て失敗時は None。
    """
    if not history:
        return None
    from ..runtime.lightning._placeholders import substitute_braced
    from ..runtime.lightning.types import OptimizePartial

    try:
        completed = {
            entry["slot"]: substitute_braced(current[entry["slot"]], vars_map.get(entry["slot"]))
            for entry in history
        }
        return OptimizePartial(
            completed_slots=completed,
            # to_dict と同様に各 entry を浅コピーして独立化する（利用者が partial 経由で
            # 書き換えても run_apo ローカルの history と共有参照にならない）。
            history=[dict(entry) for entry in history],
            failed_slot=failed_slot,
        )
    except Exception:
        logger.warning(
            "[apo] 部分成果の組み立てに失敗しました（partial なしで継続）", exc_info=True
        )
        return None


def _raise_run_apo_failure(
    exc: BaseException,
    *,
    partial: Any,
    trainer_message: str,
) -> None:
    """run_apo の失敗を 3 系統へ振り分けて送出する（X1 / X2 共有・partial 保全ポリシーの単一定義）。

    - 既に `OptimizeError` の失敗（NFR-8 fail-closed の CONFIG_MISSING 等）: kind / message を
      保つため再ラップせず、`partial` が空のときのみ属性後付けして re-raise する。
    - `ImportError`（agentlightning の `[apo]` 系サブ依存（poml 等）の遅延 import 失敗、または
      rollout 内の利用者コード import 失敗）: kind 契約（`EXTRA_MISSING`）を維持しつつ
      `OptimizeError` に包み `partial` を保全する（生 raise は完了済み slot の成果を捨てる）。
    - その他: `TRAINER_FAILED` として `trainer_message` で送出する。

    Args:
        exc: 捕捉した例外。
        partial: `_build_partial` の結果（None 可）。
        trainer_message: `TRAINER_FAILED` 用の完成済みメッセージ。

    Raises:
        OptimizeError: 常に送出する（本関数は戻らない）。
    """
    from ..runtime.lightning.types import (
        FailureKind,
        OptimizeError,
        _format_exception_message,
    )

    if isinstance(exc, OptimizeError):
        if exc.partial is None:
            exc.partial = partial
        raise exc
    if isinstance(exc, ImportError):
        hint = (
            "（完了済み slot の部分成果は error.partial から取得できます）"
            if partial is not None
            else ""
        )
        raise OptimizeError(
            FailureKind.EXTRA_MISSING,
            "APO の実行に必要な import に失敗しました"
            "（agentlightning のサブ依存（poml 等）の欠落、または rollout 内の"
            f"利用者コード import 失敗）: {_format_exception_message(exc)}{hint}",
            partial=partial,
        ) from exc
    raise OptimizeError(FailureKind.TRAINER_FAILED, trainer_message, partial=partial) from exc


async def run_apo(
    *,
    seeds: dict[str, str],
    train: Sequence[Any],
    val: Sequence[Any] | None,
    rollout: Callable[[dict[str, str], Any], Awaitable[float]],
    config: Any,
    vars_per_slot: dict[str, dict[str, str]] | None = None,
) -> OptimizeResult:
    """APO 最適化ループを agent-lightning Trainer / APO へ委譲し plain 結果へ変換する（NFR-1）。

    APO 0.3.0 は単一プロンプトの beam search 最適化のため、複数スロット mapping は順次最適化する
    （各スロットを順に APO に通し、最良候補で他スロットの seed を上書き）。Trainer.fit 完了後に
    最良候補で train / val を改めて rollout して合成スコアを再計算する（複数スロット時の合成効果を
    正しく反映）。

    seed / 最適化済みテキストに `vars_per_slot` を再注入し、before / after の unified diff を
    `OptimizeResult.diff` に算出して詰める。新 shape slot の固定セグメントを含む full 再合成は
    optimizer 側の `_recompose_new_shape_results` が `Slot.segments` を SoT に組み直す（本関数は
    tune 側の `${var}` 再注入のみ担う）。custom build / 生 seed + rebind 経路（segments 空）では
    tune そのままが返る。

    Args:
        seeds: 最適化対象スロットの seed テキスト（`{名前: seed}`・`${var}` 保持・tune 部分のみ）。
        train: 学習用ケース列（必須・利用者供給）。
        val: 検証用ケース列（APO 必須）。None / 空は呼び出し側 `optimizer` が
            `OptimizeError(CONFIG_MISSING)` へ倒す。
        rollout: `(候補スロット mapping, ケース) -> 報酬`（non-blocking）。候補適用 rollout と
            利用者 reward を内部実行する。
        config: 実行制御設定（`OptimizeConfig` 想定・`apo_client` 必須）。`apo_client` 未供給は
            呼び出し側 `optimizer` が `OptimizeError(CONFIG_MISSING)` へ倒す。
        vars_per_slot: `{名前: Slot.vars}` の mapping（`${var}` 再注入対象・None / 空は no-op）。
            `substitute_braced`（braced `${name}` のみ・bare `$var` は非対象）で tune 側に注入する。

    Returns:
        plain `OptimizeResult`（合成済み full の seed / prompt・unified diff・train / val スコア
        + 履歴）。
    """
    _require_agentlightning()

    # NFR-1: コア（`_adapters`）は runtime を module top で import しない（`docs/architecture.md`
    # 単方向依存）。本関数内の遅延 import で参照する（`_make_litagent` / `_run_apo_single_slot` /
    # `_compose_full` でも同じパターン）。
    from ..runtime.lightning.types import (
        HistoryEntry,
        OptimizeResult,
        _format_exception_message,
    )

    current = dict(seeds)
    history: list[HistoryEntry] = []
    vars_map = dict(vars_per_slot or {})
    slot_names = list(seeds)
    for index, slot_name in enumerate(slot_names, start=1):
        # X1: slot 途中失敗。完了済み slot（1..index-1）の最良テキストと履歴は API コストを
        # 払って確定済みで、失敗とともにローカル変数ごと破棄しない（`OptimizeError.partial`
        # へ添付・pre-flight の `coverage` 保全と同じ動機）。
        try:
            best_text, entry = await _run_apo_single_slot(
                slot_name=slot_name,
                other_candidates=current,
                train=train,
                val=val,  # type: ignore[arg-type]  # 呼び出し側で None でないことを保証
                rollout=rollout,
                config=config,
            )
        except Exception as exc:
            partial = _build_partial(
                current=current, history=history, vars_map=vars_map, failed_slot=slot_name
            )
            hint = (
                "（完了済み slot の部分成果は error.partial から取得できます）"
                if partial is not None
                else ""
            )
            _raise_run_apo_failure(
                exc,
                partial=partial,
                trainer_message=(
                    f"slot {index}/{len(slot_names)} {slot_name!r} の APO 実行に失敗しました: "
                    f"{_format_exception_message(exc)}{hint}"
                ),
            )
        current[slot_name] = best_text
        history.append(entry)

    # X2: 全 slot 完了後の失敗（スコア再計算・`${var}` 再注入・diff 算出）。最適化そのものは
    # 全部成功しているため、全 slot の最良テキストを partial に載せて全損を防ぐ。
    try:
        # 全スロット最良の合成候補で train / val を再計算（複数スロット時の合成効果を反映）。
        train_score = await _score_candidate(current, train, rollout)
        val_score = await _score_candidate(current, val or [], rollout) if val else None

        # tune 側 `${var}` を rollout 実体と同じ規則で再注入する（固定セグメントは新 shape の
        # `_recompose_new_shape_results` が `Slot.segments` から compose_from_marked で組み直す・
        # custom build / 生 seed 経路は再注入なしで返す）。braced (`${var}`) のみ置換で bare
        # `$var` 副作用を避ける。
        from ..runtime.lightning._placeholders import substitute_braced, unified_diff_labeled

        composed_seeds = {
            name: substitute_braced(seeds[name], vars_map.get(name)) for name in seeds
        }
        composed_prompts = {
            name: substitute_braced(current[name], vars_map.get(name)) for name in current
        }

        diffs = {
            name: unified_diff_labeled(composed_seeds[name], composed_prompts[name])
            for name in current
        }
    except Exception as exc:
        partial = _build_partial(
            current=current, history=history, vars_map=vars_map, failed_slot=None
        )
        hint = (
            "（全 slot の最適化は完了済みです。最良テキストは error.partial から取得できます）"
            if partial is not None
            else ""
        )
        _raise_run_apo_failure(
            exc,
            partial=partial,
            trainer_message=(
                f"最適化結果のスコア再計算・整形に失敗しました: "
                f"{_format_exception_message(exc)}{hint}"
            ),
        )

    prompt: str | dict[str, str]
    seed_out: str | dict[str, str]
    diff_out: str | dict[str, str]
    if len(current) == 1:
        only_name = next(iter(current))
        prompt = composed_prompts[only_name]
        seed_out = composed_seeds[only_name]
        diff_out = diffs[only_name]
    else:
        prompt = dict(composed_prompts)
        seed_out = dict(composed_seeds)
        diff_out = dict(diffs)

    return OptimizeResult(
        prompt=prompt,
        seed=seed_out,
        diff=diff_out,
        train_score=train_score,
        val_score=val_score,
        history=history,
    )


async def judge_score(
    *,
    rubric: str,
    model: Any,
    output: str,
    case: Any,
) -> float:
    """利用者 Judge モデルで rollout 出力を 0.0..1.0 で採点する（SDK 結合を閉じる・NFR-1）。

    最小エージェント（instructions=rubric）を 1 ターン実行し、最終出力テキストから先頭の数値を
    0.0..1.0 にクランプして報酬とする（数値が取れなければ 0.0・fail-closed）。`model` は SDK
    `Model` / モデル名文字列等の不透明値で、`agents.Runner.run` の `model` として渡す。SDK 結合
    （`agents` の `Agent` / `Runner`）は本 `_adapters` に閉じ、reward 層へ SDK 型を出さない。

    Args:
        rubric: 採点観点文（利用者供給・最小エージェントの instructions に使う）。
        model: 採点に使う LLM（不透明値）。
        output: rollout が生成した最終出力テキスト。
        case: 採点対象の入力ケース（rubric の参照用に文字列化して渡す）。

    Returns:
        0.0..1.0 にクランプした報酬（数値抽出不能なら 0.0）。
    """
    from agents import Agent, Runner

    agent = Agent(name="oai-agentspec-lightning-judge", instructions=rubric, model=model)
    prompt = f"case: {case}\noutput: {output}\nscore (0.0-1.0):"
    result = await Runner.run(agent, prompt)
    text = "" if result.final_output is None else str(result.final_output)
    return _parse_score(text)


def _parse_score(text: str) -> float:
    """採点テキストから先頭の数値を 0.0..1.0 にクランプして返す（抽出不能は 0.0）。

    Args:
        text: 採点モデルの出力テキスト。

    Returns:
        0.0..1.0 にクランプした float（数値が無ければ 0.0）。
    """
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match is None:
        return 0.0
    return max(0.0, min(1.0, float(match.group())))


async def _score_candidate(
    candidate: dict[str, str],
    cases: Sequence[Any],
    rollout: Callable[[dict[str, str], Any], Awaitable[float]],
) -> float:
    """候補スロットを全ケースで rollout し平均報酬を返す（空ケースは 0.0）。

    Args:
        candidate: 候補スロット mapping（`{名前: 候補テキスト}`・`${var}` 保持）。
        cases: 評価に使う入力ケース群。
        rollout: `(候補スロット mapping, ケース) -> 報酬`（non-blocking）。

    Returns:
        全ケースの平均報酬（ケースが空なら 0.0）。
    """
    cases_list = list(cases)
    if not cases_list:
        return 0.0
    total = 0.0
    for case in cases_list:
        total += float(await rollout(dict(candidate), case))
    return total / len(cases_list)
