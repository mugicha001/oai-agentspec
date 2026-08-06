"""内容ガードレール helper ファクトリ（plain 検知 × SDK 接着・利用者向け公開 API）。

`_detectors.py` の plain 検知ファクトリと `_adapters/guardrails.py` の SDK 接着を組み合わせ、
利用者が `AgentSpec.input_guardrails` / `output_guardrails` 専用フィールドへ直接渡せる SDK 互換
`InputGuardrail` / `OutputGuardrail`、および `FunctionTool` へ tool guardrail を装着した
ラップ済みツールを返す helper を提供する（`agents.Agent` と同型の宣言面）。

helper はファクトリに徹する。重い専門検知（PII / モデレーション / 注入検知サービス）は lib 非同梱で
利用者 DI、既定 helper（注入ベースライン等）は DI で上書き / 拡張できる。判定 model / prompt は
すべて利用者 DI（プロンプト / モデル非同梱）。
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from ..._adapters.guardrails import (
    attach_tool_guardrails,
    build_async_input_guardrail,
    build_async_output_guardrail,
    build_context_output_guardrail,
    build_input_guardrail,
    build_output_guardrail,
    build_tool_input_guardrail,
    build_tool_output_guardrail,
    run_judge_prompt,
)
from ._detectors import (
    Detection,
    allow_deny_detector,
    canary_detector,
    injection_baseline_detector,
    length_detector,
    regex_detector,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    # SDK 型は `_adapters.guardrails` 経由で参照する（`from agents` を直接書かず SDK 隔離を保つ・
    # 戻り型注釈の具体化用。実行時は不要なので TYPE_CHECKING に閉じる）。
    from ..._adapters.guardrails import (
        FunctionTool,
        InputGuardrail,
        OnTrip,
        OutputGuardrail,
        ToolInputGuardrail,
        ToolOutputGuardrail,
    )

# guardrail 名（トレース用・公開窓口で安定名を固定）。
_PROMPT_LLM_NAME = "prompt_llm_guardrail"
_CANARY_NAME = "canary_guardrail"
_PREDICATE_NAME = "predicate_guardrail"
_REGEX_NAME = "regex_guardrail"
_LENGTH_NAME = "length_guardrail"
_ALLOW_DENY_NAME = "allow_deny_guardrail"
_INJECTION_NAME = "injection_baseline_guardrail"
_EXTERNAL_NAME = "external_detector_guardrail"

# ツール境界 guardrail 名（agent 境界名を流用するとトレース識別子が誤誘導になるため専用名）。
_TOOL_INPUT_NAME = "tool_input_guardrail"
_TOOL_OUTPUT_NAME = "tool_output_guardrail"

# prompt 駆動 LLM judge が trip と判定したとみなす既定トークン（判定出力に含まれれば trip）。
_DEFAULT_TRIP_TOKEN = "UNSAFE"


def _default_verdict(text: str) -> Detection:
    """prompt 駆動 LLM judge の出力テキストから trip 判定を導く既定パーサ。

    判定出力に既定トークン（`UNSAFE`・大文字小文字無視）が含まれていれば trip とみなす。判定の
    解釈方法を変えたい場合は `prompt_llm_guardrail(verdict=...)` で利用者 DI のパーサに差し替える
    （判定 prompt 側で出力フォーマットを指定する想定）。

    Args:
        text: judge model の出力テキスト。

    Returns:
        trip 判定の `Detection`。
    """
    triggered = _DEFAULT_TRIP_TOKEN.lower() in text.lower()
    return Detection(
        triggered=triggered,
        reason="llm judge flagged content" if triggered else None,
        info={"verdict": text},
    )


def prompt_llm_guardrail(
    model: Any,
    prompt: str,
    *,
    on: str,
    verdict: Callable[[str], Detection] | None = None,
    name: str | None = None,
    run_in_parallel: bool = True,
) -> InputGuardrail | OutputGuardrail:
    """prompt 駆動 LLM guardrail を作る（LLM-as-judge・判定 model / prompt は利用者 DI・NFR-1）。

    判定 model と判定 prompt（本文は利用者提供・lib 非同梱）で内容を判定し、judge 出力の解釈を
    `verdict`（既定は `UNSAFE` トークン照合）で trip 判定へ写す。LLM 呼び出しは `_adapters` 経由
    （`run_judge_prompt`）へ寄せ外部直叩きを避ける。`on="input"` で `InputGuardrail`、`on="output"`
    で `OutputGuardrail` を返す。

    既定 `verdict` は judge 出力が空 / 不正のとき trip しない（fail-open）。fail-closed が必要なら
    `verdict` DI で空応答を trip 扱いにするパーサを渡すこと。

    `run_in_parallel`（既定 True・SDK 既定）は**入力境界（`on="input"`）にのみ効く**。True だと
    判定 LLM の呼び出しがエージェントのターンと並行に走るため、判定が trip する前にモデルがツールを
    呼びうる。ツール実行の副作用はツール境界ガードレール（`guard_tool` / `ToolInputGuardrail`）が
    実行前にゲートする役割分担を前提とする。ツールガードレールを併用せず本 guardrail 単体で実行前
    ブロックを保証したい場合は `run_in_parallel=False` を指定する（SDK が判定完了を待ってから
    ターンを開始する）。`on="output"` のときは無視される（`OutputGuardrail` に該当フィールドなし）。

    Args:
        model: 判定に使う LLM（SDK `Model` / モデル名文字列 等の不透明値・利用者 DI）。
        prompt: 判定 prompt 本文（利用者提供）。
        on: 適用境界（"input" or "output"・キーワード必須）。
        verdict: judge 出力テキスト → `Detection` の解釈関数（None で既定トークン照合）。
        name: guardrail 名（None で既定名）。
        run_in_parallel: 入力検査を並行に走らせるか（既定 True・`on="input"` のときのみ効く）。

    Returns:
        SDK 互換 `InputGuardrail`（on="input"）または `OutputGuardrail`（on="output"）。

    Raises:
        ValueError: `on` が "input" / "output" 以外の場合（他 factory と契約統一）。
    """
    _check_on(on)
    parse = verdict or _default_verdict

    async def _detect(content: str) -> Detection:
        return parse(await run_judge_prompt(model, prompt, content))

    guardrail_name = name or _PROMPT_LLM_NAME
    return _user_agent_guardrail(guardrail_name, _detect, on, run_in_parallel=run_in_parallel)


def canary_guardrail(
    canary: str | Iterable[str] | Callable[..., Any], *, name: str | None = None
) -> OutputGuardrail:
    """canary（漏洩トークン）の出力漏洩検知 guardrail を作る（システムプロンプト漏洩・C 家族）。

    出力テキストに canary が逐語で含まれていれば trip する `OutputGuardrail` を返す。canary 値は
    利用者 DI（システムプロンプトへ埋め込んだトークン等）。

    固定値（`str` / `Iterable[str]`）に加えて **resolver（callable）** を受ける。resolver の契約は
    `(context, agent) -> str | Iterable[str] | None` で、`context` は SDK の `RunContextWrapper` の
    まま渡る（動的 instructions / `is_enabled` と同じ鏡写し規約）。resolver は構築時には評価せず
    **検知呼び出しごとに再評価**し、その回の戻り値で逐語照合する（run ごとに変わるトークンを扱う
    ため）。`None` / 空文字列 / 空 iterable を返した場合は「この run にカナリアが無い」状態として
    trip しない。resolver が上記以外の型（`Mapping` / `bytes` / スカラー / 非 str 要素を含む
    iterable 等）を返した場合は検知呼び出しで `TypeError` を上げる（`Mapping` を通すとキー列が
    照合対象になり、実トークンが一切照合されないまま恒久 fail-open になるため）。構築時に行うのは
    引数規約の bind 検証と同期関数であることの検証のみ。

    Args:
        canary: 照合する canary 値（単一 or 複数）、または run ごとに解決する resolver。
        name: guardrail 名（None で既定名）。

    Returns:
        SDK 互換 `OutputGuardrail`。

    Raises:
        ValueError: resolver が `(context, agent)` の 2 引数で呼び出せない場合、または
            resolver が `async def`（コルーチン関数・`async def __call__` を持つ callable object を
            含む）の場合。
    """
    guardrail_name = name or _CANARY_NAME
    if isinstance(canary, str) or not callable(canary):
        return build_output_guardrail(guardrail_name, canary_detector(canary))

    resolver = canary
    _validate_canary_resolver(resolver)

    def detect(context: Any, agent: Any, text: str) -> Detection:
        tokens = _canary_tokens(resolver(context, agent))
        return canary_detector(tokens)(text)

    return build_context_output_guardrail(guardrail_name, detect)


def _canary_tokens(resolved: Any) -> tuple[str, ...]:
    """canary resolver の解決値を照合用トークン列へ正規化する（型制約の適用点）。

    受理するのは `str` / `Iterable[str]` / `None` のみ。`Mapping` はキー列が照合対象になり
    実トークンが一切照合されないまま恒久 fail-open になるため、iterable であっても拒否する
    （`bytes` も反復すると int 列になるため拒否する）。使い切り iterable（generator 等）を
    壊さないよう、要素検査の前に tuple 化して以降その tuple を使う。

    例外メッセージには型名のみを載せ、解決値（トークン候補）は載せない（漏洩面を広げない）。

    Args:
        resolved: resolver の戻り値。

    Returns:
        逐語照合に使うトークンの tuple（`None` は空 tuple）。

    Raises:
        TypeError: `str` / `Iterable[str]` / `None` 以外の値を受けた場合。
    """
    if resolved is None:
        return ()
    if isinstance(resolved, str):
        return (resolved,)
    if isinstance(resolved, Mapping | bytes | bytearray) or not isinstance(resolved, Iterable):
        raise TypeError(
            "canary_guardrail の resolver は str / Iterable[str] / None を返す必要がありますが "
            f"{type(resolved).__name__!r} を返しました"
        )
    tokens = tuple(resolved)
    for token in tokens:
        if not isinstance(token, str):
            raise TypeError(
                "canary_guardrail の resolver が返す iterable の要素は str である必要が"
                f"ありますが {type(token).__name__!r} が含まれます"
            )
    return tokens


def _validate_canary_resolver(resolver: Callable[..., Any]) -> None:
    """canary resolver が同期関数かつ `(context, agent)` の 2 引数で呼び出せることを検証する。

    `validate_instructions_callable` と同種の bind 検証（デフォルト引数・可変長は許容）に加え、
    公開契約が同期のみ（ADR 0023 判断 9）であることを構築時に確かめる（`async def` を通すと検知時に
    未 await の coroutine が照合へ流れ、guardrail が壊れたことに気付きにくい）。実行時まで遅延させ
    ると、guardrail は登録済みで正常に見えるまま run で初めて壊れる。シグネチャ取得不能な callable
    （builtin 等）は bind 検証をスキップする。

    同期性の検査は関数形（`async def` / `functools.partial(async def)`）だけでなく
    `type(resolver).__call__` も対象にする（`inspect.iscoroutinefunction` は `async def __call__`
    を持つ callable object に対して False を返すため、関数形だけの検査ではすり抜けて検知時の
    未 await coroutine 事故になる）。同期 `__call__` を持つ callable object は受理する。

    Args:
        resolver: 検証対象の resolver。

    Raises:
        ValueError: resolver が `async def`（コルーチン関数・`async def __call__` を持つ callable
            object を含む）の場合、または `(context, agent)` の 2 引数で呼び出せない場合。
    """
    # resolver は callable であることが呼び出し側で保証されるため `type(resolver).__call__` は
    # 必ず存在する（関数の場合は slot wrapper で非コルーチン扱いになる）。
    if inspect.iscoroutinefunction(resolver) or inspect.iscoroutinefunction(
        type(resolver).__call__
    ):
        raise ValueError(
            "canary_guardrail の resolver は同期関数である必要があります"
            "（async def は受理しません）"
        )
    try:
        sig = inspect.signature(resolver)
    except (ValueError, TypeError):
        return
    try:
        sig.bind(object(), object())
    except TypeError:
        raise ValueError(
            "canary_guardrail の resolver は (context, agent) の 2 引数で呼び出せる必要があります"
        ) from None


def predicate_guardrail(
    predicate: Callable[[str], bool | Awaitable[bool]],
    *,
    on: str,
    reason: str | None = None,
    name: str | None = None,
    run_in_parallel: bool = True,
) -> InputGuardrail | OutputGuardrail:
    """汎用 predicate guardrail を作る（任意ロジックを DI 述語で差し込む・C 家族）。

    `predicate(text)` が True を返したら trip する。`on="input"` / `on="output"` で適用境界を選ぶ。
    述語は同期（`bool` を返す）でも非同期（`async def` / `async __call__`・`Awaitable[bool]` を
    返す）でもよく、戻り値が awaitable なら await して扱う。

    `run_in_parallel`（既定 True・SDK 既定）は**入力境界（`on="input"`）にのみ効く**。並行実行が
    既定のため遅い / async 述語が trip する前にモデルがツールを呼びうる（実行前ブロックが必要なら
    `run_in_parallel=False` を指定するか、ツール境界ガードレールを併用する）。

    Args:
        predicate: テキストを受けて検知有無（bool）を返す述語（DI・同期 / 非同期どちらも可）。
        on: 適用境界（"input" or "output"・キーワード必須）。
        reason: 検知時の理由（任意）。
        name: guardrail 名（None で既定名）。
        run_in_parallel: 入力検査を並行に走らせるか（既定 True・`on="input"` のときのみ効く）。

    Returns:
        SDK 互換 `InputGuardrail`（on="input"）または `OutputGuardrail`（on="output"）。
    """

    async def detect(text: str) -> Detection:
        result = predicate(text)
        if inspect.isawaitable(result):
            result = await result
        if result:
            return Detection(triggered=True, reason=reason or "predicate matched")
        return Detection(triggered=False)

    return _user_agent_guardrail(
        name or _PREDICATE_NAME, detect, on, run_in_parallel=run_in_parallel
    )


def regex_guardrail(
    patterns: str | Iterable[str],
    *,
    on: str,
    flags: int = 0,
    name: str | None = None,
    run_in_parallel: bool = True,
) -> InputGuardrail | OutputGuardrail:
    """正規表現 guardrail を作る（DI パターンへのマッチで trip・C 家族）。

    `run_in_parallel`（既定 True・SDK 既定）は**入力境界（`on="input"`）にのみ効く**（並行実行が
    既定・実行前ブロックが必要なら `run_in_parallel=False` か、ツール境界ガードレールを併用する）。

    Args:
        patterns: 検知に使う正規表現（単一 or 複数・利用者 DI）。
        on: 適用境界（"input" or "output"・キーワード必須）。
        flags: `re.compile` フラグ（既定 0）。
        name: guardrail 名（None で既定名）。
        run_in_parallel: 入力検査を並行に走らせるか（既定 True・`on="input"` のときのみ効く）。

    Returns:
        SDK 互換 `InputGuardrail`（on="input"）または `OutputGuardrail`（on="output"）。
    """
    detect = regex_detector(patterns, flags=flags)
    return _agent_guardrail(name or _REGEX_NAME, detect, on, run_in_parallel=run_in_parallel)


def length_guardrail(
    *,
    max_length: int | None = None,
    min_length: int | None = None,
    on: str,
    name: str | None = None,
    run_in_parallel: bool = True,
) -> InputGuardrail | OutputGuardrail:
    """長さ / サイズ閾値 guardrail を作る（無制限消費の粗い網・C 家族）。

    テキスト長が `max_length` 超過 / `min_length` 未満なら trip する。`max_length` /
    `min_length` の**少なくとも一方は必須**で、両方 None なら（無言の no-op を避けるため）
    `ValueError` を上げる。

    `run_in_parallel`（既定 True・SDK 既定）は**入力境界（`on="input"`）にのみ効く**（並行実行が
    既定・実行前ブロックが必要なら `run_in_parallel=False` か、ツール境界ガードレールを併用する）。

    Args:
        max_length: 上限文字数（超過で trip）。None で上限なし。
        min_length: 下限文字数（未満で trip）。None で下限なし。
        on: 適用境界（"input" or "output"・キーワード必須）。
        name: guardrail 名（None で既定名）。
        run_in_parallel: 入力検査を並行に走らせるか（既定 True・`on="input"` のときのみ効く）。

    Returns:
        SDK 互換 `InputGuardrail`（on="input"）または `OutputGuardrail`（on="output"）。

    Raises:
        ValueError: `max_length` / `min_length` が両方 None の場合（閾値が無く no-op になるため）。
    """
    if max_length is None and min_length is None:
        raise ValueError("length_guardrail は max_length / min_length の少なくとも一方が必須です")
    detect = length_detector(max_length=max_length, min_length=min_length)
    return _agent_guardrail(name or _LENGTH_NAME, detect, on, run_in_parallel=run_in_parallel)


def allow_deny_guardrail(
    *,
    deny: Iterable[str] | None = None,
    allow: Iterable[str] | None = None,
    case_sensitive: bool = True,
    on: str,
    name: str | None = None,
    run_in_parallel: bool = True,
) -> InputGuardrail | OutputGuardrail:
    """allow / deny リスト guardrail を作る（部分文字列照合・C 家族）。

    `deny` のいずれかが含まれれば trip、`allow` 指定時はいずれも含まれなければ trip する。

    `run_in_parallel`（既定 True・SDK 既定）は**入力境界（`on="input"`）にのみ効く**（並行実行が
    既定・実行前ブロックが必要なら `run_in_parallel=False` か、ツール境界ガードレールを併用する）。

    Args:
        deny: 含まれていたら trip する拒否語の集合（任意・利用者 DI）。
        allow: いずれも含まれなければ trip する許可語の集合（任意・利用者 DI）。
        case_sensitive: 大文字小文字を区別するか（既定 True）。
        on: 適用境界（"input" or "output"・キーワード必須）。
        name: guardrail 名（None で既定名）。
        run_in_parallel: 入力検査を並行に走らせるか（既定 True・`on="input"` のときのみ効く）。

    Returns:
        SDK 互換 `InputGuardrail`（on="input"）または `OutputGuardrail`（on="output"）。
    """
    detect = allow_deny_detector(deny=deny, allow=allow, case_sensitive=case_sensitive)
    return _agent_guardrail(name or _ALLOW_DENY_NAME, detect, on, run_in_parallel=run_in_parallel)


def injection_baseline_guardrail(
    extra_patterns: Iterable[str] | None = None,
    *,
    name: str | None = None,
    run_in_parallel: bool = True,
) -> InputGuardrail:
    """注入ベースライン guardrail を作る（SQLi / コマンド注入 / パストラバーサルの補助検知）。

    既定パターンに `extra_patterns` を追記して入力を検査する `InputGuardrail` を返す。**網羅的検知
    ではなく補助検知**であり、注入対策の本丸はパラメータ化クエリ / 安全 API 利用である（既定
    パターンは DI で上書き / 拡張可・rationale 参照）。

    `run_in_parallel`（既定 True・SDK 既定）は入力検査をエージェントのターンと並行に走らせるか
    （レイテンシ優先）を制御する。並行実行が既定のため検知が trip する前にモデルがツールを呼びうる
    ため、ツール実行の副作用はツール境界ガードレール（`guard_tool` / `ToolInputGuardrail`）が実行前
    にゲートする役割分担を前提とする。本 guardrail 単体で実行前ブロックを保証したい場合は
    `run_in_parallel=False` を指定する（SDK が検査完了を待ってからターンを開始する）。

    Args:
        extra_patterns: 既定に追記する正規表現パターン（任意・利用者 DI）。
        name: guardrail 名（None で既定名）。
        run_in_parallel: 入力検査を並行に走らせるか（既定 True・SDK 既定）。

    Returns:
        SDK 互換 `InputGuardrail`。
    """
    detect = injection_baseline_detector(extra_patterns)
    return build_input_guardrail(name or _INJECTION_NAME, detect, run_in_parallel=run_in_parallel)


def external_detector_guardrail(
    detect: Callable[[str], Detection | Awaitable[Detection]],
    *,
    on: str,
    name: str | None = None,
    run_in_parallel: bool = True,
) -> InputGuardrail | OutputGuardrail:
    """外部検知器 guardrail を作る（Presidio / モデレーション等の利用者検知を薄く包む・A 家族）。

    利用者の検知 callable（テキスト → `Detection`）をそのまま SDK 互換 guardrail へ接着する。
    検知本体は lib 非同梱（外部 DI）。検知器は同期でも非同期でもよく（`async def` 関数・OpenAI
    Moderation など `async __call__` を持つ DI オブジェクト・同期関数のいずれも可）、builder が
    `detect(text)` の戻り値が awaitable かを見て await するため一様に扱える。

    `run_in_parallel`（既定 True・SDK 既定）は**入力境界（`on="input"`）にのみ効く**。True だと外部
    検知（Presidio / モデレーション等のネットワーク I/O）がエージェントのターンと並行に走るため、
    検知が trip する前にモデルがツールを呼びうる。ツール実行の副作用はツール境界ガードレール
    （`guard_tool` / `ToolInputGuardrail`）が実行前にゲートする役割分担を前提とする。本 guardrail
    単体で実行前ブロックを保証したい場合は `run_in_parallel=False` を指定する（SDK が検知完了を
    待ってからターンを開始する）。`on="output"` のときは無視される（`OutputGuardrail` に該当なし）。

    Args:
        detect: テキストを受けて `Detection`（同期）または `Awaitable[Detection]`（非同期）を返す
            利用者検知関数（利用者 DI）。
        on: 適用境界（"input" or "output"・キーワード必須）。
        name: guardrail 名（None で既定名）。
        run_in_parallel: 入力検査を並行に走らせるか（既定 True・`on="input"` のときのみ効く）。

    Returns:
        SDK 互換 `InputGuardrail`（on="input"）または `OutputGuardrail`（on="output"）。
    """
    return _user_agent_guardrail(
        name or _EXTERNAL_NAME, detect, on, run_in_parallel=run_in_parallel
    )


def tool_guardrail(
    detector: Callable[[str], Detection | Awaitable[Detection]],
    *,
    on: str,
    on_trip: OnTrip | Callable[[Detection], Any] = "reject",
    name: str | None = None,
) -> ToolInputGuardrail | ToolOutputGuardrail:
    """検知器を SDK ネイティブの tool guardrail へ接着して返す（ツール定義時に宣言する用途）。

    `function_tool(_func, tool_input_guardrails=[...], tool_output_guardrails=[...])` のように
    ツール定義時にそのまま渡せる `ToolInputGuardrail` / `ToolOutputGuardrail` を生成する（agent
    境界ファクトリと対称の宣言面）。`on="input"` でツール引数を、`on="output"` で中間ツール出力を
    検査する guardrail を返す。既存ツール（`as_tool` 等 `function_tool` で定義できないもの）への
    **後付け**は `guard_tool` を使う。

    検知器は同期（`Detection` を返す）でも非同期（`async def` / `async __call__`・
    `Awaitable[Detection]` を返す）でもよく、`build_tool_*_guardrail` が `detector(text)` の戻り値が
    awaitable かを見て await する（Presidio / モデレーション等の async 検知器をそのまま渡せる）。

    trip 時の挙動は `on_trip`（"reject" 既定 = 注釈付き返却で続行 / "raise" = 中断 / "allow" =
    通過、または `Detection` を受ける callable DI）で選ぶ。`on_trip` に不正な文字列を渡すと
    `ValueError`（typo の早期検出・callable はそのまま許可）。

    Args:
        detector: テキスト（ツール引数 / 出力）を受けて `Detection` を返す検知関数（同期/非同期）。
        on: 適用境界（"input" or "output"・キーワード必須）。
        on_trip: trip 時挙動の選択（文字列 or callable DI・既定 "reject"）。
        name: guardrail 名（None で既定の tool 専用名）。

    Returns:
        SDK 互換 `ToolInputGuardrail`（on="input"）または `ToolOutputGuardrail`（on="output"）。

    Raises:
        ValueError: `on` が "input" / "output" 以外の場合。
    """
    _check_on(on)
    if on == "input":
        return build_tool_input_guardrail(name or _TOOL_INPUT_NAME, detector, on_trip=on_trip)
    return build_tool_output_guardrail(name or _TOOL_OUTPUT_NAME, detector, on_trip=on_trip)


def guard_tool(
    tool: FunctionTool,
    *,
    input_detector: Callable[[str], Detection | Awaitable[Detection]] | None = None,
    output_detector: Callable[[str], Detection | Awaitable[Detection]] | None = None,
    on_trip: OnTrip | Callable[[Detection], Any] = "reject",
) -> FunctionTool:
    """既存 `FunctionTool` へネイティブ tool guardrail を**後付け**装着して返す（ツール境界）。

    `function_tool` で定義し直せない既存ツール（`as_tool` 等）へ retrofit する用途。ツール定義時に
    宣言できる場合は `function_tool(_func, tool_*_guardrails=[tool_guardrail(...)])` を使う。

    `input_detector` でツール引数を、`output_detector` で中間ツール出力を検査する SDK ネイティブ
    tool guardrail を `tool_guardrail` で生成し、`dataclasses.replace` で `tool_input_guardrails` /
    `tool_output_guardrails` へ装着する。`name` / `description` / `params_json_schema` /
    `needs_approval` / `on_invoke_tool` は維持する（実行本体・宣言メタは変えない）。ここで行うのは
    **内容検査のみ**であり、実行可否の allow / deny 制御（ポリシー強制）は新設しない（AGT ガバナンス
    の責務）。trip 時の挙動は `on_trip`（"reject" 既定 = 注釈付き返却で続行 / "raise" = 中断 /
    "allow" = 通過、または `Detection` を受ける callable DI）で選ぶ。`on_trip` に不正な文字列を
    渡すと `ValueError`（typo の早期検出・callable はそのまま許可）。

    検知器は同期（`Detection` を返す）でも非同期（`async def` / `async __call__`・
    `Awaitable[Detection]` を返す）でもよく、`detector(text)` の戻り値が awaitable かを見て await
    する（Presidio / モデレーション等の async 検知器をそのまま渡せる）。

    Args:
        tool: 装着対象の `FunctionTool`。
        input_detector: ツール引数の検知関数（任意・利用者 DI・同期 / 非同期どちらも可）。
        output_detector: 中間ツール出力の検知関数（任意・利用者 DI・同期 / 非同期どちらも可）。
        on_trip: trip 時挙動の選択（文字列 or callable DI・既定 "reject"）。

    Returns:
        guardrail 装着済みの新しい `FunctionTool`（検知器未指定なら元 tool）。
    """
    input_guard = (
        tool_guardrail(input_detector, on="input", on_trip=on_trip)
        if input_detector is not None
        else None
    )
    output_guard = (
        tool_guardrail(output_detector, on="output", on_trip=on_trip)
        if output_detector is not None
        else None
    )
    return attach_tool_guardrails(tool, input=input_guard, output=output_guard)


def _check_on(on: str) -> None:
    """適用境界 `on` が "input" / "output" のいずれかであることを検証する。

    各 factory（決定的 / 利用者検知 / prompt 駆動）の `on` バリデーションを 1 か所へ集約し、
    不正値で一様に `ValueError` を上げる（黙ったフォールスルーを避け契約を統一する）。

    Args:
        on: 適用境界。

    Raises:
        ValueError: `on` が "input" / "output" 以外の場合。
    """
    if on not in ("input", "output"):
        raise ValueError(f"on must be 'input' or 'output', got {on!r}")


def _agent_guardrail(
    name: str, detect: Callable[[str], Detection], on: str, *, run_in_parallel: bool = True
) -> InputGuardrail | OutputGuardrail:
    """lib 内部の**同期**決定的検知を `on` に応じた agent 境界 guardrail へ接着する内部ヘルパ。

    regex / length / allow_deny 等の同期検知（`_detectors.py`・agents 非依存）専用。利用者が渡す
    検知器（同期 / 非同期どちらも取りうる）は `_user_agent_guardrail` を使う。`run_in_parallel` は
    入力 builder にのみ伝播する（`OutputGuardrail` に該当フィールドがないため `on=="output"` では
    使わない）。

    Args:
        name: guardrail 名。
        detect: テキストを受けて `Detection` を返す同期検知関数。
        on: 適用境界（"input" or "output"）。
        run_in_parallel: 入力検査を並行に走らせるか（既定 True・`on=="input"` のときのみ伝播）。

    Returns:
        SDK 互換 `InputGuardrail`（on="input"）または `OutputGuardrail`（on="output"）。

    Raises:
        ValueError: `on` が "input" / "output" 以外の場合。
    """
    _check_on(on)
    if on == "input":
        return build_input_guardrail(name, detect, run_in_parallel=run_in_parallel)
    return build_output_guardrail(name, detect)


def _user_agent_guardrail(
    name: str, detect: Callable[[str], Any], on: str, *, run_in_parallel: bool = True
) -> InputGuardrail | OutputGuardrail:
    """**利用者検知器**を `on` に応じた agent 境界 guardrail へ接着する内部ヘルパ。

    利用者検知器は同期 / 非同期どちらも取りうるため、常に非同期 builder
    （`build_async_*_guardrail`）へ接着する。builder 側が `detect(text)` の戻り値が awaitable か
    （型ではなく結果で判定）を見て await するため、`async def` 関数・`async __call__` を持つ DI
    オブジェクト・同期関数のいずれも取りこぼさず一様に扱える（未 await の coroutine が後段へ流れて
    `AttributeError` になる不具合を防ぐ・新たな実行エンジンは作らず既存 builder を再利用する）。
    `run_in_parallel` は入力 builder にのみ伝播する（`on=="output"` では使わない）。

    Args:
        name: guardrail 名。
        detect: テキストを受けて `Detection`（同期）または `Awaitable[Detection]`（非同期）を返す
            利用者検知器。
        on: 適用境界（"input" or "output"）。
        run_in_parallel: 入力検査を並行に走らせるか（既定 True・`on=="input"` のときのみ伝播）。

    Returns:
        SDK 互換 `InputGuardrail`（on="input"）または `OutputGuardrail`（on="output"）。

    Raises:
        ValueError: `on` が "input" / "output" 以外の場合。
    """
    _check_on(on)
    if on == "input":
        return build_async_input_guardrail(name, detect, run_in_parallel=run_in_parallel)
    return build_async_output_guardrail(name, detect)
