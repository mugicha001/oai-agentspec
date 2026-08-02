"""内容ガードレールの SDK 結合窓口（agent / tool 双方の guardrail 型を `_adapters` に閉じる）。

`from agents import ...`（guardrail 型・デコレータ・`FunctionTool` の `dataclasses.replace`）の
SDK 結合を本モジュール内に閉じる。openai-agents はコア依存のため `builders.py` と同様にトップ
レベルで `agents` を import する（`judge.py` の deepeval のような遅延 import は不要）。

上位の `runtime/guardrails/factories.py` は **plain な検知結果**（agents 非依存層 `_detectors` の
`Detection`）と本モジュールの薄い接着関数のみを扱い、SDK 型を直接見ない。検知結果 →
`GuardrailFunctionOutput` / `ToolGuardrailFunctionOutput` の写像は本モジュールで行う。

prompt 駆動 LLM guardrail の判定実行は `judge.py` を流用せず本モジュール専用の薄い実行ヘルパ
（`run_judge_prompt`）に閉じる（SDK 隔離のため・判定 model / prompt は利用者 DI）。
"""

from __future__ import annotations

import inspect
from dataclasses import replace as _dataclass_replace
from typing import TYPE_CHECKING, Any, Literal

from agents import (
    Agent,
    FunctionTool,
    GuardrailFunctionOutput,
    InputGuardrail,
    OutputGuardrail,
    Runner,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolInputGuardrailData,
    ToolOutputGuardrail,
    ToolOutputGuardrailData,
    input_guardrail,
    output_guardrail,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..runtime.guardrails._detectors import Detection

# ツール guardrail の trip 時挙動（reject_content / raise_exception / allow）の選択肢。
OnTrip = Literal["reject", "raise", "allow"]


def _to_guardrail_output(detection: Detection) -> GuardrailFunctionOutput:
    """plain な `Detection` を SDK の `GuardrailFunctionOutput` へ写す。

    `detection.triggered` を `tripwire_triggered` に写し、`reason` / `info` を `output_info`
    へ載せる（トレース用の付帯情報）。

    Args:
        detection: agents 非依存層の検知結果。

    Returns:
        SDK 互換の `GuardrailFunctionOutput`。
    """
    return GuardrailFunctionOutput(
        output_info={"reason": detection.reason, "info": detection.info},
        tripwire_triggered=detection.triggered,
    )


def build_input_guardrail(
    name: str, detect: Callable[[str], Detection], *, run_in_parallel: bool = True
) -> InputGuardrail[Any]:
    """plain 検知関数（テキスト → `Detection`）を SDK 互換 `InputGuardrail` へ接着する。

    SDK の入力 guardrail シグネチャ `(context, agent, input) -> GuardrailFunctionOutput` を
    満たす関数を `input_guardrail` デコレータで包む。`input` は文字列 / 入力アイテム列のため
    `str(input)` でテキスト化して `detect` に渡す。

    `run_in_parallel`（SDK 既定 True）は入力検査をエージェントのターンと**並行**に走らせるか
    （レイテンシ優先）を制御する。True のままだと遅い / async な検知器が trip する前にモデルが
    ツールを呼びうるため、ツール実行の副作用はツール境界ガードレール（`guard_tool` /
    `ToolInputGuardrail`）が実行前にゲートする役割分担を前提とする。ツールガードレールを併用せず
    入力ガードレール単体で実行前ブロックを保証したい場合は `run_in_parallel=False` を渡す
    （SDK が検査完了を待ってからターンを開始する）。

    Args:
        name: guardrail 名（トレース用）。
        detect: テキストを受けて `Detection` を返す plain 検知関数。
        run_in_parallel: 入力検査をエージェントのターンと並行に走らせるか（既定 True・SDK 既定）。

    Returns:
        SDK 互換 `InputGuardrail`。
    """

    @input_guardrail(name=name, run_in_parallel=run_in_parallel)
    def _guardrail(context: Any, agent: Any, input: Any) -> GuardrailFunctionOutput:  # noqa: A002, ARG001
        return _to_guardrail_output(detect(_text_of(input)))

    return _guardrail


def build_output_guardrail(name: str, detect: Callable[[str], Detection]) -> OutputGuardrail[Any]:
    """plain 検知関数（テキスト → `Detection`）を SDK 互換 `OutputGuardrail` へ接着する。

    SDK の出力 guardrail シグネチャ `(context, agent, agent_output) -> GuardrailFunctionOutput`
    を満たす関数を `output_guardrail` デコレータで包む。`agent_output` は不透明値のため
    `str(...)` でテキスト化して `detect` に渡す。

    Args:
        name: guardrail 名（トレース用）。
        detect: テキストを受けて `Detection` を返す plain 検知関数。

    Returns:
        SDK 互換 `OutputGuardrail`。
    """

    @output_guardrail(name=name)
    def _guardrail(context: Any, agent: Any, agent_output: Any) -> GuardrailFunctionOutput:  # noqa: ARG001
        return _to_guardrail_output(detect(_text_of(agent_output)))

    return _guardrail


async def _call_detect(detect: Callable[[str], Any], text: str) -> Detection:
    """検知器を呼び、戻り値が awaitable なら await して `Detection` に正規化する。

    同期関数（`Detection` を返す）・コルーチン関数（`async def`）・`async __call__` を持つ callable
    オブジェクトのいずれでも同一に扱う。型（`inspect.iscoroutinefunction`）ではなく**戻り値が
    awaitable か**（`inspect.isawaitable`）で判定するため、`async __call__` を持つ DI オブジェクトの
    取りこぼし（未 await の coroutine が後段へ流れて `AttributeError` になる）を防ぐ。

    Args:
        detect: テキストを受けて `Detection` または `Awaitable[Detection]` を返す検知器。
        text: 検知対象テキスト。

    Returns:
        正規化された `Detection`。
    """
    result = detect(text)
    if inspect.isawaitable(result):
        return await result
    return result


def build_async_input_guardrail(
    name: str, detect: Callable[[str], Any], *, run_in_parallel: bool = True
) -> InputGuardrail[Any]:
    """検知関数（同期 / 非同期どちらも可）を SDK 互換 `InputGuardrail` へ接着する。

    常に async な guardrail 関数で包み、`detect(text)` の戻り値が awaitable なら await して
    `Detection` に正規化する（`_call_detect`）。prompt 駆動 LLM guardrail のような async 検知、
    `async __call__` を持つ DI オブジェクト、同期検知のいずれも一様に扱う。

    `run_in_parallel`（SDK 既定 True）の意味と役割分担は `build_input_guardrail` と同一
    （並行実行が既定・実行前ブロックが必要なら False か、ツール境界ガードレールを併用する）。

    Args:
        name: guardrail 名（トレース用）。
        detect: テキストを受けて `Detection` または `Awaitable[Detection]` を返す検知関数。
        run_in_parallel: 入力検査をエージェントのターンと並行に走らせるか（既定 True・SDK 既定）。

    Returns:
        SDK 互換 `InputGuardrail`。
    """

    @input_guardrail(name=name, run_in_parallel=run_in_parallel)
    async def _guardrail(context: Any, agent: Any, input: Any) -> GuardrailFunctionOutput:  # noqa: A002, ARG001
        return _to_guardrail_output(await _call_detect(detect, _text_of(input)))

    return _guardrail


def build_async_output_guardrail(name: str, detect: Callable[[str], Any]) -> OutputGuardrail[Any]:
    """検知関数（同期 / 非同期どちらも可）を SDK 互換 `OutputGuardrail` へ接着する。

    常に async な guardrail 関数で包み、戻り値が awaitable なら await して正規化する
    （`_call_detect`・`build_async_input_guardrail` と同じ方針）。

    Args:
        name: guardrail 名（トレース用）。
        detect: テキストを受けて `Detection` または `Awaitable[Detection]` を返す検知関数。

    Returns:
        SDK 互換 `OutputGuardrail`。
    """

    @output_guardrail(name=name)
    async def _guardrail(context: Any, agent: Any, agent_output: Any) -> GuardrailFunctionOutput:  # noqa: ARG001
        return _to_guardrail_output(await _call_detect(detect, _text_of(agent_output)))

    return _guardrail


def _text_of(value: Any) -> str:
    """guardrail 入力 / 出力（文字列・入力アイテム列・不透明値）をテキスト化する。

    SDK の入力は `str | list[TResponseInputItem]`、出力は不透明値で渡る。検知器は plain な
    テキストを受け取るため、文字列はそのまま、それ以外は `str(...)` で平坦化する。

    Args:
        value: guardrail が受け取った入力 / 出力。

    Returns:
        検知器に渡すテキスト。
    """
    if isinstance(value, str):
        return value
    return str(value)


def _trip_output(on_trip: OnTrip, message: str, info: Any) -> ToolGuardrailFunctionOutput:
    """trip 時の `ToolGuardrailFunctionOutput` を `on_trip` 選択に従って組む。

    `OnTrip` Literal 外の不正文字列（typo 等）は黙って reject へフォールバックせず `ValueError` を
    上げる（agent 境界の `on` 引数不正で `ValueError` を上げるのと対称・契約統一）。

    Args:
        on_trip: trip 時挙動（"reject" = 出力差し替えで続行 / "raise" = 中断 / "allow" = 通過）。
        message: reject_content 時にモデルへ返す注釈メッセージ。
        info: 検知の付帯情報（`output_info` へ載せる）。

    Returns:
        選択に応じた `ToolGuardrailFunctionOutput`。

    Raises:
        ValueError: `on_trip` が "reject" / "raise" / "allow" 以外の文字列の場合。
    """
    if on_trip == "reject":
        return ToolGuardrailFunctionOutput.reject_content(message=message, output_info=info)
    if on_trip == "raise":
        return ToolGuardrailFunctionOutput.raise_exception(output_info=info)
    if on_trip == "allow":
        return ToolGuardrailFunctionOutput.allow(output_info=info)
    raise ValueError(f"on_trip must be 'reject' / 'raise' / 'allow', got {on_trip!r}")


def build_tool_input_guardrail(
    name: str,
    detect: Callable[[str], Detection],
    *,
    on_trip: OnTrip | Callable[[Detection], ToolGuardrailFunctionOutput] = "reject",
) -> ToolInputGuardrail[Any]:
    """plain 検知をツール入力（引数）guardrail へ接着する（SDK ネイティブ tool guardrail）。

    SDK の `(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput` を満たす関数を
    `ToolInputGuardrail` へ包む。ツール引数 JSON（`data.context.tool_arguments`）をテキスト化し
    `detect` で検査する。常に async な guardrail 関数で包み、`detect` の戻り値が awaitable なら
    await して正規化する（`_call_detect`）。これにより同期検知・`async def`・`async __call__` を
    持つ DI オブジェクトのいずれも一様に扱える（SDK は async な tool guardrail 関数を受け付ける）。
    trip 時の挙動は `on_trip`（"reject" 既定 / "raise" / "allow" または `Detection` を受けて
    `ToolGuardrailFunctionOutput` を返す callable）で選ぶ。

    Args:
        name: guardrail 名。
        detect: テキスト（ツール引数）を受けて `Detection`（同期）または `Awaitable[Detection]`
            （非同期）を返す検知関数。
        on_trip: trip 時挙動の選択（文字列 or callable DI）。

    Returns:
        SDK 互換 `ToolInputGuardrail`。
    """

    async def _guardrail(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        return _tool_output_for(await _call_detect(detect, _tool_input_text(data)), on_trip)

    return ToolInputGuardrail(guardrail_function=_guardrail, name=name)


def build_tool_output_guardrail(
    name: str,
    detect: Callable[[str], Detection],
    *,
    on_trip: OnTrip | Callable[[Detection], ToolGuardrailFunctionOutput] = "reject",
) -> ToolOutputGuardrail[Any]:
    """plain 検知をツール出力 guardrail へ接着する（SDK ネイティブ tool guardrail）。

    SDK の `(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput` を満たす関数を
    `ToolOutputGuardrail` へ包む。中間ツール出力（`data.output`）をテキスト化し `detect` で
    検査する。常に async な guardrail 関数で包み、`detect` の戻り値が awaitable なら await して
    正規化する（`_call_detect`・`build_tool_input_guardrail` と同じ方針）。trip 時の挙動は
    `on_trip` で選ぶ（既定 "reject" = 注釈付き返却で続行）。

    Args:
        name: guardrail 名。
        detect: テキスト（ツール出力）を受けて `Detection`（同期）または `Awaitable[Detection]`
            （非同期）を返す検知関数。
        on_trip: trip 時挙動の選択（文字列 or callable DI）。

    Returns:
        SDK 互換 `ToolOutputGuardrail`。
    """

    async def _guardrail(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
        return _tool_output_for(await _call_detect(detect, _text_of(data.output)), on_trip)

    return ToolOutputGuardrail(guardrail_function=_guardrail, name=name)


def _resolve_trip(
    on_trip: OnTrip | Callable[[Detection], ToolGuardrailFunctionOutput],
    detection: Detection,
) -> ToolGuardrailFunctionOutput:
    """trip 時挙動の選択（文字列定数 or callable DI）を `ToolGuardrailFunctionOutput` へ解決する。

    callable のときはそのまま呼び出して委ね、文字列のときは `_trip_output` で組む（reject 時の
    注釈メッセージは検知理由を使う）。

    Args:
        on_trip: trip 時挙動の選択。
        detection: 検知結果（trip 済み）。

    Returns:
        解決した `ToolGuardrailFunctionOutput`。
    """
    if callable(on_trip):
        return on_trip(detection)
    message = detection.reason or "guardrail tripped"
    return _trip_output(on_trip, message, {"reason": detection.reason, "info": detection.info})


def _tool_output_for(
    detection: Detection,
    on_trip: OnTrip | Callable[[Detection], ToolGuardrailFunctionOutput],
) -> ToolGuardrailFunctionOutput:
    """検知結果からツール guardrail 出力を組む（非 trip は allow・trip は `on_trip` 解決）。

    同期 / 非同期どちらの tool guardrail 経路からも共通で使う写像（検知済み `Detection` → SDK
    出力）。trip していなければ通過（allow）、trip していれば `on_trip` を解決する。

    Args:
        detection: 検知結果。
        on_trip: trip 時挙動の選択（文字列 or callable DI）。

    Returns:
        SDK 互換の `ToolGuardrailFunctionOutput`。
    """
    if not detection.triggered:
        return ToolGuardrailFunctionOutput.allow()
    return _resolve_trip(on_trip, detection)


def _tool_input_text(data: ToolInputGuardrailData) -> str:
    """ツール入力 guardrail データからツール引数テキストを取り出す。

    ツール引数は `data.context.tool_arguments`（JSON 文字列）に入る（通常は常に存在する）。
    `getattr` は SDK 版差異 / 将来互換のための防御的フォールバックで、無ければ空文字を返す。

    Args:
        data: SDK のツール入力 guardrail データ。

    Returns:
        ツール引数テキスト。
    """
    arguments = getattr(data.context, "tool_arguments", None)
    return "" if arguments is None else str(arguments)


def attach_tool_guardrails(
    tool: FunctionTool,
    *,
    input: ToolInputGuardrail[Any] | None = None,  # noqa: A002
    output: ToolOutputGuardrail[Any] | None = None,
) -> FunctionTool:
    """`FunctionTool` へ tool guardrail を装着した新しい `FunctionTool` を返す。

    `dataclasses.replace` で `tool_input_guardrails` / `tool_output_guardrails` を差し替える
    （`mock_spec_tools` と同型の SDK 型操作）。`name` / `description` / `params_json_schema` /
    `needs_approval` / `on_invoke_tool` は維持する（実行本体・宣言メタは変えない）。既存の
    guardrails が tool に付いている場合は連結する。`input` / `output` が共に None なら元 tool を
    そのまま返す。

    Args:
        tool: 装着対象の `FunctionTool`。
        input: 装着するツール入力 guardrail（任意）。
        output: 装着するツール出力 guardrail（任意）。

    Returns:
        guardrail 装着済みの新しい `FunctionTool`（元 tool は不変）。装着なしなら元 tool。
    """
    if input is None and output is None:
        return tool
    changes: dict[str, Any] = {}
    if input is not None:
        changes["tool_input_guardrails"] = [*(tool.tool_input_guardrails or []), input]
    if output is not None:
        changes["tool_output_guardrails"] = [*(tool.tool_output_guardrails or []), output]
    return _dataclass_replace(tool, **changes)


def guardrail_visible_name(guardrail: Any) -> str:
    """guardrail の上流 SDK 可視名（トレース等に表示される名前）を返す（NFR-1）。

    上流 4 型はいずれも `get_name()` を持ち、生成時に渡した `name` があればそれを、無ければ
    guardrail 関数名を返す。登録簿はこの値を登録キーとの照合に用いるため、可視名の取得経路を
    本関数 1 箇所へ寄せて `agents` 型の参照を `_adapters` に閉じる。

    Args:
        guardrail: 上流 SDK 互換の guardrail 実体。

    Returns:
        `get_name()` の戻り値を `str()` で文字列化した値。

    Raises:
        AttributeError: `get_name()` を持たない実体を渡した場合、または上流 4 型の
            インスタンスでも `name` が未設定で `guardrail_function` が `__name__` を持たない
            場合（`functools.partial` / `__call__` を持つオブジェクト等）。上流 4 型の
            `get_name()` は `name or guardrail_function.__name__` 相当のため、境界判定
            （`guardrail_boundary`）を通った実体でも後者は起こりうる。
    """
    return str(guardrail.get_name())


def guardrail_boundary(guardrail: Any) -> str | None:
    """guardrail の実体型から適用境界の文字列を判定する（NFR-1）。

    上流 4 型は相互に継承関係を持たない別型として分離されており、`isinstance` で一意に
    判別できる。宣言された境界との突合や、上流 guardrail 型でない実体の拒否に用いる。

    Args:
        guardrail: 判定対象。上流 SDK 互換の guardrail 実体を想定する。

    Returns:
        `"input"` / `"output"` / `"tool_input"` / `"tool_output"` のいずれか。上流 4 型の
        いずれのインスタンスでもない場合は `None`（境界を推測せず、呼び出し側が拒否できる形
        にする）。
    """
    if isinstance(guardrail, InputGuardrail):
        return "input"
    if isinstance(guardrail, OutputGuardrail):
        return "output"
    if isinstance(guardrail, ToolInputGuardrail):
        return "tool_input"
    if isinstance(guardrail, ToolOutputGuardrail):
        return "tool_output"
    return None


async def run_judge_prompt(model: Any, prompt: str, content: str) -> str:
    """判定 model を SDK Runner 経由で 1 ターン実行し判定テキストを返す（prompt 駆動 LLM・NFR-1）。

    `judge.py` を流用せず本モジュール専用の薄い実行ヘルパとして閉じる（責務分離）。判定 prompt
    本文と検査対象 content を結合した最小エージェントを 1 ターン走らせ、最終出力テキストを返す
    （trip 判定は上位 factories 側がテキストから決定する）。LLM 呼び出しを `_adapters` 経由へ
    寄せ外部直叩きを避ける（プロンプト / モデルは利用者 DI で非同梱）。

    Args:
        model: 判定に使う LLM（SDK `Model` / モデル名文字列 等の不透明値・利用者 DI）。
        prompt: 判定 prompt 本文（利用者提供・lib 非同梱）。
        content: 検査対象テキスト（入力 / 出力）。

    Returns:
        判定モデルの最終出力テキスト（空なら ""）。
    """
    agent = Agent(name="oai-agentspec-guardrail-judge", instructions=prompt, model=model)
    result = await Runner.run(agent, content)
    output = result.final_output
    return "" if output is None else str(output)
