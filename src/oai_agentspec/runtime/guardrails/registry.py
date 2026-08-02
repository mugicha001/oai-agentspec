"""宣言的ガードレール登録簿（宣言の保持 + 生成 facade・値域検証と境界正規化に徹する）。

`GuardrailSpec` を名前キーで保持する `GuardrailRegistry` を提供する。登録は 2 経路ある。
(1) 利用者が組んだ実体を `register()` で受け取る経路（実体が上流 4 型か・可視名が登録キーと
一致するか・宣言境界と実体境界が一致するかまで突き合わせる）、(2) `factories` の helper へ
代理呼び出しして生成と登録を 1 呼び出しで済ませる facade 9 経路（`HELPER_DEFAULTS` にキーを
持つ helper では labels / severity を自動付与する）。照会は `get` / `names` / `metadata` /
`boundary_of` / `specs` と、`RunConfig(**kwargs)` へそのまま展開できる `run_config_kwargs()`。

属性アクセス（`registry.<name>`）は提供せず、参照はメソッド経由に限る（guardrail 名に識別子
制約を課さないため）。

SDK 型の判定（実体の境界・可視名）は `_adapters/guardrails.py` の 2 関数へ委譲し、本層は
`agents` を一切 import しない（NFR-1）。実行時に不要な SDK 型注釈は `TYPE_CHECKING` に閉じる。
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..._adapters.guardrails import guardrail_boundary, guardrail_visible_name
from . import factories
from .catalog import HELPER_DEFAULTS
from .types import Boundary, GuardrailSpec, Severity

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Sequence

    # SDK 型は `_adapters.guardrails` 経由で参照する（SDK パッケージを直接 import せず SDK
    # 隔離を保つ・facade シグネチャの注釈用。実行時は不要なので TYPE_CHECKING に閉じる）。
    from ..._adapters.guardrails import OnTrip
    from ._detectors import Detection

#: `run_config_kwargs()` が返すキー（境界別・空でも欠落させない）。
_INPUT_KEY = "input_guardrails"
_OUTPUT_KEY = "output_guardrails"


class GuardrailRegistry:
    """ガードレール宣言の登録簿（単一スレッド前提・並行制御は利用者責任）。

    `register()` は利用者が組んだ実体つき宣言を検証して保持し、facade 9 メソッドは対応する
    `factories` の helper へ代理呼び出しして生成 + 登録を 1 呼び出しで済ませる。いずれの経路でも
    登録キーは実体の上流 SDK 可視名と一致する（照合キーと可視名の食い違いによる silent no-op を
    構造的に排除する）。保持する `boundary` は `Boundary` メンバへ正規化する。
    """

    def __init__(self) -> None:
        """空の登録簿を生成する。"""
        self._specs: dict[str, GuardrailSpec] = {}

    # ------------------------------------------------------------------
    # 登録（利用者が組んだ実体を受け取る経路）
    # ------------------------------------------------------------------
    def register(self, spec: GuardrailSpec) -> GuardrailSpec:
        """実体つきの `GuardrailSpec` を登録する（labels / severity の自動付与はしない）。

        検証は「name 値域 → boundary 値域 → severity 値域 → guardrail 非 None → 重複名 →
        実体の整合」の順に行い、どの段で落ちても登録簿の状態は変えない。実体の整合は上流 4 型か
        （`guardrail_boundary`）・可視名が登録キーと一致するか（`guardrail_visible_name`）・
        実体境界が宣言 `boundary` と一致するかの 3 点で、ツール境界 2 型も受理する。

        Args:
            spec: 登録する宣言。`boundary` は `Boundary` メンバまたは値域内の文字列。

        Returns:
            登録された宣言（`boundary` を `Boundary` メンバへ正規化したもの）。`metadata()` が
            返すインスタンスと同一。

        Raises:
            ValueError: name / boundary / severity が値域外、`guardrail` が None、登録名の重複、
                実体の整合（上流 4 型・可視名・境界）が取れない場合、または実体から可視名を
                取得できない場合（`name` 未設定 + `guardrail_function` に `__name__` なし）。
        """
        self._validate_name(spec.name)
        boundary = self._coerce_boundary(spec.boundary)
        self._validate_severity(spec.severity, field="severity")
        if spec.guardrail is None:
            raise ValueError(f"guardrail entity is required (got None): {spec.name!r}")
        self._reject_duplicate(spec.name)
        actual = guardrail_boundary(spec.guardrail)
        if actual is None:
            raise ValueError(
                f"guardrail entity is not a supported SDK guardrail type: {spec.name!r}"
            )
        try:
            visible = guardrail_visible_name(spec.guardrail)
        except AttributeError as exc:
            # 上流 4 型は `name=None` を許し `get_name()` は `guardrail_function.__name__` へ
            # フォールバックする。`functools.partial` / `__call__` オブジェクトでは属性が無く
            # `AttributeError` になるため、登録時の検証は必ず `ValueError` という契約へ包む。
            raise ValueError(
                f"guardrail visible name is unavailable: {spec.name!r} "
                f"（name 未設定で guardrail_function に __name__ がありません: {exc}）"
            ) from exc
        if visible != spec.name:
            raise ValueError(
                f"guardrail visible name does not match the registration key: "
                f"{spec.name!r} != {visible!r}"
            )
        if actual != boundary.value:
            raise ValueError(
                f"declared boundary does not match the guardrail entity: {spec.name!r} "
                f"declared={boundary.value}, actual={actual}"
            )
        normalized = spec if spec.boundary is boundary else replace(spec, boundary=boundary)
        self._specs[spec.name] = normalized
        return normalized

    # ------------------------------------------------------------------
    # 照会
    # ------------------------------------------------------------------
    def get(self, name: str) -> Any:
        """登録された guardrail 実体を返す（不透明型のまま返し構造を解釈しない）。

        Args:
            name: 登録名。

        Returns:
            登録時の guardrail 実体（copy ではない）。

        Raises:
            KeyError: `name` が未登録の場合（文言は `_unknown_guardrail_message` で単一ソース化）。
        """
        return self._require(name).guardrail

    def names(self, *, boundary: Boundary | str | None = None) -> list[str]:
        """登録名を昇順で返す（`boundary` 指定時はその境界のみ）。

        Args:
            boundary: 絞り込む境界（`Boundary` メンバまたは値域内の文字列・None で全件）。

        Returns:
            登録名の昇順リスト（該当なしは空リスト）。

        Raises:
            ValueError: `boundary` が値域外の場合（無言の空振りを返さない）。
        """
        return [spec.name for spec in self.specs(boundary=boundary)]

    def metadata(self, name: str) -> GuardrailSpec:
        """登録された宣言（`GuardrailSpec`）を返す。

        Args:
            name: 登録名。

        Returns:
            登録された宣言そのもの（`boundary` は `Boundary` メンバへ正規化済み）。

        Raises:
            KeyError: `name` が未登録の場合。
        """
        return self._require(name)

    def boundary_of(self, name: str) -> Boundary:
        """登録された宣言の適用境界を返す（`Boundary` は `str` 併用のため文字列比較可）。

        Args:
            name: 登録名。

        Returns:
            適用境界（`Boundary` メンバ。`== "output"` のように素の文字列と比較できる）。

        Raises:
            KeyError: `name` が未登録の場合。
        """
        return Boundary(self._require(name).boundary)

    def specs(
        self,
        *,
        boundary: Boundary | str | None = None,
        min_severity: Severity | None = None,
    ) -> list[GuardrailSpec]:
        """登録された宣言を名前昇順で返す（両フィルタは AND 条件）。

        `min_severity` 指定時は `severity is None` の宣言を例外にせず除外する（未分類の宣言を
        閾値照会の対象外として扱う）。

        Args:
            boundary: 絞り込む境界（`Boundary` メンバまたは値域内の文字列・None でフィルタなし）。
            min_severity: 下限の深刻度（`Severity` メンバ・None でフィルタなし）。

        Returns:
            条件を満たす宣言の名前昇順リスト（毎回新しい list。要素は `metadata()` と同一
            インスタンス）。

        Raises:
            ValueError: `boundary` / `min_severity` が値域外の場合。
        """
        wanted = None if boundary is None else self._coerce_boundary(boundary)
        self._validate_severity(min_severity, field="min_severity")
        selected: list[GuardrailSpec] = []
        for name in sorted(self._specs):
            spec = self._specs[name]
            if wanted is not None and Boundary(spec.boundary) is not wanted:
                continue
            if min_severity is not None and (spec.severity is None or spec.severity < min_severity):
                continue
            selected.append(spec)
        return selected

    def run_config_kwargs(self, names: Sequence[str] | None = None) -> dict[str, list[Any]]:
        """agent 境界 guardrail を `RunConfig(**kwargs)` へ展開できる形へ束ねる。

        検証は「要素の型 → 名前の解決 → 境界の受け入れ可否」の順に、それぞれ全要素について
        一巡させる（複数の違反が混在しても例外型が入力順に依存せず決定的になる）。ツール境界の
        登録は `RunConfig` へ渡せないため、`names=None`（登録全件）でも静かに除外せず
        `ValueError` を上げる。

        Args:
            names: 対象の登録名（宣言順を保持し重複も排除しない）。None で登録全件を名前昇順。

        Returns:
            `{"input_guardrails": [...], "output_guardrails": [...]}`（対象が空でも 2 キーを
            欠落させない。毎回新しい dict / list で、要素は `get()` と同一インスタンス）。

        Raises:
            ValueError: 要素が str でない場合、または対象にツール境界の登録が含まれる場合。
            KeyError: 未登録の名前が含まれる場合。
        """
        if isinstance(names, str):
            # `str` は `Sequence[str]` を満たすため注釈では防げない。1 文字ずつ分解されると
            # 空文字で「guardrail 0 件の RunConfig」が静かに生成される（fail-open）ので弾く。
            raise ValueError(f"names must be a sequence of str, not a bare str: {names!r}")
        selected = self.names() if names is None else list(names)
        for item in selected:
            if not isinstance(item, str):
                raise ValueError(f"guardrail name must be a str: {item!r}")
        for item in selected:
            self._require(item)
        for item in selected:
            boundary = Boundary(self._specs[item].boundary)
            if boundary not in (Boundary.INPUT, Boundary.OUTPUT):
                raise ValueError(
                    f"guardrail is not an agent boundary guardrail and cannot be passed to "
                    f"RunConfig: {item!r} (boundary={boundary.value})"
                )
        kwargs: dict[str, list[Any]] = {_INPUT_KEY: [], _OUTPUT_KEY: []}
        for item in selected:
            spec = self._specs[item]
            key = _INPUT_KEY if Boundary(spec.boundary) is Boundary.INPUT else _OUTPUT_KEY
            kwargs[key].append(spec.guardrail)
        return kwargs

    # ------------------------------------------------------------------
    # 生成 facade（agent 境界 8 + ツール境界 1・`factories` へ代理呼び出し）
    # ------------------------------------------------------------------
    def prompt_llm_guardrail(
        self,
        model: Any,
        prompt: str,
        *,
        on: str,
        verdict: Callable[[str], Detection] | None = None,
        name: str,
        run_in_parallel: bool = True,
        labels: dict[str, Any] | None = None,
        severity: Severity | None = None,
    ) -> GuardrailSpec:
        """`factories.prompt_llm_guardrail` へ代理呼び出しして生成 + 登録する。

        境界は `on` から導出する（`Boundary(on)`）。DI 依存 helper のため labels / severity の
        自動付与はなく、宣言された値のみ保持する。

        Args:
            model: 判定に使う LLM（利用者 DI）。
            prompt: 判定 prompt 本文（利用者提供）。
            on: 適用境界（"input" or "output"）。
            verdict: judge 出力テキスト → `Detection` の解釈関数（None で既定）。
            name: 登録名（実体の可視名にも注入する・キーワード必須）。
            run_in_parallel: 入力検査を並行に走らせるか（`on="input"` のときのみ効く）。
            labels: 任意のラベル。
            severity: 深刻度（`Severity` メンバまたは None）。

        Returns:
            登録された宣言（`metadata()` と同一インスタンス）。

        Raises:
            ValueError: name / severity が値域外、登録名の重複、または factory の引数が不正な場合
                （factory 由来の例外は伝播し、登録は増えない）。
        """
        self._validate_declaration(name, severity)
        guardrail = factories.prompt_llm_guardrail(
            model,
            prompt,
            on=on,
            verdict=verdict,
            name=name,
            run_in_parallel=run_in_parallel,
        )
        return self._register_generated(
            "prompt_llm_guardrail", name, Boundary(on), guardrail, labels, severity
        )

    def canary_guardrail(
        self,
        canary: str | Iterable[str],
        *,
        name: str,
        labels: dict[str, Any] | None = None,
        severity: Severity | None = None,
    ) -> GuardrailSpec:
        """`factories.canary_guardrail` へ代理呼び出しして生成 + 登録する。

        境界は helper 自体で OUTPUT に固定される。`HELPER_DEFAULTS` にキーを持つため labels /
        severity の既定値が自動付与される（同一キーの labels と `Severity` メンバの明示は
        利用者宣言が優先）。

        Args:
            canary: 照合する canary 値（単一 or 複数）。
            name: 登録名（実体の可視名にも注入する・キーワード必須）。
            labels: 任意のラベル（既定 labels とキー単位でマージする）。
            severity: 深刻度（`Severity` メンバまたは None。None で既定を付与）。

        Returns:
            登録された宣言（`metadata()` と同一インスタンス）。

        Raises:
            ValueError: name / severity が値域外、または登録名が重複する場合。
        """
        self._validate_declaration(name, severity)
        guardrail = factories.canary_guardrail(canary, name=name)
        return self._register_generated(
            "canary_guardrail", name, Boundary.OUTPUT, guardrail, labels, severity
        )

    def predicate_guardrail(
        self,
        predicate: Callable[[str], bool | Awaitable[bool]],
        *,
        on: str,
        reason: str | None = None,
        name: str,
        run_in_parallel: bool = True,
        labels: dict[str, Any] | None = None,
        severity: Severity | None = None,
    ) -> GuardrailSpec:
        """`factories.predicate_guardrail` へ代理呼び出しして生成 + 登録する。

        境界は `on` から導出する（`Boundary(on)`）。DI 依存 helper のため labels / severity の
        自動付与はない。

        Args:
            predicate: 検知有無を返す述語（同期 / 非同期どちらも可・利用者 DI）。
            on: 適用境界（"input" or "output"）。
            reason: 検知時の理由（任意）。
            name: 登録名（実体の可視名にも注入する・キーワード必須）。
            run_in_parallel: 入力検査を並行に走らせるか（`on="input"` のときのみ効く）。
            labels: 任意のラベル。
            severity: 深刻度（`Severity` メンバまたは None）。

        Returns:
            登録された宣言（`metadata()` と同一インスタンス）。

        Raises:
            ValueError: name / severity が値域外、登録名の重複、または factory の引数が不正な場合。
        """
        self._validate_declaration(name, severity)
        guardrail = factories.predicate_guardrail(
            predicate, on=on, reason=reason, name=name, run_in_parallel=run_in_parallel
        )
        return self._register_generated(
            "predicate_guardrail", name, Boundary(on), guardrail, labels, severity
        )

    def regex_guardrail(
        self,
        patterns: str | Iterable[str],
        *,
        on: str,
        flags: int = 0,
        name: str,
        run_in_parallel: bool = True,
        labels: dict[str, Any] | None = None,
        severity: Severity | None = None,
    ) -> GuardrailSpec:
        """`factories.regex_guardrail` へ代理呼び出しして生成 + 登録する。

        境界は `on` から導出する（`Boundary(on)`）。DI 依存 helper のため labels / severity の
        自動付与はない。

        Args:
            patterns: 検知に使う正規表現（単一 or 複数・利用者 DI）。
            on: 適用境界（"input" or "output"）。
            flags: `re.compile` フラグ（既定 0）。
            name: 登録名（実体の可視名にも注入する・キーワード必須）。
            run_in_parallel: 入力検査を並行に走らせるか（`on="input"` のときのみ効く）。
            labels: 任意のラベル。
            severity: 深刻度（`Severity` メンバまたは None）。

        Returns:
            登録された宣言（`metadata()` と同一インスタンス）。

        Raises:
            ValueError: name / severity が値域外、登録名の重複、または factory の引数が不正な場合。
        """
        self._validate_declaration(name, severity)
        guardrail = factories.regex_guardrail(
            patterns, on=on, flags=flags, name=name, run_in_parallel=run_in_parallel
        )
        return self._register_generated(
            "regex_guardrail", name, Boundary(on), guardrail, labels, severity
        )

    def length_guardrail(
        self,
        *,
        max_length: int | None = None,
        min_length: int | None = None,
        on: str,
        name: str,
        run_in_parallel: bool = True,
        labels: dict[str, Any] | None = None,
        severity: Severity | None = None,
    ) -> GuardrailSpec:
        """`factories.length_guardrail` へ代理呼び出しして生成 + 登録する。

        境界は `on` から導出する（`Boundary(on)`）。DI 依存 helper のため labels / severity の
        自動付与はない。

        Args:
            max_length: 上限文字数（超過で trip）。None で上限なし。
            min_length: 下限文字数（未満で trip）。None で下限なし。
            on: 適用境界（"input" or "output"）。
            name: 登録名（実体の可視名にも注入する・キーワード必須）。
            run_in_parallel: 入力検査を並行に走らせるか（`on="input"` のときのみ効く）。
            labels: 任意のラベル。
            severity: 深刻度（`Severity` メンバまたは None）。

        Returns:
            登録された宣言（`metadata()` と同一インスタンス）。

        Raises:
            ValueError: name / severity が値域外、登録名の重複、または factory の引数が不正な場合
                （閾値が両方 None のときを含む）。
        """
        self._validate_declaration(name, severity)
        guardrail = factories.length_guardrail(
            max_length=max_length,
            min_length=min_length,
            on=on,
            name=name,
            run_in_parallel=run_in_parallel,
        )
        return self._register_generated(
            "length_guardrail", name, Boundary(on), guardrail, labels, severity
        )

    def allow_deny_guardrail(
        self,
        *,
        deny: Iterable[str] | None = None,
        allow: Iterable[str] | None = None,
        case_sensitive: bool = True,
        on: str,
        name: str,
        run_in_parallel: bool = True,
        labels: dict[str, Any] | None = None,
        severity: Severity | None = None,
    ) -> GuardrailSpec:
        """`factories.allow_deny_guardrail` へ代理呼び出しして生成 + 登録する。

        境界は `on` から導出する（`Boundary(on)`）。DI 依存 helper のため labels / severity の
        自動付与はない。

        Args:
            deny: 含まれていたら trip する拒否語（任意・利用者 DI）。
            allow: いずれも含まれなければ trip する許可語（任意・利用者 DI）。
            case_sensitive: 大文字小文字を区別するか（既定 True）。
            on: 適用境界（"input" or "output"）。
            name: 登録名（実体の可視名にも注入する・キーワード必須）。
            run_in_parallel: 入力検査を並行に走らせるか（`on="input"` のときのみ効く）。
            labels: 任意のラベル。
            severity: 深刻度（`Severity` メンバまたは None）。

        Returns:
            登録された宣言（`metadata()` と同一インスタンス）。

        Raises:
            ValueError: name / severity が値域外、登録名の重複、または factory の引数が不正な場合。
        """
        self._validate_declaration(name, severity)
        guardrail = factories.allow_deny_guardrail(
            deny=deny,
            allow=allow,
            case_sensitive=case_sensitive,
            on=on,
            name=name,
            run_in_parallel=run_in_parallel,
        )
        return self._register_generated(
            "allow_deny_guardrail", name, Boundary(on), guardrail, labels, severity
        )

    def injection_baseline_guardrail(
        self,
        extra_patterns: Iterable[str] | None = None,
        *,
        name: str,
        run_in_parallel: bool = True,
        labels: dict[str, Any] | None = None,
        severity: Severity | None = None,
    ) -> GuardrailSpec:
        """`factories.injection_baseline_guardrail` へ代理呼び出しして生成 + 登録する。

        境界は helper 自体で INPUT に固定される。`HELPER_DEFAULTS` にキーを持つため labels /
        severity の既定値が自動付与される（同一キーの labels と `Severity` メンバの明示は
        利用者宣言が優先）。

        Args:
            extra_patterns: 既定に追記する正規表現パターン（任意・利用者 DI）。
            name: 登録名（実体の可視名にも注入する・キーワード必須）。
            run_in_parallel: 入力検査を並行に走らせるか（既定 True・SDK 既定）。
            labels: 任意のラベル（既定 labels とキー単位でマージする）。
            severity: 深刻度（`Severity` メンバまたは None。None で既定を付与）。

        Returns:
            登録された宣言（`metadata()` と同一インスタンス）。

        Raises:
            ValueError: name / severity が値域外、または登録名が重複する場合。
        """
        self._validate_declaration(name, severity)
        guardrail = factories.injection_baseline_guardrail(
            extra_patterns, name=name, run_in_parallel=run_in_parallel
        )
        return self._register_generated(
            "injection_baseline_guardrail", name, Boundary.INPUT, guardrail, labels, severity
        )

    def external_detector_guardrail(
        self,
        detect: Callable[[str], Detection | Awaitable[Detection]],
        *,
        on: str,
        name: str,
        run_in_parallel: bool = True,
        labels: dict[str, Any] | None = None,
        severity: Severity | None = None,
    ) -> GuardrailSpec:
        """`factories.external_detector_guardrail` へ代理呼び出しして生成 + 登録する。

        境界は `on` から導出する（`Boundary(on)`）。DI 依存 helper のため labels / severity の
        自動付与はない。

        Args:
            detect: `Detection`（同期）または `Awaitable[Detection]`（非同期）を返す利用者検知器。
            on: 適用境界（"input" or "output"）。
            name: 登録名（実体の可視名にも注入する・キーワード必須）。
            run_in_parallel: 入力検査を並行に走らせるか（`on="input"` のときのみ効く）。
            labels: 任意のラベル。
            severity: 深刻度（`Severity` メンバまたは None）。

        Returns:
            登録された宣言（`metadata()` と同一インスタンス）。

        Raises:
            ValueError: name / severity が値域外、登録名の重複、または factory の引数が不正な場合。
        """
        self._validate_declaration(name, severity)
        guardrail = factories.external_detector_guardrail(
            detect, on=on, name=name, run_in_parallel=run_in_parallel
        )
        return self._register_generated(
            "external_detector_guardrail", name, Boundary(on), guardrail, labels, severity
        )

    def tool_guardrail(
        self,
        detector: Callable[[str], Detection | Awaitable[Detection]],
        *,
        on: str,
        on_trip: OnTrip | Callable[[Detection], Any] = "reject",
        name: str,
        labels: dict[str, Any] | None = None,
        severity: Severity | None = None,
    ) -> GuardrailSpec:
        """`factories.tool_guardrail` へ代理呼び出しして生成 + 登録する。

        境界は `on` からツール境界へ導出する（`on="input"` で TOOL_INPUT、`on="output"` で
        TOOL_OUTPUT）。DI 依存 helper のため labels / severity の自動付与はない。

        Args:
            detector: `Detection`（同期）または `Awaitable[Detection]`（非同期）を返す検知器。
            on: 適用境界（"input" or "output"）。
            on_trip: trip 時挙動の選択（文字列 or callable DI・既定 "reject"）。
            name: 登録名（実体の可視名にも注入する・キーワード必須）。
            labels: 任意のラベル。
            severity: 深刻度（`Severity` メンバまたは None）。

        Returns:
            登録された宣言（`metadata()` と同一インスタンス）。

        Raises:
            ValueError: name / severity が値域外、登録名の重複、または factory の引数が不正な場合。
        """
        self._validate_declaration(name, severity)
        guardrail = factories.tool_guardrail(detector, on=on, on_trip=on_trip, name=name)
        boundary = Boundary.TOOL_INPUT if on == "input" else Boundary.TOOL_OUTPUT
        return self._register_generated(
            "tool_guardrail", name, boundary, guardrail, labels, severity
        )

    # ------------------------------------------------------------------
    # 内部ヘルパ
    # ------------------------------------------------------------------
    def _validate_declaration(self, name: str, severity: Severity | None) -> None:
        """facade 経路の事前検証（name 値域 → severity 値域 → 重複名）をまとめて行う。

        factory 呼び出しより前に済ませることで、生成に失敗しても名前だけが登録された部分適用が
        残らないようにする。

        Args:
            name: 登録名。
            severity: 深刻度の指定（`Severity` メンバまたは None）。

        Raises:
            ValueError: name / severity が値域外、または登録名が重複する場合。
        """
        self._validate_name(name)
        self._validate_severity(severity, field="severity")
        self._reject_duplicate(name)

    def _register_generated(
        self,
        helper: str,
        name: str,
        boundary: Boundary,
        guardrail: Any,
        labels: dict[str, Any] | None,
        severity: Severity | None,
    ) -> GuardrailSpec:
        """生成済み実体から宣言を組んで登録する（labels / severity の既定を自動付与する）。

        `HELPER_DEFAULTS` にキーを持つ helper では既定 labels を土台に利用者 labels をキー単位で
        マージし（同一キーは利用者宣言が優先）、severity 未指定なら既定を付与する。labels は毎回
        新しい dict を作り、`HELPER_DEFAULTS` の Mapping を共有しない。

        Args:
            helper: 代理呼び出しした helper 名（`HELPER_DEFAULTS` のキーに使う）。
            name: 登録名。
            boundary: 導出済みの適用境界。
            guardrail: factory が返した guardrail 実体。
            labels: 利用者が宣言したラベル（None で宣言なし）。
            severity: 利用者が宣言した深刻度（None で宣言なし）。

        Returns:
            登録された宣言。
        """
        defaults = HELPER_DEFAULTS.get(helper)
        merged: dict[str, Any] = dict(defaults.labels) if defaults is not None else {}
        if labels is not None:
            merged.update(labels)
        resolved = severity
        if resolved is None and defaults is not None:
            resolved = defaults.severity
        spec = GuardrailSpec(
            name=name, boundary=boundary, guardrail=guardrail, labels=merged, severity=resolved
        )
        self._specs[name] = spec
        return spec

    def _require(self, name: str) -> GuardrailSpec:
        """登録済み宣言を取得する（未登録は名前を含む `KeyError`）。

        Args:
            name: 登録名。

        Returns:
            登録された宣言。

        Raises:
            KeyError: `name` が未登録の場合。
        """
        if name not in self._specs:
            raise KeyError(self._unknown_guardrail_message(name))
        return self._specs[name]

    def _reject_duplicate(self, name: str) -> None:
        """登録名の重複を拒否する（文言に `already` を含め後段の検証と区別できるようにする）。

        Args:
            name: 登録名。

        Raises:
            ValueError: `name` が既登録の場合。
        """
        if name in self._specs:
            raise ValueError(f"guardrail already registered: {name!r}")

    @staticmethod
    def _validate_name(name: str) -> None:
        """登録名が非空白の `str` であることを検証する。

        属性アクセスを提供しないため識別子制約（`str.isidentifier()` 等）は課さない。

        Args:
            name: 検証対象の名前。

        Raises:
            ValueError: str でない、空文字、または空白文字のみの場合。
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"invalid guardrail name (must be a non-blank str): {name!r}")

    @staticmethod
    def _coerce_boundary(boundary: Boundary | str) -> Boundary:
        """境界指定を `Boundary` メンバへ正規化する（値域外は `ValueError`）。

        Args:
            boundary: `Boundary` メンバまたは値域内の文字列。

        Returns:
            対応する `Boundary` メンバ。

        Raises:
            ValueError: `Boundary` メンバでも値域内の文字列でもない場合。
        """
        try:
            return Boundary(boundary)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in Boundary)
            raise ValueError(
                f"invalid guardrail boundary: {boundary!r} (expected one of {allowed})"
            ) from exc

    @staticmethod
    def _validate_severity(severity: Severity | None, *, field: str) -> None:
        """深刻度が `Severity` メンバまたは None であることを検証する。

        `Severity` は `IntEnum` のため `isinstance(x, int)` では素の int を取り逃がす。メンバ性
        （`isinstance(x, Severity)`）で判定し、素の int（`3` 等）も拒否する。

        Args:
            severity: 検証対象の値。
            field: 例外文言に埋め込む引数名（`severity` / `min_severity`）。

        Raises:
            ValueError: `Severity` メンバでも None でもない場合。
        """
        if severity is not None and not isinstance(severity, Severity):
            raise ValueError(
                f"invalid guardrail {field}: {severity!r} (expected a Severity member or None)"
            )

    def _unknown_guardrail_message(self, name: str) -> str:
        """未登録名エラーの単一ソース文言を組み立てる（`get` / `metadata` 等で共有）。

        Args:
            name: 未登録の参照名。

        Returns:
            `unknown guardrail: <name>. registered guardrails: <一覧>` 形式の文字列。登録が
            空の場合は一覧を `(none)` と表示する。
        """
        registered = ", ".join(sorted(self._specs)) or "(none)"
        return f"unknown guardrail: {name!r}. registered guardrails: {registered}"


__all__ = ["GuardrailRegistry"]
