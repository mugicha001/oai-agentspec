"""Agent Lightning 最適化の plain 結果型・スロット型（外部 SDK 非依存）。

本モジュールは openai-agents（`agents`）・Agent Lightning（`agentlightning`）を一切 import
しない。`_adapters/lightning` が最適化エンジンの結果を本モジュールの plain dataclass へ変換し、
最適化ロジック層（`optimizer`）と公開窓口はこの plain 型のみを扱う（NFR-1）。

すべて `@dataclass(frozen=True)`（会話の `SendResult` / llmops の結果型と一致・Pydantic 非導入）。
`OptimizeResult.save` は利用者指定パスへの opt-in 書込のみで `PromptStore` / ライブラリ管理領域を
一切書き換えない（FR-9・PromptStore 非書込）。

型定義に加え、失敗メッセージ整形の共有ヘルパ `_format_exception_message`（依存ゼロの純関数）を
置く（optimizer / `_rollout` / `_adapters` の 3 箇所で使い、整形規則の drift を防ぐ）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from ..._validation import validate_bool

if TYPE_CHECKING:
    from collections.abc import Callable


class HistoryEntry(TypedDict):
    """`OptimizeResult.history` に詰まる 1 ラウンド分の plain dict schema（FR-2）。

    各スロットを順次最適化する `_run_apo_single_slot` が 1 件返す。`placeholder_fallback=True`
    のときは `best_score` / `best_version` を None にする（候補が `${var}` を喪失したため seed へ
    戻したことを示し、利用者の集計が「最適化が成功した best_score」と取り違えないため）。

    Attributes:
        slot: 当該ラウンドで最適化したスロット名。
        best_score: APO `Optimization.best_score`（fallback 時は None）。
        best_version: APO `Optimization.best_version`（fallback 時は None）。
        placeholder_fallback: APO 最良候補が `${var}` を喪失したため seed へフォールバックしたか。
    """

    slot: str
    best_score: float | None
    best_version: int | None
    placeholder_fallback: bool


class FailureKind(StrEnum):
    """最適化失敗の種別（FR-8・構造化エラーで判別可能にする）。

    `OptimizeError.kind` に載せ、利用者が失敗の種別ごとに分岐できるようにする。

    Attributes:
        EXTRA_MISSING: `[lightning]` extra（agentlightning）未導入。
        CONFIG_MISSING: 必須設定（algorithm / train / reward / slot・rebind / registry 等）不在。
        TRAINER_FAILED: 最適化実行（Trainer / rollout / reward）中の失敗。
    """

    EXTRA_MISSING = "extra_missing"
    CONFIG_MISSING = "config_missing"
    TRAINER_FAILED = "trainer_failed"


@dataclass(frozen=True)
class CoverageReport:
    """pre-flight route coverage 観測の集計（`OptimizeError.coverage` に添付・診断用・NFR-1）。

    Attributes:
        covered: 到達済み slot 名の union（frozenset・順序非依存）。到達観測は常に陽性証拠で、
            `complete` の値によらず「実際に routing された」ことを意味する。
        missing: 未到達 slot 名（`slots.keys() - covered`・frozenset）。**「train 全件を観測した
            結果、一度も routing されなかった」ことの確定であるのは
            `complete=True` かつ `invalid_cases == 0` のときに限る**。`complete=False` なら
            「未到達」と「まだ観測していない」の和、`invalid_cases > 0` なら無効化で観測
            できなかった case のぶん判定が欠けており、いずれも train の作り直し等の対処判断に
            直接使ってはならない。
        per_case: `(case, route_steps)` の tuple 列（train の全 case を順に含む）。
            `route_steps` は **3 値**: `None` = 候補無効化（rollout の観測が得られていない・
            `${var}` 喪失 / `vars=callable` の非 dict 戻り値 / 境界マーカー崩れ等）/
            `()` = 実行済みだが観測が空（防御的経路）/ 非空 tuple = 到達観測。case 要素は
            `RolloutResult.case` と型を揃えて `Any`（利用者任意型の多態性を保持）。case 本文の
            accidental dump を防ぐため `repr()` には含めない（`report.per_case` への明示
            アクセスは可）。
        interrupted_cases: `RunOutcome.interrupted=True` で途中打ち切りとなった case 数。
            診断カウンタであり coverage 判定には使わない（interrupted case で観測できた到達も
            `covered` に算入される）。
        complete: train 全件の**観測ループを完走したか**（既定 True）。False は観測が途中の
            例外（timeout / モデル API エラー等）で打ち切られた部分レポートであることを表す。
            **`complete=True` は `missing` の確定を単独では意味しない**（無効化 case があれば
            そのぶん判定が欠ける。確定条件は `complete and invalid_cases == 0`・上記参照）。
            `OptimizeError.kind` からも `CONFIG_MISSING`=完走 / `TRAINER_FAILED`=部分と
            推し量れるが、kind は失敗種別の軸であり観測完了度とは別関心。レポート単体を
            例外から切り離してログ・保存しても完了度が読めるよう、自己記述フィールドとして
            持つ（repr にも出る）。
        invalid_cases: 候補無効化により rollout の観測が得られなかった case 数（既定 0・
            `per_case` の `None` エントリ数と一致）。`interrupted_cases` と同じく診断カウンタで
            あり coverage 判定には使わない。0 より大きいとき `missing` は確定でない（上記）。
    """

    covered: frozenset[str]
    missing: frozenset[str]
    per_case: tuple[tuple[Any, tuple[str, ...] | None], ...] = field(repr=False)
    interrupted_cases: int
    complete: bool = True
    invalid_cases: int = 0

    def __post_init__(self) -> None:
        """`complete` が bool であることを構築時に検証する。

        Raises:
            ValueError: `complete` が bool でない場合。
        """
        validate_bool(self.complete, "complete")


def _format_exception_message(exc: BaseException) -> str:
    """例外を「型名は常に・本文は非空のときだけ」の形式で整形する（共有ヘルパ）。

    `str(TimeoutError())` のように本文が空になる例外型があり、無条件に `型名: 本文` を連結すると
    `'TimeoutError: '` とコロンで終わる情報ゼロの文字列になる。失敗メッセージを組む全境界
    （optimizer の catch-all / `_rollout` の pre-flight 観測 / `_adapters` の APO 実行）で
    本関数を使い、整形規則の drift を防ぐ。

    Args:
        exc: 整形対象の例外。

    Returns:
        `"型名: 本文"`（本文非空時）または `"型名"`（本文空時）。
    """
    body = str(exc)
    return f"{type(exc).__name__}: {body}" if body else type(exc).__name__


@dataclass(frozen=True)
class OptimizePartial:
    """APO 逐次実行が途中で失敗したときの部分成果（`OptimizeError.partial` に添付・診断用）。

    複数 slot の逐次 APO では、slot i の失敗時点で slot 1..i-1 の最適化は完了しており、
    その成果（API コストを払って得た最良テキストと履歴）を失敗とともに破棄しない。
    `except OptimizeError` 節から `error.partial` でプログラム的に取得できる。

    Attributes:
        completed_slots: 完了済み slot の最良テキスト（`{slot 名: テキスト}`・`${var}` 再注入
            済み）。**診断・救出用の中間表現**であり、新 shape slot では固定セグメントを含む
            full 合成（`OptimizeResult.prompt` の契約）ではない tune 側テキストのまま。
            そのまま instructions に使う値ではない。prompt 本文の accidental dump を防ぐため
            `repr()` には含めない（明示アクセスは可・`CoverageReport.per_case` と同方針）。
        history: 完了済み slot の履歴（`HistoryEntry` の列・`OptimizeResult.history` と同 schema。
            stdlib に frozen mapping が無く `OptimizeResult.history` と型を揃えるため list）。
        failed_slot: 失敗した slot 名。None は「全 slot の最適化は完了し、合成スコア再計算段で
            失敗した」ことを表す（このとき `completed_slots` は全 slot を含む）。
    """

    completed_slots: dict[str, str] = field(repr=False)
    history: list[HistoryEntry]
    failed_slot: str | None


class OptimizeError(Exception):
    """最適化が送出する構造化エラー（未捕捉例外でプロセスを止めないための変換先・FR-8）。

    extra 不在 / 設定不在 / Trainer 実行失敗を `kind` で判別できる明確なエラーに統一する。SDK /
    agentlightning の生例外を上位へ漏らさず、本型へ変換して送出する（NFR-1 と整合）。

    Attributes:
        kind: 失敗種別（`FailureKind`）。
        message: 人間可読のエラーメッセージ。
        coverage: pre-flight route coverage 診断（`CoverageReport`）。pre-flight の
            **未到達検出（`CONFIG_MISSING`・`complete=True`）と観測失敗（`TRAINER_FAILED`・
            `complete=False`）の両経路**で非 None。他の raise 経路では None（既存呼び出しには
            影響なし）。例外: pre-flight 中に承認安全違反（NFR-8 fail-closed）で送出される
            `CONFIG_MISSING` は既存の kind / message を保つため `coverage=None` のまま。
    """

    def __init__(
        self,
        kind: FailureKind,
        message: str,
        *,
        coverage: CoverageReport | None = None,
        partial: OptimizePartial | None = None,
    ) -> None:
        """最適化エラーを生成する。

        Args:
            kind: 失敗種別。
            message: 人間可読メッセージ。
            coverage: pre-flight route coverage の診断情報（keyword-only・pre-flight 経路のみ）。
            partial: APO 逐次実行の部分成果（keyword-only・複数 slot APO の途中失敗経路のみ。
                非 None は「保全された成果がある」ことを意味し、先頭 slot 失敗等の保全対象が
                ない失敗では None のまま）。

        Note:
            `coverage` / `partial` は keyword-only。位置引数で渡すと `TypeError`
            （既存呼び出し互換）。
        """
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.coverage = coverage
        self.partial = partial


class _CandidateInvalid(Exception):
    """rollout 候補を無効化するための内部シグナル例外（reward 0.0 経路へ倒す）。

    lightning ライブラリ内部で「この候補は破棄・reward 0.0」を伝えるための sentinel。
    利用者 `build=` 関数や利用者 `vars=callable` 関数が投げる無関係な `ValueError` を
    候補無効化に silent 吸収してしまわないよう、意味を明示した専用型で signal する
    （C1/C3 フォローアップ: 旧 shape custom build の ValueError の silent 化 regression と
    dynamic instructions closure から raise される例外の rollout 全体 abort を解消する）。

    catch は `_rollout._apply_candidate` と `_rollout._make_rollout` の rollout closure でのみ
    行い、それ以外の例外（TypeError / RuntimeError / OptimizeError / 利用者 ValueError 等）は
    従来どおり伝搬させる（暴走防止・診断性維持）。
    """


@dataclass(frozen=True)
class RolloutResult:
    """1 rollout の plain な観測（reward へ渡す・SDK 型に非依存・NFR-1）。

    `optimizer` が rollout（`run_with_observation`）から抽出した plain データを reward callable へ
    渡すための型。`case` は `OptimizeCase` または利用者定義の dict で、reward ファクトリが当該
    フィールドを解釈する。

    Attributes:
        case: 入力ケース（`OptimizeCase` または利用者定義の任意型・dict 等）。
        output: rollout が生成した最終出力テキスト。
        tool_calls: 観測したツール呼び出し名の列（順序保持・承認 resume 後の segment も含む）。
        fired_approvals: 承認ゲートが発火した（中断時に pending に出た）ツール名の列。各ラウンドで
            新たに pending に出た tool_name を順次連結する（approve / reject を問わず・llmops の
            `ObservedApproval` と同型の recall 用観測）。
        route_steps: 実行経路（起点を含む agent 名の列・llmops `HandoffRoute` と同型）。単体
            agent は `["bot"]`、handoff があれば `["triage", "billing"]` のように順序・経由回数を
            保持する。
        last_agent: 最終応答を返した agent 名（経路の終端）。rollout が応答する前に中断した場合は
            None になりうる。
    """

    case: Any
    output: str
    tool_calls: list[str] = field(default_factory=list)
    fired_approvals: list[str] = field(default_factory=list)
    route_steps: list[str] = field(default_factory=list)
    last_agent: str | None = None


@dataclass(frozen=True)
class SlotSegment:
    """スロットを構成する 1 セグメント（qualified 参照付き・新 shape の構造情報）。

    `ref` は `"base:main"` / `"part:style"` / `"agent:triage"` のような qualified 参照文字列で、
    セグメントの由来（base / part / agent 等の種別と名前）を示す。`tune=True` のセグメントのみ
    APO の最適化対象になり、`tune=False` は固定セグメントとして候補テキストへそのまま連結される。

    Attributes:
        ref: qualified 参照文字列（例: `"base:main"` / `"part:style"` / `"agent:triage"`）。
        text: セグメント本文（`${var}` プレースホルダ保持）。
        tune: 最適化対象かを示すフラグ（True = APO 対象・False = 固定セグメント）。
    """

    ref: str
    text: str
    tune: bool

    def __post_init__(self) -> None:
        """`tune` が bool であることを構築時に検証する。

        Raises:
            ValueError: `tune` が bool でない場合。
        """
        validate_bool(self.tune, "tune")


@dataclass(frozen=True)
class Slot:
    """APO の最適化対象スロット 1 件（`prompt_slot` の戻り値・plain）。

    seed（`${var}` 未展開・プレースホルダ保持）と build（候補テキスト → `AgentSpec`）・vars
    （最適化対象外・rollout 再注入）を保持する。`build` を内包するため `optimizer` が rebind を
    自動導出でき、利用者は手書き rebind を渡さなくてよい（FR-3 / FR-9）。`PromptStore` は
    `prompt_slot` が読み取り参照するのみで本型は SDK / `PromptStore` 型を保持しない。

    構成情報は `segments`（構成順の `SlotSegment` 列）が SoT として保持し、`_new_default_build`
    と optimizer の OptimizeResult 合成の双方が参照する。custom build 経路 / 手書き `Slot` /
    生 seed + rebind 経路では空タプルで、その場合 optimizer は run_apo の返却をそのまま
    `OptimizeResult` にする（再合成しない）。

    Attributes:
        name: スロット名（対象エージェント / セグメント名）。
        seed: vars 未展開の seed テキスト（`${var}` プレースホルダ保持・tune 部分のみ）。
        build: 候補テキストから `AgentSpec` を構築する関数。
        vars: `${var}` 置換値（最適化対象外・各 rollout で再注入）。
        segments: 構成情報 SoT（既定 build と OptimizeResult 合成の両方が参照する）。空タプルは
            custom build / 手書き Slot / 生 seed 経路（optimizer は run_apo 返却を素通し）。
        vars_fn: vars=callable を受けた場合の保持先（`vars` の dict 契約とは分離）。既定 build が
            これを見て動的 instructions を生成する。callable を受けない場合は None。
    """

    name: str
    seed: str
    build: Callable[[str], Any]
    vars: dict[str, Any] = field(default_factory=dict)
    segments: tuple[SlotSegment, ...] = ()
    vars_fn: Callable[[Any], dict[str, Any]] | None = None


@dataclass(frozen=True)
class OptimizeResult:
    """最適化全体の構造化結果（plain・必ず返る・FR-2 / FR-9）。

    APO の結果は `${var}` プレースホルダを保持した最適化済みスロットテキスト（単一スロットは
    str・複数スロットは `{名前: テキスト}` mapping）。`save(path)` は利用者指定パスへの opt-in
    書込のみで、未呼び出し時は何も書かず `PromptStore` を触らない。

    Attributes:
        prompt: 最適化済みプロンプトテキスト（単一は str・複数は `{名前: str}` mapping・`${var}`
            保持）。**rollout 時に agent が実際に受け取る合成済み full テキスト**（構成順の
            固定・tune セグメントを `\\n\\n` 連結したもの）を返す。custom build / 生 seed + rebind
            経路では run_apo 返却をそのまま返す。
        seed: 最適化前のプロンプトテキスト（`prompt` と同じ shape・**合成済み full**）。利用者が
            「before / after」を比較表示する際のボイラープレートを不要にする。空文字
            （`""` / `{}`）は seed が解決できなかった例外的経路の既定値（通常は呼び出し側で
            CONFIG_MISSING へ倒れる）。
        diff: `seed` (before) と `prompt` (after) の **unified diff** 表記（同じ shape）。
            stdlib `difflib.unified_diff` で算出し、複数パーツ合成の中で APO がどこを変えたかが
            一目で分かる。差分なしのときは空文字。利用者は `print(result.diff)` するだけで読める。
        train_score: train 上で測った最適化結果のスコア。
        val_score: val 上で測った汎化スコア。`val` 省略時は None。
        history: 最適化の履歴（各スロット 1 件・`HistoryEntry` schema の plain dict 列）。
            `slot` / `best_score` / `best_version` / `placeholder_fallback` の 4 キーを含む。
            `placeholder_fallback=True` のときは `best_score` / `best_version` が None
            （APO 最良候補が `${var}` を喪失したため seed へフォールバックしたケース）。
    """

    prompt: str | dict[str, str]
    train_score: float
    val_score: float | None = None
    history: list[HistoryEntry] = field(default_factory=list)
    seed: str | dict[str, str] = ""
    diff: str | dict[str, str] = ""

    def to_dict(self) -> dict[str, Any]:
        """結果を plain dict として返す（ログ / 外部保存に使える）。

        Returns:
            `prompt` / `seed` / `diff` / `train_score` / `val_score` / `history` を含む plain dict。
        """
        prompt = dict(self.prompt) if isinstance(self.prompt, dict) else self.prompt
        seed = dict(self.seed) if isinstance(self.seed, dict) else self.seed
        diff = dict(self.diff) if isinstance(self.diff, dict) else self.diff
        # `list(self.history)` だけだと内側の TypedDict は同じ参照が共有され、利用者が
        # `to_dict()` の戻り値を後から書き換えると `OptimizeResult.history` も silent に
        # 書き換わる（frozen dataclass の不変契約を破る）。各 entry を浅コピーして独立化する
        # （history entry は plain な scalar のみ・shallow copy で十分・Codex 第4 round）。
        return {
            "prompt": prompt,
            "seed": seed,
            "diff": diff,
            "train_score": self.train_score,
            "val_score": self.val_score,
            "history": [dict(entry) for entry in self.history],
        }

    def save(self, path: str | Path) -> None:
        """最適化結果を利用者指定パスへ書き出す（opt-in・FR-9）。

        `prompt` が str（単一スロット）の場合はテキストをそのまま書き、mapping（複数スロット）の
        場合は JSON として書く。`${var}` プレースホルダは展開せず保持したまま書く。`PromptStore`
        のテンプレートやライブラリ管理領域は一切書き換えない（PromptStore 非書込）。

        Args:
            path: 書き出し先パス（利用者指定）。

        Raises:
            OSError: 書込先が書込不能 / 不正な場合（fail-closed・呼び出し側へ伝播）。
        """
        target = Path(path)
        if isinstance(self.prompt, str):
            target.write_text(self.prompt, encoding="utf-8")
        else:
            target.write_text(
                json.dumps(self.prompt, ensure_ascii=False, indent=2), encoding="utf-8"
            )
