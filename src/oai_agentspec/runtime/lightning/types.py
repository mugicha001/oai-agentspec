"""Agent Lightning 最適化の plain 結果型・スロット型（外部 SDK 非依存）。

本モジュールは openai-agents（`agents`）・Agent Lightning（`agentlightning`）を一切 import
しない。`_adapters/lightning` が最適化エンジンの結果を本モジュールの plain dataclass へ変換し、
最適化ロジック層（`optimizer`）と公開窓口はこの plain 型のみを扱う（NFR-1）。

すべて `@dataclass(frozen=True)`（会話の `SendResult` / llmops の結果型と一致・Pydantic 非導入）。
`OptimizeResult.save` は利用者指定パスへの opt-in 書込のみで `PromptStore` / ライブラリ管理領域を
一切書き換えない（FR-9・PromptStore 非書込）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

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


class OptimizeError(Exception):
    """最適化が送出する構造化エラー（未捕捉例外でプロセスを止めないための変換先・FR-8）。

    extra 不在 / 設定不在 / Trainer 実行失敗を `kind` で判別できる明確なエラーに統一する。SDK /
    agentlightning の生例外を上位へ漏らさず、本型へ変換して送出する（NFR-1 と整合）。

    Attributes:
        kind: 失敗種別（`FailureKind`）。
        message: 人間可読のエラーメッセージ。
    """

    def __init__(self, kind: FailureKind, message: str) -> None:
        """最適化エラーを生成する。

        Args:
            kind: 失敗種別。
            message: 人間可読メッセージ。
        """
        super().__init__(message)
        self.kind = kind
        self.message = message


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
class Slot:
    """APO の最適化対象スロット 1 件（`prompt_slot` の戻り値・plain）。

    seed（`${var}` 未展開・プレースホルダ保持）と build（候補テキスト → `AgentSpec`）・vars
    （最適化対象外・rollout 再注入）を保持する。`build` を内包するため `optimizer` が rebind を
    自動導出でき、利用者は手書き rebind を渡さなくてよい（FR-3 / FR-9）。`PromptStore` は
    `prompt_slot` が読み取り参照するのみで本型は SDK / `PromptStore` 型を保持しない。

    `fixed` は base / parts を合成した固定部分テキスト（`${var}` プレースホルダ保持）。`prompt_slot`
    が `_compose_fixed` の戻り値を保持し、(1) 既定 build が candidate と連結して agent の
    instructions を組み立てる際、(2) `OptimizeResult.seed` / `OptimizeResult.prompt` を「合成済み
    full テキスト」として返す際に参照する。利用者が custom `build` を明示する場合や生 seed +
    rebind 経路では空文字（合成不要）。

    Attributes:
        name: スロット名（対象エージェント / セグメント名）。
        seed: vars 未展開の seed テキスト（`${var}` プレースホルダ保持・tune 部分のみ）。
        build: 候補テキストから `AgentSpec` を構築する関数。
        vars: `${var}` 置換値（最適化対象外・各 rollout で再注入）。
        fixed: base / parts を合成した固定部分テキスト（`${var}` 保持・空なら合成なし）。
    """

    name: str
    seed: str
    build: Callable[[str], Any]
    vars: dict[str, Any] = field(default_factory=dict)
    fixed: str = ""


@dataclass(frozen=True)
class OptimizeResult:
    """最適化全体の構造化結果（plain・必ず返る・FR-2 / FR-9）。

    APO の結果は `${var}` プレースホルダを保持した最適化済みスロットテキスト（単一スロットは
    str・複数スロットは `{名前: テキスト}` mapping）。`save(path)` は利用者指定パスへの opt-in
    書込のみで、未呼び出し時は何も書かず `PromptStore` を触らない。

    Attributes:
        prompt: 最適化済みプロンプトテキスト（単一は str・複数は `{名前: str}` mapping・`${var}`
            保持）。**rollout 時に agent が実際に受け取る合成済み full テキスト**（`Slot.fixed`
            と tune を `\\n\\n` 連結したもの）を返す。固定部分（base / parts）が無い場合は tune
            そのものと一致する（生 seed + rebind 経路 / `prompt_slot` で base/parts 未指定 /
            custom build 経路）。
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
