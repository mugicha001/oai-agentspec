"""意図予測の型定義（pydantic BaseModel 全面採用）。

すべての description は pydantic 標準の Field(description=...) で表現し、
model_json_schema() に自動反映される。ConfidenceLevel の各値の意味は
IntentCandidate.level の Field description（_CONFIDENCE_LEVEL_DESCRIPTION 経由）
に集約し、Field description は schema 向けメタ、IntentCategory.description は
各カテゴリ値の LLM 説明という使い分けをする。
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from ._readonly import ReadOnlyAnyMapping


class ConfidenceLevel(StrEnum):
    """5 段階の意図分類信頼度。各値の意味は _CONFIDENCE_LEVEL_DESCRIPTION を参照。"""

    CERTAIN = "certain"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPECULATIVE = "speculative"


# ConfidenceLevel 各値の意味（単一ソース）。Field description と render_prompt が共有する。
_CONFIDENCE_LEVEL_MEANINGS: dict[str, str] = {
    "certain": "ユーザーが明示的に自分の言葉で述べている",
    "high": "言い換えまたは 1 段の推論で確実に導ける",
    "medium": "妥当な文脈解釈だが、他解釈も残る",
    "low": "弱い手がかりで、他解釈と同程度もっともらしい",
    "speculative": "ほぼ根拠がない推測",
}

# IntentCandidate.level の Field description（JSON schema 経由で LLM に届く）。
_CONFIDENCE_LEVEL_DESCRIPTION = "Confidence level. Choose exactly one of:\n" + "\n".join(
    f"- '{k}': {v}" for k, v in _CONFIDENCE_LEVEL_MEANINGS.items()
)


class IntentCategory(BaseModel):
    """分類器が返せる意図カテゴリ 1 つの定義。"""

    model_config = {"frozen": True}
    name: str = Field(description="Category identifier. Used as the value of IntentCandidate.text.")
    description: str = Field(
        description="Human/LLM-readable description of when this category applies."
    )


class IntentCandidate(BaseModel):
    """分類候補 1 件。confidence 降順で並ぶ前提。"""

    model_config = {"frozen": True}
    text: str = Field(description="Category name. Must be one of the policy categories.")
    level: ConfidenceLevel = Field(description=_CONFIDENCE_LEVEL_DESCRIPTION)
    rationale: str | None = Field(default=None, description="One-sentence justification.")


class ConsistencyReport(BaseModel):
    """候補の一貫性判定結果。分類器が判定しない場合は IntentPrediction.report=None。"""

    model_config = {"frozen": True}
    conflicts: tuple[str, ...] = Field(
        default=(), description="Contradictions with prior conversation context."
    )
    stale_context: tuple[str, ...] = Field(
        default=(), description="Context that has become outdated."
    )
    over_inference: tuple[str, ...] = Field(
        default=(), description="Inferences that outrun the evidence."
    )


class IntentPrediction(BaseModel):
    """分類器の出力（固定契約）。"""

    model_config = {"frozen": True}
    candidates: tuple[IntentCandidate, ...] = Field(
        description=(
            "Sorted by ConfidenceLevel descending; within a level, LLM output order preserved."
        )
    )
    report: ConsistencyReport | None = Field(
        default=None, description="Optional consistency judgment."
    )
    metadata: Mapping[str, Any] | None = Field(
        default=None, description="Implementation-specific metadata."
    )


class IntentPolicy(BaseModel):
    """分類器が守るべき契約（意図集合 + 返却制約）。"""

    model_config = {"frozen": True, "extra": "forbid"}
    categories: tuple[IntentCategory, ...] = Field(
        description="Allowed intent categories. Non-empty, unique names."
    )
    max_candidates: int = Field(
        default=3,
        ge=1,
        description="Maximum number of candidates the classifier should return (>=1).",
    )
    extra_instructions: str = Field(
        default="",
        description=(
            "利用側が render_prompt の先頭に差し込む追加指示。空文字なら追加なし。"
            " **Trusted developer-authored text only**: エンドユーザー入力を直接渡さないこと"
            " (system プロンプトの意図を書き換えられるリスクがある)。"
        ),
    )
    include_rationale_in_prompt: bool = Field(
        default=False,
        description=(
            "True なら render_prompt の出力例 JSON に rationale フィールドを載せて LLM に生成を"
            " 促す。False (既定) では出力例から rationale を外し、生成トークン/レイテンシを抑える。"
            " どちらでも parser 側は rationale を optional として受け入れる (pass-through)。"
        ),
    )

    @field_validator("categories")
    @classmethod
    def _validate_categories(cls, v: tuple[IntentCategory, ...]) -> tuple[IntentCategory, ...]:
        """categories が非空かつ name の重複がないことを検証する。

        Args:
            v: バリデーション対象の categories tuple。

        Returns:
            検証済みの categories tuple。

        Raises:
            ValueError: 空 tuple または重複 name を含む場合。
        """
        if not v:
            raise ValueError("IntentPolicy.categories must not be empty")
        names = [c.name for c in v]
        if len(names) != len(set(names)):
            raise ValueError("IntentPolicy.categories has duplicate names")
        return v

    def render_prompt(self) -> str:
        """LLM に渡す最小の指示プロンプトを組み立てる。

        固定のタスク指示 1 行（分類 + JSON のみ出力）と、事前定義値からの引用
        （カテゴリ一覧・信頼度の意味・出力形式・制約）を Markdown 見出し 4 セクション
        で区切って出力する。タスク指示と「JSON のみ」制約は、prompt callable が発話を
        素通しする最小構成（`lambda ctx: ctx.utterance`）でも低精度・高速モデルが
        分類タスクとして応答するための最低限の固定文。few-shot 例やロール定義等の
        追加テキストは lib では持たない（必要な利用者は `extra_instructions` か
        prompt callable 側で追加するか、`include_policy_in_system=False` で全制御）。

        Returns:
            LLM に渡す prompt 文字列。
        """
        cat_lines = "\n".join(f"- {c.name}: {c.description}" for c in self.categories)
        level_lines = "\n".join(f"- {k}: {v}" for k, v in _CONFIDENCE_LEVEL_MEANINGS.items())
        # include_rationale_in_prompt で LLM に rationale 生成を促すかどうかを切り替える
        # (True 時のみ出力例に rationale フィールドを載せる)
        if self.include_rationale_in_prompt:
            item = '{"text": "<カテゴリ名>", "level": "<信頼度>", "rationale": "<理由>"}'
        else:
            item = '{"text": "<カテゴリ名>", "level": "<信頼度>"}'
        # 空白のみの extra_instructions は空扱い (先頭に無駄な空行を挿入しない)
        stripped_extra = self.extra_instructions.rstrip()
        prefix = f"{stripped_extra}\n\n" if stripped_extra else ""
        return (
            f"{prefix}"
            "ユーザー発話を以下のカテゴリに分類し、JSON のみを出力してください。\n"
            "\n"
            f"# カテゴリ\n{cat_lines}\n"
            "\n"
            f"# 信頼度 (level)\n{level_lines}\n"
            "\n"
            "# 出力形式\n"
            '{"candidates": [\n'
            f"  {item},\n"
            "  ...\n"
            "]}\n"
            "\n"
            "# 制約\n"
            "- text はカテゴリ名のいずれか\n"
            f"- 最大 {self.max_candidates} 件、level 降順で並べる\n"
            "- JSON 以外のテキスト（説明文・コードフェンス）を含めない"
        )


class IntentQuery[TContext](BaseModel):
    """分類器への入力。継承拡張可。"""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}
    utterance: str = Field(
        default="",
        description=(
            "User utterance to classify. 空文字の場合は現在発話なし（履歴のみで分類する"
            "モード）を意味し、history とどちらか一方は必要。"
        ),
    )
    history: Any | None = Field(
        default=None, description="agents.Session 相当（不透明型・docstring 契約）"
    )
    run_context: TContext | None = Field(
        default=None, description="利用側の run_context（型は TContext）"
    )


class IntentContext[TContext](BaseModel):
    """ContextBuilder が組み立てる整形済み内部型。prompt callable の入力。"""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}
    utterance: str = Field(description="User utterance.")
    history_items: tuple[Mapping[str, Any], ...] = Field(
        default=(),
        description=(
            "ContextBuilder が抽出した直近履歴アイテムのタプル。各要素は role/content 等の "
            "dict-like で、prompt callable 側が用途に応じて整形する。"
        ),
    )
    run_context: TContext | None = Field(default=None, description="Pass-through run_context.")


class ExecutableIntent(IntentCandidate):
    """実行可能アクションを指す候補 1 件。候補生成方式に依存しない固定契約（FR-4）。

    既存 `IntentCandidate` のサブクラスであるため `IntentPrediction.candidates` へそのまま
    載せられる。親型の必須フィールド `text` は `action_id` から自動補完するので、利用者は
    同じ名前を 2 度書かない。
    """

    model_config = {"frozen": True}
    action_id: str = Field(description="Registered action_id this candidate points at.")
    # 読み取り専用へ正規化する（`_readonly.ReadOnlyAnyMapping`）。frozen は属性の再束縛だけを
    # 禁じるため、素の dict のままだと候補が運ぶ実行入力を宣言後に差し替えられる。
    # `validate_default=True` は既定の空 Mapping も同じ正規化へ通すために必要である。
    parameters: ReadOnlyAnyMapping = Field(
        default_factory=dict,
        validate_default=True,
        description="Parameter values the candidate already carries. Keys are parameter names.",
    )
    source: str = Field(
        description="Which generator produced this candidate. Not validated by the library."
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_text_from_action_id(cls, data: Any) -> Any:
        """`text` を `action_id` から補完し、双方明示された場合の不一致を拒否する。

        `mode="before"` なのは、親型の `text` が必須であり、必須チェックより前に埋める必要が
        あるためである。`action_id` を持たない入力（親型そのままの dict やモデルインスタンス
        の再検証）では何もせずそのまま返す。ここで例外を出すと、本来「`action_id` が必須」
        という素直な `ValidationError` になるべき入力が別の失敗へすり替わる。

        Args:
            data: 検証前の入力。Mapping とは限らない。

        Returns:
            `text` を補完した新しい dict。入力が Mapping でないか `action_id` を持たない
            場合は入力そのもの。

        Raises:
            ValueError: `text` と `action_id` の双方が明示され両者が一致しない場合。黙って
                どちらかを採ると、候補の表示名と実行先の宣言が食い違ったまま下流へ流れる。
        """
        if not isinstance(data, Mapping) or "action_id" not in data:
            return data
        action_id = data["action_id"]
        text = data.get("text")
        if text is None:
            return {**data, "text": action_id}
        if text != action_id:
            raise ValueError(
                f"ExecutableIntent.text must match action_id: text={text!r} action_id={action_id!r}"
            )
        return data


class ExecutableSuggestion(BaseModel):
    """候補生成の結果一式（FR-4）。`IntentPrediction` を丸ごと持たない。

    `candidates` を `tuple[ExecutableIntent, ...]` として直接持つのは、`IntentPrediction` を
    経由すると純 dict 経路で親型へ coerce され `action_id` / `parameters` が落ちるためである
    （設計 §3.10・実測 1）。`report` / `metadata` は `IntentPrediction` を丸ごと持つ代わりに
    分解して持つが、保持の仕方は同じではない。`report` は generator が返したモデルをそのまま
    保持し、`metadata` は値を保ったまま読み取り専用へ正規化して保持する（防御的コピーを
    取り込むため、渡した Mapping との同一性は保たれない）。

    `context.run_context` に利用者の任意型が載るため、**直列化の成立は契約に含めない**
    （FR-1）。
    """

    model_config = {"frozen": True}
    candidates: tuple[ExecutableIntent, ...] = Field(
        description="Candidates in generator order, already filtered to registered actions."
    )
    context: IntentContext[Any] = Field(
        description="The IntentContext the candidates were generated from."
    )
    report: ConsistencyReport | None = Field(
        default=None, description="Pass-through of IntentPrediction.report."
    )
    # `parameters` と同じく読み取り専用へ正規化する（`_readonly.ReadOnlyAnyMapping`）。
    # frozen は属性の再束縛だけを禁じるため、素の Mapping のままだと宣言後に中身を
    # 差し替えられる。None は「判定材料を返さない generator」の表明であり素通しする。
    metadata: ReadOnlyAnyMapping | None = Field(
        default=None,
        description="Pass-through of IntentPrediction.metadata. Read-only; None means absent.",
    )
