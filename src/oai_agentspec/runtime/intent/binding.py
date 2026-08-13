"""結線の宣言型（`CandidateSource` / `LLMFiller`）。`catalog.bind()` の引数に渡す。

`bind` の平坦な引数列を関心事ごとに分ける。`CandidateSource` は「候補の出どころ」、
`LLMFiller` は「不足パラメータの埋め方」であり、後者を渡さないことが「穴埋め経路を
持たない = 従量課金が発生しない」という利用者の明示的な意思表示になる（FR-3）。

方針:
- `generator` / `context_builder` / `model` は**不透明値**として `Any` で受ける
  （`arbitrary_types_allowed=True`）。候補生成方式（ルール / 学習 / LLM）や接続先を宣言時に
  縛らないためであり、`agents` / `openai` は import しない（NFR-1）。
- **検証は `bind` まで持ち越さず宣言時に落とす。** 誤りが `plan()` の実行時に初めて出ると、
  候補が押された瞬間まで発覚しない。排他検証は `CandidateSource` の validator、
  `on_invalid_response` の値域は `LLMFiller` の `Literal` が担う。
- `guardrail_registry` は `LLMFiller` のフィールドではなく `bind` の引数である（設計 §3.4a）。
  「`guardrails` が非空なのに解決簿が無い」の検出は起動時検証（`planner.validate()`）が担う。
  `LLMFiller` 単体では解決簿を知らないため宣言時には落とせない。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class CandidateSource(BaseModel):
    """候補の出どころ。`generator` は必須（候補生成は代替不能）。"""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}
    generator: Any = Field(
        description=(
            "CandidateGenerator-ish object. Held opaquely and never type-checked here."
            " Only None is rejected, at declaration time, since Any lets it pass the"
            " required-field check and surface as an AttributeError inside plan()."
        )
    )
    context_builder: Any = Field(
        default=None,
        description="ContextBuilder-ish object. None means the default builder is used.",
    )
    history_limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Convenience argument for the default builder. None (the sentinel) lets the"
            " default builder pick its own 20, which keeps 'explicitly 20' distinguishable."
            " Values below 1 are rejected at declaration time instead of waiting for the"
            " default builder to raise ValueError on the first plan() call."
        ),
    )

    @model_validator(mode="after")
    def _check_generator_is_not_none(self) -> CandidateSource:
        """`generator` が None でないことを宣言時に確かめる。

        判定は `is None` だけで行う（`if not self.generator` にすると `__bool__` が偽の
        不透明値まで巻き込む。lib は generator を不透明値として扱うため中身で判断しない）。

        Returns:
            検証済みの自分自身。

        Raises:
            ValueError: `generator` が None の場合。`Any` は pydantic の必須検証を素通りする
                ため、誤りが `plan()` 実行時の `AttributeError` まで持ち越される。
        """
        if self.generator is None:
            raise ValueError("generator must not be None; candidate generation has no fallback")
        return self

    @model_validator(mode="after")
    def _check_exclusive_context_builder(self) -> CandidateSource:
        """`context_builder` と `history_limit` の同時指定を拒否する。

        Returns:
            検証済みの自分自身。

        Raises:
            ValueError: 双方に非 `None` を渡した場合。`history_limit` は既定 builder 専用の
                便宜引数であり、差し替えた builder には効かない。黙って無視すると
                「20 件に絞ったつもり」の宣言が効かないまま実行される。
        """
        if self.context_builder is not None and self.history_limit is not None:
            raise ValueError(
                "history_limit only applies to the default context builder; it cannot be "
                "combined with an explicit context_builder"
            )
        return self


class LLMFiller(BaseModel):
    """不足パラメータの埋め方。渡さなければ穴埋め経路そのものが存在しない。"""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}
    model: Any = Field(
        description=(
            "Model to drive the prediction agent with. Held opaquely; the library builds the"
            " agent itself so the user never passes an agent instance."
        )
    )
    on_invalid_response: Literal["error", "skip"] = Field(
        default="error",
        description="What to do when the prediction response fails validation.",
    )
    guardrails: tuple[str, ...] = Field(
        default=(),
        description=(
            "Registered guardrail names to attach to the prediction agent. Empty means none"
            " are attached (opt-in). Names are resolved by bind(guardrail_registry=...)."
        ),
    )
