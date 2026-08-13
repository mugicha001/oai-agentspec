"""L1: `runtime.intent._validate` の起動時検証（設計 §5 タスク 1-11 / §4a-2 の 9 種）。

FR-3 の受け入れ基準を pin する。9 種の検査それぞれについて **違反が確実に落ちること**と
**正常な宣言では落ちないこと**の両方を固定する（検査が空振りしても外からは正常に見えるため、
片側だけでは「何も見ていない実装」を通してしまう）。あわせて、判定対象のスコープが検査ごとに
異なること（検査 4 = マージ後 / 検査 5 = カタログ全体 / 検査 8 = 当該 `ActionSpec` 自身の宣言
のみ）と、`PromptTemplateIntegrityError` を捕捉せず伝播すること（設計 §3.12）を対象とする。

例外の型は `KeyError`（検査 1 / 9）と `ValueError`（検査 2-8）が混在する。`PromptResolutionError`
は `KeyError` 派生であり `except`/`raise` の取り違えが型階層に隠れるため、送出型は必ず
`type(excinfo.value) is ...` で固定する。

外部依存（agents / openai）なし。`AgentRegistry` / `GuardrailRegistry` は duck-typed な Fake
（`names()` / `get()`）を使い、`PromptStore` は tmp_path 上の実物を使う。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from oai_agentspec.exceptions import PromptTemplateIntegrityError
from oai_agentspec.prompts import PromptLayout, PromptStore
from oai_agentspec.runtime.intent._validate import _validate_catalog
from oai_agentspec.runtime.intent.actions import ActionCatalog, ActionSpec, param
from oai_agentspec.runtime.intent.binding import LLMFiller

pytestmark = pytest.mark.unit


_LOGGER_NAME = "oai_agentspec.runtime.intent._validate"
_PRIVATE_SYMBOL = "_validate_catalog"
_LAYOUT = PromptLayout(base="base", parts="parts", agents="agents")


# ---- Fake（SDK を使わない duck-typed 登録簿） ----


class _FakeAgentRegistry:
    """`AgentRegistry` の最小 Fake。`names()` / `get()` の双方を提供する。"""

    def __init__(self, *names: str) -> None:
        self._names = sorted(names)

    def names(self) -> list[str]:
        """登録済みエージェント名を昇順で返す。"""
        return list(self._names)

    def get(self, name: str) -> object:
        """未登録名なら `KeyError`（本物と同じ契約）。"""
        if name not in self._names:
            raise KeyError(name)
        return object()


class _FakeGuardrailRegistry:
    """`GuardrailRegistry` の最小 Fake。`names()` / `get()` の双方を提供する。"""

    def __init__(self, *names: str) -> None:
        self._names = sorted(names)

    def names(self) -> list[str]:
        """登録済みガードレール名を昇順で返す。"""
        return list(self._names)

    def get(self, name: str) -> object:
        """未登録名なら `KeyError`（本物と同じ契約）。"""
        if name not in self._names:
            raise KeyError(name)
        return object()


class _IntegrityPromptStore:
    """lockdown 済み `PromptStore` の代役。`compose()` が常に整合性違反を送出する。"""

    def compose(self, *args: Any, **kwargs: Any) -> str:
        """常に `PromptTemplateIntegrityError` を送出する。"""
        raise PromptTemplateIntegrityError("manifest に未掲載のセグメントです")


class _Tenant:
    """`context` の入れ子。属性は存在するが値が `None` のものを含む。"""

    id = "t-001"
    plan = None


class _Ctx:
    """検査 7 用の代表インスタンス（run_context 相当）。"""

    def __init__(self) -> None:
        self.tenant = _Tenant()
        self.region = "jp"


# ---- ヘルパ ----


def _agents(*names: str) -> _FakeAgentRegistry:
    """指定名だけを持つエージェント登録簿を返す。"""
    return _FakeAgentRegistry(*names)


def _store(tmp_path: Path, **bodies: str) -> PromptStore:
    """`parts/<name>.md` を書き出した実物の `PromptStore` を返す。"""
    parts = tmp_path / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for name, body in bodies.items():
        (parts / f"{name}.md").write_text(body, encoding="utf-8")
    return PromptStore(tmp_path, _LAYOUT)


def _spec(
    action_id: str = "run_load_test",
    *,
    action_agent: str = "load_test_agent",
    label: str = "負荷試験 ${seconds} 秒",
    parameters: tuple[Any, ...] | None = None,
    prompt: tuple[str, ...] = (),
    prompt_vars: Mapping[str, str] | None = None,
) -> ActionSpec:
    """他の検査に一切触れない健全な `ActionSpec` を組み、指定分だけを崩す。"""
    return ActionSpec(
        action_id=action_id,
        description="負荷試験を実行する",
        action_agent=action_agent,
        label=label,
        parameters=parameters if parameters is not None else (param("seconds", int, default=30),),
        prompt=prompt,
        prompt_vars=dict(prompt_vars or {}),
    )


def _catalog(*specs: ActionSpec, **kwargs: Any) -> ActionCatalog:
    """宣言簿を組んで `specs` を登録した `ActionCatalog` を返す。"""
    catalog = ActionCatalog(**kwargs)
    for spec in specs:
        catalog.register(spec)
    return catalog


def _validate(catalog: ActionCatalog, **kwargs: Any) -> None:
    """`registry` の既定を補って `_validate_catalog` を呼ぶ。"""
    kwargs.setdefault("registry", _agents("load_test_agent"))
    return _validate_catalog(catalog, **kwargs)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """`_validate` の logger が出した WARNING レコードだけを取り出す。"""
    return [r for r in caplog.records if r.levelno == logging.WARNING and r.name == _LOGGER_NAME]


# ---- 正常系（すべての検査を通す） ----


def test_valid_catalog_returns_none(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """全検査を通る宣言では `None` を返し、WARNING も出さない（FR-3）。"""
    prompts = _store(tmp_path, hint="テナント ${tenant} 向けのヒント")
    spec = _spec(
        parameters=(
            param("seconds", int, default=30),
            param("target", str, from_context="tenant.id"),
            param("note", str, by_llm=True, prompt="part:hint"),
        ),
        prompt=("part:hint",),
        prompt_vars={"tenant": "tenant.id"},
    )
    with caplog.at_level(logging.WARNING):
        result = _validate(
            _catalog(spec),
            prompts=prompts,
            guardrail_registry=_FakeGuardrailRegistry("pii"),
            llm_filler=LLMFiller(model="gpt-x", guardrails=("pii",)),
            context=_Ctx(),
        )

    assert result is None
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_empty_catalog_is_valid() -> None:
    """宣言が 0 件でも例外にならない。"""
    assert _validate(_catalog()) is None


# ---- 検査 1: action_agent が registry に存在するか（集約 KeyError） ----


def test_unknown_action_agent_raises_key_error() -> None:
    """未登録の `action_agent` は `KeyError`（`ValueError` ではない）。"""
    catalog = _catalog(_spec(action_agent="ghost_agent"))
    with pytest.raises(KeyError) as excinfo:
        _validate(catalog)

    assert type(excinfo.value) is KeyError
    assert "ghost_agent" in str(excinfo.value)


def test_unknown_action_agents_aggregate_into_one_key_error() -> None:
    """複数違反は 1 つの `KeyError` へ集約する（最初の 1 件で止めない）。"""
    catalog = _catalog(
        _spec("zeta_action", action_agent="ghost_zeta"),
        _spec("alpha_action", action_agent="ghost_alpha"),
    )
    with pytest.raises(KeyError) as excinfo:
        _validate(catalog)

    message = str(excinfo.value)
    assert "ghost_zeta" in message
    assert "ghost_alpha" in message


def test_registered_action_agent_passes() -> None:
    """`registry.names()` にある `action_agent` は通る。"""
    catalog = _catalog(_spec(action_agent="load_test_agent"))
    assert _validate(catalog, registry=_agents("load_test_agent", "other")) is None


# ---- 検査 2: label のプレースホルダ ⊆ 宣言パラメータ名 ----


def test_label_placeholder_not_declared_raises_value_error() -> None:
    """`label` の `${...}` が宣言パラメータに無ければ差分を挙げて `ValueError`。"""
    catalog = _catalog(_spec(label="負荷試験 ${seconds} 秒 / ${hours} 時間"))
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog)

    assert type(excinfo.value) is ValueError
    assert "hours" in str(excinfo.value)


def test_label_placeholder_subset_is_valid() -> None:
    """宣言パラメータの一部だけを使う `label` は正常（等しさではなく包含）。"""
    catalog = _catalog(
        _spec(
            label="負荷試験 ${seconds} 秒",
            parameters=(param("seconds", int, default=30), param("target", str, default="all")),
        )
    )
    assert _validate(catalog) is None


def test_label_without_placeholders_is_valid() -> None:
    """プレースホルダを持たない `label` は正常。"""
    assert _validate(_catalog(_spec(label="負荷試験"))) is None


# ---- 検査 2 の続き: render 不能な `label` を宣言時に落とす（セキュリティレビュー指摘 #88-W4） ----
#
# `Template(spec.label).get_identifiers()` は `"100$ "` のような不正なプレースホルダを
# 黙って無視するため、包含チェックだけでは render 不能な `label` が起動時検証を通過する。
# 一方 render 側（`ActionPlan.label`）は宣言の取りこぼしを黙って通さないため意図的に
# `substitute` を使っており、実行時に `ValueError: Invalid placeholder` になる。宣言時に
# 落とせる誤りを毎ターンの候補提示まで持ち越さない。


@pytest.mark.parametrize(
    "label",
    [
        pytest.param("負荷試験 100$ で ${seconds} 秒", id="bare-dollar-before-digit"),
        pytest.param("コスト $ を ${seconds} 秒", id="bare-dollar-before-space"),
        pytest.param("${} を ${seconds} 秒", id="empty-braces"),
        pytest.param("負荷試験 ${seconds} 秒$", id="trailing-dollar"),
    ],
)
def test_label_with_an_invalid_placeholder_raises_value_error(label: str) -> None:
    """render 不能な `label` は宣言時に `ValueError`（指摘 #88-W4）。

    いずれも `get_identifiers()` では `['seconds']` としか見えず、宣言パラメータとの包含
    チェックだけでは違反にならない。`substitute` が実行時に落とす誤りを起動時検証で落とす。
    """
    catalog = _catalog(_spec(label=label))
    with pytest.raises(ValueError, match="label") as excinfo:
        _validate(catalog)

    assert type(excinfo.value) is ValueError
    assert "run_load_test" in str(excinfo.value)


def test_invalid_labels_aggregate_into_one_value_error() -> None:
    """複数アクションの render 不能な `label` は 1 つの `ValueError` へ集約する（指摘 #88-W4）。

    検査ごとに全違反を集約する既存方針（`_validate` のモジュール docstring）へ揃える。
    """
    catalog = _catalog(
        _spec("zeta_action", label="コスト 100$ / ${seconds} 秒"),
        _spec("alpha_action", label="${} / ${seconds} 秒"),
    )
    with pytest.raises(ValueError, match="label") as excinfo:
        _validate(catalog)

    message = str(excinfo.value)
    assert "zeta_action" in message
    assert "alpha_action" in message


def test_label_with_an_escaped_dollar_is_valid() -> None:
    """`$$` でエスケープした `label` は従来どおり通る（指摘 #88-W4 の回帰防止）。

    `$` を含むだけで落とす実装にすると、render できる正当な宣言まで拒否してしまう。
    """
    assert _validate(_catalog(_spec(label="コスト 100$$ / ${seconds} 秒"))) is None


def test_label_check_still_reports_undeclared_placeholders_first() -> None:
    """未宣言プレースホルダの包含チェックは従来どおり働く（指摘 #88-W4 の回帰防止）。

    render 可否の検査を足したことで既存の差分報告が置き換わっていないことを固定する。
    """
    catalog = _catalog(_spec(label="負荷試験 ${seconds} 秒 / ${hours} 時間"))
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog)

    assert type(excinfo.value) is ValueError
    assert "hours" in str(excinfo.value)


# ---- 検査 3: prompt セグメントが prompts で解決できるか ----


def test_unresolvable_segment_raises_value_error(tmp_path: Path) -> None:
    """未解決セグメントは `ValueError` へ変換する（`KeyError` のまま漏らさない）。"""
    prompts = _store(tmp_path, hint="ヒント")
    catalog = _catalog(
        _spec(
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:missing_segment",),
        )
    )
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, prompts=prompts)

    # PromptResolutionError は KeyError 派生。素通しさせると型で区別できなくなる。
    assert type(excinfo.value) is ValueError
    assert not isinstance(excinfo.value, KeyError)
    assert "missing_segment" in str(excinfo.value)


def test_unresolvable_segment_chains_the_original_error(tmp_path: Path) -> None:
    """原因の `PromptResolutionError` を `raise ... from` で残す（FR-3 実現手順）。"""
    prompts = _store(tmp_path, hint="ヒント")
    catalog = _catalog(
        _spec(
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:missing_segment",),
        )
    )
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, prompts=prompts)

    assert excinfo.value.__cause__ is not None


def test_unresolvable_segments_are_aggregated(tmp_path: Path) -> None:
    """複数の未解決セグメントを 1 つの `ValueError` へ集約する。"""
    prompts = _store(tmp_path, hint="ヒント")
    catalog = _catalog(
        _spec(
            "zeta_action",
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:missing_zeta",),
        ),
        _spec(
            "alpha_action",
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:missing_alpha",),
        ),
    )
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, prompts=prompts)

    message = str(excinfo.value)
    assert "missing_zeta" in message
    assert "missing_alpha" in message


def test_catalog_level_segment_is_checked(tmp_path: Path) -> None:
    """`ActionCatalog.prompt` のセグメントも検査対象（アクション側だけを見ない）。"""
    prompts = _store(tmp_path, hint="ヒント")
    catalog = _catalog(
        _spec(parameters=(param("seconds", int, default=30), param("note", str, by_llm=True))),
        prompt=("part:missing_common",),
    )
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, prompts=prompts)

    assert "missing_common" in str(excinfo.value)


def test_param_level_segment_is_checked(tmp_path: Path) -> None:
    """`param(prompt=...)` のセグメントも検査対象（穴埋め時に実際に積まれるため）。"""
    prompts = _store(tmp_path, hint="ヒント")
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, default=30),
                param("note", str, by_llm=True, prompt="part:missing_param_segment"),
            )
        )
    )
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, prompts=prompts)

    assert "missing_param_segment" in str(excinfo.value)


def test_resolvable_segments_pass(tmp_path: Path) -> None:
    """解決できるセグメントだけなら通る。"""
    prompts = _store(tmp_path, hint="ヒント本文", common="共通本文")
    catalog = _catalog(
        _spec(
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:hint",),
        ),
        prompt=("part:common",),
    )
    assert _validate(catalog, prompts=prompts) is None


# ---- 検査 4: テンプレのプレースホルダ ⊆ prompt_vars のキー（マージ後） ----


def test_template_placeholder_missing_from_prompt_vars_raises(tmp_path: Path) -> None:
    """テンプレの `${...}` が `prompt_vars` に無ければ差分を挙げて `ValueError`。"""
    prompts = _store(tmp_path, hint="テナント ${tenant} 向け")
    catalog = _catalog(
        _spec(
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:hint",),
        )
    )
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, prompts=prompts)

    assert type(excinfo.value) is ValueError
    assert "tenant" in str(excinfo.value)


def test_placeholder_supplied_by_catalog_prompt_vars_is_valid(tmp_path: Path) -> None:
    """判定対象は**マージ後**。カタログ側が供給する変数で足りていれば通る。"""
    prompts = _store(tmp_path, hint="テナント ${tenant} / 地域 ${region}")
    catalog = _catalog(
        _spec(
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:hint",),
            prompt_vars={"region": "region"},
        ),
        prompt_vars={"tenant": "tenant.id"},
    )
    assert _validate(catalog, prompts=prompts) is None


def test_catalog_level_template_placeholder_is_checked(tmp_path: Path) -> None:
    """判定対象は**マージ後**。カタログ側セグメントのプレースホルダも検査される。

    テンプレ側をマージ前（`spec.prompt`）で走査すると、カタログ既定として置いた
    セグメントの `${...}` が誰にも供給されないまま検査を素通りする。
    """
    prompts = _store(tmp_path, common="全体共通 ${company}")
    catalog = _catalog(
        _spec(
            parameters=(
                param("target", str, default="h"),
                param("seconds", int, default=30),
                param("note", str, by_llm=True),
            )
        ),
        prompt=("part:common",),
    )
    with pytest.raises(ValueError, match="company"):
        _validate(catalog, prompts=prompts)


def test_placeholder_error_lists_only_the_missing_name(tmp_path: Path) -> None:
    """差分の列挙であり、供給済みの変数名を巻き込まない。"""
    prompts = _store(tmp_path, hint="テナント ${tenant} / 地域 ${region}")
    catalog = _catalog(
        _spec(
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:hint",),
            prompt_vars={"tenant": "tenant.id"},
        )
    )
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, prompts=prompts)

    assert "region" in str(excinfo.value)


# ---- 検査 5: prompt_vars のキーがどのテンプレでも未使用でないか（カタログ全体） ----


def test_unused_prompt_vars_key_raises(tmp_path: Path) -> None:
    """どのテンプレにも現れない `prompt_vars` のキーは効果がないため `ValueError`。"""
    prompts = _store(tmp_path, hint="テナント ${tenant} 向け")
    catalog = _catalog(
        _spec(
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:hint",),
            prompt_vars={"tenant": "tenant.id", "unused_key": "region"},
        )
    )
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, prompts=prompts)

    assert type(excinfo.value) is ValueError
    assert "unused_key" in str(excinfo.value)


def test_catalog_prompt_vars_key_used_by_one_spec_is_valid(tmp_path: Path) -> None:
    """判定対象は**カタログ全体**。1 つの `ActionSpec` で使われていれば足りる。"""
    prompts = _store(tmp_path, hint="テナント ${tenant} 向け", plain="変数を使わない本文")
    catalog = _catalog(
        _spec(
            "uses_tenant",
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:hint",),
        ),
        _spec(
            "uses_nothing",
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:plain",),
        ),
        prompt_vars={"tenant": "tenant.id"},
    )
    assert _validate(catalog, prompts=prompts) is None


# ---- 検査 4 / 5 の走査範囲: param(prompt=...) のテンプレも含める（要件 149 / 150 行） ----


def test_param_level_template_placeholder_missing_from_prompt_vars_raises(tmp_path: Path) -> None:
    """`param(prompt=...)` のテンプレの `${...}` も検査 4 の対象（過小検知を塞ぐ）。

    検査 3 は既に `param.prompt` のセグメントを解決しているのに、検査 4 の走査範囲が
    `resolve_prompt`（アクション + カタログ）だけだと、パラメータ側テンプレの
    プレースホルダが誰にも供給されないまま起動時検証を素通りする。
    """
    prompts = _store(tmp_path, ask_note="上限 ${limit} まででメモを書く")
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, default=30),
                param("note", str, by_llm=True, prompt="part:ask_note"),
            )
        )
    )
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, prompts=prompts)

    assert type(excinfo.value) is ValueError
    assert "limit" in str(excinfo.value)


def test_prompt_vars_key_used_only_by_param_level_template_is_valid(tmp_path: Path) -> None:
    """`param(prompt=...)` のテンプレだけで使われるキーは未使用ではない（過剰検知を塞ぐ）。

    検査 5 の走査範囲が `resolve_prompt` だけだと、パラメータ側テンプレでのみ使う
    `prompt_vars` のキーが「どのテンプレでも未使用」と誤判定されて起動できなくなる。
    """
    prompts = _store(tmp_path, ask_note="上限 ${limit} まででメモを書く")
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, default=30),
                param("note", str, by_llm=True, prompt="part:ask_note"),
            ),
            prompt_vars={"limit": "job.limit"},
        )
    )
    assert _validate(catalog, prompts=prompts) is None


# ---- 検査 6: 同一 prompt_vars キーが異なるパスへ宣言されていないか ----


def test_same_prompt_vars_key_with_different_paths_raises(tmp_path: Path) -> None:
    """カタログ全体で同一キーが別パスへ宣言されていれば `ValueError`。"""
    prompts = _store(tmp_path, hint="テナント ${tenant} 向け")
    catalog = _catalog(
        _spec(
            "first_action",
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:hint",),
            prompt_vars={"tenant": "tenant.id"},
        ),
        _spec(
            "second_action",
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:hint",),
            prompt_vars={"tenant": "region"},
        ),
    )
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, prompts=prompts)

    assert type(excinfo.value) is ValueError
    assert "tenant" in str(excinfo.value)


def test_same_prompt_vars_key_with_same_path_is_valid(tmp_path: Path) -> None:
    """同一キー・同一パスの重複は許容する。"""
    prompts = _store(tmp_path, hint="テナント ${tenant} 向け")
    catalog = _catalog(
        _spec(
            "first_action",
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:hint",),
            prompt_vars={"tenant": "tenant.id"},
        ),
        _spec(
            "second_action",
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:hint",),
            prompt_vars={"tenant": "tenant.id"},
        ),
    )
    assert _validate(catalog, prompts=prompts) is None


# ---- 検査 7: context 指定時のパス構造解決 ----


def test_unresolvable_from_context_path_raises_with_context() -> None:
    """`context` を渡したとき解決できない `from_context` のパスは `ValueError`。"""
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, default=30),
                param("target", str, from_context="tenant.nonexistent"),
            )
        )
    )
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, context=_Ctx())

    assert type(excinfo.value) is ValueError
    assert "tenant.nonexistent" in str(excinfo.value)


def test_unresolvable_prompt_vars_path_raises_with_context(tmp_path: Path) -> None:
    """`prompt_vars` の値（パス）も構造解決の対象。"""
    prompts = _store(tmp_path, hint="テナント ${tenant} 向け")
    catalog = _catalog(
        _spec(
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:hint",),
            prompt_vars={"tenant": "tenant.nonexistent"},
        )
    )
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, prompts=prompts, context=_Ctx())

    assert "tenant.nonexistent" in str(excinfo.value)


def test_path_resolving_to_none_is_not_a_violation() -> None:
    """値が `None` でも構造的には解決できているため違反にしない。"""
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, default=30),
                param("target", str, from_context="tenant.plan"),
            )
        )
    )
    assert _validate(catalog, context=_Ctx()) is None


def test_missing_mapping_key_is_a_violation() -> None:
    """mapping にキーが無いパスは違反として検出される（検査 7）。

    属性フォールバックへ落として `hasattr` で判定すると、dict の
    メソッド名（`items` など）と衝突するキーだけが偶然通る。
    """
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, default=30),
                param("target", str, from_context="tenant.missing"),
            )
        )
    )
    with pytest.raises(ValueError, match="tenant.missing"):
        _validate(catalog, context={"tenant": {"id": "t-001"}})


def test_mapping_context_is_walked_by_key() -> None:
    """mapping の起点はキーで辿る（属性へ落とさない）。"""
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, default=30),
                param("target", str, from_context="tenant.id"),
            )
        )
    )
    assert _validate(catalog, context={"tenant": {"id": "t-001"}}) is None


def test_paths_are_not_checked_without_context() -> None:
    """`context` を省略したらパスの構造検査は行わない（他の検査のみ）。"""
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, default=30),
                param("target", str, from_context="tenant.nonexistent"),
            )
        )
    )
    assert _validate(catalog) is None


def test_unresolvable_paths_are_aggregated(tmp_path: Path) -> None:
    """`from_context` と `prompt_vars` の違反を 1 つの `ValueError` へ集約する。"""
    prompts = _store(tmp_path, hint="テナント ${tenant} 向け")
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, default=30),
                param("target", str, from_context="tenant.absent_param"),
                param("note", str, by_llm=True),
            ),
            prompt=("part:hint",),
            prompt_vars={"tenant": "tenant.absent_var"},
        )
    )
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, prompts=prompts, context=_Ctx())

    message = str(excinfo.value)
    assert "tenant.absent_param" in message
    assert "tenant.absent_var" in message


# ---- 検査 7 の続き: 中間セグメントの `None` は違反にしない（レビュー指摘・非対称 1） ----
#
# `actions._resolve_path` は「途中のセグメントが `None` なら `None` を返す」（宣言順に次の
# パスを試す正常系）が、検査 7 の `_is_resolvable` にはこの分岐が無く `hasattr(None, "id")`
# が偽になって違反へ倒れる。代表 context の任意項目が未設定なだけで起動時に落ちるため、
# 「解決結果の値が `None` であることは違反として扱わない」（FR-3 L152・既存
# `test_path_resolving_to_none_is_not_a_violation`）と同じ扱いへ揃える。
#
# ただし「検査 7 を丸ごと素通しにする」修正でも緑にならないよう、非 `None` のオブジェクトに
# 存在しない属性 / キーを指すパスは引き続き違反であることを対で pin する。


class _OptionalUser:
    """`user` 配下の任意項目。属性は存在する。"""

    id = "u-001"


class _OptionalCtx:
    """任意項目が未設定になりうる代表インスタンス。"""

    def __init__(self, user: object | None) -> None:
        self.user = user


def _from_context_catalog(path: str) -> ActionCatalog:
    """`from_context=path` だけを持つ健全な宣言簿を組む（他の検査には触れない）。"""
    return _catalog(
        _spec(
            parameters=(
                param("seconds", int, default=30),
                param("target", str, from_context=path),
            )
        )
    )


@pytest.mark.parametrize(
    "context",
    [
        pytest.param(_OptionalCtx(None), id="attribute-context"),
        pytest.param({"user": None}, id="mapping-context"),
    ],
)
def test_intermediate_none_in_a_from_context_path_is_not_a_violation(context: Any) -> None:
    """中間セグメントが `None` のパスは違反にしない（非対称 1）。

    `_resolve_path` はこの状態を「解決できないので次のパスを試す」正常系として扱う。検査 7
    だけが違反にすると、代表 context の任意項目が未設定なだけでアプリが起動できなくなる。
    """
    assert _validate(_from_context_catalog("user.id"), context=context) is None


def test_intermediate_none_deeper_in_the_path_is_not_a_violation() -> None:
    """`None` より奥のセグメントが 2 段以上あっても違反にしない（非対称 1）。"""
    assert _validate(_from_context_catalog("user.profile.locale"), context=_OptionalCtx(None)) is (
        None
    )


@pytest.mark.parametrize(
    ("context", "path"),
    [
        pytest.param(_OptionalCtx(_OptionalUser()), "user.nonexistent", id="attribute-context"),
        pytest.param({"user": {"id": "u-001"}}, "user.nonexistent", id="mapping-context"),
    ],
)
def test_missing_member_after_a_non_none_object_is_still_a_violation(
    context: Any, path: str
) -> None:
    """非 `None` のオブジェクトに無い属性 / キーは引き続き違反（非対称 1 の回帰防止）。

    中間 `None` を許すために検査 7 そのものを緩めると、宣言の打ち間違いが起動時に落ちなく
    なる。許容するのは「構造は正しいが値が無い」ケースだけである。
    """
    with pytest.raises(ValueError, match="user.nonexistent") as excinfo:
        _validate(_from_context_catalog(path), context=context)

    assert type(excinfo.value) is ValueError


def test_intermediate_none_in_a_prompt_vars_path_is_not_a_violation(tmp_path: Path) -> None:
    """`prompt_vars` のパスでも中間 `None` は違反にしない（非対称 1）。

    検査 7 は `from_context` と `prompt_vars` の双方を同じ `_is_resolvable` で判定するため、
    片方だけ直すと非対称が残る。
    """
    prompts = _store(tmp_path, hint="担当 ${owner} 向け")
    catalog = _catalog(
        _spec(
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:hint",),
            prompt_vars={"owner": "user.id"},
        )
    )
    assert _validate(catalog, prompts=prompts, context=_OptionalCtx(None)) is None


# ---- 検査 8: by_llm=True が 0 件なのに prompt / prompt_vars を宣言していないか ----


def test_prompt_without_by_llm_parameter_raises(tmp_path: Path) -> None:
    """埋める経路が無いのに `prompt` を宣言していれば効果がないため `ValueError`。"""
    prompts = _store(tmp_path, hint="ヒント本文")
    catalog = _catalog(_spec(prompt=("part:hint",)))
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, prompts=prompts)

    assert type(excinfo.value) is ValueError
    assert "run_load_test" in str(excinfo.value)


def test_prompt_vars_without_by_llm_parameter_raises(tmp_path: Path) -> None:
    """`prompt_vars` だけの宣言も同じく `ValueError`。

    カタログ側にテンプレートを置き `${tenant}` を使わせることで、検査 4 / 5 は満たしたまま
    検査 8 だけが違反になる構成にしている（別の検査で落ちても同じ `ValueError` になるため）。
    """
    prompts = _store(tmp_path, hint="テナント ${tenant} 向け")
    catalog = _catalog(_spec(prompt_vars={"tenant": "tenant.id"}), prompt=("part:hint",))
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, prompts=prompts)

    assert type(excinfo.value) is ValueError
    assert "run_load_test" in str(excinfo.value)


def test_catalog_prompt_does_not_trigger_check_for_specs_without_by_llm(tmp_path: Path) -> None:
    """判定対象は**当該 `ActionSpec` 自身の宣言のみ**。カタログ由来のマージ分は対象外。"""
    prompts = _store(tmp_path, hint="テナント ${tenant} 向け")
    catalog = _catalog(
        _spec("uses_llm", parameters=(param("note", str, by_llm=True),), label="LLM 利用"),
        _spec("no_llm", label="LLM 非利用", parameters=(param("seconds", int, default=30),)),
        prompt=("part:hint",),
        prompt_vars={"tenant": "tenant.id"},
    )
    assert _validate(catalog, prompts=prompts) is None


def test_prompt_with_by_llm_parameter_is_valid(tmp_path: Path) -> None:
    """`by_llm=True` が 1 件でもあれば `prompt` の宣言は有効。"""
    prompts = _store(tmp_path, hint="ヒント本文")
    catalog = _catalog(
        _spec(
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:hint",),
        )
    )
    assert _validate(catalog, prompts=prompts) is None


# ---- 検査 8 の続き: パラメータ側の穴埋め専用宣言も対象（レビュー指摘・非対称 2） ----
#
# 検査 8 は `spec.prompt` / `spec.prompt_vars` しか見ないため、`by_llm=True` が 0 件でも
# `param(prompt=...)` / `param(max_suggestions=N)` のような**穴埋め段でしか効かない宣言**は
# 「効かないのに宣言されている」まま通る。検査 4 / 5 の走査範囲が `param(prompt=...)` を
# 含むよう拡張されたのと同じ系統の取りこぼしであり、検査 8 も同じ方向へ揃える。


def test_parameter_prompt_without_by_llm_parameter_raises(tmp_path: Path) -> None:
    """`param(prompt=...)` だけの宣言も効果が無いため `ValueError`（非対称 2）。

    セグメントは解決できるため検査 3 は通り、プレースホルダを持たないため検査 4 / 5 も通る。
    検査 8 だけが違反になる構成である。
    """
    prompts = _store(tmp_path, hint="ヒント本文")
    catalog = _catalog(_spec(label="負荷試験", parameters=(param("env", str, prompt="part:hint"),)))
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, prompts=prompts)

    assert type(excinfo.value) is ValueError
    assert "run_load_test" in str(excinfo.value)


def test_parameter_max_suggestions_without_by_llm_parameter_raises() -> None:
    """`max_suggestions>1` だけの宣言も効果が無いため `ValueError`（非対称 2）。

    候補値の上限は予測段でのみ効くため、`by_llm=True` が 0 件なら宣言しても効かない。
    """
    catalog = _catalog(_spec(label="負荷試験", parameters=(param("env", str, max_suggestions=5),)))
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog)

    assert type(excinfo.value) is ValueError
    assert "run_load_test" in str(excinfo.value)


def test_parameter_only_violations_aggregate_into_one_value_error(tmp_path: Path) -> None:
    """パラメータ側だけの違反も 1 つの `ValueError` へ集約する（非対称 2）。"""
    prompts = _store(tmp_path, hint="ヒント本文")
    catalog = _catalog(
        _spec("zeta_action", label="負荷試験", parameters=(param("env", str, prompt="part:hint"),)),
        _spec("alpha_action", label="負荷試験", parameters=(param("env", str, max_suggestions=5),)),
    )
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, prompts=prompts)

    message = str(excinfo.value)
    assert "zeta_action" in message
    assert "alpha_action" in message


def test_parameter_prompt_with_by_llm_parameter_is_valid(tmp_path: Path) -> None:
    """`by_llm=True` が 1 件でもあればパラメータ側の宣言も有効（非対称 2 の回帰防止）。"""
    prompts = _store(tmp_path, hint="ヒント本文")
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, default=30),
                param("note", str, by_llm=True, prompt="part:hint", max_suggestions=3),
            )
        )
    )
    assert _validate(catalog, prompts=prompts) is None


def test_parameter_prompt_on_another_parameter_with_by_llm_is_valid(tmp_path: Path) -> None:
    """穴埋め経路が同一 `ActionSpec` 内にあれば別パラメータの宣言も有効（非対称 2 の回帰防止）。

    判定の単位は `ActionSpec` であり、宣言したパラメータ自身が `by_llm=True` である必要は
    無い（検査 8 の既存スコープと揃える）。
    """
    prompts = _store(tmp_path, hint="ヒント本文")
    catalog = _catalog(
        _spec(
            label="負荷試験",
            parameters=(
                param("env", str, prompt="part:hint", max_suggestions=5),
                param("note", str, by_llm=True),
            ),
        )
    )
    assert _validate(catalog, prompts=prompts) is None


def test_default_max_suggestions_is_not_a_prompt_declaration() -> None:
    """既定の `max_suggestions=1` は宣言と見なさない（非対称 2 の誤検知防止）。

    既定値まで「効かない宣言」に数えると、`by_llm=True` を持たない全アクションが違反になる。
    """
    catalog = _catalog(_spec(label="負荷試験", parameters=(param("env", str, max_suggestions=1),)))
    assert _validate(catalog) is None


def test_plain_parameters_without_by_llm_are_still_valid() -> None:
    """穴埋め専用宣言を持たないパラメータだけの宣言は従来どおり通る（非対称 2 の誤検知防止）。"""
    catalog = _catalog(
        _spec(
            parameters=(
                param("seconds", int, default=30),
                param("target", str, from_context="tenant.id", confirm=True),
            )
        )
    )
    assert _validate(catalog) is None


# ---- 検査 9: LLMFiller.guardrails の登録名が解決できるか ----


def test_unresolvable_guardrail_names_raise_key_error() -> None:
    """解決できない登録名は集約した `KeyError`（`ValueError` ではない）。"""
    catalog = _catalog(_spec())
    with pytest.raises(KeyError) as excinfo:
        _validate(
            catalog,
            guardrail_registry=_FakeGuardrailRegistry("pii"),
            llm_filler=LLMFiller(model="gpt-x", guardrails=("pii", "ghost_a", "ghost_b")),
        )

    assert type(excinfo.value) is KeyError
    message = str(excinfo.value)
    assert "ghost_a" in message
    assert "ghost_b" in message


def test_guardrails_without_registry_raise_value_error() -> None:
    """解決簿そのものが未結線なら `ValueError`（名前が無いのではなく結線の欠落）。"""
    catalog = _catalog(_spec())
    with pytest.raises(ValueError) as excinfo:
        _validate(catalog, llm_filler=LLMFiller(model="gpt-x", guardrails=("pii",)))

    assert type(excinfo.value) is ValueError


def test_registered_guardrail_names_pass() -> None:
    """解決できる登録名だけなら通る。"""
    catalog = _catalog(_spec())
    assert (
        _validate(
            catalog,
            guardrail_registry=_FakeGuardrailRegistry("pii", "jailbreak"),
            llm_filler=LLMFiller(model="gpt-x", guardrails=("pii", "jailbreak")),
        )
        is None
    )


def test_empty_guardrails_without_registry_is_valid() -> None:
    """`guardrails` が空なら解決簿が無くても通る（opt-in）。"""
    catalog = _catalog(_spec())
    assert _validate(catalog, llm_filler=LLMFiller(model="gpt-x")) is None


def test_no_llm_filler_is_valid() -> None:
    """`llm_filler` 未結線でも検査 9 は空振りしない（対象が無いだけ）。"""
    assert _validate(_catalog(_spec())) is None


# ---- WARNING: 埋まる経路が宣言に無いパラメータ（検査種別には数えない） ----


def test_parameter_without_fill_path_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    """`from_context` / `by_llm` / `default` のいずれも無ければ WARNING 1 行・例外なし。"""
    catalog = _catalog(_spec(parameters=(param("seconds", int),), label="負荷試験"))
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = _validate(catalog)

    assert result is None
    records = _warnings(caplog)
    assert len(records) == 1
    assert "seconds" in records[0].getMessage()


def test_filled_by_candidate_suppresses_the_warning(caplog: pytest.LogCaptureFixture) -> None:
    """`filled_by_candidate=True` は明示宣言なので WARNING を出さない。"""
    catalog = _catalog(
        _spec(parameters=(param("seconds", int, filled_by_candidate=True),), label="負荷試験")
    )
    with caplog.at_level(logging.WARNING):
        assert _validate(catalog) is None

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


@pytest.mark.parametrize(
    "declared",
    [
        param("seconds", int, default=30),
        param("seconds", int, from_context="region"),
        param("seconds", int, by_llm=True),
    ],
    ids=["default", "from_context", "by_llm"],
)
def test_declared_fill_paths_produce_no_warning(
    declared: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """埋まる経路が 1 つでも宣言されていれば WARNING は出ない。"""
    catalog = _catalog(_spec(parameters=(declared,), label="負荷試験"))
    with caplog.at_level(logging.WARNING):
        assert _validate(catalog) is None

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_warning_comes_from_module_logger(caplog: pytest.LogCaptureFixture) -> None:
    """WARNING の logger 名は `_validate` モジュール（`logging.getLogger(__name__)`）。"""
    catalog = _catalog(_spec(parameters=(param("seconds", int),), label="負荷試験"))
    with caplog.at_level(logging.WARNING):
        _validate(catalog)

    names = {r.name for r in caplog.records if r.levelno == logging.WARNING}
    assert names == {_LOGGER_NAME}


# ---- 結線が欠けた場合（設計 §3.4a） ----


def test_declared_segments_without_prompts_raise_runtime_error() -> None:
    """セグメント宣言があるのに `prompts` 未結線なら `RuntimeError`（検査の空振りを防ぐ）。"""
    catalog = _catalog(
        _spec(
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:hint",),
        )
    )
    with pytest.raises(RuntimeError) as excinfo:
        _validate(catalog, prompts=None)

    assert type(excinfo.value) is RuntimeError


def test_no_segments_without_prompts_is_valid() -> None:
    """セグメント宣言が 1 件も無ければ `prompts` 未結線でも正常に完了する。"""
    assert _validate(_catalog(_spec())) is None


# ---- lockdown 環境（設計 §3.12） ----


def test_integrity_error_is_not_converted_to_value_error() -> None:
    """`PromptTemplateIntegrityError` は捕捉せず伝播する（集約 `ValueError` に埋めない）。"""
    catalog = _catalog(
        _spec(
            parameters=(param("seconds", int, default=30), param("note", str, by_llm=True)),
            prompt=("part:hint",),
        )
    )
    with pytest.raises(PromptTemplateIntegrityError) as excinfo:
        _validate(catalog, prompts=_IntegrityPromptStore())

    assert type(excinfo.value) is PromptTemplateIntegrityError
    assert not isinstance(excinfo.value, ValueError)
    assert not isinstance(excinfo.value, KeyError)


# ---- 非公開性と SDK 隔離（設計 §3.13 / NFR-1 / NFR-6） ----


def test_validate_symbol_is_not_publicly_exported() -> None:
    """起動時検証の実装関数は公開シンボルではない（窓口は `ActionPlanner.validate`）。"""
    import oai_agentspec.runtime.intent as intent_mod

    assert _PRIVATE_SYMBOL not in intent_mod.__all__
    assert not any(name.startswith("_") for name in intent_mod.__all__)
    with pytest.raises(AttributeError):
        getattr(intent_mod, _PRIVATE_SYMBOL)


def test_validate_module_is_private() -> None:
    """モジュール名は `_` 始まり（`_validate.py`）。"""
    from oai_agentspec.runtime.intent import _validate as validate_mod

    assert validate_mod.__name__ == _LOGGER_NAME
    assert Path(validate_mod.__file__ or "").name == "_validate.py"


def test_validate_module_does_not_import_sdk_or_re() -> None:
    """`agents` / `openai` を import せず `import re` も持たない（NFR-1 / NFR-6）。"""
    from oai_agentspec.runtime.intent import _validate as validate_mod

    text = Path(validate_mod.__file__ or "").read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]
    assert not any(line.startswith(("import agents", "from agents")) for line in lines)
    assert not any(line.startswith(("import openai", "from openai")) for line in lines)
    assert not any(line.startswith("import re") or line.startswith("from re ") for line in lines)
