"""プロンプトテンプレートのロードと合成。

`PromptStore` は利用側が渡す root 配下の `.md`（YAML frontmatter）/ `.yaml` をロードし、
共通ベース・パーツ・エージェント個別テンプレートを連結して instructions を生成する。
合成結果（静的 str または動的 callable）を `AgentSpec.instructions` に渡して使う。
`agents` には依存しない。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

import yaml

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(frozen=True)
class PromptLayout:
    """合成セグメント（base/part/agent）のディレクトリ構成。

    `PromptStore` の必須引数。プリセット（暗黙の既定）を設けず、利用側に 3 つの
    ディレクトリ名を必ず明示させることで「規約フォルダを勝手に仮定して無音で誤合成
    する」ミスを防ぐ。各ディレクトリ名に空文字 "" を渡すと root 直下を探索する。

    例:
        PromptLayout(base="base", parts="parts", agents="agents")     # 規約構成
        PromptLayout(base="common", parts="snippets", agents="roles") # 既存構成に合わせる
        PromptLayout(base="", parts="", agents="")                    # 全て root 直下

    Attributes:
        base: base:<name> セグメントのサブディレクトリ名。
        parts: part:<name> セグメントのサブディレクトリ名。
        agents: agent:<name> セグメントのサブディレクトリ名。
    """

    base: str
    parts: str
    agents: str


@dataclass(frozen=True)
class PromptTemplate:
    """テンプレート文字列ラッパー（本文 + メタデータ）。

    openai-agents の `agents.Prompt`（id 参照型 TypedDict）とは別物であり、
    本クラスは本文を保持するローカルテンプレートである。
    """

    name: str
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        """frontmatter の version（未指定は "0"）。"""
        return str(self.metadata.get("version", "0"))

    def render(self, **vars: Any) -> str:
        """`${var}` を安全置換してレンダリングする（不足キーはそのまま残す）。"""
        return Template(self.body).safe_substitute(**vars)


class PromptResolutionError(KeyError):
    """合成セグメントの解決に失敗したことを示す例外。"""


class PromptStore:
    """root 配下のテンプレートをロードし、instructions を合成するストア。

    ディレクトリ構成は `layout`（PromptLayout）で**明示必須**。3 つのディレクトリ名を
    必ず指定する（暗黙の既定なし）。

    規約レイアウトの例:

        <root>/base/main.md     全 main 共通ベース
        <root>/base/sub.md      全 sub 共通ベース
        <root>/parts/<name>.md  使い回しパーツ
        <root>/agents/<name>.md エージェント個別

    各セグメントのサブディレクトリ配下はさらに階層化してよい。`agent`/`part`/`base` 名は
    サブディレクトリ配下を再帰探索し stem 一致で解決する（例: `agents/billing/refund.md`
    を `agent="refund"` で取得）。同 stem が複数ある場合は曖昧エラーとなるため
    `agent="billing/refund"` のようにサブパスを含めて指定する。

    使い方::

        store = PromptStore("prompts", PromptLayout(base="base", parts="parts", agents="agents"))
        spec = AgentSpec(
            name="triage",
            instructions=store.compose(agent="triage", base="main", parts=["style"], vars=VARS),
        )
    """

    _EXTS = (".md", ".yaml", ".yml")
    _SEGMENT_KINDS = ("base", "part", "agent")

    def __init__(self, root: str | Path, layout: PromptLayout):
        """ストアを生成する。

        Args:
            root: プロンプトファイルのルートディレクトリ（利用側が指定）。
            layout: 合成セグメントのディレクトリ構成（必須。暗黙デフォルトなし）。
        """
        self.root = Path(root)
        self.layout = layout
        self._segment_dirs = {"base": layout.base, "part": layout.parts, "agent": layout.agents}
        self._cache: dict[str, PromptTemplate] = {}
        # ``_preload()`` 末尾で True に遷移し、以降の ``reload()`` を禁止する。lockdown 後の
        # 「cache only / disk 不参照」契約を維持し、disk 改竄 → reload → 改竄プロンプト流入を防ぐ。
        self._locked: bool = False

    # ------------------------------------------------------------------
    # ファイルロード
    # ------------------------------------------------------------------
    def _load_file(self, path: Path) -> PromptTemplate:
        text = path.read_text(encoding="utf-8")
        name = path.stem
        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(text) or {}
            body = data.pop("body", "")
            return PromptTemplate(name=name, body=body.strip(), metadata=data)
        m = _FRONTMATTER_RE.match(text)
        if m:
            meta = yaml.safe_load(m.group(1)) or {}
            body = text[m.end() :]
        else:
            meta, body = {}, text
        return PromptTemplate(name=name, body=body.strip(), metadata=meta)

    def _find_flat(self, name: str) -> Path | None:
        for ext in self._EXTS:
            path = self.root / f"{name}{ext}"
            if path.exists():
                return path
        return None

    def _find_in(self, subdir: str, name: str) -> Path | None:
        """セグメントのサブディレクトリ配下から name を探す。

        - `subdir` が空文字（フラット指定）の場合は root 直下のみを非再帰で探す。
        - `subdir` 指定時はその配下を再帰探索し stem 一致で解決する。
        - name に "/" を含む場合は subdir 起点の相対パスとして完全一致で探す。

        Raises:
            PromptResolutionError: 再帰探索で同 stem が複数見つかり曖昧な場合。
        """
        base = self.root / subdir if subdir else self.root
        if "/" in name or not subdir:
            for ext in self._EXTS:
                path = base / f"{name}{ext}"
                if path.exists():
                    return path
            return None
        matches: list[Path] = []
        for ext in self._EXTS:
            matches.extend(sorted(base.rglob(f"{name}{ext}")))
        if len(matches) > 1:
            rels = sorted(str(m.relative_to(base)) for m in matches)
            raise PromptResolutionError(
                f"名前 {name!r} が {subdir} 配下で複数見つかり曖昧です: {rels}"
                "（サブパスを含めて指定してください。例: agent='billing/refund'）"
            )
        return matches[0] if matches else None

    # ------------------------------------------------------------------
    # 単体テンプレート（フラット配置）
    # ------------------------------------------------------------------
    def get(self, name: str) -> PromptTemplate:
        """フラット配置（`<root>/<name>.*`）のテンプレートをロードする。

        Raises:
            KeyError: テンプレートが見つからない場合。
            PromptTemplateIntegrityError: ``_locked=True``（lockdown 後）に cache miss
                した場合。disk アクセスを行わず即時 raise する（cache only 契約維持）。
        """
        if name in self._cache:
            return self._cache[name]
        if self._locked:
            # lockdown 後の disk アクセスを禁止（manifest 未掲載のテンプレ要求）。
            from .integrity import PromptTemplateIntegrityError

            raise PromptTemplateIntegrityError(
                f"lockdown 後の cache miss（manifest 未掲載）: {name!r}"
            )
        path = self._find_flat(name)
        if path is None:
            raise KeyError(f"prompt not found: {name} (looked in {self.root})")
        template = self._load_file(path)
        self._cache[name] = template
        return template

    def render(self, name: str, **vars: Any) -> str:
        """フラット配置テンプレートをレンダリングする。"""
        return self.get(name).render(**vars)

    # ------------------------------------------------------------------
    # セグメント解決と合成
    # ------------------------------------------------------------------
    def _load_segment(self, segment: str) -> PromptTemplate:
        """セグメント参照（"base:main" / "part:style" / "agent:triage"）をロードする。

        Raises:
            PromptResolutionError: 記法が不正、またはファイルが見つからない場合。
            PromptTemplateIntegrityError: ``_locked=True``（lockdown 後）に cache miss
                した場合。disk アクセスを行わず即時 raise する（cache only 契約維持）。
        """
        kind, _, name = segment.partition(":")
        if kind not in self._segment_dirs or not name:
            raise PromptResolutionError(
                f"不正なセグメント参照: {segment!r}（base:<name> / part:<name> / agent:<name>）"
            )
        cache_key = f"{kind}:{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        if self._locked:
            # lockdown 後の disk アクセスを禁止（manifest 未掲載のセグメント要求）。
            from .integrity import PromptTemplateIntegrityError

            raise PromptTemplateIntegrityError(
                f"lockdown 後の segment cache miss（manifest 未掲載）: {segment!r}"
            )
        path = self._find_in(self._segment_dirs[kind], name)
        if path is None:
            subdir = self._segment_dirs[kind]
            where = self.root / subdir if subdir else self.root
            raise PromptResolutionError(
                f"セグメント {segment!r} のファイルが見つかりません (searched under {where})"
            )
        template = self._load_file(path)
        self._cache[cache_key] = template
        return template

    def _segments(
        self,
        agent: str | None,
        base: str | None,
        parts: Sequence[str],
        layout: Sequence[str] | None,
    ) -> list[str]:
        if layout is not None:
            return list(layout)
        segments: list[str] = []
        if base:
            segments.append(f"base:{base}")
        segments += [f"part:{p}" for p in parts]
        if agent:
            segments.append(f"agent:{agent}")
        if not segments:
            raise PromptResolutionError(
                "合成対象がありません（agent / base / parts / layout のいずれかを指定してください）"
            )
        return segments

    def _render(self, segments: Sequence[str], vars: dict[str, Any]) -> str:
        return "\n\n".join(self._load_segment(s).render(**vars) for s in segments)

    def compose(
        self,
        agent: str | None = None,
        *,
        base: str | None = None,
        parts: Sequence[str] = (),
        layout: Sequence[str] | None = None,
        vars: dict[str, Any] | Callable[[Any], dict[str, Any]] | None = None,
    ) -> str | Callable[[Any, Any], str]:
        """共通ベース・パーツ・個別テンプレートを連結した instructions を返す。

        デフォルト順は base -> parts -> agent。`layout` 指定時はセグメント参照
        （"base:main" 等）の列をそのまま順序として使う。各セグメントは frontmatter を
        除いた本文を `${var}` で置換し `\\n\\n` 連結する。

        `vars` の型で静的/動的が決まり、戻り値は `Agent.instructions`（`str | callable`）
        にそのまま渡せる:

        - `vars` が dict / None: ビルド時に置換した **静的な str** を返す。
        - `vars` が callable（`RunContextWrapper -> dict`）: 各 run で ctx から変数を
          生成して合成する **2 引数 callable `(context, agent) -> str`** を返す。

        Args:
            agent: agent:<name> セグメント（個別テンプレート名）。
            base: base:<name> セグメント（共通ベース名。例 "main" / "sub"）。
            parts: part:<name> セグメント名の列。
            layout: セグメント参照の明示列（指定時は agent/base/parts を無視）。
            vars: `${var}` 置換変数（dict）または ctx から dict を返す関数。

        Returns:
            静的合成なら str、動的合成なら `(context, agent) -> str` の callable。

        Raises:
            PromptResolutionError: セグメント参照が不正、または解決できない場合。
        """
        segments = self._segments(agent, base, parts, layout)
        if callable(vars):
            vars_fn = vars

            def instructions(context: Any, agent_: Any) -> str:
                return self._render(segments, vars_fn(context))

            return instructions
        return self._render(segments, dict(vars or {}))

    # ------------------------------------------------------------------
    # その他
    # ------------------------------------------------------------------
    def all(self) -> dict[str, PromptTemplate]:
        """フラット配置（root 直下）の全テンプレートをロードして返す。

        合成セグメントのキャッシュ（"agent:triage" のようなコロン付きキー）は含めない。
        """
        for path in self.root.glob("*"):
            if path.suffix in self._EXTS and path.stem not in self._cache:
                self._cache[path.stem] = self._load_file(path)
        return {k: v for k, v in self._cache.items() if ":" not in k and "/" not in k}

    def reload(self) -> None:
        """テンプレートキャッシュをクリアする。次回 render/compose でファイル再読込。

        ``lockdown(store=...)`` 経由で ``_preload()`` が走った後（``_locked=True``）に呼ぶと
        ``PromptTemplateIntegrityError`` を raise する。lockdown 後の「cache only / disk
        不参照」契約を維持するため。継続検証は ``lockdown()`` の再呼び出しで行う。

        Raises:
            PromptTemplateIntegrityError: lockdown 経由で固定済みの場合（``_locked=True``）。
        """
        if self._locked:
            # 遅延 import で循環回避（integrity → prompts は TYPE_CHECKING 内のみ）。
            from .integrity import PromptTemplateIntegrityError

            raise PromptTemplateIntegrityError(
                "lockdown 後の PromptStore.reload は禁止です: 継続検証は lockdown() の "
                "再呼び出しで行ってください"
            )
        self._cache.clear()

    # ------------------------------------------------------------------
    # 整合性検証（lockdown 内部用）
    # ------------------------------------------------------------------
    def _verify_integrity(self, checks: list[Callable[[], None]]) -> None:
        """利用者の検知関数群を順次発火する（fail-closed・最初の違反で残りスキップ）。

        ``lockdown`` 内部のヘルパとして呼ばれ、公開 API ではない。各 check 関数は
        違反時に ``IntegrityError`` 系の例外を raise する契約。

        Args:
            checks: 順次発火させる検知関数のリスト。

        Raises:
            Exception: いずれかの check が raise した例外をそのまま伝播する。
        """
        for check in checks:
            check()

    def _preload(self) -> None:
        """root 配下の全テンプレートを eager-load して ``_cache`` を充填する。

        ``lockdown`` の段 2（store verify + preload）から呼ばれる非公開メソッド。
        以降の ``get`` / ``compose`` は cache のみを参照し disk アクセスを行わない。

        探索対象は ``root`` 配下を再帰的に走査し、サポート拡張子（``_EXTS``）を持つ
        ファイルのみを対象とする。cache キーは次の 3 種を登録する:

        - flat key（``"name"``）— root 直下のフラット配置に対応。
        - segment full path key（``"agent:billing/refund"``）— サブディレクトリ配下の
          フルパス指定（``_find_in`` の "/" 含む経路）と一致。
        - segment stem alias（``"agent:refund"``）— サブディレクトリ配下を ``rglob``
          したときの **stem が unique** な場合のみ登録。``_find_in`` の再帰 stem 解決と
          同じ振る舞いを cache 経由でも保証する。複数候補がある stem は alias を入れず、
          ``compose(agent="refund")`` 等の曖昧呼び出しを既存の ``PromptResolutionError``
          経路に流す（disk 走査時の挙動と一致）。

        末尾で ``_locked=True`` をセットし、以降の ``reload()`` を禁止する（lockdown 後の
        cache only 契約を維持するため）。

        Raises:
            OSError: テンプレートファイルの読み取りに失敗した場合。
        """
        # 1) flat / segment full path キーを cache 充填しつつ、segment 配下の stem 出現数を集計。
        # 集計マップ: (kind, stem) -> [(name_path, file_path), ...]
        segment_paths: dict[str, list[tuple[str, Path]]] = {kind: [] for kind in self._segment_dirs}
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix not in self._EXTS:
                continue
            try:
                relative = path.relative_to(self.root)
            except ValueError:
                continue
            # flat 配置（root 直下）。
            if relative.parent == Path("."):
                template = self._cache.get(path.stem)
                if template is None:
                    template = self._load_file(path)
                    self._cache[path.stem] = template
                # 空 subdir のセグメント種別（flat layout）は root 直下を直接参照するため
                # （_find_in("", name) と同経路）、segment key も同テンプレで登録する。
                for kind, subdir in self._segment_dirs.items():
                    if not subdir:  # 空 subdir = flat layout
                        seg_key = f"{kind}:{path.stem}"
                        if seg_key not in self._cache:
                            self._cache[seg_key] = template
                continue
            # サブディレクトリ配下: セグメント種別を判定する。
            top_segment = relative.parts[0]
            for kind, subdir in self._segment_dirs.items():
                if subdir and top_segment == subdir:
                    name = relative.relative_to(subdir).with_suffix("").as_posix()
                    full_key = f"{kind}:{name}"
                    if full_key not in self._cache:
                        self._cache[full_key] = self._load_file(path)
                    segment_paths[kind].append((name, path))
                    break

        # 2) 各 (kind, stem) で unique なら stem alias キーで cache に再登録する。
        # _find_in の再帰 stem 解決と同じく、stem 衝突時は alias を入れない。colon を含む
        # stem は cache キーの ``kind:name`` 解釈と曖昧化するため alias を作らない。
        for kind, entries in segment_paths.items():
            stem_count: dict[str, int] = {}
            for name, _ in entries:
                stem = name.rsplit("/", 1)[-1]
                stem_count[stem] = stem_count.get(stem, 0) + 1
            for name, path in entries:
                stem = name.rsplit("/", 1)[-1]
                if stem == name:
                    continue  # フルパスと alias が同一（既に full_key で cache 済み）。
                if stem_count[stem] != 1:
                    continue  # 曖昧 stem は alias を作らない。
                if ":" in stem:
                    continue  # colon を含む stem は alias を作らない（曖昧化回避）。
                alias_key = f"{kind}:{stem}"
                if alias_key not in self._cache:
                    self._cache[alias_key] = self._load_file(path)

        # 3) lockdown 後の cache only 契約を固定する（以降 reload 禁止）。
        self._locked = True


def dynamic_prompt(
    extractor: Callable[[Any], dict[str, Any]],
) -> Callable[[Any], dict[str, Any]]:
    """ctx から id/version/variables を生成する DynamicPromptFunction を作る。

    `agents.Prompt` は id 参照型 TypedDict（本文を持たない）であり、本ヘルパーは
    ローカルテンプレート本文を扱わない。OpenAI Responses API 専用で、`AgentSpec.prompt`
    に渡す。

    Args:
        extractor: `GenerateDynamicPromptData.context`（RunContextWrapper）から
            `{"id": ..., "version": ..., "variables": ...}` を返す関数。

    Returns:
        `GenerateDynamicPromptData -> dict`（agents.Prompt TypedDict）。
    """

    def generate(data: Any) -> dict[str, Any]:
        return extractor(data.context)

    return generate


__all__ = [
    "PromptStore",
    "PromptLayout",
    "PromptTemplate",
    "PromptResolutionError",
    "dynamic_prompt",
]
