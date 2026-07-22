# プロンプト合成（PromptStore）

## 何を解決するか

プロンプト本体をコードにハードコードすると、レビュー・A/B・多言語対応が困難になります。`PromptStore` は利用側 root 配下の Markdown を `base -> parts -> agent` の順で合成し、`AgentSpec.instructions` に渡します。ライブラリはプロンプトを同梱しません。

`vars` で `${var}` プレースホルダを埋め、`RunContextWrapper -> dict` の callable を渡せば run 毎の動的合成も可能です。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| デフォルト合成 | `compose(agent, base, parts)` = base -> parts -> agent 順 | 共通ベース + 差し込み + 個別で組み立てる |
| `layout=[...]` で順序上書き | `layout=["agent:x", "base:main", "part:safety"]` | agent を先頭に置く等の順序調整 |
| `vars=dict` | 静的な値注入 | tenant 名・会社名など固定文字列 |
| `vars=callable` | `RunContextWrapper -> dict` | run 毎に user / plan で切り替え |

## 使い方

- import: `from oai_agentspec import PromptStore, PromptLayout, PromptTemplate, dynamic_prompt`
- extras: なし
- 依存 env: なし

```python
from pathlib import Path
from oai_agentspec import AgentSpec, PromptLayout, PromptStore

store = PromptStore(Path("prompts"), PromptLayout(base="base", parts="parts", agents="agents"))
spec = AgentSpec(
    name="triage",
    instructions=store.compose(
        agent="triage", base="main", parts=["style", "safety"],
        vars={"company": "AgentSpec Inc."},
    ),
)
```

ディレクトリ構成:

```
prompts/
├── base/    # main.md, sub.md
├── parts/   # style.md, safety.md
└── agents/  # triage.md, billing.md
```

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `PromptLayout`（frozen・全 3 引数）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `base` | `str` | 必須 | base:<name> セグメントのサブディレクトリ名（空文字で root 直下） |
| `parts` | `str` | 必須 | part:<name> セグメントのサブディレクトリ名 |
| `agents` | `str` | 必須 | agent:<name> セグメントのサブディレクトリ名 |

### `PromptStore.__init__`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `root` | `str \| Path` | 必須 | プロンプトファイルのルート |
| `layout` | `PromptLayout` | 必須 | 合成セグメントのディレクトリ構成 |

### `PromptStore.compose`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `agent` | `str \| None` | `None` | agent:<name> セグメント |
| `base` | `str \| None` | `None` | base:<name> セグメント（kw_only） |
| `parts` | `Sequence[str]` | `()` | part:<name> セグメント列（kw_only） |
| `layout` | `Sequence[str] \| None` | `None` | 明示セグメント列（指定時 agent/base/parts を無視・kw_only） |
| `vars` | `dict[str, Any] \| Callable[[Any], dict[str, Any]] \| None` | `None` | `${var}` 置換値（dict なら静的 str、callable なら 2 引数 callable を返す・kw_only） |

### `PromptTemplate`（frozen・3 引数のためコメントで列挙）

`name: str` / `body: str` / `metadata: dict[str, Any] = {}`。`.render(**vars)` と `.version` プロパティを持つ。

### `dynamic_prompt(extractor)`

引数は `extractor: Callable[[Any], dict[str, Any]]` の 1 個のみ（詳細はコード docstring 参照）。

## 判断軸

- 共通文言は **`base` / `parts`** に切り出し、agent 固有のみ `agents/<name>.md` に置く
- コードから静的に注入できるなら **`vars=dict`**、run 毎に変えたい場合のみ **`vars=callable`** を使う
- `layout=` で順序を触りたくなったら、まず `base -> parts -> agent` の並びで書けないか再検討する（可読性が下がる）

## 落とし穴

- `PromptLayout(base="", parts="", agents="")` でフラット構成も可能だが名前衝突に注意
- lib にプロンプト文字列をハードコードしない（`runtime/intent` の固定文のみ例外）
- `lockdown` 経由で `_preload` すると `reload()` は `PromptTemplateIntegrityError` になる（disk 再読込禁止）

## 参照

- 詳細設計: `docs/architecture.md`（プロンプト合成節）
- 具体例: `examples/basic/composition.py` / `examples/basic/custom_layout.py` / `examples/prompts/`

## 次

[tools.md](./tools.md) — ToolRegistry と HITL 宣言
