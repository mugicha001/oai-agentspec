# Fine-Tuning データセット整形・検証（oai-agentspec[finetune]）の使い方

OpenAI / Azure OpenAI のマネージド fine-tuning API（SFT / DPO）向けのデータセット整形（chat /
preference 形式への変換）と、持ち込み JSONL の検証を提供する層。データセット変換・検証は
`agents` / `openai` を import せずネットワークにも一切触れない**純データ層**であり、
本 examples 群は**API キー・`.env` なしで実行できる**（`examples/_shared/_azure.py` は使わない）。

## インストール

```bash
pip install 'oai-agentspec[finetune]'
```

## example 一覧

| ファイル | 内容 | 主な道具 |
|---|---|---|
| `01_sft_dataset.py` | ケース列を SFT（chat 形式）データセットへ変換し、保存・検証まで通す | `to_sft_dataset` + `save` + `validate_dataset` |
| `02_dpo_dataset.py` | `DpoCase` / plain dict を DPO（preference 形式）データセットへ変換する | `DpoCase` + `to_dpo_dataset` |
| `03_tools_from_registry.py` | `ToolRegistry` 登録ツールを学習データの `tools=` へそのまま渡す | `ToolRegistry` + `ToolSpec` + `to_sft_dataset(tools=...)` |
| `04_validate_byo_jsonl.py` | 持ち込み JSONL の検証・`raise_on_invalid` / `skip_missing` の挙動 | `validate_dataset` + `DatasetValidationReport` |

実行（API キー不要）:

```bash
uv run python examples/finetune/01_sft_dataset.py
uv run python examples/finetune/02_dpo_dataset.py
uv run python examples/finetune/03_tools_from_registry.py
uv run python examples/finetune/04_validate_byo_jsonl.py
```

## 使い方の要点

- **入力の二形受理**: `input` / 出力側は文字列（1 件のメッセージへ包む）または messages 形式の
  リスト（複数ターン・非改変透過）のいずれも受ける。
- **`system=`**: `to_sft_dataset` の全レコード先頭へ挿入する（`input` リスト内に system が
  既にあると競合エラー）。`to_dpo_dataset` は `system=` を持たず、`input` リスト内の system
  透過で表現する。
- **`tools=`**: plain dict と `ToolRegistry.<name>` が返す SDK `FunctionTool` の混在リストを
  渡せる。`FunctionTool` は `name` / `params_json_schema` 属性のダックタイピングで検出し、
  公式 tools 定義形式へ写像する。学習データの tools 定義と推論時に Agent へ渡すツール定義を
  同じ Registry から出せる（`03_tools_from_registry.py`）。
- **`weight`**: SFT の assistant メッセージにのみ付けられる（整数 0 / 1）。loss masking したい
  メッセージへ明示する（暗黙補完はしない）。
- **保存は opt-in**: `DatasetBuildResult.save(path)` を呼ばない限り何も書き込まない。
  examples はリポジトリを汚さないよう一時ディレクトリ（`tempfile.TemporaryDirectory()`）へ
  書き出している。
- **検証は fail-closed**: `validate_dataset(source, method="sft"|"dpo")` は違反ゼロのときのみ
  `ok=True`。`raise_on_invalid=True` を明示したときのみ `FineTuneError` を送出する
  （既定は `DatasetValidationReport` を返すのみ）。
- **`skip_missing`**: `to_sft_dataset` / `to_dpo_dataset` に渡すと、必須フィールド欠落等の
  ケースを既定エラーにせず除外し、`DatasetBuildResult.skipped` に件数報告する。

## スコープ外

段階 1（本 examples 群が示す範囲）はデータセット整形・検証のみで、ジョブ投入（`submit_job` 等）
は含まない。データ分割は `oai_agentspec.runtime.lightning.train_val_split` がレコード列にも
そのまま使える（finetune 側に分割 API は持たない）。
