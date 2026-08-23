# Fine-Tuning データセット整形・検証（oai-agentspec[finetune]）の使い方

OpenAI / Azure OpenAI のマネージド fine-tuning API（SFT / DPO）向けのデータセット整形（chat /
preference 形式への変換）・持ち込み JSONL の検証と、学習ジョブの投入・状態照会・完了待機を
提供する層。

データセット変換・検証は `agents` / `openai` を import せずネットワークにも一切触れない
**純データ層**で、ジョブ管理も SDK 接触を `_adapters/finetune.py` の 1 ファイルに閉じている。
**`06` / `07` / `08` を除く全 example は API キーなしで実行できる**（この 3 本のみ実 API へ
ジョブを投入するため接続情報が必要で、従量課金が発生する）。

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
| `05_job_body_preview.py` | 引数がジョブ作成リクエストのどのフィールドになるかを可視化する（SFT / DPO / tools / 衝突検出） | `submit_job`（記録用の疑似 client） |
| `06_submit_job_live.py` | 実 API へ SFT ジョブを投入し状態を照会する（**課金あり**） | `submit_job` + `get_job` |
| `07_submit_dpo_job_live.py` | 実 API へ DPO ジョブを投入し状態を照会する（**課金あり**） | `to_dpo_dataset` + `submit_job(method="dpo")` |
| `08_submit_tools_job_live.py` | ツール定義つきの学習データを実 API へ投入する（SFT / DPO 切替・**課金あり**） | `ToolRegistry` + `tools=` + `submit_job` |

実行（API キー不要）:

```bash
uv run python examples/finetune/01_sft_dataset.py
uv run python examples/finetune/02_dpo_dataset.py
uv run python examples/finetune/03_tools_from_registry.py
uv run python examples/finetune/04_validate_byo_jsonl.py
uv run python examples/finetune/05_job_body_preview.py
```

実行（接続情報が必要・**従量課金が発生する**）:

```bash
uv run python examples/finetune/06_submit_job_live.py                      # SFT
uv run python examples/finetune/07_submit_dpo_job_live.py                  # DPO
uv run python examples/finetune/08_submit_tools_job_live.py                # tools つき SFT
uv run python examples/finetune/08_submit_tools_job_live.py --method dpo   # tools つき DPO
```

いずれも実行前に確認プロンプトを出す。標準入力が対話的でない環境（パイプ実行・CI 等）では
確認を取れないため中止する。その場合に実行したいときは `--yes` を付ける（課金が発生する）:

```bash
uv run python examples/finetune/06_submit_job_live.py --yes
```

`06` / `07` / `08` は学習ファイルのアップロードとジョブ作成を実 API に対して行う。実行前に
確認プロンプトを出し、費用を抑えるため最小データ（10 件）・1 エポックを既定にしている。

最小の設定は次の 2 行（gpt-4.1-mini で SFT を試す場合）:

```bash
FINETUNE_SFT_BASE_MODEL=gpt-4.1-mini-2025-04-14
FINETUNE_TRAINING_TYPE=GlobalStandard              # 任意・Azure 専用（下記の注意を参照）
```

`FINETUNE_TRAINING_TYPE` は Azure 専用で、OpenAI 直接続では設定されていても送らない。値は
リソース / リージョンによって受理されるものが異なり、使えない値を指定するとプラットフォームは
`The fineTuningJob field is required.` という一見無関係なエラーを返す（実測）。`Developer` が
使えない環境では `GlobalStandard` を試すこと。

環境変数は 3 ブロックに分かれる（詳細は `.env.example`）。

| ブロック | 変数 | 未設定のとき |
|---|---|---|
| 学習対象モデル | `FINETUNE_SFT_BASE_MODEL` / `FINETUNE_DPO_BASE_MODEL` | 各 example がエラー表示して終了 |
| FT 共通 | `FINETUNE_PROVIDER` | `AZURE_OPENAI_FINETUNE_ENDPOINT` があれば azure、無ければ `EXAMPLES_LLM_PROVIDER` |
| FT 共通（**Azure 専用**） | `FINETUNE_TRAINING_TYPE` | フィールド自体を送信しない（OpenAI 直接続では設定されていても送らない） |
| FT 共通 | `AZURE_OPENAI_FINETUNE_API_VERSION` | `2025-04-01-preview`（推論用は継承しない） |
| 接続（Azure） | `AZURE_OPENAI_FINETUNE_ENDPOINT` / `_API_KEY` | 推論用（`AZURE_OPENAI_*`）へフォールバック |
| 接続（OpenAI） | `OPENAI_FINETUNE_API_KEY` | `OPENAI_API_KEY` へフォールバック |
| 接続（OpenAI） | `OPENAI_FINETUNE_BASE_URL` | `https://api.openai.com/v1`（`OPENAI_BASE_URL` は継承しない） |

**推論と FT で接続先を分けられる**。「推論は OpenAI 互換ゲートウェイ・FT は Azure」のような
構成では、`AZURE_OPENAI_FINETUNE_ENDPOINT` / `_API_KEY` を設定すれば FT だけ Azure へ向く
（`FINETUNE_PROVIDER` の明示も可）。推論用ゲートウェイは Files / fine_tuning API を持たない
ことがあり、`OPENAI_BASE_URL` を継承すると 404 になるため、OpenAI 経路の base_url も継承しない。

api-version の既定を推論用から分けているのは、`trainingType` の指定が公式手順で dated 版
（`2025-04-01-preview`）を要求するため。v1 preview 方式を試す場合は
`AZURE_OPENAI_FINETUNE_API_VERSION=preview` を明示する。

`07`（DPO）は対応モデルが SFT より狭い点に注意する。本ライブラリは対応モデル一覧を保持しない
ため、非対応の組み合わせはプラットフォームのエラー（`API_ERROR`）で判明する。SFT の後に DPO を
重ねる 2 段構成は、`06` が返した `model_ref` を `FINETUNE_DPO_BASE_MODEL` へ渡して `07` を
実行する（利用者が 2 回のジョブとして実行する）。

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

## ジョブ管理の要点（05 / 06 / 07 / 08）

- **設定は解釈せず透過する**: lib はハイパーパラメータの構造・対応モデル一覧・`training_type` の
  許容値・`suffix` の長さ制約（OpenAI は 64 文字 / Azure は 18 文字・ドット不可）を保持しない。
  値の妥当性はプラットフォームが判定し、そのエラーは理由文言を保全した `API_ERROR` で返る。
- **未指定のフィールドは送らない**: 最小引数なら送信 body のキーは `model` / `training_file` /
  `method` の 3 つだけ。学習完了後の自動デプロイを有効化するフィールドも lib は付加しない
  （有効にしたい場合は `extra_body=` で明示する）。
- **`method` の写像**: `"sft"` は API 側の `supervised` へ写像する（`"dpo"` はそのまま）。
  `hyperparameters=` は写像後の type 値の下（`method.supervised.hyperparameters`）へ入る。
  未知のメソッド識別子や Mapping 形の `method` は非解釈で透過する。
- **`train` / `val` の受理形**: `str` は**アップロード済みのファイル id**、`Path` はローカル
  JSONL、`DatasetBuildResult` / レコード列はメモリ内容をアップロードする。`validate_dataset` の
  `source` では `str` がファイルパスを意味するのと逆なので注意する。
- **`tools` はデータ側に入る**: `to_sft_dataset(tools=...)` / `to_dpo_dataset(tools=...)` で渡した
  ツール定義は**アップロードされる JSONL のレコード**へ入り（SFT はレコード直下の `tools`、DPO は
  `input.tools`）、ジョブ作成リクエストの body 直下には現れない。body 直下に載るのは
  `training_type` / `suffix` / `seed` のようなジョブ設定だけである（`05` で実際の送信内容を確認できる）。
  学習データと推論時のツール定義を同じ `ToolRegistry` から出せるため、両者の一致が構造的に保たれる。
- **`extra_body=` の衝突検出**: lib が組み立てるキー（実際に指定した引数の担当キーのみ）と
  交差したら送信前に `FineTuneError`（`CONFIG_MISSING`）で失敗する。省略した引数のキーは
  占有しないので `extra_body` 側から指定してよい。
- **`wait_job` は opt-in・`timeout` 必須**: lib 内で唯一のポーリングループ（`poll_interval`
  既定 30 秒）。無限待機の経路を持たない。詳細は `docs/adr/0031-wait-job-polling-isolation.md`。
- **Azure の `model_ref` はデプロイ前参照**: 推論に使うには Azure 側でのデプロイ操作が別途必要
  （本ライブラリのスコープ外・利用者責任）。
- **接続先の分離**: `06` / `07` / `08` は既定で推論用の接続情報を再利用するが、FT が使える
  リージョンは限られるため別リソースになることがある。その場合は
  `AZURE_OPENAI_FINETUNE_ENDPOINT` / `_API_KEY`（OpenAI 直接続なら
  `OPENAI_FINETUNE_API_KEY`）を設定する。未設定なら推論用へフォールバックする
  （`examples/_shared/_azure.py` の `build_finetune_client`）。
  **接続先を分けたら API キーも必ず分けて設定する**: エンドポイント / base_url は推論用から
  継承しないのに API キーは継承するため、キーだけ未設定にすると推論用のキーが別ホストへ
  送信される（例: 推論を OpenAI 互換ゲートウェイへ向けている構成で FT 用キーを設定しないと、
  ゲートウェイの仮想キーが `api.openai.com` へ送られる）。

## RFT（強化学習ファインチューニング）を使う場合

RFT は Azure でも利用できる（`o4-mini` は GA、`gpt-5` は招待制）。本ライブラリは `method` を
非解釈で透過するため投入自体は通るが、SFT / DPO とは扱いが変わるので example は用意していない。

- **`method` は Mapping 形で渡す**: RFT は grader（報酬関数）が必須で、それを `method` の中へ
  埋める必要がある。`hyperparameters=` 引数だけでは表現できないため、`method={...}` を丸ごと
  渡す escape hatch を使う。

  ```python
  await submit_job(
      client,
      train=Path("rft_train.jsonl"),
      val=Path("rft_val.jsonl"),          # RFT は検証データが必須
      model="o4-mini-2025-04-16",
      method={
          "type": "reinforcement",
          "reinforcement": {
              "grader": {
                  "type": "string_check",
                  "name": "answer",
                  "operation": "eq",
                  "input": "{{ sample.output_text }}",
                  "reference": "{{ item.solution }}",
              },
              "hyperparameters": {"reasoning_effort": "medium"},
          },
      },
  )
  ```

- **データは持ち込み JSONL で渡す**: RFT のデータは `messages` の最後を `user` ロールにし、
  grader が参照する追加フィールド（上例の `solution` 等）を持つ。この形式は `to_sft_dataset` /
  `to_dpo_dataset` の対象外で、`validate_dataset` も `sft` / `dpo` のみを検証する。整形済みの
  JSONL を `Path` かアップロード済みファイル id で渡すこと。
- **課金上限**: RFT は 1 ジョブあたり $5,000 に達すると自動で一時停止し、チェックポイントが
  作られる（再開すると以降は上限なしで課金が続く）。本ライブラリはこの挙動を検知・制御しない。

## スコープ外

デプロイ / ホスティング（Azure の control plane 操作）は含まない。完成モデルは `model_ref`
（モデル id の文字列）を返すところまでで、`AgentSpec` の model への流し込みは利用者が行う。
会話ログ（SDK `Session`）からのデータセット生成も本 examples 群の範囲外。データ分割は
`oai_agentspec.runtime.lightning.train_val_split` がレコード列にもそのまま使える
（finetune 側に分割 API は持たない）。
