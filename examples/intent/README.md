# 意図予測（oai-agentspec[intent]）の使い方

`runtime/intent/` は「LLM を用いた意図予測」の汎用土台。入力（現在発話・履歴・run_context）から
`IntentPrediction`（信頼度付き候補群 + 一貫性判定）を返す **分類器のみ** を提供し、分類結果で
何をするか（下流エージェント選択・実行分岐）は application 側の責務にする（PolicyEngine は
lib スコープ外）。

## インストール（extra）

```bash
pip install 'oai-agentspec[intent]'
```

追加依存は `pydantic>=2` のみ。extra 未導入でも `import oai_agentspec.runtime.intent` は
壊れず（PEP 562 遅延再エクスポート）、シンボルへのアクセス時に初めて pydantic を要求する。

## 実行

Azure OpenAI の環境変数（`AZURE_OPENAI_*` -- `examples/_shared/_azure.py` 参照）を `.env` 等で
設定し、任意の例を直接実行:

```bash
uv run python examples/intent/01_basic_classification.py
```

各例は分類実行時に LLM へ流れる合成プロンプトを `[SYSTEM]` / `[USER]` として stdout に表示する。

## 例の一覧

| # | ファイル | パターン |
|---|---|---|
| 01 | `01_basic_classification.py` | `intent_classifier_from_model` 1 行ヘルパの最小例。`IntentPolicy` 3 カテゴリ |
| 02 | `02_with_session_history.py` | `agents.SQLiteSession` を `IntentQuery.history` に渡し、`DefaultContextBuilder` が抽出した履歴は `IntentContext.history_items` として保持され、SDK 経由で multi-turn として LLM に届く。utterance を省略する履歴のみモードのデモも含む |
| 03 | `03_custom_prompt.py` | `include_policy_in_system=False` で prompt callable が system 込みで全制御。fenced block で間接プロンプトインジェクションの信頼境界を明示 |
| 04 | `04_custom_context_builder.py` | 独自 `ContextBuilder` を差し替え、`run_context`（`UserProfile`）を素通しし prompt callable が user_content に反映。`DefaultIntentClassifier` を直接組み立て。`include_rationale_in_prompt=True` で判断理由を生成させるオプションのデモも兼ねる |
| 05 | `05_intent_based_routing.py` | 分類結果で下流 `AgentSpec` を SDK `Runner` に dispatch する統合フロー。分類と下流応答のレイテンシを分離計測 |

## 信頼境界（読んでおくべき）

- **`IntentContext.history_items` は sanitize / role フィルタされない生の SDK 互換 dict tuple**。
  既定構成では `history_items` は `prompt` callable を通らず SDK の `Runner.run(input=list)` に直接
  流れる（callable の責務は「現在発話の user content 生成のみ」）。そのため `role: "system"` /
  `"developer"` を含む item が履歴に混入すると、そのまま LLM に system 権限メッセージとして届き、
  分類器プロンプトの前段に注入される（role 昇格経路）。tool 出力や retrieval 結果など外部由来
  コンテンツが履歴に混ざる利用形態では、**独自 `ContextBuilder` を差し替えて role の allowlist
  （`user` / `assistant` のみ通す等）を適用するか、items を文字列化 + fence した専用 item に整形して
  から渡す**必要がある。prompt callable 側の fence は現在発話にのみ有効で、履歴由来インジェクション
  には効かない。
- **`pydantic.ValidationError`** は LLM 生出力の一部を例外メッセージに含む。外部露出（ログ・API
  レスポンス）前に握り替えを検討する。
- **allowlist 除外は silent**（`metadata` に記録されない）。除外があった場合は
  `logger.warning`（logger 名 `oai_agentspec.runtime.intent._llm`）で通知される。詳細な追跡は
  SDK Span を参照する。
- **rationale 生成は既定 off**（`IntentPolicy.include_rationale_in_prompt=False`）。LLM の生成
  トークン・レイテンシを抑えるため、分類器の判断理由が欲しい場合のみ `True` に切り替える。
  parser 側は常に rationale を optional として受け入れる。
- **低精度・高速モデルでの運用**: プロンプトには固定のタスク指示行と「JSON のみ出力」制約が
  含まれ、コードフェンス付き応答はパース前に自動で剥がされる。それでも分類が不安定な場合は
  `extra_instructions` に few-shot 例を 1-2 件追加すると安定しやすい。
- **履歴のみでの分類**: `IntentQuery(history=session)`（`utterance` 省略・既定 `""`）で、現在発話
  なしに会話履歴だけを multi-turn として送り「ここまでの会話」を分類できる。utterance と
  history の両方が空の場合は `ValueError` になる。固定タスク指示行は「ユーザー発話を分類」の
  文言のままなので、履歴のみモード向けに調整したい場合は `extra_instructions` で補足する
  （例: 「発話がない場合はここまでの会話全体から次の意図を予測してください」）。

## 位置づけ

- **build-don't-run**: `LLMCandidateGenerator` / `DefaultIntentClassifier` は宣言・検証と薄い結線
  のみ。実行は SDK `Runner.run`（例 05 参照）に寄せる。
- **プロンプト非同梱**: `IntentPolicy.render_prompt()` は事前定義値（カテゴリ / 信頼度 / 出力形式 /
  制約）から手書き 4 セクションを組み立てる薄い骨格で、prompt engineering は含まない
  （`model_json_schema` は prompt に埋め込まない）。カスタムプロンプトは `extra_instructions` の
  先頭挿入か、`include_policy_in_system=False` + `prompt` callable による全制御で行う。
- **SDK 隔離 (NFR-1)**: lib 本体は `agents` / `openai` を import しない
  （`_adapters/intent.py` の関数内遅延 import に閉じる）。
