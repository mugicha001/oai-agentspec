# オブザーバビリティ連携（oai-agentspec[observability]）の使い方

宣言したエージェント（`AgentSpec` / `HandoffGraph` / `WorkflowGraph`）の実行トレースを
Microsoft Agent 365 オブザーバビリティへ送り、あわせて標準 `logging` のログを OpenTelemetry Logs
として送出する。実行中のログには実行中スパンの trace_id / span_id が付くため、トレースとログを
突き合わせて追跡できる。

宣言物は read-only で扱い変更しない。有効化は利用者が明示的に 1 回呼ぶ薄い結線で、lib 独自の
実行ループは持たない（実行は SDK `Runner.run` のまま）。

## インストール（extra）

```bash
pip install 'oai-agentspec[observability]'
```

トレース連携（Agent 365 拡張）とログ連携（OpenTelemetry Logs）を単一 extra でまとめて導入する。
未導入のまま有効化 API を呼ぶと、導入方法を案内する `ImportError` になる。

## 最小例（コンソール確認のみ・Agent 365 の認証情報なし）

```python
from agents import Runner, set_tracing_disabled
from oai_agentspec import AgentRegistry, AgentSpec
from oai_agentspec.runtime.observability import (
    Agent365TracingConfig,
    OtelLoggingConfig,
    enable_agent365_tracing,
    enable_otel_logging,
)

set_tracing_disabled(False)  # トレーシングが無効だと何も送信されない
enable_agent365_tracing(
    Agent365TracingConfig(service_name="my-app", service_namespace="my-team")
)
enable_otel_logging(OtelLoggingConfig(service_name="my-app"))
```

エクスポート先を指定しなければトレース・ログとも標準出力へ書き出されるため、Agent 365 の実
サービスや認証情報なしで動作を確認できる。

## エクスポート先の切り替え

トレース側の切り替えは Agent 365 拡張の仕様に委譲しており、本ライブラリは独自の選択方式を
持たない。`Agent365TracingConfig` の `token_resolver` / `exporter_options` / `cluster_category`
がそのまま渡る。

ログ側は Agent 365 拡張がログ機構を持たないため素の OpenTelemetry Logs で組む。宛先はコンソール
と OTLP の 2 つで、`otlp_enabled=True` はコンソールを**置換せず併用追加**する（接続先は
OpenTelemetry 標準の環境変数で解決される）。

```python
enable_otel_logging(OtelLoggingConfig(service_name="my-app", otlp_enabled=True))
```

## 標準出力をログ収集基盤で拾う場合（1 行 JSON）

コンソール出力は既定で整形済み JSON（`indent=4`）のため、1 レコードが 20 行以上に分かれる。
コンテナの標準出力を「1 行 = 1 レコード」で取り込む収集基盤（Azure Monitor Agent / Fluent Bit 等）
では 1 つのログが複数レコードに分割され、`trace_id` と本文が別レコードになって相関で絞れなくなる。

`console_json_lines=True` でログのコンソール出力を 1 行 JSON（JSON Lines）にできる。

```python
enable_otel_logging(OtelLoggingConfig(service_name="my-app", console_json_lines=True))
```

```
{"body": "LINE-CHECK", "severity_text": "INFO", "trace_id": "0x...", "resource": {...}}
```

**ただしスパン側は 1 行化できない**。スパンのコンソール出力は Agent 365 の構成関数が内部で組む
エクスポータが担っており、整形方法を指定する引数が公開されていないため、本ライブラリからは
制御できない。標準出力の収集でトレースも扱いたい場合は、トレース側を OTLP で直接送る
（`ENABLE_OTLP_EXPORTER`）のが本来の経路になる。

## 送出されるデータ（外部エクスポータを有効にする前に必ず確認する）

以下の内容が送出対象になる。`otlp_enabled=True` や `exporter_options` / `token_resolver` を
設定した時点で外部へ出るのはもちろんだが、**外部送出は設定を渡さなくても環境変数だけで起動する**。
トレース側は委譲先の Agent 365 が `ENABLE_OTLP_EXPORTER` / `ENABLE_A365_OBSERVABILITY_EXPORTER` を
直接読むため、本ライブラリへ既定の設定を渡していても環境次第で外部エンドポイントへ送られる
（本ライブラリ本体は環境変数を読まないが、送出可否の最終決定権は委譲先にある）。コンソール出力も、
stdout がログ収集基盤へ転送される環境では外部送出と同義になる。

- **スパンに載るもの**: ユーザー入力・モデル出力・エージェントの instructions（system メッセージ）・
  ツールの引数と結果。
- **ログに載るもの**: root logger へハンドラを付けるため、アプリケーション全体のログに加えて
  依存ライブラリのログも対象になる（実測では httpx がリクエスト URL を出力する）。

### `suppress_invoke_agent_input` の実際の効果

`Agent365TracingConfig.suppress_invoke_agent_input=True` は抑止手段として不十分である。除去される
のは InvokeAgent スパンの `gen_ai.input.messages` 属性（完全一致キー）だけで、instructions は
接頭辞付きキー（`gen_ai.input.messages.0.*`）で残り、chat スパンの入出力とツール入出力も送出
され続ける。

このフラグは**本ライブラリのものではなく Agent 365 SDK の `configure()` の引数**で、除去処理も
SDK 側にある。本ライブラリは値を保持して渡すだけなので、除去範囲を変えることはできない。

### 本文を落とす（span enricher によるリダクション）

プロンプトや応答の本文そのものを送出対象から外したい場合は、Agent 365 SDK が公開している
span enricher に自前のリダクション関数を登録する。全スパン（chat / execute_tool を含む）が
エクスポート前に通るため、対象キーを指定して除去できる。

```python
from microsoft_agents_a365.observability.core import register_span_enricher
from microsoft_agents_a365.observability.core.exporters.enriched_span import EnrichedReadableSpan

_SENSITIVE = ("gen_ai.input.messages", "gen_ai.output.messages")


def redact(span):
    """本文を含む属性をエクスポート前に除去する。"""
    keys = {k for k in (span.attributes or {}) if k.startswith(_SENSITIVE)}
    if not keys:
        return span
    return EnrichedReadableSpan(span, extra_attributes={}, excluded_attribute_keys=keys)


register_span_enricher(redact)   # enable_agent365_tracing() より前に呼ぶ
```

注意点:

- **enricher はプロセスで 1 つしか登録できない**（登録済みなら `RuntimeError`）。本ライブラリは
  この枠を使わない方針で、リダクションの基準（どのキーを落とすか・削除かマスクか）は組織ごとに
  異なるため利用者側に委ねる。
- 除去対象は上記のキーだけではない。`gen_ai.execution.payload`（エンドポイント URL）や
  ツールの引数・結果（`gen_ai.tool.call.arguments` / `gen_ai.tool.call.result`）も必要に応じて
  加える。実際に何が載るかは `examples/observability/02_json_lines_and_handoff.py` の出力で確認できる。
- enricher が例外を送出した場合、SDK は元のスパンをそのまま使う（つまり**除去されずに送出される**）。
  リダクションを確実にしたいなら関数内で例外を出さない実装にする。
- **enricher は Agent 365 の計装経路にのみ効く**。`configure()` が失敗して `enable_agent365_tracing`
  が計装せずに戻った場合、SDK 既定のトレースプロセッサ列がそのまま残り既定のエクスポート先へ
  送信が継続するが、その経路には enricher も `suppress_invoke_agent_input` も適用されない
  （警告文でもこの旨を通知する）。構成失敗時も送出を止めたいなら
  `agents.set_tracing_disabled(True)` を併用する。

これでも足りない場合は、エクスポート先の選択そのもの（外部へ送らない）で制御する。

## コンソールのログ書式を変える（プロジェクト名を入れる等）

有効化すると、標準エラー出力に次の形式で人間可読のログが流れる。

```
INFO:microsoft_agents_a365.observability.core.config:Creating new TracerProvider for a365 observability.
INFO:examples.observability:agent run starting
```

これは `logging.basicConfig()` の既定書式（`%(levelname)s:%(name)s:%(message)s`）で、**Agent 365 拡張が
import 時に `basicConfig()` を呼ぶ**ために付くハンドラの出力である（本ライブラリが付けるものではない）。

書式を変えたい場合は、**有効化より前に自分で `logging` を設定する**。`basicConfig()` は root に
ハンドラが既にあると何もしないため、先に設定しておけば拡張側の呼び出しが no-op になり、拡張自身の
ログも含めて自分の書式が適用される。

```python
import logging

# 1) 先に自分の書式で設定する
logging.basicConfig(level=logging.INFO, format="%(levelname)s:[my-project]:%(name)s:%(message)s")

# 2) その後で有効化する
enable_agent365_tracing(Agent365TracingConfig(service_name="my-project", service_namespace="my-team"))
enable_otel_logging(OtelLoggingConfig(service_name="my-project"))
```

```
INFO:[my-project]:microsoft_agents_a365.observability.core.config:Creating new TracerProvider for a365 observability.
INFO:[my-project]:examples.observability:agent run starting
```

有効化した後に変えたい場合は、既存ハンドラの formatter を差し替える。

```python
for h in logging.getLogger().handlers:
    if isinstance(h, logging.StreamHandler):
        h.setFormatter(logging.Formatter("%(levelname)s:[my-project]:%(name)s:%(message)s"))
```

注意点:

- **`logging.basicConfig(..., force=True)` は使わない**。`force=True` は root の既存ハンドラを
  すべて削除するため、ログ連携の `LoggingHandler` も一緒に消える。OTel Logs への送出が黙って
  止まり、ハンドラ数は元のままなので気づきにくい。
- `force` を付けない `basicConfig(format=...)` は、root にハンドラがある状態では**何も起きない**
  （no-op）。有効化の後から書式だけ指定しても効かないのはこのため。
- ここで変わるのは標準エラー出力の人間可読ログだけで、OTel Logs（標準出力の JSON）の内容は
  変わらない。構造化ログ側でサービスを識別するには `OtelLoggingConfig(service_name=...)` を使う
  （`resource.attributes["service.name"]` として載る）。本ライブラリは利用者が設定したハンドラや
  フォーマッタを書き換えないため、この書式変更は利用者側で明示的に行う必要がある。

## 落とし穴

- **root logger に付けた Filter は、子 logger 由来のレコードには適用されない**。Python の
  `logging` は発生元 logger の Filter だけを適用し、祖先の Filter を再適用しないため、
  「root に redaction Filter を付ける」方式の秘匿は本連携のハンドラを素通りする。秘匿は各 logger
  もしくは各ハンドラ側に持たせる。
- **トレーシングを無効化したままだと何も送られない**。`examples/_shared/_azure.py` の
  `azure_model()` / `azure_client()` は内部で `set_tracing_disabled(True)` を呼ぶため、これらを
  使う場合はモデル構築の後に `set_tracing_disabled(False)` で戻す。`set_tracing_disabled(True)`
  による無効化は有効化 API が検知して `RuntimeWarning` を出す。ただし環境変数
  `OPENAI_AGENTS_DISABLE_TRACING` による無効化は**検知できない**（SDK が内部フラグを最初の
  スパン生成まで更新しないため）。警告が無いことは送信されていることの保証にならない。
- **既定のトレース送信先が置き換わる**。Agent 365 の計装は SDK のトレースプロセッサ列を
  差し替えるため、SDK 既定の送信先（OpenAI プラットフォーム）へは送られなくなる。
- **ログ連携は root logger に作用する**。アプリケーション全体のログ（依存ライブラリのものを
  含む）が送出対象になる。既存のハンドラ・フォーマッタ・登録順は変更せず追加のみ行う。
- **root logger のレベルは変更しない**。既定のままだと INFO 以下がハンドラへ届かないため、
  必要に応じて `logging.getLogger().setLevel(logging.INFO)` を利用者側で設定する。
- **Agent 365 拡張は import しただけで root logger にハンドラを 1 つ追加する**（拡張側が
  `logging.basicConfig` を呼ぶため）。本ライブラリからは制御できない。

## example

いずれも Azure OpenAI の環境変数（`AZURE_OPENAI_*`。`examples/_shared/_azure.py` を参照）が必要。
Agent 365 の実サービスや認証情報は不要で、出力はコンソールに閉じる。

### 01: トレースとログの相関

```bash
uv run python examples/observability/01_trace_and_logs.py
```

スパンとログがコンソールへ出力され、ツール実行中に出したログには実行中スパンと同じ trace_id /
span_id が入る。スパン外で出したログには相関 ID が付かない（偽の相関を作らない）。

### 02: 1 行 JSON とハンドオフ時のスパン種別

```bash
uv run python examples/observability/02_json_lines_and_handoff.py
```

`console_json_lines=True` でログが 1 行 JSON になる様子と、ハンドオフを含む 2 エージェント構成で
スパン種別（`gen_ai.operation.name`）が複数現れる様子を確認できる。実行例:

| スパン | `gen_ai.operation.name` |
|---|---|
| `invoke_agent triage` / `invoke_agent billing` | `invoke_agent` |
| `handoff to billing` | `execute_tool`（ハンドオフはツール呼び出しとして表現される） |
| `chat <model>` | `chat` |
| `turn` / `Agent workflow` | `chain` |

OTel 標準の `kind` は全て `INTERNAL` になるため、分類には `gen_ai.operation.name` を使う。
ハンドオフ先のスパンには `graph_node_parent_id` が付き、どのエージェントから引き継がれたかが分かる。

この example の出力は「送出されるデータ」の実例にもなっている。スパン属性にシステムプロンプト・
ユーザー入力・モデル応答・エンドポイント URL・トークン使用量がそのまま載ることを確認できる。
