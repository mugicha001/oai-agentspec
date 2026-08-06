# オブザーバビリティ連携（Agent 365 トレース + OTel ログ）

## 何を解決するか

既存の `AgentSpec` / `HandoffGraph` / `WorkflowGraph` を無改変のまま、有効化関数を 1 回呼ぶだけで (1) OpenAI Agents SDK のトレーシングに Microsoft Agent 365 拡張がグローバルにフックされ span が送出され、(2) Python 標準 `logging` のログが OTel Logs として root logger 経由で送出されます。エクスポート先は既定でコンソール（実バックエンド・認証不要で確認可能）です。

有効化は明示的な関数呼び出しのみで行われ、import 副作用はありません（`import oai_agentspec` はトレーシング・root logger に非接触）。extra 未導入でもコア import・既存機能は壊れず、有効化 API に触れた場合のみ導入案内の `ImportError` を返します。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| `enable_agent365_tracing(config)` のみ | SDK トレーシングへの MS 拡張フック | span だけ観測したい |
| `enable_otel_logging()` のみ | 標準 `logging` を OTel Logs 化 | 既存ログを構造化して流したい |
| 両方 | span とログの併走（span 存在時は trace_id / span_id で相関） | 実行の全体観測 |
| 既定（コンソール） | Console exporter | ローカル確認・CI・入門 |
| トレースを sidecar / 実 Agent365 API / OTLP へ | MS 拡張 `configure()` の仕様に従う | 実基盤への送出 |
| ログを OTLP 併用 | `OtelLoggingConfig(otlp_enabled=True)` | ログを OTel Collector 等へも送る |

## 使い方

- import: `from oai_agentspec.runtime.observability import (enable_agent365_tracing, Agent365TracingConfig, enable_otel_logging, OtelLoggingConfig)`
- extras: `pip install 'oai-agentspec[observability]'`（`microsoft-agents-a365-observability-extensions-openai` + `opentelemetry-sdk` + `opentelemetry-exporter-otlp-proto-http`。最後の 1 つは別配布物で、ログの OTLP 併用時にのみ使われる）
- 依存 env: 既定のコンソール確認では不要。エクスポート先切替時のみ委譲先（MS 拡張 / OTel SDK）が読む env を設定

```python
import logging

from oai_agentspec.runtime.observability import (
    Agent365TracingConfig,
    OtelLoggingConfig,
    enable_agent365_tracing,
    enable_otel_logging,
)

# プロセスで 1 回だけ呼ぶ（service_name / service_namespace は必須）
enable_agent365_tracing(
    Agent365TracingConfig(service_name="my-app", service_namespace="my-team")
)
enable_otel_logging(OtelLoggingConfig(service_name="my-app"))

logging.getLogger(__name__).info("handoff resolved")  # OTel LogRecord として送出される
# 以降は通常どおり Runner.run(...) するだけで span が送出される
```

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）

### `enable_agent365_tracing(config)`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `config` | `Agent365TracingConfig` | 必須 | トレース連携の宣言的設定。省略できません |

### `Agent365TracingConfig`（frozen）

`service_name` / `service_namespace` が必須で、残りは MS 拡張 `configure()` へのパススルー引数です（sidecar 向け `exporter_options`・実 Agent365 API 向け `token_resolver`・`cluster_category`・`logger_name`・`suppress_invoke_agent_input`）。エクスポータ選択の列挙型は持たず、切替仕様は委譲先（MS 拡張）が正です。

`suppress_invoke_agent_input=True` は本文送出の全面的な抑止手段ではありません（InvokeAgent スパンの `gen_ai.input.messages` 完全一致キーのみが対象で、instructions・chat スパン・ツール入出力は残ります）。本文を落としたい場合は下記「送出されるデータ」を参照してください。

### `enable_otel_logging(config=None)`

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `config` | `OtelLoggingConfig \| None` | `None` | None で既定コンソール出力・INFO レベル |

### `OtelLoggingConfig`（frozen）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `otlp_enabled` | `bool` | `False` | True で既定コンソールに OTLP エクスポータを**追加**（併用。置換しない） |
| `level` | `int` | `logging.INFO` | root logger に付与する `LoggingHandler` のレベル |
| `service_name` | `str \| None` | `None` | OTel Resource の `service.name` |
| `console_json_lines` | `bool` | `False` | True でコンソール出力を 1 行 JSON（JSON Lines）にする。標準出力を「1 行 = 1 レコード」で取り込む収集基盤向け（既定の整形済み出力は 1 レコードが複数行に分かれる）。OTLP 側の形式は変わりません |

## エクスポート先の切替

トレース側とログ側で切替の仕組みが**非対称**です。

- **トレース**: 切替（sidecar / 実 Agent365 API / 汎用 OTLP / 既定コンソール）は MS 拡張 `configure()` の既存仕様に委譲します。`SpectraExporterOptions`（sidecar）・`ENABLE_A365_OBSERVABILITY_EXPORTER` + token_resolver（実 API）・`ENABLE_OTLP_EXPORTER`（OTLP 併用）の詳細は MS 拡張のドキュメントが正であり、本ライブラリは選択方式を持ちません。
- **ログ**: 宛先は「既定コンソール + `otlp_enabled=True` での OTLP 併用追加」のみです。sidecar / 実 Agent365 API 宛は存在しません（Agent365 側に Logs 機構が無いため、素の OTel SDK で構築しています）。OTLP のエンドポイント等は標準 OTEL 環境変数（`OTEL_EXPORTER_OTLP_*`）に従います。

いずれの OTLP も**併用（additive）**であり、コンソール出力を置換しません。

## 送出されるデータ

スパンにはユーザー入力・モデル出力・エージェントの instructions・ツールの引数と結果が載ります。ログは root logger 経由のため、アプリケーション全体と依存ライブラリのログが対象になります。

- **環境変数だけで外部送出が起動する**: トレース側の委譲先（MS 拡張）が `ENABLE_OTLP_EXPORTER` / `ENABLE_A365_OBSERVABILITY_EXPORTER` を直接読むため、本ライブラリへ既定の設定しか渡していなくても環境次第で外部エンドポイントへ送られます。送出可否の最終決定権は委譲先にあります
- **本文を落とす（リダクション）**: MS 拡張が公開する `register_span_enricher` に自前の関数を登録し、`EnrichedReadableSpan` で対象属性を除いたスパンを返します（`enable_agent365_tracing` より前に呼ぶ）。enricher はプロセスで 1 つしか登録できず、本ライブラリはこの枠を使いません。除去対象キーの選び方・enricher が例外を出した場合の挙動を含む詳細は `examples/observability/README.md` を参照してください
- **スパン種別は `gen_ai.operation.name` で見る**: `invoke_agent` / `execute_tool`（ハンドオフを含む）/ `chat` / `chain` が現れます。OTel 標準の `kind` は全て `INTERNAL` になるため分類には使えません

## 判断軸

- まず既定コンソールで span とログが出ることを確認し、その後にエクスポート先を切り替える
- トレースだけ・ログだけの片方利用も可能（依存も副作用も独立）
- 将来 Log Analytics 等の実バックエンドへ入れたい場合は汎用 OTLP パス（Collector 経由）で到達する

## 落とし穴

- **root logger へのグローバル副作用**: `enable_otel_logging()` は root logger にハンドラを追加します（既存ハンドラ・フォーマッタには触れず、複数回呼んでも冪等）。アプリ側で logging 構成を厳密に管理している場合は付与されるハンドラとレベルを把握した上で有効化してください
- **2 回目以降の設定変更は効かない**: 初回と異なる設定で `enable_otel_logging` を呼び直しても適用済みの結線は変わりません（黙殺せず `RuntimeWarning` で通知されます）。設定を変える場合はプロセス起動時に 1 度だけ有効化してください
- **既定のトレース送信先が置き換わる**: MS 拡張の計装は SDK のトレースプロセッサ列を差し替えるため、SDK 既定の送信先（OpenAI プラットフォーム）へは送られなくなります
- **`set_tracing_disabled(True)` との衝突**: SDK トレーシングを無効化しているとトレース生成自体が止まり、MS 拡張をフックしても span は送出されません。この状態で `enable_agent365_tracing(config)` を呼ぶと `RuntimeWarning` で通知されます（例外にはなりません）
- **環境変数 `OPENAI_AGENTS_DISABLE_TRACING` による無効化は検知できません**（SDK が内部フラグを最初のスパン生成まで更新しないため）。警告が無いことは送信されていることの保証になりません
- **構成失敗時も例外にしません**: MS 拡張の `configure()` が失敗した場合・未構成の場合は `RuntimeWarning` を出して計装せずに戻ります（観測の失敗で利用者のアプリを止めないベストエフォート方針）
- **構成失敗時は既定の送信先が生き続けます**: 計装しないため SDK 既定のトレースプロセッサ列はそのまま残り、既定のエクスポート先（OpenAI プラットフォーム）へは送信が継続します。このとき本文抑止（`suppress_invoke_agent_input`）と span enricher は MS 拡張の計装経路にのみ効くため**適用されません**。ユーザー入力・instructions・ツール入出力を外部へ出したくない場合は、構成失敗時の送出も止まるよう `agents.set_tracing_disabled(True)` を併用するか、SDK 側の機微データ抑止設定を使ってください
- **`configure()` が成功しても Agent 365 へ届かない場合があり、本連携はそれを検知しません**: 委譲先の内部フォールバック（エクスポート先が解決できないときのコンソール出力）や、同一プロセスで 2 回目以降の有効化が既存構成を保ったまま無視されること、アプリ側で用意済みのトレースプロバイダへ相乗りしたときのそのプロバイダの状態などにより、警告が出ないまま Agent 365 へ届かないことがあります（ベストエフォート）
- **MS 拡張は import しただけで root logger にハンドラを 1 つ追加します**（拡張側が `logging.basicConfig` を呼ぶため）。本ライブラリからは制御できません。書式を自分で決めたい場合は有効化より前に `logging.basicConfig(...)` を呼んでください（`force=True` は本連携のハンドラも消すため使わない）
- **相関はアクティブ span がある時だけ**: span が無い時点のログは無効 trace_id 相当となり、相関情報は付きません
- 有効化はプロセスグローバルに効きます。エージェント単位の opt-in / opt-out はできません

## example

- `examples/observability/01_trace_and_logs.py` — スパンとログの相関（スパン内のログに trace_id / span_id が入る）
- `examples/observability/02_json_lines_and_handoff.py` — 1 行 JSON 出力と、ハンドオフを含む構成でのスパン種別

運用上の詳細（リダクション・ログ書式の変更・実測されるスパン属性）は `examples/observability/README.md` にまとめてあります。

## 参照

- 詳細設計: `docs/architecture.md`（オブザーバビリティ連携節）
- 設計判断の経緯: `docs/adr/0022-observability-global-hook-exception.md`
- 具体例と運用詳細: `examples/observability/README.md`

## 次

[../runtime/realtime.md](../runtime/realtime.md) — Realtime エージェント
