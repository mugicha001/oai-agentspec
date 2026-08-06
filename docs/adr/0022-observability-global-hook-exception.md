# 0022: オブザーバビリティ有効化のグローバル結線を build-don't-run の第 3 の例外とする

- Status: accepted
- Date: 2026-08-05

## Context

Microsoft Agent 365 オブザーバビリティ連携（トレース）と標準 `logging` の OTel Logs 連携（ログ）を、
既存の `AgentSpec` / `HandoffGraph` / `WorkflowGraph` を無改変のまま有効化できる単一 extra
`observability` として追加するにあたり、有効化の結線点をどこに置くかを検討した。

本連携が用いる 2 つの機構はいずれもプロセス全体に効くグローバル登録である。

- トレース: MS 拡張（`microsoft-agents-a365-observability-extensions-openai`）が公開する
  `OpenAIAgentsTraceInstrumentor().instrument()` による SDK トレーシングへの計装。
  SDK トレーシング全体へのフックであり、`AgentSpec` / `registry` 単位の宣言的 build フローに乗らない。
- ログ: root logger への OTel `LoggingHandler` 付与。root logger はプロセス全体で共有される。

lib の不変条件は build-don't-run（宣言・build-time 検証・薄い結線に徹し、実行は SDK `Runner.run` に
寄せる）であり、既存の例外は `fit_ml_estimator`（ADR 0004）と `failsafe_call`（ADR 0012）の 2 例のみ
である。いずれも「lib 独自の実行ループを持たず、利用者が明示的に呼ぶ 1 点の薄い駆動 / 結線に限定する」
という共通の位置づけで例外化されている。

検討した選択肢:

1. **import 副作用での自動有効化（却下）**: `import oai_agentspec` または extra 導入だけでトレース
   フック登録・root logger 付与を行う。「本体 import は副作用を持たない」不変条件
   （`tests/test_extra_isolation.py` が強制する extra 未導入耐性・遅延 import 境界）と正面から衝突し、
   利用者の logging 構成を無断で書き換える。extra を導入しただけの利用者にもグローバル副作用が及ぶ。
2. **AgentSpec 宣言への組み込み（却下）**: observability をエージェント単位の宣言フィールドとして
   持たせ、build フローに載せる。実体はプロセスグローバルな登録であるため、宣言の見かけ
   （エージェント単位）と効果範囲（プロセス全体）が乖離し、複数 spec に宣言した場合の重複登録・
   「どの宣言が効いているのか」の不透明さを生む。個別エージェントへの組み込みを要求しない本機能の
   要件特性とも不整合。
3. **明示的な有効化関数（採用）**: 利用者が明示的に 1 回呼ぶ薄い結線関数として提供する。lib 独自の
   実行ループ・Runner 代行は一切持たず、既存例外 2 例と同じ位置づけで build-don't-run の例外に
   加えられる。

トレースの登録経路についても併せて検討した。

- **MS 拡張の内部プロセッサを `agents.add_trace_processor(...)` へ直接登録する（却下）**: SDK 既定の
  プロセッサ列を残したまま追加できる利点はあるが、登録対象の `OpenAIAgentsTraceProcessor` は MS 拡張の
  `__all__` に含まれない内部クラスであり、公開契約のないベンダー内部実装へ依存することになる。拡張側の
  リファクタで容易に壊れるうえ、公式の計装 API が提供する初期化順序の保証（構成完了後にプロセッサを
  生成する）を自前で再現する必要がある。
- **公式 API `OpenAIAgentsTraceInstrumentor().instrument()` を使う（採用）**: MS 拡張が公開契約として
  提供する計装 API に乗る。内部で `set_trace_processors` により SDK のトレースプロセッサ列を置換する
  ため、SDK 既定の送信先（OpenAI プラットフォーム）への送信は行われなくなる。この副作用を受け入れる
  代わりに、公開 API のみへ依存する。

## Decision

`runtime/observability` 公開窓口の有効化関数 2 つを build-don't-run の第 3 の例外とする。

- `enable_agent365_tracing(config)`: MS 拡張 `configure()` をラップし、その後に公式 API
  `OpenAIAgentsTraceInstrumentor().instrument()` で SDK トレーシングへ計装を適用する（構成前に計装を
  生成すると MS 拡張側が `RuntimeError` を送出するため、この順序が契約）。計装は内部で
  `set_trace_processors` により SDK のトレースプロセッサ列を置換するため、SDK 既定の送信先へは送られ
  なくなる。冪等化は MS 拡張側へ委譲し（再構成は警告付きで無視され、再計装も no-op）、本関数は独自の
  状態を持たない。構成に失敗した場合・未構成の場合は `RuntimeWarning` で通知して計装せずに戻る。
- `enable_otel_logging(config)`: `LoggerProvider` / `LogExporter` を構築し、OTel `LoggingHandler` を
  root logger へ冪等に付与する（モジュールフラグでプロセス内 1 回に限定。既存ハンドラ・フォーマッタ
  には触れない）。

いずれも有効化は明示的な関数呼び出しのみで行い、import 副作用を持たない（`import oai_agentspec` は
root logger・SDK トレーシングに非接触）。グローバル結線の実装は `_adapters/observability.py` に物理
隔離し、`./CLAUDE.md`「設計の核」の build-don't-run 例外リストを 2 件から 3 件へ更新して逸脱範囲を
grep 的に検知可能な状態に保つ。

付随する設計判断:

- **トレースのエクスポート先切替は MS 拡張 `configure()` へ委譲する**: sidecar / 実 Agent365 API /
  汎用 OTLP / 既定コンソールの切替は MS 拡張の既存仕様（`SpectraExporterOptions` /
  `ENABLE_A365_OBSERVABILITY_EXPORTER` + token_resolver / `ENABLE_OTLP_EXPORTER`）に委譲し、
  oai-agentspec 独自のエクスポータ選択方式・列挙型を新設しない。`Agent365TracingConfig` は
  `configure()` へのパススルー引数のみを保持し、切替仕様の SoT は MS 拡張側に置く。
- **ログの OTLP は併用（additive）とする**: `OtelLoggingConfig.otlp_enabled=True` は既定の
  `ConsoleLogExporter` を置換せず `OTLPLogExporter` を追加する（`LoggerProvider` に
  `LogRecordProcessor` を 2 本構成する）。トレース側 `ENABLE_OTLP_EXPORTER` の併用仕様との対称性を
  保つため。ログ側は Agent365 core に Logs 機構が無いため素の `opentelemetry-sdk` で構築し、宛先は
  「既定コンソール + OTLP 併用追加」のみとする（sidecar / 実 Agent365 API 宛は存在しない。トレース側
  との非対称は usage docs に明記する）。
- **ログの trace 相関は spike で確定させ、不成立時は要件オーナー確認を経る**: `LoggingHandler` が
  付与する trace_id / span_id は OTel current span 文脈由来であり、相関の成立は「MS 拡張の
  `TracingProcessor` が OTel span を開始し context に attach するか」に依存する。spike で実挙動を
  確認し、成立するなら成立条件を docs に明記する。不成立と判明した場合は受け入れ基準（span 存在時の
  trace_id / span_id 付与・example での相関ログ観測）を満たさない状態にあたるため、相関 ID なしの
  フォールバック実装へ自動移行せず、要件オーナーへ「受け入れ基準を修正するか、相関を成立させる代替
  手段（OTel context への手動ブリッジ等）を追加検討するか」を確認して扱いを確定してから後続タスクへ
  進む。
- **`set_tracing_disabled` 検知時は warning とする**: SDK トレーシングが無効化された状態での
  `enable_agent365_tracing` 呼び出しは `warnings.warn(..., RuntimeWarning)` で「トレース未送信」を
  通知し、処理は継続する（例外にしない。送信失敗時も実行を止めないベストエフォート方針と整合させ
  つつ、無警告の黙殺を禁止する）。
- **構成失敗も warning とする**: `configure()` が失敗した場合・`is_configured()` が偽の場合は
  `RuntimeWarning` で「トレース未送信」を通知し、計装せずに戻る（未構成のまま計装を生成すると MS 拡張
  側が `RuntimeError` を送出するため進まない）。観測の失敗で利用者のアプリを停止させないベストエフォート
  方針を、`set_tracing_disabled` 検知と同一の形で適用する。
- **ログ連携の再設定は warning とする**: 冪等化により 2 回目以降の `enable_otel_logging` は適用済みの
  結線を変更しない。初回と異なる設定を渡された場合は `RuntimeWarning` で通知し、「より制限的な設定へ
  変えたつもりが効いていない」という誤解を防ぐ（適用済み設定を後勝ちで上書きはしない）。

## Consequences

- + 既存の `AgentSpec` / `HandoffGraph` / `WorkflowGraph` を無改変のまま、有効化関数を 1 回呼ぶだけで
  トレース・ログが送出される。
- + 逸脱範囲が有効化関数 2 つ・`_adapters/observability.py` 1 ファイルに閉じ、`./CLAUDE.md` の例外文言と
  併せて範囲逸脱を検知できる。
- + import 副作用ゼロのため、extra 未導入耐性・「本体 import は副作用を持たない」不変条件は不変の
  まま保たれる。
- - 公式計装 API がプロセッサ列を置換するため、有効化すると SDK 既定のエクスポート先（OpenAI
  プラットフォーム）へのトレース送信が外れる。トレースの宛先は MS 拡張の構成が単一の決定点になる。
- - 有効化はプロセスグローバルに効き、エージェント単位・run 単位での opt-in / opt-out は表現でき
  ない（グローバル登録という機構特性に由来する意図的なトレードオフ）。
- - ログの trace 相関の成立可否が MS 拡張の実装詳細に依存し、spike 完了まで確定しない（上記の
  要件オーナー確認ゲートで扱いを確定する）。

## Confirmation

本 ADR は設計フェーズで受理されたものであり、以下の強制手段は**実装フェーズで追加する対象**として
実装タスクへ引き継ぐ（設計時点では未実装）。

- 遅延 import 境界（有効化関数を呼ばない限り observability 依存が `sys.modules` に載らないこと）の
  強制手段: `tests/test_extra_isolation.py` の subprocess 隔離テスト。既存
  `::test_importing_package_does_not_force_load_extra_deps` の `_FORBIDDEN_EXTRAS` へ観測系パッケージ
  名を追加する。
- root logger への冪等付与（複数回呼び出しで重複付与・二重出力なし・既存ハンドラ / フォーマッタ非
  接触）の強制手段: `tests/_adapters/test_observability_l1.py`（新設）の冪等・非接触テスト。
- ログ連携の再設定検知（初回と異なる設定での呼び直しが黙殺されず、かつ適用済みの結線が変わらないこと）
  の強制手段: `tests/_adapters/test_observability_l1.py::test_enable_otel_logging_different_settings_second_call_warns`
  / `::test_enable_otel_logging_different_settings_does_not_change_wiring`。
- 構成失敗時の非例外化（`RuntimeWarning` に倒して計装せずに戻ること）の強制手段:
  `tests/_adapters/test_observability_l1.py::test_enable_agent365_tracing_warns_when_configure_returns_false`
  / `::test_enable_agent365_tracing_warns_when_not_configured`。
- 上記は `docs/QUALITY-GUARANTEES.md` に source = ADR 0022 として登録済み（相互参照）。
- グローバル結線の物理隔離（`_adapters/observability.py` に閉じること）と、有効化が明示関数呼び出し
  のみであることは、モジュール構成そのものとコードレビューで担保する。
