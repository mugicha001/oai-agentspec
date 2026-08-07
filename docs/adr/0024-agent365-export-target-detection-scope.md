# 0024: Agent 365 トレースの到達先について、config だけから誤検知なく確定できる範囲のみを警告する

- Status: accepted
- Date: 2026-08-06

## Context

`enable_agent365_tracing(config)` は MS 拡張の `configure()` の戻り値と `is_configured()` で構成の
成否を判定し、偽なら `RuntimeWarning` を出して計装せずに戻る（ADR 0022 の決定）。しかし
「構成は成功したのに Agent 365 実サービスへは届かない」状態が別に存在し、この状態は無警告で
成功したように見える。観測基盤としては「送っているつもりで送れていない」が最も気付きにくい失敗
であるため、開示だけで足りるかを再検討した。

### 上流の到達先は公開 API から判別できない

上流はエクスポート先を 3 分岐で決める（`microsoft_agents_a365/observability/core/config.py:164-191`）。

1. `exporter_options` が `SpectraExporterOptions` なら sidecar（OTLP）。
   `ENABLE_A365_OBSERVABILITY_EXPORTER` は意図的に無視される（`config.py:167`）。
2. そうでなく `is_agent365_exporter_enabled()` が真かつ `exporter_options.token_resolver` が非 `None`
   なら実 Agent 365 API。
3. それ以外は `ConsoleSpanExporter` へフォールバックし、上流は WARNING ログのみを出す。

`configure()` は分岐 3 でも `True` を返し（`config.py:223`）、`is_configured()` は
`self._tracer_provider is not None` を返すだけである（`config.py:225-227`）。実測でも 3 分岐すべてで
両者が真になることを確認した。したがって戻り値は到達の証拠にならない。

### 実測で判明した追加の失敗経路

上流は `exporter_options` が渡された場合はそれを素のまま使い、トップレベル引数の `token_resolver` /
`cluster_category` を参照しない（`config.py:149-154`。上流 docstring は両引数を Deprecated と明記
している。`config.py:75-78`）。分岐 2 が見るのは `exporter_options.token_resolver` のみであるため、
利用者が認証手段を渡していても、`exporter_options` を併用しただけでコンソールへ落ちる。

実 SDK に対する実測結果（ケースごとに別プロセスで `configure()` を 1 回だけ呼び、上流が構築した
exporter の型を観測。span を生成しないため外部通信は発生しない）。

| ケース | 渡した引数 | env フラグ | 実際の exporter | `configure()` | `is_configured()` |
|---|---|---|---|---|---|
| A | `token_resolver` のみ | ON | `_Agent365Exporter`（実 API） | True | True |
| B | `token_resolver` + `exporter_options`（resolver 無し） | ON | `ConsoleSpanExporter` | True | True |
| A | `token_resolver` のみ | OFF | `ConsoleSpanExporter` | True | True |
| C | `exporter_options=object()` | ON / OFF とも | 未構築（`AttributeError`） | False | True |

ケース B が本 ADR の検知対象である。env フラグが有効で resolver も渡しているにもかかわらず、
`exporter_options` を併用しただけでコンソールへ落ち、戻り値からは一切判別できない。
ケース A（フラグ OFF）とケース B は外から見て完全に同一の症状（同一の上流 WARNING 文言・同一の
戻り値・同一の exporter 型）であり、上流ログ文言も両者を区別しない（"not enabled **or**
token_resolver not set"）。上流ログの捕捉は原因の切り分けに使えない。
ケース C は既存の構成失敗判定が捕まえるため本 ADR の対象外である。

### 検知の制約

- 到達先の判定に必要な `is_agent365_exporter_enabled()` は上流の `__all__` に含まれない非公開関数
  であり（`core/exporters/utils.py:301`。`core/__init__.py` / `core/exporters/__init__.py` の
  いずれの `__all__` にも無く、パッケージ内では `config.py` からしか参照されない）、直接呼び出しは
  できない。ADR 0022 が `OpenAIAgentsTraceProcessor` を「MS 拡張の `__all__` に含まれない内部クラス」
  として却下した先例と同カテゴリである。
- OpenTelemetry には `TracerProvider` に登録済みの span processor / exporter を列挙する公開 API が
  存在しない（内部の `_active_span_processor._span_processors` のみ）。
- `./CLAUDE.md` の「本体（`_adapters` を含む）は環境変数に依存しない」不変条件があり、env フラグを
  直読できない。

### コンソール出力は「意図された既定」でもある

usage docs は「エクスポート先は既定でコンソール（実バックエンド・認証不要で確認可能）」と明記して
おり（`docs/usage/ops/observability.md:5,16`）、`examples/observability/01_trace_and_logs.py` は
`token_resolver` も `exporter_options` も渡さずこの既定を使う。同じ分岐 3 が「意図した既定」でも
「意図せぬ未達」でもあるという非対称が本件の核心であり、「コンソールへ落ちたら警告」という素朴な
実装は文書化済みの正常系すべてで誤検知になる。

### 検討した選択肢

1. **docs 開示のみ（却下）**: 実測で存在が確定したケース B（意図が明示されているのに未達）を
   機械的に塞げない。
2. **`effective_resolver is None` の全体を警告（却下）**: 意図された既定利用（options 未指定）と、
   バッチ調整目的の options 単独指定で誤検知する。上流はバッチ処理パラメータを全分岐で使うため
   （`config.py:156-162` -> `:197-201`）、コンソール出力のままバッチ挙動を調整したい利用者は
   `Agent365ExporterOptions` を渡す以外の手段がない。動いている構成へ警告を出すことは、警告を
   無視する習慣を作る。
3. **`getattr(options, "token_resolver", None) is None` で判定（却下）**: `SpectraExporterOptions` が
   当該属性を持たないため sidecar 構成を巻き込み、案 2 と同じ誤検知も残る。
4. **到達先を返す診断 API を新設（却下）**: 公開 API 契約（`__all__`）を増やす割に、最も重要な
   「非 Spectra かつ実効 resolver あり」のケースは env 次第であり「判定不能」を返すしかない。
5. **`Agent365TracingConfig` へ意図宣言フィールドを追加（却下）**: config 契約の変更を伴い、
   ADR 0022 の「`configure()` へのパススルー引数のみを保持する」決定と衝突する。env 依存のケースは
   宣言と照合できない。
6. **env フラグを読んで完全判定（却下）**: 全ケースを判定できるが、env 非依存の不変条件に正面から
   抵触する。本ライブラリには「env 由来の要因は検知せず docs で開示する」受理済みの先例がある
   （`OPENAI_AGENTS_DISABLE_TRACING` による無効化。`_adapters/observability.py:159-161` /
   `docs/usage/ops/observability.md:105`）。要件オーナーの判断により env は読まないことが確定した。
7. **`token_resolver` 属性の有無と値のみを読み、resolver が捨てられる構成に限って警告（採用）**:
   誤検知ゼロで、実測で存在が確定した実害を塞ぐ。追加の公開シンボル・config フィールド・env 参照・
   上流非公開 API 参照はいずれも不要。
8. **lib が resolver を `exporter_options` へ注入して救済（却下）**: 上流の引数合成規則を lib 側で
   再現することになり、上流仕様の変更で黙って壊れる。利用者が渡したオブジェクトを改変する副作用も
   持つ。

## Decision

1. **検知範囲を、config だけから誤検知なく確定できる範囲に限定する**。具体的には
   「トップレベル `token_resolver` を渡しているのに、`token_resolver` 属性を持つ `exporter_options`
   （Agent365 形式）を併用したため上流が当該 resolver を参照しない」構成のみを `RuntimeWarning` で
   通知する。意図された既定（`exporter_options` 未指定）・sidecar 構成・`exporter_options` 側に
   resolver が設定済みの構成・バッチ調整目的の options 単独指定では警告しない。
2. **判定はダックタイピングで行い、`isinstance` による型判別はしない**。実測事実として
   `Agent365ExporterOptions` は `token_resolver` 属性を常に持ち（既定 `None`。
   `core/exporters/agent365_exporter_options.py:41`）、`SpectraExporterOptions` は当該属性を持たない
   （`core/exporters/spectra_exporter_options.py:26-56`）。これにより `_adapters` へ上流型を持ち込ま
   ずに sidecar を除外できる。
   属性の読み取りは**専用の番兵を既定値とする `getattr` で 1 回だけ行い、あらゆる例外を吸収する**。
   `hasattr` は使わない: `hasattr` が吸収するのは `AttributeError` のみであり、`exporter_options` は
   利用者が渡す不透明値で property や `__getattr__` を持ちうるため、それ以外の例外が
   `enable_agent365_tracing` から伝播して**観測の構成判定の失敗がアプリを停止させてしまう**
   （観測の失敗でアプリを止めないベストエフォート方針に反する。従来は上流 `configure()` の
   `except Exception` が吸収して「構成失敗の `RuntimeWarning` + 継続」で着地していたため、
   `hasattr` 実装は可用性の退行にあたる）。既定値に `None` ではなく専用の番兵を使うのは、
   「属性が無い」（sidecar 構成・options 未指定 = 警告不要）と「属性はあるが `None`」
   （認証手段が失われる構成 = 警告対象）を区別する必要があるためである。属性値が取得できない
   場合は番兵へ倒し、判定不能として**警告しない**。
3. **`exporter_options` の解釈は 1 属性の有無と値のみに限定する**。ADR 0022 は
   「`Agent365TracingConfig` は `configure()` へのパススルー引数のみを保持し、切替仕様の SoT は
   MS 拡張側に置く」と定め、config docstring も「lib は解釈せず素通しする」と書いている。本 ADR は
   その原則に対する限定的な例外を置く。逸脱範囲は「読むのは `token_resolver` 属性のみ / 型判別は
   しない / エクスポータ選択はしない」であり、切替仕様の SoT は依然 MS 拡張側にある。
4. **警告は `configure()` 呼び出しの前に置く**。判定材料は config のみで構成の成否と独立しており、
   構成失敗時の early return（`_adapters/observability.py:230`）で警告が失われない。既存の順序
   （トレース無効検知が前・構成失敗検知が後）とも整合する。
5. **警告文言は結果を断定しない**。同一プロセスで 2 回目以降の `configure()` は渡した設定を丸ごと
   無視して真を返すため（`config.py:115-119`）、「コンソールへフォールバックする」と断定するとその
   経路で偽になる。「この設定が適用される場合」という条件節を置き、事実（resolver が参照されない）
   と確認すべきこと（`exporter_options` 側に resolver を設定する）を既存警告と同じ語調で並べる。
6. **env フラグ由来の未達は検知しない**。代わりに利用者が自力で判定できる材料を usage docs で
   提供する: (a) 到達先の判定条件（env 変数名と真と扱う値・実 API には
   `exporter_options.token_resolver` が必須である旨。ただし切替仕様の SoT は MS 拡張側であることを
   明記し、上流内部 dispatch の全再現はしない）、(b) コンソール出力の有無による自己診断手順
   （分岐 3 のみ標準出力へ span が出る。`config.py:187`。「実 API へ切り替えたのにコンソールへ span が
   出続けている = フォールバックしている」で判別できる）、(c) 利用者アプリ側での前提セルフチェック例
   （利用者は env を読めるため lib の env 非依存不変条件に触れない）。
7. **本 ADR は ADR 0022 を supersede しない**。0022 の決定（グローバル結線の build-don't-run 例外化・
   エクスポート先切替の MS 拡張への委譲）はそのまま有効であり、本 ADR は「到達先の検知範囲」という
   別の論点を補う。

公開 API 契約は変更しない。`runtime/observability` の `__all__` のメンバ集合・
`enable_agent365_tracing` のシグネチャと戻り値・`Agent365TracingConfig` のフィールド集合と既定値は
いずれも不変であり、追加されるのは特定の構成でのみ `RuntimeWarning` が 1 件増えるという観測可能な
振る舞いのみである。

## Consequences

- + 実測で存在が確定した「認証手段を渡しているのに届かない」構成を、誤検知ゼロで通知できる。上流は
  当該構成でも `configure()` / `is_configured()` がともに真を返すため、警告が唯一の検知手段である。
- + 公開 API 契約（`__all__` のメンバ集合・シグネチャ・config のフィールド集合）は不変であり、
  既存の正常系（options 未指定・sidecar・resolver 設定済み）の挙動は一切変わらない。
- + env 非依存の不変条件と SDK 隔離を維持したまま実現できる。実装は `_adapters/observability.py` の
  判定 1 つと警告 1 つに閉じ、新しい通知機構・診断 API・config フィールドを増やさない。
- - env フラグ未設定による未達は lib 側では警告されない。利用者は usage docs の判定条件表・
  コンソール出力による自己診断手順・アプリ側セルフチェック例に依拠する必要がある。
- - 判定が上流のクラス構造（`Agent365ExporterOptions` が `token_resolver` 属性を持ち、
  `SpectraExporterOptions` が持たないこと）に依存する。`SpectraExporterOptions` に当該属性が
  追加されると sidecar 構成で誤検知が始まるため、L2 契約テストでこの前提を pin する。
- - 上流が `exporter_options` 指定時にもトップレベル `token_resolver` を合成するよう変更された場合、
  本警告は誤検知になる。依存は上限なし（`pyproject.toml:72`）であるため、品質保証台帳の retire 条件に
  この条件を明記する。
- - `token_resolver` 属性を持ちバッチ処理属性を持たない自作 options では、本警告と既存の構成失敗警告が
  二重に出る（上流が `max_queue_size` 参照で `AttributeError` になるため）。害はないため意図として
  許容する。
- + 属性取得を番兵付き `getattr` + 例外吸収で行うため、利用者が渡す不透明値が属性アクセスで例外を
  投げても有効化は失敗せず、観測の構成判定がアプリの可用性へ影響しない。判定不能な構成は警告せず
  番兵へ倒すため、誤警告も増やさない。
- - 属性は `configure()` を呼ぶ**前**に 1 回だけ読む。上流が当該属性を読まない構成（sidecar・
  有効化フラグ未設定）でも読むため、利用者の計算プロパティは上流より広い経路で 1 回評価される。
  例外は吸収するが評価そのものは避けられないため、ブロッキングな property を渡さないことを
  `Agent365TracingConfig.exporter_options` の docstring で利用者契約として明示する。吸収した例外は
  警告せず DEBUG ログへトレースバック付きで記録し、判定不能を「判定対象外」と区別できるようにする。

## Confirmation

本決定を機械強制するテストは実装済みである。**個々のテスト node ID は
`docs/QUALITY-GUARANTEES.md` の source = `ADR-0024` の行を参照する**（テスト名は改名されうる可変な
情報であり、`/spec-sync` が鮮度維持を担う台帳側を単一の置き場とする）。本節は改名で古くならない
「何を pin する必要があるか・なぜその形でないと無音化するか」だけを記す。

- **発火**: 通常発火と、構成失敗時にも発火すること（判定材料は config のみで構成の成否と独立して
  いるため、警告は `configure()` の前に置く）。警告文面は原因・結果を断定しない条件節・是正指示の
  3 点を個別に照合する（単一の `match=` では条件節と是正指示が消えても緑になる）。警告の帰属
  （`stacklevel=2`）も検査する（帰属の退行は警告を出し続けるためメッセージ照合では検知できない）。
- **誤検知しないこと**: 意図された既定（`exporter_options` 未指定）・sidecar 構成・options 側に
  resolver 設定済み・バッチ調整目的の options 単独指定の 4 経路。sidecar 構成の除外は L1
  （`token_resolver` 属性を持たないスタブで番兵分岐を pin）と L2（実 `SpectraExporterOptions` が
  当該属性を持たないことを pin）の**合成**で保証する。L1 側を実 SDK のクラスで書くと observability
  extra 未導入環境で skip され、番兵分岐の変異検知が黙って失われる。本リポジトリは `filterwarnings`
  未設定のため、非発火テストは `warnings.catch_warnings()` + `simplefilter("error")` または
  `record=True` での 0 件 assert を必須とする（指定しないと「発火しないこと」を検査できない）。
- **上流前提**: `Agent365ExporterOptions` が `token_resolver` を持ち既定が `None` であること、
  `SpectraExporterOptions` が持たないこと。後者は「存在しない属性の不在」を主張するだけでは属性名の
  typo で無音成立するため、実在する属性に対する正の assert を併記する。加えて本決定の唯一の前提
  （`exporter_options` 指定時にトップレベル `token_resolver` が使われないこと）を、子プロセスで
  `configure()` を 1 回だけ呼ぶ probe で実測する。上流が両者を合成する実装へ変われば本警告は
  「届いている構成への誤警告」へ反転するため、`exporter_options` 未指定時に実 exporter が選ばれる
  ことを正の対照として併置する（対照が無いと有効化フラグが効いていないだけの環境でも緑になる）。
  span を生成しないため外部通信は発生しない。
- **例外吸収と痕跡**: `token_resolver` が `AttributeError` 以外を投げる options で、例外を伝播させず・
  誤警告もせず・計装まで到達すること。属性の読み取り回数は判定する構成で 1 回・判定しない構成で 0 回
  （評価の広さそのものが Consequences で開示した利用者契約であり、その上限も契約に含まれる）。
  判定不能時の DEBUG ログは、レベル・`exc_info` の存在・options の型・logger 名リテラルを個別に
  照合する（logger 名は docs が `logging.getLogger(...)` の引数として公開しているため、モジュール
  改名にテストが追随すると docs だけが静かに偽になる）。
- **確認の方法**: 上記の各 pin は変異注入で実効性を確認する。「テストが実在する」「スイートが全緑」
  のみでは確認済みとしない。集合の一致を主張する pin は過大側・過小側の両方向を注入する。変異の
  復元は退避した全文の `cp` で行い、`git checkout` / `git restore` / `git stash` / `git reset` は
  使わない。
- **保守条件**: 本モジュールへ 5 番目以降の `RuntimeWarning` を追加する場合、帰属の検査ヘルパの
  適用を必須とする（強制手段が列挙方式のため、適用漏れは全緑のまま帰属だけを失う）。
