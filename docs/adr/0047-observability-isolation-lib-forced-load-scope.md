# 0047: 観測系の分離保証を「lib が強制ロードしない」と定め、SDK 本体は引き続き禁止する

- Status: accepted
- Date: 2026-10-02

## Context

### 出発点の保証

- ADR-0022 の Confirmation は、遅延 import 境界を「有効化関数を呼ばない限り observability 依存が
  `sys.modules` に載らないこと」と定め、強制手段を `tests/test_extra_isolation.py` の subprocess 隔離テストに
  置いた。同じ保証を観測系の窓口側で検査するテストとして、
  `tests/runtime/observability/test_config_l1.py::test_config_module_does_not_load_observability_sdks` と
  `tests/runtime/observability/test_reexport_l1.py::test_importing_window_does_not_load_observability_sdks`
  がある。いずれも、窓口 import 後の `sys.modules` に `opentelemetry` 系・`microsoft_agents_a365` 系の
  モジュールが、誰が読んだかを問わず 1 つも無いことを検査していた。
- ADR-0043 の Context（「mcp 2 系での観測系分離テスト」）は、mcp 2.2.0 が import 時に opentelemetry-api を
  読み込むため、`import agents` の時点で `opentelemetry` が `sys.modules` に入り、上記のテストが失敗することを
  記録した。lock で mcp が 1 系に留まるのは lightning extra の依存が `mcp<2` を要求するためで、lightning を
  入れない構成には mcp 2 系が入り、この構成は CI に現れない。同 ADR は、保証を「lib が強制ロードしない」へ
  定義し直し、SDK 経由で推移的に入るモジュールを対象外にし、観測系 SDK 本体は引き続き禁止する形を想定して、
  SDK 経由の推移の範囲を実測で確定したうえで新しい ADR を起こすとした。本 ADR がそれである。

### 実測

環境は一時の uv プロジェクトで、openai-agents 0.22.3 / mcp 2.2.0 / opentelemetry-api 1.45.0 /
opentelemetry-sdk 1.45.0 / microsoft-agents-a365-observability-core 1.0.0。課金 API への接続は無い。

- M1（observability extra なし）: `import mcp` 単体・`import agents`・`import oai_agentspec`・
  `import oai_agentspec.runtime.observability`・`import oai_agentspec.runtime.observability.config` の
  いずれでも、`sys.modules` 上の `opentelemetry` 系・`microsoft_agents_a365` 系のモジュールは同一の集合だった。
  その全モジュールが配布物 opentelemetry-api の files に属し、`opentelemetry.sdk` 系・`microsoft_agents_a365`
  系は無かった。lib の窓口 import が追加でロードする観測系モジュールは無い。
- M2（observability extra あり。opentelemetry-sdk と a365 core を導入済み）: 上記の各 import で M1 と同じ集合で、
  `opentelemetry.sdk` 系・`opentelemetry.exporter` 系・`microsoft_agents_a365` 系は無かった。
- M3（M2 の環境で上記のテストを実行）: いずれも失敗し、原因は opentelemetry-api のモジュールだけだった。
  SDK 本体は含まれない。
- M4（extras から lightning と llmops-langfuse だけを除いた構成。llmops = deepeval 4.2.7 を含む）: 依存は
  mcp 2.2.0 で解決し、観測系モジュールの集合・SDK 本体の不在・`deepeval` の非ロードとも M2 と同じだった。

この実測から、「SDK 経由の推移」を「`import agents` の時点で `sys.modules` に載っている観測系モジュール」と
定める。決定時点の実体は opentelemetry-api のモジュールのみである。なお CI（lock。mcp 1.28.1）では
`import agents` の時点で観測系モジュールは載らない。

### 検討した切り分け方式

| 方式 | 版変動への耐性 | 偽陰性（lib の強制ロードを見逃す） | 偽陽性（SDK 経由を lib のせいにする） | 判定 |
|---|---|---|---|---|
| (a) opentelemetry-api のモジュール名の allowlist | 低い。api の版でモジュールが増減すると allowlist が古くなり失敗する | あり。lib が api を強制ロードしても allowlist 内なら通る（mcp 1 系でも検出できない） | api に新しいモジュールが加わると発生する | 却下 |
| (b) 配布物（`importlib.metadata`）で opentelemetry-api 由来を除外 | 中。名前空間パッケージ `opentelemetry` を api / sdk / exporter が共有するため、トップレベル名では配布物を決められず、モジュールごとの `__file__` と配布物の files の突合が要る | あり。(a) と同じく lib による api の強制ロードを常に見逃す | 少ない | 却下。複雑さに対して (a) と同じ偽陰性が残る |
| (c) `import agents` 直後の `sys.modules` を baseline とした差分判定と、SDK 本体の絶対禁止の併用 | 高い。api のモジュール名を持たず、SDK が読む範囲が版で変わっても baseline が追随する | mcp 2 系では、SDK が既に読んだ観測系モジュールを lib が import しても検出できない（新たなロードが生じないため、定義上の強制ロードに当たらない）。mcp 1 系では検出する | `import agents` で読まれない SDK サブモジュールが将来観測系を読むと発生する。失敗側に倒れるため、保証が黙って消えることはない | **採用** |
| (d) 絶対禁止を SDK 本体だけに縮める（差分判定なし） | 高い | あり。lib が api を強制ロードしても全環境で通り、「lib が強制ロードしない」を検査しない | なし | 却下 |

(c) を採用する理由:

- 「lib が強制ロードしない」をそのまま判定式にできる。lib による追加と SDK 本体の存在を別々の判定で検査する。
- `tests/runtime/observability/test_config_l1.py::test_config_module_does_not_load_observability_sdks` は既に、
  baseline（`set(sys.modules)`）からの差分で `agents.*` の追加を判定している。同じ型を観測系に適用する。
- 同型の判断の先例が ADR-0020 にある。同 ADR は、窓口 import より前に必ずロード済みになる `agents` の
  非発火を不変条件にせず、不変条件を lib が制御するもの（実装実体モジュールの非ロード）に定め直した。
  本 ADR はこの原則を観測系に適用する。

baseline を `import agents` だけにする理由:

- lib の `_adapters` は `agents` のサブモジュール（`agents.sandbox` 等）をトップレベルで import する。
  これらを baseline に列挙すると、テストが lib の import 構成に結合する。
- M1 / M4 で、`import agents` と lib の窓口 import の観測系モジュールの集合は同一だった。`import agents`
  以外の SDK サブモジュールが観測系を追加で読む事実は、決定時点では無い。
- 将来それが起きた場合は、違反モジュール名を伴う失敗として表面化し、保証の黙った消失にはならない。

## Decision

### 1. 保証の定義

ADR-0022 の Confirmation の遅延 import 境界の項を、次の 2 つの判定で定義し直す。

- **判定 A（lib による追加の禁止）**: `import agents` の直後の `sys.modules` を baseline とする。その後の
  lib の窓口 import（設定型の構築を含む場合はそれも含む）で baseline から新たに増えたモジュールのうち、
  `opentelemetry` 系・`microsoft_agents_a365` 系に一致するものがあってはならない。
- **判定 B（SDK 本体の絶対禁止）**: baseline に含まれるかどうかに関わらず、`sys.modules` に観測系 SDK 本体の
  名前空間に一致するモジュールがあってはならない。

名前の一致は、名前空間 `p` に対して `m == p or m.startswith(p + ".")` とする（サブモジュール単位の判定。
トップレベル名だけの一致は、`opentelemetry` 自体が baseline に載る環境では差分判定として成立しない）。

`import agents` の時点で SDK 経由で載っている観測系モジュール（決定時点の実体は mcp 2 系での
opentelemetry-api）は、判定 B の名前空間に一致しない限り対象外とする。

### 2. 観測系 SDK 本体の範囲

観測系 SDK 本体は、observability extra（`pyproject.toml` の `[project.optional-dependencies]` の
`observability`）が直接宣言する配布物のトップ名前空間とする。docs 上の列挙はここだけに置き、他の箇所は
本節を参照する。

- `opentelemetry.sdk`（opentelemetry-sdk）
- `opentelemetry.exporter`（opentelemetry-exporter-otlp-proto-http）
- `microsoft_agents_a365`（microsoft-agents-a365-observability-extensions-openai と、その依存の core）

observability extra が直接宣言する配布物を変える場合は、本節の範囲も見直す。

### 3. 対象の窓口

- `import oai_agentspec`
- `import oai_agentspec.runtime.observability`
- `import oai_agentspec.runtime.observability.config` と、設定型（`Agent365TracingConfig` /
  `OtelLoggingConfig`）の構築

### 4. 変えないもの

- `tests/test_extra_isolation.py` が検査する観測系以外の extra 依存（`fastapi` / `websockets` / `deepeval` /
  `langfuse`）は、従来どおりトップレベル名の一致による絶対禁止のままとする。保証を狭める根拠となる実測が無い
  ため（M4 でも `deepeval` はロードされない）。
- src の挙動は変えない。変えるのは保証の定義と、それを検査するテストの判定式である。

### 5. ADR-0022 のうち置き換える範囲

ADR-0022 の Confirmation のうち、遅延 import 境界（有効化関数を呼ばない限り observability 依存が
`sys.modules` に載らないこと）の項だけを本 ADR で置き換える。同 Confirmation の他の項（root logger への冪等
付与・ログ連携の再設定検知・構成失敗時の非例外化・グローバル結線の物理隔離）と、同 ADR の Decision /
Consequences は変えない。「`import oai_agentspec` は root logger・SDK トレーシングに非接触」という import
副作用の不在も変わらない。

## Consequences

- + mcp 2 系の構成（lightning を入れない構成）でも保証が成り立ち、テストが通る。
- + 判定が opentelemetry-api のモジュール名に依存しない。SDK や api の版で SDK 経由のモジュールが変わっても、
  baseline が追随する。
- + lib の関数内遅延 import をトップレベルへ昇格させる退行のうち、SDK 本体を読むものは mcp の版に関わらず
  検出する。
- - mcp 2 系では、SDK が既に読んだ観測系モジュール（決定時点では opentelemetry-api）を lib が import しても
  検出できない。例として、`_adapters/observability.py` のトップに `import opentelemetry.trace` を注入する変異は
  mcp 2 系では緑のままになる。新たなロードが生じないため定義上の強制ロードに当たらず、本 ADR が保証を狭めた
  範囲そのものである。CI（mcp 1 系）では baseline に観測系モジュールが無いため、同じ変異を判定 A が検出する。
- - `import agents` で読まれない SDK サブモジュール（lib が直接 import するもの）が将来観測系モジュールを読む
  ようになると、lib のせいではない追加として判定 A が失敗する（偽陽性）。失敗側に倒れるため保証は黙って
  消えず、違反モジュール名から原因を辿れる。
- - mcp 2 系の構成そのものは CI では継続的に検証しない（ADR-0043 の Consequences と同じ状態）。SDK が
  `import agents` の時点で opentelemetry-api を読み込む状況は、テストの中で opentelemetry-api を
  `import agents` より前に読み込むことで再現し、CI（mcp 1 系）でも検査する。

## Confirmation

- 強制手段は次のテストである。いずれもクリーンな子プロセスで `sys.modules` を検査し、対象の窓口ごとに
  1 組ずつある。子プロセスの probe は `import agents` より前に任意の文（preamble）を差し込めるように
  組み立て、SDK が観測系モジュールを先に読み込んだ状況をこの preamble で再現する。
  - 窓口の import で判定 A / 判定 B の違反が無いこと:
    - `tests/test_extra_isolation.py::test_importing_package_does_not_force_load_extra_deps`
    - `tests/runtime/observability/test_config_l1.py::test_config_module_does_not_load_observability_sdks`
    - `tests/runtime/observability/test_reexport_l1.py::test_importing_window_does_not_load_observability_sdks`
  - `import agents` より前に載った opentelemetry-api を違反にしないこと（判定 A が差分であること。
    preamble で opentelemetry-api を先に読み込み、mcp 2 系の baseline を再現する）:
    - `tests/test_extra_isolation.py::test_observability_api_loaded_before_agents_is_not_attributed_to_lib`
    - `tests/runtime/observability/test_config_l1.py::test_observability_api_loaded_before_agents_is_not_attributed_to_config`
    - `tests/runtime/observability/test_reexport_l1.py::test_observability_api_loaded_before_agents_is_not_attributed_to_window`
  - baseline に含まれる観測系 SDK 本体も違反にすること（判定 B。preamble で SDK 本体を先に読み込む。
    `opentelemetry.sdk` と `opentelemetry.exporter` の名前空間ごとに検査する。`microsoft_agents_a365` は
    読み込むと他の名前空間も伴うため、単独の状況は作れない）:
    - `tests/test_extra_isolation.py::test_sdk_body_loaded_before_agents_is_a_violation`
    - `tests/runtime/observability/test_config_l1.py::test_sdk_body_loaded_before_agents_is_a_config_violation`
    - `tests/runtime/observability/test_reexport_l1.py::test_sdk_body_loaded_before_agents_is_a_window_violation`
- 検出力は次の変異で、対応するテストが失敗することにより確認する。
  - 判定 A: 観測系モジュールの import（`opentelemetry.trace` 等）を
    `src/oai_agentspec/_adapters/observability.py` のモジュールトップへ注入する。窓口の import のテストが
    失敗する（CI の mcp 1 系で検出）。
  - 判定 A の差分: probe の `set(sys.modules) - sdk_baseline` を `sys.modules` に置き換える。
    opentelemetry-api を先に読み込むテストが失敗する。
  - 判定 B: probe の違反の和集合から判定 B の項を外す。SDK 本体を先に読み込むテストが失敗する。
- `docs/QUALITY-GUARANTEES.md` の observability 遅延 import 境界の行に、source = ADR 0047 として登録する
  （相互参照）。
