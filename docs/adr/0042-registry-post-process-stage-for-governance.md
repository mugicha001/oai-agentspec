# 0042: registry に第 3 段 post-process を追加し、sub_agents の as_tool と factory Agent をオプトインで統治する

- Status: accepted (partially superseded by 0045: 6 の複製時の同一性保持と、SDK 複製が包み直さないことへの依存)
- Date: 2026-09-29

## Context

`GovernedAgentBuilder` は `AgentBuilder.build(spec)` の中で `_adapters.govern_spec` を呼び、`spec.tools` の
`FunctionTool` の実行本体をポリシー評価付きへ置換し、監査 `AgentHooks` を `spec.hooks` と合成する。
registry の構築は `_registry_core.build_two_pass` の 2 パス（パス 1 = bare build・パス 2 = wire）であり、
統治は build（パス 1）で完結する。この構造では次の 2 経路が build 時の統治を経ない。

- **`sub_agents` の as_tool**: `AgentRegistry._wire`（パス 2）が `make_agent_tool(sub_agent, ...)` で生成した
  as_tool を `agent.tools` へ直接 append する。build の後に注入されるため govern ラップを受けず、per-call の
  allow / deny 評価と `tool:` の決定記録を持たない（監査フックの `tool_start:` / `tool_end:` 記録のみ）。
- **`register_factory` の Agent**: `AgentRegistry.get` が factory の戻り値をそのままキャッシュし、builder の
  `build` も `_wire` も通らない。

利用側のアプリケーションは統治下で `sub_agents` / `register_factory` を使いたいが、この 2 経路の使用を
検知して起動を拒否する回避策で運用していた。

ADR-0025 は `AGENT_AS_TOOL` origin を監査フックで評価する案を「既存利用者の `allowed_tools` に as_tool 名が
書かれておらず、対象化すると既存デプロイが即 deny で機能停止する」理由で却下し、「`sub_agents` の統治は
`allowed_tools` へ as_tool 名を書く規約の追加を伴う独立した意思決定」（ADR-0025 の「origin 判定方式の候補」節）
と記録している。本 ADR がその意思決定である。既定無効・明示オプトインとするため、ADR-0025 の却下理由とは
衝突しない。ADR-0025 の決定（`spec.tools` は build 時の実行本体ラップ・MCP は監査フックで評価）は変えない。

構造上の制約:

- wire は `agent.handoffs.append(target)` と `agent.tools.append(as_tool)` で `built` 内の Agent オブジェクト
  そのものを参照に取り込み、SDK の as_tool は `_agent_instance` にサブ Agent を捕捉して nested run が
  サブ Agent の `tools` / `hooks` を run 時に読む。したがって注入済み as_tool を Agent 層で統治するなら
  (a) 全 wire の後に走り、(b) spec 経路の Agent の identity を変えない（clone すると他 Agent が未統治
  インスタンスを指す）必要がある。
- SDK の `Agent.clone` は `tools=` を渡さない限り tools の list を共有する（浅いコピー）。wire の途中で
  factory が `registry.get` の戻りを clone すると、その clone は第 3 段より前の list を指す。
- factory は到達可能収集の対象外で、他 Agent からの参照は `get` 経由でキャッシュ値を取る。factory が
  共有 Agent を返しうるため、factory 経路で in-place 加工すると利用者のオブジェクトを汚す。
- 第 3 段を足すと統治点は build 時と第 3 段の 2 つになり、build 時に統治済みの tool と wire が注入した
  as_tool を第 3 段で判別する手段が要る。
- SDK の as_tool 生成物は `ToolOriginType.AGENT_AS_TOOL` の origin と `_is_agent_tool` / `_agent_instance`
  を dataclass の kw_only フィールドに持ち、`dataclasses.replace` によるラップで引き継がれる。SDK の
  `FunctionTool.__post_init__` / `__copy__` は `on_invoke_tool` を包み直さず、同じ関数オブジェクトを保持する。

### 検討した案（本 ADR で却下したもの）

| 案 | 却下理由 |
|---|---|
| build 時の統治をやめ、`spec.tools` / hooks を含む統治を第 3 段へ一本化する | 第 3 段を呼ばない構成で統治が外れる。`build` だけを委譲する装飾 builder や入れ子の builder の下では第 3 段の呼び出しが届かず、`spec.tools` の統治まで黙って消える。加えて build から第 3 段までの間（wire の途中）に factory が取得した Agent は未統治のまま漏れる。いずれも既定経路へ fail-open を持ち込む |
| builder が `AgentPostProcessor` を兼ね、registry が `isinstance(builder, AgentPostProcessor)` で発見して第 3 段を呼ぶ | 装飾 builder・入れ子 builder の下では最外の builder しか判定されず、オプトインが黙って無効になる |
| 装飾 builder の実装者に `post_process` の委譲を書かせる | 書き忘れても例外も警告も出ず、builder 側から「呼ばれていない」を判定する時点も無い |
| build 時の `govern_spec` を残し、registry に getattr で探す任意フック（magic name）を足す | 契約が型に現れず、Mock 系オブジェクトが任意属性に応答して誤検出する。採用案との差は Protocol か magic name か・注入点が明示か暗黙かにある |
| 監査フック（`on_tool_start`）で `AGENT_AS_TOOL` origin も評価する | `spec.tools` に直接置いた as_tool との二重評価・deny 時の利用者フック非到達・`Agent.hooks` 差し替えでの強制消失を新経路へ持ち込む。ラップ対象が第 3 段の時点に存在するため「存在する tool は実行本体ラップ・run 時解決の MCP のみフック」の原則を維持できる |
| 統治済みの印をラッパ関数の属性（`setattr`）で付ける | 利用者が同名属性を付ければ偽造でき、`functools.wraps` が属性をコピーし、truthy な値を印と誤判定しうる |
| 統治済みラッパを `weakref.WeakSet` で保持する | 所属判定が hash / eq に依存し、hash / eq を参照先へ委譲するプロキシや利用者定義の `__eq__` を所属と誤判定する |
| build 時に統治後 tool の `id()` を spec 名ごとに記録する | GC 後の id 再利用で未統治 as_tool を統治済みと誤判定する（fail-open）。clone した registry で記録が混ざる |
| origin が `AGENT_AS_TOOL` の tool だけを第 3 段の対象にする | 利用者が `spec.tools` に直接置いた as_tool（build 時に統治済み）を二重ラップする。由来の判別に結局は印が要る |
| post-processor を公開の独立クラス（`runtime.governance.__all__` に追加）にする | 機能は同じで公開シンボルが増えるだけになる |
| 独自のポリシー・sink を持つ post-processor にする | ポリシーを二重に宣言することになり、sink の渡し忘れで監査チェーンが分断し、override の適用記録が 2 か所に割れる |
| `AgentBuilder` Protocol に post-process のメソッドを追加する | 既存の自作 builder が `isinstance(x, AgentBuilder)` を通らなくなる |
| factory 経路は対応しない（as_tool だけを対象化する） | `register_factory` の Agent を統治下で使うという要求を満たさない |
| オプトインを単一フラグにする | `sub_agents` と `register_factory` は別の要求であり、片方だけを有効化する手段が無くなる |

## Decision

1. **構築を build → wire →（任意の）第 3 段 post-process とする**: `AgentRegistry.__init__(agent_builder=None,
   *, guardrail_registry=None, post_processor=None)` に kw-only の `post_processor`（`AgentPostProcessor`）を
   追加し、明示的に渡したときだけ第 3 段を行う。
   - registry は builder を `isinstance` で発見しない（builder が `AgentPostProcessor` を満たしていても呼ばない）。
     未指定なら第 3 段も、第 3 段のための `_builder()` 呼び出しも無く、構築は従来の 2 パスと同一である。
   - `AgentPostProcessor` を満たさないオブジェクト（builder 自身を渡す誤用を含む）は `__init__` で `TypeError`。
   - `clone()` は `post_processor` を共有継承する（builder / `guardrail_registry` と同じ扱い）。
   - spec 経路: `_registry_core.build_two_pass` の kw-only 引数 `post_process`（既定 None）として注入し、
     パス 1 / 2 と同じ try の内側・収集した全 spec の wire 完了後に各 spec につき 1 回呼ぶ。post-processor は
     受け取った Agent を in-place で加工して同一オブジェクトを返す契約で、registry は `result is agent` を検査し、
     違えば `ValueError` を送出する。パス 1〜第 3 段のどこで落ちても本呼び出しで新規キャッシュした Agent を
     巻き戻す。
   - factory 経路: `get` が factory を呼んだ直後に `spec=None` で 1 回呼び、戻り値をキャッシュする。
     post-processor は受け取った Agent を変更せず、加工が必要なら新インスタンスを返す。
   - `AgentPostProcessor`（`post_process(agent, *, name: str, spec: AgentSpec | None) -> Agent` の 1 メソッド・
     `runtime_checkable`）は `oai_agentspec.protocols` に置き `protocols.__all__` に載せる（コア `__all__` には
     載せない）。`AgentBuilder` は変更しない。`RealtimeAgentRegistry` は第 3 段を持たない。
2. **`spec.tools` と hooks の統治は build 時のまま**: `GovernedAgentBuilder.build` は従来どおり `govern_spec` で
   統治する（ADR-0025 の決定は不変）。第 3 段は build 時の統治を代替せず、`agent_builder` の構築経路に同じ
   `GovernedAgentBuilder` の `build` が含まれる前提で動く（装飾・入れ子にしてよい）。
3. **オプトインは `GovernedAgentBuilder.post_processor(*, sub_agent_tools=False, factory_agents=False)`**:
   戻り値は builder.py の private クラスで、生成元 builder のポリシー（YAML 解決スナップショット）・override・
   既定 sink・override 適用記録を共有する（監査チェーンは build 時の記録と連続する）。両方 False は
   `ValueError`（「渡した = オプトインした」を不変条件にする）。`runtime.governance.__all__` は増やさない。
4. **`sub_agent_tools=True`（spec 経路）**: `_adapters.govern_ungoverned_tools` が `agent.tools` のうち印の無い
   `FunctionTool`（wire が注入した as_tool 等）だけを `_govern_tool` でラップする。origin では絞らない。
   `agent.tools` の list を再束縛せず同じ list の要素を置換するため、wire の途中で作られた list 共有の clone
   にも統治が届く。hooks は build 時に合成済みのため再合成しない（再合成するとライフサイクル記録・MCP 評価が
   重複する）。
5. **`factory_agents=True`（factory 経路）**: `_adapters.govern_agent` が factory の戻り値を clone し、全
   `FunctionTool` のラップと監査フックの合成（`agent.hooks` との「監査記録 → 既存フックへ委譲」）を行う。
   印は見ない（統治済みを skip すると factory 名のポリシーが黙って効かなくなるため）。factory が返した
   Agent は変更しない。
6. **統治済みの印はラッパ関数との同一性**: `_govern_tool` が作ったラッパ関数を module-private の
   `_GOVERNED_WRAPPERS: dict[int, weakref.ref]` へ登録し、`_is_governed(tool)` は
   `tool.on_invoke_tool` が登録済みの関数そのもの（`ref() is fn`）かで判定する。hash / eq に依存しない。
   弱参照のためラッパが GC されるとエントリは消え、削除コールバックは同じ id で後から登録されたエントリを
   消さない。`dataclasses.replace` / `copy` を経ても同じ関数を指すため判定は保たれる。
7. **監査 `tool:` の `agent_id` と適用記録**: `tool:` レコードの `agent_id` は registry の登録名である
   （build 時の `spec.tools` は `spec.name`、注入 as_tool は親エージェントの登録名、factory Agent は
   `register_factory` の名前）。override の適用記録は、spec 経路は従来どおり build の成功時、factory 経路は
   `factory_agents=True` の統治の成功時に行う（既定の素通しでは記録しない）。既定 sink は従来どおり初回
   build で生成し、factory だけの registry で `factory_agents=True` の場合は初回の factory 統治で生成する。
8. 評価点は 2 つのまま（tool 実行本体のラップ / MCP のみ監査フック）で、ADR-0025 の positive 判定・
   fail-closed / fail-open・監査形式・合成形・宣言面は変えない。統治を適用する結線点は build 時と、オプトイン時の
   第 3 段の 2 つになる。第 3 段は結線のみで実行ループ・`Runner` 参照を持たず、build-don't-run の例外を新設しない。

## Consequences

- `post_processor` を渡さない registry は、監査レコード列・identity・委譲順・`spec.hooks` の受理 / 拒否集合を
  含めて従来と同一である。`build` だけを委譲する装飾 builder・入れ子の builder でも build 時の統治は外れない。
- オプトインの設定を持つオブジェクトと registry が第 3 段で呼ぶオブジェクトが同じ 1 つになり、
  `agent_builder` をどう装飾・入れ子にしても効く。誤用（builder 自身を渡す・両フラグ False）は構築時の例外になる。
- breaking change は無い。非 breaking の追加は `AgentRegistry.__init__` の kw-only `post_processor`・
  `GovernedAgentBuilder.post_processor()`・`protocols.AgentPostProcessor`・private leaf `build_two_pass` の
  kw-only `post_process=None`（None なら従来の 2 パス）。
- `sub_agent_tools=True` を有効化する利用者は、親エージェントの `allowed_tools` に as_tool の公開名
  （`sub_agent_tools` 未指定時は SDK がエージェント名から導出）を追記する必要がある。追記しないと deny になる。
- `sub_agent_tools=True` では as_tool の入力文が `tool:` レコードの `details.arguments` に全文記録される
  （個人情報等を含みうる。記録先を `audit_sink` で選定する）。
- `factory_agents=True` のとき `registry.get(name)` は factory 戻り値の clone を返し、identity が変わる。
- 境界（詳細は `GovernedAgentBuilder.post_processor()` の docstring を正とする）:
  - `agent_builder` の構築経路に `GovernedAgentBuilder` が無いと、spec 経路の Agent は第 3 段で印の無い tool が
    ラップされるだけで監査フックが付かず、MCP ツールの評価とライフサイクル監査は行われない（例外も警告も
    出ない）。
  - 統治済みの Agent（例: `registry.get` の戻り）を返す factory は、元のポリシーと factory 名のポリシーの
    両方で評価され、`tool:` レコードとライフサイクル記録が 2 回ずつ残る（どちらかが deny なら deny）。
  - 結線途中の Agent X から factory F が作った Agent における X の注入 as_tool は、F が X そのもの / list を
    共有する clone を返すなら `factory_agents=False` で X の第 3 段（`sub_agent_tools=True` のとき）により
    統治され、新しい list へ写したものは未統治である。`factory_agents=True` では F 名のポリシーだけで統治され、
    clone 後に注入された as_tool は含まれない。X の `get` がロールバックした場合、F のキャッシュは第 3 段前の X を
    持ち続ける。
  - 印は「いずれかのポリシーで統治済み」を示す。自作 `inner` が別 agent / 別 builder で統治済みの tool を
    足すと、その tool は元のポリシーだけで評価される。逆に自作 `inner` が独自に追加した印の無い `FUNCTION`
    tool はオプトイン時に統治される。
- 印の判定は SDK の `FunctionTool` 複製（`__post_init__` / `__copy__` / `dataclasses.replace`）が
  `on_invoke_tool` を包み直さないことに依存する。包み直すようになると印が外れて build 時に統治済みの tool が
  第 3 段で再ラップされ、`tool:` レコードが 2 行になる（fail-closed）。トリップワイヤで検知する。
- as_tool メタの保持は SDK の非公開フィールド（`_tool_origin` / `_is_agent_tool` / `_agent_instance`）が
  dataclass フィールドであり続けることに依存する。フィールド外へ移ると `dataclasses.replace` は無警告で
  落とす。

## Confirmation

強制手段は `docs/QUALITY-GUARANTEES.md` に登録済み（source = ADR-0042）。

- 第 3 段の契約: `tests/test_registry_post_process_l1.py::test_builder_with_post_process_is_not_discovered_without_post_processor` /
  `::test_factory_only_registry_with_default_builder_returns_factory_result` /
  `::test_post_processor_rejects_non_protocol_object` / `::test_clone_inherits_post_processor` /
  `::test_spec_path_post_process_called_once_per_spec_after_all_wiring` /
  `::test_spec_path_returning_different_object_raises_and_rolls_back` /
  `::test_factory_path_post_process_called_once_with_spec_none` /
  `::test_factory_path_new_instance_from_post_process_is_cached` /
  `::test_post_processor_called_under_decorating_builder`
- 既定不変: `tests/_adapters/test_governance_l2.py::test_audit_record_sequence_unchanged_by_default` /
  `::test_decorated_builder_delegating_only_build_still_governs` / `::test_nested_governed_builder_still_governs`
- オプトイン API: `tests/runtime/governance/test_builder_l1.py::test_post_processor_requires_at_least_one_flag` /
  `::test_registry_rejects_governed_builder_as_post_processor` /
  `::test_post_process_flags_select_sub_agent_and_factory_governance` /
  `::test_default_sink_created_on_first_factory_post_process_when_opted_in` /
  `::test_unapplied_overrides_on_factory_path_follow_governance`
- 印: `tests/_adapters/test_governance_l1.py::test_ungoverned_predicate_by_marker` /
  `::test_governed_wrappers_are_held_weakly` / `::test_drop_callback_keeps_newer_entry_with_same_id` /
  `::test_govern_spec_outputs_are_marked`、
  `tests/_adapters/test_governance_l2.py::test_governed_registration_survives_replace_and_copy_tripwire` /
  `::test_opt_in_sub_agent_tools_adds_only_injected_tool_record`
- 要素置換と clone: `tests/_adapters/test_governance_l1.py::test_govern_ungoverned_tools_replaces_elements_in_same_list` /
  `::test_govern_agent_clone_does_not_mutate_source`、
  `tests/_adapters/test_governance_l2.py::test_factory_clone_during_wire_sees_governed_injected_as_tool` /
  `::test_opt_in_factory_agent_governed_as_clone_allow_and_deny` /
  `::test_decorated_builder_with_explicit_post_processor_governs_injected_as_tool` /
  `::test_factory_returning_governed_agent_is_evaluated_by_both_policies`
