# 次ターン開始エージェントの上書き（NextTurnPolicy）

## 何を解決するか

ハンドオフで専門エージェントへ遷移すると、SDK 標準の last_agent 継続（`Runner.run(result.last_agent, ...)`）では次ターンも専門エージェントから始まります。「請求の話が終わったら次は窓口（triage）へ戻す」といった会話設計を毎ターンの if 文で書くと、判断が呼び出しコードに散らばります。`NextTurnPolicy` は「ハンドオフ経由で X が回答を終えたら次ターンは Y から」という上書きルールをグラフ宣言と同じレベルで固定し、run 完了結果から次ターンの開始エージェントを副作用なく解決します。

あわせて「ハンドオフで X に到達したターンでは X の全 handoff を無効化し、X 自身に回答を終えさせる」到達時ハンドオフ禁止をルール単位で opt-in できます（たらい回しの遮断）。次ターンの `Runner.run` は利用者コードが呼びます（build-don't-run）。

## 使い分け

| パターン | 仕組み | 最適な場合 |
|---|---|---|
| `NextTurnRule(next_agent=...)` | 次ターンの開始先を固定 | 回答後に必ず窓口へ戻したい |
| `NextTurnRule(no_handoff_on_arrival=True)` のみ | 到達ターンの handoff を無効化（次ターンは last_agent 継続） | たらい回しだけ止めたい |
| `source=...` 付きルール | 特定の遷移元からの到達に限定 | 到達元で戻し先・禁止有無を変えたい |
| `next_turn_agent(...)` | 解決 + registry 解決 + last_agent フォールバックを 1 回で | 毎ターンの分岐を書きたくない |
| `resolve_next_agent(...)` | 名前（`str`）または「上書きなし」（`None`）だけ返す | 分岐を自前で書きたい |
| SDK 標準 `Agent.clone(handoffs=[])` | ターン開始エージェントの handoff を単発で外す | 宣言ではなく 1 回限りの無効化がしたい |

より広い（handoff vs agent-as-tool vs WorkflowGraph の）使い分けは [multi_agent](./multi_agent.md) を参照。

## 使い方

- import: `from oai_agentspec import NextTurnPolicy, NextTurnRule, apply_next_turn_policy, next_turn_agent, resolve_next_agent`
- extras: なし
- 依存 env: なし

主経路は「宣言 -> `apply_next_turn_policy` -> `Runner.run` -> `next_turn_agent`」です。

```python
from agents import Runner
from oai_agentspec import (
    NextTurnPolicy, NextTurnRule, apply_next_turn_policy, next_turn_agent,
)

# 前提: triage -> billing / tech、server -> billing の handoff を持つ registry
policy = NextTurnPolicy(
    rules={
        "billing": (
            # triage からの到達: handoff を止めたうえで、次ターンは triage から
            NextTurnRule(next_agent="triage", no_handoff_on_arrival=True, source="triage"),
            # それ以外（server 等）からの到達: 禁止のみ（次ターンは last_agent 継続）
            NextTurnRule(no_handoff_on_arrival=True),
        ),
        "tech": "triage",  # 次ターン指定だけの単一ルールは名前の str 略記
    }
)

# 名前整合を検証し、到達時ハンドオフ禁止を結線した派生 registry を返す（元 registry は不変）
runtime_registry = apply_next_turn_policy(policy, registry)

result = await Runner.run(runtime_registry.get("triage"), user_input, session=session)

# 上書き発動時は Y の Agent、非発動時は result.last_agent が返る
next_agent = next_turn_agent(policy, result, runtime_registry)
if next_agent is not None:
    result = await Runner.run(next_agent, next_input, session=session)
else:
    ...  # last_agent も取得できず開始エージェントを決められない場合（扱いは利用者の判断）
```

`next_turn_agent` を使わず、解決結果で自分で分岐することもできます。

```python
from oai_agentspec import resolve_next_agent

name = resolve_next_agent(policy, result)  # str = 上書き先の名前 / None = 上書きなし
agent = runtime_registry.get(name) if name is not None else result.last_agent
result = await Runner.run(agent, next_input, session=session)
```

到達時ハンドオフ禁止だけを使う（次ターンは従来どおり last_agent 継続）宣言は次の形です。

```python
policy = NextTurnPolicy(rules={"billing": NextTurnRule(no_handoff_on_arrival=True)})
runtime_registry = apply_next_turn_policy(policy, registry)
# billing にハンドオフで到達したターンは billing の全 handoff が無効化され、billing が回答を終える。
# resolve_next_agent は「上書きなし」を返すため、次ターンは result.last_agent で継続する。
```

ターン開始エージェントの handoff を単発で外したい（宣言ではなく 1 回限り）場合は、SDK 標準の `Agent.clone(handoffs=[])` を利用者コードで直接使います（lib は API を提供しません。変形インスタンスの管理は利用者責務）。

```python
solo = runtime_registry.get("billing").clone(handoffs=[])
result = await Runner.run(solo, user_input, session=session)
```

`Runner.run_streamed` の完了結果（`RunResultStreaming`）も、`resolve_next_agent` / `next_turn_agent` にそのまま渡せます。判定材料である `new_items`（handoff アイテム）と `last_agent` を `RunResult` と同名・同義で持つため、同じ観測が得られます（ストリーム消費が完了した結果を渡してください）。

## パラメータ一覧
（下表は現時点のシグネチャ抜粋。乖離時は `docs/architecture.md` を正とする）


### `NextTurnPolicy`（frozen）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `rules` | `Mapping[str, NextTurnRule \| Sequence[NextTurnRule] \| str]` | `{}` | 回答者名 X -> 単一ルール / ルールの列 / 次ターン名の str 略記。build 時に `dict` へ正規化・検証したうえで `MappingProxyType` へ差し替えられ不変化する（事後注入・元 dict 変更は反映されない）。空の `rules` は no-op 宣言として許容 |

### `NextTurnRule`（frozen）

| パラメータ | 型 | 既定 | 説明 |
|---|---|---|---|
| `next_agent` | `str \| None` | `None` | 次ターンの開始エージェント名（Y）。X と同名も可（継続の明示固定） |
| `no_handoff_on_arrival` | `bool` | `False` | 到達時ハンドオフ禁止の opt-in。`True` で、ハンドオフによる X 到達以降そのターン中は X の全 handoff が無効化される |
| `source` | `str \| None` | `None` | 到達元条件。指定した遷移元からのハンドオフ到達に限定する（未指定は到達元不問の包括ルール。1 ルールに 1 名） |

`next_agent` と `no_handoff_on_arrival` のいずれも持たないルール、同一 X のルール列内の `source` 重複・包括ルール 2 件以上・空列は build-time `ValueError`。

### `resolve_next_agent(policy, result)`

`resolve_next_agent(policy: NextTurnPolicy, result: Any) -> str | None`。run 完了結果から次ターン開始エージェント名を解決する。「ターン内にハンドオフ遷移が 1 件以上」かつ「最終回答者名が X」の AND 条件が成立し、X の発動ルールが `next_agent` を持つときのみ名前を返し、それ以外は `None`（上書きなし）。副作用なし・決定的。

### `next_turn_agent(policy, result, registry)`

`next_turn_agent(policy: NextTurnPolicy, result: Any, registry: AgentRegistry) -> Any | None`。上書き発動時は Y を registry から解決した Agent、非発動時は `result.last_agent` を返す。`last_agent` も取得できない場合のみ `None`（開始エージェント決定不能）。Y が registry に未登録なら `KeyError` がそのまま伝播する。

### `apply_next_turn_policy(policy, registry)`

`apply_next_turn_policy(policy: NextTurnPolicy, registry: AgentRegistry) -> AgentRegistry`。宣言中の全エージェント名（キー・`next_agent`・`source`）を registry の登録名と突合し、不在なら `ValueError` で fail-fast する。検証後、到達時ハンドオフ禁止を結線した派生 registry（`registry.clone()` 由来）を返す。元 registry は変更されない。frozen な registry にも適用でき、派生 registry は元の freeze 状態を引き継ぐ（元が frozen なら派生も frozen）。到達時ハンドオフ禁止が確実に効かない宣言（禁止対象 X がファクトリ登録 / 禁止ルールの `source` がファクトリ登録名）は `ValueError` で拒否する（合成の差し込み口が無いため）。

## 判断軸

- 戻し先が宣言時に決まる → **`next_agent`**。ターン毎に動的に決めたいなら本機能ではなく利用者コードで分岐する（宣言は実行時に変更しない）
- たらい回しだけ止めたい → **禁止のみルール**（次ターンは last_agent 継続のまま）
- 到達元で挙動を変えたい → **`source` 付きルールを先に、包括ルールを後に**列で並べる（選定は「一致 `source` -> 包括 -> なし」）
- 毎ターンの分岐を減らしたい → **`next_turn_agent`**。返り値をそのまま次の `Runner.run` に渡せる
- 宣言ではなく 1 回限りの無効化 → **SDK 標準 `Agent.clone(handoffs=[])`**

## 落とし穴

- `apply_next_turn_policy` は**派生 registry を返す**。実行（`get`）も `next_turn_agent` の registry 引数も返り値の registry を使う。元 registry のまま実行すると到達時ハンドオフ禁止は働かない
- 到達時ハンドオフ禁止は「ハンドオフによる到達」でのみ発動する。X をターン開始エージェントとして直接使うターンでは発動しない（直接開始した X へ同一ターン内で再到達した場合は、その到達以降について発動する）
- 禁止のみルール（`next_agent` なし）が発動したターンは `resolve_next_agent` が `None` を返す。「禁止が働いたのに上書きされない」のは仕様で、次ターンは last_agent 継続になる
- `resolve_next_agent` の `None` は「上書きなし（正常系）」、`next_turn_agent` の `None` は「開始エージェント決定不能」。意味が異なる
- 宣言単体（`apply_next_turn_policy` を通さない）では名前の実在は検証されない。キー X が実際の回答者名と一致しなければ発動せず、Y の実在は `next_turn_agent` の registry 解決時に `KeyError` で検出される
- 次ターン開始エージェント Y 側の handoff 構成には制限を掛けない。次ターンで再び X へハンドオフされうるが、その到達でも禁止が再度働く
- 保証範囲は「到達以降そのターンで X からの handoff 遷移が実行されない」ことまで。LLM の応答文面（他部署へ回したがる表現等）は制御せず、session 履歴も改変しない
- 禁止対象 X の出辺に宣言した `is_enabled` は、到達済みのターンでは**評価されない**（合成したゲートが先に `False` を返して短絡するため）。カウント・ログ・外部呼び出し等の副作用を `is_enabled` に持たせているとその副作用も起きないため、`is_enabled` は純粋な判定関数として書く
- `RunContextWrapper` インスタンスを利用者が自作して**複数 run で使い回す形は非対応**（到達記録が run を跨ぎ、次ターンまで禁止が残りうる）。plain な context オブジェクトを `Runner.run(context=...)` に渡す通常の使い方は run ごとに分離される
- 到達時ハンドオフ禁止は**ファクトリ登録（`register_factory`）のエージェントには結線できない**。拒否されるのは「禁止対象 X 自身がファクトリ登録」「禁止ルールの `source` でファクトリ登録名を明示指定」の 2 つで、いずれも `ValueError`。包括ルール（`source` なし）の到達元候補にファクトリ登録が含まれるだけなら適用は通り、`logging.getLogger("oai_agentspec.next_turn")` に warning が 1 回出る（そのファクトリが実際に X へ handoff する場合はそのターンの禁止が効かないため、確実に効かせるなら `AgentSpec` 登録にする）。次ターン指定のみのルールはファクトリ登録でも使える
- 到達時ハンドオフ禁止の結線は `apply_next_turn_policy` を呼んだ時点の登録内容で確定する。適用後に登録したエージェントからの到達は禁止の対象外になるため、policy の適用は全登録を終えた後に行う
- HITL 承認再開（resume）はスコープ外で、上書きと到達時ハンドオフ禁止の**両方**が失効する。SDK の `RunState.from_string` は復元時に新しい `RunContextWrapper` を生成するため到達記録が引き継がれず、承認待ちで中断・再開したターンは（利用者の体感では同じターンでも）禁止が解けた状態で続行する。`RealtimeRunner` / `RealtimeSession` も同じくスコープ外
- マルチスレッドから同一の派生 registry を共有する場合の同期は、registry と同じく利用者責務

## 参照

- 詳細設計: `docs/architecture.md`（Next-Turn Agent Override 節）
- 設計判断: `docs/adr/0014-next-turn-agent-override.md`
- 関連: [handoffs.md](./handoffs.md)（ハンドオフの宣言）/ [multi_agent.md](./multi_agent.md)（オーケストレーション手段の選び方）

## 次

[workflow.md](./workflow.md) — WorkflowGraph の経路パターン
