# 0044: 決定論モデルが返す既定 call_id を、要求と当該 tool call の内容から導出する

- Status: accepted
- Date: 2026-09-30

## Context

### 決定時点の既定 call_id

- 応答ビルダ `tool_call_response` は `call_id` を省略すると既定値 `call_deterministic`
  （`src/oai_agentspec/_adapters/deterministic.py` の `DEFAULT_CALL_ID`）を `ModelResponse` へ焼き込む。
  値は ADR-0019 の「8. 命名と既定 id 値」で固定値として決めた。
- `DeterministicResponseModel.get_response` はルール関数の戻り値をそのまま返し、`stream_response` は
  `get_response` を経由して応答を確定させる。
- 既定値のまま 1 run で 2 回以上 tool call すると、同じ call_id が同じ run の中で再利用される。

### SDK 側の制約（ADR-0043 の実測）

- SDK 0.20.0 以降は、同じ run の中で call_id が同じで指紋（tool または承認スコープと、正規化した引数）が
  違う呼び出しを `ModelBehaviorError` で拒否する。指紋まで同じ場合は拒否せず、出力済みなら黙って実行を
  省く。範囲は run 単位で、Session を跨いだ run は対象外である。
- このため固定の既定値のまま 2 回以上 tool call する構成は SDK 0.22.3 で失敗する（実測の内訳は ADR-0043 の
  「要修正」の call_id の項を正とする）。

### 守るべき契約

- ステートレス契約（ADR-0019 の「3. 入力は `ModelRequest` として渡し、多ターン判別フィールドを入力から
  導出する」と、`docs/QUALITY-GUARANTEES.md` の決定的応答モデルのステートレス契約の行）: 同一インスタンスの
  再実行・複数 Agent での共有で応答が変わらない。インスタンスに可変な状態を持たせない。
- 入力が刈り込まれる構成（handoff の `input_filter`、`nest_handoff_history`、`call_model_input_filter`）でも
  成立すること。
- 公開シンボル・ビルダのシグネチャと既定引数は変えない。

### 検討した方式

| 方式 | 内容 | 判定と理由 |
|---|---|---|
| 固定 ID の維持と案内の拡張のみ | 既定値は `call_deterministic` のまま、明示指定を案内する | 却下。SDK 0.20.0 以降では既定値のまま 2 回以上 tool call する構成がすべて失敗し、同じ引数の繰り返しは `MaxTurnsExceeded` までの分かりにくい失敗になる |
| インスタンスのカウンタで採番 | 呼び出しごとに連番を振る | 却下。ステートレス契約を破る |
| ターン番号・tool 出力件数と連番から導く | 入力から数えた値を材料にする | 却下。入力が刈り込まれる構成で失敗した（ADR-0043 の実測） |
| instructions と入力全体のハッシュと連番から導く | 要求のみを材料にする | 採用案に包含。ADR-0043 の実測したシナリオすべてで成立したが、instructions と入力が同じで応答が違うケース（tools や model_settings で分岐するルール、非純粋なルール）で衝突する |
| `ModelRequest` の全フィールドのハッシュから導く | 要求のフィールドをすべて材料にする | 却下。`user_text` / `turn` / `tool_outputs` は `input` から導出する値で冗長である。`model_settings` / `tools` / `handoffs` / `output_schema` は SDK の不透明値で安定した直列化の手段が無い（repr はプロセスごとに変わるアドレスを含みうる。属性へ触ると `ModelRequest` の「lib はフィールドの属性へ触らない」方針に反する） |
| 既定値の撤去（call_id の必須化） | ビルダの `call_id` を必須にする | 却下。衝突は原理的に起きないが、利用者のコード修正が要り、ビルダの既定値の保証（`docs/QUALITY-GUARANTEES.md` の既定 id 値の行）も退役する |
| **instructions・正規化した入力・当該 tool call の位置・名前・引数のハッシュから導く** | 要求と応答内容を材料にする | **採用**（Decision） |

採用した方式の根拠:

- 要求だけを材料にする方式で残る衝突は「instructions と入力が同じで応答が違う」ケースである。応答側の
  tool call（位置・名前・引数）を材料に入れると、応答が違えば call_id も違うので、分岐の原因によらずこの
  衝突は消える見込みである（実装時にテストで固定する）。
- 残る衝突は材料がすべて同じケースに限られる。このとき SDK の指紋（tool と正規化した引数）も同じなので、
  SDK は拒否しない見込みである（SDK の拒否条件からの推論。実装時に SDK 0.22.x で確認する）。
- 位置（応答 `output` 内の index）を材料に入れるので、1 応答に既定 call_id の tool call が複数ある場合
  （利用者が `output` を連結した場合）も衝突しない。
- 材料は要求と応答だけで、インスタンスの状態を使わないので、ステートレス契約を保つ。

## Decision

### 1. 置換は `DeterministicResponseModel.get_response` の後処理で行い、ビルダは変えない

- `get_response` がルール関数の応答を返す直前に、純関数の後処理を 1 つ置く。`stream_response` は
  `get_response` を経由するので両経路に効く。
- ビルダは要求を知らないので、要求由来の値を作れない。ビルダの戻り値と `DEFAULT_CALL_ID` の値は変えない。
- 元の `ModelResponse` と item は変更しない（`dataclasses.replace` と pydantic の `model_copy(update=...)` で
  新しいものを作る）。ルール関数がモジュール定数の応答を使い回す場合に、呼び出し間で値が漏れないように
  するためである。

### 2. 置換の対象

- `type == "function_call"` かつ `call_id == DEFAULT_CALL_ID` の item だけを置換する。明示した call_id
  （`tool_call_response(..., call_id=...)`・`multi_tool_call_response` / `mixed_response` の call_id）は
  置換しない。
- item id（`fc_deterministic`）は変えない。SDK が拒否するのは call_id の再利用である（ADR-0043）。item id の
  重複を SDK 0.22.x が拒否しないことは実装時に確認する。
- 後処理は既定 call_id の item があるときだけ材料を組む。該当 item が無い応答では直列化もしない。

### 3. 導出値

- 材料は `{instructions, input, index, name, arguments}` とする。`instructions` は要求の
  `system_instructions`、`input` は既存の入力正規化（`_input_items`）の結果、`index` は応答 `output` 内の
  位置、`name` と `arguments` は当該 tool call の名前と引数の文字列である。
- 導出値は `call_deterministic_` に、材料を直列化した文字列の SHA-256 の 16 進表記の先頭 24 文字を連ねた
  ものとする。接頭辞は ADR-0019 の禁止語（`Fake` / `Mock` / `Dummy` / `workflow` / `wf`）を含まない。

### 4. 材料の直列化

- `json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_canonical)` で
  直列化する。数値・文字列・真偽値・None・list・dict は `json.dumps` の標準の扱いで決定的で、dict の順序には
  依存しない。
- 入力の正規化は SDK の入力変換（決定時点の 0.17.4 では `ItemHelpers.input_to_new_input_list` と、その内部の
  dump 互換化）を通る。0.17.4 のソースを読んだ結果では、dict は値を再帰的に変換した dict、list / tuple は
  要素を再帰的に変換した list、str / bytes / bytearray 以外の Iterable は list になり（pydantic の
  `BaseModel` は `[フィールド名, 値]` の組の列になる）、それ以外はそのまま残る。
- `_canonical` は次の規則だけを持つ。
  - pydantic の `BaseModel` は `model_dump(mode="json")` にする。0.17.4 ではこの分岐に到達しない（上の list 化が
    先に効く）。SDK の入力変換が `BaseModel` を list 化しなくなった場合への防御である。SDK 0.22.x での到達可否は
    実装時に確認する。
  - それ以外（JSON 化できない値）は `TypeError` を送出する。`str` / `repr` へは落とさない。アドレスを含みうる
    値でプロセスごとに call_id が変わり、決定性を壊すためである。0.17.4 で実際にここへ落ちるのは bytes /
    bytearray と、Iterable でないオブジェクト（dict の値や list の要素に入ったものを含む）である。
- `TypeError` の文言には「既定 call_id を導出できない入力なので、ビルダの `call_id` 引数で明示すること」を
  含める。既定 call_id を使わない応答ではこの経路に入らない。

### 5. 公開契約との関係

- `DEFAULT_CALL_ID` の値、ビルダの既定引数と戻り値は変えない。ADR-0019 の「8. 命名と既定 id 値」の表の
  call_id の値は、ビルダが返す値としてはそのまま成り立つ。
- `DeterministicResponseModel` が返す応答（`Runner` 経由・`get_response` 経由で観測される応答）の既定
  call_id は導出値になる。ADR-0019 の「8」のうち、この値に関する部分を本 ADR が置き換える。
- ルール関数が受け取る値も導出値になる。次ターンの `request.input` 中の function_call /
  function_call_output と、`request.tool_outputs` の call_id がこれに当たる。
- ビルダの docstring にある handoff 併用時の注意は、「既定値のままでも `DeterministicResponseModel` 経由では
  呼び出しごとに導出値になる。値を固定したい・call_id で tool 結果を探したい場合は明示する」の趣旨に改める。

### 6. テストヘルパ

- `tests/_helpers/fake_model.py` の `FakeModel` / `ChoiceAwareModel` は `DeterministicResponseModel` を経由しない
  キュー消費型のステートフルな非公開ヘルパで、ADR-0019 の「9」で公開せず残置すると決めている。状態を持つ
  採番でよいので、`FakeModel` はキューへ積むたびに増える連番から `call_fake_<n>`、`ChoiceAwareModel` は既存の
  呼び出し回数から `call_choice_<n>` を明示的に渡す。
- 明示 call_id を使うヘルパ（承認系の QueuedFakeModel）は変えない。

## Consequences

- + 既定 call_id のまま 1 run で 2 回以上 tool call しても、SDK の call_id 再利用拒否に当たらない。
- + ステートレス契約を保つ。材料は要求と応答だけで、同一インスタンスの再実行では同じ call_id の列になる。
- + ビルダの既定 id 値の保証（`docs/QUALITY-GUARANTEES.md` の既定 id 値の行）は変わらず成り立つ。
- - breaking: `DeterministicResponseModel` 経由で観測される既定 call_id が導出値になる。
  `call_id == "call_deterministic"`（または `DEFAULT_CALL_ID`）で tool 結果を探すルール関数は一致しなくなる。
  回避は、tool 呼び出し時に call_id を明示してその値で探すことである。
- - 既定 call_id の item を含む応答で、入力に JSON 化できない要素があると `TypeError` になる。回避は call_id の
  明示である。
- - 既定 call_id の item を含む応答では、呼び出しごとに材料の直列化とハッシュ計算が走る。
- - 残る限界（回避はいずれも call_id の明示）:
  - 別の Agent が、同じ instructions・同じ入力・同じ tool 名・同じ引数で呼ぶと call_id が同じになる。SDK の
    指紋に承認スコープ等の Agent 依存要素が入ると拒否されうる（未実測）。
  - 利用者が明示で `call_id="call_deterministic"` を渡した場合も既定値と区別できず置換される（値を目印に
    使うため）。
  - 材料が完全一致する呼び出し（例: `call_model_input_filter` で入力を毎回同じ形に刈り込み、同じ tool を同じ
    引数で呼び続けるルール）では、2 回目以降の tool 実行を SDK が黙って省く。
  - 入力正規化が空列へ倒す入力（list へ正規化できない入力）は入力の材料がすべて `[]` になり、instructions と
    応答内容が同じなら call_id が衝突する。

## Confirmation

本 ADR は設計フェーズで受理したものであり、以下の強制手段は実装フェーズで追加する対象である。追加した
テストは `docs/QUALITY-GUARANTEES.md` へ登録する（source = ADR-0044）。個別 assert とテスト名の確定はテスト
実装時に行い、一次情報は各テストの docstring とする。

- 既定 call_id のまま 1 run で異なる引数の tool call を 2 回行っても、観測される call_id がすべて異なり、
  いずれも既定値そのものでないことを pin するテスト
- 同じ引数の tool call を 2 ターン続けても call_id が異なることを pin するテスト
- 入力が刈り込まれる構成（handoff の `input_filter`、`nest_handoff_history`、`call_model_input_filter`）で
  call_id が run 内で一意であることを pin するテスト
- 同一インスタンスで 2 回 run したときに call_id の列が一致すること（決定性）を pin するテスト
- 明示した call_id が置換されず、モジュール定数の応答を返すルール関数で定数側の item が変更されないことを
  pin するテスト
- `Runner.run_streamed` の経路でも置換されることを pin するテスト

ADR-0019 の Confirmation が挙げるステートレス契約と既定 id 値のテストは、本 ADR の下でも変更せず成り立つ。
