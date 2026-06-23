# Rationale: ハンドオフ循環解決の設計経緯

本ファイルは不変な検討経緯（immutable）を保持する archival ドキュメントである。
現在の確定仕様は `docs/architecture.md` を参照すること。本ファイルは実装変更に追随して更新しない。

## 背景となる SDK 制約

openai-agents の `Agent` は `frozen=False` であり、`handoffs` は可変 list である。
実行時、ハンドオフ候補は `get_handoffs` が毎ターン `agent.handoffs` を読み直して解決する。
このため Agent 構築後に `agent.handoffs.append(...)` で後付けした参照も実行時に有効になる。

## 検討した代替案とトレードオフ

### 案 A: コンストラクタ焼き込み（不採用）

`Agent(handoffs=[registry.get(h) for h in spec.handoffs])` の形で構築時に解決する方式。
`a -> b -> a` のような循環では `get("a")` が `get("b")` を呼び、`get("b")` が再び `get("a")` を呼ぶため
無限再帰し `RecursionError` となる。循環ユースケースを満たせないため不採用。

### 案 B: 早期登録のみ（不採用）

各 Agent を `handoffs=[]` で先に `_built` に登録してから handoffs を解決する案。
ただし「登録のみ」では、循環内の各 Agent が相手の確定インスタンスではなく
空または陳腐化したプレースホルダを指す危険があり、object identity 検証
（`a.handoffs[0] is registry.get("b")`）を満たせないケースが残る。2 パスで明示的に
append する手順が必要となるため、登録だけでは不十分と判断した。

### 案 C: 全 spec 一括 2 パス（不採用）

レジストリ内の全 spec を一括でビルドしてから handoffs を解決する案。
到達不能な spec まで巻き込んでビルドするため、遅延構築の利点を損なう。
未登録・無関係な spec のビルドエラーが起点と無関係に発生し、原因特定を難しくする。

### 案 D: 局所 2 パス遅延バインド（採用）

`get(name)` を起点に、`name` から到達可能かつ未ビルドの spec 集合のみを収集し、
その集合に対してのみ 2 パス（パス 1: 空 handoffs でビルド、パス 2: append で結線）を実行する。

採用理由:

- SDK が handoffs を実行時読みするため、後付け append が確実に成立する（案 A の再帰を回避）。
- 到達可能 spec のみに局所化することで遅延構築の利点を維持し、無関係 spec を巻き込まない（案 C の問題を回避）。
- パス 2 で明示的に確定インスタンスを append するため object identity が保証される（案 B の問題を回避）。

サブエージェント（`as_tool` 配線）もサブ Agent インスタンスを参照として取り込むため、handoff と同じビルド順序問題を持つ。よって到達可能 spec の収集および結線の依存辺は handoffs ∪ sub_agents の和集合とする。両者は参照取り込みという点で同じ局所 2 パスの枠組みで扱える。
