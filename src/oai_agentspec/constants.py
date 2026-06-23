"""ライブラリ全体で共有する定数。

不変値は `Final` で宣言する（Python 規約 4）。
"""

from __future__ import annotations

from typing import Final

# ワークフロー: 経路A（as_facade_spec）でファサード Agent に強制するツール選択。
# tool_choice は agents.ModelSettings のフィールドであり、extra に積むと build_agent の
# 未知キーガードで ValueError になるため必ず model_settings 経由で設定する（FR-9）。
WORKFLOW_TOOL_CHOICE_REQUIRED: Final[str] = "required"

# ワークフロー: 経路A のファサード Agent が最初のツール呼び出しで停止する挙動。
# tool_use_behavior は agents.Agent のフィールドなので extra で渡せる（FR-9・非対称）。
WORKFLOW_TOOL_USE_BEHAVIOR: Final[str] = "stop_on_first_tool"

# ワークフロー: 経路A の既定 input_filter が流入履歴を有界化する件数（直近 N 件）。
WORKFLOW_DEFAULT_INPUT_HISTORY_LIMIT: Final[int] = 1

# ワークフロー: 1 run の総ノード実行数の上限の既定値（超過で実行時エラー・C-5）。
# node/edge グラフはノードを 1 つ実行するたびにカウントし、この回数を超えると実行時例外を
# 送出する（無限ループ防止を兼ねる）。ループの無いグラフでもノード総数がこの値を超えると
# 停止するため、大きなグラフでは WorkflowGraph(recursion_limit=...) で引き上げる（FR-2）。
WORKFLOW_DEFAULT_RECURSION_LIMIT: Final[int] = 25

# 会話 Helper: conversation_id 生成に使う UUID プレフィックス（識別子の可読性用）。
CONVERSATION_ID_PREFIX: Final[str] = "conv-"

# runtime インテグリティ防御: ``lockdown`` が emit する構造化ログの logger 名（固定）。
# 利用者は ``logging.getLogger(INTEGRITY_LOGGER_NAME)`` で観測性基盤に接続する。
INTEGRITY_LOGGER_NAME: Final[str] = "oai_agentspec.integrity"

# runtime インテグリティ防御: root verify / store verify が参照する manifest の固定相対パス。
# GNU coreutils ``sha256sum`` 互換フォーマット（``<sha256>  <relative-path>``）を採用する。
INTEGRITY_MANIFEST_RELATIVE_PATH: Final[str] = ".integrity/sha256.manifest"

# runtime インテグリティ防御: libs detect で明示的に拒否する hash アルゴリズム名（小文字）。
# 暗号学的に弱い md5 / sha1 は PEP 376 RECORD であっても受け入れない。
INTEGRITY_REJECTED_HASH_ALGORITHMS: Final[frozenset[str]] = frozenset({"md5", "sha1"})
