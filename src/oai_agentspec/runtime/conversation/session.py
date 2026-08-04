"""session 管理（session_id 連動の永続化方針・agents 非依存）。

履歴は SDK `Session`（既定 `SQLiteSession`）に委ねる（NFR-2）。session_id 連動の方針:
名前付き（session_id 明示）→ 既定 `memory/` 配下にファイル永続化（再起動後 resume 可）、
無指定 → in-memory（揮発）。実際の SDK `Session` 生成は `_adapters.make_session` に委譲し、
本モジュールは agents を import しない（NFR-1。生成された `Session` は不透明型で扱う）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..._validation import validate_bool

if TYPE_CHECKING:
    # NFR-1: openai はランタイム import しない（型注釈のみ）。client は不透明値として保持する。
    from openai import AsyncOpenAI

# session_id 連動でファイル永続化する既定ディレクトリ（D3）。プロジェクト直下の見える
# フォルダに置く（隠しホームフォルダにはしない）。カレントからの相対パス。
DEFAULT_SESSION_DIR: Path = Path("memory")

# 永続化 db のファイル名（DEFAULT_SESSION_DIR 配下）。
DEFAULT_SESSION_DB_NAME: str = "conversations.db"


@dataclass(frozen=True)
class CompactionConfig:
    """compaction（履歴圧縮）設定の型付き契約（有効化フラグ・client・model）。

    `enabled` フラグで有効化を明示制御し、client/model の受け渡しと有効化判定を分離する。
    `enabled=False`（既定）なら client を渡しても圧縮しない（暗黙有効化を行わない）。
    compaction は OpenAI Responses API 専用で、client は `AsyncOpenAI` /
    `AsyncAzureOpenAI` のいずれでも Responses API を叩ければよい（不透明値として保持）。

    Attributes:
        enabled: compaction を有効化するか（既定 False）。
        client: OpenAI Responses API 互換クライアント（`AsyncOpenAI` 系）。`enabled=True`
            のとき必須。型注釈のみで `openai` をランタイム import しない（NFR-1）。
        model: 圧縮に使うモデル名。None なら SDK 既定モデルを使う（キーを渡さない）。
        options: `OpenAIResponsesCompactionSession` へ素通しする追加オプション。

    Raises:
        ValueError: `enabled=True` かつ `client` が欠けている場合（`__post_init__` で検知）。
    """

    enabled: bool = False
    client: AsyncOpenAI | None = None
    model: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """`enabled` の型と、有効化時の client 欠落を早期検知する（誤用を構築時に弾く）。

        Raises:
            ValueError: `enabled` が bool でない場合、または `enabled=True` かつ `client` が
                None の場合。
        """
        validate_bool(self.enabled, "enabled")
        if self.enabled and self.client is None:
            raise ValueError("compaction を有効化する場合は client（AsyncOpenAI 系）が必須です")


@dataclass(frozen=True)
class SessionPolicy:
    """session 生成方針（永続化先・compaction 設定の保持）。

    Attributes:
        base_dir: ファイル永続化の基底ディレクトリ（既定 `memory/`・プロジェクト直下）。
        db_name: 永続化 db ファイル名。
        persist: ファイル永続化を許可するか（既定 True）。False（揮発モード）なら
            session_id 明示でも常に in-memory にする（CLI の `--ephemeral` 用）。
        compaction: compaction 設定（`CompactionConfig`）。None または `enabled=False` で
            plain SQLite（client/model を渡しても圧縮しない）。`compaction_kwargs` で
            `_adapters.make_session` の plain 引数へ展開する。
    """

    base_dir: Path = DEFAULT_SESSION_DIR
    db_name: str = DEFAULT_SESSION_DB_NAME
    persist: bool = True
    compaction: CompactionConfig | None = None

    def __post_init__(self) -> None:
        """`persist` が bool であることを構築時に検証する。

        Raises:
            ValueError: `persist` が bool でない場合。
        """
        validate_bool(self.persist, "persist")

    def compaction_kwargs(self) -> dict[str, Any]:
        """`CompactionConfig` を `make_session` の plain kwargs へ展開する（非公開ヘルパ）。

        戻りキーは `_adapters.make_session` のシグネチャと完全一致させる。compaction が
        None なら `enable_compaction=False`（plain SQLite 経路）のみを返す。

        Returns:
            `make_session` へ `**` 展開して渡す plain kwargs。
        """
        cfg = self.compaction
        if cfg is None:
            return {"enable_compaction": False}
        return {
            "enable_compaction": cfg.enabled,
            "client": cfg.client,
            "model": cfg.model,
            "compaction_options": cfg.options,
        }

    def db_path_for(self, *, persist: bool) -> str | None:
        """永続化するかに応じた db_path を返す（揮発モードでは常に None）。

        本ポリシーが揮発モード（`self.persist=False`）なら、引数 `persist` に関わらず
        None（in-memory）を返す。

        Args:
            persist: 当該会話を永続化したいか（session_id 明示なら True）。

        Returns:
            ファイルパス文字列、または in-memory のとき None。
        """
        if not persist or not self.persist:
            return None
        self.base_dir.mkdir(parents=True, exist_ok=True)
        return str(self.base_dir / self.db_name)

    def pending_db_path(self, *, persist: bool) -> str | None:
        """HITL 中断状態の永続化先 db_path を返す（揮発会話/揮発モードでは None）。

        中断状態は会話履歴と同一 db（既定 `memory/conversations.db`）の専用テーブルに同居する
        （方針B）。永続会話（`persist=True`・session_id 明示）かつポリシーが永続化を許す場合のみ
        パスを返し、`db_path_for` と同じ判定でディレクトリも用意する。揮発会話/揮発モードは
        None（メモリのみ・復元しない・FR-10）。

        Args:
            persist: 当該会話を永続化するか（`ConversationEntry.persist`）。

        Returns:
            永続化先 db のファイルパス。揮発なら None。
        """
        return self.db_path_for(persist=persist)

    def persisted_db_path(self) -> str:
        """永続化 db のファイルパスを返す（ディレクトリは作らない）。

        session 一覧（D5）の読み取り専用列挙に使う。db ファイルが未作成でも例外にせず、
        パス文字列のみを返す（実在判定は呼び出し側 / `list_session_ids` が行う）。

        Returns:
            `base_dir/db_name` のファイルパス文字列。
        """
        return str(self.base_dir / self.db_name)


__all__ = [
    "DEFAULT_SESSION_DB_NAME",
    "DEFAULT_SESSION_DIR",
    "CompactionConfig",
    "SessionPolicy",
]
