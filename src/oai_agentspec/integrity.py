"""runtime インテグリティ防御（lockdown 1 関数集約）。

oai-agentspec の稼働中改竄を fail-closed で検知する高レベルヘルパー。
6 段順次（root verify → store verify+preload → libs detect → custom checks →
registry freeze → workflow freeze）で実行し、最初の違反で残りの段はスキップする。

詳細仕様は ``docs/integrity.md`` を参照。本モジュールは標準 lib のみ依存とし、
``agents`` / ``openai`` を import しない（NFR-4 SDK 隔離）。
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import logging
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .constants import (
    INTEGRITY_LOGGER_NAME,
    INTEGRITY_MANIFEST_RELATIVE_PATH,
    INTEGRITY_REJECTED_HASH_ALGORITHMS,
)

if TYPE_CHECKING:
    from .prompts import PromptStore
    from .registry import AgentRegistry
    from .workflow import WorkflowGraph

logger = logging.getLogger(INTEGRITY_LOGGER_NAME)

IntegrityCheck = Callable[[], None]
"""利用者が ``lockdown(checks=[...])`` 経由で渡す検知関数のシグネチャ規約。

違反時は ``IntegrityError``（または継承例外）を raise する契約。
"""


class IntegrityError(Exception):
    """ファイル整合性違反の基底例外。

    ``lockdown`` の root verify / libs detect / custom check 段で raise される。
    """


class PromptTemplateIntegrityError(IntegrityError):
    """``PromptStore`` manifest 不一致時に raise される例外。

    ``lockdown`` の store verify 段で発生する。
    """


_STAGE_ROOT_VERIFY: Final[str] = "root_verify"
_STAGE_STORE_VERIFY: Final[str] = "store_verify"
_STAGE_LIBS_DETECT: Final[str] = "libs_detect"
_STAGE_CUSTOM_CHECKS: Final[str] = "custom_checks"
_STAGE_REGISTRY_FREEZE: Final[str] = "registry_freeze"
_STAGE_WORKFLOW_FREEZE: Final[str] = "workflow_freeze"

_HASH_CHUNK_SIZE: Final[int] = 65536


# ----------------------------------------------------------------------
# 低レベル helper（非公開）
# ----------------------------------------------------------------------
def _parse_sha256_manifest(manifest_path: Path) -> dict[str, str]:
    """sha256sum 互換 manifest を解析する。

    フォーマット: ``<sha256>  <relative-path>`` 行（GNU coreutils ``sha256sum`` 互換）。
    空行や ``#`` 始まりのコメント行はスキップする。

    Args:
        manifest_path: manifest ファイルの絶対パス。

    Returns:
        相対パス -> sha256 hex digest（小文字）の dict。

    Raises:
        IntegrityError: manifest 不在・読み取り失敗・パース不能な行を含む場合。
    """
    if not manifest_path.is_file():
        raise IntegrityError(f"manifest が見つかりません: {manifest_path}")
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IntegrityError(f"manifest 読み取りに失敗しました: {manifest_path} ({exc})") from exc

    entries: dict[str, str] = {}
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip("\r\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # sha256sum 出力は "<hex>  <path>"（2 スペース区切り）。
        parts = line.split("  ", 1)
        if len(parts) != 2:
            # フォールバック: 単一スペース区切り（一部実装互換）。
            parts = line.split(None, 1)
        if len(parts) != 2:
            raise IntegrityError(
                f"manifest のパースに失敗しました: {manifest_path} 行 {lineno}: {raw_line!r}"
            )
        digest, relative = parts[0].strip().lower(), parts[1].strip()
        if not digest or not relative:
            raise IntegrityError(
                f"manifest の行が不正です: {manifest_path} 行 {lineno}: {raw_line!r}"
            )
        # バイナリモードの先頭 "*" 付与に対応（sha256sum -b）。
        if relative.startswith("*"):
            relative = relative[1:]
        entries[relative] = digest
    return entries


def _hash_file(path: Path, algorithm: str) -> bytes:
    """ファイルを streaming hash で読み取り raw digest を返す共通 helper。

    Args:
        path: hash 計算対象のファイル（シンボリックリンクは解決後の target）。
        algorithm: ``hashlib`` のアルゴリズム名。

    Returns:
        raw digest（バイナリ）。

    Raises:
        IntegrityError: アルゴリズム未サポート、またはファイルが読めない場合。
    """
    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise IntegrityError(f"未サポートの hash アルゴリズム: {algorithm}") from exc
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(_HASH_CHUNK_SIZE), b""):
                hasher.update(chunk)
    except OSError as exc:
        raise IntegrityError(f"ファイル読み取りに失敗しました: {path} ({exc})") from exc
    return hasher.digest()


def _compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """ファイルの hash hex digest を計算する（sha256sum 互換 manifest 用）。

    GNU coreutils ``sha256sum`` 互換 manifest（FR-1 PromptStore manifest / FR-4
    任意 path manifest）の照合で使う。PEP 376 RECORD 照合には
    ``_compute_file_hash_b64`` を使う（hash 形式が異なるため）。

    Args:
        path: hash 計算対象のファイル（シンボリックリンクは解決後の target）。
        algorithm: ``hashlib`` のアルゴリズム名（既定 ``sha256``）。

    Returns:
        hex digest（小文字）。

    Raises:
        IntegrityError: ファイルが読めない場合。
    """
    return _hash_file(path, algorithm).hex()


def _compute_file_hash_b64(path: Path, algorithm: str = "sha256") -> str:
    """ファイルの hash を ``urlsafe-base64-nopad`` 形式で計算する（PEP 376 RECORD 互換）。

    PEP 376 / PEP 627 は RECORD の hash 値を ``urlsafe-base64`` の **パディング無し**
    形式で格納する。``_compute_file_hash``（hex）とは形式が異なるため、配布物照合
    （``_distribution_check``）では本 helper を使う。

    Args:
        path: hash 計算対象のファイル（シンボリックリンクは解決後の target）。
        algorithm: ``hashlib`` のアルゴリズム名（既定 ``sha256``）。

    Returns:
        urlsafe-base64 形式の digest（パディング ``=`` を除去した ASCII 文字列）。

    Raises:
        IntegrityError: ファイルが読めない場合。
    """
    digest = _hash_file(path, algorithm)
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _resolve_path(path: Path) -> Path:
    """シンボリックリンクを解決した実体パスを返す。

    Args:
        path: 解決対象のパス。

    Returns:
        ``resolve()`` 済みパス（リンクでなければ self）。

    Raises:
        IntegrityError: シンボリックリンクの target が存在しない場合。
    """
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IntegrityError(f"パス解決に失敗しました: {path} ({exc})") from exc
    return resolved


def _ensure_regular_file_or_symlink(path: Path) -> None:
    """通常ファイル / シンボリックリンク以外（FIFO / デバイス / ソケット等）なら raise。

    Args:
        path: 検査対象のパス。

    Raises:
        IntegrityError: 通常ファイル / シンボリックリンク以外の特殊ファイルを検出した場合。
    """
    try:
        st = path.lstat()
    except OSError as exc:
        raise IntegrityError(f"ファイル種別の確認に失敗しました: {path} ({exc})") from exc
    mode = st.st_mode
    if stat.S_ISLNK(mode) or stat.S_ISREG(mode):
        return
    if stat.S_ISDIR(mode):
        # ディレクトリは走査対象として呼び出し側で扱う（ここでは raise しない）。
        return
    raise IntegrityError(
        f"通常ファイル / シンボリックリンク以外の特殊ファイルを検出しました: {path}"
    )


def _supported_hash_algorithm(name: str) -> str:
    """RECORD で許容する hash アルゴリズム名を検証して正規化名を返す。

    ``hashlib.algorithms_guaranteed`` に含まれ、かつ md5 / sha1 でないものだけを許容する。

    Args:
        name: RECORD の ``<alg>=<value>`` 形式から取り出したアルゴリズム名。

    Returns:
        正規化された（小文字）アルゴリズム名。

    Raises:
        IntegrityError: md5 / sha1 / その他サポート対象外アルゴリズムを検出した場合。
    """
    normalized = name.strip().lower()
    if not normalized:
        raise IntegrityError("RECORD の hash アルゴリズム名が空です")
    if normalized in INTEGRITY_REJECTED_HASH_ALGORITHMS:
        raise IntegrityError(f"暗号学的に弱い hash アルゴリズムは拒否されます: {normalized}")
    if normalized not in hashlib.algorithms_guaranteed:
        raise IntegrityError(f"未サポートの hash アルゴリズムです: {normalized}")
    return normalized


def _iter_files(root: Path) -> list[Path]:
    """root 配下の全エントリを走査し、特殊ファイル検出時は raise する。

    通常ファイル / シンボリックリンクのみを収集する。``.integrity/`` 配下の manifest 自身は
    照合対象から除外する（manifest が自分を検証することはできない）。

    Args:
        root: 走査開始ディレクトリ。

    Returns:
        root に対する相対パス一覧（通常ファイル / シンボリックリンクのみ）。

    Raises:
        IntegrityError: 特殊ファイル（FIFO / デバイス / ソケット等）を検出した場合。
    """
    files: list[Path] = []
    integrity_dir = root / ".integrity"
    for entry in root.rglob("*"):
        # manifest 自身は照合対象から除外（自己照合は意味を持たない）。パス名ベースで
        # 判定して resolve 失敗時の不確実な fallback を避ける。
        if entry.is_relative_to(integrity_dir):
            continue
        try:
            st = entry.lstat()
        except OSError as exc:
            raise IntegrityError(f"ファイル種別の確認に失敗しました: {entry} ({exc})") from exc
        mode = st.st_mode
        if stat.S_ISDIR(mode):
            continue
        if not (stat.S_ISLNK(mode) or stat.S_ISREG(mode)):
            raise IntegrityError(
                f"通常ファイル / シンボリックリンク以外の特殊ファイルを検出しました: {entry}"
            )
        files.append(entry)
    return files


# ----------------------------------------------------------------------
# 3 ファクトリ（非公開）
# ----------------------------------------------------------------------
def _verify_directory_against_manifest(
    root: Path,
    manifest: Path,
    *,
    exception_factory: Callable[[str], IntegrityError] = IntegrityError,
) -> None:
    """root 配下を manifest と sha256 照合する。

    - root 配下のファイル全件を走査
    - manifest に未掲載のファイルがあれば違反
    - manifest 掲載のファイルが存在しなければ違反
    - sha256 不一致があれば違反
    - シンボリックリンクは target を解決して照合
    - 特殊ファイル（FIFO / デバイス / ソケット等）が存在すれば違反

    Args:
        root: 検証対象のディレクトリ。
        manifest: ``<root>/.integrity/sha256.manifest`` 等の manifest ファイル。
        exception_factory: 違反時に raise する例外を生成する callable。
            既定は ``IntegrityError``。

    Raises:
        IntegrityError: ``exception_factory`` の戻り値型で raise される。
    """
    if not root.exists() or not root.is_dir():
        raise exception_factory(f"検証対象のディレクトリが存在しません: {root}")

    try:
        expected = _parse_sha256_manifest(manifest)
    except IntegrityError as exc:
        # manifest 不在 / 破損も exception_factory 系統で再 raise（公開契約の例外型を維持）。
        # store verify では PromptTemplateIntegrityError を保つ。
        raise exception_factory(str(exc)) from exc
    # manifest 内の相対パスを正規化（OS 区切り依存を排除し POSIX 表記に統一）。
    expected_normalized = {p.replace("\\", "/"): h for p, h in expected.items()}

    try:
        files = _iter_files(root)
    except IntegrityError as exc:
        # 走査自体の失敗（特殊ファイル検出等）も exception_factory 系統で再 raise。
        raise exception_factory(str(exc)) from exc

    seen_relative: set[str] = set()
    for path in files:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise exception_factory(
                f"root 配下にないパスを検出しました: {path}（root={root}）"
            ) from exc
        seen_relative.add(relative)
        if relative not in expected_normalized:
            raise exception_factory(
                f"manifest 未掲載のファイルが存在します: {relative}（root={root}）"
            )
        target = _resolve_path(path) if path.is_symlink() else path
        actual = _compute_file_hash(target, algorithm="sha256")
        if actual != expected_normalized[relative]:
            raise exception_factory(
                f"sha256 不一致: {relative}（expected={expected_normalized[relative]} "
                f"actual={actual} root={root}）"
            )

    missing = set(expected_normalized) - seen_relative
    if missing:
        missing_sample = sorted(missing)[0]
        raise exception_factory(
            f"manifest 記載のファイルが存在しません: {missing_sample}（root={root} "
            f"missing_count={len(missing)}）"
        )


def _prompt_manifest_check(manifest: Path, root: Path) -> IntegrityCheck:
    """``PromptStore`` root 用の manifest 照合 check を返す。

    Args:
        manifest: ``<store.root>/.integrity/sha256.manifest`` のパス。
        root: ``PromptStore.root``。

    Returns:
        呼ぶと ``PromptTemplateIntegrityError`` を raise しうる ``IntegrityCheck``。
    """

    def check() -> None:
        _verify_directory_against_manifest(
            root,
            manifest,
            exception_factory=PromptTemplateIntegrityError,
        )

    return check


def _path_manifest_check(root: Path, manifest: Path) -> IntegrityCheck:
    """任意 path 用の manifest 照合 check を返す。

    Args:
        root: 検証対象ディレクトリ。
        manifest: 照合に使う manifest ファイル。

    Returns:
        呼ぶと ``IntegrityError`` を raise しうる ``IntegrityCheck``。
    """

    def check() -> None:
        _verify_directory_against_manifest(
            root,
            manifest,
            exception_factory=IntegrityError,
        )

    return check


def _distribution_check(name: str) -> IntegrityCheck:
    """PEP 376 RECORD で配布物を照合する check を返す。

    PEP 376 / PEP 627 に従い、hash 値は ``urlsafe-base64-nopad`` 形式で比較する
    （``_compute_file_hash_b64`` を使う）。RECORD の hash フィールドが空のエントリ
    （RECORD 自身など）はスキップする。

    Args:
        name: 配布物名（``importlib.metadata.distribution`` に渡す名前）。

    Returns:
        呼ぶと ``IntegrityError`` を raise しうる ``IntegrityCheck``。
    """

    def check() -> None:
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise IntegrityError(f"配布物が見つかりません: {name}") from exc

        files = dist.files
        if files is None:
            raise IntegrityError(f"配布物の RECORD（files メタデータ）がありません: {name}")

        base = Path(dist.locate_file(""))
        for entry in files:
            hash_field = getattr(entry, "hash", None)
            if hash_field is None:
                continue
            mode_value = getattr(hash_field, "mode", None)
            value = getattr(hash_field, "value", None)
            if not mode_value or not value:
                # 空 hash（RECORD 自身など）は PEP 376 でスキップ対象。
                continue
            algorithm = _supported_hash_algorithm(mode_value)
            target_path = Path(dist.locate_file(entry))
            if not target_path.exists():
                raise IntegrityError(
                    f"配布物のファイルが存在しません: {name} {entry}（base={base}）"
                )
            resolved = _resolve_path(target_path) if target_path.is_symlink() else target_path
            # PEP 376 RECORD は urlsafe-base64-nopad で hash を格納する仕様。
            actual = _compute_file_hash_b64(resolved, algorithm=algorithm)
            if actual != value:
                raise IntegrityError(
                    f"配布物の hash 不一致: distribution={name} file={entry} "
                    f"algorithm={algorithm} expected={value} actual={actual}"
                )

    return check


def _detect_used_distributions() -> set[str]:
    """``sys.modules`` 全件を ``packages_distributions()`` で配布物名にマップする。

    環境依存の I/O 失敗（``OSError`` 等）は ``IntegrityError`` に包んで fail-closed する
    （検査対象 0 件で成功扱いになる偽陰性を防ぐ）。一方、``AttributeError`` /
    ``TypeError`` / ``KeyError`` 等の bug 由来例外は捕捉せず propagate させ、stdlib /
    依存ライブラリ側のバグを「環境依存の I/O 失敗」に紛れ込ませない。

    Returns:
        現在 import 済みの全配布物名の集合（重複排除）。

    Raises:
        IntegrityError: ``packages_distributions()`` の I/O 失敗（``OSError`` 系）。
    """
    try:
        mapping = importlib.metadata.packages_distributions()
    except OSError as exc:
        # I/O / 環境依存の失敗のみ wrap。bug 由来（AttributeError 等）は propagate。
        raise IntegrityError(f"配布物メタデータ取得に失敗しました（環境依存）: {exc}") from exc
    found: set[str] = set()
    for module_name in list(sys.modules):
        top = module_name.split(".", 1)[0]
        for dist_name in mapping.get(top, ()):
            found.add(dist_name)
    return found


# ----------------------------------------------------------------------
# 公開関数
# ----------------------------------------------------------------------
def lockdown(
    root: Path,
    store: PromptStore | None = None,
    registry: AgentRegistry | None = None,
    workflow: WorkflowGraph | None = None,
    *,
    libs: bool = True,
    checks: list[IntegrityCheck] | None = None,
) -> None:
    """6 段順次・fail-closed で runtime インテグリティを守る。

    実行順序:

    1. root verify: ``<root>/.integrity/sha256.manifest`` と root 配下を sha256 照合。
    2. store verify + preload: ``store`` 指定時、``<store.root>/.integrity/sha256.manifest``
       と照合し、全テンプレを eager-load して ``_cache`` 充填。
    3. libs detect: ``libs=True`` 時、``sys.modules`` 配下配布物の PEP 376 RECORD を照合。
    4. custom checks: ``checks`` 指定時、リスト順に発火（fail-closed）。
    5. registry freeze: ``registry`` 指定時、``registry.freeze()`` を呼ぶ（冪等）。
    6. workflow freeze: ``workflow`` 指定時、``workflow.freeze()`` を呼ぶ（冪等）。

    最初の違反で残りの段はスキップされる（fail-closed）。冪等性: 同一引数で再呼び出し
    すると検証系（1〜4 段）は毎回再実行、freeze 系（5〜6 段）は冪等 no-op となる。

    詳細仕様は ``docs/integrity.md`` を参照。

    Args:
        root: 検証対象のルートディレクトリ。
        store: 検証対象の ``PromptStore``。``None`` で段 2 をスキップ。
        registry: 凍結対象の ``AgentRegistry``。``None`` で段 5 をスキップ。
        workflow: 凍結対象の ``WorkflowGraph``。``None`` で段 6 をスキップ。
        libs: ``True`` で ``sys.modules`` 配下配布物の RECORD 照合を実行。
        checks: 利用者の独自検知関数リスト。``None`` または空で段 4 をスキップ。

    Raises:
        IntegrityError: root verify / libs detect / custom check で違反検出時。
        PromptTemplateIntegrityError: store verify で manifest 不一致検出時。
    """
    start_time = time.monotonic()
    logger.info(
        "lockdown.start",
        extra={
            "root": str(root),
            "libs": libs,
            "store": store is not None,
            "registry": registry is not None,
            "workflow": workflow is not None,
            "checks_count": len(checks) if checks else 0,
        },
    )

    current_stage = _STAGE_ROOT_VERIFY
    try:
        # 1. root verify
        _run_stage(
            current_stage,
            lambda: _verify_directory_against_manifest(
                root, root / INTEGRITY_MANIFEST_RELATIVE_PATH
            ),
        )

        # 2. store verify + preload
        if store is not None:
            current_stage = _STAGE_STORE_VERIFY
            store_root = Path(store.root)
            store_manifest = store_root / INTEGRITY_MANIFEST_RELATIVE_PATH
            store_check = _prompt_manifest_check(store_manifest, store_root)

            def _store_verify() -> None:
                # `_verify_integrity` が fail-closed 順次実行 helper として
                # check 群を順次発火する責務を持つ（設計サマリ準拠）。
                store._verify_integrity([store_check])  # noqa: SLF001
                store._preload()  # noqa: SLF001

            _run_stage(current_stage, _store_verify)

        # 3. libs detect
        if libs:
            current_stage = _STAGE_LIBS_DETECT

            def _libs_detect() -> None:
                for name in sorted(_detect_used_distributions()):
                    _distribution_check(name)()

            _run_stage(current_stage, _libs_detect)

        # 4. custom checks
        if checks:
            current_stage = _STAGE_CUSTOM_CHECKS

            def _custom_checks() -> None:
                for fn in checks:
                    fn()

            _run_stage(current_stage, _custom_checks)

        # 5. registry freeze
        if registry is not None:
            current_stage = _STAGE_REGISTRY_FREEZE
            _run_stage(current_stage, registry.freeze)

        # 6. workflow freeze
        if workflow is not None:
            current_stage = _STAGE_WORKFLOW_FREEZE
            _run_stage(current_stage, workflow.freeze)

    except Exception as exc:
        logger.warning(
            "lockdown.violation",
            extra={
                "stage": current_stage,
                "error_type": type(exc).__name__,
                "identifier": str(exc),
            },
        )
        raise

    duration_ms = int((time.monotonic() - start_time) * 1000)
    logger.info("lockdown.complete", extra={"duration_ms": duration_ms})


def _run_stage(stage: str, fn: Callable[[], object]) -> None:
    """各段の start / success を構造化ログに出しつつ ``fn`` を実行する。

    Args:
        stage: 段の識別子（``root_verify`` 等）。
        fn: 当該段の実装関数（引数なし）。

    Raises:
        Exception: ``fn`` が raise した具象例外をそのまま伝播する（fail-closed の
            ため握り潰さない）。具象型は呼び出し元の段に依存し、``IntegrityError`` /
            ``PromptTemplateIntegrityError`` / ``RegistryFrozenError`` /
            ``WorkflowFrozenError`` 等を含む。
    """
    logger.debug("lockdown.stage", extra={"stage": stage, "status": "start"})
    fn()
    logger.debug("lockdown.stage", extra={"stage": stage, "status": "success"})


__all__ = [
    "IntegrityCheck",
    "IntegrityError",
    "PromptTemplateIntegrityError",
    "lockdown",
]
