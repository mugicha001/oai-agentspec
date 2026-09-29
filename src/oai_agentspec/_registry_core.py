"""registry の到達可能収集 + トランザクショナル構築（局所 2 パス + 任意の第 3 段）の共有ヘルパ。

通常ルート（`AgentRegistry`）と Realtime 専用ルート（`RealtimeAgentRegistry`）が、遅延構築の
到達可能収集アルゴリズムと 2 パス build/wire（+ 任意の第 3 段 post-process）+ 巻き戻し
セマンティクスを単一ソースで保つための最下層の
純ヘルパ。`agents` には依存せず、plain な `Mapping` / `MutableMapping`（`_specs` / `_built`）+
narrow なコールバックのみを受け取る（`_validation` / `_mermaid` と同じ共有 leaf パターン）。

両ルートは宣言型・registry を共用しないが、片側修正による挙動乖離を最も招きやすい安全
クリティカルな巻き戻し部分を単一ソース化する。差分点（依存辺プロバイダ・bare ビルド・結線・
post-process）はコールバックで注入し、`_wire` / `_require` / factory 分岐は各 registry に据え置く。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, MutableMapping


def collect_reachable(
    name: str,
    specs: Mapping[str, Any],
    built: Mapping[str, Any],
    deps: Callable[[Any], Iterable[str]],
) -> list[str]:
    """name から依存辺を辿り未ビルドの spec 名を集める（visited で循環を打ち切る）。

    到達不能 spec は含めない。spec でない依存名（factory / 未登録）は収集対象外
    （factory は get() 時に自前構築、未登録は結線フェーズでエラーになる）。差分点である
    依存辺の算出は `deps` コールバックに委ねる（通常ルート＝handoffs ∪ sub_agents ∪ dynamic
    候補・Realtime ルート＝handoffs のみ）。

    Args:
        name: 収集の起点となるエージェント名。
        specs: 登録済み spec のマッピング（`_specs`）。
        built: 構築済みキャッシュのマッピング（`_built`）。
        deps: spec から依存辺（依存先名の列）を返すプロバイダ。

    Returns:
        起点から到達可能で未ビルドの spec 名リスト（走査順）。
    """
    collected: list[str] = []
    visited: set[str] = set()
    stack = [name]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        if current not in specs:
            continue
        if current not in built:
            collected.append(current)
        for dep in deps(specs[current]):
            if dep not in visited:
                stack.append(dep)
    return collected


def build_two_pass(
    reachable: list[str],
    specs: Mapping[str, Any],
    built: MutableMapping[str, Any],
    build_bare: Callable[[Any], Any],
    wire: Callable[[Any, Any], None],
    *,
    post_process: Callable[[Any, Any], Any] | None = None,
) -> None:
    """到達可能 spec を局所 2 パス + 任意の第 3 段でトランザクショナルに構築する。

    パス 1 で handoffs 空・サブツール未注入の bare agent を `build_bare` でビルドして
    `built` に登録し、パス 2 で `wire` により handoffs / sub_agents を後付け結線する。
    `post_process` が与えられた場合は、全 spec の結線完了後の第 3 段で各 spec 1 回ずつ
    呼び、戻り値で `built` を置き換える。いずれかの段で例外が出たら本呼び出しで新規
    キャッシュした agent を巻き戻し、不完全なインスタンスを残さない（差分点＝bare ビルド・
    結線・post-process はコールバックで注入）。

    Args:
        reachable: `collect_reachable` が返した未ビルド spec 名リスト。
        specs: 登録済み spec のマッピング（`_specs`）。
        built: 構築済みキャッシュのマッピング（`_built`・本関数が破壊的に更新する）。
        build_bare: spec から handoffs 空の agent を構築するコールバック。
        wire: `(spec, agent)` を受け取り handoffs / sub_agents を後付け結線するコールバック。
        post_process: `(spec, agent)` を受け取り加工後の agent を返す第 3 段のコールバック
            （任意）。None なら第 3 段を行わない。
    """
    newly_built: list[str] = []
    try:
        # パス 1: handoffs 空・サブツール未注入でビルドして登録
        for target in reachable:
            if target not in built:
                built[target] = build_bare(specs[target])
                newly_built.append(target)
        # パス 2: handoffs / sub_agents を後付け結線
        for target in reachable:
            wire(specs[target], built[target])
        # 第 3 段: 全結線の完了後に post-process（任意）
        if post_process is not None:
            for target in reachable:
                built[target] = post_process(specs[target], built[target])
    except Exception:
        for target in newly_built:
            built.pop(target, None)
        raise
