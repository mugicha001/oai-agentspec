"""データ源インテグリティ（取得結果の改竄検知）の最小例（実 API）。

RAG 検索結果・会話メモリ・静的資産を返すツールの出力が、取得時点から改変されていないかを
ツール出力 guardrail（`Boundary.TOOL_OUTPUT`）で検知する。新機構は使わず、既存の
`tool_guardrail(detector, on="output")` + `predicate_detector` へ利用者の照合述語を DI する
だけで組み立てる。

検知の実行主体は SDK ネイティブのツール出力 guardrail フックで、lib は接着のみを提供する。
**ベースライン（取得時点の正解ダイジェスト）の生成・保管・保護・失効は利用者責務**であり、
lib はハッシュ機構もベースライン保存機構も持たない。本例ではモジュール読み込み時に 1 回だけ
ベースラインを算出しているが、実運用では取得時点に生成し読み取り専用配置へ保管する。

攻撃者が対象データとベースラインの双方を書き換えられる環境では検知保証が失効する
（`docs/integrity.md` の manifest 信頼境界と同じ前提）。また本例はデモのため検査対象と同じ
`_CANONICAL` からベースラインを算出している。取得元から取得したデータでベースラインを作ると
照合が恒真になり検知が無効化されるため、実運用では取得時点の値を別途保管したものを読み込む。

trip 時の挙動は `on_trip` で選ぶ（既定 'reject'）。改竄検知では続行前提の 'reject' ではなく
`on_trip="raise"` を明示し `ToolOutputGuardrailTripwireTriggered` で実行を中断する（fail-closed）。
trip は改竄の証明ではなく「取得時点の本文と一致しない」ことの検知である。ツール実行が失敗した
場合も、SDK が返すエラーメッセージがベースラインに一致しないため trip する。

本例では同一の detector と同一の `on_trip="raise"` を装着した正常系ツールと改竄系ツールを
`function_tool` で 2 本作り、本文の違いだけで trip の有無が分かれることを示す。
既存ツール（`as_tool` 等 `function_tool` で定義し直せないもの）へ後付けする場合は `guard_tool`
（コメント参照）。

Azure OpenAI の環境変数（AZURE_OPENAI_* 。examples/_shared/_azure.py 参照）を設定して実行:
    uv run python examples/guardrails/09_data_integrity_detector.py

導入: pip install 'oai-agentspec[guardrails]'（依存ゼロ opt-in extra）。
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

from agents import Runner, ToolOutputGuardrailTripwireTriggered

from oai_agentspec import AgentRegistry, AgentSpec, function_tool
from oai_agentspec.runtime.guardrails import predicate_detector, tool_guardrail

# examples/ 共有ヘルパ _azure を解決するため _shared を import パスへ。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from _azure import azure_model  # noqa: E402

# 従量課金 API に接続するため、開始からの絶対上限を script 内 watchdog で強制する
# （想定所要 60s + マージン）。
WATCHDOG_SECONDS = 60

# 取得時点の正解の本文（doc_id -> text）。実運用では取得時点に生成し読み取り専用配置へ保管する。
_CANONICAL: dict[str, str] = {
    "DOC-1042": "退職金は基本給に勤続年数を乗じた額を基礎として計算します。",
}

# ベースライン（doc_id -> sha256 hex）。モジュール読み込み時に 1 回だけ算出する。
BASELINE: dict[str, str] = {
    doc_id: hashlib.sha256(text.encode("utf-8")).hexdigest() for doc_id, text in _CANONICAL.items()
}
# 照合に使う実効値（読み込み時のスナップショット）。単一資産前提のためここで doc_id を落として
# いる。ベースラインを更新する運用では再算出が必要。
KNOWN_DIGESTS = frozenset(BASELINE.values())

# 配信側だけが書き換えられた本文（取得時点のベースラインは正しいまま）。
_TAMPERED: dict[str, str] = {
    doc_id: text + "ただし退職理由により減額される場合があります。"
    for doc_id, text in _CANONICAL.items()
}


def _fetch_document(doc_id: str) -> str:
    """正常系: 取得時点から改変されていない本文を返す。

    Args:
        doc_id: ドキュメント ID。

    Returns:
        ドキュメント本文。
    """
    return _CANONICAL[doc_id]


def _fetch_tampered_document(doc_id: str) -> str:
    """改竄系: 配信側だけが書き換わった本文を返す（利用者側のベースラインは正しいまま）。

    Args:
        doc_id: ドキュメント ID。

    Returns:
        改竄されたドキュメント本文。
    """
    return _TAMPERED[doc_id]


def _mismatches_baseline(text: str) -> bool:
    """返却テキスト全体の sha256 がベースラインに無ければ改竄とみなす（True で trip）。

    Args:
        text: ツールが返した出力テキスト。

    Returns:
        ベースラインのダイジェスト集合に含まれなければ True（guardrail が trip する）。

    Note:
        本例は単一資産（返却テキスト全体 = 1 ドキュメント）を前提とし、ダイジェストの集合照合で
        判定する。このため `doc_id` と本文の対応までは検証しておらず、別の正当なドキュメントへの
        すり替えは検知できない。検知器の契約は `Callable[[str], bool]` でツール引数を受け取れない
        ため、識別子の同送は利用者側の設計事項になる。資産が複数ある構成では、ツールが `doc_id` と
        ダイジェストを含む構造化出力を返し、detector がそれをパースして資産単位で照合する形を使う。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest() not in KNOWN_DIGESTS


integrity_detector = predicate_detector(
    _mismatches_baseline, reason="retrieved content does not match the acquisition-time baseline"
)


def build_registry() -> AgentRegistry:
    """正常系 / 改竄系のツールを `function_tool` で定義した 2 エージェントを登録する。

    `function_tool(_func, tool_output_guardrails=[tool_guardrail(...)])` でツール定義時に
    guardrail を宣言する。2 ツールは同じ detector・同じ `on_trip="raise"` を共有し、本文の
    違いだけで trip の有無が分かれることを示す。

    Returns:
        2 エージェントを登録済みの `AgentRegistry`。
    """
    instructions = (
        "あなたはドキュメント検索係です。ドキュメントの内容を聞かれたら必ず fetch_document "
        "ツールを使って回答してください。"
    )

    clean_tool = function_tool(
        _fetch_document,
        name_override="fetch_document",
        tool_output_guardrails=[tool_guardrail(integrity_detector, on="output", on_trip="raise")],
    )
    tampered_tool = function_tool(
        _fetch_tampered_document,
        name_override="fetch_document",
        tool_output_guardrails=[tool_guardrail(integrity_detector, on="output", on_trip="raise")],
    )

    # 既存ツール（function_tool で定義し直せない as_tool 等）への後付けは guard_tool:
    #   from oai_agentspec.runtime.guardrails import guard_tool
    #   guarded = guard_tool(existing_tool, output_detector=integrity_detector, on_trip="raise")

    # guardrail を載せても name は元のまま（差し替えるのは guardrail だけ）。
    assert clean_tool.name == "fetch_document"

    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="agent-intact",
            instructions=instructions,
            model=azure_model(),
            tools=[clean_tool],
        )
    )
    registry.register(
        AgentSpec(
            name="agent-tampered",
            instructions=instructions,
            model=azure_model(),
            tools=[tampered_tool],
        )
    )
    registry.validate()
    return registry


async def run() -> None:
    """正常系（ベースライン一致）と改竄系（ベースライン不一致で中断）の 2 経路を実行する。"""
    registry = build_registry()
    prompt = "DOC-1042 の内容を教えて。"

    print("--- 正常系: ベースライン一致 ---")
    result = await Runner.run(registry.get("agent-intact"), input=prompt)
    print("output:", result.final_output[:120])

    print("\n--- 改竄系: ベースライン不一致で中断 ---")
    try:
        await Runner.run(registry.get("agent-tampered"), input=prompt)
        print("（中断されませんでした）")
    except ToolOutputGuardrailTripwireTriggered:
        print("ツール出力 guardrail が改竄を検知して実行を中断しました")


async def main() -> None:
    try:
        await asyncio.wait_for(run(), timeout=WATCHDOG_SECONDS)
    except TimeoutError:
        print(f"watchdog: {WATCHDOG_SECONDS}s を超過したため強制終了します", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
