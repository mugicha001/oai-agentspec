"""`ToolRegistry` に登録したツールをそのまま SFT データセットへ渡す例（API キー不要）。

`registry.<name>` は属性アクセス時に SDK `FunctionTool` を遅延構築して返す。`to_sft_dataset`
の `tools=` はこの `FunctionTool` を `name` / `params_json_schema` 属性のダックタイピングで
検出し、公式 tools 定義形式へ写像する。これにより **学習データの tools 定義と、推論時に
Agent へ渡すツール定義が同じ `ToolRegistry` から出る**ことを示す。

あわせて `tool_calls` 付き assistant / role `"tool"` を含む複数ターンのケース、`weight` に
よる loss masking の例、plain dict のツール定義との混在も 1 件ずつ示す。

実行:
    uv run python examples/finetune/03_tools_from_registry.py

導入: pip install 'oai-agentspec[finetune]'
"""

from __future__ import annotations

from oai_agentspec import ToolRegistry, ToolSpec
from oai_agentspec.runtime.finetune import to_sft_dataset


def get_order_status(order_id: str) -> str:
    """注文番号から配送状況を返す（example 用のダミー実装）。

    Args:
        order_id: 注文番号。

    Returns:
        配送状況の説明文。
    """
    return f"{order_id} は本日発送予定です"


def main() -> None:
    """Registry 登録ツールを `to_sft_dataset(tools=...)` へそのまま渡す。"""
    registry = ToolRegistry()
    registry.register(ToolSpec(func=get_order_status))

    # registry.<name> は SDK FunctionTool を遅延構築する。name / params_json_schema を持つため
    # to_sft_dataset の tools= はこれを FunctionTool 相当として検出し FT 形式へ写像する。
    tools = [registry.get_order_status]

    # plain dict のツール定義との混在も許容される（1 行例）。
    mixed_tools = [*tools, {"type": "function", "function": {"name": "noop", "parameters": {}}}]

    cases = [
        # tool_calls 付き assistant / role "tool" を含む複数ターンケース。
        {
            "input": [
                {"role": "user", "content": "A-1234 の配送状況を教えて"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_order_status",
                                "arguments": '{"order_id": "A-1234"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "本日発送予定"},
            ],
            # weight=0 を付けると当該 assistant メッセージは loss masking で学習対象から外れる。
            "expected_output": [
                {"role": "assistant", "content": "A-1234 は本日発送予定です", "weight": 1},
            ],
        },
        {"input": "営業時間を教えてください", "expected_output": "平日 9 時から 18 時です"},
    ]

    result = to_sft_dataset(cases, tools=mixed_tools)

    print(f"records={len(result.records)} skipped={result.skipped}")
    for record in result.records:
        print("tools   =", record["tools"])
        print("messages=", record["messages"])
        print("---")


if __name__ == "__main__":
    main()
