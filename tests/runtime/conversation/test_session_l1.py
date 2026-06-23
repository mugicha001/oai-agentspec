"""L1: CompactionConfig / SessionPolicy.compaction_kwargs を検証する（unit・外部依存なし）。

`enabled` フラグと client/model の受け渡しの分離（暗黙有効化の排除）と、有効化時の
client 欠落の早期検知（`__post_init__`）・frozen 性を確認する。あわせて
`SessionPolicy.compaction_kwargs` が `make_session` の plain kwargs へ正しく展開する
（None 経路・`options` -> `compaction_options` の名前変換）ことを確認する。実 API は叩かない。
"""

from __future__ import annotations

import dataclasses

import pytest

from oai_agentspec.runtime.conversation import CompactionConfig, SessionPolicy

pytestmark = pytest.mark.unit


class _FakeClient:
    """実 API を叩かないダミー AsyncOpenAI 互換クライアント。"""


def test_enabled_without_client_raises() -> None:
    """enabled=True かつ client 欠落は構築時 ValueError を送出する。"""
    with pytest.raises(ValueError, match="client"):
        CompactionConfig(enabled=True)


def test_disabled_with_client_constructs() -> None:
    """enabled=False は client を渡しても構築成功する（圧縮しない契約）。"""
    cfg = CompactionConfig(enabled=False, client=_FakeClient())
    assert cfg.enabled is False
    assert cfg.client is not None


def test_default_is_disabled() -> None:
    """既定は enabled=False / client None / options 空（暗黙有効化しない）。"""
    cfg = CompactionConfig()
    assert cfg.enabled is False
    assert cfg.client is None
    assert cfg.model is None
    assert cfg.options == {}


def test_enabled_with_client_constructs() -> None:
    """enabled=True かつ client 指定は構築成功する。"""
    cfg = CompactionConfig(enabled=True, client=_FakeClient(), model="gpt-4.1")
    assert cfg.enabled is True
    assert cfg.model == "gpt-4.1"


def test_is_frozen() -> None:
    """frozen dataclass のためフィールド再代入は FrozenInstanceError になる。"""
    cfg = CompactionConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.enabled = True  # type: ignore[misc]


def test_compaction_kwargs_none_returns_disabled_only() -> None:
    """compaction=None は enable_compaction=False のみを返す（他キーを含めない）。"""
    kwargs = SessionPolicy().compaction_kwargs()
    assert kwargs == {"enable_compaction": False}


def test_compaction_kwargs_expands_config_with_options_rename() -> None:
    """CompactionConfig を make_session の plain kwargs へ展開する。

    特に `options` フィールドが `compaction_options` キーへ名前変換されることを確認する。
    """
    client = _FakeClient()
    policy = SessionPolicy(
        compaction=CompactionConfig(
            enabled=True, client=client, model="gpt-4.1", options={"k": "v"}
        )
    )
    kwargs = policy.compaction_kwargs()
    assert kwargs == {
        "enable_compaction": True,
        "client": client,
        "model": "gpt-4.1",
        "compaction_options": {"k": "v"},
    }
