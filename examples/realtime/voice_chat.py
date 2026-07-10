"""マイク / スピーカーで音声会話する例（宣言ルート + sounddevice 音声 I/O）。

OpenAI Agents SDK 公式の realtime CLI デモ（examples/realtime/cli/demo.py）の音声 I/O
（24kHz int16 mono・40ms チャンク・ジッタバッファ・割り込み時フェードアウト・barge-in）を
踏襲し、エージェント構築部分だけを本ライブラリの宣言ルート（RealtimeAgentSpec /
RealtimeAgentRegistry）に差し替えた構成。責務分担:

  - 宣言側（本ライブラリ）: triage <-> support の相互 handoff（循環）を宣言・構築する。
  - 実行側（本ファイル = 利用者）: 接続先・turn_detection 等の実行時 Config と、
    マイク入力（session.send_audio）/ 音声イベント再生などの音声 I/O をすべて担う。

実行（sounddevice はライブラリ本体の依存に含めないため --with で一時導入する）:

    uv run --with sounddevice python examples/realtime/voice_chat.py

Ctrl+C で終了。接続先は Azure OpenAI 優先・OPENAI_API_KEY フォールバック（.env.example 参照）。
"""

from __future__ import annotations

import asyncio
import contextlib
import queue
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print(
        "sounddevice が見つかりません。次のコマンドで実行してください:\n"
        "    uv run --with sounddevice python examples/realtime/voice_chat.py"
    )
    raise SystemExit(1) from None

from agents.realtime import (
    RealtimePlaybackTracker,
    RealtimeRunner,
    RealtimeSession,
    RealtimeSessionEvent,
)

from oai_agentspec.realtime import (
    RealtimeAgentRegistry,
    RealtimeAgentSpec,
    RealtimeHandoffGraph,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _connection import build_model_config, require_credentials, scrub  # noqa: E402

from _azure import load_env  # noqa: E402

# 音声設定（SDK 公式デモと同値。realtime の既定に合わせた 40ms チャンク）
CHUNK_LENGTH_S = 0.04
SAMPLE_RATE = 24000
FORMAT = np.int16
CHANNELS = 1
ENERGY_THRESHOLD = 0.015  # アシスタント発話中に割り込みとみなすマイク RMS 閾値
PREBUFFER_CHUNKS = 3  # 再生開始前のジッタバッファ（40ms x 3 = 約 120ms）
FADE_OUT_MS = 12  # 割り込み時のクリックノイズ回避用フェードアウト
PLAYBACK_ECHO_MARGIN = 0.002  # 再生エコーより大きい入力だけを発話とみなすマージン


def build_registry() -> RealtimeAgentRegistry:
    """triage <-> support の相互 handoff（循環）を宣言・登録して registry を返す。

    Returns:
        validate 済みの RealtimeAgentRegistry（entry は triage）。
    """
    # spec はエージェントの中身のみ宣言する（トポロジはグラフ側の責務）。
    specs = [
        RealtimeAgentSpec(
            name="triage",
            instructions=(
                "あなたは音声受付担当。最初に一言挨拶し、技術的な問い合わせは"
                "サポート担当へ引き継ぐ。"
            ),
            handoff_description="最初の受付・振り分け担当。",
        ),
        RealtimeAgentSpec(
            name="support",
            instructions=(
                "製品の技術的な問い合わせに音声で簡潔に答えるサポート担当。"
                "解決したら、または技術以外の話題になったら受付担当へ戻す。"
            ),
            handoff_description="技術サポート担当。",
        ),
    ]

    # 相互 handoff（循環）をグラフ DSL で宣言し spec 群へ一括反映する
    # （spec.handoffs 直接宣言と同一の結線になる）。
    graph = RealtimeHandoffGraph(entry="triage")
    graph.edge("triage", "support", tool_description="技術的な問い合わせを引き継ぐ")
    graph.edge("support", "triage", tool_description="解決したら受付へ戻す")
    graph.apply(specs)

    registry = RealtimeAgentRegistry()
    for spec in specs:
        registry.register(spec)
    registry.validate()
    return registry


class VoiceChat:
    """sounddevice でマイク入力と再生を行う音声会話ループ（SDK 公式デモ踏襲）。"""

    def __init__(self) -> None:
        """音声 I/O・再生キュー・barge-in 判定の状態を初期化する。"""
        self.session: RealtimeSession | None = None
        self.audio_stream: sd.InputStream | None = None
        self.audio_player: sd.OutputStream | None = None
        self.recording = False
        # capture タスクへの強参照（イベントループは弱参照しか持たないため GC 対策）と
        # セッション終了時の確実な停止に使う
        self.capture_task: asyncio.Task[None] | None = None

        # 実再生位置をモデルへ知らせる（割り込み時に「どこまで聞こえたか」を正しく扱う）
        self.playback_tracker = RealtimePlaybackTracker()

        # 再生キュー: (samples, item_id, content_index)。ドロップによる音飛びを避けるため無制限
        self.output_queue: queue.Queue[Any] = queue.Queue(maxsize=0)
        self.interrupt_event = threading.Event()
        self.current_audio_chunk: tuple[np.ndarray[Any, np.dtype[Any]], str, int] | None = None
        self.chunk_position = 0

        # ジッタバッファとフェードアウトの状態
        self.prebuffering = True
        self.fading = False
        self.fade_total_samples = 0
        self.fade_done_samples = 0
        self.fade_samples = int(SAMPLE_RATE * (FADE_OUT_MS / 1000.0))
        self.playback_rms = 0.0  # 再生エコー除去用の平滑化エネルギー

    def _output_callback(self, outdata, frames: int, time, status) -> None:
        """再生コールバック。キューの音声を流し、割り込み時はフェードアウトして破棄する。"""
        if status:
            print(f"[audio] output status: {status}")

        # どの経路も無音を土台に必要分だけ上書きする
        outdata.fill(0)

        if self.interrupt_event.is_set():
            if self.current_audio_chunk is None:
                self._flush_queue()
                self.prebuffering = True
                self.interrupt_event.clear()
                return
            if not self.fading:
                self.fading = True
                self.fade_done_samples = 0
                remaining = len(self.current_audio_chunk[0]) - self.chunk_position
                self.fade_total_samples = min(self.fade_samples, max(0, remaining))
            samples, item_id, content_index = self.current_audio_chunk
            filled = 0
            while filled < len(outdata) and self.fade_done_samples < self.fade_total_samples:
                n = min(len(outdata) - filled, self.fade_total_samples - self.fade_done_samples)
                src = samples[self.chunk_position : self.chunk_position + n].astype(np.float32)
                idx = np.arange(
                    self.fade_done_samples, self.fade_done_samples + n, dtype=np.float32
                )
                gain = 1.0 - (idx / float(self.fade_total_samples))
                ramped = np.clip(src * gain, -32768.0, 32767.0).astype(np.int16)
                outdata[filled : filled + n, 0] = ramped
                self._update_playback_rms(ramped)
                self._report_playback(item_id, content_index, ramped)
                filled += n
                self.chunk_position += n
                self.fade_done_samples += n
            if self.fade_done_samples >= self.fade_total_samples:
                self.current_audio_chunk = None
                self.chunk_position = 0
                self._flush_queue()
                self.fading = False
                self.prebuffering = True
                self.interrupt_event.clear()
            return

        filled = 0
        while filled < len(outdata):
            if self.current_audio_chunk is None:
                try:
                    if self.prebuffering and self.output_queue.qsize() < PREBUFFER_CHUNKS:
                        break
                    self.prebuffering = False
                    self.current_audio_chunk = self.output_queue.get_nowait()
                    self.chunk_position = 0
                except queue.Empty:
                    break
            samples, item_id, content_index = self.current_audio_chunk
            n = min(len(outdata) - filled, len(samples) - self.chunk_position)
            if n > 0:
                chunk = samples[self.chunk_position : self.chunk_position + n]
                outdata[filled : filled + n, 0] = chunk
                self._update_playback_rms(chunk)
                self._report_playback(item_id, content_index, chunk)
                filled += n
                self.chunk_position += n
            # チャンクを使い切ったら（長さ 0 のチャンクを含む）解放して次へ進む。
            # n > 0 の内側でのみ解放するとサイズ 0 のチャンクが永遠に残り
            # コールバックが無限ループするため、解放判定は書き込みの外で行う
            if self.chunk_position >= len(samples):
                self.current_audio_chunk = None
                self.chunk_position = 0

        # 無音（書き込みなし）の間は再生エコー推定を減衰させる。減衰しないと直前の
        # 大きな値が次の応答開始直後の barge-in ゲートを不当に高くする
        if filled == 0:
            self.playback_rms *= 0.8

    def _flush_queue(self) -> None:
        """再生キューを空にし、エコー推定をリセットする（割り込み時の破棄処理）。"""
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
        # 破棄した音声はもう鳴らないため、エコー推定もリセットする
        self.playback_rms = 0.0

    def _report_playback(self, item_id: str, content_index: int, chunk: np.ndarray) -> None:
        """実際に再生したサンプルを playback tracker へ通知する（割り込み位置の精度向上用）。"""
        try:
            self.playback_tracker.on_play_bytes(
                item_id=item_id, item_content_index=content_index, bytes=chunk.tobytes()
            )
        except Exception:
            pass

    async def run(self) -> None:
        """宣言ルートで構築したエージェントを RealtimeRunner で実行し、音声会話ループを回す。"""
        registry = build_registry()
        entry = registry.get("triage")

        chunk_size = int(SAMPLE_RATE * CHUNK_LENGTH_S)
        self.audio_player = sd.OutputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype=FORMAT,
            callback=self._output_callback,
            blocksize=chunk_size,
        )
        self.audio_player.start()

        try:
            runner = RealtimeRunner(entry)
            # 接続先（Azure/OpenAI）と turn_detection は実行時 Config（宣言側は関知しない）。
            model_config: dict[str, Any] = build_model_config() or {}
            model_config["playback_tracker"] = self.playback_tracker
            model_config["initial_model_settings"] = {
                "voice": "alloy",
                "turn_detection": {
                    "type": "semantic_vad",
                    "interrupt_response": True,
                    "create_response": True,
                },
            }
            print(f"接続先: {'Azure OpenAI' if 'url' in model_config else 'OpenAI'}")
            print("接続中...")
            async with await runner.run(model_config=model_config) as session:  # type: ignore[arg-type]
                self.session = session
                print("接続完了。マイクに向かって話してください（Ctrl+C で終了）")
                await self.start_audio_recording()
                try:
                    async for event in session:
                        await self._on_event(event)
                finally:
                    # セッションが閉じる（async with を抜ける）前に capture を止める。
                    # 逆順だとクローズ済みセッションへ send_audio する窓ができる
                    await self.stop_audio_recording()
        finally:
            # capture 系（recording / capture_task / audio_stream）は start/stop_audio_recording
            # が単独所有する。ここでは run() が所有する audio_player のみを片付ける
            if self.audio_player and self.audio_player.active:
                self.audio_player.stop()
            if self.audio_player:
                self.audio_player.close()
        print("セッション終了")

    async def start_audio_recording(self) -> None:
        """マイク入力ストリームを開始し、キャプチャタスクを起動する。"""
        self.audio_stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype=FORMAT,
        )
        self.audio_stream.start()
        self.recording = True
        # 強参照を保持する（イベントループはタスクを弱参照でしか持たず GC されうるため）
        self.capture_task = asyncio.create_task(self.capture_audio())

    async def stop_audio_recording(self) -> None:
        """キャプチャタスクを停止して完了を待つ（セッションクローズ前に呼ぶ）。"""
        self.recording = False
        if self.capture_task is not None:
            self.capture_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.capture_task

    async def capture_audio(self) -> None:
        """マイク音声を読み取り session.send_audio へ流す（barge-in 判定付き）。"""
        if not self.audio_stream or not self.session:
            return
        read_size = int(SAMPLE_RATE * CHUNK_LENGTH_S)
        try:
            while self.recording:
                if self.audio_stream.read_available < read_size:
                    await asyncio.sleep(0.01)
                    continue
                data, _ = self.audio_stream.read(read_size)
                audio_bytes = data.tobytes()

                # アシスタント発話中は、再生エコーより十分大きい入力（= 人の割り込み発話）
                # だけを送る。送る際はローカル再生も即フラッシュして体感を良くする
                assistant_playing = (
                    self.current_audio_chunk is not None or not self.output_queue.empty()
                )
                if assistant_playing:
                    mic_rms = self._compute_rms(data.reshape(-1))
                    gate = max(ENERGY_THRESHOLD, self.playback_rms * 0.6 + PLAYBACK_ECHO_MARGIN)
                    if mic_rms >= gate:
                        self.interrupt_event.set()
                        await self.session.send_audio(audio_bytes)
                else:
                    await self.session.send_audio(audio_bytes)
                await asyncio.sleep(0)
        except Exception as e:
            print(f"[audio] capture error: {scrub(str(e))[:200]}")
        finally:
            if self.audio_stream and self.audio_stream.active:
                self.audio_stream.stop()
            if self.audio_stream:
                self.audio_stream.close()

    async def _on_event(self, event: RealtimeSessionEvent) -> None:
        """セッションイベントの処理（音声はキューへ、その他は表示）。"""
        if event.type == "agent_start":
            print(f"[agent] start: {event.agent.name}")
        elif event.type == "agent_end":
            print(f"[agent] end: {event.agent.name}")
        elif event.type == "handoff":
            print(f"[handoff] {event.from_agent.name} -> {event.to_agent.name}")
        elif event.type == "audio":
            np_audio = np.frombuffer(event.audio.data, dtype=np.int16)
            # 長さ 0 のチャンクはキューに入れない（出力コールバック側にもガードがあるが、
            # そもそも再生する意味がなくキューを汚すだけのため入口で捨てる）
            if np_audio.size > 0:
                self.output_queue.put_nowait((np_audio, event.item_id, event.content_index))
        elif event.type == "audio_interrupted":
            print("[audio] interrupted")
            self.prebuffering = True
            self.interrupt_event.set()
        elif event.type == "error":
            print(f"[error] {scrub(str(event.error))[:200]}")

    @staticmethod
    def _compute_rms(samples: np.ndarray[Any, np.dtype[Any]]) -> float:
        """int16 サンプル列の RMS エネルギーを [-1, 1] 正規化で計算する。"""
        if samples.size == 0:
            return 0.0
        x = samples.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x * x)))

    def _update_playback_rms(self, samples: np.ndarray[Any, np.dtype[Any]]) -> None:
        """再生エコー推定（平滑化 RMS）を実再生サンプルで更新する。"""
        sample_rms = self._compute_rms(samples)
        self.playback_rms = 0.9 * self.playback_rms + 0.1 * sample_rms


def main() -> None:
    """env 読み込みと認証チェックを行い、音声会話ループを起動する。"""
    load_env()
    require_credentials()
    chat = VoiceChat()
    try:
        asyncio.run(chat.run())
    except KeyboardInterrupt:
        print("\n終了します")


if __name__ == "__main__":
    main()
