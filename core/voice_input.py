# core/voice_input.py
# 語音輸入模組 — 麥克風收音 + VAD 偵測 + Whisper 辨識

import threading

# ⚠️ config 必須排在重量套件之前：faster_whisper 會連帶 import huggingface_hub，
#    而後者在 import 當下就讀 HF_HUB_OFFLINE（config 開頭設定，見 main.py 開頭）。
from config import VOICE_CONFIG
from core import console
from core.logger import setup_logger

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel

try:
    import msvcrt   # Windows 終端的按鍵緩衝
except ImportError:  # 非 Windows
    msvcrt = None

logger = setup_logger("VoiceInput")

SAMPLE_RATE = VOICE_CONFIG["sample_rate"]
CHANNELS = VOICE_CONFIG["channels"]
VAD_THRESHOLD = VOICE_CONFIG["vad_threshold"]
SILENCE_DURATION = VOICE_CONFIG["silence_duration"]
RMS_GATE = VOICE_CONFIG["rms_gate"]
CHUNK_SIZE = VOICE_CONFIG["chunk_size"]
DEBUG_PRINT_RMS = VOICE_CONFIG["debug_print_rms"]

_ESC = "\x1b"


def _drain_keys() -> bool:
    """讀光終端按鍵緩衝，回傳其中有沒有 ESC。

    ⚠️ 讀的是本終端的輸入緩衝，只有終端視窗在前景時按鍵才會進來。不要改成全域鍵盤 hook
       （pynput 之類）：那會讓在遊戲或其他程式裡按 ESC 也退出語音模式。
    """
    if msvcrt is None:
        return False
    hit = False
    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):   # 方向鍵等特殊鍵是兩碼，第二碼一起讀掉
            msvcrt.getwch()
        elif ch == _ESC:
            hit = True
    return hit


class VoiceInput:
    def __init__(self, model_size: str = "small", language: str = "zh"):
        console.emit("[語音模型載入中…]")
        logger.info(f"載入 Whisper 模型（{model_size}）")
        self.whisper = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )
        self.language = language

        logger.info("載入 Silero VAD 模型")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.vad_utils

        self._stop = threading.Event()   # close() 用：讓進行中的 listen() 在下一個音框結束

        if msvcrt is None:
            logger.warning("非 Windows 終端：ESC 退出語音模式不可用（說 quit 或 Ctrl+C 結束）")
        console.emit("[語音準備完成，可以開始說話（在本視窗按 ESC 回到文字模式）]\n")

    def listen(self) -> str | None:
        """
        收音、偵測語音、靜音超過 SILENCE_DURATION 秒後停止並回傳辨識文字（沒聽到東西＝空字串）。
        使用者在本視窗按 ESC ⇒ 回傳 None（由呼叫端決定後續，見 core/producers/user_input.py）。
        """
        console.emit("🎙 聆聽中...（按 ESC 可退出語音模式）")

        self.vad_model.reset_states()  # ⚠️ 必須重置，否則上一輪殘留狀態會污染本輪判斷
        _drain_keys()                  # 她說話期間按的鍵不算（與上一輪的退出旗標同理）

        chunk_size = CHUNK_SIZE
        audio_buffer = []
        silence_chunks = 0
        silence_limit = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
        is_speaking = False

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
            while True:
                if self._stop.is_set():
                    return ""
                if _drain_keys():
                    console.emit("\n⏹ 偵測到退出鍵，結束語音模式聆聽。")
                    return None

                chunk, _ = stream.read(chunk_size)
                chunk_mono = chunk[:, 0] if chunk.ndim > 1 else chunk.flatten()
                tensor = torch.from_numpy(chunk_mono)

                rms = float(torch.sqrt(torch.mean(tensor ** 2)))
                if DEBUG_PRINT_RMS:
                    logger.debug(f"rms={rms:.4f}")   # 校準用，看 log 視窗

                # 先用音量門檻過濾底噪，未過門檻不送進 VAD（省一次模型推論）
                if rms < RMS_GATE:
                    if is_speaking:
                        silence_chunks += 1
                        if silence_chunks >= silence_limit:
                            break
                    continue

                confidence = self.vad_model(tensor, SAMPLE_RATE).item()

                if confidence > VAD_THRESHOLD:
                    is_speaking = True
                    silence_chunks = 0
                    audio_buffer.append(chunk_mono)
                elif is_speaking:
                    audio_buffer.append(chunk_mono)
                    silence_chunks += 1
                    if silence_chunks >= silence_limit:
                        break  # 靜音夠久，判定說話結束

        if not audio_buffer:
            return ""

        audio_np = np.concatenate(audio_buffer)
        segments, _ = self.whisper.transcribe(
            audio_np,
            language=self.language,
            beam_size=5,
        )
        text = "".join(seg.text for seg in segments).strip()
        console.emit(f"📝 辨識結果：{text}")
        return text

    def close(self):
        """程式結束時呼叫：進行中的 listen() 在下一個音框（約 32 ms）內結束。"""
        self._stop.set()
