# core/voice_input.py
# 語音輸入模組 — 麥克風收音 + VAD 偵測 + Whisper 辨識

import threading

import sounddevice as sd
import numpy as np
import torch
from faster_whisper import WhisperModel
from pynput import keyboard

from core.logger import setup_logger

logger = setup_logger("VoiceInput")

SAMPLE_RATE = 16000
CHANNELS = 1
VAD_THRESHOLD = 0.5       # 語音判定信心值門檻
SILENCE_DURATION = 1.5    # 靜音多久（秒）判定為說完
RMS_GATE = 0.02           # 音量門檻，過濾底噪；麥克風增益不同時需一併校準
CHUNK_SIZE = 512          # Silero VAD 要求固定 512 samples

DEBUG_PRINT_RMS = False   # 校準麥克風時開啟，印出即時 rms 數值


class VoiceModeExit(Exception):
    """使用者按下退出鍵（預設 ESC）時拋出，呼叫端需 catch 並切回文字模式或結束程式。"""
    pass


class VoiceInput:
    def __init__(self, model_size: str = "small", language: str = "zh", exit_key=keyboard.Key.esc):
        print("[VoiceInput] 載入 Whisper 模型中...")
        self.whisper = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )
        self.language = language

        print("[VoiceInput] 載入 Silero VAD 模型中...")
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

        # 背景鍵盤監聽：整個 VoiceInput 生命週期只啟動一次，不放在 listen() 裡重複建立
        self.exit_key = exit_key
        self._exit_event = threading.Event()
        self._key_listener = keyboard.Listener(on_press=self._on_key_press)
        self._key_listener.daemon = True
        self._key_listener.start()

        print(f"[VoiceInput] 準備完成，可以開始說話。（按 {self.exit_key} 可隨時退出語音模式）\n")

    def _on_key_press(self, key):
        if key == self.exit_key:
            self._exit_event.set()

    def listen(self) -> str:
        """
        收音、偵測語音、靜音超過 SILENCE_DURATION 秒後停止並回傳辨識文字。
        按下退出鍵時拋出 VoiceModeExit，由呼叫端決定後續行為。
        """
        print("🎙 聆聽中...（按 ESC 可退出語音模式）")

        self.vad_model.reset_states()  # 重置 VAD 內部狀態，避免上一輪殘留污染這一輪判斷
        self._exit_event.clear()       # 清空上一輪殘留的退出旗標

        chunk_size = CHUNK_SIZE
        audio_buffer = []
        silence_chunks = 0
        silence_limit = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
        is_speaking = False

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
            while True:
                if self._exit_event.is_set():
                    print("\n⏹ 偵測到退出鍵，結束語音模式聆聽。")
                    raise VoiceModeExit("使用者按下退出鍵，要求離開語音模式")

                chunk, _ = stream.read(chunk_size)
                chunk_mono = chunk[:, 0] if chunk.ndim > 1 else chunk.flatten()
                tensor = torch.from_numpy(chunk_mono)

                rms = float(torch.sqrt(torch.mean(tensor ** 2)))
                if DEBUG_PRINT_RMS:
                    print(f"[debug] rms={rms:.4f}")

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
        print(f"📝 辨識結果：{text}")
        return text

    def close(self):
        """停止背景鍵盤監聽 thread，於程式結束或切換模式時呼叫。"""
        self._key_listener.stop()