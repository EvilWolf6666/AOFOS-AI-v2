# core/voice_input.py
# 語音輸入模組 — 麥克風收音 + VAD 偵測 + Whisper 辨識

import threading

import sounddevice as sd
import numpy as np
import torch
from faster_whisper import WhisperModel
from pynput import keyboard

from config import VOICE_CONFIG
from core.logger import setup_logger

# ⚠️ 其他模組若要 catch VoiceModeExit，請直接從 core.ctrl_stream_signal import，
#    **不要**寫成 `from core.voice_input import VoiceModeExit`。後者雖然寫得出來、
#    也會動，但會在模組層級把整個語音棧（faster_whisper ~6s、torch ~4s）一併載入，
#    而且不會有任何徵兆。VoiceModeExit 之所以被搬出去就是為了避免這件事
#    （見《決策脈絡歸檔》三十二章）。此處 import 僅供本檔 raise 使用。
from core.ctrl_stream_signal import VoiceModeExit

logger = setup_logger("VoiceInput")

SAMPLE_RATE = VOICE_CONFIG["sample_rate"]
CHANNELS = VOICE_CONFIG["channels"]
VAD_THRESHOLD = VOICE_CONFIG["vad_threshold"]
SILENCE_DURATION = VOICE_CONFIG["silence_duration"]
RMS_GATE = VOICE_CONFIG["rms_gate"]
CHUNK_SIZE = VOICE_CONFIG["chunk_size"]
DEBUG_PRINT_RMS = VOICE_CONFIG["debug_print_rms"]


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

        self.vad_model.reset_states()  # ⚠️ 必須重置，否則上一輪殘留狀態會污染本輪判斷
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
        print(f"📝 辨識結果：{text}")
        return text

    def close(self):
        """停止背景鍵盤監聽 thread，於程式結束時呼叫（切換模式時不呼叫，見 Agent._get_input）。"""
        self._key_listener.stop()