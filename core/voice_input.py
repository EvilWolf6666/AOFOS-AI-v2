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
VAD_THRESHOLD = 0.5       # 降低門檻，更容易觸發語音檢測
SILENCE_DURATION = 1.5    # 縮短靜音判定時間，更快反應
RMS_GATE = 0.02           # 原本寫死在 listen() 內部的 magic number，集中管理方便校準
CHUNK_SIZE = 512          # Silero VAD 要求固定 512 samples

DEBUG_PRINT_RMS = False   # 【開關】要看 rms 數值校準麥克風時，改成 True 即可


class VoiceModeExit(Exception):
    """
    使用者在語音模式中按下退出鍵（預設 ESC）時拋出。
    外層（agent/pipeline）應該 catch 這個例外，用來切回文字模式或結束程式，
    而不是強制關閉終端機。
    """
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

        # 【新增】背景鍵盤監聽：整個 VoiceInput 生命週期只啟動一次，
        # 不放在 listen() 裡面重複開，避免每輪對話都疊加監聽器造成洩漏。
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
        開始收音，偵測到說話後錄音，
        靜音超過 SILENCE_DURATION 秒後停止，
        回傳辨識出的文字。

        若使用者按下退出鍵（預設 ESC），拋出 VoiceModeExit，
        呼叫端須自行 catch 並決定後續行為（切回文字模式 / 結束程式）。
        """
        print("🎙 聆聽中...（按 ESC 可退出語音模式）")

        # 每次開始聆聽前重置 VAD 內部狀態，避免上一輪殘留狀態污染這一輪判斷
        self.vad_model.reset_states()
        # 每次開始聆聽前清空退出旗標，避免上一輪殘留的按鍵事件影響這一輪
        self._exit_event.clear()

        chunk_size = CHUNK_SIZE
        audio_buffer = []
        silence_chunks = 0
        silence_limit = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
        is_speaking = False

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
            while True:
                # 每個 chunk（約32ms）檢查一次退出旗標，反應夠即時
                if self._exit_event.is_set():
                    print("\n⏹ 偵測到退出鍵，結束語音模式聆聽。")
                    raise VoiceModeExit("使用者按下退出鍵，要求離開語音模式")

                chunk, _ = stream.read(chunk_size)
                chunk_mono = chunk[:, 0] if chunk.ndim > 1 else chunk.flatten()
                tensor = torch.from_numpy(chunk_mono)

                # 音量門檻過濾（過濾底噪）
                rms = float(torch.sqrt(torch.mean(tensor ** 2)))
                if DEBUG_PRINT_RMS:
                    print(f"[debug] rms={rms:.4f}")

                if rms < RMS_GATE:
                    if is_speaking:
                        silence_chunks += 1
                        if silence_chunks >= silence_limit:
                            break
                    continue

                # VAD 偵測
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

        # 合併音訊並辨識
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
        """
        程式結束或切換模式時呼叫，停止背景鍵盤監聽 thread。
        建議在 agent.py 的關閉流程（或 /mode t 切換離開語音模式時）呼叫一次。
        """
        self._key_listener.stop()