# core/voice_input.py
# 語音輸入模組 — 麥克風收音 + VAD 偵測 + Whisper 辨識

import sounddevice as sd
import numpy as np
import torch
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHANNELS = 1
VAD_THRESHOLD = 0.5       # 降低門檻，更容易觸發語音檢測
SILENCE_DURATION = 1.5    # 縮短靜音判定時間，更快反應
CHUNK_SIZE = 512          # Silero VAD 要求固定 512 samples


class VoiceInput:
    def __init__(self, model_size: str = "small", language: str = "zh"):
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

        print("[VoiceInput] 準備完成，可以開始說話。\n")

    def listen(self) -> str:
        """
        開始收音，偵測到說話後錄音，
        靜音超過 SILENCE_DURATION 秒後停止，
        回傳辨識出的文字。
        """
        print("🎙 聆聽中...")

        chunk_size = CHUNK_SIZE
        audio_buffer = []
        silence_chunks = 0
        silence_limit = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
        is_speaking = False

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
            while True:
                chunk, _ = stream.read(chunk_size)
                chunk_mono = chunk[:, 0] if chunk.ndim > 1 else chunk.flatten()
                tensor = torch.from_numpy(chunk_mono)

                # 音量門檻過濾（過濾底噪）
                rms = float(torch.sqrt(torch.mean(tensor ** 2)))
                if rms < 0.02:
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