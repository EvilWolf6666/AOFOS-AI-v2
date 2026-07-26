"""
AgentState：純狀態容器，只被動讀寫，不含任何決策邏輯（決策邏輯屬於 pipeline.py）。
"""

from __future__ import annotations
import queue
import threading
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.voice_input import VoiceInput


class AgentState:
    """
    Attributes:
        mode: 目前輸入模式，"t"（文字）或 "v"（語音）
        voice_input: VoiceInput 實例，語音模式下才建立（惰性初始化）
        running: 主迴圈是否應繼續執行
        text_input_thread: 文字模式扛住阻塞 input() 的背景執行緒
        text_input_queue: 上述執行緒與主執行緒之間傳遞輸入結果的佇列
        proactive_enabled: 主動對話觸發開關，/proactive on|off 控制
    """

    def __init__(self, mode: str = "t") -> None:
        self.mode: str = mode
        self.voice_input: Optional["VoiceInput"] = None
        self.running: bool = True

        self.text_input_thread: Optional[threading.Thread] = None
        self.text_input_queue: Optional["queue.Queue[str]"] = None

        self.proactive_enabled: bool = True

    def switch_to_text(self) -> None:
        self.mode = "t"

    def switch_to_voice(self, voice_input_factory: Callable[[], "VoiceInput"]) -> None:
        """第一次切換語音模式時才透過 factory 建立 VoiceInput，之後沿用同一實例。"""
        if self.voice_input is None:
            self.voice_input = voice_input_factory()
        self.mode = "v"

    def stop(self) -> None:
        self.running = False