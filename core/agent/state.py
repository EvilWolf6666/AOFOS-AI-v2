# core/agent/state.py
"""
Agent 狀態容器(AgentState)

職責：只保存「現在系統處於什麼狀態」，不做任何決策邏輯。
決策邏輯屬於 pipeline.py 的職責，state.py 只負責被讀寫。

拆分理由（對應技術文件 Agent Layer 風險章節）：
God Object 的其中一個成因是「狀態」與「決策」混在同一個物件裡，
導致要改邏輯時不小心動到狀態，或反之。這裡先把狀態單獨隔開，
之後若要加「主動對話是否正在等待中」「is_ai_thinking」這類新狀態，
只需要動這個檔案，不影響 pipeline 的判斷邏輯。
"""

from __future__ import annotations
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.voice_input import VoiceInput


class AgentState:
    """
    純狀態容器，不含業務邏輯，不呼叫任何外部模組（除了被動接受 factory 建立結果）。

    Attributes:
        mode: 目前輸入模式，"t"（文字）或 "v"（語音）
        voice_input: VoiceInput 實例，語音模式下才會被建立（惰性初始化，維持原行為）
        running: 主迴圈是否應該繼續執行
    """

    def __init__(self, mode: str = "t") -> None:
        self.mode: str = mode
        self.voice_input: Optional["VoiceInput"] = None
        self.running: bool = True

    def switch_to_text(self) -> None:
        self.mode = "t"

    def switch_to_voice(self, voice_input_factory: Callable[[], "VoiceInput"]) -> None:
        """
        切換至語音模式。voice_input_factory 是惰性建立 VoiceInput 的 callable，
        只有第一次切換時才真正實例化（維持原 main.py 的惰性初始化行為，
        避免文字模式使用者被迫載入語音模型）。
        """
        if self.voice_input is None:
            self.voice_input = voice_input_factory()
        self.mode = "v"

    def stop(self) -> None:
        self.running = False
