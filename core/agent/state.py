# core/agent/state.py
"""
Agent 狀態容器。

只保存「現在系統處於什麼狀態」，不做任何決策；決策邏輯屬於 pipeline.py。
新增狀態欄位時只需動這個檔案，不影響 Pipeline 的判斷邏輯。
"""

from __future__ import annotations
import queue
import threading
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.voice_input import VoiceInput


class AgentState:
    """
    純狀態容器，不含業務邏輯，不主動呼叫任何外部模組。

    Attributes:
        mode: 輸入模式，"t"（文字）或 "v"（語音）
        voice_input: VoiceInput 實例，語音模式下才惰性建立
        running: 主迴圈是否應繼續執行
        text_input_thread: 文字模式下扛住阻塞 input() 的背景執行緒
        text_input_queue: 上述執行緒與主執行緒之間傳遞輸入結果的佇列
        proactive_enabled: 主動對話觸發開關，預設關閉，需 /proactive on 手動開啟
            （關閉原因見《決策脈絡歸檔》十一之2）。此為執行期狀態、不持久化，
            重新啟動一律回到預設值。
    """

    def __init__(self, mode: str = "t") -> None:
        self.mode: str = mode
        self.voice_input: Optional["VoiceInput"] = None
        self.running: bool = True

        # 主動對話觸發：文字模式背景輸入執行緒狀態
        self.text_input_thread: Optional[threading.Thread] = None
        self.text_input_queue: Optional["queue.Queue[str]"] = None

        self.proactive_enabled: bool = False

    def switch_to_text(self) -> None:
        self.mode = "t"

    def switch_to_voice(self, voice_input_factory: Callable[[], "VoiceInput"]) -> None:
        """
        切換至語音模式。voice_input_factory 惰性建立 VoiceInput，只有第一次切換時
        才真正實例化，避免文字模式使用者被迫載入 Whisper / VAD 模型。

        實例一旦建立就跨模式常駐，不重複建立（相關約束見 Agent._get_input()）。
        """
        if self.voice_input is None:
            self.voice_input = voice_input_factory()
        self.mode = "v"

    def stop(self) -> None:
        self.running = False