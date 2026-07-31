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
import queue
import threading
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
        text_input_thread: 文字模式下扛住阻塞 input() 的背景執行緒
            （2026-07-25 新增，供主動對話靜默觸發使用）
        text_input_queue: 上述執行緒與主執行緒之間傳遞輸入結果的佇列
        proactive_enabled: 主動對話觸發開關。
            【2026-07-25 預設值變更】原預設 True（開發測試階段），
            實測發現連續多次無回應時內容可能重複（已記錄於《決策脈絡歸檔》
            十一之2子問題八，解法延後至螢幕/遊戲事件觸發源建置時一併處理），
            在此之前改為預設 False（關閉），需要時手動 /proactive on 開啟。
            此為執行期狀態，每次重新啟動程式都會回到此預設值，不會跨執行
            階段記住上次的開關狀態。
    """

    def __init__(self, mode: str = "t") -> None:
        self.mode: str = mode
        self.voice_input: Optional["VoiceInput"] = None
        self.running: bool = True

        # ── 主動對話觸發：文字模式背景輸入執行緒狀態 ──────────────────
        self.text_input_thread: Optional[threading.Thread] = None
        self.text_input_queue: Optional["queue.Queue[str]"] = None

        # ── 主動對話觸發：開關（2026-07-25 改為預設關閉，見上方屬性說明）──
        self.proactive_enabled: bool = False

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