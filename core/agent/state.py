# core/agent/state.py
"""
Agent 狀態容器。

只保存「現在系統處於什麼狀態」，不做任何決策；決策邏輯屬於 pipeline.py。
新增狀態欄位時只需動這個檔案，不影響 Pipeline 的判斷邏輯。
"""

from __future__ import annotations
import threading
from typing import Optional, Callable, TYPE_CHECKING

from core.agent.event_queue import EventQueue

if TYPE_CHECKING:
    from core.voice_input import VoiceInput


class AgentState:
    """
    純狀態容器，不含業務邏輯，不主動呼叫任何外部模組。

    Attributes:
        mode: 輸入模式，"t"（文字）或 "v"（語音）
        voice_input: VoiceInput 實例，語音模式下才惰性建立
        running: 主迴圈是否應繼續執行
        event_queue: 所有事件來源共用的單一佇列。多 producer 寫入，
            Agent 在唯一取用點讀出——該取用點即為仲裁層的位置。
            同來源的 observation 會互相覆蓋（環境狀態是當前值不是事件流），
            utterance 不覆蓋。見 core/agent/event_queue.py
        text_input_thread: 扛住阻塞 input() 的背景執行緒。每取得一次輸入即結束，
            下一輪重建；queue 則長駐不重建。
        proactive_enabled: 是否對 observation 事件作出反應。預設關閉，
            需 /proactive on 手動開啟。執行期狀態、不持久化。
        last_speech_at: 奧狐上次輸出的時間（time.monotonic），供仲裁層計算冷卻。
            初始為 0.0 表示尚未發言過，任何事件皆可通過冷卻檢查。
            ⚠️ 用 monotonic 而非 time.time()：後者會受系統時間調整影響，
            校時或跨時區可能讓冷卻瞬間失效或永久卡住。
        observation_test_index: /proactive test 的循環取用位置（腳手架，
            清理條件見 config.EVENT_CONFIG）
    """

    def __init__(self, mode: str = "t") -> None:
        self.mode: str = mode
        self.voice_input: Optional["VoiceInput"] = None
        self.running: bool = True

        # 長駐，不隨輪次重建——多個 producer 需要一個穩定的寫入目標
        self.event_queue = EventQueue()
        self.text_input_thread: Optional[threading.Thread] = None

        self.proactive_enabled: bool = False
        self.observation_test_index: int = 0
        self.last_speech_at: float = 0.0

    def switch_to_text(self) -> None:
        self.mode = "t"

    def switch_to_voice(self, voice_input_factory: Callable[[], "VoiceInput"]) -> None:
        """
        切換至語音模式。voice_input_factory 惰性建立 VoiceInput，只有第一次切換時
        才真正實例化，避免文字模式使用者被迫載入 Whisper / VAD 模型。

        實例一旦建立就跨模式常駐，不重複建立（相關約束見 Agent._get_next_event()）。
        """
        if self.voice_input is None:
            self.voice_input = voice_input_factory()
        self.mode = "v"

    def stop(self) -> None:
        self.running = False