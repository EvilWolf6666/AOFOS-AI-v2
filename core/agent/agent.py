"""
Agent（中央控制核心 / Single Control Point）

職責：
    - 持有並初始化核心模組（LLM / Persona / Memory / VTube / VoiceInput）
    - 模組初始化失敗時的降級處理（例如 VTube 連不上就跳過，不中斷主流程）
    - 驅動主迴圈：取得輸入 → 交給 Pipeline 判斷 → 印出結果 / 驅動 VTube / 結束

不做「輸入該怎麼處理」的決策，那是 Pipeline 的職責；狀態則由 AgentState 持有。
"""

from __future__ import annotations

import queue
import threading

from core.llm_client import LLMClient
from core.persona import Persona
from core.memory_manager import MemoryManager
from core.logger import setup_logger
from core.agent.state import AgentState
from core.agent.pipeline import Pipeline
from core.voice_input import VoiceModeExit

logger = setup_logger("Agent")

# 主動對話：靜默多久（秒）視為使用者沒有回應，觸發主動發話。暫定值，待實測校準。
SILENCE_THRESHOLD_SECONDS = 30


class Agent:
    def __init__(self) -> None:
        # 核心模組：初始化失敗直接拋錯，不做 fallback
        self.llm = LLMClient()
        self.persona = Persona()
        self.memory = MemoryManager(self.llm)

        # VTube Studio：可選模組，連不上就跳過，不中斷主流程
        self.vtube = None
        try:
            from core.vtube_client import VTubeClient
            self.vtube = VTubeClient()
            self.vtube.connect()
        except Exception as e:
            logger.warning(f"VTube 未連線，跳過：{e}")

        self.state = AgentState(mode="t")
        self.pipeline = Pipeline(self.llm, self.persona, self.memory)

    @staticmethod
    def _voice_input_factory():
        """語音模式切換時才建立 VoiceInput，避免文字模式使用者被迫載入語音模型。"""
        from core.voice_input import VoiceInput
        return VoiceInput(model_size="small", language="zh")

    def _drive_vtube_on_reply(self) -> None:
        """每次奧狐回覆後觸發嘴型動作，失敗不影響對話流程。"""
        if self.vtube:
            try:
                self.vtube.set_mouth(0.8)
            except Exception as e:
                logger.warning(f"VTube 嘴型控制失敗：{e}")

    def _close_voice_input(self) -> None:
        """關閉 VoiceInput 的背景鍵盤監聽 thread，只在程式整體結束時呼叫。"""
        if self.state.voice_input:
            try:
                self.state.voice_input.close()
            except Exception as e:
                logger.warning(f"VoiceInput 關閉時發生例外（不影響流程）：{e}")

    def _close_text_input_thread(self) -> None:
        """
        清空文字輸入背景執行緒的參照。

        不呼叫 thread.join()：該執行緒卡在 input()，只有使用者打字才會結束，
        quit 情境下不會發生，join() 會卡死程式。交由 daemon=True 隨程式結束回收。
        """
        self.state.text_input_thread = None
        self.state.text_input_queue = None

    def _get_input(self) -> str:
        if self.state.mode == "t":
            return self._get_text_input_with_timeout()

        try:
            text = self.state.voice_input.listen()  # type: ignore[union-attr]
        except VoiceModeExit:
            print("[已退出語音模式，切回文字模式]\n")
            # voice_input 實例跨模式切換保留（惰性建立，避免重複載入 Whisper/VAD 模型），
            # 這裡只切模式、不呼叫 close()，否則下次 /mode v 會沿用已停止監聽的舊實例。
            self.state.switch_to_text()
            return ""

        return text.strip() if text else ""

    def _get_text_input_with_timeout(self) -> str:
        """
        文字輸入 + 靜默計時，兩者用同一個 queue 達成。

        input() 無法中斷，故背景執行緒扛住它，主執行緒用 queue.get(timeout=...)：
        逾時 → 判定靜默，觸發主動對話；未逾時 → 取得真正輸入。
        執行緒跨輪次沿用（尚未拿到輸入前不重建），避免多條執行緒同時搶 stdin。
        """
        if (
            self.state.text_input_thread is None
            or not self.state.text_input_thread.is_alive()
        ):
            self.state.text_input_queue = queue.Queue()
            self.state.text_input_thread = threading.Thread(
                target=self._blocking_input_worker,
                args=(self.state.text_input_queue,),
                daemon=True,
            )
            self.state.text_input_thread.start()

        try:
            user_input = self.state.text_input_queue.get(
                timeout=SILENCE_THRESHOLD_SECONDS
            )
            self.state.text_input_thread = None
            self.state.text_input_queue = None
            return user_input.strip()
        except queue.Empty:
            # 靜默超過閾值。先檢查是否被手動暫停（開發測試用）。
            if self.state.proactive_enabled:
                self._trigger_proactive_message()
            return ""

    @staticmethod
    def _blocking_input_worker(q: "queue.Queue[str]") -> None:
        text = input("你：")
        q.put(text)

    def _trigger_proactive_message(self) -> None:
        result = self.pipeline.handle_proactive_trigger()
        if result.reply:
            print(f"\n奧狐：{result.reply}\n")
            logger.debug(f"奧狐主動發起：{result.reply[:50]}...")
            self._drive_vtube_on_reply()

    def run(self) -> None:
        print("=== 奧狐 AOFOS 啟動 ===")
        print("[記憶已載入]" if self.memory.compressed_memory else "[首次啟動，無歷史記憶]")
        print("模式：t = 文字輸入，v = 語音輸入（預設文字）")
        print("指令：/mode t、/mode v、/proactive on、/proactive off、quit\n")

        while self.state.running:
            user_input = self._get_input()
            if not user_input:
                continue

            if self.pipeline.is_command(user_input):
                result = self.pipeline.handle_command(
                    user_input, self.state, self._voice_input_factory
                )
                if result.system_message:
                    print(f"{result.system_message}\n")
                if result.should_exit:
                    if self.vtube:
                        self.vtube.disconnect()
                    self._close_voice_input()
                    self._close_text_input_thread()
                    break
                continue

            result = self.pipeline.handle_turn(user_input)
            if result.reply:
                print(f"\n奧狐：{result.reply}\n")
                logger.debug(f"奧狐回應：{result.reply[:50]}...")
                self._drive_vtube_on_reply()