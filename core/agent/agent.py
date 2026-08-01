# core/agent/agent.py
"""
Agent（中央控制核心 / Single Control Point）

職責：
    - 建立與持有所有核心模組（LLM / Persona / Memory / VTube / VoiceInput）
    - 可選模組初始化失敗時降級（如 VTube 連不上就跳過），不中斷主流程
    - 驅動主迴圈：取得輸入 → 交給 Pipeline → 輸出 / 驅動 VTube / 結束

「該怎麼處理這句輸入」的決策細節屬於 Pipeline，Agent 不介入。
拆分理由與例外記錄見《決策脈絡歸檔》六之6、十四章。
"""

from __future__ import annotations

import queue
import threading

from config import SILENCE_THRESHOLD_SECONDS
from core.llm_client import LLMClient
from core.persona import Persona
from core.memory_manager import MemoryManager
from core.logger import setup_logger
from core.agent.state import AgentState
from core.agent.pipeline import Pipeline
from core.voice_input import VoiceModeExit

logger = setup_logger("Agent")


class Agent:
    def __init__(self) -> None:
        # 必要模組：初始化失敗應直接報錯，不做 fallback
        self.llm = LLMClient()
        self.persona = Persona()
        self.memory = MemoryManager(self.llm)

        # 可選模組：失敗需 fallback，不中斷主流程
        self.vtube = None
        try:
            from core.vtube_client import VTubeClient
            self.vtube = VTubeClient()
            self.vtube.connect()
        except Exception as e:
            logger.warning(f"VTube 未連線，跳過：{e}")

        self.state = AgentState(mode="t")
        self.pipeline = Pipeline(self.llm, self.persona, self.memory)

    # ── 模組建立與清理 ───────────────────────────────────────────────

    @staticmethod
    def _voice_input_factory():
        """惰性建立，文字模式使用者不必載入語音模型。"""
        from core.voice_input import VoiceInput
        return VoiceInput(model_size="small", language="zh")

    def _close_voice_input(self) -> None:
        """關閉 VoiceInput 的背景鍵盤監聽 thread。只在程式整體結束時呼叫。"""
        if self.state.voice_input:
            try:
                self.state.voice_input.close()
            except Exception as e:
                logger.warning(f"VoiceInput 關閉時發生例外（不影響流程）：{e}")

    def _close_text_input_thread(self) -> None:
        """
        清理文字模式背景輸入執行緒的參照。

        ⚠️ 絕對不可對該執行緒呼叫 join()。input() 是 OS 層級阻塞呼叫，卡在裡面的
        執行緒無法被主動中斷，只有使用者打字才會結束——而 quit 情境下這不會發生，
        join() 會讓整個程式卡死。此處能做的只有清掉參照，並依賴 daemon=True
        讓它不阻擋主程式退出；殘留執行緒會隨行程結束由 OS 回收。
        """
        if self.state.text_input_thread is not None:
            logger.debug("背景文字輸入執行緒尚未取得輸入，交由 daemon thread 機制回收")
        self.state.text_input_thread = None
        self.state.text_input_queue = None

    # ── 輸出端副作用 ─────────────────────────────────────────────────

    def _drive_vtube_on_reply(self) -> None:
        if self.vtube:
            try:
                self.vtube.set_mouth(0.8)
            except Exception as e:
                logger.warning(f"VTube 嘴型控制失敗：{e}")

    # ── 輸入取得 ─────────────────────────────────────────────────────

    def _get_input(self) -> str:
        if self.state.mode == "t":
            return self._get_text_input_with_timeout()

        # 語音模式：voice_input 必定已在切換時建立
        try:
            text = self.state.voice_input.listen()  # type: ignore[union-attr]
        except VoiceModeExit:
            print("[已退出語音模式，切回文字模式]\n")
            # ⚠️ 此處刻意不呼叫 _close_voice_input()。voice_input 實例跨模式常駐，
            # switch_to_voice() 判斷 is not None 就不重建；在這裡關掉鍵盤監聽 thread，
            # 下次 /mode v 會沿用同一個已停止監聽的舊實例，ESC 從此失效。
            self.state.switch_to_text()
            return ""  # 由 run() 的 `if not user_input: continue` 跳過這輪

        return text.strip() if text else ""

    def _get_text_input_with_timeout(self) -> str:
        """
        文字模式輸入，同時支援主動對話的靜默觸發。

        input() 無法中斷，故用背景執行緒扛住它，主執行緒改用 queue.get(timeout=...)
        等待：逾時代表靜默超過閾值，觸發主動對話；沒逾時代表拿到真正輸入。

        尚未取得輸入的執行緒跨輪次沿用、不重建——多條執行緒同時等 stdin 會使輸入
        歸屬不可預期。
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
            # 該執行緒的 input() 已返回、生命週期結束，下一輪需重建
            self.state.text_input_thread = None
            self.state.text_input_queue = None
            return user_input.strip()
        except queue.Empty:
            if not self.state.proactive_enabled:
                return ""

            self._trigger_proactive_message()
            # 主動訊息會把 input("你：") 印出的提示往上推走，補印一次讓使用者知道
            # 仍可打字。不建立新執行緒，不影響背景執行緒的等待狀態。
            print("你：", end="", flush=True)
            return ""

    @staticmethod
    def _blocking_input_worker(q: "queue.Queue[str]") -> None:
        text = input("你：")
        q.put(text)

    # ── 主動對話觸發 ─────────────────────────────────────────────────

    def _trigger_proactive_message(self) -> None:
        result = self.pipeline.handle_proactive_trigger()
        if result.reply:
            print(f"\n奧狐：{result.reply}\n")
            logger.debug(f"奧狐主動發起：{result.reply[:50]}...")
            self._drive_vtube_on_reply()
            self.pipeline.finalize_turn()  # 與一般輪次相同，壓縮必須在輸出之後

    # ── 主迴圈 ───────────────────────────────────────────────────────

    def run(self) -> None:
        print("=== 奧狐 AOFOS 啟動 ===")
        print("[記憶已載入]" if self.memory.compressed_memory else "[首次啟動，無歷史記憶]")
        print("模式：t = 文字輸入，v = 語音輸入（預設文字）")
        print("指令：/mode t、/mode v、/proactive on、/proactive off、"
              "/test on、/test off、/status、quit")

        # 測試模式會改變記憶與 RAG 行為，必須在畫面上持續可見，
        # 否則容易在忘記自己開著的情況下把測試資料當成正常對話。
        if self.memory.test_mode:
            print("\n" + "=" * 46)
            print("  ⚠️  測試模式（TEST_MODE）目前為開啟狀態")
            print("     RAG 停用中、記憶寫入獨立測試檔")
            print("     此為 config.py 的設定，非 /test 指令")
            print("=" * 46)
        print()

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

            # 記憶壓縮一律在輸出之後。即使本輪無回應也要呼叫：Working Memory
            # 可能在先前輪次就已超長，略過會讓壓縮無限期延後。
            self.pipeline.finalize_turn()