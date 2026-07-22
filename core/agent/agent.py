# core/agent/agent.py
"""
Agent(中央控制核心 / Single Control Point)

職責：
    - 建立與持有所有核心模組(LLM / Persona / Memory / VTube / VoiceInput)
    - 負責模組初始化失敗時的降級(fallback)，例如 VTube 連不上就跳過，不中斷主流程
    - 驅動主迴圈：取得輸入 → 交給 Pipeline 判斷 → 依結果印出 / 驅動 VTube / 結束
    - 不自己做「該怎麼處理這句輸入」的決策細節，那是 Pipeline 的職責

拆分依據（呼應技術文件 Agent Layer 風險章節「God Object」的解法:
「拆分子模組，如 state、pipeline」):
    Agent 只做「持有 + 驅動」，State 只做「儲存狀態」，Pipeline 只做「流程判斷」，
    三者職責互不重疊，任一模組要換掉都不影響其他兩個
    (例如未來要加 Tool Calling，只需要在 Pipeline 裡加，State/Agent 不必動)。

【例外決定記錄】
本次拆分時，文件原訂的「抽出訊號」(線性 if/else 開始難以維護)尚未真正出現
——原 main.py 只有 4 個分支，並不難維護。這是邪狼知情下的主動決定：
提前把架構準備好，以便後續加「主動對話觸發」「Memory Router 第二層」時
可以直接掛進 Pipeline，不必屆時因為卡在舊架構裡而被迫重寫一次。
此例外已於 2026-07-18 對齊確認，應同步記錄進《AOFOS-決策脈絡歸檔.md》
「六、Agent Layer 現況說明」一節，避免日後回顧時誤判為自相矛盾。
"""

from __future__ import annotations

from core.llm_client import LLMClient
from core.persona import Persona
from core.memory_manager import MemoryManager
from core.logger import setup_logger
from core.agent.state import AgentState
from core.agent.pipeline import Pipeline

logger = setup_logger("Agent")


class Agent:
    def __init__(self) -> None:
        # ── 核心模組初始化（必要模組，失敗即應直接報錯，不做 fallback）──────
        self.llm = LLMClient()
        self.persona = Persona()
        self.memory = MemoryManager(self.llm)

        # ── VTube Studio（可選模組，失敗需 fallback，不中斷主流程）──────────
        self.vtube = None
        try:
            from core.vtube_client import VTubeClient
            self.vtube = VTubeClient()
            self.vtube.connect()
        except Exception as e:
            logger.warning(f"VTube 未連線，跳過：{e}")

        # ── 狀態與流程控制 ────────────────────────────────────────────
        self.state = AgentState(mode="t")
        self.pipeline = Pipeline(self.llm, self.persona, self.memory)

    # ── 語音模組惰性建立（維持原 main.py 行為，文字模式使用者不必載入語音模型）──

    @staticmethod
    def _voice_input_factory():
        from core.voice_input import VoiceInput
        return VoiceInput(model_size="small", language="zh")

    # ── 輸出端副作用（VTube 嘴型），由 Agent 決定要不要驅動 ─────────────────

    def _drive_vtube_on_reply(self) -> None:
        if self.vtube:
            try:
                self.vtube.set_mouth(0.8)
            except Exception as e:
                logger.warning(f"VTube 嘴型控制失敗：{e}")

    # ── 輸入取得 ─────────────────────────────────────────────────────

    def _get_input(self) -> str:
        if self.state.mode == "t":
            return input("你：").strip()
        # 語音模式：voice_input 一定已在切換時被惰性建立
        text = self.state.voice_input.listen()  # type: ignore[union-attr]
        return text.strip() if text else ""

    # ── 主迴圈 ───────────────────────────────────────────────────────

    def run(self) -> None:
        print("=== 奧狐 AOFOS 啟動 ===")
        print("[記憶已載入]" if self.memory.compressed_memory else "[首次啟動，無歷史記憶]")
        print("模式：t = 文字輸入，v = 語音輸入（預設文字）")
        print("指令：/mode t、/mode v、quit\n")

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
                    break
                continue

            result = self.pipeline.handle_turn(user_input)
            if result.reply:
                print(f"\n奧狐：{result.reply}\n")
                logger.debug(f"奧狐回應：{result.reply[:50]}...")
                self._drive_vtube_on_reply()
