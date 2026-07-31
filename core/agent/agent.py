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
（依 2026-07-29 訂立的四類判定，此決定實為「既有需求的缺口」，非違例。）

【2026-07-22 追加】語音模式退出處理：
core/voice_input.py 的 VoiceInput.listen() 在使用者按下退出鍵（預設 ESC）時
會拋出 VoiceModeExit，職責上屬於「輸入取得」這件事本身的異常狀況，
所以在 _get_input() 這一層 catch，不下放給 Pipeline
（Pipeline 只處理「拿到輸入之後」的流程，不該認識輸入取得階段的例外）。

【2026-07-25 追加】主動對話觸發（文字模式，靜默計時版）：
input() 是無法中斷的 OS 層級阻塞呼叫，故用背景執行緒扛住它，主執行緒
改用 queue.get(timeout=...) 同時達成「等輸入」與「等靜默計時」不互相卡住。
這是 kimjammer/Neuro（獨立執行緒 + 共用狀態物件）架構原則的最小實例，
未來若加入螢幕事件、聊天室訊息等第二個以上的觸發源，會需要仲裁機制
避免多執行緒同時判定觸發（設計原則已記錄於《決策脈絡歸檔》十一之1，
參考 Lumi_Nox 的 speech_output_arbiter 三態仲裁），現階段僅單一觸發源，
不存在競態，故不現在建置仲裁層。

已知技術限制：input() 卡住的背景執行緒無法被主動中斷（不像 VoiceInput
監聽迴圈逐 chunk 檢查旗標那樣可中斷）。此執行緒用 daemon=True 確保
不會阻止主程式退出，但絕對不能對它呼叫 join()——它只有使用者打字才會
結束，quit 情境下這不會發生，join() 會讓程式卡死。詳見 _close_text_input_thread()。

新增 /proactive on / /proactive off 指令，供開發測試階段手動暫停主動
觸發（見 state.proactive_enabled），避免測試時被反覆觸發打擾。

【2026-07-25 追加修正】靜默觸發後補印輸入提示符號：
輸入執行緒跨輪次沿用、逾時觸發時刻意不重建的設計（見
_get_text_input_with_timeout()），代價是 input("你：") 這個提示字串
只在執行緒第一次建立時印出一次——逾時觸發主動對話後，畫面上不會再
自動出現新的「你：」提示，即使背景執行緒其實一直都在正常等待輸入。
這不是功能性 bug（使用者仍可直接打字送出，執行緒會正常收到），純粹是
視覺提示消失，容易讓人誤以為系統卡住。故在觸發後補印一次提示符號，
不建立新執行緒、不影響任何既有邏輯。

【2026-07-30 追加】記憶壓縮移出顯示路徑：
原本 Pipeline.handle_turn() 內的 memory.add_turn() 會在 Working Memory
超長時直接觸發一次完整的 LLM 壓縮呼叫，而它是在 print(reply) 之前執行的
——等於每累積約 5 輪，使用者要多等一整次 LLM 呼叫才看得到回應。
現改為：回應輸出完畢後才呼叫 pipeline.finalize_turn() 執行壓縮。
尖峰因此移到「下一次輸入之前」，使用者感受不到，但尖峰本身仍然存在
（仍為同步執行）。若實測後仍不可接受，再評估改背景執行緒；那會引入
競態寫入 compressed_memory 的風險（該欄位為整份覆寫非追加），需另行處理。
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

# ── 主動對話觸發：靜默閾值（秒）───────────────────────────────
# 暫定預設值，未經實測校準。與 VAD_THRESHOLD 等參數同樣性質，
# 屬於「先用合理起始值、依實際使用感受調整」的參數，非最終定案。
SILENCE_THRESHOLD_SECONDS = 30


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

    # ── 語音模組清理 ─────────────────────────────────────────────────

    def _close_voice_input(self) -> None:
        """
        關閉 VoiceInput 的背景鍵盤監聽 thread。
        呼叫時機：使用者按退出鍵離開語音模式 / 程式整體結束時，
        避免背景 thread 沒被停掉就留著（即使是 daemon thread，養成隨手清理的習慣較好）。
        """
        if self.state.voice_input:
            try:
                self.state.voice_input.close()
            except Exception as e:
                logger.warning(f"VoiceInput 關閉時發生例外（不影響流程）：{e}")

    # ── 文字輸入背景執行緒清理 ───────────────────────────────────────

    def _close_text_input_thread(self) -> None:
        """
        清理文字模式的背景輸入執行緒參照。

        重要限制：input() 是 OS 層級阻塞呼叫，卡在裡面的執行緒無法被
        主動中斷（不像 VoiceInput 監聽迴圈逐 chunk 檢查旗標、天生可中斷）。
        因此這裡能做的只有：
        1. 清掉 state 上的 thread/queue 參照，避免任何後續程式碼誤用舊參照
        2. 依賴 daemon=True 讓它不阻擋主程式退出

        刻意不做、且不能做的事：不呼叫 thread.join()。join() 會卡住等待
        該執行緒自然結束，但它只有使用者打字才會結束，quit 情境下這不會
        發生，join() 會讓 quit 指令卡死整個程式。這條殘留的背景執行緒
        會隨主程式行程結束被 OS 一併回收，不會有資源外洩，只是無法
        「乾淨結束」，只能「放它去」——這是 input() 本身的限制，非疏漏。
        """
        if self.state.text_input_thread is not None:
            logger.debug(
                "背景文字輸入執行緒尚未取得輸入，"
                "隨主程式結束交由 daemon thread 機制回收（無法主動中斷 input()）"
            )
        self.state.text_input_thread = None
        self.state.text_input_queue = None

    # ── 輸入取得 ─────────────────────────────────────────────────────

    def _get_input(self) -> str:
        if self.state.mode == "t":
            return self._get_text_input_with_timeout()

        # 語音模式：voice_input 一定已在切換時被惰性建立
        try:
            text = self.state.voice_input.listen()  # type: ignore[union-attr]
        except VoiceModeExit:
            print("[已退出語音模式，切回文字模式]\n")
            # 注意：這裡刻意不呼叫 _close_voice_input()。
            # voice_input 實例跨模式切換是持續存在的（見 state.py switch_to_voice()
            # 的惰性初始化設計，避免重複載入 Whisper/VAD 模型），
            # 若在這裡把鍵盤監聽 thread 關掉，下次 /mode v 會沿用同一個已關閉監聽的
            # 舊實例（switch_to_voice 判斷 voice_input is not None 就不重建），
            # ESC 會從此失效。背景監聽 thread 留著跨模式常駐是刻意行為，
            # 真正該關閉的時機只有整個程式結束時（見 run() 的 quit 分支）。
            self.state.switch_to_text()
            return ""  # 交給 run() 迴圈的 `if not user_input: continue` 自然跳過這輪

        return text.strip() if text else ""

    def _get_text_input_with_timeout(self) -> str:
        """
        文字模式輸入，同時支援主動對話的靜默觸發。

        原理：input() 無法中斷，故用背景執行緒扛住它，主執行緒改用
        queue.get(timeout=...) 等待——逾時代表靜默超過閾值，觸發主動對話；
        沒逾時代表拿到真正輸入。

        跨輪次沿用同一條輸入執行緒（尚未拿到輸入時不重建），避免重複開
        執行緒同時等 stdin 導致輸入歸屬不可預期（此坑於 asyncio 方案討論
        時已確認過，thread 版本必須同樣避開）。
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
            # 這條執行緒的 input() 已返回、生命週期結束，
            # 下一輪需要重新建立新的，故清空參照。
            self.state.text_input_thread = None
            self.state.text_input_queue = None
            return user_input.strip()
        except queue.Empty:
            # 靜默超過閾值。先檢查是否被手動暫停（開發測試用）。
            if not self.state.proactive_enabled:
                # 【2026-07-31 修正】主動觸發關閉時，直接回去繼續等，不補印提示。
                # 補印的原意是「主動對話訊息把畫面上的 你： 擠掉了，補一個回來」，
                # 前提是真的有訊息被印出來。主動觸發關閉時沒有任何東西擠掉它，
                # 補印只會讓畫面每 30 秒多長出一個 你：——使用者離開座位幾分鐘
                # 回來就看到一排 你：你：你：，看起來像程式出錯。
                return ""

            self._trigger_proactive_message()
            # 主動訊息已把提示符號往上推走，補印一次讓使用者知道仍可打字。
            # 不建立新執行緒、不影響背景執行緒原本的等待狀態。
            print("你：", end="", flush=True)
            return ""  # 交給 run() 的 `if not user_input: continue`

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
            # 與一般輪次相同：壓縮必須在輸出之後才執行。
            self.pipeline.finalize_turn()

    # ── 主迴圈 ───────────────────────────────────────────────────────

    def run(self) -> None:
        print("=== 奧狐 AOFOS 啟動 ===")
        print("[記憶已載入]" if self.memory.compressed_memory else "[首次啟動，無歷史記憶]")
        print("模式：t = 文字輸入，v = 語音輸入（預設文字）")
        print("指令：/mode t、/mode v、/proactive on、/proactive off、"
              "/test on、/test off、/status、quit")

        # 測試模式會改變記憶與 RAG 行為，必須在畫面上持續可見，
        # 否則很容易在忘記自己開著的情況下把測試資料當成正常對話。
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

            # 輪次收尾（記憶壓縮）一律在輸出之後執行。
            # 即使本輪沒有回應也要呼叫：Working Memory 可能在先前輪次就已超長，
            # 略過會讓壓縮無限期延後。
            self.pipeline.finalize_turn()