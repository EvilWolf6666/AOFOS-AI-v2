# core/agent/agent.py
"""
Agent（中央控制核心 / Single Control Point）

職責：
    - 建立與持有所有核心模組（LLM / Persona / Memory / VTube / VoiceInput）
    - 可選模組初始化失敗時降級（如 VTube 連不上就跳過），不中斷主流程
    - 驅動主迴圈：從 event_queue 取事件 → 依 kind 分流 → 輸出 / 驅動 VTube / 結束

「該怎麼處理這個事件」的決策細節屬於 Pipeline，Agent 不介入。
拆分理由與例外記錄見《歸檔》F0006。

⚠️ event_queue.get() 是所有事件的唯一取用點，仲裁層即建於此（_arbitrate）。
   新增仲裁規則一律加在該方法內，不得在各 producer 內分頭決定要不要發言
   （分散決策點的失效案例見《歸檔》L0005）。
"""

from __future__ import annotations

import threading
import time
from typing import Optional

# PERSONA_CONFIG 已不在此使用：回應前後綴改由 Pipeline.REPLY_PREFIX/TRAILING 單一來源提供
from config import ARBITRATION_CONFIG
from core.llm_client import LLMClient
from core.persona import Persona
from core.memory import MemoryManager
from core.logger import setup_logger
from core.agent.state import AgentState
from core.agent.pipeline import Pipeline
from core.agent.events import Event, UTTERANCE, OBSERVATION
from core.producers.window_watcher import WindowWatcher
from core import console
# 從 ctrl_signals 取，不是從 voice_input 取：後者會在模組層級拉進整個語音棧，
# 使下方 _voice_input_factory() 的惰性建立失效（函式內 import 只是查 sys.modules）。
from core.ctrl_signals import VoiceModeExit

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

        # 事件來源。producer 只推事件、不決定是否發言（見 _arbitrate）
        self.window_watcher = WindowWatcher(self.state.event_queue)

    # ── 模組建立與清理 ───────────────────────────────────────────────

    @staticmethod
    def _voice_input_factory():
        """惰性建立，文字模式使用者不必載入語音模型。

        ⚠️ 前提是本檔模組層級沒有任何路徑會先載入 core.voice_input，
        否則此處 import 只是查 sys.modules，一毛錢都省不到。
        """
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
        清理文字輸入執行緒的參照。

        ⚠️ 絕對不可對該執行緒呼叫 join()。input() 是 OS 層級阻塞呼叫，卡在裡面的
        執行緒無法被主動中斷，只有使用者打字才會結束——而 quit 情境下這不會發生，
        join() 會讓整個程式卡死。此處能做的只有清掉參照，並依賴 daemon=True
        讓它不阻擋主程式退出；殘留執行緒會隨行程結束由 OS 回收。

        event_queue 不清空：它長駐且無外部資源，殘留事件隨行程結束消失。
        """
        if self.state.text_input_thread is not None:
            logger.debug("背景文字輸入執行緒尚未取得輸入，交由 daemon thread 機制回收")
        self.state.text_input_thread = None

    # ── 輸出端副作用 ─────────────────────────────────────────────────

    def _emit_reply(self, reply: str) -> None:
        """
        非串流路徑的回應輸出。

        🔴 **前後綴一律取自 Pipeline，不在此重寫。** 兩條路徑的畫面格式必須逐字元
        一致，否則開關 stream 時行距會跳動——而那看起來像程式出了問題，會誘使人
        去查一個不存在的 bug（同 console.end_stream() 的註解）。
        ⚠️ 這裡原本是手寫的 f"\n{名字}：{回應}\n" 加一行「改動需兩處同步」的註解，
        也就是**同一個格式有兩份真相、只靠紀律維持**。依《現況》20之0——
        **副本數才是變因，同步紀律不是**——2026-09-03 改為單一來源。
        """
        console.emit(
            f"{Pipeline.REPLY_PREFIX}{reply}{Pipeline.REPLY_TRAILING}", end=""
        )

    def _drive_vtube_on_reply(self) -> None:
        if self.vtube:
            try:
                self.vtube.set_mouth(0.8)
            except Exception as e:
                logger.warning(f"VTube 嘴型控制失敗：{e}")

    # ── 事件取得 ─────────────────────────────────────────────────────

    def _get_next_event(self) -> Optional[Event]:
        """
        取得下一個待處理事件。文字模式阻塞在 event_queue 上，語音模式阻塞在
        麥克風上。

        ⚠️ 語音模式不經 event_queue，因此語音錄音期間無法接收 observation 事件。
        統一走 queue 需要把 VoiceInput 改為獨立 producer 執行緒，屬語音線工作，
        本次不處理。
        """
        if self.state.mode == "t":
            return self._get_text_mode_event()

        # 語音模式：voice_input 必定已在切換時建立
        try:
            text = self.state.voice_input.listen()  # type: ignore[union-attr]
        except VoiceModeExit:
            console.emit("[已退出語音模式，切回文字模式]\n")
            # ⚠️ 此處刻意不呼叫 _close_voice_input()。voice_input 實例跨模式常駐，
            # switch_to_voice() 判斷 is not None 就不重建；在這裡關掉鍵盤監聽 thread，
            # 下次 /mode v 會沿用同一個已停止監聽的舊實例，ESC 從此失效。
            self.state.switch_to_text()
            return None

        text = text.strip() if text else ""
        if not text:
            return None
        return Event(source="user_voice", kind=UTTERANCE, payload=text)

    def _get_text_mode_event(self) -> Event:
        """
        文字模式的事件取得。input() 無法中斷，故用背景執行緒扛住它，取得的輸入
        包成 Event 推進共用佇列；主執行緒只負責阻塞讀取佇列。

        尚未取得輸入的執行緒跨輪次沿用、不重建——多條執行緒同時等 stdin 會使輸入
        歸屬不可預期。
        """
        if (
            self.state.text_input_thread is None
            or not self.state.text_input_thread.is_alive()
        ):
            self.state.text_input_thread = threading.Thread(
                target=self._blocking_input_worker,
                args=(self.state.event_queue,),
                daemon=True,
            )
            self.state.text_input_thread.start()

        event = self.state.event_queue.get()

        # 取到的若是其他來源（如測試事件），輸入執行緒仍卡在 input() 中，
        # 不可清掉參照，否則下一輪會再開一條同時等 stdin
        if event.source == "user_text":
            self.state.text_input_thread = None

        return event

    @staticmethod
    def _blocking_input_worker(q) -> None:
        # 經 console 取得輸入：提示字元登記後，背景輸出被打斷時能重畫
        text = console.input_with_prompt("你：")
        q.put(Event(source="user_text", kind=UTTERANCE, payload=text.strip()))

    # ── 仲裁 ─────────────────────────────────────────────────────────

    def _arbitrate(self, event: Event) -> bool:
        """
        決定這個事件要不要處理。**所有仲裁規則一律加在這裡。**

        這是系統中唯一的仲裁點——它成立的前提是 event_queue.get() 是唯一取用點。
        規則若下放到各 producer 自行判斷，多來源時必然出現同時輸出
        （Lumi_Nox 失效案例，見《歸檔》L0005）。

        現行兩條規則，皆只作用於 observation：
          1. proactive 開關關閉 → 不處理
          2. 距上次發言未滿冷卻時間 → 不處理

        utterance（有人在說話）一律通過，不受冷卻限制——冷卻是為了避免奧狐
        自言自語過於頻繁，不是用來擋人講話的。

        ⚠️ 尚未實作但位置已預留（① 類，TTS 接入主迴圈後才存在）：
        奧狐正在說話時使用者插話的 QUEUE / INTERRUPT 選擇、中斷與靜音通道。
        目前文字模式為同步阻塞，輸出期間不可能有新輸入，故兩者皆不存在。
        """
        if event.kind != OBSERVATION:
            return True

        if not self.state.proactive_enabled:
            logger.debug(f"事件略過（proactive 關閉）：{event.payload[:40]}")
            return False

        cooldown = ARBITRATION_CONFIG["observation_cooldown_seconds"]
        elapsed = time.monotonic() - self.state.last_speech_at
        if self.state.last_speech_at > 0.0 and elapsed < cooldown:
            # 一併清掉佇列裡同來源的殘留。只丟棄當下這一則的話，冷卻期間
            # 累積的其他則會在下一輪被取出，冷卻形同虛設
            dropped = self.state.event_queue.drop_source(event.source)
            extra = f"，另清除同來源殘留 {dropped} 則" if dropped else ""
            logger.debug(
                f"事件略過（冷卻中，剩 {cooldown - elapsed:.0f}s{extra}）："
                f"{event.payload[:40]}"
            )
            return False

        return True

    # ── 事件處理 ─────────────────────────────────────────────────────

    def _handle_observation(self, event: Event) -> None:
        """環境事件：沒有人在說話，情境走標記過的 context 區塊注入。"""
        result = self.pipeline.handle_observation(event)
        if result.reply:
            # printed=True 代表回應已在生成過程中逐段印出（串流），再印一次會出現兩遍。
            if not result.printed:
                self._emit_reply(result.reply)
            logger.debug(f"奧狐主動發起：{result.reply[:50]}...")
            self.state.last_speech_at = time.monotonic()
            self._drive_vtube_on_reply()
        # 🔺 2026-09-30：移出 `if result.reply`，與 _handle_utterance 一致（無回應也收尾）——
        #    準 v4 的背景判斷在輪次開始暫停、在這裡恢復，漏呼叫會卡到暫停逾時才恢復。
        self.pipeline.finalize_turn()  # 與一般輪次相同，收尾必須在輸出之後

    def _handle_utterance(self, event: Event) -> bool:
        """有人在說話。回傳 True 表示應結束主迴圈。"""
        text = event.payload
        if not text:
            return False

        if self.pipeline.is_command(text):
            result = self.pipeline.handle_command(
                text, self.state, self._voice_input_factory
            )
            if result.system_message:
                console.emit(f"{result.system_message}\n")
            return result.should_exit

        # ⚠️ 傳整個 Event 而不是 text：source／kind 是 Event 自己的欄位，
        # 只傳 payload 等於在入口把它們丟掉（events.py 檔頭：Event 由 Agent 產生、
        # 由 Pipeline 消費——傳整個才是它原本的設計意圖）。
        result = self.pipeline.handle_turn(event)
        if result.reply:
            if not result.printed:
                self._emit_reply(result.reply)
            logger.debug(f"奧狐回應：{result.reply[:50]}...")
            self.state.last_speech_at = time.monotonic()
            self._drive_vtube_on_reply()

        # 收尾一律在輸出之後，即使本輪無回應也要呼叫（v1 時是壓縮；準 v3 目前空轉，
        # 掛點保留給日後的離線處理——略過會讓那類處理無限期延後）。
        self.pipeline.finalize_turn()
        return False

    # ── 主迴圈 ───────────────────────────────────────────────────────

    def run(self) -> None:
        # 必須在所有 logger 建立之後——此時各模組已於 __init__ 完成初始化
        bridged = console.install_logging_bridge()
        logger.debug(f"log 輸出已導向終端協調器（{bridged} 個 handler）")

        console.emit("=== 奧狐 AOFOS 啟動 ===")
        # 🔺 2026-09-26：改問 has_history()——原本看 compressed_memory，
        #    而準 v3 切片沒有摘要 ⇒ 有記憶也永遠印「首次啟動」（《現況》二十一章 #7）。
        console.emit("[記憶已載入]" if self.memory.has_history() else "[首次啟動，無歷史記憶]")
        console.emit("模式：t = 文字輸入，v = 語音輸入（預設文字）")
        # 🔺 指令清單已改為取自 Pipeline.command_names()（單一來源），不再是字面清單。
        #    改的原因：2026-08-16 /proactive test 已隨腳手架刪除，這裡的字面清單仍列著它，
        #    照著打會靜默落入對話分支（奧狐會認真回答「/proactive test」這句話）。
        console.emit("指令：" + "、".join(self.pipeline.command_names()) )

        # 測試模式會改變記憶的寫入去向，必須在畫面上持續可見，
        # 否則容易在忘記自己開著的情況下把測試資料當成正常對話。
        if self.memory.test_mode:
            console.emit("\n" + "=" * 46)
            console.emit("  ⚠️  測試模式（TEST_MODE）目前為開啟狀態")
            console.emit("     記憶寫入只進測試區（讀取仍含日常區）")
            console.emit("     此為 config.py 的設定，非 /test 指令")
            console.emit("=" * 46)
        console.emit()

        self.window_watcher.start()

        while self.state.running:
            event = self._get_next_event()
            if event is None:
                continue

            if not self._arbitrate(event):
                continue

            if event.kind == OBSERVATION:
                self._handle_observation(event)
                continue

            if self._handle_utterance(event):
                self.window_watcher.stop()
                if self.vtube:
                    self.vtube.disconnect()
                self._close_voice_input()
                self._close_text_input_thread()
                break
