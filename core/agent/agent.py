# core/agent/agent.py
"""
Agent（中央控制核心 / Single Control Point）

職責：
    - 建立與持有所有核心模組（LLM / Persona / Memory / VTube / VoiceInput）
    - 可選模組初始化失敗時降級（如 VTube 連不上就跳過），不中斷主流程
    - 驅動主迴圈：從 event_queue 取事件 → 依 kind 分流 → 輸出 / 結束

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

from config import ARBITRATION_CONFIG
from core.llm_client import LLMClient
from core.persona import Persona
from core.memory.manager import MemoryManager
from core.logger import setup_logger
from core.agent.state import AgentState
from core.agent.pipeline import Pipeline
from core.agent.events import Event, OBSERVATION
from core.producers.window_watcher import WindowWatcher
from core.producers.user_input import OnDemandReader, text_reader, voice_reader
from core import console

logger = setup_logger("Agent")

# 主迴圈每次最多等多久就回來一次。不是輪詢（有事件會立刻醒）：只為了讓 Ctrl+C 進得來
# （EventQueue.get 註解）。
_WAIT_SLICE_S = 0.5


class Agent:
    def __init__(self) -> None:
        # 必要模組：初始化失敗應直接報錯，不做 fallback
        self.llm = LLMClient()
        self.persona = Persona()
        self.memory = MemoryManager(self.llm)

        # 可選模組：失敗需 fallback，不中斷主流程。連不上就不留物件（之後每輪不必再試）
        # ⚠️ 目前只連線、不驅動任何參數：沒有語音時動嘴是假訊號（嘴型等 TTS 接上再做）。
        self.vtube = None
        try:
            from core.vtube_client import VTubeClient   # 套件沒裝也算「連不上」
            vtube = VTubeClient()
            vtube.connect()
            self.vtube = vtube
        except Exception as e:
            logger.warning(f"VTube 未連線，跳過：{e}")

        self.state = AgentState(mode="t")
        self.pipeline = Pipeline(self.llm, self.persona, self.memory)

        # 事件來源。producer 只推事件、不決定是否發言（見 _arbitrate）
        self.window_watcher = WindowWatcher(self.state.event_queue)
        # 使用者輸入（core/producers/user_input.py）：文字的開機就建；語音的第一次要語音輸入時才建
        self._text = text_reader(self.state.event_queue)
        self._voice: Optional[OnDemandReader] = None
        self._voice_input = None
        self._closed = False
        self._shutdown_done = threading.Event()   # 關視窗時，系統給的幾秒內要等到它

    # ── 模組建立與清理 ───────────────────────────────────────────────

    def _reader_for_mode(self) -> OnDemandReader:
        """目前模式的輸入讀取器。語音的第一次用到才載模型，載不起來就留在文字模式。"""
        if self.state.mode != "v":
            return self._text
        if self._voice is None:
            try:
                # 函式內 import：文字模式的使用者不必載入語音模型。
                # ⚠️ 前提是沒有任何模組在模組層級 import core.voice_input（core/agent/selftest.py 會驗）。
                from core.voice_input import VoiceInput
                self._voice_input = VoiceInput(model_size="small", language="zh")
            except Exception as e:
                logger.error(f"語音輸入啟動失敗：{type(e).__name__}: {e}")
                console.emit(f"[語音模式啟動失敗，維持文字模式：{e}]\n")
                self.state.switch_to_text()
                return self._text
            self._voice = voice_reader(self.state.event_queue, self._voice_input)
        return self._voice

    def _shutdown(self) -> None:
        """
        收尾的唯一位置：quit、Ctrl+C、主迴圈例外都走這裡。各步獨立，前一步失敗不擋後一步；
        記憶排第一（最不能丟）。重複呼叫是安全的。

        ⚠️ 文字讀取若正卡在 input() 裡無法中斷（OS 層阻塞），靠 daemon 隨行程結束，不 join。
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._run_shutdown_steps()
            logger.info("收尾完成")   # 關視窗時畫面看不到，驗收看 logs/daily 有沒有這行
        finally:
            self._shutdown_done.set()

    def _run_shutdown_steps(self) -> None:
        steps = [("記憶", self.memory.close), ("視窗監看", self.window_watcher.stop),
                 ("文字輸入", self._text.close)]
        if self._voice is not None:
            steps += [("語音輸入", self._voice.close), ("麥克風", self._voice_input.close)]
        if self.vtube is not None:
            steps.append(("VTube", self.vtube.disconnect))
        for name, step in steps:
            try:
                step()
            except Exception as e:
                logger.error(f"收尾失敗（{name}，其餘照常）：{type(e).__name__}: {e}")

    # ── 輸出端副作用 ─────────────────────────────────────────────────

    def _emit_reply(self, reply: str) -> None:
        """
        非串流路徑的回應輸出。

        🔴 **前後綴一律取自 Pipeline，不在此重寫。** 兩條路徑的畫面格式必須逐字元
        一致，否則開關 stream 時行距會跳動——而那看起來像程式出了問題，會誘使人
        去查一個不存在的 bug（同 console.end_stream() 的註解）。
        """
        console.emit(
            f"{Pipeline.REPLY_PREFIX}{reply}{Pipeline.REPLY_TRAILING}", end=""
        )

    # ── 事件取得 ─────────────────────────────────────────────────────

    def _next_event(self) -> Optional[Event]:
        """
        取下一個事件；等不到回傳 None（主迴圈會再來）。

        佇列空了才向目前模式的讀取器要輸入：佇列裡還有事件時先開讀，「你：」會在
        下一輪回應前就跳出來。讀取器同一時間只會有一次讀取，重複要求不會疊加。
        """
        event = self.state.event_queue.get(timeout=0)
        if event is None:
            self._reader_for_mode().request()
            event = self.state.event_queue.get(timeout=_WAIT_SLICE_S)
        return event

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

        ⚠️ 尚未實作但位置已預留（TTS 接入主迴圈後才存在）：奧狐正在說話時使用者插話
        怎麼處理（《現況》第八章 P12、〈🔒 事件驅動〉，未裁決）。目前沒有語音輸出，
        輸出期間進來的輸入照一般事件排隊、下一輪處理。
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
        """環境事件：沒有人在說話。情境怎麼送見 Pipeline.handle_observation()。"""
        result = self.pipeline.handle_observation(event)
        if result.reply:
            # printed=True 代表回應已在生成過程中逐段印出（串流），再印一次會出現兩遍。
            if not result.printed:
                self._emit_reply(result.reply)
            logger.debug(f"奧狐主動發起：{result.reply[:50]}...")
            self.state.last_speech_at = time.monotonic()
        # 無回應也要收尾：背景判斷在輪次開始暫停、在這裡恢復，漏呼叫會卡到暫停逾時才恢復。
        self.pipeline.finalize_turn()  # 收尾必須在輸出之後

    def _handle_utterance(self, event: Event) -> None:
        """有人在說話（含指令）。quit 由 pipeline 設 state.running=False，主迴圈自己會停。"""
        text = event.payload
        if not text:
            return

        if self.pipeline.is_command(text):
            result = self.pipeline.handle_command(text, self.state)
            if result.system_message:
                console.emit(f"{result.system_message}\n")
            return

        # ⚠️ 傳整個 Event 而不是 text：source／kind 是 Event 自己的欄位，
        # 只傳 payload 等於在入口把它們丟掉（events.py 檔頭：Event 由 Agent 產生、
        # 由 Pipeline 消費——傳整個才是它原本的設計意圖）。
        result = self.pipeline.handle_turn(event)
        if result.reply:
            if not result.printed:
                self._emit_reply(result.reply)
            logger.debug(f"奧狐回應：{result.reply[:50]}...")
            self.state.last_speech_at = time.monotonic()

        # 收尾一律在輸出之後，即使本輪無回應也要呼叫（恢復背景判斷，同 _handle_observation）。
        self.pipeline.finalize_turn()

    # ── 主迴圈 ───────────────────────────────────────────────────────

    def run(self) -> None:
        console.install_logging_bridge()
        console.on_window_close(self._shutdown_done)   # 按 X 關視窗＝Ctrl+C，一樣收尾
        try:
            self._banner()
            self.window_watcher.start()
            while self.state.running:
                event = self._next_event()
                if event is None or not self._arbitrate(event):
                    continue
                if event.kind == OBSERVATION:
                    self._handle_observation(event)
                else:
                    self._handle_utterance(event)
        except KeyboardInterrupt:
            console.emit("\n[Ctrl+C：收尾後結束]")
        finally:
            self._shutdown()

    def _banner(self) -> None:
        console.emit("=== 奧狐 AOFOS 啟動 ===")
        console.emit("[記憶已載入]" if self.memory.has_history() else "[首次啟動，無歷史記憶]")
        console.emit("模式：t = 文字輸入，v = 語音輸入（預設文字）")
        # 指令清單取自 Pipeline.command_names()（單一來源），不在這裡寫字面清單。
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
