# core/agent/pipeline.py
"""
Pipeline（流程控制層）

決定一個事件該怎麼被處理：系統指令、正常對話輪次，還是環境事件。
正常輪次的處理順序固定為：取記憶 context → 組裝 Prompt → 呼叫 LLM → 寫回記憶，
輪次收尾另由 finalize_turn() 負責，由 Agent 在輸出之後呼叫
（🔺 2026-09-27：準 v3 不壓縮，該掛點目前空轉；留著給日後的離線整理用）。

Prompt 組裝的優先順序由 persona.build_messages() 決定，此處只負責呼叫順序。

指令一律登記於 _commands，is_command() 與 handle_command() 共用同一份來源。
⚠️ 不得再開第二份清單——漏改任一邊都會靜默失效。

註：/remember、/list 已停用且不在 _commands 內，輸入這兩個字串會落入
handle_turn() 當一般對話處理（預期行為，非 bug）。
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple

from config import EVENT_CONFIG, PERSONA_CONFIG
from core.agent.events import Event, OBSERVATION
from core import console


class TurnResult:
    """單輪處理結果，供 Agent 決定要不要印出來 / 驅動 VTube / 結束迴圈。"""

    def __init__(
        self,
        should_exit: bool = False,
        reply: Optional[str] = None,
        system_message: Optional[str] = None,
        printed: bool = False,
    ) -> None:
        self.should_exit = should_exit
        self.reply = reply
        self.system_message = system_message

        # printed=True 代表 reply 已逐段印到畫面上，Agent 不可再印一次
        # （否則串流開啟時每句話出現兩遍）。
        self.printed = printed


class _StreamSink:
    """
    串流輸出的接收端：第一次真的收到內容時才開始串流輸出。

    延遲到第一個片段才 begin_stream()：非串流路徑下回呼從未被呼叫，used 保持
    False，Agent 照舊整段印出——**Pipeline 因此不需要自己判斷會不會串流**，
    該判斷留在 LLMClient 一處。另可避免呼叫失敗時畫面留下孤零零的前綴。
    """

    def __init__(self, prefix: str, trailing: str = "\n") -> None:
        self.prefix = prefix
        self.trailing = trailing
        self.used = False

    def __call__(self, text: str) -> None:
        if not self.used:
            self.used = True
            console.begin_stream(self.prefix)
        console.write_stream(text)

    def close(self) -> None:
        if self.used:
            console.end_stream(self.trailing)


class Pipeline:
    """不負責模組的建立與連線（那是 Agent 的職責），只負責呼叫順序與分流邏輯。"""


    # 串流輸出的前後綴。⚠️ 必須與 agent.py 的非串流輸出格式逐字元一致，
    # 否則開關 stream 時畫面行距會跳動。
    REPLY_PREFIX = f"\n{PERSONA_CONFIG['name']}："
    REPLY_TRAILING = "\n\n"

    def __init__(self, llm, persona, memory) -> None:
        self.llm = llm
        self.persona = persona
        self.memory = memory

        # 指令的唯一來源。key 一律小寫、以空白分段，比對前經 _key() 正規化。
        # 值為 (handler, takes_arg)；takes_arg=True 者允許在指令後接參數。
        self._commands = {
            "quit":            (self._cmd_quit, False),
            "/mode t":         (self._cmd_mode_text, False),
            "/mode v":         (self._cmd_mode_voice, False),
            "/proactive on":   (self._cmd_proactive_on, False),
            "/proactive off":  (self._cmd_proactive_off, False),
            "/test on":        (self._cmd_test_on, False),
            "/test off":       (self._cmd_test_off, False),
            "/status":         (self._cmd_status, False),
        }

    # ── 指令分流 ──────────────────────────────────────────────────

    @staticmethod
    def _key(user_input: str) -> str:
        return user_input.strip().lower()

    def _resolve(self, user_input: str) -> Optional[Tuple[Callable, str]]:
        """
        找出對應的 handler 與參數字串，找不到回傳 None。

        由長至短逐段比對，因為指令 key 本身含空白（"/mode t"）。取第一段會讓
        "/proactive on" 誤命中不存在的 "/proactive"。

        ⚠️ takes_arg 目前無任何使用者（《現況》第八章 舊C4（已結案）清理後），留著的成本是一個 bool；
        新增帶參數的指令時登記 takes_arg=True 即可。
        ⚠️ 參數取自 _key() 正規化後的字串，已轉小寫；需大小寫敏感時改由原始輸入取。
        """
        key = self._key(user_input)
        parts = key.split()

        for i in range(len(parts), 0, -1):
            entry = self._commands.get(" ".join(parts[:i]))
            if entry is None:
                continue
            handler, takes_arg = entry
            arg = " ".join(parts[i:])
            if takes_arg or not arg:
                return handler, arg
        return None

    def command_names(self) -> list[str]:
        """
        供啟動訊息列出可用指令。

        ⚠️ 存在理由是消滅第三份真相——agent.py 的啟動訊息不得再寫字面清單。

        排序照登記順序（dict 有序），不另外排序：登記順序依相關性分組，
        字母序會讓 /mode 與 /proactive 打散。
        """
        return list(self._commands.keys())

    def is_command(self, user_input: str) -> bool:
        return self._resolve(user_input) is not None

    def handle_command(self, user_input: str, state, voice_input_factory: Callable) -> TurnResult:
        """
        處理系統指令，回傳結果讓 Agent 決定後續動作（此處不直接操作 VTube 等模組）。

        刻意不做保底回傳：呼叫端必定先經 is_command() 過濾，走到這裡卻找不到
        handler 代表分流被繞過，應以 KeyError 大聲失敗，而非靜默回傳空結果。
        """
        resolved = self._resolve(user_input)
        if resolved is None:
            raise KeyError(user_input)
        handler, arg = resolved
        return handler(state, voice_input_factory, arg)

    # ── 指令 handler（簽名統一為 (state, voice_input_factory, arg)）─

    def _cmd_quit(self, state, _factory, _arg) -> TurnResult:
        # 離場收尾（補寫原句、關背景判斷、重建事件層；毫秒級，不需要「請稍候」提示）。
        # 🔺 2026-10-09 v1 壓縮的相容殼（needs_compress／COMPRESS_WAIT_MESSAGE）已移除。
        self.memory.compress_on_exit()
        state.stop()
        return TurnResult(should_exit=True, system_message="掰掰。")

    def _cmd_mode_text(self, state, _factory, _arg) -> TurnResult:
        state.switch_to_text()
        return TurnResult(system_message="[切換為文字模式]")

    def _cmd_mode_voice(self, state, voice_input_factory, _arg) -> TurnResult:
        state.switch_to_voice(voice_input_factory)
        return TurnResult(system_message="[切換為語音模式]")

    def _cmd_proactive_on(self, state, _factory, _arg) -> TurnResult:
        state.proactive_enabled = True
        return TurnResult(system_message="[已開啟：會對環境事件作出反應]")

    def _cmd_proactive_off(self, state, _factory, _arg) -> TurnResult:
        state.proactive_enabled = False
        return TurnResult(system_message="[已關閉：環境事件將被忽略]")

    def _cmd_test_on(self, _state, _factory, _arg) -> TurnResult:
        self._set_test_mode(True)
        return TurnResult(system_message=(
            "[測試模式已開啟]\n"
            "  · 記憶：讀取共用（日常區＋測試區）、寫入只進測試區；日常區整輪不變動\n"
            "  · 推理參數切為貪婪解碼（temperature=0、top_k=1）\n"
            "    ⚠ 2026-09-27 實測：開機後第一次請求的輸出曾與之後不同，逐字重現不保證\n"
            "  · 串流輸出已停用（串流與非串流是否逐字相同尚未驗證，施測不賭這個假設）\n"
            "  · Working Memory 已清空（本場已說的話先前已寫進日常區，沒有遺失）\n"
            "  · 本設定不寫入 config，重開程式後自動回到正常模式"
        ))

    def _cmd_test_off(self, _state, _factory, _arg) -> TurnResult:
        self._set_test_mode(False)
        return TurnResult(system_message=(
            "[測試模式已關閉]\n"
            "  · 推理參數已還原為日常設定\n"
            "  · 串流輸出已恢復（依 config LLM_CONFIG['stream']）\n"
            "  · 記憶已切回日常區；測試區已整包丟棄（測試期間的對話不進日常記憶）\n"
            "  · Working Memory 已清空"
        ))

    def _set_test_mode(self, on: bool) -> None:
        """
        記憶隔離與推理參數是兩件不同的事，但施測時必須同時成立，
        故由 Pipeline 統一協調，不讓任一模組去驅動另一個模組。
        """
        self.memory.set_test_mode(on)
        self.llm.set_test_mode(on)

    def _cmd_status(self, state, _factory, _arg) -> TurnResult:
        return TurnResult(system_message=(
            "[目前狀態]\n"
            f"  · 模式：{'文字' if state.mode == 't' else '語音'}\n"
            f"  · 環境事件反應：{'開啟' if state.proactive_enabled else '關閉'}\n"
            f"  · 佇列待處理事件：{self._queue_summary(state)}\n"
            f"  · 測試模式：{'開啟' if self.memory.test_mode else '關閉'}\n"
            f"  · 推理參數：temperature={self.llm.temperature} "
            f"top_p={self.llm.top_p} freq_pen={self.llm.frequency_penalty} "
            f"extra_body={self.llm.extra_body}\n"
            f"  · 串流輸出：{'開啟' if self.llm.stream else '關閉'}"
            f"{'（測試模式下強制停用中）' if self.memory.test_mode else ''}\n"
            f"  · 記憶檢索：{self.memory.status_text()}\n"
            f"  · 記憶寫入：{self.memory.memory_path}\n"
            f"  · Working Memory：{len(self.memory.working_memory)} 則"
        ))

    @staticmethod
    def _queue_summary(state) -> str:
        """佇列內容摘要。只印數量看不出是哪個 producer 在累積事件。"""
        events = state.event_queue.snapshot()
        if not events:
            return "0 則"
        by_source: dict[str, int] = {}
        for e in events:
            by_source[e.source] = by_source.get(e.source, 0) + 1
        detail = "、".join(f"{k}×{v}" for k, v in by_source.items())
        return f"{len(events)} 則（{detail}）"

    # ── 正常對話輪次 ──────────────────────────────────────────────

    def _chat_streamed(self, messages: list, purpose: str = "dialogue") -> Tuple[str, bool]:
        """
        呼叫 LLM 並在支援時逐段印出。回傳 (完整回應, 是否已印出)。

        ⚠️ close() 必須放在 finally：串流中途拋例外若沒還原 console 的串流狀態，
        之後所有 emit() 都停在「不重畫提示」模式，症狀看起來像當機而非例外。
        """
        sink = _StreamSink(self.REPLY_PREFIX, self.REPLY_TRAILING)
        try:
            reply = self.llm.chat(messages, purpose=purpose, on_delta=sink)
        finally:
            sink.close()
        return reply, sink.used

    def handle_turn(self, event: Event) -> TurnResult:
        """取記憶 context → 組 Prompt → 呼叫 LLM → 追加記憶。輸出後收尾見 finalize_turn()。

        ⚠️ 收整個 Event 而不是字串：`source` 與 `kind` 是 Event 自己的欄位，
        在此只取 payload 等於把它們丟掉。Pipeline 本來就是 Event 的消費者
        （見 events.py 檔頭），收整個才是它原本的設計意圖。

        🔴 **但只傳到這一層為止。** 往下呼叫 MemoryManager 時改傳明確的值，
        不傳 Event 物件——core/memory/ 是 core/agent/ 的下層，讓它 import
        core.agent.events 會造成反向依賴，而那正是 events.py 與 ctrl_signals.py
        兩處檔頭都明寫要避免的事。
        ⭐ 附帶好處：**記憶的欄位集是它自己的規格，不該隨 Event 新增欄位而漂移。**
        """
        user_input = event.payload
        # 準 v4：背景判斷先讓出推論槽（推理後端單槽，背景請求在跑會讓本輪排隊）。
        # 恢復在 finalize_turn（輸出之後）。
        self.memory.bg_pause()
        # 記憶區塊可能併進本輪 user 訊息（config 的 block_position）；寫回記憶仍用原始輸入
        context, user_content = self.memory.build_turn(user_input)
        messages = self.persona.build_messages(context, user_content)
        reply, printed = self._chat_streamed(messages)

        if reply:
            self.memory.add_turn(
                user_input, reply, source=event.source, kind=event.kind
            )

        return TurnResult(reply=reply, printed=printed)

    # ── 輪次收尾 ──────────────────────────────────────────────────

    def finalize_turn(self) -> None:
        """
        輸出後的收尾掛點：恢復背景判斷（輪次開始時 bg_pause）。
        🔴 必須由 Agent 在回應「已經輸出之後」呼叫——早了背景請求會跟還沒送完的對話搶推論槽。
        失敗不得中斷主流程。
        🔺 2026-10-09 v1 壓縮（maybe_compress）的相容殼已移除；日後要在這裡掛任何離線處理，
           同樣只能放在輸出之後、且不得放在使用者等待的路徑上。
        """
        try:
            self.memory.bg_resume()
        except Exception as e:
            # 此處不用 logger：Pipeline 不持有 logger，錯誤細節已由 MemoryManager 記錄
            console.emit(f"[背景記憶恢復失敗，不影響本輪對話：{e}]")

    # ── 環境事件輪次 ──────────────────────────────────────────────

    def handle_observation(self, event: Event) -> TurnResult:
        """
        環境狀態變化（螢幕/遊戲事件等）觸發的輪次：沒有人在說話。情境怎麼送看
        EVENT_CONFIG["observation_delivery"]（2026-10-23 起 "in_user"：情境就是這一輪
        user 訊息的內容；舊做法 "separate_role"：一則 observation_role 訊息＋空 user）。
        與 handle_turn() 相同，此處只追加記憶。

        🔴 兩種送法的實測差異、為什麼不能改成尾端 system，唯一記載在
        config.EVENT_CONFIG["observation_delivery"]，此處不複述——改動前讀該處。

        寫回 WM 的 user 內容是 placeholder，情境掛在 situation 欄位；之後幾輪看不看得到情境，
        由 EVENT_CONFIG["situation_in_history"] 決定（方案 α；現值與各選項的依據只記在該處註解，此處不抄值），渲染在 manager._history()。
        ⚠️ 照抄前一組回應：舊做法（"tool" 放尾端）在貪婪解碼、真實情境格式下，
        連續觸發會回同一句（2026-10-23，《現況》22之84）；in_user 0 次。

        ⚠️ separate_role 時情境訊息 append 在歷史「之後」、緊鄰生成點，不可前移
        （prompt cache 比對最長共同前綴，前移會讓整段歷史重算）。in_user 時情境本來就在最後一則。

        ⚠️ placeholder、suffix、送法、role 皆由 config 帶入，不寫死在此。

        🔴 **情境描述會進記憶（方案 β：掛欄位；之後幾輪顯不顯示看 situation_in_history）。**
        `add_turn(..., situation=description)` 把它掛在**該輪的 user 那一則**上
        ——那一則的 content 是空 placeholder，情境正是那個位置該有的東西。
        ⭐ **「user 空 ＋ 有 situation」是自洽的形狀；「user 空 ＋ 沒有 situation」
        一眼看得出是異常。** 先前情境只存在於這一次呼叫，事件層會留下一批
        「她突然說了一段話而沒有人知道為什麼」的紀錄。
        ⚠️ 非標準欄位一律由 manager._to_message() 投影掉——**那個投影是本做法成立的前提，
        不是防禦性程式碼**。situation 是唯一例外：依 situation_in_history 渲染成那則 user 的內容
        （方案 α，2026-10-24 邪狼裁；實測見《現況》22之84 五），欄位本身仍不送。

        ⚠️ **observation 輪次不做記憶檢索，這是實際行為但尚未確認是否應該如此。**
        傳給 build_turn() 的是空 placeholder，該方法的 `if user_input:` 因此
        跳過查詢，即使 description 本身是可用的檢索字串（開場帶入仍照常）。
        改動前先確認收益，不要當成 bug 直接「修好」。
        """
        description = event.payload
        placeholder = EVENT_CONFIG["user_placeholder"]
        self.memory.bg_pause()   # 同 handle_turn
        if EVENT_CONFIG["observation_delivery"] == "in_user":
            # 選項 D（2026-10-23）：情境就是這一輪 user 訊息的內容；檢索仍看空 placeholder（不檢索）
            context, user_content = self.memory.build_turn(placeholder, situation=description)
        elif EVENT_CONFIG["observation_delivery"] == "separate_role":
            context, user_content = self.memory.build_turn(placeholder)
            context.append({"role": EVENT_CONFIG["observation_role"],
                            "content": self.memory.render_situation(description)})
        else:
            raise ValueError(f"EVENT_CONFIG['observation_delivery'] 只能是 in_user／separate_role："
                             f"{EVENT_CONFIG['observation_delivery']!r}")
        messages = self.persona.build_messages(context, user_content)

        reply, printed = self._chat_streamed(messages)

        if reply:
            self.memory.add_turn(
                placeholder, reply,
                source=event.source, kind=event.kind, situation=description,
                user_speaker=None,   # 準 v4：沒有人說話（user 那則是 placeholder）
            )

        return TurnResult(reply=reply, printed=printed)

