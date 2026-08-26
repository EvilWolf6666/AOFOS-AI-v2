# core/agent/pipeline.py
"""
Pipeline（流程控制層）

決定一個事件該怎麼被處理：系統指令、正常對話輪次，還是環境事件。
正常輪次的處理順序固定為：取記憶 context → 組裝 Prompt → 呼叫 LLM → 寫回記憶，
輪次收尾（記憶壓縮）另由 finalize_turn() 負責，由 Agent 在輸出之後呼叫。

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
from core.short_time_test import dump_observation_messages
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

    # 壓縮期間畫面完全沒有動靜，看起來像當機。兩條路徑共用（輪次收尾與離場）。
    # ⚠️ 寧可寫長不寫短：秒數隨 Working Memory 長度與背景 CPU 競爭浮動，
    #    提前結束無害，超出提示上限會被當成當機（實測依據見《現況》15之2）。
    COMPRESS_WAIT_MESSAGE = "[整理記憶中，約需 1 分鐘，請稍候……]"

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

        ⚠️ takes_arg 目前無任何使用者（C4 清理後），留著的成本是一個 bool；
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
        # ⚠️ compress_on_exit() 不檢查 needs_compress()，只要 working_memory
        # 非空就壓縮——講兩句話就退出也會等，故此處條件只看「有沒有東西要壓」。
        if self.memory.working_memory and not self.memory.test_mode:
            console.emit(self.COMPRESS_WAIT_MESSAGE)

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
            "  · RAG 已停用（不查詢、不寫入）\n"
            "  · 推理參數切為貪婪解碼（temperature=0、top_k=1），輸出可重現\n"
            "  · 串流輸出已停用（串流與非串流是否逐字相同尚未驗證，施測不賭這個假設）\n"
            "  · 記憶改用獨立測試檔，正式記憶檔完全未被改動\n"
            "  · Working Memory 已清空（僅存在記憶體、從不寫檔）\n"
            "  ⚠ 本次開機以來尚未壓縮的對話已直接丟棄，不會進入長期記憶\n"
            "  · 本設定不寫入 config，重開程式後自動回到正常模式"
        ))

    def _cmd_test_off(self, _state, _factory, _arg) -> TurnResult:
        self._set_test_mode(False)
        return TurnResult(system_message=(
            "[測試模式已關閉]\n"
            "  · RAG 已恢復\n"
            "  · 推理參數已還原為日常設定\n"
            "  · 串流輸出已恢復（依 config LLM_CONFIG['stream']）\n"
            "  · 記憶已切回正式檔\n"
            "  · Working Memory 已清空\n"
            "  ⚠ 測試期間尚未壓縮的對話已直接丟棄"
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
            f"  · RAG：{'啟用' if self.memory.rag.enabled else '停用'}\n"
            f"  · 記憶檔：{self.memory.memory_path}\n"
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

    def handle_turn(self, user_input: str) -> TurnResult:
        """取記憶 context → 組 Prompt → 呼叫 LLM → 追加記憶。壓縮見 finalize_turn()。"""
        context = self.memory.get_context(user_input)
        messages = self.persona.build_messages(context, user_input)
        reply, printed = self._chat_streamed(messages)

        if reply:
            self.memory.add_turn(user_input, reply)

        return TurnResult(reply=reply, printed=printed)

    # ── 輪次收尾 ──────────────────────────────────────────────────

    def finalize_turn(self) -> None:
        """
        視 Working Memory 長度決定是否壓縮記憶。

        🔴 必須由 Agent 在回應「已經輸出之後」呼叫——這是本方法存在的唯一理由。
        壓縮是一次完整的 LLM 呼叫，放在輸出前會直接落在使用者等待的路徑上。
        目前仍為同步執行，尖峰只是被移到下一次輸入之前。

        壓縮失敗不得中斷主流程。
        """
        try:
            if self.memory.needs_compress():
                console.emit(self.COMPRESS_WAIT_MESSAGE)
            self.memory.maybe_compress()
        except Exception as e:
            # 此處不用 logger：Pipeline 不持有 logger，錯誤細節已由 MemoryManager 記錄
            console.emit(f"[記憶壓縮失敗，不影響本輪對話：{e}]")

    # ── 環境事件輪次 ──────────────────────────────────────────────

    def handle_observation(self, description: str) -> TurnResult:
        """
        環境狀態變化（螢幕/遊戲事件等）觸發的輪次：沒有人在說話，故 user 欄位
        填 EVENT_CONFIG["user_placeholder"]（不偽裝成使用者發言），情境改由
        EVENT_CONFIG["observation_role"] 指定的角色帶入。與 handle_turn() 相同，
        此處只追加記憶不壓縮。

        🔴 role 的現行值、待定狀態與兩套 role 定義交錯的問題，唯一記載在
        config.EVENT_CONFIG["observation_role"]，此處不複述——改動前讀該處。

        情境訊息只存在於這次呼叫，不寫入 Working Memory。
        ⚠️ 已知會造成兩個症狀（連續觸發時模型照抄前一組回應、`_compress` 讀不到
        情境導致摘要抓不到事件）。待 observation 類訓練資料就緒後重新評估——
        模型學到「U 空時看情境說話」可能即可壓過複製傾向，故不預先修。

        🔴 情境訊息一律 append 在歷史「之後」、緊鄰生成點，順序不可前移。
        prompt cache 比對的是最長共同前綴，變動內容排最尾端時分岔點之後只需重算
        情境與空 user；前移則整段歷史被牽連重算，代價隨對話變長而增大。
        同一原理見 memory_manager.get_context()。

        ⚠️ placeholder、suffix、role 皆由 config 帶入，不寫死在此。

        ⚠️ **observation 輪次不查 RAG，這是實際行為但尚未確認是否應該如此。**
        傳給 get_context() 的是空 placeholder，該方法的 `if user_input:` 因此
        跳過查詢，即使 description 本身是可用的檢索字串。屬 ① 類，暫不改動——
        改動前先確認收益，不要當成 bug 直接「修好」。
        """
        placeholder = EVENT_CONFIG["user_placeholder"]
        suffix = EVENT_CONFIG["observation_prompt_suffix"]

        context = self.memory.get_context(placeholder)
        context.append({
            "role": EVENT_CONFIG["observation_role"],
            "content": f"[目前情境]\n{description}" + (f"\n{suffix}" if suffix else ""),
        })
        messages = self.persona.build_messages(context, placeholder)

        dump_observation_messages(messages)  # 腳手架，見 core/short_time_test.py

        reply, printed = self._chat_streamed(messages)

        if reply:
            self.memory.add_turn(placeholder, reply)

        return TurnResult(reply=reply, printed=printed)