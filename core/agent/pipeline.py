# core/agent/pipeline.py
"""
Pipeline（流程控制層）

決定一筆輸入該怎麼被處理：系統指令，還是正常對話輪次。
正常輪次的處理順序固定為：取記憶 context → 組裝 Prompt → 呼叫 LLM → 寫回記憶，
輪次收尾（記憶壓縮）另由 finalize_turn() 負責，由 Agent 在輸出之後呼叫。

Prompt 組裝的優先順序由 persona.build_messages() 決定，此處只負責呼叫順序。

註：/remember、/list 已停用且不在 is_command() 攔截範圍內，輸入這兩個字串
會落入 handle_turn() 當一般對話處理（預期行為，非 bug）。
"""

from __future__ import annotations
from typing import Callable, Optional


class TurnResult:
    """單輪處理結果，供 Agent 決定要不要印出來 / 驅動 VTube / 結束迴圈。"""

    def __init__(
        self,
        should_exit: bool = False,
        reply: Optional[str] = None,
        system_message: Optional[str] = None,
    ) -> None:
        self.should_exit = should_exit
        self.reply = reply
        self.system_message = system_message


class Pipeline:
    """不負責模組的建立與連線（那是 Agent 的職責），只負責呼叫順序與分流邏輯。"""

    def __init__(self, llm, persona, memory) -> None:
        self.llm = llm
        self.persona = persona
        self.memory = memory

    # ── 指令分流 ──────────────────────────────────────────────────

    def is_command(self, user_input: str) -> bool:
        return user_input.lower() == "quit" or user_input in (
            "/mode t", "/mode v", "/proactive on", "/proactive off",
            "/test on", "/test off", "/status",
        )

    def handle_command(self, user_input: str, state, voice_input_factory: Callable) -> TurnResult:
        """處理系統指令，回傳結果讓 Agent 決定後續動作（此處不直接操作 VTube 等模組）。"""
        if user_input.lower() == "quit":
            self.memory.compress_on_exit()
            state.stop()
            return TurnResult(should_exit=True, system_message="掰掰。")

        if user_input == "/mode t":
            state.switch_to_text()
            return TurnResult(system_message="[切換為文字模式]")

        if user_input == "/mode v":
            state.switch_to_voice(voice_input_factory)
            return TurnResult(system_message="[切換為語音模式]")

        if user_input == "/proactive off":
            state.proactive_enabled = False
            return TurnResult(system_message="[主動對話觸發已暫停]")

        if user_input == "/proactive on":
            state.proactive_enabled = True
            return TurnResult(system_message="[主動對話觸發已恢復]")

        if user_input in ("/test on", "/test off"):
            on = user_input == "/test on"
            # 記憶隔離與推理參數是兩件不同的事，但施測時必須同時成立，
            # 故由 Pipeline 統一協調，不讓任一模組去驅動另一個模組。
            self.memory.set_test_mode(on)
            self.llm.set_test_mode(on)
            if on:
                return TurnResult(system_message=(
                    "[測試模式已開啟]\n"
                    "  · RAG 已停用（不查詢、不寫入）\n"
                    "  · 推理參數切為貪婪解碼（temperature=0、top_k=1），輸出可重現\n"
                    "  · 記憶改用獨立測試檔，正式記憶檔完全未被改動\n"
                    "  · Working Memory 已清空（僅存在記憶體、從不寫檔）\n"
                    "  ⚠ 本次開機以來尚未壓縮的對話已直接丟棄，不會進入長期記憶\n"
                    "  · 本設定不寫入 config，重開程式後自動回到正常模式"
                ))
            return TurnResult(system_message=(
                "[測試模式已關閉]\n"
                "  · RAG 已恢復\n"
                "  · 推理參數已還原為日常設定\n"
                "  · 記憶已切回正式檔\n"
                "  · Working Memory 已清空\n"
                "  ⚠ 測試期間尚未壓縮的對話已直接丟棄"
            ))

        if user_input == "/status":
            return TurnResult(system_message=(
                "[目前狀態]\n"
                f"  · 模式：{'文字' if state.mode == 't' else '語音'}\n"
                f"  · 主動對話觸發：{'開啟' if state.proactive_enabled else '關閉'}\n"
                f"  · 測試模式：{'開啟' if self.memory.test_mode else '關閉'}\n"
                f"  · 推理參數：temperature={self.llm.temperature} "
                f"top_p={self.llm.top_p} freq_pen={self.llm.frequency_penalty} "
                f"extra_body={self.llm.extra_body}\n"
                f"  · RAG：{'啟用' if self.memory.rag.enabled else '停用'}\n"
                f"  · 記憶檔：{self.memory.memory_path}\n"
                f"  · Working Memory：{len(self.memory.working_memory)} 則"
            ))

        # 呼叫前已經過 is_command() 過濾，走到這裡代表兩處清單不同步，保底防呆
        return TurnResult(system_message=None)

    # ── 正常對話輪次 ──────────────────────────────────────────────

    def handle_turn(self, user_input: str) -> TurnResult:
        """取記憶 context → 組 Prompt → 呼叫 LLM → 追加記憶。壓縮見 finalize_turn()。"""
        context = self.memory.get_context(user_input)
        messages = self.persona.build_messages(context, user_input)
        reply = self.llm.chat(messages)

        if reply:
            self.memory.add_turn(user_input, reply)

        return TurnResult(reply=reply)

    # ── 輪次收尾 ──────────────────────────────────────────────────

    def finalize_turn(self) -> None:
        """
        視 Working Memory 長度決定是否壓縮記憶。

        ⚠️ 必須由 Agent 在回應「已經輸出之後」呼叫，這是本方法存在的唯一理由——
        壓縮是一次完整的 LLM 呼叫（實測 20～30 秒），放在輸出前會直接落在
        使用者等待的路徑上。目前仍為同步執行，尖峰只是被移到下一次輸入之前。

        壓縮失敗不得中斷主流程：它是輔助功能，不該讓一次失敗拖垮對話。
        """
        try:
            # 壓縮期間畫面完全沒有動靜，看起來像當機（實測時使用者因此連按 Enter）
            if self.memory.needs_compress():
                print("[整理記憶中，約需 20～30 秒，請稍候……]")
            self.memory.maybe_compress()
        except Exception as e:
            # 此處不用 logger：Pipeline 不持有 logger，錯誤細節已由 MemoryManager 記錄
            print(f"[記憶壓縮失敗，不影響本輪對話：{e}]")

    # ── 主動對話觸發 ──────────────────────────────────────────────

    def handle_proactive_trigger(self) -> TurnResult:
        """
        無使用者輸入時觸發：U 內容固定為空字串（符合 LoRA 訓練資料格式規則，
        不偽裝成使用者發言），另附一則臨時 system 訊息說明目前是主動觸發情境。

        該 system 訊息只存在於這次呼叫，不寫入 Working Memory，避免情境說明
        過期後仍卡在對話歷史裡。與 handle_turn() 相同，此處只追加記憶不壓縮。
        """
        context = self.memory.get_context("")
        context.append({
            "role": "system",
            "content": "[目前情境]\n邪狼已經一段時間沒有回應，妳可以主動找話題延續對話，"
            "如果妳剛才已經問過類似的話，這次換個不同的方向。",
        })
        messages = self.persona.build_messages(context, "")
        reply = self.llm.chat(messages)

        if reply:
            self.memory.add_turn("", reply)

        return TurnResult(reply=reply)