"""
Pipeline（流程控制層）

決定一筆輸入該怎麼被處理：系統指令（quit / mode 切換 / 主動對話開關）
還是正常對話輪次；正常對話輪次的處理順序固定為：
取得記憶 context → 組裝 Prompt → 呼叫 LLM → 寫回記憶。

Prompt 組裝的優先順序由 persona.build_messages() 內部決定，
Pipeline 只負責呼叫順序，不重新定義組裝邏輯。

/remember、/list 已停用且不在 is_command() 攔截範圍內，
使用者輸入這兩個字串會直接落入 handle_turn() 當一般對話處理。
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
    """不持有模組的建立/連線責任（那是 Agent 的職責），只負責呼叫順序與分流邏輯。"""

    def __init__(self, llm, persona, memory) -> None:
        self.llm = llm
        self.persona = persona
        self.memory = memory

    # ── 指令分流 ──────────────────────────────────────────────────

    def is_command(self, user_input: str) -> bool:
        return user_input.lower() == "quit" or user_input in (
            "/mode t", "/mode v", "/proactive on", "/proactive off",
        )

    def handle_command(self, user_input: str, state, voice_input_factory: Callable) -> TurnResult:
        """處理系統指令，回傳結果讓 Agent 決定後續動作（Pipeline 不直接操作 VTube 等模組）。"""
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

        # 不會走到這裡（呼叫前已經過 is_command() 過濾），保底防呆
        return TurnResult(system_message=None)

    # ── 正常對話輪次 ──────────────────────────────────────────────

    def handle_turn(self, user_input: str) -> TurnResult:
        """
        取記憶 context → 組 Prompt → 呼叫 LLM → 寫回記憶。
        VTube 驅動屬於輸出端的可選副作用，不在此決策。
        """
        context = self.memory.get_context(user_input)
        messages = self.persona.build_messages(context, user_input)
        reply = self.llm.chat(messages)

        if reply:
            self.memory.add_turn(user_input, reply)

        return TurnResult(reply=reply)

    # ── 主動對話觸發 ──────────────────────────────────────────────

    def handle_proactive_trigger(self) -> TurnResult:
        """
        無使用者輸入時觸發：U 內容固定為空字串（符合 LoRA 訓練資料格式規則，
        不偽裝成使用者發言），另外附加一則臨時 system 訊息說明目前是主動觸發情境。

        該 system 訊息只存在於這次呼叫，不透過 add_turn() 寫入 Working Memory，
        避免情境說明過期後仍卡在對話歷史裡。
        """
        context = self.memory.get_context("")
        context.append({
            "role": "system",
            "content": "[目前情境]\n邪狼已經一段時間沒有回應，妳可以主動找話題延續對話。",
        })
        messages = self.persona.build_messages(context, "")
        reply = self.llm.chat(messages)

        if reply:
            self.memory.add_turn("", reply)

        return TurnResult(reply=reply)