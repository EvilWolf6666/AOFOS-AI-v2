# core/agent/pipeline.py
"""
Pipeline(流程控制層)

職責：決定「一筆使用者輸入該怎麼被處理」——
是系統指令(quit / mode 切換）還是正常對話輪次，
以及正常對話輪次的處理順序：
    取得記憶 context → 組裝 Prompt → 呼叫 LLM → 寫回記憶

這裡對應技術文件「Prompt 組裝策略」與「記憶調用方式」的決策職責。
目前組裝順序仍沿用 persona.build_messages() 內部固定的優先級
(Persona > 多模態 > Compressed Memory > RAG > Working Memory > 使用者輸入），
Pipeline 本身不重新定義這個順序，只負責「呼叫順序」（先取記憶、再組 Prompt、再推理)。

刻意不做的事（避免違反「不在問題出現前預先修」）：
- 不做 Tool Calling 判斷（現階段無工具可呼叫）
- 不做 Chain-of-Thought 拆解（用不到）
- 不做 Memory Router 第二層 Agent 語意確認
  (RAG 資料量仍是個位數，《決策脈絡歸檔》七章已明確記錄延後啟用條件）

已停用指令記錄（保留註解，不刪除，理由同原 main.py):
    /remember、/list 兩個 CLI 指令已於 2026-06-20 停用(見《決策脈絡歸檔》五之3)。
    停用後若使用者輸入這兩個字串，不會被 is_command() 攔截，
    會直接落入 handle_turn() 當一般對話處理——這是預期行為，不是 bug。
"""

from __future__ import annotations
from typing import Callable, Optional


class TurnResult:
    """單輪處理結果，讓 Agent 決定要不要印出來 / 要不要驅動 VTube / 要不要結束迴圈。"""

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
    """
    流程控制核心。不持有模組的建立/連線責任（那是 Agent 的職責），
    只負責「拿到已建好的模組」後，決定呼叫順序與分流邏輯。
    """

    def __init__(self, llm, persona, memory) -> None:
        self.llm = llm
        self.persona = persona
        self.memory = memory

    # ── 指令分流 ──────────────────────────────────────────────────

    def is_command(self, user_input: str) -> bool:
        return user_input.lower() == "quit" or user_input in ("/mode t", "/mode v")

    def handle_command(self, user_input: str, state, voice_input_factory: Callable) -> TurnResult:
        """
        處理系統指令。回傳 TurnResult 讓 Agent 決定要不要印訊息、要不要結束迴圈、
        要不要斷開 VTube 連線（那些是 Agent 持有的模組，Pipeline 不直接碰）。
        """
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

        # 理論上不會走到這裡（呼叫前應已經過 is_command() 過濾），保底防呆
        return TurnResult(system_message=None)

    # ── 正常對話輪次 ──────────────────────────────────────────────

    def handle_turn(self, user_input: str) -> TurnResult:
        """
        正常對話輪次：取記憶 context → 組 Prompt → 呼叫 LLM → 寫回記憶。

        VTube 表情/嘴型驅動不在這裡決策——那屬於「輸出端的可選副作用」，
        Pipeline 只保證「有沒有拿到回覆、記憶有沒有正確寫回」這件事本身正確，
        避免 Pipeline 也要認識 VTube 這個可選模組，維持職責單一。
        """
        context = self.memory.get_context(user_input)
        messages = self.persona.build_messages(context, user_input)
        reply = self.llm.chat(messages)

        if reply:
            self.memory.add_turn(user_input, reply)

        return TurnResult(reply=reply)
