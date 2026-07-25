# core/agent/pipeline.py
"""
Pipeline(流程控制層)

職責：決定「一筆使用者輸入該怎麼被處理」——
是系統指令(quit / mode 切換 / 主動對話開關）還是正常對話輪次，
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

【2026-07-25 追加】主動對話觸發（handle_proactive_trigger）：
無使用者輸入時，U 內容固定傳空字串，不使用獨立 system 角色注入取代 U/A
結構本身（曾規劃比照 neuro-sama 的 Prompt Injection 模式完全取代 U，
經核對 LoRA 訓練資料格式規則後推翻，完整推理過程見《決策脈絡歸檔》
十一之2子問題四）。U 內容為空字串，即是「這輪為系統主動觸發、非真人
輸入」的訊號本身。

另外附加一則獨立的 system 情境訊息（說明目前是主動觸發情境），這符合
六之3既有規則的例外條款（u: 內容本身完全無法看出場合、且該場合資訊有
實際功能需求時才補 system），與「取代 U/A 結構」是兩件不同的事，不衝突
——此設計與 get_context() 既有的 Compressed Memory / RAG 附加 system
訊息模式一致，非新機制。此情境訊息不透過 add_turn() 寫入 Working Memory，
僅該次呼叫臨時附加，避免情境說明過期後仍卡在對話歷史裡。
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
        return user_input.lower() == "quit" or user_input in (
            "/mode t", "/mode v", "/proactive on", "/proactive off",
        )

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

        if user_input == "/proactive off":
            state.proactive_enabled = False
            return TurnResult(system_message="[主動對話觸發已暫停]")

        if user_input == "/proactive on":
            state.proactive_enabled = True
            return TurnResult(system_message="[主動對話觸發已恢復]")

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

    # ── 主動對話觸發（新增）─────────────────────────────────────────

    def handle_proactive_trigger(self) -> TurnResult:
        """
        主動對話觸發：無使用者輸入，U 內容為空字串（符合 LoRA 訓練資料
        「三欄位結構必須存在、內容可為空值」的格式規則），不偽裝成使用者
        發言、不取代 U/A 結構本身。

        另外附加一則獨立的 system 情境訊息，說明「這是系統主動觸發」——
        這符合六之3既有規則的例外條款（u: 內容本身完全無法看出場合，
        且該場合資訊有實際功能需求時才補 system），不是重新引入被推翻的
        injection 機制。被推翻的是「用 system 注入取代 U/A 結構」，這裡是
        「U 維持空字串不變，另外補一則不影響 U/A 結構的情境說明」，兩者
        性質不同，不衝突。

        情境說明訊息只存在於這一次呼叫 LLM 的當下，不會透過 add_turn()
        寫入 Working Memory——若持久化，下一輪組裝 conversation_history
        時會被原封不動回放出來，變成過期的情境描述卡在對話歷史裡（那時
        邪狼可能已經回話了，「已經一段時間沒有回應」這句話就是錯的資訊），
        所以只能每次呼叫當下臨時附加。

        情境說明內容目前為靜態字串；未來若加入話題轉移升級邏輯
        （見《決策脈絡歸檔》十一之2子問題七：多次觸發後可拉長間隔、
        轉移話題），此處會需要依觸發次數/觸發來源動態產生內容，
        現階段僅單一觸發源，維持靜態即可。
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