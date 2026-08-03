# core/agent/pipeline.py
"""
Pipeline（流程控制層）

決定一個事件該怎麼被處理：系統指令、正常對話輪次，還是環境事件。
正常輪次的處理順序固定為：取記憶 context → 組裝 Prompt → 呼叫 LLM → 寫回記憶，
輪次收尾（記憶壓縮）另由 finalize_turn() 負責，由 Agent 在輸出之後呼叫。

Prompt 組裝的優先順序由 persona.build_messages() 決定，此處只負責呼叫順序。

指令一律登記於 _commands（見 __init__），is_command() 與 handle_command() 共用
同一份來源。新增指令只需加一個 handler 並登記一行——這是刻意的結構：先前兩處
各有一份清單，漏改任一邊都會靜默失效（漏 A 則指令被當對話送給 LLM，看起來像
正常回話；漏 B 則什麼都不印，看起來像當機）。

註：/remember、/list 已停用且不在 _commands 內，輸入這兩個字串會落入
handle_turn() 當一般對話處理（預期行為，非 bug）。
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple

from config import EVENT_CONFIG
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
    ) -> None:
        self.should_exit = should_exit
        self.reply = reply
        self.system_message = system_message


class Pipeline:
    """不負責模組的建立與連線（那是 Agent 的職責），只負責呼叫順序與分流邏輯。"""

    # 壓縮期間畫面完全沒有動靜，看起來像當機（實測時使用者因此連按 Enter）。
    # 抽成常數是因為兩條路徑都要用（輪次收尾與離場），分開寫會在改秒數時漏一邊。
    COMPRESS_WAIT_MESSAGE = "[整理記憶中，約需 20～30 秒，請稍候……]"

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
            "/proactive test": (self._cmd_proactive_test, True),
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

        由長至短逐段比對，因為指令 key 本身就含空白（"/mode t"、"/proactive on"），
        單純取第一段會讓 "/proactive test 3" 誤命中不存在的 "/proactive"。
        完整比對優先，確保 "/proactive test" 不會被當成別的指令加參數。

        ⚠️ 參數取自 _key() 正規化後的字串，已轉小寫。目前唯一的參數是情境編號，
        不受影響；日後若加入大小寫敏感的參數，需改由原始輸入取。
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
        # 離場壓縮同樣是一次完整的 LLM 呼叫（實測 17～25 秒），且 quit 之後
        # 使用者更傾向認為程式已經在關了，卡住不動的誤判機率比一般輪次更高。
        # ⚠️ compress_on_exit() 不檢查 needs_compress()，只要 working_memory
        # 非空就壓縮——講兩句話就退出也會等，故條件只看「有沒有東西要壓」。
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

    def _cmd_proactive_test(self, state, _factory, arg) -> TurnResult:
        """
        手動注入一則測試用 observation 事件（腳手架，清理條件見
        config.EVENT_CONFIG["test_observations"]）。

        用法：/proactive test        依序循環取下一則
              /proactive test 3     指定第 3 則

        指定編號的用途是變數隔離：每次開機只讓她開口一次，才能確保多次比較之間
        working memory 皆為空、唯一變動的是情境文字。靠空打指令推進循環位置，
        每推進一次就先污染一次記憶。

        只推事件進佇列，不直接呼叫 LLM——走的必須是與真實 producer 完全相同的
        路徑，否則測不到分流本身是否正確。
        """
        observations = EVENT_CONFIG["test_observations"]
        if not observations:
            return TurnResult(system_message="[未設定測試情境]")

        if arg:
            if not arg.isdigit() or not (1 <= int(arg) <= len(observations)):
                return TurnResult(
                    system_message=f"[編號需為 1～{len(observations)} 的整數]"
                )
            index = int(arg) - 1
        else:
            index = state.observation_test_index % len(observations)
            state.observation_test_index += 1

        description = observations[index]
        state.event_queue.put(
            Event(source="test", kind=OBSERVATION, payload=description)
        )

        note = "" if state.proactive_enabled else "\n  ⚠ proactive 目前關閉，此事件會被忽略"
        return TurnResult(system_message=(
            f"[注入測試事件 {index + 1}/{len(observations)}] {description}{note}"
        ))

    def _cmd_test_on(self, _state, _factory, _arg) -> TurnResult:
        self._set_test_mode(True)
        return TurnResult(system_message=(
            "[測試模式已開啟]\n"
            "  · RAG 已停用（不查詢、不寫入）\n"
            "  · 推理參數切為貪婪解碼（temperature=0、top_k=1），輸出可重現\n"
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

        ⚠️ role 現行值為 "tool"，不是 "system"。LM Studio 有 role 白名單擋在
        chat_template 之前，"observation" 與 "ipython" 皆送不出去；白名單內的
        "tool" 會被 jinja 渲染成 <|start_header_id|>ipython，與訓練端
        LLaMA-Factory 的 format_observation 輸出一致。見《決策脈絡歸檔》二十二之4。

        情境訊息只存在於這次呼叫，不寫入 Working Memory。
        ⚠️ 已知會造成兩個症狀（連續觸發時模型照抄前一組回應、`_compress` 讀不到
        情境導致摘要抓不到事件）。待 observation 類訓練資料就緒後重新評估——
        模型學到「U 空時看情境說話」可能即可壓過複製傾向，故不預先修。
        見《決策脈絡歸檔》二十之7 之5、二十二之7。

        情境訊息一律 append 在歷史「之後」、緊鄰生成點。prompt cache 是 token
        序列的最長共同前綴比對，變動內容排在最尾端時，分岔點之後只需重算情境與
        空 user；若前移到歷史之前，整段歷史都會被牽連重算，且代價隨對話變長而
        增大。訓練端只能把 system 放在最前面、無法表達此結構，這正是情境必須走
        observation role 的原因，見《決策脈絡歸檔》二十二之3。

        ⚠️ placeholder、suffix、role 皆由 config 帶入，不寫死在此：訓練端格式
        若再變動，推理端只需改 config 一處。

        ⚠️ **observation 輪次不查 RAG——這是目前的實際行為，但尚未確認是否應該
        如此**（2026-08-03 程式碼審查 I-2 提出）。傳給 get_context() 的是空的
        placeholder，該方法的 `if user_input:` 因此跳過 RAG 查詢，即使
        description 本身（例如「邪狼剛才在 VS Code 待了快 40 分鐘」）其實是
        可用的檢索字串。暫不改動的理由：RAG 查詢實測約 2 秒，加在這條本來就慢的
        路徑上，而「observation 需不需要長期記憶」尚未被驗證，屬 ① 類。
        改動前請先確認收益，不要當成 bug 直接「修好」。
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

        reply = self.llm.chat(messages)

        if reply:
            self.memory.add_turn(placeholder, reply)

        return TurnResult(reply=reply)