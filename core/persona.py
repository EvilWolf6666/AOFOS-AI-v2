# core/persona.py
# 載入固定人格 prompt，組裝送給 LLM 的完整 messages

import os

from core.logger import setup_logger

logger = setup_logger("Persona")

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "aofos_persona.txt")

# 此標記行以下的內容不送進模型。
#
# 存在理由：人格檔同時是「給模型的指令」與「給人看的維護備忘」，但模型讀得到
# 備忘段就會受其影響（例如備忘寫著「這是精簡測試版」，模型便知道自己是測試版本）。
#
# ⚠️ 標記字串刻意選用不會出現在正文的形式。改動此字串必須同步改 persona.txt，
#    否則剝除靜默失效——故未命中時一律發 WARNING，不靜默通過。
STRIP_MARKER = "[[NOT_SENT_TO_MODEL]]"


class Persona:
    def __init__(self, send_system: bool = True):
        """send_system=False：不送人格 system（只給 tools/eval_runner --no-persona 對照組用）。"""
        self.system_prompt = self._load_prompt()
        self.send_system = send_system

    def _load_prompt(self) -> str:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            raw = f.read()

        if STRIP_MARKER in raw:
            kept = raw.split(STRIP_MARKER, 1)[0].strip()
            logger.info(
                f"人格 prompt 已剝除維護備忘段：送出 {len(kept)} 字，"
                f"剝除 {len(raw.strip()) - len(kept)} 字"
            )
            return kept

        # 沒有標記時整份送出是正確行為（人格檔可以不含備忘段），但必須看得見，
        # 否則「標記被改壞」與「本來就沒有備忘段」兩種情況無法區分
        logger.warning(
            f"人格 prompt 未發現剝除標記 {STRIP_MARKER}，整份送出（{len(raw.strip())} 字）"
        )
        return raw.strip()

    def build_messages(self, conversation_history: list[dict], user_input: str) -> list[dict]:
        """順序固定：人格 system → 記憶 context → 本輪輸入。"""
        messages = [{"role": "system", "content": self.system_prompt}] if self.send_system else []
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_input})
        return messages