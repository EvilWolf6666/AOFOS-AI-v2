# core/persona.py
# 載入固定人格 prompt，組裝送給 LLM 的完整 messages

import os

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "aofos_persona.txt")


class Persona:
    def __init__(self):
        self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()

    def build_messages(self, conversation_history: list[dict], user_input: str) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_input})
        return messages