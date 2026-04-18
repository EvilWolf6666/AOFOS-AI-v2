# core/llm_client.py

from openai import OpenAI
from config import LLM_CONFIG
from core.text_utils import fix_traditional


class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=LLM_CONFIG["base_url"],
            api_key=LLM_CONFIG["api_key"],
        )
        self.model = LLM_CONFIG["model"]
        self.temperature = LLM_CONFIG["temperature"]
        self.max_tokens = LLM_CONFIG["max_tokens"]

    def chat(self, messages: list[dict]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            reply = response.choices[0].message.content.strip()
            return fix_traditional(reply)  # ← 後處理在這裡
        except Exception as e:
            print(f"[LLMClient ERROR] {e}")
            return ""