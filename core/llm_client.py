# core/llm_client.py
# 透過 OpenAI 相容 API 連接 LM Studio 本機模型

from openai import OpenAI
from config import LLM_CONFIG
from core.text_utils import fix_traditional
from core.logger import setup_logger

logger = setup_logger("LLMClient")


class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=LLM_CONFIG["base_url"],
            api_key=LLM_CONFIG["api_key"],
        )
        self.model = LLM_CONFIG["model"]
        self.temperature = LLM_CONFIG["temperature"]
        self.max_tokens = LLM_CONFIG["max_tokens"]
        logger.info("初始化完成")

    def chat(self, messages: list) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            reply = (response.choices[0].message.content or "").strip()
            reply = fix_traditional(reply)
            logger.debug(f"回應：{reply[:50]}...")
            return reply
        except Exception as e:
            logger.error(f"LLM 呼叫失敗：{e}")
            return ""