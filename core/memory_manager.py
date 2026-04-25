# core/memory_manager.py

import json
import os
from core.llm_client import LLMClient
from core.rag_manager import RAGManager

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "memory.json")
MAX_WORKING_MEMORY = 10


class MemoryManager:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.working_memory: list[dict] = []
        self.compressed_memory: str = ""
        self.rag = RAGManager()
        self._load_memory()

    # ── 讀取 / 儲存 ──────────────────────────────

    def _load_memory(self):
        os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
        if os.path.exists(MEMORY_PATH):
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.compressed_memory = data.get("compressed_memory", "")

    def save_memory(self):
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump({"compressed_memory": self.compressed_memory}, f, ensure_ascii=False, indent=2)

    # ── 對話管理 ─────────────────────────────────

    def add_turn(self, user_input: str, assistant_reply: str):
        self.working_memory.append({"role": "user", "content": user_input})
        self.working_memory.append({"role": "assistant", "content": assistant_reply})

        if len(self.working_memory) > MAX_WORKING_MEMORY * 2:
            self._compress_oldest()

    def get_context(self, user_input: str = "") -> list[dict]:
        """
        組裝完整 context：
        1. 壓縮記憶（如果有）
        2. RAG 查詢結果（如果有相關知識）
        3. 當前對話
        """
        context = []

        # 壓縮記憶
        if self.compressed_memory:
            context.append({
                "role": "system",
                "content": f"[過去對話摘要]\n{self.compressed_memory}"
            })

        # RAG 查詢
        if user_input:
            rag_results = self.rag.query(user_input)
            if rag_results:
                rag_text = "\n".join(f"- {r}" for r in rag_results)
                context.append({
                    "role": "system",
                    "content": f"[相關記憶]\n{rag_text}"
                })

        context.extend(self.working_memory)
        return context

    # ── 記憶壓縮 ─────────────────────────────────
    def _compress(self, turns: list[dict]):
        """把指定的對話輪次壓縮成摘要並儲存"""
        dialogue_text = "\n".join(
            f"{'邪狼' if m['role'] == 'user' else '奧狐'}：{m['content']}"
            for m in turns
        )
        prompt = [{
            "role": "user",
            "content": (
                "請用繁體中文，將以下對話摘要成3到5句話，"
                "保留重要資訊（人名、事件、關係、重要決定），"
                "去掉閒聊細節。只輸出摘要，不要說明。\n\n"
                f"{dialogue_text}"
            )
        }]
        summary = self.llm.chat(prompt)
        if self.compressed_memory:
            self.compressed_memory = f"{self.compressed_memory}\n{summary}"
        else:
            self.compressed_memory = summary
        self.save_memory()
        return summary

    def _compress_oldest(self):
        to_compress = self.working_memory[:12]
        self.working_memory = self.working_memory[12:]
        summary = self._compress(to_compress)
        print(f"[Memory] 已壓縮記憶：{summary[:50]}...")

    def compress_on_exit(self):
        if not self.working_memory:
            return
        self._compress(self.working_memory)
        print("[Memory] 對話已儲存。")