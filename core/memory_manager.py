# core/memory_manager.py
# 記憶系統：Working Memory（近期對話）+ Compressed Memory（摘要）+ RAG（長期知識）

import json
import os
from core.llm_client import LLMClient
from core.rag_manager import RAGManager
from core.logger import setup_logger

logger = setup_logger("MemoryManager")

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "memory.json")
MAX_WORKING_MEMORY = 10
MAX_SUMMARY_SENTENCES = 8  # Compressed Memory 上限句數，避免無限累積


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
            logger.info("記憶載入完成。")
        else:
            logger.info("無歷史記憶，從零開始。")

    def save_memory(self):
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump({"compressed_memory": self.compressed_memory}, f, ensure_ascii=False, indent=2)
        logger.debug("記憶已存檔。")

    # ── 對話管理 ─────────────────────────────────

    def add_turn(self, user_input: str, assistant_reply: str):
        self.working_memory.append({"role": "user", "content": user_input})
        self.working_memory.append({"role": "assistant", "content": assistant_reply})

        if len(self.working_memory) > MAX_WORKING_MEMORY * 2:
            self._compress_oldest()

    def get_context(self, user_input: str = "") -> list[dict]:
        """組裝 context：Compressed Memory（若有）→ RAG 查詢結果（若有相關）→ 當前對話。"""
        context = []

        if self.compressed_memory:
            context.append({
                "role": "system",
                "content": f"[過去對話摘要]\n{self.compressed_memory}"
            })

        if user_input:
            rag_results = self.rag.query(user_input)
            if rag_results:
                rag_text = "\n".join(f"- {r}" for r in rag_results)
                context.append({
                    "role": "system",
                    "content": f"[相關記憶]\n{rag_text}"
                })
                logger.debug(f"注入 {len(rag_results)} 筆 RAG 記憶。")
            else:
                logger.debug("無相關 RAG 記憶，跳過注入。")

        context.extend(self.working_memory)
        return context

    # ── 記憶壓縮 ─────────────────────────────────

    def _compress(self, turns: list[dict]):
        """
        將指定對話輪次與既有摘要整合覆寫（非疊加），長度受 MAX_SUMMARY_SENTENCES 限制。
        舊摘要中沒出現在新摘要裡的內容（不論是被取代還是排不進句數上限），
        一律轉存 RAG，不區分原因——避免 LLM 誤以為「沒被提到」等於可以直接省略。
        """
        dialogue_text = "\n".join(
            f"{'邪狼' if m['role'] == 'user' else '奧狐'}：{m['content']}"
            for m in turns
        )

        existing_block = f"[之前的摘要]\n{self.compressed_memory}\n\n" if self.compressed_memory else ""

        prompt = [{
            "role": "user",
            "content": (
                "你要幫忙整理「奧狐」的近期記憶摘要。\n\n"
                f"{existing_block}[新對話]\n{dialogue_text}\n\n"
                "請完成兩件事，並嚴格遵守以下規則：\n"
                "規則：「之前的摘要」中的每一條資訊，預設都要「保留」在新摘要裡，"
                "除非新對話明確更新、推翻、或讓它過時（例如某件事後來解決了、狀態變了），"
                "或受限於新摘要的句數上限而排不進去。"
                "新對話沒有提到的舊資訊，不代表它不重要，原則上不可以省略，"
                "只有在真的擠不進新摘要時才可以不保留。\n\n"
                f"1. 輸出整合後的新摘要，不超過 {MAX_SUMMARY_SENTENCES} 句話，"
                "包含：未被取代的舊資訊（照規則保留）+ 新對話中重要的新資訊。"
                "若空間不足，優先保留人名、關係、近況、重要決定，次要內容可精簡或不放入新摘要。"
                "每一句話結束後換行，一行一句，不要把多句擠在同一行。\n"
                "2. 列出「之前的摘要」中，沒有出現在新摘要裡的所有舊資訊"
                "（無論原因是被新對話取代、還是因為句數上限排不進去，只要新摘要裡沒有，就列出來；"
                "若新摘要完整保留了之前摘要的所有內容，就寫「無」）。\n\n"
                "請只用以下格式輸出，不要其他說明文字：\n"
                "摘要：<新摘要內容>\n"
                "淘汰：<沒出現在新摘要裡的舊資訊，或「無」>"
            )
        }]

        result = self.llm.chat(prompt)
        new_summary, discarded = self._parse_compress_result(result)

        self.compressed_memory = new_summary
        self.save_memory()

        if discarded and discarded != "無":
            try:
                self.rag.add(discarded)
                logger.debug(f"舊記憶已轉存 RAG：{discarded[:50]}...")
            except Exception as e:
                logger.warning(f"舊記憶轉存 RAG 失敗（不影響主流程）：{e}")

        return new_summary

    def _parse_compress_result(self, result: str) -> tuple[str, str]:
        """解析 LLM 回傳的『摘要 / 淘汰』結果。"""
        try:
            lines = result.strip().splitlines()
            summary_line_idx = next(
                i for i, l in enumerate(lines) if l.strip().startswith("摘要：")
            )
            discard_line_idx = next(
                (i for i, l in enumerate(lines) if l.strip().startswith("淘汰：")),
                None,
            )

            if discard_line_idx is not None:
                summary = "\n".join(lines[summary_line_idx:discard_line_idx]).replace("摘要：", "", 1).strip()
                discarded = "\n".join(lines[discard_line_idx:]).replace("淘汰：", "", 1).strip()
            else:
                summary = "\n".join(lines[summary_line_idx:]).replace("摘要：", "", 1).strip()
                discarded = "無"

            return summary, discarded
        except Exception as e:
            logger.warning(f"壓縮結果解析失敗：{e}，原始結果：{result}")
            # 保底：解析失敗時直接用原始輸出當摘要（避免記憶遺失），不寫入 RAG（避免存入格式錯亂內容）
            return result.strip(), "無"

    def _compress_oldest(self):
        to_compress = self.working_memory[:12]
        self.working_memory = self.working_memory[12:]
        summary = self._compress(to_compress)
        logger.info(f"已壓縮記憶：{summary[:50]}...")

    def compress_on_exit(self):
        if not self.working_memory:
            return
        self._compress(self.working_memory)
        logger.info("對話已儲存。")