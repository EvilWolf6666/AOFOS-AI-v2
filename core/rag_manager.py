# core/rag_manager.py
# RAG 長期記憶 — 用 ChromaDB 儲存/查詢具體知識

import json
import os
import uuid
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from config import TEST_CONFIG, TEST_MODE
from core.logger import setup_logger, timed

logger = setup_logger("RAGManager")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "aofos_memory"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.7  # 低於這個分數的結果直接丟棄

# Write/Search 計數器：暫時性診斷用途，追蹤淘汰內容轉存 RAG 後是否查得到。
# 問題釐清後（見 AOFOS-決策脈絡歸檔.md 六之10）整段直接刪除，不保留。
#
# 【2026-07-29 修正】原本只有 writes / hits 兩個欄位，缺少「查詢總次數」，
# 因此算不出命中率——而命中率正是這個計數器存在的目的。
# 另外原本 hits 是在每筆命中文件時累加、且每次都寫檔（單次查詢最多寫 3 次），
# 語意上混淆了「命中幾次查詢」與「回傳幾筆文件」。現拆成三個欄位。
STATS_FILE = Path(os.path.join(os.path.dirname(__file__), "..", "data", "rag_stats.json"))

_DEFAULT_STATS = {
    "writes": 0,             # 寫入次數
    "searches": 0,           # 查詢總次數
    "searches_with_hit": 0,  # 至少命中一筆的查詢次數
    "docs_returned": 0,      # 累計回傳文件數（通過門檻的）
}


def _load_stats() -> dict:
    stats = dict(_DEFAULT_STATS)
    if STATS_FILE.exists():
        try:
            stats.update(json.loads(STATS_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return stats


def _save_stats(stats: dict):
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class RAGManager:
    def __init__(self):
        # 後端採惰性初始化：ChromaDB 客戶端與 embedding 模型載入要好幾秒，
        # 測試模式下完全用不到，沒必要付這個成本。
        self.client = None
        self.collection = None
        self.embedding_fn = None
        self._stats = dict(_DEFAULT_STATS)

        self.enabled = not (TEST_MODE and TEST_CONFIG.get("disable_rag", True))
        if self.enabled:
            self._init_backend()
        else:
            logger.warning("TEST_MODE 啟用：RAG 已停用（不查詢、不寫入、不初始化）")

    def _init_backend(self):
        """實際建立 ChromaDB 連線與 embedding 函式。重複呼叫安全（已初始化則跳過）。"""
        if self.collection is not None:
            return

        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            # cosine 距離空間：query() 的 `similarity = 1 - distance` 只在此空間下成立
            # （ChromaDB 預設 l2，換算會失真）。此設定在 collection 建立當下即固定，
            # 事後無法修改，換空間需刪除整個 data/chroma_db 資料夾重建。
            metadata={"hnsw:space": "cosine"},
        )
        self._stats = _load_stats()
        # 初始化時就把統計檔寫出來。
        # 原本只在 add() 或非空查詢時才寫檔，導致「檔案不存在」這個狀態同時可能代表
        # 「沒有任何淘汰內容需要寫入」或「寫入路徑壞掉」——兩者無法區分，
        # 而區分這兩者正是這個計數器存在的目的。
        _save_stats(self._stats)
        logger.info(f"RAG 初始化完成（目前 {self.collection.count()} 筆）。")

    def set_enabled(self, enabled: bool):
        """供 /test 指令在執行期切換。關閉時不銷毀既有連線，只是不再查詢/寫入。"""
        if enabled == self.enabled:
            return
        self.enabled = enabled
        if enabled:
            self._init_backend()
            logger.info("RAG 已啟用。")
        else:
            logger.info("RAG 已停用（查詢一律回傳空、寫入一律略過）。")

    # ── 新增知識 ──────────────────────────────────

    def add(self, text: str, doc_id: str = None, metadata: dict = None):
        if not self.enabled:
            logger.debug("TEST_MODE：略過 RAG 寫入。")
            return

        if doc_id is None:
            doc_id = str(uuid.uuid4())

        upsert_kwargs = {
            "documents": [text],
            "ids": [doc_id],
        }
        if metadata:
            upsert_kwargs["metadatas"] = [metadata]

        with timed("RAG寫入"):
            self.collection.upsert(**upsert_kwargs)

        logger.debug(f"新增知識：{text[:40]}...")

        self._stats["writes"] += 1
        _save_stats(self._stats)

    # ── 查詢知識（含相似度門檻）─────────────────

    def query(self, text: str) -> list[str]:
        """查詢最相關的知識，只回傳相似度超過 SIMILARITY_THRESHOLD 的結果。"""
        if not self.enabled:
            return []

        count = self.collection.count()
        if count == 0:
            # 空集合不計入查詢統計，否則命中率會被大量必然落空的查詢稀釋，
            # 使這份診斷資料失去意義。
            return []

        with timed("RAG查詢"):
            results = self.collection.query(
                query_texts=[text],
                n_results=min(TOP_K, count),
                include=["documents", "distances"],
            )

        docs = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []

        filtered = []
        for doc, distance in zip(docs, distances):
            similarity = 1 - distance
            if similarity >= SIMILARITY_THRESHOLD:
                filtered.append(doc)
                logger.debug(f"RAG 命中（相似度 {similarity:.2f}）：{doc[:30]}...")
            else:
                logger.debug(f"RAG 過濾（相似度 {similarity:.2f}）：{doc[:30]}...")

        # 統計每次查詢只更新一次
        self._stats["searches"] += 1
        if filtered:
            self._stats["searches_with_hit"] += 1
            self._stats["docs_returned"] += len(filtered)
        _save_stats(self._stats)

        logger.debug(f"查詢：{text[:30]}，命中：{len(filtered)}/{len(docs)} 筆")
        return filtered

    # ── 刪除知識 ──────────────────────────────────

    def delete(self, doc_id: str):
        if not self.enabled:
            logger.debug("TEST_MODE：略過 RAG 刪除。")
            return
        self.collection.delete(ids=[doc_id])
        logger.info(f"已刪除知識：{doc_id}")

    # ── 列出所有知識 ──────────────────────────────

    def list_all(self) -> list[dict]:
        if not self.enabled:
            return []
        results = self.collection.get()
        output = []
        for i, doc_id in enumerate(results["ids"]):
            output.append({
                "id": doc_id,
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
            })
        return output