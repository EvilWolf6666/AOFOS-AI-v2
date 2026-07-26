# core/rag_manager.py
# RAG 長期記憶 — 用 ChromaDB 儲存/查詢具體知識

import os
import json
import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions
from core.logger import setup_logger

logger = setup_logger("RAGManager")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "aofos_memory"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.7  # 低於這個分數的結果直接丟棄

# Write/Search 計數器：暫時性診斷用途，追蹤淘汰內容轉存 RAG 後是否查得到。
# 問題釐清後（見 AOFOS-決策脈絡歸檔.md 六之10）整段直接刪除，不保留。
STATS_FILE = Path(os.path.join(os.path.dirname(__file__), "..", "data", "rag_stats.json"))


def _load_stats() -> dict:
    if STATS_FILE.exists():
        try:
            return json.loads(STATS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"writes": 0, "hits": 0}


def _save_stats(stats: dict):
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class RAGManager:
    def __init__(self):
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
        logger.info("RAG 初始化完成。")

    # ── 新增知識 ──────────────────────────────────

    def add(self, text: str, doc_id: str = None, metadata: dict = None):
        if doc_id is None:
            import uuid
            doc_id = str(uuid.uuid4())

        upsert_kwargs = {
            "documents": [text],
            "ids": [doc_id],
        }
        if metadata:
            upsert_kwargs["metadatas"] = [metadata]

        self.collection.upsert(**upsert_kwargs)
        logger.debug(f"新增知識：{text[:40]}...")

        self._stats["writes"] += 1
        _save_stats(self._stats)

    # ── 查詢知識（含相似度門檻）─────────────────

    def query(self, text: str) -> list[str]:
        """查詢最相關的知識，只回傳相似度超過 SIMILARITY_THRESHOLD 的結果。"""
        count = self.collection.count()
        if count == 0:
            return []

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
                self._stats["hits"] += 1
                _save_stats(self._stats)
            else:
                logger.debug(f"RAG 過濾（相似度 {similarity:.2f}）：{doc[:30]}...")

        logger.debug(f"查詢：{text[:30]}，命中：{len(filtered)}/{len(docs)} 筆")
        return filtered

    # ── 刪除知識 ──────────────────────────────────

    def delete(self, doc_id: str):
        self.collection.delete(ids=[doc_id])
        logger.info(f"已刪除知識：{doc_id}")

    # ── 列出所有知識 ──────────────────────────────

    def list_all(self) -> list[dict]:
        results = self.collection.get()
        output = []
        for i, doc_id in enumerate(results["ids"]):
            output.append({
                "id": doc_id,
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
            })
        return output