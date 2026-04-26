# core/rag_manager.py
# RAG 長期記憶 — 用 ChromaDB 儲存/查詢具體知識

import os
import chromadb
from chromadb.utils import embedding_functions
from core.logger import setup_logger

logger = setup_logger("RAGManager")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "aofos_memory"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.7  # 低於這個分數的結果直接丟掉


class RAGManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )
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

    # ── 查詢知識（含相似度門檻）─────────────────

    def query(self, text: str) -> list[str]:
        """
        查詢最相關的知識，只回傳相似度超過門檻的結果。
        ChromaDB 回傳的 distance 是越小越相似（0 = 完全相同）。
        轉換成相似度：similarity = 1 - distance
        """
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

        # 過濾低相似度結果
        filtered = []
        for doc, distance in zip(docs, distances):
            similarity = 1 - distance
            if similarity >= SIMILARITY_THRESHOLD:
                filtered.append(doc)
                logger.debug(f"RAG 命中（相似度 {similarity:.2f}）：{doc[:30]}...")
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