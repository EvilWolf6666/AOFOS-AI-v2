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

    # ── 查詢知識 ──────────────────────────────────

    def query(self, text: str) -> list[str]:
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_texts=[text],
            n_results=min(TOP_K, count),
        )
        docs = results["documents"][0] if results["documents"] else []
        logger.debug(f"查詢：{text[:30]}，結果：{len(docs)} 筆")
        return docs

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