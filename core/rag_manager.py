# core/rag_manager.py
# RAG 長期記憶 — 用 ChromaDB 儲存/查詢具體知識

import os
import chromadb
from chromadb.utils import embedding_functions

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "aofos_memory"
TOP_K = 3  # 每次查詢回傳最相關的幾筆


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

    # ── 新增知識 ──────────────────────────────────

    def add(self, text: str, doc_id: str = None, metadata: dict = None):
        """
        新增一筆知識到資料庫。
        text     : 要存的內容
        doc_id   : 唯一ID，不填會自動生成
        metadata : 額外標記，例如 {"category": "喜好"}
        """
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
        
    # ── 查詢知識 ──────────────────────────────────

    def query(self, text: str) -> list[str]:
        """
        用自然語言查詢最相關的知識，回傳字串列表。
        如果資料庫是空的，回傳空列表。
        """
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_texts=[text],
            n_results=min(TOP_K, count),
        )
        return results["documents"][0] if results["documents"] else []

    # ── 刪除知識 ──────────────────────────────────

    def delete(self, doc_id: str):
        """刪除指定 ID 的知識"""
        self.collection.delete(ids=[doc_id])

    # ── 列出所有知識 ──────────────────────────────

    def list_all(self) -> list[dict]:
        """列出所有已儲存的知識，回傳 [{id, text, metadata}] 列表"""
        results = self.collection.get()
        output = []
        for i, doc_id in enumerate(results["ids"]):
            output.append({
                "id": doc_id,
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
            })
        return output