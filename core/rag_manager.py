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
SIMILARITY_THRESHOLD = 0.7  # 低於這個分數的結果直接丟掉

# --- Write/Search 最簡計數器（2026-07-24 新增，暫時性診斷用途） ---
# 用途：驗證「淘汰內容轉存RAG後是否測得到」的方向性訊號（AutoMem Write/Search比例概念）
# ⚠️ 移除條件：見 AOFOS-決策脈絡歸檔.md 六之10。
#    RAG 資料量增長到有意義規模、且透過 writes/hits 比例確認「淘汰內容查詢命中率」
#    這個問題是否存在之後，不論結論為何，這整段（含下方 add()/query() 裡呼叫的三行）
#    都應該直接刪除，不比照 behavior_scorer.py 移到 archive 保留。
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
# --- 計數器區塊結束 ---


class RAGManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            # 【修正,2026-07-23】明確指定 cosine 距離空間。
            # ChromaDB 預設是 l2（歐氏距離平方），此時 query() 裡的
            # `similarity = 1 - distance` 換算並不成立，會讓 SIMILARITY_THRESHOLD
            # 篩選失真。只有在 cosine 空間下，Chroma 回傳的 distance 才等於
            # `1 - cosine_similarity`，1 - distance 才是真正的相似度。
            # 注意：HNSW 索引空間在 collection 建立當下即固定，事後無法修改，
            # 若舊的 data/chroma_db 是用 l2 建的，需要刪掉整個資料夾讓它用這個設定重建。
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

        # Write/Search 計數器：寫入 +1（暫時性診斷用途，見上方區塊註解）
        self._stats["writes"] += 1
        _save_stats(self._stats)

    # ── 查詢知識（含相似度門檻）─────────────────

    def query(self, text: str) -> list[str]:
        """
        查詢最相關的知識，只回傳相似度超過門檻的結果。
        ChromaDB 在 cosine 距離空間下回傳的 distance = 1 - cosine_similarity，
        故 similarity = 1 - distance 在此設定下成立（需搭配 __init__ 的
        metadata={"hnsw:space": "cosine"} 一起看，兩處是同一個假設的兩端）。
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
                # Write/Search 計數器：命中且真正被使用 +1（暫時性診斷用途，見檔案頂部區塊註解）
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