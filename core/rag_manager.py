# core/rag_manager.py
# RAG 長期記憶 — 用 ChromaDB 儲存/查詢具體知識

import os
import uuid

from config import RAG_SIMILARITY_THRESHOLD, RAG_TOP_K, TEST_CONFIG, TEST_MODE

# ⚠️ chromadb 刻意不在此 import，見 _init_backend()。
from core.short_time_test import RagStats   # 診斷腳手架，清理時連同下方呼叫點一併移除
from core.logger import setup_logger, timed

logger = setup_logger("RAGManager")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "aofos_memory"

# 知識來源。寫入 metadata，供查詢端區分統計；缺這個欄位的資料視為 SOURCE_EVICT。
SOURCE_EVICT = "compress_evict"   # _compress 整合覆寫時被擠出摘要、轉存進來的
SOURCE_SEED = "seed"              # 人工撰寫的世界觀，由 tools/seed_rag.py 灌入

# 退場機制。RAG 是唯一沒有下一層可退的記憶層——Working Memory 會滾掉、
# Compressed Memory 靠 _compress 整合覆寫，而 RAG 進來就永遠在。過期的知識
# （例如「她看不到畫面」在 VL 接上之後）留在庫裡會與新事實一起被召回，變成
# 矛盾注入，且沒有任何訊號提示該刪它。見《歸檔》二十九之3、三十之2。
#
# ⚠️ 只做「寫入欄位 + 查詢過濾」這一半。下線的 setter、觸發條件與介面刻意不做：
#    欄位已經在每一列上，之後補寫 setter 不需要資料遷移；而現在做要猜「什麼時候
#    算過期」的規格，猜錯的成本是重做。
#
# ⚠️ 不用「失效時間戳」是因為時點形狀不對——過期是由能力變化觸發（VL 接上那天），
#    不是由日期觸發。布林值猜不錯，時間戳會猜錯。
RETIRED_FIELD = "retired"


class RAGManager:
    def __init__(self):
        # 後端採惰性初始化：ChromaDB 客戶端與 embedding 模型載入要好幾秒，
        # 測試模式下完全用不到，沒必要付這個成本。
        self.client = None
        self.collection = None
        self.embedding_fn = None
        self._stats = RagStats()

        self.enabled = not (TEST_MODE and TEST_CONFIG.get("disable_rag", True))
        if self.enabled:
            self._init_backend()
        else:
            logger.warning("TEST_MODE 啟用：RAG 已停用（不查詢、不寫入、不初始化）")

    def _init_backend(self):
        """建立 ChromaDB 連線與 embedding 函式。重複呼叫安全（已初始化則跳過）。

        ⚠️ chromadb 在此 import 而非模組層級，這是刻意的：TEST_MODE 下本方法
        永遠不會被呼叫，而 `import chromadb` 實測 2.8～7.4 秒（雜訊大，只能看
        量級）。放在模組層級的話，即使 RAG 全程停用也照付這筆錢——那正是
        本檔檔頭「測試模式下完全用不到，沒必要付這個成本」想避免的事，
        原本卻只做到一半。見《決策脈絡歸檔》三十二之2。

        ⚠️ 這裡的 import 必須排在 config 被載入之後才有意義（config 頂部設定
        HF_HUB_OFFLINE，而 huggingface_hub 在 import 當下把它讀成模組常數）。
        目前結構天然滿足：本方法只可能在 __init__ 或 set_enabled() 中被呼叫，
        屆時模組層級的 `from config import ...` 早已執行完畢。

        ⚠️ 真正吃掉十幾秒的 sentence-transformers 不在這裡，是 ChromaDB 於
        SentenceTransformerEmbeddingFunction.__init__ 內部才載入的——那一層
        原本就已經被本方法的惰性保護涵蓋，不需另外處理。
        """
        if self.collection is not None:
            return

        import chromadb
        from chromadb.utils import embedding_functions

        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            # ⚠️ cosine 距離空間：query() 的 `similarity = 1 - distance` 只在此空間下
            # 成立（ChromaDB 預設 l2，換算會失真）。此設定在 collection 建立當下即
            # 固定，事後無法修改，換空間需刪除整個 data/chroma_db 資料夾重建。
            metadata={"hnsw:space": "cosine"},
        )
        self._stats.load()
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

    def add(self, text: str, doc_id: str = None, metadata: dict = None,
            source: str = SOURCE_EVICT, answer: str = None):
        """寫入一筆知識。

        `text` 是拿去比對的索引文本，`answer` 是實際要注入模型的內容。
        兩者分開的理由：使用者的查詢永遠是問句，索引文本若也是問句，兩邊
        同分佈，相似度的絕對值才有意義（實測 0.437 → 0.760）。但問句不該
        被注入模型，所以答案另存 metadata。見《歸檔》三十之1。

        `answer` 省略時（`_compress` 淘汰內容即是）查詢端會退回傳 `text` 本身，
        故淘汰路徑的呼叫端一行都不用改。
        """
        if not self.enabled:
            logger.debug("TEST_MODE：略過 RAG 寫入。")
            return

        if doc_id is None:
            doc_id = str(uuid.uuid4())

        meta = dict(metadata) if metadata else {}
        meta["source"] = source
        meta[RETIRED_FIELD] = False
        if answer:
            meta["answer"] = answer

        with timed("RAG寫入"):
            self.collection.upsert(
                documents=[text],
                ids=[doc_id],
                metadatas=[meta],
            )

        logger.debug(f"新增知識（{source}）：{text[:40]}...")

        # 只計入 SOURCE_EVICT。種子資料是人工照「會被怎麼問」的措辭寫的，命中率
        # 天然偏高；混進來會稀釋掉這個計數器唯一要量的東西——摘要式書面語 vs
        # 口語提問的措辭落差是否讓淘汰內容命不中門檻。
        if source == SOURCE_EVICT:
            self._stats.record_write()

    # ── 查詢知識（含相似度門檻）─────────────────

    def query(self, text: str) -> list[str]:
        """查詢最相關的知識，回傳實際要注入的內容（有 answer 者回 answer）。

        ⚠️ RAG_SIMILARITY_THRESHOLD 的職責是**地板不是閘門**。實測 recall@3 與
        MRR 都是滿分——排序本身已經正確，門檻的工作只剩「擋掉整個查詢與知識庫
        無關的情況」（例如問天氣）。當成閘門用會砍掉已經排對的結果，這正是
        0.7 造成 0/15 的原因。見《歸檔》三十之1。

        ⚠️ 依 answer 去重：一個答案可掛多個問句，不去重的話三個同義問句會佔滿
        TOP_K，實際只回到一則知識，recall 反而下降。故多取候選再去重。
        """
        if not self.enabled:
            return []

        count = self.collection.count()
        if count == 0:
            # 空集合不計入查詢統計，否則命中率會被大量必然落空的查詢稀釋
            return []

        with timed("RAG查詢"):
            results = self.collection.query(
                query_texts=[text],
                # 多取候選：同一答案可能掛多個問句，去重後才湊得滿 TOP_K
                n_results=min(RAG_TOP_K * 3, count),
                include=["documents", "distances", "metadatas"],
                # 已下線的知識在檢索階段就排除，不是撈回來再過濾——一旦進了
                # 上下文就很難保證不以某種形式洩漏進回答裡
                where={RETIRED_FIELD: False},
            )

        docs = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metas = results["metadatas"][0] if results.get("metadatas") else [None] * len(docs)

        filtered = []
        seen = set()
        evict_hits = 0
        for doc, distance, meta in zip(docs, distances, metas):
            similarity = 1 - distance
            if similarity < RAG_SIMILARITY_THRESHOLD:
                logger.debug(f"RAG 過濾（相似度 {similarity:.2f}）：{doc[:30]}...")
                continue

            meta = meta or {}
            # 沒有 answer 的是 _compress 淘汰內容，索引文本本身就是要注入的內容
            payload = meta.get("answer") or doc
            if payload in seen:
                continue
            seen.add(payload)
            filtered.append(payload)

            # 無 source 欄位者視為淘汰內容：該欄位是後加的，早於它寫入的
            # 資料只可能來自 _compress。
            if meta.get("source", SOURCE_EVICT) == SOURCE_EVICT:
                evict_hits += 1
            logger.debug(f"RAG 命中（相似度 {similarity:.2f}）：{payload[:30]}...")

            if len(filtered) >= RAG_TOP_K:
                break

        # ⚠️ 判讀限制：分母是「所有非空查詢」，包含當下根本沒有相關淘汰內容可命中
        # 的查詢，因此比例會系統性偏低。要看的是趨勢與 writes 的相對關係，不是絕對值。
        self._stats.record_search(evict_hits)

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