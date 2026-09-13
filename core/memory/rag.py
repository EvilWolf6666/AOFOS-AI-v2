# core/memory/rag.py
# RAG 長期記憶 — 用 ChromaDB 儲存/查詢具體知識

import os
import uuid

from config import (PROJECT_ROOT, RAG_INCLUDE_SEED, RAG_SIMILARITY_THRESHOLD,
                    RAG_TOP_K, TEST_CONFIG, TEST_MODE)

# ⚠️ chromadb 刻意不在此 import，見 _init_backend()。
from core.logger import setup_logger, timed

logger = setup_logger("RAGManager")

# ⚠️ 專案根一律取自 config.PROJECT_ROOT，不得在此用 __file__ + ".." 推算——
#    搬檔時它會靜默指到別的位置，然後 get_or_create_collection 建一個空庫。
DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")
COLLECTION_NAME = "aofos_memory"

# 知識來源。寫入 metadata，供 tools/seed_rag.py 的整份同步辨識哪些是種子
# （檔案裡刪掉的段落要一併從庫裡移除，而淘汰內容不能被那個動作波及）。
SOURCE_EVICT = "compress_evict"   # _compress 整合覆寫時被擠出摘要、轉存進來的
SOURCE_SEED = "seed"              # 人工撰寫的世界觀，由 tools/seed_rag.py 灌入

# 退場機制。RAG 是唯一沒有下一層可退的記憶層——Working Memory 會滾掉、
# Compressed Memory 靠 _compress 整合覆寫，而 RAG 進來就永遠在。過期的知識
# （例如「她看不到畫面」在 VL 接上之後）留在庫裡會與新事實一起被召回，變成
# 矛盾注入，且沒有任何訊號提示該刪它。見《歸檔》K0001、E0005。
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
        原本卻只做到一半。見《歸檔》I0028。

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
        seed_state = "含種子" if RAG_INCLUDE_SEED else "**種子已隔離停用**"
        logger.info(f"RAG 初始化完成（目前 {self.collection.count()} 筆，檢索範圍：{seed_state}）。")

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
        被注入模型，所以答案另存 metadata。見《歸檔》E0004。

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

    # ── 查詢知識（含相似度門檻）─────────────────

    @staticmethod
    def _build_where() -> dict:
        """檢索階段的 metadata 過濾條件。單一組裝點，不在呼叫端各自拼。"""
        if RAG_INCLUDE_SEED:
            return {RETIRED_FIELD: False}
        return {"$and": [{RETIRED_FIELD: False}, {"source": {"$ne": SOURCE_SEED}}]}


    def query(self, text: str) -> list[str]:
        """查詢最相關的知識，回傳實際要注入的內容（有 answer 者回 answer）。

        ⚠️ RAG_SIMILARITY_THRESHOLD 的職責是**地板不是閘門**。實測 recall@3 與
        MRR 都是滿分——排序本身已經正確，門檻的工作只剩「擋掉整個查詢與知識庫
        無關的情況」（例如問天氣）。當成閘門用會砍掉已經排對的結果，這正是
        0.7 造成 0/15 的原因。見《歸檔》E0004。

        ⚠️ 依 answer 去重：一個答案可掛多個問句，不去重的話三個同義問句會佔滿
        TOP_K，實際只回到一則知識，recall 反而下降。故多取候選再去重。
        """
        if not self.enabled:
            return []

        count = self.collection.count()
        if count == 0:
            # 下方 n_results 取 min(TOP_K*3, count)，空集合會算出 0 而 ChromaDB
            # 不接受 n_results=0
            return []

        with timed("RAG查詢"):
            results = self.collection.query(
                query_texts=[text],
                # 多取候選：同一答案可能掛多個問句，去重後才湊得滿 TOP_K
                n_results=min(RAG_TOP_K * 3, count),
                include=["documents", "distances", "metadatas"],
                # ⚠️ 過濾一律下在檢索階段，不是撈回來再篩。兩個理由不同但同樣硬：
                #    已下線的知識一旦進了上下文，就很難保證不以某種形式洩漏進回答；
                #    而種子若撈回來再篩，n_results 的名額會先被它佔滿，
                #    過濾完剩不到 TOP_K——那是「看起來有過濾、實際砍掉召回」。
                # ⚠️ 種子條件寫 `$ne SEED` 而非 `$eq EVICT`：缺 source 欄位的舊資料
                #    依慣例視為淘汰內容，用後者會把它們一併漏掉。
                where=self._build_where(),
            )

        docs = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metas = results["metadatas"][0] if results.get("metadatas") else [None] * len(docs)

        filtered = []
        seen = set()
        for doc, distance, meta in zip(docs, distances, metas):
            similarity = 1 - distance
            if similarity < RAG_SIMILARITY_THRESHOLD:
                # ⚠️ 這裡印的是**索引文本**（多為問句），命中那行印的是 answer。
                #    兩者是同一筆知識的兩面，不標明會被誤讀成庫裡塞滿問句條目。
                logger.debug(f"RAG 過濾（相似度 {similarity:.2f}，索引文本）：{doc[:30]}...")
                continue

            meta = meta or {}
            # 沒有 answer 的是 _compress 淘汰內容，索引文本本身就是要注入的內容
            payload = meta.get("answer") or doc
            if payload in seen:
                continue
            seen.add(payload)
            filtered.append(payload)
            logger.debug(f"RAG 命中（相似度 {similarity:.2f}）：{payload[:30]}...")

            if len(filtered) >= RAG_TOP_K:
                break

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