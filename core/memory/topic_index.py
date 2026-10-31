# core/memory/topic_index.py
"""
準 v4 讀取端：主題句的向量（`topic_vec` 表）與檢索。

⭐ 衍生物：`topic_vec` 完全由 `topic.text` 算出，換 embedding 模型時舊向量自動不採用、背景重算。
⭐ FAISS：IndexFlatIP＝逐筆內積，結果與 numpy 逐筆算完全相同
   （確定性不變），只是量大時快。🔴 沒裝 faiss 時退回 numpy，**記一次 WARNING**——
   結果一樣、只差速度，所以不擋啟動。

⚠️ 執行緒分工同 embed.py：寫 `topic_vec` 在背景（判斷執行緒建完主題順手算）；
   主執行緒只讀、只編碼查詢句。

關鍵字那一路（《現況》〈🔒 隔離區〉準 v4 檢索列）：
   `topic_fts`（FTS5，rowid＝topic.id）＝主題句的 bigram
   （core/memory/bigram.py 的 `bigrams()`／`_match_expr()`，寫入與查詢必須同一個函式）。
   同樣是衍生物、可重建；由判斷執行緒建完主題順手補（`fill_fts`）。
   合併方式不在本檔：兩路各自回名單，由 manager 依各路配額取聯集（不重排）。
"""

from __future__ import annotations

import sqlite3

from core.logger import setup_logger
from core.memory.db import write_txn
from core.memory.bigram import _match_expr, bigrams
from core.memory.topics import TopicStore

logger = setup_logger("MemoryTopicIndex")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS topic_vec (
    id    INTEGER PRIMARY KEY,     -- topic.id（每個版本各一個向量）
    model TEXT NOT NULL,
    v     BLOB NOT NULL            -- float32、已正規化
);
CREATE VIRTUAL TABLE IF NOT EXISTS topic_fts USING fts5(
    tokens,                        -- bigrams(topic.text)；rowid＝topic.id
    tokenize = 'unicode61 remove_diacritics 0'
);
"""

_faiss_warned = False


def _load_faiss():
    global _faiss_warned
    try:
        import faiss  # noqa: F401
        return faiss
    except Exception as e:  # noqa: BLE001
        if not _faiss_warned:
            logger.warning(f"faiss 不可用，主題檢索改用 numpy 逐筆算（結果相同、較慢）："
                           f"{type(e).__name__}: {e}")
            _faiss_warned = True
        return None


def ensure_schema(cx: sqlite3.Connection) -> None:
    cx.executescript(_SCHEMA)


def fill_vectors(cx: sqlite3.Connection, emb, batch: int = 32) -> int:
    """補算還沒有向量的主題（背景執行緒呼叫）。回傳補了幾筆。"""
    ensure_schema(cx)
    n = 0
    while True:
        rows = cx.execute(
            "SELECT t.id, t.text FROM topic t LEFT JOIN topic_vec v ON v.id = t.id AND v.model = ? "
            "WHERE v.id IS NULL ORDER BY t.id LIMIT ?", (emb.model_name, batch)).fetchall()
        if not rows:
            return n
        m = emb.encode([r["text"] for r in rows])
        with write_txn(cx):
            cx.executemany("INSERT OR REPLACE INTO topic_vec(id, model, v) VALUES(?,?,?)",
                           [(r["id"], emb.model_name, m[i].tobytes()) for i, r in enumerate(rows)])
        n += len(rows)


def fill_fts(cx: sqlite3.Connection) -> int:
    """補主題句的關鍵字索引（背景執行緒呼叫）。回傳補了幾筆。"""
    ensure_schema(cx)
    rows = cx.execute("SELECT id, text FROM topic WHERE id > "
                      "(SELECT COALESCE(MAX(rowid), 0) FROM topic_fts) ORDER BY id").fetchall()
    if rows:
        with write_txn(cx):
            cx.executemany("INSERT INTO topic_fts(rowid, tokens) VALUES(?, ?)",
                           [(r["id"], bigrams(r["text"])) for r in rows])
    return len(rows)


class TopicSearch:
    """主執行緒用：增量讀 `topic_vec`，只在「每個 root 的最新有效版本」裡找。
    emb 可為 None（向量關閉）：此時只有 search_bm25 有作用。"""

    def __init__(self, cx: sqlite3.Connection, emb) -> None:
        self.cx = cx
        self.emb = emb
        self.store = TopicStore(cx, writable=False)
        self._ids: list[int] = []
        self._mat = None
        self._max = 0
        self._faiss = _load_faiss()
        self._index = None
        self._fts_warned = False

    def _refresh(self) -> None:
        if self.emb is None:
            return
        if self.cx.execute("SELECT 1 FROM sqlite_master WHERE name='topic_vec'").fetchone() is None:
            return
        np = self.emb._np
        rows = self.cx.execute("SELECT id, v FROM topic_vec WHERE model = ? AND id > ? ORDER BY id",
                               (self.emb.model_name, self._max)).fetchall()
        if not rows:
            return
        new = np.stack([np.frombuffer(r["v"], dtype=np.float32) for r in rows])
        self._mat = new if self._mat is None else np.vstack([self._mat, new])
        self._ids += [r["id"] for r in rows]
        self._max = self._ids[-1]
        if self._faiss is not None:
            if self._index is None:
                self._index = self._faiss.IndexFlatIP(new.shape[1])
            self._index.add(np.ascontiguousarray(new))

    def search(self, qvec, *, top_k: int, min_sim: float) -> list[tuple[dict, float]]:
        """回傳 [(topic, 餘弦)]，由高到低；只含最新版、未被取代、未退役的。"""
        self._refresh()
        if self._mat is None:
            return []
        live = {t["id"]: t for t in self.store.latest()}
        if not live:
            return []
        np = self.emb._np
        n = len(self._ids)
        if self._index is not None:
            sims, idx = self._index.search(np.asarray(qvec, dtype=np.float32)[None, :], n)
            ranked = [(int(i), float(s)) for i, s in zip(idx[0], sims[0]) if i >= 0]
        else:
            sims = self._mat @ qvec
            order = np.argsort(-sims, kind="stable")
            ranked = [(int(i), float(sims[i])) for i in order]
        out: list[tuple[dict, float]] = []
        for i, s in ranked:
            if s < min_sim or len(out) >= top_k:
                break
            t = live.get(self._ids[i])
            if t is not None:
                out.append((t, s))
        return out

    def search_bm25(self, query: str, *, top_k: int, max_terms: int) -> list[tuple[dict, float]]:
        """關鍵字那一路：回傳 [(topic, bm25 分數)]，越相關越前（分數越小越相關，FTS5 慣例）。
        只含最新版、未被取代、未退役、非待審／駁回的。索引表不存在（唯讀舊庫）⇒ 記一次 WARNING、回空。"""
        expr = _match_expr(bigrams(query), max_terms)
        if not expr or top_k <= 0:
            return []
        if self.cx.execute("SELECT 1 FROM sqlite_master WHERE name='topic_fts'").fetchone() is None:
            if not self._fts_warned:
                logger.warning("這個 DB 還沒有主題關鍵字索引（背景判斷建主題時才補；唯讀不代建），本區只走向量那一路")
                self._fts_warned = True
            return []
        live = {t["id"]: t for t in self.store.latest()}
        out: list[tuple[dict, float]] = []
        for r in self.cx.execute("SELECT rowid AS id, bm25(topic_fts) AS score FROM topic_fts "
                                 "WHERE topic_fts MATCH ? ORDER BY score, rowid", (expr,)):
            t = live.get(r["id"])
            if t is not None:
                out.append((t, float(r["score"])))
                if len(out) >= top_k:
                    break
        return out

    def pending(self) -> int:
        """還沒有向量的主題數（積壓）。向量關閉時回 0（沒有東西要補）。"""
        if self.emb is None:
            return 0
        if self.cx.execute("SELECT 1 FROM sqlite_master WHERE name='topic'").fetchone() is None:
            return 0
        if self.cx.execute("SELECT 1 FROM sqlite_master WHERE name='topic_vec'").fetchone() is None:
            return self.store.count()
        return self.cx.execute(
            "SELECT COUNT(*) c FROM topic t LEFT JOIN topic_vec v ON v.id = t.id AND v.model = ? "
            "WHERE v.id IS NULL", (self.emb.model_name,)).fetchone()["c"]
