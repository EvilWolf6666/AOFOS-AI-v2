# core/memory/topic_index.py
"""
準 v4 讀取端：主題句的向量（`topic_vec` 表）與檢索。

⭐ 衍生物：`topic_vec` 完全由 `topic.text` 算出，換 embedding 模型時舊向量自動不採用、背景重算。
⭐ FAISS 保留（邪狼 2026-09-30 裁定）：IndexFlatIP＝逐筆內積，結果與 numpy 逐筆算完全相同
   （確定性不變），只是量大時快。🔴 沒裝 faiss 時退回 numpy，**記一次 WARNING**——
   結果一樣、只差速度，所以不擋啟動。

⚠️ 執行緒分工同 embed.py：寫 `topic_vec` 在背景（判斷執行緒建完主題順手算）；
   主執行緒只讀、只編碼查詢句。
"""

from __future__ import annotations

import sqlite3

from core.logger import setup_logger
from core.memory.db import write_txn
from core.memory.topics import TopicStore

logger = setup_logger("MemoryTopicIndex")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS topic_vec (
    id    INTEGER PRIMARY KEY,     -- topic.id（每個版本各一個向量）
    model TEXT NOT NULL,
    v     BLOB NOT NULL            -- float32、已正規化
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


class TopicSearch:
    """主執行緒用：增量讀 `topic_vec`，只在「每個 root 的最新有效版本」裡找。"""

    def __init__(self, cx: sqlite3.Connection, emb) -> None:
        self.cx = cx
        self.emb = emb
        self.store = TopicStore(cx, writable=False)
        self._ids: list[int] = []
        self._mat = None
        self._max = 0
        self._faiss = _load_faiss()
        self._index = None

    def _refresh(self) -> None:
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

    def pending(self) -> int:
        """還沒有向量的主題數（積壓）。"""
        if self.cx.execute("SELECT 1 FROM sqlite_master WHERE name='topic'").fetchone() is None:
            return 0
        if self.cx.execute("SELECT 1 FROM sqlite_master WHERE name='topic_vec'").fetchone() is None:
            return self.store.count()
        return self.cx.execute(
            "SELECT COUNT(*) c FROM topic t LEFT JOIN topic_vec v ON v.id = t.id AND v.model = ? "
            "WHERE v.id IS NULL", (self.emb.model_name,)).fetchone()["c"]
