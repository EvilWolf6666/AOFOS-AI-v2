# core/memory/embed.py
"""
向量檢索（準 v3 組合 6 的向量那一路），與 bigram＋FTS5 那一路在 manager 以 RRF 合併。

⭐ **衍生物，不是資料**：`vec` 表完全由 `transcript` 算出來；換 embedding 模型時
   舊模型的向量自動不被採用、背景重算，原句一筆不動。
⭐ **確定性**：逐筆算餘弦（不用 ANN）——同一個庫、同一句話，今天撈到的和昨天一樣
   （《現況》〈🔒 隔離區〉準 v3〈八〉「穩定且對」的第一項）。資料量到萬筆級仍是毫秒級。

🔴 **成本都在啟動，不在每輪**（2026-09-27 Windows 實測）：
   import＋載入約 18 s ⇒ **一律在背景執行緒載入**，載好之前只用字面檢索、不擋對話；
   每輪查詢編碼約 33 ms；常駐約 800 MB（LLM 總耗時影響約 1.7%，見 config 註解）。

🔴 **依賴缺失不擋啟動，但要大聲**：sentence-transformers／numpy／模型快取任一缺，
   記一次 WARNING、退回只用字面檢索。⚠️ 與 FTS5 相反（那個缺了拒絕啟動）——
   FTS5 是唯一的取回路徑，向量只是第二路，少了它記憶仍然撈得到。

⚠️ **執行緒分工**：模型載入、補算向量、寫 `vec` 表全在背景執行緒（自己開連線）；
   主執行緒只讀 `vec` 與編碼查詢句。兩邊共用模型時以鎖序列化（最多等一次編碼）。
"""

from __future__ import annotations

import queue
import sqlite3
import threading

from core.logger import setup_logger
from core.memory.db import connect, write_txn

logger = setup_logger("MemoryEmbed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vec (
    serial INTEGER PRIMARY KEY,
    model  TEXT NOT NULL,
    v      BLOB NOT NULL          -- float32、已正規化（內積＝餘弦）
);
"""

_BATCH = 32


class Embedder:
    """背景載入的 embedding 模型。`factory` 只給自測注入假模型用。"""

    def __init__(self, model_name: str, *, factory=None) -> None:
        self.model_name = model_name
        self._factory = factory
        self._model = None
        self._np = None
        self._lock = threading.Lock()
        self.ready = threading.Event()
        self.failed: str | None = None

    def start(self) -> None:
        threading.Thread(target=self._load, name="embed-load", daemon=True).start()

    def _load(self) -> None:
        try:
            import numpy as np
            if self._factory is not None:
                model = self._factory()
            else:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.model_name, device="cpu")
            model.encode(["暖機"])
            self._np, self._model = np, model
            self.ready.set()
            logger.info(f"向量檢索就緒（{self.model_name}）")
        except Exception as e:  # noqa: BLE001 —— 任何原因都退回字面檢索，但要大聲
            self.failed = f"{type(e).__name__}: {e}"
            logger.warning(f"向量檢索停用，只用字面檢索：{self.failed}")

    def encode(self, texts: list[str]):
        """回傳正規化後的 float32 矩陣。**未就緒時呼叫是程式錯誤**，直接拋。"""
        if not self.ready.is_set():
            raise RuntimeError("embedding 模型尚未就緒")
        np = self._np
        with self._lock:
            m = np.asarray(self._model.encode(texts), dtype=np.float32)
        if m.ndim == 1:
            m = m[None, :]
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return m / n


def ensure_schema(cx: sqlite3.Connection) -> None:
    cx.executescript(_SCHEMA)


class VectorWorker:
    """背景補算向量：收到某個 DB 的通知就把「還沒有向量的原句」補齊。

    🔴 **它會默默死掉的那種執行緒**（《現況》〈📐 規則〉背景執行緒要有積壓告警）：
       每批例外都接住並記 ERROR、不讓執行緒結束；積壓數由 `backlog()` 對外可查，
       manager 的 /status 會印出來。
    """

    def __init__(self, embedder: Embedder) -> None:
        self.emb = embedder
        self._q: queue.Queue = queue.Queue()
        self._cx: dict[str, sqlite3.Connection] = {}
        threading.Thread(target=self._run, name="embed-worker", daemon=True).start()

    def notify(self, path: str) -> None:
        self._q.put(("fill", path, None))

    def release(self, path: str, timeout: float = 5.0) -> None:
        """關掉背景對某個 DB 的連線（丟棄測試區前必須先做——Windows 不能刪開著的檔）。"""
        done = threading.Event()
        self._q.put(("release", path, done))
        if not done.wait(timeout):
            logger.error(f"背景向量執行緒沒有在 {timeout}s 內放開 {path}")

    def _run(self) -> None:
        while True:
            op, path, done = self._q.get()
            try:
                if op == "release":
                    cx = self._cx.pop(path, None)
                    if cx is not None:
                        cx.close()
                    continue
                # 等模型載好；載入失敗就放棄（failed 會被設、ready 永遠不會）
                while not self.emb.ready.wait(timeout=1.0):
                    if self.emb.failed:
                        break
                if self.emb.ready.is_set():
                    self._fill(path)
            except Exception as e:  # noqa: BLE001
                logger.error(f"背景補算向量失敗（{path}），下次通知再補：{type(e).__name__}: {e}")
            finally:
                if done is not None:
                    done.set()

    def _fill(self, path: str) -> None:
        cx = self._cx.get(path)
        if cx is None:
            cx = self._cx[path] = connect(path)
            ensure_schema(cx)
        model = self.emb.model_name
        while True:
            rows = cx.execute(
                "SELECT t.serial, COALESCE(t.clean, t.content) AS c, t.situation FROM transcript t "
                "LEFT JOIN vec v ON v.serial = t.serial AND v.model = ? "
                "WHERE v.serial IS NULL ORDER BY t.serial LIMIT ?", (model, _BATCH)).fetchall()
            if not rows:
                return
            texts = [f'{r["c"]} {r["situation"] or ""}'.strip() or "（空）" for r in rows]
            m = self.emb.encode(texts)
            with write_txn(cx):
                cx.executemany("INSERT OR REPLACE INTO vec(serial, model, v) VALUES(?,?,?)",
                               [(r["serial"], model, m[i].tobytes()) for i, r in enumerate(rows)])


class VectorSearch:
    """主執行緒用：讀 `vec` 表（增量快取）並做逐筆餘弦。每個 DB 連線一個。"""

    def __init__(self, cx: sqlite3.Connection, embedder: Embedder) -> None:
        self.cx = cx
        self.emb = embedder
        self._serials: list[int] = []
        self._mat = None
        self._max = 0

    def _refresh(self) -> None:
        if self.cx.execute("SELECT 1 FROM sqlite_master WHERE name='vec'").fetchone() is None:
            return
        np = self.emb._np
        rows = self.cx.execute(
            "SELECT serial, v FROM vec WHERE model = ? AND serial > ? ORDER BY serial",
            (self.emb.model_name, self._max)).fetchall()
        if not rows:
            return
        new = np.stack([np.frombuffer(r["v"], dtype=np.float32) for r in rows])
        self._mat = new if self._mat is None else np.vstack([self._mat, new])
        self._serials += [r["serial"] for r in rows]
        self._max = self._serials[-1]

    def search(self, qvec, *, candidates: int, min_sim: float,
               exclude: set[int] | frozenset[int] = frozenset()) -> list[tuple[int, float]]:
        """回傳 `[(serial, 餘弦), ...]`，由高到低；已退役（retired）的不回。"""
        self._refresh()
        if self._mat is None:
            return []
        np = self.emb._np
        sims = self._mat @ qvec
        order = np.argsort(-sims, kind="stable")
        out: list[tuple[int, float]] = []
        for i in order:
            s = float(sims[i])
            if s < min_sim or len(out) >= candidates:
                break
            serial = self._serials[i]
            if serial in exclude:
                continue
            out.append((serial, s))
        if not out:
            return []
        live = {r["serial"] for r in self.cx.execute(
            f"SELECT serial FROM transcript WHERE retired = 0 AND serial IN "
            f"({','.join('?' * len(out))})", [s for s, _ in out])}
        return [(s, sim) for s, sim in out if s in live]

    def pending(self) -> int:
        """還沒有向量的原句數（積壓）。"""
        if self.cx.execute("SELECT 1 FROM sqlite_master WHERE name='vec'").fetchone() is None:
            return self.cx.execute("SELECT COUNT(*) c FROM transcript").fetchone()["c"]
        return self.cx.execute(
            "SELECT COUNT(*) c FROM transcript t LEFT JOIN vec v "
            "ON v.serial = t.serial AND v.model = ? WHERE v.serial IS NULL",
            (self.emb.model_name,)).fetchone()["c"]
