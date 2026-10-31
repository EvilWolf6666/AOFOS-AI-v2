# core/memory/embed.py
"""
embedding 模型（背景載入）。準 v4 主題向量用（core/memory/topic_index.py）。

舊 DB 可能殘留 `vec` 表（原句向量，已不用；衍生物，要清可直接 DROP）。

🔴 **成本都在啟動，不在每輪**（Windows 實測）：
   import＋載入約 18 s ⇒ **一律在背景執行緒載入**，載好之前主題只走關鍵字那一路、不擋對話；
   每輪查詢編碼約 33 ms；常駐約 800 MB（LLM 總耗時影響約 1.7%，見 config 註解）。

🔴 **依賴缺失不擋啟動，但要大聲**：sentence-transformers／numpy／模型快取任一缺，
   記一次 WARNING、主題只走關鍵字那一路。⚠️ 與 FTS5 相反（那個缺了拒絕啟動）。

⚠️ 模型由主執行緒（編碼查詢句）與背景判斷執行緒（補主題向量）共用，以鎖序列化。
"""

from __future__ import annotations

import threading

from core.logger import setup_logger

logger = setup_logger("MemoryEmbed")


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
            logger.warning(f"向量檢索停用，主題只走關鍵字那一路：{self.failed}")

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
