# core/diagnostics.py
"""
暫時性診斷腳手架集中處。

本檔案內的所有東西都是為了釐清特定問題而臨時加入的，問題釐清後（不論結論為何）
應整段刪除，不保留、不註解掉、不比照 archive/ 的做法保存。

集中在單一檔案的理由：這些程式碼原本散在 text_utils.py 與 rag_manager.py 內部，
刪除時要在多個檔案裡辨認「哪幾行是腳手架、哪幾行是正式功能」。集中之後，
清理動作變成「刪掉本檔 + 移除各呼叫點的 import 與單行呼叫」。

⚠️ 延遲量測 timed() 刻意不放在這裡——它是長期需求，不是腳手架，
   放進本檔會在日後整檔刪除時被一併誤刪。

目前掛著的腳手架與刪除條件：

  1. log_opencc_diff()
     呼叫點：core/text_utils.py fix_traditional()
     目的：釐清「纔／喫／牀／爲」等異體字來自模型輸出還是 opencc s2t 轉換
     刪除條件：確認來源後（若為 opencc，改用 s2tw 即可）

  2. RagStats
     呼叫點：core/rag_manager.py __init__ / add() / query()
     輸出：data/rag_stats.json
     目的：追蹤 _compress 淘汰內容轉存 RAG 後是否查得到（命中率）
     刪除條件：命中率問題釐清後
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from config import DIAGNOSTICS
from core.logger import setup_logger

logger = setup_logger("Diagnostics")


# ══════════════════════════════════════════════════════════════
# 1. opencc 轉換差異記錄
# ══════════════════════════════════════════════════════════════

def log_opencc_diff(before: str, after: str) -> None:
    """
    記錄 opencc 實際改動了哪些字。

    s2t 走大陸繁體字形標準（才→纔、吃→喫、床→牀、為→爲）；若確認異體字是
    opencc 造成而非模型輸出，改用 s2tw（台灣字形標準，不動詞彙）即可解決。

    只記差異字元而非整句，避免洗版且便於統計。開關未開時直接返回。
    """
    if not DIAGNOSTICS.get("log_opencc_changes"):
        return
    if before == after:
        return

    if len(before) == len(after):
        pairs = [f"{b}→{a}" for b, a in zip(before, after) if b != a]
        if pairs:
            logger.debug(f"opencc 改動 {len(pairs)} 字：{'、'.join(pairs)}")
    else:
        # 長度不同代表不是單純的一對一字元替換，直接記兩邊供人工判讀
        logger.debug(f"opencc 改動（長度變化）：{before[:60]} → {after[:60]}")


# ══════════════════════════════════════════════════════════════
# 2. RAG Write/Search 計數器
# ══════════════════════════════════════════════════════════════

STATS_FILE = Path(
    os.path.join(os.path.dirname(__file__), "..", "data", "rag_stats.json")
)

_DEFAULT_STATS = {
    "writes": 0,             # 寫入次數
    "searches": 0,           # 查詢總次數
    "searches_with_hit": 0,  # 至少命中一筆的查詢次數
    "docs_returned": 0,      # 累計回傳文件數（通過相似度門檻的）
}


class RagStats:
    """
    RAG 寫入／查詢次數統計。

    searches 與 searches_with_hit 分開記錄才算得出命中率；docs_returned 另計，
    避免「命中幾次查詢」與「回傳幾筆文件」兩個語意被混在同一個數字裡。

    開關關閉時所有方法皆為 no-op，不建立也不寫入統計檔。
    """

    def __init__(self) -> None:
        self.enabled: bool = bool(DIAGNOSTICS.get("rag_stats", True))
        self._data: dict = dict(_DEFAULT_STATS)

    def load(self) -> None:
        """讀取既有統計並立即寫回一次。

        寫回是刻意的：若只在有寫入/查詢時才建檔，「檔案不存在」會同時代表
        「沒有任何淘汰內容需要寫入」與「寫入路徑壞掉」，而區分這兩者正是
        本計數器存在的目的。
        """
        if not self.enabled:
            return
        if STATS_FILE.exists():
            try:
                self._data.update(json.loads(STATS_FILE.read_text(encoding="utf-8")))
            except Exception:
                pass
        self._save()

    def record_write(self) -> None:
        if not self.enabled:
            return
        self._data["writes"] += 1
        self._save()

    def record_search(self, hit_docs: int) -> None:
        """每次查詢只呼叫一次。hit_docs 為本次通過門檻的文件數，0 代表未命中。"""
        if not self.enabled:
            return
        self._data["searches"] += 1
        if hit_docs > 0:
            self._data["searches_with_hit"] += 1
            self._data["docs_returned"] += hit_docs
        self._save()

    def _save(self) -> None:
        try:
            STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATS_FILE.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            # 診斷工具不得成為新的故障點
            logger.warning(f"RAG 統計寫入失敗（不影響主流程）：{e}")
