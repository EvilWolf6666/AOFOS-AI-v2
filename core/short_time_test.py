# core/short_time_test.py
"""
暫時性診斷腳手架集中處。

本檔案內的所有東西都是為了釐清特定問題而臨時加入的，問題釐清後（不論結論為何）
應整段刪除，不保留、不註解掉、不比照 archive/ 的做法保存。

集中在單一檔案的理由：散在多個模組時，刪除要逐檔辨認「哪幾行是腳手架、哪幾行是正式功能」。集中之後，
清理動作變成「刪掉本檔 + 移除各呼叫點的 import 與單行呼叫」。

⚠️ 延遲量測 timed() 刻意不放在這裡——它是長期需求，不是腳手架，
   放進本檔會在日後整檔刪除時被一併誤刪。

目前掛著的腳手架與刪除條件：

  1. log_opencc_diff()
     呼叫點：core/text_utils.py fix_traditional()
     保留理由（《現況》第八章 C13）：opencc 關閉下輸出仍有簡體字，恢復只剩時機，
        而恢復時本函式是唯一能看見轉換內容的工具。舊目的與判讀規則已作廢（《歸檔》I0035、I0042）。
     刪除條件：opencc 恢復並穩定後重新評估。
     ⚠️ 施測期間不要開它（它是新變數），兩組對照必須同值。
"""

from __future__ import annotations

from config import DIAGNOSTICS
from core.logger import setup_logger

logger = setup_logger("Diagnostics")


# ══════════════════════════════════════════════════════════════
# 1. opencc 轉換差異記錄
# ══════════════════════════════════════════════════════════════

def log_opencc_diff(before: str, after: str) -> None:
    """
    記錄 opencc 實際改動了哪些字。

    ⚠️ 現行轉換模式為 s2tw（台灣字形標準）。用途見檔頭第 1 項。

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
