# core/short_time_test.py
"""
暫時性診斷腳手架集中處。

本檔案內的所有東西都是為了釐清特定問題而臨時加入的，問題釐清後（不論結論為何）
應整段刪除，不保留、不註解掉、不比照 archive/ 的做法保存。

集中在單一檔案的理由：這些程式碼原本散在 text_utils.py 與 RAG 模組（現 core/memory/rag.py）內部，
刪除時要在多個檔案裡辨認「哪幾行是腳手架、哪幾行是正式功能」。集中之後，
清理動作變成「刪掉本檔 + 移除各呼叫點的 import 與單行呼叫」。

⚠️ 延遲量測 timed() 刻意不放在這裡——它是長期需求，不是腳手架，
   放進本檔會在日後整檔刪除時被一併誤刪。

目前掛著的腳手架與刪除條件：

  1. log_opencc_diff()
     呼叫點：core/text_utils.py fix_traditional()
     ⚠️ 舊目的（釐清異體字來自模型還是 opencc s2t）已於 2026-08-06 達成且
        **證偽**——掛載期間抓到的每一次都是簡體字，異體字一次都沒出現過。
        本項因新用途而延長壽命，非默默留著；詳見《歸檔》I0035。
     ⚠️ **「新目的」那條也已作廢**（2026-08-16，《歸檔》I0042）：用「它有沒有動作」
        判斷 opencc 存廢的判讀規則未經驗證，量測已改為直接關掉 opencc 看輸出。
     現行保留理由（《現況》第八章 C13，2026-08-28 由假設轉為排程）：
           2026-08-26 三組 36 題實測在 opencc 關閉下仍抓到簡體字，**恢復已成定局、
           只剩何時**（2026-09-03 裁定等換設備後再議），而恢復時本函式是唯一能
           看見轉換內容的工具。
     現行刪除條件：綁在 opencc 恢復排程上——恢復並穩定後才重新評估。
        ⚠️ **施測期間不要開它**，它是新變數。

  （🔺 2026-10-24 移除 dump_observation_messages()：刪除條件「空 U 複製問題釐清後」已達成——
     照抄的成因是「system 訊息接在上一句回覆後面」，情境改併進 user 後 0 次（《現況》22之84、第八章 C14）。）
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

    ⚠️ 現行轉換模式為 s2tw（台灣字形標準）。舊目的是釐清「纔／喫／牀／爲」等
    異體字來自模型輸出還是 opencc s2t——該假設已證偽，抓到的全是簡體字。

    現行目的：換基礎模型後跑一輪固定評測題組，看本函式是否完全不輸出。
    完全不輸出 = 新模型不吐簡體字 = opencc 這一層可以整個拿掉，
    少一層後處理就少一個變數。

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
