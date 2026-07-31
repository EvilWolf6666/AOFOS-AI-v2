# core/text_utils.py

import opencc

from config import DIAGNOSTICS
from core.logger import setup_logger

logger = setup_logger("TextUtils")

_converter = opencc.OpenCC("s2t")  # 純簡體字→繁體字，不替換詞彙


def fix_traditional(text: str) -> str:
    """把輸出裡的簡體字轉成繁體"""
    converted = _converter.convert(text)

    # ── 診斷腳手架（暫時性，見 config.DIAGNOSTICS）─────────────────
    if DIAGNOSTICS.get("log_opencc_changes") and converted != text:
        _log_conversion_diff(text, converted)
    # ── 診斷腳手架結束 ─────────────────────────────────────────

    return converted


# ══════════════════════════════════════════════════════════════
# 以下為診斷腳手架，問題釐清後整段刪除（含上方 fix_traditional 內的呼叫）
# ══════════════════════════════════════════════════════════════

def _log_conversion_diff(before: str, after: str) -> None:
    """
    記錄 opencc 實際改動了哪些字。

    存在目的：釐清「纔、喫、牀、爲」這類現代中文罕見的異體字，
    是模型本來就這樣輸出，還是 opencc 的 s2t 模式轉出來的。
    s2t 走大陸繁體字形標準（才→纔、吃→喫、床→牀、為→爲），
    若確認是 opencc 造成，改用 s2tw（台灣字形標準，不動詞彙）即可解決。

    只記差異字元而非整句，避免洗版且便於統計。
    """
    if len(before) == len(after):
        pairs = [f"{b}→{a}" for b, a in zip(before, after) if b != a]
        if pairs:
            logger.debug(f"opencc 改動 {len(pairs)} 字：{'、'.join(pairs)}")
    else:
        # 長度不同代表不是單純的一對一字元替換，直接記兩邊供人工判讀
        logger.debug(f"opencc 改動（長度變化）：{before[:60]} → {after[:60]}")