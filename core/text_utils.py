# core/text_utils.py

import opencc

from core.diagnostics import log_opencc_diff   # 診斷腳手架，清理時連同下方呼叫一併移除

_converter = opencc.OpenCC("s2t")  # 純簡體字→繁體字，不替換詞彙


def fix_traditional(text: str) -> str:
    """把輸出裡的簡體字轉成繁體。"""
    converted = _converter.convert(text)
    log_opencc_diff(text, converted)   # 診斷腳手架
    return converted