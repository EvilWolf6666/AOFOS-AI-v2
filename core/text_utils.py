# core/text_utils.py

import opencc

_converter = opencc.OpenCC("s2t")  # 純簡體字→繁體字，不替換詞彙

def fix_traditional(text: str) -> str:
    """把輸出裡的簡體字轉成繁體"""
    return _converter.convert(text)