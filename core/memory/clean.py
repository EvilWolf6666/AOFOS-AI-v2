# core/memory/clean.py
"""
原樣輸出 → 乾淨文字。**表情標籤的剝除規則唯一住在這裡。**

準 v3 最小記憶切片「標籤走 C」（《現況》〈🔒 隔離區〉記憶模塊〈準 v3〉〈十二〉）：
`transcript.content` 存原樣輸出（什麼都不丟），`transcript.clean` 存本函式的結果，
檢索與注入一律讀 `clean`。

🟦 欄位是目標形；🟩 規則目前是**恆等**——表情格式還沒定，現在寫剝除規則等於猜格式。
⚠️ 規則定了之後要做兩件事，缺一不可：
   ① 改本函式
   ② 對既有列重算一次 `clean`，再 `python -m core.memory.recall --rebuild`
      （索引是從 `clean` 衍生的，不重建會留著舊字）
"""

from __future__ import annotations


def clean_text(content: str) -> str:
    """🟩 目前恆等。⚠️ 呼叫端不得假設它恆等——那只是規則未定的暫時狀態。"""
    return content
