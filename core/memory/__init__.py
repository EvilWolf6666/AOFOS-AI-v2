# core/memory/
"""
記憶模組（Working Memory ／ Compressed Memory ／ RAG ／ 預寫區 ／ 原文存放區）。

本套件是記憶這條線的唯一分群：**這幾個檔會一起被改，而且不跟外面一起改**
——分群判準是這個，不是行數（《現況》零章第八條）。

對外入口
    from core.memory import MemoryManager

⚠️ **診斷工具例外，這是刻意的**：tools/ 底下的腳本需要模組層級的東西
（RETIRED_FIELD、SOURCE_SEED、_build_where），一律直接
`from core.memory.rag import ...`。它們要回答的是「檢索層實際做了什麼」，
不是「正式路徑給了什麼」，所以繞過本層是對的。

🔴 **套件內部互相 import 一律寫完整子模組路徑**（`from core.memory.rag import RAGManager`），
   不得寫 `from core.memory import RAGManager`——後者會在本檔還沒執行完時
   回頭 import 本套件，形成循環 import。

⚠️ **路徑一律取自 config.PROJECT_ROOT**，本套件內任何檔案都不得用
   `os.path.dirname(__file__) + ".."` 推算專案根：那個寫法把檔案層級寫死，
   搬動時會靜默指到錯的位置（理由詳見 config.py 的 PROJECT_ROOT 區塊）。
"""

# 🆕 2026-09-26 manager.py 已整份換成準 v3 最小記憶切片
#    （《現況》〈🔒 隔離區〉記憶模塊〈準 v3〉〈十二〉）。
#    ⚠️ **本檔不再 import rag.py**——v1 的 rag.py 移出後本套件照常載入；
#    但 tools/ 底下直接 import `core.memory.rag` 的診斷腳本會隨之失效，這是預期的。
from core.memory.manager import MemoryManager

__all__ = ["MemoryManager"]

