# core/memory/
"""
記憶模組（準 v4：原句存放 transcript ／ 預寫區 pre_write ／ 清理 clean ／ 資料庫 db ／
主題 topics、topic_index ／ 背景判斷 judge ／ 切詞 bigram ／ 向量模型 embed ／ 事件層 episode ／
執行鎖 run_lock，入口 manager）。

本套件是記憶這條線的唯一分群：**這幾個檔會一起被改，而且不跟外面一起改**
——分群判準是這個，不是行數（《現況》零章第八條）。

對外入口
    from core.memory.manager import MemoryManager

🔴 **一律寫完整子模組路徑**（套件內外皆同）。本檔刻意不 re-export 任何東西：
   一旦在這裡 import manager，`python -m core.memory.db --selftest` 會先經由本檔
   把 db 載入一次、runpy 再以 __main__ 執行第二份 ⇒ 每次都印 RuntimeWarning
   （一個每次都出現的警告會訓練人忽略警告）。

⚠️ **路徑一律取自 config.PROJECT_ROOT**，本套件內任何檔案都不得用
   `os.path.dirname(__file__) + ".."` 推算專案根：那個寫法把檔案層級寫死，
   搬動時會靜默指到錯的位置（理由詳見 config.py 的 PROJECT_ROOT 區塊）。
"""
