# core/console.py
"""
終端輸出協調。

問題：主執行緒阻塞在 input("你：") 時，背景執行緒（事件 producer、log）的輸出
會直接插進去，把提示字元推走。使用者看到自己打的字散落在別的位置，
且不確定輸入還有沒有效。

做法：所有輸出經同一把鎖，印之前先清掉當前行，印完重畫提示。

⚠️ 這不是排版美化。語音模式接上 TTS 之後，「奧狐正在說話時畫面該顯示什麼」
是同一個問題的延伸，屆時的中斷與靜音提示也走這裡。現在建立是為了避免
那時再回頭改所有 print 呼叫點。

⚠️ 已輸入但尚未按 Enter 的內容無法取回——那些字元在 OS 的終端緩衝區裡，
Python 讀不到。重畫時只能還原提示字元本身，使用者已打的字會消失（但仍在
緩衝區內，按 Enter 依然送得出去）。要完整還原需接管逐字元輸入
（readline / prompt_toolkit），成本遠高於收益，不做。
"""

from __future__ import annotations

import sys
import threading

# ANSI：\r 回到行首，\033[K 清除該行從游標到行尾
_CLEAR_LINE = "\r\033[K"

_lock = threading.RLock()
_prompt: str = ""          # 目前顯示中的提示字元，空字串表示沒有提示在等輸入


def set_prompt(prompt: str) -> None:
    """
    登記目前正在等待輸入的提示字元。由呼叫 input() 的那一方在呼叫前設定。

    只登記不印出——input() 自己會印。這裡記住它，是為了在被打斷後能重畫。
    """
    global _prompt
    with _lock:
        _prompt = prompt


def clear_prompt() -> None:
    """輸入已取得，提示不再顯示中。"""
    global _prompt
    with _lock:
        _prompt = ""


def emit(text: str = "", end: str = "\n") -> None:
    """
    輸出一段文字，並在有提示等待中時重畫提示。

    所有非 input() 的畫面輸出都應走這裡，包括背景執行緒。直接呼叫 print()
    會繞過鎖，重新產生本模組要解決的問題。
    """
    with _lock:
        if _prompt:
            sys.stdout.write(_CLEAR_LINE)
        sys.stdout.write(text + end)
        if _prompt:
            sys.stdout.write(_prompt)
        sys.stdout.flush()


def input_with_prompt(prompt: str) -> str:
    """
    帶提示的阻塞輸入。與 emit() 共用同一把鎖的登記狀態，故被打斷時能重畫。

    ⚠️ input() 本身不上鎖——它會阻塞，持鎖阻塞會讓所有背景輸出跟著卡住。
    只有「登記提示」和「清除登記」上鎖。
    """
    set_prompt(prompt)
    try:
        return input(prompt)
    finally:
        clear_prompt()


# ── logging 橋接 ─────────────────────────────────────────────────

class _ConsoleStream:
    """
    file-like 包裝，把 logging 的輸出導進 emit()。

    logging 的 StreamHandler 會分兩次寫入（訊息本體、換行符），故 write() 收到
    純換行時直接忽略——否則每筆 log 都會多觸發一次重畫，畫面閃爍。
    """

    def write(self, text: str) -> int:
        if text and text != "\n":
            emit(text.rstrip("\n"))
        return len(text)

    def flush(self) -> None:
        pass


def install_logging_bridge() -> int:
    """
    把本專案 logger 的「終端 handler」導向 emit()，回傳改接的 handler 數。

    ⚠️ 兩個必須排除的東西，漏掉任一個都會出事：

      1. **FileHandler 是 StreamHandler 的子類別**。只用
         isinstance(h, StreamHandler) 過濾會一併改接檔案 handler——
         結果是檔案 log 被印到畫面（每筆訊息出現兩次，第二次帶時間戳），
         而 logs/*.log 從此空白。長期記錄斷掉且無任何徵兆。

      2. **第三方 logger**。掃 logging.root.manager.loggerDict 會取到
         chromadb / sentence-transformers / urllib3 等所有 import 過的套件。
         改接它們沒有意義，且數量會讓回報值失去參考價值。

    故改接對象限定為 core.logger.KNOWN_LOGGERS 登記的 logger，
    且只取非 FileHandler 的 StreamHandler。

    ⚠️ 呼叫時尚未建立的 logger 由 logger.set_console_stream() 涵蓋——本函式
    只能改接「當下已存在」的，而 TIMING logger 是惰性建立的（第一次 timed()
    區塊結束時才誕生，必定在本函式之後），且它是全專案輸出最密集的 logger。
    兩者缺一，終端協調就會在最主要的來源上失效。
    """
    import logging

    try:
        from core.logger import KNOWN_LOGGERS, set_console_stream
    except ImportError:
        logging.getLogger(__name__).warning(
            "core.logger 未提供 KNOWN_LOGGERS，log 未導向終端協調器"
            "（背景輸出可能插斷輸入行）"
        )
        return 0

    stream = _ConsoleStream()

    # 之後才建立的 logger（含惰性建立的 TIMING）由此涵蓋
    set_console_stream(stream)

    count = 0

    for name in list(KNOWN_LOGGERS):
        for handler in logging.getLogger(name).handlers:
            if isinstance(handler, logging.FileHandler):
                continue  # 檔案輸出不得改接，見上方說明
            if isinstance(handler, logging.StreamHandler) and hasattr(handler, "stream"):
                handler.stream = stream
                count += 1

    return count