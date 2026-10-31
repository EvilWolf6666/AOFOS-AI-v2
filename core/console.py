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
_streaming: bool = False   # 是否正在逐 token 輸出（見 begin_stream）


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

    ⚠️ 串流期間（begin_stream 與 end_stream 之間）**不清行**，改為先斷行再印。
    提示字元與已生成的回應在這裡有本質差異：提示可以清掉再重畫（它是可重建的），
    但已經印出來的 token 是最終產物，清掉就真的沒了——後續 delta 只會接在
    被清空的行上，使用者看到的是被腰斬的一句話。故串流內容一律只被推開，不被覆蓋。
    """
    with _lock:
        if _streaming:
            sys.stdout.write("\n" + text + end)
            sys.stdout.flush()
            return
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


# ── 串流輸出 ─────────────────────────────────────────────────────
#
# 為什麼不直接用 emit() 逐 token 呼叫：emit() 每次都會清行並重畫提示，
# 逐字呼叫等於每個 token 閃一次畫面。串流需要的是「開頭印一次前綴、中間只追加、
# 結尾收一次」，這是不同的輸出形狀，不是 emit 的參數能表達的。
#
# ⚠️ 刻意不在串流全程持鎖。持鎖會讓背景執行緒的 log 卡住整段生成時間，
# 而那些 log 有些是要當場看的警告。改為逐次取鎖 +
# _streaming 旗標，代價是 log 會把回應切成多行——可接受，因為字沒有消失。


def begin_stream(prefix: str = "") -> None:
    """
    進入串流輸出狀態，印出前綴（例如「奧狐：」）。

    必須與 end_stream() 成對呼叫，且 end_stream() 要放在 finally——中途例外
    若沒還原 _streaming，之後所有 emit() 都會停在「只斷行不重畫提示」的模式，
    輸入提示從此不再出現，而且看起來像當機。
    """
    global _streaming
    with _lock:
        if _prompt:
            sys.stdout.write(_CLEAR_LINE)
        _streaming = True
        if prefix:
            sys.stdout.write(prefix)
        sys.stdout.flush()


def write_stream(text: str) -> None:
    """追加一段已生成的內容，不換行、不重畫。空字串直接略過，避免無謂 flush。"""
    if not text:
        return
    with _lock:
        sys.stdout.write(text)
        sys.stdout.flush()


def end_stream(trailing: str = "\n") -> None:
    """
    結束串流輸出，補上收尾空白並在有提示等待中時重畫提示。

    trailing 由呼叫端指定，用來讓串流與非串流兩條路徑的畫面**逐字元一致**。
    ⚠️ 這不是排版偏好：格式不一致時，開關 stream 會讓畫面行距跳動，
    而那看起來像程式出了問題，會誘使人去查一個不存在的 bug。

    重複呼叫是安全的（第二次為 no-op），因為呼叫端多半寫在 finally，
    而 finally 有可能與正常路徑重疊。
    """
    global _streaming
    with _lock:
        if not _streaming:
            return
        _streaming = False
        sys.stdout.write(trailing)
        if _prompt:
            sys.stdout.write(_prompt)
        sys.stdout.flush()


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
         而 logs/daily/*.log 從此空白。長期記錄斷掉且無任何徵兆。

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