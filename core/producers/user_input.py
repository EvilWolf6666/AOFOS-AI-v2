# core/producers/user_input.py
"""
使用者輸入的事件來源（文字、語音共用同一個機制）。

每個來源一條常駐執行緒，只在主迴圈 request() 時做一次阻塞讀取，讀到就包成 Event
推進共用佇列。「什麼時候讀」由主迴圈決定（閒著才要），這裡只負責讀與包裝。

⚠️ 為什麼要等 request() 才讀，不一直讀：
   · 文字：input() 一啟動就印出「你：」並開始回顯打字。她還在輸出時就先開讀，
     提示與打的字會混進回應中間。
   · 語音：她輸出時不收音（說話期間收音與打斷屬《現況》〈🔒 事件驅動〉，未裁決）。

⚠️ 讀取中的 input() 無法從外部中斷（OS 層阻塞），只能靠 daemon 隨行程結束。
   所以全程只有一條文字讀取執行緒，且從不 join。

本檔不 import core.voice_input：語音讀取只拿現成的實例（Agent 第一次切語音時才建），
文字模式的使用者不必載入語音模型。
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from core import console
from core.agent.events import Event, UTTERANCE
from core.logger import setup_logger

logger = setup_logger("UserInput")

_IDLE, _REQUESTED, _READING = "idle", "requested", "reading"


class OnDemandReader:
    """
    read_once()：做一次阻塞讀取，回傳 Event；回傳 None 表示這次沒讀到東西（空行、雜音），
    直接再讀一次，不回主迴圈。回傳 Event 後回到閒置，等下一次 request()。

    on_error：read_once 拋例外時改推的事件（讀取端壞掉時，主迴圈仍要等得到一則事件，
    否則會永遠卡在等輸入）。
    """

    def __init__(self, name: str, read_once: Callable[[], Optional[Event]], queue,
                 on_error: Event) -> None:
        self.name = name
        self._read_once = read_once
        self._queue = queue
        self._on_error = on_error
        self._state = _IDLE
        self._closed = False
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    def request(self) -> None:
        """要一則輸入。已在等或正在讀時不重複（同一時間只會有一次讀取）。"""
        with self._cond:
            if self._state == _IDLE and not self._closed:
                self._state = _REQUESTED
                self._cond.notify()

    def close(self) -> None:
        """不再接受 request()。進行中的讀取由讀取端自己的停止機制結束（文字讀取停不了，見檔頭）。"""
        with self._cond:
            self._closed = True
            self._cond.notify()

    def _loop(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._state == _REQUESTED or self._closed)
                if self._closed:
                    return
                self._state = _READING
            event = None
            while event is None and not self._closed:
                try:
                    event = self._read_once()
                except Exception as e:  # noqa: BLE001
                    logger.error(f"{self.name} 讀取失敗：{type(e).__name__}: {e}")
                    event = self._on_error
            # 🔴 先回到閒置再推事件：主迴圈拿到事件、處理完、再 request() 時這裡必須已是閒置，
            #    否則那次 request() 會被當成重複而丟掉，主迴圈從此等不到輸入。
            with self._cond:
                self._state = _IDLE
            if event is not None:
                self._queue.put(event)


def _command(source: str, text: str) -> Event:
    return Event(source=source, kind=UTTERANCE, payload=text)


# ── 文字 ─────────────────────────────────────────────────────────

def read_text_line() -> Optional[Event]:
    """一行文字輸入。空行不算輸入（回 None 讓讀取器再讀）。

    EOF（Windows 上 Ctrl+Z 再 Enter、或 stdin 被關掉）＝之後不可能再有輸入 ⇒ 當成 quit，
    走正常路徑收尾（寫完記憶再結束），不留一個永遠等不到輸入的主迴圈。
    """
    try:
        text = console.input_with_prompt("你：").strip()
    except EOFError:
        return _command("user_text", "quit")
    return _command("user_text", text) if text else None


def text_reader(queue) -> OnDemandReader:
    # 讀取端壞掉＝之後讀不到任何文字 ⇒ 收尾離開（同 EOF）
    return OnDemandReader("user_text", read_text_line, queue, on_error=_command("user_text", "quit"))


# ── 語音 ─────────────────────────────────────────────────────────

def voice_reader(queue, voice_input) -> OnDemandReader:
    """voice_input.listen()：回傳辨識文字（可能是空字串）；使用者按 ESC 時回傳 None。"""

    def read_once() -> Optional[Event]:
        text = voice_input.listen()
        if text is None:          # ESC ＝ 使用者要離開語音模式，跟打 /mode t 同一件事
            return _command("user_voice", "/mode t")
        text = text.strip()
        return _command("user_voice", text) if text else None

    # 麥克風壞掉（拔掉、被占用）⇒ 切回文字模式，不讓整個程式跟著倒
    return OnDemandReader("user_voice", read_once, queue, on_error=_command("user_voice", "/mode t"))
