# core/agent/event_queue.py
"""
事件佇列。

介面與 queue.Queue 相容（put / get / qsize），額外支援「同來源覆蓋」——
producer 推入新事件時，佇列中該來源的舊事件會被取代而非累積。

⚠️ 為何不用 queue.Queue：它無法窺看或修改內容，做不到覆蓋。而覆蓋是必要的，
因為**環境狀態是當前值，不是事件流**。「他現在在瀏覽器」有意義，
「他 5 分鐘前切到瀏覽器」在冷卻結束後已經過期——舊事件湧出會讓奧狐
連續講好幾件已經不成立的事。

⚠️ utterance 不覆蓋。使用者說的每一句都必須處理，覆蓋等於吞掉發言。
覆蓋只作用於 kind == OBSERVATION 且 source 相同者。
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Optional

from core.agent.events import Event, OBSERVATION


class EventQueue:
    """
    執行緒安全的事件佇列。

    Condition 同時擔任鎖與等待機制：put 後 notify、get 在空佇列時 wait。
    所有對 _items 的存取都在 with self._cond 之內——deque 本身的 append/popleft
    雖是原子操作，但「檢查是否有同來源事件再取代」是複合操作，必須整段上鎖。
    """

    def __init__(self) -> None:
        self._items: deque[Event] = deque()
        self._cond = threading.Condition()

    def put(self, event: Event) -> None:
        """
        推入事件。若為 observation 且佇列中已有同來源的 observation，取代之。

        取代而非「移除後append」是刻意的：保持原本的排隊位置，否則反覆更新的
        來源會一直被推到隊尾，永遠排在其他來源之後。
        """
        with self._cond:
            if event.kind == OBSERVATION:
                for i, existing in enumerate(self._items):
                    if existing.kind == OBSERVATION and existing.source == event.source:
                        self._items[i] = event
                        self._cond.notify()
                        return
            self._items.append(event)
            self._cond.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[Event]:
        """取出最早的事件；佇列為空時最多等 timeout 秒，等不到回傳 None（timeout=0 不等）。

        ⚠️ 主執行緒一律帶 timeout 分段等：Windows 上不帶逾時的 wait 收不到 Ctrl+C
           （Python 3.14 前），主迴圈會卡死在這裡。
        """
        with self._cond:
            if not self._cond.wait_for(lambda: self._items, timeout):
                return None
            return self._items.popleft()

    def qsize(self) -> int:
        with self._cond:
            return len(self._items)

    def drop_source(self, source: str) -> int:
        """
        丟棄指定來源的所有 observation，回傳丟棄數量。

        供仲裁層在「決定不處理某來源的事件」時清場用——若只丟棄當下取出的那則、
        佇列裡還留著同來源的其他則，下一輪又會被取出，冷卻形同虛設。
        """
        with self._cond:
            before = len(self._items)
            self._items = deque(
                e for e in self._items
                if not (e.kind == OBSERVATION and e.source == source)
            )
            return before - len(self._items)

    def snapshot(self) -> list[Event]:
        """回傳目前內容的複本，供 /status 等診斷用途。不影響佇列本身。"""
        with self._cond:
            return list(self._items)