# core/producers/window_watcher.py
"""
前景視窗監看 producer。

背景執行緒輪詢前景視窗，偵測到「使用者改去做別的事」或「久無切換」時，
把一則 observation 事件推進 event_queue。

⚠️ 只負責「偵測與變化判斷」，不決定奧狐要不要說話——那是仲裁層的事
（Agent._arbitrate()）。producer 自行判斷是否發言會造成分散決策點，
多來源時必然衝突（見《決策脈絡歸檔》十一之1、十一之3）。

⚠️ 純 Win32 API（ctypes），不引入 pywin32 / psutil。理由是這個 producer
的價值就在成本近乎為零，為它裝套件不划算。非 Windows 平台會在 start()
時降級為不啟動，不中斷主流程。
"""

from __future__ import annotations

import ctypes
import threading
import time
from typing import Optional

from config import WINDOW_WATCHER_CONFIG
from core.agent.events import Event, OBSERVATION
from core.logger import setup_logger

logger = setup_logger("WindowWatcher")

MAX_PATH = 260
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class WindowWatcher:
    """
    Attributes:
        _last_process: 上次偵測到的前景程式名（小寫），None 表示尚未取得
        _stay_since: 目前這個程式從何時開始持續在前景（time.monotonic）
        _idle_reported: 本次停留是否已推過「久無切換」事件，避免每輪重複推
    """

    def __init__(self, event_queue) -> None:
        self.event_queue = event_queue
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._last_process: Optional[str] = None
        self._stay_since: float = 0.0
        self._idle_reported: bool = False

    # ── 生命週期 ─────────────────────────────────────────────────

    def start(self) -> bool:
        """啟動背景輪詢。回傳是否實際啟動。"""
        if not WINDOW_WATCHER_CONFIG["enabled"]:
            logger.info("視窗監看已於設定中關閉，不啟動")
            return False

        if not hasattr(ctypes, "windll"):
            logger.warning("非 Windows 平台，視窗監看不啟動（不影響其他功能）")
            return False

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(
            f"視窗監看已啟動（輪詢 {WINDOW_WATCHER_CONFIG['poll_interval_seconds']}s、"
            f"久無切換門檻 {WINDOW_WATCHER_CONFIG['idle_threshold_seconds']}s）"
        )
        return True

    def stop(self) -> None:
        """
        要求背景執行緒結束。不 join——執行緒最長只會再跑一個輪詢間隔，
        且為 daemon，不阻擋程式退出。
        """
        self._stop.set()

    # ── 輪詢主體 ─────────────────────────────────────────────────

    def _loop(self) -> None:
        interval = WINDOW_WATCHER_CONFIG["poll_interval_seconds"]
        while not self._stop.wait(interval):
            try:
                self._tick()
            except Exception as e:
                # producer 不得成為主流程的故障點；記錄後繼續輪詢
                logger.warning(f"視窗偵測失敗（本輪略過）：{e}")

    def _tick(self) -> None:
        process = self._foreground_process()
        if process is None:
            return

        if process in WINDOW_WATCHER_CONFIG["ignored_processes"]:
            # 略過的程式不重置停留計時：使用者短暫切到檔案總管再切回來，
            # 不算「換了一件事在做」
            return

        now = time.monotonic()

        if process != self._last_process:
            previous = self._last_process
            previous_since = self._stay_since
            self._last_process = process
            self._stay_since = now
            self._idle_reported = False

            # 第一次偵測（previous is None）不推事件——那是程式剛啟動時的
            # 初始狀態，不是「切換」，推出去會讓奧狐一開機就評論
            if previous is not None:
                stayed = now - previous_since
                self._emit(
                    f"邪狼剛才在{self._display(previous)}"
                    f"{self._humanize_duration(stayed)}，"
                    f"現在切到{self._display(process)}。"
                )
            return

        # 同一程式持續在前景
        threshold = WINDOW_WATCHER_CONFIG["idle_threshold_seconds"]
        if not self._idle_reported and (now - self._stay_since) >= threshold:
            self._idle_reported = True
            minutes = int(threshold // 60)
            self._emit(
                f"邪狼已經連續使用{self._display(process)}超過 {minutes} 分鐘，"
                f"中間沒有切換到其他程式。"
            )

    def _emit(self, description: str) -> None:
        self.event_queue.put(
            Event(source="window", kind=OBSERVATION, payload=description)
        )
        logger.debug(f"推入視窗事件：{description}")

    @staticmethod
    def _humanize_duration(seconds: float) -> str:
        """
        停留時長轉為口語描述。

        給精確秒數沒有意義——奧狐要判斷的是「這件事值不值得講」，
        待 2 分鐘就切走跟待 3 小時的反應本來就該不同，而中間的精確值不影響判斷。
        分段而非給原始數字，也避免她把數字直接複誦出來。
        """
        minutes = seconds / 60
        if minutes < 3:
            return "只待了一下下"
        if minutes < 15:
            return f"待了大約 {int(minutes)} 分鐘"
        if minutes < 60:
            return f"待了快 {int(minutes)} 分鐘"
        hours = minutes / 60
        if hours < 2:
            return "待了一個多小時"
        return f"待了超過 {int(hours)} 小時"

    @staticmethod
    def _display(process: str) -> str:
        """程式名 → 可讀名稱。查不到時回傳程式名本身，不隱藏未知程式。"""
        return WINDOW_WATCHER_CONFIG["process_display_names"].get(process, process)

    # ── Win32 ────────────────────────────────────────────────────

    @staticmethod
    def _foreground_process() -> Optional[str]:
        """
        取得前景視窗所屬的程式檔名（小寫）。取不到回傳 None。

        用 QueryFullProcessImageNameW 而非 GetModuleFileNameEx：後者需要
        PROCESS_VM_READ 權限，對某些程式會失敗；前者只要
        PROCESS_QUERY_LIMITED_INFORMATION。
        """
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None

        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return None

        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value
        )
        if not handle:
            return None

        try:
            buf = ctypes.create_unicode_buffer(MAX_PATH)
            size = ctypes.c_ulong(MAX_PATH)
            if not kernel32.QueryFullProcessImageNameW(
                handle, 0, buf, ctypes.byref(size)
            ):
                return None
            return buf.value.rsplit("\\", 1)[-1].lower()
        finally:
            kernel32.CloseHandle(handle)