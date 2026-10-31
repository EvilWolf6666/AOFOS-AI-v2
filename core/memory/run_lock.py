# core/memory/run_lock.py
"""
「AOFOS 正在執行」的訊號：一個由作業系統持有的檔案鎖（給 tools/judge_worker.py 用）。

為什麼要它：獨立終端的背景判斷（judge_worker）收不到「使用者開口」的取消訊號——
它若在 AOFOS 開著時跑，對話就得排在它後面（推理後端只有一個推理槽），兩邊也會同時判同一個 DB。
⇒ AOFOS 執行期間持有這把鎖；judge_worker 每 0.5 秒試一次，**拿不到＝AOFOS 開著 ⇒ 取消並結束**。

⭐ 用作業系統的鎖、不用「檔案存不存在」：AOFOS 當掉時鎖由作業系統自動放掉，不會留下一個
   永遠擋路的舊鎖檔（檔案本身留著沒關係，鎖才是訊號）。
⚠️ 鎖檔放在記憶檔同一個資料夾（data/，已在 .gitignore）。
"""
from __future__ import annotations

import os
import time


def lock_path(db_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(db_path)), "aofos.run.lock")


class RunLock:
    def __init__(self, path: str) -> None:
        self.path = path
        self._f = None

    def try_acquire(self) -> bool:
        """不等待；拿到回 True。同一個行程內另開一個 RunLock 也會拿不到（鎖綁在開檔上）。"""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        f = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        self._f = f
        return True

    def acquire(self, timeout_s: float = 3.0) -> bool:
        """等一下再放棄：judge_worker 每 0.5 秒會「試拿再放」一次，可能剛好撞上。"""
        end = time.monotonic() + timeout_s
        while True:
            if self.try_acquire():
                return True
            if time.monotonic() >= end:
                return False
            time.sleep(0.1)

    def release(self) -> None:
        if self._f is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self._f.seek(0)
                msvcrt.locking(self._f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._f, fcntl.LOCK_UN)
        finally:
            self._f.close()
            self._f = None


def held_by_other(path: str) -> bool:
    """別人拿著鎖嗎？（試拿，拿到就馬上放掉。）"""
    probe = RunLock(path)
    if probe.try_acquire():
        probe.release()
        return False
    return True

