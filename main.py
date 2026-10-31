# main.py
"""啟動入口。中央控制/流程控制已移交 core/agent/，這裡只建立 Agent 並驅動主迴圈。"""

from __future__ import annotations

# ⚠️ 必須排在所有其他 import 之前，且不可因為「看起來沒用到」而被移除。
# config 頂部會 load_dotenv() 並設定 HF_HUB_OFFLINE，而 huggingface_hub 是在
# import 當下把這些環境變數讀成模組常數——只要任何一個會連帶拉進 huggingface_hub
# 的模組（sentence-transformers、faster-whisper）先被 import，設定就來不及了，
# 而且不會報錯，只會靜默地變回連網。
#
# 環境設定集中於 config.py：tools/ 底下的腳本直接 import config、不經過 main.py（《歸檔》I0026）。
import config  # noqa: F401

import os  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

from core.agent.agent import Agent  # noqa: E402
from core.memory.run_lock import RunLock, lock_path  # noqa: E402


def main() -> None:
    # 執行期間持有「AOFOS 正在執行」鎖：獨立終端的背景判斷（tools/judge_worker.py）看到就讓出推理槽並結束。
    # 拿不到＝已經有另一個 AOFOS 在跑（兩個同時寫同一個記憶檔不行）。
    lock = RunLock(lock_path(config.MEMORY_CONFIG["db_path"]))
    if not lock.acquire():
        print("⚠️ 已經有另一個 AOFOS 在執行（記憶檔被占用中），本次不啟動。")
        sys.exit(1)
    try:
        agent = Agent()
        agent.run()
    finally:
        lock.release()


if __name__ == "__main__":
    code = 1
    try:
        main()
        code = 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except BaseException:
        traceback.print_exc()
    finally:
        # ⚠️ 收尾在 Agent.run() 裡已做完，這裡直接結束行程、不走 Python 的正常退出：
        #    文字讀取執行緒可能還卡在 input()（OS 層阻塞、停不了），正常退出時 C 執行環境要先
        #    鎖住 stdin 才能清理，整個行程就卡在那裡（Linux 終端實測，Ctrl+C 後關不掉）。
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
