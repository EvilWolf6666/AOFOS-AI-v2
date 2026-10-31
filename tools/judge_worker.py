# tools/judge_worker.py
"""
獨立終端的背景判斷：AOFOS 關了、推理後端還開著時，把還沒判的原句判完。

用法：雙擊 start/背景判斷.bat（或 python -m tools.judge_worker）。判完自動結束。

規則：
  · AOFOS 開著時**不跑**：開始前看到 AOFOS 在執行就直接結束；跑到一半 AOFOS 啟動了
    （每 0.5 秒檢查一次「AOFOS 正在執行」鎖，core/memory/run_lock.py）⇒ 立刻取消手上那次呼叫、結束。
    理由：本程式收不到「使用者開口」的取消訊號，留著會讓對話排在它後面（推理後端只有一個推理槽）。
  · 同時只准一個 worker（另一把鎖），避免兩個一起判同一個 DB。
  · 判斷邏輯與 AOFOS 裡的完全同一份（core/memory/judge.py 的 BackgroundJudge），只差：
      閒置門檻＝0（AOFOS 已關，沒有要等的對話）；不載 embedding（省 18 秒與約 800 MB）——
      主題的關鍵字索引照補，向量由 AOFOS 下次開啟後的背景判斷順手補（補完之前向量那一路撈不到這些新主題）。
  · 一段只有在「最後一句已過 judge_segment_close_s」才算講完，所以 AOFOS 剛關時最多等那麼久才開判。
  · 人工審核不在這裡做：判完若有待審的，只提醒去開 start/審核.bat。
  · 連續出錯 3 次（例如後端沒開）⇒ 結束並回 1（.bat 會停住讓你看訊息）。
  · config 的 MEMORY_CONFIG["v4_enabled"] 關閉時不啟動（與 AOFOS 一致：關閉期間不建主題）。
"""
from __future__ import annotations

import os
import sys
import time

from config import MEMORY_CONFIG
from core.llm_client import BackgroundLLM
from core.memory.db import connect
from core.memory.judge import BackgroundJudge
from core.memory.run_lock import RunLock, held_by_other, lock_path
from core.memory.topics import TopicStore


def main(*, db_path: str | None = None, llm=None, cfg_override: dict | None = None,
         poll_s: float = 0.5, out=print) -> int:
    """自測可注入 db_path／llm／cfg_override。回傳 0＝判完或讓路、1＝出錯中止。"""
    cfg = {**MEMORY_CONFIG, "judge_idle_s": 0.0, **(cfg_override or {})}
    db = db_path or cfg["db_path"]
    run = lock_path(db)
    if held_by_other(run):
        out("AOFOS 正在執行：背景判斷由它自己做，本程式不啟動。")
        return 0
    if not cfg["v4_enabled"]:
        out("MEMORY_CONFIG[\"v4_enabled\"] 關閉中（不建主題），本程式不啟動。")
        return 0
    if not os.path.exists(db):
        out(f"記憶檔不存在：{db}")
        return 0
    me = RunLock(run + ".worker")
    if not me.try_acquire():
        out("已經有另一個背景判斷在跑，本程式不啟動。")
        return 0
    judge = BackgroundJudge(llm or BackgroundLLM(params=cfg["judge_params"]), None, cfg)
    cx = connect(db)
    rc, last_print = 0, 0.0
    try:
        st = TopicStore(cx, writable=False)
        out(f"背景判斷開始：待判 {st.pending_count()} 句（AOFOS 一啟動就會自動讓路）")
        judge.set_path(db)
        while True:
            time.sleep(poll_s)
            if held_by_other(run):
                judge.pause()
                out("偵測到 AOFOS 啟動：已取消手上的判斷並結束（沒判完的下次再判，不會重複建）")
                break
            pend = st.pending_count()
            if judge.stats["errors"] >= 3:
                out(f"連續出錯，停止（llama-server 有開嗎？）：{judge.status(pend)}")
                rc = 1
                break
            if pend == 0 and not judge.busy:
                out(f"全部判完：{judge.status(pend)}")
                n_rev = len(TopicStore(cx, writable=False).pending_review())
                if n_rev:
                    out(f"⚠️ 有 {n_rev} 件要取代核心設定的事實在等人工審核：請另外雙擊 start/審核.bat（本程式不做審核）")
                break
            if time.monotonic() - last_print > 10:
                out(judge.status(pend))
                last_print = time.monotonic()
    except KeyboardInterrupt:
        out("中斷：沒判完的下次再判")
    finally:
        judge.close()
        cx.close()
        me.release()
    return rc


if __name__ == "__main__":
    sys.exit(main())

