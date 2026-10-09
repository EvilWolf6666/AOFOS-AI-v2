# tools/memory_review.py
"""
人工審核（2026-10-09 邪狼裁，《現況》〈🔒 隔離區〉準 v4〈鎖定區與人工審核〉）。

用法：雙擊 start/審核.bat（或 python -m tools.memory_review）。
  逐筆列出待審的東西：輸入 yes（或 y）通過、no（或 n）駁回、直接 Enter 跳過（維持待審，下次再審）。
  全部過完自動結束；一路 Enter 就是全部跳過然後結束。

審核來源（SOURCES）：目前只有「記憶：要取代鎖定事實的新事實」。日後要擴到整個專案，
在 SOURCES 加一個 (名稱, 函式) 即可——函式自己列項目、自己寫入，這裡只管問答。

⭐ AOFOS 開著也能用：寫入走 write_txn（BEGIN IMMEDIATE），與背景判斷不會互相蓋掉。
⚠️ 會寫正式記憶檔（只改待審那幾列的 status 與被取代者的 valid_to），不刪任何列。
"""
from __future__ import annotations

import os
import sys

from config import MEMORY_CONFIG
from core.memory.db import connect
from core.memory.topics import TopicStore


def _ask(prompt: str, ask) -> str:
    """回 'yes'／'no'／''（跳過）。y／n 等同 yes／no；其他輸入再問一次，再錯當跳過。"""
    for _ in range(2):
        a = ask(prompt).strip().lower()
        a = {"y": "yes", "n": "no"}.get(a, a)
        if a in ("yes", "no", ""):
            return a
        print("  只收 yes／y、no／n、Enter")
    return ""


def review_memory(ask=input, out=print) -> tuple[int, int, int]:
    """記憶的待審事實。回傳（通過, 駁回, 跳過）。"""
    path = MEMORY_CONFIG["db_path"]
    if not os.path.exists(path):
        out("（記憶檔不存在，沒有待審）")
        return 0, 0, 0
    names = MEMORY_CONFIG["speaker_names"]
    labels = MEMORY_CONFIG["speaker_labels"]
    cx = connect(path)
    try:
        cols = {r[1] for r in cx.execute("PRAGMA table_info(topic)")}
        if "status" not in cols:
            out("（主題表還沒有審核欄位——新版 AOFOS 跑過一次就會有；目前沒有待審）")
            return 0, 0, 0
        st = TopicStore(cx, writable=False)
        items = st.pending_review()
        ok = no = skip = 0
        for n, t in enumerate(items, 1):
            out(f"\n[{n}/{len(items)}] 新：#{t['id']} [{t['type']}|{t['attr_key'] or '-'}] {t['text']}")
            if t["replaces"] is not None:
                old = [o for o in st.latest() if o["root"] == t["replaces"]]
                out(f"  要取代鎖定的：#{t['replaces']} {old[0]['text'] if old else '（找不到，可能已被取代）'}")
            out(f"  原句（serial {t['first_serial']}–{t['last_serial']}）：")
            for r in st.source_rows(t):
                who = names.get(r.get("speaker_id") or "", "") or labels.get(r["role"], r["role"])
                out(f"    {who}：{r.get('clean') or r.get('content') or ''}")
            a = _ask("  y 通過／n 駁回／Enter 跳過 > ", ask)
            if a == "yes":
                st.approve(t["id"]); ok += 1
            elif a == "no":
                st.reject(t["id"]); no += 1
            else:
                skip += 1
        if not items:
            out("（記憶：沒有待審）")
        return ok, no, skip
    finally:
        cx.close()


SOURCES = [("記憶", review_memory)]


def main(ask=input, out=print) -> int:
    for name, fn in SOURCES:
        out(f"══ {name} ══")
        ok, no, skip = fn(ask, out)
        if ok or no or skip:
            out(f"  → 通過 {ok}、駁回 {no}、跳過 {skip}（跳過的維持待審）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
