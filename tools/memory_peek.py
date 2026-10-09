# tools/memory_peek.py
"""
診斷用：印出「模型實際收到的記憶區塊」與日常區最近的原句，不寫入日常區。

用法：python -m tools.memory_peek "上次我們說了甚麼?"
      python -m tools.memory_peek            （只看開場帶入）

⭐ 以測試模式開啟：日常區唯讀、寫入只進測試區，結束時測試區整包丟棄。
   向量關閉（省 18 s 載入）⇒ 下面的 messages 只含開場與「關鍵字那一路」命中的主題。
   主題表另外直接列出（唯讀）。🔺 2026-10-09 準 v3 撈原句淘汰後，messages 不再有原句。準 v3 撈原句若在 config 關了，messages 會是空的——正常。
"""
import sqlite3
import sys

from config import MEMORY_CONFIG
from core.memory.manager import MemoryManager


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    cx = sqlite3.connect(MEMORY_CONFIG["db_path"])
    cx.row_factory = sqlite3.Row
    print("══ 日常區最近 30 句 ══")
    for r in cx.execute("SELECT serial, ts, role, content, situation FROM transcript "
                        "ORDER BY serial DESC LIMIT 30").fetchall()[::-1]:
        print(f'{r["serial"]:>4} {r["ts"][:16]} {r["role"][:4]:<4} {r["content"]}'
              + (f'  ［{r["situation"]}］' if r["situation"] else ""))
    try:
        print("\n══ 準 v4 主題（每件最新版；✗＝已被取代；［鎖］［待審］［駁回］＝審核狀態）══")
        mark = {"locked": "［鎖］", "pending": "［待審］", "rejected": "［駁回］"}
        for r in cx.execute("SELECT t.* FROM topic t JOIN (SELECT root, MAX(id) mid FROM topic "
                            "GROUP BY root) m ON t.id = m.mid ORDER BY t.id DESC LIMIT 30"
                            ).fetchall()[::-1]:
            st = mark.get(r["status"], "") if "status" in r.keys() else ""
            print(f'  {"✗" if r["valid_to"] else " "}#{r["root"]}{st} [{r["type"]}|{r["attr_key"] or "-"}] '
                  f'{r["text"]}  ← serial {r["first_serial"]}–{r["last_serial"]} {(r["event_ts"] or "")[:16]}')
        done = cx.execute("SELECT value FROM meta WHERE key='judged_upto'").fetchone()
        print(f"  （背景判斷已處理到 serial {done[0] if done else 0}）")
    except sqlite3.OperationalError as e:
        print(f"  （還沒有主題表：{e}）")
    try:
        print("\n══ 事件層 ══")
        for r in cx.execute("SELECT * FROM episode ORDER BY id"):
            print(f'  #{r["id"]} serial {r["start_serial"]}–{r["end_serial"]} '
                  f'{r["start_ts"][:16]}～{r["end_ts"][11:16]}  詞={r["keywords"]} 代表={r["reps"]}')
    except sqlite3.OperationalError as e:
        print(f"  （沒有事件層：{e}）")
    cx.close()

    m = MemoryManager(embed_enabled=False)
    m.set_test_mode(True)
    try:
        ctx = m.get_context(query)
        print(f"\n══ 送給模型的 messages（查詢：{query!r}）══")
        for x in ctx:
            print(f'--- role={x["role"]} ---\n{x["content"]}')
    finally:
        # 🔺 只呼叫 compress_on_exit()：測試模式下它會自己丟棄測試區、不碰日常區。
        #    原本先 set_test_mode(False) 再 compress_on_exit()，會把日常區以可寫開啟
        #    ⇒ 檢索索引 catch-up ＋ 事件層整張重建，與檔頭「不寫入日常區」矛盾。
        m.compress_on_exit()


if __name__ == "__main__":
    main()
