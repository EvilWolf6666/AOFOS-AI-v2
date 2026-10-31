# tools/memory_peek.py
"""
診斷用：印出「模型實際收到的記憶區塊」與日常區最近的原句，不寫入日常區。

用法：python -m tools.memory_peek "上次我們說了甚麼?"
      python -m tools.memory_peek            （只看開場帶入）

⭐ 以測試模式開啟：日常區唯讀、寫入只進測試區，結束時測試區整包丟棄。
   向量關閉（省 18 s 載入）⇒ 下面的 messages 只含開場與「關鍵字那一路」命中的主題。
   主題表另外直接列出（唯讀）。`v4_enabled` 關閉時不撈主題，只會看到開場
   （第一場沒有上一場時連開場都沒有）——正常。
⭐ 日常區以唯讀開啟（mode=ro）：檔案不存在時只印提示，不會替它建出空檔。
⭐ 印的是正式路徑實際送出的形狀（build_turn()：記憶區塊依 block_position 可能在最後一則 user 裡）。
"""
import os
import sqlite3
import sys

from config import MEMORY_CONFIG
from core.memory.manager import MemoryManager


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    path = MEMORY_CONFIG["db_path"]
    if os.path.exists(path):
        _print_daily(path)
    else:
        print(f"══ 日常區不存在：{path}（正式記憶為空；不建檔）══")
    _print_messages(query)


def _print_daily(path: str) -> None:
    cx = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
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


def _print_messages(query: str) -> None:
    m = MemoryManager(embed_enabled=False)
    m.set_test_mode(True)
    try:
        ctx, user_content = m.build_turn(query)
        print(f"\n══ 送給模型的 messages（人格之後；查詢：{query!r}）══")
        for x in ctx + [{"role": "user", "content": user_content}]:
            print(f'--- role={x["role"]} ---\n{x["content"]}')
    finally:
        # 只呼叫 close()：測試模式下它會自己丟棄測試區、不碰日常區。
        #    ⚠️ 先 set_test_mode(False) 會把日常區以可寫開啟並重建事件層，違反「不寫入日常區」。
        m.close()


if __name__ == "__main__":
    main()
