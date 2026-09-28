# tools/memory_peek.py
"""
診斷用：印出「模型實際收到的記憶區塊」與日常區最近的原句，不寫入日常區。

用法：python -m tools.memory_peek "上次我們說了甚麼?"
      python -m tools.memory_peek            （只看開場帶入）

⭐ 以測試模式開啟：日常區唯讀、寫入只進測試區，結束時測試區整包丟棄。
   向量那一路關閉（省 18 s 載入）⇒ 看到的是「字面檢索」那一路的結果。
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

