# core/memory/selftest.py
"""
記憶的儲存層與開場自測：`python -m core.memory.selftest`

🔺 2026-10-09 準 v3 讀取路徑淘汰後改寫：撈原句（RRF、原句向量、WM 不重撈、命中計數）的測試隨程式刪除；
   留下的是準 v4 也依賴的部分——原句寫入、壞資料、寫入失敗、測試模式分流、兩條執行緒同時寫、
   跨執行緒誤用、事件層與開場（上次聊天多久以前＋上一場關鍵詞）。主題（判斷、檢索）在 selftest_v4.py。

全部在暫存資料夾跑，不碰 data/。
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone

from config import MEMORY_CONFIG
from core.memory.bigram import bigrams, check_fts5
from core.memory.db import connect
from core.memory.episode import EpisodeIndex, time_bucket
from core.memory.manager import MemoryManager
from core.memory.pre_write import PreWrite

_fail = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _fail
    print(("  ✅ " if ok else "  ❌ ") + name + (f"　— {detail}" if detail and not ok else ""))
    if not ok:
        _fail += 1


def open_handles(d: str) -> list[str]:
    """本行程還開著、指向 d 底下的檔案（Linux 用 /proc 看；其他平台回空＝不檢查）。

    🔴 存在理由：Windows 不能刪開著的檔，Linux 可以 ⇒ 連線漏關在 Linux 自測永遠看不出來
       （2026-09-30 Windows 實跑兩支自測都在清暫存資料夾時炸）。這裡在 Linux 上把同一件事變成可見。
    """
    import gc
    gc.collect()
    fd_dir = "/proc/self/fd"
    if not os.path.isdir(fd_dir):
        return []
    real = os.path.realpath(d)
    out = []
    for fd in os.listdir(fd_dir):
        try:
            t = os.readlink(os.path.join(fd_dir, fd))
        except OSError:
            continue
        if t.startswith(real):
            out.append(os.path.basename(t))
    return sorted(out)


def mm(d: str, **kw) -> MemoryManager:
    # 只測儲存與開場：主題（背景判斷）關掉，向量不載
    kw.setdefault("embed_enabled", False)
    kw.setdefault("cfg_override", {"v4_enabled": False})
    return MemoryManager(None, db_path=os.path.join(d, "memory.db"),
                         test_db_path=os.path.join(d, "memory_test.db"), **kw)


def block(ctx: list[dict]) -> str:
    """整個 ⟦記憶⟧ 區塊（沒有就空字串）。只認以 block_open 開頭的那則（固定說明不算）。"""
    return "\n".join(m["content"] for m in ctx
                     if m["role"] == "system" and m["content"].startswith(MEMORY_CONFIG["block_open"]))


def count(path: str) -> int:
    cx = sqlite3.connect(path)
    try:
        return cx.execute("SELECT COUNT(*) FROM transcript").fetchone()[0]
    finally:
        cx.close()


def main() -> int:  # noqa: C901
    print(f"SQLite {sqlite3.sqlite_version}／Python {sys.version.split()[0]}")

    print("\n[依賴]")
    try:
        check_fts5()
        check("FTS5 可用且中文 bigram 查得到", True)
    except RuntimeError as e:
        check("FTS5", False, str(e))
        return 1

    print("\n[切詞]")
    check("中文兩字組", bigrams("去高雄玩") == "去高 高雄 雄玩", bigrams("去高雄玩"))
    check("全形英數正規化＋混排", bigrams("ＬＯＬ好玩！") == "lol 好玩", bigrams("ＬＯＬ好玩！"))
    check("單字段保留", bigrams("貓。") == "貓", bigrams("貓。"))
    check("純標點 → 空", bigrams("……！？") == "")

    with tempfile.TemporaryDirectory() as d:
        print("\n[正常路徑與開場]")
        m = mm(d)
        m.add_turn("我上次去高雄玩了三天", "高雄的海風很舒服吧", source="user_text", kind="utterance")
        m.add_turn("", "你在打 Minecraft 耶", source="window", kind="observation",
                   situation="使用者切換到 Minecraft")
        check("第一場沒有上一場 ⇒ 不注入開場", block(m.get_context("嗨")) == "")
        m.compress_on_exit()
        cx = sqlite3.connect(os.path.join(d, "memory.db"))
        check("原句落地、clean 欄已寫", cx.execute("SELECT COUNT(*) FROM transcript").fetchone()[0] == 4
              and cx.execute("SELECT COUNT(*) FROM transcript WHERE clean IS NULL").fetchone()[0] == 0)
        cx.close()

        m2 = mm(d)   # 新場次
        ctx = m2.get_context("")
        b0 = block(ctx)
        check("新場次開頭帶狀態（包夾）",
              b0.startswith(MEMORY_CONFIG["block_open"]) and b0.endswith(MEMORY_CONFIG["block_close"])
              and "[狀態] 上次聊天是剛剛" in b0 and "那次聊了大約" in b0, b0[:300])
        if MEMORY_CONFIG["note_position"] == "fixed":
            fixed = ctx[0]["content"] if ctx and ctx[0]["role"] == "system" else ""
            check("固定說明：在最前面、區塊裡不再帶 block_note",
                  MEMORY_CONFIG["block_open"] in fixed and not fixed.startswith(MEMORY_CONFIG["block_open"])
                  and MEMORY_CONFIG["block_note"] not in b0, fixed[:120])
            check("固定說明：每輪逐字相同（快取前綴不變）",
                  m2.get_context("")[0] == ctx[0] and m2.get_context("隨便問")[0] == ctx[0])
        else:
            check("區塊自帶說明（inline）", MEMORY_CONFIG["block_note"] in b0, b0[:300])
        mi = mm(d, cfg_override={"v4_enabled": False, "note_position": "inline"})
        bi = block(mi.get_context(""))
        check("inline 模式＝舊形狀（說明在區塊第二行、沒有固定說明）",
              bi.split("\n")[1] == MEMORY_CONFIG["block_note"]
              and not any(MEMORY_CONFIG["block_open"] in x["content"] and not x["content"].startswith(MEMORY_CONFIG["block_open"])
                          for x in mi.get_context("")), bi[:200])
        mi.compress_on_exit()
        check("開場不帶原句（代表句／最後幾句已淘汰）",
              "那次最後幾句" not in b0 and "那次你特別說過" not in b0 and "我上次去高雄" not in b0, b0)
        check("開場不含日期數字", "2026-" not in b0 and "天前" not in b0)
        check("注入在尾端、送模型的訊息只有 role/content",
              ctx and ctx[-1]["role"] == "system" and all(set(x) == {"role", "content"} for x in ctx))

        print("\n[測試模式：讀取共用、寫入分流]")
        daily = os.path.join(d, "memory.db")
        before = count(daily)
        m2.set_test_mode(True)
        check("WM 清空", m2.working_memory == [])
        m2.add_turn("測試區才有的句子：藍莓鬆餅", "好")
        check("開場仍讀日常區", "[狀態]" in block(m2.get_context("")))
        check("測試區有寫入", count(os.path.join(d, "memory_test.db")) == 2)
        check("日常區整輪靜止（筆數不變）", count(daily) == before)
        m2.set_test_mode(False)
        check("關閉後測試區整包丟棄",
              not any(os.path.exists(os.path.join(d, "memory_test.db" + s)) for s in ("", "-wal", "-shm")))
        m2.compress_on_exit()

    with tempfile.TemporaryDirectory() as d:
        print("\n[壞資料]")
        p = os.path.join(d, "memory.db")
        old = sqlite3.connect(p)
        old.executescript("""
            CREATE TABLE transcript (serial INTEGER PRIMARY KEY, ts TEXT NOT NULL, role TEXT NOT NULL,
              content TEXT NOT NULL, hash TEXT NOT NULL, source TEXT, kind TEXT, situation TEXT,
              recalled INTEGER NOT NULL DEFAULT 0, retired INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
            INSERT INTO meta VALUES('next_serial',3),('landed_upto',2),('compressed_upto',0);
            INSERT INTO transcript(serial,ts,role,content,hash) VALUES
              (1,'2026-09-01T10:00:00+08:00','user','舊資料裡的台南小吃','x'),
              (2,'2026-09-01T10:00:00+08:00','assistant','牛肉湯','y');
        """)
        old.commit(); old.close()
        m = mm(d)
        b = block(m.get_context("台南"))
        check("舊檔補欄位後照常運作、開場時間是口語", "上次聊天是" in b and "2026-" not in b, b[:160])
        cx = connect(p)
        pw = PreWrite(cx)
        pw.append([{"role": "user", "content": "好的", "ts": "2026-09-25T00:00:00+08:00"}])
        cx.execute("UPDATE pre_write SET content='被改過'")
        cx.close()
        try:
            m.add_turn("竄改之後的一輪", "嗯")
            check("預寫區 hash 不符時不崩潰", True)
        except Exception as e:  # noqa: BLE001
            check("預寫區 hash 不符時不崩潰", False, repr(e))
        m.compress_on_exit()

    with tempfile.TemporaryDirectory() as d:
        print("\n[寫入失敗]")
        os.makedirs(os.path.join(d, "memory.db"))   # 路徑被資料夾佔住 → 開不了 DB
        m = mm(d)
        try:
            m.add_turn("存不進去的一句", "嗯")
            ctx = m.get_context("存不進去")
            check("寫入失敗不中斷對話", True)
            check("沒寫進去的留在 WM", len([t for t in m.working_memory if "serial" not in t]) == 2)
            check("WM 照常進 context", any(x["content"] == "存不進去的一句" for x in ctx))
        except Exception as e:  # noqa: BLE001
            check("寫入失敗不中斷對話", False, repr(e))

    with tempfile.TemporaryDirectory() as d:
        print("\n[兩條執行緒同時寫同一個 DB]")
        errs: list[str] = []

        def worker(tag: str) -> None:
            try:
                w = mm(d)
                for i in range(40):
                    w.add_turn(f"{tag}說的第{i}句", f"回{tag}{i}")
                w.compress_on_exit()
            except Exception as e:  # noqa: BLE001
                errs.append(repr(e))

        ts = [threading.Thread(target=worker, args=(t,)) for t in ("甲", "乙")]
        [t.start() for t in ts]; [t.join() for t in ts]
        check("無例外", not errs, "; ".join(errs))
        n = count(os.path.join(d, "memory.db"))
        check("160 則全部落地（無重複、無遺漏）", n == 160, str(n))

        print("\n[跨執行緒誤用要大聲]")
        m = mm(d)
        m.get_context("甲")   # 主執行緒開了連線
        m._opening_cache = None
        out: list = []
        th = threading.Thread(target=lambda: out.append(m.get_context("甲說的")))
        th.start(); th.join()
        check("別的執行緒呼叫不崩潰、回傳不含注入（錯誤已記 log）", out and block(out[0]) == "")
        m.compress_on_exit()

    with tempfile.TemporaryDirectory() as d:
        print("\n[事件層與開場]")
        import core.memory.manager as mgr
        real_now = mgr._now_iso
        base = datetime(2026, 9, 20, 20, 0, tzinfo=timezone(timedelta(hours=8)))
        clock = [base]
        mgr._now_iso = lambda: clock[0].isoformat(timespec="seconds")
        try:
            m = mm(d)
            for i, (u, a) in enumerate([("今天去高雄吃了牛肉湯", "高雄的牛肉湯很有名"),
                                        ("牛肉湯要配飯", "嗯嗯")]):
                clock[0] = base + timedelta(minutes=i)
                m.add_turn(u, a)
            clock[0] = base + timedelta(hours=3)          # 相隔 3 小時 ⇒ 新的一段
            for i in range(5):
                clock[0] = base + timedelta(hours=3, minutes=i)
                m.add_turn(f"在寫記憶模塊第{i}次", f"加油{i}")
            m.compress_on_exit()
        finally:
            mgr._now_iso = real_now
        cx = connect(os.path.join(d, "memory.db"))
        ep = EpisodeIndex(cx)
        n = cx.execute("SELECT COUNT(*) c FROM episode").fetchone()["c"]
        check("依時間間隔切成 2 段", n == 2, str(n))
        last = ep.last_before(10 ** 9)
        check("上一段的關鍵詞抓到話題", any("記憶模塊" in k for k in last["keywords"]),
              str(last["keywords"]))
        cx.close()
        m2 = mm(d)
        b1 = block(m2.get_context("嗨"))
        check("開場帶上一場關鍵詞", "[上一場] 聊到：" in b1 and "記憶模塊" in b1, b1[:300])
        check("開場時間是口語（很久以前的那場）", "上次聊天是" in b1 and "2026-" not in b1, b1[:120])
        for i in range(3):
            m2.add_turn(f"閒聊{i}", "嗯")
        check("過了開頭幾輪就不再帶開場", "[上一場]" not in block(m2.get_context("閒聊")))
        m2.compress_on_exit()
        now = datetime(2026, 9, 27, 12, 0, tzinfo=timezone(timedelta(hours=8)))
        check("時間感：昨天／上週／很久",
              (time_bucket("2026-09-26T12:00:00+08:00", now), time_bucket("2026-09-18T12:00:00+08:00", now),
               time_bucket("2026-06-01T12:00:00+08:00", now)) == ("昨天", "上週", "很久沒聊了"))
        h = open_handles(d)
        check("離場後沒有任何連線還開著（Windows 才刪得掉）", not h, str(h))

    print("\n" + ("全部通過" if not _fail else f"❌ {_fail} 項失敗"))
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())

