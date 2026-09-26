# core/memory/selftest.py
"""
準 v3 最小記憶切片自測：`python -m core.memory.selftest`

全部在暫存資料夾跑，不碰 data/。涵蓋：依賴、正常路徑、壞資料、寫入失敗、
測試模式分流、兩條執行緒同時寫。⚠️ 不測「撈回的東西模型用不用得上」——
那是回憶題（《現況》15之6 量測手段第一版）的事，不是程式自測能回答的。
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone

from core.memory.db import connect
from core.memory.manager import MemoryManager
from core.memory.pre_write import PreWrite
from core.memory.recall import RecallIndex, bigrams, check_fts5, time_ago

_fail = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _fail
    print(("  ✅ " if ok else "  ❌ ") + name + (f"　— {detail}" if detail and not ok else ""))
    if not ok:
        _fail += 1


def mm(d: str, **kw) -> MemoryManager:
    return MemoryManager(None, db_path=os.path.join(d, "memory.db"),
                         test_db_path=os.path.join(d, "memory_test.db"), **kw)


def injected(ctx: list[dict]) -> str:
    return "\n".join(m["content"] for m in ctx if m["role"] == "system")


def main() -> int:
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

    print("\n[時間差]")
    now = datetime(2026, 9, 25, 20, 0, tzinfo=timezone.utc)
    check("3 天前", time_ago((now - timedelta(days=3)).isoformat(), now) == "3 天前")
    check("壞時間不猜", time_ago("昨天", now) == "時間不明")

    with tempfile.TemporaryDirectory() as d:
        print("\n[正常路徑]")
        m = mm(d)
        m.add_turn("我上次去高雄玩了三天", "高雄的海風很舒服吧", source="user_text", kind="utterance")
        for i in range(14):
            m.add_turn(f"第{i}輪閒聊內容", f"回應第{i}輪")
        check("WM 為滑動視窗（≤ 20 則）", len(m.working_memory) == 20, str(len(m.working_memory)))
        m.add_turn("", "你在打 Minecraft 耶", source="window", kind="observation",
                   situation="使用者切換到 Minecraft")
        m.compress_on_exit()

        m2 = mm(d)   # 新場次
        check("新場次不注入摘要（摘要注入已停用）", m2.get_context("") == [])
        ctx = m2.get_context("高雄好玩嗎")
        inj = injected(ctx)
        check("跨場次撈回原句", "我上次去高雄玩了三天" in inj, inj[:200])
        check("帶出鄰域（對方回應）", "高雄的海風很舒服吧" in inj)
        check("附時間差", "剛才" in inj or "分鐘前" in inj, inj[:120])
        check("注入在尾端", ctx and ctx[-1]["role"] == "system")
        check("送模型的訊息只有 role/content",
              all(set(x) == {"role", "content"} for x in ctx))
        inj2 = injected(m2.get_context("Minecraft 好玩"))
        check("情境可被撈回且改印情境行", "［當時情境］使用者切換到 Minecraft" in inj2, inj2[:300])
        check("無關查詢不注入", injected(m2.get_context("量子色動力學")) == "")
        check("FTS5 語法字不炸", isinstance(m2.get_context('AND "NEAR* OR ) ('), list))

        cx = sqlite3.connect(os.path.join(d, "memory.db"))
        rec = cx.execute("SELECT recalled FROM transcript WHERE content='我上次去高雄玩了三天'").fetchone()[0]
        check("命中計數有記", rec >= 1, str(rec))
        check("clean 欄已寫", cx.execute(
            "SELECT COUNT(*) FROM transcript WHERE clean IS NULL").fetchone()[0] == 0)
        cx.close()

        print("\n[WM 內的不重撈]")
        m2.add_turn("我養了一隻叫麻糬的貓", "麻糬好可愛")
        check("剛說過的不會被自己撈回", "麻糬" not in injected(m2.get_context("麻糬今天好嗎")))

        print("\n[測試模式：讀取共用、寫入分流]")
        daily = os.path.join(d, "memory.db")
        cx = sqlite3.connect(daily)
        before = cx.execute("SELECT COUNT(*), SUM(recalled) FROM transcript").fetchone()
        cx.close()
        m2.set_test_mode(True)
        check("WM 清空", m2.working_memory == [])
        m2.add_turn("測試區才有的句子：藍莓鬆餅", "好")
        inj = injected(m2.get_context("高雄 藍莓"))
        check("讀得到日常區", "高雄" in inj, inj[:200])
        for i in range(11):   # 推出 WM（10 輪），否則會被「WM 內不重撈」排除
            m2.add_turn(f"填充第{i}輪", f"好{i}")
        inj = injected(m2.get_context("藍莓鬆餅"))
        check("讀得到測試區", "藍莓鬆餅" in inj, inj[:200])
        cx = sqlite3.connect(daily)
        after = cx.execute("SELECT COUNT(*), SUM(recalled) FROM transcript").fetchone()
        cx.close()
        check("日常區整輪靜止（筆數與命中計數不變）", before == after, f"{before} → {after}")
        m2.set_test_mode(False)
        check("關閉後測試區整包丟棄",
              not any(os.path.exists(os.path.join(d, "memory_test.db" + s)) for s in ("", "-wal", "-shm")))
        check("關閉後撈不到測試區", "藍莓" not in injected(m2.get_context("藍莓鬆餅")))
        m2.compress_on_exit()

    with tempfile.TemporaryDirectory() as d:
        print("\n[壞資料]")
        # 舊檔：沒有 clean 欄
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
        inj = injected(m.get_context("台南"))
        check("舊檔補欄位＋回填＋補索引後撈得到", "舊資料裡的台南小吃" in inj, inj[:200])
        check("舊資料時間差以天計", "天前" in inj, inj[:120])

        # 預寫區被竄改：land 應停在壞列之前，不崩潰
        cx = connect(p)
        pw = PreWrite(cx)
        pw.append([{"role": "user", "content": "好的", "ts": "2026-09-25T00:00:00+08:00"}])
        cx.execute("UPDATE pre_write SET content='被改過'")
        cx.close()
        try:
            m.add_turn("竄改之後的一輪", "嗯")
            check("預寫區 hash 不符時不崩潰", True)
        except Exception as e:
            check("預寫區 hash 不符時不崩潰", False, repr(e))
        m.compress_on_exit()

        # 退役列不被撈
        cx = sqlite3.connect(p)
        cx.execute("UPDATE transcript SET retired=1 WHERE serial=1"); cx.commit(); cx.close()
        m = mm(d)
        check("retired 列不被撈回", "舊資料裡的台南小吃" not in injected(m.get_context("台南小吃")))
        m.compress_on_exit()

    with tempfile.TemporaryDirectory() as d:
        print("\n[寫入失敗]")
        os.makedirs(os.path.join(d, "memory.db"))   # 路徑被資料夾佔住 → 開不了 DB
        m = mm(d)
        try:
            m.add_turn("存不進去的一句", "嗯")
            ctx = m.get_context("存不進去")
            check("寫入與取回失敗都不中斷對話", True)
            check("沒寫進去的留在 WM", len([t for t in m.working_memory if "serial" not in t]) == 2)
            check("WM 照常進 context", any(x["content"] == "存不進去的一句" for x in ctx))
        except Exception as e:
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
        cx = connect(os.path.join(d, "memory.db"))
        n = cx.execute("SELECT COUNT(*) c FROM transcript").fetchone()["c"]
        idx = RecallIndex(cx)
        idx.catch_up()
        nf = cx.execute("SELECT COUNT(*) c FROM recall_fts").fetchone()["c"]
        check("無例外", not errs, "; ".join(errs))
        check("160 則全部落地", n == 160, str(n))
        check("索引 160 則（無重複、無遺漏）", nf == 160, str(nf))
        cx.close()

        print("\n[跨執行緒誤用要大聲]")
        m = mm(d)
        m.get_context("甲")   # 主執行緒開了連線
        out: list = []
        th = threading.Thread(target=lambda: out.append(m.get_context("甲說的")))
        th.start(); th.join()
        check("別的執行緒呼叫不崩潰、回傳不含注入（錯誤已記 log）",
              out and injected(out[0]) == "")

    print("\n" + ("全部通過" if not _fail else f"❌ {_fail} 項失敗"))
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
