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

from config import MEMORY_CONFIG
from core.memory.db import connect
from core.memory.episode import EpisodeIndex, time_bucket
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
    kw.setdefault("embed_enabled", False)   # 向量那一路另有專段測（用假模型）
    return MemoryManager(None, db_path=os.path.join(d, "memory.db"),
                         test_db_path=os.path.join(d, "memory_test.db"), **kw)


def block(ctx: list[dict]) -> str:
    """整個 ⟦記憶⟧ 區塊（沒有就空字串）。"""
    return "\n".join(m["content"] for m in ctx if m["role"] == "system")


def injected(ctx: list[dict]) -> str:
    """只取〔相關記憶〕那一段（檢索結果）；開場的狀態／上一場另外看。"""
    b = block(ctx)
    i = b.find("[相關記憶]")
    return "" if i < 0 else b[i:].replace(MEMORY_CONFIG["block_close"], "").strip()


class FakeModel:
    """假 embedding：同義詞表裡的詞映到同一維，其餘逐字雜湊。只給自測用。"""
    SYN = {"高雄": 0, "港都": 0, "貓": 1, "喵星人": 1}

    def encode(self, texts):
        import numpy as np
        out = []
        for t in ([texts] if isinstance(texts, str) else texts):
            v = np.zeros(64, dtype=np.float32)
            for w, dim in self.SYN.items():
                if w in t:
                    v[dim] += 3.0
            for ch in t:
                v[2 + (ord(ch) % 62)] += 0.05
            out.append(v)
        return np.stack(out)


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
        b0 = block(m2.get_context(""))
        check("新場次開頭帶狀態與上一場（包夾＋自帶說明）",
              b0.startswith(MEMORY_CONFIG["block_open"]) and MEMORY_CONFIG["block_note"] in b0
              and "[狀態] 上次聊天是剛剛" in b0 and "那次最後幾句" in b0, b0[:300])
        check("開場不含日期數字", "2026-" not in b0 and "天前" not in b0)
        ctx = m2.get_context("高雄好玩嗎")
        inj = injected(ctx)
        whole = block(ctx)
        # 使用者那句已是開場「那次你特別說過」的代表句 ⇒ 檢索段只補對方回應、不重複
        check("跨場次撈回（開場＋檢索合計、不重複）",
              whole.count("我上次去高雄玩了三天") == 1 and "高雄的海風很舒服吧" in inj, whole[:400])
        check("附口語時間感（不給日期）", "── 剛剛 ──" in inj and "2026-" not in inj, inj[:120])
        check("注入在尾端", ctx and ctx[-1]["role"] == "system")
        check("送模型的訊息只有 role/content",
              all(set(x) == {"role", "content"} for x in ctx))
        inj2 = block(m2.get_context("Minecraft 好玩"))
        # 那一輪是上一場的最後幾句 ⇒ 出現在開場段、不在檢索段重複（兩處擇一）
        check("情境可被撈回且改印情境行", "［當時情境］使用者切換到 Minecraft" in inj2
              and inj2.count("使用者切換到 Minecraft") == 1, inj2[:400])
        check("無關查詢不注入", injected(m2.get_context("量子色動力學")) == "")
        check("FTS5 語法字不炸", isinstance(m2.get_context('AND "NEAR* OR ) ('), list))

        cx = sqlite3.connect(os.path.join(d, "memory.db"))
        rec = cx.execute("SELECT SUM(recalled) FROM transcript WHERE content LIKE '%高雄%'").fetchone()[0]
        check("命中計數有記", (rec or 0) >= 1, str(rec))
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
        inj = block(m.get_context("台南"))
        check("舊檔補欄位＋回填＋補索引後撈得到", "舊資料裡的台南小吃" in inj, inj[:200])
        check("舊資料的時間是口語、不給日期", "上次聊天是" in inj and "2026-" not in inj, inj[:160])

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
        # ⚠️ Windows 不能刪「還開著」的檔：離開 with 前必須關連線，
        #    否則暫存資料夾清不掉而拋 PermissionError（2026-09-26 Windows 實跑撞到，
        #    Linux 不會發作——同一段測試碼在兩個平台行為不同）。
        m.compress_on_exit()

    with tempfile.TemporaryDirectory() as d:
        print("\n[事件層與開場帶入]")
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
        check("代表句只挑使用者的話", all(r["role"] == "user" for r in
              [dict(x) for x in cx.execute(
                  f"SELECT role FROM transcript WHERE serial IN ({','.join(map(str, last['reps'])) or '0'})")]))
        cx.close()
        m2 = mm(d)
        b1 = block(m2.get_context("嗨"))
        check("開場帶上一場關鍵詞", "[上一場] 聊到：" in b1, b1[:300])
        check("開場時間是口語（很久以前的那場）", "上次聊天是" in b1 and "2026-" not in b1, b1[:120])
        for i in range(3):
            m2.add_turn(f"閒聊{i}", "嗯")
        check("過了開頭幾輪就不再帶上一場", "[上一場]" not in block(m2.get_context("閒聊")))
        m2.compress_on_exit()
        now = datetime(2026, 9, 27, 12, 0, tzinfo=timezone(timedelta(hours=8)))
        check("時間感：昨天／上週／很久",
              (time_bucket("2026-09-26T12:00:00+08:00", now), time_bucket("2026-09-18T12:00:00+08:00", now),
               time_bucket("2026-06-01T12:00:00+08:00", now)) == ("昨天", "上週", "很久沒聊了"))

    with tempfile.TemporaryDirectory() as d:
        print("\n[向量那一路（假模型）＋RRF]")
        import time as _t
        m = mm(d, embed_enabled=True, embed_factory=FakeModel)
        m.add_turn("我上次去高雄玩了三天", "好好喔")
        m.add_turn("我家的貓很胖", "好可愛")
        for i in range(11):
            m.add_turn(f"填充{i}", f"嗯{i}")
        for _ in range(100):
            if m._emb.ready.is_set() and "未補向量 0 句" in m.status_text():
                break
            _t.sleep(0.05)
        check("背景補完向量、狀態看得到積壓為 0", "未補向量 0 句" in m.status_text(), m.status_text())
        inj = injected(m.get_context("港都好玩嗎"))
        check("同義換句（字面零重疊）靠向量撈回", "高雄" in inj, inj[:200])
        inj = injected(m.get_context("喵星人最近怎樣"))
        check("第二組同義詞也撈得回", "貓很胖" in inj, inj[:200])
        m.set_test_mode(True)
        m.add_turn("測試區的一句話", "好")
        m.set_test_mode(False)
        check("有向量背景執行緒時，測試區仍能整包丟棄",
              not os.path.exists(os.path.join(d, "memory_test.db")))
        m.compress_on_exit()
        m3 = mm(d, embed_enabled=True, embed_factory=lambda: (_ for _ in ()).throw(ImportError("假裝沒裝")))
        for _ in range(40):
            if m3._emb.failed:
                break
            _t.sleep(0.05)
        check("向量套件缺失 ⇒ 退回字面檢索、不擋啟動", "向量停用" in m3.status_text()
              and "高雄" in injected(m3.get_context("高雄")), m3.status_text())
        m3.compress_on_exit()

    print("\n" + ("全部通過" if not _fail else f"❌ {_fail} 項失敗"))
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
