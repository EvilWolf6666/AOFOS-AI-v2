# core/memory/db.py
"""
記憶資料庫的連線與結構（SQLite）。

一個檔、多張表：`pre_write`（預寫區佇列）／`transcript`（原文存放區，永久）／
`topic`／`topic_vec`／`topic_fts`（準 v4 主題與其索引，topics.py／topic_index.py 持有）／
`episode`（事件層，衍生）／`meta`（`next_serial`、`landed_upto`＝N、`judged_upto`）。
舊 DB 的 meta 可能殘留 `compressed_upto` 一列（v1），沒人讀。

🔴 **為什麼是同一個 DB 檔**：讓「原文寫進 transcript ＋ 推進 N」落在**同一個交易**裡
   ——分成兩個檔就沒有交易保證，那個「原文寫成功但 N 沒更新」的窗口會回來。
   `serial` 發號同理：取號與寫入同一交易 ⇒ **不會有「號發了但紀錄沒寫」。**

🔴 **為什麼是 SQLite 不是 JSONL**（量測與舊做法見《歸檔》）：交易保證讓「寫到一半的紀錄」
   這個狀態不存在 ⇒ 不必自己做檔尾換行檢查、解析失敗檢查、啟動截尾。

⚠️ **每個執行緒各自 connect()。** Python 的 sqlite3 連線預設不可跨執行緒共用
   （check_same_thread）。producer（主線程）與 consumer（背景）各開各的，
   指向同一個檔——**這正是 WAL 模式設計來支援的形狀**（一個 writer ＋ 多 reader，
   寫入由鎖序列化，不會交錯寫壞）。

多個執行緒各自寫入是安全的：SQLite 交易有原子性，不需要「只由單一寫入者清除」這類繞路。
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

from core.logger import setup_logger

logger = setup_logger("MemoryDB")

# 交易等鎖的上限。⚠️ 正常情況下每個交易都是毫秒級（append 約 0.7 ms），
# 5 秒等於「不可能等到」——⇒ 真的逾時代表有人把長工作包進交易了，
# 那要大聲失敗、不是靜默重試（見本檔末的紀律說明）。
BUSY_TIMEOUT_MS = 5000

SCHEMA = """
-- 預寫區：落盤的暫存佇列。裝的是「待處理」，不是「已保存」。
CREATE TABLE IF NOT EXISTS pre_write (
    serial    INTEGER PRIMARY KEY,
    ts        TEXT    NOT NULL,   -- 事件發生時間，不是寫入時間
    role      TEXT    NOT NULL,
    content   TEXT    NOT NULL,
    hash      TEXT    NOT NULL,
    source    TEXT,               -- 開放值域：只記錄，不得做分支判斷
    kind      TEXT,               -- 封閉值域：utterance / observation
    situation TEXT,               -- 只有 observation 輪次的 user 那則會有
    speaker_id TEXT               -- 準 v4：說話者 ID（邪狼／奧狐／觀眾平台 ID）；NULL＝未知（舊資料）
);

-- 原文存放區：逐字原句的權威副本，永久。不刪，只標 retired。
CREATE TABLE IF NOT EXISTS transcript (
    serial    INTEGER PRIMARY KEY,
    ts        TEXT    NOT NULL,
    role      TEXT    NOT NULL,
    content   TEXT    NOT NULL,   -- 原樣輸出（含表情標籤等），權威副本
    hash      TEXT    NOT NULL,
    source    TEXT,
    kind      TEXT,
    situation TEXT,
    clean     TEXT,               -- 乾淨文字：供檢索與注入（規則見 clean.py）
    recalled  INTEGER NOT NULL DEFAULT 0,  -- 被撈次數計數器
    retired   INTEGER NOT NULL DEFAULT 0,  -- 可標失效；⚠️ 永不刪除列
    speaker_id TEXT                        -- 準 v4：這一句是誰說的；NULL＝未知（欄位加入前的舊列）
);
-- speaker_id 每句一個（《現況》11之2 丁；舊做法 participants 見《歸檔》D0030）。
--    「誰在場」由同一段對話的 speaker_id 收集推出，不另存（兩份會對不上）。
--    舊列不回填：「猜的值」與「記錄到的值」長得一樣，寧可留 NULL。

-- 單一鍵值表：next_serial（下一個可用號）／landed_upto（N）／judged_upto（topics.py 建）
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT    PRIMARY KEY,
    value INTEGER NOT NULL
);
"""

_DEFAULT_META = {"next_serial": 1, "landed_upto": 0}


def connect(path: str) -> sqlite3.Connection:
    """開一個連線並確保結構就緒。**每個執行緒各自呼叫一次。**

    ⚠️ `synchronous=FULL` 是刻意的：WAL 預設的 NORMAL 在斷電時可能丟掉最後幾個
    交易，而預寫區的整個價值就是「裁切 Working Memory 之前原文已經落地」。
    代價是每次寫入不到 1 ms，相對 TTFT 可忽略。
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # 🔴 **isolation_level=None（autocommit）是必要的**，不是風格選擇：
    #    Python sqlite3 的預設會在寫入前自動開一個 **DEFERRED** 交易，而 DEFERRED
    #    **不會在 SELECT 時取寫鎖** ⇒ 兩個執行緒可以同時讀到同一個 next_serial，
    #    然後雙雙嘗試寫入，其中一個炸 UNIQUE constraint。
    #    ⇒ 改為 autocommit，並由 write_txn() 明確 BEGIN IMMEDIATE（見下）。
    cx = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
    cx.row_factory = sqlite3.Row
    cx.execute("PRAGMA journal_mode=WAL")
    cx.execute("PRAGMA synchronous=FULL")
    cx.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    cx.executescript(SCHEMA)
    _migrate(cx)
    with write_txn(cx):
        cx.executemany(
            "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)",
            list(_DEFAULT_META.items()),
        )
    return cx


def _migrate(cx: sqlite3.Connection) -> None:
    """舊檔補欄位。`CREATE TABLE IF NOT EXISTS` 對已存在的表**什麼都不做**，
    所以新加的欄位只會出現在新檔——舊檔要靠這裡補，否則寫入時才炸。

    `transcript.clean`：
    `ALTER TABLE ADD COLUMN` 在 SQLite 是零遷移成本（見上方 SCHEMA 的註解），
    補完後把既有列回填成 `clean = content`——剝除規則未定前兩欄本來就相同，
    ⚠️ 不回填的話舊列的 clean 是 NULL，而「NULL」與「剝完是空字串」是兩件事。
    """
    cols = {r["name"] for r in cx.execute("PRAGMA table_info(transcript)")}
    if "clean" not in cols:
        with write_txn(cx):
            cx.execute("ALTER TABLE transcript ADD COLUMN clean TEXT")
            n = cx.execute(
                "UPDATE transcript SET clean = content WHERE clean IS NULL").rowcount
        logger.warning(f"原文存放區補上 clean 欄位（一次性），回填 {n} 列")

    # speaker_id（預寫區與原文存放區都要有，預寫區是原樣轉寫的上游）。
    # ⚠️ 一次性結構改動（《現況》〈📐 規則〉二「架構上只做一次的事」）：動手前先備份 data/memory.db。
    for table in ("pre_write", "transcript"):
        cols = {r["name"] for r in cx.execute(f"PRAGMA table_info({table})")}
        if "speaker_id" not in cols:
            with write_txn(cx):
                cx.execute(f"ALTER TABLE {table} ADD COLUMN speaker_id TEXT")
            logger.warning(f"{table} 補上 speaker_id 欄位（一次性；舊列為 NULL＝未知）")


@contextmanager
def write_txn(cx: sqlite3.Connection):
    """一個**立刻取得寫鎖**的交易。所有寫入一律經由它。

    🔴 **必須是 IMMEDIATE，不可以是 DEFERRED。**
    DEFERRED（Python sqlite3 的預設）在第一次寫入時才升級成寫鎖，
    **中間的 SELECT 完全不排他** ⇒ 「讀 next_serial → +1 → 寫回」這種
    read-modify-write **不是原子的**：兩個執行緒同時讀到同一個號，
    然後一個寫成功、另一個炸 `UNIQUE constraint failed`。

    ⚠️ **這個缺陷只在併發下發作，單執行緒測試全綠**（自測撞到過：兩條執行緒各寫 40 筆，
    只進去 47 筆）。SQLite 序列化的是**寫**，DEFERRED 交易的**讀**不排他。

    ⚠️ **交易只能包住 DB 操作**——LLM 呼叫、網路、睡眠一律在外面（見本檔末）。
    """
    cx.execute("BEGIN IMMEDIATE")
    try:
        yield cx
    except Exception:
        cx.execute("ROLLBACK")
        raise
    cx.execute("COMMIT")


def get_meta(cx: sqlite3.Connection, key: str) -> int:
    row = cx.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row is None:
        # 缺鍵一律大聲：它代表 schema 與程式碼對不上，靜默補 0 會讓
        # 「從沒處理過」與「鍵不見了」長得一模一樣。
        raise KeyError(f"meta 缺少鍵：{key}")
    return row["value"]


def set_meta(cx: sqlite3.Connection, key: str, value: int) -> None:
    """⚠️ **不自己開交易**——呼叫端負責，這樣它才能與資料寫入放進同一個交易。"""
    cx.execute("UPDATE meta SET value=? WHERE key=?", (value, key))


# ══════════════════════════════════════════════════════════════
# 🔴 交易紀律（本套件唯一靠結構而非靠自覺維持的東西）
# ══════════════════════════════════════════════════════════════
#
# **交易只能包住 DB 操作。LLM 呼叫、網路、睡眠一律在交易之外。**
#
# ⚠️ 違反的症狀很具體：背景判斷的 LLM 呼叫每段 5～25 秒，
#    若它被包在交易裡，producer 每一次 append 都要等那把鎖——**而前台會直接感覺到**。
#
# ⭐ **本套件不靠註解維持這條，靠結構**：
#    `pre_write`、`transcript`、`topics` 的公開方法**一律自己開自己關交易**（用 write_txn），
#    不對外回傳任何可持有的 transaction 物件或 cursor。
#    ⇒ **呼叫端在結構上拿不到「把 LLM 包進交易」的機會。**
#
# ⚠️ 因此 consumer 的迴圈固定切三段（現行實例：core/memory/judge.py 的背景判斷）：
#      ① 開交易讀待處理 → 關交易
#      ② **交易外**呼叫 LLM（那 5～25 秒）
#      ③ 開交易寫結果 ＋ 推進 checkpoint（judged_upto）→ 關交易


# ══════════════════════════════════════════════════════════════
# 自測：python -m core.memory.db --selftest
# ══════════════════════════════════════════════════════════════
#
# 🔴 **每一項都用一個已知該被抓到的樣本撞一次**（《現況》20之3）——
#    只印「通過」只證明射程內沒問題，沒證明射程夠寬。
# ⚠️ **一律在暫存目錄造假 DB，不碰任何真檔**。
# 📌 本自測涵蓋整個儲存層（pre_write ＋ transcript ＋ meta），因為它們是同一個 DB。

def _selftest() -> int:  # noqa: C901
    import shutil, tempfile, threading, time

    from core.memory.pre_write import PreWrite, content_hash
    from core.memory.transcript import Transcript

    fails: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(("  ✅ " if ok else "  🔴 ") + name + (f"　{detail}" if detail else ""))
        if not ok:
            fails.append(name)

    def turn(role: str, content: str, **extra) -> dict:
        return {"role": role, "content": content, "ts": "2026-09-13T17:00:00", **extra}

    d = tempfile.mkdtemp(prefix="memdb_selftest_")
    try:
        path = os.path.join(d, "memory.db")
        cx = connect(path)
        w, t = PreWrite(cx), Transcript(cx)

        print("\n── 預寫區：取號與讀回 ──")
        s1 = w.append([turn("user", "晚上好呀", source="user_text", kind="utterance"),
                       turn("assistant", "嗨，今天過得怎樣", source="user_text",
                            kind="utterance")])
        check("取號從 1 開始且連續", s1 == [1, 2], f"{s1}")
        got, halt = w.read_pending()
        check("讀得回來、無中止", len(got) == 2 and halt is None)
        check("額外欄位原樣轉寫", got[0]["source"] == "user_text")
        check("沒帶的欄位是 None 不是空字串", got[0]["situation"] is None)

        s2 = w.append([turn("user", "", source="window", kind="observation",
                            situation="邪狼剛切到 Minecraft。")])
        check("續號不重來", s2 == [3], f"{s2}")
        got, _ = w.read_pending(after_serial=2)
        check("🔴 回歸：續讀不被誤判成 serial 不遞增",
              [r["serial"] for r in got] == [3],
              "（JSONL 版踩過：prev 與 after_serial 混用，只在第二輪以後發作）")
        check("observation 的 situation 有進來",
              got[0]["situation"].startswith("邪狼剛切到"))

        print("\n── 🔴 撞擊：交易原子性（JSONL 版那三道檢查消失的根據）──")
        before = w.backlog()
        try:
            with write_txn(cx):
                cx.execute("INSERT INTO pre_write(serial,ts,role,content,hash) "
                           "VALUES(99,'','user','會被回滾','x')")
                raise RuntimeError("模擬交易中途失敗")
        except RuntimeError:
            pass
        check("中途失敗 ⇒ 整個交易回滾，一筆都沒進去", w.backlog() == before,
              f"{before} → {w.backlog()}　⇒ **「寫到一半的紀錄」這個狀態不存在**")

        print("\n── 原文存放區：落地與 N ──")
        pend, _ = w.read_pending()
        r = t.land(pend)
        check("三筆全部寫入", r["written"] == 3 and r["skipped"] == 0)
        check("N 推進到 3", t.landed_upto == 3, f"N={t.landed_upto}")

        r = t.land(pend)
        check("重跑同一批 ⇒ 全部跳過（冪等）", r["skipped"] == 3 and r["written"] == 0)

        print("\n── 🔴 撞擊：hash 不同 ⇒ 修補並大聲 ──")
        bad = [dict(pend[0], content="被改過的內容")]
        r = t.land(bad)
        check("hash 不同 ⇒ 覆寫修補（上方應有 ERROR）", r["repaired"] == 1)

        print("\n── 🔴 撞擊：read_pending 的 hash 檢查 ──")
        with write_txn(cx):
            cx.execute("UPDATE pre_write SET content='偷偷換掉' WHERE serial=2")
        got, halt = w.read_pending()
        check("停在壞紀錄之前，不跳過", len(got) == 1 and halt is not None, f"{halt}")
        with write_txn(cx):  # 還原，後面還要用
            cx.execute("UPDATE pre_write SET content=? WHERE serial=2",
                       (pend[1]["content"],))

        print("\n── 預寫區清除 ──")
        removed = w.purge_upto(t.landed_upto)
        check("清掉已落地的 3 筆", removed == 3 and w.backlog() == 0)
        check("原文存放區不受影響", t.count() == 3, f"{t.count()} 筆")

        print("\n── 🔴 撞擊：兩個執行緒同時寫（producer ＋ consumer 的形狀）──")
        errors: list[str] = []

        def hammer(tag: str, n: int) -> None:
            c = connect(path)          # ⚠️ 每個執行緒各自 connect
            p = PreWrite(c)
            try:
                for i in range(n):
                    p.append([turn("user", f"{tag}-{i}")])
                    time.sleep(0)
            except Exception as e:      # noqa: BLE001
                errors.append(f"{tag}: {type(e).__name__}: {e}")
            finally:
                c.close()

        ths = [threading.Thread(target=hammer, args=(f"t{k}", 40)) for k in range(2)]
        for x in ths:
            x.start()
        for x in ths:
            x.join()
        check("兩執行緒各寫 40 筆，無例外（無 SQLITE_BUSY 失敗）",
              not errors, "；".join(errors)[:120])
        got, halt = w.read_pending()
        serials = [r["serial"] for r in got]
        check("80 筆全到、serial 無重複無亂序",
              len(serials) == 80 and serials == sorted(set(serials)),
              f"{len(serials)} 筆")
        check("讀取端四道檢查全過（hash ＋ 遞增）", halt is None)

        cx.close()

        print("\n── 重啟：狀態從 DB 續得回來 ──")
        cx2 = connect(path)
        w2, t2 = PreWrite(cx2), Transcript(cx2)
        check("next_serial 續得回來", get_meta(cx2, "next_serial") == 84,
              f"next_serial={get_meta(cx2, 'next_serial')}")
        check("N 續得回來", t2.landed_upto == 3, f"N={t2.landed_upto}")
        check("積壓筆數續得回來", w2.backlog() == 80)
        cx2.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print()
    if fails:
        print(f"🔴 自測未通過：{len(fails)} 項 —— " + "、".join(fails))
        return 1
    print("✅ 自測全部通過（交易原子性、冪等、修補、併發、重啟續跑皆撞過）")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print("用法：python -m core.memory.db --selftest")
