# core/memory/topics.py
"""
準 v4 的主題表（《現況》〈🔒 隔離區〉記憶模塊〈準 v4〉）。

它是什麼：**原句存放區（transcript）的索引**，不是資料本身。
  · 每一列＝一件事的一句主題句（主詞寫明），帶「指回原句」的 serial 範圍。
  · 原句永遠在 transcript，一筆不刪；日常主題壞掉可以從原句重建（重跑背景判斷）。
    例外是核心設定與審核結果（status／replaces），見下方〈鎖定與人工審核〉。
    ⇒ 寫入端做的歸納是**可重建的衍生物**，不是不可逆的結論
      （《現況》〈📐 規則〉三「歸納放在讀取端」要防的是不可逆，這裡沒有違反）。

版本（重寫）：
  · 一件事第一次被記下來＝一列，`root = 自己的 id`、`parent = NULL`。
  · 之後依當下狀況重寫的新版本＝另一列，`parent = 上一版 id`、`root = 同一個源頭`。
  · 檢索只看每個 root 的**最新一版**；重寫的輸入＝源頭原句＋最新一版
    （只拿最新版會讓記錯的細節一路傳下去——轉述偏移）。
  ⚠️ 本檔只提供版本的資料形狀；「重寫」本身（第 4 步）尚未實作。

時間盲區（「上週在上海」vs「這週搬到北京」）：
  · 狀態類的主題帶 `attr_key`（例：住地）。同一個 attr_key 寫進新的一件事時，
    **機械地**把舊的那件事（每個版本）填 `valid_to`、`superseded_by`——不經 LLM。
  · 判斷「是不是同一件事的新狀態」需要語意，那一步已經搭在標主題那次呼叫裡（產出 attr_key）。
  · 檢索預設跳過 `valid_to IS NOT NULL` 的列；原句照留。

鎖定與人工審核（《現況》〈🔒 隔離區〉準 v4〈鎖定區與人工審核〉）：
  · `status`：normal（日常）／locked（核心設定）／pending（待審）／rejected（駁回）。
  · 人工審核只管「要不要進核心設定」：背景判斷寫進來的新事實若要**機械取代一件核心設定**
    （同屬性鍵），不直接取代；新的那件寫成 pending、`replaces` 指向被它挑戰的鎖定 root，
    等人工審核（tools/memory_review.py）。通過＝取代舊的、成為核心設定；駁回＝不改核心設定。
  · 駁回的事日後要當日常記憶撈（加註「有人提過、未採用」，同時帶入現行核心設定，矛盾時核心優先）。
    ⏸ 這段讀取行為等有鎖定條目時才實作；**現行**檢索只看 normal／locked，
    pending、rejected 不注入（原句照留，駁回的列也不刪）。
  ⚠️ 目前沒有任何路徑會產生 locked（世界觀灌入未做，《現況》17之1）⇒ 欄位先在、機制先不會被觸發。
  ⚠️ 核心設定（locked）與審核結果（status／replaces）**不能從原句重建**：前者日後由世界觀灌入、
     不一定來自對話，後者是人的決定。重建日常主題（重跑背景判斷）前要先把它們另外留下，
     否則會跟著一起消失。重寫版本（add_version）沿用上一版的 status／replaces。

判斷來源（judge_model／judge_prompt）：
  · 背景判斷寫進來的源頭列記下「哪顆模型、哪版規則檔」判出來的（模型 alias、規則檔內容雜湊）。
  · 用途：換判斷模型或改規則檔後，只挑舊來源產的重判，不必整張重來。
  · NULL＝不是背景判斷寫的（自測、日後的世界觀灌入），或是加欄位之前寫的舊列。
    重寫版本不填（它是重寫產的、不是判斷產的；來源看它的 root 那一列）。

🔴 **交易紀律同 db.py**：公開方法自己開自己關交易，不對外回傳 cursor。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from core.logger import setup_logger
from core.memory.db import get_meta, set_meta, write_txn

logger = setup_logger("MemoryTopics")

_JUDGED_KEY = "judged_upto"   # 背景判斷處理到哪個 serial（含「判了不建」的）

SCHEMA = """
CREATE TABLE IF NOT EXISTS topic (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    root          INTEGER,               -- 源頭那一列的 id（第一版＝自己）
    parent        INTEGER,               -- 上一版的 id；第一版為 NULL
    text          TEXT    NOT NULL,      -- 主題句（主詞寫明，人稱換成名字）
    type          TEXT,                  -- 類型（封閉值域，來自 config；未定前為佔位值）
    attr_key      TEXT,                  -- 屬性鍵：狀態類事實才有（住地、工作…）；其他為 NULL
    first_serial  INTEGER NOT NULL,      -- 指回原句範圍（含）
    last_serial   INTEGER NOT NULL,
    event_ts      TEXT,                  -- 事件發生時間（取 last_serial 那句的 ts），存絕對時間
    created_at    TEXT    NOT NULL,      -- 這一列寫進來的時間
    speakers      TEXT,                  -- 這段原句裡出現過的 speaker_id（逗號分隔，排序過）
    valid_to      TEXT,                  -- 被同 attr_key 新事實取代的時間；NULL＝仍有效
    superseded_by INTEGER,               -- 取代它的那件事的 root
    retired       INTEGER NOT NULL DEFAULT 0,
    status        TEXT    NOT NULL DEFAULT 'normal',  -- normal／locked／pending／rejected
    replaces      INTEGER,               -- pending 才有：它要取代的鎖定 root
    judge_model   TEXT,                  -- 判出這件事的模型 alias；NULL＝非背景判斷寫入
    judge_prompt  TEXT                   -- 當時規則檔的內容雜湊（judge.prompt_digest）
);
CREATE INDEX IF NOT EXISTS topic_root ON topic(root);
CREATE INDEX IF NOT EXISTS topic_attr ON topic(attr_key) WHERE attr_key IS NOT NULL;
"""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


_LIVE = "('normal','locked')"   # 會被檢索的狀態


def ensure_schema(cx: sqlite3.Connection) -> None:
    cx.executescript(SCHEMA)
    # 舊表補欄位：沒有 status／replaces 的補上（既有列一律 normal，行為不變）；
    # 沒有 judge_model／judge_prompt 的補上（既有列為 NULL＝來源不明）。
    # ⚠️ 這是對正式記憶檔的結構改動：第一次跑新版前先備份 data/memory.db。
    cols = {r[1] for r in cx.execute("PRAGMA table_info(topic)")}
    for col, ddl in (("status", "TEXT NOT NULL DEFAULT 'normal'"), ("replaces", "INTEGER"),
                     ("judge_model", "TEXT"), ("judge_prompt", "TEXT")):
        if col not in cols:
            with write_txn(cx):
                cx.execute(f"ALTER TABLE topic ADD COLUMN {col} {ddl}")
            logger.warning(f"主題表補欄位 {col}（舊表遷移，既有列不變）")
    with write_txn(cx):
        cx.execute("INSERT OR IGNORE INTO meta(key, value) VALUES(?, 0)", (_JUDGED_KEY,))


class TopicStore:
    """主題表的存取。**每個執行緒傳入自己的連線。**"""

    def __init__(self, cx: sqlite3.Connection, *, writable: bool = True) -> None:
        self.cx = cx
        self.writable = writable
        if writable:
            ensure_schema(cx)

    def _has_table(self) -> bool:
        return self.cx.execute(
            "SELECT 1 FROM sqlite_master WHERE name='topic'").fetchone() is not None

    def _live_cond(self, alias: str = "t") -> str:
        """檢索條件。唯讀開啟的舊表（施測中的日常區、還沒被新版開過）沒有 status 欄 ⇒ 不篩。"""
        cols = {r[1] for r in self.cx.execute("PRAGMA table_info(topic)")}
        return f"AND {alias}.status IN {_LIVE}" if "status" in cols else ""

    # ── 背景判斷的進度 ─────────────────────────────────────────

    @property
    def judged_upto(self) -> int:
        if not self._has_table():
            return 0
        try:
            return get_meta(self.cx, _JUDGED_KEY)
        except KeyError:
            # 建表與寫入 judged_upto 是先後兩步（ensure_schema）：別條連線剛好夾在中間讀到
            # ＝還沒判過任何一句（Windows 自測撞到：judge_worker 主執行緒讀、判斷執行緒建表）。
            return 0

    def pending_rows(self, limit: int) -> list[dict]:
        """已落地、還沒判斷過的原句（serial > judged_upto），依 serial 排序。"""
        rows = self.cx.execute(
            "SELECT serial, ts, role, content, clean, situation, speaker_id FROM transcript "
            "WHERE serial > ? AND serial <= (SELECT value FROM meta WHERE key='landed_upto') "
            "AND retired = 0 ORDER BY serial LIMIT ?", (self.judged_upto, limit)).fetchall()
        return [dict(r) for r in rows]

    def pending_count(self) -> int:
        """已落地、還沒判斷過的原句數（/status 即時查；主題表還沒建＝全部都還沒判）。"""
        return self.cx.execute(
            "SELECT COUNT(*) c FROM transcript WHERE serial > ? AND retired = 0 AND serial <= "
            "COALESCE((SELECT value FROM meta WHERE key='landed_upto'), 0)",
            (self.judged_upto,)).fetchone()["c"]

    def mark_judged(self, serial: int) -> None:
        """推進 judged_upto。⚠️ 只在該段的主題**已寫入**（或判定不建）之後呼叫。"""
        with write_txn(self.cx):
            if serial > get_meta(self.cx, _JUDGED_KEY):
                set_meta(self.cx, _JUDGED_KEY, serial)

    # ── 寫入 ──────────────────────────────────────────────────

    def add_topics(self, items: list[dict], *, first_serial: int, last_serial: int,
                   event_ts: str, speakers: list[str], judged_upto: int,
                   judge_model: str | None = None, judge_prompt: str | None = None) -> list[int]:
        """寫入一段對話判出的主題，並在**同一個交易**裡：
          ① 同 attr_key 的舊事實標 valid_to／superseded_by（機械取代）
          ② 推進 judged_upto
        ⇒ 「主題寫了但進度沒推（下次重複建）」與「進度推了但主題沒寫（漏建）」都不存在。

        items：[{text, type, attr_key}]；空清單＝這段判了不建，只推進度。
        judge_model／judge_prompt：背景判斷一律要帶（見檔頭〈判斷來源〉）；不帶＝NULL。
        回傳新列的 id。
        ⚠️ 要取代的舊事實裡有**鎖定**的 ⇒ 鎖定的那件不動，新的這件寫成 pending 等人工審核
           （非鎖定的照常取代）。
        """
        now = _now_iso()
        spk = ",".join(sorted({s for s in speakers if s}))
        ids: list[int] = []
        with write_txn(self.cx):
            for it in items:
                cur = self.cx.execute(
                    "INSERT INTO topic(text, type, attr_key, first_serial, last_serial, "
                    "event_ts, created_at, speakers, judge_model, judge_prompt) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (it["text"], it.get("type"), it.get("attr_key") or None,
                     first_serial, last_serial, event_ts, now, spk or None,
                     judge_model, judge_prompt))
                tid = cur.lastrowid
                self.cx.execute("UPDATE topic SET root=? WHERE id=?", (tid, tid))
                ids.append(tid)
                if it.get("attr_key"):
                    # 同屬性鍵、仍有效、而且是「更早發生」的事 ⇒ 被取代（整個 root 的所有版本）
                    rows = self.cx.execute(
                        "SELECT root, MAX(status='locked') locked FROM topic WHERE attr_key=? "
                        f"AND valid_to IS NULL AND status IN {_LIVE} AND root<>? AND last_serial < ? "
                        "GROUP BY root ORDER BY root", (it["attr_key"], tid, first_serial)).fetchall()
                    olds = [r["root"] for r in rows if not r["locked"]]
                    locked = [r["root"] for r in rows if r["locked"]]
                    if locked:
                        self.cx.execute("UPDATE topic SET status='pending', replaces=? WHERE id=?",
                                        (locked[0], tid))
                        logger.warning(f"屬性鍵「{it['attr_key']}」碰到鎖定的 #{locked[0]}："
                                       f"新事實 #{tid} 改為待審，等人工審核（start/審核.bat）")
                    if olds:
                        self.cx.execute(
                            f"UPDATE topic SET valid_to=?, superseded_by=? WHERE root IN "
                            f"({','.join('?' * len(olds))})", [now, tid, *olds])
                        logger.info(f"屬性鍵「{it['attr_key']}」：{len(olds)} 件舊事實標為已被取代")
            if judged_upto > get_meta(self.cx, _JUDGED_KEY):
                set_meta(self.cx, _JUDGED_KEY, judged_upto)
        return ids

    def add_version(self, parent_id: int, text: str) -> int:
        """寫一個重寫版本（第 4 步用）：parent＝上一版、root＝同源頭，其餘欄位沿用。
        ⚠️ status／replaces 一定要沿用：latest() 取每個 root 最新一版，新版若退回預設 normal，
           鎖定的事一被重寫就不再是鎖定（之後同屬性鍵的新說法會直接取代它，不進待審）。
        judge_model／judge_prompt 不沿用（見檔頭〈判斷來源〉）。"""
        with write_txn(self.cx):
            p = self.cx.execute("SELECT * FROM topic WHERE id=?", (parent_id,)).fetchone()
            if p is None:
                raise KeyError(f"topic id={parent_id} 不存在")
            cur = self.cx.execute(
                "INSERT INTO topic(root, parent, text, type, attr_key, first_serial, last_serial, "
                "event_ts, created_at, speakers, valid_to, superseded_by, status, replaces) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (p["root"], parent_id, text, p["type"], p["attr_key"], p["first_serial"],
                 p["last_serial"], p["event_ts"], _now_iso(), p["speakers"],
                 p["valid_to"], p["superseded_by"], p["status"], p["replaces"]))
            return cur.lastrowid

    # ── 讀取 ──────────────────────────────────────────────────

    def latest(self, *, include_superseded: bool = False) -> list[dict]:
        """每個 root 的最新一版（id 最大者）。預設跳過已被取代與已退役的。"""
        if not self._has_table():
            return []
        cond = "" if include_superseded else "AND t.valid_to IS NULL"
        rows = self.cx.execute(
            "SELECT t.* FROM topic t JOIN (SELECT root, MAX(id) mid FROM topic GROUP BY root) m "
            f"ON t.id = m.mid WHERE t.retired = 0 {self._live_cond()} {cond} "
            "ORDER BY t.id").fetchall()
        return [dict(r) for r in rows]

    # ── 人工審核（tools/memory_review.py）────────────────────────

    def pending_review(self) -> list[dict]:
        """待審的事實，舊的先。"""
        if not self._has_table():
            return []
        return [dict(r) for r in self.cx.execute(
            "SELECT * FROM topic WHERE status='pending' ORDER BY id").fetchall()]

    def lock(self, topic_id: int) -> None:
        """把一件事（整個 root）設為鎖定。目前只給自測與日後的世界觀灌入用。"""
        with write_txn(self.cx):
            self.cx.execute("UPDATE topic SET status='locked' WHERE root=(SELECT root FROM topic "
                            "WHERE id=?) AND status='normal'", (topic_id,))

    def approve(self, topic_id: int) -> None:
        """通過：被挑戰的鎖定事實標為被取代，新的這件接手鎖定。同一個交易。"""
        now = _now_iso()
        with write_txn(self.cx):
            t = self.cx.execute("SELECT * FROM topic WHERE id=? AND status='pending'",
                                (topic_id,)).fetchone()
            if t is None:
                raise KeyError(f"topic id={topic_id} 不是待審")
            if t["replaces"] is not None:
                self.cx.execute("UPDATE topic SET valid_to=?, superseded_by=? WHERE root=?",
                                (now, t["root"], t["replaces"]))
            self.cx.execute("UPDATE topic SET status='locked' WHERE id=?", (topic_id,))

    def reject(self, topic_id: int) -> None:
        """駁回：不刪列（原句與判斷紀錄都留著），不改核心設定。
        現行不進檢索；日後改為當日常記憶撈（見檔頭〈鎖定與人工審核〉）。"""
        with write_txn(self.cx):
            n = self.cx.execute("UPDATE topic SET status='rejected' WHERE id=? AND status='pending'",
                                (topic_id,)).rowcount
            if n != 1:
                raise KeyError(f"topic id={topic_id} 不是待審")

    def get(self, ids: list[int]) -> list[dict]:
        if not ids or not self._has_table():
            return []
        rows = self.cx.execute(
            f"SELECT * FROM topic WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()
        by = {r["id"]: dict(r) for r in rows}
        return [by[i] for i in ids if i in by]

    def source_rows(self, topic: dict) -> list[dict]:
        """一件事指回的原句（重寫與人工檢查用）。"""
        rows = self.cx.execute(
            "SELECT serial, ts, role, content, clean, situation, speaker_id FROM transcript "
            "WHERE serial BETWEEN ? AND ? ORDER BY serial",
            (topic["first_serial"], topic["last_serial"])).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        if not self._has_table():
            return 0
        return self.cx.execute("SELECT COUNT(*) c FROM topic").fetchone()["c"]
