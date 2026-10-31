# core/memory/pre_write.py
"""
預寫區（Write-Ahead Log）。

⚠️ **中文文件一律稱它「預寫區」，程式碼檔名是 `pre_write`** ——兩套名字刻意不一致，
   在此互指一次即可（否則日後 grep「預寫區」找不到檔案）。

它是什麼：**落盤的暫存**。裝的是「待處理」，不是「已保存」。
  · **暫存**講保留期限——被 consumer 處理完就清掉
  · **落盤**講耐久性——crash 之後還在
🔴 **兩者不衝突，但不可混談**：若把它當「已保存」，就會有人以為原文永久留在這裡
   而不去用原文存放區——**而預寫區清掉之後那些原文就真的沒了。**

存在 SQLite 表裡（理由見 `db.py` 檔頭）：交易保證讓「寫到一半的紀錄」這個狀態不成立。
   ⚠️ **仍然保留的兩道檢查**：
     · **內容 hash 重算**——防磁碟層腐蝕（機率低，但成本是零：hash 本來就要算）
     · **`serial` 嚴格遞增**——防邏輯錯（例如發號重複）
"""

from __future__ import annotations

import hashlib
import sqlite3

from core.logger import setup_logger
from core.memory.db import get_meta, set_meta, write_txn

logger = setup_logger("PreWrite")

_COLS = ("serial", "ts", "role", "content", "hash", "source", "kind", "situation", "speaker_id")


def content_hash(content: str) -> str:
    """內容雜湊。截短到 16 個十六進位字元：碰撞機率對本用途遠遠足夠，
    而整串 64 字元會讓每一筆平白多 48 位元組，人眼比對也更難看。"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


class PreWrite:
    """預寫區佇列。**每個執行緒傳入自己的連線。**"""

    def __init__(self, cx: sqlite3.Connection) -> None:
        self.cx = cx

    # ── producer 側（主線程）────────────────────────────────────

    def append(self, turns: list[dict]) -> list[int]:
        """把一批輪次寫進預寫區，回傳配發的 serial 清單。

        🔴 **呼叫端必須等本方法回傳之後才裁切 Working Memory。**
        本方法回傳即代表交易已提交（`synchronous=FULL` ⇒ 已落到磁碟）；
        在它回傳之前，原文只存在記憶體裡。

        ⭐ **取號與寫入在同一個交易裡** ⇒ 不會有「號發了但紀錄沒寫進去」。
        ⚠️ `turns` 的元素就是 Working Memory 的紀錄（role／content ＋ 選填的
        source／kind／situation）——本層**原樣轉寫，不認識也不判斷那些欄位**。
        記憶的欄位集是上層的規格，不該由暫存層決定。
        """
        if not turns:
            return []

        for t in turns:
            if "role" not in t or "content" not in t:
                # 缺欄位大聲失敗，不靜默補空：補空會寫進一筆看起來正常、
                # 但內容是假的紀錄。
                raise ValueError(f"預寫區紀錄缺 role/content：{str(t)[:80]}")

        # 🔴 **必須是 write_txn（BEGIN IMMEDIATE）**：取號是 read-modify-write，
        #    用預設的 DEFERRED 交易時兩個執行緒會讀到同一個號（自測撞過）。
        with write_txn(self.cx):   # 一個交易：取號 ＋ 寫入 ＋ 推進 next_serial
            start = get_meta(self.cx, "next_serial")
            rows, serials = [], []
            for i, t in enumerate(turns):
                s = start + i
                serials.append(s)
                rows.append((
                    s,
                    t.get("ts", ""),
                    t["role"],
                    t["content"],
                    content_hash(t["content"]),
                    t.get("source") or None,
                    t.get("kind") or None,
                    t.get("situation") or None,
                    t.get("speaker_id") or None,
                ))
            self.cx.executemany(
                f"INSERT INTO pre_write({','.join(_COLS)}) VALUES(?,?,?,?,?,?,?,?,?)", rows
            )
            set_meta(self.cx, "next_serial", start + len(turns))
        return serials

    def purge_upto(self, serial: int) -> int:
        """移除 serial ≤ 指定值的紀錄，回傳移除筆數。

        誰呼叫都安全（寫入由鎖序列化，不會交錯寫壞）。實務上由推進 checkpoint 的那一段呼叫，
        因為它才知道處理到哪。
        """
        with write_txn(self.cx):
            cur = self.cx.execute("DELETE FROM pre_write WHERE serial <= ?", (serial,))
        return cur.rowcount

    def backlog(self) -> int:
        """目前積壓筆數。

        🔴 **給「背景執行緒靜默死掉」那個檢查用。** Python 背景執行緒的未捕捉例外
        不會傳到主線程——它只印一行 traceback，然後那條執行緒悄悄死掉，
        **程式照常跑、對話照常回，而預寫區一直長、記憶再也不進庫。**
        ⚠️ 崩潰至少會停，這個不會。
        """
        return self.cx.execute("SELECT COUNT(*) c FROM pre_write").fetchone()["c"]

    # ── consumer 側（背景執行緒）────────────────────────────────

    def read_pending(self, after_serial: int = 0, limit: int = 0
                     ) -> tuple[list[dict], str | None]:
        """讀出 serial > after_serial 的紀錄。回傳 `(紀錄清單, 中止原因)`。

        🔴 **中止時回傳的是「中止點之前」的紀錄**——呼叫端照樣可以處理它們，
        但 checkpoint **不得推過中止點**。
        ⚠️ 任何一道檢查不過一律「停在那一行之前」，**不是跳過它繼續讀**：
        預寫區是有序佇列，跳過壞的繼續消費會讓 checkpoint 推過一個
        **從未被處理的 serial**，那一筆就永久遺失且無徵兆。
        """
        sql = f"SELECT {','.join(_COLS)} FROM pre_write WHERE serial > ? ORDER BY serial"
        args: tuple = (after_serial,)
        if limit:
            sql += " LIMIT ?"
            args = (after_serial, limit)

        out: list[dict] = []
        # ⚠️ **兩個變數不可合併。** `prev` 是「上一筆的 serial」（遞增檢查用），
        #    `after_serial` 是「已處理到哪」（過濾用）——語意不同。
        #    🔴 合成一個的代價：續讀時第一筆的 serial 等於 after_serial，會被誤判成
        #    「未嚴格遞增」而整批停住。**而它只在第二輪以後才出現**——第一次跑
        #    （after_serial=0）完全正常，是典型的射程缺口（自測實際抓到過）。
        prev: int | None = None
        for r in self.cx.execute(sql, args):
            rec = {k: r[k] for k in _COLS}

            if prev is not None and rec["serial"] <= prev:
                return out, _halt(rec["serial"],
                                  f"serial 未嚴格遞增（前一筆 {prev}）")
            if content_hash(rec["content"]) != rec["hash"]:
                return out, _halt(rec["serial"], "內容 hash 對不上（寫入後被改動）")

            prev = rec["serial"]
            out.append(rec)
        return out, None


def _halt(serial: int, why: str) -> str:
    """統一的中止訊息。**一律大聲**——這兩種情況在正常運行下都不該出現。"""
    msg = f"預寫區 serial={serial} {why}，本輪停止讀取（不跳過、不往後讀）"
    logger.error(msg)
    logger.error(
        "   ⚠️ 跳過壞紀錄繼續消費，會讓 checkpoint 推過一個從未被處理的 serial，"
        "那一筆就永久遺失且無徵兆。"
    )
    return msg
