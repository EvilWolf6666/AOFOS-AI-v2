# core/memory/transcript.py
"""
原文存放區：逐字原句的**權威副本**，永久保存。

⚠️ **刻意不叫 `archive.py`** ——專案根目錄已經有一個 `archive/` 資料夾，同名會讓
   兩者在對話與文件裡持續混淆。`transcript`（逐字稿）也更精確描述它裝的東西。

🔴 **它是權威副本這件事的後果超出字面**：原文不在向量庫裡 ⇒ 換 embedding、
   換距離空間時，**刪掉向量庫、從這裡重跑一次就好，資料一筆不丟**。

**存逐字，不存抽取**：抽取保住「發生了什麼」、
丟掉「**怎麼說的**」——語氣、用詞、當下喊的稱呼。而陪伴型要的正是後者：
「你上次去高雄也是三天」這種**她自己講過的話**，抽成「邪狼曾去高雄三天」之後
就回指不了。⚠️ 粒度＝**一則**（一個 user／assistant 對是兩則）。

🔴 **本模組同時持有 checkpoint N（landed_upto）**，而這是刻意的：
   「原文落地 ＋ 推進 N」必須在**同一個交易**裡——拆成兩個模組就拆成兩段，
   那個「原文寫成功但 N 沒更新」的窗口會回來。⇒ checkpoint 不獨立成檔。
背景判斷的進度另有 `judged_upto`（core/memory/topics.py），明寫在 meta，不靠「還沒建主題」推斷。
"""

from __future__ import annotations

import sqlite3

from core.logger import setup_logger
from core.memory.db import get_meta, set_meta, write_txn
from core.memory.pre_write import content_hash
from core.memory.clean import clean_text

logger = setup_logger("Transcript")

_COLS = ("serial", "ts", "role", "content", "hash", "source", "kind", "situation", "speaker_id")
# 寫入時多一欄 clean（由 clean_text() 從 content 算出）。
# ⚠️ 只加在寫入端：預寫區不認識 clean（它由本檔落地時才算）。
_WRITE_COLS = _COLS + ("clean",)


class Transcript:
    """原文存放區 ＋ checkpoint N。**每個執行緒傳入自己的連線。**"""

    def __init__(self, cx: sqlite3.Connection) -> None:
        self.cx = cx

    # ── checkpoint ──────────────────────────────────────────────

    @property
    def landed_upto(self) -> int:
        """N：原文已落地到哪個 serial。producer 據此清預寫區。"""
        return get_meta(self.cx, "landed_upto")

    # ── 落地 ────────────────────────────────────────────────────

    def land(self, records: list[dict]) -> dict:
        """把一批原文寫進原文存放區並推進 N，回傳各結果的計數。

        ⭐ **寫入與推進 N 在同一個交易**：要嘛都成功，要嘛都不算——
        那個「原文寫成功但 N 沒更新」的窗口不存在。
        ⚠️ N 只推進到**本批實際成功寫入的最大 serial**，不是「讀到的最大」。
        兩者在正常情況下一樣，**出錯時差一整批，而且不會報錯**。

        四種結果：
          · serial 不存在              → 寫入（靜默，計入批次匯總）
          · serial 存在、hash 相同      → 跳過（重跑造成的，不是錯）
          · 🔴 serial 存在、hash 不同   → **覆寫（修補），並大聲**
          · 例外                       → 交易整個回滾，N 不動，下次重跑
        """
        if not records:
            return {"written": 0, "skipped": 0, "repaired": 0, "landed_upto": self.landed_upto}

        written = skipped = repaired = 0
        top = 0
        with write_txn(self.cx):
            for rec in records:
                s = rec["serial"]
                h = content_hash(rec["content"])
                row = self.cx.execute(
                    "SELECT hash FROM transcript WHERE serial=?", (s,)
                ).fetchone()
                vals = (s, rec.get("ts") or "", rec["role"], rec["content"], h,
                        rec.get("source"), rec.get("kind"), rec.get("situation"),
                        rec.get("speaker_id"), clean_text(rec["content"]))

                if row is None:
                    self.cx.execute(
                        f"INSERT INTO transcript({','.join(_WRITE_COLS)}) "
                        f"VALUES(?,?,?,?,?,?,?,?,?,?)", vals)
                    written += 1
                elif row["hash"] == h:
                    skipped += 1
                else:
                    # 🔴 **覆寫的唯一合法情況是「修補」，不是「更新」。**
                    #    原文是逐字歷史事實，性質上就不該被改——用「內容被更新」
                    #    當理由等於偷偷假設原文可變，與「權威副本、不刪」抵觸。
                    #    ⇒ 它只可能來自「前一次寫入被截斷或寫壞」。
                    # ⚠️ **所以它要大聲**：正常運行不該出現，出現就是有東西壞過。
                    self.cx.execute(
                        "UPDATE transcript SET ts=?, role=?, content=?, hash=?, "
                        "source=?, kind=?, situation=?, speaker_id=?, clean=? WHERE serial=?",
                        vals[1:] + (s,))
                    repaired += 1
                    logger.error(
                        f"原文存放區 serial={s} 內容與既有紀錄不同，已覆寫修補。"
                        f"⚠️ 原文是歷史事實、正常運行不該被改——請查前一次寫入為何壞掉。"
                    )
                top = max(top, s)

            if top > self.landed_upto:
                set_meta(self.cx, "landed_upto", top)

        # ⚠️ 依《現況》0之3「成功靜默」：常態只印一行批次匯總，不逐筆。
        #    逐筆留給診斷模式。（修補那一行是失敗訊號，在上面已經大聲過了。）
        logger.info(
            f"本批 {len(records)} 筆：{written} 寫入／{skipped} 跳過／{repaired} 修補"
            f"（已落地到 serial={self.landed_upto}）"
        )
        return {"written": written, "skipped": skipped, "repaired": repaired,
                "landed_upto": self.landed_upto}

    # ── 讀取 ────────────────────────────────────────────────────

    def count(self) -> int:
        return self.cx.execute("SELECT COUNT(*) c FROM transcript").fetchone()["c"]
