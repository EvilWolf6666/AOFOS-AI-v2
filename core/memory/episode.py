# core/memory/episode.py
"""
事件層（準 v3 組合 9，甲案）：把原句切成一段一段，每段由程式挑關鍵詞與代表句。

⭐ **不經 LLM**（2026-09-27 邪狼裁定甲案）：不編造、不用等、內容全是原句。
   「用她的話概括一次」刻意不做——歸納放在讀取端，她拿到代表句後當場自己串。
⭐ **衍生物，不是資料**：`episode` 表完全由 `transcript` 算出來，隨時可整張重建
   （`build()` 就是整張重建）。切段規則或挑詞規則改了，重跑一次即可，原句一筆不動。

切段：相鄰兩句相隔超過 `gap_minutes` 就切成新的一段（時間是唯一依據，確定性）。
關鍵詞：段內「在兩句以上出現」的片語（相鄰兩字組會接成長片語，如「記憶模塊」）
        × 跨段稀有度。只出現一次的碎片（斷錯的兩字組）自然被濾掉，不需要斷詞器。
代表句：只挑使用者那一方的句子，含越多關鍵詞越前；都沒有就取最長的。
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from datetime import datetime

from core.memory.db import write_txn
import unicodedata

from core.memory.recall import _is_cjk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episode (
    id           INTEGER PRIMARY KEY,
    start_serial INTEGER NOT NULL,
    end_serial   INTEGER NOT NULL,
    start_ts     TEXT    NOT NULL,
    end_ts       TEXT    NOT NULL,
    n_rows       INTEGER NOT NULL,
    keywords     TEXT    NOT NULL,   -- JSON list[str]
    reps         TEXT    NOT NULL    -- JSON list[int]（代表句的 serial）
);
"""


def _parse(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def time_bucket(ts: str, now: datetime | None = None) -> str:
    """ts → 口語的時間感（剛剛／今天稍早／昨天／前幾天／上週／這個月稍早／很久沒聊）。

    ⭐ 刻意不給數字：給「3 天前」她會照唸（《現況》12之0 逐字複製傾向），
       給模糊說法她才會自己決定要不要提、怎麼提。解析失敗回空字串，**不猜**。
    """
    t = _parse(ts)
    if t is None:
        return ""
    now = now or datetime.now(t.tzinfo)
    if (t.tzinfo is None) != (now.tzinfo is None):
        return ""
    sec = (now - t).total_seconds()
    if sec < 0:
        return ""
    if sec < 1800:
        return "剛剛"
    if t.date() == now.date():
        return "今天稍早"
    days = (now.date() - t.date()).days
    if days == 1:
        return "昨天"
    if days < 7:
        return "前幾天"
    if days < 14:
        return "上週"
    if days < 31:
        return "這個月稍早"
    return "很久沒聊了"


def _units(text: str) -> list[str]:
    """一句話 → 中日韓連續段與英數詞（與 recall.bigrams 同一套正規化）。"""
    text = unicodedata.normalize("NFKC", text or "").lower()
    out, cur, cjk = [], [], None
    for ch in text + " ":
        k = True if _is_cjk(ch) else (False if ch.isalnum() else None)
        if k is None or (cur and k != cjk):
            if cur:
                out.append("".join(cur))
            cur, cjk = ([ch], k) if k is not None else ([], None)
        else:
            cur.append(ch); cjk = k
    return out


def _phrases(sentences: list[str], min_sf: int = 2) -> Counter:
    """在一段對話裡找「重複出現的詞」：回傳 {片語: 出現在幾句}。

    做法：兩字組在 ≥ min_sf 句裡出現才算數；同一句裡相鄰的合格兩字組**接成一個片語**
    （記憶＋憶模＋模塊 ⇒ 記憶模塊）。⭐ 只出現一次的兩字組多半是斷錯的碎片
    （「周都」「在研」），用「重複」過濾掉，不需要斷詞器。英數詞同理。
    """
    per_sent = [_units(t) for t in sentences]
    sf = Counter()
    for units in per_sent:
        seen = set()
        for u in units:
            if _is_cjk(u[0]):
                seen.update(u[i:i + 2] for i in range(len(u) - 1))
            elif len(u) >= 2 and not u.isdigit():
                seen.add(u)
        sf.update(seen)
    ok = {g for g, c in sf.items() if c >= min_sf}
    out = Counter()
    for units in per_sent:
        found = set()
        for u in units:
            if not _is_cjk(u[0]):
                if u in ok:
                    found.add(u)
                continue
            i = 0
            while i < len(u) - 1:
                if u[i:i + 2] in ok:
                    j = i
                    while j + 1 < len(u) - 1 and u[j + 1:j + 3] in ok:
                        j += 1
                    found.add(u[i:j + 2])
                    i = j + 1
                else:
                    i += 1
        out.update(found)
    return out


class EpisodeIndex:
    """一個 DB 檔的事件層。**每個執行緒傳入自己的連線**（同其他存取物件）。"""

    def __init__(self, cx: sqlite3.Connection, *, writable: bool = True) -> None:
        self.cx = cx
        self.writable = writable
        if writable:
            cx.executescript(_SCHEMA)

    def _has_table(self) -> bool:
        return self.cx.execute(
            "SELECT 1 FROM sqlite_master WHERE name='episode'").fetchone() is not None

    def build(self, *, gap_minutes: int, n_keywords: int, n_reps: int) -> int:
        """整張重建，回傳段數。唯讀模式下不做（回 0）。

        ⚠️ 讀原句在交易外、寫結果在交易內——計算不佔寫鎖（db.py 末的交易紀律）。
        """
        if not self.writable:
            return 0
        rows = self.cx.execute(
            "SELECT serial, ts, role, COALESCE(clean, content) AS c, situation "
            "FROM transcript WHERE retired = 0 ORDER BY serial").fetchall()

        # ── 切段 ──
        segs: list[list] = []
        prev_t = None
        for r in rows:
            t = _parse(r["ts"])
            new = (not segs or t is None or prev_t is None
                   or (t - prev_t).total_seconds() > gap_minutes * 60)
            if new:
                segs.append([])
            segs[-1].append(r)
            if t is not None:
                prev_t = t

        # ── 關鍵詞（段內重複出現的片語 × 跨段稀有度）與代表句 ──
        def text_of(r):
            return f'{r["c"]} {r["situation"] or ""}'
        seg_ph = [_phrases([text_of(r) for r in s]) for s in segs]
        df = Counter(ph for p in seg_ph for ph in p)
        n = len(segs)
        out = []
        for s, ph in zip(segs, seg_ph):
            score = {k: c * math.log((n + 1) / (df[k] + 0.5)) for k, c in ph.items()}
            # 長片語優先於它自己的子片語（「記憶模塊」在時不另列「記憶」）
            ranked = sorted(score.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
            kws: list[str] = []
            for k, sc in ranked:
                if sc <= 0 or any(k in longer for longer in kws):
                    continue
                kws.append(k)
                if len(kws) >= n_keywords:
                    break
            users = [r for r in s if r["role"] == "user" and r["c"].strip()]
            # 代表句：含越多關鍵詞越前；同分取較長的（內容多）；都沒有關鍵詞就取最長的
            cands = sorted(users, key=lambda r: (-sum(k in r["c"] for k in kws),
                                                 -len(r["c"]), r["serial"]))
            reps = sorted(r["serial"] for r in cands[:n_reps])
            out.append((s[0]["serial"], s[-1]["serial"], s[0]["ts"], s[-1]["ts"], len(s),
                        json.dumps(kws, ensure_ascii=False), json.dumps(reps)))

        with write_txn(self.cx):
            self.cx.execute("DELETE FROM episode")
            self.cx.executemany(
                "INSERT INTO episode(start_serial, end_serial, start_ts, end_ts, n_rows, "
                "keywords, reps) VALUES(?,?,?,?,?,?,?)", out)
        return len(out)

    def last_before(self, serial: int) -> dict | None:
        """end_serial < serial 的最後一段（＝「上一場」）。沒有就回 None。"""
        if not self._has_table():
            return None
        r = self.cx.execute(
            "SELECT * FROM episode WHERE end_serial < ? ORDER BY end_serial DESC LIMIT 1",
            (serial,)).fetchone()
        if r is None:
            return None
        return {"start_serial": r["start_serial"], "end_serial": r["end_serial"],
                "start_ts": r["start_ts"], "end_ts": r["end_ts"], "n_rows": r["n_rows"],
                "keywords": json.loads(r["keywords"]), "reps": json.loads(r["reps"])}

    def count_on(self, day: str) -> int:
        """某天（YYYY-MM-DD）開始的段數。給「今天第幾次聊」用。"""
        if not self._has_table():
            return 0
        return self.cx.execute(
            "SELECT COUNT(*) c FROM episode WHERE substr(start_ts, 1, 10) = ?",
            (day,)).fetchone()["c"]
