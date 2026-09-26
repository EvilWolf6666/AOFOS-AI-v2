# core/memory/recall.py
"""
取回：bigram ＋ SQLite FTS5（準 v3 最小記憶切片「取回走 A」）。

🟩 最小可行。目標形：混合檢索（本檔 ＋ 向量）；升級條件：回憶題中**實際觀察到**
   同義換句撈不到（《現況》〈🔒 隔離區〉記憶模塊〈準 v3〉〈十二〉）。

⭐ **索引是衍生物，不是資料**：`recall_fts` 從 `transcript.clean` 算出來，
   刪掉重建一筆不丟（`python -m core.memory.recall --rebuild <db>`）。
   ⇒ 本模組**只讀** `transcript`，唯一會寫的是自己的索引表、`indexed_upto`，
   以及命中計數 `recalled`（呼叫端可關）。

🔴 **為什麼自己做 bigram，不用 FTS5 內建分詞**：
   · `unicode61` 以非字母數字切詞，一整串中文沒有空白 ⇒ 會變成**一個超長 token**，
     查「高雄」撈不到「我上次去高雄玩」。
   · `trigram` 至少要三個字才查得到，而中文詞最常見的是兩個字。
   ⇒ 寫入與查詢**都先過同一個 `bigrams()`**，切成以空白分隔的兩字組，再交給
     `unicode61` 按空白切。⚠️ **兩端必須用同一個函式**，否則索引與查詢切法不同、
     靜默查不到。

⚠️ **索引以 catch-up 方式跟上**（`indexed_upto` → `landed_upto`），不掛 trigger：
   trigger 要呼叫 Python 註冊的函式，而任何**沒註冊那個函式的連線**一寫 transcript
   就炸——包括 db.py 自測與未來的 consumer。⇒ 用 catch-up，transcript 那邊完全
   不需要知道有索引存在（**這是去耦合的關鍵**）。
"""

from __future__ import annotations

import sqlite3
import unicodedata
from datetime import datetime

from core.logger import setup_logger
from core.memory.db import get_meta, set_meta, write_txn

logger = setup_logger("Recall")

_INDEX_KEY = "indexed_upto"

_INDEX_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS recall_fts USING fts5(
    tokens,
    tokenize = 'unicode61 remove_diacritics 0'
);
"""

_ROW_COLS = ("serial", "ts", "role", "content", "clean", "source", "kind", "situation")


# ══════════════════════════════════════════════════════════════
# 依賴檢查
# ══════════════════════════════════════════════════════════════

def check_fts5() -> None:
    """確認這個 Python 的 SQLite 有 FTS5，且 bigram 切法在 unicode61 下行為正確。

    🔴 **大聲失敗**：FTS5 是 SQLite 的編譯選項，不是每個 Python 都有。
    靜默退回「不撈」的話，對話照常、記憶照寫，**只是她永遠想不起來**——
    而那正是這個切片要解決的事。
    ⚠️ 不只測「能不能建表」，還測「中文 bigram 查得到」：unicode61 若把整串中文
    當一個 token，建表會成功、查詢會全落空，這才是會騙人的那種失敗。
    """
    try:
        cx = sqlite3.connect(":memory:")
        cx.executescript(_INDEX_SCHEMA)
        cx.execute("INSERT INTO recall_fts(rowid, tokens) VALUES(1, ?)",
                   (bigrams("我上次去高雄玩了三天"),))
        hit = cx.execute("SELECT rowid FROM recall_fts WHERE recall_fts MATCH ?",
                         (_match_expr(bigrams("高雄"), 8),)).fetchall()
        cx.close()
    except sqlite3.OperationalError as e:
        raise RuntimeError(
            f"這個 Python 的 SQLite（{sqlite3.sqlite_version}）不支援 FTS5：{e}\n"
            "   ⇒ 準 v3 的取回無法運作。處理：換一個內建 FTS5 的 Python"
            "（python.org 官方 Windows 安裝檔有），或放回 v1 的 manager.py／rag.py。"
        ) from e
    if [r[0] for r in hit] != [1]:
        raise RuntimeError(
            "FTS5 存在，但中文 bigram 查不到（unicode61 分詞行為與預期不同）。"
            "⇒ 取回會全數落空，不可啟動。"
        )


# ══════════════════════════════════════════════════════════════
# 切詞
# ══════════════════════════════════════════════════════════════

def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2FFFF
            or 0x3040 <= o <= 0x30FF or 0xAC00 <= o <= 0xD7AF)


def bigrams(text: str) -> str:
    """文字 → 以空白分隔的 token 串。**寫入與查詢共用。**

    · 中日韓連續段 → 相鄰兩字組（單字段保留單字）
    · 英數連續段 → 整段小寫當一個詞
    · 其他（標點、空白、表情符號）→ 分隔
    ⚠️ 先做 NFKC：全形英數 → 半形，否則「ＬＯＬ」與「lol」是兩個詞。
    """
    text = unicodedata.normalize("NFKC", text or "").lower()
    out: list[str] = []
    run, run_cjk = [], None

    def flush():
        if not run:
            return
        if run_cjk:
            out.extend([run[0]] if len(run) == 1 else
                       ["".join(run[i:i + 2]) for i in range(len(run) - 1)])
        else:
            out.append("".join(run))

    for ch in text:
        if _is_cjk(ch):
            kind = True
        elif ch.isalnum():
            kind = False
        else:
            flush(); run, run_cjk = [], None
            continue
        if run and kind != run_cjk:
            flush(); run = []
        run.append(ch); run_cjk = kind
    flush()
    return " ".join(out)


def _match_expr(tokens: str, cap: int) -> str:
    """token 串 → FTS5 MATCH 式（OR 連接，每個加引號）。

    ⚠️ 必須加引號：使用者輸入可能含 FTS5 語法字（AND／NEAR／*／"），不加引號
    會變成查詢語法，輕則查錯、重則 OperationalError。引號內的 `"` 以 `""` 跳脫。
    """
    seen: list[str] = []
    for t in tokens.split():
        if t not in seen:
            seen.append(t)
        if len(seen) >= cap:
            break
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in seen)


# ══════════════════════════════════════════════════════════════
# 索引與查詢
# ══════════════════════════════════════════════════════════════

class RecallIndex:
    """一個 DB 檔的檢索索引。**每個執行緒傳入自己的連線**（同 PreWrite／Transcript）。"""

    def __init__(self, cx: sqlite3.Connection, *, writable: bool = True) -> None:
        """`writable=False`：唯讀使用（`/test on` 下的日常區）——不建表、不 catch-up、
        不記命中。⚠️ 此時若索引表不存在，查詢一律回空並警告一次，不代為建立。"""
        self.cx = cx
        self.writable = writable
        self._missing_warned = False
        if writable:
            cx.executescript(_INDEX_SCHEMA)
            with write_txn(cx):
                cx.execute("INSERT OR IGNORE INTO meta(key, value) VALUES(?, 0)",
                           (_INDEX_KEY,))

    def _has_index(self) -> bool:
        return self.cx.execute(
            "SELECT 1 FROM sqlite_master WHERE name='recall_fts'").fetchone() is not None

    def catch_up(self) -> int:
        """把 `indexed_upto < serial ≤ landed_upto` 的原文補進索引，回傳補了幾筆。

        ⭐ 索引與 `indexed_upto` 在同一個交易：不會「進了索引但號沒推」而重複收錄。
        ⚠️ 只收 `landed_upto` 以內——那之外的 serial 還沒確定落地。
        """
        if not self.writable:
            return 0
        with write_txn(self.cx):
            lo = get_meta(self.cx, _INDEX_KEY)
            hi = get_meta(self.cx, "landed_upto")
            if hi <= lo:
                return 0
            rows = self.cx.execute(
                "SELECT serial, COALESCE(clean, content) AS c, situation FROM transcript "
                "WHERE serial > ? AND serial <= ? ORDER BY serial", (lo, hi)).fetchall()
            self.cx.executemany(
                "INSERT OR REPLACE INTO recall_fts(rowid, tokens) VALUES(?, ?)",
                [(r["serial"], bigrams(f'{r["c"]} {r["situation"] or ""}')) for r in rows])
            set_meta(self.cx, _INDEX_KEY, hi)
        return len(rows)

    def rebuild(self) -> int:
        """整個索引砍掉重建。⚠️ clean 規則改過、或 transcript 有列被修補後要跑。"""
        with write_txn(self.cx):
            self.cx.execute("DELETE FROM recall_fts")
            set_meta(self.cx, _INDEX_KEY, 0)
        return self.catch_up()

    def search(self, query: str, *, top_k: int, max_terms: int,
               exclude: set[int] | frozenset[int] = frozenset()) -> list[tuple[int, float]]:
        """回傳 `[(serial, bm25 分數), ...]`，分數越小越相關（FTS5 慣例）。

        `exclude`：不得命中的 serial（目前 Working Memory 裡那幾則——已經在
        context 裡的東西再撈一次只會重複）。
        """
        expr = _match_expr(bigrams(query), max_terms)
        if not expr or top_k <= 0:
            return []
        if not self._has_index():
            if not self._missing_warned:
                logger.warning("這個 DB 沒有檢索索引（唯讀模式不代建），本區不參與取回")
                self._missing_warned = True
            return []
        rows = self.cx.execute(
            "SELECT f.rowid AS serial, bm25(recall_fts) AS score "
            "FROM recall_fts f JOIN transcript t ON t.serial = f.rowid "
            "WHERE recall_fts MATCH ? AND t.retired = 0 "
            "ORDER BY score LIMIT ?",
            (expr, top_k + len(exclude))).fetchall()
        return [(r["serial"], r["score"]) for r in rows
                if r["serial"] not in exclude][:top_k]

    def neighborhood(self, hits: list[int], n: int,
                     exclude: set[int] | frozenset[int] = frozenset()) -> list[list[dict]]:
        """每個命中帶出前後各 n 則，重疊的合併成一段；回傳依時間排序的段落清單。

        每段是 `[row, ...]`，row 多一個 `hit` 布林欄。
        ⚠️ 鄰域以 serial 算，會跨場次——`serial`＋`ts` 足以讓注入端標出時間差，
           這是〈十二〉收回 `session_id` 的理由。
        """
        if not hits:
            return []
        spans = sorted((max(1, s - n), s + n) for s in hits)
        merged: list[list[int]] = []
        for lo, hi in spans:
            if merged and lo <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        hit_set = set(hits)
        blocks = []
        for lo, hi in merged:
            rows = self.cx.execute(
                f"SELECT {','.join(_ROW_COLS)} FROM transcript "
                "WHERE serial BETWEEN ? AND ? AND retired = 0 ORDER BY serial",
                (lo, hi)).fetchall()
            block = [{**{k: r[k] for k in _ROW_COLS}, "hit": r["serial"] in hit_set}
                     for r in rows if r["serial"] not in exclude]
            if block:
                blocks.append(block)
        return blocks

    def mark_recalled(self, serials: list[int]) -> None:
        """命中計數 +1（只算命中本身，不算鄰域）。唯讀模式下不寫。"""
        if not self.writable or not serials:
            return
        with write_txn(self.cx):
            self.cx.executemany(
                "UPDATE transcript SET recalled = recalled + 1 WHERE serial = ?",
                [(s,) for s in serials])


# ══════════════════════════════════════════════════════════════
# 注入文字
# ══════════════════════════════════════════════════════════════

def time_ago(ts: str, now: datetime | None = None) -> str:
    """ts（ISO 8601）→「3 天前」。解析失敗回「時間不明」，**不猜**。"""
    try:
        t = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return "時間不明"
    now = now or datetime.now(t.tzinfo)
    if (t.tzinfo is None) != (now.tzinfo is None):
        return "時間不明"
    sec = (now - t).total_seconds()
    if sec < 0:
        return "時間在未來（時鐘可能被改過）"
    if sec < 3600:
        return "剛才" if sec < 120 else f"{int(sec // 60)} 分鐘前"
    if sec < 86400:
        return f"{int(sec // 3600)} 小時前"
    days = int(sec // 86400)
    if days < 60:
        return f"{days} 天前"
    if days < 730:
        return f"約 {days // 30} 個月前"
    return f"約 {days // 365} 年前"


def format_blocks(blocks: list[list[dict]], labels: dict[str, str],
                  now: datetime | None = None) -> str:
    """段落 → 注入文字。原句逐字（取 clean），每段標時間差與日期。

    ⚠️ observation 輪次的 user 那則 content 是空 placeholder，情境在 situation
    ——這裡改印情境，否則會出現一行「邪狼：」後面什麼都沒有。
    """
    parts = ["[相關記憶] 以下是過去對話的原句，依時間排列；時間差以現在為準。"]
    for block in blocks:
        head = block[0]["ts"] or ""
        parts.append(f"── {time_ago(head, now)}（{head[:16].replace('T', ' ')}）──")
        for r in block:
            text = r["clean"] if r["clean"] is not None else r["content"]
            if r["role"] == "user" and not text.strip() and r["situation"]:
                parts.append(f"［當時情境］{r['situation']}")
            else:
                parts.append(f"{labels.get(r['role'], r['role'])}：{text}")
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════
# 手動重建：python -m core.memory.recall --rebuild <db 路徑>
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    from core.memory.db import connect

    if len(sys.argv) == 3 and sys.argv[1] == "--rebuild":
        check_fts5()
        c = connect(sys.argv[2])
        print(f"重建完成：{RecallIndex(c).rebuild()} 筆")
        sys.exit(0)
    print("用法：python -m core.memory.recall --rebuild <db 路徑>")
    sys.exit(2)
