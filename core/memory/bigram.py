# core/memory/bigram.py
"""
中文 bigram 切詞＋FTS5 查詢式。

用的人：主題的關鍵字那一路（core/memory/topic_index.py 的 `topic_fts`）、事件層挑關鍵詞
（core/memory/episode.py 的 `_is_cjk`）、MemoryManager 啟動時的 FTS5 檢查。

🔴 **為什麼自己做 bigram，不用 FTS5 內建分詞**：
   · `unicode61` 以非字母數字切詞，一整串中文沒有空白 ⇒ 會變成**一個超長 token**，
     查「高雄」撈不到「我上次去高雄玩」。
   · `trigram` 至少要三個字才查得到，而中文詞最常見的是兩個字。
   ⇒ 寫入與查詢**都先過同一個 `bigrams()`**，切成以空白分隔的兩字組，再交給
     `unicode61` 按空白切。⚠️ **兩端必須用同一個函式**，否則索引與查詢切法不同、靜默查不到。
"""

from __future__ import annotations

import sqlite3
import unicodedata

_PROBE_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS probe_fts USING fts5(
    tokens,
    tokenize = 'unicode61 remove_diacritics 0'
);
"""


def check_fts5() -> None:
    """確認這個 Python 的 SQLite 有 FTS5，且 bigram 切法在 unicode61 下行為正確。

    🔴 **大聲失敗**：FTS5 是 SQLite 的編譯選項，不是每個 Python 都有。
    靜默退回「不撈」的話，對話照常、記憶照寫，**只是關鍵字那一路永遠撈不到**。
    ⚠️ 不只測「能不能建表」，還測「中文 bigram 查得到」：unicode61 若把整串中文
    當一個 token，建表會成功、查詢會全落空，這才是會騙人的那種失敗。
    """
    try:
        cx = sqlite3.connect(":memory:")
        cx.executescript(_PROBE_SCHEMA)
        cx.execute("INSERT INTO probe_fts(rowid, tokens) VALUES(1, ?)",
                   (bigrams("我上次去高雄玩了三天"),))
        hit = cx.execute("SELECT rowid FROM probe_fts WHERE probe_fts MATCH ?",
                         (_match_expr(bigrams("高雄"), 8),)).fetchall()
        cx.close()
    except sqlite3.OperationalError as e:
        raise RuntimeError(
            f"這個 Python 的 SQLite（{sqlite3.sqlite_version}）不支援 FTS5：{e}\n"
            "   ⇒ 主題的關鍵字檢索無法運作。處理：換一個內建 FTS5 的 Python"
            "（python.org 官方 Windows 安裝檔有）。"
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
