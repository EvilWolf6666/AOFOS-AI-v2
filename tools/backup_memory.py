"""backup_memory：把記憶檔完整複製一份（手動備份）。

為什麼有這支
  〈現況-開場〉基本準則：動正式記憶檔（data/memory.db）前先備份。原本靠人記得手動複製檔案；
  而 SQLite 開著 WAL 時直接複製 .db 檔可能少了還在 -wal 裡的最後幾筆。這支用 SQLite 的 backup API，
  AOFOS 開著也能拿到一致的一份。

用法（專案根目錄）：
  python -m tools.backup_memory                 備份 config 的日常記憶檔
  python -m tools.backup_memory --tag 換模型前    檔名加一段說明
  python -m tools.backup_memory --db <路徑>       備份別的記憶檔

輸出：<記憶檔所在資料夾>/backups/<檔名>_<時間>[_<tag>].db；完成後核對兩邊的原句筆數。
離開碼：0 成功／1 失敗／2 找不到檔案
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime

import config


def backup(src: str, tag: str = "") -> str:
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    d = os.path.join(os.path.dirname(os.path.abspath(src)), "backups")
    os.makedirs(d, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src))[0]
    safe = "".join(c for c in tag if c not in '\\/:*?"<>|').strip()
    dst = os.path.join(d, f"{stem}_{datetime.now():%Y%m%d_%H%M%S}" + (f"_{safe}" if safe else "") + ".db")
    if os.path.exists(dst):
        raise FileExistsError(dst)
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)   # 來源唯讀：備份不得動到正式檔
    b = sqlite3.connect(dst)
    try:
        s.backup(b)
        n_src = s.execute("SELECT COUNT(*) FROM transcript").fetchone()[0]
        n_dst = b.execute("SELECT COUNT(*) FROM transcript").fetchone()[0]
    finally:
        b.close()
        s.close()
    if n_src != n_dst:   # 失敗要大聲：筆數對不上的備份不能當備份
        raise RuntimeError(f"備份後原句筆數對不上：來源 {n_src}／備份 {n_dst}（{dst}）")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description="記憶檔備份（SQLite backup API）")
    ap.add_argument("--db", default=config.MEMORY_CONFIG["db_path"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    try:
        dst = backup(a.db, a.tag)
    except FileNotFoundError:
        print(f"❌ 找不到記憶檔：{a.db}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"❌ 備份失敗：{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"✅ 已備份：{dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
