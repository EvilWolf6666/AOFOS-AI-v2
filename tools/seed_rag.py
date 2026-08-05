# tools/seed_rag.py
# 把 worldview/ 底下的世界觀原始檔灌進 RAG。
#
# ⚠️ 這不是診斷腳手架，不要放進 core/short_time_test.py 的清理清單。
#    腳手架的定義是「有明確刪除條件的暫時性程式碼」；本檔沒有刪除條件——只要世界觀
#    還會增補、或 data/chroma_db 還可能再被刪掉重建，它就一直有用。理由同
#    TIMING_CONFIG 被刻意排除於腳手架清單之外。
#
# 用法（於專案根目錄執行）：
#     python -m tools.seed_rag --dry-run    先看會寫進去什麼
#     python -m tools.seed_rag              實際寫入
#
# 語意是「整份同步」而非「附加」：檔案裡有什麼，RAG 裡的 seed 就是什麼。
# 從檔案刪掉的段落，下次執行時會從 RAG 一併移除。

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_manager import RAGManager, SOURCE_SEED  # noqa: E402

WORLDVIEW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worldview"
)


def parse_file(path: str) -> list[str]:
    """一個空行分隔的段落 = 一筆知識。`#` 開頭的標題與 `>` 開頭的備註不算內容。

    刻意維持極簡：分段規則要能用肉眼在編輯器裡看出來，不需要記語法。
    """
    entries, buf = [], []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip()
            if line.startswith("#") or line.startswith(">"):
                continue
            if line.strip():
                buf.append(line.strip())
            elif buf:
                entries.append(" ".join(buf))
                buf = []
    if buf:
        entries.append(" ".join(buf))
    return entries


def collect() -> dict[str, str]:
    """回傳 {doc_id: text}。doc_id 由「檔名 + 段落序號」構成，故重跑是 upsert 而非重複寫入。"""
    if not os.path.isdir(WORLDVIEW_DIR):
        sys.exit(f"❌ 找不到 {WORLDVIEW_DIR}／世界觀原始檔應放在專案根目錄的 worldview/ 底下")

    docs = {}
    for name in sorted(os.listdir(WORLDVIEW_DIR)):
        if not name.endswith(".md"):
            continue
        stem = os.path.splitext(name)[0]
        for i, text in enumerate(parse_file(os.path.join(WORLDVIEW_DIR, name))):
            docs[f"worldview:{stem}:{i:03d}"] = text
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只印出結果，不寫入")
    args = ap.parse_args()

    docs = collect()
    if not docs:
        sys.exit("❌ worldview/ 底下沒有解析出任何內容，中止（避免把 RAG 裡既有的種子全部清掉）")

    print(f"解析到 {len(docs)} 筆：\n")
    for doc_id, text in docs.items():
        print(f"  [{doc_id}] {text}")

    if args.dry_run:
        print("\n（dry-run，未寫入）")
        return

    rag = RAGManager()

    # RAG 在 TEST_MODE 下會靜默略過所有寫入。灌完顯示成功、實際一筆都沒進去是最糟的
    # 失敗樣態，故在此擋下並大聲報錯，而不是讓它安靜跑完。
    if not rag.enabled:
        sys.exit("❌ RAG 目前為停用狀態（TEST_MODE）。請先關閉 TEST_MODE 再執行，否則寫入會被靜默略過。")

    # 先寫入、後清理：反過來做的話，中途失敗會讓 RAG 停在「舊的已刪、新的沒進」的狀態。
    # 破壞性操作以成功為前提，不是先做再檢查（〇之6）。
    for doc_id, text in docs.items():
        rag.add(text, doc_id=doc_id, source=SOURCE_SEED)
    print(f"\n✅ 已寫入 {len(docs)} 筆")

    stale = [
        item["id"]
        for item in rag.list_all()
        if (item.get("metadata") or {}).get("source") == SOURCE_SEED
        and item["id"] not in docs
    ]
    for doc_id in stale:
        rag.delete(doc_id)
    if stale:
        print(f"🧹 已移除 {len(stale)} 筆已從檔案刪除的舊種子")

    print(f"📊 目前 RAG 總筆數：{rag.collection.count()}")


if __name__ == "__main__":
    main()
