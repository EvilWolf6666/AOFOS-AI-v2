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
#
# 檔案格式（問答分離索引，見《歸檔》E0004）：
#     Q: 妳叫什麼名字
#     Q: 妳是誰
#     A: 她叫奧狐，英文寫作 AoFos。
#
# 一個區塊 = 一個答案 + 一到多個問句。每個問句各自成為一筆 ChromaDB 文件，
# 答案存在 metadata。這樣同一則知識可以被多種問法命中，而注入模型的內容
# 始終只有答案本身、不帶問句贅詞。

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_manager import RAGManager, SOURCE_SEED  # noqa: E402

WORLDVIEW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worldview"
)


def parse_file(path: str) -> list[tuple[list[str], str]]:
    """解析成 [(問句列表, 答案), ...]。`#` 標題與 `>` 備註略過。

    區塊以 A: 行結束。連續的 Q: 行累積為同一個區塊的問法。
    刻意維持極簡：格式要能用肉眼在編輯器裡看出來，不需要記語法。

    格式錯誤一律大聲失敗而非跳過——靜默略過一個區塊的結果是「灌完了但少一則
    知識」，而少的那則不會有任何症狀，只會在某天被問到時答不出來。
    """
    blocks, questions = [], []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith(">"):
                continue
            if line.startswith("Q:"):
                q = line[2:].strip()
                if not q:
                    sys.exit(f"❌ {path}:{lineno} Q: 後面是空的")
                questions.append(q)
            elif line.startswith("A:"):
                a = line[2:].strip()
                if not a:
                    sys.exit(f"❌ {path}:{lineno} A: 後面是空的")
                if not questions:
                    sys.exit(f"❌ {path}:{lineno} 這個 A: 前面沒有任何 Q:")
                blocks.append((questions, a))
                questions = []
            else:
                sys.exit(f"❌ {path}:{lineno} 無法辨識的行（需以 Q: 或 A: 開頭）：{line[:40]}")
    if questions:
        sys.exit(f"❌ {path} 結尾有 {len(questions)} 個 Q: 沒有對應的 A:")
    return blocks


def collect() -> dict[str, tuple[str, str]]:
    """回傳 {doc_id: (問句, 答案)}。

    doc_id 為「檔名 + 區塊序號 + 問句序號」，故重跑是 upsert 而非重複寫入。
    ⚠️ 在檔案中間插入區塊會讓後面所有 id 位移，該次會重寫很多筆——最終狀態
    正確、不會重複，但 log 看起來很嚇人。
    """
    if not os.path.isdir(WORLDVIEW_DIR):
        sys.exit(f"❌ 找不到 {WORLDVIEW_DIR}／世界觀原始檔應放在專案根目錄的 worldview/ 底下")

    docs = {}
    for name in sorted(os.listdir(WORLDVIEW_DIR)):
        if not name.endswith(".md"):
            continue
        stem = os.path.splitext(name)[0]
        for bi, (questions, answer) in enumerate(parse_file(os.path.join(WORLDVIEW_DIR, name))):
            for qi, q in enumerate(questions):
                docs[f"worldview:{stem}:{bi:03d}:{qi}"] = (q, answer)
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只印出結果，不寫入")
    args = ap.parse_args()

    docs = collect()
    if not docs:
        sys.exit("❌ worldview/ 底下沒有解析出任何內容，中止（避免把 RAG 裡既有的種子全部清掉）")

    answers = {a for _, a in docs.values()}
    print(f"解析到 {len(answers)} 則知識、{len(docs)} 個問句索引：\n")
    last_answer = None
    for doc_id, (q, a) in docs.items():
        if a != last_answer:
            print(f"\n  A: {a}")
            last_answer = a
        print(f"     Q: {q}   [{doc_id}]")

    if args.dry_run:
        print("\n（dry-run，未寫入）")
        return

    rag = RAGManager()

    # RAG 在 TEST_MODE 下會靜默略過所有寫入。灌完顯示成功、實際一筆都沒進去是最糟的
    # 失敗樣態，故在此擋下並大聲報錯，而不是讓它安靜跑完。
    if not rag.enabled:
        sys.exit("❌ RAG 目前為停用狀態（TEST_MODE）。請先關閉 TEST_MODE 再執行，否則寫入會被靜默略過。")

    # 先寫入、後清理：反過來做的話，中途失敗會讓 RAG 停在「舊的已刪、新的沒進」的狀態。
    # 破壞性操作以成功為前提，不是先做再檢查（《現況》0之4）。
    for doc_id, (q, a) in docs.items():
        rag.add(q, doc_id=doc_id, source=SOURCE_SEED, answer=a)
    print(f"\n✅ 已寫入 {len(docs)} 個問句索引（{len(answers)} 則知識）")

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