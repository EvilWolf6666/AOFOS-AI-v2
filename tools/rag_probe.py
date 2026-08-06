# tools/rag_probe.py
# 逐題印出 RAG 檢索結果與相似度，並計算 recall@k 與 MRR。
#
# ⚠️ 與 seed_rag.py 同，這不是診斷腳手架，不列入 core/short_time_test.py 的清理清單。
#    每次擴充 worldview/ 之後都會再用到，沒有刪除條件。
#
# ⚠️ 刻意不呼叫 RAGManager.query()：那條路徑會 record_search()，跑一輪題本會讓
#    rag_stats.json 的 searches 平白增加十幾次，污染「淘汰內容命中率」的量測
#    （《歸檔》二十九之5）。本檔直接打 collection，只讀不寫。
#
# ⚠️ 前提假設：正式路徑是用「使用者輸入原文」去查 RAG。已於 2026-08-05 確認
#    memory_manager.py:160 為 self.rag.query(user_input)，無改寫或加料。
#
# ⚠️ 要看的是 recall@k 與 MRR，不是「過門檻數」。門檻是地板不是閘門——只要正確
#    答案進了上下文，模型就有機會用它。用過門檻數當成績會把「檢索夠好但閘門設錯」
#    誤判成「檢索不夠好」，這兩者的修法完全不同。見《歸檔》三十之1。
#
# 題本格式（tools/probe_questions.txt）：
#     問題文字 | 正確答案的關鍵片段
#     問題文字                      ← 省略關鍵片段者不計入指標，只印結果
#     # 開頭為註解
#
# 用法：
#     python -m tools.rag_probe                      讀 tools/probe_questions.txt
#     python -m tools.rag_probe my_questions.txt     指定題本

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAG_SIMILARITY_THRESHOLD, RAG_TOP_K  # noqa: E402
from core.rag_manager import RAGManager, RETIRED_FIELD  # noqa: E402

DEFAULT_QUESTIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_questions.txt")


def load(path):
    cases = []
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            q, expect = line.split("|", 1)
            cases.append((q.strip(), expect.strip()))
        else:
            cases.append((line, None))
    return cases


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTIONS
    if not os.path.exists(path):
        sys.exit(f"❌ 找不到題本：{path}")

    cases = load(path)
    if not cases:
        sys.exit("❌ 題本沒有任何題目")

    rag = RAGManager()
    if not rag.enabled:
        sys.exit("❌ RAG 目前為停用狀態（TEST_MODE）。請先關閉 TEST_MODE。")

    count = rag.collection.count()
    if count == 0:
        sys.exit("❌ RAG 是空的，請先跑 python -m tools.seed_rag")

    print(f"庫內 {count} 筆　門檻 {RAG_SIMILARITY_THRESHOLD}　TOP_K {RAG_TOP_K}\n")

    scored = 0          # 有標注正確答案的題數
    recall_hits = 0     # 正確答案出現在去重後前 K 名的題數
    rr_sum = 0.0        # 倒數排名累計
    floor_pass = 0      # 通過地板的題數（僅供對照，不是主要指標）

    for q, expect in cases:
        res = rag.collection.query(
            query_texts=[q],
            n_results=min(RAG_TOP_K * 3, count),
            include=["documents", "distances", "metadatas"],
            where={RETIRED_FIELD: False},
        )
        docs = res["documents"][0]
        dists = res["distances"][0]
        metas = res["metadatas"][0] if res.get("metadatas") else [None] * len(docs)

        # 與 RAGManager.query() 相同的去重邏輯：依實際注入內容去重，多取候選後截斷
        rows, seen = [], set()
        for doc, dist, meta in zip(docs, dists, metas):
            meta = meta or {}
            payload = meta.get("answer") or doc
            if payload in seen:
                continue
            seen.add(payload)
            rows.append((1 - dist, doc, payload))
            if len(rows) >= RAG_TOP_K:
                break

        print(f"❓ {q}")
        for sim, doc, payload in rows:
            mark = "✅" if sim >= RAG_SIMILARITY_THRESHOLD else "  "
            shown = payload if payload == doc else f"{doc}　→　{payload[:40]}"
            print(f"   {mark} {sim:.3f}  {shown[:88]}")
        if rows and rows[0][0] >= RAG_SIMILARITY_THRESHOLD:
            floor_pass += 1

        if expect:
            scored += 1
            rank = next((i for i, r in enumerate(rows, 1) if expect in r[2]), None)
            if rank:
                recall_hits += 1
                rr_sum += 1 / rank
                if rank > 1:
                    print(f"   ↳ 正確答案排第 {rank}")
            else:
                print(f"   ⚠️ 正確答案未進前 {RAG_TOP_K}（找 '{expect}'）")
        print()

    print("─" * 60)
    if scored:
        print(f"recall@{RAG_TOP_K} : {recall_hits}/{scored} = {recall_hits / scored:.1%}   ← 最重要")
        print(f"MRR         : {rr_sum / scored:.3f}   （1.000 代表每題正確答案都排第一）")
    print(f"首名過地板  : {floor_pass}/{len(cases)}   （僅供對照，陰性對照題應該不過）")


if __name__ == "__main__":
    main()