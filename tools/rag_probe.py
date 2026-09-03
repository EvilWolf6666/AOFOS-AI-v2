# tools/rag_probe.py
# 逐題印出 RAG 檢索結果與相似度，並計算 recall@k 與 MRR。
#
# ⚠️ 與 seed_rag.py 同，這不是診斷腳手架，不列入 core/short_time_test.py 的清理清單。
#    每次擴充 worldview/ 之後都會再用到，沒有刪除條件。
#
# ⚠️ 刻意不呼叫 RAGManager.query()，直接打 collection，只讀不寫。
#    ⚠️ **原理由已不成立**：本行原本寫的是「避免 record_search() 讓 rag_stats.json
#    的 searches 平白增加、污染淘汰內容命中率的量測（《歸檔》K0003）」，而 RagStats
#    已於 2026-08-08 整組移除（《歸檔》E0013）。
#    現行理由是本檔要算的東西：recall@k 與 MRR 需要**原始排序與 distances**，而
#    RAGManager.query() 只回傳過濾去重後的 payload 字串，排名與相似度都取不到。
#    ⚠️ 代價是量到的**不完全等同正式路徑**（《現況》16之1 已登記）；要比對三條路徑
#    差異請用 tools/check_path_parity.py。
#
# 🔴 **本檔刻意不套用種子過濾（`RAG_INCLUDE_SEED`），這是有意義的分岔，不是漏改。**
#    本檔量的是**檢索側品質**——「這則世界觀寫成這樣、被這樣問的時候撈不撈得到」，
#    每次擴充 worldview/ 之後都要看。若跟著正式路徑把種子濾掉，題本會全數命中 0、
#    工具直接失效，而失效的原因與它要量的東西無關。
#    ⚠️ **代價：`RAG_INCLUDE_SEED = False` 時本檔的 recall@k／MRR 不描述正式路徑的任何
#    行為**——那邊一筆種子都不會回傳。故啟動時強制印出落差警告（見 main()），
#    不靠人讀到這段註解。
#    ⚠️ **與 tools/check_path_parity.py 的處置相反是刻意的**：那支要回答「正式路徑做了
#    什麼」，依單一來源判準必須共用 `_build_where()`；本檔要回答「該撈的撈不撈得到」。
#    📌 本檔的長期定位跟著《現況》11之2「種子的長期去留」那個未裁決項走。
#
# ⚠️ 前提假設：正式路徑是用「使用者輸入原文」去查 RAG。已於 2026-08-05 確認、
#    2026-09-03 複查：memory_manager.get_context() 內為 self.rag.query(user_input)，
#    無改寫或加料。
#    ⚠️ 這裡刻意寫函式名不寫行號——原註解寫「memory_manager.py:160」，複查時該行已漂到
#    209 行。行號會隨任何一次無關的編輯而過期，而過期時**不會有任何徵兆**：它仍然指向
#    一個真實存在的行，只是內容變成別的東西了（同《現況》19之5「指向真實但錯誤的條目」）。
#
# ⚠️ 要看的是 recall@k 與 MRR，不是「過門檻數」。門檻是地板不是閘門——只要正確
#    答案進了上下文，模型就有機會用它。用過門檻數當成績會把「檢索夠好但閘門設錯」
#    誤判成「檢索不夠好」，這兩者的修法完全不同。見《歸檔》E0004。
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

from config import (RAG_INCLUDE_SEED, RAG_SIMILARITY_THRESHOLD,  # noqa: E402
                    RAG_TOP_K)
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

    print(f"庫內 {count} 筆　門檻 {RAG_SIMILARITY_THRESHOLD}　TOP_K {RAG_TOP_K}")
    # 落差要印在數字之前——印在最後會被整頁結果推走
    if not RAG_INCLUDE_SEED:
        print("🔴 RAG_INCLUDE_SEED = False：正式路徑會過濾掉種子，本檔刻意不過濾。")
        print("   ⇒ 以下 recall@k／MRR 是**檢索側品質**，"
              "不描述正式對話的行為（那邊一筆種子都不會回傳）。")
    print()

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