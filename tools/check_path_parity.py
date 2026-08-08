# tools/check_path_parity.py
# 一次性診斷：比對三條檢索路徑對同一個查詢的結果是否一致。
#
#   A. collection.query()  ← tools/rag_probe.py 走的路（近似最近鄰）
#   B. RAGManager.query()  ← 正式對話走的路（A + 門檻 + 去重）
#   C. 暴力全量餘弦        ← 標準答案（精確，不是近似）
#
# ⚠️ C 存在的理由：HNSW 是**近似**最近鄰，順序本來就不保證等於真實排序。
#    沒有 C 的話，A 與 B 不一致時只知道兩者不同，不知道誰對——而「ANN 抖動」
#    與「程式碼有 bug」的修法完全相反。
#
# ⚠️ 唯讀：不呼叫 add()／delete()／upsert()。B 路徑會真的跑 RAGManager.query()，
#    但 RagStats 已於本輪移除，不再有統計副作用。
#
# 用法（於專案根目錄執行）：
#     python -m tools.check_path_parity
#     python -m tools.check_path_parity "妳跑在什麼電腦上"    指定查詢

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: F401,E402  ← HF_HUB_OFFLINE 必須在 huggingface_hub 之前生效
from config import RAG_SIMILARITY_THRESHOLD, RAG_TOP_K  # noqa: E402
from core.rag_manager import RAGManager, RETIRED_FIELD  # noqa: E402

DEFAULT_Q = "妳叫什麼名字"
REPEATS = 3


def short(s, n=34):
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def main():
    q = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_Q

    rag = RAGManager()
    if not rag.enabled:
        sys.exit("❌ RAG 目前為停用狀態（TEST_MODE）。請先關閉 TEST_MODE。")

    count = rag.collection.count()
    n_req = min(RAG_TOP_K * 3, count)
    print(f"\n查詢：{q}")
    print(f"庫內 {count} 筆　n_results={n_req}　門檻 {RAG_SIMILARITY_THRESHOLD}　TOP_K {RAG_TOP_K}")

    # ── A. collection.query()，同一進程內重複多次 ──────────────
    runs = []
    for i in range(REPEATS):
        res = rag.collection.query(
            query_texts=[q],
            n_results=n_req,
            include=["documents", "distances", "metadatas"],
            where={RETIRED_FIELD: False},
        )
        rows = [
            (1 - dist, doc, (meta or {}).get("answer") or doc)
            for doc, dist, meta in zip(
                res["documents"][0], res["distances"][0], res["metadatas"][0]
            )
        ]
        runs.append(rows)

    print("\n" + "═" * 70)
    print(f"A. collection.query() 原始候選（rag_probe 走的路，同進程跑 {REPEATS} 次）")
    print("═" * 70)
    for i, sim_doc in enumerate(runs[0], 1):
        print(f"  {i:>2}. {sim_doc[0]:.4f}  {short(sim_doc[1])}")

    same = all(
        [(round(s, 6), d) for s, d, _ in r] == [(round(s, 6), d) for s, d, _ in runs[0]]
        for r in runs[1:]
    )
    print(f"\n  同進程 {REPEATS} 次是否完全相同：{'✅ 是' if same else '🔴 否——檢索本身非決定性'}")
    if not same:
        for i, r in enumerate(runs[1:], 2):
            print(f"    第 {i} 次：" + "、".join(f"{s:.3f} {short(d, 12)}" for s, d, _ in r))

    # ── B. RAGManager.query() 實際回傳 ────────────────────────
    print("\n" + "═" * 70)
    print("B. RAGManager.query() 實際回傳（正式對話走的路）")
    print("═" * 70)
    payloads = rag.query(q)
    for i, p in enumerate(payloads, 1):
        print(f"  {i:>2}. {short(p, 60)}")
    if not payloads:
        print("  （空）")

    # ── C. 暴力全量餘弦：標準答案 ─────────────────────────────
    print("\n" + "═" * 70)
    print("C. 暴力全量餘弦（精確排序，不是近似——這是標準答案）")
    print("═" * 70)
    try:
        import numpy as np

        all_rows = rag.collection.get(include=["documents", "embeddings", "metadatas"])
        mat = np.asarray(all_rows["embeddings"], dtype=float)
        qv = np.asarray(rag.embedding_fn([q])[0], dtype=float)

        mat_n = mat / np.linalg.norm(mat, axis=1, keepdims=True)
        qv_n = qv / np.linalg.norm(qv)
        sims = mat_n @ qv_n

        order = np.argsort(-sims)[:n_req]
        exact = [(float(sims[j]), all_rows["documents"][j]) for j in order]
        for i, (s, d) in enumerate(exact, 1):
            print(f"  {i:>2}. {s:.4f}  {short(d)}")

        # A 與 C 的差集：ANN 漏掉了什麼
        a_docs = [d for _, d, _ in runs[0]]
        c_docs = [d for _, d in exact]
        missed = [d for d in c_docs if d not in a_docs]
        extra = [d for d in a_docs if d not in c_docs]
        print("\n  A 相對於 C：", end="")
        if not missed and not extra and a_docs == c_docs:
            print("✅ 完全一致（含順序）")
        elif not missed and not extra:
            print("⚠️ 集合相同但順序不同")
            print(f"     A 順序：{[short(d, 10) for d in a_docs]}")
            print(f"     C 順序：{[short(d, 10) for d in c_docs]}")
        else:
            print("🔴 集合不同——HNSW 漏抓")
            for d in missed:
                print(f"     C 有 A 沒有：{short(d, 40)}")
            for d in extra:
                print(f"     A 有 C 沒有：{short(d, 40)}")
    except Exception as e:
        print(f"  ⚠️ 暴力比對失敗（不影響 A/B 判讀）：{type(e).__name__}: {e}")

    # ── 判讀提示 ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("判讀：")
    print("  · A 同進程多次不同        → 檢索非決定性，rag_probe 的數字不可重現")
    print("  · A 同進程相同、但與跨進程觀察到的不同 → 進程間差異（索引載入順序等）")
    print("  · A 與 C 集合不同         → HNSW 近似誤差，非程式碼問題")
    print("  · A 與 C 一致但 B 少了高分項 → RAGManager.query() 有 bug，我改壞了")
    print("─" * 70 + "\n")


if __name__ == "__main__":
    main()