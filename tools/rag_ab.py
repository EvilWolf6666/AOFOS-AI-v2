# tools/rag_ab.py
# 對同一個問題比較多種寫法的相似度，不寫入 ChromaDB。
#
# 用途：世界觀寫法的變數隔離實驗。灌進 DB 再測一輪要花好幾分鐘，而寫法比較
# 只需要 embedding，不需要向量庫。
#
# ⚠️ 與 seed_rag.py、rag_probe.py 同，不是腳手架，不列入清理清單——每次調整
#    寫法都會再用到。
#
# ⚠️ 直接用 RAGManager 的 embedding_fn，確保與正式路徑同一個模型、同一組權重。
#    自己另外載一次 SentenceTransformer 會變成第二個來源。
#
# 題本格式（tools/ab_cases.txt）：
#     ? 問題
#     - 候選寫法一
#     - 候選寫法二
#     （空行分隔下一組；# 開頭為註解）
#
# 用法：python -m tools.rag_ab  [題本路徑]

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAG_SIMILARITY_THRESHOLD  # noqa: E402
from core.memory.rag import RAGManager  # noqa: E402

DEFAULT_CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ab_cases.txt")


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def parse(path):
    cases, q, cands = [], None, []
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("?"):
            if q:
                cases.append((q, cands))
            q, cands = line[1:].strip(), []
        elif line.startswith("-"):
            cands.append(line[1:].strip())
    if q:
        cases.append((q, cands))
    return cases


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CASES
    if not os.path.exists(path):
        sys.exit(f"❌ 找不到題本：{path}")

    cases = parse(path)
    if not cases:
        sys.exit("❌ 題本沒有解析出任何比較組")

    rag = RAGManager()
    if not rag.enabled:
        sys.exit("❌ RAG 為停用狀態（TEST_MODE）。請先關閉 TEST_MODE。")
    ef = rag.embedding_fn

    print(f"目前門檻 {RAG_SIMILARITY_THRESHOLD}\n")

    for q, cands in cases:
        if not cands:
            continue
        vecs = ef([q] + cands)
        qv, cvs = vecs[0], vecs[1:]
        scored = sorted(
            ((cosine(qv, v), c) for v, c in zip(cvs, cands)),
            key=lambda x: -x[0],
        )
        best = scored[0][0]
        print(f"❓ {q}")
        for sim, c in scored:
            delta = sim - best
            tag = "★" if sim == best else f"{delta:+.3f}"
            print(f"   {sim:.3f} {tag:>7}  {c}")
        print()


if __name__ == "__main__":
    main()
