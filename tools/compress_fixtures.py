# tools/compress_fixtures.py
"""
`_compress` 回歸 fixture（《現況》第八章 P5）。

🔴 **本檔是「先立 fixture 再改 prompt」的前半**——⚠️ 立完之前不要動
`memory_manager._compress()` 的 prompt，否則改完仍然只能靠下一次偶遇來驗證
（P5 那一列的原話）。

每一組 fixture 是一個三元組：
    existing  既有的 compressed_memory（可為空字串）
    turns     這一批要壓縮的對話（與 _compress 收到的 turns 同形狀）
    expect    期望行為，以「必須成立的斷言」表示，不是期望的字面輸出

⚠️ **刻意不比對字面輸出**：壓縮是 LLM 呼叫，逐字比對必然脆弱且每次改 prompt
都要重寫期望值。改為檢查**三條規則各自的失效形狀有沒有出現**——那才是
《歸檔》D0001 記錄的東西。

fixture 來源：《歸檔》D0011（觸發條件由第二個樣本收斂：低資訊量時失效，
一輪只有一兩句可壓就會照抄）。
⚠️ 原始現場已因刪除 data/memory.json 滅失，本檔是**重新構造**不是回放。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Fixture:
    name: str
    existing: str
    turns: list[dict]
    # 每個檢查是 (描述, 函式)，函式收 (summary, evicted) 回傳 True 表示通過
    checks: list[tuple[str, Callable[[str, str], bool]]] = field(default_factory=list)
    note: str = ""


def _no_verbatim(min_len: int = 12):
    """規則二：新摘要不得整句照抄對話原文。

    ⚠️ 門檻取 min_len 是因為短句天然會重合（「好」「知道了」），
    把它們算成照抄會讓檢查變成偽陽性製造機——而過度敏感的檢查
    會被習慣性忽略，最後等於沒有檢查（《歸檔》M0018 記過同一形狀）。
    """
    def check(summary: str, _evicted: str, turns: list[dict]) -> bool:
        for m in turns:
            line = m["content"].strip()
            if len(line) >= min_len and line in summary:
                return False
        return True
    return check


# ── 五組 fixture ──────────────────────────────────────────────────

FIXTURES = [
    # 1. 🔴 主目標：低資訊量。這是 D0011 收斂出來的觸發條件本身。
    Fixture(
        name="low_information_single_turn",
        existing=(
            "邪狼正在開發奧狐這個專案。\n"
            "他最近在處理記憶壓縮的部分。\n"
            "他養了一隻叫麻糬的貓。\n"
        ),
        turns=[
            {"role": "user", "content": "今天好累"},
            {"role": "assistant", "content": "那就早點睡吧。"},
        ],
        note="一輪、兩句、幾乎沒有新資訊 ⇒ 這是最容易觸發「照抄」的形狀",
        checks=[
            ("舊摘要三條資訊不得消失（規則一）",
             lambda s, e: all(k in s or k in e for k in ("奧狐", "記憶壓縮", "麻糬"))),
            ("不得整句照抄對話原文（規則二）", None),   # 由 _no_verbatim 掛上
            ("不得出現『沒有新的決定』這類填充句（規則三）",
             lambda s, e: not any(x in s for x in ("沒有新的", "沒有改變", "未提及", "無明顯"))),
        ],
    ),

    # 2. 規則一的正面案例：新對話明確更新舊狀態 ⇒ 舊的該被取代且列進淘汰
    Fixture(
        name="explicit_update_supersedes",
        existing="邪狼的筆電還沒修好。\n他打算這週把它送修。\n",
        turns=[
            {"role": "user", "content": "筆電修好了，昨天拿回來的，換了電池"},
            {"role": "assistant", "content": "那就好，這次撐久一點吧。"},
        ],
        note="舊狀態被明確推翻 ⇒ 新摘要應反映『已修好』，舊句應進淘汰",
        checks=[
            ("新摘要不得仍宣稱『還沒修好』",
             lambda s, e: "還沒修好" not in s),
            ("被取代的舊資訊要出現在淘汰清單",
             lambda s, e: e.strip() not in ("", "無")),
        ],
    ),

    # 3. 規則三：完全沒有可寫的東西
    Fixture(
        name="nothing_worth_recording",
        existing="邪狼在寫奧狐的記憶模組。\n",
        turns=[
            {"role": "user", "content": "嗯"},
            {"role": "assistant", "content": "嗯。"},
        ],
        note="⭐ 正確行為是『維持既有摘要不變』——這正是 P5 那一列寫的"
             "「修法方向不是加強禁令，是給無話可壓時一條合法出路」",
        checks=[
            ("既有摘要必須原樣保留",
             lambda s, e: "記憶模組" in s or "奧狐" in s),
            ("不得憑空產生新內容",
             lambda s, e: len([x for x in s.split("\n") if x.strip()]) <= 2),
        ],
    ),

    # 4. 句數上限：擠不進去的要進淘汰，不得無聲消失
    Fixture(
        name="overflow_goes_to_evicted",
        existing="\n".join(f"舊資訊第 {i} 條，內容各不相同。" for i in range(1, 9)) + "\n",
        turns=[
            {"role": "user", "content": "我決定把發話權那條線獨立出來做，不跟記憶綁在一起"},
            {"role": "assistant", "content": "那樣兩邊都好改。"},
        ],
        note="舊摘要已滿 8 句 ＋ 一條明確決定 ⇒ 必然有東西排不進去",
        checks=[
            ("新摘要不得超過 MAX_SUMMARY_SENTENCES",
             lambda s, e: len([x for x in s.split("\n") if x.strip()]) <= 8),
            ("排不進去的必須出現在淘汰清單，不得無聲消失",
             lambda s, e: e.strip() not in ("", "無")),
            ("『明確做出的決定』應優先於『其他』被保留",
             lambda s, e: "發話權" in s or "獨立" in s),
        ],
    ),

    # 5. 冷啟動：既有摘要為空
    Fixture(
        name="cold_start_empty_summary",
        existing="",
        turns=[
            {"role": "user", "content": "我叫邪狼，在做一個叫奧狐的 AI VTuber"},
            {"role": "assistant", "content": "所以你想做出比 neuro 更像那回事的東西。"},
        ],
        note="⚠️ existing 為空時 prompt 少掉整個 [之前的摘要] 區塊，"
             "是一條與其他四組不同的路徑",
        checks=[
            ("人名必須被保留（取捨順序最高）",
             lambda s, e: "邪狼" in s),
            ("淘汰清單應為『無』",
             lambda s, e: e.strip() in ("無", "")),
        ],
    ),
]

# 把 _no_verbatim 掛進第一組（需要 turns，故不能寫在 lambda 裡）
_f = FIXTURES[0]
_vb = _no_verbatim()
_f.checks[1] = (_f.checks[1][0], lambda s, e, _t=_f.turns: _vb(s, e, _t))


# ── 執行 ──────────────────────────────────────────────────────────
#
# ⚠️ 本檔刻意**不**自己建立 MemoryManager：壓縮需要 LLM 連線，而是否要在
#    CI／離線環境跑是另一個決定。呼叫端自行注入一個
#    `run(existing, turns) -> (summary, evicted)`。
#
# 🔴 建議的最小跑法（手動）：
#      1. /test on（記憶走獨立測試檔，正式記憶完全不動）
#      2. 對每組 fixture：設 compressed_memory = existing、
#         working_memory = turns，呼叫 _compress(turns)
#      3. 逐條檢查，失敗的印出來
#    ⚠️ 步驟 1 不可省——《現況》第八章 C8 記過 `/test on` 會刪掉整份測試記憶檔，
#       而那正是我們要的隔離；但也代表**不要在有正式記憶時直接跑**。

def run_all(run) -> list[str]:
    """回傳失敗描述清單；空清單表示全過。"""
    failures = []
    for f in FIXTURES:
        summary, evicted = run(f.existing, f.turns)
        for desc, check in f.checks:
            try:
                ok = check(summary, evicted)
            except Exception as exc:            # 檢查自己壞掉要大聲
                failures.append(f"{f.name}｜{desc}｜檢查拋例外：{exc}")
                continue
            if not ok:
                failures.append(f"{f.name}｜{desc}")
    return failures
