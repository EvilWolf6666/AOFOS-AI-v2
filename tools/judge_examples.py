# tools/judge_examples.py
"""
例句實測：把句子逐句送真的推理後端（llama-server），看 P(建)、建不建判對沒有、主題句寫得像不像。

用法：python -m tools.judge_examples            （只跑下方 HELD_OUT：不在提示詞裡的句子）
      python -m tools.judge_examples --all      （另外把 prompts/memory_judge.txt【例子】也跑一遍）
（llama-server 照平常開著（start\啟動推理後端.bat）。不碰記憶檔；原始回應整份落檔 logs/judge_examples_*.json。）

⭐ 分兩組是為了不「拿考古題考」（2026-10-09 第一次實測 19 句全在提示詞裡、分數偏樂觀）：
   HELD_OUT 刻意不放進提示詞；提示詞的例子只用 --all 順便看。
⭐ 建不建自動判對錯；主題句只並排印出（期望 vs 模型），由人看——寫法沒有唯一答案。
⭐ 送法與正式判斷同一份（core/memory/judge.py 的 build_messages／p_build／split_answer），不另寫。
⚠️ 連續跑時規則前綴吃得到 prompt cache（第一句最慢）；實際對話中背景判斷前後夾著對話呼叫，
   **每段都可能是冷的** ⇒ 第一句的秒數比較接近實際背景的耗時。
"""
from __future__ import annotations

import json
import os
import sys
import time

from config import MEMORY_CONFIG
from core.llm_client import BackgroundLLM
from core.memory.judge import build_messages, decide, load_prompt, norm_topic, p_build, split_answer

# （對話，「／」分句；期望輸出：第一行建／不建，之後是期望主題句）——不得與提示詞【例子】重複
HELD_OUT: list[tuple[str, list[str]]] = [
    ("邪狼：嗯，對。", ["不建"]),
    ("邪狼：哈哈哈。／奧狐：笑什麼啦", ["不建"]),
    ("邪狼：其實我是外星人派來的。", ["不建"]),
    ("奧狐：我覺得你每次熬夜都是在逃避事情。", ["建", "看法｜無｜奧狐覺得邪狼每次熬夜都是在逃避事情"]),
    ("邪狼：今天直播第一次有人斗內你耶。", ["建", "事件｜無｜奧狐的直播第一次有人斗內"]),
    ("邪狼：你的生日就定在 3 月 11 號吧。", ["建", "身分｜奧狐.生日｜奧狐的生日是 3 月 11 號"]),
    ("邪狼：我不搬台中了，改去高雄。", ["建", "狀態｜邪狼.住處｜邪狼不搬台中了，改搬去高雄"]),
    ("奧狐：我收回，你熬夜不是在逃避啦。", ["建", "看法｜無｜奧狐收回說法，不再覺得邪狼熬夜是在逃避"]),
    ("邪狼：最近在打艾爾登法環。／奧狐：打到哪了？／邪狼：卡在女武神。／奧狐：那隻超難的吧！",
     ["建", "話題｜無｜邪狼和奧狐聊了艾爾登法環的女武神"]),
    ("邪狼：最近在打艾爾登法環。／奧狐：好玩嗎？／邪狼：還行。對了我明天要面試。／奧狐：加油！",
     ["建", "狀態｜無｜邪狼明天要面試（先提了幾句艾爾登法環）"]),
]


def main(argv: list[str]) -> int:
    cfg = MEMORY_CONFIG
    system, _types, examples = load_prompt(cfg["judge_prompt_path"], max_topics=cfg["judge_max_topics"])
    cases = [("未見過", d, o) for d, o in HELD_OUT]
    if "--all" in argv:
        cases += [("提示詞內", d, o) for d, o in examples]
    ex_texts = {norm_topic(ln.replace("|", "｜").split("｜")[-1]) for _, o in examples for ln in o[1:]}
    llm = BackgroundLLM(params=cfg["judge_params"])
    thr = cfg["judge_build_threshold"]
    rows, wrong = [], 0
    for group, dialog, want_lines in cases:
        seg = "\n".join(x.strip() for x in dialog.split("／"))
        want = want_lines[0]
        t0 = time.monotonic()
        out, lps = llm.call_lp(build_messages(system, seg), purpose="例句判斷",
                               max_tokens=cfg["judge_max_tokens"], top_logprobs=cfg["judge_top_logprobs"])
        dt = time.monotonic() - t0
        said, body = split_answer(out)
        p = p_build(lps)
        got = "建" if decide(said, p, thr)[0] else "不建"   # 與正式判斷同一條規則
        hit = got == want
        wrong += not hit
        pb = "—" if p is None else f"{p:.2f}"
        print(f"{'✅' if hit else '❌'} [{group}] P(建)={pb} 模型答「{said}」→{got}（應{want}）{dt:4.1f}s｜{dialog}")
        for ln in want_lines[1:]:
            print(f"      期望：{ln}")
        for ln in body.strip().splitlines():
            mark = "　⛔照抄例子（正式判斷會濾掉；整段被濾光時改存原句）" if norm_topic(ln.replace("|", "｜").split("｜")[-1]) in ex_texts else ""
            print(f"      模型：{ln}{mark}")
        rows.append({"group": group, "dialog": dialog, "want": want_lines, "said": said, "p_build": p,
                     "decided": got, "ok": hit, "seconds": round(dt, 2), "raw": out, "logprobs": lps})
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", time.strftime("judge_examples_%Y%m%d_%H%M%S.json"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"threshold": thr, "rows": rows}, f, ensure_ascii=False, indent=2, sort_keys=True)
    if any(r["p_build"] is None for r in rows):
        print("⚠️ 有回應拿不到 logprobs（串流沒附？）⇒ 那些 P(建) 是「—」、一律看文字判")
    rule = "看字" if thr is None else f"門檻 {thr}"
    print(f"\n建不建：{len(rows) - wrong}/{len(rows)} 對（{rule}）｜主題句請對照「期望／模型」看｜原始回應：{path}")
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

