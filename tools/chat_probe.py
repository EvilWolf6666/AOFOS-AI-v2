"""對話探測：照劇本自動跑一場對話，量延遲、看注入、測背景取消與 observation。

用途：換後端（llama-server）與記憶說明位置（MEMORY_CONFIG["note_position"]）之後的回歸實測，
不用人在終端機一句一句打。走的是正式路徑：LLMClient＋Persona＋MemoryManager＋Pipeline。

🔴 不碰正式記憶檔：先用 SQLite backup 把 data/memory.db 複製到 logs/chat_probe_<時間>/，
   整場（含背景判斷建主題）都只寫那份副本。正式檔一個字不動。
🔴 拿 AOFOS 的執行鎖：AOFOS 開著就不跑（兩邊會搶同一個推理槽，量到的數字不能用）；
   獨立終端的背景判斷（judge_worker）看到鎖會自己讓路。
⚠️ 為了不用等太久，副本上的背景判斷閒置門檻與「一段講完」門檻都改成 PROBE_IDLE_S 秒
   （只影響這次探測，不改 config）。

用法（專案根目錄，llama-server 先開著）：python -m tools.chat_probe
結果：畫面印摘要；每輪完整紀錄（含送出的 messages）落檔 logs/chat_probe_<時間>/result.json
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics
import sys
import time
from datetime import datetime

import config
from core.agent.events import OBSERVATION, UTTERANCE, Event
from core.agent.pipeline import Pipeline
from core.llm_client import LLMClient
from core.memory.manager import MemoryManager
from core.memory.run_lock import RunLock, lock_path
from core.persona import Persona

PROBE_IDLE_S = 10.0

# 劇本：內容各不相同（seed 修正後應不再重複），並刻意放幾句「以前講過的事」讓主題被撈出來
SCRIPT = [
    "晚安，今天好累喔",
    "你還記得我上次說要搬去哪裡嗎？",
    "你的生日是幾月幾號？",
    "我最近在打艾爾登法環",
    "你覺得我熬夜是在逃避事情嗎？",
    "今天中午吃了一碗拉麵",
    "你記得我們上次聊了什麼嗎？",
    "明天要早起上班，好煩",
    "如果有人在直播斗內你，你會說什麼？",
    "好啦我先去洗澡",
]
CANCEL_INPUT = "我回來了，剛剛在發呆"
OBSERVATION_TEXT = "使用者切換到 Minecraft"


def backup_db(src: str, dst: str) -> None:
    """用 SQLite backup API 複製（正式檔若正被別人開著也一致），來源唯讀開啟。"""
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(dst)
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()


class Recorder:
    """包住 LLMClient／Persona 的幾個方法，記下每輪的 TTFT、server timings 與送出的 messages。"""

    def __init__(self, llm: LLMClient, persona: Persona) -> None:
        self.cur: dict = {}
        orig_chat, orig_timings, orig_build = llm.chat, llm._log_timings, persona.build_messages

        def chat(messages, purpose="dialogue", on_delta=None):
            t0 = time.monotonic()

            def delta(text):
                if "ttft_ms" not in self.cur:
                    self.cur["ttft_ms"] = round((time.monotonic() - t0) * 1000)
                if on_delta is not None:
                    on_delta(text)
            reply = orig_chat(messages, purpose=purpose, on_delta=delta)
            self.cur["total_ms"] = round((time.monotonic() - t0) * 1000)
            return reply

        def log_timings(obj, purpose):
            t = (getattr(obj, "model_extra", None) or {}).get("timings") if obj is not None else None
            if t:
                self.cur["timings"] = {k: t.get(k) for k in
                                       ("prompt_n", "prompt_ms", "cache_n", "predicted_n", "predicted_ms")}
            return orig_timings(obj, purpose)

        def build(history, user_input):
            msgs = orig_build(history, user_input)
            self.cur["messages"] = msgs
            return msgs

        llm.chat, llm._log_timings, persona.build_messages = chat, log_timings, build

    def start(self, label: str, text: str) -> None:
        self.cur = {"label": label, "input": text}


def queue_wait(rec: dict) -> int | None:
    """TTFT 扣掉 server 自己算 prompt 的時間＝在 server 排隊（推理槽被別的請求佔著）的時間。
    ⭐ 背景取消有沒有真的讓出推理槽，看這個，不看 TTFT：TTFT 混了 prefill 長短。"""
    t = rec.get("timings") or {}
    if rec.get("ttft_ms") is None or t.get("prompt_ms") is None:
        return None
    return max(0, round(rec["ttft_ms"] - t["prompt_ms"]))


def summarize(rec: dict, memory_note_fixed: bool) -> str:
    t = rec.get("timings") or {}
    msgs = rec.get("messages") or []
    sysm = [m["content"] for m in msgs if m["role"] == "system"]
    block = next((s for s in sysm if s.startswith(config.MEMORY_CONFIG["block_open"])), "")
    n_topics = sum(1 for line in block.splitlines() if line.startswith("·"))
    wait = queue_wait(rec)
    return (f"[{rec['label']}] TTFT {rec.get('ttft_ms', '—')} ms｜排隊 {wait if wait is not None else '—'} ms"
            f"｜重算 {t.get('prompt_n', '—')} tok"
            f"（快取 {t.get('cache_n', '—')}）｜記憶 {n_topics} 件"
            f"\n    你：{rec['input']}\n    奧狐：{(rec.get('reply') or '（空）')[:80]}")


def main() -> int:
    real_db = config.MEMORY_CONFIG["db_path"]
    lock = RunLock(lock_path(real_db))
    if not lock.acquire():
        print("❌ AOFOS 正在執行（或另一個探測在跑），先關掉再探測。")
        return 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(config.PROJECT_ROOT, "logs", f"chat_probe_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    work_db = os.path.join(out_dir, "memory.db")
    records: list[dict] = []
    memory = None
    try:
        if os.path.exists(real_db):
            backup_db(real_db, work_db)
            print(f"已複製正式記憶檔到 {work_db}（正式檔不動）")
        else:
            print("⚠️ 找不到正式記憶檔，從空的記憶開始（撈不到任何主題）")

        llm = LLMClient()
        persona = Persona()
        memory = MemoryManager(llm, db_path=work_db,
                               test_db_path=os.path.join(out_dir, "memory_test.db"),
                               cfg_override={"judge_idle_s": PROBE_IDLE_S,
                                             "judge_segment_close_s": PROBE_IDLE_S})
        pipeline = Pipeline(llm, persona, memory)
        rec = Recorder(llm, persona)
        fixed = config.MEMORY_CONFIG.get("note_position") == "fixed"
        print(f"note_position={config.MEMORY_CONFIG.get('note_position')}｜seed={llm.seed}")

        # 等向量載入：沒載好主題只走關鍵字一路，量到的注入不是平常的樣子
        emb = getattr(memory, "_emb", None)
        if emb is not None:
            print("等向量檢索載入（最多 180 秒）…")
            end = time.monotonic() + 180
            while not emb.ready.wait(1.0) and not emb.failed and time.monotonic() < end:
                pass
            if not emb.ready.is_set():
                print(f"⚠️ 向量沒載好（{emb.failed or '逾時'}），本次主題只走關鍵字")

        def run(label: str, event: Event, observation: bool = False) -> dict:
            rec.start(label, event.payload)
            res = pipeline.handle_observation(event) if observation else pipeline.handle_turn(event)
            pipeline.finalize_turn()
            rec.cur["reply"] = res.reply
            records.append(rec.cur)
            print(summarize(rec.cur, fixed), flush=True)
            return rec.cur

        print("\n══ 劇本對話 ══")
        for i, text in enumerate(SCRIPT, 1):
            run(f"第{i}輪", Event("user_text", UTTERANCE, text))

        print(f"\n══ 背景取消：閒置等背景判斷開跑（門檻 {PROBE_IDLE_S:.0f} 秒），開跑後 1 秒插話 ══")
        judge = getattr(memory, "_judge", None)
        started = False
        end = time.monotonic() + 120
        while judge is not None and time.monotonic() < end:
            if judge.busy:
                started = True
                break
            time.sleep(0.2)
        if not started:
            print("⚠️ 120 秒內背景判斷沒開跑：本項沒測到（看 log 的「背景判斷」相關行）")
        else:
            time.sleep(1.0)
        r = run("插話" if started else "插話（背景沒開跑）", Event("user_text", UTTERANCE, CANCEL_INPUT))
        r["background_was_running"] = started

        print("\n══ observation（role＝" + config.EVENT_CONFIG["observation_role"] + "）══")
        o = run("observation", Event("window", OBSERVATION, OBSERVATION_TEXT), observation=True)
        o["ok"] = bool(o.get("reply"))

        # ── 摘要 ──
        base = [x["ttft_ms"] for x in records[1:len(SCRIPT)] if x.get("ttft_ms")]   # 第 1 輪沒快取，不算
        recompute = [x["timings"]["prompt_n"] for x in records[1:len(SCRIPT)] if x.get("timings")]
        replies = [x.get("reply") for x in records[:len(SCRIPT)]]
        dup = len(replies) - len(set(replies))
        print("\n══ 摘要 ══")
        if base:
            rc = f"{statistics.median(recompute):.0f} tok" if recompute else "—（回應沒有 timings）"
            print(f"劇本第 2～{len(SCRIPT)} 輪：TTFT 中位數 {statistics.median(base):.0f} ms"
                  f"（{min(base)}～{max(base)}）｜每輪重算中位數 {rc}")
        print(f"回答重複：{dup} 句（seed 修正後應為 0）")
        waits = [(x["label"], queue_wait(x)) for x in records]
        slow = [f"{lb} {w} ms" for lb, w in waits if w is not None and w > 2000]
        if started:
            print(f"插話：排隊 {queue_wait(r)} ms（>2000＝取消後推理槽沒立刻讓出）")
        print("排隊超過 2 秒的輪次：" + ("、".join(slow) if slow else "無"))
        print(f"observation：{'✅ 有回應' if o['ok'] else '❌ 回應是空的（看 log 有沒有 HTTP 400／LLM 呼叫失敗）'}")
        return 0
    finally:
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump({"note_position": config.MEMORY_CONFIG.get("note_position"),
                       "seed": config.LLM_CONFIG.get("seed"), "records": records},
                      f, ensure_ascii=False, indent=2)
        print(f"\n完整紀錄：{os.path.join(out_dir, 'result.json')}")
        if memory is not None:
            memory.compress_on_exit()   # 只收尾副本
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
