"""對話探測：照劇本自動跑一場或連續幾場對話，量延遲、看注入、測背景取消與 observation。

版本：2026-10-25 14:49（前一版：2026-10-25 14:45）
🔺 2026-10-25 改版（《現況》第八章 P1 第 ③ 點基線＋P22）：
   ① 劇本改成 23 輪：中段兩則 observation，後面接「剛剛在玩什麼」「現在在幹嘛」（P22：驗方案 α
      `situation_in_history="all"` 在正式路徑上的歷史）；長度超過 Working Memory 上限，量丟批那一輪。
   ② 可連跑多場（--sessions），第 k 場用第 k−1 場的副本當來源 ⇒ 第二場起才有開場可帶。
      也可用 --from-prev／--from 指定上一次探測的副本當來源。
   ③ 記憶區塊 2026-10-23 起併進 user 訊息（block_position="in_user"），舊版摘要只在 system
      訊息裡找區塊 ⇒ 注入件數永遠是 0。改成兩處都找。
   ④ 每輪自動標記四項觀察（字面 `[狀態]`、照講上一句、「你們」、丟批那輪 TTFT）與 P22 核對；
      判讀門檻在 THRESHOLDS，是 Claude 擬的草案，邪狼可直接改。
   ⑤ 14:45：--v4 on／off 只在副本上開關主題建立（不改 config）；--baseline 預設 on。
      邪狼 2026-10-25：基線開主題只看效能（向量載入、主題檢索與注入、背景判斷與取消對延遲的影響），
      不看主題內容——判斷模型還沒換（P1 第 ⑤ 點），建出來的主題品質不當結論。
   ⑥ 14:49：劇本第 16 輪後加一段閒置，等背景判斷開跑後插話（第 17 輪）⇒ 量「歷史已長、判斷進行中」
      的取消代價（舊劇本只在結尾量，那時判斷很少有東西可判）。跑完把本次副本上的主題全部停用
      （retired=1，不刪）：邪狼 2026-10-25，測試時的主題句不作為實際使用。

用途：換後端、改參數、改記憶注入之後的回歸實測，不用人在終端機一句一句打。
走的是正式路徑：LLMClient＋Persona＋MemoryManager＋Pipeline。

🔴 不碰正式記憶檔：先用 SQLite backup 把來源 DB 複製到 logs/runs/chat_probe/chat_probe_<時間>/s<k>/，
   整場（含背景判斷建主題）都只寫那份副本。正式檔一個字不動。
🔴 拿 AOFOS 的執行鎖：AOFOS 開著就不跑（兩邊會搶同一個推理槽，量到的數字不能用）；
   獨立終端的背景判斷（judge_worker）看到鎖會自己讓路。整個連跑期間都拿著鎖。
⚠️ 為了不用等太久，副本上的背景判斷閒置門檻與「一段講完」門檻都改成 PROBE_IDLE_S 秒
   （只影響這次探測，不改 config）。
⚠️ 連跑多場時，事件層切段門檻改成 PROBE_EPISODE_GAP_MIN 分鐘、場與場之間等稍久一點：
   config 的 30 分鐘會讓連著跑的幾場被切成同一段，開場的「上一場」就不是上一場。
   只影響副本，不改 config。

用法（專案根目錄，推理後端先開著）：
  python -m tools.chat_probe                     一場，來源＝正式記憶檔（舊行為）
  python -m tools.chat_probe --baseline          基線：空庫、核對基線條件、預設連跑 5 場
  python -m tools.chat_probe --sessions 3        連跑 3 場
  python -m tools.chat_probe --from-prev         來源＝最近一次探測最後一場的副本
  python -m tools.chat_probe --from <db 路徑>    來源＝指定的副本
  python -m tools.chat_probe --empty             來源＝空庫
結果：畫面印摘要；每輪完整紀錄（含送出的 messages）落檔 logs/runs/chat_probe/chat_probe_<時間>/s<k>/result.json，
      全部場次的總表在 logs/runs/chat_probe/chat_probe_<時間>/summary.json
"""
from __future__ import annotations

import argparse
import difflib
import glob
import hashlib
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

# 探測紀錄放 logs/runs/chat_probe/（2026-10-26 紀錄結構統一，見 core/logger.py 開頭）。
PROBE_RUNS_DIR = os.path.join(config.PROJECT_ROOT, "logs", "runs", "chat_probe")
PROBE_IDLE_S = 10.0
PROBE_EPISODE_GAP_MIN = 3        # 連跑時的切段門檻（分鐘）；劇本內最長的停頓（背景取消等待 ≤120 秒）要小於它
BASELINE_SESSIONS = 5

# 劇本：("say", 文字) 是邪狼說話；("obs", 描述) 是 observation（真實情境格式同 window_watcher，
#    《現況》22之84 五）；第三欄是 P22 的答案核對（回覆要提到其中一個詞）。
# ⚠️ 長度刻意超過 MAX_WORKING_MEMORY（現為 20）：第 21 輪寫回後一次丟 WM_DROP_ROUNDS 輪，
#    第 22 輪送出的開頭改變 ⇒ 快取斷、整段重讀，那一輪就是「丟批那輪」。改 WM 上限時劇本要跟著改。
MINECRAFT = ("Minecraft", "minecraft", "麥塊", "創世神")
YOUTUBE = ("YouTube", "Youtube", "youtube", "YT", "油管", "影片")
SCRIPT = [
    ("say", "晚安，今天好累喔", None),
    ("say", "你還記得我上次說要搬去哪裡嗎？", None),
    ("say", "你的生日是幾月幾號？", None),
    ("say", "我最近在打艾爾登法環", None),
    ("say", "你覺得我熬夜是在逃避事情嗎？", None),
    ("obs", "邪狼剛才在 Discord 待了 3 分鐘，現在切到 Minecraft。", None),
    ("say", "今天中午吃了一碗拉麵", None),
    ("say", "你記得我們上次聊了什麼嗎？", None),
    ("say", "明天要早起上班，好煩", None),
    ("obs", "邪狼剛才在 Minecraft 待了 40 分鐘，現在切到 YouTube。", None),
    ("say", "剛剛在玩什麼？", MINECRAFT),
    ("say", "現在在幹嘛？", YOUTUBE),
    ("say", "如果有人在直播斗內你，你會說什麼？", None),
    ("say", "我想說週末去爬山", None),
    ("say", "你喜歡下雨天還是晴天？", None),
    ("say", "最近工作好多報告要寫", None),
    ("idle", None, None),                     # ← 閒置等背景判斷開跑，下一輪在判斷進行中插話
    ("say", "你會不會覺得我很煩？", None),
    ("say", "晚餐不知道要吃什麼", None),
    ("say", "我剛剛是在玩什麼遊戲？", MINECRAFT),
    ("say", "你覺得我該早點睡嗎？", None),
    ("say", "好啦我先去洗澡", None),
    ("say", "洗好了，好舒服", None),           # ← 丟批那輪（第 21 輪寫回後滑出）
    ("say", "我們剛剛聊到哪了？", None),       # ← 丟批後下一輪：快取應該回來
]
ROUNDS = [x for x in SCRIPT if x[0] != "idle"]   # 真正送出的輪次（idle 只是停頓，不算一輪）
CANCEL_INPUT = "我回來了，剛剛在發呆"

# 判讀門檻（2026-10-25 Claude 擬的草案，未經邪狼確認；依據寫在每項後面）
THRESHOLDS = {
    # 照講上一句：和上一句回覆的相似度 ≥ 這個值就算（difflib ratio；0.8 約等於只改幾個字）
    "repeat_ratio": 0.8,
    # 全部場次合計：0 ✅、≤ warn ⚠️、更多 ❌（2026-10-23 區塊併進 user 後 12 輪 0 次）
    "repeat_warn": 1,
    # 「你們」：每場平均出現幾句以下算 ⚠️（0 ✅；任一場 ≥ fail_per_session ❌；2026-10-24 前 12 句 7 次）
    "youmen_warn_avg": 1.0,
    "youmen_fail_per_session": 2,
    # 丟批那輪 TTFT：平常 1.2～3.3 秒；超過 warn 已是明顯卡頓，超過 fail 需要處理（2026-10-24 量到 82 秒）
    "drop_ttft_warn_ms": 10_000,
    "drop_ttft_fail_ms": 30_000,
    # P22 答案核對：答對的場次比例（瀏覽器直打 all：「現在在幹嘛」6/6、「剛剛在玩什麼」完整 4/6）
    "p22_now_min": 0.8,
    "p22_before_min": 0.5,
}


def backup_db(src: str, dst: str) -> None:
    """用 SQLite backup API 複製（正式檔若正被別人開著也一致），來源唯讀開啟。"""
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(dst)
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()


def latest_probe_db() -> str | None:
    """最近一次探測的最後一場副本（目錄名含時間，字典序＝時間序）。舊版探測的副本在目錄根。"""
    dirs = sorted(glob.glob(os.path.join(PROBE_RUNS_DIR, "chat_probe_*")))
    for d in reversed(dirs):
        cands = sorted(glob.glob(os.path.join(d, "s*", "memory.db"))) or \
            [p for p in [os.path.join(d, "memory.db")] if os.path.exists(p)]
        if cands:
            return cands[-1]
    return None


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


def find_block(msgs: list[dict]) -> str:
    """本輪的記憶區塊：in_user 時在最後一則 user 開頭，tail_system 時是一則 system。"""
    op = config.MEMORY_CONFIG["block_open"]
    if msgs and msgs[-1]["role"] == "user" and msgs[-1]["content"].startswith(op):
        return msgs[-1]["content"].split(config.MEMORY_CONFIG["block_close"], 1)[0]
    return next((m["content"] for m in msgs if m["role"] == "system" and m["content"].startswith(op)), "")


def annotate(rec: dict, prev_reply: str | None, sit_prefix: str, s2t) -> None:
    """每輪自動標記（只標可機械判定的；要人判的留給 result.json）。"""
    msgs = rec.get("messages") or []
    block = find_block(msgs)
    reply = rec.get("reply") or ""
    rec["block"] = {
        "has_opening": f"[{config.MEMORY_CONFIG['label_remember']}]" in block,
        "n_topics": sum(1 for line in block.splitlines() if line.startswith("·")),
    }
    # 歷史裡（不含本輪）顯示成情境的 user 訊息數（P22）
    rec["hist_situations"] = sum(1 for m in msgs[:-1] if m["role"] == "user" and m["content"].startswith(sit_prefix))
    rec["flags"] = {
        "status_tag_leak": "[狀態]" in reply,
        "youmen": "你們" in reply,
        "repeat_prev": (prev_reply is not None and bool(reply) and
                        difflib.SequenceMatcher(None, prev_reply, reply).ratio() >= THRESHOLDS["repeat_ratio"]),
        "empty": not reply,
    }
    if s2t is not None and reply:
        rec["flags"]["simplified_chars"] = sum(1 for a, b in zip(reply, s2t(reply)) if a != b)
    if rec.get("expect"):
        rec["p22_hit"] = any(w in reply for w in rec["expect"])


def summarize(rec: dict) -> str:
    t = rec.get("timings") or {}
    wait = queue_wait(rec)
    b = rec.get("block") or {}
    tags = [k for k, v in (rec.get("flags") or {}).items() if v is True]
    if "p22_hit" in rec:
        tags.append("P22✅" if rec["p22_hit"] else "P22❌")
    if rec.get("dropped_before"):
        tags.append("丟批")
    return (f"[{rec['label']}] TTFT {rec.get('ttft_ms', '—')} ms｜排隊 {wait if wait is not None else '—'} ms"
            f"｜重算 {t.get('prompt_n', '—')} tok（快取 {t.get('cache_n', '—')}）"
            f"｜開場 {'有' if b.get('has_opening') else '無'}｜主題 {b.get('n_topics', 0)} 件"
            f"｜歷史情境 {rec.get('hist_situations', 0)} 則" + (f"｜{'、'.join(tags)}" if tags else "")
            + f"\n    你：{rec['input']}\n    奧狐：{(rec.get('reply') or '（空）')[:80]}")


def check_baseline_conditions() -> list[str]:
    """P1 第 ③ 點的基線條件（2026-10-24）。人格是不是 V3 程式驗不了，只記雜湊。"""
    bad = []
    if config.MEMORY_CONFIG.get("block_position") != "in_user":
        bad.append(f"block_position={config.MEMORY_CONFIG.get('block_position')!r}（應為 'in_user'）")
    if config.EVENT_CONFIG.get("observation_delivery") != "in_user":
        bad.append(f"observation_delivery={config.EVENT_CONFIG.get('observation_delivery')!r}（應為 'in_user'）")
    if config.EVENT_CONFIG.get("situation_in_history") != "all":
        bad.append(f"situation_in_history={config.EVENT_CONFIG.get('situation_in_history')!r}（應為 'all'）")
    return bad


def run_params(persona: Persona, llm: LLMClient, gap_min: int | None, v4: bool | None) -> dict:
    """本場實際用的參數全部落檔（施測前六步第 4、5 步的探測版：人格兩組相同、參數全帶入）。"""
    keys_mem = ("block_position", "note_position", "opening_turns", "v4_enabled", "episode_gap_minutes")
    keys_evt = ("observation_delivery", "situation_in_history", "situation_template")
    return {
        "persona_sha256": hashlib.sha256(persona.system_prompt.encode("utf-8")).hexdigest(),
        "persona_first_line": persona.system_prompt.splitlines()[0] if persona.system_prompt else "",
        "llama_server": {k: config.LLAMA_SERVER.get(k) for k in ("build", "model", "alias")},
        "llm": {k: v for k, v in config.LLM_CONFIG.items() if k not in ("api_key",)},
        "seed_in_client": llm.seed,
        "memory": {k: config.MEMORY_CONFIG.get(k) for k in keys_mem},
        "event": {k: config.EVENT_CONFIG.get(k) for k in keys_evt},
        "wm": {"MAX_WORKING_MEMORY": config.MAX_WORKING_MEMORY, "WM_DROP_ROUNDS": config.WM_DROP_ROUNDS},
        "probe": {"idle_s": PROBE_IDLE_S, "episode_gap_min_override": gap_min,
                  "v4_enabled_override": v4, "thresholds": THRESHOLDS},
    }


def make_s2t():
    """簡體字偵測（近似：s2tw 轉換前後不同的字數）。venv 沒有 opencc 就略過，不影響其他項。"""
    try:
        import opencc
    except ImportError:
        return None
    for name in ("s2tw", "s2tw.json"):
        try:
            return opencc.OpenCC(name).convert
        except Exception:  # noqa: BLE001
            continue
    return None


def run_session(k: int, src: str | None, out_dir: str, gap_min: int | None, v4: bool | None,
                s2t) -> tuple[dict, str]:
    """跑一場。回傳 (本場摘要, 本場副本路徑)。"""
    os.makedirs(out_dir, exist_ok=True)
    work_db = os.path.join(out_dir, "memory.db")
    records: list[dict] = []
    memory = None
    params: dict = {}
    print(f"\n████ 第 {k} 場 ████")
    try:
        if src:
            backup_db(src, work_db)
            print(f"來源：{src} → 副本 {work_db}（來源不動）")
        else:
            print("來源：空庫（撈不到任何主題、第一場沒有開場）")

        llm = LLMClient()
        persona = Persona()
        override = {"judge_idle_s": PROBE_IDLE_S, "judge_segment_close_s": PROBE_IDLE_S}
        if gap_min is not None:
            override["episode_gap_minutes"] = gap_min
        if v4 is not None:
            override["v4_enabled"] = v4
        memory = MemoryManager(llm, db_path=work_db,
                               test_db_path=os.path.join(out_dir, "memory_test.db"),
                               cfg_override=override)
        pipeline = Pipeline(llm, persona, memory)
        rec = Recorder(llm, persona)
        params = run_params(persona, llm, gap_min, v4)
        v4_now = memory._v4   # 實際生效的值（override 優先於 config）
        print(f"note_position={config.MEMORY_CONFIG.get('note_position')}｜seed={llm.seed}"
              f"｜v4_enabled={v4_now}{'（探測覆寫，config 不動）' if v4 is not None else ''}"
              f"｜situation_in_history={config.EVENT_CONFIG['situation_in_history']}")

        # 等向量載入：沒載好主題只走關鍵字一路，量到的注入不是平常的樣子
        emb = getattr(memory, "_emb", None)
        if emb is not None:
            print("等向量檢索載入（最多 180 秒）…")
            end = time.monotonic() + 180
            while not emb.ready.wait(1.0) and not emb.failed and time.monotonic() < end:
                pass
            if not emb.ready.is_set():
                print(f"⚠️ 向量沒載好（{emb.failed or '逾時'}），本次主題只走關鍵字")

        sit_prefix = config.EVENT_CONFIG["situation_template"].split("{", 1)[0]
        # drop_pending：上一輪寫回時 Working Memory 變少了（_slide 丟了一批）⇒ 這一輪送出的開頭改變
        state = {"prev_reply": None, "drop_pending": False}

        def run(label: str, event: Event, observation: bool = False, expect=None) -> dict:
            rec.start(label, event.payload)
            before = len(memory.working_memory) // 2
            rec.cur["wm_rounds_before"] = before
            rec.cur["dropped_before"] = state["drop_pending"]
            if expect:
                rec.cur["expect"] = list(expect)
            try:
                res = pipeline.handle_observation(event) if observation else pipeline.handle_turn(event)
                rec.cur["reply"] = res.reply
            except Exception as e:  # noqa: BLE001  一輪失敗不中斷整場，但要記下、摘要裡大聲列出
                rec.cur["reply"] = ""
                rec.cur["error"] = f"{type(e).__name__}: {e}"
            finally:
                pipeline.finalize_turn()
            annotate(rec.cur, state["prev_reply"], sit_prefix, s2t)
            if rec.cur.get("reply"):
                state["prev_reply"] = rec.cur["reply"]
            state["drop_pending"] = len(memory.working_memory) // 2 < before
            records.append(rec.cur)
            print(summarize(rec.cur) + (f"\n    ❌ 例外：{rec.cur['error']}" if rec.cur.get("error") else ""), flush=True)
            return rec.cur

        judge = getattr(memory, "_judge", None)

        def wait_judge(where: str) -> bool | None:
            """閒置等背景判斷開跑，開跑後 1 秒回來（讓下一輪在判斷進行中插話）。
            None＝背景判斷不存在；False＝等不到。"""
            if judge is None:
                print(f"\n⚠️ 背景判斷不存在（v4_enabled={v4_now}）：{where}的取消測試略過")
                return None
            print(f"\n══ {where}：閒置等背景判斷開跑（門檻 {PROBE_IDLE_S:.0f} 秒），開跑後 1 秒插話 ══")
            end = time.monotonic() + 120
            while time.monotonic() < end:
                if judge.busy:
                    time.sleep(1.0)
                    return True
                time.sleep(0.2)
            print("⚠️ 120 秒內背景判斷沒開跑：本項沒測到（看 log 的「背景判斷」相關行）")
            return False

        print("\n══ 劇本對話 ══")
        i, mid = 0, None   # mid：對話中途閒置後的 (結果, 下一輪標記)
        for kind, text, expect in SCRIPT:
            if kind == "idle":
                mid = wait_judge("對話中途")
                continue
            i += 1
            if kind == "obs":
                r = run(f"第{i}輪 observation", Event("window", OBSERVATION, text), observation=True)
            else:
                r = run(f"第{i}輪", Event("user_text", UTTERANCE, text), expect=expect)
            if mid is not None:
                r["background_was_running"] = mid
                r["mid_cancel"] = True
                mid = None

        # 結尾的背景取消（舊劇本就有；歷史短時的對照）
        cancel = None
        started = wait_judge("結尾")
        if started is not None:
            cancel = run("插話" if started else "插話（背景沒開跑）", Event("user_text", UTTERANCE, CANCEL_INPUT))
            cancel["background_was_running"] = started

        return session_summary(k, records, cancel), work_db
    finally:
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump({"session": k, "source": src, "params": params, "records": records},
                      f, ensure_ascii=False, indent=2)
        print(f"本場完整紀錄：{os.path.join(out_dir, 'result.json')}")
        if memory is not None:
            memory.compress_on_exit()   # 只收尾副本（事件層重建 ⇒ 下一場的開場從這裡來）


def session_summary(k: int, records: list[dict], cancel: dict | None) -> dict:
    script_recs = records[:len(ROUNDS)]
    mid = next((r for r in script_recs if r.get("mid_cancel")), None)
    drop = next((r for r in script_recs if r.get("dropped_before")), None)
    normal = [r["ttft_ms"] for i, r in enumerate(script_recs)
              if i > 0 and r.get("ttft_ms") and not r.get("dropped_before")]
    s = {
        "session": k,
        "ttft_median_ms": statistics.median(normal) if normal else None,
        "ttft_range_ms": [min(normal), max(normal)] if normal else None,
        "drop_round": drop["label"] if drop else None,
        "drop_ttft_ms": drop.get("ttft_ms") if drop else None,
        "drop_prompt_n": (drop.get("timings") or {}).get("prompt_n") if drop else None,
        "status_tag_leak": sum(r["flags"]["status_tag_leak"] for r in script_recs),
        "repeat_prev": sum(r["flags"]["repeat_prev"] for r in script_recs),
        "youmen": sum(r["flags"]["youmen"] for r in script_recs),
        "empty_or_error": [r["label"] for r in script_recs if r["flags"]["empty"] or r.get("error")],
        "simplified_chars": sum(r["flags"].get("simplified_chars", 0) for r in script_recs),
        "opening_rounds": [r["label"] for r in script_recs if r["block"]["has_opening"]],
        # 主題那一路的效能（只看數量與排隊，不看主題內容）
        "topic_rounds": sum(1 for r in script_recs if r["block"]["n_topics"]),
        "topics_injected": sum(r["block"]["n_topics"] for r in script_recs),
        "max_queue_ms": max((w for w in (queue_wait(r) for r in script_recs) if w is not None), default=None),
        "p22": {r["input"]: r["p22_hit"] for r in script_recs if "p22_hit" in r},
        # P22 機械核對：第一則 observation 之後、丟批之前的每一輪，歷史裡都要看得到情境
        "p22_history_missing": [],
        "cancel_queue_ms": queue_wait(cancel) if cancel and cancel.get("background_was_running") else None,
        # 對話中途插話（歷史已長、背景判斷進行中）：排隊＝取消後推理槽多久才讓出
        "mid_cancel_round": mid["label"] if mid else None,
        "mid_cancel_queue_ms": queue_wait(mid) if mid and mid.get("background_was_running") else None,
        "mid_cancel_ttft_ms": mid.get("ttft_ms") if mid and mid.get("background_was_running") else None,
    }
    first_obs = next((i for i, (kind, _, _) in enumerate(ROUNDS) if kind == "obs"), None)
    if first_obs is not None:
        for r in script_recs[first_obs + 1:]:
            if r.get("dropped_before"):
                break
            if r.get("hist_situations", 0) < 1:
                s["p22_history_missing"].append(r["label"])
    return s


def verdict(sums: list[dict]) -> list[str]:
    """依 THRESHOLDS 判讀（草案門檻；要人判的項目只提示去看哪裡）。"""
    th, n, out = THRESHOLDS, len(sums), []

    def mark(ok, warn):
        return "✅" if ok else ("⚠️" if warn else "❌")

    leak = sum(s["status_tag_leak"] for s in sums)
    out.append(f"{mark(leak == 0, False)} 第 1 點 `[狀態]` 字面外洩：{leak} 句"
               "｜有沒有『用到』開場的狀態要人判：看第 2 場起的第 1～3 輪")
    rep = sum(s["repeat_prev"] for s in sums)
    out.append(f"{mark(rep == 0, rep <= th['repeat_warn'])} 第 2 點 照講上一句：合計 {rep} 句"
               f"（相似度 ≥ {th['repeat_ratio']}）")
    ym = [s["youmen"] for s in sums]
    avg = sum(ym) / n if n else 0
    out.append(f"{mark(sum(ym) == 0, avg <= th['youmen_warn_avg'] and max(ym, default=0) < th['youmen_fail_per_session'])}"
               f" 第 3 點 「你們」：各場 {ym}（平均 {avg:.1f}）")
    dt = [s["drop_ttft_ms"] for s in sums if s["drop_ttft_ms"] is not None]
    if dt:
        worst = max(dt)
        out.append(f"{mark(worst <= th['drop_ttft_warn_ms'], worst <= th['drop_ttft_fail_ms'])}"
                   f" 第 4 點 丟批那輪 TTFT：各場 {dt} ms（最差 {worst / 1000:.1f} 秒）")
    else:
        out.append("❌ 第 4 點 丟批那輪沒有出現（劇本長度沒超過 Working Memory 上限？）——本項沒測到")
    miss = [f"第{s['session']}場 {s['p22_history_missing']}" for s in sums if s["p22_history_missing"]]
    out.append(f"{mark(not miss, False)} P22 歷史帶著情境：" + ("每輪都有" if not miss else "缺 " + "；".join(miss)))
    for q, need in (("現在在幹嘛？", th["p22_now_min"]), ("剛剛在玩什麼？", th["p22_before_min"]),
                    ("我剛剛是在玩什麼遊戲？", th["p22_before_min"])):
        hits = [s["p22"].get(q) for s in sums if q in s["p22"]]
        rate = sum(bool(h) for h in hits) / len(hits) if hits else 0
        out.append(f"{mark(rate >= need, False)} P22「{q}」答到：{sum(bool(h) for h in hits)}/{len(hits)}"
                   f"（門檻 {need:.0%}）")
    bad = [f"第{s['session']}場 {s['empty_or_error']}" for s in sums if s["empty_or_error"]]
    if bad:
        out.insert(0, "❌ 有空回應或例外（該輪不能當模型表現）：" + "；".join(bad))
    return out


def retire_probe_topics(root: str) -> dict:
    """本次探測各場副本上建出的主題全部停用（retired=1）。

    邪狼 2026-10-25：測試時的主題句不作為實際使用。⇒ 跑完才停用、不在跑的時候關：
    跑的時候要量主題檢索與注入的效能；跑完停用，日後 --from-prev／--from 拿這份副本當來源時
    就撈不到（兩路檢索都經 TopicStore.latest()，只取 retired=0）。
    ⭐ 停用不刪：主題列與原句都留著，result.json 對得回去；要看內容時改回 0 即可。
    只動 logs/ 底下本次的副本，正式記憶檔不在這裡。
    """
    out = {}
    for db in sorted(glob.glob(os.path.join(root, "s*", "memory.db"))):
        cx = sqlite3.connect(db)
        try:
            if cx.execute("SELECT 1 FROM sqlite_master WHERE name='topic'").fetchone() is None:
                out[os.path.basename(os.path.dirname(db))] = 0
                continue
            cx.execute("BEGIN IMMEDIATE")
            n = cx.execute("UPDATE topic SET retired = 1 WHERE retired = 0").rowcount
            cx.execute("COMMIT")
            out[os.path.basename(os.path.dirname(db))] = n
        except Exception as e:  # noqa: BLE001  停用失敗要大聲：那份副本之後會被當成來源撈到
            out[os.path.basename(os.path.dirname(db))] = f"❌ 失敗：{type(e).__name__}: {e}"
        finally:
            cx.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="AOFOS 對話探測")
    ap.add_argument("--sessions", type=int, default=None, help="連跑幾場（第 k 場用第 k−1 場的副本當來源）")
    src_g = ap.add_mutually_exclusive_group()
    src_g.add_argument("--from-prev", action="store_true", help="第一場來源＝最近一次探測最後一場的副本")
    src_g.add_argument("--from", dest="from_db", default=None, help="第一場來源＝指定的 DB 路徑")
    src_g.add_argument("--empty", action="store_true", help="第一場來源＝空庫")
    ap.add_argument("--v4", choices=("on", "off"), default=None,
                    help="只在副本上開關主題建立（不改 config）；不給＝照 config，--baseline 預設 on")
    ap.add_argument("--baseline", action="store_true",
                    help="P1 第 ③ 點基線：空庫、核對基線條件（不符即停）、預設連跑 5 場")
    args = ap.parse_args()

    if args.baseline:
        if args.from_prev or args.from_db:
            print("❌ --baseline 固定從空庫開始，不能和 --from-prev／--from 一起用。")
            return 1
        bad = check_baseline_conditions()
        if bad:
            print("❌ 基線條件不符，不跑（改 config 或拿掉 --baseline）：\n  " + "\n  ".join(bad))
            return 1
        args.empty = True
        if args.v4 is None:
            args.v4 = "on"
    v4 = None if args.v4 is None else args.v4 == "on"
    n_sessions = args.sessions or (BASELINE_SESSIONS if args.baseline else 1)
    if n_sessions < 1:
        print("❌ --sessions 至少 1。")
        return 1

    real_db = config.MEMORY_CONFIG["db_path"]
    if args.empty:
        src = None
    elif args.from_prev:
        src = latest_probe_db()
        if src is None:
            print("❌ --from-prev：logs/runs/chat_probe/ 底下找不到任何探測副本（舊紀錄搬過來了嗎？）。")
            return 1
    elif args.from_db:
        src = args.from_db
        if not os.path.exists(src):
            print(f"❌ --from：找不到 {src}")
            return 1
    else:
        src = real_db if os.path.exists(real_db) else None
        if src is None:
            print("⚠️ 找不到正式記憶檔，從空的記憶開始（撈不到任何主題）")

    lock = RunLock(lock_path(real_db))
    if not lock.acquire():
        print("❌ AOFOS 正在執行（或另一個探測在跑），先關掉再探測。")
        return 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(PROBE_RUNS_DIR, f"chat_probe_{stamp}")
    gap_min = PROBE_EPISODE_GAP_MIN if n_sessions > 1 else None
    s2t = make_s2t()
    if s2t is None:
        print("⚠️ 沒有 opencc：簡體字偵測略過")
    sums: list[dict] = []
    try:
        for k in range(1, n_sessions + 1):
            if k > 1:
                wait = gap_min * 60 + 30
                print(f"\n…等 {wait} 秒再開下一場（切段門檻 {gap_min} 分鐘，讓兩場切成不同段）")
                time.sleep(wait)
            s, src = run_session(k, src, os.path.join(root, f"s{k}"), gap_min, v4, s2t)
            sums.append(s)

        print("\n══ 摘要（判讀門檻是 Claude 擬的草案，見 THRESHOLDS）══")
        for s in sums:
            print(f"第{s['session']}場：TTFT 中位數 {s['ttft_median_ms']} ms（{s['ttft_range_ms']}）"
                  f"｜丟批 {s['drop_round']} TTFT {s['drop_ttft_ms']} ms、重算 {s['drop_prompt_n']} tok"
                  f"｜開場出現在 {s['opening_rounds'] or '無'}｜簡體字 {s['simplified_chars']}"
                  f"｜主題注入 {s['topic_rounds']} 輪共 {s['topics_injected']} 件｜最大排隊 {s['max_queue_ms']} ms"
                  + (f"｜中途插話（{s['mid_cancel_round']}）排隊 {s['mid_cancel_queue_ms']} ms、"
                     f"TTFT {s['mid_cancel_ttft_ms']} ms" if s["mid_cancel_queue_ms"] is not None else "")
                  + (f"｜結尾插話排隊 {s['cancel_queue_ms']} ms" if s["cancel_queue_ms"] is not None else ""))
        for line in verdict(sums):
            print(line)
        return 0
    finally:
        os.makedirs(root, exist_ok=True)
        # 中途失敗也要停用（放 finally）：已跑完的那幾場副本一樣會被 --from-prev 撈到
        retired = retire_probe_topics(root)
        print(f"\n測試主題已停用（retired=1，只動本次副本）：{retired}")
        with open(os.path.join(root, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"sessions": sums, "verdict": verdict(sums) if sums else [],
                       "thresholds": THRESHOLDS, "retired_topics": retired}, f, ensure_ascii=False, indent=2)
        print(f"\n總表：{os.path.join(root, 'summary.json')}")
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
