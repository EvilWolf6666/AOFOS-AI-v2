# core/memory/selftest_v4.py
"""
準 v4 自測：`python -m core.memory.selftest_v4`

全部在暫存資料夾、對假 LLM 伺服器（本檔 FakeLLM）與假 embedding 跑，不碰 data/、不碰真的推理後端。
涵蓋：切段／解析（純函式）、主題表（屬性鍵機械取代、版本、進度原子性）、speaker_id 落地、
端到端（閒置 → 判斷 → 建主題 → 向量 → 注入）、取消（prefill 中／串流中）、施測隔離、
暫停逾時保險、壞輸出。
🆕 2026-10-09：判斷一次呼叫＋logprobs 算 P(建)（含「答不建但機率過門檻 ⇒ 補一次」與「拿不到機率」）、
規則檔解析（用真的 prompts/memory_judge.txt）、關鍵字那一路與配額聯集、鎖定／待審／人工審核、舊表遷移。
⚠️ 不測「真的推理後端斷線後會不會停止運算」——那要在 Windows 實測（看取消後下一輪對話 TTFT）。
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import math

from config import MEMORY_CONFIG
from core.llm_client import BackgroundLLM
from core.memory.db import connect
from core.memory.judge import (_FORCE_BUILD, drop_copies, fallback_topic, load_prompt, norm_topic, p_build,
                               parse_topics, split_answer, split_segment)
from core.memory.manager import MemoryManager
from core.memory.topic_index import TopicSearch, fill_fts
from core.memory.topics import SCHEMA as TOPIC_SCHEMA
from core.memory.topics import TopicStore


# ── 假 LLM 伺服器（只給本自測用，不接任何真服務；2026-10-04 由 fake_llm_server.py 併入）──
# 模擬 LM Studio 的兩段延遲：送出回應標頭前的 prefill，以及逐 token 串流；
# 記錄每個請求是「送完」還是「被客戶端中途斷線」，用來驗證 BackgroundLLM 的取消。

class FakeLLM:
    def __init__(self) -> None:
        self.prefill_s = 0.0          # 回應標頭前的延遲（模擬 prefill）
        self.token_s = 0.0            # 每個 chunk 之間的延遲
        self.responder = lambda req: "OK"   # req(dict) → 回應文字，或 (文字, logprobs 規格)
        self.log: list[dict] = []     # 每個請求：{purpose, status: done/aborted, body}
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.base_url = f"http://127.0.0.1:{self._srv.server_address[1]}/v1"
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()

    def _handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # 安靜
                pass

            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                entry = {"body": req, "status": "started"}
                fake.log.append(entry)
                time.sleep(fake.prefill_s)
                got = fake.responder(req)
                text, spec = got if isinstance(got, tuple) else (got, None)
                # logprobs 規格：[(實際選的 token, [(候選, 機率), ...]), ...]——模擬 LM Studio 串流逐塊附上
                pieces: list[tuple[str, dict | None]] = []
                if spec and req.get("logprobs"):
                    head = "".join(tok for tok, _ in spec)
                    assert text.startswith(head), (text, head)
                    for tok, cands in spec:
                        pieces.append((tok, {"content": [{"token": tok, "logprob": math.log(dict(cands)[tok]),
                                                          "top_logprobs": [{"token": c, "logprob": math.log(p)}
                                                                           for c, p in cands]}]}))
                    text = text[len(head):]
                pieces += [(ch, None) for ch in _chunks(text) if ch]
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for i, (ch, lp) in enumerate(pieces):
                        if i:
                            time.sleep(fake.token_s)
                        choice = {"index": 0, "delta": {"content": ch}, "finish_reason": None}
                        if lp is not None:
                            choice["logprobs"] = lp
                        data = {"id": "x", "object": "chat.completion.chunk", "created": 0,
                                "model": req.get("model", "m"), "choices": [choice]}
                        self.wfile.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    entry["status"] = "done"
                except (BrokenPipeError, ConnectionResetError, OSError):
                    entry["status"] = "aborted"

        return H


def _chunks(text: str, size: int = 2) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


_fail = 0
TZ = timezone(timedelta(hours=8))


def check(name: str, ok: bool, detail: str = "") -> None:
    global _fail
    print(("  ✅ " if ok else "  ❌ ") + name + (f"　— {detail}" if detail and not ok else ""))
    if not ok:
        _fail += 1


class FakeModel:
    """假 embedding：同義詞映到同一維，其餘逐字雜湊。"""
    SYN = {"高雄": 0, "港都": 0, "貓": 1, "喵星人": 1, "住": 2}

    def encode(self, texts):
        import numpy as np
        out = []
        for t in ([texts] if isinstance(texts, str) else texts):
            v = np.zeros(64, dtype=np.float32)
            for w, dim in self.SYN.items():
                if w in t:
                    v[dim] += 3.0
            for ch in t:
                v[3 + (ord(ch) % 61)] += 0.05
            out.append(v)
        return np.stack(out)


BUILD = [("建", [("建", 0.9), ("不", 0.1)])]


def responder(req: dict):
    """假模型（一次呼叫）：第一行建／不建＋主題句，附第一行的 logprobs。依片段內容決定。"""
    seg = req["messages"][-1]["content"]
    if _FORCE_BUILD in seg:                          # 補呼叫：要它直接寫主題句
        return "建\n事件｜無｜邪狼說了邊界的事"
    if "天氣" in seg:                                # P(建)=0.05＋0.95×0.02≈0.07 ⇒ 不建
        return "不建", [("不", [("不", 0.95), ("建", 0.05)]), ("建", [("建", 0.98), ("確", 0.02)])]
    if "邊界" in seg:                                # 模型答不建、但 P(建)=0.4 ≥ 門檻 0.3 ⇒ 補一次
        return "不建", [("不", [("不", 0.6), ("建", 0.4)]), ("建", [("建", 0.9), ("確", 0.1)])]
    if "亂碼判斷" in seg:                            # 不附 logprobs、第一行也看不懂 ⇒ 當建
        return "我覺得要建\n其他｜無｜邪狼說了亂碼判斷"
    if "解析不了" in seg:
        return "建\n這段很有意思喔", BUILD             # 沒有 ｜ ⇒ 解析不出
    if "全抄" in seg:                                # 只寫了照抄的例子 ⇒ 濾光 ⇒ 退回原句
        return "建\n身分｜無｜邪狼是奧狐的哥哥，也是奧狐的創造者", BUILD
    if "抄例子" in seg:                              # 照抄規則檔例子＋與現存重複＋一件新的
        return ("建\n身分｜無｜邪狼是奧狐的哥哥，也是奧狐的創造者。\n狀態｜無｜邪狼住在高雄\n"
                "事件｜無｜邪狼說了抄例子的事"), BUILD
    lines = []
    if "高雄" in seg:
        lines.append("狀態｜邪狼.住地｜邪狼住在高雄")
    if "台北" in seg:
        lines.append("狀態｜邪狼.住地｜邪狼搬到台北住了")
    if "貓" in seg:
        lines.append("事件|無|邪狼家的貓很胖")      # 半形 | 也要吃
    if "阿凱" in seg:
        lines.append("狀態｜無｜阿凱下次會來邪狼的直播")
    return "建\n" + ("\n".join(lines) or "其他｜無｜沒什麼"), BUILD


def wait(pred, timeout=8.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def open_handles(d: str) -> list[str]:
    """本行程還開著、指向 d 底下的檔案（Linux 用 /proc 看；其他平台回空＝不檢查）。

    🔴 存在理由：Windows 不能刪開著的檔，Linux 可以 ⇒ 連線漏關在 Linux 自測永遠看不出來
       （2026-09-30 Windows 實跑兩支自測都在清暫存資料夾時炸）。這裡在 Linux 上把同一件事變成可見。
    """
    import gc
    gc.collect()   # sqlite3 連線可能卡在循環參考裡，先收掉，只留真的還被持有的
    fd_dir = "/proc/self/fd"
    if not os.path.isdir(fd_dir):
        return []
    real = os.path.realpath(d)
    out = []
    for fd in os.listdir(fd_dir):
        try:
            t = os.readlink(os.path.join(fd_dir, fd))
        except OSError:
            continue
        if t.startswith(real):
            out.append(os.path.basename(t))
    return sorted(out)


def has_table(path: str, name: str) -> bool:
    # 🔴 一定要關：Windows 上沒關的連線會讓測試區刪不掉、暫存資料夾清不掉
    #    （2026-09-30 Windows 實跑抓到；Linux 刪得掉開著的檔，所以本機自測看不出來）。
    cx = sqlite3.connect(path)
    try:
        return cx.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone() is not None
    finally:
        cx.close()


def injected(ctx: list[dict]) -> str:
    # 只認以 block_open 開頭的區塊：固定說明（note_position="fixed"）也提到標記，但不是注入內容
    return "\n".join(m["content"] for m in ctx
                     if m["role"] == "system" and m["content"].startswith(MEMORY_CONFIG["block_open"]))


def mm(d: str, fake: FakeLLM, **over) -> MemoryManager:
    cfg = {"v4_enabled": True,
           "judge_idle_s": 0.2, "judge_segment_close_s": 0, "judge_pause_timeout_s": 60}
    cfg.update(over)
    return MemoryManager(None, db_path=os.path.join(d, "memory.db"),
                         test_db_path=os.path.join(d, "memory_test.db"),
                         embed_enabled=True, embed_factory=FakeModel, cfg_override=cfg,
                         bg_llm=BackgroundLLM(base_url=fake.base_url, model="fake",
                                              params={"temperature": 0.0}))


def turn(m: MemoryManager, u: str, a: str, **kw) -> None:
    """模擬 Pipeline 一輪：開始 pause → 寫入 → 收尾 resume。"""
    m.bg_pause()
    m.add_turn(u, a, **kw)
    m.bg_resume()


def judged(m: MemoryManager) -> tuple[int, int]:
    cx = connect(m.memory_path)
    try:
        g = lambda k: (cx.execute("SELECT value FROM meta WHERE key=?", (k,)).fetchone() or [0])[0]
        return g("judged_upto"), g("landed_upto")
    finally:
        cx.close()


def main() -> int:
    ts = lambda mnt: datetime(2026, 9, 30, 20, mnt, tzinfo=TZ).isoformat()

    print("\n[切段／解析（純函式）]")
    rows = [{"serial": i + 1, "ts": ts(i), "role": "user"} for i in range(5)]
    now = datetime(2026, 9, 30, 20, 4, 10, tzinfo=TZ)
    check("最後一句還沒過 close_s ⇒ 不判", split_segment(rows, max_rows=12, gap_minutes=30,
                                                        close_s=60, now=now) is None)
    check("過了 close_s ⇒ 整段", len(split_segment(rows, max_rows=12, gap_minutes=30, close_s=5,
                                                   now=now) or []) == 5)
    check("滿 max_rows 就切", len(split_segment(rows, max_rows=3, gap_minutes=30, close_s=999,
                                              now=now) or []) == 3)
    gap = rows[:2] + [{"serial": 3, "ts": ts(50), "role": "user"}]
    check("中間隔了空白 ⇒ 只取空白前那段", [r["serial"] for r in split_segment(
        gap, max_rows=12, gap_minutes=30, close_s=999, now=now) or []] == [1, 2])
    items, bad = parse_topics("1. 狀態｜邪狼.住地｜邪狼住高雄\n亂寫一行\n事件|無|貓很胖\n\n",
                              types=["狀態", "事件"], max_n=5)
    check("解析：編號去掉、半形分隔可、「無」→ None、壞行計數",
          items == [{"type": "狀態", "attr_key": "邪狼.住地", "text": "邪狼住高雄"},
                    {"type": "事件", "attr_key": None, "text": "貓很胖"}] and bad == 1, f"{items} {bad}")

    print("\n[判斷：讀機率（純函式）]")
    L = lambda tok, cands: {"token": tok, "logprob": math.log(dict(cands)[tok]),
                            "top_logprobs": [{"token": c, "logprob": math.log(p)} for c, p in cands]}
    # 2026-10-09 LM Studio 實測的形狀：不 0.873／建 0.117，接著 建 0.705／確 0.287
    real = [L("不", [("不", 0.873), ("建", 0.117)]), L("建", [("建", 0.705), ("確", 0.287)])]
    check("P(建)＝建＋不×確（實測形狀 ≈0.37）", abs(p_build(real) - (0.117 + 0.873 * 0.287)) < 1e-6,
          str(p_build(real)))
    check("前面的空白 token 跳過", abs(p_build([{"token": "\n", "logprob": 0.0, "top_logprobs": []}] + real)
                                     - p_build(real)) < 1e-9)
    check("「不確定」整個是一個 token 也算建",
          abs(p_build([L("不確定", [("不確定", 0.6), ("不建", 0.4)])]) - 0.6) < 1e-6)
    check("選了「建」時，「不」底下的不確定路徑不算（看不到）",
          abs(p_build([L("建", [("建", 0.7), ("不", 0.3)]), L("\n", [("\n", 1.0)])]) - 0.7) < 1e-6)
    check("沒有 logprobs ⇒ None", p_build([]) is None and p_build([{"token": "建", "logprob": 0.0}]) is None)
    check("第一行判斷與主題句分開", split_answer("建\n狀態｜無｜x") == ("建", "狀態｜無｜x")
          and split_answer("不建") == ("不建", "") and split_answer("嗯\n狀態｜無｜x")[0] == "?")

    items = [{"text": "邪狼是奧狐的哥哥，也是奧狐的創造者。"}, {"text": "邪狼 住在高雄"},
             {"text": "新的一件"}, {"text": "新的一件！"}]
    keep, cp, dp = drop_copies(items, examples={norm_topic("邪狼是奧狐的哥哥，也是奧狐的創造者")},
                               existing={norm_topic("邪狼住在高雄")})
    check("機械過濾：照抄例子、與現存重複、同次重複都丟（忽略空白與句尾標點）",
          [k["text"] for k in keep] == ["新的一件"] and (cp, dp) == (1, 2), f"{keep} {cp} {dp}")

    fb = fallback_topic("（對話時間：2026-10-10 12:00）\n邪狼：你的生日就定在 3 月 11 號吧。\n\n奧狐：好！")
    check("退回原句：去掉時間標頭、各句以／接起、說話者保留、無屬性鍵",
          fb == {"type": "原句", "attr_key": None, "text": "邪狼：你的生日就定在 3 月 11 號吧。／奧狐：好！"}, str(fb))

    print("\n[判斷：規則檔（真的 prompts/memory_judge.txt）]")
    system, types, examples = load_prompt(MEMORY_CONFIG["judge_prompt_path"], max_topics=5)
    check("類型＝【類型】段的八類", types == ["身分", "狀態", "偏好", "看法", "約定", "事件", "話題", "其他"],
          str(types))
    check("例子解析成（對話, 期望輸出）、第一行都是建／不建、「最多幾件」已代入",
          len(examples) >= 5 and all(out and out[0] in ("建", "不建") for _, out in examples)
          and examples[0][1][1].count("｜") == 2 and "{最多幾件}" not in system
          and "最多 5 行" in system, f"{examples[:2]}")
    try:
        load_prompt("/不存在的路徑/x.txt", max_topics=5)
        check("規則檔不在 ⇒ 大聲失敗", False)
    except FileNotFoundError:
        check("規則檔不在 ⇒ 大聲失敗", True)

    print("\n[滑動視窗：一次丟一批]")
    with tempfile.TemporaryDirectory() as d:
        m = MemoryManager(None, db_path=os.path.join(d, "m.db"), test_db_path=os.path.join(d, "t.db"),
                          embed_enabled=False,
                          cfg_override={"v4_enabled": False, "wm_max_rounds": 20, "wm_drop_rounds": 10})
        sizes = []
        for i in range(31):
            m.add_turn(f"u{i}", f"a{i}")
            sizes.append(len(m.working_memory) // 2)
        m.compress_on_exit()
        check("滿 20 輪後一次丟 10 輪、之後再長回 20（開頭只在丟的那輪改變）",
              sizes[:20] == list(range(1, 21)) and sizes[20:31] == list(range(11, 21)) + [11], str(sizes))

    print("\n[主題表]")
    with tempfile.TemporaryDirectory() as d:
        cx = connect(os.path.join(d, "t.db"))
        st = TopicStore(cx)
        a = st.add_topics([{"text": "邪狼住高雄", "type": "狀態", "attr_key": "邪狼.住地"}],
                          first_serial=1, last_serial=4, event_ts=ts(1), speakers=["evilwolf", "aofos"],
                          judged_upto=4)
        v2 = st.add_version(a[0], "邪狼住在高雄（重寫版）")
        check("版本：latest 取最新一版、root 相同", [t["id"] for t in st.latest()] == [v2]
              and st.get([v2])[0]["root"] == a[0])
        st.add_topics([], first_serial=5, last_serial=6, event_ts=ts(2), speakers=[], judged_upto=6)
        check("判了不建：只推進度", st.judged_upto == 6 and st.count() == 2)
        b = st.add_topics([{"text": "邪狼搬到台北", "type": "狀態", "attr_key": "邪狼.住地"},
                           {"text": "奧狐住在電腦裡", "type": "狀態", "attr_key": "奧狐.住地"}],
                          first_serial=7, last_serial=8, event_ts=ts(3), speakers=["evilwolf"],
                          judged_upto=8)
        lat = {t["text"] for t in st.latest()}
        check("同屬性鍵新事實 ⇒ 舊的（含所有版本）被機械取代，不同人不互相取代",
              lat == {"邪狼搬到台北", "奧狐住在電腦裡"}, str(lat))
        old = st.get([a[0], v2])
        check("被取代的標 valid_to／superseded_by", all(o["valid_to"] and o["superseded_by"] == b[0]
                                                      for o in old))
        check("被取代的仍可用 include_superseded 看到", len(st.latest(include_superseded=True)) == 3)

        print("\n[鎖定／待審／人工審核]")
        g = st.add_topics([{"text": "奧狐的生日是 1 月 1 號", "type": "身分", "attr_key": "奧狐.生日"}],
                          first_serial=9, last_serial=10, event_ts=ts(4), speakers=["evilwolf"], judged_upto=10)
        st.lock(g[0])
        h2 = st.add_topics([{"text": "奧狐的生日是 3 月 11 號", "type": "身分", "attr_key": "奧狐.生日"}],
                           first_serial=11, last_serial=12, event_ts=ts(5), speakers=["evilwolf"],
                           judged_upto=12)
        lat = {t["text"] for t in st.latest()}
        pend = st.pending_review()
        check("碰到鎖定的 ⇒ 新的寫成待審、鎖定的不動、待審不進檢索",
              "奧狐的生日是 1 月 1 號" in lat and "奧狐的生日是 3 月 11 號" not in lat
              and [p_["id"] for p_ in pend] == h2 and pend[0]["replaces"] == g[0], f"{lat} {pend}")
        st.approve(h2[0])
        lat = {t["text"]: t for t in st.latest()}
        check("通過 ⇒ 舊的被取代、新的接手鎖定",
              "奧狐的生日是 1 月 1 號" not in lat and lat.get("奧狐的生日是 3 月 11 號", {}).get("status") == "locked",
              str(list(lat)))
        h3 = st.add_topics([{"text": "奧狐的生日是 7 月 7 號", "type": "身分", "attr_key": "奧狐.生日"}],
                           first_serial=13, last_serial=14, event_ts=ts(6), speakers=["evilwolf"],
                           judged_upto=14)
        st.reject(h3[0])
        check("駁回 ⇒ 列還在（rejected）、不進檢索、鎖定的照舊",
              st.get(h3)[0]["status"] == "rejected" and "奧狐的生日是 3 月 11 號" in {t["text"] for t in st.latest()}
              and not st.pending_review())

        print("\n[關鍵字那一路（主題）]")
        fill_fts(cx)
        st.add_topics([{"text": "邪狼的朋友阿凱下次會來直播", "type": "狀態", "attr_key": None}],
                      first_serial=15, last_serial=16, event_ts=ts(7), speakers=["evilwolf"], judged_upto=16)
        n = fill_fts(cx)
        ts_ = TopicSearch(cx, None)
        hit = ts_.search_bm25("阿凱什麼時候來", top_k=3, max_terms=64)
        check("增量補索引、專有名詞逐字命中（不需向量）", n == 1 and [t["text"] for t, _ in hit] ==
              ["邪狼的朋友阿凱下次會來直播"], f"n={n} {hit}")
        got = {t["text"] for q in ("生日 1 月 1 號", "生日 7 月 7 號")
               for t, _ in ts_.search_bm25(q, top_k=10, max_terms=64)}
        check("被取代／駁回的不會被關鍵字撈到（只剩現行的那件）",
              got == {"奧狐的生日是 3 月 11 號"}, str(got))
        cx.close()

    print("\n[舊表遷移：沒有 status／replaces 欄]")
    with tempfile.TemporaryDirectory() as d:
        cx = connect(os.path.join(d, "old.db"))
        old_schema = TOPIC_SCHEMA.replace(",\n    status        TEXT    NOT NULL DEFAULT 'normal',  -- normal／locked／pending／rejected\n    replaces      INTEGER                -- pending 才有：它要取代的鎖定 root\n", "\n")
        assert "status" not in old_schema.split("CREATE INDEX")[0]
        cx.executescript(old_schema)
        cx.execute("INSERT INTO topic(root, text, first_serial, last_serial, created_at) VALUES(1,'舊主題',1,2,'x')")
        cx.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value INTEGER)")
        st = TopicStore(cx)
        cols = {r[1] for r in cx.execute("PRAGMA table_info(topic)")}
        check("補上兩欄、既有列為 normal、照常被檢索", {"status", "replaces"} <= cols
              and [t["text"] for t in st.latest()] == ["舊主題"] and st.latest()[0]["status"] == "normal")
        cx.close()

    print("\n[人工審核工具（tools/memory_review.py，腳本化輸入）]")
    with tempfile.TemporaryDirectory() as d:
        from tools import memory_review
        path = os.path.join(d, "memory.db")
        cx = connect(path)
        st = TopicStore(cx)
        a1 = st.add_topics([{"text": "設定甲", "type": "身分", "attr_key": "k1"},
                            {"text": "設定乙", "type": "身分", "attr_key": "k2"},
                            {"text": "設定丙", "type": "身分", "attr_key": "k3"}],
                           first_serial=1, last_serial=2, event_ts=ts(1), speakers=[], judged_upto=0)
        for i in a1:
            st.lock(i)
        b1 = st.add_topics([{"text": "新甲", "type": "身分", "attr_key": "k1"},
                            {"text": "新乙", "type": "身分", "attr_key": "k2"},
                            {"text": "新丙", "type": "身分", "attr_key": "k3"}],
                           first_serial=3, last_serial=4, event_ts=ts(2), speakers=[], judged_upto=0)
        cx.close()
        saved = MEMORY_CONFIG["db_path"]
        MEMORY_CONFIG["db_path"] = path
        answers = iter(["y", "?", "NO", ""])
        lines: list[str] = []
        try:
            res = memory_review.review_memory(ask=lambda _p: next(answers), out=lines.append)
        finally:
            MEMORY_CONFIG["db_path"] = saved
        cx = connect(path)
        st = TopicStore(cx, writable=False)
        stat = {t["text"]: t["status"] for t in st.get(b1)}
        lat = {t["text"] for t in st.latest()}
        cx.close()
        check("y 通過／錯字再問一次後 NO 駁回（大小寫、縮寫都收）／Enter 跳過（維持待審）",
              res == (1, 1, 1) and stat == {"新甲": "locked", "新乙": "rejected", "新丙": "pending"}
              and lat == {"新甲", "設定乙", "設定丙"}, f"{res} {stat} {lat}")
        h = open_handles(d)
        check("審核工具結束後沒有連線還開著", not h, str(h))

    fake = FakeLLM()
    fake.responder = responder
    try:
        with tempfile.TemporaryDirectory() as d:
            print("\n[端到端：閒置 → 判斷 → 主題 → 注入]")
            m = mm(d, fake)
            check("啟動時不碰 DB 建主題表（連線延後到真的要判）",
                  not has_table(os.path.join(d, "memory.db"), "topic"))
            turn(m, "我住在高雄", "高雄很熱吧")
            turn(m, "我家的貓很胖", "好可愛")
            ok = wait(lambda: judged(m)[0] == judged(m)[1] == 4)
            check("閒置後背景判完（judged_upto 追上 landed_upto）", ok, str(judged(m)))
            check("/status 待判句數即時（判完＝0）", "待判 0 句" in m.status_text(), m.status_text())
            ok = wait(lambda: "主題未補向量 0" in m.status_text() and "主題 2" in m.status_text())
            check("建 2 件主題並補完向量", ok, m.status_text())
            cx = connect(m.memory_path)
            spk = [r[0] for r in cx.execute("SELECT speaker_id FROM transcript ORDER BY serial")]
            cx.close()
            check("speaker_id 落地（使用者／奧狐）", spk == ["evilwolf", "aofos"] * 2, str(spk))
            inj = injected(m.get_context("港都熱不熱"))
            check("同義查詢撈回主題句（帶口語時間）", "邪狼住在高雄" in inj and "（剛剛）" in inj, inj)
            inj = injected(m.get_context("高雄熱不熱"))
            check("兩路都命中同一件 ⇒ 只出現一次（聯集去重）", inj.count("邪狼住在高雄") == 1, inj)
            inj = injected(m.get_context("喵星人好嗎"))
            check("第二件也撈得回", "貓很胖" in inj, inj)

            turn(m, "我搬到台北了，現在住那邊", "恭喜搬家")
            # 進度推進（add_topics）與補向量是先後兩步 ⇒ 要等向量補完再查，否則偶發撈不到新的那件
            ok = wait(lambda: judged(m)[0] == 6 and "主題未補向量 0" in m.status_text())
            inj = injected(m.get_context("你現在住哪"))
            check("新住地取代舊住地：只注入台北", ok and "台北" in inj and "高雄" not in inj, inj)

            turn(m, "今天天氣不錯", "對啊")
            ok = wait(lambda: judged(m)[0] == 8)
            check("判「不建」：進度推進、不建主題", ok and "不建 1" in m.status_text(), m.status_text())

            turn(m, "", "我自己說話", user_speaker=None, situation="畫面切到遊戲")
            wait(lambda: judged(m)[0] == 10)
            cx = connect(m.memory_path)
            r = cx.execute("SELECT speaker_id FROM transcript WHERE serial=9").fetchone()[0]
            cx.close()
            check("observation 輪次 user 那列 speaker_id＝NULL", r is None, str(r))

            turn(m, "這句解析不了", "嗯")
            ok = wait(lambda: judged(m)[0] == 12)
            check("主題句解析不出 ⇒ 退回存原句、不卡住、計數",
                  ok and "解析丟行 1" in m.status_text() and "退回原句 1" in m.status_text()
                  and "主題 5" in m.status_text(), m.status_text())
            turn(m, "亂碼判斷", "嗯")
            ok = wait(lambda: judged(m)[0] == 14)
            check("拿不到機率、第一行看不懂 ⇒ 當建、計數（observation 那段也建了 1 件）",
                  ok and "主題 6" in m.status_text() and "缺機率 1" in m.status_text(), m.status_text())
            n0 = len(fake.log)
            turn(m, "邊界的事", "嗯")
            ok = wait(lambda: judged(m)[0] == 16)
            forced = [e for e in fake.log[n0:] if _FORCE_BUILD in e["body"]["messages"][-1]["content"]]
            check("模型答不建、但 P(建) 過門檻 ⇒ 補一次呼叫並建",
                  ok and len(forced) == 1 and "補呼叫 1" in m.status_text() and "主題 7" in m.status_text(),
                  m.status_text())
            req0 = fake.log[n0]["body"]
            check("判斷請求：要 logprobs、不用 json_schema、規則在 system（片段在後）",
                  req0.get("logprobs") is True and "response_format" not in req0
                  and "【類型】" in req0["messages"][0]["content"]
                  and req0["messages"][-1]["content"].startswith("【對話片段】"), str(list(req0)))

            print("\n[取消]")
            fake.prefill_s = 3.0
            n0 = len(fake.log)
            turn(m, "我家的貓又胖了", "哈哈")
            started = wait(lambda: len(fake.log) > n0)
            time.sleep(0.2)
            t0 = time.monotonic()
            m.bg_pause()
            dt = time.monotonic() - t0
            aborted = wait(lambda: fake.log[-1]["status"] == "aborted", 5.0)
            check("prefill 中取消：pause 立刻返回、伺服器看到斷線", started and dt < 0.2 and aborted,
                  f"dt={dt:.3f} status={fake.log[-1]['status']}")
            check("被取消的段不推進度", judged(m)[0] == 16, str(judged(m)))
            fake.prefill_s = 0.0
            fake.token_s = 0.3
            m.bg_resume()
            n1 = len(fake.log)
            wait(lambda: len(fake.log) > n1)
            time.sleep(0.4)
            m.bg_pause()
            ok = wait(lambda: fake.log[-1]["status"] == "aborted", 5.0)
            check("串流中取消：伺服器看到斷線、計數", ok and "切斷進行中 2" in m.status_text(),
                  f"{fake.log[-1]['status']} {m.status_text()}")
            fake.token_s = 0.0
            m.bg_resume()
            ok = wait(lambda: judged(m)[0] == 18)
            # 這段的主題句（邪狼家的貓很胖）與前面建過的一字不差 ⇒ 被重複過濾丟掉，主題數不變
            check("恢復後整段重判完成（冪等：主題不重複；與現存一字不差的被過濾）",
                  ok and "主題 7" in m.status_text() and "重複 1" in m.status_text(), m.status_text())

            print("\n[施測隔離]")
            before = judged(m)[0]
            m.set_test_mode(True)
            turn(m, "施測題：我住在高雄", "好")
            time.sleep(1.0)
            check("施測中不判（測試區沒有主題表）", not has_table(m.memory_path, "topic"))
            check("/status 即時顯示施測區待判 2 句（背景停用、不判）", "待判 2 句" in m.status_text(),
                  m.status_text())
            inj = injected(m.get_context("你住哪"))
            check("施測中仍讀得到日常區主題", "台北" in inj, inj)
            m.set_test_mode(False)
            check("離場測試區整包丟棄", not os.path.exists(os.path.join(d, "memory_test.db")))
            check("日常區進度沒被施測動到", judged(m)[0] == before)

            m.compress_on_exit()
            h = open_handles(d)
            check("離場後沒有任何連線還開著（Windows 才刪得掉）", not h, str(h))

        print("\n[讀取：向量撈不到時關鍵字那一路補上（配額聯集）]")
        with tempfile.TemporaryDirectory() as d:
            m = mm(d, fake, topic_min_sim=0.999, topic_vec_k=3, topic_bm25_k=1)
            turn(m, "我朋友阿凱下次會來直播", "好期待")
            turn(m, "我住在高雄", "嗯")
            ok = wait(lambda: judged(m)[0] == 4 and "主題未補向量 0" in m.status_text())
            inj = injected(m.get_context("阿凱什麼時候來"))
            check("向量下限調到撈不到 ⇒ 仍由關鍵字撈回專有名詞那件、且只有配額內 1 件",
                  ok and "阿凱下次會來" in inj and "高雄" not in inj, inj)
            turn(m, "這段會抄例子", "嗯")
            ok = wait(lambda: judged(m)[0] == 6)
            st_ = m.status_text()
            check("實際判斷時：照抄規則檔例子的、與現存重複的被丟，新的那件照建",
                  ok and "濾掉 抄例子 1／重複 1" in st_ and "主題 3" in st_ and "退回原句 0" in st_, st_)
            turn(m, "我的生日是全抄那天", "記住了")
            ok = wait(lambda: judged(m)[0] == 8)
            st_ = m.status_text()
            cx = connect(m.memory_path)
            fb = cx.execute("SELECT text, type, attr_key FROM topic ORDER BY id DESC LIMIT 1").fetchone()
            cx.close()
            check("照抄的被濾光 ⇒ 退回存原句（帶說話者、類型「原句」、無屬性鍵），不是建 0 件",
                  ok and "退回原句 1" in st_ and "主題 4" in st_ and fb is not None
                  and fb["type"] == "原句" and fb["attr_key"] is None
                  and "全抄那天" in fb["text"] and "／" in fb["text"] and "對話時間" not in fb["text"],
                  f"{st_} {dict(fb) if fb else None}")
            m.compress_on_exit()
            h = open_handles(d)
            check("離場後沒有任何連線還開著", not h, str(h))

        print("\n[獨立終端的背景判斷（tools/judge_worker.py）與執行鎖]")
        from core.memory.run_lock import RunLock, held_by_other, lock_path
        from tools import judge_worker
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "memory.db")
            lk = lock_path(db)
            a_ = RunLock(lk)
            check("執行鎖：拿得到、別人看得到、放掉後別人看不到",
                  a_.try_acquire() and held_by_other(lk) and not RunLock(lk).try_acquire()
                  and (a_.release() or True) and not held_by_other(lk))
            m = MemoryManager(None, db_path=db, test_db_path=os.path.join(d, "t.db"), embed_enabled=False,
                              cfg_override={"v4_enabled": False})
            for u in ("我住在高雄", "我家的貓很胖", "今天天氣不錯"):
                m.add_turn(u, "嗯")
            m.compress_on_exit()
            mk = lambda: BackgroundLLM(base_url=fake.base_url, model="fake", params={"temperature": 0.0})
            over = {"judge_segment_close_s": 0, "judge_segment_max": 2}
            lines: list[str] = []
            rc = judge_worker.main(db_path=db, llm=mk(), cfg_override=over, poll_s=0.05, out=lines.append)
            cx = connect(db)
            jd = TopicStore(cx, writable=False)
            n_top, pend = jd.count(), jd.pending_count()
            fts = cx.execute("SELECT COUNT(*) FROM topic_fts").fetchone()[0]
            cx.close()
            check("AOFOS 沒開 ⇒ 全部判完、自己結束；關鍵字索引補了、不載 embedding",
                  rc == 0 and pend == 0 and n_top == 2 and fts == 2 and "全部判完" in lines[-1],
                  f"rc={rc} pend={pend} n={n_top} fts={fts} {lines[-1:]}")
            hold = RunLock(lk)
            hold.try_acquire()
            lines = []
            check("AOFOS 開著 ⇒ 不啟動", judge_worker.main(db_path=db, llm=mk(), out=lines.append) == 0
                  and "不啟動" in lines[0], str(lines))
            hold.release()
            m = MemoryManager(None, db_path=db, test_db_path=os.path.join(d, "t.db"), embed_enabled=False,
                              cfg_override={"v4_enabled": False})
            m.add_turn("我搬到台北了", "恭喜")
            m.compress_on_exit()
            fake.prefill_s = 3.0
            n0 = len(fake.log)
            lines = []
            res: list[int] = []
            th = threading.Thread(target=lambda: res.append(judge_worker.main(
                db_path=db, llm=mk(), cfg_override=over, poll_s=0.05, out=lines.append)))
            th.start()
            wait(lambda: len(fake.log) > n0)
            time.sleep(0.2)
            t0 = time.monotonic()
            hold.try_acquire()                      # 模擬 AOFOS 啟動
            th.join(5.0)
            dt = time.monotonic() - t0
            aborted = wait(lambda: fake.log[-1]["status"] == "aborted", 5.0)
            hold.release()
            fake.prefill_s = 0.0
            # ⚠️ 門檻是「假伺服器 prefill 3 秒＋1」而不是 1 秒：Windows 上 shutdown() 不會打斷別條執行緒
            #    卡著的 recv（Linux 會），要等伺服器下一次寫入才返回（2026-10-09 Windows 實測 2.77 秒）。
            #    斷線本身（FIN）在取消當下就送出 ⇒ 伺服器端照樣看得到；只是 worker 本身晚一點結束。
            check("跑到一半 AOFOS 啟動 ⇒ 取消手上的呼叫並結束（伺服器看到斷線；Windows 最多等到伺服器下一次回應）",
                  res == [0] and dt < 4.0 and aborted and "偵測到 AOFOS 啟動" in lines[-1],
                  f"res={res} dt={dt:.2f} aborted={aborted} {lines[-1:]}")
            h = open_handles(d)
            check("worker 結束後沒有連線還開著（鎖檔除外）", [x for x in h if not x.startswith("aofos.run")] == [], str(h))

        print("\n[暫停逾時保險]")
        with tempfile.TemporaryDirectory() as d:
            m = mm(d, fake, judge_pause_timeout_s=0.5)
            m.bg_pause()
            m.add_turn("我住在高雄", "嗯")      # 模擬收尾掛點漏呼叫 bg_resume
            ok = wait(lambda: judged(m)[0] == 2, 6.0)
            check("暫停沒被恢復 ⇒ 逾時自動恢復並完成", ok, str(judged(m)))
            m.compress_on_exit()
            h = open_handles(d)
            check("離場後沒有任何連線還開著", not h, str(h))
    finally:
        fake.close()

    print("\n" + ("✅ 準 v4 自測全部通過" if _fail == 0 else f"❌ {_fail} 項失敗"))
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())


