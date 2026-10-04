# core/memory/selftest_v4.py
"""
準 v4 自測：`python -m core.memory.selftest_v4`

全部在暫存資料夾、對假 LLM 伺服器（本檔 FakeLLM）與假 embedding 跑，不碰 data/、不碰 LM Studio。
涵蓋：切段／解析（純函式）、主題表（屬性鍵機械取代、版本、進度原子性）、speaker_id 落地、
端到端（閒置 → 判斷 → 建主題 → 向量 → 注入）、取消（prefill 中／串流中）、施測隔離、
暫停逾時保險、壞輸出。
⚠️ 不測「真的 LM Studio 斷線後會不會停止運算」——那要在 Windows 實測（看取消後下一輪對話 TTFT）。
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

from core.llm_client import BackgroundLLM
from core.memory.db import connect
from core.memory.judge import parse_topics, split_segment
from core.memory.manager import MemoryManager
from core.memory.topics import TopicStore


# ── 假 LLM 伺服器（只給本自測用，不接任何真服務；2026-10-04 由 fake_llm_server.py 併入）──
# 模擬 LM Studio 的兩段延遲：送出回應標頭前的 prefill，以及逐 token 串流；
# 記錄每個請求是「送完」還是「被客戶端中途斷線」，用來驗證 BackgroundLLM 的取消。

class FakeLLM:
    def __init__(self) -> None:
        self.prefill_s = 0.0          # 回應標頭前的延遲（模擬 prefill）
        self.token_s = 0.0            # 每個 chunk 之間的延遲
        self.responder = lambda req: "OK"   # req(dict) → 回應文字
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
                text = fake.responder(req)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for i, ch in enumerate(_chunks(text)):
                        if i:
                            time.sleep(fake.token_s)
                        data = {"id": "x", "object": "chat.completion.chunk", "created": 0,
                                "model": req.get("model", "m"),
                                "choices": [{"index": 0, "delta": {"content": ch},
                                             "finish_reason": None}]}
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


def responder(req: dict) -> str:
    """假模型：判斷類看有沒有 schema；主題類依片段內容回固定句子。"""
    seg = req["messages"][-1]["content"].split("【指示】")[0]
    if "response_format" in req:
        if "亂碼判斷" in seg:
            return "我覺得要建"                      # 不是 JSON ⇒ 當不確定
        return '{"decision":"不建"}' if "天氣" in seg else '{"decision":"建"}'
    if "解析不了" in seg:
        return "這段很有意思喔"                        # 沒有 ｜ ⇒ 解析不出
    lines = []
    if "高雄" in seg:
        lines.append("狀態｜邪狼.住地｜邪狼住在高雄")
    if "台北" in seg:
        lines.append("狀態｜邪狼.住地｜邪狼搬到台北住了")
    if "貓" in seg:
        lines.append("事件|無|邪狼家的貓很胖")      # 半形 | 也要吃
    if "亂碼判斷" in seg:
        lines.append("其他｜無｜邪狼說了亂碼判斷")
    return "\n".join(lines) or "其他｜無｜沒什麼"


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
    return "\n".join(m["content"] for m in ctx if m["role"] == "system" and "⟦記憶⟧" in m["content"])


def mm(d: str, fake: FakeLLM, **over) -> MemoryManager:
    cfg = {"v3_recall": False, "v3_opening": False, "v4_enabled": True,
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
        cx.close()

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
            inj = injected(m.get_context("喵星人好嗎"))
            check("第二件也撈得回", "貓很胖" in inj, inj)

            turn(m, "我搬到台北了，現在住那邊", "恭喜搬家")
            ok = wait(lambda: judged(m)[0] == 6)
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
            check("主題句解析不出 ⇒ 視為不建、不卡住、計數", ok and "解析丟行 1" in m.status_text(),
                  m.status_text())
            turn(m, "亂碼判斷", "嗯")
            ok = wait(lambda: judged(m)[0] == 14)
            check("判斷輸出非 JSON ⇒ 當不確定 ⇒ 照建（observation 那段也建了 1 件）", ok and "主題 5" in m.status_text(), m.status_text())

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
            check("被取消的段不推進度", judged(m)[0] == 14, str(judged(m)))
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
            ok = wait(lambda: judged(m)[0] == 16)
            check("恢復後整段重判完成（冪等：主題不重複）", ok and "主題 6" in m.status_text(), m.status_text())

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
