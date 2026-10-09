# core/memory/judge.py
"""
準 v4 寫入端的背景判斷（乙：同一顆對話模型，閒置時跑、一開口就取消）。

一段對話 → **一次呼叫**（🔺 2026-10-09 改，舊做法 json_schema 兩段式見《歸檔》D0033）：
  第一行「建／不建／不確定」，要建就接著一行一件「類型｜屬性鍵｜主題句」，**JSON 由 Python 組**。
  · 建不建看**機率不看字**：由第一行的 logprobs 算 P(建)（「不確定」那條路算進建，寧可多建——
    主題是可重建的索引，漏建才救不回來），**只有一個門檻**（config `judge_build_threshold`）。
  · 模型答「不建」但 P(建) 過門檻 ⇒ 它沒寫主題句 ⇒ 補一次呼叫要它寫。
  · 拿不到 logprobs ⇒ WARNING（只記一次）＋計數，退回看文字（不確定／看不懂都當建）。
  · 不用 json_schema：LM Studio 在格式限制下回的是限制「之前」的分布，讀不出選項機率，
    而且同一段結論會變（2026-10-09 實測，《現況》11之2 己）。
  ⭐ prompt 順序＝「規則在前（system，固定不變）、對話片段在後」：規則是每段共用的前綴，
    連續判好幾段時吃得到 prompt cache。規則、類型、例子都在 prompts/memory_judge.txt（邪狼直接排版）。

取消（乙能不能成立的關鍵）：
  · Pipeline 在輪次**開始**呼叫 pause() ⇒ BackgroundLLM.cancel() 切斷連線；
    在輪次**收尾**呼叫 resume()。被取消的那段**不推進度**，下次整段重判（冪等）。
  · 🔴 `cancels_live`＝真的切斷了進行中請求的次數。乙成不成立要看的是
    「取消之後下一輪對話的 TTFT 有沒有變長」——那要在 Windows＋LM Studio 實測，
    本檔只負責把次數與時間點記下來（log 搜「背景取消」對照下一行「LLM呼叫」耗時）。
    量不到改善或做不到 ⇒ 換甲（極小判斷模型），見《現況》準 v4。

🔴 **會默默死掉的那種執行緒**：每步例外都接住記 ERROR、不讓執行緒結束；
   待判積壓超過門檻記 WARNING；/status 看得到積壓與計數。
"""

from __future__ import annotations

import math
import os
import threading
import time
from datetime import datetime

from core.llm_client import BackgroundCancelled
from core.logger import setup_logger
from core.memory.db import connect
from core.memory.topic_index import fill_fts, fill_vectors
from core.memory.topics import TopicStore

logger = setup_logger("MemoryJudge")

_FORCE_BUILD = "這段要建：第一行寫「建」，接著逐件寫主題句。"


# ── 提示詞檔（prompts/memory_judge.txt）────────────────────────

def load_prompt(path: str, *, max_topics: int) -> tuple[str, list[str], list[tuple[str, list[str]]]]:
    """讀判斷規則檔 → （system 文字, 類型名稱清單, 例子[(對話, 期望輸出的各行)]）。

    ⭐ 類型名稱的單一來源就是這個檔的【類型】段（「名稱：說明」每行一個）——config 不另列一份。
    ⭐ 例子格式：「對話：…」一行，接「輸出：」，之後到下一個「對話：」為止是期望輸出（第一行建／不建）。
       🔺 2026-10-09 實測：例子只寫「→ 建：身分」時，模型照抄類型說明、欄位錯位、編造事實
       ⇒ 例子改成完整輸出格式。
    🔴 檔案不在、或【類型】段解析不出任何類型 ⇒ 直接拋（啟動就失敗，不帶著空規則默默跑）。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"記憶判斷規則檔不存在：{path}（config MEMORY_CONFIG['judge_prompt_path']）")
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    sections: dict[str, list[str]] = {}
    cur = None
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("【") and s.endswith("】"):
            cur = s[1:-1]
            sections[cur] = []
        elif cur is not None and s:
            sections[cur].append(s)
    types = [ln.split("：", 1)[0].strip() for ln in sections.get("類型", []) if "：" in ln]
    if not types:
        raise ValueError(f"記憶判斷規則檔的【類型】段解析不出任何類型（每行「名稱：說明」）：{path}")
    examples: list[tuple[str, list[str]]] = []
    for ln in sections.get("例子", []):
        if ln.startswith("對話："):
            examples.append((ln[3:].strip(), []))
        elif examples and ln != "輸出：":
            examples[-1][1].append(ln)
    return raw.replace("{最多幾件}", str(max_topics)).strip(), types, examples


def build_messages(system: str, segment: str, *, force: bool = False) -> list[dict]:
    """正式判斷與 tools/judge_examples.py 共用（驗證工具不另寫一份送法）。
    ⭐ 規則在前（每段共用的前綴）、片段在後；補呼叫只在最後多一句。"""
    user = f"【對話片段】\n{segment}"
    if force:
        user += f"\n\n{_FORCE_BUILD}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ── 讀機率 ────────────────────────────────────────────────────

def classify(word: str) -> str:
    """第一行的字 → 建／不建／不確定／?（看不懂）。"""
    w = word.strip().lstrip("「『*").strip()
    if w.startswith("建"):
        return "建"
    if w.startswith(("不確", "不确")):
        return "不確定"
    if w.startswith("不建"):
        return "不建"
    return "?"


def p_build(tokens: list[dict]) -> float | None:
    """由輸出 token 的 logprobs 算 P(建)＝P(建)＋P(不確定)。拿不到分布回 None。

    第一個非空白 token 的候選：「建…」與「不確…」直接算；「不」要乘上第二個 token 是「確」的機率——
    ⚠️ 那個條件機率只有在模型**實際選了「不」**時才看得到；選了「建」時「不」底下的「不確定」路徑
       看不到、不算（此時 P(建) 已過半，不影響判斷）。
    純函式 ⇒ 自測直接餵資料。
    """
    i = 0
    while i < len(tokens) and not (tokens[i].get("token") or "").strip():
        i += 1
    if i >= len(tokens) or not tokens[i].get("top_logprobs"):
        return None
    first = tokens[i]
    nxt = tokens[i + 1] if i + 1 < len(tokens) else None
    total = 0.0
    for c in first["top_logprobs"]:
        t, p = (c.get("token") or "").strip(), math.exp(c["logprob"])
        if t.startswith("建") or t.startswith(("不確", "不确")):
            total += p
        elif t == "不" and (first.get("token") or "").strip() == "不" and nxt and nxt.get("top_logprobs"):
            total += p * sum(math.exp(cc["logprob"]) for cc in nxt["top_logprobs"]
                             if (cc.get("token") or "").strip().startswith(("確", "确")))
    return min(total, 1.0)


def split_answer(text: str) -> tuple[str, str]:
    """回應 → （第一行判斷, 其餘的主題句部分）。第一行看不懂時整段都當主題句部分。"""
    lines = text.strip().splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return "?", ""
    head = classify(lines[0])
    if head == "?":
        return "?", "\n".join(lines)
    return head, "\n".join(lines[1:])


def _parse_ts(ts: str | None):
    try:
        return datetime.fromisoformat(ts) if ts else None
    except ValueError:
        return None


def split_segment(rows: list[dict], *, max_rows: int, gap_minutes: float,
                  close_s: float, now: datetime | None = None) -> list[dict] | None:
    """從待判原句的開頭切出**一段講完了的**對話；還沒講完回 None。

    講完了＝後面隔了一段空白（gap）、或滿 max_rows、或最後一句已經過了 close_s。
    ⚠️ 純函式（不碰 DB、不碰時鐘以外的狀態）⇒ 自測直接餵資料。
    """
    if not rows:
        return None
    seg = [rows[0]]
    for prev, r in zip(rows, rows[1:]):
        a, b = _parse_ts(prev["ts"]), _parse_ts(r["ts"])
        if a and b and (b - a).total_seconds() > gap_minutes * 60:
            return seg                       # 後面隔了空白 ⇒ 這段講完了
        seg.append(r)
        if len(seg) >= max_rows:
            return seg
    last = _parse_ts(seg[-1]["ts"])
    if last is None:
        return seg                           # 時間不明就不等（等不出結果）
    now = now or datetime.now(last.tzinfo)
    if (now - last).total_seconds() >= close_s:
        return seg
    return None


def render_segment(seg: list[dict], names: dict[str, str]) -> str:
    """原句 → 給背景模型看的片段。說話者用名字，observation 那列印情境。"""
    t0 = _parse_ts(seg[0]["ts"])
    head = f"（對話時間：{t0.strftime('%Y-%m-%d %H:%M')}）" if t0 else "（對話時間不明）"
    lines = [head]
    for r in seg:
        who = names.get(r.get("speaker_id") or "", "")
        if not who:
            who = {"user": "對方", "assistant": "奧狐"}.get(r["role"], r["role"])
        text = (r.get("clean") or r.get("content") or "").strip()
        if r["role"] == "user" and not text and r.get("situation"):
            lines.append(f"[情境] {r['situation']}")
            continue
        lines.append(f"{who}：{text}")
    return "\n".join(lines)


def parse_topics(text: str, *, types: list[str], max_n: int) -> tuple[list[dict], int]:
    """「類型｜屬性鍵｜主題句」→ [{type, attr_key, text}]。回傳（結果, 丟掉的行數）。"""
    out, bad = [], 0
    for line in text.splitlines():
        line = line.strip().lstrip("-•*0123456789.、) ").strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace("|", "｜").split("｜")]
        if len(parts) != 3 or not parts[2]:
            bad += 1
            continue
        typ, key, sent = parts
        if typ not in types:
            logger.warning(f"主題類型不在清單內，照存：「{typ}」")
        if key in ("", "-", "－", "無", "none", "None", "null"):
            key = None
        out.append({"type": typ, "attr_key": key, "text": sent})
        if len(out) >= max_n:
            break
    return out, bad


class BackgroundJudge:
    """背景判斷執行緒。對外只有 set_path／notify／pause／resume／status／close。"""

    def __init__(self, llm, emb, cfg: dict) -> None:
        self.llm = llm
        self.emb = emb
        self.cfg = cfg
        self.names = cfg["speaker_names"]
        self.system, self.types, self.examples = load_prompt(
            cfg["judge_prompt_path"], max_topics=cfg["judge_max_topics"])
        self._path: str | None = None
        self._want_path: str | None = None
        self._cx = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._paused_at: float | None = None
        self._last_activity = time.monotonic()
        self.stats = {"segments": 0, "built": 0, "skipped": 0, "topics": 0, "cancelled": 0,
                      "cancels_live": 0, "parse_bad": 0, "errors": 0, "backlog": 0,
                      "no_lp": 0, "extra": 0}
        self._no_lp_warned = False
        self._backlog_warned = False
        self._working = False          # 正在判一段（給 /status 看，不參與任何決策）
        self._thread = threading.Thread(target=self._run, name="memory-judge", daemon=True)
        self._thread.start()

    # ── 對外 ──────────────────────────────────────────────────

    def set_path(self, path: str | None) -> None:
        """換要判的 DB；None＝停用（施測中）。會取消進行中的呼叫。"""
        with self._lock:
            self._want_path = path
        self.llm.cancel()
        self._wake.set()

    def detach(self, timeout: float = 5.0) -> bool:
        """停用並**等到**執行緒放開 DB 連線（切換／丟棄 DB 前必須先做——Windows 不能刪開著的檔，
        施測進場時日常區也必須已經沒有人在寫）。回傳是否在時限內放開。"""
        self.set_path(None)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._path is None and self._cx is None:
                return True
            self._wake.set()
            time.sleep(0.02)
        logger.error(f"背景判斷執行緒沒有在 {timeout}s 內放開 DB（{self._path}）")
        return False

    @property
    def busy(self) -> bool:
        """正在判一段（給 tools/judge_worker.py 判斷「做完了沒」）。"""
        return self._working

    def notify(self) -> None:
        self._wake.set()

    def pause(self) -> None:
        """輪次開始：立刻讓出推論槽。"""
        with self._lock:
            self._paused_at = time.monotonic()
        if self.llm.cancel():
            self.stats["cancels_live"] += 1
            logger.info("背景取消：切斷進行中的背景請求（對照下一行對話 LLM 耗時）")

    def resume(self) -> None:
        """輪次收尾：重新計算閒置時間。"""
        with self._lock:
            self._paused_at = None
            self._last_activity = time.monotonic()
        self.llm.resume()
        self._wake.set()

    def close(self, timeout: float = 5.0) -> None:
        self._stop = True
        self.llm.cancel()
        self._wake.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.error(f"背景判斷執行緒沒有在 {timeout}s 內結束")

    def status(self, pending: int | None = None) -> str:
        """pending：呼叫端即時查的待判句數；None 時退回上一次判斷時的快照（可能過時）。"""
        s = self.stats
        state = "停用" if self._path is None and self._want_path is None else (
            "判斷中" if self._working else "暫停" if self._paused_at is not None else "待命")
        n = f"{pending}" if pending is not None else f"{s['backlog']}（上次判斷時）"
        return (f"背景判斷{state}：待判 {n} 句｜段 {s['segments']}（建 {s['built']}／"
                f"不建 {s['skipped']}）｜主題 {s['topics']}｜取消 {s['cancelled']}"
                f"（切斷進行中 {s['cancels_live']}）｜補呼叫 {s['extra']}｜缺機率 {s['no_lp']}"
                f"｜解析丟行 {s['parse_bad']}｜錯誤 {s['errors']}")

    # ── 執行緒 ────────────────────────────────────────────────

    def _reconcile_path(self) -> None:
        with self._lock:
            want = self._want_path
        if want == self._path:
            return
        if self._cx is not None:
            self._cx.close()
            self._cx = None
        self._path = want
        with self._lock:
            if self._paused_at is None:
                self.llm.resume()   # set_path 的 cancel 只為打斷舊對象，不該留著擋新對象
        if want is not None:
            logger.info(f"背景判斷對象：{want}")
        # ⚠️ 連線延後到真的要判時才開（_step）：閒置門檻沒到就結束的程序（診斷工具）
        #    不會因為建表而寫到日常區。

    def _wait_s(self) -> float | None:
        """還要等多久才能做事；None＝現在可以做。"""
        with self._lock:
            paused_at, last = self._paused_at, self._last_activity
        now = time.monotonic()
        if paused_at is not None:
            limit = self.cfg["judge_pause_timeout_s"]
            if now - paused_at < limit:
                return min(5.0, limit - (now - paused_at))
            logger.warning(f"背景判斷暫停超過 {limit}s 沒被恢復（收尾掛點漏呼叫？），自動恢復")
            self.resume()
            return 0.0
        idle = self.cfg["judge_idle_s"] - (now - last)
        return idle if idle > 0 else None

    def _run(self) -> None:
        while not self._stop:
            try:
                self._reconcile_path()
                if self._path is None:
                    self._wake.wait(5.0)
                    self._wake.clear()
                    continue
                w = self._wait_s()
                if w is not None:
                    self._wake.wait(w)
                    self._wake.clear()
                    continue
                self._working = True
                try:
                    did = self._step()
                finally:
                    self._working = False
                if not did:   # 沒有講完的段：等新句子（notify）或等它變「講完了」
                    self._wake.wait(max(0.5, self.cfg["judge_segment_close_s"] / 2))
                    self._wake.clear()
            except BackgroundCancelled:
                self.stats["cancelled"] += 1
            except Exception as e:  # noqa: BLE001
                self.stats["errors"] += 1
                logger.error(f"背景判斷失敗，稍後重試：{type(e).__name__}: {e}")
                self._wake.wait(10.0)
                self._wake.clear()
        if self._cx is not None:
            self._cx.close()

    def _step(self) -> bool:
        """判一段。回傳 True＝有做事（馬上接著做下一段）。"""
        cfg = self.cfg
        if self._cx is None:
            self._cx = connect(self._path)
        st = TopicStore(self._cx)             # 首次：建表＋judged_upto（舊表補 status 欄）
        fill_fts(self._cx)                    # 首次：補上舊主題的關鍵字索引（衍生物，冪等）
        if self.emb is not None and self.emb.ready.is_set():
            fill_vectors(self._cx, self.emb)
        rows = st.pending_rows(cfg["judge_segment_max"] * 20)
        self.stats["backlog"] = len(rows)
        if len(rows) >= cfg["judge_backlog_warn"] and not self._backlog_warned:
            logger.warning(f"背景判斷積壓 {len(rows)}+ 句（跟不上、或一直被取消）")
            self._backlog_warned = True
        elif len(rows) < cfg["judge_backlog_warn"]:
            self._backlog_warned = False
        seg = split_segment(rows, max_rows=cfg["judge_segment_max"],
                            gap_minutes=cfg["judge_segment_gap_minutes"],
                            close_s=cfg["judge_segment_close_s"])
        if seg is None:
            return False

        text = render_segment(seg, self.names)
        first, last = seg[0]["serial"], seg[-1]["serial"]
        # 開始也記一行：量「撞取消」時要知道背景此刻在跑（完成那行要等呼叫結束才出現）
        logger.info(f"背景判斷開始 serial {first}–{last}（{len(seg)} 句）")
        said, p, build, items = self._judge(text, first, last)
        thr = cfg["judge_build_threshold"]

        speakers = [r.get("speaker_id") for r in seg]
        ids = st.add_topics(items, first_serial=first, last_serial=last,
                            event_ts=seg[-1]["ts"], speakers=speakers, judged_upto=last)
        self.stats["segments"] += 1
        self.stats["built" if items else "skipped"] += 1
        self.stats["topics"] += len(ids)
        # 🔴 第 6 點（定門檻）的資料來源就是這一行：P(建)、模型自己的答案、結果
        pb = "—" if p is None else f"{p:.3f}"
        logger.info(f"背景判斷 serial {first}–{last}：模型答「{said}」P(建)={pb}（門檻 {thr}）"
                    f"→ {'建' if build else '不建'}，建 {len(ids)} 件")
        if ids:
            fill_fts(self._cx)
            if self.emb is not None and self.emb.ready.is_set():
                fill_vectors(self._cx, self.emb)
        return True

    def _judge(self, text: str, first: int, last: int) -> tuple[str, float | None, bool, list[dict]]:
        """一段 → （模型自己的答案, P(建), 建不建, 主題）。會打 1～2 次 LLM（可被取消）。"""
        cfg = self.cfg
        out, lps = self.llm.call_lp(self._messages(text), purpose="記憶判斷",
                                    max_tokens=cfg["judge_max_tokens"],
                                    top_logprobs=cfg["judge_top_logprobs"])
        said, body = split_answer(out)
        p = p_build(lps)
        thr = cfg["judge_build_threshold"]
        if p is None:
            self.stats["no_lp"] += 1
            if not self._no_lp_warned:
                logger.warning("背景判斷拿不到 logprobs（LM Studio 串流沒附？），改看文字判斷："
                               "「不確定」與看不懂都當建。只記這一次，次數見 /status「缺機率」")
                self._no_lp_warned = True
            build = said != "不建"
        else:
            build = p >= thr
        if said == "?":
            logger.warning(f"判斷第一行看不懂（serial {first}–{last}）：{out[:80]!r}")

        items: list[dict] = []
        if build:
            if said == "不建":   # 機率過門檻、但模型自己答了不建 ⇒ 它沒寫主題句，補一次
                self.stats["extra"] += 1
                out = self.llm.call(self._messages(text, force=True), purpose="記憶主題",
                                    max_tokens=cfg["judge_max_tokens"])
                body = split_answer(out)[1]
            items, bad = parse_topics(body, types=self.types, max_n=cfg["judge_max_topics"])
            self.stats["parse_bad"] += bad
            if not items:
                logger.warning(f"判建但主題句解析不出任何一件（serial {first}–{last}），"
                               f"本段視為不建；原句仍在、可重判：{out[:120]!r}")

        return said, p, build, items

    # ── 提示詞 ──────────────────────────────────────────────

    def _messages(self, segment: str, *, force: bool = False) -> list[dict]:
        return build_messages(self.system, segment, force=force)
