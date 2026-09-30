# core/memory/judge.py
"""
準 v4 寫入端的背景判斷（乙：同一顆對話模型，閒置時跑、一開口就取消）。

一段對話 → 兩次呼叫：
  ① 判斷類：「這段有沒有值得記的事」——`response_format: json_schema`，enum 限死
     ["建","不建","不確定"]；「不確定」當「建」（寧可多建：主題是可重建的索引，漏建才救不回來）。
  ② 生成類：主題句——純文字「類型｜屬性鍵｜主題句」一行一件，**JSON 由 Python 組**
     （生成類不套 schema：輸出越短越快，這是延遲考量，《歸檔》準 v4 決策）。
  ⭐ prompt 順序＝「對話片段在前、指示在後」：兩次呼叫前綴相同 ⇒ 第二次吃到 prompt cache。

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

import json
import threading
import time
from datetime import datetime

from core.llm_client import BackgroundCancelled
from core.logger import setup_logger
from core.memory.db import connect
from core.memory.topic_index import fill_vectors
from core.memory.topics import TopicStore

logger = setup_logger("MemoryJudge")

DECIDE_SCHEMA = {
    "type": "object",
    "properties": {"decision": {"type": "string", "enum": ["建", "不建", "不確定"]}},
    "required": ["decision"],
    "additionalProperties": False,
}

_SYSTEM = "你是記憶整理助手。只依照指示輸出，不要多說任何話。"


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
        self._path: str | None = None
        self._want_path: str | None = None
        self._cx = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._paused_at: float | None = None
        self._last_activity = time.monotonic()
        self.stats = {"segments": 0, "built": 0, "skipped": 0, "topics": 0, "cancelled": 0,
                      "cancels_live": 0, "parse_bad": 0, "errors": 0, "backlog": 0}
        self._backlog_warned = False
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

    def status(self) -> str:
        s = self.stats
        state = "停用" if self._path is None and self._want_path is None else (
            "暫停" if self._paused_at is not None else "待命")
        return (f"背景判斷{state}：待判 {s['backlog']} 句｜段 {s['segments']}（建 {s['built']}／"
                f"不建 {s['skipped']}）｜主題 {s['topics']}｜取消 {s['cancelled']}"
                f"（切斷進行中 {s['cancels_live']}）｜解析丟行 {s['parse_bad']}｜錯誤 {s['errors']}")

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
                did = self._step()
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
        st = TopicStore(self._cx)             # 首次：建表＋judged_upto
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
        mt = cfg["judge_max_tokens"]
        decide_raw = self.llm.call(
            self._messages(text, self._decide_instruction()),
            purpose="記憶判斷", max_tokens=mt["decide"], schema=DECIDE_SCHEMA)
        try:
            decision = json.loads(decide_raw)["decision"]
        except (ValueError, KeyError, TypeError):
            logger.warning(f"判斷輸出不是預期 JSON，當「不確定」：{decide_raw[:80]!r}")
            decision = "不確定"

        items: list[dict] = []
        if decision != "不建":
            out = self.llm.call(self._messages(text, self._topic_instruction()),
                                purpose="記憶主題", max_tokens=mt["topics"])
            items, bad = parse_topics(out, types=cfg["topic_types"], max_n=cfg["judge_max_topics"])
            self.stats["parse_bad"] += bad
            if not items:
                logger.warning(f"判「{decision}」但主題句解析不出任何一件（serial {first}–{last}），"
                               f"本段視為不建；原句仍在、可重判：{out[:120]!r}")

        speakers = [r.get("speaker_id") for r in seg]
        ids = st.add_topics(items, first_serial=first, last_serial=last,
                            event_ts=seg[-1]["ts"], speakers=speakers, judged_upto=last)
        self.stats["segments"] += 1
        self.stats["built" if items else "skipped"] += 1
        self.stats["topics"] += len(ids)
        logger.info(f"背景判斷 serial {first}–{last}：{decision}，建 {len(ids)} 件")
        if ids and self.emb is not None and self.emb.ready.is_set():
            fill_vectors(self._cx, self.emb)
        return True

    # ── 提示詞 ────────────────────────────────────────────────

    @staticmethod
    def _messages(segment: str, instruction: str) -> list[dict]:
        # ⭐ 片段在前、指示在後：兩次呼叫前綴相同（prompt cache）
        return [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"【對話片段】\n{segment}\n\n【指示】\n{instruction}"}]

    def _decide_instruction(self) -> str:
        return (f"上面這段對話裡，有沒有值得長期記住的事？\n{self.cfg['judge_criteria']}\n"
                "有就回「建」，沒有就回「不建」，拿不準回「不確定」。")

    def _topic_instruction(self) -> str:
        types = "、".join(self.cfg["topic_types"])
        return ("把上面這段對話裡值得長期記住的事逐件寫下來，一行一件，格式：\n"
                "類型｜屬性鍵｜主題句\n"
                f"· 類型只能是：{types}\n"
                "· 屬性鍵：這件事若是某人會變動的狀態（住哪、工作、身體狀況…），寫「人名.屬性」，"
                "例如「邪狼.住地」；不是狀態就寫「無」\n"
                "· 主題句：一句話說完一件事，主詞寫出名字（不要用你、我、他），不要加入對話裡沒有的內容\n"
                f"· 最多 {self.cfg['judge_max_topics']} 行，不要編號、不要其他說明")
